"""Typo-tolerant reading of user input (see the 2026-09-24 session: the user
types fast; "web serarch", "implment", "Tes th e web search feature" reached
the model verbatim and -- worse -- reached the *slash-command gate* verbatim,
where a mistyped command name was rejected instead of understood.

Two tiers, deliberately independent:

1. Offline (always available, zero dependencies, deterministic): a small
   English dev-vocabulary dictionary plus difflib near-miss repair for words
   the dictionary nearly matches. This runs synchronously on every submitted
   objective and never requires network, model, or credentials.

2. Optional local HF model (opt-in via config ``typo_autocorrect_model``):
   a small seq2seq grammar-correction model (default
   ``vennify/t5-grammar-correction``, ~80M params) loaded lazily through
   ``transformers`` ONLY when configured. Never downloaded implicitly: the
   first load fetches weights from Hugging Face using whatever ambient
   credentials the host already has (HF_TOKEN in the environment -- e.g.
   tamgpt6's .env, which the canonical .env loader already exports -- or
   `huggingface-cli login`'s cached token). Any failure degrades to tier 1.

The corrected text is a *reading aid*: the original text is preserved in
session state, and the submission path reports the correction so the user
can see exactly what was changed. Corrections are never applied inside code
blocks, file paths, URLs, quoted strings, flags, or identifiers -- a typo in
prose must not become a corrupted path.
"""
from __future__ import annotations

import ast
import json
import os
import re
from dataclasses import dataclass
from typing import Any, Optional

# Words a developer plausibly types in an objective. Kept deliberately
# small: this repairs common misspellings of domain vocabulary, it does not
# attempt general English proofreading (that is tier 2's job when enabled).
_VOCAB = (
    "the and for with that this from into your about which there their "
    "please implement implementation installed install installer "
    "feature features search searching research server service services "
    "repository repositories workspace workspaces "
    "function functions class classes variable variables "
    "configuration configure config configs environment environments "
    "authentication authorize authorization credential credentials "
    "database databases migration migrations "
    "orchestration orchestrator orchestrating "
    "vector embedding embeddings memory "
    "model models routing router provider providers "
    "terminal command commands directory directories file files "
    "project projects module modules package packages "
    "typo typos correction corrections "
    "understand understood integration integrated "
    "capabilities capability compatibility compatible "
    "documentation document documents "
    "validation verify verified verifier "
    "web internet browser "
    "stream streaming streamed "
    "queued queue "
    "immediately immediately "
    "borrow borrowed borrowing "
    "consolidate consolidated consolidation "
    "behavior behaviours behaviour "
    "because though through "
    "before after during between "
    "tamfis tamfiscode tamfisgpt tamgpt finitron "
    "mcp llm api url json html http https"
).split()

_WORD_RE = re.compile(r"[A-Za-z][A-Za-z0-9_.-]*")
_MAX_REPAIR_CHARS = 20_000
_MAX_REPAIRED_WORDS = 30
# difflib's distance tolerance: short words need a tighter ratio or every
# "the"→"they" becomes a false repair.
_RATIO_BY_LENGTH = {1: 0.99, 2: 0.85, 3: 0.8, 4: 0.78}
_DEFAULT_RATIO = 0.76
# How much the best candidate must beat the runner-up by before an
# ambiguous-looking token is still repaired to the best one.
_CLEAR_WIN_MARGIN = 0.1

# Protected spans: paths (/x/y, ~/x), URLs, emails, quoted strings, long
# identifier-ish tokens (snake_case/camelCase with _ or digits suggest a
# real identifier rather than prose), single characters, and anything in a
# fenced code block.
_FENCE_RE = re.compile(r"```.*?```", re.DOTALL)
_PROTECT_RES = (
    re.compile(r"https?://\S+"),
    re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.]+\b"),
    re.compile(r"(?<![\w~])/(?:[\w.-]+/)+[\w.-]+"),  # multi-segment path
    re.compile(r"~/\S+"),
    re.compile(r"\b\w+_\w+\b"),        # snake_case identifiers
    re.compile(r"\b\w+\.\w{1,4}\b"),   # filenames like main.py
    re.compile(r"--?[A-Za-z][\w-]*"),  # CLI flags
    re.compile(r"'[^']*'"),
    re.compile(r'"[^"]*"'),
)


@dataclass(frozen=True)
class CorrectionResult:
    corrected: str
    changed: bool
    repairs: tuple[tuple[str, str], ...]  # (original, replacement) pairs
    source: str  # "none" | "dictionary" | "model"


def _is_protected(text: str, start: int, end: int) -> bool:
    """Whether [start, end) overlaps a span whose literal text must survive."""
    for pattern in _PROTECT_RES:
        for match in pattern.finditer(text):
            if match.start() < end and start < match.end():
                return True
    return False


def _repair_word(word: str, dictionary: set[str]) -> Optional[str]:
    """The dictionary word `word` was a near miss of, or None.

    Case is preserved from the original token (leading capital or ALLCAPS).
    A candidate is accepted only when it beats the runner-up by a clear
    margin: 'serarch' is 0.92-similar to 'search' and only ~0.7 to
    'research' (repair to search), but a token sitting between two
    dictionary words at nearly the same ratio is genuinely ambiguous and
    must be left alone -- guessing between near-equal candidates is how an
    autocorrector corrupts a correct sentence.
    """
    import difflib

    lowered = word.casefold()
    if lowered in dictionary:
        return None
    if len(lowered) < 2 or len(lowered) > 24:
        return None
    ratio = _RATIO_BY_LENGTH.get(len(lowered), _DEFAULT_RATIO)
    candidates = difflib.get_close_matches(lowered, dictionary, n=2, cutoff=ratio)
    if not candidates:
        return None
    best = candidates[0]
    if len(candidates) > 1:
        best_score = difflib.SequenceMatcher(None, lowered, best).ratio()
        second_score = difflib.SequenceMatcher(None, lowered, candidates[1]).ratio()
        if best_score - second_score < _CLEAR_WIN_MARGIN:
            return None
    if word.isupper():
        return best.upper()
    if word[0].isupper():
        return best.capitalize()
    return best


def _offline_correct(text: str) -> CorrectionResult:
    """Tier 1: deterministic dictionary repair of prose tokens."""
    if not text or len(text) > _MAX_REPAIR_CHARS:
        return CorrectionResult(text, False, (), "none")
    fences = list(_FENCE_RE.finditer(text))
    dictionary = set(_VOCAB)

    repairs: list[tuple[str, str]] = []
    repaired_word_count = 0

    def inside_fence(pos: int) -> bool:
        return any(match.start() <= pos < match.end() for match in fences)

    def replace(match: re.Match) -> str:
        nonlocal repaired_word_count
        word = match.group(0)
        if inside_fence(match.start()) or _is_protected(text, match.start(), match.end()):
            return word
        # Identifier-looking tokens (embedded capitals like writeFile) are
        # almost certainly real identifiers, not prose typos.
        if len(word) > 4 and any(c.isupper() for c in word[1:]):
            return word
        fixed = _repair_word(word, dictionary)
        if fixed is None or fixed == word:
            return word
        if repaired_word_count >= _MAX_REPAIRED_WORDS:
            return word
        repaired_word_count += 1
        repairs.append((word, fixed))
        return fixed

    corrected = _WORD_RE.sub(replace, text)
    if not repairs:
        return CorrectionResult(text, False, (), "none")
    return CorrectionResult(corrected, True, tuple(repairs), "dictionary")


_MODEL_STATE: dict[str, Any] = {}


def _model_pipeline():  # pragma: no cover - exercised only when configured
    """Lazily loaded transformers pipeline for the configured model.

    Returns None when transformers is unavailable or the model was never
    configured. Loads at most once per process.
    """
    if "pipeline" in _MODEL_STATE:
        return _MODEL_STATE["pipeline"]
    _MODEL_STATE["pipeline"] = None  # fail-safe default while loading
    try:
        from transformers import pipeline as hf_pipeline

        _MODEL_STATE["pipeline"] = hf_pipeline(
            "text2text-generation", model=TYPO_MODEL_DEFAULT, device=-1,
        )
    except Exception:
        _MODEL_STATE["pipeline"] = None
    return _MODEL_STATE.get("pipeline")


def _model_correct(text: str) -> CorrectionResult:
    """Tier 2: the optional small HF grammar-correction model.

    Runs only on text tier 1 could not repair, and only for short objectives
    (the model is slow; a 5,000-char paste must not wait on it). Any
    exception degrades to tier 1's result.
    """
    offline = _offline_correct(text)
    if offline.changed or not text or len(text) > 600:
        return offline
    pipe = _model_pipeline()
    if pipe is None:
        return offline
    try:
        output = pipe(
            f"grammar: {text}", max_length=max(64, len(text) * 2), num_beams=1,
        )
        corrected = str((output[0] or {}).get("generated_text") or "").strip()
    except Exception:
        return offline
    if not corrected or corrected.casefold() == text.casefold():
        return offline
    # The model gets one shot at prose; never let it shrink the request to
    # something meaningless (a degenerate model output must not truncate
    # the user's objective).
    if len(corrected) < len(text) // 2:
        return offline
    return CorrectionResult(corrected, True, ((text, corrected),), "model")


def correct_objective_text(text: str, *, use_model: bool = False) -> CorrectionResult:
    """Public entry point: read a user-submitted objective past its typos.

    Never raises. Returns CorrectionResult with source="none" and
    changed=False for anything it cannot confidently improve.
    """
    try:
        if use_model:
            return _model_correct(text)
        return _offline_correct(text)
    except Exception:
        return CorrectionResult(text or "", False, (), "none")


def correction_note(result: CorrectionResult, original: str) -> Optional[str]:
    """One diagnostics line describing what was corrected, if anything."""
    if not result.changed:
        return None
    pairs = "; ".join(f"{bad!r} → {good!r}" for bad, good in result.repairs[:6])
    more = f" (+{len(result.repairs) - 6} more)" if len(result.repairs) > 6 else ""
    return f"Read as: {' ' if pairs else ''}{pairs}{more} (typo auto-correction: {result.source})"


def hf_token_configured() -> bool:
    """Whether any ambient Hugging Face credential is available for tier 2.

    The canonical .env loader (providers._load_project_env) exports HF_TOKEN
    from tamgpt6's .env on this host; huggingface_hub also accepts its own
    cached login token without any env var.
    """
    try:
        from huggingface_hub import HfFolder  # type: ignore

        if HfFolder.get_token():
            return True
    except Exception:
        pass
    return bool(
        os.environ.get("HF_TOKEN")
        or os.environ.get("HF_API_KEY")
        or os.environ.get("HUGGINGFACE_API_KEY")
    )


# Model id for tier 2, overridable per-call in _model_pipeline's caller if
# config ever grows a dedicated knob (kept as a module constant so tests can
# monkeypatch it).
TYPO_MODEL_DEFAULT = "vennify/t5-grammar-correction"

# Keep json/ast imports alive for the object-argument repair parity with
# capability_gateway (removed if tier 2 grows structured output).
_ = (json, ast)
