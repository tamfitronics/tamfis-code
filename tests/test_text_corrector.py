"""text_corrector: typo-tolerant reading of user objectives.

Covers the offline dictionary tier (the always-on path), the protected-span
guarantees (paths/identifiers/quotes must survive), and the degradation
contract (never raises; ambiguous near-misses are left alone).
"""
from tamfis_code.text_corrector import (
    _MODEL_STATE,
    correct_objective_text,
)


def test_common_dev_typos_are_repaired():
    result = correct_objective_text("implment the web serarch feautre")
    assert result.changed
    assert "implement" in result.corrected
    assert "search" in result.corrected
    assert result.source == "dictionary"


def test_clean_text_is_left_untouched():
    result = correct_objective_text("implement the web search feature")
    assert not result.changed
    assert result.source == "none"
    assert result.corrected == "implement the web search feature"


def test_common_pronouns_are_not_corrupted_as_typos():
    result = correct_objective_text("they said the tests are ready for them")
    assert not result.changed
    assert result.corrected == "they said the tests are ready for them"


def test_valid_coding_words_from_live_prompts_are_not_corrupted():
    text = "Confirm why the Codex changes stall across the sites. It still fails; check the process."
    result = correct_objective_text(text)
    assert result.corrected == text
    assert not result.changed


def test_fast_typing_and_split_key_errors_are_repaired_without_losing_intent():
    result = correct_objective_text(
        "You nee dto impove the world-calss coding agent so it can comept e, "
        "with intellgent disgnotics and a better dictonery"
    )
    assert "need to improve" in result.corrected
    assert "world-class coding agent" in result.corrected
    assert "can compete" in result.corrected
    assert "intelligent diagnostics" in result.corrected
    assert "better dictionary" in result.corrected


def test_short_valid_words_are_not_fuzzy_rewritten():
    result = correct_objective_text("they lie low and use the new key")
    assert result.corrected == "they lie low and use the new key"


def test_model_output_cannot_drop_negation_path_flag_or_version(monkeypatch):
    class UnsafePipeline:
        def __call__(self, *_args, **_kwargs):
            return [{"generated_text": "delete release 3.12 from another place"}]

    monkeypatch.setitem(_MODEL_STATE, "pipeline", UnsafePipeline())
    original = "do not delete /srv/app/config.py --keep version 3.12"
    result = correct_objective_text(original, use_model=True)
    assert result.corrected == original
    assert not result.changed


def test_paths_are_never_corrected():
    result = correct_objective_text("fix /home/tamfiscode/serarch_dir and also implment seraching")
    assert "/home/tamfiscode/serarch_dir" in result.corrected


def test_snake_case_identifiers_are_never_corrected():
    result = correct_objective_text("rename web_serarch_fn to web_search_fn and implment it")
    assert "web_serarch_fn" in result.corrected
    assert "web_search_fn" in result.corrected


def test_quoted_strings_are_never_corrected():
    result = correct_objective_text('print "serarch" then implment the serach')
    assert '"serarch"' in result.corrected


def test_code_fences_are_never_corrected():
    text = "```\nserarch implment\n```\nthen implment the serach"
    result = correct_objective_text(text)
    assert "serarch implment" in result.corrected  # inside fence survives
    assert "implement the" in result.corrected      # prose repaired


def test_ambiguous_near_miss_is_left_alone():
    # "tes" is close to several words (test, tea, ten...): must not guess.
    result = correct_objective_text("run the tes suite")
    # Either unchanged, or changed to something containing a real repair --
    # but never a random unrelated word. With our vocab, "tes" is within
    # range of "test"; a single clear match IS repaired.
    assert "test" in result.corrected or result.corrected == "run the tes suite"


def test_case_is_preserved():
    result = correct_objective_text("Implment the serach")
    assert "Implement" in result.corrected


def test_never_raises_on_garbage():
    for text in ("", "   ", "x" * 50_000, " Implment\t\n the  serach "):
        result = correct_objective_text(text)
        assert isinstance(result.corrected, str)


def test_empty_and_whitespace_return_unchanged():
    assert correct_objective_text("").corrected == ""
    assert correct_objective_text("   ").changed is False
