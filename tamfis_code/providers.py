"""Provider integrations for Tamfis-Code.

Supports Ollama Cloud, xAI Grok, NVIDIA NIM, Hugging Face, OpenRouter, the
TamfisGPT subscription API, and the internal Tier IV orchestration service
through a canonical OpenAI-compatible client interface.

Automatic standalone routing is deterministic NVIDIA NIM-first after
capability, configuration and live-health filtering. Other providers are an
ordered resilience chain, not routine random traffic. Tamfis/Tier IV/local
routes remain explicitly selectable compatibility routes, not AUTO candidates.

Ollama Cloud is accessed through the signed-in local Ollama daemon at
``http://127.0.0.1:11434/v1``. The daemon forwards ``:cloud`` models to
Ollama Cloud while preserving the OpenAI-compatible tools/streaming contract
already used throughout Tamfis-Code.

In a source checkout, a sibling ``.env`` is loaded for development. Installed
packages use process/user configuration and never assume a builder's path.
"""

from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, AsyncIterator, Dict, List, Optional, TYPE_CHECKING

from openai import AsyncOpenAI

from .provider_protocols import system_messages_first


def _load_project_env() -> None:
    """Load Tamfis-Code's root .env without requiring python-dotenv.

    Existing process environment variables always win. The canonical file is
    ``/home/tamfiscode/.env`` unless ``TAMFIS_CODE_ENV_FILE`` overrides it.
    """

    default_root = Path(
        os.environ.get("TAMFIS_CODE_ROOT") or Path(__file__).resolve().parents[1]
    ).expanduser()
    env_path = Path(
        os.environ.get(
            "TAMFIS_CODE_ENV_FILE",
            str(default_root / ".env"),
        )
    ).expanduser()

    if not env_path.is_file():
        return

    try:
        for raw_line in env_path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("export "):
                line = line[7:].strip()
            if "=" not in line:
                continue

            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip()

            if not key or not key.replace("_", "").isalnum():
                continue

            if (
                len(value) >= 2
                and value[0] == value[-1]
                and value[0] in {"'", '"'}
            ):
                value = value[1:-1]

            os.environ.setdefault(key, value)
    except OSError:
        # Environment loading must never make the CLI unstartable.
        return


_load_project_env()


if TYPE_CHECKING:
    from .routing import TaskProfile


class ProviderType(str, Enum):
    """Supported provider identifiers."""

    TIER_IV = "tier_iv"
    TAMFIS = "tamfis"
    OLLAMA_CLOUD = "ollama_cloud"
    GROK = "grok"
    HF = "hf"
    NVIDIA = "nvidia"
    OPENROUTER = "openrouter"
    LOCAL = "local"
    AUTO = "auto"


# Providers whose OpenAI-compatible endpoint accepts `reasoning_effort` and,
# when given it, streams a real `reasoning_content` delta ahead of the
# answer (confirmed live against NVIDIA NIM) -- other OpenAI-compatible
# providers/models reject or ignore the field. Single source of truth so
# providers.py's own request-building and runner_local.py's streaming loop
# (which needs the same set to decide whether "thought for Xs" tracking is
# meaningful) can't drift into two different lists.
REASONING_EFFORT_CAPABLE_PROVIDERS = frozenset({
    ProviderType.NVIDIA,
    ProviderType.OPENROUTER,
    ProviderType.TIER_IV,
})


def reasoning_effort_capable(provider: ProviderType, model: str) -> bool:
    """Whether `model` on `provider` actually handles reasoning_effort,
    rather than just being served by a provider that generally supports it.

    Confirmed live: on NVIDIA NIM, reasoning_effort works cleanly on the
    nemotron family (NVIDIA's own reasoning-tuned models) but the plain
    instruct models NVIDIA also hosts -- meta/llama-3.1-*, moonshotai/
    kimi-k2.6, mistralai/mistral-large-2-123b, google/gemma-2-27b-it,
    microsoft/phi-3-medium-128k-instruct -- HANG indefinitely when given
    it: a real streaming call to meta/llama-3.1-70b-instruct (NVIDIA's
    default_model at the time, briefly, per the v0.4.39 kimi-k2.6-404 fix)
    with reasoning_effort="high" produced zero chunks in 40+ seconds, while
    the identical call without it returned in well under a second.
    REASONING_EFFORT_CAPABLE_PROVIDERS alone silently broke the moment
    NVIDIA's default_model moved off nemotron that time -- nobody had
    re-verified reasoning_effort against the new default, since the
    original "confirmed live" note only ever tested nemotron. This
    model-name substring check is why later default_model changes within
    the nemotron family (2026-07-26: nemotron-3-ultra-550b-a55b) stay
    correctly gated without needing another update here. OpenRouter/Tier IV
    are left as provider-level (unverified per-model here; no live evidence
    either way).
    """
    if provider not in REASONING_EFFORT_CAPABLE_PROVIDERS:
        return False
    if provider == ProviderType.NVIDIA:
        return "nemotron" in (model or "").lower()
    return True

# TaskType.value (routing.py) -> Tier IV `mode` category (tamgpt6's
# model_catalog.py model "categories" tags: chat, coding, research,
# academic, business, project). Anything not listed falls through to
# "coding" -- tamfis-code is a coding CLI, so the safe default for an
# unrecognised/mixed task is still a coding-vetted model pool rather than
# Tier IV's own "auto" (which draws from every category, coding included
# only by chance).
_TIER_IV_MODE_BY_TASK_TYPE: Dict[str, str] = {
    "conversation": "chat",
    "question": "chat",
    "research": "research",
}


def _tier_iv_mode_for_task(task_profile: Optional["TaskProfile"]) -> str:
    """Map a classified task to the Tier IV `mode` category that keeps
    model selection inside a coding-appropriate (or otherwise matching)
    pool instead of TamfisRuntime's uncategorised random draw."""

    task_type = getattr(task_profile, "task_type", None)
    task_type_value = getattr(task_type, "value", None) or str(task_type or "")
    return _TIER_IV_MODE_BY_TASK_TYPE.get(task_type_value, "coding")


# Task types demanding enough to justify a paid model even where a
# provider has a free tier available -- research (which needs a strong,
# up-to-date model to be worth doing at all) and the two "real coding
# work" classifications (routing.py marks both DEBUG and EDIT "high"
# complexity, same as AUDIT). Deliberately excludes QUESTION/CONVERSATION/
# INSPECT/TEST/GIT/EXECUTE/PLAN -- routine chat, reads, and simple
# commands are exactly what a free-tier model handles fine.
_PAID_TIER_TASK_TYPES = frozenset({"research", "audit", "debug", "edit"})


def _task_needs_paid_tier(task_profile: Optional["TaskProfile"]) -> bool:
    """True when the task is demanding enough (deep research, or complex
    coding/analysis) to spend a paid-tier model instead of a free one."""

    task_type = getattr(task_profile, "task_type", None)
    task_type_value = getattr(task_type, "value", None) or str(task_type or "")
    if task_type_value in _PAID_TIER_TASK_TYPES:
        return True
    from .routing import ComplexityLevel, complexity_at_least

    return complexity_at_least(
        str(getattr(task_profile, "complexity", "")),
        ComplexityLevel.COMPLEX,
    )


@dataclass(frozen=True)
class ProviderConfig:
    """Static provider and default-model capabilities."""

    name: str
    base_url: str
    api_key_env: str
    default_model: str
    models: List[str] = field(default_factory=list)

    # A free-tier model this provider can serve without consuming paid
    # credits, e.g. OpenRouter's `:free`-suffixed routes. When set,
    # ProviderManager.select_model uses this by default and only falls
    # back to `default_model` (the paid tier) for tasks that actually
    # need it -- see _task_needs_paid_tier. None means this provider has
    # no meaningfully free option, so `default_model` is always used.
    free_model: Optional[str] = None

    # Lower values are preferred by canonical automatic routing.
    priority: int = 999

    # Secondary provider/model characteristics.
    weight: int = 1
    reasoning_supported: bool = False
    vision_supported: bool = False
    context_window: int = 32768
    coding_quality: int = 1
    tool_calling: bool = True
    structured_output: bool = True
    long_context: bool = False
    local_only: bool = False


@dataclass
class RouteHealth:
    """Bounded runtime health for one provider/model route."""

    failures: int = 0
    successes: int = 0
    circuit_open_until: float = 0.0
    last_failure: float = 0.0
    last_success: float = 0.0
    last_error_class: str = ""


@dataclass
class RoutingTelemetry:
    provider_requests: Dict[str, int] = field(default_factory=dict)
    provider_successes: Dict[str, int] = field(default_factory=dict)
    provider_failures: Dict[str, int] = field(default_factory=dict)
    provider_fallbacks: Dict[str, int] = field(default_factory=dict)
    provider_tool_failures: Dict[str, int] = field(default_factory=dict)
    provider_stream_failures: Dict[str, int] = field(default_factory=dict)
    nim_eligible_requests: int = 0
    nim_selected_requests: int = 0
    nim_successful_requests: int = 0
    nim_fallback_requests: int = 0

    @property
    def nim_share_of_eligible(self) -> float:
        if not self.nim_eligible_requests:
            return 0.0
        return self.nim_selected_requests / self.nim_eligible_requests


_HEALTH_LOCK = threading.RLock()
_ROUTE_HEALTH: Dict[tuple[str, str], RouteHealth] = {}
_ROUTING_TELEMETRY = RoutingTelemetry()


def _increment(mapping: Dict[str, int], key: str) -> None:
    mapping[key] = mapping.get(key, 0) + 1


class ProviderManager:
    """Initialise, inspect, rank, and access configured AI providers."""

    # Deterministic fallback order after an already-selected route fails.
    # Initial AUTO selection uses AUTO_PROVIDER_WEIGHTS below instead.
    PRIORITY_ORDER: tuple[ProviderType, ...] = (
        ProviderType.NVIDIA,
        ProviderType.OLLAMA_CLOUD,
        ProviderType.HF,
        ProviderType.OPENROUTER,
        # Grok last: real billed xAI spend showed up within the first week
        # of enabling it (2026-08-06). Ollama Cloud's included models and
        # the free/cheap NIM+HF+OpenRouter tiers now all get tried first.
        ProviderType.GROK,
        ProviderType.TAMFIS,
        ProviderType.TIER_IV,
    )

    # AUTO is deterministic NIM-first after capability and live-health
    # filtering. These weights remain compatibility/diagnostic metadata;
    # they are not used to randomly divert healthy NIM-eligible traffic.
    # The >=85% economic target is therefore a floor: healthy eligible AUTO
    # traffic is 100% NIM, while the ordered chain remains available for a
    # genuine route failure or capability gap.
    AUTO_PROVIDER_WEIGHTS: dict[ProviderType, int] = {
        ProviderType.NVIDIA: 85,
        ProviderType.OLLAMA_CLOUD: 5,
        ProviderType.HF: 5,
        ProviderType.OPENROUTER: 3,
        ProviderType.GROK: 2,
    }

    PROVIDERS: Dict[ProviderType, ProviderConfig] = {
        ProviderType.OLLAMA_CLOUD: ProviderConfig(
            name="Ollama Cloud",
            # The signed-in local daemon forwards :cloud models to Ollama.
            # This preserves Tamfis-Code's existing OpenAI SDK, streaming,
            # structured-output, and function-tool calling implementation.
            base_url=os.environ.get(
                "OLLAMA_BASE_URL",
                "http://127.0.0.1:11434/v1",
            ).rstrip("/"),
            api_key_env="OLLAMA_API_KEY",
            # FIX 2026-09-03 (operator request, "remove gemma4 entirely from
            # ollama list, since it isn't helpful"): gemma4:cloud removed as
            # both default_model and a selectable model. Same failure class
            # TamfisGPT's gemma4-cloud route was live-disproven on the same
            # day (tier_iv_orchestration/config/orchestration.yaml): under
            # forced tool_choice, it answered with unparseable pseudo-tool-
            # call text instead of a real structured call -- directly
            # damaging for an agentic coding CLI that depends on real tool
            # calls for every file edit/shell command. kimi-k2.7-code:cloud
            # (the included-plan coding route select_model() already always
            # returns for Ollama Cloud, per its own comment a few lines
            # below) is the safe default here instead.
            default_model=os.environ.get(
                "TAMFIS_CODE_OLLAMA_GENERAL_MODEL",
                "kimi-k2.7-code:cloud",
            ),
            models=[
                # Moonshot AI's newest Ollama Cloud model as of 2026-07-28:
                # 2.81T-param native multimodal agentic MoE, 1M-token
                # context, tools/thinking/vision confirmed on the model's
                # own ollama.com/library/kimi-k3 page before wiring this in.
                # It remains available for explicit operator-approved heavy
                # jobs, but is not an automatic/default route because it can
                # consume Ollama extra usage.
                "kimi-k3:cloud",
                "kimi-k2.7-code:cloud",
                # GLM cloud route kept directly behind the current
                # included-plan Kimi coding default. This makes it available
                # as an explicit Ollama Cloud priority without unexpectedly
                # changing the automatic model for every task.
                "glm-5.2:cloud",
                # Replaces the prior minimax-m2.7:cloud route outright
                # (2026-08-03): same MiniMax family already routed via
                # Ollama Cloud, now the latest generation with a 1M-token
                # context window (vs. m2.7's 204800) -- see also
                # minimaxai/minimax-m3 already selectable on NVIDIA NIM
                # above. Not the default -- explicit-select route for
                # long-context coding work.
                "minimax-m3:cloud",
                # DeepSeek-V4-Flash-0731 (2026-08-04): Ollama's announced
                # agentic-capability refresh of the DeepSeek V4 Flash line --
                # reliable tool calling at long contexts, three selectable
                # reasoning_effort tiers (low/high/max), hosted US+EU with
                # zero data retention. Explicit-select route like glm-5.2 and
                # minimax-m3 above, not the default, since it hasn't been
                # live-verified against this CLI's own tool-calling loop yet.
                "deepseek-v4-flash:0731-cloud",
            ],
            # priority=3 (2026-08-08, was 0/first): Ollama Cloud's weekly
            # usage limit is a real multi-day-reset 429, not a transient
            # blip -- being priority 0 meant every auto-routed request hit
            # that quota first and burned it out. NVIDIA NIM is now
            # priority 0 (see ProviderType.NVIDIA below) as the reliable
            # free-tier default across all subscription tiers/model groups.
            priority=3,
            weight=15,
            reasoning_supported=True,
            vision_supported=True,
            # NOTE: this budget is shared by the whole provider bucket
            # (gemma4:cloud, minimax-m3:cloud too, not just kimi-k3), and
            # gates real request-size/truncation logic in runner_local.py --
            # left at the existing conservative value rather than raised to
            # kimi-k3's real 1M window, since that would misrepresent the
            # other two models' actual limits.
            context_window=262144,
            coding_quality=5,
            tool_calling=True,
            structured_output=True,
            long_context=True,
            local_only=False,
        ),
        # xAI Grok -- direct native endpoint (api.x.ai/v1). Last-resort
        # provider for Tamfis-Code, tried only after Ollama Cloud, NVIDIA
        # NIM, Hugging Face, and OpenRouter have all been exhausted or ruled
        # out (2026-08-06: demoted from priority 2 after real billed xAI
        # spend showed up within the first week of enabling it -- see the
        # PRIORITY_ORDER comment above). Enabled by the operator setting
        # GROK_API_KEY in .env; when unset, the provider is simply not
        # initialised and routing falls through the remaining chain (which
        # still carries the x-ai/grok-4.6 OpenRouter relay as a selectable
        # model, ahead of this direct route). Grok's native API accepts bare
        # model names (grok-4.6, grok-4.3, grok-4-fast) and is
        # OpenAI-compatible for chat, tools, streaming, and structured
        # outputs. Image (grok-imagine) and video (grok-imagine-video)
        # generation are exposed through the same endpoint. Still directly
        # selectable by name at any time, regardless of automatic order.
        ProviderType.GROK: ProviderConfig(
            name="xAI Grok",
            base_url=os.environ.get(
                "GROK_BASE_URL",
                "https://api.x.ai/v1",
            ).rstrip("/"),
            api_key_env="GROK_API_KEY",
            default_model=os.environ.get("TAMFIS_CODE_GROK_MODEL", "grok-4.6"),
            models=[
                "grok-4.6",
                "grok-4.3",
                "grok-4-fast",
                "grok-2-vision",
                "grok-imagine",
                "grok-imagine-video",
            ],
            # priority=6: last among the auto-routing chain (after
            # OpenRouter's 5) -- see the class-level comment above `.priority`
            # feeds provider selection directly via min(..., key=priority),
            # so this number, not just PRIORITY_ORDER's tuple position, is
            # what actually keeps Grok as the last-tried paid route.
            priority=6,
            weight=15,
            reasoning_supported=True,
            vision_supported=True,
            context_window=256000,
            coding_quality=5,
            tool_calling=True,
            structured_output=True,
            long_context=True,
        ),
        # Public subscription API. Unlike TIER_IV this endpoint is intended
        # for portable installs and authenticates with a user-owned key.
        # Tool calls are returned to the CLI so local workspace tools remain
        # local to the machine where Tamfis-Code is installed.
        ProviderType.TAMFIS: ProviderConfig(
            name="TamfisGPT Subscription API",
            base_url=os.environ.get(
                "TAMFIS_API_BASE",
                "https://gpt.tamfitronics.com/api/v1/openai",
            ).rstrip("/"),
            api_key_env="TAMFIS_API_KEY",
            default_model="tamfis-gpt-auto",
            # FIX 2026-08-11: this list used to be just ["tamfis-gpt-auto"],
            # a single placeholder id. resolve_public_model_alias() matches
            # a requested tier (e.g. "TamfisGPT Ultima") against
            # public_model_name(candidate) for each entry in `models` --
            # with only one placeholder candidate that could never equal
            # "TamfisGPT Ultima", every tier selection (Smart/Pro/Ultra/
            # Ultima) silently fell through to default_model and sent the
            # literal string "tamfis-gpt-auto" to the server regardless of
            # what --model was actually requested. The server-side gateway
            # (tier_ii_gateway/api/openai_compat.py's
            # _get_actual_model_from_registry, see its matching 2026-08-11
            # fix) now recognizes these exact ids and maps them to a real
            # model_tier/escalate request instead of always resolving via
            # generic mode="chat" auto-selection.
            models=[
                "tamfis-gpt-smart",
                "tamfis-gpt-pro",
                "tamfis-gpt-ultra",
                "tamfis-gpt-ultima",
            ],
            priority=4,
            weight=0,  # explicit-only compatibility route; excluded from AUTO
            context_window=128000,
            coding_quality=5,
            tool_calling=True,
            structured_output=True,
            long_context=True,
        ),
        # The shared TamfisGPT Tier IV orchestration service (tamgpt6,
        # tier_iv_orchestration/tamgpt_api.py) -- a "SINGLE entry point for
        # all model execution" that tamgpt6 itself already uses for its own
        # agentic/tool-calling flows. Exposed only as an OpenAI-compatible
        # /v1/chat/completions endpoint (confirmed live: no separate
        # routing-decision endpoint exists) -- treated as one more execution
        # provider rather than a pre-flight routing advisor. Retained as an
        # explicit remote compatibility route, not part of weighted AUTO.
        ProviderType.TIER_IV: ProviderConfig(
            name="TamfisGPT Tier IV",
            base_url=f"{os.environ.get('TAMFIS_TIER_IV_URL', 'http://127.0.0.1:9555').rstrip('/')}/v1",
            api_key_env="TAMFIS_ACCESS_TOKEN",
            default_model="",  # empty -- let Tier IV's own orchestrator pick, same as tamgpt6's own callers do when no model is pinned.
            models=[],
            priority=5,
            weight=0,  # explicit-only compatibility route; excluded from AUTO
            reasoning_supported=True,
            vision_supported=False,
            context_window=128000,
            coding_quality=5,
            # NOT True. Confirmed live against tamgpt_api.py/runtime.py:
            # the endpoint never reads an incoming `tools` field at all, and
            # when TamfisRuntime's OWN internal tool loop decides to call a
            # tool, it executes that tool itself server-side (against its
            # own ~91 MCP tools/remote_exec/file-gen registry) and never
            # returns a tool_calls response to the caller. tamfis-code's
            # local tools (read_file/write_file/edit/bash, running against
            # the user's own working directory) have no way to reach the
            # model through this provider, and any tool_calls this provider
            # did emit would never come back for local execution anyway.
            # Declaring True here used to make capability-aware routing
            # (routing.py's requires_tools) offer this provider for real
            # coding/edit/debug work, where the model would see none of the
            # CLI's own tools and either refuse or hallucinate a fake call.
            # False here means routing.py naturally keeps this provider to
            # tasks that don't need local tools (plain chat/Q&A) and falls
            # back to NVIDIA/OpenRouter/HF -- which DO honor real
            # client-side function-calling -- for everything else.
            tool_calling=False,
            structured_output=True,
            long_context=True,
        ),
        ProviderType.NVIDIA: ProviderConfig(
            name="NVIDIA NIM",
            base_url="https://integrate.api.nvidia.com/v1",
            api_key_env="NVIDIA_API_KEY",
            # 2026-07-26: re-prioritized after nemotron-3-nano-omni-30b-a3b-
            # reasoning (the prior default) was caught live wrapping a fake
            # tool call in CLI-flag/XML-ish text under real agentic use (see
            # runner_local.py's _FAKE_TOOL_CALL_FLAG_RE/_FAKE_TOOL_CALL_XML_RE
            # fix) -- a failure mode a plain single-call smoke test doesn't
            # surface, since that model also returns clean real tool_calls
            # on simple prompts. Re-verified all candidates below live
            # against this account immediately before reordering (real
            # chat-completions calls with tools attached, real tool_calls
            # returned, not guessed from NVIDIA's public catalog page):
            # nemotron-3-ultra-550b-a55b (1.8s), nemotron-3-super-120b-a12b
            # (6.1s), nemotron-3-nano-30b-a3b, llama-3.3-nemotron-super-49b-
            # v1.5, llama-3.3-nemotron-super-49b-v1, minimaxai/minimax-m3
            # all returned genuine tool_calls. ultra-550b-a55b was NVIDIA's
            # largest/most capable model verified at the time and became
            # the default then; super-120b-a12b (already separately
            # confirmed to handle reasoning_effort without hanging) is the
            # next fallback. The nano-omni-reasoning route stays
            # selectable -- do not remove it entirely, since an account
            # without ultra/super entitlement still needs a working route.
            #
            # DEFAULT CHANGED 2026-09-01 (owner directive, same session as
            # AUTO_PROVIDER_WEIGHTS's NVIDIA/Ollama Cloud rebalance above):
            # moonshotai/kimi-k3 is now the default instead of
            # nemotron-3-ultra-550b-a55b -- free/plan-covered like every
            # other NVIDIA route here, and the same model tamgpt6
            # (TamfisGPT) already made its own highest-weighted, highest-
            # priority NIM model for the identical reason.
            # nemotron-3-ultra-550b-a55b drops to first fallback, not
            # removed. Live-verified 2026-08-30 directly against
            # integrate.api.nvidia.com/v1/chat/completions with a real
            # account key -- three probes: (1) plain chat completion, 200
            # OK with real content; (2) a real function-calling probe
            # (get_weather tool), returned a genuine tool_calls event, not
            # narrated text; (3) NVIDIA's own catalog vision payload shape
            # (text + image_url content parts), returned an accurate
            # description of the real test image, not a refusal/generic
            # non-answer. Same underlying model also available via Ollama
            # Cloud (kimi-k3:cloud, see above) and HF (see
            # model_registry.py); this is a third, independently verified
            # route.
            default_model="moonshotai/kimi-k3",
            models=[
                "moonshotai/kimi-k3",
                "nvidia/nemotron-3-ultra-550b-a55b",
                "nvidia/nemotron-3-super-120b-a12b",
                "nvidia/nemotron-3-nano-30b-a3b",
                # NVIDIA-hosted, live-verified real tool_calls on this
                # account; high accuracy on reasoning/tool-calling per
                # NVIDIA's own catalog, kept as mid-tier fallbacks below the
                # newer nemotron-3 family.
                "nvidia/llama-3.3-nemotron-super-49b-v1.5",
                "nvidia/llama-3.3-nemotron-super-49b-v1",
                # Third-party (MiniMax AI) multimodal MoE hosted on NVIDIA
                # NIM, live-verified real tool_calls on this account.
                # Marketplace models have previously turned out to be a
                # per-account entitlement gap disguised as a working route
                # (see the kimi-k2.6 404 fix) -- kept selectable, not
                # promoted to default, until it has more real-world use.
                "minimaxai/minimax-m3",
                # Confirmed live: real tool_calls response (not narrated
                # text), reasoning_effort and reasoning_budget both work
                # without hanging. Demoted from default after the fake-
                # tool-call-as-text incident above; still selectable.
                "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning",
                # NVIDIA currently exposes these DeepSeek V4 hosted routes
                # as mature coding/agent models. They remain selectable
                # fallbacks; the verified Nemotron routes stay the default
                # until an account-specific smoke test proves otherwise.
                # deepseek-ai/deepseek-v4-flash removed 2026-08-08: reached
                # NVIDIA NIM end-of-life 2026-08-07 (HTTP 410 confirmed live
                # in tamgpt6's intent classifier), no longer callable.
                "deepseek-ai/deepseek-v4-pro",
                "meta/llama-3.1-405b-instruct",
                "meta/llama-3.1-70b-instruct",
                "moonshotai/kimi-k2.6",
                # moonshotai/kimi-k3 moved to the front of this list as
                # default_model above (2026-09-01) -- see that comment for
                # the verification details, not repeated here.
                "mistralai/mistral-large-2-123b",
                "google/gemma-2-27b-it",
                "microsoft/phi-3-medium-128k-instruct",
            ],
            # priority=0 (2026-08-08, was 3): NIM made the top-priority
            # auto-routed provider -- reliable free tier vs. Ollama Cloud's
            # multi-day weekly-quota 429s. See ProviderType.OLLAMA_CLOUD's
            # priority comment above for the incident this responds to.
            priority=0,
            weight=40,
            reasoning_supported=True,
            # FIX 2026-08-30: was False -- true before kimi-k3 above was
            # added, since none of this bucket's other models had a
            # confirmed vision route. Now live-verified true for at least
            # one model in this provider's bucket (see the comment above).
            vision_supported=True,
            context_window=128000,
            coding_quality=5,
            tool_calling=True,
            structured_output=True,
            long_context=True,
        ),
        ProviderType.HF: ProviderConfig(
            name="Hugging Face",
            base_url="https://router.huggingface.co/v1",
            api_key_env="HF_TOKEN",
            # HF Router delegates these official models to its configured
            # Inference Provider. Prefer the strong Qwen 3.6 coding route;
            # older small instruct models remain explicit fallbacks only.
            default_model="Qwen/Qwen3.6-35B-A3B",
            models=[
                "Qwen/Qwen3.6-35B-A3B",
                "Qwen/Qwen3.6-27B",
                "Qwen/Qwen3-Coder-480B-A35B-Instruct",
                "deepseek-ai/DeepSeek-V4-Pro",
                "deepseek-ai/DeepSeek-V4-Flash",
                "meta-llama/Llama-3.2-3B-Instruct",
                "mistralai/Mistral-7B-Instruct-v0.3",
                "microsoft/Phi-3.5-vision-instruct",
                "meta-llama/Llama-3.2-11B-Vision-Instruct",
                "Qwen/Qwen2-VL-7B-Instruct",
                # Confirmed live (real chat-completions call, real 200) on
                # HF's router -- exact casing matters here: the lowercase
                # "moonshotai/kimi-k2.6" (NVIDIA's own id casing) 400s with
                # model_not_found on HF, only this exact casing resolves.
                # Not the default -- offered as an additional selectable
                # route so NVIDIA's per-account kimi-k2.6 entitlement gap
                # (see providers.py's NVIDIA default_model comment) doesn't
                # have to mean losing access to this model entirely.
                "moonshotai/Kimi-K2.6",
            ],
            priority=4,
            weight=15,
            reasoning_supported=False,
            vision_supported=True,
            context_window=262144,
            coding_quality=5,
            tool_calling=True,
            structured_output=True,
            long_context=True,
        ),
        ProviderType.OPENROUTER: ProviderConfig(
            name="OpenRouter",
            base_url="https://openrouter.ai/api/v1",
            api_key_env="OPENROUTER_API_KEY",
            # Paid coding/repair work gets a coder-tuned route. Routine work
            # still uses the free model below, so this does not turn every
            # inspection into a paid request.
            default_model="qwen/qwen3-coder",
            # Used for plain chat/Q&A/inspection -- anything that isn't
            # deep research or genuinely demanding coding/analysis work --
            # so routine turns don't spend paid OpenRouter credits at all.
            # See _task_needs_paid_tier for exactly which tasks escalate
            # to `default_model` instead.
            free_model="openrouter/free",
            models=[
                # Deliberately excludes openai/* defaults.
                "google/gemini-2.5-flash",
                "qwen/qwen3-coder",
                "deepseek/deepseek-v4-pro",
                "deepseek/deepseek-v4-flash",
                "deepseek/deepseek-chat-v3-0324",
                "anthropic/claude-sonnet-4",
                "anthropic/claude-3.5-haiku",
                "openai/gpt-4.1-mini",
                "openai/gpt-4.1-nano",
                "openrouter/free",
                "qwen/qwen3-coder:free",
                "meta-llama/llama-3.3-70b-instruct:free",
                "mistralai/mistral-7b-instruct:free",
                # Confirmed live: present in OpenRouter's real /v1/models
                # catalog and a real chat-completions call against it
                # returned a genuine billing (402 insufficient credits)
                # response rather than a bad-model/400 -- i.e. OpenRouter
                # itself accepts and would run this model, this account
                # just has no credit balance right now. Not the default --
                # an additional selectable route for the same reason as
                # HF's entry above (NVIDIA's per-account kimi-k2.6
                # entitlement gap, see providers.py's NVIDIA default_model
                # comment).
                "moonshotai/kimi-k2.6",
                # Real xAI-hosted Grok, via OpenRouter -- same ids already
                # live-verified against OpenRouter's /v1/models catalog for
                # TamfisGPT's own Tier IV routing (see tamgpt6's
                # orchestration.yaml grok-4.6-or/grok-4.3-or entries, added
                # 2026-08-03). xAI does not distribute Grok weights for
                # local/Ollama use, so OpenRouter is the only
                # policy-permitted route to the genuine model (standing
                # policy: never call a vendor's native API directly -- see
                # providers.py module docstring / PRIORITY_ORDER comment).
                "x-ai/grok-4.6",
                "x-ai/grok-4.3",
            ],
            # OpenRouter is last in AUTO because paid coding routes can fail
            # with HTTP 402 when the account has no credits. It remains
            # explicitly selectable and is still tried after HF/NVIDIA.
            priority=5,
            weight=15,
            reasoning_supported=True,
            vision_supported=True,
            context_window=200000,
            coding_quality=4,
            tool_calling=True,
            structured_output=True,
            long_context=True,
        ),
    }

    def __init__(self, *, runtime_mode: str = "standalone") -> None:
        self.runtime_mode = (runtime_mode or "standalone").strip().lower()
        if self.runtime_mode not in {"standalone", "remote"}:
            raise ValueError(f"Unsupported provider runtime mode: {runtime_mode!r}")
        self.clients: Dict[ProviderType, AsyncOpenAI] = {}
        self.config = self._load_config()
        self._init_clients()

    @property
    def routing_order(self) -> tuple[ProviderType, ...]:
        """Provider order permitted for this runtime boundary.

        Standalone Tamfis-Code must never escape into the shared TamfisGPT
        Tier IV service: that service owns a different workspace allow-list
        and cannot execute this process's local MCP tools.  Remote mode may
        still expose Tier IV explicitly.
        """
        if getattr(self, "runtime_mode", "standalone") == "remote":
            return self.PRIORITY_ORDER
        return tuple(p for p in self.PRIORITY_ORDER if p != ProviderType.TIER_IV)

    def provider_allowed(self, provider: ProviderType) -> bool:
        return provider in self.routing_order

    def fallback_chain_names(self, current: ProviderType) -> list[str]:
        return [provider.value for provider in self.routing_order if provider != current and self._fallback_provider_allowed(provider)]

    def auto_fallback_enabled(self) -> bool:
        """Return whether AUTO may use safe cross-provider recovery."""
        return os.environ.get("TAMFIS_CODE_DISABLE_PROVIDER_FALLBACK", "false").strip().lower() != "true"

    def paid_fallback_enabled(self) -> bool:
        return os.environ.get("TAMFIS_CODE_ALLOW_PROVIDER_FALLBACK", "false").strip().lower() == "true"

    def ollama_cloud_is_premium_primary(self) -> bool:
        """Deprecated compatibility hook; AUTO is always weighted now.

        Premium entitlement still controls Ollama model access, but no
        environment flag may bypass the operator-approved 40/15/15/15/15
        provider distribution. Explicit ``--provider ollama_cloud`` remains
        available when a caller genuinely wants to pin that route.
        """
        return False

    def _fallback_provider_allowed(self, provider: ProviderType) -> bool:
        # These are the agreed automatic recovery providers. OpenRouter is
        # permitted automatically only where a free route exists, unless paid
        # fallback is explicitly enabled.
        if provider in self.AUTO_PROVIDER_WEIGHTS:
            return True

        # The Tamfis subscription/local service is retained for explicit
        # compatibility only. It is not one of the operator-approved cloud
        # AUTO/fallback routes and must not dilute their 100% distribution.
        if provider in {ProviderType.TAMFIS, ProviderType.TIER_IV, ProviderType.LOCAL}:
            return False

        config = self.PROVIDERS.get(provider)
        return bool(config and config.free_model) or self.paid_fallback_enabled()

    def _load_config(self) -> Dict[str, bool]:
        """Read provider enable/disable flags from the environment."""

        config: Dict[str, bool] = {}
        for provider in self.PROVIDERS:
            env_var = f"TAMFIS_PROVIDER_{provider.value.upper()}_ENABLED"
            config[provider.value] = (
                self.provider_allowed(provider)
                and os.environ.get(env_var, "true").strip().lower() == "true"
            )
        return config

    def _get_api_key(self, provider_type: ProviderType) -> Optional[str]:
        """Return a provider API key, or None when not configured."""

        if not self.provider_allowed(provider_type):
            return None

        config = self.PROVIDERS.get(provider_type)
        if config is None:
            return None

        if provider_type == ProviderType.TIER_IV:
            # No auth is enforced by the real endpoint (internal-only
            # service) -- a token is forwarded if present, but the OpenAI
            # SDK still needs a non-empty api_key string to construct.
            return os.environ.get(config.api_key_env, "").strip() or "tier-iv-local"

        if provider_type == ProviderType.OLLAMA_CLOUD:
            # Authentication to cloud models is owned by the signed-in local
            # Ollama daemon. The OpenAI SDK still requires a non-empty value,
            # but the local Ollama endpoint ignores it. Preserve the real key
            # in the environment for direct native API use elsewhere.
            return os.environ.get(config.api_key_env, "").strip() or "ollama"

        if provider_type == ProviderType.TAMFIS:
            # An explicitly supplied developer key wins.  Otherwise reuse
            # the renewable credential created by `tamfis-code login`; the
            # subscription gateway accepts either credential type while all
            # tools and orchestration remain in this local runtime.
            developer_key = os.environ.get(config.api_key_env, "").strip()
            if developer_key:
                return developer_key
            try:
                from .api_client import load_secure_credentials

                credentials = load_secure_credentials()
            except Exception:
                credentials = None
            return str(getattr(credentials, "access_token", "") or "").strip() or None

        key = os.environ.get(config.api_key_env, "").strip()
        return key or None

    def _has_valid_api_key(self, provider_type: ProviderType) -> bool:
        """Perform a conservative configuration-level key check."""

        if not self.provider_allowed(provider_type):
            return False

        if provider_type == ProviderType.TIER_IV:
            return self._check_tier_iv_available()

        if provider_type == ProviderType.OLLAMA_CLOUD:
            return self._check_ollama_available()

        config = self.PROVIDERS.get(provider_type)
        if config is None:
            return False

        key = self._get_api_key(provider_type)
        if not key or len(key) < 8:
            return False

        upper_key = key.upper()
        placeholder_markers = (
            "YOUR_",
            "CHANGE_ME",
            "REPLACE_ME",
            "_API_KEY",
            "EXAMPLE",
            "DUMMY",
        )
        return not any(marker in upper_key for marker in placeholder_markers)

    def _check_ollama_available(self) -> bool:
        """Return True when the local Ollama daemon is reachable.

        A successful daemon health check does not guarantee that every cloud
        model is included in the current account plan. A model-specific 403 is
        therefore handled as a retryable provider error and AUTO falls back to
        NVIDIA, Hugging Face, then OpenRouter.
        """

        explicit = os.environ.get("TAMFIS_PROVIDER_OLLAMA_CLOUD_ENABLED")
        if explicit is not None and explicit.strip().lower() != "true":
            return False

        config = self.PROVIDERS.get(ProviderType.OLLAMA_CLOUD)
        if config is None:
            return False

        import httpx

        base = config.base_url
        if base.endswith("/v1"):
            base = base[:-3]

        try:
            response = httpx.get(f"{base}/api/tags", timeout=3.0)
            return response.status_code == 200
        except httpx.HTTPError:
            return False

    def _check_tier_iv_available(self) -> bool:
        """Return True when Tier IV is reachable, unless explicitly disabled.

        On by default (mirrors how every other provider here is
        enabled-by-default and gated only by its own reachability/key
        check) -- the 2s health-probe below is the actual gate against
        probing a box that doesn't run it, not an env var the user has to
        know to set. TAMFIS_TIER_IV_ENABLED=false remains the explicit
        opt-out.
        """

        explicit = os.environ.get("TAMFIS_TIER_IV_ENABLED")
        if explicit is not None and explicit.strip().lower() != "true":
            return False

        config = self.PROVIDERS.get(ProviderType.TIER_IV)
        if config is None:
            return False

        import httpx

        health_url = f"{config.base_url.rsplit('/v1', 1)[0]}/health"
        try:
            response = httpx.get(health_url, timeout=2.0)
            return response.status_code == 200
        except httpx.HTTPError:
            return False

    def _init_clients(self) -> None:
        """Initialise clients only for enabled and configured providers."""

        for provider_type, config in self.PROVIDERS.items():
            if not self.config.get(provider_type.value, True):
                continue

            if not self._has_valid_api_key(provider_type):
                continue

            try:
                self.clients[provider_type] = AsyncOpenAI(
                    base_url=config.base_url,
                    api_key=self._get_api_key(provider_type),
                    timeout=120.0,
                    # The standalone runner owns stream recovery so retries
                    # are visible, back off sensibly, retain partial output,
                    # and update the durable turn checkpoint. SDK-level
                    # retries are immediate and opaque to all three.
                    max_retries=0,
                )
            except Exception:
                # Availability is reported by list_available_providers().
                continue

    def get_client(self, provider: ProviderType) -> Optional[AsyncOpenAI]:
        """Return a client, resolving AUTO through capability routing."""

        resolved = (
            self._select_best_provider()
            if provider == ProviderType.AUTO
            else provider
        )
        if not self.provider_allowed(resolved):
            return None
        return self.clients.get(resolved)

    def select_model(
        self,
        config: ProviderConfig,
        task_profile: Optional["TaskProfile"] = None,
    ) -> str:
        """Pick which of a provider's models to actually call.

        Providers with no `free_model` (everything except OpenRouter today)
        are unaffected -- this always returns `default_model` for them.
        For OpenRouter, routine tasks get the free-tier model and only
        deep research / complex coding-or-analysis tasks (see
        _task_needs_paid_tier) escalate to the paid `default_model`, so
        ordinary chat/inspection turns don't spend credits at all.
        """
        if config.name == "Ollama Cloud":
            # FIX 2026-08-08: TAMFIS_CODE_OLLAMA_PREMIUM used to gate BOTH
            # (a) whether AUTO is allowed to force Ollama Cloud as its
            # exclusive primary and (b) which model gets
            # picked once Ollama Cloud IS being called. Those are unrelated
            # concerns: kimi-k2.7-code:cloud is the included-plan coding
            # route (confirmed not an extra-usage cost, unlike kimi-k3
            # below), so there is no reason to withhold it just because
            # Ollama isn't forced as AUTO's primary. It is now always the
            # default model whenever Ollama Cloud is used -- via automatic
            # fallback after NIM, or explicit --provider selection -- while
            # the genuinely extra-cost kimi-k3 escalation stays gated
            # behind TAMFIS_CODE_OLLAMA_PREMIUM + TAMFIS_CODE_OLLAMA_EXTRA_USAGE
            # exactly as before.
            premium_enabled = os.environ.get(
                "TAMFIS_CODE_OLLAMA_PREMIUM",
                "false",
            ).strip().lower() in {"1", "true", "yes", "on"}

            if premium_enabled:
                # Kimi K3 consumes Ollama "extra usage" on accounts where it
                # is not included in plan usage. Never select it merely
                # because Ollama Cloud is being used. A machine
                # administrator must explicitly enable extra usage, and the
                # task must independently qualify as heavy.
                extra_usage_enabled = os.environ.get(
                    "TAMFIS_CODE_OLLAMA_EXTRA_USAGE",
                    "false",
                ).strip().lower() in {"1", "true", "yes", "on"}
                if extra_usage_enabled and _task_needs_paid_tier(task_profile):
                    return os.environ.get(
                        "TAMFIS_CODE_OLLAMA_HEAVY_MODEL",
                        "kimi-k3:cloud",
                    )

            # The included-plan Kimi coding route is the all-round default
            # whenever Ollama Cloud handles the request, whether or not
            # Ollama is AUTO's forced primary. It handles routine and heavy
            # coding without silently escalating to an extra-usage-only
            # model.
            return os.environ.get(
                "TAMFIS_CODE_OLLAMA_CODING_MODEL",
                "kimi-k2.7-code:cloud",
            )

        if config.name == "NVIDIA NIM":
            # A failed NIM deployment is a model-route failure first, not a
            # reason to abandon NIM. The bounded cooldown automatically
            # makes it probe-eligible again later.
            candidates = [config.default_model, *config.models]
            for candidate in dict.fromkeys(item for item in candidates if item):
                if self.route_is_healthy(ProviderType.NVIDIA, candidate):
                    return candidate

        if config.free_model and not _task_needs_paid_tier(task_profile):
            return config.free_model
        return config.default_model

    @staticmethod
    def routing_telemetry() -> RoutingTelemetry:
        return _ROUTING_TELEMETRY

    @staticmethod
    def reset_runtime_routing_state() -> None:
        """Clear process-local health/metrics for deterministic diagnostics."""
        with _HEALTH_LOCK:
            _ROUTE_HEALTH.clear()
            fresh = RoutingTelemetry()
            _ROUTING_TELEMETRY.__dict__.update(fresh.__dict__)

    @staticmethod
    def route_is_healthy(provider: ProviderType, model: str = "") -> bool:
        now = time.monotonic()
        with _HEALTH_LOCK:
            state = _ROUTE_HEALTH.get((provider.value, model or "*"))
            return state is None or state.circuit_open_until <= now

    def record_route_attempt(self, provider: ProviderType, model: str) -> None:
        with _HEALTH_LOCK:
            _increment(_ROUTING_TELEMETRY.provider_requests, provider.value)

    def record_route_success(self, provider: ProviderType, model: str) -> None:
        now = time.monotonic()
        with _HEALTH_LOCK:
            state = _ROUTE_HEALTH.setdefault((provider.value, model), RouteHealth())
            state.successes += 1
            state.failures = 0
            state.circuit_open_until = 0.0
            state.last_success = now
            provider_state = _ROUTE_HEALTH.setdefault((provider.value, "*"), RouteHealth())
            provider_state.failures = 0
            provider_state.circuit_open_until = 0.0
            provider_state.last_success = now
            _increment(_ROUTING_TELEMETRY.provider_successes, provider.value)
            if provider == ProviderType.NVIDIA:
                _ROUTING_TELEMETRY.nim_successful_requests += 1

    def record_route_failure(
        self,
        provider: ProviderType,
        model: str,
        exc: Exception,
        *,
        stream: bool = False,
        tool_call: bool = False,
    ) -> None:
        status = self.provider_error_status(exc)
        deterministic = status in {400, 401, 402, 403, 404}
        cooldown = 300.0 if deterministic else 30.0
        now = time.monotonic()
        with _HEALTH_LOCK:
            state = _ROUTE_HEALTH.setdefault((provider.value, model), RouteHealth())
            state.failures += 1
            state.last_failure = now
            state.last_error_class = type(exc).__name__
            state.circuit_open_until = max(state.circuit_open_until, now + cooldown)
            # Non-NIM providers currently expose a single selected route per
            # task, so cool the provider immediately. NIM is demoted only
            # after every configured sibling model is unavailable.
            provider_config = self.PROVIDERS.get(provider)
            configured_models = list(dict.fromkeys(
                [
                    getattr(provider_config, "default_model", ""),
                    *(getattr(provider_config, "models", ()) or ()),
                ]
            ))
            all_models_unhealthy = bool(configured_models) and all(
                not self.route_is_healthy(provider, candidate)
                for candidate in configured_models if candidate
            )
            if provider != ProviderType.NVIDIA or all_models_unhealthy:
                provider_state = _ROUTE_HEALTH.setdefault((provider.value, "*"), RouteHealth())
                provider_state.failures += 1
                provider_state.last_failure = now
                provider_state.last_error_class = type(exc).__name__
                provider_state.circuit_open_until = max(
                    provider_state.circuit_open_until, now + cooldown,
                )
            _increment(_ROUTING_TELEMETRY.provider_failures, provider.value)
            if stream:
                _increment(_ROUTING_TELEMETRY.provider_stream_failures, provider.value)
            if tool_call:
                _increment(_ROUTING_TELEMETRY.provider_tool_failures, provider.value)

    def record_fallback(self, provider: ProviderType) -> None:
        with _HEALTH_LOCK:
            _increment(_ROUTING_TELEMETRY.provider_fallbacks, provider.value)
            if provider == ProviderType.NVIDIA:
                _ROUTING_TELEMETRY.nim_fallback_requests += 1

    def provider_health_status(self, provider: ProviderType) -> str:
        if not self.route_is_healthy(provider, "*"):
            return "circuit_open"
        config = self.PROVIDERS.get(provider)
        models = list(dict.fromkeys([
            getattr(config, "default_model", ""),
            *(getattr(config, "models", ()) or ()),
        ]))
        if any(model and not self.route_is_healthy(provider, model) for model in models):
            return "degraded"
        return "healthy"

    def resolve_route(
        self,
        provider: ProviderType,
        task_profile: Optional["TaskProfile"] = None,
        *,
        quality_mode: str = "quality",
    ) -> tuple[ProviderType, ProviderConfig]:
        """Resolve AUTO to a concrete provider and configuration."""

        if provider == ProviderType.AUTO and self.ollama_cloud_is_premium_primary():
            resolved = ProviderType.OLLAMA_CLOUD
            if resolved not in self.clients:
                raise ValueError(
                    "Ollama Cloud is enabled as the AUTO primary route, but its local daemon is unavailable "
                    f"at {self.PROVIDERS[resolved].base_url}. Start/sign in to Ollama Cloud, or explicitly "
                    "select another provider with --provider."
                )
        else:
            resolved = (
                self._select_best_provider(
                    task_profile, quality_mode=quality_mode,
                    allowed_providers=(
                        tuple(provider for provider in self.routing_order if self._fallback_provider_allowed(provider))
                        if provider == ProviderType.AUTO else None
                    ),
                )
                if provider == ProviderType.AUTO
                else provider
            )

        if not self.provider_allowed(resolved):
            raise ValueError(
                f"Provider {resolved.value!r} is not available in {self.runtime_mode} runtime mode"
            )
        config = self.PROVIDERS.get(resolved)
        if config is None:
            raise ValueError(f"Unknown provider: {resolved.value}")

        return resolved, config

    def _select_best_provider(
        self,
        task_profile: Optional["TaskProfile"] = None,
        *,
        quality_mode: str = "quality",
        allowed_providers: Optional[tuple[ProviderType, ...]] = None,
    ) -> ProviderType:
        """Select the first healthy capability-compatible AUTO provider."""

        available = [
            provider
            for provider in self.routing_order
            if provider in self.AUTO_PROVIDER_WEIGHTS
            if allowed_providers is None or provider in allowed_providers
            if provider in self.clients and self._has_valid_api_key(provider)
            if self.route_is_healthy(provider, "*")
        ]

        if not available:
            raise ValueError("No configured AI provider is available")

        requires_tools = bool(
            getattr(task_profile, "requires_tools", False)
        )
        requires_long_context = bool(
            getattr(task_profile, "requires_long_context", False)
        )
        candidates = available

        eligible: List[ProviderType] = []
        for provider in candidates:
            config = self.PROVIDERS[provider]

            if requires_tools and not config.tool_calling:
                continue
            if requires_long_context and not config.long_context:
                continue

            eligible.append(provider)

        if not eligible:
            eligible = candidates

        del quality_mode  # retained in the public signature for compatibility
        with _HEALTH_LOCK:
            if ProviderType.NVIDIA in eligible:
                _ROUTING_TELEMETRY.nim_eligible_requests += 1
        selected = next(provider for provider in self.routing_order if provider in eligible)
        with _HEALTH_LOCK:
            if selected == ProviderType.NVIDIA:
                _ROUTING_TELEMETRY.nim_selected_requests += 1
        return selected

    @staticmethod
    def provider_error_status(exc: Exception) -> Optional[int]:
        """Extract an HTTP status code from common provider exceptions."""

        status = getattr(exc, "status_code", None)
        if isinstance(status, int):
            return status

        response = getattr(exc, "response", None)
        response_status = getattr(response, "status_code", None)
        return response_status if isinstance(response_status, int) else None

    @classmethod
    def is_retryable_provider_error(cls, exc: Exception) -> bool:
        """Return True when automatic routing should try another provider."""

        status = cls.provider_error_status(exc)
        if status is not None:
            # NVIDIA can return HTTP 400 when a model's registered function
            # is degraded or unavailable. This is a route/model capability
            # failure, not a malformed user request; AUTO must continue with
            # the next eligible provider instead of stopping the task.
            if status == 400:
                message = str(exc).lower()
                if (
                    "degraded function cannot be invoked" in message
                    or "function cannot be invoked" in message
                ):
                    return True
            return status in {
                401,
                402,
                403,
                # Inference APIs use 404 for a model/deployment that was
                # removed or is unavailable to the current account. Some
                # providers (notably NVIDIA NIM) return no response detail
                # beyond ``Error code: 404``, so message-shape matching
                # cannot reliably identify that live failure. Endpoint
                # paths are owned by our static provider config; at request
                # time the useful recovery is the next provider/model.
                404,
                408,
                409,
                # 422 (Unprocessable Entity) was previously NOT in this set,
                # so it hit the exact "Provider streaming failed ... type
                # `continue` to resume" stop path documented below in
                # runner_local.py -- with configured fallback routes sitting
                # unused, same class of bug as the 429 case that comment
                # describes. Like the HTTP 400 special-case above, a 422
                # here is a provider/model-specific rejection of this
                # request's shape (an unsupported tool-schema field, a
                # parameter that provider doesn't accept), not evidence the
                # user's actual task is malformed -- the next provider may
                # well accept the same logical request. Route around it
                # instead of stopping, matching TamfisGPT's own multi-
                # provider fallback behavior for the same status class.
                422,
                425,
                429,
            } or status >= 500

        explicit_retryable = getattr(exc, "retryable", None)
        if isinstance(explicit_retryable, bool):
            return explicit_retryable

        message = str(exc).lower()
        retryable_markers = (
            "insufficient credits",
            "payment required",
            "quota",
            "rate limit",
            "resourceexhausted",
            "resource_exhausted",
            "total request limit reached",
            "worker local",
            "worker capacity",
            "internal_server_error",
            "service unavailable",
            "overloaded",
            "temporarily unavailable",
            "connection error",
            "connection refused",
            # Some OpenAI-compatible SDKs discard the HTTP/body detail once
            # an already-open response stream fails and surface only this
            # wrapper.  It is still a transport/provider-route failure, not
            # evidence that the coding task itself failed.  In AUTO mode the
            # runner must therefore try the next configured provider.
            "error occurred during streaming",
            "streaming error",
            "stream disconnected",
            "timed out",
            "timeout",
            "authentication",
            "unauthorized",
            "forbidden",
            # NVIDIA NIM's detailed shape for "this account has no
            # deployment access to this specific model". Kept for SDK
            # wrappers that lose the numeric HTTP status but preserve the
            # response text.
            "not found for account",
            "degraded function cannot be invoked",
            "function cannot be invoked",
        )
        return any(marker in message for marker in retryable_markers)

    @classmethod
    def is_quota_or_rate_limit_error(cls, exc: Exception) -> bool:
        """True specifically for quota-exhaustion/rate-limit failures.

        FIX 2026-08-07: narrower than is_retryable_provider_error, which also
        returns True for daemon-unavailability-class errors (connection
        refused, timeouts). ollama_cloud_is_premium_primary's fallback-block
        in fallback_candidates() was written specifically for the
        daemon-down case ("caller receives a concrete Ollama availability
        error instead") -- a 429 weekly-quota exhaustion is a different
        failure mode where that reasoning doesn't apply, and callers need to
        distinguish the two to pass allow_premium_primary correctly.
        """
        status = cls.provider_error_status(exc)
        if status is not None:
            return status in {402, 429}

        message = str(exc).lower()
        quota_markers = (
            "insufficient credits",
            "payment required",
            "quota",
            "rate limit",
            "weekly usage limit",
            "resourceexhausted",
            "resource_exhausted",
            "total request limit reached",
        )
        return any(marker in message for marker in quota_markers)

    def fallback_candidates(
        self,
        current: ProviderType,
        task_profile: Optional["TaskProfile"] = None,
        *,
        allow_premium_primary: bool = False,
    ) -> List[ProviderType]:
        """Return usable alternatives in canonical policy order."""

        if (
            self.ollama_cloud_is_premium_primary()
            and ProviderType.OLLAMA_CLOUD in self.clients
            and not allow_premium_primary
        ):
            return []

        requires_tools = bool(
            getattr(task_profile, "requires_tools", False)
        )
        requires_long_context = bool(
            getattr(task_profile, "requires_long_context", False)
        )

        candidates: List[ProviderType] = []
        # A failed NIM model must not immediately eject the task from NIM.
        # The caller records the failed route before asking for candidates;
        # select_model() will therefore choose a healthy sibling deployment.
        if current == ProviderType.NVIDIA:
            config = self.PROVIDERS[current]
            now = time.monotonic()
            with _HEALTH_LOCK:
                failed_nim_model_exists = any(
                    provider_name == current.value
                    and model_name != "*"
                    and state.circuit_open_until > now
                    for (provider_name, model_name), state in _ROUTE_HEALTH.items()
                )
            healthy_models = [
                model
                for model in dict.fromkeys([config.default_model, *config.models])
                if model and self.route_is_healthy(current, model)
            ]
            if failed_nim_model_exists and healthy_models:
                candidates.append(current)
        for provider in self.routing_order:
            if provider == current:
                continue
            if provider not in self.clients:
                continue
            if not self._has_valid_api_key(provider):
                continue
            if not self.route_is_healthy(provider, "*"):
                continue

            config = self.PROVIDERS[provider]

            if not self._fallback_provider_allowed(provider):
                continue

            if requires_tools and not config.tool_calling:
                continue
            if requires_long_context and not config.long_context:
                continue

            candidates.append(provider)

        return candidates

    def list_available_providers(self) -> List[Dict[str, Any]]:
        """Return complete routing metadata for every configured provider."""

        result: List[Dict[str, Any]] = []
        for provider_type in self.routing_order:
            config = self.PROVIDERS[provider_type]
            enabled = self.config.get(provider_type.value, True)
            valid_key = self._has_valid_api_key(provider_type)
            client_initialised = provider_type in self.clients
            health_status = self.provider_health_status(provider_type)
            available = enabled and valid_key and client_initialised and health_status != "circuit_open"

            key = self._get_api_key(provider_type)
            key_preview = "local"
            if provider_type != ProviderType.TIER_IV:
                key_preview = f"{key[:8]}..." if key else "Not set"

            result.append(
                {
                    "name": config.name,
                    "type": provider_type.value,
                    "enabled": enabled,
                    "available": available,
                    "healthy": available and health_status == "healthy",
                    "health_status": health_status if enabled and valid_key else "unconfigured",
                    "client_initialised": client_initialised,
                    "default_model": config.default_model,
                    "models": list(config.models),
                    "priority": config.priority,
                    "weight": config.weight,
                    "coding_quality": config.coding_quality,
                    "tool_calling": config.tool_calling,
                    "structured_output": config.structured_output,
                    "long_context": config.long_context,
                    "context_window": config.context_window,
                    "local_only": config.local_only,
                    "api_key_set": valid_key,
                    "reasoning_supported": config.reasoning_supported,
                    "vision_supported": config.vision_supported,
                    "key_preview": key_preview,
                }
            )

        return result

    async def chat_completion(
        self,
        provider: ProviderType,
        messages: List[Dict[str, Any]],
        model: Optional[str] = None,
        stream: bool = True,
        temperature: float = 0.10,
        max_tokens: int = 16384,
        reasoning_effort: Optional[str] = "high",
        task_profile: Optional["TaskProfile"] = None,
        allow_fallback: bool = True,
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        """Yield text from one provider, with safe automatic fallback."""

        resolved, config = self.resolve_route(
            provider,
            task_profile,
            quality_mode=str(
                getattr(
                    task_profile,
                    "preferred_quality_tier",
                    "quality",
                )
            ),
        )
        client = self.clients.get(resolved)
        if client is None:
            raise ValueError(f"Provider {resolved.value} is not available")

        from .public_identity import resolve_public_model_alias

        selected_model = resolve_public_model_alias(
            model,
            models=config.models,
            default_model=config.default_model,
            free_model=config.free_model,
        ) or (
            self.select_model(config, task_profile)
            if resolved == ProviderType.OLLAMA_CLOUD
            else (
                config.free_model
                if config.free_model and not self.paid_fallback_enabled()
                else self.select_model(config, task_profile)
            )
        )

        request_kwargs: Dict[str, Any] = {
            "model": selected_model,
            "messages": system_messages_first(messages),
            "stream": stream,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        # Some OpenAI-compatible providers reject unsupported reasoning fields.
        # Tier IV's OrchestrationContext explicitly reads reasoning_effort,
        # so it's safe to forward there too.
        if reasoning_effort and reasoning_effort_capable(resolved, selected_model):
            request_kwargs["reasoning_effort"] = reasoning_effort

        request_kwargs.update(kwargs)

        # Tier IV's OrchestrationContext reads `mode` (data.get("mode", "auto"))
        # to pick a category-vetted, weight-ordered model pool via
        # TamfisRuntime._resolve_category()/_category_candidates(). Leaving
        # it unset defaults to "auto", which falls through to
        # model_catalog.py's select_model() drawing a WEIGHTED RANDOM model
        # across every category (chat/coding/research/academic/business/
        # project) with no bias toward coding models at all -- confirmed
        # live: three calls in ~90s landed on three different models
        # (qwen3-next-80b, mistral-large-3, nemotron-3-super). A coding CLI
        # has no business rolling that dice on every request.
        # `mode` is not a real OpenAI SDK parameter (unlike reasoning_effort
        # above) -- the SDK's create() has a fixed, fully-typed signature
        # with no **kwargs passthrough, so it must travel via extra_body,
        # which the SDK merges into the literal outgoing JSON body.
        if resolved == ProviderType.TIER_IV:
            from .runtime.telemetry import current_trace_id
            extra_body = dict(request_kwargs.get("extra_body") or {})
            extra_body.setdefault("mode", _tier_iv_mode_for_task(task_profile))
            extra_body.setdefault("trace_id", current_trace_id())
            metadata = dict(extra_body.get("metadata") or {})
            metadata.setdefault("trace_id", current_trace_id())
            metadata.setdefault("source_system", "tamfis-code")
            metadata.setdefault("ai_component", "main_completion")
            extra_body["metadata"] = metadata
            request_kwargs["extra_body"] = extra_body

        self.record_route_attempt(resolved, selected_model)
        from .runtime.telemetry import record_usage, span
        try:
            with span(
                "provider.invoke",
                provider=resolved.value,
                model=selected_model,
                operation="chat_completion",
            ):
                response = await client.chat.completions.create(**request_kwargs)

                if stream:
                    async for chunk in response:
                        usage = getattr(chunk, "usage", None)
                        if usage is not None:
                            record_usage(input_tokens=getattr(usage, "prompt_tokens", None),
                                         output_tokens=getattr(usage, "completion_tokens", None),
                                         reasoning_tokens=getattr(getattr(usage, "completion_tokens_details", None), "reasoning_tokens", None),
                                         cached_input_tokens=getattr(getattr(usage, "prompt_tokens_details", None), "cached_tokens", None))
                        if not chunk.choices:
                            continue
                        content = chunk.choices[0].delta.content
                        if content:
                            yield content
                    self.record_route_success(resolved, selected_model)
                    return

                if response.choices:
                    content = response.choices[0].message.content
                    if content:
                        yield content
                usage = getattr(response, "usage", None)
                if usage is not None:
                    record_usage(input_tokens=getattr(usage, "prompt_tokens", None),
                                 output_tokens=getattr(usage, "completion_tokens", None),
                                 reasoning_tokens=getattr(getattr(usage, "completion_tokens_details", None), "reasoning_tokens", None),
                                 cached_input_tokens=getattr(getattr(usage, "prompt_tokens_details", None), "cached_tokens", None))
                self.record_route_success(resolved, selected_model)
                return

        except Exception as exc:
            self.record_route_failure(resolved, selected_model, exc, stream=stream)
            # FIX 2026-08-07: `provider != ProviderType.AUTO` used to block
            # fallback entirely whenever a specific provider was resolved
            # (including the app's ordinary configured default, not just an
            # explicit --provider override) -- confirmed live 2026-08-06: a
            # turn died with "Error code: 429 ... you (tamfitron) have
            # reached your weekly usage limit" on the default provider, with
            # zero attempt to retry on a different configured provider, even
            # though is_retryable_provider_error already correctly classifies
            # 429 as retryable. Standing expectation (this is the same class
            # of issue as Remote-outage auto-fallback) is that infra-side
            # failures degrade-and-continue regardless of how the provider
            # was selected. auto_fallback_enabled() (TAMFIS_CODE_DISABLE_PROVIDER_FALLBACK)
            # remains the one on/off switch for this behavior.
            if (
                not allow_fallback
                or not self.auto_fallback_enabled()
                or not self.is_retryable_provider_error(exc)
            ):
                raise

            last_error: Exception = exc
            for fallback in self.fallback_candidates(
                resolved,
                task_profile,
                # FIX 2026-08-07: ollama_cloud_is_premium_primary's guard in
                # fallback_candidates() was written for the LOCAL DAEMON
                # UNAVAILABLE case, not quota exhaustion -- passing True
                # unconditionally here would silently mask real daemon-down
                # errors behind other providers instead of surfacing the
                # concrete Ollama availability error that guard exists to
                # produce. Only bypass it for the specific failure mode it
                # was never meant to cover.
                allow_premium_primary=self.is_quota_or_rate_limit_error(exc),
            ):
                try:
                    self.record_fallback(resolved)
                    # Do not carry a model identifier across providers.
                    async for chunk in self.chat_completion(
                        fallback,
                        messages,
                        model=None,
                        stream=stream,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        reasoning_effort=reasoning_effort,
                        task_profile=task_profile,
                        allow_fallback=False,
                        **kwargs,
                    ):
                        yield chunk
                    return
                except Exception as fallback_error:
                    last_error = fallback_error
                    continue

            raise last_error

    async def chat_completion_sync(
        self,
        provider: ProviderType,
        messages: List[Dict[str, Any]],
        model: Optional[str] = None,
        temperature: float = 0.10,
        max_tokens: int = 16384,
        task_profile: Optional["TaskProfile"] = None,
        **kwargs: Any,
    ) -> str:
        """Return a complete non-streaming response."""

        chunks: List[str] = []
        async for chunk in self.chat_completion(
            provider=provider,
            messages=messages,
            model=model,
            stream=False,
            temperature=temperature,
            max_tokens=max_tokens,
            task_profile=task_profile,
            **kwargs,
        ):
            chunks.append(chunk)
        return "".join(chunks)


async def chat_with_ollama_cloud(
    messages: List[Dict[str, Any]],
    **kwargs: Any,
) -> AsyncIterator[str]:
    manager = ProviderManager()
    async for chunk in manager.chat_completion(
        ProviderType.OLLAMA_CLOUD,
        messages,
        **kwargs,
    ):
        yield chunk


async def chat_with_grok(
    messages: List[Dict[str, Any]],
    **kwargs: Any,
) -> AsyncIterator[str]:
    manager = ProviderManager()
    async for chunk in manager.chat_completion(
        ProviderType.GROK,
        messages,
        **kwargs,
    ):
        yield chunk


async def chat_with_hf(
    messages: List[Dict[str, Any]],
    **kwargs: Any,
) -> AsyncIterator[str]:
    manager = ProviderManager()
    async for chunk in manager.chat_completion(
        ProviderType.HF,
        messages,
        **kwargs,
    ):
        yield chunk


async def chat_with_nvidia(
    messages: List[Dict[str, Any]],
    **kwargs: Any,
) -> AsyncIterator[str]:
    manager = ProviderManager()
    async for chunk in manager.chat_completion(
        ProviderType.NVIDIA,
        messages,
        **kwargs,
    ):
        yield chunk


async def chat_with_openrouter(
    messages: List[Dict[str, Any]],
    **kwargs: Any,
) -> AsyncIterator[str]:
    manager = ProviderManager()
    async for chunk in manager.chat_completion(
        ProviderType.OPENROUTER,
        messages,
        **kwargs,
    ):
        yield chunk


def get_provider_status() -> Dict[str, Any]:
    """Return provider availability and current automatic default."""

    manager = ProviderManager()

    try:
        default = manager._select_best_provider().value
    except ValueError:
        default = "none"

    return {
        "available": manager.list_available_providers(),
        "default": default,
        "config": {
            provider.value: {
                "enabled": manager.config.get(provider.value, True),
                "api_key_set": manager._has_valid_api_key(provider),
                "key_preview": (
                    (
                        manager._get_api_key(provider)[:8] + "..."
                        if manager._get_api_key(provider)
                        else "Not set"
                    )
                    if provider != ProviderType.TIER_IV
                    else "local"
                ),
            }
            for provider in manager.PRIORITY_ORDER
        },
    }
