"""Provider integrations for Tamfis-Code.

Supports the TamfisGPT subscription API, Hugging Face, NVIDIA NIM, and
OpenRouter through a canonical OpenAI-compatible client interface.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, AsyncIterator, Dict, List, Optional, TYPE_CHECKING

from openai import AsyncOpenAI

if TYPE_CHECKING:
    from .routing import TaskProfile


class ProviderType(str, Enum):
    """Supported provider identifiers."""

    TIER_IV = "tier_iv"
    TAMFIS = "tamfis"
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
    it: a real streaming call to meta/llama-3.1-70b-instruct (NVIDIA's own
    current default_model, below) with reasoning_effort="high" produced
    zero chunks in 40+ seconds, while the identical call without it
    returned in well under a second. REASONING_EFFORT_CAPABLE_PROVIDERS
    alone silently broke the moment NVIDIA's default_model moved off
    nemotron (the v0.4.39 kimi-k2.6-404 fix) -- nobody had re-verified
    reasoning_effort against the new default, since the original "confirmed
    live" note only ever tested nemotron. OpenRouter/Tier IV are left as
    provider-level (unverified per-model here; no live evidence either way).
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
    return str(getattr(task_profile, "complexity", "")) == "high"


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


class ProviderManager:
    """Initialise, inspect, rank, and access configured AI providers."""

    PRIORITY_ORDER: tuple[ProviderType, ...] = (
        ProviderType.TIER_IV,
        ProviderType.TAMFIS,
        ProviderType.HF,
        ProviderType.NVIDIA,
        ProviderType.OPENROUTER,
    )

    PROVIDERS: Dict[ProviderType, ProviderConfig] = {
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
            models=["tamfis-gpt-auto"],
            priority=0,
            weight=6,
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
        # provider rather than a pre-flight routing advisor. Tried
        # automatically whenever reachable -- see _check_tier_iv_available
        # -- with TAMFIS_TIER_IV_ENABLED=false as the explicit opt-out.
        ProviderType.TIER_IV: ProviderConfig(
            name="TamfisGPT Tier IV",
            base_url=f"{os.environ.get('TAMFIS_TIER_IV_URL', 'http://127.0.0.1:9555').rstrip('/')}/v1",
            api_key_env="TAMFIS_ACCESS_TOKEN",
            default_model="",  # empty -- let Tier IV's own orchestrator pick, same as tamgpt6's own callers do when no model is pinned.
            models=[],
            priority=0,
            weight=5,
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
            # The plain Llama instruct route can answer fluently but has
            # repeatedly narrated/fabricated local tool results in this CLI.
            # This NVIDIA reasoning model was verified to return genuine
            # tool_calls and to accept reasoning parameters without the
            # indefinite stream stall seen on the plain instruct models.
            # Keep the older models selectable for accounts where this route
            # is not enabled, but do not make them the automatic coding path.
            default_model="nvidia/nemotron-3-nano-omni-30b-a3b-reasoning",
            models=[
                "nvidia/nemotron-3-super-120b-a12b",
                "nvidia/nemotron-3-ultra-550b-a55b",
                # Confirmed live: real tool_calls response (not narrated
                # text), reasoning_effort and reasoning_budget both work
                # without hanging (unlike the plain instruct models below).
                "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning",
                "nvidia/nemotron-3-nano-30b-a3b",
                # NVIDIA currently exposes these DeepSeek V4 hosted routes
                # as mature coding/agent models. They remain selectable
                # fallbacks; the verified Nemotron route stays the default
                # until an account-specific smoke test proves otherwise.
                "deepseek-ai/deepseek-v4-pro",
                "deepseek-ai/deepseek-v4-flash",
                "meta/llama-3.1-405b-instruct",
                "meta/llama-3.1-70b-instruct",
                "moonshotai/kimi-k2.6",
                "mistralai/mistral-large-2-123b",
                "google/gemma-2-27b-it",
                "microsoft/phi-3-medium-128k-instruct",
            ],
            # HF is the preferred external coding route: its automatic model
            # is the official Qwen 3.6 coding model. NVIDIA remains the first
            # mature fallback when HF is unavailable or its account is out.
            priority=1,
            weight=4,
            reasoning_supported=True,
            vision_supported=False,
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
            priority=0,
            weight=3,
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
            ],
            # OpenRouter is last in AUTO because paid coding routes can fail
            # with HTTP 402 when the account has no credits. It remains
            # explicitly selectable and is still tried after HF/NVIDIA.
            priority=3,
            weight=2,
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

    def _fallback_provider_allowed(self, provider: ProviderType) -> bool:
        if provider == ProviderType.NVIDIA:
            return True
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

        key = os.environ.get(config.api_key_env, "").strip()
        return key or None

    def _has_valid_api_key(self, provider_type: ProviderType) -> bool:
        """Perform a conservative configuration-level key check."""

        if not self.provider_allowed(provider_type):
            return False

        if provider_type == ProviderType.TIER_IV:
            return self._check_tier_iv_available()

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
        if config.free_model and not _task_needs_paid_tier(task_profile):
            return config.free_model
        return config.default_model

    def resolve_route(
        self,
        provider: ProviderType,
        task_profile: Optional["TaskProfile"] = None,
        *,
        quality_mode: str = "quality",
    ) -> tuple[ProviderType, ProviderConfig]:
        """Resolve AUTO to a concrete provider and configuration."""

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
        """Select the strongest eligible configured provider."""

        available = [
            provider
            for provider in self.routing_order
            if allowed_providers is None or provider in allowed_providers
            if provider in self.clients and self._has_valid_api_key(provider)
        ]

        if not available:
            raise ValueError("No configured AI provider is available")

        requires_tools = bool(
            getattr(task_profile, "requires_tools", False)
        )
        requires_long_context = bool(
            getattr(task_profile, "requires_long_context", False)
        )
        preferred_tier = str(
            getattr(task_profile, "preferred_quality_tier", quality_mode)
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

        if quality_mode == "economy" or preferred_tier == "economy":
            return min(
                eligible,
                key=lambda provider: (
                    self.PROVIDERS[provider].priority,
                    -self.PROVIDERS[provider].coding_quality,
                ),
            )

        if quality_mode == "balanced" or preferred_tier == "balanced":
            return min(
                eligible,
                key=lambda provider: (
                    self.PROVIDERS[provider].priority,
                    -self.PROVIDERS[provider].coding_quality,
                ),
            )

        # Quality mode favours capability while preserving the stated policy
        # as the tie-breaker.
        return max(
            eligible,
            key=lambda provider: (
                self.PROVIDERS[provider].coding_quality,
                self.PROVIDERS[provider].context_window,
                -self.PROVIDERS[provider].priority,
            ),
        )

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
                408,
                409,
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
            # NVIDIA NIM's real shape for "this account has no deployment
            # access to this specific model" (confirmed live: a genuine,
            # permanent 404 for moonshotai/kimi-k2.6 on an ordinary API
            # key, despite the model being listed in the general
            # /v1/models catalog). Retrying the exact same model would
            # never help, but AUTO mode falling back to the next candidate
            # provider/model is exactly the right response -- this is an
            # account-entitlement failure, not a bad request.
            "not found for account",
            "degraded function cannot be invoked",
            "function cannot be invoked",
        )
        return any(marker in message for marker in retryable_markers)

    def fallback_candidates(
        self,
        current: ProviderType,
        task_profile: Optional["TaskProfile"] = None,
    ) -> List[ProviderType]:
        """Return usable alternatives in canonical policy order."""

        requires_tools = bool(
            getattr(task_profile, "requires_tools", False)
        )
        requires_long_context = bool(
            getattr(task_profile, "requires_long_context", False)
        )

        candidates: List[ProviderType] = []
        for provider in self.routing_order:
            if provider == current:
                continue
            if provider not in self.clients:
                continue
            if not self._has_valid_api_key(provider):
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
            available = enabled and valid_key and client_initialised

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
                    "healthy": available,
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

        selected_model = model or (
            config.free_model
            if config.free_model and not self.paid_fallback_enabled()
            else self.select_model(config, task_profile)
        )

        request_kwargs: Dict[str, Any] = {
            "model": selected_model,
            "messages": messages,
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
            extra_body = dict(request_kwargs.get("extra_body") or {})
            extra_body.setdefault("mode", _tier_iv_mode_for_task(task_profile))
            request_kwargs["extra_body"] = extra_body

        try:
            response = await client.chat.completions.create(**request_kwargs)

            if stream:
                async for chunk in response:
                    if not chunk.choices:
                        continue
                    content = chunk.choices[0].delta.content
                    if content:
                        yield content
                return

            if response.choices:
                content = response.choices[0].message.content
                if content:
                    yield content
            return

        except Exception as exc:
            if (
                not allow_fallback
                or not self.auto_fallback_enabled()
                or provider != ProviderType.AUTO
                or not self.is_retryable_provider_error(exc)
            ):
                raise

            last_error: Exception = exc
            for fallback in self.fallback_candidates(
                resolved,
                task_profile,
            ):
                try:
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
