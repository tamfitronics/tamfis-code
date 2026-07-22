import os
from unittest.mock import patch

from tamfis_code.providers import ProviderManager, ProviderType
from tamfis_code.routing import TaskType, classify_task


def test_greeting_requires_no_tools_or_repo_context():
    profile = classify_task("hey")
    assert profile.task_type == TaskType.CONVERSATION
    assert not profile.requires_tools
    assert not profile.requires_repository_context


def test_closure_confirmation_is_conversation_not_debug():
    # Regression: "fix"/"bug" are substrings of "fixed"/"bug", so a plain
    # closure message ("yeah, that bug is fixed now, thanks") used to hit
    # the DEBUG branch below and get handed edit tools for a task with
    # nothing left to do -- live-reported as the agent redundantly
    # re-applying an already-shipped fix.
    profile = classify_task("yeah that bug is fixed now, thanks")
    assert profile.task_type == TaskType.CONVERSATION
    assert not profile.requires_tools


def test_closure_confirmation_variants_are_conversation():
    for text in (
        "the issue is fixed, no need to touch it again",
        "already resolved, thanks",
        "that fixed it",
        "confirmed working now",
    ):
        assert classify_task(text).task_type == TaskType.CONVERSATION, text


def test_genuine_debug_request_is_unaffected():
    profile = classify_task("please fix the bug in calc.py")
    assert profile.task_type == TaskType.DEBUG
    assert profile.requires_tools


def test_audit_requires_frontier_long_context_tools():
    profile = classify_task("audit the entire stack and implement fixes")
    assert profile.task_type == TaskType.AUDIT
    assert profile.requires_tools


def test_explicit_web_search_request_is_research_not_inspect():
    # Regression: TaskType.RESEARCH had no classify_task branch at all, so
    # RESEARCH_TOOLS (browser, web_search) was unreachable dead code -- the
    # model could never actually be offered either tool through the normal
    # agent loop. "search"/"find" alone are INSPECT's own broad keywords, so
    # this must be checked first and only trigger on web-directed phrasing.
    for text in (
        "search the web for the latest FastAPI release notes",
        "look up online what the current bitcoin price is",
        "please google the error message for me",
        "what's the latest news on this library",
    ):
        assert classify_task(text).task_type == TaskType.RESEARCH, text


def test_research_request_requires_tools_but_not_repository_context():
    profile = classify_task("search the web for current Node LTS version")
    assert profile.task_type == TaskType.RESEARCH
    assert profile.requires_tools
    assert not profile.requires_repository_context


def test_ordinary_in_repo_search_stays_inspect_not_research():
    profile = classify_task("search for the config file in this repository")
    assert profile.task_type == TaskType.INSPECT


def _manager_with(*providers):
    manager = ProviderManager.__new__(ProviderManager)
    manager.clients = {p: object() for p in providers}
    manager.config = {p.value: True for p in providers}
    manager._has_valid_api_key = lambda p: p in providers
    return manager


def test_auto_prefers_hf_qwen36_for_audit():
    manager = _manager_with(ProviderType.HF, ProviderType.NVIDIA, ProviderType.OPENROUTER)
    assert manager._select_best_provider(classify_task("audit the whole repository")) == ProviderType.HF


def test_auto_prefers_openrouter_for_edit_when_nvidia_unavailable():
    manager = _manager_with(ProviderType.OPENROUTER, ProviderType.HF)
    assert manager._select_best_provider(classify_task("fix and refactor the code")) == ProviderType.HF


def test_openrouter_default_is_not_openai_family():
    cfg = ProviderManager.PROVIDERS[ProviderType.OPENROUTER]
    assert not cfg.default_model.startswith("openai/")
    assert all(not model.startswith("openai/") for model in cfg.models)


def test_nvidia_default_model_is_tool_capable_and_not_unentitled_kimi():
    # The plain Llama route is fluent but has been observed fabricating local
    # tool results. Use the verified NVIDIA reasoning/tool route instead;
    # never use the account-unentitled Kimi route as the default.
    default_model = ProviderManager.PROVIDERS[ProviderType.NVIDIA].default_model
    assert default_model != "moonshotai/kimi-k2.6"
    assert default_model == "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning"


def test_kimi_k2_6_is_still_selectable_on_openrouter_and_hf():
    # NVIDIA's own kimi-k2.6 route is a per-account entitlement gap (see
    # the test above), not evidence the model itself is unusable -- both
    # confirmed live with real chat-completions calls: OpenRouter accepted
    # it and returned a genuine billing response (real model, no credit
    # balance), HF returned a real 200. NVIDIA's own id casing
    # ("moonshotai/kimi-k2.6") is what OpenRouter also uses; HF requires
    # the differently-cased "moonshotai/Kimi-K2.6" -- confirmed live that
    # the lowercase id 400s with model_not_found there.
    assert "moonshotai/kimi-k2.6" in ProviderManager.PROVIDERS[ProviderType.OPENROUTER].models
    assert "moonshotai/Kimi-K2.6" in ProviderManager.PROVIDERS[ProviderType.HF].models
    # Neither is the default -- an explicit --model selection, not a
    # behavior change to AUTO routing.
    assert ProviderManager.PROVIDERS[ProviderType.OPENROUTER].default_model != "moonshotai/kimi-k2.6"
    assert ProviderManager.PROVIDERS[ProviderType.HF].default_model != "moonshotai/Kimi-K2.6"


def test_openrouter_paid_coding_default_is_qwen_coder():
    config = ProviderManager.PROVIDERS[ProviderType.OPENROUTER]
    assert config.default_model == "qwen/qwen3-coder"
    assert config.default_model in config.models


def test_hf_prefers_official_qwen36_coding_route_and_keeps_deepseek_fallbacks():
    config = ProviderManager.PROVIDERS[ProviderType.HF]
    assert config.default_model == "Qwen/Qwen3.6-35B-A3B"
    assert config.default_model in config.models
    assert "Qwen/Qwen3.6-27B" in config.models
    assert "Qwen/Qwen3-Coder-480B-A35B-Instruct" in config.models
    assert "deepseek-ai/DeepSeek-V4-Pro" in config.models
    assert config.coding_quality >= 5
    assert config.context_window >= 262144


def test_nvidia_exposes_deepseek_v4_routes_without_replacing_verified_default():
    config = ProviderManager.PROVIDERS[ProviderType.NVIDIA]
    assert config.default_model == "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning"
    assert "deepseek-ai/deepseek-v4-pro" in config.models
    assert "deepseek-ai/deepseek-v4-flash" in config.models


def test_remote_fallback_candidates_stay_in_policy_order():
    manager = _manager_with(ProviderType.NVIDIA, ProviderType.OPENROUTER, ProviderType.HF)
    # Use a tool-requiring task that does not require long context, so HF is
    # eligible; long-context tasks correctly skip its 32k route.
    assert manager.fallback_candidates(ProviderType.NVIDIA, classify_task("inspect app.py")) == [
        ProviderType.HF, ProviderType.OPENROUTER,
    ]


def test_http_402_is_retryable_provider_failure():
    class CreditError(Exception):
        status_code = 402

    assert ProviderManager.is_retryable_provider_error(CreditError("Insufficient credits"))


def test_resource_exhausted_worker_limit_is_retryable():
    exc = RuntimeError("ResourceExhausted: Worker local total request limit reached (32/32)")
    assert ProviderManager.is_retryable_provider_error(exc)


def test_generic_open_stream_failure_is_retryable():
    # Live OpenAI-compatible SDKs sometimes discard the underlying response
    # detail and expose only this wrapper after streaming already began.
    exc = RuntimeError("An error occurred during streaming")
    assert ProviderManager.is_retryable_provider_error(exc)


def test_nvidia_degraded_function_400_is_retryable():
    exc = RuntimeError(
        "Error code: 400 - {'status': 400, 'title': 'Bad Request', "
        "'detail': \"Function id 'abc': DEGRADED function cannot be invoked\"}"
    )
    assert ProviderManager.is_retryable_provider_error(exc)


def test_unrelated_http_400_is_not_retryable():
    class BadRequest(Exception):
        status_code = 400

    assert not ProviderManager.is_retryable_provider_error(BadRequest("invalid request"))


def test_nim_account_entitlement_404_is_retryable():
    # Live-reported: moonshotai/kimi-k2.6 returns a real 404 with this exact
    # shape when the account has no deployment access to that specific
    # model. Not a bad request -- AUTO mode should fall back to the next
    # candidate rather than hard-failing the whole turn.
    exc = RuntimeError(
        "Error code: 404 - {'status': 404, 'title': 'Not Found', 'detail': "
        "\"Function '23d4f03a-b8a6-4adb-a183-7daa083a09cc': Not found for "
        "account 'T0ktMu-NCoEGEm9N8eE19EvsHqn9CiQAk-DN7TF22WM'\"}"
    )
    assert ProviderManager.is_retryable_provider_error(exc)


def test_plain_404_without_the_entitlement_shape_is_not_retryable():
    # A generic 404 (bad endpoint path, genuinely missing resource) must
    # not be swallowed into an infinite fallback loop -- only the specific
    # NIM account-entitlement message shape above is treated as retryable.
    class NotFoundError(Exception):
        status_code = 404

    assert not ProviderManager.is_retryable_provider_error(NotFoundError("not found"))


def test_check_status_is_an_inspection_requiring_tools():
    profile = classify_task("check your previous status and continue")
    assert profile.task_type == TaskType.INSPECT
    assert profile.requires_tools


def test_standalone_provider_manager_excludes_tier_iv_from_routing_order():
    manager = ProviderManager.__new__(ProviderManager)
    manager.runtime_mode = "standalone"
    assert ProviderType.TIER_IV not in manager.routing_order
    assert manager.fallback_chain_names(ProviderType.NVIDIA) == [
            ProviderType.TAMFIS.value,
            ProviderType.HF.value,
            ProviderType.OPENROUTER.value,
    ]


def test_remote_provider_manager_may_include_tier_iv():
    manager = ProviderManager.__new__(ProviderManager)
    manager.runtime_mode = "remote"
    assert manager.routing_order[0] == ProviderType.TIER_IV


def test_standalone_explicit_tier_iv_route_is_rejected():
    manager = ProviderManager.__new__(ProviderManager)
    manager.runtime_mode = "standalone"
    manager.clients = {ProviderType.TIER_IV: object()}
    manager.config = {ProviderType.TIER_IV.value: True}
    try:
        manager.resolve_route(ProviderType.TIER_IV)
    except ValueError as exc:
        assert "not available in standalone runtime mode" in str(exc)
    else:
        raise AssertionError("standalone Tier IV route was not rejected")
