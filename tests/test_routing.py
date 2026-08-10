from tamfis_code.providers import ProviderManager, ProviderType
from tamfis_code.routing import TaskType, classify_task, is_explicit_read_only_request
from tamfis_code.orchestrator.planner import should_plan
from tamfis_code.tool_policy import allowed_tools


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


def test_explicit_no_edit_status_review_is_inspection_even_when_filename_contains_fix():
    profile = classify_task(
        "read tamgpt6/ATTACHMENT_PIPELINE_FIX_PROGRESS.md and check against "
        "the TamfisGPT status and provide recommendations. no file edit"
    )
    assert profile.task_type == TaskType.INSPECT
    assert profile.requires_tools
    assert not profile.requires_validation
    assert not should_plan(profile)
    assert "execute_command" not in allowed_tools(profile, read_only=False)
    assert "edit_file" not in allowed_tools(profile, read_only=False)


def test_exact_background_status_prompt_has_shared_read_only_intent():
    text = (
        "read tamgpt6/ATTACHMENT_PIPELINE_FIX_PROGRESS.md and check against hte "
        "TamfsiGPT status and provide recommendationd. no file edit. workin in the backgorund"
    )
    assert is_explicit_read_only_request(text)
    assert classify_task(text).task_type == TaskType.INSPECT


def test_read_only_fix_recommendations_do_not_become_a_debug_task():
    for text in (
        "review the fix progress and provide recommendations only",
        "inspect the reported bug without editing anything",
        "check the repair status; don't modify files",
        "read-only review of the failing pipeline",
    ):
        assert classify_task(text).task_type == TaskType.INSPECT, text


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


def test_auto_prefers_nvidia_over_hf_for_audit():
    # Prior to the Ollama Cloud priority routing change, HF's Qwen 3.6 route
    # was preferred for audits. PRIORITY_ORDER now ranks NVIDIA (priority=1)
    # above HF (priority=2) unconditionally, so selection among these three
    # is purely priority-based regardless of task type -- there is no
    # audit-specific override in _select_best_provider.
    manager = _manager_with(ProviderType.HF, ProviderType.NVIDIA, ProviderType.OPENROUTER)
    assert manager._select_best_provider(classify_task("audit the whole repository")) == ProviderType.NVIDIA


def test_premium_ollama_is_authoritative_for_auto(monkeypatch):
    monkeypatch.setenv("TAMFIS_PROVIDER_OLLAMA_CLOUD_ENABLED", "true")
    monkeypatch.setenv("TAMFIS_CODE_OLLAMA_PREMIUM", "true")
    monkeypatch.setenv("TAMFIS_CODE_OLLAMA_AUTO_PRIMARY", "true")
    manager = _manager_with(ProviderType.OLLAMA_CLOUD, ProviderType.NVIDIA)
    resolved, _ = manager.resolve_route(ProviderType.AUTO, classify_task("fix the API"))
    assert resolved == ProviderType.OLLAMA_CLOUD


def test_ollama_primary_uses_kimi_k27_without_extra_usage(monkeypatch):
    monkeypatch.setenv("TAMFIS_CODE_OLLAMA_PREMIUM", "true")
    monkeypatch.delenv("TAMFIS_CODE_OLLAMA_EXTRA_USAGE", raising=False)
    monkeypatch.delenv("TAMFIS_CODE_OLLAMA_CODING_MODEL", raising=False)
    manager = _manager_with(ProviderType.OLLAMA_CLOUD)

    for prompt in ("hello", "fix the API", "audit the entire repository"):
        profile = classify_task(prompt)
        assert manager.select_model(
            manager.PROVIDERS[ProviderType.OLLAMA_CLOUD], profile
        ) == "kimi-k2.7-code:cloud"


def test_premium_ollama_remains_enabled_without_auto_primary(monkeypatch):
    # Subscription entitlement must remain enabled without forcing AUTO to
    # Ollama Cloud. Historically one flag controlled both concerns and
    # disabling forced-primary also downgraded the included-plan model from
    # kimi-k2.7-code:cloud to the much weaker gemma4:cloud any time Ollama
    # Cloud was used (automatic fallback after NIM, or explicit
    # --provider ollama_cloud). kimi-k2.7-code:cloud is not an extra-usage
    # cost (unlike kimi-k3:cloud, still gated below), so there is no reason
    # to withhold it just because AUTO isn't forced onto Ollama Cloud.
    monkeypatch.setenv("TAMFIS_CODE_OLLAMA_PREMIUM", "true")
    monkeypatch.setenv("TAMFIS_CODE_OLLAMA_AUTO_PRIMARY", "false")
    monkeypatch.delenv("TAMFIS_CODE_OLLAMA_EXTRA_USAGE", raising=False)
    monkeypatch.delenv("TAMFIS_CODE_OLLAMA_CODING_MODEL", raising=False)
    manager = _manager_with(ProviderType.OLLAMA_CLOUD)

    for prompt in ("hello", "fix the API", "audit the entire repository"):
        profile = classify_task(prompt)
        assert manager.select_model(
            manager.PROVIDERS[ProviderType.OLLAMA_CLOUD], profile
        ) == "kimi-k2.7-code:cloud"

    # But AUTO still must not force Ollama Cloud as primary with the flag
    # off -- when NVIDIA is also available, NIM (priority 0) wins, not
    # Ollama Cloud (priority 3), confirming the AUTO-forcing behavior this
    # dedicated AUTO-primary flag stays off.
    manager_with_nim = _manager_with(ProviderType.OLLAMA_CLOUD, ProviderType.NVIDIA)
    resolved, _ = manager_with_nim.resolve_route(ProviderType.AUTO, classify_task("fix the API"))
    assert resolved == ProviderType.NVIDIA


def test_ollama_exposes_glm_52_as_a_priority():
    config = ProviderManager.PROVIDERS[ProviderType.OLLAMA_CLOUD]
    assert "glm-5.2:cloud" in config.models
    assert config.models.index("glm-5.2:cloud") == (
        config.models.index("kimi-k2.7-code:cloud") + 1
    )


def test_ollama_exposes_deepseek_v4_flash_0731():
    config = ProviderManager.PROVIDERS[ProviderType.OLLAMA_CLOUD]
    assert "deepseek-v4-flash:0731-cloud" in config.models


def test_ollama_extra_usage_requires_operator_opt_in_and_heavy_task(monkeypatch):
    monkeypatch.setenv("TAMFIS_CODE_OLLAMA_PREMIUM", "true")
    monkeypatch.setenv("TAMFIS_CODE_OLLAMA_EXTRA_USAGE", "true")
    monkeypatch.delenv("TAMFIS_CODE_OLLAMA_HEAVY_MODEL", raising=False)
    manager = _manager_with(ProviderType.OLLAMA_CLOUD)
    config = manager.PROVIDERS[ProviderType.OLLAMA_CLOUD]

    assert manager.select_model(config, classify_task("hello")) == "kimi-k2.7-code:cloud"
    assert manager.select_model(
        config, classify_task("audit the entire repository")
    ) == "kimi-k3:cloud"


def test_premium_ollama_does_not_fallback_to_nvidia(monkeypatch):
    monkeypatch.setenv("TAMFIS_PROVIDER_OLLAMA_CLOUD_ENABLED", "true")
    monkeypatch.setenv("TAMFIS_CODE_OLLAMA_PREMIUM", "true")
    monkeypatch.setenv("TAMFIS_CODE_OLLAMA_AUTO_PRIMARY", "true")
    manager = _manager_with(ProviderType.OLLAMA_CLOUD, ProviderType.NVIDIA)
    assert manager.fallback_candidates(ProviderType.OLLAMA_CLOUD) == []


def test_unavailable_premium_ollama_fails_explicitly_in_auto(monkeypatch):
    monkeypatch.setenv("TAMFIS_PROVIDER_OLLAMA_CLOUD_ENABLED", "true")
    monkeypatch.setenv("TAMFIS_CODE_OLLAMA_PREMIUM", "true")
    monkeypatch.setenv("TAMFIS_CODE_OLLAMA_AUTO_PRIMARY", "true")
    manager = _manager_with(ProviderType.NVIDIA)
    try:
        manager.resolve_route(ProviderType.AUTO, classify_task("fix the API"))
    except ValueError as exc:
        assert "Ollama Cloud is enabled as the AUTO primary route" in str(exc)
    else:
        raise AssertionError("AUTO silently selected a non-Ollama provider")


def test_auto_prefers_openrouter_for_edit_when_nvidia_unavailable():
    manager = _manager_with(ProviderType.OPENROUTER, ProviderType.HF)
    assert manager._select_best_provider(classify_task("fix and refactor the code")) == ProviderType.HF


def test_openrouter_default_is_not_openai_family():
    # The default route must still avoid the OpenAI family (cost/quota
    # reasons this project has documented elsewhere), but the model catalog
    # has since grown to deliberately include openai/gpt-4.1-mini and
    # openai/gpt-4.1-nano as additional *selectable* OpenRouter options --
    # only the default is constrained, not the whole list anymore.
    cfg = ProviderManager.PROVIDERS[ProviderType.OPENROUTER]
    assert not cfg.default_model.startswith("openai/")


def test_nvidia_default_model_is_tool_capable_and_not_unentitled_kimi():
    # The plain Llama route is fluent but has been observed fabricating local
    # tool results. Use a verified NVIDIA nemotron tool-calling route instead;
    # never use the account-unentitled Kimi route as the default.
    # 2026-07-26: re-prioritized off nemotron-3-nano-omni-30b-a3b-reasoning
    # after it was caught live wrapping a fake tool call in CLI-flag/XML-ish
    # text (see runner_local.py's fake-tool-call detection fix) -- a failure
    # mode a single simple tool-calling smoke test doesn't surface, since
    # that model also returns clean real tool_calls on simple prompts.
    default_model = ProviderManager.PROVIDERS[ProviderType.NVIDIA].default_model
    assert default_model != "moonshotai/kimi-k2.6"
    assert default_model == "nvidia/nemotron-3-ultra-550b-a55b"


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


def test_nvidia_exposes_deepseek_v4_pro_without_replacing_verified_default():
    config = ProviderManager.PROVIDERS[ProviderType.NVIDIA]
    assert config.default_model == "nvidia/nemotron-3-ultra-550b-a55b"
    assert "deepseek-ai/deepseek-v4-pro" in config.models
    # deepseek-ai/deepseek-v4-flash removed 2026-08-08: reached NVIDIA NIM
    # end-of-life 2026-08-07 (HTTP 410 confirmed live in tamgpt6's intent
    # classifier) -- must not silently reappear as a selectable NIM route.
    assert "deepseek-ai/deepseek-v4-flash" not in config.models


def test_nvidia_exposes_newly_added_coding_and_tool_calling_models():
    # 2026-07-26: added per a live NVIDIA NIM catalog re-check for models
    # described as strong at coding/tool-calling. All were live-verified
    # against this account (real chat-completions calls with tools
    # attached, real tool_calls returned) before being added here.
    config = ProviderManager.PROVIDERS[ProviderType.NVIDIA]
    assert "nvidia/nemotron-3-ultra-550b-a55b" in config.models
    assert "nvidia/nemotron-3-super-120b-a12b" in config.models
    assert "nvidia/nemotron-3-nano-30b-a3b" in config.models
    assert "nvidia/llama-3.3-nemotron-super-49b-v1.5" in config.models
    assert "nvidia/llama-3.3-nemotron-super-49b-v1" in config.models
    assert "minimaxai/minimax-m3" in config.models
    # The nano-omni-reasoning route stays selectable (not removed), just no
    # longer the default -- see test_nvidia_default_model_is_tool_capable_
    # and_not_unentitled_kimi.
    assert "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning" in config.models


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
    # Ollama Cloud priority routing put OLLAMA_CLOUD at priority=0 (ahead of
    # everything, including automatic fallback candidates) and moved TAMFIS
    # to the back of PRIORITY_ORDER -- it no longer appears in NVIDIA's
    # automatic fallback chain unless a free/paid-fallback route applies.
    # xAI Grok was added, then demoted to priority=6 / last in PRIORITY_ORDER
    # (2026-08-06: real billed xAI spend within the first week of enabling
    # it), so it now appears last in NVIDIA's automatic fallback chain
    # instead of straight after Ollama Cloud.
    assert manager.fallback_chain_names(ProviderType.NVIDIA) == [
            ProviderType.OLLAMA_CLOUD.value,
            ProviderType.HF.value,
            ProviderType.OPENROUTER.value,
            ProviderType.GROK.value,
    ]


def test_remote_provider_manager_may_include_tier_iv():
    manager = ProviderManager.__new__(ProviderManager)
    manager.runtime_mode = "remote"
    # NIM priority routing (2026-08-08) ranks NVIDIA (priority=0) ahead of
    # TIER_IV (priority=5) in PRIORITY_ORDER even in remote mode -- remote
    # mode's only remaining distinction from standalone is that it does not
    # exclude TIER_IV from the order entirely.
    assert manager.routing_order[0] == ProviderType.NVIDIA
    assert ProviderType.TIER_IV in manager.routing_order


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
