import asyncio
import json
from io import StringIO
from types import SimpleNamespace

from rich.console import Console

from tamfis_code import state as state_module
from tamfis_code.providers import ProviderManager, ProviderType
from tamfis_code.routing import TaskType, classify_task
from tamfis_code.orchestrator import ToolEnvelope
from tamfis_code.orchestrator.engine import AgentOrchestrator
from tamfis_code.runner_local import (
    _audit_evidence_targets,
    _read_only_audit_has_sufficient_evidence,
    run_local_agent_turn,
)


class RouteError(RuntimeError):
    status_code = 503


def _manager_with(*providers):
    manager = ProviderManager.__new__(ProviderManager)
    manager.clients = {provider: object() for provider in providers}
    manager.config = {provider.value: True for provider in providers}
    manager._has_valid_api_key = lambda provider: provider in providers
    return manager


def setup_function():
    ProviderManager.reset_runtime_routing_state()


def teardown_function():
    ProviderManager.reset_runtime_routing_state()


def test_read_only_audit_coverage_uses_successful_concrete_tool_evidence():
    records = []
    for index in range(7):
        envelope = ToolEnvelope(
            tool_call_id=f"call-{index}",
            tool_name="read_file",
            arguments={"path": f"module_{index}.py"},
            purpose="inspect implementation",
        )
        envelope.finish(result={"path": f"module_{index}.py"}, success=True)
        records.append(envelope)

    assert len(_audit_evidence_targets(records)) == 7
    assert _read_only_audit_has_sufficient_evidence(TaskType.AUDIT, True, records)
    assert not _read_only_audit_has_sufficient_evidence(TaskType.EDIT, True, records)
    assert not _read_only_audit_has_sufficient_evidence(TaskType.AUDIT, False, records)


def test_nim_primary_failure_uses_alternate_nim_model_before_other_provider():
    manager = _manager_with(ProviderType.NVIDIA, ProviderType.HF)
    profile = classify_task("fix the validation bug and run tests")
    config = manager.PROVIDERS[ProviderType.NVIDIA]

    primary = manager.select_model(config, profile)
    assert primary == config.default_model
    manager.record_route_failure(ProviderType.NVIDIA, primary, RouteError("worker unavailable"))

    candidates = manager.fallback_candidates(ProviderType.NVIDIA, profile)
    assert candidates[:2] == [ProviderType.NVIDIA, ProviderType.HF]
    assert manager.select_model(config, profile) != primary


def test_nim_circuit_is_model_scoped_and_not_permanent(monkeypatch):
    manager = _manager_with(ProviderType.NVIDIA)
    config = manager.PROVIDERS[ProviderType.NVIDIA]
    primary = config.default_model
    manager.record_route_failure(ProviderType.NVIDIA, primary, RouteError("temporary 503"))
    assert not manager.route_is_healthy(ProviderType.NVIDIA, primary)
    assert manager.route_is_healthy(ProviderType.NVIDIA, config.models[1])

    import tamfis_code.providers as providers_module

    monkeypatch.setattr(providers_module.time, "monotonic", lambda: 10**12)
    assert manager.route_is_healthy(ProviderType.NVIDIA, primary)


def test_explicit_non_nim_selection_is_not_rewritten_by_economic_policy():
    manager = _manager_with(ProviderType.NVIDIA, ProviderType.HF)
    resolved, config = manager.resolve_route(
        ProviderType.HF,
        classify_task("inspect the repository"),
    )
    assert resolved == ProviderType.HF
    assert config is manager.PROVIDERS[ProviderType.HF]


def test_all_nim_unavailable_falls_to_next_healthy_capable_provider():
    manager = _manager_with(ProviderType.NVIDIA, ProviderType.HF, ProviderType.OPENROUTER)
    profile = classify_task("fix the validation bug and run tests")
    config = manager.PROVIDERS[ProviderType.NVIDIA]
    for model in dict.fromkeys([config.default_model, *config.models]):
        manager.record_route_failure(ProviderType.NVIDIA, model, RouteError("unavailable"))

    assert manager.fallback_candidates(ProviderType.NVIDIA, profile)[:2] == [
        ProviderType.HF,
        ProviderType.OPENROUTER,
    ]


def test_route_telemetry_distinguishes_eligible_selected_success_and_fallback():
    manager = _manager_with(ProviderType.NVIDIA, ProviderType.HF)
    profile = classify_task("inspect the repository")
    selected = manager._select_best_provider(profile)
    model = manager.select_model(manager.PROVIDERS[selected], profile)
    manager.record_route_attempt(selected, model)
    manager.record_route_success(selected, model)
    manager.record_fallback(selected)

    telemetry = manager.routing_telemetry()
    assert telemetry.nim_eligible_requests == 1
    assert telemetry.nim_selected_requests == 1
    assert telemetry.nim_successful_requests == 1
    assert telemetry.nim_fallback_requests == 1
    assert telemetry.provider_requests == {"nvidia": 1}
    assert telemetry.provider_successes == {"nvidia": 1}


def test_fallback_keeps_requested_and_effective_route_distinct(monkeypatch):
    orchestrator = AgentOrchestrator.__new__(AgentOrchestrator)
    orchestrator.session_id = 7
    orchestrator.run = SimpleNamespace(route={})
    monkeypatch.setattr(orchestrator, "transition", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        "tamfis_code.orchestrator.engine.local_state.save_session_state",
        lambda *_args, **_kwargs: None,
    )

    orchestrator.record_route(
        provider="nvidia",
        model="moonshotai/kimi-k3",
        requested_provider="auto",
        requested_model="auto",
        reason="capability-aware automatic routing",
    )
    orchestrator.record_route(
        provider="hf",
        model="Qwen/Qwen3.6-35B-A3B",
        reason="automatic provider fallback",
        fallback_reason="NIM stream failed with HTTP 503",
    )

    assert orchestrator.run.route["requested_provider"] == "auto"
    assert orchestrator.run.route["requested_model"] == "auto"
    assert orchestrator.run.route["effective_provider"] == "hf"
    assert orchestrator.run.route["effective_model"] == "Qwen/Qwen3.6-35B-A3B"
    assert orchestrator.run.route["fallback_reason"] == "NIM stream failed with HTTP 503"


def test_real_agent_loop_survives_primary_nim_stream_failure_without_duplicate_mutation(
    tmp_path, monkeypatch,
):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    target = workspace / "calc.py"
    target.write_text("def total(n):\n    return n + 2\n", encoding="utf-8")
    state_module.CONFIG_DIR = tmp_path / ".config"
    state_module.STATE_PATH = state_module.CONFIG_DIR / "state.json"

    def delta(*, content=None, tool_calls=None):
        return SimpleNamespace(content=content, tool_calls=tool_calls)

    def call(index, call_id, name, arguments):
        return SimpleNamespace(
            index=index,
            id=call_id,
            function=SimpleNamespace(name=name, arguments=arguments),
        )

    def chunk(value, finish_reason=None):
        return SimpleNamespace(
            choices=[SimpleNamespace(delta=value, finish_reason=finish_reason)],
        )

    class Stream:
        def __init__(self, chunks):
            self.chunks = chunks

        def __aiter__(self):
            async def generate():
                for item in self.chunks:
                    yield item
            return generate()

    class Completions:
        def __init__(self):
            self.calls = []
            self.alternate_rounds = [
                [chunk(delta(tool_calls=[call(
                    0, "edit-1", "edit_file", json.dumps({
                        "path": str(target),
                        "old_string": "return n + 2",
                        "new_string": "return n + 1",
                    }),
                )]))],
                [chunk(delta(tool_calls=[call(
                    0, "verify-1", "execute_command", json.dumps({"command": "true"}),
                )]))],
                [chunk(delta(content="Fixed and verified."), finish_reason="stop")],
            ]

        async def create(self, **kwargs):
            self.calls.append(kwargs)
            if kwargs["model"] == ProviderManager.PROVIDERS[ProviderType.NVIDIA].default_model:
                raise RouteError("NIM primary stream unavailable")
            return Stream(self.alternate_rounds.pop(0))

    completions = Completions()
    client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    manager = _manager_with(ProviderType.NVIDIA, ProviderType.HF)
    manager.clients[ProviderType.NVIDIA] = client
    manager.clients[ProviderType.HF] = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(
            create=lambda **_kwargs: (_ for _ in ()).throw(AssertionError("HF should not be reached")),
        )),
    )

    class Renderer:
        debug = False

        def __init__(self):
            self.events = []
            self.background_requested = asyncio.Event()

        def handle_event(self, event):
            self.events.append(event)

    renderer = Renderer()
    monkeypatch.setattr("tamfis_code.runner_local.should_plan", lambda *_args: False)
    async def no_sleep(_seconds):
        return None
    monkeypatch.setattr("tamfis_code.runner_local.asyncio.sleep", no_sleep)
    monkeypatch.setattr(
        "tamfis_code.runner_local.detect_validation_commands",
        lambda *_args: [("no-op", "true")],
    )

    outcome = asyncio.run(run_local_agent_turn(
        manager,
        ProviderType.AUTO,
        None,
        [{"role": "user", "content": "fix the bug in calc.py and run tests"}],
        Console(file=StringIO(), no_color=True, width=200),
        renderer,
        workspace_root=str(workspace),
        session_id=1,
        approval_policy="auto",
        interactive=False,
    ))

    assert outcome.status == "completed"
    assert target.read_text(encoding="utf-8").count("return n + 1") == 1
    attempted_models = [item["model"] for item in completions.calls]
    assert attempted_models[0] == ProviderManager.PROVIDERS[ProviderType.NVIDIA].default_model
    assert any(model != attempted_models[0] for model in attempted_models[1:])
    mutation_events = [event for event in renderer.events if event["event_type"] == "file_mutation"]
    assert len(mutation_events) == 1
    assert not any(
        event.get("payload", {}).get("provider") == "hf"
        for event in renderer.events
    )
