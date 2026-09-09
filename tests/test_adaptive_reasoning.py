from tamfis_code.providers import ProviderType
from tamfis_code.routing import classify_task
import tamfis_code.runner_local as runner_local


def test_reasoning_effort_scales_with_task_complexity(monkeypatch):
    monkeypatch.setattr(runner_local, "DEFAULT_REASONING_EFFORT", "auto")
    model = "nvidia/nemotron-3-super-120b-a12b"

    simple = classify_task("Explain what this repository does")
    complex_task = classify_task(
        "Fix the request lifecycle across api.py and worker.py, add integration tests, "
        "run the build, and verify database persistence end-to-end."
    )

    assert runner_local._reasoning_effort(ProviderType.NVIDIA, model, simple) == "medium"
    assert runner_local._reasoning_effort(ProviderType.NVIDIA, model, complex_task) == "high"

    agent_upgrade = classify_task(
        "Improve this coding agent's reasoning abilities and developer experience"
    )
    assert runner_local._reasoning_effort(ProviderType.NVIDIA, model, agent_upgrade) == "high"


def test_explicit_reasoning_effort_override_wins(monkeypatch):
    monkeypatch.setattr(runner_local, "DEFAULT_REASONING_EFFORT", "low")
    model = "nvidia/nemotron-3-super-120b-a12b"
    profile = classify_task("audit the entire repository architecture")

    assert runner_local._reasoning_effort(ProviderType.NVIDIA, model, profile) == "low"


def test_unsupported_model_never_receives_reasoning_effort(monkeypatch):
    monkeypatch.setattr(runner_local, "DEFAULT_REASONING_EFFORT", "high")
    profile = classify_task("audit the entire repository architecture")

    assert runner_local._reasoning_effort(
        ProviderType.HF, "Qwen/Qwen2.5-Coder-32B-Instruct", profile,
    ) is None
