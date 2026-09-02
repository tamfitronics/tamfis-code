from tamfis_code import state
from tamfis_code.orchestrator.protocols import classify_failure


def test_task_checkpoint_persists_structured_resume_ledger(tmp_path, monkeypatch):
    monkeypatch.setattr(state, "STATE_PATH", tmp_path / "state.json")
    session_id = 991001
    state.save_session_state(session_id, workspace_root=str(tmp_path))
    state.update_task_state(
        session_id,
        objective="repair and validate artifacts",
        phase="testing",
        completed_steps=["inspect"],
        failures=[{"category": "test_failure", "error": "assertion"}],
        retries=[{"retry_number": 1, "disposition": "pending_diagnosis"}],
        artifacts_created=[{"filename": "analysis.xlsx", "status": "validated"}],
        completion_evidence=["targeted test passed"],
    )
    checkpoint = state.task_checkpoint(session_id, reason="tests_passed", next_action="validate artifacts")
    restored = state.get_session_state(session_id)
    assert checkpoint["reason"] == "tests_passed"
    assert restored.task_state["objective"] == "repair and validate artifacts"
    assert restored.context_checkpoints[-1]["next_action"] == "validate artifacts"
    assert restored.task_state["artifacts_created"][0]["status"] == "validated"


def test_failure_classifier_is_provider_and_domain_neutral():
    assert classify_failure("pytest assertion failed") == "test_failure"
    assert classify_failure("429 rate limit from provider") == "provider_failure"
    assert classify_failure("artifact workbook validation failed") == "artifact_validation_failure"
    assert classify_failure("unexpected process crash") == "runtime_failure"
