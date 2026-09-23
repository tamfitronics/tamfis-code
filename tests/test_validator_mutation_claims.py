from tamfis_code.orchestrator.validator import validate_completion
from tamfis_code.routing import classify_task


def test_created_claim_without_mutation_evidence_is_rejected_for_execution_task():
    objective = "do this for Finitron: prepare the training pipeline"
    report = validate_completion(
        profile=classify_task(objective),
        tool_records=[
            {"tool_name": "read_file", "success": True, "arguments": {"path": "README.md"}},
        ],
        any_mutation=False,
        final_text="Created `/home/finitron/DATA_AUDIT_MULTIMODAL.md` and configured the pipeline.",
        objective=objective,
        workspace_root="/home/finitron",
    )
    assert report.passed is False
    assert report.severity == "error"
    assert any("mutation" in item.lower() for item in report.unresolved)


def test_generated_claim_without_mutation_evidence_is_rejected():
    objective = "train the Finitron model"
    report = validate_completion(
        profile=classify_task(objective),
        tool_records=[{"tool_name": "read_file", "success": True}],
        any_mutation=False,
        final_text="Generated the distillation dataset and updated the registry.",
        objective=objective,
        workspace_root="/home/finitron",
    )
    assert report.passed is False
    assert report.severity == "error"
