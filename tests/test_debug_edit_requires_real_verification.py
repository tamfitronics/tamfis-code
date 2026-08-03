"""FIX: for EDIT/DEBUG tasks, validate_completion used to accept a bare
read_file/search_code/list_directory/get_git_info tool result as
"validation evidence" -- a model could report a bug fixed after only
re-reading the file it just edited, with zero execute_command calls ever
made, and this check passed. Narrowed to real execution evidence for
exactly the task types where "I fixed it" is a testable claim.
"""
from tamfis_code.orchestrator.validator import validate_completion
from tamfis_code.routing import classify_task


def test_debug_task_with_only_read_file_evidence_is_not_validated():
    objective = "fix the off-by-one bug in parser.py"
    report = validate_completion(
        profile=classify_task(objective),
        tool_records=[
            {"tool_name": "write_file", "success": True, "arguments": {"path": "parser.py"}},
            {"tool_name": "read_file", "success": True, "arguments": {"path": "parser.py"}},
        ],
        any_mutation=True, final_text="Fixed the off-by-one bug.", objective=objective,
    )
    assert report.passed is False
    assert any("validation" in item.lower() for item in report.unresolved)


def test_debug_task_with_a_real_execute_command_is_validated():
    objective = "fix the off-by-one bug in parser.py"
    report = validate_completion(
        profile=classify_task(objective),
        tool_records=[
            {"tool_name": "write_file", "success": True, "arguments": {"path": "parser.py"}},
            {"tool_name": "execute_command", "success": True, "exit_code": 0, "arguments": {"command": "pytest tests/test_parser.py"}},
        ],
        any_mutation=True, final_text="Fixed the off-by-one bug; tests pass.", objective=objective,
    )
    assert report.passed is True


def test_edit_task_with_only_search_code_evidence_is_not_validated():
    objective = "edit the config loader to support YAML"
    report = validate_completion(
        profile=classify_task(objective),
        tool_records=[
            {"tool_name": "edit_file", "success": True, "arguments": {"path": "config.py"}},
            {"tool_name": "search_code", "success": True, "arguments": {"query": "load_config"}},
        ],
        any_mutation=True, final_text="Updated the config loader.", objective=objective,
    )
    assert report.passed is False


def test_question_classification_cannot_bypass_false_mutation_and_live_claim_gates():
    objective = "Can you align these settings?"
    report = validate_completion(
        profile=classify_task(objective),
        tool_records=[
            {"tool_name": "read_file", "success": True, "arguments": {"path": ".env"}},
            {"tool_name": "execute_command", "success": True, "exit_code": 0,
             "arguments": {"command": "sudo systemctl restart tamfisseo"}},
        ],
        any_mutation=False,
        final_text=(
            "I have updated `.env` and rewritten `api/workers/discovery.ts`. "
            "The service restart is complete and the frontend will now pull data successfully."
        ),
        objective=objective,
        workspace_root="/srv/app",
    )

    assert report.passed is False
    assert report.severity == "error"
    assert any("no successful file mutation" in item.lower() for item in report.unresolved)
    assert any("behavioral test" in item.lower() for item in report.unresolved)


def test_deployment_claims_pass_with_exact_mutations_restart_and_behavioral_check():
    objective = "Can you align these settings?"
    report = validate_completion(
        profile=classify_task(objective),
        tool_records=[
            {"tool_name": "edit_file", "success": True, "files_changed": ["/srv/app/.env"],
             "arguments": {"path": ".env"}},
            {"tool_name": "edit_file", "success": True,
             "files_changed": ["/srv/app/api/workers/discovery.ts"],
             "arguments": {"path": "api/workers/discovery.ts"}},
            {"tool_name": "execute_command", "success": True, "exit_code": 0,
             "arguments": {"command": "sudo systemctl restart tamfisseo"}},
            {"tool_name": "execute_command", "success": True, "exit_code": 0,
             "arguments": {"command": "curl -fsS https://example.test/health"}},
        ],
        any_mutation=True,
        final_text=(
            "I have updated `.env` and rewritten `api/workers/discovery.ts`. "
            "The service restart is complete and the API is now working."
        ),
        objective=objective,
        workspace_root="/srv/app",
    )

    assert report.passed is True
