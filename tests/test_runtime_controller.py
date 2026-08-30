from tamfis_code.runtime import ExecutionController, RuntimeBudgets, RuntimePhase


def _result(stdout="", *, success=True, items=None):
    payload = {"stdout": stdout}
    if items is not None:
        payload["items"] = items
    return {"success": success, "result": payload}


def test_useful_observation_resets_empty_streak():
    controller = ExecutionController(RuntimeBudgets(max_runtime_seconds=60))
    assert controller.guard_action("search_code", {"root": "/tmp", "query": "x"}).allowed
    first = controller.observe("search_code", {"root": "/tmp", "query": "x"}, _result("(empty)"))
    assert not first.useful
    assert controller.snapshot.consecutive_empty_observations == 1
    assert controller.guard_action("read_file", {"path": "/tmp/a.py"}).allowed
    second = controller.observe("read_file", {"path": "/tmp/a.py"}, _result("print('ok')"))
    assert second.useful
    assert controller.snapshot.consecutive_empty_observations == 0
    assert controller.snapshot.evidence_items >= 1


def test_three_empty_observations_signal_recoverable_stall():
    controller = ExecutionController(RuntimeBudgets(max_consecutive_empty_observations=3, max_runtime_seconds=60))
    decision = None
    for index in range(3):
        args = {"root": "/tmp", "query": f"missing-{index}"}
        assert controller.guard_action("search_code", args).allowed
        decision = controller.observe("search_code", args, _result("(empty)"))
    assert decision is not None and decision.stalled
    assert not decision.terminal
    assert controller.snapshot.phase == RuntimePhase.EXECUTE
    assert not controller.snapshot.failure_reason
    assert "stalled" in decision.reason.casefold()


def test_tool_failures_are_actionable_evidence_not_empty_stalls():
    controller = ExecutionController(RuntimeBudgets(max_consecutive_empty_observations=3, max_runtime_seconds=60))
    for index in range(3):
        args = {"command": f"bad-{index}"}
        assert controller.guard_action("execute_command", args).allowed
        decision = controller.observe(
            "execute_command", args,
            {"success": False, "error": "Command path is outside the resolved task scope"},
        )
        if index == 0:
            assert decision.useful
        else:
            assert not decision.useful
            assert "Duplicate evidence" in decision.reason
        assert not decision.terminal
    assert controller.snapshot.consecutive_empty_observations == 2


def test_identical_action_is_blocked_after_two_attempts():
    controller = ExecutionController(RuntimeBudgets(max_identical_actions=2, max_runtime_seconds=60))
    args = {"root": "/tmp", "query": "same"}
    for _ in range(2):
        assert controller.guard_action("search_code", args).allowed
        controller.observe("search_code", args, _result("(empty)"))
    third = controller.guard_action("search_code", args)
    assert not third.allowed
    assert "repeated action" in third.reason.casefold()


def test_different_actions_returning_duplicate_evidence_trigger_stall():
    controller = ExecutionController(
        RuntimeBudgets(max_consecutive_empty_observations=3, max_runtime_seconds=60)
    )
    for index in range(4):
        args = {"command": f"grep variant-{index} app.py"}
        assert controller.guard_action("execute_command", args).allowed
        decision = controller.observe(
            "execute_command", args, _result("the same already-known line"),
        )
    assert decision.stalled
    assert not decision.terminal
    assert "no new evidence" in decision.reason
    assert controller.snapshot.novel_observations == 1


def test_novel_evidence_allows_a_reasonable_follow_up_check():
    controller = ExecutionController(RuntimeBudgets(max_identical_actions=2, max_runtime_seconds=60))
    args = {"path": "/tmp/app.py"}
    for content in ("before", "after"):
        assert controller.guard_action("read_file", args).allowed
        observation = controller.observe("read_file", args, _result(content))
        assert observation.useful
    # The same check is now valid again because the file's observed state
    # changed; it must not inherit the pre-edit repetition count forever.
    assert controller.guard_action("read_file", args).allowed


def test_tool_budget_is_hard():
    controller = ExecutionController(RuntimeBudgets(max_tool_calls=2, max_runtime_seconds=60))
    for index in range(2):
        args = {"path": f"/tmp/{index}"}
        assert controller.guard_action("read_file", args).allowed
        controller.observe("read_file", args, _result("content"))
    blocked = controller.guard_action("read_file", {"path": "/tmp/third"})
    assert blocked.terminal
    assert blocked.tool_call_budget_exhausted
    assert "TAMFIS_CODE_MAX_TOOL_CALLS" in blocked.reason
    assert controller.snapshot.phase == RuntimePhase.FAILED


def test_tool_budget_extension_grants_a_fresh_window():
    # Regression: a "round" (runner_local.py) can contain several tool
    # calls, so this raw-count ceiling was reachable well before the round
    # budget's own separate extension logic ever got a chance to run --
    # unlike the wall-clock budget, it previously had no extension at all
    # and was an unconditional hard failure. extend_tool_call_budget()
    # mirrors extend_runtime()'s behaviour: bounded by
    # max_tool_call_extensions, and it un-fails a FAILED phase that was set
    # purely by the tool-call ceiling so execution can resume.
    controller = ExecutionController(
        RuntimeBudgets(max_tool_calls=2, max_tool_call_extensions=1, max_runtime_seconds=60)
    )
    for index in range(2):
        args = {"path": f"/tmp/{index}"}
        assert controller.guard_action("read_file", args).allowed
        controller.observe("read_file", args, _result("content"))
    blocked = controller.guard_action("read_file", {"path": "/tmp/third"})
    assert blocked.terminal and blocked.tool_call_budget_exhausted
    assert controller.snapshot.phase == RuntimePhase.FAILED

    assert controller.extend_tool_call_budget() is True
    assert controller.snapshot.phase == RuntimePhase.EXECUTE
    assert controller.snapshot.failure_reason == ""
    # The extension grants one more full window (base budget), not just +1.
    resumed = controller.guard_action("read_file", {"path": "/tmp/third"})
    assert resumed.allowed


def test_tool_budget_extensions_are_bounded():
    controller = ExecutionController(
        RuntimeBudgets(max_tool_calls=1, max_tool_call_extensions=1, max_runtime_seconds=60)
    )
    assert controller.guard_action("read_file", {"path": "/a"}).allowed
    controller.observe("read_file", {"path": "/a"}, _result("content"))
    assert controller.guard_action("read_file", {"path": "/b"}).terminal
    assert controller.extend_tool_call_budget() is True
    assert controller.guard_action("read_file", {"path": "/b"}).allowed
    controller.observe("read_file", {"path": "/b"}, _result("content"))
    assert controller.guard_action("read_file", {"path": "/c"}).terminal
    # The single configured extension is already used -- no more are granted.
    assert controller.extend_tool_call_budget() is False
