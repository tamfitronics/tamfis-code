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


def test_three_empty_observations_fail_terminally():
    controller = ExecutionController(RuntimeBudgets(max_consecutive_empty_observations=3, max_runtime_seconds=60))
    decision = None
    for index in range(3):
        args = {"root": "/tmp", "query": f"missing-{index}"}
        assert controller.guard_action("search_code", args).allowed
        decision = controller.observe("search_code", args, _result("(empty)"))
    assert decision is not None and decision.terminal
    assert controller.snapshot.phase == RuntimePhase.FAILED
    assert "stalled" in controller.snapshot.failure_reason.casefold()


def test_identical_action_is_blocked_after_two_attempts():
    controller = ExecutionController(RuntimeBudgets(max_identical_actions=2, max_runtime_seconds=60))
    args = {"root": "/tmp", "query": "same"}
    for _ in range(2):
        assert controller.guard_action("search_code", args).allowed
        controller.observe("search_code", args, _result("(empty)"))
    third = controller.guard_action("search_code", args)
    assert not third.allowed
    assert "repeated action" in third.reason.casefold()


def test_tool_budget_is_hard():
    controller = ExecutionController(RuntimeBudgets(max_tool_calls=2, max_runtime_seconds=60))
    for index in range(2):
        args = {"path": f"/tmp/{index}"}
        assert controller.guard_action("read_file", args).allowed
        controller.observe("read_file", args, _result("content"))
    blocked = controller.guard_action("read_file", {"path": "/tmp/third"})
    assert blocked.terminal
    assert controller.snapshot.phase == RuntimePhase.FAILED
