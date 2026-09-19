"""Coordinator/worker approval mailbox (Pillar 3).

The properties that matter here are the ones that make parallel sub-agents
safe: exactly one claim per request under concurrency, exactly one resolution,
a worker that cannot approve itself, and a worker that fails closed when the
coordinator never answers.
"""
from __future__ import annotations

import asyncio
import threading
from pathlib import Path

import pytest

from tamfis_code.mailbox import (
    DECISION_APPROVE,
    DECISION_DENY,
    DECISION_TIMEOUT,
    STATUS_APPROVED,
    STATUS_CLAIMED,
    STATUS_DENIED,
    STATUS_EXPIRED,
    STATUS_PENDING,
    Mailbox,
    WorkerContext,
    current_worker,
    worker_context,
)


@pytest.fixture()
def mailbox(tmp_path: Path) -> Mailbox:
    return Mailbox(tmp_path / "mailbox.sqlite3")


def test_open_claim_resolve_roundtrip(mailbox: Mailbox):
    request_id = mailbox.open_request(
        worker="w1", tool="execute_command", arguments={"command": "git push --force origin main"},
        risk="dangerous", session_id=7, command="git push --force origin main",
    )
    record = mailbox.get(request_id)
    assert record["status"] == STATUS_PENDING
    assert record["arguments"] == {"command": "git push --force origin main"}

    claimed = mailbox.claim_next("coordinator")
    assert claimed["id"] == request_id
    assert claimed["status"] == STATUS_CLAIMED
    assert mailbox.claim_next("coordinator") is None  # no second claim of the same row

    assert mailbox.resolve(request_id, "approve_once", coordinator="coordinator") is True
    assert mailbox.get(request_id)["status"] == STATUS_APPROVED
    # First answer wins: a second answer (even the opposite) changes nothing.
    assert mailbox.resolve(request_id, "deny", coordinator="coordinator") is False


def test_a_worker_cannot_approve_its_own_request(mailbox: Mailbox):
    request_id = mailbox.open_request(worker="worker_a", tool="rm", arguments={}, risk="dangerous")
    assert mailbox.resolve(request_id, "approve_once", coordinator="worker_a") is False
    assert mailbox.get(request_id)["status"] == STATUS_PENDING
    assert mailbox.resolve(request_id, "deny", coordinator="coordinator") is True


def test_concurrent_coordinators_never_double_claim(mailbox: Mailbox):
    """The atomic-claim requirement: N coordinators polling the same mailbox
    must collectively claim each request exactly once."""
    expected = [
        mailbox.open_request(worker=f"w{i}", tool="execute_command", arguments={"command": f"rm -rf /tmp/{i}"})
        for i in range(12)
    ]
    claimed: list[str] = []
    lock = threading.Lock()

    def coordinator(number: int) -> None:
        local: list[str] = []
        while True:
            record = mailbox.claim_next(f"coordinator_{number}")
            if record is None:
                break
            local.append(record["id"])
        with lock:
            claimed.extend(local)

    threads = [threading.Thread(target=coordinator, args=(n,)) for n in range(6)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert sorted(claimed) == sorted(expected)
    assert len(claimed) == len(set(claimed)) == len(expected)


def test_expire_stale_clears_abandoned_requests(mailbox: Mailbox):
    request_id = mailbox.open_request(worker="w1", tool="rm", arguments={}, risk="dangerous")
    import time

    time.sleep(0.05)
    assert mailbox.expire_stale(max_age_seconds=0.01) == 1
    # A live request is never swept up by the same call.
    fresh = mailbox.open_request(worker="w2", tool="rm", arguments={}, risk="dangerous")
    assert mailbox.expire_stale(max_age_seconds=300.0) == 0
    assert mailbox.get(fresh)["status"] == STATUS_PENDING
    record = mailbox.get(request_id)
    assert record["status"] == STATUS_EXPIRED
    assert record["decision"] == DECISION_TIMEOUT


def test_pending_and_stats_report_the_backlog(mailbox: Mailbox):
    first = mailbox.open_request(worker="w1", tool="a", arguments={}, session_id=1)
    second = mailbox.open_request(worker="w2", tool="b", arguments={}, session_id=2)
    mailbox.resolve(second, "deny", coordinator="coordinator")

    assert [record["id"] for record in mailbox.pending(session_id=1)] == [first]
    assert len(mailbox.pending()) == 1
    stats = mailbox.stats()
    assert stats[STATUS_PENDING] == 1
    assert stats[STATUS_DENIED] == 1
    assert stats["total"] == 2


def test_worker_context_is_bound_to_the_task_not_globally(mailbox: Mailbox):
    assert current_worker() is None
    with worker_context(WorkerContext(worker_id="w1", mailbox=mailbox)) as bound:
        assert current_worker() is bound
        assert current_worker().worker_id == "w1"
    assert current_worker() is None


def test_worker_request_is_approved_while_it_waits(mailbox: Mailbox):
    context = WorkerContext(worker_id="delegated_1", mailbox=mailbox, timeout_seconds=5.0)

    def coordinator() -> None:
        import time

        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            record = mailbox.claim_next("coordinator")
            if record is not None:
                mailbox.resolve(record["id"], "approve_once", coordinator="coordinator")
                return
            time.sleep(0.01)

    thread = threading.Thread(target=coordinator)
    thread.start()
    decision = asyncio.run(context.request_approval(
        tool="execute_command", arguments={"command": "git push --force origin main"}, risk="dangerous",
    ))
    thread.join()
    assert decision == DECISION_APPROVE
    assert len(context.request_ids) == 1


def test_worker_fails_closed_when_the_coordinator_never_answers(mailbox: Mailbox):
    context = WorkerContext(worker_id="delegated_2", mailbox=mailbox, timeout_seconds=0.05)
    decision = asyncio.run(context.request_approval(
        tool="execute_command", arguments={"command": "rm -rf /tmp/x"}, risk="dangerous", timeout=0.05,
    ))
    assert decision == DECISION_TIMEOUT
    assert mailbox.get(context.request_ids[0])["status"] == STATUS_PENDING


def test_the_subagent_approval_gate_denies_unless_the_coordinator_approves(mailbox: Mailbox):
    """The runner-side gate (`_worker_mailbox_decision`) is the thing that
    actually routes a sub-agent's call; the mailbox only stores the answer."""
    from tamfis_code.runner_local import _worker_mailbox_decision

    context = WorkerContext(worker_id="delegated_3", mailbox=mailbox, timeout_seconds=0.05)

    def coordinator(decision: str) -> None:
        import time

        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            record = mailbox.claim_next("coordinator")
            if record is not None:
                mailbox.resolve(record["id"], decision, coordinator="coordinator")
                return
            time.sleep(0.01)

    thread = threading.Thread(target=coordinator, args=("deny",))
    thread.start()
    denied = asyncio.run(_worker_mailbox_decision(
        context, tool_name="execute_command", arguments={"command": "git push --force origin main"},
        display_command="git push --force origin main", risk="dangerous",
    ))
    thread.join()
    assert denied == "deny"

    thread = threading.Thread(target=coordinator, args=("approve_once",))
    thread.start()
    approved = asyncio.run(_worker_mailbox_decision(
        context, tool_name="write_file", arguments={"path": "src/app.py"},
        display_command="write_file", risk="medium",
    ))
    thread.join()
    assert approved == "approve_once"


def test_swarm_runs_workers_through_the_coordinator_mailbox(tmp_path: Path, monkeypatch):
    """End-to-end: a mutating swarm's worker asks the coordinator for a
    destructive call and only proceeds once the coordinator's policy approves
    it -- the whole point of the mailbox pattern."""
    from tamfis_code import mailbox as mailbox_module
    from tamfis_code import swarm as swarm_module

    shared = Mailbox(tmp_path / "swarm.sqlite3")
    monkeypatch.setattr(mailbox_module, "mailbox_for_swarm", lambda session_id=None: shared)

    seen: dict[str, str] = {}

    class _FakeAgentManager:
        async def execute_tasks(self, descriptions, **kwargs):
            factory = kwargs.get("worker_context_factory")
            results = []
            for index, description in enumerate(descriptions):
                context_manager = factory(f"delegated_{index}", 100 + index, description) if factory else None
                if context_manager is not None:
                    context_manager.__enter__()
                try:
                    worker = current_worker()
                    decision = await worker.request_approval(
                        tool="execute_command",
                        arguments={"command": "git push --force origin main"},
                        risk="dangerous", command="git push --force origin main",
                        timeout=10.0,
                    )
                    seen[f"delegated_{index}"] = decision
                    results.append({"task_id": f"delegated_{index}", "status": "completed"})
                finally:
                    if context_manager is not None:
                        context_manager.__exit__(None, None, None)
            return results

    monkeypatch.setattr("tamfis_code.agents.AgentManager", _FakeAgentManager)

    class _Console:
        is_terminal = False

        def print(self, *args, **kwargs):
            pass

    results = asyncio.run(swarm_module.run_swarm(
        ["force-push the release branch"], manager=None, provider=None, model=None,
        console=_Console(), workspace_root=str(tmp_path), session_id=1,
        approval_policy="full-auto", mutate=True, max_concurrency=1,
    ))

    assert [item["status"] for item in results] == ["completed"]
    assert seen == {"delegated_0": DECISION_APPROVE}
    # The coordinator answered it, and the record is auditable afterwards.
    history = shared.history(session_id=100)
    assert history and history[0]["status"] == STATUS_APPROVED
    assert history[0]["worker"] == "delegated_0"
    assert history[0]["coordinator"].startswith("coordinator_")


def test_read_only_swarm_needs_no_mailbox(tmp_path: Path, monkeypatch):
    """A read-only swarm can never issue an approvable call, so it must not
    pay for a mailbox (and must not change behaviour for existing callers)."""
    from tamfis_code import swarm as swarm_module

    captured: dict[str, object] = {}

    class _FakeAgentManager:
        async def execute_tasks(self, descriptions, **kwargs):
            captured["factory"] = kwargs.get("worker_context_factory")
            return [{"task_id": "delegated_0", "status": "completed"}]

    monkeypatch.setattr("tamfis_code.agents.AgentManager", _FakeAgentManager)

    class _Console:
        is_terminal = False

        def print(self, *args, **kwargs):
            pass

    results = asyncio.run(swarm_module.run_swarm(
        ["explain how auth works"], manager=None, provider=None, model=None,
        console=_Console(), workspace_root=str(tmp_path), session_id=1,
        approval_policy="ask", mutate=False, max_concurrency=1,
    ))

    assert captured["factory"] is None
    assert results == [{"task_id": "delegated_0", "status": "completed"}]



def test_coordinator_sweeps_abandoned_requests_before_answering(tmp_path: Path, monkeypatch):
    """A request left by a swarm that died must not sit in front of a live
    swarm's own requests, nor get answered long after its worker is gone."""
    from tamfis_code import swarm as swarm_module

    mailbox = Mailbox(tmp_path / "sweep.sqlite3")
    # The broken() shim: an abandoned request parked in the mailbox.
    mailbox.open_request(worker="dead-worker", tool="rm", arguments={}, risk="dangerous")

    swept: list[float] = []
    original = mailbox.expire_stale

    def spy(*, max_age_seconds: float = 900.0) -> int:
        swept.append(max_age_seconds)
        return original(max_age_seconds=0.0)  # sweep everything, for the test

    monkeypatch.setattr(mailbox, "expire_stale", spy)

    class _Console:
        is_terminal = False

        def print(self, *args, **kwargs):
            pass

    coordinator = swarm_module.SwarmCoordinator(
        mailbox, approval_policy="auto", console=_Console(), interactive=False,
    )

    async def main() -> None:
        stop = asyncio.Event()
        task = asyncio.ensure_future(coordinator.drain(stop))
        await asyncio.sleep(0.1)
        stop.set()
        await task

    asyncio.run(main())
    assert swept, "the coordinator must sweep stale requests before serving"
    assert mailbox.pending() == []
