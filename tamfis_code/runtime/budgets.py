"""Hard execution budgets that prevent unbounded agent loops."""
from __future__ import annotations
from dataclasses import dataclass


@dataclass(frozen=True)
class RuntimeBudgets:
    # Match runner_local's three bounded 40-round windows. This is still a
    # hard safety ceiling; identical-action and empty-observation guards stop
    # pathological loops much earlier.
    max_tool_calls: int = 120
    # A "round" (runner_local.py's MAX_AGENT_ROUNDS, with its own separate
    # auto-extension) is one model turn and can contain several tool calls,
    # so this ceiling -- being a raw tool-call count -- was reachable well
    # before the round budget's own extensions ever kicked in on any task
    # that made more than one tool call per round on average. Unlike the
    # round and wall-clock budgets, this one had no extension at all: it was
    # an unconditional hard task failure telling the user to go edit
    # config.toml and restart, which is a genuine "less capable than Claude
    # Code/Codex on a long task" gap -- neither of those hard-fails a session
    # for making "too many" tool calls. See max_tool_call_extensions below.
    max_tool_call_extensions: int = 2
    max_identical_actions: int = 2
    max_consecutive_empty_observations: int = 3
    max_plan_revisions: int = 4
    # Without an extension, replace_plan() unconditionally killed the whole
    # task the moment the model wanted to revise its plan for the 5th time --
    # the one budget (of rounds/wall-clock/tool-calls/repair/plan-revisions)
    # that had none at all, even though "learn more, update the plan" is
    # exactly the adaptive behaviour a long, evolving task needs and Claude
    # Code/Codex never hard-cap. See extend_plan_revision_budget on
    # ExecutionController.
    max_plan_revision_extensions: int = 2
    max_repair_rounds: int = 3
    max_runtime_seconds: int = 900
    # How many times a turn may reset its wall-clock budget and keep going
    # instead of the task failing outright when it runs out of time. This
    # is a continuation, not a bigger single budget: every other guard
    # (tool-call count, repeated actions, stall detection) still applies
    # across the whole task and is untouched by an extension.
    max_runtime_extensions: int = 3
    # record_repair() is shared across ~10 unrelated recovery classes --
    # provider fallback, empty-continuation recovery, fabricated-tool-result
    # correction, capitulation redirection, AND genuine "the model tried to
    # fix a real failure (e.g. a broken build)" repairs -- see the many
    # orchestrator.mark_repair(...) call sites in runner_local.py. A single
    # small shared counter means 3 unrelated infra hiccups earlier in a turn
    # can exhaust the budget before the model gets a real shot at the
    # actual failing step. Extensions grant a fresh max_repair_rounds
    # window (bounded by this count) instead of failing the whole task the
    # first time the shared counter runs out.
    max_repair_extensions: int = 2

    def __post_init__(self) -> None:
        for name, value in self.__dict__.items():
            if value < 1:
                raise ValueError(f"{name} must be at least 1")
