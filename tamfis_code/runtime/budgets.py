"""Hard execution budgets that prevent unbounded agent loops."""
from __future__ import annotations
from dataclasses import dataclass


@dataclass(frozen=True)
class RuntimeBudgets:
    max_tool_calls: int = 40
    max_identical_actions: int = 2
    max_consecutive_empty_observations: int = 3
    max_plan_revisions: int = 4
    max_repair_rounds: int = 3
    max_runtime_seconds: int = 900

    def __post_init__(self) -> None:
        for name, value in self.__dict__.items():
            if value < 1:
                raise ValueError(f"{name} must be at least 1")
