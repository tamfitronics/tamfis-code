# Tamfis-Code 0.7.0 Runtime Rebuild

This release replaces the unbounded model-driven loop contract with a deterministic execution controller.

## Core behaviour

- Explicit runtime phases: discover, plan, execute, observe, validate, repair, complete, failed and cancelled.
- Hard budgets for total tool calls, identical actions, consecutive empty observations, plan revisions, repair rounds and wall-clock runtime.
- Stable fingerprints for tool actions and observations.
- Repeated identical calls are refused before dispatch.
- Three consecutive empty/no-evidence observations terminate the task with an explicit stall error.
- Plan steps complete only when a successful observation gains useful evidence.
- Useful evidence resets the empty-result streak.
- Runtime state is persisted into session progress metadata.
- Agent round ceiling reduced from 200 to 40.
- Package version advanced to 0.7.0.

## Package recovery

The supplied source archive omitted `routing.py` and `indexer.py` while other modules imported them. Both modules have been restored. Package-level exports are now lazy, preventing an optional component from breaking CLI startup.

## Validation performed

- All Python modules compile with `py_compile`.
- Deterministic runtime controller tests: 4 passed.
- Wheel built successfully: `tamfis_code-0.7.0-py3-none-any.whl`.
