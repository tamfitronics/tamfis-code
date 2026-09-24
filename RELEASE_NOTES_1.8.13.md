# Tamfis Code 1.8.13

- Enforce generated-plan scope validation before a plan is displayed or executed.
- Reject plan steps that mention unselected sibling projects, even when they omit an explicit path separator.
- Prevent cross-project hallucinations such as inspecting `betpredict` while the task is scoped to `tamgpt6`.
