# Tamfis-Code 1.8.3

## Correct interrupted-task completion state

- Do not report a saved plan as delivered when the final turn was cancelled,
  interrupted, failed, or lost during provider recovery.
- Resume those turns through acceptance verification instead of returning
  “The saved plan is already complete; no further work was started.”
- Add regression coverage for a completed-looking plan with an interrupted
  checkpoint.

## Verification

- Recovery and resume suites: 34 passed.
