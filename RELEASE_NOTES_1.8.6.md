# Tamfis-Code 1.8.6

## Slash-command completeness

- Implement `/notifications`, which reports completed, failed, and stopped
  background jobs plus queued follow-up notifications for the active session.
- Add regression coverage for the notification command and keep the full
  resume/history alias behavior covered.

## Verification

- Slash-command and standalone session tests: 20 passed, 17 subtests passed.
