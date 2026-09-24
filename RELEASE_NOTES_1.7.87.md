# Tamfis-Code 1.7.87

## Fixed

- Recognize explicit in-place shell edits such as `sed -i` and `perl -pi`
  as mutation intent. Follow-up coding instructions now upgrade stale
  inspect/audit turns into execute mode while retaining the normal scope and
  approval gates.
- Recognize common `comment out` and `uncomment` coding instructions as
  mutation intent.

## Verification

- Added routing regression coverage for shell-edit follow-ups.
