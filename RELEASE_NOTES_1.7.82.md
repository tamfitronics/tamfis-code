# Tamfis-Code 1.7.82

## Approved external directories

- Explicitly approved external paths can now be used by read tools and
  `execute_command` during an otherwise read-only turn.
- The approval gate remains mandatory for the external scope; unapproved
  paths and external mutations remain blocked.
- Added clearer recovery guidance for approved external inspection.
