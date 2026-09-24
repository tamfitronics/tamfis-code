# Tamfis-Code 1.7.83

## Command path compatibility

- `execute_command` now accepts provider-generated `path` and
  `working_directory` aliases as `cwd`, preventing handler crashes from
  unexpected `path` keywords.
- Added regression coverage for the alias.
