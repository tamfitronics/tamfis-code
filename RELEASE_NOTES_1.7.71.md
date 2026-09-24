# Tamfis-Code 1.7.71

## Checkpoint recovery hardening

- Resumed inspection tasks now execute a bounded deterministic
  `list_directory` or `read_file` step when a provider repeatedly emits
  fabricated tool results or refuses to call tools.
- Recovery uses only the already-approved pending plan target and remains
  inside the workspace scope; it never invents paths or performs recursive
  scans.
- Fixed the installed-package resume path so `return_recap` is available in
  the released CLI artifact.

## Verification

- 223 focused tests passed, plus 17 subtests.
- Python compilation passed.
- Installed resume command reached the interactive session without an import
  failure.
