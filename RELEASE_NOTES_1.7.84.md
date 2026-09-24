# Tamfis-Code 1.7.84

## Bounded approved directory discovery

- Approved external recursive `find` commands are automatically bounded to
  one filesystem and depth 8 unless the caller already specifies
  `-maxdepth`.
- This prevents `/etc` and service-managed directories from consuming the
  full command timeout while preserving approved cross-directory inspection.
