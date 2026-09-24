# Tamfis-Code 1.7.80

## Release metadata reliability

- Release publication now automatically selects the matching versioned notes
  file when no notes argument is supplied.
- Missing `latest.json`, release notes, installer, or wheel files now fail the
  release instead of being silently ignored.
- The published manifest and live release directory are checked against the
  exact package version before the served manifest is verified.
