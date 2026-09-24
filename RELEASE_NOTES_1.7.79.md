# Tamfis-Code 1.7.79

## Provider-safe numeric tool arguments

- Coerce schema-declared integer and number fields when providers send them as
  JSON strings, fixing valid searches rejected for string `offset` or
  `max_results` values.
- Invalid numeric strings remain unchanged and are still rejected by strict
  schema validation.
- Added gateway regression coverage.
