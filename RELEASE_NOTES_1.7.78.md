# Tamfis-Code 1.7.78

## Reliable approved file inspection

- Plain `cat` reads of approved files now use the bounded direct reader,
  avoiding intermittent shell/sandbox hangs on external read-only paths such
  as `/etc/cron.d/*`.
- Scope checks remain mandatory; shell features, options, pipes, redirects,
  and substitutions continue through the normal command policy and timeout.
- Added regression coverage for the direct-read path.
