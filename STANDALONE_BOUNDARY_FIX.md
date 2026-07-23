Tamfis-Code standalone boundary repair

Changes:
- ProviderManager defaults to runtime_mode="standalone".
- Standalone routing_order excludes ProviderType.TIER_IV.
- Tier IV cannot initialise, resolve, list, or participate in fallback in standalone mode.
- runtime_mode="remote" retains Tier IV support.
- local_chat no longer accepts tier_iv/tier4 aliases.
- runner fallback metadata derives from the manager's runtime-safe order.
- failed evidence validation returns TaskOutcome(status="failed") and preserves the checkpoint.
- regression tests cover standalone exclusion, remote allowance, explicit rejection, and legacy test doubles.
