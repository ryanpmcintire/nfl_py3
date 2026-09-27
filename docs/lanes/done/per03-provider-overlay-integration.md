# PER-03 optional injury provider integration

## Goal
Make a verified Sportradar capture usable by the specialist absence overlay.

## State
Implemented the runtime loader and an explicit optional provider path in the
specialist overlay. The default source remains unchanged. This completes the
integration slice; historical revision coverage and a live capture remain open.

## Tried
- **Measured:** all six existing capture tests pass after updating the loader
  import and explicit season/week arguments (`.tmp/per03-existing-tests-root.log`).
- **Read:** the runtime loader verifies target season/week/type, capture cutoff,
  coverage, file sizes and hashes, row identity, and availability timestamps
  (`src/nfl_ats/sportradar_injury_snapshot.py`).
- **Measured:** the isolated overlay check accepts a valid provider
  snapshot, preserves the default source, and rejects tampering, incomplete
  coverage, a wrong-week row, and a future-only capture
  (`.tmp/per03-provider-root-verification.log`).

## Next
Obtain an authorized current capture when credentials are available, then verify
its manifest and exercise the explicit provider path against that capture.

## Open
No provider credential or current live-response verification is available.
Capture time is the availability bound; provider status dates are not revision
history. No historical evidence or served-side rule changed in this slice.
