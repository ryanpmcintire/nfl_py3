# MOD-17 redundant challenger enrollment

## Goal
Stop recording a lattice-centre challenger after that centre became the served implementation.

## State
Complete implementation cleanup; MOD-17 research remains unresolved.
**Read:** `src/nfl_ats/tiebreaker.py:401` uses `challenger_centre` for the served lattice;
`docs/lanes/done/mod17-unified-served-numbers.md` dates adoption to September 24.
`src/nfl_ats/lattice_centre_challenger.py` now settles historical rows first, then
skips enrollment from `2026-09-24T00:00:00Z` onward.

## Tried
**Measured:** root ran the production recorder against a temporary ledger copy
with September 27 as the recording time. It skipped enrollment, preserved all
three identities and their provenance, and retained two settled rows.
Source and copy SHA-256 remained
`f63eae2965c9fbdd37aecc89b9b8411ca0549eb7bbb603feeede057d562a47d1`.
Evidence: `.tmp/mod17-root-recorder-probe.json`; locked Ruff check and format passed.

## Next
Keep settling the pre-adoption ledger. Continue the separate joint-residual research.

## Open
The cutoff has date precision and conservatively excludes every new row on the
adoption date. The registry remains active because its other statuses imply a
research verdict or promotion; this cleanup asserts neither. A truthful enrollment-ended
status is separate work. Zero crossing does not close a signal, and one fitted
calibrated probability must select the served side.
