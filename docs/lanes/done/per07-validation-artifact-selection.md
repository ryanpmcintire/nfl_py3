# PER-07 coordinator validation input selection

## Goal
Make coordinator-change replays select and identify their adjudicated input.

## State
Implemented `--change-validation` for the screen and production LEAD-29
commands. Omission retains the existing latest-snapshot selection. Each replay
records the selected file path, SHA-256, and embedded validation provenance.
This completes the input-selection repair; the broader research remains open.

## Tried
- **Measured:** the real input-building command produced 8,698 rows, 12 counted
  events, and 12 signal rows. Explicit and default selection agreed.
- **Measured:** the selected file hash matched
  `61fc88853b85165fe47f6805e05beb30ef0eacda3564c050965dd23133cf571f`.
  Source: `data/raw/coordinators/20260924T194157799750Z/change_validation.json`.
- **Reported (worker verification):** missing files, malformed payloads, and
  unknown statuses were rejected; CLI help and scoped Ruff checks passed.
- Evidence: `.tmp/per07-root-input-verification.log` and
  `.tmp/per07-implementation-verification.md`.

## Next
Use an explicit validation artifact for a predeclared future research replay.
Keep the existing unresolved coordinator result and serving state unchanged.

## Open
Other replay inputs retain their existing selection rules. Embedded validation
provenance is preserved, not independently attested. This change adds no new
appointment evidence and does not close the coordinator signal.
