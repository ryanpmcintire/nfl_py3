# LEAD-64 card designations

## Goal

Surface actionable official final designations for teams still to play without changing the model, picks or availability ingestion.

## State

The presentation subtask is complete. `injury_feed_coverage_note` counts only Out, Doubtful and Questionable final statuses for pending teams. It resolves each identified player to the latest feed status, enriches Out and Doubtful rows with names from roster weeks at or before the requested week, and retains the existing missing or stale feed warning. This does not close the full LEAD-64 Friday/Sunday acquisition workflow.

## Tried

- Measured: a temporary official-schema parquet invocation of the production function selected three unique pending players, removed an earlier Out after a later cleared status, kept a later Out after an earlier Questionable status, excluded completed teams and ignored a future roster week.
- Measured by root: Ruff format and check, MyPy, and the full test suite passed; the suite reported 4,529 passed and 9 skipped.
- Measured by root: `publish-board` exited 0 after the presentation change.

## Next

Complete the remaining LEAD-64 acquisition workflow so Friday final designations are captured before Sunday and Sunday status updates refresh the same canonical feed.

## Open

The Friday/Sunday capture workflow remains open. The display shows at most eight Out and eight Doubtful entries, then reports the remaining count. Team status counts accompany rather than replace availability provenance.
