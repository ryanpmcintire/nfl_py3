# LEAD-53 skip diagnostics

## Goal

Retain the next recording-enabled Sunday re-nomination outcome so a skipped pairing can be diagnosed
without reconstructing history.

## State

**Read:** Every recording-enabled Sunday call writes
`artifacts/prospective/best_pick_refresh_latest_attempt.json` atomically with the invocation, result, and skip
reason when present.

**Read:** The wrapper delegates to the unchanged recorder first. Settlement, ledger, and current-row
validation failures retain their existing skip behavior; there is no fallback or invariant weakening.

**Measured:** No latest-attempt artifact exists yet because no recording-enabled Sunday call has run since
this diagnostic was added. No historical diagnostic or decision row was fabricated.

## Tried
- **Measured:** all 11 existing focused tests pass; Ruff and format checks pass.
- **Measured:** the real read-only `nfl-ats refresh-picks` command exited 0; no
  decision was recorded. The card/ledger check still reports five side differences.
  Evidence: `.tmp/lead53-refresh-readonly.log`; diagnostic writing remains covered
  by the existing recorder tests, since this command used no record flag.

Traced all recorder returns and the scheduler output contract. The old scheduler retained only the last
stdout line, which explains why the Week 1 detailed result disappeared. A current production paper-ledger
read exited 0 with 48 rows, so the historical Week 2 loader failure does not reproduce now.

## Next

After the next live Sunday recording attempt, inspect the latest-attempt artifact and scheduler log together.

## Open

The diagnostic is a latest-attempt snapshot rather than an archive. The exact Week 1 skip reason remains
unrecoverable from retained artifacts.
