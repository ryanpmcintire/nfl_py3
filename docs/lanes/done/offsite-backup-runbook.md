# OPS-06 capture reconciliation runbook

## Goal

Make the backup runbook describe the shipped capture reconciliation behavior.

## State

**Read, 2026-09-26:** platform selection and capture-only scheduling already
exist. The runbook now separates current source behavior from the September 10
deployment account. Current remote health and deployed revision remain unverified.

## Tried

- Reviewed `scripts/sync_captures.py` and the owning scheduler job and role guard.
- Documented same-name and near-window deletion, gap-fill name/size verification,
  `--keep-remote`, `--dry` log writes, and feature-table size-only freshness.
- Corrected the scheduler description to cover deletion of accounted-for copies.
- **Measured:** scoped Ruff check, format check, and the real local scheduler
  help command all exited 0; logs are in `.tmp/offsite-runbook-*.log`.
- No SSH, capture, backup, deletion, credential access, or deployment was run.

## Next

This documentation task is complete. OPS-06 remains open for operational
verification of the deployed server and capture reconciliation.

## Open

The sync treats equal file sizes as current and can delete differing near-window
observations. These are documented behaviors, not content-integrity guarantees.
