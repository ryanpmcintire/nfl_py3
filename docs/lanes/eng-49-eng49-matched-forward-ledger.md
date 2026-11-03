# ENG-49 Matched-game forward ledger

## Goal
Matched-game forward ledger: one schema carrying every frozen candidate probability on the same game ids (ROADMAP ENG-49).

## State
Added 2026-11-03, unmeasured. Each forward writer has its own shape, so paired comparisons and the monitor need ad-hoc joins. One weekly table with served, market and each frozen candidate per game gives free paired differences and decisive-game records.

## Tried
Nothing yet. Novelty: forward-log-audit-2026-11-03 (read-only coverage) and scripts/mod24_forward_log.py (one arm). Grep found no unified table.

## Next
First unit: Write the join for Weeks 4-9 from existing run files into a single parquet under artifacts/, with a manifest listing any missing week or candidate.

## Open
Data: Existing forward artifact runs. Zero crossing closes nothing; record unresolved results with a closing ground.
