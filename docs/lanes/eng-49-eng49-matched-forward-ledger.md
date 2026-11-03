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

## Unit 1 (2026-11-03): built, infrastructure only, no verdict
- scripts/forward_ledger.py writes artifacts/forward_ledger*/ledger.parquet + manifest.json. A record counts only if written_at < the game deadline (min of row deadline, kickoff) and, for forward_terms, written_at >= frozen_created_at_utc; earliest record per candidate and game kept. MOD-27 backfill rows are excluded. POL-10/11 are side-only (no probability) and enter only via --pol-ids.
- Measured Weeks 4-9 (88 games): strict all-candidate match = 0 games. forward_terms (sky01/01snap/03/03sep/env02/surface) exist for Week 9 only; sky04 has null term values; mod35_cold_exposure has frozen coefficients but no forward file; draftcap Week 7 only; mod25_wfj and mod27 live rows Week 9 only; lead99 Weeks 7-9.
- Core match (mod24, mod25_combined, lead99): 28 graded games (Weeks 7, 8); Week 9 pending, 0 matched yet. Missing: lead99 45, mod24 15, mod25 16.
- Weekly command (for a scheduler Job after Sunday results load): `uv run --no-sync python scripts/forward_ledger.py --out artifacts/forward_ledger` (strict) and `... --require mod24 mod25_combined lead99 --out artifacts/forward_ledger_core`.
## Next
Add the Job; rerun after Week 9 results; extend --require as candidates accumulate games. XLG-12 CFB ledgers kept separate.
