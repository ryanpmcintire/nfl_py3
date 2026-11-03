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

## Fix 2026-11-03 16:12 ET (measured)
- scripts/forward_ledger.py: new _played_card() (last published_picks row per 2026 game with published_at_utc <= pick_deadline_utc; served_p = displayed_score if pick_team is home else 1 - displayed_score; served_side = pick_team side, not p>=0.5, because Week 1 CHI_CAR is an exact 0.5). card_line = market_spread (equals game_features spread_line on every game). Old candidate-record base kept as ledger column base_at_record_p and per-candidate <name>__base_p.
- Check (artifacts/forward_ledger_allweeks, Weeks 1-9): served_side matches pick_team on 136/136 games; ledger vs prospective_scorecard 20261103T192211Z served_card W-L: W1 9-7, W2 14-2, W3 6-10, W4 9-7, W5 8-7, W6 6-8, W7 7-7, W8 6-8, all equal.
- Reran artifacts/forward_ledger and forward_ledger_core (28 matched graded), forward_monitor, eng50 stacker. Monitor vs played card now: mean log-loss gain lead99 +0.0479 (final e 1.92), mod25_combined +0.0231 (e 0.44), mod24 +0.0073 (e 0.39); previously +0.0198, -0.0050, -0.0208. No e-value reaches the threshold; sign flipped for mod24 and mod25, all unresolved.
