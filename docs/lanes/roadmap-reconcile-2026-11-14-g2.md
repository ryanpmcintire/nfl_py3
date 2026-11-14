# Roadmap reconcile 2026-11-14, group 2

Rows: MKT-21..28, XLG-10..15, PER-11..16, ENV-02, LEAD-75 (22).

## Goal
Transcribe the recorded lane state of 22 ROADMAP rows still marked not started into a verdict file so the orchestrator can update the roadmap. Read-only: no analysis, no new research verdict, no ROADMAP, registry, src, scripts or tests edit.

## State
Done 2026-11-14. Verdicts in `tests/scratch/reconcile_2026_11_14/g2.json` (22 objects, parses, plain ASCII).
- done 2: MKT-27, MKT-28.
- forward_only 4: MKT-21, MKT-23, ENV-02 (2027-01-11, after Week 18), PER-13 (2027-01-05, after Week 17). Dates are inferred from the Tuesday cycle dates in lead75.md; no schedule file was read.
- in_progress 16. Row-declared work still owed (9): MKT-26, XLG-10, XLG-12, XLG-13, XLG-15, PER-11, PER-14, PER-16, LEAD-75. Row's first unit complete and the lane names a further unit (7): MKT-22, MKT-24, MKT-25, XLG-11, XLG-14, PER-12, PER-15; the orchestrator may bucket these as done.
- not_started 0, blocked 0.
- Every recorded cell is unresolved_below_power. Nothing is served and nothing is closed.

## Tried
- Read all 21 lanes, the ENV-02 entries in backlog-all-2026-11-02.md, and done/surface-switch-fitted-term.md.
- Registry (Grep and Glob): all 29 cited family files exist under registry/weak_signals. File existence only; cell counts were taken from the lanes.
- Scheduler log `xlg12_`: 8 lines; last runs OK 2026-11-08 (score) and 2026-11-10 (freeze).
- No script run, no process started.

## Next
In_progress units, most valuable for the weekly picks first (full list in g2.json `next_unit`):
1. PER-16 step 4: coach-keyed shrunk Questionable play-rate offset inside the player availability input, graded on the served base with intercept-only and line-move-only controls. Held-out gain is the strongest in this group. Availability features may fall under the pinned-source guard.
2. MKT-22: leave-one-season-out add-one of pass-rate-when-ahead on the served probability, extending scripts/mkt22_trait_pricing.py.
3. XLG-10: leave-one-season-out check and served-path grade of short_week_diff (scripts/xlg10_addone.py, scripts/pooled_screen.py).
4. XLG-12: `uv run --no-sync python scripts/xlg12_weekly.py score`, then repair the cached-week skip in the Tuesday freeze.
5. XLG-11: college-centred prior inside the 88-input ridge against the served model.
6. LEAD-75: pregame no-leakage check of the pass-rate input before the next look.

## Open
- XLG-12 (measured, file listing and scheduler log): both jobs exit OK but produce nothing. 11-08 score printed no finished ledger rows; 11-10 freeze wrote 0 rows for week 11. Both forward_ledger_open.jsonl files are unchanged since 2026-11-03 (393 rows: week 10 372, week 11 21) and no forward_scores.csv exists. Inferred cause: the score job does not ingest results first, and the capture skips a week it already cached, so week 11 kept the 7 games that had lines on 11-03.
- XLG-13 (read): registry/weak_signals/xlg13_screen_transfer.json still holds the pre-fix values (Pearson -0.84, Spearman -0.40, file dated 11-03 13:53) that the lane struck; the fixed-harness result is unrecorded.
- ENV-02 (read): the row still says the 4,380 issuance values are timezone-naive and blocked; the backlog lane resolved them as UTC and the roof term was fitted and recorded (env02_roof_fitted_term).
- Row and lane differ: XLG-15 row names wind elasticity, lane measured three other traits; PER-14 row says OC or DC, lane is OC only and its unit 1 correlations are not in the registry; MKT-21 (2023-25, not 2020-25), MKT-26 (2023-25, not 2013-2025) and PER-11 (2013 on, not 2009) ran on shorter windows than the row states. The id PER-14 is also used by an older registry family, per14_unit_prior (2026-09-05).
- Cells whose whole interval lies below zero yet are recorded unresolved, with the lane's reason: MKT-27 tercile add-one, ENV-02 roof term, XLG-14 load residual, MKT-28 SIM-18 terms. No verdict issued here; the orchestrator verifies the classification.
- LEAD-75 (measured, scheduler log): the 2026-11-03 Tuesday leg was MISSED, so that week has no pairs; 2026-11-10 captured 14 games.
