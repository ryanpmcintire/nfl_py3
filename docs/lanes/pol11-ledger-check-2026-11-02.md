# pol11-ledger-check-2026-11-02

## Goal
Read-only: card-ledger-check W1-8, map POL-11 deliverables to evidence, grade side-changing late-week refreshes.

## State
- card-ledger-check (measured 2026-11-02): only week 8 checked; 14 paper rows, 0 pick-revision rows; no disagreements. W1-7 not checkable (pre-kickoff scope only).
- Graded ledger artifacts/prospective/pick_revisions.graded.parquet (arms before_refresh / refreshed / model_only_off_arm, 43 games, no pushes). Side changed before->refreshed: 38 games, refreshed 21-17, Tuesday side 17-21. W2-8 only: 32 changed, refreshed 20-12, Tuesday 12-20. W1: 6 changed, refreshed 1-5 (retired follow rule/handle). All 43: refreshed 25-18 vs 21-22.
- Graded against the Tuesday line, not the opener (opener grade not in this ledger).
- Deliverables: refresh built (src/nfl_ats/pick_refresh.py, docs/late_week_refresh.md); follow rule LATE_WEEK_FOLLOW_SERVED=False (pick_refresh.py:125); move as fitted term via pick_probability.py MARKET_MOVE columns (line 98); consistency guard card_ledger_check.py. Gap: repair --record-decisions --replace-week unrun/owner-blocked; Section 5 inactives experiment and refresh-path settlement not done; evidence base (P+ 0.935-0.981) was for the retired follow rule, not the fitted term.
- late-flip-mechanism lane: W2-8 Tue->final 13-10 vs Tuesday 10-13; defect sharp_book_movement_features.py:136-139 (single-book baseline, LV_NYJ).

## Tried
Reads only; no refits.

## Next
Named unit: LOSO baseline-outlier guard for move feature; opener-line grade of changed sides.

## Open
Is the --replace-week repair still needed now the check is clean for W8?
