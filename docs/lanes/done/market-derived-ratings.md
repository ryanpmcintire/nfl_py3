# market-derived-ratings

## Goal
MOD-21: fit team ratings from closing lines; predict opener minus rating-implied line.

## State
- 2026-09-16: queued by owner; ROADMAP row written.
- Unit 1 DONE 2026-09-16 (subagent-measured, parent-verified by rerun).
  Script `scripts/market_derived_ratings_eval.py`; artifact
  `artifacts/market_derived_ratings/20260916T192813Z/` (`summary.json`,
  `per_game.csv`). Two cells recorded in family
  `market_derived_ratings_v1` (registry now 6,587 with the day's other
  cells).

## Tried
- Walk-forward ratings on closes (decay 0.9, ridge 3.0, week-1 HFA 2.5,
  point-in-time asserted on all 104 season-weeks), 1,503 games 2020-2025.
  Variance ratio close/margin is 0.18, not the row's 1/70 guess.
- (a) residual predicts the move: pooled OOS corr -0.056, MSE gain
  +0.0075 [-0.0196, +0.0214], P+ 0.73 — nothing there.
- (b) R2 (served + residual) vs served accuracy: -1.93 pts
  [-3.32, -0.29], P+ 0.009, decisive 140-169 on 309 — recorded
  refuted_mechanism / wrong_sign_resolved. Brier companion +0.00175,
  P+ 0.976 — recorded unresolved (opposite axis, tiny).
- IS -1.26 vs OOS -1.93 (gap +0.67). 16 looks, all reported.

## Next
- Closed on accuracy; Brier axis stays unresolved. No follow-up unless a
  different residual construction is proposed.

## Open
- None.
