# clv-metric-everywhere (ENG-47)

## Goal
Make the signed opener-to-close line move toward the pick (points) a standard, low-variance yardstick in every backtest and opener evaluation, and use it to test whether the served card adds anything beyond following the line.

## State
- Unit 1 DONE, committed `df535bd`: `backtest.summarize_predictions` and `clv.opener_pick_evaluation` / `opener_evaluation_metrics` / `_metric_draws` emit `line_move_toward_pick_{mean,games,pushes,no_close}`; opener-eval cache version `3-line-move-toward-pick`. Run with `--features data/processed/game_features_weak_stack.parquet` (default features file is stale for the active model).
- Measured unpaired, `artifacts/opener_evaluation/20260923T172849Z/`, 1,537 games: raw-model pick +0.131 [0.089, 0.178]; discrete probability-rule pick +0.094; always-favourite +0.103; always-home +0.099.
- Unit 2 DONE 2026-09-23 (`scripts/line_move_yardstick_paired_eval.py`, `artifacts/line_move_yardstick/20260923T180055Z/`, family `line_move_yardstick_v1`, 5 cells, all unresolved_below_power), paired season-block:
  - model-only vs always-favourite (clean): -0.007 [-0.076, +0.070], P+ 0.42, decisive 305-252 on 557, sign flips by season.
  - opener-only four-term (model logit + flag sum, LOSO) vs model-only: +0.044 [-0.029, +0.110], P+ 0.88.
  - opener-only vs always-favourite: +0.028 [-0.058, +0.127], P+ 0.72.
  - served four-term vs model-only +0.236 and vs favourite +0.222 are CONTAMINATED: its move term uses post-open sharp movement (`leader_median_through_sunday_prekick_v1`). Never quote them as beating the market.

## Tried
Composition flags confirmed pregame-only, so the flag-sum term is clean on this yardstick.

## Next
The raw model shows no line-move edge over always-favourite; the flag sum leans positive (P+ 0.88) but is below power. Any new signal lane should report its paired line-move cell beside accuracy.

## Open
None.
