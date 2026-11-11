# MOD-57 P2-N - margin error against the line

## Goal
Does anything we can build predict the final margin better than the closing line, and by how much (absolute and squared error, residual calibration slope). Parent: mod57-sixty-percent.md. Script scripts/mod57_p2n_margin_error_vs_line.py; outputs artifacts/mod57_p2n/ (oos_predictions.parquet per-game predictions of all five arms, results.json, batch.json).

## State (2026-11-11 10:14, measured)
Novelty: ENG-46 u1 (every_metric_served_backtest_v1) had served ridge margin MAE 9.952 v spread 9.844 2018-25; MOD-24 finer grade had low-capacity arms v the opener. No gradient-boosting or elastic-net residual model, shrink arm or residual slope existed.
Run: 4,431 regular-season games 2009-2025, 2,895 test games 2015-25 (11 season blocks), 295 joined columns over 8 tables (83/185/273/248/254/248/134/273), 88 served; three outcome columns (home_cover, home_score, away_score) found by the leak screen and removed before the run.
Line MAE 9.818, RMSE 12.729. Gain over line (positive better), abs / sq / slope [95%]: R -0.191 / -5.53 / 0.05 [-0.09,0.22]; S -0.007 / -0.02 / 0.42 [-0.87,2.07]; G -0.069 / -1.50 / -0.10 [-0.75,0.19]; H -0.024 / -0.35 / 0.22 [-0.31,0.65]; E -0.002 / +0.05 / 0.69 [-0.06,2.01], P+ slope .96. 15 cells recorded, family mod57_margin_error_vs_line (file confirmed).

## Tried
All five arms once. Shrink factors 0 to 0.4, zero in 6 of 11 seasons; boosting chose 1-4 iterations mostly.

## Next
Accuracy cells (vs close and opener) are reported not recorded (no interval computed). Next unit: any arm that predicts a residual must carry real new information (E slope is the only one near 1); candidate: E restricted to features with stable sign, or targeted opener residual.

## Open
Leak screen top corr 0.099 (away_qb_cpoe, QB rate columns); pregame status of pbp/player tables read from builders, not truncation-tested. Opener comparison uses open + predicted residual though arms were trained on the closing residual.
