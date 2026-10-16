# MOD-23 unit 7 - train on the opener, not the close

## Goal
Test whether the served base model (market_residual, ridge alpha 10, weak_stack) improves on the fair opener test when its training target and market input are the opener instead of the nflverse closing line. Baseline 802-701 (53.36%) on 1,503 decisive 2020-2025 games.

## State
- Target finding (read): features.py:49 `ats_margin = result - spread_line`, spread_line is the nflverse (closing) line; margin.py:883-957 `fit_margin_model` trains on `ats_margin` with `spread_line` as a market input column. At grading clv.py:2201-2207 swaps `spread_line` to the Tuesday opener (or close) in the scoring frame only. So training line (close) differs from the line the pool grades at (opener).
- History check: ROADMAP MOD-20/MKT-20 (line move as target) and XLG-09 opener_error_transfer_v1-5 (registry families line_move_target_v1, opener_error_transfer_v*) test move prediction / pooling, not retraining the margin residual on opener. MOD-24 (ROADMAP 824) searched 2026-10-01: no family trains on a denoised or per-half target. grep of docs/ROADMAP/scripts finds no opener-trained base. Untested.
- Declared before scoring (2 looks, family mod23_unit7_target_line): A = rows with an opener (SBR 2011-2019, archive opener 2020+) with `spread_line := opener`, `ats_margin := result - opener`, scored at the opener. C = control, same rows (2011+ only), close-trained, to separate the window change from the target change. Compared to the official base artifact. total_line stays the closing total in both (not changed).
- Metrics: decisive accuracy diff, season-blocked bootstrap (20000, seed 20260817) with probability_positive, log loss, Brier, per season.

## Tried
- DONE 2026-10-16 (artifacts/mod23_unit7/report.json). A open-trained 801-702 (53.29%), diff -0.07 pts, season-blocked [-1.04,+0.81], P+ 0.43; LL 0.6972 vs 0.6969 (P+ of LL rise 0.61), Brier 0.2518 vs 0.2517. C close-trained same rows 801-702, -0.07 [-1.39,+1.03], P+ 0.45, LL 0.6976. Per-season A d_acc: -1.8,+1.3,-1.6,0.0,+0.4,+1.1. Recorded unresolved_below_power x2.

## Next
- Target mismatch is not the lever. Next: a different base-model change with a prior-work check.

## Open
- Pre-2011 rows have no opener, so arm A trains on fewer seasons than base.
