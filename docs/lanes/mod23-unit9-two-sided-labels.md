# MOD-23 unit 9b: two-sided points labels (candidate 3)

## Goal
Fit home and away points as separate ridge targets on the base inputs, predict the margin as their difference, and grade against the base on the fair opener test (base 802-701 2020-25; 1142-1089 2011-19, SBR open).

## State (2026-10-18)
Prior-work check (read/measured by grep): no equivalent opener-graded base comparison. Nearest: MOD-17 joint residual screen (registry/experiments/mod17-joint-residual-screen/20260905T160219Z.json, src/nfl_ats/joint_residual_model.py): a joint ridge on margin_residual and total_residual with a union feature set (base margin features plus drive features), alpha 10; it reports margin_opener_delta_accuracy_points -0.73, P+ .24, and a total MAE gain. It is a joint (margin, total) target pair on a different input set, not separate home/away points targets, and it is a margin/total-residual coupling test. Home and away team points targets are not used anywhere else in registry/experiments, ROADMAP, or docs/lanes (grep of home_points/away_points/two_sided/team_points hits only unrelated lanes).
Algebra: the base target is ats_margin = result - spread_line. Residual points targets H - (total+spread)/2 and A - (total-spread)/2 differ by exactly ats_margin; ridge is linear in y for fixed X, alpha and preprocessing, so Hhat - Ahat equals the base fit. Arm P0 (same inputs, same alpha) is therefore identical by construction and is run only as a numeric identity check on one fit, not as a graded look.
Meaningful variants declared before running (scripts/mod23_unit9b.py):
- P1: per-side alpha, chosen by inner leave-one-season-out MSE of each residual side target, grid {1,3,10,30,100,300,1000} (10 = base), selection uses only seasons strictly before the latest training season, cached per season; fewer than 3 prior seasons falls back to 10.
- P2: side-specific inputs, alpha 10: the home-points model drops away_off_* and home_def_* columns; the away-points model drops home_off_* and away_def_* columns (diff_*, context, QB/injury, bias columns kept in both). Mechanism: each side's points depend on its own offence and the other's defence.
Eras: 2011-19 (SBR open) and 2020-25 (Tuesday open); base reproduced 802-701 first from base_per_game.parquet. Looks: 2 arms x 2 eras = 4, family mod23_unit9_two_sided; the P0 check is not a look. Grade: paired decisive accuracy vs base, season-block bootstrap probability_positive, log loss and Brier vs base and market-only, per-season d_acc, flips, selected alphas per season. Eval cache disabled for non-base arms. Leak check: residual targets use only training rows; inner selection excludes the latest training season by construction (assert) and corrupting outcomes at/after a cutoff leaves selected alphas identical.
Closing grounds: only whole interval below zero or a powered positive control; else unresolved_below_power.

## Tried
- Unit 9 EWMA span request was a duplicate of unit-8 candidate 2 (scripts/mod23_unit8_span.py, artifacts/mod23_unit8_span/report_all.json, family mod23_unit8_span, 4 looks, unresolved_below_power): family-LOSO 2011-19 -0.31 [-1.25,0.49] P+.24, 2020-25 +0.27 [-0.28,1.07] P+.69; global-LOSO 2011-19 +1.35 [0.09,2.50] P+.98, 2020-25 -0.87 [-2.37,0.65] P+.13; span 24 (grid edge) chosen in all global folds. No new run. Optional wider-grid extension (32, 48, season-long) deferred, low value.
- Span-adjacent work that does not vary builder span: base-model-recency (row weights), era_weighting_screen, acc-u7b, mod06.

## Next
Run scripts/mod23_unit9b.py, record cells, write results here.

## Open
Whether the wider-grid span extension is worth CPU.
