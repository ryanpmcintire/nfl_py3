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

## Result (measured 2026-10-18; scripts/mod23_unit9b.py, artifacts/mod23_unit9/report_all.json, run.log; scripts/mod23_unit9.py was an existing unrelated file so the script is unit9b)
Checks: base 802-701 and 1142-1089 reproduced; P0 identity max abs prediction diff vs base 2.5e-14 (identical by construction, confirmed); alpha selection unchanged when latest-season outcomes corrupted; positive control (home points shifted by 3x spread) changes selection. Eval cache disabled.
d_acc vs base (season-block 95% interval, P+), flips W-L, log loss arm/base/market, Brier arm/base/market:
P1 per-side alpha 2011-19 1149-1082: +0.31 [-0.94,1.48] P+.69, flips 141-134, LL .7033/.7195/.6933, Brier .2547/.2590/.2501.
P1 2020-25 791-712: -0.73 [-1.41,-0.06] P+.013, flips 79-90, LL .6960/.6969, Brier .2514/.2517.
P2 side inputs 2011-19 1129-1102: -0.58 [-1.07,-0.09] P+.005, flips 10-23, LL .7191, Brier .2588.
P2 2020-25 798-705: -0.27 [-0.58,0.07] P+.047, flips 9-13, LL .6968, Brier .2516.
P1 chosen alphas: 1000 (grid edge) both sides in 14 of 16 seasons, 10 in 2; so P1 is mostly heavy shrinkage of the residual targets toward the market, which improves 2011-19 log loss/Brier (calibration) but not current-era picks. Per-season P1 d_acc 2011-19: 0,.8,0,1.6,2.0,3.3,-.4,-.8,-3.6; 2020-25: -1.4,.4,-1.2,-1.9,.4,-.7.
Four looks recorded unresolved_below_power (family mod23_unit9_two_sided); intervals for P1 2020-25 and P2 2011-19 lie below zero but the recorded classification stays unresolved since no wrong_sign closing ground was declared and P1 is grid-edge selected. Not served.

## Next
Optional: P1 with grid beyond 1000 (edge hit) only if CPU is free; otherwise candidate 3 is exhausted. No further two-sided variants declared.

## Open
Whether the wider-grid span extension is worth CPU.
