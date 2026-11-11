# fix-picks u5 - generalized crossing term (LV) leave-one-season-out

## Goal
Decide whether one term LV (push-mass crossed between a line and the pool line) should replace move + hook in the served stack. Parent: fix-picks-2026-11-10.md. Measurement only.

## State (all measured, 2026-11-10)
Scripts tests/scratch/fixpicks/u5_build.py, u5_loso.py, u5_fwd.py; outputs u5_loso.out, u5_fwd.out, u5_oos.parquet, u5_fit.pkl.
- Population: served build_fit_population keep_pushes=True, 1537 games 2020-25 (34 pushes at open). Whole-number Tuesday lines 661 split into pool lines M+0.5 and M-0.5 (weight 0.5 each); others graded at their line; 2198 rows. Deadline leader median (loso.py logic) present for all 1537; books per game 3.0 in 2021-25, 2.99 in 2020; served move_available is 0 in 2020-22, so G sees a gap there that S does not.
- Push table per fold: seasons before each row's season and never the held season. M (hook generalized, move kept) is identical to S on history (pool is always within 0.5 of the Tuesday line): checked allclose.
- S reproduction: hook coef full 0.805, folds .745-.843 (lane .77-.87); log loss gain of hook on whole-line rows .00123 (lane .00123). Level and counts do NOT match the lane (mine 757-565 vs 742-580 no-hook, ll .6785 vs .6798; lane 729-593 vs 726-596, ll .68223 vs .68343): population or move table version differs; unresolved.
- LOSO vs S (arm minus S, week-block bootstrap, 107 blocks): G 2023-25 ll -0.0031 [-.0093,+.0029] P+ .16, acc +0.31 pts P+ .60, differ 120.5 games arm 61.5 / S 59.0; 2020-22 ll +0.0028 [-.0017,+.0076] P+ .88; all ll -0.0003 P+ .44, acc +0.07 P+ .52, differ 247 games 124-123. G1, G805 same within .0001. G coef c .82 (folds .75-1.02, sd .089) vs S c .805 (sd .031); G stage-1 model_logit .373 (S .308). Stage 1 without move and no LV (G_stage1only) is worse than S, ll -0.0039 P+ .05. LV-only and intercept controls lose to S (LV-only acc 53.7%, ll .6890; intercept 49.6%).
- In-sample minus OOS ll gap: S .00098, G .00075.
- 2026 W1-9 forward (136 games, flags recovered by inversion from served probability, noisy: mean .196 from integer): card 70-66 (served p 74-62), S 77-59, M 78-58, G 77-59, G1 78-58, G805 77-59. Log loss vs card: S -.0007, M +.0011, G -.0027, G1 -.0009, G805 -.0029 (all P+ .32-.69). Flag-free: S 83-53, M 83.5, G 77-59 (ll .6821), LV-only 77-59 (.6830). Differing vs card: G 23 games arm 15 card 8; S 15 games 11-4; M 16 games 12-4.

## Tried
S, S no-hook, G, G1, G805, G stage-1 only, intercept, LV-only, S served-style (no held-season exclusion in push table; equals S within 0.0001 ll).

## Next
Serve decision (inferred): G does not beat S on the historical out-of-season test where quotes are dense (2023-25 ll worse, P+ .16) and gains only where S has no move input; keep S this week. M (move kept, hook generalized to any offset size) is the better candidate for the 2026 case where pool and Tuesday market differ by more than 0.5; history cannot test it (always within 0.5), forward 78-58 is not out-of-fit evidence. Closing grounds: none met, so unresolved_below_power. Proposed record (not run): nfl-ats weak-signals record --name lv_generalized_crossing_vs_move_hook --classification unresolved_below_power --effect-units log_loss_improvement --effect -0.0003 --interval-low -0.00436 --interval-high 0.00364 --probability-positive 0.444 --season-start 2020 --season-end 2025 --family fix_picks_lv_2026_11_10 --source tests/scratch/fixpicks/u5_loso.out --league nfl.

## Open
Why S level differs from lane; whether to fit G with the served move-table in stage 1 plus LV beyond the move's coverage; true flags for 2026.
