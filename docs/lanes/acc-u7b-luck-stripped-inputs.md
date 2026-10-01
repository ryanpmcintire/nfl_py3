# acc-u7b-luck-stripped-inputs

## Goal
Rebuild margin/EPA history inputs with non-repeating scores stripped; grade as a base-model change at the opener (paired vs base). Parent: docs/lanes/acc-u7-new-information.md. Family for recording: mod24_luck_stripped_inputs.

## State
2026-10-01. Absence verified (read): base inputs are EWMA (span 8, min 3, retention 0.67) of raw per-game team_stats metrics in src/nfl_ats/features.py build_team_game_metrics/build_team_states; point_diff and ats_residual are raw score differences, EPA from passing_epa+rushing_epa. Nothing strips defensive/return TDs or fumble luck. Registry close_game_luck_* and special_teams_* cells are prior-season quartile fades or separate terms, not cleaned base inputs.
Code scripts/mod24_u7b.py (stages: prep, grade). Prep done (artifacts/mod24_u7b/prep.log, clean_minus_raw.parquet): PBP counts 1079 non-fumble def/return/blocked TDs, 786 INT TDs, 4574 run/pass fumbles lost; expected values from seasons before each season (2009-10 use 2009-10). Cleaning via monkeypatch of F.build_team_game_metrics (no src edits), delta added to the parquet columns. Raw rebuild vs parquet: max abs diff 1.59 on 4703 common games, NaN pattern identical (not exact; delta approach used).
Grade stage launched in background (detached): log artifacts/mod24_u7b/grade.log; writes b1_run.parquet, b2_run.parquet, b3_selection.json, b3_per_game.parquet, report.json (compare5 vs artifacts/mod24_u5/base_per_game.parquet). B1 = point_diff/ats_residual cleaned; B2 = B1 + six EPA inputs; B3 = blend raw + w*(B2-raw), w from {0,.25,.5,.75,1} by inner-season MSE.

## Tried
Graded 2026-10-01 (measured, report artifacts/mod24_u7b/report.json). Pooled 2011-25 vs base 1944-1790: B1 1921-1813 d_acc -0.62 (P+ .03, 6/15 seasons, flips 67-90); B2 1931-1803 -0.35 (P+ .14, 6/15, flips 108-121); B3 1932-1802 -0.32 (P+ .06, 3/15, flips 54-66). True era 2020-25 (base 802-701): B1 -0.53, B2 -0.73, B3 -0.07 (801-702). Proxy 2011-19 (base 1142-1089): B1 -0.67, B2 -0.09, B3 -0.49. Temp-recalibrated log loss differences within +-0.0002 (P+ .08-.76); true-era lattice RPS slightly worse for all (-0.0009 to -0.0021). B3 picked w=1 in 2011-14, 2019-22 and w=0 in other seasons (inner MSE gain tiny, about 0.1 percent). Recorded 3 cells, family mod24_luck_stripped_inputs, unresolved_below_power.

## Next
Inference (inferred): cleaning the inputs does not help picks; if anything accuracy dips, no mechanism-level sign resolved. Not queued: FG and short-field cleaning, Elo cleaning.

## Open
Raw rebuild not exact vs parquet (max diff 1.59), delta approach used. About 50 looks (3 arms x 3 eras plus 15 w selections).
