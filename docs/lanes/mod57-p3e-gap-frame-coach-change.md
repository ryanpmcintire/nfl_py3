# MOD-57 P3-E: gap-frame harness and coach-change run (2026-11-11 12:55)

## Goal
Reusable "hindsight gap, weeks 1-6" harness `scripts/mod57_p3_gap_frame.py`; first input = new HC/OC/DC count, home minus away. Family `mod57_team_change_gap_frame` (13 looks; this unit 2: gap label, margin label, both weeks 1-6).

## State
Built and run (measured, artifacts/mod57_p3e/): `scripts/mod57_p3e_coach_change_features.py` makes `coach_change_features.parquet` (4,703 games; columns coach_new_hc_diff, _oc_diff, _dc_diff, _sum_diff). A role is "new" when the holder in the latest revision dated before that game's kickoff differs from the holder in the prior season's last revision dated before that season's final kickoff. Slots with no prior or no pre-kickoff holder count 0 (3,119 slots of 28,218, mostly 2009 and missing OC/DC; 5 no-holder-before-kickoff). Outputs: `coach_change_results.json`, `coach_change_table.txt`, `coach_change_batch.json`, run.out/err.
Results weeks 1-6, 1,016 games 2015-25 (measured). Decisive-game record vs close: gap-label model 541-458 (intercept-only 531-468), margin-label 508-491; vs opener 2020-25 both 280-244 (the same constant home lean; mean |pred| 0.12 and 0.26 pts). Share of gap predicted -0.0067 (in-sample +0.0036, gap +0.0103; perm percentile 0.013, i.e. worse than shuffled). MAE vs close 9.8401 model vs 9.8415 line, gain +0.0014 [-0.0062, +0.0094] P+ 0.62. Slope on margin residual -1.14 [-4.37, +4.50] P+ 0.23. Margin label: gain -0.0068 [-0.0241, +0.0080] P+ 0.23, slope -1.26 P+ 0.21, perm percentile (share) 0.86. Placebo weeks 7-12 gap: gain -0.0034 P+ 0.07, slope +0.06; margin gain -0.0177 P+ 0.08, slope +1.19 P+ 0.94. On top of the 295-col EL: added gain -0.0020 (gap, P+ 0.05) and -0.0007 (margin). Fold signs (gap, wk1-6): DC +6/-0, HC -6/+0, OC -3/+0, sum all zero.
Recorded 2 cells (mae_improvement, unresolved_below_power) into registry/weak_signals/mod57_team_change_gap_frame.json; `weak-signals pool --league nfl --effect-units accuracy_points` loads.

## Tried
Single-process, OMP_NUM_THREADS=1; run takes about 35 min (4,000 permutation walks dominate). Stored P2-P `pred_label_EL` and P2-N `pred_E` serve as the 295-col baseline (read, not refit). Elastic net trains on same-window weeks only.

## Next
Run the harness on the next input (roster continuity, QB change, draft class):
`F:\Repos\nfl_py3\.tools\uv.exe run --project F:\Repos\nfl_py3 --no-sync python F:\Repos\nfl_py3\scripts\mod57_p3_gap_frame.py --features <parquet with game_id> --columns a,b --tag <name> --out F:\Repos\nfl_py3\artifacts\mod57_p3<x>` (flags `--permutations 1000`, `--skip-stack`), then `nfl-ats weak-signals record --batch <out>\<tag>_batch.json`.

## Open
The gap-label pick advantage over the intercept (+10 games) is not from a mechanism claim; no flip rule. Interim_coaches table not used (coordinator history carries in-season HC swaps from 2022). In-season revisions exist only 2022+, so 2015-21 holders are preseason snapshots. Placebo cells and slope/share are in the JSON, not the registry.
