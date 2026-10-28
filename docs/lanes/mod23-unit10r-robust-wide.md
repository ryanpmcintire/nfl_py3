# MOD-23 unit 10r: wide-grid robust target + arm T

## Goal
Test the range v1 (mod23_unit10_robust_target) left untested: q grid {0.40,0.50,0.60,0.70,0.80,1.00}. Family mod23_unit10_robust_target_wide, script scripts/mod23_unit10r_wide.py, artifacts artifacts/mod23_unit10r_wide/. Zero crossing closes nothing; one fitted probability picks the side (grading path unchanged from v1).

## Declaration (before running)
Arms W (ridge alpha 10, ats_margin winsorized at c = quantile q of |label|), H (Huber, alpha 10/sigma0, epsilon = quantile q of |resid|/sigma0), q chosen by inner LOSO over seasons before the latest training season, criterion held-out MAE on raw labels. Arm T mechanism: touchdowns scored on interceptions, lost fumbles, punts and kickoffs are luck of defense/special teams not offensive strength, and Q4 scoring at win probability <0.05 or >0.95 reflects prevent play and surrender; removing the points (plus the XP/2pt following a removed score) from the label removes noise unrelated to team strength. Target = ats_margin minus (removed home pts - removed away pts), from data/pbp (labels may use the game itself; inputs stay pregame). T has no tuned parameter (garbage threshold is the standard nflfastR-style 0.05/0.95 convention, not fitted: a declared convention, not leave-one-season-out chosen). Residual distribution for probabilities uses raw ats_margin.
Looks: 3 arms x 2 eras = 6 in this family; cumulative with v1's 4 = 10 (T is a new mechanism, but counted).
Eras: 2011-19 proxy (base 1142-1089), 2020-25 true opener (base must reproduce 802-701).

## State
Run launched in background; see artifacts/mod23_unit10r_wide/run.log. Reconstruction check on 2023: pbp final scores equal home/away_score in 95.4% of games' home end value. PBP coverage of feature rows 92.6% (rest keep raw label).

## Next
Read report_all.json, record cells via weak_signals_batch.json.

## Update (in flight)
Run started 8:41 PM 2026-10-27 as tool background task b3xtr70wy, python PID 11952 (parent 19304). Passed: base 802-701 / 1142-1089 reproduced, q=1 identity diff 0.0, W selection unchanged under corrupted latest season (q 0.4). Positive control (labels shifted by a pattern) did NOT change W selection: control invalid/unproven, report as such. H leak check slow (Huber hits max_iter at low q). Outputs when done: artifacts/mod23_unit10r_wide/{run.log,report_all.json,W_q.json,H_q.json,weak_signals_batch.json,*_run.parquet}.
Next: when "better arm" appears in run.log, read report_all.json, then `.\.tools\uv.exe run --no-sync nfl-ats weak-signals record --batch artifacts/mod23_unit10r_wide/weak_signals_batch.json` (6 cells, unresolved_below_power); not yet recorded. Kill PIDs 11952/19304 if abandoning.

## Update 2 (T relaunch)
First run (PIDs 11952/19304) ended: W and H finished and cached (W_run/H_run.parquet); arm T crashed KeyError ats_margin_t (harness drops extra columns). Fixed via game_id map CTX["t_map"]. Relaunched as tool task bw96zctbl, log artifacts/mod23_unit10r_wide/run2.log (reuses W/H parquet; reruns checks). W/H results (d_acc pp, pp+): W 2011-19 -0.72 (0.134), W 2020-25 -1.00 (0.077); H 2011-19 -0.13 (0.351), H 2020-25 -0.87 (0.000, interval [-1.42,-0.32]). Record only after report_all.json exists.

## Coordinator update
W and H cells already recorded by the orchestrator (names mod23_unit10r_wide_{W,H}_{sbr_2011_2019,tuesday_2020_2025}; H 2020-25 is refuted_mechanism / wrong_sign_resolved). Do not re-record them or use --replace. After run2 finishes, record only T: mod23_unit10r_wide_T_sbr_2011_2019 and mod23_unit10r_wide_T_tuesday_2020_2025 (build a T-only batch with these names; the script's weak_signals_batch.json has W/H/T under other names).
Overlap note: arm T overlaps acc-u5 arm T2 (scrimmage EPA, wp 0.05-0.95, no special teams), which already went -0.73 pts in 2020-25; T here is a points-based variant (removes non-offensive TDs and Q4 garbage-time scoring, special teams included), so it is not independent evidence.

## RESULT (done)
run2 finished; T recorded (2 cells, unresolved_below_power; W/H untouched). d_acc pp (pp+): W -0.72 (0.134) / -1.00 (0.077); H -0.13 (0.351) / -0.87 (0.000, wrong_sign_resolved by orchestrator); T -0.45 (0.282, 1132-1099, flips 184-194) / -0.86 (0.098, 789-714, flips 129-142). Base reproduced. Chosen q: W 0.4 (grid edge, every fold from 2012), H 0.7-1.0. Leak checks passed (W, H); positive control invalid (W selection unchanged). Reliability (H): 2020-25 deciles mean_p 0.35-0.63 vs mean_y 0.53-0.56 low end, flat. Overall: wider grid does not rescue; W still sits at the new edge 0.4. Artifacts artifacts/mod23_unit10r_wide/ (report_all.json, run2.log, weak_signals_batch_T.json).
Next: none required; optional W q below 0.40 is not worth running (negative both eras).
