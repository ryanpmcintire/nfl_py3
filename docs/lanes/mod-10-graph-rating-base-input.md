# MOD-10 unit 1: graph (Massey ridge) rating as a base-model input

## Goal
ROADMAP MOD-10 (Graph model). Add one input to the base model via the add-one harness: home-minus-away ridge/Massey rating (game margin, and EPA/play margin), exponential recency decay, home field, prior games only. Opener grade, LOSO 2020-25 plus 2014-19 proxy.

## State
DONE 2026-11-03 (18:00-18:20). scripts/mod10_graph_rating.py (modes true|proxy|true_control|proxy_control; kinds margin|epa). Grid declared before looking: tau 60/120/240/480/960 days, lambda 1/4/16/64/256, chosen per outer fold by mean held-out MSE over seasons 2011..s-1. Outputs artifacts/mod10/unit1/. 4 cells recorded (family mod10_graph_rating_base_input, unresolved_below_power).

## Tried
Baseline reproduced 802-701. 8 runs (4 arms + 4 permuted controls).
- 2020-25 margin: 800-703, flips 2-4, d_acc -0.13 [-0.28,0.00], coef ~0 and sign unstable, R2 on existing features 0.93 (duplicate).
- 2020-25 EPA: 805-698, flips 27-24, d_acc +0.20 [-0.61,1.14] P+ .60, ll gain .00054 (control .00018), coef -4.9..-7.4 per unit (stable negative), R2 0.92.
- 2014-19 margin: 769-718 = base, flips 8-8. 2014-19 EPA: 779-708 vs 769-718, flips 29-19, d_acc +0.67 [-0.07,1.48] P+ .94, ll .00042 P+ .92.
- Chosen params tau 240d lambda 4 in nearly every fold.

## Next
EPA rating coefficient is negative (corrects overshoot of existing EPA state): test it as a replacement for the diff_off/def_epa rolling features, not an add-on.

## Open
8 looks (4 arms + 4 controls); grid selection inside folds is not a look.

## Unit 2 (started 2026-11-03 18:25, in progress)
Declared: replace the 18 rolling EPA columns (home_/away_/diff_ x off/def x all/pass/rush _epa_per_play; from STATE_METRICS in src/nfl_ats/constants.py) with the EPA graph rating. Arms: replace3 (-18, +3: mod10_rh, mod10_ra, mod10_diff), replace1 (-18, +1 diff), addone (reference, unit 1). Script scripts/mod10_unit2_replace.py (ruff clean), args: mode(true|proxy|true_control|proxy_control) arm. Outputs artifacts/mod10/unit2/<mode>_<arm>_report.json and .log.
Running in background (task b8xwxmopj): true replace3, log artifacts/mod10/unit2/true_replace3.log. Nothing else run yet.
Next: run sequentially one at a time: true/proxy x replace3/replace1, then controls; read reports (rec, compare, ll/brier boot, folds gap); record cells via nfl-ats weak-signals record family mod10_graph_rating_base_input; write served-change description (pinned feature builder, refit, v8).
18:22 chain launched (task bcj8vt475, artifacts/mod10/unit2/chain.ps1): waits PID 28104 (true replace3) then runs true replace1, proxy replace3/1, true_control replace3/1, proxy_control replace3/1; logs <mode>_<arm>.log, reports <mode>_<arm>_report.json. Then read, record, write served-change note.
18:21 in flight: chain powershell PIDs 11368 19848 33716 23212, true replace3 python PIDs 28104/29912 (the chain python children appear after). Remaining steps: read 8 reports, record cells, served-change note. Do not restart chain.
18:40 Unit 2 DONE. 8 runs finished; 16 cells recorded (family mod10_graph_rating_base_input, unresolved_below_power).
2020-25 (base 802-701): replace3 801-702 dacc -0.07 [-1.57,1.48] P+ .43, ll +.00274 [.00067,.00520] P+ .998, brier +.00132; replace1 799-704 dacc -0.20 [-1.87,1.52] P+ .32, ll +.00253 P+ .995. Controls: replace3 789-714 ll +.00169 P+ .91; replace1 791-712 ll +.00178 P+ .92.
2014-19 (base 769-718): replace3 766-721 dacc -0.20 [-1.14,0.82] ll +.00221 P+ .996; replace1 769-718 dacc 0 [-0.80,0.81] ll +.00200 P+ .998. Controls: replace3 759-728 ll +.00036 P+ .71; replace1 773-714 ll +.00109 P+ .999.
Mean in-minus-out MSE gap -0.6 to -2.2 (out-of-sample gain exceeds in-sample).
Served change would need: edit pinned feature builder/STATE_METRICS (drop 18 columns, add rating columns computed prior-games-only, chosen tau/lambda per fold), constants, refit, re-enroll as v8; not done.
