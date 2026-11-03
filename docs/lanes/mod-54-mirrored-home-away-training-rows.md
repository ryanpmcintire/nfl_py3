# MOD-54

## Goal
Mirrored home and away training rows: fit each game from both teams' views so the base is exactly antisymmetric (ROADMAP MOD-54).

## State
Unit 1 done 2026-11-03 18:17 (scripts/mod54_mirrored_rows.py, artifacts/mod54_unit1/, 4 cells recorded, family mod54_mirrored_rows). Mechanism: A game currently gives one row oriented home minus away; the home-field offset and any orientation-specific noise are fitted together with team strength. Measured mechanism: each game enters twice (home view and sign-flipped away view) with the home-field offset as its own column, so the penalty sees orientation-free pairs. Fitted term: Base model refit on the mirrored table; compared on the same games with the same penalty selection.

## Tried
Unit 1: base is NOT antisymmetric (home_/away_ blocks, intercept 1.12, elo_diff carries +55 home, elo_home_win_prob). Mirrored rows (swap home/away, negate diff_*/rest_diff/elo_diff/spread_line, 1-p for elo_home_win_prob, home_view column +1/-1, alpha 10, 2x rows); base reproduced 1142-1089 and 802-701 first. Flips W-L first: 2011-19 225-229, 2020-25 122-119. Mirror 2011-19 1138-1093 (-0.18 [-2.20,1.73] P+ 0.43); 2020-25 805-698 (+0.20 [-0.81,1.23] P+ 0.61). Log loss 2020-25 0.6944 v 0.6969 base v 0.6933 market; Brier 0.2505 v 0.2517 v 0.2501. Train acc 0.545 v 0.560 base. Intercept control 2020-25 -3.06 [-3.82,-2.26], 2011-19 -0.31. 4 looks, unresolved_below_power.
Novelty check only. Nearest: MOD-23 units 9 and 16 (per-side targets, estimator class), unit 17 (block penalties). Differs: Nothing in registry/weak_signals or docs/lanes fits on mirrored rows; the identity check may show the base is already equivariant, which is a valid result.

## Next
Unit 2 candidate: alpha halved to match doubled rows; a served change needs a pinned margin.py refit path plus v8 enrollment.
First unit for one agent: Read the feature builder to confirm whether inputs are differences or home and away blocks, build the mirrored table in a scratch script, and report whether predictions equal the base when the table is already antisymmetric (identity check) before reporting any record difference.

## Open
Rules: Zero crossing closes nothing; one fitted probability selects the side; every penalty, weight and cut point chosen leave-one-season-out; every arm counted as a look; unresolved results go through nfl-ats weak-signals record with a closing ground; graded on the fair opener test (base 802-701 in 2020-25 and 1142-1089 in 2011-19 reproduced first, opener not a model input) on the same market and simple-model baselines. Pinned model source stays untouched: run in a scratch harness script, no src edit unless the result is adopted. Look count and family declared before any sign is seen. Data: Base input table and feature builder (local).

## Unit 2 (started Tue Nov 3 2026 6:18 PM)
Script scripts/mod54_unit2_penalty.py (arms A mirrored alpha 20, H base alpha 5; alphas follow from the doubled/halved row count). Running in background (task bxwa2ev2m), log C:\Users\Ryan\AppData\Local\Temp\u2.log, outputs artifacts/mod54_unit2/report.json and weak_signals_batch.json. Next: read the log, record cells with nfl-ats weak-signals record under family mod54_mirrored_rows, fill results here.
Unit 2 done Tue Nov 3 2026 6:21 PM; 4 cells recorded (unresolved_below_power). Base 1142-1089 / 802-701. Decisive W-L flips first.
A (mirror, alpha 20): 2011-19 1142-1089 flips 233-233, +0.00 [-2.28,2.03] P+ .50; 2020-25 815-688 flips 124-111, +0.86 [-0.26,2.07] P+ .90; LL .6941 v base .6969 v mkt .6933; Brier .2504 v .2517 v .2501; train acc .544.
H (base, alpha 5): 2011-19 1145-1086 flips 17-14, +0.13 [-0.31,0.54] P+ .70; 2020-25 788-715 flips 7-21, -0.93 [-1.59,-0.39] P+ 0.00; LL .6971; Brier .2518; train acc .559 (base .560).
Mirroring effect (A v base) +0.86 in 2020-25; penalty effect (H v base) -0.93 in 2020-25, so unit 1's +0.20 was mirroring gain masked by weaker penalty. Not closed: H is a penalty-direction probe, not a refuted served mechanism. Next: season stability of A, alpha chosen LOSO rather than doubled.
