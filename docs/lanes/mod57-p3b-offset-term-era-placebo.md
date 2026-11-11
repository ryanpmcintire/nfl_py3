# MOD-57 P3-B - why the pool-line offset term loses in 2020-22

Stamped 2026-11-11 12:01.

## Goal
Name which 2020-22 difference (09:00 ET pre-noon snapshot, no move input in stage 1) turns the DraftKings-built offset term negative, by degrading 2023-25 inputs one at a time and together. No refit of 0.805. Family mod57_offset_term_era_placebo, 3 arms x {accuracy, log loss} = 6 looks (base is a replication, not recorded).

## State (measured; tests/scratch/mod57/p3b/run.py, run.out, cells.json, batch.json)
Snapshots: 2020-22 has one snapshot per week at 3.08 h before noon (about 09:00 ET). 2023-25 has hourly pre-noon quotes from 4.07 to 0.07 h before noon; arm A takes the quote nearest 3.08 h (18,186 quotes within 0.5 h). Post-noon 2023-25 line is the first quote at 0.93 h after noon. Stage 1 move: available 0% in 2020-22, 98% in 2023-25 (nonzero 65-72%). Arm B zeroes move and availability in 2023-25 and refits stage 1 leave-one-season-out (refit reproduces artifact oos probs, max diff 0.0).
Base replication: 2023-25 +2.02 pts [0.81, 3.27] P+ 1.00; 2020-22 -2.35 [-4.25, -0.57] P+ .006 (matches u1c). Coef .805 all-seasons, folds .68-.97.
2023-25 two-stage minus stage 1 (accuracy pts [interval] P+; log loss improvement P+; changed picks two-stage right / stage-1 right; fold coefs):
- A snapshot line: +1.37 [-0.31, 2.94] P+ .94; ll +.00211 P+ .987; changed 37.0 vs 26.0 of 63; coefs .79-1.33, all 1.03; gap -0.25 pts.
- B no move: +0.55 [-1.11, 2.34] P+ .72; ll +.00130 P+ .949; changed 45.0 vs 40.5 of 85.5; coefs .64-.94, all .78; gap 0.12.
- C both: +0.12 [-1.80, 1.95] P+ .56; ll +.00196 P+ .979; changed 43.5 vs 42.5 of 86; coefs .75-1.29, all 1.00; gap 0.62.
Base changed picks 37.0 vs 20.5 of 57.5. 2020-22 changed picks 29.0 vs 45.5 of 74.5 (base).

## Tried
One declared run, no variants. Verdict (inferred): the 2020-22 loss (-2.35, P+ .006) is NOT reproduced on 2023-25 by any degradation; the arms are not significantly different from the base gain or from zero. Removing move cuts the accuracy gain from +2.02 to +0.55 (+0.12 with the earlier line), so the move input accounts for most of the 2023-25 accuracy gain, but log loss stays positive (P+ .95-.99) in every arm. Remaining differences (inferred): era itself, half-point side split of source whole-number games (pool<anchor .59 in 2023-25 vs .48 in 2020-22, measured), whole-source share .43 vs .48, coefficient per fold drifts up to 1.0 under the snapshot line.

## Next
Test the half-point side split by era directly (does the pool shift toward the home side only in 2023-25); check whether the 2020-22 loss is a single-book (DraftKings) quality issue by running FD and the all-books median under the same arms.

## Open
Pool posting time and half-point shift rule unknown. Registry: 6 cells recorded unresolved_below_power in registry/weak_signals/mod57_offset_term_era_placebo.json; weak-signals pool (accuracy_points) loads.
