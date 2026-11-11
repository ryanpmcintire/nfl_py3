# P3-G returning offensive snap share as a fifth fitted term (2026-11-11 13:51)

## Goal
Does last season's returning snap share (home minus away, from artifacts/mod57_p3c/team_change_features.parquet, main columns only) add inside the served four-term stack in weeks 2-6, leave-one-season-out at the opener 2020-25. Family mod57_returning_share_stack_term, 6 cells.

## State
Decision implication (inferred): neither serve nor hold as a candidate beside T. Held-out it did not add; nothing on the 2026 card changes (term is zero outside weeks 2-6). Closed only as unresolved_below_power, no closing ground.
Measured (artifacts/mod57_p3g/run.out, results.json, script scripts/mod57_p3g_returning_share_term.py), weeks 2-6, 436 games, four-term 259-177 (59.4%):
- A (off): 251-185; changed 48 picks, new 20 vs base 28; accuracy -1.85 pts [-5.10, +1.58] P+ 0.127; log loss worse 0.0026 (improvement P+ 0.149). In-sample 58.9%, gap +1.38. Fold coefficients +,+,+,-,+,+ (2023 negative).
- B (off+def): 259-177; changed 98, 49 vs 49; accuracy -0.01 [-4.43, +4.59] P+ 0.470; log loss improvement P+ 0.232. Def coefficient positive in 6 of 6, off sign +,-,+,-,+,+.
- C (A + T): 248-188; changed 93, 41 vs 52; accuracy -2.50 [-6.05, +0.95] P+ 0.075; log loss improvement +0.0007 P+ 0.552. T coefficient negative in all 6, A positive 5 of 6 (2023 -0.008).
- Placebo, same term weeks 7-12 (all games): 825-678 vs 829-674, log loss improvement P+ 0.627. Intercept-shift only: 249-187, accuracy -2.27 P+ 0.083 (equals the A loss, so A's loss is mostly the fifth-term parameter cost, inferred).
- Permutation (500, within season, weeks 2-6): real log-loss gain -0.0026 sits at 25.2th percentile from the top (74.8% of permutations did as well or better); accuracy 67% at or above real.
- All 1,503 games: A 823-680 vs 829-674, accuracy -0.40 P+ 0.198.
- Frozen 2026 (read: raw_model_home_prob_final replay of current structure, tests/scratch/fixpicks/root_m2_games.csv, 77 weeks 2-6 games of weeks 1-9, coefficients fitted 2020-25 with proper (x-mean)/std): current 50-27; A 47-30 (changed 5, 1 vs 4; accuracy -3.9 [-6.3, -1.3]); B 41-36 (changed 13, 2 vs 11); C 50-27 (changed 18, 9 vs 9).
Cells recorded: 6, shard registry/weak_signals/mod57_returning_share_stack_term.json; `pool --league nfl --effect-units accuracy_points` exit 0.

## Tried
Arms A, B, C, placebo, intercept shift, permutation, 2026 frozen. The earlier standalone lead (9 of 11 folds positive against the close) did not carry into the stack: the served probability already holds the information (inferred).

## Next
Nothing from this line unless the base model drops its roster information; re-test only in a new season window (2026 full weeks 2-6, 2027).

## Open
Mean-offset: 2026 contribution uses (x-mean)/std, unlike P2-R which omitted the mean. Cell population is weeks 2-6 (the all-games delta is in results.json, not recorded).
