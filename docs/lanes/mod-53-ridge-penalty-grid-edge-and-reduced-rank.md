# MOD-53

## Goal
Ridge penalty at the grid edge: extend the base penalty grid, report effective degrees of freedom, and test a reduced-rank base (ROADMAP MOD-53).

## State
Added 2026-11-03 (backlog expansion 9), not started, nothing run. Mechanism: MOD-23 units 16 and 17 chose the ridge penalty at the top of the grid (1000) in every fit from 2011 on, so the optimum may lie beyond the grid and the base may behave like a few-factor model. Measured mechanism: the held-out error curve over penalty and over the number of retained principal components, with effective degrees of freedom per fold. Fitted term: Base model refit with the penalty (or component count) chosen leave-one-season-out on a grid that brackets the minimum; no new input and no side rule.

## Tried
Novelty check only. Nearest: MOD-23 units 16 and 17 (chosen penalty at the grid edge, noted but not followed up), unit 9 (per-side alpha). Differs: No prior arm extends the penalty grid past 1000 or fits a reduced-rank base.

## Next
First unit for one agent: Extend the grid to 10^(-1..6) and to component counts 1..30, report the chosen penalty and effective degrees of freedom per outer fold, the held-out error curve with a bracketed minimum, and the 2011-19 and 2020-25 opener-test record, log loss and Brier against the base.

## Open
Rules: Zero crossing closes nothing; one fitted probability selects the side; every penalty, weight and cut point chosen leave-one-season-out; every arm counted as a look; unresolved results go through nfl-ats weak-signals record with a closing ground; graded on the fair opener test (base 802-701 in 2020-25 and 1142-1089 in 2011-19 reproduced first, opener not a model input) on the same market and simple-model baselines. Pinned model source stays untouched: run in a scratch harness script, no src edit unless the result is adopted. Look count and family declared before any sign is seen. Data: Base 88-input table and the unit 16 and 17 scripts (local).
