# POOL-09

## Goal
Hook over-pick against lattice value: is the field's half-point lean proportional to the discrete-margin value of the half point (ROADMAP POOL-09).

## State
Added 2026-11-03 (backlog expansion 7), not started, nothing run. Mechanism: POOL-08 measured the field over-picking the half-point side by +7.7 pp (perm p .029). If the lean tracks the probability the margin lands exactly on the whole number, the field is pricing it; if flat, it is a heuristic. Fitted term: lattice hook value as a covariate in the field-share model feeding the POOL-01 leverage term. Data: POOL-08 artifact, served lattice (local).

## Tried
Novelty check only. Nearest: POOL-08 (the lean), MOD-26 (hook term), UI-32 (key-number chip). Differs: explains the lean, not its existence.

## Next
First unit for one agent: Compute per-game push mass from the served discrete lattice, regress the POOL-08 residual on it with a within-week permutation null and report slope with interval.

## Open
Rules: Zero crossing closes nothing; one fitted probability selects the side; weights, cut points and alphas chosen leave-one-season-out; every cell counted as a look; unresolved results go through nfl-ats weak-signals record with a closing ground; graded at the opener (opener as model input, with intercept-only and line-move-only controls) on the same market and simple-model baselines. Look count and family declared before any sign is seen.

## Unit 1 (2026-11-03 17:32, family pool09_hook_value_slope)
scripts/pool09_hook_value_slope.py -> artifacts/pool_rank_replay/pool09_hook_value_slope.json. Effect = _pool_hook_effect from resolve_card_probabilities (served model, hook on vs off). Only W3-8 recorded forecasts carry the discrete base the served model needs (W1-2 raise), so n = 33 of 44 hook games. Served sign agrees with hook side 33/33. Effect mean 2.66 pp (0.42-4.21, sd 1.32). Not an outcome look; hook side covered 26-18 per POOL-08.
- POOL-08 residual (pp) on effect (pp): slope -1.27 [-5.07, 3.06] P+ 0.283; within-week perm two-sided p .51 (one-sided .65). Spearman -0.10 [-0.45, 0.25] P+ 0.278. Early (W3-5, n15) +0.06, late (W6-8) -3.11: unstable.
- Looks: 2 estimators on one question + 1 half split. Recorded 1 cell unresolved_below_power. Residual mean 5.3 pp in this subset.
- Reading: flat-to-slightly-negative; no sign the lean prices key-number value. Challenger only: discount hook-side leverage uniformly, not by value; no side rule.
## Next
Extend to W9+ and W1-2 only if discrete base is backfilled; test lattice push mass (not served effect) as covariate.
