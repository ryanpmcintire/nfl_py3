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
