# POOL-07

## Goal
Standings-conditional variance policy: how far ahead or behind we sit changes which picks to take (ROADMAP POOL-07).

## State
Added 2026-11-03 (backlog expansion 5), unmeasured, nothing run. Mechanism: A leader should avoid contrarian coin flips; a trailer needs them. Fitted term: a leverage coefficient on the simulated finish-probability slope, fitted leave-one-week-out on the pool simulator.

## Tried
Novelty check only. Nearest: POOL-01, POOL-03 (coin-flip-band leverage), UI-21 (outlook panel); grep found no standings-conditional policy. Differs: Policy depends on standing, not game ownership alone.

## Next
First unit: Using POOL-04's calibrated simulator, compute the finish-probability gain of each pick as standings vary and test whether the served card would change.

## Open
Data: Pool standings and field simulator (captured). Zero crossing closes nothing; one fitted probability selects the side; weights, cut points and alphas chosen leave-one-season-out; every cell counted as a look; unresolved results go through nfl-ats weak-signals record with a closing ground; graded at the opener (opener as model input, with intercept-only and line-move-only controls) on the same market and simple-model baselines. Look count and family declared before any sign is seen.
