# MKT-24

## Goal
Kickoff-slot opener residual: opener error by Thursday, Sunday night, Monday, early, late, holiday and international slot (ROADMAP MKT-24).

## State
Added 2026-11-03 (backlog expansion 5), unmeasured, nothing run. Mechanism: The opener may be priced the same across slots while scoring variance and rest vary; slot residuals shrunk toward zero could be a small term. Fitted term: shrunk slot effect on the opener margin residual.

## Tried
Novelty check only. Nearest: MKT-21, MOD-29 (kickoff-clock rest), POOL-05 (primetime as ownership covariate); grep 'international' found none. Differs: Slot as a market-error covariate, not rest or ownership.

## Next
First unit: Compute slot residual means 2013-2025 with season-block bootstrap, partial pooling, and a permutation null across slots; count slots as looks.

## Open
Data: nflverse schedule slot and location fields (free). Zero crossing closes nothing; one fitted probability selects the side; weights, cut points and alphas chosen leave-one-season-out; every cell counted as a look; unresolved results go through nfl-ats weak-signals record with a closing ground; graded at the opener (opener as model input, with intercept-only and line-move-only controls) on the same market and simple-model baselines. Look count and family declared before any sign is seen.
