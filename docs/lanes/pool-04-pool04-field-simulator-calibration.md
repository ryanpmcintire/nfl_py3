# POOL-04 Field simulator calibration

## Goal
Field simulator calibration: score the pool-finish simulation against the observed 250-entry weekly score distribution ROADMAP row POOL-04.

## State
Added 2026-11-03 as an unmeasured backlog row; nothing run, no registry window spent. Mechanism: The chance of finishing first shown to the owner comes from a simulated field. POOL-02 showed entrant spread equals binomial spread on average, but the simulator upper tail and cross-entrant correlation through shared public sides were never checked against the observed weeks. Data: data/pool_observables standings and picks W1-8, simulate_pool_finish field model, market-implied shares.

## Tried
Novelty check only: Nearest: POOL-01 and pool01-standing-aware-2026-11-02.md (the simulation itself), POOL-02 (persistence only, no distribution check), UI-21 (displays the output). Differs: validates the field model that feeds the displayed first-place chance.

## Next
First unit: Posterior predictive check: per-week mean, standard deviation, top-decile score and share tied at the maximum, observed against simulated with intervals; report each mismatch as measured and let the field model inputs change only through named mechanisms.
Rules: Zero crossing closes nothing; one fitted probability selects the side; weights, cut points and alphas chosen leave-one-season-out; every cell counted as a look; unresolved results go through nfl-ats weak-signals record with a closing ground; graded at the opener on the same market and simple-model baselines.

## Open
Look count and family to be declared before any sign is seen. Nothing here closes on zero crossing; record unresolved results with nfl-ats weak-signals record.
