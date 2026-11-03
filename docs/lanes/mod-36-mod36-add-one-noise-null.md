# MOD-36 Add-one noise ensemble

## Goal
Add-one noise ensemble: run random matched features through the served add-one pipeline to set the noise floor for every candidate verdict ROADMAP row MOD-36.

## State
Added 2026-11-03 as an unmeasured backlog row; nothing run, no registry window spent. Mechanism: Add-one candidates on the 88-input model move accuracy by flipping a handful of coin-flip games, so a gain of a few tenths of a point may be the pipeline own noise. A negative control of random features with matched marginals shows the distribution of gain, log loss and flip count that noise alone produces. Data: Served-model add-one pipeline and the opener population 2020-2025; random columns permuted within season from real candidate marginals.

## Tried
Novelty check only: Nearest: positive-control-power.md and sim07-power-2026-11-03.md (planted real effects), leak_ceiling_control.md (leakage ceiling). Differs: a negative control for the add-one noise floor; grep for noise feature, random feature and placebo feature returned no hit.

## Next
First unit: Run 200 matched random features through the pipeline, store gain, log loss, flip count and decisive-game record, then place each completed MOD-23 candidate in that null distribution and report probability_positive of its excess over noise.
Rules: Zero crossing closes nothing; one fitted probability selects the side; weights, cut points and alphas chosen leave-one-season-out; every cell counted as a look; unresolved results go through nfl-ats weak-signals record with a closing ground; graded at the opener on the same market and simple-model baselines.

## Open
Look count and family to be declared before any sign is seen. Nothing here closes on zero crossing; record unresolved results with nfl-ats weak-signals record.
