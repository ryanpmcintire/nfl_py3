# MOD-27 unit 4 - serve the MOD-26 pool hook term

## Goal
Serve the hook (pool half-point over a whole-number Tuesday market line) as one fitted term without changing any pick where the hook cannot matter.

## State
- Branch mod27-hook-serve (worktree F:\Repos\nfl_py3_wt_hook): d39b98d1 (hook term, loader, card, validation v7 protocol) and a5ce3a1e (signal atlas reproduces the stacked five-term fit from fit_population.parquet). Production is on master's four-term fit pick_probability/20261101T171318Z; v6 source guard matches; site shows 833-670.
- 2026-11-02 merged and activated (pick_probability/20261102T154016Z): fit, atlas, refresh dry run, pytest 395 and publish all passed. Rolled back the same hour (merge aborted, pointers restored, site republished) because of a measured regression:
  - Opener view, 1,503 games: new 819-684 vs served 833-670; hook term zeroed 824.
  - Non-whole-number lines (hook is 0, 876 games): new 472 vs served 487; changed picks 34-49 (P(new better) .051). Whole-number lines (627): 347 vs 346.
  - Cause (inferred from coefficients): the stacked push-inclusive fit counts each whole-number game twice and refits every term (model_logit .309 -> .249, flag_sum .319 -> .419, move_available .017 -> .083, intercept -.012 -> -.031). MOD-26 compared against a four-term model refit on the same stacked rows, never against the served fit.

## Tried
- Pointer-copy activation (loader fixes); three atlas failures before a5ce3a1e; one full activation rolled back for the regression above.

## Next
- Two-stage design on the branch: keep the served four-term fit exactly (same population, coefficients and per-game probabilities), then fit only the hook coefficient leave-one-season-out on the stacked whole-number rows with the served logit as an offset. Gate: on non-whole-number games every probability equals the served one; report the hook coefficient per fold and the whole-number record at simulated pool lines against the served fit. Atlas: four-term reproduction unchanged, hook fixed as an offset.
- Bundle the generalized offset term (lane mod27-u6-pool-offset.md, coefficient about 0.88) only if the same two-stage gate passes.

## Open
- Strength bands for whole-number games use one simulated pool line; a weighted version would use both.
- Pinned files must keep the bytes v6/v7 enroll with; a checkout that rewrites line endings breaks the guard.
