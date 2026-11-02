# MOD-27 unit 4 - serve the MOD-26 pool hook term

## Goal
Serve the hook (pool half-point over a whole-number Tuesday market line) as one fitted term without changing any pick where the hook cannot matter.

## State
- Branch mod27-hook-serve (worktree F:\Repos\nfl_py3_wt_hook) now holds the two-stage design. Stage 1 is the served four-term fit byte-for-byte (population, coefficients, per-game probabilities). Stage 2 fits only the hook coefficient (ridge 1e-3, no intercept) leave-one-season-out on stacked whole-number rows (pool lines M+0.5 and M-0.5, pushes kept, hook from prior-season push mass) with the fold's stage-1 logit as offset. hook_rows.parquet holds those rows; pool_hook is in coefficients.json; metadata features stay four; signal_atlas.py is unchanged from master (it reproduces stage 1).
- Measured in a copied artifacts root: gate passed (non-whole and all probabilities identical, 4 coefficients equal, 833-670 unchanged, hook coefficient 0.805, folds 0.77-0.87). Whole-number rows at simulated pool lines: log loss .68343 -> .68223, Brier .24520 -> .24459, probability_positive .994/.995; accuracy 726-596 -> 729-593 (185 changed picks).
- Earlier stacked five-term design rolled back 2026-11-02 (472 vs 487 on non-whole games).

## Tried
- Stacked five-term refit (regressed); two-stage (this commit).

## Next
- Orchestrator: activation (merge, fit in production artifacts, atlas, publish) and the weak-signals record for pool_hook_term_2026_10_27.
- Bundle the generalized offset term (lane mod27-u6-pool-offset.md) only if the same two-stage gate passes.

## Open
- Strength bands for whole-number games use one simulated pool line.
- Pinned files must keep v6/v7 enrolled bytes.
