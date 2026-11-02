# MOD-27 unit 4 - serve the MOD-26 pool hook term

## Goal
Serve the hook (pool half-point over a whole-number Tuesday market line) as one fitted term without changing any pick where the hook cannot matter.

## State
- Branch mod27-hook-serve (worktree F:\Repos\nfl_py3_wt_hook) now holds the two-stage design. Stage 1 is the served four-term fit byte-for-byte (population, coefficients, per-game probabilities). Stage 2 fits only the hook coefficient (ridge 1e-3, no intercept) leave-one-season-out on stacked whole-number rows (pool lines M+0.5 and M-0.5, pushes kept, hook from prior-season push mass) with the fold's stage-1 logit as offset. hook_rows.parquet holds those rows; pool_hook is in coefficients.json; metadata features stay four; signal_atlas.py is unchanged from master (it reproduces stage 1).
- Measured in a copied artifacts root: gate passed (non-whole and all probabilities identical, 4 coefficients equal, 833-670 unchanged, hook coefficient 0.805, folds 0.77-0.87). Whole-number rows at simulated pool lines: log loss .68343 -> .68223, Brier .24520 -> .24459, probability_positive .994/.995; accuracy 726-596 -> 729-593 (185 changed picks).
- Earlier stacked five-term design rolled back 2026-11-02 (472 vs 487 on non-whole games).

- LIVE 2026-11-02 11:01 ET: branch merged; `nfl-ats fit-pick-probability` activated pick_probability/20261102T155750Z (four coefficients and all 1,503 per-game probabilities identical to 20261101T171318Z, opener 833-670, hook 0.805); atlas loads; refresh-picks dry run exit 0; pytest 452 passed; publish-board shows 833-670. Week 8 replay: the 9 non-hook games identical, 5 hook games moved 0.4-2.3 pts toward the hook side, no flips. Validation v7 enrolled (combined_vs_raw_2026_w09_w18_v7, 151 games, source_unchanged true); v6 stops at Week 8. Rollback: session scratchpad active_pick_probability.before_merge.json and active_signal_atlas.before_merge.json.

## Tried
- Stacked five-term refit (regressed); two-stage (this commit).

## Next
- Tuesday 14:47 ET check: Week 9 card built by weekly_lock with the hook term; v7 capture runs Sat/Sun.
- Bundle the generalized offset term (lane mod27-u6-pool-offset.md) only if the same two-stage gate passes.

## Open
- Strength bands for whole-number games use one simulated pool line.
- Pinned files must keep v6/v7 enrolled bytes.
- Review (lane hook-review-2026-11-02.md): no blocking findings. Leak-free re-measure holds: log loss gain .00123 [.00032, .00224] P+ .9966, Brier P+ .998, 727-595 vs 726-596; folds .74-.85. Decisions: the v7 capture pairs the enrolled coefficient with the live push table, which is derived from the fixed 2020-25 population, so Tuesday's check compares the new artifact's hook_push_mass with 20261102T155750Z; the late-hook risk is covered by the Tuesday 12:10 ET odds capture before the 14:20 lock.
