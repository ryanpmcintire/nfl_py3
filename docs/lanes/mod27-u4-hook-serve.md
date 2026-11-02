# MOD-27 unit 4 - serve the MOD-26 pool hook term

## Goal
Serve the hook (pool half-point over a whole-number Tuesday market line) as a fifth fitted term in the pick probability, base re-read at the market line so the hook counts once.

## State
Code written, gates pass, NOT activated, committed, or enrolled. New artifact artifacts/pick_probability/20261102T150028Z (active pointer untouched: 20261101T171318Z).
Files: src/nfl_ats/pool_hook.py (new: push table, stacking, Tuesday market line from first quote snapshot after capture-6h, served hook frame), pick_probability.py (hook coefficient + push table optional in artifact; base converted to market line: pool x/(1-a) when pool=M+0.5, (x-a)/(1-a) when pool=M-0.5, a = pooled prior push mass within +-1 point), pick_probability_fit.py (push-inclusive stacked population, FITTED_FEATURES, FIT_FEATURES kept 4 for signal_atlas), card_view.py (serves hook), independent_validation.py (uses market-line base + hook in the recompute).
Measured (tests/scratch/hook_serve/fit_out.txt): 2,198 stacked rows = 1,537 games; hook coef 0.808 full fit, per fold .81 .876 .798 .77 .814 .782; LOSO accuracy +0.098 pts P+ .585, Brier +0.00026 P+ .997, log loss +0.00051 P+ .996 (identical to MOD-26 unit 1).
Week 8 dry run (week8_out.txt): 5 hook games, hook -0.19..+0.14 logit; 2 side changes, both non-hook games from the refit: CAR at GB AWAY .501 -> HOME .511, CLE at PIT AWAY .507 -> HOME .507; max prob move .037 (KC at DEN). Gates: ruff, ruff format, mypy clean on 5 touched src + pool_hook.py; pytest 391 passed (board, pick_refresh, cli, clv, publishing, discrete read, lineage).

## Tried
Base kept at pool line with hook added (double counts); rejected for the algebra above.

## Next (orchestrator)
1. Activation: write pointer for pick_probability/20261102T150028Z (fit with activate=True equivalent; copy artifacts/pick_probability/20261102T150028Z/activation.json to artifacts/active_pick_probability.json) then run refresh-picks --dry-run to confirm.
2. v7: copy registry/studies/combined_vs_raw_2026_v6.json to combined_vs_raw_2026_v7.json (new study_id, start_week 9); set DEFAULT_STUDY = "independent_validation_v7" (independent_validation.py:30) and the CLI protocol default; `nfl-ats independent-validation enroll --study independent_validation_v7 <protocol>`; then capture via the scheduler job independent_validation_sat_1210. Same session as activation. v6 stops (pinned edits already made).
3. Consider adding pool_hook.py to PINNED_NAMES (it changes served probability, name not matched by the guard).

## Open
Strength-band landing rates now computed on stacked pool-line rows unweighted (whole-line games count twice). Game-specific push mass uses the pooled empirical table, not the live lattice atom. Pre-edit _sources() hashes were not saved (post-edit saved: tests/scratch/hook_serve/sources_post_edit.json). No new weak-signals record (no new measurement).
