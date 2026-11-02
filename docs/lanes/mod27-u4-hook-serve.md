# MOD-27 unit 4 - serve the MOD-26 pool hook term

## Goal
Serve the hook (pool half-point over a whole-number Tuesday market line) as a fifth fitted term, base re-read at the market line so the hook counts once.

## State
- Work parked on local branch `mod27-hook-serve` (commit d39b98d1, not pushed, not on master). Master and production unchanged: active pick_probability/20261101T171318Z, atlas loads, v6 source guard matches (CRLF restored on card_view.py and pick_probability.py after the branch checkout).
- On the branch: src/nfl_ats/pool_hook.py (push table, stacking, Tuesday market line = first snapshot at or after the pool's Tuesday-noon ET freeze via tuesday_noon_et_cutoff_utc; the agent's underived 6-hour lookback removed), pick_probability.py (loader accepts the hook term only with a push table), pick_probability_fit.py (stacked push-inclusive fit; per_game.parquet, bands and graded_games use one row per game: the up row, opener pushes dropped, 1,503 games; stacked_fit_rows kept), card_view.py (hook fail-open, cannot drop flags or move), independent_validation.py (hook in recompute, pool_hook.py pinned, DEFAULT_STUDY v7), v7 protocol.
- Measured: hook coef 0.808 (folds .77-.88), LOSO accuracy +0.098 pts P+ .585, Brier P+ .997, log loss P+ .996. Production loader dry run Week 8: 5 hook games, signs correct; CAR at GB and CLE at PIT flip to HOME from the refit (both coin flips). pytest 395 passed on the branch.
- BLOCKER (measured): `nfl-ats fit-pick-probability` fails its atlas rebuild: "Signal atlas did not reproduce saved out_of_season probabilities". signal_atlas.py refits the four-term stack from per_game.parquet (FIT_FEATURES) and must reproduce the served fit; it does not know the stacked population or the hook.

## Tried
- Pointer copy activation: loader rejected the fifth term, then the features list (fixed). Activated through the real command three times; each failed at the atlas and the pointer was rolled back each time.

## Next
- After the Week 9 Tuesday weekly_lock succeeds: on the branch, make signal_atlas.py reproduce the five-term stacked fit (or read the fit's saved out-of-season probabilities instead of refitting), then verify the whole chain on the branch: fit-pick-probability (activates + atlas), refresh-picks --dry-run, publish-board, pytest. Merge, activate, enroll v7 (start week = first week with no kicked-off game), restore CRLF if checkout rewrites pinned files.
- The weekly run calls fit-pick-probability, so nothing from the branch reaches master until the atlas passes.

## Open
- Strength bands now use the up row for whole-line games (away side holds the hook); a weighted version would use both.
