# MOD-57 P2-S where the served probability wins; offseason market re-rating (A0, A1)

## Goal
Part 1 descriptive record by week of the served probability, model alone and line-move side. Part 2 whether the market's own offseason re-rating or prior-season surprise carries over into early-season results. Family mod57_how_wrong_is_the_line (shared). Script scripts/mod57_p2s_offseason_adjustment.py, outputs artifacts/mod57_p2s/.

## State
2026-11-11 10:54:40 DECLARED before any outcome read. Novelty (read): grep of ROADMAP, docs, lanes, weak_signals. Offseason retention 0.67 is DEFAULT_OFFSEASON_RETENTION in src/nfl_ats/constants.py, a feature-builder regression of rate inputs (docs/offseason_retention.md); transaction_aware_preseason_prior and mod50_mod41_early grade feature priors and in-season terms. No document grades the market's own offseason re-rating against early results. Not a duplicate. Prior count in shard mod57_how_wrong_is_the_line is read from the registry before the run.
CONSTRUCTION DEVIATION (read data layout, no outcome): the declared first-three-games implied rating uses lines of weeks 2-3 that postdate a Week 1 game and embed Week 1-2 results, so it is not pregame for the games scored; the P2-Q weeks 1-3 fallback has the same flaw. Used instead: prior(s-1) = P2-Q line-only least squares on all of season s-1 closing lines, ratings r and home term h. For game g in season s, the market total adjustment A0 = (line_g - h) - (r_home - r_away), the game's own line minus what the prior-season ratings imply; per side it is the implied entering rating of that team given the opponent's prior rating, home minus away. The line is the closing line for closing grades, and the Tuesday opener (open_line, 2020+) for opener grades. Seasons 2010-25 (2009 has no prior).
- A1 = home minus away of each team's mean (margin minus close) over the whole prior season.
- Outcomes: real residual margin minus close, weeks 1-4 and 1-6; hindsight gap R_H minus close (leak, descriptive outcome only).
- Reads: slope with season-block bootstrap and probability_positive; walk-forward pick (sign of covariance of real residual on the quantity over all earlier seasons in the same window; test 2015-25 vs close, 2020-25 vs opener), decisive games (margin != line, quantity != 0), record first, by season. Opener grade uses A0 built from open_line, sign trained on the close-based quantity.
- Looks: 16 (2 quantities x 2 windows x slope real, slope hindsight, accuracy close, accuracy opener). Part 1 is one descriptive family of week and season cells, no fitting.
- Truncation: ten games, drop rows from g's date on; A0 and A1 must not change.

## Tried
2026-11-11 10:56:29 RUN DONE, 16 cells recorded (shard 102,338 to 123,149 bytes; prior count 76). Output artifacts/mod57_p2s_run.out, results.json, batch.json. Truncation diff 0.0 on 10 games. Part 1: served 40-48 week 1, 259-177 weeks 2-6 (by season 41-23, 38-30, 44-31, 45-31, 49-27, 42-35), 268-215 weeks 7-12, 262-234 weeks 13-18; strongest weeks 3, 4, 10. Part 2: slopes on real residual A0 -0.14 (P+ .10), A1 +0.08 (P+ .90) weeks 1-4; walk-forward accuracy 50.2 to 52.2% every cell, P+ .59-.89, season-to-season sign flips for A0 weeks 1-4. No tell found.

## Next
Weeks 3-4 and 10 carry the served 2-6 win; look at what those weeks share (bye, divisional schedule) as a mechanism, not a flip.

## Open
Opener A0 only exists 2020+ with 86-98% coverage.

