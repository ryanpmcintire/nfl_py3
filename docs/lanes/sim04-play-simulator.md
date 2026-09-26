# SIM-04 play-by-play simulator

## Goal

Build the simulator in `docs/sim04_play_simulator_plan.md`. It exists for the
discrete margin distribution, what-if runs (SIM-05) and a dashboard view, not
for better side picks. Before any of its probabilities can enter the pick
probability, it must beat the served `DiscretePushReader` read on held-out
calibration at the opener, and even then it enters only as a fitted term. Owner
2026-09-25: finish it end to end, with no part-way updates. The follow-on lane
is `sim04-loso-eval.md` (team conditioning, LOSO grade, what-if runs, dashboard).

## State

- The engine is `scripts/sim04_engine.py`. It is self-contained and plays one
  snap at a time: each snap draws one real 2009-2014 REG play from situation
  cells (down, distance, field position, score, time, timeouts, with back-off)
  and copies its transition. Scoring and the clock come from the same plays.
  Validation is 2015-2017 REG. It takes about 15 s for 10k games.
- Bar (predeclared, same as unit 1b): at least 4 of 5 key-number hits (sim
  mass inside the actual bootstrap CI) and a log-loss delta vs the naive
  histogram of +0.02 or less.
- Latest run, `artifacts/sim04_engine/20260926T021517Z`: 1/5 hits, log loss
  +0.0207, SD ratio 1.08. Mass at 3 is .098 vs .152 and at 7 is .063 vs .091.
  Ties are 1.6% vs about 0.4%. The tied-at-5:00 OT rate is 49% vs 15%, and the
  late-Q4 possession scoring rate is 0.136 vs 0.215. NO-GO.
- Fixes so far: TD PAT bonus; OT ending on a tying FG; game-final plays kept
  in the pool (they were dropped, which removed every OT walk-off); halftime
  possession change. Details are in `docs/sim04_unit_log.md` under "Engine
  unit" and "Engine fix 1".

## Tried

- Units 1-9 (drive-outcome chain plus patches) are abandoned. Their lesson:
  the shortfall at 3 comes from how games finish, not how they get close.
  2018-2025 has had 5 looks and stays off limits. The final grade is LOSO or
  2026 prospective.
- `game_features_pbp.parquet` filters on season only. Always filter
  `game_type=='REG'` for actual-margin sets.

## Next

- Split the engine error into finishing vs reaching. Start the engine from
  the real 2015-2017 game states at 5:00 left in Q4 and simulate only the
  finish, then compare final margins to actual. Separately compare the sim's
  margin at 5:00 to the actual one. Fix whichever mechanism is named, in this
  file.

## Open

- `scripts/team_style_features.py` needs a `--refresh-raw` run (old schema
  cache, research-only).
