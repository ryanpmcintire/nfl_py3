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
- Latest run, `artifacts/sim04_engine/20260926T023148Z`: 2/5 hits (10, 17),
  log-loss delta +0.0092 (clears the <=0.02 bar for the first time). SD ratio
  1.056. Mass at 3 is .102 vs .152, at 7 .072 vs .091. Ties 1.8% vs ~0.4%.
  Tied-at-5:00 OT rate 52.4% vs 14.9%. Late-Q4 possession scoring rate 0.129
  vs 0.215. Still NO-GO (need >=4/5 hits).
- Fixes so far: TD PAT bonus; OT ending on a tying FG; game-final plays kept
  in the pool; halftime possession change (`docs/sim04_unit_log.md` "Engine
  unit", "Engine fix 1"); relative yards/distance delta on non-flip
  transitions instead of copying the drawn row's absolute next
  yardline/distance, and a clock-elapsed fix for the 309 training rows where
  the next row's game_seconds_remaining resets upward (mostly regulation-to-
  OT crossings) ("Engine unit 2" -- this fixed the log-loss/SD-ratio/mass-3-7
  numbers but did not move the tied-at-5:00 OT rate or late-Q4 scoring rate).
  A backoff-reorder fix (coarsen field position/distance before score/time at
  L1) was tried and measured WORSE on every endgame metric and was reverted;
  do not retry it without a different mechanism.
- Diagnostic split (`sim04_engine.simulate_from_states`, new function) run
  against the fixed engine: finishing error (from each real 2015-2017 game's
  own Q4<=300s state to the end, 200 reps x 768 games) is well calibrated --
  margin mean/SD 2.26/14.13 vs actual 2.20/13.87 (ratio 1.02), mass@3 .114 vs
  .152, mass@7 .075 vs .091. Reaching error (full-game sim's home margin at
  5:00 left vs actual, aggregate over all games) is also fairly close. This
  localizes the remaining gap to the tied-at-5:00 subgroup specifically (a
  small slice of all games, invisible in the aggregate reaching check), not
  to the general finishing mechanics.

## Tried

- Units 1-9 (drive-outcome chain plus patches) are abandoned. Their lesson:
  the shortfall at 3 comes from how games finish, not how they get close.
  2018-2025 has had 5 looks and stays off limits. The final grade is LOSO or
  2026 prospective.
- `game_features_pbp.parquet` filters on season only. Always filter
  `game_type=='REG'` for actual-margin sets.
- Engine unit 2: reordering the backoff hierarchy to coarsen field
  position/distance before score/time (meant to protect the tied-late
  urgency signal) measured worse on every endgame metric once field
  position/distance were already delta-based; reverted. Don't retry that
  specific reorder.

## Next

- Named mechanism, not yet implemented: the tied+late (qtr==4, gsr<=300,
  score_diff==0) corner has 320 L0 cells averaging 4.7 rows, 96.6% must back
  off, and real per-play scoring there is 0.077 vs 0.055 overall -- backoff
  dilutes exactly the signal that matters. Fix by pooling ALL small-margin
  (|score_diff|<=3) late-Q4 historical rows together, using the sim's own
  score_diff sign/magnitude only to pick offense-vs-defense aggression
  symmetrically, instead of coarsening any existing axis. Measure on the
  tied-at-5:00 OT rate (52.4% vs 14.9%) and late-Q4 scoring rate (0.129 vs
  0.215) specifically, then rerun full validation.
- Points/game still short (40.8 vs ~45); ~0.2 pts/game of that is confirmed
  structural (kickoff-return TDs, 76/15,503 kickoffs in train, impossible
  under the current `build_opening_pool` design) and deferred as small; the
  rest is presumably the same under-scoring mechanism above.

## Open

- `scripts/team_style_features.py` needs a `--refresh-raw` run (old schema
  cache, research-only).
