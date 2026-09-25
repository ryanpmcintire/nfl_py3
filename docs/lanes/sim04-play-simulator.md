# SIM-04 play-by-play simulator

## Goal

Build the play-by-play game simulator in `docs/sim04_play_simulator_plan.md`.
Its case is the discrete margin distribution, counterfactuals (SIM-05) and a
dashboard view, not better side picks. It must beat the served
`DiscretePushReader` read on held-out margin calibration before any
probability it makes enters the pick probability, and then only as a fitted term.

## State

- Plan written; ROADMAP SIM-04 is in progress. Full unit history is in `docs/sim04_unit_log.md`.
- Unit 1 (`scripts/sim04_unit1_drive_chain.py`): field-position drive chain,
  train 2009-2017, test 2018-2025. 3/5 key numbers, log loss +0.046 vs the naive
  histogram. NO-GO.
- Unit 1b (`scripts/sim04_unit1b_state_chain.py`): plus score, time and OT.
  2/5, log loss +0.0175. NO-GO (joint bar).
- Unit 1c (`scripts/sim04_unit1c_diagnostic.py`): six configs on a 2015-2017
  validation split; none recover 7/14; the try-after-TD idea is refuted (PATs are
  already folded into drives). Of real 3-point games, 24% go to OT, and in 57% of
  the rest the final drive runs out the clock.
- Unit 6 (`scripts/sim04_unit6_endgame.py`) DONE, NO-GO: timeout-conditioned
  late-window resampling (4 configs, predeclared) did not beat Unit 1b's
  unmodified cascade on validation; config A won the predeclared selection.
  Test split (config A, 4th look at 2018-2025): 2/5 hits (10, 17), log-loss
  delta +0.01748 -- parity with Unit 1b, confirms the prior NO-GO. Full tables
  in `docs/sim04_unit_log.md` "Unit 6".
- Unit 2 DONE (root, 2026-09-25): `PBP_SNAPSHOT_COLUMNS` widened 45 -> 56
  (penalty team/type, play_type_nfl, four timeout columns, rusher, receiver
  and fumble player ids). Personnel/formation do not exist upstream. New
  snapshot `data/pbp/raw/20260925T202544Z` (2009-2025 plus postseason); timeouts,
  penalty type and play_type_nfl are fully covered every season. The rebuilt
  `game_features_pbp.parquet` matches except EPA features up to 0.00075,
  traced to an upstream nflverse revision of 2020 EPA (1,381 plays, 253
  games) carried forward by the EWM.
- Unit 1d diagnostic DONE (2026-09-25, `scripts/sim04_unit1d_divergence.py`,
  artifact `artifacts/sim04_unit1d/20260925T204442Z/report.json`, full tables
  in `docs/sim04_unit_log.md` "Unit 1d diagnostic"): decomposed the margin-3
  shortfall via a predeclared reaching-vs-finishing swap on the 5:00-left-Q4
  transition matrix. **Finishing explains ~87% of the gap, reaching only
  ~13%.** Named mechanism: no clock-expiration/kneel-down rule -- the sim
  gives a late drive its full drawn outcome instead of running out the
  clock, so real games' last drive is "End of half" 85.3% of the time (61.5%
  of margin-3 games) vs the sim's 46.6% (17.3% of margin-3 games); sim
  manufactures 3-point finals mostly via a live go-ahead FG instead (37.8%
  vs actual 23.9%). Dispersion does build gradually pre-Q4 too (sim/actual
  SD ratio 1.08 at end of Q1 rising to 1.15 at final margin) but is the
  smaller piece. Per-drive scoring-rate-by-state shape already matches
  between sim and actual (rules out missing marginal negative dependence).
  Data caveat found in passing: `game_features_pbp.parquet` filters only by
  `season`, so Units 1b/1c/6's "801 actual REG games" for 2015-2017 silently
  includes 33 playoff games; this unit used the clean 768 REG-only games and
  did not retroactively fix 1b/1c/6.

## Tried

- Outcome-pool conditioning (units 1, 1b, 1c) does not reproduce the 7/14
  pile-ups; 2018-2025 has been looked at 3 times. Future tuning uses the
  2015-2017 validation split, and the final comparison should use
  leave-one-season-out or 2026 prospective games, not 2018-2025 again.

## Next

- Unit 1d's decomposition (measured, see State) upgrades Unit 6's null
  result from qualitative to quantitative: build an explicit
  clock-expiration/kneel-down rule for the final drive of a half (checks
  remaining time vs a drive's normal duration, not another resampling-pool
  conditioning axis) -- this is Units 3-5's per-play territory. Escalate to
  the orchestrator: invest there next, with ~87%-of-gap justification, given
  four successive drive-level-conditioning-only attempts (1b, 1c, 6, 1d) all
  point at the same missing mechanism rather than a conditioning fix.

## Open

- `scripts/team_style_features.py` has a cached raw parquet with the old
  schema; run it with `--refresh-raw` next time (research-only, not scheduled).
- `game_features_pbp.parquet`'s season-only filter silently includes
  playoff games in what Units 1b/1c/6 called "actual REG" 2015-2017/2018-
  2025 slices (Unit 1d found 33 such games in 2015-2017 alone); not fixed
  retroactively, flagged for the orchestrator.
