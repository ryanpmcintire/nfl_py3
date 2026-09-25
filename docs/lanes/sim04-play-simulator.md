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

## Tried

- Outcome-pool conditioning (units 1, 1b, 1c) does not reproduce the 7/14
  pile-ups; 2018-2025 has been looked at 3 times. Future tuning uses the
  2015-2017 validation split, and the final comparison should use
  leave-one-season-out or 2026 prospective games, not 2018-2025 again.

## Next

- Unit 6 is done (see State). Its null result plus Unit 1c's decomposition
  point at a near-deterministic clock-kill rule (not probability-weighted
  resampling) as the remaining lever, which needs a per-play loop -- closer
  to Units 3-5 than to more drive-level state-cell conditioning. Escalate to
  the orchestrator: invest in Units 3-5 (per-play submodels) next, given
  three successive drive-level conditioning attempts (1b, 1c, 6) have all
  failed to move the margin-3/7/14 gap.

## Open

- `scripts/team_style_features.py` has a cached raw parquet with the old
  schema; run it with `--refresh-raw` next time (research-only, not scheduled).
