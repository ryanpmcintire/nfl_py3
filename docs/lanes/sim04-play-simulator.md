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

- Unit 7 DONE, NO-GO (2026-09-25, `scripts/sim04_unit7_clock.py`): play-level
  clock-race model for the final drive(s) of each half. Predeclaration,
  configs, and both results (validation +
  `artifacts/sim04_unit7/20260925T205718Z/report.json`; test +
  `artifacts/sim04_unit7/20260925T210113Z/report.json`) are in
  `docs/sim04_unit_log.md` "Unit 7 predeclaration"/"Unit 7 result". REG-only
  actual-set fix (confirmed `game_type=='REG'` == raw-snapshot
  `season_type=='REG'`) used for every actual set here; on 2018-2025 it
  removes 100 playoff games (2227->2127) but shifts key-number mass only
  slightly (<=0.004 per number). Config B (no timeout conditioning) won
  validation (2/5 hits) and ran once on test 2018-2025 (5th look): **2/5
  hits (10, 17)** -- same count/numbers as Unit 1b/1c/6 -- log-loss delta
  +0.01610 (passes, better than Unit 1b's +0.0175), sd ratio 1.076 (down
  from Unit 1b's 1.090). The targeted mechanism itself is fixed:
  final-drive clock-expiration share is now **90.1%** (vs. actual 85.3%),
  up from Unit 1b's implicit ~46.6%. No sign flips (all misses are
  simulator-under-actual) so this is `unresolved_below_power`, not a
  refuted mechanism.

- Unit 7b diagnostic DONE (2026-09-25, `scripts/sim04_unit7b_scoring_mix.py`,
  artifact `artifacts/sim04_unit7b/20260925T211101Z/report.json`, full
  writeup in `docs/sim04_unit_log.md` "Unit 7b diagnostic"). Resolved the
  Unit 7 "Next" question the opposite way it was guessed: the clock-survives
  branch **under**-scores in the late window, not over-scores. Tied-at-5:00
  P(margin=3): actual 0.714 vs sim 0.573 (gap +0.141), and splitting by
  OT-vs-regulation shows the gap is 100%+ from the regulation (no-OT) path
  (+0.438, partly offset by -0.297 via-OT where sim already over-produces
  3s). Root cause is measured: sim's tied-at-5:00 OT rate is 0.869 vs actual
  0.347 (2.5x too high) because the final-5-minutes scoring rate is flat-out
  low (mean scoring plays after 5:00: 0.99 sim vs 1.33 actual; both FG and
  TD+PAT shares ~1.5-1.6x short, not a mix problem). Cell audit on 2009-2014
  train drives found the mechanism: `draw_cell`'s fine (tied, late-Q4,
  field-position) cell has n=2-49 (median ~10), almost always below
  `MIN_CELL_N=25`, so it falls back to a coarse pool that
  `time_bucket_coarse` merges with pre-halftime Q2 tied drives -- a lower-
  urgency population that dilutes the true endgame scoring rate.

## Next

- Build change named by Unit 7b: split the coarse-cell fallback so Q4
  endgame tied cells (`tb_fine` in {5,6}) never pool with Q2 pre-half tied
  cells (`tb_fine==2`) in `build_state_cells`/`draw_cell` (e.g. give
  `time_bucket_coarse` a separate bucket for {5,6}, or add a Q4-only
  intermediate level). Re-measure the tied-bucket OT rate and via-regulation
  margin-3 share on the same 2015-2017 split before touching OT resolution
  (secondary finding: sim OT ties 14.8% vs actual 0/17, and OT_SECONDS=600
  applied uniformly though 2015-2016 used the 15-min rule -- pre-existing,
  not introduced here). Escalate to the orchestrator for the next unit
  assignment (src/ build change vs. another diagnostic).

## Open

- `scripts/team_style_features.py` has a cached raw parquet with the old
  schema; run it with `--refresh-raw` next time (research-only, not scheduled).
- `game_features_pbp.parquet`'s season-only filter silently includes
  playoff games in what Units 1b/1c/6 called "actual REG" 2015-2017/2018-
  2025 slices (Unit 1d found 33 such games in 2015-2017 alone); Unit 7 fixes
  this for its own actual sets only, not retroactively for 1b/1c/6.
