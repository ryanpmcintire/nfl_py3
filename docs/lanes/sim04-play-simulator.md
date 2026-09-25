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

- Unit 8 DONE, NO-GO (2026-09-25, `scripts/sim04_unit8_cells.py`, artifact
  `artifacts/sim04_unit8/20260925T212234Z/report.json`, full writeup in
  `docs/sim04_unit_log.md` "Unit 8 predeclaration"/"Unit 8 result"). Built
  the Q2/Q4 coarse-cell split + Q4-only/score-only backoff levels Unit 7b
  named, plus era-correct OT length (15 min through 2016, 10 min from
  2017) and an audit of the OT possession rule (no bug found there).
  **Fix verified working at the single-drive level but does not close the
  gap**: cell audit confirms the new Q4-only level pools tied+late-Q4
  drives correctly (n=91-165, scoring rate 0.24-0.35, close to actual
  ~0.355) with no more Q2 dilution, yet the tied-at-5:00 aggregate barely
  moved (sim OT rate 84-85% vs actual 34.7%, was 86.9% pre-fix; sim
  via-regulation margin-3 share 0.09-0.11 vs actual 0.531, unchanged from
  pre-fix 0.092). Validation hits **regressed** 2/5 (Unit 7 B: 7,10) to
  1/5 (7 only) across all 3 predeclared configs; selected config C
  (coarse-split only, no extra levels) per the predeclared tie-break, log
  loss best-in-series at +0.00812. OT era fix confirmed working
  independently: sim OT tie rate 14.8% -> 6.5-9.4%. Not a refuted
  mechanism (no sign flip on the Unit 7b fix itself) --
  `unresolved_below_power`; most of the tied-at-5:00 gap is not explained
  by the coarse-cell dilution after all.

- Unit 8b trace DONE (2026-09-25, `scripts/sim04_unit8b_trace.py`, copy of
  Unit 8 config C, artifact
  `artifacts/sim04_unit8b_trace/20260925T214746Z/report.json`, full
  writeup `docs/sim04_unit_log.md` "Unit 8b trace"). Found and fixed (in
  the copy only) the real bug behind the 84% vs 34.7% tied-at-5:00 OT-rate
  contradiction: `build_play_rows` (`scripts/sim04_unit7_clock.py:92-148`)
  rebuilds its `selected` play list per-drive, so every drive-ending play
  (punt/FG/score) that isn't literally the last play of the period gets
  `elapsed = gsr - boundary` -- the *entire remaining clock* -- instead of
  the true few seconds to the next snap (measured: terminal-row mean
  elapsed 267.8s vs non-terminal 28.3s at `sl_bucket=4`, tied bucket).
  `run_race` checks clock-expiry before `is_terminal`
  (`sim04_unit7_clock.py:268`), so these corrupted rows get misread as
  "clock expired," discarding real scores and starving the window of
  possessions. Ruled out the Q2/Q4-pooling hypothesis directly (elapsed/
  terminal-rate stats are nearly identical split by qtr). Fixed in the
  copy (`build_play_rows_fixed`, accumulates `selected` across the whole
  game before computing elapsed): possessions/game 1.63->3.04 (now matches
  actual's 3.04 almost exactly), OT rate 84.3%->66.5% (actual 34.7%),
  via-regulation margin-3 0.094->0.162 (actual 0.531), log-loss delta
  +0.00812->+0.00681. Gap roughly halved, not closed -- a second, unnamed
  mechanism remains (post-fix clock-expired share still 32.9% vs actual's
  15.4% "End of half"; per-possession scoring rate still half actual's).
  Not ported to `src/` or Unit 7/8 (this was a bounded copy-only debugging
  task).

- Unit 9 DONE, NO-GO (2026-09-25, `scripts/sim04_unit9.py`, artifact
  `artifacts/sim04_unit9/20260925T215733Z/report.json`, full writeup in
  `docs/sim04_unit_log.md` "Unit 9"). Built on Unit 8 config C, train
  2009-2014 / validate 2015-2017. Found and fixed two more clear bugs:
  (1) `sim04_unit8b_trace.py`'s whole-game elapsed fix mixes late-Q2 and
  late-Q4 rows (nothing selected in between), so the true last late-Q2
  play of a game reads a bogus cross-quarter gap as `elapsed` instead of
  falling back to the period boundary -- fixed with a same-quarter-only
  "next" search (`build_play_rows_qtr_safe`); (2) `run_race`
  (`sim04_unit7_clock.py:268`) checks clock-expiry *before* `is_terminal`,
  so a drawn scoring/terminal play whose `elapsed` reaches the remaining
  clock gets discarded as a bare expiry instead of crediting the score --
  fixed by checking `is_terminal` first (`run_race_fixed`). Audited
  `reconstruct_drives` for the same class of bug: no comparable defect
  found (period-ending drives are legitimately shorter, capped by the
  period boundary at their start; `_emit_drive` always uses the drive's
  own real last-play gsr, never a boundary fallback). Re-measured
  tied-at-5:00 validation: possessions/game 3.04->3.40 (actual 3.04,
  overshoots now), duration 64.7s->58.1s (actual 59.4s, improved),
  clock-expired/"End of half" share 32.9%->25.1% (actual 15.4%, ~40%
  of the gap closed), OT rate 66.5%->59.9% (actual 34.7%, ~20% of the
  gap closed), but **scoring rate per possession barely moved: 0.153->
  0.161 (actual 0.302)**, via-regulation margin-3 0.162->0.171 (actual
  0.531), key-number hits stayed 1/5 (only "10"), log-loss delta
  **worsened** +0.00681->+0.00968 (still under the 0.02 GO threshold on
  its own, but hits are the binding constraint). NO-GO: needs >=4/5 hits.
  Named mechanism for the near-zero movement in scoring rate: when the
  race says "not expired," the scored outcome is not the terminal play
  the race drew -- it is a second, independent draw from `draw_cell2`'s
  historical whole-drive pool (`sim04_unit8_cells.py:100-138`), keyed
  only by score/time/field-position and, under config C, missing the
  Q4-specific `level_q4`/`level_score` levels. The race's terminal signal
  and the drive's scored outcome are structurally decoupled, so Defect 2
  could not raise the drive-outcome pool's own scoring rate -- this
  matches Unit 8's finding that the Q4-specific level doesn't close the
  gap even firing cleanly at the cell level.

## Next

- Orchestrator decision 2026-09-25: stop patching the drive-chain hybrid
  (units 1b-9 are chained copies: 1b -> 7 -> 8 -> 8b -> 9, each patching the
  last). Their lasting value is diagnostic: the gap is late-game finishing;
  possessions and duration now match actual; the remaining defect is that
  scoring is drawn separately from the play sequence, so late possessions score
  at half the real rate (0.161 vs 0.302).
- Next unit: build the plan's Unit 3 as ONE clean module
  (`scripts/sim04_engine.py`), a play-level engine for the whole game. The state
  is (qtr, clock, score diff, possession, down, distance, yardline, timeouts);
  each play draws its type and result (yards, clock, turnover, penalty, score)
  from empirical cells with back-off, and down, distance and field position
  advance, so scoring and the clock come out of the same play sequence. Validate
  on train 2009-2014 / 2015-2017 (regular season) against units 1b and 9. Report
  the late-possession scoring rate, the tied-at-5:00 OT rate, the key-number
  table, log loss and sd. No 2018-2025 runs (it has had 5 looks); the final
  evaluation is leave-one-season-out or 2026 prospective.

## Open

- `scripts/team_style_features.py` has a cached raw parquet with the old
  schema; run it with `--refresh-raw` next time (research-only, not scheduled).
- `game_features_pbp.parquet`'s season-only filter silently includes
  playoff games in what Units 1b/1c/6 called "actual REG" 2015-2017/2018-
  2025 slices (Unit 1d found 33 such games in 2015-2017 alone); Unit 7 fixes
  this for its own actual sets only, not retroactively for 1b/1c/6.
