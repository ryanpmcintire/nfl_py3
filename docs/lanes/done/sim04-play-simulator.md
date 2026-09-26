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

- CLOSED 2026-09-26. Final results are in `docs/sim04_unit_log.md` ("Unit LOSO grade result", "SIM-05 and dashboard") and ROADMAP SIM-04/SIM-05. Anything below is history.

- The engine is `scripts/sim04_engine.py`. Each snap draws one real 2009-2014
  REG play. As of Engine unit 3 the draw is a k=40 nearest-neighbour pick from
  a per-(down, phase) `sklearn.neighbors.KDTree` on continuous state
  (`build_neighbor_index`/`pick_index_nn`, ~line 236/259) with neighbour sets
  memoized on a rounded key (`round_state_key`, ~line 222) -- NOT the old
  categorical l0-l3 back-off (removed; `build_levels`/`pick_index` no longer
  exist). Validation is train 2009-2014 / valid 2015-2017 REG
  (`TRAIN_SEASONS`/`VALID_SEASONS` in the file). Full validation
  (`--n-games-per-season 3334`, 10,002 games) now takes ~2 minutes.
- Bar (predeclared, unchanged since unit 1b): >=4/5 key-number hits (sim mass
  inside the actual bootstrap CI) AND log-loss delta vs the naive histogram
  <=+0.02.
- **Engine unit 3 result (measured, artifact
  `artifacts/sim04_engine/20260926T024735Z/report.json`, full predeclaration
  and result narrative in `docs/sim04_unit_log.md` "Engine unit 3" -- that
  section's predeclaration is written but its "### Result" body was NOT
  appended before the tool-call cap hit; the numbers below are the source of
  truth, transcribe them into that header verbatim):**
  Fixed two named bugs first (both in `scripts/sim04_engine.py`): (A)
  `run_one_game`'s non-flip branch (~line 686) now scores a TD when the
  relative-delta yardline crosses the goal instead of clamping at the 1,
  drawing a PAT/2pt bonus from `tables["pat_bonus_pool"]` and a receiving
  field position from `tables["post_score_pool"]` (both built in
  `build_tables`, ~line 456/467); (B) `build_transition_frame` (~line
  350-351) now credits punt-return/blocked-kick TDs (`return_score`: TD row,
  not INT/fumble, `posteam_score_post==posteam_score`) to `points_def` with
  the PAT bonus, matching how INT/fumble-return TDs already worked, instead
  of losing the 6 into a zero-delta `points_off`.
  Backoff-level usage measured (`scripts/sim04_backoff_diag.py`, old l0-l3
  mechanism, post A/B fix, one 3,000-game run): overall l0 67.2% / l1 25.3% /
  l2 6.4% / l3 1.1% (n=433,470). Q4<=300s: l0 27.9% / l1 29.5% / l2 36.4% /
  l3 6.3% (n=47,186) -- 72% must leave the exact cell. Q2<=120s: l0 44.7% /
  l1 20.5% / l2 30.8% / l3 3.9% (n=28,396).
  Full validation with the NN replacement (2/5 hits -- 7, 10; changed
  composition from unit 2's 10, 17):

  | number | sim mass | actual pooled | actual 90% CI | in CI? |
  |---|---|---|---|---|
  | 3 | 0.0936 | 0.1523 | [0.1380, 0.1667] | no |
  | 7 | 0.0785 | 0.0911 | [0.0755, 0.1068] | yes |
  | 10 | 0.0548 | 0.0482 | [0.0391, 0.0573] | yes |
  | 14 | 0.0440 | 0.0508 | [0.0456, 0.0560] | no (just below) |
  | 17 | 0.0400 | 0.0299 | [0.0208, 0.0391] | no (just above) |

  Log-loss delta **+0.01302** (within the +0.02 bound but worse than unit 2's
  +0.0092). Margin SD ratio **1.105** (worse than unit 2's 1.056; sim sd
  15.34 vs actual 13.87). Tie rate **0.96%** vs actual ~0.4% (better than
  unit 2's 1.8%). **Tied-at-5:00 OT rate 30.3% vs actual 14.9%** (big
  improvement from unit 2's 52.4%, driven almost entirely by fix A -- a
  small-N smoke test of fix A alone, before the NN replacement, already
  showed this rate at ~15%). **Late-Q4 possession scoring rate 0.1432 vs
  actual 0.2149** (improved from unit 2's 0.129, still low). **Possessions
  per game: sim 23.17 vs actual (measured this run, real 2015-2017 REG
  pbp) 23.06 -- now matched** (previously 21.1 vs a ~22.4 pool target).
  **Plays per game: sim 145.09 vs actual 158.51** -- gap widened in relative
  terms now that possessions match: plays-per-possession sim 6.26 vs actual
  6.87, a new, more specific finding (drives resolve in fewer snaps on
  average even though drive *count* is now right). Points/game sim 42.92 (up
  from ~40.4-40.8, closer to the ~43.8 target). FG/offensive-TD/defensive-TD
  per-game counts were NOT obtained this run -- `run_one_game` was mid-edit to
  add `fg_count`/`off_td_count`/`def_td_count` local counters (declared at
  the top of the function, ~line 559) when the tool-call cap hit; the
  increments (after the points-apply block, ~line 616, and inside the fix-A
  overshoot branch, ~line 694) and the record-dict/`summarize_sim` wiring
  were never added. These three unused locals are harmless dead code as
  left (no syntax/runtime break) but should be finished or removed.
  **GO/NO-GO: NO_GO** (2/5 hits, need 4; log-loss and possessions both
  improved or held, margin-3 mass and SD ratio both moved slightly the wrong
  way). Not a refuted mechanism (no interval flipped to the wrong side of a
  control) -- `unresolved_below_power` for "bug fixes + NN backoff" jointly.
- Fixes prior to unit 3 (unit 2 and earlier): TD PAT bonus; OT ending on a
  tying FG; game-final plays kept in the pool; halftime possession change;
  relative yards/distance delta on non-flip transitions; clock-elapsed fix
  for the 309 rows where next-row gsr resets upward. A backoff-reorder fix
  (unit 2, coarsen fp/distance before score/time at L1) measured worse and
  was reverted -- moot now that the categorical back-off itself is gone.
- Diagnostic split (`sim04_engine.simulate_from_states`) from unit 2, before
  the unit-3 fixes: finishing error from each real game's own Q4<=300s state
  was well calibrated (mass@3 .114 vs .152) while full-game reaching error
  was not, localizing the old gap to the tied-at-5:00 subgroup specifically.
  Superseded in practice by unit 3's direct fix (OT rate 52%->30%) but the
  original diagnostic method is still valid if needed again.

## Tried

- Units 1-9 (drive-outcome chain plus patches) are abandoned. 2018-2025 has
  had 5 looks and stays off limits. The final grade is LOSO or 2026
  prospective.
- `game_features_pbp.parquet` filters on season only. Always filter
  `game_type=='REG'` for actual-margin sets.
- Engine unit 2's backoff-reorder fix: reverted, see above; now moot.
- Engine unit 3: NN backoff replacement (k=40, KDTree per down x phase, see
  `docs/sim04_unit_log.md` "Engine unit 3" for the full scale/feature
  predeclaration) plus TD-overshoot (A) and return-TD (B) bug fixes. Result
  above: fixed A/B cleanly (OT rate, possessions/game, tie rate, points/game
  all moved toward actual); the NN replacement itself did not clear the
  hit-count bar and slightly worsened log-loss delta and SD ratio relative to
  the old categorical mechanism's last measurement (unit 2). Do not re-try
  the NN replacement with the same k/scales expecting a different result --
  if revisited, change a named, measured input (see Next).

## Next

1. Finish or revert the fg_count/off_td_count/def_td_count instrumentation
   in `run_one_game` (see State) and rerun a ~3,000-game `simulate()` call
   against `build_tables(TRAIN_SEASONS)` (not a full validation look) to get
   the FG/offensive-TD/defensive-TD per-game tallies against the 2009-2014
   pool, matching the orchestrator's original comparison format (plays 145.4
   vs 150.2, possessions 21.1 vs 22.4, FGs 2.79 vs 3.21, offensive TDs 4.29
   vs 4.56, points 40.4 vs 43.8) so the FG/TD gap can finally be quantified
   directly instead of inferred from points/game.
2. Named next mechanism (not yet implemented): plays-per-possession is now
   the cleanest lead -- sim drives resolve in 6.26 plays vs actual 6.87 even
   though drive *count* per game now matches almost exactly (23.17 vs 23.06).
   Measure the real vs. simulated down-progression distribution (P(reach 2nd
   down | 1st down attempt), etc.) conditional on down alone, to check
   whether the k=40 NN pool at a given (down, phase) is pulling in enough
   distance-similar neighbours or diluting the down-to-down persistence rate
   with neighbours from a wider ydstogo range than real drives see.
3. Second candidate: `post_score_pool` (post-score kickoff field position)
   currently pools ALL scoring types (offensive TD, defensive/return TD, FG)
   together undifferentiated; a made FG's ensuing kickoff and a pick-six's
   ensuing kickoff have different real starting-field-position distributions
   only if fake-punt/onside-kick rates differ by scoring type context (likely
   small) -- lower priority than #2, but cheap to split and measure if #2
   doesn't move the margin-3/14/17 numbers.
4. Per the task's "at most one additional predeclared config" allowance, only
   ONE of #2/#3 (or a jointly-predeclared combination) may be tried on the
   2015-2017 validation split before the next full test-style report; predeclare
   before running.
5. Margin-3 mass moved the wrong way this unit (0.102 -> 0.094 vs actual
   .152) even though fix A should have helped near the goal line -- worth
   checking whether more OT games now ending in sudden-death TDs (fix A
   always ends OT immediately on any TD) redistribute some historical
   OT-field-goal-decided 3-point margins into other margins; not yet
   measured.

## Open

- `scripts/team_style_features.py` needs a `--refresh-raw` run (old schema
  cache, research-only).
- `docs/sim04_unit_log.md` "Engine unit 3" section has its predeclaration
  written but is missing the "### Result" body (numbers are in this lane's
  State section above) -- append verbatim before this lane's next unit
  starts, so the unit log stays the source of truth for verdicts.
