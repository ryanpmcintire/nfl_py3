# SIM-08 simulator rebuild to catch the model

## Goal

Make the play simulator (`scripts/sim04_engine.py`) at least match the served
discrete read at the opener, then test whether it adds anything. Done = a LOSO
2020-2025 re-grade where the market-anchored sim ties or beats the served read
on cover-vs-miss and push log loss, recorded with `weak-signals record`.

## State

- 2026-09-26: opened. SIM-04 closed with the engine still NO-GO on its own
  validation gate (2/5 key numbers). Its registry rows
  (`sim04_engine_team_conditioned_*`) refute this engine version only, not
  play simulation. Numbers: `docs/sim04_unit_log.md` lines 1806-1866.
- 2026-09-26 unit 1 (measured, this session): the task's cited finishing-split
  baseline (mass@3 .114) is stale (it was the pre-NN fix1+2 engine, log line
  1655). Rerun against the CURRENT committed engine
  (`scripts/sim04_engine.py`, unconditioned, same 768 real 2015-2017 REG
  games x 200 reps from their own Q4<=300s state): mass@3 .1331, mass@7
  .0845, SD ratio 1.015, OT reach rate .0622 (actual .0625, matched). Actual
  season-block bootstrap CI for mass@3 on these 768 games: mean .1525, 90%
  CI [.1380, .1667] -- sim sits just below the CI floor, a real but much
  smaller gap than the task assumed. Mechanism breakdown (sim vs actual,
  same script): 4th-down go rate trailing 1-3 .427/.420 (matched), leading
  kneel rate .172/.183 (matched), hurry-up pass rate trailing <=120s
  .775/.772 (matched), FG make rate by distance/score-state noisy on the
  actual side (n=2-57 per cell) with no consistent directional bug. The one
  clear, well-powered anomaly: OT outcomes. Sim OT games tie 21.3% of the
  time vs actual 4.2% (n=48 real OT games in this set), FG-win share 49.0%
  vs actual 60.4%; at the play level sim's OT touchdown rate is .0066 vs
  actual .0203 (n=934 real OT plays, 2015-2017) while OT FG-attempt/make
  rates already matched (.044/.047 attempt, .84/.80 make). Named mechanism:
  the down x phase KDTree for OT (phase 4) is built from OT rows only, the
  smallest, sparsest pool of all five phases, so red-zone OT snaps often
  can't find 40 true neighbors and pull in field-position-distant rows,
  suppressing the relative-gain-crosses-goal TD path unit 3's fix relies on.
- 2026-09-26 unit 1 fix (measured): `scripts/sim04_engine.py` -- added
  `phase_pool_mask` (~line 241) and used it in both `build_neighbor_index`
  (~line 253-260) and `build_neighbor_index_scipy` (~line 328-335) so the
  down x phase=4 (OT) tree draws from phase in {3, 4} (OT unioned with
  Q4<=300s) instead of phase==4 alone; every other phase's tree is
  unchanged. Rerun, same 768x200 split: sim OT TD rate rose .0066 -> .0174
  (actual .0203, much closer); OT tie rate fell 21.3% -> 10.3% (actual
  4.2%); OT FG-win share fell 49.0% -> 40.6% (actual 60.4%, moved the wrong
  way -- the reclaimed scoring mostly became OT touchdowns, not OT field
  goals). Net effect on the headline finishing split: mass@3 .1331 -> .1289
  (actual .1523, slightly worse, ~5 SE at n=153,600 so not noise), mass@7
  .0845 -> .0949 (actual .0911, closer), SD ratio 1.015 -> 1.016 (flat), OT
  reach rate .0622 -> .0631 (flat, matched either way). Verdict: the named
  OT-pool-sparsity mechanism was real and the fix corrected it in the
  predicted direction (TD rate, tie rate) but OT is only ~6% of games, and
  it did not close the aggregate mass-at-3 gap -- the larger driver is
  still open. Kept the fix (real, measured, targeted improvement to a named
  mechanism); not reverted. `unresolved_below_power`, no registry write per
  the task (mechanism check, not a pick decision).

## Tried (from SIM-04, do not redo)

- Seven engine looks on validation 2015-2017; best unit 3d: log loss +0.0045
  vs naive, mass at 3 .103 vs .152, at 7 .074 vs .091, OT rate from tied at
  5:00 31% vs 15%, FGs 2.84 vs 3.21/game, SD 8% high.
- Finishing split (real Q4 5:00 states to end): mass at 3 .114 vs .152, SD
  matched. Reaching split (margin at 5:00) is close at 3. So the 3s go missing
  in the last five minutes.
- Team conditioning: backup QB moves margin 40% of the rating gap (damped).
- LOSO grade with no market input: raw -0.034, re-centred -0.011 (all from
  pushes), side split dead heat at 78% agreement.

## Next

1. Endgame unit continued: the OT-pool fix (unit 1) did not close the
   aggregate mass-at-3 gap (.1289 vs .1523, actual 90% CI [.1380,.1667]).
   Next candidate mechanism, not yet measured: regulation-time (non-OT)
   dilution in the same down x phase pools -- check whether phase=3
   (Q4<=300s) itself is sparse enough to pull neighbors from phase=2
   (Q4>300s) or phase=0 in a way that mutes the true late-game FG-attempt
   rate, using the same play-level trace method as unit 1 (throwaway
   script, policy hook capture) restricted to non-OT plays. Diagnostic
   script (gitignored, reusable): `tests/scratch/sim08_unit1_diag.py`.
2. Undamp conditioning (kernel bandwidth h, k_state) and recheck SIM-05 QB run.
3. Anchor the sim on the opening spread and total; re-grade LOSO with 1000+
   draws; add a fitted-term blend with the served probability.

## Open

- None needing the owner.
