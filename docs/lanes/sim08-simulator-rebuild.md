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
- Unit 1 (measured + fix kept): finishing split (768 real 2015-2017 REG games
  x 200 reps from their own Q4<=300s state) on the then-current engine gave
  mass@3 .1331, mass@7 .0845, SD ratio 1.015, OT reach .0622 (actual mass@3
  .1525, 90% CI [.1380, .1667] on these 768 games -- sim just under the CI
  floor, a real but smaller gap than SIM-04's task framing assumed). Named
  mechanism: OT's down x phase KDTree pool (phase 4) drew only from the
  sparsest phase, missing the TD path. Fix: `phase_pool_mask`
  (`scripts/sim04_engine.py` ~line 241) makes the phase=4 tree draw from
  phase in {3, 4}. Result: OT TD rate .0066 -> .0174 (actual .0203), OT tie
  rate 21.3% -> 10.3% (actual 4.2%), but OT FG-win share moved the wrong way
  (49.0% -> 40.6%, actual 60.4%). Headline barely moved: mass@3 .1331 ->
  .1289 (actual .1523), mass@7 .0845 -> .0949, SD ratio 1.015 -> 1.016, OT
  reach .0622 -> .0631. Kept (real, measured, targeted; OT is only ~6% of
  games so it didn't close the aggregate gap). Full detail:
  `docs/sim04_unit_log.md` "SIM-08 unit 1" section.
- Unit 2 (measured; fix tried and reverted, no net engine change): built
  `tests/scratch/sim08_unit2_diag.py` (gitignored) to check five things on
  the same 768x200 non-OT finishing split: (1) real-unit KDTree neighbor
  spread by down x phase, (2) 4th-down FG-attempt rate by field position
  when tied/trailing 1-3, (3) red-zone (yardline<=20) drive TD:FG split for
  the last 5:00, (4) net point change per late possession, (5) the sim's
  outcome distribution restricted to the 117 real games whose actual final
  margin is exactly 3. Findings: neighbor spread (1) and the red-zone TD:FG
  split (3) already match actual closely -- not the driver. The dominant,
  well-powered gap is (2): actual teams kick a field goal 93-100% of the
  time on 4th down in makeable range (kick distance <50) when tied or
  trailing 1-3 with <=5:00 left (n=10-25/cell); the sim only does so 61-64%
  of the time trailing 1-3 (88-93% tied) at the same field-position buckets
  (n=1956-3002/cell) -- it draws punt/go-for-it far too often exactly where
  real coaches essentially never do. Consequence measured directly in (5):
  among the true-margin-3 games, the sim's own last scoring play is a field
  goal only 44% of the time vs a touchdown 51% of the time, while actual is
  65% field goal. Root cause read from the code: `scaled_score_diff`
  (`scripts/sim04_engine.py` line 204-210, `SCORE_INNER=8`,
  `SCORE_INNER_SCALE=2`) compresses a 0-8 point real score gap into only 0-4
  scaled units, so the measured mean neighbor score distance at down=4,
  phase=3 (4.2 raw points) is comparable to or wider than the 3-point
  trail_1_3 bucket itself -- field position and time dominate the KDTree
  distance budget over score exactly where the real decision is score-driven
  and near-deterministic. One fix tried (predeclared, `pick_index_nn`,
  `scripts/sim04_engine.py` ~line 270-296): kept the KDTree distances and
  replaced the uniform `rng.integers` neighbor pick with an adaptive Gaussian
  kernel weight (`bw` = median of the k=40 distances). Rerun (finishing
  split + all five measurements): moved nothing -- FG rate trail_1_3 <30
  .606->.599, 30-39 .629->.654 (wrong direction), tied 50+ .108->.110; true
  margin-3 sim mass@3 .3106->.3157 (+0.0051, ~1.7 SE at n=23,400, noise);
  headline mass@3 .1289->.1297, mass@7 .0949->.0956, SD ratio 1.016->1.015,
  OT reach .0631->.0629 (all flat). Diagnosis: the adaptive bandwidth only
  mildly discriminates near vs far neighbors (~7x nearest:farthest), not
  enough when the nearest 40 neighbors themselves are already off on score
  state because of the compression above. Verdict: fix did not move the
  named mechanism toward actual, so it was reverted (`git diff --stat
  scripts/sim04_engine.py` confirms no diff against the unit-1 committed
  state). Full-game validation of the current (unit-1-only) engine,
  `python scripts/sim04_engine.py` default `--n-games-per-season 3334`,
  2015-2017: key-number hits 2/5 (unchanged), log-loss delta +0.0044 vs
  naive (unit 3d was +0.0045, 2/5 -- same within noise, as expected since no
  net engine change was kept), mass@3 sim .0978 vs actual .1523 pooled, SD
  ratio 1.089. The FG-attempt-rate mechanism and its `scaled_score_diff`
  root cause remain open and unresolved -- not closed, not refuted (this was
  a refuted FIX, not a refuted signal; the underlying gap is still real and
  measured). No registry write (mechanism check, not a pick decision). Full
  detail: `docs/sim04_unit_log.md` "SIM-08 unit 2" section.

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
- Unit 1: OT-pool KDTree fix (kept). See State above.
- Unit 2: distance-weighted `pick_index_nn` resample (reverted, no effect).
  See State above.

- Unit 3 (measured; both attempts tried and reverted, no net engine change):
  Attempt A (predeclared: no compression inside 8 points for phase in
  {2,3,4} via a phase-conditional `SCORE_INNER_SCALE`) moved the trailing 1-3
  4th-down FG rate only .606/.629/.637 -> .628/.671/.641 (target >0.85, not
  reached) -- reverted, `git diff --stat scripts/sim04_engine.py` clean.
  Attempt B (predeclared: multinomial logistic go/FG/punt classifier fit on
  TRAIN_SEASONS-only 4th-down rows from score differential, seconds left,
  field position, distance, gating a play-type-filtered KDTree redraw) moved
  most trailing/tied makeable-range buckets to or above actual but broke the
  40-49 kick-distance bucket (tied .901->.530, trail .637->.443, both away
  from actual ~1.0/.96); full-game validation gave key-number hits 2/5 ->
  3/5 but log-loss delta +0.0044 -> +0.01156, over the task's +0.0094 keep
  bar -- reverted, `git diff --stat scripts/sim04_engine.py` clean (confirmed
  after manual reversal, `git checkout --` is blocked by ENG-31). Full
  detail: `docs/sim04_unit_log.md` "SIM-08 unit 3" section.

## Next

1. Unit 4 candidate (not yet measured): the down=4/phase=3 FG-rate gap is
   real (units 2-3 all confirm it) but every fix inside the existing KDTree
   metric or a linear decision layer either does nothing or trades one
   field-position bucket for another. Consider a decision layer with an
   explicit kick-distance feature (`fp_raw + 17`) and a non-linear model
   (small tree, not logistic) so the 40-49 bucket isn't forced through a
   single linear boundary shared with the other buckets; or accept the gap
   as `unresolved_below_power` and move to items 2-3.
2. Undamp conditioning (kernel bandwidth h, k_state) and recheck SIM-05 QB run.
3. Anchor the sim on the opening spread and total; re-grade LOSO with 1000+
   draws; add a fitted-term blend with the served probability.

## Open

- None needing the owner.
