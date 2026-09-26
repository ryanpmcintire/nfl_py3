# SIM-08 simulator rebuild to catch the model

## Goal

Make the play simulator (`scripts/sim04_engine.py`) at least match the served
discrete read at the opener, then test whether it adds anything. Done = a LOSO
2020-2025 re-grade where the market-anchored sim ties or beats the served read
on cover-vs-miss and push log loss, recorded with `weak-signals record`.

## State

- SIM-04 was closed while the engine failed its own gate (2/5 key numbers);
  its registry rows refute that engine version only.
- Unit 1 (committed `69fe2f1`): OT neighbour pool includes phase 3. OT TDs per
  play .007 to .017 (actual .020), OT ties 21% to 10% (4%).
- Unit 4 (committed with this lane update): late-phase 4th-down layer,
  HistGradientBoosting on score, seconds, distance, kick distance, fit on
  2009-2014 (LOSO accuracy .903, in-sample .919, majority .642), redraws from
  neighbours of the chosen play type in phases 1, 3, 4 only. Full-game
  validation 2015-2017 (`artifacts/sim04_engine/20260926T132734Z`): **4/5 key
  numbers, log loss +0.0080 vs naive** (first pass of the predeclared GO gate:
  >=4/5 and <=+0.02). Mass at 3 still .102 vs .152; SD ratio 1.10.
- The validation split has now had about 11 engine looks; the LOSO 2020-2025
  grade is the untouched test.
- Full numbers: `docs/sim04_unit_log.md`, sections "SIM-08 unit 1" to "unit 4".

## Tried

- Unit 2: Gaussian neighbour weighting, no movement, reverted. Named the gap:
  trailing 1-3 late, real teams kick 93-100%, sim 61-64%.
- Unit 3: late score-scale tightening (.61 to .63, reverted); logistic
  4th-down layer on all phases (3/5, but 40-49 yd kicks halved and log loss
  +0.0116, reverted).

## Next

1. Generalize the 4th-down layer: `build_tables` fits it only when
   `seasons == TRAIN_SEASONS and not condition_on_team`, so the LOSO grade
   (tables 2009..S-1, team-conditioned) would run without it. Fit it on
   whatever seasons the tables use, in both paths.
2. Undamp team conditioning (kernel bandwidth h, k_state); recheck the SIM-05
   backup-QB run (currently 40% of the rating gap).
3. Remaining shape gaps: mass at 3 (.102 vs .152) and SD ratio 1.10.
4. Anchor on the opening spread and total; re-grade LOSO 2020-2025 with 1000+
   draws via `scripts/sim04_loso.py`; add a fitted-term blend with the served
   probability; record with `weak-signals record`.

## Open

- None needing the owner.
