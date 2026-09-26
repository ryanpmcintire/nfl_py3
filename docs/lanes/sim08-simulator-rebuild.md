# SIM-08 simulator rebuild to catch the model

## Goal

Make the play simulator (`scripts/sim04_engine.py`) at least match the served
discrete read at the opener, then test whether it adds anything. Done = a LOSO
2020-2025 re-grade where the market-anchored sim ties or beats the served read
on cover-vs-miss and push log loss, recorded with `weak-signals record`.

## State

- SIM-04 was closed while its engine failed its own gate (2/5 key numbers);
  those registry rows refute that engine version only.
- Engine now (validation 2015-2017, `artifacts/sim04_engine/20260926T161032Z`):
  4/5 key numbers, log loss +0.0024 vs naive, SD ratio 1.10. Built from:
  OT neighbour pool (unit 1, `69fe2f1`); late boosted-tree 4th-down layer
  (unit 4, `0014146`), fitted on whatever seasons the tables use (unit 5,
  `0324ecd`); team yard shift 0.64 yd per unit of rating gap (unit 6,
  `65cd15e`: home edge +1.89, backup QB -1.37); `SCALE_FP` 2.5 stops
  goal-line plays borrowing open-field yardage (`3cc779e`); about 7x faster
  with identical single-process output (`0ed7c9c`).
- `scripts/sim04_loso.py` grades raw, old shift, spike-keeping tilt and a LOSO
  blend with the served cover probability (`a91cbfe`); `--n-reps`/`--workers`
  (`d9e42e8`). Detail for every unit: `docs/sim04_unit_log.md`.
- Early re-grades died on 2026-09-26 from memory (22 workers, then an
  unbounded per-game weight cache, fixed in `f62011e`). Workers are capped at
  min(8, free RAM less 16 GB over 3 GB per worker, CPUs less 4).

## Tried

- Unit 2: Gaussian neighbour weighting, no movement, reverted. Named the gap:
  trailing 1-3 late, real teams kick 93-100%, sim 61-64%.
- Unit 3: late score-scale tightening (.61 to .63, reverted); logistic
  4th-down layer on all phases (3/5, 40-49 yd kicks halved, log loss +0.0116,
  reverted).
- Unit 5: k_state 50 helped the slope less than the bandwidth; not applied.

## Next

1. Re-grade DONE 2026-09-26 (`artifacts/sim04_loso/20260926T203158Z/`,
   table in `docs/sim04_unit_log.md` "SIM-08 LOSO re-grade"; five rows
   recorded under family `sim04_play_simulator`, names `sim08_*` and
   `served_cover_probability_loso_recalibration_at_open`). Goal NOT met: raw
   and shift refuted (wrong sign resolved); tilt -0.0023 P+ 0.19; blend
   +0.0043 P+ 0.93 over served, but -0.0004 [-0.0011,+0.0003] P+ 0.14 over a
   served-only LOSO recalibration, so the gain is shrinkage, not the sim.
2. Engine shape is the remaining defect: mass at 3 (.096 vs .152), SD ratio
   1.10, total barely tracks the opening total (slope 0.34 vs 0.87); tilting
   does not fix shape. Next unit, if the lane continues: fix mass at 3 in the
   generator, then re-grade the blend against recalibrated served (checkpoints
   are keyed by engine hash, so a changed engine re-simulates; ~45 min on 6
   workers, workers peak about 2.3 GB each).
3. Served cover probability is overconfident (LOSO slope 0.27-0.43, all six
   seasons gain from shrinking); this is MKT-17's known finding, owned by
   `docs/lanes/joint-probability-model.md`, not this lane.

## Open

- None needing the owner.
