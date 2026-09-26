# SIM-08 simulator rebuild to catch the model

## Goal

Make the play simulator (`scripts/sim04_engine.py`) at least match the served
discrete read at the opener, then test whether it adds anything. Done = a LOSO
2020-2025 re-grade where the market-anchored sim ties or beats the served read
on cover-vs-miss and push log loss, recorded with `weak-signals record`.

## State

- UNIT 7 IN FLIGHT (2026-09-26, subagent hit 50-tool-call cap; background
  task `bvw7920cf` keeps running independently of this session): full
  detail and next-agent instructions are in "Next" item 2 below. Do not
  relaunch the run; check for a fresh `artifacts/sim04_loso/<timestamp>/`
  directory first (newer than 20260926T034537Z).
- SIM-04 was closed while the engine failed its own gate (2/5 key numbers);
  its registry rows refute that engine version only.
- Unit 1 (`69fe2f1`): OT neighbour pool includes phase 3; OT TDs per play .007
  to .017 (actual .020).
- Unit 4 (`0014146`): late-phase boosted-tree 4th-down layer. Validation
  2015-2017: 4/5 key numbers, log loss +0.0080 vs naive (first GO). Mass at 3
  .102 vs .152; SD ratio 1.10.
- Unit 5: the 4th-down layer now fits on whatever seasons the tables use and
  runs in the team-conditioned path too (before, it was silently absent from
  LOSO and SIM-05); unconditioned validation reproduces unit 4 exactly.
  Kernel bandwidth stays `TEAM_KERNEL_H_SCALE = 0.5`: narrowing it to 0.10
  (chosen on a noisy 150-game slope check) wrong-signed the backup-QB what-if
  (+1.64, SE 0.26; home edge +0.23), `artifacts/sim05_whatif/20260926T141045Z`.
  At 0.5 with the layer (`20260926T141946Z`, 5000 games): home edge +1.39
  (SE 0.19), QB shift -0.73 (SE 0.27) vs -1.24 (SE 0.14) without the layer.
- Unit 6: damping closed without touching the bandwidth. On 2009-2014 tables
  the kernel draw itself already carries ~77% of the target rating gap into
  the drawn play's label (`tests/scratch/sim08_unit6_transmission.py`). Added
  a fixed-effects-slope yard shift (`TEAM_RATING_YARD_GAIN = 0.6425`,
  `scripts/sim04_engine.py:34,902-909`) on top of the unchanged kernel draw.
  5000-game acceptance run (`artifacts/sim05_whatif/20260926T144104Z`): home
  edge +1.8938 (SE 0.1935, in [1.8, 3.0]), backup-QB shift -1.3694
  (SE 0.2753, more negative than -0.73). Unconditioned validation unchanged:
  4/5, +0.008016485953585839, GO. Detail: `docs/sim04_unit_log.md` "SIM-08
  unit 6".
- Validation 2015-2017 has had about 13 engine looks; LOSO 2020-2025 is the
  untouched test. Detail: `docs/sim04_unit_log.md` "SIM-08 unit 1" to "unit 6".

## Tried

- Unit 2: Gaussian neighbour weighting, no movement, reverted. Named the gap:
  trailing 1-3 late, real teams kick 93-100%, sim 61-64%.
- Unit 3: late score-scale tightening (.61 to .63, reverted); logistic
  4th-down layer on all phases (3/5, 40-49 yd kicks halved, log loss +0.0116,
  reverted).
- Unit 5: k_state 50 helped the slope less than the bandwidth; not applied.

## Next

1. Unit 8 (second LOSO look, post goal-line fix) LAUNCHED and RUNNING:
   `scripts/sim04_loso.py --n-reps 1000 --workers 22`, pid 34068, started
   2026-09-26 16:22:55Z, artifact dir
   `artifacts/sim04_loso/20260926T162257Z/` (do not touch unit 7's own
   `artifacts/sim04_loso/20260926T150827Z`, pid 7024/28028, still running
   separately; do not relaunch either job). Change: `scripts/sim04_loso.py`
   gained `--n-reps`/`--workers` argparse in `main()` (default
   `TEAM_COND_N_REPS`=180/1, serial path byte-unchanged); `workers>1` routes
   `build_team_conditioned_hists` (new helpers `_team_ratings_for_row`,
   `_append_team_cond_hists`) through `sim04_engine.simulate_games_multiprocess`
   once per held-out season, seed `RNG_SEED + season*1_000_000` + per-game
   index. Reads (raw, shift, tilt, blend) and all report keys unchanged; only
   `n_reps_per_game_used` now reflects `--n-reps` and `n_reps_reduction_reason`
   branches on whether a reduction actually happened. Smoke
   (`tests/scratch/sim08_unit8_mp_smoke.py`, season 2020, 20 reps, 8 workers,
   run twice): 227 games, identical=True, ~88s/run (measured). ETA: 51.2
   games-reps/s at 8 workers (season-2020 table, measured, concurrent with
   unit 7) extrapolated linearly to 22 workers and slowed per-season by the
   unit-7 single-process 2020-vs-2025 ratio (30.5/27.3, inferred to hold under
   multiprocessing) -> ~3.2 h for 1537 games x 1000 reps, under the 4 h cap;
   Monte Carlo cost (K-1)/2N = 1/1000 = 0.001. When
   `artifacts/sim04_loso/20260926T162257Z/report.json` exists and is stable,
   append "SIM-08 unit 8 LOSO re-grade" to `docs/sim04_unit_log.md` (same
   keys as unit 7, plus the shape fix now in effect) and have root run the
   drafted `weak-signals record` commands after checking them.
2. Shape: goal-line borrowing fixed (`SCALE_FP` 2.5, `3cc779e`; log loss
   +0.0024, 4/5). The unit 7 run grades the engine before this change, so a
   second LOSO look is needed after it. Still open: mass at 3 (.096 vs .152),
   SD ratio 1.10, Q4 variance.
3. Parallel on copies: anchoring check (`tests/scratch/sim08_condition/`:
   does sim key-number mass track spread and total; joint spread+total tilt)
   and speed (`tests/scratch/sim08_speed/`, engine now ~28 games/s, output
   must stay bit-identical; its copy predates `SCALE_FP` 2.5). Port what passes after the re-grade finishes.

## Open

- None needing the owner.
