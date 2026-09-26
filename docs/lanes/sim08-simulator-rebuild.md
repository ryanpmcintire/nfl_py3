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
- Both LOSO re-grades died on 2026-09-26 when a 22-worker run (~59 GB) ran
  the 64 GB host out of memory and crashed the session; neither wrote output
  (`artifacts/sim04_loso/20260926T150827Z` and `...T162257Z` are empty).
  `simulate_games_multiprocess` now caps workers at min(8, free RAM less
  16 GB over 3 GB per worker, CPUs less 4) and refuses to start below that.

## Tried

- Unit 2: Gaussian neighbour weighting, no movement, reverted. Named the gap:
  trailing 1-3 late, real teams kick 93-100%, sim 61-64%.
- Unit 3: late score-scale tightening (.61 to .63, reverted); logistic
  4th-down layer on all phases (3/5, 40-49 yd kicks halved, log loss +0.0116,
  reverted).
- Unit 5: k_state 50 helped the slope less than the bandwidth; not applied.

## Next

1. Re-grade attempt 3 (2026-09-26, `--n-reps 500 --workers 8`) was killed by
   Claude Code's low-memory reaper about 45 min into the first season's
   8-worker phase (warm-up 18:55-19:25 UTC); no season checkpoint was written
   (the `engine-fd9b29826e52_reps-500_mp-1/` directory is empty). Free RAM
   was 42 GB before the workers started, so 8 workers use more than the
   3 GB each the cap assumes. Next run, only when the owner asks: same
   command with `--workers 4`, in a terminal the owner starts (not a
   background shell), or with CLAUDE_CODE_DISABLE_BG_SHELL_PRESSURE_REAP=1 set
   before Claude Code starts. Measure per-worker memory in the first minutes and
   fix `simulate_games_multiprocess`'s 3 GB assumption. Checkpoints resume
   finished seasons. After: append "SIM-08 LOSO re-grade" to
   `docs/sim04_unit_log.md` and run the drafted `weak-signals record`
   commands after checking them.
2. Engine shape: mass at 3 (.096 vs .152), SD ratio 1.10, and the total that
   barely tracks the opening total (slope 0.34 vs 0.87); tilting does not fix
   shape (unit log "anchoring check").

## Open

- None needing the owner.
