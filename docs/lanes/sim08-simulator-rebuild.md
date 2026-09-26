# SIM-08 simulator rebuild to catch the model

## Goal

Make the play simulator (`scripts/sim04_engine.py`) at least match the served
discrete read at the opener, then test whether it adds anything. Done = a LOSO
2020-2025 re-grade where the market-anchored sim ties or beats the served read
on cover-vs-miss and push log loss, recorded with `weak-signals record`.

## State

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

1. Re-grade the current engine once, when no scheduler weekly-run is active
   and at least 40 GB is free: `scripts/sim04_loso.py --n-reps 500 --workers 8`
   (about 4-5 h; Monte Carlo cost 0.002). One heavy job at a time; no other
   simulation running alongside. Add per-season checkpointing first so a kill
   does not lose the whole run. Then append "SIM-08 LOSO re-grade" to the unit
   log and run the drafted `weak-signals record` commands after checking them.
2. Engine shape: mass at 3 (.096 vs .152), SD ratio 1.10, and the total that
   barely tracks the opening total (slope 0.34 vs 0.87); tilting does not fix
   shape (unit log "anchoring check").

## Open

- None needing the owner.
