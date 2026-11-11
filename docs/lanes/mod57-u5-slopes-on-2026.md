# MOD-57 U5 slopes on 2026

## Goal
Score unit 4's Arm A (move slope split at abs 0.5) and Arm M (model slope by state) on 2026 Weeks 1-9 (136 games, pool-line grade), coefficients frozen on 2020-25. One look per arm.

## State
Done 2026-11-11 08:53 (date stamp). Frozen coefficients tests/scratch/mod57/u5/frozen.json sha256 a65e62d8e63e9010a9d43c90b973b29895966c19328240d17c1fdb8db310a24f; four-term matches artifact 20261111T011859Z (0.308 / 0.220 / flag 0.234). Harness reproduced: 85-51 (no-conversion row of root_m5). Scripts freeze.py, score.py, results.json, batch.json in tests/scratch/mod57/u5/. 6 cells recorded unresolved_below_power, registry/weak_signals/mod57_quiet_game_slopes.json.
Results (current 85-51 ll .6787): Arm A 81-55 ll .6806 (acc -2.93 pts, P+ .15; ll P+ .24; Brier P+ .24; 24 changed picks, arm 10-14; 15 at |move|=.5, arm 6-9). Arm M 82-54 ll .6770 (acc -2.18, P+ .00; ll P+ .78; Brier P+ .77; 3 changed, arm 0-3).
Inputs: move available 118/136; of those abs 0 44%, 0.5 33%, 1 8%, more 13%, other 3%; 3 leader books quoted per game in all 136.

## Tried
Flag term is zero on 2026 (harness is flag-free); arms scored z-then-zero for out-of-regime columns as trained. Own frozen four-term 83-53, differs from harness current on 2 picks.

## Next
Not served. No further arm; wait for Weeks 10-18 forward cohort only if the owner wants a second look.

## Open
None.
