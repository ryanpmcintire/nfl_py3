# POOL-01 unit 2 alternative: standing-aware Best Pick

## Goal
Standing-aware variant of POOL-01 using `simulate_pool_finish` field machinery (`_field_scores`), field from market-implied public shares only. Sides stay as served; the layer only picks which game gets the Best Pick, by simulated expected final rank given the realised standing and weeks remaining (carry-in score vs simulated field, future weeks at served Best Pick = top-probability game).

## State
Script `scripts/pool01_standing_aware.py`; artifacts `artifacts/pool_rank_replay/standing_aware_*_{results.json,weekly.csv}`. 107 season-weeks, 6 seasons, 0 side flips. Pool entrants, Best Pick bonus/penalty are UNKNOWN (no captured pool observables); runs take them as sensitivity arguments.
- 400 samples: noise-chasing (winner's curse over ~14 candidates), 65 decisive weeks; superseded.
- 3000 samples, 100 entrants, bonus 1 / penalty 0 (headline): 39 decisive weeks; aware pick covered/served missed 7, same 21, served covered/aware missed 11; 77% of aware picks on the public side; mean win probability given up 0.019. Final-rank gain -1.34 per season [-4.38, +1.04], P+ 0.173; points -0.037/week [-0.112, +0.037], P+ 0.142. Season gains: +3.07, -1.19, -8.17, +0.20, -0.03, -1.95.
- Unit 1 (+1.47 ranks/week) flipped sides, so it is not comparable and not admissible under the one-probability rule.

## Tried
Argmin of simulated expected rank per candidate with common random numbers; no significance threshold (would be a magic number).

## Next
Needs real pool Best Pick scoring and field size (`nfl-ats pool-observables`) before any served use. Record cells via weak-signals once final sensitivity run is read.

## Open
Look count: 3 runs of one family (policy A vs standing-aware B). Unresolved_below_power; no closing ground.
