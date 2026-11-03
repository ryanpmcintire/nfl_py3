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

## POL-05 and POOL-01 on measured observables (2026-11-03)
Script `scripts/pool_observed_replay.py`, artifact `artifacts/pool_rank_replay/observed_results.json` (+ `observed_*_weekly.csv`). Inputs read from `data/pool_observables/20261103T130853Z`: 250 entries, 1 paid place, Best Pick worth 0 points (tiebreak only: points, then Best Pick wins), co-winner share fractional. Field lean 0.638 measured from 2026 w1-7 pick counts (107 games), 0.651 from pre-lock splits; our agreement with the field's majority side 51.4%.
- Replay, 6 seasons, 0 side flips, 1500 samples, lean arms 2: aware Best Pick differs from served in 69 of 107 weeks (64 on split lean) but simulated P(first) gain is 0.0005, i.e. noise; realised P(first) gain per season -0.57 pp [-1.71, 0.00], P+ 0.167 (split lean -1.39 pp, P+ 0.009). Decisive 16 aware covered / 21 served covered / 32 same; probability given up 0.071. Recorded: `pool01_standing_aware_observed_tiebreak`.
- POL-05 (250 entries, 18x15 games, served hit 55.5%): P(first) served 20.4% (split lean 21.3%); flip 10 public sides 17.4%; flip 25 13.6%; +2 accuracy points 41.4%. Contrarianism loses; accuracy dominates.
- Limits: hit rate 55.5% is the population hit rate (in-sample); field Best Pick distribution per game not measurable (best_picks only in w1-2 tsv), so rival tiebreak wins are independent of our game choice; tiebreak only, no weekly Last-game-total modelled; today's actual standing (19th, 65 pts) not simulated.
## Next
Fit field best-pick game shares once more weeks of `best_picks` are captured; simulate forward from rank 19 / 65 points.
