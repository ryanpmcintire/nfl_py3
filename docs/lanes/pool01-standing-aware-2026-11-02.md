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

## Field Best Pick model and forward sim (2026-11-03)
- Parquet data/splash/field/2026_field_best_picks_w01-08.parquet (242 rows, 121 games). Autopicks absent in raw (0 for W3-8, NaN W1-2). Script scripts/pool01_field_best_pick.py -> artifacts/pool_rank_replay/field_best_pick_model.json: conditional logit over sides (fav, |spread|, model p), leave-one-week-out; held-out loglik/best pick -3.289 v uniform -3.408, gain +0.118/wk [0.059, 0.178]; game-share calibration slope 0.79 (shrinks; quintile table in json). With lock pick counts: -3.128, slope 1.36.
- scripts/pool01_forward_sim.py (10 weeks x 15 games, rank 19 / 65 pts conditioned by rejection, 1200 paths, inner 200): P(first) served 18.5% [16.3, 20.7], aware 18.8% [16.7, 21.1], gain +0.375 pp [0.083, 0.75] P+ 0.999 but policies differ on only 6 paths; BP tiebreak value +1.38 pp [0.87, 1.92]. Recorded pool01_standing_aware_forward_from_rank19.
- Caveats: model probs assumed calibrated (inflates P(first)); future-game features bootstrapped from W1-8; full-fit BP coefficients used in sim; W8 standings page not read, rival points simulated.
## Next
Read Week 8 standings for real rival scores; replace bootstrap with real remaining schedule.

## Real weekly points and rerun (2026-11-03)
- Parsed data/splash/field/2026_weekly_points_w01-08.parquet via scripts/pool01_weekly_points.py -> artifacts/pool_rank_replay/weekly_points_field.json (241 of 250 entries; games/week 16,16,16,16,15,14,14,14).
- Measured: field hit rate 0.483; total sd 5.38 v binomial 5.50 (ratio 0.98, no over-dispersion); per-week sd 1.75-2.00 v binomial 1.87-2.00. Odd/even split-half r 0.039, Spearman-Brown 0.075 [-0.23, 0.29]: entry skill essentially unmeasurable at 8 weeks. Our 65 pts: 15 read entries above, 9 tied (incl. us), percentile ~92 (z +1.21).
- scripts/pool01_forward_sim.py now uses real rival season totals (our row removed, 9 unread rivals resampled from read totals), rival skill offsets = SB reliability x (rate - mean), posterior sd sqrt(r(1-r)var); rejection to rank 19. Output -> forward_sim_real_rivals_results.json.
- Full run (--paths 1200 --inner 200) was STILL RUNNING at hand-back (PIDs 17688, 30408; stdout in scratchpad fs.out). Not yet read or recorded.
## Next
When artifacts/pool_rank_replay/forward_sim_real_rivals_results.json exists: read it, compare to 18.5% [16.3,20.7] / aware +0.375 pp / BP value +1.38 pp, run `nfl-ats weak-signals record` family pool01_standing_aware (name pool01_standing_aware_forward_real_rivals, payout_first_pp, unresolved_below_power, source the json), decide whether Week 15 revisit trigger changes. Kill leftover PIDs if still alive.
## Rerun result (2026-11-03, measured)
forward_sim_real_rivals_results.json (1200 paths, inner 200): P(first) served 6.67% [5.29, 8.08], aware 6.83% [5.42, 8.33], gain +0.167 pp [0.0, 0.375] P+ 0.975 but policies differ on 3 of 1200 paths; BP tiebreak value +0.59 pp [0.27, 0.95]. Earlier simulated-rival result was 18.5% / +0.375 pp / BP +1.38 pp: real rivals (15 above us, 9 tied, near-zero skill spread) cut P(first) about 2.8x. Recorded pool01_standing_aware_forward_real_rivals (unresolved_below_power). Week 15 revisit trigger: unchanged in kind (aware choice is noise-level; BP matters only as tiebreak), but P(first) base is ~7% not 18%; the earlier figure is superseded. Processes reaped.
