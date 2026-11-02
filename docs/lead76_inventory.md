# LEAD-76 rule-aware margin prior: sources, rule eras, measurement

## Sources (read/measured)
- **Measured:** nflverse schedules via `nflreadpy.load_schedules` (the repo's own source, `src/nfl_ats/data.py:75`) fetched for 1999-2008: 2,500 REG games, all with scores and `spread_line`. Saved at `artifacts/lead76/nflverse_schedules_1999_2008.parquet` (artifact, untracked).
- **Measured:** `data/raw/20261027T220410Z/schedules.parquet` holds 2009-2021 REG games with scores (earlier inventory undercounted by scanning only to 2019). Pool = 5,880 REG games, 1999-2021. Pool lines are nflverse (closing) lines: older games supply shape only.
- **Measured:** opener proxy for grading is `artifacts/sbr_era_opener_eval/20260819T233013Z/scored.parquet` (2011-2021, served model residual and probability at the proxy opener). Authentic pool captures for 2020-2025: 0. `data/market/historical/open_close` covers 2025 only (285 games), not used. Declared population 2020-2025 is replaced by proxy openers 2013-2021 (folds), 2015-2021 evaluated: a deviation from Protocol B (F=6), stated here.
- Elo baseline: not available in this unit; market-only and model-only baselines used.

## Rule eras (reported; from league rule history, not re-fetched this session)
| Era | Seasons | PAT spot | Regular-season overtime |
| --- | --- | --- | --- |
| 0 | 1999-2011 | 20-yard line (kick), 2-pt attempt since 1994 | sudden death (the 2010 modified format applied to playoffs only) |
| 1 | 2012-2014 | 20-yard line | modified sudden death (adopted March 2012) |
| 2 | 2015-2016 | 15-yard line (adopted May 2015) | modified, 15 minutes |
| 3 | 2017-2024 | 15-yard line | modified, 10 minutes (adopted 2017) |
| 4 | 2025- | 15-yard line | both teams possess even after a first-drive touchdown, 10 minutes (new regime; inherits the pooled prior) |
Sources to confirm: NFL Operations rules-history pages and the NFL Rulebook overtime and extra-point sections; Wikipedia "Overtime (sports)#National Football League". Dates verified this session (**read**, secondary-to-primary: the NFL.com and Pro Football Hall of Fame pages; operations.nfl.com rule-history and rulebook pages were fetched and carry no dated text):
- Modified sudden death adopted for the 2012 regular season (ties allowed): https://www.nfl.com/news/nfl-overtime-rules-09000d5d827ee2c0 and https://www.profootballhof.com/football-history/football-history/2010-2019/2012 . Confirmed.
- Extra point snap to the 15-yard line: owners approved 2015-05-19 (30-2), effective 2015 season: https://www.cbsnews.com/boston/news/nfl-votes-to-move-extra-point-kick-attempts-to-15-yard-line/ and https://www.washingtontimes.com/news/2015/may/19/nfl-changes-extra-point-kick-distance/ . Confirmed.
- Overtime 15 to 10 minutes: adopted 2017-05-23 at the Chicago spring meeting: https://www.nfl.com/news/nfl-overtime-rules-09000d5d827ee2c0 . Confirmed.
- CORRECTION: the prior unit ended the rules chronology at 2017. The 2025 season changed regular-season overtime so both teams get a possession: https://www.nfl.com/news/looking-at-tweak-to-kickoff-other-rules-changes-for-2025-season . Era break added for 2025.
- The 2010 postseason-only modification (era 0 sudden death in regular season) is consistent with the nfl.com overtime page above.

## Measured results (script `scripts/lead76_rule_prior.py`, artifacts in `artifacts/lead76/`)
Key-number mass |margin|=3,7,10,14,17 by era (Wilson 95%): all era intervals overlap; PAT-era (pre vs 2015+) permutation p: 3 0.48, 7 0.70, 10 0.13, 14 0.15, 17 0.28. Era magnitude, not absence: 10 fell 6.1% to 4.3-5.4%, 14 rose 4.6% to 5.1-5.6%, unresolved.
Fitted era weight (earlier seasons only, folds 2015-2021): 0.000006-0.0094; folds 2013-2014 had no era fit and inherit the pooled prior.
Cover, non-push, 1,745 games 2015-2021, accuracy / log loss / Brier: market-only 51.1 / 0.6944 / 0.2506; served 51.4 / 0.7024 / 0.2542; served recalibrated 51.4 / 0.6919 / 0.2494; pooled prior 50.5 / 0.6988 / 0.2527; rule prior 50.5 / 0.6988 / 0.2527; served+rule term 51.9 / 0.6916 / 0.2492; joint 51.6 / 0.6917 / 0.2493.
In-sample vs out-of-sample log loss gap: term 0.6913 vs 0.6916 (gap 0.0003), joint gap 0.0004.
Decisive games (side differs from served): rule term 58-49 (107 games), joint 85-82 (167), rule prior alone 155-171 (326), pooled 156-172.
Paired log-loss improvement, season-block bootstrap (7 blocks): rule vs pooled +1.1e-5 [0.8e-6, 2.6e-5] probability_positive 0.979 (3 of 7 seasons); term vs recalibrated served +2.7e-4 [-1.0e-4, 9.3e-4] 0.671 (2/7); joint vs recalibrated +2.4e-4 [-1.7e-4, 8.8e-4] 0.736; pooled vs served +0.0037 0.985 (the gain is the recalibration, not the rule). Whole-margin log score rule vs pooled +2e-5 [~0, 4e-5] 0.919 (9 blocks, 38 games out of support); pooled-at-model-point vs market-only -0.029 [-0.053, -0.010] 0.0003 (model point worsens whole-margin score).
Reliability tables: `artifacts/lead76/reliability.csv` (five bands per model, edges from early-season quantiles).
Looks: 19 paired bootstrap cells + 3 whole-margin + 5 permutation + 20 era key-mass intervals + 7 summary models = 54 reported cells (family lead76_rule_prior). Not the declared 361; fewer folds and no Elo arm.

## Implication
**Inferred:** the rule-conditioned prior is unresolved_below_power: direction positive, size ~1e-5 log loss, three registry cells recorded. It changed no served pick and is not a flip rule. Authentic 2020-2025 openers remain required for the declared population.

## Declared-protocol run (2020-2025 authentic openers), `scripts/lead76_declared.py`, artifacts `artifacts/lead76_declared/`
Population (**measured**): 1,537 served-model opener games (1,503 non-push), 2020-2025; results match the nflreadpy schedule on 1,537 of 1,537. Pool for fold s = 1999-2019 nflverse margins plus authentic openers of graded seasons before s. Era weight fitted on earlier seasons only (0.006 to 0.024). Elo arm: 538-published constants (K 20, home 48, reversion 1/3, 25 Elo per point, margin multiplier), pregame ratings from all prior games, run through the same discrete read.
Decisive games first (side differs from served; new side right / served right): rule prior 110-128 of 238; rule term chronological 50-58 (4 folds); term leave-one-season-out 57-68; market-only 253-289; Elo 323-395.
Cover non-push, accuracy / log loss / Brier (n=1,503, 6 folds): market-only 50.7 / 0.6944 / 0.2506; served 53.1 / 0.6983 / 0.2524; pooled prior 51.9 / 0.6957 / 0.2512; rule prior 51.9 / 0.6957 / 0.2512; Elo 48.3 / 0.7143 / 0.2597. Chronological folds 2022-2025 (n=1,047): recalibrated served 52.5 / 0.6927 / 0.2498; term 51.8 / 0.6942 / 0.2505; joint 51.7 / 0.6950 / 0.2509. Leave-one-season-out (n=1,503): recalibrated served 0.6923, term 0.6931, joint 0.6936.
In-sample vs out-of-sample log loss gap: recalibrated 0.0016, term 0.0033, joint 0.0042 (chronological); 0.0003 / 0.0013 / 0.0018 (leave-one-season-out). Term coefficients on the rule-minus-pool difference swing in sign and size across folds (about -61 to +76) because the difference is nearly collinear with the pooled read: unstable.
Paired season-block log loss improvement (mean [95%] probability_positive, blocks): rule vs pooled +1.0e-5 [-1.0e-5, 3.1e-5] 0.657 (3/6); whole-margin score rule vs pooled +2.4e-4 [8e-5, 4.8e-4] 1.0 (5/6); term vs recalibrated served -1.5e-3 [-2.8e-3, -2.2e-4] 0.0 (0/4), leave-one-season-out -7.9e-4 [-1.6e-3, -7e-5] 0.005 (1/6); pooled read vs served +2.6e-3 [9e-4, 4.3e-3] 0.9998 (5/6; calibration, not rules); term vs Elo +0.0198 1.0. Rule-era 2025 is an unseen regime so rule equals pooled there.
Reliability (five training-quantile bands, rule prior): observed cover 47.3, 47.5, 52.7, 49.0, 53.5 percent against mean probability 39.7 to 57.6 percent: the prior spreads probability wider than outcomes (`reliability.csv`). Per-season in `by_season.csv`.
Looks: declared L=361; reported 173 cells (3 summary metrics for each of 21 model-set rows plus IS and gap cells, 18 decisive cells, 31 paired cells, 4 margin cells, reliability bands, F=6 not 7 fold pooling, no extra grids). Fewer than declared; the unreported declared cells are per-fold metric cells not computed.
**Inferred:** the rule-era effect is a tiny shape gain unresolved_below_power; it is not usable as a probability term (out-of-season loss); nothing flips a pick.
