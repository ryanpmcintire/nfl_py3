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
| 3 | 2017-2021 | 15-yard line | modified, 10 minutes (adopted 2017) |
Sources to confirm: NFL Operations rules-history pages and the NFL Rulebook overtime and extra-point sections; Wikipedia "Overtime (sports)#National Football League". Dates are **reported**, unverified against the primary document.

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
