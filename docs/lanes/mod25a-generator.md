# mod25a-generator

## Goal
Season generator with latent team strength plus predeclared verifier gates. Parent docs/lanes/mod25-synthetic-data.md. No model grading here.

## State
Gates declared 2026-10-01 BEFORE any generator run. Real values **measured** by `scripts/mod25_generator.py real` (2011-2025 REG, 3,919 games; artifact `artifacts/mod25_generator/real_targets.json`). Interval = season-block bootstrap, 2,000 draws, 2.5-97.5 percent. A setting passes a gate iff its pooled synthetic value (many 15-season chains, same builder code, same week buckets) lies inside the interval. Pass-all is required to be called accepted.

| gate | real | 95% interval |
|---|---|---|
| margin mass at 3 | .1429 | .1335-.1523 |
| at 7 | .0880 | .0783-.0981 |
| at 10 | .0513 | .0457-.0564 |
| at 14 | .0498 | .0419-.0576 |
| at 17 | .0332 | .0269-.0391 |
| margin SD | 14.47 | 13.97-14.87 |
| home edge | 2.02 | 1.55-2.44 |
| share decided by 3 or fewer | .2360 | .2243-.2485 |
| lag-1 autocorr team off EPA/play (deviation from league-season mean, within team-season) | .156 | .133-.177 |
| R2 rolling EMA(span 8) team EPA net diff vs next margin, weeks 1-4 | .079 | .041-.125 |
| weeks 5-9 | .121 | .089-.155 |
| weeks 10-18 | .148 | .132-.166 |

Rolling state is production `build_team_states` (span 8, min 3, retention .67), pregame via previous-row shift plus offseason retention as in `attach_team_states`. Real team stats are built from nflverse plays with the same aggregation used on synthetic plays.

## Tried
- Pilots (3 worlds x 5 seasons, **measured**): engine default (scale 1) autocorr .037 (damped), margin mass at 3 = .10-.11 (real .143) at every setting; neutral engine totals 39.8 points/game vs real about 45 (engine under-scores, **measured** 2,500 neutral games); yard_bias 1.0 gives 47.2, so 0.75 is fixed for the grid. Pilot scale 3 / yard gain 3 matched the three R2 gates but overshot autocorr (.239) and SD (16.6).
- Read: engine yard shift uses off_row - def_row where def is EPA allowed (higher = worse defense), so a bad defense lowers the offense's shift. Grid tests def_sign -1 (original) vs +1 (additive, correct sign).
- Declared grid (before running): def_sign {+1,-1} x scale {2,3} x yard_gain {2,4}, yard_bias .75, drift 1; plus drift .5 and 2 at def_sign +1 / scale 2.5 / yard_gain 3. 6 worlds x 5 seasons each, 2 burn-in seasons dropped. Logs `artifacts/mod25_generator/log_g_*.txt`, `gen_g_*.json`.

## State update
Code done: scripts/mod25_generator.py (real, fit, gen) and scripts/mod25_produce.py (batch producer to data/processed/synthetic/<tag>/). Fit params in artifacts/mod25_generator/fit_params.json (offense phi .999, defense phi .974, backup QB rate .16, effect -.057 EPA). Throughput measured about 105-120 games/s at 6-8 workers (workers use about 1 GB each). Grid started 14:47 via artifacts/mod25_generator/grid.sh (background, 10 runs, about 90 s each); done marker grid_done.txt. Not yet read. Engine shape defect confirmed (mass at 3 stays about .10 at every setting; real .143), likely fails 3, 7 and share-le3 gates.

## Rule (owner 2026-10-01)
The key-number shape is a held-out check that the generator reflects football,
never a target. No parameter, reweighting or repair may aim at mass on 3, 7,
10, 14 or 17. A shape failure points to a wrong mechanism, such as coach
decisions by score, time and field (4th down, playing for the field goal),
endgame and clock behaviour, or scoring units. Fix the mechanism and re-check.
Scoring-level and strength parameters are fitted only to their own real
targets (points per game, EPA spread), never to the margin shape.

## Next
1. Read gen_g_*.json / log_g_*.txt, tabulate gates per setting, pick best (or closest, labelled failing).
2. python scripts/mod25_produce.py --tag best --setting '{json of chosen setting incl yard_bias .75}' --workers 12 --budget-seconds 1800 (about 2,000 seasons needs about 75 min at 120 g/s; report seasons reached).
3. Report to the parent with gate table. Consider a mechanism-named key-number repair only if scoring-composition diagnosis supports it.

## Open
Interval crossing zero closes nothing here; gates are fidelity filters, not signal tests.
