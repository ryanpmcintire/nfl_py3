# MOD-43

## Goal
QB in-season learning curve: does EPA/play rise with start index (young or new-to-team QBs) and is it priced at the opener (ROADMAP MOD-43).

## State
Unit 1 done 2026-11-03 21:36Z via scripts/mod43_qb_learning_curve.py (outputs tests/scratch/mod43/: curve_cells.csv, result.json). Starter = most dropbacks per team game, REG 2009-2025; EPA/dropback minus season mean; within-QB (FE) dropback-weighted slope on log(1+starts), QB-cluster bootstrap 2000 draws. Early = within 1 season of debut (career) or of stint start (team), debut/stint observed after 2009; veteran = rest observed. Declared before looking.
Curve (EPA/db per log-start): career early +0.021 [0.002,0.040] P+ .987 (157 QBs, 2161 g; odd/even seasons +.009/+.013); career veteran -0.030 [-0.058,-0.005] P+ .009; team early +0.011 [-0.001,0.023] P+ .96 (split halves -.010/+.025); team veteran -0.014 [-0.048,0.021] P+ .22.
Feature: slope (LOSO, other seasons) x (log1p starts now - log1p starts at season start) of prior-game starter, home minus away. Opener 2020-25, n 1503, LOSO logit:
career: decisive 135 flips, aug 69-66 vs served; LL oos .69217 vs served .69216 (dLL -0.00001 [-0.0012,0.0012] P+ .48); in-sample-minus-oos gap -0.00055; seasons+ 3/6; W-L 803-700 vs served 800-703.
team: decisive 81, aug 37-44; dLL -0.00054 [-0.0012,0.0001] P+ .056; seasons+ 0/6; W-L 793-710.
Intercept-only LL .69328. Looks: 4 curve cells + 2 pick terms = 6, family mod43_qb_learning_curve, all recorded unresolved_below_power.

## Tried
One definition of early/start/projection; prior-game starter as pregame QB proxy.

## Next
Per-fold coefficients are in result.json coef; check weeks 4-12 only; projection with Bayes-shrunk per-QB slopes; actual-starter-known (depth chart) instead of prior starter.

## Open
Curve is real for early-career (positive, odd/even consistent) but vets negative (age/regression to mean?), and no pricing gap detected: opener already absorbs it or power is lacking.
