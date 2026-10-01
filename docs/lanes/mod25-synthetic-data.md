# MOD-25 synthetic training data

## Goal
Beat the base model at the opener by training it on synthetic games from a
generator checked against reality: play-level knowledge into the game model.

## State
2026-10-01 opened on owner prompt. Generator candidate: scripts/sim04_engine.py
(real-play nearest neighbour; about 10k games in 16 s; known defects: 3-point
mass .103 vs .152, team effects damped to about 40%; see SIM-08 lane).

## Design
- Generator: seasons with latent team strengths (with drift and shocks) that
  drive play sampling, emitting play rows with real columns.
- Verifier: accept a generator setting only if synthetic seasons match real
  2011-2025 on margin key-number mass (3, 7, 10, 14), margin SD, home edge,
  week-to-week autocorrelation of team EPA, and the real predictability curve
  (R-squared of rolling stats to next margin by week). Gates written before
  the run.
- Book: a pricing model fitted on real openers (features to line) prices the
  synthetic games, so margin minus line exists in the synthetic world.
- Student: the production features on synthetic histories train the ridge;
  the real ridge shrinks toward that, with strength chosen out of season.
  Graded on the real opener 2020-2025 and 2011-2019 separately.

## Next
Unit a: generator plus verifier (docs/lanes/mod25a-generator.md).

## Open
Synthetic residuals can only teach book errors the generator gets right. The
real-opener grade is the only judge.
