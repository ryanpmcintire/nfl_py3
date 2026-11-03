# MOD-37 Stadium false-start premium

## Goal
Venue trait: visiting offense false-start excess at a stadium. Family mod37_venue_false_start (3 cells + 2 sensitivities).

## State
Unit 1 done 2026-11-03. Scripts scripts/mod37_false_start.py, scripts/mod37_step3.py; output artifacts/mod37/. Recorded 3 cells, unresolved_below_power.
- Reliability (measured): 4431 games, 48 venues, expected = visitor team-season rate (leave-game-out) x crew ratio. Odd/even-season r 0.29 (SB 0.45), venue-permutation p 0.025; random halves r 0.21; no-crew r 0.33 p 0.014. Roof: closed +0.13, open +0.19, dome -0.01, outdoors -0.02 FS/game; dome-only r -0.11, outdoors-only 0.38.
- Visitor EPA/play (measured, as-of strength + roof, LOSO, 2011-2025, 3919 games): coef -0.004, interval [-0.076, +0.054], P(neg) 0.56, 12/14 folds negative, LOSO sq-error gain negative. About 0.0004 EPA per 1 sd.
- Opener term (measured, 2020-2025, 1503 games): 200 flips, arm 96 right v base 104. Log loss gain vs intercept-only +0.000006, P+ 0.67; the +0.0012 vs base is only the intercept recalibration.

## Tried
Premium shrinkage by empirical Bayes from prior seasons only.

## Next
Premium is reliable but does not reach scoring. Retest with dome-only vs outdoor venue split, or with delay-of-game and encroachment added, only if a mechanism is named.

## Open
Opener-graded base log loss sits above intercept-only; not examined here.
