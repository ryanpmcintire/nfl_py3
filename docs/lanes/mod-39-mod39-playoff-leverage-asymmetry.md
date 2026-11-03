# MOD-39 Playoff leverage asymmetry for Weeks 10-16

## Goal
Pregame playoff-probability swing (P(playoffs|win) - P(playoffs|loss)), home minus away (levdiff), as one LOSO term at the opener. Family mod39_playoff_leverage (declared before running; 7 looks counted, 4 recorded).

## State
Unit 1 done 2026-11-03. scripts/mod39_leverage.py -> artifacts/mod39/mod39_results.json. Swing reused from artifacts/sim09/stakes_teamweek.csv (standings Monte Carlo, 6000 sims, as-of margin ratings; k,c,slope fit on 2009-17 so 2009-17 is in-sample for 4 rating parameters). 1812 games wk10-16 2009-25; levdiff SD 0.162.
Measured (opener, 2020-25, n=592): slope +3.18 ats pts per 1.0 levdiff, interval -2.85..9.33, probability_positive 0.849; with open spread control 2.75 (0.820). Close-graded 2009-19: -1.59 (-6.33..3.08, p+ 0.257); odd/even-season halves -3.82 / +3.49 (no split-half stability).
LOSO term (offset = opener model prob) vs intercept-only: log loss gain -0.00094 (-0.0028..0.0010, p+ 0.18), accuracy +0.68 pp (p+ 0.74); in-sample logloss +0.00017, gap 0.0011. Decisive games (pick differs from control) 32: term 18, control 14.
Overlap with SIM-09 stakes: levdiff corr with elimination diff -0.51, clinch diff -0.33; levdiff+stakes LOSO logloss -0.0032 (p+ 0.10), decisive 32 v 36. Mostly shared signal, no gain.

## Tried
All four cells recorded unresolved_below_power (registry/weak_signals). Not closed: no wrong-sign resolution, no positive control.

## Next
Pool the three stake measures (levdiff, elim, clinch) as one declared family via weak-signals pool; MOD-36 noise-floor comparison; pre-2020 openers absent so 2009-19 stays close-graded.

## Open
Swing is Monte Carlo (6000 sims) with ties unmodeled and tiebreakers by random jitter. Opener coverage is 2020-25 only.
