# SIM-16 Pregame attrition expectation

## Goal
Pregame attrition expectation: measured in-game exit cost times the chance a thin position group loses a starter (ROADMAP SIM-16).

## State
Added 2026-11-03, unmeasured. SIM-12 measured an offensive-line exit at -0.038 EPA per play. A team with a thin backup line has a larger expected loss from exits that are invisible on Tuesday; compute that expectation from depth and hazard.

## Tried
Nothing yet. Novelty: SIM-12 (the hazard and the effect), MOD-33 (last-game exits), PER-10 (inactives). Grep found no pregame expected-attrition term.

## Next
First unit: Compute per-team expected EPA loss from exits (hazard times backup-starter gap) for 2018-2025 and test it against the opener margin residual with a permutation null.

## Open
Data: SIM-12 hazard table, depth charts and snap counts. Zero crossing closes nothing; record unresolved results with a closing ground.

## Unit 1 (measured 2026-11-03)
Script scripts/sim16_attrition.py (out tests/scratch/sim16), ruff clean. Family sim16_pregame_attrition: 3 term looks (hazard x gap, flat hazard x gap, hazard only) + 3x200 within-week permutation placebos. Hazard per OL starter slot (25220 slots 2016-25, exit 4.96%) logit on prior-game exit, Questionable (same-week report, may postdate opener), years_exp (age proxy; no age field), prior snaps/game; fit leave-one-season-out; z-coefs +0.085, +0.143, +0.200, -0.081 (se ~0.025). Gap = 1 - best bench OL prior snaps/game over starters' (mean 0.57). Mean cost 0.0053 EPA/play x 0.038 scale.
Opener grade, 1368 games, intercept-only control acc 52.8% LL 0.6921. Term (home-away): acc 51.8%, LL 0.6935 (gap ins/oos 0.002), coef +0.033 (se 0.054, P>0 0.73), folds +.07 +.09 +.02 -.01 +.02 +.01; decisive 112 games term 49 v control 63; LL gain -0.0015 (-0.0026,-0.0004), P>0 0.004; placebo p>=obs 0.95. Flat-hazard 43-57; hazard-only 52-55.
Availability corr: OL unavailability 0.00, OL continuity 0.11, offense continuity 0.06. Positive coef is the wrong sign for a cost. Recorded 3 cells unresolved_below_power.
Next: none unless a mechanism for sign; backup quality by EPA not snaps.
