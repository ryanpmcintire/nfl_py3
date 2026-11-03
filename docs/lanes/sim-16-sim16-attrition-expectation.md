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
