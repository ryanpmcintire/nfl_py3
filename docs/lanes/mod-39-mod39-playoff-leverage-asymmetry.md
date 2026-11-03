# MOD-39 Playoff leverage asymmetry for Weeks 10-16

## Goal
Playoff leverage asymmetry for Weeks 10-16: win-probability swing to the playoff chance for each side as a margin term (ROADMAP MOD-39).

## State
Added 2026-11-03, unmeasured. A team one win from elimination and one that has clinched are not equally motivated, and the opener prices team quality, not stakes. The swing in playoff probability from winning versus losing is computable from the schedule alone, so the term is strictly pregame.

## Tried
Nothing yet. Novelty: SIM-09 (stakes in the simulator, drive level), MOD-28 (late-game habit), docs/lanes/sim09-urgency-stakes.md. Grep found no pregame leverage term.

## Next
First unit: Compute each side's leverage for Weeks 10-16, 2009-2025, as playoff probability if win minus if loss, then fit the leverage difference as one term in the add-one pipeline scored leave-one-season-out against the MOD-36 noise floor, and report the decisive-game record.

## Open
Data: nflverse schedules and results; remaining schedule simulated with the served team ratings to get playoff chance with and without the win. Zero crossing closes nothing; record unresolved results with a closing ground.
