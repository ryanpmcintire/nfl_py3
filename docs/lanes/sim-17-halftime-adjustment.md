# SIM-17

## Goal
Halftime adjustment: second-half success rate over first-half, conditioned on halftime margin, per coach (ROADMAP SIM-17).

## State
Added 2026-11-03 (backlog expansion 5), unmeasured, nothing run. Mechanism: Coaches adjust at halftime; teams behind or ahead by scheme differ in second-half performance. Enters the simulator as a coach-level second-half shift on SIM-10 drive outcomes, never as a global constant; the pregame side gets the same shift as a level and variance term.

## Tried
Novelty check only. Nearest: SIM-09 (urgency and stakes), SIM-10, LEAD-61 (second-half line archive, failed); grep 'halftime' found only LEAD-61. Differs: Within-game adjustment at the break.

## Next
First unit: Measure second-half minus first-half EPA per play by coach-season, split-half reliability with a coach permutation null, then test as a drive-outcome shift in the SIM-10 resampler.

## Open
Data: nflverse play-by-play by half (free). Zero crossing closes nothing; one fitted probability selects the side; weights, cut points and alphas chosen leave-one-season-out; every cell counted as a look; unresolved results go through nfl-ats weak-signals record with a closing ground; graded at the opener (opener as model input, with intercept-only and line-move-only controls) on the same market and simple-model baselines. Look count and family declared before any sign is seen.
