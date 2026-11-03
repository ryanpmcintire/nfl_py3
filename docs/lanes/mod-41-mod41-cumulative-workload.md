# MOD-41 Cumulative workload

## Goal
Cumulative workload: offensive and defensive plays over the last three games, overtime, and short-week stacking (ROADMAP MOD-41).

## State
Added 2026-11-03, unmeasured. Rest days are priced; snaps absorbed are not. A defense that faced 80 plays in each of the last three games, or played overtime on a short week, enters with a load that days of rest do not describe.

## Tried
Nothing yet. Novelty: SIM-13 (defensive snap load, simulator side), MOD-29 (kickoff-clock rest), existing rest and bye features. Grep for 'snap load' found SIM-13 only.

## Next
First unit: Build three-game plays-defended and plays-run totals plus an overtime flag, regress the opener margin residual on the home-minus-away load difference with the rest-day term in the model, and run a within-week permutation null with a split-half by season parity.

## Open
Data: nflverse play-by-play play counts and schedule dates. Zero crossing closes nothing; record unresolved results with a closing ground.
