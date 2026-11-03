# MOD-44

## Goal
Head-coach tenure: games under the current head coach, first-year and interim, as a margin term (ROADMAP MOD-44).

## State
Added 2026-11-03 (backlog expansion 5), unmeasured, nothing run. Mechanism: New and interim head coaches change scheme and culture on a time scale the opener rating does not update. Fitted term: tenure bucket difference (first season, interim, established) in the margin ridge.

## Tried
Novelty check only. Nearest: PER-07 (coordinator changes, done), SIM-14 (aggression latent); grep 'tenure' found none. Differs: Head-coach tenure rather than coordinator change.

## Next
First unit: Tabulate opener residual by tenure bucket 2009-2025 with season-block bootstrap and permutation null, then LOSO add-one.

## Open
Data: nflverse schedule coach columns (free). Zero crossing closes nothing; one fitted probability selects the side; weights, cut points and alphas chosen leave-one-season-out; every cell counted as a look; unresolved results go through nfl-ats weak-signals record with a closing ground; graded at the opener (opener as model input, with intercept-only and line-move-only controls) on the same market and simple-model baselines. Look count and family declared before any sign is seen.
