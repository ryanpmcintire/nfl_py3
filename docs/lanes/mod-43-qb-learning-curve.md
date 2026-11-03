# MOD-43

## Goal
Quarterback in-season learning curve: starts index for rookie and first-year starters as a margin term (ROADMAP MOD-43).

## State
Added 2026-11-03 (backlog expansion 5), unmeasured, nothing run. Mechanism: Young starters improve over their first dozen starts faster than the opener's season-lagged rating assumes; the gap should be largest in weeks 4-12. Fitted term: log(starts+1) times the rookie flag, difference home minus away, entering the margin ridge.

## Tried
Novelty check only. Nearest: PER-12 (return from absence), PER-11 (acquired players), MOD-22; grep 'learning curve' found only LEAD-28 text. Differs: Development within a season, not absence or arrival.

## Next
First unit: Estimate EPA per dropback versus start index for first-year starters 2009-2025, then LOSO add-one on the opener residual with a permutation null over weeks.

## Open
Data: nflverse rosters and play-by-play EPA per dropback (free). Zero crossing closes nothing; one fitted probability selects the side; weights, cut points and alphas chosen leave-one-season-out; every cell counted as a look; unresolved results go through nfl-ats weak-signals record with a closing ground; graded at the opener (opener as model input, with intercept-only and line-move-only controls) on the same market and simple-model baselines. Look count and family declared before any sign is seen.
