# SIM-19

## Goal
Trailing risk shape: pass depth, sack rate and turnover rate as the deficit grows and time shrinks (ROADMAP SIM-19).

## State
Added 2026-11-03 (backlog expansion 5), unmeasured, nothing run. Mechanism: Catch-up football raises variance through deeper targets, more sacks and interceptions. The simulator needs the shape of risk, not only the go rate. Enters SIM-10 as deficit-conditioned play outcomes; the pregame side gets a trailing-team variance term in the line-conditional margin distribution.

## Tried
Novelty check only. Nearest: SIM-09 Tier B (stakes hurt Q4 slope), SIM-14 (go rate); grep 'trailing' found only MOD-18 text. Differs: Play-level risk shape rather than fourth-down go rate.

## Next
First unit: Fit aDOT, sack, interception and explosive rates versus deficit and time bins; test whether SIM-10 reproduces the Q4 slope with them.

## Open
Data: Local play-by-play (free). Zero crossing closes nothing; one fitted probability selects the side; weights, cut points and alphas chosen leave-one-season-out; every cell counted as a look; unresolved results go through nfl-ats weak-signals record with a closing ground; graded at the opener (opener as model input, with intercept-only and line-move-only controls) on the same market and simple-model baselines. Look count and family declared before any sign is seen.
