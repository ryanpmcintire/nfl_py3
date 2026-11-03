# POOL-10

## Goal
Best Pick placement on half-point sides: does the field weight its single Best Pick toward the hook side (ROADMAP POOL-10).

## State
Added 2026-11-03 (backlog expansion 7), not started, nothing run. Mechanism: The Best Pick carries extra weight; a field that over-picks hook sides may concentrate that weight there. Fitted term: hook indicator in the Best Pick share model inside the standing-aware leverage. Data: captured field Best Pick shares if present (local).

## Tried
Novelty check only. Nearest: POOL-08, POOL-05 (slot and team ownership), confidence-best-pick-unification lane. Differs: Best Pick weight, not side share.

## Next
First unit for one agent: Count captured weeks with Best Pick shares before any test; if fewer than eight, record the count and defer.

## Open
Rules: Zero crossing closes nothing; one fitted probability selects the side; weights, cut points and alphas chosen leave-one-season-out; every cell counted as a look; unresolved results go through nfl-ats weak-signals record with a closing ground; graded at the opener (opener as model input, with intercept-only and line-move-only controls) on the same market and simple-model baselines. Look count and family declared before any sign is seen.
