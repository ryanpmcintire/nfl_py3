# LEAD-109

## Goal
Position-value weighting of the wording read: which absences the read prices, by position group and trailing snap share (ROADMAP LEAD-109).

## State
Added 2026-11-03 (backlog expansion 7), not started, nothing run. Mechanism: An availability gain only matters through the player's marginal value; QB and offensive line absences move margin more than depth absences. Fitted term: read availability weighted by position-group coefficients fitted out of season, replacing equal weighting. Data: LEAD-106 reads, snap counts, PER-09 ratings (local).

## Tried
Novelty check only. Nearest: MOD-42 (line continuity), SIM-16 (position-group attrition), LEAD-106. Differs: weights the language read, no new rating build.

## Next
First unit for one agent: Fit group coefficients (QB, OL, skill, front seven, secondary) leave-one-season-out on read-weighted absence, with a permutation null, and report the count of groups as looks.

## Open
Rules: Zero crossing closes nothing; one fitted probability selects the side; weights, cut points and alphas chosen leave-one-season-out; every cell counted as a look; unresolved results go through nfl-ats weak-signals record with a closing ground; graded at the opener (opener as model input, with intercept-only and line-move-only controls) on the same market and simple-model baselines. Look count and family declared before any sign is seen.
