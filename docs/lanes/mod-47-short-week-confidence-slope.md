# MOD-47

## Goal
Short-week confidence slope: the calibration slope of the model logit inside short-week games (ROADMAP MOD-47).

## State
Added 2026-11-03 (backlog expansion 7), not started, nothing run. Mechanism: XLG-10 reran with short-week accuracy up but log loss worse, the signature of probabilities too confident rather than wrong sides. Fitted term: short-week interaction on the model logit (slope), not a side rule. Data: opener predictions with rest days (local).

## Tried
Novelty check only. Nearest: MOD-29 (hour rest), MOD-41 (workload), XLG-10. Differs: calibrates confidence, not the pick.

## Next
First unit for one agent: Fit slope and intercept inside short-week games leave-one-season-out and report the reliability table, log loss and Brier against model-only and market baselines, with the look count.

## Open
Rules: Zero crossing closes nothing; one fitted probability selects the side; weights, cut points and alphas chosen leave-one-season-out; every cell counted as a look; unresolved results go through nfl-ats weak-signals record with a closing ground; graded at the opener (opener as model input, with intercept-only and line-move-only controls) on the same market and simple-model baselines. Look count and family declared before any sign is seen.
