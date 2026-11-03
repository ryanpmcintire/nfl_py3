# MOD-52

## Goal
MOD-41 stability profile: coefficient and gain by season, week block and spread band before any serving read (ROADMAP MOD-52).

## State
Added 2026-11-03 (backlog expansion 8), not started, nothing run. Mechanism: A real load effect should show a stable coefficient across seasons and weeks; a one-season artifact or a late-season-only effect would change how it is served. Fitted term: season and week-block interaction on the frozen coefficient, not a gate.

## Tried
Novelty check only. Nearest: MOD-41 units 1-3 (pooled and LOSO folds only), MOD-47. Differs: a stability profile of one frozen term, with no new feature.

## Next
First unit for one agent: Report the frozen coefficient and log-loss gain per season 2020-25, per week block (5-8, 9-13, 14-18), and per spread tercile as a stability profile with permutation nulls on the lopsided splits, counting every cell as a look.

## Open
Rules: Zero crossing closes nothing; one fitted probability selects the side; weights, cut points and alphas chosen leave-one-season-out; every cell counted as a look; unresolved results go through nfl-ats weak-signals record with a closing ground; graded at the opener on the served base (out_of_season_home_probability from the latest artifacts/pick_probability run, with raw-model, intercept-only and line-move-only controls) on the same market and simple-model baselines. Look count and family declared before any sign is seen.
