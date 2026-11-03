# MOD-51

## Goal
Served-base re-test of the MOD-40 and MOD-42 to MOD-47 pregame terms graded on the raw model probability (ROADMAP MOD-51).

## State
Added 2026-11-03 (backlog expansion 8), not started, nothing run. Mechanism: Memory served-probability-vs-raw-model records that many 2026-11-03 add-on units used the raw model probability as served. A term that looked null or positive against the raw base may change against the four-term served probability. Fitted term: the same declared term per row, refit on the served logit.

## Tried
Novelty check only. Nearest: MOD-41 unit 3 (served rerun done), MOD-47 unit 2, ENG-51. Differs: reruns the other terms on the true served base; MOD-41 already was.

## Next
First unit for one agent: List which of MOD-40, MOD-42, MOD-43, MOD-44, MOD-45, MOD-46 and MOD-47 used the raw base (read their scripts), rerun only those terms on out_of_season_home_probability with one declared family, and report gain, decisive record and which conclusion changes against the raw-base cell.

## Open
Rules: Zero crossing closes nothing; one fitted probability selects the side; weights, cut points and alphas chosen leave-one-season-out; every cell counted as a look; unresolved results go through nfl-ats weak-signals record with a closing ground; graded at the opener on the served base (out_of_season_home_probability from the latest artifacts/pick_probability run, with raw-model, intercept-only and line-move-only controls) on the same market and simple-model baselines. Look count and family declared before any sign is seen.
