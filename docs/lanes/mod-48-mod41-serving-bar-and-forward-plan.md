# MOD-48

## Goal
MOD-41 serving bar and forward-weeks plan: a pre-declared decision rule and the week it can first be met (ROADMAP MOD-48).

## State
Added 2026-11-03 (backlog expansion 8), not started, nothing run. Mechanism: MOD-41 residual load is frozen (coefficient -0.0439) and scored forward weekly, but no rule says what matured result serves it. Fitted term: the frozen term enters the one served probability only if the pre-declared bar is met; the bar is a serving rule, not a closing rule, and a miss closes nothing.

## Tried
Novelty check only. Nearest: ENG-48 (generic anytime-valid monitor), ENG-50 (online stacker), MOD-41 lane unit 3. Differs: a MOD-41-specific pre-declared serving rule with a power timeline, not the generic monitor or the combination.

## Next
First unit for one agent: Before more weeks score, write the bar: paired forward log-loss gain against the served probability with probability positive and decisive record, then simulate from the 2020-25 per-week gain variance how many forward games and which week the bar is reachable under the LOSO-sized effect (.004) and under half of it, and file the bar with the freeze hash.

## Open
Rules: Zero crossing closes nothing; one fitted probability selects the side; weights, cut points and alphas chosen leave-one-season-out; every cell counted as a look; unresolved results go through nfl-ats weak-signals record with a closing ground; graded at the opener on the served base (out_of_season_home_probability from the latest artifacts/pick_probability run, with raw-model, intercept-only and line-move-only controls) on the same market and simple-model baselines. Look count and family declared before any sign is seen.
