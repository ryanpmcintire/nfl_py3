# MOD-49

## Goal
MOD-41 feature-construction sensitivity: window length, play definition and norm, all declared before signs (ROADMAP MOD-49).

## State
Added 2026-11-03 (backlog expansion 8), not started, nothing run. Mechanism: The residual-load term depends on construction choices made once (window 3, pass and run plays only, season-to-date norm, tempo parts removed by regression). A real mechanism should survive reasonable alternatives; one that vanishes outside window 3 is a fit to the construction. Fitted term: the same single coefficient on each declared variant, compared leave-one-season-out on the served base.

## Tried
Novelty check only. Nearest: MOD-41 units 1-3 (window 3 only), MOD-29, SIM-13. Differs: varies the construction, not the mechanism or the injury mediation already run.

## Next
First unit for one agent: Declare the family first (windows 2, 3, 4, 5; plays with and without kneels and spikes; norm as season-to-date versus prior-season mean; seconds of possession instead of plays; same-week future-game placebo), count the looks, and report the gain, sign and coefficient per variant on the served base with the held-out gap.

## Open
Rules: Zero crossing closes nothing; one fitted probability selects the side; weights, cut points and alphas chosen leave-one-season-out; every cell counted as a look; unresolved results go through nfl-ats weak-signals record with a closing ground; graded at the opener on the served base (out_of_season_home_probability from the latest artifacts/pick_probability run, with raw-model, intercept-only and line-move-only controls) on the same market and simple-model baselines. Look count and family declared before any sign is seen.
