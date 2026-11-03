# MOD-38 Kicker wind decay

## Goal
Kicker wind decay: forecast wind times each kicker distance profile as expected field-goal points lost ROADMAP row MOD-38.

## State
Added 2026-11-03 as an unmeasured backlog row; nothing run, no registry window spent. Mechanism: Three points is the largest key number. Wind lowers make rates at long distances and changes go or kick choices, so the same forecast wind removes different amounts of expected points from different kickers and shifts margin mass off 3. SIM-15 covers calls, not kick outcomes. Data: nflverse pbp kick distance, result, kicker id and wind 2009-2025; dome placebo.

## Tried
Novelty check only: Nearest: special_teams_battery.md and st_player_ratings.md (kicker quality without wind), kicker_change_underdog (registry), LEAD-75 (wind revision times passing exposure), SIM-15 (calls). Differs: per-kicker wind decay and its key-number consequence.

## Next
First unit: Logistic make model on distance and wind with a kicker random slope, test slope variance against a within-kicker permutation null, then convert to expected points lost per team at the Tuesday forecast and check the margin and total residual at the opener.
Rules: Zero crossing closes nothing; one fitted probability selects the side; weights, cut points and alphas chosen leave-one-season-out; every cell counted as a look; unresolved results go through nfl-ats weak-signals record with a closing ground; graded at the opener on the same market and simple-model baselines.

## Open
Look count and family to be declared before any sign is seen. Nothing here closes on zero crossing; record unresolved results with nfl-ats weak-signals record.
