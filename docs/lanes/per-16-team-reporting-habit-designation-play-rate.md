# PER-16

## Goal
Team reporting habit: play rate given the same designation, by team and coach, shrunk, as the availability prior (ROADMAP PER-16).

## State
Added 2026-11-03 (backlog expansion 9), not started, nothing run. Mechanism: Teams word and use designations differently: one staff lists a player Questionable and he plays, another lists the same word and he sits. A team-keyed play rate by designation, shrunk to the league rate, may carry information the league-wide ladder and the wording read both average away. Fitted term: Expected-availability difference with team-keyed shrunken play rates replacing the league ladder, entering the existing availability input and then the calibrated probability.

## Tried
Novelty check only. Nearest: LEAD-106 (wording by player), LEAD-109 (position value), PER-15 (return snap fraction). Differs: Keys the availability prior to the reporting team and coach, not to wording or position.

## Next
First unit for one agent: For each team-season and head coach, compute the share of Questionable and Doubtful starters who played at least half the snaps, report odd and even split-half reliability and the shrinkage factor, and score a team-keyed prior against the league ladder on realised snaps (Brier) out of season.

## Open
Rules: Zero crossing closes nothing; one fitted probability selects the side; weights, cut points and alphas chosen leave-one-season-out; every cell counted as a look; unresolved results go through nfl-ats weak-signals record with a closing ground; graded at the opener on the served base (out_of_season_home_probability from the latest artifacts/pick_probability run, with raw-model, intercept-only and line-move-only controls) on the same market and simple-model baselines. Look count and family declared before any sign is seen. Data: Injury reports and snap counts 2012-2025 (free), coach table from PER-07.
