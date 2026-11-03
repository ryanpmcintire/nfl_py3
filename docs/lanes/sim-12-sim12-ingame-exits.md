# SIM-12 In-game starter exits

## Goal
Measure in-game starter exit hazard and its drive-level cost for the simulator. ROADMAP row SIM-12.

## State
Added 2026-11-03 as an unmeasured backlog row; nothing run, no registry window spent. Mechanism: The simulator draws a fixed lineup; real starters leave mid-game and the remaining drives run on backups. Estimate exit hazard by position and game state from participation, then the efficiency change of the next drives, as a named mechanism. Data: nflverse participation and pbp 2016-2025 (game_features_player_participation).

## Tried
Novelty check only (grep of ROADMAP, docs/lanes incl. done, docs/*.md, registry/weak_signals). Nearest prior: sim09_personnel_catalogue.md B2 (listed, never measured), PER-10 injury scenario mixture (pregame mixture, not in-game), inactives_channel.md. No lane measures the in-game exit rate.

## Next
Build the exit table from participation (first versus last offensive snap per starter). First unit: First unit: measure per-game probability a team's starting QB and each OL starter exits, by position, and the next-drive EPA change against matched drives; no engine edit.
Research rules: Zero crossing closes nothing; one fitted probability selects the side; weights, cut points and alphas chosen leave-one-season-out; every cell counted as a look; unresolved results go through nfl-ats weak-signals record with a closing ground; graded at the opener on the same market and simple-model baselines.

## Open
Participation has no injury cause; ejections and blowout rests will be mixed with injuries and must be separable by game state.
