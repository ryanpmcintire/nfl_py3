# MOD-33 Mid-game starter exits in last week game as a Tuesday signal the Sunday-night opener cannot contain

## Goal
Mid-game starter exits in last week game as a Tuesday signal the Sunday-night opener cannot contain ROADMAP row MOD-33.

## State
Added 2026-11-03 as an unmeasured backlog row; nothing run, no registry window spent. Mechanism: Whoever leaves a Sunday game early is invisible to an opener posted that night and reaches the injury report only on Wednesday. Participation data show it by Monday. SIM-12 models exit hazard inside a game; this uses last week observed exits as a pregame availability term for next week. Data: nflverse participation 2016-2025 (offense and defense player ids per play) and snap_counts; injury designations for timing from the injury feed archive.

## Tried
Novelty check only: Nearest: qb_exit_mixture.md and registry/weak_signals/qb_exit_mixture.json (quarterback exits inside the model), SIM-12 (in-game hazard for the simulator), injury_timing_audit.md and injury_trajectory_leads.md (designation timing), snap-counts-2026.md (starter key sets). Differs: all starters, inferred from in-game participation before any designation.

## Next
First unit: Define an exit from each starter own play-participation trajectory against his team-position norm (cut point chosen leave-one-season-out), measure how often it precedes an Out or Doubtful designation versus never being designated, and then the opener residual of next week margin for undesignated exits with a within-week permutation null.
Rules: Zero crossing closes nothing; one fitted probability selects the side; weights, cut points and alphas chosen leave-one-season-out; every cell counted as a look; unresolved results go through nfl-ats weak-signals record with a closing ground; graded at the opener on the same market and simple-model baselines.

## Open
Look count and family to be declared before any sign is seen. Nothing here closes on zero crossing; record unresolved results with nfl-ats weak-signals record.
