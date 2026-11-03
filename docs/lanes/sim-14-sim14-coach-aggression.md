# SIM-14 Coach aggression latent

## Goal
Measure the persistence of coach decision aggressiveness for the simulator. ROADMAP row SIM-14.

## State
Added 2026-11-03 as an unmeasured backlog row; nothing run, no registry window spent. Mechanism: The simulator uses league-level fourth-down choice; coaches differ persistently. A per-coach go rate over a league state model (field position, distance, score, time, kick range), shrunk across seasons, drives the simulator decision and is checked against held-out seasons. Data: nflverse pbp 2009-2025 with schedules home_coach and away_coach.

## Tried
Novelty check only (grep of ROADMAP, docs/lanes incl. done, docs/*.md, registry/weak_signals). Nearest prior: SIM-04 and mod25e fourth-down choice (league level), sim09_personnel_catalogue.md A9 (listed), docs/closure_audit.md:229 (found only roadmap prose for a coach aggression term), PER-07 (coordinator changes, not decisions).

## Next
Fit the go model by season and compute coach residuals with a split-half null. First unit: First unit: fit the league go model leave-one-season-out, compute coach residual go rate, and report coach split-half reliability; only then wire a simulator arm.
Research rules: Zero crossing closes nothing; one fitted probability selects the side; weights, cut points and alphas chosen leave-one-season-out; every cell counted as a look; unresolved results go through nfl-ats weak-signals record with a closing ground; graded at the opener on the same market and simple-model baselines.

## Open
Coach-by-coordinator attribution is ambiguous in midseason changes; use the schedule head coach and record the limit.
