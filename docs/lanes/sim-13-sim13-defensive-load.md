# SIM-13 Defensive snap load

## Goal
Measure whether plays defended so far shifts late-game scoring beyond team strength. ROADMAP row SIM-13.

## State
Added 2026-11-03 as an unmeasured backlog row; nothing run, no registry window spent. Mechanism: A defence that has been on the field for many plays against tempo has fewer effective snaps late; the simulator treats each drive as independent of plays defended. Measure late-game points per drive against plays defended so far, team strength held fixed. Data: nflverse pbp 2009-2025 (local).

## Tried
Novelty check only (grep of ROADMAP, docs/lanes incl. done, docs/*.md, registry/weak_signals). Nearest prior: sim09_personnel_catalogue.md B1 (listed only), mod25e-e182-pace (pace as a rate), mod25e-e175 (leader Q4 gap). No measured fatigue term.

## Next
Compute plays-defended and possession imbalance per drive and fit the contrast. First unit: First unit: regression of fourth-quarter points per drive on defensive plays so far and time of possession imbalance, opponent and defence strength fixed, season-week block bootstrap.
Research rules: Zero crossing closes nothing; one fitted probability selects the side; weights, cut points and alphas chosen leave-one-season-out; every cell counted as a look; unresolved results go through nfl-ats weak-signals record with a closing ground; graded at the opener on the same market and simple-model baselines.

## Open
Possession imbalance is partly caused by strength and score; the contrast must condition on both.
