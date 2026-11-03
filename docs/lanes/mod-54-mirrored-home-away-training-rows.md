# MOD-54

## Goal
Mirrored home and away training rows: fit each game from both teams' views so the base is exactly antisymmetric (ROADMAP MOD-54).

## State
Added 2026-11-03 (backlog expansion 9), not started, nothing run. Mechanism: A game currently gives one row oriented home minus away; the home-field offset and any orientation-specific noise are fitted together with team strength. Measured mechanism: each game enters twice (home view and sign-flipped away view) with the home-field offset as its own column, so the penalty sees orientation-free pairs. Fitted term: Base model refit on the mirrored table; compared on the same games with the same penalty selection.

## Tried
Novelty check only. Nearest: MOD-23 units 9 and 16 (per-side targets, estimator class), unit 17 (block penalties). Differs: Nothing in registry/weak_signals or docs/lanes fits on mirrored rows; the identity check may show the base is already equivariant, which is a valid result.

## Next
First unit for one agent: Read the feature builder to confirm whether inputs are differences or home and away blocks, build the mirrored table in a scratch script, and report whether predictions equal the base when the table is already antisymmetric (identity check) before reporting any record difference.

## Open
Rules: Zero crossing closes nothing; one fitted probability selects the side; every penalty, weight and cut point chosen leave-one-season-out; every arm counted as a look; unresolved results go through nfl-ats weak-signals record with a closing ground; graded on the fair opener test (base 802-701 in 2020-25 and 1142-1089 in 2011-19 reproduced first, opener not a model input) on the same market and simple-model baselines. Pinned model source stays untouched: run in a scratch harness script, no src edit unless the result is adopted. Look count and family declared before any sign is seen. Data: Base input table and feature builder (local).
