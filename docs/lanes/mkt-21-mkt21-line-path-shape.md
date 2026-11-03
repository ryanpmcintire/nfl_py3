# MKT-21 Line-path shape

## Goal
Test whether the shape of the line path carries pick information beyond net movement. ROADMAP row MKT-21.

## State
Added 2026-11-03 as an unmeasured backlog row; nothing run, no registry window spent. Mechanism: Two games with the same net move can differ: one drifted steadily, another jumped and reverted. Features: count of direction changes, time share spent at the final number, and distance travelled over net move; fitted terms beside the existing move term. Data: Existing multi-book captures (data/raw odds snapshots, capture_odds_tape).

## Tried
Novelty check only (grep of ROADMAP, docs/lanes incl. done, docs/*.md, registry/weak_signals). Nearest prior: LEAD-74 within-spread price revisions, SKY-04 lead-lag and sharpness term (unresolved), market_move_decomposition.py (move decomposition by key-number crossings). Grep for reversal count, direction change, move count finds zero hits.

## Next
Extract per-game quote paths from the Tuesday-to-lock captures and compute the features. First unit: First unit: build the three path features for the captured weeks 2020-2025, report their correlation with net move, then a LOSO term on the four-term probability.
Research rules: Zero crossing closes nothing; one fitted probability selects the side; weights, cut points and alphas chosen leave-one-season-out; every cell counted as a look; unresolved results go through nfl-ats weak-signals record with a closing ground; graded at the opener on the same market and simple-model baselines.

## Open
Capture cadence changed across seasons; only seasons with sub-daily snapshots are usable.
