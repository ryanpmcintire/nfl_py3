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

## Update 2026-11-03 (unit 1 done)
scripts/mkt21_line_path.py, artifacts/mkt21/ (line_path.json, path_features.parquet, record_batch.json). Data: 2023-2025 only (2020-2022 have no per-book snapshots), hourly (SKY-04 cadence), 10-16 books, 798 of 799 non-push games with paths. Window Tuesday noon ET to min(kickoff, Sun 4 PM ET); consensus = per-snapshot median of forward-filled books.
Family mkt21_line_path_shape, 3 looks, each feature x signed by move direction, one term on out_of_season probability, LOSO. Corr with |net|: reversals 0.14, last-move 0.33, efficiency 0.47. OOS Brier gain -0.00087 / -0.00058 / -0.00050, p+ 0.15 / 0.07 / 0.08; in-sample ~0 to +0.0007 (gap 0.0005-0.0015). Decisive new v base: 14-18, 13-14, 11-12. Fold coefs unstable. Recorded 3 cells, unresolved_below_power.
Next: re-score on 2026 once hourly captures exist; sub-hourly tape (capture_odds_tape) for reversals.
