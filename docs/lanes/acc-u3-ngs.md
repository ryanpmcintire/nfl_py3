# acc-u3-ngs

## Goal
Ingest nflverse Next Gen Stats weekly aggregates as leak-free team inputs; feasibility then model-alone grade. Parent: docs/lanes/accuracy-ceiling-theory.md.

## State
Data + features DONE (no grading). Measured:
- Raw: data/raw/ngs/20261001T173403Z/ (ngs_passing/rushing/receiving.parquet, sha256 in data/processed/game_features_ngs.manifest.json). Seasons 2016-2026 (2026 weeks 1-3), REG+POST. Week 0 = season totals (e.g. Mahomes 2024 w0 581 att); excluded from team-week table.
- Prior-work grep: no NGS-aggregate feature in ROADMAP.md or registry/weak_signals.json (only unrelated fourth_down_aggressiveness).
- Output: data/processed/game_features_ngs.parquet (3033 games 2016-2026, 24 diff_ngs_* columns = own + opponent-allowed versions of 12 inputs, plus max_week_used_home/away). Built by scripts/mod24_u3_ngs.py (`describe` arg prints correlations).
- Rule: game in season s week w uses same-season weeks < w only, volume-weighted (attempts/targets/rush att), shrunk to last season's team value (league mean fallback) with k = 4 mean team-week volumes computed from pre-cutoff rows only. 2016 week 1 is NaN (16 games).
- Leak check: asserts max_week_used < week for every row; 12 random (season, week) cutoffs recomputed from a table truncated to weeks < w gave max abs diff 0.0.
- Descriptive corr (2020-2025, n=1613 games with Tuesday opener; read-only): r vs margin-over-opener within |0.046|, same as vs close (spread_line); strongest vs raw result is rush_yards_over_expected_per_att 0.142, CPOE 0.132, yac_above_expectation 0.154 (these mostly restate team quality already in the line, so r vs opener ~0.02).

## Tried
Fixed two bugs in-session: shrink constant used full-table volume (future volume leak, caught by truncation check); initial opener sign wrong (opener spread has spread_line sign: margin vs opener = result - opener).

## Next
Unit acc-u3b: grade the diff_ngs_* columns on the model alone (leave-one-season-out, same opener test), 24 features = 24 looks in one family declared now; record via weak-signals record. Zero crossing closes nothing.

## Open
Opener coverage 1613/1693 games for 2020-2025 (80 lack Tuesday opener). Week-of-game NGS for week w is not used by design (published after games).
