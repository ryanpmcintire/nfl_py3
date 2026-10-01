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
NGS model-alone is done and negative-leaning (U3b). Next: no NGS arm adds to the base; only revisit with a forward log. Pool across families via weak-signals pool when asked.

## U3b result (measured, scripts/mod24_u3b.py, artifacts/mod24_u3b/{results.csv,json,run.log,weak_signals_batch.json})
Base reproduced 802-701. Arms declared before scoring: a all24, b compact6 own, c compact6 + allowed. 3 arms x 5 metrics x 2 training variants + 6 coefficient reads = about 36 looks. Zero-fill: a 798-705 acc -0.27 P+ 0.29; b 791-712 -0.73 P+ 0.11; c 776-727 -1.73 P+ 0.012 (CI -3.07..-0.19). Recal log loss diff (positive better): a -0.00045 P+ 0.29, b -0.00045 P+ 0.10, c -0.00047 P+ 0.19; Brier same sign; RPS all negative (P+ 0.20/0.24/0.08). Flip W-L (arm-correct): a 153-157, b 70-81, c 110-136. Train-2016+ variant (own base): acc +0.60/+0.53/-0.67, LL P+ 0.012/0.081/0.014 (arms worse). Arm b coefficients mostly sign-stable (time to throw, aggressiveness, RYOE all negative 6/6; 8+ defenders positive 5/6; separation and CPOE mixed). 3 LL cells recorded (family mod24_ngs_model_alone, unresolved_below_power). Wrong-sign not claimed as closure (6 season blocks).

## Open
Opener coverage 1613/1693 games for 2020-2025 (80 lack Tuesday opener). Week-of-game NGS for week w is not used by design (published after games).
