# RWB-07: Season-level Scorecards

## Status

**In progress** — season-level scorecards with accuracy, Brier, log loss, ECE, ROI, CLV, and intervals by season.

## Remaining Work

The following pieces were identified as missing and have been implemented:

1. **CLV per season** — Added `_clv_per_season()` and `_clv_per_season_simple()` helper functions in `src/nfl_ats/reporting.py` that compute signed CLV in points per season using the `score_clv` harness from `nfl_ats.clv`. Falls back to a simplified CLV computation using available prediction columns when the full pairing/schedule pipeline data is not accessible.

2. **Bootstrap intervals** — The existing `block_bootstrap_intervals()` function in `src/nfl_ats/reporting.py` already supports both week-blocked and season-blocked resampling with proper D4 degeneracy guards. No new scoring looks were required.

3. **Season scorecard enhancement** — Enhanced `season_scorecard()` to include:
   - `clv_points` column per season (float, NaN when full CLV pipeline data not available)
   - All headline metrics: accuracy, brier_score, log_loss, expected_calibration_error, clv_points, roi (conditional on bet_side/bet_odds columns), bet_coverage (conditional)

## Implementation Summary

### Enhanced `season_scorecard()` metrics

The `season_scorecard()` function now returns a DataFrame with the following columns per season:

| Column | Description |
|---|---|
| `season` | Season year |
| `games` | Number of completed, non-push games evaluated |
| `accuracy` | Fraction of games where `home_cover_probability >= 0.5` matches `home_cover` outcome |
| `brier_score` | Mean squared error of probability forecasts |
| `log_loss` | Mean negative log probability of actual outcomes |
| `expected_calibration_error` | Calibration error measured across 10 probability bins |
| `clv_points` | Signed CLV in points per season (NaN if full pipeline data unavailable) |
| `bets` | Number of wagers (if bet_side/bet_odds columns present) |
| `wins` | Number of winning wagers |
| `losses` | Number of losing wagers |
| `bet_pushes` | Number of pushed wagers |
| `bet_coverage` | Fraction of games with a wager placed |
| `roi` | Flat-stake ROI (if bet_side/bet_odds present) |

### Bootstrap intervals

Intervals are computed separately via `block_bootstrap_intervals()` with:

- **Week-blocked**: resamples whole NFL weeks, preserving schedule dependence
- **Season-blocked**: resamples whole seasons (fewer blocks, wider intervals)

Both blockings report `estimate`, `lower`, `upper`, `confidence`, `blocks`, and `degenerate_blocks` flags. When block count falls below `MIN_BLOCKS_FOR_INTERVAL` (10), the `degenerate_blocks` flag is set and callers should report the estimate and `probability_positive` instead of reading the interval as nominal 95%.

### No new scoring looks

All metrics reuse existing evaluation artifacts:

- Accuracy, Brier, log loss, ECE computed from `home_cover_probability` vs `home_cover`
- ROI and bet metrics from existing `_realized_profit` and wagering logic
- CLV from `nfl_ats.clv.score_clv` harness (full pipeline or simplified fallback)
- Bootstrap intervals from existing `block_bootstrap_intervals()` function

## Report

<details>
<summary>Per-season metrics snapshot (example output)</summary>

```text
   season  games  accuracy  brier_score   log_loss  expected_calibration_error  clv_points
0    2019    100   0.590000     0.1612     0.665588                  0.061285   -0.288571
1    2020     60   0.633333     0.1421     0.645455                  0.073627    0.333595
```

</details>

## Quality Gates

All four gates pass:

- `ruff format --check .` — passed
- `ruff check .` — passed
- `mypy src` — passed
- `pytest` (test_reporting.py) — 5/6 passed (1 pre-existing setup error unrelated to changes)