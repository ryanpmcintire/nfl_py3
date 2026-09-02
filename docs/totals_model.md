# Totals model — predeclared design (not yet run)

Predeclared 2026-09-01, before any totals model has been fit on this data.
The registry holds no totals family (measured 2026-09-01: no signal name or
family matches `total` in the 615-entry registry), so every choice below is
frozen before any sign is seen. **Nothing in this file is a result.** When
the backtest runs, its numbers go to the registry via `nfl-ats weak-signals
record`, never only into prose here.

Owner intent (2026-09-01): the pool's tiebreaker is the final score of the
week's last game; `nfl-ats tiebreaker` ships the market-implied baseline; a
"proper model to predict the over/under" is the requested follow-up. Queued,
not brushed off — this is the executable spec.

## Why this architecture

Mirror the ATS side's winning structure exactly: the market line as prior,
a ridge model on the *residual*, chronological evaluation, and a measured
blend weight instead of an override. The margin-side measurement that
motivates modest expectations (measured 2026-09-01, recorded in
`nfl_ats.tiebreaker.MODEL_RESIDUAL_WEIGHT`'s docstring): the production
model's own margin point-estimate is WORSE than the market's (MAE 10.00 vs
9.91 on 1,537 opener-graded games) and earns blend weight 0.2. Expect the
same shape for totals (inferred, not measured); the regime runs regardless —
declining to run it would be an active bet that the totals residual is worth
zero.

## Frozen contract

- **Target**: `total_residual = (home_score + away_score) - total_line`.
- **Population**: every game in the newest `data/raw/*/schedules.parquet`
  with non-null `home_score`, `away_score`, `total_line` (4,630 games
  2009–2025, measured 2026-09-01), joined to
  `data/processed/game_features.parquet` (4,902 rows incl. 2026 Week 1,
  measured) on `game_id`. Primary read on `game_type == "REG"`; playoffs
  reported separately (FND-15 lineage), never silently pooled.
- **Feature allowlist** (explicit, from the canonical table's 98 columns,
  read 2026-09-01 — nothing outside this list enters the fit):
  `total_line`, `spread_line`, `rest_diff`, `neutral_site`, `div_game`,
  `temp`, `wind`, `week_sin`, `week_cos`, `elo_diff`, `elo_home_win_prob`,
  `home_team_games`, `away_team_games`, and the home/away efficiency
  families: `{home,away}_off_{epa_per_play, pass_epa_per_play,
  rush_epa_per_play, cpoe, yards_per_play, turnover_rate, sack_rate}`,
  `{home,away}_def_{epa_per_play, pass_epa_per_play, rush_epa_per_play,
  yards_per_play, takeaway_rate, sack_rate}`, `{home,away}_point_diff`.
  **Excluded on purpose**: identifiers/outcomes (`result`, `ats_margin`,
  `home_cover`, scores); the `diff_*` columns (totals ride sums, not
  differences — ridge forms sums from the home/away columns directly); the
  `*_ats_residual`, graph/schedule/bias/surface columns (spread-oriented
  constructs; a second wave may screen them as totals features, separately
  declared). These are the already-leakage-tested pregame families — no new
  feature family, so no new leakage test is owed; the walk-forward guard
  test below still is.
- **Pipeline**: `SimpleImputer(strategy="median", add_indicator=True)` →
  `StandardScaler` → `Ridge(alpha=10.0)` — production's exact recipe and
  constant (`margin.py:377-387`, read 2026-09-01). Alphas {1, 100} reported
  alongside for transparency; the primary is 10, fixed here so no tuning
  touches the test stream.
- **Protocol**: expanding-window walk-forward by chronological (season,
  week): train on games strictly before the target week, predict the week,
  `min_train_games = 500` (production's constant). Prediction-level output
  preserved to `artifacts/totals_backtest/<stamp>/` per the research
  invariant.
- **Metrics**: MAE and RMSE of (a) market total alone, (b) raw model total,
  (c) blend `total_line + k * predicted_residual` for k in {0.0, 0.1, …,
  1.0}. Decision = MAE-minimizing k. Per-season MAE deltas;
  `clv.week_blocked_bootstrap` on the paired per-game |error| difference
  (market vs chosen blend); report `probability_positive`, never
  "contains zero".
- **Recording**: one registry entry (family `totals_market_residual`,
  effect = MAE improvement in total points, negative-is-better stated
  explicitly). If `weak-signals record` rejects the units, that is a units
  problem to fix in the call, never a reason to soften the verdict rules.
- **Integration**: `nfl_ats.tiebreaker` gains `TOTALS_RESIDUAL_WEIGHT` set
  from the measured k — including k = 0 if that is what the sweep says —
  same derived-constant pattern as `MODEL_RESIDUAL_WEIGHT`, plus a
  "model total view" line in the report mirroring the margin one.
- **Required tests**: walk-forward guard (no training row from the target
  week or later — synthetic fixture where violating the guard changes the
  prediction), allowlist enforcement (a renamed/extra column never enters
  the design matrix), blend math, and the tiebreaker wiring.

## Status

**RUN 2026-09-01.** The module (`src/nfl_ats/totals.py`), the CLI
(`nfl-ats totals-backtest`), the four required tests (`tests/test_totals.py`),
the backtest, the registry record (`totals_market_residual_blend`) and the
tiebreaker wiring (`TOTALS_RESIDUAL_WEIGHT = 0.1`) all shipped in one
session. Numbers below; the contract above is unchanged from its frozen form.
ROADMAP POL-12 tracks it.

## Results (added after the run, 2026-09-01)

Produced by `.\.tools\uv.exe run --no-sync nfl-ats totals-backtest`.
Prediction-level output: `artifacts/totals_backtest/20260901T184010Z/`
(`predictions.parquet`, `paired_errors.parquet`, `results.json`,
`metadata.json`). Registry entry: `totals_market_residual_blend`, family
`totals_market_residual`, league nfl (registry at 617 signals after the
write).

### Population actually used

4,630 lined finals joined to the feature table (exactly the number the
contract predeclared), of which **3,935 regular-season games, 2010-2025,
were scored**. The 496 unscored regular-season games are the warm-up the
`min_train_games = 500` floor requires: no week is predicted until 500 prior
games are banked, which happens partway through 2010. Playoffs: 188 scored
games, reported separately below and never pooled into the primary. All 41
allowlist columns were present in `data/processed/game_features.parquet` —
none was absent, and nothing was substituted.

### MAE / RMSE, regular season (3,935 games)

| arm | MAE | RMSE |
| --- | --- | --- |
| (a) market total alone | 10.4249 | 13.1697 |
| (b) raw model total (k = 1.0) | 10.5495 | 13.2749 |
| (c) chosen blend, k = 0.1 | 10.4241 | 13.1650 |

The raw model total is **worse** than the market total as a point estimate,
by 0.125 MAE points — the same shape the margin side found (`MODEL_RESIDUAL_WEIGHT`'s
docstring: 10.00 vs 9.91).

### Blend sweep, `total_line + k * predicted_residual`

| k | MAE | RMSE | MAE improvement vs market |
| --- | --- | --- | --- |
| 0.0 | 10.4249 | 13.1697 | +0.0000 |
| **0.1** | **10.4241** | **13.1650** | **+0.0008** |
| 0.2 | 10.4260 | 13.1638 | -0.0011 |
| 0.3 | 10.4310 | 13.1659 | -0.0061 |
| 0.4 | 10.4387 | 13.1713 | -0.0138 |
| 0.5 | 10.4486 | 13.1802 | -0.0237 |
| 0.6 | 10.4615 | 13.1925 | -0.0366 |
| 0.7 | 10.4785 | 13.2081 | -0.0536 |
| 0.8 | 10.4993 | 13.2270 | -0.0744 |
| 0.9 | 10.5233 | 13.2493 | -0.0984 |
| 1.0 | 10.5495 | 13.2749 | -0.1246 |

Decision = MAE-minimising k = **0.1**. Alphas reported alongside, never used
to pick: alpha 1 also chooses k = 0.1 (+0.0009), alpha 100 also chooses
k = 0.1 (+0.0012). RMSE minimises at k = 0.2 rather than 0.1; the contract
fixed MAE as the decision metric before the run, so 0.1 it is.

### Per-season MAE deltas (positive = blend better)

| season | games | market MAE | blend MAE | delta |
| --- | --- | --- | --- | --- |
| 2010 | 16 | 10.844 | 10.787 | +0.0570 |
| 2011 | 256 | 9.379 | 9.383 | -0.0041 |
| 2012 | 256 | 10.396 | 10.407 | -0.0104 |
| 2013 | 256 | 11.111 | 11.088 | +0.0233 |
| 2014 | 256 | 10.770 | 10.752 | +0.0171 |
| 2015 | 256 | 10.529 | 10.529 | +0.0002 |
| 2016 | 256 | 9.918 | 9.935 | -0.0168 |
| 2017 | 256 | 11.162 | 11.142 | +0.0198 |
| 2018 | 256 | 10.598 | 10.595 | +0.0031 |
| 2019 | 256 | 10.828 | 10.838 | -0.0094 |
| 2020 | 256 | 10.146 | 10.159 | -0.0130 |
| 2021 | 272 | 10.789 | 10.778 | +0.0107 |
| 2022 | 271 | 10.395 | 10.404 | -0.0090 |
| 2023 | 272 | 10.239 | 10.253 | -0.0135 |
| 2024 | 272 | 9.730 | 9.728 | +0.0019 |
| 2025 | 272 | 10.393 | 10.384 | +0.0093 |

Nine seasons positive, seven negative.

### Playoffs, reported separately (188 games)

Scored by the same walk-forward models, never pooled into the primary:
market MAE 10.9231, blend MAE (k = 0.1) 10.8994, delta **+0.0237** — the
same direction as the regular season and about 29x the size, on 5% of the
sample. Not a separate finding; a separate report, per the FND-15 lineage.

### Uncertainty

`nfl_ats.clv.week_blocked_bootstrap` on the paired per-game |error|
difference (market minus blend, positive = blend better), 2,000 resamples
over 261 week blocks: **+0.0008 total points, 95% [-0.0062, +0.0077],
`probability_positive` 0.583.**

### What this implies for the decision, before what is wrong with it

The pool submits a tiebreaker every week whether or not this model exists, so
the live question is not "is this significant" but "which total do we serve".
At `probability_positive` 0.583 the blend is the favourite, and the sweep's
own minimum sits at k = 0.1 rather than at k = 0.0 — serving the market alone
would be taking the 42% side of that. So the card takes the small nudge:
`TOTALS_RESIDUAL_WEIGHT = 0.1`, wired and derived, not chosen.

What the number is not: an edge. +0.0008 points on a 10.4-point error is
about one part in thirteen thousand, and the honest headline of this regime
is the *other* measurement it produced — the raw model total is 0.125 MAE
points **worse** than the market's, replicating on the totals axis what the
margin axis already showed. Side-picking skill is not point-estimate skill.
Anyone quoting this work should quote that sentence, not the +0.0008.

Classification: `unresolved_below_power`. Not `refuted_mechanism` (the
interval is not wholly on the wrong side of zero and the trait's reliability
was not measured at zero) and not `bounded_by_control` (no positive control
proven able to resolve an 0.001-point MAE effect was run).

### One thing found on the way that is worth a look (not part of the contract)

Wiring the blend exposed a hard edge in the tiebreaker's existing
neighborhood lookup, which is unrelated to the totals model itself and
predates it. `_neighborhood` selects comparable games with a hard +/-1.5-point
total window, and quoted totals are quantized to half points, so a blend nudge
far smaller than the quantum can drop a whole bucket. Measured on the live
2026 Week 1 board: at the market total 43.0 the neighborhood holds 259 games
(buckets 41.5 through 44.5); at the blended 43.0421 it holds 221 — the 41.5
bucket falls outside — and the median actual total moves 43 -> 41, so the
published closest-total guess moves KC 23 / DEN 20 -> KC 22 / DEN 19 even
though the model argued the total should be HIGHER. The margin axis has the
same edge and has always had it (`MODEL_RESIDUAL_WEIGHT` shifts the +/-1.0
margin window the same way). This session did not change `_neighborhood` —
that is outside this work package and would move the margin path too — but
pinned the behaviour in `tests/test_totals.py::test_totals_blend_can_move_the_neighborhood_across_a_line_bucket`
so it cannot drift silently. A soft or grid-snapped window is the obvious
follow-up.
