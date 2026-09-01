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

Designed and frozen; **not run**. Execution is one session of work: the
module (`src/nfl_ats/totals.py`), the CLI (`nfl-ats totals-backtest`), the
tests above, one backtest run, the registry record, and the tiebreaker
wiring. ROADMAP POL-12 tracks it.
