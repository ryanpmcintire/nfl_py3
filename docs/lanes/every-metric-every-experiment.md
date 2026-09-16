# Every metric, every experiment — ENG-46

## Goal
Every experiment reports accuracy, Brier, log loss and margin MAE from the
same run, and the registry's existing families get the metrics they never
recorded, backfilled from preserved predictions. Done when the runner emits
all four by default and the backfill cells are in the registry.

## State
- 2026-09-16: ROADMAP row ENG-46 written. Correction recorded the same day:
  the served model is a ridge regression of the margin relative to the
  spread (`margin.py` `_target_values`, target market_residual); an earlier
  claim that it fit the binary cover label was wrong.
- Measured: 4,107 of 6,578 cells in accuracy points, 88 in margin error;
  105 `predictions.parquet` files under artifacts (55 in `artifacts/margins/`)
  carry `predicted_margin`, `home_cover_probability`, `ats_margin`.
- Unit 1 DONE 2026-09-16. Script `scripts/every_metric_backfill.py`
  (read-only); artifact `artifacts/every_metric/20260916T152725Z/`. Two
  cells recorded in family `every_metric_served_backtest_v1` (registry now
  6,582 cells). Nothing committed.

## Tried
- Served model (ridge market-residual, `artifacts/margins/20260915T183932Z`),
  2,127 games 2018-2025, verified by the parent session: margin MAE 9.952 vs
  the spread's 9.844, worse in all 8 seasons, season-block interval
  [-0.140, -0.071], P+ 0.0; recorded refuted_mechanism / wrong_sign_resolved
  for the claim that the served margin beats the spread. Its cover accuracy
  is 52.3% [50.8, 53.6] on 2,091 (companion cell, unresolved). Opposite
  signs on the two axes: the model's value is the side it leans to, not
  the number it predicts.
- Inventory: 105 predictions files, 97 supportable, 25 families; 8 families
  have no registry cells matched by source path, including the served
  model's own `margins` directory (55 files). Sign convention: the spread
  columns are already home-margin oriented (no negation). 1,090 looks.

## Next (unit 2 DONE 2026-09-16, measured)
- Runner: `backtest.summarize_predictions` now emits `model_margin_mae`,
  `model_margin_rmse`, `model_margin_games` alongside accuracy/Brier/log
  loss (None when the frame carries no margin columns; existing test only
  pins the empty-frame error, still green); `evaluation.SELECTION_METRICS`
  gains `model_margin_mae`.
- `nfl-ats weak-signals record --batch FILE` records many sibling cells
  under one family in one call (shared defaults + per-cell overrides);
  single-cell flags relaxed with manual validation preserving exit 2;
  smoked single + batch on a tmp registry.
- Backfill: `scripts/every_metric_backfill_record.py` pools each
  family's preserved predictions (game_id dedupe, latest wins), paired
  per game vs market, season-block bootstrap 2000 draws; 73 cells across
  19 families recorded (`artifacts/every_metric_backfill/20260916T194151Z/`,
  registry now 6,660). 8 files in 6 families have neither margin nor
  probability columns, nothing to backfill. A duplicate invocation
  correctly refused without --replace.
- My one wrong SE offer (fraction vs points) was caught by the
  validator's plausibility floor and corrected with --replace; noted.
- Unit 3 (pool + findings page read the margin family) not started.

## Open
- Whether backfilled cells count as new looks (they are new metrics on old
  looks; proposal: same family, flagged `backfilled` in notes, not counted).
