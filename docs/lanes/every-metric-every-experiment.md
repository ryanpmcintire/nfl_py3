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
- Unit 3 DONE 2026-09-16, details below (measured this session).

- Unit 3 DONE 2026-09-16 (measured this session). The Signal Ledger
  renders the three improvement units in words
  (`signal_ledger._UNIT_META`: points of average-error improvement,
  Brier-score / log-loss points of improvement, same words as the
  findings recent-activity list); the findings evergreen cites the
  `mae_improvement` pool beside accuracy; live pool on the corrected
  registry reads -0.0133 pts [-0.0200, -0.0067], P+ 0.00004 on 103
  signals, sign test 38/103 (p 0.010), a diagnostic over correlated
  inputs, not a verdict. No registry cell for the pool read, by the
  evergreen's own rule. Full suite 4529 passed / 9 skipped, mypy
  clean, ruff clean on touched files (37 pre-existing ruff errors in
  `every_metric_backfill.py` left untouched).

## Correction 2026-09-16: the 16 margin cells were wrong, now replaced
(measured this session)
- The record script's margin frame used the absolute line as the market
  baseline instead of the absolute line-minus-final error, about 5 points
  off on every cell (e.g. pbp_replication -5.39, true -0.37). Unit 1 was
  never affected (its own path reads 10.40 vs 10.09 there). One-line fix in
  `scripts/every_metric_backfill_record.py`, recomputed per (family, tag)
  with identical files, dedupe, draws and seed (all corrected effects are
  fractions of a point; margins-family -0.1052 on 2,143 corroborates the
  served 9.952 vs 9.844 read), re-recorded through batch `--replace`
  (validators accepted all 16), artifact
  `artifacts/every_metric_backfill/20260916T220513Z/`. Registry still 6,660
  signals. Several corrected cells sit wholly below zero and now meet the
  letter of wrong_sign_resolved for "the model beats the spread on MAE";
  left unresolved per the backfill's predeclared no-adjudication rule.

## Open
- Whether backfilled cells count as new looks (they are new metrics on old
  looks; proposal: same family, flagged `backfilled` in notes, not counted).
