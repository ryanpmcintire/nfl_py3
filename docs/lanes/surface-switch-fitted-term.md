# Surface-switch flag as a fifth fitted term

## Goal
Test the ENV-02 grass-modal-visitor-on-turf lead (`surface_switch_flag`,
home covers +1.16 full-slate pts, P+ 0.995, 2009-2025, replicates on CFB,
grass-venue mirror null) as a fitted fifth term in the served four-term
pick-probability logistic (`model_logit`, `composition_flag_sum`,
`market_move_toward_home`, `market_move_available`), never as a flip.

## Predeclaration (before any fit)
- Family: one member, `surface_switch_flag_pit` (point-in-time, home-signed
  0/1: +1 when today's home venue is turf AND the visiting team's own modal
  home-venue surface computed ONLY from that team's home games strictly
  before this game's date is grass; else 0). Mirror direction (turf-modal
  visitor at a grass venue) is excluded by predeclaration -- ENV-02 already
  found that mirror null in both leagues.
- Populations: (A) served `build_fit_population` 2020-2025, n~1503; (B)
  extended `artifacts/extended_fit_population/20260923T205910Z/population.parquet`,
  2011-2025, n~3734 for accuracy, restricted to seasons with an `open_move`
  source (SBR 2013-2019 + true 2020-2025) for the line-move grading only.
- Fit: LOSO by season, `FIT_RIDGE=1e-3`, same ridge/iteration Newton solver
  as `pick_probability_fit._fit_logit`. Base = four served terms. Plus =
  base + `surface_switch_flag_pit`.
- Grading: (1) LOSO out-of-season accuracy, base vs plus, both populations,
  decisive record on games where the pick differs. (2) line movement toward
  the pick (`open_move` signed by pick side), base vs plus, on the
  populations/seasons where `open_move` exists. Two line-move variants:
  4-term-based (both sides already carry the served market-move terms, so
  the diff is clean even though the absolute served-vs-naive-baseline
  yardstick is documented CONTAMINATED in
  `artifacts/line_move_yardstick/*/results.json`) and an opener-only 2- vs
  3-term variant (drops market-move terms entirely) as an uncontaminated
  cross-check, mirroring `scripts/line_move_yardstick_paired_eval.py`.
- Looks counted: 2 populations x 2 gradings x (2 line-move variants for
  grading 2) = up to 6 named cells; count is written into the artifact's
  `look_log`.
- Closing grounds available: none targeted; this run reports
  `unresolved_below_power` unless a resolved wrong sign with the whole
  interval on the wrong side appears (that would be new information, not
  predeclared).

## State
Script `scripts/surface_switch_fitted_term.py` ran once for real:
`artifacts/surface_switch_fitted_term/20260925T194502Z/summary.json`
(ruff format + ruff check both pass, 11 looks logged).

**Measured** results (accuracy diff = plus5-term minus served four-term,
mean full-slate accuracy points, LOSO by season, `FIT_RIDGE=1e-3`):
- Population A (served 2020-2025, n=1503): record 839-664 -> 828-675,
  diff -0.0073, season-block 95% [-0.0164, +0.0013] P+ 0.038, week-block
  [-0.0157, +0.0007] P+ 0.033. Decisive (35 games where picks differ):
  base wins 23, plus wins 12.
- Population B (extended 2011-2025, n=3734): record 1994-1740 -> 2019-1715,
  diff +0.0067, season-block 95% [-0.0085, +0.0212] P+ 0.802, week-block
  [-0.0072, +0.0205] P+ 0.824. Decisive (673 games): base wins 324,
  plus wins 349.
- Line move toward pick, four-term-based (both sides already carry
  market-move terms -- diff is clean, see contamination note in the
  artifact): Pop A diff -0.0093, P+(season) 0.158, P+(week) 0.094,
  decisive 24 (base 16, plus 8). Pop B (n=3234, seasons with an open-line
  source) diff -0.0312, P+(season) 0.053, P+(week) 0.106, decisive 428
  (base 239, plus 189).
- Line move, opener-only uncontaminated cross-check (2-term vs 3-term, no
  market-move terms): Pop A diff -0.0283, P+ 0.038; Pop B diff -0.0073,
  P+ 0.396. Same negative lean as the four-term cells.
- No cell has its whole interval on the wrong side -- every cell touches
  zero -- so none is `wrong_sign_resolved`; all six graded cells are
  `unresolved_below_power`.
- Per-fold natural-coefficient betas for `surface_switch_flag_pit`: in
  population B the sign is **positive and stable across all 15 LOSO
  folds** (range +0.176 to +0.247, e.g. 2025 held out: +0.207), matching
  ENV-02's original direction. In population A (6 folds, 2020-2025 only)
  the sign is noisy and small (2020 +0.064, 2021 -0.017, 2022 +0.004,
  2023 +0.043, 2024 +0.102, 2025 -0.012) -- 2 of 6 folds flip negative.
  So the fitted coefficient itself points the predeclared direction
  reliably only once pre-2020 seasons are in the training pool; it does
  not yet reliably turn into more correct out-of-season picks in either
  population.
- Flag coverage (games flagged per season, point-in-time modal surface,
  computed only from each visiting team's own home games strictly before
  the current game's date): 2011 67, 2012 69, 2013 63, 2014 68, 2015 70,
  2016 57, 2017 62, 2018 56, 2019 53, 2020 52, 2021 65, 2022 65, 2023 65,
  2024 75, 2025 77 (roughly 21-29% of each season's games).
- Look count: 11 (builder + 2 populations x [base fit, plus fit, accuracy
  cell] + 2 populations x [four-term line-move cell, opener-only line-move
  cell]).

## Tried
- Reused `scripts/line_move_yardstick_paired_eval.py`'s `_design`,
  `_standardise`, `_predict`, `cell_stats` (parameterized by feature
  tuple) and `pick_probability_fit._fit_logit`/`FIT_RIDGE`/
  `build_fit_population` rather than duplicating the Newton solver.
- Reused `scripts/opener_error_transfer_unit4.py`'s
  `load_line_move_population` join pattern (teams from
  `game_features_weak_stack.parquet`, opener/move from the served
  population + `sbr_odds.parquet` 2013-2019) to attach `open_move` onto
  the extended population for the line-move grading only; 2011-2012 stay
  excluded from that grading (no open-line source), matching existing
  precedent.
- Built a point-in-time surface-switch flag independent of
  `nfl_ats.surface_switch_tilt_overlay.surface_switch_flag_by_game`,
  which computes its modal surface from the WHOLE season (including
  future weeks) -- leaky for a mid-season prediction even though the
  underlying venue surface essentially never changes within a season.
  The new builder uses `pd.merge_asof(..., direction="backward",
  allow_exact_matches=False)` per visiting team against that team's own
  chronological home-game surface history.

## Next

- 2026-09-25 root: the drafted commands used flags the CLI lacks and fractions
  as accuracy points; root recorded six corrected cells (registry 7,264-7,269,
  family `surface_switch_fitted_term_v1`), all unresolved_below_power.
  Accuracy +0.67 pts [-0.86,+2.13] P+ 0.80 on 2011-2025 (349-324 decisive),
  -0.73 [-1.65,+0.13] P+ 0.04 on 2020-2025 (12-23). Betas positive in all 15
  extended folds. Not served. Revisit with more seasons, or if the per-season
  pattern (positive pre-2020, negative 2020-2024) gets a mechanism.

## Open
- None of the six cells crossed into a closing ground; this stays a
  tracked challenger like the rest of ENV-02, not a promotion. The next
  useful step, if picked back up, is widening population A's window (more
  seasons of served four-term data) before re-testing whether the
  positive-fold-beta / negative-holdout gap in 2020-2025 narrows, since
  15 LOSO folds already show the coefficient is stable once pre-2020 data
  is included.
- Line-move grading needed `open_move`/`gameday`/team names joined in from
  `data/processed/game_features_weak_stack.parquet` +
  `data/processed/sbr_odds.parquet` (2013-2019) + the served opener
  evaluation (2020-2025), mirroring `scripts/opener_error_transfer_unit4.py`'s
  `load_line_move_population`; 2011-2012 have no open-line source and stay
  excluded from that grading only (not from accuracy grading), which is why
  population B's line-move cells report n=3234 rather than n=3734.
