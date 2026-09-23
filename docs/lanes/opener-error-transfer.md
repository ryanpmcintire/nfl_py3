# Opener error transfer (XLG-09)

## Goal

Same books set NFL and CFB openers by a similar procedure. Measure whether an
opener-error model learned on CFB data ONLY transfers to the NFL opener: (a)
predicting close-minus-open line move, (b) predicting cover at the opener,
both as a standalone comparator and as one added fitted term inside the
served four-term pick probability. Separate transfer value (CFB-trained) from
feature value (an NFL-only-trained twin of the same features).

## Predeclared feature set (written before any fit ran)

Point-in-time features that exist in both leagues at opener-post time:

1. `home_favorite` = 1 if the opener home spread is negative (home favored)
2. `spread_size` = abs(opener home spread)
3. `key_number_distance` = min distance from `spread_size` to {3,7,10,14,17}
4. `home_prior_move` = home team's own most recent prior game's
   close-minus-open move (team-signed), `away_prior_move` likewise, and
   `prior_move_diff` = home minus away
5. `rest_diff` = home team days rest minus away team days rest since each
   team's own previous game

Targets: (a) `move` = close_home_spread - open_home_spread (points); (b)
`home_covered` = home actual margin plus open spread > 0, pushes dropped.

Both ridge (a) and logistic (b) fits standardize features and use the
project-standard ridge constant 1e-3 (fixed, not tuned in-sample; codebase
convention in `pick_probability_fit.FIT_RIDGE`) — the "fixed" branch of
"hyperparameters fixed or chosen inside CFB."

CFB training population: seasons 2021-2024, book Bovada (only book with
continuous, high-coverage `opening_lines` across those four seasons in
`data/cfb/lines/raw`), non-neutral-site completed games only.

NFL evaluation population: the 1,503-game graded opener population from
`pick_probability_fit.build_fit_population`, 2020-2025, which already carries
`close_home_spread`, `open_move`, `margin_vs_open`, `home_covered`,
`model_logit`, the flag-sum, and the market-move columns needed for the
four-term base.

Comparators: (i) CFB-trained predictor scored directly on NFL; (ii) the same
predictor's logit/prediction added as a fifth fitted term inside the
four-term probability, LOSO over NFL seasons, vs. the four-term base; (iii)
an NFL-only-trained twin of the identical five features, to separate transfer
value from feature value.

## State

Unit 2 (this pass, 2026-09-23): the `src/` import blocker from Unit 1 was
already fixed by whoever owned `displayed_confidence.py` before this session
started; confirmed with `.tools/uv.exe run python -c "import
nfl_ats.pick_probability_fit"` -> `ok`. Ran
`scripts/opener_error_transfer_eval.py` unmodified end to end (no bugs hit,
no script edits needed). Artifacts at
`artifacts/opener_error_transfer/20260923T192851Z/` (`summary.json`,
`per_game.csv`). CFB training population 3,054 games (2021-2024, Bovada,
non-neutral, completed); NFL evaluation population 1,503 games (2020-2025),
matching the predeclared population. `look_count=13` for this run (listed in
`summary.json["look_log"]`).

All 5 predeclared cells recorded via `nfl-ats weak-signals record --batch`,
family `opener_error_transfer_v1`:

- `opener_error_transfer_cfb_move_vs_zero` (target a, move MAE): CFB-trained
  ridge move predictor **worse** than a naive zero-move guess.
  effect=-0.0181 mae_improvement, interval [-0.0327, -0.0077] (entirely
  below zero), P+=0.0. **refuted_mechanism / wrong_sign_resolved.**
- `opener_error_transfer_cfb_direct_vs_base`: CFB-trained cover model used
  directly (standalone pick source) vs served four-term base. effect=-6.92
  accuracy points, interval [-10.98, -3.76], P+=0.0; on the 750 games where
  picks differed, CFB-direct 323-427, base 427-323.
  **refuted_mechanism / wrong_sign_resolved** (not a standalone substitute).
- `opener_error_transfer_added_term_vs_base` (the decision-relevant cell:
  CFB-transfer logit as a 5th fitted term, LOSO): effect=-0.399 accuracy
  points, interval [-1.50, 0.26] (crosses zero), P+=0.326; only 40/1,503
  games flip pick vs base (17-23 for plus, 23-17 for base on those). Brier
  0.24579 vs base 0.24539, log loss 0.68478 vs 0.68395 — essentially flat.
  Reliability tables (both) well aligned bin-by-bin.
  **unresolved_below_power** — zero-crossing interval, no closing ground per
  AGENTS.md.
- `opener_error_transfer_nflonly_twin_vs_base`: NFL-only-trained twin of the
  same 5 features, standalone, vs base. effect=-7.385 accuracy points,
  interval [-11.04, -4.40], P+=0.0. **refuted_mechanism /
  wrong_sign_resolved** — the 5 features alone (even NFL-fit) underperform
  the four-term base; the shortfall in the CFB-direct cell is feature-set
  poverty, not only transfer loss.
- `opener_error_transfer_nflonly_vs_cfbdirect` (isolates transfer vs feature
  value): effect=-0.466 accuracy points, interval [-6.12, 4.76] (crosses
  zero), P+=0.459. **unresolved_below_power** — cannot yet distinguish a
  CFB-transfer-specific cost from a generic feature-set cost.

Split-half reliability of the CFB-trained predictor scored on NFL (odd vs
even CFB training seasons): 0.475 (not near zero, so `no_split_half_
reliability` is not an admissible closing ground here; not used).

Coefficient stability per fold and an in-sample-vs-out-of-sample gap were
**not** computed — the predeclared script (owned, not to be redesigned this
pass) does not fit or export per-fold betas or in-sample scores, only LOSO
out-of-sample predictions. Flagged in Open below rather than added
post-hoc, since adding new statistics after seeing the results would be an
undeclared look.

Decision implication: the served four-term probability is not changed. The
only cell that could gate a change (`added_term_vs_base`) is unresolved, not
positive; do not add the CFB-transfer term to `src/`. Not closed as
negative either — AGENTS.md bars closing on a zero-crossing interval alone.

## Tried

- Checked `data/cfb/lines/raw` book coverage of `opening_lines` per season;
  Bovada is the only book with continuous multi-season coverage
  (2021-2024, ~1600-1750 spread rows/season); DraftKings only 2023-2024;
  William Hill/ESPN Bet/Caesars carry no opening_lines at all in this
  snapshot.
- Confirmed CFB `lines.parquet` rows join to `schedules.parquet` on
  `game_id`; the home row is the one whose `abbr` equals the schedule's
  `home_team`.
- Confirmed NFL side needs no separate assembly: the latest
  `artifacts/opener_evaluation/*/per_game.parquet` (read inside
  `build_fit_population`) already carries `close_home_spread`, `open_move`,
  `margin_vs_open`, `home_covered` and the four-term feature columns.

## Next

This unit is closed out for the run itself. If a future session wants to
push this further: (a) widen CFB training seasons/books (2020 CFB excluded,
no Bovada opening coverage that season) to shrink the standard error on
`opener_error_transfer_added_term_vs_base` and
`opener_error_transfer_nflonly_vs_cfbdirect`, the two unresolved cells, since
neither closed on a wrong sign nor on reliability; (b) if repeating this
family, extend the script to export per-fold LOSO betas and an in-sample fit
alongside the out-of-sample one, so coefficient stability and an IS-OOS gap
can be reported without a new undeclared look; (c) do not add the CFB-
transfer term to `src/` on this evidence — the gating cell is unresolved,
not positive.

## Open

- Only 6 NFL seasons for outer scoring; season-block bootstrap intervals will
  be wide. That is expected, not a rejection reason.
- CFB 2020 excluded from training (no Bovada opening coverage that season);
  if transfer looks promising, a future unit should widen CFB book/season
  coverage before treating this as settled.
- Per-fold coefficient stability and an in-sample-vs-out-of-sample gap were
  not part of the predeclared script output; not computed this pass (see
  State). A future unit should predeclare and add that before reporting it.
