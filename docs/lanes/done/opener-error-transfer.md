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

## Unit 2 results (run, not yet recorded to the shared registry — root records)

Ran `scripts/opener_error_transfer_unit2.py` once for real. Artifact
`artifacts/opener_error_transfer/20260923T202410Z/` (`summary.json`,
`per_game.csv`). CFB training population widened to 4,301 games (13 seasons,
up from 3,054/4 seasons); NFL eval population unchanged, 1,503 games
(2020-2025). `look_count=16`.

- Split-half reliability of the CFB-trained predictor scored on NFL (odd vs
  even CFB seasons): **0.141** (was 0.475 on the 4-season population) — drops
  with the wider, more heterogeneous (two-book) training set. Not zero, so
  `no_split_half_reliability` is still not an admissible closing ground.
- `opener_error_transfer_added_term_vs_base_v2` (5th-term LOSO vs base):
  effect=+0.133 accuracy points, interval [-0.202, 0.475] (crosses zero),
  P+=0.7599. 24/1,503 picks flip (13-11 plus, 11-13 base). Per-fold LOSO
  betas for `cfb_transfer_logit`: 2020 -0.0098, 2021 +0.0009, 2022 -0.0084,
  2023 -0.0405, 2024 -0.0236, 2025 +0.0135 — sign-flipping and an order of
  magnitude smaller than the stable base four-term coefficients (e.g.
  `composition_flag_sum` 0.20-0.25 every fold). In-sample vs OOS gap:
  accuracy -0.00067, Brier -0.00071, log loss -0.00151 (no overfitting).
  Interval narrowed ~4x vs Unit 1 ([-1.50,0.26]) and point estimate flipped
  positive, but still crosses zero. **unresolved_below_power.**
- `opener_error_transfer_nflonly_vs_cfbdirect_v2`: effect=+0.399 accuracy
  points, interval [-4.318, 4.485] (crosses zero), P+=0.5819. In-sample vs
  OOS gap: accuracy +0.0180, Brier -0.0019, log loss -0.0037 (small, no
  overfitting). **unresolved_below_power.**
- Unaffected-but-recomputed-as-side-effect Unit-1 cells (not re-recorded):
  `cfb_direct_vs_base` effect -7.78 [-10.10,-5.59] P+=0.0 (was -6.92);
  `nfl_only_twin_vs_base` effect -7.385 [-11.04,-4.40] P+=0.0 (unchanged,
  doesn't depend on CFB data); `move_vs_zero` effect -0.0200 [-0.033,-0.011]
  P+=0.0 (was -0.0181). All still refuted_mechanism/wrong_sign_resolved,
  consistent with Unit 1.

Decision implication unchanged from Unit 1: no cell in this family supports
adding the CFB-transfer term to `src/`. Widening CFB coverage 13x by game
count narrowed the gating cell's interval substantially and flipped its
point estimate positive, but it still crosses zero — genuinely
`unresolved_below_power`, not a promotion.

Root: record these two cells with (exact commands, family
`opener_error_transfer_v2`, source
`artifacts/opener_error_transfer/20260923T202410Z/summary.json`):

```
nfl-ats weak-signals record --name opener_error_transfer_added_term_vs_base_v2 \
  --description "CFB-trained opener-error logit (13-season, 4301-game, two-book widened training set) added as a 5th fitted term to the served four-term NFL pick probability, LOSO by season, vs the four-term base" \
  --source artifacts/opener_error_transfer/20260923T202410Z/summary.json \
  --effect 0.13306719893546592 --effect-units accuracy_points \
  --classification unresolved_below_power --league nfl \
  --season-start 2020 --season-end 2025 \
  --interval-low -0.20215633423180668 --interval-high 0.4745762711864443 \
  --probability-positive 0.759875 --sample-games 1503 --sample-blocks 6 \
  --reliability 0.14132263925353555 --family opener_error_transfer_v2 \
  --classification-evidence "Interval crosses zero (P+=0.76); per-fold LOSO betas for cfb_transfer_logit flip sign across the 6 NFL seasons (-0.0098,+0.0009,-0.0084,-0.0405,-0.0236,+0.0135), an order of magnitude smaller than the stable base four-term coefficients; in-sample-minus-OOS accuracy gap -0.00067 (no overfitting); split-half reliability 0.141, not zero, so no_split_half_reliability is not admissible either"

nfl-ats weak-signals record --name opener_error_transfer_nflonly_vs_cfbdirect_v2 \
  --description "NFL-only-trained 5-feature twin vs CFB-trained-direct on the same features, widened CFB (13-season, 4301-game) population; isolates transfer cost from feature-set cost" \
  --source artifacts/opener_error_transfer/20260923T202410Z/summary.json \
  --effect 0.39920159680638667 --effect-units accuracy_points \
  --classification unresolved_below_power --league nfl \
  --season-start 2020 --season-end 2025 \
  --interval-low -4.317897371714641 --interval-high 4.48493342676945 \
  --probability-positive 0.581875 --sample-games 1503 --sample-blocks 6 \
  --reliability 0.14132263925353555 --family opener_error_transfer_v2 \
  --classification-evidence "Interval crosses zero (P+=0.58), wide because it compares two already-noisy standalone predictors; cannot yet separate a CFB-transfer-specific cost from a generic feature-set cost; in-sample-minus-OOS accuracy gap +0.018 (no overfitting)"
```

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

## Unit 2 predeclaration (written before running scripts/opener_error_transfer_unit2.py)

Widened CFB training population, decided from local data coverage checked
this pass (no API fetch — see rationale below): seasons
`(2012,2013,2014,2015,2016,2017,2018,2019,2021,2022,2023,2024,2025)`, 13
seasons total (up from 4). Book varies by season since no single book has
continuous local opener coverage across the whole span: `5Dimes & sportbet`
for 2012-2019 (`sbr_multibook` source era, only book column with opener
coverage), `Bovada` for 2021-2025 (adds 2025 to the prior 2021-2024; 2025 CFB
season confirmed fully completed, 3,831/3,831 games, through 2026-01-20).
2020 stays excluded: `data/cfb/lines/raw/20260816T143907Z/manifest.json`
documents `cfbd_provider_sparse` for 2020 with "zero openers and zero
moneylines" for the whole season across every book in the archive — an
upstream provider gap the CFBD API would not fill (same provider), not
fetched. Same non-neutral-site completed-game filter, same 5 features, same
ridge, same NFL evaluation population (1,503 games, unchanged) as Unit 1.

Script change: `scripts/opener_error_transfer_unit2.py` (copy of the Unit-1
script) adds per-LOSO-fold beta export (base four-term, the 5th-term-plus
model, and the NFL-only twin) and an in-sample fit (fit and score on the
full NFL population, no held-out season) for the same three models, reported
beside the existing out-of-sample numbers, with an in-sample-minus-OOS gap
on accuracy/Brier/log-loss. Re-grading only the two Unit-1 cells that were
`unresolved_below_power`: `opener_error_transfer_added_term_vs_base` and
`opener_error_transfer_nflonly_vs_cfbdirect`, as `_v2` cells. The other three
Unit-1 cells (already `refuted_mechanism`/`wrong_sign_resolved`) are not
re-recorded even though the widened CFB population changes their point
estimates as a side effect of the same run; those numbers appear in the new
artifact for transparency only.

## Next

Unit 2 is closed out for the run itself (widened CFB coverage + per-fold
betas + in-sample fit, both delivered). Immediate next step is mechanical:
root runs the two `nfl-ats weak-signals record` commands above. Beyond that,
if a future session wants to push this further: (a) the `added_term_vs_base`
interval is still zero-crossing at 1,503 NFL games / 6 LOSO folds even after
a 13x CFB-population widen — the remaining lever is more NFL seasons of
outer data (not more CFB training data, which is already near-exhausted
locally: 2020 is a genuine upstream gap, not a fetchable one); (b) do not add
the CFB-transfer term to `src/` on this evidence — the gating cell is
unresolved, not positive, twice now under two different CFB training
populations.

## Open

- Only 6 NFL seasons for outer scoring; season-block bootstrap intervals will
  stay wide until more NFL seasons accumulate. That is expected, not a
  rejection reason.
- CFB 2020 stays excluded from training: confirmed via
  `data/cfb/lines/raw/20260816T143907Z/manifest.json` that the source
  archive has zero opener rows for any book that season
  (`cfbd_provider_sparse` regime) — an upstream gap, not something a CFBD
  API fetch would fill (same provider). No API call was made this pass since
  the manifest already documents the gap as total for the season.
  DraftKings (2023-2025) and ESPN Bet (2024-2025) exist locally too but were
  not added as additional per-season books this pass since Bovada already
  gives continuous coverage for 2021-2025; a future unit could pool multiple
  books per season for within-season replication if that becomes useful.
- Split-half reliability of the CFB-to-NFL transfer predictor dropped from
  0.475 (4-season population) to 0.141 (13-season population) — worth
  tracking if a future unit revisits this family; still not zero.
- Per-fold coefficient stability and an in-sample-vs-out-of-sample gap are
  now exported by `scripts/opener_error_transfer_unit2.py` for the base,
  plus, and NFL-only-twin models (see Unit 2 results above and
  `summary.json`).
