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

## Unit 3 predeclaration (written before running scripts/opener_error_transfer_unit3.py)

Grade the same 5-feature CFB-trained opener-error term (`FEATURES =
[home_favorite, spread_size, key_number_distance, prior_move_diff,
rest_diff]`, same ridge, same logit/ridge fitters) as a 5th fitted term added
to the served four-term NFL pick probability, now on the new
`artifacts/extended_fit_population/20260923T205910Z/population.parquet`
(3,734 NFL games, 2011-2025; that parquet itself only carries
`game_id, season, week, home_covered, model_logit, composition_flag_sum,
market_move_toward_home, market_move_available, opener_source`, so
`home_team`/`away_team` are re-joined from
`data/processed/game_features_weak_stack.parquet` by `game_id`, and the
opener spread + gameday are re-joined per source: `tue_open` rows (2020-2025,
n=1,503) from `build_fit_population`'s own output (`tue_open_home_spread`,
`open_move`, `gameday` — identical to Units 1-2's NFL population);
`sbr_proxy_discrete` rows (2011-2019, n=2,231) from
`artifacts/sbr_era_opener_eval/20260819T233013Z/scored.parquet`
(`proxy_open_home_spread`, `gameday`), which carries no close-line, so
`open_move` is set to 0.0 for those rows — the same convention the extended
population already applies to `market_move_toward_home`/`_available` pre-2020
(stated in Unit 2/backfill lane: "market_move terms are 0 before 2020"),
extended here to the transfer feature set's own move-derived terms
(`prior_move_diff`, computed from `open_move` via `attach_transfer_features`,
is therefore 0 for every pre-2020 team-game). Target A (predicting
close-minus-open move) has no real target for `sbr_proxy_discrete` rows (no
close line), so Target A stays evaluated on the `tue_open` (2020-2025) subset
only, unchanged from Units 1-2 — not re-run as a new cell this unit.

**Point-in-time CFB-training rule (new this unit, per root instruction):** a
fixed pooled CFB training set (Unit 1/2's approach) cannot legitimately score
every NFL season once the population reaches back to 2011, because Unit 2's
pooled CFB seasons run through 2025, which does not precede early NFL
seasons. Unit 3 instead fits a season-specific CFB model per held-out NFL
season Y: train only on local CFB seasons strictly `< Y` from the same
13-season local pool used in Unit 2
(`{2012..2019} u {2021..2025}`, 2020 excluded as a genuine upstream gap, both
per Unit 2's Open section), then score NFL season Y with that model. A NFL
season only counts as **graded** if at least one CFB season strictly precedes
it in that local pool. Consequence, stated before running: **NFL 2011 and
2012 are excluded** (no local CFB season precedes 2011 or 2012; earliest
local CFB season is 2012, which does not strictly precede NFL season 2012
itself). **Graded NFL seasons: 2013-2025 (13 seasons, n=3,234 of the 3,734
extended-population games)** — reported as "all graded seasons" — with the
2020-2025 subset (n=1,503, matching Units 1-2's population exactly) reported
separately per the root's instruction. Per-season CFB training-set size grows
from 1 CFB season (NFL 2013, trained on CFB 2012 only) to 12 CFB seasons (NFL
2025, trained on CFB 2012-2019+2021-2024); logged per fold in the artifact.

Cells to record (decision-relevant term only, matching the task's "grade the
same predeclared term" scope — not re-deriving the standalone
CFB-direct/NFL-only-twin cells, which are reported in the artifact for
context but not re-recorded, same as Unit 2's treatment of its unchanged
Unit-1 cells):

- `opener_error_transfer_added_term_vs_base_v3_all_graded`: 5th-term LOSO
  (standard NFL leave-one-graded-season-out, same convention as Units 1-2)
  vs. the served four-term base, over all 13 graded seasons (2013-2025,
  n=3,234).
- `opener_error_transfer_added_term_vs_base_v3_2020_2025`: the same LOSO
  predictions, restricted to the 2020-2025 subset (n=1,503) for direct
  comparability with Units 1-2's population.

Both use `paired_accuracy_effect` (paired season-block bootstrap, block =
`season`, `samples=4000`, `seed=20260923`, same as Units 1-2) on accuracy
points. Classification follows AGENTS.md: an interval crossing zero is
`unresolved_below_power`, not a rejection; a wrong-sign whole-interval result
is `refuted_mechanism`/`wrong_sign_resolved`. Run once, no re-runs after
seeing results.

## Unit 3 results (run once, not yet recorded to the shared registry — root records)

Ran `scripts/opener_error_transfer_unit3.py` once for real. Artifact
`artifacts/opener_error_transfer_unit3/20260923T210810Z/` (`summary.json`,
`per_game.csv`). NFL population: the new
`artifacts/extended_fit_population/20260923T205910Z/population.parquet`
(3,734 games) re-joined to `home_team`/`away_team`
(`data/processed/game_features_weak_stack.parquet`) and to the opener spread
+ gameday (`tue_open` rows from `build_fit_population`; `sbr_proxy_discrete`
rows from `sbr_era_opener_eval`'s scored artifact, `open_move=0.0` for those
2011-2019 rows, no close line exists to compute a real move). **Graded NFL
seasons per the point-in-time rule: 2013-2025 (13 seasons, n=3,234)**; **2011
and 2012 excluded** (no local CFB season strictly precedes them). CFB
training is season-specific this unit: for held-out NFL season Y, the CFB
opener-error model trains only on local CFB seasons `< Y`. Early folds are
CFB-data-starved: NFL 2013 trains on 48 CFB games (season 2012 only, 5Dimes
book); NFL 2020/2021 both train on the same 389 games (seasons 2012-2019, the
2020 CFB gap holds both folds back); NFL 2022-2025 grow fast once Bovada
enters (1,196 to 3,443 games) since Bovada's per-season coverage is far
denser than 5Dimes'. `look_count=6`.

- `opener_error_transfer_added_term_vs_base_v3_all_graded` (5th-term LOSO,
  standard NFL leave-one-graded-season-out, vs. served four-term base, all 13
  graded seasons): effect=+0.1237 accuracy points, interval [-0.4593,
  +0.6839] (crosses zero), P+=0.6694. 174/3,234 picks flip (plus 89-85, base
  85-89 on those). **unresolved_below_power.**
- `opener_error_transfer_added_term_vs_base_v3_2020_2025_subset` (same LOSO
  predictions, restricted to the 1,503-game 2020-2025 subset used by Units
  1-2): effect=+0.2661 accuracy points, interval [-0.5316, +0.8748] (crosses
  zero), P+=0.7735. 28/1,503 picks flip (plus 16-12, base 12-16).
  **unresolved_below_power.** Directionally consistent with Unit 2's
  `_v2` cell (+0.133, P+=0.76) though not numerically identical (Unit 3's
  base/plus LOSO folds pool across all 13 graded NFL seasons rather than just
  6, and every fold's CFB training set is now point-in-time-restricted).
- Per-fold `cfb_transfer_logit` betas in the plus model: **positive in all 13
  folds** (0.0187 to 0.0629, no sign flips) — unlike Unit 2's `_v2` cell,
  where 2 of 6 folds were negative. Still an order of magnitude smaller than
  the stable base four-term coefficients (0.20-0.25 per Unit 2). Sign
  consistency across every held-out season is notable but does not by itself
  clear AGENTS.md's zero-crossing bar; the aggregate bootstrap interval still
  crosses zero.
- `cfb_direct_vs_base_all_graded_context` (standalone CFB-trained predictor
  vs. base, all 13 graded seasons, reported for context only, not recorded):
  effect=-1.4224 accuracy points, interval [-4.1147, +1.6405] (crosses zero),
  P+=0.1736. Weaker sign and much wider than Units 1-2's `cfb_direct_vs_base`
  (which was resolved negative, -6.92/-7.78 pts, wrong-sign, on the smaller
  2020-2025-only population) — expected, since most graded-season folds here
  use far less CFB training data than Units 1-2's pooled 13-season set.
- Reliability: split-half reliability of the CFB-trained predictor class was
  not recomputed this unit (the per-season point-in-time training design has
  no single fixed CFB training set to split); Unit 2's pooled-set value
  (0.141, same 13-season CFB pool, same features/ridge) is reused as the
  `--reliability` field below since it characterizes the same predictor
  class, not re-measured against this unit's population.
- Records: `base_4term` 1726-1508, `base_plus_cfb_transfer` 1730-1504,
  `cfb_direct` 1680-1554 (all-graded population, n=3,234).

Decision implication unchanged from Units 1-2: no cell in this family
supports adding the CFB-transfer term to `src/`. Extending to the 2011-2025
population under a point-in-time-correct CFB training rule keeps both
gating-cell intervals crossing zero; sign consistency across LOSO folds grew
(notable, worth tracking) but the classification stays
`unresolved_below_power` under AGENTS.md (a crossing interval never
justifies closure either way).

Root: record these two cells with (exact commands, family
`opener_error_transfer_v3`, source
`artifacts/opener_error_transfer_unit3/20260923T210810Z/summary.json`):

```
nfl-ats weak-signals record --name opener_error_transfer_added_term_vs_base_v3_all_graded \
  --description "CFB-trained opener-error logit (season-specific, point-in-time CFB training strictly preceding each held-out NFL season) added as a 5th fitted term to the served four-term NFL pick probability, LOSO by season, vs the four-term base, on the extended 2011-2025 NFL population restricted to the 13 seasons a preceding local CFB season exists for (2013-2025)" \
  --source artifacts/opener_error_transfer_unit3/20260923T210810Z/summary.json \
  --effect 0.12368583797155441 --effect-units accuracy_points \
  --classification unresolved_below_power --league nfl \
  --season-start 2013 --season-end 2025 \
  --interval-low -0.45931266748014443 --interval-high 0.6838882411666622 \
  --probability-positive 0.669375 --sample-games 3234 --sample-blocks 13 \
  --reliability 0.14132263925353555 --family opener_error_transfer_v3 \
  --category market \
  --classification-evidence "Interval crosses zero (P+=0.67) on the 13-season point-in-time-correct population (NFL 2011-2012 excluded, no local CFB season precedes them); per-fold LOSO betas for cfb_transfer_logit are positive in all 13 folds (0.0187-0.0629) but an order of magnitude smaller than the stable base four-term coefficients (0.20-0.25); early folds train the CFB model on very few games (48 for NFL 2013, up to 389 through the 2020 CFB gap) before Bovada coverage widens training from 2022 on; reliability reused from Unit 2's pooled-set measurement (0.141), not zero, so no_split_half_reliability is not admissible" \
  --plain-summary "We tested whether a model trained on college football's own opening-line errors helps predict which side covers the NFL opener, now checked back to 2013 instead of just 2020. The extra signal points the right direction almost every year, but it is still too small and too noisy to say for sure it helps -- so the pick stays exactly as it would be without it."

nfl-ats weak-signals record --name opener_error_transfer_added_term_vs_base_v3_2020_2025_subset \
  --description "Same 5th-term LOSO cell as _all_graded, restricted to the 2020-2025 subset (n=1,503) used by Units 1-2, for direct comparability" \
  --source artifacts/opener_error_transfer_unit3/20260923T210810Z/summary.json \
  --effect 0.26613439787092075 --effect-units accuracy_points \
  --classification unresolved_below_power --league nfl \
  --season-start 2020 --season-end 2025 \
  --interval-low -0.5315614617940168 --interval-high 0.8748317631224856 \
  --probability-positive 0.7735 --sample-games 1503 --sample-blocks 6 \
  --reliability 0.14132263925353555 --family opener_error_transfer_v3 \
  --category market \
  --classification-evidence "Interval crosses zero (P+=0.77) on the 2020-2025 subset of the point-in-time-correct population; directionally consistent with Unit 2's _v2 cell (+0.133 pts, P+=0.76) though not identical (base/plus LOSO folds here are drawn from all 13 graded NFL seasons, not just these 6); reliability reused from Unit 2 (0.141), not zero" \
  --plain-summary "Looking only at the last six NFL seasons (the same window used before), the college-football transfer signal again nudges toward helping the pick, but the range of plausible outcomes still includes no effect at all -- unresolved, not served."
```

## Next

Unit 2 is closed out for the run itself (widened CFB coverage + per-fold
betas + in-sample fit, both delivered). Unit 3 is closed out for the run
itself too (point-in-time-correct CFB training per NFL season, on the new
2011-2025 extended population, both graded-population cuts delivered).
**Immediate next step is mechanical: root runs all four `nfl-ats
weak-signals record` commands above ((Unit 2 cells were already recorded 2026-09-23 as registry 7,000-7,001; v3 cells recorded, registry 7,021) two from Unit 2, family
`opener_error_transfer_v2`; two from Unit 3, family
`opener_error_transfer_v3`)** — including `--category market` and
`--plain-summary` on the Unit 3 pair per this pass's instruction (the Unit 2
commands predate that requirement and were left as originally written; root
may add `--category market` to them too when recording if it wants uniform
registry metadata). Beyond that, if a future session wants to push this
further: (a) both gating cells (`_v2` on 1,503 games/6 folds, `_v3_all_graded`
on 3,234 games/13 folds) are still zero-crossing even after a 13x CFB-widen
and a 2.5x NFL-season widen — every LOSO fold's `cfb_transfer_logit` beta is
now positive (new in Unit 3, not true of Unit 2's folds), which is worth
tracking, but sign-per-fold consistency is not itself an AGENTS.md closing
ground; (b) do not add the CFB-transfer term to `src/` on this evidence — the
gating cell is unresolved, not positive, three times now under three
different CFB training designs; (c) the remaining lever is more NFL seasons
or a way to give the earliest folds (2013-2021) more than 48-389 CFB training
games — CFB 2011 and earlier are not known to exist locally and were not
checked this pass.

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
