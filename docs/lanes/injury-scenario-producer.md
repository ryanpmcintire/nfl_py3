# Injury scenario producer (PER-10)

## Goal
Build a research-only joint lineup-scenario producer wired to the injury
scenario mixture kernel, run once for 2026 Week 3, and compare its mixed
discrete cover probability to the served one per game.

## State
Not built. Both documented blockers in `docs/injury_scenario_mixture.md`
still hold today, and a third blocker was discovered this session: **the
kernel itself no longer exists in `src/`.**

## Tried (this session, read-only)
- `docs/injury_scenario_mixture.md` (2026-09-02): names two blockers —
  (1) no joint multi-player lineup probability, only marginals; (2) no
  accepted mapping from player EPA/value state to a scenario margin-point
  center.
- `git log --oneline -- src/nfl_ats/injury_scenarios.py` and
  `tests/test_injury_scenarios.py`: both were **deleted** in commit
  `b7ed31d` ("Repository cut", 2026-09-10) as one of "60 src modules
  imported by nothing but their own tests" — the kernel was orphaned
  (no producer ever called it) and got swept in the hygiene cut. The
  271-line module is recoverable verbatim at `git show b7ed31d~1:src/nfl_ats/injury_scenarios.py`.
  ROADMAP.md:549 (`PER-10` row) still asserts "Distribution kernel
  complete" — that line is now stale/inaccurate; nothing in `src/`
  imports or exposes `injury_scenarios.py` today (confirmed via
  `find src -iname "*injury*scenario*"` -> only a stale `.pyc`).
- Checked whether blocker 1 (joint probability) is closeable with
  existing measured results: `docs/absence_pairwise_dependence.md` /
  `registry` (ROADMAP.md:549) already measured pairwise absence coupling
  (independence rejected; observed/null coupling ~1.13-1.25x, all pairs
  sit together 20-27% less than pooled independence implies) — this is
  reusable, not a new decision, and would plausibly close blocker 1 for
  small "questionable" sets (full-unit sits never occur per the same
  doc).
- Checked whether blocker 2 (margin-center mapping) is closeable with
  data already in the repo:
  - `artifacts/latent_ratings_on_production/expected_lineup_ratings.parquet`
    (built by `scripts/latent_ratings_on_production.py`) has per-game
    `lineup_total` / `full_strength_total` / `divergence`, built from
    per-player offense/defense ridge ratings
    (`data/processed/player_participation_ratings.parquet`) times EWMA
    role shares times **marginal** play probability
    (`scripts/latent_ratings_on_production.py:300-345`). This is in raw
    EPA-weighted rating units, not margin points, and is used nowhere in
    `src/` (`grep -rln "lineup_total|diff_divergence" src` -> no hits):
    it is a standalone research artifact with no fitted conversion to
    margin points.
  - `src/nfl_ats/players.py:1441-1442` (`_injury_value_features`) builds
    `injury_skill_epa_value_lost` / `injury_defense_disruption_value_lost`
    as a **linear sum over players** of
    `severity(marginal) * role_share * player_value_rate`, which would in
    principle let a scenario plug in a deterministic 0/1 in place of
    `severity` — **if** a frozen margin-point coefficient on these
    features existed.
  - No such coefficient exists. The only consumers of these two columns
    are `src/nfl_ats/surgical_gating.py:11-13` (a magnitude-threshold
    gate that picks between two already-computed model outputs, not a
    margin adjustment) and
    `src/nfl_ats/injury_value_tilt_overlay.py:82-108`
    (`apply_injury_value_tilt_overlay`), which only **flips the pick's
    side** when `value_lost_diff` has a favorable sign for the healthier
    team — no magnitude-to-points conversion anywhere, and its own
    disclosure text says "Prospective evidence only -- not applied to
    the published card" (`injury_value_tilt_overlay.py:125-131`). A
    sign-only flip is exactly the inadmissible flip pattern AGENTS.md's
    "One calibrated probability decides every pick" section bans from
    the served path, so it cannot be reused as the kernel's per-scenario
    margin center either.

## Next
Blocker 2 is unresolved and is a genuine new modeling decision (fit and
leave-one-season-out validate a mapping from lineup EPA divergence, or
from `injury_*_value_lost`-style per-player deltas, to margin points),
not a wiring task. Before a producer can be built:
1. Restore `src/nfl_ats/injury_scenarios.py` and its test from
   `git show b7ed31d~1:...` (or re-derive) if/when a producer is ready
   to consume it.
2. Predeclare and fit the margin-point mapping out of season (candidate
   input: the already-computed `diff_divergence` /
   `injury_*_value_lost` features), report in-sample/out-of-sample gap
   per AGENTS.md's calibration section, before any scenario center is
   trusted.
3. Combine per-player marginal play probabilities
   (`nfl_ats.availability.resolve_unavailability`) with the measured
   pairwise coupling multiplier from `docs/absence_pairwise_dependence.md`
   to build the joint scenario set (blocker 1 path, already closeable).
4. Only then wire both into the restored kernel and run 2026 Week 3.
5. Separately: fix ROADMAP.md:549's stale "kernel complete" claim, since
   the module is gone from `src/`.

## Unit 2 predeclaration (2026-09-23, before fitting)
Blocker-2 margin-mapping fit. Predictor: `value_lost_diff` = sum of
`diff_injury_skill_epa_value_lost` + `diff_injury_defense_disruption_value_lost`
from `data/processed/game_features_player_value.parquet` (home - away;
verified via direct subtraction check), the same combination
`injury_value_tilt_overlay.raw_value_lost_diff` already uses. Point-in-time
seasons = rows where `home_injury_observed_at` is populated: 2010-2024
essentially 100% (2009 is 0%, 2025-2026 are 0% — no observed-at attestation,
excluded as not confirmed pregame-only). Fit population source:
`nfl_ats.pick_probability_fit.build_fit_population`, whose opener-evaluation
window only spans seasons 2020-2025 (`model_logit`, `composition_flag_sum`,
`market_move_toward_home`, `market_move_available` — production's served
FIT_FEATURES four terms — only exist there). Intersection of both
constraints = **seasons 2020-2024** (5 seasons) used for both the margin
regression LOSO and the probability paired look; 2025/2026 excluded for lack
of point-in-time attestation, 2009-2019 excluded for lack of the served
4-term baseline to pair against.
Two fits, both LOSO by season on 2020-2024:
(A) OLS `margin_vs_open ~ 1 + value_lost_diff` — reports points/unit slope
per fold, stability, OOS MAE/RMSE vs opener-alone (predict 0).
(B) Logistic `home_covered ~ model_logit + composition_flag_sum +
market_move_toward_home + market_move_available` (baseline, matches
production FIT_FEATURES exactly) vs the same 4 terms plus `value_lost_diff`
as a 5th fitted term (candidate) — this satisfies AGENTS.md "one calibrated
probability ... every situational signal as a fitted term," not a
margin-to-probability bolt-on. Paired accuracy/log-loss/Brier on the same
held-out graded games.
Script: `scripts/injury_value_margin_map.py`. Output:
`artifacts/injury_value_margin_map/<ts>/`. No `src/` or served-path changes.

## Unit 2 result (2026-09-23, run once for real)
Ran `.tools/uv.exe run python scripts/injury_value_margin_map.py`. Output:
`artifacts/injury_value_margin_map/20260923T210029Z/` (`summary.json`,
`scoped_population.parquet`, `paired_probability_population.parquet`).
1228 scoped games (seasons 2020-2024, point-in-time only). `ruff check` clean.

(A) Margin regression, LOSO by season, points per unit `value_lost_diff`:
2020 -0.274, 2021 -0.373, 2022 -0.310, 2023 -0.302, 2024 -0.313 — 5/5 folds
negative, sign as expected (team that lost more value underperforms its
opener). Pooled block-bootstrap (season/week blocks, 2000 draws) slope
-0.320, 95% CI [-0.593, -0.038], probability slope is positive = 0.0115 ->
**resolved_directional on sign**. OOS margin MAE: opener-alone 10.146 ->
candidate 10.128 (bootstrap probability improvement positive = 0.7135, CI
[-0.045, 0.088] — crosses zero). OOS RMSE 13.039 -> 13.025 (prob positive
0.704). Practical OOS improvement is **unresolved_below_power** — small and
crosses zero, not refuted.

(B) Cover-probability paired look: baseline = served 4-term LOSO logistic
(`model_logit`, `composition_flag_sum`, `market_move_toward_home`,
`market_move_available`, matching production `FIT_FEATURES` exactly);
candidate = same 4 terms + `value_lost_diff` as a 5th fitted term (per
AGENTS.md "one calibrated probability ... every situational signal as a
fitted term"). Accuracy delta -0.814 points (candidate worse), probability
candidate is better = 0.28. Brier improvement +0.00047 (prob positive 0.65),
log loss improvement +0.00083 (prob positive 0.63). 158 decisive games:
candidate wins 74, baseline wins 84, exact binomial p=0.474. All three
intervals cross zero -> **unresolved_below_power**, no refutation (sign not
reversed with the whole interval on the wrong side), no positive control run.

Decision-relevant conclusion: the injury value-lost differential has a
directionally-confirmed but practically tiny relationship to the margin
residual; adding it as a 5th fitted term to the served probability does not
clearly help or hurt over 2020-2024 and stays open, not closed.

Record commands for the root (not run — registry write is out of scope here):
```
nfl-ats weak-signals record --name injury_value_lost_margin_slope --description "LOSO-by-season OLS of margin_vs_open on value_lost_diff (diff_injury_skill_epa_value_lost + diff_injury_defense_disruption_value_lost), 2020-2024, point-in-time injury data only. Per-fold slopes: 2020 -0.274, 2021 -0.373, 2022 -0.310, 2023 -0.302, 2024 -0.313 points per unit (5/5 negative). Pooled block-bootstrap slope -0.320 [-0.593,-0.038], probability positive=0.0115. OOS MAE 10.146->10.128, RMSE 13.039->13.025; bootstrap probability the MAE improvement is positive=0.7135." --source artifacts/injury_value_margin_map/20260923T210029Z/summary.json --effect 0.0184 --effect-units mae_improvement --classification unresolved_below_power --league nfl --season-start 2020 --season-end 2024 --interval-low -0.0454 --interval-high 0.0877 --probability-positive 0.7135 --sample-games 1228 --sample-blocks 89 --classification-evidence "Slope sign is resolved negative (prob_positive=0.0115, as expected: team that lost more value underperforms its opener) but the OOS margin-error improvement crosses zero (prob_positive=0.7135 MAE, 0.704 RMSE); not a refuted mechanism and no positive control run, so AGENTS.md leaves the practical-improvement claim unresolved_below_power." --category health --plain-summary "The team that lost more value to injury tends to underperform the opening line by a little, in the expected direction, but the nudge is too small yet to reliably sharpen the final point-spread prediction."

nfl-ats weak-signals record --name injury_value_lost_margin_map_cover_probability --description "value_lost_diff added as a 5th fitted term to the served 4-term LOSO logistic (model_logit, composition_flag_sum, market_move_toward_home, market_move_available), seasons 2020-2024, point-in-time injury data only, paired against the served 4-term baseline on the same held-out games." --source artifacts/injury_value_margin_map/20260923T210029Z/summary.json --effect -0.8143 --effect-units accuracy_points --classification unresolved_below_power --league nfl --season-start 2020 --season-end 2024 --standard-error 1.4155 --interval-low -3.6351 --interval-high 1.9467 --probability-positive 0.28 --sample-games 1228 --sample-blocks 89 --classification-evidence "Bootstrap probability_positive=0.28 for candidate beating baseline on accuracy (2000 draws, season/week blocks); exact_null_p=0.474 on 158 decisive games (74 candidate wins vs 84 baseline wins); brier/log-loss also cross zero (prob_positive 0.65/0.63). Sign not reversed (not refuted), no positive control run, so unresolved_below_power." --category health --plain-summary "Adding up injuries lost to value and folding that into the pick model did not clearly help or hurt picks over 2020-2024; the effect is too small to tell apart from noise yet."
```

## Next
Both signals above need `weak-signals record` by the root before any
write-up calls blocker 2 settled. Blocker 2 itself remains formally open
(unresolved, not closed) — a margin-point mapping exists now with a
confirmed-sign LOSO slope, but AGENTS.md's calibration bar for trusting a
scenario center is not met (OOS improvement and the fitted-term paired look
both cross zero). Producer steps 1/3/4/5 from the prior Next are unchanged
and still blocked on this being resolved with more power or accepted as
permanently small; do not restore `injury_scenarios.py` on the strength of
this unit alone.

## Open
No Week 3 comparison was produced this session — blocked as above, no
code was written. `ruff` was not run (nothing changed under `src/` or
`scripts/`).
