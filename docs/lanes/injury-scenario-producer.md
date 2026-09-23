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

## Unit 3 (2026-09-23, second session): built and ran, ruff not yet clean

Built `scripts/injury_scenario_producer.py` (research-only, vendors kernel
logic, no `src/` or served-path writes). Ran once for real:
`.tools/uv.exe run python scripts/injury_scenario_producer.py`. Output
`artifacts/injury_scenario_producer/20260923T211155Z/` (`summary.json`,
`week3_scenario_comparison.csv`).

Design (deviations from the literal Unit-3 ask, forced by data reality,
disclosed):
- No 2026 game yet has a `report_status` (Q/D/O) in the latest injury
  snapshot (`data/raw/nflverse_injuries/20260923T203033Z/injuries.parquet`,
  season 2026 week 3, all 22 rows `report_status=NaN`) — too early in the
  week (Tue). Used `practice_status` (DNP=0.25, Limited=0.10 sit
  probability, matching `nfl_ats.availability.fixed_unavailability`'s own
  practice-only fallback branch) as the borderline signal in place of Q/D.
- Did NOT fit fresh historical designation play rates
  (`build_season_lagged_availability_rates` needs an outcomes join of
  injuries x snap counts x games; no consolidated historical snap-counts
  table exists in `data/processed` and no caller of that function exists
  anywhere in the repo — it and `injury_scenarios.py` were both orphaned).
  Used the practice-status fallback severities above as the per-player sit
  probability instead of a fresh fit. This is the single biggest gap vs.
  the literal ask — flagged, not hidden.
- Per-player value-if-out: no per-player decomposition of
  `injury_skill_epa_value_lost`/`injury_defense_disruption_value_lost`
  exists outside the full feature builder. Allocated each side's already-
  served aggregate value-lost equally in severity-weighted proportion
  across that side's borderline players (self-consistent: expected
  scenario value_lost_diff equals the served baseline).
- Pairwise coupling: applied the measured per-unit excess multiplier
  (`docs/absence_pairwise_dependence.md` section 8: OFF_OL 1.192, OFF_SKILL
  1.235, DEF_FRONT 1.175, DEF_SECONDARY 1.245) once per co-sitting pair
  sharing a unit within a scenario subset, then renormalized.
- Margin-center shift: `FITTED_SLOPE_PER_UNIT_VALUE_LOST_DIFF = -0.320`
  (Unit 2's pooled block-bootstrap slope, itself `unresolved_below_power`
  on practical OOS improvement — used here as a research point estimate,
  not a served constant).
- Discrete distribution reuse: shifting the center by `delta` and reading
  cover probability at the served `spread_line` is algebraically identical
  to keeping the center fixed and reading at `line_offset = -delta` in the
  production `line_sweep.parquet` for the active model
  (`artifacts/margin_predictions/2026-week-03-20260923T161033Z/line_sweep.parquet`,
  `method="market_residual"`, column `home_cover_probability`, policy
  `discrete_conditional_non_push_v1` — confirmed this equals the served
  `home_cover_probability` at `line_offset=0` for every game checked). This
  reuses the served discrete margin distribution exactly, satisfying
  AGENTS.md's "Margins are multimodal" without needing the raw residual
  array. Offsets are linearly interpolated on the 0.5-step sweep grid and
  clamped at +/-4.0 (no game needed clamping this run).

Result (16 games, `artifacts/injury_scenario_producer/20260923T211155Z/summary.json`):
only 1 of 16 games (`2026_03_ATL_GB`) has any borderline (DNP/Limited)
player yet — 14 games have zero injury-report rows at all for week 3 this
early (Tuesday), 1 game (ARI@SF) has rows but all "Full Participation".
ATL@GB: 10 borderline players on GB (home), 2 on ATL (away), 4,096
enumerated joint scenarios, `base_value_lost_diff=0.0481`,
`served_home_cover_probability=0.428377`,
`scenario_mixed_home_cover_probability=0.429536`, shift **+0.116
percentage points** (well under 1 pt), pick side unchanged (away/ATL both
ways). **No game showed a shift over 1 point and no game showed a pick-side
difference this run** — an honest null, not a data gap, for the 1 game
with real injury data; the other 15 are nulls by data absence, not by the
mixture math. One data-quality note (not corrected, out of scope): the raw
snapshot lists Tua Tagovailoa and Michael Penix Jr. under `team=ATL`
(both Miami/unrelated), but both are `Full Participation` (severity 0) so
this did not affect the result.

Data coverage for a later out-of-season grade of the mixture: **3,886
historical games (seasons 2010-2024)** in
`data/processed/game_features_player_value.parquet` have both sides'
`*_injury_observed_at` populated (pregame-attested, point-in-time safe) —
the same population Unit 2 scoped from, so the margin-slope and
value-lost-diff inputs already exist for all of them. What is still
missing for a real historical grade of the SCENARIO MIXTURE specifically
(not just the point-estimate slope) is per-player historical
report/practice status joined to actual play/sit outcomes for those same
3,886 games — i.e. exactly the outcomes table (injuries x snap counts x
games) that `build_availability_outcomes` needs and that has no
consolidated processed artifact yet.

## Open / unfinished this call (ruff not clean, cap hit)
`ruff check scripts/injury_scenario_producer.py` reported 10 E501
line-too-long errors. Two were fixed in this session (removed a redundant
ternary at the old line ~173; wrapped `fitted_slope_source`). The edit
wrapping `coupling_source` (old lines ~225-227) was IN FLIGHT when the
50-tool-call cap hit and did NOT apply — the file on disk right now still
has the long `coupling_source` line and likely 1-2 other unseen E501s from
the same `ruff check` run (only the tail of that output was read). The
computed artifact `artifacts/injury_scenario_producer/20260923T211155Z/`
is already correct and complete (ruff cleanliness does not affect
already-computed numbers); only the script's style is unfinished.

## Next
1. Finish ruff: `.tools/uv.exe run ruff check scripts/injury_scenario_producer.py`,
   fix remaining E501s (wrap `coupling_source` string and any others it
   reports), re-run until clean.
2. Re-run the script once more after ruff fixes purely to confirm the
   output is byte-identical (formatting-only changes should not change
   any number) and note the new timestamp artifact if one is produced;
   the 20260923T211155Z artifact already stands as the real result either
   way.
3. Record both Unit 2 signals with `weak-signals record` (commands are
   already fully assembled above, unrun, root's job) before any write-up
   calls this settled.
4. If a real historical grade of the mixture (not just the slope) is
   wanted later: build the outcomes table
   (`nfl_ats.availability.build_availability_outcomes` needs injuries +
   snap-counts-with-gsis-ids + games; no consolidated snap-counts parquet
   exists in `data/processed` today) and fit
   `build_season_lagged_availability_rates` for real, then re-run this
   producer against a completed season's Q/D-reported snapshots (once
   report_status exists, unlike this week's Tuesday snapshot) and grade
   scenario-mixed vs served cover probability against actual results.
5. Do not restore `injury_scenarios.py` into `src/` on the strength of
   this unit; it stays vendored in the research script only.

## Unit 4 (2026-09-23, third session): historical outcomes table built, run once

Built `scripts/injury_outcomes_table.py` (ruff clean). Reused existing
production infrastructure rather than reimplementing: `nfl_ats.availability`
already has `build_availability_outcomes`, `build_season_lagged_availability_rates`,
`score_availability_rates`, `summarize_availability_scores` (previously called
only from the orphaned `_cmd_build_learned_availability_features` CLI path,
per `grep`; no consolidated historical output existed). Sources: latest
consolidated player snapshot `data/players/raw/20260923T204548Z/`
(injuries/rosters/snap_counts, via `latest_player_snapshot` +
`load_player_snapshot`), games/kickoff from `data/processed/game_features.parquet`
(2009-2026). Decision cutoff 24h before kickoff (same constant as the one
existing caller). Rate scheme is **season-lagged / earlier-seasons-only**
(expanding window, matches the task's "other/earlier seasons only" and the
one existing served-feature caller), not a two-sided LOSO — this is a reuse
choice, not a new design.

Ran once for real: `.tools/uv.exe run python scripts/injury_outcomes_table.py`.
Output table: `data/processed/injury_play_outcomes.parquet` (67,989 rows, 20
columns: game_id, season, week, team, gsis_id, position, report_category,
practice_category, position_group, played, unavailable, fixed_unavailability,
observed_at_is_proxy, report_status, practice_status, decision_observed_at,
offense_pct, defense_pct, st_pct, snap_share). Summary:
`artifacts/injury_outcomes_table/20260923T211644Z/summary.json` +
`cell_calibration.csv` in the same directory.

**Coverage deviation from the literal ask, disclosed**: seasons **2013-2025**
(13 seasons), not 2010-2025. The injuries snapshot itself goes back to 2009,
but `build_availability_outcomes` inner-joins to snap-count seasons to know
whether a player actually played, and nflverse snap-count data starts 2013
(confirmed: `snap_counts.parquet` season min is 2013) — this is a source-data
floor, not a code choice. 3,406 games, 5,495 unique players. 63,077 of 67,989
rows are scorable against a fitted rate (2013 itself has no earlier season to
train on and is dropped from scoring, 4,912 rows, same expanding-window edge
case the existing rate-fit code already handles).

**Calibration, fitted (season-lagged) vs fallback (`fixed_unavailability`),
63,077 scored player-games**: Brier 0.0867 (fitted) vs 0.0912 (fixed) —
fitted is better overall. Classification accuracy 0.8827 (fitted) vs 0.8761
(fixed). Biggest single-cell fix: `questionable`+`dnp` (2,701 obs) — fixed
predicts 0.65 play probability, actual is 0.421 (fixed overpredicts playing
by 23pp); fitted predicts 0.356, much closer. `questionable`+`limited`
(10,437 obs) fixed 0.65 vs actual 0.681, fitted 0.678, also closer. Caveat,
not hidden: in a few **sparse** cells fixed's hard 0/1 assignment beats
fitted — e.g. `out`+`full` (n=200, actual play rate 0.0) fixed correctly
predicts 0.0, fitted predicts 0.166; `out`+`none` (n=51, actual 0.0) fixed
0.0, fitted 0.361. These are rare designation/practice combinations
(player ruled out but reported full practice, or no practice status logged)
where the season-lagged rate is noisy; full per-cell table is in
`cell_calibration.csv`.

## Next
1. If the producer (`scripts/injury_scenario_producer.py`) is revisited, its
   per-player `practice_severity()` ad hoc fallback (DNP=0.25, Limited=0.10,
   the same values as `fixed_unavailability`'s practice-only branch) should
   switch to `nfl_ats.availability.resolve_unavailability()` fed by
   `build_season_lagged_availability_rates` fit on this unit's
   `injury_play_outcomes.parquet` (keyed off `report_category`+
   `practice_category`, optionally `position_group`) — `resolve_unavailability`
   already falls back to `fixed_unavailability` when a season/cell has no
   fitted rate, which is the safe integration path given the sparse-cell
   caveat above (do not use the raw fitted rate unguarded on rare cells).
   This also lets the producer use `report_category` (Q/D/O) once a season
   is far enough along to have it, not just practice status.
2. This same table now closes the last remaining Unit-3 gap ("no
   consolidated historical outcomes to grade the scenario mixture itself
   against"): `injury_play_outcomes.parquet` spans 2013-2025 / 3,406 games
   and can be joined against `game_features_player_value.parquet`'s
   point-in-time population (2020-2024 intersection, per Unit 2) to grade
   scenario-mixed vs served cover probability on real outcomes, not just the
   margin-slope point estimate.
3. Not yet run: no `weak-signals record` for this unit — it is an outcomes
   table + calibration measurement of an auxiliary rate fit, not itself a
   pick-probability signal decision. If a fitted rate from this table is
   later wired into a served or research pick path, that wiring is the
   decision point requiring a record.

## Unit 5 predeclaration (2026-09-23, fourth session, before grading)

Part (a): `scripts/injury_scenario_producer.py` severity source switches from
the fixed practice-only fallback to `nfl_ats.availability.resolve_unavailability`-
style season-lagged rates fit on `data/processed/injury_play_outcomes.parquet`
via `build_season_lagged_availability_rates(target_seasons=[2026])` (prior
seasons only, matches the existing contract). Sparse-cell guard: a cell
(report_category, practice_category, position_group or its `__all__`
aggregate) is only trusted if its fitted-rate row has
`observations >= 250` (chosen from Unit 4's own calibration table, where the
`out`+`full` (n=200) and `out`+`none` (n=51) cells were the ones fitted beat
by fixed; `questionable`+`dnp` n=2,701 and `questionable`+`limited` n=10,437
were comfortably fitted-favoring); below the floor, falls back to
`fixed_unavailability`.

Part (b): grading population = seasons 2020-2024 intersection already
established in Unit 2 (`opener_evaluation` per_game via
`pick_probability_fit.build_fit_population`, `game_features_player_value.parquet`
point-in-time rows, i.e. both `*_injury_observed_at` populated). Borderline
players per team-game come from `injury_play_outcomes.parquet` (report/practice
category + position_group), severity resolved the same way as part (a) but
with a genuine leave-earlier-seasons-only rate per target season (2013..s-1
training window, s in 2020..2024) — no fold sees its own season's outcomes.
To bound the joint-scenario enumeration, each side keeps at most the 6
highest-severity borderline players (2^6=64 subsets/side, 4,096 scenarios/game
cap); games/sides needing the cap are counted and disclosed, not silently
dropped.

Margin-shift slope is the season-specific Unit 2 LOSO fold slope (loaded from
`artifacts/injury_value_margin_map/20260923T210029Z/summary.json`
`margin_regression.loso_folds`, e.g. 2020 -0.274 ... 2024 -0.313), i.e. the
2020 games use the slope fit on 2021-2024 only, matching "Unit 2 slope refit
LOSO" literally (a stricter, per-season LOSO, not the single pooled slope
Unit 3's live producer used for 2026).

Cover-probability read: the live producer's technique (shift the margin
center by `delta` == read the discrete distribution at `line - delta` with
center fixed) needs a per-game discrete lattice, which is not persisted for
historical games. Deviation, disclosed: reuse the actual served discrete
primitive `nfl_ats.mass_preserving_lattice.band_read` /
`discrete_margin_mapping.discrete_side_read` directly (not reimplemented),
with `pool_line`/`pool_margin` = `data/processed/game_features.parquet`
`spread_line`/`result` for all seasons strictly before the target game's
season (2009 floor, point-in-time safe), and a per-game `point` (center)
solved once by bisection so that
`discrete_side_read(pool, pool, tue_open_home_spread, point).home_cover_probability`
reproduces that game's own known served `home_cover_probability_at_open`
exactly (this makes every zero-shift scenario collapse back to the served
probability by construction, and keeps the read genuinely nonlinear/discrete,
unlike a local-linear proxy which would erase exactly the curvature effect
Unit 3 found). Scenario probability = the same read at
`line = tue_open_home_spread - margin_shift`. Games where bisection fails to
bracket a root are flagged and fall back to the served probability
(no-shift), counted and disclosed, not silently included as a match.

## Unit 5 complete (2026-09-23, fifth session)

Pre-work: `Get-CimInstance Win32_Process` found 12 stray processes from the
prior session's two hung background tasks (`bhxm0adxc`: 3 bash wrappers +
3 python leaves running `injury_scenario_producer.py`; `bqer9icvr`: 3 bash
wrappers running `ruff check ... && injury_scenario_grade.py` + 3 python
leaves running `injury_scenario_grade.py`), all confirmed by command line as
this repo's `F:\Repos\nfl_py3\scripts\injury_scenario_*.py` — killed all 12
(`Stop-Process -Force`).

Bug 2 fix applied in both files: `SIGNAL_REPORT_CATEGORIES =
frozenset(("out","doubtful","questionable"))`,
`SIGNAL_PRACTICE_CATEGORIES = frozenset(("dnp","limited"))`. In
`scripts/injury_scenario_producer.py`'s `borderline_players`, a player is
skipped unless its report or practice category is in one of those sets,
before `resolve_severity` is used for magnitude. In
`scripts/injury_scenario_grade.py`'s `build_borderline_table`, the outcomes
frame is filtered the same way on its precomputed `report_category`/
`practice_category` columns before severity resolution.

Enumeration cap added to the producer (it previously had none):
`MAX_ENUMERATED_PLAYERS_PER_SIDE = 12`. `side_subsets` now sorts a side's
gated borderline players by `sit_probability` descending, enumerates 2^n
over at most the top 12, and folds any remainder's summed `sit_probability`
into every scenario's `sit_severity` as a constant expected-value addend
(not part of the combinatorics or pairwise coupling) — so a long report
still bounds runtime at 4,096 scenarios/side without silently dropping the
excluded players' contribution to the value-lost estimate. Returns
`(subsets, truncated)` now; caller records `side_truncated_at_cap` per game
and the summary reports `games_with_side_truncated_at_cap` and
`max_enumerated_players_per_side`. The grader's existing
`MAX_BORDERLINE_PER_SIDE=6` (simple truncation, predeclared in Unit 5,
already a hard cap) was left as-is — untouched by this fix.

`ruff check scripts/injury_scenario_producer.py scripts/injury_scenario_grade.py`
-> all checks passed.

**Producer run** (`.tools/uv.exe run python scripts/injury_scenario_producer.py`,
finished well under a minute): output
`artifacts/injury_scenario_producer/20260923T213811Z/`. 16 games,
`severity_source_counts` = `{season_lagged_rate: 12, fixed_status_prior: 0}`
(sparse floor not rejecting everything, gate is working). Only
`2026_03_ATL_GB` has borderline players (10 GB-home, 2 ATL-away, all
DNP/Limited — qualitative gate correctly admits only real DNP/Limited rows,
no cap truncation since 10<12), 4,096 scenarios,
`served_home_cover_probability=0.428377`,
`scenario_mixed_home_cover_probability=0.430144`, shift **+0.177 percentage
points** (up slightly from Unit 3's fixed-severity +0.116, still well under
1 point), pick side unchanged (away/ATL both ways). All other 15 games:
zero borderline players (no report yet this early in the week or all Full
Participation), zero shift. `games_with_side_truncated_at_cap=0`. Same
honest null as Unit 3, now on the corrected qualitative-gate + fitted-rate
severity path instead of the fixed-practice-only fallback.

**Grader run** (`.tools/uv.exe run python scripts/injury_scenario_grade.py`,
finished in well under the 480s timeout): output
`artifacts/injury_scenario_grade/20260923T213822Z/summary.json` +
`scoped_population.parquet`. 1,227 scoped games (seasons 2020-2024,
point-in-time only, same population as Unit 2/5-predeclaration).
`center_bisection_failures=0`, `games_zero_borderline_both_sides_with_nonzero_base_diff=0`.
`games_with_side_truncated_at_cap=793` of 1,227 (real historical injury
reports routinely list more than 6 qualitatively-borderline players/side,
unlike this week's sparse Tuesday snapshot — disclosed, not hidden; the
6-cap's expected-value handling was NOT added this unit, only the producer
got it, per this session's scope decision to leave the grader's already-
predeclared cap untouched).

Cell 1, `raw_scenario_vs_served` (scenario-mixed probability vs served,
paired, decisive_games=70, full/reduced wins 32/38, exact_null_p=0.5504):
accuracy_delta_points **-0.489** [-2.277, 1.100], probability_positive
**0.29225**. Brier improvement -0.0000940 [-0.00124, 0.00102], probability
positive 0.4495. Log-loss improvement -0.000201 [-0.00259, 0.00210],
probability positive 0.448. Per-season accuracy_delta_points: 2020 -0.457,
2021 +0.424, 2022 -3.279, 2023 +0.758, 2024 0.0 — no consistent sign.
**unresolved_below_power** (probability_positive not <=0.025 or >=0.975;
interval crosses zero both directions; not a refuted mechanism).

Cell 2, `fitted_term_scenario_shift_vs_4term_baseline` (scenario_shift as a
5th fitted term on the served 4-term LOSO logistic vs the served 4-term
baseline, decisive_games=50, full/reduced wins 19/31, exact_null_p=0.1189):
accuracy_delta_points **-0.978** [-2.653, 0.239], probability_positive
**0.0695**. Brier improvement -0.000410 [-0.00130, 0.000318], probability
positive 0.178. Log-loss improvement -0.000818 [-0.00261, 0.000654],
probability positive 0.183. Per-season accuracy_delta_points: 2020 0.0,
2021 -0.847, 2022 -2.869, 2023 -1.136, 2024 0.0. **unresolved_below_power**
(probability_positive=0.0695 is directionally suggestive but does not clear
the <=0.025 resolved-directional bar; interval crosses zero; not refuted).

Decision-relevant conclusion: on 2020-2024 historical outcomes, neither
mixing lineup scenarios into the served cover probability nor adding the
scenario shift as a 5th fitted term shows a resolved effect either way;
both point estimates are negative (mixture slightly hurts paired accuracy
on this population) but neither clears the resolved-directional bar, so
per AGENTS.md this stays open/unresolved, not closed.

Record commands for the root (not run — registry write is out of scope
here):
```
nfl-ats weak-signals record --name injury_scenario_mixture_raw_vs_served --description "Joint lineup-scenario mixture (qualitative Q/D/O-or-DNP/Limited gate, season-lagged fitted severity with sparse-cell fallback, top-12-enumerated-plus-expected-value-remainder per side, measured pairwise unit coupling, per-season LOSO margin slope, exact discrete-lattice re-read via bisected center) vs served cover probability, paired on seasons 2020-2024 point-in-time injury population (same scope as Unit 2)." --source artifacts/injury_scenario_grade/20260923T213822Z/summary.json --effect -0.4889975550122249 --effect-units accuracy_points --classification unresolved_below_power --league nfl --season-start 2020 --season-end 2024 --standard-error 0.8860039282576608 --interval-low -2.276983459489766 --interval-high 1.1004249130211567 --probability-positive 0.29225 --sample-games 1227 --sample-blocks 89 --classification-evidence "Bootstrap probability_positive=0.29225 (2000 draws, season/week blocks), exact_null_p=0.5504 on 70 decisive games (32 scenario wins vs 38 served wins); Brier and log-loss also cross zero (probability_positive 0.4495/0.448); per-season accuracy_delta_points has no consistent sign (2020 -0.46, 2021 +0.42, 2022 -3.28, 2023 +0.76, 2024 0.0). Not a refuted mechanism (sign not consistently reversed) and no positive control run, so unresolved_below_power." --category health --plain-summary "Building out realistic game-day injury scenarios and mixing their cover probabilities together did not clearly sharpen or dull picks compared to just using the served number, on five seasons of real outcomes; the difference is too small and inconsistent to call yet."

nfl-ats weak-signals record --name injury_scenario_mixture_fitted_term_vs_4term_baseline --description "Scenario-mixture shift (scenario_mixed_home_cover_probability minus served) added as a 5th fitted term to the served 4-term LOSO logistic (model_logit, composition_flag_sum, market_move_toward_home, market_move_available), paired against the served 4-term baseline on the same seasons 2020-2024 point-in-time population, both refit LOSO by season." --source artifacts/injury_scenario_grade/20260923T213822Z/summary.json --effect -0.9779951100244498 --effect-units accuracy_points --classification unresolved_below_power --league nfl --season-start 2020 --season-end 2024 --standard-error 0.7315433172923517 --interval-low -2.6528599737210365 --interval-high 0.23907253292673966 --probability-positive 0.0695 --sample-games 1227 --sample-blocks 89 --classification-evidence "Bootstrap probability_positive=0.0695 (2000 draws, season/week blocks) is directionally suggestive (candidate worse) but does not clear the <=0.025 resolved-directional bar; exact_null_p=0.1189 on 50 decisive games (19 candidate wins vs 31 baseline wins); Brier/log-loss also cross zero (probability_positive 0.178/0.183). Not refuted (interval not entirely on one side) and no positive control run, so unresolved_below_power." --category health --plain-summary "Adding the injury-scenario mixture as one more fitted ingredient in the pick model, instead of using it standalone, still did not clearly help or hurt picks over 2020-2024 -- it leans toward hurting a little but not enough to call it settled."
```

## Unit 5 status (2026-09-23, fourth session, superseded above): NOT complete, 50-call cap hit

Both scripts are written and ruff-clean but neither has produced a real
result yet. Two real bugs were found and one is still unfixed. Do not trust
any artifact under `artifacts/injury_scenario_grade/` or a fresh
`artifacts/injury_scenario_producer/` run from this session without
re-checking against the fixes below.

**Bug 1 (found, fixed in both files): coupling multiplier used the wrong
unit.** In `side_subsets` (both `scripts/injury_scenario_producer.py` and
`scripts/injury_scenario_grade.py`), the original Unit-3 code counted
same-unit sitting pairs, then applied `UNIT_COUPLING_MULTIPLIER[sitting[0]["unit"]]`
once per pair — using the FIRST sitting player's unit for every pair
regardless of which unit that pair actually shared. This crashed
(`KeyError: 'other'`) the first time a game had a sitting player whose unit
is `"other"` (kicker/punter/long-snapper position groups) ahead of a same-
unit pair elsewhere in the sitting list — which the old fixed-practice-only
severity never surfaced enough borderline players to trigger, but the new
fitted-rate severity does. Fixed in both files: apply
`UNIT_COUPLING_MULTIPLIER[left["unit"]]` per matching pair directly, no
`sitting[0]` indirection. **This fix is already saved on disk in both
files** (confirmed via the successful Edit calls before the cap hit).

**Bug 2 (found, NOT yet fixed anywhere): fitted rates make almost every
player "borderline," causing combinatorial hang.** `practice_severity()`
(the old fixed-only function) returned exactly `0.0` for
`Full Participation`/no-report players, so only real DNP/Limited/Q-D-O
players ever entered the scenario enumeration (small n, fast). The new
`resolve_severity()` returns whatever the season-lagged fitted rate is for
a cell, and a `(report=none, practice=full, position_group=X)` cell can have
a small but nonzero fitted rate (normal game-day-inactive noise in the
2013-2025 training data) that clears the 250-observation sparse floor. That
turns nearly every listed player into a "borderline" player with tiny
severity, blowing up `side_subsets`'s `2**n` enumeration per side (n can be
15+ for a full injury report) — `scripts/injury_scenario_producer.py`'s
week-3 background run (task `bhxm0adxc`) hung with zero output for 6+
minutes before the cap hit and was never confirmed to finish; presumed
still running or effectively stuck. A second producer run plus grading run
(`bqer9icvr`) were also launched in the background after Bug 1's fix but
before Bug 2 was diagnosed, so they carry the same hang risk and their
output must not be trusted without verifying they actually completed.

**Required fix, not yet applied**: gate "borderline" on a qualitative signal
before ever computing/using the fitted magnitude, in both
`borderline_players` (producer) and `build_borderline_table` (grader): only
keep a player if `report_category in {"out","doubtful","questionable"}` OR
`practice_category in {"dnp","limited"}` (i.e. never treat a `none`/`full`
row as borderline regardless of what a sparse-cell-cleared fitted rate says
— the fitted rate should only refine the MAGNITUDE for players who already
show a real signal, not manufacture new borderline players out of normal
noise). The grading script also already has a 6-per-side severity cap
(`MAX_BORDERLINE_PER_SIDE = 6`) as a second safety net once the gate is
added; the producer script has no such cap and should probably get one too
now that fitted rates are wired in (even gated players could exceed 6 on a
long Wednesday/Thursday report later in a real week). `SIGNAL_REPORT_CATEGORIES`
and `SIGNAL_PRACTICE_CATEGORIES` constants were being added to
`scripts/injury_scenario_producer.py` when the cap hit — the edit with those
two frozensets did NOT apply (cap hit mid-call); the file on disk right now
has Bug 1's fix but NOT the qualitative gate.

## Unit 5 Next (for the root / a fresh subagent)

1. In `scripts/injury_scenario_producer.py`: add
   `SIGNAL_REPORT_CATEGORIES = frozenset(("out","doubtful","questionable"))`
   and `SIGNAL_PRACTICE_CATEGORIES = frozenset(("dnp","limited"))` (module
   level, after the `nfl_ats.availability` imports), then in
   `borderline_players` compute `report_cat`/`practice_cat` via
   `availability_report_category(row.get("report_status"))` /
   `availability_practice_category(row["practice_status"])` and skip the
   player unless `report_cat in SIGNAL_REPORT_CATEGORIES or practice_cat in
   SIGNAL_PRACTICE_CATEGORIES`, BEFORE calling `resolve_severity` (or call
   it and just gate on the result — either order is fine, but the gate must
   exist). Same gate in `scripts/injury_scenario_grade.py`'s
   `build_borderline_table` using `row.report_category`/`row.practice_category`
   (already precomputed columns on the `injury_play_outcomes.parquet` rows,
   no need to re-normalize).
2. Kill any stray background python processes from this session first
   (task ids `bhxm0adxc`, `bqer9icvr` — check with the shell's job list /
   `tasklist | grep python` on Windows; if still running they are almost
   certainly hung on the combinatorial bug above, not doing useful work).
   Confirmed post-handback: `bhxm0adxc` finished on its own with
   **exit code 255 (failed)**, not an infinite hang — consistent with the
   Bug 2 diagnosis (most likely the same `KeyError`/combinatorial-blowup
   family, or a related crash from the unguarded fitted-rate severity
   change) rather than a true hang. Its output file is
   `C:\Users\Ryan\AppData\Local\Temp\claude\F--Repos-nfl-py3\5ea705ef-c6be-4b65-8730-46dcbf2a8514\tasks\bhxm0adxc.output`
   (not read this session — read it first before re-running, it likely has
   the exact traceback). `bqer9icvr` also confirmed finished, also
   **exit code 255 (failed)** — output file
   `C:\Users\Ryan\AppData\Local\Temp\claude\F--Repos-nfl-py3\5ea705ef-c6be-4b65-8730-46dcbf2a8514\tasks\bqer9icvr.output`
   (not read this session either). Both background runs are confirmed
   over, neither is still occupying anything, and neither needs to be
   killed — the "kill stray processes" step above is now moot; go straight
   to reading both output files for the exact tracebacks, apply the Bug 2
   qualitative-gate fix, then re-run fresh.
3. Re-run `.tools/uv.exe run ruff check scripts/injury_scenario_producer.py
   scripts/injury_scenario_grade.py`, fix anything new, then run the
   producer once
   (`.tools/uv.exe run python scripts/injury_scenario_producer.py`) and
   confirm it finishes in well under a minute now that the gate bounds `n`.
   Sanity-check `severity_source_counts` in its summary.json — expect a mix
   of `season_lagged_rate` and `fixed_status_prior`, not all-fixed (that
   would mean the sparse floor of 250 is rejecting everything, worth a
   second look) and not a huge borderline-player blowup (that would mean
   the gate didn't take).
4. Run the grader once
   (`.tools/uv.exe run python scripts/injury_scenario_grade.py`) — expect a
   few minutes given the per-game bisection over ~1,200+ scoped games; if it
   is still slow, the next lever is lowering `MAX_BORDERLINE_PER_SIDE` from
   6 before touching anything else. Read `summary.json`'s
   `raw_scenario_vs_served` and `fitted_term_scenario_shift_vs_4term_baseline`
   cells (each already has `accuracy_delta_points`, `probability_positive`,
   `brier_improvement`/`brier_probability_positive`,
   `log_loss_improvement`/`log_loss_probability_positive`, `decisive_games`,
   `full_decisive_wins`/`reduced_decisive_wins`, `exact_null_p`, and a
   per-season `seasons` breakdown for season-block intervals) plus
   `games_zero_borderline_both_sides_with_nonzero_base_diff` and
   `center_bisection_failures` for disclosure. Classify each cell with the
   same `probability_positive>=0.975 or <=0.025 -> resolved_directional`
   rule as Unit 2 (else `unresolved_below_power`), then write two
   `nfl-ats weak-signals record --category health --plain-summary ...`
   commands (one per cell, same style/fields as Unit 2's two commands
   above) and add them to this lane before any write-up calls this
   settled. Neither cell's numbers exist yet — do not fabricate or guess
   them; run the script for real first.
5. If the qualitative gate alone doesn't bound runtime for some outlier
   game (a real Q/D-heavy week can list 10+ players even after the gate),
   the existing `MAX_BORDERLINE_PER_SIDE=6` truncation in the grader (and a
   same-style cap worth adding to the producer) is the disclosed release
   valve — `games_with_side_truncated_at_cap` in the grader's summary
   already counts how often this binds.

Reported (per AGENTS.md calibration + "count every look"): for (1) raw
scenario-mixed vs served probability and (2) scenario-shift added as a 5th
fitted term (`scenario_mixed_home_cover_probability - served_home_cover_probability`)
on top of the served 4-term LOSO logistic (`FIT_FEATURES`) refit LOSO by
season — both via `nfl_ats.signal_atlas._cell` (same machinery Unit 2 used):
accuracy_delta_points, decisive-game record with exact binomial p, Brier and
log-loss improvement with block-bootstrap 95% CI and probability_positive,
and the per-season breakdown (season-block intervals). Classification uses
the same `probability_positive>=0.975 or <=0.025 -> resolved_directional`
rule as Unit 2; anything else `unresolved_below_power`, no interval-crossing
closures. Script: `scripts/injury_scenario_grade.py`. Output:
`artifacts/injury_scenario_grade/<ts>/`. No `src/` or served-path changes,
run once.

(Steps 1-5 above are now DONE — see "Unit 5 complete" earlier in this file
for the real fix, the real producer/grader runs, both real result cells,
and the two fully-assembled record commands. This section is kept verbatim
as history of the predeclared plan; do not redo it.)

## Next (for the root, 2026-09-23 fifth session handback)
1. Run the two `nfl-ats weak-signals record` commands in "Unit 5 complete"
   above (`injury_scenario_mixture_raw_vs_served`,
   `injury_scenario_mixture_fitted_term_vs_4term_baseline`) — registry
   write is the root's job, out of scope for this subagent.
2. Both cells are `unresolved_below_power`; neither closes blocker 1/2 from
   the original "Next" section, and neither promotes the scenario mixture
   to a served or research-path input. Do not restore `injury_scenarios.py`
   into `src/` on the strength of this unit.
3. If revisited later: `games_with_side_truncated_at_cap=793/1227` in the
   grader shows the 6-per-side cap binds often on real historical reports;
   an expected-value-remainder upgrade (same style as the producer's new
   12-cap) could be added to the grader if truncation bias is ever
   suspected of masking a real effect — not needed to close this unit.
