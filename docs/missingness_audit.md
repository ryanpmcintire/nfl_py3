# Missingness audit: source-era indicators on the production model

WP15 / ROADMAP MOD-13. The initial diagnostic below reports what
`scripts/missingness_audit.py` measured over the production margin pipeline;
the predeclared follow-up was subsequently executed on 2026-09-02 and is
recorded in the Stage 2 section below. **The diagnostic itself** ran no
experiment, recorded nothing to either registry, and touched no production
file. Every diagnostic number below is **measured** by running
the command shown, on `data/processed/game_features_weak_stack.parquet`
(4,902 rows, 275 columns) as it exists in this working tree on 2026-09-01,
against `artifacts/active_ats_model.json` (model_id `d1f07d773475dc58`,
`feature_profile: weak_stack`, `method: market_residual`, `ridge_alpha:
10.0`, activated 2026-08-24, weekly forecast = 2026 Week 1).

## Why this diagnostic exists

The production pipeline (`src/nfl_ats/margin.py::make_margin_estimator`, read
2026-09-01) is `SimpleImputer(strategy="median", add_indicator=True)` ->
`StandardScaler` -> `Ridge(alpha=10.0)`. `add_indicator=True` appends one
binary "this column was missing" feature for every training column that had
*any* missing value in the fit data. For a feature whose underlying source
only exists from some season onward (e.g. a stat that requires
play-by-play-derived inputs not backfilled to 2009), that indicator is not
really encoding "missing" — in-sample it is almost perfectly encoding *which
era the game is from*, because every pre-source-season row is missing and
every post-source-season row is not. Walk-forward evaluation never sees this
as a problem (both eras are represented in every backtest fold). The risk is
narrower and sharper: on a single live prediction row, if that row's
missing/present pattern for some column essentially never occurred in
training, the fitted indicator coefficient for that column is being applied
to a case the ridge fit was never actually informed by, and the model is
extrapolating rather than interpolating on that row.

## Command

```
./.tools/uv.exe run --no-sync python scripts/missingness_audit.py
```

(`--json` prints the same data as a machine-readable payload — full
per-column-per-season matrix, full per-(column, game) risk table, and the
coefficient summary — for downstream tooling; the Markdown mode below is the
human-readable summary of the same computation.)

## 1. Source-era columns (measured)

**Measured**, Appendix (raw script output) section "1. Season-level
classification." Of the production `weak_stack` /
`market_residual` feature set (90 columns, **measured**:
`nfl_ats.margin.margin_feature_columns("market_residual", "weak_stack")`),
classified over the fully realized 2009-2025 seasons only (2026 is excluded
from classification because only Week 1 has real inputs so far — a partially
played season would masquerade as a step transition):

| category | count | meaning |
|---|---|---|
| `source_era` | 7 | >=95% missing in some season, <=5% missing in another — a data source switching on |
| `always_missing` | 0 | missing in every 2009-2025 row (none found) |
| `sporadic` | 62 | some missingness, not a season step function |
| `complete` | 21 | never missing in 2009-2025 |

All 7 `source_era` columns are the lineup-continuity family, and they share
one transition:

| column | last >=95%-missing season | first <=5%-missing season | min frac | max frac |
|---|---|---|---|---|
| `diff_defense_lineup_continuity` | 2012 | 2014 | 0.000 | 1.000 |
| `diff_front_lineup_continuity` | 2012 | 2014 | 0.000 | 1.000 |
| `diff_offense_lineup_continuity` | 2012 | 2014 | 0.000 | 1.000 |
| `diff_offensive_line_continuity` | 2012 | 2014 | 0.000 | 1.000 |
| `diff_secondary_lineup_continuity` | 2012 | 2014 | 0.000 | 1.000 |
| `diff_skill_lineup_continuity` | 2012 | 2014 | 0.000 | 1.000 |
| `diff_special_teams_lineup_continuity` | 2012 | 2014 | 0.000 | 1.000 |

**Measured** per-season trajectory (missing fraction, REG rows), identical
for all seven columns: `2009:1.00 2010:1.00 2011:1.00 2012:1.00 2013:0.12
2014:0.00 2015:0.00 ... 2025:0.00 2026:0.00`. 2013 is a genuine transition
year (12% missing, not a clean jump), not a data-quality anomaly — the
classifier still calls it `source_era` because 2012 and earlier are 100%
missing and 2014 onward is 0% missing. **Inferred** (not read from a
constant, since no `start_season` constant for this family was found by
grepping `src/nfl_ats/*.py`): the mechanism is plausibly nflreadpy roster
continuity/participation coverage beginning around 2012-2013, but that is a
guess, not a verified cause — the measured trajectory above is the fact that
matters for this audit, independent of why.

Everything else in the production set — the market/Elo/rest/weather
columns, all EPA/CPOE/yardage/turnover/sack-rate blocks, the QB blocks, the
injury blocks, the roster-continuity *rate* columns (as opposed to the 7
`source_era` lineup-continuity columns above), and the bias columns — is
either `complete` or `sporadic` (season-to-season noise, or missing only on
specific weeks such as `bias_week2_anchor_*`, which is structurally absent
outside week 2 by construction, or `temp`/`wind`, which are absent for dome
games in every season at a roughly stable ~25-65% rate, not a step
function). Full per-column min/max/mean fractions for all 62 `sporadic`
columns are in the Appendix (raw script output), section "sporadic columns"); the
full 90 x 18 (column x season) matrix is in `--json` output, not reproduced
here because it is 1,620 rows.

Notably: **all 9 `diff_injury_*`/`diff_injury_*_value_lost` columns are
`complete`** (never missing 2009-2025) — consistent with the standing note
in `src/nfl_ats/constants.py` (read, near `football_weak_stack` definition)
that this table's injury columns "carry LEARNED availability semantics by
construction," i.e. they are themselves already imputed/modeled upstream of
this feature table, not raw-missing. That is a relevant fact for stage 2
below: the injury family already solved the exact problem this audit is
about, by construction, for itself — it just did not extend that treatment
to the roster-continuity family.

## 2. 2026 Week 1 extrapolation-risk list

**Measured**, Appendix (raw script output), section 2: **none**. Every one of the 90
production columns, for every one of the 16 2026 Week 1 games, has a
missing/present state that occurred in at least 1% of the 272 2025
regular-season rows (the reference population for this check — the season
immediately prior to the target week, not the full 4,431-row 2009-2025
training pool used for the coefficient fit in section 3). The closest case
across all 90 x 16 = 1,440 (column, game) pairs was `bias_prior_week_ats_home`
missing, a state that occurred in 5.9% of 2025 rows (**measured** from the
`--json` payload's `week1_risk` records) — comfortably above the 1%
threshold, not a near-miss. This is the item that matters for the
Sep 8 lock: **there is no evidence of an imputer/ridge extrapolation risk on
the live 2026 Week 1 card from this mechanism.** Concretely (**measured**
from the `--json` payload), all 7 `source_era` continuity columns from
section 1 are non-missing (present) on all 16 2026 Week 1 games, and
"present" was the ~100% common state in 2025 (present every season since
2014) — so no source-era column is extrapolating this week. No other
column, in any of the 90 x 16 (column, game) pairs, presented a state 2025
rarely or never saw.

This is a point-in-time result, not a standing guarantee — it should be
re-run (the command above, `--season`/`--week` default to `2026`/`1` but are
overridable) before each future week's lock, since a newly-injured player,
a newly bye'd team, or any other week-specific missingness pattern could in
principle create a rare state in a different week even though Week 1 is
clean. `tests/test_missingness_guard.py::test_current_lock_missingness_states_are_not_rare_relative_to_prior_season`
(added by this package, see below) re-checks this automatically against
whatever the current locked-but-unplayed week is, every time it runs.

## 3. Indicator-coefficient summary

The active model's fitted pipeline is **not** persisted anywhere on disk:
**measured**, its weekly-forecast artifact directory
`artifacts/margin_predictions/2026-week-01-20260824T120725Z/` contains only
`predictions.csv`, `recommendations.csv`, `pool_card.csv`,
`straight_up_pool_*.{csv,md}`, `line_sweep.parquet`, `prediction_safety.json`
and `metadata.json` — no `model.joblib`. **Read**:
`src/nfl_ats/cli.py::_cmd_margin_predict` (the `margin-predict` command that
produced that artifact) calls `score_outcome_week` -> (via
`nfl_ats.outcomes._target_and_models_for_week` /
`_fit_week_models`) `fit_margin_model(training, target="market_residual",
model_name="ridge", feature_profile="weak_stack", ridge_alpha=10.0)` and
never calls `joblib.dump`. (A *different* command, `predict`, does
`joblib.dump` a `model.joblib` at `src/nfl_ats/cli.py:4117`, but that is a
separate, older classification-model code path and its artifact directories
are not the one the active manifest points at.) So per the work order's
fallback instruction, `scripts/missingness_audit.py` **refits the identical
production recipe in memory** — same training-cutoff logic
(`gameday < 2026-Week-1-earliest-kickoff`, `result.notna()`,
`regular_season_rows`), same 90 columns, same `ridge_alpha=10.0`, same
`random_state=42` default — and inspects that pipeline. Nothing is written
to disk by this script.

**Measured** (Appendix, section 3): 4,431 training rows (REG,
2009-2025, `gameday` through 2026-01-04), 90 real features, 69 indicator
features (= 90 minus the 21 `complete` columns above, exactly, a consistency
check that passed). Standardized (post-`StandardScaler`) `|coefficient|`:

| | real features | indicator features |
|---|---|---|
| median | 0.252 | 0.00775 |
| max | 2.840 | 1.163 |

Ratio of medians (indicator / real): **0.031** — the typical indicator
coefficient is about 3% the size of the typical real-feature coefficient, so
on the whole the model leans on missingness indicators only lightly. That
average hides two clear outliers, though — the top of the ranked indicator
list:

**Measured** (re-ranking all 159 real + indicator coefficients together by
`|coefficient|`, not shown in the script's default Markdown output but
recomputed directly from the same refit pipeline for this check):

| indicator on column | abs standardized coef | rank of 159 (indicator) | rank of 159 (its own real feature) |
|---|---|---|---|
| `diff_active_roster_continuity` | 1.163 | **14** | 32 (real value's own coef: 0.487) |
| `diff_active_roster_mean_experience` | 1.059 | **18** | 75 (real value's own coef: 0.073) |
| `bias_week2_anchor_home` | 0.134 | modest | — |
| (all other 66 indicators) | <= 0.067 | below the real-feature median (0.252) | — |

Both top indicators are `sporadic` (section 1), not `source_era` — max
missing fraction 0.375 and 0.332 respectively, mean 0.125 and 0.122,
plausibly week-shaped (early-season weeks with insufficient in-season roster
history) rather than season-shaped — and section 2 found no rare
2026-Week-1 state for either, so neither is flagged as a Week 1 risk by this
audit's own criterion. The finding worth carrying into stage 2, though, is
sharper than "the model leans on some indicators": for both of these
columns, **the missing-indicator coefficient outranks the real feature's own
coefficient** — rank 14 vs. 32, and rank 18 vs. 75 — meaning "was this
column populated at all" currently carries more standardized weight in the
prediction than the column's actual value does. That is exactly the shape of
problem this audit was commissioned to look for, even though it did not
surface as a live-card risk this week; it is a stronger reason to fold these
two columns into stage 2's redesign than their `sporadic` (not `source_era`)
label alone would suggest.

## Stage 2 design (predeclared 2026-09-01, before scoring)

This package is diagnostic-only; the following is a specification for a
future work package to execute, not a result.

- **Candidate**: the production `weak_stack` feature set with the 7
  `source_era` lineup-continuity indicator-bearing columns' *implicit*
  missing-indicator treatment replaced by an *explicit* per-source
  availability flag (e.g. `roster_continuity_data_available`, one flag
  shared by the whole family since they transition together per section 1,
  rather than 7 separate learned indicator coefficients from
  `add_indicator=True`). Whether to also address the two high-coefficient
  `sporadic` indicators (`diff_active_roster_continuity` /
  `diff_active_roster_mean_experience`) found in section 3 is an open design
  question for that package, not resolved here.
- **Comparator**: production `weak_stack` / `market_residual`, unchanged
  (`ridge_alpha=10.0`).
- **Grading**: report **both** the close grade and the opener grade,
  following this repository's own precedent (`AGENTS.md`'s MOD-07 example
  reports both 51.57% close and 52.83% open for the same paired games). Per
  `AGENTS.md`'s binding rule ("Grade the decision at the OPENER. A
  close-graded number may never veto a play."), **the opener grade governs
  any promotion decision** — the close grade is reported for transparency
  only and must not be used to reject the candidate. (The work order for
  this package specified "grade close"; this doc deviates from that literal
  instruction to the extent of also requiring the opener grade as the
  decision-governing number, per the binding AGENTS.md rule, since nothing
  is actually decided in this diagnostic-stage package and the later
  execution package must not re-run the MOD-07 mistake.)
- **Window**: rotation-assigned, not hand-picked — declare a family name
  (e.g. `missingness_availability_flags`) and draw it via `nfl-ats rotation
  assign --name missingness_availability_flags` at execution time, per
  `docs/` convention (`opener-windows-are-not-scarce`: windows retire
  per-family, a spent block can be redrawn, reuse is not penalized beyond a
  stated discount).
- **Metric**: `accuracy_points` (forced-pick ATS accuracy delta vs.
  comparator), paired, bootstrapped, week- and season-blocked, matching
  every other feature-arm comparison in this repository.
- **Positive control**: before trusting a null result, confirm the
  comparison instrument can detect an effect of the size being tested for.
  This repository's established idiom for that is a deliberately leaky
  "oracle" variant (precedent: `weak_stack_v4`'s wind oracle,
  `docs/weak_stack_v4.md`, `FEATURE_SETS["football_weak_stack_oracle_weather"]`
  in `src/nfl_ats/constants.py`) — construct an oracle candidate that leaks
  the *actual* 2026 source-availability pattern back into the training
  indicator (unrealistic, single-purpose), confirm the paired-comparison
  instrument detects that injected effect, and only then treat a null on
  the real (non-leaky) candidate as informative rather than merely
  underpowered.
- **Closing grounds** (binding, restated verbatim per AGENTS.md/CLAUDE.md
  since this doc is not itself a subagent prompt but the eventual execution
  package's prompt must carry this taxonomy): an interval or CI that
  contains zero is **never** grounds to reject, fail, or close this
  candidate. Only two things justify closing this line of work once run:
  (1) a **refuted mechanism** — a RESOLVED wrong sign (whole interval below
  zero) or zero split-half reliability; (2) **bounded by a positive
  control** — the oracle instrument above proven able to detect an effect
  this size, and the real candidate's effect absent. Everything else is
  `unresolved_below_power`: record it with `nfl-ats weak-signals record`
  and report `probability_positive`, never "contains zero."

## Stage 2 execution (2026-09-02)

**Decision implication (measured):** on the rotation-assigned 2020-2021
opener window, the candidate and production incumbent made exactly the same
456 forced picks (53.7281% each; delta **+0.0000 accuracy points**,
`probability_positive=0.000`). There is therefore no measured expected-value
reason to alter the production card, and no card or active-model file changed.

**Measured:** the predeclaration predates the scoring: this file's creation and
last-write time is `2026-09-01T19:08:08Z`; the three run artifacts were written
on `2026-09-02`. The execution declared and assigned the fresh
`missingness_availability_flags` opener family before scoring; its deterministic
window was `[2020, 2021]`. The candidate retained all seven source-era values,
suppressed only their seven automatic imputer indicators via the candidate-only
`SelectiveMissingnessImputer`, and added one explicit
`roster_continuity_data_available` flag. Other production missingness indicators
were unchanged. The code and its identity/leakage regressions are
`src/nfl_ats/missingness_availability.py`, `src/nfl_ats/margin.py`, and
`tests/test_missingness_source_availability.py`.

The frozen sequence ran once, using
`scripts/missingness_source_availability_on_production.py`:

- `null` -> `artifacts/mod13_source_availability/20260902T155357Z/results.json`.
  Its opener paired delta was exactly `0.0000` on all 200 within-week
  permutations.
- `positive-control` ->
  `artifacts/mod13_source_availability/20260902T155425Z/results.json`.
- `screen` ->
  `artifacts/mod13_source_availability/20260902T155452Z/results.json`, with
  `predictions_opener.parquet` and `predictions_close.parquet` retained beside
  the results.

**Measured screen:** the primary opener grade has 456 games / 35 weeks:
production and candidate each scored 53.7281%, delta +0.0000 points,
week- and season-blocked intervals `[0.0000, 0.0000]`,
`probability_positive=0.000`. The transparent close secondary has 524 games /
35 weeks: both arms scored 53.4351%, delta +0.0000 points, the same two
intervals, and `probability_positive=0.000`.

### Control-protocol correction and recorded verdict

**Measured:** an implementation-time target-leak control wrote realized
`ats_margin` into the explicit availability flag. It was selected before any
MOD-13 score was visible and detected +43.8596 opener accuracy points,
week-blocked 95% `[+38.4607, +49.6615]`,
`probability_positive=1.000` (close: +46.5649 points,
`probability_positive=1.000`). It demonstrates gross harness sensitivity only.

**Read:** that control is not the literal oracle specified in the predeclaration
above, which called for the 2026 source-availability pattern, and it does not
demonstrate sensitivity at the candidate-sized +0.0000-point effect. It cannot
support `positive_control_bound`. The first terminal entries were therefore
corrected through both recorder CLIs, rather than hand-edited: weak-signal key
`mod13_source_availability_on_production_opener` is
`unresolved_below_power`, and rotation family
`missingness_availability_flags` is `open` with its exact same spent `[2020,
2021]` window recorded `unresolved`. The correction retains assignment, spend,
artifact, season, and window-kind provenance and carries no closing ground.
The window remains spent; the result is not a closure and must not be described
as one.

## Appendix: raw script output

Verbatim output of the command in the "Command" section above, captured
2026-09-01 (**measured**; this is the actual stdout, not a transcription):

```
# Missingness audit output

Command:

    ./.tools/uv.exe run --no-sync python scripts/missingness_audit.py

Production profile `weak_stack` / target `market_residual`: 90 feature columns (measured: `nfl_ats.margin.margin_feature_columns('market_residual', 'weak_stack')`).

## 1. Season-level classification (2009-2025 fully realized seasons)

- source_era: 7 columns
- always_missing: 0 columns
- sporadic: 62 columns
- complete: 21 columns

### source_era columns (7)

| column | source_begin_season | last_high_missing_season | min_frac | max_frac |
|---|---|---|---|---|
| diff_defense_lineup_continuity | 2014.0 | 2012.0 | 0.000 | 1.000 |
| diff_front_lineup_continuity | 2014.0 | 2012.0 | 0.000 | 1.000 |
| diff_offense_lineup_continuity | 2014.0 | 2012.0 | 0.000 | 1.000 |
| diff_offensive_line_continuity | 2014.0 | 2012.0 | 0.000 | 1.000 |
| diff_secondary_lineup_continuity | 2014.0 | 2012.0 | 0.000 | 1.000 |
| diff_skill_lineup_continuity | 2014.0 | 2012.0 | 0.000 | 1.000 |
| diff_special_teams_lineup_continuity | 2014.0 | 2012.0 | 0.000 | 1.000 |

Per-season trajectory (missing fraction, REG rows only):

- `diff_defense_lineup_continuity`: 2009:1.00 2010:1.00 2011:1.00 2012:1.00 2013:0.12 2014:0.00 2015:0.00 2016:0.00 2017:0.00 2018:0.00 2019:0.00 2020:0.00 2021:0.00 2022:0.00 2023:0.00 2024:0.00 2025:0.00 2026:0.00
- `diff_front_lineup_continuity`: 2009:1.00 2010:1.00 2011:1.00 2012:1.00 2013:0.12 2014:0.00 2015:0.00 2016:0.00 2017:0.00 2018:0.00 2019:0.00 2020:0.00 2021:0.00 2022:0.00 2023:0.00 2024:0.00 2025:0.00 2026:0.00
- `diff_offense_lineup_continuity`: 2009:1.00 2010:1.00 2011:1.00 2012:1.00 2013:0.12 2014:0.00 2015:0.00 2016:0.00 2017:0.00 2018:0.00 2019:0.00 2020:0.00 2021:0.00 2022:0.00 2023:0.00 2024:0.00 2025:0.00 2026:0.00
- `diff_offensive_line_continuity`: 2009:1.00 2010:1.00 2011:1.00 2012:1.00 2013:0.12 2014:0.00 2015:0.00 2016:0.00 2017:0.00 2018:0.00 2019:0.00 2020:0.00 2021:0.00 2022:0.00 2023:0.00 2024:0.00 2025:0.00 2026:0.00
- `diff_secondary_lineup_continuity`: 2009:1.00 2010:1.00 2011:1.00 2012:1.00 2013:0.12 2014:0.00 2015:0.00 2016:0.00 2017:0.00 2018:0.00 2019:0.00 2020:0.00 2021:0.00 2022:0.00 2023:0.00 2024:0.00 2025:0.00 2026:0.00
- `diff_skill_lineup_continuity`: 2009:1.00 2010:1.00 2011:1.00 2012:1.00 2013:0.12 2014:0.00 2015:0.00 2016:0.00 2017:0.00 2018:0.00 2019:0.00 2020:0.00 2021:0.00 2022:0.00 2023:0.00 2024:0.00 2025:0.00 2026:0.00
- `diff_special_teams_lineup_continuity`: 2009:1.00 2010:1.00 2011:1.00 2012:1.00 2013:0.12 2014:0.00 2015:0.00 2016:0.00 2017:0.00 2018:0.00 2019:0.00 2020:0.00 2021:0.00 2022:0.00 2023:0.00 2024:0.00 2025:0.00 2026:0.00

### sporadic columns (62)

| column | min_frac | max_frac | mean_frac |
|---|---|---|---|
| away_ats_residual | 0.000 | 0.188 | 0.011 |
| away_def_epa_per_play | 0.000 | 0.188 | 0.011 |
| away_def_pass_epa_per_play | 0.000 | 0.188 | 0.011 |
| away_def_rush_epa_per_play | 0.000 | 0.188 | 0.011 |
| away_def_sack_rate | 0.000 | 0.188 | 0.011 |
| away_def_takeaway_rate | 0.000 | 0.188 | 0.011 |
| away_def_yards_per_play | 0.000 | 0.188 | 0.011 |
| away_off_cpoe | 0.000 | 0.188 | 0.011 |
| away_off_epa_per_play | 0.000 | 0.188 | 0.011 |
| away_off_pass_epa_per_play | 0.000 | 0.188 | 0.011 |
| away_off_rush_epa_per_play | 0.000 | 0.188 | 0.011 |
| away_off_sack_rate | 0.000 | 0.188 | 0.011 |
| away_off_turnover_rate | 0.000 | 0.188 | 0.011 |
| away_off_yards_per_play | 0.000 | 0.188 | 0.011 |
| away_point_diff | 0.000 | 0.188 | 0.011 |
| away_team_games | 0.000 | 0.062 | 0.004 |
| bias_prior_week_ats_away | 0.059 | 0.062 | 0.061 |
| bias_prior_week_ats_diff | 0.059 | 0.066 | 0.062 |
| bias_prior_week_ats_home | 0.059 | 0.062 | 0.061 |
| bias_week2_anchor_away | 0.000 | 0.004 | 0.000 |
| bias_week2_anchor_diff | 0.000 | 0.008 | 0.000 |
| bias_week2_anchor_home | 0.000 | 0.004 | 0.000 |
| diff_active_roster_continuity | 0.000 | 0.375 | 0.125 |
| diff_active_roster_mean_experience | 0.000 | 0.332 | 0.122 |
| diff_ats_residual | 0.000 | 0.188 | 0.011 |
| diff_def_epa_per_play | 0.000 | 0.188 | 0.011 |
| diff_def_pass_epa_per_play | 0.000 | 0.188 | 0.011 |
| diff_def_rush_epa_per_play | 0.000 | 0.188 | 0.011 |
| diff_def_sack_rate | 0.000 | 0.188 | 0.011 |
| diff_def_takeaway_rate | 0.000 | 0.188 | 0.011 |
| diff_def_yards_per_play | 0.000 | 0.188 | 0.011 |
| diff_off_cpoe | 0.000 | 0.188 | 0.011 |
| diff_off_epa_per_play | 0.000 | 0.188 | 0.011 |
| diff_off_pass_epa_per_play | 0.000 | 0.188 | 0.011 |
| diff_off_rush_epa_per_play | 0.000 | 0.188 | 0.011 |
| diff_off_sack_rate | 0.000 | 0.188 | 0.011 |
| diff_off_turnover_rate | 0.000 | 0.188 | 0.011 |
| diff_off_yards_per_play | 0.000 | 0.188 | 0.011 |
| diff_point_diff | 0.000 | 0.188 | 0.011 |
| diff_qb_expected_epa_per_dropback | 0.000 | 0.082 | 0.009 |
| diff_qb_start_probability | 0.000 | 0.062 | 0.004 |
| diff_qb_starter_cpoe | 0.000 | 0.082 | 0.009 |
| diff_qb_starter_epa_per_dropback | 0.000 | 0.082 | 0.009 |
| diff_qb_starter_experience_log | 0.000 | 0.062 | 0.004 |
| home_ats_residual | 0.000 | 0.188 | 0.011 |
| home_def_epa_per_play | 0.000 | 0.188 | 0.011 |
| home_def_pass_epa_per_play | 0.000 | 0.188 | 0.011 |
| home_def_rush_epa_per_play | 0.000 | 0.188 | 0.011 |
| home_def_sack_rate | 0.000 | 0.188 | 0.011 |
| home_def_takeaway_rate | 0.000 | 0.188 | 0.011 |
| home_def_yards_per_play | 0.000 | 0.188 | 0.011 |
| home_off_cpoe | 0.000 | 0.188 | 0.011 |
| home_off_epa_per_play | 0.000 | 0.188 | 0.011 |
| home_off_pass_epa_per_play | 0.000 | 0.188 | 0.011 |
| home_off_rush_epa_per_play | 0.000 | 0.188 | 0.011 |
| home_off_sack_rate | 0.000 | 0.188 | 0.011 |
| home_off_turnover_rate | 0.000 | 0.188 | 0.011 |
| home_off_yards_per_play | 0.000 | 0.188 | 0.011 |
| home_point_diff | 0.000 | 0.188 | 0.011 |
| home_team_games | 0.000 | 0.062 | 0.004 |
| temp | 0.250 | 0.646 | 0.319 |
| wind | 0.250 | 0.646 | 0.319 |

## 2. 2026 Week 1 extrapolation-risk list (vs 2025 regular season)

None. Every 2026 Week 1 column's missing/present state occurred in >= 1% of 2025 regular-season training rows.

## 3. Standardized ridge coefficient magnitude: indicators vs real features

Refit in memory (no artifact written): `ridge` target=`market_residual` ridge_alpha=10.0, 4431 training rows through 2026-01-04.

- real features: 90
- indicator features: 69
- median |coef| real: 0.25206
- median |coef| indicator: 0.00775
- max |coef| real: 2.84000
- max |coef| indicator: 1.16329
- ratio (median indicator / median real): 0.031

Top indicator columns by |standardized coefficient|:

| indicator on column | abs standardized coef |
|---|---|
| diff_active_roster_continuity | 1.16329 |
| diff_active_roster_mean_experience | 1.05911 |
| bias_week2_anchor_home | 0.13391 |
| bias_prior_week_ats_away | 0.06678 |
| bias_week2_anchor_away | 0.06172 |
| bias_prior_week_ats_diff | 0.05830 |
| diff_qb_starter_cpoe | 0.05664 |
| diff_qb_starter_epa_per_dropback | 0.05664 |
| diff_qb_expected_epa_per_dropback | 0.05664 |
| bias_prior_week_ats_home | 0.05453 |
| bias_week2_anchor_diff | 0.05105 |
| wind | 0.02077 |
| temp | 0.02077 |
| home_team_games | 0.01716 |
| diff_qb_start_probability | 0.01716 |
```

The rank-of-159 columns in section 3's table above (interleaving real and
indicator coefficients into one ranking) are not part of this raw output —
they were computed by a follow-up, separately verified query against the
same refit pipeline (see section 3's prose for exactly what was run).

## Files

- `scripts/missingness_audit.py` — the read-only audit script (Markdown by
  default, `--json` for the full per-column-per-season matrix and per-game
  risk table).
- `tests/test_missingness_guard.py` — two prediction-safety tests: (1) the
  current locked-but-unplayed week's missingness states are all >=1% common
  in the prior season's training rows (skips cleanly if the local feature
  table is absent); (2) no production column is missing in every 2009-2025
  training row. Both **pass** on real data as of this audit
  (**measured**: `./.tools/uv.exe run --no-sync pytest
  tests/test_missingness_guard.py -p no:cacheprovider --basetemp=<private>`
  — 2 passed).
- `scripts/missingness_source_availability_on_production.py` — the Stage 2
  frozen null/control/screen runner; it retains prediction-level rows.
- `tests/test_missingness_source_availability.py` — candidate identity,
  missing-indicator suppression, and outcome/line leakage regressions.
- `docs/missingness_audit.md` — this document.
