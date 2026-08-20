# weak_stack_v3: every buildable positive-lean gap feature, scored at the opener

Written 2026-08-20. Task: find every recorded NFL weak signal not yet feeding
the production model, build the buildable ones into a candidate feature
profile, and score it against the active model at the opener grade (the
grade the Splash Sports pool actually settles on, per AGENTS.md). This
document is the full record: the inclusion rule, the gap list, what got
built vs deferred and why, the opener/close scores, and the EV read.

**Everything below is measured this session** (`registry/weak_signals.json`,
`artifacts/weak_stack_v3_opener_eval/20260820T110308Z/`,
`registry/experiment_specs/` are not used here since `experiment_runner.py`'s
`feature_arm` type does not yet support `population.grade == "opener"` --
this comparison used the same hand-adapted `opener_pick_evaluation` pattern
`scripts/surface_profile_opener_eval.py` established for MOD-08) unless
explicitly marked **read** (a fact taken from a file) or **reported**
(quoted from the registry, not independently re-verified this session).

## 1. The inclusion rule (predeclared before reading effect signs)

The candidate set is every `registry/weak_signals.json` entry that satisfies
**all** of:

1. `league == "nfl"` and `effect_units == "accuracy_points"`.
2. `probability_positive >= 0.60` (leans toward the candidate direction; not
   a promotion bar, an inclusion bar for "worth building").
3. `classification` is not `refuted_mechanism` or `bounded_by_control` (a
   closed negative must never be smuggled back in).
4. `reliability` is not exactly `0.0` (the one AGENTS.md-admissible
   "no split-half reliability" refutation; `None`/unmeasured and negative
   values pass this filter -- they are read as *weak* evidence, not
   disqualifying, per AGENTS.md's binding rule that a crossing-zero interval
   is never grounds for rejection).
5. The underlying construct is computable **point-in-time** (pregame-safe)
   from data **already local to this repo**: `game_features*.parquet`
   families, the SBR/nflverse odds archive, CDC FluView, the IEM forecast
   archive, Wikipedia-pageview attention data, injury news, special-teams
   stats, referee assignments, team-style stats, division-revenge/era/
   travel-rest schedule facts -- checked against `data/raw/` and
   `data/processed/` (**read**, `Bash` `ls`/`glob` this session), not
   assumed.
6. The construct is **not already inside**
   `FEATURE_SETS["football_weak_stack"]` (the production allowlist -- **read**,
   `src/nfl_ats/constants.py`) and is **not** a re-measurement, ablation, or
   subgroup slice of a feature block that is already inside it (an
   "already-priced" re-audit is not a gap).

This rule was written down and run as a mechanical filter (a Python script
over the loaded registry) **before** any candidate's effect sign was used to
decide inclusion -- rule 2 only gates the direction already implied by the
registry's own `probability_positive` field, which is itself the project's
declared EV-relevant statistic, not something chosen after seeing results.

Applying rules 1-4 mechanically against the 264-entry (**measured**, `nfl-ats
weak-signals status` after this session's write; 239 at session start)
registry returns **87 candidates**. Rules 5-6 (data availability,
already-priced exclusion) then require reading each one; the classification
is below.

## 2. The gap list

### 2a. Excluded by rule 6 (already inside `weak_stack`, or not a feature at all)

Not a gap, however positive: `learned_availability_ats_2018_2025` and the
whole `injury_value_lost_*`/`injury_value_lost_gradient`/
`injury_value_lost_narrowed` family (the production injury-value columns
*are* the candidate they describe); `player_family_base_vs_qb_continuity`,
`_value`, `_continuity`, `_injuries`, `_injury_value`, `_injuries_continuity`
(ablations of blocks already in `weak_stack`, not new columns);
`weak_stack_v2_penalty_only` and `surface_switch_feature_arm` (these
literally **are** the feature-arm tests for two of the constructs built
below -- counted under §3, not excluded); `mod08_smooth_cdf_mapping`
(a probability-calibration method, not a training feature);
`best_pick_opener_ranker_dispersion_filtered_candidate_vs_*` (a Best-Pick
*nomination* method, not a per-game feature); `sbr_opener_era_*`,
`sbr_opener_pooled_2011_2021`, `proxy_opener_production_rule_2009_2019`,
`pick_conditioned_spread_gap_zone_pre2018`, `pick_conditioned_rest_mismatch_pre2018`
(era-stratified/subgroup **re-measurements of the production model itself**,
not a candidate feature); every `era_trend_*` entry (a magnitude-drift
re-measurement of an already-covered construct, not a new one);
`odds_microstructure_*` and `observed_movement_*` (the movement-oracle /
post-Tuesday-information channel -- by construction not point-in-time-safe
for a training feature fit at the opener; AGENTS.md already treats this as a
*pick-refresh* lever, not a model feature, and it stays there); the
actual-weather versions in `weather_battery_*`/`weather_followup_*` (their
own registry `description` fields say **"NOT pregame-available"** in so many
words -- fails rule 5 directly; only the `forecast_weather_*` variants below
pass).

### 2b. Passes the rule -- built this session (§3)

| Registry entry | P+ | effect (pts) | reliability |
|---|---|---|---|
| `bias_battery_division_revenge_game` (+ `_opener` re-screen) | 0.8825 / 0.8642 | +0.19 / +0.29 | n/a (situational) |
| `bias_battery_sandwich_spot` | 0.603 | +0.044 | n/a (situational) |
| `bias_battery_post_blowout_win_letdown` | 0.7844 | +0.159 | n/a (situational) |
| `bias_battery_post_blowout_loss_bounce` | 0.6344 | +0.073 | n/a (situational) |
| `penalty_discipline` / `weak_stack_v2_penalty_only` | 0.6828 / 0.6939 | +0.33 / +0.13 | 0.261 |
| `surface_switch_feature_arm` | 0.6181 | +0.24 | n/a (structural) |
| `travel_rest_thursday_pure` | 0.7592 | +0.135 | n/a (situational) |
| `travel_rest_return_trip_hangover` | 0.7528 | +0.212 | n/a (situational) |

(**reported**, from `registry/weak_signals.json`, unverified beyond the
mechanical filter above -- these are the numbers that motivated building
each column, not re-derived here.)

### 2c. Passes the rule -- identified, NOT built this session (deferred, with reasons)

Every one of these genuinely meets rules 1-6 above. Deferral is a scope/time
call this session, not a finding against them -- they are **v4 candidates**,
not closed.

| Registry entry(ies) | P+ | reliability | Why deferred |
|---|---|---|---|
| `forecast_weather_warm_team_cold_late`, `forecast_weather_temp_gap_cold_visitor` | 0.9711, 0.9029 | n/a | Data exists (`data/raw/forecast_archive`, **read**, `ls` this session) and `scripts/nfl_forecast_weather_screen.py` has the construct, but wiring a Tuesday-noon-forecast-archive join into a reusable point-in-time feature builder (team's climatological-normal temp vs. this game's forecast) is new integration surface not completed this session. |
| `fluview_away_market_elevated`, `fluview_home_market_elevated`, `fluview_peak_home_elevated` | 0.8826, 0.8179, 0.6228 | 0.9814 (high -- the strongest deferred candidate) | Data exists (`data/raw/fluview`, **read**), but needs a team-to-state mapping plus an AS-OF weekly ILI merge; not built this session. Highest priority for v4 given the reliability. |
| `special_teams_return_top_quartile`, `_composite_edge_top_quartile`, `_composite_edge_bottom_quartile`, `_return_bottom_quartile`, `_punt_net_bottom_quartile` | 0.9547, 0.6425, 0.7941, 0.7207, 0.6726 | 0.065-0.313 (low) | `scripts/special_teams_features.py` exists, so this is buildable, but every cell's measured split-half reliability is low (0.065-0.313) -- not zero (so not rule-excluded) but weak enough that this session prioritized the higher-reliability/simpler-mechanism candidates instead. |
| `team_style_short_game_identity`, `team_style_pace_mismatch_dog_cover` | 0.8705, 0.7113 | 0.408, 0.489 | `scripts/team_style_features.py` exists; `pace_mismatch` is conditioned on `dog_cover` (a subset construct, not a plain pregame diff), which needs more design work to turn into a clean ridge input than the schedule-flag families below. Reliability is decent -- a real v4 candidate. |
| `referee_battery_home_penalty_tilt_top/bottom_quartile`, `_penalty_rate_top/bottom_quartile` | 0.72, 0.71, 0.70, 0.65 | **-0.101** (tilt), 0.370 (rate) | Data exists (`data/raw/officials`, **read**) and flag-builder code already exists (`nfl_ats.experiment_runner.FLAG_BUILDERS`), so this was the closest call. The penalty-*tilt* metric's measured reliability is **negative** -- not the AGENTS.md-admissible "measured zero" refutation, but weak enough (indistinguishable from a non-persistent trait) that it was deprioritized behind stronger candidates this session, not excluded. |
| `attention_battery_both_cold`, `attention_battery_away_hot`, `attention_followup_*` (3 cells) | 0.8568, 0.6999, 0.62-0.62 | 0.132 (low, shared across all cells) | Low reliability, and every registry entry in this family explicitly says "mined pilot battery, uncorrected multiplicity" in its own notes -- the project's own documented caution. Local data location for the Wikipedia-pageview source was not verified this session. |
| `surface_familiarity_r1_turf_venue_visitor_split`, `_r3_era_2018_2025`, `_r3_era_2009_2017` | 0.9332, 0.9577, 0.6482 | n/a | Same underlying mechanism as `surface_switch_flag` (grass-modal visitor on turf), a venue-controlled refinement of it. Not built as a second, near-duplicate/collinear column -- `surface_switch_flag` already carries the core mechanism into `weak_stack_v3`. |

## 3. What was built: `weak_stack_v3`

New module `src/nfl_ats/weak_stack_v3_features.py`. Three sub-families, 15
new columns, every one computed from the newest `data/raw/*/schedules.parquet`
snapshot (and, for penalty rate, the newest PBP snapshot) alone -- **never**
from `result`/`spread_line` at prediction time -- ported from
already-reviewed constructs rather than re-derived, so the registry's own
measured numbers above describe exactly these columns:

- **`gap_v3_bias`** (12 columns, `_home`/`_away`/`_diff` each):
  `gap_division_revenge`, `gap_sandwich_spot`,
  `gap_post_blowout_win_letdown`, `gap_post_blowout_loss_bounce`. Ported
  from `nfl_ats.experiment_runner.FLAG_BUILDERS`
  (`division_revenge_game`, `sandwich_spot`) and
  `scripts/nfl_bias_battery_screen.py`'s identically-named hypotheses
  (post-blowout).
- **`gap_v3_penalty`** (1 column): `diff_penalty_rate_prior`, ported
  verbatim from `scripts/weak_stack_v2_eval.py`'s
  `team_season_penalty_rate`/`add_penalty_discipline_feature` (already
  verified there to reproduce the registered `penalty_discipline` signal's
  mean/sd/reliability).
- **`gap_v3_travel`** (2 columns): `gap_thursday_pure_flag`,
  `gap_return_trip_hangover_flag`, ported from
  `scripts/nfl_travel_rest_battery_screen.py`'s cells 8 and 4, using the
  same `registry/stadium_coordinates.json` reference table and haversine
  formula.

`surface_switch_flag` (the fourth registry gap candidate) is **not**
recomputed: it is already a real, tested production column
(`nfl_ats.features.add_surface_switch_features`), and
`data/processed/game_features_weak_stack_surface.parquet` already carries
it. `weak_stack_v3` is built from that table, so it gets the column for
free.

**Wiring** (mirrors the `weak_stack_surface`/`weak_stack_js_prior`
precedent exactly, in `src/nfl_ats/constants.py` and `src/nfl_ats/margin.py`):
`GAP_V3_BIAS_FEATURE_COLUMNS`/`GAP_V3_PENALTY_FEATURE_COLUMNS`/
`GAP_V3_TRAVEL_FEATURE_COLUMNS` stay out of `MODEL_FEATURE_COLUMNS`
(the production allowlist, unchanged); `FEATURE_SETS["football_weak_stack_v3"]`/
`["full_weak_stack_v3"]` = `weak_stack_surface`'s own sets plus the three new
families; `MarginFeatureProfile`/`MARGIN_FEATURE_PROFILES`/
`_MARGIN_PROFILE_FEATURE_SETS` gain one new entry, `"weak_stack_v3"`. The
active model's `feature_profile` (`"weak_stack"`) and every other production
default are untouched.

**Table build**: `scripts/build_weak_stack_v3_table.py` is a pure additive
merge-by-`game_id` enrichment of
`data/processed/game_features_weak_stack_surface.parquet` (never
`game_features_weak_stack.parquet`, the production table) --
`data/processed/game_features_weak_stack_v3.parquet`, 4,902 rows x 290
columns (275 + 15). The script asserts every pre-existing column is
bit-identical before writing (**measured**, this session: assertion passed).

**Leakage tests**: `tests/test_weak_stack_v3_features.py`, 14 tests, one
leakage-regression test per new family per AGENTS.md ("a leakage regression
test for every new feature family"): worked examples for division revenge,
sandwich spot, and post-blowout letdown/bounce with hand-verified
True/False assertions; a test that mutating a game's own
`result`/`spread_line` never changes its own flags (only a *later* game
that looks back changes); a season-boundary leak test; a penalty-rate lag
worked example plus the `prev_season > season` structural self-check
(promoted from `scripts/weak_stack_v2_eval.py`'s inline assertion into a
real pytest); travel/rest worked examples (hangover fires/doesn't fire,
thursday flag, a known-city-pair haversine sanity check); and an orchestrator
additivity test. All 14 pass (**measured**, this session).

## 4. Score: `weak_stack_v3` vs the active `weak_stack`, opener grade

`scripts/weak_stack_v3_opener_eval.py`, adapted line-for-line from
`scripts/surface_profile_opener_eval.py` (the MOD-08 precedent, itself
adapted from `scripts/ridge_alpha_promotion_eval.py`'s `run_opener_grade` /
the MOD-07 52.83%-vs-52.50% precedent). Both arms hold `ridge_alpha=10.0`,
`regressor="ridge"`, `target="market_residual"` fixed at the incumbent's own
values -- only `feature_profile` differs, isolating the 15 gap columns'
combined effect. Paired Tuesday-opener archive, 1,537 games, 107 weeks,
2020-2025 (**measured**, this session, reproducing
`docs/opener_evaluation.md`'s population exactly). Uses
`clv.opener_pick_evaluation` directly, so both pick rules are reported; the
production **probability rule** (`home_cover_probability_at_open >= 0.5`) is
primary per `docs/opener_evaluation.md`'s 2026-08-19 addendum.

Sanity check: the baseline arm's opener accuracy under the probability rule
reproduced **53.36%**, exactly the tracked production headline
(`artifacts/active_ats_model.json`) -- trusted before reading the candidate.

### Primary: opener grade, production probability rule

| | baseline (`weak_stack`) | candidate (`weak_stack_v3`) |
|---|---|---|
| opener accuracy | 53.36% | 53.03% |
| close accuracy | 52.09% | 52.42% |

**Paired delta (candidate minus baseline), week-blocked (primary):
-0.333 accuracy points, 95% [-2.107, +1.467], `probability_positive` 0.3415.**
Season-blocked (6 blocks -- below `MIN_BLOCKS_FOR_INTERVAL=10`, degenerate,
reported for completeness only, not the governing interval): -0.333 points,
[-1.596, +0.915], P+ 0.2966.

Both intervals cross zero. Per AGENTS.md's binding rule this is
**`unresolved_below_power`**, not refuted -- the naive week-blocked interval
does not even sit entirely below zero, so no widening-factor check is needed
to rule out `wrong_sign_resolved`, and no positive control was run, so
`bounded_by_control` is unavailable either.

### Secondary readings (never veto the primary, per AGENTS.md)

- **Opener grade, historical sign rule** (`residual_at_open > 0`): -0.732
  points, week-blocked P+ 0.211 -- same direction, a somewhat stronger lean
  against.
- **Close grade** (both rules; reported per AGENTS.md, never gates the
  opener-graded decision): probability rule +0.332 points, P+ 0.619; sign
  rule +0.133 points, P+ 0.545. The direction **flips** at close -- exactly
  the situation the opener-grading rule exists to adjudicate. The primary
  verdict is the opener number above.
- **Calibration direction** (Brier/log-loss, opener-graded): both lean
  against the candidate -- Brier P+ 0.193 (week) / 0.246 (season), log-loss
  P+ 0.181 / 0.230. Consistent with the accuracy lean, not contradicting it.
- **Pick flips**: 189 of 1,537 games (12.3%) disagree under the production
  probability rule, 207 (13.5%) under the sign rule -- the 15 gap columns
  move a real share of picks, not a token amount.

Full artifact: `artifacts/weak_stack_v3_opener_eval/20260820T110308Z/`
(`opener_summary.json`, `opener_baseline.parquet`, `opener_candidate.parquet`,
`opener_paired.parquet`).

### Registry record

`weak_stack_v3_nfl_opener_confirmation`, `unresolved_below_power`, no
`closing_ground`, recorded via `nfl-ats weak-signals record` and verified
present with `nfl-ats weak-signals status` / a direct registry read
(**measured**, this session; registry now holds 264 signals, up from 239 at
session start). Full classification evidence and notes (including the
close-grade flip, the calibration direction, and a DO-NOT-POOL flag for its
shared baseline arm with every other opener-graded `weak_stack`-baseline
comparison) are in the recorded entry itself.

## 5. The EV read

**This is a pooled candidate, not a single mechanism.** Its combined-arm
result does not individually confirm or refute any one input family --
each input (division revenge, sandwich spot, post-blowout, penalty rate,
surface switch, thursday-pure, return-trip hangover) keeps its own existing
registry entry and classification untouched.

**At the opener -- the primary, pool-relevant grade -- `weak_stack_v3` leans
BELOW the incumbent `weak_stack`: `probability_positive` 0.34 for the
candidate**, i.e. roughly a 2-to-1 lean that the incumbent is actually
better here. This is not strong enough to call it refuted (the interval
crosses zero, nowhere close to the 1.099x-widening bar that would even let
`wrong_sign_resolved` be considered), and it is not strong enough to
promote. Per AGENTS.md's EV framing (P+ above 0.5 favours the candidate; a
promotion bar is not a decision bar), **the honest read is: do not play this
combined candidate over the incumbent today** -- P+ 0.34 is a real lean
against it, not a coin flip, even though it stays a legitimate open
`unresolved_below_power` entry rather than a closed negative. The close-grade
flip (P+ 0.619 in the candidate's favour) is recorded and is a genuine
disagreement between grades worth tracking, but per the binding "grade the
decision at the opener" rule it does not change the verdict.

This does not close any of the eight input mechanisms individually -- their
own registry entries, mostly at healthier P+ (0.60-0.88) on their own
narrower comparisons, stand as recorded. What it says is narrower: **stacked
together, at the current uniform `ridge_alpha=10`, these 15 columns do not
clear the incumbent at the grade that matters.** Two directions worth a
future look, neither executed this session: (a) a per-family ablation of
`weak_stack_v3` at the opener grade, to see whether one or two columns are
dragging the pooled result while the rest hold up individually; (b) the
higher-reliability deferred candidates in §2c (FluView in particular,
reliability 0.981) as a v4 stack, scored the same way.

## 6. Gates

`ruff format --check`, `ruff check`, and `mypy src` all pass clean on every
file this session touched or created. The full `pytest` suite: **1,465
passed, 1 failed** -- the one failure
(`test_experiment_registry.py::test_every_script_writing_artifacts_json_uses_the_provenance_helper`)
is caused by `scripts/roof_decision_screen.py`, an untracked file from a
**concurrent agent session** in this shared working tree (not authored by
this session, not touched by this work; **read**, `git status`/file mtime
this session). Before this fix that same command reported 4 additional
failures, all caused by this session's own changes (two `FEATURE_SETS`
exception-pinning tests needing `weak_stack_v3` added to their expected
admitting set, a `FAMILY_PHRASES` coverage test needing the three new
`gap_v3_*` phrases, and this script needing an allowlist entry matching its
`surface_profile_opener_eval.py` precedent) -- all four are fixed and
verified passing.

## 7. Files

- `src/nfl_ats/weak_stack_v3_features.py` -- the three builder sub-families
  plus the orchestrator.
- `src/nfl_ats/constants.py` -- `GAP_V3_BIAS_METRICS`,
  `GAP_V3_BIAS_FEATURE_COLUMNS`, `GAP_V3_PENALTY_FEATURE_COLUMNS`,
  `GAP_V3_TRAVEL_FEATURE_COLUMNS`, `FEATURE_FAMILIES["gap_v3_*"]`,
  `FEATURE_SETS["football_weak_stack_v3"]`/`["full_weak_stack_v3"]`.
- `src/nfl_ats/margin.py` -- `weak_stack_v3` added to
  `MarginFeatureProfile`/`MARGIN_FEATURE_PROFILES`/
  `_MARGIN_PROFILE_FEATURE_SETS`.
- `src/nfl_ats/market_decomposition.py` -- `FAMILY_PHRASES` entries for the
  three new families (plain-English attribution coverage).
- `scripts/build_weak_stack_v3_table.py` -- builds
  `data/processed/game_features_weak_stack_v3.parquet` (gitignored, local
  only).
- `scripts/weak_stack_v3_opener_eval.py` -- the opener-graded comparison
  script.
- `tests/test_weak_stack_v3_features.py` -- 14 leakage/registration tests.
- `tests/test_features.py`, `tests/test_experiment_registry.py` -- updated
  exception-pinning/allowlist assertions for the new profile and script.
- `registry/weak_signals.json` -- `weak_stack_v3_nfl_opener_confirmation`.
- `artifacts/weak_stack_v3_opener_eval/20260820T110308Z/` -- full result
  artifact (local, gitignored).

**Not touched**: `artifacts/active_ats_model.json`, `cli.py`,
`artifacts/prospective/challengers.json`, any production default profile,
or `data/processed/game_features_weak_stack.parquet` (the active model's own
table).
