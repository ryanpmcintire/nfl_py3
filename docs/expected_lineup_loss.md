# Expected lineup loss from the play-probability model (LEAD-62)

Predeclared 2026-09-05, lane AS, BEFORE any candidate arm was scored. This
document is written and committed to before `nfl-ats rotation record` /
`nfl-ats weak-signals record` are run; results are appended below the
predeclaration, never edited into it.

## Binding closing-grounds taxonomy (verbatim, quoted per AGENTS.md/CLAUDE.md)

An interval or CI that contains zero is NEVER grounds to reject, fail, or
close an experiment. At this evaluator's ~2-point resolution, "contains
zero" is the EXPECTED outcome for a real small signal. Only two grounds ever
close a line of work: (1) refuted mechanism — a RESOLVED wrong sign (whole
interval on the wrong side of zero) or zero split-half reliability; (2)
bounded by a positive control proven able to detect an effect that size.
Everything else is `unresolved_below_power`: record it with
`nfl-ats weak-signals record`, report `probability_positive`, never the
binary "contains zero". The registry code hard-rejects inadmissible
closures; if a record command errors, the verdict is wrong, not the
validator. Verdicts flow only through `nfl-ats weak-signals record` /
`nfl-ats rotation record`, never through prose.

## Motivation

Lane AB shipped a walk-forward, isotonic-calibrated per-player P(plays)
model (`src/nfl_ats/play_probability.py`, `docs/play_probability_model.md`;
serving-realistic Brier improvement +0.124 over the previous no-designation
base rate). The PRODUCTION injury block
(`diff_injury_*` in `data/processed/game_features_weak_stack.parquet`, built
in `src/nfl_ats/players.py`) still uses only designation-based
unavailability (a player is either "out"/injury-weighted or not) and never
consults this model, the depth chart, or trailing snap share directly. This
lead asks whether a depth-chart-anchored, snap-share-weighted expected
lineup loss built from the new P(plays) model adds information PRODUCTION's
own injury features do not already carry.

## Frozen construct (predeclared, not to change after this doc is committed)

For each **team-game** (season, week, team):

```
expected_lineup_loss[group] = sum over that team's WEEK-OF depth-chart
    starters in `group` (depth_rank == 1 at each position slot) of
    (1 - P(plays)) * trailing-4-week snap share
```

split into three groups:

- **`qb`** — the team's depth-rank-1 quarterback only.
- **`offense`** — every other depth-rank-1 starter whose
  `position_group` (`nfl_ats.lineup_availability.depth_chart_position_group`)
  is `offensive_line` or `skill`, excluding the QB row counted above.
- **`defense`** — every depth-rank-1 starter whose `position_group` is
  `front` or `secondary`.

Special-teams-only starters (`position_group == "other"`: K/P/LS) are
excluded from all three groups — the construct is offense/defense/QB only,
per the lead's own predeclared split.

`P(plays)` is `nfl_ats.play_probability`'s own `play_probability` output
(`predict_play_probabilities`, the `played` label — see
"Strict pregame safety" below for exactly which model and which injury
snapshot). `trailing-4-week snap share` is that same module's own
`trailing4_snap_share` feature (mean own-side share over the player's own
last up to 4 games played, strictly before this week — already leak-safe by
construction, see the module's `attach_history_features` docstring). A
starter with no snap history at all (`trailing4_snap_share` is `NaN`, e.g. a
rookie or a player with zero recorded snaps before this week) contributes
**zero** to the sum, not `NaN` — documented choice: an unknown-history
player's *lost value* if he sits is treated conservatively as unmeasured
rather than poisoning the whole team-week sum.

**Signed production column** (one per group, three total):

```
diff_expected_lineup_loss_{offense,defense,qb} = home[group] - away[group]
```

Positive = the HOME team expects to lose more availability-weighted value
than the away team.

## Predeclared direction

This is a QUALITY feature (a continuous, signed column set), not a flag. The
predeclared hypothesis: the side that is expected to lose LESS should be
favoured, i.e. `diff_expected_lineup_loss_*` should carry a **negative**
coefficient when regressed against the home **margin residual** (higher
home-relative expected loss predicts a WORSE home margin). No claim is made
about magnitude before scoring.

## Decision arm

Because this is a quality-feature addition (not a discrete flag), the
comparison is: PRODUCTION `weak_stack` (ridge, `market_residual` target,
identical estimator/hyperparameters to every other on-production
confirmation) **plus the three `diff_expected_lineup_loss_*` columns**, vs.
PRODUCTION `weak_stack` alone. Graded at the Tuesday OPENER (binding:
"Grade the decision at the OPENER"), close-graded numbers are a screen only,
never the decision. `nfl-ats rotation declare --grade opener` assigns the
confirmation window; only that window's games are scored (single look).

## Strict pregame safety

`players.py`'s own `_injury_rows_asof` decides visibility for PRODUCTION's
`diff_injury_*` block: an injury revision is visible to a team-game's
features iff `effective_observed_at <= decision_at`, where
`decision_at = kickoff - decision_hours_before_kickoff` (production's own
built table, measured from `data/processed/game_features_weak_stack.manifest.json`:
`"decision_hours_before_kickoff": 24`, `"injury_timestamp_fallback": "week_proxy"`,
`"source_player_snapshot": "20260905T123614Z"`). This lead uses **the exact
same decision instant, the exact same player snapshot, and the exact same
`effective_observed_at` column** (already computed at ingest time on that
snapshot's own `injuries.parquet` — measured, 100% coverage, no
recomputation needed) to build its own asof-visible injury lookup, so a
player's `report_category`/`practice_category` inputs to the P(plays) model
are never drawn from a revision that would not yet have been visible to
PRODUCTION's own feature build for that same game.

`weeks_since_last_snap`/`trailing4_snap_share` are the play-probability
panel's own features, already leak-safe by construction (`merge_asof`,
`allow_exact_matches=False`, strictly earlier games only, any team) — no
sub-week timing precision is needed there since a player's OWN prior games
are, by definition, from strictly earlier weeks.

`roster_status` is forced to `"ACT"` at scoring time for every row (never
the true historical `INA`), matching `play_probability.serving_feature_frame`'s
own documented convention: the true weekly designation for the CURRENT week
is not normally known 24 hours before kickoff, so scoring with it would be
optimistic relative to a real deployment. This is the same "serving-realistic"
choice `docs/play_probability_model.md` already measures and prefers
(+0.124 Brier improvement) over the more flattering "full-information"
number (+0.181) for exactly this reason.

**Documented simplification (training only, not scoring):** the walk-forward
`HistGradientBoostingClassifier` itself is fit on `data/processed/play_probability_panel.parquet`
exactly as lane AB built it — i.e. each TRAINING row's injury/practice
category is the panel's own "last observed status for that player-week",
not an hour-precise asof cutoff. This is the same simplification
`docs/play_probability_model.md` already documents and accepts for the
model's own walk-forward protocol; it affects only which historical rows
the booster is fit on (strictly prior seasons in every case — no
cross-season leakage), never which information a SCORED game's own features
may see. A leakage regression test (`tests/test_expected_lineup_loss_features.py`)
verifies the scoring-time asof property directly: a late injury revision
recorded after `decision_at` must never change a scored game's
`diff_expected_lineup_loss_*` value.

## Population

2013-2025 (the play-probability panel's own coverage; snap_counts starts
2013). Because `fit_play_probability_model` requires at least one strictly
prior season, 2013 itself can never be SCORED (no model exists for it,
exactly as `docs/play_probability_model.md`'s own walk-forward table starts
at 2014) — the confirmable population is 2014-2025.

## Reliability plan

Split-half (odd/even weeks) reliability of the team-season mean expected
loss, computed BEFORE the decision arm is graded: for each (team, season),
compute the mean `expected_lineup_loss_offense + expected_lineup_loss_defense
+ expected_lineup_loss_qb` (the combined per-team-game total, before the
home-minus-away signing) separately over odd-numbered and even-numbered
weeks, then the Pearson correlation across all (team, season) cells between
the odd-week mean and the even-week mean. Per the binding taxonomy, a low
or zero-crossing reliability is not by itself a closing ground — only a
reliability of exactly (or statistically indistinguishable from) zero,
established as `no_split_half_reliability`, would refute the mechanism.

## Procedure (predeclared order)

1. This document, committed, before any arm is scored.
2. Build `src/nfl_ats/expected_lineup_loss_features.py` (the construct
   above) plus the `weak_stack_expected_lineup_loss` profile
   (append-only in `constants.py`/`margin.py`), `inherited_suffixes`
   (`tests/test_features.py`), and a `FAMILY_PHRASES` entry
   (`market_decomposition.py`).
3. `nfl-ats rotation declare --name expected_lineup_loss_on_production
   --description ... --grade opener`, then
   `nfl-ats rotation assign --name expected_lineup_loss_on_production`.
4. `scripts/expected_lineup_loss_on_production.py --mode null`, then
   `--mode positive-control`, then `--mode screen` (single look), against a
   scratch-built augmented table (`--features <scratch path>`; the
   augmented table is never written under `data/processed`).
5. Record via `nfl-ats rotation record` and
   `nfl-ats weak-signals record --category health`, in `accuracy_points`,
   with the reliability figure, per-season coverage, and per-season
   magnitude in notes. `probability_positive` is reported; "contains zero"
   is never used as a verdict.
6. Results appended below, and the ROADMAP.md LEAD-62 row updated with a
   dated note.

## Results (appended after scoring — nothing above this line was edited after predeclaration)

See the dated note below and `docs/expected_lineup_loss.md`'s own
`## Measured results, 2026-09-05` section for the full numbers, or
ROADMAP.md's LEAD-62 row for the summary the reader sees first.

## Implementation audit, 2026-09-05 (CX5)

**Measured** (SHA-256 of this file before appending this section): the original
predeclaration bytes hash to
`eb46c5cd3afff8a28fe462794eb6d1d95236437b8b68481da69f19d186206135`;
all text above this section is preserved. **Read** (owner's CX5 lane task):
the required decision instant is now `min(kickoff, Sunday 16:00 ET)` and the
corrected play-probability code takes precedence over the old timing and
training descriptions above; the three-group expected-loss formula stays frozen.

**Read** (`src/nfl_ats/expected_lineup_loss_features.py:94,168,217`): features
use the shared pool cutoff, latest visible injury revisions, strictly earlier
daily depth observations, and freshly computed probabilities fitted only on
strictly prior seasons; supplied probability columns are overwritten.
**Read** (`src/nfl_ats/play_probability.py:947`): booster training precedes
the previous-season calibration block, with an explicitly uncalibrated
fallback when that split is unavailable. **Read**
(`src/nfl_ats/play_probability.py:132`): the current probability model does
not consume `roster_status`; the feature wrapper retains the harmless ACT
compatibility column and no longer requires it in the panel.

**Measured** (`artifacts/experiments/expected_lineup_loss_cx5/build.json`):
the rebuilt input panel has 320,700 rows, seasons 2013-2024; its hash is
`54fadaf1820f96c811b613011cecba7cdda243a5b959ed128b4ac1984c4db8c6`.
The production feature-table hash is
`41a778f26a38e63bede7e7bf01f4a4a30254c09164cae3c5ee2cce87bc2547f6`,
and the injury input is its declared snapshot
`data/players/raw/20260905T123614Z/injuries.parquet`.
**Read** (`data/processed/play_probability_panel.parquet.provenance.json:9`):
the upstream rebuild excluded 34,745 daily depth rows with unverifiable
observation times. **Read** (`src/nfl_ats/expected_lineup_loss_features.py:168`):
legacy weekly depth charts still rely on the archive's week-labelled pregame
assumption; their exact observation times are unavailable, so the historical
screen is not proof that every legacy depth observation preceded the cutoff.

**Measured** (`build.json`, computed before either decision arm): odd/even
team-season mean-total-loss reliability is **0.8262005 on 352 team-seasons**.
The same artifact records game coverage: 256 each season 2014-2020, 272 in
2021, 271 in 2022, and 272 each in 2023-2024; 2025 has no covered games.
**Read** (`scripts/expected_lineup_loss_on_production.py:59`): the candidate
profile is registered within the experiment process only, as the exact 90
production columns plus the three frozen columns, and restored after use;
shared profile modules are outside the owner's allowed file scope.

**Measured** (`nfl-ats rotation declare` then `rotation assign`): the assigned
opener confirmation window is **2020-2021**, with the mined-window
acknowledgment recorded. **Measured** (initial `--mode null` artifact at
`artifacts/experiments/expected_lineup_loss_cx5/null/results.json`): the first
instrument run used a coverage-filtered training table. **Read**
(`scripts/expected_lineup_loss_on_production.py:158`): that mistake was
corrected before positive-control or screen execution; the final harness
keeps all production training rows, leaves absent early loss history missing
for the standard production imputer, and restricts only the paired evaluation
population to games with measured loss features. **Measured** (corrected
`--mode null` artifact under `production/null`): the null check was repeated
on that corrected comparison, with 200 permutations, mean **+0.1579 accuracy
points**, standard deviation **1.0889**, and central 95% range
**[-1.7599, +2.1930]**. The initial artifact is retained as an audit trail,
not used as the confirmation result.

**Read** (`src/nfl_ats/clv.py:830`): this existing opener harness substitutes
the opener spread while other production inputs retain their existing build
timing. This is opener-graded pool-deadline feature research, not a claim
that these injury/depth features were already available on Tuesday.

**Measured** (`production/positive-control/results.json`, same artifact root
as above): replacing only the candidate offense column with realized ATS
margin yielded **+44.2982 accuracy points**, week-blocked 95%
**[+38.6160, +50.0000]**, `probability_positive=1.0`, on 456 opener non-push
games across 35 weeks; control accuracy was 97.5877% versus production's
53.2895%. **Inferred:** this deliberately leaky instrument check establishes
sensitivity to a large injected signal; it does not bound the smaller real
candidate and cannot supply a candidate-sized positive-control closing ground.

## Measured results, 2026-09-05

**Inferred:** I think the forced-pick decision favors adding the three loss
features on expected value; this is a decision preference, not a stable-edge
claim. **Measured** (`production/screen/results.json`, independently checked
against `production/screen/paired_predictions.csv`): the opener probability
rule scores **246/456 (53.9474%)** versus production's **243/456 (53.2895%)**,
an improvement of **+0.657895 accuracy points**, week-blocked 95%
**[-1.569507, +3.118040]**, **`probability_positive=0.66505`**.
The comparison has 466 paired games including 10 opener pushes, 35 weeks,
and 28 pick disagreements; 20,000 resamples use seed 20260902.
**Measured** (same artifact): season-blocked 95% is
[-1.363636, +2.542373] accuracy points, `probability_positive=0.74865`.

**Measured** (`production/screen/results.json`, per-season opener probability
rule):

| Season | Non-push games | Candidate | Production | Change (accuracy points) | Week-blocked 95% (points) | probability_positive |
| --- | ---: | ---: | ---: | ---: | --- | ---: |
| 2020 | 220 | 50.4545% | 51.8182% | -1.363636 | [-4.090909, +0.934579] | 0.10900 |
| 2021 | 236 | 57.2034% | 54.6610% | +2.542373 | [-0.847458, +6.521739] | 0.89465 |

**Measured** (same artifact, secondary grades): opener sign-rule change is
+0.219298 accuracy points, week-blocked [-1.826589, +2.401747],
`probability_positive=0.53105` on 456 games; close probability-rule change
is -0.216920 points, [-2.542373, +2.212389],
`probability_positive=0.38175` on 461 games. **Read** (AGENTS.md,
"Grade the decision at the OPENER"): the close result cannot veto the
opener decision.

**Measured** (`nfl-ats weak-signals record`, then `nfl-ats rotation record`):
`expected_lineup_loss_on_production` is recorded in category `health`, units
`accuracy_points`, classification **`unresolved_below_power`**, and the
2020-2021 rotation look is spent with verdict **`unresolved`**; both records
include the probability, interval, reliability and per-season magnitudes.
**Inferred:** neither a resolved wrong sign, absent split-half reliability,
nor a candidate-sized positive-control bound has been established.

**Measured** (active-model digest comparison after scoring):
`artifacts/active_ats_model.json` still matches the screened production source
digest and the weak_stack / market_residual / ridge / alpha-10 configuration.
**Read** (owner's CX5 file scope and publish prohibition): production profile
wiring, live forecast regeneration and deployment remain the orchestrator's
lane; this task delivers the scored addition and its decision evidence.

**Measured** (CX5 final targeted validation): `pytest -n 2
tests/test_expected_lineup_loss_features.py tests/test_roadmap_inventory.py
tests/test_features.py --basetemp "$env:TEMP\nfl_pytest_cx5"
-p no:cacheprovider -q` passed **32 tests**. **Measured** (`ruff format` and
`ruff check` on the three CX5 Python files plus the inventory test): four
files were already formatted and lint passed. **Measured** (artifact audit):
baseline predictions are identical across the final null, positive-control
and screen modes; both curated registry records exist; the predeclaration
prefix retains its original hash. **Measured** (inventory test): the single
LEAD-62 row raises the pinned count from 264 to 265.
