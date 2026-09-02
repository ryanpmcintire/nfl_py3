# FluView elevated-illness indicators, replicated on COLLEGE FOOTBALL: predeclaration

Written **before any ATS outcome, cover rate, accuracy delta or sign is
computed on college-football data by this line of work**. Sections 1-8 are the
predeclaration. Section 9 was added after the look and reports what it found;
it changes nothing above it.

This is a **cross-league replication**, not a new NFL look. It spends **no NFL
evaluation window and no rotation window** — CFB is this project's sanctioned
free replication ground, exactly as `scripts/cfb_surface_familiarity_screen.py`
used it for the surface-familiarity lead (**read**,
`scripts/cfb_surface_familiarity_screen.py:16-17`: "No NFL evaluation window is
spent here -- CFB is this project's sanctioned free cross-league replication
ground"), and as `registry/weak_signals.json`'s
`cfb_surface_familiarity_turf_venue_visitor_split` entry records it (**read**:
`league: "cfb"`, no rotation entry, `classification:
unresolved_below_power`). `ROADMAP.md` names "CFB-replicated mechanisms" as one
of the three admissible paths forward (**read**, `ROADMAP.md:610-612`).

## Closing-grounds taxonomy (binding, restated verbatim per AGENTS.md)

An interval or CI that contains zero is **NEVER** grounds to reject, fail, or
close an experiment. At this evaluator's ~2-point resolution, "contains zero"
is the EXPECTED outcome for a real small signal. Only two grounds ever close a
line of work: (1) **refuted mechanism** — a RESOLVED wrong sign (whole interval
on the wrong side of zero) or zero split-half reliability; (2) **bounded by a
positive control** — the instrument was PROVEN able to detect an effect that
size and it was absent. Everything else is `unresolved_below_power`: record it
with `nfl-ats weak-signals record`, report `probability_positive`, never the
binary "contains zero". The registry code hard-rejects inadmissible closures;
if a record command errors, the verdict is wrong, not the validator. A
promotion threshold governs only what the docs may CLAIM; it never governs
which card is PLAYED, which is expected value.

## 1. What is being replicated, and what it is not

The construct under replication is the FluView state-level as-of
influenza-like-illness (ILI) **"elevated market"** indicator frozen in
`docs/fluview_battery.md` sections 2-3 (**read** in full; not re-derived here).
Its NFL reads, as they stand today:

| NFL read | grade | comparator | effect | 95% CI (week-blocked) | P+ |
|---|---|---|---|---|---|
| `fluview_home_market_elevated` (**read**, `docs/fluview_battery.md:382`) | close | bare market baseline, `home_cover` subset-vs-complement | +0.3090 pts | [-0.4092, +0.9491] | 0.8179 |
| `fluview_away_market_elevated` (**read**, `docs/fluview_battery.md:383`) | close | bare market baseline | +0.3681 pts | [-0.2566, +1.0005] | 0.8826 |
| `fluview_home_market_elevated_on_production` (**read**, `docs/fluview_on_production.md:374`) | close | PRODUCTION `weak_stack` chain, paired accuracy delta | +0.969 pts | [-1.150, +3.119] | 0.792 |
| `fluview_away_market_elevated_on_production` (**read**, `docs/fluview_on_production.md:373`) | close | PRODUCTION `weak_stack` chain, paired accuracy delta | 0.000 pts | [-1.156, +1.161] | 0.403 |
| underlying trait split-half reliability (**read**, `docs/fluview_battery.md:348-351`) | — | — | Pearson r 0.9636, Spearman-Brown 0.9814 | [0.9487, 0.9752] | 1.0000 |

The reliability figure is why this construct is worth a replication at all: at
Spearman-Brown 0.9814 the trait is not noise, so `no_split_half_reliability` is
unavailable as a closing ground and the only open question is magnitude.

An **opener-graded** NFL look on 2020-2021 read -0.439 pts, P+ 0.341 (**read**,
`docs/fluview_opener_look.md` section 7); a second NFL opener window is being
run in parallel by another agent and will append to that document. This
document neither reads nor touches that file, and **a CFB result never by
itself changes an NFL card** (section 8).

**What this document adds that the NFL reads cannot**: an independent sample,
in a different league, at zero NFL-window cost. It is replication evidence
about the mechanism's sign and magnitude. It is not, and cannot be, a
promotion or play/no-play decision for the NFL card.

## 2. Population

`data/processed/cfb_game_features.parquet` — the XLG-03 canonical benchmark
table built by `nfl-ats cfb-build-features` (**read**, `docs/cfb_data.md`
section "Derived benchmark table (XLG-03)"): completed regular-season
FBS-vs-FBS games carrying both an orientable spread and play-by-play, with the
NFL ATS sign convention (`ats_margin = result - spread_line`, `home_cover`
1/0/NaN-on-push).

Restricted to `nfl_ats.cfb_benchmark.CFB_CLEAN_CORE_SEASONS` (2012-2019 plus
2021-2025), **reused verbatim, never redeclared** — the same restriction
`scripts/cfb_surface_familiarity_screen.py` applies (**read**, its
`load_population`).

**Coverage-defined restriction, declared as a RULE before it is measured.**
FluView's point-in-time-recoverable floor is a property of the Delphi archive,
not of this project: state-level version history is effectively absent before
the 2017-18 season (**read**, `docs/fluview_battery.md` section 1, measured
2026-08-19). The scored population is therefore the clean-core seasons whose
**measured feature coverage is greater than zero**, where coverage is computed
on the PREDICTOR ONLY (the fraction of games with a non-missing as-of ILI
value) and before any outcome column is touched — the same admissible
pre-scoring exception `docs/team_style.md`'s reliability gate and
`docs/fluview_battery.md` section 4's peak-week window already use. The
measured season set is frozen in section 7 below, before any mode is run.

The **full clean core including the zero-coverage seasons** is reported as a
secondary read in section 9, so nothing is hidden: it is exactly the shape the
NFL family already disclosed for its own zero-information leg 2011 (**read**,
`docs/fluview_on_production.md:381-384`).

**Neutral-site handling, inherited unchanged from the NFL construction.**
`docs/fluview_battery.md` section 2 restricts the NFL battery to
`location == "Home"` because the home-market illness mechanism does not apply
at a neutral or displaced site. The CFB mirror is `neutral_site`: both feature
columns are **NaN (missing, not "not elevated")** for a neutral-site game.
Mirroring `nfl_ats.fluview_production_feature` exactly, the row is KEPT in the
scored population with a NaN feature (this is a feature builder feeding a model
with its own training-fold median imputation, not an evaluator), because that
is what the NFL on-production harness does — it scores every regular-season
game in its window and lets the NaN speak for itself.

## 3. The construct, replicated exactly, swapping only the league

Every element below is the frozen NFL construction with the league swapped and
nothing else changed. Where a function already exists it is **imported, not
reimplemented**.

| element | NFL (frozen) | CFB (this document) |
|---|---|---|
| ILI source | CDC Delphi FluView state ILI, full multi-issue history | identical archive, identical endpoint |
| as-of algorithm | per-state `release_date`-ordered running-max-epiweek checkpoint table, `merge_asof` backward | `build_checkpoint_tables` / `asof_lookup` **imported verbatim** from `scripts/fluview_battery_screen.py` |
| decision cutoff | the Tuesday on or before `gameday`, `tuesday_offset = (weekday - 1) % 7` | identical formula, identical code path |
| team → state | static 23-state `STATE_BY_TEAM` (`scripts/fluview_battery_ingest.py`) | per-season school → venue state from cfbfastR-data `team_info`, joined on CFBD/ESPN `team_id` (section 4) |
| "elevated" | as-of ILI ≥ that state's own 90th percentile, computed ONCE on the league's own state-week as-of panel, ≥10-observation floor | `compute_state_thresholds` **imported verbatim**, applied to the CFB state-week panel |
| missing handling | missing as-of value or missing threshold ⇒ excluded / NaN, never defaulted to "not elevated" | identical |
| home-market restriction | `location == "Home"` | `neutral_site` false |

**The threshold is computed on the CFB panel, not copied from the NFL panel,
and that is the faithful replication of the RULE.** The frozen rule
(`docs/fluview_battery.md` section 3) is "the top decile of **that state's own
history**", where the history is one value per state per league-season-week.
Copying the NFL's numeric thresholds would replicate a *number* measured on a
different panel of states, weeks and seasons, not the *construct*. The CFB
panel is: one row per (state, season, week) over the whole XLG-03 table's
non-neutral games (both home and away sides, all seasons the table carries),
computed once and frozen before scoring, exactly as the NFL panel was.

## 4. School → venue state: the one element with no NFL counterpart

The NFL construction uses a 34-code static dict. CFB needs a per-season
school → state map, and **no local CFB snapshot carries it** (**measured, this
session**): `data/processed/cfb_game_features.parquet` has 60 columns and no
venue, city or state field; the schedules snapshot has `venue_id` and a venue
*name* whose "(City, ST)" parenthetical resolves a state for only **4.2%** of
2012-2025 FBS non-neutral home games (6 distinct states); the ESPN
play-by-play, roster and betting snapshots carry no venue-location field;
`registry/stadium_coordinates.json` is NFL-only by its own README.

**Source used**: `team_info/parquet/cfb_team_info_<year>.parquet` from
**cfbfastR-data** — the same repository, same branch, same free no-key
`raw.githubusercontent.com` host that `src/nfl_ats/cfb.py` already declares as
a sanctioned CFB source for schedules and lines (**read**,
`src/nfl_ats/cfb.py:59-60`, `CFBFASTR_DATA_REPOSITORY`). Each file carries one
row per school per year with `team_id`, `school`, `venue_id`, `venue_name`,
`city`, **`state`**, `zip`, `latitude`, `longitude`. The commit SHA is pinned
via `nfl_ats.cfb.resolve_cfbfastr_commit()` (imported, not reimplemented) and
recorded in the snapshot manifest, matching the repository's own CFB
provenance convention.

**No CFBD API credit is spent.** The surface-familiarity replication reached
the same class of fact through an authenticated CFBD `/venues` call (**read**,
`scripts/cfb_surface_familiarity_screen.py:24-40`); this document deliberately
does not, because the free cfbfastR-data mirror carries the field.

**Join discipline: no name joins.** The benchmark table's `home_id`/`away_id`
are CFBD/ESPN team ids and `team_info.team_id` is the same id space.
**Measured, this session**: joining on `(season, team_id)` resolves a state for
**8,666 of 8,666** clean-core non-neutral scored games on both the home and the
away side (0 missing), all `country_code == "US"`, spanning **42 states**. The
same join attempted on school NAME leaves 39 home-side and 47 away-side rows
unresolved (Connecticut/UConn, UMass/Massachusetts, UT San Antonio/UTSA, …) —
which is why the id join is the declared path and the name join is not used at
all.

`team_info.state` is the state of the school's own listed **venue** (it sits
beside `venue_id`/`venue_name`/`city`/`zip`/`latitude`/`longitude`), so it is
the home-market state the mechanism is about, and it is per-season, so a venue
change is carried correctly.

## 5. FluView state coverage: what was already local, and what was fetched

The local snapshot `data/raw/fluview/20260820T003258Z/fluview_raw.parquet`
(809,716 rows) covers **23 states + `nat`** — the states hosting an NFL
franchise, by construction (**read**, `scripts/fluview_battery_ingest.py:51-93`).
CFB's clean-core home venues span **42** states (section 4), so **19 states were
genuinely missing**: `al ar ct de hi ia id ks ky ms ne nm ok or sc ut va wv wy`.

**They were fetched this session**, one bulk request per state, reusing
`scripts/fluview_battery_ingest.run_ingest` verbatim (same polite GET, same
429 backoff, same manifest with per-request status/bytes/rows and an
`output_sha256`), into a **new directory** `data/raw/fluview_cfb/<UTC>/` that
nothing else in the repository globs — so `data/raw/fluview/*`'s "latest
snapshot" resolution, and therefore every concurrent NFL FluView run, is
untouched. The 23 overlapping states are **not** re-fetched: the already-frozen
NFL snapshot is concatenated in as-is, so every shared state's checkpoint table
is bit-identical to the one the NFL battery froze.

**Measured, this session** (predictor-only, no outcome touched): all 19 new
states returned HTTP 200 with 33,827-34,037 rows each, 645,288 rows total, zero
retries, and **every one has a non-null `release_date` from 2017-10-24
onward** — the same version-history floor `docs/fluview_battery.md` section 1
measured for the NFL states, confirmed independently on 19 more states.

Two coverage gaps are inherited and disclosed, not corrected: `ny` returns a
null `release_date` on every row upstream (**read**,
`docs/fluview_battery.md:93-102`), so New York schools carry a missing value on
both sides; and `fl`'s earliest checkpoint is 2021-10-15 rather than 2017-10-24
(**read**, `docs/fluview_battery.md:362-372`), so Florida schools are covered
only from 2021.

## 6. The comparator: the XLG-03 benchmark arm, plus exactly one column

Mirroring `docs/fluview_on_production.md` section 4 in structure, with the CFB
benchmark standing where PRODUCTION `weak_stack` stands in the NFL version.

| arm | feature columns | estimator |
|---|---|---|
| baseline (shared) | `nfl_ats.cfb_features.CFB_MODEL_FEATURE_COLUMNS` (35 columns, the frozen XLG-03 contract) | `nfl_ats.cfb_benchmark.fit_cfb_residual_model`, `target="market_residual"`, ridge, `alpha=10.0` |
| PRIMARY: away-market | the same 35 **plus** `cfb_fluview_away_market_elevated` | identical |
| SECONDARY: home-market | the same 35 **plus** `cfb_fluview_home_market_elevated` | identical |

Both arms hold regressor, alpha and target fixed at the benchmark's own frozen
values; only the feature contract differs, isolating each column's marginal
contribution against everything the benchmark already explains. The extension
point is the benchmark's own: `fit_cfb_residual_model`'s `feature_columns`
parameter, whose docstring already declares it (**read**,
`src/nfl_ats/cfb_benchmark.py:100-103`): "a declared candidate family … may
extend it without touching the frozen benchmark path." The two candidate
columns are never mixed with each other — one column per cell, matching the
NFL family's own F1/F2 separation.

**Walk-forward.** Every scored week's models are trained on all completed games
in the table that kicked off **strictly before that week's own earliest
kickoff**, with the benchmark's own `CFB_BENCHMARK_MIN_TRAIN_GAMES = 500`
floor — the same forward-chaining `cfb_walk_forward_benchmark` performs
(**read**, `src/nfl_ats/cfb_benchmark.py:200-209`), and the same per-week refit
`scripts/fluview_elevated_on_production.py::run_leg` performs on the NFL side.
Training draws on the whole table (all seasons), not only the scored seasons.

**Primary/secondary assignment, mirrored rather than assumed.**
`docs/fluview_on_production.md` section 4 (**read**, lines 126-127 and 140-143)
declares **away-market the PRIMARY cell and home-market the SECONDARY cell**
for the NFL on-production family. This document mirrors that assignment
verbatim, per the instruction to mirror the NFL family's own assignment rather
than assume one. Both cells are recorded regardless of sign, both are reported
with equal prominence, and the decision rule (section 8) is expected value, so
the label changes nothing about what is recorded or concluded — it only fixes
which cell's numbers lead the write-up. Noted explicitly because the
NFL family's *stronger* on-production read was the secondary (home) cell, and
because the work package that commissioned this replication named the home
cell as the construct of interest.

**Grade, named.** The CFB benchmark grades on `spread_line`, which is "the
median across books of each book's home-oriented **close-proxy** spread"
(**read**, `docs/cfb_data.md`, XLG-03 section). The CFB line archive is
"opener plus an unidentified-time resolved quote (a close proxy)" and "no
source records quote observation times" (**read**, `docs/cfb_data.md`,
"Betting-line source regimes"), so **CFB can be graded at a close proxy and
never at a verified opener**. This replication is therefore **close-graded**,
which is the right match: the NFL read it replicates
(`fluview_*_on_production`) is itself close-graded. Per the binding "grade the
decision at the opener" rule, a close-graded number settles no NFL play/no-play
or promotion decision — and a CFB number could not do so in any case.

## 7. Metric, uncertainty, instrument checks, leakage

**Metric.** The paired **candidate-minus-baseline forced-pick accuracy delta**,
in `accuracy_points` (percentage points), picks taken at
`home_cover_probability >= 0.5` and graded with `nfl_ats.clv.pick_correct`
(pushes NaN, excluded) — identical to
`scripts/fluview_elevated_on_production.py::_paired_metric`, reused by import
where possible and otherwise by verbatim mirror.

**Uncertainty.** `nfl_ats.clv.week_blocked_bootstrap`, **week-blocked
primary** (within-week correlation is ZERO by owner mandate — no ICC term),
**season-blocked secondary**, never averaged together. 1,000 samples (the same
`BOOTSTRAP_SAMPLES` the NFL sibling harness uses, for comparability), seed
`20260901` (repo convention: today's date).

**Within-week permutation null**, **200 draws**: both arms' models are fit ONCE
on the REAL `ats_margin`; only the grading margin is shuffled within week, so
200 draws cost no extra fits. This null is **not** centred on zero by design
(it preserves each week's realised home-cover rate and the two arms carry
different home-pick rates) and is reported ALONGSIDE the bootstrap-vs-zero
interval, never instead of it. Identical mechanism to
`docs/fluview_on_production.md` section 6.

**Positive control**, run BEFORE the real screen, per cell: the candidate's one
new column is REPLACED by the realised `ats_margin` — a deliberate, large
leak — so the harness must show an obvious, large effect. This proves the
FULL 36-column ridge fit (not a single-feature model) can detect a real effect
of meaningful size when one is present.

**Leakage.** Two guarantees, both regression-tested in
`tests/test_fluview_cfb_feature.py`:

1. **Publication lag.** A game in week *w* may only see ILI whose Delphi
   `release_date` is `<= ` that game's decision-cutoff Tuesday. The as-of
   algorithm enforces this by construction (`merge_asof(..., direction=
   "backward")` on `release_date`) and returns **missing**, never a leaked
   final value, when no qualifying checkpoint exists. The test injects a
   revision released after the cutoff and asserts it is invisible; it also
   asserts a late re-issue of an OLD epiweek never overwrites a newer
   already-known epiweek.
2. **Cutoff is never after kickoff.** For every game in the CFB population the
   computed cutoff Tuesday is `<= gameday`. This matters more in CFB than in
   the NFL because CFB plays Tuesday and Wednesday games in November: for a
   Tuesday kickoff the formula yields the cutoff = the gameday itself, which is
   still pregame, and the test asserts `cutoff <= gameday` on the real
   population, not only on a fixture.

A third test covers **join correctness**: the `(season, team_id)` school→state
join resolves every row of the real population, a two-school state shares one
panel row per week, and a neutral-site game comes back NaN on both columns
rather than 0.

**Frozen scored season set** (**measured** 2026-09-01 by
`.\.tools\uv.exe run --no-sync python scripts\fluview_cfb_replication.py
--mode coverage`, PREDICTOR ONLY — no outcome column is read by that mode —
before any other mode was run; artifact
`artifacts/fluview_cfb_replication/coverage.json`). Clean-core seasons with
non-zero home-side feature coverage: **2017, 2018, 2019, 2021, 2022, 2023,
2024, 2025**. Measured home-side coverage by season: 2006-2016 all **0.0%**;
2017 **36.9%** (partial — the first checkpoint is 2017-10-24, mid-season);
2018 89.9%, 2019 90.0%, [2020 90.7%, outside the clean core], 2021 92.2%,
2022 95.9%, 2023 95.3%, 2024 92.9%, 2025 95.0%. So 2012-2016 are excluded
from the primary population by the rule in section 2, and 2017 enters as a
genuinely partial season rather than a full one.

Also measured by that same predictor-only mode: **42 states** present across
both sides, **0** rows with an unresolved state, **327** neutral-site games in
the 12,500-row table, and the **CFB panel's own split-half reliability** —
`n_state_seasons=351`, Pearson r **0.9716**, Spearman-Brown corrected
**0.9856**, `probability_positive` 1.0000. That is an independent replication
of the NFL trait's own reliability (0.9636 / 0.9814, **read**,
`docs/fluview_battery.md:348-351`) on a different league and 19 additional
states, and it is the figure recorded in `--reliability` for both CFB cells.
`no_split_half_reliability` is therefore unavailable as a closing ground here,
exactly as it is on the NFL side.

**Era split, declared before scoring.** The window spans the benchmark's own
declared 2020 regime gap, which is the obvious boundary. Two eras, magnitudes
reported separately and **never averaged across a sign flip** (owner rule "era
magnitude, not presence"): **era A = 2017-2019**, **era B = 2021-2025**.

## 8. Decision rule and recording, frozen before scoring

**Decision rule.** Expected value, never a threshold: `probability_positive`
above 0.5 favours the candidate over the baseline. Predeclared thresholds
govern only what a document may CLAIM. **A CFB result is replication evidence
about a mechanism; it never by itself changes an NFL card**, and this run is
close-graded besides, so it settles no play/no-play or promotion decision in
either league.

**Recording.** Two `nfl-ats weak-signals record` entries, one per cell:

| cell | registry name | role |
|---|---|---|
| away-market | `cfb_fluview_away_market_elevated_on_benchmark` | PRIMARY |
| home-market | `cfb_fluview_home_market_elevated_on_benchmark` | SECONDARY |

`--league cfb`, `--effect-units accuracy_points`, `--family
fluview_cfb_replication`, `--category health`, week-blocked interval and
`probability_positive`, `--reliability` carrying the CFB panel's own measured
split-half reliability (not the NFL figure).

**A separate pooling bucket, stated explicitly.** `fluview_cfb_replication` is
a different family from `fluview_battery` and from
`fluview_elevated_on_production`. AGENTS.md's commensurability rule forbids
pooling non-commensurable comparators, and these three measure the same
construct against three different comparators in two different leagues
(bare market baseline / NFL production chain / CFB benchmark arm). They are
never pooled together.

**Classification.** `unresolved_below_power` for both cells unless a cell
literally meets a terminal ground: `wrong_sign_resolved` requires the WHOLE
week-blocked interval on the wrong side of zero, and `positive_control_bound`
requires the control to have PROVEN detection of an effect of the size in
question. An interval containing zero is not a ground; if a record command
errors, the verdict is wrong, not the validator.

**No rotation window is spent.** `nfl-ats rotation` is not invoked by this
document, matching the surface-familiarity CFB replication's own precedent.

## 9. Results (added after the look, 2026-09-01)

Every number in this section is **measured** by
`.\.tools\uv.exe run --no-sync python scripts\fluview_cfb_replication.py`
this session; each table names its own artifact. Nothing above this line was
edited after the first outcome sign was computed, except section 7's frozen
season set, which was filled in from the PREDICTOR-ONLY `--mode coverage` run
before any of the three scoring modes was launched.

**The scored population, as frozen.** 5,671 games across 122 week blocks and 8
seasons (2017-2019, 2021-2025). The XLG-03 baseline arm's own forced-pick
accuracy on that population is **51.402%**, and its home-pick rate is 43.70%.
Feature coverage inside the scored population: 575 games flagged
away-market-elevated (801 feature-missing) and 579 flagged
home-market-elevated (796 feature-missing).

**Instrument check 1 — within-week permutation null** (200 draws,
`--mode null`; artifacts `20260901T185347Z` away, `20260901T185447Z` home):

| cell | null mean | null sd | null 95% |
|---|---|---|---|
| away | +0.152 pts | 0.487 | [-0.742, +1.200] |
| home | +0.116 pts | 0.420 | [-0.688, +0.848] |

Both finite, both sane, neither centred exactly on zero — the predeclared
shape, since the null preserves each week's realised home-cover rate.

**Instrument check 2 — positive control** (`--mode positive-control`;
artifacts `20260901T185526Z` away, `20260901T185503Z` home). Identical for
both cells by construction (the same leaked column): pooled **+48.598
accuracy points**, week-blocked P+ **1.000**, 95% [+47.302, +49.751], n=5,671.
Per era: 2017-2019 **+48.778** (P+ 1.000), 2021-2025 **+48.493** (P+ 1.000).
The full 36-column ridge fit is not blind to a real effect of meaningful size
in either era.

**The real screen** (`--mode screen`, primary population; artifacts
`20260901T185607Z` away, `20260901T185542Z` home):

| cell (role) | pooled delta | week 95% CI | week P+ | season 95% CI | season P+ | null percentile |
|---|---|---|---|---|---|---|
| away (PRIMARY) | **-0.423** pts | [-1.553, +0.694] | **0.213** | [-1.275, +0.457] | 0.173 | 10.5th |
| home (SECONDARY) | **-0.388** pts | [-1.272, +0.460] | **0.200** | [-1.100, +0.279] | 0.125 | 10.5th |

**Per-era magnitudes, never averaged across a sign flip** (owner rule "era
magnitude, not presence"):

| cell | era 2017-2019 (n=2,087) | era 2021-2025 (n=3,584) |
|---|---|---|
| away | **+0.527** pts, P+ 0.707, 95% [-1.245, +2.564], 72 flagged | **-0.977** pts, P+ 0.063, 95% [-2.284, +0.308], 503 flagged |
| home | **-0.767** pts, P+ 0.205, 95% [-2.624, +1.069], 87 flagged | **-0.167** pts, P+ 0.305, 95% [-1.010, +0.666], 492 flagged |

**The away cell's two eras carry OPPOSITE SIGNS**, so its pooled -0.423 is an
average across a sign flip and must not be read as one era's magnitude. The
home cell keeps the same sign in both eras, with the earlier, thinner era
about 4.6x the magnitude of the later one. Note how thin the 2017-2019 flagged
counts are (72 and 87 games) — that era is where the coverage floor bites, and
its wide intervals say so honestly.

Per-season deltas (accuracy points, home cell / away cell): 2017 -0.73/-0.15,
2018 +0.28/+1.71, 2019 -1.87/0.00, 2021 -0.29/-1.87, 2022 +0.72/+0.29,
2023 -0.27/-1.23, 2024 -1.80/-2.50, 2025 +0.80/+0.40.

**Secondary read, the full clean core including the zero-coverage 2012-2016
seasons** (artifacts `20260901T185656Z` away, `20260901T185636Z` home), n=8,933
games / 199 weeks, with 4,112 and 4,107 feature-missing rows respectively:
away **-0.269** pts, 95% [-0.956, +0.475], P+ 0.247; home **-0.246** pts,
95% [-0.835, +0.340], P+ 0.195. As predeclared, adding an era in which the
column is 100% missing pulls both estimates toward zero without changing
either sign or either conclusion.

### What this implies for the decision, before what is wrong with it

**On EV grounds — `probability_positive` above 0.5 favours the candidate, the
only decision rule this project uses — the CFB replication favours the
BASELINE over both candidates**: P+ 0.200 (home) and 0.213 (away) week-blocked,
0.125 and 0.173 season-blocked. On CFB, adding either FluView column to the
frozen XLG-03 benchmark arm is roughly a 4-to-1 bet against.

**Stated plainly, since that is what this replication was commissioned to
settle: the NFL close-graded direction did NOT replicate.** The NFL
on-production home-market cell — the strongest reading this construct has, and
the one the work package named — was **+0.969 points, P+ 0.792** (**read**,
`docs/fluview_on_production.md:374`). The same construct, built the same way,
stacked the same way on the analogous frozen benchmark, measured at the
analogous close grade, on an independent league with **5,671 games against the
NFL read's 516**, comes back **-0.388 points, P+ 0.200**. The direction
reversed, and it reversed in **both** CFB eras for that cell.

**And it agrees with both NFL reads that were not close-graded.** The NFL
opener-graded windows on this same construct read **-0.439 points, P+ 0.341**
(2020-2021) and **-1.751 points, P+ 0.094** (2022-2023) — **read**,
`docs/fluview_opener_look.md:570-571`, the second appended by a parallel agent
this same session and read here after it landed; this document never edited
that file. So the ledger for `fluview_home_market_elevated` now stands at **one
positive close-graded NFL reading against three independent negative-leaning
reads**: two non-overlapping NFL opener windows and a CFB sample eleven times
the size of either.

**The decision implication for the NFL card**: spending further NFL window on
the FluView home-market cell now has materially worse expected value than it
did before this run, and any write-up still resting on the +0.969 close-graded
reading has to be read alongside all three. The four reads are not
commensurable and are never pooled (three different comparators, two leagues,
two grades — section 8), but they are four independent looks at one mechanism
and three of them lean the same way.

**What is NOT concluded, and why.** Neither cell is closed. `wrong_sign_resolved`
requires the WHOLE week-blocked interval on the wrong side of zero; away is
[-1.553, +0.694] and home is [-1.272, +0.460], both crossing it.
`bounded_by_control` requires the instrument to have been PROVEN able to detect
an effect of the size in question; the control proved detection of a +48.6-point
leak, which says nothing about a sub-1-point effect, and both week-blocked CIs
span more than 1.7 points — wider than the +0.969 NFL reading they are testing.
Both cells are recorded `unresolved_below_power`, and the trait's own
reliability (0.9856, section 7) rules out `no_split_half_reliability`
independently. **An interval containing zero is not grounds for rejection, and
none of the above rests on one.**

**Caveats, after the numbers rather than instead of them.** (1) CFB is a
different, softer market: the XLG-03 baseline picks at 51.402% here, so a null
on CFB constrains but does not strictly bound what the same construct can do
against the NFL market. (2) The 2017-2019 era carries only 72-87 flagged games
because the FluView version-history floor is 2017-10-24; its magnitudes are the
weakest-supported numbers in this document, which is exactly why they are
reported per era rather than folded into one figure. (3) `ny` (zero resolvable
coverage upstream) and `fl` (covered only from 2021-10-15) narrow coverage for
the schools in those states, disclosed in `docs/fluview_battery.md` section 1
and unchanged here. (4) This is close-graded and CFB besides, so it settles no
NFL play/no-play or promotion decision by itself; it is evidence about the
mechanism, weighed with everything else.

### Registry, verified by reading it back

`registry/weak_signals.json`: **619 -> 621** signals (measured before and after
each write; the registry was concurrently being written by other agents this
session, which is why every write went through the cross-process lock wrapper).
Both entries read back with `classification: unresolved_below_power`,
`closing_ground: null`, `family: fluview_cfb_replication`, `league: cfb`,
`reliability: 0.9856`:

- `cfb_fluview_away_market_elevated_on_benchmark` — effect -0.4232, interval
  [-1.5529, +0.6943], P+ 0.213.
- `cfb_fluview_home_market_elevated_on_benchmark` — effect -0.3879, interval
  [-1.2723, +0.4605], P+ 0.200.

`registry/rotation_registry.json` is **untouched**: no NFL rotation window was
declared, assigned or spent by this document.

### Files added

- `docs/fluview_cfb_replication.md` (this document).
- `src/nfl_ats/fluview_cfb_feature.py` — the two CFB candidate columns,
  reusing the frozen as-of/threshold engine by import.
- `scripts/fluview_cfb_replication.py` — `--mode fetch-inputs | coverage |
  null | positive-control | screen`.
- `tests/test_fluview_cfb_feature.py` — 13 leakage and join-correctness tests.
- `artifacts/fluview_cfb_replication/` — `coverage.json` plus seven run
  artifacts.
- `data/raw/fluview_cfb/20260901T184534Z/` (gitignored) — the 19 CFB-only
  states' FluView history, and `data/cfb/team_info/raw/20260901T185247Z/`
  (gitignored) — the pinned cfbfastR-data school -> venue-state snapshot.
