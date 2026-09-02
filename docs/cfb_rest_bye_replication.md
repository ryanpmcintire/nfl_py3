# NFL rest / bye mechanisms, replicated on COLLEGE FOOTBALL: predeclaration

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
ground"), and exactly as `docs/fluview_cfb_replication.md` used it for the
FluView construct earlier the same day.

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

Four NFL rest-and-bye constructs, frozen in `docs/travel_rest_battery.md`
(cells 5, 6, 7) and `docs/bye_overvaluation_screen.md` (cell 1), transcribed
into college football with the league swapped and nothing else changed. Their
NFL reads, **read** this session out of `registry/weak_signals.json`:

| NFL entry | effect (accuracy pts) | 95% CI (week-blocked) | P+ | n games | seasons |
|---|---|---|---|---|---|
| `travel_rest_home_off_bye` | -0.1523 | [-0.5137, +0.2085] | 0.2079 | 4,317 | 2009-2025 |
| `travel_rest_away_off_bye` | -0.0622 | [-0.4684, +0.3461] | 0.3839 | 4,317 | 2009-2025 |
| `bye_overval_home_edge_post2011` | -0.3304 | [-0.7563, +0.0965] | 0.0637 | 3,573 | 2012-2025 |
| `travel_rest_short_week_road` | +0.0442 | [-0.3116, +0.4041] | 0.5933 | 4,317 | 2009-2025 |

`bye_overval_home_edge_post2011` is the strongest directional read in the
family and it points at the market **over**pricing the bye: P+ 0.0637 against a
predicted-negative home-cover edge means the data leans the way the
overvaluation mechanism predicts. All four NFL entries carry
`reliability: null` (**read**, same file), so none of them has a measured
split-half reliability to inherit; section 7 measures CFB's own rather than
importing anything.

### The one thing that must be said before anything else

`nfl_ats.cfb_features.CFB_MODEL_FEATURE_COLUMNS` — the frozen 35-column XLG-03
benchmark contract — **already contains `rest_diff`** (**read**,
`src/nfl_ats/cfb_features.py:82`, inside `CFB_CONTEXT_FEATURES`; **measured**
this session: `len(CFB_MODEL_FEATURE_COLUMNS) == 35` and `"rest_diff"` is a
member). The baseline arm therefore already prices the **linear rest
difference** for every game it scores.

Every cell below is consequently a **MARGINAL on top of a model that already
knows the rest differential**, not a test of whether rest matters at all. That
is the honest comparison and it is the project's own discipline: "an overlay
positive alone can be negative stacked; evaluate marginals on top of what's
PLAYED" (memory `composition-is-not-the-signal`). What each cell asks is
narrower and sharper than the NFL siblings asked: *does a side-specific,
threshold-shaped rest fact carry anything the linear differential has already
missed?*

This is also why the CFB reads are not directly comparable in magnitude to the
NFL entries in the table above: those are subset-cover-rate-vs-complement
measurements against a bare market baseline, and these are paired accuracy
deltas against a 35-feature model that already carries rest. Section 6 states
that comparator difference formally; section 8 keeps the two in separate
pooling families because of it.

**What this document adds that the NFL reads cannot**: an independent sample,
in a different league, at zero NFL-window cost, against a stronger baseline.
It is replication evidence about each mechanism's sign and magnitude. It is
not, and cannot be, a promotion or play/no-play decision for the NFL card.

## 2. Population

`data/processed/cfb_game_features.parquet` — the XLG-03 canonical benchmark
table built by `nfl-ats cfb-build-features` (**read**, `docs/cfb_data.md`,
section "Derived benchmark table (XLG-03)"): completed regular-season
FBS-vs-FBS games carrying both an orientable spread and play-by-play, with the
NFL ATS sign convention (`ats_margin = result - spread_line`, `home_cover`
1/0/NaN-on-push). **Measured** this session: 12,500 rows x 60 columns, seasons
2006-2025, `kickoff` tz-aware UTC with 0 nulls.

Restricted to `nfl_ats.cfb_benchmark.CFB_CLEAN_CORE_SEASONS` — 2012-2019 plus
2021-2025 (**read**, `src/nfl_ats/cfb_benchmark.py:46`) — **reused verbatim,
never redeclared**, the same restriction the surface-familiarity and FluView
CFB replications apply.

**Coverage-defined restriction, declared as a RULE before it is measured.** The
scored population is the clean-core seasons whose **measured coverage for that
cell's own column is greater than zero**, where coverage is computed on the
PREDICTOR ONLY (the fraction of games whose candidate column is non-missing)
and before any outcome column is touched — the same admissible pre-scoring
exception `docs/fluview_cfb_replication.md` section 2 uses. The measured season
set is frozen in section 5 below, before any scoring mode is run. Unlike
FluView, this family has no archive floor to trip over, so the expectation
(stated before measuring) is that the rule excludes nothing; it is declared
anyway so the rule is the same rule either way.

**Neutral-site games are KEPT and flagged normally.** Rest is rest wherever the
game is played, and neither NFL sibling excluded neutral sites — the FluView
replication's `neutral_site` NaN rule exists because a *home-market illness*
mechanism cannot apply at a neutral venue, and no part of that argument
transfers to a calendar fact. Disclosed rather than assumed: **measured**, the
table carries 327 neutral-site games.

**Pushes** are handled by `nfl_ats.clv.pick_correct` (settle margin exactly 0
returns NaN and is excluded from accuracy), never scored as a loss.

## 3. The four cells: the NFL definition transcribed exactly, then the CFB adaptation

Sign convention for every cell, identical to the NFL entries: the effect is the
**candidate-minus-baseline paired accuracy delta**, so **positive means the
extra column helped the model**. The "predicted direction" column carries the
NFL family's own frozen prediction about the underlying `home_cover` edge, kept
so section 9 can say per cell whether CFB matched the NFL DIRECTION.

### Cell 1 — `home_off_bye`

**NFL definition, transcribed verbatim** from `registry/weak_signals.json`
(**read**, entry `travel_rest_home_off_bye`, `description` field):

> home_rest >= 13 days (side-specific absolute threshold, captures both true
> byes and MNF-to-Sunday extra-rest turnarounds) -- predicted positive
> home_cover edge (pregame-safe schedule/geometry fact, no leakage caveat).

**CFB adaptation**: the home team's own rest is `>= 13` days. Column
`cfb_rest_home_off_bye`.

**Is 13 the right CFB threshold? Justified before scoring, on predictor-only
evidence.** The NFL's own construction rule is stated in
`docs/travel_rest_battery.md:214-224` (**read**): set the threshold **one day
below the modal bye** so the cell captures the true 14-day bye cluster *and*
the adjacent weekday-shifted extra-rest cluster, rather than fragmenting into
two thinner cells. Applying that RULE — not the number — to the CFB calendar:
college football's modal open-date turnaround is also 14 days (Saturday to
Saturday), and **measured** this session on the whole 12,500-row table the
home-side rest histogram reads 11 days: 99, 12 days: 112, **13 days: 232, 14
days: 930**, 15 days: 194. The 14-day cluster dominates exactly as in the NFL,
and 13 is its adjacent shoulder. **13 therefore survives the CFB calendar
unchanged and is kept as the primary**; no shift is made, so no "NFL threshold
as a sensitivity arm" fallback is needed.

One CFB-specific case the NFL calendar cannot produce is nevertheless declared
as a **sensitivity arm** rather than left unexamined: CFB plays Thursday and
Friday games, so a Saturday → open date → Thursday turnaround is **12** days
and a `>= 13` threshold misses it. The arm `home_off_bye_gap12`
(`cfb_rest_home_off_bye_gap12`, `home_rest >= 12`) is frozen here, alongside
the primary and never in place of it. 12 is not an invented number: it is the
project's own strict-bye gap (`scripts/venue_milestone_screen.py`'s
`POST_BYE_GAP_DAYS = 12`, **read** via `docs/bye_overvaluation_screen.md`,
"Strict bye definition"), and it is the gap cell 3 already uses.

### Cell 2 — `away_off_bye`

**NFL definition, transcribed verbatim** (**read**, entry
`travel_rest_away_off_bye`, `description` field):

> away_rest >= 13 days (side-specific absolute threshold, mirror of cell 5 on
> the away side) -- predicted negative home_cover edge (pregame-safe
> schedule/geometry fact, no leakage caveat).

**CFB adaptation**: the away team's own rest is `>= 13` days. Column
`cfb_rest_away_off_bye`. Threshold justification identical to cell 1
(**measured** away-side histogram: 11 days: 88, 12 days: 115, **13 days: 244,
14 days: 832**, 15 days: 188). Sensitivity arm `away_off_bye_gap12`
(`away_rest >= 12`), frozen here.

### Cell 3 — `bye_edge_home`

**NFL definition, transcribed verbatim** (**read**, entry
`bye_overval_home_edge_post2011`, `description` field):

> HOME team off strict bye (>=12-day gap) AND opponent NOT off bye, seasons
> 2012-2025; market-overprices-bye mechanism predicted negative home cover.

**CFB adaptation**: `home_rest >= 12` **and** `away_rest < 12`. Column
`cfb_rest_bye_edge_home`. The 12-day gap is transcribed unchanged — it is the
NFL family's own strict-bye definition and no CFB calendar argument moves it.
The NFL cell restricted itself to seasons 2012-2025 as its post-CBA era; the
CFB clean core is 2012-2019 + 2021-2025 for an unrelated reason (the
benchmark's own 2020 regime gap), so the two windows coincide by accident, not
by transfer, and no CBA-era claim is made about college football.

**A game where BOTH sides are off a strict bye must NOT be flagged.** That is
the whole point of the cell — it isolates the side holding the rest EDGE — and
`tests/test_cfb_rest_bye_feature.py` pins it with a hand-built both-sides case.

### Cell 4 — `short_week_road`

**NFL definition, transcribed verbatim** (**read**, entry
`travel_rest_short_week_road`, `description` field):

> away_rest <= 5 days (game-level, side-specific: does short rest cost more
> specifically when it is the traveling side) -- predicted positive home_cover
> edge (pregame-safe schedule/geometry fact, no leakage caveat).

**CFB adaptation**: `away_rest <= 5` days. Column `cfb_rest_short_week_road`.
Transcribed unchanged: in CFB a `<= 5`-day turnaround is precisely the
Saturday → Thursday short week, the direct analogue of the NFL's
Sunday → Thursday. It is **thin**, and that is disclosed before scoring rather
than after: **measured**, the whole table carries 176 games at `away_rest <= 5`
(clean core: 136 of 9,093, 1.5%), against the NFL cell's 261 of 4,317 (6.0%).
The CFB market simply schedules fewer short weeks.

Because it is thin, a **sensitivity arm** is frozen here alongside it:
`short_week_road_le6` (`away_rest <= 6`), which adds the Saturday → Friday
turnaround — CFB's other genuine short week, with no NFL counterpart —
**measured** at 1,533 flagged games on the whole table. The arm is declared
with the primary and never substituted for it.

### Cell summary

| cell | column | rule | NFL sibling | NFL frozen direction (on `home_cover`) |
|---|---|---|---|---|
| `home_off_bye` | `cfb_rest_home_off_bye` | `home_rest >= 13` | `travel_rest_home_off_bye` | positive |
| `away_off_bye` | `cfb_rest_away_off_bye` | `away_rest >= 13` | `travel_rest_away_off_bye` | negative |
| `bye_edge_home` | `cfb_rest_bye_edge_home` | `home_rest >= 12 & away_rest < 12` | `bye_overval_home_edge_post2011` | negative |
| `short_week_road` | `cfb_rest_short_week_road` | `away_rest <= 5` | `travel_rest_short_week_road` | positive |
| *sens.* `home_off_bye_gap12` | `cfb_rest_home_off_bye_gap12` | `home_rest >= 12` | — | positive |
| *sens.* `away_off_bye_gap12` | `cfb_rest_away_off_bye_gap12` | `away_rest >= 12` | — | negative |
| *sens.* `short_week_road_le6` | `cfb_rest_short_week_road_le6` | `away_rest <= 6` | — | positive |

**The four cells are correlated subsets of one window and are NOT four
independent votes.** `home_off_bye` and `bye_edge_home` overlap heavily by
construction (a home team at 13+ days is also at 12+); `away_off_bye` is the
complement condition inside `bye_edge_home`. This is carried in every registry
`--notes` field and in section 8.

## 4. Per-side rest: the one element with no ready-made column

The benchmark table carries `rest_diff` but **not** the two per-side rest
values it is built from (**measured**: `rest_diff` is present, `home_rest` /
`away_rest` are not among the 60 columns). Every cell above needs a side's own
rest, so this is the one quantity that has to be derived.

**Source: the FULL CFB schedules snapshot, never the benchmark table.**
`nfl_ats.cfb.latest_cfb_snapshot(data/cfb, "schedules")` resolves
`data/cfb/schedules/raw/20260816T162105Z/` (**measured**: seasons 2001-2025,
36,915 rows, `season_type` 36,636 regular / 279 postseason, `completed` True on
36,903). This matters and is not a detail: the benchmark table is a **filtered
subset** (FBS-vs-FBS with an orientable spread and play-by-play), so a team's
actual immediately-preceding game is frequently absent from it, and a rest
value computed from the subset alone would be **wrong** — it would manufacture
byes out of games the table simply does not carry. A hand-built known-answer
case in `tests/test_cfb_rest_bye_feature.py` pins exactly this failure: a home
team whose previous game is schedule-only reads 7 days from the full schedule
and would read 21 days (a false bye) from the subset.

**Derivation: reused, not reinvented.** `nfl_ats.cfb_rest_bye_feature` imports
`nfl_ats.cfb_features._rest_base_schedule` and `_add_rest_features` — the exact
helpers that built the frozen `rest_diff` column (**read**,
`src/nfl_ats/cfb_features.py:747-808`, called at line 869). They keep completed
regular-season appearances of **any division** (only dates are consumed, never
outcomes), normalise each kickoff to its UTC date, shift within `(team_id,
season)` groups, and take the day difference. Grouping within season is what
stops a season opener inheriting the previous season's finale — the exact
cross-season bug `docs/bye_overvaluation_screen.md` records finding and fixing
on the NFL side (**read**, "Correction 2026-08-22").

**First-game rule, frozen: NaN, never 0 and never "not off bye".** A team's
first game of a season has no previous game, so its rest is undefined and every
cell needing that side is missing for that row. 0 would assert "not off bye",
which is a claim the schedule cannot support. This is the identical treatment
`rest_diff` itself already gets in the frozen contract, and the model's own
`SimpleImputer(strategy="median", add_indicator=True)` (**read**,
`src/nfl_ats/margin.py:429-435`) handles it in the training fold — the same way
it already handles `rest_diff`'s own missingness. Rows are **kept**, never
dropped: this is a feature builder feeding a model, not an evaluator.
**Measured** on the whole table: 724 rows missing home rest, 717 missing away
rest, 765 missing at least one.

**The reproduce-`rest_diff` check (known-answer test on the real table).**
`home_rest - away_rest` must equal the benchmark's own frozen `rest_diff`
wherever both are defined, with an identical missingness pattern. **Measured**
this session: **11,735 of 11,735 rows exact**, max |difference| **0.0**, **0**
missingness-pattern mismatches (765 missing on both sides). This pins the
schedules source, the season range, the regular-season/completed filters and
the within-`(team, season)` grouping against a column this session did not
build. It is asserted in `tests/test_cfb_rest_bye_feature.py`.

## 5. Coverage, and the frozen scored season set

*(Filled from the PREDICTOR-ONLY `--mode coverage` run — it reads no outcome
column — before any scoring mode was launched. Nothing else in sections 1-8 was
touched afterwards.)*

Run: `.\.tools\uv.exe run --no-sync python scripts\cfb_rest_bye_replication.py
--mode coverage`; artifact `artifacts/cfb_rest_bye_replication/coverage.json`.

**Frozen scored season set: the FULL clean core, 2012-2019 + 2021-2025, for
every one of the seven columns.** Every clean-core season has non-zero coverage
for every cell, so the section-2 rule excludes nothing — as predicted there,
before it was measured.

Measured per-season coverage (fraction of games whose `cfb_rest_home_off_bye`
is non-missing; the other columns differ by at most the away-side opener
count): 2012 94.1%, 2013 93.5%, 2014 95.6%, 2015 95.1%, 2016 94.5%, 2017 94.0%,
2018 94.3%, 2019 93.8%, [2020 88.1%, outside the clean core], 2021 94.3%, 2022
93.3%, 2023 93.7%, 2024 95.1%, 2025 94.0%. The uncovered ~6% is the season
openers of section 4, and nothing else.

Measured flag counts on the whole 12,500-row table (clean-core counts in
parentheses, **measured** separately): `cfb_rest_home_off_bye` 1,424 (1,063),
`cfb_rest_away_off_bye` 1,341 (1,001), `cfb_rest_bye_edge_home` 1,071 (785),
`cfb_rest_short_week_road` 176 (136), `cfb_rest_home_off_bye_gap12` 1,536,
`cfb_rest_away_off_bye_gap12` 1,456, `cfb_rest_short_week_road_le6` 1,533.

### Split-half reliability of the underlying trait, on two declared instruments

The panel is **one row per team per game** over the clean-core population
(**measured**: 18,186 rows), stacking both sides, carrying that team's own rest
days for that game and the cell's own propensity indicator evaluated on that
team (`own_off_bye_13`, `own_strict_bye_edge`, `own_short_week_5`, and the two
sensitivity variants). It is the team-season panel
`nfl_ats.cfb_qb_dependence.split_half_reliability` wants (`team_id`, `season`,
`week`, metric), and it is the right panel because the trait each cell reads is
a team's own propensity to arrive rested — the cell differs only in which side
of the game that same trait is read on. **Home and away cells therefore share
one reliability figure by construction, and that is correct, not a shortcut.**

Two instruments, both declared here before either was reported:

1. **`within_season_odd_even_week`** — the repo's standard instrument, splitting
   each team-season by odd/even calendar week.
2. **`across_season_odd_even_year`** — the identical function on a re-framed
   panel (one pooled "season", the calendar year in the `week` slot), so the
   odd/even split falls between SEASONS of the same program.

Both are declared because instrument 1 is **structurally inappropriate for a
calendar quantity and was known to be so before it was run**: a team's season
contains a fixed number of days and a fixed number of games, so extra rest in
one part of the calendar is arithmetically less rest elsewhere. That
compositional constraint drags any within-season split-half correlation
negative for a schedule fact, which is a property of the instrument, not
evidence about the trait.

**Measured** (clean-core panel, seed 20260901, 4,000 reliability bootstrap
draws, `n_team_seasons` 1,671 within-season / 135 across-season):

| metric | within-season odd/even week (Pearson r) | across-season odd/even year (Pearson r) | across-season Spearman-Brown | across-season P+ |
|---|---|---|---|---|
| `own_rest_days` | **-0.4745** | **+0.6208** [+0.2585, +0.7959] | 0.7660 | 1.0000 |
| `own_off_bye_13` | **-0.3368** | **+0.7052** [+0.5180, +0.8300] | 0.8271 | 1.0000 |
| `own_strict_bye_edge` | **-0.2375** | **+0.4658** [+0.2606, +0.6311] | 0.6355 | 1.0000 |
| `own_short_week_5` | **-0.0509** | **+0.4808** [+0.2956, +0.6228] | 0.6494 | 1.0000 |

The within-season column is negative on every metric, in the exact pattern the
compositional argument predicts and in descending magnitude with the rarity of
the flag. The between-season instrument, free of that constraint, says the
trait is **stable and clearly non-zero**: a program's propensity to arrive off
13+ days rest correlates +0.705 between its odd-numbered and even-numbered
seasons, Spearman-Brown 0.827, `probability_positive` 1.0000.

**`no_split_half_reliability` is not an admissible closing ground for any cell
in this document, and this is declared before the results exist**, for two
independent reasons. First, the ground requires reliability at or below 0.10
(**read**, `src/nfl_ats/weak_signals.py:83`, `NO_SPLIT_HALF_RELIABILITY_MAX`),
and the correctly-specified instrument reads 0.6355-0.8271. Second, and
sufficient on its own: these cells are **deterministic calendar facts with zero
measurement error**. There is no noisy construct here for a sample size to fail
to rescue. `--reliability` is recorded from the **across-season** instrument,
with the within-season number and this explanation carried in `--notes`.

## 6. The comparator: the XLG-03 benchmark arm, plus exactly one column

| arm | feature columns | estimator |
|---|---|---|
| baseline (shared by all cells) | `nfl_ats.cfb_features.CFB_MODEL_FEATURE_COLUMNS` (35 columns, the frozen XLG-03 contract, **including `rest_diff`**) | `nfl_ats.cfb_benchmark.fit_cfb_residual_model`, `target="market_residual"`, ridge, `alpha=10.0` |
| candidate (one per cell) | the same 35 **plus exactly one** of the seven columns in section 3 | identical |

Both arms hold regressor, alpha and target fixed at the benchmark's own frozen
values; only the feature contract differs, isolating each column's marginal
contribution against everything the benchmark already explains — **`rest_diff`
first among them**. The extension point is the benchmark's own:
`fit_cfb_residual_model`'s `feature_columns` parameter, whose docstring already
declares it (**read**, `src/nfl_ats/cfb_benchmark.py:100-103`): "a declared
candidate family … may extend it without touching the frozen benchmark path."
Candidate columns are never mixed with each other — one column per cell.

**Walk-forward.** Every scored week's models are trained on all completed games
in the table that kicked off **strictly before that week's own earliest
kickoff**, with the benchmark's own `CFB_BENCHMARK_MIN_TRAIN_GAMES = 500` floor
(**read**, `src/nfl_ats/cfb_benchmark.py:38`) — the same forward chaining
`cfb_walk_forward_benchmark` performs (**read**, same file, lines 200-209).
Training draws on the whole table (all seasons), not only the scored seasons.

**Grade, named.** The CFB benchmark grades on `spread_line`, "the median across
books of each book's home-oriented **close-proxy** spread" (**read**,
`docs/cfb_data.md`, XLG-03 section). The CFB line archive is "opener plus an
unidentified-time resolved quote (a close proxy)" and "no source records quote
observation times" (**read**, same file, "Betting-line source regimes"), so
**CFB can be graded at a close proxy and never at a verified opener**. This
replication is therefore **close-graded**. Per the binding "grade the decision
at the opener" rule, a close-graded number settles no NFL play/no-play or
promotion decision — and a CFB number could not do so in any case. The NFL
siblings were graded at the week-blocked opener-era convention of their own
batteries, which is a second reason the two are not commensurable.

### Overlap disclosure (required, not optional)

`registry/weak_signals.json` already holds two CFB rest entries, both from
`scripts/cfb_bias_battery_screen.py` and both **read** this session:

| existing CFB entry | its construct | its effect / P+ |
|---|---|---|
| `cfb_bias_battery_bye_week_rest_edge` | "Rest edge >=6 days (large rest advantage, bye-week proxy) -- **subset cover rate vs. complement**" | +0.1671, 95% [-0.2626, +0.5986], P+ 0.7719, n 16,782 team-side rows |
| `cfb_bias_battery_short_week_rest_disadvantage` | "Rest edge <=-4 days (short-week disadvantage) -- **subset cover rate vs. complement**" | +0.2248, 95% [-0.2115, +0.6606], P+ 0.8477, n 16,782 team-side rows |

They sit on an **overlapping population** (the same CFB clean core) and they
measure a **different quantity**: a subset's cover rate against its complement,
full-slate scaled, on a team-perspective relative-rest differential. This
document's quantity is the **paired accuracy delta of the XLG-03 estimator with
exactly one extra column** — a marginal against a 35-feature model, not a
cover-rate gap against a bare slate.

Therefore: **`cfb_rest_bye_replication` is declared a separate pooling family
and is NEVER pooled with `cfb_bias_battery`.** AGENTS.md's commensurability rule
("pooled inputs must be commensurable — same units, same scale, same
population — and the family must be declared before the signs are seen")
forbids it, and the family is declared here, before any sign exists. The same
disclosure is carried in every `--notes` field, in the harness docstring, and
in every artifact payload.

## 7. Metric, uncertainty, instrument checks, leakage, era split

**Metric.** The paired **candidate-minus-baseline forced-pick accuracy delta**,
in `accuracy_points` (percentage points), picks taken at
`home_cover_probability >= 0.5` and graded with `nfl_ats.clv.pick_correct`
(pushes NaN, excluded). Positive = the extra column helped.

**Uncertainty.** `nfl_ats.clv.week_blocked_bootstrap`, **week-blocked primary**
(within-week correlation is ZERO by owner mandate — no ICC term),
**season-blocked secondary**, never averaged together. **1,000 samples**, seed
**20260901**.

**Within-week permutation null**, **200 draws**: both arms' models are fit ONCE
on the REAL `ats_margin`; only the grading margin is shuffled within week, so
the draws cost no extra fits. This null is **not** centred on zero by design
(it preserves each week's realised home-cover rate and the two arms carry
different home-pick rates) and is reported ALONGSIDE the bootstrap-vs-zero
interval, never instead of it.

**Positive control**, run BEFORE the real screen, per cell: the candidate's one
new column is REPLACED by the realised `ats_margin` — a deliberate, large
leak — so the harness must show an obvious, large effect. This proves the FULL
36-column ridge fit (not a single-feature model) can detect a real effect of
meaningful size when one is present. It is identical across cells by
construction (the same leaked column) and is run per cell anyway.

**Leakage.** Regression-tested in `tests/test_cfb_rest_bye_feature.py`, which
must pass before any scoring mode runs:

1. **Pure function of pregame schedule facts.** Every candidate column is
   bit-identical after `result`, `ats_margin`, `home_points` and `away_points`
   are permuted — one test per column, plus one permuting all four at once,
   plus one that drops all four columns entirely and asserts nothing changes.
   The only inputs read anywhere in the derivation are `season`, `week`,
   kickoff dates, team ids, `season_type` and `completed`.
2. **Backward-looking in time.** Appending a LATER game to the schedule never
   changes an earlier game's rest.
3. **The first-game rule.** Season openers are NaN on every cell, never 0.
4. **Known answers.** A hand-built eleven-game fixture with hand-computed rest
   for every team, covering a season opener, a true open date, a
   both-sides-off-bye game `bye_edge_home` must not flag, a one-side-undefined
   game, a game whose correct answer is only reachable from the full schedule,
   the 12-vs-13 threshold boundary, and the 5-vs-6 short-week boundary.
5. **The real-table known answer**: the reproduce-`rest_diff` check of
   section 4.

**Era split, declared before scoring.** The window spans the benchmark's own
declared 2020 regime gap, the same boundary `docs/fluview_cfb_replication.md`
uses. Two eras, magnitudes reported separately and **never averaged across a
sign flip** (owner rule "era magnitude, not presence"): **era A = 2012-2019**,
**era B = 2021-2025**. Per-season deltas are also reported.

## 8. Decision rule and recording, frozen before scoring

**Decision rule.** Expected value, never a threshold: `probability_positive`
above 0.5 favours the candidate over the baseline. Predeclared thresholds
govern only what a document may CLAIM. **A CFB result is replication evidence
about a mechanism; it never by itself changes an NFL card**, and this run is
close-graded besides, so it settles no play/no-play or promotion decision in
either league.

**Recording.** Four `nfl-ats weak-signals record` entries, one per primary cell,
written through the cross-process lock because other agents are writing the
same registry this session:

| cell | registry name | NFL sibling |
|---|---|---|
| `home_off_bye` | `cfb_rest_home_off_bye_on_benchmark` | `travel_rest_home_off_bye` |
| `away_off_bye` | `cfb_rest_away_off_bye_on_benchmark` | `travel_rest_away_off_bye` |
| `bye_edge_home` | `cfb_rest_bye_edge_home_on_benchmark` | `bye_overval_home_edge_post2011` |
| `short_week_road` | `cfb_rest_short_week_road_on_benchmark` | `travel_rest_short_week_road` |

`--league cfb`, `--effect-units accuracy_points`, `--family
cfb_rest_bye_replication`, `--category schedule`, week-blocked interval and
`probability_positive`, `--reliability` carrying the CFB-measured across-season
figure from section 5 (never an NFL figure — all four NFL siblings carry
`reliability: null`). Era slices are recorded as
`<name>_era_2012_2019` / `<name>_era_2021_2025` in the same family **where the
era magnitudes differ materially** — judgement, exercised in section 9 and
stated there.

Every `--notes` field carries all five required disclosures: (i) close-graded
CFB, no verified opener exists; (ii) the four cells are correlated subsets of
one window and are **not** independent votes; (iii) the NFL sibling entry name;
(iv) the `cfb_bias_battery` overlap disclosure of section 6; (v) that the
baseline **already carries `rest_diff`**, so this is a marginal.

**A separate pooling bucket, stated explicitly.** `cfb_rest_bye_replication` is
a different family from `cfb_bias_battery`, from `travel_rest_battery` and from
`bye_overvaluation_screen`. Three different comparators across two leagues; not
commensurable; never pooled together.

**Classification.** `unresolved_below_power` for every cell unless a cell
literally meets a terminal ground: `wrong_sign_resolved` requires the WHOLE
week-blocked interval on the wrong side of zero, and `positive_control_bound`
requires the control to have PROVEN detection of an effect of the size in
question. `no_split_half_reliability` is unavailable to every cell for the two
independent reasons in section 5. An interval containing zero is not a ground;
if a record command errors, the verdict is wrong, not the validator.

**No rotation window is spent.** `nfl-ats rotation` is not invoked by this
document, matching both prior CFB replications' precedent.

## 9. Results (added after the look, 2026-09-01)

Every number in this section is **measured** by
`.\.tools\uv.exe run --no-sync python scripts\cfb_rest_bye_replication.py`
this session; each table names its own artifact. Nothing above this line was
edited after the first outcome sign was computed, except section 5, which was
filled in from the PREDICTOR-ONLY `--mode coverage` run before any of the three
scoring modes was launched.

**The scored population, as frozen.** 8,933 games across 199 week blocks and 13
seasons (2012-2019, 2021-2025). The XLG-03 baseline arm's own forced-pick
accuracy on that population is **51.595%** and its home-pick rate is **41.67%**.
Feature-missing (season-opener) rows inside the scored population: 522-548
depending on the cell.

### Instrument check 1 — within-week permutation null (200 draws, `--mode null`)

| cell | artifact | null mean | null sd | null 95% |
|---|---|---|---|---|
| `home_off_bye` | `20260901T192914Z` | **+0.015** pts | 0.204 | [-0.414, +0.403] |
| `away_off_bye` | `20260901T193006Z` | **+0.044** pts | 0.233 | [-0.403, +0.505] |
| `bye_edge_home` | `20260901T193047Z` | **+0.011** pts | 0.139 | [-0.280, +0.291] |
| `short_week_road` | `20260901T193127Z` | **-0.001** pts | 0.138 | [-0.235, +0.269] |

All four finite and centred within 0.05 points of zero. **The harness
manufactures no effect**, so nothing below is an artifact of the evaluator.

### Instrument check 2 — positive control (`--mode positive-control`)

Artifacts `20260901T193234Z` / `193332Z` / `193432Z` / `193536Z`. Identical for
all four cells by construction (the same leaked column, as predeclared in
section 7): pooled **+48.405 accuracy points**, week-blocked P+ **1.000**, 95%
[+47.374, +49.464], n=8,933. Per era: 2012-2019 **+48.345** (P+ 1.000, 95%
[+47.012, +49.567]), 2021-2025 **+48.493** (P+ 1.000, 95% [+46.767, +50.058]).
The full 36-column ridge fit is **not blind** to a real effect of meaningful
size, in either era.

### The real screen (`--mode screen`)

Artifacts `20260901T193647Z` (home_off_bye), `193759Z` (away_off_bye),
`193908Z` (bye_edge_home), `194017Z` (short_week_road).

| cell | pooled delta | week 95% CI | week P+ | season 95% CI | season P+ | n games / weeks / flagged | null percentile |
|---|---|---|---|---|---|---|---|
| `home_off_bye` | **-0.146** pts | [-0.615, +0.301] | **0.273** | [-0.537, +0.290] | 0.233 | 8,933 / 199 / 1,063 | 21.0th |
| `away_off_bye` | **+0.101** pts | [-0.375, +0.606] | **0.646** | [-0.333, +0.526] | 0.676 | 8,933 / 199 / 1,001 | 60.5th |
| `bye_edge_home` | **-0.078** pts | [-0.377, +0.215] | **0.296** | [-0.424, +0.253] | 0.343 | 8,933 / 199 / 785 | 22.0th |
| `short_week_road` | **-0.011** pts | [-0.300, +0.280] | **0.442** | [-0.316, +0.268] | 0.484 | 8,933 / 199 / 136 | 44.0th |

### Per-era magnitudes, never averaged across a sign flip

Owner rule "era magnitude, not presence". Era A n=5,349 (122 weeks), era B
n=3,584 (77 weeks).

| cell | era 2012-2019 | era 2021-2025 | sign flip? |
|---|---|---|---|
| `home_off_bye` | **+0.093** pts, P+ 0.632, 95% [-0.431, +0.606], 633 flagged | **-0.502** pts, P+ 0.105, 95% [-1.341, +0.333], 430 flagged | **yes** |
| `away_off_bye` | **+0.093** pts, P+ 0.645, 95% [-0.508, +0.665], 605 flagged | **+0.112** pts, P+ 0.588, 95% [-0.731, +0.906], 396 flagged | no |
| `bye_edge_home` | **+0.093** pts, P+ 0.743, 95% [-0.164, +0.354], 485 flagged | **-0.335** pts, P+ 0.125, 95% [-0.954, +0.245], 300 flagged | **yes** |
| `short_week_road` | **+0.187** pts, P+ 0.909, 95% [-0.076, +0.451], 85 flagged | **-0.307** pts, P+ 0.125, 95% [-0.846, +0.222], 51 flagged | **yes** |

**Era A is positive in all four cells; era B is negative in three of the four.**
Because the four columns are correlated subsets of one window (section 3), that
is closer to **one** observation about a regime than to four independent ones,
and it is reported as such. The three sign-flipping cells therefore have their
per-era magnitudes recorded as separate registry rows; `away_off_bye`, whose
eras agree in sign and to within 0.02 points in magnitude, does not.

**A coincidence, disclosed and verified rather than left to look like a bug.**
Three different cells read exactly **+0.093** points in era A. That is a net of
exactly +5 correct picks out of 5,349 for each, and it is a genuine collision of
small integers, not a duplicated computation: **measured** this session by
re-running the walk-forward for all four cells and comparing per game, the cells
flip a different number of picks against the baseline (`home_off_bye` 200 flips,
99 gains / 94 losses; `away_off_bye` 246 flips, 123/118; `bye_edge_home` 51
flips, 27/22; `short_week_road` 68 flips, 38/28) and their candidate
probabilities differ by up to 0.145 game-by-game. `short_week_road` nets +10,
not +5.

Per-season deltas (accuracy points): `home_off_bye` 2012 +0.48, 2013 +0.62, 2014
+0.77, 2015 -0.45, 2016 +0.75, 2017 -0.58, 2018 -0.57, 2019 -0.14, 2021 0.00,
2022 -0.87, 2023 +0.82, 2024 -1.66, 2025 -0.80. `away_off_bye` +0.48, +0.94,
+0.92, 0.00, +0.30, -1.02, +0.57, -1.29, +0.29, 0.00, +1.37, -0.83, -0.27.
`bye_edge_home` +0.16, 0.00, +0.61, +0.15, +0.30, +0.29, -0.43, -0.29, -0.43,
-0.14, +0.96, -1.66, -0.40. `short_week_road` +0.32, -0.47, +0.31, +0.15, +0.45,
+0.15, +0.14, +0.43, -0.72, -0.14, +0.55, -1.39, +0.13.

### The declared sensitivity arms

Frozen in section 3 with the primaries, run on the identical population and
never substituted for them. Artifacts `20260901T194146Z`, `194246Z`, `194353Z`.

| arm | pooled delta | week 95% CI | week P+ | season P+ | flagged | era 2012-2019 | era 2021-2025 |
|---|---|---|---|---|---|---|---|
| `home_off_bye_gap12` (`>=12`) | -0.101 pts | [-0.423, +0.204] | 0.268 | 0.196 | 1,142 | +0.093, P+ 0.676 | -0.391, P+ 0.065 |
| `away_off_bye_gap12` (`>=12`) | **+0.157** pts | [-0.317, +0.632] | **0.734** | 0.758 | 1,088 | +0.206, P+ 0.764 | +0.084, P+ 0.574 |
| `short_week_road_le6` (`<=6`) | -0.090 pts | [-0.460, +0.268] | 0.279 | 0.267 | 1,123 | -0.131, P+ 0.293 | -0.028, P+ 0.453 |

Widening the off-bye gap from 13 to 12 days moves both off-bye cells in the
same direction they already pointed and by less than 0.06 points — the primary
threshold is not carrying the result. Widening the short week from `<=5` to
`<=6` takes the flagged count from 136 to 1,123 and takes the reading from
-0.011 (P+ 0.442) to -0.090 (P+ 0.279); the Saturday-to-Friday turnaround the
NFL calendar cannot produce does not rescue the cell. The sensitivity arms are
**not** recorded in the registry: they are threshold robustness for the four
predeclared cells, not four more cells.

### Did the NFL direction replicate? Per cell, plainly

**First, the caveat that has to come before the answer, because without it the
answer is misread.** The NFL entries' `effect` is a **subset-cover-rate gap
against the complement**, full-slate scaled (**read**,
`docs/travel_rest_battery.md`, "Method"). This document's effect is a **paired
accuracy delta of a 35-feature model with one extra column**. They are not the
same quantity, they are not on the same scale, and section 6 keeps them in
separate pooling families for exactly this reason. A sign agreement below is
**suggestive, not a like-for-like replication**, and a like-for-like CFB
cover-rate arm was not predeclared here and is therefore not computed post-hoc.

**Correction to section 3's wording, made here rather than by editing it.**
Section 3 opens "Sign convention for every cell, identical to the NFL entries:
the effect is the candidate-minus-baseline paired accuracy delta". The first
four words are wrong: the NFL entries are **not** signed that way, they are
subset-cover-rate gaps, as the paragraph above establishes from
`docs/travel_rest_battery.md`. The sign convention this document actually uses
is exactly as stated — positive means the extra column helped the model — and
nothing in the measurement changes. Section 3 is left untouched because it is a
predeclaration; the error is corrected in place here, following the precedent
`docs/bye_overvaluation_screen.md` sets with its own "Correction 2026-08-22"
section.

| cell | NFL recorded effect / P+ | CFB recorded effect / P+ | sign match? |
|---|---|---|---|
| `home_off_bye` | -0.1523, P+ 0.208 | **-0.1455**, P+ 0.273 | **yes** — and within 0.007 points |
| `away_off_bye` | -0.0622, P+ 0.384 | **+0.1008**, P+ 0.646 | **no** — opposite |
| `bye_edge_home` | -0.3304, P+ 0.064 | **-0.0784**, P+ 0.296 | **yes** — same sign, ~1/4 the magnitude |
| `short_week_road` | +0.0442, P+ 0.593 | **-0.0112**, P+ 0.442 | **no** — opposite, both within 0.05 of zero |

Two of four match on sign. The closest correspondence is `home_off_bye`
(-0.152 NFL vs -0.146 CFB); the strongest NFL reading in the family,
`bye_edge_home`, keeps its sign on CFB but at roughly a quarter of the magnitude
and with P+ moving from 0.064 to 0.296 — the CFB sample does not sharpen it.

### What this implies for the decision, before what is wrong with it

**On EV grounds — `probability_positive` above 0.5 favours the candidate, the
only decision rule this project uses — exactly one of the four columns is worth
carrying on the CFB benchmark arm today.** `away_off_bye` reads P+ **0.646**
week-blocked and **0.676** season-blocked: taking the baseline over it is taking
the short side of a roughly 2-to-1 bet, and its declared `>= 12` sensitivity arm
is stronger still at P+ **0.734 / 0.758**. If a card had to be submitted with or
without that column, it goes on. The other three go the other way —
`home_off_bye` P+ 0.273, `bye_edge_home` P+ 0.296, `short_week_road` P+ 0.442 —
and the correct decision on those, on the same EV rule, is to leave them off.

**The most decision-relevant structure in the battery is the era split, and it
must not be averaged away.** Era 2012-2019 favours the candidate in **all four**
cells (P+ 0.632, 0.645, 0.743, 0.909) and era 2021-2025 opposes it in **three of
four** (P+ 0.105, 0.588, 0.125, 0.125). The single strongest reading anywhere in
this document is `short_week_road` in 2012-2019: **+0.187 points, P+ 0.909
week-blocked, 0.952 season-blocked**, on only 85 flagged games. Because the four
columns are correlated subsets of one window, that pattern is closer to one
observation about a regime change than to four; it is recorded per era so a
later reader can weigh it, and it is not claimed as four votes.

**For the NFL card: nothing changes, and nothing was ever going to.** A CFB
result is replication evidence about a mechanism and never by itself changes an
NFL card (section 8), and this run is close-graded besides. What it does change
is the expected value of spending further NFL window on these constructs. The
NFL family's strongest entry, `bye_overval_home_edge_post2011` (P+ 0.064, the
market-overprices-the-bye mechanism), gets **weak, same-signed, non-sharpening**
support here: on 8,933 independent CFB games against a model that already prices
rest, the analogous column reads -0.078 points at P+ 0.296. That is a lean in
the same direction, not a confirmation, and it is a quarter of the size.
`away_off_bye` is the one cell whose CFB reading is *better* than its NFL
sibling's (P+ 0.646 vs 0.384), and it is the cell the NFL battery treated as the
mirror afterthought.

**What is NOT concluded, and why.** No cell is closed. `wrong_sign_resolved`
requires the WHOLE week-blocked interval on the wrong side of zero; every
interval in the screen table crosses it. `bounded_by_control` requires the
instrument to have been PROVEN able to detect an effect of the size in
question — the control proved detection of a +48.4-point leak, which says
nothing about a sub-0.2-point effect, and every week-blocked CI here spans
0.58-0.92 points, several times wider than the effects being tested.
`no_split_half_reliability` is unavailable for the two independent reasons
frozen in section 5 (the across-season instrument reads Spearman-Brown
0.635-0.827, far above the 0.10 ceiling; and a deterministic calendar fact has
zero measurement error). All four cells are recorded `unresolved_below_power`.
**An interval containing zero is not grounds for rejection, and none of the
above rests on one.**

**Caveats, after the numbers rather than instead of them.** (1) Every cell here
is a **marginal on a baseline that already carries `rest_diff`**, so a null
reading says the linear rest term already captured what the threshold adds — it
does not say rest is unpriced or that rest does not matter. (2) CFB is a
different, softer market: the XLG-03 baseline picks at 51.595% here, so a null
on CFB constrains but does not strictly bound what the same construct can do
against the NFL market. (3) `short_week_road` is genuinely thin at 136 flagged
games (85 in era A, 51 in era B) because the CFB calendar schedules few
Saturday-to-Thursday turnarounds; its era-A P+ 0.909 rests on 85 games and its
interval says so. (4) The four cells are correlated subsets of one window and
are not four independent votes; the registry notes carry that on every row.
(5) This is close-graded and CFB besides, so it settles no NFL play/no-play or
promotion decision by itself.

### Registry, verified by reading it back

`registry/weak_signals.json`: **653 -> 663** signals (**measured** before and
after; every write went through the cross-process lock wrapper because other
agents were writing the same file this session). Ten entries, all read back and
checked field-by-field against their artifacts, all with `league: cfb`,
`family: cfb_rest_bye_replication`, `category: schedule`,
`classification: unresolved_below_power`, `closing_ground: null`, and all five
required disclosures present in `notes`:

| entry | effect | interval | P+ | n games / blocks | reliability | seasons |
|---|---|---|---|---|---|---|
| `cfb_rest_home_off_bye_on_benchmark` | -0.1455 | [-0.6148, +0.3009] | 0.273 | 8,933 / 199 | 0.8271 | 2012-2025 |
| `cfb_rest_home_off_bye_on_benchmark_era_2012_2019` | +0.0935 | [-0.4310, +0.6062] | 0.632 | 5,349 / 122 | 0.8271 | 2012-2019 |
| `cfb_rest_home_off_bye_on_benchmark_era_2021_2025` | -0.5022 | [-1.3411, +0.3327] | 0.105 | 3,584 / 77 | 0.8271 | 2021-2025 |
| `cfb_rest_away_off_bye_on_benchmark` | +0.1008 | [-0.3754, +0.6059] | 0.646 | 8,933 / 199 | 0.8271 | 2012-2025 |
| `cfb_rest_bye_edge_home_on_benchmark` | -0.0784 | [-0.3769, +0.2146] | 0.296 | 8,933 / 199 | 0.6355 | 2012-2025 |
| `cfb_rest_bye_edge_home_on_benchmark_era_2012_2019` | +0.0935 | [-0.1636, +0.3542] | 0.743 | 5,349 / 122 | 0.6355 | 2012-2019 |
| `cfb_rest_bye_edge_home_on_benchmark_era_2021_2025` | -0.3348 | [-0.9544, +0.2451] | 0.125 | 3,584 / 77 | 0.6355 | 2021-2025 |
| `cfb_rest_short_week_road_on_benchmark` | -0.0112 | [-0.2998, +0.2801] | 0.442 | 8,933 / 199 | 0.6494 | 2012-2025 |
| `cfb_rest_short_week_road_on_benchmark_era_2012_2019` | +0.1870 | [-0.0755, +0.4508] | 0.909 | 5,349 / 122 | 0.6494 | 2012-2019 |
| `cfb_rest_short_week_road_on_benchmark_era_2021_2025` | -0.3069 | [-0.8460, +0.2222] | 0.125 | 3,584 / 77 | 0.6494 | 2021-2025 |

`away_off_bye` has no era rows because its two eras agree in sign and differ by
0.019 points — the judgement section 8 reserved, exercised.

`registry/rotation_registry.json` is **untouched**: no NFL rotation window was
declared, assigned or spent by this document.

### Files added

- `docs/cfb_rest_bye_replication.md` (this document).
- `src/nfl_ats/cfb_rest_bye_feature.py` — per-side rest derivation, the four
  candidate columns, the three declared sensitivity columns, the team-season
  reliability panel.
- `scripts/cfb_rest_bye_replication.py` — `--mode coverage | null |
  positive-control | screen`, `--cell` over seven columns.
- `tests/test_cfb_rest_bye_feature.py` — 38 leakage, known-answer,
  first-game-rule and contract tests.
- `artifacts/cfb_rest_bye_replication/` — `coverage.json` plus fifteen run
  artifacts (4 null, 4 positive-control, 7 screen).
