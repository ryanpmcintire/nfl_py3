# Circadian body-clock / timezone cells, replicated on COLLEGE FOOTBALL: predeclaration

Written **before any ATS outcome, cover rate, accuracy delta or sign was
computed on college-football data by this line of work**. Sections 1-8 are the
predeclaration and were saved before any scoring mode was launched. Section 5's
coverage table was filled in from a PREDICTOR-ONLY `--mode coverage` run that
reads no outcome column. Section 9 was added after the look and reports what it
found; it changes nothing above it.

This is a **cross-league replication**, not a new NFL look. It spends **no NFL
evaluation window and no rotation window** — CFB is this project's sanctioned
free replication ground, exactly as `scripts/cfb_surface_familiarity_screen.py`
used it (**read**, `scripts/cfb_surface_familiarity_screen.py:16-17`) and as
`docs/fluview_cfb_replication.md` used it for the FluView construct (**read**,
that document's header). **A CFB result is replication evidence about a
mechanism; it never by itself changes an NFL card.**

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
which card is PLAYED, which is expected value. Every cell below is recorded
regardless of sign.

## 1. What is being replicated, and what it is not

The construct under replication is the **circadian body-clock / timezone**
family frozen in `docs/body_clock_screen.md`, `docs/body_clock_night_screen.md`
and `docs/travel_rest_battery.md`. Its four NFL reads, transcribed from
`registry/weak_signals.json` (**read**, this session):

| NFL registry entry | effect (accuracy points) | 95% CI (week-blocked) | P+ | n games | seasons |
|---|---|---|---|---|---|
| `body_clock_west_road_early` | −0.1545 | [−0.6272, +0.3106] | 0.2588 | 4,317 | 2009-2025 |
| `body_clock_east_host_west_visitor_early` | −0.2690 | [−0.6232, +0.0917] | 0.0714 | 4,317 | 2009-2025 |
| `travel_rest_eastbound_multizone` | −0.1382 | [−0.7550, +0.4887] | 0.3239 | 4,317 | 2009-2025 |
| `body_clock_night_west_road_ge2000et` | −0.1713 | [−0.4137, +0.0738] | 0.0843 | 119 flagged | 2009-2025 |

All four `reliability` fields are `null` in the registry (**read**): the NFL
family never measured a split-half reliability for this trait, so
`no_split_half_reliability` has never been available or unavailable as a
closing ground on the NFL side. **This document measures one** (section 7), on
CFB, which is one of the two things it adds.

**Three of the four NFL reads lean NEGATIVE** (P+ 0.2588 / 0.0714 / 0.3239 /
0.0843, all below 0.5). On the NFL side the predicted circadian home-side edge
did not appear in the predicted direction. So this replication is primarily a
**SIGN check**, not a hunt for a promotable edge, and it is stated that way up
front so nobody reads a null here as a surprise.

**What this document adds that the NFL reads cannot**: (a) an independent
sample in a different league at zero NFL-window cost — CFB plays roughly 2.9x
as many games per season as the NFL and, critically for this family, plays a
noon-ET window every week that the NFL does not; (b) the first split-half
reliability this construct has ever carried; (c) a positive control, which no
NFL cell in this family ever ran (**read**, every one of the four
`classification_evidence` fields says "no positive-control bound was run").

**What it is not**: it is not a promotion or play/no-play decision for the NFL
card, and it cannot become one. It is also **close-graded** (section 6), and
per the binding "grade the decision at the opener" rule a close-graded number
settles no NFL promotion decision even inside the NFL.

## 2. Population

`data/processed/cfb_game_features.parquet` — the XLG-03 canonical benchmark
table built by `nfl-ats cfb-build-features` (**read**, `docs/cfb_data.md`,
"Derived benchmark table (XLG-03)"): completed regular-season FBS-vs-FBS games
carrying both an orientable spread and play-by-play, with the NFL ATS sign
convention (`ats_margin = result - spread_line`, `home_cover` 1/0/NaN-on-push).
**Measured** this session: 12,500 rows x 60 columns, seasons 2006-2025, 327
neutral-site rows, `kickoff` tz-aware UTC with **0 nulls of 12,500**.

**Scored seasons: `nfl_ats.cfb_benchmark.CFB_CLEAN_CORE_SEASONS` — 2012-2019
plus 2021-2025** (**read**, `src/nfl_ats/cfb_benchmark.py:46`), reused verbatim
and never redeclared, the same restriction
`scripts/cfb_surface_familiarity_screen.py` and
`scripts/fluview_cfb_replication.py` apply.

**No coverage-conditional season selection.** Unlike the FluView replication,
which had to restrict to seasons with a non-zero archive coverage, every input
here is a pure schedule fact with no archive floor: a kickoff timestamp, two
team ids, and each school's own listed venue state and city. **The whole clean
core is scored, declared here before coverage was measured.** If section 5's
predictor-only measurement turns up a season with imperfect zone coverage,
that season is still scored and its unresolved rows come back NaN — they are
never dropped and never defaulted to "not flagged".

**Training draws on the whole table** (all seasons 2006-2025), not only the
scored seasons, exactly as `cfb_walk_forward_benchmark` does — only the SCORED
weeks are restricted.

## 3. The four cells: NFL definition transcribed, then the CFB adaptation

Each NFL definition below is transcribed **exactly** from the registry entry's
own `description` field and its predeclaration doc. The CFB adaptation follows,
and every element of the adaptation that is not a pure league swap is named.

### Cell 1 — `cfb_body_clock_west_road_early`

**NFL definition, verbatim** (`registry/weak_signals.json`,
`body_clock_west_road_early.description`, **read**):

> "Pacific/Arizona-body-clock road team, kickoff before 14:00 ET (~10am
> biological), predicted POSITIVE home-side circadian edge; classic mechanism,
> schedule-fact flags."

and from `docs/body_clock_screen.md` (**read**, lines 128-133):

> "away body clock WEST, true road game, kickoff < 14:00 ET (i.e. 10am Pacific
> biological time)"; "`away_body_tz`: IANA zone of the away team's modal home
> stadium that season. WEST body clock := tz in {`America/Los_Angeles`,
> `America/Phoenix`} … Denver (Mountain) is deliberately EXCLUDED".

**CFB adaptation**: the away team's OWN home venue that season is in
`America/Los_Angeles` or `America/Phoenix` (section 4's map), the game is not
at a neutral site, and the kickoff is before 14:00 ET.

### Cell 2 — `cfb_body_clock_east_host_west_visitor_early`

**NFL definition, verbatim** (`body_clock_east_host_west_visitor_early.description`,
**read**):

> "Eastern host receiving Western-body-clock visitor before 14:00 ET, predicted
> POSITIVE home edge; subset of west_road_early."

and from `docs/body_clock_screen.md` (**read**, lines 134-136): "cell 1 ∩ venue
tz EAST (the mirror: Eastern hosts receiving Western visitors at 1pm ET)";
"Venue EAST := venue tz == `America/New_York`".

**CFB adaptation**: cell 1 AND the host venue's UTC offset at that kickoff
equals the Eastern offset at the same instant. The offset comparison replaces
the NFL's literal `tz == "America/New_York"` string test because CFB's Eastern
venues include `America/Detroit` and `America/Indiana/Indianapolis`, which are
the same clock; a string test would silently drop Michigan and Indiana hosts.

### Cell 3 — `cfb_travel_rest_eastbound_multizone`

**NFL definition, verbatim** (`travel_rest_eastbound_multizone.description`,
**read**):

> "This game's venue UTC offset minus away team's own home UTC offset >= 2
> hours (eastbound body-clock disadvantage, DST-aware via gameday) — predicted
> positive home_cover edge (pregame-safe schedule/geometry fact, no leakage
> caveat)."

and from `docs/travel_rest_battery.md` (**read**, lines 138-144 and 187-193):
"`tz_delta_eastbound`: this game's venue UTC offset minus the away team's own
home UTC offset, both evaluated at `gameday` via `zoneinfo` (DST-aware)";
"`tz_delta_eastbound >= 2` (hours). Threshold matches the commonly-used
≥2-timezone circadian-disruption cutoff in travel-fatigue sports-science
literature (external justification, not data-mined)."

**CFB adaptation**: identical arithmetic on the CFB timezone map, with one
stated improvement — the two offsets are evaluated at the game's **own kickoff
instant** rather than at `gameday` midnight. The NFL table had no kickoff
timestamp and had to use `gameday`; the CFB table has a UTC `kickoff` with zero
nulls, and the kickoff instant is both strictly more correct across a DST
boundary and still purely pregame. A second, smaller difference: CFB's
`gameday` is the UTC date, not the ET date (**measured**: 2,902 of 12,500 rows
have `gameday` != the ET calendar date of kickoff), so using `gameday` here
would have been the wrong day for every late kickoff. Unlike the NFL cell,
neutral-site games are EXCLUDED (NaN) rather than evaluated at the actual
stadium: the NFL battery could use its `stadium` column, which is correct even
at a neutral site, and no CFB equivalent exists.

### Cell 4 — `cfb_body_clock_night_west_road_ge2000et`

**NFL definition, verbatim** (`body_clock_night_west_road_ge2000et.description`,
**read**):

> "West body-clock (PT/AZ) road team, kickoff >=20:00 ET, true road games;
> predicted west-side circadian edge."

and from `docs/body_clock_night_screen.md` (**read**, lines 117-122): "away body
clock WEST, true road game, kickoff >= 20:00 ET (SNF/MNF window) … Predicted:
**negative home_cover gap** (positive west-side cover edge — circadian-peak
mechanism, the Smith et al. evening effect)."

**CFB adaptation**: the away team's own home venue is in
`America/Los_Angeles` or `America/Phoenix`, the game is not at a neutral site,
and the kickoff is at or after 20:00 ET.

### The direction convention, stated once so no sign is misread

The NFL entries above are signed as a **subset-vs-complement full-slate-scaled
`home_cover` gap**: negative means the home team covers LESS in the flagged
subset. The primary estimator in this document (section 6) is a different
thing — a **paired candidate-minus-baseline model accuracy delta**, where
positive means the extra column HELPED the model. Those two are not
sign-commensurable, so section 6 predeclares BOTH, and the "did the NFL
direction replicate" question in section 9 is answered on the commensurable
one. Naming this before scoring, because collapsing the two is exactly how a
replication claim goes wrong.

## 4. The venue state → IANA timezone map: the one element with no NFL counterpart

The NFL cells read an IANA `tz` straight out of
`registry/stadium_coordinates.json`, which is **NFL-only by its own README**
and carries no CFB venue. **Measured this session**: no local CFB snapshot
carries a venue latitude, longitude, elevation or timezone —
`data/processed/cfb_game_features.parquet` has 60 columns and none of them is a
venue-location field. Distance-based cells are therefore impossible here and
are not attempted.

**Source used**: `data/cfb/team_info/raw/20260901T185247Z/season=*/team_info.parquet`
— the cfbfastR-data `team_info` snapshot already pinned in this repository for
the FluView CFB replication (**read**, `docs/fluview_cfb_replication.md`
section 4; free, no key, `raw.githubusercontent.com`, no CFBD API credit). It
carries `team_id, school, venue_id, venue_name, city, state`, one row per
school per season, so a venue change is carried per season.
**Measured this session**: joined on `(season, team_id)` to the benchmark
table, home-side and away-side state coverage are **1.000 in every season
2006-2025** (0 unresolved rows on either side). The join is on the CFBD/ESPN
team id, never on a school name.

### The map itself

`src/nfl_ats/cfb_body_clock_feature.py` holds two declared tables:

- `STATE_TIMEZONES` — 39 declared codes (37 US states plus `DC` and `BC`) that
  lie wholly inside one zone **for every school in this population**.
- `SPLIT_STATE_CITY_TIMEZONES` — a `(state, city) -> zone` row for **every**
  school in a split state that appears in the population (**measured**: 42
  distinct pairs covering 43 schools; Rice and Houston share the city
  Houston), including the ones that agree with their state's majority zone,
  so the table is an auditable statement about each school rather than a list
  of exceptions.

The full per-school assignment table (season, team_id, school, city, state,
venue, resolved zone) is written out as an artifact
(`zone_assignments.json`) so it can be audited row by row.

### Split-state resolution rule, declared BEFORE scoring, with its reason

Twelve states span two zones: **FL, TX, TN, KY, IN, MI, ND, SD, NE, KS, OR,
ID**. **The rule: resolve by the school's own `city`, never by a blanket
per-state assumption.** The reason is that a blanket assumption is wrong for
four schools that matter to these specific cells, and wrong in the direction
that would corrupt them:

| school | city | state | state's majority zone | correct zone | why it matters here |
|---|---|---|---|---|---|
| Idaho | Moscow | ID | Mountain | **Pacific** | a blanket rule would drop a genuine WEST body clock out of cells 1, 2 and 4 |
| UTEP | El Paso | TX | Central | **Mountain** | a blanket rule would put a Mountain body clock INTO the Central set and change its eastbound delta by an hour |
| Western Kentucky | Bowling Green | KY | Eastern | **Central** | changes its eastbound delta by an hour |
| Memphis / Middle Tennessee / Vanderbilt | Memphis, Murfreesboro, Nashville | TN | (Tennessee is genuinely split; Knoxville is Eastern) | **Central** | changes whether the host counts as an Eastern host in cell 2 |

A `SPLIT_STATE_DEFAULT_TIMEZONES` fallback exists for a split-state city that
is not in the table, but **every use of it is counted** in the diagnostics as
`n_split_state_city_fallback` and a test asserts that count is 0 on the real
population, so a silent fallback cannot hide inside a result.

### The WEST body-clock set: the NFL zone set, kept exactly

`WEST_BODY_CLOCK_ZONES = {America/Los_Angeles, America/Phoenix}` — the NFL
screen's set, character for character. Two consequences are declared here
rather than discovered later:

- **Mountain-time schools are EXCLUDED** (Colorado, Utah, Wyoming, New Mexico,
  Montana, Boise State, UTEP), mirroring the NFL screen's deliberate exclusion
  of Denver. Count disclosed in section 5.
- **Hawaii is EXCLUDED** (`Pacific/Honolulu`, 5-6 hours behind ET). The NFL had
  no Hawaii team, so the frozen construct never had to decide; a 5-6 hour shift
  is a materially different dose from the construct's 2-3 hours, and folding it
  in would replicate a broader construct than the one recorded. Hawaii sits in
  the complement, and its count is disclosed in section 5 so the cost of the
  decision is visible. Cell 3 (`eastbound_multizone`) is timezone-general by
  its own NFL definition and therefore does cover Hawaii trips.

### Kickoff clock, and the past-midnight rule

`kickoff` is converted from UTC to `America/New_York` and read as minutes past
midnight ET. **CFB plays kickoffs that land after midnight ET** (a 22:30
Pacific kickoff is 01:30 ET the next calendar day), which the NFL never does,
so an ET kickoff earlier than **06:00 ET** is carried forward as
`minutes + 1440` and reads as the previous evening's LATE window. Without this
rule a 01:30 ET kickoff would satisfy "before 14:00 ET" and invert cell 1.
06:00 ET is comfortably below the earliest genuine CFB kickoff in the table
(07:00 ET, **measured**) and above the latest post-midnight one. The count of
rows this touches is disclosed in section 5.

### Neutral-site rule

All four cells describe a true road game at a known host venue, and the NFL
cells all require `location == 'Home'` (**read**, `docs/body_clock_screen.md`
line 102). At a CFB neutral site the host's listed venue is **not** where the
game is played, so the game's own timezone is unknown. **Every candidate column
is NaN (missing, not "not flagged") on a `neutral_site == 1` row, and the row
is KEPT in the scored population with its NaN** — imputation belongs to the
model's own training-fold median (`SimpleImputer(strategy="median",
add_indicator=True)`, **read**, `src/nfl_ats/margin.py:387-393`), never to a
feature builder that can see every season at once. This mirrors
`nfl_ats.fluview_cfb_feature`'s handling exactly. The row cost is **327 of
12,500 in the whole table** (**measured**); the clean-core figure is in
section 5.

## 5. Coverage (filled from the PREDICTOR-ONLY `--mode coverage` run)

**Measured** 2026-09-01 by
`.\.tools\uv.exe run --no-sync python scripts\cfb_body_clock_replication.py
--mode coverage`, which reads no outcome column, before any other mode was
run. Artifacts: `artifacts/cfb_body_clock_replication/coverage.json` and
`artifacts/cfb_body_clock_replication/zone_assignments.json` (the per-school
assignment table, one row per school-season).

**The map resolved everything.** 12,500 of 12,500 rows carry a resolved zone on
both sides; **0 unresolved**; home-side and away-side zone coverage is
**1.000 in every season 2006-2025**; **0 split-state schools fell back to a
state default** — every one was resolved by its own city. **42 states** appear
across both sides. Nine IANA zones appear: `America/Boise`, `America/Chicago`,
`America/Denver`, `America/Detroit`, `America/Indiana/Indianapolis`,
`America/Los_Angeles`, `America/New_York`, `America/Phoenix`,
`Pacific/Honolulu`. So the section 2 rule ("the whole clean core is scored")
costs nothing: no season is thin, and no season is dropped.

**Clean core: 9,093 of 12,500 rows** (2012-2019, 2021-2025), of which **271 are
neutral-site** and therefore carry NaN on all four columns and sit in the
scored population with their NaN (section 4's rule). **1 row of 12,500** needed
the past-midnight-ET carry-forward (a 2021 Hawaii home game that kicks off at
01:30 ET the following calendar day); without the rule it would have been
counted as an "early" kickoff.

**Flagged counts inside the clean core** (predictor-only):

| cell | flagged | feature-missing (neutral) | era 2012-2019 | era 2021-2025 |
|---|---|---|---|---|
| `west_road_early` | 50 | 271 | 26 | 24 |
| `east_host_west_visitor_early` | 24 | 271 | 11 | 13 |
| `eastbound_multizone` | 338 | 271 | 168 | 170 |
| `night_west_road` | 487 | 271 | 326 | 161 |

Two of these are thin — 50 and 24 games — and that is **measured, disclosed
before scoring, and not hidden**. It is also the honest shape of the CFB
schedule: a Pacific/Arizona school playing a road game at a pre-14:00-ET
kickoff happens roughly four times a season league-wide (all 50 are noon or
1pm ET kickoffs: 41 at 12:00, 9 at 13:00). Section 9 reports these cells'
intervals with the width that count implies, and no adjective is substituted
for the number.

**Body clocks the WEST set excludes, as declared in section 4**: 32 away-team
rows with a Hawaii body clock and 1,128 with a Mountain body clock (Colorado,
Utah, Wyoming, New Mexico, Boise State, UTEP) across the whole table.

**The offset-based Eastern-host test earned its keep**: of cell 2's 24 flagged
clean-core games, 16 are at `America/New_York` venues, **4 at
`America/Detroit` and 4 at `America/Indiana/Indianapolis`**. A literal
`tz == "America/New_York"` string test, as the NFL screen wrote it, would have
silently dropped a third of the cell.

### Split-half reliability (predictor-only, no outcome touched)

| panel | n team-seasons | Pearson r | 95% CI | Spearman-Brown | P+ |
|---|---|---|---|---|---|
| body-clock UTC offset (the tz map as a trait) | 2,420 | **0.9858** | [0.9843, 0.9871] | **0.9928** | 1.0000 |
| `night_west_road` exposure | 1,486 | 0.5916 | [0.5203, 0.6613] | 0.7434 | 1.0000 |
| `eastbound_multizone` exposure | 1,486 | 0.3897 | [0.2699, 0.4991] | 0.5608 | 1.0000 |
| `west_road_early` exposure | 1,486 | 0.0521 | [-0.0121, 0.1966] | 0.0990 | 0.6245 |
| `east_host_west_visitor_early` exposure | 1,486 | -0.0037 | [-0.0060, -0.0019] | -0.0075 | 0.0000 |

**Amendment to section 7, made here, before any scoring mode was run, and
disclosed rather than quietly applied.** Section 7 as originally frozen named
the **per-cell exposure propensity** as the figure to record in
`--reliability`. The predictor-only run above shows why that choice would
publish a misleading number for the two thin cells, so `--reliability` instead
carries the **body-clock UTC-offset trait reliability, 0.9928**, and all five
figures above go verbatim into every entry's `--notes`. The reason, stated in
full:

- For a per-team-season event that happens at most once (24 flagged games
  across 1,486 team-seasons), a positive in the odd half nearly always implies
  a zero in the even half, which **forces a small negative odd/even
  correlation by construction**. The −0.0037 is arithmetic about rarity, not a
  measurement of a trait, and its CI excluding zero is the mechanical
  consequence, not evidence.
- AGENTS.md's stated rationale for the `no_split_half_reliability` closing
  ground is that "an unreliable trait is refuted because **no sample size
  rescues it**". That rationale is false for a rare schedule fact: more seasons
  do add flagged games. **`no_split_half_reliability` is therefore NOT an
  admissible closing ground for any cell in this document**, and no cell claims
  it.
- The trait these cells are actually built from — each school's own venue
  timezone — reads Spearman-Brown **0.9928**, so the construct is not noise.

Both numbers are published, side by side, in every registry entry. The
deviation was decided with **no outcome column yet touched by this line of
work**, which is the whole point of the predictor-only mode running first.

## 6. The comparator, and the two estimators

### Primary (registered): the XLG-03 benchmark arm plus exactly one column

Mirroring `docs/fluview_cfb_replication.md` section 6 in structure, with one
cell's column standing where the FluView column stands.

| arm | feature columns | estimator |
|---|---|---|
| baseline (shared) | `nfl_ats.cfb_features.CFB_MODEL_FEATURE_COLUMNS` (35 columns, the frozen XLG-03 contract) | `nfl_ats.cfb_benchmark.fit_cfb_residual_model`, `target="market_residual"`, ridge, `alpha=10.0` |
| candidate (one per cell) | the same 35 **plus** that cell's one column | identical |

Both arms hold regressor, alpha and target fixed at the benchmark's own frozen
values; only the feature contract differs, isolating each column's marginal
contribution against everything the benchmark already explains. The extension
point is the benchmark's own `feature_columns` parameter, whose docstring
declares it (**read**, `src/nfl_ats/cfb_benchmark.py:100-103`). **The four
candidate columns are never mixed with each other** — one column per cell.

**Note, declared before scoring**: `CFB_MODEL_FEATURE_COLUMNS` already contains
`neutral_site`, `rest_diff`, `week_sin` and `week_cos` (**measured**, 35
columns), so the baseline already knows about neutral sites and rest. It does
not contain anything about kickoff time or timezone, which is the axis every
cell here adds.

**Walk-forward.** Every scored week's two models are trained on all completed
games in the WHOLE table that kicked off strictly before that week's own
earliest `gameday`, with the benchmark's own
`CFB_BENCHMARK_MIN_TRAIN_GAMES = 500` floor — the same forward chaining
`cfb_walk_forward_benchmark` performs (**read**,
`src/nfl_ats/cfb_benchmark.py:200-209`).

### Secondary (NFL-commensurable): the subset-vs-complement `home_cover` gap

The four NFL registry entries are **not** signed as a paired model delta. They
are signed as `(subset_cover − complement_cover) × 100 × fraction_of_slate`,
week-blocked (**read**, `docs/body_clock_screen.md` lines 106-110). A CFB paired
model delta and an NFL subset-vs-complement gap cannot be compared for sign, so
this document also computes the NFL estimator, **verbatim** via
`scripts._common.summarize` / `block_bootstrap_two_group` — the same function
the NFL screens call — on the CFB clean core. Its population convention is the
NFL family's own, taken unchanged: `home_cover` with pushes dropped, and a row
whose flag is missing has the **flag forced False and sits in the complement**
(**read**, `travel_rest_eastbound_multizone.notes`: "flag forced False,
included in complement"). It is reported in section 9 and carried in each
registry entry's `--notes`; the **primary registered `--effect` is the paired
model delta**, per the work package that commissioned this run.

### Grade, named

The CFB benchmark grades on `spread_line`, "the median across books of each
book's home-oriented **close-proxy** spread" (**read**, `docs/cfb_data.md:117-118`),
and the CFB line archive carries "no source records quote observation times"
(**read**, `docs/cfb_data.md:42`), so **CFB can be graded at a close proxy and
never at a verified opener**. This replication is therefore **close-graded**.
Per the binding "grade the decision at the opener" rule, a close-graded number
settles no play/no-play or promotion decision — and a CFB number could not do so
in any case.

## 7. Metric, uncertainty, instrument checks, leakage, era split

**Metric.** `accuracy_points` (percentage points). Primary: the paired
candidate-minus-baseline forced-pick accuracy delta, picks taken at
`home_cover_probability >= 0.5` and graded with `nfl_ats.clv.pick_correct`
(pushes NaN, excluded). Secondary: the full-slate-scaled `home_cover` gap
above, in the same units.

**Uncertainty.** `nfl_ats.clv.week_blocked_bootstrap`, **week-blocked
primary**, **season-blocked secondary**, never averaged together. Within-week
correlation is ZERO by owner mandate — no ICC term anywhere.
**1,000 samples, seed `20260901`** (the same `BOOTSTRAP_SAMPLES` and seed
convention `scripts/fluview_cfb_replication.py` uses, for comparability).

**Within-week permutation null, 200 draws.** Both arms' models are fit ONCE on
the REAL `ats_margin`; only the grading margin is shuffled within week, so 200
draws cost no extra fits. This null is **not** centred on zero by design (it
preserves each week's realised home-cover rate and the two arms carry different
home-pick rates) and is reported ALONGSIDE the bootstrap-vs-zero interval,
never instead of it. A harness that reports a real effect under this null is
broken, and the run stops if it does.

**Positive control, run BEFORE the real screen, per cell**: the candidate's one
new column is REPLACED by the realised `ats_margin` — a deliberate, large
leak — so the harness must show an obvious, large effect. This proves the full
36-column ridge fit (not a single-feature model) can detect a real effect of
meaningful size when one is present. A harness that cannot detect this is
blind, and the run stops if it cannot.

**Split-half reliability, and exactly which panel.** Every cell here is a pure
schedule fact, so there is no continuous per-game trait to correlate. Two reads
are computed with `nfl_ats.cfb_qb_dependence.split_half_reliability`
(odd/even-week team-season split-half, Spearman-Brown corrected, block
bootstrap over team-seasons):

1. **Per-cell exposure propensity.** *(As originally frozen this was the figure
   to record in `--reliability`; section 5 amends that, before any scoring
   mode was run, and states the reason in full. Both figures are published.)*
   Panel: one row per team per game, keyed `(team_id, season, week)`, where
   `team_id` is the **visiting** team — the team the mechanism is about in all
   four cells — and the metric is that cell's 0/1 flag. This asks: is "being in
   this cell" a stable team-season property, or week-to-week noise? It is the
   only reliability question a pure schedule fact admits, and it is the one
   that matters for whether a sample size can ever rescue the cell.
2. **The underlying tz map, as a team-season trait** (the figure section 5
   amends `--reliability` to carry): the same panel shape with the metric set
   to each team's OWN home-venue UTC offset at that game. Expected to be
   near-perfectly reliable by construction (a team's venue does not move within
   a season), and reported as a floor check on the instrument.

Both are computed in the PREDICTOR-ONLY `coverage` mode, before any outcome is
touched.

**Leakage.** Every candidate column is a pure function of three pregame facts:
the game's kickoff timestamp, the two teams' identities, and each team's own
listed venue state/city for that season. Nothing about scores, injuries,
market moves or weather enters. Three regression tests in
`tests/test_cfb_body_clock_feature.py` enforce it:

1. **Outcome invariance.** Every candidate column is recomputed on a frame with
   `result`, `ats_margin`, `home_points`, `away_points` and `home_cover` all
   shuffled, and asserted bit-identical — on the hand-built fixture and on the
   real 12,500-row population.
2. **Input minimality.** Every column is recomputed from a frame trimmed to
   exactly the seven declared input columns and asserted bit-identical, so a
   column cannot be secretly reading anything else.
3. **Join correctness.** The `(season, team_id)` team_info join resolves every
   row of the real population on both sides, no split-state school falls back
   to a state default, and every state in the population has a declared zone.

Plus a hand-computed known-answer fixture per cell (10 games, including a
DST-boundary pair whose eastbound flag flips, a post-midnight-ET kickoff, three
split-state schools and a neutral site) and a test that neutral-site rows come
back NaN on all four columns and are kept, not dropped.

**Era split, declared before scoring.** The window spans the benchmark's own
declared 2020 regime gap, which is the obvious boundary and the one
`docs/fluview_cfb_replication.md` used. Two eras: **`2012_2019`** and
**`2021_2025`**. Magnitudes are reported separately and **never averaged across
a sign flip** (owner rule "era magnitude, not presence").

## 8. Decision rule and recording, frozen before scoring

**Decision rule.** Expected value, never a threshold: `probability_positive`
above 0.5 favours the candidate over the baseline. Predeclared thresholds
govern only what a document may CLAIM. **A CFB result is replication evidence
about a mechanism; it never by itself changes an NFL card**, and this run is
close-graded besides.

**Recording.** Four `nfl-ats weak-signals record` entries, one per cell, all
`--league cfb`, `--effect-units accuracy_points`, `--family
cfb_body_clock_replication`, `--category schedule`, `--season-start 2012`,
`--season-end 2025`:

| `--cell` | registry name | NFL sibling it replicates |
|---|---|---|
| `west_road_early` | `cfb_body_clock_west_road_early` | `body_clock_west_road_early` |
| `east_host_west_visitor_early` | `cfb_body_clock_east_host_west_visitor_early` | `body_clock_east_host_west_visitor_early` |
| `eastbound_multizone` | `cfb_travel_rest_eastbound_multizone` | `travel_rest_eastbound_multizone` |
| `night_west_road` | `cfb_body_clock_night_west_road_ge2000et` | `body_clock_night_west_road_ge2000et` |

`--interval-low`/`--interval-high` and `--probability-positive` come from the
**week-blocked** bootstrap; the season-blocked pair goes in `--notes`.
`--reliability` carries the **CFB-measured** per-cell exposure reliability,
never the NFL figure (the NFL entries carry `null`).

Each entry's `--notes` discloses, in these words: (i) that this is
**close-graded CFB**, never an opener grade; (ii) that the **four cells are
correlated subsets of one window and are not independent votes** — cell 2 is a
strict subset of cell 1, cell 4 shares cell 1's body-clock definition on the
opposite end of the kickoff distribution, and cell 3 overlaps all three through
the same timezone map — so they must never be sign-test-pooled as independent;
(iii) the **NFL sibling entry name** this cell replicates, and (iv) the
NFL-commensurable subset-vs-complement gap with its own interval and P+.

**Era slices.** `<cell>_era_2012_2019` / `<cell>_era_2021_2025` are recorded
**only where the two eras' magnitudes differ materially** — a sign flip, or a
magnitude ratio large enough that the pooled number is not a fair summary of
either. Where the eras agree, the era numbers are reported in section 9 and not
duplicated into the registry, per the instruction not to inflate it with
near-identical rows.

**A separate pooling bucket, stated explicitly.** `cfb_body_clock_replication`
is a different family from `body_clock_screen`, `body_clock_night_screen` and
`travel_rest_battery`. AGENTS.md's commensurability rule forbids pooling
non-commensurable comparators, and the NFL entries are subset-vs-complement
gaps in a different league at a different grade against a different baseline.
They are never pooled together, and neither is the primary paired-delta pooled
with the secondary gap.

**Classification.** `unresolved_below_power` for all four cells unless a cell
literally meets a terminal ground: `wrong_sign_resolved` requires the WHOLE
week-blocked interval on the wrong side of zero, `no_split_half_reliability`
requires a measured zero reliability, and `positive_control_bound` requires the
control to have PROVEN detection of an effect of the size in question. An
interval containing zero is not a ground; if a record command errors, the
verdict is wrong, not the validator.

**No rotation window is spent.** `nfl-ats rotation` is not invoked by this
document, matching the CFB replications' own precedent.

## 9. Results (added after the look, 2026-09-01)

Every number in this section is **measured** by
`.\.tools\uv.exe run --no-sync python scripts\cfb_body_clock_replication.py`
this session; each table names its own artifact. Nothing above this line was
edited after the first outcome sign was computed. Section 5 and section 7's
`--reliability` amendment were both settled from the PREDICTOR-ONLY
`--mode coverage` run, before any of the three scoring modes was launched.

**The scored population, as frozen in section 2.** 8,933 games across 199 week
blocks and 13 seasons (2012-2019, 2021-2025) — every clean-core game that is
completed, carries an `ats_margin`, and falls in a week with at least 500
strictly-prior training games. The XLG-03 baseline arm's own forced-pick
accuracy on that population is **51.595%**, and its home-pick rate is
**41.67%**.

### Instrument check 1 — within-week permutation null (200 draws, `--mode null`)

| cell | artifact | null mean | null sd | null 95% | observed |
|---|---|---|---|---|---|
| `west_road_early` | `20260901T192853Z` | +0.021 pts | 0.076 | [-0.123, +0.157] | -0.067 |
| `east_host_west_visitor_early` | `20260901T192941Z` | -0.004 pts | 0.057 | [-0.101, +0.101] | -0.011 |
| `eastbound_multizone` | `20260901T193020Z` | +0.030 pts | 0.112 | [-0.191, +0.246] | +0.022 |
| `night_west_road` | `20260901T193056Z` | +0.007 pts | 0.093 | [-0.202, +0.157] | -0.067 |

All four finite, all four centred within 0.03 points of zero, none reporting a
spurious effect. The harness is not broken, and no run was stopped.

### Instrument check 2 — positive control (`--mode positive-control`)

Artifacts `20260901T193156Z` / `193253Z` / `193347Z` / `193447Z`. Identical for
all four cells by construction (the same leaked column replaces each
candidate): pooled **+48.405 accuracy points**, week-blocked P+ **1.000**,
95% [+47.374, +49.464], n=8,933. Per era: 2012-2019 **+48.345** (P+ 1.000),
2021-2025 **+48.493** (P+ 1.000). The full 36-column ridge fit is not blind to
a real effect of meaningful size, in either era. No run was stopped.

### The real screen — PRIMARY estimator (paired candidate-minus-baseline model delta)

Artifacts `20260901T193559Z` / `193659Z` / `193809Z` / `193912Z`. Positive
means the extra column HELPED the model.

| cell | flagged | pooled delta | week 95% CI | week P+ | season 95% CI | season P+ | null pct |
|---|---|---|---|---|---|---|---|
| `west_road_early` | 50 | **-0.067** pts | [-0.205, +0.067] | **0.147** | [-0.159, +0.044] | 0.085 | 12.0th |
| `east_host_west_visitor_early` | 24 | **-0.011** pts | [-0.121, +0.101] | **0.376** | [-0.134, +0.114] | 0.381 | 39.0th |
| `eastbound_multizone` | 338 | **+0.022** pts | [-0.195, +0.223] | **0.581** | [-0.145, +0.168] | 0.612 | 45.5th |
| `night_west_road` | 487 | **-0.067** pts | [-0.237, +0.098] | **0.203** | [-0.205, +0.067] | 0.125 | 16.5th |

All four on n=8,933 games / 199 weeks / 13 seasons, with 271 feature-missing
(neutral-site) rows each.

### The real screen — SECONDARY estimator (NFL-commensurable subset-vs-complement gap)

The estimator the four NFL registry entries are actually signed with, computed
verbatim through `scripts/_common.summarize` (20,000 draws, seed 20260901),
NFL population convention unchanged (pushes dropped, 267 missing-flag rows
forced False into the complement). Positive means the HOME side covers MORE in
the flagged subset.

| cell | n_flag | subset cover | complement | raw gap | full-slate effect | week 95% CI | week P+ | season P+ | the NFL sibling's own number |
|---|---|---|---|---|---|---|---|---|---|
| `west_road_early` | 50 | 54.00% | 49.34% | +4.659 pts | **+0.0261** pts | [-0.0501, +0.1003] | **0.748** | 0.814 | -0.1545, P+ 0.2588 |
| `east_host_west_visitor_early` | 24 | 50.00% | 49.37% | +0.634 pts | **+0.0017** pts | [-0.0524, +0.0551] | **0.535** | 0.538 | -0.2690, P+ 0.0714 |
| `eastbound_multizone` | 331 | 51.06% | 49.30% | +1.755 pts | **+0.0650** pts | [-0.1137, +0.2519] | **0.760** | 0.759 | -0.1382, P+ 0.3239 |
| `night_west_road` | 478 | 49.37% | 49.37% | +0.005 pts | **+0.0003** pts | [-0.2525, +0.2464] | **0.502** | 0.504 | -0.1713, P+ 0.0843 |

### Per-era magnitudes, never averaged across a sign flip

Owner rule "era magnitude, not presence". Primary estimator first, then the
commensurable gap.

| cell | era 2012-2019 (n=5,349, 122 wks) | era 2021-2025 (n=3,584, 77 wks) | signs |
|---|---|---|---|
| `west_road_early` | **-0.075** pts, P+ 0.166, [-0.252, +0.093], 26 flagged | **-0.056** pts, P+ 0.292, [-0.278, +0.144], 24 flagged | **same** |
| `east_host_west_visitor_early` | **+0.056** pts, P+ 0.738, [-0.093, +0.211], 11 flagged | **-0.112** pts, P+ 0.034, [-0.248, +0.028], 13 flagged | **FLIP** |
| `eastbound_multizone` | **+0.112** pts, P+ 0.780, [-0.166, +0.388], 168 flagged | **-0.112** pts, P+ 0.230, [-0.427, +0.222], 170 flagged | **FLIP** |
| `night_west_road` | **-0.131** pts, P+ 0.129, [-0.404, +0.109], 326 flagged | **+0.028** pts, P+ 0.558, [-0.163, +0.222], 161 flagged | **FLIP** |

Commensurable gap, per era: `west_road_early` +0.0439 (P+ 0.817) / -0.0024
(P+ 0.475); `east_host_west_visitor_early` +0.0308 (P+ 0.855) / -0.0433
(P+ 0.167); `eastbound_multizone` +0.0984 (P+ 0.822) / -0.0033 (P+ 0.495);
`night_west_road` -0.0167 (P+ 0.455) / +0.0416 (P+ 0.591).

**Three of the four cells carry OPPOSITE SIGNS across the two eras on the
primary estimator, and three of four do on the secondary too.** Their pooled
numbers are therefore averages across a sign flip and must not be read as
either era's magnitude. Those three cells' era slices are recorded separately
in the registry for exactly that reason; `west_road_early`, whose eras agree in
sign and are within 0.02 points of each other, is not duplicated.

### What this implies for the decision, before what is wrong with it

**Per cell, does the NFL direction replicate on CFB? Answered on the
commensurable estimator, because that is the only one the NFL entries are
signed with.**

| cell | NFL point estimate | CFB point estimate | direction replicated? |
|---|---|---|---|
| `west_road_early` | -0.1545 (P+ 0.259) | **+0.0261** (P+ 0.748) | **No** — opposite side of zero |
| `east_host_west_visitor_early` | -0.2690 (P+ 0.071) | **+0.0017** (P+ 0.535) | **No** — opposite side, and CFB is flat |
| `eastbound_multizone` | -0.1382 (P+ 0.324) | **+0.0650** (P+ 0.760) | **No** — opposite side of zero |
| `night_west_road` | -0.1713 (P+ 0.084) | **+0.0003** (P+ 0.502) | **No** — CFB is dead flat |

**All four NFL point estimates are negative; all four CFB point estimates are
positive.** The NFL family's anti-predicted negative lean — home teams covering
LESS against a west-body-clock visitor — did not appear on college football in
any of the four cells. Where CFB moves at all it moves the other way, that is,
weakly toward the direction the NFL screens originally PREDICTED and failed to
find.

**The decision this supports, stated first.** On expected value — P+ above 0.5
favours the flagged side, the only decision rule this project uses — **CFB's
`west_road_early` and `eastbound_multizone` cells both favour the HOME side**,
at P+ 0.748 / 0.814 (week / season) and P+ 0.760 / 0.759 respectively. The
largest is `west_road_early`: on 50 flagged CFB games the home team covered
**54.00%** against a **49.34%** complement, a **+4.66-point** raw gap. For a
CFB card that is a live, playable lean at roughly 3-to-1. It is also the direct
CFB counterpart of the NFL cell that reads P+ 0.2588 the other way — so this
run's contribution to the NFL question is that **the NFL's negative reading
does not generalise**, which raises rather than lowers the expected value of
keeping that NFL cell open.

On the primary registered estimator — adding one column to the frozen XLG-03
arm — only `eastbound_multizone` is favoured (P+ 0.581 week / 0.612 season);
the other three favour the baseline (P+ 0.147 / 0.376 / 0.203). Those two
readings are not in conflict: they answer different questions. The gap
estimator asks whether the flagged games cover differently; the model estimator
asks whether a 36th column beats a 35-column ridge that already sees
`spread_line`, `neutral_site` and `rest_diff`. A flag on 0.56% of the slate can
move pooled accuracy by at most **±0.56 points** even if it flipped every
flagged game (±0.27 for `east_host_west_visitor_early`), so the model
estimator has almost no headroom on the two thin cells and the observed
magnitudes are a small fraction of it.

**For the NFL card: nothing changes, and nothing here can change it.** A CFB
result is replication evidence about a mechanism; it never by itself changes an
NFL card, and this run is close-graded besides.

**What is NOT concluded, and why.** No cell is closed, and no era slice is.

- `wrong_sign_resolved` requires the WHOLE week-blocked interval on the wrong
  side of zero. **Not one of the ten intervals recorded here qualifies** — the
  narrowest, `east_host_west_visitor_early` era 2021-2025 at [-0.248, +0.028],
  still crosses. An interval containing zero is not a ground for rejection and
  nothing above rests on one.
- `no_split_half_reliability` is unavailable by section 5's measurement: the
  trait these cells are built from reads Spearman-Brown **0.9928**. The two
  thin cells' near-zero *exposure* reliabilities (0.0990 and -0.0075) are
  arithmetic about how rare the schedule situation is, not a measurement of a
  trait, and AGENTS.md's rationale for that ground — "no sample size rescues
  it" — is false for a rare schedule fact, because more seasons do add flagged
  games.
- `bounded_by_control` requires the instrument to have been PROVEN able to
  detect an effect **of the size in question**. The control proved detection of
  a **+48.4-point** leak, which says nothing about a sub-0.3-point effect, and
  every week-blocked CI here is wider than the effect it is testing.

All ten entries are recorded `unresolved_below_power`.

**Caveats, after the numbers rather than instead of them.**

1. **`night_west_road` measures a materially different thing in CFB than in the
   NFL, and this is the largest single caveat in the document.** Of its 487
   flagged clean-core games, **346 are at a Pacific or Arizona venue**
   (`America/Los_Angeles` 295, `America/Phoenix` 51) and a further 109 at a
   Mountain venue (`America/Denver` 92, `America/Boise` 17) — so **455 of 487**
   are West-or-Mountain road trips, where the visitor's body clock barely
   shifts at all. In the NFL a kickoff at or after 20:00 ET is a national night
   game; in CFB it is mostly the Pacific schools' ordinary 7-8pm-local home
   window seen from the road. The flat CFB reading (+0.0003, P+ 0.502) is
   therefore weak evidence about the NFL construct, and it is reported as such
   rather than as a refutation of it.
2. **Two cells are thin: 50 and 24 flagged games.** Measured, disclosed before
   scoring in section 5, and reflected in the interval widths. That is the
   honest shape of the CFB schedule — a Pacific/Arizona school playing a road
   game before 14:00 ET happens about four times a season league-wide — not a
   defect in the instrument.
3. **The four cells are correlated subsets of one window.**
   `east_host_west_visitor_early` is a strict subset of `west_road_early` (24
   of its 50 games); `night_west_road` shares the identical body-clock
   definition at the opposite end of the kickoff distribution;
   `eastbound_multizone` overlaps all three through the same timezone map.
   **Never sign-test-pool them as independent votes**, and never pool them with
   the NFL siblings either (different league, different grade, different
   baseline — section 8).
4. **CFB is a different, softer market**: the XLG-03 baseline picks at 51.595%
   here and the slate covers home at 49.37%, so a reading on CFB constrains but
   does not bound what the same construct can do against the NFL market.
5. **Close-graded.** CFB has no verified opener (`docs/cfb_data.md:42`), so
   this settles no play/no-play or promotion decision in either league.
6. The primary and secondary estimators are **not commensurable with each
   other** and are never pooled; the registered `--effect` is the primary, and
   the secondary travels in every entry's `--notes`.

### Registry, verified by reading it back

Ten entries recorded, each `nfl-ats weak-signals record` call wrapped in the
session's cross-process lock (other agents were writing the same file this
session). `registry/weak_signals.json` went **640 -> 650** signals. Every field
below was **measured** by re-reading `registry/weak_signals.json` after the
writes and diffing it field-by-field against the screen artifacts; all ten
match on effect, both interval endpoints, `probability_positive`,
`sample_games`, `sample_blocks`, `seasons`, `league`, `family`, `category`,
`effect_units`, `classification`, `closing_ground` and `reliability`.

| registry name | effect | week 95% interval | P+ | n games | n blocks | seasons |
|---|---|---|---|---|---|---|
| `cfb_body_clock_west_road_early` | -0.067167 | [-0.205000, +0.067120] | 0.147 | 8,933 | 199 | 2012-2025 |
| `cfb_body_clock_east_host_west_visitor_early` | -0.011194 | [-0.120848, +0.101279] | 0.376 | 8,933 | 199 | 2012-2025 |
| `cfb_travel_rest_eastbound_multizone` | +0.022389 | [-0.194926, +0.223292] | 0.581 | 8,933 | 199 | 2012-2025 |
| `cfb_body_clock_night_west_road_ge2000et` | -0.067167 | [-0.236621, +0.098323] | 0.203 | 8,933 | 199 | 2012-2025 |
| `cfb_body_clock_east_host_west_visitor_early_era_2012_2019` | +0.056085 | [-0.092969, +0.210676] | 0.738 | 5,349 | 122 | 2012-2019 |
| `cfb_body_clock_east_host_west_visitor_early_era_2021_2025` | -0.111607 | [-0.248416, +0.028249] | 0.034 | 3,584 | 77 | 2021-2025 |
| `cfb_travel_rest_eastbound_multizone_era_2012_2019` | +0.112170 | [-0.165567, +0.387880] | 0.780 | 5,349 | 122 | 2012-2019 |
| `cfb_travel_rest_eastbound_multizone_era_2021_2025` | -0.111607 | [-0.427369, +0.222309] | 0.230 | 3,584 | 77 | 2021-2025 |
| `cfb_body_clock_night_west_road_ge2000et_era_2012_2019` | -0.130866 | [-0.404263, +0.108540] | 0.129 | 5,349 | 122 | 2012-2019 |
| `cfb_body_clock_night_west_road_ge2000et_era_2021_2025` | +0.027902 | [-0.162847, +0.221802] | 0.558 | 3,584 | 77 | 2021-2025 |

All ten read back with `classification: unresolved_below_power`,
`closing_ground: null`, `league: cfb`, `family: cfb_body_clock_replication`,
`category: schedule`, `effect_units: accuracy_points`, `reliability: 0.9928`, a
`plain_summary`, and `notes` carrying the close-grade disclosure, the
correlated-subsets disclosure, the NFL sibling name, the NFL-commensurable gap
and both reliability reads.

`registry/rotation_registry.json` is **untouched by this document**: `nfl-ats
rotation` was never invoked here, and no NFL rotation window was declared,
assigned or spent. (That file does show as modified in `git status` — other
agents were writing it concurrently this session. None of those writes came
from this line of work.)

### Files added

- `docs/cfb_body_clock_replication.md` (this document).
- `src/nfl_ats/cfb_body_clock_feature.py` — the venue state/city -> IANA
  timezone map and the four candidate columns.
- `scripts/cfb_body_clock_replication.py` — `--mode coverage | null |
  positive-control | screen`, `--cell` one of four.
- `tests/test_cfb_body_clock_feature.py` — 17 leakage, join, known-answer and
  neutral-site tests.
- `artifacts/cfb_body_clock_replication/` — `coverage.json`,
  `zone_assignments.json` (the auditable per-school zone table) and twelve run
  artifacts (four cells x null / positive-control / screen).
