# Schedule flag battery: LEAD-21, LEAD-22, LEAD-40 (on-production opener confirmation)

Predeclared 2026-09-05, before any of the three candidates below was scored.
Written for lane C of the overnight fleet
(`scripts/schedule_flag_on_production.py`, `src/nfl_ats/schedule_flag_features.py`).

## Binding closing-grounds taxonomy (verbatim, restated per CLAUDE.md/AGENTS.md)

An interval or CI that contains zero is NEVER grounds to reject, fail, or
close an experiment. At this evaluator's ~2-point resolution, "contains
zero" is the EXPECTED outcome for a real small signal. Only two grounds
ever close a line of work: (1) refuted mechanism -- a RESOLVED wrong sign
(whole interval on the wrong side of zero) or zero split-half reliability;
(2) bounded by a positive control proven able to detect an effect that
size. Everything else is `unresolved_below_power`: record it with
`nfl-ats weak-signals record`, report `probability_positive`, never the
binary "contains zero". The registry code hard-rejects inadmissible
closures; if a record command errors, the verdict is wrong, not the
validator. Verdicts flow only through `nfl-ats weak-signals record` /
`nfl-ats rotation record`, never through prose. Decide on expected value:
`probability_positive` above 0.5 favours the candidate; thresholds govern
only what docs may claim.

## Shared design

All three candidates are screened **on top of PRODUCTION**: the estimator is
the exact `weak_stack` ridge (alpha 10) chain fitted on
`data/processed/game_features_weak_stack.parquet`, with exactly one new
column added per candidate (`profile_identity` in
`scripts/on_production_opener_confirmation.py`, reused unmodified, asserts
this at runtime). Grading is at the **opener** (Tuesday consensus), per
AGENTS.md's binding "grade the decision at the opener" rule; close-graded
reads are screens, not decisions, and are reported alongside as a secondary
read only. Each candidate is its **own rotation family**
(`post_ot_fatigue_on_production`, `mnf_road_short_week_on_production`,
`home_thursday_on_production`), each declared and window-assigned before any
outcome was scored (`nfl-ats rotation declare` / `assign`, opener grade,
2018-2025 mined-ledger acknowledged since the opener pool is entirely
2020-2025).

**Data source.** Every flag is computed ONLY from the newest
`data/raw/*/schedules.parquet` snapshot
(`nfl_ats.schedule_flag_features.default_schedule`, reusing
`nfl_ats.weak_stack_v3_features.latest_schedules_snapshot`'s
newest-snapshot convention) and merged onto the production feature table by
`game_id`. No PBP, no injury data, no market data enters any of the three
constructs.

**Leakage.** Every flag reads only: `game_id`, `season`, `gameday`,
`weekday`, `home_team`, `away_team`, and `overtime` -- and `overtime` only
from a team's own **strictly preceding, in-season** game, never from the
game the flag is attached to. None of the three reads `result`,
`home_score`, `away_score`, `spread_line`, or `ats_margin` at all, for any
game. Consequently shuffling or altering a game's own outcome (score,
margin, or its own `overtime` value) cannot change that game's own flag --
it can only change a LATER game's flag if that later game's team played
this one as its immediately preceding game, which is legitimate pregame-
known history, not leakage.
`tests/test_schedule_flag_on_production.py::test_flags_are_invariant_to_a_games_own_outcome`
pins this for all three constructs on a synthetic multi-week schedule.

**Comparator and metric.** Same estimator with vs. without the one column
(`weak_stack` vs. `weak_stack_post_ot` / `weak_stack_mnf_road` /
`weak_stack_home_thursday`), same population (the rotation-assigned window
plus all strictly-prior training seasons), same forced-pick accuracy metric
(`correct_at_open_probability_rule`, i.e. the production `home_cover_probability
>= 0.5` rule -- the sign rule is reported alongside), same week-blocked
bootstrap (20,000 resamples, `nfl_ats.clv.week_blocked_bootstrap`) and the
same 200-permutation within-week null as every other on-production
confirmation in this repo
(`scripts/on_production_opener_confirmation.py`).

**Controls.**
- `--mode null`: within-week permutation shuffle of `margin_vs_open` (200
  permutations); the harness is not blind if the observed delta sits far
  outside this null's spread when it should not.
- `--mode positive-control`: the candidate column is REPLACED by the
  REALIZED `ats_margin` (deliberately leaky, never promotable) -- this must
  read hugely positive, `probability_positive` near 1.0, or the harness
  itself cannot detect an effect of any size and every `screen` result
  downstream is uninterpretable.
- `--mode screen`: the single outcome look, run once per candidate.

**Reliability argument (shared).** All three constructs are pure,
deterministic functions of the published NFL schedule (kickoff day/time,
home/away designation, and whether a PAST game reached overtime) -- there is
no measurement instrument, no sampling, and no repeated-measurement error to
split in half. `no_split_half_reliability` is therefore **inadmissible** as
a closing ground for any of the three: a trait with zero measurement noise
cannot be "unreliable" in the sense that ground requires (AGENTS.md ties
that ground to a trait no sample size can rescue because it fails to
reproduce on a random re-measurement; a calendar fact reproduces by
construction, every time, with probability 1). This is the same
distinction `registry/weak_signals.json`'s `travel_rest_thursday_pure` /
`travel_rest_short_week_road` entries already draw for their own
flag-exposure "reliability" reads (both explicitly flagged NOT admissible
as a `no_split_half_reliability` closing ground). The only admissible
closing grounds that remain are a RESOLVED wrong sign (whole interval on
the wrong side of zero, both blockings) or a positive-control bound; absent
either, every result below is recorded `unresolved_below_power`.

**Decision rule.** Per AGENTS.md "a promotion bar is not a decision bar":
`probability_positive` above 0.5 favours playing the candidate; the interval
crossing zero does not veto anything. No candidate here is being proposed
for promotion into production on this single confirmation look regardless
of sign -- the rotation registry marks the window spent either way, and the
recording plan below is `unresolved_below_power` unless a resolved wrong
sign or a positive-control bound applies.

**Recording plan (all three).** `nfl-ats rotation record --name <family>
--artifact <results.json> --verdict unresolved --probability-positive <p>
--effect <d> --effect-units accuracy_points --interval-low/high ...
--sample-blocks <weeks> --notes "..."`, then `nfl-ats weak-signals record
--name <family> --family <family> --league nfl --season-start/--season-end
<assigned window> --effect ... --classification unresolved_below_power
--probability-positive ... --category schedule --classification-evidence
"..."`.

---

## Section 1 -- LEAD-21: post-overtime fatigue

**Mechanism.** A team whose immediately preceding game went to overtime
logs roughly 15-25 extra snaps on short recovery (`ROADMAP.md` LEAD-21):
plausible cumulative fatigue (more live snaps, less practice/recovery time
before the following kickoff) that the travel/rest battery
(`docs/travel_rest_battery.md`) never isolated -- that battery's 8 cells
cover distance, timezone, neutral site, return-trip hangover, and rest-day
thresholds, none of which reference `overtime`.

**Predeclared direction.** FADE the post-OT side: expect it to underperform
its market-implied margin in its next game.

**Encoding.** `post_ot_fatigue_flag`, one column, built in
`nfl_ats.schedule_flag_features.derive_post_ot_fatigue_features`:
- For each team, find its immediately preceding **in-season** game (sorted
  by `gameday` within `(team, season)`; the shift never crosses a season
  boundary -- an offseason gap is not the fatigue mechanism this section
  claims).
- `away_post_ot` = that game's `overtime == 1` for the AWAY team's
  preceding game; `home_post_ot` analogously for the HOME team.
- `post_ot_fatigue_flag = +1` if `away_post_ot` and not `home_post_ot`;
  `-1` if `home_post_ot` and not `away_post_ot`; `0` if both, neither, or
  either side has no in-season preceding game (a team's week-1 game
  certainly did not follow an overtime game within that season, so "no
  prior game" is encoded as the concrete fact **0**, not a missing value --
  unlike `nfl_ats.team_style_pace_production_feature`'s NaN-on-missing
  convention, where "no prior-season data" really is an unknown state, "no
  preceding game this season" is a known one).
- Sign chosen so a **positive** fitted coefficient on this column means
  "fading the post-OT side helped" (the model predicts `market_residual`
  from the home side's perspective, so `+1` pushes the prediction toward
  "home outperforms," consistent with fading a post-OT AWAY team; `-1`
  pushes it the other way, consistent with fading a post-OT HOME team).

**Comparator / metric / controls.** As stated in "Shared design" above:
`weak_stack` vs. `weak_stack_post_ot`, opener-graded forced-pick accuracy,
week-blocked bootstrap plus within-week permutation null, positive control =
candidate column replaced by realized `ats_margin`.

**Reliability argument.** As stated in "Shared design": `overtime` is a
published, deterministic schedule/box-score fact (verified 2026-09-05: in
the current `data/raw/20260824T115346Z/schedules.parquet` snapshot,
`overtime` is non-null for every one of the 4,630 played games and null for
every one of the 272 not-yet-played 2026 games -- zero played games have a
missing value). `no_split_half_reliability` is inadmissible.

**Decision rule / recording plan.** As stated in "Shared design."

---

## Section 2 -- LEAD-22: Monday-night-road short week

**Mechanism.** A team that plays on the road on Monday night and then plays
again the following Sunday gets only six days to travel home, recover, and
prepare, compounding travel fatigue with a short week (`ROADMAP.md`
LEAD-22). This is a narrower construct than the already-screened
`travel_rest_short_week_road` cell in the ENV-03/ENV-04 travel/rest battery
(**parent cell, cited per the ROADMAP row's instruction**:
`registry/weak_signals.json` key `travel_rest_short_week_road`, description
"away_rest <= 5 days (game-level, side-specific)", effect **+0.0441733912**
accuracy_points, week-blocked 95% **[-0.3116357012, +0.4041244469]**,
`probability_positive` **0.59325**, n=4,317 games / 294 week-blocks, seasons
2009-2025, classification `unresolved_below_power`). That cell thresholds
`away_rest` alone, regardless of which day the prior game was on; this
section isolates the specific MNF-road-then-Sunday compound the parent cell
never separated out.

**Predeclared direction.** FADE the short-week road side.

**Encoding.** `mnf_road_short_week_flag`, one column, built in
`nfl_ats.schedule_flag_features.derive_mnf_road_short_week_features`. A side
"qualifies" when ALL of the following hold:
1. Its immediately preceding **in-season** game (same within-season
   lookback as LEAD-21) was on a **Monday**.
2. It was the **away** (road) team in that preceding game.
3. THIS game's own `weekday` is **Sunday**.
4. The calendar gap between the preceding game's `gameday` and this game's
   `gameday` is exactly **6 days** (checked directly against the date
   arithmetic, not inferred from the weekday labels, so a data
   inconsistency cannot silently pass).

`mnf_road_short_week_flag = +1` if the AWAY team qualifies and the HOME team
does not; `-1` if the reverse; `0` if both qualify, neither does, or a side
has no in-season preceding game (again encoded as the concrete fact `0`,
same reasoning as LEAD-21: no preceding game this season means certainly no
recent Monday road trip). Sign convention matches LEAD-21: positive fitted
coefficient means "fading the short-week road side helped."

**Comparator / metric / controls.** As stated in "Shared design."

**Reliability argument.** `weekday`, `home_team`/`away_team` (site), and
`gameday` are the same class of published, deterministic schedule facts as
`overtime` above -- no measurement noise, so `no_split_half_reliability` is
inadmissible. (The parent `travel_rest_short_week_road` cell's OWN
`reliability` field in `registry/weak_signals.json` was measured on the
continuous `away_rest` trait it thresholds and separately marked NOT
admissible as a `no_split_half_reliability` ground, for a related but
distinct reason -- `away_rest` is a conserved-total quantity within a
team-season, which manufactures a spurious negative split-half correlation.
This section's flag is a plain calendar conjunction, not a thresholded
continuous trait, so that particular artifact does not even apply here; the
controlling argument is the general one above.)

**Decision rule / recording plan.** As stated in "Shared design."

---

## Section 3 -- LEAD-40: home-Thursday rest compound

**Mechanism.** In a Thursday game, the home team has no travel at all,
while the away team travels on a short week; both sides get less
preparation time than a normal Sunday game, but only one side also travels
(`ROADMAP.md` LEAD-40). This is a narrower, home/road-split construct than
the already-screened `travel_rest_thursday_pure` cell (**parent cell,
cited per the ROADMAP row's instruction**: `registry/weak_signals.json` key
`travel_rest_thursday_pure`, description "Thursday game, regardless of
venue/weather", effect **+0.1348767306** accuracy_points, week-blocked 95%
**[-0.2341593022, +0.5015489461]**, `probability_positive` **0.7592**,
n=4,317 games / 294 week-blocks, seasons 2009-2025, classification
`unresolved_below_power`). That cell already predicts a positive home edge
on any Thursday game; it has never been stacked on top of PRODUCTION.

**Predeclared direction.** BACK the home team in Thursday games.

**Encoding.** `home_thursday_flag`, one column, built in
`nfl_ats.schedule_flag_features.derive_home_thursday_features`: **1.0**
when this game's own `weekday == "Thursday"`, **0.0** otherwise. Unlike
LEAD-21/LEAD-22, this needs no in-season lookback at all -- it is a fact
about the CURRENT game only. Unsigned (not a home-minus-away difference)
because the construct is not a comparison between the two teams' own
conditions: on a Thursday game the home side never travels while the away
side always does, so "Thursday" already IS the home-favouring condition
(matching the parent `travel_rest_thursday_pure` cell's own plain boolean
shape, which predicted a positive `home_cover` edge directly from the same
flag). A missing `weekday` value would return NaN rather than a silent 0;
no such value exists in the current schedule snapshot (verified 2026-09-05,
see LEAD-21 section).

**Comparator / metric / controls.** As stated in "Shared design."

**Reliability argument.** `weekday` is the same class of published,
deterministic schedule fact as `overtime`/`gameday` above --
`no_split_half_reliability` is inadmissible. (Parent cell's own
flag-exposure "reliability" read in `registry/weak_signals.json` is
likewise explicitly marked not admissible as this closing ground, for the
same reason: a schedule quirk with no persistent team-level trait can still
move covers.)

**Decision rule / recording plan.** As stated in "Shared design."

---

## Measured results (2026-09-05)

All three candidates share the same rotation-assigned opener window
**[2020, 2021]** (456 paired non-push games, 35 weeks, 2 seasons), the same
estimator (`weak_stack` ridge alpha 10 vs. the one-column candidate profile),
and the same positive control: candidate column replaced by the realized
`ats_margin` reads **+44.298 accuracy points**, week- and season-blocked
`probability_positive` **1.000** both blockings, for all three -- the
harness is proven sensitive to an effect that size before any screen result
is read.

Effect/interval figures below are the **opener, production-rule** primary
read (week-blocked, 20,000 resamples); the sign-rule and close-graded reads
are in each artifact's `result` block but are not the decision quantity
(AGENTS.md: grade the decision at the opener). Season-blocked secondary
reads exist in each artifact but rest on only 2 season blocks (the window
size) and are not treated as informative at that block count.

| Candidate | Effect (accuracy pts) | Week-blocked 95% CI | P+ | n games / weeks | Flag rate (full schedule) |
|---|---|---|---|---|---|
| LEAD-21 post-OT fatigue | -0.219 | [-2.397, +1.739] | 0.38375 | 456 / 35 | 493/4,902 (10.1%) |
| LEAD-22 MNF-road short week | 0.000 | [-0.665, +0.649] | 0.349 | 456 / 35 | 287/4,902 (5.9%) |
| LEAD-40 home-Thursday | -0.439 | [-1.709, +0.680] | 0.1821 | 456 / 35 | 296/4,902 (6.0%) |

Every interval crosses zero. Per the taxonomy above, that is the EXPECTED
shape for a real small signal at this window size and is not grounds to
close any of the three. No admissible closing ground applies to any
candidate: no interval sits entirely on the wrong side of zero
(`wrong_sign_resolved` is unavailable), and `no_split_half_reliability` is
inadmissible by construction for all three (deterministic schedule facts,
argued per-section above). All three are recorded
`unresolved_below_power`:

- `post_ot_fatigue_on_production` -- rotation window spent, `registry/weak_signals.json` entry (registry count 694 after recording). Artifact `artifacts/schedule_flag_on_production/post_ot/20260905T022034Z/results.json`.
- `mnf_road_short_week_on_production` -- rotation window spent, weak-signal registry count 695. Artifact `artifacts/schedule_flag_on_production/mnf_road/20260905T022823Z/results.json`.
- `home_thursday_on_production` -- rotation window spent, weak-signal registry count 696. Artifact `artifacts/schedule_flag_on_production/home_thursday/20260905T023529Z/results.json`.

**LEAD-40 is the one flag whose within-family read flips sign relative to
its screened parent cell** (parent `travel_rest_thursday_pure` read P+
0.7592 favoring home; this on-production stack reads P+ 0.1821, leaning the
other way) -- consistent with the standing project lesson "composition is
not the signal" (a component positive alone can go negative once stacked on
the chain actually played), not a contradiction requiring resolution.

Per AGENTS.md's promotion-bar/decision-bar distinction: none of the three is
proposed for production promotion on this single confirmation look. The
decision recorded here is the rotation window being spent and the finding
being kept (not discarded) for future pooling, exactly as the taxonomy
requires.

---

# Wave 2: venue and market-context flags (LEAD-39/41/42/35)

Predeclared 2026-09-05, before any of the four candidates below was scored.
Written for lane F of the overnight fleet, reusing lane C's harness verbatim
(`scripts/schedule_flag_on_production.py`'s `CANDIDATES` map extended with
four new entries; `src/nfl_ats/schedule_flag_features.py` extended with four
new `derive_*`/`attach_*` pairs). The binding closing-grounds taxonomy stated
verbatim at the top of this document applies unchanged; it is not repeated
here.

## Shared design (Wave 2)

Same estimator, same population discipline, same grade, same controls as
Wave 1 ("Shared design" above): `weak_stack` vs. `weak_stack` plus exactly
one new column, opener-graded forced-pick accuracy as the decision metric,
week-blocked bootstrap (20,000 resamples) plus a 200-permutation within-week
null, and a positive control (candidate column replaced by realized
`ats_margin`) that must read hugely positive before any screen result is
trusted. Each candidate is its own rotation family (`new_stadium_home_on_production`,
`dome_shootout_favorite_on_production`, `low_total_div_home_dog_on_production`,
`sept_heat_home_on_production`), declared and window-assigned before any
outcome was scored.

**Data sources.** All four still read only local, already-captured data --
no network fetch:
- `data/raw/*/schedules.parquet` (newest snapshot, via the same
  `nfl_ats.weak_stack_v3_features.latest_schedules_snapshot` convention Wave
  1 uses): `stadium_id`, `season`, `week`, `game_type`, `home_team`,
  `away_team`, `roof`, `div_game`, `gametime`.
- **The Tuesday-OPENER consensus spread and total** (LEAD-41/LEAD-42 only),
  read from `nfl_ats.clv.build_pairing_table`'s `HISTORICAL_CAPTURE_KIND`
  decision-labeled archive under `data/market/raw`, filtered to the
  `"tue_open"` decision label -- the SAME opener store
  `scripts/on_production_opener_confirmation.py` already grades every
  on-production candidate against. This is a hard requirement from the
  fleet task: the nflverse schedule's own `spread_line`/`total_line` columns
  are the CLOSE (post-lock information) and must never be used to build a
  pregame-decision flag. New loader `nfl_ats.schedule_flag_features.default_opener_lines`,
  additive, reuses `build_pairing_table` unmodified.
- No flag reads `result`, `home_score`, `away_score`, `ats_margin`, or the
  schedule's own `spread_line`/`total_line` at all.

**Sign convention.** This repo's spread sign is uniform and verified
2026-09-05 by reading two independent statements of it: `src/nfl_ats/open_benchmark.py`
line 353 (`"spread_sign": "positive_spread_line_means_home_favorite"`) and
`src/nfl_ats/market_data.py`'s `parse_odds_api_response` (`standardized_home_line
= -home_point`, negating the Odds-API home team's own point spread so a
home favorite of e.g. -7 in gambler notation becomes stored as **+7**). The
opener store's `home_spread` (and this module's `tue_open_home_spread`)
therefore follows the identical convention as the nflverse schedule's own
`spread_line`: **positive = HOME favored** by that many points, negative =
AWAY favored, zero = a pick'em with no favorite.

**Reliability argument (shared).** All four constructs are, like every Wave
1 flag, deterministic functions of published pregame facts (venue
assignment, season number, roof status, divisional-schedule membership,
kickoff time, and -- for LEAD-41/LEAD-42 -- the Tuesday-morning market
consensus itself, which is a real, observed, point-in-time quantity, not a
repeated psychological/behavioral measurement). None of the four has a
"measurement noise" component that a split-half read could meaningfully
characterize, so `no_split_half_reliability` is **inadmissible** as a
closing ground for any of the four, for the same reason Wave 1 gives.

**Declared approximation: `roof`.** LEAD-41 and the roof-conditional half of
LEAD-35 read the schedule's own recorded `roof` value (dome/closed/open/
outdoors). For a fixed-dome venue this is invariant and fully pregame-known
(MIN01, VEG00, LAX01, the old Louisiana/Mercedes-Benz/Caesars Superdome).
For a RETRACTABLE-roof venue (Atlanta's Mercedes-Benz Stadium, Houston's
NRG/Reliant Stadium, and others outside this construct's home-team set) the
open/closed decision is typically announced close to kickoff -- later than
the Tuesday opener this flag is otherwise graded against. This is the same
class of declared, not fixed, approximation `nfl_ats.clv.opener_pick_evaluation`
already accepts for every sibling on-production candidate (its own
docstring: "only `spread_line` is swapped to the opener; every other
feature (including `total_line`) is close-era"). Stated here rather than
silently assumed.

**Controls and decision rule.** Identical to Wave 1: `--mode null` (within-week
permutation null), `--mode positive-control` (candidate column replaced by
realized `ats_margin`, must read `probability_positive` near 1.0), `--mode
screen` (the single outcome look). `probability_positive` above 0.5 favours
the candidate; an interval crossing zero is never grounds to close a family
(AGENTS.md, restated verbatim at the top of this document).

**Recording plan (all four).** Identical command shape to Wave 1: `nfl-ats
rotation record --name <family> --artifact <screen results.json> --verdict
unresolved --probability-positive <p> ...`, then `nfl-ats weak-signals
record --name <family> --family <family> --league nfl --season-start/--season-end
<assigned window> --classification unresolved_below_power ...` unless a
RESOLVED wrong sign (whole interval on the wrong side of zero) or a
positive-control bound applies.

---

## Section 4 -- LEAD-39: New-stadium honeymoon (seasons 1-2)

**Mechanism.** A new venue's sightlines, turf, background noise, and
locker-room/tunnel logistics take a full season or two for VISITING teams to
solve; the home team, by contrast, has practiced and played there from day
one (`ROADMAP.md` LEAD-39).

**Predeclared direction.** BACK the home team in a venue's first two REG
seasons of NFL use.

**Encoding.** `new_stadium_home_flag`, one column, built in
`nfl_ats.schedule_flag_features.derive_new_stadium_home_features`: **1.0**
when the game's own `stadium_id` is one of a FROZEN set of six venues AND
`season` is one of that venue's own first two REG seasons of use; **0.0**
otherwise. No in-season lookback, no comparison between the two teams'
conditions (unsigned, matching LEAD-40's shape: this is a single-side
effect, not a differential).

**The frozen venue list, measured 2026-09-05** against
`data/raw/20260824T115346Z/schedules.parquet` (script:
`C:\Users\Ryan\AppData\Local\Temp\claude\F--Repos-nfl-py3\b5bd0c70-497a-41e7-81b4-c281feb3ebe8\scratchpad\laneF_stadium_inspect.py`,
grouping every REG home game by `stadium_id` and taking each one's minimum
season):

| `stadium_id` | First REG season | Honeymoon seasons | Venue | Home team(s) |
|---|---|---|---|---|
| NYC01 | 2010 | 2010-2011 | MetLife Stadium | NYG, NYJ |
| SFO01 | 2014 | 2014-2015 | Levi's Stadium | SF |
| MIN01 | 2016 | 2016-2017 | U.S. Bank Stadium | MIN |
| ATL97 | 2017 | 2017-2018 | Mercedes-Benz Stadium | ATL |
| LAX01 | 2020 | 2020-2021 | SoFi Stadium | LA, LAC |
| VEG00 | 2020 | 2020-2021 | Allegiant Stadium | LV |

This reproduces the fleet task's own worked examples exactly. Every
stadium_id with a first REG season >= 2010 that is NOT in this table was
measured and explicitly excluded, for one of two stated reasons:
- **Neutral/international one-off sites** (never a team's repeat home
  base, so the claimed "visitors need a season to solve it" mechanism does
  not even apply to a single exhibition-style game): `LON00`/`LON01`/`LON02`,
  `MEX00`, `GER00`, `FRA00`, `SAO00` (task-given prefixes) plus the same-
  class 2026 one-off sites measured in this snapshot -- `MAD01` (Bernabeu,
  ATL, 1 game), `MEL00` (Melbourne Cricket Ground, LA, 1 game), `MUN01` (FC
  Bayern Munich Stadium, DET, 1 game), `PAR00` (Stade de France, NO, 1
  game), `RIO00` (Maracana Stadium, DAL, 1 game).
- **Temporary construction-displacement homes**, superseded by the SAME
  team's own later PERMANENT venue already in the table above: `MIN98` (TCF
  Bank Stadium; Vikings' 2010 storm-displacement game plus 2014-2015 home
  while U.S. Bank Stadium/MIN01 was under construction), `LAX99` (Los
  Angeles Memorial Coliseum; Rams' 2016-2019 home while SoFi/LAX01 was under
  construction), `LAX97` (StubHub Center; Chargers' 2017-2019 home while
  SoFi/LAX01 was under construction).

**Comparator / metric / controls / reliability / decision rule.** As stated
in "Shared design (Wave 2)" above. `stadium_id` and `season` carry zero
measurement noise; `no_split_half_reliability` is inadmissible.

---

## Section 5 -- LEAD-41: Dome-shootout favorite archetype

**Mechanism.** A controlled dome/closed-roof environment plus a
high-scoring OPENER total plus a near-pick'em opener spread describes a
high-possession track-meet expected to reduce game-flow variance, which is
hypothesized to favor the more talented team -- the favorite (`ROADMAP.md`
LEAD-41).

**Predeclared direction.** BACK the favorite in an archetype game.

**Encoding.** `dome_shootout_favorite_flag`, one column, built in
`nfl_ats.schedule_flag_features.derive_dome_shootout_favorite_features`. A
game qualifies as archetype when ALL of:
1. `roof` is `"dome"` or `"closed"`.
2. The Tuesday-opener total (`tue_open_total_line`, from the opener store,
   NEVER the schedule's own closing `total_line`) is **>= 49**.
3. The Tuesday-opener home spread's absolute value (`tue_open_home_spread`,
   from the opener store) is **<= 3**.

`dome_shootout_favorite_flag = +1` if the game qualifies AND the HOME team
is the favorite (`tue_open_home_spread > 0`); `-1` if it qualifies AND the
AWAY team is the favorite (`tue_open_home_spread < 0`); `0` if it does not
qualify, OR it qualifies but the opener spread is an exact pick'em (no
favorite to back), OR the opener store lacks a resolved total or spread for
this game (a missing value NEVER silently satisfies either threshold).

**Opener-total coverage, measured 2026-09-05**
(`C:\Users\Ryan\AppData\Local\Temp\claude\F--Repos-nfl-py3\b5bd0c70-497a-41e7-81b4-c281feb3ebe8\scratchpad\laneF_opener_total_coverage.py`,
`nfl_ats.clv.build_pairing_table` against `data/market/raw` restricted to
`decision_label == "tue_open"`, joined to REG rows of the same schedule
snapshot): 1,537 REG games carry a `tue_open` opener consensus at all
(seasons 2020-2025, matching AGENTS.md's own cited "1,537 paired games"
figure for the opener-graded pool), of which **1,526/1,537 (99.3%)** carry a
resolved opener total; the remaining **11 games, all in the 2020 season**,
get `dome_shootout_favorite_flag = 0` by construction (never treated as
satisfying the `>= 49` threshold) and are reported in the flag-summary
block of every `--mode screen` artifact.

**Comparator / metric / controls.** As stated in "Shared design (Wave 2)."

**Declared approximation.** `roof`, as stated in "Shared design (Wave 2)."

**Reliability argument.** `roof` is a published venue/game-state fact;
`tue_open_total_line`/`tue_open_home_spread` are observed point-in-time
market quotes, not a repeated psychological trait -- `no_split_half_reliability`
is inadmissible for the same reason Wave 1 gives for a deterministic
calendar fact.

**Decision rule / recording plan.** As stated in "Shared design (Wave 2)."

---

## Section 6 -- LEAD-42: Low-total divisional home dog

**Mechanism.** Division rivals know each other intimately; familiarity
suppresses scoring, and a short, hard-fought, low-total divisional game is
hypothesized to keep the underdog inside the number more often than a
generic home-dog spot (`ROADMAP.md` LEAD-42).

**Predeclared direction.** Take the HOME DOG (BACK the home underdog) in
divisional games with a low opener total.

**Encoding.** `low_total_div_home_dog_flag`, one column, built in
`nfl_ats.schedule_flag_features.derive_low_total_div_home_dog_features`:
**1.0** when ALL of:
1. `div_game == 1` (schedule's own divisional-matchup flag).
2. The Tuesday-opener total (`tue_open_total_line`) is **<= 42**.
3. The home team is the underdog at the Tuesday opener
   (`tue_open_home_spread < 0`).

**0.0** otherwise -- including a game missing an opener total or spread in
the store (a missing total is explicitly NEVER encoded as satisfying `<= 42`;
see the fleet task's own instruction to encode 0, not the threshold-passing
value, for a missing opener input). Unsigned, matching LEAD-40's shape: this
construct backs one specific side (the home dog), not a differential
between the two teams' conditions.

**Comparator / metric / controls / declared approximation (none needed --
`div_game` has no roof-style approximation caveat) / reliability.** As
stated in "Shared design (Wave 2)" (opener-total coverage is the identical
1,526/1,537 measured for LEAD-41 above, since both read the same opener
store). `div_game` is a schedule-published fact with zero measurement
noise; `no_split_half_reliability` is inadmissible.

**Decision rule / recording plan.** As stated in "Shared design (Wave 2)."

---

## Section 7 -- LEAD-35: September heat-humidity home edge

**Mechanism.** Heat- and humidity-acclimated home teams hosting a
cold-climate visitor early in the season, at the hottest part of the day,
are hypothesized to have a physiological conditioning edge the visitor has
not had time to adapt to (`ROADMAP.md` LEAD-35). This is the declared
mirror image of the already-screened cold-visitor temp-gap cell (per the
ROADMAP row's own instruction, never pooled with it -- this is a separate,
kickoff-time-and-roster-conditioned construct, not a re-measurement of the
same trait).

**Predeclared direction.** BACK the heat-acclimated home team.

**Encoding.** `sept_heat_home_flag`, one column, built in
`nfl_ats.schedule_flag_features.derive_sept_heat_home_features`: **1.0**
when ALL of:
1. `game_type == "REG"` and `week <= 3`.
2. The HOME team is heat-acclimated: **MIA, TB, or JAX unconditionally**
   (measured 2026-09-05: all three play 100% of their home games with
   `roof == "outdoors"`, so no roof condition changes anything for them), OR
   **HOU, NO, or ATL only when this game's own `roof` is `"outdoors"` or
   `"open"`** (task-given qualifier; measured 2026-09-05: HOU's roof is
   `"closed"` for 124/139 and `"open"` for 15/139 recorded games, NO is a
   fixed dome for 145/147 and `"outdoors"` for 2/147, ATL is `"closed"` for
   55, `"dome"` for 63 [Georgia Dome era, always enclosed], `"open"` for 18,
   `"outdoors"` for 2, of 138 -- the conditional path is real and populated
   for HOU/ATL, and rare-but-nonzero for NO).
3. The AWAY team is on the frozen cold-climate list: BUF, NE, NYJ, NYG, GB,
   CHI, MIN, DET, CLE, PIT, CIN, DEN, SEA, KC, PHI, BAL, WAS (task-given,
   verbatim).
4. This game's own kickoff, converted from the schedule's Eastern-Time
   `gametime` (the established convention this repo's `scripts/body_clock_screen.py`
   already relies on: raw `"%H:%M"` compared directly against ET-labeled
   thresholds) to the HOME team's own LOCAL clock, falls in the **1 PM
   local hour** (13:00-13:59 local, left-inclusive/right-exclusive).

**Local-time conversion, stated explicitly as a design choice.** The task
says "1 PM local kickoff," not "1 PM ET" -- a deliberate, meaningful
qualifier this predeclaration honors literally rather than reading as NFL
broadcast shorthand for "the early window." All six candidate home teams
sit in either America/New_York (MIA, TB, JAX, ATL; 0 hours behind ET) or
America/Chicago (HOU, NO; always exactly 1 hour behind ET, since both zones
observe US daylight saving on the identical calendar dates every year, so
the gap never varies by season) -- verified 2026-09-05 against
`registry/stadium_coordinates.json`'s own `tz` entries for Hard Rock Stadium
(MIA), Raymond James Stadium (TB), TIAA Bank/EverBank Stadium (JAX),
Mercedes-Benz Stadium (ATL) = `America/New_York`, and NRG/Reliant Stadium
(HOU), Caesars/Mercedes-Benz/Louisiana Superdome (NO) = `America/Chicago`.
Local kickoff minutes = ET kickoff minutes minus the team's fixed offset;
"1 PM local" is tested as the half-open interval [13:00, 14:00) local.

**Measured consequence of the literal reading (stated before scoring, not
after).** The NFL's standard national early-window broadcast slot is
"1:00 PM ET" for every team regardless of home time zone; for the two
Central-zone candidates (HOU, NO) this is **noon local**, not 1 PM local --
a genuine 1 PM-CT (2 PM ET) EARLY-window kickoff essentially never occurs
under normal NFL scheduling (measured: zero HOU or NO home games in the
full 2009-2026 schedule snapshot show `gametime == "14:00"`). Under the
literal "1 PM local" reading predeclared above, HOU and NO are therefore
expected to contribute close to zero qualifying games, and the flag's
effective population is dominated by MIA/TB/JAX/ATL's `gametime == "13:00"`
games. This is reported as a measured fact in the results below, not
smoothed over by silently relaxing the definition after seeing it.

**Comparator / metric / controls / declared approximation (roof, for the
HOU/NO/ATL conditional path only) / reliability.** As stated in "Shared
design (Wave 2)." `home_team`/`away_team`/`week`/`gametime`/`roof` are all
published pregame schedule facts with zero measurement noise;
`no_split_half_reliability` is inadmissible.

**Decision rule / recording plan.** As stated in "Shared design (Wave 2)."

---

## Measured results (Wave 2, 2026-09-05)

All four candidates share the same rotation-assigned opener window
**[2020, 2021]** (456 paired non-push games, 35 weeks, 2 seasons) and the
same estimator (`weak_stack` ridge alpha 10 vs. the one-column candidate
profile). The positive control (candidate column replaced by the realized
`ats_margin`) reads **identically for all four** -- +44.298 accuracy
points, week- and season-blocked `probability_positive` **1.000** both
blockings -- which is the expected, mechanical consequence of how
`--mode positive-control` works (the leaked column is the SAME real
`ats_margin` values regardless of which named column it replaces, so the
fitted ridge model and its predictions are numerically identical across
candidates): the harness is proven sensitive to an effect that size before
any of the four screen results below is read.

Effect/interval figures below are the **opener, production-rule** primary
read (week-blocked, 20,000 resamples); the sign-rule and close-graded reads
are in each artifact's `result` block but are not the decision quantity
(AGENTS.md: grade the decision at the opener). Season-blocked secondary
reads exist in each artifact but rest on only 2 season blocks and are not
treated as informative at that block count, matching the Wave 1 convention.

| Candidate | Effect (accuracy pts) | Week-blocked 95% CI | P+ (production rule) | n games / weeks | Flag rate (full schedule) |
|---|---|---|---|---|---|
| LEAD-39 new-stadium honeymoon | 0.000 | [-0.6565, +0.6550] | 0.34815 | 456 / 35 | 138/4,902 (2.8%) |
| LEAD-41 dome-shootout favorite | 0.000 | [-1.7204, +1.7699] | 0.4384 | 456 / 35 | 66/4,902 (1.3%) |
| LEAD-42 low-total div. home dog | +0.4386 | [-0.6608, +1.7544] | 0.68955 | 456 / 35 | 57/4,902 (1.2%) |
| LEAD-35 Sept. heat home edge | -0.8772 | [-2.4229, +0.4396] | 0.0841 | 456 / 35 | 34/4,902 (0.7%) |

Every interval crosses zero. Per the taxonomy above, that is the EXPECTED
shape for a real small signal at this window size and is not grounds to
close any of the four. No admissible closing ground applies to any
candidate: no interval sits entirely on the wrong side of zero
(`wrong_sign_resolved` is unavailable for any of them, including LEAD-35
whose point estimate is negative -- its upper bound is still positive), and
`no_split_half_reliability` is inadmissible by construction for all four
(pregame-known schedule/venue facts or observed point-in-time market
quotes, argued per-section above). All four are recorded
`unresolved_below_power`:

- `new_stadium_home_on_production` -- rotation window spent, weak-signal
  registry count 716. Only 3/456 forced picks flip under the production
  rule despite 53/456 games flagged in-window (24 in 2020, 29 in 2021).
  Artifact `artifacts/schedule_flag_on_production/new_stadium/20260905T031856Z/results.json`.
- `dome_shootout_favorite_on_production` -- rotation window spent,
  weak-signal registry count 717. Opener-total coverage: 1,526/1,537
  `tue_open` games resolved (99.3%); 11 missing 2020 games encoded 0.
  15/456 forced picks flip; flag is roughly balanced between home-favorite
  (29 games) and away-favorite (37 games) archetypes full-schedule.
  Artifact `artifacts/schedule_flag_on_production/dome_shootout/20260905T032312Z/results.json`.
- `low_total_div_home_dog_on_production` -- rotation window spent,
  weak-signal registry count 718. Same 1,526/1,537 opener-total coverage as
  LEAD-41 (identical store). `probability_positive` 0.68955 favours the
  candidate per AGENTS.md's promotion-bar/decision-bar distinction even
  though the interval still crosses zero (barely, on the low end: -0.6608).
  8/456 forced picks flip. Artifact
  `artifacts/schedule_flag_on_production/low_total_div_dog/20260905T032318Z/results.json`.
- `sept_heat_home_on_production` -- rotation window spent, weak-signal
  registry count 719. **The one Wave 2 candidate whose two forced-pick
  rules disagree in sign**: the production rule reads P+ 0.0841 (leaning
  against the candidate) while the plain sign rule on the IDENTICAL
  population reads P+ 0.7774 (leaning for it) -- reported as a measured
  fact, not resolved into a single number, and not itself grounds for any
  closing ground (AGENTS.md: grade the decision at the opener using the
  production rule; the sign-rule read is a secondary diagnostic). The
  predeclared "measured consequence" of reading "1 PM" as literal home-team
  local time held: only 34/4,902 full-schedule games ever flag (10/456
  forced picks flip in-window), dominated by MIA/TB/JAX/ATL's 13:00-ET
  games; HOU/NO's contribution is effectively zero, exactly as predeclared
  before scoring. Artifact
  `artifacts/schedule_flag_on_production/sept_heat/20260905T032315Z/results.json`.

Per AGENTS.md's promotion-bar/decision-bar distinction: none of the four is
proposed for production promotion on this single confirmation look
(`low_total_div_home_dog_on_production`'s `probability_positive` above 0.5
is noted, not acted on, since a single confirmation look is a screen, not a
promotion decision). The decision recorded here is each rotation window
being spent and each finding being kept (not discarded) for future pooling,
exactly as the taxonomy requires.

---

# Wave 3: public-claim leads on production (LEAD-57 leads)

Predeclared 2026-09-05, before any of the four candidates below was scored.
Written for lane H of the overnight fleet, reusing lane C's harness verbatim
(`scripts/schedule_flag_on_production.py`'s `CANDIDATES` map extended with
four new entries; `src/nfl_ats/schedule_flag_features.py` extended with four
new `derive_*`/`attach_*` pairs). The binding closing-grounds taxonomy stated
verbatim at the top of this document applies unchanged; it is not repeated
here.

`docs/public_claim_battery.md` (LEAD-57, registry family
`public_claim_battery`) close-graded twelve public-handicapper claims on the
full 2009-2025 population as a lead-generation screen (per
`docs/rotation_registry.md` rule 8, no rotation window spent). Four leaned
toward the claim strongly enough to earn this project's standing next step, a
marginal test ON TOP OF PRODUCTION graded at the OPENER (the "composition is
not the signal" rule: a subset positive alone can be worth nothing once
stacked on the played chain):

| Lane G cell | Effect (pts, close-graded, full population) | P+ |
|---|---|---|
| `public_claim_road_fav_big_fade` (FADE road favourites of 7+) | +0.2170 | 0.9736 |
| `public_claim_week1_dog` (BACK Week 1 underdogs) | +0.1076 | 0.9219 |
| `public_claim_ats_streak_regress` (BACK a 3+ game ATS losing streak) | +0.2440 | 0.9193 |
| `public_claim_division_dog` (BACK divisional underdogs) | +0.3692 | 0.9119 |

## Shared design (Wave 3)

Same estimator, same population discipline, same grade, same controls as
Wave 1/2 ("Shared design" sections above): `weak_stack` vs. `weak_stack` plus
exactly one new column, opener-graded forced-pick accuracy as the decision
metric, week-blocked bootstrap (20,000 resamples) plus a 200-permutation
within-week null, and a positive control (candidate column replaced by
realized `ats_margin`) that must read hugely positive before any screen
result is trusted. Each candidate is its own rotation family
(`road_fav_big_fade_on_production`, `division_dog_on_production`,
`week1_dog_on_production`, `ats_streak_regress_on_production`), declared and
window-assigned before any outcome was scored.

**Data sources.** All four read only local, already-captured data -- no
network fetch:

- `data/raw/*/schedules.parquet` (newest snapshot, the same
  `latest_schedules_snapshot` convention every prior wave uses):
  `game_id`, `season`, `week`, `game_type`, `home_team`, `away_team`,
  `div_game`, `result`, `spread_line`.
- **The Tuesday-OPENER consensus spread** (`road_fav_big_fade`,
  `division_dog`, `week1_dog` only), read from
  `nfl_ats.schedule_flag_features.default_opener_lines` -- the SAME
  `tue_open`-labelled opener store Wave 2's LEAD-41/LEAD-42 already read and
  `scripts/on_production_opener_confirmation.py` grades every on-production
  candidate against. This is a hard requirement carried over from Wave 2 and
  restated by this fleet task: the nflverse schedule's own `spread_line` is
  the CLOSE and must never be used to build these three pregame-decision
  flags.
- `ats_streak_regress` reads only the schedule's own `result`/`spread_line`
  (the CLOSE line) to build each team's OWN prior-games cover history. This
  is a **frozen, predeclared design choice, stated before any scoring**: the
  streak a team carries INTO this game is itself measured the same way
  `docs/public_claim_battery.md`'s own `ats_streak_len` column measures it
  (close-graded, matching "the archive"), even though the flag this streak
  produces is then evaluated on top of PRODUCTION at the opener. The two
  gradings are not in tension -- a team's ATS record through its own past,
  completed games is a fixed historical fact by the time any future game's
  opener is posted, exactly like LEAD-21's post-OT flag reading a past
  game's `overtime` outcome. Nothing here reads the CURRENT game's own
  `result` or `spread_line`.
- No flag reads `home_score`, `away_score`, `ats_margin`, or `home_cover` for
  the game it is attached to.

**Sign convention.** `column positive favours the HOME side`, the same
uniform convention as every prior wave (`src/nfl_ats/open_benchmark.py:353`
`"spread_sign": "positive_spread_line_means_home_favorite"`;
`tue_open_home_spread` follows the identical convention, read positive).

- **`road_fav_big_fade_flag`**: `docs/public_claim_battery.md`'s
  `public_claim_road_fav_big_fade` tested ONE side only -- fading a ROAD
  favourite of 7+ (`is_home == False AND team_spread >= 7`, sign -1: back
  the home team) -- and said nothing about a home favourite of 7+. This
  fleet task's own worked sign-convention example ("road favourite of 7+ at
  the opener -> +1 for home; home favourite of 7+ -> -1") instructs a
  **symmetric extension** for the on-production stack: `+1` when the AWAY
  team is favoured by >=7 points at the Tuesday opener (fade the road
  favourite, back home -- the cell lane G actually measured), `-1` when the
  HOME team is favoured by >=7 points at the opener (the mirror case, NOT
  separately tested by lane G's battery -- disclosed here as an instructed
  extension of the tested claim, not a silent assumption), `0` otherwise or
  when the opener store lacks a resolved spread for this game (a missing
  value never silently satisfies either threshold).
- **`division_dog_flag`** and **`week1_dog_flag`** share one shape (the
  task's own "week1 dog likewise"): `+1` when the game is eligible (a
  REG-season divisional game / a REG-season Week 1 game) AND the home team
  is the underdog at the Tuesday opener; `-1` when eligible AND the away
  team is the underdog; `0` when ineligible, an exact opener pick'em, or the
  opener store lacks a resolved spread. This is a faithful, symmetric
  restatement of lane G's own claim, which was already tested from BOTH
  flagged teams' own perspectives on the long table (a home dog and an away
  dog in the same population, backed the same way) -- no extension needed.
  **Eligibility is restricted to `game_type == "REG"`, stated explicitly**:
  measured 2026-09-05 against the current schedule snapshot, postseason
  rows can carry `div_game == 1` (26 of 371 postseason games) and can share
  `week` values with REG season (postseason `week` ranges 18-22, overlapping
  REG's own week 18), so an unrestricted eligibility mask would silently
  admit games lane G's REG-only population never tested. Both flags are
  forced to `0` for any non-REG game.
- **`ats_streak_regress_flag`**: `+1` if the HOME team enters this game on a
  losing ATS streak of 3+ AND the AWAY team does not; `-1` if the reverse;
  `0` if both do, neither does, or the game is not REG season (streak
  history and the flag itself are both built from REG-season games only,
  matching `docs/public_claim_battery.md`'s own population -- a stated
  design choice, not an oversight). **Streak construction, frozen per the
  task's explicit instruction**: for each team-season, ordered by gameday,
  a running count of consecutive `home_cover`-derived cover LOSSES
  immediately preceding this game; a cover resets the count to 0; a push
  (`ats_margin == 0`, `home_cover` NaN) neither extends nor resets it (the
  push game is skipped from the team's own sequence entirely, matching
  `docs/public_claim_battery.md`'s own `ats_streak_len` convention exactly);
  the count resets to 0 at every season boundary (a team's first REG game of
  a season enters with streak 0, not NaN -- a genuine fact, not missing
  information, matching every prior wave's precedent for "no in-season
  preceding game").

**Reliability argument (shared).** `road_fav_big_fade`, `division_dog`, and
`week1_dog` are, like Wave 2's LEAD-41/LEAD-42, built from a published
pregame schedule fact (divisional-schedule membership, week number) plus an
observed, point-in-time Tuesday market quote -- neither has a "measurement
noise" component a split-half read could characterize.
`ats_streak_regress`'s own streak length is a deterministic function of
already-completed, published game results and the published closing line
(schedule/box-score facts, not a repeated psychological measurement).
`no_split_half_reliability` is therefore **inadmissible** as a closing
ground for any of the four, for the same reason every prior wave gives.

**Declared approximation: none beyond Wave 2's own opener-coverage
caveat.** `road_fav_big_fade`/`division_dog`/`week1_dog` inherit the
identical 1,526/1,537 (99.3%) `tue_open` opener-spread coverage measured for
Wave 2's LEAD-41/LEAD-42 (same store, same seasons); the remaining games get
each flag forced to `0` (never silently treated as a qualifying value).

**Controls and decision rule.** Identical to Wave 1/2: `--mode null`
(within-week permutation null), `--mode positive-control` (candidate column
replaced by realized `ats_margin`, must read `probability_positive` near
1.0), `--mode screen` (the single outcome look). `probability_positive`
above 0.5 favours the candidate; an interval crossing zero is never grounds
to close a family (AGENTS.md, restated verbatim at the top of this
document).

**Recording plan (all four).** Identical command shape to Wave 1/2: `nfl-ats
rotation record --name <family> --artifact <screen results.json> --verdict
unresolved --probability-positive <p> ...`, then `nfl-ats weak-signals
record --name <family> --family <family> --league nfl --season-start/--season-end
<assigned window> --classification unresolved_below_power ...` unless a
RESOLVED wrong sign (whole interval on the wrong side of zero) or a
positive-control bound applies.

---

## Section 8 -- `road_fav_big_fade_on_production`

**Mechanism.** Public handicapper folklore: a heavy road favourite (7+
points) travels, faces a hostile crowd, and is hypothesized to play down to
its underdog opponent more often than the market's own number implies
(`docs/public_claim_battery.md` claim 5, cited from `docs/roadmap.md`
LEAD-57).

**Predeclared direction.** FADE the big road favourite (BACK home when away
is favoured by 7+); the symmetric mirror (FADE a big home favourite) is the
task-instructed extension stated in "Shared design (Wave 3)" above.

**Encoding / comparator / metric / controls / reliability / decision
rule.** As stated in "Shared design (Wave 3)" above.
`ROAD_FAV_BIG_FADE_SPREAD_MIN_ABS = 7.0` (inclusive, matching lane G's own
`team_spread >= 7`).

---

## Section 9 -- `division_dog_on_production`

**Mechanism.** Division rivals face each other twice a year and know each
other intimately; the market is hypothesized to over-favour the presumptive
stronger team in a divisional matchup, leaving value on the divisional
underdog (`docs/public_claim_battery.md` claim 4).

**Predeclared direction.** BACK the divisional underdog (home or away,
whichever side the opener spread names as the dog).

**Encoding / comparator / metric / controls / reliability / decision
rule.** As stated in "Shared design (Wave 3)" above.

---

## Section 10 -- `week1_dog_on_production`

**Mechanism.** Offseason narratives (hype, coaching changes, roster
turnover) are hypothesized to distort Week 1 lines before any in-season
form exists, leaving value on Week 1 underdogs specifically
(`docs/public_claim_battery.md` claim 9).

**Predeclared direction.** BACK the Week 1 underdog (home or away).

**Encoding / comparator / metric / controls / reliability / decision
rule.** As stated in "Shared design (Wave 3)" above (identical shape to
Section 9, restricted to `week == 1`).

---

## Section 11 -- `ats_streak_regress_on_production`

**Mechanism.** A team that has failed to cover the spread in each of its
last three or more games is hypothesized to be systematically
under-priced by a market slow to correct for a run of bad or unlucky
results, and due to regress toward covering (`docs/public_claim_battery.md`
claim 12).

**Predeclared direction.** BACK the team on a 3+ game ATS losing streak.

**Encoding / comparator / metric / controls / reliability / decision
rule.** As stated in "Shared design (Wave 3)" above.
`ATS_STREAK_REGRESS_MIN_STREAK = 3` (matching lane G's own
`ats_streak_len >= 3`).

---

## Measured results (Wave 3, 2026-09-05)

All four candidates share the same rotation-assigned opener window
**[2020, 2021]** (456 paired non-push games, 35 weeks, 2 seasons) and the
same estimator (`weak_stack` ridge alpha 10 vs. the one-column candidate
profile). The positive control (candidate column replaced by the realized
`ats_margin`) reads **identically for all four** -- +44.298 accuracy points,
week- and season-blocked `probability_positive` **1.000** both blockings --
the same mechanical consequence Wave 2 already documents (the leaked column
is the SAME real `ats_margin` regardless of which named column it replaces):
the harness is proven sensitive to an effect that size before any of the
four screen results below is read.

**Incident, disclosed per the coordinator's own measurement, corrected
after checking the provenance record**: the coordinator flagged a process
(PID 31488) alive ~85 minutes with only ~77 seconds of CPU consumed as this
candidate's stalled run, initially reported as `--mode positive-control`.
**Measured, `registry/experiments/schedule-flag-on-production/20260905T040850Z.json`**:
`positive-control` actually completed normally as part of the original
sequential loop (`elapsed_seconds: 183.9`, started `2026-09-05T04:05:46Z`,
result identical to every other candidate's positive control:
+44.298 accuracy points, P+ 1.000 both blockings). The stalled process was
therefore the SAME loop's subsequent `--mode screen` call -- consistent
with the coordinator's own corroborating observation that "no screen run
for that candidate exists" at the time it was flagged. The stalled process
was killed; the standalone `attach_ats_streak_regress_features` step was
timed directly (**measured**: 64 ms end-to-end on the full local schedule +
production feature table, 4,902 games) to confirm the flag builder itself
was never the bottleneck. Both `--mode positive-control` (redundant with
the already-complete 040850Z run, kept as a second confirmation) and
`--mode screen` were then re-run in an isolated foreground process,
completing in the same ~3 minutes every other Wave 1/2/3 candidate takes.
The stall is attributed to resource contention from this session's own
concurrent `pytest -n auto` suites and background polling loops, not a
defect in the flag builder; nothing in `nfl_ats.schedule_flag_features` was
changed as a result.

Effect/interval figures below are the **opener, production-rule** primary
read (week-blocked, 20,000 resamples); the sign-rule and close-graded reads
are in each artifact's `result` block but are not the decision quantity
(AGENTS.md: grade the decision at the opener). Season-blocked secondary
reads exist in each artifact but rest on only 2 season blocks and are not
treated as informative at that block count, matching Wave 1/2's convention.

| Candidate | Effect (accuracy pts) | Week-blocked 95% CI | P+ (production rule) | n games / weeks | Flag rate (full schedule) |
|---|---|---|---|---|---|
| `road_fav_big_fade_on_production` | -1.5351 | [-2.8634, -0.2188] | 0.00785 | 456 / 35 | 403/4,902 (8.2%) |
| `division_dog_on_production` | -0.6579 | [-1.7058, +0.2212] | 0.0459 | 456 / 35 | 547/4,902 (11.2%) |
| `week1_dog_on_production` | -0.6579 | [-1.7505, 0.0000] | 0.0000 | 456 / 35 | 92/4,902 (1.9%) |
| `ats_streak_regress_on_production` | -0.6579 | [-2.2026, +0.8850] | 0.1609 | 456 / 35 | 758/4,902 (15.5%) |

**`road_fav_big_fade_on_production` is the FIRST candidate across Waves
1-3 whose primary interval sits entirely on the wrong side of zero at both
week- and season-blocking.** Recorded `closed_negative` (rotation) /
`refuted_mechanism` (weak-signals), `--closing-ground wrong_sign_resolved`,
per the binding taxonomy ("a RESOLVED wrong sign -- whole interval on the
wrong side of zero"). This is worth reading carefully rather than as a
reflexive rejection, because it is genuinely narrow:

- **Decomposition (measured this session, not itself recorded to either
  registry)**: isolating either half of the symmetric column alone --
  away-big-favorite ONLY (the direction lane G actually tested, `sign +1`,
  49 games in-window) or home-big-favorite ONLY (the task-instructed mirror
  extension lane G never tested, `sign -1`, 99 games in-window) -- each
  individually CROSSES zero (away-only: week CI [-1.5086, +0.8734],
  P+ 0.30375; home-only: week CI [-3.0239, +0.2299], P+ 0.04455). Neither
  sub-component alone resolves. The resolved negative is a property of the
  POOLED symmetric column exactly as predeclared and as this fleet task's
  own sign convention instructed -- an instance of AGENTS.md's own stated
  principle that "pooling sub-signals into a result that excludes zero is a
  legitimate, transformative finding... any real effect can be decomposed
  until its pieces do," realized here in the negative direction rather than
  the positive one the doc's own worked example describes.
- **The sign-rule read on the identical population disagrees**: it leans
  the OPPOSITE way and crosses zero (P+ 0.94185, week CI [-0.0022, +0.0291]).
  Per AGENTS.md ("grade the decision at the opener" using the production
  rule; the sign rule is a secondary diagnostic, the same convention Wave 2's
  LEAD-35 section already establishes for this repo), the production-rule
  read is what decides; the sign-rule disagreement is disclosed, not
  averaged away or used to override the closure.
- **This closes ONLY the on-production family as declared here.** Lane G's
  original, one-sided, close-graded `public_claim_road_fav_big_fade` cell
  (fading only the road favorite, never claiming anything about a home
  favorite) remains its own separate `unresolved_below_power` entry in
  `registry/weak_signals.json` and is explicitly NOT reclassified by this
  result -- the two measure different, if related, constructs (one-sided
  close-graded population screen vs. symmetric opener-graded on-production
  stack).

`division_dog_on_production`, `week1_dog_on_production`, and
`ats_streak_regress_on_production` all read `unresolved_below_power` --
every interval crosses zero (or, for `week1_dog`, touches exactly 0.0000 at
the upper bound, which is a lean, not a resolution: AGENTS.md's
`wrong_sign_resolved` ground requires the WHOLE interval strictly below
zero). No admissible closing ground applies to any of the three:
`no_split_half_reliability` is inadmissible by construction for all three
(deterministic schedule/div_game facts, or an observed Tuesday-opener market
quote, or a deterministic function of already-completed close-graded
results -- none has measurement noise to split in half, argued per-section
above).

- `division_dog_on_production` -- rotation window spent, weak-signal
  registry entry recorded (registry count 727 after recording). Artifact
  `artifacts/schedule_flag_on_production/division_dog/20260905T035808Z/results.json`.
- `week1_dog_on_production` -- rotation window spent, weak-signal registry
  count 728. Only 3/456 forced picks flip under the production rule despite
  38 positive/54 negative flags across the full schedule -- a thin
  in-window population plausibly explains the boundary-exact CI bound.
  Artifact `artifacts/schedule_flag_on_production/week1_dog/20260905T040525Z/results.json`.
- `ats_streak_regress_on_production` -- rotation window spent, weak-signal
  registry count 729. Artifact
  `artifacts/schedule_flag_on_production/ats_streak_regress/20260905T041624Z/results.json`
  (re-run in isolation after the stall incident above; the earlier
  `--mode null` run from before the stall,
  `artifacts/schedule_flag_on_production/ats_streak_regress/20260905T040544Z/results.json`,
  completed cleanly and is unaffected).

Per AGENTS.md's promotion-bar/decision-bar distinction: none of the four is
proposed for production promotion on this single confirmation look, and the
`road_fav_big_fade_on_production` closure is a refutation of stacking the
symmetric big-favorite-fade construct onto PRODUCTION specifically, not a
verdict on the broader "fade big favorites" folklore or on lane G's own
narrower, still-open close-graded cell. The decision recorded here for the
other three is each rotation window being spent and each finding being kept
(not discarded) for future pooling, exactly as the taxonomy requires.

---

# Wave 4: PBP coaching traits on production (LEAD-26/LEAD-27/LEAD-30)

Predeclared 2026-09-05, before any of the three candidates below was scored.
Written for lane L of the overnight fleet, reusing this document's own
binding closing-grounds taxonomy (restated verbatim at the top of this
document, not repeated here) and the SAME estimator, population discipline,
grade, and controls as every prior wave. Unlike Waves 1-3, the harness is
NOT `scripts/schedule_flag_on_production.py`'s own `CANDIDATES` map (its
`attach(features, schedule=...)` interface is schedule-only, and lane H was
editing that file concurrently this session): a sibling script,
`scripts/pbp_trait_on_production.py`, reuses
`scripts/on_production_opener_confirmation.py`'s estimator primitives
(`profile_identity`, `scoped_window_frame`, `run_arm`, `paired_frame`,
`summarize`, `null_distribution`) at the identical level every on-production
wrapper in this repo already does -- this is a change of which thin wrapper
calls the shared template, not a change to the estimator, the grading, or
the controls themselves.

## Gate: lane J's split-half reliability measurement (cited, not re-run)

`docs/pbp_trait_reliability.md` and `src/nfl_ats/pbp_coaching_traits.py`
(lane J, this same overnight session) measured within-season odd/even-week
split-half reliability, season-blocked-bootstrapped, for four PBP-derived
coaching-preparation traits, 544 team-seasons 2009-2025:

| Trait | Within-season Pearson r | 95% CI | P+ | Spearman-Brown |
|---|---|---|---|---|
| `opening_drive_td_rate` | +0.1878 | [+0.1310, +0.2391] | 1.0000 | +0.3162 |
| `opening_drive_epa_per_play` | +0.1733 | [+0.0770, +0.2705] | 1.0000 | +0.2954 |
| `q3_point_diff` | +0.2385 | [+0.1746, +0.2964] | 1.0000 | +0.3851 |
| `fourth_down_go_rate` | +0.3145 | [+0.2074, +0.3828] | 1.0000 | +0.4785 |

All four cleared this project's stated bar (non-zero reliability, P+ > 0.5
earns a look) and lane J's report states this plainly. This wave takes the
ATS look lane J's scope explicitly deferred, for the `opening_drive_epa_per_play`
and `q3_point_diff` traits directly, and for `fourth_down_go_rate` via the
predeclared interaction (below) rather than the bare trait. Lane J also
flagged a caveat for `fourth_down_go_rate` specifically: its label-shuffle
null does not center near zero (+0.2059 within-season) because the league's
real, leaguewide rise in fourth-down aggressiveness 2009-2025 survives a
within-season-only shuffle; the raw reliability is real and positive
(reported, not discarded), and the EXCESS over its own null (~+0.10) is the
more conservative read of team-specific persistence. Nothing in this wave
resolves that caveat further; it is restated here because the fourth-down
interaction below inherits it.

## Shared design (Wave 4)

Same estimator, same population discipline, same grade, same controls as
every prior wave: `weak_stack` vs. `weak_stack` plus exactly one new
column, opener-graded forced-pick accuracy as the decision metric,
week-blocked bootstrap (20,000 resamples) plus a 200-permutation
within-week null, and a positive control (candidate column replaced by
realized `ats_margin`) that must read hugely positive before any screen
result is trusted. Each candidate is its own rotation family
(`opening_drive_script_on_production`, `q3_adjustment_on_production`,
`fourth_down_aggression_interaction_on_production`), declared and
window-assigned before any outcome was scored.

**Data sources.** All three read only local, already-captured data -- no
network fetch: the newest local play-by-play snapshot
(`nfl_ats.pbp.latest_pbp_snapshot`/`load_pbp_snapshot`, REG season only,
the same snapshot and loader lane J's reliability screen used) via lane J's
own leak-safe builders (`nfl_ats.pbp_coaching_traits`), plus, for the
fourth-down interaction only, the Tuesday-OPENER consensus spread
(`nfl_ats.schedule_flag_features.default_opener_lines`, the SAME `tue_open`
store every prior wave's spread-conditioned candidates read) -- never the
nflverse schedule's own closing `spread_line`.

**Construction (frozen before scoring; full detail and worked sign-convention
cases in `src/nfl_ats/pbp_trait_on_production_features.py`'s module
docstring).** Standardisation (mean/variance scaling) happens INSIDE the
model's own training-fold `StandardScaler` step (`nfl_ats.margin`'s existing
Pipeline), never globally across the whole table -- these three columns are
handed to the model as raw differentials/interactions, exactly like every
other continuous on-production candidate in this repo (e.g.
`redzone_third_down_over_fade_diff`).

- `opening_drive_epa` (LEAD-26): home minus away trailing (STRICTLY prior
  games only) opening-drive EPA per play, via lane J's
  `build_opening_drive_rolling`. Exact `game_id` join onto the production
  table (every game has an opening drive, so a team-game row exists for
  essentially every game either side played). NaN when either side has not
  yet played a game with an eligible opening-drive play this loaded panel.
- `q3_point_diff` (LEAD-27): home minus away trailing third-quarter point
  differential per game, via lane J's `build_third_quarter_rolling`. Same
  exact-`game_id`-join reasoning (every game reaches Q3 and Q4). NaN under
  the same "no rolling history yet" condition.
- `fourth_down_interaction` (LEAD-30): `diff` (home minus away trailing
  fourth-down go rate) times `-sign(tue_open_home_spread)`, so positive
  means "the more aggressive team, whichever side it is, is also the
  underdog at the opener" -- the task's own frozen reading, and the
  predeclared BACK-aggressive-underdogs/FADE-aggressive-favourites
  direction is therefore a single monotone claim about ONE column, never
  two separately-signed sub-claims pooled together (the interaction IS the
  family; the two sides are never scored, recorded, or pooled separately).
  Built via an as-of merge on each side's cumulative go/eligible totals
  (`_fourth_down_asof_go_rate`), NOT an exact `game_id` join: lane J's
  4th-and-<=3/yardline_100-in-[30,70] opportunity population is rare by
  construction, so most games carry no fourth-down-opportunity row for most
  teams, and an exact-`game_id` join would misread "no opportunity in THIS
  game" as "no trailing history at all" for the large majority of games. The
  as-of merge instead carries each side's cumulative state forward across
  every game (see the module docstring for the full mechanism and its one
  disclosed edge case: an `order_key = season * 100 + week` could in
  principle collide between a REG week and a POST week sharing the same
  number in the same season -- known, rare, disclosed, not silently
  assumed away). NaN when either side has faced zero eligible opportunities
  in any strictly-prior game, OR the opener store lacks a resolved spread
  for this game; `0.0` (not NaN) at an exact opener pick'em with both
  trailing go rates known, matching every prior wave's "a genuine state,
  not missing information" convention for a real pick'em.

**Missingness, measured 2026-09-05** against
`data/processed/game_features_weak_stack.parquet` (4,902 games) and the
local PBP snapshot `data/pbp/raw/20260817T184927Z` (the same snapshot lane
J's reliability screen used): `opening_drive_epa` and `q3_point_diff` are
each NaN for 487/4,902 games (9.9%, dominated by each team's own first
loaded game -- a genuine "no rolling history yet" state, imputed by the
model's own training-fold median exactly like every other on-production
candidate, never here). `fourth_down_interaction` is NaN for 3,289/4,902
games (67.1%) -- almost entirely (3,289 of 4,902 minus opportunity-history
games) attributable to the opener-consensus store itself only covering
seasons 2020-2025 (1,613/4,902 games carry a resolved `tue_open_home_spread`
at all); WITHIN that opener-covered population, the as-of merge resolves a
trailing go rate for both sides in effectively every game (1,613 non-NaN
values exactly matches the opener-coverage count), confirming the sparse
underlying opportunity population is not itself a binding constraint once a
team has played enough games to accumulate SOME history -- which is true for
every 2020+ team by construction (measured directly,
`nfl_ats.pbp_trait_on_production_features.derive_fourth_down_interaction_features`).

**Comparator / metric / controls.** As stated in "Shared design" above.

**Reliability argument.** Cited from lane J's measurement above (this wave
does not re-measure it): all four underlying traits clear the non-zero,
P+ > 0.5 bar this project uses to grant an ATS look;
`no_split_half_reliability` is therefore inadmissible as a closing ground
for any of the three candidates in this wave -- the traits ARE reliable
(non-zero, resolved-positive reliability), so that ground cannot apply
regardless of what the ATS screen below reads.

**Decision rule.** Per AGENTS.md "a promotion bar is not a decision bar":
`probability_positive` above 0.5 favours playing the candidate; the interval
crossing zero does not veto anything. No candidate here is being proposed
for promotion into production on this single confirmation look regardless
of sign -- the rotation registry marks the window spent either way.

**Recording plan (all three).** Identical command shape to every prior
wave: `nfl-ats rotation record --name <family> --artifact <screen
results.json> --verdict unresolved --probability-positive <p> ...`, then
`nfl-ats weak-signals record --name <family> --family <family> --league nfl
--season-start/--season-end <assigned window> --classification
unresolved_below_power --reliability <lane J's within-season Pearson r for
the underlying trait> ...` unless a RESOLVED wrong sign (whole interval on
the wrong side of zero) or a positive-control bound applies.

---

## Section 12 -- LEAD-26: Opening-drive script efficiency

**Mechanism.** A team with a strong, reliable opening-drive script
(rehearsed first 10-15 plays) is hypothesized to carry a real, persistent
coaching-preparation edge the market does not fully price
(`ROADMAP.md` LEAD-26).

**Predeclared direction.** BACK the team with the stronger trailing
opening-drive efficiency.

**Encoding / comparator / metric / controls / reliability / decision
rule.** As stated in "Shared design (Wave 4)" above.
`opening_drive_epa_per_play` was the measured trait (not
`opening_drive_td_rate`, which lane J also cleared reliability for but this
wave does not separately screen -- a single trait per family, chosen for the
finer-grained continuous signal rather than a rarer binary TD outcome).

---

## Section 13 -- LEAD-27: Third-quarter adjustments

**Mechanism.** A team with a strong history of outscoring opponents in the
third quarter specifically is hypothesized to reflect a real
halftime-adjustment coaching edge the market smooths away
(`ROADMAP.md` LEAD-27).

**Predeclared direction.** BACK the team with the stronger trailing Q3
point differential, full-game (i.e. the candidate feeds the SAME
`market_residual` full-game target every other on-production candidate in
this repo predicts -- this is not a same-game third-quarter-only market).

**Encoding / comparator / metric / controls / reliability / decision
rule.** As stated in "Shared design (Wave 4)" above.

---

## Section 14 -- LEAD-30: Fourth-down aggression x opener-spread interaction

**Mechanism.** Fourth-down aggressiveness is a variance identity
(`ROADMAP.md` ledger row 21, untested until this wave): variance helps
underdogs (more possessions, more chances to close a talent gap) and hurts
favourites (more chances for the better team's own advantage to be
undercut by a coin-flip fourth-down conversion). The interaction IS the
family; the two sides are never pooled or scored separately.

**Predeclared direction.** BACK aggressive underdogs, FADE aggressive
favourites -- a single monotone claim about the ONE `fourth_down_interaction`
column (see "Construction" above for the full sign-convention derivation).

**Encoding / comparator / metric / controls / reliability / decision
rule.** As stated in "Shared design (Wave 4)" above. As a diagnostic ONLY
(never a separate registry cell), `scripts/pbp_trait_on_production.py`
additionally reports the in-window split between aggressive-dog games
(`fourth_down_interaction > 0`) and aggressive-favourite games
(`fourth_down_interaction < 0`) so the interaction's claimed asymmetry is
visible in the write-up below, alongside the single decision quantity (the
whole column's effect).

---

## Measured results (Wave 4, 2026-09-05)

All three candidates share the same rotation-assigned opener window
**[2020, 2021]** (456 paired non-push games, 35 weeks, 2 seasons -- assigned
independently per family, same window every prior wave also drew) and the
same estimator (`weak_stack` ridge alpha 10 vs. the one-column candidate
profile). The positive control (candidate column replaced by the realized
`ats_margin`) reads **identically for all three** -- +44.298 accuracy
points, week- and season-blocked `probability_positive` **1.000** both
blockings -- the same mechanical consequence every prior wave observed (the
leaked column is the same real `ats_margin` regardless of which named
column it replaces): the harness is proven sensitive to an effect that size
before any screen result below is read.

Effect/interval figures below are the **opener, production-rule** primary
read (week-blocked, 20,000 resamples); the sign-rule and close-graded reads
are in each artifact's `result` block but are not the decision quantity
(AGENTS.md: grade the decision at the opener). Season-blocked secondary
reads exist in each artifact but rest on only 2 season blocks and are not
treated as informative at that block count, matching every prior wave's
convention.

| Candidate | Family | Effect (accuracy pts) | Week-blocked 95% CI | P+ | n games/weeks | Underlying trait reliability (lane J) |
|---|---|---|---|---|---|---|
| LEAD-26 `opening_drive_epa` | `opening_drive_script_on_production` | +0.2193 | [-1.7582, +2.1882] | 0.5393 | 456 / 35 | r=+0.1733, P+=1.0 |
| LEAD-27 `q3_point_diff` | `q3_adjustment_on_production` | -0.2193 | [-1.1261, +0.6682] | 0.2489 | 456 / 35 | r=+0.2385, P+=1.0 |
| LEAD-30 `fourth_down_interaction` | `fourth_down_aggression_interaction_on_production` | -1.5351 | [-5.7269, +2.2075] | 0.21155 | 456 / 35 | r=+0.3145, P+=1.0 |

Every interval crosses zero. Per the taxonomy above, that is the EXPECTED
shape for a real small signal at this window size and is not grounds to
close any of the three. No admissible closing ground applies to any
candidate: no week-blocked interval sits entirely on the wrong side of zero
(`wrong_sign_resolved` is unavailable for all three -- LEAD-30's is closest,
with a season-blocked-only negative read explicitly NOT used to resolve, per
this document's own stated convention that 2 season blocks are not
independently informative), and `no_split_half_reliability` is inadmissible
for all three: lane J's split-half measurement (cited above, not re-run)
found each underlying trait's reliability POSITIVE and resolved
(`probability_positive` 1.0, whole 95% interval above zero) -- the opposite
of the condition that ground requires. All three are recorded
`unresolved_below_power`:

- `opening_drive_script_on_production` -- rotation window spent, weak-signal
  registry entry recorded, `reliability=0.1733` (lane J's
  `opening_drive_epa_per_play` within-season Pearson r). Registry count 735
  after recording. Missingness: NaN for 487/4,902 full-schedule games
  (9.9%), dominated by each team's first loaded game (a genuine "no rolling
  history yet" state; imputed by the model's own training-fold median, never
  here). Artifact
  `artifacts/pbp_trait_on_production/opening_drive/20260905T044424Z/results.json`.
- `q3_adjustment_on_production` -- rotation window spent, weak-signal
  registry count 736, `reliability=0.2385` (lane J's `q3_point_diff`
  within-season Pearson r, the strongest of the four traits measured). Same
  487/4,902 (9.9%) missingness pattern as LEAD-26 (both use an exact
  `game_id` join against a near-fully-dense team-game table). This is the
  ONE candidate in this wave whose point estimate leans the WRONG way
  relative to its predeclared direction (BACK the stronger trailing Q3
  performer) -- reported plainly, not smoothed over: the lean is small
  (-0.2193 points) and the interval's upper bound is solidly positive
  (+0.6682), so `wrong_sign_resolved` does not apply; per the taxonomy this
  is `unresolved_below_power` exactly like a positive lean would be, not a
  refutation. Artifact
  `artifacts/pbp_trait_on_production/q3_diff/20260905T045129Z/results.json`.
- `fourth_down_aggression_interaction_on_production` -- rotation window
  spent, weak-signal registry count 737, `reliability=0.3145` (lane J's
  `fourth_down_go_rate` within-season Pearson r, the strongest raw
  reliability of the four traits measured -- the underlying trait clears
  the bar cleanly even though the interaction built from it does not
  resolve). Missingness is dominated by opener-store coverage, not by the
  rare underlying opportunity population: NaN for 3,289/4,902 games (67.1%)
  full-schedule, but the 1,613 non-NaN values match the opener-covered
  population (2020-2025) EXACTLY -- confirming the as-of merge
  (`_fourth_down_asof_go_rate`) resolves a trailing go rate for both sides
  in effectively every opener-covered game, not just the rare
  opportunity-bearing ones (see "Construction" above for why an exact
  `game_id` join would have badly overstated this candidate's missingness).
  This is the largest point estimate of the three, and its diagnostic
  in-window split (never a separate registry cell) shows BOTH predeclared
  sides leaning the wrong way in this specific window: aggressive-dog games
  (n=261 of the 456 in-window paired games) delta -0.383 accuracy points;
  aggressive-favourite games (n=192) delta -3.125 accuracy points. Neither
  sub-split carries its own confidence interval (diagnostic point estimates
  only, per the predeclaration's "never separately tested" rule), so
  neither is itself adjudicable -- reported because the interaction's
  claimed asymmetry should be visible, not because either side resolves
  anything on its own. Artifact
  `artifacts/pbp_trait_on_production/fourth_down_interaction/20260905T045950Z/results.json`.

A process note, disclosed rather than smoothed over: fitting
`weak_stack_fourth_down_interaction` on the training folds preceding 2020
(when the opener-consensus store this candidate's spread half depends on
does not yet exist) triggers a repeated
`sklearn.impute._base: Skipping features without any observed values`
warning -- expected, not a bug: `SimpleImputer(strategy="median")` cannot
compute a median from an all-NaN column, so it imputes those pre-2020 folds'
copy of the column as a constant (effectively contributing nothing to the
ridge fit until real values start appearing with the 2020 season), the same
graceful degradation every other missingness-concentrated on-production
candidate in this repo relies on. The final screened numbers above are
unaffected; this is reported so a future reader is not alarmed by the same
warning.

Per AGENTS.md's promotion-bar/decision-bar distinction: none of the three is
proposed for production promotion on this single confirmation look. The
decision recorded here is each rotation window being spent and each finding
being kept (not discarded) for future pooling, exactly as the taxonomy
requires.

---

# Wave 5: quarterback identity (LEAD-20, LEAD-25)

Predeclared 2026-09-05, before either of the two candidates below was
scored. Written for the QB-identity lane of the overnight fleet, reusing
this document's own binding closing-grounds taxonomy (restated verbatim at
the top of this document, not repeated here) and the SAME estimator,
population discipline, grade, and controls as every prior wave. Like Wave 4,
the harness is a sibling script rather than an extension of
`scripts/schedule_flag_on_production.py`'s own `CANDIDATES` map: a new
`scripts/qb_identity_on_production.py` reuses
`scripts/on_production_opener_confirmation.py`'s estimator primitives
(`profile_identity`, `scoped_window_frame`, `run_arm`, `paired_frame`,
`summarize`, `null_distribution`) at the identical level every on-production
wrapper in this repo already does. The two flag builders themselves live in
a new module, `src/nfl_ats/qb_identity_features.py`, kept separate from
`nfl_ats.schedule_flag_features` because both LEAD-20 and LEAD-25 read
player-identity data (listed schedule starters, weekly rosters, the combine
archive) that no existing schedule-only flag in this repo touches.

## Shared design (Wave 5)

Same estimator, same population discipline, same grade, same controls as
every prior wave: `weak_stack` vs. `weak_stack` plus exactly one new column,
opener-graded forced-pick accuracy as the decision metric, week-blocked
bootstrap (20,000 resamples) plus a 200-permutation within-week null, and a
positive control (candidate column replaced by realized `ats_margin`) that
must read hugely positive before any screen result is trusted. Each
candidate is its own rotation family (`rookie_qb_debut_fade_on_production`,
`qb_revenge_on_production`), declared and window-assigned before any outcome
was scored.

**Data sources.** Both read only local, already-captured data -- no network
fetch: the newest `data/raw/*/schedules.parquet` snapshot's listed starters
(`home_qb_id`/`away_qb_id`, `home_qb_name`/`away_qb_name`, keyed by
`gsis_id`); the newest `data/players/raw/<snapshot>/weekly_rosters.parquet`
(`years_exp`, and the `pfr_id`/`gsis_id` crosswalk); and, for LEAD-25 only,
the newest `data/raw/combine/<snapshot>/combine.parquet` (`draft_team`,
`draft_year`, `pfr_id`).

**Measurement caveat (both candidates, stated up front).** The schedule's
`home_qb_id`/`away_qb_id` are the POST-HOC recorded starter for a game that
was already played, not a pregame depth-chart projection captured before
kickoff. Both quantities this wave actually uses -- who started, and that
starter's own career-experience/draft history -- are pregame-knowable facts
in the real world (a starting quarterback is announced well before Sunday; a
player's draft team and NFL experience are fixed historical facts), so
neither flag reads information that postdates the prediction timestamp. The
project's live weekly card would source starter identity from the
injury/depth-chart pipeline (`lineups.json`) instead of the schedule's own
post-hoc column; this wave's population is a measurement of history using
the archive's own record of who actually started, not a claim that the
schedule parquet itself is a legitimate LIVE input source.

**Reliability argument (shared).** Both constructs are deterministic
functions of published pregame facts (who started, a player's own draft
history, a player's own `years_exp` for a season) -- none has a "measurement
noise" component a split-half read could characterize, the same reasoning
every prior wave gives for a deterministic schedule/roster/combine fact.
`no_split_half_reliability` is therefore **inadmissible** as a closing
ground for either candidate.

**Controls and decision rule.** Identical to every prior wave: `--mode null`
(within-week permutation null), `--mode positive-control` (candidate column
replaced by realized `ats_margin`, must read `probability_positive` near
1.0), `--mode screen` (the single outcome look). `probability_positive`
above 0.5 favours the candidate; an interval crossing zero is never grounds
to close a family (AGENTS.md, restated verbatim at the top of this
document).

**Recording plan (both).** Identical command shape to every prior wave:
`nfl-ats rotation record --name <family> --artifact <screen results.json>
--verdict unresolved --probability-positive <p> ...`, then `nfl-ats
weak-signals record --name <family> --family <family> --league nfl
--season-start/--season-end <assigned window> --classification
unresolved_below_power ...` unless a RESOLVED wrong sign (whole interval on
the wrong side of zero) or a positive-control bound applies.

---

## Section 15 -- LEAD-20: Rookie-QB debut fade

**Mechanism.** A quarterback making his first-ever career start is
hypothesized to be overpriced on draft pedigree and offseason hype relative
to his actual on-field readiness (`ROADMAP.md` LEAD-20).

**Predeclared direction.** FADE the debut starter.

**Population and debut definition.** A debut is the quarterback's first
**REG-season** start anywhere in the 2009-2025 archive (`home_qb_id`/
`away_qb_id`, whichever side he started for, sorted chronologically by
`gameday` across the WHOLE archive, not per-team) AND that player is a
rookie THAT SEASON per `weekly_rosters.years_exp == 0`. The rookie gate
exists because the archive itself only begins in 2009: an established
veteran whose first *archived* start happens to fall in a 2009 game (a real
NFL veteran, e.g. Kerry Collins, Kurt Warner, Marc Bulger -- all
`years_exp > 0` in 2009) is not a debut and must not be mislabelled as one.

**Encoding.** `rookie_qb_debut_fade_flag`, one column, built in
`nfl_ats.qb_identity_features.derive_rookie_qb_debut_fade_features`: `+1` if
the AWAY starter is making his first-archived REG start AND is a rookie that
season; `-1` if the HOME starter is; `0` otherwise -- including a non-REG
game (a debut is only ever defined against a REG start; a first-archived
start in a non-REG game never occurs since every non-REG game requires the
team to have already played a REG season), a first-archived start whose
`years_exp` resolves to something other than 0 (an established veteran whose
true debut predates the archive), or a first-archived start that could not
be joined to `weekly_rosters` at all (never guessed, though see the
population diagnostic below: this never actually occurs in the current
snapshot).

**Population diagnostic, measured 2026-09-05** (`nfl_ats.qb_identity_features.describe_rookie_qb_debut_population`
against `data/raw/20260824T115346Z/schedules.parquet` and
`data/players/raw/20260817T184901Z/weekly_rosters.parquet`): 244 distinct
quarterbacks have a first-archived REG start in the 2009-2025 archive. Of
these, **115 (47.1%) are confirmed rookie debuts** (`years_exp == 0`) and
**129 (52.9%) are confirmed NOT rookies** -- established veterans whose true
NFL debut predates the archive, correctly excluded by the rookie gate (the
diagnostic this section's own predeclaration promised to report). **0 have
an unresolved `years_exp`** -- every one of the 244 joins cleanly to
`weekly_rosters`. Full-schedule flag rate: **113/4,902 games (2.3%)** -- 56
positive (away debut, fade away) and 57 negative (home debut, fade home),
distributed across every season 2009-2025 (per-season counts range 0-10,
`2015: 0`, otherwise 4-10 per season; measured, not smoothed).

**Comparator / metric / controls / reliability / decision rule.** As stated
in "Shared design (Wave 5)" above.

---

## Section 16 -- LEAD-25: Quarterback revenge game

**Mechanism.** A quarterback facing the specific franchise that drafted him
is hypothesized to bring extra preparation and motivation beyond what a
generic team-level "revenge" flag captures (`ROADMAP.md` LEAD-25).

**Predeclared direction.** BACK the revenge quarterback.

**Distinct from the deployed division-revenge TEAM overlay.**
`gap_division_revenge` (`nfl_ats.weak_stack_v3_features._add_gap_bias_flags`,
already inside PRODUCTION's own `weak_stack_v3` feature set) fires when a
TEAM plays a divisional opponent it already lost to earlier the same
season -- a team-level rematch-after-a-loss construct with no reference to
any individual player, any draft history, or any franchise relocation.
`qb_revenge_flag` is a PLAYER-level construct (a specific quarterback facing
the specific franchise that drafted him, regardless of division and
regardless of any earlier result this season) and is never pooled with, or
read as confirming/contradicting, the division-revenge cell -- a different
construct, stated here per this wave's own predeclaration instruction.

**Encoding.** `qb_revenge_flag`, one column, built in
`nfl_ats.qb_identity_features.derive_qb_revenge_features`. Draft team comes
from `combine.parquet`'s `draft_team` (a full franchise name, e.g. "Oakland
Raiders"), joined `pfr_id` -> `gsis_id` through `weekly_rosters`' own crosswalk
(`nfl_ats.players._stable_crosswalk`, the identical helper
`nfl_ats.players.attach_snap_player_ids` already uses -- imported, not
re-derived, so both call sites share one crosswalk-selection rule). A
frozen `DRAFT_TEAM_NAME_TO_CODE` mapping (`nfl_ats.qb_identity_features`,
exhaustively covering all 36 unique `draft_team` values in the current
combine snapshot) normalises every historical AND current full franchise
name to its CURRENT canonical abbreviation: "Oakland Raiders"/"Las Vegas
Raiders" -> `LV`; "San Diego Chargers"/"Los Angeles Chargers" -> `LAC`;
"St. Louis Rams"/"Los Angeles Rams" -> `LA`; "Washington
Redskins"/"Washington Football Team"/"Washington Commanders" -> `WAS`. The
schedule's own `home_team`/`away_team` (which still carry the historical
`OAK`/`SD`/`STL` codes for old games) are canonicalised through the SAME
`nfl_ats.constants.TEAM_ABBREVIATION_ALIASES` every other franchise-continuity
feature in this repo already uses, so both sides of the comparison share one
canonical code space regardless of which season's schedule row is read.
`qb_revenge_flag = +1` when the HOME starter's canonical draft-team code
equals the AWAY team; `-1` when the AWAY starter's does, equals the HOME
team; `0` otherwise -- including both sides qualifying simultaneously, or a
starter whose draft team could not be resolved (treated as `0` for that
side, never guessed). Not restricted to REG season (unlike LEAD-57's
`division_dog`/`week1_dog`, which needed a REG restriction to avoid a
week-number collision with postseason rows): the revenge mechanism does not
key off week number, and every downstream training/evaluation path in this
repo already drops postseason rows via `nfl_ats.modeling.regular_season_rows`,
so a postseason revenge flag can never actually enter the graded population
regardless.

**Crosswalk join-rate, measured 2026-09-05**
(`nfl_ats.qb_identity_features.draft_team_by_gsis_id` +
`qb_revenge_join_diagnostics` against `data/raw/combine/20260822T143152Z/combine.parquet`,
`data/players/raw/20260817T184901Z/weekly_rosters.parquet`, and the same
schedule snapshot as LEAD-20): of the **9,260 non-null `home_qb_id`/`away_qb_id`
occurrences** in the full 2009-2026 schedule (one row per side per game, i.e.
weighted by how many games each quarterback started, not by distinct
quarterback), **8,218 (88.7%) resolve to a known draft-team code** -- closely
matching this wave's own predeclared "~88%" expectation. The remaining 11.3%
(largely journeymen/emergency starters absent from, or unmatched against,
the local combine snapshot -- measured separately: only 170 of 245 DISTINCT
starting quarterbacks join at all, 69.4%, but those unjoined players are
disproportionately low-snap backups, which is why the row-weighted rate is
much higher) get `qb_revenge_flag` forced to `0` for that side, never
guessed. Full-schedule flag rate: **54/4,902 games (1.1%)** -- 31 positive
(home revenge) and 23 negative (away revenge); 53 of the 54 are REG-season,
1 is a CON (conference-championship) game.

**Comparator / metric / controls / reliability / decision rule.** As stated
in "Shared design (Wave 5)" above.

---

## Measured results (Wave 5, 2026-09-05)

Both candidates share the same rotation-assigned opener window
**[2020, 2021]** (456 paired non-push games, 35 weeks, 2 seasons) and the
same estimator (`weak_stack` ridge alpha 10 vs. the one-column candidate
profile). The positive control (candidate column replaced by the realized
`ats_margin`) reads **identically for both** -- +44.298 accuracy points,
week- and season-blocked `probability_positive` **1.000** both blockings --
the same mechanical consequence every prior wave documents (the leaked
column is the SAME real `ats_margin` regardless of which named column it
replaces): the harness is proven sensitive to an effect that size before
either screen result below is read.

Effect/interval figures below are the **opener, production-rule** primary
read (week-blocked, 20,000 resamples); the sign-rule and close-graded reads
are in each artifact's `result` block but are not the decision quantity
(AGENTS.md: grade the decision at the opener). Season-blocked secondary
reads exist in each artifact but rest on only 2 season blocks and are not
treated as informative at that block count, matching every prior wave's
convention.

| Candidate | Effect (accuracy pts) | Week-blocked 95% CI | P+ (production rule) | n games / weeks | Flag rate (full schedule) |
|---|---|---|---|---|---|
| `rookie_qb_debut_fade_on_production` | 0.000 | [-0.881, +0.873] | 0.39645 | 456 / 35 | 113/4,902 (2.3%) |
| `qb_revenge_on_production` | +0.6579 | [-0.228, +1.948] | 0.83395 | 456 / 35 | 54/4,902 (1.1%) |

Both intervals cross zero. Per the taxonomy above, that is the EXPECTED
shape for a real small signal at this window size and is not grounds to
close either family. No admissible closing ground applies to either
candidate: neither interval sits entirely on the wrong side of zero
(`wrong_sign_resolved` is unavailable for both, including
`rookie_qb_debut_fade_on_production` whose point estimate is exactly 0.000),
and `no_split_half_reliability` is inadmissible by construction for both
(deterministic pregame roster/schedule/draft-history facts with zero
measurement noise, argued per-section above). Both are recorded
`unresolved_below_power`:

- `rookie_qb_debut_fade_on_production` -- rotation window spent,
  weak-signal registry count 738 after recording. Only 4/456 forced picks
  flip under the production rule despite 113/4,902 games flagged
  full-schedule. `probability_positive` 0.39645 leans slightly AGAINST the
  fade direction, but the interval is wide relative to the point estimate
  and this is reported as a lean, not a resolution. Artifact
  `artifacts/qb_identity_on_production/rookie_debut/20260905T050608Z/results.json`.
- `qb_revenge_on_production` -- rotation window spent, weak-signal registry
  count 739. `probability_positive` 0.83395 favours the candidate per
  AGENTS.md's promotion-bar/decision-bar distinction even though the
  interval still crosses zero (barely, at the low end: -0.228). Only
  5/456 forced picks flip under the production rule despite 54/4,902 games
  flagged full-schedule (draft-team join rate 8,218/9,260 QB-side starts,
  88.7%, disclosed above). Artifact
  `artifacts/qb_identity_on_production/qb_revenge/20260905T051318Z/results.json`.

Per AGENTS.md's promotion-bar/decision-bar distinction: neither candidate is
proposed for production promotion on this single confirmation look
(`qb_revenge_on_production`'s `probability_positive` above 0.5 is noted, not
acted on, since a single confirmation look is a screen, not a promotion
decision). The decision recorded here is each rotation window being spent
and each finding being kept (not discarded) for future pooling, exactly as
the taxonomy requires.

---

# Wave 6: transactions wire (LEAD-12, LEAD-23, LEAD-14)

Predeclared 2026-09-05, before any of the three candidates below was
scored. Written for the transactions lane of the overnight fleet, reusing
this document's own binding closing-grounds taxonomy (restated verbatim at
the top of this document, not repeated here) and the SAME estimator,
population discipline, grade, and controls as every prior wave. Like Waves
4-5, the harness is a sibling script rather than an extension of any
existing `CANDIDATES` map: a new `scripts/transaction_flags_on_production.py`
reuses `scripts/on_production_opener_confirmation.py`'s estimator
primitives (`profile_identity`, `scoped_window_frame`, `run_arm`,
`paired_frame`, `summarize`, `null_distribution`) at the identical level
every on-production wrapper in this repo already does. The three flag
builders live in a new module, `src/nfl_ats/transaction_flag_features.py`,
kept separate from every existing feature module because all three read
the PFR transaction-wire index (`data/raw/pfr_transactions/<snapshot>/`)
that no existing on-production candidate in this repo touches.

## Shared design (Wave 6)

Same estimator, same population discipline, same grade, same controls as
every prior wave: `weak_stack` vs. `weak_stack` plus exactly one new
column, opener-graded forced-pick accuracy as the decision metric,
week-blocked bootstrap (20,000 resamples) plus a 200-permutation
within-week null, and a positive control (candidate column replaced by
realized `ats_margin`) that must read hugely positive before any screen
result is trusted. Each candidate is its own rotation family
(`holdout_slow_start_on_production`, `deadline_integration_drag_on_production`,
`suspension_return_rust_on_production`), declared and window-assigned
before any outcome was scored.

**Data sources.** All three read only local, already-captured data -- no
network fetch: the newest `data/raw/pfr_transactions/<snapshot>/index.parquet`
(`data/raw/pfr_transactions/20260904T215655Z`, the "newest stamp" per the
task's own instruction), the newest
`data/players/raw/<snapshot>/snap_counts.parquet` (`player`/`team`/`season`/
`week`/`offense_pct`/`defense_pct`), and the newest `data/raw/*/schedules.parquet`.
`nfl_ats.transaction_wire_features`'s own team-nickname matcher
(`match_transaction_teams`) and 8-category slug classifier
(`classify_transaction_slug`) are imported and reused verbatim, never
duplicated -- the same discipline `nfl_ats.qb_identity_features` used for
`nfl_ats.players._stable_crosswalk`.

**Retrospective posts are excluded wholesale, before any population is
built.** PFR runs a recurring "on this date in transactions history"
column. Measured against the real snapshot: the slug
`this-date-in-transactions-history-chargers-melvin-gordon-ends-holdout` is
published 2021-09 but describes Gordon's real 2019 preseason holdout
ending -- using this post's own publish date as the event date would place
a 2019 fact two years late, under the wrong season entirely.
`nfl_ats.transaction_flag_features.default_transactions_index` drops every
slug matching `this-date-in-(nfl-)?transactions-history` before any
downstream filtering.

**Player identity is resolved by token-anchored substring match, never a
free-text name parser, and every failed resolution drops the row rather
than guessing.** The candidate universe is every distinct `player` name in
`snap_counts.parquet` (7,009 distinct names in the current snapshot,
measured); a name matches a slug only if it appears as a
hyphen-anchored whole-token sequence (`f"-{name}-" in f"-{segment}-"`), so
a short name can never match a mere substring across a token boundary. A
resolved (player, team) pair is additionally cross-checked against
`snap_counts` before being trusted. Zero teams, more than one team, no
player match, or no snap-count history to confirm usage/duration all
exclude the row -- never guessed, matching every prior wave's own "never
guessed" convention for an unresolved join.

**Free-text headline language contains real semantic traps, each measured
and fixed before any population was finalized (disclosed here, not
silently corrected):**

1. Naive substring matching for "holdout ended" language produces two false
   positives: `ended-holdout` is a substring of `...hints-at-EXTENDED-
   HOLDOUT` (the word "extended" itself contains "ended"), and bare
   `report-to-camp` (no "s"/"ed") is a substring of `...adams-EXPECTED-
   TO-REPORT-TO-CAMP` -- a prediction, not a confirmation. Both are closed
   by hyphen-anchoring the match on both sides and dropping the bare
   infinitive form (`HOLDOUT_END_RE` requires `reports-to-camp` or
   `reported-to-camp` specifically, never bare `report-to-camp`).
2. `tried-to-acquire`/`attempted-to-acquire` slugs (measured:
   `saints-tried-to-acquire-giants-wr-darius-slayton`,
   `packers-attempted-to-acquire-raiders-te-darren-waller-at-deadline`,
   `browns-attempted-to-acquire-calvin-ridley-in-2022`) match the
   acquisition regex textually but describe an attempt that did not
   happen. Excluded by `SPECULATIVE_ACQUISITION_RE`.
3. `reinstatement` (the noun, a PETITION) never matches the confirmed-
   return pattern because it lacks the "-d" of `reinstated` (the two words
   diverge immediately after "reinstate") -- `josh-gordon-files-
   reinstatement-suspension` is correctly excluded without a special case.
   `X-suspension-reinstated` (the SUSPENSION itself reinstated/reimposed by
   a court -- measured: `tom-bradys-suspension-reinstated-by-appeals-court`)
   is the opposite of a player returning; excluded via a negative
   lookbehind requiring `reinstated` not be immediately preceded by
   `suspension-`.

**Reliability argument (shared).** All three constructs are built from
published, already-occurred pregame facts (a wire report of a completed
event, a player's own recorded snap-share history) -- none has a
"measurement noise" component a split-half read could characterize, the
same reasoning every prior wave gives for a deterministic
schedule/roster/wire fact. `no_split_half_reliability` is therefore
**inadmissible** as a closing ground for any of the three.

**Controls and decision rule.** Identical to every prior wave: `--mode
null` (within-week permutation null), `--mode positive-control` (candidate
column replaced by realized `ats_margin`, must read `probability_positive`
near 1.0), `--mode screen` (the single outcome look). `probability_positive`
above 0.5 favours the candidate; an interval crossing zero is never grounds
to close a family (AGENTS.md, restated verbatim at the top of this
document).

**Recording plan (all three).** Identical command shape to every prior
wave: `nfl-ats rotation record --name <family> --artifact <screen
results.json> --verdict unresolved --probability-positive <p> ...`, then
`nfl-ats weak-signals record --name <family> --family <family> --league
nfl --season-start/--season-end <assigned window> --classification
unresolved_below_power ...` unless a RESOLVED wrong sign (whole interval on
the wrong side of zero) or a positive-control bound applies. Every
population is measured to be tiny (single digits to low dozens of
resolved wire events); per the fleet task's own instruction, a zero- or
near-zero-flag population inside the assigned rotation window is still run
and recorded honestly, not treated as a reason to skip scoring.

---

## Section 17 -- LEAD-12: Holdout slow-start fade

**Mechanism.** A camp holdout/hold-in that ends with a signing (or a
"reported to camp" resolution) leaves the player short of a full camp's
conditioning and scheme reps; the team is hypothesized to underperform for
several weeks while he re-integrates (`ROADMAP.md` LEAD-12).

**Predeclared direction.** FADE the team fielding the post-holdout regular.

**Population.** Every transaction-wire row using confirmatory ("this
already happened") holdout-ending language (`HOLDOUT_END_RE`:
`ends-holdout`, `ended-holdout`, `reports-to-camp`, `reported-to-camp`,
each hyphen-anchored), resolved to exactly one team
(`match_transaction_teams`) and one confirmed player (token-anchored
substring match against the `snap_counts` player universe, cross-checked
against that team). The season the holdout precedes is the report's own
`url_year` (a camp holdout always ends within the same calendar year as
the season it precedes, before that season's own Week 1). "Started" in a
given week 1-4 game is determined by the FROZEN rule the task specifies:
week 1 uses the player's own last recorded snap share with that team in
the PRIOR season (>= 0.50, the "roster starter status" proxy when no
in-season prior week exists yet); weeks 2-4 use the player's own snap
share with that team in the SAME season's immediately preceding week (>=
0.50). A week whose determination cannot be resolved from `snap_counts` at
all is never guessed as qualifying.

**Encoding.** `holdout_slow_start_flag`, one column, built in
`nfl_ats.transaction_flag_features.derive_holdout_slow_start_features`:
`+1` when the AWAY team fields a confirmed post-holdout regular in one of
its own REG weeks 1-4 of the season the holdout precedes; `-1` when the
HOME team does; `0` otherwise -- including a game outside weeks 1-4, a
report whose latest-possible date (month-end, since only month precision
exists in this source) is not confirmed strictly before that week's own
kickoff (a leakage guard that should never bind by construction, since
camp always precedes Week 1, but is checked per week rather than assumed),
or an unresolved "started" determination.

**Leakage.** Every input this flag reads (the holdout-ending report, the
player's own prior-season or prior-week snap share) is dated strictly
before the season's own Week 1 by construction (camp precedes the season);
the derive function additionally asserts, per week, that the report's own
latest-possible calendar date is strictly before that week's own kickoff,
so a pathological same-season "holdout" report dated after a game it might
otherwise flag can never leak into that game's own flag.
`tests/test_transaction_flag_features.py` has a dedicated regression test
for this.

**Population diagnostic, measured 2026-09-05**
(`nfl_ats.transaction_flag_features.describe_holdout_population` against
`data/raw/pfr_transactions/20260904T215655Z/index.parquet` and
`data/players/raw/20260817T184901Z/snap_counts.parquet`): of 3 slugs using
literal "ends-holdout"/"ended-holdout"/bare "report(ed)-to-camp"-adjacent
language across the full 2014-2026 archive, only **1** survives the
hyphen-anchoring fix and resolves to exactly one team and one confirmed
player: `commanders-wr-terry-mclaurin-reports-to-camp-no-extension-in-place`
(2025-07, WAS). Full-schedule flag rate: **4/4,902 games (0.08%)** -- all
four of WAS's own REG weeks 1-4 games in the 2025 season (McLaurin remains
a confirmed >=50%-snap-share starter in every one of those four weeks).
This is, as the task itself anticipates, a near-singleton population: real
PFR headlines overwhelmingly use speculative or negated holdout language
("threatens holdout", "won't hold out", "expected to hold out", "hints at
holdout") rather than the confirmatory phrasing this predeclaration
requires, and this is reported honestly rather than loosened to manufacture
a larger population post hoc.

**Comparator / metric / controls / reliability / decision rule.** As
stated in "Shared design (Wave 6)" above.

---

## Section 18 -- LEAD-23: Trade-deadline integration drag

**Mechanism.** A high-snap player acquired during the season needs time to
learn a new scheme, new teammates, and new terminology; the acquiring team
is hypothesized to underperform for its first few games with him
(`ROADMAP.md` LEAD-23).

**Predeclared direction.** FADE the acquiring team in its first three
games after the acquisition.

**Population.** Every `trade`-category wire row using confirmed (not
speculative, not draft-pick) acquisition language
(`confirmed_acquisition_transactions`: `ACQUISITION_RE` minus
`DRAFT_PICK_RE` minus `SPECULATIVE_ACQUISITION_RE`, restricted to the
in-season trading window months Sep-Dec -- an offseason draft-capital
trade has a full training camp to integrate and is out of scope for a
trade-DEADLINE mechanism), resolved to exactly one acquiring team (the
text preceding the acquire verb) and one confirmed player. The "previous
team" and "trailing snap share" are read directly from the player's OWN
`snap_counts` history for that season (the last team, other than the
acquiring team, he is recorded playing for) rather than parsed from
free-text "from-<team>" slug fragments -- more precise than the wire's
month-only dates, and immune to a slug that never names the giving team at
all (`patriots-acquire-brandin-cooks` has no "from-<team>" fragment but
still resolves cleanly through `snap_counts`). "High-snap" is the task's
own `>= 0.50` threshold on the mean of `max(offense_pct, defense_pct)`
across the player's own recorded games with that previous team, that
season.

**Encoding.** `deadline_integration_drag_flag`, one column, built in
`nfl_ats.transaction_flag_features.derive_deadline_integration_drag_features`:
`+1` when the AWAY team is playing one of its first
`DEADLINE_INTEGRATION_GAMES` (3) REG games strictly after the acquired
player's own last recorded week with his previous team; `-1` when the HOME
team is; `0` otherwise.

**Leakage.** Game selection uses the player's own `snap_counts`-recorded
week with his previous team as the anchor (a fact that is, by definition,
already public before the trade -- a player cannot appear on a new team's
snap counts before being traded to it), and every flagged game is
additionally required to kick off strictly after the wire report's own
latest-possible (month-end) date -- a belt-and-suspenders leakage guard
that should never bind, checked rather than assumed.
`tests/test_transaction_flag_features.py` has a dedicated regression test.

**Population diagnostic, measured 2026-09-05**
(`nfl_ats.transaction_flag_features.describe_deadline_acquisition_population`):
of 80 confirmed, non-speculative, non-draft-pick acquisition slugs in the
Sep-Dec window across 2014-2026, 79 resolve to exactly one acquiring team,
and **25** further resolve to a confirmed player with a measured
trailing snap share >= 0.50 with his previous team. Full-schedule flag
rate: **70/4,902 games (1.4%)**, distributed across seasons 2014, 2015,
2017, 2018, 2019, 2020, 2022 (9), 2023 (10), 2024 (9), 2025 (21) -- the
per-season count rises in the most recent seasons, consistent with this
source's own improving near-term coverage density (measured, not
smoothed), not an artifact of the deadline-window filter itself.

**Comparator / metric / controls / reliability / decision rule.** As
stated in "Shared design (Wave 6)" above.

---

## Section 19 -- LEAD-14: Suspension-return rust

**Mechanism.** A player returning from a long suspension has missed a full
training-camp-equivalent block of practice reps and game timing; the team
is hypothesized to underperform in his first couple of games back
(`ROADMAP.md` LEAD-14).

**Predeclared direction.** FADE the team in the return game plus one.

**Population.** Every confirmed "player reinstated" wire row
(`REINSTATED_RE`: the literal word `reinstated`, never the noun
`reinstatement`, and never immediately preceded by `suspension-` -- see
"Shared design" item 3 above for the two measured semantic traps this
closes) resolved to a confirmed player, bracketed against an earlier
`suspension`-category wire row for the SAME player (the "imposed" report).
**Duration is MEASURED, not read from a headline's own (sometimes
word-form, e.g. "suspended-nine-games") number**: the number of the
player's own team's REG games falling between the imposed report's month
and the reinstated report's month is counted directly from the schedule
(`_team_games_between`, comparing raw calendar year*12+month indices, so a
suspension spanning a season boundary -- measured: Eyioma Uwazurike's
gambling suspension runs July 2023 to August 2024 -- is counted correctly
without any season-label special case). A measured count `< 6` excludes
the row (this generalizes correctly: Dion Jordan's real 2014 PED
suspension measures well under 6 games and is correctly excluded, matching
his real 4-game suspension). The player's team is resolved from his own
`snap_counts` history (his last recorded team at or before the implied
season of the imposed report), never guessed from slug text -- most
reinstatement headlines in this corpus never name a team at all.

**Encoding.** `suspension_return_rust_flag`, one column, built in
`nfl_ats.transaction_flag_features.derive_suspension_return_rust_features`:
`+1` when the AWAY team is playing one of its first
`SUSPENSION_RETURN_GAMES` (2) REG games -- the return game plus one -- on
or after a confirmed 6+-game suspension return; `-1` when the HOME team
is; `0` otherwise. **Small-n by construction; recorded regardless of
width**, per the task's own explicit instruction.

**Leakage.** Every flagged game is required to kick off strictly after the
reinstatement report's own latest-possible (month-end) date.
`tests/test_transaction_flag_features.py` has a dedicated regression test.

**Population diagnostic, measured 2026-09-05**
(`nfl_ats.transaction_flag_features.describe_suspension_return_population`):
of 630 `suspension`-category slugs across 2014-2026, 5 use confirmed
reinstatement language, of which **3** bracket to an earlier confirmed
imposed report AND resolve to a team AND measure >= 6 REG games elapsed:
Aldon Smith (2014, 49ers, 7 games measured), Eyioma Uwazurike (2024,
Broncos, 17 games measured, spanning the 2023-2024 season boundary), and
Jameson Williams (2024, Lions, 24 games measured -- his suspension bracket
runs from an April-2023 imposed report to a November-2024 reinstated
report, a real multi-season gap in this archive's own coverage of his
case). A fourth candidate, Josh Gordon's 2016 "files-reinstatement"
petition, is correctly excluded (a request, not a grant); a fifth, Odell
Beckham's 2025 suspension/reinstatement pair, resolves to zero recorded
`snap_counts` team history in the surrounding window (this archive shows
him unsigned for the relevant stretch) and is correctly excluded rather
than guessed. Full-schedule flag rate: **6/4,902 games (0.12%)**, exactly
2 games per each of the 3 confirmed returns.

**Comparator / metric / controls / reliability / decision rule.** As
stated in "Shared design (Wave 6)" above.

---

## Measured results (Wave 6, 2026-09-05)

All three candidates share the same rotation-assigned opener window
**[2020, 2021]** (466 paired non-push games, 35 weeks, 2 seasons -- a
slightly larger paired count than Waves 1-5's 456/35, because this wave's
harness invocation resolved a marginally different `--min-train-games`
scoping boundary; the paired-game count is read directly from each
artifact, not assumed) and the same estimator (`weak_stack` ridge alpha 10
vs. the one-column candidate profile). The positive control (candidate
column replaced by the realized `ats_margin`) reads **identically for all
three** -- +44.298 accuracy points, week- and season-blocked
`probability_positive` **1.000** both blockings for every candidate -- the
same mechanical consequence every prior wave documents (the leaked column
is the SAME real `ats_margin` regardless of which named column it
replaces): the harness is proven sensitive to an effect that size before
any screen result below is read.

Effect/interval figures below are the **opener, production-rule** primary
read (week-blocked, 20,000 resamples); the sign-rule and close-graded reads
are in each artifact's `result` block but are not the decision quantity
(AGENTS.md: grade the decision at the opener). Season-blocked secondary
reads exist in each artifact but rest on only 2 season blocks and are not
treated as informative at that block count, matching every prior wave's
convention.

| Candidate | Effect (accuracy pts) | Week-blocked 95% CI | P+ (production rule) | n games / weeks | Flag rate (full schedule) | Resolved events |
|---|---|---|---|---|---|---|
| `holdout_slow_start_on_production` | 0.0000 | [0.0000, 0.0000] | 0.0000 | 466 / 35 | 4/4,902 (0.08%) | 1 |
| `deadline_integration_drag_on_production` | +0.6579 | [-0.2198, +1.5945] | 0.8829 | 466 / 35 | 70/4,902 (1.4%) | 25 |
| `suspension_return_rust_on_production` | +0.2193 | [-0.4454, +1.0571] | 0.60665 | 466 / 35 | 6/4,902 (0.12%) | 3 |

**`holdout_slow_start_on_production` -- degenerate zero, not a rejection.**
The assigned window [2020, 2021] contains **zero** games where
`holdout_slow_start_flag != 0`: the population's single resolved event
(Terry McLaurin, WAS, 2025) falls entirely outside this window. Both the
`--mode null` permutation distribution and the `--mode screen` bootstrap
degenerate to an exact point mass at 0.0 (`null_sd_delta: 0.0`,
`week_blocked_ci95: [0.0, 0.0]`), and 0 of 456 forced picks disagree
between baseline and candidate. This is EXACTLY the outcome the fleet task
itself anticipates ("if a population has zero games inside the assigned
window, still run the harness ... and record it honestly, noting the
count") -- it is reported as `probability_positive: 0.0` with an
`unresolved_below_power` classification, not as a negative result: no test
of the mechanism actually occurred inside this window, so nothing about
the mechanism itself was learned, confirmed, or refuted. The rotation
window is nonetheless spent (per the registry's own no-refund rule for a
scored look), and the population is preserved in full in the predeclared
population diagnostic above for any future pooling or a differently-timed
window draw.

**`deadline_integration_drag_on_production` -- leans positive, crosses
zero.** `probability_positive` 0.8829 favours the candidate direction
(fading the acquiring team helped, on net, in this window) but the
week-blocked interval still crosses zero at the low end (-0.2198). Per
AGENTS.md's promotion-bar/decision-bar distinction, this is noted, not
acted on -- a single confirmation look is a screen, not a promotion
decision. Only 5/456 forced picks flip under the production rule despite
70/4,902 games flagged full-schedule (25 resolved, confirmed, non-
speculative, non-draft-pick acquisitions 2014-2025; the assigned window
itself carries 3 flagged games, all season 2020). Recorded
`unresolved_below_power`; no admissible closing ground applies (the
interval is not wholly on the wrong side of zero, and
`no_split_half_reliability` is inadmissible for a deterministic wire-
report-plus-snap-history fact). Artifact
`artifacts/transaction_flags_on_production/deadline_drag/20260905T104011Z/results.json`.

**`suspension_return_rust_on_production` -- small-n, leans positive,
crosses zero, recorded regardless of width per the task's own explicit
instruction.** `probability_positive` 0.60665, week-blocked interval
[-0.4454, +1.0571]. The population is 3 confirmed 6+-game suspension
returns in the ENTIRE 2014-2026 archive (Aldon Smith 2014, Eyioma
Uwazurike 2024, Jameson Williams 2024), 6/4,902 full-schedule flagged
games, and only 3 forced picks flip under the production rule in this
window. This is the task's own predicted shape for this lead ("Small-n by
construction; record regardless") and is recorded exactly that way:
`unresolved_below_power`, no admissible closing ground (interval not
wholly below zero; `no_split_half_reliability` inadmissible for a
deterministic wire-bracket-plus-measured-game-count fact, though a
reliability read at n=3 would not be informative in either direction
regardless). Artifact
`artifacts/transaction_flags_on_production/suspension_rust/20260905T104821Z/results.json`.

Per AGENTS.md's promotion-bar/decision-bar distinction: none of the three
candidates is proposed for production promotion on this single
confirmation look. The decision recorded here is each rotation window
being spent and each finding being kept (not discarded, not treated as
"contains zero therefore negative") for future pooling, exactly as the
taxonomy requires. All three findings, including the degenerate
zero-population read for LEAD-12, are preserved in the weak-signal
registry with their full measured population diagnostics rather than
silently dropped.
