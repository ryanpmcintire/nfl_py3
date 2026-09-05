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
