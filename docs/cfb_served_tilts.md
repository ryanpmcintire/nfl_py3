# The six served NFL tilts, replicated on COLLEGE FOOTBALL: predeclaration

Written **before any ATS outcome, cover rate, accuracy delta or sign is
computed on college-football data by this line of work.** Sections 1-9 are the
predeclaration. Section 10 was added after the look and reports what it found;
it changes nothing above it.

This is a **cross-league replication**, not a new NFL look. It spends **no NFL
evaluation window and no rotation window** — CFB is this project's sanctioned
free replication ground, exactly as `docs/cfb_rest_bye_replication.md` and
`docs/cfb_body_clock_replication.md` used it. **A replication on college
football is corroboration of a mechanism, never a second independent NFL
evidence point.** Nothing here changes any served behaviour on the NFL card.

## Closing-grounds taxonomy (binding, restated verbatim per AGENTS.md)

An interval or CI that contains zero is **NEVER** grounds to reject, fail, or
close an experiment. At this evaluator's ~2-point resolution, "contains zero"
is the EXPECTED outcome for a real small signal. Only two grounds ever close a
line of work: (a) **refuted mechanism** — a RESOLVED wrong sign (whole interval
on the wrong side of zero) or zero split-half reliability; (b) **bounded by a
positive control** proven able to detect an effect that size. Everything else
is `unresolved_below_power`: record it with `nfl-ats weak-signals record`,
report `probability_positive`, never the binary "contains zero". The registry
code hard-rejects inadmissible closures; if a record command errors, the
verdict is wrong, not the validator. A promotion threshold governs only what
the docs may CLAIM; it never governs which card is PLAYED, which is expected
value.

Within-week correlation is ZERO (mandated); the primary blocking is
week-blocked. Era readings differ in MAGNITUDE; a weaker era is never absence.

## 1. What is being replicated

Six tilts were served on the NFL card 2026-09-09 on an in-sample composition
(`docs/unserved_tilt_marginals.md`,
`artifacts/unserved_tilt_marginals/20260909T210338Z`). Each one's mechanism and
its NFL flag builder live in `src/nfl_ats/<tilt>_overlay.py`. Each is a
**pick-level, post-prediction transform** of the active model's own forced
pick, so the CFB port is the same shape: take the frozen CFB benchmark model's
own forced pick, apply the identical flip rule, and score the two arms paired.

| NFL tilt | NFL served effect | NFL flag + rule (read from the overlay module) |
| --- | --- | --- |
| bye-edge fade | +0.53 pts, P+ 0.87 | exactly one team off a strict bye (>= 12-day gap to its own immediately preceding game *this season*); if the model's pick IS the bye-holding side, flip away (`bye_edge_fade_overlay.py:143`, `POST_BYE_GAP_DAYS = 12`) |
| forecast cold visitor | +0.40, P+ 0.82 | outdoor AND (away team's climatological home temp − this game's Tuesday-noon forecast temp) >= 25 °F; flip AWAY -> HOME (`forecast_cold_visitor_tilt_overlay.py:150`, `TEMP_GAP_THRESHOLD_F = 25.0`) |
| pass-protection mismatch | +0.40, P+ 0.77 | offense's 4-game pressure-allowed rate top-quartile AND opposing defense's pressure-generated rate top-quartile; back the DEFENSE, flip only off the flagged offense (`pbp08_matchup_flags.py`, `WINDOW_GAMES = 4`, `MIN_WINDOW_OBS = 3`, `MIN_QUANTILE_POOL = 200`, pressure = sack OR qb_hit on dropbacks) |
| interim HC first game | +0.13, P+ 0.93, 6 flips | a team's FIRST regular-season game under a newly appointed interim head coach; flip the pick TOWARD that team (`interim_hc_first_game_tilt_overlay.py`) |
| tank-zone fade | +0.07, P+ 0.63 | weeks 14-18 only; a team whose record places it in the league's BOTTOM TWO league-wide; if exactly one side is flagged and the pick IS that team, flip away (`tank_zone_fade_tilt_overlay.py:153-156`, `OVERLAY_WEEK_MIN = 14`, `OVERLAY_WEEK_MAX = 18`, `TANK_ZONE_SIZE = 2`) |
| precip on high totals | +0.07, P+ 0.63 | outdoor AND kickoff-nearest forecast precipitation probability >= 60% AND this game's `total_line` >= 47; flip AWAY -> HOME (`forecast_weather_kn_precip_high_total_tilt_overlay.py:150-151`) |

## 2. Replicability verdict per mechanism, decided on data availability alone

Every verdict below is a statement about what CFB DATA EXISTS, measured this
session before any outcome was scored. **A data-availability block is a
resource limit, never a mechanism verdict, and never an admissible closing
ground.**

1. **bye-edge fade — REPLICABLE, faithfully.** CFB schedules carry every team's
   full regular-season game sequence with dates
   (`data/cfb/schedules/raw/20260816T162105Z`, seasons 2001-2025). The NFL
   flag is a pure calendar-gap construct and ports verbatim: same 12-day
   threshold, same within-`(team, season)` grouping, same "first game of a
   season has no defined gap, so it is not off a bye" rule.
2. **forecast cold visitor — NOT REPLICABLE on available data.** No local CFB
   snapshot carries any temperature field: `schedules` has 30 columns and none
   is weather (`venue_id`/`venue` only), `team_info` carries `city`/`state` but
   no coordinates, and the play-by-play carries none. CollegeFootballData's
   `/games/weather` endpoint is the only source and it is Patreon-gated:
   **measured this session, it returns HTTP 401** with this project's key.
   Recorded as a coverage block, not a result.
3. **pass-protection mismatch — REPLICABLE ONLY IN DEGRADED FORM.** The NFL
   pressure definition is `sack OR qb_hit` on dropbacks. CFB play-by-play has
   **no QB-hit field**: `type.text` carries `Sack` as a play type but nothing
   for a hit or hurry (measured on
   `data/cfb/pbp/raw/20260816T163313Z/season=2019/plays.parquet`, 53 columns,
   3,354 sacks of 58,774 pass plays in 2019). The port therefore substitutes
   **sack rate on dropbacks** for pressure rate — a strictly narrower
   instrument (roughly a third of NFL pressure volume). Everything else ports
   byte-for-byte: 4-game strictly-prior window, `MIN_WINDOW_OBS = 3`,
   expanding strictly-prior quartile thresholds with `MIN_QUANTILE_POOL = 200`,
   top-quartile allow x top-quartile generate, exactly-one-side-flagged game
   rule, back the defence. **This is declared a DEVIATION, and the cell is
   named `..._sackonly_...` so it can never be mistaken for a faithful port.**
   A null here is as consistent with the narrower instrument as with the
   mechanism.
4. **interim HC first game — REPLICABLE via a coaching-change table that had to
   be fetched.** `data/cfb` holds no coaches source. CFBD's `/coaches`
   endpoint is on the free tier (**measured this session: HTTP 200, 134 records
   for 2019**) and each record carries `hireDate` plus per-season
   `teamId`/`games`. Mid-season changes are visible: 2019 shows Rutgers
   (Chris Ash 4 games; Nunzio Campanile hired 2019-09-30, 8 games) and three
   others. **Declared flag definition, frozen here:** a team-season has a
   mid-season head-coach change when a coach's `hireDate` falls strictly
   between that team's first and last REGULAR-season kickoff of that season;
   the flagged game is that team's first regular-season game kicking off
   strictly after the `hireDate`. `hireDate` is used rather than the `games`
   count because `games` includes bowl games and cannot be aligned to a
   week. CFBD carries no "interim" label, but in college football a
   replacement appointed mid-season IS an interim appointment; that
   equivalence is the declared, disclosed assumption of this cell.
5. **tank-zone fade — NO CFB ANALOGUE OF THE MECHANISM.** The NFL construct's
   causal claim is a draft-position incentive: the bottom two teams league-wide
   gain from losing. College football has no record-ordered draft, so **the
   incentive the NFL flag names does not exist in CFB at all.** Two cells are
   declared, and neither is a replication of the incentive:
   - `tank_zone_literal` — the flag definition ported mechanically (bottom two
     league-wide by prior-weeks record, final five regular-season weeks). Its
     value is as a **falsification probe**: if the same flag pays in a league
     with no draft incentive, the NFL mechanism attribution is wrong. Note the
     incidence differs by construction — bottom-two of ~130 FBS teams is 1.5%
     of the league against 6.25% in a 32-team NFL — and rescaling the flag to
     match that fraction would be re-tuning, which is banned here.
   - `bowl_dead_analogue` — the nearest MOTIVATIONAL analogue, named in this
     task: a team mathematically unable to reach six wins (bowl eligibility)
     with its remaining regular-season games, in the final five weeks. This is
     a **different mechanism** (nothing to play for) and is labelled as an
     analogue, never as a replication of the tank-zone cell.
6. **precip on high totals — NOT REPLICABLE on available data.** Same weather
   block as (2): no precipitation field anywhere in local CFB data, and
   `/games/weather` returns 401. `total_line` IS present in the CFB feature
   table (99.97% of clean-core rows), so the total half of the flag would port;
   the precipitation half has no source.

## 3. Population, frozen before scoring

The XLG-03 canonical CFB benchmark table
(`data/processed/cfb_game_features.parquet`, 12,500 completed FBS-vs-FBS
regular-season games with an orientable spread) restricted to the clean core
`nfl_ats.cfb_benchmark.CFB_CLEAN_CORE_SEASONS` = 2012-2019 + 2021-2025, reused
verbatim, not redeclared. The paired arms come from the frozen walk-forward
benchmark run `artifacts/cfb_benchmark/20260818T115149Z/predictions.parquet`,
rows with `method == "market_residual"` (the ridge arm) — **9,093 clean-core
games**, one row per game. Pushes are excluded by `nfl_ats.clv.pick_correct`
returning NaN on a zero settle margin, never scored as losses.

Rest history for the bye cell is derived from the FULL CFB schedules snapshot
(every appearance, any division), never from this filtered table — a team's
actual previous game is frequently absent from the benchmark subset, the same
distinction `docs/cfb_rest_bye_replication.md` section 4 froze.

## 4. Comparator and arms

* **Baseline arm** — the CFB benchmark model's own forced pick,
  `home_cover_probability >= 0.5` picks home. This mirrors the NFL served
  tilts exactly: each is a transform of the ACTIVE MODEL's forced pick, not a
  new feature in a refit.
* **Candidate arm** — the identical picks with the tilt's flip rule applied,
  in the direction the NFL rule serves, with the same asymmetry. Every tilt
  is asymmetric by construction; none of them ever moves a pick ONTO the side
  the mechanism fades.
* **Metric** — paired `delta_accuracy` = mean(candidate correct) −
  mean(baseline correct), in accuracy points, on the games where both arms
  are graded.

## 5. Grading

CFB carries a usable opener: `spread_open` is present on **99.81%** of
clean-core rows (`docs/cfb_data.md` "Openers" table; measured this session).
Both grades are reported and **the OPENER is primary**, per AGENTS.md ("Grade
the decision at the OPENER. A close-graded number may never veto a play"),
which is a change from `docs/cfb_rest_bye_replication.md`'s close-only
convention and is stated here rather than made silently:

* opener settle margin = `result − spread_open` (primary);
* close settle margin = `result − spread_line`, the median-book close proxy
  (secondary).

Note the benchmark model itself was fit against the close-proxy spread, so the
opener grade asks the pool-relevant question of a close-fit model — the same
asymmetry the NFL opener evaluation already lives with.

## 6. Uncertainty

Week-blocked bootstrap, **20,000 samples, seed 20260821**, via
`nfl_ats.clv.week_blocked_bootstrap` (whole `(season, week)` blocks resampled;
within-week correlation is mandated ZERO, so the block is the week).
Season-blocked secondary on the pooled window. `probability_positive` is
reported for every cell; the binary "contains zero" is never reported.

## 7. Eras

The same split `docs/cfb_rest_bye_replication.md` froze: **2012-2019** and
**2021-2025** (2020 is excluded from the clean core by the benchmark contract:
zero openers, no moneylines). Era readings differ in MAGNITUDE; a weaker era is
never absence.

## 8. Split-half reliability of each flag

Odd/even-week team-season split-half via
`nfl_ats.cfb_qb_dependence.split_half_reliability`, on a team-game long panel
whose metric is that team's own flag indicator for that game. Declared in
advance, because AGENTS.md makes zero reliability one of only two admissible
closing grounds. For the bye cell the same compositional caveat
`docs/cfb_rest_bye_replication.md` section 7 records applies — a season holds a
fixed number of days and games, so a within-season split-half of a schedule
quantity is pushed negative by arithmetic, which is a property of the
instrument, not evidence about the trait.

## 9. Positive control

**Perfect-foresight flips on the same population**, declared before scoring:
on exactly the games each tilt is ELIGIBLE to touch (its flagged set), the
candidate pick is replaced by the realised winning side. That measures what an
effect of maximum size would look like at this cell's own flip volume, and is
the only route to the `bounded_by_control` classification. A tilt whose real
reading is null while its positive control is comfortably resolved is
instrument-bounded at that magnitude; a tilt whose positive control ALSO fails
to resolve proves the cell cannot answer the question at this sample size, and
is `unresolved_below_power` by definition.

## 10. What the look found

Added after scoring. Nothing in sections 1-9 was changed after the numbers were
seen. Artifacts: `artifacts/cfb_served_tilts/20260910T014734Z/results.json`
(primary), `artifacts/cfb_served_tilts/20260910T020641Z/results.json` (the
corrected interim addressing of section 10.2), `reliability_supplement.json`
and `record_commands.json` alongside each. Twenty-four registry rows named
`cfb_served_tilt_*` carry the cells; every one is `unresolved_below_power` with
no closing ground, because no admissible ground applies to any of them.

**Harness validation.** The close-graded baseline arm scores **0.5160 on 8,933
graded games**, reproducing the frozen benchmark's own
`clean_core_headline.market_residual.cover_accuracy` of 0.5159520877644688 on
8,933 cover games exactly. The opener-graded baseline is **0.5178 on 8,925**.
The benchmark arm picks home on 41.67% of games, so it carries the opposite
side-tilt from the NFL arms, which pick home 55-67%.

### 10.1 Opener-graded pooled readings

| Cell | delta (points) | 95% week-blocked | `probability_positive` | eligible / flips | subject side covers |
| --- | --- | --- | --- | --- | --- |
| bye-edge fade | **-0.2577** | [-0.9665, +0.4659] | **0.2395** | 1,497 / 1,029 | 0.5164 |
| protection mismatch (sack-only) | **-0.4034** | [-0.9363, +0.1240] | **0.0673** | 1,149 / 510 | 0.4943 |
| interim HC first game (predeclared) | **+0.0000** | [-0.1300, +0.1261] | **0.5011** | 64 / 38 | 0.4844 |
| interim HC first game (corrected) | **+0.0560** | [-0.0693, +0.1813] | **0.8070** | 59 / 35 | 0.5763 |
| tank zone (literal port) | **+0.0112** | [-0.1011, +0.1267] | **0.5786** | 85 / 27 | 0.4941 |
| bowl-dead (analogue) | **+0.2017** | [-0.1930, +0.6475] | **0.8279** | 734 / 290 | 0.4646 |

"Subject side covers" is the model-free read declared in section 4: the side
the NFL rule names, on the eligible set, against 0.5 under no effect. The fade
cells claim below 0.5; the back cells claim above.

### 10.2 A defect found in the predeclared interim flag, and its correction

CFBD's `hireDate` carries **no time of day**, so it lands at midnight UTC. An
evening game kicking at 00:30 UTC on that date was played the night BEFORE in
US local time — and is usually the game that got the previous coach fired.
Measured on the 81 predeclared events, **13 flag a game kicking within 12 hours
of the hire date**, and the modal gap is 0.5 hours.

That is a pregame-safety violation, not a tuning question, so it was corrected
and BOTH readings are reported. The corrected variant
(`interim_flags_gamecount`) addresses the successor's first game by the
**predecessor's own CFBD game count** — a fired coach never coaches the bowl,
so his `games` value is exactly his regular-season games. It produces 75
events, 60 of them the same game as the predeclared cell. Spot-checked against
known cases: Tennessee 2017 moves from the Missouri loss that ended Butch
Jones to Brady Hoke's actual first game at LSU. Neither addressing is exact —
CFBD carries one hire date per coach, sometimes the permanent-contract date
rather than the interim appointment — so the flag carries a measured
game-addressing uncertainty of roughly 15-20%, disclosed here and in the
registry notes.

### 10.3 Positive controls, and why `bounded_by_control` is inadmissible

| Cell | control delta (points) | 95% | control flips | cell's own half-width | NFL served effect |
| --- | --- | --- | --- | --- | --- |
| bye-edge fade | +8.4034 | [+7.4512, +9.3733] | 750 | 0.716 | +0.53 |
| protection mismatch | +6.1064 | [+5.5069, +6.7255] | 545 | 0.530 | +0.40 |
| interim (corrected) | +0.3361 | [+0.2273, +0.4560] | 30 | 0.125 | +0.13 |
| tank zone (literal) | +0.4818 | [+0.3296, +0.6510] | 43 | 0.114 | +0.07 |
| bowl-dead | +4.0224 | [+2.8984, +5.2329] | 359 | 0.420 | +0.07 |

Every control resolves far from zero, so the harness is not blind. But the
control proves detection at **6-8 points** on the two large cells, against NFL
served effects of **0.4-0.5 points** and a cell half-width of **0.5-0.7
points**. The instrument was therefore never proven able to detect an
NFL-sized effect here, and `positive_control_bound` is inadmissible for every
cell. Each stays `unresolved_below_power`.

### 10.4 Split-half reliability, both instruments

| Flag | within-season odd/even week | across-season odd/even year |
| --- | --- | --- |
| off a strict bye | SB **-1.0964** (n=4,873) | SB **+0.6678** (n=703) |
| top-quartile sack-allowed | SB +0.9097 (n=1,688) | SB +0.4771 (n=140) |
| top-quartile sack-generated | SB +0.8728 (n=1,684) | SB +0.3541 (n=140) |
| interim first game | SB -0.0145 (n=4,873) | SB **+0.1900** (n=703) |
| tank zone | SB +0.6519 (n=1,754) | SB +0.6057 (n=137) |
| bowl-dead | SB +0.8197 (n=1,754) | SB +0.8332 (n=137) |

The bye flag's within-season reading of **-1.0964** is exactly the
compositional artefact section 8 predeclared: a season holds a fixed number of
days and games, so extra rest in one part of the calendar is arithmetically
less rest elsewhere, and the within-season instrument is pushed negative for
ANY schedule quantity. The across-season instrument, which is free of that
constraint, reads **+0.6678** and closely matches the +0.6355 already recorded
on `cfb_rest_bye_edge_home_on_benchmark`. **`no_split_half_reliability` is
inadmissible for every cell here**; the lowest correctly-specified reading is
the interim flag's +0.1900, and even that is above any no-reliability ceiling.

### 10.5 Fidelity disclosures on the ported flags

* **Date convention.** Bye gaps use the UTC calendar date, which is the repo's
  own frozen CFB rest convention (`nfl_ats.cfb_features._rest_base_schedule`
  parses `start_date` with `utc=True` then normalises), so this port is
  consistent with the `rest_diff` column the benchmark already carries.
  Measured sensitivity: 12.8% of CFB games have a different UTC and Eastern
  calendar date, but the >=12-day flag count moves by only 33 of 5,750 (0.57%).
* **Completed games.** The bye sequence uses every regular-season row in the
  snapshot; 12 of 29,238 rows (0.04%) are not completed.
* **Tank-zone ties.** The literal bottom-two cut is nearly arbitrary in a
  130-team league: on average **13.5 teams tie at the cut line**, and the
  NFL flag's final tiebreak is the team name. The cell is reported for what it
  is — a falsification probe with 85 eligible games — not as a measurement of
  the NFL construct.
* **Pressure proxy.** CFB sack rate on competitive dropbacks is roughly a
  third of NFL pressure volume (3,354 sacks of 58,774 pass plays in 2019). A
  null on the sack-only cell is as consistent with the narrower instrument as
  with the mechanism.

### 10.6 Decision line

* **Sign agreement with the NFL served direction, at the opener.** Corrected
  interim HC **agrees** (P+ 0.807 against the NFL's 0.93). Tank-zone literal
  port is a dead heat (P+ 0.579) and was never a mechanism test, since college
  football has no record-ordered draft. Bye-edge fade **disagrees** (P+ 0.240),
  and this is now the SECOND CFB construction to lean that way — the existing
  `cfb_rest_bye_edge_home_on_benchmark` reads -0.0784 at P+ 0.296. Protection
  mismatch **disagrees most strongly** (P+ 0.067), on a degraded instrument.
  The bowl-dead analogue leans with a fade-the-unmotivated-team direction
  (P+ 0.828) but tests a different mechanism.
* **None of that closes anything.** Every interval above contains zero, which
  per AGENTS.md is the EXPECTED shape at this resolution and is NEVER grounds
  to reject. No cell has a resolved wrong sign, no cell has zero reliability,
  and no positive control bounds an NFL-sized effect.
* **The served tilt with no corroboration anywhere is precipitation on high
  totals.** Its two registry rows (`forecast_weather_kn_precip_high_total_full`
  and `_pre2020`) are nested windows of one cell; `wxtot_precip60_top_total`
  re-cuts the SAME 50 flagged games on a tercile total rather than a >= 47
  line, so it is not an independent construction; the tuesday-noon forecast
  archive never captured precipitation, so no second NFL source exists; and
  CFB cannot check it at all. By contrast the cold-visitor tilt, which also
  cannot be checked on CFB, has four genuinely different NFL constructions
  (game-time actual weather, tuesday-noon forecast, kickoff-nearest forecast,
  and an attention-battery cut) all leaning the same way. **Precip-on-high-
  totals is the one to watch first when the 2026 NFL rows arrive.**
