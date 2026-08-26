# New lead CLASSES — scouted and verified 2026-08-26

## What this document is, and what it is not

The weak-signal registry holds **453 recorded signals** (**measured**, this
session: `registry/weak_signals.json`, `len(d['signals']) == 453`), but they are
roughly two dozen ideas with many variants each — **34** weather entries
(`weather_*` + `forecast_weather_*` + `wxtot_*`), **15** `body_clock_*`, **14**
`era_weighting_*`, **13** `residual_location_*` (**measured**, this session, by
prefix count over the same file). What is scarce here is a new **class**, not a
new variant.

Also **measured** by the same count, and the reason the eight candidates below
qualify as classes: the registry contains **zero** entries whose key mentions
illness, birthplace/hometown, disaster/FEMA, concert, or sun/solar. The four
`venue_milestone_*` entries are home-opener and new-stadium-debut flags, not
anything about venue condition.

The generative principle used, which is the project's own: *the betting market
prices football extremely well; it does not price things that are not football
but affect football.* The project's single most reliable measured trait is
state-level influenza (split-half reliability 0.981 — **reported** from the
parent session's brief, not re-derived here), and flu is not football.

**Nothing in this document has been built, scored, or tested against an ATS
outcome.** That is the predeclared step and it comes later; mining before
predeclaration is what the repository rules forbid. Every claim below is about
*data existence, coverage, cadence, and point-in-time recoverability*. No
candidate here is being closed, rejected, or graded — the binding closing-grounds
taxonomy is not in play because no experiment has been run.

Provenance tags, per `AGENTS.md`:
**measured** = a command run this session, with the command or path given;
**read** = a page or file opened this session;
**reported** = a search snippet or secondhand claim, explicitly unverified;
**inferred** = my reasoning, not evidence.

Four leads are already in flight with other agents and are deliberately absent:
CDC broad respiratory illness, the Pro Football Rumors transaction wire, the
daylight-saving transition week, and the Reddit/Arctic-Shift battery
(**reported**, from the task brief).

---

## Ranked shortlist

| # | Class | Mechanism strength | Data verified? | PIT quality | Cost |
|---|---|---|---|---|---|
| 1 | Roster-level contagion (illness designations) | strong | **yes, measured** | A (per-row `date_modified`) | S |
| 2 | Federal disaster declarations in the team's county | strong | **yes, measured** | A (declaration + incident dates) | S |
| 3 | Venue non-football event load (field condition) | strong | **partial, measured** | A (events announced months ahead) | M |
| 4 | Player geography / homecoming distance | medium | **yes, measured** | A (static facts) | S–M |
| 5 | NFLPA player-graded workplace conditions | medium | **yes, measured** | A, with one hard rule | S |
| 6 | Solar geometry and glare windows | medium-low | **partial, measured** | A (deterministic) | S |
| 7 | Aviation-system disruption on travel day | medium | **yes, measured** | B (indirect linkage) | L |
| 8 | Geomagnetic activity | low / speculative | **yes, measured** | A (definitive index) | XS |
| — | *Unverified section below*: uniform thermal load; betting-legalization regime; rule-enforcement regime | — | **no** | — | — |

---

## 1. Roster-level contagion — the `illness` designation on the injury report

**Mechanism.** A team with four players carrying an `illness` designation on
Friday has a locker-room outbreak, not four independent injuries: respiratory
illness spreads through a shared facility, meeting room and plane, it degrades
players who still dress and play, and it peaks and clears inside one or two
weeks. **Why a sharp book does not have it:** the market's injury adjustment is
calibrated on musculoskeletal injuries, where the relevant question is binary
availability of a valuable player. An `illness — Questionable` player almost
always plays, so the participation-based machinery scores him at roughly zero,
and the *correlated, roster-wide, transient* nature of contagion — the part that
matters — has no representation in that machinery at all (**inferred**; this is
my mechanism argument, not a measurement).

**Why this is a new class, not an injury variant.** Every existing injury entry
in the registry conditions on *severity or value* (`injury_value_lost_*`) or on
*designation counts* (`nflcom_friday_out_count_ge2`,
`nflcom_refresh_out2_starters_on_chain`) — **measured**, from the registry name
listing. None conditions on the *reason*. The unit of analysis is also different:
this is a team-week clustering statistic, not a player-availability sum. And it
is the missing bridge between the project's most reliable measured trait
(ambient state-level ILI) and the actual roster, which is a different construct
from either endpoint.

**Data, VERIFIED.**
- **measured**, this session: `data/raw/nflcom_injuries/20260821T222602Z/injuries.parquet`
  holds 17,483 rows for seasons 2022–2024, and its `injury` column contains
  **409 rows matching "illness"** (348 exactly `illness`) plus 135 rows matching
  not-injury / NIR / personal / rest.
- The repo's nflverse snapshot
  `data/players/raw/20260817T184901Z/injuries.parquet` covers **2009–2024,
  79,818 rows**, but has only 9 columns and **the injury-description columns
  were dropped on ingest** (**measured**: column list is
  `season, game_type, team, week, gsis_id, position, report_status,
  practice_status, date_modified`). The description fields exist upstream.
- **measured**, this session: downloaded
  `https://github.com/nflverse/nflverse-data/releases/download/injuries/injuries_{2010,2012,2016,2020,2023,2024}.parquet`
  (all HTTP 200, 107–140 KB each). Each has **16 columns** including
  `report_primary_injury`, `report_secondary_injury`, `practice_primary_injury`,
  `practice_secondary_injury`, and `date_modified`.

Measured density per season (script:
`scratchpad/ill_probe.py`, run this session):

| season | rows | player-weeks flagged `illness` | team-weeks ≥1 | ≥2 | ≥3 | max | `date_modified` non-null |
|---|---|---|---|---|---|---|---|
| 2010 | 4,491 | 120 | 94 | 22 | 3 | 4 | 4,429 |
| 2012 | 5,533 | 166 | 129 | 22 | 9 | 6 | 5,533 |
| 2016 | 5,115 | 161 | 130 | 23 | 7 | 4 | 5,115 |
| 2020 | 5,661 | 248 | 179 | 52 | 14 | 5 | 5,661 |
| 2023 | 5,599 | 265 | 189 | 54 | 12 | 6 | 5,599 |
| 2024 | 6,215 | 202 | 155 | 39 | 7 | 4 | 6,215 |

2023 alone: 126 `report_primary_injury == illness` and 247
`practice_primary_injury == illness` (**measured**). 2020's elevation is
COVID-era and must be handled as its own regime, not pooled naively
(**inferred**).

**Point-in-time honesty: A, the best in this document.** `date_modified` is a
real per-row UTC revision timestamp (**measured**: `2023-09-08 18:49:43+00:00`
on the first 2023 rows) and is essentially fully populated in every season
sampled. That is exactly the `as_of` field that makes the flu work defensible,
and it means a Wednesday-vs-Friday trajectory is reconstructable rather than
assumed. The practice-report cadence (Wed/Thu/Fri) sits *after* the Tuesday line
freeze and *before* the pick deadline, which is the window this project has
measured value in before.

**Cost: S.** No new source, no new scraper. Re-pull the nflverse injuries release
with all 16 columns (16 files, ~2 MB total), which also fixes a real data loss in
the current snapshot. Hours, not days.

**Caveat, stated up front:** `illness` is a self-reported team designation with
known strategic under-reporting, so the measure is a floor, not a census
(**inferred**). That biases toward attenuation, not toward a false positive.

---

## 2. Federal disaster declarations in the team's county

**Mechanism.** A federally declared disaster in a team's home county — a
hurricane, a wildfire, a flood, an ice storm — disrupts the practice week, the
facility, and the players' own households, and it does so for the *away* team
too, since a team's home county disruption travels with it. **Why a sharp book
does not have it:** books unquestionably price the handful of famous cases where
a game is relocated or postponed. The mass of the distribution is the other kind:
a DR/EM declaration for flooding in Allegheny County during a normal week, where
nothing about the game changes and no line moves. Nobody at a book is joining
FEMA's county table to the schedule (**inferred**).

**Why this is a new class.** There is no natural-disaster, civil-emergency, or
facility-disruption construct anywhere in the 453-signal registry (**measured**:
name listing contains no such family). It is adjacent to the weather family only
at the surface — the operative variable is *declared civil emergency in the
county where the team lives and practices*, most of which is not gameday weather
at the venue at all, and it therefore is not bounded by the weather-oracle
ceiling result (**inferred**; the oracle replaced forecast gameday weather with
actual gameday weather and never touched practice-week civil disruption).

**Data, VERIFIED.** OpenFEMA `DisasterDeclarationsSummaries`, v2, free, keyless.
- **measured**: `GET https://www.fema.gov/api/open/v2/DisasterDeclarationsSummaries?$filter=state eq 'LA' and fyDeclared eq 2021&$top=3`
  → HTTP 200, real rows for `DR-4611-LA HURRICANE IDA` including
  `declarationDate`, `incidentBeginDate`, `incidentEndDate`, `fipsStateCode`,
  `fipsCountyCode`, `designatedArea`, `declarationType`, `incidentType`.
- **measured**, density sweep (script: `scratchpad/fema_probe.py`, run this
  session) over the **34 in-scope stadium counties** already resolved in
  `registry/reference/stadium_county_fips.csv`:
  **387 declaration-county rows with `declarationDate >= 2009-01-01`**, of which
  **149 were declared in September–January**. Declaration types: DR 198, EM 146,
  FM 43. Incident types: Hurricane 92, Biological 68, Severe Storm 57, Fire 53,
  Flood 53, Snowstorm 15, Tropical Storm 14, Severe Ice Storm 12, Winter Storm 11,
  Other 4, Tornado 4, Coastal Storm 3.
- Per-county counts ranged from Lambeau Field's Brown County (3) to the LA
  Coliseum's Los Angeles County (39) — **measured**, full table in the script
  output.
- The 68 `Biological` rows are the COVID declarations and are a separate regime
  that should be excluded or modelled apart (**inferred**).

**Point-in-time honesty: A.** Every row carries both `declarationDate` (the date
the fact became public) and `incidentBeginDate`/`incidentEndDate` (the physical
window). A pregame feature keyed on `declarationDate <= kickoff` is trivially
leak-free, and the richer version keyed on incident-window overlap is defensible
because incident begin dates precede declarations. FEMA also stamps `lastRefresh`
per row, so retroactive amendments are detectable.

**Cost: S.** 34 API calls (already written and run this session), joined to a
county FIPS table that already exists in the repo. The stadium→county mapping
work is done.

---

## 3. Venue non-football event load — field condition as an exogenous variable

**Mechanism.** A stadium that hosted a stadium-tour concert nine days before
kickoff had a stage, flooring and forty trucks parked on the field, and the grass
that comes back is not the grass that was there. Same for an international soccer
friendly on temporary sod, or a second tenant playing on the same surface. Footing
and injury exposure change, and they change *asymmetrically* — a speed-dependent
offense loses more than a power one. **Why a sharp book does not have it:** the
information lives in a concert promoter's calendar, not in any football feed. It
is announced months in advance and is therefore not even a late-breaking edge the
market would have machinery to catch; it is simply in a different industry's
database (**inferred**).

**Why this is a new class.** Registry has 6 `surface_*` (grass vs turf, surface
switches), 6 `roof_battery_*`, and 4 `venue_milestone_*` (home opener, new
stadium debut, post-bye) — **measured**, prefix counts. All treat the venue as a
*static property* or the game as a calendar slot. This treats the field as a
**depreciating
asset with a time-varying condition driven by non-football usage**, which is a
different variable entirely.

**Data, PARTIALLY VERIFIED — this is the weak link and I am flagging it.**
- **measured**: MusicBrainz `ws/2` is live, keyless, 1 req/s.
  `GET /ws/2/place?query=MetLife Stadium&fmt=json` → HTTP 200, place
  `da18a856-a99c-4b1d-ada8-89c3f1bbd989` with coordinates and address.
  `GET /ws/2/event?place=<id>&fmt=json&limit=100` → **63 dated events**,
  each with `life-span.begin`/`end` to the day and often a `time`
  (e.g. `2012-09-19`, `20:29`, Bruce Springsteen; `2014-02-02` Super Bowl XLVIII
  halftime show). Year distribution 2011–2026.
- **measured**, coverage sweep across 18 current NFL venues (script:
  `scratchpad/mb_probe.py`): 7 venues returned real dated event lists —
  Arrowhead 10 (span 1988–2023), AT&T 19 (2011–2025), Lumen 20 (2003–2025),
  Hard Rock 20 (1994–2026), Mercedes-Benz 24 (2018–2025), State Farm 28
  (2008–2026), Bank of America 7 (2020–2026). **11 venues failed**: MusicBrainz
  returned HTTP 503 rate-limit errors for 10 of them, and Lambeau Field returned
  no place match. **M&T Bank Stadium returned a place match with zero events**,
  which is certainly wrong for a Baltimore stadium and proves the catalogue is
  incomplete, not that the venue is quiet.
- **measured**, negative: `concertarchives.org` is behind Cloudflare
  (HTTP 403 on a plain fetch), so the obvious purpose-built source is not
  directly scrapable.
- **inferred**: a serious build backfills from Wikipedia venue "Concerts" and
  "Notable events" sections and from the schedule table itself (shared-tenant
  stadiums, international soccer at NFL venues, college games at NFL venues are
  all derivable from free schedule data), and treats MusicBrainz as one input
  among several rather than the spine.

**Point-in-time honesty: A on the event dates themselves** — a concert on
2023-08-12 is a public, months-in-advance fact and there is no possibility of
hindsight contamination in the date. The honest weakness is **coverage
completeness**, not timing: an incomplete catalogue produces false negatives
(games scored "no recent event" that had one), which attenuates rather than
inflates any effect (**inferred**).

**Cost: M.** Rate-limited crawl of ~40 venue histories at 1 req/s is minutes, but
the Wikipedia backfill and the hand-audit needed to trust coverage is the real
work — a day or two. Recommend the cheap version first: **audit one season by
hand for the ~10 grass venues** and measure catalogue recall before building
anything.

---

## 4. Player geography — homecoming distance

**Mechanism.** A player playing 20 miles from where he grew up is handling ticket
requests, family, and old coaches all week, and is playing in front of people
whose opinion he cares about. The published literature on "homecoming" effects
in sport is real but mixed on sign (**inferred** — I have not opened the papers
this session, so treat the sign as genuinely open). The team-level construct is
the *share of the active roster* within some radius of the venue, which for a few
games a season is a large fraction. **Why a sharp book does not have it:** the
inputs are 53 birthplaces and one venue coordinate, which nobody joins, and the
effect — if real — is small enough that it never becomes a narrative a trader
prices (**inferred**).

**Why this is a new class.** The registry's travel family (`travel_rest_*`,
`body_clock_*`) is about *the trip*: distance flown, time zones crossed, rest
days. This is about *where the players are from*, which is a static player
attribute unrelated to the trip and orthogonal to every existing geographic
feature (**measured**: no birthplace or hometown construct appears in the
registry name listing).

**Data, VERIFIED.**
- **measured**: Wikidata Query Service, keyless SPARQL, HTTP 200.
  `SELECT (COUNT(DISTINCT ?p)) WHERE { ?p wdt:P106 wd:Q19204627 ; wdt:P19 ?pob . ?pob wdt:P625 ?coord . }`
  → **33,918 American-football players with a geocoded place of birth.** Adding
  `wdt:P569` (date of birth) still gives **33,646**, so name+DOB is a viable join
  key at near-full coverage.
- **measured, and this is the important negative**: joining on
  `wdt:P3539` (Pro-Football-Reference player ID) yields only **296** players.
  Do not plan a PFR-ID join; use name + date of birth against nflverse rosters.
- **measured**: Sleeper's keyless `https://api.sleeper.app/v1/players/nfl`
  (14.6 MB, 12,225 players) has **`birth_city` populated for 0 players** — that
  field is empty — but **`high_school` populated for 10,829**, formatted as
  `"Cedar Grove (GA)"`, i.e. **state-level only, no city**. Usable as a coarse
  "playing in his home state" cross-check; not usable for distance.
  `gsis_id` is present for only 3,893 players (**measured**), so Sleeper is not a
  clean bridge to nflverse either.
- Venue coordinates already exist: `registry/stadium_coordinates.json`, 83
  entries with lat/lon/tz/city, covering every `stadium` string in REG games
  2009–2025 (**read**, this session, including its `_README`).

**Point-in-time honesty: A.** Birthplace is a static fact and cannot leak.
Roster membership as of the game week is already solved in this repo by the
participation and roster pipelines. The only PIT care needed is using the roster
*as of that week*, not the season-end roster.

**Cost: S–M.** One SPARQL dump (minutes), one name+DOB fuzzy join to nflverse
rosters (the fiddly part — expect 5–15% unmatched and report the match rate),
then great-circle distance against a coordinate table that already exists.

---

## 5. NFLPA player-graded workplace conditions

**Mechanism.** Players anonymously grade their own employer on eleven operational
categories including **team travel**, **training staff**, **weight room**,
**strength coaches**, **nutritionist**, and **treatment of families**. These are
the exact inputs to recovery and fatigue, measured by the only people who
experience them, and they vary enormously between clubs. **Why a sharp book does
not have it:** it is an HR survey published by a labour union in February. A
book's model has no column for "the visiting team's charter seating is graded
F-", and to the extent the effect works through cumulative fatigue it would show
up in late-season and short-week games rather than as a constant, which is
exactly the shape a season-long power rating absorbs badly (**inferred**).

**Why this is a new class.** Nothing in the registry measures organizational or
workplace quality. It is the closest thing available to a *causal* input to the
travel-and-rest family, rather than another schedule-derived proxy for it
(**inferred**).

**Data, VERIFIED.**
- **measured**: `https://nflpa.com/report-cards/2025` → HTTP 200, 445,954 bytes.
  `https://nflpa.com/report-cards/2025/new-england-patriots` → HTTP 200,
  122,544 bytes, containing per-category labels (`Weight Room`, `Training Staff`,
  `Team Travel`, `Ownership`, `Head Coach`) and letter grades.
- **read**: the 2025 index page states the survey was **administered
  2024-08-26 to 2024-11-20**, that **1,695 players (77% of membership)**
  responded, that this is the **third annual** edition, and lists the 11
  categories: treatment of families, food/dining, nutritionist/dietician,
  locker room, training room, training staff, weight room, strength coaches,
  team travel, head coach, team ownership.
- **measured, negative**: `https://nflpa.com/report-cards/2023` → HTTP 404. The
  2023 and 2024 editions exist (the 2025 page calls itself the third annual) but
  the URL pattern for prior years differs and I did not find it. Budget an hour
  to locate them, or pull them from Wayback.

**Point-in-time honesty: A, but only under one hard rule.** The survey window for
the "2025" card is **inside the 2024 season** (Aug–Nov 2024), and it publishes in
February 2025. Therefore: **the card labelled year S may only be used for season
S onward — never for season S-1**, whose games were being played while the survey
was in the field. Using the 2025 card on 2024 games is leakage. Stated this
explicitly because the year label invites exactly that mistake.

**Cost: S.** 32 team pages × 3 editions ≈ 96 fetches of a plain HTML page.
Coverage is only **3 seasons**, which is thin — that is a power statement, not a
reason to skip it, and the repository rules are explicit that below-power is not
a negative.

---

## 6. Solar geometry — glare and sun-angle windows

**Mechanism.** Sun position at kickoff is fully determined by latitude,
longitude, date and time, and the direction a receiver or a kicker is looking is
determined by the field's compass orientation. Late-afternoon games in stadiums
whose long axis points into the setting sun produce a real, documented glare
window; open-ended stadiums with a west-facing gap are the notorious cases.
**Why a sharp book does not have it:** it is astronomy plus building orientation.
There is no feed. The market prices kickoff time (primetime effects are
well-known) but kickoff time is a terrible proxy for sun angle, because the same
4:05pm kickoff is a glare game in November in Denver and a non-event in September
in Miami (**inferred**).

**Why this is a new class.** Not weather: it does not depend on any forecast and
is unaffected by the weather-oracle ceiling. Not body clock: `body_clock_*`
measures the *player's internal* time, this measures *external illumination
geometry* (**measured**: registry has 15 `body_clock_*` entries and zero keys
mentioning sun, solar or glare — the only `sun` substring matches are
`*_sunday_*` slot flags).

**Data, PARTIALLY VERIFIED.**
- Sun position needs **no network at all** — it is closed-form from lat/lon/UTC,
  and the repo already has lat/lon plus IANA time zones for every venue in
  `registry/stadium_coordinates.json` (**read**).
- Field orientation must come from building geometry. **measured**: Overpass API
  (`https://overpass-api.de/api/interpreter`, keyless) returned MetLife Stadium's
  `leisure=stadium, sport=american_football` way (id 24221553) with **62 geometry
  vertices**, plus a `leisure=pitch, sport=soccer` polygon — enough to compute a
  principal axis.
- **measured, negative**: a follow-up sweep over 12 venues failed. Python
  `urllib` got HTTP 406 without a User-Agent, and after adding one, both `urllib`
  and `curl` hit connection timeouts to `overpass-api.de` — I was rate-limited
  after the successful calls. **Per-venue coverage of explicit
  `sport=american_football` pitch polygons is therefore UNVERIFIED beyond
  MetLife**, where no such polygon existed and the stadium outline had to serve.
  A real build should retry against a mirror (`overpass.kumi.systems`) with
  polite pacing, or download the geometry once from a Geofabrik US extract.

**Point-in-time honesty: A.** Wholly deterministic from the schedule. Kickoff
time is known at the Tuesday freeze; the sun does not revise.

**Cost: S.** One-time pull of ~40 stadium polygons plus ~30 lines of solar math.
The honest expectation is a small effect on a narrow subset of games
(**inferred**) — which is precisely the shape this project exists to accumulate.

---

## 7. Aviation-system disruption on the travel day

**Mechanism.** Teams fly the day before. When the origin or destination airport
is under an ATC ground stop or a systemic delay event, a charter sits on a ramp
for four hours and the team lands at 2am. That is a real, occasional, large
fatigue shock that the schedule-derived travel features cannot see, because on
paper the trip was ordinary. **Why a sharp book does not have it:** the schedule
is priced; the *execution* of the schedule is not, and Saturday-evening delay
data is not something a football model ingests (**inferred**).

**Why this is a new class.** `travel_rest_*` and `body_clock_*` measure the
*planned* trip. This measures whether the trip actually happened as planned —
executed rather than scheduled travel. The previously-scouted JetTip charter
blog (`docs/data_source_scout_v5.md` §C rank 1, **read**) covers *planned*
charter legs; this covers *disruption to them*.

**Data, VERIFIED (with an honest linkage caveat).**
- **measured**: `HEAD https://transtats.bts.gov/PREZIP/On_Time_Reporting_Carrier_On_Time_Performance_1987_present_2023_11.zip`
  → HTTP 200, `Content-Length: 27,573,257`, `Content-Type: application/x-zip-compressed`,
  `Last-Modified: Fri, 02 Feb 2024`. Monthly files, 1987 to present, no auth.
- **measured**: `https://www.transtats.bts.gov/ONTIME/Departures.aspx` → HTTP 200
  (the interactive query UI; the PREZIP monthlies are the sane ingest path).
- **The linkage caveat, stated plainly:** NFL team charters are non-scheduled
  flights and **do not appear** in BTS carrier on-time data (**inferred**, from
  what the BTS reporting-carrier universe is — I did not open a file to confirm
  the absence). The usable construct is therefore **airport-and-date-level
  systemic delay** as a proxy for regional ATC/weather disruption that charters
  share, not the team's own flight. That is indirect, and it partially overlaps
  weather, which is why this sits at rank 7 rather than higher.

**Point-in-time honesty: B.** The *event* (Saturday delays at Newark) precedes
kickoff and was publicly observable that day, so a backtest is legitimate. But
BTS *publishes* monthly with a ~2-month lag, so a prospective feature needs a
different live source (FAA NAS status), and that is a second build. Say B, not A.

**Cost: L.** Roughly 200 monthly files at ~27 MB is about **5.5 GB** to download
and aggregate down to an airport-day delay index. That is the largest ingest in
this document by two orders of magnitude, for the least direct mechanism.

---

## 8. Geomagnetic activity

**Mechanism.** Peer-reviewed work links geomagnetic storm activity to human
physiology — sleep quality, heart-rate variability — and there is a well-known
finance literature applying the same index to asset returns
(Krivelyova & Robotti, "Playing the field: Geomagnetic storms and international
stock markets", **reported** — I did not open it this session). **Why a sharp
book does not have it:** obviously nobody prices the Kp index into a football
spread (**inferred**).

**I am ranking this last on purpose and labelling the mechanism honestly: it is
speculative.** It is included because the data is definitive, complete, and free,
and the whole ingest is one file — it is the cheapest lottery ticket available,
not a considered physiological hypothesis. It should be predeclared as an
*a-priori-weak* arm so its result is read correctly whichever way it lands.

**Data, VERIFIED.**
- **measured**: `https://kp.gfz.de/app/files/Kp_ap_since_1932.txt` → HTTP 200,
  **16,596,042 bytes, 276,598 lines**, one row per **three-hour interval**, from
  1932 through the last row `2026 08 25 21.0` — i.e. current to yesterday. CC BY
  4.0, cite Matzka et al. 2021. (The `kp.gfz-potsdam.de` host 301-redirects to
  `kp.gfz.de`; follow it.)

**Point-in-time honesty: A.** Kp is a definitive published geophysical index with
a fixed three-hourly grid; the nowcast/definitive distinction exists but every
historical value used in a backtest is final.

**Cost: XS.** One file, one parse. An hour.

---

## Unverified — data not confirmed, listed separately as instructed

These have mechanisms I can defend but data I could **not** confirm today. They
are not closed; they are unfinished scouting.

**U1. Uniform choice and thermal load.** The home team elects white or colour and
the visitor takes the other, so "home team wore white in a September heat game"
is a deliberate, well-documented tactic (the Miami September white-jersey trick)
that forces a visitor into dark cloth in high heat. **measured**:
`gridiron-uniforms.com` is live (HTTP 200, 118,957 bytes on the main controller;
a team page for MIA returned HTTP 200, 50,809 bytes) and exposes structured
`controller.php?action=decades&decade=YYYY` and `action=teams&team_id=XXX`
routes. **But**: the `&team=&year=` parameters I guessed did not change the
response (identical 118,957 bytes), and the per-game uniform record appears to be
rendered as images rather than text, so **I did not confirm that per-game jersey
colour is extractable**. Verify before planning any build.

**U2. Betting-legalization and market-composition regime.** When a state launches
online sportsbooks, local retail money floods onto the local team, and books
shade the number to balance it — a distortion in the *line* driven by a
*regulatory calendar*, not by football. **measured, negative**:
`legalsportsreport.com/sportsbetting-bill-tracker/` → HTTP 403 and
`americangaming.org/research/state-gaming-map/` → HTTP 403 (both bot-blocked);
Wikipedia's "Sports betting in the United States" was checked and **read** — it
has a legal/not-legal table but **no launch dates**. No free dated
machine-readable source located. Genuinely promising class, unsolved sourcing.

**U3. Rule-enforcement regime — annual points of emphasis.** Officiating
behaviour shifts when the league declares a point of emphasis, penalty rates move
with it, and the market re-learns over some number of weeks — so the early-season
window each year is the interesting one. **measured, negative**: every
`operations.nfl.com` URL I tried returned HTTP 404
(`/the-rules/rules-changes/`, `/updates/the-rules/2025-rules-change-proposals/`,
`/updates/football-ops/2024-nfl-rules-changes-and-points-of-emphasis/`),
including URLs that appeared in search results — so the site is either moved or
bot-blocked from here. Only **reported** search snippets confirm the pages exist.
`footballzebras.com` is the likely alternative and was not tested.

---

## Considered and set aside, with reasons — so the next session does not re-walk this

- **Personnel movement between opponents** (a player who just left Team A and
  signed with Team B before they play): mechanism is fine, but the natural data
  source is the Pro Football Rumors wire, which is **already in flight with
  another agent** (**reported**, task brief). Dropped to avoid duplication. Note
  that `data/raw/pfr_transactions/20260820T011126Z/` is a **news-article
  archive**, not a structured transaction log (**measured**: it holds
  `index.parquet`, `yearly/`, and a `sample_articles/` directory of JSON news
  items), so a structured transaction feed would be new work either way.
- **Non-injury absence codes** (NIR-Personal, NIR-Rest): **measured** — 2023 has
  573 `practice_primary_injury` rows matching not-injury/personal/rest, and the
  NFL.com archive has 135. Real and cheap, but it is close enough to candidate 1
  that it should ride along as a second arm of the same predeclaration rather
  than compete as its own class.
- **Practice-week weather at the team facility**: I judged this too close to the
  weather family the brief excluded, even though the oracle result technically
  bounds only gameday venue weather (**inferred**). Flagging that the exclusion
  is my judgement call, not a measured bound, in case a later session disagrees.
- **Co-located pro teams' playoff runs, documentary crews in the building
  (Hard Knocks / All or Nothing), lunar phase, local unemployment**: mechanisms
  too thin or samples in the low tens of team-seasons; not worth a class slot
  against the eight above (**inferred**).

---

## Recommended execution order

1 and 2 first, together: both are S-cost, both have A-grade point-in-time
provenance, and candidate 1 needs a data re-pull that fixes an existing column
loss in the repo regardless of what the experiment finds. Then 4 and 5, which are
also small. Then 3, which is the highest-mechanism idea in the document but needs
a coverage audit before it can be trusted. 6 and 8 are cheap enough to run
whenever there is idle capacity. 7 only if the 5.5 GB is otherwise free.

Every one of these must be predeclared before any ATS scoring, and every verdict
must flow through `nfl-ats weak-signals record` — never through prose in a doc.
