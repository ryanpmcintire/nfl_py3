# Environmental-exposure archive: air quality + drought, ingestion + coverage

Build-next ranks 3 (EPA AirNow / air quality) and 4 (US Drought Monitor) from
`docs/data_source_scout_v4.md`. **Scope of the original session that wrote
sections 1-6: ingestion + coverage report only.** No experiments were run, no
`weak-signals`/rotation registry entries were written, and no `src/nfl_ats`
code changed in that session. **Update, same day (2026-08-20), a follow-up
session**: both predeclared screens in section 6 were run and recorded --
see **section 7** for the full results table, `probability_positive` per
cell, and the four `environmental_battery_*` weak-signal registry entries.
Sections 1-6 below are left as originally written (historical record); every
claim in them is tagged **measured** (run in the ORIGINAL session, command/path
given), **read** (a file opened that session), **reported** (unverified
secondhand), or **inferred** (reasoning), per the binding AGENTS.md labeling
rule. Section 7's claims carry the same tags, scoped to the follow-up session
that produced them.

## Bottom line

- **Both sources ingested end-to-end via no-auth paths, exactly as
  instructed.** AirNow itself was NOT used (its historical endpoint 401s
  without a free-account key, per `docs/data_source_scout_v4.md` sec 3,
  **reported** from that doc, not re-verified this session since the point
  was to avoid it). Instead: EPA's own pre-generated **AQS** annual
  "daily AQI by county" CSVs (`aqs.epa.gov/aqsweb/airdata/`, zero
  authentication) and the **US Drought Monitor** `CountyStatistics` REST API
  (`usdmdataservices.unl.edu`, zero authentication).
- **Coverage is 100% for both sources across every in-scope domestic game,
  2009 through 2025**, at the home stadium's county, as-of the Tuesday of
  the game's week. 2026 is a different story -- see the staleness section
  below; its "100%" is stale carry-forward, not real coverage, and is
  reported honestly as such rather than folded into the headline number.
- **A real bug was hit and fixed this session**: the drought ingestion
  script's first run hard-failed all 34 counties with
  `SSL: CERTIFICATE_VERIFY_FAILED` against `usdmdataservices.unl.edu` under
  Python's default OS-trust-store lookup -- the same host-specific CA-bundle
  gap `docs/data_source_scout_v4.md` found via `curl -k`, just hitting
  Python's `ssl` module too. An earlier version of this session's own script
  docstring claimed (wrongly, unverified) that Python's certifi-backed
  bundle would sidestep it. Fixed by explicitly building the SSL context from
  `certifi`'s bundle (`ssl.create_default_context(cafile=certifi.where())`);
  **measured** working via a direct `urlopen` test (HTTP 200) before being
  wired into the ingestion script. Flagging this because any future session
  hitting this exact host from Python on this machine will see the same
  failure if it trusts the OS default.
- **Stadium/county reference table reused two existing assets rather than
  re-deriving locations**: `registry/stadium_coordinates.json` (lat/lon,
  built 2026-08-19 for the travel-rest battery, covers every `stadium`
  string in nflverse schedules 2009-2026) and
  `registry/reference/stadium_station_map.csv` (domestic/international
  split, built for the GFS-MOS weather-forecast work). County FIPS codes are
  new this session, resolved via the FCC's public Census Block API (no
  auth), written to `registry/reference/stadium_county_fips.csv`.
- **Dome/retractable-roof games are flagged, not dropped**, using nflverse
  schedules' own `roof` field, which this session **measured** to be fully
  populated 2009-2025 (only the unplayed tail of 2026 is null) --
  contradicting the scout doc's caution that pre-2020 completeness was
  unconfirmed.

## 1. Stadium -> county FIPS reference table

`scripts/build_stadium_county_fips.py` (new this session) joins
`registry/stadium_coordinates.json`'s lat/lon to
`registry/reference/stadium_station_map.csv`'s domestic/international
`mappable` flag, then calls the FCC's Census Block Conversions API
(`geo.fcc.gov/api/census/block/find`, no key) once per unique domestic
coordinate pair. **Measured**: `curl -s
"https://geo.fcc.gov/api/census/block/find?latitude=44.5013&longitude=-88.0622&format=json"`
returns HTTP 200 with `{"County":{"FIPS":"55009","name":"Brown County"},
"State":{"code":"WI",...}}` for Lambeau Field's coordinates. Ran once,
**measured** clean (no errors) across 39 unique domestic lookups.

Output: `registry/reference/stadium_county_fips.csv` -- **82 stadium-name
rows** (every `stadium` string in the schedule 2009-2026, including
franchise-move temporary homes and sponsor-name history), of which:

- **67 rows are in-scope domestic** stadiums, mapping to **34 distinct
  county FIPS codes** (renamed/sponsor-name variants of the same physical
  building share a FIPS, e.g. Heinz Field/Acrisure Stadium both -> Allegheny
  County, PA).
- **15 rows are out-of-scope international** venues (London x4 name
  variants, Munich x2, Mexico City x2, Frankfurt, Sao Paulo, Toronto,
  Melbourne, Madrid, Paris) -- kept in the table with empty `county_fips`
  and `in_scope=False`, not silently dropped.

Each row also carries `roof_values_seen` (the union of nflverse's `roof`
values observed for games at that stadium name) and `season_min`/
`season_max`, both read directly from the schedule rather than hand-curated.

Franchise-move / temporary-home coverage confirmed present: LA Rams'
Edward Jones Dome (STL, pre-2016) and LA Memorial Coliseum (2016-2019)
->SoFi Stadium (2020-); LA Chargers' StubHub Center (2017-2019) -> SoFi
Stadium (shared, 2020-); Las Vegas Raiders' Oakland-Alameda/O.co/Ring
Central Coliseum (pre-2020) -> Allegiant Stadium (2020-); all resolve to
distinct, correct counties (Los Angeles County CA for both SoFi-era teams,
Alameda County CA for old Oakland, Clark County NV for Las Vegas).

## 2. Air quality: EPA AQS daily AQI by county (not AirNow)

`scripts/ingest_air_quality.py` (new this session). Per this session's
instruction to prefer a no-auth path, this deliberately bypasses AirNow's
key-gated historical endpoint (**reported**, `docs/data_source_scout_v4.md`
sec 3: a dummy-key call there returns 401) and instead pulls EPA's own
pre-generated annual archive files from `aqs.epa.gov/aqsweb/airdata/` --
one ZIP per calendar year, covering the whole US, no authentication.
**Measured**: `curl -I https://aqs.epa.gov/aqsweb/airdata/daily_aqi_by_county_2020.zip`
returns HTTP 200, `Content-Type: application/zip`; the unzipped CSV header
is `"State Name","county Name","State Code","County Code","Date","AQI",
"Category","Defining Parameter","Defining Site","Number of Sites Reporting"`.
Years 2009 through 2025 were all **measured** present (HTTP 200) before
building the script; all 17 fetched cleanly on the real run, no failures.

Snapshot: `data/raw/air_quality/20260820T110407Z/` --
`fetched_at 2026-08-20T11:05:31Z`, `index.parquet` = **209,533 county-day
rows** across all 34 stadium counties, 17 annual files under `annual/`,
`manifest.json` with per-run metadata. `counties_with_zero_rows: []`.

### Per-county AQI coverage (2009-01-01 through 2025-12-31)

Every one of the 34 counties has essentially a full daily series (~6,190-
6,209 of a possible 6,209 days; the small shortfalls are real missing-station
days in EPA's own data, not an ingestion gap). Three West Coast counties
lose 2025 specifically: Alameda County CA (old Oakland Raiders site,
retired from the schedule since 2019 anyway), San Francisco County CA
(Candlestick Park, retired since 2013), and Santa Clara County CA (Levi's
Stadium, still active) each stop reporting AQI after 2024-12-31 in this
archive -- **measured**, not investigated further this session (label:
unexplained gap, flagged not guessed at).

## 3. Drought: US Drought Monitor weekly county statistics

`scripts/ingest_drought_monitor.py` (new this session). Source:
`usdmdataservices.unl.edu`'s `CountyStatistics/GetDroughtSeverityStatisticsByAreaPercent`
endpoint, a USDA/National Drought Mitigation Center product, free, no auth.
One request per county for the *entire* 2009-2025 date range succeeded
(no pagination needed) -- 34 requests total, not 34 x 17.

Snapshot: `data/raw/drought/20260820T111232Z/` -- `fetched_at
2026-08-20T11:14:18Z`, `index.parquet` = **30,192 weekly rows** (34 counties
x 888 weeks each, 2008-12-30 through 2025-12-30 -- USDM's own weekly map
grid starts a few days before the requested `1/1/2009` because weeks don't
align to calendar-year boundaries), `manifest.json`, per-county files under
`county/<FIPS>.parquet`. `counties_failed: []` on the corrected run.

**Measured**: USDM's `d0`-`d4` fields are **cumulative** ("at least this
severe"), confirmed by checking Maricopa County AZ's full series: `none +
d0 == 100.0` for every one of 888 rows, and `d0 >= d1 >= d2 >= d3 >= d4`
holds for every row. This matters for the `drought_primary_category`
derived field below (walks from D4 down to D0, first category with >=50%
county-area coverage wins).

## 4. Join: as-of-Tuesday, per (season, week, game), home stadium county

`scripts/build_environmental_exposure_join.py` (new this session). For each
schedule row, `tuesday_date` = the most recent Tuesday on/before `gameday`
(matching the pool's Tuesday-line-freeze checkpoint used elsewhere in this
project as the natural "what was knowable by the frozen-line moment" cut --
these readings are being aligned to that checkpoint as a convenient
reference point, not because AQI/drought are themselves pool inputs).

- **AQI**: `merge_asof(direction="backward")` on the home county's daily
  series -- most recent `date <= tuesday_date`. No extra lag buffer (a
  daily AQI reading is same/next-day in the live system this archive
  descends from).
- **Drought**: `merge_asof(direction="backward")` on `valid_start + 3 days`
  (a conservative publication-lag buffer, since each week's map is
  conventionally released a few days after its own `validStart`, per
  `docs/data_source_scout_v4.md` sec 4).

Output: `data/processed/environmental_exposures/game_join.parquet`
(**4,842 in-scope domestic games**, 2009-2026, REG + playoffs) and
`data/processed/environmental_exposures/out_of_scope_games.parquet`
(**60 international games**, flagged not dropped).

### Per-season join coverage

`merge_asof(direction="backward")` will silently carry the *last available*
archive value forward for any `tuesday_date` past the archive's own max date
-- both archives stop at end-of-2025. A naive "match found" boolean would
therefore report "100% coverage" for 2026 games using stale, carried-forward
late-2025 readings. This report distinguishes **coverage** (any match found)
from **fresh** (match found within a staleness threshold: <=10 days for
AQI, <=17 days for drought, beyond which a match is archive-exhaustion
carry-forward, not a real as-of-Tuesday reading) so the 2026 gap is visible
rather than hidden inside a green number.

| Season | Games | AQI coverage | AQI fresh | Drought coverage | Drought fresh | Outdoor-exposed | Dome/closed |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 2009 | 265 | 100.0% | 100.0% | 100.0% | 100.0% | 197 | 68 |
| 2010 | 265 | 100.0% | 100.0% | 100.0% | 100.0% | 198 | 67 |
| 2011 | 265 | 100.0% | 100.0% | 100.0% | 100.0% | 195 | 70 |
| 2012 | 265 | 100.0% | 100.0% | 100.0% | 100.0% | 194 | 71 |
| 2013 | 264 | 100.0% | 100.0% | 100.0% | 100.0% | 194 | 70 |
| 2014 | 264 | 100.0% | 100.0% | 100.0% | 100.0% | 200 | 64 |
| 2015 | 264 | 100.0% | 100.0% | 100.0% | 100.0% | 202 | 62 |
| 2016 | 263 | 100.0% | 100.0% | 100.0% | 100.0% | 196 | 67 |
| 2017 | 262 | 100.0% | 100.0% | 100.0% | 100.0% | 200 | 62 |
| 2018 | 264 | 100.0% | 100.0% | 100.0% | 100.0% | 199 | 65 |
| 2019 | 262 | 100.0% | 100.0% | 100.0% | 100.0% | 201 | 61 |
| 2020 | 269 | 100.0% | 100.0% | 100.0% | 100.0% | 176 | 93 |
| 2021 | 283 | 100.0% | 100.0% | 100.0% | 100.0% | 200 | 83 |
| 2022 | 279 | 100.0% | 100.0% | 100.0% | 100.0% | 193 | 86 |
| 2023 | 280 | 100.0% | 100.0% | 100.0% | 100.0% | 194 | 86 |
| 2024 | 280 | 100.0% | 100.0% | 100.0% | 100.0% | 182 | 98 |
| 2025 | 285 | 100.0% | **94.7%** | 100.0% | **98.9%** | 193 | 92 |
| 2026 | 263 | 100.0% | **0.0%** | 100.0% | **0.0%** | 173 | 49 |

2009-2024: **fully fresh, no gaps, both sources.** 2025's small fresh-coverage
shortfall is late-December games whose Tuesday falls within the staleness
window of year-end (archives run through 2025-12-31/2025-12-30, so only the
very last week or two of the season is affected). **2026 is 100% stale
carry-forward for both sources and should not be described as "covered" in
any feature-readiness claim** -- see section 5.

### Out-of-scope international games by season

| Season | Int'l games |
|---:|---:|
| 2009-2012 | 2 each |
| 2013-2015 | 3 each |
| 2016 | 4 |
| 2017 | 5 |
| 2018 | 3 |
| 2019 | 5 |
| 2020 | 0 |
| 2021 | 2 |
| 2022-2024 | 5 each |
| 2025 | 0 |
| 2026 | 9 |

60 total, matching `4,902 total schedule rows - 4,842 in-scope = 60`.

### Dome / retractable-roof flag (not a drop)

Read directly from nflverse's own `roof` field, aggregated per physical
stadium via `stadium_county_fips.csv`'s `roof_values_seen`:

- **Fixed dome** (11 stadiums): Allegiant Stadium, Caesars/Mercedes-Benz/
  Louisiana Superdome (New Orleans, one building), Edward Jones Dome,
  Ford Field, Georgia Dome, Hubert H. Humphrey Metrodome/Mall of America
  Field (Minneapolis pre-2016, one building), SoFi Stadium, U.S. Bank
  Stadium.
- **Retractable roof** (8 stadium-name rows / 5 physical buildings): AT&T
  Stadium/Cowboys Stadium (Dallas), Lucas Oil Stadium, Mercedes-Benz
  Stadium (Atlanta), NRG Stadium/Reliant Stadium (Houston), State Farm
  Stadium/University of Phoenix Stadium (Glendale AZ) -- these show both
  `open` and `closed` roof values across their game history.
- **Outdoors** (the remaining 48 stadium-name rows): no roof mechanism at
  all.

Aggregate: **3,487 outdoor-exposed games** (`roof in {outdoors, open}`) vs.
**1,314 dome-or-closed games** (`roof in {dome, closed}`) across all
4,842 in-scope games (the remainder are 2026's not-yet-played, `roof=NaN`
rows). AQI's playing-conditions mechanism is expected to be much weaker
indoors (filtered/conditioned air); drought's turf-hardness mechanism does
not apply at all to a dome played on any surface. **Both sources are
flagged per-game via `is_outdoor_exposed`/`is_dome_or_closed` in the join
output, not dropped** -- a future experiment can restrict to the outdoor
subset explicitly rather than have that filtering silently baked into
ingestion.

### Distribution snapshots (in-scope games, all seasons)

Drought `drought_primary_category` (most severe D-level covering >=50% of
the home county, cumulative-percentage convention):

| Category | Games |
|---|---:|
| none/mixed<50% | 2,737 |
| D0 (abnormally dry) | 892 |
| D1 (moderate) | 676 |
| D2 (severe) | 326 |
| D3 (extreme) | 170 |
| D4 (exceptional) | 41 |

AQI `aqi_category` (EPA's own bucketing of the as-of-Tuesday reading):

| Category | Games |
|---|---:|
| Good | 2,256 |
| Moderate | 2,478 |
| Unhealthy for Sensitive Groups | 88 |
| Unhealthy | 16 |
| Very Unhealthy | 4 |

## 5. Point-in-time honesty: this archive is backtest-safe, not live

**These are retrospective annual/weekly archive files.** Each row's date is
a genuine measurement date, so using it as a pregame feature in a backtest
does not leak future information into an earlier day -- the value itself
was physically true on that calendar day. **This ingestion path cannot
serve a live, in-season 2026 feature today**, for two independent reasons:

1. EPA's AQS annual "daily AQI by county" file for 2026 does not exist yet
   (it is published well after the year completes, following EPA's own
   established annual-archive cadence for this product) -- confirmed by
   this session only requesting/receiving 2009-2025.
2. USDM's per-county API was queried through `enddate=12/31/2025` this
   session; a live query with `enddate` extended to "today" would return
   real current-week data (the API itself has no such gap -- **inferred**,
   not tested this session since it wasn't needed for the 2009-2025
   backtest scope), but the AQI half specifically has no equivalent live
   no-auth path.

**What a live 2026 AQI feed would take**: register a free AirNow API key
(`docs.airnowapi.org`), then a per-stadium lat/long REST pull against
AirNow's current/historical observation endpoints -- mechanically simple
once the key exists (this session already has every stadium's lat/lon
ready in `registry/stadium_coordinates.json`), effort **S-M** per the scout
doc's own estimate. Not done this session because the task explicitly
preferred the no-auth path and scoped this session to backtest ingestion.
A live drought feed needs no new registration -- the same API this session
already used, with a rolling `enddate`.

## 6. Predeclared next-step experiments (NOT run this session)

Per this session's binding invariant, restated verbatim:

> An interval or CI that contains zero is NEVER grounds to reject, fail, or
> close an experiment. Only two grounds ever close a line of work: (1)
> refuted mechanism -- a RESOLVED wrong sign (whole interval on the wrong
> side of zero) or zero split-half reliability; (2) bounded by a positive
> control proven able to detect an effect that size. Everything else is
> `unresolved_below_power`: record it with `nfl-ats weak-signals record`,
> report `probability_positive`, never the binary "contains zero".

Two families are predeclared here -- population, mechanism, and effect
units fixed *before* either is run, per the separate commensurability
discipline in AGENTS.md (pooled inputs must share units/scale/population;
the family must be declared before signs are seen):

**A. High-AQI outdoor games -- visiting-team-from-clean-air gap.**
Population: outdoor-exposed games (`roof in {outdoors, open}`) with a fresh
as-of-Tuesday home-county AQI >= 101 ("Unhealthy for Sensitive Groups" or
worse). **Measured** this session: **44 such games**, 2009-2023 (none found
fresh in 2024-2025 in this archive; the single-digit-per-year rate is
itself worth reporting as a finding, not editorializing about). Mechanism:
a visiting team based in a persistently cleaner-air market may be
disproportionately affected by acute smoke/haze exposure relative to a
home team acclimated to it, or vice versa if the home team is *also*
unacclimated (a genuinely new, discrete-event-driven hypothesis, distinct
from the project's already-built day-of weather actuals). Effect units:
`accuracy_points` (ATS-relevant, forced-pick framing) computed against the
existing strong-market/simple-model baselines already used elsewhere in
this project. n=44 is small; per the crossing-zero rule above, a wide
interval or one crossing zero here is the **expected** outcome for a real
small-sample signal, not grounds to close it.

**B. Drought-category home-field turf effect.**
Population: outdoor-exposed, grass-surface games (`surface` containing
"grass") with a fresh as-of-Tuesday home-county drought category D2
(severe) or worse covering >=50% of the county. **Measured** this session:
**222 such games**, 2009-2025. Mechanism: cumulative soil-moisture deficit
hardens turf, plausibly speeding up play and shifting injury/footing risk
over an accumulation window a single day-of weather reading does not
capture -- mechanistically and administratively distinct from the
already-built weather-actuals battery (USDA/NDMC product, not an NWS/NOAA
feed). Effect units: `accuracy_points`, same baseline convention as (A).

Both families, when eventually run, must record every outcome --
including a small effect whose interval contains zero -- via
`nfl-ats weak-signals record --effect-units accuracy_points
--classification unresolved_below_power` (or `refuted_mechanism`/
`bounded_by_control` only if one of the two admissible closing grounds
above is actually met), reporting `probability_positive` in the writeup
rather than a contains-zero verdict.

## Files

- `registry/reference/stadium_county_fips.csv` -- new reference table, 82
  rows (67 in-scope domestic / 34 counties, 15 out-of-scope international).
- `data/raw/air_quality/20260820T110407Z/` -- `index.parquet` (209,533
  rows), `annual/2009.parquet`..`annual/2025.parquet`, `manifest.json`.
- `data/raw/drought/20260820T111232Z/` -- `index.parquet` (30,192 rows),
  `county/<FIPS>.parquet` x34, `manifest.json`.
- `data/processed/environmental_exposures/game_join.parquet` -- 4,842
  in-scope games with AQI + drought joined.
- `data/processed/environmental_exposures/out_of_scope_games.parquet` --
  60 international games, flagged not dropped.
- `scripts/build_stadium_county_fips.py`, `scripts/ingest_air_quality.py`,
  `scripts/ingest_drought_monitor.py`,
  `scripts/build_environmental_exposure_join.py` -- new in the original
  ingestion session, all idempotent/resumable (skip-if-present unless
  `--force`).
- `scripts/environmental_exposure_battery.py` -- new in the 2026-08-20
  follow-up session (section 7); measure-only, writes
  `artifacts/environmental_exposure_battery/<timestamp>/results.json` via
  `write_experiment_artifact`. Recording each cell in
  `registry/weak_signals.json` is a separate, explicit
  `nfl-ats weak-signals record` call (done this session; see section 7).

## 7. Results: the two predeclared screens (run 2026-08-20, follow-up session)

Both families predeclared in section 6 were run this follow-up session via
`scripts/environmental_exposure_battery.py`. The predeclaration (population
definitions reproduced to match section 6's 44/222 headline counts exactly,
the whole-league-complement method choice, era-split boundary, opener
re-screen plan) was frozen in `<scratchpad>/environmental_battery/
predeclaration.md` **before any effect/interval/P+ was computed**, per the
binding crossing-zero discipline. **Measured** this session unless labeled
otherwise.

**Direction was NOT predeclared for either family.** Section 6.A's mechanism
paragraph hedges explicitly ("...or vice versa if the home team is *also*
unacclimated") and section 6.B's never names a team side at all. Both cells
are therefore scored **exploratory, sign fixed at +1** (raw gap =
`home_cover(subset) - home_cover(complement)`, reported as-is, not spun into
a directional claim the source doc never actually committed to) --
consistent with this project's own precedent for a mined cell with no
predeclared direction (`roof_battery_closed_benign_forecast_vs_open`).

**Method** (frozen before scoring, matching the established mined-battery
precedent exactly so these `accuracy_points` figures stay poolable with the
rest of the registry): one row per game, `home_cover` as the value column,
population = the WHOLE REG-season 2009-2025 game set with a resolved
(non-push) cover and a FRESH (non-stale-carryforward) reading of the
relevant source, complement = "everyone else in the league" (dome games,
turf games, normal-AQI/drought games) -- not a mechanism-restricted control
group, exactly matching `roof_decision_screen.py`'s convention. Week-blocked
(`season*100+week`) joint block bootstrap, 20,000 draws, seed `20260820`.
**REG-only** (matching `nfl_bias_battery_screen.py`/`roof_decision_screen.py`
precedent) means the doc's playoff rows are NOT part of the tested
population -- section 6.A's 44 games (2 playoff: 1 WC, 1 DIV) becomes **42**
here, and section 6.B's 222 games (6 playoff) becomes **212** -- reconciled
explicitly, not silently dropped.

### Close grade (2009-2025 REG)

| Cell | n_flag / n_total | subset cover | complement cover | full-slate effect (accuracy_points) | week-blocked 95% CI | P+ |
|---|---:|---:|---:|---:|---:|---:|
| `environmental_battery_aqi_high_outdoor` | 42 / 4,259 | 59.52% | 48.83% | **+0.1055** | [-0.0378, +0.2541] | 0.9264 |
| `environmental_battery_drought_severe_grass` | 212 / 4,267 | 44.81% | 49.15% | **-0.2155** | [-0.5437, +0.1221] | 0.1035 |

Both intervals contain zero -- **the expected shape for a real small signal
at this evaluator's resolution, per the binding AGENTS.md rule, not grounds
to reject either cell.**

**Era split** (predeclared boundary 2009-2016 / 2017-2025, applied only
where both arms clear the predeclared floor of 5 flagged games):

| Cell | Era | n_flag | full-slate effect | 95% CI | P+ |
|---|---|---:|---:|---:|---:|
| AQI high-outdoor | 2009-2016 | 23 | +0.1425 | [-0.0880, +0.3959] | 0.8909 |
| AQI high-outdoor | 2017-2025 | 19 | +0.0741 | [-0.1123, +0.2515] | 0.7928 |
| Drought severe-grass | 2009-2016 | 130 | -0.5650 | [-1.1066, -0.0198] | 0.0206 |
| Drought severe-grass | 2017-2025 | 82 | +0.0800 | [-0.3208, +0.4706] | 0.6467 |

The AQI cell's two eras agree in direction and rough magnitude. **The
drought cell's two eras disagree in direction** -- 2009-2016's own interval
is entirely negative (excludes zero), 2017-2025 flips positive. This is
reported plainly, per the project's era-magnitude-not-presence convention,
not smoothed into one number. **This does NOT meet `wrong_sign_resolved`**:
there is no predeclared direction for this cell to be resolved wrong
against (the taxonomy's wrong-sign ground requires a predeclared sign that
turned out backwards), and this is one era cut of several on a mined,
uncorrected-multiplicity, non-directional family -- a within-family
disagreement across eras to carry forward, not a resolution one way or the
other.

### Opener grade (2020-2025, paired Tuesday-opener/close REG archive)

Reused `nfl_ats.experiment_runner._opener_graded_features` (imported
read-only, `src/nfl_ats` untouched) against the same population, restricted
to the 1,489-game paired archive intersection. Opener regrading recomputes
`home_cover` off the opener line, which pushed 25 games that were not
pushes at the close -- re-dropped explicitly (1,464 scored after that
second push-drop).

| Cell | n_flag / n_total | subset cover | complement cover | full-slate effect | 95% CI | P+ |
|---|---:|---:|---:|---:|---:|---:|
| `environmental_battery_aqi_high_outdoor_opener` | 9 / 1,457 | 55.56% | 49.52% | +0.0373 | [-0.1830, +0.2475] | 0.6459 |
| `environmental_battery_drought_severe_grass_opener` | 63 / 1,464 | 52.38% | 49.39% | +0.1286 | [-0.4267, +0.6549] | 0.6764 |

**n=9 for the AQI opener cell is very small** (2020-2025 only has 9 flagged
games in the whole paired archive) -- recorded for completeness per AGENTS.md,
not as an informative estimate on its own; direction agrees with the
close-grade full-period cell. The drought opener cell's direction (positive)
disagrees with the close-grade full-period cell (negative) but agrees with
the close-grade 2017-2025 era split, consistent with the opener window
(2020-2025) overlapping that later era.

### Registry recording

All four cells recorded via `nfl-ats weak-signals record`
(`--effect-units accuracy_points`, `--classification unresolved_below_power`
in every case -- no admissible closing ground was met: no predeclared
direction exists to resolve a wrong sign against for either family, and
n=9-212 populations cannot bound a positive control). **Verified present**
via `nfl-ats weak-signals status` immediately after recording (registry
293 -> 297 signals, all four names confirmed, no re-record needed -- no
parallel-writer race lost).

- `environmental_battery_aqi_high_outdoor`
- `environmental_battery_aqi_high_outdoor_opener`
- `environmental_battery_drought_severe_grass`
- `environmental_battery_drought_severe_grass_opener`

Source artifact: `artifacts/environmental_exposure_battery/20260820T113156Z/results.json`
(provenance-stamped via `write_experiment_artifact`, registry row under
`registry/experiments/environmental-exposure-battery/`). Script:
`scripts/environmental_exposure_battery.py`. Predeclaration:
`<scratchpad>/environmental_battery/predeclaration.md`.

**Honest small-n caveat, stated plainly rather than as a reason to discard
anything**: 44/222 (this screen's REG-only 42/212) are small populations by
construction -- both families are rare-event archives (single-digit-per-year
AQI breaches; drought is more common but still a minority of grass games).
Every cell here is mined (2 predeclared families, uncorrected multiplicity),
every interval either contains zero or, in one era sub-cut, sits close to
it. None of that is grounds to close either line of work. Both are recorded
as `unresolved_below_power` and belong in the same pooling family as every
other `accuracy_points` entry in this registry -- worth carrying forward,
not worth a verdict from four cells alone.

## Follow-up commands

```powershell
# Re-run / extend air quality (add --end-year 2026 once EPA publishes it):
.\.tools\uv.exe run --no-sync python scripts/ingest_air_quality.py

# Re-run / extend drought (bump --end-date for a rolling live pull):
.\.tools\uv.exe run --no-sync python scripts/ingest_drought_monitor.py --end-date 12/31/2026

# Rebuild the join after either source refreshes:
.\.tools\uv.exe run --no-sync python scripts/build_environmental_exposure_join.py

# Rebuild the stadium/county reference only if new stadiums/renames appear:
.\.tools\uv.exe run --no-sync python scripts/build_stadium_county_fips.py

# Re-run the two predeclared screens (section 7) after either source refreshes:
.\.tools\uv.exe run --no-sync python scripts/environmental_exposure_battery.py
```
