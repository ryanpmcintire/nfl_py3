# ENV-01 payoff screen: forecast-vs-actual weather battery (predeclaration)

Written **before** `scripts/nfl_forecast_weather_screen.py` scores anything.
This is the ENV-01 "payoff screen" the sourcing/build docs deferred: the
actual-weather battery cells (`weather_battery_*`, `weather_followup_*`,
predeclared in `docs/weather_followup.md`) were **explicitly recorded as
upper bounds** on any forecast-time feature (see each cell's
`description`/`classification_evidence` field in `registry/weak_signals.json`
-- "actual-weather mechanism screen, NOT pregame-available; upper bound for a
forecast-time feature"). This doc predeclares the re-measurement of the four
strongest of those mechanisms with the **Tuesday-noon GFS-MOS forecast**
substituted for the game-time actual, on the population that forecast now
covers.

## Binding closing-grounds taxonomy (governs every verdict from this screen)

An interval or CI that contains zero is NEVER grounds to reject, fail, or
close an experiment. At this evaluator's ~2-point resolution, "contains
zero" is the EXPECTED outcome for a real small signal. Only two grounds ever
close a line of work: (1) refuted mechanism -- a RESOLVED wrong sign (whole
interval on the wrong side of zero) or zero split-half reliability; (2)
bounded by a positive control proven able to detect an effect that size.
Everything else is `unresolved_below_power`: record it with
`nfl-ats weak-signals record`, report `probability_positive`, never the
binary "contains zero". The registry code hard-rejects inadmissible
closures; if a record command errors, the verdict is wrong, not the
validator. Every cell below is recorded regardless of sign.

## What this screen is, and is not

This is a **timing re-screen, not an independent discovery**. All four cells
below test the SAME mechanisms already predeclared and recorded in
`docs/weather_followup.md` / `scripts/nfl_weather_battery_screen.py` --
only the information-timing changes (Tuesday-noon forecast instead of
game-time actual). A future pool across the weak-signal registry must not
treat a `forecast_weather_*` cell and its `weather_battery_*`/
`weather_followup_*` sibling as independent evidence points: they overlap in
mechanism and (for 2020-2025) in population.

## Data source (read from `docs/forecast_archive_build.md` and measured this
session against the parquet directly)

- `data/raw/forecast_archive/full_2020_2025/forecasts.parquet`: the
  `tuesday_noon`-cutoff GFS-MOS (model=MEX) archive, one row per REG game,
  2020-2025. **Measured** directly this session: 1,615 rows, `fetch_status`
  `ok` for 1,598 (domestic), `unmappable_international_stadium` for 17;
  `forecast_temp_f`/`forecast_wind_mph` null only on those 17 international
  rows (100% coverage of domestic games). Columns used:
  `forecast_temp_f`, `forecast_wind_mph` (already knots->mph converted),
  `fetch_status`, joined to schedules on `game_id`.
- ~~The pool's picks lock **Tuesday at 12:00** (`docs/pool_edge_plan.md` line
  80), essentially the opener -- so `tuesday_noon` is the correct, and only,
  cutoff mode for a pool-playable feature; `kickoff_nearest` (also built,
  2024 + a 2020-2023 spot check) is NOT pool-playable and is out of scope
  for this screen.~~ **Owner-corrected 2026-08-20:** wrong. Only the pool's
  LINE locks Tuesday at 12:00; our picks are editable up to each game's real
  deadline (**refined 2026-08-20: min(kickoff, Sunday 16:00 ET) -- SNF/MNF
  lock early at Sunday 4pm**) (`docs/pool_edge_plan.md`). `tuesday_noon` is used in this screen
  because it mirrors the grading line's own information set (a legitimate
  reason to screen it), not because it is the only pool-playable cutoff --
  `kickoff_nearest` is, if anything, the MORE pool-playable construction for
  a late-week-refreshed pick, since it validates far tighter against actuals
  (temp r=0.972 vs. 0.897, `docs/forecast_archive_build.md`) and is not
  ruled out of scope by anything about pick timing. This screen's four
  cells were scored against `tuesday_noon` only and were not re-run against
  `kickoff_nearest`; that is a real gap left by this correction, not
  resolved here.
- **Narrower window than the actual-weather originals, disclosed up front**:
  this screen's population is REG **2020-2025** (archive coverage), vs. the
  2009-2025 population the `weather_battery_*`/`weather_followup_*` siblings
  were recorded on. Six seasons of week-blocks instead of seventeen -- the
  season-blocked secondary bootstrap here has far fewer blocks (n<=6) and is
  correspondingly weaker as a robustness check; this is reported, not
  hidden.

## Leakage posture -- the one thing this screen changes

Every cell below is built by taking its `weather_battery_*`/
`weather_followup_*` sibling's exact subset definition and swapping ONLY
the "this game's own weather" term from the schedules parquet's game-time
`temp`/`wind` (actual) to the forecast archive's `forecast_temp_f`/
`forecast_wind_mph` (Tuesday-noon forecast). Nothing else changes:

- `roof` (outdoor/dome/closed) and `surface` stay the schedules parquet's
  own actual-recorded values, unchanged from the parent scripts. This is a
  disclosed, precedented convention already carried by both prior batteries,
  not a new leakage source: for fixed roofs (the large majority of outdoor
  stadiums and domes) roof status is a known-in-advance stadium fact, not a
  forecast; the caveat is narrower than the temp/wind caveat it replaces and
  applies identically to a small number of retractable-roof games whose
  Tuesday-noon roof decision is not yet public. This screen does not resolve
  that narrower caveat -- it is inherited, not introduced.
- `away_modal_roof` / `away_modal_surface` (cells 2 and the surface half of
  nothing new here -- surface itself is not gapped in this battery) and the
  `climate_temp` away-team-own-climate baseline (cell 4) are same-season
  aggregates over the away team's OTHER home games' ACTUAL weather -- exactly
  the `weather_followup_screen.py` convention (not season-causal, disclosed
  there, unchanged here). Only the FOCAL game's own temp/wind term swaps to
  forecast; the climatological baseline a game is compared against is still
  built from actual weather at other games.
- `forecast_temp_f` / `forecast_wind_mph` themselves ARE genuinely
  pregame-available before kickoff (and, for this specific
  `tuesday_noon`-cutoff construction, at the pool's Tuesday-noon LINE lock
  too -- **owner-corrected 2026-08-20:** that lock constrains the grading
  line, not our own pick timing, which runs up to each game's real
  deadline, min(kickoff, Sunday 16:00 ET) -- SNF/MNF lock early at Sunday
  4pm)
  -- this is the whole
  point of the screen: these four cells, unlike their actual-weather
  siblings, are candidate POOL-PLAYABLE features, not mechanism-screen upper
  bounds, to the extent the roof/surface/climate-baseline caveats above do
  not apply.

## The 4 predeclared cells (mirrors, exact subset definitions, frozen before scoring)

All cells score `home_cover` (pushes dropped, `add_ats_outcomes`) on REG
2020-2025 games with a successful (or attempted) forecast join. Week-blocked
joint bootstrap primary (block=`season*100+week`), season-blocked secondary
(block=`season`), 20,000 samples, seed `20260819` -- same method, sample
count, and seed as both prior batteries, for direct comparability.
Full-slate-scaled effect (`raw_gap_pts * fraction_of_slate`), reused
verbatim from `scripts/nfl_weather_battery_screen.py`.

1. **`forecast_weather_high_wind_outdoor`** -- mirrors
   `weather_battery_high_wind_outdoor` (registry: +0.1585pts, 95%
   [-0.2863, +0.6037], P+ 0.7567, n=4,317 REG 2009-2025). Flag: outdoor/open
   roof AND `forecast_wind_mph >= 15`. Unsigned (no predicted direction --
   same as the sibling).

2. **`forecast_weather_dome_team_outdoors_cold`** -- mirrors
   `weather_battery_dome_team_outdoors_cold` (registry: +0.1052pts, 95%
   [-0.118, +0.3264], P+ 0.8249). Flag: away team's modal home roof this
   season is dome/closed AND this game is outdoor/open AND
   `forecast_temp_f <= 40`. **Predicted: positive home_cover edge.**

3. **`forecast_weather_warm_team_cold_late`** -- mirrors
   `weather_battery_warm_team_cold_late` (registry: +0.1576pts, 95%
   [-0.0043, +0.3094], P+ 0.9723, the strongest first-generation cell other
   than surface-switch). Flag: away team's season code in the static
   warm-winter-metro list AND outdoor AND `forecast_temp_f <= 35` AND
   `week >= 13`. **Predicted: positive home_cover edge.**

4. **`forecast_weather_temp_gap_cold_visitor`** -- mirrors
   `weather_followup_temp_gap_cold_visitor` (registry: +0.3836pts, 95%
   [+0.0017, +0.7541], P+ 0.9755 -- the strongest follow-up-battery cell, and
   the only registry sibling among these four whose recorded interval
   already excludes zero). Flag: away team's own climatological-normal
   OUTDOOR home temp this season (actual, same-season aggregate) MINUS this
   game's `forecast_temp_f` `>= 25`F, AND outdoor. **Predicted: positive
   home_cover edge.**

## Diagnostic: forecast-vs-actual flag agreement (per cell, not a registry field)

For each cell, also compute the SAME subset definition using the game's
ACTUAL temp/wind (the sibling's original construction) on the identical
2020-2025 population, then report:

- the agreement rate (fraction of rows where the forecast-based flag and the
  actual-based flag agree), restricted to rows where BOTH flags have their
  required inputs present;
- the confusion breakdown (both-true / both-false / forecast-only /
  actual-only counts);
- the same-population actual-weather effect/interval/P+ (computed here, not
  looked up from the registry) alongside the forecast-weather effect, so the
  "playable fraction" (forecast full-slate effect / actual full-slate effect
  on the IDENTICAL 2020-2025 games) is measured on matched populations, not
  confounded by the registry sibling's broader 2009-2025 window. The
  registry sibling's original 2009-2025 number is reported alongside as a
  second, disclosed-as-broader-population comparison.

This diagnostic is exploratory/descriptive, not itself recorded to the
registry as a signal (it has no predicted direction of its own -- it is an
information-timing decay measurement, not a home_cover mechanism).

## Recording commitment

Every one of the 4 cells above records to `registry/weak_signals.json` via
`nfl-ats weak-signals record` under a `forecast_weather_*` name (checked for
collisions against the current registry before this doc was written -- none
exist), `league=nfl`, `effect_units=accuracy_points`, `unresolved_below_power`
regardless of what the interval looks like, via a script that reads the
computed results JSON and passes every numeric field through unmodified (no
hand-typed numbers) -- same discipline as
`scripts/record_weather_followup.py`. The only admissible alternative
classification under AGENTS.md would be a RESOLVED wrong sign (whole
interval on the wrong side of the predicted direction, cells 2-4 only, cell
1 has no predicted direction) or a positive-control bound (not run in this
screen); if neither applies the classification is
`unresolved_below_power`, full stop -- an interval crossing zero is never
itself that ground.

## 2026-08-20 extension: 2009-2019 archive backward + a 6-cell family (predeclaration)

Backlog item 3 from the 2026-08-20 session queue. Written **before** running
`nfl-ats experiment run` on any of the 12 specs below -- the specs
themselves (`registry/experiment_specs/forecast_weather_kn_*.json`) were
committed to disk first, this predeclaration second, and only THEN was
`nfl-ats experiment run` invoked. Results, when this section is next edited,
are appended below the predeclaration, never mixed into it.

### Binding closing-grounds taxonomy (verbatim, governs every verdict below)

An interval or CI that contains zero is NEVER grounds to reject, fail, or
close an experiment. At this evaluator's ~2-point resolution, "contains
zero" is the EXPECTED outcome for a real small signal. Only two grounds ever
close a line of work: (1) refuted mechanism -- a RESOLVED wrong sign (whole
interval on the wrong side of zero) or zero split-half reliability; (2)
bounded by a positive control proven able to detect an effect that size.
Everything else is `unresolved_below_power`: record it with
`nfl-ats weak-signals record` (here, via `nfl-ats experiment run`, which
performs the same mechanical classification -- see
`docs/experiment_pipeline.md`), report `probability_positive`, never the
binary "contains zero". An interval or 0.90 threshold is never a decision
bar for what gets PLAYED (`AGENTS.md`'s "a promotion bar is not a decision
bar"); it only gates what the docs may claim as resolved.

### Why kickoff_nearest, not tuesday_noon -- a disclosed pivot from the task's literal framing

The archive extension was scoped as "extend the Tuesday-noon forecast
archive backward to 2009-2019." **Measured** this session (a live curl probe
against the IEM MOS JSON API, reconfirming `docs/forecast_archive_build.md`'s
prior-session finding): `model=MEX` (the model `tuesday_noon` uses) returns
zero rows for `runtime=2015-09-08T12:00Z` and `runtime=2009-09-08T12:00Z` at
KDFW ("Database query found no results..."), while `model=GFS` (the model
`kickoff_nearest` uses) returns 21 rows for the same 2015 runtime. A
tuesday_noon-cutoff archive genuinely cannot be extended to 2009-2019 with
this free data source -- this is not a new finding, `docs/forecast_archive_build.md`
already flagged it ("tuesday_noon ... still genuinely blocked for
2009-2019"), reconfirmed rather than re-litigated here.

Given that constraint, this extension pivots to **kickoff_nearest** (model
GFS, whose IEM archive reaches back to at least 2005) for the ENTIRE
2009-2025 window, not just 2009-2019 -- a single, internally consistent
cutoff mode across all 17 seasons, rather than splicing kickoff_nearest
(2009-2019) onto the existing tuesday_noon archive (2020-2025) and
disclosing a cutoff-mode seam in the middle of a "full window" comparison.
This is not a downgrade: `docs/forecast_archive_build.md`'s own
2026-08-20 owner-corrected framing already concludes kickoff_nearest is
"the primary feature target for a pool-playable pregame weather feature,
not the fallback" (picks are editable up to each game's real deadline, not
frozen at Tuesday noon) and validates far tighter against actuals (temp
r=0.972 vs. 0.897, MAE 3.05F vs. 7.63F, on the 2020-2025 spot checks already
run). **Consequence, disclosed up front:** none of the 12 cells below are
byte-for-byte reproductions of their tuesday_noon `forecast_weather_*`
namesakes in this same file -- same mechanism, different information
timing AND a different (wider) population -- so every name below carries a
`_kn_` infix and must not be pooled against its tuesday_noon sibling as a
second independent evidence point (same overlap-disclosure convention this
file already uses between the tuesday_noon cells and THEIR
`weather_battery_*`/`weather_followup_*` actual-weather siblings).

### Archive build

`scripts/ingest_forecast_archive.py --start-season 2009 --end-season 2025
--cutoff-mode kickoff_nearest` (model defaults to GFS via
`MOS_MODEL_BY_CUTOFF_MODE`) -- the exact walking/cutoff/station-mapping
machinery from the ENV-01 build, unchanged. Two additive changes made this
session, both backward-compatible (old archives are unaffected, new columns
only):

1. **Field extraction extended to capture precipitation probability**
   (`forecast_precip_prob_pct`, from GFS MOS `p06`, falling back to `p12`).
   **Measured** this session: `p06`/`p12` are populated on only every OTHER
   3h row within a bulletin (the 6h-boundary rows), so the plain
   `nearest_row` pick (nearest by valid time to kickoff, whichever field)
   frequently lands on a null. `nearest_row_with_field` (new function) does a
   second, field-restricted nearest-by-valid-time pick over the SAME
   already-fetched bulletin rows -- no extra HTTP call, no relaxation of the
   point-in-time issuance walk.
2. **`registry/reference/stadium_station_map.csv` extended with 23 historic
   stadium-name rows.** **Measured** this session (a full population-load
   dry run across REG 2009-2025 against the existing table): 23 stadium
   display strings used by seasons before ~2018 are absent from the table
   built for the 2020-2025 window (sponsor-name eras, demolished/relocated
   venues, and international one-off hosts) -- `Candlestick Park`,
   `Cleveland Browns Stadium`, `Cowboys Stadium`, `Dolphin Stadium`,
   `EverBank Field`, `Giants Stadium`, `Hubert H. Humphrey Metrodome`,
   `Invesco Field at Mile High`, `Jacksonville Municipal Stadium`, `LP
   Field`, `Los Angeles Memorial Coliseum`, `Louisiana Superdome`, `Mall of
   America Field`, `New Meadowlands Stadium`, `Qwest Field`, `Ralph Wilson
   Stadium`, `Rogers Centre`, `Sports Authority Field at Mile High`,
   `StubHub Center`, `Sun Life Stadium`, `TCF Bank Stadium`, `Twickenham
   Stadium`, `University of Phoenix Stadium`. 21 of 23 mapped to the SAME
   ICAO station as a modern-era row already in the table (same physical
   building under an earlier sponsor name, or a demolished/relocated
   predecessor at the same site); `Los Angeles Memorial Coliseum` and
   `StubHub Center` (LA Rams'/Chargers' pre-SoFi temporary homes) mapped to
   KLAX (domestic, outdoor, no modern-era sibling row); `Rogers Centre`
   (Toronto, Bills' 2009-2013 international home games) and `Twickenham
   Stadium` (London) were added as `mappable=false`, matching this table's
   existing international convention (CYYZ/EGLL: zero GFS MOS coverage).
   One genuinely new fact worth flagging: `TCF Bank Stadium` (MIN's
   temporary home in 2010 and 2014-2015 while the Metrodome/US Bank Stadium
   were unavailable) is **outdoor**, unlike the dome it substituted for --
   this matters for the `away_modal_roof`/`dome_cold_windy` construction and
   is handled correctly by the schedules parquet's own per-game `roof`
   column (not something the station map itself encodes).

**Coverage, measured this session** (`scripts/validate_forecast_archive.py`
against the FINISHED, complete
`data/raw/forecast_archive/kickoff_nearest_2009_2025/forecasts.parquet`, plus
the ingestion script's own manifest -- the fetch that was still running when
an earlier version of this section reported on an interim checkpoint has
since completed; see "Status" below for that history, kept rather than
deleted per this project's provenance discipline):

- **Per-season coverage: 100% complete, all 17 seasons, every REG game.**
  2009-2020: 256/256 each; 2021, 2023, 2024, 2025: 272/272 each; 2022:
  271/272 (nflverse's own schedule has 271 REG games that season -- the
  Bills-Bengals game cancelled after the Damar Hamlin injury is not replayed
  in the schedule feed, not a fetch gap). **4,431 total REG games in the
  population**, 51 unmappable international (matches the international-venue
  list), 4,380 domestic, **4,379 fetched `ok`** (coverage_of_domestic =
  99.98%), 1 `transport_error` (a single transient failure, not a
  station-coverage gap), 0 `no_bulletin_within_lookback` -- every domestic
  game that could be fetched, was.
- **Instrument validation holds up across the full 17-season window, not
  just 2020+**: temp r=0.964, MAE=3.25°F, bias=-0.55°F; wind r=0.650,
  MAE=3.15mph, bias=+1.40mph (n=2,974 outdoor pairs) -- essentially
  unchanged from the smaller 2009-2015 checkpoint measured mid-fetch (temp
  r=0.965/MAE=3.26°F; wind r=0.660/MAE=3.08mph) and consistent with the
  2020-2025 `kickoff_nearest` numbers already validated in
  `docs/forecast_archive_build.md` (2024: r=0.972/MAE=3.05°F; 2020-2023 spot
  check: r=0.924/MAE=2.35°F). The GFS-MOS kickoff-nearest instrument does
  not degrade going back to 2009.
- Elapsed time for the full fetch: 3,018s (~50 min) at 1.45 games/sec once
  it had the machine's network to itself (measured slower mid-session while
  a concurrent full pytest run and other agents' API traffic shared the same
  connection -- see Status below).
- **Total on-disk footprint** stayed small; `data/raw/**` is gitignored, none
  of this is tracked.

### The 6-cell family (frozen before any effect on 2009-2019 or the full window was computed)

Two RERUNS of the strongest already-recorded tuesday_noon cells, plus 4 NEW
cells declared as one family before signs were seen. All six share the
`_forecast_weather_game_table` construction in `src/nfl_ats/experiment_runner.py`
(one row per REG game, `team_covered`=`home_cover` -- deliberately NOT the
team-long `_base_team_game_table` shape every other builder in that module
uses, to stay numerically faithful to `scripts/nfl_forecast_weather_screen.py`'s
original subset-vs-complement design; see the code comment for the full
derivation of why the team-long shape would silently change what's measured
for a flag with no team-relative framing of its own). Each cell is run on
BOTH the full 2009-2025 window and the 2009-2019-only window separately
(era magnitude matters -- a weaker pre-2020 reading is never reported as
absence, per the owner's magnitude-not-presence rule), giving 12 specs total
in `registry/experiment_specs/forecast_weather_kn_*_{full,pre2020}.json`.
Seed `20260820`, 20,000 bootstrap draws, week-blocked primary / season-blocked
secondary -- same method and sample count as every prior battery in this
file.

1. **`forecast_weather_kn_warm_team_cold_late`** (rerun of
   `forecast_weather_warm_team_cold_late`). Flag: away team's season code in
   the static warm-winter-metro list AND outdoor AND kickoff_nearest
   forecast temp<=35F AND week>=13. Predicted positive home_cover edge.
   `reliability_check.method=not_applicable` (per-game situational
   condition).
2. **`forecast_weather_kn_temp_gap_cold_visitor`** (rerun of
   `forecast_weather_temp_gap_cold_visitor`, the strongest tuesday_noon cell
   -- the only one whose recorded 2020-2025 interval already excludes
   zero). Flag: away team's own climatological-normal outdoor home temp
   (ACTUAL, same-season aggregate) minus kickoff_nearest forecast temp
   >=25F, AND outdoor. Predicted positive home_cover edge.
   `reliability_check.method=not_applicable`.
3. **`forecast_weather_kn_wind_passing_away_favorite`** (NEW). Flag:
   outdoor AND kickoff_nearest forecast wind>=15mph AND the AWAY team is
   both the market favorite (`spread_line<0`, this module's home-perspective
   sign convention) AND that team's PRIOR-season pass-attempt-rate quartile
   is Q4 (most pass-heavy; global qcut(4) over every (team, season) pair
   with a valid year-over-year lag, mirroring `penalty_rate_quartile`'s own
   construction exactly but on `pass_attempt/(pass_attempt+rush_attempt)`
   instead of penalty rate). Predicted positive home_cover edge (a
   pass-heavy road favorite's game plan is disrupted by real wind,
   benefiting the home underdog). Deliberately one-sided (away-favorite
   only) so the flag has a single, unambiguous sign -- a home-favorite
   mirror would need the opposite sign under the same flag name, which the
   generic pipeline's single `sign` field cannot express cleanly.
   `reliability_check.method=split_half`: this is the one cell in the
   family with a genuine persistent per-team trait, so it gets a real
   year-over-year reliability number, unlike every other cell here.
4. **`forecast_weather_kn_precip_high_total`** (NEW). Flag: outdoor AND
   kickoff_nearest forecast precipitation probability (`p06`, falling back
   to `p12`) >=60% AND `total_line`>=47. Predicted positive home_cover
   edge -- the SAME unverified folk mechanism the sibling cold-weather
   cells already carry (home teams conventionally assumed better adapted to
   their own site's weather), not a new assumption; reported as such, not
   as a validated mechanism. `reliability_check.method=not_applicable`.
   Could not be built against the OLD tuesday_noon archive at all (it never
   captured a precipitation field), so this cell has no tuesday_noon
   sibling to mirror.
5. **`forecast_weather_kn_temp_swing_prior_week`** (NEW). Flag: outdoor AND
   `|kickoff_nearest forecast temp - away team's own immediately preceding
   same-season game's actual temp| >= 30F`. Predicted positive home_cover
   edge (a large temperature swing since the visitor's last game is a
   disruption borne only by the visitor; pregame-safe -- only already-played
   games are used for the "prior" side). A team's first game of a season has
   no same-season prior game and the flag is forced False on it (folded
   into the complement, not dropped). `reliability_check.method=not_applicable`.
6. **`forecast_weather_kn_dome_cold_windy`** (NEW). Flag: away team's modal
   home roof this season is dome/closed AND this game is outdoor AND
   kickoff_nearest forecast temp<=32F AND forecast wind>=10mph. A compound,
   stricter version of `forecast_weather_dome_team_outdoors_cold`
   (temp<=40F alone) -- tests whether cold+windy together compounds the
   dome-team disadvantage beyond cold alone; a distinct predeclared
   hypothesis, not a threshold retune of the sibling cell. Predicted
   positive home_cover edge. `reliability_check.method=not_applicable`.

### Recording commitment

Every one of the 12 runs records to `registry/weak_signals.json` via
`nfl-ats experiment run <spec.json>` (no `--dry-run`), which performs the
SAME mechanical classification `nfl-ats weak-signals record` would (see
`docs/experiment_pipeline.md`'s "Mechanical classification" section): the
runner's ONLY self-authorized non-default verdict is `refuted_mechanism`/
`wrong_sign_resolved`, and only when the PRIMARY (week-blocked) interval
sits entirely below zero (predicted-positive cells) AND the widening factor
needed to re-cross zero exceeds 1.099x. Every other outcome -- including a
naive interval that excludes zero but would re-cross under <1.099x widening
-- records `unresolved_below_power`, full stop, regardless of how the
interval looks. Per AGENTS.md, most or all of these 12 are expected to land
there, which is the EXPECTED shape for a real small signal at this
evaluator's resolution, not a negative result.

### Status: complete (superseding the interim checkpoint below)

**Update, same session:** the background fetch this section originally
reported as still-running completed shortly after the interim checkpoint
numbers below were recorded (`built_at_utc: 2026-08-20T11:47:02Z`,
4,431/4,431 rows, elapsed 3,018s). All 12 specs were **re-run with
`--replace`** against the complete, final archive
(`data/raw/forecast_archive/kickoff_nearest_2009_2025/forecasts.parquet`),
and `registry/weak_signals.json` now holds the FINAL numbers, not the
checkpoint ones -- **the "Results" table below is the final, complete-archive
result; read it, not the interim numbers that follow it for the historical
record.**

<details>
<summary>Interim checkpoint numbers (superseded; kept for provenance/audit
trail, not the current registry state)</summary>

The background fetch was still running when this section was first written
(confirmed by file-modification-time checks on its `results.jsonl` --
actively appending, not stalled/dead; other concurrent agent sessions on
this machine were also hitting the same IEM API, the likely reason this
fetch ran slower than the 2024 pilot's measured 1.46 games/sec at that
point). Rather than hold the session idle, a read-only checkpoint was cut
from the in-progress `results.jsonl` covering 2009-2014 complete + partial
2015 (1,576 rows), and the 12 specs were pointed at it. At that point EVERY
cell's `n_flag` was identical between its `_full` and `_pre2020` entries
(both windows drew from the same 2009-2015 pocket of data) --
`warm_team_cold_late` +0.0771/+0.1265 pts (P+ 0.926/0.935);
`temp_gap_cold_visitor` +0.0008/+0.0161 (P+ 0.499/0.521);
`wind_passing_away_favorite` +0.0242/+0.0388 (P+ 0.996/0.997, n_flag=4 both
windows -- flagged as the most fragile-looking read in the family, correctly
as it turned out); `precip_high_total` -0.0088/-0.0122 (P+ 0.409/0.414);
`temp_swing_prior_week` +0.0521/+0.0931 (P+ 0.705/0.722); `dome_cold_windy`
-0.0098/-0.0144 (P+ 0.418/0.418). Comparing these against the final numbers
below is itself informative (see per-cell notes) -- this is why the interim
numbers are kept rather than deleted.

</details>

### Results (measured this session, FINAL -- complete 2009-2025 archive, 4,431 games)

All 12 recorded to `registry/weak_signals.json` (via `--replace` over the
interim checkpoint entries), verified present via `nfl-ats weak-signals
status` (12/12 found by name). Every one classified `unresolved_below_power`
-- the runner's mechanical classifier never self-authorizes anything else
without a resolved-below-zero interval past the 1.099x widening bar, and
none of these 12 qualify (several sit entirely ABOVE zero, a strong
confirming read, not a closing ground either -- the runner does not
auto-close on that side, and neither does this write-up).

| cell | window | n_flag/n_total | effect (pts) | 95% CI | P+ | reliability |
|---|---|---|---|---|---|---|
| `warm_team_cold_late` | full (2009-2025) | 65/4,317 | +0.1697 | [+0.0091, +0.3169] | 0.9800 | n/a |
| `warm_team_cold_late` | pre2020 (2009-2019) | 43/2,735 | +0.2285 | [+0.0208, +0.4167] | 0.9848 | n/a |
| `temp_gap_cold_visitor` | full | 245/4,317 | +0.2749 | [-0.1106, +0.6562] | 0.9223 | n/a |
| `temp_gap_cold_visitor` | pre2020 | 176/2,735 | +0.2997 | [-0.2269, +0.8200] | 0.8707 | n/a |
| `wind_passing_away_favorite` | full | 22/4,317 | -0.0176 | [-0.1214, +0.0875] | 0.3668 | 0.3929 (512 pairs) |
| `wind_passing_away_favorite` | pre2020 | 12/2,735 | +0.0434 | [-0.0778, +0.1650] | 0.7915 | 0.3929 (512 pairs) |
| `precip_high_total` | full | 50/4,317 | +0.0832 | [-0.0898, +0.2462] | 0.8324 | n/a |
| `precip_high_total` | pre2020 | 29/2,735 | +0.1086 | [-0.1107, +0.3065] | 0.8406 | n/a |
| `temp_swing_prior_week` | full | 190/4,317 | +0.1233 | [-0.1864, +0.4373] | 0.7794 | n/a |
| `temp_swing_prior_week` | pre2020 | 133/2,735 | +0.1352 | [-0.2844, +0.5629] | 0.7372 | n/a |
| `dome_cold_windy` | full | 20/4,317 | -0.0182 | [-0.1338, +0.0814] | 0.3644 | n/a |
| `dome_cold_windy` | pre2020 | 12/2,735 | -0.0300 | [-0.1551, +0.1016] | 0.3248 | n/a |

`n_flag` now genuinely differs between `_full` and `_pre2020` for every
cell (e.g. `warm_team_cold_late`: 65 vs 43, i.e. 22 additional flagged games
land in 2020-2025) -- this IS now a real full-vs-pre2020 comparison, not the
checkpoint's shared-population artifact. Reading these (labeled per AGENTS.md
-- P+, not "contains zero"; era magnitude, never absence):

- **`warm_team_cold_late` is now the strongest read in the family, on both
  windows.** Full: +0.1697 pts, 95% [+0.0091, +0.3169], P+ 0.980 -- the
  interval EXCLUDES zero. Pre2020: +0.2285 pts, [+0.0208, +0.4167], P+
  0.985 -- also excludes zero, and the point estimate is noticeably larger
  pre-2020 (era magnitude, not absence: the mechanism reads stronger in the
  earlier era, consistent with the checkpoint's own early read once it had
  real, if partial, data). Neither interval clears the runner's own
  `refuted_mechanism` bar (that requires the WRONG sign, not this one) --
  both are strong, unresolved, confirming reads of the rerun's ORIGINAL
  hypothesis (mirrors `forecast_weather_warm_team_cold_late`, registered
  +0.1576/P+0.9723 on tuesday_noon/2020-2025 -- the kickoff_nearest,
  full-history rerun landing at a comparable-or-stronger P+ on a much wider
  population is a real corroboration of that original cell).
- **`temp_gap_cold_visitor` swung from a coin-flip in the checkpoint
  (P+ 0.499/0.521, n_flag=94 both windows) to the strongest point estimate
  in the family on full data** (+0.2749/+0.2997 pts, P+ 0.922/0.871,
  n_flag=245/176). This is exactly the correction flagged as likely when the
  checkpoint numbers were first reported (missing 2020-2025 coverage was
  suppressing this cell specifically) -- confirmed. It now sits closer to,
  though still below, its tuesday_noon sibling's registered
  +0.4305/P+0.9029 (2020-2025) and the `forecast_cold_visitor_tilt`
  challenger's actual-weather ancestor (+0.3836/P+0.9755, 2009-2025) --
  three independent-construction reads of the same underlying mechanism
  (temp-gap cold visitor) now all point the same direction with P+ in the
  0.87-0.98 range, which is the kind of convergence AGENTS.md's "pooling
  sub-signals" section treats as meaningful even though none individually
  clears a 0.95 bar.
- **`wind_passing_away_favorite` -- the checkpoint's "standout" evaporated on
  full data, exactly as its own fragility warning predicted.** Full window
  now reads NEGATIVE and near coin-flip (-0.0176 pts, P+ 0.367, n_flag=22,
  up from 4); pre2020 still reads positive but the interval now crosses
  zero (+0.0434, P+ 0.792, n_flag=12). Neither is refuted (no resolved wrong
  sign), but neither is confirming either -- this was the correct call to
  flag as fragile rather than promote in the earlier write-up, and the
  split-half reliability (0.393, 512 pairs) is real but clearly was not
  enough to rescue a 4-game read. Demoted accordingly below.
- **`precip_high_total` flipped from slightly negative to moderately
  positive** on full data (+0.0832/+0.1086 pts, P+ 0.832/0.841, n_flag
  50/29, up from 11) -- still unresolved (interval crosses zero both
  windows) but a meaningfully stronger, same-direction-as-predicted read
  than the checkpoint suggested.
- **`temp_swing_prior_week` strengthened somewhat** (+0.1233/+0.1352 pts,
  P+ 0.779/0.737, n_flag 190/133, up from 65) -- consistent direction,
  unresolved, a real if modest signal on more data.
- **`dome_cold_windy` stayed negative-leaning and, if anything, weakened
  further** (P+ 0.364/0.325, down from ~0.42 on the checkpoint) -- the
  compounded cold+wind condition continues to NOT outperform the
  cold-only sibling (`forecast_weather_dome_team_outdoors_cold`, P+ 0.8249).
  Neither interval sits entirely below zero (full: [-0.1338, +0.0814]), so
  this stays `unresolved_below_power`, not refuted -- but it is now the
  clearest candidate in this family for "the compounding hypothesis
  specifically doesn't help," a real, reportable negative-leaning finding
  in its own right, distinct from "no data."

### Wiring recommendations, ranked by EV (FINAL -- for the orchestrator; this session did not touch `artifacts/prospective/challengers.json` or wire anything)

**2026-08-20 build-agent update:** recommendations #1 and #3 below are now
**wired** (see `src/nfl_ats/forecast_weather_kn_warm_team_cold_late_tilt_overlay.py`,
`src/nfl_ats/forecast_weather_kn_precip_high_total_tilt_overlay.py`, and their
`artifacts/prospective/challengers.json` entries `forecast_weather_kn_warm_team_cold_late_tilt`
/ `forecast_weather_kn_precip_high_total_tilt`), sharing ONE live
kickoff-nearest/GFS fetch (`fetch_shared_kickoff_nearest_forecasts_fail_open`)
rather than each making its own network call. #2, #4, #5, #6 were left
UNWIRED, per the reasoning already recorded in each numbered item below
(read before assuming a skip needs re-litigating) -- #2 in particular is not
"rejected," it is a flagged follow-up: **retargeting the existing
`forecast_cold_visitor_tilt` challenger's live fetch from tuesday_noon/MEX to
kickoff_nearest/GFS would be the higher-EV move for that mechanism**, but was
deliberately not done in this build (a live-fetch cutoff swap on an
already-registered, already-accruing challenger is a bigger/riskier change
than adding a new dual-tracked one, outside this build's scope). The original
text below is preserved verbatim beneath this note.

Precedent read this session: `src/nfl_ats/forecast_cold_visitor_tilt_overlay.py`
+ its `artifacts/prospective/challengers.json` entry
(`forecast_cold_visitor_tilt`, `status: ACTIVE_PROSPECTIVE`) is the exact
template -- a pregame-safe, FAIL-OPEN, pick-level tilt overlay, dual-tracked
in the prospective challenger ledger, NOT wired into `publishing.py`. Ranked
by expected value (probability_positive above 0.5 favors wiring it, per
AGENTS.md's "a promotion bar is not a decision bar" -- EV, not a 0.90
threshold, is what should drive a challenger-wiring call). **This ranking
supersedes the interim one based on checkpoint data** (`wind_passing_away_favorite`
was ranked #1 there; it drops to the bottom here on full data):

1. **`forecast_weather_kn_warm_team_cold_late` -- highest EV, wire first.
   WIRED 2026-08-20** (`forecast_weather_kn_warm_team_cold_late_tilt`, see the
   build-agent update above).
   Both windows' intervals exclude zero (P+ 0.980 full / 0.985 pre2020) on
   4,317/2,735 games with 65/43 flagged -- the strongest, best-powered read
   in this family. What wiring it would take: a new overlay reusing
   `forecast_cold_visitor_tilt_overlay.py`'s live-temp-fetch machinery
   almost unchanged (same MOS station-mapping/fetch pattern, switch cutoff
   to kickoff_nearest per this cell's registered evidence, swap the flag
   condition for the static warm-metro-team list + week>=13 test -- no new
   data source, no new plumbing beyond the cutoff-mode swap). Its
   tuesday_noon sibling (`forecast_weather_warm_team_cold_late`, registered
   +0.1576/P+0.9723, 2020-2025) is not currently wired as a live challenger
   either, so this would be a first for the mechanism.
2. **`forecast_weather_kn_temp_gap_cold_visitor` -- second-highest EV, but
   likely REDUNDANT with the existing `forecast_cold_visitor_tilt`
   challenger, not a reason to skip -- a reason to consider whether to
   retarget that challenger's cutoff instead of adding a new one.** P+
   0.922/0.871 on full data, the largest point estimate in the family
   (+0.27 to +0.30 pts), and now converges with two other independent
   constructions of the same mechanism (see Results discussion). The live
   `forecast_cold_visitor_tilt` challenger already covers this exact
   mechanism at the tuesday_noon cutoff; the higher EV option here may be
   re-registering that challenger (or a sibling) at the kickoff_nearest
   cutoff -- validated tighter (temp r=0.964 vs. tuesday_noon's r=0.897,
   `docs/forecast_archive_build.md`) and now with a full 2009-2025 measured
   base -- rather than wiring a second, separate challenger for the same
   underlying idea.
3. **`precip_high_total` -- moderate EV, genuinely new mechanism, worth a
   second look before wiring. WIRED 2026-08-20** (`forecast_weather_kn_precip_high_total_tilt`,
   sharing recommendation #1's live fetch -- see the build-agent update
   above). P+ 0.832/0.841, n_flag 50/29 -- a real
   strengthening from the checkpoint's near-null read. No tuesday_noon
   sibling exists (that archive never captured precip probability), so this
   would be a first for a precip-based live signal. What wiring would take:
   a new overlay needs a live precip-probability fetch (`p06`/`p12`,
   `nearest_row_with_field` from `scripts/ingest_forecast_archive.py`, not
   yet ported into any overlay module) plus the game's own live
   `total_line` (already read by the active model's weekly card).
4. **`temp_swing_prior_week` -- lower EV, exploratory.** P+ 0.779/0.737, a
   real but modest signal, genuinely new mechanism (no tuesday_noon
   sibling). Wiring would need the live overlay to also read the away
   team's own MOST RECENT ACTUAL game temp (a new schedules-lookback join,
   not present in any existing overlay) -- more plumbing than #1-#3 for a
   currently weaker read. Worth tracking, not yet worth building.
5. **`forecast_weather_kn_wind_passing_away_favorite` -- DO NOT wire from
   this measurement; demoted from the interim #1 ranking on full data.**
   Full window is now negative/coin-flip (P+ 0.367); pre2020 still positive
   but its interval crosses zero (P+ 0.792). The split-half reliability
   (0.393, 512 pairs) on the underlying pass-rate trait is real, so this is
   not dead -- but the mechanism itself (wind disrupting a pass-heavy road
   favorite) is not currently supported by the full-history read that
   looked strong on 4 games. Worth revisiting with a THIRD predeclared
   window (e.g. 2020-2025-only) to see whether the pre2020 lean is a real
   era effect or noise, before any wiring decision.
6. **`dome_cold_windy` -- not recommended to wire.** P+ 0.364/0.325 on full
   data, weaker than the checkpoint suggested and weaker than its cold-only
   sibling (`forecast_weather_dome_team_outdoors_cold`, P+ 0.8249). Neither
   interval is resolved-wrong-sign, so this stays `unresolved_below_power`,
   not refuted -- but EV clearly does not favor wiring a compounded
   condition that reads weaker than the simpler cell it compounds. The
   reportable finding here is that compounding cold+wind did NOT help,
   which is itself worth keeping in the registry rather than discarding.
