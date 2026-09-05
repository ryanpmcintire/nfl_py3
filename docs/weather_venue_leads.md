# Phase 12 weather/venue leads: LEAD-36, LEAD-37, LEAD-38

## Closing-grounds taxonomy (binding, verbatim, restated before any scoring)

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
`probability_positive` above 0.5 favours the candidate.

## Source-gap check (measured, done first, before any predeclaration)

**No observed historical precipitation or snow column exists anywhere in
this repo's local data.** Measured against the newest
`data/raw/20260824T115346Z/schedules.parquet`'s full column list:
`away_coach`, `away_moneyline`, `away_qb_id`, `away_qb_name`, `away_rest`,
`away_score`, `away_spread_odds`, `away_team`, `div_game`, `espn`, `ftn`,
`game_id`, `game_type`, `gameday`, `gametime`, `gsis`, `home_coach`,
`home_moneyline`, `home_qb_id`, `home_qb_name`, `home_rest`, `home_score`,
`home_spread_odds`, `home_team`, `location`, `nfl_detail_id`,
`old_game_id`, `over_odds`, `overtime`, `pff`, `pfr`, `referee`, `result`,
`roof`, `season`, `spread_line`, `stadium`, `stadium_id`, `surface`,
`temp`, `total`, `total_line`, `under_odds`, `week`, `weekday`, `wind` --
none names precipitation, rain, or snow in any form. A repo-wide grep for
`precip`/`snow` under `data/raw` and `data/processed` (excluding this
session's own new scratch files) found no matching parquet column either.

The only locally available substitute is
`nfl_ats.forecast_weather_features.FORECAST_WEATHER_COLUMNS`'s
`forecast_precip_prob_pct` -- a FORECAST probability of precipitation, not
an observed rain/snow outcome, built from the validated `pool_decision`
archive (`data/raw/forecast_archive/pool_decision_2009_2025/`). Its
manifest (`manifest.json`, read this session) declares `start_season:
2009`, `end_season: 2025`, 4,431 rows, `fetch_status_counts`
`{"ok": 4380, "unmappable_international_stadium": 51}` -- the archive's
full manifested window, comfortably past the task's own "3 scored
seasons" bar for using it as a disclosed screened proxy rather than
stopping at a source-gap note.

**Decision, per the task's own instruction:**

- **LEAD-37 (rain-on-grass fumble chaos) is SCREENED** using
  `forecast_precip_prob_pct` as a disclosed proxy (grass surface AND
  forecast precip probability >= 60%, underdog direction), since the
  archive covers the assigned rotation window.
- **LEAD-38 (snow-game home preparation) is NOT screened.** A forecast
  precipitation *probability* cannot distinguish rain from snow (no
  temperature-conditioned precip-type field exists in the archive or
  anywhere else locally), so there is no proxy at all for "a snow game" --
  only a genuinely observed precipitation-type column (or a much larger
  , currently unbuilt, temp-conditioned forecast join) would let this be
  screened rather than skipped. **What would be needed:** either (a) an
  observed-precipitation-type join (a historical METAR/ASOS station
  archive keyed to each stadium and kickoff hour, giving actual
  precipitation type, not just probability), or (b) combining the
  existing `forecast_precip_prob_pct` with `forecast_temp_f`/
  `observed_temp_f` at a cold threshold as a cruder proxy for "snow
  rather than rain" -- not attempted this session since the task
  instructed a source-gap note rather than inventing a compound proxy for
  this specific lead, and a temperature-gated precip-probability compound
  was not predeclared before this measurement.

## LEAD-36: open-corner stadium wind (predeclared BEFORE scoring)

**Mechanism.** Buffalo/Chicago/Foxboro-style open-end/open-corner stadium
geometries expose the field to sustained ambient wind more than a mostly
enclosed bowl or a dome does; sustained wind >= 15 mph is hypothesised to
suppress passing offense more at these venues than the venue-blind wind
cells already measure, and a suppressed passing game is hypothesised to
help the underdog (chaos/variance favors the side priced to lose).

**Predeclared direction.** Take the UNDERDOG (at the Tuesday opener) in a
qualifying high-wind game at a frozen open-corner venue.

**Disclosure, stated up front.** `wind`/`roof` are the schedule's own
OBSERVED, game-time-actual columns -- NOT a pregame forecast. This is
identical to the data convention already used by the two "venue-blind"
wind cells this task named as this lead's parent (both already recorded
in `registry/weak_signals.json`, both from
`scripts/nfl_weather_battery_screen.py`):

- `weather_battery_high_wind_outdoor` -- outdoor/open roof AND wind >= 15
  mph, unsigned. Effect +0.1585 accuracy points, 95% [-0.2863, +0.6037],
  P+ 0.7567, n=4,317 games/294 weeks (**read**,
  `registry/weak_signals.json`).
- `weather_battery_high_wind_road_favorite` -- outdoor AND wind >= 15 mph
  AND away team favored, predicted home_cover edge. Effect -0.0979 points,
  95% [-0.3407, +0.1467], P+ 0.2176, same population (**read**,
  `registry/weak_signals.json`).

Both are disclosed there verbatim as "an actual-weather mechanism screen
(game-time actuals, not pregame-available)... upper bound for a
forecast-time feature." LEAD-36 runs the identical methodology, adding
only the open-corner VENUE interaction those two cells never tested (they
are venue-blind). It is a mechanism/magnitude question, never a claim of
a deployable pregame feature; the column is never wired into
`MODEL_FEATURE_COLUMNS` and never promoted to a live weekly card.

**Frozen open-corner venue list (JUDGEMENT CALL, disclosed).** Exactly the
task's own given anchor list -- BUF, CHI, NE, CLE, GB, PIT, KC, DEN,
NYJ/NYG (MetLife only), PHI. No further venue was added: a repo-wide grep
for "open corner"/"open end" stadium geometry found only one general essay
(`docs/new_lead_classes_20260826.md` section 6, sun-glare/building
orientation -- a different mechanism, not a wind-venue catalogue) and no
other locally documented open-corner list. Gated on this game's own
`stadium_id` (each team's own PRIMARY, most-frequent code), never on
`home_team` alone -- measured against
`data/raw/20260824T115346Z/schedules.parquet`, five of the eleven teams
also host a handful of international/blizzard-relocation one-off "home"
games at a DIFFERENT stadium (BUF at `DET00`/`LON02`; NE at
`FRA00`/`IND00`/`MIN01`/`SFO01`; GB at `DAL00`/`LON02`; CHI at `LON02`;
CLE at `LON01`; KC at `LON00`/`MIA00`/`FRA00`/`VEG00`; DEN at
`NYC01`/`SFO01`; PHI at `PHO00`/`SAO00`/`NOR00`), none of which is the
open-corner design this lead is about:

| team | frozen stadium_id | n games at that code (2009-2026) |
|---|---|---|
| BUF | `BUF00` | 148 |
| CHI | `CHI98` | 151 |
| NE | `BOS00` | 164 |
| CLE | `CLE00` | 146 |
| GB | `GNB00` | 155 |
| PIT | `PIT00` | 154 |
| KC | `KAN00` | 162 |
| DEN | `DEN00` | 156 |
| NYJ/NYG | `NYC01` (MetLife, 2010+) | 279 combined |
| PHI | `PHI00` | 157 |

NYJ/NYG are restricted to `NYC01` specifically (per the task's own
"MetLife" qualifier); their earlier shared Giants Stadium (`NYC00`,
pre-2010, 8 games each) is excluded, mirroring
`nfl_ats.schedule_flag_features.NEW_STADIUM_HONEYMOON_SEASONS`'s own use
of `NYC01` for the identical venue.

**Encoding.** Signed `open_corner_wind_dog_flag`: `+1` when the HOME team
is the underdog at the Tuesday opener AND this game qualifies (a frozen
open-corner venue, this game's own roof is outdoors/open, AND this game's
own wind is >= 15 mph); `-1` when the AWAY team is the opener underdog AND
the game qualifies; `0` otherwise. Family `open_corner_wind_dog_on_production`,
opener-graded, built in `src/nfl_ats/weather_venue_flag_features.py`
(`derive_open_corner_wind_dog_features`/`attach_open_corner_wind_dog_features`),
run via `scripts/weather_venue_flags_on_production.py --candidate
open_corner_wind`.

**Population, measured** (against
`data/raw/20260824T115346Z/schedules.parquet`, before any opener-store
join): 1,673 games at a frozen open-corner venue (all recorded outdoors,
i.e. none of these ten stadiums has ever recorded a dome/closed-roof
game in this window), 1,521 of those have a recorded wind value, 227
clear the >= 15 mph threshold (213 REG, 6 WC, 6 DIV, 2 CON). Season
range 2009-2025, 6-21 qualifying games per season, no obvious trend.
Per-venue count of the 227: CLE 49, PHI 31, BUF 30, NYJ/NYG 26, KC 25, CHI
18, GB 16, PIT 13, NE 11, DEN 8.

**Leakage.** No column read here depends on this game's own SCORING
outcome (`result`/`home_score`/`away_score`/`spread_line` are never read
by the derivation); the leakage test in
`tests/test_weather_venue_flag_features.py` shuffles those columns and
confirms the flag is unchanged. `wind`/`roof` are, however, this game's
own OBSERVED weather -- not knowable at the Tuesday-opener decision
timestamp -- and that non-pregame-safety is disclosed above, not
asserted away by the test.

## LEAD-37: rain-on-grass fumble chaos (predeclared BEFORE scoring)

**Mechanism.** Wet natural (grass) surfaces are hypothesised to increase
ball-security chaos (fumbles, footing) more than a synthetic surface
does; chaos is hypothesised to favor the underdog.

**Predeclared direction.** Take the UNDERDOG (at the Tuesday opener) in a
qualifying rain-forecast grass game.

**Proxy disclosure.** See the source-gap section above: no observed
precipitation column exists locally, so `forecast_precip_prob_pct` (the
validated `pool_decision` forecast archive) is used as a disclosed proxy
-- it measures a forecast PROBABILITY of precipitation at the pool's real
decision timestamp, not whether rain actually fell. "ATS chaos framing is
new" per ROADMAP's own note: the existing `forecast_weather_kn_*` precip
cells (`docs/weak_stack_v4.md`) test a TOTALS-market tilt, a different
market, never poolable with this ATS-underdog construct.

**Encoding.** Signed `rain_on_grass_dog_flag`: `+1` when the HOME team is
the underdog at the Tuesday opener AND this game qualifies (this game's
own surface normalizes to grass, per
`nfl_ats.surface_switch_tilt_overlay.GRASS_SURFACES`, AND this game's own
`forecast_precip_prob_pct` >= 60); `-1` when the AWAY team is the opener
underdog AND the game qualifies; `0` otherwise. Family
`rain_on_grass_dog_on_production`, opener-graded, built in
`src/nfl_ats/weather_venue_flag_features.py`
(`derive_rain_on_grass_dog_features`/`attach_rain_on_grass_dog_features`),
run via `scripts/weather_venue_flags_on_production.py --candidate
rain_on_grass`.

**Population, measured** (schedule join to the forecast archive, before
any opener-store join): 2,734 grass-surface games in the schedule, 2,451
of those have a resolved forecast precip probability (the archive is
REG-season only, per its own manifest), median precip prob among grass
games is 1%, mean 11.6%. 154 games clear the >= 60% threshold, ALL REG
season, spread fairly evenly across 2009-2025 (5-13 per season).

**Leakage.** No column read here depends on this game's own scoring
outcome. `forecast_precip_prob_pct` is genuinely pregame-safe by
construction -- the archive's own loader
(`nfl_ats.forecast_weather_features.load_forecast_archive`) verifies every
consumed row's `issuance_runtime_utc <= decision_cutoff_utc ==
min(kickoff, Sunday 16:00 America/New_York)` before returning any value,
so this candidate, unlike LEAD-36, does not need the observed-weather
disclosure.

## LEAD-38: snow-game home preparation -- source-gap, NOT screened

See the source-gap section above. No observed precipitation-TYPE source
(rain vs. snow) exists locally, and `forecast_precip_prob_pct` alone
cannot distinguish the two, so no proxy exists for "a snow game"
specifically. Stays unscreened this session; what would be needed is
listed above.

## Measured results (both candidates, 2026-09-05)

Both run via `scripts/weather_venue_flags_on_production.py`
(`--features artifacts/backups/20260905_pre_eng39/game_features_weak_stack.parquet`
-- the frozen pre-EGG-39-rebuild copy, since another lane is concurrently
rewriting `data/processed/*.parquet` this session), all three modes in the
foreground, rotation window `[2020, 2021]` (opener grade,
`--acknowledge-mined` since the opener pool sits entirely inside
2018-2025), 456 paired games / 35 weeks / 2 seasons for both (the same
opener-store-limited population every sibling on-production candidate this
session shares).

**Positive control (both candidates, identical harness/population):**
+44.298 accuracy points, week- and season-blocked `probability_positive`
**1.000** both blockings -- harness proven sensitive before either screen
was read.

| Family | Verdict | Effect (pts, production rule) | Week 95% CI | P+ | Sign-rule effect / P+ |
|---|---|---|---|---|---|
| `open_corner_wind_dog_on_production` | unresolved | -0.2193 | [-1.7316, +1.3514] | 0.32955 | +1.535 / 0.902 |
| `rain_on_grass_dog_on_production` | unresolved | +0.6579 | [-1.7058, +2.8446] | 0.69175 | +0.439 / 0.635 |

**LEAD-36 (`open_corner_wind`).** Production-rule delta -0.2193 accuracy
points, week-blocked 95% [-1.7316, +1.3514], `probability_positive`
0.32955, season-blocked 95% [-0.4237, 0.0000] P+ 0.0 (11 of 456 picks
flipped vs. baseline). The PLAIN sign rule on the identical population
reads the OPPOSITE lean: +1.535 accuracy points, P+ 0.902 -- the two
decision rules disagree at this size, same pattern already seen this
session on `sept_heat_home_on_production` (ROADMAP LEAD-35) and
`home_thursday_on_production` (LEAD-40); reported as disagreeing, not
resolved, with the production rule as the headline per this repo's own
convention (it is the rule actually played). Interval crosses zero on the
production rule -> `unresolved_below_power` per the taxonomy above; NOT a
resolved wrong sign (the interval is not entirely on the negative side)
and no positive-control-bound applies (the harness is proven sensitive,
not proven insensitive to an effect this size). Recorded
`nfl-ats rotation record` (verdict `unresolved`) and
`nfl-ats weak-signals record` (`unresolved_below_power`, registry total
747 after recording). Full-schedule flag rate: 227/4,902 games (4.6%)
across the frozen 10-stadium list; 150 of those 227 lack a resolved
opener-store spread (population diagnostic,
`open_corner_wind_population_diagnostic`). Artifacts:
`artifacts/weather_venue_flags_on_production/open_corner_wind/{20260905T130709Z(null),20260905T131023Z(positive-control),20260905T131348Z(screen)}/`.

**LEAD-37 (`rain_on_grass`).** Production-rule delta +0.6579 accuracy
points, week-blocked 95% [-1.7058, +2.8446], `probability_positive`
0.69175, season-blocked 95% [-0.4237, +1.8182] P+ 0.74545 (21 of 456
picks flipped vs. baseline). The plain sign rule reads the SAME direction,
smaller magnitude: +0.439 accuracy points, P+ 0.635. Interval crosses
zero -> `unresolved_below_power`; `probability_positive` 0.69175 is above
0.5 and favours the candidate per the binding EV rule (a promotion
decision is separate from this screen and not made here). Recorded
`nfl-ats rotation record` (verdict `unresolved`) and
`nfl-ats weak-signals record` (`unresolved_below_power`, registry total
748 after recording). Full-schedule flag rate: 154/4,902 games (3.1%,
grass surface AND forecast precip probability >= 60%); 103 of those 154
lack a resolved opener-store spread. Artifacts:
`artifacts/weather_venue_flags_on_production/rain_on_grass/{20260905T131431Z(null),20260905T131748Z(positive-control),20260905T132107Z(screen)}/`.

Neither candidate is promoted; both rotation windows are now spent for
these two families.
