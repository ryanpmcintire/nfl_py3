# Travel/rest battery (ENV-03 travel geometry + ENV-04 rest context) — predeclaration

Written **before** `scripts/nfl_travel_rest_battery_screen.py` scores any
cover-rate outcome. Per AGENTS.md, this is a mined/exploratory lead-generation
family: every cell here is predeclared to record `unresolved_below_power`
regardless of interval shape (an interval crossing zero is the EXPECTED
outcome for a real small signal at this evaluator's ~2-point resolution,
never a rejection ground). Method, population, thresholds, and blocking are
locked before any cell's sign is seen — only population/flag-size
diagnostics (never cover-rate outcomes) were examined before freezing this
document, exactly the same discipline `docs/weather_followup.md` used.

## Binding closing-grounds taxonomy (verbatim)

An interval or CI that contains zero is NEVER grounds to reject, fail, or
close an experiment. At this evaluator's ~2-point resolution, "contains zero"
is the EXPECTED outcome for a real small signal. Only two grounds ever close
a line of work: (1) refuted mechanism — a RESOLVED wrong sign (whole interval
on the wrong side of zero) or zero split-half reliability; (2) bounded by a
positive control proven able to detect an effect that size. Everything else
is `unresolved_below_power`: record it with `nfl-ats weak-signals record`,
report `probability_positive`, never the binary "contains zero". The registry
code hard-rejects inadmissible closures; if a record command errors, the
verdict is wrong, not the validator. Every cell below is recorded regardless
of sign.

## ROADMAP status (read 2026-08-19, before this work)

- `ENV-03` — ⬜ unbuilt — "Travel geometry: Distance, time-zone change,
  international games, return travel"
- `ENV-04` — ⬜ unbuilt — "Rest context: Bye, short week, mini-bye,
  consecutive road games"

## Avoiding duplicate ground (read from the registry before designing cells)

`registry/weak_signals.json` (147 entries at session start) already covers
several rest/travel constructs, all **team-perspective, pooled across
home/away, or already-existing NFL/CFB pairs** — this battery is designed to
be genuinely new relative to every one of them, not a re-screen:

| existing entry | construct | why this battery differs |
|---|---|---|
| `bias_battery_extra_rest_edge`(`_opener`) | own_rest − opp_rest ≥ 4, team-perspective, either side | This battery's off-bye cells are **side-specific absolute thresholds** (home_rest≥13, away_rest≥13 separately), not a pooled differential |
| `bias_battery_short_week`(`_opener`) | own_rest ≤ 5, team-perspective, either side | This battery's `short_week_road` cell is **game-level and side-specific**: does being short-rested cost MORE specifically when it's the traveling side, using `home_cover` directly rather than the pooled own-team-cover framing |
| `bias_battery_three_plus_road_games` | 3rd+ consecutive true road game | Consecutive-road-games ground is **already built and recorded** (P+ 0.2209) — deliberately NOT re-built here; distance/timezone geometry is the new axis instead |
| `bias_battery_west_coast_early_kickoff`(`_opener`) | PT-timezone road team, non-PT opponent, kickoff<14:00 ET | This battery's `eastbound_multizone` cell is **timezone-general** (any originating timezone, not PT-specific) and does **not** require an early kickoff — a broader body-clock construct |
| `weather_followup_rest_disadvantage_cold` | away_rest<home_rest AND outdoor AND temp≤35F | Weather-compounded, not a pure travel/rest construct; this battery's cells never condition on temperature |
| `cfb_bias_battery_neutral_site_designated_home` | CFB only | No NFL equivalent exists yet; this battery adds one (`travel_rest_international_game`) |
| `cfb_bias_battery_bye_week_rest_edge`, `cfb_bias_battery_short_week_rest_disadvantage` | CFB only, relative-differential framing | NFL side-specific absolute-threshold cells here are a different construct even where the mechanism rhymes |
| `pick_conditioned_off_bye_fade_pre2018`, `pick_conditioned_rest_mismatch_pre2018` | pick-conditioned replication against the ACTIVE production model, 2011-2017 close-grade walk-forward | Different population (pick-conditioned vs. full-slate subset-vs-complement), different grading (close vs. this battery's week-blocked opener-era convention), different mechanism framing |

No stadium lat/lon table existed in this repo before this session (**measured**:
`registry/stadium_coordinates.json` did not exist; grep for "stadium" +
"lat"/"lon" across `src/`, `registry/`, `docs/` returned nothing before this
work).

## Data source and leakage posture

- Newest `data/raw/*/schedules.parquet` snapshot only (**read**,
  `data/raw/20260817T235649Z/schedules.parquet`, same snapshot the weather
  batteries use). Columns used: `game_id`, `season`, `week`, `game_type`,
  `weekday`, `gameday`, `home_team`, `away_team`, `result`, `spread_line`,
  `home_rest`, `away_rest`, `location`, `stadium`, `stadium_id`.
- REG games only, **full 2009-2025 window** (not restricted to 2018-2025).
  `home_rest`/`away_rest`/`location`/`stadium`/`weekday` are pregame-known
  schedule facts (not game-time actuals like the weather battery's
  temp/wind), so unlike that battery **every cell in this one is genuinely
  pregame-safe on its own terms** — there is no forecast-vs-actual leakage
  caveat here. The one caveat: stadium geometry (lat/lon/timezone) is a
  static reference fact about a known, scheduled venue, always knowable
  before kickoff once the schedule is set.
- **Era concentration is disclosed, following the ENV-02 precedent.**
  International/neutral-site games are heavily back-loaded: of 61 games
  2009-2025 (**measured**, `location=='Neutral'` count), the London/Mexico
  City slate expanded materially after 2013 and Germany/Brazil venues only
  exist from 2022 onward — `travel_rest_international_game` is mechanically
  an increasingly-modern-era cell even though it is scored on the full
  2009-2025 window; no other cell in this battery is meaningfully
  era-concentrated (distance/timezone geometry and rest thresholds are
  structural scheduling facts present throughout the window).

## Stadium coordinate table (new reference data)

Built this session: `registry/stadium_coordinates.json`, keyed by the exact
`stadium` string as it appears in `schedules.parquet` (not `stadium_id` —
**measured**, `stadium_id` is NOT a reliable physical-location key for
neutral-site games; nflverse leaves `stadium_id` set to the schedule-
designated home team's own code — e.g. `DAL00` — even when `stadium`
correctly names the true venue, e.g. "Maracana Stadium" for a 2026
Cowboys-designated "home" game actually played in Rio de Janeiro). Verified
by reading the raw rows for every `location=='Neutral'` game.

- 82 stadium-name entries: every stadium/roof-name variant a home team used
  2009-2025 (including renames at the same physical site — Denver's three
  names, Buffalo's three, etc. — each gets its own entry with identical
  coordinates, simple and correct) plus the international venues used as
  neutral-site hosts, plus a few 2026 (future, unscored) international
  venues for documentation completeness.
- **Coverage: 0 unresolved stadium names** in the 2009-2025 REG population
  (**measured**, exhaustive diff of `schedules['stadium'].unique()` against
  the coordinate table's keys — see the exploration script's output below).
  No games are dropped for missing stadium geometry.
- Coordinates are reported general knowledge (not fetched from a live
  geocoding API this session) — labelled **reported**, not measured, per
  AGENTS.md's provenance rule.
- Time zones are IANA zone names; the actual UTC offset used for a given
  game is computed at scoring time via Python's stdlib `zoneinfo`, evaluated
  at that game's date, so Daylight Saving Time is handled correctly rather
  than hardcoded (e.g. Arizona/`America/Phoenix` never observes DST and is
  distinguished from Denver/`America/Denver`, which does).

### Coordinate sanity check (measured this session)

Great-circle (haversine) distance computed from the table's lat/lon, checked
against commonly-cited flight great-circle distances for four well-known
city pairs:

| pair | computed | commonly-cited |
|---|---|---|
| MetLife Stadium (NYC) ↔ SoFi Stadium (LA) | 2449 mi | ~2451 mi |
| Wembley Stadium (London) ↔ MetLife Stadium (NYC) | 3452 mi | ~3459 mi |
| Arrowhead Stadium (KC) ↔ Gillette Stadium (Foxboro) | 1233 mi | ~1240 mi |
| Lumen Field (Seattle) ↔ Hard Rock Stadium (Miami) | 2720 mi | ~2724 mi |

All four land within 10 miles of the commonly-cited figure — the haversine
implementation and coordinate table are validated before any cover-rate
scoring.

## Derived quantities

- **`away_travel_mi`**: haversine distance from the away team's own modal
  home stadium THAT SEASON (`groupby(["home_team","season"])["stadium"]`
  mode over that team's own `location=='Home'` rows — same convention as
  `away_modal_roof`/`away_modal_surface` in the weather batteries, so
  relocations — STL→LA, SD→LAC, OAK→LV — resolve automatically per season
  from the schedule itself, no hand-maintained roster needed) to THIS game's
  actual venue (`stadium`, which is correct even for neutral-site games).
- **`tz_delta_eastbound`**: this game's venue UTC offset minus the away
  team's own home UTC offset, both evaluated at `gameday` via `zoneinfo`
  (DST-aware). Positive = the away team traveled toward an earlier sunset
  (eastbound in the Western Hemisphere / into Europe), the direction
  circadian research treats as the harder body-clock adjustment (advancing
  the clock forward vs. the easier delay of westbound travel).
- **`prev_own_travel_mi`** (home side only): the HOME team's OWN travel
  distance in ITS immediately preceding game this season (0 if that
  preceding game was played at that team's own home venue), built from a
  team-perspective long table (`game_id, team, side, own_travel_mi`, one row
  per team per game) sorted by team+season+date, shifted by 1 within each
  team-season group. Week-1 games (272 of 4431, exactly the count of
  season-opening games — **measured**, confirms the shift correctly excludes
  only true season-starts) have no previous game and are excluded via the
  missing mask, not zero-filled.

All three quantities are population/structural diagnostics only — computed
and range-checked before any cover-rate sign was examined.

## Method (reused verbatim from `scripts/nfl_weather_battery_screen.py` /
`scripts/nfl_weather_followup_screen.py`)

- `home_cover` from `nfl_ats.features.add_ats_outcomes` (pushes dropped).
- Subset-vs-complement full-slate-scaled effect: `(subset_cover −
  complement_cover) × 100 × fraction_of_slate`.
- **Week-blocked joint bootstrap primary** (block = `season*100+week`,
  matching the owner mandate that within-week game correlation is ZERO —
  blocking is a conservative convenience carried over from precedent, not an
  admission of within-week correlation), **season-blocked secondary**
  (block = `season`), same `block_bootstrap_two_group` algorithm.
- **20,000 samples, seed 20260819** (identical to every prior battery in
  this family, for direct comparability).
- `probability_positive` = fraction of bootstrap draws with gap > 0.

## The 8 predeclared cells

All score `home_cover` on the same REG 2009-2025 population (pushes/missing
spread dropped), subset vs. complement, thresholds fixed at round,
externally-justified values BEFORE any flag was crossed against outcomes.
Population sizes below are **measured** (population diagnostics only, not
outcome peeking).

1. **`travel_rest_long_distance_road`** — `away_travel_mi >= 1500`.
   n_flag=927 (20.9% of 4431-row diagnostic population; final scored n may
   differ slightly after the push/missing-spread drop). Round threshold
   (roughly the 79th percentile of the observed distribution, chosen for
   round-number defensibility, not tuned to outcomes). **Predicted:
   positive home_cover edge** (away team travel fatigue).

2. **`travel_rest_eastbound_multizone`** — `tz_delta_eastbound >= 2` (hours).
   n_flag=590 (13.3%). Threshold matches the commonly-used ≥2-timezone
   circadian-disruption cutoff in travel-fatigue sports-science literature
   (external justification, not data-mined). Distinct from
   `bias_battery_west_coast_early_kickoff` (timezone-general vs. PT-only,
   no early-kickoff requirement). **Predicted: positive home_cover edge**
   (eastbound body-clock disadvantage for the traveling side).

3. **`travel_rest_international_game`** — `location == 'Neutral'`. n_flag=61
   (1.4%, thin — disclosed, not hidden; this is what exists in the data).
   Direct NFL analog of `cfb_bias_battery_neutral_site_designated_home`
   (CFB, recorded P+ 0.0009 — a resolved-shaped negative on that league;
   **reported** from the registry, offered as motivating prior only, not
   determinative for NFL). **Predicted: negative home_cover edge**
   (schedule-designated "home" team is not actually playing at its true
   home venue, so the home-field advantage baked into the spread should be
   overstated for these games).

4. **`travel_rest_return_trip_hangover`** — HOME team's own `prev_own_travel_mi
   >= 1500` (in ITS immediately preceding game this season) AND
   `home_rest <= 8` (excludes bye-reset cases, where a 13+ day gap would
   plausibly erase any hangover — keeps the mechanism clean). n_flag=597
   (13.5%). This is the "return travel" construct named explicitly in
   ROADMAP's ENV-03 description text. **Predicted: negative home_cover
   edge** (fatigue/logistics hangover from the team's own prior long road
   trip persists into this game, even though this game is at home).

5. **`travel_rest_home_off_bye`** — `home_rest >= 13`. n_flag=266 (6.0%).
   Side-specific absolute threshold (distinct from the pooled
   `bias_battery_extra_rest_edge` differential). Threshold set at 13 (not
   14) to also capture the Monday-night-to-Sunday extra-rest turnaround
   named explicitly in ROADMAP's ENV-04 text, not only true 14-day byes —
   **measured**, `home_rest` distribution has a real cluster at 13 (27
   games) distinct from the 14-day bye cluster (216 games); both are
   captured together as one "meaningfully extra-rested home team" cell
   rather than fragmented into two thinner ones. **Predicted: positive
   home_cover edge** (extra home preparation time).

6. **`travel_rest_away_off_bye`** — `away_rest >= 13`. n_flag=278 (6.3%).
   Mirror of cell 5 on the away side — side-specific, distinct from the
   pooled differential construct. **Predicted: negative home_cover edge**
   (extra-rested visitor closes the standard home-field gap).

7. **`travel_rest_short_week_road`** — `away_rest <= 5`. n_flag=261 (5.9%).
   Same threshold as `bias_battery_short_week`'s own-rest cutoff (kept
   identical for direct comparability) but **game-level and side-specific**:
   tests whether short rest costs MORE specifically when it is the
   traveling side (using `home_cover` directly), a question the pooled
   either-side construct cannot answer on its own. **Predicted: positive
   home_cover edge** (short-rested + traveling compounds).

8. **`travel_rest_thursday_pure`** — `weekday == 'Thursday'`. n_flag=277
   (6.3%). Distinct from `weather_battery_thursday_outdoor_cold` (that cell
   additionally requires outdoor AND temp≤35F, a much narrower weather-
   compounded subset); this cell is the plain calendar effect across every
   Thursday game regardless of venue/weather. **Predicted: positive
   home_cover edge** (short-notice road logistics disadvantage the visitor
   more than the standard rest-differential story alone would predict).

## Recording commitment

Every cell above records to `registry/weak_signals.json` via
`nfl-ats weak-signals record` as `unresolved_below_power`, `league=nfl`,
`effect_units=accuracy_points`, `season_start=2009`, `season_end=2025`,
regardless of interval shape, via `scripts/record_travel_rest_battery.py`,
which reads every numeric field from the screen's output JSON and passes it
through unmodified (no hand-typed numbers). The only admissible alternative
under AGENTS.md would be a RESOLVED wrong sign (whole interval on the wrong
side of the predicted direction) or a positive-control bound — this
measure-only screen is not designed to produce either, and none is claimed
here regardless of what the numbers show.

This battery spends no rotation-registry window (measure-only, same posture
as both weather batteries) and does not touch `src/nfl_ats` or any
production code path.
