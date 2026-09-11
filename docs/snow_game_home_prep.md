# LEAD-38: snow-game home preparation

Predeclared design (`ROADMAP.md` LEAD-38): equipment, routine, and footing
familiarity in snow favour the home team beyond what the market's weather
price already reflects. **Predeclared direction: BACK the home team in snow
games.** This doc states the snow-game definition and thresholds BEFORE any
cover-rate or production number appears below.

## Binding closing-grounds taxonomy (governs every verdict in this doc)

An interval or CI that contains zero is NEVER grounds to reject, fail, or
close an experiment. At this evaluator's ~2-point resolution, "contains
zero" is the EXPECTED outcome for a real small signal. Only two grounds ever
close a line of work: (1) refuted mechanism -- a RESOLVED wrong sign (whole
interval on the wrong side of zero) or zero split-half reliability; (2)
bounded by a positive control proven able to detect an effect that size.
Everything else is `unresolved_below_power`: recorded with `nfl-ats
weak-signals record` / `nfl-ats rotation record`, reporting
`probability_positive`, never the binary "contains zero." Nothing below is
closed on a crossing-zero interval alone.

## The source gap this doc resolves

`ROADMAP.md`'s prior LEAD-38 entry (2026-09-05) found no local column names
precipitation TYPE (rain vs. snow) anywhere in `data/raw/*/schedules.parquet`
or the forecast archive, and flagged a temp-gated compound of the forecast
precipitation-PROBABILITY field with temperature as the untried, "cruder"
proxy. This session's task predeclares exactly that compound before scoring
anything, which is what makes it admissible now.

## Snow-game definition (predeclared, stated before scoring)

A game is flagged **snow** iff, using ONLY information visible at the pool's
per-game decision deadline (`min(kickoff, Sunday 16:00 America/New_York)`,
the `pool_decision` forecast-archive cutoff, `docs/forecast_archive_build.md`):

1. the venue is **outdoor** (`roof` in `{outdoors, open}`, schedules parquet,
   the same convention `_FORECAST_OUTDOOR_ROOFS` uses elsewhere in this
   repo), AND
2. **forecast precipitation probability >= 50%** (`forecast_precip_prob_pct`,
   GFS-MOS `p06` falling back to `p12`, from
   `data/raw/forecast_archive/pool_decision_2009_2025/forecasts.parquet`), AND
3. **forecast temperature <= 32F** (`forecast_temp_f`, at or below freezing,
   the same archive).

50% and 32F are round, predeclared numbers, not tuned to the outcome --
picked before any cover-rate was computed (**measured**, a threshold grid
run first to confirm the choice was not vacuous: `precip>=50, temp<=32`
gives 22 games out of 4,431 REG games 2009-2025 with a resolved forecast
join; nearby grid points range 12-49 games, so 50/32 sits in the middle of a
smooth surface, not a cliff-edge pick).

**"All other outdoor cold games"** = outdoor AND forecast temp <= 32F AND
NOT snow-flagged (i.e., freezing but under the precipitation threshold) --
isolates the marginal contribution of precipitation given cold.

**"The population"** = every REG 2009-2025 game with a resolved
(domestic) forecast-archive fetch, regardless of weather or roof (n=4,380;
51 international games are excluded from the archive entirely and are not
part of any group below).

**Game-time actual-weather upper bound** (never the playable read): the
same rule with `forecast_temp_f` replaced by the schedules parquet's own
game-time `actual_temp_f`. The precipitation term is UNCHANGED (still the
forecast probability) because no observed precipitation-type or
probability field exists anywhere in this project's data (**read**, the
ROADMAP LEAD-38 source-gap note, reconfirmed this session by the same
column search). This upper bound is reported for context only and is
excluded from every cover-rate comparison, market-pricing check, and
production screen below.

Script: `scripts/snow_game_home_prep_screen.py` (`screen` and `production`
subcommands). Data sources: `data/processed/game_features.parquet`,
`data/processed/game_features_weak_stack.parquet`, the latest
`data/raw/*/schedules.parquet` snapshot, and
`data/raw/forecast_archive/pool_decision_2009_2025/forecasts.parquet`.
Bootstrap: week-blocked (block = `season*100+week`) primary, season-blocked
secondary, 20,000 draws, seed `20260911`, using this repo's shared
`probability_positive_from_draws` helper throughout.

## 1. Counts and home cover rate

**Measured**, `scripts/snow_game_home_prep_screen.py screen`,
`artifacts/snow_game_home_prep/screen_results.json`.

Count by season (deadline-visible snow flag vs. the game-time actual-weather
upper bound; `n_cold_outdoor_other` is the "all other outdoor cold games"
group):

| season | n games | n snow (deadline) | n snow (actual upper bound) | n cold-outdoor-other |
|---|---:|---:|---:|---:|
| 2009 | 254 | 1 | 1 | 12 |
| 2010 | 254 | 2 | 2 | 18 |
| 2011 | 254 | 0 | 0 | 4 |
| 2012 | 254 | 1 | 0 | 7 |
| 2013 | 253 | 5 | 4 | 17 |
| 2014 | 253 | 0 | 0 | 10 |
| 2015 | 253 | 1 | 0 | 10 |
| 2016 | 252 | 5 | 4 | 6 |
| 2017 | 251 | 2 | 2 | 14 |
| 2018 | 253 | 0 | 0 | 8 |
| 2019 | 251 | 1 | 2 | 5 |
| 2020 | 256 | 2 | 2 | 3 |
| 2021 | 270 | 1 | 1 | 7 |
| 2022 | 266 | 0 | 0 | 15 |
| 2023 | 267 | 0 | 0 | 3 |
| 2024 | 267 | 0 | 0 | 12 |
| 2025 | 272 | 1 | 1 | 17 |

22 snow games total (deadline-visible), 19 under the actual-weather upper
bound, 168 (167 graded, one push) "other cold outdoor" games, 4,380 (4,267
graded) in the population. Snow games are rare and lumpy by season, exactly
as expected for a weather-tail construct, never grounds on its own to treat
the family as unscreenable.

Home cover rate (pushes dropped):

| group | n | n graded | home cover rate |
|---|---:|---:|---:|
| Population (all REG 2009-2025, resolved forecast) | 4,380 | 4,267 | 48.93% |
| Snow games (deadline-visible) | 22 | 22 | **50.00%** |
| All other outdoor cold games (freezing, not snow) | 168 | 167 | 52.69% |
| Snow games, game-time actual-weather UPPER BOUND (not playable) | 19 | 19 | 42.11% |

Home-cover-rate gap, snow vs. population: **+1.07 accuracy points**,
week-blocked 95% **[-18.94, +29.37]**, `probability_positive` **0.5495**;
season-blocked **[-20.38, +27.03]**, P+ 0.5605.

Home-cover-rate gap, snow vs. all-other-outdoor-cold: **-2.69 accuracy
points**, week-blocked 95% **[-24.66, +26.79]**, P+ **0.4277**;
season-blocked **[-24.36, +22.56]**, P+ 0.4254.

Both intervals cross zero by a wide margin, exactly the expected shape for
an n=22 rare-event construct at this evaluator's power. Per the taxonomy
above, this is `unresolved_below_power`, not a negative: the point estimate
against the population leans (weakly) the predeclared direction; the point
estimate against the closer "other cold outdoor" control leans the other
way. Neither is resolved. Recorded as
`snow_game_home_prep_cover_gap_vs_population` and
`snow_game_home_prep_cover_gap_vs_cold_outdoor_other`.

## 2. Does the market already price it?

**Measured**, same script/artifact. For each of the 22 snow games, matched
against that same home team's OTHER REG home games in the SAME season that
are not snow-flagged (all 22 had at least one eligible control; 189 paired
rows total). Paired difference = snow game's home-perspective `spread_line`
minus the mean `spread_line` of the team's other home games that season
(home-perspective convention: negative means the home team is more favoured).

**Mean paired difference: -0.186 points**, 95% (season-blocked, 17 blocks)
**[-2.24, +2.56]**, probability the market already shades toward home
(`P(diff<0)`) **0.562**. The interval is wide and crosses zero; there is no
detectable market shading toward the home team in snow games relative to
that same team's other home games that season, but 22 games cannot rule out
a small one either. Recorded as `snow_game_home_prep_market_pricing_check`
(effect units `ats_points`, an unsigned diagnostic, not a directional
accuracy candidate).

## 3. Production screen (opener-graded, PRODUCTION weak_stack)

**Measured**, `scripts/snow_game_home_prep_screen.py production`,
`artifacts/snow_game_home_prep/production_results.json`. Rotation family
`snow_game_home_prep_on_production` (opener grade, `acknowledges_mined
_2018_2025=true`), assigned window **[2020, 2021]** via `nfl-ats rotation
assign` -- the same window other lanes drew tonight, taken as the default
earliest-eligible block rather than requested by size.

Candidate: production's (`weak_stack`, active model `d49194e04945a5e5`)
opener pick, with an override that flips an AWAY pick to HOME whenever the
game is snow-flagged (the "back home in snow" rule applied as a tilt on top
of whatever production already picks, matching this repo's existing tilt-
overlay convention).

- 456 paired opener-graded games in the window (35 weeks, 2 seasons).
- **2 snow games** fall inside the population that has both a captured
  Tuesday-opener and a close snapshot (the third 2020-2021 snow game listed
  in section 1 has no captured pairing and drops out here); **1 pick
  changed**.
- Accuracy delta (candidate minus baseline): **-0.219 accuracy points**
  (53.07% vs. 53.29%). Week-blocked 95% **[-0.673, 0.000]**,
  `probability_positive` **0.1827**. Season-blocked 95% **[-0.455, 0.000]**,
  P+ 0.1238.

Both intervals lean negative and their upper edge sits AT zero rather than
strictly below it, so this is **not** a resolved wrong sign (the whole
interval does not sit below zero) and no positive control has been run.
With exactly 2 flagged games, this result is entirely a description of what
happened on 1 flipped pick; it is `unresolved_below_power` by construction,
recorded via `nfl-ats rotation record --verdict unresolved`.

## 4. Split-half reliability (odd vs. even seasons)

**Measured**, same screen artifact. `snow_flag` is a per-game situational
condition (this week's forecast at an outdoor venue), not a persistent
per-team trait with a year-over-year value -- a formal split-half
correlation coefficient does not apply to it the way it would to a team
trait (matching this project's existing convention for other single-game
weather flags, e.g. `forecast_weather_kn_warm_team_cold_late`'s
`reliability_check.method=not_applicable`). What is reported instead: the
snow-vs-population home-cover gap measured independently on odd- and
even-numbered seasons.

- Odd seasons: 12 snow games, gap **+1.27 accuracy points**, 95%
  **[-23.99, +51.46]**, P+ **0.5704**.
- Even seasons: 10 snow games, gap **+0.85 accuracy points**, 95%
  **[-34.11, +37.63]**, P+ **0.5259**.

Both halves lean the same (positive) direction as the pooled read in
section 1, which is consistent with -- not proof of -- a real effect; at
12 and 10 games per half neither interval is remotely resolved. Recorded as
`snow_game_home_prep_cover_gap_odd_seasons` /
`_even_seasons`.

## 5. CFB replication

**Measured** (directory search, this session): `data/cfb/` contains
`draft_picks, espn_betting, lines, participants, pbp, portal,
recruiting_players, recruiting_teams, returning_production, rosters,
schedules, team_info, usage` and no weather or forecast directory of any
kind; `src/nfl_ats/forecast_weather_features.py` has no CFB reference.
**No CFB forecast-weather archive exists**, so the construct cannot be
replicated on CFB data. Per the task's own instruction, this is skipped
rather than built from nothing.

## What this does and does not establish

Every number above crosses zero (parts 1, 2, 4) or has its interval edge
touching zero (part 3). Per the binding taxonomy, none of that is grounds
to close the lead: no interval sits entirely on the wrong side of zero, and
no positive control has been run to bound the family. This is
`unresolved_below_power` across the board, and every cell is recorded as
such with `probability_positive` attached. The honest read: snow games are
rare enough (22 of 4,431 REG games under this predeclared rule, roughly 1.3
per season) that this evaluator cannot currently distinguish the
predeclared home-preparation mechanism from a coin flip, in either
direction, at either grade. The construct is registered and poolable; it is
not resolved, and per the pooling section of `AGENTS.md` this is exactly
the kind of small, real-or-not signal that should be kept rather than
discarded.

## Files

- `scripts/snow_game_home_prep_screen.py` -- `screen` (parts 1, 2, 4) and
  `production` (part 3) subcommands.
- `artifacts/snow_game_home_prep/screen_results.json`,
  `snow_game_table.parquet`, `market_pricing_pairs.csv`,
  `production_results.json`, `production_paired.parquet`.
- `registry/weak_signals.json` --
  `snow_game_home_prep_cover_gap_vs_population`,
  `snow_game_home_prep_cover_gap_vs_cold_outdoor_other`,
  `snow_game_home_prep_market_pricing_check`,
  `snow_game_home_prep_cover_gap_odd_seasons`,
  `snow_game_home_prep_cover_gap_even_seasons`.
- `registry/rotation_registry.json` -- family
  `snow_game_home_prep_on_production`, window [2020, 2021] spent, verdict
  `unresolved`.

## Provenance

- **measured**: every count, cover rate, bootstrap interval, and
  `probability_positive` in sections 1-4, all from this session's script
  run; the threshold-grid calibration in section "Snow-game definition";
  the CFB directory search in section 5.
- **read**: `ROADMAP.md`'s prior LEAD-38 row (the source-gap note this
  session resolves), `docs/forecast_weather_screen.md` and
  `docs/forecast_archive_build.md` (archive provenance and validation
  numbers), `docs/pool_edge_plan.md` (opener-grade population and
  methodology conventions), `src/nfl_ats/forecast_weather_features.py`
  and `src/nfl_ats/clv.py` (the pool_decision archive loader and
  `opener_pick_evaluation`).
- **inferred**: none load-bearing; the closing paragraph's reading of what
  the measured numbers imply for the family's status is a summary of the
  measurements above, not a new claim.
