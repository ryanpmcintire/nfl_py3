# Travel geometry feature contract (ENV-03)

## Scope

[Read: `ROADMAP.md`, ENV-03 row] ENV-03 asks for distance, time-zone change,
international games, and return travel. [Read:
`docs/travel_rest_battery.md`] The 2026-08-19 battery already defined and
screened related schedule-level quantities; this work does not rerun or
reinterpret that screen.

[Read: `src/nfl_ats/travel_geometry.py`] The reusable implementation is a
paper-research feature API only. It contains no prediction, staking, wagering,
policy-selection, or active-model path. [Read: `src/nfl_ats/constants.py`]
`FEATURE_FAMILIES["travel_geometry"]` declares the columns, but no
`FEATURE_SETS` profile and no `MODEL_FEATURE_COLUMNS` contract includes them.

## Columns

[Read: `src/nfl_ats/constants.py`] The family emits one row per supplied game:

| Column | Contract |
|---|---|
| `travel_home_distance_mi`, `travel_away_distance_mi` | Great-circle miles from each team's resolved home venue to the scheduled game venue. |
| `travel_home_tz_change_hours`, `travel_away_tz_change_hours` | Scheduled venue UTC offset minus origin UTC offset on the game date. Positive is eastbound/body-clock advance; negative is westbound/delay. |
| `travel_home_body_clock_direction`, `travel_away_body_clock_direction` | Sign of the time-zone change: `+1` eastbound, `0` unchanged, `-1` westbound. |
| `travel_international_game` | `1` when the game venue's validated IANA zone is outside the five US venue zones represented by the checked-in NFL registry. |
| `travel_neutral_site` | `1` when the schedule's `location` is `Neutral`; this is intentionally separate from international status. |
| `travel_home_prior_game_distance_mi`, `travel_away_prior_game_distance_mi` | That team's computed travel distance in its immediately preceding game of the same season. Missing at a season opener or when the earlier origin was unavailable. |

[Read: `src/nfl_ats/travel_geometry.py`] Distances use the same 3,958.8-mile
haversine earth radius as the frozen ENV-03 screen. Time-zone offsets use
`zoneinfo` on the game date, so DST is date-specific rather than a hard-coded
regional offset.

## Decision-time and missingness rule

[Read: `scripts/nfl_travel_rest_battery_screen.py`] The retrospective screen
resolved a team's origin from its full-season modal true-home stadium. [Read:
`src/nfl_ats/travel_geometry.py`] The reusable builder deliberately tightens
that rule: a decision row may resolve a team origin only from the latest
same-season `location == "Home"` row at or before that game. It never backfills
an early row from a later schedule row.

[Inferred] This stricter rule trades some early-season coverage for an explicit
chronological invariant: a future stadium or location correction cannot
rewrite an earlier decision row. Missing geometry remains `NaN`; it is not
silently replaced by zero. [Read: `src/nfl_ats/travel_geometry.py`] Current
named venues absent from the registry fail closed by default; callers may opt
into `strict_venues=False`, which retains the row with explicit missing
geometry.

## Registry provenance and validation

[Read: `registry/stadium_coordinates.json`] Venue entries supply latitude,
longitude, IANA time zone, and city. [Read: `src/nfl_ats/travel_geometry.py`]
The loader validates numeric finiteness and coordinate bounds, resolves every
IANA time-zone name, rejects empty registries, freezes the typed mapping, and
computes a canonical SHA-256 digest.

[Read: `src/nfl_ats/travel_geometry.py`] Returned feature frames carry
`travel_geometry_provenance` metadata with the registry source and digest,
the exact schedule columns read, an empty outcome-column list, the home-origin
cutoff rule, and the signed time-zone rule.

## Verification

[Measured: `uv run pytest tests/test_travel_geometry.py -q`, 2026-09-02] The
focused suite covers known NY-LA mileage, eastbound/westbound signs, London
international and neutral flags, chronological prior travel, missing origins,
strict/permissive unresolved venues, shuffled input, registry validation,
attachment provenance, and a leakage regression that mutates current-game
postgame values plus a later game's venue/location without changing the earlier
decision row.

[Measured: implementation actions, 2026-09-02] No empirical screen was rerun,
no outcome was used to choose a feature definition or weight, and no promotion
decision was made.
