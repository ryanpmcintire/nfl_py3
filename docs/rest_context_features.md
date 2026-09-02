# Rest-context feature contract (ENV-04)

## Scope

ENV-04 now has a reusable NFL feature builder in
`src/nfl_ats/rest_context.py`. **Read from that module:** it consumes only
`game_id`, `season`, `gameday`, `home_team`, `away_team`, and `location`, and
it records that exact allowlist in output provenance. It does not read a
result, betting line, observed weather field, or any later row when deriving
an earlier decision row.

This work did **not** rerun, score, or adjudicate an experiment. The historical
rest-cell measurements remain documented in `docs/travel_rest_battery.md` and
the weak-signal registry; this contract neither replaces nor reinterprets
them.

## Columns and frozen definitions

**Read from `src/nfl_ats/rest_context.py`:** rest is the integer calendar-day
gap from a team's immediately preceding game in the same season. A season
opener has unknown rest, so every rest-dependent value is `NaN`, never zero.

| Column | Definition |
| --- | --- |
| `rest_home_days`, `rest_away_days` | Per-side calendar days since the prior same-season appearance |
| `rest_days_diff` | Home rest minus away rest; missing unless both sides are known |
| `rest_home_off_bye`, `rest_away_off_bye` | Rest >= 13 days, preserving the threshold frozen in `docs/travel_rest_battery.md` |
| `rest_home_short_week`, `rest_away_short_week` | Rest <= 5 days, preserving the battery's short-week threshold |
| `rest_home_mini_bye`, `rest_away_mini_bye` | Rest of 9--11 days, the extended turnaround centered on Thursday-to-Sunday's 10 days, distinct from a full bye |
| `rest_away_consecutive_road_games` | Away team's current true-road streak including this game; neutral and true-home games break the streak |

Neutral-site games are not treated as road games. **Read from the historical
implementation cited in `docs/travel_rest_battery.md`:** that is the same
true-road convention used by the already-recorded
`bias_battery_three_plus_road_games` construct.

## Contracts

- **Read from `src/nfl_ats/constants.py`:** `rest_context` is registered in
  `FEATURE_FAMILIES` but its columns are absent from every `FEATURE_SETS`
  profile. Building the family does not admit it to the active model.
- **Read from `src/nfl_ats/rest_context.py`:** duplicate game ids, invalid
  dates/seasons, blank teams, identical opponents, invalid location values,
  and multiple appearances by one team on one date fail closed with
  `DataContractError`.
- **Read from `tests/test_rest_context.py`:** known-answer tests pin bye,
  short-week, mini-bye, road-streak, neutral-site reset, and season reset
  semantics; shuffled input must produce identical output.
- **Read from `tests/test_rest_context.py`:** the leakage regression mutates
  current/future results, lines, observed weather, and a later schedule row,
  then requires the earlier decision row to remain bit-identical.
- **Read from `src/nfl_ats/rest_context.py`:** attachment is by unique
  `game_id`, preserves the caller's row order and existing columns, and adds
  explicit provenance including an empty `outcome_columns_read` list.

## Promotion posture

This family is registered infrastructure only. **Read from
`src/nfl_ats/constants.py`:** no production or candidate feature profile
includes these columns. Any future evaluation must preserve chronological
selection/calibration/test separation and the weak-signal decision taxonomy;
none was needed to complete ENV-04's deterministic feature contract.
