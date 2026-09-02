# Roster continuity feature contract

Status: **implementation complete 2026-09-02; no ATS experiment or model
promotion performed**.

## Existing continuity paths

**Read from `src/nfl_ats/players.py`:** `player_continuity` already exposes
seven snap-lineup overlap measures plus active-roster continuity and mean
experience. A target game reads only the two latest completed lineups and
weekly roster rows whose `(season, week)` key is strictly earlier than the
target. The current game's snap lineup updates state only after its feature
row is emitted.

**Read from
`artifacts/player_experiments/20260813T122348Z/{summary.csv,paired_comparisons.csv}`:**
the existing 2018–2025 close-graded ablation scored continuity at 51.95% versus
51.08% for its base on 2,075 paired games, an improvement of +0.87 accuracy
points with week-blocked 95% [-0.77, +2.49]. The registry correctly keeps
`player_family_base_vs_continuity` unresolved at `probability_positive=0.852`;
this implementation task does not reinterpret or rerun that evidence.

## Returning-snap offseason prior

The isolated `roster_returning_snaps` family adds three side-level measures
and their home-minus-away differences:

- `returning_offense_snap_share`
- `returning_defense_snap_share`
- `returning_special_teams_snap_share`

For target season T and team X, each denominator is X's total resolved player
snap mass in T-1 for that channel. The numerator is the same mass contributed
by players listed `ACT` or `INA` on X's latest weekly roster strictly before
the target `(season, week)`. A prior-season roster is never treated as proof
that a player returned: the latest eligible roster must itself be from T.

That rule deliberately makes Week 1 missing. The nflverse weekly-roster source
has no observation timestamp, so using its target Week-1 row at a 24-hour
decision cutoff would be an unsupported availability claim. Missing prior
season snaps, a missing earlier current-season roster, or an empty active
roster likewise emits `NaN` for all three measures.

Target-season and future snap outcomes cannot enter season T's prior: the
aggregation stores every source season S only under target key S+1. The
leakage regression in `tests/test_players.py` mutates target-season snaps,
future roster rows, and the target game's roster row; none can change the
current target. The target-game roster mutation first becomes visible to the
following week, matching the existing conservative roster contract.

## Source coverage and provenance

**Measured 2026-09-02** from immutable player snapshot
`data/players/raw/20260817T184901Z/manifest.json` (roster SHA-256
`69d067e8...e7`, snap SHA-256 `f7b45c14...46f`) joined to
`data/raw/20260824T115346Z/schedules.parquet`:

- all three returning-snap values exist for 5,918/6,302 (93.91%) 2014–2025
  regular-season team-games;
- coverage is 5,918/5,920 (99.97%) after Week 1;
- the 382 Week-1 team-games all fail closed as designed;
- the remaining two gaps are Tampa Bay and Miami in 2017 Week 2, whose Week-1
  game was postponed for Hurricane Irma, leaving no earlier current-season
  roster row;
- stable GSIS linkage covers 99.43% of snap rows and 99.36–99.55% of rows with
  positive offense, defense, or special-teams snaps. Unresolved identities are
  excluded from both numerator and denominator rather than guessed.

The audit also found nflverse roster aliases `ARZ`, `BLT`, `CLV`, `HST`, and
`SL` were not canonicalized to schedule identities. The shared team-alias map
now normalizes them, restoring current-season roster coverage for Arizona,
Baltimore, Cleveland, Houston, and the Rams. A focused regression pins those
mappings.

## Model isolation

**Read from `src/nfl_ats/constants.py`:** the three differences are registered
as `FEATURE_FAMILIES["roster_returning_snaps"]`, but no existing `FEATURE_SET`
includes that family. Building the columns therefore does not silently alter
`player_continuity`, `weak_stack`, the active model, or any recorded arm. A
future ATS comparison requires its own predeclaration and is outside PER-06.
