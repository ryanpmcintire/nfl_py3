# Referee battery: officiating-crew effects on ATS cover rates

**Predeclared 2026-08-19, before any cover-rate sign in this family was
examined.** Family name: `referee_battery`. Units: `accuracy_points`.
Blocking: week primary, season secondary. Grade: `close`. Seed: `20260819`
throughout. Samples: 20,000. Every cell below is recorded to
`registry/weak_signals.json` via `nfl-ats experiment run`, **regardless of
which way the sign comes out** -- per `AGENTS.md`'s binding rule, a interval
crossing zero is never grounds to skip recording or to call a cell a
non-finding.

## Data source and pregame-safety argument

- **Head referee identity per game**: `nflreadpy.load_officials()` (the
  nflverse-data `officials/officials` release), filtered to
  `position == "Referee"` (the on-field crew chief, distinct from the six
  other crew positions and the numbered/replay alternates also present in the
  feed) and `season_type == "REG"`. Fetched into
  `data/raw/officials/<UTC timestamp>/officials.parquet` (gitignored, mirrors
  the existing `data/raw/<timestamp>/schedules.parquet` snapshot convention).
  MEASURED coverage: 2015-2025, 21,900 raw rows, 3,029 `Referee`-position rows
  (2,896 REG-season after the `season_type` filter).
- **Join key**: `officials.parquet`'s own `game_id` is the LEGACY numeric GSIS
  format (e.g. `"2015091000"`), not `game_features.parquet`'s
  `"2015_01_PIT_NE"`-shaped id. The crosswalk is the newest
  `data/raw/*/schedules.parquet` snapshot's own `old_game_id` column (already
  the join key `_bias_battery_*` builders use for the same snapshot).
  MEASURED: 2,892 of 2,896 REG referee-assignment rows (99.86%) matched a
  schedules row via this crosswalk; the 4 unmatched rows (3 in 2021, 1 in
  2022) simply drop out of the population, same silent-drop convention as
  every other inner-merge trait builder in this module.
- **Penalty counts per game**: nflverse PBP's own `penalty`/`penalty_team`
  columns (NOT the repo's existing trimmed local PBP snapshot loader --
  that snapshot's stored column list omits `penalty_team`, see
  `PBP_SNAPSHOT_COLUMNS` in `src/nfl_ats/pbp.py`). Fetched fresh via
  `nflreadpy.load_pbp(seasons=list(range(2015, 2026)))` and immediately
  aggregated to one row per `game_id` (`penalties_total`,
  `penalties_on_home`, `penalties_on_away`) -- the raw ~532k-row PBP itself is
  NOT persisted, only this small derived aggregate, written alongside
  `officials.parquet` as `data/raw/officials/<timestamp>/game_penalties.parquet`.
- **Why this is pregame-safe**: NFL officiating crew assignments are
  published by the league office before kickoff (this is the premise this
  task was scoped under, and is the same premise the repo's existing
  `home_qb_name`-based `backup_qb_start` builder relies on for starter
  identity). The construct never uses the game's OWN penalty count or
  outcome -- every trait below is the referee's PRIOR-season aggregate,
  lagged exactly the way `penalty_rate_quartile` lags a team's own prior
  season penalty rate (`_lag_and_quartile` in `experiment_runner.py`, ported
  here as a referee-keyed sibling). See "Leakage test" below.

## Known data-coverage caveat (read, not hidden)

`load_officials()` only goes back to 2015 (nflreadr's own documented floor).
An official's FIRST appearance in this dataset is indistinguishable from a
genuine NFL debut -- someone who officiated for 15 years before 2015 and is
still active in 2015 looks identical, in this data, to a true rookie. MEASURED:
of the 29 distinct head referees who appear 2015-2025, all 17 who worked in
season 2015 show "0 prior seasons" purely from left-censoring, while only 12
referees have a genuine documented debut season (2016-2025) inside the
window. The experience cells (3 and 4 below) therefore exclude season 2015
from their population (`population.seasons: [2016, 2025]`) to remove the
contaminated all-referees-are-"rookies" 2015 slate; this does not fully
remove the caveat (a referee debuting in, say, 2013 and still active in 2016
is invisible to us as a veteran until their 6th dataset-visible season), so
`referee_rookie_home_cover`'s "rookie" label should be read as "first
dataset-visible season," not a certified NFL debut.

## Cells

Officials, PBP, and schedules are joined and aggregated to one row per
(official name, season): `mean_total` (mean total penalties/game, both teams
combined, across REG games that official worked as `Referee` that season) and
`mean_diff` (`mean(penalties_on_away) - mean(penalties_on_home)`, i.e. how
much more that official's crews penalize the road team than the home team,
per game, that season). Officials are identified by `official_name`, not
`official_id` -- MEASURED: `official_id` is NOT stable for the same person
across nflverse-data's own history (16 of 29 referees carry two different
`official_id` values across their careers, e.g. Adrian Hill is `official_id`
10 through 2022 and `493` from 2023 onward; `official_name` has no such
break in this window).

Two lagged lookback traits are used, each qcut into global quartiles over
every (official, season) pair with a valid year-over-year lag
(`season == prev_season + 1`), exactly mirroring `penalty_rate_quartile`'s
own `_lag_and_quartile`:

| Trait | MEASURED split-half (year-over-year Pearson r, consecutive-season pairs) |
|---|---|
| `mean_total` (penalty-rate) | **+0.370** (158 pairs) -- a real, moderate persistent trait. |
| `mean_diff` (home-away penalty differential) | **-0.101** (158 pairs) -- close to zero; this trait shows little-to-no measured year-over-year persistence in this window. Cells 5/6 below still use `reliability_check.method: split_half` (the construct IS a well-defined per-referee-season trait, so the METHOD is applicable regardless of what number comes out -- this mirrors how `penalty_rate_quartile` always uses split_half regardless of outcome), but the LOW measured reliability itself is reported plainly, not treated as a reason to reclassify or skip recording. |

1. **`referee_penalty_rate_top_quartile_home_cover`** -- flag: home team's
   game officiated by a referee whose PRIOR-season `mean_total` (penalty
   rate) sits in the top quartile, vs. everyone else. Mechanism: more
   penalty stoppages per game are hypothesized to disrupt the ROAD team's
   offensive rhythm and silent-count communication disproportionately more
   than the home team's (home communication is unaffected by crowd noise at
   the snap; road teams already face a communication handicap that extra
   stoppages compound), producing a small home-cover edge under
   high-penalty-rate crews. Sign: **+1**. Reliability: split_half
   (`mean_total`, +0.370).
2. **`referee_penalty_rate_bottom_quartile_home_cover`** -- same trait,
   bottom quartile (fewest penalties/game). Mechanism: the mirror of (1) --
   fewer stoppages means less disruption differential, so the flagged games
   are hypothesized to show a SMALLER (or negative) home-cover edge relative
   to the rest of the slate. Sign: **-1**. Reliability: split_half
   (`mean_total`, +0.370, same trait as cell 1).
3. **`referee_veteran_home_cover`** -- flag: home team's game officiated by
   a referee with >= `params.veteran_threshold` (default 5) distinct PRIOR
   seasons appearing as `Referee` in this dataset, vs. everyone else.
   Population restricted to `[2016, 2025]` (see caveat above). Mechanism:
   documented officiating-bias literature (e.g. Moskowitz & Wertheim,
   *Scorecasting*) attributes home-field penalty bias partly to
   crowd-noise pressure on officials' split-second judgment calls; more
   experienced referees are hypothesized to be less susceptible to that
   pressure, producing a SMALLER home-cover edge in veteran-officiated games
   than the rest of the slate. Sign: **-1**. Reliability: `not_applicable`
   -- prior-season count is a monotonically increasing career-stage counter,
   not a trait whose value could repeat or correlate year-over-year in the
   split-half sense (analogous to `backup_qb_start`'s per-game career-stage
   condition, also `not_applicable` in this same module).
4. **`referee_rookie_home_cover`** -- flag: home team's game officiated by a
   referee with exactly 0 prior dataset-visible seasons as `Referee`, vs.
   everyone else. Population restricted to `[2016, 2025]` (excludes the
   contaminated 2015 all-"rookies" slate; see caveat above). Mechanism: the
   mirror of (3) -- less-experienced officials hypothesized MORE susceptible
   to home-crowd pressure, producing a LARGER home-cover edge. Sign: **+1**.
   Reliability: `not_applicable`, same reasoning as cell 3.
5. **`referee_home_penalty_tilt_top_quartile_home_cover`** -- flag: home
   team's game officiated by a referee whose PRIOR-season `mean_diff`
   (away-minus-home penalty differential) sits in the top quartile (the
   referee's crews have historically penalized road teams the most relative
   to home teams), vs. everyone else. Mechanism: this is the direct
   implementation of the documented home-field penalty-differential effect;
   a referee's own history of protective home-team officiating is
   hypothesized to persist, producing a home-cover edge. Sign: **+1**.
   Reliability: split_half (`mean_diff`, -0.101 -- MEASURED near-zero;
   reported, not hidden; read this cell's outcome as more exploratory than
   cell 1/2's given the weak measured persistence of the underlying trait).
6. **`referee_home_penalty_tilt_bottom_quartile_home_cover`** -- same trait,
   bottom quartile (referees whose crews show the LEAST home-favoring
   penalty split, i.e., closest to or below penalizing the home team as much
   as the road team). Mechanism: the mirror of (5) -- hypothesized SMALLER
   (or negative) home-cover edge. Sign: **-1**. Reliability: split_half,
   same trait and same caveat as cell 5.

All six flags are the one-sided `eligible=None` design (`fraction_of_slate =
n_flag / n_total`), the same design `home_underdog` and the bias-battery
builders use: `flag = is_home & <referee condition>`, restricting the
"flagged" arm to the home-side row of a flagged game (so `team_covered` on
that row is exactly `home_cover`), while the complement is "every other
team-game row in the population" (both sides of every non-flagged game, plus
the away-side row of flagged games) -- consistent with the existing
`_flag_home_underdog` precedent this whole family borrows its shape from.

## Leakage test

`tests/test_experiment_runner.py::test_referee_flags_do_not_use_this_games_own_penalty_count`
mutates a game's OWN `game_penalties.parquet` row (the current season's
penalty count for the very game being flagged) to a value that would flip
the quartile if it were (incorrectly) used directly, and asserts the flag is
unchanged -- because the trait merge only ever reads the prior-SEASON
aggregate (`shift(1)` over `(official_name, season)`, requiring
`season == prev_season + 1`), a game's own penalty count structurally cannot
reach its own flag.

## Registry

Specs: `registry/experiment_specs/referee_battery_*.json` (six files, one per
cell above). All run via `nfl-ats experiment run <spec> --dry-run` first,
then for real; the runner's mechanical classifier (never a hand-typed
verdict) determines `unresolved_below_power` vs. `refuted_mechanism` per
`AGENTS.md`'s two-admissible-grounds rule. Per-cell measured numbers are
reported in the session's final report, not duplicated here (this doc is the
predeclaration; the registry and the run artifacts under
`artifacts/experiment_runner/` are the measurement record).
