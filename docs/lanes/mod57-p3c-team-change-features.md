# MOD-57 P3-C: team-change feature table (2026-11-11 12:10)

## Goal
Build roster-continuity and quarterback-change inputs, home/away/diff, with as-of proof. Build only; no outcome column read, nothing recorded.

## State
Rebuild: `F:\Repos\nfl_py3\.tools\uv.exe run --project F:\Repos\nfl_py3 --no-sync python F:\Repos\nfl_py3\scripts\mod57_p3c_team_change_features.py` (OMP_NUM_THREADS=1, about 1 min). Outputs in `artifacts/mod57_p3c/`: `team_change_features.parquet` (3,423 rows, one per REG game 2014-2026, key `game_id` = game_features.parquet key used by p2p), `asof_proof.json`, `coverage_by_season.csv`, `distribution_by_third.csv`, `distribution_by_season_third.csv`. ruff check/format clean.

Columns (side = home_/away_/diff_): `ret_snap_share_off|def` = prior-season (S-1 REG) snaps of players on the team's roster (status ACT or INA) who took them for the same team / team's S-1 total snaps on that side. `arr_snap_vol_off|def` = S-1 snaps for other teams of roster players / league-mean team S-1 total on that side (plain denominator choice). `qb_changed` = expected starter gsis != S-1 primary QB (most QB-position offense snaps for that team). `qb_starter_prior_qb_snap_share` = starter's S-1 QB snaps for that team / team S-1 QB snaps (0 for a rookie or new arrival). Audit: `*_qb_starter_id`, `*_qb_depth_age_days`, `*_roster_key_used` (season*100+week). Variants `*_sameweek` use the game-week roster: NOT provably pre-kickoff; do not serve.
Roster used (main columns): each team's latest roster week strictly before the game's (season, week); week 1 therefore uses the prior season's last roster for that team (read; matches the data contract in the snapshot manifest, weekly_rosters strictly earlier week). Expected starter: rank-1 QB of the latest depth chart with effective_at <= kickoff (historical file, chart for week W effective after week W's last kickoff, so a week-N game sees the week N-1 chart; 2025 from raw live depth charts dt). Never actual starter.

As-of measurements (measured, 2014-2025, 6,302 team-weeks, snaps mapped pfr->gsis, 0.07% unmapped):
- Week-N roster includes 98.5% of weeks 2-18 game participants (97.2% in week 1). Arrivals vs the prior roster who also played that week = 1.6% of week-N rows (weeks 2+), 25.4% of week-1 rows (vs prior-season last roster). Only 36 participants (week 2+; 6 in week 1) absent from week N but on N+1. Arrival rows cannot be split into Tuesday signings and game-day elevations, so week-N rows are not provably pre-kickoff: upper bound 1.6% (weeks 2+) / 25.4% (week 1) of rows could be post-kickoff.
- Truncation: 40 of 40 sampled games identical on all 20 main columns after deleting roster rows from the game's week on, snap rows from games kicking off at or after, depth rows effective at or after kickoff. The `_sameweek` columns are excluded (they vanish when week-N rows are deleted by construction).
- Coverage: all main columns non-null for every game 2014-2025 (3,167 of 3,167); 2026 roster columns 272/272, QB columns null (no 2026 depth chart; live charts end 2026-03).

## Tried
Same-week roster as main (rejected, not provable). pfr_id roster map alone (47% coverage, replaced by players.parquet crosswalk, 99.6% of snap ids).

## Next
Tester: read distribution CSVs by third; note cost below; run the P3 test design in `mod57-p3a-team-change-inventory.md` with these columns.

## Open
Cost: week-1 main roster columns reflect in-season churn only, not the offseason (week-1 median QB depth age 245 days, week-1 qb_changed mean 0.23 vs 0.32/0.38 weeks 2/3 because the chart updates only after week 1). Offseason moves appear first in `_sameweek` week 1 (unprovable). No raw timestamped source for week-1 roster exists in data. diff_qb_changed SD about 0.65-0.70 all thirds.
