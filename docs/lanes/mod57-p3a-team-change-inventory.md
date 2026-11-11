# MOD-57 P3-A: team-change inputs, novelty and data map (2026-11-11 11:58 ET)

## Goal
Prove which early-season team-change inputs (QB change, roster continuity, HC/OC/DC change, draft capital, preseason market, preseason games) are new in the weeks 1-6 / hindsight-gap frame, and map data on disk. Read-only; nothing built or recorded.

## State
Prior work (read from registry shards and ROADMAP lines; none was run in weeks 1-6 against the hindsight-gap label of scripts/mod57_p2o/p2p; all graded at the opener as served-logit add-ons, 2020-25, 1,503 games, or full-season):
- per07_coord_change_on_production: Sept HC/OC/DC turnover diffs added to weak stack, accuracy -0.33 pts [-1.59, +0.93] (read, registry).
- lead29_playcaller_change_v1 / playcaller_change: first game 6-6, nothing (read).
- mod44_coach_tenure: tenure-state residual vs served prob, 2011-19 +0.19 rate_pp [-1.5, +1.9] P+ .585, 18 looks (read).
- mod42_ol_continuity (MOD-42): in-season five-starter streak, log loss -0.00067 [-0.00126, -0.00007], P+ .014 (read).
- pbp08 continuity split (LEAD-65, ROADMAP 707): weeks 1-4 low-continuity protection tilt +2.64 [+0.58, +4.65] P+ .977, 21 decisive games; different input (read).
- mod23_unit10/10b draft capital: starter snap-weighted draft_ovr, +0.47 / +0.13 pts, P+ .75/.60 (read). Draft CLASS added by team: not found.
- mod43_qb_learning_curve: log loss -1.3e-5 P+ .48; qb_continuity_replication +0.45 pts [-0.30, +1.20]; per12_qb_return, qb_expected_starter, qb_quality_loss, rookie_qb_debut_fade (read; QB state as cover add-on).
- P2-S (read, mod57 lane): offseason re-rating of line vs prior-season market ratings, weeks 1-4 slope -0.14 [-0.38, +0.07] P+ .10; prior-season surprise carried over: nothing.
- P2-L: true actives/QB handed as leak: +0.73 / +0.67 pts. MKT-25 17-week rating flat.
- Not found in ROADMAP/lanes/registry grep: NFL roster turnover/returning-snap-share; preseason futures or win totals; preseason games; any input scored against the hindsight gap in weeks 1-6.

Data on disk (read via pyarrow this session):
- Roster/continuity: data/players/raw/20261111T141547Z/weekly_rosters.parquet 702,040 rows 2009-2026 (team, status, years_exp, week); snap_counts.parquet 324,612 rows 2013-2025 (offense/defense snaps by player-game). Builds prior-season returning snap share. Needs as-of check that week-1 roster is pre-kickoff.
- QB: data/quarterbacks/depth/historical quarterbacks.parquet 22,962 rows with observed_at/effective_at; weekly_rosters + snap_counts.
- Coaches: data/raw/coordinators/*/coordinator_history.parquet (latest 20261110T123309744451Z) 3,168 rows HC/OC/DC by season, with observed-at fields; interim_coaches table.
- Draft: data/raw/draft_picks/20261028T004039Z 12,927 picks 1980-2026; combine.parquet 8,968 rows 2000-2026.
- Depth charts: data/raw/depth_charts 554,215 rows, dt 2025-08..2026-03 only (one season; no history).
- Preseason market: data/market/raw/<year>0825-futures, Super Bowl outright prices, ~10 books, 2020-2025 only (6 seasons, 320 rows each). NO season win totals on disk; free source: sportsoddshistory.com season win totals (reported, not downloaded). Sagarin as-of Tuesday ratings 2009-2025, 9,848 rows (week-1 value carries the preseason prior; tested as ratings in MKT work).
- Preseason games: nflverse schedules.parquet is 2026 only, 272 rows, regular season; NO preseason results on disk; free source: Pro-Football-Reference or ESPN preseason scoreboards.
- participation/ values dirs empty by glob; pfr_transactions are 73,749 headline index rows (text, not structured).

## Tried
Nothing built. Ranking (inferred):
1. Returning snap share + new-player snap share (offense, defense; prior-season snaps of players on week-1 roster / total), home minus away. Mechanism: Tuesday line in weeks 1-6 leans on last season's team; heavy turnover means last-season rating stale in a direction the line prices only partly. Buildable now.
2. QB changed from prior-season primary QB (indicator plus prior-season QB snap share of the current starter), home minus away. Mechanism: line carries QB name, not that the team around him was rebuilt. Prior work is cover add-on, not gap frame.
3. HC/OC/DC new count interacted with weeks 1-6 (coordinators history); per07 nearest, so lower.
4. Draft class capital added by team (picks 2013-26, round weights fixed by round, not fitted on outcomes). Weak mechanism.
5. Preseason Super Bowl price rank 2020-25 only: too few seasons for walk-forward; needs win totals first.

## Next
Test design for 1 and 2 (one unit each):
- Label: hindsight gap = hindsight-rating margin minus closing line (scripts/mod57_p2o, p2p definition), weeks 1-6, tested 2015-25 walk-forward, max label season = test season minus 1 (2013-14 warm-up; snaps start 2013 so continuity needs 2014+ prior-season data; test seasons start 2015).
- Fit: elastic net with the single input (plus pooled turnover family of units 1+2 as a declared pool), also entered after the 295 columns. Score: mean absolute error against the closing line 9.818 (weeks 1-6 subset), calibration slope of predicted residual on real margin with interval, share of the hindsight gap predicted, picks vs close and opener (2020-25).
- Controls: intercept-only; shuffled input within season (1,000 permutations); placebo weeks 7-12.
- Looks: unit 1 = 2 inputs x 2 label windows (gap, real margin) = 4; unit 2 = 2 x 2 = 4; family 8 plus the pooled arm = 9. Record every cell with weak-signals record, unresolved_below_power; interval containing zero closes nothing.
- As-of: roster rows from week 1 of the season being predicted with status date before kickoff; truncation test on the roster file before use.

## Open
Whether weekly_rosters week-1 rows are stamped pre-kickoff (must verify, not read here). Win totals and preseason results need a public download (not done). Registry shard contents beyond the first signal per shard were not read.
