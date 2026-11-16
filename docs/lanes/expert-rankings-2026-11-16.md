# expert-rankings-2026-11-16

## Goal
Outside-leads-2 candidate 2: FantasyPros weekly expert consensus rankings (ECR) summed to a team offensive expectation, compared with the market. Part 1 source and timing; Part 2 screen at the close, one served look on QW only for a survivor. Owned files: scripts/expert_rankings.py, data/external/fp_ecr/ (gitignored), artifacts/expert_rankings/.

## State (done 2026-11-16 18:16 EST; ready for done/)
- Source (measured): github.com/dynastyprocess/data files/db_fpecr.parquet (nflreadr load_ff_rankings type all), 1,885,069 rows. Weekly pages (QB, PPR RB/WR/TE): 2020-2026 only, one scrape per week, mostly Friday (2019 has one Dec 27 scrape). No 2017-19 weekly history. Policy entry dynastyprocess_fpecr added to config/source_policies.json. Joins: db_playerids.csv (fantasypros_id to gsis_id), nflverse stats_player_week 2020-25 for actual PPR points.

## Declaration (written 2026-11-16 18:12 EST, before any ATS outcome was joined)
- Population: REG 2020-25, weeks 2+, games with a pregame scrape for that week: latest scrape whose date is strictly before the ET date of the pick deadline min(kickoff, Sun 4 PM ET) and whose next Sunday is the game's Sunday. Thursday games after a Friday scrape drop out.
- Rank to points: per season, week and position, isotonic (decreasing) fit of actual PPR points on ECR position rank over that season's earlier weeks (ranked, joined players; did-not-play = 0). No hand constant.
- Team expectation: sum of mapped points over the team's ranked QB/RB/WR/TE in the scrape; converted to real points by OLS of actual team points on that sum over the season's earlier weeks.
- Implied team totals: home minus away = the spread (total cancels). As-of: Tuesday quote (hindsight_audit_team.tuesday_quotes) where available, else the close; coverage reported.
- Cells (2 looks): gap = E_diff_points - spread_asof; gap_orth = E_diff_points minus its OLS fit on spread_asof over the season's earlier weeks. Declared sign POSITIVE on home cover.
- Screen vs the close, pushes dropped: OLS slope per SD in accuracy points, 200 within-season shuffles, halves 2020-22 / 2023-25, season bootstrap. Survivor = above null p95 and positive in both halves; a survivor gets ONE served look at the opener 2020-25, pushes kept, on QW.

## Tried
- Part 1 (measured, artifacts/expert_rankings/part1.json): weekly scrapes 2020 12, 2021 19, 2022 16, 2023 16, 2024 15, 2025 16, 2026 10; weekdays Fri 90, Thu 10, Sat 2, Tue 1, Wed 1. Scrape to deadline date: 2 days for 1,048 games, 3 for 132, 1 for 63. Pregame-covered REG games 152/237/221/221/193/219 (2020-25); missing mostly Week 1 (scrapes start Weeks 2-4 in 2022-24) and Thursday games. gsis join 99.2-99.96% per year; all team codes mapped. Spread as-of Tuesday/SBR open 1,117 games, close 14 (corr to close .967). E_diff vs spread corr .779; k median .279 real points per fantasy point.
- Screen vs close, REG 2020-25 weeks 2+, 1,131 games, 2 looks (measured, screen.json; rerun after lint byte-identical):
  - gap: +1.73 acc pts/SD [-0.36, 3.48], P+ .941, null pct 87.5 (p95 +2.82), halves +1.64/+1.83, 4/6 seasons, gap side 575-556. Not a survivor.
  - gap_orth: +1.32 [-0.07, 2.53], P+ .965, null pct 81.5 (p95 +2.45), halves +0.02/+2.53, 5/6 seasons, 567-564. Not a survivor.
- Recorded: registry/weak_signals/expert_rankings_2026_11_16.json (2 cells, unresolved_below_power, accuracy_points); pool ran clean. No served look (protocol).

## Next
Optional frozen forward check: 2026 Weeks 2-10 scrapes exist (10 through 2026-11-13); score gap and gap_orth once, declared positive, at the close and the pool line, with Weeks 11-18 logged forward. Not started.

## Open
- Bootstrap over 6 seasons is narrower than the shuffle null; read the null percentile, not the interval.
