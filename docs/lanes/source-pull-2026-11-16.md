# Source pull 2026-11-16 (free public sources)

## Goal
Acquire five free public sources for research (no outcomes read, no accuracy computed): NFL preseason win totals, Open-Meteo kickoff forecasts, 538 Elo/QB Elo, nfelo outputs, Kalshi/Polymarket game prices. Per source: coverage, join rate to data/processed/game_features.parquet game_id, as-of statement.

## State
- Script: scripts/source_pull_2026_11_16.py, subcommands wins, weather, elo538, nfelo, kalshi, polymarket. Writes data/external/<dir>/ with a local `.gitignore` (`*`) and inventory.json each.
- Policy entries added to config/source_policies.json (append only): sportsoddshistory_nfl_win_totals, open_meteo_historical_forecast, fivethirtyeight_nfl_elo, github_nfelo_public_ratings, kalshi_public_market_data, polymarket_public_market_data.
- All six sources pulled 2026-11-16 20:44-21:00 UTC; each dir has inventory.json. Weather caches raw responses under data/external/open_meteo/raw and never refetches. No processes left running.
- Open-Meteo semantics (read, docs 2026-11-16): historical-forecast stitches the first few hours of each successive model run (lead about 0-6 h, issue time not exposed); archive starts 2016 (JMA), 2017 (ECMWF IFS HRES), 2018 (HRRR), 2021-03-23 (GFS), most others 2022. Previous Runs API gives fixed 1..7-day lead offsets (provably issued at least 24 h before valid time for previous_day1).

## Results (measured 2026-11-16, inventory.json per dir; all six runs exit 0, every dir has a local .gitignore, git check-ignore confirms)
- open_meteo (data/external/open_meteo/kickoff_forecast_weather.parquet, raw cache in raw/): 2,150 of 2,151 outdoor/open-roof games 2016-2026, keyed by game_id (100% join by construction, 0 stadiums missing coordinates). Historical-forecast best_match temp/wind/gust/precip at kickoff hour: 2016 0/200, 2017 13/205, 2018-2025 complete, 2026 101/182 (games through 2026-11-15). Previous-runs day1/day2 lead: 2021 200/202, 2022 195/198, 2023 172/199, 2024-2025 complete, 2026 101. As-of: historical-forecast = first hours of the latest run before the valid hour (lead about 0-6 h, issue time not exposed), pregame relative to kickoff but NOT provably before the pick timestamp; previous_day1/day2 = fixed 24/48 h lead, provably issued before a Sunday 4 PM ET lock for any game at least a day later.
- win_totals (data/external/win_totals/nfl_win_totals.parquet): 575 rows, 2009-2026, 32/season except 2011 = 31; team-season join 99.8%. As-of: page header "As of <date>" = early-September (season-opening week) number, 14/18 seasons parsed; 2012, 2013, 2023, 2026 header not parsed. That is a preseason closing-week number, not the opening May/June number. Settled week, actual wins, result not stored.
- fivethirtyeight_elo: Wayback capture 20250306125358 of nfl_elo.csv; 3,775 games 2009-2022; repo join 100% (2009-2014 263-265/267 rows, a few unmatched duplicates absorbed by swap join). As-of: pregame by construction (elo/qbelo _pre); file is a retroactive full-history publication, not provable per game. Post/score/quality columns dropped unread.
- nfelo: nfelo_games 4,780 rows, join 86.5% (233-251/267-285 per season; 2026 131/272 = games so far); historic_projected_spreads 1,565 rows 2021-2026 only, join 2021+ about 87%; prediction_tracker 14 rows; elo_snapshot 32. Commit history fetched (capped 1,000 commits, so earliest dates are truncated: nfelo_games >= 2025-09-15 in the capped window, projected spreads from 2021-12-28). As-of provable only where a commit precedes kickoff.
- kalshi (data/external/kalshi/): KXNFLGAME series, live + historical endpoints; 1,090 markets / 545 events; 447 joined (2025: 284 = full season incl. playoffs, 2026: 163 to date); the 98 unjoined are inferred preseason games. Hourly home-yes candles 203,183 rows on 445 games, only candles ending at or before repo kickoff kept. As-of: exchange timestamps, provably pregame. Nothing before 2025 in this series.
- polymarket (data/external/polymarket/): 2,371 NFL events pulled, 824 game-slug events (nfl-away-home-date), 670 joined (81%; 2024: 278, 2025: 285, 2026: 107). Hourly outcome0 price 117,553 rows on 670 games, only points at or before kickoff kept. As-of: CLOB timestamps, provably pregame. 2026 count is below Kalshi's 163, so gamma offset cap (about 2,000) likely truncates; /events/keyset pagination would finish it. The first two runs failed with 406 (pages above 50 need gzip) and 422 (offset cap).

## Tried
- robots: sportsoddshistory (now under covers.com) allows; Kalshi empty; Polymarket gamma 404 (none); Open-Meteo API host robots `Disallow: /` (crawler rule; API documented for programmatic use, used as API only, recorded in policy).
- 538 original host redirects (302); fetched the Internet Archive capture instead.
- nfelo LICENSE permits read/analyze/republish outputs; forbids running the code (we only read CSVs).

## Next
- Acquisition unit is complete. Optional follow-ups: (1) 2016-2017 weather via models=jma_seamless (2016+) or ecmwf_ifs (2017+), new cache key; (2) Polymarket /events/keyset pagination to fill 2026; (3) narrow OUTCOME_COLUMN and rerun nfelo; (4) parse missing win-total as-of headers.
- Research use is a separate declared unit (leave-one-season-out, opener-only and intercept-only controls).

## Open
- OUTCOME_COLUMN regex in the script also dropped pregame model columns containing "cover" (nfelo_home_cover_prob_close, home_cover_prob, projected_winner*); these are model outputs, not outcomes. Narrow the regex and rerun nfelo if wanted.
- nfelo commit pagination capped at 10 pages; raise to prove as-of before 2025-09.
- Win totals as-of regex missed 4 seasons' headers; read those pages' header text.

## Re-pull 2026-11-16 (Kalshi prices, Polymarket keyset)
- Kalshi: candles now read price.close_dollars (fallback close; 2025 stored close was already dollars), volume_fp, open_interest_fp. Candle price_close non-null 51,105 (2025 only) -> 98,251 (2025 51,105; 2026 47,146); bid/ask close 126,330 -> 203,197 rows. Events 447 joined (284 in 2025, 163 in 2026), unchanged. Whole series re-pulled (script has no season filter).
- Polymarket: /events/keyset (limit 50, after_cursor) replaces offset paging. Tag events 2,371 -> 2,951; game events 824 -> 896; joined 670 -> 742 (2026: 107 -> 179; 2024 278, 2025 285 unchanged); history rows 117,553 -> 124,686.
- 2026 W1-10, 145 played-card games, price within 1 h: Tuesday noon Kalshi 0 -> 145, Polymarket 107 -> 145, any 107 -> 145; pick deadline Kalshi 0 -> 145, Polymarket 100 -> 138, any 100 -> 145. Fit-season coverage (2024-25) unchanged (2025 Kalshi tue 258, deadline 267).
- `prediction_market_gap.py pregame --week 10 --slot tue --dry` read both venues for 2026_10_LAC_BAL (Kalshi 0.50, Polymarket 0.50). No fit, score2026 or write run; frozen coefficients untouched. Prior 2026 score (Polymarket-only prices) was made on partial coverage; rescoring is the owner/orchestrator's decision.
