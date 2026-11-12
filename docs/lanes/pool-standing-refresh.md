# pool-standing-refresh

## Goal
Keep the dashboard's "Where we stand in the pool" panel on the last finished week. Done each week = standings and pick distribution read from Splash, three scripts rerun, board published and pushed.

## State (2026-11-12 07:50 ET)
- Owner asked Thursday why the dashboard was not updated for the week. Measured: the site has shown Week 10 since Tue 20:55 ET (the noon lock was missed: `splash_board_tue` FAIL x5 at 12:05, `weekly_lock` MISSED, manual lock 20:37). The part still stale was the pool panel, frozen at "After Week 8": it reads three hand-built files from the one-off 2026-11-03 read (board_content.py `_load_pool_outlook`) and no job refreshes them.
- Week 9 read in Chrome, view only (contests.app.splashsports.com, Reports > Weekly results and Pick Distribution): 250 of 250 entries, no names stored. Our entry: rank 65 of 250, 70 points; leader 81; 61 entries above, 18 on 70.
- Files written: data/splash/field/2026_weekly_points_w01-09_raw.txt, 2026_week03-09_pick_distribution_raw.txt, 2026_field_best_picks_w01-09.parquet, 2026_weekly_points_w01-09.parquet; artifacts/pool_rank_replay/{field_best_pick_model,weekly_points_field,forward_sim_real_rivals_results}.json.
- Scripts no longer hard-code Week 8: pool01_field_best_pick.py, pool01_weekly_points.py (`--points`), pool01_forward_sim.py (`--weeks --points --rank` required) take the newest file by name. board_site_content.py `_field_week_texts` takes the newest weekly-points parquet.
- Result (1200 paths, inner 200): chance of first 6.0% [4.7, 7.4] from rank 65, 9 weeks left; standing-aware Best Pick gain +0.17 pp [0.0, 0.375], policies differ on 3 of 1200 paths (unchanged in kind; unresolved_below_power, no verdict issued).
- Field skill: split-half Spearman-Brown now -0.087 [-0.43, 0.16] on 9 weeks. A reliability is a variance share and cannot be below zero, so the simulation uses 0 (no rival skill spread) and records both values in the results json. Before this the negative value produced NaN rival offsets.
- Honest-assumption text: the simulation has our side covering at its stated chance (mean 55.2% over 136 games); those sides have covered 50.7% (69 of 136). Results json now carries both and the panel says so.
- 08:02 ET: `nfl-ats publish-board` exit 0; rendered diff reviewed (panel reads After Week 9, stale note gone). Gates: ruff format and check clean on the five edited files, mypy adds no error (22 pre-existing in board_* files). Committed and pushed 1fec1cb7 at 08:21 ET; Pages build `built`; live page reads After Week 9, 65 of 250 (measured by fetch).
- 08:20 ET: pytest (durable tests) 1643 passed, 2 failed. (a) `test_week_refresh_schedule_matches_enabled_pick_refresh_commands`: commits 0fe26b88 and 2573dc24 (11-11) added refresh jobs fri 12:40, 15:15, 19:00 and sun 15:35 without listing them in `WEEK_REFRESH_PASSES`; added, lists now equal (21). (b) `tests/test_public_board.py::test_build_public_site_writes_six_pages`: measured cause 08:41 ET, `assert_public_safe` found a sportsbook name. The signal ledger shows a record's raw description when `plain_summary` is empty (signal_ledger.py `_idea_text`), and 18 records committed 11-11 (12 `mod57_u8_*_anchor_*` in mod57_pool_source_book.json, 6 in mod57_offset_term_era_placebo.json) name a book there. Each now has a plain summary with no book name (only those 18 lines changed; `weak-signals pool` loads; rescan finds 0). The four live pages never carried a book name (0 matches). Rerun 08:44 ET: 1 passed. Both failures from the 08:20 run are resolved.
- Regression caught before push: `_half_point_week_texts` concatenates every `*_field_best_picks_w*.parquet`, so the cumulative w01-09 file beside w01-08 halved the Weeks 1-8 crowd shares on the history page. It now drops duplicate week/game/side rows; shares match the live page again and Week 9 is added.
- Splash entry checked against the served card 07:45 ET: all 14 sides, Best Pick CIN -3.5 and tiebreaker 37 match.

## Tried
- Screenshots are refused on the contests subdomain; page text and DOM reads work. The weekly-results table is virtualised: scroll its container and collect rows keyed in-page. The unpicked side of each game carries class `opacity-50`.

## Next
Each Tuesday after the Monday game settles:
1. Read Weekly results (all rows, Weeks 1-N) into data/splash/field/2026_weekly_points_w01-NN_raw.txt and append the WN line to a new 2026_week03-NN_pick_distribution_raw.txt (formats in the file headers).
2. `.tools/uv.exe run --no-sync python scripts/pool01_field_best_pick.py`, then `scripts/pool01_weekly_points.py --points P`, then `scripts/pool01_forward_sim.py --weeks R --points P --rank K --paths 1200 --inner 200` (background, about five minutes).
3. `nfl-ats publish-board`, review the diff, commit, push.
4. Not done: a scheduler check that fails when the newest weekly-points file is behind the last settled week (pattern of `splash_board_tue` / scripts/check_splash_board.py), so the gap raises an alert instead of a stale note.

## Open
- None.
