# mod27-u5-pool-move-log

## Goal
Forward log of the pool-relative challenger beside the card, Weeks 9-18, no serving change. Parent: docs/lanes/mod27-better-picks.md.

## State
- Built scripts/pool_relative_move_forward_log.py (no src or scheduler edits). Live run: latest feed's week, one row per game not yet past its pick deadline, only from the newest feed; `--backfill --week N` takes the last pre-deadline feed per game and sets column backfill=True; `--grade` scores the frozen (last) row per game.
- Log: artifacts/pool_relative_forward_log/log.parquet. Challenger move coefficient read from artifacts/pool_relative_forward_log/challenger_coefficients.json (full-sample challenger fit, market_move_toward_home 0.1455, produced by appending a json dump to tests/scratch/pool_relative_move/loso.py and re-running it; source key inside the file). Served coefficients read from the active pick_probability artifact.
- challenger z = logit(feed final_probability) - served terms + coef * (leader median at cutoff - Splash pool line). Served terms use the feed creation time; active coefficients, not the feed-time set (approximate, as in U2).
- Verified (measured): `--week 8 --backfill` wrote 14 rows; `--grade`: 13 decided at the pool line, card 5-8, challenger 6-7, 1 differs (card 0-1, challenger 1-0). Matches U2's stand-in.

## Tried
- Reading the coefficient from the weak-signals registry: not stored there, so re-ran loso.py (3 s).

## Next
- Orchestrator adds the jobs below to scripts/capture_scheduler.py (each Job as `_log_job` style, Job(name, day, at, grace, argv, True, why, added_on=...)); argv = [str(UV), "run", "--no-sync", "python", str(REPO / "scripts" / "pool_relative_move_forward_log.py")].
  - pool_move_log_tue, tue 14:30, grace 90 (after weekly_lock 12:20 closes 14:20)
  - pool_move_log_wed, wed 18:45, grace 60 (after refresh_wed 18:15)
  - pool_move_log_thu, thu 15:30, grace 60 (after refresh_thu 15:00)
  - pool_move_log_sat, sat 10:50, grace 40 (after refresh_sat 10:30)
  - pool_move_log_sat_1245, sat 12:45, grace 10 (before 13:00 Saturday games, after 12:15 last call)
  - pool_move_log_sun_1020, sun 10:20, grace 60 (after refresh_sun 10:00)
  - pool_move_log_sun_1255, sun 12:55, grace 4 (before 13:00 kickoffs; after 12:45 last call)
  - pool_move_log_sun_1545, sun 15:45, grace 14 (before 16:00 kickoffs and the Sunday cap)
- Week 9 onward: read the log weekly; after Week 18 score with `--grade` plus Brier and log loss against the card, same metrics as Unit 1.

## Open
- Rows share the feed time; a late flip after the last feed is not in the log (card_side is the latest published pick at run time).
- Feed final_probability is the home-cover probability; confirm after the first live run.
