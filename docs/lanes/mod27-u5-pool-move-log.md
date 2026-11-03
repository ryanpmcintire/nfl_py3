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

**Measured 2026-11-03 Week 8 scored (CHI at SEA settled):** see docs/lanes/week8-forward-scoring-2026-11-03.md.

## Fenced repairs replay (measured 2026-11-03, scripts/mod27_fenced_repairs_replay.py)
Inputs copied to artifacts/mod27_replay_inputs (g26 feed frame, per-book base/final pickles, pool_offset2.csv); output artifacts/mod27_fenced_repairs_replay/results.json. No src edit. Recorded family mod27_fenced_repairs_replay (4 cells, unresolved_below_power).
- Pool-offset term: 2026 Weeks 2-8 (103 games) 0 sides changed, mean |p shift| 0.009. Simulated-offset LOSO 2020-25 (5885 stacked rows): log loss gain 0.00238 [0.00099, 0.00379] P+ 1.0; Brier gain 0.00119 P+ 1.0; accuracy +0.92 pts P+ 0.87; offset coef 0.85-0.96 per fold.
- Thin-book (<3 leader books): drop move+availability: 5 games 2020-25, LOSO gain 0.00007 [-0.00033, 0.00056] P+ 0.60, accuracy -0.27 pts; disagreement 46 games served 25-21 v repair 21-25. 2026: 5 sides change (W5 2, W7 1, W8 2), served 4-1 v repair 1-4 (IND_JAX 0.575 to 0.490 is the only non-coin-flip; harness agrees with feed side on 75 percent only, so approximate). All-books median variant changes 0 rows and 0 sides.
- Implication: pool offset costs nothing to defer; thin-book drop has no historical support, so keep the v7 fence; both wait for Week 18.

## Observed pool offsets, Weeks 2-8 (measured 2026-11-03, scripts/mod27_pool_offset_observed.py)
- `--mode observed` writes artifacts/mod27_pool_offset_forward/observed_weeks_1_8.json (frozen coefficients, hash in frozen_coefficients.json). Week 1 has no discrete base in the feed, so 103 games, Weeks 2-8. 0 sides changed, mean |p shift| 0.0092.
- Graded at pool line (103): log loss gain +0.0042 [-0.0008, 0.0095] P+ 0.95, Brier +0.0021 P+ 0.96. Graded at Tuesday market/opener (99): log loss +0.0009 [-0.0037, 0.0056] P+ 0.66, Brier +0.0005 P+ 0.68. Per-week P+ 0.17-0.98 (n 13-16). Recorded as two weak-signals cells, family mod27_fenced_repairs_replay.
- Week 9: Splash Week 9 file not in data/splash at 2026-11-03 morning. Run after weekly_lock 14:20: `uv run --no-sync python scripts/mod27_pool_offset_observed.py --mode forward --week 9` (writes artifacts/mod27_pool_offset_forward/2026_week09_<stamp>.json once; refuses if present). Orchestrator may add as scheduler job; not in forward_terms_week.py (different structure).
