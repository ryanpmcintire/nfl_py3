# ENG-58

## Goal
Flexed and rescheduled games: opener captured for one kickoff slot, game played in another (ROADMAP ENG-58).

## State
Unit 1 done 2026-11-03 18:25 (scripts/eng58_flex_audit.py; artifacts/eng58/audit.csv, live_schedule_changes.csv, processed_kickoff_stale.csv, run.log). Measured: tue_open historical quotes carry commence_time_utc for 1,619 of 4,368 games 2011-2026 (all 1,537 graded opener games covered); 294 differ from the final schedule kickoff: 252 are the feed's 1-minute tick, 42 are real moves (34 graded, 21 shift the min(kickoff, Sun 4 PM ET) deadline, 0 change date). Moves: 2020 COVID slots, 2021_17 LA-BAL, 2022_16 HOU-TEN, 2023 wk19 MIA-KC/LA-DET, 2024_09 IND-MIN, 2025_06 NE-NO (16:25 to 13:00 ET), 2025 DEN-NYJ, 2025_07 TB-DET and HOU-SEA. 2026: schedule snapshots (16, from 2026-08-12) show one flex, 2026_05_CHI_GB 16:25 to 13:00 ET (changed between snapshots 20260923T005026Z and 20260929T191306Z). Opener grades use the opener line, not the slot: no grade changes. Not yet checked: which graded studies consume slot or rest for the 34 moved games (MKT-24 slot from final schedule, MOD-29 rest_hours from final schedule gameday/gametime). Games without opener commence (2011-2019 and live) cannot be compared; no pre-2026 schedule snapshots on disk.
Forward guard (read): mod24/mod25 week_frame reads kickoff from data/processed/game_features_weak_stack.parquet and refuses past deadline at write (scripts/mod24_forward_log.py:47-53, 221-224); processed kickoff matches final schedule for 272 of 272 2026 rows (0 stale); CHI-GB week 5 runs all recorded with kickoff 17:00Z, before it. Risk: the feature parquet is a static build, so a flex after the last rebuild would carry the old kickoff; forward_ledger.py:144-154 takes min(row deadline, game_features kickoff).

## Tried
Novelty check only. Nearest: ENG-52 (opener timestamp audit), ENG-55, MKT-24. Differs: Audits the kickoff slot itself, not the capture time.

## Next
First unit for one agent: Compare the schedule row at opener capture with the final schedule row for every graded game, list games whose date or kickoff changed, and report how many feed a slot, rest-day or forecast input.

## Open
Rules: audit only, no research verdict; report counts, rows and the exact join; any graded result that changes is re-run and re-recorded through the registry validator, never weakened. Look count and family declared before any sign is seen. Data: nflverse schedules, captured schedule snapshots and opener archive timestamps (local).

## Unit 2 (2026-11-03 18:35 ET, scripts only)
Added scripts/live_kickoff.py (latest data/raw schedules snapshot, gameday+gametime ET -> UTC). Guards now use min(own kickoff, live): mod24_forward_log.py week_frame, mod25_wfj_forward.py deadlines_for, mod27_pool_offset_observed.py forward limit, forward_terms_week.py main (pre-filter), lead106_injury_language.py live_inputs, lead99_llm_read.py build_packet deadline_et (lead101 inherits via L99.build_packet). Check: tests/scratch/eng58/guard_check.py, Wk9 identical (272/272 rows 2026 unchanged), simulated -8 day move refused by mod24 and mod25. Cadence: schedule snapshots on disk are weekly (Tuesdays: 1006, 1013, 1020, 1027, 1103), no capture_scheduler job found writing them; guard is as fresh as the last weekly snapshot. Next: add a daily schedule-snapshot job if flex detection needs to be faster; mod24 ruff E501 pre-existing (7).

## Unit 3 (2026-11-03 18:35 EST)
Added scripts/capture_schedule_snapshot.py (current-season nflreadpy.load_schedules via nfl_ats.data helpers, validate_schedules, writes data/raw_schedules/<stamp>/schedules.parquet + manifest.json with fetched_at_utc, rows, sha256). Separate root chosen so latest_snapshot/load_snapshot callers on data/raw never see a snapshot lacking team_stats.parquet. scripts/live_kickoff.py now takes the newest folder by stamp across data/raw and data/raw_schedules. Ran once: data/raw_schedules/20261103T233105Z; live_kickoff 2026 rows identical (272 rows, 0 diffs, Wk9 15/15 same). ruff clean.
Proposed scheduler entry (orchestrator adds; not edited). Sunday 08:00 precedes the 09:30 ET London window and 13:00 early window:
    Job("schedule_snapshot_daily", day, "09:00", 30, [str(UV), "run", "--no-sync", "python", str(REPO / "scripts" / "capture_schedule_snapshot.py")], True, "ENG-58: daily nflverse schedule snapshot so a mid-week flex reaches live_kickoff guards.", season_guarded=True, added_on="2026-11-03", catch_up=True) for day in DAYS except sun, plus the same with ("sun", "08:00").
