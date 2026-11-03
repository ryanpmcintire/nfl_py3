# LEAD-59 officials Wayback sweep, unattended nightly windows

## Goal
Run scripts/officials_wayback_sweep.py (2009-2014 PFR officials via web.archive.org) unattended in bounded overnight windows.

## State
- Resume (read): manifest.json keyed by pfr_id; a game with its html on disk and parsed officials is skipped, so rerunning with the same --run-id and season window continues exactly. Added (this lane): append-only data/raw/officials_pfr_wayback/<run_id>/progress.jsonl, one line per game attempt.
- Added `--max-minutes N`: checked before each new game; stops with stop_reason=max_minutes_reached, manifest and parquet written, summary line now ends items_remaining=<n> estimated_hours_left=<h>.
- Pacing (read): existing >=8 s floor between requests (config/source_policies.json internet_archive_pfr_boxscores condition minimum_delay_seconds_8, terms_reviewed 2026-09-05), contact User-Agent, backoff from 60 s doubling on 429/5xx, hard stop after 5 consecutive failures, single process. No change.
- Measured disk state (2026-11-02): run 20260907T175420Z (2009-2013) 1,279 of 1,280 games parsed, 1 parsed-0 (2009); run 20260907T140309Z (2014) 255 of 256 parsed, 1 replay_fetch_failed. So the 51-102 h estimate is stale: only 2 games remain.
- 3-minute run done; 2014 now 256 of 256 parsed.

## Tried
Measured: `--max-minutes 3` on the 2014 run: new_fetch_attempts=1 fetched_ok=1 total_http_requests=2 items_remaining=0 estimated_hours_left=0.0 stopped_early=False (finished before budget; the max_minutes_reached branch was not exercised). progress.jsonl got 1 line.

## Next
Only 2009 has 1 parsed-0 game left. Optional job for root after Week 9 lock: name `officials_wayback_2009_retry`, day `wed`, at `02:15` ET (existing jobs at 00:30, 03:30, 04:00; cap ends before 03:30), grace 60, timeout default 3600 s, season_guarded=False, one-shot then disable, argv:
`uv run --no-sync python scripts/officials_wayback_sweep.py --season-start 2009 --season-end 2013 --run-id 20260907T175420Z --max-minutes 45 --retry-unparsed`
The 51-102 h triage estimate is stale: 1,535 of 1,536 games already parsed.

## Open
Whether to merge the two run directories into one parquet for the downstream battery (owner/root decision).

## Rerun 2026-11-02 (as-of table + type-trait terms, measured)
- Table: load_officials(include_archive=True): 1,536 archive referee rows (6x256) + 2,896 nflverse; 0 duplicate game ids; 4 nflverse rows unjoined to schedules (2021-12-19 x3, 2023-01-02, pre-existing). Names: 40 archive-era, 29 nflverse-era, 16 shared; 72.6% of 2009-14 referee-games carry a name seen in 2015+. Unmerged candidate aliases (never guessed): Al/Alberto Riveron, Brad/Bradley Rogers.
- Penalty types for 2009-14 did not exist (snapshot is 2015-25); fetched via nflreadpy pbp with the snapshot script's own function (tests/scratch/lead59/pen_2009_2014.parquet, ignored).
- Script tests/scratch/lead59/run.py (ignored), output artifacts/lead59_officials_sweep_rerun/summary.json. Served fit has moved since 09-23, so variant A (2015+ history) rebaselines at DPI -0.466 P+ 0.137, holding -0.798 P+ 0.051 (not -0.133/-0.599).
- Recorded 4 cells, family lead59_type_trait_history_2009_fit_terms_v1, all unresolved_below_power.
## Next
Owner/root: decide LEAD-59 done; commit nothing from this lane yet.

## 2026-11-03 test repair
- test_cli_exposes_fallback_captures_and_retry_unparsed failed KeyError items_remaining: commit 7164e885 added items_remaining and estimated_hours_left to the run_sweep summary and the CLI print; the test's fake summary lacked them.
- Fix: added both keys to the fake summary in the test (stub only). Code untouched.
