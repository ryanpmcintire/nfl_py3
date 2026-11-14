# eng-served-terms-liveness-check

## Goal
Fail loudly when a served probability term goes silent (offset anchor collapse 2026-11-10, Week 5 no move input).

## State
- scripts/check_served_terms.py built (production functions only); exit 1 on: offset term zero while pool differs from anchor; no anchors past Tue 12:30 ET with board captured; no move input from Thu 00:00 ET.
- scripts/capture_scheduler.py: jobs served_terms_check_tue 12:50 (requires weekly_lock), _thu 18:30, _sun 09:15. Daemon needs one restart to load them.
- Verified: Week 10 exit 0 (14 anchors, 9 differ, 9 nonzero); MANUAL-RUN OK; demo tests/scratch/fixpicks/demo_served_terms.py (hook zeroed) exits 1 with the problem; ruff clean; test_capture_scheduler 57 passed.

- 2026-11-14 14:55 ET: the script also reports each game's anchor book count, cross-book range and crossed mass against its room, and exits 1 on a one-book anchor, a range above 2.0, or mass above room; `--as-of` replays an earlier instant; problems also go to stderr. Measurement, derivations and verification: fix-picks-2026-11-10.md State. Uncommitted.

## Tried
- Nothing rejected.

## Next
- Restart daemon (orchestrator). Move this lane to done/ after the first Thursday run passes.

## Open
- move_available_games is 0 on Tuesday night by design; the Thursday check is the first live test of condition c.
