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

## 2026-11-14 21:08Z line-move depth and late-file alarms
- scripts/check_served_terms.py now reports `thin_move_games` (game, books of the fitted 3, move) for every unplayed game below full leader-book depth, plus `late_written_quote_files`. Problem only when books < ANCHOR_MINIMUM_BOOKS (2, smallest count with a cross-book median); two of three is reported, not alarmed. Any quote file saved after Wednesday began for a Mon-Tue snapshot is always a problem (the 10-31 re-parse moved live picks). Lane flip-attribution-audit-2026-11-14 holds the evidence.
- Live Week 10 (measured): LAC at BAL 2/3 -0.5, LA at ARI 2/3 -0.25, MIN at GB 1/3 -1.5; problem "MIN at GB: the line move behind this pick comes from one book, not a market consensus."; exit 1.
- Replay: --as-of cannot reach Week 8 (forecast file holds Week 10 only); `_baselines_written_late` at Week 8 Wednesday returns 20261027T182541Z-odds-gap-private-reparse.
- On failure the owner sees only a scheduler_log line `FAIL(1) served_terms_check_<day>: ...`; capture_scheduler.py notify_after_job pushes only for refresh_/lineups_ jobs, so no phone alert. ruff clean; pytest -k "served_terms or scheduler": 64 passed.
