# Weekly card readiness proposal

## Goal
Recommend one bounded, useful task after the owner rejected Findings: tell a reader which week's card is displayed and why the next card is unavailable.

## State
- Task selection only; implementation has not been requested or started. Leave the rejected Findings lane unchanged.
- Measured 2026-09-22: a structured read of `docs/index.html` found `SEASON 2026 &middot; WEEK 2` in the week tag and Week 2 in the board summary.
- Measured: `.venv\Scripts\python.exe scripts\check_splash_board.py --season 2026 --week 3` reported no `data/splash/2026_week03_*.json` capture. Missing input must not be replaced by invented pool lines.
- Measured: scheduler `--status --brief` reported a running daemon and a missed Tuesday weekly lock. Restarting a running daemon does not supply the missing manual pool capture.

## Tried
- Read the current execution order and bounded backlog candidates. UI-19 already documents real-row verification; ENG-23 and ENG-34 are historical implementation notes, not verified current gaps. Do not propose those as new features.
- The prescribed `uv` scheduler invocation failed on cache permissions. Running the existing Windows environment's Python with `capture_scheduler.py --once` succeeded, then status was read. No forecast or publication job was run.

## Next
If selected, add a compact notice inside the existing This Week design: identify the displayed week and say that the upcoming week's picks await pool lines when that is the verified blocker. Start at `board_content.py` and the This Week renderer in `board_terminal.py`; use the existing pool-capture validator and schedule context. Distinguish missing lines from other failures. Clear the notice only when the new card is available. Verify missing-input and ready states, mobile rendering, required repository checks, and actual publication. No new tests under the moratorium.

## Open
- This is a recommendation, not an accepted implementation scope. No new mockup or research experiment is needed.
- Preserved unrelated changes: `CURRENT_PREDICTIONS.md`, `tiebreaker.json`, and six untracked experiment registry records. No source changes or code tests in this selection task.
