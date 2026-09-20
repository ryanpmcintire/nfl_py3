# Lineup refresh dry run

## Goal

Make `capture_scheduler.py --run-job lineups_sun_am --dry` avoid nested card,
ledger and site writes while still exercising the scheduled command path.

## State

- 2026-09-20: found `dry_command` only removed top-level recording flags.
  `refresh_lineup_forecast.py` then called `weekly-run`, `publish-predictions`,
  `refresh-picks --publish-card`, and `publish-board` with no dry propagation.
- Added `--dry-run` to the lineup script. It passes the flag to the existing
  `weekly-run --dry-run` plan path and returns before lineup builds or publish
  calls. Scheduler appends the flag only for this nested script command.
- Existing scheduler dry argv test extended; 57 scheduler tests pass, Ruff
  passes, and comment/docstring check finds zero. Exact manual argv exercise
  passed at 11:17 ET after real recovery finished: `MANUAL-DRY-RUN OK
  lineups_sun_am`; scheduled child argv included `--dry-run` and returned in
  4.3 seconds.

## Tried

- `refresh_lineup_forecast.py --help` lists the new flag. Read `weekly.py`:
  `run_weekly(dry_run=True)` returns the plan before executing its steps.

## Next

- Keep the nested dry flag in future scheduler refactors; it prevents weekly
  and publish writes.

## Open

- Private current-spread capture is documented in `free-odds-sources.md`;
  public Books-now excludes private vendor quotes.
