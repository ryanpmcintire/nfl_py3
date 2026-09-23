# midweek-model-activation

## Goal
Explain why `publish-board` failed closed on 2026-09-23 with a stale served refresh-chain measurement, and stop it recurring.

## State
Closed 2026-09-23, no code change. Measured: the scheduled `lineups_wed` job (retrain + `weekly-run` + `publish-predictions --with-board` + `refresh-picks` + `publish-board`, all `check=True` where it matters) ran 16:00-16:20 UTC and logged `OK lineups_wed` (`data/scheduler_log.txt:1702`), so the scheduled path published cleanly on the new model `d5da2c0670e17eba`. The fail-closed error appeared only after a research lane ran `nfl-ats opener-evaluation` by hand at 17:28 UTC (ENG-47, new cache version), adding new opener evaluations of the active model; `served-refresh-card` (reuse-or-measure) then re-measured and `publish-board` passed.

## Tried
A subagent proposed making `weekly.py`'s optional-step handler raise when the model changed; rejected because the scheduled run did not fail (inferred from the OK log line and the 16:16/16:18 publication stamps).

## Next
After any hand-run `opener-evaluation` on the active model, run `nfl-ats served-refresh-card` before `publish-board`.

## Open
None.
