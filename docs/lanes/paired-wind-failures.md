# paired-wind-failures

## Goal
Fix paired_wind_* scheduler FAILs (10-11 .. 10-18).

## State
- Cause (read): scripts/capture_paired_wind_forecast.py:277-278 called atomic_parquet/atomic_json with (path, data); src/nfl_ats/io.py:93,119 take (data, path). AttributeError 'DataFrame' has no 'parent'.
- Fixed by swapping args; real command `capture_paired_wind_forecast.py --phase deadline` run once: exit 0, 5 games, 10 ok rows, 14 receipts, output written.
- eligible_product_rows=0 on that run (2026-10-18 18:36 UTC, past deadline for those games), so it is not an eligible pre-deadline observation.

## Tried
- Earlier FAIL at fetch_forecast line 131 (10-11..10-15) not reproduced; it was the same saver failing downstream or a transient fetch; unverified.

## Next
- Watch the next scheduled paired_wind slot (Tue 12:05) for OK.
- Missed Sunday 08:15/11:45 captures are unrecoverable (catch_up cannot replace a forecast observation).

## Open
- Whether the 10-11..10-15 line-131 failures had a separate cause.
