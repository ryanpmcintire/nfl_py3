# Continuous backlog and watchdog

## Goal
Keep three bounded backlog workers busy and check their queue every 30 minutes.
The owner explicitly authorized continuous work and automatic replenishment.

## State
Three workers active: SIM-08 timeout mechanism audit, POL-10 scorecard integration,
and MOD-18 paired opener evaluation core. Refill on completion messages as well as
30-minute session checks. Last check: 2026-09-27 08:55 UTC;
next check: 2026-09-27 09:25 UTC.

## Tried
- **Reported (owner):** `codex --no-daemon` stopped shell popups.
- **Measured:** the installed CLI rejects `codex --no-daemon queue` because
  queueing requires the shared server. Evidence: `.tmp/backlog-queue-check.log`.
- The session timer emitted its scheduled check at 08:55 UTC; its next check is 09:25 UTC, then every 30 minutes.
  It writes a status file and process output for the primary agent to inspect and refill all three assignments.
  The 08:55 check found all three workers running. No shared daemon was started.
- **Measured:** project capture scheduler is running; 218 jobs current, one
  acknowledged missed lock job. Its `--once` check exited 0.

## Next
Check live workers by 09:25 UTC, refill finished assignments immediately,
and checkpoint reviewed changes while root handles operational work.

## Open
A persistent wake-up after this session closes is not installed. The available
CLI queue path conflicts with the owner's working no-daemon launch mode.
The active-session checks cannot wake a closed session.
