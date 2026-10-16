# backup-offsite-timeout

## Goal
Weekly backup_offsite (Sun 22:30, scripts/offsite_backup.py) succeeds again.

## State
- measured: last OK 2026-09-27 (snapshot 359255eb, 3.2 GB repo); 10-04 and 10-11 FAIL backup step timed out at 1500 s.
- measured: artifacts/mod25e3 grew to 12.4 GB / 20,636 files, all written after 09-27; 10.5 GB is play_*.parquet and 1.5 GB events_*.parquet (simulator play dumps, regenerable). rehearsal_lockday 2.1 GB, data/raw 20.9 GB unchanged.
- change: scripts/offsite_backup.py EXCLUDES gains mod25e3/**/play_*.parquet and mod25e3/**/events_*.parquet. Summaries, sim_games, npz, bin still backed up.
- verification: real daemon argv run of --run-job backup_offsite, result in the MANUAL-RUN line of scheduler_log.

- measured 10-16 run 1: backup step finished in about 12 min (no timeout) but exit 1 on unreadable data/environment_recovery/pytest_* dirs; added exclude environment_recovery/pytest_*.
- measured 10-16 run 2 (daemon argv): backup step passed in under 1 min; forget --prune failed (11): repo locked by PID 17084 since 2026-10-11 22:50 (the failed timed-out run). PID 17084 is not running (stale lock).
- blocked: `restic unlock` was denied by the permission classifier; owner/orchestrator must run it (stale lock, no live holder), then rerun --run-job backup_offsite.

## Tried
Nothing deleted, no lock touched, repository untouched.

## Next
If the run still times out, scale --timeout from measured bytes or check sftp throughput.

## Open
Whether other future artifact dirs of regenerable sim dumps need excludes.

- 2026-10-16 orchestrator: stale lock (PID 17084, 10-11 run, not running) blocks forget --prune; restic unlock was denied to the agent by the permission classifier, so it is not retried by another agent. After the owner runs unlock, rerun `--run-job backup_offsite` and confirm forget --prune passes.
