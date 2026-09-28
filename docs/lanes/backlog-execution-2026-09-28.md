# Backlog execution, September 28

## Goal

Complete executable work from the single `ROADMAP.md` backlog with stable counts.

## State

OPS-06 backup/sync/player repairs are pushed in `e350601` (handoff `1c6d966`).
PER-09's lagged ST builder is pushed in `f3ff113` (handoff `46d1e47`).
**Measured:** four declared fits produced 7,413 unserved player-season ratings.
Scheduler status/manual-receipt repair is reviewed and verified; the primary
daemon runs its source hash with 193 enabled jobs and unchanged schedule.
Evidence: `docs/offsite_backup.md`, `docs/st_player_ratings.md` sections 10–11.
No full row closed: **measured**, 56 total, 13 done, **43 remaining**.

## Tried

**Measured:** capture deployment retained rollback; seven new aliases passed,
five player jobs failed with OOM. Backup status/one-file restore and sync dry run
passed. Memory/output, cache, atomicity and transfer probes passed. ST artifact
verification covered 12 source partitions, hashes, chronology and exposure.
Early fits contain only 141/131 plays; retain all folds, make no serving claim.
Real-state status replay exposes all five failed player jobs. The corrected
primary status exposes four older dated failures: lineups_tue, lineups_sat,
splash_board_tue, refresh_last_call_sat_1215. These remain outstanding.
Final commands with inherited `UV_CACHE_DIR=.tmp/uv-cache`:
`.tools/uv.exe run --no-sync ruff format --check .`; `ruff check .`;
`mypy src`; `pytest -q --basetemp .tmp/backlog-execution-20260928/pytest-scheduler-complete`.
All passed: 1,645 tests. No tests added or existing-test deletions proposed.

## Next

After explicit approval, deploy final player and scheduler repairs, rerun the
five full-history player aliases on the 512 MiB host, then run final primary sync.
Player deployment packet targets checkpoint `916ba64`; add the scheduler source
with its own remote-hash/rollback preflight before upload. Helpers, logs and
verification are in `.tmp/backlog-execution-20260928/`.
ST ratings need a separately declared ATS study with coverage accounting.

## Open

Automatic approval review rejected external source and possible feature-table
upload to `backup-server`; explicit approval is pending. Its fallback daemon runs;
reboot persistence needs administrator-enabled lingering (`loginctl`: Access denied).
Pool entrant/prize inputs and five-directory 527.9 MiB cleanup approval remain
pending; nothing deleted. Preserve unrelated policy edits, four page greeting
changes, popup lane, and September 27 registry records.
