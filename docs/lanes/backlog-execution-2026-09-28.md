# Backlog execution, September 28

## Goal

Complete executable work from the single `ROADMAP.md` backlog without changing
its counting rules or preserving a separate implementation-category list.

## State

OPS-06 repairs are pushed in `e350601` (handoff `1c6d966`): incomplete backups
fail before pruning; sync verifies SHA-256 and retains all remote captures;
player ingestion processes seasons/datasets sequentially, bypasses its cache,
and stages complete snapshots. Evidence: `docs/offsite_backup.md`.
PER-09's season-lagged ST builder is implemented and ran its four declared fits:
7,413 player-season ratings, all unserved. Evidence and command:
`docs/st_player_ratings.md` sections 10–11. No full row closed: **measured**,
56 total, 13 done, 43 remaining. The remaining ATS evaluation is distinct from
this completed construction step, but stays in the same PER-09 backlog row.

## Tried

**Measured:** deployed current capture code with hashes/rollback; seven new
aliases passed, five player jobs hit OOM. Backup status and one-file restore
passed; final sync dry run retained five conflicts. Memory/output, cache-restore,
atomicity, and sync-integrity probes passed with independent review. ST artifact
verification checked 12 source partitions, declaration/code/output hashes,
chronology and exposure arithmetic. Early fit windows have only 141/131 plays;
all declared outputs remain retained. No research closure or serving change.
Final checks use `.tools/uv.exe run --no-sync` with inherited
`UV_CACHE_DIR=.tmp/uv-cache`: `ruff format --check .`, `ruff check .`, `mypy src`,
`pytest -q --basetemp .tmp/backlog-execution-20260928/pytest-verified-final`.
All passed: 1,645 tests. No tests added or redundant existing-test deletion proposed.

## Next

After explicit approval, deploy the final player repair and rerun the five real
player aliases on the 512 MiB host, then run the final primary sync job. Reviewed
source, hash manifest against remote checkpoint `916ba64`, rollback-preserving
helper, command logs and artifact verification are in
`.tmp/backlog-execution-20260928/`. ST ratings need a separately declared ATS
study with coverage accounting before any serving consideration.

## Open

Automatic approval review rejected new source upload and possible feature-table
upload to `backup-server`; explicit approval is pending. The fallback daemon runs;
reboot persistence needs administrator-enabled lingering (`loginctl`: Access denied).
Pool entrant/prize inputs and exact five-directory 527.9 MiB cleanup approval
remain pending; nothing was deleted. Preserve unrelated policy edits, four page
greeting changes, popup lane, and September 27 registry records.
