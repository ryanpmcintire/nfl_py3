# Backlog execution, September 28

## Goal

Complete executable work from the single `ROADMAP.md` backlog without changing
its counting rules or preserving a separate list of implementation categories.

## State

Three reviewed repairs are ready: backup completeness, capture-sync integrity,
and lower-memory player ingestion. Files: `scripts/offsite_backup.py`,
`scripts/sync_captures.py`, `src/nfl_ats/players.py`. Current operations evidence
is in `docs/offsite_backup.md`; OPS-02/OPS-06 progress is recorded in `ROADMAP.md`.
No full backlog row closed: **measured**, 56 total, 13 done, 43 remaining.

## Tried

**Measured:** deployed 499 current capture-code files from `916ba64` with hashes
and rollback; seven new aliases passed, five full-history player jobs hit OOM.
Backup status and a 1,650-byte restore passed with identical content. Final sync
dry run retained five conflicts. Lower-memory output equivalence, integrity
probes, and independent reviews passed. Full validation (locked uv, inherited
`UV_CACHE_DIR=.tmp/uv-cache`): `ruff format --check .`, `ruff check .`, `mypy src`,
`pytest -q --basetemp .tmp/backlog-execution-20260928/pytest-ops-final` all passed;
1,645 tests, no tests added. No redundant test deletion is proposed.

## Next

On explicit deployment approval, upload the reviewed player source and helpers
from `.tmp/backlog-execution-20260928/`, apply the hash-checked rollback-preserving
patch, and rerun the five real player aliases on the 512 MiB host. Run the final
primary sync scheduler job, then update measured receipts. Do not equate local
memory probes with a constrained-host success. Keep the canonical count intact.

## Open

Automatic approval review rejected new source upload and possible feature-table
upload to `backup-server`; explicit approval is pending. The fallback daemon is
running, but reboot persistence needs administrator-enabled lingering (`loginctl`
returned Access denied). Pool entrant/prize inputs and exact five-directory
527.9 MiB cleanup approval are pending; nothing was deleted. Preserve unrelated
policy edits, four page greeting changes, popup lane, and September 27 registry
records. Command receipts and deployment manifests: `.tmp/backlog-execution-20260928/`.
