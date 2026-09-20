# Windows/Linux environment and Shopify startup recovery

## Goal

Restore the Windows project after a shared-checkout Linux session and fix the
unrelated Shopify skill startup warning without removing the plugin.

## State

- Measured 2026-09-20: Windows Python existed but `nfl_ats` and `tzdata` were
  missing; `.tools/uv.exe sync --locked` installed all 48 locked packages.
  Imports and `nfl-ats doctor` pass. Scheduler restarted headlessly using
  `scripts/start_capture_scheduler.cmd`; `--is-running` confirms PID 24128.
- Read: prior Linux overwrite was documented in `free-odds-sources.md`.
  Prevention now lives in `AGENTS.md` and `docs/windows_linux_environment.md`:
  shared-checkout Linux uses `.venv-linux`, Windows keeps `.venv`.
- Measured: repaired metadata/hooks YAML in the user-level cached Shopify
  4.0.1 `skills/shopify-shopifyql/SKILL.md`. Codex `skills/list` reproduces
  the exact original error and loads the repaired copy with no errors.
  Original backup and loader evidence: ignored `data/environment_recovery/`.
- Dashboard source adds keyboard Skip to content navigation on four pages.
  Mechanical cleanup of 37 pre-existing lint errors in
  `scripts/every_metric_backfill.py`; no experiment was executed.

## Tried

- Measured: `uv run --no-sync ruff format --check .` and `ruff check .` pass;
  `mypy src` passes (235 files); `pytest -q`: 4,529 passed, 9 skipped,
  133 warnings (including a non-fatal pytest cache permission warning).
  Commands use `.tools/uv.exe`; logs are in `data/environment_recovery/`.
- Measured: scheduler `--once` completed catch-up captures. Six Saturday
  lineup/refresh windows remain truthfully MISSED; no historical captures
  were fabricated. Other missed entries predate this outage or are disabled.
- `publish-board` initially refused a stale player-arrests snapshot;
  `--run-job player_arrests_tue` refreshed it with the real scheduled argv:
  `2026-09-20T07:39:34-04:00 MANUAL-RUN OK player_arrests_tue`.
  `nfl-ats publish-board` then passed and wrote all four pages. Measured
  rendered diff: +8/-1 lines per page, each with one skip link and focusable
  main landmark; current-week page also updates the next check and injury
  feed freshness. Full visible diff: `data/environment_recovery/board_*`.
- Measured: `scripts/strip_comments.py --check` and `git diff --check` pass.
  `nfl-ats handoff` refreshed `HANDOFF.md`; final freshness check passes.
- Post-review residue proposal: existing
  `test_terminal_style_css_constant_matches_asset_file` repeats the same CSS
  file read as production. Left intact for owner review; no tests added.

## Next

- Repair complete. Restart Codex to clear the already-displayed startup
  warning. Free-odds backlog continues from `../free-odds-sources.md`.
- Local Git changes: `.gitignore`, `AGENTS.md`, `HANDOFF.md`, `ROADMAP.md`,
  lane index and this lane, environment guide, free-odds lane, four rendered
  pages, board renderer/CSS, mechanical backfill lint fix, and the preserved
  pre-existing `config/source_policies.json` edit. Nothing committed/pushed.

## Open

- No commit/push requested; repository hygiene forbids either without an
  explicit request. Existing free-odds source-policy changes are preserved.
- A plugin reinstall/update may overwrite the local Shopify YAML repair.
