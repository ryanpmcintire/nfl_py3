# Current-season injury capture integrity

## Goal

Reject injury captures that omit the current season, without blocking useful current data because an older season failed.

## State

Completed and verified on 2026-09-22. `scripts/nflverse_injuries_ingest.py::run_ingest` checks the current-season fetch before creating a capture directory or replacing output. Historical failures remain tolerated. CLI help uses a literal description.

Measured: a failed current-season CLI capture leaves no directory and preserves existing parquet and manifest bytes. A failed historical season still permits a successful current-season capture. Read: `scripts/capture_scheduler.py` counts timestamp directories when checking capture freshness, so leaving an empty directory would suppress retries.

## Tried

Root verification, all exit 0:
- `uv run --no-sync python .tmp/verify_injury_current_season.py` (ignored CLI fixture, provider responses controlled).
- `uv run --no-sync ruff format --check .` and `ruff check .`.
- `uv run --no-sync mypy src`: 237 source files.
- `uv run --no-sync pytest -q --basetemp .tmp/week-navigation/pytest-capture-final`: 4529 passed, 9 skipped, 162 warnings.
- `uv run --no-sync nfl-ats publish-board`: all four pages generated; measured rendered-page diff: zero for this follow-up.

Logs are under `.tmp/week-navigation/`, including `injury-cli-capture-final.log`, `format-capture-final.log`, `lint-capture-final.log`, `mypy-capture-integrity.log`, `pytest-capture-final.log`, and `publish-capture-integrity.log`. Post-review residue sweep found no added committed tests, assumed-behavior fixtures, or research-only assertions. Scheduler job definitions and argv did not change.

## Next

Continue Week 3 acquisition from `../week3-lines-2026-09-22.md` when authentic locked pool lines arrive. Check live Git state and HANDOFF for release status. Broader LEAD-64 work needs observed Friday/Sunday acquisition cycles with current data.

## Open

This guard does not establish upstream publication coverage. The user can supply Week 3 lines in normal chat; no interaction with the queued input control is required. Existing unrelated prediction-card, tiebreaker, and experiment-registry changes remain preserved.
