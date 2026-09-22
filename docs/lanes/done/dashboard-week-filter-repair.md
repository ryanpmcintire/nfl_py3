# Dashboard week filter repair

## Goal
Preserve the dashboard and filter its weekly card and game details. Default to the current week. The owner rejected replacing the page with a plain archive table; future filtering must retain the existing design and interaction.

## State
- Implemented: only the weekly card is swapped; dashboard hero, metrics and evidence remain mounted. Original published game-room state survives week changes. Archived cards use the existing table/inspector design with selection, keyboard and previous/next controls. Week-specific notes follow their card; stale game tickers are hidden.
- Measured: desktop and 390px mobile checks passed for Weeks 1, 2 and 3, including URL selection, current-week reset, browser Back, archive selection, live keyboard selection after returning, analysis tabs and game room. One card and one selected detail remain visible; no horizontal overflow. Screenshots and check logs: `.tmp/week-navigation/`.
- Measured: `ruff format --check .`, `ruff check .`, `mypy src`, and `pytest -q --basetemp=.tmp/week-navigation/pytest-final` via Windows `uv run --no-sync` passed: 4,529 tests, 9 skipped. `node --check src/nfl_ats/board_week_navigation.js`, `git diff --check`, and `nfl-ats publish-board` passed.
- Measured rendered diff: `docs/index.html` +130/-93; findings, history and model pages each +2/-2 (shared keyboard guard and generation stamp). No new tests. Post-review sweep found no added research-only assertions or redundant test coverage.

## Tried
- Compared the rejected page with the original dashboard in an isolated browser. Kept inactive archives in a template so existing document-wide game handlers bind only to their own live card.
- Measured scheduler startup: `--once` then `--status --brief` showed the existing daemon running, 206 OK and 10 MISSED jobs, including today's weekly lock and nine older rows. No scheduler job changed; the running daemon was retained.

## Next
Continue bounded backlog work from the active lane index after this repair is deployed. Keep filtering local to the card.

## Open
- Week 3 is the default and truthfully shows picks pending until genuine locked pool lines are acquired. Acquisition is tracked in `../week3-lines-2026-09-22.md`; never ask the owner to transcribe lines or fabricate picks.
- Preserve unrelated prediction card, tiebreaker and six untracked experiment-registry changes.
