# Contributing

Create changes from a clean environment with Python 3.12 and uv:

```powershell
uv sync --all-groups
uv run ruff format .
uv run ruff check .
uv run mypy src
uv run pytest
```

Any feature change must document when the value becomes knowable and include a
test showing that changing a current game's outcome cannot change that game's
pregame predictors. Any model comparison must use the walk-forward runner and
retain prediction-level output, not only aggregate scores.

Do not commit raw nflverse files, credentials, cookies, fitted models, or
recommendation artifacts. Report suspected credential exposure privately and
rotate the credential before cleaning history.

Before handing work to a new session:

1. Update the roadmap and relevant design documentation.
2. Republish `CURRENT_PREDICTIONS.md` if the active weekly forecast changed.
3. Run the complete quality-gate command sequence above.
4. Run `uv run nfl-ats handoff` and review `HANDOFF.md` against live Git state.
5. Commit or push only when explicitly requested, and report the branch/hash.
