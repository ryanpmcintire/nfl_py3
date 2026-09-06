# Dashboard designs

The interactive design is now the [main dashboard](https://ryanpmcintire.github.io/nfl_py3/). The old `ball-week.html`, `ball-model.html`, `ball-history.html`, and `ball-findings.html` addresses redirect to the corresponding main pages, preserving matchup anchors.

The publisher renders the current card through `src/nfl_ats/board_interactive.py`. Layout, motion, and interaction assets are packaged beside that renderer. Picks, scores, explanations, lineup dates, and confirmed finals come from the same content used by the main board; the retired saved card and its duplicate assets have been removed.

The dashboard keeps the game room, both teams' field formations, player details, score scenarios, receipts, keyboard navigation, and interactive Model, History, and Findings pages. Pinning, the watchlist, and share-card export were removed at the owner's request on 2026-09-06.

Earlier `command-center*.html` files are archived design sketches.

## Verification on 2026-09-06

Measured: `node .tmp/verify_interactive.cjs` passed 41/41 browser checks with zero exceptions, including all four pages at mobile width and Week 8 fixtures with unavailable lineups. Screenshots: `.tmp/interactive-main-desktop.png`, `.tmp/interactive-*-mobile.png`. Fixture and site tests: `tests/test_board_interactive.py` and `tests/test_board_site.py`, 13 passed.

Rendered changes: all four main pages use the interactive presentation and normal artifact-backed publication; old preview pages redirect. `publish-board` remains the sole production generator.

Measured final checks (all via `.\.tools\uv.exe run`): `ruff format --check .`, `ruff check .`, and `mypy src` passed. Ruff reported the existing inaccessible `.pytest-temp-lineup` directory. Its formatter initially crashed while reporting an unformatted new string; formatting that source file resolved the crash, and the required whole-repository command then passed.

Measured: `pytest --basetemp=F:\Repos\nfl_py3\.tmp\pytest-interactive-full -o cache_dir=F:\Repos\nfl_py3\.tmp\pytest-interactive-full-cache -n 4 -q --tb=short` completed with 5,504 passed and six failures (`.tmp/interactive-pytest.txt`). All six also failed in this conversation's pre-change run (`.tmp/publish-pytest.txt`): two experiment-registry scanner contracts, drift step ordering, expectation copy, tiebreaker test arguments, and local totals-wave2 data alignment. No new Python test failed.

The owner requested removing the deleted-feature absence assertion after that full run; it was removed. Measured final rerun: 13 fixture/site tests passed, and 41 browser checks passed.
