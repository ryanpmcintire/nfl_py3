# Tiebreaker total move serving

## Goal
Serve unit 3 for the score guess only, preserving Tuesday's baseline, the pick, and its deadline.

## State
**Measured:** code and verification complete; no publication, commit, registry write, or live tiebreaker mutation. `artifacts/tiebreaker_total_move/fit.json` stores slope 0.8, 1,333 usable games, hashes and coverage. Production reproduces all 1,333 OOS score pairs and zero-adjustment identities. Six LOSO slopes: 0.667, 0.667, 0.8, 0.5, 1, 1.
**Measured:** OOS MAE 10.315079 versus 10.368342; improvement 0.053263 [0.009901, 0.097524], probability_positive 0.9923. IS improvement 0.075019 [0.026219, 0.126583]; gap -0.021755 [-0.048013, 0.003779]. Last-game improvement 0.099010 [-0.019802, 0.217822], probability_positive 0.95255, n=101.
**Measured:** week-4 dry refresh: Tuesday total 48.5, current 48, adjustment -0.4, applied 0 (same cell), NO 25–ATL 23 retained; written=false. Deadline October 4, 4 PM Eastern.

## Protocol (declared before outcome access)
Completed 2020+ regular-season games with unit-3 paired totals/four-term baseline; historical opener proxies the pool line. Target actual total; no-intercept LAD slope against served integer total. Reuse six LOSO folds and two last/all MAE looks; 10,000 season-stratified week-block draws, seed 88. Fit all usable rows for serving. Preserve zero/same-cell identity and fixed-side fallback. One fitted probability selects the side; zero crossing closes nothing. No protocol revisions/upstream rebuild.

## Tried
Files: `src/nfl_ats/tiebreaker_total_move.py`, `src/nfl_ats/publishing.py`, `src/nfl_ats/cli_commands/publishing.py`, `src/nfl_ats/board_content.py`, existing recorder mock `tests/test_cli.py:848`; no new tests.
With `.tools/uv.exe run --no-sync` and local UV_CACHE_DIR:
- `python -m nfl_ats.tiebreaker_total_move --baseline tests/scratch/codex/lead88_unit2/baseline.parquet`
- `nfl-ats refresh-picks --help`; `nfl-ats refresh-picks --season 2026 --week 4 --dry-run`
- `mypy src`; `ruff check` and `ruff format --check` on all five Python files
- `pytest -q -n 2 -k 'tiebreaker or refresh or board' --basetemp .tmp/pytest-tiebreaker-total-move-20260930T0024`
**Measured:** Ruff passes; mypy 254 files clean; pytest 413 passed, 15 warnings. Resolved sandbox snapshot/temp access and an existing mock failure. First pytest inherited auto-concurrency; final run explicitly used two workers.
Production replay regenerated IS/OOS intervals. Rows/summary: `tests/scratch/codex/tiebreaker_total_move_serve/`; logs: `.tmp/tiebreaker-total-move-*.log`.

## Next
Orchestrator reviews/integrates code and retains the ignored fit artifact; orchestrator owns publication/commits.

## Open
**Measured:** 1,663 completed games; 1,343 baselines, 10 declined, 320 missing construction. All 48 completed 2026 games lack Tuesday totals: live totals cover 31 games, Tuesday-window coverage zero. Additional baseline files can extend the fit. **Read:** unit-3 retrospective upstream limitations remain. Owner may review/consolidate mock-heavy recorder coverage; none deleted.

## Record commands
None: reproduces existing unit-3 looks, no new verdict. Orchestrator owns existing record commands.
