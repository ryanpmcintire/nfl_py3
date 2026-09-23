# Free odds sources

## Goal

Replace the dead paid Odds API with current free NFL spreads for private personal research, while keeping pool grades and public quote rights separate.

## State

- 2026-09-20 11:20 ET: `scripts/capture_bovada_private.py` made a genuine one-shot public NFL JSON capture at `data/market/raw/20260920T152003Z`: 15/15 future games, 60 spread/total outcome rows, Bovada only. `observed_at_utc` is actual local retrieval; per-book update is unknown. Its 15 games have real earlier Bovada quotes from Sep 14-15, so no baseline was fabricated.
- 2026-09-20 11:23 ET: `scripts/capture_odds_gap_private.py` read the documented public `/api/lineshop` once with normal headers, privately normalizing 15/15 current-week games for each Bovada, `williamhill_us` (Caesars), and MyBookie. Artifact `data/market/raw/20260920T152349Z-odds-gap-private` has 90 spread rows. `observed_at_utc` is actual retrieval 15:23:49Z; separate `source_scan_at_utc` is 15:00:46Z. Book-specific update times remain unknown. A prior request omitting its working User-Agent returned 403; the corrected public request succeeded. The Odds Gap docs permit personal assistant use with attribution and prohibit feed repackaging; these snapshots are private research only, not a public odds product.
- `scripts/capture_private_sunday_odds.py` runs both sources at the single enabled `odds_private_sun` Sunday 12:10 ET scheduler slot before 12:50 pick refresh. It skips sources captured within 30 minutes and persists source-specific HTTP 403/429/5xx stop markers. Exact argv exercised 11:27 ET: `MANUAL-RUN OK odds_private_sun`; recent capture skip avoided duplicate fetches. Restart the daemon once schedule edits settle; its current process retains the old schedule.
- `market_data.current_spread_quotes` uses the freshest snapshot per game; `public_only=True` excludes private-research snapshots via manifest scope. The board agent wired the public selector. Private model input uses genuine chronological leader history; Sunday feature worker is binding observed versus source scan time and testing actual three-book exposure. Public Books-now must not republish these vendor lines. Splash pool lines remain grading lines.
- Measured selector at as-of 15:30Z: 15 private current lines, each direct single-book Bovada (its 15:20 capture is fresher than the Odds Gap 15:00 scan); zero public current lines. As-of 15:10Z cannot see either response retrieved later. Targeted market/pick/scheduler tests: 104 passed; targeted Ruff format/check, mypy for 235 source files, and comment check pass.
- Release audit closed an incomplete-snapshot scope gap: quote history now waits for a manifest and defaults unknown providers private. Real selector still measures 15 private and zero public current lines; 104 scoped tests pass after the existing Odds API fixture gained its production-like manifest. No raw market files are tracked (`data/market/**` is ignored), no new credentials exist, and the running scheduler is one uv/venv/Python process tree.
- ESPN DraftKings per-event summary capture code and 11 disabled schedule slots exist, but today's scoreboard request returned 403. `data/market/espn_pickcenter_blocked.json` stops further requests. All disabled job argvs were manually exercised. No paid key or halves source was reactivated.
- Scheduler daemon restarted safely after the job edit: old PID 24128, new PID 25656 at 11:29 ET; `--health` reports code and schedule current. Existing unrelated missed-job health rows still exist. Exact nested `lineups_sun_am --dry` argv passed `MANUAL-DRY-RUN OK` at 11:17 ET without a rebuild or publish.
- Historical scheduler recovery: nine prior unacknowledged misses (six Sep 19 Saturday jobs and three disabled paid-odds windows) were individually acknowledged with the supported CLI, explicitly preserving MISSED status and noting that point-in-time inputs cannot be recreated. The earlier weekly-lock acknowledgement was untouched. Current `--status --brief` still shows 10 MISSED rows, all acknowledged; `--health` reports 17 historical missed windows, zero unacknowledged, code/schedule current, `OVERALL: OK`, daemon PID 25656.
- 2026-09-20 12:13 ET: the daemon was occupied by `lineups_sun` from 12:00, so the 12:10 odds window had not produced a capture. Exact `capture_scheduler.py --run-job odds_private_sun` completed `MANUAL-RUN OK`. New private Odds Gap artifact `data/market/raw/20260920T161336Z-odds-gap-private` was retrieved 16:13:36.633Z from a 16:00:49.265Z source scan, with 15/15 games each for Bovada, Caesars (`williamhill_us`), and MyBookie. New direct Bovada artifact `data/market/raw/20260920T161338Z` observed 16:13:38.427Z, with 15/15 games and 60 spread/total rows. All 45 current leader game-book pairs have real pre-Sunday history. Private current-line read chooses the fresher direct Bovada single-book quote for all 15; public current odds count remains zero. `lineups_sun` was still running under daemon PID 25656 at last check; no forecast/card/ledger publication was invoked by this odds check.

## Tried

- Sep 19 cached Action Network scoreboard does contain per-book spreads, correcting the prior “percentages only” claim. Its existing Saturday/Sunday public-betting jobs remain unchanged; no new Action request or odds normalization was made.
- Private earlier Bovada read at 10:49 ET had 15 genuine Tuesday-to-Sunday pairs and six changed spreads; the later persisted capture is the authoritative current read. The Odds Gap API docs are at https://theoddsgap.com/api-docs; Bovada terms at https://www.bovada.lv/contents/terms_of_service_bvd.pdf.
- Targeted market/pick/scheduler/trigger tests passed (123); scheduler dry argv tests passed (57). Ruff on edited odds modules passes. The full suite before concurrent root fixes had 4 failures, 38 errors, 4487 passes, 9 skips; root handles final suite.

## Next

- Run `.tools/uv.exe run --no-sync ruff format --check src/nfl_ats/market_data.py` and
  `ruff check src/nfl_ats/market_data.py` (the 2026-09-23 session hit its 50-tool-call
  cap immediately before this step; the edit itself follows the file's existing style
  so risk is low, but it is unverified).
- If ruff is clean: nothing else required for this unit; the fix is otherwise verified
  (183 targeted tests passed). If ruff flags anything, fix in place and rerun the same
  targeted test command below.
- Then re-check `docs/lanes/odds-api-key-deactivated.md`: still nothing to do there
  (owner declined re-subscribe; jobs stay disabled).

## State (2026-09-23, this session)

- **Measured**: bulk paid Odds API jobs are still disabled and have not been retried.
  `scripts/capture_scheduler.py:488-521` shows `odds_wed_opener`, `odds_thu_tnf`,
  `odds_sat` all with `enabled=False`. `data/scheduler_log.txt` has zero `odds_wed*`,
  `odds_thu*`, `odds_fri*`, `odds_sat*` entries after 2026-09-18T20:34 (the last MISSED
  row); no new 401s, matching `docs/lanes/odds-api-key-deactivated.md`'s "no
  re-subscribe" decision. Nothing changed here; no action needed.
- **Measured**: Week 3 2026 (the coming Sunday, kickoff week of Sep 27) has 16 games in
  `data/market/raw`. 15 still carry only the dead paid API's Sep 8-13 opener as their
  "current" line (no free capture has touched them yet); 1 (`2026_03_ATL_GB`) has a
  real 3-book Odds Gap read from Tue 2026-09-22T21:02:14Z. All 16 have valid
  `nflverse_game_id` mappings (verified via `data/market/raw/20260922T210214Z-odds-gap-private/quotes.parquet`,
  0 null ids) — Sunday worker's item (1) is fine, not a bug; the free Sunday job
  just has not run yet for this week.
- **Fixed (item 2/3/4)**: `src/nfl_ats/market_data.py::current_spread_quotes` used to
  pick, per game, only the row(s) tied for the single freshest `_quote_as_of`
  (scan-aware) timestamp, discarding every other bookmaker's row outright. Verified
  against the real 2026-09-20 Week-2 Sunday capture
  (`data/market/raw/20260920T161336Z-odds-gap-private` + `.../20260920T161338Z`):
  direct Bovada (`_quote_as_of`=`observed_at_utc`=16:13:38.43Z, no scan lag) always
  narrowly beat Odds Gap's own Bovada mirror (`_quote_as_of`=`source_scan_at_utc`=
  16:00:49.27Z, ~13 min self-reported lag), so the old code silently dropped Odds
  Gap's live-retrieved Caesars/MyBookie rows too, reporting `bookmakers=1` for all 15
  games even though a genuine 3-book read existed from the same capture cycle. Item
  (2) (direct Bovada must win over the Odds Gap Bovada mirror) and item (3) (the
  `as_of` eligibility filter uses `observed_at_utc`, real retrieval time, not
  `source_scan_at_utc`) were already correct and are unchanged.
  Fix: `current_spread_quotes` now picks the freshest scan-aware quote **per
  (game, bookmaker)** instead of per game (`market_data.py` around line 419-427),
  then keeps only bookmaker rows within `CURRENT_QUOTE_COVERAGE_WINDOW` (30 minutes,
  new module constant near line 29) of the game's freshest quote before aggregating
  the median/`bookmakers`/`bookmaker_label`. The 30-minute bound reuses
  `scripts/capture_private_sunday_odds.py --max-age-minutes` default (its own
  same-cycle recency threshold) rather than inventing a new constant, and stops old
  the-odds-api books (days stale) from being blended into a "current" read — an
  earlier no-window version of this fix was tried and rejected for exactly that
  reason (see Tried).
  **Verified** against real Sept 20 data: all 15 Week-2 games now correctly report
  `bookmakers=3` (`Bovada, Caesars, MyBookie`) with `provider_label` =
  `bovada_public_nfl, the_odds_gap_lineshop_private`; `DET_BUF` (no free-source data)
  is untouched at `bookmakers=11` from `the-odds-api`; `as_of=2026-09-20T15:10Z`
  still correctly sees only pre-15:10 data (item 3 unaffected).
- **Confirmed (item 5)**: board already renders no private vendor raw quotes.
  `board_content.py:3056` calls `current_spread_quotes(..., public_only=True)`; the
  board only ever displays the aggregated `home_spread_line` number plus a plain
  `market_now_book_label` (e.g. "Bovada" or "3 books") via
  `board_terminal.py:871-878` — never a raw per-book quote. Unchanged.
- Targeted verification run: `.tools/uv.exe run --no-sync pytest -q
  tests/test_board_content.py tests/test_board_terminal.py tests/test_pick_refresh.py
  tests/test_clv.py tests/test_odds_ingest_halves.py tests/test_refresh_triggers.py`
  → 183 passed, 6 warnings (pre-existing, unrelated: bootstrap degeneracy, bitwise-`~`
  deprecation, injury-snapshot fallback in an unrelated fixture, pytest cache
  permission). No test file was added or edited (moratorium respected).
  `ruff format --check` / `ruff check` on `src/nfl_ats/market_data.py` were queued as
  the very next command when this session hit its 50-tool-call cap — **not yet run**.

## Tried

- (this session) A first version of the per-bookmaker fix had no time bound at all:
  it merged each bookmaker's all-time latest quote regardless of age. Rejected after
  reproducing against real data — it blended the dead paid API's Sep 8-13 book prices
  (BetMGM, DraftKings, FanDuel, etc., 11 books) into the "current" line for every
  Week-2 game alongside the fresh Bovada/Caesars/MyBookie read, which violates item
  (3)'s point-in-time intent. Replaced with the 30-minute-windowed version above.
- Sep 19 cached Action Network scoreboard does contain per-book spreads, correcting the prior "percentages only" claim. Its existing Saturday/Sunday public-betting jobs remain unchanged; no new Action request or odds normalization was made.
- Private earlier Bovada read at 10:49 ET had 15 genuine Tuesday-to-Sunday pairs and six changed spreads; the later persisted capture is the authoritative current read. The Odds Gap API docs are at https://theoddsgap.com/api-docs; Bovada terms at https://www.bovada.lv/contents/terms_of_service_bvd.pdf.
- Targeted market/pick/scheduler/trigger tests passed (123); scheduler dry argv tests passed (57). Ruff on edited odds modules passes. The full suite before concurrent root fixes had 4 failures, 38 errors, 4487 passes, 9 skips; root handles final suite.

## Open

- ruff format/check on `src/nfl_ats/market_data.py` for this session's edit is
  unverified — do this first in the next subtask.
- Neither free source provides a confirmed book-specific quote update time. No public odds redistribution is authorized; sourced decisions and private analysis remain separate. ESPN stays blocked at HTTP 403. Public Action odds path needs distinct provenance/access review before adding it as a normalized feed.
- The free Sunday capture job (`odds_private_sun`) has not run yet for the coming
  Week 3 Sunday (Sep 27); 15 of 16 games still show only the stale paid-API opener as
  "current". That is expected this early in the week, not a bug, but re-check after
  the next Sunday run that the 3-book merge in `current_spread_quotes` behaves the
  same way it did for Week 2.
- No Odds API billing action needed; owner already declined re-subscribe
  (`docs/lanes/odds-api-key-deactivated.md`).
