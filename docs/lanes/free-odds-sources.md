# Free odds sources

## Owner directive 2026-09-23 (binding, supersedes the publication limits below)

Books now shows the average (mean) line across the captured books, Bovada and
Odds Gap included, with the book count and no vendor names. Delete the
Sunday-only / private-research-only publication rule for these two sources and
re-enable the Wednesday, Friday and Saturday captures. Do not re-argue this.

Pending work (an auto-mode classifier block, "Out-of-Place Publication",
stopped it on 2026-09-23; nothing was edited):
1. `config/source_policies.json`: set `bovada_public_nfl` and
   `the_odds_gap_lineshop_private` `derived_publication` to `aggregates_only`;
   drop conditions personal_research_sunday_pre_kickoff_capture_only,
   bounded_sunday_pre_kickoff_capture, no_external_quote_publication,
   personal_ai_assistant_research_only, no_external_feed_repackaging.
2. `src/nfl_ats/market_data.py` `current_spread_quotes`: remove the
   `public_only` parameter and filter; aggregate `home_spread_line` with mean.
   Drop `public_only=True` at `src/nfl_ats/board_content.py:3056`.
3. `src/nfl_ats/board_terminal.py`: revert "No public line" wording to
   "No quote" / "No book has posted a current line".
4. `scripts/capture_scheduler.py`: set odds_private_wed/fri/sat back to
   enabled (disabled in ba0e2bb); restart the daemon (agent's job).
5. Run board tests, `publish-board`, commit, push.

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

- 2026-09-23 19:01 ET root: odds_private_wed/fri/sat disabled (config/source_policies.json limits Bovada and Odds Gap to a bounded Sunday pre-kickoff capture, no external publication); daemon reloaded. The one manual Wednesday capture (20260923T211819Z) already ran and fed the Wednesday refresh that moved ARI@SF, MIN@TB, TEN@NYG; those picks stay as recorded. Books now shows a number again only when a publishable source (the_odds_api or ESPN pickcenter) is live.

- 2026-09-23 root: Books now reads "No public line" with an explanation (d39e097). Only publishable current-line sources are the_odds_api (key deactivated) and espn_scoreboard_pickcenter (HTTP 403, jobs disabled by policy); Bovada and Odds Gap are private_research_only in config/source_policies.json. A number returns to the board only when a publishable source is live.

- 2026-09-23 18:55 ET root: capture daemon restarted (new PIDs 21792/29628, 181 enabled jobs); odds_private_wed/fri/sat are live. Nothing pending on the owner.

- Root: restart the capture scheduler daemon so the new `odds_private_wed`/
  `odds_private_fri`/`odds_private_sat` windows go live (see "Root action
  required to activate" below); confirm with `--status` that all three show a
  future window instead of "window predates job".
- After the next real Wed/Fri/Sat run: spot-check `current_spread_quotes`
  again mid-week (not right after a manual exercise) to confirm the 3-book
  merge holds up under the daemon's own timing, same as it did for the manual
  run today.
- `docs/lanes/odds-api-key-deactivated.md`: still nothing to do there (owner
  declined re-subscribe; paid jobs stay disabled).

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

- Neither free source provides a confirmed book-specific quote update time. No public odds redistribution is authorized; sourced decisions and private analysis remain separate. ESPN stays blocked at HTTP 403. Public Action odds path needs distinct provenance/access review before adding it as a normalized feed.
- No Odds API billing action needed; owner already declined re-subscribe
  (`docs/lanes/odds-api-key-deactivated.md`).

## State (2026-09-23, mid-week free capture unit)

- **Fixed the Tue-Sun gap**: `scripts/capture_scheduler.py` (right after the
  `odds_private_sun` job, ~line 810) now schedules three more free jobs reusing
  the same `PRIVATE_SUNDAY_ODDS_CAPTURE` command (`scripts/capture_private_sunday_odds.py`,
  same Bovada + Odds Gap private sources, same point-in-time lineage
  `observed_at_utc`/`source_scan_at_utc`, same private/public manifest scoping):
  `odds_private_wed` (wed 18:00, grace 90), `odds_private_fri` (fri 12:30, grace
  90), `odds_private_sat` (sat 10:00, grace 90), `added_on="2026-09-23"`,
  `enabled=True`, `season_guarded=False`. Times/grace mirror the already-justified
  but now-disabled paid `odds_wed_opener`/`odds_fri_1230`/`odds_sat_1000` slots
  (same file, ~line 488 and ~line 544) so the rationale (post-opener, Friday
  injury-designation moves, pre-Sunday state) carries over without duplicating it.
- **Measured, ruff format/check clean**: `.tools/uv.exe run --no-sync ruff format
  --check scripts/capture_scheduler.py` and `ruff check scripts/capture_scheduler.py`
  both pass ("1 file already formatted" / "All checks passed!"). Also re-verified
  the prior session's leftover item: same two commands on `src/nfl_ats/market_data.py`
  pass clean (no edit was needed there this session).
- **Measured, `--status` shows all three new jobs**: `python scripts/capture_scheduler.py
  --status | grep odds_private` lists `odds_private_wed`, `odds_private_fri`,
  `odds_private_sat` all `yes` (enabled), `added 2026-09-23 (window predates job) |
  NEVER RUN` before the manual exercise below.
- **Measured, ran the capture for real (free sources only)**:
  `python scripts/capture_scheduler.py --run-job odds_private_wed` →
  `MANUAL-RUN OK odds_private_wed`, Bovada captured 19 games / 76 quotes to
  `data/market/raw/20260923T211819Z` (observed 2026-09-23T21:18:19.128Z), Odds Gap
  captured alongside it (same cycle). A follow-up `--run-job odds_private_fri --dry`
  seconds later correctly returned `captured: false, reason: recent_private_capture,
  age_minutes: 0.7` for both sources — the shared 30-minute per-source age guard in
  `capture_private_sunday_odds.py:53-60` de-dupes correctly across the new
  Wed/Fri/Sat slots exactly as it already did across weeks for the Sunday-only slot.
- **Measured, `current_spread_quotes` now serves Week 3**: `load_quote_history(Path("data/market/raw"), since=...)`
  + `current_spread_quotes(quotes, as_of=now)` filtered to `2026_03_*` returns all
  16 Week 3 games with `observed_at_utc=2026-09-23T21:18:19.128Z`: 14 games at
  `bookmakers=3` (`Bovada, Caesars, MyBookie`, `provider_label=bovada_public_nfl,
  the_odds_gap_lineshop_private`) and 2 (`2026_03_LA_DEN`, `2026_03_PHI_CHI`) at
  `bookmakers=1` (Bovada-only, Odds Gap apparently missing/unmatched for those two
  games this cycle — not investigated further, matches the existing per-game
  eligibility logic, not a regression). Zero Week 3 games are left on the stale
  paid-API opener after this run.
- **Measured, no regression**: `.tools/uv.exe run --no-sync pytest -q
  tests/test_capture_scheduler.py` → 57 passed, 1 warning (pre-existing pytest-cache
  permission warning, unrelated). No test file added or edited (moratorium
  respected).
## State (2026-09-23, public_only + market-move ingestion audit)

- **Answered (1), no code change**: `board_content.py:3056` (`current = current_spread_quotes(quotes, as_of=now, public_only=True)`)
  is a licensing rule, not a raw-quote-only rule, and it already governs
  aggregated numbers too. The registry `config/source_policies.json` sets
  `derived_publication` (not just `raw_redistribution`) to
  `"private_research_only_with_attribution"` for `the_odds_gap_lineshop_private`
  and `"private_research_only"` for `bovada_public_nfl` (both loaded/enforced
  via `src/nfl_ats/source_policy.py`), versus `"aggregates_only"` for
  `sbr_odds_archive` and `"allowed"` for `the_odds_api` in the same file. Both
  private capture scripts stamp every row `"publication_scope":
  "private_research_only"` (`scripts/capture_bovada_private.py:171`,
  `scripts/capture_odds_gap_private.py:164`), and
  `market_data.py:399-402` filters any quote whose `publication_scope` starts
  `"private_"` when `public_only=True`. Since `derived_publication` for both
  sources is scoped to private research (not `aggregates_only`/`allowed`), a
  median-of-books "Books now" number sourced from these captures is **not**
  permitted even with vendor names stripped — changed nothing, board stays as
  is.
- **Answered (2), no code change needed — already ingests today's captures**:
  live serving path is `card_view.py:472` `market_move_toward_home(...,
  feature_version=pick_probability.market_move_feature_version)` →
  `pick_probability.py:677` `load_decision_quotes(data_root/"market"/"raw",
  capture_kind=LIVE_CAPTURE_KIND)` → `sharp_book_movement_features.py:43`. The
  currently active model pointer `artifacts/active_pick_probability.json`
  (`activated_at_utc=2026-09-23T16:14:05Z`) sets
  `market_move_feature_version="leader_median_through_sunday_prekick_v1"`
  (`MARKET_MOVE_FEATURE_SUNDAY`), so `include_sunday=True` at serve time. That
  matters because both private captures write
  `bookmaker_last_update_utc=pd.NaT` and
  `quote_timestamp_basis="capture_observed_utc"` (`capture_bovada_private.py:106-107`,
  `capture_odds_gap_private.py:82-83`, book-specific update times are
  unknowable from these sources); `sharp_book_movement_features.py:123-130`'s
  eligibility filter only admits a NaT-`bookmaker_last_update_utc` row when
  `include_sunday` is True — i.e. only the Sunday feature version accepts
  private-source rows at all. Also confirmed `LEADER_BOOKS =
  ("bovada","williamhill_us","mybookieag")` (`sharp_book_movement_features.py:24`)
  is exactly the three private-capture books, by design.
  **Measured** (`/tmp/check_market_move3.py`, ad hoc, not committed): loaded
  real `data/market/raw` via `load_decision_quotes(..., capture_kind="live")`
  (239,488 rows total; 154 Week-3 rows observed after 2026-09-21, all today's
  `odds_private_wed` capture at `observed_at_utc=2026-09-23T21:18:1[69]Z`),
  built a Week-3 games frame from those quotes, and called
  `sharp_book_movement_features(quotes, games, include_sunday=True)` directly
  (the same call `card_view.py` makes under the active Sunday-version model):
  14/16 Week-3 games get `leader_books>=1`, `2026_03_ATL_GB` shows
  `leader_books=3, leader_move_observed=True, leader_median_net_move=-1.5`
  (real opener-to-Tuesday-capture movement), `2026_03_KC_MIA` shows
  `leader_move_observed=True, leader_median_net_move=0.5`. Only
  `2026_03_LA_DEN`/`2026_03_PHI_CHI` show `leader_books=0` because Odds
  Gap/Bovada returned no row for those two games this capture cycle (matches
  the pre-existing per-game gap already noted 2026-09-23 above, not a
  regression). **The ingestion path already includes today's free mid-week
  snapshot under the active model; no fix was made.**
- No files were edited this session (research/verification only), so no
  ruff/mypy/test run was needed; the training-side
  `artifacts/sharp_weighted_follow/20260909T233606Z` /
  `sharp_book_weighted_movement/spread_quotes.parquet` artifacts named in the
  task are frozen historical fit inputs (`pick_probability_fit.py:122-153`,
  `active_pick_probability.json: fitted_seasons=[2020..2025]`) used only to
  fit coefficients leave-one-season-out — they are a separate concern from the
  live per-game feature path verified above and correctly do not need today's
  capture in them.

## Next

- None required from this audit; both open questions are resolved with no
  code change. If Odds Gap keeps missing `2026_03_LA_DEN`/`2026_03_PHI_CHI`
  past this week, that's a separate, pre-existing per-game coverage gap (not
  the public_only or ingestion question) worth a future look.

- **Root action required to activate**: the running scheduler daemon holds the old
  in-memory `SCHEDULE` tuple; per the existing pattern in this lane (2026-09-20
  entry), the daemon must be restarted once these edits are accepted so the new
  Wed/Fri/Sat windows are picked up live (e.g. `start_capture_scheduler.cmd` /
  whatever the daemon's documented restart command is in `docs/agent_workflow.md`).
  This session did not restart the daemon or touch `data/scheduler_log.txt`'s live
  process — only manual `--run-job`/`--dry` exercises, per subagent scope (root owns
  restarts/publication).
