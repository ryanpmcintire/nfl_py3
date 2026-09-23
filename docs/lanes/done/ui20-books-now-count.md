# UI-20: Books-now count in the This Week board

## Goal
Make the "Books now" cell/tooltip tell the reader how many books the current
line reflects (e.g. "3 books" vs "1 book"), reading the count from the same
current-line data the cell already used, without showing any sportsbook
vendor name (AGENTS.md: no vendor names of private sources in reader text).

## State (uncommitted, in working tree, not committed/pushed)
The plumbing (`GameRow.market_now_books` / `market_now_book_label`,
`_market_now_by_game`, the `market-source` span in board_terminal.py) already
existed since commit 5fb88a2 (Sep 20) and already read `bookmakers`/
`bookmaker_label` from `current_spread_quotes` (fixed in 6ac6a2e to count
every fresh book, not always 1). The bug: for `books == 1` it rendered the
raw `bookmaker_label` (a real sportsbook name, e.g. "DraftKings") as reader
text — a vendor-name violation. Fixed by three small edits:

1. `src/nfl_ats/board_content.py` `_market_now_by_game` (~line 3072): label
   is now always a plain count — `"1 book"` if `books == 1` else
   `f"{books} books"` — never the vendor name. `bookmaker_label` column is
   still read into the dataframe but no longer used for display text.
2. `src/nfl_ats/board_terminal.py` `market_help` tooltip paragraph
   (~line 892-894): "The source appears below the line." ->
   "How many books agree appears below the line."
3. `src/nfl_ats/board_terminal_style.css` (~line 264, before the existing
   `.market-now .market-move` rule): added
   `.market-now .market-source{display:block;font-size:9px;color:var(--text-faint);margin-top:2px;white-space:nowrap}`
   so the count renders as a quiet muted line under the spread number
   (previously unstyled — this class had no CSS rule at all before this
   change, so it just ran into the number).

Reader sees, on the This Week board's "Books now" column: the current
spread for the pick's side, then a small faint line underneath reading
"1 book" or "3 books" (mobile row label uses the same text via
`data-label`), then the move-since-pool-line note below that.

## Tried
- Confirmed via `git log -S` that `market_now_book_label` and
  `_market_now_by_game`'s bookmaker_label branch predate this task
  (introduced 5fb88a2, Sep 20); the vendor-name leak was pre-existing, not
  introduced by 6ac6a2e.
- Confirmed no existing test asserts on `market_now_book_label` /
  `market-source` text (grep of tests/test_board_content.py,
  tests/test_board_terminal.py) — no contract test needed editing.
- `docs/index.html` currently shows "No quote" for every Week 3 game
  (market_now is None in the committed artifact), so a raw
  `diff docs/index.html` vs a fresh scratch regen is dominated by unrelated
  live-data drift (same Week 3, but probabilities/picks refreshed since
  docs/index.html was last published) — not usable to isolate this change.
- Instead ran an isolated A/B: reverted the 3 edits with the Edit tool,
  regenerated to scratch dir `ui20-baseline`
  (`nfl-ats publish-board --site-destination
  <scratchpad>/ui20-baseline`), reapplied the 3 edits, regenerated again to
  `ui20-site`, then `diff ui20-baseline/index.html ui20-site/index.html` ->
  saved to `<scratchpad>/isolated.diff`, **6 lines total, not yet read**
  (hit the 50-tool-call cap on the Read of that diff file). Expect: 1 CSS
  line (the new `.market-source` rule) + the market_help sentence line +
  1-2 lines per game row that currently has a live quote with the "1 book"/
  "N books" span text. Scratch dirs both live under
  `C:\Users\Ryan\AppData\Local\Temp\claude\F--Repos-nfl-py3\3de407a1-acbe-4e06-9e02-f857ffb073d6\scratchpad\`
  (`ui20-baseline`, `ui20-site`) — session-scoped temp, may not survive a
  fresh session; regenerate again if gone (cheap, ~1-2 min each).

## Next
1. Read `<scratchpad>/isolated.diff` (6 lines) and confirm every line is
   one of: the CSS addition, the market_help sentence, or a "1 book"/"N
   books" span — nothing else. If anything else changed, investigate before
   proceeding.
2. Run `/f/Repos/nfl_py3/.tools/uv.exe run ruff format src/nfl_ats/board_content.py src/nfl_ats/board_terminal.py` and
   `ruff check` on those two files, plus `mypy` on the same (paths only,
   not full repo) — not yet run this session.
3. Run
   `/f/Repos/nfl_py3/.tools/uv.exe run pytest tests/test_board_content.py tests/test_board_terminal.py tests/test_public_board.py -q -p no:cacheprovider`
   once — not yet run this session.
4. Add one dated sentence to the `ROADMAP.md` UI-20 row (not done yet —
   blocked by tool-call cap, lane-file writes only).
5. Do not publish/commit/push (task scope forbids it) — hand back to the
   orchestrator for that.

## Open
- Whether the owner wants the mobile `data-label` attribute (currently
  `game.market_now_book_label or "Books now"`, i.e. the per-row count text
  replaces the column header on mobile) kept as-is — it predates this task
  and was left untouched; flag it only if review flags it as confusing.
- market_data.py, pick_probability_fit.py, scripts/, registry/ were not
  touched (out of scope per task).
