# sim04 dashboard chart

## Goal
Add a "how this game could finish" simulated final-margin histogram chart to
each game's deep-dive panel on the board (TERMINAL skin), sourced from
scripts/sim04_engine.py simulations, recentred on the served predicted home
margin the card already uses. Do not edit scripts/sim04_engine.py or
scripts/sim04_loso.py (grading job depends on them). No code comments/
docstrings. No new tests. No commit/push/publish.

## State

- CLOSED 2026-09-26. Final results are in `docs/sim04_unit_log.md` ("Unit LOSO grade result", "SIM-05 and dashboard") and ROADMAP SIM-04/SIM-05. Anything below is history.
Step 1 output was WRONG and has been fixed; steps 2-4 not started (edit to
board_content.py did not land -- blocked by the 50-tool-call subagent cap
mid-edit; see Next #1). No board_content.py or board_terminal.py changes are
on disk yet.

**Critical finding this session:** `dive.adjuster` (from
`_build_adjuster`/`SpreadAdjusterParams.center`) is `None` for every game this
week, by design, not a bug. `board_content.py:3634-3640` sets
`spread_explorer_params = {} if calibrated_mass else _load_spread_explorer_params(...)`.
The currently active served model has `calibrated_mass=True` (discrete
mass-preserving-lattice policy, per AGENTS.md "Margins are multimodal") so
spread_explorer_params is always `{}` and `_build_adjuster` always returns
`None`. Confirmed by direct check: `load_board_content(...).dives[0].adjuster
is None` for all 16 games this week; `board.headline.model_id ==
'8587951e3acc6055'`. The original `scripts/sim04_week.py` gated on
`dive.adjuster is None: continue`, so it silently produced an EMPTY
`margins.json` (`"games": {}`) -- verified by loading
`artifacts/sim04_week/20260926T040054Z/margins.json` and finding `len(games)
== 0`. That file is stale/wrong; do not use it.

**Fix applied and verified working:** `artifacts.predictions` (from
`load_public_board_artifacts(artifacts_root)`, already imported by
board_content.py) carries a `predicted_margin` column directly, for every
game, independent of `calibrated_mass`/gaussian gating. Confirmed sign
convention matches home-margin (positive = home favored/winning), matching
`dive.adjuster.center`'s convention, by cross-checking 4 games' `home_team`,
`market_spread`, `predicted_margin`, `home_cover_probability` against
`board.games[i].pick_team`. Also present on the same predictions frame (not
yet used, may be useful later): `margin_lower_50/80`, `margin_upper_50/80`
(discrete-lattice quantile bounds).

Edited `F:/Repos/nfl_py3/scripts/sim04_week.py` (already-owned new file, not
engine/loso):
- Import `load_public_board_artifacts` alongside `load_board_content`.
- `build_week_margins` now builds `predicted_margin_by_id =
  load_public_board_artifacts(artifacts_root).predictions.set_index
  ("game_id")["predicted_margin"].to_dict()` and uses `center =
  predicted_margin_by_id.get(game.game_id)` (skip if `None`/`NaN`) instead of
  `dive.adjuster.center` / skipping when `dive is None or dive.adjuster is
  None`. `dive_by_id`/the `dive` variable were removed from this function
  entirely -- no more dependency on `board.dives` or the adjuster gate.
  Nothing else in the file changed (ratings lookup, `simulate(...)`,
  recentring, clipping, histogram building, JSON shape are all unchanged from
  the version described in the previous lane entry).
- Re-ran in background (`F:/Repos/nfl_py3/.venv/Scripts/python
  scripts/sim04_week.py`, bash id `b9ih8gk5y`, output file
  `C:\Users\Ryan\AppData\Local\Temp\claude\F--Repos-nfl-py3\7897f86b-3563-4cf5-9d78-28ad2b5e8fef\tasks\b9ih8gk5y.output`)
  -- NOT CONFIRMED DONE. Only `artifacts/sim04_week/20260926T040054Z/` existed
  on disk as of the last check (the stale/empty one); the new run's output
  directory had not yet appeared. `build_tables(range(2009,2026),
  condition_on_team=True)` plus 16 games x 2000-play simulations is the likely
  reason it runs past 120s.

Attempted `src/nfl_ats/board_content.py` edits this session:
- Added `import json` right after `from __future__ import annotations` (line
  3, now landed on disk -- this one DID apply before the cap fired).
- Attempted to insert a `SimMarginChart` frozen dataclass (fields:
  `histogram: tuple[tuple[int, int], ...]`, `break_even: float`, `n: int`)
  immediately before the `GameDive` dataclass (was ~line 441, will have
  shifted by +1 for the `import json` line and further once this lands) --
  this Edit call was BLOCKED by the tool-call cap (PreToolUse hook fired
  before execution) and did NOT land. Re-check the file before assuming it is
  there.

## Tried
(Carried over from the previous unit, still valid -- sign convention and
loader pattern references below are unaffected by the center-source fix
above.)
- `load_board_content(artifacts_root, data_root=None, generated_at=None,
  require_fresh_arrest_overlay=False) -> BoardContent` at
  `src/nfl_ats/board_content.py` (search, line numbers have shifted by the
  `import json` line); `.games: tuple[GameRow,...]`, `.dives:
  tuple[GameDive,...]`, `.headline.model_id`.
- Sign convention (unchanged, still the one to use):
  `pick_line_value = market_spread * sign` with `sign = -1.0 if pick_team ==
  home else 1.0` (matches `pick_spread_text`, board_content.py ~line
  320-326); `pick_margin = home_margin if pick_team == home else
  -home_margin`. Pick covers iff `pick_margin > -pick_line_value`, pushes at
  equality (`-pick_line_value` is the break-even margin -- store this
  precomputed value as `SimMarginChart.break_even` so board_terminal.py needs
  zero sign logic, just render bars and shade left/right of `break_even`).
- `simulate(...)` / `build_tables(...)` contracts, ratings dict shape, CLI
  root helpers (`_artifacts_root`/`_data_root` in `src/nfl_ats/cli_common.py`)
  -- all unchanged, see previous entries in git history of this file if
  needed; scripts/sim04_week.py already encodes all of this correctly.
- Loader pattern to copy for "latest artifact file, fail-soft" is
  `load_lineups` in `src/nfl_ats/lineup_view.py:105-124` (glob timestamped
  subdirs `artifacts_root/"sim04_week"/*/margins.json`, sort reverse,
  try/except OSError/JSONDecodeError, skip bad files, return `{}` if none
  parse).
- `_build_dive` call site: `lineups = load_lineups(artifacts_root)` then
  `_build_dive(game, sweep=..., waterfall_feed=..., waterfall_document=...,
  active=..., spread_explorer_params=..., raw_home_cover_probability=...,
  lineups=lineups)` inside a generator over `games` (was ~board_content.py
  line 3838-3855, will have shifted).
- TERMINAL CSS tokens confirmed in `src/nfl_ats/board_terminal_style.css`:
  `--green` (cover), `--red` (miss), `--amber` (push/marker, used by
  `.marker`), `--cyan` (used by `.curve-path`/`.adjuster-marker`),
  `--text-dim`/`--text-faint` (labels), `--line`/`--line-soft` (gridlines),
  `--font-mono`. Reuse via inline `style="fill:var(--green)"` etc. or by
  adding new classes to that CSS file -- no need to invent new colors either
  way.
- `_game_dive_chart_html(dive)` (board_terminal.py, was line 1041, will have
  shifted) is the pattern to copy: inline `<svg viewBox="0 0 280 100"
  width="100%" height="140" role="img" aria-label="...">`, `escape()` helper
  already imported in this file, `.grid`/`.ref`/`.marker` CSS classes give
  responsive phone-width behavior for free via `width="100%"` + viewBox --
  copy this, don't reinvent scaling.
- Call site to wire in (board_terminal.py, was ~line 1290-1313, will have
  shifted): inside the dive-panel builder, `_game_dive_chart_html(dive)` is
  called at line ~1311 inside `'<div class="dive-body"><div>'` (2-column:
  attribution left, cover-curve chart right) which closes with `"</div></div>"`
  at line ~1312, then `f"{_lineups_html(dive)}</div></div>"` closes out the
  panel. Plan: insert `_sim_margin_html(dive)` call AFTER the `dive-body`
  closes (i.e. after line ~1312, before `_lineups_html`), as its own
  full-width block -- a bar chart needs more horizontal room than the
  2-column layout gives, and this keeps phone width simple.

## Next
1. DONE (verified this session, background run `b9ih8gk5y` completed exit 0):
   `artifacts/sim04_week/20260926T041245Z/margins.json` is the good artifact
   -- use this one, not the stale empty `20260926T040054Z/` one. Confirmed:
   `len(d['games'])==16`, every `home_margin_histogram` sums to `d['lineage']
   ['n']==2000` (zero bad sums), and game `2026_03_ATL_GB` (home GB,
   market_spread 6.5, center/`recentred_home_margin` 5.4765) has a histogram
   spanning margins -49..53 with mean 5.332 -- matches the center almost
   exactly and is right-shifted/positive as expected for a home favorite.
   Step 2 (quick-check) is therefore also DONE -- skip both, start at step 3
   below.
3. Re-check whether `import json` and nothing else landed in
   `src/nfl_ats/board_content.py` (`grep -n "^import json" -A2 -B2` near the
   top, and confirm `SimMarginChart`/`GameDive` are NOT yet touched -- the
   dataclass insert was blocked by the cap, don't assume it's there). Then
   apply, in order:
   - Insert `SimMarginChart` frozen dataclass (fields above) immediately
     before `GameDive` (currently right after `SpreadAdjusterParams`, search
     for `class SpreadAdjusterParams` then `class GameDive`).
   - Add `sim_margin: SimMarginChart | None = None` to `GameDive` (after
     `adjuster: SpreadAdjusterParams | None`).
   - Add `_load_sim_margins(artifacts_root: Path) -> dict[str, dict]`: glob
     `artifacts_root / "sim04_week"` / `"*/margins.json"`, sort reverse
     (lexicographic UTC timestamp sort works), try/except
     `(OSError, json.JSONDecodeError)`, parse first that succeeds, return
     `parsed.get("games", {})` if it's a dict else `{}` (fail-soft, mirrors
     `load_lineups`).
   - In `load_board_content`, call `sim_margins = _load_sim_margins
     (artifacts_root)` near `lineups = load_lineups(artifacts_root)`, thread
     as a new kwarg into every `_build_dive(...)` call in the generator over
     `games`.
   - In `_build_dive` signature, add `sim_margins: Mapping[str, dict]`
     parameter. Body: look up `raw = sim_margins.get(game.game_id)`; if
     `raw` is `None` or missing/malformed keys, set `sim_margin=None`. Else
     convert: `sign = -1.0 if game.pick_team == game.home else 1.0` (or
     reuse `pick_is_home` local), `pick_line_value = game.market_spread *
     sign`, `break_even = -pick_line_value`; build `histogram` by mapping
     each `(home_margin_str, count)` in `raw["home_margin_histogram"].items()`
     to `pick_margin = int(home_margin_str) if game.pick_team == game.home
     else -int(home_margin_str)`, aggregate counts per `pick_margin` (two
     different home margins can map to the same pick_margin only if pick is
     away and margins are negatives of each other -- unlikely but sum
     defensively), sort by margin, wrap as `tuple(sorted(...))`.  Set
     `sim_margin=SimMarginChart(histogram=..., break_even=break_even,
     n=raw.get("n") or sum(counts))` on the returned `GameDive`. Wrap the
     whole conversion in try/except (KeyError/ValueError/TypeError) ->
     `None` for fail-soft.
4. Edit `src/nfl_ats/board_terminal.py`:
   - Add `_sim_margin_html(dive: GameDive) -> str` beside
     `_game_dive_chart_html` (re-find by searching
     `def _game_dive_chart_html`, was line 1041). Fail-soft: if
     `dive.sim_margin is None`, return `""` (render nothing, no broken
     layout -- do NOT render a "not published" placeholder here, this is a
     bonus chart not a core one).
   - Inline SVG bar chart: x-axis = picked side's margin, roughly -21..+21
     (clip/bucket bars outside that range into the edge bins so the chart
     stays readable), y = bar height by count. Mark `dive.sim_margin
     .break_even` with a vertical line/marker (reuse `--amber`, matches
     `.marker` styling elsewhere). Label key numbers 3, 7, 10, 14 (and their
     negatives if they fall in range) as small ticks using `--text-faint`.
     Tint bars: `pick_margin > break_even` -> `--green` (cover), `<
     break_even` -> `--red` (miss), the single bar at `== break_even` (only
     possible when `break_even` is a whole number) -> `--amber` (push).
     Caption text (plain English, exactly this or equivalent): "How this
     game could finish: 2,000 simulated games played out snap by snap." Use
     `dive.sim_margin.n` instead of a hardcoded 2,000 if it's ever not 2000.
     No new percentages -- bars/counts only, no computed cover-probability
     text here (that's what the cover-curve chart already shows). Must work
     at phone width -- copy `viewBox="0 0 280 ..." width="100%"` pattern from
     `_game_dive_chart_html`, don't hardcode pixel widths.
   - Wire the call in: re-find the `_game_dive_chart_html(dive)` call site
     (search `_game_dive_chart_html(`, was ~line 1311) and the `"</div></div>"`
     that closes `dive-body` right after it (was ~line 1312). Insert
     `f"{_sim_margin_html(dive)}"` between that `"</div></div>"` and
     `f"{_lineups_html(dive)}</div></div>"` (was line 1313) -- i.e. full
     width, below the two-column attribution/cover-curve row, above lineups.
5. Render once for real to verify:
   - `publish-board` CLI handler: search `src/nfl_ats/cli_commands
     /publishing.py` for the `"publish-board"` argparse subcommand
     (reported near line 1766 by the previous unit, not yet located this
     session) and a `build_site(...)` call (reported near line 336). Prefer
     calling `load_board_content(...)` + whatever render function turns
     `BoardContent` into the TERMINAL HTML string directly in a `python -c`
     snippet over invoking the full CLI, to guarantee no push/publish
     side-effect. If a dry-run/no-push flag exists on the CLI, that's fine
     too. Do NOT call anything that writes to `docs/` and DO NOT commit/push
     if it does -- if the only render path writes to `docs/`, run it, inspect
     with `git diff -- docs/` scoped to one game, then leave the diff
     unstaged (do not commit).
   - Confirm the chart appears (non-empty `_sim_margin_html` output) for at
     least one game this week, and read the rendered HTML for that one game's
     dive panel to sanity-check the SVG (bars present, break_even marker
     present, caption text present, no stray percentages).
6. Run only board-rendering tests once: find test files matching
   `board` under `tests/` (e.g. `grep -rl board tests/ --include=*.py` or
   similar, exclude `tests/scratch/`), run that file (or files) once via
   pytest. If a legitimate change breaks one, edit the test minimally (no
   new test files/functions, no expanding coverage). Do not run the full
   suite.

## Open
- RESOLVED: background rerun `b9ih8gk5y` finished (exit 0), output is
  `artifacts/sim04_week/20260926T041245Z/margins.json`, verified good (see
  Next #1). Use this path when wiring `_load_sim_margins`' glob test / manual
  checks; don't hardcode it in code (the loader globs for latest).
- Exact current line numbers in `board_content.py` and `board_terminal.py`
  for everything referenced above have shifted by at least the `import json`
  line already landed, and will shift further once the dataclass/loader
  edits go in -- always re-search by symbol name, never trust the line
  numbers in this file as exact.
- Whether two different raw home-margin bins could collide onto the same
  pick_margin bucket when aggregating in step 3 was reasoned about
  defensively (sum the counts) but not exercised against real data yet --
  spot check after step 3 lands.
- `publish-board` dry-run/no-push mechanism still unconfirmed (same open item
  as the previous unit -- nobody has located it yet this session).
- Delete or ignore the stale `artifacts/sim04_week/20260926T040054Z/
  margins.json` (empty games -- confirmed wrong, produced by the
  pre-fix script). `artifacts/` is gitignored so this is not a repo-hygiene
  problem, just don't read from it.
