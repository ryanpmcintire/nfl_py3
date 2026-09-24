# Card freshness staleness + follow-rule log wording

## Goal
Bug 1: card's "Source freshness" line disagreed with the live
`source_freshness_policy.report_for_publication` evaluation (showed inactives
COMPLETE when live eval says not_due). Bug 2: `describe_pick_change` in
`scripts/capture_scheduler.py` wording claimed a "follow rule" when hard
follow rules are off (AGENTS.md: one fitted probability picks every side).

## State
Both code fixes made and functionally verified. Tool-call cap hit before
running ruff/mypy/pytest, so those checks are still Next.

### Bug 1 root cause (measured)
The top `**Source freshness: ...**` line in `CURRENT_PREDICTIONS.md` is
written ONLY by `publish_active_predictions` (`src/nfl_ats/publishing.py:453,
530`, calling `source_report.summary_line()`) during a full card publish.
`nfl-ats refresh-picks --publish-card` never calls that function — it only
calls `append_refresh_to_card` (`src/nfl_ats/pick_refresh.py`), which
previously just appended a late-week refresh section between
`LATE_WEEK_REFRESH_START/END` markers and never touched the top freshness
line. So the 15:00 ET `refresh_thu` job republished the card with a line
that was actually a stale snapshot from whenever the card was last fully
published (days earlier), disagreeing with the live 15:08 ET evaluation.
This matches the "stale cached report" hypothesis, not a `now`/week mismatch
or the zero-row branch (that branch, at `source_freshness_policy.py:520-542`,
is correct and already fixed per `docs/lanes/inactives-capture-empty.md`).

### Bug 1 fix
- `src/nfl_ats/pick_refresh.py`: added `_SOURCE_FRESHNESS_LINE_PATTERN` and
  `_refreshed_source_freshness_line(text, result, data_root, artifacts_root)`
  (local `import` of `report_for_publication` inside the function body, not
  top-level, to avoid a circular import: `source_freshness_policy` ->
  `public_board` -> `card_view` -> `published_picks` -> `pick_refresh`).
  `append_refresh_to_card` now takes optional `data_root`/`artifacts_root`
  kwargs and calls this helper on `text` before appending the refresh
  section, replacing the stale top line with
  `report_for_publication(now=result.computed_at_utc, ...).summary_line()`.
  Falls back to leaving `text` unchanged if both roots are `None` or the
  live call raises.
- `src/nfl_ats/cli_commands/publishing.py:1539-1546`: the `--publish-card`
  call site now passes `data_root=_data_root(), artifacts_root=_artifacts_root()`.
- Known minor scope narrowing: the refreshed line omits `arrest_snapshot_at`/
  `arrest_snapshot_id` (not threaded through), so `player_arrests` drops out
  of the recomputed line's buckets entirely rather than showing "complete".
  Acceptable: it is never miscategorized, just omitted from this
  narrower recompute; the full Tuesday publish still reports it correctly.

**Verified** (`measured`, scratch script, not committed):
called `_refreshed_source_freshness_line` directly on the real
`CURRENT_PREDICTIONS.md` text with `data_root=Path("data")`,
`artifacts_root=Path("artifacts")`, `now=datetime.now(UTC)` (2026-09-24,
~15:1x ET):
  - OLD line: `**Source freshness: COMPLETE.** Complete: ... inactives ...`
  - NEW line: `**Source freshness: COMPLETE.** Complete: odds opener, odds
    refresh, injuries nflverse, injuries nflverse timestamps, projected
    lineups, referee assignments, pfr transactions, airnow weather.
    Degraded (allowed fallback): none. Blocked: none. Not due yet:
    inactives. Not set up: injuries sportradar.`
  Confirms inactives now correctly reports `not_due` instead of `complete`.

### Bug 2 fix
`scripts/capture_scheduler.py` `describe_pick_change` (~line 2086-2090):
replaced `f"; line moved {float(delta):.1f} toward {now.split()[0]}, follow
rule"` with a two-line f-string ending `"one fitted input to the
probability"` (no more "follow rule" claim).

**Verified** (`measured`): ran a stub payload
(`TEN at NYG, previous HOME, new AWAY, movement_policy set, movement_delta
-1.0, model_only_pick_side HOME`) through `describe_pick_change` via
`PYTHONPATH=. .tools/uv.exe run --no-sync python <scratch script>`. Output:
`TEN at NYG: was NYG -3.5, now TEN +3.5 (50%; line moved -1.0 toward TEN,
one fitted input to the probability)`.

## Tried
- Traced the render call graph: `board_content.py:_load_source_policy_view`
  (board/HTML path) already recomputes live via `_live_source_policy_view`
  when no persisted `source_policy.json`/metadata block exists — that path
  was fine. The bug was specific to the markdown card's top summary line,
  which is plain text baked in at last full-publish time, not read from
  `SourcePolicyView`/`source_policy.json` at all.
- First attempt put `from nfl_ats.source_freshness_policy import
  report_for_publication` at module level in `pick_refresh.py` -> circular
  import (`pick_refresh` <- `published_picks` <- `card_view` <-
  `public_board` <- `source_freshness_policy`). Fixed by moving the import
  inside `_refreshed_source_freshness_line`.
- Import-order fix: `from nfl_ats.source_freshness_policy import
  report_for_publication` also had to be removed from the sorted top-level
  block (isort/ruff ordering) since it moved to a local import.

## Next
1. Run and report real output for:
   - `.tools\uv.exe run --no-sync ruff format --check src/nfl_ats/pick_refresh.py src/nfl_ats/cli_commands/publishing.py scripts/capture_scheduler.py`
   - `.tools\uv.exe run --no-sync ruff check src/nfl_ats/pick_refresh.py src/nfl_ats/cli_commands/publishing.py scripts/capture_scheduler.py`
   - `.tools\uv.exe run --no-sync mypy src/nfl_ats/pick_refresh.py src/nfl_ats/cli_commands/publishing.py scripts/capture_scheduler.py`
   - `.tools\uv.exe run --no-sync pytest -k "freshness or scheduler"`
   Fix anything these surface (e.g. ruff format may want to reflow the new
   multi-line f-string or the new helper function; mypy may flag
   `now=result.computed_at_utc` since `RefreshResult.computed_at_utc` is
   `pd.Timestamp` typed against `report_for_publication(now: datetime, ...)`
   — pandas `Timestamp` subclasses `datetime.datetime` at runtime so this
   should type-check, but confirm).
2. Check `tests/test_pick_refresh.py` (`test_append_refresh_to_card_fails_closed_without_a_published_card`)
   and `tests/test_cli.py:840-889` (mocks `append_refresh_to_card` with
   `lambda *_args, **_kwargs`) still pass unmodified — they should, since
   the new kwargs are optional and the existing mock accepts `**_kwargs`.
3. No commit/push yet (orchestrator owns that per AGENTS.md scope rules).

## Open
- None blocking. Both bugs are root-caused, fixed, and each independently
  exercised with real output above; only the standard verification commands
  (ruff/mypy/pytest) remain to run in a fresh subagent turn.
