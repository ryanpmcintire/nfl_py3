# REF-SCORECARDS-FIX: re-task of the rejected blog-rwb07-scorecards code

## Status

**Complete.** Optional CLV columns threaded through `season_scorecard` using the
REAL `nfl_ats.clv` contract, with explicit missing-data markers and fixture
tests. All four gates pass.

## What went wrong last time (read, reports/wave1/blog-rwb07-scorecards.md)

The previous worker's commit (`14b759e`, branch `swarm/blog-rwb07-scorecards`,
verified via `git show 14b759e --stat` this session) invented the clv API:

- called `build_pairing_table(predictions)` — one positional DataFrame; the
  real signature (read, src/nfl_ats/clv.py:390) is
  `build_pairing_table(root: Path, *, capture_kind, labels, seasons, schedule)`
  and takes the market-capture ARCHIVE directory;
- called `close_reference_table(pairings=..., ...)` — wrong kwarg; the real
  signature is positional `(pairing, schedule)`;
- a "simplified fallback" that averaged `spread_line` and labelled it a "CLV
  proxy" — that is not CLV under any definition;
- blanket `except Exception: nan` handlers that turned every failure mode into
  an indistinguishable silent NaN.

That code was never merged; this worktree started from clean master
(`cdc7c8b`, measured via `git log -1`).

## What was built

### src/nfl_ats/reporting.py

- `season_scorecard(predictions, *, market_capture_root=None,
  clv_decision_label="tue_open")` — optional capture-root parameter threaded
  through the season-scorecard path.
- Three new columns on every returned scorecard row:
  - `clv_points` (float): mean signed CLV in points for that season's forced
    picks, or NaN **only when accompanied by a non-measured status below**;
  - `clv_status`: explicit provenance marker —
    `measured` | `capture_unavailable` (no root supplied / root not a
    directory) | `no_paired_games` (archive exists but pairs none of that
    season's games);
  - `clv_games` (int): how many of the season's games produced a CLV value,
    so partial archive coverage stays visible instead of collapsing into NaN.
- The measurement path calls exactly the documented pipeline:
  `build_pairing_table(root, capture_kind=HISTORICAL_CAPTURE_KIND,
  labels=(decision_label, *CLOSE_LABEL_PRIORITY), seasons=..., schedule=...)`
  → `close_reference_table(pairing, schedule)` → `score_clv(picks, pairing,
  close_reference)`, with picks carrying `game_id`, `side` (HOME/AWAY implied
  by `home_cover_probability >= 0.5`, i.e. the model's own forced pick), and
  `decision_label`. Sign conventions verified against the hand-computed test
  in tests/test_clv.py (`score_clv = side * (close - decision)`).
- The empty-frame return now also carries the three CLV columns so the output
  schema is stable for callers (`model_card.build_model_card` serializes the
  scorecard to JSON; its existing test still passes).
- `nfl_ats.clv` is imported lazily inside `_clv_per_season` because `clv` →
  `active_model` → `reporting` would be circular at module level (verified by
  grep this session). A missing `spread_line`/`game_id`/`season`/`week`
  column still raises `DataContractError` from the clv contracts — that is a
  caller contract error, deliberately NOT swallowed into a status marker.

### tests/test_reporting.py

Three new tests plus shared fixtures, all synthetic; no scoring looks were run
or needed (this is reporting plumbing over already-evaluated prediction frames):

1. `test_season_scorecard_marks_clv_unavailable_without_capture_root` — no
   root and a non-existent root both yield `capture_unavailable` everywhere,
   NaN points, zero games.
2. `test_season_scorecard_measures_clv_from_fixture_archive` — builds a real
   historical-backfill snapshot store (reusing `_store_snapshot`/`_event`/
   `_spread_book` from tests/test_clv.py) with tue_open + sun_late_close
   snapshots for two games across two seasons, and checks hand-computed CLV:
   KC-CIN HOME pick at +1.5 decision, +4.0 store close → +2.5; SEA-NE AWAY
   pick at +2.5 decision, +1.0 store close → +1.5; per-season split correct.
3. `test_season_scorecard_flags_seasons_the_archive_does_not_pair` — after
   deleting the 2024 snapshots from the copied store, 2023 stays `measured`
   (+1.5) while 2024 gets `no_paired_games`, NaN, 0 games.

Two fixture bugs were hit and fixed during development (both measured via
pytest failures): `attach_nflverse_game_ids` needs a `kickoff` column
(src/nfl_ats/market_data.py:204), and the historical parser normalizes team
names to abbreviations, so the fixture schedule uses "KC"/"SEA"; books must be
passed through `_event` so `"__HOME__"` placeholders resolve to real team
names (otherwise `_outcome_side` marks every quote OTHER).

## Binding-rule compliance

- Rule 3 (windows): no experiment window was scored; fixtures only. Nothing to
  record with `weak-signals record` — this change introduces no signal claim,
  only plumbing, and asserts hand-computed arithmetic.
- Rule 1: nothing here closes or rejects anything; no interval was read as
  evidence.
- Rule 2: all claims above are tagged measured (commands run this session),
  read (paths cited inline), or inferred (labelled).

## Quality gates (all four, this session)

```
ruff format --check .   -> 636 files already formatted
ruff check .            -> All checks passed!
mypy src                -> Success: no issues found in 105 source files
pytest                  -> 1858 passed, 5 skipped (documented data-absence skips)
```

One transient full-suite failure (`test_cli_model_workflow`, FileExistsError
inside pytest's basetemp factory) reproduced as a basetemp-collision artifact,
not a code failure: it passes in isolation and in a clean second full run
(1858 passed) with a fresh basetemp.

## Not done / out of scope

- No CLI flag wiring for `market_capture_root` (the task scoped the parameter
  to the season-scorecard path); callers pass it explicitly.
- `model_card.season_history` rows now include the three CLV fields; existing
  model-card tests pass unchanged.
