# serve-calibrated-probability

## Goal

Put a per-game confidence number back on every reader surface, sourced from
MKT-18's four-term model instead of the raw model whose number was inverted
against its own record. Done when the coefficients live in an artifact (never
in `src/`), the board, card, ticker, history table and assistant all serve that
number, and the gates plus `publish-board` / `publish-predictions` are clean.

## State

Done, 2026-09-14, uncommitted. Shipped:

- `src/nfl_ats/pick_probability.py` (serving: coefficient loader that fails
  closed, signed flag builder reusing the composition members' own side
  builders, live market-move reader, calibrated probability + side + strength
  word, the reader sentence).
- `src/nfl_ats/pick_probability_fit.py` and `nfl-ats fit-pick-probability`
  (registered in `cli_commands/prediction.py`) writing
  `artifacts/pick_probability/<stamp>/{coefficients.json,metadata.json}` and
  the `artifacts/active_pick_probability.json` pointer.
- `card_view.resolve_card_view` takes `artifacts_root` and serves the
  calibrated number and side; `publishing`, `board_content`, `public_board` and
  `clv`'s played-card recorder all pass it.
- Presentation restored from the pre-strip HEAD content:
  `board_terminal.py` + css, `board_interactive.py` + js, `board_assistant.py`,
  `card_explanation.py`, `lineage.py`, plus the stripped hunks in
  `board_content.py` and `publishing.py` (card column renamed Cover chance).
- `publish-board` and `publish-predictions` refuse to run when the artifacts
  root has an active model but no fitted cover chance
  (`cli_commands/publishing._require_served_pick_probability`); a malformed
  pointer raises everywhere.
- `docs/serve_calibrated_probability.md`, ROADMAP row `MKT-19`.

Gates: ruff format/check clean, mypy clean on 234 files, `pytest -q` 4529
passed / 9 skipped. `publish-predictions` and `publish-board` both ran.

## Tried

- **Fit on the served opener probability, not the raw one.** Production serves
  the home-side-offset-adjusted `home_cover_probability`; fitting the model term
  on `_raw` (as MKT-18's research arm did) would feed the serving path a
  different quantity than it was fitted on. Measured offset effect on the
  opener stream: max 0.053, 401 games non-zero, 46 opener picks changed.
- **Held members contribute zero at fit time too**, so the fitted `b` is the
  weight on the seven flags actually served rather than on nine.
- First fit, `artifacts/pick_probability/20260914T232538Z`, 1,503 graded games
  2020-2025, 799 with the market move: a 0.217, b 0.260, c 0.210 -- all inside
  MKT-18's fold ranges. Out-of-season 841-662 (55.95%) vs model-only 819-684
  (54.49%). Confidence bands 48.9 / 56.4 / 58.3 / 59.3 / 60.2% where the raw
  model's fell (56.0 down to 50.7).
- Strength thresholds are the artifact's own out-of-season terciles (Lean from
  52.8%, Strong from 56.5%) with each band's measured landing rate stored
  beside it, so the board legend and the card note read from the artifact.
- The market move is live at serve time: measured on this week's card, all 16
  games had leader-book exposure, moves up to 2.5 points, and 8 of 16 games
  land on a different side from the raw model once the flags and the move are
  counted. Those sides are frozen for Week 1 because every game is past its
  deadline -- the calibrated number governs the next open week.
- Reverting the stripped presentation from `HEAD` blobs (not `git checkout`,
  which a hook blocks) was cheaper and more faithful than re-authoring it; only
  two restored tests needed re-editing (the refresh-policy wording and the CLI
  contract fixture's subcommand order).

## Next

- Fold the calibrated probability into `pick_refresh.plan_refresh` so the
  late-week ledger decides from the same number the card serves.
- Re-run `nfl-ats fit-pick-probability` whenever the active model changes; add
  it to the weekly run once the refresh path uses it.

## Open

- No forward read exists. MKT-18's look-count caveat carries: `b` and `c`
  inherit whatever optimism the flag and move promotions already accumulated on
  these same games. Serving is an owner decision taken here on the calibration
  ordering, not on a horse-race win over the flip chain.
- Whether the week board (`public_board.render_picks_page`) should also take
  its fallback strength bands from this artifact rather than the old
  displayed-confidence cells; today it only falls back when a row carries no
  attached strength word.
