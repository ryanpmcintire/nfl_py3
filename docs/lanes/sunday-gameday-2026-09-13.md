# sunday-gameday-2026-09-13

## Goal

Week 1 Sunday morning: every served pick reflects Sunday's injuries, lineups
and books before the 13:00 ET kickoffs, the Week 1 paper ledger agrees with
the served card, and the Sunday scheduler chain is proven to run. Done when
the ledger check reports zero disagreements, the 09:30 refit, the 11:35
inactives capture and the 11:55 refresh have run, and the board is pushed.

## State

- Odds captured by hand at 07:23 ET (`scripts/odds_capture.ps1`,
  `data/market/capture_log.txt` last line, snapshot 20260913T112351Z).
- New scheduler job `lineups_sun_am` (sun 09:30, `LINEUP_CAPTURE`, dedupe
  60 min on `artifacts/margin_predictions`) in `scripts/capture_scheduler.py`,
  exercised by hand 07:27-07:57 ET (`MANUAL-RUN OK lineups_sun_am` in the
  scheduler log); daemon restarted 07:27:29 ET so the schedule on disk is
  live. The refit activated model 425790b83c903fdb, republished the card (no
  pick changed; only timestamps moved in `CURRENT_PREDICTIONS.md`) and
  published the board. Uncommitted.
- Board: confidence-word legend under the picks table
  (`board_content.py` `confidence_legend_text`, `_confidence_legend_text`;
  `board_terminal.py` `_confidence_legend_html`), rendered on
  `docs/index.html` line 1108 ("Lean from ..., Strong from 57.2%"), both
  numbers read from the served strength bands. Uncommitted.
- Official injuries 06:00 and player snapshot 06:15 captured today; the
  09:00 / 09:15 captures and the 09:30 refit are the daemon's.
- Week 1 paper ledger re-recorded by the owner at about 08:20 ET
  (`publish-predictions --record-decisions --replace-week --no-board`:
  14 pre-kickoff rows replaced, 2 played games left, .bak kept). Then
  `refresh-picks --publish-card --note sun_ledger_realign`,
  `card-ledger-check` (no disagreements, measured) and `publish-board`.
  Served picks unchanged; the late-week table no longer lists ATL at PIT and
  DEN at KC as changes because the Tuesday baseline now carries them.

## Tried

- The daemon's stop script kills every process whose command line matches
  `capture_scheduler`, including a hand `--run-job`; restart the daemon by
  pid tree while a manual run is in flight.
- LEAD-64 morning comparison is recorded in
  `docs/lanes/lead64-headline-parser.md` (Tried); next there is the
  designations dedupe.

## Next

- Commit and push (handoff refreshed first).
- Read the 09:30 `lineups_sun_am` row in `capture_scheduler.py --status`
  after 10:00 ET; it must say OK.

## Open

- None.
