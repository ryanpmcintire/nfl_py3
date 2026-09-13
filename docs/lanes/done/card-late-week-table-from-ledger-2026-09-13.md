# card-late-week-table-from-ledger-2026-09-13

## Goal

The card's late-week table, the board and the phone alert always name the
same side for every game. Done when a recorded refresh that flips a pick
republishes the card and the board itself, and the card's table shows the
last recorded side for games already kicked off.

## State

- Failure (measured, 2026-09-13): the 12:45 last-call pass recorded CHI at
  CAR back to CAR and sent the alert, but had no card publish; the card's
  late-week table (12:25) and the board (12:28) still said CHI at the 13:00
  lock. Each surface was rendered from its own run, not from the ledger.
- Fix (`src/nfl_ats/pick_refresh.py`, `src/nfl_ats/cli_commands/publishing.py`):
  the late-week table is built per game from the live run when the game is
  still eligible and from the latest pick-revision row when it is past
  kickoff (`latest_revisions_by_game`, `_served_side_rows`); a row appears
  only when that side differs from the Tuesday side. `refresh-picks` now
  writes the card section and publishes the board whenever `--publish-card`
  is passed OR `--record-decisions` changed a pick (`card.trigger` says
  which). Republished 13:21 ET: table lists ARI at LAC -> LAC and WAS at
  PHI -> PHI only; board ticker reads LAC -9.5, CAR +2.5, PHI -5.5;
  card-ledger-check ok, no disagreements (measured).
- Tests: refresh/card/publish files 303 passed (measured).

## Next

- Commit, push.

## Open

- The owner's locked picks may have followed the stale board for CHI at
  CAR; the paper ledger records CAR as served. Grading follows the ledger.
