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

- Resolved 17:30 ET: the played side is CHI, the live pick on the board at
  the 13:00 lock (measured: commit 996763c `docs/index.html` ticker read
  CHI -2.5 at 50.0%). Recorded as `board_at_lock` in the pick-revision
  ledger, a `site_push` row at 16:50:36Z in `clv_ledger/published_picks`
  (the board freezes from that ledger; its 12:45:29Z CAR row came from a
  content build, not a site publish, which is a defect to fix), and the
  card's late-week table by hand. Board republished: CHI -2.5.
- `refresh-picks` fails after every deadline has passed when the paper
  ledger carries two composition policy ids (v2 rows for the Wednesday and
  Thursday games, v3 for the Sunday re-record): "Refresh requires one
  frozen production composition policy". The 15:00 pass ran; the 17:16
  publish did not. Must be fixed before the Tuesday lock.
