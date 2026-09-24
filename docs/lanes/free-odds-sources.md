# Free odds sources

## Goal

Current NFL spreads from free sites (Odds Gap line shop, Bovada) feed the model's
market-move term and the board's Books now column. Posting lines found on other
websites on the dashboard is allowed; the owner has said so repeatedly and it is
settled. Never reintroduce a private-only, no-publication, or Sunday-only rule for
odds sources.

## State

- 2026-09-24: the invented no-publication rule is removed everywhere.
  `config/source_policies.json` gives both sources `derived_publication:
  aggregates_only`; the `public_only` filter and `publication_scope` stamps are
  gone from `market_data.py` and both capture scripts.
- Books now shows one number when every captured book agrees, otherwise the
  lowest-to-highest range for the picked side (e.g. `CAR -3 to -2.5`), with the
  book count underneath. The move label uses the book average rounded to the half
  point. Lines captured in the last 48 hours count. The model's input stays the
  median (`pick_refresh.py`, `home_spread_line`).
- `odds_private_wed/fri/sat` are enabled again next to `odds_private_sun`. The
  daemon was restarted (pid 33320, code and schedule current).
  `--run-job odds_private_fri` returned MANUAL-RUN OK: Odds Gap had 15 games from
  3 books.
- The board is published with the ranges. 357 board, pick-refresh and scheduler
  tests pass.
- The Odds API is cancelled for good (`done/odds-api-key-deactivated.md`).

## Tried

- 2026-09-24: the direct Bovada capture failed with `Bovada response must be an
  array`. The `coupon/events` endpoint now returns `{}`. Fixed by switching to
  `services/sports/event/v2/events/...`, which returns the same payload format.
  A direct run captured 18 games and 72 quotes; `--run-job odds_private_sat`
  returned MANUAL-RUN OK (Bovada was correctly skipped by the 30-minute age
  guard).
- ESPN pickcenter returns 403; those jobs stay disabled.

## Next

- After Friday's 12:30 run, confirm Books now updates by itself.

## Open

- None.
