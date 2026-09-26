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
  point. The range renders as two bold numbers joined by a short bar on one line
  (`mn-range` in `board_terminal_style.css`), and the column widths were rebalanced
  so it does not wrap. The board counts books whose latest line falls within 6
  hours of the newest (`coverage_window`); the model keeps the 30-minute window.
  Lines captured in the last 48 hours count. The model's input stays the
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

- 2026-09-26 measured: Friday 2026-09-25 12:30 ET `odds_private_fri` ran but
  did NOT capture — `OK odds_private_fri: {"captured": false, "reason":
  "no_upcoming_nfl_games"}` (`data/scheduler_log.txt:1842-1843`). Read
  `scripts/capture_private_sunday_odds.py:34-39`: it requires a kickoff within
  `(now, now + 2 days)`. Friday 12:30 ET + 2 days = Sunday 12:30 ET, which is
  ~30 min before the earliest 1:00 PM ET Sunday kickoff, so the window
  structurally excludes every Sunday game whenever this job runs at its
  scheduled Friday-noon time. This is a distinct bug from the 9/24 Bovada
  endpoint fix — not yet diagnosed further, not fixed here (read-only check).
  Saturday's 10:00 ET `odds_private_sat` run then succeeded:
  `the_odds_gap_lineshop_private` captured=true, 15 games, 90 quotes, written
  to `data/market/raw/20260926T140051Z-odds-gap-private/`
  (`data/scheduler_log.txt:1886-1887`). `docs/index.html` mtime is
  2026-09-26T13:56:20-04:00, matching the `lineups_sat_pm` republish that ran
  right after that capture, so Books now does reflect data captured after
  Friday 12:30 ET — but from Saturday's run, not Friday's. The original Next
  ("confirm Friday's run updates Books now by itself") is not confirmed as
  stated; Books now self-updates only because Saturday's job papered over
  Friday's structural miss. Next: widen or fix the 2-day upcoming-game window
  in `capture_private_sunday_odds.py` so the Friday job itself captures.
- 2026-09-26 measured: the same Saturday run's `bovada_public_nfl` job (a
  separate direct-scrape job in `scripts/capture_bovada_private.py`, distinct
  from the Bovada quotes already included via Odds Gap's 3-book aggregate)
  failed again with `"reason": "Bovada response must be an array"` even
  though that script already uses the fixed endpoint
  (`capture_bovada_private.py:25`, `services/sports/event/v2/events/...`,
  raised at line 40). The 9/24 endpoint fix has not made this job reliable;
  looks intermittent (anti-bot/rate limit) rather than the original wrong-URL
  bug. Not fixed here.

## Open

- Two open items, both read-only findings from 2026-09-26, not fixed:
  1. FIXED 2026-09-26: `capture_private_sunday_odds.py:capture` window widened
     to 3 days; at Friday 16:30 UTC it now sees 14 games (was 0, measured
     against `game_features.parquet`). Confirm next Friday's 12:30 log line
     reads captured true.
  2. FIXED 2026-09-26: `bovada_public_nfl` failures (2 of 4 logged runs:
     `data/scheduler_log.txt:1764`, `:1887`) were a real intermittent
     PerimeterX bot-check on the endpoint (`X-Px` header present on every
     response, live-checked twice, both 200 with a proper JSON array —
     could not reproduce the block live within the 2-request budget).
     `capture_bovada_private.py:_rows` (now ~38-53) previously raised a
     generic "must be an array" with no payload evidence, so the block
     shape was never captured for diagnosis. Now: JSON-decode errors and
     non-list/non-`events`-dict payloads raise with the decoded type plus
     a 200-char snippet of the actual body; a single coupon dict (has an
     `events` list) is normalized to `[dict]` instead of rejected. Verified
     with a real run: `python scripts/capture_bovada_private.py` ->
     `{"captured": true, "games": 17, "quotes": 68, ...}`. Next occurrence
     of the failure will log the real payload shape instead of a blind
     message.
