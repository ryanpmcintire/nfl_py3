# Free odds sources

## Goal

Replace the dead paid Odds API captures with free present-day line sources.
Done when one or more free sources are proven to carry pregame spreads
(and totals) on the Tuesday-to-Sunday cadence the board needs, and the
replacement capture job is specified (not yet built).

## State

- 2026-09-19: paid key dead (DEACTIVATED_KEY), owner will not re-subscribe;
  all 18 paid jobs disabled (lane `odds-api-key-deactivated.md`). Owner
  confirms no specific free source was ever agreed. Probe launched same day.

## Tried

- In-tree candidates noted (unverified): hand-captured Splash board (pool
  lines only, no cross-book market), `public_betting_live_capture.py`
  (Action Network splits, percentages not lines), nflverse schedules
  (closing lines, post-season, not live).
- Feasibility probe DONE 2026-09-19 (subagent, 7 requests; coordinator
  re-verified the two load-bearing claims with 2 more requests plus the
  vendor pricing page):
  - ESPN: scoreboard carries no odds (measured, 14 events); per-game
    summary `pickcenter` carries DraftKings spread+total (measured:
    CAR -2.5, o/u 43.5). Single book, per-event (16 calls/slot), free,
    no key. Tuesday availability for Sunday games unverified.
    Licensing red-leaning: ESPN pages already RED in
    `config/source_policies.json`; a DK line must never publish as
    consensus.
  - Same-vendor free tier: Starter FREE, 500 credits/month, all
    sports/markets, most bookmakers (measured from the pricing page).
    Bulk-only board fits (~10 jobs x 3 credits = ~130/month); halves
    (64/run) and props do not. Licensing GREEN (registered source).
  - nflverse schedules: lines filled weeks 1-3 only (measured, 272 rows
    via nflreadpy), single number, no books or timestamps. Close
    reference only, not a live feed.
  - Action Network: percentages only, not a line source (read).
  - Nothing else free and live found (Pinnacle/Betfair need funded
    accounts; archives are closes-only).
- Standing fact under every option: all six halves jobs die (LEAD-61
  loses its prospective channel); no free per-event halves source exists.

## Next

- Feasibility probe (read-only, no registry cells, no looks spent): for each
  candidate source verify spread+total pregame, book coverage, Tuesday
  availability, cadence, and access/licensing terms; rank and specify the
  replacement capture. Then build it as its own unit.

## Open

- Owner decision 2026-09-19: use AS MANY sources as possible (federation,
  not one feed). Coverage matrix from verified facts:
  - Splash pool board (hand, already scheduled Tue 12:05): the grade
    line, authoritative for the card. Unchanged.
  - ESPN DraftKings single-book (free, per-event): spread+total proxy
    for market moves and Books-now, always labeled single-book, never
    consensus. Needs: Tuesday-availability probe, licensing registration
    (source currently RED).
  - nflverse schedules (free, weekly): close reference for CLV once
    updated. Not a live feed.
  - Public betting splits (already scheduled): field behavior only.
- Build order proposed: (1) ESPN Tuesday probe + licensing call;
  (2) ESPN capture job on the old bulk slots; (3) consumer audit
  (Books-now/market-move/dispersion pools read single-book honestly or
  degrade); halves stay dead under every option.
- Same-vendor free tier (500/mo, bulk-only fits) stays a fallback; owner
  leans away from the vendor entirely.
