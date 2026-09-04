# Historical officials archive probe (LEAD-59)

**Measured 2026-09-04, two single fetches total.** The question was
whether a 2009–2025 per-game crew archive is buildable for LEAD-31/32/33.

## 2015–2025: already covered, no build needed

The nflverse `officials` feed (`nflreadpy.load_officials()`) is already
snapshotted locally at `data/raw/officials/20260819T190537Z/` (see
`docs/referee_battery.md` for the measured coverage: 2015–2025, 2,896
REG-season referee rows, 99.86% schedule crosswalk). LEAD-59's premise
("no accepted historical source") is half-answered without new work:
eleven seasons suffice for every crew-trait reliability gate currently
queued.

## 2009–2014: blocked on both paths today

- **Direct PFR boxscore fetch: HTTP 403.** One polite fetch of
  `pro-football-reference.com/boxes/2015091000.htm` with the repo's
  registered contact UA was refused — same bot wall measured 2026-09-03
  on PFR team-season staff pages (PER-07 row). No spend, no retry loop.
- **Wayback replay of the same page: HTTP 429.** One
  `web.archive.org/web/2020id_/...` fetch was throttled — consistent
  with the 2026-09-03 IP-throttle finding (MKT-14 row) after prior
  sessions' heavy crawling.

## Unblock conditions (in order)

1. Ride MKT-14's planned polite Wayback pass: a backoff-scheduled
   sweep can collect 2009–2014 PFR boxscore captures with zero new
   source negotiation.
2. If PFR stays walled, assess ESPN game archives for official listings
   (presence unconfirmed — a one-page probe, not a build).
3. Fallback decision the owner can take anytime: declare 2015+ the
   population. Eleven seasons resolve crew-rate traits; only
   long-tenure referees' early years are lost.

## What was NOT done

No archive built, no bulk fetch, no ATS screen, no window spent.
