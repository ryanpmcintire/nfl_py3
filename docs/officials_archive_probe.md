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

## 2026-09-05 (lane AN): the polite Wayback sweep built, throttle still holds

Built the "MKT-14 polite Wayback pass" this row names as unblock path 1:
`scripts/officials_wayback_sweep.py` (two-step per game — CDX API lookup on
`web.archive.org/cdx/search/cdx` for a capture timestamp, then a replay
fetch at `web.archive.org/web/<ts>id_/<original>` — with a >=8s floor
between every request, exponential backoff starting at 60s and doubling on
any 429/5xx, and a hard stop after 5 consecutive game-level failures),
resumable by construction (skips any game whose HTML is already on disk),
writing raw captures + a manifest under
`data/raw/officials_pfr_wayback/<run-id>/` and parsed crew rows to
`data/processed/officials_pfr_wayback/<run-id>/officials_2009_2014.parquet`.
A new `internet_archive_pfr_boxscores` entry in `config/source_policies.json`
governs it (`acquisition_allowed: true`, polite-crawl conditions). 15 tests
in `tests/test_officials_wayback_sweep.py` cover the parser, backoff
schedule, resume-skip logic, hard-stop counter, and manifest shape — all
offline (injected fake fetch functions, no network).

**Measured 2026-09-05, before running the packaged script**: a diagnostic
backoff probe against the exact endpoint the script uses
(`web.archive.org/cdx/search/cdx?url=pro-football-reference.com/boxscores/
201409040sea.htm...`, same contact User-Agent) drew **HTTP 429 on all 6
requests attempted**: an initial check, an immediate retry, then one retry
each after 60s, 120s, 240s, and 480s of backoff (900s of cumulative
escalating backoff across the scripted portion alone). The throttle from
2026-09-03/04 (`docs/officials_archive_probe.md`'s original finding above)
had not cleared as of this session, even at an 8-minute single backoff —
the same persistent pattern, not a one-off. Per this lane's own binding
"do not hammer" instruction, no further live requests were sent once that
pattern was unambiguous (6 consecutive 429s spanning a doubling backoff
sequence, well under the 20-request budget the task allowed before a
mandatory stop). Because the real-world block never cleared, the packaged
script itself was **not** run against live network this session (it would
reproduce the identical result while spending more of the host's
tolerance); its mechanics were instead verified with injected fake fetch
functions reproducing this exact 429-persistent scenario (see the test
file). **Yield this session: 0 games fetched, 0 rows parsed** — the 2014
first-tranche run named in this lane's task could not be attempted live.
No detached background continuation was started (the task's own condition
for launching one — "if the tranche succeeds" — was not met).

This does not change the unblock-order list above: path 1 (the polite
sweep) is now genuinely BUILT and ready to run the moment the throttle
clears (it may be time-of-day or cumulative-crawl-volume driven, not
permanent — worth a retry on a later day without further code changes),
but it did not itself clear the block. A future session should re-run
`scripts/officials_wayback_sweep.py --season-start 2014 --season-end 2014`
directly (no probing first — the mechanics are proven) before falling back
to path 2 (ESPN listings) or path 3 (declare 2015+ the population).
