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

## 2026-09-07: throttle cleared; a capture-selection defect found on the first live fetch

Measured this session: the polite sweep (`scripts/officials_wayback_sweep.py
--season-start 2014 --season-end 2014`) got HTTP 200 from both the CDX lookup
and the replay on its first game, so the 2026-09-03/04 429 throttle had
cleared after three days without crawling. The first fetched page, however,
parsed **0 officials**: the CDX query returned captures in ascending order
and the selector took the earliest one -- `20140530011957`, a capture of the
2014_01_GB_SEA boxscore URL taken 2014-05-30, more than three months BEFORE
the 2014-09-04 game. PFR boxscore URLs exist pre-game as placeholder pages;
the officials block only appears on the post-game page, so the selector's
own comment ("any capture ... carries the same pregame-fixed officiating
assignment") was wrong. Fix: the CDX query now carries
`from=<gameday + 1 day>` and `_select_capture_timestamp` re-applies that
bound client-side, choosing the earliest POST-game capture
(`capture_not_before`); pinned by
`tests/test_officials_wayback_sweep.py::test_pre_game_captures_are_never_selected`.
The one pre-game capture the aborted run wrote was deleted and the sweep
relaunched on a fresh run id.

**Second defect, same first batch (measured):** with post-game captures
selected, the first five 2014 pages (captures 2014-09-24 to 2014-10-07) still
parsed 0 officials. The archived 2014-era boxscore carries the crew in
`<table id="ref_info">` with each label in `<b>...</b>`, not the
`id="officials"` table the parser was written against. The table-id
alternation now accepts both; the 2014_01_GB_SEA table is checked in verbatim
as `tests/fixtures/pfr_boxscore_officials_ref_info_2014.html` (Referee John
Parry, seven positions) and pinned by
`test_parses_the_2014_era_ref_info_table_from_a_real_capture`. The sweep was
resumed on the same run id so the five fetched pages are re-parsed from disk
with zero new requests.

## 2026-09-07 (lane N): why 2009-2010 parsed 0 of 418, and the capture-selection fix

**Offline, measured on run `20260907T175420Z`** (423 manifest rows: 418
`fetched`, 4 `cdx_fetch_failed`, 1 `no_capture_found`; seasons 2009 = 251
fetched, 2010 = 167; captures dated 2009-09-25 to 2011-11-17, i.e. the
EARLIEST post-game capture under the old `limit=5` policy). Over the 418
HTML files: `grep -l -E "Umpire|Referee"` matches 0 files;
`grep -l -i Officials` 0; `grep -l ref_info` 0; `grep -l scorebox_meta` 0;
`grep -l 'id="game_info"'` 0; `grep -l -i Attendance` 0;
`grep -l -i scoring` 418. The pages are complete final boxscores in PFR's
pre-2012 layout (title, linescore, Scoring, Team Stats, Rushing & Receiving,
Defense & Returns; 30-57 KB each against ~316 KB for a 2014 page) with no
game-info or officials block of any kind. Nothing to parse; the parser is
not at fault. For comparison, run `20260907T140309Z` (2014): 248 of 254
fetched pages carry `id="ref_info"`; the 6 that do not are captures taken
1-2 days after kickoff (e.g. `201409250was` captured 2014-09-26), and 5
further rows show `officials_parsed: 0` only because they were fetched
before the ref_info parser fix and the resume path never wrote the re-parse
back to the manifest (fixed below).

**Live probe, measured, 25 requests to `web.archive.org` only** (cap was 40;
8 s floor; two probe directories, one file per fetch:
`data/raw/officials_pfr_wayback/laneN_probe_20260907T233340Z/` and
`.../laneN_probe_20260907T233532Z/`, each with `manifest.json`, `cdx/`,
`html/<pfr_id>__<capture_ts>.html`). Full unbounded CDX lists (no `limit`):

| game | post-game captures | parsed 0 at | parsed 7 at |
| --- | --- | --- | --- |
| 200909100pit (TEN@PIT, 2009-09-10) | 61, 2009-10-02 .. 2026-09-02 | 2010-04-19, 2011-12-09, 2012-08-19 | 2013-09-14, 2016-06-29, 2026-09-02 (Referee Bill Leavy) |
| 201109080gnb (NO@GB, 2011-09-08) | 73, 2011-09-27 .. 2026-08-21 | 2012-06-14 | 2012-10-25, 2012-11-05, 2013-11-24, 2017-06-12, 2026-08-21 (Referee Clete Blakeman) |
| 201309050den (BAL@DEN, 2013-09-05) | 74, 2013-09-08 .. 2026-02-23 | (none fetched) | 2013-09-29, 2013-11-05, 2014-04-02, 2015-09-26, 2017-07-02, 2026-02-23 (Referee Walt Coleman) |

So the officials block appeared on PFR boxscores between **2012-08-19
(absent) and 2012-10-25 (present)** on the same URLs, and every capture
from 2012-10-25 onward carries it -- for 2009 and 2011 games alike -- in two
layouts the parser already reads: `<table id="ref_info">` (2012-2015
captures) and `<table id="officials">` inside an HTML comment (2016+
captures, including every 2026 capture fetched). The coordinator's
hypothesis (earliest capture predates the block; a later capture carries
it) is confirmed. Two throttle notes: the first probe pass drew HTTP 429 on
two CDX calls with no backoff (a 60 s backoff cleared a single replay 429
in the second pass), and a CDX `limit=-3` ("last N") query drew HTTP 504,
so the sweep now sends no `limit` at all -- the unbounded lists above came
back 200 at 61-74 rows.

**Changed in `scripts/officials_wayback_sweep.py`** (31 tests in
`tests/test_officials_wayback_sweep.py`, up from 17; all offline):

- Selection prefers the NEWEST post-game capture (`_rank_capture_timestamps`,
  newest first, `from=` bound re-applied client-side). An officiating
  assignment is fixed before kickoff, so capture recency changes nothing
  about pregame safety; `effective_time` stays the game date.
- Per-game fallback: if the chosen capture parses 0 officials, up to
  `--fallback-captures` (default 2) further captures are tried newest-first
  before the game is recorded parsed-0. A fallback replay failure keeps the
  page already fetched and still counts toward the consecutive-failure stop.
- `--retry-unparsed` (default off): a resumed run re-attempts games whose
  manifest row parsed 0 -- re-parsing the on-disk page under the current
  parser first (zero requests if it now parses), then fetching newer
  captures only if still 0, excluding captures already on disk. Rows are
  upserted (no duplicate rows for re-attempted games, which also fixes the
  pre-existing failed-row duplication on plain resume).
- Every fetched page is its own immutable file
  `html/<pfr_id>__<capture_ts>.html`; the manifest row's `html_file` names
  the page the crew rows came from and `attempted_captures` lists every
  capture fetched (old rows keep `html/<pfr_id>.html` and stay valid).
- The resume path now writes the current parser's `officials_parsed` back
  to the manifest row, and `--limit` caps new network work without skipping
  the re-parse of pages already on disk.

Rehearsed offline (measured, on scratchpad COPIES of both real run
directories with a fake CDX that returns only the already-fetched
timestamp): the 2014 copy re-parses with 0 requests and its 5 stale rows
go to 7 officials; the 2009-2013 copy retries one CDX call per game, keeps
423 rows, and marks each `no post-game capture beyond those already fetched`.

**Re-sweep commands (coordinator launches; not run by this lane).** The
2009-2013 run first (418 retry games + 862 unfetched 2011-2013 games + 5
failed rows; roughly 2 requests per game, so ~2,570 requests at the 8 s
floor is ~5.7 h before any backoff -- inferred from the counts, not
measured), then the 2014 run's 6 genuine zero-parse games and 2 failed rows
(~16 requests):

```powershell
.\.tools\uv.exe run --no-sync python scripts\officials_wayback_sweep.py `
    --season-start 2009 --season-end 2013 --run-id 20260907T175420Z --retry-unparsed

.\.tools\uv.exe run --no-sync python scripts\officials_wayback_sweep.py `
    --season-start 2014 --season-end 2014 --run-id 20260907T140309Z --retry-unparsed
```

Both are resumable with the same flags if interrupted. The first thing to
check after a few games: `parsed_zero=` in the summary line should stay
near 0 once the newest captures are being read.
