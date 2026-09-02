# Jeff Sagarin NFL ratings backfill via the Wayback Machine

Follow-up build to rank #2 in `docs/archive/data_source_scout_v4.md` ("Sagarin
ratings via the Wayback Machine"). Scope of this document, per the task that
produced it: **ingestion plus a coverage report only** -- no experiment was
run, no `nfl-ats weak-signals` registry entry was written, and no
`src/nfl_ats` code was touched. Section 6 predeclares the follow-up
experiment (Sagarin-implied spread minus market opener, as a divergence
signal) for a later session to run.

Every claim below is tagged **measured** (fetched/run this session, exact
command or artifact path given), **read** (a file opened this session),
**reported** (unverified), or **inferred** (reasoning), per the binding
`AGENTS.md` labeling rule.

---

## 1. What was built

`scripts/ingest_sagarin_ratings.py` (new). CDX-enumerates Wayback captures of
Jeff Sagarin's NFL power ratings across two URL eras, fetches each distinct-
content capture, parses the fixed-width ratings table out of the HTML with
an era-tolerant regex parser, normalizes team names to current nflverse team
codes, and builds a per-(season, week, team) "as-of-Tuesday" alignment view
against the project's own already-ingested nflverse schedule snapshot.

Rate-limited to ~1 request/sec throughout (CDX queries and page fetches
alike), per the task's instruction. Fetches go through `curl` as a
subprocess rather than Python's `urllib` -- **measured** this session:
identical CDX requests that `urllib.request.urlopen` reliably timed out on
(repeated 30s "read operation timed out" against
`web.archive.org/cdx/search/cdx`) succeeded via `curl` in seconds, matching
`docs/archive/data_source_scout_v4.md`'s own prior finding for this domain ("direct
WebFetch to web.archive.org URLs fails at the tool level... curl
succeeded"). Separately **measured**: passing the CDX API's own
`collapse=digest` parameter (server-side de-duplication of consecutive
identical-content captures) made several of the lower-traffic `nfl<YY>.htm`
queries go from a normal ~2-9s round trip to a consistent 30s+ timeout /
connection failure -- a server-side cost of that parameter, not a client
issue. The script fetches the full, uncollapsed capture list per URL and
collapses consecutive same-digest rows itself in Python instead.

**Three more archive.org quirks found and fixed live this session** (all
three now permanent behavior in `_fetch`, not one-off workarounds):

1. **Gzip bodies with no negotiation.** 28 of the first 113 sagarin.com
   captures fetched parsed as `no_header_match` despite a 200 status; the
   cached bytes started with the gzip magic number `\x1f\x8b`. **Measured**:
   `curl` without `--compressed` does not auto-decompress a `Content-Encoding:
   gzip` body it never asked for; Wayback evidently replays the *original*
   2023+-era response verbatim, gzip header included, regardless of what the
   client requested. Fixed by adding `--compressed` to every `curl` call.
2. **Silent same-timestamp redirects.** After the gzip fix, 2 captures still
   failed with an *empty* body despite a "successful" status. **Measured**
   (`curl -I`): those exact CDX-listed timestamps return HTTP 302 with
   `x-archive-redirect-reason: found capture at <nearby timestamp>` -- the
   CDX index listed a timestamp Wayback itself does not serve independently.
   Fixed by adding `-L` (follow redirects) to `curl`.
3. **A 0-byte cache poisons the resume check forever.** Because `_fetch`
   originally accepted any 200/301/302 response including an empty body, the
   2 redirect-affected captures got written to disk as literal 0-byte `.html`
   files -- and `html_path.exists()` (the resume/dedup check) treats a 0-byte
   file as "already fetched," so a naive re-run would never self-heal. Fixed
   by treating an empty response body as a retryable failure inside `_fetch`
   itself, not just at the caller.

After all three fixes, the primary 2010-2025 target (`sagarin.com` era)
reached **113/113 fetched, 113/113 parsed** -- see section 5.

---

## 2. The two URL eras

### Era A -- `sagarin.com/sports/nflsend.htm` (current, 2010-2026)

**Measured** (CDX, `web.archive.org/cdx/search/cdx?url=sagarin.com/sports/nflsend.htm`):
one URL for the entire "current season," continuously re-published and
re-crawled since 2010, through the most recent captures found this session
(2026-08). Header line format:

```
NFL 2014 through games of October 13 Monday - Week #6
```

Home-edge line carries **four** bracketed values, one per rating method:

```
HOME ADVANTAGE=[  2.73]        [  2.75]       [  2.75]       [  2.71]
```

Each team row carries three method columns after the overall `RATING`:
`GOLDEN_MEAN`, `PURE POINTS` (Sagarin's own "PREDICTOR"), `ELO_SCORE`.

### Era B -- `www.usatoday.com/sports/sagarin/nfl<YY>.htm` (1998-2011)

**Measured** (CDX): one URL *per season* (`nfl98.htm` .. `nfl11.htm`,
`YY` = two-digit season year), live with `statuscode 200` through the
season's own USA Today hosting; from ~2012 the path 301s/404s (USA Today's
site restructure orphaned it) and `sagarin.com` became the sole "current
season" URL. Two header sub-variants were measured within this era:

```
Final NFL 2009 Ratings through results of  2010 FEBRUARY 7 SUNDAY - Super Bowl
NFL 2003 Ratings thru results of  SUNDAY, FEBRUARY 1, 2004 - Super Bowl - FINAL
Final 1998 NFL ratings                                    <- terse, no date/week at all
```

("Final" as a leading prefix, "FINAL" as a trailing word after the date
clause, or -- for the earliest, thinnest-archived 1998 season -- no
through-date clause at all, just season + "NFL ratings.") Home-edge line
carries **one** value:

```
HOME ADVANTAGE=  2.96          RATING    W   L   T  SCHEDL(RANK) ...
```

Each team row carries two method columns: `ELO_CHESS`, `PURE POINTS`. The
very earliest measured sample (1998 season, 30 teams -- correct for that
season, pre-Cleveland-1999/Houston-2002 expansion) carries **zero** method
columns at all, just the core rank/name/RATING/W-L-T/SCHEDL/vs-top10/vs-top16
fields -- the parser's team-row regex treats the trailing method-column
group as optional for exactly this reason.

Both eras' header lines repeat the current season's `Week #N` marker in
Era A but Era B's own header rarely does (measured: none of the sampled Era
B captures carried a `Week #N` token, only a spelled-out date and/or
"Super Bowl"/"FINAL"). **Week alignment therefore does not depend on this
in-page label at all** -- section 4 aligns every capture to an nflverse
schedule week purely from the Wayback capture timestamp, independent of
whatever the page's own header says.

---

## 3. Parsing

Both eras share one row shape after HTML-tag/entity stripping:

```
   1  Denver Broncos          =  28.30    4   1   0   21.45(   4)   2  1  0 |   3  1  0 |   28.56    1 |   28.06    1 |   28.36    1
rank  team name                  RATING   W   L   T   SCHEDL(RANK)  vs-top10   vs-top16    method1 val+rank  method2 ...  method3 ...
```

One era-tolerant regex (`TEAM_ROW_RE` in the script) captures rank through
vs-top16 unconditionally, then an *optional* trailing `| VALUE RANK ...`
tail whose method-column count (0, 2, or 3) is read off however many
`VALUE RANK` pairs the tail actually contains -- no era needs its own
separate row regex. The header line is matched by a primary regex covering
both the `Final NFL <year> ... through|thru results of|games of ...` shapes
(with `is_final` detected either from a leading `Final ` prefix or a
trailing `FINAL` token), falling back to the terse `Final 1998 NFL ratings`
shape when the primary regex finds nothing.

The full page repeats the same 32-team table three times for readability (a
"top-to-bottom by RATING" listing, then division-average summaries, then a
"by division" listing covering the same 32 teams again) -- the parser
dedupes by team name, keeping the first occurrence (the top-to-bottom list),
since every repeat within one capture carries identical values.

**Team-name normalization.** Sagarin spells out full team names, not
nflverse codes. `NAME_TO_CODE` (in the script) maps every historical
spelling straight to the **current** nflverse code per the task brief:
Oakland Raiders / Los Angeles Raiders / Las Vegas Raiders -> `LV`; San Diego
Chargers / Los Angeles Chargers -> `LAC`; St. Louis Rams / Los Angeles Rams
-> `LA`; Washington Redskins / Washington Football Team / Washington
Commanders -> `WAS`; Tennessee Oilers (1998-99, the pre-Titans name, present
in the earliest measured USA Today sample) -> `TEN`. The raw parsed string
is kept in `team_name_raw` for audit; every mapped name in the samples
checked this session mapped cleanly (see section 5's per-era spot check).

**Sample-verified parse correctness (measured, four hand-checked captures
spanning both eras and every row-shape variant found):**

| sample | era | season | teams parsed | unmapped teams | notes |
|---|---|---:|---:|---:|---|
| `sagarin.com` 2014-10-18 capture | A | 2014 | 32 | 0 | 3 method columns, 4-value home edge |
| `usatoday` nfl09.htm, 2010-10-06 capture | B | 2009 | 32 | 0 | "Final NFL 2009... Super Bowl" header, 2 method columns |
| `usatoday` nfl03.htm, 2004-07-01 capture | B | 2003 | 32 | 0 | "thru results of... - FINAL" trailing-FINAL header variant |
| `usatoday` nfl98.htm, 2000-01-18 capture | B | 1998 | 30 (correct: 30-team league) | 0 | terse header, zero method columns |

---

## 4. As-of-Tuesday / as-of-prekickoff alignment

Every Sagarin capture is one NFL-wide snapshot (32 team rows, one Wayback
timestamp). `build_week_windows` reads the project's own already-ingested
nflverse schedule snapshot (`data/raw/<UTC>/schedules.parquet`, the most
recent one found under `data/raw/`) and, for every `(season, week)` --
regular season and postseason together, using the schedule's own continuous
`week` numbering -- computes:

- `first_kickoff_utc`: the earliest `gameday` + `gametime` in that week,
  `gametime` treated as US/Eastern and converted to UTC.
- `tuesday_cutoff_utc`: midnight UTC of the Tuesday on or immediately before
  `first_kickoff_utc`'s UTC calendar date.

For each `(season, week)`, `build_asof_view` finds the **latest** Sagarin
capture (by `capture_ts`) at or before `tuesday_cutoff_utc` (`has_tuesday_snapshot`)
and, separately, the latest capture strictly before `first_kickoff_utc`
(`has_prekickoff_snapshot`), then joins that capture's 32 team rows in.

**Known imprecision, stated plainly rather than smoothed over:** because
`tuesday_cutoff_utc` is built from the UTC *calendar date* of first kickoff
(not a timezone-correct back-conversion through the game's own local Tuesday
noon ET), it lands at Monday ~8pm ET rather than Tuesday noon ET -- roughly
16 hours **earlier** (more conservative) than the pool's actual Tuesday-noon
lock convention (`MEMORY.md`, "picks lock at kickoff... only LINES freeze
Tuesday"). This makes the `has_tuesday_snapshot` counts in section 5 a
conservative **undercount** of true Tuesday-lock coverage, not an overcount
-- appropriate for a coverage report, but this alignment view is not
precise enough on its own for a leakage-sensitive production join without
redoing the timezone arithmetic properly first.

---

## 5. Coverage, measured

Snapshot: `data/raw/sagarin/20260820T112501Z/` (gitignored, local only).
**Measured** from `manifest.json`, `captures_log.parquet`, `index.parquet`,
and `asof_tuesday_view.parquet` after the final clean rebuild this session
(`fetched_at_utc: 2026-08-20T11:52:16Z`).

### 5.1 Run-level summary -- Era A, `sagarin.com` (primary target, 2010-2025)

| | |
|---|---:|
| Captures attempted (CDX, distinct content, client-side digest-collapsed) | 113 |
| Captures fetched successfully | **113 / 113 (100%)** |
| Captures parsed successfully | **113 / 113 (100%)** |
| Team-rating rows in `index.parquet` | 3,616 (113 captures x 32 teams) |
| Unmapped team names (`team_code` null) | **0** |
| Seasons covered | 2010-2025 (16 seasons) |
| `era_format` breakdown (rows) | `sagarin_com` (4-method, 3-bracket-plus-overall home edge) 3,296; `usatoday`-shaped (2-method, single home-edge value -- Sagarin used the simpler layout on his own domain in some 2010-2011 captures too) 128; `unknown` (the preseason "Starting Ratings" 3-bracket variant, section 3) 192 |

### 5.2 Per-season captures and parse rate (Era A)

| season | captures attempted | fetch ok | parse ok | parse rate |
|---:|---:|---:|---:|---:|
| 2010 | 2 | 2 | 2 | 100% |
| 2011 | 3 | 3 | 3 | 100% |
| 2012 | 3 | 3 | 3 | 100% |
| 2013 | 4 | 4 | 4 | 100% |
| 2014 | 17 | 17 | 17 | 100% |
| 2015 | 15 | 15 | 15 | 100% |
| 2016 | 7 | 7 | 7 | 100% |
| 2017 | 8 | 8 | 8 | 100% |
| 2018 | 9 | 9 | 9 | 100% |
| 2019 | 6 | 6 | 6 | 100% |
| 2020 | 7 | 7 | 7 | 100% |
| 2021 | 1 | 1 | 1 | 100% |
| 2022 | 6 | 6 | 6 | 100% |
| 2023 | 12 | 12 | 12 | 100% |
| 2024 | 9 | 9 | 9 | 100% |
| 2025 | 4 | 4 | 4 | 100% |

Capture density is real but uneven -- 2021 has only one distinct-content
capture the whole season (Sagarin's page apparently changed rarely, or
Wayback's crawl of it that year was thin), while 2014-2015 and 2023-2024 are
much denser (12-17). Section 5.3 is the number that actually matters for
pool relevance: how many *schedule weeks* each season's captures land in.

### 5.3 Per-(season, week) as-of alignment (Era A)

One row per `(season, week)` (regular season + postseason together, using
the schedule's own continuous week numbering). `weeks_with_tuesday`: a
capture exists at or before that week's conservative Tuesday cutoff (section
4's known undercount-biased definition). `weeks_with_prekickoff`: a capture
exists strictly before that week's first kickoff (a looser bound -- "was any
Sagarin snapshot public before ANY game in that week kicked off").

| season | weeks | weeks with Tuesday snapshot | weeks with any pre-kickoff snapshot |
|---:|---:|---:|---:|
| 2010 | 21 | 9 (43%) | 9 (43%) |
| 2011 | 21 | 11 (52%) | 11 (52%) |
| 2012 | 21 | 12 (57%) | 13 (62%) |
| 2013 | 21 | 19 (90%) | 20 (95%) |
| 2014 | 21 | 19 (90%) | 19 (90%) |
| 2015 | 21 | 20 (95%) | 21 (100%) |
| 2016 | 21 | 20 (95%) | 20 (95%) |
| 2017 | 21 | 21 (100%) | 21 (100%) |
| 2018 | 21 | 18 (86%) | 18 (86%) |
| 2019 | 21 | 17 (81%) | 18 (86%) |
| 2020 | 21 | 21 (100%) | 21 (100%) |
| 2021 | 22 | 9 (41%) | 9 (41%) |
| 2022 | 22 | 11 (50%) | 12 (55%) |
| 2023 | 22 | 20 (91%) | 20 (91%) |
| 2024 | 22 | 19 (86%) | 20 (91%) |
| 2025 | 22 | 19 (86%) | 19 (86%) |

Read plainly: 2013-2020 and 2023-2025 all clear 80%+ weekly Tuesday-snapshot
coverage (several seasons at or near 100%), while 2010-2012 and 2021-2022
are the thin years (41-62%). 2021 in particular is thin on BOTH axes (only
one distinct-content capture all season, section 5.2), so its 9/22 is a real
archive-density gap, not a parsing miss.

### 5.4 Era B (USA Today, 1998-2011) -- consolidated, with seven fetch gaps

Per the task's explicit fallback instruction ("if the archive is denser than
[the fetch budget] allows, prioritize complete recent seasons and report the
exact resume command"): Era A (the primary 2010-2025 target) was completed
first and is 100%/100% clean (above). Era B was then started as the
explicitly-bonus "deeper if cheap" extension and did **not** finish inside
this session's fetch budget -- archive.org's CDX endpoint was measured this
session to be intermittently very slow (`Operation timed out after 30000ms`,
`Failed to connect... port 443`, and one literal "Internet Archive:
Temporarily Offline" banner page, all measured directly), and Era B's CDX
enumeration alone (14 separate per-season queries) repeatedly hit this.
**Measured, initial partial state left on disk**: 37 raw HTML pages cached under
`data/raw/sagarin/20260820T112501Z/pages/usatoday/{nfl98,nfl99,nfl00,nfl01}/`
(4 of 14 season URL keys touched). The background extension process was
stopped cleanly (not crashed), leaving that partial cache safe to resume.

**Measured 2026-08-20, first bounded resume:** the exact section 7 command was
run against the same snapshot until the owner requested the process stop at
the next bounded checkpoint. The USA Today cache grew from **37 to 242 pages
(+205)** and from four to eight URL keys / parsed seasons (1998-2005).

**Measured 2026-08-20, second bounded resume:** the same exact command then ran
for its full **1,200-second** cap. It timed out with exit code 124 before the
end-of-run consolidation, after growing the cache from **242 to 394 pages
(+152 this pass; +357 cumulatively)** and reaching twelve URL keys / parsed
seasons (1998-2009). The exact surviving uv/Python process chain was stopped
and a process-command-line audit found no matching process afterward. Current
durable cache at that intermediate checkpoint was 394 pages.

**Measured 2026-08-20, final bounded resume:** the exact same command completed
normally in **953.3 seconds** (inside its 1,200-second cap), fetched the
remaining 2010-2011 season keys, processed all cached captures, and rewrote all
four consolidated outputs.

**Measured 2026-08-20, post-consolidation retry:** the exact same command was
run behind a 600-second wrapper. The wrapper timed out with exit code 124 after
604.0 seconds, but on Windows its uv/Python/curl child tree survived the wrapper
and completed the consolidated writes at `2026-08-20T16:49:26.801702+00:00`
before the follow-up command-line audit could terminate it. Process start and
manifest timestamps put the actual run at approximately 628.8 seconds, about 25
seconds beyond the wrapper timeout; no matching child remained afterward. This
is recorded as a bound overrun, not as a within-bound completion. Two of the
five prior curl-7 connection failures recovered and parsed cleanly:
`nfl08@20080910215335` (season 2008, 32 teams) and
`nfl06@20080917012429` (season 2006, 32 teams). Current USA Today cache and
capture-log reconciliation:

| URL key | season | CDX captures | pages cached / parsed | final fetch failures |
|---|---:|---:|---:|---:|
| `nfl98` | 1998 | 38 | 37 | 1 |
| `nfl99` | 1999 | 58 | 58 | 0 |
| `nfl00` | 2000 | 56 | 56 | 0 |
| `nfl01` | 2001 | 59 | 59 | 0 |
| `nfl02` | 2002 | 27 | 25 | 2 |
| `nfl03` | 2003 | 9 | 8 | 1 |
| `nfl04` | 2004 | 14 | 14 | 0 |
| `nfl05` | 2005 | 87 | 87 | 0 |
| `nfl06` | 2006 | 21 | 20 | 1 |
| `nfl07` | 2007 | 17 | 16 | 1 |
| `nfl08` | 2008 | 29 | 28 | 1 |
| `nfl09` | 2009 | 7 | 7 | 0 |
| `nfl10` | 2010 | 11 | 11 | 0 |
| `nfl11` | 2011 | 46 | 46 | 0 |
| **total** | **1998-2011** | **479** | **472** | **7** |

**Measured** (read-only audit with
`scripts.ingest_sagarin_ratings.parse_capture_html` over every cached page):
all **585/585** durable pages parse successfully (472 USA Today plus the 113
Era A pages), with zero parser exceptions, zero `parse_error` statuses, zero
unmapped team names, and zero zero-byte pages. Parsed team-count distribution
is 30 teams on 37 pages, 31 on 173, and 32 on 375. Of the 585 pages, 465
identify as USA Today format, 103 as Sagarin.com format, and 17 as `unknown`
format while still passing the header/team-row parser. The 17 are 11 USA Today
2011 pages plus one Sagarin.com 2011 page, three 2012 pages, and two 2013 pages;
they have no parsed home-edge value, so they are row-usable but not
implied-spread-usable. **Inferred:**
that distribution is consistent with the league's historical expansion rather
than parser truncation.

**Measured consolidated checkpoint:** `manifest.json` now records **592
captures attempted, 585 fetch-ok, 7 fetch-failed, 585 parse-ok, and 18,473
index rows**; `captures_log.parquet` has 592 rows, `index.parquet` has 18,473,
and `asof_tuesday_view.parquet` has 9,848. The selected schedule snapshot
(`data/raw/20260817T235649Z/schedules.parquet`) begins in 2009, so seasons
1998-2008 are indexed but cannot appear in this as-of view. Measured weekly
Tuesday/pre-kickoff coverage is 21/21 and 21/21 in 2009, 20/21 and 20/21 in
2010, and 20/21 and 20/21 in 2011.

**Measured artifact integrity** (SHA-256 after the successful consolidation):

| file | bytes | SHA-256 |
|---|---:|---|
| `manifest.json` | 885 | `f9e33196ec5c8a09d4696e48fcd2532764ae6c4b6b2a71ebd29965979e5e7d09` |
| `captures_log.parquet` | 50,644 | `5802ea44c2dca193c6d869cdd773826bbb94a93188f6ff85e702d0b50eca4da0` |
| `index.parquet` | 213,969 | `41ce2aa9cdded9218eabebf95e84317814b35bfacbd393c78becd16c3759c52d` |
| `asof_tuesday_view.parquet` | 138,089 | `051f1dc3b0d2b42ef5743ce97b96f4e06515039fca812c4e545e33afbd66bcb5` |

**Measured current fetch failures:** three captures still end with curl 61
(`incorrect header check` while decoding content):
`nfl07@20080511204634`, `nfl06@20080511205104`, and
`nfl02@20080512003808`. Four captures now end with `[Errno 22] Invalid
argument`: `nfl03@20080725163954`, `nfl98@20080726171311`,
`nfl02@20081002130352`, and `nfl08@20091027194548`. The first of those four
had been a curl-61 gzip failure in the prior pass; the other three had been
curl-7 connection failures. Therefore all four captures originally associated
with gzip/decoding errors remain gaps, while two of the five originally
transient curl-7 gaps recovered. These seven are fetch gaps, not parser
failures; they remain explicit rows in `captures_log.parquet`, not silently
omitted.

**Measured provenance warning:** section 8's ATS screen was **not rerun** in
any resume pass. Its recorded numbers belong to the pre-Era-B consolidated
checkpoint and must not be presented as a result computed from the new combined
18,473-row index / 9,848-row as-of files. This section reports ingestion only.

---

## 6. Predeclared next-step experiment (NOT run this session)

**Question:** does `(Sagarin-implied spread) - (market opening spread)`, read
at the same as-of-Tuesday cutoff this ingestion builds, carry signal for
against-the-spread outcome beyond the opener itself? Sagarin's `RATING`
column, plus its own stated home-edge number
(`home_edge_rating`/`home_edge_golden_mean`/`home_edge_pure_points`/
`home_edge_elo_score` in `index.parquet`), converts directly to an
implied-spread number per Sagarin's own stated convention (rating
difference, home team credited the stated home edge) -- comparable, not
identical, units to a market point spread. **Construction note per
`AGENTS.md`'s commensurability rule:** the pooled/compared quantity must be
declared before signs are seen -- the natural candidate is
`sagarin_rating_diff_plus_home_edge - market_opening_spread`, both in point
units, on the same population of games, with the family (which Sagarin
method column: `RATING`, `GOLDEN_MEAN`, `PURE_POINTS`, or `ELO_SCORE`/
`ELO_CHESS`) fixed before running, not chosen after seeing which one
performs best.

**Method (not run):** for each game, take the Sagarin capture selected by
this ingestion's `asof_tuesday_view.parquet` (or a stricter cutoff, if this
session's Monday-evening-vs-Tuesday-noon imprecision, section 4, is fixed
first), difference it against the market opener already in the project's
own odds data, and test whether the divergence predicts ATS outcome
above/below what the opener alone predicts, on out-of-sample chronological
folds, exactly as `AGENTS.md`'s evaluation invariants require (leakage
regression test for the new feature family; validation/selection/
calibration/outer-test periods kept distinct).

**Binding closing-grounds taxonomy, restated verbatim per `AGENTS.md` and
`CLAUDE.md`:**

> An interval or CI that contains zero is NEVER grounds to reject, fail, or
> close an experiment. Only two grounds ever close a line of work: (1)
> refuted mechanism — a RESOLVED wrong sign (whole interval on the wrong
> side of zero) or zero split-half reliability; (2) bounded by a positive
> control proven able to detect an effect that size. Everything else is
> `unresolved_below_power`: record it with `nfl-ats weak-signals record`,
> report `probability_positive`, never the binary 'contains zero'.

Whatever this experiment finds, it must be recorded through
`nfl-ats weak-signals record` with an admissible classification (most likely
`unresolved_below_power` on first measurement, per the same taxonomy that
governs every other candidate in the registry) -- never described as
"settled" or "negative" from prose alone.

---

## 7. Resume / follow-up commands

The real snapshot from this session is `20260820T112501Z`. Era A and Era B now
have a combined, self-consistent consolidated checkpoint. The first command
below is an optional retry of Era B's seven explicit fetch gaps; the second
extends Era A forward as new weeks get archived.

```powershell
# Retry the seven explicit Era B fetch failures. All 585 successful pages are
# cache hits; captures_log.parquet will retain any capture that still fails.
# A successful retry keeps index.parquet / asof_tuesday_view.parquet combined:
.\.tools\uv.exe run --no-sync python scripts/ingest_sagarin_ratings.py `
    --out data/raw/sagarin --snapshot 20260820T112501Z `
    --start-season 1998 --end-season 2011

# Extend Era A forward as new weeks are archived (omit --snapshot to resume
# the most recent existing snapshot automatically; --no-usatoday keeps this
# fast by skipping the already-slow Era B CDX queries):
.\.tools\uv.exe run --no-sync python scripts/ingest_sagarin_ratings.py `
    --out data/raw/sagarin --start-season 2010 --end-season 2026 --no-usatoday
```

---

## 8. Section 6's predeclared experiment, run (measured)

**Ceiling caveat up front, per the project's own prior measurement**: a
feature that only measures team quality *better than the market* is bounded
near zero (`team-quality-is-already-priced`, project MEMORY.md). Sagarin
divergence from the market line is exactly that family -- an independent
power rating's disagreement with the market's own consensus price. The job
of this screen is to locate WHERE in the divergence distribution any
residual signal lives (tails vs. bulk, era, model-agreement), not to assume
one exists going in. Every result below is `unresolved_below_power` --
**measured** this session, none of it is a negative finding, per the binding
closing-grounds taxonomy restated in section 6 and in `AGENTS.md`.

**Predeclaration**, frozen before any effect was computed:
`<scratchpad>/sagarin_divergence/predeclaration.json` (session-local,
not tracked in the repo; its construction and cell definitions are
summarized in full below). Method family fixed to
Sagarin's `RATING` column plus its own `home_edge_rating` value --
`GOLDEN_MEAN`/`PURE_POINTS`/`ELO_SCORE`/`ELO_CHESS` were not touched.
`sagarin_implied_spread_home = home_rating - away_rating + home_edge_rating`
(home-positive, matching `spread_line`'s own home-favored-positive
convention, verified via `nfl_ats.features.add_ats_outcomes`). Two grades:
CLOSE (`divergence_close = sagarin_implied_spread_home - spread_line`, the
schedule's own recorded line, REG 2010-2025) and OPENER
(`divergence_open = sagarin_implied_spread_home - tue_open_home_spread`,
the project's historical odds-snapshot archive, REG 2020-2025, restricted
to `nfl_ats.experiment_runner._opener_graded_features`'s own paired
tue_open+close population -- the same 1,537-game archive
`docs/opener_evaluation.md` documents). Implemented in
`scripts/sagarin_divergence_battery.py`; recorded via
`scripts/record_sagarin_divergence_battery.py`.

### 8.1 Join coverage, honestly (measured)

Team codes normalized through `nfl_ats.constants.TEAM_ABBREVIATION_ALIASES`
(OAK->LV, SD->LAC, STL->LA) before joining, matching the current-code
convention the Sagarin ingestion already applied to its own team names.

**A real coverage gap found only by actually performing the join, not
visible in section 5.3's week-level table**: section 5.3 counts ANY
Tuesday-asof capture as covered. But **every one of 2012's captures, and
2013's weeks 1-12, parsed with `era_format="unknown"` and a null
`home_edge_rating`** (measured: `home_edge_rating`, `home_edge_golden_mean`,
`home_edge_pure_points`, and `home_edge_elo_score` are ALL null on these
captures -- not just the preseason "Starting Ratings" variant section 3
already flagged, but several genuine in-season 2012 captures too, e.g. a
"Week #8" and an "End of Regular Season" header). A game needs a non-null
`home_edge_rating` to get a `sagarin_implied_spread_home` at all, so **2012
contributes zero usable games to this screen and 2013 only contributes
usable games from week 13 onward** -- a real archive-format gap, not a join
bug (verified by inspecting `index.parquet`'s raw `home_edge_*` columns for
every `era_format="unknown"` capture; all four are null on all of them).

Close-grade population: **2,684 REG games, 2010-2025** with a usable
Sagarin snapshot and a non-push `home_cover`.

| season | games | games w/ usable Sagarin | coverage |
|---:|---:|---:|---:|
| 2010 | 256 | 80 | 31.2% |
| 2011 | 256 | 110 | 43.0% |
| 2012 | 256 | **0** | **0.0%** |
| 2013 | 256 | 80 | 31.2% |
| 2014 | 256 | 224 | 87.5% |
| 2015 | 256 | 240 | 93.8% |
| 2016 | 256 | 240 | 93.8% |
| 2017 | 256 | 256 | 100.0% |
| 2018 | 256 | 208 | 81.2% |
| 2019 | 256 | 193 | 75.4% |
| 2020 | 256 | 256 | 100.0% |
| 2021 | 272 | 78 | 28.7% |
| 2022 | 271 | 107 | 39.5% |
| 2023 | 272 | 240 | 88.2% |
| 2024 | 272 | 224 | 82.4% |
| 2025 | 272 | 224 | 82.4% |

Opener-grade population: the `_opener_graded_features` paired archive is
**1,537 REG games, 2020-2025**; intersected with usable Sagarin coverage it
is **1,053 games**.

| season | games (paired archive) | games w/ usable Sagarin | coverage |
|---:|---:|---:|---:|
| 2020 | 227 | 227 | 100.0% |
| 2021 | 239 | 63 | 26.4% |
| 2022 | 255 | 99 | 38.8% |
| 2023 | 272 | 240 | 88.2% |
| 2024 | 272 | 224 | 82.4% |
| 2025 | 272 | 224 | 82.4% |

Model-agreement population (cell b): **1,492 games**, active model
`method="market_residual"` (matching `artifacts/active_ats_model.json`),
walk-forward evaluation artifact `margins/20260820T004951Z`, **season range
2018-2025** -- narrower than the 2010-2025 close population because that is
the walk-forward evaluation's own `min_train_games` cutoff, not a choice
made in this screen.

### 8.2 Cells, week-blocked bootstrap (measured)

20,000 draws, seed 20260820, block = season\*100+week. Metric =
`(mean(sagarin_side_cover) - 0.5) * 100` accuracy points for the 6
single-group cells; for the model-agreement cell, `(agree accuracy -
disagree accuracy) * 100`. `P+` = fraction of bootstrap draws with the
metric > 0 -- reported for every cell regardless of whether the interval
crosses zero, per the binding taxonomy.

| cell | seasons | n | effect (pts) | 95% CI | P+ |
|---|---:|---:|---:|---|---:|
| `sagarin_battery_large_divergence_close` (\|div\|>=3, close) | 2010-2025 | 1,072 | -0.466 | [-3.588, +2.627] | 0.371 |
| `sagarin_battery_large_divergence_open` (\|div\|>=3, opener) | 2020-2025 | 406 | -1.478 | [-5.978, +2.987] | 0.254 |
| `sagarin_battery_top_decile_close` (top 10% \|div\|, close) | 2010-2025 | 269 | +3.532 | [-2.245, +9.449] | 0.879 |
| `sagarin_battery_top_decile_open` (top 10% \|div\|, opener) | 2020-2025 | 106 | -0.943 | [-9.794, +7.895] | 0.393 |
| `sagarin_battery_large_divergence_era_2010_2016` | 2010-2016 | 376 | +2.926 | [-2.572, +8.486] | 0.845 |
| `sagarin_battery_large_divergence_era_2017_2025` | 2017-2025 | 696 | -2.299 | [-6.052, +1.506] | 0.115 |
| `sagarin_battery_model_agreement_close` (agree - disagree) | 2018-2025 | 1,492 | -0.947 | [-6.134, +4.274] | 0.360 |

Read plainly, per the taxonomy (no interval here is grounds to reject
anything): the close-grade large-divergence cell sits almost exactly at a
coin flip (P+ 0.371, slightly favoring the *market's* side over Sagarin's
at |div|>=3). The opener-grade version of the same cell leans the same
direction, more so (P+ 0.254) -- opener grading did not surface a larger
Sagarin edge here the way it has for other project signals; if anything the
smaller opener-paired sample (406 games) points further away from zero on
the market's side, not Sagarin's. The top-decile close-grade sub-cell is the
most Sagarin-favoring reading in the table (P+ 0.879, +3.53 pts on 269
games, 124 week-blocks) -- exactly the kind of "does the tail carry more
signal than the bulk" question this screen was built to ask -- but its own
opener-grade counterpart on the smaller 2020-2025 paired subset does not
replicate that direction (P+ 0.393, 106 games, 56 blocks); with only 106
opener-paired games in the top decile, that is a thin, likely underpowered
comparison, not a contradiction that resolves anything. The era split shows
real magnitude movement, not presence/absence, per the project's era
convention: 2010-2016 leans toward Sagarin (P+ 0.845) while 2017-2025 leans
toward the market (P+ 0.115) on the identical close-grade large-divergence
subset -- a magnitude difference across eras worth keeping in view, not
grounds for any verdict on its own (376 and 696 games respectively, far
from a positive-control-sized sample). The model-agreement cell (does
Sagarin confirming the active model's forced pick predict the model doing
better?) leans slightly the wrong way at this measurement (agree accuracy
50.20% vs. disagree accuracy 51.15%, gap -0.947 pts, P+ 0.360) -- not a
resolved wrong sign (the whole interval does not sit below zero; it spans
[-6.13, +4.27]), so `unresolved_below_power`, not `refuted_mechanism`.

### 8.3 Registry (measured)

All 7 cells recorded via `nfl-ats weak-signals record`
(`scripts/record_sagarin_divergence_battery.py`), classification
`unresolved_below_power` for every cell (none carries a predeclared sign, so
`wrong_sign_resolved` cannot apply to any of them; no positive control was
run, so `positive_control_bound` cannot apply either -- decided in the
predeclaration, before the numbers were seen). Verified present via
`nfl-ats weak-signals status` immediately after recording (registry total
went 320 -> 327, all 7 names confirmed, no race-condition re-record needed).
Source artifact:
`artifacts/sagarin_divergence_battery/20260820T120937Z/results.json`.

---

## 9. Complete-source replication after Era B consolidation (2026-08-21)

**[Measured]** The exact frozen seven-cell battery was rerun against consolidated
snapshot `20260820T112501Z`; no method, threshold, era split, grade, bootstrap
seed, or candidate sign changed. The authoritative artifact is
`artifacts/sagarin_divergence_battery/20260821T170345Z/results.json`, and its
tracked experiment record is
`registry/experiments/sagarin-divergence-battery/20260821T170345Z.json`.

**[Measured]** Source provenance is now part of the result itself: 592 captures
attempted, 585 fetched and parsed, seven explicit fetch failures, 18,473 index
rows, and SHA-256 hashes for `manifest.json`, `captures_log.parquet`,
`index.parquet`, and `asof_tuesday_view.parquet`. The screen physically projects
only season, week, team, rating, home edge, and Tuesday-availability columns; a
regression test prevents an ATS outcome or other retrospective Sagarin column
from entering that load.

**[Measured]** Usable close-grade coverage rose from 2,684 to 2,966 games.
The material source change is early-era coverage: 2010 rose from 80/256 to
240/256 games and 2011 from 110/256 to 240/256. The opener population remains
1,053 because its odds archive begins in 2020, outside the added Era-B seasons.

| cell | n | effect (accuracy pts) | `probability_positive` |
|---|---:|---:|---:|
| large divergence, close | 1,194 | -0.5863 | 0.3473 |
| large divergence, opener | 406 | -1.4778 | 0.2542 |
| top decile, close | 297 | +3.5354 | 0.8908 |
| top decile, opener | 106 | -0.9434 | 0.3928 |
| large divergence, 2010-2016 close | 498 | +1.8072 | 0.7620 |
| large divergence, 2017-2025 close | 696 | -2.2989 | 0.1152 |
| model agreement minus disagreement, close | 1,492 | -0.9474 | 0.3600 |

**[Inferred]** The complete early archive weakens, but does not reverse, the
old-era close-grade Sagarin lean: +2.926 points / P+ 0.845 became +1.807 /
P+ 0.762. The close-only top-decile tail remains the strongest Sagarin-facing
read, while its pool-relevant opener counterpart still leans away from Sagarin.
This supports no production or prospective pick change: the opener cells are
unchanged and neither is a like-for-like improvement over the 53.36% raw-model
baseline or the promoted 53.76% arrest-policy component evaluation.

**[Measured]** All seven existing weak-signal identities were replaced in place
from the authoritative artifact and remain `unresolved_below_power`; the
registry still contains exactly one row per name. The preliminary complete-source
run `20260821T170108Z` is retained as an audit record; it prompted the source-hash
improvement before the authoritative rerun and did not write the weak-signal
registry.

Exact commands:

```powershell
.\.tools\uv.exe run --no-sync python scripts/sagarin_divergence_battery.py `
    --sagarin-root data/raw/sagarin/20260820T112501Z
.\.tools\uv.exe run --no-sync python scripts/record_sagarin_divergence_battery.py `
    --results artifacts/sagarin_divergence_battery/20260821T170345Z/results.json `
    --replace
```

---

## 9. 2012 coverage fix (2026-09-01)

WP19: a data-coverage fix only. **No experiment was rerun and no
`registry/weak_signals.json` entry was rescored** -- section 8's seven
`sagarin_battery_*` cells are untouched; see the closing note at the end of
this section for exactly what a re-measure on the corrected data would
require.

### 9.1 Cause, measured

Section 8.1 already found that every 2012 capture (and 2013 weeks 1-12)
parsed with `era_format="unknown"` and a null `home_edge_rating`. Opening
the cached captures explains why. **Measured** (raw bytes,
`data/raw/sagarin/20260820T112501Z/pages/sagarin_com/nflsend/20121031035824.html`,
the "NFL 2012 ... Week #8" capture) around where the home-advantage value
should be:

```
HOME ADVANTAGE=[<font color="#9900ff">  2.02</font>]                                                   [<font color="#ff0000">  0.91</font>]       [<font color="#0000ff">  1.33</font>]</B>
   1  San Francisco 49ers     =<font COLOR="#9900ff">  30.74</font>    6   2   0   20.97(  13)   2  1  0 |   3  2  0 |<font COLOR="#ff0000">   29.11    4 </font>|<font COLOR="#0000ff">   31.24    1</font>
```

This is a **third, transitional home-advantage line format** the original
parser never handled: **three** bracketed values (`HOME ADVANTAGE=[2.02]
...[0.91]...[1.33]`), not the four the `sagarin.com`-era regex
(`HOME_ADVANTAGE_4`) requires and not the bare single value the USA-Today-era
regex (`HOME_ADVANTAGE_1`) requires -- `HOME_ADVANTAGE_4` needs all four
groups to match at all, so it silently failed and fell through to `home1`,
which also failed because the character after `HOME ADVANTAGE=` is `[`, not
a digit. Every team row in this era carries only **two** method columns
(`ELO_CHESS`, `PURE POINTS` -- **measured**, column header text
`SCHEDL(RANK) VS top 10 | VS top 16 | ELO_CHESS    | PURE POINTS` on this
same capture), not the three (`GOLDEN_MEAN`, `PURE POINTS`, `ELO_SCORE`) the
post-Nov-2013 format carries, so the three brackets are `[RATING edge]
[ELO_CHESS edge] [PURE POINTS edge]` -- the same 2-method shape as the
existing single-value USA Today format, just published with one bracket per
value instead of one bare number.

**Measured**, walking every capture in the local snapshot whose
`captures_log.parquet` row had `era_format=="unknown"` (17 of 592 captures,
both domains): 16 of the 17 shared this exact 3-bracket shape, spanning
roughly **Nov 2011 - Sep 2013** on both `sagarin.com`- and
`usatoday.com`-domain captures (including a pre-Week-1 "NFL 2013 Starting
Ratings" snapshot, previously mischaracterized in section 5.1 as its own
distinct "preseason 3-bracket variant" -- it is the same transitional
format, not a separate one). The 17th (`usatoday/nfl11@20120109071948`,
"2012 JANUARY 8 SUNDAY - Wild Card Weekend") used a fourth, one-off layout
in the same window: **measured**, comma-separated and unbracketed --
`HOME EDGE=  3.04,  2.38,  2.74` -- with a different label (`HOME EDGE=`,
not `HOME ADVANTAGE=`) and the same RATING/ELO_CHESS/PURE-POINTS value order
(the page's own explanatory text, measured on that capture: `"There are now
THREE home edges listed for: RATING, ELO_CHESS, PREDICTOR(PURE POINTS)"`).
Neither format is a fetch problem, a redirect artifact, or a genuine
absence -- both are real Sagarin page layouts the parser simply never
matched; the team rows themselves already parsed fine in every one of these
17 captures (`parse_status="ok"`, 32 teams each), only the era-tolerant
home-advantage regex pair was incomplete.

### 9.2 Fix, additive

`scripts/ingest_sagarin_ratings.py`:

- Added `HOME_ADVANTAGE_3` (three bracketed values) and `HOME_EDGE_COMMA`
  (`HOME EDGE=  X, Y, Z`), checked only when the existing four-bracket regex
  fails to match (so there is no ordering ambiguity with the untouched
  `HOME_ADVANTAGE_4`/`HOME_ADVANTAGE_1` pair). Both set `home_edge_rating`
  from the first (RATING-method) value and tag `era_format=ERA_USATODAY` --
  reusing, not inventing, the convention section 5.1 already documented
  (`sagarin.com`-domain captures using the simpler 2-method layout already
  got tagged `era_format="usatoday"`) -- so the team-row method-column
  lookup correctly resolves to `elo_chess`/`pure_points` instead of the
  generic `method_0`/`method_1` fallback. The two per-method values
  (`ELO_CHESS`, `PURE POINTS`) are deliberately left out of
  `home_edge_methods` rather than written into the
  `home_edge_golden_mean`/`home_edge_elo_score` columns, which are a
  fixed-position mapping elsewhere in the script that assumes
  `GOLDEN_MEAN`/`PURE_POINTS`/`ELO_SCORE` order -- writing `ELO_CHESS`/`PURE
  POINTS` values into those slots would mislabel them for no benefit, since
  no downstream consumer (the frozen `sagarin_battery_*` predeclaration,
  section 8) uses anything but `home_edge_rating`.
- Added `enumerate_cached_captures()` and a `--reparse-cache-only` CLI flag:
  rebuilds `captures_log.parquet`/`index.parquet`/`asof_tuesday_view.parquet`
  by walking the already-cached HTML under `<snapshot>/pages/` and
  re-running the (fixed) parser, with **zero CDX queries and zero page
  fetches**. This exists specifically so a parser fix can be measured
  against the *same* capture set already on disk, rather than conflating
  "the parser got better" with "the archive got denser since 20260820" by
  re-enumerating CDX and picking up new captures Wayback has crawled since.
- `tests/test_sagarin_ingest.py` (new) + `tests/fixtures/sagarin/*.html`
  (new, 5 files): real, trimmed excerpts (header line + home-advantage/edge
  line + 3 team rows, boilerplate paragraph dropped) copied verbatim from
  genuine cached captures, one fixture per format handled -- the existing
  4-bracket and 1-bracket formats (regression guard) plus the three new
  cases (in-season 3-bracket, preseason-snapshot 3-bracket, comma format).
  **Measured**: `.\.tools\uv.exe run --no-sync pytest tests/test_sagarin_ingest.py
  -p no:cacheprovider --basetemp=<private temp dir>` -> **8 passed**.
  `.\.tools\uv.exe run --no-sync ruff format` /
  `.\.tools\uv.exe run --no-sync ruff check` on both changed files ->
  clean.

No new Wayback fetches were needed or made: every capture the fix touches
was already cached from the original 2026-08-20 session. The rebuild ran as
`.\.tools\uv.exe run --no-sync python scripts/ingest_sagarin_ratings.py --out
data/raw/sagarin --snapshot 20260820T112501Z --reparse-cache-only
--schedules-snapshot data/raw/20260824T115346Z/schedules.parquet`
(**measured** manifest after: `captures_attempted: 585, captures_fetch_ok:
585, captures_fetch_failed: 0, captures_parse_ok: 585, index_rows: 18473`).
One provenance note: `--reparse-cache-only` walks only HTML already on disk,
so the captures_log no longer carries a row for the 7 pre-existing fetch
*failures* section 5.4 documented (`nfl07@20080511204634` and similar curl-61
gzip / `Errno 22` cases) -- those were never cached to disk in the first
place and are unrelated to this fix; they are simply outside what
`--reparse-cache-only` can represent, not resolved or regressed by it. A
future CDX-enumerating run would still see and could still retry them.

### 9.3 Coverage, before -> after (measured)

Both tables below come from re-running this project's own coverage
functions (`attach_sagarin`/`build_close_population`/`build_open_population`
in `scripts/sagarin_divergence_battery.py`, called directly as a
measurement -- **not** `main()`, so no bootstrap ran and nothing was written
to `artifacts/` or the registry) against a pre-fix backup of
`captures_log.parquet`/`index.parquet`/`asof_tuesday_view.parquet` (copied
before the rebuild) versus the rebuilt, fixed snapshot, both joined to the
same current schedule snapshot (`data/raw/20260824T115346Z/schedules.parquet`).
Every count below is grouped from the *actual* screen population
(`close_pop`/`open_pop`, i.e. after `has_sagarin`, non-null `home_cover`,
and non-null `divergence`/`sagarin_side_cover` -- pushes and the rare
exact-zero-divergence game excluded, same as the frozen battery's own
population), not the looser raw join-coverage table `attach_sagarin` prints
on its own, so every per-season count here reconciles exactly to the
reported population totals.

**Close-grade** (REG 2010-2025, screen population):

| season | REG games | usable BEFORE | usable AFTER | coverage BEFORE | coverage AFTER |
|---:|---:|---:|---:|---:|---:|
| 2010 | 256 | 235 | 235 | 91.8% | 91.8% |
| 2011 | 256 | 229 | 229 | 89.5% | 89.5% |
| 2012 | 256 | **0** | **121** | **0.0%** | **47.3%** |
| 2013 | 256 | 77 | 219 | 30.1% | 85.5% |
| 2014 | 256 | 219 | 219 | 85.5% | 85.5% |
| 2015 | 256 | 231 | 231 | 90.2% | 90.2% |
| 2016 | 256 | 235 | 235 | 91.8% | 91.8% |
| 2017 | 256 | 248 | 248 | 96.9% | 96.9% |
| 2018 | 256 | 200 | 200 | 78.1% | 78.1% |
| 2019 | 256 | 184 | 184 | 71.9% | 71.9% |
| 2020 | 256 | 256 | 256 | 100.0% | 100.0% |
| 2021 | 272 | 76 | 76 | 27.9% | 27.9% |
| 2022 | 271 | 104 | 104 | 38.4% | 38.4% |
| 2023 | 272 | 228 | 228 | 83.8% | 83.8% |
| 2024 | 272 | 221 | 221 | 81.2% | 81.2% |
| 2025 | 272 | 223 | 223 | 82.0% | 82.0% |
| **TOTAL** | | **2,966** | **3,229** | | |

**Total close-grade population: 2,966 -> 3,229 games (+263, +8.9%)**, entirely
from 2012 (+121 games) and 2013 (+142 games; 121+142=263, reconciling
exactly with the total delta); every other season is byte-for-byte
unchanged, confirming the fix is additive and did not disturb already-working
captures (also **measured** directly: of 592 pre-fix captures, exactly 17
flipped `home_edge_rating` from null to non-null, 0 regressed from non-null
to null, and 0 already-non-null values changed). These per-season figures
differ from section 8.1's original table (which read 31.2%/43.0%/etc. for
2010-2012) because that table was built from an earlier schedule snapshot
and a coarser join-coverage count; both before/after columns here use the
*same* current schedule snapshot and the *same* strict screen-population
filter, so the comparison is apples-to-apples even though neither column
individually matches section 8.1's numbers verbatim. 2010-2011 and 2021-2022
remain genuinely thin (28-92%) -- an archive-density gap (few distinct
Wayback captures that season), not a parsing miss; 2012 itself only reaches
47.3%, not 100%, for the same reason (**measured**: only 3 distinct
captures carry `season==2012` in the header text at all -- weeks 1-7 and
9-16 simply have no Sagarin snapshot published before their Tuesday cutoff,
regardless of parser correctness).

**Opener-grade** (the 1,537-game paired `tue_open`+close archive,
2020-2025, `_opener_graded_features`), screen population:

| season | usable BEFORE | usable AFTER |
|---:|---:|---:|
| 2020 | 220 | 220 |
| 2021 | 62 | 62 |
| 2022 | 96 | 96 |
| 2023 | 234 | 234 |
| 2024 | 221 | 221 |
| 2025 | 220 | 220 |
| **TOTAL** | **1,053** | **1,053** |

**Opener-grade population: 1,053 -> 1,053 games, unchanged in every season.**
Every capture this fix touches falls in 2011-2013; the opener archive only
covers 2020-2025, so the two windows never intersect. Read plainly: this fix
improves the CLOSE-grade join meaningfully (+263 games, entirely in
2012-2013) but has **zero effect** on the pool-relevant OPENER-grade
population or on any of the four
`*_open`/`*_2010_2016`/`*_2017_2025`/`model_agreement_close` registry cells
that read from 2018-2025 data -- only `sagarin_battery_large_divergence_close`
and `sagarin_battery_top_decile_close` touch seasons this fix changed at
all, and even those are frozen results, not live queries (next section).

### 9.4 What this fix does NOT do

Per the binding data-fix-vs-experiment distinction in this task and in
`AGENTS.md`: **section 8.3's seven `sagarin_battery_*` weak-signal registry
entries were not re-scored, re-run, or touched.** They remain exactly the
values recorded from `artifacts/sagarin_divergence_battery/20260821T170345Z/results.json`,
computed on the pre-fix coverage. Re-running
`scripts/sagarin_divergence_battery.py` against the now-fixed snapshot would
change the close-grade cells' input population (2,966 -> 3,229 games, with
2012 newly contributing games it never could before) and would be a **new
look at the same outcome data** those cells already scored once -- exactly
what this project's rotation-registry discipline exists to prevent. A
legitimate re-measure would need, before computing anything: (1) a fresh
predeclaration (which cells, which threshold, which era split -- likely
identical to section 6/8's, but stated fresh, not reused after seeing this
section's numbers), (2) registration as its own rotation-registry look
(`nfl-ats rotation record-look`) rather than an in-place edit of the
existing `sagarin_battery_*` entries, and (3) its own window, since the
input population materially changed in exactly the two seasons
(2012-2013) most different in era from the rest of the close-grade sample.
This section reports the coverage delta only, as scoped.
