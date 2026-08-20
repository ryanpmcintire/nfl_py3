# Public betting percentage archive: ingestion + coverage report

Ingestion of `docs/data_source_scout_v3.md`'s rank-2 candidate
("actionnetwork.com free public betting-percentage archive" -- read that
document's section 2 for the mechanism case: bet%/money% splits by side are
the direct "fade/follow the public" signal, v2's #2-ranked mechanism overall).
Clones the proven sitemap/CDX-then-per-page-fetch structure documented in
`docs/pfr_transactions_sourcing.md` and `docs/injury_news_sourcing.md`
(rate limiter, timestamped-snapshot-directory convention, manifest.json),
adapted for the Wayback Machine's CDX API instead of a live sitemap, since
these are historical percentages that only exist as archived captures of a
page that has never itself published a public API.
Scope of this document and this session: **ingestion + coverage report
only** -- no experiments run, no registry writes, no `src/nfl_ats` changes,
no challenger wiring. Every claim below is tagged **measured** (fetched or
run this session, exact command/URL/path given), **read** (a file opened
this session), or **inferred** (reasoning, not evidence); there are no
unverified **reported** claims carried over from the scout doc without
independent re-verification -- in fact section 4 below **corrects** one of
the scout doc's claims after re-fetching the exact snapshot it cited.

Code: `scripts/ingest_public_betting.py` (new; does not touch any other
ingestion script, `experiment_runner.py`, `margin.py`, `public_board.py`,
`cli.py`, or anything under `src/nfl_ats`).

---

## 1. Sources, access, and methodology

**Measured** this session, both origin sites' `robots.txt`:
- `https://www.actionnetwork.com/robots.txt`: no `Crawl-delay`; `Disallow`
  covers `/health`, `/redirect`, `/fb-*`, `/*/embed`, `/content/`, and a set
  of tracking query-string patterns -- nothing covering `/nfl/public-betting`.
- `https://www.covers.com/robots.txt`: no `Crawl-delay`; `Disallow` covers
  forum/account/redirect utility paths -- nothing covering `/picks/nfl`.

Neither site is actually fetched directly by this script, though: every
request goes through `web.archive.org`'s CDX API and raw-capture (`id_`)
endpoint, never the origin sites. The operative rate limit is therefore the
task's own ~1 req/sec instruction against `web.archive.org` itself
(`RateLimiter(WAYBACK_DELAY_SECONDS=1.0)` in the script), not either site's
robots policy -- the robots checks above are recorded for provenance, not
because they gate anything this script does.

**CDX methodology, measured**: `http://web.archive.org/cdx/search/cdx
?url=<path>&output=json&from=<Y1>&to=<Y2>&filter=statuscode:200
&collapse=timestamp:8`. `collapse=timestamp:8` day-collapses to the first
capture per calendar day, matching the scout doc's own methodology so the
counts stay directly comparable. One off-by-one bug was caught and fixed
during this session: the CDX `to=` year parameter is **inclusive of the
entire year**, not just its first day (measured: `to=2020` on a
`from=2019` query returned captures dated through 2020-12-06) -- an
initial `end_year + 1` in the script silently pulled one extra year into
every range until corrected.

**Two template eras found in actionnetwork.com's `__NEXT_DATA__` payload**,
both parsed by the script (era detection is automatic per-snapshot, not a
hardcoded date cutoff, so the exact boundary date does not need to be known
for the parser to work correctly):

- **era1** (`pageProps.initialState.gamesReducer.games`, a dict keyed by
  the site's internal game id): 2018 through Oct/Nov 2022. **Measured**:
  boundary sits between 2022-10-07 (still era1) and 2022-11-04 (already
  era2) -- not pinned to a single day, not needed for the parser. Each
  game's Consensus-book, game-period odds record carries
  `{market}_{side}_public` integer fields (`spread_home_public`,
  `ml_away_public`, `total_over_public`, etc.). **Only bet (ticket)
  percentages exist in this era's schema -- there is no money-percentage
  field at all**, confirmed structurally (not just absent in one sample) by
  inspecting every book's `game`-period odds object across 2019/2020/2022
  snapshots: the money-side keys simply do not exist in the JSON, this is
  not a parsing gap.
- **era2** (`pageProps.scoreboardResponse.games`, a list): ~Nov 2022
  onward. Each game carries an explicit `week` field (era1 has none) and a
  `markets.{book_id}.event.{market_type}` structure (`spread`, `moneyline`,
  `total`) where each outcome carries **both**
  `bet_info.tickets.percent` (bet%) and `bet_info.money.percent` (money%).
  Money% genuinely first becomes available in this era -- see section 4 for
  the measured window where era2 nonetheless returned no data at all.

**The "Consensus" book, resolved dynamically, not hardcoded**: both eras
expose a same-shaped book registry (`booksReducer.supportedBooks` in era1,
`allBooks` in era2) where one entry has `source_name: "consensus"` /
`display_name: "Consensus"`. The script looks this id up per-snapshot
(`_find_consensus_book_id_era1` / `_find_consensus_book_id_era2`) rather
than assuming a fixed id, though **measured**: it has been book id `15` in
every one of the ~150 snapshots parsed this session.

**A brace-matching bug found and fixed this session**: the 2018-era page
uses an older inline `<script>__NEXT_DATA__ = {...}</script>` form (no
`id="__NEXT_DATA__"` wrapper), immediately followed by a second, unrelated
`<script>` block with no whitespace separating the two. A first-draft
non-greedy regex (`__NEXT_DATA__\s*=\s*(\{.*?\});?\s*</script>`) under-matched:
it stopped at the first `}` that happened to precede *any* later
`</script>` tag -- including one inside the second, unrelated script block
-- truncating the real JSON and raising `json.JSONDecodeError: Extra data`
(**measured** on the 2019-01-03 snapshot, and this affected all 5 of the
pre-August-2019 captures that predate the `id="__NEXT_DATA__"` wrapper).
Fixed with a string/escape-aware balanced-brace scanner
(`_extract_balanced_json`) instead of a regex for the closing boundary.
After the fix, all 5 of those early-2019 captures parse cleanly (see
section 2's table -- zero `no_next_data` failures in that window post-fix).

**A genuine Wayback-side truncation, found and left unfixed because it is
unfixable from this side**: 5 of 153 captures (3.3%), scattered from
2021-01 to 2022-08, are all ~1,048,600-1,048,700 bytes -- clustered right at
1 MiB (1,048,576 bytes) -- and their `__NEXT_DATA__` JSON never closes
(the balanced-brace scanner runs off the end of the file with `depth > 0`).
**Measured**: `wc -c` on all 5 raw HTML files on disk
(`data/raw/public_betting/20260820T111148Z/actionnetwork/raw_html/
2021*.html`, `2022*.html`) confirms the ~1 MiB clustering directly. This
reads as Wayback's own capture pipeline truncating the response at a ~1 MiB
buffer limit for whatever tool captured these five snapshots -- the
archived content itself is incomplete, not something a better parser on
this end could recover.

**One outlier, not built into either era's parser**: the single oldest
capture in the whole archive, 2018-09-02, has a `props` object with no
`pageProps` wrapper at all (`props.title`, `props.description`,
`props.league` sit directly under `props`, **measured** by inspecting the
raw HTML) -- an even earlier page-data shape than era1. It is exactly one
data point (0.65% of the corpus) predating the archive's own real start
(the very next capture, 2018-11-02, is already era1-shaped); not worth a
third parser for one row, flagged honestly as `unrecognized_shape` rather
than silently dropped or miscounted as a "failure" of the real parsers.

---

## 2. Inventory built and per-era/per-year parse success rates

**Measured**, `data/raw/public_betting/20260820T111148Z/actionnetwork/
manifest.json` (gitignored under the repository's existing `data/raw/**`
rule, not committed):

| Metric | Value |
|---|---:|
| CDX captures found (day-collapsed, 2018-2026) | 153 |
| Parsed successfully (either era) | 147 (96.1%) |
| `era1_initial_state` successes | 59 |
| `era2_scoreboard_response` successes | 88 |
| `no_next_data` failures (Wayback ~1 MiB truncation, section 1) | 5 |
| `unrecognized_shape` failures (2018-09-02 outlier, section 1) | 1 |
| Total per-game rows parsed | 1,658 |
| Rows with >=1 non-null bet% or money% value | 1,348 (81.3% raw -- see section 4 for why the raw figure understates true availability) |

Per-year capture and parse detail (**measured**, from
`actionnetwork/cdx_index.parquet` and `actionnetwork/index.parquet`):

| Year | CDX captures | Game rows parsed | Rows with any public data | Row-level yield |
|---|---:|---:|---:|---:|
| 2018 | 2 | 13 | 13 | 100.0% |
| 2019 | 12 | 130 | 125 | 96.2% |
| 2020 | 19 | 191 | 136 | 71.2% |
| 2021 | 19 | 207 | 207 | 100.0% |
| 2022 | 23 | 297 | 151 | 50.8% |
| 2023 | 21 | 166 | 62 | 37.3% |
| 2024 | 36 | 484 | 484 | 100.0% |
| 2025 | 16 | 120 | 120 | 100.0% |
| 2026 (through Aug) | 5 | 50 | 50 | 100.0% |
| **Total** | **153** | **1,658** | **1,348** | **81.3%** |

This is close to the scout doc's **reported** "~15-45 captures/season" claim
(now **measured** directly per year above: 12-36/season in the years with a
full season on the books, 2019-2025) and, with the year extended through
2026-08, adds the current in-progress season.

The 2020 shortfall (71.2%) is explained by section 1's methodology, not a
parser gap: 3 of 2020's 19 captures (2020-05-31, 2020-06-19, 2020-06-26)
are legitimate off-season captures with 16 games listed but zero lines
posted yet (`has_any_public_data=False` on every game -- this is the
pre-training-camp calendar, not missing data), and 3 more
(2020-09-29/30, 2020-10-01) returned a page with **zero games listed at
all** (a real but unexplained page-state gap this session did not chase
further, 3 consecutive day-collapsed captures in a normal in-season week).
The 2022/2023 shortfall is a single, precisely bounded, much larger gap --
covered in its own section next because it deserves one.

---

## 3. covers.com/picks/nfl: measured dead end, corrects the scout doc

**This is a correction to `docs/data_source_scout_v3.md` section 2**, which
reported (unverified, from a search/fetch snippet) that
`covers.com/picks/nfl` has "cleaner, denser data ... real moneyline
consensus % confirmed in a 2023-08-19 snapshot." **Measured this session**:
the exact 2023-08-19 snapshot cited (`http://web.archive.org/web/
20230819011901id_/https://www.covers.com/picks/nfl`) was re-fetched and
inspected directly. It is Covers' community/handicapper **picks** page --
individual authors' pick strings and win-loss records (e.g. a handicapper's
"100%" season record) -- not a sportsbook bet%/money% consensus page. Its
own on-page copy confirms this: the page links out to a *different* URL,
`contests.covers.com/consensus/topconsensus/nfl/overall`, for actual
"consensus" data.

That linked URL was checked too, back to a 2016 Wayback archive (predating
even actionnetwork's 2018 start, so a real find if it had panned out).
**Measured**: it is also a dead end -- its own inline `<script>` block
literally calls `consensusEventHandlers()` and a jQuery `$.ajax(...)` fetch
for a promo widget, i.e. its real content (the percentage data the page's
own meta description promises: "see what side the public is on!") is
populated by **client-side AJAX after page load**, and the archived static
HTML never ran that JavaScript.

A body-only regex percent-scan (excluding `<head>`/`<style>` CSS, where
Bootstrap grid classes like `.col-xs-8{width:66.66667%}` produce false
positives) was run against 5 real snapshots this session, spanning both
URLs and both eras of Covers' own site redesigns:

| Snapshot | URL | `<body>`-only `%` matches | What they actually are |
|---|---|---:|---|
| 2016-10-21 | `contests.covers.com/.../Overall` | 2 | unrelated page furniture, not a per-game consensus table |
| 2023-08-19 | `covers.com/picks/nfl` (scout doc's own cited date) | 4 | handicapper "100%" win-rate badges |
| 2023-10-23 | `covers.com/picks/nfl` | 4 | same |
| 2024-08-26 | `covers.com/picks/nfl` | 13 | same, more handicappers |
| 2025-10-08 | `covers.com/picks/nfl` | 212 | CSS `@keyframes ... background-position: 99% 0` animation rules inside a `<style>` tag placed mid-`<body>` by the page's component framework -- not data |
| 2026-07-24 | `covers.com/picks/nfl` | 32 | same CSS-animation artifact |

Zero genuine per-game bet%/money% percentages were found in any of the 5
samples across both URLs and 10 years of site redesigns. This is not a
"maybe try harder" gap -- the mechanism (client-side AJAX fetched after
Wayback's crawler already saved the page) is structural and applies to
every historical capture of either URL, not just the ones sampled.

**Coverage still ingested for future reference**: `scripts/
ingest_public_betting.py --covers-sample-n N` fetches the CDX inventory
unconditionally (**measured**: 331 day-collapsed captures, 2023-2026 --
`covers.com/picks/nfl` genuinely starts later and denser than
actionnetwork, exactly as the scout doc said, just without the target
data) and a small verification sample (`N=8` this session, one per roughly
year/era) of raw HTML, so a future session does not have to re-derive this
finding from scratch. `data/raw/public_betting/20260820T111148Z/covers/
verification_summary.json` records the full result. **No bulk covers.com
fetch was run** -- sample-verifying a confirmed-negative source further
would have spent fetch budget without adding information; see section 7 for
what a follow-up would need to do differently (find the actual AJAX
endpoint, not another HTML pull).

---

## 4. The Oct 2022 - Oct 2023 empty-markets gap (era2's real cost)

**Measured**, precisely bounded from `actionnetwork/index.parquet`: of
era2's 88 successfully-parsed captures, **27 are contiguous and 100%
empty** -- every game on the page has a `markets` dict entirely stripped of
percentages -- running from **2022-10-16 through 2023-10-16**, almost
exactly one calendar year. Every era2 capture before this window is era1
(so N/A), and **every one of the 61 era2 captures after 2023-11-07 has real
data** (0 empty captures from Nov 2023 through the most recent capture,
2026-08-16). The gap is not a template-detection problem (the page shape
correctly parses as era2 throughout) and not a game-listing problem (games,
teams, kickoffs, and status are all populated normally throughout the gap)
-- specifically and only the `markets` object per game is empty. This
session did not determine the site-side cause (a vendor/consensus-provider
change, an A/B test, a paywall change) and does not speculate further than
what was measured.

**Practical effect on the row-level yield table in section 2**: the 2022
and 2023 rows' low yield (50.8%, 37.3%) is almost entirely this one
14-month window, not a spread of small problems. A user of this archive for
2022-10 through 2023-10 games should expect **no** actionnetwork public
betting data for that stretch specifically, full stop -- not a lower-quality
reading, an absent one.

---

## 5. Coverage join: matching parsed games to the schedule

`build_coverage_report()` in the script joins `actionnetwork/index.parquet`
against `data/processed/game_features.parquet` (**read**, the repo's
already-ingested nflverse schedule, 2009-2026, `game_id` = `{season}_
{week:02d}_{away}_{home}`) on normalized team abbreviation pairs
(`TEAM_ALIASES` extends `src/nfl_ats/constants.py`'s
`TEAM_ABBREVIATION_ALIASES` -- OAK->LV, SD->LAC, STL->LA, plus LAR->LA,
JAC->JAX, WSH->WAS, all **measured** as real alternate spellings
actionnetwork used across its own eras/relocations), filtered to
`game_type == "REG"` and a kickoff within 72 hours of the site's own
`start_time` for that game (handles minor postponements/flex moves loosely
without needing exact-timestamp matching).

**Measured**, `data/raw/public_betting/20260820T111148Z/
coverage_report.json`:

- 1,658 total parsed rows; 800 (48.3%) matched to a REG-season schedule
  game within 72 hours. The other 858 are, by design, excluded rather than
  "lost": preseason games (August "scheduled" rows, before Week 1 --
  actionnetwork's page shows the upcoming preseason slate all August) and
  stale playoff reruns (the same just-finished Wild Card matchup re-served
  by the site as "complete" for weeks into the following offseason --
  **measured** directly in the 2019 Jan-Apr captures, section 2's table) are
  real content the site served, just not REG-season games this project's
  pool cares about.

Own-week Tuesday-noon-ET cutoff, computed with the **exact same
convention** already used and tested in
`scripts/injury_tuesday_cutoff_experiment.py`'s
`team_week_tuesday_noon` (**read**, that file, lines 119-141): `(kickoff
weekday - 1) % 7` days back from kickoff in `US/Eastern`, then +12 hours,
converted to UTC. Per project convention (`MEMORY: "Picks lock at
kickoff"`), the pool's **line** freezes Tuesday but **picks stay editable
to kickoff**, so a reading captured after that Tuesday is still playable,
not discarded -- both counts are reported, neither is treated as waste:

| Season | Distinct captures | REG games in schedule | Games with >=1 pregame reading | ...before Tuesday noon ET | ...after Tuesday noon ET only |
|---|---:|---:|---:|---:|---:|
| 2018 | 1 | 256 | 3 | 0 | 3 |
| 2019 | 5 | 256 | 31 | 0 | 31 |
| 2020 | 12 | 256 | 45 | 16 | 29 |
| 2021 | 14 | 272 | 81 | 10 | 71 |
| 2022 | 19 | 271 | 92 | 8 | 84 |
| 2023 | 9 | 272 | 51 | 5 | 46 |
| 2024 | 27 | 272 | 77 | 12 | 65 |
| 2025 | 6 | 272 | 36 | 12 | 24 |
| 2026 (through Aug, preseason only so far) | 2 | 272 | 12 | 12 | 0 |

Two honest reads of this table:

1. **Coverage is a "known as of the nearest prior capture" feature, not a
   per-game-guaranteed one**, exactly as the scout doc predicted: even in
   the best-covered season (2022, 19 captures), only 92 of 271 REG games
   (34%) have any captured reading at all. A denser prospective capture job
   (the scout doc's part (b), a weekly live-capture cron, not built this
   session -- out of scope, see section 7) would be needed for anything
   beyond a backfill-quality historical feature.
2. **Most captured readings land after that week's Tuesday-noon line
   freeze, not before** (e.g. 2022: 84 of 92 games' only reading is
   post-Tuesday) -- but per this project's own binding convention these are
   still real, playable readings, since picks stay open to kickoff. A
   downstream feature built on this archive should use whichever reading is
   most recent as of the actual decision time, not silently restrict itself
   to pre-Tuesday readings only.

---

## 6. Snapshot layout

```
data/raw/public_betting/20260820T111148Z/
  actionnetwork/
    cdx_index.parquet          153 Wayback CDX rows (day-collapsed, 2018-2026)
    yearly/<YYYY>.parquet      parsed rows, keyed by capture year
    index.parquet              concatenation of all years, 1,658 rows
    raw_html/<timestamp>.html  147 raw fetched pages (audit/re-parse without refetch)
    manifest.json              run metadata + era/failure counts
  covers/
    cdx_index.parquet          331 Wayback CDX rows (day-collapsed, 2023-2026)
    sample_html/<timestamp>.html  8 verification-sample raw pages
    verification_summary.json  the negative finding, section 3
  index.parquet                == actionnetwork/index.parquet (covers contributes 0 rows)
  coverage_report.json         section 5's per-season table, machine-readable
  manifest.json                top-level run metadata
```

`manifest.json` is nested one level below `public_betting/`, matching this
repo's `data/raw/<source>/<UTC timestamp>/` convention
(`nfl_ats.snapshots.latest_snapshot()` treats any directory directly under
`data/raw/` with a `manifest.json` as a candidate schedules snapshot --
`scripts/ingest_injury_news.py`'s docstring documents the same collision
this avoids).

Per-row schema of `actionnetwork/index.parquet` (wide format, one row per
game per capture): `capture_ts`, `source`, `era`, `site_game_id`, `season`,
`week` (era2 only, null in era1), `status`, `away_team_raw`/`home_team_raw`
(as printed by the site), `away_team`/`home_team` (normalized to nflverse
abbreviations), `start_time_utc`, `book_id`, then
`spread_home_bet_pct`/`spread_away_bet_pct`/`spread_home_money_pct`/
`spread_away_money_pct`, the same four for `ml_*`, and
`total_over_bet_pct`/`total_under_bet_pct`/`total_over_money_pct`/
`total_under_money_pct`, plus `has_any_public_data` (bool).

---

## 7. Provenance summary and follow-up work

**Measured this session**: both origin `robots.txt` fetches, the CDX
methodology and the `to=` off-by-one fix, both `__NEXT_DATA__` template
eras and the consensus-book-id resolution across ~150 raw snapshots, the
brace-matching parser bug and its fix (verified against the 5 early-2019
captures it affected), the ~1 MiB Wayback-truncation finding (byte-length
measurement on all 5 affected files), the 2018-09-02 `unrecognized_shape`
outlier, the full 153-capture actionnetwork backfill and its per-year/
per-era yield table, the precisely-bounded 2022-10-16 to 2023-10-16
empty-markets gap, the covers.com re-fetch of the scout doc's own cited
2023-08-19 snapshot plus 4 more years/eras and the linked
contests.covers.com 2016 archive, the 331-capture covers.com CDX inventory,
the schedule join and its Tuesday-noon-ET table.

**Read this session**: `docs/data_source_scout_v3.md` section 2,
`docs/pfr_transactions_sourcing.md` and `docs/injury_news_sourcing.md` (style
and pattern references), `scripts/ingest_injury_news.py` and `scripts/
ingest_transaction_news.py` in full (the cloned template), `scripts/
injury_tuesday_cutoff_experiment.py` lines 119-141 (the exact Tuesday-noon-ET
convention reused), `src/nfl_ats/constants.py`'s `TEAM_ABBREVIATION_ALIASES`,
`data/processed/game_features.parquet`'s schema, `.gitignore` (confirmed
`data/raw/**` covers this snapshot).

**Inferred**: the "private research caching, never republish" policy stance
for both archives, by analogy to this project's existing CFBD/PFT/PFR
precedent (`docs/data_feasibility.md` License item 6) -- neither
actionnetwork's nor Covers' terms of use were independently reviewed this
session; the hypothesis that the 3 zero-games captures in Sept/Oct 2020
reflect a real but unexplained page-state gap rather than a fetch artifact
-- not chased further, reasoning only, flagged as such in section 2.

Nothing in this document is a **reported** (unverified subagent/search)
claim carried over without independent re-verification -- every number was
fetched or computed directly this session, and the one scout-doc claim that
did not survive re-verification (covers.com's percentage data) is corrected
in section 3 rather than silently repeated.

### What a follow-up session should fetch

1. **A live, prospective weekly capture job** (the scout doc's part (b), not
   built this session): the Wayback backfill is frozen history by
   construction. A small Task-Scheduler-style job hitting
   `actionnetwork.com/nfl/public-betting` directly (not through Wayback)
   once or twice a week, in the same `scripts/odds_capture.ps1` pattern
   already used for market lines, is the only way this source stays useful
   once the project needs the *current* week rather than a backtest.
2. **Denser era1 backfill**: era1 (2018 - Oct 2022) only had 2-23
   captures/season in this session's day-collapsed pull; re-running the CDX
   query without `collapse=timestamp:8` (or with a finer collapse) would
   recover same-day multiple-capture instances if any exist, though section
   5's finding (most seasons cap out well under 100 REG games with any
   reading at all even at current density) suggests density, not
   collapsing, is the binding constraint.
3. **If covers.com is still wanted**: do not re-pull `covers.com/picks/nfl`
   or `contests.covers.com/.../Overall` via Wayback again -- section 3
   settled that path. A real path would need the actual JSON API endpoint
   Covers' own client-side JavaScript calls (found via a live browser
   network trace, not a static HTML pull), and then a check of whether
   *that* endpoint has ever itself been crawled by Wayback.
4. **The 3.3% Wayback-truncation gap** (section 1) is not recoverable from
   this side; if those 5 specific weeks matter to a future analysis, the
   only fix is checking whether a second, non-truncated Wayback capture
   exists nearby in time (this session's day-collapse would have hidden a
   same-day second capture -- an un-collapsed CDX query for just those 5
   dates is a cheap, targeted follow-up).

### Predeclared next-step experiment (NOT run this session)

Per this task's scope (ingestion + coverage report only) and per AGENTS.md's
binding closing-grounds taxonomy, restated verbatim as required for any
downstream subagent or scoring pass:

> An interval or CI that contains zero is NEVER grounds to reject, fail, or
> close an experiment. Only two grounds ever close a line of work: (1)
> refuted mechanism -- a RESOLVED wrong sign (whole interval on the wrong
> side of zero) or zero split-half reliability; (2) bounded by a positive
> control proven able to detect an effect that size. Everything else is
> `unresolved_below_power`: record it with `nfl-ats weak-signals record`,
> report `probability_positive`, never the binary "contains zero".

Two directly testable questions, predeclared before running, mirroring the
two-question style of `docs/injury_news_sourcing.md` section 5 and
`docs/pfr_transactions_sourcing.md` section 6:

1. **Fade-the-public screen**: for REG-season games with a matched
   pregame reading (section 5's 800-row join), does the side with the
   LOWER `spread_{side}_bet_pct` (the side the public is NOT on) cover at a
   rate different from 50% / different from this project's existing
   52.05%-vs-close baseline? This is the textbook "fade the public"
   mechanism the scout doc named as v2's #2-ranked idea. Needs the
   opener-vs-close framing this project already uses for every other
   candidate (`docs/pool_edge_plan.md`'s "grade at the opener" convention,
   `AGENTS.md`'s "grade the decision at the opener" section) -- a reading
   captured well before Tuesday should be compared against the Tuesday
   opener, not the close, to avoid grading a market-efficient number
   against itself.
2. **Bet%/money% divergence ("smart money") screen**: for era2 rows only
   (money% only exists there, section 1), does a large gap between
   `spread_{side}_bet_pct` and `spread_{side}_money_pct` -- many small
   tickets on one side, more dollars on the other, the standard "sharp
   money" signature -- predict the closing-line move direction, and does
   that predicted-move side cover at a different rate than the
   `bet_info.tickets`-only signal from question 1? This tests whether
   money% adds anything beyond bet% specifically, which is exactly the
   reason this session parsed both fields separately rather than only the
   simpler bet% that era1 alone would have provided.

Neither question is run here. Given section 5's coverage ceiling (at most
~34% of REG games in the best-covered season have any captured reading),
both screens should be predeclared as exploratory/backfill-quality from the
start, not treated as a dense weekly feature until the prospective capture
job in section 7's item 1 exists. Whatever `probability_positive` either
analysis returns, an interval crossing zero is not grounds to close the
question; any resulting accuracy/coverage measurement should be recorded
through `nfl-ats weak-signals record` with the appropriate classification
(most likely `unresolved_below_power` unless one of the two admissible
closing grounds is cleared), never asserted as settled in prose.

No experiments were run, no `registry/` files were written, no files under
`src/nfl_ats` were touched, and no challenger wiring was added, per this
task's scope.

---

## 8. The predeclared screen, run (2026-08-20, follow-up session)

**Measured this session.** Predeclaration frozen first, in
`docs/public_betting_battery_predeclaration.md`, before
`scripts/public_betting_battery_screen.py` computed anything (that script
also does not touch `src/nfl_ats` or the rotation registry -- a mined,
exploratory battery, not a confirmation look, per this document's own
section 7 framing). Output: `artifacts/public_betting_battery/
20260820T113846Z/` (local, gitignored). All five cells were recorded via
`nfl-ats weak-signals record` and confirmed present in
`nfl-ats weak-signals status` (registry at 306 total signals after this
session's five writes, no parallel-writer race observed).

**Population**: 800 archive rows matched to a REG-season schedule game
within 72h (section 5's figure, reproduced identically). Collapsing to one
row per game -- the single latest capture strictly before kickoff -- gives
**416 games**, 2018-2025, with a resolvable close line and result.

**Staleness distribution** (hours from that latest pregame capture to
kickoff, n=416 games): min 0.14h, p10 3.6h, p25 12.3h, **median 45.2h**,
p75 76.3h, p90 127.5h, max 2,824h (a captured reading from many weeks
before its game, still the latest one available pregame), mean 156.0h.
Half of all matched games' most-recent pregame reading is more than
~1.9 days stale relative to kickoff -- consistent with section 5's own
finding that most captures land well after that week's Tuesday-noon line
freeze; per project convention picks stay playable to kickoff regardless.

**Cell results** (week-blocked bootstrap primary, 20,000 resamples, seed
20260818; season-blocked reported alongside; `P+` = `probability_positive`,
never reported as a bare "contains zero"; effect points are forced-pick
accuracy minus 50%, in percentage points):

| Cell | n | Accuracy | Effect (pts) | Week 95% CI | Week P+ | Season P+ |
|---|---:|---:|---:|---|---:|---:|
| A.1 fade-heavy-public, close, 2018-2025 | 91 | 46.15% | -3.85 | [-12.26, +2.25] | 0.100 | 0.203 |
| A.2 fade-heavy-public, tue_open, 2020-2025 | 80 | 47.50% | -2.50 | [-10.00, +4.44] | 0.209 | 0.278 |
| B.1 sharp-divergence (era2 only), close | 62 | 46.77% | -3.23 | [-18.29, +9.26] | 0.260 | 0.303 |
| C.1 model-interaction, public heavy against pick, close | 47 | 46.81% | -3.19 | [-13.64, +6.76] | 0.214 | 0.288 |
| C.2 model-interaction, against-minus-with diff | 91 (47 vs 44) | -- | -7.74 | [-25.93, +5.32] | 0.123 | 0.231 |

Every cell's interval crosses zero in both blocks -- per AGENTS.md's binding
rule that is the EXPECTED outcome at this project's ~2-point evaluator
resolution and is explicitly NOT grounds to reject any of these five
mechanisms. None meets either admissible closing ground (no cell has both
its week- and season-blocked interval entirely below zero, so
`wrong_sign_resolved` does not apply to any of them; none has a positive
control, so `bounded_by_control` applies to none either). All five are
therefore correctly recorded as `unresolved_below_power`, kept rather than
discarded, per the taxonomy `docs/public_betting_battery_predeclaration.md`
restates verbatim.

**What the point estimates say, stated plainly and separately from the
verdict above**: all five point estimates lean negative (fading the heavy
public side, and following the sharp-money side, both underperform 50% on
this sample; the model does not win more when the public is heavy against
its pick -- if anything C.2's point estimate leans the other way, model
accuracy trending *lower*, not higher, in the against subset). Every `P+`
sits below 0.5 (0.10-0.26 at the week block), i.e. the resamples lean
negative on all five cells, though none is anywhere near resolved at n=47-91
games on a sparse, mined archive. This is a probability-positive read, not
a rejection -- exactly the distinction AGENTS.md's labeling rule requires.

## 9. Prospective live capture (2026-08-20, same follow-up session)

Item 1 of section 7's follow-up list, built this session: a live,
prospective capture path so this source stays useful for the CURRENT week,
not just backfilled history.

**Robots.txt**: not re-fetched. **Read** this session, section 1 above:
"`https://www.actionnetwork.com/robots.txt`: no `Crawl-delay`; `Disallow`
covers `/health`, `/redirect`, `/fb-*`, `/*/embed`, `/content/`, and a set
of tracking query-string patterns -- nothing covering
`/nfl/public-betting`" -- the exact path this capture fetches, already
measured and documented in this same file, so a second robots.txt fetch
would be redundant.

**Built**: `scripts/public_betting_live_capture.py` (Python entry: one GET
against `https://www.actionnetwork.com/nfl/public-betting`, parsed with
`extract_next_data`/`parse_actionnetwork_snapshot` imported directly from
`scripts/ingest_public_betting.py` -- same-directory import, not a
duplicate of the brace-matching `__NEXT_DATA__` parser or the era1/era2
dispatch) and `scripts/public_betting_capture.ps1` (Task Scheduler
wrapper, same error-handling shape as `scripts/odds_capture.ps1`: stderr to
its own temp file rather than `2>&1`, `$LASTEXITCODE` as the only trusted
native-exe success signal, one append-only log line per run). Writes to
`data/raw/public_betting_live/<UTC timestamp>/` (`raw_html/`,
`index.parquet`, `manifest.json`), gitignored, same convention as the
backfill archive.

**Measured, run twice this session** (both direct hits against the live
site, not Wayback): `20260820T114125Z` and `20260820T114157Z`, both HTTP
200, both parsed as `era2_scoreboard_response`, 16 game rows each (2026
Week 2, all 16 rows carrying real bet%/money% splits -- the live page is
firmly era2 as expected, since era1 predates Nov 2022). The `.ps1` wrapper
was also run directly (not through Task Scheduler) and logged
`OK snapshot=20260820T114157Z era=era2_scoreboard_response rows=16
rows_with_data=16` to `data/raw/public_betting_live/capture_log.txt`,
confirming the full wrapper-to-parser-to-log path end to end.

**Task Scheduler registration: NOT performed this session.** **Measured**,
repo-wide search (`grep -rln "ScheduledTask\|schtasks\|Task Scheduler"`):
no script or doc anywhere in this repository contains a `schtasks` or
`Register-ScheduledTask`/`New-ScheduledTask*` invocation. The six existing
`Odds_*` tasks (`docs/week1_readiness.md`: `Odds_TueOpen`, `Odds_ThuTNF`,
`Odds_MonMNF`, `Odds_Sat`, `Odds_SunClose`, `Odds_SunLate`, all `Ready`)
are referenced only as already-existing state, never as something a
tracked script registers -- their registration was manual/owner-side. Per
this task's own instruction, this session mirrors that: it does NOT run
`schtasks` itself. The exact command to register the two weekly captures
this task calls for (Saturday and Sunday, noon ET, bracketing the slate)
is documented here for the owner to run:

```
schtasks /Create /TN "PublicBetting_Sat" /TR "powershell.exe -NoProfile -ExecutionPolicy Bypass -File \"F:\Repos\nfl_py3\scripts\public_betting_capture.ps1\"" /SC WEEKLY /D SAT /ST 12:00 /RL LIMITED

schtasks /Create /TN "PublicBetting_Sun" /TR "powershell.exe -NoProfile -ExecutionPolicy Bypass -File \"F:\Repos\nfl_py3\scripts\public_betting_capture.ps1\"" /SC WEEKLY /D SUN /ST 12:00 /RL LIMITED
```

**Measured** this session (`Get-TimeZone`): this machine's own local
timezone is already `Eastern Standard Time` / "Eastern Time (US & Canada)",
DST-aware -- the same machine `Odds_*` already runs on, per
`docs/ops_runbook.md`'s "the scheduled live odds captures land ~06:00-09:00
ET" convention. A Task Scheduler local-time trigger of `12:00` is therefore
already noon ET with no UTC conversion needed; re-verify with `Get-TimeZone`
before registering on any other machine. Add `/RU`/`/RP` flags matching
however the existing `Odds_*` tasks authenticate (not discoverable from the
repo -- `scripts/odds_capture.ps1`'s own comments describe reading
`THE_ODDS_API_KEY` from `HKCU\Environment` or the explicit `HKEY_USERS`
hive specifically because "S4U tasks may not map HKCU to the real user
profile", implying the existing tasks run under an S4U-style logon; this
capture needs no API key so the same constraint may not apply, but the
owner should register it under the same account/logon type as the
existing six for consistency).

**Reproducibility note**: cell C's model source is
`artifacts/margins/20260820T004951Z/predictions.parquet` (filtered to
`method="market_residual"`, `model_name="ridge"`), whose
`configuration_sha256` matches `artifacts/active_ats_model.json`'s
`evaluation_configuration_sha256` exactly (both
`d5259477727e0cdd84c5c3e17200c71002697f31f83a808401b75d2ddd29eb05`) --
confirmed the same frozen production model, not a different run reused by
coincidence. Both artifacts are local/gitignored and may be absent in a
fresh clone; `docs/public_betting_battery_predeclaration.md` item 8 records
the exact `margin-backtest` configuration to regenerate it.
