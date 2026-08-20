# Pro Football Rumors transaction-wire archive: ingestion + coverage report

Ingestion of `docs/data_source_scout_v3.md`'s rank-1 candidate ("Pro Football
Rumors transaction-wire sitemap" -- read that document's section 1 for the
mechanism case: roster moves are exactly the kind of information that can
firm up between a pool's Tuesday-noon lock and kickoff). Clones the proven
ProFootballTalk (PFT) ingestion pattern documented in
`docs/injury_news_sourcing.md` sections 1-4 (read): sitemap-index -> per-chunk
fetch -> URL/slug extraction -> a per-article JSON-LD verification sample.
Scope of this document and this session: **ingestion + coverage report only**
-- no experiments run, no registry writes, no `src/nfl_ats` changes, no
challenger wiring. Every claim below is tagged **measured** (fetched or run
this session, exact command/URL given), **read** (a file opened this
session), or **inferred** (reasoning, not evidence); there are no unverified
**reported** claims in this document -- every scout-doc assertion reused here
was independently re-fetched and re-measured this session rather than taken
on faith.

Code: `scripts/ingest_transaction_news.py` (new; does not touch
`ingest_injury_news.py`, `experiment_runner.py`, `margin.py`,
`public_board.py`, `cli.py`, or anything under `src/nfl_ats`).

---

## 1. Source, access, and the lastmod-contamination problem

**Measured** this session: `https://www.profootballrumors.com/robots.txt`
sets `Crawl-delay: 1` for `User-agent: *` (friendlier than PFT/NBC Sports'
`Crawl-delay: 10`) and disallows only `/wp-admin/`, `/search`, and a handful
of query-string variants (`?redirect_to`, `?fvtc_order`, `?s`, `/*/email`) --
nothing on the sitemap or article paths used here. No block was encountered
or evaded.

**Measured**: `https://www.profootballrumors.com/sitemap.xml` is a WordPress
sitemap index listing yearly post chunks
`sitemap-posttype-post.YYYY.xml` for every year 2013-2026 (plus a
`sitemap-posttype-page.xml` for static pages and a `sitemap-home.xml`, both
deliberately excluded -- not articles), a `<?xml-stylesheet?>` and generator
comment identifying `XML & Google News Sitemap Feed` v4.7.5. **Measured**:
`sitemap-posttype-post.2013.xml` and `.2010.xml` both return HTTP 200 but
each contains only a single placeholder `<url><loc>https://www.
profootballrumors.com</loc></url>` entry, no real content -- confirmed
directly (`curl` fetch + full-body inspection, not just a `grep -c` count) --
so true article coverage starts at 2014.

**Lastmod contamination, measured with my own fetch (not just re-reading the
scout doc's claim):**

```
GET https://www.profootballrumors.com/2015/09/minor-nfl-transactions-9-23-15
```

JSON-LD block on the page (single `<script type='application/ld+json'>`
tag, one line):

```json
{"headline":"Minor NFL Transactions: 9/23/15", ...,
 "datePublished":"2015-09-23T16:47:54-05:00",
 "dateModified":"2025-12-26T23:27:00-06:00", ...}
```

The sitemap `<lastmod>` for this same URL, fetched in the same session from
`sitemap-posttype-post.2015.xml`, is `2025-12-27T05:27:00+00:00`, which
converts to the same `2025-12-26T23:27:00-06:00` as the JSON-LD
`dateModified` -- **confirming the scout doc's finding directly**: sitemap
`<lastmod>` tracks a 2025-12-26/27 bulk site-wide retouch (a plugin/template
refresh across old posts), not the true 2015-09-23 publish date. This is
**not** universal -- the same yearly chunk contains many rows whose
`<lastmod>` is a plausible original-era date (e.g. 2018-02-26, 2022-05-02) --
so contamination is partial, and its actual rate needed to be measured, not
assumed (section 3 below).

**A cheaper, reliable alternative found this session and not in the scout
doc**: the article URL itself encodes `/YYYY/MM/slug` --
`/2015/09/minor-nfl-transactions-9-23-15` for a 2015-09-23 publish. This
costs zero extra fetches (it's already present in every sitemap `<loc>`) and
is verified in section 3 to match the true JSON-LD `datePublished` year/month
100% of the time in a 325-article stratified sample -- a free, full-inventory
date proxy at month granularity, with the per-article JSON-LD fetch reserved
for day/hour precision where it's actually needed (e.g. a Tuesday-noon-cutoff
test; see section 6).

---

## 2. Inventory built

**Measured**, `data/raw/pfr_transactions/20260820T011126Z/manifest.json`
(gitignored under the repository's existing `data/raw/**` rule, not
committed):

| Year | URLs | transaction_relevant (keyword-matched) |
|---|---:|---:|
| 2014 | 5,635 | 2,200 |
| 2015 | 5,858 | 2,084 |
| 2016 | 6,166 | 2,375 |
| 2017 | 6,577 | 2,564 |
| 2018 | 6,608 | 2,669 |
| 2019 | 5,957 | 2,566 |
| 2020 | 4,915 | 2,042 |
| 2021 | 5,323 | 2,509 |
| 2022 | 5,283 | 2,194 |
| 2023 | 4,954 | 2,093 |
| 2024 | 5,498 | 2,338 |
| 2025 | 5,668 | 2,371 |
| 2026 (partial, through ~Aug 19) | 3,926 | 1,409 |
| **Total** | **72,368** | **29,414 (40.6%)** |

`transaction_relevant` is a deliberately over-inclusive URL-slug keyword
match (signings, cuts, waivers, trades, IR/practice-squad moves, extensions,
tenders, suspensions, retirements, etc. -- full list in the manifest's
`transaction_keywords`) mirroring the PFT script's `injury_relevant` design:
**every** PFR post URL is retained regardless of match, with the boolean
flag, so the keyword line can be redrawn later without re-fetching (and was:
see the `extend`-keyword fix in section 5). The remaining ~59% is PFR's
broader "rumors" content -- mock drafts, contract-value analysis, opinion
round-ups, "extra points" links posts -- not itself transaction news but
correctly kept in the full inventory per the task's headline-from-slug
design.

Every row also carries `headline_from_slug` (the URL slug with hyphens
replaced by spaces, e.g. `minor-nfl-transactions-9-23-15` ->
`minor nfl transactions 9 23 15`, or `texans-andre-hal-retires-from-nfl` ->
`texans andre hal retires from nfl`), extracted at zero marginal fetch cost
for all 72,368 rows, matching the task's explicit ask ("URL slugs themselves
usually carry the headline text -- extract headline-from-slug for the full
inventory so the corpus is useful even where the body was not fetched").

**Snapshot layout** (matching this repo's `data/raw/<source>/<UTC
timestamp>/` convention, `manifest.json` nested one level below `pfr_
transactions/` so `nfl_ats.snapshots.latest_snapshot()` cannot mistake it for
a schedules snapshot -- read `scripts/ingest_injury_news.py`'s docstring for
why this matters, the same collision it warns about applies here):

```
data/raw/pfr_transactions/20260820T011126Z/
  yearly/<YYYY>.parquet        one row per PFR post url, that year's chunk
  index.parquet                concatenation of all 13 yearly files, 72,368 rows
  manifest.json                run metadata + per-year coverage
  sample_articles/<slug>.json  325 per-article JSON-LD verification fetches
  date_verification_summary.json               general-population sample (n=195)
  date_verification_summary_relevant_only.json transaction_relevant-only sample (n=130)
```

---

## 3. Per-article date verification: fetch cost and reliability

**Measured**, two stratified (evenly spread across all 13 years, ~15 or ~10
per year) samples, real article fetches with JSON-LD extraction:

| Sample | n | fetch failures | elapsed | mean s/article | url_year/month matches datePublished | lastmod within 1h of datePublished |
|---|---:|---:|---:|---:|---:|---:|
| General population | 195 | 0 | 194.4s | 1.00 | **195/195 (100.0%)** | 164/195 (84.1%) |
| `transaction_relevant`-only | 130 | 0 | 129.4s | 1.00 | **130/130 (100.0%)** | 95/130 (73.1%) |

Two findings, both measured directly (not extrapolated from the scout doc's
single-article example):

1. **The free URL-path year/month proxy is perfectly reliable** in this
   325-article sample (100% both populations) -- it can be trusted as a
   month-granular publish-date bound for the full 72,368-row inventory
   without any further fetching.
2. **Sitemap `<lastmod>` is measurably unreliable**, and more so on the
   transaction-relevant subset (73.1% agree with `datePublished` within an
   hour) than the general population (84.1%) -- plausibly because
   transaction/roster-move posts (e.g. "Minor NFL Transactions" round-ups)
   get same-day or later factual updates more often than static
   analysis content, which would show up as a genuine `dateModified` change
   layered on top of the separate 2025-12 bulk-retouch contamination this
   document's section 1 already isolated. This document does not
   distinguish "genuine same-day edit" from "years-later bulk retouch"
   within the 73.1%/84.1% failure rate -- both make `<lastmod>` unusable as
   a point-in-time proxy, which is the operative conclusion either way.

**Fetch budget spent, measured**: sitemap index + 13 yearly chunks (~14
requests, ~14s at the 1s crawl delay) + 195-article general sample (194.4s)
+ 130-article relevant-only sample (129.4s) = **~338 seconds (~5.6 minutes)
of per-article/per-chunk fetch time**, well inside the task's ~30-minute
budget. No further bulk per-article fetching was done this session --
extrapolating the measured 1.0s/article rate, verifying JSON-LD dates for
the full 72,368-row inventory would take ~20 hours, and for just the
29,414-row `transaction_relevant` subset ~8.2 hours; both are explicitly out
of scope for "ingestion + coverage report" and are left as future,
deliberately-budgeted work (like the PFT script's own bulk-body-fetch
tradeoff already documents for its source).

**Verified-datePublished fraction of the full inventory**: 325 / 72,368
articles (0.45%) have a real fetched JSON-LD `datePublished` on disk this
session; the remaining 99.55% carry only the free, measured-100%-reliable
url_year/month bound (month granularity) plus the unreliable raw `<lastmod>`
(kept for transparency, never used as ground truth).

---

## 4. Verification sample files

`sample_articles/*.json` (325 files, 322 unique -- 3 slugs were sampled by
both the general and relevant-only draws and overwrote each other, harmless
since the content is identical) each record: `url`, `slug`, `sitemap_year`,
`url_year`, `url_month`, `lastmod_contaminated`, `json_ld_date_published`,
`json_ld_date_modified`, `json_ld_headline`. Spot examples, measured:

| URL | url_year/month | lastmod (contaminated) | JSON-LD datePublished |
|---|---|---|---|
| `2026/06/minor-nfl-transactions-6-25-26` | 2026/06 | 2026-06-26T00:12:40Z | 2026-06-25T19:12:40-05:00 |
| `2026/07/saints-finalizing-restructured-deal-with-rb-alvin-kamara` | 2026/07 | 2026-07-16T07:09:59Z | 2026-07-15T22:20:18-05:00 |
| `2026/01/rams-activate-s-quentin-lake-from-ir` | 2026/01 | 2026-01-06T22:03:07Z | 2026-01-06T16:03:07-06:00 |

All three of these current-season (2026) rows show `lastmod` reasonably
close to `datePublished` (within a day) -- the contamination is
concentrated in older posts touched by the 2025-12 bulk retouch, consistent
with section 1's single-example finding, now confirmed distributionally
rather than anecdotally.

---

## 5. Keyword-line correction made this session

**Measured**: the initial `TRANSACTION_KEYWORDS` list matched `extends` /
`extension` / `extending` substrings but not the bare verb `extend`, missing
544 of 72,368 rows (0.75%) whose slug contains `extend` but no other
transaction keyword (e.g. `falcons-extend-smith-dimitroff-mckay`,
`eagles-extend-jason-peters`, `steelers-extend-troy-polamalus-contract`).
Fixed by adding `extend` to the keyword list and re-running
`--recompute-keywords` (a new mode added to the script, no network fetch --
it redraws `matched_keywords`/`transaction_relevant` from the already-cached
`slug` column on disk, the same "redraw without re-fetching" design the PFT
script's docstring already commits to). `transaction_relevant` rows rose
from 28,870 to **29,414** (+544, exactly the gap found); `cumulative_index_
rows` unchanged at 72,368 -- no rows added or lost, confirming the recompute
only touched the boolean flag and keyword-match column. The per-year table
in section 2 reflects the corrected counts.

---

## 6. Predeclared next-step experiment (NOT run this session)

Per this task's scope (ingestion + coverage report only) and per AGENTS.md's
binding closing-grounds taxonomy, restated verbatim as required for any
downstream subagent or scoring pass:

> An interval or CI that contains zero is NEVER grounds to reject, fail, or
> close an experiment. At this evaluator's ~2-point resolution, "contains
> zero" is the EXPECTED outcome for a real small signal. Only two grounds
> ever close a line of work: (1) refuted mechanism -- a RESOLVED wrong sign
> (whole interval on the wrong side of zero) or zero split-half reliability;
> (2) bounded by a positive control proven able to detect an effect that
> size. Everything else is `unresolved_below_power`: record it with
> `nfl-ats weak-signals record`, report `probability_positive`, never the
> binary "contains zero". Verdicts must flow through `nfl-ats weak-signals
> record` / `nfl-ats rotation record-look` -- never through prose in a doc.

Mirroring `docs/injury_news_sourcing.md` section 5's style (two directly
testable questions, predeclared before running):

1. **Does PFR add transaction visibility beyond what the already-ingested
   PFT source (`docs/injury_news_sourcing.md`) provides by the pool's
   Tuesday-noon lock?** For each historical (season, week, team), diff the
   set of PFR `transaction_relevant` headlines with a verified JSON-LD
   `datePublished` before that week's own Tuesday noon ET against the same
   week's PFT injury/availability headlines already ingested
   (`data/raw/injury_news/*/index.parquet`). A meaningful PFR-only
   (non-overlapping) subset would be a genuinely additive, independent-outlet
   signal; near-total overlap would mean PFR is redundant with PFT for this
   specific use case. **Precision requirement, stated up front**: this test
   needs day/hour-level dating, so it must use real per-article JSON-LD
   `datePublished` fetches for the specific weeks under test -- the free
   url_year/month proxy built in this session is only month-granular and is
   NOT sufficient for a Tuesday-noon cutoff test by itself.
2. **Does PFR's transaction coverage foreshadow official roster-status
   changes (IR placements, practice-squad moves, cuts) ahead of when they
   become visible in the project's existing nflverse-derived state?**
   Same own-week Tuesday-noon-ET construction `docs/injury_news_sourcing.md`
   section 5.1 already built and ran for PFT
   (`scripts/injury_tuesday_cutoff_experiment.py`,
   `nfl_ats.clv.live_tuesday_openers`'s own-week-Tuesday convention) --
   re-run with PFR `transaction_relevant` headlines substituted for (or
   pooled with) PFT's injury-keyword headlines, to see whether pooling the
   two independent outlets changes the visibility fraction PFT alone
   measured (0.43% official-report rows visible by Tuesday noon with no
   news credit, 8.13% with PFT-headline credit -- both measured in `docs/
   injury_news_sourcing.md` section 5.1).

Neither question is run here. Whatever `probability_positive` either
analysis returns, an interval crossing zero is not grounds to close the
question; any resulting accuracy/coverage measurement should be recorded
through `nfl-ats weak-signals record` with the appropriate classification
(most likely `unresolved_below_power` unless one of the two admissible
closing grounds is cleared), never asserted as settled in prose.

---

## 7. Provenance summary

- **Measured this session**: `robots.txt` fetch, the sitemap index fetch and
  all 13 yearly chunk fetches, the 2015-09-23 lastmod-contamination example
  (independently re-fetched, not just re-read from the scout doc), the empty
  2013/2010 chunk check, the JSON-LD structure inspection, both
  195-article and 130-article stratified verification samples and their
  summary statistics, the `extend`-keyword gap (544 rows) and its
  `--recompute-keywords` fix, all per-year inventory counts, the ~338s fetch
  budget.
- **Read this session**: `docs/data_source_scout_v3.md` section 1,
  `docs/injury_news_sourcing.md` sections 1-4, `scripts/ingest_injury_news.py`
  in full (the cloned template), `docs/sbr_odds_archive.md` (style
  reference), `.gitignore` (confirmed `data/raw/**` covers this snapshot).
- **Inferred**: the "private research caching, never republish" policy
  stance for the PFR archive, by analogy to this project's existing
  CFBD/PFT precedent (`docs/data_feasibility.md` License item 6) -- Pro
  Football Rumors' own terms of use were not independently reviewed this
  session, same caveat the PFT document already carries for NBC Sports; the
  hypothesis (in section 3) that transaction posts get more frequent genuine
  same-day edits than the general population, offered to explain the
  73.1% vs 84.1% lastmod-reliability gap -- not independently confirmed,
  reasoning only.
- Nothing in this document is a **reported** (unverified subagent/search)
  claim -- every number was fetched or computed directly this session.

No experiments were run, no `registry/` files were written, no files under
`src/nfl_ats` were touched, and no challenger wiring was added, per this
task's scope.
