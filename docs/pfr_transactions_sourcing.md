# Pro Football Rumors transaction-wire archive: ingestion + coverage report

Ingestion of `docs/data_source_scout_v3.md`'s rank-1 candidate ("Pro Football
Rumors transaction-wire sitemap" -- read that document's section 1 for the
mechanism case: roster moves are exactly the kind of information that can
firm up between a pool's Tuesday-noon lock and kickoff). Clones the proven
ProFootballTalk (PFT) ingestion pattern documented in
`docs/injury_news_sourcing.md` sections 1-4 (read): sitemap-index -> per-chunk
fetch -> URL/slug extraction -> a per-article JSON-LD verification sample.
**Read**: the initial scope was ingestion + coverage reporting. A same-day
follow-up subsequently ran the two predeclared coverage/additivity questions;
section 6 reconciles the original partial-cache result at
`artifacts/pfr_pft_additivity/result.json` with the complete-cache rerun at
`artifacts/pfr_pft_additivity/20260820T155757Z/result.json` and its versioned
registry record at `registry/experiments/pfr-pft-additivity-experiment/
20260820T155757Z.json`.
**Read**: this work did not run an ATS model screen, write a weak-signal or
rotation verdict, change `src/nfl_ats`, or wire a challenger. Every claim below
is tagged **measured** (fetched or run in the stated session, exact command or
artifact given), **read** (a file opened in the stated session), or **inferred**
(reasoning, not evidence); there are no unverified **reported** claims in this
document.

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

**Measured in the initial ingestion**: sitemap index + 13 yearly chunks (~14
requests, ~14s at the 1s crawl delay) + 195-article general sample (194.4s)
+ 130-article relevant-only sample (129.4s) = **~338 seconds (~5.6 minutes)
of per-article/per-chunk fetch time**. **Inferred at that point**: fetching all
72,368 articles would take about 20 hours and all 29,414
`transaction_relevant` articles about 8.2 hours at the measured 1.0s/article
rate, so the later bulk fetch targeted only the in-season windows needed by
section 6.

**Measured in the 2026-08-20 bulk-fetch follow-up**, using
`.\.tools\uv.exe run --no-sync python scripts/pfr_bulk_date_fetch.py
--snapshot data/raw/pfr_transactions/20260820T011126Z`: the targeted scope is
4,361 unique `transaction_relevant` rows in August-through-January windows for
the 2022-2025 seasons. This invocation began with 3,628 target rows already
cached and fetched the remaining 733 in 734.5 seconds: 733 dates extracted,
zero failures. **Measured after completion**, by joining the cache filenames
back to `index.parquet`: all 4,361 target rows have valid JSON, an extracted
`datePublished`, the expected URL, and a publish year/month matching the URL;
there are zero target failures, malformed records, URL mismatches, or duplicate
slugs. The shared cache now has 4,653 valid dated files: 4,537 of 29,414
`transaction_relevant` inventory rows and 116 other verification-sample rows,
or 4,653 of 72,368 total inventory rows (6.43%).

---

## 4. Verification sample files

**Measured in the initial verification**: the two samples produced 325 fetched
records and 322 unique files -- three slugs occurred in both draws and safely
overwrote identical content. **Measured after the bulk follow-up**:
`sample_articles/*.json` contains 4,653 unique valid JSON records and every
record has an extracted `json_ld_date_published`; none is marked
`fetch_failed`. The initial sample records also carry `sitemap_year`,
`url_year`, `url_month`, and `lastmod_contaminated`; the bulk records retain the
common `url`, `slug`, `json_ld_date_published`, `json_ld_date_modified`, and
`json_ld_headline` fields. Spot examples, measured:

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

## 6. Predeclared coverage experiment and recorded result

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

**Read**: mirroring `docs/injury_news_sourcing.md` section 5's style, the
following two directly testable questions were predeclared before running:

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

**Read from the recorded artifact**
`artifacts/pfr_pft_additivity/result.json` (identical metrics are registered at
`registry/experiments/pfr-pft-additivity-experiment/pfr_pft_additivity.json`):
the follow-up ran both questions with a nine-day lookback at Tuesday noon and
at the Saturday refresh. This is a source-coverage experiment, not an ATS
accuracy experiment; it emits no `probability_positive` or rotation verdict.

| Recorded metric | Tuesday noon | Saturday refresh |
|---|---:|---:|
| PFR matched rows | 277 | 228 |
| PFT matched rows | 5,272 | 5,874 |
| PFR-only rows | 197 | 152 |
| PFR-only share of the union | 3.60% | 2.52% |
| Official injury rows | 16,838 | 16,838 |
| Official-only visible share | 0.31% | 92.69% |
| PFT-augmented visible share | 12.05% | 93.65% |
| PFR-augmented visible share | 0.49% | 92.72% |
| PFT+PFR visible share | 12.16% | 93.67% |
| Rows added by PFR over PFT alone | 18 | 3 |

**Read from that artifact**: the run used only 715 PFR
`transaction_relevant` rows with precise dates from 831 cache files then on
disk. Its per-season PFR matches were correspondingly lopsided: 269/1/3/4 at
Tuesday noon and 223/0/2/3 at Saturday refresh for 2022/2023/2024/2025.
**Read from the original artifact**: that result answers what the partial date
cache could see and is retained as historical provenance rather than presented
as the final cross-season PFR additivity estimate.

### Complete-cache rerun, 2026-08-20

**Correction note, 2026-08-21**: ROADMAP PER-03 flagged this section as still
reading "not run this session"; on audit that flag was stale — the subsection
below was already present with the rerun results, and every figure in it was
re-checked against `artifacts/pfr_pft_additivity/20260820T155757Z/result.json`
on 2026-08-21 and found accurate (no numbers changed in this pass).

**Measured** (`.\.tools\uv.exe run --no-sync python
scripts/pfr_pft_additivity_experiment.py`; versioned artifact
`artifacts/pfr_pft_additivity/20260820T155757Z/result.json`): the exact frozen
questions, 9-day lookback, populations, name matching, Tuesday-noon cutoff, and
Saturday-refresh cutoff were rerun without changing a threshold.

**Measured** (same artifact): all 4,361 rows in the predeclared precise-date
target have dates, and the shared cache contains 4,537 precisely dated
`transaction_relevant` rows in total with zero cached fetch failures or
date-extraction failures.

**Measured** (same artifact): PFR is materially additive to PFT as a source at
both frozen cutoffs and across every season in the scope.

| Complete-cache metric | Tuesday noon | Saturday refresh |
|---|---:|---:|
| PFR matched rows | 2,634 | 2,505 |
| PFT matched rows | 5,272 | 5,874 |
| PFR-only rows | 1,839 | 1,708 |
| PFR-only share of PFR matches | 69.82% | 68.18% |
| PFR-only share of the union | **25.86%** | **22.53%** |
| Official injury rows | 16,838 | 16,838 |
| Official-only visible share | 0.31% | 92.69% |
| PFT-augmented visible share | 12.05% | 93.65% |
| PFR-augmented visible share | 1.95% | 92.83% |
| PFT+PFR visible share | **13.07%** | **93.71%** |
| Rows added by PFR over PFT alone | **171** | **10** |

**Measured** (same artifact): Tuesday PFR matches / PFR-only matches are
571/381 in 2022, 613/425 in 2023, 718/498 in 2024, and 732/535 in 2025; the
complete-cache answer is no longer driven almost entirely by 2022.

**Measured** (same artifact): at Tuesday noon, pooling PFR with PFT raises the
official-injury visibility count by 171 rows beyond PFT alone and the visible
share from 12.05% to 13.07%; at Saturday refresh it adds 10 rows and moves the
share from 93.65% to 93.71%.

**Inferred**: the source decision is clear before caveats -- PFR provides a
substantial independent transaction-news channel rather than redundant PFT
coverage, especially at the earlier Tuesday checkpoint.

**Read** (`scripts/pfr_pft_additivity_experiment.py`): this remains a
source-coverage experiment; it does not load ATS outcomes, estimate an accuracy
effect, emit `probability_positive`, or produce a weak-signal/rotation verdict.

**Inferred**: the result supports retaining both sources for a future frozen
feature construction, but it does not establish that the additional coverage
improves the 53.4% opener-grade production rule.

**Measured** (`registry/experiments/pfr-pft-additivity-experiment/
20260820T155757Z.json`): the rerun has a new run identity and points to the
versioned artifact; the original `pfr_pft_additivity.json` record remains the
honest provenance for the partial-cache run rather than being overwritten or
duplicated under the same identity.

**Read** (`scripts/pfr_pft_additivity_experiment.py`): the final runner reads
only schedule identifiers, teams, kickoff, season/week, and game type from the
feature table and uses an explicitly nanosecond-typed `NaT`; it neither loads
ATS result columns nor emits the NumPy deprecation warning encountered during
the local validation pass.

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
- **Measured in the bulk-fetch follow-up**: the exact resume command above,
  its 3,628-before / 733-fetched / 4,361-complete counters, and the post-run
  cache integrity and coverage audit in section 3.
- **Read this session**: `docs/data_source_scout_v3.md` section 1,
  `docs/injury_news_sourcing.md` sections 1-4, `scripts/ingest_injury_news.py`
  in full (the cloned template), `docs/sbr_odds_archive.md` (style
  reference), `.gitignore` (confirmed `data/raw/**` covers this snapshot).
- **Read in the reconciliation follow-up**:
  `artifacts/pfr_pft_additivity/result.json` and its registered copy at
  `registry/experiments/pfr-pft-additivity-experiment/pfr_pft_additivity.json`;
  section 6 transcribes their coverage metrics without rerunning them.
- **Measured in the complete-cache rerun**:
  `artifacts/pfr_pft_additivity/20260820T155757Z/result.json` and its versioned
  registry record at `registry/experiments/pfr-pft-additivity-experiment/
  20260820T155757Z.json`; section 6 reports the frozen rerun directly.
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

**Read**: the recorded PFR/PFT coverage experiment was run between the initial
ingestion and this reconciliation. **Measured in this follow-up**: no
experiment was rerun, no `registry/` file or file under `src/nfl_ats` was
changed, and no challenger wiring was added.
