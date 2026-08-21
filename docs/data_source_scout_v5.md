# Data source scout v5

Written 2026-08-21. Six parallel research agents produced the sections below
(market structure; official football data; unconventional/cross-domain; CFB
cross-league; other-sports analogs; historical-odds deepening), each instructed
to verify access live today and to avoid every source taken by
`data_source_scout_v2/v3/v4.md`. Provenance tags are inline per section:
**measured** (fetched/run today, URL given), **read** (a page opened),
**reported** (search snippet, unverified), **inferred** (reasoning).

The orchestrating session has NOT independently re-fetched these endpoints;
within each section an agent-level "measured" claim is presented as
measured-at-the-session-level following the v4 precedent, while anything the
agent could only reach via search snippets stays **reported/unverified**.

Top actions this report implies (owner-facing summary):

1. **NFL.com official weekly injury-report archive** (football §1): free,
   point-in-time-A, verified back to 2011, plain HTML. It simultaneously
   replaces the dead-after-2024 nflverse injury feed (`docs/data_feasibility.md`
   flags "live source replacement required after 2024") and delivers
   Friday-filed designations into the post-Tuesday pick window the project
   measures at roughly +1.3-1.5 accuracy points. Highest EV-per-effort item in
   this report.
2. **Big Ten (2023+) and SEC (2024+) conference availability reports** (CFB
   §1-2): the XLG-07 "no historical CFB availability source exists" fail-closed
   conclusion is outdated for 2023+; two league mandates now publish exactly
   the pregame artifact the program lacked.
3. **VegasInsider boards via Wayback** (odds-deepening §1): multi-book Tuesday
   boards + totals + movement pages, 2005-2016, dense captures measured — the
   only structurally new historical-market source found this cycle.
4. **FantasyFootballCalculator ADP API** (unconventional §2): free keyless REST,
   exact dated windows, 2007+ archive; cheapest clean build in the report.

---

## Section A — betting-market structure

### Ranked candidates

| Rank | Source | Mechanism | PIT | Access | Verified |
|---|---|---|---|---|---|
| 1 | VegasInsider multi-book midweek board via Wayback | market state between frozen Tuesday opener and kickoff, ~daily capture density incl. Tuesdays 2010-2026 | A | A | yes |
| 2 | Key-number push/teaser ladder computed on own openers (own idea) | settlement-line geometry from data already in repo | A | A (zero new access) | yes |
| 3 | Betfair historical exchange data | only true second-market liquidity source; volume needs paid ADVANCED tier, free BASIC is price-only | A | B | yes |
| 4 | Palp-error corpus (own idea) | book mistakes as information events; honest weak-mechanism assessment recorded rather than dropped | varies | varies | partial |
| 5 | Halftime/live line archives | no free halftime-line archive exists beyond repo's already-ingested SBR `2H` columns (VI halftime subpages have zero Wayback captures — measured, empty CDX) | — | — | closed |

Session summary: 6 WebSearch queries, 2 WebFetch calls, 9 curl commands against
Wayback CDX/snapshot endpoints, teamrankings.com slug probes, and the OddsPortal
results page. Closed with evidence, not vibes: no free teaser-price archive
exists (strategy prose only — reported); ScoresAndOdds' Wayback footprint too
sparse to use (measured); Betfair worth one free-month probe before any spend.

---

## Section B — official/semi-official football data

### Ranked candidates

| Rank | Source | Mechanism | PIT | Access | Verified |
|---|---|---|---|---|---|
| 1 | NFL.com official weekly injury-report archive (`nfl.com/injuries/league/{season}/{reg\|post}{week}`) | league's OWN final designations publish Friday/Saturday — after the Tuesday grading-line freeze but before our pick deadline; also the missing post-2024 live replacement | A | A | **yes** |
| 2 | FantasyData snap counts, preseason weeks 2012-2026 | answers v4's rank-11 CONFIRMED GAP: structured preseason snap counts for Weeks 1-4 readiness | A | B (CSV export JS/login-gated, unverified) | yes, page + selectors only |
| 3 | NFL media-site daily Personnel Notice PDFs | gameday practice-squad elevations announced Saturday/day-of — the LATEST roster signal that exists | A if obtainable | C (measured: anonymous fetch hits AEM login redirect) | yes, blocked |
| 4 | Next Gen Stats team-level aggregates | tracking traits refreshed nightly in-season can diverge from a stale Tuesday line | B | C (measured: API 401 unauthenticated) | yes, blocked without session |
| 5 | Full officiating-crew assignments (footballzebras) | widens referee-only battery to whole-crew continuity/mix | C | A | yes |
| 6 | nflverse combine/pro-day dataset | athletic-trait priors; cheapest new ingest; zero late-week value (reported dataset existence, not fetched) | A | A | reported |
| 7 | OC/DC change history | still unresolved; two weak new leads (single-commit GitHub scraper; altdraft.com JS database) | A | C | leads only |

### Verified detail worth keeping (Section B)

- **NFL.com injuries**: fetching `nfl.com/injuries/league/2024/reg3` returned
  the complete Week-3 2024 report — practice status AND game status per player;
  `nfl.com/injuries/league/2011/reg5` returned a full real report including the
  old fourth "Probable" tier, so the archive demonstrably reaches at least 2011
  (page selector lists 1965-2026; pre-~2004 inferred sparse). One snapshot per
  week (the final report) — the Wed/Thu/Fri REVISION stream is explicitly NOT
  recoverable from this source (measured); if wanted, it must be collected
  prospectively starting now.
- **Personnel Notices**: `curl` on the 9-30-24 URL returned HTTP 302 → AEM
  login interstitial (measured). May yield to browser session or Wayback —
  neither tested.
- **NGS**: internal API 401s without session headers (measured); the working
  recipe likely lives in `github.com/nflverse/ngs-data` R code (reported).
- Cheapest experiments: scrape 2022-2024 NFL.com injury pages and measure
  agreement vs nflverse (≥99% match ⇒ extend pipeline to present); one
  network-tab capture decides FantasyData viability.

---

## Section C — unconventional / cross-domain

### Ranked candidates

| Rank | Source | Mechanism | PIT | Access | Verified |
|---|---|---|---|---|---|
| 1 | JetTip weekly team-charter flight archive (blog.jettip.net) — revives the twice-killed charter seed via human curation instead of tail-number linkage | executed travel timing (Saturday-night red-eyes before short weeks, bus-instead-of-fly notes) is late-arriving fatigue info a Tuesday line prices only from schedule | B (posts publish Tue/Wed; some legs TBD) | A (plain blog) | **yes** |
| 2 | FantasyFootballCalculator historical ADP REST API | roster-aggregated fantasy-draft position = crowd expectation proxy; every snapshot stamped to its exact mock-draft window | A | A (free, keyless, attribution requested) | **yes** |
| 3 | Arctic Shift / PullPush Reddit archives — team-subreddit VOLUME (deliberate reframe of v2's killed NLP idea) | pregame post/comment volume as attention proxy from exact `created_utc`; counts need no sentiment model | A | A (verified live, no auth) | **yes** |
| 4 | NYT Archive API article counts | editorial-attention proxy provenance-distinct from Wikipedia pageviews | A | B (free key, rate-limited; endpoint live, 401 on dummy key) | yes |
| 5 | GDELT Television Explorer — adjacent to scouted GDELT family: same project, different modality (TV airtime share, not article counts) | broadcast-attention proxy with daily resolution back to mid-2009 | B (caption noise documented) | A (free, no key) | yes (API up; station-scope syntax needs work) |

### Verified detail worth keeping (Section C)

- JetTip: fetched `blog.jettip.net/nfl-2025-week-17-team-charter-flights`;
  real per-team rows with routes/times/aircraft; author states tabs kept since
  2019. FAA registry does NOT solve tail-number→team (chartered 767/777s
  registered to airlines — measured). Cheapest experiment: hand-code 2023-2025
  discrete flags (~50 posts), split-half reliability BEFORE building a scraper.
- FFC ADP: live call returned genuine JSON with `"total_drafts": 2403,
  "start_date": "2020-08-30", ...`; vendor help states free for personal and
  commercial use, archives to 2007 (12-team formats). Weeks 1-4 covariate only.
- Arctic Shift: live call on CHIBears returned submissions with created_utc/
  score/num_comments/subscriber fields, no auth (reported coverage Dec 2005+).
  Gate ATS work behind a shared-variance check against the existing Wikipedia
  feature.
- Killed explicitly, with reasons: YouTube/Twitch view history (API returns
  current snapshot only — prospective-only), CME weather futures (monthly index
  granularity), app-store review velocity (no history), utility-grid demand
  (invisible at aggregation), satellite tailgate imagery (arrives days AFTER
  the Tuesday lock — structurally disqualified), airport/hotel price indices
  (quarterly/paid), school calendars (no mechanism), municipal event permits
  (effort L, second-order), sports-radio listenership (quarterly), Google Books
  Ngram (annual, ends ~2019).

---

## Section D — CFB cross-league

### Ranked candidates

| Rank | Source | Unblocks | PIT | Access | Verified |
|---|---|---|---|---|---|
| 1 | Big Ten Football Availability Reports archive (2023+) — league-mandated reports ≥2h before kickoff, per-week archived PDFs on bigten.org | XLG-07 (first-ever pregame CFB availability signal) | A | A | **yes** (live 2023 hub + direct weekly PDFs) |
| 2 | SEC Student-Athlete Availability Reports (2024+) — Wed initial, Thu/Fri updates, final 90 min pregame; fines enforce accuracy | XLG-07 (multi-day trajectory richer than NFL snapshots) | A | A | **yes** (live archive read) |
| 3 | College Poll Archive (AP 1936+, Coaches 2007+, CFP 2014+; Sports-Reference dated weekly polls) | XLG-06 early-season priors where span-8 state has no data | A | A | yes |
| 4 | 247Sports Composite rankings 2000+ | fills the exact 2000-2012 recruiting gap CFBD leaves open | A signed-class / B in-cycle | B (browser UA; ToS gray, private cache OK) | yes |
| 5 | On3 Transfer Portal Wire (2021+, dated entries/commits) | the missing timestamp axis CFBD portal data lacks | B (needs Wayback for verification) | B | yes |
| 6 | Coaching-staff change history — CFBD `/coaches` alive (HC tenures, S effort); collegefootballpoll.com dated HC moves incl. mid-season firings | offseason-disruption flag | varies | A/S | yes |
| 7 | Bowl opt-out trackers (dated bullets, 2020+) | late-arriving availability; positive control for whether markets price announced absences in CFB | C (pages updated in place) | A | yes |

Bottom line: XLG-07's fail-closed conclusion is outdated for 2023+. First
experiment: hand-pull one month of B1G PDFs, join "Out" starters to the XLG-03
table, measure line movement/spread error vs non-flagged games. B1G+SEC gives
~2 seasons × ~200 flagged-team-games — near XLG-03's own detection floor.

---

## Section E — other-sports analogs

All home-sport numbers **reported** (search excerpts; primaries not opened);
NFL designs **inferred**. Nothing measured against local data.

1. **[PRICING-GAP] Rest/travel underadjustment (NBA back-to-backs)** — Ashman,
   Bowman & Lambrinos 2010, JSE: home teams on game 2 of back-to-backs vs
   rested visitors performed poorly ATS, magnified traveling 1-2 zones east.
   NFL analog: compressed-week games; rest-differential/timezone features from
   schedules (all local).
2. **[PRICING-GAP] Official identity underpriced (MLB plate umpires)** —
   extreme-umpire totals effect ~0.3-0.5 runs vs books moving 0.1-0.25;
   assignments public 2-3h pregame. NFL analog: crew fixed effects on penalty
   EPA/scoring; split-half reliability check first.
3. **[PRICING-GAP] Stale-prior anchoring (soccer promoted teams priced off
   second-tier data ~8 games)** — NFL translation: high-offseason-turnover
   teams weeks 1-8 (turnover index from rosters).
4. **[PRICING-GAP, conditional] Drift-as-information (Betfair)** — requires
   multi-timestamp line archive; verify archive granularity before designing.
5. **[PRICING-GAP, thin] Dead-rubber motivation asymmetry (NBA tanking)** —
   eliminated teams rest players; cheap test from schedules + standings.
6. Noted/skipped: favorite-longshot re-measurement ([QUALITY], bounded near
   zero here); red-card reaction speed (no pregame forced-pick analog).

Recommended order: 1 → 2 → 3, 4 gated on verifying archive timestamps.

---

## Section F — historical-odds deepening

Internet Archive was intermittently offline during scouting (measured); empty
results below may be transient.

1. **VegasInsider NFL boards via Wayback — strongest lead.** CDX density
   measured: ~100+ captures Aug 2010-Feb 2017 alone; Sept-Nov 2011 near-daily.
   Content measured (fetched snapshot `20111102063341`, saved locally):
   server-rendered table survives — rotation numbers, kickoff time, per-book
   spread AND total cells across several books, plus per-game line-movement
   trail pages archived in their own URL space. Unique value: multi-book
   dispersion + totals + midweek movement 2005-2016, complementing SBR's
   opener/close pairs. Effort S-M (same parsing shape as SBR ingest).
2. **Free CSV datasets**: `willvernon/NFL_Scores_Lines` raw CSV downloaded this
   session (HTTP 200; `spread_line,total_line,...` populated from 1999) —
   closes, adding 1999-2006 depth plus totals cross-check (S). Kaggle
   `tobycrabtree/nfl-scores-and-betting-data` claims spreads since 1979
   (reported; Kaggle auth).
3. **KillerSports**: site live (HTTP 200 measured); Wayback hosts year-by-year
   "NFL Bible" PDFs 2014-2020 status 200 (measured); SDQL database reportedly
   carries deep lines incl. openers (reported — needs account to test). M.
4. **Dr Bob weekly reviews via Wayback**: dense homepage captures since 1996;
   analysis pages 2009-2013 contain Best-Bet prose with stated lines but NO
   structured table (measured) — parse-hard, effort L, low priority.
5. Measured negatives: Covers.com grids load from `/data/nfl/*` fragments which
   have ZERO Wayback captures (shell-only archive — dead); OddsPortal zero NFL
   captures (dead); Sports Insights/BetLabs trend pages paywalled as before
   (unchanged from v2 verdict); StatFox linemoves exactly ONE capture (ghost);
   ESPN Chalk CDX empty twice (parked unresolved-low, possibly transient).
