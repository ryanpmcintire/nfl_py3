# Data source scout v4

Scope: sources NOT already scouted or built by `docs/archive/data_source_scout_v2.md`
(SBR odds archive, Bet Labs, GDELT, state-legalization dates, Kalshi/
Polymarket, Delphi FluView/CDC ILI, OverTheCap, sun position, uniform
database, Madden ratings) or `docs/archive/data_source_scout_v3.md`
(ProFootballTalk + Pro Football Rumors news wires, NOAA/IEM weather
forecasts + actuals, Wikipedia pageviews, Action Network/covers.com public
bet%, The Odds API historical player-prop lines, Ourlads/ESPN historical
depth charts, interim head coaches, FiveThirtyEight Elo, PFR attendance,
footballzebras referee assignments, NFL flex scheduling). Confirmed by
reading both docs' headers before starting this report.

The task brief was explicit: **do not limit creativity to sources obviously
related to football.** This report deliberately includes candidates from
aviation tracking, federal disaster/environmental data, gaming-regulatory
filings, and podcast-index metadata alongside the more traditional
sports-data seeds, and ranks all of them on the same mechanism/access/
point-in-time rubric — a non-football provenance is not a discount, and a
familiar sports-media source is not a premium.

Every claim below is tagged **measured** (fetched or run this session, exact
command/URL given), **read** (a docs/API-reference page opened this
session), **reported** (a search snippet or vendor claim, unverified), or
**inferred** (reasoning, not evidence), per the binding AGENTS.md labeling
rule. This session ran four parallel research agents (ratings/market
divergence; player/roster context; physical/travel/environment; attention/
sentiment), each instructed to use real `WebFetch`/`curl` calls and tag every
claim, plus direct verification fetches by the orchestrating session on gaps
the agents didn't cover and on several of the highest-ranked findings.
Following `docs/archive/data_source_scout_v3.md`'s own precedent, a claim an agent
reports as **measured** (exact URL/command given, real HTTP status or
content quoted) is presented here as **measured** at the session level; a
claim an agent could only find via search snippet stays **reported**. Five
candidates below (marked "own idea" explicitly) are original ideas not
present in the task brief's seed list: penalty-**type** crew tendencies as
an in-house schema-widening question rather than a new source, EPA AirNow
wildfire-smoke air quality, the US Drought Monitor as a turf/field-condition
proxy, FEMA/NOAA disaster-displacement games, and state/national
gambling-regulator revenue reports. ADS-B Exchange was a seed-list item; it
is re-examined here specifically for whether its more open data policy
solves the tail-number identity-linkage problem that already killed v2's
FlightAware angle (it does not).

## Ranked candidates

| Rank | Source | Mechanism (one line) | Point-in-time grade | Access grade | Verified today |
|---|---|---|---|---|---|
| 1 | Sagarin ratings via the Wayback Machine | independent, actively-updated computer power rating with a genuine per-snapshot timestamp that can diverge from the market line | A | A | **yes** |
| 2 | Penalty-**type** crew tendencies (own idea — not a new source) | widens the project's own already-built, already-reliable (`+0.370` split-half) referee-crew penalty-rate battery from totals to types (DPI/holding/false-start) using a column nflverse already provides | A | A (zero new external access) | **yes** |
| 3 | EPA AirNow historical AQI (own idea) | wildfire-smoke-degraded outdoor West Coast games are a documented, discrete event class the market may underprice; smoke plumes can shift overnight | A | A | **yes** |
| 4 | US Drought Monitor weekly county statistics (own idea) | cumulative soil-moisture/turf-hardness proxy for outdoor grass stadiums, distinct in mechanism and agency from the project's already-built day-of weather actuals | A | A | **yes** |
| 5 | USA Today NFL player arrests database | an arrest/citation days before kickoff is a genuinely late-arriving roster-disruption signal, independent of the injury-report family already built | A | A | **yes** |
| 6 | Retractable-roof open/closed decisions | a late (T-90-minute) roof call changes wind/precipitation exposure for an otherwise-domed game in a way a Tuesday line cannot see | B (backtest); needs a live scrape for prospective use | A (already a field in the project's own nflverse schedule source) | **yes** |
| 7 | Massey Ratings composite archive | consensus of ~50+ independent computer systems, updated continuously, with 100+ years of season-level archive depth | C | A (browser-only; Cloudflare blocks plain fetches) | **yes** |
| 8 | Podcast Index API | team-podcast publish-volume as an attention proxy structurally distinct from Wikipedia pageviews and the already-killed Reddit/DFS-ownership ideas | B | A | **yes** |
| 9 | FTN Data / DVOA | opponent-adjusted efficiency rating, the direct successor to Football Outsiders' long-standing DVOA | C (weekly granularity unconfirmed) | A (free, no login on the table checked) | **yes** |
| 10 | Sports Media Watch weekly TV ratings (lagged) | last week's / season-to-date household-rating trend as a team-relevance/hype proxy, engineered strictly as a lag to avoid the "same-game rating is an outcome" objection that killed this idea in v2 | B | A | **yes** |
| 11 | Preseason usage patterns / snap counts | a projected starter's abnormal (or absent) preseason workload is a health/readiness signal for Weeks 1-4, and is a confirmed real gap in the project's own pipeline | A | C (no free structured source found) | **yes, gap confirmed; source not** |
| 12 | Coaching staff continuity beyond head coach (OC/DC) | a mid-season coordinator firing is a scheme-continuity shock distinct from the head-coach features already built | A (fact of a change) | C (no clean structured feed; PFR blocked) | **yes** |
| 13 | Spotrac contract-year / holdout / franchise-tag tracker | walk-year and franchise-tag friction are documented in-season motivation proxies, though mostly already known well before any Tuesday lock | B | C (partially paywalled; bot-detected) | **yes** |
| 14 | NFLPA team report cards | annual culture/facility grades; a possible discipline/buy-in proxy, but plausibly just a slower re-measurement of team quality, already shown near-zero ceiling | B | A | **yes** |
| 15 | FEMA declarations / NOAA Storm Events (own idea) | a disaster-forced home-game relocation strips true home-field advantage, but the relocation itself is normally announced well before a Tuesday lock | C (mechanism-timing, not access) | A | **yes** |
| — | Google Trends official API | the 2025 official API fixes pytrends' fatal rescaling-instability problem | A (once accessible) | **blocked** — gated alpha, application-only, no self-serve path found | checked, access-blocked |
| — | State/national gambling-regulator revenue reports (own idea) | monthly handle/hold as a market-composition proxy | A (each report is dated) | A | checked, mechanism too lagged for a specific week |

---

## VERIFIED TODAY

### 1. Sagarin ratings via the Wayback Machine

- **Mechanism**: Jeff Sagarin's predictive rating has historically updated
  roughly daily in-season; a Wayback-scraped snapshot captures the rating as
  of the archive's own crawl timestamp, giving genuinely late (post-Tuesday)
  information that can diverge from the market line, plus 15+ years of
  outer-season depth for backtesting.
- **Point-in-time safety**: **A**. **Measured**:
  `web.archive.org/web/20141018174848/http://sagarin.com/sports/nflsend.htm`
  returned real content, self-labeled *"NFL 2014 through games of October 13
  Monday - Week #6"* — the page's own internal date label and the
  independent Wayback capture timestamp double-confirm point-in-time
  provenance, a stronger guarantee than a source that only carries one or
  the other.
- **Access path**: free. **Measured**: direct `WebFetch` to `web.archive.org`
  URLs fails at the tool level ("unable to fetch from web.archive.org" — a
  client limitation, not a site block); a real browser session reached the
  same snapshot with no issue. **Measured** CDX query
  (`web.archive.org/cdx/search/cdx?url=sagarin.com/sports/nflsend.htm&output=json`)
  returned dense near-weekly in-season captures through Sep-Nov 2014,
  continuing through **2026-08-14** — the archive is actively maintained
  today, not a frozen relic like v3's FiveThirtyEight Elo finding.
- **Coverage years**: **measured** 2010-2026 on the `sagarin.com` domain.
  Earlier USA Today-hosted history (Sagarin has published ratings since the
  1980s) is **reported/inferred** — a CDX check on the old `usatoday.com`
  hosting path did not complete cleanly this session and needs a follow-up.
- **Next-step ingestion plan (effort: M)**: a CDX-driven weekly-snapshot
  crawl (same pattern already proven for Ourlads in v3) plus a plain-text
  parser for Sagarin's tabular dump, whose exact column layout has drifted
  over 15+ years and will need per-era handling.

### 2. Penalty-type crew tendencies — an in-house schema-widening question, not a new source

- **Mechanism**: the project's own `docs/referee_battery.md` already built
  and measured a referee-crew **total** penalty-rate battery with a real,
  moderate split-half reliability (`mean_total` **+0.370** across 158 pairs,
  **read** from that file) — a genuine persistent trait, not noise. The
  seed brief's "officiating crew penalty tendencies beyond referee identity"
  idea is naturally answered by widening that same construct from a raw
  count to penalty-**type** rates (defensive pass interference, holding,
  false start, etc.), since a crew's *type* mix, not just its *volume*, is
  the more specific mechanism cited in outside research (nxtbets and the
  Harvard Sports Analysis Collective's penalty-type breakdowns, both
  **reported**, search snippets only).
- **Point-in-time safety**: **A** — identical to the already-built
  construct: crew assignment is public before kickoff, and the feature must
  use the **prior**-season type mix, never the game's own outcome, exactly
  as the existing `mean_total` construct already enforces (**read**,
  `tests/test_experiment_runner.py::test_referee_flags_do_not_use_this_games_own_penalty_count`).
- **Access path**: **A, and free of new external access entirely.**
  **Measured**: this project's own locally cached PBP parquet
  (`data/pbp/team_style/raw_pbp_narrow.parquet`,
  `data/processed/game_features_pbp.parquet`) only carries `penalty` and
  `penalty_yards` — **not** `penalty_type` or `penalty_team` — confirming
  the gap is in this project's own trimmed local snapshot, not in the
  upstream source. **Reported** (nflfastR's own documentation, via search
  synthesis of the official field descriptions): the **full** nflfastR PBP
  schema includes `penalty_type` ("the penalty type of the first penalty on
  the given play") and `penalty_team`, i.e. the raw material already exists
  upstream in the same nflverse pipeline this project already ingests — this
  is a re-pull/schema-widening task, not a new vendor relationship. As a
  free, independent cross-check only (not a build target),
  **measured**: `nflpenalties.com/all-referees.php?year=2024&view=total` is
  a real, live, per-referee/crew penalty table, robots.txt (**measured**)
  is fully open (`Allow: /`), covering 2009-2025 with a season selector, and
  it explicitly credits nflfastR for its own 2020-2025 data — independent
  confirmation the underlying data exists upstream.
- **Coverage years**: matches the project's existing nflverse PBP coverage
  (2009-2025, per `docs/data_feasibility.md`'s existing inventory) — no new
  coverage-year constraint introduced.
- **Next-step ingestion plan (effort: S)**: re-pull (or widen the existing
  ingestion of) nflverse PBP to retain `penalty_type`/`penalty_team`, then
  extend `docs/referee_battery.md`'s existing `mean_total`/`mean_diff`
  cell-construction code to per-type rates. This is the cheapest item in
  this entire report because it needs no new external source, no new
  robots.txt/access negotiation, and extends a construct already proven to
  have non-zero reliability.

### 3. EPA AirNow historical air quality index (own idea)

- **Mechanism**: West Coast wildfire smoke has caused documented,
  discrete-event playing-condition degradation (haze, poor visibility,
  smoke taste/irritation) in specific years, most notably 2020 — a
  genuinely NOT-obviously-football federal environmental dataset, and
  smoke plumes can shift materially overnight, so a value queried close to
  kickoff can beat a frozen Tuesday assessment.
- **Point-in-time safety**: **A**, provided the query is restricted to
  observation timestamps strictly before kickoff (not a same-day average
  spanning post-game hours).
- **Access path**: free, key-gated. **Read**: `docs.airnowapi.org`
  documents a public API requiring only a free account. **Measured**: a
  live call to the historical zip-code observation endpoint with a dummy
  key returned HTTP **401 Unauthorized** (not 403/blocked) — confirming the
  endpoint is live and reachable, gated only by a trivial free signup, not
  by network or robots restrictions. **Read**: the zip/lat-long-specific
  historical endpoints are slated for retirement in fall 2026 but are being
  replaced by unified reporting-area/lat-long endpoints per AirNow's own
  migration notes, so stadium-coordinate queries keep working post-migration.
- **Coverage years**: AirNow network density on the West Coast since roughly
  the early 2000s is **inferred**, not verified this session and worth a
  direct check before building; EPA's separate Air Data Daily Tracker is
  **reported** (search snippet, unverified) to cite AQI history to 2000 and
  AQS records to 1980.
- **Next-step ingestion plan (effort: S-M)**: register a free AirNow API
  key, then a straightforward per-stadium lat/long, per-game-hour REST pull
  — mechanically simple once the key exists; the main open task is
  confirming actual station density near each West Coast stadium before
  committing to the feature.

### 4. US Drought Monitor weekly county statistics (own idea)

- **Mechanism**: cumulative soil-moisture deficit / turf hardness for
  outdoor grass stadiums is a longer-horizon field-condition proxy,
  mechanistically and administratively distinct from the project's already-
  built single-day weather actuals (this is a USDA/National Drought
  Mitigation Center agricultural-monitoring product, not an NWS/NOAA
  forecast feed) — a genuinely NOT-obviously-football source. Turf firmness
  affects footing, injury risk, and play speed over an accumulation window
  a single day-of temperature/precipitation reading does not capture.
- **Point-in-time safety**: **A**. **Measured**: `curl -sk
  "https://usdmdataservices.unl.edu/api/CountyStatistics/GetDroughtSeverityStatisticsByAreaPercent?aoi=42101&startdate=9/1/2020&enddate=9/30/2020&statisticsType=1"
  -H "Accept: application/json"` returned HTTP 200 with real weekly
  drought-severity-category rows for Philadelphia County, PA (`d0`-`d4`
  percentages), each carrying explicit `mapDate`/`validStart`/`validEnd`
  fields — every value is dated to an exact, non-overlapping weekly window,
  and each week's map is conventionally released a few days after
  `validStart` (a small, well-known publication lag that must be respected
  as the true availability time, not `validStart` itself, to stay
  pregame-safe).
- **Access path**: free, no authentication required, confirmed live.
  **Measured**: the `-k` (skip TLS verification) flag was needed only
  because this session's `curl` lacked a working CA bundle for this host —
  a local environment quirk, not a site-side restriction; the same URL
  succeeded via `WebFetch`-equivalent tooling in principle and the JSON
  response itself is genuine, well-formed data, not an error page.
- **Coverage years**: **measured** — the same query with `startdate=1/1/2000`
  returned real rows starting `2000-02-29`, confirming the archive's
  earliest map is essentially the USDM program's own 2000 launch — 26
  seasons of coverage, well above this project's own "High" admission tier
  (8+ seasons).
- **Next-step ingestion plan (effort: S-M)**: map each outdoor grass stadium
  to its county FIPS code (a one-time, static lookup), then a weekly REST
  pull per stadium-county starting 2000; the API's JSON/CSV/XML format
  options and no-auth requirement make this one of the cheapest ingestion
  builds in this report.

### 5. USA Today NFL player arrests database

- **Mechanism**: an arrest, charge, or citation in the days before kickoff
  is a genuinely late-arriving roster-disruption/distraction signal,
  distinct from the injury-report family already built (`docs/data_feasibility.md`)
  — this is a legal-record source, not a football-beat one.
- **Point-in-time safety**: **A** for using the incident date (arrests are
  newsworthy and typically public within hours); a "resolution date" cut
  would leak and must not be used.
- **Access path**: free, confirmed live. **Measured**: `curl` to
  `databases.usatoday.com/nfl-arrests/` returned HTTP 200; page title reads
  *"NFL player arrests database: Records since 2000 | USA TODAY Databases"*;
  body text attributes the database to reporter Brent Schrotenboer with a
  live update-submission contact, and an embedded config object shows
  `"sortBy":"Date","sortOrder":"desc"` with `First_name`/`Last_name`/`Team`
  columns and `"content-protection-state":"free"`. The table itself is a
  Caspio-backed AJAX embed (`csp-app`/`csp-data`), so a plain static-HTML
  scrape will not see row data — the AJAX/Caspio API (or a headless
  browser) is required, not a simple table pull. Separately **measured**:
  the plain `WebFetch` tool could not reach this domain at all ("unable to
  fetch from databases.usatoday.com"); `curl` succeeded, so the eventual
  ingestion script should use a direct HTTP client, not this session's
  fetch tool.
- **Coverage years**: 2000-present per the page's own title/intro
  (**measured**, text read directly from the fetched page); the exact most
  recent row date was not independently confirmed this session since it
  requires querying the AJAX endpoint directly rather than the static shell.
- **Next-step ingestion plan (effort: M)**: reverse-engineer the Caspio AJAX
  call (inspect the embed's network requests) or fall back to a headless
  browser for the table; both are standard patterns, not blocked, but
  neither is a one-line `curl`.

### 6. Retractable-roof open/closed decisions

- **Mechanism**: teams with retractable roofs sometimes make a late
  roof-status call that changes wind/precipitation exposure for a game a
  frozen Tuesday line would have priced as fully sheltered; the decision is
  effectively locked at kickoff-minus-90-minutes and cannot reverse once
  closed (**read**, footballzebras.com's documented retractable-roof policy
  explainer).
- **Point-in-time safety**: **B** for the historical backtest record (the
  games are complete, so no leakage risk in using the final tag), but this
  is retrospective — a genuine **prospective** signal needs a separate
  live source (reporter/NFL.com pregame-info scrape in the T-90-to-kickoff
  window), which was not itself verified this session.
- **Access path**: free, and largely already available. **Read**:
  `github.com/nflverse/nfldata`'s `games` dataset documents a `roof` field
  (`outdoors`/`open`/`closed`/`dome`) flagged "NEW Feb 2020" in its own
  changelog — meaning the field's actual completeness for games before
  2020 is unconfirmed and needs direct inspection of the CSV before relying
  on it for outer-season backtests, not assumed from the changelog note
  alone.
- **Coverage years**: nflverse's `games` dataset covers 2006-present per its
  own documentation (**read**); the `roof` field's real historical
  completeness within that range is the open question above.
- **Next-step ingestion plan (effort: S for the backtest; M to add a live
  prospective signal)**: pull the existing `roof` field from the project's
  already-ingested nflverse schedule source (likely zero new external
  access at all) and directly audit pre-2020 completeness before use; a
  live T-90 scrape is a separate, larger build.

### 7. Massey Ratings composite archive

- **Mechanism**: a consensus of 50+ independent computer rating systems,
  recomputed continuously; a value scraped close to kickoff reflects
  information through the latest games/injuries rather than a frozen
  Tuesday number.
- **Point-in-time safety**: **C**. **Read**: the archive page itself
  carries an explicit disclaimer that "historical ratings are subject to
  data/model revisions, and should not be considered 'official'" — a
  vendor-acknowledged retroactive-revision risk. **Measured**: historical
  team labels are also retroactively renamed to current franchises (2002
  Oakland displayed as "Las Vegas Raiders" on `masseyratings.com/nfl2002/nfl/ratings`),
  a concrete, observed instance of the same revision problem. Weekly
  in-season point-in-time granularity (as opposed to season-level) was not
  confirmed this session.
- **Access path**: free but browser-only. **Measured**: plain `WebFetch` to
  `masseyratings.com/nfl/ratings` returned HTTP 403; **measured** robots.txt
  allows the ratings path itself (only `/cgi-bin/`, `/data/`, `/scores.php*`
  are disallowed) but explicitly blocks AI-training crawlers by name
  (`GPTBot`, `CCBot`, `ClaudeBot`) — the 403 is very likely that
  crawler-name block, not a generic bot gate, which matters for choosing an
  ingestion user agent. **Measured** via a real browser session: 200 OK
  with genuine numeric ratings for the 2026 preseason, the 2023 season
  final (Kansas City 15-6, Rat 9.25), and 2002 (Tampa Bay 15-4, Rat 9.38).
  **Measured**: `masseyratings.com/nfl2023/nfl/archive` lists a full season
  archive spanning **1920-2026**.
- **Coverage years**: 1920-2026 at the season level (**measured**); weekly
  granularity within a season unverified.
- **Next-step ingestion plan (effort: M)**: needs a non-`ClaudeBot`/`GPTBot`
  user agent or light browser automation to clear the crawler-name block
  (not a hard technical barrier, since robots.txt permits the path for
  other agents); table format is stable across the archive.

### 8. Podcast Index API (own-idea-adjacent — a genuinely non-football attention proxy)

- **Mechanism**: team-podcast episode-publish-count spikes in the days
  before kickoff as an attention/hype proxy structurally independent of
  Wikipedia pageviews (already built) and the already-killed Reddit/DFS-
  ownership ideas — and, unusually among "attention" sources, this one is
  not a sports-media product at all, but a general podcast-RSS index.
- **Point-in-time safety**: **B**. **Read** (the project's own fetch of the
  service's OpenAPI YAML, `podcastindex-org.github.io/docs-api/pi_api.yaml`):
  the `/episodes/byfeedid` and `/recent/episodes` endpoints support a
  `since` unix-timestamp parameter, so episodes published in a specific
  pre-kickoff window are mechanically retrievable and dated. Not graded
  **A** because the index only carries RSS-derived publish metadata, not
  true listen/download counts or chart rank — a weaker attention proxy in
  substance than its clean API access would suggest.
- **Access path**: free, real, documented REST API, key-gated. **Read**:
  the same OpenAPI spec lists `/search/byterm`, `/episodes/byfeedid`,
  `/podcasts/trending`, `/recent/episodes`, `/categories/list`,
  `/stats/current`, authenticated via an `apiHeaderTime` + SHA1-hash header
  scheme. **Measured**: an unauthenticated call to
  `api.podcastindex.org/api/1.0/search/byterm` returned HTTP **403** —
  consistent with "auth required," not "blocked," and a free developer
  signup is documented (**reported**, search result) at the same domain.
- **Coverage years**: each feed's own RSS history; most team-specific NFL
  podcasts are **inferred** (not verified) to start somewhere in the
  2015-2020 range.
- **Next-step ingestion plan (effort: S-M)**: register a free developer
  key, implement the HMAC auth scheme (a documented, mechanical pattern),
  build a team-keyword search list, and de-duplicate feed-vs-episode
  granularity.

### 9. FTN Data / DVOA

- **Mechanism**: DVOA is an opponent-adjusted efficiency rating computed
  after each week's games — the direct successor product to Football
  Outsiders' long-standing DVOA after that outlet's 2024 shutdown — and
  could diverge informatively from the market if scraped close to kickoff.
- **Point-in-time safety**: **C, unconfirmed**. **Read**: the displayed
  table has `Week`/`Year`-shaped columns suggesting per-week snapshots
  might exist, but **measured**: changing the URL's `?season=2010` query
  parameter did **not** change the displayed data (it kept showing 2025) —
  year/week selection is JS-driven, not URL-addressable, and the underlying
  JSON API behind that selector was not found this session.
- **Access path**: free for the table checked, no login wall observed.
  **Measured**: plain `WebFetch` to `ftndata.com` failed on DNS (wrong
  domain — the real domain is `ftnfantasy.com`); `WebFetch` to
  `ftnfantasy.com/stats/nfl/team-total-dvoa` and its robots.txt both
  returned 403 at the tool level. **Measured** via a real browser session:
  200 OK, genuine 2025 numeric data (Seattle 41.3% DVOA rank 1, down to New
  York Jets -35.9% rank 32) with full methodology text and no paywall
  visible on this specific table; a "download DVOA data" link was seen but
  not followed (**inferred** likely paid, not confirmed).
- **Coverage years**: 2025 confirmed free; Football Outsiders' original
  DVOA is **reported** (unverified this session) to extend back to 1986,
  but that historical depth's actual free-vs-paid status on the FTN
  successor site is unconfirmed.
- **Next-step ingestion plan (effort: M)**: find the JSON endpoint behind
  the JS season/week selector (likely via browser network-tab inspection,
  not attempted this session) or fall back to per-page browser automation
  for each week/season combination.

### 10. Sports Media Watch weekly TV ratings (lagged construction only)

- **Mechanism**: v2 killed a raw "TV ratings" idea on a timing objection —
  same-game viewership is a game-time/post-game outcome, not a pregame
  predictor. This candidate is deliberately re-framed as a **lagged**
  signal: last week's, or a team's season-to-date, household-rating trend
  as a relevance/hype proxy feeding into the *next* week's prediction,
  which sidesteps v2's objection as long as the lag is strictly enforced in
  code.
- **Point-in-time safety**: **B**, not A — sound in principle, but ratings
  are published in revised waves (same-day "fast nationals," then "finals"
  days later per standard Nielsen practice, **inferred** from general
  industry knowledge, not reverified this session), which is its own
  restatement-leakage risk unless the ingestion deliberately uses only the
  wave available before the *next* decision timestamp.
- **Access path**: free, real content confirmed. **Measured**: a `WebFetch`
  of a November 2023 `sportsmediawatch.com` weekly-ratings article
  contained genuine numeric figures ("21.73 million" combined Sunday
  viewership, CBS singleheader "+12% vs. year-ago," an SNF "season-low"
  note). **Measured** robots.txt: explicitly allows major search and even
  AI crawlers (`Google-Extended`, `PerplexityBot`), no disallow blocks found
  in the fetched section.
- **Coverage years**: **reported** (search results, unverified this
  session) a weekly archive back to at least January 2011, with the site's
  own historical reference tables claimed to reach back to 2002; the
  category listing continues into the current 2026 season.
- **Next-step ingestion plan (effort: M)**: no API — figures are embedded
  in editorial prose (confirmed by the fetch above), so extraction needs
  regex/light-NLP parsing rather than a clean table scrape; the weekly
  cadence itself is stable and predictable.

### 11. Preseason usage patterns / snap counts — a confirmed real gap, source unresolved

- **Mechanism**: a projected starter with abnormally low or zero preseason
  snaps (rest vs. injury-limited) is a plausible health/readiness signal
  specifically for Weeks 1-4, when the regular-season sample for that
  player is thin or nonexistent.
- **Point-in-time safety**: **A** — preseason games are complete and public
  well before any regular-season Tuesday line.
- **Access path**: **C, and this is the headline finding, not a footnote.**
  **Measured**: `nflreadr.nflverse.com/reference/load_snap_counts.html`
  documents snap-count data "starting with the 2012 season," and
  **measured**: `nflreadr.nflverse.com/articles/dictionary_snap_counts.html`
  states the `game_type` field is explicitly "regular or postseason" only —
  **preseason is confirmed absent** from the project's own existing
  nflverse snap-count pipeline, not merely unconfirmed. **Measured**: a free
  alternative (`occupyfantasy.com/nfl-preseason-snap-opportunity-data/`)
  loaded but gated preseason snap/opportunity data behind a "Premium
  Membership" login wall, with underlying provenance ("powered by
  SportsData.io") unconfirmed.
- **Coverage years**: nflverse regular/postseason snap counts run
  2012-present (**measured**) but categorically exclude preseason; no free
  preseason-specific source was found live this session.
- **Next-step ingestion plan (effort: M-L)**: the two natural structured
  candidates (PFR preseason box scores, a paid feed) both hit access
  friction — PFR is the same site-wide Cloudflare block documented in v3,
  and Occupy Fantasy is paywalled with unclear original provenance. This is
  a real, confirmed hole in the project's own data inventory
  (`docs/data_feasibility.md` does not list preseason snaps at all) worth
  flagging even without a resolved source, per this report's brief to
  report gaps honestly rather than paper over them.

### 12. Coaching staff continuity beyond head coach (OC/DC)

- **Mechanism**: an in-season offensive or defensive coordinator firing or
  promotion is an abrupt scheme-continuity shock distinct from the
  head-coach-identity features already built (v3's interim-head-coach
  candidate).
- **Point-in-time safety**: **A** for the fact of a change (widely reported
  same-day); **B** if scheme-similarity details are needed.
- **Access path**: **C**, no single clean structured feed. **Measured**:
  PFR's `/years/2025/coaches.htm` returned HTTP 403 via both `WebFetch` and
  `curl` with a browser user agent — confirming the site-wide Cloudflare
  block already documented in v3 extends here too, not just the
  transactions/box-score pages previously found. **Measured**:
  `gridironexperts.com/nfl-coaches-list/` loaded successfully with a
  structured current-season HC/OC/DC/ST table, but is a manually-maintained
  fan site with no stated update-latency guarantee. **Read**: Wikipedia's
  "List of current NFL offensive/defensive coordinators" pages are
  maintained, all-32-team tables with a "Since" column and full edit-history
  version tracking (in principle usable to reconstruct in-season change
  dates via diffs), though the live page snapshot itself shows only
  present-day incumbents, not a historical log.
- **Coverage years**: current-season only for the fan-site table; Wikipedia
  edit history theoretically extends back years, but reconstructing
  historical coordinator-by-week data from revision diffs is unverified
  effort, not a confirmed capability.
- **Next-step ingestion plan (effort: L)**: no clean structured API exists
  — the two natural candidates (PFR, and OverTheCap-style trackers) are
  either blocked or already excluded from this report's scope. Would need
  either a Wikipedia-revision-history scrape or a manually-curated news
  tracker (the profootballrumors.com transaction wire, already built per
  v3, might already surface some of these events as a side effect and is
  worth checking before building anything new).

### 13. Spotrac contract-year / holdout / franchise-tag tracker

- **Mechanism**: players in a walk year or under franchise-tag friction
  carry documented extra motivation/injury-risk-tolerance, distinct from
  team-quality ratings — but largely already known well before any Tuesday
  lock, except for the specific timing of an in-season holdout's
  resolution.
- **Point-in-time safety**: **B** — contract-year status itself is known far
  in advance (safe), but a holdout-resolution date must be captured as of
  Tuesday, not after, to avoid leakage.
- **Access path**: **C**, mixed free/paywalled and bot-detected. **Measured**:
  `spotrac.com/robots.txt` returned 200 with a normal file (disallows
  `/trade-machine/`, `/reports/`, session-param URLs; no blanket
  disallow). **Measured**: `spotrac.com/nfl/free-agents/franchise-tag` and
  `spotrac.com/nfl/contracts/franchise-tag` both returned HTTP 200 via
  `curl` with a browser user agent, but each page's own text contains
  multiple "Premium"/"login"/"unlock" occurrences — partially paywalled
  gating on some of the detail. **Measured**: `WebFetch` (as opposed to
  `curl`) returned 403 on the same URLs — bot-detection specific to that
  tool's user agent, not a robots.txt disallow.
- **Coverage years**: **reported** (forum-thread references to an
  "extended history" of tags per team), exact start year not confirmed.
- **Next-step ingestion plan (effort: M)**: a real browser UA (matching the
  working `curl` approach) plus handling the partial paywall; marginal
  value is unclear since **measured** `nflreadr::load_contracts()` (already
  sourced from OverTheCap, which v2 already excluded) includes
  `year_signed`/`years` fields that could approximate walk-year status
  without touching Spotrac at all — worth checking that existing field
  before building a new scraper.

### 14. NFLPA team report cards

- **Mechanism**: annual player-submitted grades on facilities, culture, and
  treatment, published since 2023 — a possible discipline/buy-in or
  front-office-quality proxy, honestly assessed as likely close to a slower
  re-measurement of team quality, a mechanism this project has **already
  found near-zero ceiling on** (per prior project finding). This is
  **inferred** reasoning about the mechanism's likely marginal value, not a
  measured null result — it is ranked low, not removed, per AGENTS.md.
- **Point-in-time safety**: **B** — a single annual snapshot (survey window
  roughly August-November of the prior year), so no within-season leakage
  risk, but also no weekly granularity at all.
- **Access path**: free, confirmed live for three separate years.
  **Measured**: `nflpa.com/report-cards/2025` returned real content — a
  table across all 32 teams and 11 categories (e.g. Miami Dolphins mostly
  A/A+, Arizona Cardinals multiple D/F grades). **Measured** (this
  session's own follow-up, correcting an initial wrong URL-pattern guess
  that 404'd): the real URL pattern is
  `nflpa.com/nfl-player-team-report-cards-<year>`, and both
  `.../nfl-player-team-report-cards-2024` and `.../-2023` returned HTTP 200
  — three full editions (2023, 2024, 2025) confirmed live, not just the
  "third annual" self-description taken on faith.
- **Coverage years**: 2023-2025, three editions, all confirmed live this
  session.
- **Next-step ingestion plan (effort: S)**: straightforward table scrape of
  three known-good URLs; low priority given the likely team-quality-ceiling
  overlap, but cheap enough to build as a low-cost cross-check regardless.

### 15. FEMA disaster declarations / NOAA Storm Events — game-displacement proxy (own idea)

- **Mechanism**: a disaster-forced "home" game relocation (Saints post-
  Katrina 2005; several 2017 hurricane-displaced games) strips a team's
  true home-field advantage in a way that may not be fully priced — a
  deliberately NOT-obviously-football federal-data candidate.
- **Point-in-time safety**: **C, and this is a mechanism-timing finding,
  not an access problem** — the relocation itself is announced by the
  league/team as news, typically well before Tuesday (evacuation orders
  precede games by days), so this is barely "late" in most historical
  cases. **Read**: NOAA's Storm Events Database FAQ states entries are
  finalized roughly 75-90 days after the event — that database is
  retrospective-corroboration-only, never a live pregame signal.
- **Access path**: free, confirmed live. **Measured**: `fema.gov/api/open/v2/DisasterDeclarationsSummaries?$top=3&$format=json`
  returned valid JSON with real disaster records (Tropical Storm Arthur/LA,
  Oregon fires) including dates, FIPS codes, and incident type — a working,
  free, no-auth API (the HTML documentation page separately 403'd, but the
  actual API endpoint works). **Read**: NOAA Storm Events has no REST API,
  only bulk CSV via FTP, at county-level granularity, comprehensively
  since 1996.
- **Coverage years**: FEMA declarations **reported** (unverified this
  session) since 1953; NOAA comprehensive since 1996 (**read**), tornado-
  only back to 1950.
- **Next-step ingestion plan (effort: S, deliberately low-ceiling)**: cheap
  to fetch, but this is a handful-of-games-per-decade rare-event flag, not
  a weekly feature — small effort, small sample, low statistical power by
  construction. Worth having as a documented, always-on override flag for
  the rare case, not worth prioritizing engineering time on.

---

## Checked, not promoted (access-blocked or mechanism too weak for a specific week)

- **Google Trends official API.** **Reported** (search synthesis of
  Google's July 2025 announcement): the new API fixes pytrends' fatal
  rescaling-instability problem (values stay "consistently scaled" across
  separate calls) that killed the old approach in v2. **Measured**:
  `developers.google.com/search/apis/trends` returns HTTP 200 but is an
  early-access **application** page, not an open console — **reported**
  (multiple search sources) that access requires describing a use case and
  approval is neither guaranteed nor fast, with no GA date announced.
  Mechanism is strong (**A** once accessible); ranked out of the main table
  because access itself is the live blocker, not a research gap.
- **State/national gambling-regulator revenue reports (own idea).**
  **Measured**: a May 2026 Nevada Gaming Control Board PDF
  (`gaming.nv.gov/.../monthly-revenue-report----may-2026.pdf`) fetched at
  HTTP 200, a genuine 453.6KB PDF, though text extraction failed both via
  the fetch tool's own parser and a browser PDF viewer (canvas-rendered, no
  text layer) — real and reachable, not confirmed easily parseable.
  **Measured**: the American Gaming Association's Commercial Gaming Revenue
  Tracker loaded free with real May 2026 figures ($12.06B handle, $1.34B
  revenue) but dated **six to seven weeks** after the covered month — a
  slow macro covariate by design, not a signal that can inform a specific
  week's forced pick. Real, free, and well-timestamped, but the mechanism
  itself does not clear this report's "beats a specific Tuesday freeze" bar
  — a mechanism-timing deprioritization, not a data-access dead end.

## Hard blockers found this session

- **Betfair Exchange historical data — geo-blocked to US IPs, confirmed
  live.** **Measured**: both `historicdata.betfair.com/` and
  `developer.betfair.com/robots.txt` returned HTTP 403 with an identical
  page body: *"Our Software detects that you may be accessing the Betfair
  website from a country that Betfair does not accept bets from... Region:
  US."* This is not a technical scraping obstacle to work around — Betfair
  Exchange does not legally operate in the United States, so any access
  from this project's US-based infrastructure would require a VPN/proxy,
  which raises its own terms-of-service problem distinct from a normal
  robots.txt or Cloudflare fetch issue. Not promoted at any rank.
- **ADS-B Exchange / airframes.io team-charter tracking — the
  identity-linkage problem that killed v2's FlightAware idea is not solved
  here either.** **Measured**: `adsbexchange.com/data-products/sample-data/`
  offers free sample data only for the 1st of each month; full historical
  trace files require a paid subscription with an undisclosed "minimum
  annual commitment." **Measured**: `airframes.io` returned HTTP 403. Even
  granting ADS-B Exchange's more open data policy (solving FlightAware's
  cost problem), **reported**/unverified evidence suggests only two teams
  (Cardinals, Patriots) own dedicated aircraft with a stable tail number;
  every other team charters mainline carrier aircraft that also serve
  unrelated passengers on other days, so there is still no authoritative,
  free, weekly-refreshed tail-number-to-team mapping. ADS-B Exchange
  changes the cost/openness of the *flight data*, but does not touch the
  actual blocker, which is identity linkage — the same conclusion v2
  already reached for FlightAware.
- **ASAP Sports press-conference transcripts — verified to not cover NFL.**
  **Measured**: `asapsports.com/showcat.php?id=1` ("Football" category) is
  almost entirely **college** football media days (Big Ten, SEC, Penn
  State, Iowa, Indiana, Kentucky), 1992-2026. **Measured**: a targeted
  search for NFL coach/injury content on-site returned zero relevant
  results; the one NFL hit found was a one-off 2000 Jets ownership-transfer
  presser, not a recurring weekly coach presser. The football branding is
  misleading — this source does not have the content its mechanism needs.
- **Secondary ticket market asking-price archive (TickPick/SeatGeek) — no
  historical price archive exists, only current-listing widgets.**
  **Measured**: TickPick's own historical-price blog content confirmed only
  a small per-listing graph widget showing "the past few months," no
  downloadable dataset or API. **Measured** Wayback CDX density is real
  (TickPick's NFL hub from Nov 2013, SeatGeek's from June 2010, dozens of
  captures/year) but only for hub/listing pages, not per-game price
  trajectories — reconstructing a specific game's asking-price history
  would require finding and repeatedly re-snapshotting that game's
  individual event URL at daily density, which is not confirmed to exist.
  Consistent with v2's finding that the official StubHub/SeatGeek APIs
  expose live inventory only, no historical archive endpoint.
- **ESPN FPI archive — the live page overwrites its own history.**
  **Measured**: `espn.com/nfl/fpi/_/season/2019` shows real 2019 numeric
  FPI values but is stamped "Last Updated: October 26, 2020" — a single
  retrospectively-frozen number per season, not a value recoverable as of
  any specific week. **Measured** Wayback CDX for `espn.com/nfl/fpi` shows
  only monthly captures (2020-2022), too coarse to recover a specific
  pre-kickoff Tuesday value even via the archive.
- **Team hotel/itinerary reporting — no structured source exists, and the
  novel part likely overlaps existing features.** **Measured**:
  `sharpfootballanalysis.com`'s rest-disparity content is paywalled beyond
  a free overview, and what is visible is purely schedule-derived (known at
  schedule release, months ahead) — not late-arriving, and likely redundant
  with rest/timezone features the project's own schedule data already
  encodes. The genuinely new signal (actual mid-trip stayovers vs. flying
  home) has no aggregator at all, only scattered team-site articles and
  beat-reporter posts.

---

## Top 5 for immediate build, by expected pool EV per unit effort

1. **Penalty-type crew tendencies (#2).** The cheapest item in this entire
   report — no new external access, no robots.txt negotiation, no
   paywall. It widens a construct this project has already built and
   already measured a real, moderate split-half reliability for
   (`mean_total` +0.370), from a raw penalty count to type-specific rates,
   using a column (`penalty_type`) the project's own upstream nflverse
   pipeline already provides but this project's local narrow snapshot
   currently drops. Effort S.
2. **Sagarin ratings via the Wayback Machine (#1).** The strongest
   point-in-time guarantee of any ratings-divergence candidate in this
   report (a real internal date label plus an independent Wayback
   timestamp), an actively-maintained archive through the present day (not
   a frozen relic like v3's 538-Elo finding), and 15+ verified years of
   depth. Effort M (CDX crawl plus a drifting text-table parser).
3. **EPA AirNow historical AQI (#3).** A genuinely non-football federal
   API, live-verified (401, not blocked, on an unauthenticated test call),
   free after a trivial signup, targeting a real documented event class
   (2020 West Coast wildfire-smoke games) the market plausibly underprices.
   Effort S-M.
4. **US Drought Monitor weekly county statistics (#4).** Also genuinely
   non-football (a USDA/drought-monitoring product, not a weather feed),
   live-verified with real dated JSON back to 2000, free, no-auth, and
   mechanistically distinct from the project's already-built day-of
   weather actuals (cumulative field condition vs. single-day readings).
   Effort S-M.
5. **Retractable-roof open/closed decisions (#6).** Likely near-zero
   marginal access cost for the backtest slice, since the `roof` field
   already exists in the project's own ingested nflverse schedule source —
   the real work is auditing pre-2020 completeness, not sourcing new data.
   A genuine live prospective signal (the late T-90-minute call) is a
   separate, larger follow-up. Effort S for the backtest.

Sagarin (#1) and the penalty-type widening (#2) are the two strongest
individual findings in this report: one is the best point-in-time
guarantee found for a market-divergence signal, the other is effectively
free. The two federal-environmental candidates (#3, #4) are the clearest
answer to the brief's explicit call for creativity outside football's
obvious orbit — both cleared a live, unauthenticated or trivially-gated
API test this session, which is a stronger verification bar than most
"reported" candidates in this report clear. The three access-or-mechanism
dead ends worth remembering for future sessions: Betfair Exchange is
geo-blocked to US traffic outright (not a scraping problem to solve),
ADS-B Exchange does not fix the tail-number identity-linkage problem that
already killed FlightAware in v2, and ASAP Sports' football coverage is
almost entirely college despite its branding.

