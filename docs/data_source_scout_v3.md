# Data source scout v3

Scope: sources NOT already scouted or built by `docs/data_source_scout_v2.md`
(SBR odds archive, Bet Labs, GDELT, state-legalization dates, Kalshi/
Polymarket, Delphi FluView, OverTheCap, sun position, uniform database,
Madden ratings, and that doc's dead ends), `docs/injury_news_sourcing.md`
(ProFootballTalk sitemap, already built), `docs/sbr_odds_archive.md` (already
ingested and validated), `docs/weather_forecast_sourcing.md`/
`docs/forecast_archive_build.md` (NOAA NDFD / GFS MOS, settled, skipped per
task brief), and `docs/attention_followup.md` (Wikipedia pageviews, proven).
Every claim below is tagged **measured** (fetched or run this session, exact
command/URL given), **read** (a docs/API-reference page opened this
session), **reported** (a search snippet or vendor claim, unverified), or
**inferred** (reasoning, not evidence). This session ran the verification
work through four parallel research agents plus direct live API testing by
the orchestrating session; one thread (referee-assignment timing / NFL flex
scheduling) did not return before this report's deadline and is marked
**PENDING** rather than guessed at.

## Ranked candidates

| Rank | Source | Mechanism (one line) | Point-in-time grade | Access grade | Verified today |
|---|---|---|---|---|---|
| 1 | Pro Football Rumors transaction-wire sitemap | roster moves/signings/IR/practice-squad news that firms up after the Tuesday lock, from a second outlet independent of the already-ingested PFT feed | B | A | **yes** |
| 2 | actionnetwork.com free public betting-percentage archive | fade/follow-the-public, the single most on-mechanism idea in v2, now with a FREE verified path (v2's Bet Labs was paywalled/unverified) | B | A | **yes** |
| 3 | The Odds API historical player-prop lines | player-level prop lines (yardage/reception O/U) encode availability and role information that can move independently of, and earlier or later than, the spread | A | B | **yes** |
| 4 | Ourlads.com historical depth charts via Wayback Machine | who is practicing/listed with the 1s, extending the project's own timestamped depth-chart coverage back from 2025-only to 2011+ | A | A | **yes** |
| 5 | Interim head coach games (PFR box scores + Pro Football Rumors' ready list) | documented motivation/effort discontinuity when a team fires its coach mid-season | A | C | **yes** |
| 6 | FiveThirtyEight NFL Elo / QB-Elo historical archive | independent, free, vendor-documented pregame-safe power rating that can diverge from the market line | A | B | **yes** (data recovered; source is dead) |
| 7 | NFL game attendance (PFR box scores) | requested by brief; mechanism assessed as weak — attendance is a distributed-tickets count mostly known in advance and plausibly already priced via home-field advantage | A | C | **yes** (data exists; mechanism weak) |
| 8 | Referee weekly-assignment archive (footballzebras.com) | extends the project's own referee-identity backtest depth (2010+ vs. nflverse's 2015 floor); timing does NOT clearly beat the Tuesday lock | C | A | **yes, reported-from-subagent** |
| — | NFL flex-scheduling announcement archive | primetime-flex decisions; timing appears to land BEFORE, not after, a Tuesday lock | — | — | checked, no mechanism found (see below) |

---

## VERIFIED TODAY

### 1. Pro Football Rumors transaction-wire sitemap

- **Mechanism**: roster moves — signings, cuts, injured-reserve placements,
  practice-squad elevations, trades — are exactly the kind of information
  that can arrive or firm up between a Tuesday-noon lock and kickoff. This
  is a second, independent outlet from the already-built ProFootballTalk
  ingestion (`docs/injury_news_sourcing.md`), so it adds coverage breadth
  and a cross-check, not a duplicate of existing work.
- **Point-in-time safety**: **B**, with a specific caveat found this
  session. `https://www.profootballrumors.com/sitemap.xml` (**measured**,
  HTTP 200) is a WordPress sitemap index of yearly chunks
  `sitemap-posttype-post.YYYY.xml`, 2014-2026. Its `<lastmod>` field is
  **not** a reliable publish-date proxy — **measured**, a 2015-09-23 article
  (`.../minor-nfl-transactions-9-23-15`) carries sitemap `<lastmod>
  2025-12-27`, while the article's own JSON-LD `datePublished` correctly
  reads `2015-09-23T16:47:54-05:00` (`dateModified` reads `2025-12-26`,
  evidence of a bulk site-wide retouch, likely a plugin/template refresh,
  that silently bumped old posts' `lastmod`). A third, untouched Jan-2015
  post had `lastmod` matching `datePublished` exactly, so the contamination
  is partial, not universal. **Net effect**: article URLs and a reliable
  article count come free from the sitemap, but true point-in-time dating
  requires one extra fetch per article for its JSON-LD `datePublished` —
  the same per-article-verification pattern the PFT ingestion already uses,
  not a new capability gap.
- **Access path**: free. `robots.txt` (**measured**) sets `Crawl-delay: 1`
  (friendlier than nbcsports' 10) and disallows only unrelated paths
  (`/wp-admin/`, `/search`, etc.).
- **Coverage years**: 2014-2026 confirmed (**measured**: `sitemap-posttype-
  post.2013.xml` and `.2010.xml` both return HTTP 200 but with an empty
  urlset — no earlier content). Per-year URL counts sampled directly
  (**measured**, `grep -c "<loc>"` on each chunk): 2014: 5,635; 2015: 5,858;
  2016: 6,166; 2018: 6,608; 2020: 4,915; 2022: 5,283; 2024: 5,498; 2026
  (partial): 3,925 — roughly 70,000+ articles across the full span.
- **Next-step ingestion plan (effort: S)**: clone
  `scripts/ingest_injury_news.py`'s structure for a
  `scripts/ingest_transaction_news.py` — fetch the sitemap index once, then
  each yearly chunk, filter to real transaction-keyword URLs, and fetch each
  matched article's JSON-LD `datePublished` (not `<lastmod>`) for the true
  timestamp. Low effort because the pattern, rate-limiting, and snapshot
  layout are all already proven and reusable from the PFT build.

### 2. actionnetwork.com free public betting-percentage archive

- **Mechanism**: bet%/money% splits by side are the direct "fade/follow the
  public" signal — v2 ranked this mechanism #2 overall but could only find
  it behind Bet Labs' $20-30/mo paywall, unverified. This session found a
  **free** path with real historical data.
- **Point-in-time safety**: **B**. The live page (**measured**,
  `actionnetwork.com/nfl/public-betting`) shows real bet%/money% splits
  (e.g., Raiders 39%→61% bets / 43%→57% money vs. an opponent; 49ers
  37%→63% bets / 19%→81% money — some deeper money% data gated behind
  "Action PRO"). Wayback CDX (**measured**) shows the earliest capture is
  **2018-09-02**, with roughly 15-45 captures per season through 2024 —
  denser than a monthly cadence but not guaranteed weekly for every
  game/week. A **measured** 2019-12-07 snapshot's raw static HTML contains
  "Public Betting" (17 hits), "Money %" (2 hits), "sharp/Sharp" (15 hits)
  plus real percentage figures, confirming the historical data is
  server-rendered and genuinely archived, not a JS-injected shell that
  Wayback only captured as empty markup.
- **Access path**: free scrape; `robots.txt` was not separately re-checked
  this session for actionnetwork.com specifically (flagged as an open item
  before building a scraper).
- **Coverage years**: 2018-2024+ via Wayback; prospective (2026+) needs an
  own weekly capture job since Wayback's own density has gaps.
- **Secondary options found, not primary**: `covers.com/picks/nfl`
  (**measured**) has cleaner, denser data (119 captures in 2023, 107 in
  2024, real moneyline consensus % confirmed in a 2023-08-19 snapshot) but
  only from 2023-02-01 onward — a strong complement for 2023+, not a
  substitute for actionnetwork's longer 2018+ history. `vegasinsider.com`
  (**measured**) has a much deeper Wayback archive (old URL scheme back to
  2005, hundreds of captures/year through 2012) but a **measured** spot
  check of a Jan-2019 snapshot found **no** percentage data at all (just
  layout CSS and a promo banner) — the deep archive exists but does not
  actually contain the target data, so it is not usable despite its age.
  VSiN was a **measured** dead end (nav-only page / an empty-body redirect
  on its data subdomain). `oddsshark.com/nfl/consensus-picks`, `betql.co`,
  `oddsassist.com`, `cleatz.com`, `wunderdog.com`, `streakforthecash.com`
  are **reported** only (found via search, not fetched) as other
  current-season-only free tools; not pursued once actionnetwork's
  historical depth was confirmed.
- **Next-step ingestion plan (effort: M)**: two parts — (a) backfill via
  Wayback CDX + snapshot HTML parsing for 2018-2024 (accept the ~15-45
  captures/season density, i.e., treat this as a "known public sentiment as
  of the nearest prior capture" feature, not a per-game-guaranteed one);
  (b) stand up a small weekly live-capture job (same Task Scheduler pattern
  as `scripts/odds_capture.ps1`) for 2026+ prospective coverage, since a
  frozen Wayback-only archive stops mattering the moment this project needs
  the current week. Supplement with covers.com for a denser 2023+ slice.

### 3. The Odds API historical player-prop lines

- **Mechanism**: a player's own prop line (e.g., rushing yards O/U, receptions
  O/U) is a direct market read on that player's expected role/health,
  distinct from and often timed differently than the point-spread market.
  Sudden appearance, disappearance, or a sharply reduced O/U number for a
  specific player is a real-world technique bettors already use to infer
  starter status before an official announcement — this is the exact
  "player prop lines encode player-level availability info" mechanism the
  task brief called out by name.
- **Point-in-time safety**: **A**, the strongest of any candidate in this
  report. **Measured** live this session (key retrieved from the Windows
  user-registry environment via the same lookup `scripts/odds_capture.ps1`
  already uses — not present in this session's plain shell `$env:`/`.env`,
  confirmed by checking both and finding neither set; the credential itself
  was never printed, only used and its value replaced with `***` in any
  logged output):
  - `GET /v4/historical/sports/americanfootball_nfl/events?date=2023-11-01T12:00:00Z`
    → HTTP 200, a real list of dated events (e.g., Pittsburgh @ Tennessee,
    commence `2023-11-03T00:15:00Z`), cost 1 request
    (`x-requests-remaining: 2872`, `x-requests-used: 97128` at time of test).
  - `GET /v4/historical/sports/americanfootball_nfl/events/{id}/odds?regions=us&markets=player_pass_yds&date=2023-11-01T12:00:00Z`
    → HTTP 200, real player-prop data: FanDuel and DraftKings both carried
    `player_pass_yds` Over/Under lines for Will Levis (Titans, 204.5/202.5)
    and Kenny Pickett (Steelers, 213.5/212.5), each with a **per-bookmaker,
    per-market `last_update` timestamp to the second** (FanDuel
    `2023-11-01T11:52:30Z`, DraftKings `2023-11-01T11:52:13Z` — the two
    books updated at different moments, which is itself informative). Cost:
    10 requests for this one market/event/snapshot combination
    (`x-requests-last: 10`).
- **Access path**: paid endpoint (**read**, the-odds-api.com docs: "only
  available on paid usage plans"), but already provisioned and live on the
  project's existing account (per user memory, a ~$30/20K-request plan) —
  no new signup or spend needed to start. Cost is real, though: a full
  weekly, multi-market backfill (pass yards, rush yards, receptions, etc.,
  times multiple books/regions) would consume quota fast at ~10 requests per
  market/event/snapshot; a budgeted sampling plan (e.g., one Tuesday-noon
  snapshot and one Thursday/Friday snapshot per game, a handful of markets)
  is required rather than a dense pull.
- **Coverage years**: **read** (the-odds-api.com docs): general historical
  odds since 2020-06-06 (already the basis of the project's existing paired
  archive), but player props/alternate lines/period markets specifically
  are available only **after 2023-05-03T05:30:00Z** — roughly 3.25 seasons
  (partial 2023 through the current 2026 season), which is below this
  project's own "High" admission tier (8+ seasons,
  `docs/data_feasibility.md`) and below "Medium" (5-7 seasons) — this is a
  **Low-tier, frozen exploratory candidate** by the project's own gates, not
  ready for broad tuning, though per AGENTS.md that is not grounds to
  dismiss it.
- **Next-step ingestion plan (effort: M)**: extend
  `src/nfl_ats/market_data.py`'s existing Odds-API adapter (currently
  `markets="spreads,h2h"` live-only, no historical/props calls at all —
  **measured**, `grep` found zero references to `player_prop`/`prop_line` in
  the repo) with a new historical-props puller, budgeted to a small number
  of markets and two snapshots per game per week, starting at the 2023-05-03
  floor. Treat the 3-season depth honestly as exploratory, not
  production-ready, per the admission-tier rules.

### 4. Ourlads.com historical depth charts via the Wayback Machine

- **Mechanism**: who is listed/practicing with the first team can shift
  after a Tuesday lock (an emerging starter, a medical elevation); this
  directly extends the project's own depth-chart source, whose only
  genuinely per-observation-timestamped data starts in 2025
  (`docs/data_feasibility.md`: "Low for retrospective estimation" before
  then).
- **Point-in-time safety**: **A** by a different mechanism than a normal
  "as-of" field — the Wayback crawl timestamp itself is proof the page's
  content was publicly visible by that date, regardless of whether the
  source page carries its own explicit timestamp.
- **Access path**: free. `robots.txt` (**measured**) does not block the
  relevant path. **Measured**: `ourlads.com/nfldepthcharts/depthchart/{ABBR}`
  has dense, gap-free Wayback coverage 2011-2026 — roughly 4-12 day-collapsed
  captures per year, every single year, for the Patriots (139 total captures
  sampled). A **measured** January 2015 snapshot rendered real player names
  (Edelman, Gronkowski), confirming genuine content, not a shell.
  ESPN's depth-chart URL was also checked (**measured**): three URL-scheme
  eras, sparse pre-2010 (only 10 total captures found for one team across
  multiple years), thin 2010-2016 (2-9/yr), dense 2016+ (39-175/yr, day-
  collapsed) — a real 2016-08-06 snapshot rendered genuine content (Brady,
  Gronkowski, "QB" labels). Recommendation: Ourlads for pre-2016 seasons
  (denser, more consistent), ESPN's modern URL as a post-2016 supplement.
- **Coverage years**: Ourlads 2011-2026; ESPN thin before 2016, dense after.
- **Next-step ingestion plan (effort: M)**: build a per-team, per-season
  Wayback CDX puller (reusable pattern, same CDX API this session already
  used to verify density) targeting Ourlads first, ESPN as a post-2016
  cross-check/gap-filler. Given ~4-12 captures/year, treat this as
  "known-as-of-the-nearest-prior-snapshot" state, not a guaranteed
  weekly-resolution feed — state that limitation explicitly in any
  downstream feature.

### 5. Interim head coach games

- **Mechanism**: a documented motivation/effort discontinuity in
  sports-betting research when a team fires its head coach mid-season and
  plays under an interim — distinct from, and not obviously subsumed by,
  the team-quality features already shown to be bounded near zero.
- **Point-in-time safety**: **A** — coach identity is public and discrete,
  known well before kickoff by construction (a firing is public news; there
  is no ambiguity about who is coaching a specific upcoming game).
- **Access path**: **C**, the one real friction point. **Measured**: both
  plain `curl` (two browser user-agents) and the WebFetch tool got HTTP 403
  on pro-football-reference.com's team page, box score, **and homepage** —
  a site-wide Cloudflare block for automated fetches (broader than the
  previously-known `/years/{Y}/transactions.htm` block from
  `docs/injury_news_sourcing.md`). **Measured** via a real browser session
  (claude-in-chrome): PFR loads fine after auto-passing the Cloudflare
  challenge; a real box score
  (`pro-football-reference.com/boxscores/201012260den.htm`) explicitly
  shows "Denver Broncos ... Coach: Eric Studesville" vs. "Houston Texans ...
  Coach: Gary Kubiak" — real, per-game, per-team coach attribution.
  Separately, **measured** via WebFetch on profootballrumors.com: a
  ready-made list of every NFL interim head coach and the exact date they
  took over, 2000-2025 (e.g., "Eric Studesville, Denver Broncos, replaced
  Josh McDaniels, Dec. 6" — independently matches the PFR box score found
  above), which likely avoids needing to scrape PFR game-by-game at all.
- **Coverage years**: 2000-2025 via the Pro Football Rumors list.
- **Next-step ingestion plan (effort: S)**: join the Pro Football Rumors
  interim-coach list (a single, small, one-time page fetch) directly onto
  the project's own `nflverse` schedule by team/date range — no PFR
  scraping needed for the primary build; treat the PFR box-score match as
  confirmation-only, to be revisited only if the ready-made list needs
  independent verification for a specific disputed case.

### 6. FiveThirtyEight NFL Elo / QB-Elo historical archive

- **Mechanism**: an independent, free, algorithmic power-rating system,
  explicitly documented by its own vendor as pregame-safe, that could
  diverge informatively from the Vegas line.
- **Point-in-time safety**: **A**. **Read** (the archived README/raw file):
  columns `elo1_pre`/`elo2_pre` are explicitly documented as "before the
  game" and `elo1_post`/`elo2_post` as "after" — pregame-safe by the
  vendor's own column contract, not an assumption this project has to make,
  back to 1920, including QB-adjusted ratings with named starting QBs.
- **Access path**: **B**, and this is the real catch. **Measured**:
  `github.com/fivethirtyeight/data/tree/master/nfl-elo` now contains only a
  README (no CSV); `nfl-forecasts` 404s entirely. **Measured**: the live
  data URL (`projects.fivethirtyeight.com/nfl-api/nfl_elo.csv`) now
  302-redirects to `abcnews.com/politics` — the data API is dead. **Measured**:
  a Wayback snapshot recovers the full file (17,380 rows), but its last row
  is **2023-02-12** (Super Bowl LVII) — frozen after the 2022 season, never
  updated for 2023-2026. A separate, still-live static file
  (`github.com/fivethirtyeight/nfl-elo-game/blob/master/data/nfl_games.csv`,
  **measured**, 16,811 rows) has a simpler non-QB Elo, 1920-2021 (frozen at
  Super Bowl LV).
- **Coverage years**: 1920 through 2021 (simple Elo) or 2022 (QB-Elo), then
  frozen — no current-season data exists anywhere, by any path.
  - **Next-step ingestion plan (effort: S for a backfill-only feature; L to
  make it live again)**: pull the frozen CSV as a static historical feature
  for backtests through the 2021/2022 seasons. Extending it to 2023-2026
  would mean independently reimplementing 538's Elo update algorithm from
  its documented rules — a real project, not a data-sourcing task; flag this
  honestly rather than quietly treating the frozen file as current.

### 7. NFL game attendance

- **Mechanism**: requested explicitly by the task brief ("attendance records
  ... encode demand/attendance/motivation"). Data exists and is free
  (**measured**, the same PFR box score above also shows "Attendance:
  73,691" in both a summary line and a Game Info box), but the mechanism is
  the weakest in this report: attendance is a distributed-tickets count,
  driven by a stable season-ticket base, and largely knowable well before
  kickoff — a genuinely incremental signal would need a specific
  discontinuity (e.g., an attendance-vs-capacity shortfall as a bad-weather
  or apathy proxy), not raw attendance level, since the spread already
  prices home-field advantage, team quality, and travel. This is
  **inferred** reasoning, not a measured null result, and per AGENTS.md is
  not grounds to kill the candidate outright — it is ranked low, not
  removed.
- **Point-in-time safety**: A for the underlying fact (attendance figures
  are historical record), but the mechanism's own pregame-availability
  story is weaker than every other candidate above.
- **Access path**: **C**, same Cloudflare gate as the interim-coach
  candidate (PFR box scores).
- **Incidental finding**: the same PFR box score also carries a free
  historical closing Vegas line and total (**measured**: "Vegas Line Houston
  Texans -2.5, Over/Under 49.5") — a possible free secondary cross-check
  against the project's existing close data, subject to the same
  browser-only access constraint. Not pursued as its own ranked candidate
  since the project's existing close archive is already stronger; noted for
  completeness.
- **Next-step ingestion plan (effort: S/M, low priority)**: only worth
  building if paired with a specific attendance-vs-capacity or walk-up-gap
  construction, not raw attendance; deprioritized below every source above
  given the weak mechanism.

---

## Referee assignment timing and flex scheduling (reported-from-subagent, folded in late)

A dedicated research thread investigated (1) footballzebras.com's weekly
officiating-crew assignment posts and (2) NFL flex-scheduling announcement
timing. This thread's own fetches are relayed here as **reported** (findings
from a subagent this session, not independently re-verified by the
orchestrating session) rather than **measured** by this document's author
directly; the subagent's own claims within its thread were, per its brief,
based on real fetches.

### 8. Referee weekly-assignment archive (footballzebras.com)

- **Mechanism**: crew identity for a game is public before kickoff by
  construction (this is not in question); the open question was whether it
  is knowable **before** the pool's Tuesday-noon lock, which would make it a
  genuinely late-arriving signal versus a same-time or already-priced one.
- **Point-in-time safety**: **C** — the timing evidence found does not
  support a "beats the lock" story. **Reported** (subagent fetch): Week 1
  2025 assignments were posted **Sept 2, 2025, a Tuesday**, and Week 17
  2015 assignments were posted **Dec 29, 2015, also a Tuesday** — the same
  day as the pool's Tuesday-noon lock, with no confirmed multi-day cushion
  before it in either sampled case. The site's own posts additionally flag
  that crews are subject to late substitutions, meaning even a
  Tuesday-posted assignment is not necessarily final. This weakens, rather
  than supports, the "late-arriving free edge" framing this report's brief
  was built around for this specific candidate.
- **Access path**: **A** — free, public, no API; **reported** (subagent
  fetch): the site's category archive goes back roughly to 2010 (spot-check
  around "page 26" of the category index), which exceeds the project's
  already-ingested `nflreadpy.load_officials()` table's 2015 floor
  (`docs/referee_battery.md`).
- **Coverage years**: ~2010-present (**reported**), vs. the existing
  official-of-record source's 2015 floor.
- **Assessment**: the genuinely new value here is **not** a late-arriving
  signal against the Tuesday lock (the timing evidence argues against that),
  but **extra backtest depth** for the referee-identity features the
  project already built in `docs/referee_battery.md` — potentially pushing
  that family's left-censoring problem (every referee active in 2015 reads
  as a false "rookie" in the current 2015-2025 source, per that doc) back by
  up to five more seasons. That is a real but different use case than this
  report's primary "beat the freeze" framing, and it was not independently
  re-verified by this session — treat as **reported**, not measured, until
  someone opens the actual archive pages directly.
- **Next-step ingestion plan (effort: S, reframed)**: not as a live-edge
  signal; instead, scrape footballzebras.com's 2010-2014 assignment posts
  specifically to extend `referee_battery.md`'s left-censored "rookie"
  population back further, then re-run that family's already-built cells
  over the longer window.

### NFL flex-scheduling announcement archive — mechanism does not clear the bar

- **Mechanism hypothesis**: primetime-flex decisions (moving a Sunday
  afternoon game to Sunday/Monday/Thursday night) might land after a
  Tuesday lock and be informative about a specific game's expected
  interest/rest profile.
- **Finding (reported, subagent fetch)**: `nfl.com/legal/flexible-
  scheduling-procedures` documents only the *current-season* rules, not a
  historical archive of past announcement dates: SNF (weeks 5-13) and MNF
  (weeks 12-17) decisions come **≤12 days** before the game; SNF (weeks
  14-17) **≤6 days**; TNF (weeks 13-17) **≤21 days**. Working through the
  arithmetic for a Sunday kickoff: even the tightest 6-day window lands on
  the **Monday** before that week's game, i.e. **before**, not after, the
  following Tuesday-noon lock for that same week's slate — the opposite of
  the "arrives after the freeze" pattern this report is built around.
  Wikipedia's Sunday Night Football results list documents which games were
  historically flexed back to 2006, but **has no announcement-date column**
  (**reported**) — reconstructing actual decision timestamps would require
  an article-by-article search, for a mechanism that the rules themselves
  already suggest lands before, not after, the lock.
- **Verdict**: **not promoted**. This is a case where the access question
  (a laborious historical reconstruction) is moot because the timing
  mechanism itself does not clear the bar this report is built around — not
  a data-access dead end, a **mechanism-timing** dead end, which is an
  admissible reason to deprioritize under this project's own rules (a
  refuted-timing story, not "effect probably small").

---

## Top 3 for immediate ingestion, by expected pool EV per unit effort

1. **Pro Football Rumors transaction-wire sitemap.** Lowest effort of
   everything in this report — it is a direct clone of the already-proven
   PFT injury-news ingestion pattern (same sitemap-index-plus-per-article-
   JSON-LD shape, same rate-limit discipline), adds a second independent
   outlet, and its mechanism sits squarely on the task's own structural
   insight: roster moves are exactly the kind of thing that can happen
   between a Tuesday lock and kickoff. Effort S.
2. **actionnetwork.com free public betting-percentage archive.** This
   converts v2's #2-ranked mechanism (fade/follow the public, previously
   paywalled and unverified) into a free, live-verified path with real
   historical depth to 2018 — the single biggest upgrade to an already-known
   high-value mechanism found this session. Effort M (Wayback backfill plus
   a small weekly live-capture job).
3. **The Odds API historical player-prop lines.** The strongest
   point-in-time guarantee of any candidate here (per-bookmaker, per-second
   `last_update` timestamps) and a direct hit on the brief's explicit
   callout that prop lines encode player-level availability information —
   already live-verified against the project's own paid account with quota
   to spare. Ranked third rather than first because the real coverage floor
   (2023-05-03 onward, ~3.25 seasons) is honestly a Low/exploratory tier by
   the project's own admission-tier rules, and per-call quota cost (10
   requests per market/event/snapshot) requires a deliberately budgeted
   sampling plan rather than a dense pull. Effort M.

Ourlads' Wayback depth-chart archive (#4) and the interim-coach signal (#5,
via the Pro Football Rumors ready-made list, avoiding PFR scraping
entirely) are both strong, low-friction follow-ups once the top three are
underway. The FiveThirtyEight Elo archive (#6) is a legitimate free,
pregame-safe backtest feature through 2021/2022 but cannot be extended to
the current season without reimplementing its algorithm from scratch, which
caps its near-term value. Attendance (#7) is the weakest mechanism in this
report and should not be prioritized without a specific
attendance-vs-capacity reformulation.
