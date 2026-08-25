# Data source scout v2

Scope: sources NOT already in the repo (nflverse, Odds API snapshots, CFBD,
Wikipedia pageviews, weather realized/forecast, officials — all excluded by
brief). Every claim tagged **measured** (fetched this session, command/URL
given), **read** (page opened and read this session), **reported** (search
snippet/vendor claim, unverified), or **inferred** (my reasoning). Mechanism
filter applied per AGENTS.md: kill only for no mechanism or refuted access,
never for "effect probably small."

## Ranked candidates (top 10)

| # | Source | Mechanism | Point-in-time safety | Depth | Access | Verified? |
|---|---|---|---|---|---|---|
| 1 | sportsbookreviewsonline.com historical odds archive | opener is the project's actual target metric | opener/close columns, stage known; exact quote timestamp unknown (same caveat as existing free sample) | 2007-08 → 2021-22 in full, populated; 2022-25 pages exist but are empty (0 rows) — **measured** | free, static HTML, scrape | **measured** — curl pulled real Date/Rot/VH/Team/1-4th/Final/Open/Close/ML/2H tables for 2007-08 (535 rows) and 2021-22 (571 rows) |
| 2 | Bet Labs / Sports Insights historical bet% | direct "fade the public" signal — the single most on-target source in the whole scout | real bets at contributing books, timestamped by design (vendor claim) | back to 2003 (vendor claim) | $20-30/mo or $299 lifetime, no free tier found | **reported** only — paywalled, could not verify actual columns |
| 3 | GDELT DOC 2.0 API | raw press attention/tone by entity — "opener is soft to attention" mechanism, distinct from Wikipedia pageviews already in repo | article `seendate` is a real publish timestamp, provably ≤ any decision cutoff | 2017+ (DOC 2.0); GKG tone back to 2015 (reported) | free, no key, REST+BigQuery | **measured** — live query returned real dated articles, but a bare team-name query is noisy (Taylor Swift/Kelce stories) and needs domain/theme filtering |
| 4 | State sports-betting legalization rollout dates | natural experiment: public-money composition and opener softness plausibly shift as a state's market matures/professionalizes | fully known in advance (law/launch dates are historical record) | 2018 (PASPA repeal) → present, all 38+ states | free, AGA "State of the States" annual PDF report | **reported** — found the report exists (pp. 23-24 of the 2025 edition per search snippet), did not open the PDF myself |
| 5 | Kalshi / Polymarket historical NFL prediction-market prices | distinct trader population from sportsbook bettors — a genuine second market, not a repackaged one | on-chain/exchange timestamps are exact | NFL markets only liquid ~2023+ — shallow | Kalshi has a free public REST API; full historical backfill needs a paid aggregator (ScoreTape, Bitquery) | **reported** — not fetched; access mechanics described by multiple vendor pages, not tested live |
| 6 | Delphi Epidata FluView API (CMU) | speculative: crowd energy / subclinical player illness by home-market flu load | **best-in-class** — every record carries `release_date`/`issue`/`lag`, i.e. it is natively versioned for point-in-time queries | national data since ~2009 (measured range tested), state-level confirmed | free, no key required | **measured** — live query for `nat`, epiweeks 200940-45 returned versioned rows; a `pa` (state) query also returned real data |
| 7 | OverTheCap historical dead money / cap space | in-season motivation ("tanking") proxy, arguably distinct from static team quality | cap charges are known publicly at start of league year, pregame-safe | multi-year history exists per team (search-confirmed Saints 2015-18 figures) | free, HTML only, no documented API/CSV | **reported** for depth; **read** confirms the "Historical Team Spending" page and per-team dead-money figures exist, page-by-page scrape needed |
| 8 | Kickoff sun position / glare | referee/QB-passing anecdotal mechanism (AT&T Stadium cases); NFL's own internal study found no broad drop-rate effect elsewhere | perfect — pure astronomy, no future information possible | 100% of games, any date, forever | **zero-cost, no external source at all** — computable from stadium lat/lon + kickoff time already in schedule data | **measured** — ran a NOAA-style solar-position formula in `.\.tools\uv.exe run --no-sync python`, got a plausible altitude/azimuth for AT&T Stadium |
| 9 | Gridiron Uniform Database (black-uniform referee bias) | real academic mechanism: Frank & Gilovich (1988) found black NFL/NHL uniforms draw more penalties, replicated in lab referee-judgment experiments | game-by-game uniform choice is known pregame (mostly home-team elected, some league-mandated alternate weeks) | claims full NFL coverage since 1920 (reported) | free, but it is a Blogspot + fan forum — no API, heavy manual/scrape effort | **read** — confirmed the site and its game-by-game claim exist; did not extract data |
| 10 | Madden ratings historical archive | reputation-anchoring: markets may over-weight a once-a-year, stale public rating vs. current-season form | ratings are published once per season, before most games — pregame-safe with a lag caveat | full year-by-year since 1992 across several free trackers (nfldraftbuzz, GitHub `theedgepredictor/nfl-madden-data`, LeagueStation) | free | **read** — sites confirmed to exist and describe full historical coverage; risk this just re-measures team quality already shown bounded near zero |

## Top 5, with fetch evidence

**1. sportsbookreviewsonline.com historical odds archive.** This is the
classic academic-backtest odds source, now sitting behind an offshore-book
affiliate-marketing redesign — the domain looks dead in a browser skim, which
is probably why nobody has re-scouted it. It is not dead: **measured**, `curl
-A "Mozilla/5.0" https://www.sportsbookreviewsonline.com/scoresoddsarchives/nfl-odds-2007-08`
returned HTTP 200 with a real per-game table (`Date, Rot, VH, Team, 1st, 2nd,
3rd, 4th, Final, Open, Close, ML, 2H`), 535 `<tr>` rows ≈ 267 games, matching
a 2007 regular season. The same check on `nfl-odds-2021-22` returned 571
rows. I also checked `nfl-odds-2022-23` through `nfl-odds-2024-25`: all
return HTTP 200 but **0** table rows — the archive stopped being populated
after 2021-22, confirming an incidental search-result claim that "the NFL
scores and odds archive will not be updated." Net new value: **13 seasons
(2007-08 through 2019-20) of opener+close data that predate the repo's
existing Odds API window (2020-2025)** — directly extends outer-season depth
for the project's own primary metric. Caveat (**inferred** from the classic
SBR format, not reverified live): the "Open"/"Close" numbers interleave
point-spread and total across a game's two rows by a known magnitude
convention, a widely-documented parsing quirk, not a blocker.

**2. Bet Labs / Sports Insights historical bet%.** This is the most directly
on-mechanism source in the whole scout — actual historical percentage of
bets/money by side, marketed as going back to 2003, "unlike other sites that
only show consensus data... taken from real bets placed at actual
sportsbooks" (**reported**, vendor copy). It also has independent academic
backing: Corey Shank's SSRN paper (**reported**, found via search, not read
in full) found bettor consensus above ~60% correlates with higher accuracy —
direct support for the "fade/follow the public" mechanism at the historical
level, not just anecdote. I could not verify the actual data (paywalled,
$20-30/mo or a $299 lifetime tier per search snippets) so this stays
**reported** across the board; it is ranked #2 despite that because mechanism
match is unmatched and cost is trivial relative to project value.

**3. GDELT DOC 2.0 API.** **Measured**: `curl
"https://api.gdeltproject.org/api/v2/doc/doc?query=%22Kansas City
Chiefs%22&mode=artlist&format=json&maxrecords=5&startdatetime=20240905000000&enddatetime=20240906000000"`
returned HTTP 200 with 5 real, dated articles. It is free, needs no key, and
is queryable by date range down to the article level with a genuine publish
timestamp (`seendate`), which is about as strong a point-in-time guarantee as
any source in this list. The catch, also **measured**: a bare team-name query
pulls entertainment noise (a Travis Kelce/Taylor Swift story, a Hallmark
movie about the Chiefs) alongside real football coverage — usable as an
attention-volume proxy (more noise ≈ more attention, arguably informative on
its own) but needs `domainis:`/theme filters before using tone/sentiment as a
feature.

**4. State sports-betting legalization rollout dates.** Not a per-game
feature by itself — a regime variable. Mechanism (**inferred**): as a state's
retail/mobile market matures (PASPA fell 2018; most states rolled out
2019-2023), the composition and sophistication of "the public" backing a home
team plausibly shifts, which would change how stale or fresh any
public-betting-bias feature is over time — this is a modifier for candidate
#2, not a standalone signal. **Reported** only: search results confirm the
AGA's annual "State of the States" report carries a state-by-state table
(cited at pp. 23-24 of the 2025 edition) but I did not open the PDF.

**5. Delphi Epidata FluView API (Carnegie Mellon).** **Measured**: `curl
"https://api.delphi.cmu.edu/epidata/fluview/?regions=nat&epiweeks=200940-200945"`
and a second call with `regions=pa` both returned real, versioned ILI
records — critically, each row carries `release_date`, `issue`, and `lag`
fields, meaning the API natively answers "what did this look like as of
date X," which is a stronger point-in-time guarantee than almost any other
source here, including several already in the repo. The mechanism is the
weakest link, not the data: illness-by-market as a crowd-energy or
subclinical-player-illness proxy is speculative and I would flag it
**inferred**/exploratory rather than promising on priors. Included at #6
anyway per AGENTS.md — never kill a real, cheap, well-timestamped candidate
for "effect probably small," and this one is nearly free to test.

## Verified dead ends

- **StubHub / SeatGeek APIs** — official developer docs (**read**) describe
  live inventory/purchase endpoints only; no historical secondary-price
  archive endpoint found in either. Access blocked, not mechanism-killed.
- **pytrends / Google Trends** — the pytrends GitHub repo was archived
  (dead) in April 2025 (**reported**); independent sources document that
  Google rescales/rounds older data and reindexes on re-pull, which breaks
  the reproducible-timestamp requirement even though the underlying search
  behavior was real-time-knowable. Deprioritized on reliability, not
  mechanism.
- **Nielsen local TV ratings** — mechanism is backwards: ratings measure
  viewership *during* the broadcast, i.e. they are a game-time/post-game
  outcome, not a pregame predictor (**inferred**). No free bulk historical
  API surfaced either. Only a lagged "last week's rating → this week's hype"
  version would even have a mechanism, and no free archive supports that.
- **FlightAware AeroAPI for team charters** — historical flight data exists
  back to 2011 (**reported**, vendor page), but there is no public
  tail-number-to-team mapping, and enterprise pricing references a
  $64,000/month tier before volume discounts (**reported**). Solving the
  identity-linkage problem alone would be its own project. Killed on
  access-mechanics/cost, not mechanism (schedule-derived timezone/rest
  features already cover most of this mechanism for free).
- **Yahoo Pick'em pick distribution** — **measured**: `curl
  ".../pickem/pickdistribution?week=1&season=2023"` returned HTTP 200 and a
  ~1MB page, but the only percentage-looking values in it were CSS gradient
  stops, not real pick data — the season/week query params did not visibly
  change server-rendered content in a plain fetch. Likely needs an
  authenticated session or client-side JS the fetch tool doesn't execute.
  Unverified/likely blocked, not confirmed dead.
- **ESPN Pigskin Pick'em** — no public historical pick-percentage archive
  surfaced in search; only current-season play pages.
- **DFS ownership % (RotoGrinders / FantasyLabs)** — search results
  (**reported**) note RotoGrinders' public API now returns 403 (access shut
  off); FantasyLabs has a historical ownership tool but export/API terms are
  unclear and likely paid. Not dead, but low-confidence access — deprioritized
  below GDELT for the same "public sentiment" mechanism.
- **Pushshift/Reddit r/nfl, r/sportsbook dumps** — the bulk archive genuinely
  exists (3.4TB on Academic Torrents, **reported**), but per-team NFL
  sentiment extraction from raw 20-year Reddit dumps is a large NLP lift for
  a diffuse, unvalidated mechanism when GDELT already offers a structured,
  queryable equivalent. Deprioritized on effort, not killed.

## Recommended top 3 for immediate ingestion

1. **sportsbookreviewsonline.com historical odds archive (2007-08 to
   2019-20).** Biggest, cleanest, free win, and it serves the project's
   actual primary metric (beat the opener) instead of a proxy. Next steps:
   scrape the 13 net-new season pages (static HTML, no rate limiting
   observed), decode the known interleaved spread/total row convention,
   normalize to the existing opener/close schema, then run the leakage /
   point-in-time audit the AGENTS.md required-audit checklist demands before
   any feature built on it counts as more than pipeline validation.
2. **GDELT DOC 2.0 API.** Free, live-verified, and the most direct hit on the
   angle the owner explicitly called out as most promising ("public
   attention/sentiment, the opener is soft"). Next step: build a
   disambiguated per-team weekly query (city + team name, domain-filtered to
   sports/news outlets, themes filter to exclude entertainment crossover
   noise like the Kelce/Swift hits seen in testing), backfill 2017-2025,
   and gate it on `seendate ≤ decision cutoff` in the leakage test.
3. **Delphi Epidata FluView (state-level, versioned).** Free, live-verified,
   and uniquely solves point-in-time safety by design (`release_date`/`issue`
   fields), which is rarer than the football mechanism itself. Next step:
   pull state ILI history 2010-2025 matched to each home team's market,
   test as a lagged crowd-energy covariate. Treat any result per the binding
   rule — an interval crossing zero is not grounds to close this, record it
   with `nfl-ats weak-signals record` and report `probability_positive`.

State legalization rollout dates (#4) is a strong complement once Bet Labs /
Sports Insights bet% (#2) is ingested — it's a regime variable, not a
standalone feature, so it should follow rather than lead.
