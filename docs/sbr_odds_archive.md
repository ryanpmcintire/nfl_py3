# sportsbookreviewsonline.com historical NFL odds archive

Ingestion + validation of the free, public historical odds archive
identified in `docs/data_source_scout_v2.md` (candidate #1). Every claim
below is tagged **measured** (run this session, command/path given),
**read** (file opened this session), **reported** (the scout doc's claim,
not independently re-verified here), or **inferred** (reasoning, not
evidence). Code: `scripts/ingest_sbr_odds.py`. No experiment was run and
nothing was written to `registry/`, per this task's scope.

## 1. Access, provenance, and licensing

- **Measured** (this session): `https://www.sportsbookreviewsonline.com/scoresoddsarchives/nfl/nfloddsarchives.htm`
  lists exactly 15 NFL season links, `nfl-odds-2007-08` through
  `nfl-odds-2021-22`. All 15 returned HTTP 200 with a populated table.
  `nfl-odds-2022-23` also returns HTTP 200 but **0** `<tr>` rows (checked
  directly this session, independent of the scout's report of the same
  finding) -- the archive was not updated past the 2021-22 season.
- **Measured**: `robots.txt` at the site root is
  `User-agent: * / Disallow: /go/` plus `User-agent: * / Allow: /` -- no
  `Crawl-delay` directive, and `/scoresoddsarchives/` is not under the one
  disallowed path. No block was encountered or evaded; a 1.5s delay was kept
  between requests as a courtesy (no delay was required by the site).
- **Measured**: no `.xlsx` link exists anywhere on the index page or any of
  the 15 season pages (plain substring scan of every fetched HTML file for
  `.xlsx`). The task brief noted SBR historically also shipped `.xlsx`
  files; on the current redesigned site, HTML is the only form available, so
  HTML is what was ingested. This also matches the locked environment: none
  of `openpyxl`/`lxml`/`bs4`/`html5lib` are installed (**measured** via a
  plain `importlib.import_module` probe), so no new dependency was added --
  the parser uses only stdlib `html.parser.HTMLParser` plus `requests`
  (present in the environment, though not listed in `pyproject.toml`'s
  explicit dependency set).
- Access date: 2026-08-19 (UTC timestamps recorded per-file in
  `data/raw/sbr_odds/20260819T192226Z/manifest.json`). Public archive page,
  fetched for private research use only, matching this project's existing
  CFBD/nflverse precedent (`docs/data_feasibility.md`). Raw HTML is cached
  locally only (`data/raw/**` is gitignored, confirmed via `git status`
  showing nothing under `data/` after this session's fetch); no raw rows are
  republished anywhere in this repo, only the derived parquet used for the
  validation reported below.

## 2. What was ingested

One HTML `<table>` per season, header row `Date, Rot, VH, Team, 1st, 2nd,
3rd, 4th, Final, Open, Close, ML, 2H`, two data rows per game (`VH`:
`V`=away, `H`=home, `N`=neutral -- Super Bowl only, one game/season).
**Measured**: every season's row count reproduces known NFL scheduling
history exactly, which is a structural cross-check that the table is being
read correctly, not merely without exceptions:

| Seasons | Games/season | Why |
|---|---|---|
| 2007-08 .. 2019-20 (13 seasons) | 267 | 256 regular season (16-game slate) + 11 playoff games |
| 2020-21 | 269 | 256 regular season + 13 playoff games (first season with an expanded 14-team, 6-game wild-card round) |
| 2021-22 | 285 | 272 regular season (first 17-game slate) + 13 playoff games |

Total: **4,025 games** parsed across 15 seasons, written to
`data/processed/sbr_odds.parquet` (manifest:
`data/processed/sbr_odds.manifest.json`).

### Parsing decisions

- **Spread/total split** (**inferred**, the classic documented SBR
  convention, reproduced correctly on every hand-checked example this
  session): of a game's two same-column values, the one with the smaller
  *magnitude* is the point spread (favored team's row), the larger is the
  total; `"pk"` parses to 0 (pick'em).
- **Bug found and fixed this session**: the 2021-22 season's Open column
  switches to an explicit signed convention for one whole week (11 games
  dated `"1114"`, e.g. raw Pittsburgh Open `"-9"` vs Detroit Open `"44"`)
  plus 2 isolated rows elsewhere -- 13 of 570 rows, all in 2021-22, zero in
  every other season (**measured** via a full scan of all 8,050 Open/Close
  cells). Comparing the *signed* values directly flips the spread's sign
  (`min(44, -9) = -9`, the wrong team); comparing *absolute* values fixes
  it, and reproduces `game_features.parquet`'s `spread_line` for that game
  (Pittsburgh favored by 9, not Detroit) exactly.
- **`spread_line` sign convention** (**measured**, not assumed): in
  `game_features.parquet`, a positive `spread_line` means the HOME team is
  favored by that many points -- confirmed by reproducing the stored
  `ats_margin` column exactly (`ats_margin = (home_score - away_score) -
  spread_line`) on inspected rows, not by recalling a convention. This
  script's `open_home_spread`/`close_home_spread` columns use the same sign.
- **Team codes**: `game_features.parquet` uses current franchise codes
  retroactively for its entire history (**measured**: `LA`/`LAC`/`LV`
  appear in every season 2009-2026; `STL`/`SD`/`OAK` never appear at all),
  so SBR's `St.Louis`/`SanDiego`/`Oakland` tokens map to `LA`/`LAC`/`LV`.
  All 44 unique raw team tokens across all 15 seasons were enumerated
  (**measured**) and mapped exactly (`TEAM_MAP` in the script) -- including
  typos (`"Washingtom"`, one occurrence, 2020-21) and truncations
  (`"Kansas"`, `"Tampa"`, both 2020-21 only).
- **Bug found and fixed this session**: one token, bare `"NewYork"`
  (2013-14, single occurrence, Nov 10 2013 vs Oakland), was first resolved
  from an *unverified memory* of a Jets-Raiders score that week -- and that
  memory was wrong. Checking `game_features.parquet` directly showed the
  real game is `2013_10_OAK_NYG` (Giants, not Jets; the Jets had a bye that
  week). This is exactly the failure mode `AGENTS.md`'s "label how you know
  it" rule exists to catch, and it was caught by doing what the rule
  requires -- opening the source before trusting a recalled fact.
- **Missing opens**: the task brief flagged "occasional missing opens" as a
  known SBR quirk. **Measured**: this did not reproduce -- all 8,050
  Open/Close/ML cells across all 15 seasons are either numeric or `"pk"`,
  zero blanks, zero `"NL"` markers. Reported as a measured zero, not assumed
  absent.
- **Date parsing**: SBR's `Date` column is month+day with no year and no
  separator (`"906"` = Sep 6, `"1230"` = Dec 30, `"203"` = Feb 3); the
  season's start year is used for Aug-Dec dates, start year + 1 for Jan-Mar.
- **Neutral-site games** (`VH="N"`, Super Bowl only): which row is "home"
  is a positional assumption (first row = away slot, second = home slot),
  not a verified designation -- flagged via a `neutral_site` column. This
  affects at most 15 games total (1/season) and was not separately audited
  against `game_features.parquet`'s own neutral-game home designation.
- **`week`**: not present in the SBR source at all. Populated only by
  joining to `game_features.parquet` on `(season, home_team, away_team,
  game_date)`; null for every unmatched game (all of 2007-2008, see
  below) and for a small number of within-season date mismatches described
  in the coverage check.

## 3. Validation

### 3a. CLOSE check (SBR's Close vs `game_features.parquet`'s `spread_line`)

Joined on `(season, home_team, away_team, game_date)`, with a 1-day
tolerance fallback where the exact date didn't match (see 3c). **Measured**,
`scripts/ingest_sbr_odds.py --validate`:

| Season | SBR games | Matched | Match rate | Mean \|diff\| | Median \|diff\| | Share ≤0.5pt | Share exact |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 2007 | 267 | 0 | 0% | -- | -- | -- | -- |
| 2008 | 267 | 0 | 0% | -- | -- | -- | -- |
| 2009 | 267 | 267 | 100% | 0.301 | 0.0 | 88.4% | 58.1% |
| 2010 | 267 | 267 | 100% | 0.285 | 0.0 | 89.5% | 57.3% |
| 2011 | 267 | 267 | 100% | 0.260 | 0.0 | 90.3% | 61.0% |
| 2012 | 267 | 267 | 100% | 0.305 | 0.0 | 88.4% | 60.3% |
| 2013 | 267 | 267 | 100% | 0.380 | 0.5 | 83.1% | 47.2% |
| 2014 | 267 | 267 | 100% | 0.219 | 0.0 | 95.5% | 61.4% |
| 2015 | 267 | 267 | 100% | 0.193 | 0.0 | 94.4% | 67.8% |
| 2016 | 267 | 267 | 100% | 0.191 | 0.0 | 92.9% | 70.0% |
| 2017 | 267 | 267 | 100% | 0.191 | 0.0 | 94.0% | 70.0% |
| 2018 | 267 | 267 | 100% | 0.232 | 0.0 | 93.3% | 61.8% |
| 2019 | 267 | 267 | 100% | 0.208 | 0.0 | 96.3% | 62.5% |
| 2020 | 269 | 269 | 100% | 0.312 | 0.0 | 89.9% | 55.8% |
| 2021 | 285 | 285 | 100% | 0.595 | 0.5 | 67.0% | 36.1% |

2007-2008 show 0 matches because `game_features.parquet` itself has no rows
before season 2009 (**measured**: `first_season: 2009` in
`data/processed/game_features.manifest.json`) -- this is the archive's
actual net-new territory, not a join bug.

2009-2020 agree tightly with the repo's own closing line (mean |diff|
0.19-0.38 pts, 83-96% within half a point) -- consistent with both sources
describing the same real closing market, captured by different
aggregations/books. **2021 is a measured outlier**: mean |diff| jumps to
0.595 and share-within-0.5pt drops to 67%. Spot-checking the largest 2021
diffs found isolated cases where SBR's raw Close cell disagrees sharply with
`spread_line` even after the sign-convention bug fix above -- e.g. a
Baltimore @ Chicago game (2021-11-21) where SBR's raw Close for Baltimore is
literally `"pk"` (pick'em) but `game_features.parquet` records Chicago as a
5-point underdog. This reads as one or a few genuine source-vs-source
disagreements for that season, not a parsing defect (the disambiguation
heuristic's `ambiguous` flags, which catch implausible spread/total splits,
are **not** elevated for 2021 -- 2 ambiguous Open rows out of 285 games,
0 ambiguous Close rows) -- reported as measured, not explained away.

### 3b. OPENER check (SBR's Open vs the repo's Tuesday-opener archive)

This is the question that decides what the archive is worth for
opener-graded evaluation: **is SBR's "Open" comparable to the Tuesday
opener the pool's paired archive (`nfl_ats.clv.build_pairing_table` /
`nfl_ats.market_data.tuesday_opener_quotes`) actually grades against?**

Overlap is limited to the two seasons where both archives have coverage:
the purchased point-in-time market store (`data/market/raw`,
`capture_kind="historical_backfill"`) has `tue_open` decision-label
snapshots for seasons 2020-2025 (**measured**), and SBR's populated range
tops out at 2021-22 -> overlap seasons are **2020 and 2021 only**.

**Measured**, joined on `game_id` (via the season/home/away/date join to
`game_features.parquet`, 1-day tolerance included):

| Season | SBR games | tue_open available | Matched | Mean \|diff\| | Median \|diff\| | Share ≤0.5pt | Share exact |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 2020 | 269 | 239 | 239 | 1.403 | 1.0 | 43.5% | 17.6% |
| 2021 | 285 | 252 | 252 | 1.313 | 1.0 | 46.0% | 17.9% |
| **Overall** | | | **491** | **1.357** | **1.0** | **44.8%** | **18.0%** |

Signed-diff supplement (SBR open minus repo `tue_open`, **measured**, not
in the script's printed output but computed the same way this session):
mean **+0.28**, std **2.02**, min -14.5, max +8.5, Pearson correlation
**0.949**. As a robustness check, the same comparison was also run
(**measured**, ad hoc this session, not wired into the script) against the
store's `true_open` decision label -- an even earlier Monday snapshot that
exists for the same 2020-2021 seasons but is outside
`odds_backfill.DECISION_LABELS`, so it required calling
`decision_market_consensus` directly rather than
`build_pairing_table`: 471 matched games, mean |diff| **1.29**, median
**0.75**, share ≤0.5pt **49.5%**, correlation **0.944** -- marginally
closer than `tue_open` on point-error, essentially the same on
correlation.

**What this does and does not prove.** SBR's Open is strongly *correlated*
with both the repo's Tuesday-opener and its even-earlier Monday `true_open`
consensus (r≈0.94-0.95) -- it is clearly tracking the same real market, not
noise. But it is **not point-identical to either**: only ~18% of games
match exactly, under half fall within half a point, and the median absolute
gap is a full point. **This means SBR's "Open" cannot be substituted for
the repo's Tuesday-opener line and treated as equivalent without a stated
error band.** The provenance of SBR's own Open timestamp is not established
here (SBR aggregates from multiple books with an unpublished capture
methodology on the current site) -- this check bounds the disagreement
empirically without resolving what specific moment SBR's Open represents.

### 3c. COVERAGE check (SBR games matched to `game_features.parquet`)

**Measured**:

| Season | SBR games | Matched | Exact date | 1-day tolerance | `game_features` games | Match rate |
|---:|---:|---:|---:|---:|---:|---:|
| 2007 | 267 | 0 | 0 | 0 | 0 | 0% |
| 2008 | 267 | 0 | 0 | 0 | 0 | 0% |
| 2009 | 267 | 267 | 267 | 0 | 267 | 100% |
| 2010 | 267 | 267 | 266 | 1 | 267 | 100% |
| 2011 | 267 | 267 | 267 | 0 | 267 | 100% |
| 2012 | 267 | 267 | 267 | 0 | 267 | 100% |
| 2013 | 267 | 267 | 267 | 0 | 267 | 100% |
| 2014 | 267 | 267 | 266 | 1 | 267 | 100% |
| 2015 | 267 | 267 | 253 | 14 | 267 | 100% |
| 2016 | 267 | 267 | 267 | 0 | 267 | 100% |
| 2017 | 267 | 267 | 267 | 0 | 267 | 100% |
| 2018 | 267 | 267 | 267 | 0 | 267 | 100% |
| 2019 | 267 | 267 | 267 | 0 | 267 | 100% |
| 2020 | 269 | 269 | 269 | 0 | 269 | 100% |
| 2021 | 285 | 285 | 284 | 1 | 285 | 100% |

Every 2009-2021 game matches once team codes are correct (post-bugfix).
**17 of 3,491 matched games (0.5%) needed the 1-day date tolerance**,
concentrated in two patterns, both measured directly from the raw data, not
inferred:

- **2010 (1 game) and 2014 (1 game)**: SBR's raw `Date` is one day *after*
  `game_features.parquet`'s `gameday` -- the 2014 case is a Monday Night
  Football game (Patriots @ Chiefs, real kickoff Monday Sep 29 2014; SBR
  logs it as `"930"`, Sep 30).
- **2015 (14 games, in two clusters)**: SBR's raw `Date` is one day
  *before* `gameday` for an entire Sunday slate -- all 12 games of Week 6
  2015 (real date Sunday Oct 18 2015; SBR logs `"1017"`, Saturday Oct 17)
  plus both 2016 conference championship games (real date Sunday Jan 24
  2016; SBR logs `"123"`, Saturday Jan 23).

Both directions occur, so this is not a single correctable global offset --
it looks like a genuine SBR-side date-entry quirk for specific weeks
(**inferred**; no SBR changelog or errata was found to confirm a cause),
not a bug in this parser (regular-season dates surrounding these clusters
match exactly). The processed parquet keeps the original SBR date
(`game_date`) and records the size of any tolerance used
(`week_match_date_diff_days`) rather than silently overwriting it.

Unmatched-game examples (all from 2007-2008, i.e. real non-overlap with
`game_features.parquet`'s season-2009 floor, not a mapping failure):

```
2007-09-06  NO @ IND
2007-09-09  KC @ HOU
2007-09-09  DEN @ BUF
2008-09-04  WAS @ NYG
2008-09-07  CIN @ BAL
2008-09-07  NYJ @ MIA
```

## 4. What this unlocks -- and what it doesn't (no experiment run)

- **Net-new opener/close coverage**: 2007-2008 (534 games), predating
  `game_features.parquet`'s season-2009 floor entirely. For seasons
  2009-2021, this archive adds a second, independently-sourced *close* that
  agrees tightly with the repo's own (3a), which is useful as a
  cross-validation signal for the existing close, not as new opener
  ground-truth for those years -- the repo's paired-archive opener
  coverage (2020-2025, per `docs/pool_edge_plan.md`) already exceeds what
  SBR's Open can substitute for (3b).
- **What section 3b rules in**: SBR's Open is usable as an *approximate,
  correlated* opener proxy (r≈0.94-0.95 vs both the repo's `tue_open` and
  `true_open`) for 2007-2019, where no purchased point-in-time archive
  exists at all. That is a real, previously-unavailable signal for
  era-stratified opener-movement research on those 13 seasons.
- **What section 3b rules out**: treating SBR's Open as interchangeable
  with the pool's actual Tuesday-lock line. Median disagreement is a full
  point and only ~18% match exactly -- any opener-graded backtest that
  uses 2007-2019 SBR-Open data alongside 2020+ purchased-archive Tuesday
  opens must either (a) treat the two eras as a different, coarser-grained
  measurement with a stated error band, or (b) restrict claims to the
  correlational/directional level, not point-accuracy claims. Which of
  those a future evaluation should do, and any resulting era-stratified
  performance numbers, is out of scope here -- this document validates the
  pipeline and characterizes the gap; it does not run that evaluation.
- **Timestamp semantics remain unproven, not just uncertain-but-close**:
  nothing in this session establishes what real-world moment SBR's Open
  column represents (a specific book's opener, an aggregated first
  print, something else) -- only that it correlates with, and measurably
  differs from, two independently-timestamped snapshots this project
  already trusts.

## 5. Files

- `scripts/ingest_sbr_odds.py` -- fetch, parse, join, and validate (all four
  steps in one script; `--skip-fetch --snapshot <dir>` reuses an existing
  raw snapshot instead of re-fetching).
- `data/raw/sbr_odds/20260819T192226Z/` -- raw HTML snapshot (15 season
  pages + index + robots.txt) and `manifest.json` (per-file URL, HTTP
  status, byte count, SHA-256, fetch timestamp). Gitignored
  (`data/raw/**`), confirmed via `git status` showing nothing tracked under
  `data/` after this session.
- `data/processed/sbr_odds.parquet` (4,025 rows) and
  `data/processed/sbr_odds.manifest.json`. Gitignored (`data/processed/**`).
