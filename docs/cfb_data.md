# College football data sources and governance

XLG-02 ingests the college-football sources selected by the XLG-01
feasibility audit: six no-key bulk sources (phase 1) plus six CFBD API
gap-fillers (phase 2, documented at the end of this file). CFB rows are never
appended to NFL rows; the data exists to estimate shared football mechanisms,
replicate hypotheses, and build priors that are judged on NFL-only outer
weeks (see `ROADMAP.md`, cross-league section).

## Sources and provenance

| Source key | Upstream | Seasons | Pinned by |
|---|---|---|---|
| `schedules` | cfbfastR-data `schedules/parquet/cfb_schedules_{year}.parquet` (CFBD `/games` flavor, all divisions) | 2001-2025 | git commit SHA recorded in the manifest |
| `lines` | cfbfastR-data `betting/parquet/cfb_line_odds.parquet` (single multi-book archive, ~1.18M rows) | 2006-2025 | git commit SHA recorded in the manifest |
| `pbp` | sportsdataverse-data release `espn_cfb_pbp`, asset `play_by_play_{year}.parquet` (477 columns incl. EPA/WPA) | 2004-2025 | asset `updated_at` + SHA-256 of downloaded bytes |
| `rosters` | sportsdataverse-data release `espn_cfb_game_rosters`, asset `game_rosters_{year}.parquet` | 2004-2025 | asset `updated_at` + SHA-256 of downloaded bytes |
| `participants` | sportsdataverse-data release `espn_cfb_play_participants`, asset `play_participants_{year}.parquet` | 2014-2025 | asset `updated_at` + SHA-256 of downloaded bytes |
| `espn_betting` | sportsdataverse-data release `espn_cfb_betting`, asset `betting_{year}.parquet` (cross-check only) | 2012-2025 | asset `updated_at` + SHA-256 of downloaded bytes |

Upstream release assets are rebuilt in place, so a URL alone never identifies
data. Every snapshot stores the verbatim downloaded bytes under `source/`,
the canonical contract table per season partition, and SHA-256 digests of
both, with the manifest written last as the commit marker:

```text
data/cfb/<source>/raw/<UTC snapshot>/source/<original upstream file>
data/cfb/<source>/raw/<UTC snapshot>/season=2024/<table>.parquet
data/cfb/<source>/raw/<UTC snapshot>/manifest.json
```

`nfl-ats cfb-ingest --source <key> --start-season Y --end-season Z` downloads
a snapshot; `--dry-run` resolves the pinned commit or release, file names,
and byte sizes without downloading anything (use it before a large play-by-play
backfill; PBP assets are 25-59MB per season). `nfl-ats cfb-summary` reports
the latest local snapshot per source.

## Betting-line source regimes

The line archive is opener plus an unidentified-time resolved quote (a close
proxy). `date_time` is scheduled kickoff; **no source records quote
observation times**, so CFB supports closing/near-closing benchmarks and
opener-vs-close movement, never intraweek timing. Each canonical row carries a
queryable `source_regime` column:

| Seasons | `source_regime` | Books | Openers |
|---|---|---|---|
| 2006-2011 | `sbr_multibook` | ~15-20 offshore books (Pinnacle, 5Dimes, BetCRIS, bet365, ...) | essentially absent (<2% of rows) |
| 2012-2019 | `sbr_multibook` | same offshore set | ~96% of FBS games, from the single "5Dimes & sportbet" SBR opener column |
| 2020-2022 | `cfbd_provider_sparse` | consensus, teamrankings, William Hill NJ, Caesars, Bovada | **2020 has zero openers and zero moneylines**; 2021-2022 openers return |
| 2023+ | `espn_book_era` | ESPN Bet, DraftKings, Bovada | ~99-100% of games |

The schema contract enforces the landmines: a requested season with no rows
fails; a season from 2012 onward (except 2020) with zero openers fails as an
upstream regression; unknown market types fail. Exact duplicate rows and rows
without a game id (unjoinable SBR remnants) are dropped and counted in the
manifest audit; the verbatim archive remains under `source/`.

The `espn_betting` release is one resolved line per game and is retained only
as a cross-check of the primary archive. Its 2004-2011 assets are placeholder
junk (constant spread 2.5/total 55.5, `odds_source='default'`,
`game_spread_available=False` on every row): those seasons are refused before
download, placeholder rows in later seasons are dropped, and a season that is
placeholder-only after filtering fails loudly.

## Availability semantics (fail closed)

No historical CFB injury or availability source exists. The
`espn_cfb_injuries` release has zero assets, and CFBD v5 has no injuries
endpoint. Any CFB availability signal must come from realized, postgame
participation; XLG-07's historical branch fails closed.

- **Game rosters are scrape-time listings, not game-day availability.**
  The audit showed zero of 27,471 players changed their Active/Inactive flag
  across all 2024 games, and `did_not_play`/`starter`/`valid` are False on
  every row. These columns are **quarantined**: the schema contract excludes
  them from the canonical table, `load_cfb_snapshot` refuses a roster table
  that contains them, and the manifest records each excluded column with its
  reason. 2004-2013 roster files list only stat-credited players (~32 per
  team-game); 2014+ files list full ~120-player rosters.
- **Play participants are credited actors from play text only**
  (passer/rusher/tackler/...). They are positive evidence of participation.
  Absence of credit is never evidence of absence; there are no lineups or
  snap counts for CFB.

## Play-by-play

Canonical PBP keeps a declared ~55-column subset (identity, teams, situation,
play flags, EPA/WPA family, credited passer/rusher/receiver ids) keyed on
`game_id` + `game_play_number`; the full 477-column upstream file is preserved
verbatim under `source/`. `seasonType` code 2 is regular season and 3 is
postseason; both are stored so downstream work chooses its own window.
Coverage regimes recorded in the manifest: 2004-2013 is the thinner early
ESPN era, 2014+ covers ~98% of completed FBS games (verified on 2024).

Two known upstream defects in the 2026-08-03 asset rebuild are dropped and
counted in the manifest audit rather than failing the season: rows with a null
`game_id`/`game_play_number`/`season`/`week`/`seasonType` key (19-2,128 junk
rows in nine seasons between 2006 and 2016), and ESPN off-season/all-star
exhibitions labeled `seasonType` 4 (every season 2008-2022) or 5 (2020 only),
typically 2-4 games per season outside the competitive calendar. Any other
unknown `seasonType` code still fails ingestion closed.

## Derived benchmark table (XLG-03)

`nfl-ats cfb-build-features` derives the canonical benchmark table
(`data/processed/cfb_game_features.parquet`) from the snapshots above:
completed regular-season FBS-vs-FBS games 2006-2025 carrying both an
orientable spread and play-by-play. Aggregation semantics, declared once:

- **Team-side identification without name joins.** Each season's line
  abbreviation is resolved by intersecting the ESPN home/away team-id pairs
  of every game it quotes; per-game partner repair covers abbreviations that
  stay ambiguous (two teams sharing a label, or a single quoted game).
  Games whose sides cannot be identified are excluded and counted in the
  build audit, never guessed.
- **`spread_line`** is the median across books of each book's home-oriented
  close-proxy spread, using the NFL sign convention (positive = home
  favored, `ats_margin = result - spread_line`). Book count, population-std
  dispersion, the median opener where present, median side prices, the
  median total, and `source_regime` are recorded per game. The opener and
  prices never feed the residual model; prices back only the market
  baseline's no-vig probability (2006-2019 carry prices; later regimes fall
  back to a 0.5 cover prior).
- **Pregame team state** mirrors the NFL base-state design verbatim
  (span-8 EWM, three-game maturity, offseason retention 0.67 toward the
  prior season's league mean): EPA/play offense and defense, success rate,
  explosive rate, and pace from competitive scrimmage plays (rush/pass,
  kneels removed, possession win probability 5-95%) of regular-season
  FBS-vs-FBS games, strictly earlier than the game being scored.
- Postseason rows exist in the schedule source only from 2024, so the
  benchmark is regular-season only; every exclusion class is counted in the
  manifest audit.

`nfl-ats cfb-benchmark` runs the frozen CFB-only walk-forward market-residual
evaluator (Ridge alpha 10, no calibration, minimum 500 training games, weekly
refits, clean-core headline window 2012-2019 plus 2021-2025), and
`nfl-ats cfb-sensitivity-audit` ports the NFL positive-control protocol
(permuted nulls plus known 0.5/1/2-point-per-SD synthetic effects across
repeated draws) to measure the benchmark's detection power.

## Licensing

- CollegeFootballData terms permit private caching, normalization, and
  retention (also after access ends) and commercial use; they prohibit
  republishing raw data as a bulk dataset, mirror, or proxy.
- cfbfastR-data declares CC BY 4.0; sportsdataverse-data is MIT. Both reshape
  ESPN/CFBD data whose upstream rights the maintainers cannot enlarge.
- Repository rule, recorded in every CFB manifest: **raw CFB source tables
  are never republished from this repository**. Publish derived aggregates
  only, with "Data provided by CollegeFootballData.com" attribution
  recommended. `data/cfb/**` is gitignored like all generated data.

## Identity linkage

CFBD athlete ids are ESPN athlete ids, ESPN ids persist from college into the
NFL, and nflverse `players.parquet` maps `espn_id -> gsis_id`. Player linkage
across leagues therefore requires no name joins. The sportsdataverse
`cfb_crosswalk` release is itself name-matched and current-season only; do
not rely on it.

## CFBD API gap-fillers

Six additional sources ingest directly from the CollegeFootballData API
(Bearer key in the `CFBD_API_KEY` environment variable; free tier 1,000
calls/month; snapshots keep the raw JSON, endpoint, params, and API version):
`draft_picks` (1967-2026; 1988 and 1996 absent upstream), `returning_production`
(2014+), `recruiting_teams` (2002+), `recruiting_players` (recent classes;
older classes await a future backfill), `usage` (2023+ ingested; 2013-2022
awaits a future backfill), and `portal` (2021+). The draft-pick contract
requires the `collegeAthleteId`/`nflAthleteId` columns and audits their
non-null rate (99.45% for 2015+ picks).

The identity chain the draft snapshot verified: `collegeAthleteId` is the ESPN
id and joins nflverse `espn_id -> gsis_id` at 99-100% for the 2019-2026
classes, ~69% for 2018, ~11% for 2015. `nflAthleteId` is *not* an ESPN id and
must not be joined against nflverse espn_id. CFBD rejects request bursts with
non-quota-consuming 429s; the client throttles and retries.
