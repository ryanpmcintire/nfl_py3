# Data sources and governance

The source catalog below describes storage and time semantics. The separate
[historical feasibility audit](data_feasibility.md) records verified season and
row coverage and determines whether each proposed research lead has enough
independent history for its intended model complexity.

## Primary source

The maintained source is [nflverse](https://nflverse.nflverse.com/), loaded
through [nflreadpy](https://nflreadpy.nflverse.com/). The pipeline downloads:

- schedules, results, closing spreads, prices, totals, rest, and game context;
- weekly team passing and rushing aggregates.

Optional maintained nflverse layers download season-partitioned play-by-play
and timestamped depth-chart observations. The canonical PBP v1 filter stores
only fields needed to reproduce filters, drives, team efficiencies, and QB
states rather than all 372 upstream columns.

The source adapter validates the fields the feature builder depends on. An
upstream breaking change should stop ingestion with a readable error instead
of producing a subtly different model table.

## Storage

Each refresh creates this ignored directory:

```text
data/raw/20260812T140000Z/
  schedules.parquet
  team_stats.parquet
  manifest.json
```

The manifest records the fetch time, season range, source loaders, row counts,
and SHA-256 digest of each file. A directory without a manifest is incomplete
and will not be selected as the latest snapshot.

Schedule and team-stat season ranges are recorded separately. This lets a
preseason snapshot contain the upcoming schedule while team statistics stop at
the latest completed season. It does not fabricate empty current-season stats.

`data/processed/game_features.parquet` is replaceable derived data. Its sibling
manifest records the raw snapshot and feature parameters used to build it.
The optional `game_features_pbp.parquet` manifest additionally records its PBP
feature version, rolling-state parameters, and opponent-adjustment parameters.

```text
data/pbp/raw/<UTC snapshot>/season=2025/plays.parquet
data/quarterbacks/depth/raw/<UTC snapshot>/quarterbacks.parquet
data/market/raw/<UTC snapshot>/quotes.parquet
data/market/historical/raw/<UTC snapshot>/market.parquet
```

Each directory has a manifest with the source, contract/filter version, row
counts, observation range, and SHA-256 digests. PBP is written one season at a
time and the manifest is published last.

## Point-in-time rules

- A completed game's team statistics may affect only later game dates.
- A game's result, score, ATS margin, and cover label are outcomes, never model
  inputs.
- A historical closing line is labelled as such. Live-odds observations include
  book, market, line, price, and the time observed.
- Raw snapshots are immutable. Refreshing means writing a new snapshot.
- The feature boundary canonicalizes historical `OAK`, `SD`, and `STL` schedule
  abbreviations to the `LV`, `LAC`, and `LA` franchise IDs used by nflverse
  team statistics; raw game IDs and snapshots remain unchanged.
- PBP features use a strict earlier-game join; a game's own plays cannot affect
  that game's pregame state.
- Opponent-adjusted PBP features are fit once per NFL week from earlier weeks
  and earlier game dates only. Their manifest records the time-decay half-life,
  ridge penalty, and minimum team-game warm-up.
- Graph and schedule-strength features are frozen for the entire NFL week
  before any result from that week updates PageRank, HITS, or ridge ratings.
- A quarterback depth row must have a real observation timestamp before the
  decision cutoff. Untimestamped historical weekly depth rows are not silently
  treated as pregame observations.
- The [nflverse schedule dictionary](https://nflreadr.nflverse.com/articles/dictionary_schedules.html)
  defines `gametime` in Eastern time. The feature builder combines it with
  `gameday`, converts the kickoff to UTC, and retains it for prospective freeze
  checks; kickoff is never a model input.

The `smoke-source` command validates the current schedule and latest completed
team-stat schemas without writing a snapshot. A scheduled GitHub Actions job
runs it weekly so missing fields or empty feeds fail visibly.

## Prospective records

`predict --freeze` is deliberately stricter than ordinary historical scoring.
Every game must have a UTC kickoff later than the freeze timestamp, no outcome
may be present, and game IDs must be unique. Each accepted run gets a new
directory under `artifacts/prospective/` containing prediction rows, model and
source provenance, and a SHA-256 digest. Existing records are never replaced.

## Licensing

nflverse documents its data attribution and licensing in the
[nflverse data repository](https://github.com/nflverse/nflverse-data). Most
datasets are CC-BY 4.0; individual upstream datasets can have different terms.
Review the source-specific license before redistributing a snapshot or using it
commercially. Generated source data is intentionally not committed here.

## Optional live odds

The implemented provider adapter uses [The Odds API](https://the-odds-api.com/)
for current NFL spread and moneyline observations. It reads
`THE_ODDS_API_KEY` from the environment, never writes the key, retains the raw
response hash plus normalized book/market/line/price timestamps, and records
quota response headers. Repeated calls append immutable snapshots. Local tools
derive a latest same-book pre-kickoff quote, cross-book median/dispersion, and
same-book closing-line value.

The provider's historical snapshot endpoint is not required by the project.
The free/current API cannot retroactively create a valid decision-time history,
so local observations are archived without making an earlier CLV claim. Book
coverage, limits, redistribution rights, retention terms, quotas, and API
availability remain provider dependencies. The nflverse-only workflow remains
reproducible without an odds key.

## Free historical market cross-check

`market-backfill` downloads the public `tobycrabtree/nfl-scores-and-betting-data`
Kaggle archive without requiring an API key. The raw ZIP, its Kaggle version and
CC BY-NC-SA 4.0 license, normalized home-oriented lines, SHA-256 hashes, and an
nflverse comparison are stored together in an immutable snapshot.

The source contains one reported closing spread per game. It has no sportsbook
identifier or observation timestamp, so `is_timestamped_quote` is always false
and `observed_at_utc` is null. It can validate sign conventions and quantify
source disagreement; it cannot answer what line was available at an arbitrary
earlier decision time.

## Free 2025 opener and multi-book close sample

`market-open-close-backfill` preserves the CC BY-NC 4.0 Kaggle sample at:

```text
data/market/historical/open_close/raw/<UTC snapshot>/source.zip
data/market/historical/open_close/raw/<UTC snapshot>/quotes.parquet
data/market/historical/open_close/raw/<UTC snapshot>/games.parquet
data/market/historical/open_close/raw/<UTC snapshot>/manifest.json
```

The normalized long table contains opener and nine named-book closing outcomes
for spreads, moneylines, and totals. In version 1 this is 17,100 rows covering
all 285 games in the 2025 season/postseason; 272 regular-season games match the
canonical nflverse feature table. `quote_stage` distinguishes `opening` from
`reported_closing`, while `is_timestamped_quote` remains false and
`observed_at_utc` remains null because the source does not publish quote times.
The one-game table records opener, median book close, book dispersion, movement,
and nflverse close comparison. The importer also checks whether each spread and
same-source moneyline identify the same favorite. Version 1 flags 12 opener
games and 29 closing book/game rows with opposite directions; these remain in
the raw normalized quote table but are excluded from derived movement and
consensus summaries. Pick'em spreads are not treated as contradictions.
