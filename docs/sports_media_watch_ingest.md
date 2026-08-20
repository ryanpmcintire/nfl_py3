# Sports Media Watch NFL TV-ratings ingestion

## Decision and source status

- **Measured (2026-08-20, direct request):** the primary 2022 seasonal page
  `https://www.sportsmediawatch.com/nfl-tv-ratings-viewership/2022-season/`
  returned HTTP 200 and 456,172 bytes with the repository's descriptive user
  agent.
- **Read (primary seasonal pages, 2026-08-20):** Sports Media Watch links the
  2023 page to season pages for 2014 through 2022. The pages are free and do
  not require authentication.
- **Measured (primary HTML, 2026-08-20):** the 2014 page exposes a structured
  `smwtable5` table whose rows include week, event date, featured game,
  network, household rating, and viewers. The 2022 page instead exposes its
  weekly tables as `wp-content/uploads/...png` images.
- **Measured (the current primary pages):** these are living seasonal pages;
  they do not expose a row-level timestamp identifying when each final or
  revised rating first became public. Therefore the current archive revision
  is reproducibly accessible but is not, by itself, a historically
  point-in-time-safe feature source.

### Completed bounded archive pass

- **Measured (`data/raw/sports_media_watch/20260820T164046Z/manifest.json`):**
  all 10 requested primary season pages for 2014-2023 were fetched and hashed.
- **Measured (same snapshot):** the parser recovered 1,065 structured ratings
  rows across 2014-2021, including 895 rows assigned to regular-season weeks
  and 769 rows with both teams identified. The primary HTML supplies 17
  regular-season weeks for 2014-2020 and 18 for 2021.
- **Measured (same snapshot):** all 43 indexed ratings images were downloaded
  and hashed: one on the 2017 page and 21 each for 2022 and 2023. The 2022 and
  2023 asset indexes each cover 18 regular-season week labels.
- **Measured (same snapshot audit):** all 43 stored asset hashes match their
  index values, no asset is empty, regular-season structured identities have
  zero duplicates, and every structured row and image remains marked
  `point_in_time_usable=false` with `source_published_at=null`.

## Ingestion and safety contract

Run a bounded pass with:

```powershell
.\.tools\uv.exe run --no-sync python scripts\ingest_sports_media_watch.py `
  --output data\raw\sports_media_watch\<snapshot> `
  --seasons 2014-2023 --max-assets 20
```

Resume the same output path without `--max-assets` to fetch remaining indexed
images. Cached pages and assets are hashed and not fetched again.

The snapshot contains:

- `pages/<season>.html`: exact primary archive response;
- `assets/<season>/<filename>`: relevant ratings images fetched so far;
- `ratings_rows.parquet`: structured rows from older HTML-table seasons;
- `source_index.parquet`: indexed newer-season image sources;
- `manifest.json`: coverage, hashes, resume command, and the leakage contract.

`source_published_at` deliberately remains null for rows parsed from a current
seasonal archive revision. `point_in_time_view` fails closed when any candidate
row lacks a verified publication timestamp. When timestamps are available, it
admits only events strictly before the decision date and sources published no
later than the decision timestamp.

The only frozen eventual constructs are **prior-game viewership** and a
**season-to-date viewership trend computed exclusively from prior games**.
Same-game viewership is an outcome and is forbidden as a feature.

## Exact alternative path to feature-ready data

The next ingestion step is to map each weekly row or image to the site's dated
weekly article, verify that the identical figure or image occurs there, and
record that article's primary publication timestamp as `source_published_at`.
**Measured (direct primary fetch, 2026-08-20):** the Week 9 article at
`https://www.sportsmediawatch.com/2023/11/weekly-sports-ratings-nfl-season-high-sec-nascar-finale-world-series/`
contains both the reported `27.14 million` figure and
`article:published_time=2023-11-08T17:28:45+00:00`; it separately reports
`article:modified_time=2023-12-20T04:28:45+00:00`. This confirms that dated
article metadata is a concrete free path, while also showing why publication
and modification timestamps must remain distinct.
Where the primary article cannot establish identity, use an archived capture
whose capture timestamp predates the prediction decision. A year/month encoded
in an image URL is not a publication timestamp and must not be promoted into
the field. Until one of those identities is established, the downloaded
archive is parser/source evidence only and cannot enter an ATS design matrix.
