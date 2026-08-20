# ENV-01 sourcing verdict: is a free point-in-time weather-FORECAST archive accessible?

**Verdict: yes, a free source exists and is instantly self-serve for 2020-present,
and free (but request-gated, unverified turnaround) for 2004-present, which
covers this project's full 2009-2025 window.** Actually verified this session
by downloading and inspecting one real archived forecast bulletin (evidence
below) -- not just documentation claims.

This resolves the open half of ROADMAP ENV-01 ("the sourcing question this
row now turns on," recorded 2026-08-19): NOAA's own NDFD archive, not
Open-Meteo, is the free path.

## Bottom line by source

| source | point-in-time? | coverage | cost | access mechanics | verified this session |
|---|---|---|---|---|---|
| **NOAA NDFD via AWS Open Data (NODD)** | yes | **measured**: 2020-2026 only (year prefixes `wmo/maxt/2020/` .. `2026/`; no 2019 or earlier) | free, no signup, no API key | anonymous HTTPS/S3 `GET`, GRIB2 files, no rate limit encountered | **yes** -- downloaded and parsed a real file, see below |
| **NOAA NDFD via NCEI HAS/AIRS ("NDFD - By WMO Header")** | yes | **read** (NCEI HAS page): "June 6, 2004 to August 16, 2026" -- covers the full 2009-2025 ATS window | free ("we provide the data and products at no cost to customers," **read** from the HAS page) | request/order system ("Order Data" links through a file-application router), NOT a plain instant bulk download | **not yet** -- the listing and coverage window are read from the HAS page; I did not complete an actual order this session, so turnaround time (instant vs. queued/email) is unverified |
| **Open-Meteo Historical Forecast API** | yes, by design (stitches first hours of successive model runs) | **read** (Open-Meteo docs): GFS from 2021-03-23, ECMWF IFS HRES from 2017-01-01, most other models from Nov 2022 | **read** (Open-Meteo pricing page): free tier explicitly excludes this API (❌ in the plan table); requires the paid Professional plan or higher | REST API, JSON | not fetched (paid, out of scope to pay for a verification pass) |
| **NWS api.weather.gov** | **no** | current + 7-day-forward only | free | REST API, JSON | **yes** -- confirmed the docs state alerts keep a 7-day window and forecasts are "over the next seven days," with no archived-forecast endpoint |
| **Iowa Environmental Mesonet (IEM)** | not established | IEM's archive strengths are METAR/ASOS obs, NEXRAD, and NWS *text* products (AFD, etc.), not a gridded NDFD point-forecast time series product | free | varies | not fetched in depth -- **inferred** from the IEM site's application/archive index that a ready-made NDFD point-forecast puller does not exist there (their archived-data page lists NEXRAD since 2002, ASOS since ~1928, storm reports since 2003, nothing named NDFD); this is my reading of what's listed, not a confirmed absence, and IEM was not pursued further once the AWS/NCEI sources above worked |

## What was actually fetched (evidence, not just documentation)

**Measured this session**, via plain `curl` (no auth, no API key):

1. Listed the AWS NODD bucket's element and year structure:
   ```
   curl -s "https://noaa-ndfd-pds.s3.amazonaws.com/?list-type=2&delimiter=/&prefix=wmo/maxt/&max-keys=100"
   ```
   returned year prefixes `2020/ 2021/ 2022/ 2023/ 2024/ 2025/ 2026/` only -- the
   free instant-download tier does **not** reach 2009-2019.

2. Confirmed the element set relevant to this project's weather battery is
   present as sibling folders under `wmo/`: `maxt` (daily max temp), `temp`
   (temperature), `wspd` (wind speed), `wgust` (wind gust), `wdir` (wind
   direction) -- i.e. both fields (`temp`, `wind`) the existing
   `scripts/nfl_weather_battery_screen.py` battery uses from game-time
   actuals have a forecast-archive counterpart.

3. Listed one day's files for **Sunday 2022-12-11** (an NFL Sunday) and
   confirmed **multiple issuance times per day**, each timestamped in the
   filename, e.g. `YGRZ98_KWBN_202212112153` = issued 2022-12-11 21:53 UTC.
   Issuances that day: 00:53, 01:59, 02:58, 03:57, 04:56, 05:54/38,
   06:58, 07:57/00, 08:56/57, 09:55/59, 10:59/56, 11:52/36/58,
   12:56/50/55, 13:55/57, 14:54/59, 15:53/56, 16:57/58, 17:34/56/55,
   18:55/57, 19:54/59/33, 20:58/56, 21:57/58/40/41, 22:55/00,
   23:38/54/57 UTC -- roughly hourly. A file issued **17:34 UTC** would be a
   genuine pregame forecast for that day's 1pm ET (18:00 UTC) kickoffs,
   issued ~26 minutes before kickoff; earlier files (e.g. 05:33 or 11:33 UTC)
   give a forecast issued many hours out.

4. Downloaded that specific file
   (`wmo/maxt/2022/12/11/YGRZ98_KWBN_202212112153`, 419,007 bytes) to
   `<scratchpad>/ndfd_sample/maxt_20221211_2153.grib` and inspected it byte
   for byte:
   - First bytes are a WMO abbreviated heading, not binary:
     `****0000418988**\nYGRZ98 KWBN 112153\r\r\n****0000102197****\nYGRB00 KWBN 112153\r\r\n`
     -- this **is** the point-in-time stamp: WMO header confirms issuance
     day/time `112153` (11th, 21:53 UTC), matching the S3 key's own
     timestamp independently.
   - The binary payload starts with `GRIB` immediately after the WMO
     header, followed by `\x00\x00\x00\x02` (GRIB edition 2).
   - The magic string `GRIB` occurs **4 times** in the file -- it is a
     WMO-bulletin wrapper around **4 separate GRIB2 messages** (this
     element's file bundles multiple forecast valid-times/grids per
     issuance).
   - This is real, valid-looking archived NWS forecast product data, not an
     error page, a redirect stub, or an empty placeholder.

## The real cost: decoding, not access

Access is free and (for 2020+) instant. **Using** it is a separate, nontrivial
cost that this session did not pay:

- **Measured** in this repo's own uv environment: `pygrib`, `cfgrib`,
  `eccodes`, and `xarray` are all `ImportError` -- no GRIB2 decoding library
  is currently installed (`.tools/uv.exe run --no-sync python -c "import
  pygrib"` etc., all failed). Reading these files requires adding one of
  those dependencies.
- Each file is a **CONUS-wide multi-message grid** (hundreds of KB per
  issuance), not a single number -- turning it into "what was forecast for
  stadium X at issuance time Y" requires (a) selecting the right message for
  the target valid date/hour, (b) full GRIB2 decode, (c) nearest-grid-point
  or bilinear lookup against a stadium lat/lon table. **No stadium
  lat/lon table exists in this codebase yet** (**measured**: `grep -rl
  "latitude" src/nfl_ats` returned nothing) -- that is a second prerequisite
  before any point value can be extracted, for either NDFD source.
- The 2004-2019 depth needed to cover this project's full 2009-2025 window
  sits behind the HAS/AIRS request system, whose turnaround (instant vs.
  queued) is **unverified** -- the natural next step, not completed here.

## What this changes in ROADMAP ENV-01

The prior entry called the NOAA NDFD backstop "scouted 2026-08-19,
unverified" and flagged Open-Meteo's paid/thin-history status as the open
question. Both halves are now settled by direct measurement, not
documentation-reading alone:

- Open-Meteo's Historical Forecast API is confirmed paid-only (free tier
  explicitly excludes it) -- not a free path, full stop.
- NOAA NDFD **is** a free, real, point-in-time archive. The instantly
  self-serve slice (AWS NODD) only covers 2020-present -- roughly a third of
  the project's window (2020-2025 REG of 2009-2025) -- but a request-based
  free path (NCEI HAS/AIRS) claims full 2004-present coverage, unverified
  end-to-end this session. Building an actual ENV-01 feature would mean: (1)
  add a GRIB2 decoder dependency, (2) build a stadium lat/lon table, (3) pull
  2020-2025 instantly from AWS NODD and separately test one HAS/AIRS order
  to learn its real turnaround for 2009-2019, (4) select, per historical
  game, the issuance file timestamped closest-before that game's actual
  kickoff. None of that pipeline work was built this session -- this is a
  sourcing verdict, not an ENV-01 implementation.

## Provenance

- **measured**: bucket listings and the downloaded sample, this session, via
  `curl` (commands and byte inspection reproduced above); Python import
  probe for GRIB tooling; `grep` for a stadium coordinate table.
- **read**: Open-Meteo docs/pricing pages, NCEI NDFD product page, NCEI HAS
  dataset-selector page, weather.gov API services documentation (all fetched
  this session via WebFetch, quotes quoted above).
- **inferred**: IEM's suitability judgment (its archive index does not list
  an NDFD point-forecast product among its named datasets); not independently
  disproven, just not pursued once AWS/NCEI worked.

**2026-08-19, follow-up session:** IEM's suitability judgment above was wrong
in one specific, load-bearing way -- IEM does NOT have a gridded NDFD
point-forecast product, but it DOES separately archive plain-text **GFS MOS**
station bulletins (a different NWS product) back to 2020-07-12, which turned
out to be the actual build route. See `docs/forecast_archive_build.md` for
the built pipeline, station mapping, and validation.
