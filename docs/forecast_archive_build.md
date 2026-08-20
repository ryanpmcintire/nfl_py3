# ENV-01 forecast-archive build: pipeline, decoder decision, and validation

Replaces the placeholder stub. This documents what was actually run, not just
scouted. `docs/weather_forecast_sourcing.md` is the sourcing verdict (why a
free point-in-time forecast archive exists at all); this doc is the build --
the two scripts that existed on disk (`scripts/ingest_forecast_archive.py`,
`scripts/validate_forecast_archive.py`) run to completion, plus a
`kickoff_nearest` mode added this session, plus real coverage/accuracy
numbers.

## Bottom line

- **Decoder decision: no GRIB2 decoder was ever added, because none is
  needed.** The build route pivoted away from NOAA NDFD GRIB2 (the
  sourcing doc's original target) to the Iowa Environmental Mesonet's
  **GFS MOS JSON API**, which serves plain station bulletins with explicit
  `runtime` (issuance) and `ftime` (forecast valid) fields -- point-in-time
  by construction, station-based (no gridded lat/lon lookup), no binary
  decode at all. This was a prior-session decision (recorded in
  `scripts/ingest_forecast_archive.py`'s docstring and the addendum at the
  bottom of `docs/weather_forecast_sourcing.md`); **read** and reconfirmed
  this session, not re-litigated.
- **Two decision-time cutoffs are both built and both validated**:
  `tuesday_noon` (Tuesday 12:00 ET of the game's week -- ~~the pool-relevant
  decision time~~ **owner-corrected 2026-08-20: the pool's LINE locks then,
  not our picks; this cutoff is the grading-line context, not the
  decision-time information set** -- see below) was already run for the
  full 2020-2025 REG population
  before this session (**read** from its manifest, `built_at_utc
  2026-08-19T20:00:38Z`) and is re-validated below. `kickoff_nearest`
  (issuance closest-before kickoff -- **owner-corrected 2026-08-20: this,
  not `tuesday_noon`, is the actual pool-relevant decision-time
  information set, since picks are editable up to each game's real
  deadline, min(kickoff, Sunday 16:00 ET) -- SNF/MNF lock early at Sunday
  4pm, so for those games specifically "closest-before-kickoff" is later
  than the real decision time; this archive was not rebuilt to that
  finer cutoff, flagged only**) is **new this session** -- added as a
  `--cutoff-mode` flag, run for the full 2024 season plus a 60-game spot
  check across 2020-2023.
- **A real bug was caught and fixed this session**: `kickoff_nearest`'s
  first run reused the `tuesday_noon` run's model (`MEX`, GFS MOS
  Extended) and scored *worse* than `tuesday_noon` despite issuing much
  closer to kickoff (temp MAE 9.97°F vs. 8.71°F on the same 2024 games) --
  because MEX's near-term granularity starts ~12-24h after its own
  issuance (it's built for days-out guidance), so "closest-before-kickoff"
  landed on a forecast valid ~18.5h from kickoff on average, not at
  kickoff. Switching `kickoff_nearest` to model `GFS` (short-range MOS,
  issuance+6h in 3h steps) fixed it: temp MAE dropped to **3.05°F**, temp
  correlation rose from 0.870 to **0.972**. Full numbers below.
- **GFS's IEM archive reaches back further than MEX's, covering this
  project's entire window for free**: **measured** this session (KDFW
  probes), GFS returns data for `runtime=2005-01-01T00:00Z` and
  `2009-09-01T00:00Z`, and returns nothing for `2003-01-01T00:00Z` -- so
  the lower bound sits between 2003 and 2005, comfortably before the
  project's 2009 start. **This means a kickoff-nearest forecast archive
  for 2009-2019 needs no NCEI HAS/AIRS order and no GRIB2 decoder --
  it's available today, free and instant, the same way as 2020-2025.**
  That pilot wasn't run this session (out of the modest-pilot scope given),
  but the archive-depth check that would gate it is done and positive.
- **What genuinely remains for 2009-2019 is narrower than the sourcing
  doc implied**: only a `tuesday_noon`-style (multi-day-out) forecast for
  2009-2019, since MEX (needed for that longer lead) is confirmed absent
  before 2020-07-12 and GFS's ~69h range doesn't reach 5-6 days out. The
  NCEI HAS/AIRS order form was reached this session (see below) -- no
  login/account needed to get there -- but placing an actual order
  requires submitting an email address into that form, which this
  unattended build session did not do (see "NCEI HAS/AIRS" section).
- **Owner-corrected 2026-08-20:** this build originally treated
  `tuesday_noon` as the pool-relevant cutoff and `kickoff_nearest` as a
  non-playable extra (see the corrected bullet above). Since our picks are
  editable up to each game's real deadline (**refined 2026-08-20:
  min(kickoff, Sunday 16:00 ET) -- SNF/MNF lock early**) rather than forced
  at Tuesday
  noon, **`kickoff_nearest` (temp r=0.972, MAE=3.05°F, the far tighter
  archive validated below) is the primary feature target for a
  pool-playable pregame weather feature, not the fallback** -- `tuesday_noon`
  remains useful only as context for what the grading line itself could
  have seen.

## Pipeline

### Station mapping

`registry/reference/stadium_station_map.csv` (built by a prior session,
**read** this session, not modified): 59 stadium-name rows keyed on the
schedules parquet's own `stadium` display string (not `stadium_id`, which
mislabels neutral-site international games). 46 rows are `mappable=true`
(domestic, including historical name variants of the same physical
buildings, e.g. Heinz Field/Acrisure Stadium both -> `KPIT`), mapping to 33
distinct ICAO stations; 13 are `mappable=false` (international: Wembley,
Tottenham, Munich x2, Frankfurt, Mexico City x2, Sao Paulo x2, Madrid,
Melbourne, Toronto), each carrying a note on why (**measured**, prior
session: zero GFS MOS stations in the international ICAO region probed).

A *different* stadium table, `registry/stadium_coordinates.json`
(lat/lon/tz, built today for ENV-03/04's travel-rest battery), exists in
the repo but was **not** used here -- this pipeline is station-based
(ICAO codes into the MOS API), not grid-based, so it has no lat/lon
dependency.

### Cutoff modes (`scripts/ingest_forecast_archive.py --cutoff-mode`)

| mode | decision cutoff | MOS model used | model's IEM archive start (measured) | max useful lead |
|---|---|---|---|---|
| `tuesday_noon` (default) | Tuesday 12:00 ET of the game's week | `MEX` (GFS MOS Extended, 00Z/12Z, +192h) | 2020-07-12 (confirmed absent 2020-06-01, 2015-09-01, 2009-09-01; present 2020-07-12) | ~192h -- reaches Sunday kickoffs from a Tuesday cutoff |
| `kickoff_nearest` (new) | kickoff itself, floored to the last 00Z/12Z cycle at-or-before kickoff | `GFS` (GFS MOS short-range, 00Z/12Z/06Z/18Z runs but only 00Z/12Z walked here, +69h) | ≤2005-01-01 (present 2005-01-01 and 2009-09-01; absent 2003-01-01) | ~69h -- more than enough for a same-week near-kickoff forecast |

Both modes share the same walk-backward machinery
(`candidate_runtimes`/`fetch_one_game`): starting from the cutoff floored
to the nearest 00Z/12Z cycle, step back 12h at a time (up to
`--max-lookback-steps`, default 10 = 5 days) until a non-empty bulletin is
found. This **never selects a bulletin issued after its cutoff** in either
mode -- point-in-time discipline is the same walk, just a different
starting cutoff and a different model. The model choice is table-driven
(`MOS_MODEL_BY_CUTOFF_MODE` in the script) and overridable with `--model`.

Within a found bulletin, `nearest_row` picks the row whose `ftime`
(forecast valid time) is closest to actual kickoff -- this can be before
*or* after kickoff (it's a valid-time nearness pick, not another
point-in-time constraint), which is fine: the point-in-time guarantee is
already enforced by the issuance-time walk, not by this step.

### Validation (`scripts/validate_forecast_archive.py`)

Joins each archive's own carried `actual_temp_f`/`actual_wind_mph`
(sourced from schedules) against `forecast_temp_f`/`forecast_wind_mph`,
restricted to outdoor games (`roof in {outdoors, open}`) with a successful
fetch. Reports Pearson r, MAE, and bias (forecast minus actual) for both
temp and wind -- this is **instrument validation** (does the forecast
correlate with what actually happened, the sanity check that nothing is
misjoined), not a cover-rate screen. No registry entry was written; none
was in scope for this build (per the task brief, a cover-rate cell is a
separate, out-of-scope step).

## Results (measured this session unless noted)

### `tuesday_noon`, full 2020-2025 REG population (built by a prior session, validated this session)

- 1,615 REG games; 17 unmappable international; 1,598 domestic, **100%
  fetched OK** (`coverage_of_domestic: 1.0`).
- 1,077 outdoor games with a successful fetch; 937 have both a forecast and
  actual temp (schedules' own temp/wind is incomplete for some seasons --
  the pre-existing "49% missing for 2022, 22% for 2023" note in
  `ROADMAP.md`'s ENV-01 row, not something this pipeline can fix).
- **Temp**: r = 0.897, MAE = 7.63°F, bias = -5.80°F (forecast runs cold --
  expected for a several-days-out lead over a seasonal cooling window).
- **Wind**: r = 0.387, MAE = 4.89 mph, bias = +2.79 mph.
- `lookback_steps_used` = 0 for all outdoor-OK rows: the Tuesday-noon
  cutoff itself always had a bulletin: no lookback was ever needed.

### `kickoff_nearest`, full 2024 REG season (this session, model=GFS after the fix)

- 272 REG games; 5 unmappable international; 267 domestic, **100% fetched
  OK**.
- Issuance lead ahead of kickoff: mean 5.5h, min 15 min, max 11.5h
  (`lead_issuance_h`, computed from `issuance_runtime_utc` vs.
  `kickoff_utc`) -- genuinely near-kickoff, unlike the MEX attempt below.
- 173 outdoor games fetched OK; 170 have both forecast and actual temp.
- **Temp**: r = 0.972, MAE = 3.05°F, bias = +0.49°F (essentially
  unbiased).
- **Wind**: r = 0.718, MAE = 3.11 mph, bias = +2.24 mph.
- For comparison, the **same 267 games** re-scored under `tuesday_noon`
  (filtered from the full archive above, not a separate fetch): temp r =
  0.899, MAE = 8.71°F, bias = -7.64°F -- confirms the accuracy gain is a
  genuine near-kickoff-lead effect, not a 2024-specific season quirk.

### The MEX mistake (kept here as a documented negative, not silently deleted)

The first `kickoff_nearest` run (before the model-selection fix) reused
`MEX` and produced temp r = 0.870, MAE = 9.97°F, bias = -8.53°F --
*worse* than `tuesday_noon` on the identical 267 games, despite a much
shorter issuance lead. Diagnosis (**measured**, from the raw output rows):
`forecast_valid_utc` sat a mean of **18.5h after kickoff** (min 12.5h,
max 23.75h) -- MEX's first available forecast row after a 12Z run is not
12Z+3h, it's roughly 12Z+24h, because MEX is an extended-range product not
meant for near-term use. The fix (switch to `GFS`, whose first row is
issuance+6h) is described above and confirmed by the improved numbers.
This is exactly the kind of "interval/number that looked bad" the project
is supposed to interrogate rather than discard -- in this case the
diagnosis pointed to a real, fixable modeling choice, not a dead end.

### `kickoff_nearest` spot check, 2020-2023 (this session, 15 games/season = 60 games, model=GFS)

Sampled as the first 15 chronological REG games of each season (early
weeks; population order is season-then-week, not random -- a modest,
declared-scope check, not a stratified sample).

- 60/60 domestic games fetched OK (5 seasons x 0 unmappable in this
  particular 60-game sample -- 2020's early weeks and the sampled 2021-2023
  games happened to all be domestic).
- 38 outdoor games OK; 20 have both forecast and actual temp (small n --
  this is a coverage/plumbing check, not a precision estimate).
- **Temp**: r = 0.924, MAE = 2.35°F, bias = +1.05°F.
- **Wind**: r = 0.520, MAE = 4.22 mph, bias = +2.89 mph.
- Confirms the GFS-based `kickoff_nearest` pipeline works identically
  across 2020-2023, consistent with the full-2024 result above.

## Coverage summary

| archive | seasons | games | domestic | fetch OK | coverage of domestic |
|---|---|---|---|---|---|
| `data/raw/forecast_archive/full_2020_2025/` (`tuesday_noon`, model=MEX) | 2020-2025 | 1,615 | 1,598 | 1,598 | 100% |
| `data/raw/forecast_archive/kickoff_nearest_2024/` (model=GFS) | 2024 | 272 | 267 | 267 | 100% |
| `data/raw/forecast_archive/spotcheck_kickoff_nearest_{2020,2021,2022,2023}/` (model=GFS) | 2020-2023 (15 games/season sample) | 60 | 60 | 60 | 100% |

Total on-disk footprint: **1.3 MB** (`du -sh data/raw/forecast_archive`) --
well inside "pilot, not a full mirror." `data/raw/**` is gitignored
(verified in `.gitignore`), so none of this is tracked.

## What remains for 2009-2019

Split by cutoff mode, since the answer is now different for each:

- **`kickoff_nearest` (near-term, model=GFS): nothing blocks it.**
  **Measured** this session: GFS's IEM archive returns data for
  `runtime=2005-01-01T00:00Z` and `2009-09-01T00:00Z` (KDFW probe), and
  the earliest project season is 2009. A 2009-2019 `kickoff_nearest` pull
  would use the exact same code path already validated above -- it simply
  wasn't run this session because the task scoped a modest 2024-centered
  pilot, not a full mirror.
- **`tuesday_noon` (multi-day-out, model=MEX): still genuinely blocked for
  2009-2019.** MEX's IEM archive starts 2020-07-12 (**measured**, both
  last session and re-confirmed this session: `runtime=2020-07-12T00:00Z`
  returns 15 rows, `2020-06-01T00:00Z` and `2015-09-01T00:00Z` return
  zero). No IEM-archived model has both the +150h-class range needed for
  a Tuesday-to-Sunday cutoff and pre-2020 coverage.
- **NCEI HAS/AIRS ("NDFD - By WMO Header") -- reached this session, not
  completed.** Navigated (via browser) from the AIRS dataset-selection
  page through to the actual order form for this dataset
  (`HAS.FileAppRouter?datasetname=9959_02...`). **Measured** directly from
  the form's own header list (not just the earlier documentation read):
  period of record confirmed **06/06/2004 to 08/16/2026** for the
  long-lived WMO headers, covering the full 2009-2025 project window. No
  login or account was required to reach this form. The form itself
  requires: (1) selecting one or more of ~600 **WMO Header** codes (3-4
  letter bulletin identifiers, e.g. `YDU`, `ZBA` -- these are **not** ICAO
  airport codes, so using this route would need a new WMO-header-to-
  stadium mapping, not the existing `stadium_station_map.csv`); (2) a
  start/end date range; (3) delivery destination (FTP) and a batch-submit
  toggle; (4) a **required Email Address field**; then a "Proceed With
  Order" submit button. This session did not fill in or submit that form:
  entering an email address and submitting an order is a form submission
  with personal data attached, which is outside what an unattended build
  session may do on its own -- it needs the project owner's explicit
  go-ahead, the same as any other form submission. If pursued: the
  remaining prerequisites after an order is placed are still (a) a
  WMO-header-to-stadium mapping (unbuilt), (b) a GRIB2 decoder (still not
  installed in this env -- unneeded for the MOS route used here, but would
  be needed for this GRIB2-format order), and (c) unknown turnaround time
  (the order was never submitted, so turnaround remains unverified).
- **Net effect on ROADMAP ENV-01's framing**: the sourcing doc's original
  worry ("2009-2019 needs a request-gated order of unverified turnaround")
  is now only true for the Tuesday-noon-style decision point. A
  kickoff-nearest pregame weather feature can be built for the entire
  2009-2025 window today, free, with the pipeline already in this repo.

## Files

- `scripts/ingest_forecast_archive.py` -- unchanged core logic; this
  session added `--cutoff-mode {tuesday_noon,kickoff_nearest}` (default
  `tuesday_noon`, preserving prior behavior/resumability), `--model`
  override, and `MOS_MODEL_BY_CUTOFF_MODE`. Record fields renamed
  `tuesday_cutoff_utc` -> `decision_cutoff_utc` (mode-agnostic) with a new
  `cutoff_mode` field; manifest gained `cutoff_mode` and `mos_model`. No
  other file in the repo referenced the old field name (**measured**,
  grep before renaming).
- `scripts/validate_forecast_archive.py` -- unchanged.
- `data/raw/forecast_archive/full_2020_2025/` -- pre-existing
  `tuesday_noon` archive (prior session), re-validated this session.
- `data/raw/forecast_archive/kickoff_nearest_2024/`,
  `spotcheck_kickoff_nearest_{2020..2023}/` -- new this session.
- `registry/reference/stadium_station_map.csv` -- pre-existing, read-only
  this session.

## Provenance

- **measured**: all coverage/fetch-status counts, all validate.py
  correlation/MAE/bias numbers, the MEX-vs-GFS forecast-valid-time gap
  diagnosis, the GFS/MEX archive-depth curl probes (KDFW, multiple
  runtimes), the ruff format/check pass on both edited files, the NCEI HAS
  order-form field list (via browser navigation, this session).
- **read**: `docs/weather_forecast_sourcing.md` (prior session's sourcing
  verdict and MOS pivot), `scripts/ingest_forecast_archive.py` and
  `scripts/validate_forecast_archive.py` docstrings and full source before
  editing, `registry/reference/stadium_station_map.csv`,
  `registry/stadium_coordinates.json` (confirmed not needed here),
  `data/raw/forecast_archive/full_2020_2025/manifest.json`.
- **inferred**: none load-bearing; the "net effect on ENV-01's framing"
  paragraph above is a reading of what the measured facts imply, not a new
  measurement itself.
