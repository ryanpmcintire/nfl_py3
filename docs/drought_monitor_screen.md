# US Drought Monitor county screen

## Decision first

**Measured** (`nfl-ats experiment run
registry/experiment_specs/environmental_battery_drought_severe_grass_opener.json
--replace`, artifact `artifacts/experiment_runner/20260820T152033Z`): the
predeclared opener-grade severe-drought/grass cell has a full-slate effect of
**+0.1618 accuracy points** across 1,486 games, with 64 flagged games,
`probability_positive=0.7193`, and a week-blocked interval of
[-0.3903, +0.6789].

**Measured** (same artifact): the raw flagged-minus-complement home-cover gap
is +3.7579 points, and the flag occupies 4.31% of the evaluated slate.

**Inferred**: this is evidence to retain because the current opener-grade
direction is favorable, but it does not by itself define or test a model-pick
overlay against the production probability rule.

**Reported** (project owner, unverified here): the decision-relevant production
baseline is 53.4% at the opener/production-rule grade; the 52.10% historical
field in `artifacts/active_ats_model.json` is a different quantity and must not
supersede it.

**Inferred**: no production pick or prospective challenger should change from
this screen alone, because a like-for-like 53.4%-baseline comparison and a
fully specified pick transform have not been measured; this is an EV-scope
distinction, not a confidence-threshold rejection.

## Source and reproducible ingestion

**Read** (`scripts/ingest_drought_monitor.py`): the no-auth ingestion path is
the US Drought Monitor county-statistics endpoint
`CountyStatistics/GetDroughtSeverityStatisticsByAreaPercent`, queried once per
distinct stadium county for the requested historical range.

**Measured** (`data/raw/drought/20260820T111232Z/manifest.json`): the cached
snapshot has 30,192 rows, 34 counties, 888 weekly rows per county, no failed
county, and source coverage from 2008-12-30 through 2025-12-30 for the requested
2009-2025 range.

**Read** ([official USDM current-map page](https://droughtmonitor.unl.edu/)):
the map's data cutoff is Tuesday 08:00 Eastern and the map is released Thursday
08:30 Eastern.

**Measured** (`tests/test_drought_monitor.py`): the join now converts that
Thursday 08:30 `America/New_York` wall-clock release to UTC with DST awareness,
selects the older map at 08:29:59, and exposes the new map at 08:30:00.

**Measured** (`tests/test_drought_monitor.py`): changing every drought value
whose official release is after an earlier decision checkpoint leaves that
checkpoint's joined value unchanged; this is the drought-family leakage
regression test.

**Read** (`scripts/build_environmental_exposure_join.py`): the experiment's
decision checkpoint is Tuesday noon Eastern, so the current Tuesday-valid map
is unavailable and the previously released weekly map is selected.

**Measured** (`python scripts/build_environmental_exposure_join.py`): rebuilding
the current local join produced 4,842 domestic in-scope games and separately
retained 60 out-of-scope international games.

**Measured** (same command): the drought join reports 100% fresh coverage in
every season from 2009 through 2024, 98.9% in 2025, and 0% fresh coverage in
2026 because the cached historical snapshot stops at 2025.

**Measured** (`scripts/ingest_drought_monitor.py --live`,
`data/raw/drought/20260902T221432Z/manifest.json`): the no-auth live path was
exercised on 2026-09-02 and returned 31,348 rows from the 2008-12-30 through
2026-08-25 maps for all 34 stadium counties, with zero failed or missing
counties.

**Read** (`scripts/ingest_drought_monitor.py`): default refreshes now create a
new immutable timestamped snapshot, use atomic parquet/JSON writes, record the
exact source URLs and SHA-256/byte-size provenance for every output, and raise
after writing a failed audit manifest if any requested county is absent.

**Measured** (`python scripts/build_environmental_exposure_join.py`): the
new full-history live snapshot integrates end to end; 2009-2025 drought
coverage is 100% fresh, while the full 2026 schedule is 5.7% fresh as of
2026-09-02 because future games correctly remain beyond the current map.

## Stadium-to-county provenance

**Read** (`scripts/build_stadium_county_fips.py` and
`registry/reference/stadium_county_fips.csv`): stadium coordinates come from
`registry/stadium_coordinates.json`, domestic/international scope comes from
`registry/reference/stadium_station_map.csv`, and county FIPS/name/state come
from the FCC Census Block API response recorded as `fcc_status`.

**Measured** (`Import-Csv registry/reference/stadium_county_fips.csv`): the
reference contains 82 schedule stadium-name rows: 67 domestic rows covering 34
distinct counties and 15 explicitly out-of-scope international rows.

**Measured** (`tests/test_drought_monitor.py`): every domestic row has a
five-digit county FIPS, county name, state code, and `fcc_status=OK`.

**Inferred**: county-level drought is a broad exposure proxy rather than a
stadium-soil measurement, so a stronger follow-up would add field maintenance,
irrigation, resurfacing, and grass-species data without changing the present
county-map result after seeing those new variables.

## Frozen low-dimensional family

**Read** (`docs/environmental_exposures.md` section 6.B): before the original
screen saw signs, the family was fixed as outdoor-exposed grass games whose
home county had fresh D2-or-worse coverage over at least 50% of county area.

**Read** (`registry/experiment_specs/environmental_battery_drought_severe_grass_opener.json`):
the declarative reproduction freezes one threshold (`d2_area_threshold=50`),
one boolean exposure (`outdoor AND grass AND D2+`), one outcome (opener-graded
ATS home cover), week blocking, season secondary blocking, 20,000 draws, and
seed 20260820.

**Read** (same spec): positive sign is an arbitrary fixed reporting convention
for the non-directional mechanism, not a directional hypothesis invented after
the earlier signs were known.

**Inferred**: the mechanism is cumulative soil-moisture deficit affecting turf
firmness, footing, and play speed over a longer horizon than day-of weather;
county drought may be informative only after interaction with stadium-specific
irrigation and maintenance.

**Read** (`src/nfl_ats/experiment_runner.py`, flag builder
`drought_severe_grass`): stale rows and rows not officially released by the
decision timestamp are excluded before the flag is evaluated.

## Result interpretation and registry disposition

**Measured** (`registry/weak_signals.json`, key
`environmental_battery_drought_severe_grass_opener`): the result is recorded as
`unresolved_below_power`, with no closing ground and
`probability_positive=0.7193`.

**Measured** (`artifacts/experiment_runner/20260820T152033Z`): the week-blocked
result uses 107 blocks and has standard error 0.2729 accuracy points.

**Measured** (same artifact): the season-blocked diagnostic has only six blocks
and is mechanically labeled degenerate; its `probability_positive` is 0.7311
and is secondary to the nondegenerate week-blocked result.

**Read** (`docs/environmental_exposures.md` section 7): the earlier bespoke
opener screen used an older 1,464-game archive intersection, found 63 flagged
games, +0.1286 accuracy points, and `probability_positive=0.6764`.

**Inferred**: the current declarative result is a refresh of the same frozen
cell on a larger local opener archive, not an independent replication and not
permission to multiply the evidence as if there were two separate studies.

**Inferred**: the next decision-relevant experiment is one predeclared overlay
that changes only flagged games and is evaluated head-to-head against the 53.4%
production probability rule at the opener; stadium irrigation/maintenance data
should be collected before defining that overlay rather than tuning multiple
drought thresholds on the already-read window.

## Exact commands

```powershell
$env:UV_CACHE_DIR='F:\Repos\nfl_py3\.uv-cache'

# Refresh the no-auth archive through a requested end date.
.\.tools\uv.exe run --no-sync python scripts\ingest_drought_monitor.py `
  --start-date 1/1/2009 --end-date 12/31/2025

# Live refresh through today's UTC date into a new immutable snapshot.
.\.tools\uv.exe run --no-sync python scripts\ingest_drought_monitor.py `
  --live

# Rebuild the point-in-time stadium-county join.
.\.tools\uv.exe run --no-sync python scripts\build_environmental_exposure_join.py

# Reproduce and record the opener-grade category-3 result.
.\.tools\uv.exe run --no-sync nfl-ats experiment run `
  registry\experiment_specs\environmental_battery_drought_severe_grass_opener.json `
  --replace

# Focused leakage/runner checks.
.\.tools\uv.exe run --no-sync pytest `
  tests\test_drought_monitor.py tests\test_experiment_runner.py -q
```
