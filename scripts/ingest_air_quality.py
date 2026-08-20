"""Bulk ingestion of EPA AQS daily county-level Air Quality Index (AQI) data.

Follow-up to `docs/data_source_scout_v4.md` candidate #3 (EPA AirNow), but
this script deliberately uses a DIFFERENT, no-auth EPA endpoint instead of
AirNow itself. **Measured** (2026-08-20): a live call to AirNow's historical
observation endpoint with a dummy key returns HTTP 401 (the scout doc's own
finding) -- AirNow requires a free account signup. Per this session's
instructions, a no-auth path is strongly preferred over any account signup,
so this script instead pulls EPA's own pre-generated **AQS** (Air Quality
System) annual "daily AQI by county" files, published at
`aqs.epa.gov/aqsweb/airdata/` with zero authentication. **Measured**: `curl -I
https://aqs.epa.gov/aqsweb/airdata/daily_aqi_by_county_2020.zip` returns HTTP
200, `Content-Type: application/zip`, no auth challenge; the unzipped CSV's
header row is `"State Name","county Name","State Code","County Code","Date",
"AQI","Category","Defining Parameter","Defining Site","Number of Sites
Reporting"` (**measured**, same session). Years 2009-2025 were all
**measured** present (HTTP 200) before this script was written.

Point-in-time note (label: inferred + read, stated honestly per AGENTS.md):
these are retrospective annual archive files -- EPA publishes each year's
file on a rolling basis well after the year completes, so this data is safe
to use for BACKTEST features (each row's `Date` is a real measurement date;
the county's AQI on that calendar day is not something that could leak
future information into an earlier day) but this exact ingestion path CANNOT
serve a live/prospective 2026-in-season feed -- there is no
"daily_aqi_by_county_2026.zip" until well after the season. A live feed
needs the AirNow REST API (registration required, see docs/
data_source_scout_v4.md sec 3) or the EPA AirNow API's real-time
observation endpoints -- out of scope for this ingestion session; noted here
so a future session does not conflate "this backtest archive is joinable"
with "a live 2026 feature exists."

One file per calendar year covers the ENTIRE country (~1-2 MB zipped), not
per-county, so this script downloads each year once and filters locally to
this project's stadium counties (see
registry/reference/stadium_county_fips.csv, built by
scripts/build_stadium_county_fips.py) rather than making one request per
county-year.

Usage:
    .\\.tools\\uv.exe run --no-sync python scripts/ingest_air_quality.py \\
        --out data/raw/air_quality --start-year 2009 --end-year 2025

Writes under --out/<snapshot>/ (default data/raw/air_quality/<UTC timestamp>,
gitignored under the repo's existing data/raw/** rule), matching this
project's established raw-snapshot convention (see scripts/
ingest_injury_news.py's module docstring for the same pattern):

    <snapshot>/annual/<YYYY>.parquet   full county-filtered AQI rows for
                                        that year (all US counties matching
                                        this project's stadium-county list).
    <snapshot>/index.parquet           concatenation of every annual file
                                        present in this snapshot.
    <snapshot>/manifest.json           run metadata + coverage summary.
"""

from __future__ import annotations

import argparse
import io
import json
import re
import sys
import time
import urllib.error
import urllib.request
import zipfile
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
STADIUM_COUNTY_FIPS_PATH = REPO / "registry/reference/stadium_county_fips.csv"
BASE_URL = "https://aqs.epa.gov/aqsweb/airdata/daily_aqi_by_county_{year}.zip"
USER_AGENT = "nfl-ats-research/0.1 (private research; contact ryanpmcintire@gmail.com)"
REQUEST_DELAY_SECONDS = 2.0
SNAPSHOT_DIR_RE = re.compile(r"^\d{8}T\d{6}Z$")


def resolve_snapshot_dir(out_dir: Path, snapshot: str | None) -> Path:
    """Same convention as scripts/ingest_injury_news.py's resolve_snapshot_dir:
    a manifest.json must never sit directly at out_dir, only inside a
    timestamped snapshot subdirectory (nfl_ats.snapshots.latest_snapshot()
    treats any directory directly under data/raw/ with a manifest.json as a
    candidate schedules snapshot)."""

    out_dir.mkdir(parents=True, exist_ok=True)
    if snapshot is not None:
        snapshot_dir = out_dir / snapshot
        snapshot_dir.mkdir(parents=True, exist_ok=True)
        return snapshot_dir

    existing = sorted(p for p in out_dir.glob("*") if p.is_dir() and SNAPSHOT_DIR_RE.match(p.name))
    if existing:
        return existing[-1]

    new_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    snapshot_dir = out_dir / new_id
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    return snapshot_dir


def _fetch(url: str, *, timeout: int = 60, retries: int = 3) -> bytes:
    last_error: Exception | None = None
    for attempt in range(retries):
        request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return response.read()
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError) as error:
            last_error = error
            print(f"  fetch failed ({attempt + 1}/{retries}) {url}: {error}", file=sys.stderr)
            time.sleep(3.0 * (attempt + 1))
    assert last_error is not None
    raise last_error


def load_stadium_counties() -> pd.DataFrame:
    df = pd.read_csv(STADIUM_COUNTY_FIPS_PATH, dtype={"county_fips": str})
    in_scope = df[df["in_scope"]].copy()
    in_scope["county_fips"] = in_scope["county_fips"].str.zfill(5)
    return in_scope


def fetch_year(year: int, county_fips: set[str]) -> pd.DataFrame:
    url = BASE_URL.format(year=year)
    raw = _fetch(url)
    with zipfile.ZipFile(io.BytesIO(raw)) as zf:
        csv_name = next(n for n in zf.namelist() if n.endswith(".csv"))
        with zf.open(csv_name) as fh:
            df = pd.read_csv(
                fh,
                dtype={"State Code": str, "County Code": str},
            )
    df["county_fips"] = df["State Code"].str.zfill(2) + df["County Code"].str.zfill(3)
    filtered = df[df["county_fips"].isin(county_fips)].copy()
    filtered = filtered.rename(
        columns={
            "State Name": "state_name",
            "county Name": "county_name",
            "Date": "date",
            "AQI": "aqi",
            "Category": "category",
            "Defining Parameter": "defining_parameter",
            "Defining Site": "defining_site",
            "Number of Sites Reporting": "n_sites_reporting",
        }
    )[
        [
            "county_fips",
            "state_name",
            "county_name",
            "date",
            "aqi",
            "category",
            "defining_parameter",
            "defining_site",
            "n_sites_reporting",
        ]
    ]
    filtered["date"] = pd.to_datetime(filtered["date"])
    filtered["year"] = year
    return filtered.sort_values(["county_fips", "date"]).reset_index(drop=True)


def ingest(snapshot_dir: Path, start_year: int, end_year: int, *, force: bool) -> dict:
    stadium_counties = load_stadium_counties()
    county_fips = set(stadium_counties["county_fips"])
    annual_dir = snapshot_dir / "annual"
    annual_dir.mkdir(parents=True, exist_ok=True)

    processed: list[int] = []
    skipped: list[int] = []
    failed: list[int] = []
    totals = {"rows": 0}

    for i, year in enumerate(range(start_year, end_year + 1)):
        out_path = annual_dir / f"{year}.parquet"
        if out_path.exists() and not force:
            skipped.append(year)
            continue
        if i > 0:
            time.sleep(REQUEST_DELAY_SECONDS)
        print(f"Fetching {year}...")
        try:
            frame = fetch_year(year, county_fips)
        except Exception as error:
            print(f"  FAILED year {year}: {error}", file=sys.stderr)
            failed.append(year)
            continue
        frame.to_parquet(out_path, index=False)
        totals["rows"] += len(frame)
        processed.append(year)
        n_counties = frame["county_fips"].nunique()
        print(f"  {year}: {len(frame)} county-day rows across {n_counties} counties")

    all_annual = sorted(annual_dir.glob("*.parquet"))
    if all_annual:
        combined = pd.concat([pd.read_parquet(p) for p in all_annual], ignore_index=True)
        combined = combined.sort_values(["county_fips", "date"]).reset_index(drop=True)
        combined.to_parquet(snapshot_dir / "index.parquet", index=False)
    else:
        combined = pd.DataFrame()

    coverage_by_county = (
        combined.groupby("county_fips")["date"].agg(["min", "max", "count"]).reset_index()
        if len(combined)
        else pd.DataFrame()
    )

    manifest = {
        "source": (
            "https://aqs.epa.gov/aqsweb/airdata/ "
            "(EPA AQS pre-generated daily AQI by county, no auth)"
        ),
        "fetched_at": datetime.now(UTC).isoformat(),
        "requested_year_range": [start_year, end_year],
        "years_processed_this_run": processed,
        "years_skipped_already_present": skipped,
        "years_failed": failed,
        "request_delay_seconds": REQUEST_DELAY_SECONDS,
        "user_agent": USER_AGENT,
        "n_stadium_counties_in_scope": int(stadium_counties["county_fips"].nunique()),
        "stadium_county_fips": sorted(county_fips),
        "cumulative_index_rows": len(combined),
        "cumulative_years_on_disk": len(all_annual),
        "counties_with_zero_rows": sorted(
            county_fips - set(combined["county_fips"].unique()) if len(combined) else county_fips
        ),
        "point_in_time_note": (
            "Retrospective annual archive files; each row's `date` is a real "
            "measurement date, safe for backtest features. This ingestion path "
            "does NOT serve a live/prospective feed -- a live 2026 feed needs "
            "the key-gated AirNow REST API instead (see docs/"
            "data_source_scout_v4.md sec 3 and docs/environmental_exposures.md)."
        ),
        "usage_note": (
            "Private research caching only, matching this project's existing "
            "external-source precedent. EPA AQS data is a public-domain federal "
            "government publication (no copyright restriction is expected, but "
            "this session did not independently review EPA's terms of use -- "
            "policy stance, not a verified legal fact)."
        ),
    }
    with (snapshot_dir / "manifest.json").open("w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)

    if len(coverage_by_county):
        print("\nPer-county coverage:")
        print(coverage_by_county.to_string(index=False))

    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=REPO / "data/raw/air_quality")
    parser.add_argument("--start-year", type=int, default=2009)
    parser.add_argument("--end-year", type=int, default=2025)
    parser.add_argument("--snapshot", type=str, default=None)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    snapshot_dir = resolve_snapshot_dir(args.out, args.snapshot)
    print(f"Snapshot dir: {snapshot_dir}")
    manifest = ingest(snapshot_dir, args.start_year, args.end_year, force=args.force)
    print(json.dumps({k: v for k, v in manifest.items() if k != "stadium_county_fips"}, indent=2))


if __name__ == "__main__":
    main()
