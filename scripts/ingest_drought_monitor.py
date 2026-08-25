"""Bulk ingestion of US Drought Monitor weekly county drought-severity stats.

Follow-up to `docs/archive/data_source_scout_v4.md` candidate #4. Source:
`usdmdataservices.unl.edu`'s public `CountyStatistics` REST API -- a
USDA/National Drought Mitigation Center product, free, no authentication.
**Measured** (2026-08-20, this session): `curl -sk
"https://usdmdataservices.unl.edu/api/CountyStatistics/GetDroughtSeverityStatisticsByAreaPercent?aoi=55009&startdate=1/1/2009&enddate=12/31/2025&statisticsType=1"
-H "Accept: application/json"` returned HTTP 200 with a genuine JSON array of
weekly rows for Brown County, WI (Lambeau Field's county), each carrying
`mapDate`/`validStart`/`validEnd`/`none`/`d0`..`d4` fields -- a single request
for the full 17-year range succeeded (no pagination needed), matching
`docs/archive/data_source_scout_v4.md`'s own earlier single-county/single-month
verification. `-k` (skip TLS verify) was needed because this environment's
`curl` lacks a working CA bundle for this host, a local quirk not a
site-side restriction. **Correction, same session**: this docstring
originally (wrongly, inferred and unverified) claimed Python's `urllib`
would sidestep the issue via its own certifi-backed bundle; a real ingestion
run instead hard-failed all 34 counties with `SSL: CERTIFICATE_VERIFY_FAILED
unable to get local issuer certificate` under `ssl.create_default_context()`'s
default OS-trust-store lookup -- the same host-specific CA-bundle gap, just
hitting Python too. Fixed by explicitly building the SSL context from the
`certifi` package's own bundle (`ssl.create_default_context(cafile=
certifi.where())`), **measured** working via a direct `urlopen` call (HTTP
200) before being wired into `_fetch` below.

Point-in-time note: each row is dated by `mapDate`/`validStart`/`validEnd`
(a non-overlapping weekly window). The USDM's official schedule says the map
is valid Tuesday at 08:00 ET and released Thursday at 08:30 ET. This script
records the source fields verbatim; the downstream join computes that exact,
DST-aware release timestamp and refuses to expose the row before it, rather
than treating `validStart` itself as available.

One request per stadium county for the full date range (not one per
county-year) -- the API accepts a 17-year span in a single call, so this is
34 requests total, not 34*17.

Usage:
    .\\.tools\\uv.exe run --no-sync python scripts/ingest_drought_monitor.py \\
        --out data/raw/drought --start-date 1/1/2009 --end-date 12/31/2025

Writes under --out/<snapshot>/ (default data/raw/drought/<UTC timestamp>,
gitignored under the repo's existing data/raw/** rule):

    <snapshot>/county/<FIPS>.parquet   weekly drought-severity rows for one
                                        stadium county.
    <snapshot>/index.parquet           concatenation of every county file
                                        present in this snapshot.
    <snapshot>/manifest.json           run metadata + coverage summary.
"""

from __future__ import annotations

import argparse
import json
import re
import ssl
import sys
import time
import urllib.error
import urllib.request
from datetime import UTC, datetime
from pathlib import Path

import certifi
import pandas as pd

# **Measured** 2026-08-20: a first ingestion run (Python 3.12, this Windows
# environment) hard-failed all 34 counties with `[SSL: CERTIFICATE_VERIFY_FAILED]
# unable to get local issuer certificate` using ssl.create_default_context()'s
# default (OS trust store) lookup against usdmdataservices.unl.edu -- the SAME
# host-specific CA-bundle gap the scout doc already found via `curl` (its `-k`
# workaround), just hitting Python's ssl module too, contradicting this
# script's own earlier (inferred, unverified) docstring claim that urllib's
# certifi-backed bundle would sidestep it. Fix: build the SSL context
# explicitly from certifi's bundle (measured working: a direct urlopen call
# with `context=ssl.create_default_context(cafile=certifi.where())` returned
# HTTP 200 for the same host/endpoint) instead of trusting the OS default.
_SSL_CONTEXT = ssl.create_default_context(cafile=certifi.where())

REPO = Path(__file__).resolve().parents[1]
STADIUM_COUNTY_FIPS_PATH = REPO / "registry/reference/stadium_county_fips.csv"
BASE_URL = (
    "https://usdmdataservices.unl.edu/api/CountyStatistics/"
    "GetDroughtSeverityStatisticsByAreaPercent"
    "?aoi={fips}&startdate={start}&enddate={end}&statisticsType=1"
)
USER_AGENT = "nfl-ats-research/0.1 (private research; contact ryanpmcintire@gmail.com)"
REQUEST_DELAY_SECONDS = 1.5
SNAPSHOT_DIR_RE = re.compile(r"^\d{8}T\d{6}Z$")


def resolve_snapshot_dir(out_dir: Path, snapshot: str | None) -> Path:
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


def _fetch(url: str, *, timeout: int = 30, retries: int = 3) -> bytes:
    last_error: Exception | None = None
    for attempt in range(retries):
        request = urllib.request.Request(
            url, headers={"User-Agent": USER_AGENT, "Accept": "application/json"}
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout, context=_SSL_CONTEXT) as response:
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
    dedup = in_scope.drop_duplicates(subset="county_fips")[
        ["county_fips", "county_name", "state_code"]
    ]
    return dedup.sort_values("county_fips").reset_index(drop=True)


def fetch_county(fips: str, start: str, end: str) -> pd.DataFrame:
    url = BASE_URL.format(fips=fips, start=start, end=end)
    raw = _fetch(url)
    rows = json.loads(raw)
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df["mapDate"] = pd.to_datetime(df["mapDate"])
    df["validStart"] = pd.to_datetime(df["validStart"])
    df["validEnd"] = pd.to_datetime(df["validEnd"])
    df = df.rename(
        columns={
            "fips": "county_fips",
            "county": "county_name",
            "state": "state_code",
            "mapDate": "map_date",
            "validStart": "valid_start",
            "validEnd": "valid_end",
        }
    )
    return df.sort_values("valid_start").reset_index(drop=True)


def ingest(snapshot_dir: Path, start: str, end: str, *, force: bool) -> dict:
    stadium_counties = load_stadium_counties()
    county_dir = snapshot_dir / "county"
    county_dir.mkdir(parents=True, exist_ok=True)

    processed: list[str] = []
    skipped: list[str] = []
    failed: list[str] = []
    totals = {"rows": 0}

    for i, row in stadium_counties.iterrows():
        fips = row["county_fips"]
        out_path = county_dir / f"{fips}.parquet"
        if out_path.exists() and not force:
            skipped.append(fips)
            continue
        if i > 0:
            time.sleep(REQUEST_DELAY_SECONDS)
        print(f"Fetching {fips} ({row['county_name']}, {row['state_code']})...")
        try:
            frame = fetch_county(fips, start, end)
        except Exception as error:
            print(f"  FAILED {fips}: {error}", file=sys.stderr)
            failed.append(fips)
            continue
        if frame.empty:
            print(f"  {fips}: 0 rows returned", file=sys.stderr)
            failed.append(fips)
            continue
        frame.to_parquet(out_path, index=False)
        totals["rows"] += len(frame)
        processed.append(fips)
        print(
            f"  {fips}: {len(frame)} weekly rows, "
            f"{frame['valid_start'].min()} to {frame['valid_start'].max()}"
        )

    all_county = sorted(county_dir.glob("*.parquet"))
    if all_county:
        combined = pd.concat([pd.read_parquet(p) for p in all_county], ignore_index=True)
        combined = combined.sort_values(["county_fips", "valid_start"]).reset_index(drop=True)
        combined.to_parquet(snapshot_dir / "index.parquet", index=False)
    else:
        combined = pd.DataFrame()

    coverage_by_county = (
        combined.groupby("county_fips")["valid_start"].agg(["min", "max", "count"]).reset_index()
        if len(combined)
        else pd.DataFrame()
    )

    manifest = {
        "source": (
            "https://usdmdataservices.unl.edu/api/CountyStatistics/"
            "GetDroughtSeverityStatisticsByAreaPercent (US Drought Monitor "
            "county stats API, no auth)"
        ),
        "fetched_at": datetime.now(UTC).isoformat(),
        "requested_date_range": [start, end],
        "counties_processed_this_run": processed,
        "counties_skipped_already_present": skipped,
        "counties_failed": failed,
        "request_delay_seconds": REQUEST_DELAY_SECONDS,
        "user_agent": USER_AGENT,
        "n_stadium_counties_in_scope": len(stadium_counties),
        "cumulative_index_rows": len(combined),
        "cumulative_counties_on_disk": len(all_county),
        "point_in_time_note": (
            "Each row is dated by mapDate/validStart/validEnd, a "
            "non-overlapping weekly window. The official USDM schedule is "
            "valid Tuesday 08:00 ET and released Thursday 08:30 ET; the "
            "downstream join computes that DST-aware release timestamp and "
            "must not treat validStart as the availability timestamp."
        ),
        "usage_note": (
            "Private research caching only. USDM is a public USDA/NDMC "
            "product; this session did not independently review its terms of "
            "use -- policy stance, not a verified legal fact."
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
    parser.add_argument("--out", type=Path, default=REPO / "data/raw/drought")
    parser.add_argument("--start-date", type=str, default="1/1/2009")
    parser.add_argument("--end-date", type=str, default="12/31/2025")
    parser.add_argument("--snapshot", type=str, default=None)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    snapshot_dir = resolve_snapshot_dir(args.out, args.snapshot)
    print(f"Snapshot dir: {snapshot_dir}")
    manifest = ingest(snapshot_dir, args.start_date, args.end_date, force=args.force)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
