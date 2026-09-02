"""Capture EPA AirNow's public, no-auth hourly observation file.

AirNow publishes ``HourlyAQObs_yyyymmddhh.dat`` at approximately 35 minutes
past each UTC hour under ``https://files.airnowtech.org/airnow/YYYY/YYYYMMDD/``.
The file is site-level CSV and includes AQS IDs.  For US sites, the nine-digit
local AQS ID begins with state (2 digits) + county (3 digits), so this capture
uses that source-owned county key to join active NFL stadium counties.  It does
not guess from geographic proximity.

Every run creates an immutable UTC-stamped directory.  ``source.dat`` is the
verbatim response, ``stadium_aqi.parquet`` contains the maximum active-site AQI
in each required stadium county, and ``manifest.json`` is written last with
the exact URL, capture timestamp, and SHA-256 hashes.  A stale file, malformed
schema, or missing required current-stadium county fails the capture closed.

Official format documentation:
https://docs.airnowapi.org/docs/HourlyAQObsFactSheet.pdf
"""

from __future__ import annotations

import argparse
import io
import json
import ssl
import urllib.error
import urllib.request
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import BinaryIO

import certifi
import pandas as pd

from nfl_ats.io import atomic_bytes, atomic_json, atomic_parquet, run_id
from nfl_ats.provenance import sha256_file

REPO = Path(__file__).resolve().parents[1]
STADIUM_COUNTIES = REPO / "registry/reference/stadium_county_fips.csv"
DEFAULT_ROOT = REPO / "data/raw/airnow_hourly"
BASE_URL = "https://files.airnowtech.org/airnow"
MAX_SOURCE_AGE = timedelta(hours=3)
PUBLICATION_SAFE_MINUTE = 40
AQI_COLUMNS = ("OZONE_AQI", "PM10_AQI", "PM25_AQI", "NO2_AQI")
EXPECTED_COLUMNS = (
    "AQSID",
    "SiteName",
    "Status",
    "EPARegion",
    "Latitude",
    "Longitude",
    "Elevation",
    "GMTOffset",
    "CountryCode",
    "StateName",
    "ValidDate",
    "ValidTime",
    "DataSource",
    "ReportingArea_PipeDelimited",
    "OZONE_AQI",
    "PM10_AQI",
    "PM25_AQI",
    "NO2_AQI",
    "OZONE_Measured",
    "PM10_Measured",
    "PM25_Measured",
    "NO2_Measured",
    "PM25",
    "PM25_Unit",
    "OZONE",
    "OZONE_Unit",
    "NO2",
    "NO2_Unit",
    "CO",
    "CO_Unit",
    "SO2",
    "SO2_Unit",
    "PM10",
    "PM10_Unit",
)


class AirNowCaptureError(RuntimeError):
    """The public file could not produce a complete, safe stadium capture."""


def source_url(observed_at_utc: datetime) -> str:
    instant = observed_at_utc.astimezone(UTC)
    return f"{BASE_URL}/{instant:%Y}/{instant:%Y%m%d}/HourlyAQObs_{instant:%Y%m%d%H}.dat"


def candidate_hours(captured_at_utc: datetime) -> list[datetime]:
    """Newest likely published hours, allowing five minutes after the stated ~:35."""

    instant = captured_at_utc.astimezone(UTC)
    hour = instant.replace(minute=0, second=0, microsecond=0)
    first_offset = 0 if instant.minute >= PUBLICATION_SAFE_MINUTE else 1
    return [hour - timedelta(hours=offset) for offset in range(first_offset, first_offset + 3)]


def fetch_latest(
    captured_at_utc: datetime,
    opener: Callable[..., BinaryIO] = urllib.request.urlopen,
    attempts: list[str] | None = None,
) -> tuple[bytes, str]:
    attempted_urls = attempts if attempts is not None else []
    context = ssl.create_default_context(cafile=certifi.where())
    for hour in candidate_hours(captured_at_utc):
        url = source_url(hour)
        attempted_urls.append(url)
        try:
            with opener(url, context=context, timeout=30) as response:
                payload = response.read()
        except (OSError, urllib.error.URLError):
            continue
        if payload:
            return payload, url
    raise AirNowCaptureError(f"No non-empty AirNow hourly file found: {attempted_urls}")


def current_stadiums(path: Path = STADIUM_COUNTIES) -> tuple[pd.DataFrame, int]:
    stadiums = pd.read_csv(path, dtype={"county_fips": str})
    domestic = stadiums.loc[stadiums["in_scope"]].copy()
    current_season = int(domestic["season_max"].max())
    current = domestic.loc[domestic["season_max"] >= current_season].copy()
    current["county_fips"] = current["county_fips"].str.zfill(5)
    if current.empty or current["county_fips"].isna().any():
        raise AirNowCaptureError("Current domestic stadium registry is empty or unmapped")
    return current[["stadium", "county_fips", "county_name", "state_code"]], current_season


def parse(payload: bytes) -> tuple[pd.DataFrame, dict[str, int]]:
    frame = pd.read_csv(io.BytesIO(payload), dtype={"AQSID": str})
    if tuple(frame.columns) != EXPECTED_COLUMNS:
        raise AirNowCaptureError(
            f"Unexpected AirNow schema ({len(frame.columns)} fields): {list(frame.columns)}"
        )
    if frame.empty:
        raise AirNowCaptureError("AirNow file has no observations")

    frame = frame.loc[frame["CountryCode"].eq("US") & frame["Status"].eq("Active")].copy()
    active_us_rows = len(frame)
    local_aqs_id = frame["AQSID"].str.removeprefix("840")
    valid_id = local_aqs_id.str.fullmatch(r"\d{9}")
    invalid_aqs_id_rows = int((~valid_id).sum())
    frame = frame.loc[valid_id].copy()
    local_aqs_id = local_aqs_id.loc[valid_id]
    if frame.empty:
        raise AirNowCaptureError("AirNow file has no county-addressable active US sites")
    frame["county_fips"] = local_aqs_id.str[:5]
    frame[list(AQI_COLUMNS)] = frame[list(AQI_COLUMNS)].apply(pd.to_numeric, errors="coerce")
    frame["aqi"] = frame[list(AQI_COLUMNS)].max(axis=1)
    frame = frame.loc[frame["aqi"].notna()].copy()
    parameters = {column: column.removesuffix("_AQI").lower() for column in AQI_COLUMNS}
    frame["parameter"] = frame[list(AQI_COLUMNS)].idxmax(axis=1).map(parameters)
    frame["observed_at_utc"] = pd.to_datetime(
        frame["ValidDate"] + " " + frame["ValidTime"], format="%m/%d/%Y %H:%M", utc=True
    )
    observed_hours = frame["observed_at_utc"].drop_duplicates()
    if len(observed_hours) != 1:
        raise AirNowCaptureError(f"Expected one UTC observation hour, got {len(observed_hours)}")
    return frame, {
        "active_us_rows": active_us_rows,
        "non_county_aqs_id_rows_quarantined": invalid_aqs_id_rows,
        "usable_active_us_aqi_rows": len(frame),
    }


def map_stadiums(
    observations: pd.DataFrame, stadiums: pd.DataFrame, captured_at_utc: datetime
) -> pd.DataFrame:
    # Stable ordering makes a tie select the lexically first AQS site.
    county = (
        observations.sort_values(["county_fips", "aqi", "AQSID"], ascending=[True, False, True])
        .drop_duplicates("county_fips")
        .rename(columns={"AQSID": "aqs_site_id", "SiteName": "site_name"})
    )
    columns = [
        "county_fips",
        "observed_at_utc",
        "aqi",
        "parameter",
        "aqs_site_id",
        "site_name",
        "Latitude",
        "Longitude",
    ]
    mapped = stadiums.merge(county[columns], on="county_fips", how="left", validate="many_to_one")
    missing = mapped.loc[mapped["aqi"].isna(), ["stadium", "county_fips"]]
    if not missing.empty:
        raise AirNowCaptureError(
            "No usable active-site AQI for required stadium counties: "
            + json.dumps(missing.to_dict("records"))
        )
    mapped["available_at_utc"] = pd.Timestamp(captured_at_utc.astimezone(UTC))
    return mapped


def capture(
    out_root: Path = DEFAULT_ROOT,
    *,
    now: datetime | None = None,
    opener: Callable[..., BinaryIO] = urllib.request.urlopen,
    stadium_path: Path = STADIUM_COUNTIES,
) -> Path:
    captured_at = (now or datetime.now(UTC)).astimezone(UTC)
    snapshot = out_root / run_id(captured_at)
    snapshot.mkdir(parents=True, exist_ok=False)
    attempts: list[str] = []
    source: str | None = None
    try:
        payload, source = fetch_latest(captured_at, opener, attempts)
        raw_path = snapshot / "source.dat"
        atomic_bytes(payload, raw_path)
        observations, parse_audit = parse(payload)
        observed_at = observations["observed_at_utc"].iloc[0].to_pydatetime()
        if source != source_url(observed_at):
            raise AirNowCaptureError(
                f"AirNow URL hour does not match payload observation hour: {source}"
            )
        age = captured_at - observed_at
        if age < timedelta(0) or age > MAX_SOURCE_AGE:
            raise AirNowCaptureError(f"AirNow observation hour is stale/future: age={age}")
        stadiums, registry_season = current_stadiums(stadium_path)
        mapped = map_stadiums(observations, stadiums, captured_at)
        data_path = snapshot / "stadium_aqi.parquet"
        atomic_parquet(mapped, data_path)
        manifest = {
            "status": "complete",
            "snapshot_id": snapshot.name,
            "captured_at_utc": captured_at.isoformat(),
            "observed_at_utc": observed_at.isoformat(),
            "source_url": source,
            "attempted_urls": attempts,
            "registry_season": registry_season,
            "required_stadiums": len(stadiums),
            "required_counties": int(stadiums["county_fips"].nunique()),
            "parse_audit": parse_audit,
            "files": [
                {"path": path.name, "bytes": path.stat().st_size, "sha256": sha256_file(path)}
                for path in (raw_path, data_path)
            ],
        }
        atomic_json(manifest, snapshot / "manifest.json")
        return Path(snapshot)
    except Exception as exc:
        atomic_json(
            {
                "status": "failed",
                "snapshot_id": snapshot.name,
                "captured_at_utc": captured_at.isoformat(),
                "source_url": source,
                "attempted_urls": attempts,
                "error": f"{type(exc).__name__}: {exc}",
            },
            snapshot / "manifest.json",
        )
        if isinstance(exc, AirNowCaptureError):
            raise
        raise AirNowCaptureError(str(exc)) from exc


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_ROOT)
    args = parser.parse_args()
    snapshot = capture(args.out)
    manifest = json.loads((snapshot / "manifest.json").read_text(encoding="utf-8"))
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
