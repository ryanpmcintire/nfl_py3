from __future__ import annotations

import csv
import hashlib
import io
import json
import sys
import urllib.error
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts import capture_airnow_hourly as airnow
from scripts import capture_scheduler
from scripts.build_environmental_exposure_join import (
    asof_merge_live_aqi,
    load_airnow_captures,
)


def _payload(
    *rows: dict[str, object], valid_date: str = "09/02/2026", valid_time: str = "21:00"
) -> bytes:
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=airnow.EXPECTED_COLUMNS)
    writer.writeheader()
    for values in rows:
        row = dict.fromkeys(airnow.EXPECTED_COLUMNS, "")
        row.update(
            {
                "Status": "Active",
                "CountryCode": "US",
                "ValidDate": valid_date,
                "ValidTime": valid_time,
            }
        )
        row.update(values)
        writer.writerow(row)
    return output.getvalue().encode()


def test_candidate_hours_respect_the_publication_boundary() -> None:
    before = airnow.candidate_hours(datetime(2026, 9, 2, 15, 39, tzinfo=UTC))
    after = airnow.candidate_hours(datetime(2026, 9, 2, 15, 40, tzinfo=UTC))

    assert [value.hour for value in before] == [14, 13, 12]
    assert [value.hour for value in after] == [15, 14, 13]


def test_capture_falls_back_when_the_current_hour_is_not_yet_available(tmp_path: Path) -> None:
    registry = tmp_path / "stadiums.csv"
    _stadiums(registry)
    payload = _payload(
        {"AQSID": "840421010001", "OZONE_AQI": 42},
        {"AQSID": "840550090001", "OZONE_AQI": 35},
        valid_time="14:00",
    )

    def previous_hour(url: str, **_kwargs: object) -> io.BytesIO:
        if url.endswith("2026090214.dat"):
            return io.BytesIO(payload)
        raise urllib.error.URLError("current hour is not published")

    snapshot = airnow.capture(
        tmp_path / "captures",
        now=datetime(2026, 9, 2, 15, 40, tzinfo=UTC),
        opener=previous_hour,
        stadium_path=registry,
    )
    manifest = json.loads((snapshot / "manifest.json").read_text(encoding="utf-8"))

    assert manifest["source_url"].endswith("2026090214.dat")
    assert [url[-14:] for url in manifest["attempted_urls"]] == [
        "2026090215.dat",
        "2026090214.dat",
    ]


def _stadiums(path: Path) -> None:
    pd.DataFrame(
        {
            "stadium": ["Alpha", "Beta"],
            "in_scope": [True, True],
            "season_max": [2026, 2026],
            "county_fips": ["42101", "55009"],
            "county_name": ["Philadelphia", "Brown"],
            "state_code": ["PA", "WI"],
        }
    ).to_csv(path, index=False)


def test_capture_is_immutable_hashed_and_maps_source_owned_counties(tmp_path: Path) -> None:
    payload = _payload(
        {
            "AQSID": "840421010001",
            "SiteName": "Philadelphia A",
            "Latitude": 39.9,
            "Longitude": -75.1,
            "OZONE_AQI": 42,
            "PM25_AQI": 70,
        },
        {
            "AQSID": "550090001",
            "SiteName": "Brown A",
            "Latitude": 44.5,
            "Longitude": -88.0,
            "OZONE_AQI": 35,
        },
    )
    registry = tmp_path / "stadiums.csv"
    _stadiums(registry)
    now = datetime(2026, 9, 2, 22, 15, tzinfo=UTC)

    snapshot = airnow.capture(
        tmp_path / "captures",
        now=now,
        opener=lambda *_args, **_kwargs: io.BytesIO(payload),
        stadium_path=registry,
    )

    manifest = json.loads((snapshot / "manifest.json").read_text(encoding="utf-8"))
    mapped = pd.read_parquet(snapshot / "stadium_aqi.parquet").set_index("stadium")
    assert manifest["status"] == "complete"
    assert manifest["source_url"].endswith("HourlyAQObs_2026090221.dat")
    assert manifest["required_stadiums"] == manifest["required_counties"] == 2
    assert mapped.loc["Alpha", "county_fips"] == "42101"
    assert mapped.loc["Alpha", "aqi"] == 70
    assert mapped.loc["Alpha", "parameter"] == "pm25"
    for entry in manifest["files"]:
        assert (
            entry["sha256"] == hashlib.sha256((snapshot / entry["path"]).read_bytes()).hexdigest()
        )
    with pytest.raises(FileExistsError):
        airnow.capture(
            tmp_path / "captures",
            now=now,
            opener=lambda *_args, **_kwargs: io.BytesIO(payload),
            stadium_path=registry,
        )


def test_capture_fails_closed_when_required_county_has_no_usable_aqi(tmp_path: Path) -> None:
    registry = tmp_path / "stadiums.csv"
    _stadiums(registry)
    payload = _payload({"AQSID": "840421010001", "SiteName": "Only one", "OZONE_AQI": 42})

    with pytest.raises(airnow.AirNowCaptureError, match="55009"):
        airnow.capture(
            tmp_path / "captures",
            now=datetime(2026, 9, 2, 22, 15, tzinfo=UTC),
            opener=lambda *_args, **_kwargs: io.BytesIO(payload),
            stadium_path=registry,
        )

    manifest = json.loads(
        (tmp_path / "captures/20260902T221500Z/manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["status"] == "failed"


def test_capture_fails_closed_on_stale_source_and_records_every_attempt(tmp_path: Path) -> None:
    registry = tmp_path / "stadiums.csv"
    _stadiums(registry)
    payload = _payload(
        {"AQSID": "840421010001", "OZONE_AQI": 42},
        {"AQSID": "840550090001", "OZONE_AQI": 35},
    )

    def only_oldest(url: str, **_kwargs: object) -> io.BytesIO:
        if url.endswith("2026090221.dat"):
            return io.BytesIO(payload)
        raise urllib.error.URLError("not published")

    with pytest.raises(airnow.AirNowCaptureError, match="stale"):
        airnow.capture(
            tmp_path / "captures",
            now=datetime(2026, 9, 3, 0, 39, tzinfo=UTC),
            opener=only_oldest,
            stadium_path=registry,
        )

    manifest = json.loads(
        (tmp_path / "captures/20260903T003900Z/manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["status"] == "failed"
    assert len(manifest["attempted_urls"]) == 3
    assert manifest["source_url"].endswith("2026090221.dat")


def test_live_aqi_join_uses_capture_time_not_earlier_observation_time() -> None:
    games = pd.DataFrame(
        {
            "game_id": ["before", "after", "stale"],
            "county_fips": ["42101"] * 3,
            "decision_at_utc": pd.to_datetime(
                ["2026-09-02T20:59Z", "2026-09-02T21:05Z", "2026-09-03T00:01Z"]
            ),
        }
    )
    captures = pd.DataFrame(
        {
            "stadium": ["Alpha"],
            "county_fips": ["42101"],
            "available_at_utc": pd.to_datetime(["2026-09-02T21:00Z"]),
            "observed_at_utc": pd.to_datetime(["2026-09-02T20:00Z"]),
            "aqi": [40.0],
            "parameter": ["ozone"],
            "aqs_site_id": ["421010001"],
            "site_name": ["A"],
        }
    )

    baseline = asof_merge_live_aqi(games, captures).set_index("game_id")
    future = captures.copy()
    future["available_at_utc"] = pd.Timestamp("2026-09-02T21:06Z")
    future["aqi"] = 500.0
    changed = asof_merge_live_aqi(games, pd.concat([captures, future])).set_index("game_id")

    assert pd.isna(baseline.loc["before", "live_aqi"])
    assert baseline.loc["after", "live_aqi"] == changed.loc["after", "live_aqi"] == 40.0
    assert pd.isna(baseline.loc["stale", "live_aqi"])


@pytest.mark.parametrize("tampered", ["source.dat", "stadium_aqi.parquet"])
def test_loader_rejects_a_tampered_complete_snapshot(tmp_path: Path, tampered: str) -> None:
    snapshot = tmp_path / "20260902T221500Z"
    snapshot.mkdir()
    data = snapshot / "stadium_aqi.parquet"
    source = snapshot / "source.dat"
    source.write_bytes(b"source")
    pd.DataFrame({"aqi": [40]}).to_parquet(data, index=False)
    manifest = {
        "status": "complete",
        "files": [
            {"path": source.name, "sha256": hashlib.sha256(source.read_bytes()).hexdigest()},
            {"path": data.name, "sha256": hashlib.sha256(data.read_bytes()).hexdigest()},
        ],
    }
    (snapshot / "manifest.json").write_text(
        json.dumps(manifest),
        encoding="utf-8",
    )
    with (snapshot / tampered).open("ab") as stream:
        stream.write(b"tampered")

    with pytest.raises(ValueError, match="SHA-256"):
        load_airnow_captures(tmp_path)


def test_scheduler_captures_after_publication_and_before_checkpoint() -> None:
    job = {item.name: item for item in capture_scheduler.SCHEDULE}["airnow_tue_checkpoint"]

    assert job.day == "tue"
    assert job.at == "11:40"
    assert job.grace_minutes == 15
    assert job.added_on == "2026-09-02"
    assert job.dedupe_dir == "data/raw/airnow_hourly"
    assert any(part.endswith("capture_airnow_hourly.py") for part in job.command)
