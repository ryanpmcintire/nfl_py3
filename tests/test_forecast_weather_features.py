from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from nfl_ats.data import DataContractError
from nfl_ats.forecast_weather_features import (
    attach_forecast_weather_features,
    load_forecast_archive,
)
from nfl_ats.provenance import sha256_file


def _archive_rows() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "game_id": ["early", "snf"],
            "kickoff_utc": ["2025-09-07T17:00:00Z", "2025-09-08T00:20:00Z"],
            "decision_cutoff_utc": [
                "2025-09-07T17:00:00Z",
                "2025-09-07T20:00:00Z",
            ],
            "issuance_runtime_utc": ["2025-09-07T12:00:00Z"] * 2,
            "cutoff_mode": ["pool_decision"] * 2,
            "fetch_status": ["ok"] * 2,
            "roof": ["outdoors"] * 2,
            "forecast_temp_f": [70.0, 66.0],
            "forecast_wind_mph": [8.0, 10.0],
            "forecast_precip_prob_pct": [10.0, 30.0],
        }
    )


def _write_archive(root: Path, rows: pd.DataFrame) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    archive = root / "forecasts.parquet"
    rows.to_parquet(archive, index=False)
    (root / "manifest.json").write_text(
        json.dumps(
            {
                "cutoff_mode": "pool_decision",
                "mos_model": "GFS",
                "files": {
                    "forecasts.parquet": {
                        "rows": len(rows),
                        "sha256": sha256_file(archive),
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    return archive


def test_pool_decision_archive_loads_with_snf_cut_off_at_sunday_1600(tmp_path: Path) -> None:
    loaded = load_forecast_archive(_write_archive(tmp_path, _archive_rows()))
    assert list(loaded["game_id"]) == ["early", "snf"]


@pytest.mark.parametrize(
    ("column", "value", "message"),
    [
        ("cutoff_mode", "kickoff_nearest", "outside pool_decision"),
        ("decision_cutoff_utc", "2025-09-08T00:20:00Z", "decision cutoff"),
        ("issuance_runtime_utc", "2025-09-07T22:00:00Z", "issued after"),
        ("fetch_status", "transport_error", "coverage failures"),
    ],
)
def test_archive_contract_rejects_leaky_or_incomplete_snf_rows(
    tmp_path: Path, column: str, value: str, message: str
) -> None:
    rows = _archive_rows()
    rows.loc[rows["game_id"].eq("snf"), column] = value
    with pytest.raises(DataContractError, match=message):
        load_forecast_archive(_write_archive(tmp_path, rows))


def test_archive_manifest_rejects_mutated_parquet(tmp_path: Path) -> None:
    archive = _write_archive(tmp_path, _archive_rows())
    archive.write_bytes(archive.read_bytes() + b"mutated")
    with pytest.raises(DataContractError, match="SHA-256"):
        load_forecast_archive(archive)


def test_attach_fails_closed_when_feature_population_is_not_covered(tmp_path: Path) -> None:
    archive = _write_archive(tmp_path, _archive_rows())
    features = pd.DataFrame({"game_id": ["early", "missing"], "spread_line": [1.0, 2.0]})
    with pytest.raises(DataContractError, match="missing 1 feature-table games"):
        attach_forecast_weather_features(features, archive_path=archive)


def test_post_decision_weather_mutation_cannot_enter_features(tmp_path: Path) -> None:
    before = _archive_rows()
    features = pd.DataFrame({"game_id": ["snf"], "spread_line": [1.0]})
    baseline = attach_forecast_weather_features(
        features, archive_path=_write_archive(tmp_path / "before", before)
    )

    after = before.copy()
    after.loc[after["game_id"].eq("snf"), "forecast_temp_f"] = 20.0
    after.loc[after["game_id"].eq("snf"), "issuance_runtime_utc"] = "2025-09-07T22:00:00Z"
    with pytest.raises(DataContractError, match="issued after"):
        attach_forecast_weather_features(
            features, archive_path=_write_archive(tmp_path / "after", after)
        )

    assert baseline.loc[0, "forecast_temp_f"] == 66.0
