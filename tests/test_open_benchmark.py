from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import pandas as pd
import pytest

from nfl_ats.data import DataContractError
from nfl_ats.open_benchmark import (
    MANIFEST_FILENAME,
    OBSERVATIONS_FILENAME,
    SUBMISSION_FILENAME,
    OpenBenchmarkDefinition,
    export_open_benchmark,
    export_open_benchmark_submission,
    validate_open_benchmark,
    validate_open_benchmark_submission,
)


def _observations() -> pd.DataFrame:
    rows = []
    games = (
        ("2022_01_A_B", 2022, 1, "2022-09-11T17:00:00Z", "train", 3.0),
        ("2022_02_C_D", 2022, 2, "2022-09-18T17:00:00Z", "train", -1.0),
        ("2023_01_E_F", 2023, 1, "2023-09-10T17:00:00Z", "validation", 0.0),
        ("2023_02_G_H", 2023, 2, "2023-09-17T17:00:00Z", "validation", 7.5),
        ("2024_01_I_J", 2024, 1, "2024-09-08T17:00:00Z", "test", None),
        ("2024_02_K_L", 2024, 2, "2024-09-15T17:00:00Z", "test", None),
    )
    for index, (game_id, season, week, kickoff, split, margin) in enumerate(games):
        kickoff_at = pd.Timestamp(kickoff)
        rows.append(
            {
                "game_id": game_id,
                "season": season,
                "week": week,
                "kickoff_utc": kickoff,
                "decision_time_utc": (kickoff_at - pd.Timedelta(days=5)).isoformat(),
                "inputs_observed_through_utc": (
                    kickoff_at - pd.Timedelta(days=5, hours=1)
                ).isoformat(),
                "home_team": game_id.split("_")[-2],
                "away_team": game_id.split("_")[-1],
                "spread_line": index - 2.5,
                "split": split,
                "rest_diff": index - 1.0,
                "ats_margin": margin,
                "cover_side": (
                    None
                    if margin is None
                    else "HOME"
                    if margin > 0
                    else "AWAY"
                    if margin < 0
                    else "PUSH"
                ),
            }
        )
    return pd.DataFrame(rows)


def _definition(**overrides: object) -> OpenBenchmarkDefinition:
    values: dict[str, object] = {
        "benchmark_id": "nfl-ats-open-v1",
        "dataset_version": "2026.1",
        "title": "NFL ATS point-in-time benchmark",
        "feature_columns": ("rest_diff",),
    }
    values.update(overrides)
    return OpenBenchmarkDefinition(**values)  # type: ignore[arg-type]


def _submission() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "game_id": "2024_02_K_L",
                "home_cover_probability": 0.4,
                "predicted_cover_side": "AWAY",
                "prediction_created_at_utc": "2024-09-09T12:00:00Z",
                "inputs_observed_through_utc": "2024-09-09T11:59:00Z",
            },
            {
                "game_id": "2024_01_I_J",
                "home_cover_probability": 0.6,
                "predicted_cover_side": "HOME",
                "prediction_created_at_utc": "2024-09-02T12:00:00Z",
                "inputs_observed_through_utc": "2024-09-02T11:59:00Z",
            },
        ]
    )


def test_dataset_export_is_deterministic_and_withholds_test_labels(tmp_path: Path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    result = export_open_benchmark(_observations(), first, definition=_definition())
    rerun = export_open_benchmark(
        _observations().sample(frac=1.0, random_state=7), second, definition=_definition()
    )

    assert result == rerun
    assert not result.publication_ready
    assert set(result.publication_blockers) == {
        "source licensing is not declared",
        "source provenance URLs are not declared",
        "external hosting location is not configured",
    }
    assert (first / OBSERVATIONS_FILENAME).read_bytes() == (
        second / OBSERVATIONS_FILENAME
    ).read_bytes()
    assert (first / MANIFEST_FILENAME).read_bytes() == (second / MANIFEST_FILENAME).read_bytes()
    exported = pd.read_csv(first / OBSERVATIONS_FILENAME, keep_default_na=False)
    assert (
        exported.loc[exported["split"].eq("test"), ["ats_margin", "cover_side"]].eq("").all().all()
    )


def test_declared_license_sources_and_hosting_are_publication_ready(tmp_path: Path) -> None:
    result = export_open_benchmark(
        _observations(),
        tmp_path / "benchmark",
        definition=_definition(
            license_spdx="CC-BY-4.0",
            source_urls=("https://example.test/source",),
            public_url="https://example.test/benchmark/v1",
        ),
    )
    assert result.publication_ready
    assert result.publication_blockers == ()


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda frame: frame.assign(
                inputs_observed_through_utc=frame["decision_time_utc"].where(
                    frame.index != 0, frame["kickoff_utc"]
                )
            ),
            "observed after decision",
        ),
        (
            lambda frame: frame.assign(
                ats_margin=frame["ats_margin"].where(frame["split"].ne("test"), 1.0),
                cover_side=frame["cover_side"].where(frame["split"].ne("test"), "HOME"),
            ),
            "must withhold",
        ),
        (
            lambda frame: frame.assign(
                split=frame["split"].where(frame["game_id"].ne("2023_02_G_H"), "train")
            ),
            "precede every validation",
        ),
    ],
)
def test_dataset_export_rejects_leakage(
    tmp_path: Path, mutate: Callable[[pd.DataFrame], pd.DataFrame], message: str
) -> None:
    changed = mutate(_observations())
    with pytest.raises(DataContractError, match=message):
        export_open_benchmark(changed, tmp_path / "benchmark", definition=_definition())


def test_dataset_validator_detects_tampering(tmp_path: Path) -> None:
    destination = tmp_path / "benchmark"
    export_open_benchmark(_observations(), destination, definition=_definition())
    observations = destination / OBSERVATIONS_FILENAME
    observations.write_bytes(observations.read_bytes().replace(b"2024_01_I_J", b"2024_01_X_Y"))

    with pytest.raises(DataContractError, match="hash mismatch"):
        validate_open_benchmark(destination)


def test_submission_round_trip_is_complete_deterministic_and_unscored(tmp_path: Path) -> None:
    benchmark = tmp_path / "benchmark"
    export_open_benchmark(_observations(), benchmark, definition=_definition())
    destination = tmp_path / "submission"
    result = export_open_benchmark_submission(
        _submission(), destination, benchmark_directory=benchmark, system_id="baseline-ridge"
    )

    assert result == validate_open_benchmark_submission(destination, benchmark_directory=benchmark)
    assert result.rows == 2
    exported = pd.read_csv(destination / SUBMISSION_FILENAME)
    assert exported["game_id"].tolist() == sorted(exported["game_id"])
    metadata = json.loads((destination / "submission.json").read_text(encoding="utf-8"))
    assert "score" not in metadata
    assert metadata["dataset_content_sha256"] == result.dataset_content_sha256


@pytest.mark.parametrize(
    ("submission", "message"),
    [
        (_submission().iloc[:1], "omits test game IDs"),
        (
            _submission().assign(
                prediction_created_at_utc=["2024-09-11T00:00:00Z", "2024-09-02T12:00:00Z"]
            ),
            "after the benchmark deadline",
        ),
        (
            _submission().assign(predicted_cover_side=["HOME", "HOME"]),
            "pick disagrees",
        ),
    ],
)
def test_submission_rejects_incomplete_or_invalid_rows(
    tmp_path: Path, submission: pd.DataFrame, message: str
) -> None:
    benchmark = tmp_path / "benchmark"
    export_open_benchmark(_observations(), benchmark, definition=_definition())
    with pytest.raises(DataContractError, match=message):
        export_open_benchmark_submission(
            submission,
            tmp_path / "submission",
            benchmark_directory=benchmark,
            system_id="candidate",
        )


def test_submission_validator_detects_tampering(tmp_path: Path) -> None:
    benchmark = tmp_path / "benchmark"
    destination = tmp_path / "submission"
    export_open_benchmark(_observations(), benchmark, definition=_definition())
    export_open_benchmark_submission(
        _submission(), destination, benchmark_directory=benchmark, system_id="candidate"
    )
    path = destination / SUBMISSION_FILENAME
    path.write_bytes(path.read_bytes().replace(b"0.59999999999999998", b"0.69999999999999998"))

    with pytest.raises(DataContractError, match="hash mismatch"):
        validate_open_benchmark_submission(destination, benchmark_directory=benchmark)
