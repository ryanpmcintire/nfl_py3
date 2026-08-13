from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from nfl_ats import cli
from nfl_ats.experiments import OutcomeProfileExperimentResult


def _last_json(capsys: pytest.CaptureFixture[str]) -> dict[str, object]:
    return json.loads(capsys.readouterr().out)


def _snapshot(snapshot_id: str) -> SimpleNamespace:
    return SimpleNamespace(snapshot_id=snapshot_id)


def test_build_participation_features_cli_writes_research_contract(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    data_root = tmp_path / "data"
    monkeypatch.setenv("NFL_ATS_DATA_DIR", str(data_root))
    monkeypatch.setattr(cli, "_load_features", lambda path: pd.DataFrame({"season": [2022]}))
    monkeypatch.setattr(cli, "latest_player_snapshot", lambda root: _snapshot("players"))
    monkeypatch.setattr(cli, "latest_pbp_snapshot", lambda root: _snapshot("pbp"))
    monkeypatch.setattr(cli, "latest_player_value_snapshot", lambda root: _snapshot("values"))
    monkeypatch.setattr(
        cli, "latest_participation_snapshot", lambda root: _snapshot("participation")
    )
    monkeypatch.setattr(cli, "load_pbp_snapshot", lambda snapshot: pd.DataFrame())
    monkeypatch.setattr(cli, "load_participation_snapshot", lambda snapshot: pd.DataFrame())
    monkeypatch.setattr(
        cli,
        "build_season_lagged_player_ratings",
        lambda participation, pbp, target_seasons: pd.DataFrame(
            {
                "target_season": [2022],
                "source_start_season": [2019],
                "source_end_season": [2021],
                "source_plays": [1000],
                "player_id": ["P1"],
            }
        ),
    )
    monkeypatch.setattr(
        cli,
        "load_player_snapshot",
        lambda snapshot: (pd.DataFrame(), pd.DataFrame(), pd.DataFrame()),
    )
    monkeypatch.setattr(cli, "load_player_value_snapshot", lambda snapshot: pd.DataFrame())
    monkeypatch.setattr(
        cli,
        "enrich_with_player_features",
        lambda *args, **kwargs: pd.DataFrame(
            {
                "home_injury_offense_participation_value_lost": [0.1],
                "away_injury_offense_participation_value_lost": [0.0],
            }
        ),
    )
    destination = data_root / "processed" / "participation_features.parquet"
    ratings = data_root / "processed" / "ratings.parquet"

    assert (
        cli.main(
            [
                "build-participation-features",
                "--destination",
                str(destination),
                "--ratings-destination",
                str(ratings),
            ]
        )
        == 0
    )
    output = _last_json(capsys)
    assert output["source_participation_snapshot"] == "participation"
    assert output["ratings_rows"] == 1
    assert destination.is_file()
    assert ratings.is_file()
    assert destination.with_name("participation_features.manifest.json").is_file()


def test_build_learned_availability_features_cli_writes_target_evaluation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    data_root = tmp_path / "data"
    monkeypatch.setenv("NFL_ATS_DATA_DIR", str(data_root))
    monkeypatch.setattr(cli, "_load_features", lambda path: pd.DataFrame({"season": [2022]}))
    monkeypatch.setattr(cli, "latest_player_snapshot", lambda root: _snapshot("players"))
    monkeypatch.setattr(cli, "latest_pbp_snapshot", lambda root: _snapshot("pbp"))
    monkeypatch.setattr(cli, "latest_player_value_snapshot", lambda root: _snapshot("values"))
    raw = pd.DataFrame({"raw": [1]})
    monkeypatch.setattr(cli, "load_player_snapshot", lambda snapshot: (raw, raw, raw))
    monkeypatch.setattr(cli, "canonicalize_injuries", lambda frame: frame)
    monkeypatch.setattr(cli, "canonicalize_rosters", lambda frame: frame)
    monkeypatch.setattr(cli, "canonicalize_snaps", lambda frame: frame)
    monkeypatch.setattr(cli, "attach_snap_player_ids", lambda snaps, rosters: snaps)
    monkeypatch.setattr(cli, "load_pbp_snapshot", lambda snapshot: pd.DataFrame())
    monkeypatch.setattr(cli, "load_player_value_snapshot", lambda snapshot: pd.DataFrame())
    monkeypatch.setattr(
        cli,
        "build_availability_outcomes",
        lambda *args, **kwargs: pd.DataFrame({"season": [2021], "unavailable": [1.0]}),
    )
    monkeypatch.setattr(
        cli,
        "build_season_lagged_availability_rates",
        lambda outcomes, target_seasons: pd.DataFrame(
            {"target_season": [2022], "unavailability_probability": [0.8]}
        ),
    )
    monkeypatch.setattr(cli, "score_availability_rates", lambda outcomes, rates: outcomes)
    monkeypatch.setattr(
        cli,
        "summarize_availability_scores",
        lambda scored: pd.DataFrame(
            [
                {"method": "fixed", "player_games": 100, "brier_score": 0.10},
                {"method": "learned", "player_games": 100, "brier_score": 0.08},
            ]
        ),
    )
    monkeypatch.setattr(
        cli,
        "enrich_with_player_features",
        lambda *args, **kwargs: pd.DataFrame({"game_id": ["game"]}),
    )
    destination = data_root / "processed" / "availability_features.parquet"
    rates = data_root / "processed" / "rates.parquet"
    evaluation = data_root / "processed" / "availability.csv"

    assert (
        cli.main(
            [
                "build-learned-availability-features",
                "--destination",
                str(destination),
                "--rates-destination",
                str(rates),
                "--evaluation-destination",
                str(evaluation),
            ]
        )
        == 0
    )
    output = _last_json(capsys)
    assert output["availability_brier_improvement"] == pytest.approx(0.02)
    assert output["availability_evaluation_player_games"] == 100
    assert destination.is_file() and rates.is_file() and evaluation.is_file()


def _fake_profile_result(label: str) -> OutcomeProfileExperimentResult:
    return OutcomeProfileExperimentResult(
        summary=pd.DataFrame([{"feature_profile": "player_value", "cover_accuracy": 0.52}]),
        season_summary=pd.DataFrame(
            [{"feature_profile": "player_value", "season": 2022, "cover_accuracy": 0.52}]
        ),
        predictions=pd.DataFrame(
            [
                {
                    "feature_profile": "player_value",
                    "game_id": "game",
                    "season": 2022,
                    "week": 1,
                    "home_cover": 1.0,
                    "home_cover_probability": 0.51,
                    "label": label,
                }
            ]
        ),
    )


def _paired_rows() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "baseline_feature_set": "fixed",
                "candidate_feature_set": "candidate",
                "metric": "accuracy_improvement",
                "estimate": 0.0,
                "lower": -0.01,
                "upper": 0.01,
                "block": "week",
            }
        ]
    )


def test_fixed_player_research_ablation_commands_write_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    artifacts = tmp_path / "artifacts"
    monkeypatch.setenv("NFL_ATS_ARTIFACTS_DIR", str(artifacts))
    monkeypatch.setattr(cli, "_load_features", lambda path: pd.DataFrame({"path": [str(path)]}))
    monkeypatch.setattr(
        cli,
        "run_outcome_profile_experiment",
        lambda features, **kwargs: _fake_profile_result(str(features.iloc[0, 0])),
    )
    monkeypatch.setattr(cli, "paired_feature_comparisons", lambda *args, **kwargs: _paired_rows())
    monkeypatch.setattr(cli, "artifact_provenance", lambda *args, **kwargs: {"test": True})

    assert cli.main(["participation-ablation", "--bootstrap-samples", "10"]) == 0
    participation = _last_json(capsys)
    participation_root = Path(str(participation["artifact_directory"]))
    assert participation["hypothesis_frozen_before_scoring"] is True
    assert (participation_root / "summary.csv").is_file()
    assert (participation_root / "paired_comparisons.csv").is_file()

    assert cli.main(["availability-ablation", "--bootstrap-samples", "10"]) == 0
    availability = _last_json(capsys)
    availability_root = Path(str(availability["artifact_directory"]))
    assert availability["hypothesis_frozen_before_scoring"] is True
    assert (availability_root / "summary.csv").is_file()
    assert (availability_root / "predictions.parquet").is_file()
    assert (availability_root / "metadata.json").is_file()
