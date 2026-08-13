from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from nfl_ats import cli
from nfl_ats.snapshots import write_snapshot


def _last_json(output: str) -> dict[str, object]:
    return json.loads(output)


def test_cli_data_workflow(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    schedules_and_stats: tuple[pd.DataFrame, pd.DataFrame],
) -> None:
    data_root = tmp_path / "data"
    artifacts_root = tmp_path / "artifacts"
    monkeypatch.setenv("NFL_ATS_DATA_DIR", str(data_root))
    monkeypatch.setenv("NFL_ATS_ARTIFACTS_DIR", str(artifacts_root))

    assert cli.main(["doctor"]) == 0
    assert _last_json(capsys.readouterr().out)["latest_snapshot"] is None

    schedules, stats = schedules_and_stats

    def fake_fetch(seasons: list[int], raw_root: Path, team_stat_seasons: list[int]):
        return write_snapshot(
            schedules,
            stats,
            seasons,
            raw_root,
            "20220101T000000Z",
            team_stat_seasons=team_stat_seasons,
        )

    monkeypatch.setattr(cli, "fetch_nflverse", fake_fetch)
    assert cli.main(["ingest", "--start-season", "2022", "--end-season", "2022"]) == 0
    ingest_output = _last_json(capsys.readouterr().out)
    assert ingest_output["snapshot_id"] == "20220101T000000Z"

    assert cli.main(["doctor"]) == 0
    assert _last_json(capsys.readouterr().out)["latest_snapshot"] is not None

    assert cli.main(["build-features", "--ewm-span", "3", "--min-periods", "1"]) == 0
    feature_output = _last_json(capsys.readouterr().out)
    assert feature_output["rows"] == 5
    assert (data_root / "processed" / "game_features.parquet").is_file()

    monkeypatch.setattr(
        cli,
        "check_nflverse_contract",
        lambda schedule_season, stats_season: {
            "schedule_season": schedule_season,
            "stats_season": stats_season,
        },
    )
    assert (
        cli.main(
            [
                "smoke-source",
                "--schedule-season",
                "2026",
                "--stats-season",
                "2025",
            ]
        )
        == 0
    )
    smoke_output = _last_json(capsys.readouterr().out)
    assert smoke_output == {"schedule_season": 2026, "stats_season": 2025}


def test_cli_model_workflow(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    model_frame: pd.DataFrame,
) -> None:
    data_root = tmp_path / "data"
    artifacts_root = tmp_path / "artifacts"
    monkeypatch.setenv("NFL_ATS_DATA_DIR", str(data_root))
    monkeypatch.setenv("NFL_ATS_ARTIFACTS_DIR", str(artifacts_root))
    features = data_root / "processed" / "game_features.parquet"
    features.parent.mkdir(parents=True)
    model_frame.to_parquet(features, index=False)

    assert (
        cli.main(
            [
                "backtest",
                "--features",
                str(features),
                "--start-season",
                "2020",
                "--min-train-games",
                "80",
                "--bootstrap-samples",
                "20",
            ]
        )
        == 0
    )
    backtest_output = _last_json(capsys.readouterr().out)
    backtest_directory = Path(str(backtest_output["artifact_directory"]))
    assert (backtest_directory / "metrics.json").is_file()
    assert (backtest_directory / "paper_ledger.parquet").is_file()
    assert (backtest_directory / "portfolio_metrics.json").is_file()
    assert (backtest_directory / "bankroll_simulation.json").is_file()
    assert (backtest_directory / "bankroll_paths.parquet").is_file()
    assert (backtest_directory / "uncertainty.csv").is_file()
    assert (backtest_directory / "model_card.json").is_file()
    assert (backtest_directory / "model_card.md").is_file()

    assert (
        cli.main(
            [
                "margin-backtest",
                "--features",
                str(features),
                "--start-season",
                "2020",
                "--min-train-games",
                "80",
                "--bootstrap-samples",
                "20",
            ]
        )
        == 0
    )
    margin_output = _last_json(capsys.readouterr().out)
    assert float(margin_output["timing"]["total_seconds"]) >= 0.0  # type: ignore[index]
    assert float(margin_output["timing"]["uncertainty_seconds"]) >= 0.0  # type: ignore[index]
    margin_directory = Path(str(margin_output["artifact_directory"]))
    assert (margin_directory / "summary.csv").is_file()
    assert (margin_directory / "season_summary.csv").is_file()
    assert (margin_directory / "predictions.parquet").is_file()
    assert (margin_directory / "uncertainty.csv").is_file()

    assert (
        cli.main(
            [
                "margin-predict",
                "--features",
                str(features),
                "--season",
                "2020",
                "--week",
                "1",
                "--min-train-games",
                "80",
            ]
        )
        == 0
    )
    margin_prediction_output = _last_json(capsys.readouterr().out)
    margin_prediction_directory = Path(str(margin_prediction_output["artifact_directory"]))
    assert (margin_prediction_directory / "predictions.csv").is_file()
    assert (margin_prediction_directory / "recommendations.csv").is_file()
    assert (margin_prediction_directory / "pool_card.csv").is_file()
    assert (margin_prediction_directory / "prediction_safety.json").is_file()
    assert (margin_prediction_directory / "straight_up_pool_market_residual.csv").is_file()
    assert margin_prediction_output["ats_method"] == "market_residual"
    assert margin_prediction_output["synchronization_status"] == "SYNCHRONIZED"
    active_model = json.loads(
        (artifacts_root / "active_ats_model.json").read_text(encoding="utf-8")
    )
    assert active_model["model_id"] == margin_prediction_output["active_model_id"]
    assert (
        active_model["historical_evaluation"]["artifact"]
        == (Path("margins") / margin_directory.name).as_posix()
    )
    assert (
        active_model["weekly_forecast"]["artifact"]
        == (Path("margin_predictions") / margin_prediction_directory.name).as_posix()
    )

    assert (
        cli.main(
            [
                "experiment",
                "--features",
                str(features),
                "--start-season",
                "2020",
                "--feature-sets",
                "market,market_context",
                "--min-train-games",
                "80",
                "--bootstrap-samples",
                "20",
            ]
        )
        == 0
    )
    experiment_output = _last_json(capsys.readouterr().out)
    experiment_directory = Path(str(experiment_output["artifact_directory"]))
    assert (experiment_directory / "paired_comparisons.csv").is_file()

    assert (
        cli.main(
            [
                "predict",
                "--features",
                str(features),
                "--season",
                "2020",
                "--week",
                "1",
                "--min-train-games",
                "80",
            ]
        )
        == 0
    )
    prediction_output = _last_json(capsys.readouterr().out)
    output = Path(str(prediction_output["artifact_directory"]))
    assert (output / "recommendations.md").is_file()
    assert (output / "recommendations.csv").is_file()
    assert (output / "model.joblib").is_file()
    assert (output / "coefficients.csv").is_file()
    assert (output / "pool_card.csv").is_file()
    assert (output / "pool_card.md").is_file()
    assert (output / "prediction_safety.json").is_file()
    recommendation_markdown = (output / "recommendations.md").read_text(encoding="utf-8")
    assert "ATS pick" in recommendation_markdown
    assert "Model probability" in recommendation_markdown


def test_cli_reports_user_errors(tmp_path: Path) -> None:
    missing = tmp_path / "missing.parquet"
    with pytest.raises(SystemExit) as error:
        cli.main(["backtest", "--features", str(missing)])
    assert error.value.code == 2

    with pytest.raises(SystemExit) as error:
        cli.main(["ingest", "--start-season", "2022", "--end-season", "2021"])
    assert error.value.code == 2


def test_cli_handoff(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        cli,
        "write_session_handoff",
        lambda repo_root, artifacts_root, destination: {
            "destination": str(destination),
            "branch": "master",
        },
    )

    assert cli.main(["handoff", "--destination", "SESSION.md"]) == 0
    assert _last_json(capsys.readouterr().out) == {
        "branch": "master",
        "destination": "SESSION.md",
    }

    monkeypatch.setattr(
        cli,
        "check_session_handoff",
        lambda repo_root, artifacts_root, handoff_path: {
            "handoff": str(handoff_path),
            "status": "CURRENT",
        },
    )
    assert cli.main(["handoff", "--destination", "SESSION.md", "--check"]) == 0
    assert _last_json(capsys.readouterr().out) == {
        "handoff": "SESSION.md",
        "status": "CURRENT",
    }
