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
    monkeypatch.setenv("NFL_ATS_REGISTRY_DIR", str(tmp_path / "registry"))

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
    monkeypatch.setenv("NFL_ATS_REGISTRY_DIR", str(tmp_path / "registry"))
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
    assert (margin_prediction_directory / "line_sweep.parquet").is_file()
    line_sweep = pd.read_parquet(margin_prediction_directory / "line_sweep.parquet")
    assert set(line_sweep["method"].unique()) == {"market", "fair_margin", "market_residual"}
    assert margin_prediction_output["line_sweep"]["rows"] == len(line_sweep)  # type: ignore[index]

    week_one_games = model_frame.loc[
        model_frame["season"].eq(2020) & model_frame["week"].eq(1), "game_id"
    ]
    lines_file = tmp_path / "lines.csv"
    lines_frame = pd.DataFrame({"game_id": week_one_games, "home_spread": 3.0})
    lines_frame.to_csv(lines_file, index=False)
    assert (
        cli.main(
            [
                "pool-card-at-lines",
                "--features",
                str(features),
                "--season",
                "2020",
                "--week",
                "1",
                "--min-train-games",
                "80",
                "--lines-file",
                str(lines_file),
                "--push-rule",
                "half",
            ]
        )
        == 0
    )
    pool_at_lines_output = _last_json(capsys.readouterr().out)
    pool_at_lines_directory = Path(str(pool_at_lines_output["artifact_directory"]))
    assert (pool_at_lines_directory / "predictions_at_lines.csv").is_file()
    assert (pool_at_lines_directory / "pool_card.csv").is_file()
    assert (pool_at_lines_directory / "pool_card.md").is_file()
    pool_at_lines_card = pd.read_csv(pool_at_lines_directory / "pool_card.csv")
    assert len(pool_at_lines_card) == len(week_one_games)
    assert pool_at_lines_card["confidence_rank"].tolist() == list(
        range(1, len(pool_at_lines_card) + 1)
    )

    assert (
        cli.main(
            [
                "key-number-calibration",
                "--features",
                str(features),
                "--start-season",
                "2020",
                "--min-train-games",
                "80",
            ]
        )
        == 0
    )
    key_number_output = _last_json(capsys.readouterr().out)
    key_number_directory = Path(str(key_number_output["artifact_directory"]))
    assert (key_number_directory / "key_number_mass.csv").is_file()
    assert (key_number_directory / "key_number_summary.csv").is_file()
    assert (key_number_directory / "line_bucket_reliability.csv").is_file()
    key_number_summary = pd.read_csv(key_number_directory / "key_number_summary.csv")
    assert set(key_number_summary["method"].unique()) == {
        "market",
        "fair_margin",
        "market_residual",
    }

    assert (
        cli.main(
            [
                "experiment",
                "compare",
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


def test_publish_predictions_does_not_record_by_default(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The safe default: an ordinary publish-predictions run must not append
    to the paper-decision ledger. Recording is a deliberate act
    (--record-decisions) -- see docs/prospective_evidence.md, 'Known
    divergence', for the incident this closes: a rehearsal run of this exact
    command, with its old opt-OUT --skip-clv-ledger flag not passed, wrote 16
    real rows to the real ledger on 2026-08-18."""

    monkeypatch.setenv("NFL_ATS_ARTIFACTS_DIR", str(tmp_path / "artifacts"))
    monkeypatch.setenv("NFL_ATS_REGISTRY_DIR", str(tmp_path / "registry"))
    destination = tmp_path / "CURRENT_PREDICTIONS.md"
    readme = tmp_path / "README.md"
    readme.write_text("x", encoding="utf-8")
    calls: list[Path] = []

    def fake_publish(
        artifacts_root: Path, *, destination: Path, readme_path: Path, data_root: Path | None = None
    ) -> dict:
        return {
            "model_id": "m",
            "season": 2026,
            "week": 1,
            "games": 1,
            "best_pick_game_id": None,
            "best_pick_tied": False,
            "historical_accuracy": 0.5,
            "destination": str(destination),
            "readme": str(readme_path),
            "published_at_utc": "t",
        }

    def fake_record(artifacts_root: Path) -> dict:
        calls.append(artifacts_root)
        return {"recorded": 1}

    def fake_overlay_record(artifacts_root: Path, data_root: Path) -> dict:
        calls.append(artifacts_root)
        return {"recorded": 1}

    def fake_nomination_record(artifacts_root: Path, data_root: Path) -> dict:
        calls.append(artifacts_root)
        return {"recorded": 1}

    monkeypatch.setattr(cli, "publish_active_predictions", fake_publish)
    monkeypatch.setattr(cli, "record_paper_decisions", fake_record)
    monkeypatch.setattr(cli, "record_overlay_challenger_decisions", fake_overlay_record)
    monkeypatch.setattr(cli, "record_nomination_challenger_decisions", fake_nomination_record)

    assert (
        cli.main(
            ["publish-predictions", "--destination", str(destination), "--readme", str(readme)]
        )
        == 0
    )
    payload = _last_json(capsys.readouterr().out)

    # Neither ledger call fires without the explicit flag.
    assert calls == []
    assert payload["clv_ledger"] == {
        "recorded": 0,
        "skipped": True,
        "reason": "pass --record-decisions to append this card's picks to the "
        "paper-decision ledger",
    }
    assert payload["overlay_challenger_ledger"] == {
        "recorded": 0,
        "skipped": True,
        "reason": "pass --record-decisions to append the overlay's picks to the "
        "prospective challenger ledger",
    }
    assert payload["nomination_challenger_ledger"] == {
        "recorded": 0,
        "skipped": True,
        "reason": "pass --record-decisions to append the v2 Best Pick nomination to "
        "the prospective challenger ledger",
    }


def test_publish_predictions_records_with_the_explicit_flag(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setenv("NFL_ATS_ARTIFACTS_DIR", str(tmp_path / "artifacts"))
    monkeypatch.setenv("NFL_ATS_REGISTRY_DIR", str(tmp_path / "registry"))
    destination = tmp_path / "CURRENT_PREDICTIONS.md"
    readme = tmp_path / "README.md"
    readme.write_text("x", encoding="utf-8")
    calls: list[Path] = []

    def fake_publish(
        artifacts_root: Path, *, destination: Path, readme_path: Path, data_root: Path | None = None
    ) -> dict:
        return {
            "model_id": "m",
            "season": 2026,
            "week": 1,
            "games": 1,
            "best_pick_game_id": None,
            "best_pick_tied": False,
            "historical_accuracy": 0.5,
            "destination": str(destination),
            "readme": str(readme_path),
            "published_at_utc": "t",
        }

    def fake_record(artifacts_root: Path) -> dict:
        calls.append(artifacts_root)
        return {"recorded": 1}

    overlay_calls: list[Path] = []

    def fake_overlay_record(artifacts_root: Path, data_root: Path) -> dict:
        overlay_calls.append(artifacts_root)
        return {"recorded": 1, "flip_count": 1}

    nomination_calls: list[Path] = []

    def fake_nomination_record(artifacts_root: Path, data_root: Path) -> dict:
        nomination_calls.append(artifacts_root)
        return {"recorded": 1, "nominated_game_id": "2026_01_AAA_BBB"}

    monkeypatch.setattr(cli, "publish_active_predictions", fake_publish)
    monkeypatch.setattr(cli, "record_paper_decisions", fake_record)
    monkeypatch.setattr(cli, "record_overlay_challenger_decisions", fake_overlay_record)
    monkeypatch.setattr(cli, "record_nomination_challenger_decisions", fake_nomination_record)

    assert (
        cli.main(
            [
                "publish-predictions",
                "--destination",
                str(destination),
                "--readme",
                str(readme),
                "--record-decisions",
            ]
        )
        == 0
    )
    payload = _last_json(capsys.readouterr().out)

    assert len(calls) == 1
    assert payload["clv_ledger"] == {"recorded": 1}
    assert len(overlay_calls) == 1
    assert payload["overlay_challenger_ledger"] == {"recorded": 1, "flip_count": 1}
    assert len(nomination_calls) == 1
    assert payload["nomination_challenger_ledger"] == {
        "recorded": 1,
        "nominated_game_id": "2026_01_AAA_BBB",
    }


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
