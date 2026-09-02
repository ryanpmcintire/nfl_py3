from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from nfl_ats import cli
from nfl_ats.active_model import ACTIVE_ATS_MODEL_VERSION
from nfl_ats.clv import PAPER_DECISION_COLUMNS, paper_decision_ledger_path
from nfl_ats.data import DataContractError
from nfl_ats.io import atomic_json, atomic_parquet
from nfl_ats.lines import apply_external_lines
from nfl_ats.outcomes import fit_margin_models_for_week
from nfl_ats.pick_refresh import load_pick_revisions
from nfl_ats.snapshots import write_snapshot


def _last_json(output: str) -> dict[str, object]:
    return json.loads(output)


def _record_weak_signal_args(name: str, **extra: str) -> list[str]:
    args = [
        "weak-signals",
        "record",
        "--name",
        name,
        "--description",
        "a technical description of the measurement",
        "--source",
        "docs/example.md",
        "--effect",
        "0.25",
        "--effect-units",
        "accuracy_points",
        "--classification",
        "unresolved_below_power",
        "--league",
        "nfl",
        "--season-start",
        "2020",
        "--season-end",
        "2024",
    ]
    for flag, value in extra.items():
        args += [f"--{flag.replace('_', '-')}", value]
    return args


def test_weak_signals_record_stores_plain_summary_and_category(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("NFL_ATS_REGISTRY_DIR", str(tmp_path / "registry"))
    assert (
        cli.main(
            _record_weak_signal_args(
                "cli_plain_summary_demo",
                plain_summary="A short sentence a fan can read on its own.",
                category="onfield",
            )
        )
        == 0
    )
    captured = capsys.readouterr()
    assert "warning:" not in captured.err

    registry_path = tmp_path / "registry" / "weak_signals.json"
    stored = json.loads(registry_path.read_text(encoding="utf-8"))
    entry = stored["signals"]["cli_plain_summary_demo"]
    assert entry["plain_summary"] == "A short sentence a fan can read on its own."
    assert entry["category"] == "onfield"


def test_weak_signals_record_without_plain_summary_or_category_warns_but_succeeds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # Both fields are optional -- the existing (pre-2026-08-26) registry rows
    # carry neither -- but a NEW record that skips them should say so loudly
    # rather than silently choosing the Signal Ledger page's fallback state.
    monkeypatch.setenv("NFL_ATS_REGISTRY_DIR", str(tmp_path / "registry"))
    assert cli.main(_record_weak_signal_args("cli_no_summary_demo")) == 0
    err = capsys.readouterr().err
    assert "no --plain-summary" in err
    assert "no --category" in err

    registry_path = tmp_path / "registry" / "weak_signals.json"
    stored = json.loads(registry_path.read_text(encoding="utf-8"))
    entry = stored["signals"]["cli_no_summary_demo"]
    assert entry["plain_summary"] is None
    assert entry["category"] is None


def test_weak_signals_record_rejects_an_unknown_category(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("NFL_ATS_REGISTRY_DIR", str(tmp_path / "registry"))
    with pytest.raises(SystemExit):
        cli.main(_record_weak_signal_args("cli_bad_category_demo", category="vibes"))


def test_prospective_primary_entrants_preserve_played_and_raw_policy_arms() -> None:
    active = pd.DataFrame(
        {
            "game_id": ["2026_01_A_B"],
            "model_pick_side": ["HOME"],
            "pick_side": ["AWAY"],
            "bet_side": ["AWAY"],
            "edge": [0.03],
        }
    )

    entrants = dict(cli._prospective_primary_entrants(active))

    assert list(entrants) == ["active_model", "base_model_no_pick_overlays"]
    assert entrants["active_model"].loc[0, "pick_side"] == "AWAY"
    assert entrants["base_model_no_pick_overlays"].loc[0, "pick_side"] == "HOME"
    assert entrants["base_model_no_pick_overlays"].loc[0, "bet_side"] == "PASS"
    assert np.isnan(entrants["base_model_no_pick_overlays"].loc[0, "edge"])


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
                # margin-predict below is invoked with its promoted default
                # (--probability-method gaussian, MOD-08, 2026-08-19); the
                # matching evaluation this test builds must carry the SAME
                # probability_method or synchronization below correctly
                # returns UNLINKED (nfl_ats.active_model's identity match).
                "--probability-method",
                "gaussian",
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


def test_publish_predictions_regenerates_the_site_by_default() -> None:
    """Default-on since 2026-08-19: a publish that skips the public site is how
    docs/ served picks that disagreed with the published card. ``--no-board``
    is the explicit rehearsal opt-out."""

    parser = cli.build_parser()
    assert parser.parse_args(["publish-predictions"]).with_board is True
    assert parser.parse_args(["publish-predictions", "--with-board"]).with_board is True
    assert parser.parse_args(["publish-predictions", "--no-board"]).with_board is False


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
        artifacts_root: Path,
        *,
        destination: Path,
        readme_path: Path,
        data_root: Path | None = None,
        published_at: datetime | None = None,
        registry_root: Path | None = None,
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

    def fake_record(artifacts_root: Path, **kwargs: object) -> dict:
        calls.append(artifacts_root)
        return {"recorded": 1}

    def fake_overlay_record(artifacts_root: Path, data_root: Path) -> dict:
        calls.append(artifacts_root)
        return {"recorded": 1}

    def fake_nomination_record(artifacts_root: Path, data_root: Path) -> dict:
        calls.append(artifacts_root)
        return {"recorded": 1}

    def fake_tilt_record(artifacts_root: Path, data_root: Path) -> dict:
        calls.append(artifacts_root)
        return {"recorded": 1}

    def fake_division_revenge_record(artifacts_root: Path, data_root: Path) -> dict:
        calls.append(artifacts_root)
        return {"recorded": 1}

    def fake_backup_qb_record(artifacts_root: Path, data_root: Path) -> dict:
        calls.append(artifacts_root)
        return {"recorded": 1}

    def fake_surface_switch_record(artifacts_root: Path, data_root: Path) -> dict:
        calls.append(artifacts_root)
        return {"recorded": 1}

    def fake_spread_gap_zone_record(artifacts_root: Path, data_root: Path) -> dict:
        calls.append(artifacts_root)
        return {"recorded": 1}

    def fake_ecdf_mapping_incumbent_record(artifacts_root: Path, data_root: Path) -> dict:
        calls.append(artifacts_root)
        return {"recorded": 1}

    def fake_era_weighted_record(artifacts_root: Path, data_root: Path) -> dict:
        calls.append(artifacts_root)
        return {"recorded": 1}

    def fake_forecast_cold_visitor_record(
        artifacts_root: Path, data_root: Path, registry_root: Path
    ) -> dict:
        calls.append(artifacts_root)
        return {"recorded": 1}

    monkeypatch.setattr(cli, "publish_active_predictions", fake_publish)
    monkeypatch.setattr(cli, "record_paper_decisions", fake_record)
    monkeypatch.setattr(cli, "record_overlay_challenger_decisions", fake_overlay_record)
    monkeypatch.setattr(cli, "record_nomination_challenger_decisions", fake_nomination_record)
    monkeypatch.setattr(cli, "record_injury_value_tilt_challenger_decisions", fake_tilt_record)
    monkeypatch.setattr(
        cli, "record_division_revenge_tilt_challenger_decisions", fake_division_revenge_record
    )
    monkeypatch.setattr(cli, "record_backup_qb_fade_challenger_decisions", fake_backup_qb_record)
    monkeypatch.setattr(
        cli, "record_surface_switch_tilt_challenger_decisions", fake_surface_switch_record
    )
    monkeypatch.setattr(
        cli, "record_spread_gap_zone_fade_challenger_decisions", fake_spread_gap_zone_record
    )
    monkeypatch.setattr(
        cli,
        "record_ecdf_mapping_incumbent_challenger_decisions",
        fake_ecdf_mapping_incumbent_record,
    )
    monkeypatch.setattr(
        cli, "record_era_weighted_half_life_8_challenger_decisions", fake_era_weighted_record
    )
    monkeypatch.setattr(
        cli,
        "record_forecast_cold_visitor_tilt_challenger_decisions",
        fake_forecast_cold_visitor_record,
    )

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
    assert payload["nomination_v3_challenger_ledger"] == {
        "recorded": 0,
        "skipped": True,
        "reason": "pass --record-decisions to append the v3 Best Pick nomination to "
        "the prospective challenger ledger",
    }
    assert payload["big_spread_nomination_challenger_ledger"] == {
        "recorded": 0,
        "skipped": True,
        "reason": "pass --record-decisions to append the big-spread-screened "
        "Best Pick nomination to the prospective challenger ledger",
    }
    assert payload["injury_value_tilt_challenger_ledger"] == {
        "recorded": 0,
        "skipped": True,
        "reason": "pass --record-decisions to append the injury value-lost tilt's "
        "picks to the prospective challenger ledger",
    }
    assert payload["division_revenge_tilt_challenger_ledger"] == {
        "recorded": 0,
        "skipped": True,
        "reason": "pass --record-decisions to append the division-revenge tilt's "
        "picks to the prospective challenger ledger",
    }
    assert payload["backup_qb_fade_challenger_ledger"] == {
        "recorded": 0,
        "skipped": True,
        "reason": "pass --record-decisions to append the backup-QB fade's picks to "
        "the prospective challenger ledger",
    }
    assert payload["surface_switch_tilt_challenger_ledger"] == {
        "recorded": 0,
        "skipped": True,
        "reason": "pass --record-decisions to append the surface-switch tilt's "
        "picks to the prospective challenger ledger",
    }
    assert payload["spread_gap_zone_fade_challenger_ledger"] == {
        "recorded": 0,
        "skipped": True,
        "reason": "pass --record-decisions to append the spread-gap-zone fade's "
        "picks to the prospective challenger ledger",
    }
    assert payload["four_overlay_incumbent_challenger_ledger"] == {
        "recorded": 0,
        "skipped": True,
        "reason": "pass --record-decisions to append the former coach-to-arrests "
        "incumbent's picks to the prospective challenger ledger",
    }
    assert payload["ecdf_mapping_incumbent_challenger_ledger"] == {
        "recorded": 0,
        "skipped": True,
        "reason": "pass --record-decisions to append the ECDF-mapping-incumbent "
        "overlay's picks to the prospective challenger ledger",
    }
    assert payload["era_weighted_half_life_8_challenger_ledger"] == {
        "recorded": 0,
        "skipped": True,
        "reason": "pass --record-decisions to append the era-weighted (half-life 8) "
        "refit's picks to the prospective challenger ledger",
    }
    assert payload["forecast_cold_visitor_tilt_challenger_ledger"] == {
        "recorded": 0,
        "skipped": True,
        "reason": "pass --record-decisions to append the forecast cold-visitor "
        "tilt's picks to the prospective challenger ledger",
    }

    for result_key in cli.PUBLISH_CHALLENGER_RESULT_KEYS.values():
        assert result_key in payload


def test_publish_challenger_result_map_covers_live_active_registry() -> None:
    """Every active publish-time challenger must have an observable CLI result."""

    registry_path = (
        Path(__file__).resolve().parents[1] / "artifacts" / "prospective" / "challengers.json"
    )
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    expected = {
        entry["challenger_id"]
        for entry in registry["challengers"]
        if entry["status"] == "ACTIVE_PROSPECTIVE"
        and "nfl-ats publish-predictions --record-decisions"
        in entry.get("weekly_recording_command", "")
    }

    assert set(cli.PUBLISH_CHALLENGER_RESULT_KEYS) == expected
    assert len(set(cli.PUBLISH_CHALLENGER_RESULT_KEYS.values())) == len(expected)


def test_publish_new_overlay_recorders_are_opt_in(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The six automatic arms remain inert unless recording is explicit."""

    calls: list[str] = []

    def fake_publish(*_args: object, **_kwargs: object) -> dict[str, object]:
        return {"published": True}

    def should_not_run(*_args: object, **_kwargs: object) -> dict[str, object]:
        calls.append("called")
        return {"recorded": 1}

    monkeypatch.setattr(cli, "publish_active_predictions", fake_publish)
    for name in (
        "record_bye_edge_fade_challenger_decisions",
        "record_tank_zone_fade_tilt_challenger_decisions",
        "record_third_down_reversion_fade_challenger_decisions",
        "record_turnover_luck_rebound_tilt_challenger_decisions",
        "record_special_teams_return_tilt_challenger_decisions",
        "record_pace_mismatch_dog_tilt_challenger_decisions",
    ):
        monkeypatch.setattr(cli, name, should_not_run)

    cli._cmd_publish_predictions(
        SimpleNamespace(
            destination=tmp_path / "card.md",
            readme=tmp_path / "README.md",
            with_board=False,
            site_destination=None,
            board_destination=None,
            record_decisions=False,
        )
    )

    payload = _last_json(capsys.readouterr().out)
    assert calls == []
    for key in (
        "bye_edge_fade_challenger_ledger",
        "tank_zone_fade_tilt_challenger_ledger",
        "third_down_reversion_fade_challenger_ledger",
        "turnover_luck_rebound_tilt_challenger_ledger",
        "special_teams_return_tilt_challenger_ledger",
        "pace_mismatch_dog_tilt_challenger_ledger",
    ):
        record = payload[key]
        assert isinstance(record, dict)
        assert record["recorded"] == 0
        assert record["skipped"] is True
        assert "pass --record-decisions" in str(record["reason"])


def test_refresh_crew_recorder_is_gated_and_fails_open(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Crew tracking is refresh-time only and cannot interrupt a card append."""

    plan = SimpleNamespace(changed_games=(object(),))
    calls: list[bool] = []
    destination = tmp_path / "card.md"

    monkeypatch.setattr(cli, "plan_refresh", lambda *_args, **_kwargs: plan)
    monkeypatch.setattr(cli, "refresh_summary", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(cli, "record_plan", lambda *_args, **_kwargs: {"recorded": 1})
    monkeypatch.setattr(
        cli, "record_injury_signal_refresh_tilt", lambda *_args, **_kwargs: {"recorded": 0}
    )
    monkeypatch.setattr(
        cli, "record_nflcom_refresh_overlay", lambda *_args, **_kwargs: {"recorded": 0}
    )
    monkeypatch.setattr(
        cli, "record_inactives_refresh_overlay", lambda *_args, **_kwargs: {"recorded": 0}
    )

    def fake_crew(*_args: object, **kwargs: object) -> dict[str, object]:
        calls.append(bool(kwargs["record_decisions"]))
        raise DataContractError("crew recorder test failure")

    monkeypatch.setattr(cli, "record_crew_tilt_refresh_overlay", fake_crew)
    monkeypatch.setattr(
        cli,
        "append_refresh_to_card",
        lambda *_args, **_kwargs: destination.write_text("refreshed", encoding="utf-8"),
    )

    cli._cmd_refresh_picks(
        SimpleNamespace(
            season=2026,
            week=1,
            features=None,
            min_train_games=500,
            note="test",
            record_decisions=False,
            publish_card=True,
            destination=destination,
        )
    )

    payload = _last_json(capsys.readouterr().out)
    assert calls == [False]
    assert destination.read_text(encoding="utf-8") == "refreshed"
    assert payload["crew_tilt_refresh_overlay"] == {
        "recorded": 0,
        "error": "crew recorder test failure",
    }


def test_publish_new_overlay_recorder_failures_do_not_unpublish(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A side-ledger failure is reported after, never instead of, publication."""

    destination = tmp_path / "card.md"

    def fake_publish(*_args: object, **_kwargs: object) -> dict[str, object]:
        destination.write_text("published", encoding="utf-8")
        return {"published": True}

    def fake_ok(*_args: object, **_kwargs: object) -> dict[str, object]:
        return {"recorded": 0}

    def fake_failure(*_args: object, **_kwargs: object) -> dict[str, object]:
        raise DataContractError("six-overlay test failure")

    monkeypatch.setattr(cli, "publish_active_predictions", fake_publish)
    for name in (
        "record_paper_decisions",
        "record_overlay_challenger_decisions",
        "record_nomination_challenger_decisions",
        "record_nomination_v3_challenger_decisions",
        "record_big_spread_nomination_challenger_decisions",
        "record_injury_value_tilt_challenger_decisions",
        "record_division_revenge_tilt_challenger_decisions",
        "record_backup_qb_fade_challenger_decisions",
        "record_surface_switch_tilt_challenger_decisions",
        "record_spread_gap_zone_fade_challenger_decisions",
        "record_pbp08_protection_mismatch_tilt_challenger_decisions",
        "record_former_production_incumbent_decisions",
        "record_ecdf_mapping_incumbent_challenger_decisions",
        "record_era_weighted_half_life_8_challenger_decisions",
        "record_forecast_cold_visitor_tilt_challenger_decisions",
        "record_interim_hc_first_game_tilt_challenger_decisions",
        "record_forecast_weather_kn_warm_team_cold_late_tilt_challenger_decisions",
        "record_forecast_weather_kn_precip_high_total_tilt_challenger_decisions",
        "record_movement_rule_composed_challenger_decisions",
        "record_nflcom_refresh_out2_starters_challenger_decisions",
    ):
        monkeypatch.setattr(cli, name, fake_ok)
    monkeypatch.setattr(
        cli, "fetch_shared_kickoff_nearest_forecasts_fail_open", lambda *_args, **_kwargs: None
    )
    for name in (
        "record_bye_edge_fade_challenger_decisions",
        "record_tank_zone_fade_tilt_challenger_decisions",
        "record_third_down_reversion_fade_challenger_decisions",
        "record_turnover_luck_rebound_tilt_challenger_decisions",
        "record_special_teams_return_tilt_challenger_decisions",
        "record_pace_mismatch_dog_tilt_challenger_decisions",
    ):
        monkeypatch.setattr(cli, name, fake_failure)

    cli._cmd_publish_predictions(
        SimpleNamespace(
            destination=destination,
            readme=tmp_path / "README.md",
            with_board=False,
            site_destination=None,
            board_destination=None,
            record_decisions=True,
        )
    )

    payload = _last_json(capsys.readouterr().out)
    assert destination.read_text(encoding="utf-8") == "published"
    assert payload["published"] is True
    assert payload["bye_edge_fade_challenger_ledger"] == {
        "recorded": 0,
        "error": "six-overlay test failure",
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
        artifacts_root: Path,
        *,
        destination: Path,
        readme_path: Path,
        data_root: Path | None = None,
        published_at: datetime | None = None,
        registry_root: Path | None = None,
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

    def fake_record(artifacts_root: Path, **kwargs: object) -> dict:
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

    big_spread_nomination_calls: list[Path] = []

    def fake_big_spread_nomination_record(artifacts_root: Path, data_root: Path) -> dict:
        big_spread_nomination_calls.append(artifacts_root)
        return {"recorded": 1, "nominated_game_id": "2026_01_CCC_DDD"}

    tilt_calls: list[Path] = []

    def fake_tilt_record(artifacts_root: Path, data_root: Path) -> dict:
        tilt_calls.append(artifacts_root)
        return {"recorded": 1, "flip_count": 1}

    division_revenge_calls: list[Path] = []

    def fake_division_revenge_record(artifacts_root: Path, data_root: Path) -> dict:
        division_revenge_calls.append(artifacts_root)
        return {"recorded": 1, "flip_count": 1}

    backup_qb_calls: list[Path] = []

    def fake_backup_qb_record(artifacts_root: Path, data_root: Path) -> dict:
        backup_qb_calls.append(artifacts_root)
        return {"recorded": 1, "flip_count": 1}

    surface_switch_calls: list[Path] = []

    def fake_surface_switch_record(artifacts_root: Path, data_root: Path) -> dict:
        surface_switch_calls.append(artifacts_root)
        return {"recorded": 1, "flip_count": 1}

    spread_gap_zone_calls: list[Path] = []

    def fake_spread_gap_zone_record(artifacts_root: Path, data_root: Path) -> dict:
        spread_gap_zone_calls.append(artifacts_root)
        return {"recorded": 1, "flip_count": 1}

    player_arrests_calls: list[Path] = []

    def fake_player_arrests_record(artifacts_root: Path, data_root: Path, **kwargs: object) -> dict:
        player_arrests_calls.append(artifacts_root)
        return {"recorded": 1, "flip_count": 1, "arrest_snapshot_id": "fresh"}

    ecdf_mapping_incumbent_calls: list[Path] = []

    def fake_ecdf_mapping_incumbent_record(artifacts_root: Path, data_root: Path) -> dict:
        ecdf_mapping_incumbent_calls.append(artifacts_root)
        return {"recorded": 1, "flip_count": 1}

    era_weighted_calls: list[Path] = []

    def fake_era_weighted_record(artifacts_root: Path, data_root: Path) -> dict:
        era_weighted_calls.append(artifacts_root)
        return {"recorded": 1, "flip_count": 1}

    forecast_cold_visitor_calls: list[Path] = []

    def fake_forecast_cold_visitor_record(
        artifacts_root: Path, data_root: Path, registry_root: Path
    ) -> dict:
        forecast_cold_visitor_calls.append(artifacts_root)
        return {"recorded": 1, "flip_count": 1}

    monkeypatch.setattr(cli, "publish_active_predictions", fake_publish)
    monkeypatch.setattr(cli, "record_paper_decisions", fake_record)
    monkeypatch.setattr(cli, "record_overlay_challenger_decisions", fake_overlay_record)
    monkeypatch.setattr(cli, "record_nomination_challenger_decisions", fake_nomination_record)
    monkeypatch.setattr(
        cli,
        "record_big_spread_nomination_challenger_decisions",
        fake_big_spread_nomination_record,
    )
    monkeypatch.setattr(cli, "record_injury_value_tilt_challenger_decisions", fake_tilt_record)
    monkeypatch.setattr(
        cli, "record_division_revenge_tilt_challenger_decisions", fake_division_revenge_record
    )
    monkeypatch.setattr(cli, "record_backup_qb_fade_challenger_decisions", fake_backup_qb_record)
    monkeypatch.setattr(
        cli, "record_surface_switch_tilt_challenger_decisions", fake_surface_switch_record
    )
    monkeypatch.setattr(
        cli, "record_spread_gap_zone_fade_challenger_decisions", fake_spread_gap_zone_record
    )
    monkeypatch.setattr(
        cli,
        "record_former_production_incumbent_decisions",
        fake_player_arrests_record,
    )
    monkeypatch.setattr(
        cli,
        "record_ecdf_mapping_incumbent_challenger_decisions",
        fake_ecdf_mapping_incumbent_record,
    )
    monkeypatch.setattr(
        cli, "record_era_weighted_half_life_8_challenger_decisions", fake_era_weighted_record
    )
    monkeypatch.setattr(
        cli,
        "record_forecast_cold_visitor_tilt_challenger_decisions",
        fake_forecast_cold_visitor_record,
    )

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
    assert len(big_spread_nomination_calls) == 1
    assert payload["big_spread_nomination_challenger_ledger"] == {
        "recorded": 1,
        "nominated_game_id": "2026_01_CCC_DDD",
    }
    assert len(tilt_calls) == 1
    assert payload["injury_value_tilt_challenger_ledger"] == {
        "recorded": 1,
        "flip_count": 1,
    }
    assert len(division_revenge_calls) == 1
    assert payload["division_revenge_tilt_challenger_ledger"] == {
        "recorded": 1,
        "flip_count": 1,
    }
    assert len(backup_qb_calls) == 1
    assert payload["backup_qb_fade_challenger_ledger"] == {
        "recorded": 1,
        "flip_count": 1,
    }
    assert len(surface_switch_calls) == 1
    assert payload["surface_switch_tilt_challenger_ledger"] == {
        "recorded": 1,
        "flip_count": 1,
    }
    assert len(spread_gap_zone_calls) == 1
    assert payload["spread_gap_zone_fade_challenger_ledger"] == {
        "recorded": 1,
        "flip_count": 1,
    }
    assert len(player_arrests_calls) == 1
    assert payload["four_overlay_incumbent_challenger_ledger"] == {
        "recorded": 1,
        "flip_count": 1,
        "arrest_snapshot_id": "fresh",
    }
    assert len(ecdf_mapping_incumbent_calls) == 1
    assert payload["ecdf_mapping_incumbent_challenger_ledger"] == {
        "recorded": 1,
        "flip_count": 1,
    }
    assert len(era_weighted_calls) == 1
    assert payload["era_weighted_half_life_8_challenger_ledger"] == {
        "recorded": 1,
        "flip_count": 1,
    }
    assert len(forecast_cold_visitor_calls) == 1
    assert payload["forecast_cold_visitor_tilt_challenger_ledger"] == {
        "recorded": 1,
        "flip_count": 1,
    }


def test_publish_predictions_records_cleanly_when_a_challenger_is_deactivated(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A deactivated challenger (e.g. backup_qb_fade_overlay, marked
    DEACTIVATED_STRUCTURAL_NO_OP 2026-08-19 -- docs/prospective_evidence.md
    'Tuesday-visibility audit') must record nothing and must NOT abort the
    publish. The recorder itself raises ValueError on any non-ACTIVE_PROSPECTIVE
    status (nfl_ats.prospective_scoring's shared status check, exercised
    directly in tests/test_backup_qb_fade_overlay.py); this test pins the
    publish-path contract that catches it: the command still exits 0, every
    OTHER ledger still records normally, and the deactivated challenger's own
    ledger entry reports recorded=0 with the error preserved for visibility."""

    monkeypatch.setenv("NFL_ATS_ARTIFACTS_DIR", str(tmp_path / "artifacts"))
    monkeypatch.setenv("NFL_ATS_REGISTRY_DIR", str(tmp_path / "registry"))
    destination = tmp_path / "CURRENT_PREDICTIONS.md"
    readme = tmp_path / "README.md"
    readme.write_text("x", encoding="utf-8")

    def fake_publish(
        artifacts_root: Path,
        *,
        destination: Path,
        readme_path: Path,
        data_root: Path | None = None,
        published_at: datetime | None = None,
        registry_root: Path | None = None,
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

    def fake_ok(*args: object, **kwargs: object) -> dict:
        return {"recorded": 1}

    def fake_deactivated_backup_qb(artifacts_root: Path, data_root: Path) -> dict:
        # Mirrors exactly what record_backup_qb_fade_challenger_decisions
        # itself raises for a non-ACTIVE_PROSPECTIVE status
        # (nfl_ats.prospective_scoring.ACTIVE_CHALLENGER_STATUS check).
        raise ValueError(
            "Challenger 'backup_qb_fade_overlay' is registered as "
            "'DEACTIVATED_STRUCTURAL_NO_OP'; only ACTIVE_PROSPECTIVE challengers "
            "have picks recorded"
        )

    monkeypatch.setattr(cli, "publish_active_predictions", fake_publish)
    monkeypatch.setattr(cli, "record_paper_decisions", fake_ok)
    monkeypatch.setattr(cli, "record_overlay_challenger_decisions", fake_ok)
    monkeypatch.setattr(cli, "record_nomination_challenger_decisions", fake_ok)
    monkeypatch.setattr(cli, "record_nomination_v3_challenger_decisions", fake_ok)
    monkeypatch.setattr(cli, "record_big_spread_nomination_challenger_decisions", fake_ok)
    monkeypatch.setattr(cli, "record_injury_value_tilt_challenger_decisions", fake_ok)
    monkeypatch.setattr(cli, "record_division_revenge_tilt_challenger_decisions", fake_ok)
    monkeypatch.setattr(
        cli, "record_backup_qb_fade_challenger_decisions", fake_deactivated_backup_qb
    )
    monkeypatch.setattr(cli, "record_surface_switch_tilt_challenger_decisions", fake_ok)
    monkeypatch.setattr(cli, "record_spread_gap_zone_fade_challenger_decisions", fake_ok)
    monkeypatch.setattr(cli, "record_former_production_incumbent_decisions", fake_ok)
    monkeypatch.setattr(cli, "record_ecdf_mapping_incumbent_challenger_decisions", fake_ok)
    monkeypatch.setattr(cli, "record_era_weighted_half_life_8_challenger_decisions", fake_ok)
    monkeypatch.setattr(cli, "record_forecast_cold_visitor_tilt_challenger_decisions", fake_ok)

    exit_code = cli.main(
        [
            "publish-predictions",
            "--destination",
            str(destination),
            "--readme",
            str(readme),
            "--record-decisions",
        ]
    )
    # The whole command must still succeed: a deactivated challenger's
    # refusal must never un-publish or fail the run.
    assert exit_code == 0
    payload = _last_json(capsys.readouterr().out)

    assert payload["backup_qb_fade_challenger_ledger"]["recorded"] == 0
    assert "DEACTIVATED_STRUCTURAL_NO_OP" in payload["backup_qb_fade_challenger_ledger"]["error"]
    # Every OTHER ledger still recorded normally -- one bad challenger must
    # not take down the rest of the publish.
    assert payload["clv_ledger"] == {"recorded": 1}
    assert payload["overlay_challenger_ledger"] == {"recorded": 1}
    assert payload["ecdf_mapping_incumbent_challenger_ledger"] == {"recorded": 1}
    assert payload["era_weighted_half_life_8_challenger_ledger"] == {"recorded": 1}
    assert payload["forecast_cold_visitor_tilt_challenger_ledger"] == {"recorded": 1}


def test_publish_predictions_surfaces_stale_arrest_snapshot_refusal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A stale production source refuses publication before either file changes."""

    monkeypatch.setenv("NFL_ATS_ARTIFACTS_DIR", str(tmp_path / "artifacts"))
    monkeypatch.setenv("NFL_ATS_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("NFL_ATS_REGISTRY_DIR", str(tmp_path / "registry"))
    destination = tmp_path / "CURRENT_PREDICTIONS.md"
    readme = tmp_path / "README.md"
    readme.write_text("x", encoding="utf-8")

    def fake_publish(
        artifacts_root: Path,
        *,
        destination: Path,
        readme_path: Path,
        data_root: Path | None = None,
        published_at: datetime | None = None,
        registry_root: Path | None = None,
    ) -> dict:
        raise cli.DataContractError("player-arrests snapshot is stale at 40.00 hours old")

    monkeypatch.setattr(cli, "publish_active_predictions", fake_publish)
    with pytest.raises(SystemExit) as exit_info:
        cli.main(
            [
                "publish-predictions",
                "--destination",
                str(destination),
                "--readme",
                str(readme),
                "--no-board",
            ]
        )

    assert exit_info.value.code == 2
    assert "snapshot is stale" in capsys.readouterr().err
    assert not destination.exists()
    assert readme.read_text(encoding="utf-8") == "x"
    assert not (tmp_path / "artifacts" / "prospective" / "challenger_decisions.parquet").exists()


def test_cli_handoff(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        cli,
        "write_session_handoff",
        lambda repo_root, artifacts_root, destination, registry_root=None: {
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
        lambda repo_root, artifacts_root, handoff_path, registry_root=None: {
            "handoff": str(handoff_path),
            "status": "CURRENT",
        },
    )
    assert cli.main(["handoff", "--destination", "SESSION.md", "--check"]) == 0
    assert _last_json(capsys.readouterr().out) == {
        "handoff": "SESSION.md",
        "status": "CURRENT",
    }


def test_cli_refresh_picks_end_to_end(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    model_frame: pd.DataFrame,
) -> None:
    """Smoke test for the `refresh-picks` command's CLI wiring (POL-11,
    docs/late_week_refresh.md): opt-in recording, per-game kickoff/Sunday-lock
    guard, and the additive card append all reachable through `cli.main`.
    Deep unit coverage of the recompute itself lives in
    tests/test_pick_refresh.py; this only proves the command is wired up."""

    data_root = tmp_path / "data"
    artifacts_root = tmp_path / "artifacts"
    monkeypatch.setenv("NFL_ATS_DATA_DIR", str(data_root))
    monkeypatch.setenv("NFL_ATS_ARTIFACTS_DIR", str(artifacts_root))
    monkeypatch.setenv("NFL_ATS_REGISTRY_DIR", str(tmp_path / "registry"))

    atomic_json(
        {
            "version": ACTIVE_ATS_MODEL_VERSION,
            "status": "SYNCHRONIZED",
            "method": "market_residual",
            "feature_profile": "base",
            "regressor": "ridge",
            "ridge_alpha": 10.0,
            "probability_method": "ecdf",
            "model_id": "model-1",
        },
        artifacts_root / "active_ats_model.json",
    )

    game_id = "2026_02_III_JJJ"
    # `refresh-picks` has no `--now` override (matching publish-predictions),
    # so it reads the real clock; a kickoff a couple of days out from the
    # real "now" stays inside RECORDING_LOCK_WINDOW and ahead of its own
    # per-game deadline regardless of which real day the suite runs on.
    kickoff = pd.Timestamp(datetime.now(UTC)) + pd.Timedelta(days=2)
    feature_columns = [
        c
        for c in model_frame.columns
        if c
        not in {
            "game_id",
            "season",
            "week",
            "gameday",
            "away_team",
            "home_team",
            "home_spread_odds",
            "away_spread_odds",
            "spread_line",
            "home_cover",
            "ats_margin",
            "result",
        }
    ]
    target_row = {column: model_frame.iloc[0][column] for column in feature_columns}
    target_row.update(
        {
            "game_id": game_id,
            "season": 2026,
            "week": 2,
            "gameday": pd.Timestamp(kickoff.date()),
            "away_team": "III",
            "home_team": "JJJ",
            "home_spread_odds": -110.0,
            "away_spread_odds": -110.0,
            "spread_line": 9.5,  # CURRENT line -- must never be what refresh scores at
            "home_cover": np.nan,
            "ats_margin": np.nan,
            "result": np.nan,
            "kickoff": kickoff,
        }
    )
    features = pd.concat([model_frame, pd.DataFrame([target_row])], ignore_index=True, sort=False)
    features_path = data_root / "processed" / "game_features.parquet"
    atomic_parquet(features, features_path)

    # Independently reproduce the frozen-line prediction so the fixture can
    # guarantee a real change: pick_side below is set to the OPPOSITE side.
    target, margin_models = fit_margin_models_for_week(
        features,
        season=2026,
        week=2,
        regressor="ridge",
        min_train_games=50,
        feature_profile="base",
        ridge_alpha=10.0,
        methods=("market_residual",),
    )
    frozen_line = pd.DataFrame({"game_id": [game_id], "home_spread": [1.0]})
    overridden = apply_external_lines(target, frozen_line)
    forecast = margin_models["market_residual"].predict(overridden, probability_method="ecdf")
    true_side = "HOME" if forecast["home_cover_probability"].iloc[0] >= 0.5 else "AWAY"
    original_pick_side = "AWAY" if true_side == "HOME" else "HOME"

    original = pd.DataFrame(
        [
            {
                "recorded_at_utc": pd.Timestamp("2026-09-15T14:00:00+00:00"),
                "forecast_artifact": "margin_predictions/test",
                "forecast_created_at_utc": pd.Timestamp("2026-09-15T13:00:00+00:00"),
                "model_id": "model-1",
                "method": "market_residual",
                "decision_policy_id": (
                    "overlay_union_coach_division_revenge_player_arrests_spread_gap_v1"
                ),
                "decision_policy_fingerprint": "test-policy-fingerprint",
                "game_id": game_id,
                "season": 2026,
                "week": 2,
                "kickoff": kickoff,
                "away_team": "III",
                "home_team": "JJJ",
                "model_pick_side": original_pick_side,
                "pre_arrest_pick_side": original_pick_side,
                "former_policy_pick_side": original_pick_side,
                "pick_side": original_pick_side,
                "coach_fade_flip": False,
                "division_revenge_flip": False,
                "player_arrests_flip": False,
                "spread_gap_zone_flip": False,
                "composed_overlay_flip": False,
                "player_arrests_home_flag": False,
                "player_arrests_away_flag": False,
                "player_arrests_snapshot_id": "snapshot-tuesday",
                "player_arrests_snapshot_fetched_at_utc": pd.Timestamp("2026-09-15T12:00:00+00:00"),
                "player_arrests_safe_index_sha256": "safe-index-hash",
                "schedule_snapshot_id": "schedule-tuesday",
                "schedule_parquet_sha256": "schedule-hash",
                "bet_side": original_pick_side,
                "decision_home_spread": 1.0,  # the FROZEN Tuesday line, different from 9.5 above
                "edge": 0.05,
                "is_best_pick": False,
            }
        ]
    )
    atomic_parquet(
        original[list(PAPER_DECISION_COLUMNS)], paper_decision_ledger_path(artifacts_root)
    )

    destination = tmp_path / "CURRENT_PREDICTIONS.md"
    destination.write_text(
        "# NFL ATS predictions: 2026 Week 2\n\nTuesday content.\n", encoding="utf-8"
    )

    # A rehearsal-style dry pass (no --record-decisions): computes but writes nothing.
    exit_code = cli.main(
        [
            "refresh-picks",
            "--season",
            "2026",
            "--week",
            "2",
            "--features",
            str(features_path),
            "--min-train-games",
            "50",
        ]
    )
    assert exit_code == 0
    dry_payload = _last_json(capsys.readouterr().out)
    assert dry_payload["season"] == 2026
    assert dry_payload["week"] == 2
    assert dry_payload["changed_game_ids"] == [game_id]
    assert dry_payload["ledger"]["skipped"] is True
    assert load_pick_revisions(artifacts_root).empty
    # No data/market/raw store exists in this fixture, so the observed-
    # movement policy (POL-11 addendum, docs/late_week_refresh.md) fails
    # open: the model-only pick governs, surfaced in the JSON payload.
    assert dry_payload["movement_policy"]["current_line_fresh"] is False
    assert dry_payload["movement_policy"]["current_line_reason"] == "no_market_snapshots"
    assert dry_payload["movement_policy"]["games_model_only"] == [game_id]
    # T-90 inactives is wired as a separate challenger result. With no capture
    # store it fails closed/observationally; the played refresh still runs.
    assert dry_payload["inactives_refresh_overlay"]["challenger_id"] == "inactives_refresh_v1"
    assert dry_payload["inactives_refresh_overlay"]["recorded"] == 0

    # The real, opt-in recording pass, with the card append.
    exit_code = cli.main(
        [
            "refresh-picks",
            "--season",
            "2026",
            "--week",
            "2",
            "--features",
            str(features_path),
            "--min-train-games",
            "50",
            "--record-decisions",
            "--publish-card",
            "--destination",
            str(destination),
            "--note",
            "thursday_afternoon",
        ]
    )
    assert exit_code == 0
    payload = _last_json(capsys.readouterr().out)
    assert payload["season"] == 2026
    assert payload["week"] == 2
    assert payload["record_decisions"] is True
    assert payload["changed_game_ids"] == [game_id]
    assert payload["ledger"] == {"recorded": 1, "ledger_rows": 1}
    assert payload["card"] == {"written": True, "destination": str(destination)}
    assert payload["movement_policy"]["current_line_fresh"] is False
    assert payload["movement_policy"]["games_model_only"] == [game_id]

    revisions = load_pick_revisions(artifacts_root)
    assert len(revisions) == 1
    assert revisions.iloc[0]["game_id"] == game_id
    assert revisions.iloc[0]["decision_home_spread"] == pytest.approx(1.0)
    assert revisions.iloc[0]["previous_pick_side"] == original_pick_side
    assert revisions.iloc[0]["new_pick_side"] == true_side
    # Both arms recorded on the ledger row: no fresh captured line this pass,
    # so the model-only arm governed and equals the played pick.
    assert revisions.iloc[0]["movement_policy"] == "model_only"
    assert revisions.iloc[0]["model_only_pick_side"] == true_side
    assert pd.isna(revisions.iloc[0]["movement_delta"])

    card_text = destination.read_text(encoding="utf-8")
    assert "Tuesday content." in card_text
    assert "Late-week refresh" in card_text
    assert "thursday_afternoon" in card_text
    assert "Policy" in card_text
    assert "model_only" in card_text
