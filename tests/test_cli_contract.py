from __future__ import annotations

import argparse
import dataclasses
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from nfl_ats import cli
from nfl_ats.cli_commands import REGISTRARS
from nfl_ats.cli_commands import operations as operations_cmds
from nfl_ats.cli_commands import prediction as prediction_cmds
from nfl_ats.cli_commands import publishing as publishing_cmds

REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURE = REPO_ROOT / "tests" / "fixtures" / "cli_contract.json"


def test_registration_order_is_the_help_listing_order() -> None:

    expected = json.loads(FIXTURE.read_text(encoding="utf-8"))
    order = expected["root"]["subcommands"]["order"]
    parser = cli.build_parser()
    action = next(a for a in parser._actions if isinstance(a, argparse._SubParsersAction))
    assert list(action.choices) == order
    assert len(REGISTRARS) == 18


def _parse(argv: list[str]) -> argparse.Namespace:
    return cli.build_parser().parse_args(argv)


def test_weekly_run_parse_and_validate() -> None:
    args = _parse(["weekly-run", "--season", "2026", "--week", "1"])
    request = operations_cmds.parse_weekly_run_request(args)
    assert request == operations_cmds.WeeklyRunRequest(
        season=2026,
        week=1,
        refresh_player_data=False,
        skip_ingest=False,
        skip_prospective=False,
        skip_drift=False,
        record_decisions=False,
        dry_run=False,
        no_package=False,
    )
    with pytest.raises(dataclasses.FrozenInstanceError):
        request.season = 2027  # type: ignore[misc]


def test_weekly_run_parse_and_validate_rejects_an_incomplete_namespace() -> None:

    with pytest.raises(AttributeError):
        operations_cmds.parse_weekly_run_request(SimpleNamespace(season=2026))


def test_weekly_run_missing_required_flags_uses_the_argparse_message(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as excinfo:
        cli.main(["weekly-run"])
    assert excinfo.value.code == 2
    assert "the following arguments are required: --season, --week" in capsys.readouterr().err


def test_weekly_run_handler_passes_the_parsed_request_to_orchestrate(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    seen: list[operations_cmds.WeeklyRunRequest] = []

    def fake(request: operations_cmds.WeeklyRunRequest) -> dict[str, Any]:
        seen.append(request)
        return {"ok": True}

    monkeypatch.setattr(operations_cmds, "orchestrate_weekly_run", fake)
    assert cli.main(["weekly-run", "--season", "2026", "--week", "2", "--dry-run"]) == 0
    assert len(seen) == 1
    assert seen[0].season == 2026
    assert seen[0].week == 2
    assert seen[0].dry_run is True
    assert seen[0].record_decisions is False
    assert json.loads(capsys.readouterr().out) == {"ok": True}


def test_publish_predictions_parse_and_validate(tmp_path: Path) -> None:
    args = _parse(
        [
            "publish-predictions",
            "--destination",
            str(tmp_path / "card.md"),
            "--readme",
            str(tmp_path / "README.md"),
            "--no-board",
        ]
    )
    request = publishing_cmds.parse_publish_predictions_request(args)
    assert request.destination == tmp_path / "card.md"
    assert request.readme == tmp_path / "README.md"
    assert request.with_board is False
    assert request.record_decisions is False
    assert request.board_destination == Path("docs/index.html")
    assert request.site_destination is None


def test_publish_predictions_board_is_on_by_default() -> None:
    request = publishing_cmds.parse_publish_predictions_request(_parse(["publish-predictions"]))
    assert request.with_board is True


def test_publish_predictions_rejects_an_unknown_flag(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as excinfo:
        cli.main(["publish-predictions", "--not-a-flag"])
    assert excinfo.value.code == 2
    assert "unrecognized arguments: --not-a-flag" in capsys.readouterr().err


def test_publish_predictions_handler_passes_the_parsed_request(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    seen: list[publishing_cmds.PublishPredictionsRequest] = []

    def fake(request: publishing_cmds.PublishPredictionsRequest) -> dict[str, Any]:
        seen.append(request)
        return {"published": True}

    monkeypatch.setattr(publishing_cmds, "orchestrate_publish_predictions", fake)
    exit_code = cli.main(
        [
            "publish-predictions",
            "--destination",
            str(tmp_path / "card.md"),
            "--no-board",
        ]
    )
    assert exit_code == 0
    assert len(seen) == 1
    assert seen[0].destination == tmp_path / "card.md"
    assert seen[0].with_board is False
    assert json.loads(capsys.readouterr().out) == {"published": True}


def test_margin_predict_parse_and_validate(tmp_path: Path) -> None:
    args = _parse(
        [
            "margin-predict",
            "--season",
            "2026",
            "--week",
            "1",
            "--features",
            str(tmp_path / "features.parquet"),
            "--line-sweep",
        ]
    )
    request = prediction_cmds.parse_margin_predict_request(args)
    assert request.features == tmp_path / "features.parquet"
    assert request.season == 2026
    assert request.week == 1
    assert request.line_sweep is True
    assert request.probability_method == args.probability_method
    assert request.feature_profile == args.feature_profile


def test_margin_predict_rejects_an_unknown_regressor(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as excinfo:
        cli.main(["margin-predict", "--season", "2026", "--week", "1", "--regressor", "nope"])
    assert excinfo.value.code == 2
    assert "invalid choice: 'nope'" in capsys.readouterr().err


def test_margin_predict_handler_passes_the_parsed_request(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    seen: list[prediction_cmds.MarginPredictRequest] = []

    def fake(request: prediction_cmds.MarginPredictRequest) -> prediction_cmds.PredictionArtifacts:
        seen.append(request)
        return prediction_cmds.PredictionArtifacts(metadata={"games": 16}, output=tmp_path / "out")

    monkeypatch.setattr(prediction_cmds, "orchestrate_margin_predict", fake)
    assert cli.main(["margin-predict", "--season", "2026", "--week", "3"]) == 0
    assert [(r.season, r.week) for r in seen] == [(2026, 3)]
    payload = json.loads(capsys.readouterr().out)
    assert payload == {"games": 16, "artifact_directory": str(tmp_path / "out")}


def test_predict_parse_and_validate(tmp_path: Path) -> None:
    args = _parse(
        [
            "predict",
            "--season",
            "2025",
            "--week",
            "7",
            "--features",
            str(tmp_path / "f.parquet"),
            "--freeze",
        ]
    )
    request = prediction_cmds.parse_predict_request(args)
    assert request.season == 2025
    assert request.week == 7
    assert request.freeze is True
    assert request.model == "logistic"
    assert request.feature_set == "market_context"


def test_predict_rejects_an_unknown_model(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as excinfo:
        cli.main(["predict", "--season", "2025", "--week", "7", "--model", "nope"])
    assert excinfo.value.code == 2
    assert "invalid choice: 'nope'" in capsys.readouterr().err


def test_predict_handler_passes_the_parsed_request(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    seen: list[prediction_cmds.PredictRequest] = []

    def fake(request: prediction_cmds.PredictRequest) -> prediction_cmds.PredictionArtifacts:
        seen.append(request)
        return prediction_cmds.PredictionArtifacts(metadata={"n": 1}, output=tmp_path)

    monkeypatch.setattr(prediction_cmds, "orchestrate_predict", fake)
    assert cli.main(["predict", "--season", "2024", "--week", "5"]) == 0
    assert [(r.season, r.week) for r in seen] == [(2024, 5)]
    assert json.loads(capsys.readouterr().out)["artifact_directory"] == str(tmp_path)


def test_library_errors_still_exit_two_with_the_error_prefix(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:

    def boom(request: object) -> dict[str, Any]:
        raise ValueError("no feature table")

    monkeypatch.setattr(operations_cmds, "orchestrate_weekly_run", boom)
    with pytest.raises(SystemExit) as excinfo:
        cli.main(["weekly-run", "--season", "2026", "--week", "1"])
    assert excinfo.value.code == 2
    assert capsys.readouterr().err == "error: no feature table\n"


HEAVY_IMPORT_BASELINE = frozenset({"joblib", "numpy", "pandas", "pyarrow", "scipy", "sklearn"})


def test_importing_the_cli_pulls_in_no_new_heavy_package() -> None:
    program = (
        "import json, sys; import nfl_ats.cli; "
        "print(json.dumps(sorted({m.split('.')[0] for m in sys.modules})))"
    )
    completed = subprocess.run(
        [sys.executable, "-c", program],
        capture_output=True,
        text=True,
        check=True,
        cwd=REPO_ROOT,
    )
    loaded = set(json.loads(completed.stdout))
    candidates = {
        "joblib",
        "matplotlib",
        "numpy",
        "nflreadpy",
        "pandas",
        "plotly",
        "pyarrow",
        "requests",
        "scipy",
        "sklearn",
        "statsmodels",
        "torch",
    }
    assert loaded & candidates <= HEAVY_IMPORT_BASELINE
