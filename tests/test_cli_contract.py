"""Contract tests for the ``nfl-ats`` command-line surface (ENG-10).

Three guarantees:

1. **The whole parser is pinned.** ``tests/fixtures/cli_contract.json`` records
   every command path, its help text, every argument's option strings, dest,
   default, choices, type, nargs, required flag and help, the subcommand
   ordering, and each handler's module and qualname. Any flag/default/help
   drift -- or a command silently moving to a different module -- fails here.
2. **Each public workflow has a testable seam.** ``parse_*_request`` is pure
   and returns a frozen Request; ``orchestrate_*`` takes that Request and does
   the work; the ``_cmd_*`` handler is the two calls plus the writer.
3. **Importing the CLI does not get heavier.** The set of heavy third-party
   packages pulled in by ``import nfl_ats.cli`` is pinned to what it already
   was before the ENG-10 split (which was, and still is, "all of them" -- the
   registrars are eager). Adding a new heavy dependency to the import path
   fails the test; nothing here claims the import is lazy.
"""

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

sys.path.insert(0, str(REPO_ROOT / "scripts"))
from cli_contract_snapshot import normalize_years, snapshot  # noqa: E402

# --------------------------------------------------------------------------
# 1. the pinned parser contract
# --------------------------------------------------------------------------


def _current_contract() -> dict[str, Any]:
    return dict(normalize_years(snapshot(cli.build_parser())))


def test_cli_contract_matches_the_tracked_fixture() -> None:
    """Every flag, default, help string and handler location is unchanged.

    Regenerate deliberately with::

        uv run python scripts/cli_contract_snapshot.py \\
            tests/fixtures/cli_contract.json --normalize-years
    """

    expected = json.loads(FIXTURE.read_text(encoding="utf-8"))
    assert _current_contract() == expected


def test_registration_order_is_the_help_listing_order() -> None:
    """``REGISTRARS`` order == top-level ``--help`` order, both pinned."""

    expected = json.loads(FIXTURE.read_text(encoding="utf-8"))
    order = expected["root"]["subcommands"]["order"]
    parser = cli.build_parser()
    action = next(a for a in parser._actions if isinstance(a, argparse._SubParsersAction))
    assert list(action.choices) == order
    assert len(REGISTRARS) == 18


def test_every_command_has_a_handler() -> None:
    """No subcommand may be reachable without a ``handler`` default."""

    contract = _current_contract()

    def walk(node: dict[str, Any]) -> None:
        subs = node.get("subcommands")
        if subs is None:
            assert node["handler"] is not None, node["path"]
            return
        for child in subs["parsers"]:
            walk(child)

    walk(contract["root"])


def test_handlers_live_in_the_command_packages() -> None:
    """Handlers belong to ``nfl_ats.cli_commands.*``; ``cli`` keeps none."""

    contract = _current_contract()
    modules: set[str] = set()

    def walk(node: dict[str, Any]) -> None:
        subs = node.get("subcommands")
        if subs is None:
            modules.add(str(node["handler"]).rsplit(".", 1)[0])
            return
        for child in subs["parsers"]:
            walk(child)

    walk(contract["root"])
    assert modules
    assert all(m.startswith("nfl_ats.cli_commands.") for m in modules), sorted(modules)


# --------------------------------------------------------------------------
# 2. the four public workflows: parse -> orchestrate -> write
# --------------------------------------------------------------------------


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
    """The Request layer raises exactly what reading the namespace raised."""

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
    # --board-destination keeps its historical default so the `or` in the
    # orchestrator can never be handed two Nones on the real CLI path.
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
    """``main`` keeps converting FileNotFoundError/ValueError into exit 2."""

    def boom(request: object) -> dict[str, Any]:
        raise ValueError("no feature table")

    monkeypatch.setattr(operations_cmds, "orchestrate_weekly_run", boom)
    with pytest.raises(SystemExit) as excinfo:
        cli.main(["weekly-run", "--season", "2026", "--week", "1"])
    assert excinfo.value.code == 2
    assert capsys.readouterr().err == "error: no feature table\n"


# --------------------------------------------------------------------------
# 3. import weight
# --------------------------------------------------------------------------

#: Measured 2026-09-04 on the post-ENG-10 tree, and identical to the pre-split
#: module: ``nfl_ats.cli`` eagerly imports every command module, so the whole
#: scientific stack loads. This is a ceiling, not an endorsement -- the test
#: exists so a NEW heavy dependency cannot slip onto the CLI import path.
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
