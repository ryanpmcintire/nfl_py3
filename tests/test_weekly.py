from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest

from nfl_ats import cli, weekly
from nfl_ats.io import atomic_json
from nfl_ats.weekly import WeeklyRunError, plan_weekly_run, run_weekly

PRODUCTION_PBP_SNAPSHOT = "20260812T142851Z"
PRODUCTION_PLAYER_SNAPSHOT = "20260812T200527Z"
PRODUCTION_PLAYER_VALUE_SNAPSHOT = "20260813T121050Z"


def _last_json(output: str) -> dict[str, Any]:
    return json.loads(output)


def _write_data_root(tmp_path: Path) -> Path:
    """A data root with a raw snapshot manifest and the two production manifests."""

    data_root = tmp_path / "data"
    raw = data_root / "raw" / "20260812T130036Z"
    atomic_json(
        {
            "snapshot_id": "20260812T130036Z",
            "seasons": list(range(2009, 2027)),
            "team_stat_seasons": list(range(2009, 2026)),
        },
        raw / "manifest.json",
    )
    processed = data_root / "processed"
    atomic_json(
        {"source_pbp_snapshot": PRODUCTION_PBP_SNAPSHOT},
        processed / "game_features_pbp.manifest.json",
    )
    atomic_json(
        {
            "source_pbp_snapshot": PRODUCTION_PBP_SNAPSHOT,
            "source_player_snapshot": PRODUCTION_PLAYER_SNAPSHOT,
            "source_player_value_snapshot": PRODUCTION_PLAYER_VALUE_SNAPSHOT,
        },
        processed / "game_features_player.manifest.json",
    )
    return data_root


def _write_active_model(artifacts_root: Path, *, season: int, week: int, status: str) -> None:
    atomic_json(
        {
            "version": 1,
            "status": status,
            "model_id": "80e458040e48b926",
            "historical_evaluation": {"accuracy": 0.5204819277, "games": 415},
            "weekly_forecast": {"season": season, "week": week},
        },
        artifacts_root / "active_ats_model.json",
    )


class _Recorder:
    """Step runner that records argv and returns canned per-command payloads."""

    def __init__(self, **outputs: dict[str, Any]) -> None:
        self.commands: list[list[str]] = []
        self.outputs = outputs

    def __call__(self, command: Sequence[str]) -> dict[str, Any]:
        self.commands.append(list(command))
        return self.outputs.get(command[0], {})

    @property
    def names(self) -> list[str]:
        return [command[0] for command in self.commands]


def test_plan_is_the_seven_specified_steps_in_order(tmp_path: Path) -> None:
    data_root = _write_data_root(tmp_path)
    steps = plan_weekly_run(season=2026, week=1, data_root=data_root)

    assert [step.name for step in steps] == [
        "ingest",
        "build-features",
        "build-pbp-features",
        "build-player-features",
        "margin-backtest",
        "margin-predict",
        "assert-synchronized",
        "publish-predictions",
    ]
    assert [step.number for step in steps] == [1, 2, 3, 3, 4, 5, 6, 7]
    # The synchronization assertion sits strictly between scoring and publish.
    names = [step.name for step in steps]
    assert names.index("assert-synchronized") > names.index("margin-predict")
    assert names.index("assert-synchronized") < names.index("publish-predictions")


def test_plan_pins_production_snapshot_ids_and_the_manifest_season_span(tmp_path: Path) -> None:
    data_root = _write_data_root(tmp_path)
    steps = {step.name: step for step in plan_weekly_run(season=2026, week=1, data_root=data_root)}

    assert steps["ingest"].command == (
        "ingest",
        "--start-season",
        "2009",
        "--end-season",
        "2026",
        "--stats-end-season",
        "2025",
    )
    assert steps["build-features"].command == ("build-features",)
    assert steps["build-pbp-features"].command == (
        "build-pbp-features",
        "--snapshot",
        PRODUCTION_PBP_SNAPSHOT,
    )
    assert steps["build-player-features"].command == (
        "build-player-features",
        "--player-snapshot",
        PRODUCTION_PLAYER_SNAPSHOT,
        "--player-value-snapshot",
        PRODUCTION_PLAYER_VALUE_SNAPSHOT,
        "--pbp-snapshot",
        PRODUCTION_PBP_SNAPSHOT,
    )
    player_table = str(data_root / "processed" / "game_features_player.parquet")
    assert steps["margin-backtest"].command == (
        "margin-backtest",
        "--features",
        player_table,
        "--feature-profile",
        "player",
    )
    assert steps["margin-predict"].command == (
        "margin-predict",
        "--season",
        "2026",
        "--week",
        "1",
        "--features",
        player_table,
        "--feature-profile",
        "player",
    )
    assert steps["assert-synchronized"].command == ()
    assert steps["publish-predictions"].command == ("publish-predictions", "--with-board")


def test_refresh_player_data_drops_the_pinned_snapshot_ids(tmp_path: Path) -> None:
    data_root = _write_data_root(tmp_path)
    steps = {
        step.name: step
        for step in plan_weekly_run(
            season=2026, week=1, data_root=data_root, refresh_player_data=True
        )
    }
    assert steps["build-pbp-features"].command == ("build-pbp-features",)
    assert steps["build-player-features"].command == ("build-player-features",)


def test_plan_aborts_when_the_snapshot_predates_the_requested_season(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    atomic_json(
        {"snapshot_id": "old", "seasons": list(range(2009, 2026))},
        data_root / "raw" / "old" / "manifest.json",
    )
    with pytest.raises(WeeklyRunError, match="excludes the requested season 2026"):
        plan_weekly_run(season=2026, week=1, data_root=data_root)


def test_plan_aborts_when_a_production_manifest_is_missing(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    atomic_json(
        {"snapshot_id": "s", "seasons": [2026], "team_stat_seasons": [2025]},
        data_root / "raw" / "s" / "manifest.json",
    )
    with pytest.raises(WeeklyRunError, match="--refresh-player-data"):
        plan_weekly_run(season=2026, week=1, data_root=data_root)


def test_dry_run_prints_the_plan_and_runs_nothing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    data_root = _write_data_root(tmp_path)
    monkeypatch.setenv("NFL_ATS_DATA_DIR", str(data_root))
    monkeypatch.setenv("NFL_ATS_ARTIFACTS_DIR", str(tmp_path / "artifacts"))

    def explode(command: Sequence[str]) -> dict[str, Any]:
        raise AssertionError(f"dry run executed {command!r}")

    monkeypatch.setattr(weekly, "_cli_runner", explode)

    assert cli.main(["weekly-run", "--season", "2026", "--week", "1", "--dry-run"]) == 0
    payload = _last_json(capsys.readouterr().out)

    assert payload["command"] == "weekly-run"
    assert payload["dry_run"] is True
    assert payload["published"] is False
    assert payload["season"] == 2026 and payload["week"] == 1
    assert [step["name"] for step in payload["steps"]] == [
        "ingest",
        "build-features",
        "build-pbp-features",
        "build-player-features",
        "margin-backtest",
        "margin-predict",
        "assert-synchronized",
        "publish-predictions",
    ]
    # The plan doubles as the manual fallback, so it prints runnable commands.
    assert payload["steps"][0]["command"][:4] == ["python", "-m", "nfl_ats", "ingest"]
    assert payload["steps"][6]["command"] == []
    assert all("status" not in step for step in payload["steps"])


def test_skip_ingest_marks_step_one_skipped(tmp_path: Path) -> None:
    data_root = _write_data_root(tmp_path)
    artifacts_root = tmp_path / "artifacts"
    _write_active_model(artifacts_root, season=2026, week=1, status="SYNCHRONIZED")
    runner = _Recorder(**{"margin-predict": {"synchronization_status": "SYNCHRONIZED"}})

    summary = run_weekly(
        season=2026,
        week=1,
        data_root=data_root,
        artifacts_root=artifacts_root,
        skip_ingest=True,
        runner=runner,
        progress=False,
    )

    assert runner.names == [
        "build-features",
        "build-pbp-features",
        "build-player-features",
        "margin-backtest",
        "margin-predict",
        "publish-predictions",
    ]
    assert summary["steps"][0] == {
        **summary["steps"][0],
        "name": "ingest",
        "status": "skipped",
        "skipped": True,
    }
    assert summary["published"] is True
    assert summary["active_model_id"] == "80e458040e48b926"


def test_run_executes_every_step_in_order(tmp_path: Path) -> None:
    data_root = _write_data_root(tmp_path)
    artifacts_root = tmp_path / "artifacts"
    _write_active_model(artifacts_root, season=2026, week=1, status="SYNCHRONIZED")
    runner = _Recorder(**{"margin-predict": {"synchronization_status": "SYNCHRONIZED"}})

    summary = run_weekly(
        season=2026,
        week=1,
        data_root=data_root,
        artifacts_root=artifacts_root,
        runner=runner,
        progress=False,
    )

    assert runner.names == [
        "ingest",
        "build-features",
        "build-pbp-features",
        "build-player-features",
        "margin-backtest",
        "margin-predict",
        "publish-predictions",
    ]
    assert [step["status"] for step in summary["steps"]] == ["ok"] * 8
    assert summary["historical_evaluation"]["accuracy"] == pytest.approx(0.5204819277)


def test_abort_on_desync_never_publishes(tmp_path: Path) -> None:
    """The manifest still says SYNCHRONIZED, but it points at last week's card."""

    data_root = _write_data_root(tmp_path)
    artifacts_root = tmp_path / "artifacts"
    _write_active_model(artifacts_root, season=2025, week=22, status="SYNCHRONIZED")
    runner = _Recorder()

    with pytest.raises(WeeklyRunError, match="assert-synchronized"):
        run_weekly(
            season=2026,
            week=1,
            data_root=data_root,
            artifacts_root=artifacts_root,
            runner=runner,
            progress=False,
        )

    assert "publish-predictions" not in runner.names
    assert runner.names[-1] == "margin-predict"


def test_abort_when_activation_reported_unlinked(tmp_path: Path) -> None:
    data_root = _write_data_root(tmp_path)
    artifacts_root = tmp_path / "artifacts"
    _write_active_model(artifacts_root, season=2026, week=1, status="SYNCHRONIZED")
    runner = _Recorder(**{"margin-predict": {"synchronization_status": "UNLINKED"}})

    with pytest.raises(WeeklyRunError, match="UNLINKED"):
        run_weekly(
            season=2026,
            week=1,
            data_root=data_root,
            artifacts_root=artifacts_root,
            runner=runner,
            progress=False,
        )
    assert "publish-predictions" not in runner.names


def test_abort_when_the_manifest_status_is_not_synchronized(tmp_path: Path) -> None:
    data_root = _write_data_root(tmp_path)
    artifacts_root = tmp_path / "artifacts"
    _write_active_model(artifacts_root, season=2026, week=1, status="UNLINKED")
    runner = _Recorder()

    with pytest.raises(WeeklyRunError, match="not synchronized"):
        run_weekly(
            season=2026,
            week=1,
            data_root=data_root,
            artifacts_root=artifacts_root,
            runner=runner,
            progress=False,
        )
    assert "publish-predictions" not in runner.names


def test_step_failure_names_the_step_and_stops_the_run(tmp_path: Path) -> None:
    data_root = _write_data_root(tmp_path)
    artifacts_root = tmp_path / "artifacts"
    calls: list[str] = []

    def failing(command: Sequence[str]) -> dict[str, Any]:
        calls.append(command[0])
        if command[0] == "margin-backtest":
            raise RuntimeError("evaluator blew up")
        return {}

    with pytest.raises(WeeklyRunError, match=r"margin-backtest.*evaluator blew up"):
        run_weekly(
            season=2026,
            week=1,
            data_root=data_root,
            artifacts_root=artifacts_root,
            runner=failing,
            progress=False,
        )

    assert calls == [
        "ingest",
        "build-features",
        "build-pbp-features",
        "build-player-features",
        "margin-backtest",
    ]


def test_cli_reports_an_abort_as_a_user_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data_root = _write_data_root(tmp_path)
    monkeypatch.setenv("NFL_ATS_DATA_DIR", str(data_root))
    monkeypatch.setenv("NFL_ATS_ARTIFACTS_DIR", str(tmp_path / "artifacts"))
    monkeypatch.setattr(weekly, "_cli_runner", lambda command: {})

    with pytest.raises(SystemExit) as excinfo:
        cli.main(["weekly-run", "--season", "2026", "--week", "1"])
    assert excinfo.value.code == 2
