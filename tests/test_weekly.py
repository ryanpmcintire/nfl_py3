from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest

from nfl_ats.io import atomic_json
from nfl_ats.weekly import (
    WeeklyRunError,
    plan_weekly_run,
    run_weekly,
)

PRODUCTION_PBP_SNAPSHOT = "20260812T142851Z"
PRODUCTION_PLAYER_SNAPSHOT = "20260812T200527Z"
PRODUCTION_PLAYER_VALUE_SNAPSHOT = "20260813T121050Z"

PROSPECTIVE_STEPS = [
    "build-weak-stack-features",
    "margin-predict-challenger",
    "prospective-record",
    "prospective-score",
]


def _last_json(output: str) -> dict[str, Any]:
    return json.loads(output)


def _write_data_root(tmp_path: Path) -> Path:

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
    (raw / "schedules.parquet").write_bytes(b"")
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
    atomic_json(
        {
            "source_pbp_snapshot": PRODUCTION_PBP_SNAPSHOT,
            "source_player_snapshot": PRODUCTION_PLAYER_SNAPSHOT,
            "source_player_value_snapshot": PRODUCTION_PLAYER_VALUE_SNAPSHOT,
        },
        processed / "game_features_weak_stack.manifest.json",
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
    steps = plan_weekly_run(season=2026, week=1, data_root=data_root, skip_prospective=True)

    assert [step.name for step in steps] == [
        "ingest",
        "build-features",
        "build-pbp-features",
        "build-player-features",
        "margin-backtest",
        "margin-predict",
        "assert-synchronized",
        "ingest-player-arrests",
        "opener-evaluation",
        "fit-pick-probability",
        "overlay-composition",
        "served-card-archive",
        "served-refresh-card",
        "waterfall-feed",
        "publish-predictions",
        "drift-report",
        "publish-board",
    ]
    assert [step.number for step in steps] == [1, 2, 3, 3, 4, 5, 6, 7, 7, 7, 7, 7, 7, 8, 8, 13, 15]
    assert steps[-2].optional is True
    assert steps[-1].name == "publish-board"
    assert steps[-1].optional is False
    names = [step.name for step in steps]
    assert names.index("assert-synchronized") > names.index("margin-predict")
    assert names.index("assert-synchronized") < names.index("publish-predictions")


def test_plan_aborts_when_the_snapshot_predates_the_requested_season(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    atomic_json(
        {"snapshot_id": "old", "seasons": list(range(2009, 2026))},
        data_root / "raw" / "old" / "manifest.json",
    )
    (data_root / "raw" / "old" / "schedules.parquet").write_bytes(b"")
    with pytest.raises(WeeklyRunError, match="excludes the requested season 2026"):
        plan_weekly_run(season=2026, week=1, data_root=data_root, skip_prospective=True)


def test_plan_aborts_when_a_production_manifest_is_missing(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    atomic_json(
        {"snapshot_id": "s", "seasons": [2026], "team_stat_seasons": [2025]},
        data_root / "raw" / "s" / "manifest.json",
    )
    (data_root / "raw" / "s" / "schedules.parquet").write_bytes(b"")
    with pytest.raises(WeeklyRunError, match="--refresh-player-data"):
        plan_weekly_run(season=2026, week=1, data_root=data_root, skip_prospective=True)


def test_abort_on_desync_never_publishes(tmp_path: Path) -> None:

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
            skip_prospective=True,
            runner=runner,
            progress=False,
        )

    assert "publish-predictions" not in runner.names
    assert runner.names[-1] == "margin-predict"


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
            skip_prospective=True,
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
            skip_prospective=True,
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


def test_an_optional_step_failure_is_reported_but_never_aborts_the_run(
    tmp_path: Path,
) -> None:

    data_root = _write_data_root(tmp_path)
    artifacts_root = tmp_path / "artifacts"
    _write_active_model(artifacts_root, season=2026, week=1, status="SYNCHRONIZED")
    calls: list[str] = []

    def failing(command: Sequence[str]) -> dict[str, Any]:
        calls.append(command[0])
        if command[0] == "build-learned-availability-features":
            raise RuntimeError("no participation snapshot")
        if command[0] == "margin-predict":
            return {"synchronization_status": "SYNCHRONIZED"}
        return {}

    summary = run_weekly(
        season=2026,
        week=1,
        data_root=data_root,
        artifacts_root=artifacts_root,
        runner=failing,
        progress=False,
    )

    assert summary["published"] is True
    assert summary["optional_failures"] == ["build-weak-stack-features"]
    statuses = {step["name"]: step["status"] for step in summary["steps"]}
    assert statuses["publish-predictions"] == "ok"
    assert statuses["build-weak-stack-features"] == "failed"
    assert statuses["prospective-score"] == "ok"
    assert "prospective-score" in calls


@pytest.mark.parametrize(
    "failed_step", ["opener-evaluation", "overlay-composition", "publish-board"]
)
def test_measurement_and_final_board_failures_are_fatal(tmp_path: Path, failed_step: str) -> None:
    data_root = _write_data_root(tmp_path)
    artifacts = tmp_path / "artifacts"
    _write_active_model(artifacts, season=2026, week=1, status="SYNCHRONIZED")
    calls: list[str] = []

    def runner(command: Sequence[str]) -> dict[str, Any]:
        calls.append(command[0])
        if command[0] == failed_step:
            raise ValueError("fixture failure")
        return {}

    with pytest.raises(WeeklyRunError, match=failed_step):
        run_weekly(
            season=2026,
            week=1,
            data_root=data_root,
            artifacts_root=artifacts,
            skip_ingest=True,
            skip_prospective=True,
            skip_drift=True,
            runner=runner,
            progress=False,
        )
    assert calls[-1] == failed_step
    if failed_step != "publish-board":
        assert "publish-predictions" not in calls
