from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest

from nfl_ats import cli, weekly
from nfl_ats.io import atomic_json
from nfl_ats.weekly import (
    PLAYER_FEATURE_PROFILE,
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
    steps = plan_weekly_run(season=2026, week=1, data_root=data_root, skip_prospective=True)

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
    steps = {
        step.name: step
        for step in plan_weekly_run(season=2026, week=1, data_root=data_root, skip_prospective=True)
    }

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
            season=2026,
            week=1,
            data_root=data_root,
            refresh_player_data=True,
            skip_prospective=True,
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
        plan_weekly_run(season=2026, week=1, data_root=data_root, skip_prospective=True)


def test_plan_aborts_when_a_production_manifest_is_missing(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    atomic_json(
        {"snapshot_id": "s", "seasons": [2026], "team_stat_seasons": [2025]},
        data_root / "raw" / "s" / "manifest.json",
    )
    with pytest.raises(WeeklyRunError, match="--refresh-player-data"):
        plan_weekly_run(season=2026, week=1, data_root=data_root, skip_prospective=True)


def test_dry_run_prints_the_plan_and_runs_nothing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    data_root = _write_data_root(tmp_path)
    monkeypatch.setenv("NFL_ATS_DATA_DIR", str(data_root))
    monkeypatch.setenv("NFL_ATS_ARTIFACTS_DIR", str(tmp_path / "artifacts"))
    monkeypatch.setenv("NFL_ATS_REGISTRY_DIR", str(tmp_path / "registry"))

    def explode(command: Sequence[str]) -> dict[str, Any]:
        raise AssertionError(f"dry run executed {command!r}")

    monkeypatch.setattr(weekly, "_cli_runner", explode)

    assert (
        cli.main(
            [
                "weekly-run",
                "--season",
                "2026",
                "--week",
                "1",
                "--dry-run",
                "--skip-prospective",
            ]
        )
        == 0
    )
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
        skip_prospective=True,
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
        skip_prospective=True,
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
            skip_prospective=True,
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
            skip_prospective=True,
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


def test_the_card_path_follows_the_active_profile_instead_of_reverting_it(
    tmp_path: Path,
) -> None:
    """A promotion made outside weekly-run must not be silently undone.

    ``margin-predict`` activates whatever profile step 4 just evaluated, and
    ``assert-synchronized`` cannot catch a revert because the reverted model
    still points at the right season/week. So the card path reads the ACTIVE
    profile and builds that, rather than a hardcoded one.
    """

    data_root = _write_data_root(tmp_path)
    artifacts_root = tmp_path / "artifacts"
    atomic_json(
        {
            "version": 1,
            "status": "SYNCHRONIZED",
            "model_id": "118f31d9a98c815b",
            "feature_profile": "weak_stack",
            "historical_evaluation": {"accuracy": 0.5156626506024097, "games": 2075},
            "weekly_forecast": {"season": 2026, "week": 1},
        },
        artifacts_root / "active_ats_model.json",
    )

    steps = plan_weekly_run(
        season=2026,
        week=1,
        data_root=data_root,
        artifacts_root=artifacts_root,
        skip_prospective=True,
    )
    scoring = [s for s in steps if s.name in {"margin-backtest", "margin-predict"}]
    assert scoring, "the card path must still evaluate and score"
    for step in scoring:
        assert "weak_stack" in step.command
        assert "game_features_weak_stack.parquet" in " ".join(step.command)
        assert PLAYER_FEATURE_PROFILE not in step.command

    # The active profile's table has to be built before it can be scored.
    build = [s for s in steps if s.name == "build-weak-stack-features" and not s.optional]
    assert build, "the card path must build the table the active model scores on"
    assert steps.index(build[0]) < steps.index(scoring[0])


def test_an_unknown_active_profile_is_fatal_rather_than_guessed(
    tmp_path: Path,
) -> None:
    """Guessing a feature table would reintroduce the revert this prevents."""

    data_root = _write_data_root(tmp_path)
    artifacts_root = tmp_path / "artifacts"
    atomic_json(
        {
            "version": 1,
            "status": "SYNCHRONIZED",
            "model_id": "deadbeef",
            "feature_profile": "some_future_profile",
            "weekly_forecast": {"season": 2026, "week": 1},
        },
        artifacts_root / "active_ats_model.json",
    )
    runner = _Recorder()

    with pytest.raises(WeeklyRunError, match="cannot"):
        run_weekly(
            season=2026,
            week=1,
            data_root=data_root,
            artifacts_root=artifacts_root,
            skip_prospective=True,
            runner=runner,
            progress=False,
        )

    assert runner.names == []


def test_run_proceeds_when_the_active_profile_already_matches_the_card_path(
    tmp_path: Path,
) -> None:
    data_root = _write_data_root(tmp_path)
    artifacts_root = tmp_path / "artifacts"
    atomic_json(
        {
            "version": 1,
            "status": "SYNCHRONIZED",
            "model_id": "80e458040e48b926",
            "feature_profile": "player",
            "historical_evaluation": {"accuracy": 0.5204819277, "games": 2075},
            "weekly_forecast": {"season": 2026, "week": 1},
        },
        artifacts_root / "active_ats_model.json",
    )
    runner = _Recorder(**{"margin-predict": {"synchronization_status": "SYNCHRONIZED"}})

    summary = run_weekly(
        season=2026,
        week=1,
        data_root=data_root,
        artifacts_root=artifacts_root,
        skip_prospective=True,
        runner=runner,
        progress=False,
    )

    assert summary["published"] is True


def test_cli_reports_an_abort_as_a_user_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data_root = _write_data_root(tmp_path)
    monkeypatch.setenv("NFL_ATS_DATA_DIR", str(data_root))
    monkeypatch.setenv("NFL_ATS_ARTIFACTS_DIR", str(tmp_path / "artifacts"))
    monkeypatch.setenv("NFL_ATS_REGISTRY_DIR", str(tmp_path / "registry"))
    monkeypatch.setattr(weekly, "_cli_runner", lambda command: {})

    with pytest.raises(SystemExit) as excinfo:
        cli.main(["weekly-run", "--season", "2026", "--week", "1"])
    assert excinfo.value.code == 2


# ---------------------------------------------------------------------------
# POL-10: prospective evidence collection (steps 8-11)
# ---------------------------------------------------------------------------


def test_prospective_steps_trail_the_publish_and_are_optional(tmp_path: Path) -> None:
    data_root = _write_data_root(tmp_path)
    steps = plan_weekly_run(season=2026, week=1, data_root=data_root)
    names = [step.name for step in steps]

    # The SPEC-3 core is untouched, and the evidence steps come strictly after
    # the publish -- research collection must never delay or endanger the card.
    assert names[:8] == [
        "ingest",
        "build-features",
        "build-pbp-features",
        "build-player-features",
        "margin-backtest",
        "margin-predict",
        "assert-synchronized",
        "publish-predictions",
    ]
    assert names[8:] == PROSPECTIVE_STEPS
    by_name = {step.name: step for step in steps}
    assert all(by_name[name].optional for name in PROSPECTIVE_STEPS)
    assert not any(by_name[name].optional for name in names[:8])

    processed = data_root / "processed"
    assert by_name["build-weak-stack-features"].command == (
        "build-learned-availability-features",
        "--features",
        str(processed / "game_features_pbp.parquet"),
        "--destination",
        str(processed / "game_features_weak_stack.parquet"),
        "--rates-destination",
        str(processed / "weak_stack_availability_rates.parquet"),
        "--evaluation-destination",
        str(processed / "weak_stack_availability_evaluation.csv"),
        "--player-snapshot",
        PRODUCTION_PLAYER_SNAPSHOT,
        "--player-value-snapshot",
        PRODUCTION_PLAYER_VALUE_SNAPSHOT,
        "--pbp-snapshot",
        PRODUCTION_PBP_SNAPSHOT,
    )
    # The challenger is scored on its OWN table and profile, never the player one.
    assert by_name["margin-predict-challenger"].command == (
        "margin-predict",
        "--season",
        "2026",
        "--week",
        "1",
        "--features",
        str(processed / "game_features_weak_stack.parquet"),
        "--feature-profile",
        "weak_stack",
    )
    assert by_name["prospective-record"].command == (
        "prospective-record",
        "--challenger",
        "mod07_weak_signal_stack",
        "--season",
        "2026",
        "--week",
        "1",
    )
    assert by_name["prospective-score"].command == ("prospective-score",)


def test_record_decisions_defaults_to_false_and_does_not_reach_either_ledger(
    tmp_path: Path,
) -> None:
    """The safe default: neither step 7's publish nor step 10's challenger
    record is told to write anywhere. This is the guard for the 2026-08-18
    incident (docs/prospective_evidence.md, 'Known divergence') -- an
    ordinary/rehearsal weekly-run must not be able to reach either ledger."""

    data_root = _write_data_root(tmp_path)
    steps = plan_weekly_run(season=2026, week=1, data_root=data_root)
    by_name = {step.name: step for step in steps}

    assert by_name["publish-predictions"].command == ("publish-predictions", "--with-board")
    assert "not recording" in by_name["publish-predictions"].notes[0]

    record_step = by_name["prospective-record"]
    assert record_step.skipped is True
    assert record_step.optional is True
    # The command is still shown (dry-run doubles as the manual fallback),
    # it is just not executed.
    assert record_step.command[0] == "prospective-record"
    assert "--record-decisions" in record_step.notes[0]


def test_record_decisions_true_wires_both_ledger_writes(tmp_path: Path) -> None:
    data_root = _write_data_root(tmp_path)
    steps = plan_weekly_run(season=2026, week=1, data_root=data_root, record_decisions=True)
    by_name = {step.name: step for step in steps}

    assert by_name["publish-predictions"].command == (
        "publish-predictions",
        "--with-board",
        "--record-decisions",
    )
    assert by_name["publish-predictions"].notes == ()

    record_step = by_name["prospective-record"]
    assert record_step.skipped is False
    assert record_step.command == (
        "prospective-record",
        "--challenger",
        "mod07_weak_signal_stack",
        "--season",
        "2026",
        "--week",
        "1",
    )


def test_run_weekly_forwards_record_decisions_and_reports_it(tmp_path: Path) -> None:
    data_root = _write_data_root(tmp_path)
    artifacts_root = tmp_path / "artifacts"
    _write_active_model(artifacts_root, season=2026, week=1, status="SYNCHRONIZED")
    runner = _Recorder(**{"margin-predict": {"synchronization_status": "SYNCHRONIZED"}})

    summary = run_weekly(
        season=2026,
        week=1,
        data_root=data_root,
        artifacts_root=artifacts_root,
        skip_prospective=True,
        record_decisions=True,
        runner=runner,
        progress=False,
    )

    assert summary["record_decisions"] is True
    publish_call = next(c for c in runner.commands if c[0] == "publish-predictions")
    assert "--record-decisions" in publish_call


def test_missing_challenger_manifest_skips_the_tail_without_breaking_the_plan(
    tmp_path: Path,
) -> None:
    data_root = _write_data_root(tmp_path)
    (data_root / "processed" / "game_features_weak_stack.manifest.json").unlink()

    steps = plan_weekly_run(season=2026, week=1, data_root=data_root)
    assert [step.name for step in steps][:8] == [
        "ingest",
        "build-features",
        "build-pbp-features",
        "build-player-features",
        "margin-backtest",
        "margin-predict",
        "assert-synchronized",
        "publish-predictions",
    ]
    tail = steps[8]
    assert tail.name == "build-weak-stack-features"
    assert tail.skipped and tail.optional
    assert "challenger evidence unavailable" in tail.notes[0]


def test_an_optional_step_failure_is_reported_but_never_aborts_the_run(
    tmp_path: Path,
) -> None:
    """The card is already published by step 8; losing a week of research
    evidence must not take the published card's run down with it."""

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
    # The rest of the tail still runs -- one broken step does not cancel the others.
    assert statuses["prospective-score"] == "ok"
    assert "prospective-score" in calls
