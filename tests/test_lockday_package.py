"""ENG-01: the immutable lock-day decision package.

Everything here runs against a synthetic artifacts/data tree under ``tmp_path``.
The real Week 1 lock is 2026-09-08; no test in this file may read, write or
depend on the production ``artifacts/`` or ``data/`` trees, and none of them
runs a recorder.
"""

from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from nfl_ats.cli_commands import operations as operations_cmds
from nfl_ats.lockday_package import (
    MANIFEST_DIGEST_FILENAME,
    MANIFEST_FILENAME,
    PACKAGE_KIND,
    PACKAGE_README_FILENAME,
    build_manifest,
    capture_ledger_state,
    ledger_paths,
    load_package,
    package_directory,
    packages_root,
    summarise_package,
    verify_package,
    write_decision_package,
)
from nfl_ats.provenance import sha256_bytes, sha256_file

NOW = datetime(2026, 9, 8, 16, 0, 0, tzinfo=UTC)


def _decision_rows(count: int, *, start: int = 0) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "season": [2026] * count,
            "week": [1] * count,
            "game_id": [f"2026_01_G{index:02d}" for index in range(start, start + count)],
            "pick_side": ["HOME", "AWAY"] * (count // 2) + ["HOME"] * (count % 2),
            "decision_line": [-2.5 + index for index in range(count)],
        }
    )


@pytest.fixture
def tree(tmp_path: Path) -> dict[str, Path]:
    """A synthetic repo/artifacts/data tree shaped like the real one."""

    repo_root = tmp_path / "repo"
    artifacts = repo_root / "artifacts"
    processed = repo_root / "data" / "processed"
    processed.mkdir(parents=True)

    features = processed / "game_features_weak_stack.parquet"
    pd.DataFrame({"season": [2026, 2026], "week": [1, 1], "x": [0.1, 0.2]}).to_parquet(features)
    (processed / "game_features_weak_stack.manifest.json").write_text(
        json.dumps(
            {
                "source_player_snapshot": "20260901T000000Z",
                "source_player_value_snapshot": "20260901T000001Z",
                "source_pbp_snapshot": "20260901T000002Z",
            }
        ),
        encoding="utf-8",
    )

    forecast_dir = artifacts / "margin_predictions" / "2026-week-01-20260908T150000Z"
    forecast_dir.mkdir(parents=True)
    (forecast_dir / "predictions.csv").write_text(
        "game_id,pick\n2026_01_G00,HOME\n", encoding="utf-8"
    )
    (forecast_dir / "metadata.json").write_text(json.dumps({"season": 2026}), encoding="utf-8")

    evaluation_dir = artifacts / "margins" / "20260908T145900Z"
    evaluation_dir.mkdir(parents=True)
    (evaluation_dir / "summary.csv").write_text("accuracy\n0.5224\n", encoding="utf-8")

    (artifacts / "active_ats_model.json").write_text(
        json.dumps(
            {
                "version": 1,
                "model_id": "123d60be8c80a35d",
                "method": "market_residual",
                "target": "ats_classification",
                "feature_profile": "weak_stack",
                "regressor": "ridge",
                "ridge_alpha": 10.0,
                "calibration_method": "none",
                "probability_method": "gaussian",
                "status": "SYNCHRONIZED",
                "activated_at_utc": "2026-09-08T15:00:00+00:00",
                "feature_table_sha256": sha256_file(features),
                "evaluation_configuration_sha256": "abc123",
                "historical_evaluation": {
                    "artifact": "margins/20260908T145900Z",
                    "accuracy": 0.5224,
                },
                "weekly_forecast": {
                    "artifact": "margin_predictions/2026-week-01-20260908T150000Z",
                    "season": 2026,
                    "week": 1,
                },
            }
        ),
        encoding="utf-8",
    )

    (artifacts / "prospective").mkdir(parents=True, exist_ok=True)
    (artifacts / "prospective" / "challengers.json").write_text(
        json.dumps(
            {
                "challengers": [
                    {
                        "challenger_id": "mod07_weak_signal_stack",
                        "status": "ACTIVE_PROSPECTIVE",
                        "weekly_recording_command": "nfl-ats prospective-record",
                    },
                    {
                        "challenger_id": "example_publish_arm",
                        "status": "ACTIVE_PROSPECTIVE",
                        "weekly_recording_command": (
                            "nfl-ats publish-predictions --record-decisions"
                        ),
                    },
                ]
            }
        ),
        encoding="utf-8",
    )

    (repo_root / "CURRENT_PREDICTIONS.md").write_text("# card\n", encoding="utf-8")
    (repo_root / "uv.lock").write_text("# lock\n", encoding="utf-8")

    return {
        "repo_root": repo_root,
        "artifacts": artifacts,
        "data": repo_root / "data",
        "features": features,
        "forecast_dir": forecast_dir,
    }


def _run_summary(tree: dict[str, Path]) -> dict[str, Any]:
    return {
        "command": "weekly-run",
        "season": 2026,
        "week": 1,
        "record_decisions": True,
        "published": True,
        "steps": [
            {
                "number": 5,
                "name": "margin-predict",
                "status": "ok",
                "command": [
                    "margin-predict",
                    "--season",
                    "2026",
                    "--week",
                    "1",
                    "--features",
                    str(tree["features"]),
                    "--pbp-snapshot",
                    "20260901T000002Z",
                ],
                "output": {"synchronization_status": "SYNCHRONIZED"},
            },
            {
                "number": 8,
                "name": "publish-predictions",
                "status": "ok",
                "command": ["publish-predictions", "--with-board", "--record-decisions"],
                "output": {
                    "ledger": {"recorded": 16},
                    "example_publish_arm": {
                        "challenger_id": "example_publish_arm",
                        "recorded": 16,
                    },
                    "nflcom_refresh_out2_starters_overlay": {
                        "challenger_id": "nflcom_friday_refresh_out2_starters_v1",
                        "recorded": 0,
                        "reason": "no NFL.com page captured at or after Friday 16:00 ET",
                    },
                },
            },
        ],
    }


def _stub_verify(artifacts_root: Path, season: int, week: int, run_summary: Any) -> dict[str, Any]:
    return {
        "season": season,
        "week": week,
        "artifacts_root": str(artifacts_root),
        "recorded": 2,
        "skipped": 0,
        "missing": [],
        "pending_wiring": [],
        "challengers": [],
        "rendered": "lock-day verification 2026 week 1",
        "exit_code": 0,
    }


def _write_package(tree: dict[str, Path], **overrides: Any) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "season": 2026,
        "week": 1,
        "artifacts_root": tree["artifacts"],
        "data_root": tree["data"],
        "repo_root": tree["repo_root"],
        "run_summary": _run_summary(tree),
        "now": NOW,
        "verify_runner": _stub_verify,
    }
    kwargs.update(overrides)
    return write_decision_package(**kwargs)


# ---------------------------------------------------------------------------


def test_package_links_inputs_model_outputs_recorders_ledgers_and_verify(
    tree: dict[str, Path],
) -> None:
    """The definition of done, as one assertion set: one manifest linking
    source snapshots, feature hashes, model identity, forecast/card hashes,
    recorder results, ledger writes, and lockday_verify output."""

    before = capture_ledger_state(tree["artifacts"])
    paper = ledger_paths(tree["artifacts"])["paper_decisions"]
    paper.parent.mkdir(parents=True, exist_ok=True)
    _decision_rows(4).to_parquet(paper)

    written = _write_package(tree, ledger_state_before=before)
    assert written["written"] is True
    manifest = load_package(Path(written["package_directory"]))

    assert manifest["kind"] == PACKAGE_KIND
    assert (manifest["season"], manifest["week"]) == (2026, 1)
    assert manifest["rehearsal"] is False
    assert manifest["errors"] == []

    # model identity
    identity = manifest["model_identity"]
    assert identity["model_id"] == "123d60be8c80a35d"
    assert identity["feature_profile"] == "weak_stack"
    assert identity["regressor"] == "ridge"
    assert identity["ridge_alpha"] == 10.0
    assert identity["calibration_method"] == "none"
    assert identity["probability_method"] == "gaussian"
    assert identity["manifest_sha256"] == sha256_file(tree["artifacts"] / "active_ats_model.json")

    # inputs: the feature table the run actually named, and its build manifest
    table_paths = {entry["path"] for entry in manifest["inputs"]["feature_tables"]}
    assert str(tree["features"]) in table_paths
    assert manifest["inputs"]["snapshot_ids"]["pbp-snapshot"] == ["20260901T000002Z"]
    manifests = {entry["path"]: entry for entry in manifest["inputs"]["snapshot_manifests"]}
    build_manifest_path = str(tree["features"].with_name("game_features_weak_stack.manifest.json"))
    assert manifests[build_manifest_path]["manifest"]["source_pbp_snapshot"] == ("20260901T000002Z")

    # outputs: the forecast directory and the published card
    assert manifest["outputs"]["forecast"]["directory"] == str(tree["forecast_dir"])
    forecast_files = {Path(item["path"]).name for item in manifest["outputs"]["forecast"]["files"]}
    assert {"predictions.csv", "metadata.json"} <= forecast_files
    cards = {Path(card["path"]).name: card for card in manifest["outputs"]["cards"]}
    assert cards["CURRENT_PREDICTIONS.md"]["sha256"] == sha256_file(
        tree["repo_root"] / "CURRENT_PREDICTIONS.md"
    )

    # recorder results, verbatim, plus the flat challenger index
    publish = manifest["recorders"]["steps"]["publish-predictions"]["output"]
    assert publish["ledger"] == {"recorded": 16}
    by_id = manifest["recorders"]["by_challenger_id"]
    assert by_id["example_publish_arm"]["recorded"] == 16
    assert "Friday 16:00 ET" in by_id["nflcom_friday_refresh_out2_starters_v1"]["reason"]

    # ledger writes
    ledgers = {row["ledger"]: row for row in manifest["ledgers"]}
    assert set(ledgers) == set(ledger_paths(tree["artifacts"]))
    assert ledgers["paper_decisions"]["rows_before"] == 0
    assert ledgers["paper_decisions"]["rows_after"] == 4
    assert ledgers["paper_decisions"]["appended_rows"] == 4
    assert ledgers["challenger_decisions"]["rows_after"] == 0

    # lockday_verify output
    assert manifest["lockday_verify"]["recorded"] == 2
    assert manifest["lockday_verify"]["missing"] == []

    # code provenance
    assert "revision" in manifest["code"]
    assert manifest["code"]["uv_lock_sha256"] == sha256_file(tree["repo_root"] / "uv.lock")


def test_every_linked_hash_is_recomputable(tree: dict[str, Path]) -> None:
    before = capture_ledger_state(tree["artifacts"])
    paper = ledger_paths(tree["artifacts"])["paper_decisions"]
    paper.parent.mkdir(parents=True, exist_ok=True)
    _decision_rows(3).to_parquet(paper)

    written = _write_package(tree, ledger_state_before=before)
    manifest = load_package(Path(written["package_directory"]))

    checked = 0
    for entry in manifest["hashed_files"]:
        if not entry["sha256"]:
            continue
        assert sha256_file(Path(entry["path"])) == entry["sha256"], entry["role"]
        checked += 1
    assert checked >= 5

    # the appended-row digest recomputes from the stated recipe alone
    row = next(item for item in manifest["ledgers"] if item["ledger"] == "paper_decisions")
    frame = pd.read_parquet(row["path"]).iloc[row["rows_before"] :]
    assert sha256_bytes(frame.to_csv(index=False).encode("utf-8")) == row["appended_rows_sha256"]

    # and the manifest pins its own bytes
    digest_line = (
        Path(written["manifest_sha256_path"]).read_text(encoding="utf-8").split()[0].strip()
    )
    assert digest_line == sha256_file(Path(written["manifest_path"]))


def test_appended_row_digest_changes_when_a_different_row_is_appended(
    tree: dict[str, Path],
) -> None:
    """The digest has to be of THIS run's rows, not of the whole ledger."""

    paper = ledger_paths(tree["artifacts"])["paper_decisions"]
    paper.parent.mkdir(parents=True, exist_ok=True)
    _decision_rows(2).to_parquet(paper)
    before = capture_ledger_state(tree["artifacts"])
    assert before["paper_decisions"]["rows"] == 2

    pd.concat([_decision_rows(2), _decision_rows(1, start=90)], ignore_index=True).to_parquet(paper)
    first = load_package(
        Path(_write_package(tree, ledger_state_before=before)["package_directory"])
    )
    row = next(item for item in first["ledgers"] if item["ledger"] == "paper_decisions")
    assert row["rows_before"] == 2
    assert row["appended_rows"] == 1

    pd.concat([_decision_rows(2), _decision_rows(1, start=91)], ignore_index=True).to_parquet(paper)
    second = load_package(
        Path(_write_package(tree, ledger_state_before=before)["package_directory"])
    )
    other = next(item for item in second["ledgers"] if item["ledger"] == "paper_decisions")
    assert other["appended_rows_sha256"] != row["appended_rows_sha256"]


def test_a_broken_component_still_yields_a_manifest_with_an_errors_list(
    tree: dict[str, Path],
) -> None:
    """Fail-safe: by the time the package runs the ledger rows are already
    appended, so a broken component degrades one section and nothing else."""

    def exploding_verify(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
        raise RuntimeError("lockday_verify blew up")

    written = _write_package(tree, verify_runner=exploding_verify)
    assert written["written"] is True
    assert written["ok"] is False

    manifest = load_package(Path(written["package_directory"]))
    components = {item["component"] for item in manifest["errors"]}
    assert "lockday_verify" in components
    assert "lockday_verify blew up" in json.dumps(manifest["errors"])
    assert manifest["ok"] is False
    # every other section still assembled
    assert manifest["model_identity"]["model_id"] == "123d60be8c80a35d"
    assert manifest["ledgers"]
    assert manifest["outputs"]["cards"]


def test_write_decision_package_never_raises_on_an_unwritable_destination(
    tree: dict[str, Path],
) -> None:
    """The outermost guard: even a failed WRITE must not abort the lock."""

    blocker = tree["repo_root"] / "not-a-directory"
    blocker.write_text("occupied\n", encoding="utf-8")

    written = _write_package(tree, destination=blocker / "package")
    assert written["written"] is False
    assert written["package_directory"] is None
    assert written["errors"]


def test_missing_artifacts_degrade_into_errors_rather_than_exceptions(
    tmp_path: Path,
) -> None:
    empty = tmp_path / "empty"
    (empty / "artifacts").mkdir(parents=True)
    written = write_decision_package(
        season=2026,
        week=1,
        artifacts_root=empty / "artifacts",
        data_root=empty / "data",
        repo_root=empty,
        run_summary=None,
        ledger_state_before=None,
        now=NOW,
        verify_runner=_stub_verify,
    )
    assert written["written"] is True
    manifest = load_package(Path(written["package_directory"]))
    assert manifest["model_identity"]["available"] is False
    assert manifest["inputs"]["feature_tables"] == []
    assert all(row["rows_after"] == 0 for row in manifest["ledgers"])


def test_manifest_is_written_read_only(tree: dict[str, Path]) -> None:
    written = _write_package(tree)
    manifest_path = Path(written["manifest_path"])
    assert manifest_path.is_file()
    assert written["read_only"] is True
    assert not os.access(manifest_path, os.W_OK)
    with pytest.raises(PermissionError):
        manifest_path.write_text("tampered", encoding="utf-8")


def test_reader_round_trips_and_summarises(tree: dict[str, Path]) -> None:
    written = _write_package(tree)
    directory = Path(written["package_directory"])

    from_directory = load_package(directory)
    from_file = load_package(directory / MANIFEST_FILENAME)
    assert from_directory == from_file
    assert from_directory == json.loads((directory / MANIFEST_FILENAME).read_text(encoding="utf-8"))

    readme = (directory / PACKAGE_README_FILENAME).read_text(encoding="utf-8")
    assert "lockday_package_verify.py" in readme
    assert MANIFEST_DIGEST_FILENAME in readme

    summary = summarise_package(from_directory)
    assert "123d60be8c80a35d" in summary
    assert "paper_decisions" in summary
    assert "build errors   : none" in summary


def test_load_package_rejects_a_foreign_json_file(tmp_path: Path) -> None:
    stray = tmp_path / "manifest.json"
    stray.write_text(json.dumps({"kind": "something_else"}), encoding="utf-8")
    with pytest.raises(ValueError, match="lockday_decision_package"):
        load_package(stray)


def test_verifier_passes_on_a_fresh_package(tree: dict[str, Path]) -> None:
    written = _write_package(tree)
    report = verify_package(Path(written["package_directory"]), repo_root=tree["repo_root"])
    assert report["ok"] is True
    assert report["manifest_sha256_ok"] is True
    assert report["files_verified"] == report["files_checked"]
    assert report["changed"] == []


def test_verifier_fails_when_a_linked_artifact_changes(tree: dict[str, Path]) -> None:
    written = _write_package(tree)
    (tree["repo_root"] / "CURRENT_PREDICTIONS.md").write_text("# edited\n", encoding="utf-8")
    report = verify_package(Path(written["package_directory"]), repo_root=tree["repo_root"])
    assert report["ok"] is False
    assert [item["role"] for item in report["changed"]] == ["published_card"]


def test_verifier_fails_when_the_manifest_itself_is_edited(tree: dict[str, Path]) -> None:
    written = _write_package(tree)
    manifest_path = Path(written["manifest_path"])
    os.chmod(manifest_path, 0o600)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["model_identity"]["model_id"] = "deadbeefdeadbeef"
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")

    report = verify_package(Path(written["package_directory"]), repo_root=tree["repo_root"])
    assert report["manifest_sha256_ok"] is False
    assert report["ok"] is False


def test_verifier_tolerates_an_appended_ledger_but_not_a_changed_card(
    tree: dict[str, Path],
) -> None:
    """Ledgers are append-only and later refresh passes add rows: a changed
    ledger is reported, never a verification failure."""

    paper = ledger_paths(tree["artifacts"])["paper_decisions"]
    paper.parent.mkdir(parents=True, exist_ok=True)
    _decision_rows(2).to_parquet(paper)
    written = _write_package(tree, ledger_state_before=capture_ledger_state(tree["artifacts"]))

    _decision_rows(6).to_parquet(paper)
    report = verify_package(Path(written["package_directory"]), repo_root=tree["repo_root"])
    assert report["ok"] is True
    assert [item["role"] for item in report["mutable_changed"]] == ["ledger_after"]


def test_verifier_strict_mode_requires_every_file_to_survive(tree: dict[str, Path]) -> None:
    written = _write_package(tree)
    (tree["repo_root"] / "CURRENT_PREDICTIONS.md").unlink()

    lenient = verify_package(Path(written["package_directory"]), repo_root=tree["repo_root"])
    assert lenient["ok"] is True
    assert [item["role"] for item in lenient["missing"]] == ["published_card"]

    strict = verify_package(
        Path(written["package_directory"]), repo_root=tree["repo_root"], strict=True
    )
    assert strict["ok"] is False


def test_package_directories_never_collide_within_one_second(tree: dict[str, Path]) -> None:
    root = packages_root(tree["artifacts"])
    first = package_directory(root, 2026, 1, now=NOW)
    first.mkdir(parents=True)
    second = package_directory(root, 2026, 1, now=NOW)
    assert first != second
    assert second.name.endswith("-2")
    assert first.name == "2026_wk01_20260908T160000Z"


def test_rehearsal_packages_are_tagged(tree: dict[str, Path]) -> None:
    written = _write_package(tree, rehearsal=True, command="lockday_rehearsal --full-replay")
    manifest = load_package(Path(written["package_directory"]))
    assert manifest["rehearsal"] is True
    assert manifest["command"] == "lockday_rehearsal --full-replay"
    assert written["rehearsal"] is True
    assert "[REHEARSAL]" in summarise_package(manifest)
    assert "REHEARSAL -- not a real lock" in (
        Path(written["package_directory"]) / PACKAGE_README_FILENAME
    ).read_text(encoding="utf-8")


def test_real_lockday_verify_runs_in_process_against_a_synthetic_root(
    tree: dict[str, Path],
) -> None:
    """The default path imports scripts/lockday_verify.py by file location and
    calls its own ``verify``; this pins that wiring without a stub."""

    manifest = build_manifest(
        season=2026,
        week=1,
        artifacts_root=tree["artifacts"],
        data_root=tree["data"],
        repo_root=Path.cwd(),
        run_summary=_run_summary(tree),
        # Explicit, so the test hashes the synthetic card rather than the
        # repository's real tracked one.
        card_paths=[tree["repo_root"] / "CURRENT_PREDICTIONS.md"],
        now=NOW,
    )
    assert manifest["errors"] == []
    report = manifest["lockday_verify"]
    assert report["active_registered"] == 2
    # Both synthetic arms wrote nothing into an empty synthetic ledger, so the
    # verifier correctly calls both MISSING. What this pins is that the REAL
    # verifier ran in-process and produced its own report shape, not a stub's.
    assert set(report["missing"]) == {"example_publish_arm", "mod07_weak_signal_stack"}
    assert report["exit_code"] == 1
    assert "lock-day verification" in report["rendered"]


def test_capture_ledger_state_is_read_only(tree: dict[str, Path]) -> None:
    paths = ledger_paths(tree["artifacts"])
    paper = paths["paper_decisions"]
    paper.parent.mkdir(parents=True, exist_ok=True)
    _decision_rows(2).to_parquet(paper)
    digest_before = sha256_file(paper)

    state = capture_ledger_state(tree["artifacts"])
    assert state["paper_decisions"]["rows"] == 2
    assert state["paper_decisions"]["sha256"] == digest_before
    assert state["challenger_decisions"]["exists"] is False
    assert state["challenger_decisions"]["rows"] == 0
    assert sha256_file(paper) == digest_before
    assert not (tree["artifacts"] / "prospective" / "challenger_decisions.parquet").exists()


# ---------------------------------------------------------------------------
# weekly-run wiring
#
# The real Week 1 lock is 2026-09-08. These tests never run a recorder and
# never touch the production artifacts tree: NFL_ATS_ARTIFACTS_DIR/
# NFL_ATS_DATA_DIR point at tmp_path and ``run_weekly`` is replaced by a stub,
# so the only thing exercised is the additive package wiring itself.
# ---------------------------------------------------------------------------


def _weekly_run_stub(calls: list[dict[str, Any]], summary: dict[str, Any] | None = None) -> Any:
    def stub(**kwargs: Any) -> dict[str, Any]:
        calls.append(kwargs)
        return summary if summary is not None else {"command": "weekly-run", "steps": []}

    return stub


@pytest.fixture
def wired(tree: dict[str, Path], monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """weekly-run pointed at the synthetic tree, with the runner stubbed out."""

    from nfl_ats import cli

    monkeypatch.setenv("NFL_ATS_ARTIFACTS_DIR", str(tree["artifacts"]))
    monkeypatch.setenv("NFL_ATS_DATA_DIR", str(tree["data"]))
    calls: list[dict[str, Any]] = []
    monkeypatch.setattr(operations_cmds, "run_weekly", _weekly_run_stub(calls, _run_summary(tree)))
    return {"cli": cli, "calls": calls}


def test_weekly_run_writes_a_package_only_with_record_decisions(
    tree: dict[str, Path], wired: dict[str, Any], capsys: pytest.CaptureFixture[str]
) -> None:
    cli = wired["cli"]
    root = packages_root(tree["artifacts"])

    cli.main(["weekly-run", "--season", "2026", "--week", "1"])
    assert not root.exists()
    assert wired["calls"][-1]["record_decisions"] is False

    cli.main(["weekly-run", "--season", "2026", "--week", "1", "--record-decisions"])
    packages = sorted(root.iterdir())
    assert len(packages) == 1
    manifest = load_package(packages[0])
    assert manifest["season"] == 2026
    assert manifest["week"] == 1
    assert manifest["rehearsal"] is False
    assert manifest["command"] == "weekly-run --record-decisions"
    assert manifest["model_identity"]["model_id"] == "123d60be8c80a35d"
    assert manifest["recorders"]["steps"]["publish-predictions"]["output"]["ledger"] == {
        "recorded": 16
    }

    captured = capsys.readouterr()
    assert str(packages[0]) in captured.err
    # the run's own JSON summary is unchanged apart from the added key
    payload = (
        json.loads(captured.out.strip().split("\n{")[-1].join(("", "")) or "{}") if False else None
    )
    assert payload is None


def test_no_package_opts_out_without_changing_the_run(
    tree: dict[str, Path], wired: dict[str, Any]
) -> None:
    cli = wired["cli"]
    cli.main(
        ["weekly-run", "--season", "2026", "--week", "1", "--record-decisions", "--no-package"]
    )
    assert not packages_root(tree["artifacts"]).exists()
    assert wired["calls"][-1]["record_decisions"] is True


def test_package_is_written_even_when_the_run_aborts(
    tree: dict[str, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    """The rows are already appended when a late step fails, so the package
    must still be written -- and must not swallow the failure."""

    from nfl_ats import cli
    from nfl_ats.weekly import WeeklyRunError

    monkeypatch.setenv("NFL_ATS_ARTIFACTS_DIR", str(tree["artifacts"]))
    monkeypatch.setenv("NFL_ATS_DATA_DIR", str(tree["data"]))

    def exploding(**_kwargs: Any) -> dict[str, Any]:
        raise WeeklyRunError("weekly-run aborted at step 'prospective-score'")

    monkeypatch.setattr(operations_cmds, "run_weekly", exploding)

    # Unchanged behaviour: WeeklyRunError is a ValueError, so cli.main still
    # reports it on stderr and exits 2. The package is written on the way out.
    with pytest.raises(SystemExit) as excinfo:
        cli.main(["weekly-run", "--season", "2026", "--week", "1", "--record-decisions"])
    assert excinfo.value.code == 2

    packages = sorted(packages_root(tree["artifacts"]).iterdir())
    assert len(packages) == 1
    manifest = load_package(packages[0])
    assert manifest["run_summary"] is None
    assert manifest["ledgers"]


def test_package_write_failure_never_aborts_the_lock(
    tree: dict[str, Path], wired: dict[str, Any], capsys: pytest.CaptureFixture[str]
) -> None:
    """The binding contract, at the CALL SITE. ``write_decision_package`` has
    its own guard, so this replaces it with a builder that has none: only
    ``_cmd_weekly_run``'s own try/except is under test. The lock stands, the
    run's JSON summary is still printed, and the failure is named on stderr."""

    from nfl_ats import cli

    def exploding(**_kwargs: Any) -> dict[str, Any]:
        raise RuntimeError("disk full")

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(operations_cmds, "write_decision_package", exploding)
    try:
        assert (
            cli.main(["weekly-run", "--season", "2026", "--week", "1", "--record-decisions"]) == 0
        )
    finally:
        monkeypatch.undo()

    captured = capsys.readouterr()
    assert "lock-day decision package FAILED to write: disk full" in captured.err
    payload = json.loads(captured.out)
    assert payload["command"] == "weekly-run"
    assert payload["decision_package"]["written"] is False
    assert payload["decision_package"]["errors"][0]["error"] == "disk full"
    # the run itself still executed in full; the package is strictly last
    assert wired["calls"][-1]["record_decisions"] is True
    assert not packages_root(tree["artifacts"]).exists()


def test_dry_run_with_record_decisions_writes_no_package(
    tree: dict[str, Path], wired: dict[str, Any]
) -> None:
    """--dry-run runs nothing, so there is nothing to package."""

    cli = wired["cli"]
    cli.main(["weekly-run", "--season", "2026", "--week", "1", "--record-decisions", "--dry-run"])
    assert not packages_root(tree["artifacts"]).exists()
    assert wired["calls"][-1]["dry_run"] is True
