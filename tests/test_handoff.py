from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from nfl_ats.handoff import (
    RepositoryState,
    check_session_handoff,
    write_session_handoff,
)


def _write_project_context(root: Path) -> None:
    (root / "ROADMAP.md").write_text(
        """# Roadmap

## Recommended execution order

Context before the ordered work.

1. Preserve prediction safety.
2. Build the player layer.
3. Add joint score distributions.
""",
        encoding="utf-8",
    )
    (root / "CURRENT_PREDICTIONS.md").write_text(
        """# NFL ATS predictions: 2026 Week 1

Published from synchronized model `published123` at `2026-08-12T22:00:00+00:00`.
""",
        encoding="utf-8",
    )


def test_handoff_captures_git_model_publication_and_priorities(tmp_path: Path) -> None:
    _write_project_context(tmp_path)
    artifacts = tmp_path / "artifacts"
    (artifacts / "margins/eval").mkdir(parents=True)
    (artifacts / "margin_predictions/week").mkdir(parents=True)
    manifest = {
        "version": 1,
        "status": "SYNCHRONIZED",
        "model_id": "active123",
        "feature_table_sha256": "fixture-table",
        "method": "market_residual",
        "feature_profile": "player",
        "regressor": "ridge",
        "historical_evaluation": {
            "artifact": "margins/eval",
            "correct": 1080,
            "games": 2075,
            "accuracy": 1080 / 2075,
        },
        "weekly_forecast": {
            "artifact": "margin_predictions/week",
            "season": 2026,
            "week": 1,
            "created_at_utc": "2026-08-12T21:15:33+00:00",
        },
    }
    artifacts.mkdir(exist_ok=True)
    (artifacts / "active_ats_model.json").write_text(json.dumps(manifest), encoding="utf-8")
    opener = artifacts / "opener_evaluation/20260819T174244Z"
    opener.mkdir(parents=True)
    (opener / "metadata.json").write_text(
        json.dumps(
            {
                "active_model_id": "active123",
                "provenance": {"feature_table": {"sha256": "fixture-table"}},
                "active_model_config": {
                    "feature_profile": "player",
                    "regressor": "ridge",
                    "ridge_alpha": 10.0,
                    "target": "market_residual",
                },
                "games": 1537,
                "metrics": {"opener_accuracy_probability_rule": 0.5335994677312043},
            }
        ),
        encoding="utf-8",
    )
    state = RepositoryState(
        branch="master",
        commit="abc123def456",
        subject="Test baseline",
        changes=(" M README.md", "?? notes.txt"),
    )

    result = write_session_handoff(
        tmp_path,
        artifacts,
        Path("HANDOFF.md"),
        generated_at=datetime(2026, 8, 13, 12, 0, tzinfo=UTC),
        state=state,
    )

    text = (tmp_path / "HANDOFF.md").read_text(encoding="utf-8")
    assert "Baseline commit: `abc123def456`" in text
    assert "Pending change set: 2 paths" in text
    assert "Model ID: `active123`" in text
    assert "Raw-model baseline (opener-graded probability rule): **53.36%**" in text
    assert "Promoted player-arrest policy component (opener-graded): **53.76%**" in text
    assert "**53.36%** on **1,503 games**" in text
    assert "probability_positive=0.8562" in text
    assert "Secondary close-grade historical classification" in text
    assert "1,080 / 2,075 (52.05%)" in text
    assert "2026 Week 1" in text
    assert "tracked publication does not match the local active model" in text
    assert "1. Preserve prediction safety." in text
    assert "2. Build the player layer." in text
    assert result["active_model_id"] == "active123"
    assert result["published_model_id"] == "published123"
    assert result["worktree_clean_before_refresh"] is False


def test_handoff_is_useful_without_ignored_local_artifacts(tmp_path: Path) -> None:
    _write_project_context(tmp_path)
    state = RepositoryState(
        branch="master",
        commit="abc123def456",
        subject="Fresh clone",
        changes=(),
    )

    result = write_session_handoff(
        tmp_path,
        tmp_path / "artifacts",
        Path("HANDOFF.md"),
        generated_at=datetime(2026, 8, 13, 12, 0, tzinfo=UTC),
        state=state,
    )

    text = (tmp_path / "HANDOFF.md").read_text(encoding="utf-8")
    assert "expected in a fresh clone" in text
    assert "last published state" in text
    assert "published123" in text
    assert "Pending change set: none" in text
    assert result["active_model_id"] is None

    check = check_session_handoff(tmp_path, tmp_path / "artifacts", Path("HANDOFF.md"))
    assert check["status"] == "CURRENT"
    assert check["published_model_id"] == "published123"


def test_handoff_write_regenerates_readme_generated_blocks(tmp_path: Path) -> None:
    _write_project_context(tmp_path)
    (tmp_path / "README.md").write_text(
        "# NFL ATS\n\nIntro paragraph.\n\n## Details\nMore.\n", encoding="utf-8"
    )
    artifacts = tmp_path / "artifacts"
    manifest = {
        "version": 1,
        "status": "SYNCHRONIZED",
        # Matches _write_project_context's CURRENT_PREDICTIONS.md so the
        # pre-existing "local active model and tracked weekly publication do
        # not match" check stays out of this test's way.
        "model_id": "published123",
        "method": "market_residual",
        "feature_profile": "player",
        "regressor": "ridge",
        "historical_evaluation": {
            "artifact": "margins/eval",
            "correct": 1080,
            "games": 2075,
            "accuracy": 1080 / 2075,
        },
        "weekly_forecast": {
            "artifact": "margin_predictions/week",
            "season": 2026,
            "week": 1,
            "created_at_utc": "2026-08-12T21:15:33+00:00",
        },
    }
    (artifacts / "margins/eval").mkdir(parents=True)
    (artifacts / "margin_predictions/week").mkdir(parents=True)
    (artifacts / "active_ats_model.json").write_text(json.dumps(manifest), encoding="utf-8")
    state = RepositoryState(branch="master", commit="abc123def456", subject="Test", changes=())

    result = write_session_handoff(
        tmp_path,
        artifacts,
        Path("HANDOFF.md"),
        generated_at=datetime(2026, 8, 13, 12, 0, tzinfo=UTC),
        state=state,
        registry_root=tmp_path / "registry",
    )

    readme_text = (tmp_path / "README.md").read_text(encoding="utf-8")
    assert "`published123`" in readme_text
    assert "<!-- ACTIVE_MODEL_STATE:START -->" in readme_text
    assert "<!-- RESEARCH_STATE:START -->" in readme_text
    assert result["readme"]["changed"] is True

    check = check_session_handoff(
        tmp_path, artifacts, Path("HANDOFF.md"), registry_root=tmp_path / "registry"
    )
    assert check["status"] == "CURRENT"


def test_handoff_check_flags_stale_readme_generated_block(tmp_path: Path) -> None:
    _write_project_context(tmp_path)
    artifacts = tmp_path / "artifacts"
    manifest = {
        "version": 1,
        "status": "SYNCHRONIZED",
        "model_id": "published123",
        "method": "market_residual",
        "feature_profile": "player",
        "regressor": "ridge",
        "historical_evaluation": {
            "artifact": "margins/eval",
            "correct": 1080,
            "games": 2075,
            "accuracy": 1080 / 2075,
        },
        "weekly_forecast": {
            "artifact": "margin_predictions/week",
            "season": 2026,
            "week": 1,
            "created_at_utc": "2026-08-12T21:15:33+00:00",
        },
    }
    (artifacts / "margins/eval").mkdir(parents=True)
    (artifacts / "margin_predictions/week").mkdir(parents=True)
    (artifacts / "active_ats_model.json").write_text(json.dumps(manifest), encoding="utf-8")
    (tmp_path / "README.md").write_text(
        "# NFL ATS\n\nIntro paragraph.\n\n## Details\nMore.\n", encoding="utf-8"
    )
    state = RepositoryState(branch="master", commit="abc123def456", subject="Test", changes=())
    write_session_handoff(
        tmp_path,
        artifacts,
        Path("HANDOFF.md"),
        generated_at=datetime(2026, 8, 13, 12, 0, tzinfo=UTC),
        state=state,
        registry_root=tmp_path / "registry",
    )

    # The model changes after the README was last regenerated -- e.g. a new
    # promotion ran without re-publishing. (Now mismatches the tracked
    # publication too, but that failure is independent of, and additive
    # with, the README staleness this test targets.)
    manifest["model_id"] = "active456"
    (artifacts / "active_ats_model.json").write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError) as error:
        check_session_handoff(
            tmp_path, artifacts, Path("HANDOFF.md"), registry_root=tmp_path / "registry"
        )
    assert "active-model-state block is stale" in str(error.value)


def test_handoff_check_rejects_changed_roadmap(tmp_path: Path) -> None:
    _write_project_context(tmp_path)
    state = RepositoryState(
        branch="master",
        commit="abc123def456",
        subject="Baseline",
        changes=(),
    )
    write_session_handoff(
        tmp_path,
        tmp_path / "artifacts",
        Path("HANDOFF.md"),
        generated_at=datetime(2026, 8, 13, 12, 0, tzinfo=UTC),
        state=state,
    )
    (tmp_path / "ROADMAP.md").write_text(
        """# Roadmap

## Recommended execution order

1. A newly promoted priority.
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError) as error:
        check_session_handoff(tmp_path, tmp_path / "artifacts", Path("HANDOFF.md"))
    assert "roadmap execution priorities" in str(error.value)
