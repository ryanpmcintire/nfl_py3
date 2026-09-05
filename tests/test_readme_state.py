from __future__ import annotations

import json
from pathlib import Path

import pytest

from nfl_ats import rotation, weak_signals
from nfl_ats.readme_state import (
    README_ACTIVE_MODEL_END,
    README_ACTIVE_MODEL_START,
    README_RESEARCH_STATE_END,
    README_RESEARCH_STATE_START,
    apply_generated_state_blocks,
    readme_state_failures,
    regenerate_readme_state,
    render_active_model_block,
    render_research_state_block,
)


def _write_active_model(artifacts_root: Path, *, model_id: str = "model-abc") -> None:
    manifest = {
        "version": 1,
        "status": "SYNCHRONIZED",
        "model_id": model_id,
        "method": "market_residual",
        "feature_profile": "weak_stack",
        "regressor": "ridge",
        "ridge_alpha": 10.0,
        "calibration_method": "none",
        "historical_evaluation": {
            "artifact": "margins/eval",
            "correct": 1081,
            "games": 2075,
            "accuracy": 1081 / 2075,
            "intervals": {"week": {"lower": 0.5011, "upper": 0.5423}},
        },
        "weekly_forecast": {
            "artifact": "margin_predictions/week",
            "season": 2026,
            "week": 1,
            "created_at_utc": "2026-08-24T12:07:25+00:00",
        },
    }
    artifacts_root.mkdir(parents=True, exist_ok=True)
    (artifacts_root / "active_ats_model.json").write_text(json.dumps(manifest), encoding="utf-8")


def _write_opener_evaluation(artifacts_root: Path) -> None:
    run = artifacts_root / "opener_evaluation" / "20260819T174244Z"
    run.mkdir(parents=True)
    (run / "metadata.json").write_text(
        json.dumps(
            {
                "active_model_config": {
                    "feature_profile": "weak_stack",
                    "regressor": "ridge",
                    "ridge_alpha": 10.0,
                    "target": "market_residual",
                },
                "games": 1537,
                "metrics": {"opener_accuracy_probability_rule": 0.5336},
                "uncertainty": [
                    {
                        "block": "week",
                        "metric": "opener_accuracy_probability_rule",
                        "lower": 0.5070,
                        "upper": 0.5609,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )


def test_active_model_block_absent_renders_fresh_clone_fallback(tmp_path: Path) -> None:
    text = render_active_model_block(tmp_path / "artifacts")
    assert "not built in this clone" in text
    assert "publish-predictions" in text


def test_active_model_block_reports_both_grades(tmp_path: Path) -> None:
    artifacts_root = tmp_path / "artifacts"
    _write_active_model(artifacts_root)
    _write_opener_evaluation(artifacts_root)

    text = render_active_model_block(artifacts_root)
    assert "`d1f07d773475dc58`" not in text  # sanity: fixture uses model-abc, not the real id
    assert "`model-abc`" in text
    assert "**53.36%** on **1,537 paired games**" in text
    assert "week-blocked 95% interval [50.70%, 56.09%]" in text
    assert "1,081 of 2,075 non-push games" in text


def test_active_model_block_without_matching_opener_run_says_unavailable(tmp_path: Path) -> None:
    artifacts_root = tmp_path / "artifacts"
    _write_active_model(artifacts_root)
    # No opener_evaluation/ directory at all.
    text = render_active_model_block(artifacts_root)
    assert "unavailable in local artifacts" in text
    assert "opener-evaluation" in text


def test_research_state_block_none_registry_root(tmp_path: Path) -> None:
    text = render_research_state_block(None, tmp_path / "artifacts")
    assert "not available" in text


def test_research_state_block_missing_files_render_honest_fallbacks(tmp_path: Path) -> None:
    text = render_research_state_block(tmp_path / "registry", tmp_path / "artifacts")
    assert "Weak-signal registry:** 0 results recorded yet" in text
    assert "Rotation registry:** not available in this clone" in text
    assert "Prospective challengers:** not available in this clone" in text


def test_research_state_block_counts_registries(tmp_path: Path) -> None:
    registry_root = tmp_path / "registry"
    registry_root.mkdir()

    weak_registry = weak_signals.Registry(
        version=weak_signals.WEAK_SIGNAL_REGISTRY_VERSION,
        notes=(),
        signals={
            "sig_a": weak_signals.WeakSignal(
                name="sig_a",
                recorded_at="2026-08-01",
                description="d",
                source="s",
                effect=0.01,
                effect_units="accuracy_points",
                classification="unresolved_below_power",
                league="nfl",
                seasons=(2018, 2025),
            ),
            "sig_b": weak_signals.WeakSignal(
                name="sig_b",
                recorded_at="2026-08-01",
                description="d",
                source="s",
                effect=0.02,
                effect_units="accuracy_points",
                classification="unresolved_below_power",
                league="nfl",
                seasons=(2018, 2025),
            ),
            "sig_c": weak_signals.WeakSignal(
                name="sig_c",
                recorded_at="2026-08-01",
                description="d",
                source="s",
                effect=-0.5,
                effect_units="accuracy_points",
                classification="refuted_mechanism",
                league="nfl",
                seasons=(2018, 2025),
                interval=(-0.9, -0.1),
                closing_ground="wrong_sign_resolved",
                classification_evidence="whole interval below zero",
            ),
        },
    )
    weak_signals.save_registry(weak_registry, weak_signals.default_registry_path(registry_root))

    rotation_registry = rotation.Registry(version=1, notes=(), families={})
    rotation_registry = rotation.declare_family(
        rotation_registry, "family_one", description="d", grade="opener"
    )
    rotation_registry = rotation.declare_family(
        rotation_registry, "family_two", description="d", grade="close"
    )
    rotation.save_registry(rotation_registry, registry_root / rotation.ROTATION_REGISTRY_FILENAME)

    artifacts_root = tmp_path / "artifacts"
    (artifacts_root / "prospective").mkdir(parents=True)
    (artifacts_root / "prospective" / "challengers.json").write_text(
        json.dumps(
            {
                "challengers": [
                    {"challenger_id": "c1", "status": "ACTIVE_PROSPECTIVE"},
                    {"challenger_id": "c2", "status": "ACTIVE_PROSPECTIVE"},
                    {"challenger_id": "c3", "status": "SUPERSEDED_BY_PROMOTION"},
                ]
            }
        ),
        encoding="utf-8",
    )

    text = render_research_state_block(registry_root, artifacts_root)
    assert "3 results recorded -- 2 unresolved_below_power, 1 closed" in text
    assert "1 refuted_mechanism, 0 bounded_by_control" in text
    assert "2 declared research families -- 2 open, 0 confirmed/closed/retired" in text
    assert "2 of 3 registered challengers are actively tracked prospectively" in text


def test_rotation_summary_counts_coverage_stubs_separately(tmp_path: Path) -> None:
    """ENG-37: a coverage stub is neither open nor closed and must not be counted as closed."""

    registry_root = tmp_path / "registry"
    registry_root.mkdir()
    rotation_registry = rotation.Registry(version=1, notes=(), families={})
    rotation_registry = rotation.declare_family(
        rotation_registry, "family_open", description="d", grade="opener"
    )
    rotation_registry = rotation.declare_coverage_stub(
        rotation_registry, "family_stub", weak_signal_family="family_stub", league="nfl"
    )
    rotation.save_registry(rotation_registry, registry_root / rotation.ROTATION_REGISTRY_FILENAME)

    text = render_research_state_block(registry_root, tmp_path / "artifacts")
    assert (
        "2 declared research families -- 1 open, 0 confirmed/closed/retired, "
        "1 declared for coverage only (no window yet)."
    ) in text


def test_apply_generated_state_blocks_bootstraps_missing_markers(tmp_path: Path) -> None:
    text = "# Project\n\nIntro.\n\n## Details\nMore.\n"
    updated = apply_generated_state_blocks(
        text, artifacts_root=tmp_path / "artifacts", registry_root=None
    )
    assert updated.count(README_ACTIVE_MODEL_START) == 1
    assert updated.count(README_ACTIVE_MODEL_END) == 1
    assert updated.count(README_RESEARCH_STATE_START) == 1
    assert updated.count(README_RESEARCH_STATE_END) == 1
    # Untouched surrounding prose survives the bootstrap insert.
    assert "# Project" in updated
    assert "## Details" in updated


def test_apply_generated_state_blocks_replaces_existing_pair_in_place(tmp_path: Path) -> None:
    artifacts_root = tmp_path / "artifacts"
    _write_active_model(artifacts_root, model_id="first-id")
    text = "# Project\n\nIntro.\n"
    once = apply_generated_state_blocks(text, artifacts_root=artifacts_root, registry_root=None)
    assert "`first-id`" in once

    _write_active_model(artifacts_root, model_id="second-id")
    twice = apply_generated_state_blocks(once, artifacts_root=artifacts_root, registry_root=None)
    assert "`second-id`" in twice
    assert "`first-id`" not in twice
    assert twice.count(README_ACTIVE_MODEL_START) == 1


def test_apply_generated_state_blocks_rejects_duplicated_markers(tmp_path: Path) -> None:
    text = (
        f"{README_ACTIVE_MODEL_START}\nold\n{README_ACTIVE_MODEL_END}\n\n"
        f"{README_ACTIVE_MODEL_START}\nold again\n{README_ACTIVE_MODEL_END}\n"
    )
    with pytest.raises(ValueError, match="must appear exactly once"):
        apply_generated_state_blocks(
            text, artifacts_root=tmp_path / "artifacts", registry_root=None
        )


def test_readme_state_failures_empty_when_fresh(tmp_path: Path) -> None:
    artifacts_root = tmp_path / "artifacts"
    _write_active_model(artifacts_root)
    _write_opener_evaluation(artifacts_root)
    text = apply_generated_state_blocks(
        "# Project\n\nIntro.\n", artifacts_root=artifacts_root, registry_root=None
    )
    assert readme_state_failures(text, artifacts_root=artifacts_root, registry_root=None) == []


def test_readme_state_failures_detects_stale_active_model_block(tmp_path: Path) -> None:
    """The exact drift-detection requirement: a block that no longer matches
    what the current artifacts would produce must be flagged, so CI catches it."""

    artifacts_root = tmp_path / "artifacts"
    _write_active_model(artifacts_root, model_id="stale-id")
    text = apply_generated_state_blocks(
        "# Project\n\nIntro.\n", artifacts_root=artifacts_root, registry_root=None
    )

    _write_active_model(artifacts_root, model_id="new-id")
    failures = readme_state_failures(text, artifacts_root=artifacts_root, registry_root=None)
    assert any("active-model-state block is stale" in failure for failure in failures)


def test_readme_state_failures_detects_missing_block(tmp_path: Path) -> None:
    failures = readme_state_failures(
        "# Project\n\nNo generated blocks here.\n",
        artifacts_root=tmp_path / "artifacts",
        registry_root=None,
    )
    assert any("missing the generated active-model-state block" in failure for failure in failures)
    assert any("missing the generated research-state block" in failure for failure in failures)


def test_regenerate_readme_state_writes_only_when_changed(tmp_path: Path) -> None:
    artifacts_root = tmp_path / "artifacts"
    _write_active_model(artifacts_root, model_id="v1")
    readme_path = tmp_path / "README.md"
    readme_path.write_text("# Project\n\nIntro.\n", encoding="utf-8")

    first = regenerate_readme_state(artifacts_root, None, readme_path)
    assert first["changed"] is True
    assert "`v1`" in readme_path.read_text(encoding="utf-8")

    second = regenerate_readme_state(artifacts_root, None, readme_path)
    assert second["changed"] is False

    _write_active_model(artifacts_root, model_id="v2")
    third = regenerate_readme_state(artifacts_root, None, readme_path)
    assert third["changed"] is True
    assert "`v2`" in readme_path.read_text(encoding="utf-8")
    assert "`v1`" not in readme_path.read_text(encoding="utf-8")
