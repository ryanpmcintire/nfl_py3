from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from nfl_ats.model_ledger import (
    Agreement,
    LedgerError,
    ModelLedger,
    build_model_ledger,
    render_markdown_table,
    validate_ledger,
)


def _write_json(path: Path, payload: object) -> Path:
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _registry_payload() -> dict[str, object]:
    return {
        "signals": {
            "hc_year_one_fade": {
                "classification": "unresolved_below_power",
                "effect": 0.7528,
                "probability_positive": 0.932,
                "interval": [-0.1, 1.6],
            },
            "mod08_smooth_cdf_mapping": {
                "classification": "unresolved_below_power",
                "effect": 0.684,
                "probability_positive": 0.8666,
                "interval": [-0.444, 1.841],
            },
            "mod08_gamma_signal": {
                "classification": "unresolved_below_power",
                "effect": -0.2,
                "probability_positive": 0.4,
                "interval": [-0.9, 0.5],
            },
        }
    }


def _challengers_payload() -> dict[str, object]:
    return {
        "challengers": [
            {
                "challenger_id": "gamma_signal",
                "status": "ACTIVE_PROSPECTIVE",
                "evidence": {},
            },
            {
                "challenger_id": "smooth_cdf_mapping",
                "status": "SUPERSEDED_BY_PROMOTION",
                "evidence": {
                    "registry_source": ["registry/weak_signals.json:mod08_smooth_cdf_mapping"],
                    "probability_positive": 0.5536,
                },
            },
            {
                "challenger_id": "qb_continuity_arm",
                "status": "CLOSED_BEFORE_ACTIVATION",
                "evidence": {
                    "registry_source": (
                        "registry/weak_signals.json:nope_one (the latter NOT "
                        "the basis), registry/weak_signals.json:hc_year_one_fade"
                    ),
                    "sample_games": 456,
                    "candidate_accuracy_at_opener": 0.5329,
                    "week_blocked_interval_points": [-1.1, 5.0],
                },
            },
            {
                "challenger_id": "deactivated_arm",
                "status": "DEACTIVATED_STRUCTURAL_NO_OP",
                "evidence": {"write_up": "docs/whatever.md"},
            },
            {
                "challenger_id": "scratchpad_only_arm",
                "status": "ACTIVE_PROSPECTIVE",
                "evidence": {"registry_source": "scratchpad/bestpick_opener/results.md"},
            },
        ]
    }


def _manifest_payload() -> dict[str, object]:
    return {
        "model_id": "abc123def4567890",
        "feature_profile": "weak_stack",
        "method": "market_residual",
        "historical_evaluation": {
            "accuracy": 0.5209638554216868,
            "games": 2075,
            "artifact": "margins/20260820T004951Z",
            "intervals": {"season": {"lower": 0.5078765661351946, "upper": 0.5345932252330292}},
        },
    }


@pytest.fixture()
def ledger_paths(tmp_path: Path) -> tuple[Path, Path, Path]:
    weak = _write_json(tmp_path / "weak_signals.json", _registry_payload())
    challengers = _write_json(tmp_path / "challengers.json", _challengers_payload())
    manifest = _write_json(tmp_path / "active_ats_model.json", _manifest_payload())
    return challengers, weak, manifest


def test_status_derivation_maps_all_badges(ledger_paths: tuple[Path, Path, Path]) -> None:
    challengers, weak, manifest = ledger_paths
    ledger = build_model_ledger(challengers, weak, manifest)
    badges = {row.arm_id: row.status_badge for row in ledger.rows}
    assert badges["promoted:abc123def4567890"] == "PROMOTED"
    assert badges["smooth_cdf_mapping"] == "SUPERSEDED"
    assert badges["qb_continuity_arm"] == "RETIRED"
    assert badges["deactivated_arm"] == "RETIRED"
    assert badges["gamma_signal"] == "CHALLENGER"
    assert badges["scratchpad_only_arm"] == "CHALLENGER"


def test_promoted_row_sorts_first_and_rest_by_probability_desc(
    ledger_paths: tuple[Path, Path, Path],
) -> None:
    challengers, weak, manifest = ledger_paths
    ledger = build_model_ledger(challengers, weak, manifest)
    arm_ids = [row.arm_id for row in ledger.rows]
    assert arm_ids[0] == "promoted:abc123def4567890"
    challenger_order = arm_ids[1:]
    probabilities = {
        row.arm_id: max(
            (
                ref.probability_positive
                for ref in row.evidence
                if ref.probability_positive is not None
            ),
            default=float("-inf"),
        )
        for row in ledger.rows[1:]
    }
    ranked = sorted(challenger_order, key=lambda a: (-probabilities[a], a))
    assert challenger_order == ranked


def test_evidence_linkage_parses_prose_and_falls_back_on_naming_convention(
    ledger_paths: tuple[Path, Path, Path],
) -> None:
    challengers, weak, manifest = ledger_paths
    ledger = build_model_ledger(challengers, weak, manifest)
    by_arm = {row.arm_id: row for row in ledger.rows}

    qb_keys = [ref.registry_key for ref in by_arm["qb_continuity_arm"].evidence]
    assert qb_keys == ["hc_year_one_fade"]
    gamma_keys = [ref.registry_key for ref in by_arm["gamma_signal"].evidence]
    assert gamma_keys == ["mod08_gamma_signal"]
    scratch_keys = [ref.registry_key for ref in by_arm["scratchpad_only_arm"].evidence]
    assert scratch_keys == []
    promoted = by_arm["promoted:abc123def4567890"]
    assert promoted.evidence == ()
    ref = by_arm["qb_continuity_arm"].evidence[0]
    assert ref.effect == pytest.approx(0.7528)
    assert ref.classification == "unresolved_below_power"


def test_unknown_status_raises_at_build(ledger_paths: tuple[Path, Path, Path]) -> None:
    challengers, weak, manifest = ledger_paths
    payload = json.loads(Path(challengers).read_text(encoding="utf-8"))
    payload["challengers"][0]["status"] = "SOME_NEW_STATUS"
    broken = _write_json(Path(challengers).with_name("broken.json"), payload)
    with pytest.raises(LedgerError):
        build_model_ledger(broken, weak, manifest)


def test_validate_passes_on_a_fresh_ledger(
    ledger_paths: tuple[Path, Path, Path],
) -> None:
    challengers, weak, manifest = ledger_paths
    ledger = build_model_ledger(challengers, weak, manifest)
    validate_ledger(ledger)


def test_validate_rejects_promoted_row_not_matching_manifest(
    ledger_paths: tuple[Path, Path, Path],
) -> None:
    challengers, weak, manifest = ledger_paths
    ledger = build_model_ledger(challengers, weak, manifest)
    tampered_rows = []
    for row in ledger.rows:
        if row.status_badge == "PROMOTED":
            row = replace(row, arm_id="promoted:deadbeefdeadbeef")
        tampered_rows.append(row)
    rebuilt = ModelLedger(
        rows=tuple(tampered_rows),
        active_model_id=ledger.active_model_id,
        weak_signals_path=ledger.weak_signals_path,
    )
    with pytest.raises(LedgerError, match="arm_id"):
        validate_ledger(rebuilt)


def test_validate_rejects_registry_key_absent_from_live_registry(
    tmp_path: Path, ledger_paths: tuple[Path, Path, Path]
) -> None:
    challengers, weak, manifest = ledger_paths
    ledger = build_model_ledger(challengers, weak, manifest)
    row = next(r for r in ledger.rows if r.arm_id == "qb_continuity_arm")
    ghost = replace(row.evidence[0], registry_key="deleted_signal")
    others = [r for r in ledger.rows if r.arm_id != "qb_continuity_arm"]
    rebuilt = ModelLedger(
        rows=(*others, replace(row, evidence=(ghost,))),
        active_model_id=ledger.active_model_id,
        weak_signals_path=ledger.weak_signals_path,
    )
    with pytest.raises(LedgerError, match="does not exist"):
        validate_ledger(rebuilt)


def test_validate_rejects_stale_fingerprint_after_registry_move(
    tmp_path: Path, ledger_paths: tuple[Path, Path, Path]
) -> None:
    challengers, weak, manifest = ledger_paths
    ledger = build_model_ledger(challengers, weak, manifest)
    payload = json.loads(Path(weak).read_text(encoding="utf-8"))
    payload["signals"]["hc_year_one_fade"]["effect"] = 9.99
    _write_json(weak, payload)
    with pytest.raises(LedgerError, match="stale"):
        validate_ledger(ledger)


def test_validate_rejects_summary_number_absent_from_cited_fields(
    tmp_path: Path, ledger_paths: tuple[Path, Path, Path]
) -> None:
    challengers, weak, manifest = ledger_paths
    ledger = build_model_ledger(challengers, weak, manifest)
    tampered = []
    for row in ledger.rows:
        if row.arm_id == "scratchpad_only_arm":
            row = replace(row, summary_sentence=row.summary_sentence + " P+ 0.99.")
        tampered.append(row)
    rebuilt = ModelLedger(
        rows=tuple(tampered),
        active_model_id=ledger.active_model_id,
        weak_signals_path=ledger.weak_signals_path,
    )
    with pytest.raises(LedgerError, match="no cited field"):
        validate_ledger(rebuilt)


def test_agreement_populated_only_where_per_game_frames_supplied(
    ledger_paths: tuple[Path, Path, Path],
) -> None:
    challengers, weak, manifest = ledger_paths
    without = build_model_ledger(challengers, weak, manifest)
    assert all(row.agreement is None for row in without.rows)

    frames = {
        "promoted:abc123def4567890": {"g1": "home", "g2": "away", "g3": "home"},
        "gamma_signal": {"g1": "home", "g2": "home", "g3": "away"},
    }
    with_frames = build_model_ledger(challengers, weak, manifest, per_game_frames=frames)
    by_arm = {row.arm_id: row for row in with_frames.rows}
    agreement = by_arm["gamma_signal"].agreement
    assert isinstance(agreement, Agreement)
    assert agreement.vs_promoted_games == 3
    assert agreement.agree == 1
    assert agreement.disagree == 2
    assert by_arm["promoted:abc123def4567890"].agreement is None
    assert by_arm["qb_continuity_arm"].agreement is None


def test_markdown_round_trip_contains_all_arms_and_is_deterministic(
    ledger_paths: tuple[Path, Path, Path],
) -> None:
    challengers, weak, manifest = ledger_paths
    ledger = build_model_ledger(challengers, weak, manifest)
    first = render_markdown_table(ledger)
    second = render_markdown_table(build_model_ledger(challengers, weak, manifest))
    assert first == second
    for row in ledger.rows:
        assert row.display_name.replace("|", "\\|") in first
    assert "| PROMOTED |" in first


def test_real_artifacts_build_a_valid_ledger() -> None:
    challengers = Path("artifacts/prospective/challengers.json")
    weak = Path("registry/weak_signals.json")
    manifest = Path("artifacts/active_ats_model.json")
    if not (challengers.is_file() and weak.is_file() and manifest.is_file()):
        pytest.skip("live artifacts absent")
    ledger = build_model_ledger(challengers, weak, manifest)
    assert len(ledger.rows) == 25
    assert ledger.rows[0].status_badge == "PROMOTED"
    assert ledger.rows[0].track_record is not None
    assert ledger.rows[0].track_record.games == 2075
    superseded = [r for r in ledger.rows if r.status_badge == "SUPERSEDED"]
    assert len(superseded) == 4
    validate_ledger(ledger)
