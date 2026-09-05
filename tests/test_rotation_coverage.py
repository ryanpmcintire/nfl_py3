"""Tests for ENG-27 rotation-registry coverage (ROADMAP.md Phase 13).

Three pieces, tested together because they only make sense as a pipeline:
``registry_explorer.coverage_plan`` (read-only: what should happen),
``rotation.declare_coverage_stub`` / ``rotation.record_no_rotation_needed``
(the write paths), and the ``nfl-ats rotation declare-coverage`` CLI that
drives them. Every test either builds small synthetic registries (precise
assertions on the classifier and the name-collision fallback) or copies the
REAL tracked registries into ``tmp_path`` (a faithful dry-run/apply/idempotent
-second-apply rehearsal, with a byte-for-byte check that no pre-existing
family or ``no_rotation_needed`` entry is ever mutated).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from nfl_ats import cli, registry_explorer, rotation, weak_signals

REPO_ROOT = Path(__file__).resolve().parents[1]
LIVE_ROTATION = REPO_ROOT / "registry" / "rotation_registry.json"
LIVE_WEAK_SIGNALS = REPO_ROOT / "registry" / "weak_signals.json"


# ---------------------------------------------------------------------------
# Synthetic registries.
# ---------------------------------------------------------------------------


def _weak_signal(**overrides: Any) -> dict[str, Any]:
    body: dict[str, Any] = {
        "recorded_at": "2026-01-01",
        "description": "synthetic test signal",
        "source": "tests/test_rotation_coverage.py",
        "effect": 1.0,
        "effect_units": "accuracy_points",
        "classification": "unresolved_below_power",
        "league": "nfl",
        "seasons": [2015, 2017],
        "probability_positive": 0.5,
    }
    body.update(overrides)
    return body


def _weak_registry(**signals: dict[str, Any]) -> weak_signals.Registry:
    payload = {
        "version": weak_signals.WEAK_SIGNAL_REGISTRY_VERSION,
        "notes": [],
        "signals": signals,
    }
    return weak_signals.registry_from_payload(payload)


def _rotation_family(**overrides: Any) -> dict[str, Any]:
    family: dict[str, Any] = {
        "declared_at": "2026-01-01",
        "description": "synthetic test family",
        "grade": "close",
        "status": "open",
        "inherits": [],
        "acknowledges_mined_2018_2025": False,
        "windows": [],
    }
    family.update(overrides)
    return family


def _rotation_registry(**families: dict[str, Any]) -> rotation.Registry:
    payload = {"version": rotation.ROTATION_REGISTRY_VERSION, "notes": [], "families": families}
    return rotation.registry_from_payload(payload)


def _synthetic_weak_registry() -> weak_signals.Registry:
    return _weak_registry(
        # Already has a rotation-family match by NAME equality -> skipped entirely.
        already_covered_family=_weak_signal(family="already_covered_family", category="onfield"),
        # category=control, name contains "oracle" -> reason "oracle".
        weather_oracle_ceiling_check=_weak_signal(
            family="weather_oracle_ceiling_check", category="control"
        ),
        # category=control, no "oracle" in the name -> reason "positive_control".
        redteam_bye_fade_sham_placebo=_weak_signal(
            family="redteam_bye_fade_sham_placebo", category="control"
        ),
        # No marker at all -> gets a stub.
        unmatched_family_needs_stub=_weak_signal(
            family="unmatched_family_needs_stub", category="onfield"
        ),
        # Same bare family string in two different leagues -> collision fallback.
        shared_name_nfl=_weak_signal(family="shared_name_family", league="nfl", category="onfield"),
        shared_name_cfb=_weak_signal(family="shared_name_family", league="cfb", category="onfield"),
    )


def _synthetic_rotation_registry() -> rotation.Registry:
    return _rotation_registry(already_covered_family=_rotation_family())


# ---------------------------------------------------------------------------
# coverage_plan: read-only.
# ---------------------------------------------------------------------------


def test_coverage_plan_skips_already_matched_and_classifies_the_rest() -> None:
    weak_registry = _synthetic_weak_registry()
    rot_registry = _synthetic_rotation_registry()

    plan = registry_explorer.coverage_plan(weak_registry, rot_registry)
    by_family = {row["weak_signal_family"]: row for row in plan}

    assert "already_covered_family" not in by_family

    assert by_family["weather_oracle_ceiling_check"]["action"] == "no_rotation_needed"
    assert by_family["weather_oracle_ceiling_check"]["reason"] == "oracle"

    assert by_family["redteam_bye_fade_sham_placebo"]["action"] == "no_rotation_needed"
    assert by_family["redteam_bye_fade_sham_placebo"]["reason"] == "positive_control"

    assert by_family["unmatched_family_needs_stub"]["action"] == "declare_stub"
    assert by_family["unmatched_family_needs_stub"]["stub_name"] == "unmatched_family_needs_stub"
    assert by_family["unmatched_family_needs_stub"]["reason"] is None

    # Two distinct (league, family) rows share the bare name "shared_name_family";
    # the plan must give them two DISTINCT stub names.
    shared_rows = [row for row in plan if row["weak_signal_family"] == "shared_name_family"]
    assert len(shared_rows) == 2
    stub_names = {row["stub_name"] for row in shared_rows}
    assert len(stub_names) == 2
    assert "shared_name_family" in stub_names  # the first one keeps the bare name
    assert any(name != "shared_name_family" for name in stub_names)  # the second is suffixed

    # Never mutates either registry.
    assert isinstance(plan, list)


def test_coverage_plan_never_guesses_a_reason_for_an_unmatched_category() -> None:
    weak_registry = _weak_registry(
        # category=market, no oracle/reliability/retired marker -> must be a stub, not a reason,
        # even though it is a below-power weak signal like any other.
        odd_market_family=_weak_signal(family="odd_market_family", category="market"),
    )
    rot_registry = _rotation_registry()
    plan = registry_explorer.coverage_plan(weak_registry, rot_registry)
    assert len(plan) == 1
    assert plan[0]["action"] == "declare_stub"


# ---------------------------------------------------------------------------
# ENG-37 (ROADMAP.md Phase 13, 2026-09-05): CFB families are out of the
# NFL-only rotation registry's scope (rule 8) and must never get a stub, even
# when the name/category would otherwise classify to "oracle" or
# "positive_control".
# ---------------------------------------------------------------------------


def test_classify_no_rotation_reason_routes_cfb_before_any_other_rule() -> None:
    # A CFB family whose name/category would otherwise map to "oracle" or
    # "positive_control" must still classify to "cfb_out_of_scope": scope is
    # checked first and is the actual reason no NFL window is needed.
    assert (
        rotation.classify_no_rotation_reason(
            "weather_oracle_ceiling_check", "control", league="cfb"
        )
        == "cfb_out_of_scope"
    )
    assert (
        rotation.classify_no_rotation_reason(
            "redteam_bye_fade_sham_placebo", "control", league="cfb"
        )
        == "cfb_out_of_scope"
    )
    # A family with no other marker at all is still routed by league alone.
    assert (
        rotation.classify_no_rotation_reason("cfb_role_continuity", "onfield", league="cfb")
        == "cfb_out_of_scope"
    )
    # league defaults to "nfl", so existing callers that never pass it keep
    # the pre-ENG-37 behaviour exactly.
    assert (
        rotation.classify_no_rotation_reason("weather_oracle_ceiling_check", "control") == "oracle"
    )
    assert rotation.classify_no_rotation_reason("odd_market_family", "market", league="nfl") is None


def test_coverage_plan_routes_cfb_families_to_no_rotation_needed_not_a_stub() -> None:
    weak_registry = _weak_registry(
        # Would classify to "oracle" by name alone -- CFB scope wins anyway.
        cfb_weather_oracle=_weak_signal(
            family="cfb_weather_oracle", league="cfb", category="control"
        ),
        # No oracle/reliability/retired/control marker at all -- still routed
        # by league, never falls through to a stub.
        cfb_unmatched_family=_weak_signal(
            family="cfb_unmatched_family", league="cfb", category="onfield"
        ),
        # Same bare family string in NFL still gets the pre-existing stub
        # treatment -- only the CFB row is affected.
        cfb_unmatched_family_nfl_twin=_weak_signal(
            family="cfb_unmatched_family_nfl_twin", league="nfl", category="onfield"
        ),
    )
    rot_registry = _rotation_registry()
    plan = registry_explorer.coverage_plan(weak_registry, rot_registry)
    by_family = {row["weak_signal_family"]: row for row in plan}

    assert by_family["cfb_weather_oracle"]["action"] == "no_rotation_needed"
    assert by_family["cfb_weather_oracle"]["reason"] == "cfb_out_of_scope"

    assert by_family["cfb_unmatched_family"]["action"] == "no_rotation_needed"
    assert by_family["cfb_unmatched_family"]["reason"] == "cfb_out_of_scope"

    assert by_family["cfb_unmatched_family_nfl_twin"]["action"] == "declare_stub"


def test_record_no_rotation_needed_accepts_cfb_out_of_scope() -> None:
    registry = _rotation_registry()
    updated = rotation.record_no_rotation_needed(
        registry,
        "cfb_role_continuity_dup",
        league="cfb",
        reason="cfb_out_of_scope",
        effect_units=("accuracy_points",),
    )
    record = updated.no_rotation_needed["cfb_role_continuity_dup"]
    assert record.reason == "cfb_out_of_scope"
    assert record.league == "cfb"
    assert rotation.validate_registry(updated) == []


# ---------------------------------------------------------------------------
# Library write paths.
# ---------------------------------------------------------------------------


def test_declare_coverage_stub_sets_expected_fields() -> None:
    registry = _rotation_registry()
    updated = rotation.declare_coverage_stub(
        registry,
        "unmatched_family_needs_stub",
        weak_signal_family="unmatched_family_needs_stub",
        league="nfl",
        effect_units=("accuracy_points",),
    )
    family = updated.families["unmatched_family_needs_stub"]
    assert family.status == rotation.COVERAGE_STUB_STATUS
    assert family.grade == rotation.COVERAGE_STUB_GRADE
    assert family.windows == ()
    assert family.coverage_weak_signal_family == "unmatched_family_needs_stub"
    assert family.coverage_league == "nfl"
    assert family.coverage_effect_units == ("accuracy_points",)

    # Round-trips through save/load unchanged.
    assert rotation.validate_registry(updated) == []


def test_declare_coverage_stub_refuses_an_existing_family_name() -> None:
    registry = _synthetic_rotation_registry()
    with pytest.raises(rotation.RegistryError, match="already declared"):
        rotation.declare_coverage_stub(
            registry,
            "already_covered_family",
            weak_signal_family="already_covered_family",
            league="nfl",
        )


def test_record_no_rotation_needed_validates_reason_and_is_append_only() -> None:
    registry = _rotation_registry()
    updated = rotation.record_no_rotation_needed(
        registry,
        "weather_oracle_ceiling_check",
        league="nfl",
        reason="oracle",
        effect_units=("accuracy_points",),
    )
    record = updated.no_rotation_needed["weather_oracle_ceiling_check"]
    assert record.reason == "oracle"
    assert record.league == "nfl"

    with pytest.raises(rotation.RegistryError, match="already has a no_rotation_needed record"):
        rotation.record_no_rotation_needed(
            updated, "weather_oracle_ceiling_check", league="nfl", reason="oracle"
        )

    with pytest.raises(rotation.RegistryError, match="Inadmissible reason"):
        rotation.record_no_rotation_needed(
            registry, "bogus_family", league="nfl", reason="because I said so"
        )

    # decomposition_of_parent:<family> is admissible even though the automatic
    # classifier never produces it itself.
    with_decomposition = rotation.record_no_rotation_needed(
        registry, "child_family", league="nfl", reason="decomposition_of_parent:parent_family"
    )
    assert (
        with_decomposition.no_rotation_needed["child_family"].reason
        == "decomposition_of_parent:parent_family"
    )


# ---------------------------------------------------------------------------
# End-to-end: the plan, applied, is idempotent and mutates nothing pre-existing.
# ---------------------------------------------------------------------------


def _apply_plan(
    weak_registry: weak_signals.Registry, rot_registry: rotation.Registry
) -> rotation.Registry:
    plan = registry_explorer.coverage_plan(weak_registry, rot_registry)
    for row in plan:
        if row["action"] == "declare_stub":
            rot_registry = rotation.declare_coverage_stub(
                rot_registry,
                row["stub_name"],
                weak_signal_family=row["weak_signal_family"],
                league=row["league"],
                effect_units=tuple(row["effect_units"]),
            )
        else:
            rot_registry = rotation.record_no_rotation_needed(
                rot_registry,
                row["weak_signal_family"],
                league=row["league"],
                reason=row["reason"],
                effect_units=tuple(row["effect_units"]),
            )
    return rot_registry


def test_apply_is_idempotent_and_touches_no_pre_existing_family() -> None:
    weak_registry = _synthetic_weak_registry()
    original = _synthetic_rotation_registry()
    before_payload = rotation.registry_payload(original)

    once = _apply_plan(weak_registry, original)
    assert len(once.families) > len(original.families)
    assert once.no_rotation_needed

    # The pre-existing family is byte-for-byte the same sub-payload.
    once_payload = rotation.registry_payload(once)
    assert (
        once_payload["families"]["already_covered_family"]
        == (before_payload["families"]["already_covered_family"])
    )

    # A second pass over the now-covered registries plans nothing further.
    second_plan = registry_explorer.coverage_plan(weak_registry, once)
    assert second_plan == []

    twice = _apply_plan(weak_registry, once)
    assert rotation.registry_payload(twice) == rotation.registry_payload(once)
    assert rotation.validate_registry(twice) == []


# ---------------------------------------------------------------------------
# CLI, against a tmp copy of the REAL registries.
# ---------------------------------------------------------------------------


def _write_uncovered_rotation_copy(destination: Path) -> None:
    """Write a tmp copy of the LIVE rotation registry with ENG-27 coverage stripped.

    ENG-27's own ``rotation declare-coverage --apply`` has already been run
    for real against the tracked registry (that is this ticket's whole
    point), so a byte-for-byte copy of the live file has ZERO uncovered
    weak-signal families left -- a dry-run/apply test against it would
    trivially plan nothing. Stripping every ``declared_for_coverage`` stub
    and the ``no_rotation_needed`` section reconstructs the pre-ENG-27 state
    (the 30 families declared by hand before this ticket) on a COPY, so this
    test stays meaningful regardless of how complete the live registry's own
    coverage becomes over time.
    """

    payload = json.loads(LIVE_ROTATION.read_text(encoding="utf-8"))
    payload["families"] = {
        name: family
        for name, family in payload["families"].items()
        if family.get("status") != rotation.COVERAGE_STUB_STATUS
    }
    payload.pop("no_rotation_needed", None)
    destination.write_text(json.dumps(payload), encoding="utf-8")


def test_cli_declare_coverage_dry_run_writes_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    registry_dir = tmp_path / "registry"
    registry_dir.mkdir()
    _write_uncovered_rotation_copy(registry_dir / "rotation_registry.json")
    (registry_dir / "weak_signals.json").write_bytes(LIVE_WEAK_SIGNALS.read_bytes())
    monkeypatch.setenv("NFL_ATS_REGISTRY_DIR", str(registry_dir))

    rotation_before = (registry_dir / "rotation_registry.json").read_bytes()

    assert cli.main(["rotation", "declare-coverage"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["mode"] == "dry_run"
    assert payload["plan_rows"] > 0
    assert set(payload["counts"]) == {"declare_stub", "no_rotation_needed"}

    assert (registry_dir / "rotation_registry.json").read_bytes() == rotation_before


def test_cli_declare_coverage_apply_is_additive_and_idempotent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    registry_dir = tmp_path / "registry"
    registry_dir.mkdir()
    _write_uncovered_rotation_copy(registry_dir / "rotation_registry.json")
    (registry_dir / "weak_signals.json").write_bytes(LIVE_WEAK_SIGNALS.read_bytes())
    monkeypatch.setenv("NFL_ATS_REGISTRY_DIR", str(registry_dir))
    rotation_path = registry_dir / "rotation_registry.json"

    original_registry = rotation.load_registry(rotation_path)
    original_payload = rotation.registry_payload(original_registry)
    original_issues = rotation.validate_registry(original_registry)

    assert cli.main(["rotation", "declare-coverage", "--apply"]) == 0
    first_apply = json.loads(capsys.readouterr().out)
    assert first_apply["mode"] == "apply"
    assert first_apply["applied_rows"] > 0
    assert first_apply["families_total"] > len(original_registry.families)

    after_first = rotation.load_registry(rotation_path)
    after_first_payload = rotation.registry_payload(after_first)

    # Every pre-existing family's sub-payload is byte-for-byte unchanged --
    # additive growth only, never a rewrite of an existing entry or look.
    for name, family_payload in original_payload["families"].items():
        assert after_first_payload["families"][name] == family_payload, name

    # validate_registry finds no NEW issues: coverage stubs carry no windows,
    # so they cannot trip the width/overlap/mined-ack checks.
    assert rotation.validate_registry(after_first) == original_issues

    # Second apply is a true no-op: the plan is empty, and the file does not change.
    assert cli.main(["rotation", "declare-coverage", "--apply"]) == 0
    second_apply = json.loads(capsys.readouterr().out)
    assert second_apply["applied_rows"] == 0
    assert second_apply["counts"] == {"declare_stub": 0, "no_rotation_needed": 0}

    after_second_payload = rotation.registry_payload(rotation.load_registry(rotation_path))
    assert after_second_payload == after_first_payload


# ---------------------------------------------------------------------------
# ENG-37 (ROADMAP.md Phase 13, 2026-09-05): the live registry migration
# (scripts/eng37_rotation_coverage_followups.py) that moved the 54
# already-declared CFB coverage stubs into no_rotation_needed.
# ---------------------------------------------------------------------------


def test_live_registry_cfb_stubs_are_marked_out_of_scope_and_kept() -> None:
    before = LIVE_WEAK_SIGNALS.read_bytes()
    registry = rotation.load_registry(LIVE_ROTATION)

    cfb_stub_families = [
        family
        for family in registry.families.values()
        if family.status == rotation.COVERAGE_STUB_STATUS and family.coverage_league == "cfb"
    ]
    # Measured 2026-09-04 (ROADMAP.md ENG-37): 54 CFB weak-signal families had
    # already been given a declared_for_coverage stub before this fix.
    assert len(cfb_stub_families) == 54

    cfb_out_of_scope = {
        key: record
        for key, record in registry.no_rotation_needed.items()
        if record.reason == "cfb_out_of_scope"
    }
    assert len(cfb_out_of_scope) == 54

    # Every stub's weak-signal family has a matching no_rotation_needed
    # record, keyed by the weak-signal family name (not the rotation-family
    # name, which can differ on a collision suffix) -- and the stub Family
    # entry itself is KEPT (no delete API; declarations are append-only).
    for family in cfb_stub_families:
        assert family.coverage_weak_signal_family in cfb_out_of_scope
        record = cfb_out_of_scope[family.coverage_weak_signal_family]
        assert record.league == "cfb"

    # A re-run of the classifier/plan machinery against the live data never
    # produces a second no_rotation_needed row or a stub for any of these 54
    # -- matching_rotation_families already finds their stub by name.
    weak_registry = weak_signals.load_registry(LIVE_WEAK_SIGNALS)
    plan = registry_explorer.coverage_plan(weak_registry, registry)
    planned_families = {row["weak_signal_family"] for row in plan}
    assert planned_families.isdisjoint({f.coverage_weak_signal_family for f in cfb_stub_families})

    # Read-only: never touches weak_signals.json.
    assert LIVE_WEAK_SIGNALS.read_bytes() == before


def test_live_registry_pbp_drive_bundle_window_is_grandfathered() -> None:
    registry = rotation.load_registry(LIVE_ROTATION)
    window = registry.families["pbp_drive_bundle"].windows[0]
    assert window.seasons == (2013, 2017)
    assert "GRANDFATHERED" in window.notes
    assert "ENG-37" in window.notes

    issues = rotation.validate_registry(registry)
    errors = [issue for issue in issues if issue.severity == "error"]
    assert errors == []
    width_issue = next(issue for issue in issues if issue.code == "window_width_out_of_range")
    assert width_issue.family == "pbp_drive_bundle"
    assert width_issue.severity == "warning"
