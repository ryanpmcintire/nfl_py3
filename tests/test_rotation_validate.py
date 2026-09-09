"""Tests for the ENG-27 rotation-registry validator (ROADMAP.md Phase 13).

``rotation.validate_registry`` is a full-audit pass, separate from the hard
``_validate`` gate that already runs at load/save time: it never raises, and
it returns every issue in one pass instead of stopping at the first one.
Three of its four checks (overlap, missing mined-acknowledgment) can never
actually fire on a registry that went through the strict loader -- `_validate`
already hard-refuses those -- so those two are tested against Registry
objects built directly from dataclasses, bypassing `_validate` on purpose, to
prove `validate_registry` is a complete standalone audit and not merely a
thin wrapper around the loader. The width and status checks CAN fire on a
loaded registry (the loader never checked either), so those are tested via
the normal JSON-payload path.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from nfl_ats import cli, rotation
from nfl_ats.rotation import (
    MAX_WINDOW_SIZE,
    MIN_WINDOW_SIZE,
    Family,
    Registry,
    Window,
    load_registry,
    registry_from_payload,
    validate_registry,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
LIVE_REGISTRY = REPO_ROOT / "registry" / "rotation_registry.json"


def _payload(**families: dict[str, Any]) -> dict[str, Any]:
    return {"version": rotation.ROTATION_REGISTRY_VERSION, "notes": [], "families": families}


def _family_payload(**overrides: Any) -> dict[str, Any]:
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


def _window_payload(**overrides: Any) -> dict[str, Any]:
    window: dict[str, Any] = {
        "seasons": [2011, 2013],
        "state": "assigned",
        "assigned_at": "2026-01-01",
        "spent_at": None,
        "artifact": None,
        "verdict": None,
        "probability_positive": None,
        "effect": None,
        "effect_units": None,
        "interval": None,
        "standard_error": None,
        "sample_blocks": None,
        "notes": "",
    }
    window.update(overrides)
    return window


def test_validate_flags_contiguous_window_wider_than_assign_window_limit() -> None:
    start = 2011
    too_wide_end = start + MAX_WINDOW_SIZE
    just_right_end = start + MAX_WINDOW_SIZE - 1
    too_narrow_end = start + MIN_WINDOW_SIZE - 2

    registry = registry_from_payload(
        _payload(
            too_wide=_family_payload(
                windows=[
                    _window_payload(
                        seasons=[start, too_wide_end],
                        state="spent",
                        spent_at="2026-01-02",
                        artifact="docs/too_wide.md",
                        verdict="unresolved",
                        probability_positive=0.5,
                    )
                ]
            ),
            just_right=_family_payload(
                windows=[
                    _window_payload(
                        seasons=[start, just_right_end],
                        state="spent",
                        spent_at="2026-01-02",
                        artifact="docs/just_right.md",
                        verdict="unresolved",
                        probability_positive=0.5,
                    )
                ]
            ),
            too_narrow=_family_payload(
                grade="nflverse_spread",
                windows=[
                    _window_payload(
                        seasons=[start, too_narrow_end],
                        state="spent",
                        spent_at="2026-01-02",
                        artifact="docs/too_narrow.md",
                        verdict="unresolved",
                        probability_positive=0.5,
                    )
                ],
            ),
        )
    )
    issues = validate_registry(registry)
    by_family = {issue.family: issue for issue in issues}

    assert by_family["too_wide"].severity == "error"
    assert by_family["too_wide"].code == "window_width_out_of_range"
    assert f"{MAX_WINDOW_SIZE + 1} season" in by_family["too_wide"].message

    assert by_family["too_narrow"].severity == "error"
    assert by_family["too_narrow"].code == "window_width_out_of_range"

    assert "just_right" not in by_family


def test_grandfather_exception_requires_the_exact_seasons_match() -> None:
    registry = registry_from_payload(
        _payload(
            pbp_drive_bundle=_family_payload(
                windows=[
                    _window_payload(
                        seasons=[2011, 2015],
                        state="spent",
                        assigned_at="2026-01-01",
                        spent_at="2026-01-02",
                        artifact="a.md",
                        verdict="unresolved",
                        probability_positive=0.5,
                    )
                ]
            ),
        )
    )
    issues = validate_registry(registry)
    width_issues = {i.family: i for i in issues if i.code == "window_width_out_of_range"}
    assert width_issues["pbp_drive_bundle"].severity == "error"


def test_grandfather_exception_never_applies_to_a_window_assigned_after_the_validator() -> None:
    from nfl_ats.rotation import GRANDFATHERED_WIDTH_VIOLATIONS, VALIDATOR_INTRODUCED_AT

    family_name = "pbp_drive_bundle"
    seasons = list(GRANDFATHERED_WIDTH_VIOLATIONS[family_name])
    registry = registry_from_payload(
        _payload(
            **{
                family_name: _family_payload(
                    windows=[
                        _window_payload(
                            seasons=seasons,
                            state="spent",
                            assigned_at=VALIDATOR_INTRODUCED_AT,
                            spent_at=VALIDATOR_INTRODUCED_AT,
                            artifact="a.md",
                            verdict="unresolved",
                            probability_positive=0.5,
                        )
                    ]
                )
            }
        )
    )
    issues = validate_registry(registry)
    width_issues = {i.family: i for i in issues if i.code == "window_width_out_of_range"}
    assert width_issues[family_name].severity == "error"


def test_grandfather_exception_downgrades_only_the_named_family_and_seasons() -> None:
    from nfl_ats.rotation import GRANDFATHERED_WIDTH_VIOLATIONS

    family_name = "pbp_drive_bundle"
    seasons = list(GRANDFATHERED_WIDTH_VIOLATIONS[family_name])
    families = {
        family_name: _family_payload(
            windows=[
                _window_payload(
                    seasons=seasons,
                    state="spent",
                    assigned_at="2026-08-13",
                    spent_at="2026-08-13",
                    artifact="a.md",
                    verdict="unresolved",
                    probability_positive=0.5,
                )
            ]
        ),
        "unrelated_wide_family": _family_payload(
            windows=[
                _window_payload(
                    seasons=[2011, 2015],
                    state="spent",
                    assigned_at="2026-01-01",
                    spent_at="2026-01-02",
                    artifact="b.md",
                    verdict="unresolved",
                    probability_positive=0.5,
                )
            ]
        ),
    }
    registry = registry_from_payload(_payload(**families))
    issues = validate_registry(registry)
    by_family = {i.family: i for i in issues if i.code == "window_width_out_of_range"}
    assert by_family[family_name].severity == "warning"
    assert "grandfathered" in by_family[family_name].message.lower()
    assert by_family["unrelated_wide_family"].severity == "error"


def test_validate_flags_overlapping_windows_bypassing_the_hard_loader() -> None:
    with pytest.raises(rotation.RegistryError, match="overlapping windows"):
        registry_from_payload(
            _payload(
                overlapper=_family_payload(
                    windows=[
                        _window_payload(
                            seasons=[2011, 2013],
                            state="spent",
                            spent_at="2026-01-02",
                            artifact="a.md",
                            verdict="unresolved",
                            probability_positive=0.5,
                        ),
                        _window_payload(
                            seasons=[2012, 2014],
                            state="spent",
                            spent_at="2026-01-02",
                            artifact="b.md",
                            verdict="unresolved",
                            probability_positive=0.5,
                        ),
                    ]
                )
            )
        )

    family = Family(
        name="overlapper",
        declared_at="2026-01-01",
        description="bypasses _validate on purpose",
        grade="close",
        status="open",
        windows=(
            Window(
                seasons=(2011, 2013),
                state="spent",
                assigned_at="2026-01-01",
                spent_at="2026-01-02",
                artifact="a.md",
                verdict="unresolved",
                probability_positive=0.5,
            ),
            Window(
                seasons=(2012, 2014),
                state="spent",
                assigned_at="2026-01-01",
                spent_at="2026-01-02",
                artifact="b.md",
                verdict="unresolved",
                probability_positive=0.5,
            ),
        ),
    )
    registry = Registry(version=1, notes=(), families={"overlapper": family})

    issues = validate_registry(registry)
    matches = [i for i in issues if i.code == "overlapping_windows_within_family"]
    assert len(matches) == 1
    assert matches[0].severity == "error"
    assert matches[0].family == "overlapper"


def test_validate_flags_missing_mined_acknowledgment_bypassing_the_hard_loader() -> None:
    with pytest.raises(rotation.RegistryError, match="acknowledges_mined_2018_2025"):
        registry_from_payload(
            _payload(
                unacknowledged=_family_payload(
                    acknowledges_mined_2018_2025=False,
                    windows=[
                        _window_payload(
                            seasons=[2020, 2021],
                            state="spent",
                            spent_at="2026-01-02",
                            artifact="a.md",
                            verdict="unresolved",
                            probability_positive=0.5,
                        )
                    ],
                )
            )
        )

    family = Family(
        name="unacknowledged",
        declared_at="2026-01-01",
        description="bypasses _validate on purpose",
        grade="close",
        status="open",
        acknowledges_mined_2018_2025=False,
        windows=(
            Window(
                seasons=(2020, 2021),
                state="spent",
                assigned_at="2026-01-01",
                spent_at="2026-01-02",
                artifact="a.md",
                verdict="unresolved",
                probability_positive=0.5,
            ),
        ),
    )
    registry = Registry(version=1, notes=(), families={"unacknowledged": family})

    issues = validate_registry(registry)
    matches = [i for i in issues if i.code == "missing_mined_acknowledgment"]
    assert len(matches) == 1
    assert matches[0].severity == "error"
    assert matches[0].family == "unacknowledged"


def test_validate_flags_terminal_status_with_no_spent_window() -> None:
    registry = registry_from_payload(
        _payload(
            hollow_closure=_family_payload(status="closed_negative", windows=[]),
            legit_open=_family_payload(status="open", windows=[]),
        )
    )
    issues = validate_registry(registry)
    by_family = {issue.family: issue for issue in issues}

    assert by_family["hollow_closure"].severity == "warning"
    assert by_family["hollow_closure"].code == "status_look_with_no_window"
    assert "legit_open" not in by_family


def test_validate_returns_no_issues_for_a_clean_registry() -> None:
    registry = registry_from_payload(
        _payload(
            clean=_family_payload(
                acknowledges_mined_2018_2025=True,
                windows=[
                    _window_payload(
                        seasons=[2020, 2021],
                        state="spent",
                        spent_at="2026-01-02",
                        artifact="a.md",
                        verdict="unresolved",
                        probability_positive=0.5,
                    )
                ],
            )
        )
    )
    assert validate_registry(registry) == []


def test_validate_against_the_live_registry_finds_pbp_drive_bundle_and_writes_nothing() -> None:
    before = LIVE_REGISTRY.read_bytes()
    registry = load_registry(LIVE_REGISTRY)
    issues = validate_registry(registry)

    width_issues = {i.family: i for i in issues if i.code == "window_width_out_of_range"}
    assert "pbp_drive_bundle" in width_issues
    assert width_issues["pbp_drive_bundle"].severity == "warning"
    assert "grandfathered" in width_issues["pbp_drive_bundle"].message.lower()
    assert not any(issue.severity == "error" for issue in issues)

    fluview = registry.families["fluview_elevated_on_production"]
    assert fluview.windows[0].window_kind == "stratified"
    assert "fluview_elevated_on_production" not in width_issues

    assert LIVE_REGISTRY.read_bytes() == before


def test_validate_registry_is_json_serializable() -> None:
    registry = load_registry(LIVE_REGISTRY)
    issues = validate_registry(registry)
    payload = [
        {"severity": i.severity, "code": i.code, "family": i.family, "message": i.message}
        for i in issues
    ]
    json.dumps(payload)


def test_save_registry_warns_but_does_not_refuse_a_width_violation(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    registry = registry_from_payload(
        _payload(
            too_wide=_family_payload(
                windows=[
                    _window_payload(
                        seasons=[2009, 2017],
                        state="spent",
                        spent_at="2026-01-02",
                        artifact="a.md",
                        verdict="unresolved",
                        probability_positive=0.5,
                    )
                ]
            )
        )
    )
    destination = tmp_path / "rotation_registry.json"
    rotation.save_registry(registry, destination)
    assert destination.is_file()
    stderr = capsys.readouterr().err
    assert "window_width_out_of_range" in stderr
    assert "too_wide" in stderr

    reloaded = load_registry(destination)
    assert reloaded.families["too_wide"].windows[0].seasons == (2009, 2017)


def test_cli_rotation_validate_exits_nonzero_on_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    registry_dir = tmp_path / "registry"
    registry_dir.mkdir()
    (registry_dir / "rotation_registry.json").write_text(
        json.dumps(
            _payload(
                too_wide=_family_payload(
                    windows=[
                        _window_payload(
                            seasons=[2009, 2017],
                            state="spent",
                            spent_at="2026-01-02",
                            artifact="a.md",
                            verdict="unresolved",
                            probability_positive=0.5,
                        )
                    ]
                )
            )
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("NFL_ATS_REGISTRY_DIR", str(registry_dir))

    with pytest.raises(SystemExit) as excinfo:
        cli.main(["rotation", "validate", "--json"])
    assert excinfo.value.code == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["error_count"] == 1
    assert payload["issues"][0]["code"] == "window_width_out_of_range"

    assert (registry_dir / "rotation_registry.json").is_file()


def test_cli_rotation_validate_clean_registry_exits_zero(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    registry_dir = tmp_path / "registry"
    registry_dir.mkdir()
    (registry_dir / "rotation_registry.json").write_text(
        json.dumps(_payload(clean=_family_payload())), encoding="utf-8"
    )
    monkeypatch.setenv("NFL_ATS_REGISTRY_DIR", str(registry_dir))

    assert cli.main(["rotation", "validate"]) == 0
    text = capsys.readouterr().out
    assert "no issues found" in text


def test_cli_rotation_validate_exits_zero_on_the_live_registry(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """ROADMAP.md ENG-37 (2026-09-05): the tracked registry's only validator

    error (``pbp_drive_bundle``'s pre-validator width) is now grandfathered
    to a warning, so ``nfl-ats rotation validate`` exits 0 against the real
    tracked ledger. Points ``NFL_ATS_REGISTRY_DIR`` at the repo's own
    ``registry/`` directory by absolute path (matching ``LIVE_REGISTRY``
    above) rather than relying on the test process's cwd.
    """

    monkeypatch.setenv("NFL_ATS_REGISTRY_DIR", str(REPO_ROOT / "registry"))
    assert cli.main(["rotation", "validate", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["error_count"] == 0
    assert payload["warning_count"] >= 1
    warning_codes = {issue["code"] for issue in payload["issues"] if issue["severity"] == "warning"}
    assert "window_width_out_of_range" in warning_codes


@pytest.mark.parametrize(
    "summary", ["", " ", "Fragment", "Missing punctuation", 42, ["Two words."]]
)
def test_audit_and_save_reject_invalid_plain_summary(summary: Any, tmp_path: Path) -> None:
    family = Family(
        name="example",
        declared_at="2026-09-05",
        description="Research notes",
        grade="close",
        status="open",
        plain_summary=summary,
    )
    registry = Registry(
        version=rotation.ROTATION_REGISTRY_VERSION, notes=(), families={"example": family}
    )
    issues = validate_registry(registry)
    assert [(issue.code, issue.severity, issue.family) for issue in issues] == [
        ("invalid_plain_summary", "error", "example")
    ]
    with pytest.raises(rotation.RegistryError, match="plain_summary"):
        rotation.save_registry(registry, tmp_path / "invalid.json")
    assert not (tmp_path / "invalid.json").exists()


@pytest.mark.parametrize(
    "summary", [None, "When division rivals meet, this rule backs the underdog."]
)
def test_audit_accepts_optional_plain_summary(summary: str | None) -> None:
    registry = registry_from_payload(_payload(example=_family_payload(plain_summary=summary)))
    assert validate_registry(registry) == []
