from __future__ import annotations

from datetime import date, datetime
from pathlib import Path
from typing import Any

import pytest

from nfl_ats import rotation, weak_signals
from nfl_ats.constants import (
    DEFAULT_MIN_CALIBRATION_GAMES,
    DEFAULT_MIN_TRAIN_GAMES,
    MIN_FITTABLE_TRAIN_GAMES,
)
from nfl_ats.dashboard import findings_content
from nfl_ats.findings_registry import (
    CLOSED_ACTIVITY_BADGE_TEXT,
    STORE_CHALLENGER,
    STORE_ROTATION,
    STORE_WEAK_SIGNAL,
    CurationError,
    RecentRegistryActivity,
    RegistryEntry,
    WatchingLead,
    fingerprint,
    load_all_entries,
    load_rotation_registry,
    load_weak_signal_registry,
    recent_registry_activity,
    top_open_leads,
    validate_curation,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def _signal_payload(**overrides: Any) -> dict[str, Any]:
    body: dict[str, Any] = {
        "recorded_at": "2026-08-19",
        "description": "a small measured effect",
        "source": "docs/example.md",
        "effect": 0.5,
        "effect_units": "accuracy_points",
        "classification": "unresolved_below_power",
        "league": "nfl",
        "seasons": [2020, 2025],
        "probability_positive": 0.7,
        "interval": [-0.3, 1.3],
    }
    body.update(overrides)
    return body


def _weak_signal_registry(**signals: dict[str, Any]) -> weak_signals.Registry:
    payload = {
        "version": weak_signals.WEAK_SIGNAL_REGISTRY_VERSION,
        "notes": [],
        "signals": signals,
    }
    return weak_signals.registry_from_payload(payload)


class _Finding:
    def __init__(
        self,
        question: str,
        *,
        evergreen: bool = False,
        registry_keys: tuple[str, ...] = (),
        registry_fingerprints: tuple[str, ...] = (),
    ) -> None:
        self.question = question
        self.evergreen = evergreen
        self.registry_keys = registry_keys
        self.registry_fingerprints = registry_fingerprints


def test_fingerprint_is_stable_and_content_sensitive() -> None:
    payload = {"a": 1, "b": [1, 2, 3]}
    assert fingerprint(payload) == fingerprint(dict(payload))
    assert fingerprint(payload) != fingerprint({**payload, "a": 2})


def test_load_all_entries_covers_all_three_stores(tmp_path: Path) -> None:
    weak_signals.save_registry(
        _weak_signal_registry(alpha=_signal_payload()), tmp_path / "weak_signals.json"
    )
    family = rotation.Family(
        name="beta",
        declared_at="2026-08-19",
        description="a rotation family",
        grade="opener",
        status="open",
        acknowledges_mined_2018_2025=True,
        windows=(
            rotation.Window(
                seasons=(2020, 2021),
                state="spent",
                assigned_at="2026-08-19",
                spent_at="2026-08-19",
                artifact="docs/example.md",
                verdict="unresolved",
                effect=1.0,
                effect_units="accuracy_points",
                probability_positive=0.6,
            ),
        ),
    )
    rotation.save_registry(
        rotation.Registry(
            version=rotation.ROTATION_REGISTRY_VERSION, notes=(), families={"beta": family}
        ),
        tmp_path / "rotation_registry.json",
    )
    challengers = [
        {
            "challenger_id": "gamma",
            "status": "ACTIVE_PROSPECTIVE",
            "status_reason": "testing",
            "evidence": {"probability_positive": 0.8},
        }
    ]

    entries = load_all_entries(registry_root=tmp_path, challengers=challengers)
    assert set(entries) == {
        f"{STORE_WEAK_SIGNAL}:alpha",
        f"{STORE_ROTATION}:beta",
        f"{STORE_CHALLENGER}:gamma",
    }
    assert entries[f"{STORE_WEAK_SIGNAL}:alpha"].effect == 0.5
    assert entries[f"{STORE_ROTATION}:beta"].effect == 1.0
    assert entries[f"{STORE_CHALLENGER}:gamma"].probability_positive == 0.8


def test_load_rotation_registry_missing_file_is_empty(tmp_path: Path) -> None:
    registry = load_rotation_registry(tmp_path)
    assert registry.families == {}


def test_load_weak_signal_registry_missing_file_is_empty(tmp_path: Path) -> None:
    registry = load_weak_signal_registry(tmp_path)
    assert registry.signals == {}


def test_validate_curation_passes_a_correctly_cited_finding() -> None:
    entries = {
        "weak_signal:alpha": RegistryEntry(
            key="weak_signal:alpha",
            store=STORE_WEAK_SIGNAL,
            name="alpha",
            description="d",
            classification="unresolved_below_power",
            effect=0.5,
            effect_units="accuracy_points",
            probability_positive=0.7,
            interval=(-0.3, 1.3),
            seasons=(2020, 2025),
            league="nfl",
            recorded_at="2026-08-19",
            fingerprint="deadbeef",
        )
    }
    finding = _Finding(
        "q", registry_keys=("weak_signal:alpha",), registry_fingerprints=("deadbeef",)
    )
    validate_curation([finding], entries)


def test_validate_curation_rejects_a_nonexistent_key() -> None:
    finding = _Finding("q", registry_keys=("weak_signal:missing",), registry_fingerprints=("x",))
    with pytest.raises(CurationError, match="does not exist"):
        validate_curation([finding], {})


def test_validate_curation_rejects_a_stale_fingerprint() -> None:
    entries = {
        "weak_signal:alpha": RegistryEntry(
            key="weak_signal:alpha",
            store=STORE_WEAK_SIGNAL,
            name="alpha",
            description="d",
            classification="unresolved_below_power",
            effect=0.5,
            effect_units="accuracy_points",
            probability_positive=0.7,
            interval=(-0.3, 1.3),
            seasons=(2020, 2025),
            league="nfl",
            recorded_at="2026-08-19",
            fingerprint="the-live-value",
        )
    }
    finding = _Finding(
        "q", registry_keys=("weak_signal:alpha",), registry_fingerprints=("a-stale-value",)
    )
    with pytest.raises(CurationError, match="is stale against"):
        validate_curation([finding], entries)


def test_validate_curation_rejects_evergreen_with_keys() -> None:
    finding = _Finding("q", evergreen=True, registry_keys=("weak_signal:alpha",))
    with pytest.raises(CurationError, match="evergreen"):
        validate_curation([finding], {})


def test_validate_curation_rejects_non_evergreen_with_no_keys() -> None:
    finding = _Finding("q")
    with pytest.raises(CurationError, match="names no registry_keys"):
        validate_curation([finding], {})


def test_validate_curation_rejects_mismatched_key_and_fingerprint_counts() -> None:
    finding = _Finding(
        "q",
        registry_keys=("weak_signal:alpha", "weak_signal:beta"),
        registry_fingerprints=("only-one",),
    )
    with pytest.raises(CurationError, match="parallel tuples"):
        validate_curation([finding], {})


def test_a_registry_entry_changing_under_a_curation_entry_raises() -> None:

    original = _weak_signal_registry(alpha=_signal_payload(effect=0.5))
    from nfl_ats.findings_registry import _weak_signal_entry

    entry_before = _weak_signal_entry(original.signals["alpha"])
    finding = _Finding(
        "Does alpha help?",
        registry_keys=("weak_signal:alpha",),
        registry_fingerprints=(entry_before.fingerprint,),
    )
    validate_curation([finding], {"weak_signal:alpha": entry_before})

    corrected = _weak_signal_registry(alpha=_signal_payload(effect=0.9))
    entry_after = _weak_signal_entry(corrected.signals["alpha"])
    with pytest.raises(CurationError, match="is stale against"):
        validate_curation([finding], {"weak_signal:alpha": entry_after})


def test_real_findings_content_validates_against_the_tracked_registries() -> None:

    from nfl_ats.public_board import load_prospective_challengers

    challengers = load_prospective_challengers(REPO_ROOT / "artifacts")
    entries = load_all_entries(registry_root=REPO_ROOT / "registry", challengers=challengers)
    validate_curation((*findings_content.FINDINGS, *findings_content.LEAD_BLURBS), entries)


def test_recertified_findings_do_not_repeat_known_stale_claims() -> None:

    by_question = {finding.question: finding for finding in findings_content.FINDINGS}
    gap = by_question["Does it matter which line we are graded against?"]
    pool = by_question["Can a pile of weak signals add up to one strong one?"]
    best_pick = by_question[
        "The pool scores one Best Pick a week. Can we tell which of our picks is best?"
    ]
    weather = by_question["What can't we see?"]
    weather_context = by_question["Do rest, travel and weather matter?"]
    history = by_question["How much history does the model need before its picks are trustworthy?"]

    assert "paired difference and uncertainty" in gap.detail
    assert "1.35 points" not in gap.detail
    assert "143 recorded results" not in pool.detail
    assert "84 NFL" not in pool.detail
    assert "-0.003 accuracy points" not in pool.detail
    assert "decision-time forecast" in weather.plain_answer + weather_context.detail
    assert "needs an archived forecast source" not in weather_context.detail
    assert "below-median cross-book-disagreement pool" in best_pick.plain_answer
    assert "about +0.9 points" not in best_pick.plain_answer
    assert "500 finished games" not in history.plain_answer + history.detail
    assert str(MIN_FITTABLE_TRAIN_GAMES) in history.plain_answer
    assert str(DEFAULT_MIN_TRAIN_GAMES) in history.plain_answer
    assert str(DEFAULT_MIN_CALIBRATION_GAMES) in history.plain_answer


def test_every_non_evergreen_finding_names_at_least_one_key() -> None:
    for finding in findings_content.FINDINGS:
        if finding.evergreen:
            assert finding.registry_keys == ()
        else:
            assert finding.registry_keys, f"{finding.question!r} names no registry_keys"
            assert len(finding.registry_keys) == len(finding.registry_fingerprints)


def test_top_open_leads_renders_from_a_synthetic_fixture() -> None:
    registry = _weak_signal_registry(
        strong=_signal_payload(probability_positive=0.95, description="a strong lean"),
        weak=_signal_payload(probability_positive=0.51, description="a weak lean"),
    )
    leads = top_open_leads(registry)
    assert len(leads) == 2
    assert leads[0].name == "strong"
    assert isinstance(leads[0], WatchingLead)


def test_top_open_leads_excludes_terminal_classifications() -> None:
    registry = _weak_signal_registry(
        refuted=_signal_payload(
            classification="refuted_mechanism",
            closing_ground="wrong_sign_resolved",
            classification_evidence="whole interval below zero",
            interval=[-2.0, -1.0],
            effect=-1.5,
            probability_positive=0.01,
        ),
        open_lead=_signal_payload(probability_positive=0.8),
    )
    leads = top_open_leads(registry)
    assert [lead.name for lead in leads] == ["open_lead"]


def test_top_open_leads_excludes_instrument_checks() -> None:

    registry = _weak_signal_registry(
        reliability=_signal_payload(
            probability_positive=1.0, effect_units="correlation", description="split-half"
        ),
        control=_signal_payload(probability_positive=1.0, category="control"),
        open_lead=_signal_payload(probability_positive=0.8),
    )
    leads = top_open_leads(registry)
    assert [lead.name for lead in leads] == ["open_lead"]


def test_top_open_leads_collapses_close_and_opener_pair_to_the_opener_grade() -> None:
    registry = _weak_signal_registry(
        foo=_signal_payload(probability_positive=0.9, description="close-graded"),
        foo_opener=_signal_payload(probability_positive=0.6, description="opener-graded"),
    )
    leads = top_open_leads(registry)
    assert len(leads) == 1
    assert leads[0].name == "foo_opener"


def test_top_open_leads_collapses_a_battery_to_its_most_striking_cell() -> None:
    registry = _weak_signal_registry(
        bias_battery_alpha=_signal_payload(probability_positive=0.55, description="cell alpha"),
        bias_battery_beta=_signal_payload(probability_positive=0.9, description="cell beta"),
        bias_battery_gamma=_signal_payload(probability_positive=0.6, description="cell gamma"),
    )
    leads = top_open_leads(registry)
    assert len(leads) == 1
    assert leads[0].name == "bias_battery_beta"


def test_top_open_leads_excludes_oracle_measurements() -> None:
    registry = _weak_signal_registry(
        oracle_check=_signal_payload(
            probability_positive=0.999, description="a movement-oracle sanity check"
        ),
        real_lead=_signal_payload(probability_positive=0.8, description="a real candidate lead"),
    )
    leads = top_open_leads(registry)
    assert [lead.name for lead in leads] == ["real_lead"]


def test_top_open_leads_respects_limit() -> None:
    signals = {
        f"sig_{index}": _signal_payload(probability_positive=0.5 + index / 100)
        for index in range(20)
    }
    registry = _weak_signal_registry(**signals)
    leads = top_open_leads(registry, limit=5)
    assert len(leads) == 5


def test_top_open_leads_filters_by_league() -> None:
    registry = _weak_signal_registry(
        nfl_sig=_signal_payload(league="nfl", probability_positive=0.8),
        cfb_sig=_signal_payload(league="cfb", probability_positive=0.9),
    )
    leads = top_open_leads(registry, leagues=("nfl",))
    assert [lead.name for lead in leads] == ["nfl_sig"]


def _rotation_registry_with_window(name: str = "rot_family", **window_overrides: Any) -> Any:
    window_kwargs: dict[str, Any] = {
        "seasons": (2020, 2021),
        "state": "spent",
        "assigned_at": "2026-08-30",
        "spent_at": "2026-09-02",
        "verdict": "unresolved",
        "probability_positive": 0.62,
        "effect": 0.4,
        "effect_units": "accuracy_points",
    }
    window_kwargs.update(window_overrides)
    family = rotation.Family(
        name=name,
        declared_at="2026-08-01",
        description="a rotation family recorded this week",
        grade="opener",
        status="open",
        windows=(rotation.Window(**window_kwargs),),
    )
    return rotation.Registry(
        version=rotation.ROTATION_REGISTRY_VERSION, notes=(), families={name: family}
    )


def _empty_rotation_registry() -> Any:
    return rotation.Registry(version=rotation.ROTATION_REGISTRY_VERSION, notes=(), families={})


def test_recent_registry_activity_empty_window_renders_correctly() -> None:
    registry = _weak_signal_registry()
    activity = recent_registry_activity(
        registry, _empty_rotation_registry(), date(2026, 9, 5), days=7
    )
    assert isinstance(activity, RecentRegistryActivity)
    assert activity.is_empty
    assert activity.screened_count == 0
    assert activity.resolved_count == 0
    assert activity.entries_by_category == ()


def test_recent_registry_activity_includes_a_signal_recorded_inside_the_window() -> None:
    registry = _weak_signal_registry(
        recent=_signal_payload(recorded_at="2026-09-02", category="onfield")
    )
    activity = recent_registry_activity(
        registry, _empty_rotation_registry(), date(2026, 9, 5), days=7
    )
    assert activity.screened_count == 1
    assert activity.resolved_count == 0
    [(category, entries)] = activity.entries_by_category
    assert category == "onfield"
    entry = entries[0]
    assert entry.key == f"{STORE_WEAK_SIGNAL}:recent"
    assert entry.plain_summary is None
    assert entry.direction_sentence is not None
    assert "Leans FOR" in entry.direction_sentence
    assert entry.closed is False
    assert entry.closed_label is None


def test_recent_registry_activity_excludes_a_signal_recorded_outside_the_window() -> None:
    registry = _weak_signal_registry(old=_signal_payload(recorded_at="2026-08-01"))
    activity = recent_registry_activity(
        registry, _empty_rotation_registry(), date(2026, 9, 5), days=7
    )
    assert activity.screened_count == 0


def test_recent_registry_activity_uses_plain_summary_when_present() -> None:
    registry = _weak_signal_registry(
        recent=_signal_payload(recorded_at="2026-09-01", plain_summary="A plain-English one-liner.")
    )
    activity = recent_registry_activity(
        registry, _empty_rotation_registry(), date(2026, 9, 5), days=7
    )
    [(_, entries)] = activity.entries_by_category
    assert entries[0].plain_summary == "A plain-English one-liner."


def test_recent_registry_activity_badges_a_terminal_classification_as_resolved() -> None:
    registry = _weak_signal_registry(
        refuted=_signal_payload(
            recorded_at="2026-09-03",
            classification="refuted_mechanism",
            closing_ground="wrong_sign_resolved",
            classification_evidence="whole interval below zero",
            interval=[-2.0, -1.0],
            effect=-1.5,
            probability_positive=0.01,
        )
    )
    activity = recent_registry_activity(
        registry, _empty_rotation_registry(), date(2026, 9, 5), days=7
    )
    assert activity.screened_count == 1
    assert activity.resolved_count == 1
    [(_, entries)] = activity.entries_by_category
    entry = entries[0]
    assert entry.closed is True
    assert entry.closed_label == CLOSED_ACTIVITY_BADGE_TEXT
    assert "failed" not in entry.closed_label.lower()


def test_recent_registry_activity_excludes_instrument_checks() -> None:

    registry = _weak_signal_registry(
        reliability=_signal_payload(
            recorded_at="2026-09-02",
            probability_positive=1.0,
            effect_units="correlation",
            description="split-half",
        ),
        control=_signal_payload(
            recorded_at="2026-09-02", probability_positive=1.0, category="control"
        ),
        oracle_check=_signal_payload(
            recorded_at="2026-09-02",
            probability_positive=0.999,
            description="a movement-oracle sanity check",
        ),
    )
    activity = recent_registry_activity(
        registry, _empty_rotation_registry(), date(2026, 9, 5), days=7
    )
    assert activity.screened_count == 0


def test_recent_registry_activity_category_falls_back_to_store_name() -> None:
    registry = _weak_signal_registry(
        recent=_signal_payload(recorded_at="2026-09-02", category=None)
    )
    activity = recent_registry_activity(
        registry, _empty_rotation_registry(), date(2026, 9, 5), days=7
    )
    [(category, _)] = activity.entries_by_category
    assert category == STORE_WEAK_SIGNAL


def test_recent_registry_activity_includes_a_rotation_window_screened_this_week() -> None:
    rotation_registry = _rotation_registry_with_window()
    activity = recent_registry_activity(
        _weak_signal_registry(), rotation_registry, date(2026, 9, 5), days=7
    )
    assert activity.screened_count == 1
    [(category, entries)] = activity.entries_by_category
    assert category == STORE_ROTATION
    entry = entries[0]
    assert entry.key == f"{STORE_ROTATION}:rot_family"
    assert entry.plain_summary is None


def test_recent_registry_activity_badges_a_closed_negative_rotation_window() -> None:
    rotation_registry = _rotation_registry_with_window(
        verdict="closed_negative", closing_ground="wrong_sign_resolved"
    )
    activity = recent_registry_activity(
        _weak_signal_registry(), rotation_registry, date(2026, 9, 5), days=7
    )
    assert activity.resolved_count == 1
    [(_, entries)] = activity.entries_by_category
    assert entries[0].closed is True
    assert entries[0].closed_label == CLOSED_ACTIVITY_BADGE_TEXT


def test_recent_registry_activity_rotation_window_with_no_result_yet_has_no_direction() -> None:
    rotation_registry = _rotation_registry_with_window(
        state="assigned", spent_at=None, verdict=None, probability_positive=None
    )
    activity = recent_registry_activity(
        _weak_signal_registry(), rotation_registry, date(2026, 9, 5), days=7
    )
    assert activity.screened_count == 1
    [(_, entries)] = activity.entries_by_category
    assert entries[0].direction_sentence is None
    assert entries[0].closed is False


def test_recent_registry_activity_accepts_a_datetime_as_of() -> None:
    registry = _weak_signal_registry(recent=_signal_payload(recorded_at="2026-09-02"))
    activity = recent_registry_activity(
        registry, _empty_rotation_registry(), datetime(2026, 9, 5, 14, 30), days=7
    )
    assert activity.screened_count == 1


def test_recent_registry_activity_groups_are_sorted_by_extremity_within_category() -> None:
    registry = _weak_signal_registry(
        weak=_signal_payload(recorded_at="2026-09-02", probability_positive=0.55),
        strong=_signal_payload(recorded_at="2026-09-02", probability_positive=0.95),
    )
    activity = recent_registry_activity(
        registry, _empty_rotation_registry(), date(2026, 9, 5), days=7
    )
    [(_, entries)] = activity.entries_by_category
    assert [entry.key for entry in entries] == [
        f"{STORE_WEAK_SIGNAL}:strong",
        f"{STORE_WEAK_SIGNAL}:weak",
    ]


def test_recent_rotation_activity_carries_family_plain_summary() -> None:
    registry = _rotation_registry_with_window()
    sentence = "When division rivals meet, this rule backs the underdog."
    from dataclasses import replace

    registry = replace(
        registry,
        families={"rot_family": replace(registry.families["rot_family"], plain_summary=sentence)},
    )
    activity = recent_registry_activity(_weak_signal_registry(), registry, date(2026, 9, 5), days=7)
    [(_, entries)] = activity.entries_by_category
    assert entries[0].plain_summary == sentence
    assert entries[0].plain_summary != registry.families["rot_family"].description


def test_invalidated_post_cutoff_measurement_excluded_from_findings_feeds() -> None:
    registry = _weak_signal_registry(
        valid=_signal_payload(),
        leaked=_signal_payload(probability_positive=1.0, effect=99.0),
    )
    registry = weak_signals.invalidate_signal(
        registry, name="leaked", reason="Measurement used post-cutoff inputs"
    )
    leads = top_open_leads(registry)
    assert len(leads) == 1
    assert leads[0].name == "valid"
    activity = recent_registry_activity(
        registry, _empty_rotation_registry(), as_of=date(2026, 8, 19)
    )
    assert activity.screened_count == 1
    assert activity.resolved_count == 0
