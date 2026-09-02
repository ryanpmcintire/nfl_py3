"""Tests for the findings-page evidence spine: fingerprinting, curation
validation, and the auto-generated "What we're watching" leads.

This is the freshness contract the owner asked for: findings.html must never
silently go stale again. Three things are proven here:

1. Every non-``evergreen`` curated finding in the real
   ``findings_content.FINDINGS`` names at least one existing registry key,
   and its fingerprint matches the live registry (the real audit -- run
   against the actual tracked ``registry/`` files).
2. The auto-leads section renders from a synthetic registry fixture with no
   hand-typed prose at all.
3. A synthetic registry entry that changes CONTENT under a curation entry
   makes :func:`validate_curation` raise -- the exact drift this module
   exists to catch.
"""

from __future__ import annotations

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
    STORE_CHALLENGER,
    STORE_ROTATION,
    STORE_WEAK_SIGNAL,
    CurationError,
    RegistryEntry,
    WatchingLead,
    fingerprint,
    load_all_entries,
    load_rotation_registry,
    load_weak_signal_registry,
    top_open_leads,
    validate_curation,
)

REPO_ROOT = Path(__file__).resolve().parents[1]

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


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
    """A minimal stand-in for ``findings_content.Finding``, since
    ``validate_curation`` only reads four attributes off it (see the
    function's own docstring on why it does not import ``findings_content``
    to avoid a circular import)."""

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


# ---------------------------------------------------------------------------
# Fingerprinting
# ---------------------------------------------------------------------------


def test_fingerprint_is_stable_and_content_sensitive() -> None:
    payload = {"a": 1, "b": [1, 2, 3]}
    assert fingerprint(payload) == fingerprint(dict(payload))  # key order doesn't matter
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


# ---------------------------------------------------------------------------
# validate_curation
# ---------------------------------------------------------------------------


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
    validate_curation([finding], entries)  # must not raise


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
    """The exact freshness-contract test the task calls for: record a
    curation entry against one version of a signal, then re-record the
    signal with different content (the way a real correction would), and
    prove the SAME curated fingerprint no longer validates."""

    original = _weak_signal_registry(alpha=_signal_payload(effect=0.5))
    # Build entries directly from the in-memory registry to avoid touching
    # the real tracked files for this synthetic scenario.
    from nfl_ats.findings_registry import _weak_signal_entry

    entry_before = _weak_signal_entry(original.signals["alpha"])
    finding = _Finding(
        "Does alpha help?",
        registry_keys=("weak_signal:alpha",),
        registry_fingerprints=(entry_before.fingerprint,),
    )
    validate_curation([finding], {"weak_signal:alpha": entry_before})  # passes against the original

    corrected = _weak_signal_registry(alpha=_signal_payload(effect=0.9))  # the "new evidence"
    entry_after = _weak_signal_entry(corrected.signals["alpha"])
    with pytest.raises(CurationError, match="is stale against"):
        validate_curation([finding], {"weak_signal:alpha": entry_after})


def test_real_findings_content_validates_against_the_tracked_registries() -> None:
    """The real audit: every non-evergreen finding in the actual, tracked
    ``findings_content.FINDINGS`` must trace to an existing key in the real,
    tracked registries, and its fingerprint must match. This is the test
    that fails the build the moment a future session records new evidence
    that moves a cited entry without updating the curated prose."""

    from nfl_ats.public_board import load_prospective_challengers

    challengers = load_prospective_challengers(REPO_ROOT / "artifacts")
    entries = load_all_entries(registry_root=REPO_ROOT / "registry", challengers=challengers)
    # Include hand-written lead blurbs too; they use the same fingerprint
    # contract but are easy to miss when only curated cards are audited.
    validate_curation((*findings_content.FINDINGS, *findings_content.LEAD_BLURBS), entries)


def test_recertified_findings_do_not_repeat_known_stale_claims() -> None:
    """The audit's failure modes stay visible in source-level regression tests.

    Counts and pooled estimates belong to the live command, and the current
    nomination/weather descriptions must describe the production paths rather
    than the retired 2026-08-19 wording.
    """

    by_question = {finding.question: finding for finding in findings_content.FINDINGS}
    gap = by_question["Does it matter which line we are graded against?"]
    pool = by_question["Can a pile of weak signals add up to one strong one?"]
    best_pick = by_question[
        "The pool scores one Best Pick a week. Can we tell which of our picks is best?"
    ]
    weather = by_question["What can't we see?"]
    weather_context = by_question["Do rest, travel and weather matter?"]
    history = by_question["How much history does the model need before its picks are trustworthy?"]

    assert findings_content.HEADLINE.opener_close_gap in gap.detail
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


# ---------------------------------------------------------------------------
# top_open_leads
# ---------------------------------------------------------------------------


def test_top_open_leads_renders_from_a_synthetic_fixture() -> None:
    registry = _weak_signal_registry(
        strong=_signal_payload(probability_positive=0.95, description="a strong lean"),
        weak=_signal_payload(probability_positive=0.51, description="a weak lean"),
    )
    leads = top_open_leads(registry)
    assert len(leads) == 2
    assert leads[0].name == "strong"  # ranked by |P+ - 0.5|, most extreme first
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


def test_top_open_leads_collapses_close_and_opener_pair_to_the_opener_grade() -> None:
    registry = _weak_signal_registry(
        foo=_signal_payload(probability_positive=0.9, description="close-graded"),
        foo_opener=_signal_payload(probability_positive=0.6, description="opener-graded"),
    )
    leads = top_open_leads(registry)
    assert len(leads) == 1
    assert leads[0].name == "foo_opener"  # opener grade wins even though less extreme


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
