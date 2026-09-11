from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import pytest

from nfl_ats.weak_signals import (
    EFFECT_UNITS,
    NO_SPLIT_HALF_RELIABILITY_MAX,
    WEAK_SIGNAL_REGISTRY_VERSION,
    Registry,
    WeakSignalError,
    coherence_problems,
    combination_report,
    family_overlap_warnings,
    implausible_standard_errors,
    load_registry,
    overlap_warnings,
    poolable_signals,
    pooled_effect,
    record_signal,
    registry_from_payload,
    retag_effect_units,
    rows_needing_remeasurement,
    save_registry,
    sign_test,
    signal_family,
    signal_from_payload,
)


def _signal(**overrides: Any) -> dict[str, Any]:
    body = {
        "recorded_at": "2026-08-17",
        "description": "a small measured effect",
        "source": "docs/example.md",
        "effect": 0.10,
        "effect_units": "ats_points",
        "classification": "unresolved_below_power",
        "league": "nfl",
        "seasons": [2009, 2017],
        "standard_error": 0.30,
    }
    body.update(overrides)
    return body


def _payload(**signals: dict[str, Any]) -> dict[str, Any]:
    return {"version": WEAK_SIGNAL_REGISTRY_VERSION, "notes": ["test ledger"], "signals": signals}


def test_round_trips_and_rejects_unknown_fields(tmp_path: Path) -> None:
    registry = registry_from_payload(_payload(alpha=_signal()))
    destination = tmp_path / "weak_signals.json"
    save_registry(registry, destination)
    assert load_registry(destination) == registry

    with pytest.raises(WeakSignalError, match="unknown top-level fields"):
        registry_from_payload({**_payload(), "budget": 1})
    with pytest.raises(WeakSignalError, match="unknown fields"):
        registry_from_payload(_payload(alpha=_signal(owner="ryan")))


def test_missing_registry_file_loads_as_empty(tmp_path: Path) -> None:
    assert load_registry(tmp_path / "absent.json").signals == {}


def test_validation_rejects_bad_taxonomy_units_and_seasons() -> None:
    with pytest.raises(WeakSignalError, match="unknown classification"):
        registry_from_payload(_payload(alpha=_signal(classification="probably_fine")))
    with pytest.raises(WeakSignalError, match="unknown effect_units"):
        registry_from_payload(_payload(alpha=_signal(effect_units="vibes")))
    with pytest.raises(WeakSignalError, match="seasons out of order"):
        registry_from_payload(_payload(alpha=_signal(seasons=[2020, 2011])))
    with pytest.raises(WeakSignalError, match="non-finite effect"):
        registry_from_payload(_payload(alpha=_signal(effect=float("nan"))))


def test_plain_summary_and_category_round_trip(tmp_path: Path) -> None:
    registry = registry_from_payload(
        _payload(
            alpha=_signal(
                plain_summary="A short sentence a fan can read on its own.",
                category="onfield",
            )
        )
    )
    destination = tmp_path / "weak_signals.json"
    save_registry(registry, destination)
    loaded = load_registry(destination)
    assert loaded == registry
    assert loaded.signals["alpha"].plain_summary == "A short sentence a fan can read on its own."
    assert loaded.signals["alpha"].category == "onfield"


def test_plain_summary_and_category_are_optional_and_round_trip_as_none(tmp_path: Path) -> None:
    registry = registry_from_payload(_payload(alpha=_signal()))
    signal = registry.signals["alpha"]
    assert signal.plain_summary is None
    assert signal.category is None
    destination = tmp_path / "weak_signals.json"
    save_registry(registry, destination)
    assert load_registry(destination) == registry


def test_category_rejects_an_unknown_value() -> None:
    with pytest.raises(WeakSignalError, match="unknown category"):
        registry_from_payload(_payload(alpha=_signal(category="vibes")))


def test_recording_the_same_signal_twice_needs_an_explicit_replace() -> None:
    registry = Registry(version=WEAK_SIGNAL_REGISTRY_VERSION, notes=(), signals={})
    first = signal_from_payload("alpha", _signal())
    registry = record_signal(registry, first)
    with pytest.raises(WeakSignalError, match="already recorded"):
        record_signal(registry, first)
    updated = record_signal(
        registry, signal_from_payload("alpha", _signal(effect=0.2)), replace=True
    )
    assert updated.signals["alpha"].effect == pytest.approx(0.2)


def test_standard_error_is_recovered_from_an_interval() -> None:
    signal = signal_from_payload("alpha", _signal(standard_error=None, interval=[-0.423, 0.417]))
    assert signal.resolved_standard_error() == pytest.approx((0.417 + 0.423) / 3.9199, rel=1e-3)


def test_sign_test_counts_directions_not_magnitudes() -> None:
    signals = [
        signal_from_payload(f"s{i}", _signal(effect=0.05 if i < 10 else -0.05)) for i in range(11)
    ]
    result = sign_test(signals)
    assert result["signals"] == 11
    assert result["favouring_candidate"] == 10
    assert result["p_value"] < 0.05
    assert "further than chance" in result["interpretation"]


def test_sign_test_calls_a_coin_flip_a_coin_flip() -> None:
    signals = [
        signal_from_payload(f"s{i}", _signal(effect=0.05 if i % 2 == 0 else -0.05))
        for i in range(10)
    ]
    result = sign_test(signals)
    assert result["p_value"] > 0.10
    assert "coin flip" in result["interpretation"]


def test_sign_test_on_an_empty_pile_is_not_a_finding() -> None:
    assert sign_test([])["p_value"] == 1.0


def test_sign_test_does_not_score_exact_ties_against_the_candidate() -> None:
    signals = [
        signal_from_payload("up", _signal(effect=0.05)),
        signal_from_payload("down", _signal(effect=-0.05)),
        *[signal_from_payload(f"tie{i}", _signal(effect=0.0)) for i in range(8)],
    ]
    result = sign_test(signals)
    assert result["signals"] == 10
    assert result["ties"] == 8
    assert result["favouring_candidate"] == 1
    assert result["favouring_baseline"] == 1
    assert result["informative_signals"] == 2
    assert result["p_value"] == pytest.approx(1.0)
    assert "coin flip" in result["interpretation"]


def test_sign_test_reports_both_tie_conventions_so_no_number_is_ambiguous() -> None:
    signals = [
        signal_from_payload("up", _signal(effect=0.05)),
        signal_from_payload("tie", _signal(effect=0.0)),
        signal_from_payload("down1", _signal(effect=-0.05)),
        signal_from_payload("down2", _signal(effect=-0.05)),
    ]
    result = sign_test(signals)
    assert result["favouring_candidate"] == 1
    assert result["favouring_candidate_half_credit"] == pytest.approx(1.5)
    assert result["favouring_baseline_half_credit"] == pytest.approx(2.5)
    assert result["share_favouring_candidate"] == pytest.approx(1 / 3)
    assert result["share_favouring_candidate_half_credit"] == pytest.approx(1.5 / 4)
    assert "ties excluded" in result["tie_convention"]


def test_sign_test_says_so_when_every_signal_is_a_dead_heat() -> None:
    signals = [signal_from_payload(f"tie{i}", _signal(effect=0.0)) for i in range(5)]
    result = sign_test(signals)
    assert result["ties"] == 5
    assert result["informative_signals"] == 0
    assert result["p_value"] == pytest.approx(1.0)
    assert "no direction to test" in result["interpretation"]


def test_pooling_sharpens_the_standard_error_toward_sqrt_k() -> None:
    signals = [
        signal_from_payload(f"s{i}", _signal(effect=0.20, standard_error=0.40)) for i in range(4)
    ]
    pooled = pooled_effect(signals, method="fixed")
    assert pooled["pooled_effect"] == pytest.approx(0.20)
    assert pooled["standard_error"] == pytest.approx(0.40 / math.sqrt(4))
    assert pooled["sharpening_vs_best_single"] == pytest.approx(2.0)
    assert not pooled["excludes_zero"]


def test_pooling_needs_roughly_sixteen_half_sigma_signals_to_resolve() -> None:
    def pooled_sigma(count: int) -> float:
        signals = [
            signal_from_payload(f"s{i}", _signal(effect=0.20, standard_error=0.40))
            for i in range(count)
        ]
        result = pooled_effect(signals, method="fixed")
        return abs(result["pooled_effect"]) / result["standard_error"]

    assert pooled_sigma(4) == pytest.approx(1.0, rel=1e-6)
    assert pooled_sigma(9) == pytest.approx(1.5, rel=1e-6)
    assert pooled_sigma(16) == pytest.approx(2.0, rel=1e-6)

    sixteen = [
        signal_from_payload(f"s{i}", _signal(effect=0.20, standard_error=0.40)) for i in range(16)
    ]
    assert pooled_effect(sixteen, method="fixed")["excludes_zero"]


def test_random_effects_widens_when_signals_disagree() -> None:
    disagreeing = [
        signal_from_payload("a", _signal(effect=2.0, standard_error=0.2)),
        signal_from_payload("b", _signal(effect=-2.0, standard_error=0.2)),
    ]
    fixed = pooled_effect(disagreeing, method="fixed")
    random = pooled_effect(disagreeing, method="random")
    assert random["heterogeneity_tau_squared"] > 0.0
    assert random["standard_error"] > fixed["standard_error"]


def _sized(name: str, *, effect: float, se: float, games: int) -> Any:
    return signal_from_payload(
        name,
        _signal(
            effect=effect, standard_error=se, effect_units="accuracy_points", sample_games=games
        ),
    )


def test_a_three_game_cell_cannot_hold_the_whole_pool() -> None:
    honest = [_sized(f"honest{i}", effect=0.1, se=1.0, games=4000) for i in range(20)]
    degenerate = _sized("degenerate", effect=-9.0, se=1e-5, games=3)

    legacy = pooled_effect([*honest, degenerate], method="fixed", weighting="inverse_variance")
    assert legacy["max_weight_share"] > 0.99
    assert legacy["most_influential_signal"] == "degenerate"
    assert legacy["pooled_effect"] == pytest.approx(-9.0, abs=1e-6)

    fixed = pooled_effect([*honest, degenerate], method="fixed")
    assert fixed["max_weight_share"] < 0.2
    assert fixed["most_influential_signal"] != "degenerate"
    assert fixed["pooled_effect"] > 0.0


def test_the_thin_cell_is_floored_and_flagged_but_never_dropped() -> None:
    honest = [_sized(f"honest{i}", effect=0.1, se=1.0, games=4000) for i in range(20)]
    degenerate = _sized("degenerate", effect=-9.0, se=1e-5, games=3)
    result = pooled_effect([*honest, degenerate], method="fixed")

    assert result["signals"] == 21
    assert result["standard_errors_floored"] == 1
    floored = result["floored_signals"][0]
    assert floored["name"] == "degenerate"
    assert floored["floored_to"] > floored["recorded_standard_error"]
    assert [row["name"] for row in implausible_standard_errors([*honest, degenerate])] == [
        "degenerate"
    ]


def test_pooling_leaves_plausible_standard_errors_alone() -> None:
    signals = [
        _sized("wide", effect=0.1, se=2.0, games=100),
        _sized("mid", effect=0.1, se=1.0, games=400),
        _sized("tight", effect=0.1, se=0.5, games=1600),
        _sized("tighter", effect=0.1, se=0.4, games=2500),
    ]
    result = pooled_effect(signals, method="fixed")
    assert result["standard_errors_floored"] == 0
    assert result["floored_signals"] == []
    legacy = pooled_effect(signals, method="fixed", weighting="inverse_variance")
    assert result["pooled_effect"] == pytest.approx(legacy["pooled_effect"])


def test_pooling_reports_the_superseded_weighting_so_the_change_is_auditable() -> None:
    honest = [_sized(f"honest{i}", effect=0.1, se=1.0, games=4000) for i in range(20)]
    degenerate = _sized("degenerate", effect=-9.0, se=1e-5, games=3)
    result = pooled_effect([*honest, degenerate], method="fixed")
    legacy = result["legacy_inverse_variance"]
    assert legacy["most_influential_signal"] == "degenerate"
    assert legacy["max_weight_share"] > 0.99
    assert result["max_weight_share"] < legacy["max_weight_share"]
    assert result["effective_signals"] > legacy["effective_signals"]


def test_pooling_reports_probability_positive_not_just_a_zero_crossing() -> None:
    signals = [_sized(f"s{i}", effect=0.2, se=1.0, games=1000) for i in range(4)]
    result = pooled_effect(signals, method="fixed")
    assert not result["excludes_zero"]
    assert 0.5 < result["probability_positive"] < 0.95


def test_pooling_falls_back_when_no_entry_records_a_sample_size() -> None:
    signals = [
        signal_from_payload(f"s{i}", _signal(effect=0.2, standard_error=0.4)) for i in range(4)
    ]
    result = pooled_effect(signals, method="fixed")
    assert result["weighting"] == "inverse_variance_fallback_no_curve"
    assert result["pooled_effect"] == pytest.approx(0.20)


def test_rows_recorded_under_the_strict_zero_convention_are_flagged_not_rewritten() -> None:
    signals = [
        signal_from_payload(
            "dead_heat",
            _signal(effect=0.0, standard_error=None, interval=[0.0, 0.0], probability_positive=0.0),
        ),
        signal_from_payload(
            "real_negative",
            _signal(effect=-0.4, standard_error=0.2, probability_positive=0.0),
        ),
        signal_from_payload("ordinary", _signal(effect=0.1, probability_positive=0.7)),
    ]
    flagged = rows_needing_remeasurement(signals)
    zero_atom = flagged["zero_atom_probability_positive"]
    assert zero_atom["count"] == 1
    assert zero_atom["signals"] == ["dead_heat"]
    assert zero_atom["correctable_value"] == 0.5
    assert flagged["strict_zero_with_nonzero_effect"]["signals"] == ["real_negative"]
    assert "not a verdict" in flagged["nothing_is_closed_by_this"]


def test_pooling_refuses_to_mix_units() -> None:
    mixed = [
        signal_from_payload("a", _signal(effect_units="ats_points")),
        signal_from_payload("b", _signal(effect_units="brier")),
    ]
    with pytest.raises(WeakSignalError, match="mixed effect units"):
        pooled_effect(mixed)


def test_pooling_without_any_uncertainty_reports_rather_than_guesses() -> None:
    bare = [signal_from_payload("a", _signal(standard_error=None))]
    result = pooled_effect(bare)
    assert result["signals"] == 0
    assert result["pooled_effect"] is None


def test_overlapping_seasons_are_flagged_as_shared_noise() -> None:
    overlapping = [
        signal_from_payload("a", _signal(seasons=[2009, 2015])),
        signal_from_payload("b", _signal(seasons=[2014, 2020])),
    ]
    warnings = overlap_warnings(overlapping)
    assert len(warnings) == 1
    assert "2014-2015" in warnings[0]

    disjoint = [
        signal_from_payload("a", _signal(seasons=[2009, 2012])),
        signal_from_payload("b", _signal(seasons=[2013, 2016])),
    ]
    assert overlap_warnings(disjoint) == []

    cross = [
        signal_from_payload("a", _signal(seasons=[2009, 2015], league="nfl")),
        signal_from_payload("b", _signal(seasons=[2009, 2015], league="cfb")),
    ]
    assert overlap_warnings(cross) == []


def test_signal_family_collapses_decompositions_of_one_construct() -> None:

    def family(name: str, *, league: str = "nfl") -> str:
        return signal_family(signal_from_payload(name, _signal(league=league)))

    assert family("bias_battery_short_week_opener") == family("bias_battery_short_week")
    assert (
        family("altitude_deficit_4000ft_era_2018_2025")
        == family("altitude_deficit_4000ft_era_2009_2017")
        == family("altitude_deficit_4000ft")
    )
    assert (
        family("body_clock_west_road_early_2009_2016")
        == family("body_clock_west_road_early_2017_2025")
        == family("body_clock_west_road_early")
    )
    assert family("bye_overval_home_edge_pre2011") == family("bye_overval_home_edge_post2011")
    assert family("bias_battery_home_underdog") == "bias_battery"
    assert family("cfb_bias_battery_home_underdog", league="cfb") == "cfb_bias_battery"
    declared = signal_from_payload("odd_name", _signal(family="declared_family"))
    assert signal_family(declared) == "declared_family"


def test_family_overlap_warnings_report_families_not_pairs() -> None:

    members = [
        signal_from_payload("bias_battery_cell_a", _signal(seasons=[2009, 2025])),
        signal_from_payload("bias_battery_cell_b", _signal(seasons=[2009, 2025])),
        signal_from_payload("lone_signal", _signal(seasons=[2009, 2025])),
    ]
    report = family_overlap_warnings(members)
    assert report["families"] == 2
    assert report["families_with_internal_overlap"] == 1
    assert report["pairwise_within_family_pairs"] == 1
    assert report["cross_family_shared_window_pairs"] == 1
    assert report["pairwise_overlap_pairs"] == 3
    entry = report["within_family"][0]
    assert entry["family"] == "bias_battery"
    assert entry["members"] == 2
    assert entry["shared_seasons"] == [2009, 2025]
    assert "correlated decompositions" in entry["warning"]

    disjoint = [
        signal_from_payload("x_a", _signal(seasons=[2009, 2010])),
        signal_from_payload("x_b", _signal(seasons=[2011, 2012])),
    ]
    empty = family_overlap_warnings(disjoint)
    assert empty["families_with_internal_overlap"] == 0
    assert empty["cross_family_shared_window_pairs"] == 0
    assert empty["within_family"] == []


def test_combination_report_carries_per_family_overlap_output() -> None:
    registry = registry_from_payload(
        _payload(
            widget=_signal(seasons=[2009, 2017]),
            widget_opener=_signal(seasons=[2009, 2017]),
            unrelated=_signal(seasons=[2009, 2017]),
        )
    )
    report = combination_report(registry, league="nfl")
    structured = report["overlap_warnings"]
    assert structured["families_with_internal_overlap"] == 1
    assert structured["within_family"][0]["family"] == "widget"
    assert report["overlap_pairwise_count"] >= 1


def test_effect_outside_interval_is_refused_at_record_time() -> None:

    contradictory = signal_from_payload("alpha", _signal(effect=0.05, interval=[-0.03, 0.03]))
    base = Registry(version=WEAK_SIGNAL_REGISTRY_VERSION, notes=(), signals={})
    with pytest.raises(WeakSignalError, match="outside its own interval"):
        record_signal(base, contradictory)
    problems = coherence_problems([contradictory])
    assert problems and problems[0]["signal"] == "alpha"


def test_bounded_by_control_needs_quantitative_evidence() -> None:
    with pytest.raises(WeakSignalError, match="no quantitative evidence"):
        signal_from_payload(
            "alpha",
            _signal(
                classification="bounded_by_control",
                closing_ground="positive_control_bound",
                classification_evidence="the control saw nothing",
                standard_error=None,
            ),
        )
    ok = signal_from_payload(
        "alpha",
        _signal(
            classification="bounded_by_control",
            closing_ground="positive_control_bound",
            classification_evidence="the control saw nothing at P+ 0.984",
            probability_positive=0.984,
            standard_error=None,
        ),
    )
    assert ok.classification == "bounded_by_control"


def test_no_reliability_closure_cannot_cite_a_reliable_trait() -> None:
    reliable = _signal(
        classification="refuted_mechanism",
        closing_ground="no_split_half_reliability",
        classification_evidence="trait persists across halves",
        reliability=0.719,
    )
    with pytest.raises(WeakSignalError, match="ceiling this ground admits"):
        signal_from_payload("alpha", reliable)
    assert NO_SPLIT_HALF_RELIABILITY_MAX == 0.10


def test_only_genuinely_unresolved_signals_are_poolable() -> None:
    registry = registry_from_payload(
        _payload(
            live=_signal(classification="unresolved_below_power"),
            refuted=_signal(
                classification="refuted_mechanism",
                closing_ground="wrong_sign_resolved",
                classification_evidence="whole interval below zero",
                effect=-0.6,
                standard_error=None,
                interval=[-0.9, -0.3],
            ),
            bounded=_signal(
                classification="bounded_by_control",
                closing_ground="positive_control_bound",
                classification_evidence="deliberate-leak control detected at P+ 0.984",
                probability_positive=0.984,
            ),
        )
    )
    assert [s.name for s in poolable_signals(registry)] == ["live"]
    report = combination_report(registry)
    assert report["eligible"] == ["live"]
    assert report["excluded_with_reason"] == {
        "bounded": "bounded_by_control",
        "refuted": "refuted_mechanism",
    }


def test_combination_report_filters_by_league_and_records_seasons() -> None:
    registry = registry_from_payload(
        _payload(
            nfl_one=_signal(league="nfl", seasons=[2009, 2011]),
            cfb_one=_signal(league="cfb", seasons=[2006, 2008]),
        )
    )
    report = combination_report(registry, league="nfl")
    assert report["eligible"] == ["nfl_one"]
    assert report["seasons_touched_by_inputs"] == [2009, 2010, 2011]
    assert "predeclared" in report["guidance"]


def test_pooling_refuses_to_mix_leagues_when_none_is_chosen() -> None:

    registry = registry_from_payload(
        _payload(
            nfl_one=_signal(league="nfl", seasons=[2009, 2011]),
            cfb_one=_signal(league="cfb", seasons=[2006, 2008]),
        )
    )
    with pytest.raises(ValueError, match="Refusing to pool across leagues"):
        combination_report(registry)

    assert combination_report(registry, league="cfb")["eligible"] == ["cfb_one"]


def test_live_ledger_validates_if_present() -> None:

    path = Path(__file__).resolve().parents[1] / "registry" / "weak_signals.json"
    if not path.is_file():
        pytest.skip("no weak-signal ledger committed yet")
    registry = load_registry(path)
    for name, signal in registry.signals.items():
        assert signal.classification in {
            "unresolved_below_power",
            "refuted_mechanism",
            "bounded_by_control",
        }, name
        assert signal.source, name
        assert math.isfinite(signal.effect), name
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["version"] == WEAK_SIGNAL_REGISTRY_VERSION


def test_weak_signal_direction_is_the_recorded_field() -> None:
    assert signal_from_payload("a", _signal(effect=0.01)).favours_candidate
    assert not signal_from_payload("a", _signal(effect=-0.01)).favours_candidate


def test_terminal_verdict_requires_an_admissible_closing_ground() -> None:

    with pytest.raises(WeakSignalError, match="no admissible closing_ground"):
        signal_from_payload(
            "alpha",
            _signal(
                classification="refuted_mechanism",
                classification_evidence="interval contains zero",
            ),
        )
    with pytest.raises(WeakSignalError, match="no admissible closing_ground"):
        signal_from_payload(
            "alpha",
            _signal(
                classification="bounded_by_control",
                closing_ground="wrong_sign_resolved",
                classification_evidence="mismatched ground for this classification",
            ),
        )


def test_wrong_sign_must_be_resolved_not_a_lean() -> None:
    crossing = _signal(
        classification="refuted_mechanism",
        closing_ground="wrong_sign_resolved",
        classification_evidence="negative point estimate",
        standard_error=None,
        interval=[-0.5, 0.2],
    )
    with pytest.raises(WeakSignalError, match="is not entirely"):
        signal_from_payload("alpha", crossing)

    resolved = signal_from_payload(
        "alpha",
        _signal(
            effect=-0.6,
            classification="refuted_mechanism",
            closing_ground="wrong_sign_resolved",
            classification_evidence="whole interval on the wrong side of zero",
            standard_error=None,
            interval=[-0.9, -0.3],
        ),
    )
    assert resolved.closing_ground == "wrong_sign_resolved"


def test_no_reliability_ground_needs_the_measurement_it_cites() -> None:
    without_measurement = _signal(
        classification="refuted_mechanism",
        closing_ground="no_split_half_reliability",
        classification_evidence="trait does not persist",
    )
    with pytest.raises(WeakSignalError, match="no reliability measurement"):
        signal_from_payload("alpha", without_measurement)

    recorded = signal_from_payload("alpha", _signal(**{**without_measurement, "reliability": 0.02}))
    assert recorded.reliability == pytest.approx(0.02)


def test_control_bounded_closure_must_cite_its_evidence() -> None:
    with pytest.raises(WeakSignalError, match="classification_evidence is empty"):
        signal_from_payload(
            "alpha",
            _signal(
                classification="bounded_by_control",
                closing_ground="positive_control_bound",
                classification_evidence="   ",
            ),
        )


def test_unresolved_signals_cannot_carry_a_closing_ground() -> None:
    with pytest.raises(WeakSignalError, match="cannot carry a closing_ground"):
        signal_from_payload("alpha", _signal(closing_ground="wrong_sign_resolved"))


def test_record_signal_enforces_closure_grounds_directly() -> None:

    from nfl_ats.weak_signals import WeakSignal

    registry = Registry(version=WEAK_SIGNAL_REGISTRY_VERSION, notes=(), signals={})
    bad = WeakSignal(
        name="alpha",
        recorded_at="2026-08-18",
        description="terminal verdict with no admissible ground",
        source="docs/example.md",
        effect=-0.1,
        effect_units="ats_points",
        classification="refuted_mechanism",
        league="nfl",
        seasons=(2009, 2017),
        classification_evidence="interval contains zero",
    )
    with pytest.raises(WeakSignalError, match="no admissible closing_ground"):
        record_signal(registry, bad)


def test_closing_ground_round_trips(tmp_path: Path) -> None:
    payload = _payload(
        alpha=_signal(
            effect=-0.6,
            classification="refuted_mechanism",
            closing_ground="wrong_sign_resolved",
            classification_evidence="whole interval below zero",
            standard_error=None,
            interval=[-0.9, -0.3],
        )
    )
    registry = registry_from_payload(payload)
    destination = tmp_path / "weak_signals.json"
    save_registry(registry, destination)
    reloaded = load_registry(destination)
    assert reloaded.signals["alpha"].closing_ground == "wrong_sign_resolved"


def test_existing_effect_units_are_unchanged_by_the_new_additions() -> None:
    assert EFFECT_UNITS[:5] == ("ats_points", "accuracy_points", "brier", "log_loss", "mae")
    assert set(EFFECT_UNITS[5:]) == {
        "correlation",
        "mae_improvement",
        "brier_improvement",
        "log_loss_improvement",
        "payout_first_pp",
    }


@pytest.mark.parametrize(
    "unit", ["correlation", "mae_improvement", "brier_improvement", "log_loss_improvement"]
)
def test_new_effect_units_are_accepted_and_round_trip(unit: str, tmp_path: Path) -> None:
    registry = registry_from_payload(_payload(alpha=_signal(effect_units=unit, effect=0.15)))
    destination = tmp_path / "weak_signals.json"
    save_registry(registry, destination)
    reloaded = load_registry(destination)
    assert reloaded == registry
    assert reloaded.signals["alpha"].effect_units == unit


@pytest.mark.parametrize(
    "unit", ["correlation", "mae_improvement", "brier_improvement", "log_loss_improvement"]
)
def test_new_units_follow_the_same_positive_favours_candidate_convention(unit: str) -> None:
    assert signal_from_payload("a", _signal(effect_units=unit, effect=0.01)).favours_candidate
    assert not signal_from_payload("a", _signal(effect_units=unit, effect=-0.01)).favours_candidate


def test_pool_on_a_new_unit_empty_bucket_does_not_crash() -> None:
    registry = registry_from_payload(_payload(alpha=_signal(effect_units="accuracy_points")))
    report = combination_report(registry, effect_units="correlation")
    assert report["eligible"] == []
    assert report["pooled_by_unit"] == {}


def test_pool_on_a_new_unit_one_entry_bucket_does_not_crash() -> None:
    registry = registry_from_payload(
        _payload(
            alpha=_signal(effect_units="mae_improvement", effect=0.00082, standard_error=0.0006)
        )
    )
    report = combination_report(registry, effect_units="mae_improvement")
    assert report["eligible"] == ["alpha"]
    pooled = report["pooled_by_unit"]["mae_improvement"]
    assert pooled["signals"] == 1
    assert pooled["pooled_effect"] == pytest.approx(0.00082)
    assert pooled["effect_units"] == "mae_improvement"


def test_pooling_a_correlation_and_an_accuracy_points_signal_still_refuses_mixed_units() -> None:
    mixed = [
        signal_from_payload("a", _signal(effect_units="accuracy_points")),
        signal_from_payload("b", _signal(effect_units="correlation")),
    ]
    with pytest.raises(WeakSignalError, match="mixed effect units"):
        pooled_effect(mixed)


def test_accuracy_points_pool_output_is_unchanged_by_the_new_units(tmp_path: Path) -> None:
    registry = registry_from_payload(
        _payload(
            acc_a=_signal(effect_units="accuracy_points", effect=0.20, standard_error=0.40),
            acc_b=_signal(effect_units="accuracy_points", effect=0.20, standard_error=0.40),
            corr_a=_signal(effect_units="correlation", effect=0.10, standard_error=0.05),
        )
    )
    report = combination_report(registry, effect_units="accuracy_points")
    assert report["eligible"] == ["acc_a", "acc_b"]
    pooled = report["pooled_by_unit"]["accuracy_points"]
    assert pooled["pooled_effect"] == pytest.approx(0.20)
    assert pooled["standard_error"] == pytest.approx(0.40 / math.sqrt(2))
    assert pooled["effect_units"] == "accuracy_points"


def test_retag_effect_units_changes_only_the_unit_and_appends_an_audit_note() -> None:
    original_payload = _signal(
        effect_units="mae",
        effect=0.00082,
        standard_error=0.0006,
        interval=[-0.0004, 0.0021],
        probability_positive=0.71,
        sample_games=200,
        sample_blocks=17,
        reliability=0.4,
        family="totals_market_residual",
        notes="sign already negated: positive means the candidate's MAE was lower",
        plain_summary="A short sentence a fan can read on its own.",
        category="modeling",
    )
    registry = registry_from_payload(_payload(totals_market_residual_blend=original_payload))
    original = registry.signals["totals_market_residual_blend"]

    retagged_registry = retag_effect_units(
        registry,
        "totals_market_residual_blend",
        effect_units="mae_improvement",
        reason="was mae with the sign convention explained only in notes",
        changed_at="2026-09-01T12:00:00+00:00",
    )
    retagged = retagged_registry.signals["totals_market_residual_blend"]

    assert retagged.effect_units == "mae_improvement"
    assert original.notes in retagged.notes
    assert "'mae' -> 'mae_improvement'" in retagged.notes
    assert "was mae with the sign convention explained only in notes" in retagged.notes
    assert "2026-09-01T12:00:00+00:00" in retagged.notes

    assert retagged.effect == original.effect
    assert retagged.interval == original.interval
    assert retagged.standard_error == original.standard_error
    assert retagged.probability_positive == original.probability_positive
    assert retagged.sample_games == original.sample_games
    assert retagged.sample_blocks == original.sample_blocks
    assert retagged.classification == original.classification
    assert retagged.classification_evidence == original.classification_evidence
    assert retagged.closing_ground == original.closing_ground
    assert retagged.reliability == original.reliability
    assert retagged.family == original.family
    assert retagged.plain_summary == original.plain_summary
    assert retagged.category == original.category
    assert retagged.league == original.league
    assert retagged.seasons == original.seasons
    assert retagged.source == original.source
    assert retagged.description == original.description
    assert retagged.recorded_at == original.recorded_at
    assert retagged.name == original.name

    assert registry.signals["totals_market_residual_blend"].effect_units == "mae"


def test_retag_effect_units_appends_to_empty_notes_without_a_leading_blank_line() -> None:
    registry = registry_from_payload(_payload(alpha=_signal(notes="")))
    retagged = retag_effect_units(
        registry, "alpha", effect_units="correlation", reason="was a numeric container only"
    )
    notes = retagged.signals["alpha"].notes
    assert notes.startswith("[")
    assert "correlation" in notes


def test_retag_effect_units_refuses_an_unknown_unit() -> None:
    registry = registry_from_payload(_payload(alpha=_signal()))
    with pytest.raises(WeakSignalError, match="Unknown effect_units"):
        retag_effect_units(registry, "alpha", effect_units="vibes", reason="typo")


def test_retag_effect_units_refuses_a_missing_entry() -> None:
    registry = registry_from_payload(_payload(alpha=_signal()))
    with pytest.raises(WeakSignalError, match="No recorded signal named"):
        retag_effect_units(registry, "does_not_exist", effect_units="correlation", reason="n/a")


def test_retag_effect_units_round_trips_through_save_and_load(tmp_path: Path) -> None:
    registry = registry_from_payload(
        _payload(
            xlg06_rookie_prior_stage1_qb=_signal(effect_units="accuracy_points", effect=-0.0018)
        )
    )
    retagged = retag_effect_units(
        registry,
        "xlg06_rookie_prior_stage1_qb",
        effect_units="correlation",
        reason="was a Pearson correlation forced into accuracy_points as a numeric container only",
    )
    destination = tmp_path / "weak_signals.json"
    save_registry(retagged, destination)
    reloaded = load_registry(destination)
    assert reloaded.signals["xlg06_rookie_prior_stage1_qb"].effect_units == "correlation"
    assert reloaded.signals["xlg06_rookie_prior_stage1_qb"].effect == pytest.approx(-0.0018)
    assert "numeric container only" in reloaded.signals["xlg06_rookie_prior_stage1_qb"].notes


def test_a_correction_may_restate_a_summary_but_never_a_measurement() -> None:

    base = _signal(
        description="identical picks on every game",
        effect=0.0,
        effect_units="accuracy_points",
        interval=[0.0, 0.0],
        probability_positive=0.5,
    )
    entry = {
        "at": "2026-09-08T18:00:00+00:00",
        "field": "probability_positive",
        "from": 0.0,
        "to": 0.5,
        "reason": "every resample was an exact tie; the old convention charged the zero atom",
    }

    signal_from_payload("noop", {**base, "corrections": [entry]})

    with pytest.raises(WeakSignalError, match="not correctable in place"):
        signal_from_payload("bad", {**base, "corrections": [{**entry, "field": "effect"}]})
    with pytest.raises(WeakSignalError, match="not correctable in place"):
        signal_from_payload("bad", {**base, "corrections": [{**entry, "field": "classification"}]})
    with pytest.raises(WeakSignalError, match="reason must say why"):
        signal_from_payload("bad", {**base, "corrections": [{**entry, "reason": "  "}]})
    with pytest.raises(WeakSignalError, match="missing"):
        signal_from_payload("bad", {**base, "corrections": [{"field": "probability_positive"}]})
