"""Tests for the ENG-07 read-only registry/overlap explorer.

``nfl_ats.registry_explorer`` never writes to either registry; every test
here either builds small synthetic registries in ``tmp_path`` (covering each
view's behaviour precisely) or runs the module against the real tracked
registries and asserts only structural properties plus byte-for-byte
file-unchanged, matching the module's own read-only contract.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from nfl_ats import registry_explorer, rotation, weak_signals

REPO_ROOT = Path(__file__).resolve().parents[1]
LIVE_WEAK_SIGNALS = REPO_ROOT / "registry" / "weak_signals.json"
LIVE_ROTATION = REPO_ROOT / "registry" / "rotation_registry.json"


# ---------------------------------------------------------------------------
# Synthetic weak-signal registry
# ---------------------------------------------------------------------------


def _weak_signal(**overrides: Any) -> dict[str, Any]:
    body: dict[str, Any] = {
        "recorded_at": "2026-01-01",
        "description": "synthetic test signal",
        "source": "tests/test_registry_explorer.py",
        "effect": 1.0,
        "effect_units": "accuracy_points",
        "classification": "unresolved_below_power",
        "league": "nfl",
        "seasons": [2015, 2017],
        "probability_positive": 0.5,
        "sample_games": 100,
    }
    body.update(overrides)
    return body


def _weak_payload(**signals: dict[str, Any]) -> dict[str, Any]:
    return {"version": weak_signals.WEAK_SIGNAL_REGISTRY_VERSION, "notes": [], "signals": signals}


def _write_weak_signals(tmp_path: Path, **signals: dict[str, Any]) -> weak_signals.Registry:
    registry = weak_signals.registry_from_payload(_weak_payload(**signals))
    destination = tmp_path / "weak_signals.json"
    weak_signals.save_registry(registry, destination)
    return weak_signals.load_registry(destination)


def _synthetic_weak_registry(tmp_path: Path) -> weak_signals.Registry:
    return _write_weak_signals(
        tmp_path,
        # alpha_battery: two overlapping-window members -> one shared-population group.
        alpha_high=_weak_signal(
            family="alpha_battery",
            effect=1.0,
            probability_positive=0.85,
            seasons=[2015, 2017],
            sample_games=300,
            category="market",
        ),
        alpha_low=_weak_signal(
            family="alpha_battery",
            effect=0.5,
            probability_positive=0.55,
            seasons=[2016, 2018],
            sample_games=200,
            category="market",
        ),
        # beta_battery: solo member, no overlap partner, category-fallback source.
        beta_solo=_weak_signal(
            family="beta_battery",
            effect=0.2,
            probability_positive=0.60,
            seasons=[2020, 2021],
            sample_games=150,
            category="schedule",
        ),
        # gamma_battery: terminal (refuted) -- excluded from unresolved/shared pool.
        gamma_refuted=_weak_signal(
            family="gamma_battery",
            classification="refuted_mechanism",
            closing_ground="wrong_sign_resolved",
            classification_evidence="whole interval below zero",
            effect=-1.5,
            interval=[-2.0, -1.0],
            seasons=[2012, 2014],
            probability_positive=0.02,
        ),
        # delta_battery: CFB league, no category -- tests league filter + unknown source.
        delta_cfb=_weak_signal(
            family="delta_battery",
            league="cfb",
            effect=0.3,
            probability_positive=0.70,
            seasons=[2015, 2017],
        ),
        # A family name that matches a verified FAMILY_SOURCE_RULES prefix.
        epsilon_public_betting=_weak_signal(
            family="public_betting_test_cell",
            effect=0.1,
            probability_positive=0.50,
            seasons=[2020, 2021],
            category="market",
        ),
        # Ties at probability_positive=0.70, different rotation-capacity families,
        # to pin next_shots' secondary sort key.
        eta_tied_a=_weak_signal(
            family="fam_delta",
            effect=0.4,
            probability_positive=0.70,
            seasons=[2013, 2014],
        ),
        theta_tied_b=_weak_signal(
            family="fam_epsilon",
            effect=0.4,
            probability_positive=0.70,
            seasons=[2013, 2014],
        ),
        # No probability_positive recorded -- must sort last, never treated as 0.
        iota_no_pp=_weak_signal(
            family="iota_battery",
            effect=0.4,
            probability_positive=None,
            standard_error=None,
            seasons=[2013, 2014],
        ),
    )


# ---------------------------------------------------------------------------
# Synthetic rotation registry
# ---------------------------------------------------------------------------


def _rotation_payload(**families: dict[str, Any]) -> dict[str, Any]:
    return {"version": rotation.ROTATION_REGISTRY_VERSION, "notes": [], "families": families}


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


def _rotation_window(**overrides: Any) -> dict[str, Any]:
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


def _write_rotation(tmp_path: Path, **families: dict[str, Any]) -> rotation.Registry:
    registry = rotation.registry_from_payload(_rotation_payload(**families))
    destination = tmp_path / "rotation_registry.json"
    rotation.save_registry(registry, destination)
    return rotation.load_registry(destination)


def _synthetic_rotation_registry(tmp_path: Path) -> rotation.Registry:
    return _write_rotation(
        tmp_path,
        # fam_alpha and fam_beta both touch 2020-2021 -- cross-family reuse,
        # both windows intersect the mined 2018-2025 era.
        fam_alpha=_rotation_family(
            acknowledges_mined_2018_2025=True,
            windows=[
                _rotation_window(
                    seasons=[2019, 2021],
                    state="spent",
                    spent_at="2026-01-02",
                    artifact="docs/fam_alpha.md",
                    verdict="unresolved",
                    probability_positive=0.6,
                    effect=0.4,
                    effect_units="accuracy_points",
                    interval=[-0.5, 1.3],
                    sample_blocks=10,
                )
            ],
        ),
        fam_beta=_rotation_family(
            acknowledges_mined_2018_2025=True,
            windows=[_rotation_window(seasons=[2020, 2022], state="assigned")],
        ),
        # fam_gamma: untouched by anyone else, pre-mined seasons -- no reuse, no discount.
        fam_gamma=_rotation_family(
            grade="nflverse_spread",
            windows=[
                _rotation_window(
                    seasons=[2012, 2014],
                    state="spent",
                    spent_at="2026-01-02",
                    artifact="docs/fam_gamma.md",
                    verdict="unresolved",
                    probability_positive=0.3,
                )
            ],
        ),
        # fam_delta_opener: a single window spends the ENTIRE opener pool
        # (2020-2025) in one shot -- zero remaining capacity, no assigned window.
        fam_delta_opener=_rotation_family(
            grade="opener",
            acknowledges_mined_2018_2025=True,
            windows=[
                _rotation_window(
                    seasons=[2020, 2025],
                    state="spent",
                    spent_at="2026-01-02",
                    artifact="docs/fam_delta.md",
                    verdict="unresolved",
                    probability_positive=0.9,
                )
            ],
        ),
        # fam_epsilon_close: a small spent window, plenty of eligible capacity left.
        fam_epsilon_close=_rotation_family(
            grade="close",
            windows=[
                _rotation_window(
                    seasons=[2011, 2013],
                    state="spent",
                    spent_at="2026-01-02",
                    artifact="docs/fam_epsilon.md",
                    verdict="unresolved",
                    probability_positive=0.4,
                )
            ],
        ),
    )


# ---------------------------------------------------------------------------
# (a) unresolved
# ---------------------------------------------------------------------------


def test_unresolved_signals_filters_and_excludes_terminal(tmp_path: Path) -> None:
    registry = _synthetic_weak_registry(tmp_path)

    all_nfl = registry_explorer.unresolved_signals(registry, league="nfl")
    names = {row["name"] for row in all_nfl}
    assert "gamma_refuted" not in names  # terminal classification excluded
    assert "delta_cfb" not in names  # wrong league excluded

    cfb_only = registry_explorer.unresolved_signals(registry, league="cfb")
    assert [row["name"] for row in cfb_only] == ["delta_cfb"]

    by_family = registry_explorer.unresolved_signals(registry, family="alpha_battery")
    assert {row["name"] for row in by_family} == {"alpha_high", "alpha_low"}

    # Sorted by probability_positive descending; missing values sort last.
    ordered = registry_explorer.unresolved_signals(registry, league="nfl")
    probabilities = [row["probability_positive"] for row in ordered]
    present = [p for p in probabilities if p is not None]
    assert present == sorted(present, reverse=True)
    assert ordered[-1]["name"] == "iota_no_pp"

    row = next(row for row in ordered if row["name"] == "alpha_high")
    assert row["interval"] is None
    assert row["seasons"] == [2015, 2017]
    assert row["family"] == "alpha_battery"


# ---------------------------------------------------------------------------
# (b) repeated windows
# ---------------------------------------------------------------------------


def test_repeated_windows_reports_cross_family_reuse_and_mined_discount(tmp_path: Path) -> None:
    registry = _synthetic_rotation_registry(tmp_path)
    report = registry_explorer.repeated_windows(registry)

    multi = {row["season"]: row for row in report["multi_family_seasons"]}
    # fam_alpha spent [2019,2021]; fam_beta holds (assigned) [2020,2022]; fam_delta_opener
    # spent [2020,2025]: 2020 and 2021 are touched by all three; 2019 by fam_alpha alone.
    assert 2020 in multi
    assert 2021 in multi
    assert multi[2020]["families"] == ["fam_alpha", "fam_beta", "fam_delta_opener"]
    assert multi[2020]["family_count"] == 3
    assert 2019 not in multi
    # fam_gamma [2012,2014] and fam_epsilon_close [2011,2013] share 2012-2013;
    # 2014 is fam_gamma alone, 2011 is fam_epsilon_close alone.
    assert multi[2012]["families"] == ["fam_epsilon_close", "fam_gamma"]
    assert 2014 not in multi
    assert 2011 not in multi

    mined_families = {row["family"] for row in report["mined_era_windows"]}
    # fam_alpha/fam_beta/fam_delta_opener all intersect 2018-2025; fam_gamma/fam_epsilon do not.
    assert mined_families == {"fam_alpha", "fam_beta", "fam_delta_opener"}
    for row in report["mined_era_windows"]:
        assert row["acknowledges_mined_2018_2025"] is True

    assert "rule 4" in report["reuse_discount_rule"]
    assert "Rule 6" in report["reuse_discount_rule"]


# ---------------------------------------------------------------------------
# (c) shared populations
# ---------------------------------------------------------------------------


def test_shared_population_groups_bounds_effective_sample_size(tmp_path: Path) -> None:
    registry = _synthetic_weak_registry(tmp_path)
    report = registry_explorer.shared_population_groups(registry, league="nfl")

    groups = {group["family"]: group for group in report["groups"]}
    assert "alpha_battery" in groups
    assert "beta_battery" not in groups  # solo member, nothing to overlap with

    alpha = groups["alpha_battery"]
    assert set(alpha["members"]) == {"alpha_high", "alpha_low"}
    assert alpha["shared_seasons"] == [2015, 2018]
    ess = alpha["effective_sample_size_games"]
    assert ess["naive_sum_upper_bound"] == 500  # 300 + 200, the overstated bound
    assert ess["max_single_member_lower_bound"] == 300  # the conservative floor
    assert ess["naive_sum_upper_bound"] >= ess["max_single_member_lower_bound"]

    # The canonical pool function's own output is included, unmodified, for traceability.
    assert "within_family" in report["pool_summary"]
    assert report["pool_summary"]["families_with_internal_overlap"] >= 1


# ---------------------------------------------------------------------------
# (d) source availability
# ---------------------------------------------------------------------------


def test_source_availability_cites_rules_and_falls_back_honestly(tmp_path: Path) -> None:
    registry = _synthetic_weak_registry(tmp_path)
    rows = {row["family"]: row for row in registry_explorer.source_availability(registry)}

    # Direct, citation-backed family rule match.
    betting = rows["public_betting_test_cell"]
    assert betting["status"] == "captured_scheduled"
    assert betting["citation"] is not None
    assert "capture_scheduler.py" in betting["citation"]

    # No family rule, but category='schedule' gives a labelled, lower-confidence read.
    beta = rows["beta_battery"]
    assert beta["status"] == "inferred_from_category"
    assert "not verified for this specific family" in beta["detail"]

    # No family rule and no category at all -> honestly unknown, never guessed.
    delta = rows["delta_battery"]
    assert delta["status"] == "unknown"
    assert delta["citation"] is None

    # Every non-unknown row must carry a human-checkable citation or an
    # explicit "not verified" disclaimer -- never a bare assertion.
    for row in rows.values():
        if row["status"] not in ("unknown", "inferred_from_category", "not_applicable_modeling"):
            assert row["citation"], row


def test_family_source_rules_all_carry_a_citation() -> None:
    """Every hardcoded rule must be traceable back to something read this session."""

    for rule in registry_explorer.FAMILY_SOURCE_RULES:
        assert rule.citation.strip()
        assert rule.status in {
            "captured_scheduled",
            "paused_scheduled",
            "derived_no_separate_capture",
            "bulk_ingest_unscheduled",
            "mixed",
        }


# ---------------------------------------------------------------------------
# (e) next shots
# ---------------------------------------------------------------------------


def test_next_shots_orders_by_probability_then_rotation_capacity(tmp_path: Path) -> None:
    weak_registry = _synthetic_weak_registry(tmp_path)
    rot_registry = _synthetic_rotation_registry(tmp_path)

    rows = registry_explorer.next_shots(weak_registry, rot_registry, league="nfl")
    by_name = {row["name"]: row for row in rows}

    # Primary key: probability_positive descending.
    assert rows[0]["name"] == "alpha_high"  # 0.85, the highest in the nfl pool

    # fam_delta (-> fam_delta_opener) has spent its ENTIRE opener pool: no capacity.
    assert by_name["eta_tied_a"]["matching_rotation_families"] == ["fam_delta_opener"]
    assert by_name["eta_tied_a"]["unspent_rotation_window"] is False

    # fam_epsilon (-> fam_epsilon_close) has plenty of eligible capacity left.
    assert by_name["theta_tied_b"]["matching_rotation_families"] == ["fam_epsilon_close"]
    assert by_name["theta_tied_b"]["unspent_rotation_window"] is True

    # Tied probability_positive (0.70): the one with an unspent window ranks first.
    tied_a_index = rows.index(by_name["eta_tied_a"])
    tied_b_index = rows.index(by_name["theta_tied_b"])
    assert tied_b_index < tied_a_index

    # No matching rotation family at all -> None, never a guessed False.
    assert by_name["beta_solo"]["matching_rotation_families"] is None
    assert by_name["beta_solo"]["unspent_rotation_window"] is None

    # Overlap group id is attached for alpha_battery's two members, absent for a solo entry.
    assert by_name["alpha_high"]["overlap_group_id"] is not None
    assert by_name["alpha_high"]["overlap_group_id"] == by_name.get("alpha_low", {}).get(
        "overlap_group_id"
    )
    assert by_name["beta_solo"]["overlap_group_id"] is None

    # No probability_positive recorded -> sorts last, not coerced to 0.0.
    assert rows[-1]["name"] == "iota_no_pp"

    # top= truncates without changing order.
    top2 = registry_explorer.next_shots(weak_registry, rot_registry, league="nfl", top=2)
    assert [row["name"] for row in top2] == [row["name"] for row in rows[:2]]


# ---------------------------------------------------------------------------
# Live registries: structural-only, never writes
# ---------------------------------------------------------------------------


def test_views_run_against_live_registries_and_write_nothing() -> None:
    weak_before = LIVE_WEAK_SIGNALS.read_bytes()
    rotation_before = LIVE_ROTATION.read_bytes()

    weak_registry = weak_signals.load_registry(LIVE_WEAK_SIGNALS)
    rot_registry = rotation.load_registry(LIVE_ROTATION)

    unresolved = registry_explorer.unresolved_signals(weak_registry)
    assert isinstance(unresolved, list)
    if unresolved:
        assert set(unresolved[0]) >= {
            "name",
            "league",
            "family",
            "effect",
            "effect_units",
            "interval",
            "probability_positive",
            "seasons",
        }

    repeated = registry_explorer.repeated_windows(rot_registry)
    assert isinstance(repeated["multi_family_seasons"], list)
    assert isinstance(repeated["mined_era_windows"], list)

    shared = registry_explorer.shared_population_groups(weak_registry, league="nfl")
    assert isinstance(shared["groups"], list)
    for group in shared["groups"]:
        assert group["member_count"] == len(group["members"]) >= 2

    availability = registry_explorer.source_availability(weak_registry, league="nfl")
    assert isinstance(availability, list)
    assert availability  # the live registry is never empty
    statuses = {row["status"] for row in availability}
    assert statuses <= {
        "captured_scheduled",
        "paused_scheduled",
        "derived_no_separate_capture",
        "bulk_ingest_unscheduled",
        "mixed",
        "inferred_from_category",
        "not_applicable_modeling",
        "unknown",
    }

    shots = registry_explorer.next_shots(weak_registry, rot_registry, league="nfl", top=15)
    assert isinstance(shots, list)
    assert len(shots) <= 15
    ranked_probabilities = [
        row["probability_positive"] for row in shots if row["probability_positive"] is not None
    ]
    assert ranked_probabilities == sorted(ranked_probabilities, reverse=True)

    assert LIVE_WEAK_SIGNALS.read_bytes() == weak_before
    assert LIVE_ROTATION.read_bytes() == rotation_before


def test_json_serializable_against_live_registries() -> None:
    """Every view's output must round-trip through JSON for the CLI's --json flag."""

    weak_registry = weak_signals.load_registry(LIVE_WEAK_SIGNALS)
    rot_registry = rotation.load_registry(LIVE_ROTATION)

    payloads = [
        registry_explorer.unresolved_signals(weak_registry, league="nfl"),
        registry_explorer.repeated_windows(rot_registry),
        registry_explorer.shared_population_groups(weak_registry, league="nfl"),
        registry_explorer.source_availability(weak_registry, league="nfl"),
        registry_explorer.next_shots(weak_registry, rot_registry, league="nfl", top=15),
    ]
    for payload in payloads:
        json.dumps(payload)  # must not raise
