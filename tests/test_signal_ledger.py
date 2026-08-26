from __future__ import annotations

import re
from typing import Any

import pytest

from nfl_ats.four_overlay_composition import on_the_card_registry_names
from nfl_ats.public_board import render_signal_ledger_page
from nfl_ats.signal_ledger import (
    STATUS_CLOSED,
    STATUS_CONTROL,
    STATUS_ON_CARD,
    STATUS_RECORDED,
    UNCATEGORISED,
    build_ledger_rows,
    build_signal_ledger_body,
)
from nfl_ats.weak_signals import (
    WEAK_SIGNAL_REGISTRY_VERSION,
    Registry,
    signal_from_payload,
)


def _signal_payload(**overrides: Any) -> dict[str, Any]:
    body: dict[str, Any] = {
        "recorded_at": "2026-08-17",
        "description": "a raw technical description of the measurement",
        "source": "docs/example.md",
        "effect": 0.30,
        "effect_units": "accuracy_points",
        "classification": "unresolved_below_power",
        "league": "nfl",
        "seasons": [2009, 2017],
        "interval": [-0.20, 0.80],
        "probability_positive": 0.75,
        "sample_games": 1200,
    }
    body.update(overrides)
    return body


def _registry(**signals: dict[str, Any]) -> Registry:
    return Registry(
        version=WEAK_SIGNAL_REGISTRY_VERSION,
        notes=(),
        signals={name: signal_from_payload(name, body) for name, body in signals.items()},
    )


# ---------------------------------------------------------------------------
# Row construction: the honest-gaps contract
# ---------------------------------------------------------------------------


def test_row_falls_back_to_description_when_no_plain_summary_is_recorded() -> None:
    registry = _registry(alpha=_signal_payload())
    rows = build_ledger_rows(registry)
    assert len(rows) == 1
    row = rows[0]
    assert row["fallback"] is True
    assert "raw technical description" in row["idea"]
    assert row["category"] == UNCATEGORISED


def test_row_uses_the_plain_summary_when_present() -> None:
    registry = _registry(
        alpha=_signal_payload(
            plain_summary="A short sentence a fan can read on its own.",
            category="onfield",
        )
    )
    row = build_ledger_rows(registry)[0]
    assert row["fallback"] is False
    assert row["idea"] == "A short sentence a fan can read on its own."
    assert row["category"] == "onfield"


def test_null_reliability_renders_as_none_never_zero() -> None:
    registry = _registry(alpha=_signal_payload(reliability=None))
    row = build_ledger_rows(registry)[0]
    assert row["rel"] is None
    # The page must never render a null reliability as a numeric zero -- the
    # client-side script special-cases None into "not measured" (see _JS).


def test_null_interval_is_preserved_as_none_not_a_zero_width_pair() -> None:
    registry = _registry(alpha=_signal_payload(interval=None, standard_error=None))
    row = build_ledger_rows(registry)[0]
    assert row["interval"] is None


def test_non_accuracy_unit_is_not_marked_as_accuracy_and_gets_a_caveat() -> None:
    registry = _registry(
        alpha=_signal_payload(effect_units="brier", effect=-0.024, interval=[-0.030, -0.018])
    )
    row = build_ledger_rows(registry)[0]
    assert row["is_accuracy"] is False
    assert row["units"] == "brier"
    assert any("not comparable to accuracy-point rows" in flag for flag in row["flags"])


@pytest.mark.parametrize("units", ["ats_points", "brier", "log_loss", "mae"])
def test_every_non_accuracy_unit_is_flagged(units: str) -> None:
    registry = _registry(alpha=_signal_payload(effect_units=units))
    row = build_ledger_rows(registry)[0]
    assert row["is_accuracy"] is False
    assert row["flags"]


# ---------------------------------------------------------------------------
# Status derivation: only classification and category ever decide it
# ---------------------------------------------------------------------------


def test_status_recorded_is_the_default() -> None:
    registry = _registry(alpha=_signal_payload())
    assert build_ledger_rows(registry)[0]["status"] == STATUS_RECORDED


def test_status_control_comes_only_from_the_category_field() -> None:
    registry = _registry(alpha=_signal_payload(category="control"))
    assert build_ledger_rows(registry)[0]["status"] == STATUS_CONTROL


def test_status_closed_comes_only_from_a_terminal_classification() -> None:
    registry = _registry(
        alpha=_signal_payload(
            classification="refuted_mechanism",
            closing_ground="wrong_sign_resolved",
            classification_evidence="whole interval below zero",
            interval=[-0.90, -0.10],
            effect=-0.50,
        )
    )
    assert build_ledger_rows(registry)[0]["status"] == STATUS_CLOSED


# ---------------------------------------------------------------------------
# Evidence buckets (owner spec, 2026-08-26)
# ---------------------------------------------------------------------------


def test_evidence_buckets_reliability_bands() -> None:
    registry = _registry(
        strong=_signal_payload(reliability=0.75),
        weak=_signal_payload(reliability=0.05),
        never=_signal_payload(reliability=None),
        middling=_signal_payload(reliability=0.45),
    )
    rows = {row["name"]: row for row in build_ledger_rows(registry)}
    assert rows["strong"]["evidence"] == ["repeats_well"]
    assert rows["weak"]["evidence"] == ["doesnt_repeat"]
    assert rows["never"]["evidence"] == ["never_checked"]
    # Between the two bands: neither chip claims this row.
    assert rows["middling"]["evidence"] == []


def test_evidence_found_by_sweeping_reads_notes_and_description() -> None:
    registry = _registry(
        alpha=_signal_payload(notes="mined battery cell, uncorrected multiplicity"),
        beta=_signal_payload(description="a correlated decomposition of a shared window"),
        clean=_signal_payload(notes="", description="a predeclared single test"),
    )
    rows = {row["name"]: row for row in build_ledger_rows(registry)}
    assert "found_by_sweeping" in rows["alpha"]["evidence"]
    assert "found_by_sweeping" in rows["beta"]["evidence"]
    assert "found_by_sweeping" not in rows["clean"]["evidence"]


def test_evidence_axes_are_independent_a_row_can_carry_both() -> None:
    registry = _registry(
        alpha=_signal_payload(reliability=None, notes="mined, uncorrected multiplicity")
    )
    row = build_ledger_rows(registry)[0]
    assert set(row["evidence"]) == {"never_checked", "found_by_sweeping"}


# ---------------------------------------------------------------------------
# Duplicate fingerprint (owner spec, 2026-08-26): flag, never merge or drop
# ---------------------------------------------------------------------------


def test_exact_duplicate_signals_are_flagged_but_both_still_render() -> None:
    shared = {
        "interval": [-0.4137, 0.0738],
        "probability_positive": 0.0843,
        "sample_games": 119,
        "sample_blocks": 294,
        "effect": -0.1713,
    }
    registry = _registry(
        body_clock_night_dose_ge2000=_signal_payload(**shared),
        body_clock_night_west_road_ge2000et=_signal_payload(**shared),
        unrelated=_signal_payload(effect=0.9, interval=[0.1, 1.7], sample_games=50),
    )
    rows = {row["name"]: row for row in build_ledger_rows(registry)}
    assert len(rows) == 3
    assert any(
        "appears twice in the registry" in f for f in rows["body_clock_night_dose_ge2000"]["flags"]
    )
    assert any(
        "appears twice in the registry" in f
        for f in rows["body_clock_night_west_road_ge2000et"]["flags"]
    )
    assert not rows["unrelated"]["flags"]


def test_sparse_rows_missing_the_same_fields_are_not_false_positive_duplicates() -> None:
    # Two rows that both happen to lack interval/probability_positive/games
    # must never be flagged as duplicates of each other merely for sharing
    # an absence of data.
    registry = _registry(
        alpha=_signal_payload(
            interval=None,
            probability_positive=None,
            sample_games=None,
            standard_error=None,
            effect=0.5,
        ),
        beta=_signal_payload(
            interval=None,
            probability_positive=None,
            sample_games=None,
            standard_error=None,
            effect=0.5,
        ),
    )
    rows = build_ledger_rows(registry)
    for row in rows:
        assert not any("appears twice" in f for f in row["flags"])


# ---------------------------------------------------------------------------
# Page rendering: crashes never happen on edge-case data
# ---------------------------------------------------------------------------


def _edge_case_registry() -> Registry:
    return _registry(
        no_summary=_signal_payload(),
        with_summary=_signal_payload(
            plain_summary="A short, plain sentence with an ampersand & a <tag> in it.",
            category="health",
        ),
        no_interval=_signal_payload(interval=None, standard_error=None),
        no_reliability=_signal_payload(reliability=None),
        non_accuracy=_signal_payload(effect_units="mae", effect=0.013, interval=[0.005, 0.021]),
        control_arm=_signal_payload(category="control"),
        cfb_row=_signal_payload(league="cfb", seasons=[2012, 2020]),
        closed_row=_signal_payload(
            classification="refuted_mechanism",
            closing_ground="wrong_sign_resolved",
            classification_evidence="whole interval below zero",
            interval=[-0.9, -0.1],
            effect=-0.5,
        ),
    )


def test_build_signal_ledger_body_never_crashes_on_edge_case_data() -> None:
    body, script = build_signal_ledger_body(_edge_case_registry())
    assert body
    assert script
    assert "<script" in script


def test_render_signal_ledger_page_end_to_end_with_edge_case_registry() -> None:
    html = render_signal_ledger_page(weak_signal_registry=_edge_case_registry())
    assert html.startswith("<!doctype html>")
    assert html.rstrip().endswith("</html>")
    assert 'aria-current="page"' in html
    assert "Signal ledger" in html
    # A registry-authored "<tag>" must never open a real HTML tag once
    # embedded in the row payload -- it is escaped before it reaches the
    # JSON blob the client-side renderer concatenates into innerHTML.
    assert "<tag>" not in html
    assert "&lt;tag&gt;" in html


def test_page_never_draws_a_whisker_for_a_null_interval() -> None:
    html = render_signal_ledger_page(
        weak_signal_registry=_registry(alpha=_signal_payload(interval=None, standard_error=None))
    )
    assert '"interval":null' in html


def test_page_states_reliability_gap_in_words_not_digits() -> None:
    html = render_signal_ledger_page(
        weak_signal_registry=_registry(alpha=_signal_payload(reliability=None))
    )
    assert "not measured" in html


def test_page_nav_and_chrome_include_the_new_page() -> None:
    html = render_signal_ledger_page(weak_signal_registry=_edge_case_registry())
    assert 'href="index.html"' in html
    assert 'href="pool.html"' in html
    assert '<div class="ats">' in html


# ---------------------------------------------------------------------------
# Theme invariants: role tokens only, never a raw hex colour
# ---------------------------------------------------------------------------

_HEX_COLOR = re.compile(r"(?<!&)#[0-9a-fA-F]{3,8}\b")


def test_ledger_page_body_uses_no_raw_hex_colors() -> None:
    body, script = build_signal_ledger_body(_edge_case_registry())
    assert not _HEX_COLOR.search(body)
    assert not _HEX_COLOR.search(script)


def test_every_css_variable_the_ledger_chrome_references_is_defined() -> None:
    from nfl_ats.public_board import _PAGE_CHROME

    used = set(re.findall(r"var\((--[a-z0-9-]+)", _PAGE_CHROME))
    defined = set(re.findall(r"^\s*(--[a-z0-9-]+):", _PAGE_CHROME, re.M))
    assert not used - defined


def test_filter_chip_groups_match_the_owner_spec_counts() -> None:
    """Status (5 incl. All -- no dead 'Candidates' chip), Subject (11 incl.
    All + Uncategorised), Evidence (5 incl. All): the owner's chip-count
    budget after the 2026-08-26 fix dropped the undeliverable status."""

    body, _script = build_signal_ledger_body(_edge_case_registry())
    status_group = re.search(r'<span class="lbl">Status</span>(.*?)</div>', body, re.DOTALL)
    subject_group = re.search(r'<span class="lbl">Subject</span>(.*?)</div>', body, re.DOTALL)
    evidence_group = re.search(r'<span class="lbl">Evidence</span>(.*?)</div>', body, re.DOTALL)
    assert status_group and subject_group and evidence_group
    assert status_group.group(1).count("<button") == 5
    assert subject_group.group(1).count("<button") == 11
    assert evidence_group.group(1).count("<button") == 5


def test_candidate_status_and_chip_were_dropped_not_shipped_empty() -> None:
    """No code in this project distinguishes a 'candidate' from a merely
    recorded row, so the chip was removed entirely rather than rendered
    permanently empty (owner, 2026-08-26)."""

    body, _script = build_signal_ledger_body(_edge_case_registry())
    assert 'data-value="candidate"' not in body
    # No chip carries the word -- it still appears once, in the explanatory
    # note that says the status was dropped, which is the point.
    assert ">Candidate<" not in body
    assert ">Candidates<" not in body
    rows = build_ledger_rows(_edge_case_registry())
    assert all(row["status"] != "candidate" for row in rows)


def test_on_the_card_is_derived_from_the_policy_composition_mapping() -> None:
    """A row whose name is one of
    ``four_overlay_composition.MEMBER_REGISTRY_EVIDENCE``'s registry names
    resolves to ``on_the_card`` -- the fix for the 2026-08-26 report that
    this chip matched zero rows. All five names (four members, with
    division-revenge's two grades both counted) must resolve."""

    names = sorted(on_the_card_registry_names())
    assert names == [
        "bias_battery_division_revenge_game",
        "bias_battery_division_revenge_game_opener",
        "hc_year_one_fade",
        "pick_conditioned_spread_gap_zone_pre2018",
        "player_arrests_recent_14d_back_side_policy_opener",
    ]

    registry = _registry(**{name: _signal_payload() for name in names})
    rows = {row["name"]: row for row in build_ledger_rows(registry)}
    for name in names:
        assert rows[name]["status"] == STATUS_ON_CARD

    body, script = build_signal_ledger_body(registry)
    assert 'data-value="on_the_card"' in body
    assert script.count('"status":"on_the_card"') == len(names)


def test_closed_and_control_status_outrank_on_the_card_membership() -> None:
    """A definitive registry verdict (or a declared control arm) is a more
    important fact than "shares a name with a live policy member" -- even
    though no current row actually has both, the precedence must hold if one
    ever does."""

    registry = _registry(
        hc_year_one_fade=_signal_payload(
            classification="refuted_mechanism",
            closing_ground="wrong_sign_resolved",
            classification_evidence="whole interval below zero",
            interval=[-0.9, -0.1],
            effect=-0.5,
        )
    )
    row = build_ledger_rows(registry)[0]
    assert row["status"] == STATUS_CLOSED
