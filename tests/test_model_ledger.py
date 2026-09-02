from __future__ import annotations

import json
import re
from dataclasses import replace
from html import unescape
from pathlib import Path

import pytest

from nfl_ats.dashboard.findings_content import (
    LEDGER_PROMOTED_CAVEAT,
    PLAYED_CARD_EXPECTATION_PERCENT,
)
from nfl_ats.model_ledger import (
    Agreement,
    LedgerError,
    ModelLedger,
    build_and_render,
    build_model_ledger,
    render_ledger_html,
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
            "mod08_hot_signal": {
                "classification": "unresolved_below_power",
                "effect": 0.9,
                "probability_positive": 0.998,
                "interval": [0.2, 1.6],
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
                "challenger_id": "hot_signal_arm",
                "status": "ACTIVE_PROSPECTIVE",
                "evidence": {
                    "registry_source": ["registry/weak_signals.json:mod08_hot_signal"],
                    "probability_positive": 0.998,
                },
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


#: Numerals the promoted row's pinned caveat legitimately quotes; after the
#: 2026-08-23 consolidation it is a one-sentence pointer whose ONLY numeral
#: is the frozen played-card expectation percentage (provenance in
#: nfl_ats.dashboard.findings_content, guarded by
#: tests/test_played_card_expectation.py).
_PROMOTED_CAVEAT_TOKENS = {
    str(PLAYED_CARD_EXPECTATION_PERCENT),
}


def test_promoted_row_summary_carries_the_selection_caveat_and_no_other_row_does(
    ledger_paths: tuple[Path, Path, Path],
) -> None:
    """2026-08-23: the promoted card's archive score was the best of 127
    subsets, so its summary must state the selection caveat verbatim; every
    challenger row stays untouched."""

    challengers, weak, manifest = ledger_paths
    ledger = build_model_ledger(challengers, weak, manifest)
    promoted = [row for row in ledger.rows if row.status_badge == "PROMOTED"]
    assert len(promoted) == 1
    assert promoted[0].summary_sentence.endswith(LEDGER_PROMOTED_CAVEAT.removesuffix(".") + ".")
    for row in ledger.rows:
        if row is promoted[0]:
            continue
        assert LEDGER_PROMOTED_CAVEAT.removesuffix(".") not in row.summary_sentence


def test_promoted_row_summary_quotes_no_track_record_percentage(
    ledger_paths: tuple[Path, Path, Path],
) -> None:
    """2026-08-23 consolidation law (owner directive): the played card's ledger
    row must not re-quote its own track record -- the picks page carries the
    one expectation number and the collapsed ladder carries the history, so the
    only percentage the promoted summary may contain is the caveat's."""

    challengers, weak, manifest = ledger_paths
    ledger = build_model_ledger(challengers, weak, manifest)
    promoted = next(row for row in ledger.rows if row.status_badge == "PROMOTED")
    assert promoted.summary_sentence.count("%") == 1
    assert "track record" not in promoted.summary_sentence
    assert "interval [" not in promoted.summary_sentence


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
    registered = json.loads(challengers.read_text(encoding="utf-8"))["challengers"]
    # One promoted-card row plus every registered challenger, including arms
    # that are superseded, deactivated, or closed before activation.
    assert len(ledger.rows) == len(registered) + 1
    assert ledger.rows[0].status_badge == "PROMOTED"
    assert ledger.rows[0].track_record is not None
    assert ledger.rows[0].track_record.games == 2075
    superseded = [r for r in ledger.rows if r.status_badge == "SUPERSEDED"]
    assert len(superseded) == 4
    validate_ledger(ledger)


# ---------------------------------------------------------------------------
# render_ledger_html
# ---------------------------------------------------------------------------

_TOKEN_RE = re.compile(r"\d[\d,]*(?:\.\d+)?")

_EXPECTED_BADGE_PAIRS = {
    "PROMOTED": "\u2713",
    "CHALLENGER": "\u25b2",
    "SUPERSEDED": "\u2014",
    "RETIRED": "\u2014",
}


def _float_variants(value: float) -> set[str]:
    variants = {
        repr(value),
        f"{value:.1f}",
        f"{value:.2f}",
        f"{value:.3f}",
        f"{value:.4f}",
        f"{value:+.2f}",
        f"{value:+.3f}",
        f"{value * 100:.1f}",
        f"{value * 100:.3f}",
        f"{value:.1%}",
        f"{value:.3%}",
        f"{abs(value):.1f}",
        f"{abs(value):.2f}",
        f"{abs(value):.3f}",
        f"{abs(value):.4f}",
        f"{abs(value) * 100:.1f}",
        f"{abs(value) * 100:.3f}",
    }
    if float(value).is_integer():
        variants.update({str(int(value)), f"{int(value):,}"})
    return variants


def _audit_ledger_numbers(rendered: str, ledger: ModelLedger) -> None:
    visible = re.sub(r"<[^>]+>", " ", unescape(rendered))
    allowed: set[str] = set()
    identifiers: list[str] = []
    for row in ledger.rows:
        identifiers.extend((row.arm_id, row.display_name))
        track = row.track_record
        if track is not None:
            if track.games is not None:
                allowed.update({str(track.games), f"{track.games:,}"})
            if track.accuracy is not None:
                allowed.update(_float_variants(track.accuracy))
            for bound in (track.interval_low, track.interval_high):
                if bound is not None:
                    allowed.update(_float_variants(bound))
        if row.own_probability_positive is not None:
            # The summary quotes this P+ when no registry entry carries one
            # ("registered evidence P+ x.xxx"), so it must be traceable too.
            allowed.update(_float_variants(row.own_probability_positive))
        for ref in row.evidence:
            identifiers.append(ref.registry_key)
            if ref.classification is not None:
                identifiers.append(ref.classification)
            if ref.effect is not None:
                allowed.update(_float_variants(ref.effect))
            if ref.probability_positive is not None:
                allowed.update(_float_variants(ref.probability_positive))
        allowed.add(str(len(row.evidence)))
        if row.agreement is not None:
            allowed.update(
                {
                    str(row.agreement.vs_promoted_games),
                    str(row.agreement.agree),
                    str(row.agreement.disagree),
                }
            )
    for identifier in identifiers:
        allowed.update(re.findall(r"\d+", identifier))
    total_markers = sum(len(row.evidence) for row in ledger.rows)
    allowed.update(str(i) for i in range(total_markers + 1))
    allowed.update({"0.99", "0.01"})
    allowed.update(_PROMOTED_CAVEAT_TOKENS)
    for raw in _TOKEN_RE.findall(visible):
        token = raw.replace(",", "")
        assert token in allowed, f"untraceable numeral {raw!r} in rendered HTML"


def test_render_ledger_html_is_byte_deterministic(ledger_paths: tuple[Path, Path, Path]) -> None:
    challengers, weak, manifest = ledger_paths
    first = render_ledger_html(build_model_ledger(challengers, weak, manifest))
    second = render_ledger_html(build_model_ledger(challengers, weak, manifest))
    assert first.encode("utf-8") == second.encode("utf-8")


def test_render_ledger_html_escapes_hostile_arm_names(tmp_path: Path) -> None:
    registry = _registry_payload()
    challengers_payload = _challengers_payload()
    challengers_payload["challengers"][0]["challenger_id"] = "<script>alert(1)</script>"
    weak = _write_json(tmp_path / "weak_signals.json", registry)
    broken = _write_json(tmp_path / "challengers.json", challengers_payload)
    manifest = _write_json(tmp_path / "active_ats_model.json", _manifest_payload())
    rendered = render_ledger_html(build_model_ledger(broken, weak, manifest))
    assert "<script" not in rendered
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in rendered


def test_render_ledger_html_promoted_first_and_badges_carry_glyph_and_text(
    ledger_paths: tuple[Path, Path, Path],
) -> None:
    challengers, weak, manifest = ledger_paths
    ledger = build_model_ledger(challengers, weak, manifest)
    rendered = render_ledger_html(ledger)
    assert rendered.index('<tr class="row-promoted">') < rendered.index("badge-challenger")
    for badge, glyph in _EXPECTED_BADGE_PAIRS.items():
        if badge in {row.status_badge for row in ledger.rows}:
            pair = (
                f'<span class="badge badge-{_badge_kind(badge)}">'
                f'<span class="badge-glyph">{glyph}</span>{badge}</span>'
            )
            assert pair in rendered
    assert ">—</td>" in rendered


def _badge_kind(badge: str) -> str:
    return {"PROMOTED": "promoted", "CHALLENGER": "challenger"}.get(badge, "muted")


def test_render_ledger_html_rejects_unknown_css_mode(
    ledger_paths: tuple[Path, Path, Path],
) -> None:
    challengers, weak, manifest = ledger_paths
    ledger = build_model_ledger(challengers, weak, manifest)
    with pytest.raises(LedgerError, match="css_mode"):
        render_ledger_html(ledger, css_mode="inline")


def test_render_ledger_html_number_audit_passes_on_live_artifacts() -> None:
    challengers = Path("artifacts/prospective/challengers.json")
    weak = Path("registry/weak_signals.json")
    manifest = Path("artifacts/active_ats_model.json")
    if not (challengers.is_file() and weak.is_file() and manifest.is_file()):
        pytest.skip("live artifacts absent")
    ledger = build_model_ledger(challengers, weak, manifest)
    rendered = build_and_render(challengers, weak, manifest)
    assert rendered.count("<tr") == len(ledger.rows) + 1
    _audit_ledger_numbers(rendered, ledger)


def test_build_and_render_matches_manual_pipeline(
    ledger_paths: tuple[Path, Path, Path],
) -> None:
    challengers, weak, manifest = ledger_paths
    ledger = build_model_ledger(challengers, weak, manifest)
    validate_ledger(ledger)
    expected = render_ledger_html(ledger)
    assert build_and_render(challengers, weak, manifest) == expected


def test_render_ledger_html_has_plain_headers_and_no_self_narration(
    ledger_paths: tuple[Path, Path, Path],
) -> None:
    challengers, weak, manifest = ledger_paths
    rendered = render_ledger_html(build_model_ledger(challengers, weak, manifest))
    assert "sort-glyph" not in rendered
    assert "decorative" not in rendered
    assert "carries the" not in rendered
    assert "PROMOTED badge" not in rendered


def test_render_ledger_html_promoted_row_leads_with_name_and_hashes_into_title(
    ledger_paths: tuple[Path, Path, Path],
) -> None:
    """B12: the promoted row must lead with its plain name; the raw model id
    lives in the title attribute, never as the visible headline."""

    challengers, weak, manifest = ledger_paths
    ledger = build_model_ledger(challengers, weak, manifest)
    rendered = render_ledger_html(ledger)
    assert "Played card \u2014 model + fix-up rules" in rendered
    assert "weak_stack" not in rendered
    assert 'title="promoted:abc123def4567890"' in rendered
    assert "Active model abc123def4567890" not in rendered


def test_render_ledger_html_floors_extreme_p_plus_honestly(
    ledger_paths: tuple[Path, Path, Path],
) -> None:
    """B10: a computed P+ of 0.998+ must display as ">0.99", never a fake
    "1.00" -- and always adjacent to n."""

    challengers, weak, manifest = ledger_paths
    rendered = render_ledger_html(build_model_ledger(challengers, weak, manifest))
    assert "P+ >0.99 over n=" in rendered
    assert "P+ 1.00" not in rendered


# ---------------------------------------------------------------------------
# 2026-08-24 dimension-3 fix: every interval row carries a P+ cell
# ---------------------------------------------------------------------------


def _pplus_fixtures(tmp_path: Path) -> tuple[Path, Path, Path]:
    """Ledger fixtures mirroring the three real rows the baseline flagged:
    intervals rendered with no P+ anywhere because their ``registry_source``
    names no weak_signals key (or none at all), while their own evidence
    block carries a measured ``probability_positive``."""

    registry = _write_json(
        tmp_path / "weak_signals.json",
        {
            "signals": {
                "mod08_smooth_cdf_mapping": {
                    "classification": "unresolved_below_power",
                    "effect": 0.684,
                    "probability_positive": 0.8666,
                    "interval": [-0.444, 1.841],
                }
            }
        },
    )
    challengers = _write_json(
        tmp_path / "challengers.json",
        {
            "challengers": [
                {
                    "challenger_id": "stack_no_registry_source",
                    "status": "ACTIVE_PROSPECTIVE",
                    "evidence": {
                        "candidate_accuracy_at_opener": 0.5329,
                        "week_blocked_interval_points": [-1.1, 5.0],
                        "probability_positive": 0.8745,
                    },
                },
                {
                    "challenger_id": "scratchpad_source_with_own_p_plus",
                    "status": "ACTIVE_PROSPECTIVE",
                    "evidence": {
                        "registry_source": "scratchpad/bestpick_opener/results.md",
                        "interval_points": [-3.92, 11.76],
                        "probability_positive": 0.813,
                    },
                },
                {
                    "challenger_id": "nested_own_p_plus",
                    "status": "ACTIVE_PROSPECTIVE",
                    "evidence": {
                        "registry_source": (
                            "docs/movement_attribution.md (a prose citation), "
                            "registry/weak_signals.json observed_movement_family"
                        ),
                        "interval_points": [0.79, 31.67],
                        "pop_threshold_cell": {
                            "probability_positive": 0.976,
                            "interval_points": [0.79, 31.67],
                        },
                    },
                },
                {
                    # A row whose evidence block declares NO probability at
                    # all: the interval must still say P+ is unavailable.
                    "challenger_id": "interval_without_any_probability",
                    "status": "ACTIVE_PROSPECTIVE",
                    "evidence": {"week_blocked_interval_points": [-1.1, 5.0]},
                },
            ]
        },
    )
    manifest = _write_json(tmp_path / "active_ats_model.json", _manifest_payload())
    return challengers, registry, manifest


def test_interval_rows_render_a_p_plus_cell(
    tmp_path: Path,
) -> None:
    """Dimension-3 contract: any challenger row that renders an accuracy-points
    interval also renders a P+ marker beside it -- measured when available,
    an explicit em dash when not -- so an interval can never sit bare again.
    The promoted row's interval is a season accuracy-proportion CI, not an
    accuracy-points effect interval, and is exempt."""

    challengers, weak, manifest = _pplus_fixtures(tmp_path)
    rendered = render_ledger_html(build_model_ledger(challengers, weak, manifest))
    rows = [r for r in re.findall(r"<tr[^>]*>.*?</tr>", rendered) if "<th>" not in r]
    assert len(rows) == 5  # promoted + four challengers
    checked = 0
    for row_html in rows:
        if 'class="row-promoted"' in row_html:
            continue  # proportion-CI interval, no accuracy-points P+
        cells = re.findall(r"<td>(.*?)</td>", row_html, flags=re.DOTALL)
        assert len(cells) == 6
        interval_cell = unescape(cells[3])
        if "[" not in interval_cell:
            continue
        assert "P+" in interval_cell, f"bare interval cell: {interval_cell!r}"
        checked += 1
    assert checked == 4


def test_own_evidence_probability_fills_the_registry_gap(
    tmp_path: Path,
) -> None:
    """A challenger whose registry_source links no weak_signals key still has
    its measured probability_positive surfaced: beside the interval, in the
    summary sentence, and in the markdown table."""

    challengers, weak, manifest = _pplus_fixtures(tmp_path)
    ledger = build_model_ledger(challengers, weak, manifest)
    by_arm = {row.arm_id: row for row in ledger.rows}

    direct = by_arm["stack_no_registry_source"]
    assert direct.own_probability_positive == pytest.approx(0.8745)
    scratchpad = by_arm["scratchpad_source_with_own_p_plus"]
    assert scratchpad.own_probability_positive == pytest.approx(0.813)
    nested = by_arm["nested_own_p_plus"]
    assert nested.own_probability_positive == pytest.approx(0.976)

    validate_ledger(ledger)

    rendered = render_ledger_html(ledger)
    assert '[0.790, 31.670] \u00b7 <span class="fine">P+ 0.98</span>' in rendered
    assert '[-1.100, 5.000] \u00b7 <span class="fine">P+ 0.87</span>' in rendered
    assert '[-3.920, 11.760] \u00b7 <span class="fine">P+ 0.81</span>' in rendered
    assert "registered evidence P+ 0.875" in rendered

    markdown = render_markdown_table(ledger)
    assert "[-1.100, 5.000] \u00b7 P+ 0.87" in markdown


def test_interval_without_measurable_probability_states_it(
    tmp_path: Path,
) -> None:
    """When no measured P+ exists anywhere in the row's data, the interval
    cell says so explicitly rather than silently omitting it."""

    challengers, weak, manifest = _pplus_fixtures(tmp_path)
    rendered = render_ledger_html(build_model_ledger(challengers, weak, manifest))
    assert '[-1.100, 5.000] \u00b7 <span class="fine">P+ \u2014</span>' in rendered


def test_own_probability_participates_in_confidence_ordering(
    tmp_path: Path,
) -> None:
    """The caption promises ordering by best-evidence P+ descending; own
    registered evidence counts toward that promise once it is surfaced."""

    challengers, weak, manifest = _pplus_fixtures(tmp_path)
    ledger = build_model_ledger(challengers, weak, manifest)
    arm_ids = [row.arm_id for row in ledger.rows[1:]]
    probabilities = {
        row.arm_id: (
            row.own_probability_positive
            if row.own_probability_positive is not None
            else float("-inf")
        )
        for row in ledger.rows[1:]
    }
    ranked = sorted(arm_ids, key=lambda a: (-probabilities[a], a))
    assert arm_ids == ranked
    # The nested 0.976 arm outranks the 0.8745 and 0.813 arms; the arm with
    # no probability sits last among its ties.
    assert arm_ids.index("nested_own_p_plus") < arm_ids.index("stack_no_registry_source")
    assert arm_ids.index("stack_no_registry_source") < arm_ids.index(
        "scratchpad_source_with_own_p_plus"
    )
