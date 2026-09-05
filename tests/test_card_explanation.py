from __future__ import annotations

from datetime import UTC, datetime

import pytest

from nfl_ats.card_explanation import (
    COMPUTED_NOW,
    LANGUAGE_CONTRACT,
    MEASURED_FROM_ARTIFACT,
    NO_DATA,
    REFRESH_FLIPPED,
    REFRESH_LINE_MOVED,
    REFRESH_NONE,
    REFRESH_NOT_YET,
    REFRESH_OVERLAY_CHANGED,
    LanguageContractError,
    OverlayFiring,
    PickExplanation,
    RefreshChangeInput,
    check_language,
    explain_card,
    explain_pick,
    from_json,
    overlay_firings_from_composition,
    refresh_change_from_pick_revision,
    render_markdown,
    to_json,
)
from nfl_ats.four_overlay_composition import (
    COACH_FADE,
    COMPOSITION_ORDER,
    POLICY_FINGERPRINT,
    POLICY_ID,
    FourOverlayCompositionResult,
    GameProvenance,
    MemberProvenance,
)
from nfl_ats.lineage import (
    FIELD_MARKET_LINE,
    FIELD_MODEL_PROBABILITY,
    CardLineage,
    CardLineageEntry,
    LineageRecord,
)
from nfl_ats.source_freshness_policy import SourcePolicyReport, SourceState

# ---------------------------------------------------------------------------
# Synthetic fixtures
# ---------------------------------------------------------------------------


def _row(**overrides: object) -> dict[str, object]:
    base: dict[str, object] = {
        "game_id": "2026_01_SF_LA",
        "home_team": "LA",
        "away_team": "SF",
        "spread_line": -3.5,
        "home_cover_probability": 0.62,
        "gameday": "2026-09-10",
    }
    base.update(overrides)
    return base


def _lineage_with_market_snapshot() -> CardLineage:
    record = LineageRecord(
        card_field=FIELD_MARKET_LINE,
        feature_family="market",
        source_snapshot="20260902T090000Z",
        source_captured_at="2026-09-02T09:00:00+00:00",
        effective_timestamp="2026-09-02T09:00:00+00:00",
        effective_timestamp_basis="source_capture",
        builder_version="v1",
        builder_module="nfl_ats.features",
    )
    probability_record = LineageRecord(
        card_field=FIELD_MODEL_PROBABILITY,
        feature_family="model_probability",
        source_snapshot="feature_table:sha256:deadbeef",
        source_captured_at=None,
        effective_timestamp="2026-09-02T09:00:00+00:00",
        effective_timestamp_basis="feature_table_build",
        builder_version="v1",
        builder_module="nfl_ats.outcomes",
    )
    return CardLineage(
        prediction_timestamp="2026-09-02T09:00:00+00:00",
        entries=(
            CardLineageEntry(FIELD_MARKET_LINE, True, record),
            CardLineageEntry(FIELD_MODEL_PROBABILITY, True, probability_record),
        ),
        season=2026,
        week=1,
    )


def _source_report(now: datetime) -> SourcePolicyReport:
    return SourcePolicyReport(
        state="complete",
        evaluated_at_utc=now.isoformat(),
        sources=(
            SourceState(
                source_id="odds_opener",
                state="complete",
                reason="snapshot is 30.0 min old, inside the 210 min budget",
                age_minutes=30.0,
                budget_minutes=210,
                fallback="publish on the newest opener snapshot on disk",
            ),
            SourceState(
                source_id="player_arrests",
                state="degraded",
                reason="snapshot is 200.0 min old, over the 90 min budget",
                age_minutes=200.0,
                budget_minutes=90,
                fallback="none -- fail-closed",
            ),
        ),
        unobserved=("injuries_sportradar",),
    )


def _composition_result(game_id: str) -> FourOverlayCompositionResult:
    member = MemberProvenance(
        member_id=COACH_FADE,
        order=0,
        implementation="nfl_ats.coach_fade_overlay.apply_coach_fade_overlay",
        enabled=True,
        status="applied",
        flipped_game_ids=(game_id,),
    )
    game = GameProvenance(
        game_id=game_id,
        member_ids=(COACH_FADE,),
        raw_home_cover_probability=0.38,
        final_home_cover_probability=0.62,
    )
    return FourOverlayCompositionResult(
        overlaid_predictions=None,  # type: ignore[arg-type]
        policy_id=POLICY_ID,
        policy_fingerprint=POLICY_FINGERPRINT,
        composition_order=COMPOSITION_ORDER,
        members=(member,),
        games=(game,),
        union_flipped_game_ids=(game_id,),
        overlapping_game_ids=(),
        arrest_snapshot_id="20260902T070000Z",
        arrest_snapshot_fetched_at_utc=None,  # type: ignore[arg-type]
        arrest_safe_index_sha256="deadbeef",
    )


# ---------------------------------------------------------------------------
# Every field present with explicit no_data states
# ---------------------------------------------------------------------------


def test_explain_pick_with_no_optional_inputs_reports_no_data_everywhere() -> None:
    explanation = explain_pick(_row())

    assert explanation.market_line.provenance == MEASURED_FROM_ARTIFACT
    assert explanation.model_probability.provenance == COMPUTED_NOW
    assert explanation.overlays.provenance == NO_DATA
    assert explanation.overlays.firings == ()
    assert explanation.freshness.provenance == NO_DATA
    assert explanation.freshness.sources == ()
    assert explanation.refresh.status == REFRESH_NOT_YET
    assert explanation.refresh.provenance == NO_DATA


def test_explain_pick_with_missing_market_and_probability_reports_no_data() -> None:
    row = _row(spread_line=None, home_cover_probability=None)
    explanation = explain_pick(row)

    assert explanation.market_line.home_spread_line is None
    assert explanation.market_line.provenance == NO_DATA
    assert explanation.model_probability.probability is None
    assert explanation.model_probability.provenance == NO_DATA
    # A missing market/probability degrades the text, never raises.
    assert "no market line is recorded" in explanation.text
    assert "No model probability is recorded" in explanation.text


def test_explain_pick_with_lineage_reports_market_snapshot() -> None:
    explanation = explain_pick(_row(), lineage=_lineage_with_market_snapshot())

    assert explanation.market_line.snapshot_id == "20260902T090000Z"
    assert explanation.market_line.snapshot_captured_at == "2026-09-02T09:00:00+00:00"
    assert "20260902T090000Z" in explanation.text


def test_explain_pick_with_source_report_reports_freshness_states() -> None:
    now = datetime(2026, 9, 2, 12, 0, tzinfo=UTC)
    explanation = explain_pick(_row(), source_report=_source_report(now))

    states = {entry.source_id: entry.state for entry in explanation.freshness.sources}
    assert states["odds_opener"] == "complete"
    assert states["player_arrests"] == "degraded"
    assert states["injuries_sportradar"] == NO_DATA
    assert explanation.freshness.provenance == MEASURED_FROM_ARTIFACT
    # as_of is recoverable for an observed source (evaluated_at - age_minutes).
    odds_entry = next(e for e in explanation.freshness.sources if e.source_id == "odds_opener")
    assert odds_entry.as_of is not None
    unobserved_entry = next(
        e for e in explanation.freshness.sources if e.source_id == "injuries_sportradar"
    )
    assert unobserved_entry.as_of is None


# ---------------------------------------------------------------------------
# Overlay fired vs not
# ---------------------------------------------------------------------------


def test_explain_pick_with_no_overlays_reports_none_fired() -> None:
    explanation = explain_pick(_row(), overlays=())

    assert explanation.overlays.firings == ()
    assert explanation.overlays.provenance == MEASURED_FROM_ARTIFACT
    assert "No overlay fired" in explanation.text


def test_explain_pick_with_a_fired_overlay_names_it_and_marks_changed() -> None:
    firing = OverlayFiring(
        name=COACH_FADE,
        direction="complemented toward LA",
        input_value="year-1 head coach matchup: SF vs LA",
    )
    explanation = explain_pick(_row(), overlays=(firing,))

    assert len(explanation.overlays.firings) == 1
    assert explanation.overlays.firings[0].changed_pick is True
    assert "coach_fade" in explanation.text
    assert "1 overlay fired" in explanation.text


def test_overlay_firings_from_composition_extracts_the_named_game() -> None:
    composition = _composition_result("2026_01_SF_LA")
    firings = overlay_firings_from_composition(composition, "2026_01_SF_LA")

    assert len(firings) == 1
    assert firings[0].name == "coach fade"
    assert firings[0].changed_pick is True
    assert "0.380" in firings[0].input_value

    assert overlay_firings_from_composition(composition, "no_such_game") == ()


# ---------------------------------------------------------------------------
# Refresh flip vs none vs not-yet
# ---------------------------------------------------------------------------


def test_explain_pick_with_no_refresh_input_is_no_refresh_yet() -> None:
    explanation = explain_pick(_row())
    assert explanation.refresh.status == REFRESH_NOT_YET


def test_explain_pick_refresh_confirms_no_change() -> None:
    change = RefreshChangeInput(previous_pick_side="HOME", new_pick_side="HOME", movement_delta=0.0)
    explanation = explain_pick(_row(), refresh_changes=change)
    assert explanation.refresh.status == REFRESH_NONE
    assert explanation.refresh.provenance == MEASURED_FROM_ARTIFACT


def test_explain_pick_refresh_flip_is_reported() -> None:
    change = RefreshChangeInput(previous_pick_side="HOME", new_pick_side="AWAY", movement_delta=1.5)
    explanation = explain_pick(_row(), refresh_changes=change)
    assert explanation.refresh.status == REFRESH_FLIPPED
    assert "HOME to AWAY" in explanation.refresh.detail
    assert "pick moved from HOME to AWAY" in explanation.text


def test_explain_pick_refresh_line_move_without_a_flip() -> None:
    change = RefreshChangeInput(
        previous_pick_side="HOME", new_pick_side="HOME", movement_delta=1.25
    )
    explanation = explain_pick(_row(), refresh_changes=change)
    assert explanation.refresh.status == REFRESH_LINE_MOVED


def test_explain_pick_refresh_overlay_change_without_flip_or_move() -> None:
    change = RefreshChangeInput(
        previous_pick_side="HOME",
        new_pick_side="HOME",
        overlays_added=("player_arrests_back_side_policy",),
    )
    explanation = explain_pick(_row(), refresh_changes=change)
    assert explanation.refresh.status == REFRESH_OVERLAY_CHANGED


def test_refresh_change_from_pick_revision_adapts_a_ledger_row() -> None:
    revision = {
        "previous_pick_side": "AWAY",
        "new_pick_side": "HOME",
        "movement_delta": 1.2,
        "movement_policy": "movement_ge_1.0",
    }
    change = refresh_change_from_pick_revision(revision)
    assert change is not None
    assert change.previous_pick_side == "AWAY"
    assert change.new_pick_side == "HOME"

    assert refresh_change_from_pick_revision(None) is None


# ---------------------------------------------------------------------------
# Language contract
# ---------------------------------------------------------------------------


def test_check_language_passes_for_ordinary_text() -> None:
    check_language("The model favors LA to cover this game by a small margin.")


@pytest.mark.parametrize("phrase", LANGUAGE_CONTRACT)
def test_check_language_fails_on_every_forbidden_phrase(phrase: str) -> None:
    with pytest.raises(LanguageContractError):
        check_language(f"This pick {phrase} for the bettor.")


def test_template_output_always_passes_the_language_contract() -> None:
    """The rendered template itself must never trip its own contract, across
    every combination of present/absent optional inputs."""

    firing = OverlayFiring(
        name=COACH_FADE, direction="complemented toward LA", input_value="year-1 coach"
    )
    now = datetime(2026, 9, 2, 12, 0, tzinfo=UTC)
    combos = [
        {},
        {"lineage": _lineage_with_market_snapshot()},
        {"source_report": _source_report(now)},
        {"overlays": (firing,)},
        {"refresh_changes": RefreshChangeInput("HOME", "AWAY", 1.5)},
        {"refresh_changes": RefreshChangeInput("HOME", "HOME", 0.0)},
        {
            "lineage": _lineage_with_market_snapshot(),
            "source_report": _source_report(now),
            "overlays": (firing,),
            "refresh_changes": RefreshChangeInput("HOME", "AWAY", 1.5),
        },
    ]
    for kwargs in combos:
        explanation = explain_pick(_row(), **kwargs)  # type: ignore[arg-type]
        check_language(explanation.text)  # explain_pick already checked; re-assert here


# ---------------------------------------------------------------------------
# JSON round-trip
# ---------------------------------------------------------------------------


def test_json_round_trip_preserves_every_field(tmp_path: object) -> None:
    firing = OverlayFiring(
        name=COACH_FADE, direction="complemented toward LA", input_value="year-1 coach"
    )
    now = datetime(2026, 9, 2, 12, 0, tzinfo=UTC)
    explanation = explain_pick(
        _row(),
        lineage=_lineage_with_market_snapshot(),
        source_report=_source_report(now),
        overlays=(firing,),
        refresh_changes=RefreshChangeInput("HOME", "AWAY", 1.5),
    )
    payload = to_json([explanation])
    restored = from_json(payload)

    assert len(restored) == 1
    round_tripped: PickExplanation = restored[0]
    assert round_tripped.to_dict() == explanation.to_dict()


# ---------------------------------------------------------------------------
# explain_card / render_markdown
# ---------------------------------------------------------------------------


def test_explain_card_keys_overlays_and_refresh_by_game_id() -> None:
    rows = [
        _row(game_id="game_a", home_team="LA", away_team="SF"),
        _row(game_id="game_b", home_team="KC", away_team="DEN", spread_line=-6.0),
    ]
    firing = OverlayFiring(name=COACH_FADE, direction="toward LA", input_value="year-1 coach")
    explanations = explain_card(
        rows,
        overlays_by_game={"game_a": (firing,)},
        refresh_changes_by_game={
            "game_b": RefreshChangeInput("HOME", "AWAY", 1.5),
        },
    )
    by_id = {explanation.game_id: explanation for explanation in explanations}
    assert len(by_id["game_a"].overlays.firings) == 1
    assert by_id["game_b"].overlays.firings == ()
    assert by_id["game_a"].refresh.status == REFRESH_NOT_YET
    assert by_id["game_b"].refresh.status == REFRESH_FLIPPED


def test_render_markdown_includes_every_pick() -> None:
    rows = [_row(game_id="game_a"), _row(game_id="game_b", home_team="KC", away_team="DEN")]
    explanations = explain_card(rows)
    markdown = render_markdown(explanations)

    assert "SF at LA" in markdown
    assert "DEN at KC" in markdown
    check_language(markdown)
