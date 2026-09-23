from __future__ import annotations

from datetime import UTC, datetime

import pytest

from nfl_ats.card_explanation import (
    LANGUAGE_CONTRACT,
    REFRESH_FLIPPED,
    REFRESH_LINE_MOVED,
    LanguageContractError,
    OverlayFiring,
    RefreshChangeInput,
    check_language,
    explain_pick,
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


def test_explain_pick_refresh_flip_is_reported() -> None:

    change = RefreshChangeInput(previous_pick_side="HOME", new_pick_side="AWAY", movement_delta=1.5)
    explanation = explain_pick(_row(), refresh_changes=change)
    assert explanation.refresh.status == REFRESH_FLIPPED
    assert "HOME to AWAY" in explanation.refresh.detail
    assert "Pick changed on refresh" in explanation.text


def test_explain_pick_refresh_line_move_without_a_flip() -> None:
    change = RefreshChangeInput(
        previous_pick_side="HOME", new_pick_side="HOME", movement_delta=1.25
    )
    explanation = explain_pick(_row(), refresh_changes=change)
    assert explanation.refresh.status == REFRESH_LINE_MOVED


def test_check_language_passes_for_ordinary_text() -> None:
    check_language("The model favors LA to cover this game by a small margin.")


@pytest.mark.parametrize("phrase", LANGUAGE_CONTRACT)
def test_check_language_fails_on_every_forbidden_phrase(phrase: str) -> None:
    with pytest.raises(LanguageContractError):
        check_language(f"This pick {phrase} for the bettor.")


def test_template_output_always_passes_the_language_contract() -> None:

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
        check_language(explanation.text)


def test_check_language_rejects_a_snapshot_id() -> None:
    with pytest.raises(LanguageContractError, match="snapshot id"):
        check_language("captured at snapshot 20260902T090000Z")


def test_check_language_rejects_wagering_and_research_boilerplate() -> None:
    for phrase in (
        "not a wagering recommendation",
        "this is a descriptive research summary",
        "an early, mutable research preview",
        "not proof of a profitable edge",
        "not proof of a stable edge",
    ):
        with pytest.raises(LanguageContractError):
            check_language(phrase)


def _waterfall_entry(**overrides: object) -> dict[str, object]:
    base: dict[str, object] = {
        "picked_side": "HOME",
        "steps": [
            {"kind": "family", "family": "defense", "delta_points": 1.2},
            {"kind": "family", "family": "context", "delta_points": 0.9},
            {"kind": "family", "family": "market", "delta_points": -0.3},
        ],
    }
    base.update(overrides)
    return base
