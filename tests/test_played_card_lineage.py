"""ENG-24: overlay and tiebreaker lineage on the card that is actually played.

``nfl_ats.lineage`` already has tested adapters (``overlay_sources_from_composition``,
``TiebreakerSource``) that turn a fired overlay or a tiebreaker input into a
:class:`~nfl_ats.lineage.LineageRecord`; this module tests the WIRING that was
previously missing: :func:`nfl_ats.lineage.extend_card_lineage_for_publication`
(extend a forecast's own lineage with those records) and
:func:`nfl_ats.tiebreaker.tiebreaker_lineage_sources` (the adapter from a built
``TiebreakerReport`` to ``TiebreakerSource`` records), plus one end-to-end
check that ``nfl_ats.publishing.publish_active_predictions`` actually calls
them and writes the result as ``lineage.json`` beside the published card.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from nfl_ats.lineage import (
    LINEAGE_FILENAME,
    CardLineage,
    CardLineageEntry,
    LineageError,
    OverlaySource,
    TiebreakerSource,
    build_card_lineage,
    extend_card_lineage_for_publication,
    overlay_sources_from_composition,
    validate_card_lineage,
)
from nfl_ats.provenance import sha256_file
from nfl_ats.publishing import publish_active_predictions
from nfl_ats.snapshots import write_snapshot
from nfl_ats.tiebreaker import MarketConsensus, ModelView, build_report, tiebreaker_lineage_sources
from nfl_ats.totals import TotalsView

PREDICTION_TIMESTAMP = "2026-09-03T14:32:53+00:00"
FEATURE_BUILD = "2026-09-03T14:31:38+00:00"

# ---------------------------------------------------------------------------
# A minimal base lineage to extend -- mirrors tests/test_lineage.py's own
# synthetic forecast/metadata shape, kept self-contained rather than imported
# so the two test files cannot break each other by changing a shared helper.
# ---------------------------------------------------------------------------


def _base_lineage() -> CardLineage:
    forecast = pd.DataFrame(
        {
            "game_id": ["2026_01_AAA_BBB", "2026_01_CCC_DDD"],
            "season": [2026, 2026],
            "week": [1, 1],
            "gameday": ["2026-09-13", "2026-09-13"],
            "home_team": ["BBB", "DDD"],
            "away_team": ["AAA", "CCC"],
            "spread_line": [-3.5, 2.5],
            "home_cover_probability": [0.55, 0.44],
            "method": ["market_residual", "market_residual"],
        }
    )
    metadata: dict[str, Any] = {
        "created_at_utc": PREDICTION_TIMESTAMP,
        "season": 2026,
        "week": 1,
        "feature_profile": "weak_stack",
        "provenance": {
            "feature_table": {
                "path": "data/processed/game_features_weak_stack.parquet",
                "sha256": "a" * 64,
                "manifest": {"built_at_utc": FEATURE_BUILD},
            }
        },
    }
    return build_card_lineage(
        forecast,
        metadata,
        feature_columns=("spread_line", "elo_diff"),
        display_fields={"Matchup": "formatted from team columns"},
    )


# ---------------------------------------------------------------------------
# Overlay wiring: the same duck-typed composition shape tests/test_lineage.py
# already exercises for overlay_sources_from_composition, fed through the
# publish-time extension function.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _FakeMember:
    member_id: str
    implementation: str
    flipped_game_ids: tuple[str, ...]


@dataclass(frozen=True)
class _FakeComposition:
    members: tuple[_FakeMember, ...]
    policy_fingerprint: str = "f" * 64
    arrest_snapshot_id: str = "20260902T120000Z"
    arrest_snapshot_fetched_at_utc: Any = "2026-09-02T12:00:00Z"


def test_one_fired_overlay_is_added_to_the_played_card_lineage() -> None:
    base = _base_lineage()
    composition = _FakeComposition(
        members=(
            _FakeMember("coach_fade", "nfl_ats.coach_fade_overlay.apply", ()),
            _FakeMember(
                "player_arrests_back_side_policy",
                "nfl_ats.player_arrests_back_side_overlay.apply",
                ("2026_01_AAA_BBB",),
            ),
        )
    )
    overlay_sources = overlay_sources_from_composition(
        composition, fallback_effective_timestamp=base.prediction_timestamp
    )

    played = extend_card_lineage_for_publication(base, overlay_sources=overlay_sources)

    assert "overlay:player_arrests_back_side_policy" in played.decision_bearing_fields()
    assert "overlay:coach_fade" not in played.decision_bearing_fields()
    # The base card's own fields are untouched -- this is an extension, not a
    # rebuild.
    assert set(base.decision_bearing_fields()).issubset(played.decision_bearing_fields())
    validate_card_lineage(played)  # does not raise


def test_zero_fired_overlays_add_no_overlay_records() -> None:
    base = _base_lineage()
    composition = _FakeComposition(
        members=(_FakeMember("coach_fade", "nfl_ats.coach_fade_overlay.apply", ()),)
    )
    overlay_sources = overlay_sources_from_composition(
        composition, fallback_effective_timestamp=base.prediction_timestamp
    )

    played = extend_card_lineage_for_publication(base, overlay_sources=overlay_sources)

    assert overlay_sources == ()
    assert played.entries == base.entries
    assert not any(field.startswith("overlay:") for field in played.decision_bearing_fields())


def test_prediction_timestamp_advances_to_the_publish_instant() -> None:
    base = _base_lineage()
    publish_instant = datetime(2026, 9, 5, 12, 0, tzinfo=UTC)

    played = extend_card_lineage_for_publication(base, prediction_timestamp=publish_instant)

    assert played.prediction_timestamp == "2026-09-05T12:00:00+00:00"
    assert base.prediction_timestamp == PREDICTION_TIMESTAMP  # base is unmodified


# ---------------------------------------------------------------------------
# Tiebreaker wiring
# ---------------------------------------------------------------------------


def _schedules_row() -> pd.Series:
    return pd.Series({"game_id": "2026_01_DEN_KC", "home_team": "KC", "away_team": "DEN"})


def _finals() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "home_score": [24.0, 20.0, 30.0],
            "away_score": [20.0, 23.0, 13.0],
            "spread_line": [3.0, 2.5, 7.0],
            "total_line": [43.5, 44.0, 41.0],
        }
    )


def test_tiebreaker_record_present_when_configured() -> None:
    """The market consensus alone is always recordable; a snapshot id embedded
    in its ``source`` (the real ``snapshot_consensus`` naming convention)
    resolves to a real capture instant rather than the fallback."""

    consensus = MarketConsensus(
        game_id="2026_01_DEN_KC",
        home_expected_margin=2.5,
        total_line=43.0,
        source="snapshot 20260902T180000Z (4 books)",
    )
    report = build_report(_schedules_row(), consensus, _finals())

    sources = tiebreaker_lineage_sources(report, fallback_effective_timestamp=FEATURE_BUILD)

    assert [source.input_name for source in sources] == ["market_consensus"]
    assert sources[0].source_snapshot == "20260902T180000Z"
    assert sources[0].source_captured_at == "2026-09-02T18:00:00+00:00"
    assert sources[0].effective_timestamp_basis == "source_capture"
    assert sources[0].unknown_source_reason is None

    played = extend_card_lineage_for_publication(_base_lineage(), tiebreaker_sources=sources)
    assert "tiebreaker:market_consensus" in played.decision_bearing_fields()
    assert validate_card_lineage(played)


def test_tiebreaker_adds_model_and_totals_views_when_present() -> None:
    consensus = MarketConsensus(
        game_id="2026_01_DEN_KC",
        home_expected_margin=2.5,
        total_line=43.0,
        source="schedules (fallback -- possibly stale)",
    )
    model_view = ModelView(
        predicted_margin=4.31,
        forecast_line=3.0,
        residual=1.31,
        source="forecast 2026-week-01-20260903T143253Z (market_residual)",
    )
    totals_view = TotalsView(
        predicted_total=44.2,
        market_total=43.0,
        residual=1.2,
        train_games=500,
        source=(
            "totals ridge wave 2 (65 cols, drive pace, alpha=1) trained on "
            "500 games before 2026 week 1"
        ),
    )
    report = build_report(_schedules_row(), consensus, _finals(), model_view, totals_view)

    sources = tiebreaker_lineage_sources(report, fallback_effective_timestamp=FEATURE_BUILD)

    names = [source.input_name for source in sources]
    assert names == ["market_consensus", "model_margin_view", "model_total_view"]

    consensus_source = sources[0]
    assert consensus_source.source_snapshot is None
    assert consensus_source.unknown_source_reason is not None
    assert consensus_source.effective_timestamp == FEATURE_BUILD

    margin_source = sources[1]
    assert margin_source.source_snapshot == "20260903T143253Z"
    assert margin_source.source_captured_at == "2026-09-03T14:32:53+00:00"

    totals_source = sources[2]
    assert totals_source.source_snapshot is None
    assert totals_source.unknown_source_reason is not None
    assert totals_source.effective_timestamp == FEATURE_BUILD

    played = extend_card_lineage_for_publication(_base_lineage(), tiebreaker_sources=sources)
    for name in names:
        assert f"tiebreaker:{name}" in played.decision_bearing_fields()
    assert validate_card_lineage(played)


# ---------------------------------------------------------------------------
# Fail closed
# ---------------------------------------------------------------------------


def test_safety_check_fails_closed_on_a_missing_decision_bearing_overlay_record() -> None:
    base = _base_lineage()
    composition = _FakeComposition(
        members=(
            _FakeMember(
                "player_arrests_back_side_policy",
                "nfl_ats.player_arrests_back_side_overlay.apply",
                ("2026_01_AAA_BBB",),
            ),
        )
    )
    overlay_sources = overlay_sources_from_composition(
        composition, fallback_effective_timestamp=base.prediction_timestamp
    )
    played = extend_card_lineage_for_publication(base, overlay_sources=overlay_sources)
    assert "overlay:player_arrests_back_side_policy" in played.decision_bearing_fields()

    # Blank the overlay's own record while keeping the field marked
    # decision-bearing -- the exact "declared present, but unrecordable"
    # shape a broken overlay adapter would produce.
    broken = replace(
        played,
        entries=tuple(
            CardLineageEntry(entry.card_field, True, None, None)
            if entry.card_field == "overlay:player_arrests_back_side_policy"
            else entry
            for entry in played.entries
        ),
    )

    with pytest.raises(LineageError, match="overlay:player_arrests_back_side_policy"):
        validate_card_lineage(broken)


# ---------------------------------------------------------------------------
# End-to-end: publish_active_predictions actually calls the wiring and writes
# lineage.json beside the published card (not into the forecast directory).
# ---------------------------------------------------------------------------


def _tenure_schedules() -> pd.DataFrame:
    columns = [
        "game_id",
        "season",
        "game_type",
        "week",
        "gameday",
        "home_team",
        "away_team",
        "home_coach",
        "away_coach",
        "result",
    ]
    rows = [
        ("2025_01_KEEP_OPP", 2025, "REG", 1, "2025-09-07", "KEEP", "OPP", "Steady", "OppC", 3.0),
        ("2025_01_YR1_OPP2", 2025, "REG", 1, "2025-09-07", "YR1", "OPP2", "Old1", "OppC2", -3.0),
        ("2026_01_KEEP_YR1", 2026, "REG", 1, "2026-09-10", "KEEP", "YR1", "Steady", "New1", np.nan),
    ]
    return pd.DataFrame(rows, columns=columns)


def _write_arrest_snapshot(data_root: Path, *, snapshot_id: str, fetched_at_utc: str) -> None:
    directory = data_root / "raw" / "player_arrests" / snapshot_id
    directory.mkdir(parents=True, exist_ok=True)
    incidents = pd.DataFrame(columns=["record_id", "incident_date", "team"])
    safe = directory / "incidents_point_in_time.parquet"
    incidents.to_parquet(safe, index=False)
    manifest = {
        "snapshot_id": snapshot_id,
        "fetched_at_utc": fetched_at_utc,
        "complete": True,
        "rows_cached": 0,
        "point_in_time_policy": {"safe_index": safe.name},
        "files": {safe.name: sha256_file(safe)},
    }
    (directory / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


def _write_played_card_fixture(root: Path) -> tuple[Path, Path]:
    """One coach-fade-eligible game (KEEP hosting YR1's new coach) and NO
    ``lineage.json`` for the forecast -- exercises the "build a fresh base
    lineage" fallback path in the same test as the overlay wiring."""

    forecast = root / "margin_predictions" / "forecast"
    forecast.mkdir(parents=True)
    metadata = {
        "active_model_id": "model-123",
        "synchronization_status": "SYNCHRONIZED",
        "season": 2026,
        "week": 1,
        "feature_profile": "player",
    }
    (forecast / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    pd.DataFrame(
        {
            "game_id": ["2026_01_KEEP_YR1"],
            "season": [2026],
            "week": [1],
            "game_type": ["REG"],
            "gameday": ["2026-09-10"],
            "kickoff": ["2026-09-10T17:00:00+00:00"],
            "away_team": ["YR1"],
            "home_team": ["KEEP"],
            "spread_line": [-3.5],
            "home_cover_probability": [0.35],
            "bet_side": ["AWAY"],
            "edge": [0.15],
            "method": ["market_residual"],
        }
    ).to_csv(forecast / "recommendations.csv", index=False)
    active = {
        "version": 1,
        "status": "SYNCHRONIZED",
        "target": "ats_classification",
        "model_id": "model-123",
        "method": "market_residual",
        "feature_profile": "player",
        "regressor": "ridge",
        "historical_evaluation": {
            "artifact": "margins/evaluation",
            "accuracy": 0.5205,
            "correct": 1080,
            "games": 2075,
            "intervals": {"week": {"lower": 0.4985, "upper": 0.5425}},
        },
        "weekly_forecast": {
            "artifact": "margin_predictions/forecast",
            "season": 2026,
            "week": 1,
        },
    }
    (root / "active_ats_model.json").write_text(json.dumps(active), encoding="utf-8")
    readme = root / "README.md"
    readme.write_text("# Project\n\nDescription.\n\n## Details\n", encoding="utf-8")

    data_root = root / "data"
    write_snapshot(
        _tenure_schedules(),
        pd.DataFrame({"game_id": [], "team": []}),
        seasons=[2025, 2026],
        raw_root=data_root / "raw",
    )
    return forecast, readme


def test_publish_writes_played_card_lineage_beside_the_card_not_the_forecast(
    tmp_path: Path,
) -> None:
    forecast, readme = _write_played_card_fixture(tmp_path)
    data_root = tmp_path / "data"
    instant = datetime(2026, 9, 8, 16, 0, tzinfo=UTC)
    _write_arrest_snapshot(
        data_root, snapshot_id="20260908T150000Z", fetched_at_utc="2026-09-08T15:00:00+00:00"
    )
    destination = tmp_path / "CURRENT_PREDICTIONS.md"

    result = publish_active_predictions(
        tmp_path,
        destination=destination,
        readme_path=readme,
        data_root=data_root,
        published_at=instant,
    )

    assert result["overlay_flipped_game_ids"] == ["2026_01_KEEP_YR1"]
    assert result["played_card_lineage_path"] == str(destination.parent / LINEAGE_FILENAME)
    assert result["played_card_lineage_checks_passed"]
    assert result["card_metadata"]["played_card_overlay_lineage_count"] >= 1

    # Never written into the forecast artifact directory -- that lineage.json
    # (if any) is the FORECAST's own, not the played card's.
    assert not (forecast / LINEAGE_FILENAME).exists()

    played = CardLineage.from_json((destination.parent / LINEAGE_FILENAME).read_text())
    fields = played.decision_bearing_fields()
    assert any(field.startswith("overlay:") and "coach_fade" in field for field in fields)
    assert played.prediction_timestamp == instant.astimezone(UTC).isoformat()


def test_overlay_source_dataclass_still_round_trips_through_the_played_card(tmp_path: Path) -> None:
    """Sanity check on the OverlaySource/TiebreakerSource shapes the publish
    path constructs, independent of the composition/tiebreaker machinery."""

    base = _base_lineage()
    overlay = OverlaySource(
        member_id="player_arrests_back_side_policy",
        builder_module="nfl_ats.player_arrests_back_side_overlay",
        builder_version="fingerprint0",
        effective_timestamp="2026-09-02T12:00:00+00:00",
        source_snapshot="20260902T120000Z",
        source_captured_at="2026-09-02T12:00:00+00:00",
        flipped_game_ids=("2026_01_AAA_BBB",),
    )
    tiebreaker = TiebreakerSource(
        input_name="market_consensus",
        builder_module="nfl_ats.tiebreaker",
        builder_version="v1",
        effective_timestamp="2026-09-02T18:00:00+00:00",
        source_snapshot="20260902T180000Z",
        source_captured_at="2026-09-02T18:00:00+00:00",
    )

    played = extend_card_lineage_for_publication(
        base, overlay_sources=(overlay,), tiebreaker_sources=(tiebreaker,)
    )
    restored = CardLineage.from_json(played.to_json())

    assert restored == played
    assert "overlay:player_arrests_back_side_policy" in restored.decision_bearing_fields()
    assert "tiebreaker:market_consensus" in restored.decision_bearing_fields()
