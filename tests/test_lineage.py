"""ENG-16: a card that cannot say where its decisions came from is not publishable."""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from nfl_ats.backtest import score_week
from nfl_ats.lineage import (
    BASE_SNAPSHOT_UNRECORDED,
    FIELD_MARKET_LINE,
    FIELD_MODEL_PROBABILITY,
    FIELD_PICK,
    LINEAGE_CHECKS,
    LINEAGE_FILENAME,
    CardLineage,
    CardLineageEntry,
    LineageError,
    LineageRecord,
    OverlaySource,
    TiebreakerSource,
    build_card_lineage,
    families_for_columns,
    is_decision_bearing,
    overlay_sources_from_composition,
    parse_snapshot_capture,
    read_card_lineage,
    validate_card_lineage,
    write_card_lineage,
)
from nfl_ats.prediction_safety import (
    PredictionSafetyError,
    validate_prediction_card,
    validate_prediction_lineage,
)

PREDICTION_TIMESTAMP = "2026-09-03T14:32:53+00:00"
FEATURE_BUILD = "2026-09-03T14:31:38+00:00"
PLAYER_SNAPSHOT = "20260817T184901Z"


def _synthetic_forecast() -> pd.DataFrame:
    return pd.DataFrame(
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
            "train_max_gameday": ["2026-01-04", "2026-01-04"],
        }
    )


def _synthetic_metadata(**overrides: Any) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "created_at_utc": PREDICTION_TIMESTAMP,
        "season": 2026,
        "week": 1,
        "feature_profile": "weak_stack",
        "active_model_id": "123d60be8c80a35d",
        "provenance": {
            "feature_table": {
                "path": "data/processed/game_features_weak_stack.parquet",
                "sha256": "a" * 64,
                "manifest": {
                    "built_at_utc": FEATURE_BUILD,
                    "source_player_snapshot": PLAYER_SNAPSHOT,
                    "player_feature_version": "v3-availability-v1",
                },
            }
        },
    }
    metadata.update(overrides)
    return metadata


def _lineage(**kwargs: Any) -> CardLineage:
    return build_card_lineage(
        _synthetic_forecast(),
        _synthetic_metadata(),
        feature_columns=kwargs.pop(
            "feature_columns",
            ("spread_line", "elo_diff", "diff_injury_offense_unavailability"),
        ),
        display_fields=kwargs.pop("display_fields", {"Matchup": "formatted from team columns"}),
        **kwargs,
    )


# ---------------------------------------------------------------------------
# Building and round-tripping
# ---------------------------------------------------------------------------


def test_synthetic_forecast_gets_lineage_for_every_decision_bearing_field(
    tmp_path: Path,
) -> None:
    lineage = _lineage()

    fields = lineage.decision_bearing_fields()
    assert {FIELD_PICK, FIELD_MODEL_PROBABILITY, FIELD_MARKET_LINE}.issubset(fields)
    assert "model_input:market" in fields
    assert "model_input:elo" in fields
    assert "model_input:player_injuries" in fields
    assert all(is_decision_bearing(field) for field in fields)

    # The player family names a real snapshot; the base families cannot, and
    # must say so rather than going quiet.
    injuries = lineage.field("model_input:player_injuries")
    assert injuries is not None and injuries.lineage is not None
    assert injuries.lineage.source_snapshot == PLAYER_SNAPSHOT
    assert injuries.lineage.source_captured_at == "2026-08-17T18:49:01+00:00"
    assert injuries.lineage.effective_timestamp_basis == "source_capture"
    assert injuries.lineage.builder_version == "v3-availability-v1"
    assert injuries.lineage.builder_module == "nfl_ats.players"

    market = lineage.field(FIELD_MARKET_LINE)
    assert market is not None and market.lineage is not None
    assert market.lineage.unknown_source_reason == BASE_SNAPSHOT_UNRECORDED
    assert market.lineage.effective_timestamp_basis == "feature_table_build"

    written = write_card_lineage(lineage, tmp_path)
    assert written.name == LINEAGE_FILENAME
    assert read_card_lineage(tmp_path) == lineage


def test_card_lineage_json_round_trips(tmp_path: Path) -> None:
    lineage = _lineage(
        overlay_sources=(
            OverlaySource(
                member_id="player_arrests_back_side_policy",
                builder_module="nfl_ats.player_arrests_back_side_overlay",
                builder_version="fingerprint0",
                effective_timestamp="2026-09-02T12:00:00+00:00",
                source_snapshot="20260902T120000Z",
                source_captured_at="2026-09-02T12:00:00+00:00",
                flipped_game_ids=("2026_01_AAA_BBB",),
            ),
        ),
        tiebreaker_sources=(
            TiebreakerSource(
                input_name="market_consensus",
                builder_module="nfl_ats.tiebreaker",
                builder_version="v1",
                effective_timestamp="2026-09-02T18:00:00+00:00",
                source_snapshot="20260902T180000Z",
                source_captured_at="2026-09-02T18:00:00+00:00",
            ),
        ),
    )

    restored = CardLineage.from_json(lineage.to_json())
    assert restored == lineage
    assert "overlay:player_arrests_back_side_policy" in restored.decision_bearing_fields()
    assert "tiebreaker:market_consensus" in restored.decision_bearing_fields()

    payload = json.loads(lineage.to_json())
    assert payload["schema_version"] == lineage.schema_version
    assert [entry["card_field"] for entry in payload["fields"]] == [
        entry.card_field for entry in lineage.entries
    ]
    display = next(entry for entry in payload["fields"] if entry["card_field"] == "Matchup")
    assert display["lineage"] is None
    assert display["decision_bearing"] is False
    assert display["reason"]

    (tmp_path / LINEAGE_FILENAME).write_text(lineage.to_json(), encoding="utf-8")
    assert read_card_lineage(tmp_path) == lineage


def test_prediction_timestamp_falls_back_to_forecast_metadata() -> None:
    assert _lineage().prediction_timestamp == "2026-09-03T14:32:53+00:00"


def test_families_for_columns_reports_unclaimed_inputs() -> None:
    assert families_for_columns(("spread_line",)) == ("market",)
    assert "unassigned" in families_for_columns(("spread_line", "a_column_no_family_claims"))


def test_parse_snapshot_capture_refuses_to_guess() -> None:
    assert parse_snapshot_capture("20260817T184901Z") == "2026-08-17T18:49:01+00:00"
    assert parse_snapshot_capture("20260817T184901Z-ncaaf") is None
    assert parse_snapshot_capture(None) is None


# ---------------------------------------------------------------------------
# ENG-22: an inherited source_snapshots block names a real snapshot
# ---------------------------------------------------------------------------


def test_market_and_base_families_prefer_an_inherited_snapshot_over_the_digest(
    tmp_path: Path,
) -> None:
    """A derived manifest that carries an ENG-22 source_snapshots block (see
    nfl_ats.feature_manifest.inherit_source_snapshots) resolves market_line
    and every base-table model_input family to the real snapshot id instead
    of falling back to feature_table:sha256."""

    metadata = _synthetic_metadata()
    metadata["provenance"]["feature_table"]["manifest"]["source_snapshots"] = {
        "source_snapshot": {
            "snapshot_id": "20260824T115346Z",
            "captured_at": "2026-08-24T11:53:46+00:00",
            "manifest_path": "data/processed/game_features.manifest.json",
        }
    }
    lineage = build_card_lineage(
        _synthetic_forecast(),
        metadata,
        feature_columns=("spread_line", "elo_diff"),
        display_fields={"Matchup": "formatted from team columns"},
    )

    market = lineage.field(FIELD_MARKET_LINE)
    assert market is not None and market.lineage is not None
    assert market.lineage.source_snapshot == "20260824T115346Z"
    assert market.lineage.source_captured_at == "2026-08-24T11:53:46+00:00"
    assert market.lineage.unknown_source_reason is None
    assert market.lineage.effective_timestamp_basis == "source_capture"

    market_input = lineage.field("model_input:market")
    assert market_input is not None and market_input.lineage is not None
    assert market_input.lineage.source_snapshot == "20260824T115346Z"
    assert market_input.lineage.unknown_source_reason is None

    elo_input = lineage.field("model_input:elo")
    assert elo_input is not None and elo_input.lineage is not None
    assert elo_input.lineage.source_snapshot == "20260824T115346Z"

    assert validate_prediction_lineage(lineage).status == "PASS"
    (tmp_path / LINEAGE_FILENAME).write_text(lineage.to_json(), encoding="utf-8")
    assert read_card_lineage(tmp_path) == lineage


def test_a_null_inherited_entry_still_falls_back_to_the_digest() -> None:
    """A transitively-forwarded 'upstream manifest absent' marker (see
    inherit_source_snapshots) must not be mistaken for a resolved snapshot."""

    metadata = _synthetic_metadata()
    metadata["provenance"]["feature_table"]["manifest"]["source_snapshots"] = {
        "source_snapshot": {"snapshot_id": None, "reason": "upstream manifest absent"}
    }
    lineage = build_card_lineage(
        _synthetic_forecast(),
        metadata,
        feature_columns=("spread_line",),
        display_fields={"Matchup": "formatted from team columns"},
    )

    market = lineage.field(FIELD_MARKET_LINE)
    assert market is not None and market.lineage is not None
    assert market.lineage.source_snapshot == f"feature_table:sha256:{'a' * 64}"
    assert market.lineage.unknown_source_reason == BASE_SNAPSHOT_UNRECORDED


def test_legacy_manifests_without_a_source_snapshots_block_are_unaffected() -> None:
    """No ENG-22 block at all -- e.g. every manifest on disk before this
    change -- keeps validating exactly as it did before: the digest fallback
    and its reason are unchanged."""

    market = _lineage().field(FIELD_MARKET_LINE)
    assert market is not None and market.lineage is not None
    assert market.lineage.unknown_source_reason == BASE_SNAPSHOT_UNRECORDED


# ---------------------------------------------------------------------------
# The release-blocking half
# ---------------------------------------------------------------------------


def test_safety_check_passes_on_complete_lineage() -> None:
    audit = validate_prediction_lineage(_lineage())

    assert audit.status == "PASS"
    assert audit.card_type == "lineage"
    assert set(LINEAGE_CHECKS).issubset(audit.checks_passed)
    assert validate_card_lineage(_lineage()) == LINEAGE_CHECKS


@pytest.mark.parametrize("dropped", [FIELD_PICK, FIELD_MODEL_PROBABILITY, FIELD_MARKET_LINE])
def test_safety_check_fails_on_a_missing_decision_bearing_field(dropped: str) -> None:
    lineage = _lineage()
    pruned = replace(
        lineage,
        entries=tuple(entry for entry in lineage.entries if entry.card_field != dropped),
    )

    with pytest.raises(PredictionSafetyError) as error:
        validate_prediction_lineage(pruned)
    assert "lineage" in str(error.value)
    assert dropped in str(error.value)


def test_safety_check_fails_when_a_decision_bearing_field_has_a_null_record() -> None:
    lineage = _lineage()
    blanked = replace(
        lineage,
        entries=tuple(
            CardLineageEntry(entry.card_field, True, None, None)
            if entry.card_field == FIELD_PICK
            else entry
            for entry in lineage.entries
        ),
    )

    with pytest.raises(PredictionSafetyError, match=FIELD_PICK):
        validate_prediction_lineage(blanked)


def test_safety_check_fails_on_an_effective_timestamp_after_the_prediction() -> None:
    lineage = _lineage()
    leaked = replace(
        lineage,
        entries=tuple(
            CardLineageEntry(
                entry.card_field,
                entry.decision_bearing,
                replace(entry.lineage, effective_timestamp="2026-09-14T00:00:00+00:00"),
                entry.reason,
            )
            if entry.card_field == "model_input:player_injuries"
            else entry
            for entry in lineage.entries
        ),
    )

    with pytest.raises(PredictionSafetyError) as error:
        validate_prediction_lineage(leaked)
    message = str(error.value)
    assert "after the prediction timestamp" in message
    assert "model_input:player_injuries" in message
    # Same card, graded against a later decision instant, is fine: the
    # invariant is about ordering, not about the literal string.
    assert (
        validate_prediction_lineage(
            leaked, prediction_timestamp=datetime(2026, 9, 15, tzinfo=UTC)
        ).status
        == "PASS"
    )


def test_safety_check_passes_with_a_null_lineage_display_field() -> None:
    lineage = _lineage(display_fields={"Decision score": "rendering of model_probability"})
    entry = lineage.field("Decision score")

    assert entry is not None
    assert entry.lineage is None
    assert entry.decision_bearing is False
    assert validate_prediction_lineage(lineage).status == "PASS"


def test_safety_check_fails_on_a_display_field_with_no_reason() -> None:
    lineage = _lineage()
    silent = lineage.with_entries([CardLineageEntry("Mystery column", False, None, None)])

    with pytest.raises(PredictionSafetyError, match="Mystery column"):
        validate_prediction_lineage(silent)


def test_safety_check_fails_when_an_absent_snapshot_is_not_explained() -> None:
    lineage = _lineage()
    unexplained = replace(
        lineage,
        entries=tuple(
            CardLineageEntry(
                entry.card_field,
                entry.decision_bearing,
                replace(entry.lineage, source_snapshot=None, unknown_source_reason=None),
                entry.reason,
            )
            if entry.card_field == FIELD_MARKET_LINE
            else entry
            for entry in lineage.entries
        ),
    )

    with pytest.raises(PredictionSafetyError, match="unknown_source_reason"):
        validate_prediction_lineage(unexplained)


def test_unsupported_schema_version_is_rejected() -> None:
    with pytest.raises(LineageError, match="schema version"):
        validate_card_lineage(replace(_lineage(), schema_version=99))


def test_unparseable_effective_timestamp_is_rejected() -> None:
    lineage = CardLineage(
        prediction_timestamp=PREDICTION_TIMESTAMP,
        entries=(
            CardLineageEntry(
                FIELD_PICK,
                True,
                LineageRecord(
                    card_field=FIELD_PICK,
                    feature_family="model_decision",
                    source_snapshot="snapshot",
                    source_captured_at=None,
                    effective_timestamp="whenever",
                    builder_version="v1",
                    builder_module="nfl_ats.outcomes",
                ),
            ),
        ),
    )

    with pytest.raises(LineageError, match="unparseable"):
        validate_card_lineage(lineage, required_fields=(FIELD_PICK,))


# ---------------------------------------------------------------------------
# Overlay adaptation
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


def test_only_overlays_that_fired_become_decision_bearing() -> None:
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

    sources = overlay_sources_from_composition(
        composition, fallback_effective_timestamp=FEATURE_BUILD
    )

    assert [source.member_id for source in sources] == ["player_arrests_back_side_policy"]
    assert sources[0].source_snapshot == "20260902T120000Z"
    assert sources[0].effective_timestamp_basis == "source_capture"
    assert sources[0].builder_module == "nfl_ats.player_arrests_back_side_overlay"

    lineage = _lineage(overlay_sources=sources)
    assert "overlay:player_arrests_back_side_policy" in lineage.decision_bearing_fields()
    assert "overlay:coach_fade" not in lineage.decision_bearing_fields()
    assert validate_prediction_lineage(lineage).status == "PASS"


def test_a_non_snapshot_overlay_member_still_explains_its_absent_snapshot() -> None:
    composition = _FakeComposition(
        members=(
            _FakeMember(
                "spread_gap_zone_fade",
                "nfl_ats.spread_gap_zone_fade_overlay.apply",
                ("2026_01_CCC_DDD",),
            ),
        )
    )

    sources = overlay_sources_from_composition(
        composition, fallback_effective_timestamp=FEATURE_BUILD
    )

    assert sources[0].source_snapshot is None
    assert sources[0].unknown_source_reason
    assert validate_prediction_lineage(_lineage(overlay_sources=sources)).status == "PASS"


# ---------------------------------------------------------------------------
# Integration with the pre-existing, unchanged safety contract
# ---------------------------------------------------------------------------


def test_existing_card_validation_is_unchanged_without_lineage(
    model_frame: pd.DataFrame,
) -> None:
    predictions, _ = score_week(model_frame, season=2020, week=1, min_train_games=80)

    audit = validate_prediction_card(
        predictions, min_edge=0.02, expected_season=2020, expected_week=1
    )

    assert audit.status == "PASS"
    assert not set(LINEAGE_CHECKS).intersection(audit.checks_passed)


def test_card_validation_adds_lineage_checks_and_fails_closed(
    model_frame: pd.DataFrame,
) -> None:
    predictions, _ = score_week(model_frame, season=2020, week=1, min_train_games=80)
    lineage = _lineage()

    audit = validate_prediction_card(
        predictions,
        min_edge=0.02,
        expected_season=2020,
        expected_week=1,
        lineage=lineage,
    )
    assert set(LINEAGE_CHECKS).issubset(audit.checks_passed)

    broken = replace(
        lineage,
        entries=tuple(
            entry for entry in lineage.entries if entry.card_field != FIELD_MODEL_PROBABILITY
        ),
    )
    with pytest.raises(PredictionSafetyError, match=FIELD_MODEL_PROBABILITY):
        validate_prediction_card(
            predictions,
            min_edge=0.02,
            expected_season=2020,
            expected_week=1,
            lineage=broken,
        )
