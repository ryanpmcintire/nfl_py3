"""Safety and composition contracts for the frozen four-overlay policy."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import nfl_ats.four_overlay_composition as composition
from nfl_ats.data import DataContractError
from nfl_ats.player_arrests_back_side_overlay import ArrestSnapshot


def _predictions() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "game_id": ["G_COACH", "G_DIV", "G_ARREST", "G_SPREAD"],
            "season": [2026] * 4,
            "week": [1, 10, 1, 1],
            "game_type": ["REG"] * 4,
            "gameday": ["2026-09-13", "2026-11-15", "2026-09-13", "2026-09-13"],
            "home_team": ["BAL", "BUF", "JAX", "DAL"],
            "away_team": ["IND", "MIA", "TEN", "NYG"],
            "home_cover_probability": [0.65, 0.65, 0.40, 0.60],
            "spread_line": [-3.0, -3.0, -2.0, -8.0],
        }
    )


def _schedules() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "game_id": [
                "PRIOR_COACH",
                "FIRST_DIV",
                "G_COACH",
                "G_DIV",
                "G_ARREST",
                "G_SPREAD",
            ],
            "season": [2025, 2026, 2026, 2026, 2026, 2026],
            "game_type": ["REG"] * 6,
            "gameday": [
                "2025-10-01",
                "2026-09-20",
                "2026-09-13",
                "2026-11-15",
                "2026-09-13",
                "2026-09-13",
            ],
            "home_team": ["BAL", "MIA", "BAL", "BUF", "JAX", "DAL"],
            "away_team": ["IND", "BUF", "IND", "MIA", "TEN", "NYG"],
            "home_coach": [
                "Old BAL",
                "MIA Coach",
                "New BAL",
                "BUF Coach",
                "JAX Coach",
                "DAL Coach",
            ],
            "away_coach": [
                "IND Coach",
                "BUF Coach",
                "IND Coach",
                "MIA Coach",
                "TEN Coach",
                "NYG Coach",
            ],
            "result": [3.0, -7.0, np.nan, np.nan, np.nan, np.nan],
        }
    )


def _incidents() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "record_id": ["arrest-1"],
            "incident_date": ["2026-09-07"],
            "team": ["JAX"],
        }
    )


def _snapshot(tmp_path: Path | None = None) -> ArrestSnapshot:
    directory = (tmp_path or Path(".")) / "20260908T120000Z"
    return ArrestSnapshot(
        snapshot_id=directory.name,
        directory=directory,
        manifest_path=directory / "manifest.json",
        safe_index_path=directory / "incidents_point_in_time.parquet",
        fetched_at_utc=pd.Timestamp("2026-09-08T12:00:00Z"),
        age_hours=1.0,
        safe_index_sha256="a" * 64,
        rows_cached=1,
    )


def _apply(
    predictions: pd.DataFrame | None = None,
    schedules: pd.DataFrame | None = None,
    incidents: pd.DataFrame | None = None,
) -> composition.FourOverlayCompositionResult:
    return composition.apply_four_overlay_composition(
        _predictions() if predictions is None else predictions,
        _schedules() if schedules is None else schedules,
        _incidents() if incidents is None else incidents,
        arrest_snapshot=_snapshot(),
    )


def test_policy_identity_and_joint_or_members_are_frozen() -> None:
    definition = composition.policy_definition()
    encoded = json.dumps(
        definition, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")

    assert (
        composition.POLICY_ID == "overlay_union_coach_division_revenge_player_arrests_spread_gap_v1"
    )
    assert composition.INCUMBENT_CHALLENGER_ID == "overlay_production_chain_coach_arrest_incumbent"
    assert composition.COMPOSITION_ORDER == (
        "coach_fade",
        "division_revenge_tilt",
        "player_arrests_back_side_policy",
        "spread_gap_zone_fade",
    )
    assert definition["semantics"] == "joint_or_against_raw_card_complement_once"
    assert hashlib.sha256(encoded).hexdigest() == composition.POLICY_FINGERPRINT
    assert (
        composition.POLICY_FINGERPRINT
        == "bbdd60a1712386541546c8e757615fb5ff216f49eb81397502cb360809bc5ded"
    )


def test_composition_reuses_each_member_and_unions_its_flip_set() -> None:
    result = _apply()
    actual = result.overlaid_predictions.set_index("game_id")["home_cover_probability"]

    assert [member.flipped_game_ids for member in result.members] == [
        ("G_COACH",),
        ("G_DIV",),
        ("G_ARREST",),
        ("G_SPREAD",),
    ]
    assert result.union_flipped_game_ids == (
        "G_COACH",
        "G_DIV",
        "G_ARREST",
        "G_SPREAD",
    )
    assert result.overlapping_game_ids == ()
    assert actual.to_dict() == pytest.approx(
        {"G_COACH": 0.35, "G_DIV": 0.35, "G_ARREST": 0.60, "G_SPREAD": 0.40}
    )
    assert result.policy_id == composition.POLICY_ID
    assert result.policy_fingerprint == composition.POLICY_FINGERPRINT
    assert result.arrest_snapshot_id == "20260908T120000Z"


def test_overlapping_members_complement_raw_probability_once_instead_of_cancelling() -> None:
    predictions = _predictions().iloc[[0]].copy()
    predictions["spread_line"] = -8.0
    schedules = _schedules().loc[_schedules()["game_id"].isin(["PRIOR_COACH", "G_COACH"])].copy()
    incidents = _incidents().iloc[0:0].copy()

    result = _apply(predictions, schedules, incidents)

    assert result.union_flipped_game_ids == ("G_COACH",)
    assert result.overlapping_game_ids == ("G_COACH",)
    assert result.games[0].member_ids == ("coach_fade", "spread_gap_zone_fade")
    assert result.games[0].raw_home_cover_probability == pytest.approx(0.65)
    assert result.games[0].final_home_cover_probability == pytest.approx(0.35)


def test_members_are_evaluated_in_declared_order_against_the_same_raw_card(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, int]] = []

    def fake(member_id: str):
        def apply(predictions: pd.DataFrame, *_args: object, **_kwargs: object) -> SimpleNamespace:
            calls.append((member_id, id(predictions)))
            return SimpleNamespace(overlaid_predictions=predictions.copy(), flips=(), enabled=True)

        return apply

    monkeypatch.setattr(composition, "apply_coach_fade_overlay", fake("coach_fade"))
    monkeypatch.setattr(
        composition,
        "apply_division_revenge_tilt_overlay",
        fake("division_revenge_tilt"),
    )
    monkeypatch.setattr(
        composition,
        "apply_player_arrests_back_side_overlay",
        fake("player_arrests_back_side_policy"),
    )
    monkeypatch.setattr(
        composition,
        "apply_spread_gap_zone_fade_overlay",
        fake("spread_gap_zone_fade"),
    )

    result = _apply()

    assert tuple(member for member, _frame_id in calls) == composition.COMPOSITION_ORDER
    assert len({frame_id for _member, frame_id in calls}) == 1
    assert tuple(member.member_id for member in result.members) == composition.COMPOSITION_ORDER


def test_coach_contract_error_disables_only_that_member() -> None:
    schedules = _schedules().drop(columns=["home_coach", "away_coach"])

    result = _apply(schedules=schedules)

    coach = result.members[0]
    assert coach.member_id == "coach_fade"
    assert coach.enabled is False
    assert coach.status == "disabled_contract_error"
    assert coach.flipped_game_ids == ()
    assert "coach tenure" in str(coach.detail)
    assert result.union_flipped_game_ids == ("G_DIV", "G_ARREST", "G_SPREAD")


def test_non_coach_member_contract_errors_propagate() -> None:
    schedules = _schedules().drop(columns=["result"])

    with pytest.raises(DataContractError, match="division-revenge tracking"):
        _apply(schedules=schedules)


def test_future_rows_do_not_change_earlier_composed_decisions() -> None:
    baseline = _apply()
    schedules = _schedules()
    future_schedule = pd.DataFrame(
        {
            "game_id": ["FUTURE_DIV"],
            "season": [2026],
            "game_type": ["REG"],
            "gameday": ["2026-12-20"],
            "home_team": ["MIA"],
            "away_team": ["BUF"],
            "home_coach": ["MIA Coach"],
            "away_coach": ["BUF Coach"],
            "result": [99.0],
        }
    )
    schedules = pd.concat([schedules, future_schedule], ignore_index=True)
    incidents = pd.concat(
        [
            _incidents(),
            pd.DataFrame(
                {
                    "record_id": ["future-arrest"],
                    "incident_date": ["2026-09-09"],
                    "team": ["TEN"],
                }
            ),
        ],
        ignore_index=True,
    )

    with_future = _apply(schedules=schedules, incidents=incidents)

    pd.testing.assert_series_equal(
        baseline.overlaid_predictions["home_cover_probability"],
        with_future.overlaid_predictions["home_cover_probability"],
    )
    assert baseline.union_flipped_game_ids == with_future.union_flipped_game_ids
    assert baseline.games == with_future.games


def test_outcome_and_grade_columns_are_decision_blind() -> None:
    first_predictions = _predictions().assign(
        home_score=[99, 99, 99, 99],
        away_score=[0, 0, 0, 0],
        ats_result=["WIN"] * 4,
    )
    second_predictions = first_predictions.assign(
        home_score=[0, 0, 0, 0],
        away_score=[99, 99, 99, 99],
        ats_result=["LOSS"] * 4,
    )
    first_incidents = _incidents().assign(outcome_archive_only="first")
    second_incidents = _incidents().assign(outcome_archive_only="second")

    first = _apply(first_predictions, incidents=first_incidents)
    second = _apply(second_predictions, incidents=second_incidents)

    pd.testing.assert_series_equal(
        first.overlaid_predictions["home_cover_probability"],
        second.overlaid_predictions["home_cover_probability"],
    )
    assert first.union_flipped_game_ids == second.union_flipped_game_ids
    assert first.members == second.members
    assert first.games == second.games


def _write_arrest_snapshot(
    data_root: Path,
    *,
    fetched_at_utc: str,
) -> None:
    snapshot_id = "20260908T120000Z"
    directory = data_root / "raw" / "player_arrests" / snapshot_id
    directory.mkdir(parents=True)
    safe_path = directory / "incidents_point_in_time.parquet"
    _incidents().to_parquet(safe_path, index=False)
    digest = hashlib.sha256(safe_path.read_bytes()).hexdigest()
    manifest = {
        "snapshot_id": snapshot_id,
        "fetched_at_utc": fetched_at_utc,
        "complete": True,
        "rows_cached": len(_incidents()),
        "point_in_time_policy": {"safe_index": safe_path.name},
        "files": {safe_path.name: digest},
    }
    (directory / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


def test_publication_boundary_fails_closed_when_arrest_source_is_unavailable(
    tmp_path: Path,
) -> None:
    with pytest.raises(FileNotFoundError, match="player-arrests snapshot root"):
        composition.apply_four_overlay_composition_for_publication(
            _predictions(),
            _schedules(),
            tmp_path / "data",
            now=datetime(2026, 9, 8, 13, 0, tzinfo=UTC),
        )


def test_publication_boundary_fails_closed_when_arrest_source_is_stale(
    tmp_path: Path,
) -> None:
    data_root = tmp_path / "data"
    _write_arrest_snapshot(data_root, fetched_at_utc="2026-09-06T00:00:00+00:00")

    with pytest.raises(DataContractError, match="stale"):
        composition.apply_four_overlay_composition_for_publication(
            _predictions(),
            _schedules(),
            data_root,
            now=datetime(2026, 9, 8, 13, 0, tzinfo=UTC),
        )


def test_publication_boundary_records_verified_arrest_source_provenance(
    tmp_path: Path,
) -> None:
    data_root = tmp_path / "data"
    _write_arrest_snapshot(data_root, fetched_at_utc="2026-09-08T12:00:00+00:00")

    result = composition.apply_four_overlay_composition_for_publication(
        _predictions(),
        _schedules(),
        data_root,
        now=datetime(2026, 9, 8, 13, 0, tzinfo=UTC),
    )

    assert result.arrest_snapshot_id == "20260908T120000Z"
    assert result.arrest_snapshot_fetched_at_utc == pd.Timestamp("2026-09-08T12:00:00Z")
    assert len(result.arrest_safe_index_sha256) == 64
