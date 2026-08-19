"""Division-revenge tilt overlay (docs/division_revenge_tilt_overlay.md).

Four things are load-bearing here, mirroring
``tests/test_coach_fade_overlay.py``'s structure and AGENTS.md's "add a
leakage regression test for every new feature family" spirit:

1. :func:`division_revenge_side_by_game`'s flag is derived from data, not
   hand-typed, is pregame-safe (a later meeting or a later season must never
   retroactively change an earlier meeting's flag), and correctly identifies
   the revenge side on EITHER the home or away side of the second meeting.
2. :func:`apply_division_revenge_tilt_overlay` flips ONLY when the model's
   pick sits against the (unique) revenge side, respects the REG-only gate,
   and is parameter-free.
3. :func:`overlay_disclosure_note` states the flip count and matchups.
4. :func:`record_division_revenge_tilt_challenger_decisions` writes the
   tilt's own picks to the prospective challenger ledger, dual-tracked and at
   no rotation-registry window cost.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import pytest

from nfl_ats.active_model import ACTIVE_ATS_MODEL_VERSION
from nfl_ats.data import DataContractError
from nfl_ats.division_revenge_tilt_overlay import (
    CHALLENGER_ID,
    apply_division_revenge_tilt_overlay,
    division_revenge_side_by_game,
    overlay_disclosure_note,
    record_division_revenge_tilt_challenger_decisions,
)
from nfl_ats.prospective_scoring import (
    CHALLENGER_DECISION_COLUMNS,
    artifact_model_config,
    config_fingerprint,
    load_challenger_decisions,
)
from nfl_ats.snapshots import write_snapshot

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------
#
# REV/WIN meet twice: G1 (week 1) REV loses at home to WIN; G2 (week 10) the
#   rematch, WIN hosting -- REV (away in G2) is the unique revenge side.
# REV2/WIN2 meet twice with the revenge side on the HOME side instead: G3
#   (week 1) REV2 loses on the road to WIN2; G4 (week 10) REV2 hosts the
#   rematch -- REV2 (home in G4) is the revenge side.
# REV3/WIN3 meet twice, mirroring G1/G2's shape, used only for the
#   "already agrees" no-flip case.
# TIE1/TIE2 meet twice but their first meeting (G5) is an exact tie -- G6 (the
#   rematch) has no revenge side for either team.
# SOLO1/SOLO2 meet exactly once (G7) -- a single meeting is never a "2nd
#   meeting" and is never flagged.
# G8 is the SAME shape as G2 but POST season -- the REG-only gate must block
#   it.
# G9 is a THIRD meeting between REV and WIN (week 15), used only by the
#   leakage regression: mutating it must not move G2's already-computed flag.


def _revenge_schedule() -> pd.DataFrame:
    rows = [
        # game_id, season, game_type, week, gameday, home_team, away_team, result
        ("2026_01_REV_WIN", 2026, "REG", 1, "2026-09-10", "REV", "WIN", -10.0),  # REV loses home
        (
            "2026_10_WIN_REV",
            2026,
            "REG",
            10,
            "2026-11-08",
            "WIN",
            "REV",
            3.0,
        ),  # rematch; REV away = revenge
        (
            "2026_01_WIN2_REV2",
            2026,
            "REG",
            1,
            "2026-09-10",
            "WIN2",
            "REV2",
            15.0,
        ),  # REV2 loses on the road
        (
            "2026_10_REV2_WIN2",
            2026,
            "REG",
            10,
            "2026-11-08",
            "REV2",
            "WIN2",
            -1.0,
        ),  # rematch; REV2 home = revenge
        (
            "2026_01_REV3_WIN3",
            2026,
            "REG",
            1,
            "2026-09-10",
            "REV3",
            "WIN3",
            -7.0,
        ),  # REV3 loses at home
        (
            "2026_10_WIN3_REV3",
            2026,
            "REG",
            10,
            "2026-11-08",
            "WIN3",
            "REV3",
            2.0,
        ),  # rematch; REV3 away = revenge
        ("2026_01_TIE1_TIE2", 2026, "REG", 1, "2026-09-10", "TIE1", "TIE2", 0.0),  # exact tie
        (
            "2026_10_TIE2_TIE1",
            2026,
            "REG",
            10,
            "2026-11-08",
            "TIE2",
            "TIE1",
            4.0,
        ),  # rematch; no revenge side
        (
            "2026_01_SOLO1_SOLO2",
            2026,
            "REG",
            1,
            "2026-09-10",
            "SOLO1",
            "SOLO2",
            6.0,
        ),  # only meeting all season
        (
            "2026_20_WIN_REV",
            2026,
            "POST",
            20,
            "2027-01-10",
            "WIN",
            "REV",
            3.0,
        ),  # same shape as G2, POST
        (
            "2026_15_REV_WIN",
            2026,
            "REG",
            15,
            "2026-12-13",
            "REV",
            "WIN",
            -2.0,
        ),  # 3rd REV/WIN meeting
    ]
    return pd.DataFrame(
        rows,
        columns=[
            "game_id",
            "season",
            "game_type",
            "week",
            "gameday",
            "home_team",
            "away_team",
            "result",
        ],
    )


def _predictions() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "game_id": [
                "2026_10_WIN_REV",
                "2026_10_REV2_WIN2",
                "2026_10_TIE2_TIE1",
                "2026_01_SOLO1_SOLO2",
                "2026_20_WIN_REV",
                "2026_10_WIN3_REV3",
                "2026_MISSING_GAME",
            ],
            "season": [2026, 2026, 2026, 2026, 2026, 2026, 2026],
            "week": [10, 10, 10, 1, 20, 10, 10],
            "game_type": ["REG", "REG", "REG", "REG", "POST", "REG", "REG"],
            "home_team": ["WIN", "REV2", "TIE2", "SOLO1", "WIN", "WIN3", "MISS_H"],
            "away_team": ["REV", "WIN2", "TIE1", "SOLO2", "REV", "REV3", "MISS_A"],
            "kickoff": ["2026-11-08T18:00:00+00:00"] * 7,
            "spread_line": [-3.0, 2.5, -1.0, -4.0, -3.0, -2.0, 1.0],
            # G2 (WIN home, REV away, revenge): model picks HOME (WIN) -- against
            # the revenge side -- should flip to AWAY (REV).
            # G4 (REV2 home, revenge): model picks AWAY (WIN2) -- against the
            # revenge side -- should flip to HOME (REV2).
            # G6 (tie -> no revenge side): untouched regardless of the pick.
            # G7 (single meeting -> never flagged): untouched.
            # G8 (POST, same shape as G2): REG-only gate blocks the flip.
            # G-WIN3-REV3 rematch: model already picks AWAY (REV3, the revenge
            # side) -- already agrees, no flip needed.
            # G-MISSING: no schedule row at all -- treated as no signal.
            "home_cover_probability": [0.70, 0.25, 0.60, 0.55, 0.70, 0.20, 0.50],
        }
    )


# ---------------------------------------------------------------------------
# 1. division_revenge_side_by_game: derived, pregame-safe
# ---------------------------------------------------------------------------


def test_revenge_flag_fires_on_the_away_side_of_the_rematch() -> None:
    flags = division_revenge_side_by_game(_revenge_schedule()).set_index("game_id")
    assert bool(flags.loc["2026_10_WIN_REV", "revenge_away"]) is True
    assert bool(flags.loc["2026_10_WIN_REV", "revenge_home"]) is False


def test_revenge_flag_fires_on_the_home_side_of_the_rematch() -> None:
    flags = division_revenge_side_by_game(_revenge_schedule()).set_index("game_id")
    assert bool(flags.loc["2026_10_REV2_WIN2", "revenge_home"]) is True
    assert bool(flags.loc["2026_10_REV2_WIN2", "revenge_away"]) is False


def test_revenge_flag_never_fires_on_the_first_meeting() -> None:
    flags = division_revenge_side_by_game(_revenge_schedule()).set_index("game_id")
    assert bool(flags.loc["2026_01_REV_WIN", "revenge_home"]) is False
    assert bool(flags.loc["2026_01_REV_WIN", "revenge_away"]) is False


def test_revenge_flag_is_false_on_both_sides_after_an_exact_tie() -> None:
    flags = division_revenge_side_by_game(_revenge_schedule()).set_index("game_id")
    assert bool(flags.loc["2026_10_TIE2_TIE1", "revenge_home"]) is False
    assert bool(flags.loc["2026_10_TIE2_TIE1", "revenge_away"]) is False


def test_revenge_flag_is_false_for_a_single_meeting() -> None:
    flags = division_revenge_side_by_game(_revenge_schedule()).set_index("game_id")
    assert bool(flags.loc["2026_01_SOLO1_SOLO2", "revenge_home"]) is False
    assert bool(flags.loc["2026_01_SOLO1_SOLO2", "revenge_away"]) is False


def test_revenge_flag_requires_its_schedule_columns() -> None:
    with pytest.raises(DataContractError, match="division-revenge"):
        division_revenge_side_by_game(pd.DataFrame({"game_id": ["G1"]}))


def test_revenge_flag_is_leak_safe_against_a_later_meeting_mutation() -> None:
    """AGENTS.md: a leakage regression test for every new feature family.

    G2's flag (the 2nd REV/WIN meeting) depends only on G1 (the 1st
    meeting); mutating G9 (a 3rd, LATER meeting) must not move it.
    """

    baseline = division_revenge_side_by_game(_revenge_schedule()).set_index("game_id")

    mutated = _revenge_schedule()
    mutated.loc[mutated["game_id"].eq("2026_15_REV_WIN"), "result"] = 99.0
    changed = division_revenge_side_by_game(mutated).set_index("game_id")

    assert bool(changed.loc["2026_10_WIN_REV", "revenge_away"]) == bool(
        baseline.loc["2026_10_WIN_REV", "revenge_away"]
    )
    assert bool(changed.loc["2026_01_REV_WIN", "revenge_home"]) == bool(
        baseline.loc["2026_01_REV_WIN", "revenge_home"]
    )


def test_revenge_flag_is_leak_safe_across_the_season_boundary() -> None:
    """A future season's rematch (even between the same two teams) must
    never change an earlier season's already-computed flag."""

    schedule = _revenge_schedule()
    baseline = division_revenge_side_by_game(schedule)

    future = pd.DataFrame(
        [
            ("2027_01_REV_WIN", 2027, "REG", 1, "2027-09-09", "REV", "WIN", -20.0),
            ("2027_10_WIN_REV", 2027, "REG", 10, "2027-11-07", "WIN", "REV", 6.0),
        ],
        columns=schedule.columns,
    )
    changed = division_revenge_side_by_game(pd.concat([schedule, future], ignore_index=True))

    pd.testing.assert_frame_equal(
        changed.loc[changed["season"].le(2026)].reset_index(drop=True),
        baseline.loc[baseline["season"].le(2026)].reset_index(drop=True),
        check_exact=True,
    )


# ---------------------------------------------------------------------------
# 2. apply_division_revenge_tilt_overlay: the pick-level transform
# ---------------------------------------------------------------------------


def test_overlay_flips_to_the_away_revenge_side() -> None:
    result = apply_division_revenge_tilt_overlay(_predictions(), _revenge_schedule())

    flipped_ids = {flip.game_id for flip in result.flips}
    assert "2026_10_WIN_REV" in flipped_ids
    flip = next(f for f in result.flips if f.game_id == "2026_10_WIN_REV")
    assert flip.revenge_team == "REV"
    assert flip.opponent_team == "WIN"

    overlaid = result.overlaid_predictions.set_index("game_id")
    assert overlaid.loc["2026_10_WIN_REV", "home_cover_probability"] == pytest.approx(0.30)


def test_overlay_flips_to_the_home_revenge_side() -> None:
    result = apply_division_revenge_tilt_overlay(_predictions(), _revenge_schedule())

    flipped_ids = {flip.game_id for flip in result.flips}
    assert "2026_10_REV2_WIN2" in flipped_ids
    flip = next(f for f in result.flips if f.game_id == "2026_10_REV2_WIN2")
    assert flip.revenge_team == "REV2"
    assert flip.opponent_team == "WIN2"

    overlaid = result.overlaid_predictions.set_index("game_id")
    assert overlaid.loc["2026_10_REV2_WIN2", "home_cover_probability"] == pytest.approx(0.75)


def test_overlay_does_not_flip_after_a_tied_first_meeting() -> None:
    result = apply_division_revenge_tilt_overlay(_predictions(), _revenge_schedule())
    assert all(flip.game_id != "2026_10_TIE2_TIE1" for flip in result.flips)
    overlaid = result.overlaid_predictions.set_index("game_id")
    assert overlaid.loc["2026_10_TIE2_TIE1", "home_cover_probability"] == pytest.approx(0.60)


def test_overlay_does_not_flip_a_single_meeting() -> None:
    result = apply_division_revenge_tilt_overlay(_predictions(), _revenge_schedule())
    assert all(flip.game_id != "2026_01_SOLO1_SOLO2" for flip in result.flips)
    overlaid = result.overlaid_predictions.set_index("game_id")
    assert overlaid.loc["2026_01_SOLO1_SOLO2", "home_cover_probability"] == pytest.approx(0.55)


def test_overlay_leaves_postseason_games_untouched() -> None:
    """Same shape as the flipped G2, but POST season -- the REG-only gate
    blocks it."""

    result = apply_division_revenge_tilt_overlay(_predictions(), _revenge_schedule())
    assert all(flip.game_id != "2026_20_WIN_REV" for flip in result.flips)
    overlaid = result.overlaid_predictions.set_index("game_id")
    assert overlaid.loc["2026_20_WIN_REV", "home_cover_probability"] == pytest.approx(0.70)


def test_overlay_does_not_flip_when_the_pick_already_agrees_with_the_revenge_side() -> None:
    result = apply_division_revenge_tilt_overlay(_predictions(), _revenge_schedule())
    assert all(flip.game_id != "2026_10_WIN3_REV3" for flip in result.flips)
    overlaid = result.overlaid_predictions.set_index("game_id")
    assert overlaid.loc["2026_10_WIN3_REV3", "home_cover_probability"] == pytest.approx(0.20)


def test_overlay_treats_a_missing_schedule_row_as_no_signal() -> None:
    result = apply_division_revenge_tilt_overlay(_predictions(), _revenge_schedule())
    assert all(flip.game_id != "2026_MISSING_GAME" for flip in result.flips)
    overlaid = result.overlaid_predictions.set_index("game_id")
    assert overlaid.loc["2026_MISSING_GAME", "home_cover_probability"] == pytest.approx(0.50)


def test_overlay_disabled_is_a_no_op() -> None:
    predictions = _predictions()
    result = apply_division_revenge_tilt_overlay(predictions, _revenge_schedule(), enabled=False)

    assert result.flip_count == 0
    pd.testing.assert_frame_equal(
        result.overlaid_predictions.reset_index(drop=True),
        predictions.reset_index(drop=True),
        check_exact=True,
    )


def test_overlay_changes_only_home_cover_probability_on_flipped_rows() -> None:
    """Additivity: every other column, and every untouched row, stays
    byte-identical -- the pick-level design's whole point."""

    predictions = _predictions()
    result = apply_division_revenge_tilt_overlay(predictions, _revenge_schedule())
    overlaid = result.overlaid_predictions

    assert list(overlaid.columns) == list(predictions.columns)
    other_columns = [c for c in predictions.columns if c != "home_cover_probability"]
    pd.testing.assert_frame_equal(
        overlaid[other_columns].reset_index(drop=True),
        predictions[other_columns].reset_index(drop=True),
        check_exact=True,
    )
    untouched = predictions["game_id"].isin(
        [
            "2026_10_TIE2_TIE1",
            "2026_01_SOLO1_SOLO2",
            "2026_20_WIN_REV",
            "2026_10_WIN3_REV3",
            "2026_MISSING_GAME",
        ]
    )
    pd.testing.assert_series_equal(
        overlaid.loc[untouched, "home_cover_probability"].reset_index(drop=True),
        predictions.loc[untouched, "home_cover_probability"].reset_index(drop=True),
        check_exact=True,
    )


def test_overlay_requires_its_prediction_columns() -> None:
    with pytest.raises(DataContractError, match="overlay columns"):
        apply_division_revenge_tilt_overlay(pd.DataFrame({"game_id": ["G1"]}), _revenge_schedule())


# ---------------------------------------------------------------------------
# 3. overlay_disclosure_note: the plain-English provenance sentence
# ---------------------------------------------------------------------------


def test_disclosure_note_is_empty_when_nothing_flipped() -> None:
    tie_only = _predictions().loc[lambda frame: frame["game_id"].eq("2026_10_TIE2_TIE1")]
    result = apply_division_revenge_tilt_overlay(tie_only, _revenge_schedule())
    assert overlay_disclosure_note(result) == ""

    disabled = apply_division_revenge_tilt_overlay(
        _predictions(), _revenge_schedule(), enabled=False
    )
    assert overlay_disclosure_note(disabled) == ""


def test_disclosure_note_states_the_flip_count_and_does_not_claim_production() -> None:
    result = apply_division_revenge_tilt_overlay(_predictions(), _revenge_schedule())
    note = overlay_disclosure_note(result)

    assert "Tilt applied: 2 picks flipped" in note
    assert "WIN -> REV" in note
    assert "WIN2 -> REV2" in note
    assert "not applied to the published card" in note


# ---------------------------------------------------------------------------
# 4. record_division_revenge_tilt_challenger_decisions: dual-tracked, no window
# ---------------------------------------------------------------------------

_MODEL_CONFIG = {
    "method": "market_residual",
    "target": "market_residual",
    "regressor": "ridge",
    "ridge_alpha": 10.0,
    "calibration_method": "none",
    "feature_profile": "weak_stack",
    "min_edge": 0.02,
    "min_train_games": 500,
    "feature_table": "data/processed/game_features_weak_stack.parquet",
}


def _recorder_predictions() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "game_id": ["2026_10_WIN_REV", "2026_10_TIE2_TIE1"],
            "season": [2026, 2026],
            "week": [10, 10],
            "game_type": ["REG", "REG"],
            "home_team": ["WIN", "TIE2"],
            "away_team": ["REV", "TIE1"],
            "kickoff": ["2026-11-08T18:00:00+00:00", "2026-11-08T18:00:00+00:00"],
            "spread_line": [-3.0, -1.0],
            "home_cover_probability": [0.70, 0.60],
        }
    )


def _write_registry(artifacts: Path, *, status: str = "ACTIVE_PROSPECTIVE") -> None:
    payload = {
        "ledger": "prospective_challengers",
        "schema_version": 1,
        "challengers": [
            {
                "challenger_id": CHALLENGER_ID,
                "status": status,
                "model": dict(_MODEL_CONFIG),
            }
        ],
    }
    path = artifacts / "prospective" / "challengers.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_active_model_and_card(artifacts: Path, *, ridge_alpha: float = 10.0) -> None:
    forecast = artifacts / "margin_predictions" / "2026-week-10-forecast"
    forecast.mkdir(parents=True, exist_ok=True)
    metadata = {
        "active_model_id": "model-xyz",
        "synchronization_status": "SYNCHRONIZED",
        "season": 2026,
        "week": 10,
        "created_at_utc": "2026-11-03T15:00:00+00:00",
        "ats_method": "market_residual",
        "regressor": "ridge",
        "ridge_alpha": ridge_alpha,
        "calibration_method": "none",
        "feature_profile": "weak_stack",
        "min_edge": 0.02,
        "min_train_games": 500,
        "provenance": {
            "feature_table": {
                "path": "data/processed/game_features_weak_stack.parquet",
                "sha256": "abc123",
            }
        },
    }
    (forecast / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    _recorder_predictions().to_csv(forecast / "recommendations.csv", index=False)

    active = {
        "version": ACTIVE_ATS_MODEL_VERSION,
        "status": "SYNCHRONIZED",
        "model_id": "model-xyz",
        "method": "market_residual",
        "feature_profile": "weak_stack",
        "historical_evaluation": {"accuracy": 0.52, "correct": 1, "games": 1, "intervals": {}},
        "weekly_forecast": {
            "artifact": "margin_predictions/2026-week-10-forecast",
            "season": 2026,
            "week": 10,
        },
    }
    (artifacts / "active_ats_model.json").write_text(json.dumps(active), encoding="utf-8")


def _write_data_root(tmp_path: Path) -> Path:
    data_root = tmp_path / "data"
    write_snapshot(
        _revenge_schedule(),
        pd.DataFrame({"game_id": [], "team": []}),
        seasons=[2026],
        raw_root=data_root / "raw",
    )
    return data_root


def test_record_tilt_challenger_decisions_records_the_tilt_arm(tmp_path: Path) -> None:
    artifacts = tmp_path / "artifacts"
    _write_registry(artifacts)
    _write_active_model_and_card(artifacts)
    data_root = _write_data_root(tmp_path)
    now = datetime(2026, 11, 3, 16, 0, tzinfo=UTC)

    result = record_division_revenge_tilt_challenger_decisions(artifacts, data_root, now=now)

    assert result["recorded"] == 2
    assert result["flip_count"] == 1
    assert result["flipped_game_ids"] == ["2026_10_WIN_REV"]

    ledger = load_challenger_decisions(artifacts).set_index("game_id")
    assert list(load_challenger_decisions(artifacts).columns) == list(CHALLENGER_DECISION_COLUMNS)
    assert (ledger["bet_side"] == "PASS").all()
    assert ledger["edge"].isna().all()

    # The tilt's own arm diverges from the active model's raw pick (0.70 ->
    # HOME): the tilt flips it to AWAY (REV), the revenge side.
    assert ledger.loc["2026_10_WIN_REV", "pick_side"] == "AWAY"
    # The tied-first-meeting game keeps the model's own HOME pick.
    assert ledger.loc["2026_10_TIE2_TIE1", "pick_side"] == "HOME"

    # Re-running is a no-op: append-only, never rewrites.
    again = record_division_revenge_tilt_challenger_decisions(artifacts, data_root, now=now)
    assert again["recorded"] == 0
    assert again["already_recorded"] == 2


def test_record_tilt_challenger_refuses_outside_recording_lock_window(tmp_path: Path) -> None:
    artifacts = tmp_path / "artifacts"
    _write_registry(artifacts)
    _write_active_model_and_card(artifacts)
    data_root = _write_data_root(tmp_path)

    with pytest.raises(ValueError, match="RECORDING_LOCK_WINDOW"):
        record_division_revenge_tilt_challenger_decisions(
            artifacts, data_root, now=datetime(2026, 8, 18, 1, 0, tzinfo=UTC)
        )
    assert load_challenger_decisions(artifacts).empty


def test_record_tilt_challenger_refuses_a_fingerprint_mismatch(tmp_path: Path) -> None:
    artifacts = tmp_path / "artifacts"
    _write_registry(artifacts)
    # The active model's OWN configuration moved (a promotion) since this
    # challenger was pinned -- recording must refuse, not silently switch
    # base models under the same challenger id.
    _write_active_model_and_card(artifacts, ridge_alpha=1.0)
    data_root = _write_data_root(tmp_path)

    with pytest.raises(DataContractError, match="configuration fingerprint"):
        record_division_revenge_tilt_challenger_decisions(
            artifacts, data_root, now=datetime(2026, 11, 3, 16, 0, tzinfo=UTC)
        )
    assert load_challenger_decisions(artifacts).empty


def test_record_tilt_challenger_refuses_an_inactive_registration(tmp_path: Path) -> None:
    artifacts = tmp_path / "artifacts"
    _write_registry(artifacts, status="CLOSED_BEFORE_ACTIVATION")
    _write_active_model_and_card(artifacts)
    data_root = _write_data_root(tmp_path)

    with pytest.raises(ValueError, match="only ACTIVE_PROSPECTIVE"):
        record_division_revenge_tilt_challenger_decisions(
            artifacts, data_root, now=datetime(2026, 11, 3, 16, 0, tzinfo=UTC)
        )


def test_tilt_fingerprint_helper_agrees_with_the_registered_model_block() -> None:
    """Sanity check that the fixture's config really matches CONFIG_FINGERPRINT_KEYS."""

    metadata = {
        "ats_method": "market_residual",
        "regressor": "ridge",
        "ridge_alpha": 10.0,
        "calibration_method": "none",
        "feature_profile": "weak_stack",
        "min_edge": 0.02,
        "min_train_games": 500,
        "provenance": {
            "feature_table": {"path": "data/processed/game_features_weak_stack.parquet"}
        },
    }
    assert config_fingerprint(artifact_model_config(metadata)) == config_fingerprint(_MODEL_CONFIG)
