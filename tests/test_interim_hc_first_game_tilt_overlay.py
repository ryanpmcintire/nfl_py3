"""Interim head-coach first-game tilt overlay (docs/interim_coach_screen.md).

Four things are load-bearing here, mirroring
``tests/test_surface_switch_tilt_overlay.py``'s structure and AGENTS.md's
"add a leakage regression test for every new feature family" spirit:

1. :func:`interim_first_game_flag_by_game_fail_open` fires ONLY on the FIRST
   REG-season game of a specific interim stint (never the second+), is
   derived from data (reusing
   ``nfl_ats.experiment_runner._build_interim_coach_trait_data`` verbatim),
   and FAILS OPEN (returns zero flags, never raises) when the interim-coach
   source snapshot is unavailable.
2. :func:`apply_interim_hc_first_game_tilt_overlay` flips toward the
   interim-coached team's first game whenever the model's own pick is not
   already on that side, respects the REG-only gate, leaves a simultaneous
   both-sides-first-game case untouched, and is parameter-free.
3. :func:`overlay_disclosure_note` states the flip count and matchups.
4. :func:`record_interim_hc_first_game_tilt_challenger_decisions` writes the
   overlay's own picks to the prospective challenger ledger, dual-tracked
   and at no rotation-registry window cost.
"""

from __future__ import annotations

import json
import warnings
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import pytest

from nfl_ats.active_model import ACTIVE_ATS_MODEL_VERSION
from nfl_ats.data import DataContractError
from nfl_ats.interim_hc_first_game_tilt_overlay import (
    CHALLENGER_ID,
    apply_interim_hc_first_game_tilt_overlay,
    interim_first_game_flag_by_game_fail_open,
    overlay_disclosure_note,
    record_interim_hc_first_game_tilt_challenger_decisions,
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
# TEAMA fires its coach ("Old Coach A") after week 1 of 2026 and hires
#   "Interim One": week 2 (at OPP2, road) is TEAMA's FIRST game under the
#   interim -- flagged. Week 3 (home vs OPP3) is its SECOND game -- NOT
#   flagged (the folklore only covers game 1, per docs/interim_coach_screen.md
#   section 6's own decomposition).
# TEAMD and TEAME BOTH get new interims and play each other in week 4 -- a
#   simultaneous both-sides-first-game case, used for the "no measured
#   direction, leave untouched" test.
# TEAMB/TEAMC never appear in the interim-coach source at all -- used for the
#   "no interim this week, no-op" test.
#
# The REG-only eligibility gate is tested separately (see
# test_overlay_leaves_a_flagged_game_untouched_when_marked_postseason):
# nfl_ats.experiment_runner._build_interim_coach_trait_data itself only ever
# considers REG-season schedule rows when joining (see its own module
# docstring/source), so a POST-season game_id can never appear in the flags
# in the first place -- the gate in apply_interim_hc_first_game_tilt_overlay
# operates defensively on the PREDICTIONS frame's own game_type, mirroring
# every sibling overlay, and is exercised directly against that column
# rather than via a schedule fixture that the upstream join would reject.


def _interim_schedule() -> pd.DataFrame:
    rows = [
        # game_id, season, week, game_type, gameday, home_team, away_team, home_coach, away_coach
        (
            "2026_01_TEAMA_OPP1",
            2026,
            1,
            "REG",
            "2026-09-07",
            "TEAMA",
            "OPP1",
            "Old Coach A",
            "Opp1 Coach",
        ),
        (
            "2026_02_OPP2_TEAMA",
            2026,
            2,
            "REG",
            "2026-09-14",
            "OPP2",
            "TEAMA",
            "Opp2 Coach",
            "Interim One",
        ),
        (
            "2026_03_TEAMA_OPP3",
            2026,
            3,
            "REG",
            "2026-09-21",
            "TEAMA",
            "OPP3",
            "Interim One",
            "Opp3 Coach",
        ),
        (
            "2026_01_TEAMB_TEAMC",
            2026,
            1,
            "REG",
            "2026-09-07",
            "TEAMB",
            "TEAMC",
            "Coach B",
            "Coach C",
        ),
        (
            "2026_04_TEAMD_TEAME",
            2026,
            4,
            "REG",
            "2026-09-28",
            "TEAMD",
            "TEAME",
            "Interim D",
            "Interim E",
        ),
    ]
    return pd.DataFrame(
        rows,
        columns=[
            "game_id",
            "season",
            "week",
            "game_type",
            "gameday",
            "home_team",
            "away_team",
            "home_coach",
            "away_coach",
        ],
    )


def _parsed_table() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "entry_id": 1,
                "interim_coach_name": "Interim One",
                "team_abbr": "TEAMA",
                "predecessor_coach_name": "Old Coach A",
                "season": 2026,
                "joinable_2009plus": True,
                "predecessor_status": "fired",
                "takeover_date_iso": "2026-09-08",
            },
            {
                "entry_id": 2,
                "interim_coach_name": "Interim D",
                "team_abbr": "TEAMD",
                "predecessor_coach_name": "Old Coach D",
                "season": 2026,
                "joinable_2009plus": True,
                "predecessor_status": "fired",
                "takeover_date_iso": "2026-09-22",
            },
            {
                "entry_id": 3,
                "interim_coach_name": "Interim E",
                "team_abbr": "TEAME",
                "predecessor_coach_name": "Old Coach E",
                "season": 2026,
                "joinable_2009plus": True,
                "predecessor_status": "fired",
                "takeover_date_iso": "2026-09-22",
            },
        ]
    )


def _write_repo_root(tmp_path: Path, *, with_interim_data: bool = True) -> Path:
    repo_root = tmp_path / "repo"
    data_root = repo_root / "data"
    write_snapshot(
        _interim_schedule(),
        pd.DataFrame({"game_id": [], "team": []}),
        seasons=[2026],
        raw_root=data_root / "raw",
    )
    if with_interim_data:
        interim_dir = data_root / "raw" / "interim_coaches" / "20260820T000000Z"
        interim_dir.mkdir(parents=True)
        _parsed_table().to_csv(interim_dir / "parsed_table.csv", index=False)
    return repo_root


def _predictions() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "game_id": [
                "2026_02_OPP2_TEAMA",
                "2026_03_TEAMA_OPP3",
                "2026_01_TEAMB_TEAMC",
                "2026_04_TEAMD_TEAME",
                "2026_MISSING_GAME",
            ],
            "season": [2026, 2026, 2026, 2026, 2026],
            "week": [2, 3, 1, 4, 2],
            "game_type": ["REG", "REG", "REG", "REG", "REG"],
            "home_team": ["OPP2", "TEAMA", "TEAMB", "TEAMD", "MISS_H"],
            "away_team": ["TEAMA", "OPP3", "TEAMC", "TEAME", "MISS_A"],
            "kickoff": ["2026-09-24T17:00:00+00:00"] * 5,
            "spread_line": [-3.0, 2.0, -1.5, 1.0, 1.0],
            # G-first: model picks HOME (OPP2) -- NOT the interim team
            #   (TEAMA, away, first game) -- should flip to AWAY.
            # G-second: model picks AWAY (OPP3) -- TEAMA is home but this is
            #   its SECOND interim game -- not flagged, no flip.
            # G-nointerim: no interim data for either team -- no flip.
            # G-both: both TEAMD (home) and TEAME (away) are in their own
            #   first interim game simultaneously -- no measured direction,
            #   no flip regardless of the model's pick.
            # G-missing: no schedule row at all -- treated as no signal.
            "home_cover_probability": [0.60, 0.35, 0.50, 0.45, 0.50],
        }
    )


# ---------------------------------------------------------------------------
# 1. interim_first_game_flag_by_game_fail_open: derived, fail-open
# ---------------------------------------------------------------------------


def test_flag_fires_on_the_first_interim_game_only(tmp_path: Path) -> None:
    repo_root = _write_repo_root(tmp_path)
    flags = interim_first_game_flag_by_game_fail_open(repo_root)
    pairs = set(zip(flags["game_id"], flags["team"], strict=True))
    assert ("2026_02_OPP2_TEAMA", "TEAMA") in pairs
    assert ("2026_03_TEAMA_OPP3", "TEAMA") not in pairs


def test_flag_fires_for_both_sides_of_a_simultaneous_double_firing(tmp_path: Path) -> None:
    repo_root = _write_repo_root(tmp_path)
    flags = interim_first_game_flag_by_game_fail_open(repo_root)
    pairs = set(zip(flags["game_id"], flags["team"], strict=True))
    assert ("2026_04_TEAMD_TEAME", "TEAMD") in pairs
    assert ("2026_04_TEAMD_TEAME", "TEAME") in pairs


def test_flag_is_empty_for_a_team_never_under_an_interim(tmp_path: Path) -> None:
    repo_root = _write_repo_root(tmp_path)
    flags = interim_first_game_flag_by_game_fail_open(repo_root)
    pairs = set(zip(flags["game_id"], flags["team"], strict=True))
    assert not any(team in {"TEAMB", "TEAMC"} for _, team in pairs)


def test_flag_fails_open_with_no_interim_coach_snapshot(tmp_path: Path) -> None:
    """No data/raw/interim_coaches/*/parsed_table.csv snapshot at all -- must
    warn, not raise, and return an empty-but-well-formed frame."""

    repo_root = _write_repo_root(tmp_path, with_interim_data=False)
    with pytest.warns(RuntimeWarning, match="interim_hc_first_game_tilt"):
        flags = interim_first_game_flag_by_game_fail_open(repo_root)
    assert flags.empty
    assert list(flags.columns) == ["game_id", "team", "entry_id"]


def test_flag_fails_open_with_no_repo_data_directory_at_all(tmp_path: Path) -> None:
    empty_repo = tmp_path / "empty_repo"
    empty_repo.mkdir()
    with pytest.warns(RuntimeWarning):
        flags = interim_first_game_flag_by_game_fail_open(empty_repo)
    assert flags.empty


# ---------------------------------------------------------------------------
# 2. apply_interim_hc_first_game_tilt_overlay: the pick-level transform
# ---------------------------------------------------------------------------


def test_overlay_flips_toward_the_interim_teams_first_game(tmp_path: Path) -> None:
    repo_root = _write_repo_root(tmp_path)
    result = apply_interim_hc_first_game_tilt_overlay(_predictions(), repo_root)

    flipped_ids = {flip.game_id for flip in result.flips}
    assert "2026_02_OPP2_TEAMA" in flipped_ids
    flip = next(f for f in result.flips if f.game_id == "2026_02_OPP2_TEAMA")
    assert flip.interim_team == "TEAMA"
    assert flip.opponent_team == "OPP2"

    overlaid = result.overlaid_predictions.set_index("game_id")
    assert overlaid.loc["2026_02_OPP2_TEAMA", "home_cover_probability"] == pytest.approx(0.40)


def test_overlay_does_not_flip_the_second_game_of_a_stint(tmp_path: Path) -> None:
    repo_root = _write_repo_root(tmp_path)
    result = apply_interim_hc_first_game_tilt_overlay(_predictions(), repo_root)
    assert all(flip.game_id != "2026_03_TEAMA_OPP3" for flip in result.flips)
    overlaid = result.overlaid_predictions.set_index("game_id")
    assert overlaid.loc["2026_03_TEAMA_OPP3", "home_cover_probability"] == pytest.approx(0.35)


def test_overlay_does_not_flip_when_neither_side_is_under_an_interim(tmp_path: Path) -> None:
    repo_root = _write_repo_root(tmp_path)
    result = apply_interim_hc_first_game_tilt_overlay(_predictions(), repo_root)
    assert all(flip.game_id != "2026_01_TEAMB_TEAMC" for flip in result.flips)
    overlaid = result.overlaid_predictions.set_index("game_id")
    assert overlaid.loc["2026_01_TEAMB_TEAMC", "home_cover_probability"] == pytest.approx(0.50)


def test_overlay_leaves_a_simultaneous_double_firing_untouched(tmp_path: Path) -> None:
    """Both sides are in their own interim stint's first game -- no measured
    direction for that case, so it is flagged (both_first_game_games) but
    never flipped."""

    repo_root = _write_repo_root(tmp_path)
    result = apply_interim_hc_first_game_tilt_overlay(_predictions(), repo_root)
    assert all(flip.game_id != "2026_04_TEAMD_TEAME" for flip in result.flips)
    assert "2026_04_TEAMD_TEAME" in result.both_first_game_games
    overlaid = result.overlaid_predictions.set_index("game_id")
    assert overlaid.loc["2026_04_TEAMD_TEAME", "home_cover_probability"] == pytest.approx(0.45)


def test_overlay_leaves_a_flagged_game_untouched_when_marked_postseason(tmp_path: Path) -> None:
    """The REG-only eligibility gate operates on the PREDICTIONS frame's own
    ``game_type`` column (never the schedule's) -- exercised directly here
    since ``_build_interim_coach_trait_data`` itself only ever considers
    REG-season schedule rows when joining, so a POST-season game_id can
    never appear in the flags in the first place; this proves the gate
    still holds defensively, mirroring every sibling overlay's REG-only
    test, even though the real-world path to it is structurally blocked one
    layer up."""

    repo_root = _write_repo_root(tmp_path)
    predictions = pd.DataFrame(
        {
            "game_id": ["2026_02_OPP2_TEAMA"],
            "season": [2026],
            "week": [2],
            "game_type": ["POST"],
            "home_team": ["OPP2"],
            "away_team": ["TEAMA"],
            "kickoff": ["2026-09-24T17:00:00+00:00"],
            "spread_line": [-3.0],
            "home_cover_probability": [0.60],
        }
    )
    result = apply_interim_hc_first_game_tilt_overlay(predictions, repo_root)
    assert result.flip_count == 0
    overlaid = result.overlaid_predictions.set_index("game_id")
    assert overlaid.loc["2026_02_OPP2_TEAMA", "home_cover_probability"] == pytest.approx(0.60)


def test_overlay_treats_a_missing_schedule_row_as_no_signal(tmp_path: Path) -> None:
    repo_root = _write_repo_root(tmp_path)
    result = apply_interim_hc_first_game_tilt_overlay(_predictions(), repo_root)
    assert all(flip.game_id != "2026_MISSING_GAME" for flip in result.flips)
    overlaid = result.overlaid_predictions.set_index("game_id")
    assert overlaid.loc["2026_MISSING_GAME", "home_cover_probability"] == pytest.approx(0.50)


def test_overlay_disabled_is_a_no_op(tmp_path: Path) -> None:
    repo_root = _write_repo_root(tmp_path)
    predictions = _predictions()
    result = apply_interim_hc_first_game_tilt_overlay(predictions, repo_root, enabled=False)

    assert result.flip_count == 0
    pd.testing.assert_frame_equal(
        result.overlaid_predictions.reset_index(drop=True),
        predictions.reset_index(drop=True),
        check_exact=True,
    )


def test_overlay_fails_open_with_no_interim_coach_snapshot(tmp_path: Path) -> None:
    """The whole point of fail-open: no interim-coach source data at all
    must fall back to the model's own picks for every game, never crash or
    block the caller."""

    repo_root = _write_repo_root(tmp_path, with_interim_data=False)
    predictions = _predictions()
    with pytest.warns(RuntimeWarning):
        result = apply_interim_hc_first_game_tilt_overlay(predictions, repo_root)

    assert result.flip_count == 0
    pd.testing.assert_frame_equal(
        result.overlaid_predictions.reset_index(drop=True),
        predictions.reset_index(drop=True),
        check_exact=True,
    )


def test_overlay_changes_only_home_cover_probability_on_flipped_rows(tmp_path: Path) -> None:
    """Additivity: every other column, and every untouched row, stays
    byte-identical -- the pick-level design's whole point."""

    repo_root = _write_repo_root(tmp_path)
    predictions = _predictions()
    result = apply_interim_hc_first_game_tilt_overlay(predictions, repo_root)
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
            "2026_03_TEAMA_OPP3",
            "2026_01_TEAMB_TEAMC",
            "2026_04_TEAMD_TEAME",
            "2026_MISSING_GAME",
        ]
    )
    pd.testing.assert_series_equal(
        overlaid.loc[untouched, "home_cover_probability"].reset_index(drop=True),
        predictions.loc[untouched, "home_cover_probability"].reset_index(drop=True),
        check_exact=True,
    )


def test_overlay_requires_its_prediction_columns(tmp_path: Path) -> None:
    with pytest.raises(DataContractError, match="overlay columns"):
        apply_interim_hc_first_game_tilt_overlay(pd.DataFrame({"game_id": ["G1"]}), tmp_path)


# ---------------------------------------------------------------------------
# 3. overlay_disclosure_note: the plain-English provenance sentence
# ---------------------------------------------------------------------------


def test_disclosure_note_is_empty_when_nothing_flipped(tmp_path: Path) -> None:
    repo_root = _write_repo_root(tmp_path)
    matched_only = _predictions().loc[lambda frame: frame["game_id"].eq("2026_01_TEAMB_TEAMC")]
    result = apply_interim_hc_first_game_tilt_overlay(matched_only, repo_root)
    assert overlay_disclosure_note(result) == ""

    disabled = apply_interim_hc_first_game_tilt_overlay(_predictions(), repo_root, enabled=False)
    assert overlay_disclosure_note(disabled) == ""


def test_disclosure_note_states_the_flip_count_and_does_not_claim_production(
    tmp_path: Path,
) -> None:
    repo_root = _write_repo_root(tmp_path)
    result = apply_interim_hc_first_game_tilt_overlay(_predictions(), repo_root)
    note = overlay_disclosure_note(result)

    assert "Tilt applied: 1 pick flipped" in note
    assert "OPP2 -> TEAMA" in note
    assert "not applied to the published card" in note


# ---------------------------------------------------------------------------
# 4. record_interim_hc_first_game_tilt_challenger_decisions: dual-tracked
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
            "game_id": ["2026_02_OPP2_TEAMA", "2026_01_TEAMB_TEAMC"],
            "season": [2026, 2026],
            "week": [2, 1],
            "game_type": ["REG", "REG"],
            "home_team": ["OPP2", "TEAMB"],
            "away_team": ["TEAMA", "TEAMC"],
            "kickoff": ["2026-09-24T17:00:00+00:00", "2026-09-24T17:00:00+00:00"],
            "spread_line": [-3.0, 2.0],
            "home_cover_probability": [0.60, 0.50],
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
    forecast = artifacts / "margin_predictions" / "2026-week-02-forecast"
    forecast.mkdir(parents=True, exist_ok=True)
    metadata = {
        "active_model_id": "model-xyz",
        "synchronization_status": "SYNCHRONIZED",
        "season": 2026,
        "week": 2,
        "created_at_utc": "2026-09-17T15:00:00+00:00",
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
            "artifact": "margin_predictions/2026-week-02-forecast",
            "season": 2026,
            "week": 2,
        },
    }
    (artifacts / "active_ats_model.json").write_text(json.dumps(active), encoding="utf-8")


def test_record_interim_hc_first_game_challenger_decisions_records_the_tilt_arm(
    tmp_path: Path,
) -> None:
    artifacts = tmp_path / "artifacts"
    _write_registry(artifacts)
    _write_active_model_and_card(artifacts)
    repo_root = _write_repo_root(tmp_path)
    data_root = repo_root / "data"
    now = datetime(2026, 9, 20, 16, 0, tzinfo=UTC)

    result = record_interim_hc_first_game_tilt_challenger_decisions(artifacts, data_root, now=now)

    assert result["recorded"] == 2
    assert result["flip_count"] == 1
    assert result["flipped_game_ids"] == ["2026_02_OPP2_TEAMA"]

    ledger = load_challenger_decisions(artifacts).set_index("game_id")
    assert list(load_challenger_decisions(artifacts).columns) == list(CHALLENGER_DECISION_COLUMNS)
    assert (ledger["bet_side"] == "PASS").all()
    assert ledger["edge"].isna().all()

    # The tilt's own arm diverges from the active model's raw pick (0.60 ->
    # HOME): the tilt flips it to AWAY (TEAMA), toward the interim team's
    # first game.
    assert ledger.loc["2026_02_OPP2_TEAMA", "pick_side"] == "AWAY"
    # The no-signal game keeps the model's own pick (0.50 -> HOME by the
    # >= 0.5 convention).
    assert ledger.loc["2026_01_TEAMB_TEAMC", "pick_side"] == "HOME"

    # Re-running is a no-op: append-only, never rewrites.
    again = record_interim_hc_first_game_tilt_challenger_decisions(artifacts, data_root, now=now)
    assert again["recorded"] == 0
    assert again["already_recorded"] == 2


def test_record_interim_hc_first_game_challenger_decisions_fails_open_with_no_interim_data(
    tmp_path: Path,
) -> None:
    """Recording must still succeed (both games recorded, un-flipped) when
    the interim-coach source snapshot is unavailable -- the fail-open
    contract must hold at the recording layer too, not just apply_*."""

    artifacts = tmp_path / "artifacts"
    _write_registry(artifacts)
    _write_active_model_and_card(artifacts)
    repo_root = _write_repo_root(tmp_path, with_interim_data=False)
    data_root = repo_root / "data"
    now = datetime(2026, 9, 20, 16, 0, tzinfo=UTC)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        result = record_interim_hc_first_game_tilt_challenger_decisions(
            artifacts, data_root, now=now
        )

    assert result["recorded"] == 2
    assert result["flip_count"] == 0
    ledger = load_challenger_decisions(artifacts).set_index("game_id")
    assert ledger.loc["2026_02_OPP2_TEAMA", "pick_side"] == "HOME"


def test_record_interim_hc_first_game_challenger_refuses_outside_recording_lock_window(
    tmp_path: Path,
) -> None:
    artifacts = tmp_path / "artifacts"
    _write_registry(artifacts)
    _write_active_model_and_card(artifacts)
    repo_root = _write_repo_root(tmp_path)
    data_root = repo_root / "data"

    with pytest.raises(ValueError, match="RECORDING_LOCK_WINDOW"):
        record_interim_hc_first_game_tilt_challenger_decisions(
            artifacts, data_root, now=datetime(2026, 8, 18, 1, 0, tzinfo=UTC)
        )
    assert load_challenger_decisions(artifacts).empty


def test_record_interim_hc_first_game_challenger_refuses_a_fingerprint_mismatch(
    tmp_path: Path,
) -> None:
    artifacts = tmp_path / "artifacts"
    _write_registry(artifacts)
    # The active model's OWN configuration moved (a promotion) since this
    # challenger was pinned -- recording must refuse, not silently switch
    # base models under the same challenger id.
    _write_active_model_and_card(artifacts, ridge_alpha=1.0)
    repo_root = _write_repo_root(tmp_path)
    data_root = repo_root / "data"

    with pytest.raises(DataContractError, match="configuration fingerprint"):
        record_interim_hc_first_game_tilt_challenger_decisions(
            artifacts, data_root, now=datetime(2026, 9, 20, 16, 0, tzinfo=UTC)
        )
    assert load_challenger_decisions(artifacts).empty


def test_record_interim_hc_first_game_challenger_refuses_an_inactive_registration(
    tmp_path: Path,
) -> None:
    artifacts = tmp_path / "artifacts"
    _write_registry(artifacts, status="CLOSED_BEFORE_ACTIVATION")
    _write_active_model_and_card(artifacts)
    repo_root = _write_repo_root(tmp_path)
    data_root = repo_root / "data"

    with pytest.raises(ValueError, match="only ACTIVE_PROSPECTIVE"):
        record_interim_hc_first_game_tilt_challenger_decisions(
            artifacts, data_root, now=datetime(2026, 9, 20, 16, 0, tzinfo=UTC)
        )


def test_interim_hc_first_game_fingerprint_helper_agrees_with_the_registered_model_block() -> None:
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
