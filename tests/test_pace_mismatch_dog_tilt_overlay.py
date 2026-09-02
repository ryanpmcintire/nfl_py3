"""Pace-mismatch dog tilt overlay (docs/pace_mismatch_dog_tilt_overlay.md).

Four things load-bearing, mirroring the sibling overlay test files'
structure (``tests/test_spread_gap_zone_fade_overlay.py``,
``tests/test_surface_switch_tilt_overlay.py``,
``tests/test_pbp08_protection_mismatch_tilt_overlay.py``):

1. :func:`pace_mismatch_flag_by_game` -- the trait/quartile-cut flag,
   reproducing the screen's own construction (PRIOR-SEASON join, frozen
   numeric threshold, ``>=`` comparator, missing data -> False never an
   error), plus leakage regression tests (AGENTS.md): no outcome columns
   read, no future-season contamination, and -- the construct-specific test
   this rule needs -- no CURRENT-season contamination (only the PRIOR season
   may ever be read).
2. :func:`apply_pace_mismatch_dog_tilt_overlay` -- the pick-level transform:
   both spread directions (home favourite and road favourite each flip to
   the correct underdog side), pick'em exclusion, already-underdog no-op, no
   effect outside the flagged population, REG-only gate.
3. :func:`overlay_disclosure_note`.
4. :func:`record_pace_mismatch_dog_tilt_challenger_decisions` -- dual-tracked,
   no window, fingerprint/status gates.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from _overlay_test_kit import write_active_model_and_card, write_challenger_registry

from nfl_ats.data import DataContractError
from nfl_ats.pace_mismatch_dog_tilt_overlay import (
    CHALLENGER_ID,
    PACE_DIFF_ABS_THRESHOLD,
    apply_pace_mismatch_dog_tilt_overlay,
    overlay_disclosure_note,
    pace_mismatch_flag_by_game,
    record_pace_mismatch_dog_tilt_challenger_decisions,
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
# Prior-season (2025) team pace-centered values, chosen to straddle the
# frozen threshold (PACE_DIFF_ABS_THRESHOLD ~= 2.1685):
#
#   FAST / SLOW    -> |2.0 - (-1.185)| = 3.185              >= threshold (flagged)
#   MID1 / MID2    -> |0.5 - 0.3|      = 0.2                <  threshold (not flagged)
#   EDGEHI/EDGELO  -> exactly PACE_DIFF_ABS_THRESHOLD                      (boundary, flagged)
#   NEWX has a NaN prior-season pace value; ROOKIE has no 2025 row at all.


def _team_season_style() -> pd.DataFrame:
    half = PACE_DIFF_ABS_THRESHOLD / 2.0
    return pd.DataFrame(
        {
            "season": [2025, 2025, 2025, 2025, 2025, 2025, 2025],
            "team": ["FAST", "SLOW", "MID1", "MID2", "NEWX", "EDGEHI", "EDGELO"],
            "seconds_per_play_pace_centered": [2.0, -1.185, 0.5, 0.3, np.nan, half, -half],
        }
    )


def _schedule() -> pd.DataFrame:
    rows = [
        ("2026_05_HOMEFAV", 2026, 5, "REG", "FAST", "SLOW"),
        ("2026_05_AWAYFAV", 2026, 5, "REG", "SLOW", "FAST"),
        ("2026_05_ALREADYDOG", 2026, 5, "REG", "FAST", "SLOW"),
        ("2026_05_PICKEM", 2026, 5, "REG", "FAST", "SLOW"),
        ("2026_05_NOTFLAGGED", 2026, 5, "REG", "MID1", "MID2"),
        ("2026_05_MISSINGTEAM", 2026, 5, "REG", "FAST", "ROOKIE"),
        ("2026_05_MISSINGVALUE", 2026, 5, "REG", "FAST", "NEWX"),
        ("2026_05_BOUNDARY", 2026, 5, "REG", "EDGEHI", "EDGELO"),
        ("2026_20_POSTFLAG", 2026, 20, "POST", "FAST", "SLOW"),
    ]
    return pd.DataFrame(
        rows, columns=["game_id", "season", "week", "game_type", "home_team", "away_team"]
    )


def _predictions() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "game_id": [
                "2026_05_HOMEFAV",
                "2026_05_AWAYFAV",
                "2026_05_ALREADYDOG",
                "2026_05_PICKEM",
                "2026_05_NOTFLAGGED",
                "2026_05_MISSINGTEAM",
                "2026_05_MISSINGVALUE",
                "2026_05_BOUNDARY",
                "2026_20_POSTFLAG",
            ],
            "season": [2026] * 9,
            "week": [5, 5, 5, 5, 5, 5, 5, 5, 20],
            "game_type": ["REG"] * 8 + ["POST"],
            "home_team": ["FAST", "SLOW", "FAST", "FAST", "MID1", "FAST", "FAST", "EDGEHI", "FAST"],
            "away_team": [
                "SLOW",
                "FAST",
                "SLOW",
                "SLOW",
                "MID2",
                "ROOKIE",
                "NEWX",
                "EDGELO",
                "SLOW",
            ],
            "spread_line": [6.5, -6.5, 6.5, 0.0, 6.5, 6.5, 6.5, 3.0, 6.5],
            "home_cover_probability": [0.62, 0.35, 0.42, 0.55, 0.60, 0.60, 0.60, 0.55, 0.62],
        }
    )


def _flags() -> pd.DataFrame:
    return pace_mismatch_flag_by_game(_schedule(), _team_season_style())


# ---------------------------------------------------------------------------
# 1. pace_mismatch_flag_by_game: the trait, quartile cut, and its leakage
# ---------------------------------------------------------------------------


def test_frozen_threshold_matches_the_measured_screen_cut() -> None:
    assert pytest.approx(2.1685022294778378) == PACE_DIFF_ABS_THRESHOLD


def test_flag_fires_for_a_top_quartile_prior_season_pace_mismatch() -> None:
    flags = _flags().set_index("game_id")
    assert flags.loc["2026_05_HOMEFAV", "pace_diff_abs"] == pytest.approx(3.185)
    assert bool(flags.loc["2026_05_HOMEFAV", "pace_mismatch_flag"]) is True


def test_flag_is_false_below_the_frozen_threshold() -> None:
    flags = _flags().set_index("game_id")
    assert flags.loc["2026_05_NOTFLAGGED", "pace_diff_abs"] == pytest.approx(0.2)
    assert bool(flags.loc["2026_05_NOTFLAGGED", "pace_mismatch_flag"]) is False


def test_flag_boundary_is_inclusive_at_the_frozen_threshold() -> None:
    """``>=`` per scripts/team_style_screen.py:468, not ``>``."""

    flags = _flags().set_index("game_id")
    assert flags.loc["2026_05_BOUNDARY", "pace_diff_abs"] == pytest.approx(PACE_DIFF_ABS_THRESHOLD)
    assert bool(flags.loc["2026_05_BOUNDARY", "pace_mismatch_flag"]) is True


def test_flag_is_false_never_an_error_with_no_prior_season_row_at_all() -> None:
    flags = _flags().set_index("game_id")
    assert pd.isna(flags.loc["2026_05_MISSINGTEAM", "pace_diff_abs"])
    assert bool(flags.loc["2026_05_MISSINGTEAM", "pace_mismatch_flag"]) is False


def test_flag_is_false_never_an_error_with_a_nan_prior_season_value() -> None:
    flags = _flags().set_index("game_id")
    assert pd.isna(flags.loc["2026_05_MISSINGVALUE", "pace_diff_abs"])
    assert bool(flags.loc["2026_05_MISSINGVALUE", "pace_mismatch_flag"]) is False


def test_flag_requires_its_schedule_columns() -> None:
    with pytest.raises(DataContractError, match="pace-mismatch"):
        pace_mismatch_flag_by_game(pd.DataFrame({"game_id": ["G1"]}), _team_season_style())


def test_flag_requires_its_team_season_style_columns() -> None:
    with pytest.raises(DataContractError, match="pace-mismatch"):
        pace_mismatch_flag_by_game(_schedule(), pd.DataFrame({"team": ["FAST"]}))


def test_flag_never_reads_outcome_columns() -> None:
    """AGENTS.md: a leakage regression test for every new feature family.

    ``pace_mismatch_flag_by_game`` does not even require/read ``result`` or
    ``spread_line`` -- adding them (with arbitrary values) and mutating them
    must never change the already-computed flags.
    """

    schedule = _schedule()
    schedule["result"] = 0.0
    schedule["spread_line"] = -3.0
    baseline = pace_mismatch_flag_by_game(schedule, _team_season_style()).set_index("game_id")

    mutated = schedule.copy()
    mutated.loc[mutated["game_id"].eq("2026_05_HOMEFAV"), "result"] = 99.0
    mutated.loc[mutated["game_id"].eq("2026_05_HOMEFAV"), "spread_line"] = 14.0
    changed = pace_mismatch_flag_by_game(mutated, _team_season_style()).set_index("game_id")

    pd.testing.assert_frame_equal(changed, baseline, check_exact=True)


def test_flag_is_leak_safe_across_a_future_season_boundary() -> None:
    """A future season's schedule/style data must never change an earlier
    season's already-computed flags."""

    baseline = pace_mismatch_flag_by_game(_schedule(), _team_season_style())

    future_schedule = pd.concat(
        [
            _schedule(),
            pd.DataFrame(
                [("2027_03_FUTURE", 2027, 3, "REG", "FAST", "SLOW")],
                columns=_schedule().columns,
            ),
        ],
        ignore_index=True,
    )
    future_style = pd.concat(
        [
            _team_season_style(),
            pd.DataFrame(
                {"season": [2026], "team": ["FAST"], "seconds_per_play_pace_centered": [0.0]}
            ),
        ],
        ignore_index=True,
    )
    changed = pace_mismatch_flag_by_game(future_schedule, future_style)

    pd.testing.assert_frame_equal(
        changed.loc[changed["season"].le(2026)].reset_index(drop=True),
        baseline.loc[baseline["season"].le(2026)].reset_index(drop=True),
        check_exact=True,
    )


def test_flag_uses_only_the_prior_season_never_the_current_seasons_data() -> None:
    """Pregame-only, the construct-specific leakage check: a mutated
    CURRENT-season (2026) pace row for a team playing IN 2026 must never
    change that game's flag -- only the season=2025 PRIOR row may be read
    (scripts/team_style_screen.py's ``_prior`` shift-by-one convention,
    ported in this module's own prior-season merge).
    """

    baseline = pace_mismatch_flag_by_game(_schedule(), _team_season_style()).set_index("game_id")

    contaminated_style = pd.concat(
        [
            _team_season_style(),
            pd.DataFrame(
                {
                    "season": [2026, 2026],
                    "team": ["FAST", "SLOW"],
                    # Current-season values that would ERASE the mismatch
                    # (diff 0.0) if the current season were ever read instead
                    # of the prior one.
                    "seconds_per_play_pace_centered": [0.0, 0.0],
                }
            ),
        ],
        ignore_index=True,
    )
    changed = pace_mismatch_flag_by_game(_schedule(), contaminated_style).set_index("game_id")

    pd.testing.assert_frame_equal(changed, baseline, check_exact=True)
    assert bool(changed.loc["2026_05_HOMEFAV", "pace_mismatch_flag"]) is True


# ---------------------------------------------------------------------------
# 2. apply_pace_mismatch_dog_tilt_overlay: the pick-level transform
# ---------------------------------------------------------------------------


def test_overlay_flips_a_home_favourite_pick_to_the_underdog() -> None:
    """Spread direction 1 of 2: ``spread_line > 0`` (home favoured), model
    holds the favourite (home) -- flips to the away underdog."""

    result = apply_pace_mismatch_dog_tilt_overlay(_predictions(), _flags())

    flipped_ids = {flip.game_id for flip in result.flips}
    assert "2026_05_HOMEFAV" in flipped_ids
    flip = next(f for f in result.flips if f.game_id == "2026_05_HOMEFAV")
    assert flip.original_pick_team == "FAST"
    assert flip.flipped_to_team == "SLOW"

    overlaid = result.overlaid_predictions.set_index("game_id")
    assert overlaid.loc["2026_05_HOMEFAV", "home_cover_probability"] == pytest.approx(0.38)


def test_overlay_flips_an_away_favourite_pick_to_the_underdog() -> None:
    """Spread direction 2 of 2: ``spread_line < 0`` (away favoured), model
    holds the favourite (away) -- flips to the home underdog."""

    result = apply_pace_mismatch_dog_tilt_overlay(_predictions(), _flags())

    flipped_ids = {flip.game_id for flip in result.flips}
    assert "2026_05_AWAYFAV" in flipped_ids
    flip = next(f for f in result.flips if f.game_id == "2026_05_AWAYFAV")
    assert flip.original_pick_team == "FAST"
    assert flip.flipped_to_team == "SLOW"

    overlaid = result.overlaid_predictions.set_index("game_id")
    assert overlaid.loc["2026_05_AWAYFAV", "home_cover_probability"] == pytest.approx(0.65)


def test_overlay_leaves_a_pick_already_on_the_underdog_untouched() -> None:
    result = apply_pace_mismatch_dog_tilt_overlay(_predictions(), _flags())
    assert all(flip.game_id != "2026_05_ALREADYDOG" for flip in result.flips)
    overlaid = result.overlaid_predictions.set_index("game_id")
    assert overlaid.loc["2026_05_ALREADYDOG", "home_cover_probability"] == pytest.approx(0.42)


def test_overlay_never_touches_a_pickem_game() -> None:
    """``spread_line == 0``: no defined underdog, never touched -- explicit
    per the frozen rule, regardless of the pace flag or the model's pick."""

    result = apply_pace_mismatch_dog_tilt_overlay(_predictions(), _flags())
    assert all(flip.game_id != "2026_05_PICKEM" for flip in result.flips)
    overlaid = result.overlaid_predictions.set_index("game_id")
    assert overlaid.loc["2026_05_PICKEM", "home_cover_probability"] == pytest.approx(0.55)


def test_overlay_has_no_effect_outside_the_flagged_population() -> None:
    result = apply_pace_mismatch_dog_tilt_overlay(_predictions(), _flags())
    assert all(flip.game_id != "2026_05_NOTFLAGGED" for flip in result.flips)
    overlaid = result.overlaid_predictions.set_index("game_id")
    assert overlaid.loc["2026_05_NOTFLAGGED", "home_cover_probability"] == pytest.approx(0.60)


def test_overlay_never_touches_games_with_missing_prior_season_data() -> None:
    result = apply_pace_mismatch_dog_tilt_overlay(_predictions(), _flags())
    flipped_ids = {flip.game_id for flip in result.flips}
    assert "2026_05_MISSINGTEAM" not in flipped_ids
    assert "2026_05_MISSINGVALUE" not in flipped_ids


def test_overlay_flips_at_the_frozen_boundary_inclusive() -> None:
    result = apply_pace_mismatch_dog_tilt_overlay(_predictions(), _flags())
    assert any(flip.game_id == "2026_05_BOUNDARY" for flip in result.flips)


def test_overlay_leaves_postseason_games_untouched() -> None:
    result = apply_pace_mismatch_dog_tilt_overlay(_predictions(), _flags())
    assert all(flip.game_id != "2026_20_POSTFLAG" for flip in result.flips)


def test_overlay_flip_set_is_exactly_the_expected_games() -> None:
    """No effect anywhere outside the three intended flips -- the strongest
    form of the "no effect outside the flagged population" guarantee."""

    result = apply_pace_mismatch_dog_tilt_overlay(_predictions(), _flags())
    flipped_ids = {flip.game_id for flip in result.flips}
    assert flipped_ids == {"2026_05_HOMEFAV", "2026_05_AWAYFAV", "2026_05_BOUNDARY"}


def test_overlay_disabled_is_a_no_op() -> None:
    predictions = _predictions()
    result = apply_pace_mismatch_dog_tilt_overlay(predictions, _flags(), enabled=False)

    assert result.flip_count == 0
    pd.testing.assert_frame_equal(
        result.overlaid_predictions.reset_index(drop=True),
        predictions.reset_index(drop=True),
        check_exact=True,
    )


def test_overlay_with_an_empty_flag_table_is_a_documented_no_op_not_a_crash() -> None:
    predictions = _predictions()
    result = apply_pace_mismatch_dog_tilt_overlay(predictions, pd.DataFrame())

    assert result.flip_count == 0
    pd.testing.assert_frame_equal(
        result.overlaid_predictions.reset_index(drop=True),
        predictions.reset_index(drop=True),
        check_exact=True,
    )


def test_overlay_changes_only_home_cover_probability_on_flipped_rows() -> None:
    """Additivity: every other column, and every untouched row, stays
    byte-identical."""

    predictions = _predictions()
    result = apply_pace_mismatch_dog_tilt_overlay(predictions, _flags())
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
            "2026_05_ALREADYDOG",
            "2026_05_PICKEM",
            "2026_05_NOTFLAGGED",
            "2026_05_MISSINGTEAM",
            "2026_05_MISSINGVALUE",
            "2026_20_POSTFLAG",
        ]
    )
    pd.testing.assert_series_equal(
        overlaid.loc[untouched, "home_cover_probability"].reset_index(drop=True),
        predictions.loc[untouched, "home_cover_probability"].reset_index(drop=True),
        check_exact=True,
    )


def test_overlay_requires_its_prediction_columns() -> None:
    with pytest.raises(DataContractError, match="overlay columns"):
        apply_pace_mismatch_dog_tilt_overlay(pd.DataFrame({"game_id": ["G1"]}), _flags())


# ---------------------------------------------------------------------------
# 3. overlay_disclosure_note: the plain-English provenance sentence
# ---------------------------------------------------------------------------


def test_disclosure_note_is_empty_when_nothing_flipped() -> None:
    not_flagged_only = _predictions().loc[lambda frame: frame["game_id"].eq("2026_05_NOTFLAGGED")]
    result = apply_pace_mismatch_dog_tilt_overlay(not_flagged_only, _flags())
    assert overlay_disclosure_note(result) == ""

    disabled = apply_pace_mismatch_dog_tilt_overlay(_predictions(), _flags(), enabled=False)
    assert overlay_disclosure_note(disabled) == ""


def test_disclosure_note_states_the_flip_count_and_does_not_claim_production() -> None:
    result = apply_pace_mismatch_dog_tilt_overlay(_predictions(), _flags())
    note = overlay_disclosure_note(result)

    assert "Tilt applied: 3 picks flipped" in note
    assert "FAST -> SLOW" in note
    assert "not applied to the published card" in note


# ---------------------------------------------------------------------------
# 4. record_pace_mismatch_dog_tilt_challenger_decisions: dual-tracked, no
#    window, fingerprint/status gates
# ---------------------------------------------------------------------------

_MODEL_CONFIG = {
    "method": "market_residual",
    "target": "market_residual",
    "regressor": "ridge",
    "ridge_alpha": 10.0,
    "calibration_method": "none",
    "feature_profile": "weak_stack",
    "feature_set": "full_weak_stack",
    "min_edge": 0.02,
    "min_train_games": 500,
    "feature_table": "data/processed/game_features_weak_stack.parquet",
}


def _recorder_predictions() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "game_id": ["2026_05_HOMEFAV", "2026_05_NOTFLAGGED"],
            "season": [2026, 2026],
            "week": [5, 5],
            "game_type": ["REG", "REG"],
            "home_team": ["FAST", "MID1"],
            "away_team": ["SLOW", "MID2"],
            "kickoff": ["2026-10-08T17:00:00+00:00", "2026-10-08T17:00:00+00:00"],
            "spread_line": [6.5, 6.5],
            "home_cover_probability": [0.62, 0.60],
        }
    )


def _write_registry(artifacts: Path, *, status: str = "ACTIVE_PROSPECTIVE") -> None:
    write_challenger_registry(
        artifacts, challenger_id=CHALLENGER_ID, model_config=_MODEL_CONFIG, status=status
    )


_SEASON, _WEEK, _CREATED_AT_UTC = 2026, 5, "2026-10-01T15:00:00+00:00"


def _write_active_model_and_card(artifacts: Path, *, ridge_alpha: float = 10.0) -> None:
    write_active_model_and_card(
        artifacts,
        season=_SEASON,
        week=_WEEK,
        created_at_utc=_CREATED_AT_UTC,
        ridge_alpha=ridge_alpha,
        recommendations=_recorder_predictions(),
    )


def _write_data_root(tmp_path: Path) -> Path:
    """A real schedule snapshot + team-season pace cache, so the recorder's
    fail-open flag build actually finds data and can produce a real flip."""

    data_root = tmp_path / "data"
    empty_team_stats = pd.DataFrame({"season": pd.Series(dtype="int64")})
    write_snapshot(_schedule(), empty_team_stats, seasons=[2025, 2026], raw_root=data_root / "raw")
    style_dir = data_root / "pbp" / "team_style"
    style_dir.mkdir(parents=True, exist_ok=True)
    _team_season_style().to_parquet(style_dir / "team_season_style.parquet", index=False)
    return data_root


def test_record_pace_mismatch_challenger_decisions_records_the_tilt_arm(tmp_path: Path) -> None:
    artifacts = tmp_path / "artifacts"
    _write_registry(artifacts)
    _write_active_model_and_card(artifacts)
    data_root = _write_data_root(tmp_path)
    now = datetime(2026, 10, 4, 16, 0, tzinfo=UTC)

    result = record_pace_mismatch_dog_tilt_challenger_decisions(artifacts, data_root, now=now)

    assert result["recorded"] == 2
    assert result["flip_count"] == 1
    assert result["flipped_game_ids"] == ["2026_05_HOMEFAV"]

    ledger = load_challenger_decisions(artifacts).set_index("game_id")
    assert list(load_challenger_decisions(artifacts).columns) == list(CHALLENGER_DECISION_COLUMNS)
    assert (ledger["bet_side"] == "PASS").all()
    assert ledger["edge"].isna().all()

    # The tilt's own arm diverges from the active model's raw pick (0.62 ->
    # HOME): the pace mismatch flips it to the underdog AWAY.
    assert ledger.loc["2026_05_HOMEFAV", "pick_side"] == "AWAY"
    # The not-flagged game keeps the model's own HOME pick untouched.
    assert ledger.loc["2026_05_NOTFLAGGED", "pick_side"] == "HOME"

    # Re-running is a no-op: append-only, never rewrites.
    again = record_pace_mismatch_dog_tilt_challenger_decisions(artifacts, data_root, now=now)
    assert again["recorded"] == 0
    assert again["already_recorded"] == 2


def test_record_pace_mismatch_challenger_refuses_a_fingerprint_mismatch(tmp_path: Path) -> None:
    artifacts = tmp_path / "artifacts"
    _write_registry(artifacts)
    # The active model's OWN configuration moved (a promotion) since this
    # challenger was pinned -- recording must refuse, not silently switch
    # base models under the same challenger id.
    _write_active_model_and_card(artifacts, ridge_alpha=1.0)
    data_root = tmp_path / "data"

    with pytest.raises(DataContractError, match="configuration fingerprint"):
        record_pace_mismatch_dog_tilt_challenger_decisions(
            artifacts, data_root, now=datetime(2026, 10, 4, 16, 0, tzinfo=UTC)
        )
    assert load_challenger_decisions(artifacts).empty


def test_record_pace_mismatch_challenger_refuses_an_inactive_registration(tmp_path: Path) -> None:
    artifacts = tmp_path / "artifacts"
    _write_registry(artifacts, status="CLOSED_BEFORE_ACTIVATION")
    _write_active_model_and_card(artifacts)
    data_root = tmp_path / "data"

    with pytest.raises(ValueError, match="only ACTIVE_PROSPECTIVE"):
        record_pace_mismatch_dog_tilt_challenger_decisions(
            artifacts, data_root, now=datetime(2026, 10, 4, 16, 0, tzinfo=UTC)
        )


def test_record_pace_mismatch_challenger_refuses_outside_recording_lock_window(
    tmp_path: Path,
) -> None:
    artifacts = tmp_path / "artifacts"
    _write_registry(artifacts)
    _write_active_model_and_card(artifacts)
    data_root = tmp_path / "data"

    with pytest.raises(ValueError, match="RECORDING_LOCK_WINDOW"):
        record_pace_mismatch_dog_tilt_challenger_decisions(
            artifacts, data_root, now=datetime(2026, 8, 18, 1, 0, tzinfo=UTC)
        )
    assert load_challenger_decisions(artifacts).empty


def test_pace_mismatch_fingerprint_helper_agrees_with_the_registered_model_block() -> None:
    """Sanity check that the fixture's config really matches
    CONFIG_FINGERPRINT_KEYS -- and that it equals the FROZEN fingerprint this
    challenger is registered against, ``bc77638d47e2748c`` (measured this
    session against both the ``model`` block and the live active-model
    artifact -- see the module report)."""

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
    assert config_fingerprint(_MODEL_CONFIG) == "bc77638d47e2748c"
