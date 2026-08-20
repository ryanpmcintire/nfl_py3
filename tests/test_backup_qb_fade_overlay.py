"""Backup-QB fade overlay (docs/backup_qb_fade_overlay.md).

Four things are load-bearing here, mirroring
``tests/test_division_revenge_tilt_overlay.py``'s structure and AGENTS.md's
"add a leakage regression test for every new feature family" spirit:

1. :func:`backup_qb_flag_by_game`'s flag is derived from data, not
   hand-typed, is pregame-safe (a later start must never retroactively
   change an earlier game's flag), and respects the battery's own >=3-prior-
   starts eligibility floor exactly.
2. :func:`apply_backup_qb_fade_overlay` flips ONLY the clean case (the
   model's pick sits on a flagged backup start against a non-backup
   opponent), respects the REG-only gate, and is parameter-free.
3. :func:`overlay_disclosure_note` states the flip count and matchups.
4. :func:`record_backup_qb_fade_challenger_decisions` writes the overlay's
   own picks to the prospective challenger ledger, dual-tracked and at no
   rotation-registry window cost.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import pytest

from nfl_ats.active_model import ACTIVE_ATS_MODEL_VERSION
from nfl_ats.backup_qb_fade_overlay import (
    CHALLENGER_ID,
    apply_backup_qb_fade_overlay,
    backup_qb_flag_by_game,
    overlay_disclosure_note,
    record_backup_qb_fade_challenger_decisions,
)
from nfl_ats.data import DataContractError
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
# BKUP starts "Franchise" weeks 1-3 (3 prior starts by week 4), then
#   "Journeyman" in week 4 -- eligible and flagged.
# STARTQB starts "Ace" every week -- never flagged.
# EARLYBKUP starts "Vet" week 1, "Rookie" week 2 -- only 1 prior start, so
#   week 2 is NOT eligible even though the QB changed (the eligibility
#   floor).
# BKUP2 mirrors BKUP's shape (flagged by week 4 on "Substitute" after 3
#   "Captain" starts), meeting BKUP in week 5 while BOTH are still flagged
#   -- the "no clean direction" case.
# OPP1-14 exist only to seed a schedule row for each tracked team; their own
#   QB history is irrelevant to the assertions.


def _qb_schedule() -> pd.DataFrame:
    rows = [
        # game_id, season, game_type, week, gameday, home_team, away_team, home_qb, away_qb
        ("2026_01_BKUP_OPP1", 2026, "REG", 1, "2026-09-10", "BKUP", "OPP1", "Franchise", "OppQB1"),
        ("2026_02_OPP2_BKUP", 2026, "REG", 2, "2026-09-17", "OPP2", "BKUP", "OppQB2", "Franchise"),
        ("2026_03_BKUP_OPP3", 2026, "REG", 3, "2026-09-24", "BKUP", "OPP3", "Franchise", "OppQB3"),
        (
            "2026_04_STARTQB_BKUP",
            2026,
            "REG",
            4,
            "2026-10-01",
            "STARTQB",
            "BKUP",
            "Ace",
            "Journeyman",
        ),
        (
            "2026_01_STARTQB_OPP4",
            2026,
            "REG",
            1,
            "2026-09-10",
            "STARTQB",
            "OPP4",
            "Ace",
            "OppQB4",
        ),
        (
            "2026_02_OPP5_STARTQB",
            2026,
            "REG",
            2,
            "2026-09-17",
            "OPP5",
            "STARTQB",
            "OppQB5",
            "Ace",
        ),
        (
            "2026_03_STARTQB_OPP6",
            2026,
            "REG",
            3,
            "2026-09-24",
            "STARTQB",
            "OPP6",
            "Ace",
            "OppQB6",
        ),
        (
            "2026_01_EARLYBKUP_OPP7",
            2026,
            "REG",
            1,
            "2026-09-10",
            "EARLYBKUP",
            "OPP7",
            "Vet",
            "OppQB7",
        ),
        (
            "2026_02_OPP8_EARLYBKUP",
            2026,
            "REG",
            2,
            "2026-09-17",
            "OPP8",
            "EARLYBKUP",
            "OppQB8",
            "Rookie",
        ),
        ("2026_01_BKUP2_OPP9", 2026, "REG", 1, "2026-09-10", "BKUP2", "OPP9", "Captain", "OppQB9"),
        (
            "2026_02_OPP10_BKUP2",
            2026,
            "REG",
            2,
            "2026-09-17",
            "OPP10",
            "BKUP2",
            "OppQB10",
            "Captain",
        ),
        (
            "2026_03_BKUP2_OPP11",
            2026,
            "REG",
            3,
            "2026-09-24",
            "BKUP2",
            "OPP11",
            "Captain",
            "OppQB11",
        ),
        (
            "2026_04_OPP12_BKUP2",
            2026,
            "REG",
            4,
            "2026-10-01",
            "OPP12",
            "BKUP2",
            "OppQB12",
            "Substitute",
        ),
        (
            "2026_05_BKUP2_BKUP",
            2026,
            "REG",
            5,
            "2026-10-08",
            "BKUP2",
            "BKUP",
            "Substitute",
            "Journeyman",
        ),
        (
            "2026_20_STARTQB_BKUP_POST",
            2026,
            "POST",
            20,
            "2027-01-10",
            "STARTQB",
            "BKUP",
            "Ace",
            "Journeyman",
        ),
        (
            "2026_05_OPP13_OPP14",
            2026,
            "REG",
            5,
            "2026-10-08",
            "OPP13",
            "OPP14",
            "OppQB13",
            "OppQB14",
        ),
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
            "home_qb_name",
            "away_qb_name",
        ],
    )


def _predictions() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "game_id": [
                "2026_04_STARTQB_BKUP",
                "2026_02_OPP8_EARLYBKUP",
                "2026_05_BKUP2_BKUP",
                "2026_20_STARTQB_BKUP_POST",
                "2026_05_OPP13_OPP14",
                "2026_MISSING_GAME",
            ],
            "season": [2026, 2026, 2026, 2026, 2026, 2026],
            "week": [4, 2, 5, 20, 5, 5],
            "game_type": ["REG", "REG", "REG", "POST", "REG", "REG"],
            "home_team": ["STARTQB", "OPP8", "BKUP2", "STARTQB", "OPP13", "MISS_H"],
            "away_team": ["BKUP", "EARLYBKUP", "BKUP", "BKUP", "OPP14", "MISS_A"],
            "kickoff": ["2026-10-01T17:00:00+00:00"] * 6,
            "spread_line": [-3.0, 2.0, -1.5, -3.0, -2.5, 1.0],
            # G-clean: model picks AWAY (BKUP, flagged) against a non-flagged
            # home -- against the clean case -- should flip to HOME.
            # G-early: model picks AWAY (EARLYBKUP, not yet eligible) -- no
            # signal, no flip.
            # G-both: model picks HOME (BKUP2, flagged) against BKUP (also
            # flagged) -- no clean direction, no flip.
            # G-post: same shape as G-clean but POST season -- REG-only gate.
            # G-plain: neither side has any QB history -- no signal.
            # G-missing: no schedule row at all -- treated as no signal.
            "home_cover_probability": [0.30, 0.40, 0.60, 0.30, 0.55, 0.50],
        }
    )


# ---------------------------------------------------------------------------
# 1. backup_qb_flag_by_game: derived, pregame-safe
# ---------------------------------------------------------------------------


def test_backup_flag_fires_after_three_prior_starts_with_a_different_qb() -> None:
    flags = backup_qb_flag_by_game(_qb_schedule()).set_index("game_id")
    assert bool(flags.loc["2026_04_STARTQB_BKUP", "backup_away"]) is True  # BKUP: Journeyman
    assert bool(flags.loc["2026_04_STARTQB_BKUP", "backup_home"]) is False  # STARTQB: Ace==Ace


def test_backup_flag_requires_the_eligibility_floor() -> None:
    """Only 1 prior start (week 1) by week 2 -- below the 3-prior-starts
    floor -- so EARLYBKUP's QB change is NOT flagged yet."""

    flags = backup_qb_flag_by_game(_qb_schedule()).set_index("game_id")
    assert bool(flags.loc["2026_02_OPP8_EARLYBKUP", "backup_away"]) is False


def test_backup_flag_is_false_before_any_start_this_season() -> None:
    flags = backup_qb_flag_by_game(_qb_schedule()).set_index("game_id")
    assert bool(flags.loc["2026_01_BKUP_OPP1", "backup_home"]) is False


def test_backup_flag_fires_on_both_sides_when_both_teams_are_backup_starts() -> None:
    flags = backup_qb_flag_by_game(_qb_schedule()).set_index("game_id")
    assert bool(flags.loc["2026_05_BKUP2_BKUP", "backup_home"]) is True
    assert bool(flags.loc["2026_05_BKUP2_BKUP", "backup_away"]) is True


def test_backup_flag_requires_its_schedule_columns() -> None:
    with pytest.raises(DataContractError, match="backup-QB"):
        backup_qb_flag_by_game(pd.DataFrame({"game_id": ["G1"]}))


def test_backup_flag_is_leak_safe_against_a_later_start_mutation() -> None:
    """AGENTS.md: a leakage regression test for every new feature family.

    BKUP's week 4 flag depends only on weeks 1-3; mutating week 5 (a LATER
    start) must not move it.
    """

    baseline = backup_qb_flag_by_game(_qb_schedule()).set_index("game_id")

    mutated = _qb_schedule()
    mutated.loc[mutated["game_id"].eq("2026_05_BKUP2_BKUP"), "away_qb_name"] = "SomeoneElse"
    changed = backup_qb_flag_by_game(mutated).set_index("game_id")

    assert bool(changed.loc["2026_04_STARTQB_BKUP", "backup_away"]) == bool(
        baseline.loc["2026_04_STARTQB_BKUP", "backup_away"]
    )


def test_backup_flag_is_leak_safe_across_the_season_boundary() -> None:
    """A future season's QB history (even for the same team) must never
    change an earlier season's already-computed flag."""

    schedule = _qb_schedule()
    baseline = backup_qb_flag_by_game(schedule)

    future = pd.DataFrame(
        [
            (
                "2027_01_BKUP_OPP1",
                2027,
                "REG",
                1,
                "2027-09-09",
                "BKUP",
                "OPP1",
                "NewGuy",
                "OppQB1",
            ),
        ],
        columns=schedule.columns,
    )
    changed = backup_qb_flag_by_game(pd.concat([schedule, future], ignore_index=True))

    pd.testing.assert_frame_equal(
        changed.loc[changed["season"].le(2026)].reset_index(drop=True),
        baseline.loc[baseline["season"].le(2026)].reset_index(drop=True),
        check_exact=True,
    )


# ---------------------------------------------------------------------------
# 2. apply_backup_qb_fade_overlay: the pick-level transform
# ---------------------------------------------------------------------------


def test_overlay_flips_away_from_the_clean_case_backup_start() -> None:
    result = apply_backup_qb_fade_overlay(_predictions(), _qb_schedule())

    flipped_ids = {flip.game_id for flip in result.flips}
    assert "2026_04_STARTQB_BKUP" in flipped_ids
    flip = next(f for f in result.flips if f.game_id == "2026_04_STARTQB_BKUP")
    assert flip.backup_team == "BKUP"
    assert flip.opponent_team == "STARTQB"

    overlaid = result.overlaid_predictions.set_index("game_id")
    assert overlaid.loc["2026_04_STARTQB_BKUP", "home_cover_probability"] == pytest.approx(0.70)


def test_overlay_does_not_flip_before_the_eligibility_floor() -> None:
    result = apply_backup_qb_fade_overlay(_predictions(), _qb_schedule())
    assert all(flip.game_id != "2026_02_OPP8_EARLYBKUP" for flip in result.flips)
    overlaid = result.overlaid_predictions.set_index("game_id")
    assert overlaid.loc["2026_02_OPP8_EARLYBKUP", "home_cover_probability"] == pytest.approx(0.40)


def test_overlay_does_not_touch_a_both_backup_game() -> None:
    result = apply_backup_qb_fade_overlay(_predictions(), _qb_schedule())
    assert "2026_05_BKUP2_BKUP" in result.both_backup_games
    assert all(flip.game_id != "2026_05_BKUP2_BKUP" for flip in result.flips)
    overlaid = result.overlaid_predictions.set_index("game_id")
    assert overlaid.loc["2026_05_BKUP2_BKUP", "home_cover_probability"] == pytest.approx(0.60)


def test_overlay_leaves_postseason_games_untouched() -> None:
    """Same shape as the flipped clean case, but POST season -- the REG-only
    gate blocks it."""

    result = apply_backup_qb_fade_overlay(_predictions(), _qb_schedule())
    assert all(flip.game_id != "2026_20_STARTQB_BKUP_POST" for flip in result.flips)
    overlaid = result.overlaid_predictions.set_index("game_id")
    assert overlaid.loc["2026_20_STARTQB_BKUP_POST", "home_cover_probability"] == pytest.approx(
        0.30
    )


def test_overlay_does_not_flip_a_game_with_no_backup_signal() -> None:
    result = apply_backup_qb_fade_overlay(_predictions(), _qb_schedule())
    assert all(flip.game_id != "2026_05_OPP13_OPP14" for flip in result.flips)
    overlaid = result.overlaid_predictions.set_index("game_id")
    assert overlaid.loc["2026_05_OPP13_OPP14", "home_cover_probability"] == pytest.approx(0.55)


def test_overlay_treats_a_missing_schedule_row_as_no_signal() -> None:
    result = apply_backup_qb_fade_overlay(_predictions(), _qb_schedule())
    assert all(flip.game_id != "2026_MISSING_GAME" for flip in result.flips)
    overlaid = result.overlaid_predictions.set_index("game_id")
    assert overlaid.loc["2026_MISSING_GAME", "home_cover_probability"] == pytest.approx(0.50)


def test_overlay_disabled_is_a_no_op() -> None:
    predictions = _predictions()
    result = apply_backup_qb_fade_overlay(predictions, _qb_schedule(), enabled=False)

    assert result.flip_count == 0
    assert result.both_backup_games == ()
    pd.testing.assert_frame_equal(
        result.overlaid_predictions.reset_index(drop=True),
        predictions.reset_index(drop=True),
        check_exact=True,
    )


def test_overlay_changes_only_home_cover_probability_on_flipped_rows() -> None:
    """Additivity: every other column, and every untouched row, stays
    byte-identical -- the pick-level design's whole point."""

    predictions = _predictions()
    result = apply_backup_qb_fade_overlay(predictions, _qb_schedule())
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
            "2026_02_OPP8_EARLYBKUP",
            "2026_05_BKUP2_BKUP",
            "2026_20_STARTQB_BKUP_POST",
            "2026_05_OPP13_OPP14",
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
        apply_backup_qb_fade_overlay(pd.DataFrame({"game_id": ["G1"]}), _qb_schedule())


# ---------------------------------------------------------------------------
# 3. overlay_disclosure_note: the plain-English provenance sentence
# ---------------------------------------------------------------------------


def test_disclosure_note_is_empty_when_nothing_flipped() -> None:
    plain_only = _predictions().loc[lambda frame: frame["game_id"].eq("2026_05_OPP13_OPP14")]
    result = apply_backup_qb_fade_overlay(plain_only, _qb_schedule())
    assert overlay_disclosure_note(result) == ""

    disabled = apply_backup_qb_fade_overlay(_predictions(), _qb_schedule(), enabled=False)
    assert overlay_disclosure_note(disabled) == ""


def test_disclosure_note_states_the_flip_count_and_does_not_claim_production() -> None:
    result = apply_backup_qb_fade_overlay(_predictions(), _qb_schedule())
    note = overlay_disclosure_note(result)

    assert "Tilt applied: 1 pick flipped" in note
    assert "BKUP -> STARTQB" in note
    assert "not applied to the published card" in note


# ---------------------------------------------------------------------------
# 4. record_backup_qb_fade_challenger_decisions: dual-tracked, no window
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
            "game_id": ["2026_04_STARTQB_BKUP", "2026_05_OPP13_OPP14"],
            "season": [2026, 2026],
            "week": [4, 5],
            "game_type": ["REG", "REG"],
            "home_team": ["STARTQB", "OPP13"],
            "away_team": ["BKUP", "OPP14"],
            "kickoff": ["2026-10-01T17:00:00+00:00", "2026-10-08T17:00:00+00:00"],
            "spread_line": [-3.0, -2.5],
            "home_cover_probability": [0.30, 0.55],
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
    forecast = artifacts / "margin_predictions" / "2026-week-04-forecast"
    forecast.mkdir(parents=True, exist_ok=True)
    metadata = {
        "active_model_id": "model-xyz",
        "synchronization_status": "SYNCHRONIZED",
        "season": 2026,
        "week": 4,
        "created_at_utc": "2026-09-26T15:00:00+00:00",
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
            "artifact": "margin_predictions/2026-week-04-forecast",
            "season": 2026,
            "week": 4,
        },
    }
    (artifacts / "active_ats_model.json").write_text(json.dumps(active), encoding="utf-8")


def _write_data_root(tmp_path: Path) -> Path:
    data_root = tmp_path / "data"
    write_snapshot(
        _qb_schedule(),
        pd.DataFrame({"game_id": [], "team": []}),
        seasons=[2026],
        raw_root=data_root / "raw",
    )
    return data_root


def test_record_fade_challenger_decisions_records_the_fade_arm(tmp_path: Path) -> None:
    artifacts = tmp_path / "artifacts"
    _write_registry(artifacts)
    _write_active_model_and_card(artifacts)
    data_root = _write_data_root(tmp_path)
    now = datetime(2026, 9, 26, 16, 0, tzinfo=UTC)

    result = record_backup_qb_fade_challenger_decisions(artifacts, data_root, now=now)

    assert result["recorded"] == 2
    assert result["flip_count"] == 1
    assert result["flipped_game_ids"] == ["2026_04_STARTQB_BKUP"]

    ledger = load_challenger_decisions(artifacts).set_index("game_id")
    assert list(load_challenger_decisions(artifacts).columns) == list(CHALLENGER_DECISION_COLUMNS)
    assert (ledger["bet_side"] == "PASS").all()
    assert ledger["edge"].isna().all()

    # The fade's own arm diverges from the active model's raw pick (0.30 ->
    # AWAY): the fade flips it to HOME (STARTQB), away from the backup start.
    assert ledger.loc["2026_04_STARTQB_BKUP", "pick_side"] == "HOME"
    # The no-signal game keeps the model's own HOME pick.
    assert ledger.loc["2026_05_OPP13_OPP14", "pick_side"] == "HOME"

    # Re-running is a no-op: append-only, never rewrites.
    again = record_backup_qb_fade_challenger_decisions(artifacts, data_root, now=now)
    assert again["recorded"] == 0
    assert again["already_recorded"] == 2


def test_record_fade_challenger_refuses_outside_recording_lock_window(tmp_path: Path) -> None:
    artifacts = tmp_path / "artifacts"
    _write_registry(artifacts)
    _write_active_model_and_card(artifacts)
    data_root = _write_data_root(tmp_path)

    with pytest.raises(ValueError, match="RECORDING_LOCK_WINDOW"):
        record_backup_qb_fade_challenger_decisions(
            artifacts, data_root, now=datetime(2026, 8, 18, 1, 0, tzinfo=UTC)
        )
    assert load_challenger_decisions(artifacts).empty


def test_record_fade_challenger_refuses_a_fingerprint_mismatch(tmp_path: Path) -> None:
    artifacts = tmp_path / "artifacts"
    _write_registry(artifacts)
    # The active model's OWN configuration moved (a promotion) since this
    # challenger was pinned -- recording must refuse, not silently switch
    # base models under the same challenger id.
    _write_active_model_and_card(artifacts, ridge_alpha=1.0)
    data_root = _write_data_root(tmp_path)

    with pytest.raises(DataContractError, match="configuration fingerprint"):
        record_backup_qb_fade_challenger_decisions(
            artifacts, data_root, now=datetime(2026, 9, 26, 16, 0, tzinfo=UTC)
        )
    assert load_challenger_decisions(artifacts).empty


def test_record_fade_challenger_refuses_an_inactive_registration(tmp_path: Path) -> None:
    artifacts = tmp_path / "artifacts"
    _write_registry(artifacts, status="CLOSED_BEFORE_ACTIVATION")
    _write_active_model_and_card(artifacts)
    data_root = _write_data_root(tmp_path)

    with pytest.raises(ValueError, match="only ACTIVE_PROSPECTIVE"):
        record_backup_qb_fade_challenger_decisions(
            artifacts, data_root, now=datetime(2026, 9, 26, 16, 0, tzinfo=UTC)
        )


def test_record_fade_challenger_refuses_a_deactivated_registration(tmp_path: Path) -> None:
    """Pins the 2026-08-19 deactivation (docs/prospective_evidence.md
    'Tuesday-visibility audit'): backup_qb_fade_overlay is now registered
    DEACTIVATED_STRUCTURAL_NO_OP in artifacts/prospective/challengers.json
    (an OPERATIONAL defect -- home_qb_name/away_qb_name are only populated
    post-game -- never an evidential closure of the underlying bias-battery
    cell). The recorder must refuse cleanly, exactly like any other
    non-ACTIVE_PROSPECTIVE status, and the ledger must stay empty: a
    deactivated challenger records nothing."""

    artifacts = tmp_path / "artifacts"
    _write_registry(artifacts, status="DEACTIVATED_STRUCTURAL_NO_OP")
    _write_active_model_and_card(artifacts)
    data_root = _write_data_root(tmp_path)

    with pytest.raises(ValueError, match="only ACTIVE_PROSPECTIVE"):
        record_backup_qb_fade_challenger_decisions(
            artifacts, data_root, now=datetime(2026, 9, 26, 16, 0, tzinfo=UTC)
        )
    assert load_challenger_decisions(artifacts).empty


def test_fade_fingerprint_helper_agrees_with_the_registered_model_block() -> None:
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
