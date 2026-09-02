"""Tank-zone fade tilt overlay (docs/tank_zone_fade_tilt_overlay.md).

Five things are load-bearing here, mirroring
``tests/test_interim_hc_first_game_tilt_overlay.py``'s structure and AGENTS.md's
"add a leakage regression test for every new feature family" mandate:

1. :func:`tank_zone_flag_by_game` reproduces the registered cell's flag --
   bottom TWO league-wide records, ordered wins ascending / losses descending /
   team ascending -- and is DATA-DERIVED, never a hardcoded team list.
2. It is PREGAME-SAFE: the standings entering week *W* use only completed games
   from strictly prior weeks of the same season, so neither the flagged game's
   own ``result`` nor any later week's results can move its flag. Two explicit
   leakage regression tests.
3. :func:`apply_tank_zone_fade_tilt_overlay` fades the tank-zone side ONLY in
   weeks 14-18, ONLY when exactly one side carries the flag, and ONLY when the
   model's own pick is that side. Weeks 1-13 never flip. Both-flagged games
   never flip.
4. :func:`overlay_disclosure_note` states the flip count and matchups and never
   claims the published card.
5. :func:`record_tank_zone_fade_tilt_challenger_decisions` writes the overlay's
   own picks to the prospective challenger ledger, refuses a retuned/foreign
   model configuration (fingerprint stability), and refuses a non-
   ``ACTIVE_PROSPECTIVE`` registration.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import pytest
from _overlay_test_kit import write_active_model_and_card, write_challenger_registry

from nfl_ats.data import DataContractError
from nfl_ats.prospective_scoring import (
    CHALLENGER_DECISION_COLUMNS,
    artifact_model_config,
    config_fingerprint,
    load_challenger_decisions,
)
from nfl_ats.snapshots import write_snapshot
from nfl_ats.tank_zone_fade_tilt_overlay import (
    CHALLENGER_ID,
    OVERLAY_WEEK_MAX,
    OVERLAY_WEEK_MIN,
    apply_tank_zone_fade_tilt_overlay,
    overlay_disclosure_note,
    record_tank_zone_fade_tilt_challenger_decisions,
    tank_zone_flag_by_game,
)

# ---------------------------------------------------------------------------
# Shared fixture: a synthetic six-team league season
# ---------------------------------------------------------------------------
#
# Weeks 1-13: CCC beats AAA and DDD beats BBB, every week. Entering week 14 the
#   ordering key (wins asc, losses desc, team asc) puts AAA (0-13) and BBB
#   (0-13) below EEE/FFF (0-0), so the tank zone is exactly {AAA, BBB}.
# Week 13 (AAA at CCC): AAA IS flagged, but week 13 is outside the registered
#   14-18 window -- the "weeks 1-13 never flip" guard.
# Week 14: AAA hosts EEE (AAA flagged, home) and BBB visits FFF (BBB flagged,
#   away). Both week-14 games are lost by the flagged team, so entering week 15
#   the tank zone is still {AAA, BBB}.
# Week 15: AAA hosts BBB -- BOTH sides flagged, no measured direction, never
#   flipped. AAA wins it, so entering week 16 the zone is still {BBB, AAA}.
# Week 16: AAA hosts CCC (AAA flagged) with the model picking the OPPONENT --
#   the "flip only when the pick is on the tank side" guard -- plus EEE vs FFF
#   with neither side flagged.

_SCHEDULE_COLUMNS = [
    "game_id",
    "season",
    "week",
    "game_type",
    "gameday",
    "home_team",
    "away_team",
    "result",
    "spread_line",
]

_GAME_ONE_FLAG_HOME = "2026_14_AAA_EEE"
_GAME_ONE_FLAG_AWAY = "2026_14_FFF_BBB"
_GAME_BOTH_FLAGGED = "2026_15_AAA_BBB"
_GAME_PICK_OPPONENT = "2026_16_AAA_CCC"
_GAME_NO_FLAG = "2026_16_EEE_FFF"
_GAME_WEEK_13 = "2026_13_CCC_AAA"


def _schedule() -> pd.DataFrame:
    rows: list[tuple[object, ...]] = []
    for week in range(1, 13):
        day = f"2026-{9 + (week // 5):02d}-{1 + (week % 28):02d}"
        rows.append((f"2026_{week:02d}_CCC_AAA", 2026, week, "REG", day, "CCC", "AAA", 7.0, -3.0))
        rows.append((f"2026_{week:02d}_DDD_BBB", 2026, week, "REG", day, "DDD", "BBB", 7.0, -3.0))
    # Week 13: the same two beatdowns; the AAA game is the weeks-1-13 guard.
    rows.append((_GAME_WEEK_13, 2026, 13, "REG", "2026-12-03", "CCC", "AAA", 7.0, -3.0))
    rows.append(("2026_13_DDD_BBB", 2026, 13, "REG", "2026-12-03", "DDD", "BBB", 7.0, -3.0))
    # Week 14: exactly one flagged side in each game; both flagged teams lose.
    rows.append((_GAME_ONE_FLAG_HOME, 2026, 14, "REG", "2026-12-10", "AAA", "EEE", -7.0, 3.0))
    rows.append((_GAME_ONE_FLAG_AWAY, 2026, 14, "REG", "2026-12-10", "FFF", "BBB", 7.0, -3.0))
    # Week 15: both sides flagged. AAA wins, keeping both in the bottom two.
    rows.append((_GAME_BOTH_FLAGGED, 2026, 15, "REG", "2026-12-17", "AAA", "BBB", 7.0, -3.0))
    # Week 16: one flagged side, but the model picks the opponent; plus a
    # game with neither side flagged.
    rows.append((_GAME_PICK_OPPONENT, 2026, 16, "REG", "2026-12-24", "AAA", "CCC", -7.0, 3.0))
    rows.append((_GAME_NO_FLAG, 2026, 16, "REG", "2026-12-24", "EEE", "FFF", 7.0, -3.0))
    return pd.DataFrame(rows, columns=_SCHEDULE_COLUMNS)


def _flags() -> pd.DataFrame:
    return tank_zone_flag_by_game(_schedule()).set_index("game_id")


def _predictions() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "game_id": [
                _GAME_WEEK_13,
                _GAME_ONE_FLAG_HOME,
                _GAME_ONE_FLAG_AWAY,
                _GAME_BOTH_FLAGGED,
                _GAME_PICK_OPPONENT,
                _GAME_NO_FLAG,
                "2026_14_MISSING_GAME",
            ],
            "season": [2026] * 7,
            "week": [13, 14, 14, 15, 16, 16, 14],
            "game_type": ["REG"] * 7,
            "home_team": ["CCC", "AAA", "FFF", "AAA", "AAA", "EEE", "MISS_H"],
            "away_team": ["AAA", "EEE", "BBB", "BBB", "CCC", "FFF", "MISS_A"],
            "kickoff": ["2026-12-10T18:00:00+00:00"] * 7,
            "spread_line": [-3.0, 3.0, -3.0, 7.0, 3.0, -3.0, 1.0],
            # W13:  model picks AWAY (AAA, flagged) -- week 13 is outside the
            #       registered window, so NO flip.
            # OneH: model picks HOME (AAA, flagged)      -> flip to 0.38.
            # OneA: model picks AWAY (BBB, flagged)      -> flip to 0.60.
            # Both: both sides flagged                   -> never flipped.
            # PickOpp: model picks AWAY (CCC, not flagged) -> no flip.
            # NoFlag / Missing: no signal                 -> no flip.
            "home_cover_probability": [0.30, 0.62, 0.40, 0.70, 0.30, 0.55, 0.50],
        }
    )


# ---------------------------------------------------------------------------
# 1. tank_zone_flag_by_game: the derived flag
# ---------------------------------------------------------------------------


def test_flag_identifies_the_two_worst_records_entering_the_week() -> None:
    flags = _flags()
    assert bool(flags.loc[_GAME_ONE_FLAG_HOME, "tank_zone_home"]) is True
    assert bool(flags.loc[_GAME_ONE_FLAG_HOME, "tank_zone_away"]) is False
    assert bool(flags.loc[_GAME_ONE_FLAG_AWAY, "tank_zone_away"]) is True
    assert bool(flags.loc[_GAME_ONE_FLAG_AWAY, "tank_zone_home"]) is False


def test_flag_fires_on_both_sides_of_a_tank_versus_tank_game() -> None:
    flags = _flags()
    assert bool(flags.loc[_GAME_BOTH_FLAGGED, "tank_zone_home"]) is True
    assert bool(flags.loc[_GAME_BOTH_FLAGGED, "tank_zone_away"]) is True


def test_flag_is_false_for_teams_outside_the_bottom_two() -> None:
    flags = _flags()
    assert bool(flags.loc[_GAME_NO_FLAG, "tank_zone_home"]) is False
    assert bool(flags.loc[_GAME_NO_FLAG, "tank_zone_away"]) is False


def test_flag_covers_every_reg_game_exactly_once() -> None:
    schedule = _schedule()
    flags = tank_zone_flag_by_game(schedule)
    assert len(flags) == len(schedule)
    assert not flags["game_id"].duplicated().any()


def test_flag_requires_its_schedule_columns() -> None:
    with pytest.raises(DataContractError, match="tank-zone tracking"):
        tank_zone_flag_by_game(pd.DataFrame({"game_id": ["G1"], "season": [2026]}))


# ---------------------------------------------------------------------------
# 2. Leakage regressions: pregame-only inputs (AGENTS.md mandate)
# ---------------------------------------------------------------------------


def test_flag_is_leak_safe_against_the_games_own_result() -> None:
    """A game's OWN ``result``/``spread_line`` must never move its own flag.

    Blanking every outcome from the flagged week onward -- exactly what a live
    Tuesday-lock snapshot looks like, where the current and all later weeks are
    unplayed -- must leave the week-14 flags byte-identical.
    """

    schedule = _schedule()
    baseline = tank_zone_flag_by_game(schedule).set_index("game_id")

    blanked = schedule.copy()
    future = blanked["week"].ge(14)
    blanked.loc[future, "result"] = pd.NA
    live = tank_zone_flag_by_game(blanked).set_index("game_id")

    week14 = [_GAME_ONE_FLAG_HOME, _GAME_ONE_FLAG_AWAY]
    pd.testing.assert_frame_equal(
        baseline.loc[week14, ["tank_zone_home", "tank_zone_away"]],
        live.loc[week14, ["tank_zone_home", "tank_zone_away"]],
    )

    flipped = schedule.copy()
    own_game = flipped["game_id"].eq(_GAME_ONE_FLAG_HOME)
    flipped.loc[own_game, "result"] = 42.0
    flipped.loc[own_game, "spread_line"] = -21.0
    mutated = tank_zone_flag_by_game(flipped).set_index("game_id")
    assert bool(mutated.loc[_GAME_ONE_FLAG_HOME, "tank_zone_home"]) is True
    assert bool(mutated.loc[_GAME_ONE_FLAG_HOME, "tank_zone_away"]) is False


def test_flag_is_leak_safe_against_later_weeks() -> None:
    """Results from weeks AFTER a game can never change that game's flag."""

    schedule = _schedule()
    baseline = tank_zone_flag_by_game(schedule).set_index("game_id")

    rewritten = schedule.copy()
    later = rewritten["week"].ge(15)
    # Invert every later result: the flagged teams now win out.
    rewritten.loc[later, "result"] = -rewritten.loc[later, "result"]
    after = tank_zone_flag_by_game(rewritten).set_index("game_id")

    early = [game for game in baseline.index if str(game).startswith(("2026_1", "2026_0"))]
    early_upto_14 = [
        game
        for game in early
        if int(schedule.set_index("game_id").loc[game, "week"]) <= 14  # type: ignore[call-overload]
    ]
    pd.testing.assert_frame_equal(
        baseline.loc[early_upto_14, ["tank_zone_home", "tank_zone_away"]],
        after.loc[early_upto_14, ["tank_zone_home", "tank_zone_away"]],
    )


def test_flag_never_reads_an_unplayed_current_week() -> None:
    """Dropping every week-14+ ROW entirely (a true Tuesday snapshot, where
    later games exist on the schedule but carry no result) still produces the
    same week-14 standings for the games that remain."""

    schedule = _schedule()
    tuesday = schedule.copy()
    tuesday.loc[tuesday["week"].ge(14), "result"] = pd.NA
    full = tank_zone_flag_by_game(schedule).set_index("game_id")
    live = tank_zone_flag_by_game(tuesday).set_index("game_id")
    assert bool(live.loc[_GAME_ONE_FLAG_HOME, "tank_zone_home"]) == bool(
        full.loc[_GAME_ONE_FLAG_HOME, "tank_zone_home"]
    )
    assert bool(live.loc[_GAME_ONE_FLAG_AWAY, "tank_zone_away"]) == bool(
        full.loc[_GAME_ONE_FLAG_AWAY, "tank_zone_away"]
    )


# ---------------------------------------------------------------------------
# 3. apply_tank_zone_fade_tilt_overlay: the pick-level transform
# ---------------------------------------------------------------------------


def test_overlay_fades_the_tank_zone_side_when_the_model_picks_it() -> None:
    result = apply_tank_zone_fade_tilt_overlay(_predictions(), _schedule())
    flipped = {flip.game_id for flip in result.flips}
    assert flipped == {_GAME_ONE_FLAG_HOME, _GAME_ONE_FLAG_AWAY}

    overlaid = result.overlaid_predictions.set_index("game_id")
    assert overlaid.loc[_GAME_ONE_FLAG_HOME, "home_cover_probability"] == pytest.approx(0.38)
    assert overlaid.loc[_GAME_ONE_FLAG_AWAY, "home_cover_probability"] == pytest.approx(0.60)

    home_flip = next(f for f in result.flips if f.game_id == _GAME_ONE_FLAG_HOME)
    assert home_flip.tank_zone_team == "AAA"
    assert home_flip.opponent_team == "EEE"
    away_flip = next(f for f in result.flips if f.game_id == _GAME_ONE_FLAG_AWAY)
    assert away_flip.tank_zone_team == "BBB"
    assert away_flip.opponent_team == "FFF"


def test_overlay_never_flips_before_week_14() -> None:
    """The registered cell's flag is weeks 14-18 only; weeks 1-13 carry no
    claim, so a flagged week-13 game the model picks on the tank side must be
    left exactly alone. This is also why a Week 1 card can never move."""

    result = apply_tank_zone_fade_tilt_overlay(_predictions(), _schedule())
    assert all(flip.game_id != _GAME_WEEK_13 for flip in result.flips)
    overlaid = result.overlaid_predictions.set_index("game_id")
    assert overlaid.loc[_GAME_WEEK_13, "home_cover_probability"] == pytest.approx(0.30)

    # And the flag itself really did fire on that game -- the guard is the
    # week window, not an accidentally empty flag.
    flags = _flags()
    assert bool(flags.loc[_GAME_WEEK_13, "tank_zone_away"]) is True


def test_overlay_never_flips_a_week_1_card() -> None:
    """Week 1 is structurally outside the window: zero flips, by construction."""

    week_one = _predictions().assign(week=1)
    result = apply_tank_zone_fade_tilt_overlay(week_one, _schedule())
    assert result.flip_count == 0


def test_overlay_leaves_a_both_flagged_game_untouched() -> None:
    result = apply_tank_zone_fade_tilt_overlay(_predictions(), _schedule())
    assert all(flip.game_id != _GAME_BOTH_FLAGGED for flip in result.flips)
    assert _GAME_BOTH_FLAGGED in result.both_tank_zone_games
    overlaid = result.overlaid_predictions.set_index("game_id")
    assert overlaid.loc[_GAME_BOTH_FLAGGED, "home_cover_probability"] == pytest.approx(0.70)


def test_overlay_does_not_flip_when_the_pick_is_already_off_the_tank_side() -> None:
    result = apply_tank_zone_fade_tilt_overlay(_predictions(), _schedule())
    assert all(flip.game_id != _GAME_PICK_OPPONENT for flip in result.flips)
    overlaid = result.overlaid_predictions.set_index("game_id")
    assert overlaid.loc[_GAME_PICK_OPPONENT, "home_cover_probability"] == pytest.approx(0.30)


def test_overlay_does_not_flip_a_game_with_no_flagged_side() -> None:
    result = apply_tank_zone_fade_tilt_overlay(_predictions(), _schedule())
    assert all(flip.game_id != _GAME_NO_FLAG for flip in result.flips)


def test_overlay_treats_a_missing_schedule_row_as_no_signal() -> None:
    result = apply_tank_zone_fade_tilt_overlay(_predictions(), _schedule())
    assert all(flip.game_id != "2026_14_MISSING_GAME" for flip in result.flips)
    overlaid = result.overlaid_predictions.set_index("game_id")
    assert overlaid.loc["2026_14_MISSING_GAME", "home_cover_probability"] == pytest.approx(0.50)


def test_overlay_leaves_a_flagged_game_untouched_when_marked_postseason() -> None:
    predictions = _predictions().loc[lambda frame: frame["game_id"].eq(_GAME_ONE_FLAG_HOME)]
    postseason = predictions.assign(game_type="POST")
    result = apply_tank_zone_fade_tilt_overlay(postseason, _schedule())
    assert result.flip_count == 0


def test_overlay_window_constants_match_the_registered_cell() -> None:
    assert (OVERLAY_WEEK_MIN, OVERLAY_WEEK_MAX) == (14, 18)


def test_overlay_disabled_is_a_no_op() -> None:
    predictions = _predictions()
    result = apply_tank_zone_fade_tilt_overlay(predictions, _schedule(), enabled=False)
    assert result.flip_count == 0
    pd.testing.assert_frame_equal(
        result.overlaid_predictions.reset_index(drop=True),
        predictions.assign(game_id=predictions["game_id"].astype(str)).reset_index(drop=True),
        check_exact=True,
    )


def test_overlay_changes_only_home_cover_probability_on_flipped_rows() -> None:
    predictions = _predictions()
    result = apply_tank_zone_fade_tilt_overlay(predictions, _schedule())
    overlaid = result.overlaid_predictions

    assert list(overlaid.columns) == list(predictions.columns)
    other_columns = [c for c in predictions.columns if c != "home_cover_probability"]
    pd.testing.assert_frame_equal(
        overlaid[other_columns].reset_index(drop=True),
        predictions[other_columns].reset_index(drop=True),
        check_exact=True,
    )
    untouched = ~predictions["game_id"].isin([_GAME_ONE_FLAG_HOME, _GAME_ONE_FLAG_AWAY])
    pd.testing.assert_series_equal(
        overlaid.loc[untouched, "home_cover_probability"].reset_index(drop=True),
        predictions.loc[untouched, "home_cover_probability"].reset_index(drop=True),
        check_exact=True,
    )


def test_overlay_requires_its_prediction_columns() -> None:
    with pytest.raises(DataContractError, match="overlay columns"):
        apply_tank_zone_fade_tilt_overlay(pd.DataFrame({"game_id": ["G1"]}), _schedule())


# ---------------------------------------------------------------------------
# 4. overlay_disclosure_note
# ---------------------------------------------------------------------------


def test_disclosure_note_is_empty_when_nothing_flipped() -> None:
    quiet = _predictions().loc[lambda frame: frame["game_id"].eq(_GAME_NO_FLAG)]
    assert overlay_disclosure_note(apply_tank_zone_fade_tilt_overlay(quiet, _schedule())) == ""
    disabled = apply_tank_zone_fade_tilt_overlay(_predictions(), _schedule(), enabled=False)
    assert overlay_disclosure_note(disabled) == ""


def test_disclosure_note_states_the_flip_count_and_does_not_claim_production() -> None:
    note = overlay_disclosure_note(apply_tank_zone_fade_tilt_overlay(_predictions(), _schedule()))
    assert "Tilt applied: 2 picks flipped" in note
    assert "AAA -> EEE" in note
    assert "BBB -> FFF" in note
    assert "weeks 14-18" in note
    assert "not applied to the published card" in note


# ---------------------------------------------------------------------------
# 5. record_tank_zone_fade_tilt_challenger_decisions: dual-tracked
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

_SEASON, _WEEK, _CREATED_AT_UTC = 2026, 14, "2026-12-08T15:00:00+00:00"
_NOW = datetime(2026, 12, 8, 16, 0, tzinfo=UTC)


def _write_data_root(tmp_path: Path) -> Path:
    repo_root = tmp_path / "repo"
    data_root = repo_root / "data"
    write_snapshot(
        _schedule(),
        pd.DataFrame({"game_id": [], "team": []}),
        seasons=[2026],
        raw_root=data_root / "raw",
    )
    return data_root


def _recorder_predictions() -> pd.DataFrame:
    return _predictions().loc[
        lambda frame: frame["game_id"].isin([_GAME_ONE_FLAG_HOME, _GAME_NO_FLAG])
    ]


def _write_registry(artifacts: Path, *, status: str = "ACTIVE_PROSPECTIVE") -> None:
    write_challenger_registry(
        artifacts, challenger_id=CHALLENGER_ID, model_config=_MODEL_CONFIG, status=status
    )


def _write_active_model_and_card(artifacts: Path, *, ridge_alpha: float = 10.0) -> None:
    write_active_model_and_card(
        artifacts,
        season=_SEASON,
        week=_WEEK,
        created_at_utc=_CREATED_AT_UTC,
        ridge_alpha=ridge_alpha,
        recommendations=_recorder_predictions(),
    )


def test_record_tank_zone_fade_challenger_decisions_records_the_tilt_arm(tmp_path: Path) -> None:
    artifacts = tmp_path / "artifacts"
    _write_registry(artifacts)
    _write_active_model_and_card(artifacts)
    data_root = _write_data_root(tmp_path)

    result = record_tank_zone_fade_tilt_challenger_decisions(artifacts, data_root, now=_NOW)

    assert result["recorded"] == 2
    assert result["flip_count"] == 1
    assert result["flipped_game_ids"] == [_GAME_ONE_FLAG_HOME]

    ledger = load_challenger_decisions(artifacts)
    assert list(ledger.columns) == list(CHALLENGER_DECISION_COLUMNS)
    assert (ledger["bet_side"] == "PASS").all()
    assert ledger["edge"].isna().all()

    indexed = ledger.set_index("game_id")
    # The model's raw pick was HOME (0.62 -> AAA, the tank-zone team); the
    # fade flips it to AWAY.
    assert indexed.loc[_GAME_ONE_FLAG_HOME, "pick_side"] == "AWAY"
    # The unflagged game keeps the model's own pick (0.55 -> HOME).
    assert indexed.loc[_GAME_NO_FLAG, "pick_side"] == "HOME"

    again = record_tank_zone_fade_tilt_challenger_decisions(artifacts, data_root, now=_NOW)
    assert again["recorded"] == 0
    assert again["already_recorded"] == 2


def test_record_tank_zone_fade_challenger_refuses_outside_recording_lock_window(
    tmp_path: Path,
) -> None:
    artifacts = tmp_path / "artifacts"
    _write_registry(artifacts)
    _write_active_model_and_card(artifacts)
    data_root = _write_data_root(tmp_path)

    with pytest.raises(ValueError, match="RECORDING_LOCK_WINDOW"):
        record_tank_zone_fade_tilt_challenger_decisions(
            artifacts, data_root, now=datetime(2026, 11, 1, 1, 0, tzinfo=UTC)
        )
    assert load_challenger_decisions(artifacts).empty


def test_record_tank_zone_fade_challenger_refuses_a_fingerprint_mismatch(tmp_path: Path) -> None:
    """Fingerprint stability: a retuned or foreign active model configuration
    must refuse to record, never silently switch base models under this id."""

    artifacts = tmp_path / "artifacts"
    _write_registry(artifacts)
    _write_active_model_and_card(artifacts, ridge_alpha=1.0)
    data_root = _write_data_root(tmp_path)

    with pytest.raises(DataContractError, match="configuration fingerprint"):
        record_tank_zone_fade_tilt_challenger_decisions(artifacts, data_root, now=_NOW)
    assert load_challenger_decisions(artifacts).empty


def test_record_tank_zone_fade_challenger_refuses_an_inactive_registration(tmp_path: Path) -> None:
    artifacts = tmp_path / "artifacts"
    _write_registry(artifacts, status="CLOSED_BEFORE_ACTIVATION")
    _write_active_model_and_card(artifacts)
    data_root = _write_data_root(tmp_path)

    with pytest.raises(ValueError, match="only ACTIVE_PROSPECTIVE"):
        record_tank_zone_fade_tilt_challenger_decisions(artifacts, data_root, now=_NOW)


def test_tank_zone_fade_fingerprint_helper_agrees_with_the_registered_model_block() -> None:
    """The fixture's config really does match CONFIG_FINGERPRINT_KEYS."""

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
