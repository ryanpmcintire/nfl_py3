"""Bye-edge fade overlay (docs/bye_edge_fade_overlay.md).

Four things are load-bearing here, mirroring
``tests/test_surface_switch_tilt_overlay.py``'s structure and AGENTS.md's
"add a leakage regression test for every new feature family" spirit:

1. :func:`bye_edge_flag_by_game`'s flags are derived from data, not
   hand-typed, read only structural schedule columns (never an outcome
   column -- neither ``result`` nor ``spread_line`` is even in the required
   set), and respect the frozen strict-bye threshold (``POST_BYE_GAP_DAYS =
   12``) ported verbatim from ``scripts/bye_overvaluation_screen.py``.
2. :func:`apply_bye_edge_fade_overlay` flips ONLY a pick that sits on the
   strict-bye-holding side of a game where EXACTLY ONE team is off a strict
   bye (both-off-bye and neither-off-bye games are never touched), respects
   the REG-only gate, and is parameter-free.
3. :func:`overlay_disclosure_note` states the flip count and matchups.
4. :func:`record_bye_edge_fade_challenger_decisions` writes the overlay's
   own picks to the prospective challenger ledger, dual-tracked and at no
   rotation-registry window cost.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import pytest
from _overlay_test_kit import write_active_model_and_card, write_challenger_registry

from nfl_ats.bye_edge_fade_overlay import (
    CHALLENGER_ID,
    apply_bye_edge_fade_overlay,
    bye_edge_flag_by_game,
    overlay_disclosure_note,
    record_bye_edge_fade_challenger_decisions,
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
# BYETEAM's 2026 REG-season home/away sequence:
#   week 1  (2026-09-10): home vs OPP1           -- season opener, no prior
#                                                    game -> gap undefined,
#                                                    never a bye.
#   week 3  (2026-09-24, 14 days after week 1):
#           at TWELVEHOST, home                  -- TWELVEHOST's own gap to
#                                                    its week-1 game (also
#                                                    2026-09-10) is 14 days
#                                                    >= 12 -> TWELVEHOST is
#                                                    the strict-bye team;
#                                                    BYETEAM's own gap is
#                                                    also 14 days (it too
#                                                    skipped week 2), so this
#                                                    game is a BOTH-OFF-BYE
#                                                    case (never flipped).
#   week 4 (2026-10-01, 7 days after week 3):
#           at ELEVENHOST                        -- ELEVENHOST's gap to ITS
#                                                    OWN prior game is
#                                                    EXACTLY 11 days (below
#                                                    the strict threshold) --
#                                                    used for the 11-vs-12-day
#                                                    boundary test.
#   week 6 (2026-10-15, 14 days after week 4):
#           at NEITHERHOST                       -- BYETEAM itself is off a
#                                                    strict 14-day bye here,
#                                                    but NEITHERHOST is NOT
#                                                    (NEITHERHOST played every
#                                                    week) -- the clean
#                                                    single-sided flagged case.
#
# TWELVEHOST's own sequence: week 1 vs OPP2 (2026-09-10), week 3 vs BYETEAM
#   (2026-09-24, exactly 14 days later -> strict bye).
# ELEVENHOST's own sequence: week 1 vs OPP3 (2026-09-10), week 2 vs OPP4
#   (2026-09-17), week 4 vs BYETEAM (2026-09-28 -- 11 days after week 2,
#   BELOW the strict threshold).
# NEITHERHOST plays every week (1, 4, 5, 6) with no gap >= 12 days, so it is
#   never flagged.


def _bye_schedule() -> pd.DataFrame:
    rows = [
        # game_id, season, week, game_type, gameday, home_team, away_team
        ("2026_01_BYETEAM_OPP1", 2026, 1, "REG", "2026-09-10", "BYETEAM", "OPP1"),
        ("2026_01_TWELVEHOST_OPP2", 2026, 1, "REG", "2026-09-10", "TWELVEHOST", "OPP2"),
        ("2026_01_ELEVENHOST_OPP3", 2026, 1, "REG", "2026-09-10", "ELEVENHOST", "OPP3"),
        ("2026_01_NEITHERHOST_OPP5", 2026, 1, "REG", "2026-09-10", "NEITHERHOST", "OPP5"),
        ("2026_02_ELEVENHOST_OPP4", 2026, 2, "REG", "2026-09-17", "ELEVENHOST", "OPP4"),
        # week 3: BYETEAM at TWELVEHOST, both off a 14-day strict bye.
        ("2026_03_TWELVEHOST_BYETEAM", 2026, 3, "REG", "2026-09-24", "TWELVEHOST", "BYETEAM"),
        # week 4: BYETEAM at ELEVENHOST -- ELEVENHOST's own gap is 11 days
        # (2026-09-17 -> 2026-09-28), below the strict threshold. BYETEAM's
        # own gap here (2026-09-24 -> 2026-09-28) is only 4 days.
        ("2026_04_ELEVENHOST_BYETEAM", 2026, 4, "REG", "2026-09-28", "ELEVENHOST", "BYETEAM"),
        ("2026_04_NEITHERHOST_OPP6", 2026, 4, "REG", "2026-09-28", "NEITHERHOST", "OPP6"),
        ("2026_05_NEITHERHOST_OPP7", 2026, 5, "REG", "2026-10-05", "NEITHERHOST", "OPP7"),
        # week 6: BYETEAM (14-day gap from week 4) at NEITHERHOST (played
        # every week, no bye) -- the clean single-sided flagged case.
        ("2026_06_NEITHERHOST_BYETEAM", 2026, 6, "REG", "2026-10-15", "NEITHERHOST", "BYETEAM"),
        # A POST-season game with the same flagged shape, for the REG-only gate.
        ("2026_20_POSTHOST_BYETEAM", 2026, 20, "POST", "2026-12-30", "POSTHOST", "BYETEAM"),
    ]
    return pd.DataFrame(
        rows,
        columns=["game_id", "season", "week", "game_type", "gameday", "home_team", "away_team"],
    )


def _predictions() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "game_id": [
                "2026_03_TWELVEHOST_BYETEAM",  # both off bye -- never flip
                "2026_04_ELEVENHOST_BYETEAM",  # 11-day gap -- not a bye -- no flip
                "2026_06_NEITHERHOST_BYETEAM",  # single-sided flag, pick on bye side -- flip
                "2026_20_POSTHOST_BYETEAM",  # same flagged shape, POST -- no flip
                "2026_MISSING_GAME",  # no schedule row -- no flip
            ],
            "season": [2026, 2026, 2026, 2026, 2026],
            "week": [3, 4, 6, 20, 6],
            "game_type": ["REG", "REG", "REG", "POST", "REG"],
            "home_team": ["TWELVEHOST", "ELEVENHOST", "NEITHERHOST", "POSTHOST", "MISS_H"],
            "away_team": ["BYETEAM", "BYETEAM", "BYETEAM", "BYETEAM", "MISS_A"],
            "kickoff": ["2026-10-15T17:00:00+00:00"] * 5,
            "spread_line": [-3.0, 2.0, -1.5, -2.0, 1.0],
            # G-both: both off bye -- flag never fires (XOR is False) -- no
            #   flip regardless of the model's pick (home pick here).
            # G-eleven: ELEVENHOST's own gap is 11 days, below the 12-day
            #   strict threshold -- not flagged -- no flip (away pick here).
            # G-flag: NEITHERHOST is not off bye, BYETEAM (away) IS off a
            #   strict 14-day bye, and the model's pick is on BYETEAM (away,
            #   home_cover_probability < 0.5) -- the bye-holding side --
            #   should flip to home (NEITHERHOST).
            # G-post: identical flagged shape to G-flag but POST season --
            #   REG-only gate blocks it.
            # G-missing: no schedule row at all -- treated as no signal.
            "home_cover_probability": [0.55, 0.35, 0.40, 0.60, 0.50],
        }
    )


# ---------------------------------------------------------------------------
# 1. bye_edge_flag_by_game: derived, structural (never outcome-based)
# ---------------------------------------------------------------------------


def test_flag_fires_for_a_strict_12_day_gap() -> None:
    flags = bye_edge_flag_by_game(_bye_schedule()).set_index("game_id")
    row = flags.loc["2026_06_NEITHERHOST_BYETEAM"]
    assert bool(row["away_off_bye"]) is True  # BYETEAM, 14-day gap
    assert bool(row["home_off_bye"]) is False  # NEITHERHOST, played every week


def test_flag_does_not_fire_for_an_11_day_gap() -> None:
    """The strict threshold is >=12 days -- 11 days must NOT count."""

    flags = bye_edge_flag_by_game(_bye_schedule()).set_index("game_id")
    row = flags.loc["2026_04_ELEVENHOST_BYETEAM"]
    assert bool(row["home_off_bye"]) is False  # ELEVENHOST's own gap is 11 days


def test_flag_fires_for_both_teams_off_a_strict_bye_simultaneously() -> None:
    flags = bye_edge_flag_by_game(_bye_schedule()).set_index("game_id")
    row = flags.loc["2026_03_TWELVEHOST_BYETEAM"]
    assert bool(row["home_off_bye"]) is True  # TWELVEHOST, 14-day gap
    assert bool(row["away_off_bye"]) is True  # BYETEAM, 14-day gap


def test_flag_is_false_for_a_teams_first_game_of_the_season() -> None:
    """No preceding game this season -- gap is undefined (NaN) -- never a bye."""

    flags = bye_edge_flag_by_game(_bye_schedule()).set_index("game_id")
    row = flags.loc["2026_01_BYETEAM_OPP1"]
    assert bool(row["home_off_bye"]) is False
    assert bool(row["away_off_bye"]) is False


def test_flag_requires_its_schedule_columns() -> None:
    with pytest.raises(DataContractError, match="bye-edge"):
        bye_edge_flag_by_game(pd.DataFrame({"game_id": ["G1"]}))


def test_flag_never_reads_outcome_columns() -> None:
    """AGENTS.md: a leakage regression test for every new feature family.

    ``bye_edge_flag_by_game`` does not even require/read
    ``result``/``spread_line`` -- adding them (with arbitrary values) and
    mutating them must never change the already-computed flags, proving the
    derivation is purely structural (gameday gaps within each team's own
    season), never outcome-based.
    """

    schedule = _bye_schedule()
    schedule["result"] = 0.0
    schedule["spread_line"] = -3.0
    baseline = bye_edge_flag_by_game(schedule).set_index("game_id")

    mutated = schedule.copy()
    mutated.loc[mutated["game_id"].eq("2026_06_NEITHERHOST_BYETEAM"), "result"] = 99.0
    mutated.loc[mutated["game_id"].eq("2026_06_NEITHERHOST_BYETEAM"), "spread_line"] = 14.0
    changed = bye_edge_flag_by_game(mutated).set_index("game_id")

    pd.testing.assert_frame_equal(changed, baseline, check_exact=True)


def test_flag_is_leak_safe_across_the_season_boundary() -> None:
    """A future season's schedule data (even for the same team) must never
    change an earlier season's already-computed flags -- and must never
    reintroduce the fixed cross-season bug (docs/bye_overvaluation_screen.md,
    "Correction 2026-08-22")."""

    schedule = _bye_schedule()
    baseline = bye_edge_flag_by_game(schedule)

    future = pd.DataFrame(
        [
            ("2027_01_BYETEAM_OPP1", 2027, 1, "REG", "2027-09-09", "BYETEAM", "OPP1"),
            ("2027_03_TWELVEHOST_BYETEAM", 2027, 3, "REG", "2027-09-23", "TWELVEHOST", "BYETEAM"),
        ],
        columns=schedule.columns,
    )
    changed = bye_edge_flag_by_game(pd.concat([schedule, future], ignore_index=True))

    pd.testing.assert_frame_equal(
        changed.loc[changed["season"].le(2026)].reset_index(drop=True),
        baseline.loc[baseline["season"].le(2026)].reset_index(drop=True),
        check_exact=True,
    )
    # The corrected (within-season) map: a 2027 season-opener must NOT
    # inherit the 2026 finale's gap -- exactly the bug the fix removed.
    future_flags = changed.set_index("game_id")
    assert bool(future_flags.loc["2027_01_BYETEAM_OPP1", "away_off_bye"]) is False


# ---------------------------------------------------------------------------
# 2. apply_bye_edge_fade_overlay: the pick-level transform
# ---------------------------------------------------------------------------


def test_overlay_flips_a_pick_on_the_strict_bye_holding_side() -> None:
    result = apply_bye_edge_fade_overlay(_predictions(), _bye_schedule())

    flipped_ids = {flip.game_id for flip in result.flips}
    assert "2026_06_NEITHERHOST_BYETEAM" in flipped_ids
    flip = next(f for f in result.flips if f.game_id == "2026_06_NEITHERHOST_BYETEAM")
    assert flip.bye_team == "BYETEAM"
    assert flip.opponent_team == "NEITHERHOST"

    overlaid = result.overlaid_predictions.set_index("game_id")
    assert overlaid.loc["2026_06_NEITHERHOST_BYETEAM", "home_cover_probability"] == pytest.approx(
        0.60
    )


def test_overlay_does_not_flip_a_both_off_bye_game() -> None:
    """Both-off-bye games are never touched -- the null-control case."""

    result = apply_bye_edge_fade_overlay(_predictions(), _bye_schedule())
    assert all(flip.game_id != "2026_03_TWELVEHOST_BYETEAM" for flip in result.flips)
    overlaid = result.overlaid_predictions.set_index("game_id")
    assert overlaid.loc["2026_03_TWELVEHOST_BYETEAM", "home_cover_probability"] == pytest.approx(
        0.55
    )


def test_overlay_does_not_flip_when_neither_side_is_off_a_strict_bye() -> None:
    """An 11-day gap is not a strict bye -- neither side is flagged."""

    result = apply_bye_edge_fade_overlay(_predictions(), _bye_schedule())
    assert all(flip.game_id != "2026_04_ELEVENHOST_BYETEAM" for flip in result.flips)
    overlaid = result.overlaid_predictions.set_index("game_id")
    assert overlaid.loc["2026_04_ELEVENHOST_BYETEAM", "home_cover_probability"] == pytest.approx(
        0.35
    )


def test_overlay_does_not_flip_when_the_pick_is_not_on_the_bye_side() -> None:
    """The flip only fires when the model's own pick sits on the bye-holding
    side -- a pick already on the non-bye side is left untouched."""

    predictions = _predictions()
    predictions.loc[
        predictions["game_id"].eq("2026_06_NEITHERHOST_BYETEAM"), "home_cover_probability"
    ] = 0.65  # already picks home (NEITHERHOST, not the bye team)

    result = apply_bye_edge_fade_overlay(predictions, _bye_schedule())
    assert all(flip.game_id != "2026_06_NEITHERHOST_BYETEAM" for flip in result.flips)
    overlaid = result.overlaid_predictions.set_index("game_id")
    assert overlaid.loc["2026_06_NEITHERHOST_BYETEAM", "home_cover_probability"] == pytest.approx(
        0.65
    )


def test_overlay_leaves_postseason_games_untouched() -> None:
    """Same flagged shape as the flipped clean case, but POST season -- the
    REG-only gate blocks it."""

    result = apply_bye_edge_fade_overlay(_predictions(), _bye_schedule())
    assert all(flip.game_id != "2026_20_POSTHOST_BYETEAM" for flip in result.flips)
    overlaid = result.overlaid_predictions.set_index("game_id")
    assert overlaid.loc["2026_20_POSTHOST_BYETEAM", "home_cover_probability"] == pytest.approx(0.60)


def test_overlay_treats_a_missing_schedule_row_as_no_signal() -> None:
    result = apply_bye_edge_fade_overlay(_predictions(), _bye_schedule())
    assert all(flip.game_id != "2026_MISSING_GAME" for flip in result.flips)
    overlaid = result.overlaid_predictions.set_index("game_id")
    assert overlaid.loc["2026_MISSING_GAME", "home_cover_probability"] == pytest.approx(0.50)


def test_overlay_disabled_is_a_no_op() -> None:
    predictions = _predictions()
    result = apply_bye_edge_fade_overlay(predictions, _bye_schedule(), enabled=False)

    assert result.flip_count == 0
    pd.testing.assert_frame_equal(
        result.overlaid_predictions.reset_index(drop=True),
        predictions.reset_index(drop=True),
        check_exact=True,
    )


def test_overlay_changes_only_home_cover_probability_on_flipped_rows() -> None:
    """Additivity: every other column, and every untouched row, stays
    byte-identical -- the pick-level design's whole point. Also proves 'no
    effect outside the flagged population': only the one XOR-flagged,
    pick-on-bye-side game moves."""

    predictions = _predictions()
    result = apply_bye_edge_fade_overlay(predictions, _bye_schedule())
    overlaid = result.overlaid_predictions

    assert result.flip_count == 1
    assert list(overlaid.columns) == list(predictions.columns)
    other_columns = [c for c in predictions.columns if c != "home_cover_probability"]
    pd.testing.assert_frame_equal(
        overlaid[other_columns].reset_index(drop=True),
        predictions[other_columns].reset_index(drop=True),
        check_exact=True,
    )
    untouched = predictions["game_id"].isin(
        [
            "2026_03_TWELVEHOST_BYETEAM",
            "2026_04_ELEVENHOST_BYETEAM",
            "2026_20_POSTHOST_BYETEAM",
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
        apply_bye_edge_fade_overlay(pd.DataFrame({"game_id": ["G1"]}), _bye_schedule())


# ---------------------------------------------------------------------------
# 3. overlay_disclosure_note: the plain-English provenance sentence
# ---------------------------------------------------------------------------


def test_disclosure_note_is_empty_when_nothing_flipped() -> None:
    matched_only = _predictions().loc[
        lambda frame: frame["game_id"].eq("2026_03_TWELVEHOST_BYETEAM")
    ]
    result = apply_bye_edge_fade_overlay(matched_only, _bye_schedule())
    assert overlay_disclosure_note(result) == ""

    disabled = apply_bye_edge_fade_overlay(_predictions(), _bye_schedule(), enabled=False)
    assert overlay_disclosure_note(disabled) == ""


def test_disclosure_note_states_the_flip_count_and_does_not_claim_production() -> None:
    result = apply_bye_edge_fade_overlay(_predictions(), _bye_schedule())
    note = overlay_disclosure_note(result)

    assert "Tilt applied: 1 pick flipped" in note
    assert "BYETEAM -> NEITHERHOST" in note
    assert "not applied to the published card" in note


# ---------------------------------------------------------------------------
# 4. record_bye_edge_fade_challenger_decisions: dual-tracked, no window cost
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
            "game_id": ["2026_06_NEITHERHOST_BYETEAM", "2026_04_ELEVENHOST_BYETEAM"],
            "season": [2026, 2026],
            "week": [6, 4],
            "game_type": ["REG", "REG"],
            "home_team": ["NEITHERHOST", "ELEVENHOST"],
            "away_team": ["BYETEAM", "BYETEAM"],
            "kickoff": ["2026-10-22T17:00:00+00:00", "2026-10-22T17:00:00+00:00"],
            "spread_line": [-1.5, 2.0],
            "home_cover_probability": [0.40, 0.35],
        }
    )


def _write_registry(artifacts: Path, *, status: str = "ACTIVE_PROSPECTIVE") -> None:
    write_challenger_registry(
        artifacts, challenger_id=CHALLENGER_ID, model_config=_MODEL_CONFIG, status=status
    )


_SEASON, _WEEK, _CREATED_AT_UTC = 2026, 6, "2026-10-13T15:00:00+00:00"


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
    data_root = tmp_path / "data"
    write_snapshot(
        _bye_schedule(),
        pd.DataFrame({"game_id": [], "team": []}),
        seasons=[2026],
        raw_root=data_root / "raw",
    )
    return data_root


def test_record_bye_edge_fade_challenger_decisions_records_the_fade_arm(
    tmp_path: Path,
) -> None:
    artifacts = tmp_path / "artifacts"
    _write_registry(artifacts)
    _write_active_model_and_card(artifacts)
    data_root = _write_data_root(tmp_path)
    now = datetime(2026, 10, 20, 16, 0, tzinfo=UTC)

    result = record_bye_edge_fade_challenger_decisions(artifacts, data_root, now=now)

    assert result["recorded"] == 2
    assert result["flip_count"] == 1
    assert result["flipped_game_ids"] == ["2026_06_NEITHERHOST_BYETEAM"]

    ledger = load_challenger_decisions(artifacts).set_index("game_id")
    assert list(load_challenger_decisions(artifacts).columns) == list(CHALLENGER_DECISION_COLUMNS)
    assert (ledger["bet_side"] == "PASS").all()
    assert ledger["edge"].isna().all()

    # The fade's own arm diverges from the active model's raw pick (0.40 ->
    # AWAY/BYETEAM, the bye-holding side): the fade flips it to HOME.
    assert ledger.loc["2026_06_NEITHERHOST_BYETEAM", "pick_side"] == "HOME"
    # The no-signal (11-day-gap) game keeps the model's own AWAY pick.
    assert ledger.loc["2026_04_ELEVENHOST_BYETEAM", "pick_side"] == "AWAY"

    # Re-running is a no-op: append-only, never rewrites.
    again = record_bye_edge_fade_challenger_decisions(artifacts, data_root, now=now)
    assert again["recorded"] == 0
    assert again["already_recorded"] == 2


def test_record_bye_edge_fade_challenger_refuses_outside_recording_lock_window(
    tmp_path: Path,
) -> None:
    artifacts = tmp_path / "artifacts"
    _write_registry(artifacts)
    _write_active_model_and_card(artifacts)
    data_root = _write_data_root(tmp_path)

    with pytest.raises(ValueError, match="RECORDING_LOCK_WINDOW"):
        record_bye_edge_fade_challenger_decisions(
            artifacts, data_root, now=datetime(2026, 8, 18, 1, 0, tzinfo=UTC)
        )
    assert load_challenger_decisions(artifacts).empty


def test_record_bye_edge_fade_challenger_refuses_a_fingerprint_mismatch(
    tmp_path: Path,
) -> None:
    artifacts = tmp_path / "artifacts"
    _write_registry(artifacts)
    # The active model's OWN configuration moved (a promotion) since this
    # challenger was pinned -- recording must refuse, not silently switch
    # base models under the same challenger id. This is the fingerprint
    # stability guard: a retuned/foreign model config must be refused.
    _write_active_model_and_card(artifacts, ridge_alpha=1.0)
    data_root = _write_data_root(tmp_path)

    with pytest.raises(DataContractError, match="configuration fingerprint"):
        record_bye_edge_fade_challenger_decisions(
            artifacts, data_root, now=datetime(2026, 10, 20, 16, 0, tzinfo=UTC)
        )
    assert load_challenger_decisions(artifacts).empty


def test_record_bye_edge_fade_challenger_refuses_an_inactive_registration(
    tmp_path: Path,
) -> None:
    artifacts = tmp_path / "artifacts"
    _write_registry(artifacts, status="CLOSED_BEFORE_ACTIVATION")
    _write_active_model_and_card(artifacts)
    data_root = _write_data_root(tmp_path)

    with pytest.raises(ValueError, match="only ACTIVE_PROSPECTIVE"):
        record_bye_edge_fade_challenger_decisions(
            artifacts, data_root, now=datetime(2026, 10, 20, 16, 0, tzinfo=UTC)
        )


def test_bye_edge_fade_fingerprint_helper_agrees_with_the_registered_model_block() -> None:
    """Sanity check that the fixture's config really matches CONFIG_FINGERPRINT_KEYS,
    and (via the FINGERPRINT block quoted in the task brief) the real active
    model's own configuration."""

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
