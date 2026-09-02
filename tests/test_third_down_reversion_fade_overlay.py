"""Tests for the third-down mean-reversion fade overlay
(docs/third_down_reversion_fade_overlay.md).

Mirrors ``tests/test_tank_zone_fade_tilt_overlay.py``'s structure and
AGENTS.md's "add a leakage regression test for every new feature family"
mandate. Six things are load-bearing:

1. :func:`third_down_over_flag_by_game` reproduces the registered cell's
   flag -- a team's PRIOR-season centered 3rd-down conversion rate at or
   above the registry cell's own FROZEN, GLOBAL top-quartile cutoff
   (:data:`THIRD_DOWN_TOP_QUARTILE_CENTERED`) -- and is DATA-DERIVED from a
   synthetic PBP fixture, never a hardcoded team list.
2. The threshold is proven GLOBAL/frozen, not a locally recomputed
   within-sample quantile.
3. It is PREGAME-SAFE: a game's own current-season PBP/outcome and any
   LATER season's PBP can never move an earlier season's flag. Two explicit
   leakage regression tests.
4. Missing prior-season data means ``flag=False``, never an exception.
5. :func:`apply_third_down_reversion_fade_overlay` fades the flagged side
   ONLY in the "clean case" (exactly one side flagged), ONLY when the
   model's own pick is that side, and ONLY in REG-season games.
6. :func:`record_third_down_reversion_fade_challenger_decisions` writes the
   overlay's own picks to the prospective challenger ledger, refuses a
   retuned/foreign model configuration (fingerprint stability), and refuses
   a non-``ACTIVE_PROSPECTIVE`` registration.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from _overlay_test_kit import write_active_model_and_card, write_challenger_registry

from nfl_ats.data import DataContractError
from nfl_ats.pbp import PBP_SNAPSHOT_COLUMNS, write_pbp_snapshot
from nfl_ats.prospective_scoring import (
    CHALLENGER_DECISION_COLUMNS,
    artifact_model_config,
    config_fingerprint,
    load_challenger_decisions,
)
from nfl_ats.snapshots import write_snapshot
from nfl_ats.third_down_reversion_fade_overlay import (
    CHALLENGER_ID,
    THIRD_DOWN_TOP_QUARTILE_CENTERED,
    apply_third_down_reversion_fade_overlay,
    overlay_disclosure_note,
    record_third_down_reversion_fade_challenger_decisions,
    third_down_over_flag_by_game,
)

# ---------------------------------------------------------------------------
# PBP fixture helpers
# ---------------------------------------------------------------------------


def _pbp_row(**overrides: object) -> dict[str, object]:
    row: dict[str, object] = dict.fromkeys(PBP_SNAPSHOT_COLUMNS, np.nan)
    row.update(
        {
            "season_type": "REG",
            "epa": 0.0,
            "wp": 0.5,
            "play": 1,
            "qb_kneel": 0,
            "qb_spike": 0,
            "aborted_play": 0,
            "pass_attempt": 1,
            "rush_attempt": 0,
            "fixed_drive": 1,
            "week": 1,
        }
    )
    row.update(overrides)
    return row


def _third_down_plays(
    *, season: int, team: str, opponent: str, n_total: int, n_conversions: int, prefix: str
) -> list[dict[str, object]]:
    """``n_total`` third-down plays for ``team``, ``n_conversions`` of them converted."""

    rows = []
    for index in range(n_total):
        rows.append(
            _pbp_row(
                play_id=float(index + 1),
                game_id=f"{prefix}_{index + 1}",
                season=season,
                home_team=team,
                away_team=opponent,
                posteam=team,
                defteam=opponent,
                down=3.0,
                first_down=1.0 if index < n_conversions else 0.0,
            )
        )
    return rows


# Season 2025 (the PRIOR season for every 2026 game below). Five teams:
#   AAA 6/10 = 0.60, FFF 6/10 = 0.60, BBB 4/10 = 0.40, CCC 3/10 = 0.30,
#   DDD 2/10 = 0.20 -- league mean 0.42, centered AAA=+0.18, FFF=+0.18,
#   BBB=-0.02, CCC=-0.12, DDD=-0.22. THIRD_DOWN_TOP_QUARTILE_CENTERED is
#   ~0.0339, so AAA and FFF (both +0.18) are clearly flagged and the other
#   three are clearly not -- deliberately far from the cutoff so the test
#   is robust to float rounding, not a boundary probe.
# GGG and HHH never appear in season 2025 at all (missing prior data).
def _season_2025_pbp() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    rows += _third_down_plays(
        season=2025, team="AAA", opponent="ZZZ", n_total=10, n_conversions=6, prefix="p25aaa"
    )
    rows += _third_down_plays(
        season=2025, team="FFF", opponent="ZZZ", n_total=10, n_conversions=6, prefix="p25fff"
    )
    rows += _third_down_plays(
        season=2025, team="BBB", opponent="ZZZ", n_total=10, n_conversions=4, prefix="p25bbb"
    )
    rows += _third_down_plays(
        season=2025, team="CCC", opponent="ZZZ", n_total=10, n_conversions=3, prefix="p25ccc"
    )
    rows += _third_down_plays(
        season=2025, team="DDD", opponent="ZZZ", n_total=10, n_conversions=2, prefix="p25ddd"
    )
    return rows


# Season 2020 (the PRIOR season for the 2021 global-vs-local test game).
# Four teams, all CLOSE to the frozen cutoff on purpose:
#   QQQ 29/100=0.29, RRR 27/100=0.27, SSS 25/100=0.25, TTT 23/100=0.23,
#   league mean 0.26, centered QQQ=+0.03, RRR=+0.01, SSS=-0.01, TTT=-0.03.
# QQQ's +0.03 sits just BELOW THIRD_DOWN_TOP_QUARTILE_CENTERED (~0.033926),
# so the correct (global, frozen) rule never flags it. But the LOCAL
# quantile(0.75) of these four centered values (linear interpolation) is
# ~0.015 -- comfortably BELOW QQQ's +0.03 -- so a (wrong) implementation
# that recomputed a quantile from this local sample would flag QQQ. This is
# the fixture that distinguishes "global, frozen" from "within-season/
# within-sample" behaviour.
def _season_2020_pbp() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    rows += _third_down_plays(
        season=2020, team="QQQ", opponent="YYY", n_total=100, n_conversions=29, prefix="p20qqq"
    )
    rows += _third_down_plays(
        season=2020, team="RRR", opponent="YYY", n_total=100, n_conversions=27, prefix="p20rrr"
    )
    rows += _third_down_plays(
        season=2020, team="SSS", opponent="YYY", n_total=100, n_conversions=25, prefix="p20sss"
    )
    rows += _third_down_plays(
        season=2020, team="TTT", opponent="YYY", n_total=100, n_conversions=23, prefix="p20ttt"
    )
    return rows


def _pbp_frame(*extra_seasons: list[dict[str, object]]) -> pd.DataFrame:
    rows = [*_season_2025_pbp(), *_season_2020_pbp()]
    for season_rows in extra_seasons:
        rows += season_rows
    return pd.DataFrame(rows)


_SCHEDULE_COLUMNS = ["game_id", "season", "week", "game_type", "home_team", "away_team"]

_GAME_HOME_FLAGGED = "2026_01_AAA_BBB"
_GAME_AWAY_FLAGGED = "2026_01_CCC_AAA"
_GAME_BOTH_FLAGGED = "2026_01_AAA_FFF"
_GAME_PICK_OPPONENT = "2026_01_AAA_DDD"
_GAME_NO_FLAG = "2026_01_BBB_CCC"
_GAME_MISSING_PRIOR = "2026_01_GGG_HHH"
_GAME_MISSING_SCHEDULE_ROW = "2026_01_MISSING_MISSING"
_GAME_GLOBAL_VS_LOCAL = "2021_01_QQQ_RRR"


def _schedule() -> pd.DataFrame:
    rows = [
        (_GAME_HOME_FLAGGED, 2026, 1, "REG", "AAA", "BBB"),
        (_GAME_AWAY_FLAGGED, 2026, 1, "REG", "CCC", "AAA"),
        (_GAME_BOTH_FLAGGED, 2026, 1, "REG", "AAA", "FFF"),
        (_GAME_PICK_OPPONENT, 2026, 1, "REG", "AAA", "DDD"),
        (_GAME_NO_FLAG, 2026, 1, "REG", "BBB", "CCC"),
        (_GAME_MISSING_PRIOR, 2026, 1, "REG", "GGG", "HHH"),
        (_GAME_GLOBAL_VS_LOCAL, 2021, 1, "REG", "QQQ", "RRR"),
    ]
    return pd.DataFrame(rows, columns=_SCHEDULE_COLUMNS)


def _flags() -> pd.DataFrame:
    return third_down_over_flag_by_game(_schedule(), _pbp_frame()).set_index("game_id")


def _predictions() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "game_id": [
                _GAME_HOME_FLAGGED,
                _GAME_AWAY_FLAGGED,
                _GAME_BOTH_FLAGGED,
                _GAME_PICK_OPPONENT,
                _GAME_NO_FLAG,
                _GAME_MISSING_PRIOR,
                _GAME_MISSING_SCHEDULE_ROW,
                _GAME_GLOBAL_VS_LOCAL,
            ],
            "season": [2026, 2026, 2026, 2026, 2026, 2026, 2026, 2021],
            "week": [1] * 8,
            "game_type": ["REG"] * 8,
            "home_team": ["AAA", "CCC", "AAA", "AAA", "BBB", "GGG", "MISS_H", "QQQ"],
            "away_team": ["BBB", "AAA", "FFF", "DDD", "CCC", "HHH", "MISS_A", "RRR"],
            "kickoff": ["2026-09-10T17:00:00+00:00"] * 8,
            "spread_line": [-3.0, 3.0, -3.0, -3.0, 1.0, -1.0, 1.0, -6.0],
            # HomeFlag: model picks HOME (AAA, flagged)         -> flip to 0.35.
            # AwayFlag: model picks AWAY (AAA, flagged)         -> flip to 0.70.
            # Both:     both sides flagged                      -> never flipped.
            # PickOpp:  model picks AWAY (DDD, not flagged)     -> no flip.
            # NoFlag:   neither side flagged                    -> no flip.
            # MissPrior:neither team has prior data             -> no flip.
            # Missing:  no schedule row at all                  -> no flip.
            # GlobalVsLocal: model picks HOME (QQQ); QQQ's centered rate
            #   (+0.03) is below the frozen threshold, so no flip even
            #   though QQQ would be the local top quartile of its 4-team
            #   sample.
            "home_cover_probability": [0.65, 0.30, 0.65, 0.30, 0.55, 0.60, 0.50, 0.70],
        }
    )


# ---------------------------------------------------------------------------
# 1. third_down_over_flag_by_game: the derived flag
# ---------------------------------------------------------------------------


def test_flag_fires_for_a_team_whose_prior_season_rate_is_top_quartile() -> None:
    flags = _flags()
    assert bool(flags.loc[_GAME_HOME_FLAGGED, "third_down_over_home"]) is True
    assert bool(flags.loc[_GAME_HOME_FLAGGED, "third_down_over_away"]) is False
    assert bool(flags.loc[_GAME_AWAY_FLAGGED, "third_down_over_away"]) is True
    assert bool(flags.loc[_GAME_AWAY_FLAGGED, "third_down_over_home"]) is False


def test_flag_fires_on_both_sides_of_a_double_flagged_game() -> None:
    flags = _flags()
    assert bool(flags.loc[_GAME_BOTH_FLAGGED, "third_down_over_home"]) is True
    assert bool(flags.loc[_GAME_BOTH_FLAGGED, "third_down_over_away"]) is True


def test_flag_is_false_for_teams_below_the_top_quartile() -> None:
    flags = _flags()
    assert bool(flags.loc[_GAME_NO_FLAG, "third_down_over_home"]) is False
    assert bool(flags.loc[_GAME_NO_FLAG, "third_down_over_away"]) is False


def test_flag_requires_its_schedule_columns() -> None:
    with pytest.raises(DataContractError, match="third-down reversion tracking"):
        third_down_over_flag_by_game(
            pd.DataFrame({"game_id": ["G1"], "season": [2026]}), _pbp_frame()
        )


# ---------------------------------------------------------------------------
# 2. The threshold is GLOBAL/frozen, not a locally recomputed quantile
# ---------------------------------------------------------------------------


def test_threshold_constant_matches_the_measured_registry_cell_value() -> None:
    """The frozen constant is the cell's own measured number, transcribed.

    Read: artifacts/redzone_reversion_screen/20260821T181025Z/results.json:349
    (thresholds.third_down_q75), the same value
    scripts/redzone_reversion_screen.py:383 computed.
    """

    assert pytest.approx(0.03392624406886406, abs=1e-15) == THIRD_DOWN_TOP_QUARTILE_CENTERED


def test_flag_uses_the_frozen_global_threshold_not_a_locally_recomputed_quantile() -> None:
    """QQQ's centered rate (+0.03) is below the frozen +0.033926 cutoff, so
    it must NOT be flagged -- even though it is the top of its own 4-team
    local sample and a locally recomputed quantile(0.75) (~0.015) would
    flag it. This is the fixture that proves the threshold is the
    registry cell's own GLOBAL, pooled-across-2009-2025 number, not
    something recomputed within a season or within a sample."""

    flags = _flags()
    assert bool(flags.loc[_GAME_GLOBAL_VS_LOCAL, "third_down_over_home"]) is False
    assert bool(flags.loc[_GAME_GLOBAL_VS_LOCAL, "third_down_over_away"]) is False

    result = apply_third_down_reversion_fade_overlay(_predictions(), _schedule(), _pbp_frame())
    assert all(flip.game_id != _GAME_GLOBAL_VS_LOCAL for flip in result.flips)


# ---------------------------------------------------------------------------
# 3. Leakage regressions: pregame-only inputs (AGENTS.md mandate)
# ---------------------------------------------------------------------------


def test_flag_is_leak_safe_against_the_current_seasons_own_pbp() -> None:
    """A team's CURRENT-season (2026) PBP data must never move a 2026 game's
    flag -- only the PRIOR season (2025) is ever read. Simulate AAA posting
    a perfect (10/10) current-season third-down rate; the 2026 flags,
    which depend only on 2025, must be byte-identical either way."""

    baseline = _flags()
    current_season_noise = _third_down_plays(
        season=2026, team="AAA", opponent="ZZZ", n_total=10, n_conversions=10, prefix="p26aaa"
    )
    with_current_season = third_down_over_flag_by_game(
        _schedule(), _pbp_frame(current_season_noise)
    ).set_index("game_id")

    pd.testing.assert_frame_equal(
        baseline.loc[:, ["third_down_over_home", "third_down_over_away"]],
        with_current_season.loc[:, ["third_down_over_home", "third_down_over_away"]],
    )


def test_flag_is_leak_safe_against_a_later_seasons_pbp() -> None:
    """A LATER season's (2027) PBP data must never change an earlier
    season's (2026, which reads only 2025) already-computed flags."""

    baseline = _flags()
    later_season_noise = _third_down_plays(
        season=2027, team="BBB", opponent="ZZZ", n_total=10, n_conversions=10, prefix="p27bbb"
    )
    with_later_season = third_down_over_flag_by_game(
        _schedule(), _pbp_frame(later_season_noise)
    ).set_index("game_id")

    pd.testing.assert_frame_equal(
        baseline.loc[:, ["third_down_over_home", "third_down_over_away"]],
        with_later_season.loc[:, ["third_down_over_home", "third_down_over_away"]],
    )


def test_missing_prior_season_data_means_unflagged_never_an_error() -> None:
    """GGG and HHH never appear in the 2025 PBP fixture at all -- a team
    with no observed prior season (first year in the data, an expansion
    team, a gap year) is simply unflagged, never a crash."""

    flags = _flags()
    assert bool(flags.loc[_GAME_MISSING_PRIOR, "third_down_over_home"]) is False
    assert bool(flags.loc[_GAME_MISSING_PRIOR, "third_down_over_away"]) is False


# ---------------------------------------------------------------------------
# 4. apply_third_down_reversion_fade_overlay: the pick-level transform
# ---------------------------------------------------------------------------


def test_overlay_fades_the_flagged_side_when_the_model_picks_it() -> None:
    result = apply_third_down_reversion_fade_overlay(_predictions(), _schedule(), _pbp_frame())
    flipped = {flip.game_id for flip in result.flips}
    assert flipped == {_GAME_HOME_FLAGGED, _GAME_AWAY_FLAGGED}

    overlaid = result.overlaid_predictions.set_index("game_id")
    assert overlaid.loc[_GAME_HOME_FLAGGED, "home_cover_probability"] == pytest.approx(0.35)
    assert overlaid.loc[_GAME_AWAY_FLAGGED, "home_cover_probability"] == pytest.approx(0.70)

    home_flip = next(f for f in result.flips if f.game_id == _GAME_HOME_FLAGGED)
    assert home_flip.flagged_team == "AAA"
    assert home_flip.opponent_team == "BBB"
    away_flip = next(f for f in result.flips if f.game_id == _GAME_AWAY_FLAGGED)
    assert away_flip.flagged_team == "AAA"
    assert away_flip.opponent_team == "CCC"


def test_overlay_leaves_a_both_flagged_game_untouched() -> None:
    result = apply_third_down_reversion_fade_overlay(_predictions(), _schedule(), _pbp_frame())
    assert all(flip.game_id != _GAME_BOTH_FLAGGED for flip in result.flips)
    assert _GAME_BOTH_FLAGGED in result.both_flagged_games
    overlaid = result.overlaid_predictions.set_index("game_id")
    assert overlaid.loc[_GAME_BOTH_FLAGGED, "home_cover_probability"] == pytest.approx(0.65)


def test_overlay_does_not_flip_when_the_pick_is_already_off_the_flagged_side() -> None:
    result = apply_third_down_reversion_fade_overlay(_predictions(), _schedule(), _pbp_frame())
    assert all(flip.game_id != _GAME_PICK_OPPONENT for flip in result.flips)
    overlaid = result.overlaid_predictions.set_index("game_id")
    assert overlaid.loc[_GAME_PICK_OPPONENT, "home_cover_probability"] == pytest.approx(0.30)


def test_overlay_does_not_flip_a_game_with_no_flagged_side() -> None:
    result = apply_third_down_reversion_fade_overlay(_predictions(), _schedule(), _pbp_frame())
    assert all(flip.game_id != _GAME_NO_FLAG for flip in result.flips)


def test_overlay_never_flips_outside_the_flagged_population() -> None:
    """No effect outside the flagged population: every untouched row keeps
    its exact original probability, not just "not in flips"."""

    predictions = _predictions()
    result = apply_third_down_reversion_fade_overlay(predictions, _schedule(), _pbp_frame())
    overlaid = result.overlaid_predictions.set_index("game_id")
    untouched_ids = [_GAME_PICK_OPPONENT, _GAME_NO_FLAG, _GAME_MISSING_PRIOR, _GAME_GLOBAL_VS_LOCAL]
    original = predictions.set_index("game_id")
    for game_id in untouched_ids:
        assert overlaid.loc[game_id, "home_cover_probability"] == pytest.approx(
            original.loc[game_id, "home_cover_probability"]
        )


def test_overlay_treats_a_missing_schedule_row_as_no_signal() -> None:
    result = apply_third_down_reversion_fade_overlay(_predictions(), _schedule(), _pbp_frame())
    assert all(flip.game_id != _GAME_MISSING_SCHEDULE_ROW for flip in result.flips)
    overlaid = result.overlaid_predictions.set_index("game_id")
    assert overlaid.loc[_GAME_MISSING_SCHEDULE_ROW, "home_cover_probability"] == pytest.approx(0.50)


def test_overlay_leaves_a_flagged_game_untouched_when_marked_postseason() -> None:
    predictions = _predictions().loc[lambda frame: frame["game_id"].eq(_GAME_HOME_FLAGGED)]
    postseason = predictions.assign(game_type="POST")
    result = apply_third_down_reversion_fade_overlay(postseason, _schedule(), _pbp_frame())
    assert result.flip_count == 0


def test_overlay_disabled_is_a_no_op() -> None:
    predictions = _predictions()
    result = apply_third_down_reversion_fade_overlay(
        predictions, _schedule(), _pbp_frame(), enabled=False
    )
    assert result.flip_count == 0
    pd.testing.assert_frame_equal(
        result.overlaid_predictions.reset_index(drop=True),
        predictions.assign(game_id=predictions["game_id"].astype(str)).reset_index(drop=True),
        check_exact=True,
    )


def test_overlay_changes_only_home_cover_probability_on_flipped_rows() -> None:
    predictions = _predictions()
    result = apply_third_down_reversion_fade_overlay(predictions, _schedule(), _pbp_frame())
    overlaid = result.overlaid_predictions

    assert list(overlaid.columns) == list(predictions.columns)
    other_columns = [c for c in predictions.columns if c != "home_cover_probability"]
    pd.testing.assert_frame_equal(
        overlaid[other_columns].reset_index(drop=True),
        predictions[other_columns].reset_index(drop=True),
        check_exact=True,
    )
    flipped_ids = [_GAME_HOME_FLAGGED, _GAME_AWAY_FLAGGED]
    untouched = ~predictions["game_id"].isin(flipped_ids)
    pd.testing.assert_series_equal(
        overlaid.loc[untouched, "home_cover_probability"].reset_index(drop=True),
        predictions.loc[untouched, "home_cover_probability"].reset_index(drop=True),
        check_exact=True,
    )


def test_overlay_requires_its_prediction_columns() -> None:
    with pytest.raises(DataContractError, match="overlay columns"):
        apply_third_down_reversion_fade_overlay(
            pd.DataFrame({"game_id": ["G1"]}), _schedule(), _pbp_frame()
        )


# ---------------------------------------------------------------------------
# 5. overlay_disclosure_note
# ---------------------------------------------------------------------------


def test_disclosure_note_is_empty_when_nothing_flipped() -> None:
    quiet = _predictions().loc[lambda frame: frame["game_id"].eq(_GAME_NO_FLAG)]
    result = apply_third_down_reversion_fade_overlay(quiet, _schedule(), _pbp_frame())
    assert overlay_disclosure_note(result) == ""
    disabled = apply_third_down_reversion_fade_overlay(
        _predictions(), _schedule(), _pbp_frame(), enabled=False
    )
    assert overlay_disclosure_note(disabled) == ""


def test_disclosure_note_states_the_flip_count_and_does_not_claim_production() -> None:
    result = apply_third_down_reversion_fade_overlay(_predictions(), _schedule(), _pbp_frame())
    note = overlay_disclosure_note(result)
    assert "2 picks flipped" in note
    assert "AAA -> BBB" in note
    assert "AAA -> CCC" in note
    assert "not applied to the published card" in note


# ---------------------------------------------------------------------------
# 6. record_third_down_reversion_fade_challenger_decisions: dual-tracked
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

_SEASON, _WEEK, _CREATED_AT_UTC = 2026, 1, "2026-09-08T15:00:00+00:00"
_NOW = datetime(2026, 9, 8, 16, 0, tzinfo=UTC)


def _write_data_root(tmp_path: Path) -> Path:
    repo_root = tmp_path / "repo"
    data_root = repo_root / "data"
    write_snapshot(
        _schedule(),
        pd.DataFrame({"game_id": [], "team": []}),
        seasons=[2021, 2026],
        raw_root=data_root / "raw",
    )
    write_pbp_snapshot({2025: pd.DataFrame(_season_2025_pbp())}, data_root / "pbp" / "raw")
    return data_root


def _recorder_predictions() -> pd.DataFrame:
    return _predictions().loc[
        lambda frame: frame["game_id"].isin([_GAME_HOME_FLAGGED, _GAME_NO_FLAG])
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


def test_record_third_down_reversion_fade_challenger_decisions_records_the_fade_arm(
    tmp_path: Path,
) -> None:
    artifacts = tmp_path / "artifacts"
    _write_registry(artifacts)
    _write_active_model_and_card(artifacts)
    data_root = _write_data_root(tmp_path)

    result = record_third_down_reversion_fade_challenger_decisions(artifacts, data_root, now=_NOW)

    assert result["recorded"] == 2
    assert result["flip_count"] == 1
    assert result["flipped_game_ids"] == [_GAME_HOME_FLAGGED]

    ledger = load_challenger_decisions(artifacts)
    assert list(ledger.columns) == list(CHALLENGER_DECISION_COLUMNS)
    assert (ledger["bet_side"] == "PASS").all()
    assert ledger["edge"].isna().all()

    indexed = ledger.set_index("game_id")
    # The model's raw pick was HOME (0.65 -> AAA, flagged); the fade flips
    # it to AWAY.
    assert indexed.loc[_GAME_HOME_FLAGGED, "pick_side"] == "AWAY"
    # The unflagged game keeps the model's own pick (0.55 -> HOME).
    assert indexed.loc[_GAME_NO_FLAG, "pick_side"] == "HOME"

    again = record_third_down_reversion_fade_challenger_decisions(artifacts, data_root, now=_NOW)
    assert again["recorded"] == 0
    assert again["already_recorded"] == 2


def test_record_third_down_reversion_fade_challenger_refuses_outside_recording_lock_window(
    tmp_path: Path,
) -> None:
    artifacts = tmp_path / "artifacts"
    _write_registry(artifacts)
    _write_active_model_and_card(artifacts)
    data_root = _write_data_root(tmp_path)

    with pytest.raises(ValueError, match="RECORDING_LOCK_WINDOW"):
        record_third_down_reversion_fade_challenger_decisions(
            artifacts, data_root, now=datetime(2026, 8, 1, 1, 0, tzinfo=UTC)
        )
    assert load_challenger_decisions(artifacts).empty


def test_record_third_down_reversion_fade_challenger_refuses_a_fingerprint_mismatch(
    tmp_path: Path,
) -> None:
    """Fingerprint stability: a retuned or foreign active model configuration
    must refuse to record, never silently switch base models under this id."""

    artifacts = tmp_path / "artifacts"
    _write_registry(artifacts)
    _write_active_model_and_card(artifacts, ridge_alpha=1.0)
    data_root = _write_data_root(tmp_path)

    with pytest.raises(DataContractError, match="configuration fingerprint"):
        record_third_down_reversion_fade_challenger_decisions(artifacts, data_root, now=_NOW)
    assert load_challenger_decisions(artifacts).empty


def test_record_third_down_reversion_fade_challenger_refuses_an_inactive_registration(
    tmp_path: Path,
) -> None:
    artifacts = tmp_path / "artifacts"
    _write_registry(artifacts, status="CLOSED_BEFORE_ACTIVATION")
    _write_active_model_and_card(artifacts)
    data_root = _write_data_root(tmp_path)

    with pytest.raises(ValueError, match="only ACTIVE_PROSPECTIVE"):
        record_third_down_reversion_fade_challenger_decisions(artifacts, data_root, now=_NOW)


def test_third_down_reversion_fade_fingerprint_helper_agrees_with_the_registered_model_block() -> (
    None
):
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
