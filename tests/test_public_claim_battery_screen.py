"""Flag-correctness and leakage tests for
``scripts/public_claim_battery_screen.py`` (LEAD-57).

Mirrors ``tests/test_bye_overvaluation_screen.py``'s pattern: load the
script module by path (never executed as ``__main__``), build small
synthetic frames by hand, and assert the predeclared flags in
``docs/public_claim_battery.md`` do exactly what they claim -- including
that no flag ever depends on the current game's own outcome.
"""

from __future__ import annotations

import importlib.util
import math
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]


def _load_script(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, REPO / "scripts" / filename)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


screen = _load_script("public_claim_battery_screen_test", "public_claim_battery_screen.py")


_BASELINE: dict[str, object] = {
    "weekday": "Saturday",
    "gametime_hour": 13.0,
    "team_spread": -1.0,
    "is_home": True,
    "div_game": 0,
    "own_rest": 7.0,
    "prior_team_spread": 1.0,
    "prior_score_margin": 0.0,
    "week": 5,
    "prior_games": 4,
    "opp_prior_games": 4,
    "prior_win_pct": 0.5,
    "opp_prior_win_pct": 0.5,
    "ats_streak_len": 0.0,
    # Outcome-only columns: build_claims must never read these. Included
    # only so the leakage test has something to shuffle.
    "team_covered": 1.0,
    "team_score_margin": 3.0,
}

_ROWS: dict[str, dict[str, object]] = {
    # --- claim 2: primetime dog ---
    "primetime_dog_pos_thu": {"weekday": "Thursday", "team_spread": -2.5},
    "primetime_dog_neg_thu_favorite": {"weekday": "Thursday", "team_spread": 2.5},
    "primetime_dog_neg_saturday_night": {
        "weekday": "Saturday",
        "gametime_hour": 20.0,
        "team_spread": -2.5,
    },
    "primetime_dog_pos_sunday_night": {
        "weekday": "Sunday",
        "gametime_hour": 20.0,
        "team_spread": -1.0,
    },
    "primetime_dog_neg_sunday_day": {
        "weekday": "Sunday",
        "gametime_hour": 13.0,
        "team_spread": -1.0,
    },
    # --- claim 3: post-bye back ---
    "post_bye_pos": {"own_rest": 13.0},
    "post_bye_pos_boundary": {"own_rest": 12.0},
    "post_bye_neg": {"own_rest": 7.0},
    "post_bye_missing": {"own_rest": float("nan")},
    # --- claim 4: division dog ---
    "division_dog_pos": {"div_game": 1, "team_spread": -2.0},
    "division_dog_neg_favorite": {"div_game": 1, "team_spread": 2.0},
    "division_dog_neg_nondiv": {"div_game": 0, "team_spread": -2.0},
    # --- claim 5: big road favorite fade ---
    "road_fav_pos_boundary": {"is_home": False, "team_spread": 7.0},
    "road_fav_neg_below_threshold": {"is_home": False, "team_spread": 6.5},
    "road_fav_neg_home_favorite": {"is_home": True, "team_spread": 7.0},
    # --- claim 6: home dog 3+ ---
    "home_dog_pos_boundary": {"is_home": True, "team_spread": -3.0},
    "home_dog_neg_above_threshold": {"is_home": True, "team_spread": -2.5},
    "home_dog_neg_road_dog": {"is_home": False, "team_spread": -3.0},
    # --- claim 7: upset-letdown fade ---
    "upset_letdown_pos": {"prior_team_spread": -3.0, "prior_score_margin": 7.0},
    "upset_letdown_neg_was_favorite": {"prior_team_spread": 3.0, "prior_score_margin": 7.0},
    "upset_letdown_neg_lost": {"prior_team_spread": -3.0, "prior_score_margin": -7.0},
    "upset_letdown_neg_no_prior": {
        "prior_team_spread": float("nan"),
        "prior_score_margin": float("nan"),
    },
    # --- claim 8: 21+ blowout-loss bounce ---
    "blowout_bounce_pos": {"prior_score_margin": -24.0},
    "blowout_bounce_pos_boundary": {"prior_score_margin": -21.0},
    "blowout_bounce_neg_smaller_loss": {"prior_score_margin": -14.0},
    # --- claim 9: Week 1 dog ---
    "week1_dog_pos": {"week": 1, "team_spread": -2.0},
    "week1_dog_neg_favorite": {"week": 1, "team_spread": 2.0},
    "week1_dog_neg_week2": {"week": 2, "team_spread": -2.0},
    # --- claim 10: weeks 17-18 proxy elimination fade ---
    "eliminated_pos": {
        "week": 17,
        "prior_games": 14,
        "opp_prior_games": 14,
        "prior_win_pct": 0.30,
        "opp_prior_win_pct": 0.70,
    },
    "eliminated_neg_thin_own_record": {
        "week": 17,
        "prior_games": 8,
        "opp_prior_games": 14,
        "prior_win_pct": 0.30,
        "opp_prior_win_pct": 0.70,
    },
    "eliminated_neg_not_bad_enough": {
        "week": 18,
        "prior_games": 14,
        "opp_prior_games": 14,
        "prior_win_pct": 0.50,
        "opp_prior_win_pct": 0.70,
    },
    "eliminated_neg_wrong_week": {
        "week": 10,
        "prior_games": 14,
        "opp_prior_games": 14,
        "prior_win_pct": 0.30,
        "opp_prior_win_pct": 0.70,
    },
    # --- claim 12: ATS losing streak regression ---
    "ats_streak_pos": {"ats_streak_len": 3.0},
    "ats_streak_pos_longer": {"ats_streak_len": 5.0},
    "ats_streak_neg_below_threshold": {"ats_streak_len": 2.0},
}


def _fixture_frame() -> pd.DataFrame:
    rows = []
    names = []
    for name, overrides in _ROWS.items():
        row = dict(_BASELINE)
        row.update(overrides)
        rows.append(row)
        names.append(name)
    frame = pd.DataFrame(rows, index=names)
    frame["is_home"] = frame["is_home"].astype(bool)
    return frame


def test_primetime_dog_flag() -> None:
    flag = screen.build_claims(_fixture_frame())["public_claim_primetime_dog"]["flag"]
    assert flag["primetime_dog_pos_thu"]
    assert flag["primetime_dog_pos_sunday_night"]
    assert not flag["primetime_dog_neg_thu_favorite"]
    assert not flag["primetime_dog_neg_saturday_night"]
    assert not flag["primetime_dog_neg_sunday_day"]


def test_post_bye_back_flag_and_eligibility() -> None:
    spec = screen.build_claims(_fixture_frame())["public_claim_post_bye_back"]
    flag, eligible = spec["flag"], spec["eligible"]
    assert flag["post_bye_pos"]
    assert flag["post_bye_pos_boundary"]  # >=12 is inclusive
    assert not flag["post_bye_neg"]
    assert not eligible["post_bye_missing"], "NaN own_rest must be excluded, not defaulted"
    assert eligible["post_bye_pos"]


def test_division_dog_flag() -> None:
    flag = screen.build_claims(_fixture_frame())["public_claim_division_dog"]["flag"]
    assert flag["division_dog_pos"]
    assert not flag["division_dog_neg_favorite"]
    assert not flag["division_dog_neg_nondiv"]


def test_road_favorite_big_fade_flag() -> None:
    flag = screen.build_claims(_fixture_frame())["public_claim_road_fav_big_fade"]["flag"]
    assert flag["road_fav_pos_boundary"]  # >=7 is inclusive
    assert not flag["road_fav_neg_below_threshold"]
    assert not flag["road_fav_neg_home_favorite"], "must require road, not just big favorite"


def test_home_dog_3plus_flag() -> None:
    flag = screen.build_claims(_fixture_frame())["public_claim_home_dog_3plus"]["flag"]
    assert flag["home_dog_pos_boundary"]  # <=-3 is inclusive
    assert not flag["home_dog_neg_above_threshold"]
    assert not flag["home_dog_neg_road_dog"], "must require home, not just a 3+ point dog"


def test_upset_letdown_fade_flag() -> None:
    flag = screen.build_claims(_fixture_frame())["public_claim_upset_letdown_fade"]["flag"]
    assert flag["upset_letdown_pos"]
    assert not flag["upset_letdown_neg_was_favorite"], "must have been an underdog last week"
    assert not flag["upset_letdown_neg_lost"], "must have WON last week"
    assert not flag["upset_letdown_neg_no_prior"], "no prior game must not default to True"


def test_blowout_loss_bounce_21_flag() -> None:
    flag = screen.build_claims(_fixture_frame())["public_claim_blowout_loss_bounce_21"]["flag"]
    assert flag["blowout_bounce_pos"]
    assert flag["blowout_bounce_pos_boundary"]  # <=-21 is inclusive
    assert not flag["blowout_bounce_neg_smaller_loss"]


def test_week1_dog_flag() -> None:
    flag = screen.build_claims(_fixture_frame())["public_claim_week1_dog"]["flag"]
    assert flag["week1_dog_pos"]
    assert not flag["week1_dog_neg_favorite"]
    assert not flag["week1_dog_neg_week2"]


def test_eliminated_fade_flag_and_eligibility() -> None:
    spec = screen.build_claims(_fixture_frame())["public_claim_eliminated_fade_wk17_18"]
    flag, eligible = spec["flag"], spec["eligible"]
    assert eligible["eliminated_pos"] and flag["eliminated_pos"]
    assert not eligible["eliminated_neg_thin_own_record"], (
        "must require a reliable own-record sample size"
    )
    assert eligible["eliminated_neg_not_bad_enough"] and not flag["eliminated_neg_not_bad_enough"]
    assert not eligible["eliminated_neg_wrong_week"], "must restrict to weeks 17-18"


def test_ats_streak_regress_flag() -> None:
    flag = screen.build_claims(_fixture_frame())["public_claim_ats_streak_regress"]["flag"]
    assert flag["ats_streak_pos"]
    assert flag["ats_streak_pos_longer"]
    assert not flag["ats_streak_neg_below_threshold"]


def test_build_claims_returns_exactly_ten_fresh_cells() -> None:
    claims = screen.build_claims(_fixture_frame())
    assert len(claims) == 10
    for spec in claims.values():
        assert "flag" in spec and "sign" in spec and spec["sign"] in (1, -1)
        assert "mechanism_class" in spec and "description" in spec


def test_no_flag_depends_on_current_game_outcome() -> None:
    """Leakage assertion: shuffling ONLY the current-game outcome columns
    (``team_covered``, ``team_score_margin``) must never change any flag or
    eligibility mask, because a predeclared claim is only allowed to read
    pregame market/schedule facts or strictly-prior derived columns.
    """

    frame = _fixture_frame()
    baseline = screen.build_claims(frame)

    shuffled = frame.copy()
    rng = np.random.default_rng(20260905)
    shuffled["team_covered"] = rng.permutation(shuffled["team_covered"].to_numpy())
    shuffled["team_score_margin"] = rng.permutation(shuffled["team_score_margin"].to_numpy())
    reshuffled = screen.build_claims(shuffled)

    assert set(baseline) == set(reshuffled)
    for name in baseline:
        pd.testing.assert_series_equal(
            baseline[name]["flag"], reshuffled[name]["flag"], check_names=False
        )
        if "eligible" in baseline[name]:
            pd.testing.assert_series_equal(
                baseline[name]["eligible"], reshuffled[name]["eligible"], check_names=False
            )


def _team_season_sequence() -> pd.DataFrame:
    """One team, one season, 4 strictly-ordered games: win, loss, loss, win.
    Hand-computed expectations for ``ats_streak_len``/``prior_team_spread``
    live in the test bodies below.
    """

    return (
        pd.DataFrame(
            {
                "team": ["AAA"] * 4,
                "season": [2023] * 4,
                "gameday": pd.to_datetime(["2023-09-10", "2023-09-17", "2023-09-24", "2023-10-01"]),
                "team_spread": [3.0, -2.0, -1.0, 5.0],
                "team_covered": [1.0, 0.0, 0.0, 1.0],
            }
        )
        .sort_values(["team", "season", "gameday"])
        .reset_index(drop=True)
    )


def test_ats_streak_len_is_strictly_prior_and_resets_on_a_cover() -> None:
    df = screen.add_claim_history_features(_team_season_sequence())
    streaks = df.set_index("gameday")["ats_streak_len"]
    assert streaks["2023-09-10"] == 0.0, "no prior games this season"
    assert streaks["2023-09-17"] == 0.0, "entering game 2: game 1 was a cover, streak reset"
    assert streaks["2023-09-24"] == 1.0, "entering game 3: one prior loss"
    assert streaks["2023-10-01"] == 2.0, "entering game 4: two consecutive prior losses"


def test_prior_team_spread_is_shifted_not_current() -> None:
    df = screen.add_claim_history_features(_team_season_sequence())
    prior = df.set_index("gameday")["prior_team_spread"]
    assert math.isnan(prior["2023-09-10"])
    assert prior["2023-09-17"] == 3.0
    assert prior["2023-09-24"] == -2.0
    assert prior["2023-10-01"] == -1.0


def test_ats_streak_resets_across_a_season_boundary() -> None:
    two_seasons = pd.DataFrame(
        {
            "team": ["AAA"] * 3,
            "season": [2023, 2023, 2024],
            "gameday": pd.to_datetime(["2023-09-10", "2023-09-17", "2024-09-08"]),
            "team_spread": [1.0, -1.0, 1.0],
            "team_covered": [0.0, 0.0, 0.0],
        }
    )
    df = screen.add_claim_history_features(two_seasons)
    streaks = df.set_index("gameday")["ats_streak_len"]
    assert streaks["2023-09-10"] == 0.0
    assert streaks["2023-09-17"] == 1.0
    assert streaks["2024-09-08"] == 0.0, "streak must reset at the season boundary"
