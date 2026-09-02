"""Tests for ``scripts/era_mechanism_screens.py`` (WP34, 2026-09-01).

Four things are pinned, matching section 3.3 of the predeclaration
``docs/era_mechanism_screens_20260901.md``:

1. the imported flag builders reproduce their parent batteries' flags on a
   fixture (bye base flag, post-MNF-Sunday, large divergence),
2. the reused changepoint machinery gives the known answer on a synthetic
   series with a planted break,
3. the coverage-matched population construction keeps exactly the seasons at
   or above the frozen threshold and drops the rest,
4. leakage: the install-need moderator reads only each team's season opener,
   so no game's own result can feed its own moderator value.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]


def _load_script(name: str, filename: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, REPO / "scripts" / filename)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def screens() -> Any:
    """Import the screens module, skipping if this clone has no local snapshots.

    Several of the parent batteries resolve their newest ``data/raw/*`` snapshot
    at import time, so a fresh clone without generated data cannot import them.
    """

    try:
        return _load_script("era_mechanism_screens_test", "era_mechanism_screens.py")
    except FileNotFoundError as error:  # pragma: no cover - depends on local data
        pytest.skip(f"local data snapshot required to import the screens module: {error}")


# ---------------------------------------------------------------------------
# 1. imported flag builders reproduce the parent batteries' flags
# ---------------------------------------------------------------------------


def _bye_fixture() -> pd.DataFrame:
    gamedays = [
        "2020-09-13",  # g1 openers, no prior game for anyone
        "2020-09-20",  # g2
        "2020-09-27",  # g3
        "2020-10-04",  # g4
        "2020-10-11",  # g5 BBB off a 28-day gap, AAA on 7 days
        "2020-10-25",  # g6 CCC and BBB BOTH off long gaps
    ]
    return pd.DataFrame(
        {
            "game_id": [f"g{i}" for i in range(1, 7)],
            "season": [2020] * 6,
            "home_team": ["BBB", "CCC", "AAA", "AAA", "BBB", "CCC"],
            "away_team": ["AAA", "AAA", "CCC", "CCC", "AAA", "BBB"],
            "gameday_dt": pd.to_datetime(gamedays),
        }
    )


def test_bye_base_flag_matches_the_battery_maps(screens: Any) -> None:
    population = _bye_fixture()

    flag = screens.bye_base_flag(population)

    home_pb, away_pb = screens.bye_battery.build_bye_maps(population)
    expected = home_pb.to_numpy(dtype=bool) & ~away_pb.to_numpy(dtype=bool)
    assert list(flag.to_numpy(dtype=bool)) == list(expected)
    # Only g5 has the home side off a strict bye while the opponent is not.
    assert list(flag.to_numpy(dtype=bool)) == [False, False, False, False, True, False]


def test_bye_base_flag_uses_the_batterys_own_gap_threshold(screens: Any) -> None:
    assert screens.bye_battery.POST_BYE_GAP_DAYS == 12
    assert screens.bye_battery.ERA_POST_MIN_SEASON == 2012


def _primetime_schedule_fixture(path: Path) -> Path:
    frame = pd.DataFrame(
        {
            "game_id": ["g1", "g2", "g3", "g4"],
            "season": [2020, 2020, 2020, 2020],
            "week": [1, 2, 3, 3],
            "game_type": ["REG"] * 4,
            "gameday": ["2020-09-13", "2020-09-21", "2020-09-27", "2020-09-27"],
            "weekday": ["Sunday", "Monday", "Sunday", "Sunday"],
            "gametime": ["13:00", "20:15", "13:00", "13:00"],
            "home_team": ["NE", "NE", "HOU", "KC"],
            "away_team": ["KC", "BUF", "NE", "BUF"],
            "result": [3.0, 7.0, -4.0, 6.0],
            "spread_line": [1.5, 2.5, 1.0, 3.0],
            "div_game": [0, 0, 0, 0],
            "location": ["Home"] * 4,
            "away_rest": [7, 7, 7, 7],
            "home_rest": [7, 7, 7, 7],
        }
    )
    destination = path / "schedules.parquet"
    frame.to_parquet(destination)
    return destination


def test_post_mnf_flag_matches_the_primetime_battery(screens: Any, tmp_path: Path) -> None:
    schedules = _primetime_schedule_fixture(tmp_path)

    frame, diagnostics = screens.build_post_mnf_frame(schedules)

    # Eligible = rows with a strictly prior game this season: NE(g2), NE(g3),
    # BUF(g4), KC(g4). Flagged = the two whose own prior game was a Monday.
    assert diagnostics["n_eligible_rows"] == 4
    assert diagnostics["n_flag"] == 2
    assert diagnostics["sign_convention"] == -1
    assert set(frame.columns) == {"season", "week", "team_covered", "flag", "season_idx"}
    assert frame["flag"].sum() == 2


def test_large_divergence_rows_use_the_batterys_threshold(screens: Any) -> None:
    close_pop = pd.DataFrame(
        {
            "season": [2015] * 5,
            "week": [1, 2, 3, 4, 5],
            "divergence_close": [-3.0, -2.99, 0.5, 3.0, 4.0],
            "home_cover": [1.0, 0.0, 1.0, 0.0, 1.0],
            "sagarin_side_cover": [0.0, 1.0, 1.0, 1.0, 1.0],
        }
    )

    rows = screens.large_divergence_rows(close_pop)

    assert screens.sagarin_battery.LARGE_DIVERGENCE_THRESHOLD == 3.0
    assert list(rows["divergence_close"]) == [-3.0, 3.0, 4.0]
    assert list(rows["sagarin_side_home"]) == [False, True, True]


# ---------------------------------------------------------------------------
# 2. changepoint machinery: known answer
# ---------------------------------------------------------------------------


def test_changepoint_finds_a_planted_break(screens: Any) -> None:
    seasons = list(range(2009, 2026))
    series = np.array([0.0] * 8 + [5.0] * 9)

    summary = screens._changepoint_summary(series, seasons)

    assert summary["break_index"] == 8
    assert summary["break_season"] == 2017
    assert summary["pre_break_mean"] == pytest.approx(0.0)
    assert summary["post_break_mean"] == pytest.approx(5.0)
    assert summary["break_magnitude_post_minus_pre"] == pytest.approx(5.0)
    assert summary["sse"] == pytest.approx(0.0)


def test_changepoint_respects_the_minimum_segment_length(screens: Any) -> None:
    seasons = list(range(2009, 2026))
    # The true jump is after the FIRST season, inside the 3-season minimum.
    series = np.array([9.0] + [0.0] * 16)

    summary = screens._changepoint_summary(series, seasons)

    assert screens.era_profile.MIN_SEGMENT_SEASONS == 3
    assert summary["break_index"] >= 3
    assert summary["break_index"] <= len(seasons) - 3


# ---------------------------------------------------------------------------
# 3. coverage-matched population construction
# ---------------------------------------------------------------------------


def test_coverage_matching_keeps_exactly_the_seasons_at_or_above_threshold(
    screens: Any,
) -> None:
    coverage = pd.DataFrame(
        {
            "season": [2010, 2011, 2012, 2013, 2014, 2017, 2018],
            "coverage_pct": [91.8, 89.5, 47.3, 85.5, 80.0, 96.9, 78.1],
        }
    )

    era_seasons, kept, dropped = screens.coverage_matched_seasons(
        coverage, season_lo=2010, season_hi=2016, threshold_pct=80.0
    )

    assert era_seasons == [2010, 2011, 2012, 2013, 2014]
    # 80.0 is at the threshold and is KEPT (>=, not >).
    assert kept == [2010, 2011, 2013, 2014]
    assert dropped == [2012]

    late_seasons, late_kept, late_dropped = screens.coverage_matched_seasons(
        coverage, season_lo=2017, season_hi=2025, threshold_pct=80.0
    )
    assert late_seasons == [2017, 2018]
    assert late_kept == [2017]
    assert late_dropped == [2018]


def test_coverage_threshold_is_the_frozen_eighty_percent(screens: Any) -> None:
    assert screens.COVERAGE_MATCH_THRESHOLD_PCT == 80.0


# ---------------------------------------------------------------------------
# 4. leakage: the moderator reads only each team's season opener
# ---------------------------------------------------------------------------


def _plays_fixture(late_passer: str) -> pd.DataFrame:
    """One team whose first REG game is week 2, plus later-week plays."""

    rows = []
    # Week 2 opener (week 1 postponed): P1 throws 3, P2 throws 1.
    for passer, count in (("P1", 3), ("P2", 1)):
        rows.extend(
            {
                "season": 2020,
                "season_type": "REG",
                "week": 2,
                "posteam": "AAA",
                "passer_player_id": passer,
                "qb_dropback": 1,
            }
            for _ in range(count)
        )
    # Week 5: a different passer dominates. Must not affect the answer.
    rows.extend(
        {
            "season": 2020,
            "season_type": "REG",
            "week": 5,
            "posteam": "AAA",
            "passer_player_id": late_passer,
            "qb_dropback": 1,
        }
        for _ in range(50)
    )
    return pd.DataFrame(rows)


def test_opener_starter_reads_only_the_first_game(screens: Any) -> None:
    first = screens.opener_starters_from_plays(_plays_fixture("P2"))
    second = screens.opener_starters_from_plays(_plays_fixture("P9"))

    assert list(first["passer_player_id"]) == ["P1"]
    assert list(first["dropbacks"]) == [3]
    # Perturbing every later week leaves the moderator input untouched: no
    # game's own week can feed the moderator value it is scored against.
    assert list(second["passer_player_id"]) == ["P1"]
    assert first.equals(second)


def test_install_need_needs_an_immediately_prior_season(screens: Any) -> None:
    starters = pd.DataFrame(
        {
            "season": [2019, 2020, 2021, 2019, 2021],
            "team": ["AAA", "AAA", "AAA", "BBB", "BBB"],
            "starter_id": ["X", "X", "Y", "Z", "Z"],
        }
    )

    moderator = screens.install_need_from_starters(starters)
    lookup = {
        (int(row["season"]), row["team"]): row["install_need"] for _, row in moderator.iterrows()
    }

    assert np.isnan(lookup[(2019, "AAA")])  # no observed prior season
    assert lookup[(2020, "AAA")] is np.False_ or lookup[(2020, "AAA")] == 0.0
    assert lookup[(2021, "AAA")] == 1.0  # opener QB changed X -> Y
    assert np.isnan(lookup[(2019, "BBB")])
    assert np.isnan(lookup[(2021, "BBB")])  # 2020 missing: gap, not a prior season


# ---------------------------------------------------------------------------
# statistic arithmetic (the recorded number itself)
# ---------------------------------------------------------------------------


def test_bye_contrast_statistic_arithmetic(screens: Any) -> None:
    frame = pd.DataFrame(
        {
            "season": [2020] * 8,
            "week": [1, 1, 2, 2, 3, 3, 4, 4],
            "home_cover": [1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0],
            "base_flag": [True, False, False, False, True, False, False, False],
            "install_need": [True, True, True, True, False, False, False, False],
        }
    )

    values = screens.bye_contrast_statistic(frame, frame["install_need"].to_numpy(dtype=bool))

    # install-need arm: flagged cover 1.0, complement cover 1/3, slate share 1/4
    assert values["effect_install_need"] == pytest.approx((1.0 - 1.0 / 3.0) * 100.0 * 0.25)
    # no-need arm: flagged cover 1.0, complement cover 1/3, slate share 1/4
    assert values["effect_no_need"] == pytest.approx((1.0 - 1.0 / 3.0) * 100.0 * 0.25)
    assert values["contrast_install_minus_no_need"] == pytest.approx(0.0)


def test_within_week_permutation_returns_the_observed_and_the_requested_draws(
    screens: Any,
) -> None:
    frame = pd.DataFrame(
        {
            "season": [2020] * 8,
            "week": [1, 1, 1, 1, 2, 2, 2, 2],
            "home_cover": [1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0],
            "base_flag": [True, False, True, False, True, False, True, False],
            "install_need": [True, True, False, False, True, True, False, False],
        }
    )

    summary = screens._within_week_permutation(
        frame,
        label_col="install_need",
        statistic=screens.bye_contrast_statistic,
        draws=25,
        seed=1,
    )

    observed = screens.bye_contrast_statistic(frame, frame["install_need"].to_numpy(dtype=bool))
    entry = summary["contrast_install_minus_no_need"]
    assert entry["observed"] == pytest.approx(observed["contrast_install_minus_no_need"])
    assert entry["requested_draws"] == 25
    assert 1 <= entry["draws"] <= 25
    assert 0.0 <= entry["share_null_at_or_above_observed"] <= 1.0
