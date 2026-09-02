"""The market-microstructure reliability sweep's builders and its split arithmetic.

Two guarantees, both of which a silent drift would break without any test
failing anywhere else:

1. ``scripts/reliability_market_micro.py`` takes each cell's parent quantity
   from the SCREEN that built the cell, not from a lookalike re-derivation. If
   the two ever disagree, the registry would carry a reliability belonging to a
   different construct than the effect recorded beside it. Both assertions
   below compare against the screen's own builder on a fixture where the
   answer is also computable by hand.
2. ``scripts/reliability_lib.measure_reliability`` does the split arithmetic
   the sweep claims -- and returns an UNMEASURED status rather than a number
   when there are too few units to split. A NaN written through as a number
   would manufacture the appearance of a ``no_split_half_reliability`` closing
   ground out of nothing, which AGENTS.md's taxonomy forbids.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "scripts") not in sys.path:
    sys.path.append(str(REPO / "scripts"))

import odds_microstructure_battery as omb  # noqa: E402
import reliability_lib as rlib  # noqa: E402
import reliability_market_micro as sweep  # noqa: E402
import sagarin_divergence_battery as sag  # noqa: E402

# ---------------------------------------------------------------------------
# 1. The parent quantities come from the screens' own builders
# ---------------------------------------------------------------------------


def _quote(
    game: str,
    book: str,
    side: str,
    price: float,
    observed: str,
    commence: str = "2023-09-10T17:00Z",
) -> dict[str, object]:
    return {
        "market": "spreads",
        "nflverse_game_id": game,
        "bookmaker_key": book,
        "outcome_side": side,
        "price": price,
        "observed_at_utc": pd.Timestamp(observed),
        "commence_time_utc": pd.Timestamp(commence),
    }


def test_price_dispersion_reproduces_the_screens_own_book_level_prices() -> None:
    """``price_std`` must be the screen's book-level price spread, not a lookalike.

    The fixture carries the two things that make a hand-rolled version differ:
    a STALE duplicate quote from a book that later re-posted (the screen keeps
    only the latest per book) and a POST-kickoff quote (the screen drops
    anything not strictly pregame). Getting the same number by accident is not
    possible here -- including either row changes the standard deviation.
    """

    quotes = pd.DataFrame(
        [
            _quote("G1", "bookA", "HOME", -105.0, "2023-09-05T12:00Z"),
            _quote("G1", "bookA", "AWAY", -115.0, "2023-09-05T12:00Z"),
            # bookA re-posted later: the -130 above must NOT survive the dedup.
            _quote("G1", "bookA", "HOME", -130.0, "2023-09-04T12:00Z"),
            _quote("G1", "bookB", "HOME", -110.0, "2023-09-05T12:00Z"),
            _quote("G1", "bookB", "AWAY", -110.0, "2023-09-05T12:00Z"),
            _quote("G1", "bookC", "HOME", -120.0, "2023-09-05T12:00Z"),
            _quote("G1", "bookC", "AWAY", -100.0, "2023-09-05T12:00Z"),
            # Posted after kickoff: never pregame, must be dropped entirely.
            _quote("G1", "bookD", "HOME", +200.0, "2023-09-10T19:00Z"),
            _quote("G1", "bookD", "AWAY", -240.0, "2023-09-10T19:00Z"),
        ]
    )

    dispersion = sweep.price_dispersion(quotes)
    assert list(dispersion["game_id"]) == ["G1"]
    assert int(dispersion["price_books"].iloc[0]) == 3

    surviving = omb._book_level_spread_prices(quotes)
    expected = float(surviving.groupby("nflverse_game_id")["home_price"].std().iloc[0])
    assert dispersion["price_std"].iloc[0] == pytest.approx(expected)
    # Hand-computed from the three surviving home prices only.
    assert dispersion["price_std"].iloc[0] == pytest.approx(
        float(np.std([-105.0, -110.0, -120.0], ddof=1))
    )


def test_signed_team_week_carries_the_screens_own_divergence_unchanged() -> None:
    """The home row IS ``sagarin_divergence_battery.add_divergence``'s column.

    The sweep's only transformation is the sign convention: the home row
    carries the builder's home-positive value verbatim and the away row
    carries its negation. Anything else -- a rescale, a re-derivation of
    ``home_rating - away_rating + home_edge_rating`` -- would break this.
    """

    games = pd.DataFrame(
        {
            "season": [2020, 2020, 2021],
            "week": [1, 2, 1],
            "home_team": ["AAA", "BBB", "AAA"],
            "away_team": ["CCC", "DDD", "BBB"],
            "home_rating": [80.0, 70.0, 75.0],
            "away_rating": [75.0, 72.0, 70.0],
            "home_edge_rating": [2.5, 2.5, 2.0],
            "spread_line": [-3.0, 1.0, -6.5],
        }
    )
    built = sag.add_divergence(games, market_col="spread_line", out_col="divergence_close")
    long = sweep.signed_team_week(built, "divergence_close", metric="sagarin_divergence_close")

    home_rows = long.iloc[: len(games)]
    away_rows = long.iloc[len(games) :]
    pd.testing.assert_series_equal(
        home_rows["sagarin_divergence_close"].reset_index(drop=True),
        built["divergence_close"].reset_index(drop=True),
        check_names=False,
    )
    pd.testing.assert_series_equal(
        away_rows["sagarin_divergence_close"].reset_index(drop=True),
        (-built["divergence_close"]).reset_index(drop=True),
        check_names=False,
    )
    assert list(home_rows["team_id"]) == list(games["home_team"])
    assert list(away_rows["team_id"]) == list(games["away_team"])


def test_per_side_frame_keeps_each_sides_own_percentage() -> None:
    """Public bet% already exists per side, so a row is never the other's negation."""

    games = pd.DataFrame(
        {
            "season": [2022],
            "week": [3],
            "home_team": ["AAA"],
            "away_team": ["BBB"],
            "spread_home_bet_pct": [72.0],
            "spread_away_bet_pct": [28.0],
        }
    )
    long = sweep.per_side_team_week(
        games, "spread_home_bet_pct", "spread_away_bet_pct", metric="public_bet_pct"
    )
    assert long.loc[long["team_id"] == "AAA", "public_bet_pct"].iloc[0] == pytest.approx(72.0)
    assert long.loc[long["team_id"] == "BBB", "public_bet_pct"].iloc[0] == pytest.approx(28.0)


def test_every_registry_cell_maps_to_a_declared_parent_quantity() -> None:
    """No cell may be measured against a quantity nobody wrote down."""

    assert len(sweep.CELL_TABLE) == 26
    for entry, (battery, frame_key, _oracle, secondaries) in sweep.CELL_TABLE.items():
        assert battery in sweep.BUILDER_FOR_BATTERY, entry
        if frame_key is not None:
            assert frame_key in sweep.QUANTITIES, entry
        for key in secondaries:
            assert key in sweep.QUANTITIES, entry


def test_the_three_oracle_controls_are_flagged_as_ceilings() -> None:
    """A leaked-oracle cell must never read as a playable rule."""

    oracle_entries = {entry for entry, (_b, _k, oracle, _s) in sweep.CELL_TABLE.items() if oracle}
    assert oracle_entries == {
        "odds_microstructure_H3_3_0a_full_week_oracle_2020_2025_sanity_check",
        "odds_microstructure_H3_3_0b_full_week_oracle_2023_2025_baseline",
        "odds_microstructure_H3_3_1_tue_to_wed_oracle_2023_2025",
    }
    assert "ceiling by" in sweep.ORACLE_CAVEAT


# ---------------------------------------------------------------------------
# 2. The split arithmetic, on an answer computable by hand
# ---------------------------------------------------------------------------


def _long_frame(values: dict[tuple[str, int], list[float]], metric: str) -> pd.DataFrame:
    rows = []
    for (team, season), series in values.items():
        for index, value in enumerate(series, start=1):
            rows.append({"team_id": team, "season": season, "week": index, metric: value})
    return pd.DataFrame(rows)


def test_recovers_a_hand_computed_correlation_and_its_spearman_brown_step_up() -> None:
    # Weeks 1..4: the odd half sees weeks 1 and 3, the even half weeks 2 and 4,
    # so each team-season's two half-means are set directly and the Pearson r
    # between them is computable by hand.
    odd_means = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    even_means = [1.0, 3.0, 2.0, 5.0, 4.0, 6.0]
    values = {
        f"T{index}": [odd, even, odd, even]
        for index, (odd, even) in enumerate(zip(odd_means, even_means, strict=True))
    }
    long = _long_frame({(team, 2020): series for team, series in values.items()}, "juice_lean")

    expected_r = float(np.corrcoef(odd_means, even_means)[0, 1])
    expected_sb = 2.0 * expected_r / (1.0 + expected_r)

    result = rlib.measure_reliability(
        long, "juice_lean", method=rlib.METHOD_TRAIT, n_boot=200, min_units=3
    )
    assert result["status"] == rlib.STATUS_MEASURED
    assert result["n_units"] == len(odd_means)
    assert result["pearson_r"] == pytest.approx(expected_r, abs=1e-9)
    assert result["reliability"] == pytest.approx(expected_sb, abs=1e-9)
    assert -1.0 <= result["reliability_low"] <= result["reliability"] <= result["reliability_high"]


def test_a_short_window_is_reported_unmeasured_never_as_zero() -> None:
    """Several market_micro cells own two- to four-season windows; a thin one
    must come back UNMEASURED, not as a reliability of zero."""

    long = _long_frame(
        {("T0", 2020): [1.0, 2.0, 3.0, 4.0], ("T1", 2020): [2.0, 1.0, 4.0, 3.0]}, "sbr_open_line"
    )
    result = rlib.measure_reliability(long, "sbr_open_line", method=rlib.METHOD_TRAIT, n_boot=100)

    assert result["status"] != rlib.STATUS_MEASURED
    assert result["status"] == rlib.STATUS_INSUFFICIENT_UNITS
    assert result["reliability"] is None
    assert result["reliability_low"] is None and result["reliability_high"] is None


def test_a_near_constant_column_is_flagged_rather_than_recorded() -> None:
    """A column with no cross-unit spread returns a correlation that means nothing."""

    long = _long_frame({(f"T{i}", 2020): [7.0] * 4 for i in range(30)}, "spread_hold")
    report = sweep.near_constant_report(long, "spread_hold", seasons=(2020, 2020))

    assert report["near_constant"] is True
    assert report["n_distinct_values"] == 1

    varied = _long_frame(
        {(f"T{i}", 2020): [float(i), float(i) + 1.0, float(i), float(i) + 1.0] for i in range(30)},
        "spread_hold",
    )
    assert (
        sweep.near_constant_report(varied, "spread_hold", seasons=(2020, 2020))["near_constant"]
        is False
    )
