from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nfl_ats import novig
from nfl_ats.data import DataContractError
from nfl_ats.novig import (
    bootstrap_calibration_gaps,
    calibration_bucket_edges,
    calibration_gap_metric_fn,
    favourite_longshot_calibration,
    moneyline_novig_probabilities,
    spread_novig_probabilities,
)
from nfl_ats.odds import market_hold, no_vig_probabilities

# ---------------------------------------------------------------------------
# spread_novig_probabilities / moneyline_novig_probabilities
# ---------------------------------------------------------------------------


def test_spread_novig_probabilities_matches_odds_module_row_by_row() -> None:
    frame = pd.DataFrame(
        {
            "game_id": ["G1", "G2", "G3"],
            "home_spread_price": [-110, -120, 105],
            "away_spread_price": [-110, 100, -125],
        }
    )
    result = spread_novig_probabilities(frame)
    for index, row in frame.iterrows():
        expected_home, _ = no_vig_probabilities(row["home_spread_price"], row["away_spread_price"])
        expected_hold = market_hold(row["home_spread_price"], row["away_spread_price"])
        assert result.loc[index, "no_vig_home_cover_probability"] == pytest.approx(expected_home)
        assert result.loc[index, "spread_hold"] == pytest.approx(expected_hold)
    # Original columns pass through unchanged.
    assert result["game_id"].tolist() == ["G1", "G2", "G3"]


def test_spread_novig_probabilities_missing_price_is_nan_not_minus_110_fallback() -> None:
    frame = pd.DataFrame(
        {
            "home_spread_price": pd.Series([-110, pd.NA, 0.0], dtype="Float64"),
            "away_spread_price": pd.Series([pd.NA, -110, -110], dtype="Float64"),
        }
    )
    result = spread_novig_probabilities(frame)
    assert result["no_vig_home_cover_probability"].isna().all()
    assert result["spread_hold"].isna().all()


def test_spread_novig_probabilities_requires_columns() -> None:
    with pytest.raises(DataContractError, match="missing columns"):
        spread_novig_probabilities(pd.DataFrame({"home_spread_price": [-110]}))


@pytest.mark.parametrize("invalid", [np.inf, "not-a-price"])
def test_spread_novig_probabilities_rejects_malformed_prices(invalid: object) -> None:
    with pytest.raises(DataContractError, match="finite numeric odds"):
        spread_novig_probabilities(
            pd.DataFrame({"home_spread_price": [invalid], "away_spread_price": [-110]})
        )


def test_moneyline_novig_probabilities_matches_odds_module() -> None:
    frame = pd.DataFrame({"home_moneyline": [-150, 130], "away_moneyline": [130, -150]})
    result = moneyline_novig_probabilities(frame)
    expected_home_0, _ = no_vig_probabilities(-150, 130)
    expected_home_1, _ = no_vig_probabilities(130, -150)
    assert result["no_vig_home_win_probability"].tolist() == pytest.approx(
        [expected_home_0, expected_home_1]
    )
    assert result["moneyline_hold"].iloc[0] == pytest.approx(market_hold(-150, 130))


def test_moneyline_novig_probabilities_requires_columns() -> None:
    with pytest.raises(DataContractError, match="missing columns"):
        moneyline_novig_probabilities(pd.DataFrame({"home_moneyline": [-150]}))


def test_moneyline_novig_probabilities_rejects_malformed_prices() -> None:
    with pytest.raises(DataContractError, match="finite numeric odds"):
        moneyline_novig_probabilities(
            pd.DataFrame({"home_moneyline": [-150], "away_moneyline": [np.inf]})
        )


# ---------------------------------------------------------------------------
# calibration_bucket_edges
# ---------------------------------------------------------------------------


def test_calibration_bucket_edges_widens_outer_edges_to_infinity() -> None:
    edges = calibration_bucket_edges(pd.Series([0.1, 0.3, 0.5, 0.7, 0.9]), buckets=5)
    assert edges[0] == -np.inf
    assert edges[-1] == np.inf


def test_calibration_bucket_edges_rejects_no_variation() -> None:
    with pytest.raises(ValueError, match="too little variation"):
        calibration_bucket_edges(pd.Series([0.5, 0.5, 0.5]))


def test_calibration_bucket_edges_rejects_empty() -> None:
    with pytest.raises(ValueError, match="No non-null probabilities"):
        calibration_bucket_edges(pd.Series([np.nan, np.nan]))


@pytest.mark.parametrize("buckets", [0, -1, True, 1.5])
def test_calibration_bucket_edges_requires_positive_integer_bucket_count(
    buckets: object,
) -> None:
    with pytest.raises(ValueError, match="positive integer"):
        calibration_bucket_edges(pd.Series([0.2, 0.8]), buckets=buckets)  # type: ignore[arg-type]


@pytest.mark.parametrize("probability", [-0.01, 1.01, np.inf, "not-a-number"])
def test_calibration_bucket_edges_rejects_invalid_probabilities(probability: object) -> None:
    with pytest.raises((DataContractError, ValueError), match=r"probabilit|non-numeric"):
        calibration_bucket_edges(pd.Series([0.2, 0.8, probability]))


# ---------------------------------------------------------------------------
# favourite_longshot_calibration -- synthetic frame with a known injected bias
# ---------------------------------------------------------------------------


def _injected_bias_frame(weeks: int = 4, season: int = 2024) -> pd.DataFrame:
    """Every week: two 0.3-probability rows (outcomes 0,1) and two 0.7-probability
    rows (outcomes 0,1) -- both buckets have a TRUE observed frequency of 0.5,
    so the injected miscalibration is exactly +0.2 for the low bucket and
    -0.2 for the high bucket, identically in every week block.
    """

    rows = []
    for week in range(1, weeks + 1):
        rows.extend(
            [
                {"season": season, "week": week, "probability": 0.3, "outcome": 0.0},
                {"season": season, "week": week, "probability": 0.3, "outcome": 1.0},
                {"season": season, "week": week, "probability": 0.7, "outcome": 0.0},
                {"season": season, "week": week, "probability": 0.7, "outcome": 1.0},
            ]
        )
    return pd.DataFrame(rows)


def test_favourite_longshot_calibration_recovers_injected_bias() -> None:
    frame = _injected_bias_frame()
    table = favourite_longshot_calibration(frame, "probability", "outcome", buckets=2)
    assert len(table) == 2
    low = table.loc[table["mean_predicted_probability"].round(1).eq(0.3)].iloc[0]
    high = table.loc[table["mean_predicted_probability"].round(1).eq(0.7)].iloc[0]
    assert low["n"] == 8
    assert high["n"] == 8
    assert low["mean_observed_frequency"] == pytest.approx(0.5)
    assert low["calibration_gap"] == pytest.approx(0.2)
    assert high["mean_observed_frequency"] == pytest.approx(0.5)
    assert high["calibration_gap"] == pytest.approx(-0.2)
    assert low["brier_component"] == pytest.approx(0.04)
    assert high["brier_component"] == pytest.approx(0.04)
    assert low["bucket_weight"] == pytest.approx(0.5)
    assert high["bucket_weight"] == pytest.approx(0.5)
    assert table["brier_reliability_contribution"].sum() == pytest.approx(0.04)


def test_favourite_longshot_calibration_excludes_pushes_and_missing_probability() -> None:
    frame = pd.DataFrame(
        {
            "probability": [0.3, 0.3, 0.7, np.nan],
            "outcome": [1.0, np.nan, 0.0, 1.0],
        }
    )
    table = favourite_longshot_calibration(frame, "probability", "outcome", buckets=2)
    # Only the (0.3, 1.0) and (0.7, 0.0) rows survive the NaN/push exclusion.
    assert int(table["n"].sum()) == 2


def test_favourite_longshot_calibration_requires_columns() -> None:
    with pytest.raises(DataContractError, match="missing columns"):
        favourite_longshot_calibration(
            pd.DataFrame({"probability": [0.5]}), "probability", "outcome"
        )


def test_favourite_longshot_calibration_raises_when_nothing_survives() -> None:
    frame = pd.DataFrame({"probability": [np.nan], "outcome": [np.nan]})
    with pytest.raises(ValueError, match="No rows with both"):
        favourite_longshot_calibration(frame, "probability", "outcome")


@pytest.mark.parametrize("outcome", [-1.0, 0.5, 2.0, np.inf, "not-a-number"])
def test_favourite_longshot_calibration_rejects_non_binary_outcomes(outcome: object) -> None:
    frame = pd.DataFrame({"probability": [0.2, 0.8], "outcome": [0.0, outcome]})
    with pytest.raises(DataContractError, match=r"binary|non-numeric"):
        favourite_longshot_calibration(frame, "probability", "outcome")


@pytest.mark.parametrize(
    "edges",
    [
        np.array([0.0, 0.5, np.inf]),
        np.array([-np.inf, 0.5, 0.5, np.inf]),
        np.array([-np.inf, np.nan, np.inf]),
        np.array([-np.inf, 1.1, np.inf]),
    ],
)
def test_favourite_longshot_calibration_rejects_unsafe_explicit_edges(
    edges: np.ndarray,
) -> None:
    with pytest.raises(ValueError, match="edges"):
        favourite_longshot_calibration(
            _injected_bias_frame(), "probability", "outcome", edges=edges
        )


def test_explicit_precomputed_edges_are_never_reestimated(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    edges = calibration_bucket_edges(pd.Series([0.2, 0.4, 0.6, 0.8]), buckets=2)
    frame = pd.DataFrame({"probability": [0.01, 0.99], "outcome": [0.0, 1.0]})

    def _forbid_reestimate(*_args: object, **_kwargs: object) -> np.ndarray:
        raise AssertionError("fixed calibration edges were recomputed")

    monkeypatch.setattr(novig, "calibration_bucket_edges", _forbid_reestimate)
    table = favourite_longshot_calibration(frame, "probability", "outcome", edges=edges)
    assert table["bucket_lower"].tolist() == [-np.inf, 0.5]
    assert table["bucket_upper"].tolist() == [0.5, np.inf]


# ---------------------------------------------------------------------------
# calibration_gap_metric_fn / bootstrap_calibration_gaps
# ---------------------------------------------------------------------------


def test_calibration_gap_metric_fn_matches_point_estimate() -> None:
    frame = _injected_bias_frame()
    edges = calibration_bucket_edges(frame["probability"], buckets=2)
    metric_fn = calibration_gap_metric_fn("probability", "outcome", edges)
    result = metric_fn(frame)
    assert result["bucket_0_gap"] == pytest.approx(0.2)
    assert result["bucket_1_gap"] == pytest.approx(-0.2)
    assert result["mean_abs_calibration_gap"] == pytest.approx(0.2)
    assert result["expected_calibration_error"] == pytest.approx(0.2)
    assert result["brier_reliability"] == pytest.approx(0.04)


def test_weighted_calibration_outputs_use_bucket_population() -> None:
    frame = pd.DataFrame(
        {
            "probability": [0.2, 0.8, 0.8, 0.8],
            "outcome": [1.0, 1.0, 1.0, 0.0],
        }
    )
    edges = np.array([-np.inf, 0.5, np.inf])
    table = favourite_longshot_calibration(frame, "probability", "outcome", edges=edges)
    metrics = calibration_gap_metric_fn("probability", "outcome", edges)(frame)

    assert table["bucket_weight"].tolist() == pytest.approx([0.25, 0.75])
    assert table["brier_reliability_contribution"].sum() == pytest.approx(0.1733333333)
    assert metrics["expected_calibration_error"] == pytest.approx(0.3)
    assert metrics["brier_reliability"] == pytest.approx(0.1733333333)


def test_bootstrap_calibration_gaps_is_deterministic_when_every_block_is_identical() -> None:
    frame = _injected_bias_frame(weeks=6)
    edges = calibration_bucket_edges(frame["probability"], buckets=2)
    bootstrap = bootstrap_calibration_gaps(
        frame, "probability", "outcome", edges, block="week", samples=50, seed=7
    )
    by_metric = bootstrap.set_index("metric")
    # Every week block has an identical composition, so every resample
    # reproduces the same estimate -- lower == estimate == upper, exactly
    # the pattern nfl_ats.clv's own deterministic bootstrap test uses.
    for metric_name, expected in (("bucket_0_gap", 0.2), ("bucket_1_gap", -0.2)):
        row = by_metric.loc[metric_name]
        assert row["estimate"] == pytest.approx(expected)
        assert row["lower"] == pytest.approx(expected)
        assert row["upper"] == pytest.approx(expected)


def test_bootstrap_calibration_gaps_requires_resolved_rows() -> None:
    frame = pd.DataFrame(
        {"season": [2024], "week": [1], "probability": [np.nan], "outcome": [np.nan]}
    )
    edges = np.array([-np.inf, 0.5, np.inf])
    with pytest.raises(ValueError, match="No rows with both"):
        bootstrap_calibration_gaps(frame, "probability", "outcome", edges, samples=10)


def test_bootstrap_calibration_gaps_requires_named_columns() -> None:
    with pytest.raises(DataContractError, match="missing columns"):
        bootstrap_calibration_gaps(
            pd.DataFrame({"season": [2024], "week": [1], "probability": [0.5]}),
            "probability",
            "outcome",
            np.array([-np.inf, np.inf]),
            samples=10,
        )
