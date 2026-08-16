from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nfl_ats.key_numbers import (
    DEFAULT_KEY_NUMBERS,
    cover_reliability_by_line_bucket,
    implied_key_number_mass,
    line_bucket,
    realized_key_number_frequency,
    summarize_key_number_calibration,
)


def test_implied_key_number_mass_counts_both_directions() -> None:
    # Two games, five samples each. Game 0: two samples at +3, two at -3, one at 0.
    # Game 1: all samples at 10.
    distribution = np.array(
        [
            [3.0, 3.1, -3.0, -2.6, 0.2],
            [10.0, 10.4, 9.6, 10.0, 9.9],
        ]
    )
    mass = implied_key_number_mass(distribution, key_numbers=(3, 7, 10))
    assert list(mass.columns) == ["key_number_3", "key_number_7", "key_number_10"]
    assert mass.loc[0, "key_number_3"] == pytest.approx(0.8)
    assert mass.loc[0, "key_number_7"] == 0.0
    assert mass.loc[1, "key_number_10"] == pytest.approx(1.0)


def test_implied_key_number_mass_requires_2d_array_and_key_numbers() -> None:
    with pytest.raises(ValueError, match="At least one key number"):
        implied_key_number_mass(np.zeros((2, 3)), key_numbers=())
    with pytest.raises(ValueError, match="2-D"):
        implied_key_number_mass(np.zeros(3))


def test_realized_key_number_frequency() -> None:
    results = pd.Series([3.0, -3.0, 7.0, 10.0, np.nan, 3.0])
    frequency = realized_key_number_frequency(results, key_numbers=(3, 7, 10, 14))
    assert frequency[3] == pytest.approx(3 / 5)
    assert frequency[7] == pytest.approx(1 / 5)
    assert frequency[10] == pytest.approx(1 / 5)
    assert frequency[14] == 0.0

    with pytest.raises(ValueError, match="No completed games"):
        realized_key_number_frequency(pd.Series([np.nan, np.nan]))
    with pytest.raises(ValueError, match="At least one key number"):
        realized_key_number_frequency(results, key_numbers=())


def test_summarize_key_number_calibration_reports_gap_per_method() -> None:
    mass = pd.DataFrame(
        {
            "method": ["fair_margin", "fair_margin", "market", "market"],
            "result": [3.0, -3.0, 7.0, 0.0],
            "key_number_3": [0.9, 0.9, 0.0, 0.0],
            "key_number_7": [0.05, 0.05, 0.4, 0.4],
        }
    )
    summary = summarize_key_number_calibration(mass, key_numbers=(3, 7))
    fair_margin_3 = summary.loc[
        summary["method"].eq("fair_margin") & summary["key_number"].eq(3)
    ].iloc[0]
    assert fair_margin_3["implied_mass"] == pytest.approx(0.9)
    assert fair_margin_3["realized_frequency"] == pytest.approx(1.0)
    assert fair_margin_3["gap"] == pytest.approx(-0.1)
    assert fair_margin_3["games"] == 2

    with pytest.raises(ValueError, match="missing columns"):
        summarize_key_number_calibration(mass.drop(columns="result"), key_numbers=(3, 7))


def test_line_bucket_partitions_by_magnitude() -> None:
    lines = pd.Series([1.0, -3.0, 5.0, 7.0, -9.5, -3.5])
    buckets = line_bucket(lines)
    assert buckets.tolist() == [
        "under_3",
        "three",
        "three_five_to_six_five",
        "seven",
        "over_seven",
        "three_five_to_six_five",
    ]


def test_cover_reliability_by_line_bucket_computes_gap_and_drops_pushes() -> None:
    predictions = pd.DataFrame(
        {
            "spread_line": [1.0, 1.0, 3.0, 7.0, -9.0],
            "home_cover_probability": [0.6, 0.4, 0.5, 0.7, np.nan],
            "home_cover": [1.0, 0.0, np.nan, 1.0, 0.0],
        }
    )
    reliability = cover_reliability_by_line_bucket(predictions)
    under_three = reliability.loc[reliability["line_bucket"].eq("under_3")].iloc[0]
    assert under_three["games"] == 2
    assert under_three["mean_predicted_probability"] == pytest.approx(0.5)
    assert under_three["realized_cover_rate"] == pytest.approx(0.5)
    assert under_three["calibration_gap"] == pytest.approx(0.0)
    # the "three" bucket's only row has a null home_cover (push) and a null
    # predicted probability, so it drops out entirely
    assert "three" not in set(reliability["line_bucket"])
    # bucket order follows the key-number line regions, not alphabetical order
    assert reliability["line_bucket"].tolist() == ["under_3", "seven"]


def test_cover_reliability_by_line_bucket_requires_columns_and_nonempty_result() -> None:
    with pytest.raises(ValueError, match="missing columns"):
        cover_reliability_by_line_bucket(pd.DataFrame({"spread_line": [1.0]}))
    all_missing = pd.DataFrame(
        {
            "spread_line": [1.0],
            "home_cover_probability": [np.nan],
            "home_cover": [np.nan],
        }
    )
    with pytest.raises(ValueError, match="No non-push rows"):
        cover_reliability_by_line_bucket(all_missing)


def test_default_key_numbers_are_the_documented_set() -> None:
    assert DEFAULT_KEY_NUMBERS == (1, 2, 3, 4, 6, 7, 10, 14)
