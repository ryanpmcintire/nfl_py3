import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from home_side_side_aware_opener_eval import offsets

from nfl_ats.home_side_location import prior_rows_before


def test_side_cells_and_prior():
    prior = pd.DataFrame(
        {
            "spread_line": [11.0, -11.0, 3.0],
            "result": [21.0, 9.0, 100.0],
            "point_incumbent": [11.0, -11.0, 0.0],
        }
    )
    lines = pd.Series([11.0, -11.0, 3.0, -3.0, 7.0])
    np.testing.assert_allclose(offsets(prior, lines, 100), [10 / 101, 20 / 101, 0, 0, 0])
    np.testing.assert_allclose(offsets(prior, lines, 50), [10 / 51, 20 / 51, 0, 0, 0])


def test_future_same_week_and_old_results_cannot_leak():
    stream = pd.DataFrame(
        {
            "season": [2019, 2020, 2025, 2025, 2026],
            "week": [1, 1, 1, 2, 1],
            "spread_line": [11.0] * 5,
            "point_incumbent": [0.0] * 5,
            "result": [999.0, 10.0, 20.0, 999.0, 999.0],
        }
    )
    before = prior_rows_before(stream, 2025, 2)
    np.testing.assert_allclose(offsets(before, pd.Series([11.0]), 100), [30 / 102])
    stream.loc[[0, 3, 4], "result"] = -999999.0
    np.testing.assert_allclose(
        offsets(prior_rows_before(stream, 2025, 2), pd.Series([11.0]), 100), [30 / 102]
    )


def test_missing_results_and_points_do_not_count():
    frame = pd.DataFrame(
        {
            "spread_line": [-8.0, -8.0, -8.0],
            "result": [10.0, np.nan, 99.0],
            "point_incumbent": [0.0, 0.0, np.nan],
        }
    )
    np.testing.assert_allclose(offsets(frame, pd.Series([-8.0, 8.0]), 100), [10 / 101, 0])
