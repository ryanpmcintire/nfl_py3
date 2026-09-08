"""Lane O sign, boundary, shrinkage and chronology contracts."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from nfl_ats.home_side_location import fit_home_side_offsets

SPEC = importlib.util.spec_from_file_location(
    "lane_o", Path(__file__).resolve().parents[1] / "scripts/cfb_home_side_replication.py"
)
assert SPEC is not None and SPEC.loader is not None
lane = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(lane)


def test_bucket_side_and_home_margin_arithmetic():
    frame = pd.DataFrame(
        {
            "spread_line": [-22, -21, -14.5, -10, -7, -6.5, -3, 0, 3, 6.5, 7, 7.5, 10.5, 21, 22],
            "result": np.arange(15),
            "point_incumbent": np.arange(15) - 2,
        }
    )
    diagnosed = lane.diagnose(frame)
    assert diagnosed.bucket.tolist() == [
        "10.5+",
        "10.5+",
        "10.5+",
        "7.5-10",
        "7",
        "3.5-6.5",
        "0-3",
        "0-3",
        "0-3",
        "3.5-6.5",
        "7",
        "7.5-10",
        "10.5+",
        "10.5+",
        "10.5+",
    ]
    assert diagnosed.side.tolist() == ["home_underdog"] * 7 + ["pickem"] + ["home_favourite"] * 7
    assert diagnosed.error.eq(2).all()
    buckets = dict(lane.bucket_subsets(diagnosed))
    assert buckets["14.5-21"].spread_line.tolist() == [-21, -14.5, 21]
    assert buckets["21+"].spread_line.tolist() == [-22, 22]


def test_offset_pools_sides_and_shrinks_toward_zero():
    prior = pd.DataFrame(
        {"spread_line": [14, -14, 3], "point_incumbent": [10, -10, 0], "result": [12, -6, 10]}
    )
    s3 = fit_home_side_offsets(prior)
    s2 = fit_home_side_offsets(prior, all_buckets=True)
    assert s3.offsets["10.5+"] == pytest.approx(6 / 102)
    assert s3.offsets["0-3"] == 0
    assert s2.offsets["0-3"] == pytest.approx(10 / 101)


def test_prior_excludes_target_future_stale_and_uncompleted_games():
    stream = pd.DataFrame(
        {
            "season": [2014, 2015, 2019, 2020, 2020, 2020, 2021],
            "week": [1, 1, 1, 1, 2, 1, 1],
            "gameday": pd.to_datetime(
                [
                    "2014-09-01",
                    "2015-09-01",
                    "2019-09-01",
                    "2020-09-01",
                    "2020-09-08",
                    "2020-09-07",
                    "2021-09-01",
                ]
            ),
            "result": [1, 1, np.nan, 1, 1, 1, 1],
            "point_incumbent": 0,
        }
    )
    target = stream.iloc[[4]]
    assert lane.eligible_prior(stream, target).index.tolist() == [1, 3]
    changed = stream.copy()
    changed.loc[[0, 2, 4, 5, 6], "point_incumbent"] = 9999
    pd.testing.assert_frame_equal(
        lane.eligible_prior(stream, target), lane.eligible_prior(changed, target)
    )


def test_probability_tie_and_score_directions():
    frame = pd.DataFrame(
        {
            "result": [1, -1],
            "spread_line": [0, 0],
            "p_raw": [0.5, 0.5],
            "p_s3": [0.6, 0.4],
            "p_s2": [0.6, 0.4],
            "p_market": [0.5, 0.5],
        }
    )
    scores = lane.metrics(frame)
    assert scores["raw_accuracy_points"].tolist() == [100, 0]
    assert scores["s3_accuracy_points"].mean() == 100
    assert scores["s3_brier_improvement"].mean() < scores["raw_brier_improvement"].mean()


def test_week_block_interval_deterministic_and_unpadded():
    frame = pd.DataFrame({"season": [2021] * 4, "week": [1, 1, 2, 2]})
    result = lane.interval(frame, np.array([1.0, 1.0, 3.0, 3.0]))
    assert result == lane.interval(frame, np.array([1.0, 1.0, 3.0, 3.0]))
    assert result["effect"] == 2
    assert result["interval_low"] == 1
    assert result["interval_high"] == 3
    assert result["probability_positive"] == 1
