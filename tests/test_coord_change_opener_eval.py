"""The predeclared policy preserves baseline picks when coverage is unknown."""

import sys
from pathlib import Path

import pandas as pd

from nfl_ats.coordinator_changes import COORDINATOR_SEASON_COLUMNS

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.coord_change_opener_eval import fallback


def test_partial_coverage_falls_back_without_changing_covered_picks():
    frame = pd.DataFrame(
        {
            "baseline_correct_open_pr": [1.0, 0.0],
            "candidate_correct_open_pr": [0.0, 1.0],
            "baseline_pick_home_pr": [True, False],
            "candidate_pick_home_pr": [False, True],
            **{column: [float("nan"), 1.0] for column in COORDINATOR_SEASON_COLUMNS},
        }
    )
    actual = fallback(frame)
    assert actual.candidate_correct_open_pr.tolist() == [1.0, 1.0]
    assert actual.candidate_pick_home_pr.tolist() == [True, True]
    assert frame.candidate_correct_open_pr.tolist() == [0.0, 1.0]
