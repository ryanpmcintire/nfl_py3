from __future__ import annotations

import pandas as pd
import pytest

from nfl_ats.home_side_location import fit_home_side_offsets


def test_offset_pools_sides_and_shrinks_toward_zero():
    prior = pd.DataFrame(
        {"spread_line": [14, -14, 3], "point_incumbent": [10, -10, 0], "result": [12, -6, 10]}
    )
    s3 = fit_home_side_offsets(prior)
    s2 = fit_home_side_offsets(prior, all_buckets=True)
    assert s3.offsets["10.5+"] == pytest.approx(6 / 102)
    assert s3.offsets["0-3"] == 0
    assert s2.offsets["0-3"] == pytest.approx(10 / 101)
