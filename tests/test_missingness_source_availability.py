"""MOD-13 availability-flag contract and leakage regressions."""

from __future__ import annotations

import numpy as np
import pandas as pd

from nfl_ats.constants import FEATURE_SETS, SOURCE_ERA_ROSTER_CONTINUITY_COLUMNS
from nfl_ats.margin import (
    MARGIN_FEATURE_PROFILES,
    SelectiveMissingnessImputer,
    margin_feature_columns,
)
from nfl_ats.missingness_availability import (
    ROSTER_CONTINUITY_DATA_AVAILABLE,
    add_roster_continuity_availability,
)


def _source_frame() -> pd.DataFrame:
    rows = []
    for available in (True, False, True):
        row = dict.fromkeys(SOURCE_ERA_ROSTER_CONTINUITY_COLUMNS, 1.0 if available else np.nan)
        row.update({"result": 3.0, "spread_line": -2.5, "unrelated": 0.0})
        rows.append(row)
    return pd.DataFrame(rows)


def test_candidate_profile_replaces_only_the_missingness_treatment() -> None:
    assert "weak_stack_source_availability" in MARGIN_FEATURE_PROFILES
    production = margin_feature_columns("market_residual", "weak_stack")
    candidate = margin_feature_columns("market_residual", "weak_stack_source_availability")
    assert set(candidate) - set(production) == {ROSTER_CONTINUITY_DATA_AVAILABLE}
    assert set(production).issubset(candidate)
    assert set(SOURCE_ERA_ROSTER_CONTINUITY_COLUMNS).issubset(candidate)
    assert (
        ROSTER_CONTINUITY_DATA_AVAILABLE in FEATURE_SETS["football_weak_stack_source_availability"]
    )
    assert ROSTER_CONTINUITY_DATA_AVAILABLE not in FEATURE_SETS["football_weak_stack"]


def test_availability_flag_is_exact_shared_source_identity() -> None:
    frame = _source_frame()
    flagged = add_roster_continuity_availability(frame)
    assert flagged[ROSTER_CONTINUITY_DATA_AVAILABLE].tolist() == [1.0, 0.0, 1.0]
    partial = frame.copy()
    partial.loc[0, SOURCE_ERA_ROSTER_CONTINUITY_COLUMNS[0]] = np.nan
    assert add_roster_continuity_availability(partial)[
        ROSTER_CONTINUITY_DATA_AVAILABLE
    ].tolist() == [
        0.0,
        0.0,
        1.0,
    ]


def test_availability_flag_never_reads_outcome_or_line_columns() -> None:
    frame = _source_frame()
    baseline = add_roster_continuity_availability(frame)[ROSTER_CONTINUITY_DATA_AVAILABLE]
    mutated = frame.copy()
    mutated["result"] = [99.0, -99.0, 0.0]
    mutated["spread_line"] = [14.0, -14.0, 0.0]
    changed = add_roster_continuity_availability(mutated)[ROSTER_CONTINUITY_DATA_AVAILABLE]
    pd.testing.assert_series_equal(changed, baseline, check_exact=True)


def test_selective_imputer_has_one_explicit_flag_not_seven_implicit_ones() -> None:
    frame = _source_frame().loc[:, [*SOURCE_ERA_ROSTER_CONTINUITY_COLUMNS, "unrelated"]]
    imputer = SelectiveMissingnessImputer(SOURCE_ERA_ROSTER_CONTINUITY_COLUMNS).fit(frame)
    names = imputer.get_feature_names_out().tolist()
    assert all(
        f"missingindicator_{column}" not in names for column in SOURCE_ERA_ROSTER_CONTINUITY_COLUMNS
    )
    assert "missingindicator_unrelated" not in names
    transformed = imputer.transform(frame)
    assert transformed.shape == frame.shape


def test_selective_imputer_keeps_unrelated_production_indicators() -> None:
    frame = _source_frame().loc[:, [*SOURCE_ERA_ROSTER_CONTINUITY_COLUMNS, "unrelated"]]
    frame.loc[1, "unrelated"] = np.nan
    imputer = SelectiveMissingnessImputer(SOURCE_ERA_ROSTER_CONTINUITY_COLUMNS).fit(frame)
    names = imputer.get_feature_names_out().tolist()
    assert names.count("missingindicator_unrelated") == 1
    assert not any(name.startswith("missingindicator_diff_") for name in names)
