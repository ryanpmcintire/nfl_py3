"""Leakage regression + wiring tests for the `weak_stack_fluview_home`/
`weak_stack_fluview_away` candidate profiles (docs/fluview_on_production.md),
per AGENTS.md's "a leakage regression test for every new feature family" rule.

The underlying as-of/threshold construction this module reuses
(``build_checkpoint_tables``, ``attach_asof_ili``, the per-state decile
threshold convention) already carries an exhaustive leak-safety proof in
``tests/test_fluview_battery_leakage.py``. These tests cover the thin wrapper
this module adds on top of it: that the additive merge does not disturb
pre-existing columns, that the wrapper's own ``game_id``-keyed join does not
reintroduce a leak the engine itself avoids, that the location restriction is
applied correctly, and that the wrapper uses caller-supplied (i.e. frozen)
thresholds rather than recomputing them.

Every test below passes ``fluview_raw=``/``thresholds=`` explicitly, so none
of it depends on the local, gitignored FluView snapshot or the frozen
battery's results artifact actually being present on disk (a fresh clone may
have neither -- see AGENTS.md "Generated artifacts are local and may be
absent in a fresh clone").
"""

from __future__ import annotations

import pandas as pd
import pytest

from nfl_ats.constants import (
    FEATURE_SETS,
    FLUVIEW_AWAY_ELEVATED_ON_PRODUCTION_FEATURE_COLUMNS,
    FLUVIEW_HOME_ELEVATED_ON_PRODUCTION_FEATURE_COLUMNS,
    MODEL_FEATURE_COLUMNS,
)
from nfl_ats.data import DataContractError
from nfl_ats.fluview_production_feature import (
    FLUVIEW_AWAY_ELEVATED_COLUMN,
    FLUVIEW_ELEVATED_ON_PRODUCTION_FEATURE_COLUMNS,
    FLUVIEW_HOME_ELEVATED_COLUMN,
    attach_fluview_elevated_features,
    derive_fluview_elevated_features,
)
from nfl_ats.margin import MARGIN_FEATURE_PROFILES, margin_feature_columns

THRESHOLDS = {"az": 4.0, "ca": 4.0}


def test_fluview_profiles_are_registered_and_disjoint_from_production_sets() -> None:
    assert "weak_stack_fluview_home" in MARGIN_FEATURE_PROFILES
    assert "weak_stack_fluview_away" in MARGIN_FEATURE_PROFILES
    assert set(FLUVIEW_ELEVATED_ON_PRODUCTION_FEATURE_COLUMNS).isdisjoint(MODEL_FEATURE_COLUMNS)
    for name in ("football_weak_stack", "full_weak_stack", "football", "full"):
        assert set(FEATURE_SETS[name]).isdisjoint(FLUVIEW_ELEVATED_ON_PRODUCTION_FEATURE_COLUMNS)

    # Each profile = weak_stack plus exactly its OWN one new column -- never
    # both, never mixed with the sibling graph_sack profile's column.
    home_columns = set(margin_feature_columns("market_residual", "weak_stack_fluview_home"))
    away_columns = set(margin_feature_columns("market_residual", "weak_stack_fluview_away"))
    weak_stack_columns = set(margin_feature_columns("market_residual", "weak_stack"))
    assert home_columns - weak_stack_columns == set(
        FLUVIEW_HOME_ELEVATED_ON_PRODUCTION_FEATURE_COLUMNS
    )
    assert away_columns - weak_stack_columns == set(
        FLUVIEW_AWAY_ELEVATED_ON_PRODUCTION_FEATURE_COLUMNS
    )
    assert len(FLUVIEW_HOME_ELEVATED_ON_PRODUCTION_FEATURE_COLUMNS) == 1
    assert len(FLUVIEW_AWAY_ELEVATED_ON_PRODUCTION_FEATURE_COLUMNS) == 1
    assert FLUVIEW_HOME_ELEVATED_ON_PRODUCTION_FEATURE_COLUMNS[0] == FLUVIEW_HOME_ELEVATED_COLUMN
    assert FLUVIEW_AWAY_ELEVATED_ON_PRODUCTION_FEATURE_COLUMNS[0] == FLUVIEW_AWAY_ELEVATED_COLUMN


# ---------------------------------------------------------------------------
# Synthetic fixture: two states (az=ARI, ca=LAC), a handful of games spanning
# a release-date boundary and one neutral-site game.
# ---------------------------------------------------------------------------


def _games() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "game_id": ["g1", "g2", "g3", "g4"],
            "season": [2018, 2018, 2018, 2018],
            "week": [1, 2, 3, 3],
            # Tuesday-of-week cutoffs: a Thursday gameday's cutoff is the
            # Tuesday two days prior.
            "gameday": pd.to_datetime(["2018-01-11", "2018-01-18", "2018-05-17", "2018-05-17"]),
            "home_team": ["ARI", "ARI", "ARI", "LAC"],
            "away_team": ["LAC", "LAC", "LAC", "ARI"],
            "location": ["Home", "Home", "Home", "Neutral"],
        }
    )


def _fluview_raw() -> pd.DataFrame:
    # az revised from 1.0 (below threshold) to 99.0 (elevated) by a release on
    # 2018-05-18 -- one day AFTER g3/g4's cutoff (2018-05-15, the Tuesday
    # before the Thursday 2018-05-17 gameday).
    return pd.DataFrame(
        {
            "region": ["az", "az", "ca", "ca"],
            "epiweek": [201801, 201801, 201801, 201801],
            "issue": [201802, 201820, 201802, 201820],
            "lag": [1, 19, 1, 19],
            "release_date": pd.to_datetime(
                ["2018-01-08", "2018-05-18", "2018-01-08", "2018-05-18"]
            ),
            "ili": [1.0, 99.0, 1.0, 8.0],
        }
    )


def test_derive_fluview_elevated_features_is_leak_safe_across_a_release() -> None:
    """g3/g4's cutoff (2018-05-15, the Tuesday before Thursday 2018-05-17) is
    strictly BEFORE the 2018-05-18 release that revises az's ili from 1.0
    (not elevated, threshold 4.0) to 99.0 (elevated). Both games must read
    the OLD value; only a game whose cutoff is on/after 2018-05-18 may see
    the revision."""

    games = _games()
    derived = derive_fluview_elevated_features(
        games, fluview_raw=_fluview_raw(), thresholds=THRESHOLDS
    ).set_index("game_id")

    # g3: home=ARI(az), away=LAC(ca); az still reads the OLD (not elevated)
    # value as of this cutoff.
    assert derived.loc["g3", FLUVIEW_HOME_ELEVATED_COLUMN] == 0.0

    # A game whose cutoff is ON the release date sees the revision.
    later = games.copy()
    later["gameday"] = pd.to_datetime(["2018-01-11", "2018-01-18", "2018-05-24", "2018-05-24"])
    derived_later = derive_fluview_elevated_features(
        later, fluview_raw=_fluview_raw(), thresholds=THRESHOLDS
    ).set_index("game_id")
    assert derived_later.loc["g3", FLUVIEW_HOME_ELEVATED_COLUMN] == 1.0


def test_derive_fluview_elevated_features_marks_non_home_location_as_missing() -> None:
    """g4 is a Neutral-location game (docs/fluview_battery.md section 2): both
    columns must be NaN regardless of otherwise-available as-of data, never a
    computed 0/1 value."""

    games = _games()
    derived = derive_fluview_elevated_features(
        games, fluview_raw=_fluview_raw(), thresholds=THRESHOLDS
    ).set_index("game_id")

    assert pd.isna(derived.loc["g4", FLUVIEW_HOME_ELEVATED_COLUMN])
    assert pd.isna(derived.loc["g4", FLUVIEW_AWAY_ELEVATED_COLUMN])
    # g3, the Home-location game at the same cutoff, is NOT masked.
    assert not pd.isna(derived.loc["g3", FLUVIEW_HOME_ELEVATED_COLUMN])


def test_derive_fluview_elevated_features_uses_caller_supplied_thresholds() -> None:
    """The wrapper must apply the THRESHOLDS it is given (the frozen battery's
    own, in production), never recompute a decile from the passed-in games --
    there are far too few games in any real call to recompute a stable
    decile, which is exactly why docs/fluview_on_production.md section 2
    reuses the already-recorded thresholds instead."""

    games = _games()
    low_threshold = {"az": 0.5, "ca": 0.5}  # every reading now counts as elevated
    derived = derive_fluview_elevated_features(
        games, fluview_raw=_fluview_raw(), thresholds=low_threshold
    ).set_index("game_id")
    assert derived.loc["g1", FLUVIEW_HOME_ELEVATED_COLUMN] == 1.0
    assert derived.loc["g1", FLUVIEW_AWAY_ELEVATED_COLUMN] == 1.0


def test_derive_fluview_elevated_features_missing_state_data_is_nan_not_leaked() -> None:
    """A state with no checkpoint rows at all (the measured ``ny``-style gap,
    docs/fluview_battery.md section 1) must resolve to NaN, never defaulted to
    "not elevated"."""

    games = _games()
    empty_fluview = pd.DataFrame(
        columns=["region", "epiweek", "issue", "lag", "release_date", "ili"]
    )
    derived = derive_fluview_elevated_features(
        games, fluview_raw=empty_fluview, thresholds=THRESHOLDS
    ).set_index("game_id")
    assert derived[FLUVIEW_HOME_ELEVATED_COLUMN].isna().all()
    assert derived[FLUVIEW_AWAY_ELEVATED_COLUMN].isna().all()


def test_attach_fluview_elevated_features_is_purely_additive() -> None:
    games = _games()
    base = games.copy()
    base["some_pre_existing_column"] = range(len(base))

    widened = attach_fluview_elevated_features(
        base, fluview_raw=_fluview_raw(), thresholds=THRESHOLDS
    )

    pd.testing.assert_frame_equal(widened[base.columns.tolist()], base, check_exact=True)
    new_columns = set(widened.columns) - set(base.columns)
    assert new_columns == set(FLUVIEW_ELEVATED_ON_PRODUCTION_FEATURE_COLUMNS)


def test_attach_fluview_elevated_features_refuses_a_collision() -> None:
    games = _games()
    games[FLUVIEW_HOME_ELEVATED_COLUMN] = 0.0
    with pytest.raises(DataContractError):
        attach_fluview_elevated_features(games, fluview_raw=_fluview_raw(), thresholds=THRESHOLDS)


def test_derive_fluview_elevated_features_refuses_an_unmapped_team() -> None:
    games = _games()
    games.loc[0, "home_team"] = "ZZZ"
    with pytest.raises(DataContractError):
        derive_fluview_elevated_features(games, fluview_raw=_fluview_raw(), thresholds=THRESHOLDS)
