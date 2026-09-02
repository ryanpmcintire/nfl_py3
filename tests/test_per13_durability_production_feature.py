"""Leakage regression + wiring tests for the ``weak_stack_durability`` candidate
profile (docs/per13_durability_stage2_on_production.md), per AGENTS.md's "a
leakage regression test for every new feature family" rule.

The durability prior's own point-in-time contract is already proven in
``tests/test_durability_prior.py``. These tests cover what Stage 2 adds on top of
it: the odds-ratio severity swap, the walk-forward offset model, the fact that a
zero offset reproduces production exactly, and the additive ``game_id`` join.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nfl_ats import players
from nfl_ats.constants import (
    FEATURE_SETS,
    MODEL_FEATURE_COLUMNS,
    PER13_DURABILITY_ON_PRODUCTION_FEATURE_COLUMNS,
    PER13_DURABILITY_SWAPPED_BASE_COLUMNS,
)
from nfl_ats.data import DataContractError
from nfl_ats.durability_prior import DURABILITY_COLUMNS, DurabilityHistory
from nfl_ats.margin import MARGIN_FEATURE_PROFILES, margin_feature_columns, margin_feature_groups
from nfl_ats.per13_durability_production_feature import (
    attach_durability_injury_columns,
    augmented_injury_frame,
    augmented_unavailability,
    build_durability_offsets,
    derive_durability_injury_columns,
    durability_column_name,
    durability_severity,
    offset_lookup,
    reproduction_report,
)

# ---------------------------------------------------------------------------
# Registration: a REPLACEMENT, so the column count must not move
# ---------------------------------------------------------------------------


def test_weak_stack_durability_profile_swaps_nine_columns_and_adds_none() -> None:
    assert "weak_stack_durability" in MARGIN_FEATURE_PROFILES
    assert len(PER13_DURABILITY_SWAPPED_BASE_COLUMNS) == 9
    assert len(PER13_DURABILITY_ON_PRODUCTION_FEATURE_COLUMNS) == 9
    assert set(PER13_DURABILITY_ON_PRODUCTION_FEATURE_COLUMNS).isdisjoint(MODEL_FEATURE_COLUMNS)
    for name in ("football_weak_stack", "full_weak_stack", "football", "full"):
        assert set(FEATURE_SETS[name]).isdisjoint(PER13_DURABILITY_ON_PRODUCTION_FEATURE_COLUMNS)

    production = margin_feature_columns("market_residual", "weak_stack")
    candidate = margin_feature_columns("market_residual", "weak_stack_durability")
    assert len(candidate) == len(production)
    assert set(candidate) - set(production) == set(PER13_DURABILITY_ON_PRODUCTION_FEATURE_COLUMNS)
    assert set(production) - set(candidate) == set(PER13_DURABILITY_SWAPPED_BASE_COLUMNS)
    # Every candidate column is claimed by exactly one declared feature family,
    # so a group-wise ridge penalty stays comparable between the arms.
    assert len(margin_feature_groups("market_residual", "weak_stack_durability")) == len(candidate)


def test_swapped_base_columns_are_exactly_the_availability_derived_ones() -> None:
    """The nine are the columns whose value multiplies ``_injury_unavailability``.

    ``player_continuity`` and ``player_qb`` are built from rosters, snaps and
    quarterback history; if a future change moved one of them onto the
    availability severity this assertion is where it should be revisited.
    """

    from nfl_ats.constants import PLAYER_INJURY_STATE_METRICS, PLAYER_VALUE_STATE_METRICS

    expected = tuple(
        f"diff_{metric}" for metric in (*PLAYER_INJURY_STATE_METRICS, *PLAYER_VALUE_STATE_METRICS)
    )
    assert expected == PER13_DURABILITY_SWAPPED_BASE_COLUMNS
    assert (
        tuple(durability_column_name(column) for column in expected)
        == PER13_DURABILITY_ON_PRODUCTION_FEATURE_COLUMNS
    )


# ---------------------------------------------------------------------------
# The odds-ratio severity update
# ---------------------------------------------------------------------------


def test_zero_offset_reproduces_production_severity_exactly() -> None:
    base = np.array([0.0, 0.05, 0.35, 0.5, 0.85, 1.0])
    updated = augmented_unavailability(base, np.zeros_like(base))
    assert updated.tolist() == base.tolist()


def test_endpoints_are_immovable_and_the_update_stays_in_range() -> None:
    """A player listed Out stays out; a clean full practice stays at 0.0."""

    for offset in (-8.0, -1.0, 1.0, 8.0):
        assert augmented_unavailability(np.array([0.0]), offset)[0] == 0.0
        assert augmented_unavailability(np.array([1.0]), offset)[0] == 1.0
    interior = np.array([0.05, 0.35, 0.85])
    for offset in (-8.0, -0.5, 0.5, 8.0):
        updated = augmented_unavailability(interior, offset)
        assert np.all(updated > 0.0) and np.all(updated < 1.0)


def test_the_update_is_monotone_in_the_offset() -> None:
    base = np.full(3, 0.35)
    lower = augmented_unavailability(base, -0.5)
    upper = augmented_unavailability(base, 0.5)
    assert np.all(lower < base)
    assert np.all(upper > base)


def test_augmented_injury_frame_matches_production_row_by_row_at_zero_offset() -> None:
    visible = pd.DataFrame(
        {
            "season": [2021, 2021],
            "week": [3, 3],
            "gsis_id": ["00-A", "00-B"],
            "report_status": ["Questionable", "Out"],
            "practice_status": ["Limited Participation", "Did Not Participate"],
        }
    )
    augmented = augmented_injury_frame(visible, {})
    assert augmented is not None
    expected = [players._injury_unavailability(row) for _, row in visible.iterrows()]
    assert augmented["_unavailability"].tolist() == expected
    # The caller's frame is never mutated.
    assert "_unavailability" not in visible.columns


def test_augmented_injury_frame_passes_none_and_empty_through() -> None:
    assert augmented_injury_frame(None, {}) is None
    empty = pd.DataFrame(columns=["season", "week", "gsis_id"])
    assert augmented_injury_frame(empty, {}).empty


# ---------------------------------------------------------------------------
# The patch touches the two aggregators and nothing else
# ---------------------------------------------------------------------------


def test_durability_severity_patches_only_the_two_aggregators_and_restores_them() -> None:
    original_unavailability = players._injury_unavailability
    original_injury = players._injury_features
    original_value = players._injury_value_features

    with durability_severity({(2021, 3, "00-A"): 2.0}):
        # The quarterback branch reads _injury_unavailability directly; leaving
        # it untouched is what keeps the swap at nine columns rather than eleven.
        assert players._injury_unavailability is original_unavailability
        assert players._injury_features is not original_injury
        assert players._injury_value_features is not original_value

    assert players._injury_features is original_injury
    assert players._injury_value_features is original_value


def test_durability_severity_restores_the_aggregators_after_an_exception() -> None:
    original_injury = players._injury_features
    with pytest.raises(RuntimeError), durability_severity({}):
        raise RuntimeError("boom")
    assert players._injury_features is original_injury


def test_durability_severity_moves_the_aggregated_injury_total() -> None:
    visible = pd.DataFrame(
        {
            "season": [2021],
            "week": [3],
            "gsis_id": ["00-A"],
            "position": ["WR"],
            "report_status": ["Questionable"],
            "practice_status": ["Limited Participation"],
        }
    )
    roles = {"00-A": {"offense_pct": 1.0, "defense_pct": 0.0, "st_pct": 0.0}}
    unpatched = players._injury_features(visible, roles)["injury_offense_unavailability"]
    with durability_severity({(2021, 3, "00-A"): 2.0}):
        patched = players._injury_features(visible, roles)["injury_offense_unavailability"]
    assert patched > unpatched
    with durability_severity({}):
        neutral = players._injury_features(visible, roles)["injury_offense_unavailability"]
    assert neutral == unpatched


# ---------------------------------------------------------------------------
# A synthetic panel for the offset model
# ---------------------------------------------------------------------------

_PLAYERS = [f"00-{index:02d}" for index in range(12)]
_SEASONS = (2020, 2021, 2022)
_WEEKS = tuple(range(1, 9))


def _panel() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for season in _SEASONS:
        for week in _WEEKS:
            kickoff = pd.Timestamp(f"{season}-09-01", tz="UTC") + pd.Timedelta(weeks=week - 1)
            for index, player in enumerate(_PLAYERS):
                rows.append(
                    {
                        "gsis_id": player,
                        "season": season,
                        "week": week,
                        "kickoff": kickoff,
                        "position_group": "skill" if index % 2 else "front",
                        "report_category": "questionable",
                        # A stable per-player durability difference: the low
                        # indices miss far more often than the designation cell
                        # implies, the high indices far less.
                        "unavailable": float((index + week + season) % 12 < index),
                        "cell_probability": 0.35,
                    }
                )
    return pd.DataFrame(rows)


def _rosters() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for season in _SEASONS:
        for week in _WEEKS:
            for index, player in enumerate(_PLAYERS):
                rows.append(
                    {
                        "gsis_id": player,
                        "season": season,
                        "week": week,
                        "position_group": "skill" if index % 2 else "front",
                        "status": "RES" if (index + week) % 7 == 0 else "ACT",
                        "played": bool((index + week) % 3),
                        "snap_covered": True,
                    }
                )
    return pd.DataFrame(rows)


def _history(outcomes: pd.DataFrame) -> DurabilityHistory:
    return DurabilityHistory(outcomes=outcomes.copy(), rosters=_rosters())


def _training(outcomes: pd.DataFrame) -> pd.DataFrame:
    frame = outcomes.copy()
    frame["decision_cutoff"] = frame["kickoff"] - pd.Timedelta(hours=24)
    frame["base_probability"] = frame["cell_probability"]
    return frame


def _targets(outcomes: pd.DataFrame) -> pd.DataFrame:
    frame = outcomes.loc[:, ["gsis_id", "season", "week", "kickoff", "position_group"]].copy()
    frame["decision_cutoff"] = frame["kickoff"] - pd.Timedelta(hours=24)
    return frame


def _offsets(outcomes: pd.DataFrame) -> pd.DataFrame:
    return build_durability_offsets(
        _history(outcomes), _training(outcomes), _targets(outcomes), min_train_rows=24
    )


def test_a_player_with_no_prior_history_gets_exactly_zero() -> None:
    outcomes = _panel()
    offsets = _offsets(outcomes)
    first_week = offsets.loc[offsets["season"].eq(2020) & offsets["week"].eq(1)]
    assert not first_week.empty
    assert (first_week["offset"] == 0.0).all()
    assert not first_week["has_history"].any()
    # ...and the offset does become non-zero once history and a fitted model
    # exist, so the zero above is a property of the row, not of the harness.
    assert offsets["offset"].abs().max() > 0.0


def test_future_outcomes_never_change_an_earlier_offset() -> None:
    """AGENTS.md: a leakage regression test for every new feature family."""

    outcomes = _panel()
    baseline = _offsets(outcomes)

    mutated = outcomes.copy()
    future = mutated["season"].eq(2022)
    mutated.loc[future, "unavailable"] = 1.0 - mutated.loc[future, "unavailable"]
    changed = _offsets(mutated)

    earlier = baseline["season"].lt(2022).to_numpy()
    np.testing.assert_array_equal(
        changed.loc[earlier, "offset"].to_numpy(), baseline.loc[earlier, "offset"].to_numpy()
    )
    # The perturbation is not inert: 2022's own offsets read 2022's history.
    assert not np.array_equal(
        changed.loc[~earlier, "offset"].to_numpy(), baseline.loc[~earlier, "offset"].to_numpy()
    )


def test_offset_lookup_drops_the_exact_zeros() -> None:
    offsets = _offsets(_panel())
    lookup = offset_lookup(offsets)
    assert len(lookup) == int(offsets["offset"].ne(0.0).sum())
    assert all(value != 0.0 for value in lookup.values())


def test_durability_columns_are_all_zero_exactly_when_the_offset_is() -> None:
    """The offset is a linear form in the six columns, so an all-zero row must
    produce an exactly zero offset regardless of the fitted coefficients."""

    outcomes = _panel()
    history = _history(outcomes)
    targets = _targets(outcomes)
    aggregates = history.aggregates(targets)
    from nfl_ats.durability_prior import durability_prior_columns

    calibration = history.calibration(before_season=2022)
    columns = durability_prior_columns(aggregates, calibration)
    all_zero = columns.eq(0.0).all(axis=1).to_numpy()
    offsets = _offsets(outcomes)
    assert all_zero.any()
    assert (offsets.loc[all_zero, "offset"] == 0.0).all()
    assert set(columns.columns) == set(DURABILITY_COLUMNS)


# ---------------------------------------------------------------------------
# The additive join back onto production
# ---------------------------------------------------------------------------


def _production_table(n: int = 6) -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "game_id": [f"2013_0{index + 1}_AAA_BBB" for index in range(n)],
            "season": 2013,
            "week": range(1, n + 1),
            "some_pre_existing_column": np.arange(n, dtype=float),
        }
    )
    for position, column in enumerate(PER13_DURABILITY_SWAPPED_BASE_COLUMNS):
        frame[column] = np.arange(n, dtype=float) * 0.1 + position
    return frame


def test_candidate_columns_equal_production_when_the_rebuilds_agree() -> None:
    """The identity property sec 5 of the predeclaration freezes as a test."""

    production = _production_table()
    rebuilt = production.copy()
    derived = derive_durability_injury_columns(production, rebuilt, rebuilt).set_index("game_id")
    for column in PER13_DURABILITY_SWAPPED_BASE_COLUMNS:
        pd.testing.assert_series_equal(
            derived[durability_column_name(column)],
            production.set_index("game_id")[column].rename(durability_column_name(column)),
            check_exact=True,
        )


def test_candidate_columns_transport_only_the_rebuild_difference() -> None:
    production = _production_table()
    baseline = production.copy()
    offset = production.copy()
    moved = PER13_DURABILITY_SWAPPED_BASE_COLUMNS[0]
    offset[moved] = offset[moved] + 0.25
    derived = derive_durability_injury_columns(production, offset, baseline).set_index("game_id")
    np.testing.assert_allclose(
        derived[durability_column_name(moved)].to_numpy(),
        production.set_index("game_id")[moved].to_numpy() + 0.25,
    )
    for column in PER13_DURABILITY_SWAPPED_BASE_COLUMNS[1:]:
        np.testing.assert_array_equal(
            derived[durability_column_name(column)].to_numpy(),
            production.set_index("game_id")[column].to_numpy(),
        )


def test_attach_is_purely_additive_and_preserves_row_order() -> None:
    production = _production_table()
    rebuilt = production.copy()
    rebuilt[PER13_DURABILITY_SWAPPED_BASE_COLUMNS[0]] += 1.0
    widened = attach_durability_injury_columns(production, rebuilt, production)

    pd.testing.assert_frame_equal(
        widened[production.columns.tolist()], production, check_exact=True
    )
    assert set(widened.columns) - set(production.columns) == set(
        PER13_DURABILITY_ON_PRODUCTION_FEATURE_COLUMNS
    )
    assert widened.index.tolist() == production.index.tolist()


def test_attach_refuses_a_collision() -> None:
    production = _production_table()
    production[PER13_DURABILITY_ON_PRODUCTION_FEATURE_COLUMNS[0]] = 0.0
    with pytest.raises(DataContractError):
        attach_durability_injury_columns(production, production, production)


def test_attach_refuses_a_duplicated_game_id() -> None:
    production = _production_table()
    production.loc[1, "game_id"] = production.loc[0, "game_id"]
    with pytest.raises(DataContractError):
        attach_durability_injury_columns(production, production, production)


def test_reproduction_report_flags_a_mismatch() -> None:
    production = _production_table()
    assert reproduction_report(production, production)["bit_identical"] is True
    drifted = production.copy()
    drifted[PER13_DURABILITY_SWAPPED_BASE_COLUMNS[2]] += 1e-9
    report = reproduction_report(production, drifted)
    assert report["bit_identical"] is False
    assert report["max_absolute_difference_by_column"][
        PER13_DURABILITY_SWAPPED_BASE_COLUMNS[2]
    ] == pytest.approx(1e-9)
