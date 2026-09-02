"""Contract-substitution proofs for the CFB graph ``team_stat`` REPLACEMENT arm.

Predeclared in ``docs/graph_team_stat_cfb_replacement.md`` section 8. The claim
this work package makes is that one statistic's raw columns were swapped for
its graph column and NOTHING ELSE changed. That claim is only worth as much as
an assertion on the design matrix the ridge actually saw, so these tests check
the fitted estimator's own ``feature_names_in_``, not just a declared tuple.

Leakage is not re-proved here: this work package adds no feature construction,
it imports WP8's builder unchanged, and
``tests/test_graph_team_stat_cfb_feature.py`` already proves week ``w`` reads
only through ``w-1``, that the join back is by ``game_id``, that adaptation A1
keeps a rebranded program one node, and that undeclared cells are refused. This
file imports that module's public surface so a change there breaks here too.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import scripts.cfb_graph_reliability as reliability
import scripts.graph_team_stat_cfb_replacement as replacement
from nfl_ats.cfb_benchmark import fit_cfb_residual_model
from nfl_ats.cfb_features import CFB_MODEL_FEATURE_COLUMNS
from nfl_ats.graph_team_stat_cfb_feature import (
    CFB_AWAY_ID_COLUMN,
    CFB_GRAPH_CELLS,
    CFB_HOME_ID_COLUMN,
    cfb_graph_column,
)

CELL = "def_epa_per_play"


def _training_frame(extra_columns: tuple[str, ...] = ()) -> pd.DataFrame:
    """A synthetic CFB-shaped training frame large enough for the benchmark fit.

    ``fit_cfb_residual_model`` needs at least 50 completed games and enough rows
    left after the out-of-time residual split; 120 rows clears both floors with
    room to spare. Values are deterministic so a failure is a contract failure,
    never a seed.
    """

    rows = 120
    rng = np.random.default_rng(24)
    frame = pd.DataFrame(
        {
            "game_id": np.arange(700_000, 700_000 + rows),
            "gameday": pd.Timestamp("2015-09-05") + pd.to_timedelta(np.arange(rows) // 8, unit="W"),
            "ats_margin": rng.normal(0.0, 13.0, rows),
        }
    )
    for index, column in enumerate(CFB_MODEL_FEATURE_COLUMNS):
        frame[column] = rng.normal(float(index) * 0.01, 1.0, rows)
    for column in extra_columns:
        frame[column] = rng.normal(0.0, 1.0, rows)
    return frame


# ---------------------------------------------------------------------------
# 1-2. The substitution is exact, and nothing else moves
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("cell", CFB_GRAPH_CELLS)
def test_replacement_drops_every_raw_column_and_adds_the_graph(cell: str) -> None:
    contract = replacement.replacement_feature_columns(cell)
    graph_column = cfb_graph_column(cell)

    assert graph_column in contract
    for raw in (f"home_{cell}", f"away_{cell}", f"diff_{cell}"):
        assert raw in CFB_MODEL_FEATURE_COLUMNS, "the raw column must be in the contract to remove"
        assert raw not in contract


@pytest.mark.parametrize("cell", CFB_GRAPH_CELLS)
def test_replacement_changes_exactly_four_column_names(cell: str) -> None:
    contract = replacement.replacement_feature_columns(cell)
    benchmark = set(CFB_MODEL_FEATURE_COLUMNS)
    candidate = set(contract)

    assert benchmark - candidate == set(replacement.raw_cell_columns(cell))
    assert candidate - benchmark == {cfb_graph_column(cell)}
    assert len(contract) == len(CFB_MODEL_FEATURE_COLUMNS) - 3 + 1
    assert len(contract) == len(candidate), "no column may be duplicated by the swap"


@pytest.mark.parametrize("cell", CFB_GRAPH_CELLS)
def test_surviving_columns_keep_the_benchmarks_own_order(cell: str) -> None:
    """The kept columns appear in the benchmark's order, with the graph appended."""

    contract = replacement.replacement_feature_columns(cell)
    dropped = set(replacement.raw_cell_columns(cell))
    expected = [column for column in CFB_MODEL_FEATURE_COLUMNS if column not in dropped]
    assert list(contract[:-1]) == expected
    assert contract[-1] == cfb_graph_column(cell)


@pytest.mark.parametrize("cell", CFB_GRAPH_CELLS)
def test_other_cells_team_state_triples_are_untouched(cell: str) -> None:
    contract = set(replacement.replacement_feature_columns(cell))
    for other in CFB_GRAPH_CELLS:
        if other == cell:
            continue
        for raw in (f"home_{other}", f"away_{other}", f"diff_{other}"):
            assert raw in contract
        assert cfb_graph_column(other) not in contract


# ---------------------------------------------------------------------------
# 3. The FITTED design matrix agrees with the declared contract
# ---------------------------------------------------------------------------


def test_fitted_design_matrix_matches_the_replacement_contract() -> None:
    graph_column = cfb_graph_column(CELL)
    training = _training_frame(extra_columns=(graph_column,))
    contract = replacement.replacement_feature_columns(CELL)

    model = fit_cfb_residual_model(training, feature_columns=contract)

    assert model.feature_columns == contract
    seen = list(model.estimator.feature_names_in_)
    assert seen == list(contract)
    assert graph_column in seen
    for raw in replacement.raw_cell_columns(CELL):
        assert raw not in seen


def test_fitted_benchmark_design_matrix_is_the_frozen_contract() -> None:
    """The reference arm must still be the untouched benchmark."""

    training = _training_frame(extra_columns=(cfb_graph_column(CELL),))
    model = fit_cfb_residual_model(
        training, feature_columns=replacement.arm_feature_columns(CELL)["benchmark"]
    )
    assert list(model.estimator.feature_names_in_) == list(CFB_MODEL_FEATURE_COLUMNS)


# ---------------------------------------------------------------------------
# 4. The ablation arm adds nothing
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("cell", CFB_GRAPH_CELLS)
def test_ablation_is_the_benchmark_minus_three_and_nothing_more(cell: str) -> None:
    ablation = replacement.ablation_feature_columns(cell)
    assert set(ablation) < set(CFB_MODEL_FEATURE_COLUMNS)
    assert set(CFB_MODEL_FEATURE_COLUMNS) - set(ablation) == set(replacement.raw_cell_columns(cell))
    assert cfb_graph_column(cell) not in ablation
    assert len(ablation) == len(CFB_MODEL_FEATURE_COLUMNS) - 3


@pytest.mark.parametrize("cell", CFB_GRAPH_CELLS)
def test_replacement_is_the_ablation_plus_exactly_the_graph_column(cell: str) -> None:
    ablation = replacement.ablation_feature_columns(cell)
    assert replacement.replacement_feature_columns(cell) == (*ablation, cfb_graph_column(cell))


def test_positive_control_swaps_only_the_graph_column_for_the_leak() -> None:
    """The leak replaces the graph column and leaves the other arms alone."""

    honest = replacement.arm_feature_columns(CELL, leak=False)
    leaked = replacement.arm_feature_columns(CELL, leak=True)

    assert honest["benchmark"] == leaked["benchmark"]
    assert honest["ablation"] == leaked["ablation"]
    assert leaked["replacement"][-1] == "ats_margin"
    assert honest["replacement"][-1] == cfb_graph_column(CELL)
    assert leaked["replacement"][:-1] == honest["replacement"][:-1]


# ---------------------------------------------------------------------------
# 5. Undeclared cells are refused (WP8's cell gate, inherited)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "builder",
    [
        replacement.raw_cell_columns,
        replacement.ablation_feature_columns,
        replacement.replacement_feature_columns,
    ],
)
def test_undeclared_cell_is_refused(builder: object) -> None:
    with pytest.raises(ValueError):
        builder("off_explosive_rate")  # type: ignore[operator]


def test_arm_feature_columns_refuses_an_undeclared_cell() -> None:
    with pytest.raises(ValueError):
        replacement.arm_feature_columns("off_plays_per_game")


# ---------------------------------------------------------------------------
# 6. The reliability long-frame reshape is faithful
# ---------------------------------------------------------------------------


def test_long_frame_gives_each_side_its_own_value() -> None:
    games = pd.DataFrame(
        {
            "game_id": [1, 2],
            "season": [2015, 2015],
            "week": [3, 4],
            CFB_HOME_ID_COLUMN: [11, 22],
            CFB_AWAY_ID_COLUMN: [22, 33],
            "home_metric": [0.1, 0.3],
            "away_metric": [0.2, 0.4],
        }
    )
    long = reliability.long_frame(games, {"metric": ("home_metric", "away_metric")})

    assert len(long) == 2 * len(games)
    lookup = {(int(row.game_id), int(row.team_id)): float(row.metric) for row in long.itertuples()}
    assert lookup[(1, 11)] == pytest.approx(0.1)
    assert lookup[(1, 22)] == pytest.approx(0.2)
    assert lookup[(2, 22)] == pytest.approx(0.3)
    assert lookup[(2, 33)] == pytest.approx(0.4)
    assert set(long.columns) == {"game_id", "season", "week", "team_id", "metric"}


def test_long_frame_does_not_mutate_the_caller() -> None:
    games = pd.DataFrame(
        {
            "game_id": [1],
            "season": [2015],
            "week": [3],
            CFB_HOME_ID_COLUMN: [11],
            CFB_AWAY_ID_COLUMN: [22],
            "home_metric": [0.1],
            "away_metric": [0.2],
        }
    )
    before = games.copy(deep=True)
    reliability.long_frame(games, {"metric": ("home_metric", "away_metric")})
    pd.testing.assert_frame_equal(games, before)


def test_graph_rating_pair_names_match_the_katz_columns() -> None:
    for cell in CFB_GRAPH_CELLS:
        home_column, away_column = reliability.graph_rating_pair(cell)
        assert home_column.startswith("home_")
        assert away_column.startswith("away_")
        assert home_column.endswith("_katz")
        assert away_column.endswith("_katz")
        # The differential WP8 attaches is formed from exactly this pair.
        stem = home_column[len("home_") : -len("_katz")]
        assert cfb_graph_column(cell) == f"{stem}_katz_diff"


# ---------------------------------------------------------------------------
# The picks-moved counter, which every reported delta is quoted beside
# ---------------------------------------------------------------------------


def test_picks_moved_counts_only_sides_that_actually_flip() -> None:
    graded = pd.DataFrame(
        {
            "benchmark_probability_close": [0.6, 0.4, 0.55, np.nan],
            "replacement_probability_close": [0.7, 0.6, 0.45, 0.9],
        }
    )
    moved = replacement.picks_moved(graded, "close", "benchmark", "replacement")
    assert moved["n_comparable"] == 3
    assert moved["n_moved"] == 2
    assert moved["fraction_moved"] == pytest.approx(2 / 3)
