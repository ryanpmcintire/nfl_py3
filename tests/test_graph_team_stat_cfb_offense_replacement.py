"""Contract-substitution proofs for the CFB JOINT OFFENCE replacement arm (WP35).

Predeclared in ``docs/graph_team_stat_cfb_offense_replacement.md`` section 9. The
claim this work package makes is that BOTH offence statistics' raw triples were
swapped for their graph columns simultaneously, that the DEFENCE triple survived
untouched, and that nothing else changed. That claim is only worth as much as an
assertion on the design matrix the ridge actually saw, so these tests check the
fitted estimator's own ``feature_names_in_``, not just a declared tuple.

Leakage is not re-proved here: this work package adds no feature construction, it
chains WP8's builder twice, and ``tests/test_graph_team_stat_cfb_feature.py``
already proves week ``w`` reads only through ``w-1``, that the join back is by
``game_id``, that adaptation A1 keeps a rebranded program one node, and that
undeclared cells are refused. What IS new is the chaining, so test group 5 proves
that two chained builder calls preserve the caller's row order and index and add
exactly two columns.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import scripts.graph_team_stat_cfb_offense_replacement as offense
from nfl_ats.cfb_benchmark import fit_cfb_residual_model
from nfl_ats.cfb_features import CFB_MODEL_FEATURE_COLUMNS
from nfl_ats.graph_team_stat_cfb_feature import cfb_cell_columns, cfb_graph_column

DEFENCE_CELL = "def_epa_per_play"
DEFENCE_TRIPLE = (
    f"home_{DEFENCE_CELL}",
    f"away_{DEFENCE_CELL}",
    f"diff_{DEFENCE_CELL}",
)
OFFENCE_GRAPH_COLUMNS = tuple(cfb_graph_column(metric) for metric in offense.OFFENCE_METRICS)


def _training_frame(extra_columns: tuple[str, ...] = ()) -> pd.DataFrame:
    """A synthetic CFB-shaped training frame large enough for the benchmark fit.

    ``fit_cfb_residual_model`` needs at least 50 completed games and enough rows
    left after the out-of-time residual split; 120 rows clears both floors with
    room to spare. Values are deterministic so a failure is a contract failure,
    never a seed.
    """

    rows = 120
    rng = np.random.default_rng(35)
    frame = pd.DataFrame(
        {
            "game_id": np.arange(800_000, 800_000 + rows),
            "gameday": pd.Timestamp("2015-09-05") + pd.to_timedelta(np.arange(rows) // 8, unit="W"),
            "ats_margin": rng.normal(0.0, 13.0, rows),
        }
    )
    for index, column in enumerate(CFB_MODEL_FEATURE_COLUMNS):
        frame[column] = rng.normal(float(index) * 0.01, 1.0, rows)
    for column in extra_columns:
        frame[column] = rng.normal(0.0, 1.0, rows)
    return frame


def _cfb_like_games(team_count: int = 16, weeks: int = 6) -> pd.DataFrame:
    """A synthetic CFB-shaped schedule carrying BOTH offence metrics.

    Mirrors ``tests/test_graph_team_stat_cfb_feature.py``'s fixture: rotating
    pairings so the graph is connected rather than isolated pairs, ESPN-like
    integer ids, and a fixed pregame statistic per team per metric so every edge
    weight is deterministic.
    """

    ids = [3000 + index for index in range(team_count)]
    rows: list[dict[str, object]] = []
    game_id = 600_000
    for week in range(1, weeks + 1):
        rotated = ids[week % team_count :] + ids[: week % team_count]
        for slot in range(team_count // 2):
            home_id = rotated[slot]
            away_id = rotated[team_count - 1 - slot]
            game_id += 1
            row: dict[str, object] = {
                "game_id": game_id,
                "season": 2015,
                "week": week,
                "gameday": pd.Timestamp("2015-09-05") + pd.Timedelta(weeks=week - 1),
                "home_id": home_id,
                "away_id": away_id,
                "home_team": f"Team {home_id}",
                "away_team": f"Team {away_id}",
                "result": float((home_id - away_id) % 7) - 3.0,
                "spread_line": 1.5,
            }
            # Each metric gets its OWN team ordering, not a constant offset of
            # the other: the edge signal is home minus away, so a shift applied
            # to both sides would cancel and the two graph columns would come
            # back identical -- which is exactly what
            # ``test_the_two_graph_columns_are_not_the_same_numbers`` exists to
            # rule out.
            for offset, metric in enumerate(offense.OFFENCE_METRICS):
                home_column, away_column = cfb_cell_columns(metric)
                for column, team_id in ((home_column, home_id), (away_column, away_id)):
                    position = ids.index(team_id)
                    row[column] = (
                        0.05 + 0.01 * position
                        if offset == 0
                        else 0.40 - 0.013 * ((position * 7) % team_count)
                    )
            rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 1. The joint substitution is exact
# ---------------------------------------------------------------------------


def test_both_offence_triples_are_gone_and_both_graph_columns_are_in() -> None:
    contract = offense.offence_replacement_feature_columns()

    for graph_column in OFFENCE_GRAPH_COLUMNS:
        assert graph_column in contract
    for metric in offense.OFFENCE_METRICS:
        for raw in (f"home_{metric}", f"away_{metric}", f"diff_{metric}"):
            assert raw in CFB_MODEL_FEATURE_COLUMNS, "the raw column must exist to be removed"
            assert raw not in contract


def test_offence_raw_columns_are_exactly_six_names() -> None:
    raw = offense.offence_raw_columns()
    assert len(raw) == 6
    assert len(set(raw)) == 6
    assert set(raw) < set(CFB_MODEL_FEATURE_COLUMNS)


# ---------------------------------------------------------------------------
# 2. The defence triple is intact and nothing else moves
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "arm",
    [
        "benchmark",
        "offense_replacement",
        "offense_ablation",
        "replacement_off_epa_per_play",
        "replacement_off_success_rate",
    ],
)
def test_the_defence_triple_survives_in_every_arm(arm: str) -> None:
    contract = set(offense.arm_feature_columns()[arm])
    for column in DEFENCE_TRIPLE:
        assert column in contract
    assert cfb_graph_column(DEFENCE_CELL) not in contract


def test_replacement_changes_exactly_eight_column_names() -> None:
    contract = offense.offence_replacement_feature_columns()
    benchmark = set(CFB_MODEL_FEATURE_COLUMNS)
    candidate = set(contract)

    assert benchmark - candidate == set(offense.offence_raw_columns())
    assert candidate - benchmark == set(OFFENCE_GRAPH_COLUMNS)
    assert len(contract) == len(CFB_MODEL_FEATURE_COLUMNS) - 6 + 2 == 31
    assert len(contract) == len(candidate), "no column may be duplicated by the swap"


def test_surviving_columns_keep_the_benchmarks_own_order() -> None:
    contract = offense.offence_replacement_feature_columns()
    dropped = set(offense.offence_raw_columns())
    expected = [column for column in CFB_MODEL_FEATURE_COLUMNS if column not in dropped]

    assert list(contract[:-2]) == expected
    assert tuple(contract[-2:]) == OFFENCE_GRAPH_COLUMNS


def test_the_other_team_state_triples_are_untouched() -> None:
    contract = set(offense.offence_replacement_feature_columns())
    swapped = set(offense.OFFENCE_METRICS)
    for column in CFB_MODEL_FEATURE_COLUMNS:
        for prefix in ("home_", "away_", "diff_"):
            if column.startswith(prefix) and column[len(prefix) :] not in swapped:
                assert column in contract


# ---------------------------------------------------------------------------
# 3. The ablation arm adds nothing
# ---------------------------------------------------------------------------


def test_ablation_is_the_benchmark_minus_six_and_nothing_more() -> None:
    ablation = offense.offence_ablation_feature_columns()

    assert set(ablation) < set(CFB_MODEL_FEATURE_COLUMNS)
    assert set(CFB_MODEL_FEATURE_COLUMNS) - set(ablation) == set(offense.offence_raw_columns())
    assert len(ablation) == len(CFB_MODEL_FEATURE_COLUMNS) - 6 == 29
    for graph_column in OFFENCE_GRAPH_COLUMNS:
        assert graph_column not in ablation


def test_replacement_is_the_ablation_plus_exactly_the_two_graph_columns() -> None:
    ablation = offense.offence_ablation_feature_columns()
    assert offense.offence_replacement_feature_columns() == (*ablation, *OFFENCE_GRAPH_COLUMNS)


def test_the_single_metric_arms_are_wp24s_own_contracts() -> None:
    """Cells 2 and 3 must be WP24's contracts exactly, or they are not continuity."""

    from scripts.graph_team_stat_cfb_replacement import replacement_feature_columns

    arms = offense.arm_feature_columns()
    for metric in offense.OFFENCE_METRICS:
        assert arms[f"replacement_{metric}"] == replacement_feature_columns(metric)
        assert len(arms[f"replacement_{metric}"]) == 33


# ---------------------------------------------------------------------------
# 4. The FITTED design matrix agrees with the declared contract
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "arm",
    [
        "benchmark",
        "offense_replacement",
        "offense_ablation",
        "replacement_off_epa_per_play",
        "replacement_off_success_rate",
    ],
)
def test_fitted_design_matrix_matches_each_arms_contract(arm: str) -> None:
    training = _training_frame(extra_columns=OFFENCE_GRAPH_COLUMNS)
    contract = offense.arm_feature_columns()[arm]

    model = fit_cfb_residual_model(training, feature_columns=contract)

    assert model.feature_columns == contract
    assert list(model.estimator.feature_names_in_) == list(contract)


def test_the_fitted_replacement_matrix_has_no_offence_raw_column() -> None:
    training = _training_frame(extra_columns=OFFENCE_GRAPH_COLUMNS)
    model = fit_cfb_residual_model(
        training, feature_columns=offense.offence_replacement_feature_columns()
    )
    seen = list(model.estimator.feature_names_in_)

    for raw in offense.offence_raw_columns():
        assert raw not in seen
    for graph_column in OFFENCE_GRAPH_COLUMNS:
        assert graph_column in seen
    for column in DEFENCE_TRIPLE:
        assert column in seen


def test_fitted_benchmark_design_matrix_is_the_frozen_contract() -> None:
    """The reference arm must still be the untouched 35-column benchmark."""

    training = _training_frame(extra_columns=OFFENCE_GRAPH_COLUMNS)
    model = fit_cfb_residual_model(
        training, feature_columns=offense.arm_feature_columns()["benchmark"]
    )
    assert list(model.estimator.feature_names_in_) == list(CFB_MODEL_FEATURE_COLUMNS)
    assert len(CFB_MODEL_FEATURE_COLUMNS) == 35


# ---------------------------------------------------------------------------
# 5. Chaining the two builders is order-preserving
# ---------------------------------------------------------------------------


def test_chaining_both_builders_adds_two_columns_and_keeps_row_order() -> None:
    games = _cfb_like_games()
    widened = offense.add_offence_graph_columns(games)

    assert list(widened.index) == list(games.index)
    assert list(widened["game_id"]) == list(games["game_id"])
    assert len(widened) == len(games)
    assert set(widened.columns) - set(games.columns) == set(OFFENCE_GRAPH_COLUMNS)


def test_chaining_never_mutates_the_callers_frame() -> None:
    games = _cfb_like_games()
    before = games.copy()

    offense.add_offence_graph_columns(games)

    pd.testing.assert_frame_equal(games, before)


def test_the_two_graph_columns_are_not_the_same_numbers() -> None:
    """Two metrics must produce two distinct ratings, not one column twice."""

    widened = offense.add_offence_graph_columns(_cfb_like_games())
    first, second = (widened[column] for column in OFFENCE_GRAPH_COLUMNS)
    both = first.notna() & second.notna()

    assert both.any(), "the synthetic schedule must clear the min_games warm-up gate"
    assert not np.allclose(first[both].to_numpy(), second[both].to_numpy())


# ---------------------------------------------------------------------------
# 6. The positive control touches exactly one column
# ---------------------------------------------------------------------------


def test_positive_control_leaks_exactly_one_swapped_column() -> None:
    honest = offense.arm_feature_columns(leak=False)
    leaked = offense.arm_feature_columns(leak=True)

    differing = [
        (a, b)
        for a, b in zip(honest["offense_replacement"], leaked["offense_replacement"], strict=True)
        if a != b
    ]
    assert differing == [(cfb_graph_column(offense.LEAK_METRIC), "ats_margin")]
    assert len(honest["offense_replacement"]) == len(leaked["offense_replacement"])


def test_positive_control_keeps_the_other_graph_column() -> None:
    leaked = offense.arm_feature_columns(leak=True)["offense_replacement"]
    survivor = next(metric for metric in offense.OFFENCE_METRICS if metric != offense.LEAK_METRIC)

    assert cfb_graph_column(survivor) in leaked
    assert cfb_graph_column(offense.LEAK_METRIC) not in leaked
    assert "ats_margin" in leaked


def test_positive_control_leaves_the_unleaked_arms_identical() -> None:
    honest = offense.arm_feature_columns(leak=False)
    leaked = offense.arm_feature_columns(leak=True)
    survivor = next(metric for metric in offense.OFFENCE_METRICS if metric != offense.LEAK_METRIC)

    assert honest["benchmark"] == leaked["benchmark"]
    assert honest["offense_ablation"] == leaked["offense_ablation"]
    assert honest[f"replacement_{survivor}"] == leaked[f"replacement_{survivor}"]
    assert leaked[f"replacement_{offense.LEAK_METRIC}"][-1] == "ats_margin"


# ---------------------------------------------------------------------------
# 7. Undeclared metrics are refused (WP8's cell gate, tightened to offence)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("metric", ["def_epa_per_play", "off_explosive_rate", "spread_line"])
def test_a_metric_outside_the_declared_offence_pair_is_refused(metric: str) -> None:
    with pytest.raises(ValueError):
        offense.validate_metric(metric)


def test_the_declared_pair_is_exactly_the_two_offence_metrics() -> None:
    assert offense.OFFENCE_METRICS == ("off_epa_per_play", "off_success_rate")
    assert offense.LEAK_METRIC in offense.OFFENCE_METRICS


def test_the_defence_cell_is_deliberately_excluded() -> None:
    """The seen-sign selection disclosed in section 1 must be visible in code."""

    assert DEFENCE_CELL not in offense.OFFENCE_METRICS
    assert cfb_graph_column(DEFENCE_CELL) not in offense.offence_replacement_feature_columns()


# ---------------------------------------------------------------------------
# 8. The declared cell list matches the predeclaration
# ---------------------------------------------------------------------------


def test_cell_one_is_the_primary_and_the_diagnostic_is_not_a_cell() -> None:
    labels = [label for label, _, _ in offense.CELL_COMPARISONS]
    assert len(labels) == 4
    assert labels[0].startswith("cell1_primary")
    assert offense.CELL_COMPARISONS[0][1:] == ("benchmark", "offense_replacement")

    diagnostic_labels = [label for label, _, _ in offense.DIAGNOSTIC_COMPARISONS]
    assert diagnostic_labels == ["diagnostic_report_only_offense_ablation_vs_benchmark"]
    assert not set(diagnostic_labels) & set(labels)


def test_every_comparison_names_two_real_arms() -> None:
    arms = set(offense.arm_feature_columns())
    assert arms == set(offense.ARM_NAMES)
    for _label, reference, candidate in offense.COMPARISONS:
        assert reference in arms
        assert candidate in arms
        assert reference != candidate
