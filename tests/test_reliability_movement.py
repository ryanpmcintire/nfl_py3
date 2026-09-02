"""The movement sweep measures the screens' OWN quantity, with sound arithmetic.

Three things would silently corrupt the 26 line-movement reliabilities, and
each gets a test here:

1. ``scripts/reliability_movement.checkpoint_move`` forms the spread move that
   ``scripts/observed_movement_channel.py`` and
   ``scripts/movement_expansion_battery.py`` threshold. If it ever drifted --
   a flipped sign, a different pair of columns -- the registry would carry a
   reliability belonging to a look-alike quantity rather than the cell's own.
   Proved by Series equality against BOTH screens' own pick builders.
2. ``scripts/reliability_lib.measure_reliability`` does the split arithmetic
   the sweep claims, on a frame whose Pearson r is computable by hand.
3. A window too short to split returns an UNMEASURED status rather than a
   number. A NaN written through as a number would manufacture the appearance
   of a ``no_split_half_reliability`` closing ground out of nothing, which
   AGENTS.md's taxonomy forbids: only a RESOLVED wrong sign or a genuinely
   measured zero reliability may ever close a line of work, and an interval
   containing zero never may.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "scripts") not in sys.path:
    sys.path.append(str(REPO / "scripts"))

import movement_expansion_battery as expansion  # noqa: E402
import observed_movement_channel as channel  # noqa: E402
import reliability_lib as rlib  # noqa: E402
import reliability_movement as sweep  # noqa: E402


def _checkpoint_fixture() -> pd.DataFrame:
    """A hand-built slate covering every branch: up, down, no move, sub-threshold."""

    return pd.DataFrame(
        {
            "game_id": [f"G{index}" for index in range(6)],
            "tue_open_home_spread": [-3.0, -3.0, 2.5, 2.5, 0.0, 7.0],
            "close_home_spread": [-4.5, -2.0, 2.5, 4.0, -1.0, 7.5],
            "thu_pre_tnf_home_spread": [-3.5, -1.0, 3.5, 2.5, 0.0, 6.0],
            sweep.PRODUCTION_PICK: [True, True, False, False, True, False],
        }
    )


# ---------------------------------------------------------------------------
# 1. The movement quantity is the screens' own, not a look-alike
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("threshold", [0.5, 1.0, 2.0])
@pytest.mark.parametrize("current_column", ["close_home_spread", "thu_pre_tnf_home_spread"])
def test_checkpoint_move_reproduces_both_screens_threshold_pick(
    threshold: float, current_column: str
) -> None:
    fixture = _checkpoint_fixture()
    production = fixture[sweep.PRODUCTION_PICK].astype(bool)
    move = sweep.checkpoint_move(fixture[current_column], fixture["tue_open_home_spread"])

    mine, eligible = channel._threshold_pick(move, production, threshold)
    theirs = expansion.threshold_pick(
        fixture[current_column], fixture["tue_open_home_spread"], production, threshold
    )

    assert mine.astype(bool).equals(theirs.astype(bool))
    assert eligible.equals(move.abs().ge(threshold))


@pytest.mark.parametrize("current_column", ["close_home_spread", "thu_pre_tnf_home_spread"])
def test_checkpoint_move_reproduces_both_screens_oracle_pick(current_column: str) -> None:
    fixture = _checkpoint_fixture()
    production = fixture[sweep.PRODUCTION_PICK].astype(bool)
    move = sweep.checkpoint_move(fixture[current_column], fixture["tue_open_home_spread"])

    mine = channel._oracle_pick(move, production)
    theirs = expansion.oracle_pick(
        fixture[current_column], fixture["tue_open_home_spread"], production
    )

    assert mine.astype(bool).equals(theirs.astype(bool))
    # The tie branch both screens share: no move keeps the production pick.
    no_move = move.eq(0.0)
    assert mine.astype(bool)[no_move].equals(production[no_move])


def test_verify_helper_agrees_with_a_persisted_pick_column() -> None:
    """The runtime guard the sweep runs on the real populations, on a fixture."""

    fixture = _checkpoint_fixture()
    production = fixture[sweep.PRODUCTION_PICK].astype(bool)
    fixture["_pick_close_thr_1_0"] = expansion.threshold_pick(
        fixture["close_home_spread"], fixture["tue_open_home_spread"], production, 1.0
    )
    report = sweep.verify_move_reproduces_screen_picks(
        fixture,
        current_column="close_home_spread",
        threshold=1.0,
        persisted_pick_column="_pick_close_thr_1_0",
    )
    assert report["matches_expansion_builder"] is True
    assert report["matches_persisted_artifact_pick"] is True
    assert report["n_games"] == len(fixture)


def test_a_drifted_move_is_caught_rather_than_measured_silently() -> None:
    """A sign flip in the move must break the agreement check, not pass quietly."""

    fixture = _checkpoint_fixture()
    production = fixture[sweep.PRODUCTION_PICK].astype(bool)
    move = sweep.checkpoint_move(fixture["close_home_spread"], fixture["tue_open_home_spread"])
    flipped, _eligible = channel._threshold_pick(-move, production, 1.0)
    correct = expansion.threshold_pick(
        fixture["close_home_spread"], fixture["tue_open_home_spread"], production, 1.0
    )
    assert not flipped.astype(bool).equals(correct.astype(bool))


def test_team_week_frame_signs_the_move_toward_each_side() -> None:
    """A positive move means the market moved onto the HOME side."""

    games = pd.DataFrame(
        {
            "season": [2020, 2020],
            "week": [1, 2],
            "home_team": ["AAA", "BBB"],
            "away_team": ["CCC", "DDD"],
        }
    )
    long = sweep.team_week_movement(games, pd.Series([3.0, -1.5]))

    assert len(long) == 4
    signed = long.set_index("team_id")[sweep.SIGNED_METRIC]
    assert signed["AAA"] == pytest.approx(3.0)
    assert signed["CCC"] == pytest.approx(-3.0)
    assert signed["BBB"] == pytest.approx(-1.5)
    assert signed["DDD"] == pytest.approx(1.5)
    # The magnitude the thresholds compare against is shared by both sides.
    magnitude = long.set_index("team_id")[sweep.ABS_METRIC]
    assert magnitude["AAA"] == magnitude["CCC"] == pytest.approx(3.0)


# ---------------------------------------------------------------------------
# 2. The split arithmetic, on an answer computable by hand
# ---------------------------------------------------------------------------


def _long_frame(values: dict[tuple[str, int], list[float]], metric: str) -> pd.DataFrame:
    rows = []
    for (team, season), series in values.items():
        for index, value in enumerate(series, start=1):
            rows.append({"team_id": team, "season": season, "week": index, metric: value})
    return pd.DataFrame(rows)


def test_recovers_a_hand_computed_pearson_r_and_its_spearman_brown_step_up() -> None:
    # Weeks 1..4, so the odd half is weeks 1 and 3 and the even half weeks 2
    # and 4. Each team-season's two half-means are set directly, which makes
    # the Pearson r between them computable by hand from these two lists.
    odd_means = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]
    even_means = [1.0, 3.0, 2.0, 5.0, 4.0, 7.0, 6.0]
    values = {
        f"T{index}": [odd, even, odd, even]
        for index, (odd, even) in enumerate(zip(odd_means, even_means, strict=True))
    }
    long = _long_frame(
        {(team, 2020): series for team, series in values.items()}, "move_toward_team"
    )

    expected_r = float(np.corrcoef(odd_means, even_means)[0, 1])
    expected_sb = 2.0 * expected_r / (1.0 + expected_r)

    result = rlib.measure_reliability(
        long, "move_toward_team", method=rlib.METHOD_TRAIT, n_boot=200, min_units=3
    )

    assert result["status"] == rlib.STATUS_MEASURED
    assert result["n_units"] == len(odd_means)
    assert result["pearson_r"] == pytest.approx(expected_r, abs=1e-9)
    assert result["reliability"] == pytest.approx(expected_sb, abs=1e-9)
    assert result["reliability_low"] <= result["reliability"] <= result["reliability_high"]


# ---------------------------------------------------------------------------
# 3. Too short a window is UNMEASURED, never reliability 0
# ---------------------------------------------------------------------------


def test_a_short_window_returns_unmeasured_and_no_number() -> None:
    """A cell whose own seasons hold too few units must not receive a number.

    The registry reads a low reliability as a candidate closing ground, so a
    NaN written through as 0 would manufacture that ground out of nothing.
    """

    inside = {(f"T{index}", 2020): [1.0, 2.0, 3.0, 4.0] for index in range(3)}
    outside = {(f"T{index}", 2024): [float(index), 1.0, float(index), 1.0] for index in range(40)}
    long = _long_frame({**inside, **outside}, "move_toward_team")

    result = rlib.measure_reliability(
        long, "move_toward_team", method=rlib.METHOD_TRAIT, seasons=(2020, 2020), n_boot=100
    )

    assert result["status"] != rlib.STATUS_MEASURED
    assert result["status"] == rlib.STATUS_INSUFFICIENT_UNITS
    assert result["reliability"] is None
    assert result["reliability_low"] is None and result["reliability_high"] is None


# ---------------------------------------------------------------------------
# 4. The pre-stated near-constant guard
# ---------------------------------------------------------------------------


def test_a_flag_almost_no_unit_ever_carries_is_flagged_not_informative() -> None:
    """A handful of non-constant units can return any correlation at all."""

    values: dict[tuple[str, int], list[float]] = {}
    for index in range(40):
        carries = index < 3
        values[(f"T{index}", 2020)] = [1.0, 0.0, 1.0, 0.0] if carries else [0.0, 0.0, 0.0, 0.0]
    long = _long_frame(values, "exposure")

    diagnostics = sweep.constancy_diagnostics(long, "exposure", (2020, 2020))
    assert diagnostics["n_usable_units"] == 40
    assert diagnostics["n_units_non_constant"] == 3
    assert sweep.near_constant(diagnostics) is True


def test_a_well_populated_flag_is_not_flagged_near_constant() -> None:
    values: dict[tuple[str, int], list[float]] = {}
    for index in range(40):
        share = float(index % 4) / 4.0
        values[(f"T{index}", 2020)] = [share, 1.0 - share, share, 1.0 - share]
    long = _long_frame(values, "exposure")

    diagnostics = sweep.constancy_diagnostics(long, "exposure", (2020, 2020))
    assert diagnostics["n_units_non_constant"] >= rlib.MIN_UNITS
    assert sweep.near_constant(diagnostics) is False


# ---------------------------------------------------------------------------
# 5. The group covers exactly its 26 cells, each mapped to one construct
# ---------------------------------------------------------------------------


def test_the_group_spec_covers_twenty_six_distinct_cells() -> None:
    trait_names = [spec["entry"] for spec in sweep.TRAIT_ENTRIES]
    attribution_names = [
        f"movement_attribution_{population}_{class_name}"
        for population in ("pop_unfiltered", "pop_threshold")
        for class_name in sweep.ATTRIBUTION_CLASSES
    ]
    every = trait_names + attribution_names

    assert len(trait_names) == 12
    assert len(attribution_names) == 14
    assert len(set(every)) == 26


def test_every_trait_cell_names_a_population_and_a_builder() -> None:
    for spec in sweep.TRAIT_ENTRIES:
        assert spec["battery"] in sweep.BUILDER_PROVENANCE
        assert spec["delta"].startswith("delta_")
