"""Lane AD: synthetic pins for the officials-archive battery evaluation.

Everything here is synthetic. The script itself is loaded by path (``scripts/``
is not part of the installed package -- the same import shape
``tests/test_officials_wayback_sweep.py`` and ``nfl_ats.lockday_package`` use),
and no test touches a real snapshot, fits a model, or writes an artifact.

Two things must not drift:

1. **The trait-change accounting.** ``change_report`` is what the lane's whole
   answer rests on -- "how many games' trait values move when the archive is
   switched on". A silent change in how it counts a moved row, or in how it
   treats rows only one side has, would turn a measured no-op into an
   unmeasured claim.
2. **The crew-composition statistic**, including its leakage contract:
   ``positions_off_modal_crew_prior`` may only ever see an official's EARLIER
   games in the same season, while ``positions_off_modal_crew_season`` is the
   deliberately whole-season descriptive twin. AGENTS.md requires a leakage
   regression test for a new feature family; ``test_a_later_game_cannot_move_an
   _earlier_games_prior_statistic`` is it.
"""

from __future__ import annotations

import importlib.util
import sys
from collections import Counter
from pathlib import Path
from types import ModuleType

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "officials_archive_battery_eval.py"

CREW_POSITIONS = (
    "Referee",
    "Umpire",
    "Head Linesman",
    "Line Judge",
    "Field Judge",
    "Side Judge",
    "Back Judge",
)


def _load_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location("officials_archive_battery_eval", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def lane() -> ModuleType:
    return _load_script()


def test_an_unchanged_trait_reports_zero_changed_games(lane: ModuleType) -> None:
    before = pd.DataFrame({"game_id": ["a", "b", "c"], "tenure": [0, 1, 2]})
    report = lane.change_report(before, before.copy(), "game_id", ("tenure",))
    assert report["rows_shared"] == 3
    assert report["rows_only_before"] == 0
    assert report["rows_only_after"] == 0
    assert report["columns"]["tenure"]["n_changed"] == 0
    assert report["columns"]["tenure"]["mean_delta"] == 0.0
    assert report["columns"]["tenure"]["max_abs_delta"] == 0.0


def test_a_moved_trait_reports_its_count_and_magnitude(lane: ModuleType) -> None:
    before = pd.DataFrame({"game_id": ["a", "b", "c"], "tenure": [0, 0, 5]})
    after = pd.DataFrame({"game_id": ["a", "b", "c"], "tenure": [6, 0, 5]})
    report = lane.change_report(before, after, "game_id", ("tenure",))
    assert report["columns"]["tenure"]["n_changed"] == 1
    assert report["columns"]["tenure"]["max_abs_delta"] == 6.0
    assert report["columns"]["tenure"]["mean_delta"] == pytest.approx(2.0)


def test_rows_only_one_side_has_are_counted_not_dropped(lane: ModuleType) -> None:
    before = pd.DataFrame({"game_id": ["a", "b"], "tenure": [1, 2]})
    after = pd.DataFrame({"game_id": ["a", "b", "c", "d"], "tenure": [1, 2, 9, 9]})
    report = lane.change_report(before, after, "game_id", ("tenure",))
    assert report["rows_shared"] == 2
    assert report["rows_only_after"] == 2
    assert report["rows_only_before"] == 0
    assert report["columns"]["tenure"]["n_changed"] == 0


def test_two_missing_values_are_the_same_value_not_a_change(lane: ModuleType) -> None:
    before = pd.DataFrame({"game_id": ["a", "b"], "quartile": [np.nan, 3.0]})
    after = pd.DataFrame({"game_id": ["a", "b"], "quartile": [np.nan, 3.0]})
    report = lane.change_report(before, after, "game_id", ("quartile",))
    assert report["columns"]["quartile"]["n_changed"] == 0


def test_tenure_counts_distinct_prior_seasons_only(lane: ModuleType) -> None:
    games = pd.DataFrame(
        {
            "game_id": ["g1", "g2", "g3", "g4"],
            "official_name": ["Ref A", "Ref A", "Ref A", "Ref B"],
            "season": [2009, 2009, 2012, 2012],
            "week": [1, 2, 1, 1],
        }
    )
    table = lane.tenure_table(games).set_index("game_id")["prior_seasons_experience"]
    assert table.loc["g1"] == 0
    assert table.loc["g2"] == 0
    assert table.loc["g3"] == 1
    assert table.loc["g4"] == 0


def _crew_rows(
    game_id: str, week: int, referee: str, others: str, season: int = 2020
) -> list[dict]:
    """One seven-position crew: ``referee`` plus six officials suffixed ``others``."""

    rows = [
        {
            "game_id": game_id,
            "season": season,
            "week": week,
            "position": "Referee",
            "official_name": referee,
            "referee": referee,
        }
    ]
    for position in CREW_POSITIONS[1:]:
        rows.append(
            {
                "game_id": game_id,
                "season": season,
                "week": week,
                "position": position,
                "official_name": f"{position} {others}",
                "referee": referee,
            }
        )
    return rows


def _three_week_season(final_referee: str) -> pd.DataFrame:
    rows = (
        _crew_rows("g1", 1, "Ref One", "X")
        + _crew_rows("g2", 2, "Ref Two", "X")
        + _crew_rows("g3", 3, final_referee, "X")
    )
    return pd.DataFrame(rows)


def test_week_one_has_no_prior_crew_history_and_is_left_unresolved(lane: ModuleType) -> None:
    table = lane.crew_composition(_three_week_season("Ref One")).set_index("game_id")
    assert table.loc["g1", "positions"] == 7
    assert table.loc["g1", "positions_resolved"] == 0
    assert table.loc["g1", "positions_off_modal_crew_prior"] == 0


def test_a_crew_working_under_a_new_referee_counts_every_borrowed_position(
    lane: ModuleType,
) -> None:
    table = lane.crew_composition(_three_week_season("Ref One")).set_index("game_id")
    assert table.loc["g2", "positions_resolved"] == 6
    assert table.loc["g2", "positions_off_modal_crew_prior"] == 6


def test_a_modal_crew_tie_is_broken_by_the_most_recent_prior_game(lane: ModuleType) -> None:
    table = lane.crew_composition(_three_week_season("Ref One")).set_index("game_id")
    assert table.loc["g3", "positions_resolved"] == 7
    assert table.loc["g3", "positions_off_modal_crew_prior"] == 6


def test_a_later_game_cannot_move_an_earlier_games_prior_statistic(lane: ModuleType) -> None:
    """Leakage regression (AGENTS.md) for the crew-composition family.

    Rewriting week 3's referee must leave every week-1 and week-2 prior-only
    value untouched, while the deliberately whole-season descriptive twin DOES
    move -- which is exactly why only the prior variant is ever graded.
    """

    base = lane.crew_composition(_three_week_season("Ref One")).set_index("game_id")
    mutated = lane.crew_composition(_three_week_season("Ref Two")).set_index("game_id")
    for game in ("g1", "g2"):
        assert (
            base.loc[game, "positions_off_modal_crew_prior"]
            == mutated.loc[game, "positions_off_modal_crew_prior"]
        )
        assert base.loc[game, "positions_resolved"] == mutated.loc[game, "positions_resolved"]
    assert base.loc["g2", "positions_off_modal_crew_season"] == 6
    assert mutated.loc["g2", "positions_off_modal_crew_season"] == 0


def test_a_stable_crew_is_never_flagged_as_scrambled(lane: ModuleType) -> None:
    rows = (
        _crew_rows("g1", 1, "Ref One", "X")
        + _crew_rows("g2", 2, "Ref One", "X")
        + _crew_rows("g3", 3, "Ref One", "X")
    )
    table = lane.crew_composition(pd.DataFrame(rows)).set_index("game_id")
    assert table["positions_off_modal_crew_prior"].max() == 0
    assert table["positions_off_modal_crew_season"].max() == 0


def test_modal_tie_falls_back_to_a_deterministic_order(lane: ModuleType) -> None:
    assert lane._modal(Counter(), None) is None
    assert lane._modal(Counter({"B": 1, "A": 1}), None) == "A"
    assert lane._modal(Counter({"B": 1, "A": 1}), "B") == "B"
    assert lane._modal(Counter({"B": 2, "A": 1}), "A") == "B"


def _paired_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "season": [2020, 2020, 2021, 2021],
            "week": [1, 1, 2, 2],
            "margin_vs_open": [3.0, -2.0, 0.0, 4.0],
        }
    )


def test_identical_picks_are_exactly_zero_not_a_small_number(lane: ModuleType) -> None:
    frame = _paired_frame()
    picks = np.array([True, False, True, True])
    result = lane.paired_accuracy(frame, picks, picks, samples=200)
    assert result["delta"] == 0.0
    assert result["flips"] == 0
    assert result["lower"] == 0.0
    assert result["upper"] == 0.0
    assert result["n"] == 3


def test_one_flip_moves_the_delta_by_one_games_worth(lane: ModuleType) -> None:
    frame = _paired_frame()
    baseline = np.array([True, True, True, True])
    candidate = np.array([True, False, True, True])
    result = lane.paired_accuracy(frame, candidate, baseline, samples=200)
    assert result["flips"] == 1
    assert result["delta"] == pytest.approx(100 / 3)
    assert result["baseline_accuracy"] == pytest.approx(2 / 3)
    assert result["candidate_accuracy"] == pytest.approx(1.0)


def test_attaching_a_candidate_column_keeps_the_frames_own_index(lane: ModuleType) -> None:
    features = pd.DataFrame({"game_id": ["a", "b", "c"], "x": [1, 2, 3]}, index=[10, 11, 12])
    values = pd.DataFrame({"game_id": ["a", "c"], "flag": [1.0, -1.0]})
    attached = lane.attach(features, "flag", values)
    assert list(attached.index) == [10, 11, 12]
    assert attached["flag"].tolist() == [1.0, 0.0, -1.0]


def test_jsonable_flattens_numpy_scalars(lane: ModuleType) -> None:
    payload = lane.jsonable({"a": np.int64(3), "b": [np.float64(1.5), np.bool_(True)]})
    assert payload == {"a": 3, "b": [1.5, True]}
    assert isinstance(payload["a"], int)
