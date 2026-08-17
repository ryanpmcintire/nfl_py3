"""POL-09 Best Pick selection.

The signal these tests pin is a CONFIRMED one (``docs/best_pick_ranker.md``):
it was scored on two registry windows that are now permanently spent, so its
definition can never be re-derived from data. If one of these tests starts
failing, the deployed Best Pick has drifted away from the signal that earned
its confirmation -- fix the code, never the expectation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from nfl_ats.best_pick import best_pick_scores, select_best_pick, sweep_robustness

OFFSETS = np.arange(-4.0, 4.5, 0.5)


def _sweep(**per_game: np.ndarray) -> pd.DataFrame:
    frames = [
        pd.DataFrame(
            {
                "game_id": game_id,
                "line_offset": OFFSETS,
                "home_cover_probability": probabilities,
            }
        )
        for game_id, probabilities in per_game.items()
    ]
    return pd.concat(frames, ignore_index=True)


def _predictions(**per_game: float) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "game_id": list(per_game),
            "home_cover_probability": list(per_game.values()),
        }
    )


def test_width_is_the_contiguous_run_around_the_quote() -> None:
    # Holds >= 0.5 from offset -1.0 through +2.0, i.e. a three-point run.
    probabilities = np.where((OFFSETS >= -1.0) & (OFFSETS <= 2.0), 0.6, 0.4)
    picks = pd.DataFrame({"game_id": ["g"], "pick": ["HOME"]})
    widths = sweep_robustness(_sweep(g=probabilities), picks)
    assert widths["g"] == 3.0


def test_a_run_is_contiguous_not_a_total_count() -> None:
    """An island of support away from the quote must not inflate the width."""

    probabilities = np.where((OFFSETS >= -0.5) & (OFFSETS <= 0.5), 0.6, 0.4)
    probabilities[OFFSETS >= 3.0] = 0.9  # detached island, must be ignored
    picks = pd.DataFrame({"game_id": ["g"], "pick": ["HOME"]})
    assert sweep_robustness(_sweep(g=probabilities), picks)["g"] == 1.0


def test_away_picks_use_the_complement() -> None:
    """The sweep reports HOME probability; an AWAY pick's edge is 1 - that."""

    probabilities = np.where((OFFSETS >= -1.0) & (OFFSETS <= 1.0), 0.2, 0.8)
    home = sweep_robustness(
        _sweep(g=probabilities), pd.DataFrame({"game_id": ["g"], "pick": ["HOME"]})
    )
    away = sweep_robustness(
        _sweep(g=probabilities), pd.DataFrame({"game_id": ["g"], "pick": ["AWAY"]})
    )
    assert home["g"] == 0.0  # home is behind at the quote
    assert away["g"] == 2.0  # away leads across the two-point run


def test_best_pick_takes_the_widest_run() -> None:
    narrow = np.where(np.abs(OFFSETS) <= 0.5, 0.6, 0.4)
    wide = np.where(np.abs(OFFSETS) <= 3.0, 0.6, 0.4)
    predictions = _predictions(narrow_game=0.6, wide_game=0.6)
    sweep = _sweep(narrow_game=narrow, wide_game=wide)
    assert select_best_pick(predictions, sweep) == "wide_game"


def test_ties_break_on_game_id_so_the_choice_is_reproducible() -> None:
    same = np.where(np.abs(OFFSETS) <= 2.0, 0.6, 0.4)
    predictions = _predictions(b_game=0.6, a_game=0.6)
    sweep = _sweep(b_game=same, a_game=same)
    assert select_best_pick(predictions, sweep) == "a_game"
    # Row order of the card must not change the nomination.
    assert select_best_pick(predictions.iloc[::-1], sweep) == "a_game"


def test_missing_or_malformed_sweep_degrades_to_no_best_pick() -> None:
    predictions = _predictions(g=0.6)
    assert select_best_pick(predictions, pd.DataFrame()) is None
    assert select_best_pick(pd.DataFrame(), _sweep(g=np.full(len(OFFSETS), 0.6))) is None
    assert best_pick_scores(predictions, pd.DataFrame({"game_id": ["g"]})).empty


def test_scores_cover_every_game_on_the_card() -> None:
    predictions = _predictions(one=0.6, two=0.3)
    sweep = _sweep(
        one=np.where(np.abs(OFFSETS) <= 1.0, 0.6, 0.4),
        two=np.where(np.abs(OFFSETS) <= 2.0, 0.4, 0.6),
    )
    scores = best_pick_scores(predictions, sweep)
    assert set(scores.index) == {"one", "two"}
    assert scores.notna().all()
