from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

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
    probabilities = np.where((OFFSETS >= -1.0) & (OFFSETS <= 2.0), 0.6, 0.4)
    picks = pd.DataFrame({"game_id": ["g"], "pick": ["HOME"]})
    widths = sweep_robustness(_sweep(g=probabilities), picks)
    assert widths["g"] == 3.0


def test_away_picks_use_the_complement() -> None:

    probabilities = np.where((OFFSETS >= -1.0) & (OFFSETS <= 1.0), 0.2, 0.8)
    home = sweep_robustness(
        _sweep(g=probabilities), pd.DataFrame({"game_id": ["g"], "pick": ["HOME"]})
    )
    away = sweep_robustness(
        _sweep(g=probabilities), pd.DataFrame({"game_id": ["g"], "pick": ["AWAY"]})
    )
    assert home["g"] == 0.0
    assert away["g"] == 2.0


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
    assert select_best_pick(predictions.iloc[::-1], sweep) == "a_game"


def test_missing_or_malformed_sweep_degrades_to_no_best_pick() -> None:
    predictions = _predictions(g=0.6)
    assert select_best_pick(predictions, pd.DataFrame()) is None
    assert select_best_pick(pd.DataFrame(), _sweep(g=np.full(len(OFFSETS), 0.6))) is None
    assert best_pick_scores(predictions, pd.DataFrame({"game_id": ["g"]})).empty


def test_an_unfiltered_multi_method_sweep_is_refused_not_ranked() -> None:

    wide = np.where(np.abs(OFFSETS) <= 3.0, 0.6, 0.4)
    narrow = np.where(np.abs(OFFSETS) <= 0.5, 0.6, 0.4)
    stacked = pd.concat([_sweep(g=wide), _sweep(g=narrow)], ignore_index=True)
    picks = pd.DataFrame({"game_id": ["g"], "pick": ["HOME"]})
    with pytest.raises(ValueError, match="repeated line offsets"):
        sweep_robustness(stacked, picks)
    assert best_pick_scores(_predictions(g=0.6), stacked).empty
    assert select_best_pick(_predictions(g=0.6), stacked) is None
