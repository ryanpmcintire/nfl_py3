from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from nfl_ats.dependence import prediction_dependence_audit, team_residual_panel


def _predictions() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    teams = ("A", "B", "C", "D")
    for game_index in range(40):
        home = teams[game_index % 4]
        away = teams[(game_index + 1) % 4]
        actual = float((game_index // 4) % 2)
        rows.append(
            {
                "game_id": f"game_{game_index:02d}",
                "season": 2020 + game_index // 20,
                "week": game_index % 10 + 1,
                "gameday": date(2020, 1, 1) + timedelta(days=7 * game_index),
                "home_team": home,
                "away_team": away,
                "home_cover": actual,
                "home_cover_probability": 0.55 if actual else 0.45,
            }
        )
    return pd.DataFrame(rows)


def test_team_residual_panel_has_opposite_game_errors() -> None:
    panel = team_residual_panel(_predictions())
    assert len(panel) == 80
    assert panel.groupby("game_id")["probability_error"].sum().abs().max() == pytest.approx(0.0)
    assert panel["previous_error"].notna().sum() == 76


@pytest.mark.full  # ENG-11: asserts determinism; dominates --durations
def test_prediction_dependence_audit_is_deterministic() -> None:
    first = prediction_dependence_audit(_predictions(), permutations=100, seed=7)
    second = prediction_dependence_audit(_predictions(), permutations=100, seed=7)
    assert first.summary == second.summary
    assert first.summary["games"] == 40
    assert first.summary["teams"] == 4
    assert 0.0 <= first.summary["season_shuffle_two_sided_p_value"] <= 1.0
    assert np.isfinite(first.team_summary["lag1_error_correlation"]).all()


def test_dependence_guards() -> None:
    with pytest.raises(ValueError, match="at least 100"):
        prediction_dependence_audit(_predictions(), permutations=10)
    empty = _predictions()
    empty["home_cover"] = np.nan
    with pytest.raises(ValueError, match="No evaluated"):
        prediction_dependence_audit(empty, permutations=100)
