from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nfl_ats.lines import (
    apply_external_lines,
    build_ats_pool_card_at_lines,
    load_lines_file,
    pick_expected_value,
    pool_card_at_lines_markdown,
    rescore_at_lines,
)
from nfl_ats.margin import fit_margin_model
from nfl_ats.prediction_safety import PredictionSafetyError


def _target(model_frame: pd.DataFrame) -> pd.DataFrame:
    target = model_frame.tail(3).copy()
    target.loc[:, "home_team"] = "HME"
    target.loc[:, "away_team"] = "AWY"
    return target


def test_load_lines_file_by_game_id(tmp_path) -> None:
    path = tmp_path / "lines.csv"
    pd.DataFrame({"game_id": ["g1", "g2"], "home_spread": [3.0, -6.5]}).to_csv(path, index=False)
    lines = load_lines_file(path)
    assert lines["home_spread"].tolist() == [3.0, -6.5]


def test_load_lines_file_by_teams(tmp_path) -> None:
    path = tmp_path / "lines.csv"
    pd.DataFrame({"home_team": ["SEA"], "away_team": ["NE"], "home_spread": [-3.0]}).to_csv(
        path, index=False
    )
    lines = load_lines_file(path)
    assert lines["home_spread"].tolist() == [-3.0]


def test_load_lines_file_requires_contract(tmp_path) -> None:
    path = tmp_path / "bad.csv"
    pd.DataFrame({"game_id": ["g1"]}).to_csv(path, index=False)
    with pytest.raises(ValueError, match="home_spread column"):
        load_lines_file(path)

    path2 = tmp_path / "bad2.csv"
    pd.DataFrame({"home_spread": [1.0]}).to_csv(path2, index=False)
    with pytest.raises(ValueError, match="game_id column or"):
        load_lines_file(path2)

    path3 = tmp_path / "bad3.csv"
    pd.DataFrame({"game_id": ["g1", "g1"], "home_spread": [1.0, 2.0]}).to_csv(path3, index=False)
    with pytest.raises(ValueError, match="duplicate game_id"):
        load_lines_file(path3)

    path4 = tmp_path / "bad4.csv"
    pd.DataFrame({"game_id": ["g1"], "home_spread": ["not-a-number"]}).to_csv(path4, index=False)
    with pytest.raises(ValueError, match="non-numeric"):
        load_lines_file(path4)


def test_apply_external_lines_overrides_spread_and_fails_closed_on_missing_game() -> None:
    frame = pd.DataFrame(
        {
            "game_id": ["g1", "g2"],
            "home_team": ["A", "B"],
            "away_team": ["C", "D"],
            "spread_line": [1.0, -1.0],
        }
    )
    lines = pd.DataFrame({"game_id": ["g1", "g2"], "home_spread": [3.0, -6.5]})
    overridden = apply_external_lines(frame, lines)
    assert overridden["spread_line"].tolist() == [3.0, -6.5]
    assert overridden["quoted_line"].tolist() == [1.0, -1.0]

    incomplete = pd.DataFrame({"game_id": ["g1"], "home_spread": [3.0]})
    with pytest.raises(ValueError, match="No external line supplied"):
        apply_external_lines(frame, incomplete)


def test_rescore_at_lines_reevaluates_distribution_at_supplied_line(
    model_frame: pd.DataFrame,
) -> None:
    model = fit_margin_model(model_frame, target="margin", model_name="ridge")
    target = _target(model_frame)
    lines = pd.DataFrame({"game_id": target["game_id"].tolist(), "home_spread": [3.0, -6.0, 0.5]})
    rescored = rescore_at_lines(model, target, lines)
    assert rescored["spread_line"].tolist() == [3.0, -6.0, 0.5]
    total = (
        rescored["home_cover_probability_excluding_push"]
        + rescored["push_probability"]
        + rescored["home_loss_probability"]
    )
    assert np.allclose(total, 1.0)
    # Half-point supplied lines can never push.
    assert rescored.loc[rescored["spread_line"].eq(0.5), "push_probability"].eq(0.0).all()


def test_pick_expected_value_push_rules() -> None:
    win = pd.Series([0.5, 0.5])
    push = pd.Series([0.2, 0.0])
    assert pick_expected_value(win, push, push_rule="loss").tolist() == [0.5, 0.5]
    assert pick_expected_value(win, push, push_rule="win").tolist() == [0.7, 0.5]
    assert pick_expected_value(win, push, push_rule="half").tolist() == [0.6, 0.5]
    with pytest.raises(ValueError, match="Unknown push rule"):
        pick_expected_value(win, push, push_rule="void")  # type: ignore[arg-type]


def test_build_ats_pool_card_at_lines_ranks_by_expected_value_and_applies_push_rule() -> None:
    predictions = pd.DataFrame(
        {
            "game_id": ["g1", "g2"],
            "gameday": pd.to_datetime(["2026-09-10", "2026-09-10"]),
            "away_team": ["A", "C"],
            "home_team": ["B", "D"],
            "spread_line": [3.0, -3.0],
            "home_cover_probability": [0.6, 0.3],
            "home_cover_probability_excluding_push": [0.5, 0.65],
            "push_probability": [0.2, 0.05],
            "home_loss_probability": [0.3, 0.3],
        }
    )
    card = build_ats_pool_card_at_lines(predictions, push_rule="win")
    assert set(card["pool_pick"]) == {"B", "C"}
    assert card["confidence_rank"].tolist() == [1, 2]
    assert card["push_rule"].eq("win").all()
    markdown = pool_card_at_lines_markdown(card, 2026, 2)
    assert "ATS pool card at supplied lines: 2026 week 2" in markdown
    assert "`win`" in markdown


def test_build_ats_pool_card_at_lines_requires_contract() -> None:
    with pytest.raises(ValueError, match="missing pool columns"):
        build_ats_pool_card_at_lines(pd.DataFrame({"game_id": ["g1"]}))
    with pytest.raises(ValueError, match="Unknown push rule"):
        build_ats_pool_card_at_lines(
            pd.DataFrame(
                {
                    "game_id": ["g1"],
                    "gameday": pd.to_datetime(["2026-09-10"]),
                    "away_team": ["A"],
                    "home_team": ["B"],
                    "spread_line": [3.0],
                    "home_cover_probability": [0.6],
                    "home_cover_probability_excluding_push": [0.5],
                    "push_probability": [0.2],
                    "home_loss_probability": [0.3],
                }
            ),
            push_rule="void",  # type: ignore[arg-type]
        )


def test_build_ats_pool_card_at_lines_fails_closed_on_broken_three_way_split() -> None:
    predictions = pd.DataFrame(
        {
            "game_id": ["g1"],
            "gameday": pd.to_datetime(["2026-09-10"]),
            "away_team": ["A"],
            "home_team": ["B"],
            "spread_line": [3.0],
            "home_cover_probability": [0.6],
            "home_cover_probability_excluding_push": [0.9],
            "push_probability": [0.2],
            "home_loss_probability": [0.3],
        }
    )
    with pytest.raises(PredictionSafetyError, match="three_way_sum"):
        build_ats_pool_card_at_lines(predictions)
