"""Score prediction and pool cards at externally supplied home spreads.

The main weekly pipeline evaluates every game at the schedule's own closing
``spread_line``. This module lets a card be re-evaluated at a different line
-- an externally supplied CSV of alternative spreads, or a captured Tuesday
opener snapshot -- by overriding ``spread_line`` and re-running the fitted
``MarginModel``'s predictive distribution at that line. It never fits a new
model: it only changes which line the same distribution is compared against.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

from nfl_ats.margin import MarginModel
from nfl_ats.prediction_safety import validate_three_way_split

PushRule = Literal["win", "loss", "half"]
PUSH_RULES: tuple[PushRule, ...] = ("win", "loss", "half")

_PUSH_CREDIT: dict[PushRule, float] = {"win": 1.0, "loss": 0.0, "half": 0.5}

_IDENTITY_COLUMNS = ("game_id", "season", "week", "gameday", "home_team", "away_team")


def load_lines_file(path: Path) -> pd.DataFrame:
    """Load a CSV of externally supplied home spreads.

    The file must contain a ``home_spread`` column plus either a ``game_id``
    column, or both ``home_team`` and ``away_team`` columns, to key each
    supplied line to a game.
    """

    frame = pd.read_csv(path)
    if "home_spread" not in frame.columns:
        raise ValueError("Lines file must contain a home_spread column")
    has_game_id = "game_id" in frame.columns
    has_teams = {"home_team", "away_team"}.issubset(frame.columns)
    if not has_game_id and not has_teams:
        raise ValueError(
            "Lines file must contain either a game_id column or home_team and away_team columns"
        )
    frame = frame.copy()
    frame["home_spread"] = pd.to_numeric(frame["home_spread"], errors="coerce")
    if frame["home_spread"].isna().any():
        raise ValueError("Lines file contains a non-numeric home_spread value")
    if has_game_id:
        if frame["game_id"].duplicated().any():
            raise ValueError("Lines file contains duplicate game_id values")
    elif frame.duplicated(["home_team", "away_team"]).any():
        raise ValueError("Lines file contains duplicate home_team/away_team pairs")
    return frame


def apply_external_lines(frame: pd.DataFrame, lines: pd.DataFrame) -> pd.DataFrame:
    """Override ``spread_line`` with externally supplied lines.

    ``lines`` is keyed by ``game_id`` when present, otherwise by
    ``home_team``/``away_team`` pairs. Every row in ``frame`` must have a
    matching supplied line; this fails closed rather than silently falling
    back to the schedule's own spread for unmatched games.
    """

    required = {"game_id", "home_team", "away_team", "spread_line"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"Frame is missing columns for external lines: {', '.join(missing)}")
    result = frame.copy()
    if "game_id" in lines.columns:
        mapping = lines.set_index("game_id")["home_spread"]
        matched = result["game_id"].map(mapping)
    else:
        keyed = lines.set_index(["home_team", "away_team"])["home_spread"]
        matched = pd.Series(
            [
                keyed.get((home, away), np.nan)
                for home, away in zip(result["home_team"], result["away_team"], strict=True)
            ],
            index=result.index,
            dtype=float,
        )
    if matched.isna().any():
        missing_games = sorted(result.loc[matched.isna(), "game_id"].astype(str).tolist())
        raise ValueError(f"No external line supplied for games: {', '.join(missing_games)}")
    result["quoted_line"] = result["spread_line"]
    result["spread_line"] = pd.to_numeric(matched, errors="coerce").to_numpy(dtype=float)
    result["line_source"] = "external"
    return result


def rescore_at_lines(model: MarginModel, frame: pd.DataFrame, lines: pd.DataFrame) -> pd.DataFrame:
    """Re-evaluate a fitted margin model's distribution at externally supplied lines."""

    overridden = apply_external_lines(frame, lines)
    forecasts = model.predict(overridden)
    identity_columns = [
        column
        for column in (*_IDENTITY_COLUMNS, "spread_line", "quoted_line", "line_source")
        if column in overridden.columns
    ]
    identity = overridden.loc[:, identity_columns].reset_index(drop=True)
    result = pd.concat([identity, forecasts.reset_index(drop=True)], axis=1)
    validate_three_way_split(result, line_column="spread_line")
    return result


def pick_expected_value(
    pick_win_probability: pd.Series, push_probability: pd.Series, *, push_rule: PushRule = "loss"
) -> pd.Series:
    """Expected pool credit (0 to 1) for a picked side under a configurable push rule.

    - ``"loss"``: a push earns no credit for either side (the pick "did not
      win"), the common convention for confidence pools without void grading.
    - ``"win"``: a push earns full credit for whichever side was picked.
    - ``"half"``: a push earns half credit, approximating a half-point hook.
    """

    if push_rule not in PUSH_RULES:
        raise ValueError(f"Unknown push rule {push_rule!r}; choose one of {PUSH_RULES}")
    credit = _PUSH_CREDIT[push_rule]
    return pick_win_probability.astype(float) + push_probability.astype(float) * credit


def build_ats_pool_card_at_lines(
    predictions: pd.DataFrame, *, push_rule: PushRule = "loss"
) -> pd.DataFrame:
    """Force one ATS side per game and rank by push-adjusted expected value.

    Unlike ``nfl_ats.pool.build_ats_pool_card`` (which ranks by distance from
    a 50% coin flip), this ranks by the expected pool credit of the picked
    side once ``push_rule`` has resolved what a push is worth -- meaningful
    only once the distribution has real push mass, which happens at
    externally supplied integer lines.
    """

    required = {
        "game_id",
        "gameday",
        "away_team",
        "home_team",
        "spread_line",
        "home_cover_probability",
        "home_cover_probability_excluding_push",
        "push_probability",
        "home_loss_probability",
    }
    missing = sorted(required.difference(predictions.columns))
    if missing:
        raise ValueError(f"Predictions are missing pool columns: {', '.join(missing)}")
    if push_rule not in PUSH_RULES:
        raise ValueError(f"Unknown push rule {push_rule!r}; choose one of {PUSH_RULES}")
    validate_three_way_split(predictions)

    card = predictions.copy()
    card["pool_side"] = card["home_cover_probability"].ge(0.5).map({True: "HOME", False: "AWAY"})
    card["pool_pick"] = card["home_team"].where(card["pool_side"].eq("HOME"), card["away_team"])
    card["pick_probability"] = card["home_cover_probability"].where(
        card["pool_side"].eq("HOME"), 1.0 - card["home_cover_probability"]
    )
    card["pick_line"] = card["spread_line"].where(
        card["pool_side"].eq("AWAY"), -card["spread_line"]
    )
    home_win = card["home_cover_probability_excluding_push"]
    away_win = card["home_loss_probability"]
    pick_win_probability = home_win.where(card["pool_side"].eq("HOME"), away_win)
    card["pick_expected_value"] = pick_expected_value(
        pick_win_probability, card["push_probability"], push_rule=push_rule
    )
    card["push_probability_at_pick"] = card["push_probability"]
    card["push_rule"] = push_rule
    card["confidence"] = (card["pick_probability"] - 0.5).abs()
    card = card.sort_values(
        ["pick_expected_value", "game_id"], ascending=[False, True]
    ).reset_index(drop=True)
    card["confidence_rank"] = range(1, len(card) + 1)
    return card[
        [
            "confidence_rank",
            "gameday",
            "away_team",
            "home_team",
            "pool_pick",
            "pool_side",
            "pick_line",
            "pick_probability",
            "pick_expected_value",
            "push_probability_at_pick",
            "push_rule",
            "confidence",
            "game_id",
        ]
    ]


def pool_card_at_lines_markdown(card: pd.DataFrame, season: int, week: int) -> str:
    display = card.drop(columns="game_id").copy()
    display["gameday"] = pd.to_datetime(display["gameday"]).dt.date.astype(str)
    display["pick_probability"] = display["pick_probability"].map(lambda value: f"{value:.1%}")
    display["pick_expected_value"] = display["pick_expected_value"].map(
        lambda value: f"{value:.1%}"
    )
    display["push_probability_at_pick"] = display["push_probability_at_pick"].map(
        lambda value: f"{value:.1%}"
    )
    display["confidence"] = display["confidence"].map(lambda value: f"{value:.1%}")
    push_rule = str(card["push_rule"].iloc[0]) if len(card) else "loss"
    return (
        f"# ATS pool card at supplied lines: {season} week {week}\n\n"
        f"Push rule: `{push_rule}`. Every game receives a forced side; rank 1 has the "
        "highest push-adjusted expected value at the supplied line. Confidence is not "
        "evidence of a profitable betting edge.\n\n" + display.to_markdown(index=False) + "\n"
    )
