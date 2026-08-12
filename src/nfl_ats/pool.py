"""Football-pool outputs derived from calibrated weekly probabilities."""

from __future__ import annotations

import pandas as pd


def build_ats_pool_card(predictions: pd.DataFrame) -> pd.DataFrame:
    """Force one ATS side per game and rank picks by model confidence."""

    required = {
        "game_id",
        "gameday",
        "away_team",
        "home_team",
        "spread_line",
        "home_cover_probability",
    }
    missing = sorted(required.difference(predictions.columns))
    if missing:
        raise ValueError(f"Predictions are missing pool columns: {', '.join(missing)}")
    card = predictions.copy()
    card["pool_side"] = card["home_cover_probability"].ge(0.5).map({True: "HOME", False: "AWAY"})
    card["pool_pick"] = card["home_team"].where(card["pool_side"].eq("HOME"), card["away_team"])
    card["pick_probability"] = card["home_cover_probability"].where(
        card["pool_side"].eq("HOME"), 1.0 - card["home_cover_probability"]
    )
    card["pick_line"] = card["spread_line"].where(
        card["pool_side"].eq("AWAY"), -card["spread_line"]
    )
    card["confidence"] = (card["pick_probability"] - 0.5).abs()
    card = card.sort_values(["confidence", "game_id"], ascending=[False, True]).reset_index(
        drop=True
    )
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
            "confidence",
            "game_id",
        ]
    ]


def pool_card_markdown(card: pd.DataFrame, season: int, week: int) -> str:
    display = card.drop(columns="game_id").copy()
    display["gameday"] = pd.to_datetime(display["gameday"]).dt.date.astype(str)
    display["pick_probability"] = display["pick_probability"].map(lambda value: f"{value:.1%}")
    display["confidence"] = display["confidence"].map(lambda value: f"{value:.1%}")
    return (
        f"# ATS pool card: {season} week {week}\n\n"
        "Every game receives a forced side; rank 1 is the model's highest-confidence pick. "
        "Confidence is not evidence of a profitable betting edge.\n\n"
        + display.to_markdown(index=False)
        + "\n"
    )


def build_straight_up_pool_card(
    predictions: pd.DataFrame, method: str = "market_residual"
) -> pd.DataFrame:
    """Force one winner per game from a named outcome-model probability."""

    required = {
        "game_id",
        "gameday",
        "away_team",
        "home_team",
        "method",
        "home_win_probability",
    }
    missing = sorted(required.difference(predictions.columns))
    if missing:
        raise ValueError(f"Predictions are missing straight-up pool columns: {', '.join(missing)}")
    card = predictions.loc[predictions["method"].eq(method)].copy()
    if card.empty:
        raise ValueError(f"No straight-up predictions found for method {method!r}")
    if card["game_id"].duplicated().any():
        raise ValueError(f"Method {method!r} contains duplicate game predictions")
    if card["home_win_probability"].isna().any():
        raise ValueError(f"Method {method!r} has missing winner probabilities")
    card["pool_side"] = card["home_win_probability"].ge(0.5).map({True: "HOME", False: "AWAY"})
    card["pool_pick"] = card["home_team"].where(card["pool_side"].eq("HOME"), card["away_team"])
    card["pick_probability"] = card["home_win_probability"].where(
        card["pool_side"].eq("HOME"), 1.0 - card["home_win_probability"]
    )
    card["confidence"] = card["pick_probability"] - 0.5
    card = card.sort_values(["confidence", "game_id"], ascending=[False, True]).reset_index(
        drop=True
    )
    card["confidence_rank"] = range(1, len(card) + 1)
    optional = [
        column
        for column in ("market_spread", "fair_spread", "predicted_market_residual")
        if column in card
    ]
    return card[
        [
            "confidence_rank",
            "gameday",
            "away_team",
            "home_team",
            "pool_pick",
            "pool_side",
            "pick_probability",
            "confidence",
            *optional,
            "method",
            "game_id",
        ]
    ]


def straight_up_pool_markdown(card: pd.DataFrame, season: int, week: int) -> str:
    display = card.drop(columns="game_id").copy()
    display["gameday"] = pd.to_datetime(display["gameday"]).dt.date.astype(str)
    display["pick_probability"] = display["pick_probability"].map(lambda value: f"{value:.1%}")
    display["confidence"] = display["confidence"].map(lambda value: f"{value:.1%}")
    method = str(card["method"].iloc[0])
    return (
        f"# Straight-up pool card: {season} week {week}\n\n"
        f"Method: `{method}`. Every game receives a forced winner; rank 1 is the "
        "highest-confidence pick. Optimize for the actual pool rules before using confidence "
        "as points.\n\n" + display.to_markdown(index=False) + "\n"
    )
