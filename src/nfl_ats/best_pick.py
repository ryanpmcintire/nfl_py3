from __future__ import annotations

import numpy as np
import pandas as pd

SWEEP_PROBABILITY_FLOOR = 0.50


def sweep_robustness(sweep: pd.DataFrame, picks: pd.DataFrame) -> pd.Series:

    side = picks.set_index("game_id")["pick"].to_dict()
    widths: dict[str, float] = {}
    for raw_game_id, group in sweep.groupby("game_id", sort=False):
        game_id = str(raw_game_id)
        pick = side.get(game_id)
        if pick is None:
            continue
        if group["line_offset"].duplicated().any():
            raise ValueError(
                f"sweep for {game_id} has repeated line offsets; filter it to one method"
            )
        ordered = group.sort_values("line_offset")
        probability = ordered["home_cover_probability"].to_numpy(dtype=float)
        if pick == "AWAY":
            probability = 1.0 - probability
        offsets = ordered["line_offset"].to_numpy(dtype=float)
        holds = probability >= SWEEP_PROBABILITY_FLOOR
        anchor = int(np.argmin(np.abs(offsets)))
        if not holds[anchor]:
            widths[game_id] = 0.0
            continue
        low = anchor
        while low - 1 >= 0 and holds[low - 1]:
            low -= 1
        high = anchor
        while high + 1 < len(holds) and holds[high + 1]:
            high += 1
        widths[game_id] = float(offsets[high] - offsets[low])
    return pd.Series(widths, name="sweep_robustness")


def _pick_sides(predictions: pd.DataFrame) -> pd.DataFrame:
    picks = predictions[["game_id"]].copy()
    picks["game_id"] = picks["game_id"].astype(str)
    picks["pick"] = np.where(
        pd.to_numeric(predictions["home_cover_probability"], errors="coerce") >= 0.5,
        "HOME",
        "AWAY",
    )
    return picks


def best_pick_scores(predictions: pd.DataFrame, sweep: pd.DataFrame) -> pd.Series:

    required = {"game_id", "line_offset", "home_cover_probability"}
    if predictions.empty or sweep.empty or not required.issubset(sweep.columns):
        return pd.Series(dtype=float, name="sweep_robustness")
    if "home_cover_probability" not in predictions.columns:
        return pd.Series(dtype=float, name="sweep_robustness")
    work = sweep.copy()
    work["game_id"] = work["game_id"].astype(str)
    if work.duplicated(subset=["game_id", "line_offset"]).any():
        return pd.Series(dtype=float, name="sweep_robustness")
    return sweep_robustness(work, _pick_sides(predictions))


def select_best_pick(predictions: pd.DataFrame, sweep: pd.DataFrame) -> str | None:

    scores = best_pick_scores(predictions, sweep)
    if scores.empty:
        return None
    ranked = scores.reset_index()
    ranked.columns = ["game_id", "sweep_robustness"]
    ranked = ranked.dropna(subset=["sweep_robustness"])
    if ranked.empty:
        return None
    ranked = ranked.sort_values(["sweep_robustness", "game_id"], ascending=[False, True])
    return str(ranked.iloc[0]["game_id"])


def best_pick_tie_count(predictions: pd.DataFrame, sweep: pd.DataFrame) -> int:

    scores = best_pick_scores(predictions, sweep)
    if scores.empty:
        return 0
    return int((scores == scores.max()).sum())


def best_pick_tie_note(predictions: pd.DataFrame, sweep: pd.DataFrame) -> str:

    tied = best_pick_tie_count(predictions, sweep)
    if tied <= 1:
        return ""
    return (
        f"This week {tied} games tie at the top of that signal, so choosing "
        "between them is arbitrary -- reproducible, but not a lean."
    )
