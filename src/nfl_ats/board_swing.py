from __future__ import annotations

from pathlib import Path

import pandas as pd

_QUARTERS = {1: "1st-quarter", 2: "2nd-quarter", 3: "3rd-quarter", 4: "4th-quarter"}
_SCORE_NAMES = {
    1: "extra point",
    2: "two-point score",
    3: "field goal",
    6: "touchdown",
    7: "touchdown",
    8: "touchdown",
}


def load_game_plays(data_root: Path, season: int) -> pd.DataFrame:
    raw = data_root / "pbp" / "raw"
    if not raw.is_dir():
        return pd.DataFrame()
    for snapshot in sorted((p for p in raw.iterdir() if p.is_dir()), reverse=True):
        path = snapshot / f"season={season}" / "plays.parquet"
        if path.is_file():
            columns = [
                "game_id",
                "play_id",
                "qtr",
                "posteam",
                "home_team",
                "away_team",
                "score_differential",
            ]
            return pd.read_parquet(path, columns=columns)
    return pd.DataFrame()


def _sign(value: float) -> int:
    return (value > 0) - (value < 0)


def _state_word(sign: int) -> str:
    return {1: "covering", 0: "on the number", -1: "behind the number"}[sign]


def cover_states(
    plays: pd.DataFrame,
    game_id: str,
    pick_side: str,
    home_spread: float,
    final_home_margin: float,
) -> tuple[pd.DataFrame, list[float], list[int], list[int]] | None:
    if plays.empty:
        return None
    game = plays.loc[plays["game_id"].astype(str).eq(game_id)]
    game = game.loc[game["posteam"].notna() & game["score_differential"].notna()]
    if game.empty:
        return None
    game = game.sort_values("play_id")
    home = str(game["home_team"].iloc[0])
    side = str(pick_side).strip().upper()
    if side not in {"HOME", "AWAY"}:
        return None
    pick_sign = 1 if side == "HOME" else -1
    margins = [
        float(diff) if str(team) == home else -float(diff)
        for team, diff in zip(game["posteam"], game["score_differential"], strict=True)
    ]
    quarters = [int(q) for q in game["qtr"]]
    margins.append(float(final_home_margin))
    quarters.append(quarters[-1])
    states = [_sign(pick_sign * (m - home_spread)) for m in margins]
    return game, margins, quarters, states


def swing_text(
    plays: pd.DataFrame,
    game_id: str,
    pick_side: str,
    home_spread: float,
    final_home_margin: float,
) -> str | None:
    built = cover_states(plays, game_id, pick_side, home_spread, final_home_margin)
    if built is None:
        return None
    game, margins, quarters, states = built
    home = str(game["home_team"].iloc[0])

    def first_state(quarter: int) -> int | None:
        for q, state in zip(quarters[:-1], states, strict=False):
            if q >= quarter:
                return state
        return None

    half, third = first_state(3), first_state(4)
    if half is None or third is None:
        return None
    covered = [index for index, state in enumerate(states) if state > 0]
    if not covered:
        if all(state < 0 for state in states):
            return "Never covered: behind the number all game."
        return "Never covered: at best on the number."
    if half == third:
        opening = f"{_state_word(half).capitalize()} at halftime and after three quarters"
    else:
        opening = (
            f"{_state_word(half).capitalize()} at halftime, "
            f"{_state_word(third)} after three quarters"
        )
    if half <= 0 and third <= 0:
        brief = _QUARTERS.get(quarters[covered[-1]])
        span = f"the {brief.removesuffix('-quarter')}" if brief else "overtime"
        opening = (
            f"{_state_word(half).capitalize()} at halftime and after three quarters, "
            f"briefly covering in {span}"
        )
    last = None
    for index in range(len(states) - 1):
        if states[index] != states[index + 1]:
            last = index
    if last is None:
        return f"{opening}."
    delta = round(abs(margins[last + 1] - margins[last]))
    scorer = home if margins[last + 1] > margins[last] else str(game["away_team"].iloc[0])
    quarter = quarters[last]
    when = _QUARTERS.get(quarter, "overtime")
    play = _SCORE_NAMES.get(delta, "score")
    article = "an" if play[0] in "aeiou" else "a"
    if when == "overtime":
        scene = f"in overtime on a {play} by {scorer}"
    else:
        scene = f"on {article} {when} {play} by {scorer}"
    verb = "lost the cover" if states[last] > 0 else "slipped behind the number"
    if quarter <= 2:
        return f"{verb.capitalize()} {scene} and never got it back."
    return f"{opening}; {verb} {scene}."
