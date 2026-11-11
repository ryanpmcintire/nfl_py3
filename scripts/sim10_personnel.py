import math
import os
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
TABLE = Path(
    os.environ.get(
        "SIM10_MATCHUP_TABLE", REPO / "artifacts" / "sim07" / "matchup_table_loso.parquet"
    )
)
FLOOR = float(np.finfo(np.float64).eps)
TARGETS = ((1, "td_shift"), (2, "fg_shift"))


class Personnel:
    def __init__(self, real_games_path):
        t = pd.read_parquet(TABLE)
        self.shift = {
            (int(s), int(w), a, b): (float(x), float(y))
            for s, w, a, b, x, y in zip(
                t["season"],
                t["week"],
                t["team"],
                t["opp"],
                t["td_shift"],
                t["fg_shift"],
                strict=True,
            )
        }
        g = pd.read_parquet(real_games_path)
        self.teams = {
            int(s): sorted(set(d["home_team"]) | set(d["away_team"]))
            for s, d in g.groupby("season")
        }
        self.cur = {1: (0.0, 0.0), 0: (0.0, 0.0)}

    def set_game(self, season, week, h0, a0):
        t = self.teams[int(season)]
        home, away = t[h0], t[a0]
        zero = (0.0, 0.0)
        self.cur = {
            1: self.shift.get((int(season), int(week), home, away), zero),
            0: self.shift.get((int(season), int(week), away, home), zero),
        }

    def tg_shift(self, tg, hm):
        d = self.cur[hm]
        if d == (0.0, 0.0):
            return tg
        tg = np.array(tg, dtype=np.float64)
        for (k, _), dp in zip(TARGETS, d, strict=True):
            p = min(max(tg[k], FLOOR), 1.0 - FLOOR)
            tg[k] = 1.0 / (1.0 + math.exp(-(math.log(p / (1.0 - p)) + dp / (p * (1.0 - p)))))
        return tg
