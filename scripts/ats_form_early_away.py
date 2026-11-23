from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import ats_form_and_team_record as base

OUT = base.OUT / "early_away.json"


def main():
    rng = np.random.default_rng(base.SEED + 1)
    wide, _, _ = base.build_games()
    pg = base.model_frame()
    pg = pg.merge(
        wide[["game_id", "ats_margin_diff", "away_ats_mean_margin"]], on="game_id", how="left"
    )
    early = pg["week"].between(*base.EARLY_WEEKS).astype(float)
    pg["early_margin_diff"] = pg["ats_margin_diff"].fillna(0.0) * early
    pg["early_away_margin"] = pg["away_ats_mean_margin"].fillna(0.0) * early
    res = base.on_model(pg, ("early_margin_diff", "early_away_margin"), rng)
    res["early_games"] = int(early.sum())
    OUT.write_text(json.dumps(res, indent=2, default=float))
    print(json.dumps(res, indent=2, default=float))


if __name__ == "__main__":
    main()
