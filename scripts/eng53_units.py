import os
from pathlib import Path

import pandas as pd

ENV = "ENG53_UNITS"
COLUMNS = ["trait", "unit", "entity", "season", "game_id", "parity", "resid", "weight"]


def piece(trait, unit, entity, season, game_id, parity, resid, weight):
    n = len(resid)
    frame = pd.DataFrame(
        {
            "trait": [trait] * n,
            "unit": pd.Series(unit).astype(str).to_numpy(),
            "entity": pd.Series(entity).astype(str).to_numpy(),
            "season": pd.Series(season).astype("int64").to_numpy(),
            "game_id": pd.Series(game_id).astype(str).to_numpy(),
            "parity": pd.Series(parity).astype("int64").to_numpy(),
            "resid": pd.Series(resid).astype(float).to_numpy(),
            "weight": pd.Series(weight).astype(float).to_numpy(),
        }
    )
    return frame[COLUMNS]


def emit(study, pieces):
    target = os.environ.get(ENV)
    if not target:
        return
    out = Path(target)
    out.mkdir(parents=True, exist_ok=True)
    pd.concat(pieces, ignore_index=True).to_parquet(out / f"{study}.parquet", index=False)
