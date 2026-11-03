import json

import numpy as np
import pandas as pd
import sim07_ftn_loso as lo
import sim07_matchup_table as mt
import sim07_personnel as m
import sim07_power as p

TABLE = m.OUT / "matchup_table_loso.parquet"
SUMMARY = m.OUT / "matchup_table_loso_summary.json"


def fold_deviation(d, f, season):
    w = (d.season.to_numpy() != season).astype(float)
    cell, sw, *_ = f.fit(w)
    cells = list(f.cells)
    proj, _ = p.projector(cells, sw)
    return cells, proj @ cell


def main():
    fz, dm = mt.load()
    d = lo.build()
    f = lo.fitter(d)
    parts = []
    devs = {}
    for season in sorted(d.season.unique()):
        cells, dev = fold_deviation(d, f, int(season))
        devs[int(season)] = dev
        fold = {"cells": cells, "frozen_deviation": dev.tolist()}
        t = mt.build(d, fold, dm)
        parts.append(t[t.season == season])
    t = pd.concat(parts, ignore_index=True)
    t.to_parquet(TABLE, index=False)
    ins = pd.read_parquet(mt.TABLE)
    j = t.merge(ins, on=["season", "week", "team", "opp"], suffixes=("_loso", "_in"))
    live = j[(j.s_play_loso != 0) & (j.s_play_in != 0)]
    seasons = sorted(devs)
    mat = np.array([devs[s] for s in seasons])
    sign = np.sign(mat)
    agree = float(
        np.mean([(sign[i] == sign[k]).mean() for i in range(len(seasons)) for k in range(i)])
    )
    unanimous = int(((sign == sign[0]).all(axis=0)).sum())
    corr_dev = float(
        np.mean([np.corrcoef(mat[i], mat[k])[0, 1] for i in range(len(seasons)) for k in range(i)])
    )
    out = {
        "rows": len(t),
        "rows_nonzero": int((t.s_play != 0).sum()),
        "sd_td_shift_all": float(t.td_shift.std()),
        "sd_td_shift_nonzero": float(t[t.s_play != 0].td_shift.std()),
        "sd_td_shift_insample_all": float(ins.td_shift.std()),
        "corr_with_insample_all": float(np.corrcoef(j.td_shift_loso, j.td_shift_in)[0, 1]),
        "corr_with_insample_nonzero": float(
            np.corrcoef(live.td_shift_loso, live.td_shift_in)[0, 1]
        ),
        "cells": len(cells),
        "fold_seasons": seasons,
        "cell_sign_agreement_pairwise_mean": agree,
        "cells_same_sign_all_folds": unanimous,
        "cell_dev_corr_pairwise_mean": corr_dev,
        "fold_vs_frozen_corr": {
            s: float(np.corrcoef(devs[s], fz["frozen_deviation"])[0, 1]) for s in seasons
        },
        "sd_td_shift_by_season": {
            int(k): float(v) for k, v in t[t.s_play != 0].groupby("season").td_shift.std().items()
        },
    }
    SUMMARY.write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
