import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mod25e_q4run as q

ART = q.ART


def real_pass():
    import mod25e_late as ml

    cols = ["game_id", "season_type", "posteam", "qtr", "game_seconds_remaining", "yardline_100", "play_type", "score_differential", "down", "ydstogo", "touchdown", "fumble_lost", "interception", "yards_gained", "qb_kneel", "qb_spike"]
    out = []
    for s in ml.POOL:
        p = pd.read_parquet(ml.PBP / f"season={s}" / "plays.parquet", columns=cols)
        p = p[(p.season_type == "REG") & p.posteam.notna() & p.score_differential.notna() & (p.qtr <= 4) & p.down.notna() & (p.play_type == "pass")]
        clean = ~((p.touchdown == 1) | (p.fumble_lost == 1) | (p.interception == 1) | ((p.down == 4) & (p.yards_gained < p.ydstogo)))
        out.append(pd.DataFrame({"gk": str(s) + "_" + p.game_id, "qtr": p.qtr, "gsr": p.game_seconds_remaining, "sd": p.score_differential, "yards": p.yards_gained, "clean": clean}))
    return pd.concat(out, ignore_index=True)


def sim_pass():
    import mod25e_draw as dr

    G, cfg = dr.build()
    ar = G["tables"]["arrays"]
    pc = np.asarray(ar["play_type_code"])
    flip = np.asarray(ar["possession_flip"]).astype(bool)
    po = np.asarray(ar["points_off"], dtype=np.float64)
    out = []
    for sd in (11, 12, 13):
        for f in sorted(glob.glob(str(ART / f"e5_{q.LABEL}_s{sd}" / "play_*_*.parquet"))):
            w, s = (int(x) for x in Path(f).stem.split("_")[1:])
            if s < 2:
                continue
            d = pd.read_parquet(f, columns=["g", "qtr", "gsr", "code", "po", "sd", "yards", "flip", "idx"])
            d = d[(d.code == 1) & (d.qtr <= 4)]
            ix = d.idx.to_numpy().astype(int)
            lc = (d.flip == 0).to_numpy() & (d.po < 6).to_numpy()
            pcl = (pc[ix] == 1) & ~flip[ix] & (po[ix] <= 0)
            out.append(pd.DataFrame({"gk": f"{sd}_{w}_{s}_" + d.g.astype(int).astype(str), "qtr": d.qtr, "gsr": d.gsr, "sd": d.sd, "yards": d.yards, "clean": lc & pcl, "logclean": lc}))
    return pd.concat(out, ignore_index=True)


def main():
    rng = np.random.default_rng(7)
    R = real_pass()
    S = sim_pass()
    print("rows real", len(R), "sim", len(S), "clean share real", round(float(R.clean.mean()), 4), "sim pool-clean", round(float(S.clean.mean()), 4), "sim logclean", round(float(S.logclean.mean()), 4))
    for nm, X in (("real all", R), ("real clean", R[R.clean]), ("sim all", S), ("sim logclean", S[S.logclean]), ("sim pool-clean", S[S.clean])):
        print(nm, "ypa by qtr", X.groupby("qtr").yards.mean().round(3).to_dict())
        for lab, m in (("Q4 trailers", (X.qtr == 4) & (X.sd < 0)), ("Q4 last5 trailers", (X.qtr == 4) & (X.sd < 0) & (X.gsr <= 300))):
            b, bs = q.slope(X[m], rng)
            print("  ", lab, "slope per 7 behind", round(float(b), 3), "se", round(float(bs.std()), 3), "n", int(m.sum()))
    for lab, m in (("Q4 trailers", lambda X: (X.qtr == 4) & (X.sd < 0)), ("Q4 last5 trailers", lambda X: (X.qtr == 4) & (X.sd < 0) & (X.gsr <= 300))):
        a = R[R.clean]
        b = S[S.clean]
        n = 100
        ra = []
        for _ in range(n):
            pass
        b1, s1 = q.slope(a[m(a)], rng)
        b2, s2 = q.slope(b[m(b)], rng)
        d = s2 - s1
        print("diff sim-real", lab, round(float(b2 - b1), 3), "pp", round(float((d > 0).mean()), 2), "CI approx", np.quantile(float(b2 - b1) + (d - d.mean()), [0.025, 0.975]).round(3).tolist())


main()
