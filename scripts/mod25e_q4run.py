import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
ART = REPO / "artifacts" / "mod25e3"
LABEL = "crHpqokg"


def load_sim(seed, nfiles=12):
    out = []
    for f in [x for x in sorted(glob.glob(str(ART / f"e5_{LABEL}_s{seed}" / "play_*_*.parquet"))) if int(Path(x).stem.split("_")[2]) >= 2][:nfiles]:
        out.append(pd.read_parquet(f, columns=["qtr", "gsr", "code", "po", "yl", "yards", "idx", "sd", "flip"]))
    return pd.concat(out, ignore_index=True)


def real_clean():
    import mod25e_late as ml

    cols = ["game_id", "season_type", "posteam", "qtr", "game_seconds_remaining", "yardline_100", "play_type", "score_differential", "down", "ydstogo", "touchdown", "fumble_lost", "yards_gained", "qb_kneel", "qb_spike"]
    out = []
    for s in ml.POOL:
        p = pd.read_parquet(ml.PBP / f"season={s}" / "plays.parquet", columns=cols)
        p = p[(p.season_type == "REG") & p.posteam.notna() & p.score_differential.notna() & (p.qtr <= 4) & p.down.notna() & (p.play_type == "run")]
        clean = ~((p.touchdown == 1) | (p.fumble_lost == 1) | ((p.down == 4) & (p.yards_gained < p.ydstogo)))
        out.append(pd.DataFrame({"gk": str(s) + "_" + p.game_id, "qtr": p.qtr, "gsr": p.game_seconds_remaining, "sd": p.score_differential, "yards": p.yards_gained, "clean": clean}))
    return pd.concat(out, ignore_index=True)


def sim_clean():
    out = []
    for sd in (11, 12, 13):
        for f in sorted(glob.glob(str(ART / f"e5_{LABEL}_s{sd}" / "play_*_*.parquet"))):
            w, s = (int(x) for x in Path(f).stem.split("_")[1:])
            if s < 2:
                continue
            d = pd.read_parquet(f, columns=["g", "qtr", "gsr", "code", "po", "sd", "yards", "flip", "down", "dist"])
            d = d[(d.code == 0) & (d.qtr <= 4)]
            clean = (d.flip == 0) & (d.po < 6)
            out.append(pd.DataFrame({"gk": f"{sd}_{w}_{s}_" + d.g.astype(int).astype(str), "qtr": d.qtr, "gsr": d.gsr, "sd": d.sd, "yards": d.yards, "clean": clean}))
    return pd.concat(out, ignore_index=True)


def slope(D, rng, nb=100):
    x = D.sd.to_numpy() * -1.0 / 7.0
    y = D.yards.to_numpy()
    gk, gi = np.unique(D.gk.to_numpy(), return_inverse=True)
    G = len(gk)
    n = np.bincount(gi, minlength=G).astype(float)
    sx = np.bincount(gi, x, G)
    sy = np.bincount(gi, y, G)
    sxx = np.bincount(gi, x * x, G)
    sxy = np.bincount(gi, x * y, G)

    def f(w):
        N = (w * n).sum()
        mx = (w * sx).sum() / N
        my = (w * sy).sum() / N
        return ((w * sxy).sum() / N - mx * my) / ((w * sxx).sum() / N - mx * mx)

    b = f(np.ones(G))
    bs = np.array([f(np.bincount(rng.integers(0, G, G), minlength=G).astype(float)) for _ in range(nb)])
    return b, bs


def clean_main():
    rng = np.random.default_rng(5)
    R = real_clean()
    S = sim_clean()
    print("rows real", len(R), "sim", len(S), "clean share real", round(float(R.clean.mean()), 4), "sim", round(float(S.clean.mean()), 4))
    for nm, X in (("real", R), ("sim", S)):
        print(nm, "ypc by qtr all-run", X.groupby("qtr").yards.mean().round(3).to_dict(), "clean", X[X.clean].groupby("qtr").yards.mean().round(3).to_dict(), "unclean share by qtr", (1 - X.groupby("qtr").clean.mean()).round(4).to_dict())
    for lab, sel in (("Q4 trailers", lambda X: (X.qtr == 4) & (X.sd < 0)), ("Q4 last5 trailers", lambda X: (X.qtr == 4) & (X.sd < 0) & (X.gsr < 300)), ("Q3 trailers", lambda X: (X.qtr == 3) & (X.sd < 0))):
        for cl in (False, True):
            res = {}
            for nm, X in (("real", R), ("sim", S)):
                Y = X[sel(X) & (X.clean if cl else True)]
                res[nm] = slope(Y, rng)
            d = res["real"][1] - res["sim"][1]
            dd = res["real"][0] - res["sim"][0]
            print(lab, "clean" if cl else "raw", "slope per 7 behind real %+.3f sim %+.3f diff %+.3f [%+.3f,%+.3f] pp %.2f" % (res["real"][0], res["sim"][0], dd, *np.quantile(d, [0.025, 0.975]), float((d > 0).mean())))


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "clean":
        clean_main()
        return
    import mod25e_draw as dr

    G, cfg = dr.build()
    ar = G["tables"]["arrays"]
    pc = np.asarray(ar["play_type_code"])
    py = np.asarray(ar["yards_gained"], dtype=np.float64)
    import mod25d_variance as dv
    import sim04_engine as sim

    trans = sim.build_transition_frame(sim.load_reg_seasons(tuple(dv.TRAIN)))
    ptr = trans["time_raw"].to_numpy().astype(np.float64)
    pfp = np.asarray(ar["fp_raw"], dtype=np.float64)
    flip = np.asarray(ar["possession_flip"]).astype(bool)
    po = np.asarray(ar["points_off"], dtype=np.float64)
    fl = []
    for f in [x for x in sorted(glob.glob(str(ART / f"e5_{LABEL}_s11" / "play_*_*.parquet"))) if int(Path(x).stem.split("_")[2]) >= 2]:
        fl.append(pd.read_parquet(f, columns=["g", "qtr", "gsr", "code", "po", "sd", "yards", "flip", "idx"]))
    S = pd.concat(fl, ignore_index=True)
    S = S[(S.code == 0) & (S.qtr <= 4) & (S.flip == 0) & (S.po < 6)]
    ix = S.idx.to_numpy().astype(int)
    pcode = pc[ix]
    same = pcode == 0
    pclean = same & ~flip[ix] & (po[ix] <= 0)
    print("sim clean runs", len(S), "pool row is run share", float(same.mean()), "pool row clean-run share", float(pclean.mean()))
    print("sim yards mean", float(S.yards.mean()), "pool yards at idx (all)", float(py[ix].mean()), "pool at idx clean-run", float(py[ix][pclean].mean()), "sim yards on those", float(S.yards.to_numpy()[pclean].mean()))
    print("rows pool not run: pool code counts", pd.Series(pcode[~same]).value_counts().to_dict(), "sim yards there", float(S.yards.to_numpy()[~same].mean()), "pool yards there", float(py[ix][~same].mean()))
    print("pool clean-run unweighted mean", float(py[(pc == 0) & ~flip & (po <= 0)].mean()))
    d = S.yards.to_numpy() - py[ix]
    print("sim minus pool yards at idx: mean", float(d.mean()), "share changed", float((d != 0).mean()), "mean when changed", float(d[d != 0].mean()))
    print("by qtr: sim", S.groupby("qtr").yards.mean().round(3).to_dict(), "pool-at-idx", pd.Series(py[ix]).groupby(S.qtr.to_numpy()).mean().round(3).to_dict())
    tr = (S.qtr == 4) & (S.sd < 0)
    for lab, y in (("sim yards", S.yards.to_numpy()), ("pool yards at idx", py[ix])):
        x = -S.sd.to_numpy()[tr.to_numpy()] / 7.0
        yy = y[tr.to_numpy()]
        print("Q4 trailers clean slope per 7 behind", lab, round(float(np.polyfit(x, yy, 1)[0]), 3))

    bad = ~pclean
    B = S[bad]
    print("rows pool flip/TD but sim flip0 po<6: n", int(bad.sum()), "pool flip share", float(flip[ix][bad].mean()), "pool TD share", float((po[ix][bad] > 0).mean()), "sim yards mean", float(B.yards.mean()), "pool yards mean", float(py[ix][bad].mean()))
    print("  sim yards quantiles", np.quantile(B.yards, [0.05, 0.25, 0.5, 0.75, 0.95]).tolist(), "sim yl mean", float(S.sd.mean()))
    C = S[pclean]
    print("clean pool rows only: sim yards by qtr", C.groupby("qtr").yards.mean().round(3).to_dict())
    tr2 = ((C.qtr == 4) & (C.sd < 0)).to_numpy()
    print("Q4 trailers slope on pclean rows", round(float(np.polyfit(-C.sd.to_numpy()[tr2] / 7.0, C.yards.to_numpy()[tr2], 1)[0]), 3), "n", int(tr2.sum()))
    tr3 = tr2 & (C.gsr.to_numpy() < 300)
    print("Q4 last5 trailers slope pclean", round(float(np.polyfit(-C.sd.to_numpy()[tr3] / 7.0, C.yards.to_numpy()[tr3], 1)[0]), 3), "n", int(tr3.sum()))


if __name__ == "__main__":
    main()
