import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
import mod25e_gain as mg  # noqa: E402

OUT = REPO / "artifacts" / "mod25e3" / "rz"
RZ_YL = 20
FP_GRID = [5.0, 3.0, 2.0, 1.0]
KEYS = ["final", "garb", "aftertov", "rz_td", "rz_fg", "lead4"]


def era_compare(seasons, Ds, Gs, rng, nboot):
    Dr, Gr = mg.real_drives(seasons)
    wpf = mg.wp_model(Dr, Gr)
    Dr = mg.finish(Dr, Gr, wpf)
    Dx = mg.finish(Ds.copy(), Gs, wpf)
    Tr, Ts = mg.table(Dr, Gr), mg.table(Dx, Gs)
    br, bs = mg.gains(Tr), mg.gains(Ts)
    Br, Bs = mg.boot(Tr, rng, nboot), mg.boot(Ts, rng, nboot)
    names = ["final"] + mg.CLASSES
    res = {"games": [len(Tr), len(Ts)], "drives_per_game": [len(Dr) / len(Tr), len(Dx) / len(Ts)]}
    for j, nm in enumerate(names):
        row = {}
        for k, lab in ((1, "P"), (2, "T")):
            d = Bs[:, k, j] - Br[:, k, j]
            row[lab] = {"real": float(br[k, j]), "sim": float(bs[k, j]), "diff": float(bs[k, j] - br[k, j]), "ci": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))], "probability_positive": float((d > 0).mean())}
        res[nm] = row
    res["share"] = {c: [float((Dr["cls"] == c).mean()), float((Dx["cls"] == c).mean())] for c in mg.CLASSES}
    res["var"] = {k + "_" + n: float(T[k].var()) for k in ("e", "p", "tr") for n, T in (("real", Tr), ("sim", Ts))}
    res["tr_var_ratio"] = res["var"]["tr_sim"] / res["var"]["tr_real"]
    return res


def cmd_comp(args):
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(5)
    Ds, Gs = mg.sim_drives()
    out = {}
    for lab, lo, hi in (("pool_2009_17", 2009, 2017), ("held_2018_25", 2018, 2025)):
        out[lab] = era_compare(range(lo, hi + 1), Ds, Gs, rng, args.boot)
        r = out[lab]
        print(lab, "games", r["games"], "drives/g", r["drives_per_game"], "tr_var_ratio", round(r["tr_var_ratio"], 3), flush=True)
        for nm in KEYS:
            for k in ("P", "T"):
                a = r[nm][k]
                print(f"  {nm:9s}{k} real {a['real']:.3f} sim {a['sim']:.3f} d {a['diff']:+.3f} [{a['ci'][0]:+.3f},{a['ci'][1]:+.3f}] pp {a['probability_positive']:.2f}", flush=True)
        print("  share", {c: [round(x, 3) for x in v] for c, v in r["share"].items() if c in KEYS}, flush=True)
    (OUT / "comp.json").write_text(json.dumps(out, indent=1))


def snap(d, r):
    d = d.copy()
    d["yardline_100"] = np.round(d["yardline_100"].to_numpy(float) / r) * r
    return d


def rz_rows(fit, val, r, uf):
    v = snap(val[val["yardline_100"] <= RZ_YL], r)
    sc = uf.Scorer(fit, v)
    rows = sc.rows(uf.CUR)
    return rows, v["game_id"].to_numpy()


def cmd_search(args):
    import sim09_u4d as u4d
    import sim09_u4f as uf

    OUT.mkdir(parents=True, exist_ok=True)
    tf, tv = uf.prep(u4d.real_plays(range(2009, 2014))), uf.prep(u4d.real_plays((2014, 2015)))
    fit, test = uf.prep(u4d.real_plays(range(2009, 2016))), uf.prep(u4d.real_plays((2016, 2017)))
    base_v, _ = rz_rows(tf, tv, FP_GRID[0], uf)
    ref = (base_v["lp"].mean(), base_v["cy"].mean(), base_v["l1"].mean())
    res = {"family": "RZ fp rounding grid " + str(FP_GRID) + ", J = lp + yards CRPS + joint-key LL relative to r=5 on val rows with yardline<=20", "val": {}, "test": {}}
    best, bj = None, 1e9
    for r in FP_GRID:
        rv, _ = rz_rows(tf, tv, r, uf)
        J = rv["lp"].mean() / ref[0] + rv["cy"].mean() / ref[1] + rv["l1"].mean() / ref[2]
        res["val"][str(r)] = float(J)
        print("val r", r, "J", round(float(J), 5), flush=True)
        if J < bj:
            best, bj = r, J
    res["chosen"] = best
    b_rows, gt = rz_rows(fit, test, FP_GRID[0], uf)
    c_rows, _ = rz_rows(fit, test, best, uf)
    for r in FP_GRID:
        rr, _ = rz_rows(fit, test, r, uf)
        res["test"][str(r)] = {k: uf.boot_delta(b_rows[k], rr[k], gt) for k in ("l1", "lp", "cy")}
        print("test r", r, {k: [round(x, 4) for x in v] for k, v in res["test"][str(r)].items()}, flush=True)
    (OUT / "search.json").write_text(json.dumps(res, indent=1, default=float))


def chosen_fp():
    return float(json.loads((OUT / "search.json").read_text())["chosen"])


def install_rzfp(r):
    import mod25d_variance as dv

    ns = dv._G["ns"]
    old = ns["round_state_key"]

    def rz_key(down, phase, dist, fp, score, time_raw, off_to, def_to):
        k = old(down, phase, dist, fp, score, time_raw, off_to, def_to)
        if fp <= RZ_YL:
            return k[:3] + (("z", round(fp / r)),) + k[4:]
        return k

    ns["round_state_key"] = rz_key
    t = dv._G["tables"]
    t["nn_cache_cond"].clear()
    t["nn_weight_cache_cond"].clear()


def rz_init_ss(setting):
    import mod25d_variance as dv
    import sim09_hk as hk

    hk.hk_init_ss(setting)
    if dv._G["cfg"].get("rzfp"):
        install_rzfp(float(dv._G["cfg"]["rzfp"]))


def cmd_e5(args):
    import mod25d_variance as dv
    import mod25e_scorestate as ss
    import sim09_hk as hk

    dv.DV["crzhr"] = dict(dv.DV["crzhk"], rzfp=chosen_fp())
    ss.s_init = rz_init_ss
    args.scale = 1.0
    ss.cmd_e5(args)


def cmd_cmp(args):
    import pandas as pd

    pa = sorted((REPO / "artifacts" / "mod25e3" / f"e5_{args.a}_s{args.seed}").glob("sim_games.parquet"))[0]
    pb = sorted((REPO / "artifacts" / "mod25e3" / f"e5_{args.b}_s{args.seed}").glob("sim_games.parquet"))[0]
    a, b = pd.read_parquet(pa), pd.read_parquet(pb)
    print("equal", a.reset_index(drop=True).equals(b.reset_index(drop=True)), len(a), len(b))


def sim_rz_drives(variant, seed):
    import pandas as pd

    d = REPO / "artifacts" / "mod25e3" / f"e5_{variant}_s{seed}"
    fr = []
    for fp in sorted(d.glob("play_*_*.parquet")):
        _, w, sn = fp.stem.split("_")
        if int(sn) < 2:
            continue
        x = pd.read_parquet(fp)
        x = x[x["qtr"] <= 4].copy()
        x["game_id"] = [f"w{int(w):04d}s{int(sn):02d}g{int(g):03d}" for g in x["g"]]
        fr.append(x)
    P = pd.concat(fr, ignore_index=True)
    prev = P.groupby("game_id")["offhome"].shift(1)
    pq = P.groupby("game_id")["qtr"].shift(1)
    P["drv"] = (prev.isna() | (prev != P["offhome"]) | ((pq <= 2) & (P["qtr"] >= 3))).cumsum()
    gk = P.groupby("drv", sort=False)
    D = pd.DataFrame({"game_id": gk["game_id"].first(), "ylmin": gk["yl"].min(), "po": gk["po"].sum(), "lastcode": gk["code"].last()}).reset_index(drop=True)
    D["oc"] = np.where(D["po"] >= 6, "td", np.where((D["po"] == 3) & (D["lastcode"] == 3), "fg", "other"))
    return D


def rz_stats(D):
    rz = D[D["ylmin"] <= RZ_YL]
    n = D["game_id"].nunique()
    return {"drives_per_game": len(D) / n, "rz_share": len(rz) / len(D), "td_given_rz": float((rz["oc"] == "td").mean()), "fg_given_rz": float((rz["oc"] == "fg").mean()), "n_rz": len(rz), "td_per_game": float((D["oc"] == "td").sum() / n)}


def cmd_rzval(args):
    real = {}
    for lab, lo, hi in (("real_2009_17", 2009, 2017), ("real_2018_25", 2018, 2025)):
        Dr, _ = mg.real_drives(range(lo, hi + 1))
        real[lab] = rz_stats(Dr)
    out = dict(real)
    for v in args.variants.split(","):
        out[v] = rz_stats(sim_rz_drives(v, args.seed))
    for k, v in out.items():
        print(k, {a_: round(b_, 4) for a_, b_ in v.items()}, flush=True)
    (OUT / "rzval.json").write_text(json.dumps(out, indent=1))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("comp")
    c.add_argument("--boot", type=int, default=300)
    sub.add_parser("search")
    z = sub.add_parser("rzval")
    z.add_argument("--variants", default="crzhk,crzhr")
    z.add_argument("--seed", type=int, default=11)
    e = sub.add_parser("e5")
    e.add_argument("--variant", default="crzhr")
    e.add_argument("--worlds", type=int, default=8)
    e.add_argument("--seasons", type=int, default=8)
    e.add_argument("--workers", type=int, default=3)
    e.add_argument("--seed", type=int, default=11)
    m = sub.add_parser("cmp")
    m.add_argument("--a")
    m.add_argument("--b")
    m.add_argument("--seed", type=int, default=11)
    args = ap.parse_args()
    {"comp": cmd_comp, "search": cmd_search, "e5": cmd_e5, "cmp": cmd_cmp, "rzval": cmd_rzval}[args.cmd](args)


if __name__ == "__main__":
    main()
