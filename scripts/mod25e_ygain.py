import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

import mod25d_variance as dv  # noqa: E402
import mod25e_knobs as kn  # noqa: E402

CAP = {}


def y_init(setting):
    dv.d_gen_init(setting)
    ns = dv._G["ns"]
    orig = ns["run_one_game"]
    CAP["rows"] = []

    def wrapped(state, tables, rng, tmax, pol, k, mx, hr, ar):
        rec, cap = orig(state, tables, rng, tmax, pol, k, mx, hr, ar)
        lg = np.array(dv._G["log"], dtype=np.float64).reshape(-1, 12)
        pl = np.array(ns["PLAYLOG"], dtype=np.float64).reshape(-1, 3)
        CAP["rows"].append((lg, pl, hr, ar))
        return rec, cap

    ns["run_one_game"] = wrapped


def y_play_season(task):
    CAP["rows"].clear()
    res = dv.d_play_season(task)
    world, sidx, sched = task[0], task[1], task[3]
    arr = dv._G["tables"]["attrs"]
    plays = res[3]
    frames = []
    for gi, (lg, pl, hr, ar) in enumerate(CAP["rows"]):
        week, h, a = sched[gi]
        idx = lg[:, 11].astype(np.int64)
        code = arr["code"][idx]
        keep = (code == 0) | (code == 1)
        offhome = pl[keep, 0].astype(int)
        n = int(keep.sum())
        frames.append(pd.DataFrame({
            "g": gi, "week": week,
            "off_t": np.where(offhome == 1, h, a), "def_t": np.where(offhome == 1, a, h),
            "off_r": np.where(offhome == 1, hr["off"], ar["off"]), "def_r": np.where(offhome == 1, ar["def"], hr["def"]),
            "down": lg[keep, 0], "dist": lg[keep, 1], "yl": lg[keep, 2], "code": code[keep],
        }))
    d = pd.concat(frames, ignore_index=True)
    assert len(d) == len(plays)
    d["yards"] = plays["yards"]
    d["epa"] = plays["epa"]
    d["world"] = world
    d["sidx"] = sidx
    d.to_parquet(Path(os.environ["YG_OUT"]) / f"yp_{world}_{sidx}.parquet")
    return res


def cmd_sim(args):
    import mod25_generator as gen

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    os.environ["YG_OUT"] = str(out.resolve())
    dv.c25.ensure_policies()
    cfgj = json.dumps(dict(dv.DV[args.variant], seed=args.seed, name=f"{args.variant}_ygain"))
    setting = dict(gen.SETTING_DEFAULTS)
    setting.update({"scale": args.scale, "yard_gain": args.yard_gain, "def_sign": 1.0, "yard_bias": 0.0, "drift": 1.0, "mech": cfgj})
    gen.init_worker = y_init
    gen.play_season = y_play_season
    gen.run_generation(setting, args.worlds, args.seasons, args.workers, args.seed, progress=False)
    print("done", flush=True)


def slopes(y, eo, ef, cell, cl, boot=200):
    yr = kn.resid_cells(y, cell)
    X = np.c_[kn.resid_cells(eo, cell), kn.resid_cells(ef, cell)]
    b = kn.ols(X, yr)
    r = kn.summ(b, kn.cluster_boot(X, yr, cl, boot), ["off_slope", "def_slope"])
    xs = X[:, 0] + X[:, 1]
    r["sum_slope"] = kn.summ(kn.ols(xs[:, None], yr), kn.cluster_boot(xs[:, None], yr, cl, boot), ["s"])["s"]
    return r


def cmd_analyze(args):
    fs = sorted(Path(args.out).glob("yp_*_*.parquet"))
    d = pd.concat([pd.read_parquet(f) for f in fs if int(f.stem.split("_")[2]) >= 2], ignore_index=True)
    sid = d["world"] * 1000 + d["sidx"]
    d["season"] = sid
    d["posteam"] = "W" + d["world"].astype(str) + "T" + d["off_t"].astype(int).astype(str)
    d["defteam"] = "W" + d["world"].astype(str) + "T" + d["def_t"].astype(int).astype(str)
    d["play_type"] = "run"
    d["gid"] = d["world"].astype(str) + "_" + d["sidx"].astype(str) + "_" + d["g"].astype(str)
    cell = ((d["code"].to_numpy().astype(np.int64) * 10 + d["down"].to_numpy().astype(np.int64)) * 1000 + np.clip(d["dist"].to_numpy(), 0, 20).astype(int) * 20 + (d["yl"].to_numpy() // 10).astype(int))
    y = d["yards"].to_numpy(float)
    cl = (d["season"].astype(str) + "_" + d["posteam"]).to_numpy()
    lo = d["off_r"] - d["off_r"].mean()
    lf = d["def_r"] - d["def_r"].mean()
    res = {"n_plays": int(len(d)), "yard_sd": float(y.std()), "yard_sd_state_resid": float(kn.resid_cells(y, cell).std())}
    res["latent_sd_off"], res["latent_sd_def"] = float(lo.std()), float(lf.std())
    res["latent"] = slopes(y, lo.to_numpy(), lf.to_numpy(), cell, cl)
    half = (pd.util.hash_pandas_object(d["gid"], index=False).to_numpy() % 2).astype(int)
    eo_x = np.zeros(len(d))
    ef_x = np.zeros(len(d))
    for h in (0, 1):
        sub = d[half == h].copy()
        sub["season"] = sub["season"].astype(int)
        dd = dv.team_effects_plays(sub, list(sub["season"].unique()), "epa")
        _, (oc, fc, eo, ef, ou, fu, mu) = dv.fit_play_effects(dd, "epa")
        om, fm = dict(zip(ou, eo)), dict(zip(fu, ef))
        m = half != h
        ok = (d["season"].astype(str) + "_" + d["posteam"]).to_numpy()
        fk = (d["season"].astype(str) + "_" + d["defteam"]).to_numpy()
        eo_x[m] = pd.Series(ok[m]).map(om).fillna(0.0).to_numpy()
        ef_x[m] = pd.Series(fk[m]).map(fm).fillna(0.0).to_numpy()
    res["crossfit"] = slopes(y, eo_x, ef_x, cell, cl)
    res["crossfit_sd_eo"], res["crossfit_sd_ef"] = float(eo_x.std()), float(ef_x.std())
    pbp, trans, at = kn.load()
    code = trans["play_type_code"].to_numpy()
    rp = (code == 0) | (code == 1)
    yr = at["yards"].astype(float)[rp]
    rcell = ((code.astype(np.int64) * 10 + trans["down_i"].to_numpy()) * 1000 + np.clip(trans["dist_raw"].to_numpy(), 0, 20).astype(int) * 20 + (trans["fp_raw"].to_numpy() // 10).astype(int))[rp]
    res["real_yard_sd"] = float(yr.std())
    res["real_yard_sd_state_resid"] = float(kn.resid_cells(yr, rcell).std())
    (Path(args.out) / "ygain.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("sim")
    s.add_argument("--variant", default="crf4m")
    s.add_argument("--scale", type=float, default=1.0)
    s.add_argument("--yard-gain", dest="yard_gain", type=float, default=6.5)
    s.add_argument("--worlds", type=int, default=4)
    s.add_argument("--seasons", type=int, default=8)
    s.add_argument("--workers", type=int, default=3)
    s.add_argument("--seed", type=int, default=9)
    s.add_argument("--out", required=True)
    a = sub.add_parser("analyze")
    a.add_argument("--out", required=True)
    args = ap.parse_args()
    {"sim": cmd_sim, "analyze": cmd_analyze}[args.cmd](args)


if __name__ == "__main__":
    main()
