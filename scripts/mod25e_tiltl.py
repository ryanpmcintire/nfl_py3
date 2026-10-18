import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
os.environ.setdefault("PASS_MODE", "epa")

OUT = REPO / "artifacts" / "mod25e3" / "tiltl"
FIT = OUT / "fit.json"
COLS = (1, 3)
NAMES = {1: "conv", 3: "sack"}
TRAIN_SEASONS = tuple(range(2009, 2018))


def enabled():
    return os.environ.get("TILTL") == "1"


def lower_priority():
    import ctypes

    ctypes.windll.kernel32.SetPriorityClass(ctypes.windll.kernel32.GetCurrentProcess(), 0x4000)


def league_means_for(seasons):
    g = pd.read_parquet(REPO / "data" / "processed" / "game_features_pbp.parquet")
    g = g[(g["game_type"] == "REG") & g["season"].isin(list(seasons))]
    off = pd.concat([g["home_off_epa_per_play"], g["away_off_epa_per_play"]]).mean()
    de = pd.concat([g["home_def_epa_per_play"], g["away_def_epa_per_play"]]).mean()
    return float(off), float(de)


def collect(per_season, kq, seed):
    import mod25d_variance as dv
    import mod25e_crG as crG
    import mod25e_draw as dr
    import sim04_engine as sim

    G, cfg = dr.build()
    pbp = sim.load_reg_seasons(tuple(dv.TRAIN))
    trans = sim.build_transition_frame(pbp)
    G["phase"] = trans["phase"].to_numpy()
    G["sc"] = trans["sc_raw"].to_numpy()
    G["tm"] = trans["time_raw"].to_numpy()
    G["oto"] = trans["off_to_raw"].to_numpy()
    G["dto"] = trans["def_to_raw"].to_numpy()
    a = G["tables"]["arrays"]
    at = dv.c25.attrs_from(pbp, trans)
    D, I = dv.rate_defs(trans, at)
    ok, _ = dv.team_keys(pbp, trans)
    season = pd.to_numeric(pd.Series(ok).astype(str).str[:4], errors="coerce").fillna(-1).to_numpy().astype(int)
    T = np.asarray(G["ns"]["TILT_T"], dtype=np.float32)
    rng = np.random.default_rng(seed)
    K = int(sim.K_STATE)
    qrow, qk, nbs, w0s = [], [], [], []
    for k in COLS:
        for s in TRAIN_SEASONS:
            pool = np.flatnonzero(D[:, k] & (season == s))
            pick = rng.choice(pool, size=min(per_season, len(pool)), replace=False)
            for r in pick:
                nb, key = dr.neighbors_of(G, int(r), kq)
                nb = nb[nb != r]
                w = dr.stage_weights(G, nb, int(r), key, float(crG.EPA_M))
                base = w["team"] * w["ipw"] * w["dk"] * w["epa"]
                pad = kq - len(nb)
                nbs.append(np.concatenate([nb, -np.ones(pad, dtype=nb.dtype)]) if pad > 0 else nb[:kq])
                w0s.append(np.concatenate([base, np.zeros(pad)]) if pad > 0 else base[:kq])
                qrow.append(int(r))
                qk.append(k)
    NB = np.array(nbs, dtype=np.int64)
    W0 = np.array(w0s, dtype=np.float32)
    qrow = np.array(qrow)
    qk = np.array(qk)
    safe = np.where(NB >= 0, NB, 0)
    Tn = T[safe][:, :, list(COLS)]
    DN = D[safe, qk[:, None]] & (NB >= 0)
    IN = I[safe, qk[:, None]] & DN
    SN = np.where(NB >= 0, season[safe], -2)
    y = I[qrow, qk].astype(np.float64)
    return dict(K=K, qrow=qrow, qk=qk, season=season[qrow], o=np.asarray(a["off_row"], dtype=np.float64)[qrow], d=np.asarray(a["def_row"], dtype=np.float64)[qrow], Tn=Tn, W0=W0, DN=DN, IN=IN, SN=SN, y=y)


def fold_weights(Q, hold, lo, ld, theta, mask_q):
    W0 = Q["W0"][mask_q].astype(np.float64)
    valid = (Q["SN"][mask_q] != hold) & (Q["SN"][mask_q] >= -1) if hold is not None else (Q["SN"][mask_q] >= -1)
    keep = valid & (np.cumsum(valid, axis=1) <= Q["K"])
    ao = np.array([theta[0], theta[2]])
    ad = np.array([theta[1], theta[3]])
    o = (Q["o"][mask_q] - lo)[:, None]
    d = (Q["d"][mask_q] - ld)[:, None]
    lin = (Q["Tn"][mask_q].astype(np.float64) * (ao[None, None, :] * o[:, :, None] + ad[None, None, :] * d[:, :, None])).sum(axis=2)
    return W0 * keep * np.exp(lin), keep


def probs(Q, hold, lo, ld, theta, mask_q):
    w, keep = fold_weights(Q, hold, lo, ld, theta, mask_q)
    wd = w * Q["DN"][mask_q]
    num = (wd * Q["IN"][mask_q]).sum(axis=1)
    den = wd.sum(axis=1)
    return np.clip(num / np.where(den > 0, den, 1.0), 1e-6, 1 - 1e-6)


def loglik(Q, hold, lo, ld, theta, mask_q):
    p = probs(Q, hold, lo, ld, theta, mask_q)
    y = Q["y"][mask_q]
    return float((y * np.log(p) + (1 - y) * np.log(1 - p)).sum())


def slopes(p, o, d, y):
    X = np.column_stack([np.ones(len(o)), o, d])
    bp = np.linalg.lstsq(X, p, rcond=None)[0]
    by = np.linalg.lstsq(X, y, rcond=None)[0]
    return bp[1:], by[1:]


def cmd_fit(args):
    from scipy.optimize import minimize

    import mod25_generator as gen

    lower_priority()
    OUT.mkdir(parents=True, exist_ok=True)
    Q = collect(args.per_season, args.kq, args.seed)
    cur = json.loads((REPO / "artifacts" / "mod25d" / "tilt_coef.json").read_text())
    cur_theta = np.array([cur["conv"]["off"], cur["conv"]["def"], cur["sack"]["off"], cur["sack"]["def"]])
    cur_lo, cur_ld = gen.league_means()
    res = {"K": Q["K"], "per_season": args.per_season, "queries": int(len(Q["y"])), "current": {"theta": cur_theta.tolist(), "lo": cur_lo, "ld": cur_ld}, "folds": {}}
    held = {"p": [], "pc": [], "p0": [], "o": [], "d": [], "y": [], "k": []}
    for hold in list(TRAIN_SEASONS) + [None]:
        tr = (Q["season"] != hold) if hold is not None else np.ones(len(Q["y"]), dtype=bool)
        te = (Q["season"] == hold) if hold is not None else tr
        lo, ld = league_means_for([s for s in TRAIN_SEASONS if s != hold])

        def neg(th):
            return -loglik(Q, hold, lo, ld, th, tr)

        r = minimize(neg, np.zeros(4), method="BFGS", options={"gtol": 1e-3, "maxiter": 60})
        th = r.x
        ll_in = -float(r.fun)
        ll_out = loglik(Q, hold, lo, ld, th, te)
        ll_cur = loglik(Q, hold, cur_lo, cur_ld, cur_theta, te)
        ll_zero = loglik(Q, hold, lo, ld, np.zeros(4), te)
        n_in, n_out = int(tr.sum()), int(te.sum())
        ll_in0 = loglik(Q, hold, lo, ld, np.zeros(4), tr)
        res["folds"]["all" if hold is None else str(hold)] = {
            "theta": th.tolist(), "lo": lo, "ld": ld, "n_in": n_in, "n_out": n_out,
            "ll_in": ll_in, "ll_in_zero": ll_in0, "ll_out": ll_out, "ll_out_current": ll_cur, "ll_out_zero": ll_zero,
        }
        if hold is not None:
            held["p"].append(probs(Q, hold, lo, ld, th, te))
            held["pc"].append(probs(Q, hold, cur_lo, cur_ld, cur_theta, te))
            held["p0"].append(probs(Q, hold, lo, ld, np.zeros(4), te))
            held["o"].append(Q["o"][te])
            held["d"].append(Q["d"][te])
            held["y"].append(Q["y"][te])
            held["k"].append(Q["qk"][te])
        print(hold, np.round(th, 4), round(ll_in, 1), round(ll_out, 1), round(ll_cur, 1), round(ll_zero, 1), flush=True)
    h = {k: np.concatenate(v) for k, v in held.items()}
    tr_out = {}
    for k in COLS:
        m = h["k"] == k
        sp, sy = slopes(h["p"][m], h["o"][m], h["d"][m], h["y"][m])
        spc, _ = slopes(h["pc"][m], h["o"][m], h["d"][m], h["y"][m])
        sp0, _ = slopes(h["p0"][m], h["o"][m], h["d"][m], h["y"][m])
        sdo, sdd = float(h["o"][m].std()), float(h["d"][m].std())
        tr_out[NAMES[k]] = {
            "sd_off_rating": sdo, "sd_def_rating": sdd,
            "slope_off": {"fold": sp[0], "current": spc[0], "zero": sp0[0], "real": sy[0]},
            "slope_def": {"fold": sp[1], "current": spc[1], "zero": sp0[1], "real": sy[1]},
            "rate_sd_off": {"fold": abs(sp[0]) * sdo, "current": abs(spc[0]) * sdo, "zero": abs(sp0[0]) * sdo, "real": abs(sy[0]) * sdo},
            "rate_sd_def": {"fold": abs(sp[1]) * sdd, "current": abs(spc[1]) * sdd, "zero": abs(sp0[1]) * sdd, "real": abs(sy[1]) * sdd},
        }
    res["transfer"] = json.loads(json.dumps(tr_out, default=float))
    FIT.write_text(json.dumps(res, indent=1))


def cmd_report(args):
    r = json.loads(FIT.read_text())
    f = {k: v for k, v in r["folds"].items() if k != "all"}
    th = np.array([v["theta"] for v in f.values()])
    nm = ["conv AO", "conv AD", "sack AO", "sack AD"]
    print("K", r["K"], "queries", r["queries"], "current", np.round(r["current"]["theta"], 3))
    for i, n in enumerate(nm):
        print(n, "mean", round(th[:, i].mean(), 3), "sd", round(th[:, i].std(ddof=1), 3), "min", round(th[:, i].min(), 3), "max", round(th[:, i].max(), 3), "signs+", int((th[:, i] > 0).sum()), "all-fit", round(r["folds"]["all"]["theta"][i], 3))
    ni = sum(v["n_in"] for v in f.values())
    no = sum(v["n_out"] for v in f.values())
    print("in-sample LL/play (mean over folds)", round(sum(v["ll_in"] for v in f.values()) / ni, 5), "zero in", round(sum(v["ll_in_zero"] for v in f.values()) / ni, 5))
    print("held-out LL/play fold", round(sum(v["ll_out"] for v in f.values()) / no, 5), "current", round(sum(v["ll_out_current"] for v in f.values()) / no, 5), "zero", round(sum(v["ll_out_zero"] for v in f.values()) / no, 5))
    d = np.array([v["ll_out"] - v["ll_out_current"] for v in f.values()])
    d0 = np.array([v["ll_out"] - v["ll_out_zero"] for v in f.values()])
    print("per-fold held-out LL fold minus current", np.round(d, 1), "positive folds", int((d > 0).sum()))
    print("per-fold held-out LL fold minus zero", np.round(d0, 1), "positive folds", int((d0 > 0).sum()))
    print(json.dumps(r["transfer"], indent=1))


def install(season=None):
    import mod25d_variance as dv

    fit = json.loads(FIT.read_text())
    f = fit["folds"].get(str(season)) or fit["folds"]["all"]
    ns = dv._G["ns"]
    ao = np.array(ns["TILT_AO"], dtype=np.float64)
    ad = np.array(ns["TILT_AD"], dtype=np.float64)
    th = f["theta"]
    ao[1], ad[1], ao[3], ad[3] = th[0], th[1], th[2], th[3]
    ns["TILT_AO"], ns["TILT_AD"] = ao, ad
    ns["TILT_LO"], ns["TILT_LD"] = f["lo"], f["ld"]


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("fit")
    p.add_argument("--per-season", type=int, default=3000)
    p.add_argument("--kq", type=int, default=260)
    p.add_argument("--seed", type=int, default=5)
    sub.add_parser("report")
    args = ap.parse_args()
    {"fit": cmd_fit, "report": cmd_report}[args.cmd](args)


if __name__ == "__main__":
    main()
