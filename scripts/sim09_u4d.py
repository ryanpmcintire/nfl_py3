import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
import sim04_engine as eng  # noqa: E402

OUT = REPO / "artifacts" / "sim09" / "u4d"
NV = Path("C:/Users/Ryan/AppData/Local/Temp/claude/F--Repos-nfl-py3/f6b7873f-565c-422a-9dee-b6963fcaad49/scratchpad/nv")
COLS = ["game_id", "play_id", "season", "down", "ydstogo", "yardline_100", "score_differential", "qtr", "game_seconds_remaining",
        "posteam_timeouts_remaining", "defteam_timeouts_remaining", "play_type", "air_yards", "yards_gained", "qb_kneel", "qb_spike"]
CUR = {"a": eng.SCORE_INNER_SCALE, "b": eng.SCORE_OUTER_SCALE, "T": eng.SCALE_TIME, "K": eng.K_NEIGHBORS, "H": 0}


def real_plays(seasons):
    d = pd.concat([pd.read_parquet(NV / f"pbp_{s}.parquet", columns=COLS) for s in seasons], ignore_index=True)
    d = d[d["play_type"].isin(["run", "pass"]) & d["down"].isin([1, 2, 3]) & (d["qb_kneel"].fillna(0) == 0) & (d["qb_spike"].fillna(0) == 0)]
    d = d.dropna(subset=["ydstogo", "yardline_100", "score_differential", "game_seconds_remaining", "yards_gained"]).reset_index(drop=True)
    q, g = d["qtr"].to_numpy(), d["game_seconds_remaining"].to_numpy()
    d["phase"] = eng.vectorized_phase(q, g)
    d["time_raw"] = eng.vectorized_time_raw(q, g)
    d["half"] = (q >= 3).astype(int)
    d["pass"] = (d["play_type"] == "pass").astype(float)
    return d


def feats(d, p):
    sc = np.clip(d["score_differential"].to_numpy(), -eng.SCORE_CLIP, eng.SCORE_CLIP)
    ab = np.abs(sc)
    mag = np.minimum(ab, eng.SCORE_INNER) / p["a"] + np.maximum(ab - eng.SCORE_INNER, 0) / p["b"]
    ph = d["phase"].to_numpy()
    tw = np.isin(ph, eng.LATE_PHASES).astype(float)
    X = np.column_stack([d["ydstogo"].to_numpy() / eng.SCALE_YDSTOGO, d["yardline_100"].to_numpy() / eng.SCALE_FP, np.sign(sc) * mag,
                         d["time_raw"].to_numpy() / p["T"], d["posteam_timeouts_remaining"].fillna(3).to_numpy() * tw,
                         d["defteam_timeouts_remaining"].fillna(3).to_numpy() * tw])
    if p["H"]:
        X = np.column_stack([X, d["half"].to_numpy() * 1000.0 * (ph == 0)])
    return X


def crps_rows(x, y):
    k = x.shape[1]
    s = np.sort(x, axis=1)
    w = 2 * np.arange(1, k + 1) - k - 1
    return np.abs(x - y[:, None]).mean(1) - 0.5 * (2 * (s * w).sum(1) / k**2)


def knn_all(fit, val, p, passonly=False):
    if passonly:
        fit, val = fit[fit["pass"] == 1], val[val["pass"] == 1]
        fit = fit.dropna(subset=["air_yards"])
        val = val.dropna(subset=["air_yards"])
    Xf, Xv = feats(fit, p), feats(val, p)
    K = int(p["K"])
    ph_f, ph_v = fit["phase"].to_numpy(), val["phase"].to_numpy()
    dn_f, dn_v = fit["down"].to_numpy(), val["down"].to_numpy()
    nb = np.zeros((len(val), K), dtype=np.int64)
    for dn in (1, 2, 3):
        for ph in range(5):
            qm = np.flatnonzero((dn_v == dn) & (ph_v == ph))
            if len(qm) == 0:
                continue
            pm = np.flatnonzero((dn_f == dn) & eng.phase_pool_mask(ph_f, ph))
            tr = cKDTree(Xf[pm])
            _, ind = tr.query(Xv[qm], k=K)
            nb[qm] = pm[ind]
    return fit, val, nb


def evaluate(fit, val, p):
    f1, v1, nb = knn_all(fit, val, p)
    yf = f1["yards_gained"].to_numpy()
    pf = f1["pass"].to_numpy()
    K = nb.shape[1]
    pp = (pf[nb].sum(1) + 0.5) / (K + 1)
    y = v1["pass"].to_numpy()
    ll = -np.mean(y * np.log(pp) + (1 - y) * np.log(1 - pp))
    cy = crps_rows(yf[nb], v1["yards_gained"].to_numpy()).mean()
    f2, v2, nb2 = knn_all(fit, val, p, passonly=True)
    ca = crps_rows(f2["air_yards"].to_numpy()[nb2], v2["air_yards"].to_numpy()).mean()
    return {"ll": float(ll), "crps_y": float(cy), "crps_air": float(ca)}


GRID = [0.25, 0.5, 0.7, 1.0, 1.4, 2.0, 4.0]


def descend(fit, val, key, cache, sweeps=2):
    p = dict(CUR)

    def ev(q):
        t = tuple(sorted(q.items()))
        if t not in cache:
            cache[t] = evaluate(fit, val, q)
        return cache[t]

    best = ev(p)[key]
    for _ in range(sweeps):
        for name in ("a", "b", "T", "K"):
            base = p[name]
            for m in GRID:
                q = dict(p)
                q[name] = int(round(base * m)) if name == "K" else base * m
                if name == "K" and q[name] < 5:
                    continue
                v = ev(q)[key]
                if v < best - 1e-9:
                    best, p = v, q
        q = dict(p)
        q["H"] = 1 - p["H"]
        v = ev(q)[key]
        if v < best - 1e-9:
            best, p = v, q
    return p, best


def implied(fit, test, p):
    sel = (test["qtr"] == 3) & test["score_differential"].between(5, 9)
    t = test[sel]
    ff, vv, nb = knn_all(fit, t, p)
    air_f = fit["air_yards"].to_numpy()
    sd_f = fit["score_differential"].to_numpy()
    h_f = fit["half"].to_numpy()
    ds = sd_f[nb] - vv["score_differential"].to_numpy()[:, None]
    return {"n": len(vv), "abs_ds": float(np.abs(ds).mean()), "nb_lead_le0": float((sd_f[nb] <= 0).mean()), "nb_first_half": float((h_f[nb] == 0).mean()),
            "nb_pass": float(fit["pass"].to_numpy()[nb].mean()), "nb_air": float(np.nanmean(air_f[nb])), "nb_ypp": float(fit["yards_gained"].to_numpy()[nb].mean()),
            "real_pass": float(vv["pass"].mean()), "real_air": float(vv["air_yards"].mean()), "real_ypp": float(vv["yards_gained"].mean())}


def cmd_real(args):
    OUT.mkdir(parents=True, exist_ok=True)
    tune_fit, tune_val = real_plays(range(2009, 2014)), real_plays((2014, 2015))
    fit, test = real_plays(range(2009, 2016)), real_plays((2016, 2017))
    cache = {}
    res = {"current": CUR}
    sols = {}
    for key in ("ll", "crps_y", "crps_air"):
        p, b = descend(tune_fit, tune_val, key, cache)
        sols[key] = p
        print(key, "tuned", p, "val", b, flush=True)
    for name, p in [("current", CUR)] + list(sols.items()):
        res[name] = {"p": p, "test": evaluate(fit, test, p), "implied_q3_lead7": implied(fit, test, p)}
        print(name, res[name], flush=True)
    allq3 = test[test["qtr"] == 3]
    res["real_q3_all"] = {"pass": float(allq3["pass"].mean()), "air": float(allq3["air_yards"].mean()), "ypp": float(allq3["yards_gained"].mean())}
    (OUT / "real_kernel.json").write_text(json.dumps(res, indent=1, default=float))
    print(res["real_q3_all"])


def cmd_sim(args):
    import mod25d_variance as dv

    dv.sim.PBP_SNAPSHOT_DIR = dv.m25.SNAP
    pbp = dv.sim.load_reg_seasons(tuple(dv.TRAIN))
    tr = dv.sim.build_transition_frame(pbp)
    q = pbp[["game_id", "play_id", "qtr"]].drop_duplicates(["game_id", "play_id"])
    tr = tr[["game_id", "play_id", "sc_raw", "time_raw", "phase"]].merge(q, on=["game_id", "play_id"], how="left")
    out = {}
    for name, path in [("crzk", REPO / "artifacts/sim09/u3d/play_crzk"), ("crzh", REPO / "artifacts/sim09/u3c/play_crzh")]:
        fs = sorted(path.glob("play_*_*.parquet"))
        P = pd.concat([pd.read_parquet(f) for f in fs], ignore_index=True)
        P = P[P["code"].isin([0, 1]) & P["down"].isin([1, 2, 3])]
        i = P["idx"].to_numpy(np.int64)
        P = P.assign(p_sc=tr["sc_raw"].to_numpy()[i], p_t=tr["time_raw"].to_numpy()[i], p_q=tr["qtr"].to_numpy()[i])
        P["s_t"] = eng.vectorized_time_raw(P["qtr"].to_numpy(), P["gsr"].to_numpy())
        P["ds"] = P["p_sc"] - P["sd"]
        P["dt"] = P["p_t"] - P["s_t"]
        P["same_half"] = ((P["p_q"] >= 3) == (P["qtr"] >= 3)).astype(float)
        P["p_le0"] = (P["p_sc"] <= 0).astype(float)
        cells = {"Q3 lead 5-9": (P["qtr"] == 3) & P["sd"].between(5, 9), "Q3 lead 7": (P["qtr"] == 3) & (P["sd"] == 7),
                 "Q3 trail 7": (P["qtr"] == 3) & (P["sd"] == -7), "Q3 tied": (P["qtr"] == 3) & (P["sd"] == 0),
                 "Q1 lead 7": (P["qtr"] == 1) & (P["sd"] == 7), "all Q3": P["qtr"] == 3}
        rows = {}
        for lab, m in cells.items():
            s = P[m]
            rows[lab] = {"n": int(len(s)), "mean_drawn_sd": float(s["p_sc"].mean()), "mean_sim_sd": float(s["sd"].mean()), "abs_ds": float(s["ds"].abs().mean()),
                         "frac_ds0": float((s["ds"] == 0).mean()), "frac_drawn_le0": float(s["p_le0"].mean()), "abs_dt_s": float(s["dt"].abs().mean()),
                         "same_half": float(s["same_half"].mean())}
        out[name] = rows
        print(name, json.dumps(rows, indent=1), flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "sim_bandwidth.json").write_text(json.dumps(out, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["sim", "real"])
    a = ap.parse_args()
    {"sim": cmd_sim, "real": cmd_real}[a.cmd](a)


if __name__ == "__main__":
    main()
