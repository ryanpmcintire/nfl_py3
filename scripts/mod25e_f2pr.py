import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

import mod25e_endgame as eg  # noqa: E402
import sim09_f2 as f2  # noqa: E402

OUT = f2.OUT
MODEL = OUT / "policy_pr.joblib"
META = OUT / "policy_pr.json"
HGB = dict(max_depth=4, max_iter=150, learning_rate=0.05, min_samples_leaf=300, l2_regularization=1.0, random_state=1)
CALL_QTR = 4
CALL_LOW = 30.0
CALL_HIGH = 120.0
EDGES = (30.0, 60.0, 90.0, 120.0)
DECILES = 5


def window_cut(R):
    qt = R["qtr"].to_numpy()
    gs = R["gsr"].to_numpy(float)
    win = ((qt == 4) & (gs > 0) & (gs <= f2.u4g.WARN_AT[4])) | ((qt == 2) & (gs > f2.u4g.FLOOR[2]) & (gs <= f2.u4g.WARN_AT[2]))
    W = R[win]
    return float(eg.hurry_cut(W["code"], W["yards"], W["el"]))


def stopped(code, yards, flip, scored, otu, dtu, el, cut):
    code = np.asarray(code)
    s = np.asarray(flip).astype(bool) | np.asarray(scored).astype(bool) | ((code == 1) & (np.asarray(yards) == 0)) | (code == 5) | (np.asarray(otu) > 0) | (np.asarray(dtu) > 0)
    s = s | ((code <= 1) & (np.asarray(el, float) <= cut))
    return s


def real_prevrun(d, cut):
    d = d.sort_values(["game_id", "play_id"], kind="stable").reset_index(drop=True)
    run = pd.Series((~stopped(d["code"], d["yards"], d["flip"], (d["po"] + d["pdf"]) > 0, d["otu"], d["dtu"], d["el"], cut)).astype(float))
    half = np.where(d["qtr"].to_numpy() >= 3, 2, 1)
    key = d["game_id"].astype(str) + "|" + pd.Series(half).astype(str)
    prev = run.groupby(key.to_numpy()).shift(1).fillna(0.0)
    d["prun"] = prev.to_numpy()
    return d


def feats_pr(down, dist, yl, sd, gsr, qtr, code, oto, dto, stop, prun):
    base = f2.feats(down, dist, yl, sd, gsr, qtr, code, oto, dto, stop)
    return np.column_stack([base, np.asarray(prun, dtype=float)])


def xy(d, pr):
    d, X, y = f2.frame_xy(d)
    if pr:
        X = np.column_stack([X, d["prun"].to_numpy(float)])
    return d, X, y


def sim_prevrun(log, playlog, qtr, cut, yards_arr, cls_arr):
    if not log:
        return 0.0
    row = log[-1]
    if (int(row[5]) >= 3) != (int(qtr) >= 3):
        return 0.0
    idx = int(row[11])
    shift = float(playlog[-1][2]) if playlog and int(playlog[-1][1]) == idx else 0.0
    cl = int(cls_arr[idx])
    return float(not stopped(row[6], yards_arr[idx] + shift, row[9], (row[7] + row[8]) > 0, cl & 1, (cl >> 1) & 1, row[10], cut))


POOL = {}
START_TO = 3


def cache_pool():
    if "p" not in POOL:
        POOL["p"] = pd.read_parquet(f2.u4g.OUT / "pool.parquet", columns=["otu", "dtu"])
    return POOL["p"]


def extra_fields(df, ysh, yards_arr):
    pool = cache_pool()
    idx = df["idx"].to_numpy().astype(int)
    otu = pool["otu"].to_numpy()[idx]
    dtu = pool["dtu"].to_numpy()[idx]
    yards = yards_arr[idx] + ysh
    g = df["g"].to_numpy()
    q = df["qtr"].to_numpy()
    oh = df["offhome"].to_numpy().astype(bool)
    oto = np.zeros(len(df))
    dto = np.zeros(len(df))
    home = away = START_TO
    ph = 0
    cg = None
    for i in range(len(df)):
        if g[i] != cg:
            cg = g[i]
            home = away = START_TO
            ph = 1 if q[i] <= 2 else 2
        h = 1 if q[i] <= 2 else 2
        if q[i] <= 4 and h != ph:
            home = away = START_TO
            ph = h
        if oh[i]:
            oto[i], dto[i] = home, away
            home = max(0, home - otu[i])
            away = max(0, away - dtu[i])
        else:
            oto[i], dto[i] = away, home
            away = max(0, away - otu[i])
            home = max(0, home - dtu[i])
    cut = eg.HC["cut"]
    st = stopped(df["code"].to_numpy(), yards, df["flip"].to_numpy(), (df["po"].to_numpy() + df["pdf"].to_numpy()) > 0, otu, dtu, df["el"].to_numpy(), -1.0 if cut is None else cut)
    half = np.where(q >= 3, 2, 1)
    key = pd.Series(g.astype(np.int64) * 10 + half)
    pst = pd.Series(st.astype(float)).groupby(key.to_numpy()).shift(1).fillna(1.0).to_numpy()
    out = df.copy()
    out["yards"] = yards
    out["lo_otu"] = otu
    out["lo_dtu"] = dtu
    out["oto"] = oto
    out["dto"] = dto
    out["pstop"] = pst
    return out


def ll(P, y):
    return f2.logloss(P, y)


def cmd_fit(args):
    import joblib
    from sklearn.ensemble import HistGradientBoostingClassifier

    R = pd.read_parquet(OUT / "real_fit.parquet")
    cut = window_cut(R)
    R = real_prevrun(R, cut)
    print("derived hurry cut", cut, flush=True)
    tot = {"old": [], "new": []}
    cal = []
    rel = []
    for s in sorted(R["season"].unique()):
        tr = R[R["season"] != s]
        te = R[R["season"] == s]
        res = {}
        for nm, pr in (("old", False), ("new", True)):
            _, Xa, ya = xy(tr, pr)
            dt, Xb, yb = xy(te, pr)
            clf = HistGradientBoostingClassifier(**HGB).fit(Xa, ya)
            P = clf.predict_proba(Xb)
            res[nm] = P
            tot[nm].append((ll(P, yb) * len(yb), len(yb)))
        sub = (dt["qtr"] == CALL_QTR) & (dt["gsr"] > CALL_LOW) & (dt["gsr"] <= CALL_HIGH) & (dt["sd"] <= 0) & (dt["oto"] > 0)
        for run in (1.0, 0.0):
            m = (sub & (dt["prun"] == run)).to_numpy()
            if m.any():
                cal.append({"season": int(s), "prun": run, "n": int(m.sum()), "real": float((dt["otu"].to_numpy()[m] > 0).mean()), "old": float((res["old"][m, 1] + res["old"][m, 3]).mean()), "new": float((res["new"][m, 1] + res["new"][m, 3]).mean())})
        pn = res["new"][:, 1] + res["new"][:, 3]
        rel.append(pd.DataFrame({"p": pn, "o": (dt["otu"].to_numpy() > 0).astype(float), "po": res["old"][:, 1] + res["old"][:, 3], "t": dt["gsr"].to_numpy(), "q": dt["qtr"].to_numpy(), "run": dt["prun"].to_numpy()}))
        print("season", int(s), "old", round(ll(res["old"], yb), 5), "new", round(ll(res["new"], yb), 5), flush=True)
    for nm in ("old", "new"):
        print("LOSO pooled 4-class log loss", nm, round(sum(a for a, _ in tot[nm]) / sum(b for _, b in tot[nm]), 5))
    C = pd.DataFrame(cal)
    g = C.groupby("prun").apply(lambda x: pd.Series({"n": x["n"].sum(), "real": np.average(x["real"], weights=x["n"]), "old": np.average(x["old"], weights=x["n"]), "new": np.average(x["new"], weights=x["n"])}), include_groups=False)
    print("Q4 gsr 30-120 trailing/tied offence with timeouts, off-call rate by previous-clock state (LOSO)")
    print(g.round(4).to_string())
    Rl = pd.concat(rel)
    late = Rl[(Rl["q"] == CALL_QTR) & (Rl["t"] > CALL_LOW) & (Rl["t"] <= CALL_HIGH) & (Rl["run"] == 1.0)]
    late = late.assign(b=pd.cut(late["t"], [CALL_LOW - 1, *EDGES[1:]]))
    print(late.groupby("b", observed=True).agg(n=("o", "size"), real=("o", "mean"), new=("p", "mean"), old=("po", "mean")).round(4).to_string())
    Rl["bin"] = pd.qcut(Rl["p"].rank(method="first"), DECILES, labels=False)
    print("reliability of any offence-timeout probability (new, all snaps, quintiles)")
    print(Rl.groupby("bin").agg(pred=("p", "mean"), real=("o", "mean"), n=("o", "size")).round(4).to_string())
    _, X, y = xy(R, True)
    clf = HistGradientBoostingClassifier(**HGB).fit(X, y)
    joblib.dump(clf, MODEL)
    META.write_text(json.dumps({"cut": cut}))
    print("saved", MODEL)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("fit")
    a = ap.parse_args()
    {"fit": cmd_fit}[a.cmd](a)


if __name__ == "__main__":
    main()
