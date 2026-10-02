import numpy as np
import pandas as pd

import mod25d_variance as dv
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "artifacts" / "mod25e"


def drives_games(T):
    T = T.sort_values(["sid", "g"], kind="stable").reset_index(drop=True)
    key = (T["sid"].astype(np.int64) * 100000 + T["g"].astype(np.int64)).to_numpy()
    gcode, uniq = pd.factorize(key)
    T = T.assign(gk=gcode)
    side, poss = dv.assign_sides(T.assign(g=gcode))
    T = T.assign(poss=poss)
    sg = np.where(T["offhome"].to_numpy() == 1, 1.0, -1.0)
    T = T.assign(hm=T["sd"].to_numpy(float) * sg, sgn=sg)
    R = T[T["qtr"] <= 4]
    gp = R.groupby("poss", sort=True)
    D = pd.DataFrame({"gk": gp["gk"].first(), "sid": gp["sid"].first(), "q0": gp["qtr"].first(), "po": gp["po"].sum(), "pdf": gp["pdf"].sum(), "sgn": gp["sgn"].first(), "n": gp.size()})
    D["s"] = (D["po"] - D["pdf"]) * D["sgn"]
    Gm = pd.DataFrame({"gkey": uniq}, index=np.arange(len(uniq)))
    Gm["sid"] = Gm["gkey"] // 100000
    for q, nm in ((2, "m1"), (3, "m2"), (4, "m3")):
        Gm[nm] = T[T["qtr"] == q].groupby("gk")["hm"].first().reindex(Gm.index)
    Gm["mreg"] = T[T["qtr"] > 4].groupby("gk")["hm"].first().reindex(Gm.index)
    Gm["nposs"] = D.groupby("gk").size().reindex(Gm.index).fillna(0.0)
    return D, Gm


def per_game(T, G):
    D, Gm = drives_games(T)
    G = G.copy()
    G["gkey"] = G["sid"].astype(np.int64) * 100000 + G["g"].astype(np.int64)
    X = Gm.merge(G[["gkey", "home_score", "away_score"]], on="gkey", how="left")
    X["margin"] = X["home_score"] - X["away_score"]
    return D, X


def season_re(G):
    out = {}
    for sid, d in G.groupby("sid"):
        meta = pd.DataFrame({"season": d["sid"].to_numpy(), "home_team": d["home_team"].to_numpy(), "away_team": d["away_team"].to_numpy()})
        M = pd.Series((d["home_score"] - d["away_score"]).to_numpy(float))
        r = dv.random_effects(meta, M, iters=400)
        out[sid] = (r["strength_diff_var"], r["noise_var"])
    return out


def cat_id(D):
    po, pdf = D["po"].to_numpy(), D["pdf"].to_numpy()
    return np.where(pdf > 0, 4, np.where(po >= 6, 3, np.where(po == 3, 2, np.where(po == 0, 0, 1))))


def stats(D, X, re_vals):
    o = {}
    m = X["margin"].to_numpy(float)
    o["margin_var"] = m.var(ddof=1)
    o["strength_diff_var_RE"] = float(np.mean([v[0] for v in re_vals]))
    o["noise_var_RE"] = float(np.mean([v[1] for v in re_vals]))
    h, a = X["home_score"].to_numpy(float), X["away_score"].to_numpy(float)
    o["var_home_pts"] = h.var(ddof=1)
    o["var_away_pts"] = a.var(ddof=1)
    o["cov_home_away"] = np.cov(h, a)[0, 1]
    o["var_total_pts"] = (h + a).var(ddof=1)
    o["mean_total_pts"] = (h + a).mean()
    mr = np.where(np.isnan(X["mreg"].to_numpy(float)), m, X["mreg"].to_numpy(float))
    o["reg_margin_var"] = mr.var(ddof=1)
    o["ot_increment_var"] = (m - mr).var(ddof=1)
    n = X["nposs"].to_numpy(float)
    o["nposs_mean"] = n.mean()
    o["nposs_var"] = n.var(ddof=1)
    s = D["s"].to_numpy(float)
    po = D["po"].to_numpy(float)
    o["ppp_mean"] = po.mean()
    o["ppp_var"] = po.var()
    o["signed_ppp_var"] = s.var()
    c = cat_id(D)
    for k, nm in enumerate(["none", "other", "fg", "td", "defscore"]):
        sel = c == k
        o[f"share_{nm}"] = sel.mean()
        o[f"ppp_var_contrib_{nm}"] = float(np.sum((s[sel] - s.mean()) ** 2) / len(s))
    dd = po - D["pdf"].to_numpy(float)
    o["drive_net_var"] = dd.var()
    o["iid_margin_var"] = n.mean() * dd.var()
    o["serial_excess"] = o["reg_margin_var"] - o["iid_margin_var"]
    for q in (1, 2, 3, 4):
        d = X[f"m{q}"].to_numpy(float) if q < 4 else mr
        d = np.where(np.isnan(d), mr, d)
        prev = X[f"m{q-1}"].to_numpy(float) if q > 1 else np.zeros(len(X))
        prev = np.where(np.isnan(prev), mr, prev)
        inc = d - prev
        o[f"q{q}_inc_var"] = inc.var(ddof=1)
        o[f"q{q}_cov_with_final"] = float(np.mean((inc - inc.mean()) * (m - m.mean())))
        qd = D[D["q0"] == q]
        o[f"q{q}_nposs"] = len(qd) / len(X)
        o[f"q{q}_ppp"] = qd["po"].mean()
        o[f"q{q}_signed_var_sum"] = qd["s"].var() * len(qd) / len(X)
    return o
