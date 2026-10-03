import argparse
import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
ART = REPO / "artifacts" / "mod25e3"
LABEL = "crHpqokg"
SEEDS = (11, 12, 13)
OUTD = ART / "couple"
POOL = tuple(range(2009, 2018))
BURN = 2
NB = 200
PBP = REPO / "data" / "pbp" / "raw" / "20260925T202544Z"
TAIL = 0.999
PLAYS = ["pass", "run", "punt", "field_goal", "qb_kneel", "qb_spike"]


def fmt(pt, bt):
    lo, hi = np.quantile(bt, [0.025, 0.975])
    return f"{pt:+.3f} [{lo:+.3f},{hi:+.3f}] pp {float((bt > 0).mean()):.2f}"


def real_tables():
    gf = pd.read_parquet(REPO / "data" / "processed" / "game_features_pbp.parquet", columns=["game_id", "home_team", "away_team", "home_score", "away_score"]).set_index("game_id")
    cols = ["game_id", "play_id", "season_type", "posteam", "home_team", "qtr", "game_seconds_remaining", "yardline_100", "play_type"]
    dr, gm = [], []
    for s in POOL:
        p = pd.read_parquet(PBP / f"season={s}" / "plays.parquet", columns=cols)
        p = p[(p.season_type == "REG") & (p.qtr <= 4) & p.game_seconds_remaining.notna()]
        p = p.sort_values(["game_id", "play_id"], kind="stable").reset_index(drop=True)
        g = p.game_id.to_numpy()
        gsr = p.game_seconds_remaining.to_numpy().astype(float)
        last = np.r_[g[1:] != g[:-1], True]
        p["dl"] = np.where(last, gsr, gsr - np.r_[gsr[1:], 0.0])
        ko = (p.play_type == "kickoff").to_numpy()
        p["kocum"] = np.cumsum(ko)
        p = p[p.play_type.isin(PLAYS) & p.posteam.notna()].copy()
        g = p.game_id.to_numpy()
        pos = p.posteam.to_numpy()
        kc = p.kocum.to_numpy()
        q = p.qtr.to_numpy()
        new = np.r_[True, (g[1:] != g[:-1]) | (pos[1:] != pos[:-1]) | (kc[1:] != kc[:-1]) | ((q[1:] == 3) & (q[:-1] <= 2))]
        p["d"] = np.cumsum(new)
        gp = p.groupby("d", sort=True)
        D = pd.DataFrame({"gk": f"{s}_" + gp.game_id.first(), "q0": gp.qtr.first(), "n": gp.size(), "t": gp.dl.sum(), "yl0": gp.yardline_100.first(),
                          "hm": (gp.posteam.first() == gp.home_team.first()).astype(float)})
        dr.append(D)
        k = gf.loc[gf.index.isin(p.game_id.unique())]
        gm.append(pd.DataFrame({"gk": f"{s}_" + k.index, "season": str(s), "ht": k.home_team.to_numpy(), "aw": k.away_team.to_numpy(), "H": k.home_score.to_numpy(), "A": k.away_score.to_numpy()}))
    return pd.concat(dr, ignore_index=True), pd.concat(gm, ignore_index=True)


def sim_tables():
    dr, gm = [], []
    for sd in SEEDS:
        sdir = ART / f"e5_{LABEL}_s{sd}"
        sg = pd.read_parquet(sdir / "sim_games.parquet", columns=["game_id", "home_team", "away_team", "home_score", "away_score"])
        sg["w"] = sg.game_id.str[1:5].astype(int)
        sg["s"] = sg.game_id.str[6:8].astype(int)
        sg["g"] = sg.game_id.str[-3:].astype(int)
        for f in sorted(glob.glob(str(sdir / "play_*_*.parquet"))):
            w, s = (int(x) for x in Path(f).stem.split("_")[1:])
            if s < BURN:
                continue
            d = pd.read_parquet(f)
            d = d[d.qtr <= 4].sort_values(["g"], kind="stable").reset_index(drop=True)
            g = d.g.to_numpy()
            oh = d.offhome.to_numpy()
            q = d.qtr.to_numpy()
            new = np.r_[True, (g[1:] != g[:-1]) | (oh[1:] != oh[:-1]) | ((q[1:] == 3) & (q[:-1] <= 2))]
            d["d"] = np.cumsum(new)
            gp = d.groupby("d", sort=True)
            key = f"{sd}_{w}_{s}_"
            D = pd.DataFrame({"gk": key + gp.g.first().astype(int).astype(str), "q0": gp.qtr.first(), "n": gp.size(), "t": gp.el.sum(), "yl0": gp.yl.first(), "hm": gp.offhome.first()})
            dr.append(D)
            k = sg[(sg.w == w) & (sg.s == s)]
            gm.append(pd.DataFrame({"gk": key + k.g.astype(str), "season": f"{sd}_{w}_{s}", "ht": k.home_team.to_numpy(), "aw": k.away_team.to_numpy(), "H": k.home_score.to_numpy(), "A": k.away_score.to_numpy()}))
    return pd.concat(dr, ignore_index=True), pd.concat(gm, ignore_index=True)


def clip_t(D):
    D = D.copy()
    D["t"] = D.t.clip(0, D.t.quantile(TAIL))
    return D


def game_frame(D, G):
    h = D[D.hm == 1].groupby("gk").agg(nh=("n", "size"), ph=("n", "sum"), th=("t", "sum"))
    a = D[D.hm == 0].groupby("gk").agg(na=("n", "size"), pa=("n", "sum"), ta=("t", "sum"))
    X = G.set_index("gk").join(h).join(a).dropna().reset_index()
    h_rows = pd.DataFrame({"gk": X.gk, "key": X.season + "_" + X.ht, "pf": X.H, "pq": X.A})
    a_rows = pd.DataFrame({"gk": X.gk, "key": X.season + "_" + X.aw, "pf": X.A, "pq": X.H})
    rows = pd.concat([h_rows, a_rows])
    gs = rows.groupby("key").agg(f=("pf", "sum"), a=("pq", "sum"), n=("pf", "size"))
    rows = rows.join(gs, on="key")
    rows["lo"] = (rows.f - rows.pf) / (rows.n - 1)
    rows["ld"] = (rows.a - rows.pq) / (rows.n - 1)
    hr = rows.iloc[: len(X)]
    ar = rows.iloc[len(X):]
    X["ho"], X["hd"], X["ao"], X["ad"] = hr.lo.to_numpy(), hr.ld.to_numpy(), ar.lo.to_numpy(), ar.ld.to_numpy()
    return X.dropna().reset_index(drop=True)


def resid(Y, Z):
    Z1 = np.c_[np.ones(len(Z)), Z]
    b = np.linalg.lstsq(Z1, Y, rcond=None)[0]
    return Y - Z1 @ b, Z1 @ b


def decomp(X):
    Y = X[["H", "A"]].to_numpy(float)
    blocks = [("strength", X[["ho", "hd", "ao", "ad"]].to_numpy(float)), ("drives", X[["nh", "na"]].to_numpy(float)),
              ("plays", X[["ph", "pa"]].to_numpy(float)), ("time", X[["th", "ta"]].to_numpy(float))]
    out = {}
    R = Y - Y.mean(0)
    Zprev = None
    for nm, B in blocks:
        Bo = B - B.mean(0) if Zprev is None else resid(B, Zprev)[0]
        Rn, F = resid(R, Bo)
        out[nm] = float(np.mean((F[:, 0] - F[:, 0].mean()) * (F[:, 1] - F[:, 1].mean())))
        R = Rn
        Zprev = B if Zprev is None else np.c_[Zprev, B]
    out["residual"] = float(np.mean(R[:, 0] * R[:, 1]))
    out["total"] = float(np.cov(Y[:, 0], Y[:, 1], bias=True)[0, 1])
    out["cov_drives_ab"] = float(np.cov(X.nh, X.na)[0, 1])
    out["cov_plays_ab"] = float(np.cov(X.ph, X.pa)[0, 1])
    out["cov_time_ab"] = float(np.cov(X.th, X.ta)[0, 1])
    out["corr_drives_ab"] = float(np.corrcoef(X.nh, X.na)[0, 1])
    out["var_drive_diff"] = float(np.var(X.nh - X.na))
    out["slope_H_on_ph_minus_pa"] = float(np.polyfit(X.ph - X.pa, X.H, 1)[0])
    out["corr_ppp_ab"] = float(np.corrcoef(X.H / X.ph, X.A / X.pa)[0, 1])
    return out


def boot(fn, X, rng):
    n = len(X)
    pt = fn(X)
    bs = [fn(X.iloc[rng.integers(0, n, n)]) for _ in range(NB)]
    return pt, {k: np.array([b[k] for b in bs]) for k in pt}


def cmd_part1(src):
    rng = np.random.default_rng(7)
    res = {}
    for nm, (D, G) in src.items():
        X = game_frame(D, G)
        res[nm] = boot(decomp, X, rng)
        print(nm, "games", len(X), flush=True)
    L = []
    for nm in res:
        L.append(nm + " " + " ".join(f"{k}={v:+.3f}" for k, v in res[nm][0].items()))
    L.append("sim minus real (95% boot, probability_positive)")
    for k in res["sim"][0]:
        d = res["sim"][1][k] - res["real"][1][k]
        L.append(f"  {k:24s} {fmt(res['sim'][0][k] - res['real'][0][k], d)}")
    return L


def pair_table(D, col):
    D = D.assign(i=D.groupby("gk").cumcount())
    D["tmk"] = D.gk + "_" + D.hm.astype(int).astype(str)
    D = D.sort_values(["tmk", "i"], kind="stable")
    x = D[col].to_numpy(float)
    k = D.tmk.to_numpy()
    same = k[1:] == k[:-1]
    return pd.DataFrame({"gk": D.gk.to_numpy()[1:][same], "a": x[:-1][same], "b": x[1:][same]})


def corr_boot(P, rng):
    u, inv = np.unique(P.gk.to_numpy(), return_inverse=True)
    S = np.zeros((len(u), 6))
    for j, v in enumerate([np.ones(len(P)), P.a, P.b, P.a ** 2, P.b ** 2, P.a * P.b]):
        S[:, j] = np.bincount(inv, weights=np.asarray(v, float), minlength=len(u))

    def c(T):
        n, sa, sb, saa, sbb, sab = T
        return (sab / n - sa * sb / n ** 2) / np.sqrt((saa / n - (sa / n) ** 2) * (sbb / n - (sb / n) ** 2))

    return c(S.sum(0)), np.array([c(S[rng.integers(0, len(u), len(u))].sum(0)) for _ in range(NB)])


def ratio_boot(num_by_game, den_by_game, rng):
    n = len(num_by_game)
    pt = num_by_game.sum() / den_by_game.sum()
    bs = []
    for _ in range(NB):
        ix = rng.integers(0, n, n)
        bs.append(num_by_game[ix].sum() / den_by_game[ix].sum())
    return pt, np.array(bs)


def cmd_part2(src):
    rng = np.random.default_rng(8)
    R = {}
    for nm, (D, G) in src.items():
        D = clip_t(D)
        R[nm] = {}
        R[nm]["persist_corr_secs_same_team_consecutive"] = corr_boot(pair_table(D, "t"), rng)
        R[nm]["persist_corr_plays_same_team_consecutive"] = corr_boot(pair_table(D, "n"), rng)
        g = D.gk.to_numpy()
        same = g[1:] == g[:-1]
        h = D.hm.to_numpy()
        ret = (h[1:] == h[:-1])[same].astype(float)
        pn = D.n.to_numpy()[:-1][same]
        pq = D.q0.to_numpy()[:-1][same]
        u, inv = np.unique(g[1:][same], return_inverse=True)
        for lab, m in (("all", np.ones(len(ret), bool)), ("prev plays<=3", pn <= 3), ("prev plays4-8", (pn >= 4) & (pn <= 8)), ("prev plays>8", pn > 8),
                       ("prev q1", pq == 1), ("prev q2", pq == 2), ("prev q3", pq == 3), ("prev q4", pq == 4)):
            num = np.bincount(inv[m], weights=ret[m], minlength=len(u))
            den = np.bincount(inv[m], minlength=len(u)).astype(float)
            R[nm]["same_team_again " + lab] = ratio_boot(num, den + 1e-12, rng)
    L = []
    for k in R["sim"]:
        s, r = R["sim"][k], R["real"][k]
        L.append(f"  {k:46s} sim {s[0]:+.4f} real {r[0]:+.4f} diff {fmt(s[0] - r[0], s[1] - r[1])}")
    return L


def q_game_table(D):
    D = D.assign(i=D.groupby("gk").cumcount())
    out = pd.DataFrame(index=pd.Index(D.gk.unique(), name="gk"))
    for q in (1, 2, 3, 4):
        m = D[D.q0 == q].groupby("gk")
        out[f"d{q}"] = m.size()
        out[f"p{q}"] = m.n.sum()
        out[f"t{q}"] = m.t.sum()
    out = out.fillna(0.0)
    h2 = D[D.q0 >= 3].groupby("gk").head(1).set_index("gk")
    f1 = D.groupby("gk").head(1).set_index("gk")
    out["has_h2"] = out.index.isin(h2.index).astype(float)
    out["same"] = (h2.hm.reindex(out.index) == f1.hm.reindex(out.index)).astype(float).fillna(0.0)
    out["fp"] = h2.yl0.reindex(out.index).fillna(0.0)
    out["fn"] = h2.n.reindex(out.index).fillna(0.0)
    out["ft"] = h2.t.reindex(out.index).fillna(0.0)
    q3 = D[D.q0 == 3].groupby("gk")
    out["q3_ret"] = q3.hm.apply(lambda s: float((s.to_numpy()[1:] == s.to_numpy()[:-1]).sum()))
    out["q3_pairs"] = q3.size().sub(1).clip(lower=0)
    out = out.fillna(0.0)
    return out


def cmd_part3(src):
    rng = np.random.default_rng(9)
    T = {nm: q_game_table(clip_t(D)) for nm, (D, G) in src.items()}
    spec = [(f"drives per game q{q}", f"d{q}", None) for q in (1, 2, 3, 4)]
    spec += [(f"plays per game q{q}", f"p{q}", None) for q in (1, 2, 3, 4)]
    spec += [(f"plays per drive q{q}", f"p{q}", f"d{q}") for q in (1, 2, 3, 4)]
    spec += [(f"secs per play q{q}", f"t{q}", f"p{q}") for q in (1, 2, 3, 4)]
    spec += [(f"secs per drive q{q}", f"t{q}", f"d{q}") for q in (1, 2, 3, 4)]
    spec += [("h2 first drive by the game-first team", "same", "has_h2"), ("h2 first drive start yardline", "fp", "has_h2"),
             ("h2 first drive plays", "fn", "has_h2"), ("h2 first drive secs", "ft", "has_h2"), ("q3 same team again share", "q3_ret", "q3_pairs")]
    L = []
    for lab, a, b in spec:
        r = {}
        for nm, X in T.items():
            num = X[a].to_numpy(float)
            den = np.ones(len(X)) if b is None else X[b].to_numpy(float)
            r[nm] = ratio_boot(num, den, rng)
        s, rr = r["sim"], r["real"]
        L.append(f"  {lab:40s} sim {s[0]:9.4f} real {rr[0]:9.4f} diff {fmt(s[0] - rr[0], s[1] - rr[1])}")
    return L


def decomp2(X):
    Y = X[["H", "A"]].to_numpy(float)
    S = X[["ho", "hd", "ao", "ad"]].to_numpy(float)
    Dv = X[["nh", "na"]].to_numpy(float)
    Tm = X[["th", "ta"]].to_numpy(float)
    out = {}
    R = Y - Y.mean(0)
    Zp = None
    for nm, B in (("strength", S), ("drives", Dv), ("time", Tm)):
        Bo = B - B.mean(0) if Zp is None else resid(B, Zp)[0]
        R, F = resid(R, Bo)
        out[nm] = float(np.mean((F[:, 0] - F[:, 0].mean()) * (F[:, 1] - F[:, 1].mean())))
        Zp = B if Zp is None else np.c_[Zp, B]
    out["residual"] = float(np.mean(R[:, 0] * R[:, 1]))
    out["total"] = float(np.cov(Y[:, 0], Y[:, 1], bias=True)[0, 1])
    Z = np.c_[np.ones(len(X)), S, Dv, Tm]
    b = np.linalg.lstsq(Z, Y[:, 0], rcond=None)[0]
    out["pts_per_100s_own_time"] = float(b[-2] * 100)
    out["pts_per_100s_opp_time"] = float(b[-1] * 100)
    out["corr_time_ab"] = float(np.corrcoef(X.th, X.ta)[0, 1])
    out["sd_th_minus_ta"] = float(np.std(X.th - X.ta))
    out["sd_th_plus_ta"] = float(np.std(X.th + X.ta))
    out["mean_th_plus_ta"] = float(np.mean(X.th + X.ta))
    out["drive_secs_cv"] = float(np.std(X.th / X.nh) / np.mean(X.th / X.nh))
    return out


def cmd_part4(src):
    rng = np.random.default_rng(11)
    res = {nm: boot(decomp2, game_frame(D, G), rng) for nm, (D, G) in src.items()}
    L = [nm + " " + " ".join(f"{k}={v:+.3f}" for k, v in res[nm][0].items()) for nm in res]
    for k in res["sim"][0]:
        d = res["sim"][1][k] - res["real"][1][k]
        L.append(f"  {k:28s} {fmt(res['sim'][0][k] - res['real'][0][k], d)}")
    ko = []
    for s in POOL:
        p = pd.read_parquet(PBP / f"season={s}" / "plays.parquet", columns=["game_id", "play_id", "season_type", "posteam", "play_type"])
        p = p[(p.season_type == "REG")].sort_values(["game_id", "play_id"], kind="stable").reset_index(drop=True)
        k = np.flatnonzero((p.play_type == "kickoff").to_numpy())
        nx = np.minimum(k + 1, len(p) - 1)
        kp = p.posteam.to_numpy()[k]
        ok = p.game_id.to_numpy()[k] == p.game_id.to_numpy()[nx]
        q = p.play_type.to_numpy()[nx]
        sc = ok & np.isin(q, ["pass", "run", "punt", "field_goal", "qb_kneel"])
        ko.append((sc.sum(), (sc & (p.posteam.to_numpy()[nx] == kp)).sum(), len(k)))
    a = np.array(ko).sum(0)
    L.append(f"real kickoffs {a[2]} followed by scrimmage {a[0]}, same team as kicker {a[1]} ({a[1] / a[0]:.4f}) per game {a[1] / 2304:.3f}")
    return L


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("part", choices=["p1", "p2", "p3", "p4", "all"])
    a = ap.parse_args()
    OUTD.mkdir(parents=True, exist_ok=True)
    Dr, Gr = real_tables()
    Ds, Gs = sim_tables()
    Dr, Ds = clip_t(Dr), clip_t(Ds)
    src = {"real": (Dr, Gr), "sim": (Ds, Gs)}
    print("real drives", len(Dr), "sim drives", len(Ds), flush=True)
    L = []
    if a.part in ("p1", "all"):
        L += ["PART1 cov(H,A) sequential orthogonal blocks: strength(leave-game-out), drives, plays, time, residual"] + cmd_part1(src)
    if a.part in ("p2", "all"):
        L += ["PART2 possession persistence and alternation"] + cmd_part2(src)
    if a.part in ("p3", "all"):
        L += ["PART3 by starting quarter"] + cmd_part3(src)
    if a.part == "p4":
        L += ["PART4 without plays block (real plays exclude penalty rows)"] + cmd_part4(src)
    (OUTD / f"couple_{a.part}.txt").write_text("\n".join(L))
    print("\n".join(L))


main()
