import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
ART = REPO / "artifacts" / "mod25e3"
LABEL = "crHpqokg"
SEEDS = (11, 12, 13)
OUTD = ART / "pace"
POOL = tuple(range(2009, 2018))
BURN = 2
NB = 300
PBP = REPO / "data" / "pbp" / "raw" / "20260925T202544Z"
SCRIM = ["pass", "run", "punt", "field_goal", "qb_kneel", "qb_spike"]
GF = REPO / "data" / "processed" / "game_features_pbp.parquet"


def fmt(pt, bt):
    lo, hi = np.quantile(bt, [0.025, 0.975])
    return f"{pt:+.4f} [{lo:+.4f},{hi:+.4f}] pp {float((bt > 0).mean()):.2f}"


def cell(down, qtr, sd):
    return (np.clip(np.nan_to_num(down, nan=1), 1, 4) * 10 + np.clip(qtr, 1, 4)) * 10 + np.clip(np.round(np.nan_to_num(sd) / 7), -3, 3) + 3


def real_load():
    cols = ["game_id", "play_id", "season_type", "posteam", "home_team", "qtr", "game_seconds_remaining", "play_type", "down", "score_differential", "complete_pass", "sack", "penalty"]
    gf = pd.read_parquet(GF, columns=["game_id", "home_team", "away_team", "home_score", "away_score"])
    rows = []
    for s in POOL:
        p = pd.read_parquet(PBP / f"season={s}" / "plays.parquet", columns=cols)
        p = p[(p.season_type == "REG") & (p.qtr <= 4) & p.game_seconds_remaining.notna()]
        p = p.sort_values(["game_id", "play_id"], kind="stable").reset_index(drop=True)
        g = p.game_id.to_numpy()
        gsr = p.game_seconds_remaining.to_numpy().astype(float)
        last = np.r_[g[1:] != g[:-1], True]
        p["el"] = np.where(last, gsr, gsr - np.r_[gsr[1:], 0.0])
        p["gk"] = f"{s}_" + p.game_id
        p["season"] = str(s)
        p["kocum"] = np.cumsum((p.play_type == "kickoff").to_numpy())
        rows.append(p.drop(columns=["season_type"]))
    P = pd.concat(rows, ignore_index=True)
    P["hm"] = (P.posteam == P.home_team).astype(float)
    key = P[["gk", "game_id", "season"]].drop_duplicates()
    G = key.merge(gf.rename(columns={"home_team": "ht", "away_team": "aw", "home_score": "H", "away_score": "A"}), on="game_id")
    return P, G.drop(columns=["game_id"])


def sim_load():
    rows, gm = [], []
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
            d = pd.read_parquet(f, columns=["g", "down", "qtr", "code", "sd", "el", "yards", "offhome"])
            d = d[d.qtr <= 4].sort_values("g", kind="stable").reset_index(drop=True)
            d["gk"] = f"{sd}_{w}_{s}_" + d.g.astype(int).astype(str)
            rows.append(d.drop(columns=["g"]))
            k = sg[(sg.w == w) & (sg.s == s)]
            gm.append(pd.DataFrame({"gk": f"{sd}_{w}_{s}_" + k.g.astype(str), "season": f"{sd}_{w}_{s}", "ht": k.home_team.to_numpy(), "aw": k.away_team.to_numpy(), "H": k.home_score.to_numpy(), "A": k.away_score.to_numpy()}))
    return pd.concat(rows, ignore_index=True), pd.concat(gm, ignore_index=True)


def norm_real(P):
    scr = P.play_type.isin(SCRIM).to_numpy() & P.posteam.notna().to_numpy()
    Q = P[scr].copy()
    gg, pos, kc, qq = Q.gk.to_numpy(), Q.posteam.to_numpy(), Q.kocum.to_numpy(), Q.qtr.to_numpy()
    new = np.r_[True, (gg[1:] != gg[:-1]) | (pos[1:] != pos[:-1]) | (kc[1:] != kc[:-1]) | ((qq[1:] == 3) & (qq[:-1] <= 2))]
    Q["d"] = np.cumsum(new)
    Q["rp"] = Q.play_type.isin(["pass", "run"])
    Q["pas"] = (Q.play_type == "pass").astype(float)
    Q["inc"] = ((Q.play_type == "pass") & (Q.complete_pass == 0) & (Q.sack != 1)).astype(float)
    Q["cl"] = cell(Q.down.to_numpy(float), Q.qtr.to_numpy(float), Q.score_differential.to_numpy(float))
    Q["ty"] = Q.play_type
    cat = np.select([P.play_type.isin(SCRIM), P.play_type == "kickoff", P.play_type == "no_play"], ["scrim", "kick", "noplay"], "other")
    T = pd.DataFrame({"gk": P.gk.to_numpy(), "cat": cat, "el": P.el.to_numpy()}).groupby(["gk", "cat"]).el.sum().unstack().fillna(0)
    N = P[P.penalty == 1].groupby("gk").size().rename("pen")
    return Q, T, N


def norm_sim(S):
    Q = S[np.isin(S.code.to_numpy().astype(int), [0, 1, 2, 3])].copy()
    gg, oh, qq = Q.gk.to_numpy(), Q.offhome.to_numpy(), Q.qtr.to_numpy()
    new = np.r_[True, (gg[1:] != gg[:-1]) | (oh[1:] != oh[:-1]) | ((qq[1:] == 3) & (qq[:-1] <= 2))]
    Q["d"] = np.cumsum(new)
    Q["hm"] = Q.offhome
    Q["rp"] = Q.code.isin([0, 1])
    Q["pas"] = (Q.code == 1).astype(float)
    Q["inc"] = ((Q.code == 1) & (Q.yards == 0)).astype(float)
    Q["cl"] = cell(Q.down.to_numpy(float), Q.qtr.to_numpy(float), Q.sd.to_numpy(float))
    Q["ty"] = Q.code
    cat = np.select([S.code.isin([0, 1, 2, 3, 4, 5]), S.code == 6], ["scrim", "noplay"], "other")
    T = pd.DataFrame({"gk": S.gk.to_numpy(), "cat": cat, "el": S.el.to_numpy()}).groupby(["gk", "cat"]).el.sum().unstack().fillna(0)
    N = S[S.code == 6].groupby("gk").size().rename("pen")
    return Q, T, N


def game_table(Q, N, G, cellmean):
    R = Q[Q.rp].copy()
    R["pr"] = R.pas - R.cl.map(cellmean)
    h = R[R.hm == 1].groupby("gk").agg(ph=("rp", "size"), prh=("pr", "mean"), pash=("pas", "mean"))
    a = R[R.hm == 0].groupby("gk").agg(pa=("rp", "size"), pra=("pr", "mean"), pasa=("pas", "mean"))
    pp = R[R.pas == 1]
    ih = pp[pp.hm == 1].groupby("gk").inc.mean().rename("irh")
    ia = pp[pp.hm == 0].groupby("gk").inc.mean().rename("ira")
    dh = Q[Q.hm == 1].groupby("gk").d.nunique().rename("nh")
    da = Q[Q.hm == 0].groupby("gk").d.nunique().rename("na")
    X = G.set_index("gk").join([h, a, ih, ia, dh, da, N]).dropna(subset=["ph", "pa", "irh", "ira", "nh", "na"])
    X["pen"] = X.pen.fillna(0)
    X = X.reset_index()
    rows = pd.concat([pd.DataFrame({"key": X.season + "_" + X.ht, "pf": X.H, "pq": X.A}), pd.DataFrame({"key": X.season + "_" + X.aw, "pf": X.A, "pq": X.H})])
    gs = rows.groupby("key").agg(f=("pf", "sum"), a=("pq", "sum"), n=("pf", "size"))
    rows = rows.join(gs, on="key")
    rows["lo"] = (rows.f - rows.pf) / (rows.n - 1)
    rows["ld"] = (rows.a - rows.pq) / (rows.n - 1)
    n = len(X)
    X["ho"], X["hd"], X["ao"], X["ad"] = rows.lo.to_numpy()[:n], rows.ld.to_numpy()[:n], rows.lo.to_numpy()[n:], rows.ld.to_numpy()[n:]
    X["nd"] = X.nh + X.na
    X["np_"] = X.ph + X.pa
    X["prm"] = (X.prh + X.pra) / 2
    X["irm"] = (X.irh + X.ira) / 2
    return X.dropna(subset=["ho", "hd", "ao", "ad"]).reset_index(drop=True)


def ols(Y, Z):
    Z1 = np.c_[np.ones(len(Z)), Z]
    b = np.linalg.lstsq(Z1, Y, rcond=None)[0]
    return Y - Z1 @ b, Z1 @ b


def stats(X):
    o = {}
    Y = X[["H", "A"]].to_numpy(float)
    o["cov_HA"] = float(np.cov(Y[:, 0], Y[:, 1])[0, 1])
    for c in ("nd", "np_", "pen"):
        o[f"var_{c}"] = float(X[c].var())
    o["corr_drives"] = float(np.corrcoef(X.nh, X.na)[0, 1])
    o["corr_plays"] = float(np.corrcoef(X.ph, X.pa)[0, 1])
    o["corr_pass_raw"] = float(np.corrcoef(X.pash, X.pasa)[0, 1])
    o["corr_pass_state"] = float(np.corrcoef(X.prh, X.pra)[0, 1])
    o["corr_inc"] = float(np.corrcoef(X.irh, X.ira)[0, 1])
    S = X[["ho", "hd", "ao", "ad"]].to_numpy(float)
    R0, F0 = ols(Y, S)
    base = float(np.cov(F0[:, 0], F0[:, 1])[0, 1])
    o["cov_strength"] = base
    R = R0
    for nm, cols in (("drives", ["nd"]), ("plays", ["np_"]), ("pen", ["pen"]), ("passres", ["prm"]), ("incres", ["irm"]), ("clock", ["scrim", "noplay"]), ("all5", ["nd", "np_", "pen", "prm", "irm"]), ("all7", ["nd", "np_", "pen", "prm", "irm", "scrim", "noplay"])):
        R, F = ols(Y, np.c_[S, X[cols].to_numpy(float)])
        o[f"med_{nm}"] = float(np.cov(F[:, 0], F[:, 1])[0, 1]) - base
    o["resid_after_all7"] = float(np.mean(R[:, 0] * R[:, 1]))
    o["corr_scrim_np"] = float(np.corrcoef(X.scrim, X.np_)[0, 1])
    o["sd_scrim"] = float(X.scrim.std())
    o["sd_noplay"] = float(X.noplay.std())
    o["corr_scrim_noplay"] = float(np.corrcoef(X.scrim, X.noplay)[0, 1])
    return o


def boot(X, rng):
    pt = stats(X)
    bs = [stats(X.iloc[rng.integers(0, len(X), len(X))]) for _ in range(NB)]
    return pt, {k: np.array([b[k] for b in bs]) for k in pt}


def pace_sigma(Q):
    R = Q[Q.rp].copy()
    R["mu"] = R.groupby(["ty", "qtr"]).el.transform("mean")
    R["par"] = R.groupby("gk").cumcount() % 2
    ag = R.groupby(["gk", "par"])[["el", "mu"]].sum().unstack().dropna()
    r0, r1 = (ag[("el", i)] / ag[("mu", i)] for i in (0, 1))
    c = float(np.cov(r0, r1)[0, 1] / (r0.mean() * r1.mean()))
    return float(np.sqrt(np.log1p(max(c, 0.0)))), float(np.corrcoef(r0, r1)[0, 1])


def main():
    OUTD.mkdir(parents=True, exist_ok=True)
    P, G = real_load()
    S, Gs = sim_load()
    Qr, Tr, Nr = norm_real(P)
    Qs, Ts, Ns = norm_sim(S)
    cellmean = Qr[Qr.rp].groupby("cl").pas.mean()
    Xr = game_table(Qr, Nr, G, cellmean)
    Xs = game_table(Qs, Ns, Gs, cellmean)
    Xr = Xr.join(Tr[["scrim", "noplay"]], on="gk")
    Xs = Xs.join(Ts[["scrim", "noplay"]], on="gk")
    rng = np.random.default_rng(9)
    res = {"real": boot(Xr, rng), "sim": boot(Xs, rng)}
    L = [f"games real {len(Xr)} sim {len(Xs)}"]
    for k in res["real"][0]:
        d = res["sim"][1][k] - res["real"][1][k]
        L.append(f"{k:18s} real {res['real'][0][k]:+.4f} sim {res['sim'][0][k]:+.4f} diff {fmt(res['sim'][0][k] - res['real'][0][k], d)}")
    L.append(f"incompletion rate real {Qr[Qr.pas == 1].inc.mean():.4f} sim {Qs[Qs.pas == 1].inc.mean():.4f}")
    L.append(f"pass share real {Qr[Qr.rp].pas.mean():.4f} sim {Qs[Qs.rp].pas.mean():.4f}")
    L.append("game clock by category (per-game mean/sd)")
    for nm, T, N in (("real", Tr, Nr), ("sim", Ts, Ns)):
        T = T.copy()
        T["total"] = T.sum(axis=1)
        pn = N.reindex(T.index).fillna(0)
        L.append(nm + " " + " ".join(f"{c}:{T[c].mean():.0f}/{T[c].std():.1f}" for c in T.columns) + f" pen/g {pn.mean():.2f} sd {pn.std():.2f}")
    for nm, Q in (("real", Qr), ("sim", Qs)):
        sg, rr = pace_sigma(Q)
        L.append(f"{nm} pace sigma est {sg:.4f} half-corr {rr:.3f} total-plays sd {Q[Q.rp].groupby('gk').size().std():.2f}")
    (OUTD / "pace.txt").write_text("\n".join(L))
    print("\n".join(L))


main()
