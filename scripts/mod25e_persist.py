import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
NB = int(sys.argv[1]) if len(sys.argv) > 1 else 200
sys.argv = sys.argv[:1] + [str(NB)]
import mod25_generator as gen  # noqa: E402
import mod25e_catchup as cu  # noqa: E402
import mod25e_cov as cv  # noqa: E402
import mod25e_late as late  # noqa: E402

OUTD = REPO / "artifacts" / "mod25e3" / "persist"
OUT = []
rng = np.random.default_rng(59)
NWORLD, NSEAS = 8, 8
GROUPS = {"season-mean latent": slice(0, 4), "weekly deviation latent": slice(4, 8), "backup-QB flags": slice(8, 10)}


def say(s=""):
    print(s)
    OUT.append(s)


def iv(d, pt):
    return f"{pt:+.2f} [{np.percentile(d, 5):+.2f},{np.percentile(d, 95):+.2f}] pp {float((d > 0).mean()):.2f}"


def latents(seed):
    cv.patch_generator()
    fit = gen.load_fit()
    setting = dict(gen.SETTING_DEFAULTS)
    setting.update({"scale": 1.0, "drift": 1.0})
    root = np.random.SeedSequence(seed)
    ws = root.spawn(NWORLD)
    out = {}
    for w in range(NWORLD):
        lat = gen.gen_world_latents(np.random.default_rng(ws[w]), NSEAS, fit, setting)
        for s, (weekly, qb) in enumerate(lat):
            out[(w, s)] = (weekly, qb)
    return out


def game_table(gk, seeds):
    sgs = {}
    for sd in seeds:
        g = pd.read_parquet(cu.ART / f"e5_{cu.LABEL}_s{sd}" / "sim_games.parquet", columns=["game_id", "week", "home_team", "away_team", "home_score", "away_score"])
        g["h"] = g.home_team.str.split("T").str[-1].astype(int)
        g["a"] = g.away_team.str.split("T").str[-1].astype(int)
        g["w"] = g.game_id.str[1:5].astype(int)
        g["s"] = g.game_id.str[6:8].astype(int)
        g["g"] = g.game_id.str[-3:].astype(int)
        sgs[sd] = g.set_index(["w", "s", "g"])
    lats = {sd: latents(sd) for sd in seeds}
    X = np.zeros((len(gk), 10))
    M = np.zeros(len(gk))
    for i, k in enumerate(gk):
        sd, w, s, g = (int(x) for x in k.split("_"))
        r = sgs[sd].loc[(w, s, g)]
        weekly, qb = lats[sd][(w, s)]
        wk = int(r.week) - 1
        mean = weekly.mean(0)
        for j, (t, off) in enumerate(((r.h, 0), (r.a, 4))):
            X[i, off:off + 2] = mean[t]
            X[i, off + 2:off + 4] = weekly[wk, t] - mean[t]
        X[i, 8] = float(qb[wk, r.h])
        X[i, 9] = float(qb[wk, r.a])
        M[i] = r.home_score - r.away_score
    return X, M


def contributions(X, I, Hr, idx):
    Xc = np.c_[np.ones(len(idx)), X[idx]]
    res = {}
    for r in range(1, 4):
        beta, *_ = np.linalg.lstsq(Xc, Hr[idx, r], rcond=None)
        for nm, sl in GROUPS.items():
            f = X[idx][:, sl] @ beta[1:][sl]
            for q in range(r):
                res[nm] = res.get(nm, 0.0) + 2 * np.cov(I[idx, q], f)[0, 1]
        for q in range(r):
            res["total E"] = res.get("total E", 0.0) + 2 * np.cov(I[idx, q], Hr[idx, r])[0, 1]
    res["unexplained"] = res["total E"] - sum(res[n] for n in GROUPS)
    return res


def terms(D, G, idx=None):
    W = np.ones((1, G))
    I, IBS, IS, abar = cu.cross_terms(D, G, W)
    gi, qi, bi = D.gi.to_numpy(), D.qi.to_numpy(), D.bi.to_numpy()
    sg = D.s.to_numpy()
    P = np.zeros((G, 4))
    C = np.zeros((G, 4))
    A = np.zeros((G, 4))
    np.add.at(P, (gi, qi), sg * D.p.to_numpy())
    np.add.at(C, (gi, qi), sg * D.c_strength.to_numpy())
    np.add.at(A, (gi, qi), sg * abar[0][qi * 5 + bi])
    return P, C, A


def decomp(P, C, A, idx):
    out = {"CC": 0.0, "PC": 0.0, "PP": 0.0}
    for q in range(4):
        for r in range(q + 1, 4):
            Pq, Cq = P[idx, q], C[idx, q]
            Pr, Cr, Ar = P[idx, r], C[idx, r], A[idx, r]
            out["CC"] += 2 * np.cov(Cq, Cr)[0, 1]
            out["PC"] += -2 * np.cov(Pq, Cr)[0, 1] - 2 * np.cov(Cq, Pr - Ar)[0, 1]
            out["PP"] += 2 * np.cov(Pq, Pr - Ar)[0, 1]
    out["E"] = out["CC"] + out["PC"] + out["PP"]
    return out


def section_p3(Dr, Gr, Ds, Gs):
    say("## P3. E split by the evaluator's own strength adjustment c (same estimator, real and sim): E = CC + PC + PP; CC = cov of the subtracted adjustment between quarters, PC = cross of raw points with the adjustment, PP = raw points vs raw points less band mean; 2x cov summed over 6 pairs; game bootstrap")
    tr, ts = terms(Dr, Gr), terms(Ds, Gs)
    pr, ps_ = decomp(*tr, np.arange(Gr)), decomp(*ts, np.arange(Gs))
    br = [decomp(*tr, rng.integers(0, Gr, Gr)) for _ in range(NB)]
    bs = [decomp(*ts, rng.integers(0, Gs, Gs)) for _ in range(NB)]
    for k in ("CC", "PC", "PP", "E"):
        d = np.array([b[k] for b in bs]) - np.array([b[k] for b in br])
        say(f"  {k}: real {pr[k]:+.2f} sim {ps_[k]:+.2f} sim-real {iv(d, ps_[k] - pr[k])}")


def section_p4(X, I, Hr, Ds, Gs):
    say("## P4. richer regression for the sim: add squares and products of the 8 latent terms and the first-possession flag (home receives opening kickoff = offence of first drive)")
    first = pd.Series(Ds.s.to_numpy()).groupby(Ds.gi.to_numpy()).first().reindex(range(Gs)).to_numpy()
    lat = X[:, :8]
    quad = np.stack([lat[:, i] * lat[:, j] for i in range(8) for j in range(i, 8)], 1)
    X2 = np.c_[X, first, quad]
    G2 = {"linear latents+qb": slice(0, 10), "first possession": slice(10, 11), "latent squares/products": slice(11, X2.shape[1])}
    res = {}
    for r in range(1, 4):
        Xc = np.c_[np.ones(Gs), X2]
        beta, *_ = np.linalg.lstsq(Xc, Hr[:, r], rcond=None)
        for nm, sl in G2.items():
            f = X2[:, sl] @ beta[1:][sl]
            for q in range(r):
                res[nm] = res.get(nm, 0.0) + 2 * np.cov(I[:, q], f)[0, 1]
        for q in range(r):
            res["total E"] = res.get("total E", 0.0) + 2 * np.cov(I[:, q], Hr[:, r])[0, 1]
    res["unexplained"] = res["total E"] - sum(res[n] for n in G2)
    for k, v in res.items():
        say(f"  {k}: {v:+.2f} (point; in-sample, {X2.shape[1] + 1} regressors per quarter)")


def section_p5(X, I, Hr, Gs):
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.model_selection import KFold

    say("## P5. flexible latent function: gradient boosting on the 10 recorded regressors, 5-fold out-of-fold prediction of the home-net residual per later quarter (game-level folds); contribution = 2x cov(early adjusted margin, out-of-fold prediction)")
    oof = np.zeros((Gs, 4))
    for r in range(1, 4):
        for tr, te in KFold(5, shuffle=True, random_state=59).split(X):
            m = HistGradientBoostingRegressor(max_depth=4, max_iter=150, learning_rate=0.05, min_samples_leaf=200)
            m.fit(X[tr], Hr[tr, r])
            oof[te, r] = m.predict(X[te])
    def stat(idx):
        t = 0.0
        for r in range(1, 4):
            for q in range(r):
                t += 2 * np.cov(I[idx, q], oof[idx, r])[0, 1]
        return t
    d = np.array([stat(rng.integers(0, Gs, Gs)) for _ in range(NB)])
    say(f"  out-of-fold latent function contribution to sim E {iv(d, stat(np.arange(Gs)))} (linear in-sample was +4.29)")


def main():
    OUTD.mkdir(parents=True, exist_ok=True)
    ns = late.load_xq()
    real, sim = cu.drives_all(ns)
    Dr, Gr = cu.prep(ns, real)
    Ds, Gs = cu.prep(ns, sim)
    gk = sorted(Ds.gk.unique())
    W = np.ones((1, Gs))
    I, IBS, IS, abar = cu.cross_terms(Ds, Gs, W)
    gi, qi, bi = Ds.gi.to_numpy(), Ds.qi.to_numpy(), Ds.bi.to_numpy()
    e = Ds.a.to_numpy() - abar[0][qi * 5 + bi]
    home = Ds.s.to_numpy() > 0
    EH = np.zeros((Gs, 4))
    EA = np.zeros((Gs, 4))
    np.add.at(EH, (gi[home], qi[home]), e[home])
    np.add.at(EA, (gi[~home], qi[~home]), e[~home])
    Hr = EH - EA
    X, M = game_table(gk, cu.SEEDS)
    Xc = np.c_[np.ones(Gs), X]
    beta, *_ = np.linalg.lstsq(Xc, M, rcond=None)
    r2 = 1 - ((M - Xc @ beta) ** 2).sum() / ((M - M.mean()) ** 2).sum()
    say(f"## P0. latents regenerated from seeds {cu.SEEDS}, {NWORLD} worlds x {NSEAS} seasons, mod25e_cov.patch_generator joint latents, drift 1.0 (read: mod25_generator.py:474-508 gen_world_latents is replaced by mod25e_cov.py:64-99 joint_latents; run_generation mod25_generator.py:550-556)")
    say(f"  games {Gs}; check: home final margin on 10 latent regressors R2 {r2:.3f}, coefs mean-latent {np.round(beta[1:5], 2).tolist()} weekly-dev {np.round(beta[5:9], 2).tolist()} qb {np.round(beta[9:11], 2).tolist()}")
    say(f"  latent variance per regressor (sim): {np.round(X.var(0), 4).tolist()}")
    say("## P1. strength proxy reliability: game-level sum of s*c_strength (leave-game-out team mean used by the evaluator, same code for real and sim, mod25e_xq.py:91-112) on true season-mean latents")
    cs = np.zeros(Gs)
    np.add.at(cs, gi, Ds.s.to_numpy() * Ds.c_strength.to_numpy())
    A = np.c_[np.ones(Gs), X[:, :4]]
    b2, *_ = np.linalg.lstsq(A, cs, rcond=None)
    rr = 1 - ((cs - A @ b2) ** 2).sum() / ((cs - cs.mean()) ** 2).sum()
    A2 = np.c_[np.ones(Gs), X[:, :8], X[:, 8:]]
    b3, *_ = np.linalg.lstsq(A2, cs, rcond=None)
    rr2 = 1 - ((cs - A2 @ b3) ** 2).sum() / ((cs - cs.mean()) ** 2).sum()
    say(f"  proxy R2 on season-mean latents {rr:.3f}; on all 10 regressors {rr2:.3f}; proxy sd {cs.std():.3f}")
    point = contributions(X, I, Hr, np.arange(Gs))
    boots = [contributions(X, I, Hr, rng.integers(0, Gs, Gs)) for _ in range(NB)]
    say("## P2. sim E (within-state residual) cross-quarter covariance, home perspective, 2x cov summed over pairs q<r, by regressor group (per-quarter OLS of home-net residual on the 10 regressors, refit per bootstrap; game bootstrap)")
    for nm in ("total E", *GROUPS, "unexplained"):
        d = np.array([b[nm] for b in boots])
        say(f"  {nm}: {iv(d, point[nm])}")
    say(f"  gap to explain (E58 sim-real E): +14.8 [+7.2,+22.1]; real E own+opp is -1.80 (E58 A3), so each group's share of the gap = value / 14.8 if real groups contribute 0 (inferred)")
    for nm in GROUPS:
        d = np.array([b[nm] for b in boots]) / 14.8
        say(f"  share of +14.8 by {nm}: {iv(d, point[nm] / 14.8)}")
    d = np.array([b["unexplained"] for b in boots]) / 14.8
    say(f"  share by unexplained (drawn-play noise and kNN row sharing): {iv(d, point['unexplained'] / 14.8)}")
    section_p3(Dr, Gr, Ds, Gs)
    section_p4(X, I, Hr, Ds, Gs)
    section_p5(X, I, Hr, Gs)
    say(f"looks: groups 3 x bootstrap plus total and unexplained = 5 interval prints per block; bootstrap {NB}")
    (OUTD / "persist.txt").write_text("\n".join(OUT))


if __name__ == "__main__":
    main()
