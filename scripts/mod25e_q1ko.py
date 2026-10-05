import os
import sys
from pathlib import Path

os.environ.setdefault("DC_LABEL", "crHpqokgndecsmfwtj")
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
import mod25e_fpos2 as f2  # noqa: E402

f1 = f2.f1
dc = f2.dc
OUTD = dc.ART / "q1ko"
NB = 400
NP = 500
rng = np.random.default_rng(83)
SIM_Q = {0: -0.299, 1: -0.37, 2: -0.16, 3: -2.48}
OUT = []


def say(s=""):
    print(s, flush=True)
    OUT.append(s)


def loo_team(df, key, val):
    g = df.groupby(["season", key, "gk"], sort=False)[val].agg(["sum", "count"]).reset_index()
    t = g.groupby(["season", key])[["sum", "count"]].transform("sum")
    g["v"] = np.where(t["count"] - g["count"] > 0, (t["sum"] - g["sum"]) / (t["count"] - g["count"]).clip(lower=1), np.nan)
    return df.merge(g[["season", key, "gk", "v"]], on=["season", key, "gk"], how="left")["v"].to_numpy()


def split_half(df, key, val):
    g = df.groupby(["season", key, "gk"], sort=True)[val].agg(["sum", "count"]).reset_index()
    g["r"] = g.groupby(["season", key]).cumcount() % 2
    a = g.groupby(["season", key, "r"])[["sum", "count"]].sum().unstack("r").dropna()
    m0 = a[("sum", 0)] / a[("count", 0)]
    m1 = a[("sum", 1)] / a[("count", 1)]
    return float(np.corrcoef(m0, m1)[0, 1]), len(a)


def wls(X, y, W):
    out = np.empty((W.shape[0], X.shape[1]))
    for i, w in enumerate(W):
        A = X.T @ (X * w[:, None])
        out[i] = np.linalg.solve(A, X.T @ (y * w))
    return out


def design(qi, x, extra):
    cols = [(qi == q).astype(float) for q in range(4)] + [(qi == q) * x for q in range(4)] + list(extra)
    return np.column_stack(cols)


def report(name, X, y, gi, G):
    W = np.vstack([np.ones((1, G)), f1.boot(G, NB, rng)])
    B = wls(X, y, W[:, gi])
    say(f" {name}")
    for q in range(4):
        b = B[:, 4 + q]
        sim = SIM_Q[q]
        say(f"   Q{q + 1} slope {b[0]:+.3f} [{np.percentile(b[1:], 5):+.3f},{np.percentile(b[1:], 95):+.3f}] probability_negative {float((b[1:] < 0).mean()):.3f}; vs sim {sim:+.3f}: {b[0] - sim:+.3f} probability_below_sim {float((b[1:] < sim).mean()):.3f}")
    for j in range(8, X.shape[1]):
        b = B[:, j]
        say(f"   control{j - 8} coef {b[0]:+.3f} [{np.percentile(b[1:], 5):+.3f},{np.percentile(b[1:], 95):+.3f}] probability_positive {float((b[1:] > 0).mean()):.3f}")
    return B[0]


def main():
    import mod25e_late as late

    OUTD.mkdir(parents=True, exist_ok=True)
    ns = late.load_xq()
    Dr = f2.annotate2(f2.real_build(ns), ns, True).reset_index(drop=True)
    g = Dr.gk.to_numpy()
    same_next = np.r_[g[1:] == g[:-1], False]
    Dr["noff"] = np.where(same_next, np.append(Dr.off.to_numpy()[1:], ""), "")
    scored = Dr.out.isin(["TD", "FG"]).to_numpy() & Dr.nvalid.to_numpy()
    sc = scored & (Dr.noff == Dr.dfn).to_numpy()
    say(f"label {dc.LABEL}; after-score kicks valid {int(scored.sum())}, kicker-kept excluded {int((scored & ~sc).sum())}; bootstrap {NB}, permutations {NP}")
    K = Dr[sc].copy().reset_index(drop=True)
    K["rec"] = K.dfn
    K["kic"] = K.off
    K["ret"] = K.nyl0
    K["rv"] = loo_team(K, "rec", "ret")
    K["kv"] = loo_team(K, "kic", "ret")
    mu = K.ret.mean()
    for c in ("rv", "kv"):
        K[c] = (K[c] - K[c].mean()).fillna(0.0)
    r1, n1 = split_half(K, "rec", "ret")
    r2, n2 = split_half(K, "kic", "ret")
    say(f"## 0. special-teams strength: leave-game-out team-season mean receiver start yl0 over after-score kicks (mean {mu:.2f}; lower yl0 = better for receiver)")
    say(f"  odd/even game split-half r: receiving unit {r1:+.3f} (n {n1} team-seasons), kicking unit {r2:+.3f} (n {n2}); low r means the control is attenuated")
    gf = pd.read_parquet(REPO / "data" / "processed" / "game_features_pbp.parquet", columns=["game_id", "season", "home_team", "away_team", "home_score", "away_score"])
    gk = gf.season.astype(str) + "_" + gf.game_id
    M = pd.concat([pd.DataFrame({"season": gf.season, "gk": gk, "team": gf.home_team, "m": gf.home_score - gf.away_score}),
                   pd.DataFrame({"season": gf.season, "gk": gk, "team": gf.away_team, "m": gf.away_score - gf.home_score})], ignore_index=True)
    for key in ("rec", "kic"):
        gm = M.rename(columns={"team": key})
        sm = gm.groupby(["season", key])["m"].agg(["sum", "count"]).reset_index()
        own = gm.groupby(["season", key, "gk"])["m"].sum().reset_index().rename(columns={"m": "own"})
        z = K[["season", key, "gk"]].merge(sm, on=["season", key], how="left").merge(own, on=["season", key, "gk"], how="left")
        K["q" + key] = ((z["sum"] - z["own"]) / (z["count"] - 1).clip(lower=1)).to_numpy()
    K["qd"] = K.qrec - K.qkic
    K["qd"] = (K.qd - K.qd.mean()).fillna(0.0)
    qi = K.qi.to_numpy()
    x = K.nbi.to_numpy(float) - 2.0
    y = K.ret.to_numpy(float)
    td = (K.out == "TD").to_numpy()
    say("## 1. n per receiver lead band (index 0..4 = most trailing..most leading) x quarter, TD kickoffs, real")
    for q in range(4):
        say(f"  Q{q + 1}: " + " ".join(f"{int(((qi == q) & td & (K.nbi == b).to_numpy()).sum()):5d}" for b in range(5)))
    gt = pd.factorize(K.gk[td])[0]
    Gt = int(gt.max()) + 1
    qt, xt, yt = qi[td], x[td], y[td]
    ex0 = []
    ex1 = [K.rv.to_numpy()[td], K.kv.to_numpy()[td]]
    ex2 = ex1 + [K.qd.to_numpy()[td]]
    say("## 2. (a) special-teams and team-quality confound, TD kickoffs, receiver start yl0 slope per lead-band step")
    b0 = report("A0 quarter intercepts + lead slope", design(qt, xt, ex0), yt, gt, Gt)
    b1 = report("A1 + receiving-unit and kicking-unit strength (leave-game-out)", design(qt, xt, ex1), yt, gt, Gt)
    b2 = report("A2 + season margin difference receiver minus kicker (leave-game-out)", design(qt, xt, ex2), yt, gt, Gt)
    say(f"  Q1 slope shift A0->A1 {b1[4] - b0[4]:+.3f}, A0->A2 {b2[4] - b0[4]:+.3f}")
    m1 = td & (qi == 0)
    say("  Q1 TD correlation of lead band with controls: " + ", ".join(f"{c} {np.corrcoef(x[m1], K[c].to_numpy()[m1])[0, 1]:+.3f}" for c in ("rv", "kv", "qd")))
    say("## 3. (b) strategy: touchback share, and start yl0 among non-touchbacks, TD kickoffs")
    tb = K.tb.to_numpy(float)
    report("touchback share slope", design(qt, xt, ex0), tb[td], gt, Gt)
    nt = td & (tb == 0)
    gn = pd.factorize(K.gk[nt])[0]
    report("start yl0 among non-touchbacks", design(qi[nt], x[nt], []), y[nt], gn, int(gn.max()) + 1)
    say("## 4. (c) permutation null: shuffle lead band within season x quarter among TD kicks; P(null slope <= observed)")
    keys = K.season.to_numpy()[td].astype(int) * 10 + qt
    grp = [np.flatnonzero(keys == k) for k in np.unique(keys)]
    ones = np.ones((1, Gt))[:, gt]
    Xe = {0: ex0, 1: ex1}
    obs = {k: wls(design(qt, xt, Xe[k]), yt, ones)[0][4:8] for k in Xe}
    cnt = {k: np.zeros(4) for k in Xe}
    for _ in range(NP):
        xp = xt.copy()
        for ix in grp:
            xp[ix] = xt[rng.permutation(ix)]
        for k in Xe:
            cnt[k] += wls(design(qt, xp, Xe[k]), yt, ones)[0][4:8] <= obs[k]
    for k in Xe:
        say(f"  A{k}: " + "; ".join(f"Q{q + 1} {obs[k][q]:+.3f} P {cnt[k][q] / NP:.3f}" for q in range(4)))
    say("## 5. definition check: fpos2 definition keeps kicker-kept (onside-type) kicks; TD kickoffs, same slope design")
    A = Dr[scored & (Dr.out == "TD").to_numpy()].reset_index(drop=True)
    kk = (A.noff != A.dfn).to_numpy()
    xa = A.nbi.to_numpy(float) - 2.0
    ga = pd.factorize(A.gk)[0]
    report("all valid TD kicks including kicker-kept", design(A.qi.to_numpy(), xa, []), A.nyl0.to_numpy(float), ga, int(ga.max()) + 1)
    say(f"  kicker-kept TD kicks {int(kk.sum())}; by quarter " + " ".join(str(int(((A.qi == q).to_numpy() & kk).sum())) for q in range(4)) + "; Q1 kept by band index " + " ".join(str(int((kk & (A.qi == 0).to_numpy() & (A.nbi == b).to_numpy()).sum())) for b in range(5)))
    say(f"  mean next start yl0 kept {A.nyl0.to_numpy(float)[kk].mean():.1f} vs not kept {A.nyl0.to_numpy(float)[~kk].mean():.1f}")
    say("## looks: 3 designs x 4 quarters x slope, 2 strategy designs, 2 x 4 permutation slopes = 34; family declared before results")
    (OUTD / "q1ko.txt").write_text("\n".join(OUT))


if __name__ == "__main__":
    main()
