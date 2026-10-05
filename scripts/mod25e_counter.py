import os
import sys
from pathlib import Path

os.environ.setdefault("CU_LABEL", "crHpqokgndecsmfwtjo2")
os.environ.setdefault("CU_SEEDS", "11,12,13")

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
import mod25e_catchup as cu  # noqa: E402
import mod25e_late as late  # noqa: E402

ART = REPO / "artifacts" / "mod25e3"
LABEL = cu.LABEL
SEEDS = cu.SEEDS
NB = 200
CH = 25
OUT = []
LOOKS = [0]
rng = np.random.default_rng(86)
PAIRS = cu.PAIRS
PNAME = cu.PNAME
COMPS = ["count", "startfp", "leadstate", "rest"]
RAW = REPO / "data" / "pbp" / "raw" / "20260925T202544Z"


def say(s=""):
    print(s)
    OUT.append(s)


def agg(d):
    d = d.copy()
    d["odd"] = d.groupby(["gk", "side"]).cumcount() % 2
    t = d.groupby(["gk", "side", "q", "odd"]).epa.agg(["sum", "size"]).reset_index()
    a = d.groupby(["gk", "side"])[["tk", "dk", "sk"]].first().reset_index()
    return t, a


def real_epa():
    ts, as_ = [], []
    for s in range(2009, 2018):
        f = pd.read_parquet(RAW / f"season={s}" / "plays.parquet", columns=["game_id", "play_id", "season_type", "home_team", "away_team", "posteam", "play_type", "epa", "qtr"])
        f = f[(f.season_type == "REG") & f.play_type.isin(["run", "pass"]) & (f.qtr <= 4) & f.posteam.notna()].sort_values(["game_id", "play_id"], kind="stable")
        home = f.posteam == f.home_team
        d = pd.DataFrame({"gk": str(s) + "_" + f.game_id.to_numpy(), "side": np.where(home, 0, 1), "q": f.qtr.astype(int).to_numpy() - 1, "epa": f.epa.fillna(0.0).to_numpy(),
                          "tk": f.posteam.to_numpy(), "dk": np.where(home, f.away_team, f.home_team), "sk": str(s)})
        t, a = agg(d)
        ts.append(t)
        as_.append(a)
    return pd.concat(ts, ignore_index=True), pd.concat(as_, ignore_index=True)


def sim_epa():
    import sim09_f2 as f2

    dv = f2.dv
    dv.sim.PBP_SNAPSHOT_DIR = dv.m25.SNAP
    pbp = dv.sim.load_reg_seasons(tuple(dv.TRAIN))
    tr = dv.sim.build_transition_frame(pbp)
    ep = pd.concat([pd.read_parquet(f"{dv.m25.SNAP}/season={y}/plays.parquet", columns=["game_id", "play_id", "epa"]) for y in dv.TRAIN]).drop_duplicates(["game_id", "play_id"])
    m = tr[["game_id", "play_id"]].merge(ep, on=["game_id", "play_id"], how="left")
    assert len(m) == len(tr)
    epa_by_idx = m["epa"].fillna(0.0).to_numpy()
    ts, as_ = [], []
    for sd in SEEDS:
        root = ART / f"e5_{LABEL}_s{sd}"
        sg = pd.read_parquet(root / "sim_games.parquet", columns=["game_id", "home_team", "away_team"])
        sg["w"] = sg.game_id.str[1:5].astype(int)
        sg["s"] = sg.game_id.str[6:8].astype(int)
        sg["g"] = sg.game_id.str[-3:].astype(int)
        for (w, s), tm in sg.groupby(["w", "s"]):
            if s < late.BURN:
                continue
            d = pd.read_parquet(root / f"play_{w}_{s}.parquet", columns=["g", "qtr", "offhome", "code", "idx"])
            d = d[(d.qtr <= 4) & d.code.isin([0, 1])].sort_values("g", kind="stable")
            t = tm.set_index("g")
            gi = d.g.astype(int).to_numpy()
            ht = t.home_team.reindex(gi).to_numpy()
            at = t.away_team.reindex(gi).to_numpy()
            oh = d.offhome.to_numpy() == 1
            key = f"{sd}_{w}_{s}"
            fr = pd.DataFrame({"gk": key + "_" + d.g.astype(int).astype(str).to_numpy(), "side": np.where(oh, 0, 1), "q": d.qtr.astype(int).to_numpy() - 1,
                               "epa": epa_by_idx[d.idx.astype(int).to_numpy()], "tk": np.where(oh, ht, at).astype(str), "dk": np.where(oh, at, ht).astype(str), "sk": key})
            a, b = agg(fr)
            ts.append(a)
            as_.append(b)
    return pd.concat(ts, ignore_index=True), pd.concat(as_, ignore_index=True)


def loo(g, key, col):
    s = g.groupby(["sk", key])[col].transform("sum")
    k = g.groupby(["sk", key])[col].transform("size")
    return ((s - g[col]) / (k - 1).clip(lower=1)).to_numpy()


def build(D, G, gk, T, A):
    side = (D.s.to_numpy() < 0).astype(int)
    gi, qi, bi = D.gi.to_numpy(), D.qi.to_numpy(), D.bi.to_numpy()
    flat = (gi * 2 + side) * 4 + qi
    Asum = np.bincount(flat, weights=D.a.to_numpy(), minlength=G * 8).reshape(G, 2, 4)
    Mcell = np.bincount(flat * 5 + bi, minlength=G * 40).astype(float).reshape(G, 2, 4, 5)
    Acell = np.bincount(flat * 5 + bi, weights=D.a.to_numpy(), minlength=G * 40).reshape(G, 2, 4, 5)
    Ccomp = np.stack([np.bincount(flat, weights=D["c_" + c].to_numpy(), minlength=G * 8).reshape(G, 2, 4) for c in COMPS], 3)
    ti = gk.get_indexer(T.gk)
    keep = ti >= 0
    S = np.zeros((G, 2, 4, 2))
    N = np.zeros((G, 2, 4, 2))
    np.add.at(S, (ti[keep], T.side.to_numpy()[keep], T.q.to_numpy()[keep], T.odd.to_numpy()[keep]), T["sum"].to_numpy()[keep])
    np.add.at(N, (ti[keep], T.side.to_numpy()[keep], T.q.to_numpy()[keep], T.odd.to_numpy()[keep]), T["size"].to_numpy()[keep])
    ai = gk.get_indexer(A.gk)
    ak = ai >= 0
    Aa = A[ak].assign(gi=ai[ak])
    tot_s, tot_n = S.sum((2, 3)), N.sum((2, 3))
    m = tot_s[Aa.gi.to_numpy(), Aa.side.to_numpy()] / np.maximum(tot_n[Aa.gi.to_numpy(), Aa.side.to_numpy()], 1)
    Aa = Aa.assign(m=m)
    Aa = Aa.assign(m=Aa.m - Aa.groupby("sk").m.transform("mean"))
    Aa["o"] = loo(Aa, "tk", "m")
    Aa["d"] = loo(Aa, "dk", "m")
    adj = np.zeros((G, 2))
    adj[Aa.gi.to_numpy(), Aa.side.to_numpy()] = (Aa.o + Aa.d).to_numpy() + (m - Aa.m.to_numpy())
    f = {}
    for h in (0, 1):
        So, No = S[..., h].sum(2, keepdims=True) - S[..., h], N[..., h].sum(2, keepdims=True) - N[..., h]
        f[h] = np.where(No > 0, So / np.maximum(No, 1), 0.0) - adj[:, :, None]
    ff = {h: S[..., h].sum(2) / np.maximum(N[..., h].sum(2), 1) - adj for h in (0, 1)}
    return dict(G=G, Asum=Asum, Mcell=Mcell, Acell=Acell, Ccomp=Ccomp, fA=f[0], fB=f[1], fAf=ff[0], fBf=ff[1], adj=adj, nside=int(ak.sum()))


def pcov(W, X, Y):
    sw = W.sum(1)
    w3 = W[:, :, None]
    mx = (w3 * X).sum((1, 2)) / (2 * sw)
    my = (w3 * Y).sum((1, 2)) / (2 * sw)
    return (w3 * X * Y).sum((1, 2)) / (2 * sw) - mx * my


def wc(W, x, y):
    return cu.wcov(W, np.broadcast_to(x, W.shape), np.broadcast_to(y, W.shape))


def chunk(ds, Wc):
    G = ds["G"]
    c = len(Wc)
    Asum, Mcell, Acell = ds["Asum"], ds["Mcell"], ds["Acell"]
    abar = (Wc @ Acell.sum(1).reshape(G, 20)) / np.maximum(Wc @ Mcell.sum(1).reshape(G, 20), 1e-9)
    Esum = Asum[None] - np.einsum("gsqk,cqk->cgsq", Mcell, abar.reshape(c, 4, 5))
    I = Asum[:, 0] - Asum[:, 1]
    Es = Esum[:, :, 0] - Esum[:, :, 1]
    V = pcov(Wc, ds["fAf"], ds["fBf"])
    den = np.stack([pcov(Wc, ds["fA"][:, :, q], ds["fB"][:, :, q]) for q in range(4)], 1)
    kI = np.stack([pcov(Wc, Asum[:, :, q], ds["fB"][:, :, q]) for q in range(4)], 1) / den
    kE = np.stack([pcov(Wc, Esum[..., q], ds["fB"][:, :, q]) for q in range(4)], 1) / den
    kC = np.stack([np.stack([pcov(Wc, ds["Ccomp"][:, :, q, h], ds["fB"][:, :, q]) for q in range(4)], 1) for h in range(len(COMPS))], 2) / den[:, :, None]
    r = {"V": V[:, None], "kI": kI, "kE": kE}
    r["tot_obs"] = np.stack([2 * wc(Wc, I[:, q], I[:, rr]) for q, rr in PAIRS], 1)
    r["E_obs"] = np.stack([2 * wc(Wc, I[:, q], Es[:, :, rr]) for q, rr in PAIRS], 1)
    r["tot_imp"] = np.stack([4 * kI[:, q] * kI[:, rr] * V for q, rr in PAIRS], 1)
    r["E_imp"] = np.stack([4 * kI[:, q] * kE[:, rr] * V for q, rr in PAIRS], 1)
    own = np.stack([sum(2 * wc(Wc, I[:, q], Esum[:, :, 0, rr]) for q in range(rr)) for rr in range(1, 4)], 1)
    opp = np.stack([sum(2 * wc(Wc, I[:, q], -Esum[:, :, 1, rr]) for q in range(rr)) for rr in range(1, 4)], 1)
    r["own_obs"], r["opp_obs"] = own, opp
    r["side_imp"] = np.stack([sum(2 * kI[:, q] * kE[:, rr] * V for q in range(rr)) for rr in range(1, 4)], 1)
    Xc = ds["Ccomp"][:, 0] - ds["Ccomp"][:, 1]
    r["ch_obs"] = np.stack([np.stack([sum(2 * wc(Wc, I[:, q], Xc[:, rr, h]) for q in range(rr)) for h in range(len(COMPS))], 1) for rr in range(1, 4)], 1)
    r["ch_imp"] = np.stack([np.stack([sum(4 * kI[:, q] * kC[:, rr, h] * V for q in range(rr)) for h in range(len(COMPS))], 1) for rr in range(1, 4)], 1)
    sets = [([q], [rr]) for q, rr in PAIRS] + [([0, 1], [2, 3])]
    mo, ms, mv, mi = [], [], [], []
    for eq, lr in sets:
        e = {X: sum(Esum[..., X, q] for q in eq) for X in (0, 1)}
        l = {X: sum(Esum[..., X, rr] for rr in lr) for X in (0, 1)}
        own_c = 0.5 * (wc(Wc, e[0], l[0]) + wc(Wc, e[1], l[1]))
        opp_c = 0.5 * (wc(Wc, e[0], l[1]) + wc(Wc, e[1], l[0]))
        var_e = 0.5 * (wc(Wc, e[0], e[0]) + wc(Wc, e[1], e[1]))
        mo.append(own_c)
        ms.append(opp_c)
        mv.append(var_e)
        mi.append(sum(kE[:, q] for q in eq) * sum(kE[:, rr] for rr in lr) * V)
    r["m_own"], r["m_opp"], r["m_var"], r["m_imp"] = (np.stack(x, 1) for x in (mo, ms, mv, mi))
    return r


def run(ds, seed_rng):
    G = ds["G"]
    W = np.vstack([np.ones((1, G)), seed_rng.multinomial(G, np.ones(G) / G, size=NB).astype(float)])
    parts = [chunk(ds, W[i:i + CH]) for i in range(0, len(W), CH)]
    return {k: np.concatenate([p[k] for p in parts], 0) for k in parts[0]}


def f1(a):
    LOOKS[0] += 1
    return f"{a[0]:+.2f} [{np.percentile(a[1:], 5):+.2f},{np.percentile(a[1:], 95):+.2f}]"


def f3(a, w=4):
    LOOKS[0] += 1
    return f"{a[0]:+.{w}f} [{np.percentile(a[1:], 5):+.{w}f},{np.percentile(a[1:], 95):+.{w}f}]"


def fd(a, w=2):
    LOOKS[0] += 1
    return f"{a[0]:+.{w}f} [{np.percentile(a[1:], 5):+.{w}f},{np.percentile(a[1:], 95):+.{w}f}] pp {float((a[1:] > 0).mean()):.2f}"


def report(R, S):
    say("## 1. E predicted from game-level form. k = pts per EPA/play of form (side-game, strength-adjusted drive points per quarter on the other-quarters EPA/play form, instrumented odd/even plays so sampling noise cancels); implied cov = 4 k_q k_r V (2x convention), V = true game form variance (odd/even cov). Counter-force = implied - observed. Game bootstrap 5-95.")
    say(f"  V (EPA/play^2): real {f3(R['V'][:, 0], 5)} | sim {f3(S['V'][:, 0], 5)}")
    for q in range(4):
        say(f"  Q{q + 1} k total pts per unit form: real {f1(R['kI'][:, q])} | sim {f1(S['kI'][:, q])}; k residual (E) pts: real {f1(R['kE'][:, q])} | sim {f1(S['kE'][:, q])}")
    for lab, imp, obs in (("E (within-band residual)", "E_imp", "E_obs"), ("total (state + residual)", "tot_imp", "tot_obs")):
        say(f"  {lab}: pair | real implied | real observed | real counter-force | sim implied | sim observed | sim counter-force | sim-real counter")
        for i, n in enumerate(PNAME):
            rc, sc = R[imp][:, i] - R[obs][:, i], S[imp][:, i] - S[obs][:, i]
            say(f"    {n} {f1(R[imp][:, i])} | {f1(R[obs][:, i])} | {f1(rc)} | {f1(S[imp][:, i])} | {f1(S[obs][:, i])} | {f1(sc)} | {fd(sc - rc)}")
        rc, sc = R[imp].sum(1) - R[obs].sum(1), S[imp].sum(1) - S[obs].sum(1)
        say(f"    sum {f1(R[imp].sum(1))} | {f1(R[obs].sum(1))} | {f1(rc)} | {f1(S[imp].sum(1))} | {f1(S[obs].sum(1))} | {f1(sc)} | {fd(sc - rc)}")
    say("## 2a. by side, later quarter r (sum over early q<r): own = early home margin vs home drives' residual later; opp = early home margin vs minus away drives' residual later; form implied is the same for both sides")
    for j in range(3):
        for lab in ("own", "opp"):
            rc = R["side_imp"][:, j] - R[lab + "_obs"][:, j]
            sc = S["side_imp"][:, j] - S[lab + "_obs"][:, j]
            say(f"  later Q{j + 2} {lab}: real implied {f1(R['side_imp'][:, j])} observed {f1(R[lab + '_obs'][:, j])} counter {f1(rc)} | sim implied {f1(S['side_imp'][:, j])} observed {f1(S[lab + '_obs'][:, j])} counter {f1(sc)} | sim-real counter {fd(sc - rc)}")
    for lab in ("own", "opp"):
        rc = R["side_imp"].sum(1) - R[lab + "_obs"].sum(1)
        sc = S["side_imp"].sum(1) - S[lab + "_obs"].sum(1)
        say(f"  all later {lab}: real counter {f1(rc)} | sim counter {f1(sc)} | sim-real {fd(sc - rc)}")
    say("## 2b. by outcome channel (drive-point components from mod25e_xq.components: count = drives x mean, startfp = start-field-position mean, leadstate = lead-band mean, rest = efficiency given start); sum over early q<r and later r in 2..4; 2x cov of early margin with later signed channel")
    for h, nm in enumerate(COMPS):
        ro, so = R["ch_obs"][:, :, h].sum(1), S["ch_obs"][:, :, h].sum(1)
        ri, si = R["ch_imp"][:, :, h].sum(1), S["ch_imp"][:, :, h].sum(1)
        say(f"  {nm}: real observed {f1(ro)} implied {f1(ri)} counter {f1(ri - ro)} | sim observed {f1(so)} implied {f1(si)} counter {f1(si - so)} | sim-real observed {fd(so - ro)} counter {fd((si - so) - (ri - ro))}")
        for j in range(3):
            say(f"      later Q{j + 2}: real obs {f1(R['ch_obs'][:, j, h])} imp {f1(R['ch_imp'][:, j, h])} | sim obs {f1(S['ch_obs'][:, j, h])} imp {f1(S['ch_imp'][:, j, h])} | obs sim-real {fd(S['ch_obs'][:, j, h] - R['ch_obs'][:, j, h])}")
    say("## 3. matchup adjustment: early residual (drive net points minus quarter x start-lead-band mean, so score state fixed) of side X vs later residual of X (own: form predicts +; defence tightening / regression predicts below) and of the opponent offence (opp: form predicts 0; offence adjustment predicts +). cov and slope per point; form-implied own = k_q k_r V.")
    names = PNAME + ["H1H2"]
    for i, n in enumerate(names):
        row = []
        for lab, key in (("real", R), ("sim", S)):
            sl_o = key["m_own"][:, i] / key["m_var"][:, i]
            sl_p = key["m_opp"][:, i] / key["m_var"][:, i]
            row.append(f"{lab} own slope {f3(sl_o)} opp slope {f3(sl_p)} own-form slope {f3((key['m_own'][:, i] - key['m_imp'][:, i]) / key['m_var'][:, i])}")
        so_, sp_ = S["m_own"][:, i] / S["m_var"][:, i] - R["m_own"][:, i] / R["m_var"][:, i], S["m_opp"][:, i] / S["m_var"][:, i] - R["m_opp"][:, i] / R["m_var"][:, i]
        sf_ = (S["m_own"][:, i] - S["m_imp"][:, i]) / S["m_var"][:, i] - (R["m_own"][:, i] - R["m_imp"][:, i]) / R["m_var"][:, i]
        say(f"  {n}: " + " ; ".join(row) + f" ; sim-real own {fd(so_, 4)} opp {fd(sp_, 4)} own-form {fd(sf_, 4)}")


def main():
    (ART / "counter").mkdir(parents=True, exist_ok=True)
    ns = late.load_xq()
    real, sim = cu.drives_all(ns)
    Dr, Gr = cu.prep(ns, real)
    Ds, Gs = cu.prep(ns, sim)
    say(f"E86 counter-force (scripts/mod25e_counter.py); label {LABEL} seeds {SEEDS}; real games {Gr}, sim games {Gs}; bootstrap {NB}; no sim run")
    gkr = pd.Index(sorted(Dr.gk.unique()))
    gks = pd.Index(sorted(Ds.gk.unique()))
    Tr, Ar = real_epa()
    dr = build(Dr, Gr, gkr, Tr, Ar)
    del Tr, Ar
    Ts, As = sim_epa()
    dsim = build(Ds, Gs, gks, Ts, As)
    del Ts, As
    say(f"side-games with plays: real {dr['nside']}, sim {dsim['nside']}")
    R = run(dr, rng)
    S = run(dsim, rng)
    report(R, S)
    say(f"looks counted (interval prints): {LOOKS[0]}")
    (ART / "counter" / "counter.txt").write_text("\n".join(OUT))


if __name__ == "__main__":
    main()
