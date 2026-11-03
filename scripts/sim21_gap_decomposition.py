import glob
import json
import math
import os

os.environ["OMP_NUM_THREADS"] = "1"
import numpy as np
import pandas as pd

N_BOOT = 200
N_QUERY = 12000
HALF_SECS = 1800.0
L0 = math.log1p(3600.0)
DE = np.array([4.0, 7.0, 14.0])
TE = np.array([546.0, 1432.0, 2212.0])
NT = len(TE) + 1
ND = len(DE) + 1
rng = np.random.default_rng(2101)
A = "artifacts/sim10/"

seg = pd.read_parquet(A + "segments.parquet")
c = np.load(A + "csr.npz")
ev_ptr, ev, pl_ptr = c["ev_ptr"], c["ev"], c["pl_ptr"]
n = len(seg)
nev = np.diff(ev_ptr)
rid = np.repeat(np.arange(n), nev)
po = np.bincount(rid, weights=ev[:, 1], minlength=n)
seg["plays"] = np.diff(pl_ptr).astype(float)
seg["score"] = (po > 0).astype(float)
seg["gs"] = seg["secs"] + np.where(seg["half"] == 1, HALF_SECS, 0.0)
seg["deficit"] = -seg["sd"]
oo = pd.read_parquet(A + "oof_targets.parquet")
seg["oof_el"] = np.nan
seg["oof_score"] = np.nan
seg.loc[oo.index, "oof_el"] = oo["el"].to_numpy()
seg.loc[oo.index, "oof_score"] = (oo["td"] + oo["fg"]).to_numpy()
seg["spp"] = np.where(seg["plays"] > 0, seg["el"] / seg["plays"].clip(lower=1), np.nan)
seg["gid"] = seg["game_id"]
real = seg[(seg["bad"] == 0) & (seg["half"] < 3)].copy()

nxt = seg.groupby("game_id")["sd"].shift(-1)
chk = float(nxt[(po > 0) & (seg["half"] < 3)].mean())

rows = []
for p in sorted(glob.glob(A + "instrument/instr_s11/*.jsonl")):
    with open(p) as fh:
        for ln in fh:
            r = json.loads(ln)
            sc = [0.0, 0.0]
            for hf, pos, yl, secs, po_, pd_, el, _tr in r["dr"]:
                if hf < 3:
                    rows.append(
                        (
                            r["id"],
                            sc[1 - int(pos)] - sc[int(pos)],
                            secs + (HALF_SECS if hf == 1 else 0.0),
                            yl,
                            float(po_ > 0),
                            el,
                        )
                    )
                sc[int(pos)] += po_
                sc[1 - int(pos)] += pd_
sim = pd.DataFrame(rows, columns=["gid", "deficit", "gs", "yl", "score", "el"])


def cell_of(df):
    d = df["deficit"].to_numpy()
    grp = np.where(d > 0, 0, np.where(d < 0, 1, 2))
    mag = np.digitize(np.abs(d), DE)
    mag = np.where(grp == 2, 0, mag)
    tb = np.digitize(df["gs"].to_numpy(), TE)
    return grp * (ND * NT) + mag * NT + tb


NC = 3 * ND * NT
GN = ["trailing", "leading", "tied"]


def group_of(i):
    return np.asarray(i) // (ND * NT)


def boot(df, cols):
    g, _ = pd.factorize(df["gid"])
    G = g.max() + 1
    cell = cell_of(df)
    sx = np.zeros((G, NC))
    np.add.at(sx, (g, cell), 1.0)
    cnt = rng.multinomial(G, np.full(G, 1.0 / G), size=N_BOOT).astype(float)
    res = {}
    for k in cols:
        v = df[k].to_numpy(float)
        okv = np.isfinite(v)
        s = np.zeros((G, NC))
        cn = np.zeros((G, NC))
        np.add.at(s, (g[okv], cell[okv]), v[okv])
        np.add.at(cn, (g[okv], cell[okv]), 1.0)
        est = s.sum(0) / np.maximum(cn.sum(0), 1)
        bs = (cnt @ s) / np.maximum(cnt @ cn, 1)
        res[k] = (est, bs)
    return sx.sum(0), res


def pooled(a, b, wts):
    est_a, bs_a = a
    est_b, bs_b = b
    out = {}
    for gi, nm in enumerate(GN):
        m = (group_of(np.arange(NC)) == gi) & (wts > 0)
        w = wts[m] / wts[m].sum()
        pt = float((w * (est_a[m] - est_b[m])).sum())
        bb = (bs_a[:, m] - bs_b[:, m]) @ w
        lo, hi = np.percentile(bb, [2.5, 97.5])
        out[nm] = {
            "gap": pt,
            "ci95": [float(lo), float(hi)],
            "p_positive": float((bb > 0).mean()),
            "real_mean": float((w * est_b[m]).sum()),
            "sim_mean": float((w * est_a[m]).sum()),
        }
    return out


cols_s = ["score", "el", "yl"]
ns, sm = boot(sim, cols_s)
nr, rl = boot(real, [*cols_s, "plays", "spp"])
res = {"check_sd_after_opponent_score": chk, "n_sim_drives": len(sim), "n_real_drives": len(real)}
wr = np.where(ns > 0, nr, 0.0)
res["A_sim_minus_real"] = {k: pooled(sm[k], rl[k], wr) for k in cols_s}

cells = {}
for i in range(NC):
    if nr[i] > 0 and ns[i] > 0:
        r_ = i % (ND * NT)
        nm = f"{GN[int(group_of(i))]} mag{r_ // NT} t{r_ % NT}"
        cells[nm] = {
            "n_real": int(nr[i]),
            "n_sim": int(ns[i]),
            **{f"d_{k}": float(sm[k][0][i] - rl[k][0][i]) for k in cols_s},
            "real_plays": float(rl["plays"][0][i]),
            "real_spp": float(rl["spp"][0][i]),
        }
res["A_cells"] = cells

cell_r = cell_of(real)
beta = np.full(NC, np.nan)
for i in range(NC):
    m = (cell_r == i) & (real["plays"].to_numpy() > 0) & (real["el"].to_numpy() > 0)
    if m.sum() > 30:
        lp = np.log(real["plays"].to_numpy()[m])
        le = np.log(real["el"].to_numpy()[m])
        beta[i] = np.cov(lp, le)[0, 1] / le.var(ddof=1)
split = {}
for gi, nm in enumerate(GN):
    idx = [
        i
        for i in range(NC)
        if group_of(i) == gi and nr[i] > 0 and ns[i] > 0 and np.isfinite(beta[i])
    ]
    w = np.array([nr[i] for i in idx])
    w = w / w.sum()
    dl = np.array([math.log(sm["el"][0][i] / rl["el"][0][i]) for i in idx])
    b = beta[idx]
    split[nm] = {
        "dlog_el": float((w * dl).sum()),
        "plays_share_inferred": float((w * b * dl).sum() / (w * dl).sum()),
        "beta_mean": float((w * b).sum()),
    }
res["A_el_split_plays_vs_secs_per_play_inferred"] = split

yl_cf = {}
gr = group_of(cell_r)
gs_ = group_of(cell_of(sim))
for gi, nm in enumerate(GN):
    rg = real[gr == gi]
    sg = sim[gs_ == gi]
    q = np.quantile(rg["yl"], np.linspace(0, 1, 11)[1:-1])
    br = np.digitize(rg["yl"], q)
    bsm = np.digitize(sg["yl"], q)
    rate = np.array([rg["score"].to_numpy()[br == b].mean() for b in range(10)])
    pr = np.bincount(br, minlength=10) / len(br)
    ps = np.bincount(bsm, minlength=10) / len(bsm)
    yl_cf[nm] = {
        "score_gap_from_start_field": float(rate @ (ps - pr)),
        "score_gap_total": res["A_sim_minus_real"]["score"][nm]["gap"],
    }
res["A_start_field_counterfactual"] = yl_cf

X = np.c_[
    seg["yl"],
    seg["sd"],
    np.log1p(seg["secs"].clip(lower=0.0)),
    seg["too"],
    seg["tod"],
    seg["x"],
    seg["season"],
].astype(float)
with open(A + "bandwidths.json") as fb:
    bw = json.load(fb)
h0 = np.array(bw["h"], dtype=float)
gsd = float(bw["g_sd"])
ok = (seg["bad"] == 0).to_numpy()
hv = seg["half"].to_numpy()
hm_ = seg["home"].to_numpy()
he_ = seg["he"].to_numpy() == 1
gid_ = seg["game_id"].to_numpy()
att = {
    "el": seg["el"].to_numpy(float),
    "plays": seg["plays"].to_numpy(float),
    "score": seg["score"].to_numpy(float),
    "yl": seg["yl"].to_numpy(float),
    "sd": seg["sd"].to_numpy(float),
    "lt": X[:, 2],
}
spp_all = seg["spp"].to_numpy(float)
qi = rng.choice(np.flatnonzero(ok & (hv < 3)), size=N_QUERY, replace=False)
strata = {}
for hf in (1, 2):
    for hm in (0, 1):
        strata[(hf, hm)] = np.flatnonzero(ok & (hv == hf) & (hm_ == hm))
K = {k: np.zeros(len(qi)) for k in [*att, "spp", "ess"]}
fin = np.isfinite(h0)
nd = X.shape[1]
for n_, j in enumerate(qi):
    hf, hm = int(hv[j]), int(hm_[j])
    idx = strata[(hf, hm)]
    gsq = math.expm1(X[j, 2]) + (HALF_SECS if hf == 1 else 0.0)
    h = h0.copy()
    h[1] = h[1] * math.exp(gsd * (math.log1p(gsq) - L0))
    f = fin[:nd]
    Xi = X[idx]
    z = (((Xi[:, f] - X[j, f]) / h[:nd][f]) ** 2).sum(axis=1)
    z[he_[idx] & (Xi[:, 2] < X[j, 2])] = np.inf
    z[gid_[idx] == gid_[j]] = np.inf
    w = np.exp(-0.5 * (z - z.min()))
    w /= w.sum()
    for k in att:
        K[k][n_] = w @ att[k][idx]
    s_ = spp_all[idx]
    mk = np.isfinite(s_)
    K["spp"][n_] = (w[mk] * s_[mk]).sum() / w[mk].sum()
    K["ess"][n_] = 1.0 / (w @ w)
kq = seg.iloc[qi].copy()
kq["deficit"] = -kq["sd"]
kcell = cell_of(kq)
kres = {}
for gi, nm in enumerate(GN):
    m = group_of(kcell) == gi
    kres[nm] = {
        "n": int(m.sum()),
        **{
            f"kernel_minus_real_{k}": float(np.nanmean(K[k][m]) - np.nanmean(kq[k].to_numpy()[m]))
            for k in ("el", "plays", "score", "yl", "spp")
        },
        "real_el": float(kq["el"][m].mean()),
        "real_plays": float(kq["plays"][m].mean()),
        "real_spp": float(np.nanmean(kq["spp"].to_numpy()[m])),
        "ess_median": float(np.median(K["ess"][m])),
    }
res["B_kernel_stage_only"] = kres
qsd = seg["sd"].to_numpy()[qi]
res["B_kernel_state_response"] = {
    "slope_candidate_sd_on_query_sd": float(np.polyfit(qsd, K["sd"], 1)[0]),
    "slope_candidate_logsecs_on_query_logsecs": float(np.polyfit(X[qi, 2], K["lt"], 1)[0]),
    "sd_bandwidth_at_mid_game": float(h0[1] * math.exp(gsd * (math.log1p(1800.0) - L0))),
    "query_sd_sd": float(qsd.std()),
}

oofres = {}
rr = real[np.isfinite(real["oof_el"])]
oc = group_of(cell_of(rr))
for gi, nm in enumerate(GN):
    m = oc == gi
    oofres[nm] = {
        "n": int(m.sum()),
        "oof_minus_real_el": float((rr["oof_el"][m] - rr["el"][m]).mean()),
        "oof_minus_real_score": float((rr["oof_score"][m] - rr["score"][m]).mean()),
    }
res["C_outcome_model_oof_bias"] = oofres
res["looks"] = {
    "pooled_group_metrics": 9,
    "cell_tables": len(cells) * 3,
    "kernel_group_metrics": 15,
    "oof_group_metrics": 6,
    "start_field_cf": 3,
    "el_split": 3,
}
with open("artifacts/sim21/gap_decomposition.json", "w") as fh:
    json.dump(res, fh, indent=1)
print(json.dumps({k: v for k, v in res.items() if k != "A_cells"}, indent=1))
