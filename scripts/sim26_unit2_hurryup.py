import glob
import json
import math
import os

os.environ["OMP_NUM_THREADS"] = "1"
import numpy as np
import pandas as pd

N_BOOT = 300
HALF_SECS = 1800.0
L0 = math.log1p(3600.0)
TD_PTS = 6.0
MIN_PLAYS = 3
A = "artifacts/sim10/"
rng = np.random.default_rng(2602)
with open("artifacts/sim19/trailing_risk.json") as fh0:
    edges = json.load(fh0)
LE = np.array(edges["deficit_edges"])
TE = np.array(edges["secs_edges"])
T9 = float(TE[0])
NEUTRAL = float(LE[0])

seg = pd.read_parquet(A + "segments.parquet")
c = np.load(A + "csr.npz")
ev_ptr, ev, pl_ptr, pl = c["ev_ptr"], c["ev"], c["pl_ptr"], c["pl"]
n = len(seg)
rid = np.repeat(np.arange(n), np.diff(ev_ptr))
pts = np.bincount(rid, weights=ev[:, 1], minlength=n)
tdr = np.bincount(rid, weights=(ev[:, 1] == TD_PTS).astype(float), minlength=n) > 0
seg["plays"] = np.diff(pl_ptr).astype(float)
seg["pts"] = pts
seg["td"] = tdr.astype(float)
seg["spp"] = np.where(seg.plays > 0, seg.el / seg.plays.clip(lower=1), np.nan)
raw = pd.read_parquet(
    "data/pbp/team_style/raw_pbp_narrow.parquet", columns=["play_type", "no_huddle"]
)
pp = pd.read_parquet(A + "plays.parquet")
align = float((raw.play_type.values[pp.row.values] == pp.play_type.values).mean())
nh = raw.no_huddle.fillna(0).values.astype(float)
prow = pl[:, 0].astype(np.int64)
prid = np.repeat(np.arange(n), np.diff(pl_ptr))
nh_sum = np.bincount(prid, weights=nh[prow], minlength=n)
seg["nhs"] = np.where(seg.plays > 0, nh_sum / seg.plays.clip(lower=1), np.nan)
seg["def"] = -seg.sd
ok = (seg.bad == 0).to_numpy()

base = seg[ok & (seg.plays >= MIN_PLAYS) & seg.spp.notna()]
flag = (base.nhs >= 0.5).to_numpy()
sp = base.spp.to_numpy()
cand = np.unique(np.quantile(sp, np.linspace(0.01, 0.99, 197)))
acc = np.array([((sp <= t) == flag).mean() for t in cand])
CUT = float(cand[acc.argmax()])
res = {
    "row_alignment_playtype_match": align,
    "nh_flag_definition": "drive no_huddle play share >= 0.5",
    "spp_cut_best_accuracy_vs_nh_flag": CUT,
    "spp_cut_accuracy": float(acc.max()),
    "nh_flag_rate_all_drives": float(flag.mean()),
    "td_pts": TD_PTS,
}
seg["hu"] = ((seg.nhs >= 0.5) | (seg.spp <= CUT)).astype(float)
short = seg.plays < MIN_PLAYS
seg.loc[short, "hu"] = (seg.nhs[short] >= 0.5).astype(float)


def dbin(d):
    out = np.full(len(d), -1)
    out[np.abs(d) <= NEUTRAL] = 0
    for i in range(len(LE) - 1):
        out[(d > LE[i]) & (d <= LE[i + 1])] = i + 1
    out[d > LE[-1]] = len(LE)
    return out


seg["db"] = dbin(seg["def"].to_numpy())
CLK = {
    "final9": (0.0, T9),
    "first_half_of_final9": (T9 / 2, T9),
    "second_half_of_final9": (0.0, T9 / 2),
}
DBN = {0: "neutral", 1: "def5-7", 2: "def8-14", 3: "def15+"}
late = seg[ok & (seg.half == 2) & (seg.secs <= T9) & (seg.db >= 0)].copy()
res["segments_late"] = len(late)


def cell_stats(df):
    hu = df[df.hu == 1]
    nm = df[df.hu == 0]
    return {
        "n": len(df),
        "hu_share": float(df.hu.mean()),
        "td": float(df.td.mean()),
        "pts": float(df.pts.mean()),
        "el": float(df.el.mean()),
        "n_hu": len(hu),
        "td_hu": float(hu.td.mean()) if len(hu) else np.nan,
        "pts_hu": float(hu.pts.mean()) if len(hu) else np.nan,
        "el_hu": float(hu.el.mean()) if len(hu) else np.nan,
        "n_nm": len(nm),
        "td_nm": float(nm.td.mean()) if len(nm) else np.nan,
        "pts_nm": float(nm.pts.mean()) if len(nm) else np.nan,
        "el_nm": float(nm.el.mean()) if len(nm) else np.nan,
    }


def clk_sel(df, lo, hi):
    return df[(df.secs > lo) & (df.secs <= hi)]


def ci(arr):
    a = np.asarray(arr)
    a = a[np.isfinite(a)]
    return [float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))]


def gboot(df, fn):
    g = pd.factorize(df.game_id)[0]
    ng = g.max() + 1
    order = np.argsort(g, kind="stable")
    gs = g[order]
    st = np.searchsorted(gs, np.arange(ng + 1))
    arrs = {k: df[k].to_numpy()[order] for k in ("hu", "td", "pts", "el")}
    out = []
    for _ in range(N_BOOT):
        pick = rng.integers(0, ng, ng)
        ix = np.concatenate([np.arange(st[p], st[p + 1]) for p in pick])
        out.append(fn({k: v[ix] for k, v in arrs.items()}))
    return np.array(out)


def fn_cell(a):
    m = a["hu"] == 1
    return [
        a["hu"].mean(),
        a["td"][m].mean() if m.any() else np.nan,
        a["td"][~m].mean() if (~m).any() else np.nan,
        a["td"].mean(),
        a["pts"][m].mean() if m.any() else np.nan,
        a["pts"][~m].mean() if (~m).any() else np.nan,
        a["el"][m].mean() if m.any() else np.nan,
        a["el"][~m].mean() if (~m).any() else np.nan,
    ]


NMS = ["hu_share", "td_hu", "td_nm", "td", "pts_hu", "pts_nm", "el_hu", "el_nm"]
res["real_cells"] = {}
for cn, (lo, hi) in CLK.items():
    sub = clk_sel(late, lo, hi)
    for b in (0, 1, 2, 3):
        d = sub[sub.db == b]
        st = cell_stats(d)
        bb = gboot(d, fn_cell)
        for i, nm in enumerate(NMS):
            st[nm + "_ci95"] = ci(bb[:, i])
        res["real_cells"][f"{cn}|{DBN[b]}"] = st
print("real cells done", flush=True)

kq = late[late.db >= 0]
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
hv = seg.half.to_numpy()
hm_ = seg.home.to_numpy()
he_ = seg.he.to_numpy() == 1
gid_ = seg.game_id.to_numpy()
el_ = seg.el.to_numpy()
hu_ = seg.hu.to_numpy()
td_ = seg.td.to_numpy()
pts_ = seg.pts.to_numpy()
fin = np.isfinite(h0)
nd = X.shape[1]
f = fin[:nd]
strata = {(hf, hm): np.flatnonzero(ok & (hv == hf) & (hm_ == hm)) for hf in (1, 2) for hm in (0, 1)}
off_, po_ = ev[:, 0], ev[:, 1]
qrows = kq.index.to_numpy()
KN = ("hu", "td_raw", "td_trunc", "hu_td_raw", "hu_td_trunc", "pts_raw", "el", "ess")
K = {k: np.zeros(len(qrows)) for k in KN}
for n_, j in enumerate(qrows):
    hf, hm = int(hv[j]), int(hm_[j])
    idx = strata[(hf, hm)]
    gsq = math.expm1(X[j, 2]) + (HALF_SECS if hf == 1 else 0.0)
    h = h0.copy()
    h[1] = h[1] * math.exp(gsd * (math.log1p(gsq) - L0))
    Xi = X[idx]
    z = (((Xi[:, f] - X[j, f]) / h[:nd][f]) ** 2).sum(axis=1)
    z[he_[idx] & (Xi[:, 2] < X[j, 2])] = np.inf
    z[gid_[idx] == gid_[j]] = np.inf
    w = np.exp(-0.5 * (z - z.min()))
    nzi = np.flatnonzero(w > 1e-12)
    ii, ww = idx[nzi], w[nzi]
    ww = ww / ww.sum()
    secs_q = math.expm1(X[j, 2])
    cut = he_[ii] | (el_[ii] >= secs_q)
    tdt = td_[ii].copy()
    for r in np.flatnonzero(cut):
        a, b = ev_ptr[ii[r]], ev_ptr[ii[r] + 1]
        tdt[r] = float(((po_[a:b] == TD_PTS) & (off_[a:b] <= secs_q)).any())
    hh = hu_[ii]
    sh = max(ww @ hh, 1e-12)
    K["hu"][n_] = ww @ hh
    K["td_raw"][n_] = ww @ td_[ii]
    K["td_trunc"][n_] = ww @ tdt
    K["hu_td_raw"][n_] = (ww * hh) @ td_[ii] / sh
    K["hu_td_trunc"][n_] = (ww * hh) @ tdt / sh
    K["pts_raw"][n_] = ww @ pts_[ii]
    K["el"][n_] = ww @ el_[ii]
    K["ess"][n_] = 1.0 / (ww @ ww)
    if n_ % 2000 == 0:
        print("kernel", n_, len(qrows), flush=True)
kq = kq.assign(**{f"k_{k}": v for k, v in K.items()})

rows = []
for p in sorted(glob.glob(A + "instrument/instr_s11/*.jsonl")):
    with open(p) as fh:
        for ln in fh:
            r = json.loads(ln)
            s2 = [0.0, 0.0]
            for hf, pos, _yl, secs, po, pd_, el, _tr in r["dr"]:
                if hf < 3:
                    rows.append((r["id"], s2[1 - int(pos)] - s2[int(pos)], hf, secs, po, el))
                s2[int(pos)] += po
                s2[1 - int(pos)] += pd_
sim = pd.DataFrame(rows, columns=["game_id", "def", "hf", "secs", "po", "el"])
sim = sim[(sim.hf == 2) & (sim.secs <= T9)].copy()
sim["db"] = dbin(sim["def"].to_numpy())
sim = sim[sim.db >= 0]
sim["td"] = (sim.po >= TD_PTS).astype(float)
sim["pts"] = sim.po
res["sim_po_values"] = sorted(set(np.round(sim.po.unique(), 1).tolist()))[:12]


def gm(v, g, pick):
    ng = g.max() + 1
    cnt = np.bincount(g, minlength=ng)
    sm = np.bincount(g, weights=v, minlength=ng)
    return sm[pick].sum() / max(cnt[pick].sum(), 1)


def dg(g):
    ng = g.max() + 1
    return rng.integers(0, ng, ng)


res["decomposition"] = {}
for cn, (lo, hi) in CLK.items():
    rk = clk_sel(kq, lo, hi)
    rs = sim[(sim.secs > lo) & (sim.secs <= hi)]
    rr = clk_sel(late, lo, hi)
    nk, nr, ns_ = rk[rk.db == 0], rr[rr.db == 0], rs[rs.db == 0]
    for b in (1, 2, 3):
        gk, gr, gs_ = rk[rk.db == b], rr[rr.db == b], rs[rs.db == b]
        comp = {
            "real_td": float(gr.td.mean()),
            "kernel_td_raw": float(gk.k_td_raw.mean()),
            "kernel_td_trunc": float(gk.k_td_trunc.mean()),
            "sim_td": float(gs_.td.mean()),
            "real_pts": float(gr.pts.mean()),
            "kernel_pts_raw": float(gk.k_pts_raw.mean()),
            "sim_pts": float(gs_.pts.mean()),
            "real_el": float(gr.el.mean()),
            "kernel_el": float(gk.k_el.mean()),
            "sim_el": float(gs_.el.mean()),
            "real_hu_share": float(gr.hu.mean()),
            "kernel_hu_share": float(gk.k_hu.mean()),
            "neutral_real_hu_share": float(nr.hu.mean()),
            "neutral_kernel_hu_share": float(nk.k_hu.mean()),
            "real_td_hu": float(gr[gr.hu == 1].td.mean()),
            "kernel_td_hu_raw": float(gk.k_hu_td_raw.mean()),
            "kernel_td_hu_trunc": float(gk.k_hu_td_trunc.mean()),
            "n_real": len(gr),
            "n_kernel_q": len(gk),
            "n_sim": len(gs_),
            "n_real_neutral": len(nr),
            "n_sim_neutral": len(ns_),
            "ess_median": float(gk.k_ess.median()),
        }
        R = (gr.td.to_numpy(), pd.factorize(gr.game_id)[0], gr.hu.to_numpy())
        NR = (nr.td.to_numpy(), pd.factorize(nr.game_id)[0], nr.hu.to_numpy())
        NRx = nr.td.to_numpy() * nr.hu.to_numpy()
        Rx = gr.td.to_numpy() * gr.hu.to_numpy()
        KA = gk.k_hu.to_numpy() * gk.k_hu_td_raw.to_numpy()
        KC = gk.k_td_raw.to_numpy() - KA
        KD = 1.0 - gk.k_hu.to_numpy()
        G = (
            gk.k_td_raw.to_numpy(),
            gk.k_td_trunc.to_numpy(),
            gk.k_hu.to_numpy(),
            pd.factorize(gk.game_id)[0],
        )
        NG = (nk.k_td_raw.to_numpy(), nk.k_td_trunc.to_numpy(), nk.k_hu.to_numpy())
        NGg = pd.factorize(nk.game_id)[0]
        S = (gs_.td.to_numpy(), pd.factorize(gs_.game_id)[0])
        NS = (ns_.td.to_numpy(), pd.factorize(ns_.game_id)[0])
        bs = []
        for _ in range(N_BOOT):
            pr, pnr, pk, pnk, ps, pns = dg(R[1]), dg(NR[1]), dg(G[3]), dg(NGg), dg(S[1]), dg(NS[1])
            rsg = gm(R[0], R[1], pr) - gm(NR[0], NR[1], pnr)
            ssg = gm(S[0], S[1], ps) - gm(NS[0], NS[1], pns)
            kr = gm(G[0], G[3], pk) - gm(NG[0], NGg, pnk)
            kt = gm(G[1], G[3], pk) - gm(NG[1], NGg, pnk)
            hr_, hk_ = gm(R[2], R[1], pr), gm(G[2], G[3], pk)
            nhs_ = gm(NR[2], NR[1], pnr)
            t_hu = gm(NRx, NR[1], pnr) / max(nhs_, 1e-12)
            t_nm = (gm(NR[0], NR[1], pnr) - gm(NRx, NR[1], pnr)) / max(1.0 - nhs_, 1e-12)
            ss_real = (hr_ - nhs_) * (t_hu - t_nm)
            kcon = gm(KA, G[3], pk) / max(hk_, 1e-12) - gm(KC, G[3], pk) / max(
                gm(KD, G[3], pk), 1e-12
            )
            bs.append([rsg, ssg, kr, kt, hr_, hk_, ss_real, (hr_ - hk_) * kcon])
        bs = np.array(bs)
        pt = [
            comp["real_td"] - float(nr.td.mean()),
            comp["sim_td"] - float(ns_.td.mean()),
            comp["kernel_td_raw"] - float(nk.k_td_raw.mean()),
            comp["kernel_td_trunc"] - float(nk.k_td_trunc.mean()),
        ]
        comp["surge_real"] = [pt[0], ci(bs[:, 0])]
        comp["surge_sim"] = [pt[1], ci(bs[:, 1])]
        comp["surge_kernel_raw"] = [pt[2], ci(bs[:, 2])]
        comp["surge_kernel_trunc"] = [pt[3], ci(bs[:, 3])]
        comp["gap_total_real_minus_sim"] = [pt[0] - pt[1], ci(bs[:, 0] - bs[:, 1])]
        comp["gap_kernel_stage_real_minus_kernel_raw"] = [pt[0] - pt[2], ci(bs[:, 0] - bs[:, 2])]
        comp["gap_truncation_kernel_raw_minus_trunc"] = [pt[2] - pt[3], ci(bs[:, 2] - bs[:, 3])]
        comp["gap_downstream_kernel_trunc_minus_sim"] = [pt[3] - pt[1], ci(bs[:, 3] - bs[:, 1])]
        comp["real_surge_from_hurryup_share_shift"] = [
            float(np.mean(bs[:, 6])),
            ci(bs[:, 6]),
        ]
        comp["kernel_gap_from_hurryup_undersampling"] = [
            float(np.mean(bs[:, 7])),
            ci(bs[:, 7]),
        ]
        comp["hu_share_real_minus_kernel"] = [
            comp["real_hu_share"] - comp["kernel_hu_share"],
            ci(bs[:, 4] - bs[:, 5]),
        ]
        res["decomposition"][f"{cn}|{DBN[b]}"] = comp
res["looks"] = {
    "real_cells": len(CLK) * 4,
    "decomposition_cells": len(CLK) * 3,
    "family": "hurry-up split and TD-surge attribution, 3 deficit bins x 3 clock windows",
}
os.makedirs("artifacts/sim26", exist_ok=True)
with open("artifacts/sim26/unit2_hurryup.json", "w") as fo:
    json.dump(res, fo, indent=1, default=float)
print(json.dumps(res["looks"]))
print("done")
