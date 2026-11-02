import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
import sim10_drive as S

OUT = REPO / "artifacts" / "sim11"
OUTCOMES = ["el", "td", "fg", "ds"]
LAB = OUTCOMES + ["score_idx"]
NB = 2000
SEED = 20261102
GROUPS = {"non_eoh": [0, 1, 3, 4], "all": [0, 1, 2, 3, 4, 5]}
Q_EDGES = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 1.0]
KNN = 400


def real_frame():
    seg = S.with_mu(pd.read_parquet(S.OUT / "segments.parquet"))
    seg = S.add_game_feats(seg)
    po, pd_ = S.segment_points(seg)
    ok = (seg["bad"].to_numpy() == 0) & (seg["half"].to_numpy() < 3)
    ph = np.array(
        [
            S.post_bin(h, s)
            for h, s in zip(seg["half"].to_numpy(), seg["secs"].to_numpy(), strict=True)
        ]
    )
    df = pd.DataFrame(
        {
            "mu": seg["mu"].to_numpy(float),
            "season": seg["season"].to_numpy(),
            "ph": ph,
            "el": seg["el"].to_numpy(float),
            "td": (po >= 6).astype(float),
            "fg": (po == 3).astype(float),
            "ds": (pd_ > 0).astype(float),
        }
    )
    return df[ok].reset_index(drop=True), seg


def targets_frame(seg, b):
    d = pd.read_parquet(OUT / "draws_lab21.parquet")
    d = d[(d["ph"] < 6) & d["t_td"].notna()].reset_index(drop=True)
    tab, _ = S.posterior_table(seg, b)
    cx = np.array([tab[int(k)][0] for k in d["ph"]])
    cs = np.array([tab[int(k)][1] for k in d["ph"]])
    mh = np.array([0.5 * (tab[int(k)][2] + tab[int(k)][3]) for k in d["ph"]])
    d["mt"] = cx * d["tgt"].to_numpy() + cs * (d["sd"].to_numpy() - mh)
    return d


NPH = 6


def cell_sums(cell, w, y, nbin, tmp):
    n = np.bincount(cell, weights=w, minlength=NPH * nbin).reshape(NPH, nbin)
    out = np.empty((NPH, nbin, y.shape[1]))
    for k in range(y.shape[1]):
        np.multiply(w, y[:, k], out=tmp)
        out[:, :, k] = np.bincount(cell, weights=tmp, minlength=NPH * nbin).reshape(NPH, nbin)
    return n, out


def cell_contrast(n, sm, bins, phases):
    tot = 0.0
    res = np.zeros(sm.shape[2])
    for p in phases:
        na, nm = n[p].sum(), n[p, bins].sum()
        if na <= 0 or nm <= 0:
            continue
        res += nm * (sm[p, bins].sum(axis=0) / nm - sm[p].sum(axis=0) / na)
        tot += nm
    return res / tot if tot > 0 else np.full(sm.shape[2], np.nan)


def knn_contrast(order, w, y, ph, n, sm, phases, buf):
    np.take(w, order, out=buf, mode="clip")
    np.cumsum(buf, out=buf)
    kk = int(np.searchsorted(buf, KNN)) + 1
    sel = order[:kk]
    ws, ys, ps = w[sel], y[sel], ph[sel]
    tot = 0.0
    res = np.zeros(y.shape[1])
    for p in phases:
        m = ps == p
        nm = ws[m].sum()
        na = n[p].sum()
        if nm <= 0 or na <= 0:
            continue
        res += nm * ((ws[m][:, None] * ys[m]).sum(axis=0) / nm - sm[p].sum(axis=0) / na)
        tot += nm
    return res / tot if tot > 0 else np.full(y.shape[1], np.nan)


def ext(v):
    v = np.asarray(v, dtype=float)
    if v.ndim == 1:
        return np.r_[v, 7.0 * (v[1] - v[3]) + 3.0 * v[2]]
    return np.c_[v, 7.0 * (v[:, 1] - v[:, 3]) + 3.0 * v[:, 2]]


def main():
    rng = np.random.default_rng(SEED)
    b = json.loads((S.OUT / "build_meta.json").read_text())["eiv_b"]
    R, seg = real_frame()
    T = targets_frame(seg, b)
    yr = R[OUTCOMES].to_numpy(float)
    yt = T[["t_el", "t_td", "t_fg", "t_ds"]].to_numpy(float)
    ya = T[["a_el", "a_td", "a_fg", "a_ds"]].to_numpy(float)
    mu, phr, sr = R["mu"].to_numpy(), R["ph"].to_numpy(), R["season"].to_numpy()
    tg_s, mt, pht, gt = (
        T["tgt"].to_numpy(),
        T["mt"].to_numpy(),
        T["ph"].to_numpy(),
        T["game"].to_numpy(),
    )
    seasons = np.unique(sr)
    sidx = np.searchsorted(seasons, sr)
    games, gidx = np.unique(gt, return_inverse=True)
    nbin = len(Q_EDGES) - 1
    qe = np.quantile(mu, Q_EDGES)
    bin_r = np.clip(np.searchsorted(qe, mu, side="right") - 1, 0, nbin - 1)
    te = np.quantile(tg_s, Q_EDGES)
    bin_t = np.clip(np.searchsorted(te, tg_s, side="right") - 1, 0, nbin - 1)
    top_t = tg_s >= np.quantile(tg_s, 0.9)
    mt_top = float(mt[top_t].mean())
    mu_pct = float((mu < mt_top).mean())
    rows, summ = [], {}
    cell_r = (phr * nbin + bin_r).astype(np.int64)
    cell_t = (pht * nbin + bin_t).astype(np.int64)
    order_r = np.argsort(np.abs(mu - mt_top), kind="stable")
    tb = np.arange(nbin - 3, nbin)
    tmp_r, tmp_t = np.empty(len(R)), np.empty(len(T))
    kbuf = np.empty(len(R))
    wr, wt = np.empty(len(R)), np.empty(len(T))
    ones_r, ones_t = np.ones(len(R)), np.ones(len(T))
    draw_s = rng.integers(0, len(seasons), (NB, len(seasons)))
    draw_g = rng.integers(0, len(games), (NB, len(games)))

    def stats(wr_, wt_, phases):
        nr, sr_ = cell_sums(cell_r, wr_, yr, nbin, tmp_r)
        nt, st_ = cell_sums(cell_t, wt_, yt, nbin, tmp_t)
        na_, sa_ = cell_sums(cell_t, wt_, ya, nbin, tmp_t)
        curve = np.array([cell_contrast(nr, sr_, [k], phases) for k in range(nbin)])
        tcurve = np.array([cell_contrast(nt, st_, [k], phases) for k in range(nbin)])
        acurve = np.array([cell_contrast(na_, sa_, [k], phases) for k in range(nbin)])
        ct = cell_contrast(nt, st_, tb, phases)
        ca = cell_contrast(na_, sa_, tb, phases)
        kn = knn_contrast(order_r, wr_, yr, phr, nr, sr_, phases, kbuf)
        rtop = cell_contrast(nr, sr_, tb, phases)
        return curve, tcurve, acurve, ct, ca, kn, rtop

    for gname, phases in GROUPS.items():
        base = stats(ones_r, ones_t, phases)
        reps = []
        for i in range(NB):
            cs_ = np.bincount(draw_s[i], minlength=len(seasons)).astype(float)
            cg = np.bincount(draw_g[i], minlength=len(games)).astype(float)
            np.take(cs_, sidx, out=wr, mode="clip")
            np.take(cg, gidx, out=wt, mode="clip")
            reps.append(stats(wr, wt, phases))
        rc = np.array([ext(r[0]) for r in reps])
        lo, hi = np.nanpercentile(rc, [5, 95], axis=0)
        for k in range(nbin):
            m_ = bin_r == k
            for j, nm in enumerate(LAB):
                rows.append(
                    {
                        "group": gname,
                        "pct_lo": Q_EDGES[k],
                        "pct_hi": Q_EDGES[k + 1],
                        "mu_mean": float(mu[m_].mean()),
                        "n_real": int(m_.sum()),
                        "outcome": nm,
                        "real_contrast": float(ext(base[0])[k][j]),
                        "real_lo90": float(lo[k][j]),
                        "real_hi90": float(hi[k][j]),
                    }
                )
        d_top = np.array([ext(r[3]) - ext(r[6]) for r in reps])
        d_knn = np.array([ext(r[3]) - ext(r[5]) for r in reps])
        d_ach = np.array([ext(r[4]) - ext(r[6]) for r in reps])
        e_top, e_knn, e_ach = (
            ext(base[3]) - ext(base[6]),
            ext(base[3]) - ext(base[5]),
            ext(base[4]) - ext(base[6]),
        )

        def pk(est, arr, j):
            return {
                "est": float(est[j]),
                "lo90": float(np.nanpercentile(arr[:, j], 5)),
                "hi90": float(np.nanpercentile(arr[:, j], 95)),
                "probability_positive": float(np.nanmean(arr[:, j] > 0)),
            }

        gap = {}
        for j, nm in enumerate(LAB):
            gap[nm] = {
                "tg_top_decile_contrast": float(ext(base[3])[j]),
                "achieved_top_decile_contrast": float(ext(base[4])[j]),
                "real_top_decile_by_mu_contrast": float(ext(base[6])[j]),
                "real_knn_at_mt_contrast": float(ext(base[5])[j]),
                "tg_minus_real_top_decile": pk(e_top, d_top, j),
                "tg_minus_real_knn_at_mt": pk(e_knn, d_knn, j),
                "achieved_minus_real_top_decile": pk(e_ach, d_ach, j),
            }
        slopes = {}
        xs = np.array([mu[bin_r == k].mean() for k in range(nbin)])
        ws = np.array([(bin_r == k).sum() for k in range(nbin)], float)
        tx = np.array([tg_s[bin_t == k].mean() for k in range(nbin)])
        for j, nm in enumerate(LAB):
            slopes[nm] = {
                "real_slope_on_mu": float(np.polyfit(xs, ext(base[0])[:, j], 1, w=np.sqrt(ws))[0]),
                "tg_slope_on_true_strength": float(np.polyfit(tx, ext(base[1])[:, j], 1)[0]),
                "achieved_slope_on_true_strength": float(np.polyfit(tx, ext(base[2])[:, j], 1)[0]),
            }
        summ[gname] = {"gap": gap, "slopes": slopes}
    pd.DataFrame(rows).to_csv(OUT / "frontier_curve.csv", index=False)
    meta = {
        "mu_quantiles": {f"{q:.2f}": float(v) for q, v in zip(Q_EDGES, qe, strict=True)},
        "real_mu_sd": float(mu.std()),
        "true_strength_sd": float(tg_s.std()),
        "mt_top_decile_mean": mt_top,
        "mt_top_decile_real_mu_percentile": mu_pct,
        "real_top_decile_mu_mean": float(mu[mu >= np.quantile(mu, 0.9)].mean()),
        "true_strength_top_decile_mean": float(tg_s[top_t].mean()),
        "n_real": len(R),
        "n_draws": len(T),
        "bootstrap_reps": NB,
        "interval": "90 percent percentile; season-blocked real drives, game-blocked draws",
        "knn": KNN,
        "summary": summ,
    }
    (OUT / "frontier.json").write_text(json.dumps(meta, indent=1))
    g = []
    for gn, s in summ.items():
        for o, d in s["gap"].items():
            r = {"group": gn, "outcome": o}
            for k, v in d.items():
                if isinstance(v, dict):
                    r.update({f"{k}_{kk}": vv for kk, vv in v.items()})
                else:
                    r[k] = v
            g.append(r)
    pd.DataFrame(g).to_csv(OUT / "frontier_gap.csv", index=False)
    print(json.dumps({k: v for k, v in meta.items() if k != "summary"}, indent=1))


if __name__ == "__main__":
    main()
