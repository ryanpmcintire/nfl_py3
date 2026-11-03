import argparse
import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
os.environ.setdefault("SIM10_SEASONS", "2009-2017")
os.environ.setdefault("SIM10_OUT", str(REPO / "artifacts" / "sim10"))
sys.path.insert(0, str(REPO / "scripts"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import sim10_drive as s10  # noqa: E402
from scipy.spatial import cKDTree  # noqa: E402

BINS = {
    "yl": [0, 25, 50, 75, 100],
    "sd": [-100, -14, -7, -1, 0, 7, 14, 100],
    "secs": [-1, 120, 300, 600, 1200, 1800],
    "half": [0, 1, 2],
}
FEAT = ["yl", "sd", "log_secs", "too", "tod", "week", "half", "home"]


def real_reference():
    seg = pd.read_parquet(s10.OUT / "segments.parquet")
    fit = (seg["bad"].to_numpy() == 0) & (seg["half"].to_numpy() < 3)
    cls, k = s10.head_codes(seg, fit)
    po, _ = s10.segment_points(seg)
    ok = fit & (cls >= 0)
    mpo = np.bincount(cls[ok], weights=po[ok], minlength=k) / np.maximum(
        np.bincount(cls[ok], minlength=k), 1
    )
    real = seg[fit].reset_index(drop=True)
    return real, s10.head_features(real), po[fit], mpo, k


def load_log(path):
    parts = sorted(Path(path).glob("*.npz"))
    if not parts:
        raise SystemExit(f"no npz parts in {path}")
    z = [np.load(p) for p in parts]
    return {key: np.concatenate([a[key] for a in z]) for key in z[0].files}


def bin_table(name, edges, qv, rv, d_q, e_q, hd_q, obs_r, share_w):
    rows = []
    qb = np.digitize(qv, edges[1:-1])
    rb = np.digitize(rv, edges[1:-1])
    for b in range(len(edges) - 1):
        mq, mr = qb == b, rb == b
        rows.append(
            {
                "var": name,
                "bin": f"({edges[b]},{edges[b + 1]}]",
                "n_engine": int(mq.sum()),
                "engine_share": float(mq.mean()),
                "real_share": float(mr.mean()),
                "E_head": float(hd_q[mq].mean()) if mq.any() else None,
                "E_kernel": float(e_q[mq].mean()) if mq.any() else None,
                "head_minus_kernel": float(d_q[mq].mean()) if mq.any() else None,
                "real_obs": float(obs_r[mr].mean()) if mr.any() else None,
            }
        )
    return rows


def shift_measure(Fq, Fr, seed=0, nref=20000):
    rng = np.random.default_rng(seed)
    mu, sd = Fr.mean(axis=0), Fr.std(axis=0)
    sd = np.where(sd > 0, sd, 1.0)
    Zr, Zq = (Fr - mu) / sd, (Fq - mu) / sd
    ir = rng.permutation(len(Zr))
    nh = min(nref, len(Zr) // 2)
    tree = cKDTree(Zr[ir[nh : 2 * nh]])
    d_rr = tree.query(Zr[ir[:nh]], k=1)[0]
    d_qr = tree.query(Zq, k=1)[0]
    p95 = np.quantile(d_rr, 0.95)
    return {
        "nn_dist_real_to_real_mean": float(d_rr.mean()),
        "nn_dist_engine_to_real_mean": float(d_qr.mean()),
        "ratio_engine_over_real": float(d_qr.mean() / d_rr.mean()),
        "frac_engine_beyond_real_p95": float((d_qr > p95).mean()),
        "real_p95_nn_dist": float(p95),
        "feature_mean_shift_in_real_sd": {
            f: float(v) for f, v in zip(FEAT, (Zq.mean(axis=0)), strict=True)
        },
        "feature_sd_ratio": {f: float(v) for f, v in zip(FEAT, Zq.std(axis=0), strict=True)},
    }


def analyse(L, real, Fr, po_r, mpo):
    e_head = L["p_head_masked"].astype(np.float64) @ mpo
    e_full = L["p_head"].astype(np.float64) @ mpo
    e_kern = L["p_kernel"].astype(np.float64) @ mpo
    d = e_head - e_kern
    Fq = L["f"]
    drawn = L["drawn_cls"].astype(int)
    drawn_pts = np.where(drawn >= 0, mpo[np.maximum(drawn, 0)], np.nan)
    qv = {"yl": Fq[:, 0], "sd": Fq[:, 1], "secs": np.expm1(Fq[:, 2]), "half": Fq[:, 6] + 1.0}
    rv = {
        "yl": real["yl"].to_numpy(),
        "sd": real["sd"].to_numpy(),
        "secs": real["secs"].to_numpy(),
        "half": real["half"].to_numpy().astype(float),
    }
    tables = []
    for nm, edges in BINS.items():
        tables += bin_table(nm, edges, qv[nm], rv[nm], d, e_kern, e_head, po_r, None)
    jq = (
        np.digitize(qv["yl"], BINS["yl"][1:-1]) * 100
        + np.digitize(qv["sd"], BINS["sd"][1:-1]) * 10
        + np.digitize(qv["half"], [1.5])
    )
    jr = (
        np.digitize(rv["yl"], BINS["yl"][1:-1]) * 100
        + np.digitize(rv["sd"], BINS["sd"][1:-1]) * 10
        + np.digitize(rv["half"], [1.5])
    )
    cells = np.union1d(np.unique(jq), np.unique(jr))
    dq = pd.Series(d).groupby(jq).mean()
    sr = pd.Series(jr).value_counts(normalize=True)
    cellmean = dq.reindex(cells)
    reweighted = float((sr.reindex(cells).fillna(0.0) * cellmean.fillna(0.0)).sum())
    covered = float(sr.reindex(cells).fillna(0.0)[cellmean.notna()].sum())
    return {
        "n_queries": len(d),
        "E_pts_per_drive": {
            "head_masked": float(e_head.mean()),
            "head_full": float(e_full.mean()),
            "kernel": float(e_kern.mean()),
            "head_minus_kernel": float(d.mean()),
            "drawn_outcome": float(np.nanmean(drawn_pts)),
            "real_obs": float(po_r.mean()),
        },
        "head_minus_kernel_at_real_state_shares": reweighted,
        "real_share_covered_by_engine_cells": covered,
        "head_minus_kernel_at_engine_shares": float(d.mean()),
        "covariate_shift": shift_measure(Fq, Fr),
        "bins": tables,
    }


def synthetic(real, Fr, k, n=400, seed=1):
    rng = np.random.default_rng(seed)
    ix = rng.choice(len(real), n)
    f = Fr[ix] + rng.normal(0, 0.5, (n, Fr.shape[1]))
    f[:, 6] = Fr[ix, 6]
    f[:, 7] = Fr[ix, 7]
    ph = rng.dirichlet(np.ones(k), n)
    pk = rng.dirichlet(np.ones(k), n)
    pm = ph * (pk > 0.01)
    pm = pm / pm.sum(axis=1, keepdims=True)
    return {
        "f": f,
        "p_head": ph.astype(np.float32),
        "p_head_masked": pm.astype(np.float32),
        "p_kernel": pk.astype(np.float32),
        "drawn_cls": rng.integers(0, k, n).astype(np.int16),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", default="")
    ap.add_argument("--synthetic", action="store_true")
    a = ap.parse_args()
    real, Fr, po_r, mpo, k = real_reference()
    if a.synthetic:
        res = analyse(synthetic(real, Fr, k), real, Fr, po_r, mpo)
        print(json.dumps({kk: vv for kk, vv in res.items() if kk != "bins"}, indent=1))
        print(pd.DataFrame(res["bins"]).head(6).to_string())
        return
    d = REPO / "artifacts" / "sim06" / "query_log" / a.label
    res = analyse(load_log(d), real, Fr, po_r, mpo)
    (d.parent / f"{a.label}_analysis.json").write_text(json.dumps(res, indent=1))
    print(json.dumps({kk: vv for kk, vv in res.items() if kk != "bins"}, indent=1))
    print(pd.DataFrame(res["bins"]).to_string())


if __name__ == "__main__":
    main()
