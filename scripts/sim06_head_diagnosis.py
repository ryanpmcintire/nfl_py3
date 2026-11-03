import json
import os
import sys
import time
import warnings
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
os.environ["SIM10_SEASONS"] = "2009-2017"
os.environ["SIM10_OUT"] = str(REPO / "artifacts" / "sim10")
for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_k] = "1"
sys.path.insert(0, str(REPO / "scripts"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import sim06_learned_head as h6  # noqa: E402
import sim10_drive as s10  # noqa: E402

warnings.filterwarnings("ignore")
NQ = int(os.environ.get("DIAG_NQ", "900"))
T0 = time.time()


def log(*a):
    print(f"[{time.time() - T0:5.0f}s]", *a, flush=True)


def main():
    seg = pd.read_parquet(s10.OUT / "segments.parquet")
    po, pd_ = s10.segment_points(seg)
    keep = (seg["bad"].to_numpy() == 0) & (seg["half"].to_numpy() < 3)
    seg = seg[keep].reset_index(drop=True)
    po, pd_ = po[keep], pd_[keep]
    season = seg["season"].to_numpy().astype(int)
    X = s10.design(seg)
    half, home = seg["half"].to_numpy(), seg["home"].to_numpy()
    he = seg["he"].to_numpy()
    lg = s10.game_lt(seg)
    F = h6.head_features(seg)
    spec = json.loads((REPO / "artifacts" / "sim06" / "results.json").read_text())["head_selected"]
    bw = json.loads((REPO / "artifacts" / "sim10" / "bandwidths_week.json").read_text())
    hA, alphaA, gA = list(bw["h"]), float(bw["alpha"]), float(bw["g_sd"])
    rng = np.random.default_rng(0)
    rows = []
    for s in range(2009, 2018):
        fit = season != s
        code = h6.class_codes(seg, po, pd_, fit)
        cls, uniq = h6.relabel(code, fit)
        K = len(uniq)
        floor = 1.0 / (int(fit.sum()) + K)
        ok = fit & (cls >= 0)
        mpo = np.bincount(cls[ok], weights=po[ok], minlength=K) / np.maximum(
            np.bincount(cls[ok], minlength=K), 1
        )
        mpd = np.bincount(cls[ok], weights=pd_[ok], minlength=K) / np.maximum(
            np.bincount(cls[ok], minlength=K), 1
        )
        grp = uniq // 10
        qi = np.sort(
            rng.choice(
                np.flatnonzero(season == s), size=min(NQ, int((season == s).sum())), replace=False
            )
        )
        qm = np.zeros(len(seg), bool)
        qm[qi] = True
        PK = h6.kernel_fold(
            X, cls, half, home, he, lg, qm, fit, K, hA, alphaA, gA, floor, full=True
        )
        PH = h6.head_dist(spec, F[fit], cls[fit], F[qi], K, floor)
        mk, mh, nsup, neff, nfull = [], [], [], [], []
        for j, i in enumerate(qi):
            st = (half == half[i]) & (home == home[i]) & fit & (cls >= 0)
            pi = np.flatnonzero(st)
            z = np.zeros(len(pi))
            for d_, hd in zip(s10.FIT_DIMS, hA, strict=True):
                if not np.isfinite(hd):
                    continue
                hq = hd * np.exp(gA * (lg[i] - s10.L0)) if d_ == 1 else hd
                z += ((X[pi, d_] - X[i, d_]) / hq) ** 2
            z[(he[pi] == 1) & (X[pi, 2] < X[i, 2])] = np.inf
            w = np.exp(-0.5 * (z - z.min()))
            nz = w > 1e-12
            w, pc = w[nz], cls[pi[nz]]
            mass = np.bincount(pc, weights=w, minlength=K)
            mass = mass / mass.sum()
            pm = PH[j] * (mass > 0)
            pm = pm / pm.sum()
            mk.append(mass @ mpo)
            mh.append(pm @ mpo)
            nsup.append(int((mass > 0).sum()))
            neff.append(float(w.sum() ** 2 / (w**2).sum()))
            nfull.append(float(np.minimum(PH[j][mass == 0].sum(), 1.0)))
        for j, i in enumerate(qi):
            rows.append(
                (
                    s,
                    "k_support",
                    mk[j],
                    0.0,
                    0.0,
                    0.0,
                    np.nan,
                    seg["yl"].iat[i],
                    seg["sd"].iat[i],
                    seg["secs"].iat[i],
                    half[i],
                )
            )
            rows.append(
                (
                    s,
                    "h_masked",
                    mh[j],
                    0.0,
                    0.0,
                    0.0,
                    np.nan,
                    seg["yl"].iat[i],
                    seg["sd"].iat[i],
                    seg["secs"].iat[i],
                    half[i],
                )
            )
            rows.append(
                (
                    s,
                    "supp_n",
                    nsup[j],
                    neff[j],
                    nfull[j],
                    0.0,
                    np.nan,
                    seg["yl"].iat[i],
                    seg["sd"].iat[i],
                    seg["secs"].iat[i],
                    half[i],
                )
            )
        for name, P in (("kernel", PK), ("head", PH)):
            ept = P @ mpo
            epd = P @ mpd
            td = P[:, grp == 5].sum(1) + P[:, (grp == 2)].sum(1)
            fg = P[:, grp == 1].sum(1)
            for j, i in enumerate(qi):
                rows.append(
                    (
                        s,
                        name,
                        ept[j],
                        epd[j],
                        td[j],
                        fg[j],
                        P[j, cls[i]] if cls[i] >= 0 else floor,
                        seg["yl"].iat[i],
                        seg["sd"].iat[i],
                        seg["secs"].iat[i],
                        half[i],
                    )
                )
        for j, i in enumerate(qi):  # noqa: B007
            rows.append(
                (
                    s,
                    "obs",
                    po[i],
                    pd_[i],
                    float(po[i] >= 6),
                    float(po[i] == 3),
                    np.nan,
                    seg["yl"].iat[i],
                    seg["sd"].iat[i],
                    seg["secs"].iat[i],
                    half[i],
                )
            )
        log("season", s, "K", K)
    d = pd.DataFrame(
        rows,
        columns=[
            "season",
            "model",
            "ep_off",
            "ep_def",
            "p_td",
            "p_fg",
            "p_true",
            "yl",
            "sd",
            "secs",
            "half",
        ],
    )
    n = d.groupby("model").size()
    print(d.groupby("model")[["ep_off", "ep_def", "p_td", "p_fg"]].mean().round(4))
    print(d.groupby(["season", "model"]).ep_off.mean().unstack().round(3).to_string())
    print(d.groupby("model")["p_true"].apply(lambda v: np.log(v).mean()).round(4))
    pv = d.pivot_table(index="season", columns="model", values="ep_off")
    pv["head_minus_obs"] = pv["head"] - pv["obs"]
    pv["kernel_minus_obs"] = pv["kernel"] - pv["obs"]
    pv["head_minus_kernel"] = pv["head"] - pv["kernel"]
    print(pv.round(3))
    print("n", n.to_dict())
    o = d[d.model == "obs"].reset_index(drop=True)
    for nm, edges in (
        ("yl", [0, 25, 50, 75, 100]),
        ("sd", [-100, -14, -7, -1, 0, 7, 14, 100]),
        ("secs", [-1, 120, 300, 600, 1200, 1800]),
        ("half", [0, 1, 2]),
    ):
        b = pd.cut(o[nm], edges)
        t = pd.DataFrame({"n": o.groupby(b).size(), "obs": o.groupby(b).ep_off.mean()})
        for m in ("head", "kernel"):
            x = d[d.model == m].reset_index(drop=True)
            t[m + "_minus_obs"] = x.groupby(b).ep_off.mean() - t["obs"]
        print(nm)
        print(t.round(3))

    return d


if __name__ == "__main__":
    main()
