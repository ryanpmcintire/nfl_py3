import json
import math
import os
import sys
import time
import warnings
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
for k_ in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(k_, "1")
os.environ.setdefault("SIM10_SEASONS", "2009-2017")
os.environ.setdefault("SIM10_OUT", str(REPO / "artifacts" / "sim10"))
sys.path.insert(0, str(REPO / "scripts"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import sim10_drive as s10  # noqa: E402
from sklearn.neural_network import MLPClassifier  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

warnings.filterwarnings("ignore")
NQ = int(os.environ.get("TILT_NQ", "300"))
LAMBDAS = (0.0, 0.25, 0.5, 0.75, 1.0, 1.25)
OUTFILE = REPO / "artifacts" / "sim06" / "tilt_refit.json"


def log(*a):
    print(*a, flush=True)


def head_fold(seg, fit, spec):
    cls, k = s10.head_codes(seg, fit)
    F = s10.head_features(seg)
    m = make_pipeline(
        StandardScaler(),
        MLPClassifier(
            hidden_layer_sizes=tuple(spec["hidden"]),
            alpha=spec["alpha"],
            early_stopping=True,
            random_state=0,
            max_iter=spec["max_iter"],
        ),
    ).fit(F[fit & (cls >= 0)], cls[fit & (cls >= 0)])
    return cls, k, F, m


def head_reweight(cls_pool, w, ph):
    ok = cls_pool >= 0
    mass = np.bincount(cls_pool[ok], weights=w[ok], minlength=len(ph))
    if mass.sum() <= 0:
        return w
    mass = mass / mass.sum()
    pm = ph * (mass > 0)
    if pm.sum() <= 0:
        return w
    pm = pm / pm.sum()
    ratio = np.where(mass > 0, pm / np.where(mass > 0, mass, 1.0), 0.0)
    return np.where(ok, w * ratio[np.maximum(cls_pool, 0)], w)


def main():
    t0 = time.time()
    seg = s10.with_mu(pd.read_parquet(s10.OUT / "segments.parquet"))
    po, _ = s10.segment_points(seg)
    X = s10.design(seg)
    half, home = seg["half"].to_numpy(), seg["home"].to_numpy()
    season = seg["season"].to_numpy().astype(int)
    he = seg["he"].to_numpy() == 1
    secs = seg["secs"].to_numpy()
    ok = (seg["bad"].to_numpy() == 0) & (half < 3)
    reg = np.array(
        [s10.regime(h_, s_) if h_ < 3 else -1 for h_, s_ in zip(half, secs, strict=True)]
    )
    out = s10.outcome_frame(seg)[s10.OUT_TARGETS].to_numpy(np.float64)
    oof = pd.read_parquet(s10.OUT / "oof_targets.parquet")
    m_arr = np.full_like(out, np.nan)
    m_arr[oof.index.to_numpy()] = oof[s10.OUT_TARGETS].to_numpy(np.float64)
    base = json.loads((s10.OUT / "bandwidths_tilt.json").read_text())
    scott = s10.scott_bandwidths(seg[ok])
    hreg = {}
    for r in range(5):
        hr = base["h_reg"][str(r)]
        h = s10.pad_h(hr["h"])
        if not np.isfinite(h[7]):
            h[7] = scott[7]
        hreg[r] = (h, float(hr["alpha"]))
    spec = json.loads((REPO / "artifacts" / "sim06" / "results.json").read_text())["head_selected"]
    fin_dims = [d for d in range(len(s10.DIMS)) if d != 5]
    rng = np.random.default_rng(0)
    rows = []
    for s in range(2009, 2018):
        fit = ok & (season != s)
        cls, K, F, mdl = head_fold(seg, fit, spec)
        gfreq = (np.bincount(cls[cls >= 0], minlength=K) + 0.5) / (int((cls >= 0).sum()) + 0.5 * K)
        cand = np.flatnonzero(ok & (season == s) & (cls >= 0) & np.isfinite(m_arr).all(axis=1))
        qi = np.sort(rng.choice(cand, size=min(NQ, len(cand)), replace=False))
        PH = np.zeros((len(qi), K))
        PH[:, mdl[1].classes_] = mdl.predict_proba(F[qi])
        for j, i in enumerate(qi):
            pool = np.flatnonzero((half == half[i]) & (home == home[i]) & fit)
            h, al = hreg[reg[i]]
            z = np.zeros(len(pool))
            for d in fin_dims:
                if np.isfinite(h[d]):
                    z += ((X[pool, d] - X[i, d]) / h[d]) ** 2
            z[he[pool] & (X[pool, 2] < X[i, 2])] = np.inf
            w = np.exp(-0.5 * (z - z.min()))
            nzm = w > 1e-12
            pn = pool[nzm]
            wk = w[nzm]
            wh = head_reweight(cls[pn], wk, PH[j])
            pc = np.bincount(cls[pool], minlength=K)
            pri = (pc + gfreq * 0.5 * K) / (len(pool) + 0.5 * K)
            tg = m_arr[i]
            mt = X[i, 5:6]
            tgt_obs = out[i]
            for name, wv in (("kernel", wk), ("head", wh)):
                w1 = s10.tilt_weights(wv, X[pn, 5:6], mt)
                u = w1 / w1.sum()
                adj = (u @ out[pn] - u @ m_arr[pn]) * s10.DR_MASK
                for lam in LAMBDAS:
                    tg2 = tg + lam * adj
                    wt = s10.tilt_weights(wv, np.c_[out[pn], X[pn, 5]], np.r_[tg2, mt])
                    num = wt[cls[pn] == cls[i]].sum() + al * pri[cls[i]]
                    ll = math.log(num / (wt.sum() + al))
                    ep = float(wt @ po[pn] / wt.sum())
                    rows.append(
                        (
                            s,
                            name,
                            lam,
                            ll,
                            ep,
                            float(po[i]),
                            adj[1],
                            adj[2],
                            adj[3],
                            float(tgt_obs[1]),
                        )
                    )
        log(f"season {s} done {time.time() - t0:.0f}s queries {len(qi)}")
    df = pd.DataFrame(
        rows, columns=["season", "w", "lam", "ll", "ep", "po", "adj_td", "adj_fg", "adj_ds", "y_td"]
    )
    res = {"nq_per_season": NQ, "lambdas": list(LAMBDAS), "by_season": {}}
    g = df.groupby(["w", "season", "lam"])[["ll", "ep", "po", "adj_td", "adj_fg", "adj_ds"]].mean()
    for name in ("kernel", "head"):
        res["by_season"][name] = {}
        for s in range(2009, 2018):
            tab = g.loc[name].loc[s]
            oth = g.loc[name].drop(index=s, level=0).groupby("lam")["ll"].mean()
            lam_star = float(oth.idxmax())
            res["by_season"][name][s] = {
                "lambda_chosen_loso": lam_star,
                "ll_heldout_at_chosen": float(tab.loc[lam_star, "ll"]),
                "ll_heldout_at_1": float(tab.loc[1.0, "ll"]),
                "ll_insample_at_chosen": float(oth.loc[lam_star]),
                "ep_minus_obs_at_chosen": float(tab.loc[lam_star, "ep"] - tab.loc[lam_star, "po"]),
                "ep_minus_obs_at_1": float(tab.loc[1.0, "ep"] - tab.loc[1.0, "po"]),
                "mean_adj_td_fg_ds_at_1": [
                    float(tab.loc[1.0, c]) for c in ("adj_td", "adj_fg", "adj_ds")
                ],
            }
    res["pooled"] = {}
    for name in ("kernel", "head"):
        d = df[df.w == name]
        res["pooled"][name] = {
            str(lam): {
                "ll": float(d[d.lam == lam].ll.mean()),
                "ep_minus_obs": float((d[d.lam == lam].ep - d[d.lam == lam].po).mean()),
            }
            for lam in LAMBDAS
        }
    kh = df[(df.w == "head") & (df.lam == 1.0)].reset_index(drop=True)
    kk = df[(df.w == "kernel") & (df.lam == 1.0)].reset_index(drop=True)
    dd = (kh.ll - kk.ll).to_numpy()
    res["head_minus_kernel_ll_at_1"] = {
        "mean": float(dd.mean()),
        "by_season": {
            int(s): float(dd[(kh.season == s).to_numpy()].mean())
            for s in sorted(kh.season.unique())
        },
    }
    OUTFILE.write_text(json.dumps(res, indent=1))
    log(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
