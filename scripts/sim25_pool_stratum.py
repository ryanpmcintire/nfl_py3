import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "src"))
import sim10_drive as sd  # noqa: E402

OUT = REPO / "artifacts" / "sim25"
CLASSES = ["td", "fg", "punt", "turnover", "expiry"]
MAP = {
    "Touchdown": 0,
    "Field goal": 1,
    "Punt": 2,
    "Turnover": 3,
    "Turnover on downs": 3,
    "Missed field goal": 3,
    "Opp touchdown": 3,
    "Safety": 3,
    "End of half": 4,
}
PTS = np.array([7.0, 3.0, 0.0, 0.0, 0.0])
SIGNS = ["lead8+", "lead1-7", "tied", "trail1-7", "trail8+"]
NBOOT = 1000
SEED = 2501
THIN = 30
WEIGHT_FLOOR = 1e-12


def sign_class(s):
    return np.select([s >= 8, s >= 1, s == 0, s >= -7], [0, 1, 2, 3], 4)


def outcomes(seg):
    p = sd.load_pbp()
    games = sd.real_games()
    games = games[games["game_id"].isin(p["game_id"].unique())]
    p = p[p["game_id"].isin(games["game_id"])].reset_index(drop=True)
    fr = []
    for s in sd.SEASONS:
        f = pd.read_parquet(
            sd.PBP_DIR / f"season={s}" / "plays.parquet",
            columns=["game_id", "play_id", "fixed_drive_result"],
        )
        fr.append(f)
    res = pd.concat(fr, ignore_index=True)
    p = p.merge(res, on=["game_id", "play_id"], how="left")
    rows = seg["row"].to_numpy()
    assert (p["game_id"].to_numpy()[rows] == seg["game_id"].to_numpy()).all()
    assert np.allclose(p["yardline_100"].to_numpy(float)[rows], seg["yl"].to_numpy(float))
    cls = p["fixed_drive_result"].map(MAP).to_numpy(float)[rows]
    return cls


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    seg = sd.with_mu(pd.read_parquet(sd.OUT / "segments.parquet"))
    cls = outcomes(seg)
    hv = seg["half"].to_numpy()
    secs = seg["secs"].to_numpy(float)
    sdv = seg["sd"].to_numpy(float)
    ok = (seg["bad"].to_numpy() == 0) & ~np.isnan(cls) & (hv < 3)
    X = sd.design(seg)
    _, _, _, hr = sd.load_h(str(sd.OUT / "bandwidths_week.json"))
    h_reg = [sd.pad_h(hr[str(r)]["h"]) for r in range(5)]
    home = seg["home"].to_numpy()
    qseg = sd.quarter_of(hv, secs)
    he = seg["he"].to_numpy() == 1
    gname, gidx_all = np.unique(seg["game_id"].to_numpy(), return_inverse=True)
    pairs = zip(hv, secs, strict=True)
    regs = np.array([sd.regime(int(a), float(b)) if a < 3 else -1 for a, b in pairs])
    sgn = sign_class(sdv)
    ind = np.zeros((len(seg), 5))
    good = ~np.isnan(cls)
    ind[good, cls[good].astype(int)] = 1.0
    el = seg["el"].to_numpy(float)
    V = np.c_[ind, ind @ PTS, el, np.ones(len(seg))]
    strata = {}
    for hf in (1, 2):
        for hm in (0, 1):
            for qq in (1, 2):
                sel = ok & (hv == hf) & (home == hm) & (qseg == qq)
                strata[(hf, hm, qq)] = np.flatnonzero(sel)

    q_ids = np.flatnonzero(ok & (((hv == 2) & (regs >= 2)) | ((hv == 1) & (regs == 1))))
    nq = len(q_ids)
    dA = np.zeros((nq, 8))
    dB = np.zeros((nq, 8))
    cand = np.zeros((nq, 5))
    for n, i in enumerate(q_ids):
        hf = int(hv[i])
        hm = int(home[i])
        qq = int(qseg[i])
        idx = strata[(hf, hm, qq)]
        Xs = X[idx]
        q = X[i]
        r = regs[i]
        h = h_reg[r]
        fin = np.isfinite(h)
        z = (((Xs[:, fin] - q[fin]) / h[fin]) ** 2).sum(axis=1)
        z[gidx_all[idx] == gidx_all[i]] = np.inf
        early = he[idx] & (Xs[:, 2] < q[2])
        if hf == 2 and q[1] > 0:
            early &= ~(Xs[:, 1] > 0)
        z[early] = np.inf
        w = np.exp(-0.5 * (z - z.min()))
        w[w <= WEIGHT_FLOOR] = 0.0
        same = (sgn[idx] == sgn[i]) & (regs[idx] == r) if hf == 2 else np.zeros(len(idx), bool)
        wb = np.where(same, w, 0.0)
        for arm, ww, dst in ((0, w, dA), (1, wb, dB)):
            s = ww.sum()
            if s <= 0:
                dst[n] = np.nan
                continue
            u = ww / s
            drawn = u @ V[idx][:, :7]
            dst[n, :7] = V[i, :7] - drawn
            dst[n, 7] = 1.0
            cand[n, 2 * arm] = (ww > 0).sum()
            cand[n, 2 * arm + 1] = 1.0 / (u @ u)
        cand[n, 4] = same.sum()

    g_q = gidx_all[q_ids]
    ug, gi = np.unique(g_q, return_inverse=True)
    ng = len(ug)
    W = rng.multinomial(ng, np.full(ng, 1.0 / ng), size=NBOOT).astype(float)
    names = [*CLASSES, "points", "seconds"]

    def cellrows(cellid, ncell, labels, arr):
        A = np.zeros((ng, ncell, 8))
        m = (cellid >= 0) & ~np.isnan(arr[:, 0])
        np.add.at(A, (gi[m], cellid[m]), arr[m])
        tot = A.sum(axis=0)
        B = np.einsum("bg,gck->bck", W, A)
        out = []
        for c in range(ncell):
            n_ = tot[c, 7]
            row = {"cell": labels[c], "n": int(n_)}
            for k, nm in enumerate(names):
                if n_ == 0:
                    row[nm] = None
                    continue
                with np.errstate(invalid="ignore", divide="ignore"):
                    bs = B[:, c, k] / B[:, c, 7]
                bs = bs[np.isfinite(bs)]
                row[nm] = [
                    float(tot[c, k] / n_),
                    float(np.quantile(bs, 0.025)),
                    float(np.quantile(bs, 0.975)),
                ]
            out.append(row)
        return out, tot, B

    late = ((hv[q_ids] == 2) & (regs[q_ids] >= 2)).astype(bool)
    reg2 = regs[q_ids] - 2
    cell25 = np.where(late, reg2 * 5 + sgn[q_ids], -1)
    lab25 = [f"R{r + 2}_{s}" for r in range(3) for s in SIGNS]
    resA, totA, BA = cellrows(cell25, 15, lab25, dA)
    resB, totB, BB = cellrows(cell25, 15, lab25, dB)
    cstat = []
    for c in range(15):
        m = cell25 == c
        cstat.append(
            {
                "cell": lab25[c],
                "queries": int(m.sum()),
                "arm_a_candidates_mean": float(cand[m, 0].mean()),
                "arm_a_effective_n_mean": float(cand[m, 1].mean()),
                "arm_b_candidates_mean": float(cand[m, 2].mean()),
                "arm_b_candidates_min": float(cand[m, 2].min()),
                "arm_b_effective_n_mean": float(cand[m, 3].mean()),
                "stratum_size_mean": float(cand[m, 4].mean()),
                "queries_with_stratum_below_thin": int((cand[m, 4] < THIN).sum()),
            }
        )
    sel_c = [r * 5 + s for r in (1, 2) for s in (0, 1, 3, 4)]
    nn = totA[sel_c, 7]
    gapA = BA[:, sel_c, 5] / BA[:, sel_c, 7]
    gapB = BB[:, sel_c, 5] / BB[:, sel_c, 7]
    wt = nn / nn.sum()
    absA = (np.abs(gapA) * wt).sum(axis=1)
    absB = (np.abs(gapB) * wt).sum(axis=1)
    pa = totA[sel_c, 5] / nn
    pb = totB[sel_c, 5] / totB[sel_c, 7]
    dif = absB - absA
    head = {
        "cells": [lab25[c] for c in sel_c],
        "weighted_abs_points_gap_full": float((np.abs(pa) * wt).sum()),
        "weighted_abs_points_gap_restricted": float((np.abs(pb) * wt).sum()),
        "bootstrap_full": [float(np.quantile(absA, 0.025)), float(np.quantile(absA, 0.975))],
        "bootstrap_restricted": [float(np.quantile(absB, 0.025)), float(np.quantile(absB, 0.975))],
        "restricted_minus_full_boot": [
            float(np.quantile(dif, 0.025)),
            float(np.quantile(dif, 0.975)),
        ],
        "probability_restricted_smaller": float((dif < 0).mean()),
        "signed_pooled_points_gap_full": float(
            (totA[sel_c, 5].sum() / nn.sum()),
        ),
        "signed_pooled_points_gap_restricted": float(totB[sel_c, 5].sum() / totB[sel_c, 7].sum()),
    }
    to = seg["too"].to_numpy(float)[q_ids].clip(0, 3).astype(int)
    c28 = np.where(
        (hv[q_ids] == 1) & (regs[q_ids] == 1),
        to,
        np.where((hv[q_ids] == 2) & (regs[q_ids] == 4), 4 + to, -1),
    )
    lab28 = [f"half{hh}_to{t}" for hh in (1, 2) for t in range(4)]
    res28, _, _ = cellrows(c28, 8, lab28, dA)
    out = {
        "n_queries": int(nq),
        "pool_games": len(gname),
        "real_minus_full_pool_kernel": resA,
        "real_minus_restricted_pool": resB,
        "candidate_counts": cstat,
        "headline": head,
        "sim28_kernel_check_real_minus_kernel_by_offence_timeouts": res28,
        "bandwidth_source": "artifacts/sim10/bandwidths_week.json h_reg",
    }
    (OUT / "pool_stratum.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(head, indent=1))


if __name__ == "__main__":
    main()
