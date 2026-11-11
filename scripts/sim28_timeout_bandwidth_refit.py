import argparse
import json
import os
import sys
from pathlib import Path

os.environ["OMP_NUM_THREADS"] = "1"
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import pandas as pd
import sim10_drive as E

A = Path("artifacts/sim10")
SIDE = Path("artifacts/sim28")
N_BOOT = 400
GRID = [0.25 * 2 ** (k / 2) for k in range(0, 28)]
Q0 = 600
TO_DIMS = [3, 4]
SD_DIM = 1


def load():
    seg = pd.read_parquet(A / "segments.parquet")
    cls = E.outcome_classes(seg)
    keep = np.flatnonzero((seg["bad"].to_numpy() == 0) & (seg["half"].to_numpy() < 3))
    seg = seg.iloc[keep].reset_index(drop=True)
    cls = cls[keep]
    return {
        "X": E.design(seg),
        "cls": cls,
        "half": seg["half"].to_numpy(),
        "home": seg["home"].to_numpy(),
        "season": seg["season"].to_numpy(),
        "he": seg["he"].to_numpy(),
        "gid": pd.factorize(seg["game_id"])[0],
        "reg": np.array(
            [E.regime(h_, s_) for h_, s_ in zip(seg["half"], seg["secs"], strict=True)]
        ),
    }


def per_query(D, qidx, h, alpha):
    X, cls, half, home, season, he = (D[k] for k in ("X", "cls", "half", "home", "season", "he"))
    K = int(cls.max()) + 1
    gc = np.bincount(cls, minlength=K) + 0.5
    out = np.full(len(X), np.nan)
    for hf in (1, 2):
        for hm in (0, 1):
            st = (half == hf) & (home == hm)
            for s in E.SEASONS:
                qi = qidx[st[qidx] & (season[qidx] == s)]
                if len(qi) == 0:
                    continue
                pool = np.flatnonzero(st & (season != s))
                oh = np.zeros((len(pool), K))
                oh[np.arange(len(pool)), cls[pool]] = 1.0
                pri = (oh.sum(axis=0) + gc / gc.sum() * 0.5 * K) / (len(pool) + 0.5 * K)
                z = np.zeros((len(qi), len(pool)))
                for d, hd in zip(E.FIT_DIMS, h, strict=True):
                    if np.isfinite(hd):
                        z += ((X[qi, d][:, None] - X[pool, d][None, :]) / hd) ** 2
                W = np.exp(-0.5 * z)
                W *= ~((he[pool][None, :] == 1) & (X[pool, 2][None, :] < X[qi, 2][:, None]))
                cnt = W @ oh
                p = (cnt[np.arange(len(qi)), cls[qi]] + alpha * pri[cls[qi]]) / (
                    W.sum(axis=1) + alpha
                )
                out[qi] = np.log(p)
    return out[qidx]


def profile(D, qidx, h, alpha, dim):
    j = E.FIT_DIMS.index(dim)
    grid = [*GRID, np.inf]
    mat = np.zeros((len(grid), len(qidx)))
    for i, g in enumerate(grid):
        hh = list(h)
        hh[j] = g
        mat[i] = per_query(D, qidx, hh, alpha)
    return grid, mat


def boot(D, qidx, grid, mat, rng):
    ug, inv = np.unique(D["gid"][qidx], return_inverse=True)
    gs = np.zeros((len(grid), len(ug)))
    for i in range(len(grid)):
        gs[i] = np.bincount(inv, weights=mat[i], minlength=len(ug))
    cap = GRID[-1] * 2**0.5
    val = np.array([cap if not np.isfinite(v) else v for v in grid])
    best = []
    for _ in range(N_BOOT):
        w = np.bincount(rng.integers(0, len(ug), len(ug)), minlength=len(ug))
        best.append(val[int(np.argmax(gs @ w))])
    best = np.array(best)
    pt = val[int(np.argmax(gs.sum(axis=1)))]
    lo, hi = np.percentile(best, [2.5, 97.5])
    return float(pt), float(lo), float(hi), float(np.log2(hi / lo)), gs.sum(axis=1), val


def fit_pair(D, qidx, h, alpha, rng):
    res = {}
    for _ in range(2):
        for d in TO_DIMS:
            grid, mat = profile(D, qidx, h, alpha, d)
            pt, lo, hi, w, _tot, val = boot(D, qidx, grid, mat, rng)
            h[E.FIT_DIMS.index(d)] = pt if pt < val[-1] else np.inf
            res[d] = (pt, lo, hi, w)
    return res


def run(D, r, hbase, alpha, seed, tag):
    cand = np.flatnonzero(D["reg"] == r)
    rng = np.random.default_rng(seed + r)
    perm = rng.permutation(cand)
    steps, q, prev = [], Q0, None
    while True:
        q = min(q, len(cand))
        qidx = np.sort(perm[:q])
        h = list(hbase)
        res = fit_pair(D, qidx, h, alpha, rng)
        width = max(v[3] for v in res.values())
        steps.append(
            {
                "queries": q,
                "too": res[3][:3],
                "tod": res[4][:3],
                "log2_width": width,
                "h": [float(v) for v in h],
            }
        )
        print(
            f"{tag} regime {r} q {q} too {res[3][0]:.3g} [{res[3][1]:.3g},{res[3][2]:.3g}]"
            f" tod {res[4][0]:.3g} [{res[4][1]:.3g},{res[4][2]:.3g}] w {width:.2f}",
            flush=True,
        )
        if q >= len(cand) or (prev is not None and width >= prev):
            break
        prev, q = width, q * 2
    qidx = np.sort(perm[:q])
    base_ll = float(per_query(D, qidx, list(hbase), alpha).mean())
    new_ll = float(per_query(D, qidx, steps[-1]["h"], alpha).mean())
    return {
        "regime": r,
        "n_candidates": len(cand),
        "queries_used": q,
        "steps": steps,
        "start_h": [float(v) for v in hbase],
        "final_h": steps[-1]["h"],
        "ll_start": base_ll,
        "ll_final": new_ll,
        "alpha": alpha,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--regimes", default="1,4")
    ap.add_argument("--seed", type=int, default=2801)
    ap.add_argument("--build", action="store_true")
    a = ap.parse_args()
    base = json.loads((A / "bandwidths_week.json").read_text())
    v2 = json.loads((A / "bandwidths_week_v2.json").read_text())
    SIDE.mkdir(exist_ok=True)
    if a.build:
        v3, v4 = json.loads(json.dumps(base)), json.loads(json.dumps(base))
        for r in [int(x) for x in a.regimes.split(",")]:
            for tag, tgt in (("a", v3), ("b", v4)):
                f = json.loads((SIDE / f"refit_{tag}_r{r}.json").read_text())
                full = list(base["h_reg"][str(r)]["h"])
                for d in TO_DIMS:
                    full[d] = f["final_h"][E.FIT_DIMS.index(d)]
                if tag == "b":
                    full[SD_DIM] = v2["h_reg"][str(r)]["h"][SD_DIM]
                tgt["h_reg"][str(r)]["h"] = [float(v) for v in full]
        (A / "bandwidths_week_v3_to.json").write_text(json.dumps(v3, indent=1))
        (A / "bandwidths_week_v4_to_sd.json").write_text(json.dumps(v4, indent=1))
        print("wrote v3_to and v4_to_sd", flush=True)
        return
    D = load()
    for r in [int(x) for x in a.regimes.split(",")]:
        alpha = base["h_reg"][str(r)]["alpha"]
        ha = [base["h_reg"][str(r)]["h"][d] for d in E.FIT_DIMS]
        hb = list(ha)
        hb[E.FIT_DIMS.index(SD_DIM)] = v2["h_reg"][str(r)]["h"][SD_DIM]
        for tag, h0 in (("a", ha), ("b", hb)):
            rec = run(D, r, h0, alpha, a.seed, tag)
            (SIDE / f"refit_{tag}_r{r}.json").write_text(json.dumps(rec, indent=1))


main()
