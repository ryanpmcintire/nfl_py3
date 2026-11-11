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
SIDE = Path("artifacts/sim27")
SD = 1
N_BOOT = 400
GRID = [0.25 * 2 ** (k / 2) for k in range(0, 28)]
Q0 = 600


def load():
    seg = pd.read_parquet(A / "segments.parquet")
    cls = E.outcome_classes(seg)
    keep = np.flatnonzero((seg["bad"].to_numpy() == 0) & (seg["half"].to_numpy() < 3))
    seg = seg.iloc[keep].reset_index(drop=True)
    cls = cls[keep]
    d = {
        "X": E.design(seg),
        "cls": cls,
        "half": seg["half"].to_numpy(),
        "home": seg["home"].to_numpy(),
        "season": seg["season"].to_numpy(),
        "he": seg["he"].to_numpy(),
        "lg": E.game_lt(seg),
        "gid": pd.factorize(seg["game_id"])[0],
        "reg": np.array(
            [E.regime(h_, s_) for h_, s_ in zip(seg["half"], seg["secs"], strict=True)]
        ),
    }
    return d


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


def mean_ll(D, qidx, h, alpha):
    return float(per_query(D, qidx, h, alpha).mean())


def profile(D, qidx, h, alpha):
    j = E.FIT_DIMS.index(SD)
    grid = [*GRID, np.inf]
    mat = np.zeros((len(grid), len(qidx)))
    for i, g in enumerate(grid):
        hh = list(h)
        hh[j] = g
        mat[i] = per_query(D, qidx, hh, alpha)
    return grid, mat


def boot(D, qidx, grid, mat, rng):
    g = D["gid"][qidx]
    ug, inv = np.unique(g, return_inverse=True)
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
    return float(pt), float(lo), float(hi), float(np.log2(hi / lo))


def coord_search(D, qidx, h, alpha):
    best = mean_ll(D, qidx, h, alpha)
    for gm in ([0.25, 0.5, 0.71, 1.41, 2.0, 4.0], [0.71, 0.84, 1.19, 1.41]):
        improved, rounds = True, 0
        while improved and rounds < 4:
            improved, rounds = False, rounds + 1
            for i in range(len(h) + 1):
                for m in gm:
                    hh, al = list(h), alpha
                    if i < len(h):
                        hh[i] = h[i] * m
                    else:
                        al = alpha * m
                    v = mean_ll(D, qidx, hh, al)
                    if v > best + 1e-6:
                        best, h, alpha, improved = v, hh, al, True
    return best, h, alpha


def run_regime(D, r, base, seed):
    cand = np.flatnonzero(D["reg"] == r)
    rng = np.random.default_rng(seed + r)
    perm = rng.permutation(cand)
    h = [base["h_reg"][str(r)]["h"][d] for d in E.FIT_DIMS]
    alpha = base["h_reg"][str(r)]["alpha"]
    steps = []
    q, prev = Q0, None
    while True:
        q = min(q, len(cand))
        qidx = np.sort(perm[:q])
        grid, mat = profile(D, qidx, h, alpha)
        pt, lo, hi, width = boot(D, qidx, grid, mat, rng)
        steps.append({"queries": q, "sd_h": pt, "lo": lo, "hi": hi, "log2_width": width})
        print(f"regime {r} q {q} sd {pt:.3g} [{lo:.3g},{hi:.3g}] w {width:.2f}", flush=True)
        if q >= len(cand) or (prev is not None and width >= prev):
            break
        prev, q = width, q * 2
    qidx = np.sort(perm[:q])
    h[E.FIT_DIMS.index(SD)] = steps[-1]["sd_h"]
    best, h, alpha = coord_search(D, qidx, h, alpha)
    grid, mat = profile(D, qidx, h, alpha)
    pt, lo, hi, width = boot(D, qidx, grid, mat, rng)
    marg = mean_ll(D, qidx, [np.inf] * len(h), alpha)
    full = np.full(len(E.DIMS), np.inf)
    full[E.FIT_DIMS] = h
    rec = {
        "h": [float(v) for v in full],
        "alpha": alpha,
        "loso_ll": best,
        "marginal_ll": marg,
        "queries": len(qidx),
    }
    side = {"n": len(cand), "steps": steps, "final_sd": [pt, lo, hi, width]}
    print(f"regime {r} final sd {pt:.3g} [{lo:.3g},{hi:.3g}] ll {best:.5f}", flush=True)

    return rec, side


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--regimes", default="4,3,2,1,0")
    ap.add_argument("--seed", type=int, default=2701)
    a = ap.parse_args()
    base = json.loads((A / "bandwidths_week.json").read_text())
    D = load()
    SIDE.mkdir(exist_ok=True)
    for r in [int(x) for x in a.regimes.split(",")]:
        rec, side = run_regime(D, r, base, a.seed)
        (SIDE / f"refit_r{r}.json").write_text(json.dumps({"rec": rec, "side": side}, indent=1))
    files = [SIDE / f"refit_r{r}.json" for r in range(5)]
    if all(f.exists() for f in files):
        new = dict(base)
        new["h_reg"] = {str(r): json.loads(files[r].read_text())["rec"] for r in range(5)}
        (A / "bandwidths_week_v2.json").write_text(json.dumps(new, indent=1))
        print("wrote bandwidths_week_v2.json", flush=True)


main()
