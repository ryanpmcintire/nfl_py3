import json
import os
import sys
from pathlib import Path

os.environ.setdefault("A2_LABEL", "crHpqokgndecsmfwtjo2as2")

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
import mod25e_e93 as e93  # noqa: E402
import sim09_u4g as u4g  # noqa: E402

OUTD = REPO / "artifacts" / "mod25e3" / "e94"
FIT = OUTD / "fit.json"
GRID = (10, 20, 40, 80, 160, 320, 640, 1280, 0)
CLASSES = ("punt", "fgmiss", "to_run", "to_pass", "downs_run", "downs_pass")
CNB = 300
W = {}


def class_of(code, down):
    c = np.full(len(code), "", dtype=object)
    c[code == 2] = "punt"
    c[code == 3] = "fgmiss"
    go = (code == 0) | (code == 1)
    c[go & (down < 4) & (code == 0)] = "to_run"
    c[go & (down < 4) & (code == 1)] = "to_pass"
    c[go & (down >= 4) & (code == 0)] = "downs_run"
    c[go & (down >= 4) & (code == 1)] = "downs_pass"
    return c


def load_class_pool():
    p = pd.read_parquet(u4g.OUT / "pool.parquet", columns=["gid", "qtr", "code", "flip", "down", "yl", "sd"])
    g = p.groupby("gid", sort=False)
    p["nyl"] = g.yl.shift(-1)
    p["nqtr"] = g.qtr.shift(-1)
    p["nsd"] = g.sd.shift(-1)
    ok = p.flip.to_numpy(bool) & np.isin(p.code.to_numpy(), (0, 1, 2, 3)) & p.yl.notna().to_numpy() & p.nyl.notna().to_numpy()
    ok &= (p.nqtr.to_numpy() <= 4) & ~((p.qtr.to_numpy() == 2) & (p.nqtr.to_numpy() == 3)) & (p.nqtr.to_numpy() >= p.qtr.to_numpy())
    ok &= np.abs(p.nsd.to_numpy(float) + p.sd.to_numpy(float)) < 1e-9
    p = p[ok].copy()
    p["cls"] = class_of(p.code.to_numpy(int), p.down.to_numpy(float))
    p["season"] = p.gid.str[:4].astype(int)
    p["yli"] = np.clip(np.round(p.yl.to_numpy(float)).astype(int), 1, 99)
    return p.reset_index(drop=True)


def window_bounds(cnt_cum, c, m):
    if m == 0:
        return 1, 99
    w = 0
    while True:
        lo, hi = max(c - w, 1), min(c + w, 99)
        n = cnt_cum[hi] - cnt_cum[lo - 1]
        if n >= m or (lo == 1 and hi == 99):
            return lo, hi
        w += 1


def crps_rows(pool_sorted, cs, y):
    n = len(pool_sorted)
    k = np.searchsorted(pool_sorted, y, side="right")
    mean_abs = (y * k - cs[k] + (cs[n] - cs[k]) - y * (n - k)) / n
    i = np.arange(1, n + 1)
    e = 2.0 * np.sum((2 * i - n - 1) * pool_sorted) / (n * n)
    return mean_abs - 0.5 * e


def fit_window(sub):
    yli = sub.yli.to_numpy(int)
    ny = sub.nyl.to_numpy(float)
    se = sub.season.to_numpy(int)
    seasons = np.unique(se)
    L = np.zeros((len(seasons), len(GRID)))
    for fi, f in enumerate(seasons):
        te = se == f
        tr = ~te
        o = np.argsort(yli[tr], kind="stable")
        ytr, ntr = yli[tr][o], ny[tr][o]
        cnt = np.bincount(ytr, minlength=100)
        cum = np.cumsum(cnt)
        edges = np.r_[0, cum]
        yte, nte = yli[te], ny[te]
        for gi, m in enumerate(GRID):
            tot = 0.0
            for c in np.unique(yte):
                lo, hi = window_bounds(cum, int(c), m)
                a, b = edges[lo], edges[hi + 1]
                pool = np.sort(ntr[a:b])
                cs = np.r_[0.0, np.cumsum(pool)]
                r = nte[yte == c]
                tot += crps_rows(pool, cs, r).sum()
            L[fi, gi] = tot / te.sum()
    return seasons, L


def descr(sub, say, rng):
    yl = sub.yl.to_numpy(float)
    ny = sub.nyl.to_numpy(float)
    gid = sub.gid.to_numpy()
    ug, gi = np.unique(gid, return_inverse=True)
    Wb = np.vstack([np.ones((1, len(ug))), rng.multinomial(len(ug), np.ones(len(ug)) / len(ug), size=CNB).astype(float)])
    one = np.bincount(gi, minlength=len(ug)).astype(float)
    sx, sy = np.bincount(gi, weights=yl, minlength=len(ug)), np.bincount(gi, weights=ny, minlength=len(ug))
    sxx, sxy = np.bincount(gi, weights=yl * yl, minlength=len(ug)), np.bincount(gi, weights=yl * ny, minlength=len(ug))
    n, mx, my = Wb @ one, (Wb @ sx), (Wb @ sy)
    mx, my = mx / n, my / n
    slope = ((Wb @ sxy) / n - mx * my) / ((Wb @ sxx) / n - mx * mx)
    lo, hi = np.percentile(slope[1:], [2.5, 97.5])
    say(f"  slope of next start on yl {slope[0]:.3f} [{lo:.3f},{hi:.3f}] n {len(sub)}")
    bands = ((1, 20), (21, 40), (41, 60), (61, 80), (81, 99))
    for a, b in bands:
        m = (yl >= a) & (yl <= b)
        if m.sum() < 20:
            continue
        net = yl[m] - 100.0 + ny[m]
        say(f"    yl {a}-{b}: n {int(m.sum())} mean next start {ny[m].mean():.1f} mean (yl-100+next) {net.mean():.1f} share next==80 {np.mean(ny[m] == 80.0):.3f}")


def build_windows(p):
    out = {}
    for c in CLASSES:
        sub = p[p.cls == c].sort_values("yli", kind="stable")
        yli = sub.yli.to_numpy(int)
        cnt = np.bincount(yli, minlength=100)
        cum = np.cumsum(cnt)
        edges = np.r_[0, cum]
        out[c] = {"nyl": sub.nyl.to_numpy(float), "yl": sub.yl.to_numpy(float), "cum": cum, "edges": edges}
    return out


def replay(d):
    rng = e93.rng
    d = d.sort_values("g", kind="stable").reset_index(drop=True)
    ix = d.idx.to_numpy().astype(int)
    g = d.g.to_numpy()
    yl0 = d.yl.to_numpy(float)
    yl = yl0.copy()
    code = d.code.to_numpy(int)
    cand = (d.flip == 1).to_numpy() & (d.po == 0).to_numpy() & (d.pdf == 0).to_numpy() & (d.qtr <= 4).to_numpy() & np.r_[g[1:] == g[:-1], False] & np.isin(code, (0, 1, 2, 3))
    ci = np.flatnonzero(cand)
    nxt = np.minimum(ci + 1, len(d) - 1)
    pi = ix[ci]
    down = d.down.to_numpy(float)
    ok = (e93.POOL["flip"][pi]) & (e93.POOL["code"][pi] == code[ci]) & (e93.POOL["down"][pi] == down[ci])
    e93.CNT["cand"] += len(ci)
    e93.CNT["pool_ok"] += int(ok.sum())
    e93.CNT["match"] += int((ok & (np.abs(yl0[nxt] - e93.POOL["nyl"][pi]) < 1e-6)).sum())
    ci, nxt, pi = ci[ok], nxt[ok], pi[ok]
    cd = code[ci]
    cl = class_of(cd, down[ci])
    z = np.nan_to_num(yl0[nxt] - e93.POOL["nyl"][pi])
    tb = e93.POOL["tb"]
    ys = np.clip(np.round(yl0[ci]).astype(int), 1, 99)
    s = np.zeros(len(ci))
    for c in CLASSES:
        m = np.flatnonzero(cl == c)
        if len(m) == 0:
            continue
        T = W["tabs"][c]
        mg = W["fit"][c]
        lo = np.empty(len(m), int)
        hi = np.empty(len(m), int)
        for c0 in np.unique(ys[m]):
            l, h = window_bounds(T["cum"], int(c0), mg)
            sel = ys[m] == c0
            lo[sel], hi[sel] = T["edges"][l], T["edges"][h + 1]
        j = lo + np.minimum(np.floor(rng.random(len(m)) * (hi - lo)).astype(int), hi - lo - 1)
        new = T["nyl"][j]
        s[m] = np.where(new == tb, new, np.where(new + z[m] >= 100.0, tb, new + z[m]))
    s = np.clip(s, 1.0, 99.0)
    yl[nxt] = s
    e93.CNT["applied"] += len(ci)
    e93.CNT["punt"] += int((cd == 2).sum())
    e93.CNT["to"] += int(np.isin(cd, (0, 1)).sum())
    e93.CNT["fgmiss"] += int((cd == 3).sum())
    d["yl"] = yl
    return d


def main():
    OUTD.mkdir(parents=True, exist_ok=True)
    out = []

    def say(s=""):
        print(s)
        out.append(s)

    rng = np.random.default_rng(94)
    p = load_class_pool()
    say(f"E94 redraw flip outcome from pool rows of the same class within a fitted yardline window; label {e93.a2.LABEL} seeds {e93.a2.SEEDS}")
    say("part 1 real data (pool, flip rows with a valid next-drive start): dependence on yardline and window fit by leave-one-season-out CRPS of next start")
    fit = {}
    looks = 0
    for c in CLASSES:
        sub = p[p.cls == c]
        say(f"class {c}")
        descr(sub, say, rng)
        seasons, L = fit_window(sub)
        mean = L.mean(axis=0)
        pooled = L[:, GRID.index(0)]
        bi = int(mean.argmin())
        nested = np.zeros(len(seasons))
        for k in range(len(seasons)):
            others = np.delete(np.arange(len(seasons)), k)
            nested[k] = L[k, int(L[others].mean(axis=0).argmin())]
        looks += len(GRID)
        fit[c] = int(GRID[bi])
        say("  LOSO CRPS by min group (" + ", ".join("all" if m == 0 else str(m) for m in GRID) + "): " + ", ".join(f"{v:.3f}" for v in mean))
        say(f"  chosen min group {'all' if GRID[bi] == 0 else GRID[bi]}; nested LOSO CRPS {nested.mean():.3f} vs pooled-all {pooled.mean():.3f} vs chosen {mean[bi]:.3f}; seasons chosen beats pooled {int((L[:, bi] < pooled).sum())}/{len(seasons)}")
    FIT.write_text(json.dumps(fit), encoding="utf-8")
    say(f"fit looks {looks}")
    W["fit"] = fit
    W["tabs"] = build_windows(p)
    e93.OUTD = OUTD
    e93.replay = replay
    e93.main()
    src = OUTD / f"e93_{e93.FGMODE}_{e93.TBMODE}.txt"
    out.append("part 2 offline replay (flip outcome redrawn within window, STF z carried; real vs base vs replay)")
    out.append(src.read_text())
    src.unlink()
    (OUTD / "e94.txt").write_text("\n".join(out), encoding="utf-8")


if __name__ == "__main__":
    main()
