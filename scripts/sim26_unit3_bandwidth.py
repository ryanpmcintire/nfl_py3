import json
import os

os.environ["OMP_NUM_THREADS"] = "1"
import numpy as np
import pandas as pd

A = "artifacts/sim10/"
N_BOOT = 300
N_Q = 3000
CHUNK = 400
TD_PTS = 6.0
LADDER = [-3, -2, -1, 0, 1, 2, 3]
rng = np.random.default_rng(2603)
with open("artifacts/sim19/trailing_risk.json") as fh0:
    edges = json.load(fh0)
LE = np.array(edges["deficit_edges"])
TE = np.array(edges["secs_edges"])
T9, T24 = float(TE[0]), float(TE[1])
NEUTRAL = float(LE[0])
TWO_MIN, Q4 = 120.0, 900.0
with open(A + "bandwidths_week.json") as fh1:
    bw = json.load(fh1)
HREG = {int(k): np.array(v["h"], dtype=float) for k, v in bw["h_reg"].items()}

seg = pd.read_parquet(A + "segments.parquet")
c = np.load(A + "csr.npz")
ev_ptr, ev = c["ev_ptr"], c["ev"]
n = len(seg)
rid = np.repeat(np.arange(n), np.diff(ev_ptr))
po = np.bincount(rid, weights=ev[:, 1], minlength=n)
td = np.bincount(rid, weights=(ev[:, 1] == TD_PTS).astype(float), minlength=n) > 0
seg["po"] = po
seg["td"] = td.astype(float)
seg["sc"] = (po > 0).astype(float)
ok = (seg.bad == 0).to_numpy()
X = np.c_[
    seg["yl"],
    seg["sd"],
    np.log1p(seg["secs"].clip(lower=0.0)),
    seg["too"],
    seg["tod"],
    seg["x"],
    seg["season"],
    seg["week"],
].astype(np.float64)
T = seg[["td", "sc", "po"]].to_numpy(np.float64)
gid = pd.factorize(seg.game_id)[0]
he = seg.he.to_numpy() == 1
half = seg.half.to_numpy()
home = seg.home.to_numpy()
secs = seg.secs.to_numpy()


def regime(s):
    if s > Q4:
        return 2
    return 3 if s > TWO_MIN else 4


reg = np.array([regime(s) if h == 2 else -1 for h, s in zip(half, secs, strict=True)])


def kernel_pred(qi, pool, h, fin):
    out = np.zeros((len(qi), T.shape[1]))
    for a in range(0, len(qi), CHUNK):
        q = qi[a : a + CHUNK]
        z = np.zeros((len(q), len(pool)))
        for d in np.flatnonzero(fin):
            z += ((X[q, d][:, None] - X[pool, d][None, :]) / h[d]) ** 2
        z[gid[q][:, None] == gid[pool][None, :]] = np.inf
        z[he[pool][None, :] & (X[pool, 2][None, :] < X[q, 2][:, None])] = np.inf
        w = np.exp(-0.5 * (z - z.min(axis=1, keepdims=True)))
        out[a : a + CHUNK] = (w @ T[pool]) / w.sum(axis=1, keepdims=True)
    return out


def profile(r):
    base = HREG[r]
    sel = ok & (half == 2) & (reg == r)
    qall = np.flatnonzero(sel)
    qi = np.sort(rng.choice(qall, size=min(N_Q, len(qall)), replace=False))
    var = T[qall].var(axis=0)
    sd_scale = X[qall, 1].std()
    lt_scale = X[qall, 2].std()
    sd_grid = [*sorted({*(sd_scale * 2.0**k for k in LADDER), base[1]}), np.inf]
    lt_grid = sorted({*(lt_scale * 2.0**k for k in LADDER), base[2]})
    loss = np.zeros((len(qi), len(sd_grid), len(lt_grid)))
    for hm in (0, 1):
        pool = np.flatnonzero(ok & (half == 2) & (home == hm))
        qh = np.flatnonzero(home[qi] == hm)
        for a in range(0, len(qh), CHUNK):
            q = qi[qh[a : a + CHUNK]]
            D = {
                d: ((X[q, d][:, None] - X[pool, d][None, :]) ** 2).astype(np.float32)
                for d in range(X.shape[1])
                if np.isfinite(base[d]) or d in (1, 2)
            }
            msk = gid[q][:, None] == gid[pool][None, :]
            msk |= he[pool][None, :] & (X[pool, 2][None, :] < X[q, 2][:, None])
            for i, hs in enumerate(sd_grid):
                for j, hl in enumerate(lt_grid):
                    z = np.zeros(msk.shape, dtype=np.float32)
                    for d, Dd in D.items():
                        hd = hs if d == 1 else (hl if d == 2 else base[d])
                        if np.isfinite(hd):
                            z += Dd / np.float32(hd * hd)
                    z[msk] = np.inf
                    w = np.exp(-0.5 * (z - z.min(axis=1, keepdims=True))).astype(np.float64)
                    p = (w @ T[pool]) / w.sum(axis=1, keepdims=True)
                    loss[qh[a : a + CHUNK], i, j] = (((p - T[q]) ** 2) / var).sum(axis=1)
        print("profile regime", r, "home", hm, flush=True)
    return qi, sd_grid, lt_grid, loss, base


def boot_argmin(qi, loss):
    g = gid[qi]
    ug, inv = np.unique(g, return_inverse=True)
    gs = np.zeros((len(ug), *loss.shape[1:]))
    np.add.at(gs, inv, loss)
    cnt = np.bincount(inv, minlength=len(ug))
    outs = []
    for _ in range(N_BOOT):
        pick = rng.integers(0, len(ug), len(ug))
        tot = gs[pick].sum(axis=0) / cnt[pick].sum()
        outs.append(np.unravel_index(tot.argmin(), tot.shape))
    return np.array(outs), gs.sum(axis=0) / len(qi)


res = {"engine_file": "bandwidths_week.json", "n_boot": N_BOOT, "n_queries_per_regime": N_Q}
chosen = {}
for r in (3, 4):
    qi, sdg, ltg, loss, base = profile(r)
    bo, mean_loss = boot_argmin(qi, loss)
    i0, j0 = np.unravel_index(mean_loss.argmin(), mean_loss.shape)
    ie = sdg.index(base[1])
    je = ltg.index(base[2])
    sdv = np.array([sdg[i] for i in bo[:, 0]])
    ltv = np.array([ltg[j] for j in bo[:, 1]])
    fin_sd = np.isfinite(sdv)
    res[f"regime_{r}"] = {
        "engine_sd": float(base[1]),
        "engine_lt": float(base[2]),
        "chosen_sd": float(sdg[i0]),
        "chosen_lt": float(ltg[j0]),
        "sd_boot_ci95": [float(np.percentile(sdv, 2.5)), float(np.percentile(sdv, 97.5))],
        "sd_boot_inf_share": float(1 - fin_sd.mean()),
        "lt_boot_ci95": [float(np.percentile(ltv, 2.5)), float(np.percentile(ltv, 97.5))],
        "loss_engine": float(mean_loss[ie, je]),
        "loss_chosen": float(mean_loss[i0, j0]),
        "loss_flat_sd": float(mean_loss[-1, j0]),
        "sd_grid": [float(v) for v in sdg],
        "lt_grid": [float(v) for v in ltg],
        "loss_surface": mean_loss.tolist(),
        "sd_scale": float(X[ok & (half == 2) & (reg == r), 1].std()),
    }
    chosen[r] = (sdg[i0], ltg[j0])
    print("regime", r, sdg[i0], ltg[j0], flush=True)

late = np.flatnonzero(ok & (half == 2) & (secs <= T24))
pred = {}
for name in ("engine", "measured"):
    out = np.zeros((len(late), 3))
    for r in (2, 3, 4):
        h = HREG[r].copy()
        if name == "measured" and r in chosen:
            h[1], h[2] = chosen[r]
        fin = np.isfinite(h)
        for hm in (0, 1):
            pool = np.flatnonzero(ok & (half == 2) & (home == hm))
            m = np.flatnonzero((reg[late] == r) & (home[late] == hm))
            if len(m):
                out[m] = kernel_pred(late[m], pool, h, fin)
    pred[name] = out
    print("late kernel", name, flush=True)

sdq = seg.sd.to_numpy()[late]
clk = {"final9": secs[late] <= T9, "min9_24": (secs[late] > T9) & (secs[late] <= T24)}
bins = {
    "trail_5_7": (sdq <= -5) & (sdq >= -7),
    "trail_8_14": (sdq <= -8) & (sdq >= -14),
    "trail_15p": sdq <= -15,
    "neutral": np.abs(sdq) <= NEUTRAL,
    "lead_5_7": (sdq >= 5) & (sdq <= 7),
    "lead_8_14": (sdq >= 8) & (sdq <= 14),
    "lead_15p": sdq >= 15,
}
glate = gid[late]
yt = T[late]
ug, inv = np.unique(glate, return_inverse=True)


def ci(a):
    a = np.asarray(a)
    return [float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))]


def stat(cm, bm, col, p):
    sel = cm & bm
    cnt = np.bincount(inv, weights=sel.astype(float), minlength=len(ug))
    v = p[:, col] if p is not None else yt[:, col]
    s = np.bincount(inv, weights=np.where(sel, v, 0.0), minlength=len(ug))
    return cnt, s


names = {0: "td", 1: "score"}
cells = {}
for cn, cm in clk.items():
    for col, mn in names.items():
        cells[(cn, mn)] = {
            bn: {
                "real": stat(cm, bm, col, None),
                "engine": stat(cm, bm, col, pred["engine"]),
                "measured": stat(cm, bm, col, pred["measured"]),
            }
            for bn, bm in bins.items()
        }


def rate(pt, pick):
    return pt[1][pick].sum() / max(pt[0][pick].sum(), 1)


def contrasts(parts, pick):
    o = {}
    for src in ("real", "engine", "measured"):
        for bn in bins:
            o[f"{src}|{bn}"] = rate(parts[bn][src], pick)
        tr = np.mean([o[f"{src}|{b}"] for b in ("trail_5_7", "trail_8_14", "trail_15p")])
        ld = np.mean([o[f"{src}|{b}"] for b in ("lead_5_7", "lead_8_14", "lead_15p")])
        o[f"{src}|trail_minus_lead"] = tr - ld
        for bn in ("trail_5_7", "trail_8_14", "trail_15p", "lead_5_7", "lead_8_14", "lead_15p"):
            o[f"{src}|surge_{bn}"] = o[f"{src}|{bn}"] - o[f"{src}|neutral"]
    for k in [k for k in list(o) if k.startswith("real|")]:
        sfx = k[5:]
        o[f"gap_engine|{sfx}"] = o[k] - o[f"engine|{sfx}"]
        o[f"gap_measured|{sfx}"] = o[k] - o[f"measured|{sfx}"]
        o[f"gap_removed|{sfx}"] = o[f"gap_engine|{sfx}"] - o[f"gap_measured|{sfx}"]
    return o


KEEP = ("gap_", "real|surge", "real|trail_minus", "engine|trail_minus", "measured|trail_minus")
out = {}
for key, parts in cells.items():
    point = contrasts(parts, np.arange(len(ug)))
    bs = [contrasts(parts, rng.integers(0, len(ug), len(ug))) for _ in range(N_BOOT)]
    ent = {
        k: {"value": float(v), "ci95": ci([b[k] for b in bs])}
        for k, v in point.items()
        if k.startswith(KEEP)
    }
    ent["n"] = {bn: int(parts[bn]["real"][0].sum()) for bn in bins}
    out[f"{key[0]}|{key[1]}"] = ent
res["late"] = out
with open("artifacts/sim26/unit3_bandwidth.json", "w") as fo:
    json.dump(res, fo, indent=1)
print("done", flush=True)
