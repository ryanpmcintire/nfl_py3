import json
import os
import sys
from pathlib import Path

os.environ["OMP_NUM_THREADS"] = "1"
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
import sim10_drive as S  # noqa: E402

N_BOOT = 500
CHUNK = 250
EXPLOSIVE = 15.0
W_FLOOR = 1e-12
SEED = 1926
OUT19 = REPO / "artifacts" / "sim19"
OUT26 = REPO / "artifacts" / "sim26"
A = S.OUT
BASE = ["td", "fg", "score", "turnover", "elapsed", "plays", "yards", "expl", "sack", "int"]
CLASSES = ["td", "fg", "turn", "other"]
PARTS = ["n", "plays", "yards", "expl", "sack", "int"]
STATS = BASE + [f"{c}_{q}" for c in CLASSES for q in PARTS]
RATE = {
    "td": "drives",
    "fg": "drives",
    "score": "drives",
    "turnover": "drives",
    "elapsed": "drives",
    "ypp": "plays",
    "expl_share": "plays",
    "sack_rate": "plays",
    "int_rate": "plays",
    "plays_per_drive": "drives",
}
for _c in CLASSES:
    RATE[f"{_c}_share"] = "drives"
    for _m in ("ypp", "expl", "sack", "int"):
        RATE[f"{_c}_{_m}"] = "plays"
edges = json.loads((OUT19 / "trailing_risk.json").read_text())
LE = edges["deficit_edges"]
TE = edges["secs_edges"]
T9, T24 = float(TE[0]), float(TE[1])
NEUTRAL = float(LE[0])
D1, D2 = float(LE[1]), float(LE[2])

h0, hot, g0, hreg = S.load_h(str(A / "bandwidths_week.json"))
eng = S.Engine(h0, hot, g0, hreg, tilt=())

seg = pd.read_parquet(A / "segments.parquet")
c = np.load(A / "csr.npz")
ev_ptr, ev, pl_ptr, pl = c["ev_ptr"], c["ev"], c["pl_ptr"], c["pl"]
n = len(seg)
p = S.load_pbp()
games = S.real_games()
p = p[p["game_id"].isin(games[games["game_id"].isin(p["game_id"].unique())]["game_id"])]
p = p.reset_index(drop=True)
yv = p["yards_gained"].fillna(0.0).to_numpy(float)
sk = p["sack"].fillna(0).to_numpy(float)
ic = p["interception"].fillna(0).to_numpy(float)
fl = p["fumble_lost"].fillna(0).to_numpy(float)
rows = pl[:, 0].astype(int)
assert (
    p["game_id"].to_numpy()[rows] == np.repeat(seg["game_id"].to_numpy(), np.diff(pl_ptr))
).all()

W = int(np.diff(pl_ptr).max())
PT = np.full((n, W), np.inf)
PY = np.zeros((n, W))
PE = np.zeros((n, W))
PS = np.zeros((n, W))
PI = np.zeros((n, W))
PF = np.zeros((n, W))
sid = np.repeat(np.arange(n), np.diff(pl_ptr))
pos = np.arange(len(pl)) - pl_ptr[sid]
PT[sid, pos] = pl[:, 1]
PY[sid, pos] = yv[rows]
PE[sid, pos] = (yv[rows] >= EXPLOSIVE).astype(float)
PS[sid, pos] = sk[rows]
PI[sid, pos] = ic[rows]
PF[sid, pos] = fl[rows]
WE = int(np.diff(ev_ptr).max())
EO = np.full((n, WE), np.inf)
EP = np.zeros((n, WE))
esid = np.repeat(np.arange(n), np.diff(ev_ptr))
epos = np.arange(len(ev)) - ev_ptr[esid]
EO[esid, epos] = ev[:, 0]
EP[esid, epos] = ev[:, 1]
EL = seg["el"].to_numpy(float)
HE = seg["he"].to_numpy() == 1


def drive_stats(ci, secs):
    cut = HE[ci] | (EL[ci] >= secs)
    el = np.where(cut, np.minimum(EL[ci], secs), EL[ci])
    kp = PT[ci] <= secs
    ke = EO[ci] <= secs
    po = (EP[ci] * ke).sum(axis=1)
    pls = kp.sum(axis=1)
    yds = (PY[ci] * kp).sum(axis=1)
    ex = (PE[ci] * kp).sum(axis=1)
    sks = (PS[ci] * kp).sum(axis=1)
    ints = (PI[ci] * kp).sum(axis=1)
    to = (((PI[ci] + PF[ci]) * kp).sum(axis=1) > 0).astype(float)
    base = np.c_[(po >= 6), (po == 3), (po > 0), to, el, pls, yds, ex, sks, ints].astype(float)
    td_ = base[:, 0] > 0
    fg_ = (base[:, 1] > 0) & ~td_
    tn_ = (base[:, 3] > 0) & ~td_ & ~fg_
    cls = [td_, fg_, tn_, ~(td_ | fg_ | tn_)]
    parts = [np.ones(len(base)), pls, yds, ex, sks, ints]
    cond = [m.astype(float) * v for m in cls for v in parts]
    return np.c_[base, np.column_stack(cond)]


FULL = drive_stats(np.arange(n), 1e9)


def stats_at(ci, secs):
    out = FULL[ci].copy()
    cut = HE[ci] | (EL[ci] >= secs)
    if cut.any():
        out[cut] = drive_stats(ci[cut], secs)
    return out


X = S.design(S.with_mu(seg))
gid = pd.factorize(seg["game_id"])[0]
ok = (seg["bad"].to_numpy() == 0) & (seg["half"].to_numpy() == 2)
secs = seg["secs"].to_numpy(float)
sdv = seg["sd"].to_numpy(float)
home = seg["home"].to_numpy().astype(int)
season = seg["season"].to_numpy(int)
qsel = np.flatnonzero(ok & (secs <= T24) & (secs > 0))
real = np.zeros((n, len(STATS)))
real[qsel] = np.vstack([stats_at(np.array([q]), secs[q]) for q in qsel])
drawn = np.zeros((n, len(STATS)))

for hm in (0, 1):
    idx, Xs, hem = eng.strata[(2, hm, 0)]
    qh = qsel[home[qsel] == hm]
    regs = np.array([S.regime(2, secs[q]) for q in qh])
    for r in np.unique(regs):
        h = eng.h_reg[r]
        fin = np.isfinite(h)
        qr = qh[regs == r]
        for a in range(0, len(qr), CHUNK):
            qs = qr[a : a + CHUNK]
            z = np.zeros((len(qs), len(idx)))
            for d in np.flatnonzero(fin):
                z += ((Xs[:, d][None, :] - X[qs, d][:, None]) / h[d]) ** 2
            z[gid[qs][:, None] == gid[idx][None, :]] = np.inf
            z[hem[None, :] & (Xs[:, 2][None, :] < X[qs, 2][:, None])] = np.inf
            w = np.exp(-0.5 * (z - z.min(axis=1, keepdims=True)))
            for k, q in enumerate(qs):
                nz = np.flatnonzero(w[k] > W_FLOOR)
                u = w[k, nz] / w[k, nz].sum()
                drawn[q] = u @ stats_at(idx[nz], secs[q])
        print("hm", hm, "regime", int(r), len(qr), flush=True)

chk = []
rng0 = np.random.default_rng(7)
for q in qsel[:: max(1, len(qsel) // 3)][:3]:
    hm = home[q]
    qv = X[q]
    ds = np.array([eng.draw(2, hm, qv, rng0) for _ in range(400)])
    chk.append(
        {
            "q": int(q),
            "engine_draw_score": float(stats_at(ds, secs[q])[:, 2].mean()),
            "kernel_weights_score": float(drawn[q, 2]),
            "engine_draw_elapsed": float(stats_at(ds, secs[q])[:, 4].mean()),
            "kernel_weights_elapsed": float(drawn[q, 4]),
        }
    )

clk = {"final9": secs[qsel] <= T9, "min9_24": (secs[qsel] > T9) & (secs[qsel] <= T24)}
sq = sdv[qsel]
state = {
    "off_trail_5_7": (sq <= -NEUTRAL - 1) & (sq >= -D1),
    "off_trail_8_14": (sq < -D1) & (sq >= -D2),
    "off_trail_15p": sq < -D2,
    "off_trail_5p": sq <= -NEUTRAL - 1,
    "neutral": np.abs(sq) <= NEUTRAL,
    "off_lead_5p": sq >= NEUTRAL + 1,
}
ug, inv = np.unique(gid[qsel], return_inverse=True)
sea_of_game = np.zeros(len(ug), dtype=int)
sea_of_game[inv] = season[qsel]
Rq, Dq = real[qsel], drawn[qsel]


def gsum(m, M):
    return np.vstack(
        [
            np.bincount(inv, weights=np.where(m, M[:, j], 0.0), minlength=len(ug))
            for j in range(M.shape[1])
        ]
    ).T


def cnt(m):
    return np.bincount(inv, weights=m.astype(float), minlength=len(ug))


def rates(sr, sd_, nd):
    out = {}
    s = {k: i for i, k in enumerate(STATS)}
    for k in ("td", "fg", "score", "turnover", "elapsed"):
        out[k] = (sr[s[k]] / nd, sd_[s[k]] / nd)
    out["plays_per_drive"] = (sr[s["plays"]] / nd, sd_[s["plays"]] / nd)
    for nm, num in (
        ("ypp", "yards"),
        ("expl_share", "expl"),
        ("sack_rate", "sack"),
        ("int_rate", "int"),
    ):
        out[nm] = (sr[s[num]] / sr[s["plays"]], sd_[s[num]] / sd_[s["plays"]])
    for c in CLASSES:
        nc = sr[s[f"{c}_n"]]
        out[f"{c}_share"] = (nc / nd, sd_[s[f"{c}_n"]] / nd)
        pr, pd_ = sr[s[f"{c}_plays"]], sd_[s[f"{c}_plays"]]
        for nm, num in (("ypp", "yards"), ("expl", "expl"), ("sack", "sack"), ("int", "int")):
            out[f"{c}_{nm}"] = (
                sr[s[f"{c}_{num}"]] / max(pr, 1e-9),
                sd_[s[f"{c}_{num}"]] / max(pd_, 1e-9),
            )
    return out


def cell_stats(pick, gr, gd, gn):
    nd = gn[pick].sum()
    if nd == 0:
        return None
    return rates(gr[pick].sum(axis=0), gd[pick].sum(axis=0), nd)


def interval(a):
    return [float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))]


rng = np.random.default_rng(SEED)
res = {
    "engine_bandwidths": "bandwidths_week.json h_reg 2,3,4 through Engine.strata and Engine.h_reg",
    "pool": "2009-2017 REG drives, leave-game-out",
    "n_boot": N_BOOT,
    "draw_vs_weights_check": chk,
    "looks": len(clk) * len(state) * len(RATE),
    "classes": "td, fg (not td), turn (turnover, no score), other; priority in that order",
    "cells": {},
}
boots = [rng.integers(0, len(ug), len(ug)) for _ in range(N_BOOT)]
for cn, cm in clk.items():
    for sn, sm in state.items():
        m = cm & sm
        gr, gd, gn = gsum(m, Rq), gsum(m, Dq), cnt(m)
        pt = cell_stats(np.arange(len(ug)), gr, gd, gn)
        if pt is None:
            continue
        bs = []
        for b in boots:
            r_ = cell_stats(b, gr, gd, gn)
            if r_ is not None:
                bs.append(r_)
        halves = []
        for par in (0, 1):
            pk = np.flatnonzero(sea_of_game % 2 == par)
            hp = cell_stats(pk, gr, gd, gn)
            halves.append(hp)
        ent = {"n_drives": int(gn.sum()), "n_games": int((gn > 0).sum())}
        for k in RATE:
            rv, dv = pt[k]
            resid = [b[k][0] - b[k][1] for b in bs]
            ent[k] = {
                "real": float(rv),
                "drawn": float(dv),
                "residual": float(rv - dv),
                "ci95": interval(resid),
                "probability_positive": float(np.mean(np.array(resid) > 0)),
                "half_even_odd": [
                    None if hh is None else float(hh[k][0] - hh[k][1]) for hh in halves
                ],
            }
        res["cells"][f"{cn}|{sn}"] = ent

oo = pd.read_parquet(A / "oof_targets.parquet")
od = S.outcome_frame(seg)
TG = ["el", "td", "fg", "ds"]
M = np.full((n, len(TG)), np.nan)
M[oo.index.to_numpy()] = oo[TG].to_numpy(float)
Y = od[TG].to_numpy(float)
okm = ~np.isnan(M[qsel]).any(axis=1)
Rm = np.where(okm[:, None], Y[qsel] - M[qsel], 0.0)
res["model_target_residual"] = {
    "meaning": "real outcome minus out-of-fold outcome-model mean (tilt target before q4dec)",
    "cells": {},
}
for cn, cm in clk.items():
    for sn, sm in state.items():
        m = cm & sm & okm
        gr, gn = gsum(m, Rm), cnt(m)
        ent = {"n_drives": int(gn.sum())}
        for j, nm in enumerate(TG):
            pt = gr[:, j].sum() / gn.sum()
            bs = np.array([gr[b, j].sum() / gn[b].sum() for b in boots])
            ent[nm] = {
                "residual": float(pt),
                "ci95": interval(bs),
                "probability_positive": float(np.mean(bs > 0)),
            }
        res["model_target_residual"]["cells"][f"{cn}|{sn}"] = ent
OUT26.mkdir(parents=True, exist_ok=True)
for d in (OUT19, OUT26):
    (d / "late_shape.json").write_text(json.dumps(res, indent=1))
print("done", flush=True)
