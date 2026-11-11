import json
import os
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "src"))
os.environ["OMP_NUM_THREADS"] = "1"
import sim25_pool_stratum as S25  # noqa: E402

OUT = REPO / "artifacts" / "sim25"
SRC = REPO / "tests" / "scratch" / "sim_finish_pool" / "sim10_drive.py"
NBOOT = 1000
SEED = 2502
NBINS = 10
SIGN_LABELS = S25.SIGNS
PTS = np.array([7.0, 3.0, 0.0, 0.0, 0.0])


class Rec(np.ndarray):
    last = None

    def __getitem__(self, k):
        if isinstance(k, np.ndarray) and k.ndim == 1 and k.dtype.kind == "i":
            Rec.last = k
        return np.ndarray.__getitem__(self, k)


class Stop(Exception):
    pass


def load_patched():
    mod = types.ModuleType("sim10_drive")
    mod.__file__ = str(REPO / "scripts" / "sim10_drive.py")
    sys.modules["sim10_drive"] = mod
    exec(compile(SRC.read_text(), mod.__file__, "exec"), mod.__dict__)
    return mod


def build_engine(mod, floor_file):
    os.environ["SIM10_POOLSTRATUM"] = str(floor_file)
    box = {}

    def my_sim(a):
        os.environ["SIM10_LATENT"] = a.latent
        os.environ["SIM10_Q4DEC"] = a.q4dec
        os.environ["SIM10_SITUATIONAL"] = a.sit
        os.environ["SIM10_OLEXIT"] = str(int(a.olexit))
        os.environ["SIM10_FGDEC"] = a.fgdec
        os.environ["SIM10_FORM"] = str(int(a.form))
        os.environ["SIM10_HECARRY"] = str(int(a.hecarry))
        os.environ["SIM10_STFULL"] = a.stfull
        os.environ["SIM10_ST"] = a.st
        os.environ["SIM10_MUCURVE"] = str(int(a.mucurve))
        os.environ["SIM10_JOINT"] = str(int(a.joint))
        os.environ["SIM10_HEAD"] = str(int(a.head))
        h, ho, g, hr = mod.load_h(a.h)
        tilt = tuple(int(v) for v in a.tilt.split(",")) if a.tilt else ()
        mf = tuple(int(v) for v in a.mufree.split(",")) if a.mufree else ()
        st = json.loads(Path(a.strength).read_text()) if a.strength else None
        mod._init(
            h,
            ho,
            g,
            hr,
            tilt,
            bool(a.models),
            bool(a.pat),
            mf,
            bool(a.dr),
            bool(a.qstrata),
            a.eoh,
            st,
            bool(a.droof),
            bool(a.drmu),
            bool(a.mulabel),
            False,
        )
        box["eng"] = mod._W["eng"]
        box["qstrata"] = bool(a.qstrata)

    mod.cmd_sim = my_sim
    old = sys.argv
    sys.argv = ["sim10_drive.py", "sim"]
    mod.main()
    sys.argv = old
    return box["eng"], box["qstrata"]


def draw_stage(mod, eng, hf, hm, q, seed):
    cap = {}
    orig = mod.tilt_weights

    def spy(w, t, target):
        out = orig(w, t, target)
        if np.ndim(t) == 2 and t.shape[1] > 1:
            cap["w_in"], cap["w_out"], cap["t"], cap["ix"] = w, out, t, Rec.last
            raise Stop
        return out

    mod.tilt_weights = spy
    try:
        eng.draw(hf, hm, q, np.random.default_rng(seed))
    except Stop:
        pass
    finally:
        mod.tilt_weights = orig
    return cap


def stage_stats(cap, cls_pool, nc):
    ix = cap["ix"]
    t = cap["t"]
    res = []
    for w in (cap["w_in"], cap["w_out"]):
        u = w / w.sum()
        sh = np.bincount(cls_pool[ix], weights=u, minlength=nc)
        res.append([*sh, float(u @ t[:, 0]), 1.0 / float(u @ u), float(len(u))])
    return res


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    ff = OUT / "pool_floor_probe.json"
    ff.write_text(json.dumps({"floor": 0.0}))
    mod = load_patched()
    eng, qstrata = build_engine(mod, ff)
    seg = S25.sd.with_mu(pd.read_parquet(S25.sd.OUT / "segments.parquet"))
    cls = S25.outcomes(seg)
    ok = (seg["bad"].to_numpy() == 0) & ~np.isnan(cls) & (seg["half"].to_numpy() == 2)
    secs = seg["secs"].to_numpy(float)
    regs = np.array([mod.regime(2, s) if o else -1 for s, o in zip(secs, ok, strict=True)])
    X = mod.design(seg)
    qids = np.flatnonzero(ok & (regs >= 3))
    gid = pd.factorize(seg["game_id"])[0]
    home = seg["home"].to_numpy()
    qq_all = mod.quarter_of(seg["half"].to_numpy(), secs)
    sgn = S25.sign_class(seg["sd"].to_numpy(float))
    cls_pool = np.where(np.isnan(cls), 4, cls).astype(int)
    eng.out_arr = eng.out_arr.view(Rec)
    real_cls = np.zeros((len(seg), 5))
    real_cls[np.arange(len(seg)), cls_pool] = 1.0
    nrow = len(qids)
    R = np.zeros((nrow, 2, 2, 8))
    for n, i in enumerate(qids):
        key = (2, int(home[i]), int(qq_all[i]) if qstrata else 0)
        idx0, X0, hem0 = eng.strata[key]
        m = gid[idx0] != gid[i]
        eng.strata[key] = (idx0[m], X0[m], hem0[m])
        try:
            for a, fl in enumerate((None, 0.0)):
                eng.ps_floor = fl
                cap = draw_stage(mod, eng, 2, int(home[i]), X[i], 7000 + i)
                R[n, a] = stage_stats(cap, cls_pool, 5)
        finally:
            eng.strata[key] = (idx0, X0, hem0)
        if n % 2000 == 0:
            print(n, nrow, flush=True)
    np.save(OUT / "tilt_floor_raw.npy", R)
    summarise(R, qids, real_cls, seg, gid, sgn, regs, rng)


def summarise(R, qids, real_cls, seg, gid, sgn, regs, rng):
    g = gid[qids]
    ug, gi = np.unique(g, return_inverse=True)
    ng = len(ug)
    W = rng.multinomial(ng, np.full(ng, 1.0 / ng), size=NBOOT).astype(float)
    rc = real_cls[qids]
    el = seg["el"].to_numpy(float)[qids]
    rp = rc @ PTS
    out = {"n_queries": len(qids), "games": ng}

    def gsum(v):
        A = np.zeros((ng, v.shape[1]))
        np.add.at(A, gi, v)
        return A

    def ratio(A, num, den):
        tot = A.sum(axis=0)
        point = tot[num] / tot[den]
        bs = (W @ A)[:, num] / (W @ A)[:, den]
        return [
            float(point),
            float(np.quantile(bs, 0.025)),
            float(np.quantile(bs, 0.975)),
        ]

    sel_cells = [(r, s) for r in (3, 4) for s in (0, 1, 3, 4)]
    names = ["td", "fg", "punt", "turnover", "expiry", "seconds", "points"]
    stages = ("kernel", "tilted")
    arms = ("full", "restricted")
    cells = {}
    gap = {}
    for r, s in sel_cells:
        m = (regs[qids] == r) & (sgn[qids] == s)
        for a, an in enumerate(arms):
            for st, sn in enumerate(stages):
                d = np.c_[
                    rc[m] - R[m, a, st, :5],
                    el[m] - R[m, a, st, 5],
                    rp[m] - R[m, a, st, :5] @ PTS,
                    np.ones(m.sum()),
                ]
                full = np.zeros((len(qids), d.shape[1]))
                full[m] = d
                A = gsum(full)
                gap[(r, s, an, sn)] = (A, m.sum())
                cells[f"R{r}_{SIGN_LABELS[s]}_{an}_{sn}"] = {
                    "n": int(m.sum()),
                    **{nm: ratio(A, k, 7) for k, nm in enumerate(names)},
                }
    out["cells"] = cells
    nn = np.array([gap[(r, s, "full", "kernel")][1] for r, s in sel_cells], float)
    wt = nn / nn.sum()

    def abs_gap(an, sn):
        pts = np.array([gap[(r, s, an, sn)][0][:, 6] for r, s in sel_cells])
        cnt = np.array([gap[(r, s, an, sn)][0][:, 7] for r, s in sel_cells])
        tot = pts.sum(axis=1) / cnt.sum(axis=1)
        bsn = np.einsum("bg,cg->bc", W, pts) / np.einsum("bg,cg->bc", W, cnt)
        return float((np.abs(tot) * wt).sum()), (np.abs(bsn) * wt).sum(axis=1)

    head = {}
    bs = {}
    for an in arms:
        for sn in stages:
            pt, b = abs_gap(an, sn)
            bs[(an, sn)] = b
            head[f"{an}_{sn}"] = [pt, float(np.quantile(b, 0.025)), float(np.quantile(b, 0.975))]
    for sn in stages:
        dif = bs[("restricted", sn)] - bs[("full", sn)]
        head[f"restricted_minus_full_{sn}"] = [
            head[f"restricted_{sn}"][0] - head[f"full_{sn}"][0],
            float(np.quantile(dif, 0.025)),
            float(np.quantile(dif, 0.975)),
            float((dif < 0).mean()),
        ]
    dk = bs[("restricted", "kernel")] - bs[("full", "kernel")]
    dt = bs[("restricted", "tilted")] - bs[("full", "tilted")]
    with np.errstate(invalid="ignore", divide="ignore"):
        sv = dt / dk
    point_sv = head["restricted_minus_full_tilted"][0] / head["restricted_minus_full_kernel"][0]
    head["share_of_gap_difference_surviving_tilt"] = [
        float(point_sv),
        float(np.nanquantile(sv, 0.025)),
        float(np.nanquantile(sv, 0.975)),
    ]
    pa = R[:, 0, 1, :5] @ PTS
    pb = R[:, 1, 1, :5] @ PTS
    head["mean_abs_points_difference_between_pools"] = {
        "kernel": float(np.abs(R[:, 0, 0, :5] @ PTS - R[:, 1, 0, :5] @ PTS).mean()),
        "tilted": float(np.abs(pa - pb).mean()),
    }
    out["headline"] = head
    floor = {}
    for st, sn in enumerate(stages):
        eA = ((rc[:, :2] - R[:, 0, st, :2]) ** 2).sum(axis=1)
        eB = ((rc[:, :2] - R[:, 1, st, :2]) ** 2).sum(axis=1)
        ess = R[:, 1, st, 6]
        d = eB - eA
        edges = np.quantile(ess, np.linspace(0, 1, NBINS + 1))
        bins = []
        for b in range(NBINS):
            lo, hi = edges[b], edges[b + 1]
            m = (ess >= lo) & ((ess < hi) if b < NBINS - 1 else (ess <= hi))
            full = np.zeros((len(qids), 2))
            full[m, 0] = d[m]
            full[m, 1] = 1.0
            A = gsum(full)
            bsv = (W @ A)[:, 0] / np.maximum((W @ A)[:, 1], 1e-12)
            bins.append(
                {
                    "ess_lo": float(lo),
                    "ess_hi": float(hi),
                    "n": int(m.sum()),
                    "restricted_minus_full_brier": [
                        float(d[m].mean()),
                        float(np.quantile(bsv, 0.025)),
                        float(np.quantile(bsv, 0.975)),
                    ],
                    "probability_restricted_better": float((bsv < 0).mean()),
                }
            )
        means = [b["restricted_minus_full_brier"][0] for b in bins]
        fl = None
        for b in range(NBINS - 1, -1, -1):
            if means[b] <= 0:
                fl = bins[b]["ess_lo"]
            else:
                break
        cum = []
        for b in range(NBINS):
            m = ess >= edges[b]
            cum.append(
                {
                    "ess_at_least": float(edges[b]),
                    "n": int(m.sum()),
                    "mean_brier_difference_over_all_queries": float((d * m).mean()),
                }
            )
        floor[sn] = {
            "bins": bins,
            "cumulative_serving_restricted_above": cum,
            "floor_from_bins": fl,
            "queries_with_restricted_ess_below_floor": None
            if fl is None
            else int((ess < fl).sum()),
            "empty_restricted_pool_draws": int((R[:, 1, st, 7] == 0).sum()),
        }
    out["floor"] = floor
    (OUT / "tilt_floor.json").write_text(json.dumps(out, indent=1))
    print(
        json.dumps(
            {
                "headline": head,
                "floor": {
                    k: {
                        x: v[x]
                        for x in ("floor_from_bins", "queries_with_restricted_ess_below_floor")
                    }
                    for k, v in floor.items()
                },
            },
            indent=1,
        )
    )


main()
