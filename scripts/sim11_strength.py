import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "src"))
import sim10_drive as sd  # noqa: E402

OUT = Path(os.environ.get("SIM11_OUT", str(REPO / "artifacts" / "sim11")))
PHASES = ["q1", "q2", "end1h", "q3", "q4", "end2h", "ot"]
LOG = []
GI = [0]
TR = []


def pool_arrays():
    seg = sd.with_mu(pd.read_parquet(sd.OUT / "segments.parquet"))
    c = np.load(sd.OUT / "csr.npz")
    pl, ptr = c["pl"], c["pl_ptr"]
    P = pd.read_parquet(sd.OUT / "plays.parquet").set_index("row")
    ep = P["epa"].fillna(0.0).reindex(pl[:, 0].astype(np.int64)).to_numpy()
    k = np.repeat(np.arange(len(seg)), np.diff(ptr))
    es = np.bincount(k, weights=ep, minlength=len(seg))
    n = np.diff(ptr).astype(float)
    po, pd_ = sd.segment_points(seg)
    ph = np.array(
        [sd.phase_of_state(h_, s_) for h_, s_ in zip(seg["half"], seg["secs"], strict=True)]
    )
    return seg, es, n, po - pd_, ph


def wslope(x, y, w):
    xm, ym = np.average(x, weights=w), np.average(y, weights=w)
    return float(np.sum(w * (x - xm) * (y - ym)) / np.sum(w * (x - xm) ** 2))


def logo_x(D):
    og = D.groupby(["season", "game", "off"]).agg(e=("e", "sum"), n=("n", "sum")).reset_index()
    dg = D.groupby(["season", "game", "dfn"]).agg(e=("e", "sum"), n=("n", "sum")).reset_index()
    out = {}
    for nm, t, col in (("o", og, "off"), ("d", dg, "dfn")):
        t["v"] = t["e"] / t["n"]
        grp = t.groupby(["season", col])["v"]
        t["l"] = (grp.transform("sum") - t["v"]) / (grp.transform("count") - 1) - t.groupby(
            "season"
        )["v"].transform("mean")
        out[nm] = t.set_index(["game", col])["l"]
    xo = out["o"].reindex(pd.MultiIndex.from_arrays([D["game"], D["off"]])).to_numpy()
    xd = out["d"].reindex(pd.MultiIndex.from_arrays([D["game"], D["dfn"]])).to_numpy()
    return D.assign(x=xo + xd)


def phase_slopes(D):
    D = D[np.isfinite(D["x"]) & (D["n"] > 0)]
    res = {}
    for p in range(len(PHASES)):
        m = D[D["ph"] == p]
        res[PHASES[p]] = {
            "segs": len(m),
            "epa": wslope(m["x"].to_numpy(), (m["e"] / m["n"]).to_numpy(), m["n"].to_numpy()),
            "pts": float(np.cov(m["pts"], m["x"])[0, 1] / m["x"].var()),
        }
    m = D[D["ph"] < 6]
    res["all"] = {
        "segs": len(m),
        "epa": wslope(m["x"].to_numpy(), (m["e"] / m["n"]).to_numpy(), m["n"].to_numpy()),
        "pts": float(np.cov(m["pts"], m["x"])[0, 1] / m["x"].var()),
    }
    return res


def real_frame():
    seg, es, n, pts, ph = pool_arrays()
    ok = (seg["bad"].to_numpy() == 0) & (n > 0)
    rg = pd.read_parquet(sd.OUT / "real_games.parquet").set_index("game_id")
    gid = seg["game_id"].to_numpy()
    ht = rg["home_team"].reindex(gid).to_numpy()
    at = rg["away_team"].reindex(gid).to_numpy()
    hm = seg["home"].to_numpy() == 1
    D = pd.DataFrame(
        {
            "season": seg["season"].to_numpy(),
            "game": gid,
            "off": np.where(hm, ht, at),
            "dfn": np.where(hm, at, ht),
            "e": es,
            "n": n,
            "ph": ph,
            "pts": pts,
        }
    )[ok]
    return logo_x(D)


def cmd_real(a):
    OUT.mkdir(parents=True, exist_ok=True)
    D = real_frame()
    point = phase_slopes(D)
    games = D["game"].unique()
    gi = pd.Series(np.arange(len(games)), index=games)
    D = D.assign(gi=gi.reindex(D["game"]).to_numpy())
    rng = np.random.default_rng(a.seed)
    bs = {p: [] for p in PHASES}
    for _ in range(a.boot):
        wg = np.bincount(rng.integers(0, len(games), len(games)), minlength=len(games))
        for p in range(len(PHASES)):
            m = D[D["ph"] == p]
            w = m["n"].to_numpy() * wg[m["gi"].to_numpy()]
            bs[PHASES[p]].append(wslope(m["x"].to_numpy(), (m["e"] / m["n"]).to_numpy(), w))
    for p in PHASES:
        lo, hi = np.percentile(bs[p], [5, 95])
        point[p]["epa_lo"], point[p]["epa_hi"] = float(lo), float(hi)
        point[p]["epa_boot_sd"] = float(np.std(bs[p]))
        for nm, par in (("odd", 1), ("even", 0)):
            m = D[(D["ph"] == PHASES.index(p)) & (D["season"] % 2 == par)]
            point[p][f"epa_{nm}"] = wslope(
                m["x"].to_numpy(), (m["e"] / m["n"]).to_numpy(), m["n"].to_numpy()
            )
    (OUT / "real_phase.json").write_text(json.dumps(point, indent=1))
    for p, v in point.items():
        print(p, json.dumps({k: round(x, 4) if isinstance(x, float) else x for k, x in v.items()}))


def winit(*params):
    sd._init(*params)
    eng = sd._W["eng"]
    eng.trace = TR
    od, og = eng.draw, eng.game

    def draw(hf, hm, q, rng):
        n0 = len(TR)
        j = od(hf, hm, q, rng)
        if len(TR) > n0:
            r_ = TR[-1]
            k_ = (len(r_) - 2) // 3
            tr = np.r_[r_[0:4], r_[k_ : k_ + 4], r_[2 * k_ : 2 * k_ + 4], r_[3 * k_ :]]
        else:
            tr = np.full(14, np.nan)
        LOG.append((sd.phase_of_state(hf, math.expm1(q[2])), j, GI[0], hm, q[1], q[5], *tr))
        return j

    def game(*x, **kw):
        r = og(*x, **kw)
        GI[0] += 1
        return r

    eng.draw = draw
    eng.game = game


def wrun(task):
    LOG.clear()
    TR.clear()
    GI[0] = 0
    r = sd.sim_world(task)
    return r[0], np.array(LOG, dtype=np.float64).reshape(-1, 20), r[1]


def engine_params(strength, dr_oof, dr_mu, mu_free, mu_label, eoh):
    h, ho, g, hr = sd.load_h(str(sd.OUT / "bandwidths_week.json"))
    return (
        h,
        ho,
        g,
        hr,
        (5,),
        True,
        True,
        mu_free,
        True,
        True,
        eoh,
        strength,
        dr_oof,
        dr_mu,
        mu_label,
    )


def probe(n_worlds, n_seasons, seed, workers, tag, params):
    import multiprocessing as mp

    ws = np.random.SeedSequence(seed).spawn(n_worlds)
    tasks = [(w, ws[w], n_seasons) for w in range(n_worlds)]
    with mp.get_context("spawn").Pool(
        min(workers, n_worlds), initializer=winit, initargs=params
    ) as pool:
        out = pool.map(wrun, tasks, chunksize=1)
    _, es, n, pts, _ = pool_arrays()
    frames = []
    for w, L, grows in out:
        G = pd.DataFrame(grows).iloc[:, [0, 1, 3, 4]]
        G.columns = ["game_id", "season", "home_team", "away_team"]
        gi = L[:, 2].astype(np.int64)
        j = L[:, 1].astype(np.int64)
        hm = L[:, 3] == 1
        ht = G["home_team"].to_numpy()[gi]
        at = G["away_team"].to_numpy()[gi]
        frames.append(
            pd.DataFrame(
                {
                    "season": G["season"].to_numpy()[gi] + 100000 * w,
                    "game": G["game_id"].to_numpy()[gi],
                    "off": np.where(hm, ht, at),
                    "dfn": np.where(hm, at, ht),
                    "e": es[j],
                    "n": n[j],
                    "ph": L[:, 0].astype(np.int64),
                    "pts": pts[j],
                    "sd": L[:, 4],
                    "tgt": L[:, 5],
                    "j": j,
                    **{f"m_{c}": L[:, 6 + i] for i, c in enumerate(sd.OUT_TARGETS)},
                    **{f"t_{c}": L[:, 10 + i] for i, c in enumerate(sd.OUT_TARGETS)},
                    **{f"a_{c}": L[:, 14 + i] for i, c in enumerate(sd.OUT_TARGETS)},
                    "a_mu": L[:, 18],
                    "ess": L[:, 19],
                }
            )
        )
    D = pd.concat(frames, ignore_index=True)
    D = D[D["n"] > 0]
    D = logo_x(D)
    if tag:
        D.to_parquet(OUT / f"draws_{tag}.parquet")
    return phase_slopes(D)


def cmd_probe(a):
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    st = json.loads(Path(a.strength).read_text()) if a.strength else None
    mf = tuple(int(v) for v in a.mufree.split(",")) if a.mufree else ()
    params = engine_params(st, bool(a.droof), bool(a.drmu), mf, bool(a.mulabel), a.eoh)
    res = probe(a.worlds, a.seasons, a.seed, a.workers, a.tag, params)
    (OUT / f"probe_{a.tag}.json").write_text(json.dumps({"slopes": res}, indent=1))
    real = json.loads((OUT / "real_phase.json").read_text())
    for p in [*PHASES, "all"]:
        print(
            f"{p:6s} sim epa {res[p]['epa']:.3f} real {real[p]['epa']:.3f}  "
            f"sim pts {res[p]['pts']:.3f} real {real[p]['pts']:.3f}"
        )
    print("seconds", round(time.time() - t0, 1))


BIN_TARGETS = ("td", "fg", "ds")


def strength_design(mu, ph, mbar):
    Z = np.zeros((len(mu), 6))
    for p in range(6):
        sel = ph == p
        Z[sel, p] = mu[sel] - mbar[p]
    return Z


def logit_offset_fit(Z, y, off, iters=50):
    g = np.zeros(Z.shape[1])
    for _ in range(iters):
        eta = off + Z @ g
        pr = 1.0 / (1.0 + np.exp(-eta))
        W = pr * (1 - pr)
        H = (Z * W[:, None]).T @ Z + 1e-9 * np.eye(Z.shape[1])
        step = np.linalg.solve(H, Z.T @ (y - pr))
        g = g + step
        if np.abs(step).max() < 1e-10:
            break
    return g


def apply_terms(t, raw, Z, g):
    if t == "el":
        return raw + Z @ g
    lz = np.log(raw / (1 - raw))
    return 1.0 / (1.0 + np.exp(-(lz + Z @ g)))


def cmd_strfit(a):
    OUT.mkdir(parents=True, exist_ok=True)
    seg = sd.with_mu(pd.read_parquet(sd.OUT / "segments.parquet"))
    od = sd.outcome_frame(seg)
    oof = pd.read_parquet(sd.OUT / "oof_targets.parquet")
    ok = (seg["bad"].to_numpy() == 0) & (seg["half"].to_numpy() < 3)
    ph = np.array(
        [sd.phase_of_state(h_, s_) for h_, s_ in zip(seg["half"], seg["secs"], strict=True)]
    )
    R = real_frame()
    xr = np.full(len(seg), np.nan)
    xr[R.index.to_numpy()] = R["x"].to_numpy()
    mu = seg["mu"].to_numpy(float)
    home = od["home"].to_numpy()
    sel = np.flatnonzero(ok)
    mbar = {
        hm: [float(mu[sel][(ph[sel] == p) & (home[sel] == hm)].mean()) for p in range(6)]
        for hm in (0, 1)
    }
    Z = np.zeros((len(sel), 6))
    for hm in (0, 1):
        m = home[sel] == hm
        Z[m] = strength_design(mu[sel][m], ph[sel][m], mbar[hm])
    season = seg["season"].to_numpy()[sel]
    out = {"phases": PHASES[:6], "pooled_home": True, "mbar": mbar, "terms": {}, "loso": {}}
    for t in ("el", *BIN_TARGETS):
        reg = t == "el"
        y = od[t].to_numpy(float)[sel]
        raw = oof[t].reindex(sel).to_numpy()
        if not reg:
            raw = np.clip(raw, 1e-6, 1 - 1e-6)
        cor = np.zeros(len(sel))
        gs = []
        for s_ in sd.SEASONS:
            te = season == s_
            if reg:
                g = np.linalg.lstsq(Z[~te], (y - raw)[~te], rcond=None)[0]
            else:
                g = logit_offset_fit(Z[~te], y[~te], np.log(raw[~te] / (1 - raw[~te])))
            gs.append(g)
            cor[te] = apply_terms(t, raw[te], Z[te], g)
        if reg:
            g = np.linalg.lstsq(Z, y - raw, rcond=None)[0]
            lr, lc = float(np.mean((y - raw) ** 2)), float(np.mean((y - cor) ** 2))
            fit_c = apply_terms(t, raw, Z, g)
            li = float(np.mean((y - fit_c) ** 2))
        else:
            g = logit_offset_fit(Z, y, np.log(raw / (1 - raw)))
            cc = np.clip(cor, 1e-6, 1 - 1e-6)
            fi = np.clip(apply_terms(t, raw, Z, g), 1e-6, 1 - 1e-6)
            lr = float(-np.mean(y * np.log(raw) + (1 - y) * np.log(1 - raw)))
            lc = float(-np.mean(y * np.log(cc) + (1 - y) * np.log(1 - cc)))
            li = float(-np.mean(y * np.log(fi) + (1 - y) * np.log(1 - fi)))
        gs = np.array(gs)
        sl = {}
        for p in range(6):
            m = (ph[sel] == p) & np.isfinite(xr[sel])
            xx = xr[sel][m]
            sl[PHASES[p]] = {
                "actual": float(np.cov(y[m], xx)[0, 1] / xx.var()),
                "raw": float(np.cov(raw[m], xx)[0, 1] / xx.var()),
                "corrected": float(np.cov(cor[m], xx)[0, 1] / xx.var()),
            }
        out["terms"][t] = {"gamma": g.tolist(), "fold_sd": gs.std(axis=0).tolist()}
        out["loso"][t] = {
            "raw_loss": lr,
            "corrected_loss_oos": lc,
            "corrected_loss_in": li,
            "slopes": sl,
        }
        print(
            t,
            "gamma",
            np.round(g, 3).tolist(),
            "fold sd",
            np.round(gs.std(axis=0), 3).tolist(),
            f"loss raw {lr:.5f} oos {lc:.5f} in {li:.5f}",
            flush=True,
        )
        for p, v in sl.items():
            print(
                f"   {p:6s} actual {v['actual']:.3f} raw {v['raw']:.3f} "
                f"corrected {v['corrected']:.3f}"
            )
    Path(a.out).write_text(json.dumps(out, indent=1))


def main():
    ap = argparse.ArgumentParser(
        description="SIM-11: strength transmission diagnostics and terms for the SIM-10 resampler"
    )
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("real", help="real per-phase slope of drive EPA on LOGO team strength")
    r.add_argument("--boot", type=int, default=400)
    r.add_argument("--seed", type=int, default=3)
    p = sub.add_parser("probe", help="small sim that logs every drive draw")
    p.add_argument("--worlds", type=int, default=11)
    p.add_argument("--seasons", type=int, default=3)
    p.add_argument("--seed", type=int, default=21)
    p.add_argument("--workers", type=int, default=11)
    p.add_argument("--strength", default=str(OUT / "strength_terms.json"))
    p.add_argument("--droof", type=int, default=1)
    p.add_argument("--drmu", type=int, default=0)
    p.add_argument("--mufree", default="1,4")
    p.add_argument("--mulabel", type=int, default=1)
    p.add_argument("--eoh", default="outcome")
    p.add_argument("--tag", default="probe")
    sf = sub.add_parser("strfit", help="per-phase strength terms on the outcome models, LOSO")
    sf.add_argument("--out", default=str(OUT / "strength_terms.json"))
    a = ap.parse_args()
    {"real": cmd_real, "probe": cmd_probe, "strfit": cmd_strfit}[a.cmd](a)


if __name__ == "__main__":
    main()
