import json
import os
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

import mod25e_ldr as ld  # noqa: E402
import mod25e_ldr2 as l2  # noqa: E402

OUTD = REPO / "artifacts" / "mod25e3" / "ldr3"
FIT = OUTD / "fit.json"
PMF = OUTD / "decide_counts.npz"
DRAWS = 300
NSEAS = len(ld.POOL)

COUNT = {"calls": 0, "tries": 0, "forced": 0, "rejected": 0}


def enabled():
    return os.environ.get("LDR3") == "1"


def install_ldr3():
    import mod25d_variance as dv

    spec = json.loads(FIT.read_text(encoding="utf-8"))
    ns = dv._G["ns"]
    a = dv._G["tables"]["arrays"]
    r = np.asarray(spec["ratio"])
    rmax = r.max(axis=1)
    edges = spec["edges"]
    dedges = spec["down_edges"]
    rowcls = ld.classes(a["play_type_code"], a["yards_gained"], a["dist_raw"], a["auto_first"])
    top = ns["DECIDE"]
    tries = int(np.ceil(np.log(1e-3) / np.log(0.5)))
    rng_state = np.random.default_rng(int(os.environ.get("LDR3_SEED", "7")))

    def decide(idx, rng, tbl, qtr, clock_val, in_ot, down, distance, yardline, score_diff, off_to, def_to, phase, time_feat):
        out = top(idx, rng, tbl, qtr, clock_val, in_ot, down, distance, yardline, score_diff, off_to, def_to, phase, time_feat)
        if phase != 3 or score_diff < 1:
            return out
        c = int(l2.cell_id(np.array([score_diff]), np.array([time_feat]), np.array([down]), edges, dedges)[0])
        COUNT["calls"] += 1
        for _ in range(tries):
            COUNT["tries"] += 1
            if rng_state.random() * rmax[c] <= r[c, rowcls[out]]:
                return out
            COUNT["rejected"] += 1
            out = top(idx, rng, tbl, qtr, clock_val, in_ot, down, distance, yardline, score_diff, off_to, def_to, phase, time_feat)
        COUNT["forced"] += 1
        return out

    ns["DECIDE"] = decide


LAMBDAS3 = l2.LAMBDAS
BASELINE3 = l2.BASELINE


def build_counts(limit=0):
    import time

    import mod25d_variance as dv
    import mod25e_crG as crG
    import mod25e_crH as crH
    import mod25e_scorestate as ss
    import mod25_generator as gen
    import sim04_engine as se

    OUTD.mkdir(parents=True, exist_ok=True)
    crH.install()
    crG.register("")
    ss.ensure_css()
    dv.c25.ensure_policies()
    os.environ["BUD_OUT"] = str(OUTD / "budtmp")
    cfgj = json.dumps(dict(dv.DV["crG"], seed=7, name="ldr3fit", outdir=str(OUTD / "budtmp")))
    setting = dict(gen.SETTING_DEFAULTS)
    setting.update({"scale": 1.0, "yard_gain": 0.0, "def_sign": 1.0, "yard_bias": 0.0, "drift": 1.0, "mech": cfgj})
    crH.H_ss(setting)
    dv._G["task_key"] = (0, 0, 0)
    ns = dv._G["ns"]
    tbl = dv._G["tables"]
    arr = tbl["arrays"]
    pbp = se.load_reg_seasons(ld.POOL)
    ratings = se.load_team_ratings()
    full = se.build_transition_frame(pbp, team_ratings=ratings).reset_index(drop=True)
    assert len(full) == len(arr["play_type_code"]) and np.array_equal(full["play_type_code"].to_numpy(), np.asarray(arr["play_type_code"]))
    row_season = full["game_id"].str.slice(0, 4).astype(int).to_numpy() - ld.POOL[0]
    rowcls = ld.classes(arr["play_type_code"], arr["yards_gained"], arr["dist_raw"], arr["auto_first"])
    t = full[full["phase"] == 3].reset_index(drop=True)
    sd = t["sc_raw"].to_numpy(float)
    q = np.flatnonzero(sd >= 1)
    if limit:
        q = q[:limit]
    pick = ns["pick_index_nn_conditioned"]
    decide = ns["DECIDE"]
    rng = np.random.default_rng(int(os.environ.get("LDR3_SEED", "7")))
    counts = np.zeros((len(q), NSEAS, ld.NCLS), dtype=np.int32)
    dn = t["down_i"].to_numpy().astype(int)
    c = {k: t[k].to_numpy() for k in ("dist_raw", "fp_raw", "time_raw", "off_to_raw", "def_to_raw", "off_row", "def_row", "is_home_off")}
    k_state = se.K_STATE

    def sample(rng, tables, down, distance, yardline, score_diff, off_to, def_to, gsr, off_sim, def_sim, is_home_sim):
        phase = 3
        qtr = 4
        in_ot = False
        clock_val = gsr
        time_feat = gsr
        offense = "home" if is_home_sim else "away"
        home_to = off_to if is_home_sim else def_to
        away_to = def_to if is_home_sim else off_to
        home_score = score_diff if is_home_sim else 0
        away_score = 0 if is_home_sim else score_diff
        ot_possession_index = 0
        possessions = 10
        out = np.zeros(DRAWS, dtype=np.int64)
        for m in range(DRAWS):
            idx = pick(rng, tables, down, phase, distance, yardline, score_diff, time_feat, off_to, def_to, k_state, off_sim, def_sim, is_home_sim)
            out[m] = int(decide(idx, rng, tables, qtr, clock_val, in_ot, down, distance, yardline, score_diff, off_to, def_to, phase, time_feat))
        return out

    t0 = time.time()
    done = 0
    try:
        for n, i in enumerate(q):
            o = sample(rng, tbl, int(dn[i]), float(c["dist_raw"][i]), float(c["fp_raw"][i]), float(sd[i]), c["off_to_raw"][i], c["def_to_raw"][i], float(c["time_raw"][i]), float(c["off_row"][i]), float(c["def_row"][i]), int(c["is_home_off"][i]))
            np.add.at(counts[n], (row_season[o], rowcls[o]), 1)
            done = n + 1
            if n % 50 == 0:
                tbl["nn_weight_cache_cond"].clear()
            if n % 100 == 0:
                print(n, len(q), round(time.time() - t0, 1), flush=True)
            if n % 1000 == 999:
                np.savez_compressed(PMF.with_name("decide_counts_partial.npz"), q=q[:done], counts=counts[:done])
    finally:
        np.savez_compressed(PMF.with_name("decide_counts_partial.npz"), q=q[:done], counts=counts[:done])
    np.savez_compressed(PMF, q=q, counts=counts)
    print("saved", counts.shape, counts.sum(), flush=True)


def fit3():
    F = ld.Frame()
    z = np.load(PMF)
    q, counts = z["q"], z["counts"]
    assert len(q) == int(F.lead.sum()), (len(q), int(F.lead.sum()))
    assert np.array_equal(q, np.flatnonzero(F.lead))
    useas = F.season[q]
    cache = {}

    def pm(u, excl):
        key = (u, frozenset(excl))
        if key not in cache:
            m = useas == u
            keep = [k for k in range(NSEAS) if (ld.POOL[0] + k) not in excl]
            s = counts[m][:, keep, :].sum(axis=1).astype(float)
            tot = s.sum(axis=1, keepdims=True)
            s = np.where(tot > 0, s / np.maximum(tot, 1.0), 1.0 / ld.NCLS)
            cache[key] = (q[m], s)
        return cache[key]

    def stack(seasons, excl_fn):
        parts = [pm(u, excl_fn(u)) for u in seasons]
        return np.concatenate([p[0] for p in parts]), np.vstack([p[1] for p in parts])

    combos = [(n, dn, lam) for n in ld.TIME_SCHEMES for dn in l2.DOWN_SCHEMES for lam in LAMBDAS3]
    ids = lambda ix, n, dn: l2.cell_id(F.sd[ix], F.gsr[ix], F.down[ix], ld.TIME_SCHEMES[n], l2.DOWN_SCHEMES[dn])
    old = json.loads(ld.FIT.read_text(encoding="utf-8"))
    old_r = np.asarray(old["ratio"])
    old_ids = lambda ix: ld.cell_id(F.sd[ix], F.gsr[ix], old["edges"])
    folds = {}
    for s in ld.POOL:
        rest = [u for u in ld.POOL if u != s]
        inner = {}
        for n, dn, lam in combos:
            nc = l2.ncell(ld.TIME_SCHEMES[n], l2.DOWN_SCHEMES[dn])
            ll = 0.0
            m = 0
            for t in rest:
                r_ix, r_pm = stack([u for u in rest if u != t], lambda u: {s, t, u})
                te_ix, te_pm = pm(t, {s, t})
                r = ld.ratios(ids(r_ix, n, dn), F.cls[r_ix], r_pm, nc, lam)
                a, _ = ld.score(r, ids(te_ix, n, dn), F.cls[te_ix], te_pm)
                ll += a
                m += len(te_ix)
            inner[(n, dn, lam)] = ll / m
        tr_ix, tr_pm = stack(rest, lambda u: {s, u})
        te_ix, te_pm = pm(s, {s})
        res = {}
        for key in (max(inner, key=inner.get), BASELINE3):
            n, dn, lam = key
            nc = l2.ncell(ld.TIME_SCHEMES[n], l2.DOWN_SCHEMES[dn])
            r = ld.ratios(ids(tr_ix, n, dn), F.cls[tr_ix], tr_pm, nc, lam)
            a_out, b_out = ld.score(r, ids(te_ix, n, dn), F.cls[te_ix], te_pm)
            a_in, _ = ld.score(r, ids(tr_ix, n, dn), F.cls[tr_ix], tr_pm)
            res[key] = (a_out, b_out, a_in)
        ch = max(inner, key=inner.get)
        a_out, b_out, a_in = res[ch]
        old_out, _ = ld.score(old_r, old_ids(te_ix), F.cls[te_ix], te_pm)
        n, dn, lam = ch
        nc = l2.ncell(ld.TIME_SCHEMES[n], l2.DOWN_SCHEMES[dn])
        r = ld.ratios(ids(tr_ix, n, dn), F.cls[tr_ix], tr_pm, nc, lam)
        adj = te_pm * r[ids(te_ix, n, dn)]
        adj = adj / adj.sum(axis=1, keepdims=True)
        i = np.arange(len(te_ix))
        c = F.cls[te_ix]
        l13 = (F.sd[te_ix] >= 1) & (F.sd[te_ix] <= 3) & (F.gsr[te_ix] <= 120)
        pas = np.isin(np.arange(ld.NCLS), [2, 3])
        folds[s] = dict(choice=f"{n}|{dn}|{lam:g}", n_te=int(len(te_ix)), out_adj=a_out / len(te_ix), out_base=res[BASELINE3][0] / len(te_ix), out_decide=b_out / len(te_ix), out_oldldr=old_out / len(te_ix), in_adj=a_in / len(tr_ix), n13=int(l13.sum()), ll13_adj=float(np.log(np.maximum(adj[i, c], 1e-9))[l13].sum()), ll13_dec=float(np.log(np.maximum(te_pm[i, c], 1e-9))[l13].sum()), pass13_adj=float(adj[l13][:, pas].sum() / max(l13.sum(), 1)), pass13_dec=float(te_pm[l13][:, pas].sum() / max(l13.sum(), 1)), pass13_real=float(np.isin(c[l13], [2, 3]).mean()) if l13.any() else None)
        print(s, folds[s], flush=True)
    n_tot = sum(f["n_te"] for f in folds.values())
    n13 = sum(f["n13"] for f in folds.values())
    pooled = {k: sum(f[k] * f["n_te"] for f in folds.values()) / n_tot for k in ("out_adj", "out_base", "out_decide", "out_oldldr")}
    pooled["in_adj"] = float(np.mean([f["in_adj"] for f in folds.values()]))
    pooled["ll13_per_query_adj_minus_decide"] = sum(f["ll13_adj"] - f["ll13_dec"] for f in folds.values()) / max(n13, 1)
    pooled["n13"] = n13
    d_dec = [f["out_adj"] - f["out_decide"] for f in folds.values()]
    d_old = [f["out_adj"] - f["out_oldldr"] for f in folds.values()]
    choices = [f["choice"] for f in folds.values()]
    modal = max(set(choices), key=lambda x: (choices.count(x), x))
    n, dn, lam = modal.split("|")
    lam = float(lam)
    tr_ix, tr_pm = stack(ld.POOL, lambda u: {u})
    nc = l2.ncell(ld.TIME_SCHEMES[n], l2.DOWN_SCHEMES[dn])
    r = ld.ratios(ids(tr_ix, n, dn), F.cls[tr_ix], tr_pm, nc, lam)
    OUTD.mkdir(parents=True, exist_ok=True)
    spec = dict(scheme=n, down_scheme=dn, edges=ld.TIME_SCHEMES[n], down_edges=l2.DOWN_SCHEMES[dn], lam=lam, ratio=r.tolist(), ncls=ld.NCLS, draws=DRAWS, folds=folds, pooled=pooled, fold_deltas_vs_decide=d_dec, fold_deltas_vs_old_ldr=d_old, looks=len(combos) * len(ld.POOL))
    FIT.write_text(json.dumps(spec), encoding="utf-8")
    print("modal", modal, "pooled", pooled)
    print("delta vs decide", [round(d, 4) for d in d_dec])
    print("delta vs old ldr ratios", [round(d, 4) for d in d_old])


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["counts", "fit"])
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    if a.cmd == "counts":
        build_counts(a.limit)
    else:
        fit3()
