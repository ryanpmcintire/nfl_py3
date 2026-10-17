import json
import os
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

OUTD = REPO / "artifacts" / "mod25e3"
FIT = OUTD / "lpo_fit.json"
POOLF = OUTD / "lpo" / "pool.npz"
MIN_LEAD = 9.0
POOL_SEASONS = tuple(range(2009, 2018))
EDGE_SETS = {"e3": [9.0, 17.0, 24.0], "e4": [9.0, 13.0, 17.0, 24.0], "e4h": [9.0, 17.0, 24.0, 35.0], "e5": [9.0, 13.0, 17.0, 24.0, 35.0]}
TB_SCHEMES = ("none", "half", "four")
K_GRID = (25, 50, 100, 200)
YARD_EDGES = (0.0, 1.0, 5.0, 10.0, 20.0)
NCLS = 7
TB_EDGES = (600.0, 300.0)


def enabled():
    return os.environ.get("LPO") == "1"


def tb_of(qtr, gsr, scheme):
    qtr = np.asarray(qtr)
    gsr = np.asarray(gsr, dtype=float)
    if scheme == "none":
        return np.zeros(len(qtr), dtype=int)
    if scheme == "half":
        return (qtr - 3).astype(int)
    return np.where(qtr == 3, 0, np.where(gsr > TB_EDGES[0], 1, np.where(gsr > TB_EDGES[1], 2, 3))).astype(int)


def ntb(scheme):
    return {"none": 1, "half": 2, "four": 4}[scheme]


def keys_for(sd, qtr, gsr, down, edges, scheme):
    lb = np.digitize(np.asarray(sd, dtype=float), edges[1:])
    tb = tb_of(qtr, gsr, scheme)
    nt = ntb(scheme)
    nl = len(edges)
    d = np.clip(np.asarray(down).astype(int), 1, 4) - 1
    c0 = lb * nt + tb
    return [c0 * 4 + d, lb * 4 + d, d], [nl * nt, nl, 1]


def classes(yards, flip):
    c = np.digitize(yards, YARD_EDGES) + 1
    return np.where(flip, 0, c).astype(int)


def pool_rows():
    import mod25e_bls as bl

    P = bl.load_pool()
    tr = P["trans"]
    code = tr["play_type_code"].to_numpy()
    qtr = tr["qtr_actual"].to_numpy()
    sd = tr["sc_raw"].to_numpy(dtype=float)
    gsr = tr["time_raw"].to_numpy(dtype=float)
    ok = (code == 1) & (qtr >= 3) & (qtr <= 4) & np.isfinite(sd) & np.isfinite(gsr) & (sd >= MIN_LEAD)
    rows = np.flatnonzero(ok)
    return dict(
        rows=rows,
        sd=sd[rows],
        qtr=qtr[rows].astype(int),
        gsr=gsr[rows],
        down=tr["down_i"].to_numpy()[rows].astype(int),
        dist=tr["dist_raw"].to_numpy(dtype=float)[rows],
        fp=tr["fp_raw"].to_numpy(dtype=float)[rows],
        cls=classes(tr["yards_gained"].to_numpy(dtype=float)[rows], tr["possession_flip"].to_numpy()[rows]),
        season=P["season"][rows],
        pool_len=int(len(tr)),
    )


def predict(Z, cls, keys, sizes, tr, te, K):
    from scipy.spatial import cKDTree

    g = np.bincount(cls[tr], minlength=NCLS) / max(int(tr.sum()), 1)
    out = np.tile(g, (int(te.sum()), 1))
    tri = np.flatnonzero(tr)
    tei = np.flatnonzero(te)
    assigned = np.full(len(tei), -1)
    for L, (k, ns) in enumerate(zip(keys, sizes, strict=True)):
        nkey = int(k.max()) + 1
        cnt = np.bincount(k[tri], minlength=nkey)
        free = assigned < 0
        use = free & (cnt[k[tei]] >= K)
        assigned[use] = L
    for L, (k, ns) in enumerate(zip(keys, sizes, strict=True)):
        sel = np.flatnonzero(assigned == L)
        if len(sel) == 0:
            continue
        kt = k[tei[sel]]
        order = np.argsort(k[tri], kind="stable")
        sk = k[tri][order]
        for kv in np.unique(kt):
            lo, hi = np.searchsorted(sk, kv, "left"), np.searchsorted(sk, kv, "right")
            members = tri[order[lo:hi]]
            rows_sel = sel[kt == kv]
            tree = cKDTree(Z[members])
            _, ind = tree.query(Z[tei[rows_sel]], k=K)
            ind = np.atleast_2d(ind).reshape(len(rows_sel), K)
            cc = cls[members][ind]
            cnts = np.stack([(cc == c).sum(1) for c in range(NCLS)], axis=1)
            out[rows_sel] = (cnts + g) / (K + 1.0)
    return out, assigned


def loglik(pred, cls_te):
    return float(np.sum(np.log(np.clip(pred[np.arange(len(cls_te)), cls_te], 1e-12, None))))


def fit():
    P = pool_rows()
    n = len(P["rows"])
    sdz = (float(P["dist"].std()), float(P["fp"].std()))
    Z = np.stack([P["dist"] / sdz[0], P["fp"] / sdz[1]], axis=1)
    seasons = np.unique(P["season"])
    res = {}
    for en, edges in EDGE_SETS.items():
        for tbs in TB_SCHEMES:
            keys, sizes = keys_for(P["sd"], P["qtr"], P["gsr"], P["down"], edges, tbs)
            for K in K_GRID:
                ll = 0.0
                for s in seasons:
                    te = P["season"] == s
                    pred, _ = predict(Z, P["cls"], keys, sizes, ~te, te, K)
                    ll += loglik(pred, P["cls"][te])
                res[f"{en}|{tbs}|{K}"] = ll / n
    base = 0.0
    for s in seasons:
        te = P["season"] == s
        g = np.bincount(P["cls"][~te], minlength=NCLS) / int((~te).sum())
        base += loglik(np.tile(g, (int(te.sum()), 1)), P["cls"][te])
    base /= n
    best = max(res, key=res.get)
    en, tbs, K = best.split("|")
    K = int(K)
    edges = EDGE_SETS[en]
    keys, sizes = keys_for(P["sd"], P["qtr"], P["gsr"], P["down"], edges, tbs)
    allm = np.ones(n, dtype=bool)
    pred_in, assigned = predict(Z, P["cls"], keys, sizes, allm, allm, K)
    ll_in = loglik(pred_in, P["cls"]) / n
    tables = {}
    for L, k in enumerate(keys):
        for kv in np.unique(k):
            m = k == kv
            tables[f"L{L}|{int(kv)}"] = dict(n=int(m.sum()), cls_counts=np.bincount(P["cls"][m], minlength=NCLS).tolist(), eligible=bool(m.sum() >= K))
    spec = dict(
        provenance=dict(
            script="scripts/mod25e_lpo.py fit",
            source="2009-2017 regular season pbp, pass rows (play_type_code 1), qtr 3-4, offense lead >= 9, via sim04_engine.build_transition_frame",
            seasons=[int(s) for s in seasons],
            folds=int(len(seasons)),
            looks=len(res),
            family="edge set x time scheme x K, one family",
            candidates=dict(edge_sets=EDGE_SETS, time_schemes=list(TB_SCHEMES), K=list(K_GRID)),
            outcome_classes="0 possession flip; 1 yards<0; 2 yards 0; 3 yards 1-4; 4 yards 5-9; 5 yards 10-19; 6 yards 20+",
            prior="one pseudo-row of the training class distribution",
            fallback="cell (lead bin x time bucket x down) with fewer than K pool rows falls to (lead bin x down), then (down); if still below K the engine draw is kept",
            matching="K nearest in (distance, field position), each divided by its pool standard deviation; down matched exactly",
        ),
        chosen=dict(edge_set=en, edges=edges, time_scheme=tbs, K=K),
        loso_ll_per_row=res,
        baseline_global_class_ll_per_row=base,
        best_minus_baseline=res[best] - base,
        in_sample_ll_per_row=ll_in,
        in_out_gap=ll_in - res[best],
        scale=dict(dist_sd=sdz[0], fp_sd=sdz[1]),
        n_rows=n,
        pool_len=P["pool_len"],
        level_use_in_sample=[int((assigned == L).sum()) for L in (0, 1, 2)] + [int((assigned < 0).sum())],
        tables=tables,
    )
    (OUTD / "lpo").mkdir(parents=True, exist_ok=True)
    np.savez(POOLF, rows=P["rows"], sd=P["sd"], qtr=P["qtr"], gsr=P["gsr"], down=P["down"], dist=P["dist"], fp=P["fp"], cls=P["cls"], season=P["season"])
    FIT.write_text(json.dumps(spec), encoding="utf-8")
    print("best", best, "LL", res[best], "baseline", base, "gain", spec["best_minus_baseline"], "in-sample", ll_in, "gap", spec["in_out_gap"], "rows", n, "looks", len(res))
    print({k: round(v, 5) for k, v in sorted(res.items(), key=lambda kv: -kv[1])[:12]})
    print("level use", spec["level_use_in_sample"])
    return spec


def install_lpo():
    import mod25e_bls as bl
    import mod25d_variance as dv
    from scipy.spatial import cKDTree

    spec = json.loads(FIT.read_text(encoding="utf-8"))
    pz = np.load(POOLF)
    ns = dv._G["ns"]
    arrays = dv._G["tables"]["arrays"]
    assert spec["pool_len"] == len(arrays["play_type_code"])
    ch = spec["chosen"]
    edges = ch["edges"]
    tbs = ch["time_scheme"]
    K = int(ch["K"])
    sdz = (spec["scale"]["dist_sd"], spec["scale"]["fp_sd"])
    nt = ntb(tbs)
    keys, _ = keys_for(pz["sd"], pz["qtr"], pz["gsr"], pz["down"], edges, tbs)
    rows = pz["rows"]
    Zp = np.stack([pz["dist"] / sdz[0], pz["fp"] / sdz[1]], axis=1)
    groups = {}
    for L, k in enumerate(keys):
        for kv in np.unique(k):
            m = np.flatnonzero(k == kv)
            if len(m) >= K:
                groups[(L, int(kv))] = (cKDTree(Zp[m]), rows[m])
    base = ns["pick_index_nn_conditioned"]
    code = arrays["play_type_code"]
    stats = {"calls": 0, "pass_draws": 0, "lead_pass": 0, "no_frame": 0, "redrawn": 0, "changed": 0, "no_pool": 0, "level": {"0": 0, "1": 0, "2": 0}, "by_bin": {}, "K": K, "edges": edges, "time_scheme": tbs}
    dv._G["lpo_stats"] = stats
    stat_path = OUTD / f"lpo_stats_{os.getpid()}.json"
    qcache = {}

    def pick(rng, tables, down, phase, dist, fp, score, time_raw, off_to, def_to, k_state, off_sim, def_sim, is_home_sim):
        idx = base(rng, tables, down, phase, dist, fp, score, time_raw, off_to, def_to, k_state, off_sim, def_sim, is_home_sim)
        stats["calls"] += 1
        if stats["calls"] % 2000 == 0:
            stat_path.write_text(json.dumps(stats), encoding="utf-8")
        if code[idx] != 1 or score < MIN_LEAD:
            return idx
        stats["pass_draws"] += 1
        fr = sys._getframe(1)
        while fr is not None and "offense" not in fr.f_locals:
            fr = fr.f_back
        if fr is None:
            stats["no_frame"] += 1
            return idx
        q_now = int(fr.f_locals["qtr"])
        if q_now not in (3, 4):
            return idx
        stats["lead_pass"] += 1
        ks, _ = keys_for(np.array([score]), np.array([q_now]), np.array([time_raw]), np.array([down]), edges, tbs)
        lbin = str(int(np.digitize(score, edges[1:])))
        for L in (0, 1, 2):
            g = groups.get((L, int(ks[L][0])))
            if g is not None:
                break
        else:
            stats["no_pool"] += 1
            return idx
        qk = (L, int(ks[L][0]), float(dist), float(fp))
        sel = qcache.get(qk)
        if sel is None:
            _, ind = g[0].query(np.array([dist / sdz[0], fp / sdz[1]]), k=K)
            sel = g[1][np.atleast_1d(ind)]
            qcache[qk] = sel
        new = int(sel[min(int(rng.random() * K), K - 1)])
        stats["redrawn"] += 1
        stats["level"][str(L)] += 1
        stats["by_bin"][lbin] = stats["by_bin"].get(lbin, 0) + 1
        if new != idx:
            stats["changed"] += 1
        return new

    ns["pick_index_nn_conditioned"] = pick


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "fit"
    if cmd == "fit":
        fit()
