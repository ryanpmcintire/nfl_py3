import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

OUTD = REPO / "artifacts" / "mod25e3" / "bls"
FIT = REPO / "artifacts" / "mod25e3" / "bls_fit.json"
ISB = OUTD / "isb.npy"
POOL_SEASONS = tuple(range(2009, 2018))
SD_FINE = [-9.0, 0.0, 9.0, 13.0, 17.0, 24.0]
SD_COARSE = [-8.5, -0.5, 0.5, 8.5]
TB_EDGES = (600.0, 300.0)
PRIOR_N = (0.0, 10.0, 50.0, 250.0)
SCHEMES = ("coarse_q", "lead_q", "coarse_t", "lead_t")
NB = 200
MIN_LEAD = 9.0


def enabled():
    return os.environ.get("BLS") == "1"


def tbucket(qtr, gsr):
    qtr = np.asarray(qtr)
    gsr = np.asarray(gsr, dtype=float)
    return np.where(qtr <= 2, 0, np.where(qtr == 3, 1, np.where(gsr > TB_EDGES[0], 2, np.where(gsr > TB_EDGES[1], 3, 4))))


def cells(sd, qtr, gsr, name):
    sd = np.asarray(sd, dtype=float)
    t = tbucket(qtr, gsr)
    tq = np.where(t <= 1, t, 2)
    if name == "coarse_q":
        return np.digitize(sd, SD_COARSE) * 3 + tq
    if name == "lead_q":
        return np.digitize(sd, SD_FINE) * 3 + tq
    if name == "coarse_t":
        return np.digitize(sd, SD_COARSE) * 5 + t
    if name == "lead_t":
        return np.digitize(sd, SD_FINE) * 5 + t
    raise ValueError(name)


def cell_probs(c, y, nc, a, p0):
    k = np.bincount(c, weights=y, minlength=nc)
    n = np.bincount(c, minlength=nc).astype(float)
    return (k + a * p0) / np.maximum(n + a, 1e-12), n


def load_pool():
    import sim04_engine as sim

    pbp = sim.load_reg_seasons(POOL_SEASONS)
    trans = sim.build_transition_frame(pbp)
    right = pbp[["game_id", "play_id", "passer_player_id", "posteam", "epa", "season"]].drop_duplicates(["game_id", "play_id"])
    m = trans[["game_id", "play_id"]].merge(right, on=["game_id", "play_id"], how="left")
    code = trans["play_type_code"].to_numpy()
    haspass = (code == 1) & m.passer_player_id.notna().to_numpy()
    sub = m[haspass]
    cnt = sub.groupby(["season", "posteam", "passer_player_id"]).size().reset_index(name="n").sort_values("n", ascending=False).drop_duplicates(["season", "posteam"])
    starter = {(s, t): p for s, t, p in zip(cnt.season, cnt.posteam, cnt.passer_player_id)}
    st = np.array([starter.get((s, t)) for s, t in zip(m.season, m.posteam)], dtype=object)
    isb = np.full(len(m), -1, dtype=np.int8)
    isb[haspass] = (m.passer_player_id.to_numpy()[haspass] != st[haspass]).astype(np.int8)
    return dict(trans=trans, isb=isb, epa=m.epa.to_numpy(dtype=float), season=m.season.to_numpy().astype(int), gid=pd.factorize(m.game_id)[0])


def fit():
    P = load_pool()
    tr = P["trans"]
    isb = P["isb"]
    qtr = tr["qtr_actual"].to_numpy()
    gsr_in_half = tr["time_raw"].to_numpy(dtype=float)
    sd = tr["sc_raw"].to_numpy(dtype=float)
    ok = (isb >= 0) & (qtr <= 4) & np.isfinite(sd) & np.isfinite(gsr_in_half)
    y = isb.astype(float)
    seasons = np.unique(P["season"][ok])
    p0g = y[ok].mean()
    res = {}
    looks = 0
    cell_all = {s: cells(np.where(ok, sd, 0.0), qtr, gsr_in_half, s) for s in SCHEMES}
    nc = {s: int(cell_all[s].max()) + 1 for s in SCHEMES}
    for sch in SCHEMES:
        for a in PRIOR_N:
            ll = 0.0
            for s in seasons:
                te = ok & (P["season"] == s)
                trn = ok & (P["season"] != s)
                p0 = y[trn].mean()
                p, _ = cell_probs(cell_all[sch][trn], y[trn], nc[sch], a, p0)
                pt = np.clip(p[cell_all[sch][te]], 1e-6, 1 - 1e-6)
                ll += float(np.sum(y[te] * np.log(pt) + (1 - y[te]) * np.log(1 - pt)))
            res[f"{sch}|{a:g}"] = ll / int(ok.sum())
            looks += 1
    best = max(res, key=res.get)
    sch, a = best.split("|")
    a = float(a)
    p_full, n_full = cell_probs(cell_all[sch][ok], y[ok], nc[sch], a, p0g)
    pin = np.clip(p_full[cell_all[sch][ok]], 1e-6, 1 - 1e-6)
    ll_in = float(np.mean(y[ok] * np.log(pin) + (1 - y[ok]) * np.log(1 - pin)))
    base_ll = res["coarse_q|0"]
    epa = P["epa"]
    lead17 = ok & (sd >= 17)
    g = P["gid"]
    rng = np.random.default_rng(132)
    ng = int(g.max()) + 1
    gb = np.bincount(g[lead17 & (isb == 1)], weights=np.nan_to_num(epa[lead17 & (isb == 1)]), minlength=ng), np.bincount(g[lead17 & (isb == 1)], minlength=ng).astype(float)
    gs = np.bincount(g[lead17 & (isb == 0)], weights=np.nan_to_num(epa[lead17 & (isb == 0)]), minlength=ng), np.bincount(g[lead17 & (isb == 0)], minlength=ng).astype(float)
    diffs = []
    for _ in range(NB):
        w = np.bincount(rng.integers(0, ng, ng), minlength=ng).astype(float)
        diffs.append(w @ gb[0] / max(w @ gb[1], 1) - w @ gs[0] / max(w @ gs[1], 1))
    d_pt = gb[0].sum() / gb[1].sum() - gs[0].sum() / gs[1].sum()
    OUTD.mkdir(parents=True, exist_ok=True)
    np.save(ISB, isb)
    spec = dict(
        scheme=sch,
        prior_n=a,
        probs=p_full.tolist(),
        counts=n_full.tolist(),
        sd_fine=SD_FINE,
        sd_coarse=SD_COARSE,
        tb_edges=list(TB_EDGES),
        seasons=[int(x) for x in seasons],
        folds=len(seasons),
        n_pass_rows=int(ok.sum()),
        rows=int(len(isb)),
        base_rate=float(p0g),
        loso_ll_per_row=res,
        best_minus_coarse_q=res[best] - base_ll,
        in_sample_ll_per_row=ll_in,
        in_out_gap=ll_in - res[best],
        looks=looks,
        backup_minus_starter_epa_lead17=dict(point=float(d_pt), lo=float(np.percentile(diffs, 5)), hi=float(np.percentile(diffs, 95)), p_positive=float(np.mean(np.array(diffs) > 0)), n_backup=int(gb[1].sum()), n_starter=int(gs[1].sum())),
        pool_len=int(len(isb)),
        score_clip_note="engine neighbour key clips score at +-21; weights use the raw state score",
    )
    FIT.write_text(json.dumps(spec), encoding="utf-8")
    print("best", best, "gain over coarse_q|0", spec["best_minus_coarse_q"], "in-sample gap", spec["in_out_gap"], "looks", looks)
    print({k: round(v, 6) for k, v in res.items()})
    print("backup-starter EPA lead>=17", spec["backup_minus_starter_epa_lead17"])
    return spec


def install_bls():
    import mod25d_variance as dv

    spec = json.loads(FIT.read_text(encoding="utf-8"))
    isb = np.load(ISB)
    t = dv._G["tables"]
    ns = dv._G["ns"]
    a = t["arrays"]
    assert len(isb) == len(a["play_type_code"]) == spec["pool_len"]
    sch = spec["scheme"]
    probs = np.asarray(spec["probs"], dtype=float)
    base = ns["pick_index_nn_conditioned"]
    stats = {"draws": 0, "reweighted": 0}
    dv._G["bls_stats"] = stats

    def pick(rng, tables, down, phase, dist, fp, score, time_raw, off_to, def_to, k_state, off_sim, def_sim, is_home_sim):
        idx = base(rng, tables, down, phase, dist, fp, score, time_raw, off_to, def_to, k_state, off_sim, def_sim, is_home_sim)
        if score < MIN_LEAD:
            return idx
        fr = sys._getframe(1)
        while fr is not None and "offense" not in fr.f_locals:
            fr = fr.f_back
        if fr is None:
            return idx
        q_now = int(fr.f_locals["qtr"])
        if q_now not in (3, 4):
            return idx
        stats["draws"] += 1
        key = ns["round_state_key"](down if down in (1, 2, 3, 4) else 4, phase, dist, fp, score, time_raw, off_to, def_to)
        nb = tables["nn_cache_cond"].get(key)
        cw = tables["nn_weight_cache_cond"].get((key, off_sim, def_sim, is_home_sim))
        if nb is None or cw is None:
            return idx
        w = np.diff(np.r_[0.0, cdf_of(cw)])
        b = isb[nb]
        mb = b == 1
        ms = b == 0
        sw = float(w[mb | ms].sum())
        if sw <= 0.0:
            return idx
        cur = float(w[mb].sum()) / sw
        c = int(cells(np.array([score]), np.array([q_now]), np.array([time_raw]), sch)[0])
        p = float(probs[c])
        if not (0.0 < cur < 1.0 and 0.0 < p < 1.0):
            return idx
        w2 = np.where(mb, w * p / cur, np.where(ms, w * (1.0 - p) / (1.0 - cur), w))
        c2 = np.cumsum(w2)
        stats["reweighted"] += 1
        return int(nb[min(int(np.searchsorted(c2, rng.random() * c2[-1], side="right")), len(nb) - 1)])

    def cdf_of(cw):
        return cw[0]

    ns["pick_index_nn_conditioned"] = pick


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "fit"
    if cmd == "fit":
        fit()
