import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "src"))

import sim04_engine as sim  # noqa: E402

SNAP = REPO / "data" / "pbp" / "raw" / "20260929T191306Z"
OUT = REPO / "artifacts" / "mod25_mechanisms"
sim.PBP_SNAPSHOT_DIR = SNAP
TRAIN = tuple(range(2009, 2018))
EVAL = tuple(range(2018, 2026))
COLS = ["g", "sd", "gsr", "qtr", "down", "dist", "yl", "code", "po", "pdf", "flip", "el"]
_G = {}


def real_frames(seasons):
    pbp = sim.load_reg_seasons(tuple(seasons))
    tr = sim.build_transition_frame(pbp).sort_values(["game_id", "play_id"]).reset_index(drop=True)
    gid = pd.factorize(tr["game_id"])[0]
    P = pd.DataFrame(
        {
            "g": gid,
            "sd": tr["sc_raw"].to_numpy(),
            "gsr": tr["gsr_actual"].to_numpy(),
            "qtr": tr["qtr_actual"].to_numpy(),
            "down": tr["down_i"].to_numpy(),
            "dist": tr["dist_raw"].to_numpy(),
            "yl": tr["fp_raw"].to_numpy(),
            "code": tr["play_type_code"].to_numpy(),
            "po": tr["points_off"].to_numpy(),
            "pdf": tr["points_def"].to_numpy(),
            "flip": tr["possession_flip"].to_numpy().astype(bool),
            "el": tr["clock_elapsed"].to_numpy(),
        }
    )
    last = tr.groupby(gid).tail(1)
    M = pd.Series(last["home_margin_post"].to_numpy(), index=gid[last.index.to_numpy()])
    return P, M


def sim_init(train, seed_tag):
    sim.PBP_SNAPSHOT_DIR = SNAP
    _G["tables"] = sim.build_tables(tuple(train), condition_on_team=False)
    _G["log"] = []

    def pol(down, distance, yardline, score_diff, qtr, clock_val, drawn):
        _G["log"].append(
            (
                down, distance, yardline, score_diff, clock_val, qtr,
                int(drawn["play_type_code"]), float(drawn["points_off"]), float(drawn["points_def"]),
                bool(drawn["flip"]), float(drawn["clock_elapsed"]),
            )
        )
        return drawn

    _G["pol"] = pol


def sim_batch(task):
    seed, n = task
    rng = np.random.default_rng(seed)
    t = _G["tables"]
    rows = []
    margins = []
    for gi in range(n):
        _G["log"].clear()
        state = sim.initial_kickoff_state(rng, t["opening_pool"])
        rec, _ = sim.run_one_game(state, t, rng, 600.0, _G["pol"], sim.K_NEIGHBORS, sim.MAX_PLAYS_PER_GAME)
        a = np.array(_G["log"], dtype=np.float64).reshape(-1, 11)
        rows.append(np.column_stack([np.full(len(a), gi), a]))
        margins.append(rec["margin"])
    return seed, np.concatenate(rows), np.array(margins)


def run_sim(train, n_games, workers, seed, init=sim_init):
    import multiprocessing as mp

    per = max(50, n_games // (workers * 4))
    tasks = [(seed * 100000 + i, per) for i in range((n_games + per - 1) // per)]
    ctx = mp.get_context("spawn")
    with ctx.Pool(workers, initializer=init, initargs=(train, "x")) as pool:
        res = pool.map(sim_batch, tasks, chunksize=1)
    frames = []
    ms = []
    off = 0
    for _, a, m in res:
        g = a[:, 0].astype(np.int64) + off
        d = pd.DataFrame(
            {
                "g": g, "down": a[:, 1], "dist": a[:, 2], "yl": a[:, 3], "sd": a[:, 4], "gsr": a[:, 5],
                "qtr": a[:, 6], "code": a[:, 7].astype(int), "po": a[:, 8], "pdf": a[:, 9],
                "flip": a[:, 10].astype(bool), "el": a[:, 11],
            }
        )
        frames.append(d)
        ms.append(pd.Series(m, index=np.arange(len(m)) + off))
        off += len(m)
    return pd.concat(frames, ignore_index=True), pd.concat(ms)


SD_BINS = [(-99, -9), (-8, -4), (-3, -1), (0, 0), (1, 3), (4, 8), (9, 99)]


def sd_label(lo, hi):
    return f"{lo}..{hi}"


def analyse(P, M):
    out = {}
    n = len(M)
    m = M.to_numpy()
    am = np.abs(m)
    for k in (3, 7, 10, 14, 17):
        out[f"m{k}"] = float(np.mean(am == k))
    out["margin_sd"] = float(np.std(m))
    out["le3"] = float(np.mean(am <= 3))
    out["tie"] = float(np.mean(am == 0))
    out["pts_game"] = float(((P["po"] + P["pdf"]).sum()) / n)
    P = P.sort_values(["g"], kind="stable").reset_index(drop=True)
    g = P["g"].to_numpy()
    prev_flip = np.r_[False, P["flip"].to_numpy()[:-1]]
    newg = np.r_[True, g[1:] != g[:-1]]
    start = prev_flip | newg
    P["start"] = start
    P["drive"] = np.cumsum(start)
    code = P["code"].to_numpy()
    po = P["po"].to_numpy()
    pdf = P["pdf"].to_numpy()
    out["drives_game"] = float(start.sum() / n)
    out["td_game"] = float((((po >= 6) | (pdf >= 6)).sum()) / n)
    out["fg_made_game"] = float((((code == 3) & (po == 3)).sum()) / n)
    out["fg_att_game"] = float(((code == 3).sum()) / n)
    out["punt_game"] = float(((code == 2).sum()) / n)
    out["safety_game"] = float((((pdf == 2) & (po == 0)).sum()) / n)
    out["plays_game"] = float(len(P) / n)
    tdo = (po >= 6) & ((code == 0) | (code == 1))
    for v in (6, 7, 8):
        out[f"pat_pts{v}"] = float(np.mean(po[tdo] == v)) if tdo.any() else np.nan
    sd = P["sd"].to_numpy()
    for lo, hi in ((-8, -8), (-5, -5), (-2, -2), (-1, -1), (0, 0), (1, 1), (2, 2), (5, 5), (8, 8), (10, 14)):
        mk = tdo & (sd >= lo) & (sd <= hi)
        out[f"pat8|pre{lo}..{hi}"] = float(np.mean(po[mk] == 8)) if mk.sum() >= 30 else np.nan
        out[f"pat6|pre{lo}..{hi}"] = float(np.mean(po[mk] == 6)) if mk.sum() >= 30 else np.nan
    sc = (po > 0) | (pdf > 0)
    ls = P.loc[sc].groupby("g").tail(1)
    lpo = ls["po"].to_numpy()
    lpd = ls["pdf"].to_numpy()
    lcode = ls["code"].to_numpy()
    typ = np.where(
        (lpo == 0) & (lpd == 2), "safety",
        np.where((lpo > 0) & (lcode == 3), "FG", "TD"),
    )
    lg = ls["g"].to_numpy()
    ams = np.abs(M.reindex(lg).to_numpy())
    out["last_none"] = float(1 - len(ls) / n)
    for t in ("FG", "TD", "safety"):
        mk = typ == t
        out[f"last_{t}_share"] = float(mk.sum() / n)
        out[f"m3|last_{t}"] = float(np.mean(ams[mk] == 3)) if mk.sum() else np.nan
        out[f"m7|last_{t}"] = float(np.mean(ams[mk] == 7)) if mk.sum() else np.nan
        out[f"le3|last_{t}"] = float(np.mean(ams[mk] <= 3)) if mk.sum() else np.nan
    pre = np.where(lpo > 0, ls["sd"].to_numpy(), -ls["sd"].to_numpy())
    lgs = ls["gsr"].to_numpy()
    for t in ("FG", "TD"):
        mk = typ == t
        for lo, hi in ((-99, -4), (-3, -1), (0, 0), (1, 3), (4, 99)):
            mm = mk & (pre >= lo) & (pre <= hi)
            out[f"last{t}_pre|{lo}..{hi}"] = float(mm.sum() / n)
        out[f"last{t}_inlast5min"] = float(np.mean(lgs[mk] <= 300)) if mk.sum() else np.nan
    reg = (P["qtr"].to_numpy() <= 4)
    for T in (1800, 900, 300, 120):
        mk = reg & (P["gsr"].to_numpy() <= T)
        first = P.loc[mk].groupby("g").head(1)
        ad = np.abs(first["sd"].to_numpy())
        fm = np.abs(M.reindex(first["g"].to_numpy()).to_numpy())
        for lo, hi in ((0, 0), (1, 3), (4, 7), (8, 14), (15, 99)):
            mm = (ad >= lo) & (ad <= hi)
            out[f"at{T}_share|{lo}..{hi}"] = float(mm.sum() / len(first))
            out[f"at{T}_f3|{lo}..{hi}"] = float(np.mean(fm[mm] == 3)) if mm.sum() >= 30 else np.nan
            out[f"at{T}_f7|{lo}..{hi}"] = float(np.mean(fm[mm] == 7)) if mm.sum() >= 30 else np.nan
            out[f"at{T}_fle3|{lo}..{hi}"] = float(np.mean(fm[mm] <= 3)) if mm.sum() >= 30 else np.nan
    ds = P.loc[P["start"]].groupby("g").tail(1)
    sd0 = ds["sd"].to_numpy()
    gsr0 = ds["gsr"].to_numpy()
    dr = ds["drive"].to_numpy()
    scored_last = P.loc[sc].groupby("drive").size().reindex(dr).fillna(0).to_numpy() > 0
    fgl = P.loc[(code == 3) & (po == 3)].groupby("drive").size().reindex(dr).fillna(0).to_numpy() > 0
    tdl = P.loc[tdo].groupby("drive").size().reindex(dr).fillna(0).to_numpy() > 0
    lmn = len(ds)
    for lo, hi in SD_BINS:
        mk = (sd0 >= lo) & (sd0 <= hi)
        k = sd_label(lo, hi)
        out[f"lastposs_share|{k}"] = float(mk.sum() / lmn)
        out[f"lastposs_score|{k}"] = float(np.mean(scored_last[mk])) if mk.sum() >= 30 else np.nan
        out[f"lastposs_fg|{k}"] = float(np.mean(fgl[mk])) if mk.sum() >= 30 else np.nan
        out[f"lastposs_td|{k}"] = float(np.mean(tdl[mk])) if mk.sum() >= 30 else np.nan
        out[f"lastposs_gsr|{k}"] = float(np.median(gsr0[mk])) if mk.sum() >= 30 else np.nan
    t0 = np.where(gsr0 <= 300, "le300", np.where(gsr0 <= 900, "le900", "early"))
    for t in ("le300", "le900", "early"):
        out[f"lastposs_time|{t}"] = float(np.mean(t0 == t))
    down4 = (P["down"].to_numpy() == 4) & np.isin(code, (0, 1, 2, 3))
    q = P.loc[down4]
    qc = q["code"].to_numpy()
    lab = np.where(qc == 3, "FG", np.where(qc == 2, "punt", "go"))
    qsd = q["sd"].to_numpy()
    qg = q["gsr"].to_numpy()
    qy = q["yl"].to_numpy()
    qd = q["dist"].to_numpy()
    qq = q["qtr"].to_numpy()
    tz = np.where(qq >= 5, "ot", np.where(qg <= 300, "le300", np.where(qg <= 900, "q4early", np.where(qg <= 1800, "mid", "h1"))))
    yz = np.where(qy <= 20, "red", np.where(qy <= 36, "fgr", np.where(qy <= 55, "mid", "own")))
    out["fourth_per_game"] = float(len(q) / n)
    for l in ("go", "FG", "punt"):
        out[f"4th_{l}_rate"] = float(np.mean(lab == l))
    for lo, hi in SD_BINS:
        smk = (qsd >= lo) & (qsd <= hi)
        for t in ("le300", "q4early", "mid", "h1", "ot"):
            for z in ("red", "fgr", "mid", "own"):
                mk = smk & (tz == t) & (yz == z)
                if mk.sum() >= 40:
                    for l in ("go", "FG", "punt"):
                        out[f"4th_{l}|sd{lo}..{hi}|{t}|{z}"] = float(np.mean(lab[mk] == l))
                    out[f"4th_n|sd{lo}..{hi}|{t}|{z}"] = int(mk.sum())
    for lo, hi in SD_BINS:
        smk = (qsd >= lo) & (qsd <= hi)
        for l in ("go", "FG", "punt"):
            out[f"4th_{l}|sd{lo}..{hi}"] = float(np.mean(lab[smk] == l)) if smk.sum() >= 40 else np.nan
    for z in ("red", "fgr", "mid", "own"):
        for l in ("go", "FG", "punt"):
            out[f"4th_{l}|{z}"] = float(np.mean(lab[yz == z] == l))
    allsd = P["sd"].to_numpy()
    allg = P["gsr"].to_numpy()
    allq = P["qtr"].to_numpy()
    kn = np.isin(code, (4,))
    sp = np.isin(code, (5,))
    for lo, hi in SD_BINS:
        mk = (allsd >= lo) & (allsd <= hi) & (allg <= 180) & (allq == 4)
        out[f"kneel|sd{lo}..{hi}|le180"] = float(np.mean(kn[mk])) if mk.sum() >= 40 else np.nan
        out[f"spike|sd{lo}..{hi}|le180"] = float(np.mean(sp[mk])) if mk.sum() >= 40 else np.nan
    out["kneel_game"] = float(kn.sum() / n)
    out["spike_game"] = float(sp.sum() / n)
    ot = P["qtr"].to_numpy() >= 5
    otg = np.unique(g[ot])
    out["ot_rate"] = float(len(otg) / n)
    if len(otg):
        mo = np.abs(M.reindex(otg).to_numpy())
        out["ot_tie"] = float(np.mean(mo == 0))
        out["ot_fg_end"] = float(np.mean(mo == 3))
        out["ot_td_end"] = float(np.mean(np.isin(mo, (6, 7, 8))))
    return out


import inspect  # noqa: E402

from sklearn.ensemble import HistGradientBoostingClassifier  # noqa: E402

SD_AX = np.r_[np.arange(-24, -10, 2), np.arange(-10, 11, 1), np.arange(12, 25, 2)].astype(float)
T4_AX = np.r_[np.arange(0, 421, 15), np.arange(480, 901, 60), [1200, 1800, 2700, 3600]].astype(float)
D4_AX = np.array([1, 2, 3, 4, 5, 6, 8, 10, 13, 18], dtype=float)
Y4_AX = np.arange(1, 100, 3).astype(float)
TL_AX = np.r_[np.arange(0, 601, 15), np.arange(1800, 1981, 15)].astype(float)
DL_AX = np.array([1, 3, 5, 8, 12, 20], dtype=float)
YL_AX = np.arange(2, 100, 4).astype(float)
CODE_L = {0: 0, 1: 1, 3: 2, 4: 3, 5: 4}
CODES_L = ((0,), (1,), (3,), (4,), (5,))
CODES_4 = ((0, 1), (3,), (2,))


def nearest(ax, v):
    return int(np.abs(ax - v).argmin())


def cls4_of(code):
    return np.select([code == 3, code == 2], [1, 2], default=0)


def in_late(qtr, gsr):
    return (qtr == 4 and gsr <= 600.0) or (qtr == 2 and gsr <= 1980.0)


def grid_rows(*axes):
    mesh = np.meshgrid(*axes, indexing="ij")
    return np.column_stack([m.ravel() for m in mesh]).astype(np.float32)


def batched_proba(clf, G, step=400000):
    return np.concatenate([clf.predict_proba(G[i : i + step]) for i in range(0, len(G), step)]).astype(np.float32)


def fit_pat(pbp):
    full = pbp.sort_values(["game_id", "play_id"]).reset_index(drop=True)
    m = full["play_type"].notna() & (full["play_type"] != "no_play")
    pl = full.loc[m].copy()
    grp = pl.groupby("game_id", sort=False)
    nt = grp["play_type"].shift(-1)
    nn = grp["play_type_nfl"].shift(-1)
    npost = grp["posteam"].shift(-1)
    gain = grp["posteam_score_post"].shift(-1) - grp["posteam_score"].shift(-1)
    td = (
        (pl["touchdown"] == 1)
        & pl["play_type"].isin(["run", "pass"])
        & ((pl["posteam_score_post"] - pl["posteam_score"]) >= 6)
    )
    is_pat = (nt == "extra_point") | (nn == "PAT2")
    rows = td & is_pat & (npost == pl["posteam"]) & pl["score_differential"].notna()
    d = pl.loc[rows]
    y = (nn.loc[rows] == "PAT2").to_numpy().astype(int)
    X = np.column_stack(
        [np.clip(d["score_differential"].to_numpy() + 6.0, -24, 24), d["game_seconds_remaining"].to_numpy()]
    )
    clf = HistGradientBoostingClassifier(max_depth=4, max_iter=120, learning_rate=0.08, random_state=1).fit(X, y)
    g = gain.loc[rows].to_numpy()
    seas = d["season"].to_numpy() if "season" in d.columns else np.full(len(d), 2016)
    xp_mask = (y == 0) & (seas >= 2015)
    s1 = float(np.mean(g[xp_mask] == 1))
    s2 = float(np.mean(g[y == 1] == 2))
    pg = clf.predict_proba(grid_rows(SD_AX, T4_AX))[:, 1].reshape(len(SD_AX), len(T4_AX))
    return pg, s1, s2, int(len(y)), float(y.mean())


def fit_decisions(trans):
    sd = np.clip(trans["sc_raw"].to_numpy(), -24, 24)
    gsr = trans["gsr_actual"].to_numpy()
    qtr = trans["qtr_actual"].to_numpy()
    ot = (qtr >= 5).astype(float)
    yl = trans["fp_raw"].to_numpy()
    dist = trans["dist_raw"].to_numpy()
    down = trans["down_i"].to_numpy()
    code = trans["play_type_code"].to_numpy()
    otf = (trans["off_to_raw"].to_numpy() > 0).astype(float)
    dtf = (trans["def_to_raw"].to_numpy() > 0).astype(float)
    m4 = (down == 4) & np.isin(code, (0, 1, 2, 3))
    X4 = np.column_stack([sd, gsr, ot, yl, dist, otf, dtf])[m4]
    c4 = HistGradientBoostingClassifier(max_depth=5, max_iter=200, learning_rate=0.08, random_state=1).fit(
        X4, cls4_of(code[m4])
    )
    G = grid_rows(SD_AX, T4_AX, [0.0, 1.0], Y4_AX, D4_AX, [0.0, 1.0], [0.0, 1.0])
    P4 = batched_proba(c4, G).reshape(len(SD_AX), len(T4_AX), 2, len(Y4_AX), len(D4_AX), 2, 2, 3)
    late = ((qtr == 4) & (gsr <= 600)) | ((qtr == 2) & (gsr <= 1980))
    ml = (down <= 3) & late & np.isin(code, (0, 1, 3, 4, 5))
    XL = np.column_stack([down, dist, yl, sd, gsr, otf, dtf])[ml]
    yL = np.array([CODE_L[int(c)] for c in code[ml]])
    cl = HistGradientBoostingClassifier(max_depth=5, max_iter=200, learning_rate=0.08, random_state=1).fit(XL, yL)
    G = grid_rows([1.0, 2.0, 3.0], DL_AX, YL_AX, SD_AX, TL_AX, [0.0, 1.0], [0.0, 1.0])
    PL = batched_proba(cl, G).reshape(3, len(DL_AX), len(YL_AX), len(SD_AX), len(TL_AX), 2, 2, 5)
    return P4, PL, int(m4.sum()), int(ml.sum())


def build_class_trees(trans):
    down_arr = trans["down_i"].to_numpy()
    phase_arr = trans["phase"].to_numpy()
    code = trans["play_type_code"].to_numpy()
    feats = sim.feature_matrix(
        trans["dist_raw"].to_numpy(),
        trans["fp_raw"].to_numpy(),
        trans["sc_raw"].to_numpy(),
        trans["time_raw"].to_numpy(),
        trans["off_to_raw"].to_numpy(),
        trans["def_to_raw"].to_numpy(),
        phase_arr,
    )
    trees = {}
    for dk in (1, 2, 3, 4):
        groups = CODES_4 if dk == 4 else CODES_L
        for ph in range(5):
            pm = sim.phase_pool_mask(phase_arr, ph)
            for ci, codes in enumerate(groups):
                sub = np.flatnonzero((down_arr == dk) & pm & np.isin(code, codes))
                if len(sub) >= 8:
                    trees[(dk, ph, ci)] = (sim.KDTree(feats[sub]), sub)
    return trees


def make_decide(tables, trans, pol4, poll, mech, seed):
    arrays = tables["arrays"]
    trees = build_class_trees(trans)
    cache = {}
    rng2 = np.random.default_rng(seed)
    order = (3, 2, 1, 0, 4)

    def cpick(rng, dk, ph, ci, dist, yl, sd, tf, ot_, dt_):
        for p in (ph,) + order:
            ent = trees.get((dk, p, ci))
            if ent is None:
                continue
            key = (sim.round_state_key(dk, p, dist, yl, sd, tf, ot_, dt_), dk, ci)
            nb = cache.get(key)
            if nb is None:
                tree, sub = ent
                f = sim.feature_matrix(
                    np.array([dist]),
                    np.array([yl]),
                    np.array([sd]),
                    np.array([tf]),
                    np.array([ot_]),
                    np.array([dt_]),
                    np.array([p]),
                )
                _, ind = tree.query(f, k=min(sim.K_NEIGHBORS, len(sub)))
                nb = sub[ind[0]]
                cache[key] = nb
            return int(nb[rng.integers(len(nb))])
        return None

    def draw(p, n):
        c = np.cumsum(p)
        return min(int(np.searchsorted(c, rng2.random() * c[-1])), n - 1)

    def decide(idx, rng, tbl, qtr, clock_val, in_ot, down, distance, yardline, score_diff, off_to, def_to, phase, time_feat):
        code0 = int(arrays["play_type_code"][idx])
        if code0 == 6:
            return idx
        dk = down if down in (1, 2, 3, 4) else 4
        sdc = min(max(score_diff, -24.0), 24.0)
        o1 = 1 if off_to > 0 else 0
        d1 = 1 if def_to > 0 else 0
        if dk == 4 and mech.get("fourth"):
            p = pol4[
                nearest(SD_AX, sdc), nearest(T4_AX, clock_val), 1 if in_ot else 0,
                nearest(Y4_AX, yardline), nearest(D4_AX, distance), o1, d1,
            ]
            ci = draw(p, 3)
            cur = 0 if code0 in (0, 1) else (1 if code0 == 3 else (2 if code0 == 2 else -1))
            if ci == cur:
                return idx
            j = cpick(rng, 4, phase, ci, distance, yardline, score_diff, time_feat, off_to, def_to)
            return idx if j is None else j
        if dk <= 3 and mech.get("late") and (not in_ot) and in_late(qtr, clock_val):
            p = poll[
                dk - 1, nearest(DL_AX, distance), nearest(YL_AX, yardline),
                nearest(SD_AX, sdc), nearest(TL_AX, clock_val), o1, d1,
            ]
            ci = draw(p, 5)
            cur = CODE_L.get(code0, -1)
            if ci == cur:
                return idx
            j = cpick(rng, dk, phase, ci, distance, yardline, score_diff, time_feat, off_to, def_to)
            return idx if j is None else j
        return idx

    return decide


def mech_init(train, mech_json):
    mech = json.loads(mech_json)
    sim.PBP_SNAPSHOT_DIR = SNAP
    condition = bool(mech.get("condition"))
    if mech.get("inner"):
        sim.SCORE_INNER_SCALE = mech["inner"]
        sim.SCORE_OUTER_SCALE = mech["outer"]
        sim.SCALE_TIME = mech["stime"]
    tables = sim.build_tables(tuple(train), condition_on_team=condition)
    _G["log"] = []
    if mech.get("fourth"):
        tables = dict(tables)
        tables["fourth_down_clf"] = None
    pbp = sim.load_reg_seasons(tuple(train))
    trans = sim.build_transition_frame(pbp)
    pol4 = poll = None
    if mech.get("fourth") or mech.get("late"):
        pol4, poll, _, _ = fit_decisions(trans)
    pat = fit_pat(pbp) if mech.get("pat") else None
    rngp = np.random.default_rng(mech.get("seed", 3) + 17)
    src = inspect.getsource(sim.run_one_game)
    anchor = '        down_key = down if down in (1, 2, 3, 4) else 4\n        fourth_clf = tables.get("fourth_down_clf")'
    assert anchor in src
    call = (
        "        idx = DECIDE(idx, rng, tables, qtr, clock_val, in_ot, down, distance, yardline, "
        "score_diff, off_to, def_to, phase, time_feat)\n"
    )
    src = src.replace(anchor, call + anchor)
    ns = dict(sim.__dict__)
    if mech.get("fourth") or mech.get("late"):
        ns["DECIDE"] = make_decide(tables, trans, pol4, poll, mech, mech.get("seed", 3))
    else:
        ns["DECIDE"] = lambda idx, *a: idx
    if condition:
        a = 'drawn_net = float(arrays["off_row"][idx] - arrays["def_row"][idx])'
        b = "yard_shift = TEAM_RATING_YARD_GAIN * ((off_sim - def_sim) - drawn_net)"
        assert a in src and b in src
        src = src.replace(a, 'drawn_net = float(arrays["off_row"][idx] + DEF_SIGN * arrays["def_row"][idx])')
        src = src.replace(b, "yard_shift = YARD_BIAS + YARD_GAIN * ((off_sim + DEF_SIGN * def_sim) - drawn_net)")
        ns["YARD_GAIN"] = float(mech["yard_gain"])
        ns["DEF_SIGN"] = float(mech["def_sign"])
        ns["YARD_BIAS"] = float(mech["yard_bias"])
    exec(src, ns)
    _G["tables"] = tables
    _G["ns"] = ns

    def pol(down, distance, yardline, score_diff, qtr, clock_val, drawn):
        if pat is not None and qtr < 5:
            po = drawn["points_off"]
            pdf = drawn["points_def"]
            scorer_off = po >= 6 and pdf == 0
            scorer_def = pdf >= 6 and po == 0
            if scorer_off or scorer_def:
                sda = (score_diff if scorer_off else -score_diff) + 6.0
                p2 = float(pat[0][nearest(SD_AX, min(max(sda, -24.0), 24.0)), nearest(T4_AX, clock_val)])
                if rngp.random() < p2:
                    pts = 8.0 if rngp.random() < pat[2] else 6.0
                else:
                    pts = 7.0 if rngp.random() < pat[1] else 6.0
                drawn = dict(drawn)
                if scorer_off:
                    drawn["points_off"] = pts
                else:
                    drawn["points_def"] = pts
        _G["log"].append(
            (
                down, distance, yardline, score_diff, clock_val, qtr,
                int(drawn["play_type_code"]), float(drawn["points_off"]), float(drawn["points_def"]),
                bool(drawn["flip"]), float(drawn["clock_elapsed"]),
            )
        )
        return drawn

    _G["pol"] = pol
    _G["mech"] = mech


def mech_batch(task):
    seed, n = task
    rng = np.random.default_rng(seed)
    t = _G["tables"]
    rows = []
    margins = []
    ns = _G["ns"]
    for gi in range(n):
        _G["log"].clear()
        state = sim.initial_kickoff_state(rng, t["opening_pool"])
        rec, _ = ns["run_one_game"](state, t, rng, 600.0, _G["pol"], sim.K_NEIGHBORS, sim.MAX_PLAYS_PER_GAME)
        a = np.array(_G["log"], dtype=np.float64).reshape(-1, 11)
        rows.append(np.column_stack([np.full(len(a), gi), a]))
        margins.append(rec["margin"])
    return seed, np.concatenate(rows), np.array(margins)


HEAD = [
    "m3", "m7", "m10", "m14", "m17", "margin_sd", "le3", "tie", "pts_game", "td_game", "fg_made_game",
    "punt_game", "4th_go_rate", "4th_FG_rate", "4th_punt_rate", "pat_pts8", "pat_pts6", "kneel_game",
    "spike_game", "ot_rate", "ot_tie", "last_FG_share", "m3|last_FG", "at300_f3|0..0", "at120_f3|0..0",
    "at300_f3|1..3", "at900_f3|1..3", "pat8|pre-8..-8", "pat8|pre-5..-5", "pat8|pre-2..-2",
    "kneel|sd1..3|le180", "kneel|sd9..99|le180", "p_tied2min_flip", "p_tied2min_el", "p_tied5min_el",
    "p_trail1_3_5min_el", "p_trail1_3_5min_flip", "p_ot_drive_el", "p_ot_drive_score",
]

VARIANTS = {
    "base": {},
    "fourth": {"fourth": 1},
    "late": {"late": 1},
    "pat": {"pat": 1},
    "all": {"fourth": 1, "late": 1, "pat": 1},
    "fine": {"fourth": 1, "late": 1, "pat": 1, "inner": 1.0, "outer": 4.0, "stime": 120.0},
    "fine2": {"fourth": 1, "late": 1, "pat": 1, "inner": 0.5, "outer": 2.0, "stime": 60.0},
}


def frames_from(res):
    frames = []
    ms = []
    off = 0
    for _, a, m in res:
        g = a[:, 0].astype(np.int64) + off
        frames.append(
            pd.DataFrame(
                {
                    "g": g, "down": a[:, 1], "dist": a[:, 2], "yl": a[:, 3], "sd": a[:, 4], "gsr": a[:, 5],
                    "qtr": a[:, 6], "code": a[:, 7].astype(int), "po": a[:, 8], "pdf": a[:, 9],
                    "flip": a[:, 10].astype(bool), "el": a[:, 11],
                }
            )
        )
        ms.append(pd.Series(m, index=np.arange(len(m)) + off))
        off += len(m)
    return pd.concat(frames, ignore_index=True), pd.concat(ms)


def cmd_mech(args):
    import multiprocessing as mp

    OUT.mkdir(parents=True, exist_ok=True)
    P, M = real_frames(EVAL)
    real = analyse(P, M)
    real.update({"p_" + k: v for k, v in probe_stats(P).items()})
    P2, M2 = real_frames(TRAIN)
    real_tr = analyse(P2, M2)
    real_tr.update({"p_" + k: v for k, v in probe_stats(P2).items()})
    results = {}
    for name in args.variants.split(","):
        mech = dict(VARIANTS[name], seed=args.seed)
        t0 = time.time()
        per = max(50, args.games // (args.workers * 4))
        tasks = [(args.seed * 100000 + i, per) for i in range((args.games + per - 1) // per)]
        ctx = mp.get_context("spawn")
        with ctx.Pool(args.workers, initializer=mech_init, initargs=(TRAIN, json.dumps(mech))) as pool:
            res = pool.map(mech_batch, tasks, chunksize=1)
        S, MS = frames_from(res)
        results[name] = analyse(S, MS)
        results[name].update({"p_" + k: v for k, v in probe_stats(S).items()})
        (OUT / f"mech_{name}.json").write_text(json.dumps(clean(results[name]), indent=1))
        print(f"{name} games {len(MS)} {time.time() - t0:.0f}s m3 {results[name]['m3']:.4f}", flush=True)
    names = list(results)
    print(f"{'key':32s} {'real_eval':>9s} {'real_trn':>9s} " + " ".join(f"{n:>8s}" for n in names))
    for k in HEAD:
        vals = " ".join(f"{results[n].get(k, float('nan')):8.4f}" for n in names)
        print(f"{k:32s} {real.get(k, float('nan')):9.4f} {real_tr.get(k, float('nan')):9.4f} {vals}")


def probe_stats(P):
    P = P.sort_values("g", kind="stable").reset_index(drop=True)
    out = {}
    sd = P["sd"].to_numpy(); gsr = P["gsr"].to_numpy(); q = P["qtr"].to_numpy(); dn = P["down"].to_numpy()
    code = P["code"].to_numpy(); el = P["el"].to_numpy(); po = P["po"].to_numpy(); pdf = P["pdf"].to_numpy()
    flip = P["flip"].to_numpy()
    for name, mk in (
        ("tied2min", (q == 4) & (sd == 0) & (gsr <= 120) & (dn <= 3) & np.isin(code, (0, 1, 4, 5))),
        ("tied5min", (q == 4) & (sd == 0) & (gsr <= 300) & (gsr > 120) & (dn <= 3) & np.isin(code, (0, 1, 4, 5))),
        ("trail1_3_5min", (q == 4) & (sd >= -3) & (sd <= -1) & (gsr <= 300) & (gsr > 120) & (dn <= 3) & np.isin(code, (0, 1, 4, 5))),
        ("ot", (q >= 5) & np.isin(code, (0, 1, 2, 3, 4, 5))),
    ):
        out[f"{name}_n"] = int(mk.sum())
        out[f"{name}_el"] = float(el[mk].mean())
        out[f"{name}_run"] = float(np.mean(code[mk] == 0))
        out[f"{name}_pass"] = float(np.mean(code[mk] == 1))
        out[f"{name}_kneel"] = float(np.mean(code[mk] == 4))
        out[f"{name}_score"] = float(np.mean((po[mk] > 0) | (pdf[mk] > 0)))
        out[f"{name}_flip"] = float(np.mean(flip[mk]))
    ot = q >= 5
    prev = np.r_[False, flip[:-1]]
    newg = np.r_[True, P["g"].to_numpy()[1:] != P["g"].to_numpy()[:-1]]
    drv = np.cumsum(prev | newg)
    d = pd.DataFrame({"drv": drv[ot], "el": el[ot], "sc": ((po > 0) | (pdf > 0))[ot], "fgm": ((code == 3) & (po == 3))[ot], "td": (po >= 6)[ot]})
    g = d.groupby("drv").agg(el=("el", "sum"), n=("el", "size"), sc=("sc", "max"), fgm=("fgm", "max"), td=("td", "max"))
    out["ot_drives"] = int(len(g)); out["ot_drive_el"] = float(g["el"].mean()); out["ot_drive_plays"] = float(g["n"].mean())
    out["ot_drive_score"] = float(g["sc"].mean()); out["ot_drive_fg"] = float(g["fgm"].mean()); out["ot_drive_td"] = float(g["td"].mean())
    return out


def cmd_probe(args):
    import multiprocessing as mp

    P, M = real_frames(EVAL)
    P2, M2 = real_frames(TRAIN)
    rows = {"real_eval": probe_stats(P), "real_trn": probe_stats(P2)}
    for name in ("base", "all"):
        mech = dict(VARIANTS[name], seed=args.seed)
        per = max(50, args.games // (args.workers * 4))
        tasks = [(args.seed * 100000 + i, per) for i in range((args.games + per - 1) // per)]
        with mp.get_context("spawn").Pool(args.workers, initializer=mech_init, initargs=(TRAIN, json.dumps(mech))) as pool:
            res = pool.map(mech_batch, tasks, chunksize=1)
        S, _ = frames_from(res)
        rows[name] = probe_stats(S)
    for k in rows["real_eval"]:
        print(f"{k:24s} " + " ".join(f"{n}={rows[n][k]:9.4f}" for n in rows))


def gen_batch(task):
    import mod25_generator as gen

    world, sidx, seed, sched, ratings = task
    rng = np.random.default_rng(seed)
    t = _G["tables"]
    ns = _G["ns"]
    rows = []
    for week, h, a in sched:
        _G["log"].clear()
        state = sim.initial_kickoff_state(rng, t["opening_pool"])
        hr = {"off": gen.quant(ratings[week][h][0]), "def": gen.quant(ratings[week][h][1])}
        ar = {"off": gen.quant(ratings[week][a][0]), "def": gen.quant(ratings[week][a][1])}
        rec, _ = ns["run_one_game"](state, t, rng, 600.0, _G["pol"], sim.K_NEIGHBORS, sim.MAX_PLAYS_PER_GAME, hr, ar)
        t["nn_weight_cache_cond"].clear()
        rows.append((float(rec["margin"]), float(rec["total"])))
    return world, sidx, rows


def cmd_gen(args):
    import multiprocessing as mp

    import mod25_generator as gen

    OUT.mkdir(parents=True, exist_ok=True)
    targets = json.loads((gen.OUT_DIR / "real_targets.json").read_text())
    fit = gen.load_fit()
    lo_, ld_ = gen.league_means()
    setting = dict(gen.SETTING_DEFAULTS)
    setting.update({"scale": args.scale, "yard_gain": args.yard_gain, "def_sign": 1.0, "yard_bias": 0.75, "drift": 1.0})
    root = np.random.SeedSequence(args.seed)
    wseeds = root.spawn(args.worlds)
    tasks = []
    for w in range(args.worlds):
        rng = np.random.default_rng(wseeds[w])
        lat = gen.gen_world_latents(rng, args.seasons, fit, setting)
        gs = wseeds[w].spawn(args.seasons)
        for sidx, (weekly, qb_out) in enumerate(lat):
            sched = gen.make_schedule(rng)
            ratings = gen.season_ratings(weekly, qb_out, fit, setting, lo_, ld_)
            tasks.append((w, sidx, int(gs[sidx].generate_state(1)[0]), sched, ratings))
    summary = {}
    for name in args.variants.split(","):
        mech = dict(VARIANTS[name], seed=args.seed, condition=1, yard_gain=setting["yard_gain"], def_sign=1.0, yard_bias=0.75)
        t0 = time.time()
        with mp.get_context("spawn").Pool(args.workers, initializer=mech_init, initargs=(TRAIN, json.dumps(mech))) as pool:
            res = pool.map(gen_batch, tasks, chunksize=1)
        arr = np.array([r for _, _, rows in res for r in rows])
        m = arr[:, 0]
        am = np.abs(m)
        out = {f"mass_{k}": float(np.mean(am == k)) for k in (3, 7, 10, 14, 17)}
        out["margin_sd"] = float(m.std())
        out["share_le3"] = float(np.mean(am <= 3))
        out["pts_game"] = float(arr[:, 1].mean())
        out["games"] = int(len(m))
        summary[name] = out
        print(name, f"{time.time() - t0:.0f}s", json.dumps(out), flush=True)
    keys = ["mass_3", "mass_7", "mass_10", "mass_14", "mass_17", "margin_sd", "share_le3"]
    print(f"{'gate':12s} {'real':>7s} {'lo':>7s} {'hi':>7s} " + " ".join(f"{n:>9s}" for n in summary))
    for k in keys:
        tg = targets["gates"][k]
        cells = " ".join(f"{summary[n][k]:7.4f}{'*' if tg['lo'] <= summary[n][k] <= tg['hi'] else ' '} " for n in summary)
        print(f"{k:12s} {tg['real']:7.4f} {tg['lo']:7.4f} {tg['hi']:7.4f} {cells}")
    (OUT / "heldout_gen.json").write_text(json.dumps(summary, indent=1))


def clean(d):
    return {k: (None if (isinstance(v, float) and np.isnan(v)) else v) for k, v in d.items()}


def cmd_diag(args):
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    P, M = real_frames(EVAL if args.eval else TRAIN)
    real = analyse(P, M)
    P2, M2 = real_frames(TRAIN if args.eval else EVAL)
    real_other = analyse(P2, M2)
    print(f"real games {len(M)} m3 {real['m3']:.4f} {time.time() - t0:.0f}s", flush=True)
    S, MS = run_sim(TRAIN, args.games, args.workers, args.seed)
    simm = analyse(S, MS)
    print(f"sim games {len(MS)} m3 {simm['m3']:.4f} {time.time() - t0:.0f}s", flush=True)
    (OUT / f"diag_{args.tag}.json").write_text(json.dumps({"real": clean(real), "sim": clean(simm), "real_other": clean(real_other)}, indent=1))
    show(real, simm, real_other=real_other)


def show(real, simm, keys=None, real_other=None):
    for k in real:
        if keys and not any(k.startswith(p) for p in keys):
            continue
        r, s = real[k], simm.get(k)
        if r is None or s is None or (isinstance(r, float) and np.isnan(r)):
            continue
        o = real_other.get(k) if real_other else None
        ot = "" if o is None else f" other {o:9.4f}"
        print(f"{k:55s} real {r:9.4f} sim {s:9.4f}{ot}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("diag")
    d.add_argument("--tag", default="before")
    d.add_argument("--games", type=int, default=12000)
    d.add_argument("--workers", type=int, default=6)
    d.add_argument("--seed", type=int, default=5)
    d.add_argument("--eval", action="store_true")
    m = sub.add_parser("mech")
    m.add_argument("--variants", default="base,all")
    m.add_argument("--games", type=int, default=12000)
    m.add_argument("--workers", type=int, default=6)
    m.add_argument("--seed", type=int, default=5)
    pr = sub.add_parser("probe")
    pr.add_argument("--games", type=int, default=12000)
    pr.add_argument("--workers", type=int, default=6)
    pr.add_argument("--seed", type=int, default=5)
    gn = sub.add_parser("gen")
    gn.add_argument("--variants", default="base,fine2")
    gn.add_argument("--worlds", type=int, default=8)
    gn.add_argument("--seasons", type=int, default=8)
    gn.add_argument("--workers", type=int, default=6)
    gn.add_argument("--seed", type=int, default=9)
    gn.add_argument("--scale", type=float, default=2.0)
    gn.add_argument("--yard-gain", dest="yard_gain", type=float, default=2.0)
    args = ap.parse_args()
    {"diag": cmd_diag, "mech": cmd_mech, "probe": cmd_probe, "gen": cmd_gen}[args.cmd](args)


if __name__ == "__main__":
    main()
