import argparse
import inspect
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "src"))

import mod25_mechanisms as m25  # noqa: E402
import sim04_engine as sim  # noqa: E402
from sklearn.ensemble import HistGradientBoostingClassifier  # noqa: E402

SNAP = m25.SNAP
TRAIN = m25.TRAIN
EVAL = m25.EVAL
OUT = REPO / "artifacts" / "mod25c"
_G = {}

B_CFG = dict(m25.VARIANTS["fine2"])
VARIANTS = {
    "base": {},
    "b": B_CFG,
    "c": dict(B_CFG, tov=1, clk=1, tout=1),
    "c_tov": dict(B_CFG, tov=1),
    "c_clk": dict(B_CFG, clk=1),
    "c_tout": dict(B_CFG, tout=1),
    "c_ot": dict(B_CFG, tov=1, clk=1, tout=1, ot=1),
}
DN_AX = np.array([1.0, 2.0, 3.0, 4.0])
TX_AX = np.unique(np.r_[m25.T4_AX, [1815, 1830, 1845, 1860, 1875, 1890, 1905, 1920, 1950, 2100, 2400, 3000]]).astype(float)


def zone_v(qtr, g):
    return np.select(
        [qtr >= 5, (qtr == 4) & (g <= 120), (qtr == 4) & (g <= 300), (qtr == 4) & (g <= 600), (qtr == 4) & (g <= 900), (g > 1800) & (g <= 1920)],
        [0, 1, 2, 3, 4, 5],
        default=6,
    ).astype(np.int64)


def zone_s(qtr, g):
    if qtr >= 5:
        return 0
    if qtr == 4:
        if g <= 120:
            return 1
        if g <= 300:
            return 2
        if g <= 600:
            return 3
        if g <= 900:
            return 4
    if 1800 < g <= 1920:
        return 5
    return 6


SB_EDGES = np.array([-8, -3, 0, 1, 4, 9])


def sb_v(sd):
    return np.digitize(sd, SB_EDGES).astype(np.int64)


def sb_s(sd):
    return int(np.digitize(sd, SB_EDGES))


def attrs_from(pbp, trans):
    right = pbp[["game_id", "play_id", "interception", "fumble_lost", "sack", "epa", "yards_gained"]].drop_duplicates(["game_id", "play_id"])
    at = trans[["game_id", "play_id"]].merge(right, on=["game_id", "play_id"], how="left")
    tov = ((at["interception"].fillna(0) == 1) | (at["fumble_lost"].fillna(0) == 1)).to_numpy().astype(np.int8)
    return {
        "tov": tov,
        "sack": at["sack"].fillna(0.0).to_numpy(dtype=np.int8),
        "epa": at["epa"].fillna(0.0).to_numpy(dtype=np.float32),
        "yards": at["yards_gained"].fillna(0.0).to_numpy(dtype=np.float32),
        "int": at["interception"].fillna(0.0).to_numpy(dtype=np.int8),
        "fl": at["fumble_lost"].fillna(0.0).to_numpy(dtype=np.int8),
    }


def cls_of(code, flip, po, pdf, yg):
    sc = ((po > 0) | (pdf > 0)).astype(np.int64)
    return (np.clip(code, 0, 7).astype(np.int64) * 8 + flip.astype(np.int64) * 4 + sc * 2 + (yg == 0).astype(np.int64))


def clock_table(trans):
    code = trans["play_type_code"].to_numpy()
    flip = trans["possession_flip"].to_numpy().astype(bool)
    cl = cls_of(code, flip, trans["points_off"].to_numpy(), trans["points_def"].to_numpy(), trans["yards_gained"].to_numpy())
    zn = zone_v(trans["qtr_actual"].to_numpy(), trans["gsr_actual"].to_numpy())
    sb = sb_v(trans["sc_raw"].to_numpy())
    el = trans["clock_elapsed"].to_numpy(dtype=float)
    key = (cl * 8 + zn) * 8 + sb
    n = 64 * 8 * 8
    cnt = np.bincount(key, minlength=n)
    sm = np.bincount(key, weights=el, minlength=n)
    zk = cl * 8 + zn
    zc = np.bincount(zk, minlength=512)
    zs = np.bincount(zk, weights=el, minlength=512)
    ccnt = np.bincount(cl, minlength=64)
    csum = np.bincount(cl, weights=el, minlength=64)
    full = np.full(n, np.nan)
    for k in range(n):
        if cnt[k] >= 40:
            full[k] = sm[k] / cnt[k]
        elif zc[k // 8] >= 40:
            full[k] = zs[k // 8] / zc[k // 8]
        elif ccnt[k // 64] >= 40:
            full[k] = csum[k // 64] / ccnt[k // 64]
    return full.reshape(64, 8, 8), cl, full[key]


def fit_grids(trans):
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
    return sd, gsr, ot, yl, dist, down, code, otf, dtf


def fit_flag_grids(trans, tov):
    sd, gsr, ot, yl, dist, down, code, otf, dtf = fit_grids(trans)
    rp = np.isin(code, (0, 1)) & np.isin(down, (1, 2, 3, 4))
    X = np.column_stack([down, dist, yl, sd, gsr, ot, code])[rp]
    clf = HistGradientBoostingClassifier(max_depth=4, max_iter=150, learning_rate=0.05, min_samples_leaf=300, l2_regularization=1.0, random_state=1).fit(X, tov[rp])
    G = m25.grid_rows(DN_AX, m25.DL_AX, m25.YL_AX, m25.SD_AX, TX_AX, [0.0, 1.0], [0.0, 1.0])
    ptov = m25.batched_proba(clf, G)[:, 1].reshape(4, len(m25.DL_AX), len(m25.YL_AX), len(m25.SD_AX), len(TX_AX), 2, 2)
    toflag = ((trans["off_to_used"].to_numpy() + trans["def_to_used"].to_numpy()) > 0).astype(int)
    X2 = np.column_stack([down, sd, gsr, ot, code, otf, dtf])[rp]
    clf2 = HistGradientBoostingClassifier(max_depth=4, max_iter=150, learning_rate=0.05, min_samples_leaf=300, l2_regularization=1.0, random_state=1).fit(X2, toflag[rp])
    G2 = m25.grid_rows(DN_AX, m25.SD_AX, TX_AX, [0.0, 1.0], [0.0, 1.0], [0.0, 1.0], [0.0, 1.0])
    ptout = m25.batched_proba(clf2, G2)[:, 1].reshape(4, len(m25.SD_AX), len(TX_AX), 2, 2, 2, 2)
    return ptov.astype(np.float32), ptout.astype(np.float32), float(tov[rp].mean()), float(toflag[rp].mean())


def get_policies(train, trans, tov):
    OUT.mkdir(parents=True, exist_ok=True)
    f = OUT / f"policies2_{train[0]}_{train[-1]}.npz"
    if f.exists():
        z = np.load(f)
        return z["ptov"], z["ptout"]
    ptov, ptout, a, b = fit_flag_grids(trans, tov)
    np.savez(f, ptov=ptov, ptout=ptout)
    return ptov, ptout


def ensure_policies():
    f = OUT / f"policies2_{TRAIN[0]}_{TRAIN[-1]}.npz"
    if f.exists():
        return
    sim.PBP_SNAPSHOT_DIR = SNAP
    pbp = sim.load_reg_seasons(tuple(TRAIN))
    trans = sim.build_transition_frame(pbp)
    at = attrs_from(pbp, trans)
    get_policies(TRAIN, trans, at["tov"])


def build_trees_flag(trans, tov, toflag, ot_only=False):
    down_arr = trans["down_i"].to_numpy()
    phase_arr = trans["phase"].to_numpy()
    code = trans["play_type_code"].to_numpy()
    feats = sim.feature_matrix(
        trans["dist_raw"].to_numpy(), trans["fp_raw"].to_numpy(), trans["sc_raw"].to_numpy(), trans["time_raw"].to_numpy(),
        trans["off_to_raw"].to_numpy(), trans["def_to_raw"].to_numpy(), phase_arr,
    )
    trees = {}
    for dk in (1, 2, 3, 4):
        for ph in range(5):
            pm = (phase_arr == 4) if (ot_only and ph == 4) else sim.phase_pool_mask(phase_arr, ph)
            for c in (0, 1):
                for tv in (0, 1):
                    for to in (0, 1):
                        sub = np.flatnonzero((down_arr == dk) & pm & (code == c) & (tov == tv) & (toflag == to))
                        if len(sub) >= 8:
                            trees[(dk, ph, c, tv, to)] = (sim.KDTree(feats[sub]), sub)
    return trees


def ot_group(dk, code):
    if dk == 4:
        return 1 if code == 3 else (2 if code == 2 else 0)
    return int(code)


def build_ot_pool(trans):
    down_arr = trans["down_i"].to_numpy()
    phase_arr = trans["phase"].to_numpy()
    code = trans["play_type_code"].to_numpy()
    feats = sim.feature_matrix(
        trans["dist_raw"].to_numpy(), trans["fp_raw"].to_numpy(), trans["sc_raw"].to_numpy(), trans["time_raw"].to_numpy(),
        trans["off_to_raw"].to_numpy(), trans["def_to_raw"].to_numpy(), phase_arr,
    )
    pools = {}
    for dk in (1, 2, 3, 4):
        for c in range(6):
            g = ot_group(dk, c)
            sub = np.flatnonzero((down_arr == dk) & (phase_arr == 4) & (code == c))
            if len(sub):
                pools.setdefault((dk, g), []).append(sub)
    out = {}
    for k, v in pools.items():
        sub = np.concatenate(v)
        if len(sub) >= 8:
            out[k] = (sim.KDTree(feats[sub]), sub)
    return out


def make_c_decide(base_decide, trans, tov, toflag, ptov, ptout, cfg, seed):
    trees = build_trees_flag(trans, tov, toflag, bool(cfg.get("ot")))
    otp = build_ot_pool(trans) if cfg.get("ot") else {}
    otcache = {}
    use_ot = bool(cfg.get("ot"))
    code_arr = trans["play_type_code"].to_numpy()
    cache = {}
    rng2 = np.random.default_rng(seed + 101)
    order = (3, 2, 1, 0, 4)
    use_tov = bool(cfg.get("tov"))
    use_to = bool(cfg.get("tout"))

    def pick(rng, dk, ph, c, tv, to, dist, yl, sd, tf, ot_, dt_):
        for p in (ph,) + order:
            ent = trees.get((dk, p, c, tv, to))
            if ent is None:
                continue
            key = (sim.round_state_key(dk, p, dist, yl, sd, tf, ot_, dt_), dk, c, tv, to)
            nb = cache.get(key)
            if nb is None:
                tree, sub = ent
                f = sim.feature_matrix(np.array([dist]), np.array([yl]), np.array([sd]), np.array([tf]), np.array([ot_]), np.array([dt_]), np.array([p]))
                _, ind = tree.query(f, k=min(sim.K_NEIGHBORS, len(sub)))
                nb = sub[ind[0]]
                cache[key] = nb
            return int(nb[rng.integers(len(nb))])
        return None

    def decide(idx, rng, tbl, qtr, clock_val, in_ot, down, distance, yardline, score_diff, off_to, def_to, phase, time_feat):
        idx = base_decide(idx, rng, tbl, qtr, clock_val, in_ot, down, distance, yardline, score_diff, off_to, def_to, phase, time_feat)
        c0 = int(code_arr[idx])
        dk = down if down in (1, 2, 3, 4) else 4
        if use_ot and in_ot and c0 != 6:
            ent = otp.get((dk, ot_group(dk, c0)))
            if ent is not None:
                key = (sim.round_state_key(dk, 4, distance, yardline, score_diff, time_feat, off_to, def_to), dk, ot_group(dk, c0))
                nb = otcache.get(key)
                if nb is None:
                    tree, sub = ent
                    f = sim.feature_matrix(np.array([distance]), np.array([yardline]), np.array([score_diff]), np.array([time_feat]), np.array([off_to]), np.array([def_to]), np.array([4]))
                    _, ind = tree.query(f, k=min(sim.K_NEIGHBORS, len(sub)))
                    nb = sub[ind[0]]
                    otcache[key] = nb
                idx = int(nb[rng.integers(len(nb))])
                c0 = int(code_arr[idx])
        if c0 not in (0, 1):
            return idx
        sdc = min(max(score_diff, -24.0), 24.0)
        ot1 = 1 if in_ot else 0
        o1 = 1 if off_to > 0 else 0
        d1 = 1 if def_to > 0 else 0
        tv_t = int(tov[idx])
        to_t = int(toflag[idx])
        if use_tov:
            p = ptov[dk - 1, m25.nearest(m25.DL_AX, distance), m25.nearest(m25.YL_AX, yardline), m25.nearest(m25.SD_AX, sdc), m25.nearest(TX_AX, clock_val), ot1, c0]
            tv_t = 1 if rng2.random() < p else 0
        if use_to:
            p = ptout[dk - 1, m25.nearest(m25.SD_AX, sdc), m25.nearest(TX_AX, clock_val), ot1, c0, o1, d1]
            to_t = 1 if rng2.random() < p else 0
        if tv_t == int(tov[idx]) and to_t == int(toflag[idx]):
            return idx
        j = pick(rng, dk, phase, c0, tv_t, to_t, distance, yardline, score_diff, time_feat, off_to, def_to)
        if j is None and to_t == 1:
            j = pick(rng, dk, phase, c0, tv_t, 0, distance, yardline, score_diff, time_feat, off_to, def_to)
        return idx if j is None else j

    return decide


def c_init(train, cfg_json):
    cfg = json.loads(cfg_json)
    sim.PBP_SNAPSHOT_DIR = SNAP
    if cfg.get("inner"):
        sim.SCORE_INNER_SCALE = cfg["inner"]
        sim.SCORE_OUTER_SCALE = cfg["outer"]
        sim.SCALE_TIME = cfg["stime"]
    cap = {}
    ol, ot_f = sim.load_reg_seasons, sim.build_transition_frame

    def lh(s):
        p = ol(s)
        cap["pbp"] = p
        return p

    def th(p, team_ratings=None):
        t = ot_f(p, team_ratings=team_ratings)
        cap["trans"] = t
        return t

    sim.load_reg_seasons = lh
    sim.build_transition_frame = th
    try:
        tables = sim.build_tables(tuple(train), condition_on_team=bool(cfg.get("condition")))
    finally:
        sim.load_reg_seasons = ol
        sim.build_transition_frame = ot_f
    pbp, trans = cap["pbp"], cap["trans"]
    at = attrs_from(pbp, trans)
    tables = dict(tables)
    tables["attrs"] = dict(at, code=tables["arrays"]["play_type_code"])
    if cfg.get("fourth"):
        tables["fourth_down_clf"] = None
    pol4 = poll = None
    if cfg.get("fourth") or cfg.get("late"):
        pol4, poll, _, _ = m25.fit_decisions(trans)
    pat = m25.fit_pat(pbp) if cfg.get("pat") else None
    seed = cfg.get("seed", 3)
    rngp = np.random.default_rng(seed + 17)
    toflag = ((trans["off_to_used"].to_numpy() + trans["def_to_used"].to_numpy()) > 0).astype(np.int8)
    if cfg.get("fourth") or cfg.get("late"):
        base_decide = m25.make_decide(tables, trans, pol4, poll, cfg, seed)
    else:
        def base_decide(idx, *a):
            return idx
    if cfg.get("tov") or cfg.get("tout") or cfg.get("ot"):
        ptov, ptout = get_policies(train, trans, at["tov"])
        decide = make_c_decide(base_decide, trans, at["tov"], toflag, ptov, ptout, cfg, seed)
    else:
        decide = base_decide
    clk = None
    if cfg.get("clk"):
        tab, cl, own = clock_table(trans)
        clk = (tab, cl, own)
    src = inspect.getsource(sim.run_one_game)
    anchor = '        down_key = down if down in (1, 2, 3, 4) else 4\n        fourth_clf = tables.get("fourth_down_clf")'
    call = "        idx = DECIDE(idx, rng, tables, qtr, clock_val, in_ot, down, distance, yardline, score_diff, off_to, def_to, phase, time_feat)\n"
    assert anchor in src
    src = src.replace(anchor, call + anchor)
    a1 = '"play_type_code": arrays["play_type_code"][idx],\n        }'
    assert a1 in src
    src = src.replace(a1, '"play_type_code": arrays["play_type_code"][idx],\n            "idx": int(idx),\n        }')
    c1 = "drawn = policy(down, distance, yardline, score_diff, qtr, clock_val, drawn)"
    assert c1 in src
    src = src.replace(
        c1,
        c1 + "\n        PLAYLOG.append((1 if offense == 'home' else 0, int(idx), float(drawn['yards_gained']) - float(arrays['yards_gained'][idx])))",
    )
    ns = dict(sim.__dict__)
    ns["DECIDE"] = decide
    ns["PLAYLOG"] = []
    if cfg.get("condition"):
        a = 'drawn_net = float(arrays["off_row"][idx] - arrays["def_row"][idx])'
        b = "yard_shift = TEAM_RATING_YARD_GAIN * ((off_sim - def_sim) - drawn_net)"
        assert a in src and b in src
        src = src.replace(a, 'drawn_net = float(arrays["off_row"][idx] + DEF_SIGN * arrays["def_row"][idx])')
        src = src.replace(b, "yard_shift = YARD_BIAS + YARD_GAIN * ((off_sim + DEF_SIGN * def_sim) - drawn_net)")
        ns["YARD_GAIN"] = float(cfg["yard_gain"])
        ns["DEF_SIGN"] = float(cfg["def_sign"])
        ns["YARD_BIAS"] = float(cfg["yard_bias"])
    exec(src, ns)
    log = []

    def pol(down, distance, yardline, score_diff, qtr, clock_val, drawn):
        i = drawn["idx"]
        if clk is not None:
            cur = clk[0][clk[1][i], zone_s(qtr, clock_val), sb_s(score_diff)]
            d = cur - clk[2][i]
            if d == d:
                drawn["clock_elapsed"] = max(0.0, float(drawn["clock_elapsed"]) + float(d))
        if pat is not None and qtr < 5:
            po = drawn["points_off"]
            pdf = drawn["points_def"]
            scorer_off = po >= 6 and pdf == 0
            scorer_def = pdf >= 6 and po == 0
            if scorer_off or scorer_def:
                sda = (score_diff if scorer_off else -score_diff) + 6.0
                p2 = float(pat[0][m25.nearest(m25.SD_AX, min(max(sda, -24.0), 24.0)), m25.nearest(m25.T4_AX, clock_val)])
                if rngp.random() < p2:
                    pts = 8.0 if rngp.random() < pat[2] else 6.0
                else:
                    pts = 7.0 if rngp.random() < pat[1] else 6.0
                drawn = dict(drawn)
                if scorer_off:
                    drawn["points_off"] = pts
                else:
                    drawn["points_def"] = pts
        log.append(
            (
                down, distance, yardline, score_diff, clock_val, qtr,
                int(drawn["play_type_code"]), float(drawn["points_off"]), float(drawn["points_def"]),
                bool(drawn["flip"]), float(drawn["clock_elapsed"]), i,
            )
        )
        return drawn

    _G.update(tables=tables, ns=ns, pol=pol, log=log, cfg=cfg, to_arr=(trans["off_to_used"].to_numpy() + trans["def_to_used"].to_numpy()))


def c_batch(task):
    seed, n = task
    rng = np.random.default_rng(seed)
    t = _G["tables"]
    ns = _G["ns"]
    tov = t["attrs"]["tov"]
    to_arr = _G["to_arr"]
    rows, margins = [], []
    for gi in range(n):
        _G["log"].clear()
        ns["PLAYLOG"].clear()
        state = sim.initial_kickoff_state(rng, t["opening_pool"])
        rec, _ = ns["run_one_game"](state, t, rng, 600.0, _G["pol"], sim.K_NEIGHBORS, sim.MAX_PLAYS_PER_GAME)
        a = np.array(_G["log"], dtype=np.float64).reshape(-1, 12)
        ix = a[:, 11].astype(np.int64)
        rows.append(np.column_stack([np.full(len(a), gi), a[:, :11], tov[ix], to_arr[ix]]))
        margins.append(rec["margin"])
    return seed, np.concatenate(rows), np.array(margins)


COLN = ["g", "down", "dist", "yl", "sd", "gsr", "qtr", "code", "po", "pdf", "flip", "el", "tov", "tout"]


def frames_c(res):
    frames, ms, off = [], [], 0
    for _, a, m in res:
        d = pd.DataFrame(a, columns=COLN)
        d["g"] = d["g"].astype(np.int64) + off
        d["code"] = d["code"].astype(int)
        d["flip"] = d["flip"].astype(bool)
        frames.append(d)
        ms.append(pd.Series(m, index=np.arange(len(m)) + off))
        off += len(m)
    return pd.concat(frames, ignore_index=True), pd.concat(ms)


def real_frames_c(seasons):
    pbp = sim.load_reg_seasons(tuple(seasons))
    tr = sim.build_transition_frame(pbp)
    at = attrs_from(pbp, tr)
    tr = tr.assign(tov=at["tov"]).sort_values(["game_id", "play_id"]).reset_index(drop=True)
    gid = pd.factorize(tr["game_id"])[0]
    P = pd.DataFrame(
        {
            "g": gid, "down": tr["down_i"].to_numpy(), "dist": tr["dist_raw"].to_numpy(), "yl": tr["fp_raw"].to_numpy(),
            "sd": tr["sc_raw"].to_numpy(), "gsr": tr["gsr_actual"].to_numpy(), "qtr": tr["qtr_actual"].to_numpy(),
            "code": tr["play_type_code"].to_numpy(), "po": tr["points_off"].to_numpy(), "pdf": tr["points_def"].to_numpy(),
            "flip": tr["possession_flip"].to_numpy().astype(bool), "el": tr["clock_elapsed"].to_numpy(),
            "tov": tr["tov"].to_numpy(), "tout": (tr["off_to_used"] + tr["def_to_used"]).to_numpy(),
        }
    )
    last = tr.groupby(gid).tail(1)
    M = pd.Series(last["home_margin_post"].to_numpy(), index=gid[last.index.to_numpy()])
    return P, M


def analyse_noise(P, M):
    out = {}
    n = len(M)
    P = P.sort_values("g", kind="stable").reset_index(drop=True)
    g_ = P["g"].to_numpy()
    code = P["code"].to_numpy()
    rp = np.isin(code, (0, 1))
    sd = P["sd"].to_numpy()
    gsr = P["gsr"].to_numpy()
    qtr = P["qtr"].to_numpy()
    zn = zone_v(qtr, gsr)
    sb = sb_v(sd)
    yl = P["yl"].to_numpy()
    yz = np.where(yl <= 20, 0, np.where(yl <= 36, 1, np.where(yl <= 55, 2, 3)))
    tov = P["tov"].to_numpy().astype(float)
    el = P["el"].to_numpy()
    to = P["tout"].to_numpy()
    out["tov_rate"] = float(tov[rp].mean())
    out["rp_per_game"] = float(rp.sum() / n)
    out["tov_per_game"] = float(tov[rp].sum() / n)
    ZN = ["ot", "q4le120", "q4le300", "q4le600", "q4le900", "q2end", "other"]
    SBN = ["le-9", "-8..-4", "-3..-1", "0", "1..3", "4..8", "ge9"]
    for z in range(7):
        mk = rp & (zn == z)
        out[f"tov|{ZN[z]}"] = float(tov[mk].mean()) if mk.sum() > 30 else np.nan
        out[f"snap_el|{ZN[z]}"] = float(el[mk].mean()) if mk.sum() > 30 else np.nan
    for s in range(7):
        mk = rp & (sb == s)
        out[f"tov|sd{SBN[s]}"] = float(tov[mk].mean()) if mk.sum() > 30 else np.nan
        out[f"snap_el|sd{SBN[s]}"] = float(el[mk].mean()) if mk.sum() > 30 else np.nan
    for f_, nm in enumerate(("red", "fgr", "mid", "own")):
        mk = rp & (yz == f_)
        out[f"tov|{nm}"] = float(tov[mk].mean())
        out[f"snap_el|{nm}"] = float(el[mk].mean())
    for z in (1, 2, 3, 4):
        for s in range(7):
            mk = rp & (zn == z) & (sb == s)
            if mk.sum() >= 150:
                out[f"tov|{ZN[z]}|sd{SBN[s]}"] = float(tov[mk].mean())
                out[f"snap_el|{ZN[z]}|sd{SBN[s]}"] = float(el[mk].mean())
    out["snap_el_rp"] = float(el[rp].mean())
    reg = qtr <= 4
    for T in (2700, 1800, 900, 300):
        mk = reg & (gsr <= T)
        fi = pd.DataFrame({"g": g_[mk], "sd": sd[mk]}).groupby("g").head(1)
        out[f"var_sd_at{T}"] = float(np.mean(fi["sd"].to_numpy() ** 2))
    tot = pd.Series((P["po"] + P["pdf"]).to_numpy()).groupby(g_).sum()
    out["total_pts_sd"] = float(tot.std())
    out["total_pts_mean"] = float(tot.mean())
    out["margin_sd"] = float(np.std(M.to_numpy()))
    out["margin_var"] = float(np.var(M.to_numpy()))
    out["el_game"] = float(el.sum() / n)
    out["plays_game"] = float(len(P) / n)
    out["tout_game"] = float(to.sum() / n)
    for z in range(7):
        mk = zn == z
        out[f"tout_per_game|{ZN[z]}"] = float(to[mk].sum() / n)
    for s in range(7):
        mk = (zn >= 1) & (zn <= 4) & (sb == s)
        out[f"tout_rate|late|sd{SBN[s]}"] = float(np.mean(to[mk] > 0)) if mk.sum() > 100 else np.nan
    h1 = (qtr <= 2)
    out["tout_h1"] = float(to[h1].sum() / n)
    out["tout_h2"] = float(to[(qtr >= 3) & (qtr <= 4)].sum() / n)
    g = P["g"].to_numpy()
    flip = P["flip"].to_numpy()
    po = P["po"].to_numpy()
    pdf = P["pdf"].to_numpy()
    ot = qtr >= 5
    og = np.unique(g[ot])
    out["ot_rate"] = float(len(og) / n)
    if len(og):
        mo = np.abs(M.reindex(og).to_numpy())
        out["ot_tie_given_ot"] = float(np.mean(mo == 0))
        out["ot_tie_game"] = float(np.sum(mo == 0) / n)
    prev = np.r_[False, flip[:-1]]
    newg = np.r_[True, g[1:] != g[:-1]]
    drv = np.cumsum(prev | newg)
    d = pd.DataFrame({"g": g[ot], "drv": drv[ot], "el": el[ot], "sc": ((po > 0) | (pdf > 0))[ot], "td": (po >= 6)[ot], "fg": ((code == 3) & (po == 3))[ot], "tov": tov[ot], "pl": 1})
    dd = d.groupby(["g", "drv"]).agg(el=("el", "sum"), sc=("sc", "max"), td=("td", "max"), fg=("fg", "max"), tov=("tov", "max"), pl=("pl", "sum")).reset_index()
    if len(dd):
        dd["k"] = dd.groupby("g").cumcount()
        out["ot_drives_per_otgame"] = float(len(dd) / len(og))
        out["ot_drive_el"] = float(dd["el"].mean())
        out["ot_drive_plays"] = float(dd["pl"].mean())
        out["ot_drive_score"] = float(dd["sc"].mean())
        out["ot_drive_td"] = float(dd["td"].mean())
        out["ot_drive_fg"] = float(dd["fg"].mean())
        out["ot_drive_tov"] = float(dd["tov"].mean())
        f1 = dd[dd["k"] == 0]
        out["ot_first_score"] = float(f1["sc"].mean())
        out["ot_first_td"] = float(f1["td"].mean())
        out["ot_first_fg"] = float(f1["fg"].mean())
        out["ot_first_el"] = float(f1["el"].mean())
        out["ot_total_el_per_game"] = float(d["el"].sum() / len(og))
    return out


def cmd_measure(args):
    import multiprocessing as mp

    OUT.mkdir(parents=True, exist_ok=True)
    ensure_policies()
    P, M = real_frames_c(EVAL)
    real_e = analyse_noise(P, M)
    P2, M2 = real_frames_c(TRAIN)
    real_t = analyse_noise(P2, M2)
    res_all = {"real_eval": real_e, "real_train": real_t}
    mshape = {}
    for name in args.variants.split(","):
        cfg = dict(VARIANTS[name], seed=args.seed)
        t0 = time.time()
        per = max(50, args.games // (args.workers * 4))
        tasks = [(args.seed * 100000 + i, per) for i in range((args.games + per - 1) // per)]
        with mp.get_context("spawn").Pool(args.workers, initializer=c_init, initargs=(TRAIN, json.dumps(cfg))) as pool:
            res = pool.map(c_batch, tasks, chunksize=1)
        S, MS = frames_c(res)
        res_all[name] = analyse_noise(S, MS)
        am = np.abs(MS.to_numpy())
        mshape[name] = {f"m{k}": float(np.mean(am == k)) for k in (3, 7, 10, 14, 17)}
        mshape[name]["sd"] = float(MS.std())
        mshape[name]["pts"] = float((S["po"] + S["pdf"]).sum() / len(MS))
        print(f"{name} {len(MS)} games {time.time() - t0:.0f}s", flush=True)
    (OUT / f"measure_{args.tag}.json").write_text(json.dumps({"res": {k: m25.clean(v) for k, v in res_all.items()}, "shape": mshape}, indent=1))
    names = list(res_all)
    print(f"{'key':34s} " + " ".join(f"{n:>10s}" for n in names))
    for k in real_e:
        vals = [res_all[n].get(k, float("nan")) for n in names]
        if any(v is None or (isinstance(v, float) and np.isnan(v)) for v in vals[:2]):
            continue
        print(f"{k:34s} " + " ".join(f"{v:10.4f}" for v in vals))
    print("shape (diagnostic only)", json.dumps(mshape))


def c_gen_init(setting):
    cfg = json.loads(setting["mech"])
    cfg.update(condition=1, yard_gain=setting["yard_gain"], def_sign=1.0, yard_bias=0.75)
    c_init(TRAIN, json.dumps(cfg))


def c_play_season(task):
    tables = _G["tables"]
    ns = _G["ns"]
    arr = tables["attrs"]
    setting_epa = 0.07
    world, sidx, seed, sched, ratings = task
    import mod25_generator as gen

    rng = np.random.default_rng(seed)
    gm_rows, chunks = [], []
    for gi, (week, h, a) in enumerate(sched):
        state = sim.initial_kickoff_state(rng, tables["opening_pool"])
        ns["PLAYLOG"].clear()
        _G["log"].clear()
        hr = {"off": gen.quant(ratings[week][h][0]), "def": gen.quant(ratings[week][h][1])}
        ar = {"off": gen.quant(ratings[week][a][0]), "def": gen.quant(ratings[week][a][1])}
        rec, cap_hit = ns["run_one_game"](state, tables, rng, 600.0, _G["pol"], sim.K_NEIGHBORS, sim.MAX_PLAYS_PER_GAME, hr, ar)
        tables["nn_weight_cache_cond"].clear()
        log = np.array(ns["PLAYLOG"], dtype=np.float64).reshape(-1, 3)
        idx = log[:, 1].astype(np.int64)
        code = arr["code"][idx]
        keep = (code == 0) | (code == 1)
        idx = idx[keep]
        shift = log[keep, 2]
        offhome = log[keep, 0].astype(np.int8)
        kind = np.where(code[keep] == 0, 0, np.where(arr["sack"][idx] == 1, 2, 1)).astype(np.int8)
        epa = arr["epa"][idx] + setting_epa * shift
        yards = arr["yards"][idx] + shift
        turn = np.where(arr["int"][idx] == 1, 1, np.where(arr["fl"][idx] == 1, 2, 0)).astype(np.int8)
        chunks.append(np.rec.fromarrays([np.full(len(idx), gi, dtype=np.int16), offhome, kind, epa.astype(np.float32), yards.astype(np.float32), turn], names="g,offhome,kind,epa,yards,turn"))
        total, margin = float(rec["total"]), float(rec["margin"])
        gm_rows.append((gi, (total + margin) / 2.0, (total - margin) / 2.0, bool(rec["went_ot"]), bool(cap_hit)))
    plays = np.concatenate(chunks) if chunks else np.empty(0)
    return world, sidx, gm_rows, plays


def real_gate_metrics():
    import mod25_generator as gen

    f = OUT / "real_metrics.json"
    if f.exists():
        return json.loads(f.read_text())
    games = gen.real_games(tuple(range(2009, 2026)))
    plays = gen.load_real_plays(range(2009, 2026))
    ts = gen.team_stats_from_plays(plays)
    games = games[games["game_id"].isin(ts["game_id"])].reset_index(drop=True)
    out = {}
    for nm, ss in (("2011_2017", range(2011, 2018)), ("2018_2025", range(2018, 2026))):
        m, ac, d = gen.all_metrics(games, ts, list(ss))
        m["r2_pooled"] = float(np.corrcoef(d["x"], d["y"])[0, 1] ** 2)
        m["nonstrength_var"] = float(m["margin_sd"] ** 2 * (1 - m["r2_pooled"]))
        out[nm] = m
    f.write_text(json.dumps(out, indent=1))
    return out


def gen_eval(variant, scale, yard_gain, worlds, seasons, workers, seed):
    ensure_policies()
    import mod25_generator as gen

    cfgj = json.dumps(dict(VARIANTS[variant], seed=seed))
    setting = dict(gen.SETTING_DEFAULTS)
    setting.update({"scale": scale, "yard_gain": yard_gain, "def_sign": 1.0, "yard_bias": 0.75, "drift": 1.0, "mech": cfgj})
    gen.init_worker = c_gen_init
    gen.play_season = c_play_season
    games, plays, latents, elapsed = gen.run_generation(setting, worlds, seasons, workers, seed, progress=False)
    ts = gen.team_stats_from_plays(plays)
    keep = [w * 1000 + s + 1 for w in range(worlds) for s in range(2, seasons)]
    m, ac, d = gen.all_metrics(games, ts, keep)
    m["r2_pooled"] = float(np.corrcoef(d["x"], d["y"])[0, 1] ** 2)
    m["nonstrength_var"] = float(m["margin_sd"] ** 2 * (1 - m["r2_pooled"]))
    m["pts_game"] = float((games["home_score"] + games["away_score"]).mean())
    m["games_per_sec"] = float(len(games) / elapsed)
    return m


GK = ["mass_3", "mass_7", "mass_10", "mass_14", "mass_17", "margin_sd", "share_le3", "home_edge", "epa_autocorr_lag1", "r2_w1_4", "r2_w5_9", "r2_w10_18", "r2_pooled", "nonstrength_var", "pts_game"]


def cmd_grid(args):
    real = real_gate_metrics()["2011_2017"]
    tg = json.loads((REPO / "artifacts" / "mod25_generator" / "real_targets.json").read_text())["gates"]
    rows = {}
    for sc in [float(x) for x in args.scales.split(",")]:
        m = gen_eval(args.variant, sc, args.yard_gain, args.worlds, args.seasons, args.workers, args.seed)
        loss = 0.0
        for k in ("epa_autocorr_lag1", "r2_w1_4", "r2_w5_9", "r2_w10_18"):
            hw = (tg[k]["hi"] - tg[k]["lo"]) / 2.0
            loss += ((m[k] - real[k]) / hw) ** 2
        m["loss"] = float(loss)
        rows[str(sc)] = m
        print(sc, json.dumps({k: round(m[k], 4) for k in GK + ["loss"]}), flush=True)
    (OUT / f"grid_{args.variant}.json").write_text(json.dumps({"real_2011_2017": real, "rows": rows}, indent=1))


def cmd_gate(args):
    real = real_gate_metrics()["2018_2025"]
    tg = json.loads((REPO / "artifacts" / "mod25_generator" / "real_targets.json").read_text())["gates"]
    res = {}
    for v in args.variants.split(","):
        t0 = time.time()
        res[v] = gen_eval(v, args.scale, args.yard_gain, args.worlds, args.seasons, args.workers, args.seed)
        print(v, f"{time.time() - t0:.0f}s", flush=True)
    (OUT / f"gate_{args.tag}.json").write_text(json.dumps({"real_2018_2025": real, "res": res, "scale": args.scale, "yard_gain": args.yard_gain}, indent=1))
    print(f"{'gate':20s} {'real':>9s} {'lo':>7s} {'hi':>7s} " + " ".join(f"{n:>10s}" for n in res))
    for k in GK:
        lo = tg.get(k, {}).get("lo", float("nan"))
        hi = tg.get(k, {}).get("hi", float("nan"))
        cells = " ".join(f"{res[n][k]:9.4f}{'*' if lo <= res[n][k] <= hi else ' '}" for n in res)
        print(f"{k:20s} {real.get(k, float("nan")):9.4f} {lo:7.4f} {hi:7.4f} {cells}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("measure")
    m.add_argument("--variants", default="base,b")
    m.add_argument("--games", type=int, default=12000)
    m.add_argument("--workers", type=int, default=6)
    m.add_argument("--seed", type=int, default=5)
    m.add_argument("--tag", default="m1")
    g = sub.add_parser("gate")
    g.add_argument("--variants", default="base,b,c")
    g.add_argument("--worlds", type=int, default=8)
    g.add_argument("--seasons", type=int, default=8)
    g.add_argument("--workers", type=int, default=6)
    g.add_argument("--seed", type=int, default=9)
    g.add_argument("--scale", type=float, default=2.0)
    g.add_argument("--yard-gain", dest="yard_gain", type=float, default=2.0)
    g.add_argument("--tag", default="g1")
    r = sub.add_parser("grid")
    r.add_argument("--variant", default="c")
    r.add_argument("--scales", default="1.5,2.0,2.5,3.0")
    r.add_argument("--worlds", type=int, default=6)
    r.add_argument("--seasons", type=int, default=8)
    r.add_argument("--workers", type=int, default=6)
    r.add_argument("--seed", type=int, default=21)
    r.add_argument("--yard-gain", dest="yard_gain", type=float, default=2.0)
    args = ap.parse_args()
    {"measure": cmd_measure, "gate": cmd_gate, "grid": cmd_grid}[args.cmd](args)


if __name__ == "__main__":
    main()
