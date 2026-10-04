import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

KDIR = REPO / "artifacts" / "mod25e3" / "kfit"
FIT = KDIR / "fit.json"
FIT_GAME = KDIR / "fit_game.json"
FIT_DN = KDIR / "fit_dn.json"
FIT_BLUP = KDIR / "fit_blup.json"
BLUP_OUT = FIT_BLUP
SNAP = REPO / "data" / "pbp" / "raw" / "20260929T191306Z"
SEASONS = tuple(range(2009, 2018))
H_GRID = (0.05, 0.1, 0.2, 0.35, 0.5, 0.7, 1.0, 1.4, 2.0, 2.8, 4.0, 8.0, np.inf)
LAM_GRID = (1.0, 1.5, 2.0, 3.0, 5.0, 10.0, 30.0, 100.0, 1e6)
CUR = (0.5, 1e6)
STRIDE = 3
YEDGES = np.array([-1, 0, 1, 2, 3, 4, 5, 6, 8, 10, 15, 25])


def enabled():
    return os.environ.get("KFIT") == "1"


def outcome_codes(t):
    y = np.round(t["yards_gained"].to_numpy(float))
    yb = np.searchsorted(YEDGES, y, side="right")
    flip = t["possession_flip"].to_numpy().astype(bool)
    po = t["points_off"].to_numpy()
    pd_ = t["points_def"].to_numpy()
    fd = (~flip) & (po == 0) & (pd_ == 0) & (t["next_down"].to_numpy() == 1)
    cls = np.zeros(len(t), dtype=int)
    cls[fd] = 4
    cls[flip & (po == 0) & (pd_ == 0)] = 2
    cls[po > 0] = 1
    cls[pd_ > 0] = 3
    return cls * (len(YEDGES) + 1) + yb


def build_frames():
    import mod25d_variance as dv
    import sim04_engine as sim

    sim.PBP_SNAPSHOT_DIR = SNAP
    pbp = sim.load_reg_seasons(SEASONS)
    ratings = sim.load_team_ratings()
    trans = sim.build_transition_frame(pbp, team_ratings=ratings).reset_index(drop=True)
    plain = sim.build_transition_frame(pbp).reset_index(drop=True)
    eo, ef = dv.blup_rows(pbp, plain, SEASONS)
    ratings = ratings.assign(season=ratings["game_id"].str[:4].astype(int))
    return sim, trans, ratings, eo, ef


def fold_eval(sim, trans, off, dfn, season, h_sd_of, cls, fold_season, hs, lams):
    tr = season != fold_season
    te = np.flatnonzero(~tr)[::STRIDE]
    feats = sim.feature_matrix(
        trans["dist_raw"].to_numpy(), trans["fp_raw"].to_numpy(), trans["sc_raw"].to_numpy(),
        trans["time_raw"].to_numpy(), trans["off_to_raw"].to_numpy(), trans["def_to_raw"].to_numpy(),
        trans["phase"].to_numpy(),
    )
    down = trans["down_i"].to_numpy()
    phase = trans["phase"].to_numpy()
    home = trans["is_home_off"].to_numpy()
    trees = {}
    for dn in (1, 2, 3, 4):
        for ph in range(5):
            m = tr & (down == dn) & sim.phase_pool_mask(phase, ph)
            idx = np.flatnonzero(m)
            if len(idx):
                trees[(dn, ph)] = (cKDTree(feats[idx]), idx)
    sd = h_sd_of
    n = len(te)
    nbmax = sim.K_STATE
    D2 = np.full((n, nbmax), np.inf, dtype=np.float64)
    M = np.zeros((n, nbmax), dtype=bool)
    Hm = np.zeros((n, nbmax), dtype=bool)
    counts = np.bincount(cls[tr], minlength=cls.max() + 1) / tr.sum()
    base = counts[cls[te]]
    groups = {}
    for j, r in enumerate(te):
        key = (int(down[r]), int(phase[r]))
        if key not in trees:
            key = (int(down[r]), 0)
        groups.setdefault(key, []).append(j)
    for key, js in groups.items():
        tree, idx = trees[key]
        k = min(nbmax, len(idx))
        _, ind = tree.query(feats[te[js]], k=k)
        ind = np.asarray(ind).reshape(len(js), k)
        nb = idx[ind]
        rows = te[js]
        d2 = (off[nb] - off[rows][:, None]) ** 2 + (dfn[nb] - dfn[rows][:, None]) ** 2
        D2[js, :k] = d2
        M[js, :k] = cls[nb] == cls[rows][:, None]
        Hm[js, :k] = home[nb] == home[rows][:, None]
    valid = np.isfinite(D2)
    out = np.zeros((len(hs), len(lams)))
    lh = np.where(Hm, 1.0, 0.0)
    for a, hsc in enumerate(hs):
        base_lw = np.zeros_like(D2) if np.isinf(hsc) else -np.where(valid, D2, 0.0) / (2.0 * (hsc * sd) ** 2)
        for b, lam in enumerate(lams):
            lw = np.where(valid, base_lw + lh * np.log(lam), -np.inf)
            w = np.exp(lw - lw.max(axis=1, keepdims=True))
            s = w.sum(axis=1)
            q = (w * M).sum(axis=1) / s
            ess = s * s / (w * w).sum(axis=1)
            p = (ess * q + base) / (ess + 1.0)
            out[a, b] = float(np.log(p).mean())
    return out, n


def fit():
    sim, trans, ratings, eo, ef = build_frames()
    season = trans["game_id"].str[:4].astype(int).to_numpy()
    cls = outcome_codes(trans)
    hs, lams = list(H_GRID), list(LAM_GRID)
    res = {"grid_h": [float(x) if np.isfinite(x) else "inf" for x in hs], "grid_lam": lams, "stride": STRIDE, "k_state": int(sim.K_STATE), "cur": list(CUR), "rating_sets": {}}
    for name in ("pregame_rolling", "blup_insample"):
        if name == "pregame_rolling":
            off, dfn = trans["off_row"].to_numpy(), trans["def_row"].to_numpy()
        else:
            lo, ld = float(np.nanmean(trans["off_row"])), float(np.nanmean(trans["def_row"]))
            off, dfn = lo + eo, ld + ef
        folds = np.zeros((len(SEASONS), len(hs), len(lams)))
        ns_ = []
        sds = []
        for f, s in enumerate(SEASONS):
            if name == "pregame_rolling":
                tr_g = ratings[ratings["season"] != s]
                sd = float(np.nanstd(pd.concat([tr_g["home_off_epa_per_play"], tr_g["away_off_epa_per_play"]]).to_numpy()))
            else:
                sd = float(np.std(eo[season != s]))
            sds.append(sd)
            folds[f], n = fold_eval(sim, trans, off, dfn, season, sd, cls, s, hs, lams)
            ns_.append(n)
            print(name, s, n, flush=True)
        mean = np.average(folds, axis=0, weights=ns_)
        ai, bi = [hs.index(CUR[0]), lams.index(CUR[1])]
        bh, bl = np.unravel_index(int(mean.argmax()), mean.shape)
        nested = []
        for f in range(len(SEASONS)):
            m = np.delete(np.arange(len(SEASONS)), f)
            mm = np.average(folds[m], axis=0, weights=np.array(ns_)[m])
            c = np.unravel_index(int(mm.argmax()), mm.shape)
            nested.append(float(folds[f][c] - folds[f][ai, bi]))
        inf_i = hs.index(np.inf)
        res["rating_sets"][name] = {
            "sd_by_fold": sds,
            "n_by_fold": ns_,
            "ll_mean": mean.tolist(),
            "ll_folds": folds.tolist(),
            "best_h_scale": float(hs[bh]) if np.isfinite(hs[bh]) else "inf",
            "best_lam": float(lams[bl]),
            "ll_best": float(mean[bh, bl]),
            "ll_current": float(mean[ai, bi]),
            "ll_no_team_weighting": float(mean[inf_i].max()),
            "fold_wins_best_vs_current": int((folds[:, bh, bl] > folds[:, ai, bi]).sum()),
            "fold_diff_best_vs_current": (folds[:, bh, bl] - folds[:, ai, bi]).tolist(),
            "nested_loso_diff_vs_current": nested,
            "nested_wins": int(sum(d > 0 for d in nested)),
            "looks": int(len(hs) * len(lams)),
            "best_h_by_lam": [float(hs[int(mean[:, b].argmax())]) if np.isfinite(hs[int(mean[:, b].argmax())]) else "inf" for b in range(len(lams))],
            "best_lam_by_h": [float(lams[int(mean[a].argmax())]) for a in range(len(hs))],
        }
    FIT.parent.mkdir(parents=True, exist_ok=True)
    primary = res["rating_sets"]["pregame_rolling"]
    res["h_scale"] = primary["best_h_scale"]
    res["lam"] = primary["best_lam"]
    FIT.write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in primary.items() if k in ("best_h_scale", "best_lam", "ll_best", "ll_current", "ll_no_team_weighting", "fold_wins_best_vs_current", "nested_wins")}))
    b = res["rating_sets"]["blup_insample"]
    print(json.dumps({k: v for k, v in b.items() if k in ("best_h_scale", "best_lam", "ll_best", "ll_current", "ll_no_team_weighting", "fold_wins_best_vs_current", "nested_wins")}))


def install_kf():
    import mod25d_variance as dv

    if os.environ.get("KFIT_SRC") == "dn":
        install_dn()
        return
    if os.environ.get("KFIT_SRC") == "blup":
        install_blup()
        return
    if os.environ.get("KFIT_SRC") == "eiv":
        install_eiv()
        return
    spec = json.loads((FIT_GAME if os.environ.get("KFIT_SRC") == "game" else FIT).read_text(encoding="utf-8"))
    import sim04_engine as sim

    scale = spec["h_scale"]
    lam = float(spec["lam"])
    old = float(sim.TEAM_KERNEL_H_SCALE)
    ns = dv._G["ns"]
    t = dv._G["tables"]
    if scale == "inf":
        t["team_kernel_h"] = float("inf")
    else:
        t["team_kernel_h"] = float(t["team_kernel_h"]) * float(scale) / old
    sim.TEAM_KERNEL_LAMBDA = lam
    ns["TEAM_KERNEL_LAMBDA"] = lam
    t["nn_weight_cache_cond"].clear()


GH_GRID = tuple(float(x) for x in np.geomspace(0.05, 16.0, 15)) + (np.inf,)
GLAM_GRID = (1.0, 1.5, 2.0, 3.0, 5.0, 10.0, 30.0, 100.0, 1e6)
GLAM_SHORT = (1.0, 1e6)
STATS = ("epa", "success", "yards")
MIN_PLAYS = 10
CTL_REL_SD = (0.073, 0.053)


def game_stats(pbp, trans):
    m = trans[["game_id", "play_id"]].merge(
        pbp[["game_id", "play_id", "posteam", "defteam", "epa", "success", "yards_gained"]].drop_duplicates(["game_id", "play_id"]),
        on=["game_id", "play_id"], how="left",
    )
    Y = np.column_stack([m["epa"].to_numpy(float), m["success"].to_numpy(float), m["yards_gained"].to_numpy(float)])
    return Y, m["posteam"].astype(str).to_numpy(), m["defteam"].astype(str).to_numpy()


def make_control(trans, season, pos, dfn_team, Y, gamma, seed):
    rng = np.random.default_rng(seed)
    gid = trans["game_id"].to_numpy()
    mo, md = float(np.nanmean(trans["off_row"])), float(np.nanmean(trans["def_row"]))
    lv = {}
    jt = {}

    def draw(store, key, sd):
        if key not in store:
            store[key] = rng.normal(0.0, sd)
        return store[key]

    n = len(trans)
    fo = np.empty(n)
    fd = np.empty(n)
    for i in range(n):
        fo[i] = draw(lv, ("o", season[i], pos[i]), CTL_REL_SD[0]) + draw(jt, ("o", gid[i], pos[i]), CTL_REL_SD[1])
        fd[i] = draw(lv, ("d", season[i], dfn_team[i]), CTL_REL_SD[0]) + draw(jt, ("d", gid[i], dfn_team[i]), CTL_REL_SD[1])
    delta = gamma * (fo + fd)
    sd_y = np.nanstd(Y, axis=0)
    Y2 = Y + delta[:, None] * (sd_y / sd_y[0])[None, :]
    return mo + fo, md + fd, Y2, delta * sd_y[2] / sd_y[0]


def game_fold(sim, trans, off, dfn, season, fold, sd, Yn, gcode, mode, hs, lams):
    tr = season != fold
    te = np.flatnonzero(~tr)
    te = te[np.isfinite(Yn[te]).all(axis=1)]
    feats = sim.feature_matrix(
        trans["dist_raw"].to_numpy(), trans["fp_raw"].to_numpy(), trans["sc_raw"].to_numpy(),
        trans["time_raw"].to_numpy(), trans["off_to_raw"].to_numpy(), trans["def_to_raw"].to_numpy(),
        trans["phase"].to_numpy(),
    )
    down = trans["down_i"].to_numpy()
    phase = trans["phase"].to_numpy()
    home = trans["is_home_off"].to_numpy()
    ok = np.isfinite(Yn).all(axis=1)
    trees = {}
    for dn in (1, 2, 3, 4):
        for ph in range(5):
            idx = np.flatnonzero(tr & ok & (down == dn) & sim.phase_pool_mask(phase, ph))
            if len(idx):
                trees[(dn, ph)] = (cKDTree(feats[idx]), idx)
    K = sim.K_STATE
    n = len(te)
    D2 = np.full((n, K), np.inf)
    Hm = np.zeros((n, K), dtype=bool)
    Ys = np.zeros((n, K, 3))
    groups = {}
    for j, r in enumerate(te):
        key = (int(down[r]), int(phase[r]))
        if key not in trees:
            key = (int(down[r]), 0)
        groups.setdefault(key, []).append(j)
    for key, js in groups.items():
        tree, idx = trees[key]
        k = min(K, len(idx))
        _, ind = tree.query(feats[te[js]], k=k)
        nb = idx[np.asarray(ind).reshape(len(js), k)]
        rows = te[js]
        if mode == "off":
            d2 = (off[nb] - off[rows][:, None]) ** 2
        elif mode == "def":
            d2 = (dfn[nb] - dfn[rows][:, None]) ** 2
        else:
            d2 = (off[nb] - off[rows][:, None]) ** 2 + (dfn[nb] - dfn[rows][:, None]) ** 2
        D2[js, :k] = d2
        Hm[js, :k] = home[nb] == home[rows][:, None]
        Ys[js, :k] = Yn[nb]
    Ys2 = Ys * Ys
    valid = np.isfinite(D2)
    D2z = np.where(valid, D2, 0.0)
    ug, gi = np.unique(gcode[te], return_inverse=True)
    cnt = np.bincount(gi).astype(float)
    keep = cnt >= MIN_PLAYS
    R = np.stack([np.bincount(gi, weights=Yn[te, k]) / cnt for k in range(3)])
    M = np.zeros((len(hs), len(lams), 3, len(ug)))
    V = np.zeros_like(M)
    lh = np.where(Hm, 1.0, 0.0)
    for a, hsc in enumerate(hs):
        base_lw = np.zeros_like(D2) if np.isinf(hsc) else -D2z / (2.0 * (hsc * sd) ** 2)
        for b, lam in enumerate(lams):
            lw = np.where(valid, base_lw + lh * np.log(lam), -np.inf)
            w = np.exp(lw - lw.max(axis=1, keepdims=True))
            w /= w.sum(axis=1, keepdims=True)
            for k in range(3):
                m = (w * Ys[:, :, k]).sum(axis=1)
                v = np.maximum((w * Ys2[:, :, k]).sum(axis=1) - m * m, 0.0)
                M[a, b, k] = np.bincount(gi, weights=m) / cnt
                V[a, b, k] = np.bincount(gi, weights=v) / cnt**2
    return ug[keep], M[..., keep], V[..., keep], R[:, keep], cnt[keep]


def tau_ll(m, v, r, mask_fit, mask_eval):
    from scipy.optimize import minimize_scalar

    e = (r - m) ** 2
    vr = float(np.var(r[mask_fit])) + 1e-12

    def nll(lt):
        s = v[mask_fit] + np.exp(lt)
        return float(np.mean(np.log(2 * np.pi * s) + e[mask_fit] / s))

    res = minimize_scalar(nll, bounds=(np.log(vr * 1e-8), np.log(vr * 2.0)), method="bounded")
    t = float(np.exp(res.x))
    s = v[mask_eval] + t
    return float(np.mean(-0.5 * (np.log(2 * np.pi * s) + e[mask_eval] / s))), t


def game_scan(sim, trans, off, dfn, season, Yn, gcode, mode, hs, lams, sds, tag, lab=None):
    parts = []
    for f, s in enumerate(SEASONS):
        if lab is not None:
            off, dfn, sdf = lab(s)
            sds = list(sds)
            sds[f] = sdf
        parts.append(game_fold(sim, trans, off, dfn, season, s, sds[f], Yn, gcode, mode, hs, lams))
        print(tag, mode, s, len(parts[-1][0]), flush=True)
    fid = np.concatenate([np.full(len(p[0]), f) for f, p in enumerate(parts)])
    M = np.concatenate([p[1] for p in parts], axis=-1)
    V = np.concatenate([p[2] for p in parts], axis=-1)
    R = np.concatenate([p[3] for p in parts], axis=-1)
    LL = np.zeros((len(hs), len(lams), 3, len(SEASONS)))
    for a in range(len(hs)):
        for b in range(len(lams)):
            for k in range(3):
                for f in range(len(SEASONS)):
                    LL[a, b, k, f] = tau_ll(M[a, b, k], V[a, b, k], R[k], fid != f, fid == f)[0]
    return LL, np.bincount(fid, minlength=len(SEASONS))


def summarize(LL, ng, hs, lams):
    out = {}
    hv = np.array([h if np.isfinite(h) else 1e9 for h in hs])
    ci = (int(np.argmin(np.abs(hv - CUR[0]))), len(lams) - 1)
    inf_i = hs.index(np.inf)
    nolam = lams.index(1.0)
    for k, nm in enumerate(STATS):
        mean = np.average(LL[:, :, k, :], axis=-1, weights=ng)
        bh, bl = np.unravel_index(int(mean.argmax()), mean.shape)
        nested = []
        for f in range(len(SEASONS)):
            m = np.delete(np.arange(len(SEASONS)), f)
            mm = np.average(LL[:, :, k, m], axis=-1, weights=ng[m])
            c = np.unravel_index(int(mm.argmax()), mm.shape)
            nested.append(float(LL[c[0], c[1], k, f] - LL[ci[0], ci[1], k, f]))
        out[nm] = {
            "best_h_scale": float(hs[bh]) if np.isfinite(hs[bh]) else "inf",
            "best_lam": float(lams[bl]),
            "interior": bool(0 < bh < len(hs) - 1),
            "ll_best": float(mean[bh, bl]),
            "ll_current": float(mean[ci[0], ci[1]]),
            "ll_inf_nohome": float(mean[inf_i, nolam]),
            "ll_inf_best_lam": float(mean[inf_i].max()),
            "fold_wins_best_vs_current": int((LL[bh, bl, k] > LL[ci[0], ci[1], k]).sum()),
            "fold_wins_best_vs_inf": int((LL[bh, bl, k] > LL[inf_i, nolam, k]).sum()),
            "diff_best_vs_inf_nohome": float(mean[bh, bl] - mean[inf_i, nolam]),
            "nested_wins_vs_current": int(sum(d > 0 for d in nested)),
            "ll_by_h_lam1": mean[:, nolam].tolist(),
            "ll_by_h_lamhard": mean[:, -1].tolist(),
            "ll_mean": mean.tolist(),
        }
    return out


def peplay(sim, trans, off, dfn, season, cls, sds, hs, lams):
    folds = np.zeros((len(SEASONS), len(hs), len(lams)))
    ns_ = []
    for f, s in enumerate(SEASONS):
        folds[f], n = fold_eval(sim, trans, off, dfn, season, sds[f], cls, s, hs, lams)
        ns_.append(n)
    mean = np.average(folds, axis=0, weights=ns_)
    bh, bl = np.unravel_index(int(mean.argmax()), mean.shape)
    inf_i = hs.index(np.inf)
    return {
        "best_h_scale": float(hs[bh]) if np.isfinite(hs[bh]) else "inf",
        "best_lam": float(lams[bl]),
        "ll_best": float(mean[bh, bl]),
        "ll_inf_nohome": float(mean[inf_i, lams.index(1.0)]),
        "ll_by_h_lam1": mean[:, lams.index(1.0)].tolist(),
    }


def fit_game(modes=("both", "off", "def"), gammas=(0.0, 0.1, 0.25)):
    sim, trans, ratings, eo, ef = build_frames()
    pbp = sim.load_reg_seasons(SEASONS)
    season = trans["game_id"].str[:4].astype(int).to_numpy()
    Y, pos, dteam = game_stats(pbp, trans)
    home = trans["is_home_off"].to_numpy().astype(int)
    gcode = pd.factorize(trans["game_id"].astype(str) + "_" + pd.Series(home).astype(str))[0]
    hs, lams = list(GH_GRID), list(GLAM_GRID)
    off, dfn = trans["off_row"].to_numpy(), trans["def_row"].to_numpy()
    sds = []
    for s in SEASONS:
        tr_g = ratings[ratings["season"] != s]
        sds.append(float(np.nanstd(pd.concat([tr_g["home_off_epa_per_play"], tr_g["away_off_epa_per_play"]]).to_numpy())))
    res = {"grid_h": [float(x) if np.isfinite(x) else "inf" for x in hs], "grid_lam": lams, "stats": list(STATS), "sd_by_fold": sds, "real": {}, "control": {}}
    if FIT_GAME.exists():
        res.update(json.loads(FIT_GAME.read_text(encoding="utf-8")))
    KDIR.mkdir(parents=True, exist_ok=True)
    for mode in modes:
        ls = lams if mode == "both" else list(GLAM_SHORT)
        LL, ng = game_scan(sim, trans, off, dfn, season, Y, gcode, mode, hs, ls, sds, "real")
        res["real"][mode] = summarize(LL, ng, hs, ls)
        res["real"][mode]["ng"] = ng.tolist()
        if mode == "both":
            res["real"]["peplay"] = peplay(sim, trans, off, dfn, season, outcome_codes(trans), sds, hs, ls)
            primary = res["real"]["both"]["epa"]
            res["h_scale"] = primary["best_h_scale"]
            res["lam"] = primary["best_lam"]
        FIT_GAME.write_text(json.dumps(res, indent=1), encoding="utf-8")
    cl = list(GLAM_SHORT)
    for g in gammas:
        fo, fd, Y2, dy = make_control(trans, season, pos, dteam, Y, g, 7)
        sdc = [float(np.std(fo[season != s])) for s in SEASONS]
        LL, ng = game_scan(sim, trans, fo, fd, season, Y2, gcode, "both", hs, cl, sdc, "ctl%g" % g)
        r = summarize(LL, ng, hs, cl)
        t2 = trans.copy()
        rng = np.random.default_rng(11)
        t2["yards_gained"] = t2["yards_gained"].to_numpy(float) + np.floor(dy + rng.random(len(t2)))
        r["peplay"] = peplay(sim, t2, fo, fd, season, outcome_codes(t2), sdc, hs, cl)
        res["control"]["gamma_%g" % g] = r
        FIT_GAME.write_text(json.dumps(res, indent=1), encoding="utf-8")
        print(g, json.dumps({k: {x: v[x] for x in ("best_h_scale", "best_lam", "diff_best_vs_inf_nohome", "fold_wins_best_vs_inf")} for k, v in r.items() if k in STATS}), flush=True)


class DN:
    def __init__(self, trans, Y, pos, dteam):
        self.season = trans["game_id"].str[:4].astype(int).to_numpy()
        gid = pd.Series(trans["game_id"].astype(str).to_numpy())
        epa = Y[:, 0]
        self.fin = np.isfinite(epa)
        self.e = np.where(self.fin, epa, 0.0)
        self.gid = gid.to_numpy()
        self.side = {}
        for nm, team in (("off", pos), ("def", dteam)):
            tm = pd.Series(team)
            gkey = pd.factorize(gid + "_" + tm)[0]
            tsk = pd.factorize(pd.Series(self.season.astype(str)) + "_" + tm)[0]
            ng = int(gkey.max()) + 1
            S = np.bincount(gkey, weights=self.e, minlength=ng)
            N = np.bincount(gkey, weights=self.fin.astype(float), minlength=ng)
            gts = np.zeros(ng, dtype=int)
            gts[gkey] = tsk
            gse = np.zeros(ng, dtype=int)
            gse[gkey] = self.season
            first = np.zeros(ng, dtype=int)
            first[gkey[::-1]] = np.arange(len(gkey))[::-1]
            self.side[nm] = {"gkey": gkey, "S": S, "N": N, "gts": gts, "gse": gse, "nts": int(gts.max()) + 1, "first": first}

    def comps(self, nm, train):
        d = self.side[nm]
        tr = np.isin(d["gse"], list(train)) & (d["N"] > 0)
        y = d["S"] / np.maximum(d["N"], 1.0)
        G = np.bincount(d["gts"][tr], minlength=d["nts"]).astype(float)
        ysum = np.bincount(d["gts"][tr], weights=y[tr], minlength=d["nts"])
        ok = G >= 2
        ybar = np.where(G > 0, ysum / np.maximum(G, 1.0), 0.0)
        ss = np.bincount(d["gts"][tr], weights=(y[tr] - ybar[d["gts"][tr]]) ** 2, minlength=d["nts"])
        v_e = float(ss[ok].sum() / (G[ok] - 1.0).sum())
        tau2 = max(float(np.var(ybar[ok], ddof=1)) - v_e * float(np.mean(1.0 / G[ok])), 1e-9)
        mu = float(d["S"][tr].sum() / d["N"][tr].sum())
        return {"v_e": v_e, "tau2": tau2, "mu": mu, "nbar": float(d["N"][tr].mean())}

    def game_labels(self, nm, train):
        d = self.side[nm]
        c = self.comps(nm, train)
        S_ts = np.bincount(d["gts"], weights=d["S"], minlength=d["nts"])
        N_ts = np.bincount(d["gts"], weights=d["N"], minlength=d["nts"])
        So = S_ts[d["gts"]] - d["S"]
        No = N_ts[d["gts"]] - d["N"]
        l1 = np.where(No > 0, So / np.maximum(No, 1.0), c["mu"])
        m = No / c["nbar"]
        r = c["tau2"] / (c["tau2"] + c["v_e"] / np.maximum(m, 1e-9))
        l2 = c["mu"] + r * (l1 - c["mu"])
        return l1, l2, r, c

    def rows(self, train, kind):
        out = []
        for nm in ("off", "def"):
            l1, l2, _, _ = self.game_labels(nm, train)
            out.append((l1 if kind == "l1" else l2)[self.side[nm]["gkey"]])
        return out[0], out[1]

    def train_sd(self, train, kind):
        d = self.side["off"]
        l1, l2, _, _ = self.game_labels("off", train)
        tr = np.isin(d["gse"], list(train)) & (d["N"] > 0)
        return float(np.std((l1 if kind == "l1" else l2)[tr]))

    def loso_game_labels(self, nm):
        d = self.side[nm]
        l1o = np.zeros(len(d["S"]))
        l2o = np.zeros(len(d["S"]))
        for s in SEASONS:
            l1, l2, _, _ = self.game_labels(nm, [x for x in SEASONS if x != s])
            m = d["gse"] == s
            l1o[m] = l1[m]
            l2o[m] = l2[m]
        return l1o, l2o


def wls(X, y, w):
    sw = np.sqrt(w)
    return np.linalg.lstsq(X * sw[:, None], y * sw, rcond=None)[0]


def slopes(boot=300, seed=5):
    sim, trans, ratings, eo, ef = build_frames()
    pbp = sim.load_reg_seasons(SEASONS)
    Y, pos, dteam = game_stats(pbp, trans)
    dn = DN(trans, Y, pos, dteam)
    d = dn.side["off"]
    first = d["first"]
    rows = {}
    rows["rolling"] = (trans["off_row"].to_numpy()[first], trans["def_row"].to_numpy()[first])
    l1o, l2o = dn.loso_game_labels("off")
    l1d, l2d = dn.loso_game_labels("def")
    dgk = dn.side["def"]["gkey"][first]
    rows["lgo_season_mean"] = (l1o, l1d[dgk])
    rows["eb_shrunk"] = (l2o, l2d[dgk])
    y = d["S"] / np.maximum(d["N"], 1.0)
    w = d["N"]
    keep = w >= MIN_PLAYS
    gids = dn.gid[first]
    ug, gi = np.unique(gids, return_inverse=True)
    rng = np.random.default_rng(seed)
    cnts = [np.bincount(rng.integers(0, len(ug), len(ug)), minlength=len(ug)) for _ in range(boot)]
    out = {"n_team_games": int(keep.sum()), "boot": boot, "labels": {}}
    for nm, (xo, xd) in rows.items():
        ok = keep & np.isfinite(xo) & np.isfinite(xd)
        res = {"sd_off": float(np.std(xo[ok])), "mean_off": float(np.mean(xo[ok])), "sd_def": float(np.std(xd[ok])), "mean_def": float(np.mean(xd[ok]))}
        for tag, cols in (("off_only", [xo]), ("def_only", [xd]), ("both", [xo, xd])):
            X = np.column_stack([np.ones(len(y))] + cols)
            b = wls(X[ok], y[ok], w[ok])
            bs = []
            for c in cnts:
                ww = w * c[gi]
                m = ok & (ww > 0)
                bs.append(wls(X[m], y[m], ww[m]))
            bs = np.array(bs)
            res[tag] = {"slope": [float(v) for v in b[1:]], "lo": [float(v) for v in np.percentile(bs[:, 1:], 2.5, axis=0)], "hi": [float(v) for v in np.percentile(bs[:, 1:], 97.5, axis=0)], "p_pos": [float(v) for v in (bs[:, 1:] > 0).mean(axis=0)]}
        out["labels"][nm] = res
    comps = {}
    for nm in ("off", "def"):
        l1, l2, r, c = dn.game_labels(nm, SEASONS)
        comps[nm] = dict(c, mean_reliability=float(np.average(r, weights=dn.side[nm]["N"])), sd_team_game_y=float(np.std(dn.side[nm]["S"] / np.maximum(dn.side[nm]["N"], 1.0))))
    out["components_all_seasons"] = comps
    import mod25_generator as gen

    f = gen.load_fit()
    lo, ld = gen.league_means()
    out["latent"] = {"off_sd": float(np.sqrt(f["off"]["var_mu"])), "def_sd": float(np.sqrt(f["def"]["var_mu"])), "off_mean": lo, "def_mean": ld, "off_phi": f["off"]["phi"], "def_phi": f["def"]["phi"]}
    KDIR.mkdir(parents=True, exist_ok=True)
    (KDIR / "slopes_dn.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps(out, indent=1))


def fit_dn(kinds=("l2", "l1")):
    sim, trans, ratings, eo, ef = build_frames()
    pbp = sim.load_reg_seasons(SEASONS)
    season = trans["game_id"].str[:4].astype(int).to_numpy()
    Y, pos, dteam = game_stats(pbp, trans)
    dn = DN(trans, Y, pos, dteam)
    home = trans["is_home_off"].to_numpy().astype(int)
    gcode = pd.factorize(trans["game_id"].astype(str) + "_" + pd.Series(home).astype(str))[0]
    hs, lams = list(GH_GRID), list(GLAM_GRID)
    res = {"grid_h": [float(x) if np.isfinite(x) else "inf" for x in hs], "grid_lam": lams, "stats": list(STATS), "real": {}}
    if FIT_DN.exists():
        res.update(json.loads(FIT_DN.read_text(encoding="utf-8")))
    for kind in kinds:
        def lab(s, kind=kind):
            train = [x for x in SEASONS if x != s]
            o, d_ = dn.rows(train, kind)
            return o, d_, dn.train_sd(train, kind)

        LL, ng = game_scan(sim, trans, None, None, season, Y, gcode, "both", hs, lams, [0.0] * len(SEASONS), kind, lab)
        r = summarize(LL, ng, hs, lams)
        r["ng"] = ng.tolist()
        r["sd_by_fold"] = [dn.train_sd([x for x in SEASONS if x != s], kind) for s in SEASONS]
        res["real"][kind] = r
        if kind == "l2":
            res["h_scale"] = r["epa"]["best_h_scale"]
            res["lam"] = r["epa"]["best_lam"]
            res["label"] = "eb_shrunk_leave_game_out"
        FIT_DN.write_text(json.dumps(res, indent=1), encoding="utf-8")
        print(kind, json.dumps({k: {x: r[k][x] for x in ("best_h_scale", "best_lam", "interior", "ll_best", "ll_inf_nohome", "diff_best_vs_inf_nohome", "fold_wins_best_vs_inf", "nested_wins_vs_current")} for k in STATS}), flush=True)


def install_dn():
    import mod25d_variance as dv
    import sim04_engine as sim

    spec = json.loads(FIT_DN.read_text(encoding="utf-8"))
    t = dv._G["tables"]
    a = t["arrays"]
    old_snap = sim.PBP_SNAPSHOT_DIR
    sim.PBP_SNAPSHOT_DIR = dv.m25.SNAP
    try:
        pool = tuple(t["seasons"])
        pbp = sim.load_reg_seasons(pool)
        ratings = sim.load_team_ratings()
        trans = sim.build_transition_frame(pbp, team_ratings=ratings).reset_index(drop=True)
    finally:
        sim.PBP_SNAPSHOT_DIR = old_snap
    assert len(trans) == len(a["off_row"]), (pool, len(trans), len(a["off_row"]))
    assert np.array_equal(trans["is_home_off"].to_numpy(), a["is_home_off"])
    assert np.corrcoef(trans["off_row"].to_numpy(), a["off_row"])[0, 1] > 0.3
    Y, pos, dteam = game_stats(pbp, trans)
    dn = DN(trans, Y, pos, dteam)
    o, d_ = dn.rows(pool, "l2")
    assert np.isfinite(o).all() and np.isfinite(d_).all()
    a["off_row"] = o.astype(np.float64)
    a["def_row"] = d_.astype(np.float64)
    scale = spec["h_scale"]
    lam = float(spec["lam"])
    t["team_kernel_h"] = float("inf") if scale == "inf" else float(scale) * dn.train_sd(pool, "l2")
    sim.TEAM_KERNEL_LAMBDA = lam
    dv._G["ns"]["TEAM_KERNEL_LAMBDA"] = lam
    t["nn_weight_cache_cond"].clear()


def install_blup():
    import mod25d_variance as dv
    import sim04_engine as sim

    spec = json.loads(FIT_BLUP.read_text(encoding="utf-8"))
    t = dv._G["tables"]
    a = t["arrays"]
    scale = spec["h_scale"]
    lam = float(spec["lam"])
    t["team_kernel_h"] = float("inf") if scale == "inf" else float(scale) * float(np.std(a["off_row"]))
    sim.TEAM_KERNEL_LAMBDA = lam
    dv._G["ns"]["TEAM_KERNEL_LAMBDA"] = lam
    t["nn_weight_cache_cond"].clear()


def pass_through(rows, y, w, keep, cnts, gi):
    out = {}
    for nm, (xo, xd) in rows.items():
        ok = keep & np.isfinite(xo) & np.isfinite(xd)
        res = {"sd_off": float(np.std(xo[ok])), "sd_def": float(np.std(xd[ok]))}
        for tag, cols in (("off_only", [xo]), ("def_only", [xd]), ("both", [xo, xd])):
            X = np.column_stack([np.ones(len(y))] + cols)
            b = wls(X[ok], y[ok], w[ok])
            bs = []
            for c in cnts:
                ww = w * c[gi]
                m = ok & (ww > 0)
                bs.append(wls(X[m], y[m], ww[m]))
            bs = np.array(bs)
            res[tag] = {"slope": [float(v) for v in b[1:]], "lo": [float(v) for v in np.percentile(bs[:, 1:], 2.5, axis=0)], "hi": [float(v) for v in np.percentile(bs[:, 1:], 97.5, axis=0)]}
        out[nm] = res
    return out


def fit_blup(boot=200, kinds=("blup", "l2")):
    import mod25d_variance as dv

    sim, trans, ratings, eo, ef = build_frames()
    pbp = sim.load_reg_seasons(SEASONS)
    plain = sim.build_transition_frame(pbp).reset_index(drop=True)
    season = trans["game_id"].str[:4].astype(int).to_numpy()
    Y, pos, dteam = game_stats(pbp, trans)
    dn = DN(trans, Y, pos, dteam)
    home = trans["is_home_off"].to_numpy().astype(int)
    gcode = pd.factorize(trans["game_id"].astype(str) + "_" + pd.Series(home).astype(str))[0]
    res = {"engine_labels": {"sd_eo": float(np.std(eo)), "sd_ef": float(np.std(ef)), "engine_h_epa": float(sim.TEAM_KERNEL_H_SCALE * np.std(eo)), "h_scale_engine": float(sim.TEAM_KERNEL_H_SCALE), "sd_rolling_off": float(np.nanstd(trans["off_row"])), "sd_rolling_def": float(np.nanstd(trans["def_row"]))}}
    d = dn.side["off"]
    first = d["first"]
    l1o, l2o = dn.loso_game_labels("off")
    l1d, l2d = dn.loso_game_labels("def")
    dgk = dn.side["def"]["gkey"][first]
    mu_o = dn.comps("off", SEASONS)["mu"]
    mu_d = dn.comps("def", SEASONS)["mu"]
    eo_g = eo[first] - np.mean(eo)
    ef_g = ef[first] - np.mean(ef)
    co = float(np.polyfit(l1o - mu_o, eo_g, 1)[0])
    cd = float(np.polyfit(l1d[dgk] - mu_d, ef_g, 1)[0])
    rows = {
        "blup_insample": (mu_o + eo_g, mu_d + ef_g),
        "blup_lgo_approx": (mu_o + co * (l1o - mu_o), mu_d + cd * (l1d[dgk] - mu_d)),
        "eb_lgo": (l2o, l2d[dgk]),
    }
    y = d["S"] / np.maximum(d["N"], 1.0)
    w = d["N"]
    keep = w >= MIN_PLAYS
    ug, gi = np.unique(dn.gid[first], return_inverse=True)
    rng = np.random.default_rng(5)
    cnts = [np.bincount(rng.integers(0, len(ug), len(ug)), minlength=len(ug)) for _ in range(boot)]
    res["pass_through"] = pass_through(rows, y, w, keep, cnts, gi)
    res["shrink_slopes_approx"] = [co, cd]
    BLUP_OUT.parent.mkdir(parents=True, exist_ok=True)
    BLUP_OUT.write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(json.dumps(res, indent=1), flush=True)
    hs = sorted(set(GH_GRID) | {0.5})
    lams = list(GLAM_GRID)
    res["grid_h"] = [float(x) if np.isfinite(x) else "inf" for x in hs]
    res["grid_lam"] = lams
    res["real"] = {}
    for kind in kinds:
        if kind == "blup":
            def lab(s):
                train = [x for x in SEASONS if x != s]
                e1, f1 = dv.blup_rows(pbp, plain, train)
                m = season != s
                e1 = e1 - e1[m].mean()
                f1 = f1 - f1[m].mean()
                od, dd = dn.rows(train, "l2")
                off = np.where(m, dn.comps("off", train)["mu"] + e1, od)
                dfn = np.where(m, dn.comps("def", train)["mu"] + f1, dd)
                return off, dfn, float(np.std(e1[m]))
        else:
            def lab(s, kind=kind):
                train = [x for x in SEASONS if x != s]
                o, d_ = dn.rows(train, kind)
                return o, d_, dn.train_sd(train, kind)

        LL, ng = game_scan(sim, trans, None, None, season, Y, gcode, "both", hs, lams, [0.0] * len(SEASONS), kind, lab)
        r = summarize(LL, ng, hs, lams)
        r["ng"] = ng.tolist()
        res["real"][kind] = r
        if kind == "blup":
            res["h_scale"] = r["epa"]["best_h_scale"]
            res["lam"] = r["epa"]["best_lam"]
            res["label"] = "engine_blup_pool_eb_query"
        BLUP_OUT.write_text(json.dumps(res, indent=1), encoding="utf-8")
        print(kind, json.dumps({k: {x: r[k][x] for x in ("best_h_scale", "best_lam", "interior", "ll_best", "ll_current", "ll_inf_nohome", "diff_best_vs_inf_nohome", "fold_wins_best_vs_inf", "fold_wins_best_vs_current", "nested_wins_vs_current")} for k in STATS}), flush=True)


FIT_EIV = KDIR / "fit_eiv.json"
EIV_DIR = KDIR / "eiv"
EIV_DRAW_SEED = 5
EIV_BOOT = 300


def install_eiv():
    import mod25d_variance as dv
    import sim04_engine as sim

    spec = json.loads(FIT_EIV.read_text(encoding="utf-8"))
    t = dv._G["tables"]
    a = t["arrays"]
    scale = spec["h_scale"]
    lam = float(spec["lam"])
    t["team_kernel_h"] = float("inf") if scale == "inf" else float(scale) * float(np.std(a["off_row"]))
    sim.TEAM_KERNEL_LAMBDA = lam
    dv._G["ns"]["TEAM_KERNEL_LAMBDA"] = lam
    t["nn_weight_cache_cond"].clear()


def eiv_wmean(lw, valid, y, y2=None):
    lw = np.where(valid, lw, -np.inf)
    w = np.exp(lw - lw.max(axis=1, keepdims=True))
    ws = w.sum(axis=1)
    m = (w * y).sum(axis=1) / ws
    if y2 is None:
        return m
    return m, np.maximum((w * y2).sum(axis=1) / ws - m * m, 0.0)


def eiv_run_fold(fold):
    import mod25d_variance as dv

    sim, trans, ratings, eo, ef = build_frames()
    pbp = sim.load_reg_seasons(SEASONS)
    plain = sim.build_transition_frame(pbp).reset_index(drop=True)
    season = trans["game_id"].str[:4].astype(int).to_numpy()
    Y, pos, dteam = game_stats(pbp, trans)
    Yn = Y[:, 0]
    dn = DN(trans, Y, pos, dteam)
    home = trans["is_home_off"].to_numpy().astype(int)
    gcode = pd.factorize(trans["game_id"].astype(str) + "_" + pd.Series(home).astype(str))[0]
    hs, lams = sorted(set(GH_GRID) | {0.5}), list(GLAM_GRID)
    train = [x for x in SEASONS if x != fold]
    e1, f1 = dv.blup_rows(pbp, plain, train)
    m = season != fold
    e1 = e1 - e1[m].mean()
    f1 = f1 - f1[m].mean()
    od, dd = dn.rows(train, "l2")
    off = np.where(m, dn.comps("off", train)["mu"] + e1, od)
    dfn = np.where(m, dn.comps("def", train)["mu"] + f1, dd)
    sd = float(np.std(e1[m]))
    post = []
    for nm in ("off", "def"):
        l1, l2, r, c = dn.game_labels(nm, train)
        post.append(np.sqrt(c["tau2"] * (1.0 - r))[dn.side[nm]["gkey"]])
    so_row, sd_row = post
    rng = np.random.default_rng(EIV_DRAW_SEED + int(fold))
    zo = rng.standard_normal(dn.side["off"]["N"].shape[0])[dn.side["off"]["gkey"]]
    zd = rng.standard_normal(dn.side["def"]["N"].shape[0])[dn.side["def"]["gkey"]]
    tr = season != fold
    te = np.flatnonzero(~tr)
    te = te[np.isfinite(Yn[te])]
    feats = sim.feature_matrix(
        trans["dist_raw"].to_numpy(), trans["fp_raw"].to_numpy(), trans["sc_raw"].to_numpy(),
        trans["time_raw"].to_numpy(), trans["off_to_raw"].to_numpy(), trans["def_to_raw"].to_numpy(),
        trans["phase"].to_numpy(),
    )
    down = trans["down_i"].to_numpy()
    phase = trans["phase"].to_numpy()
    homeb = trans["is_home_off"].to_numpy()
    ok = np.isfinite(Yn)
    trees = {}
    for dnn in (1, 2, 3, 4):
        for ph in range(5):
            idx = np.flatnonzero(tr & ok & (down == dnn) & sim.phase_pool_mask(phase, ph))
            if len(idx):
                trees[(dnn, ph)] = (cKDTree(feats[idx]), idx)
    K = sim.K_STATE
    n = len(te)
    NB = np.zeros((n, K), dtype=np.int64)
    valid = np.zeros((n, K), dtype=bool)
    groups = {}
    for j, r in enumerate(te):
        key = (int(down[r]), int(phase[r]))
        if key not in trees:
            key = (int(down[r]), 0)
        groups.setdefault(key, []).append(j)
    for key, js in groups.items():
        tree, idx = trees[key]
        k = min(K, len(idx))
        _, ind = tree.query(feats[te[js]], k=k)
        NB[js, :k] = idx[np.asarray(ind).reshape(len(js), k)]
        valid[js, :k] = True
    Po = off[NB]
    Pd = dfn[NB]
    lh = (homeb[NB] == homeb[te][:, None]).astype(float)
    Ys = np.where(valid, Yn[NB], 0.0)
    Ys2 = Ys * Ys
    qo, qd = off[te], dfn[te]
    Do = (Po - qo[:, None]) ** 2
    Dd = (Pd - qd[:, None]) ** 2
    so2 = so_row[te] ** 2
    sd2 = sd_row[te] ** 2
    ug, first, gi = np.unique(gcode[te], return_index=True, return_inverse=True)
    cnt = np.bincount(gi).astype(float)
    keep = cnt >= MIN_PLAYS
    R = np.bincount(gi, weights=Yn[te]) / cnt
    draws = []
    Dq = []
    for sgn in (1.0, -1.0):
        qo_k = qo + sgn * so_row[te] * zo[te]
        qd_k = qd + sgn * sd_row[te] * zd[te]
        draws.append((qo_k[first], qd_k[first]))
        Dq.append((Po - qo_k[:, None]) ** 2 + (Pd - qd_k[:, None]) ** 2)
    H, L, G = len(hs), len(lams), len(ug)
    Mn = np.zeros((H, L, G))
    Vn = np.zeros((H, L, G))
    Me = np.zeros((H, L, G))
    Ve = np.zeros((H, L, G))
    Mq = np.zeros((H, L, 2, G))
    for a, hsc in enumerate(hs):
        h2 = (hsc * sd) ** 2
        if np.isinf(hsc):
            base_n = np.zeros_like(Do)
            base_e = np.zeros_like(Do)
            base_q = [np.zeros_like(Do)] * 2
        else:
            base_n = -(Do + Dd) / (2.0 * h2)
            base_e = -(Do / (2.0 * (h2 + so2))[:, None] + Dd / (2.0 * (h2 + sd2))[:, None])
            base_q = [-d / (2.0 * h2) for d in Dq]
        for b, lam in enumerate(lams):
            ll = lh * np.log(lam)
            mn, vn = eiv_wmean(base_n + ll, valid, Ys, Ys2)
            me, ve = eiv_wmean(base_e + ll, valid, Ys, Ys2)
            Mn[a, b] = np.bincount(gi, weights=mn) / cnt
            Vn[a, b] = np.bincount(gi, weights=vn) / cnt**2
            Me[a, b] = np.bincount(gi, weights=me) / cnt
            Ve[a, b] = np.bincount(gi, weights=ve) / cnt**2
            for k in range(2):
                Mq[a, b, k] = np.bincount(gi, weights=eiv_wmean(base_q[k] + ll, valid, Ys)) / cnt
        print("eiv", fold, a, flush=True)
    EIV_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(
        EIV_DIR / ("fold_%d.npz" % fold),
        hs=np.array([h if np.isfinite(h) else 1e9 for h in hs]), lams=np.array(lams), keep=keep, cnt=cnt, R=R,
        Mn=Mn[..., keep], Vn=Vn[..., keep], Me=Me[..., keep], Ve=Ve[..., keep], Mq=Mq[..., keep],
        qo=np.array([d[0] for d in draws])[:, keep], qd=np.array([d[1] for d in draws])[:, keep],
        so=so_row[te][first][keep], sdv=sd_row[te][first][keep], sdscale=sd, ncnt=cnt[keep],
    )


def eiv_ols(Y, X, w):
    A = np.linalg.inv((X * w[:, None]).T @ X)
    return ((Y * w[None, :]) @ X) @ A


def eiv_fit():
    parts = [np.load(EIV_DIR / ("fold_%d.npz" % s)) for s in SEASONS]
    hs = parts[0]["hs"]
    hsl = [float(h) if h < 1e8 else np.inf for h in hs]
    lams = [float(x) for x in parts[0]["lams"]]
    H, L = len(hs), len(lams)
    fid = np.concatenate([np.full(len(p["R"][p["keep"]]), f) for f, p in enumerate(parts)])
    cat = lambda key: np.concatenate([p[key] for p in parts], axis=-1)
    Mn, Vn, Me, Ve, Mq = cat("Mn"), cat("Vn"), cat("Me"), cat("Ve"), cat("Mq")
    R = np.concatenate([p["R"][p["keep"]] for p in parts])
    cnt = np.concatenate([p["ncnt"] for p in parts])
    qo = cat("qo")
    qd = cat("qd")
    ng = np.bincount(fid, minlength=len(SEASONS))
    out = {"grid_h": [float(x) if np.isfinite(x) else "inf" for x in hsl], "grid_lam": lams, "ng": ng.tolist(), "n_games": int(len(R))}
    summ = {}
    for nm, M, V in (("naive", Mn, Vn), ("eiv", Me, Ve)):
        LL = np.zeros((H, L, 3, len(SEASONS)))
        for a in range(H):
            for b in range(L):
                for f in range(len(SEASONS)):
                    LL[a, b, 0, f] = tau_ll(M[a, b], V[a, b], R, fid != f, fid == f)[0]
        LL[:, :, 1] = LL[:, :, 0]
        LL[:, :, 2] = LL[:, :, 0]
        summ[nm] = summarize(LL, ng, hsl, lams)["epa"]
        summ[nm]["LL_folds"] = LL[:, :, 0, :].tolist()
    out["naive"] = summ["naive"]
    out["eiv_a"] = summ["eiv"]
    X = np.column_stack([np.ones(2 * len(R)), np.concatenate([qo[0], qo[1]]), np.concatenate([qd[0], qd[1]])])
    w2 = np.concatenate([cnt, cnt])
    Ymat = np.concatenate([Mq[:, :, 0, :], Mq[:, :, 1, :]], axis=-1).reshape(H * L, -1)
    beta = eiv_ols(Ymat, X, w2)
    slope = beta[:, 1:].reshape(H, L, 2)
    out["pass_through_grid"] = slope.tolist()
    spec = json.loads(FIT_BLUP.read_text(encoding="utf-8"))["pass_through"]["eb_lgo"]["both"]
    tgt = np.array(spec["slope"])
    out["real_pass_through"] = {"off": [spec["slope"][0], spec["lo"][0], spec["hi"][0]], "def": [spec["slope"][1], spec["lo"][1], spec["hi"][1]], "label": "eb_lgo both, fit_blup.json"}
    ci = int(np.argmin(np.abs(np.array([h if np.isfinite(h) else 1e9 for h in hsl]) - CUR[0])))
    cur = (ci, L - 1)
    ea = summ["eiv"]
    ba = (hsl.index(float(ea["best_h_scale"])) if ea["best_h_scale"] != "inf" else hsl.index(np.inf), lams.index(ea["best_lam"]))
    na = summ["naive"]
    bn = (hsl.index(float(na["best_h_scale"])) if na["best_h_scale"] != "inf" else hsl.index(np.inf), lams.index(na["best_lam"]))
    dist = ((slope - tgt[None, None, :]) ** 2).sum(axis=-1)
    dist[hsl.index(np.inf)] = np.inf
    bb = tuple(int(i) for i in np.unravel_index(int(np.argmin(dist)), dist.shape))
    rng = np.random.default_rng(EIV_DRAW_SEED)
    boots = [np.bincount(rng.integers(0, len(R), len(R)), minlength=len(R)).astype(float) for _ in range(EIV_BOOT)]
    mean_eiv = np.average(np.array(summ["eiv"]["LL_folds"]), axis=-1, weights=ng)

    def point(name, idx):
        a, b = idx
        y = Ymat[a * L + b][None, :]
        bs = []
        for c in boots:
            ww = np.concatenate([cnt * c, cnt * c])
            m = ww > 0
            bs.append(eiv_ols(y[:, m], X[m], ww[m])[0, 1:])
        bs = np.array(bs)
        fl = np.array(summ["eiv"]["LL_folds"])
        fn = np.array(summ["naive"]["LL_folds"])
        return {
            "h_scale": hsl[a] if np.isfinite(hsl[a]) else "inf", "lam": lams[b],
            "interior": bool(0 < a < H - 1),
            "pass_through": [float(slope[a, b, 0]), float(slope[a, b, 1])],
            "pass_through_lo": [float(v) for v in np.percentile(bs, 2.5, axis=0)],
            "pass_through_hi": [float(v) for v in np.percentile(bs, 97.5, axis=0)],
            "ll_eiv": float(mean_eiv[a, b]),
            "ll_naive_same_point": float(np.average(fn[a, b], weights=ng)),
            "fold_wins_vs_current_eiv": int((fl[a, b] > fl[cur[0], cur[1]]).sum()),
            "fold_wins_vs_current_scoring_naive": int((fn[a, b] > fn[cur[0], cur[1]]).sum()),
        }

    out["points"] = {"current": point("current", cur), "naive_best": point("naive_best", bn), "a_eiv_best": point("a", ba), "b_pass_through_1": point("b", bb)}
    out["h_scale"] = out["points"]["a_eiv_best"]["h_scale"]
    out["lam"] = out["points"]["a_eiv_best"]["lam"]
    out["option_b"] = {"h_scale": out["points"]["b_pass_through_1"]["h_scale"], "lam": out["points"]["b_pass_through_1"]["lam"]}
    out["label"] = "engine_blup_pool_posterior_integrated_query"
    FIT_EIV.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps(out["points"], indent=1), flush=True)
    print(json.dumps({"naive": {k: na[k] for k in ("best_h_scale", "best_lam", "interior", "ll_best", "ll_current")}, "eiv": {k: ea[k] for k in ("best_h_scale", "best_lam", "interior", "ll_best", "ll_current", "fold_wins_best_vs_current", "nested_wins_vs_current")}, "real_pass_through": out["real_pass_through"]}, indent=1), flush=True)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "eiv":
        if len(sys.argv) > 2 and sys.argv[2] == "fit":
            eiv_fit()
        else:
            for s in sys.argv[2].split(","):
                eiv_run_fold(int(s))
    elif len(sys.argv) > 1 and sys.argv[1] == "slopes":
        slopes()
    elif len(sys.argv) > 1 and sys.argv[1] == "dn":
        fit_dn(tuple(sys.argv[2].split(",")) if len(sys.argv) > 2 else ("l2", "l1"))
    elif len(sys.argv) > 1 and sys.argv[1] == "blup":
        fit_blup()
    elif len(sys.argv) > 1 and sys.argv[1] == "game":
        part = sys.argv[2] if len(sys.argv) > 2 else "all"
        fit_game(
            modes={"all": ("both", "off", "def"), "real": ("both",), "axes": ("off", "def")}.get(part, ()),
            gammas=tuple(float(x) for x in sys.argv[3].split(",")) if part == "ctl" else (),
        )
    else:
        fit()
