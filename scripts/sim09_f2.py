import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

import sim09_u4g as u4g  # noqa: E402

dv = u4g.dv
OUT = REPO / "artifacts" / "sim09" / "f2"
NV = dv.NV_DIR
FIT = tuple(range(2009, 2018))
HELD = tuple(range(2018, 2026))
EXTRA = ["game_id", "play_id", "penalty", "qb_kneel", "qb_spike", "posteam", "defteam", "home_team"]


def real_frame(seasons):
    import glob

    dv.sim.PBP_SNAPSHOT_DIR = dv.m25.SNAP
    pbp = dv.sim.load_reg_seasons(tuple(seasons))
    tr = dv.sim.build_transition_frame(pbp)
    ex = pd.concat([pd.read_parquet(f"{dv.m25.SNAP}/season={y}/plays.parquet", columns=EXTRA) for y in seasons]).drop_duplicates(["game_id", "play_id"])
    d = pd.DataFrame({
        "game_id": tr["game_id"].to_numpy(), "play_id": tr["play_id"].to_numpy(), "down": tr["down_i"].to_numpy(),
        "dist": tr["dist_raw"].to_numpy(), "yl": tr["fp_raw"].to_numpy(), "sd": tr["sc_raw"].to_numpy(),
        "gsr": tr["gsr_actual"].to_numpy(), "qtr": tr["qtr_actual"].to_numpy(), "code": tr["play_type_code"].to_numpy(),
        "flip": tr["possession_flip"].to_numpy().astype(bool), "po": tr["points_off"].to_numpy(), "pdf": tr["points_def"].to_numpy(),
        "el": tr["clock_elapsed"].to_numpy(), "yards": tr["yards_gained"].to_numpy(), "oto": tr["off_to_raw"].to_numpy(), "dto": tr["def_to_raw"].to_numpy(),
        "otu": tr["off_to_used"].to_numpy(), "dtu": tr["def_to_used"].to_numpy(),
    })
    d = d.merge(ex, on=["game_id", "play_id"], how="left")
    d = d.sort_values(["game_id", "play_id"], kind="stable").reset_index(drop=True)
    d["g"] = pd.factorize(d["game_id"])[0]
    d["season"] = d["game_id"].str.slice(0, 4).astype(int)
    d["stop"] = (d["penalty"].fillna(0) == 1) | ((d["code"] == 1) & (d["yards"] == 0)) | ((d["po"] + d["pdf"]) > 0) | d["flip"]
    d["next_code"] = d.groupby("g")["code"].shift(-1)
    d["next_qtr"] = d.groupby("g")["qtr"].shift(-1)
    return d


def cmd_real(args):
    OUT.mkdir(parents=True, exist_ok=True)
    for nm, ss in (("fit", FIT), ("held", HELD)):
        d = real_frame(ss)
        d.to_parquet(OUT / f"real_{nm}.parquet")
        print(nm, d.shape, "otu", float(d.otu.sum() / d.g.nunique()), "dtu", float(d.dtu.sum() / d.g.nunique()), flush=True)


def sim_frame(path):
    pool = pd.read_parquet(u4g.OUT / "pool.parquet")
    if Path(path).is_dir():
        parts = []
        for i, f in enumerate(sorted(Path(path).glob("play_*_*.parquet"))):
            _, w, sid = f.stem.split("_")
            if int(sid) < 2:
                continue
            x = pd.read_parquet(f)
            x["g"] = x["g"].astype(np.int64) + i * 100000
            parts.append(x)
        S = pd.concat(parts, ignore_index=True)
        ix = S["idx"].to_numpy().astype(int)
        S["s_otu"] = pool["otu"].to_numpy()[ix]
        S["s_dtu"] = pool["dtu"].to_numpy()[ix]
    else:
        S = pd.read_parquet(path)
    S = S.sort_values(["g"], kind="stable").reset_index(drop=True)
    idx = S["idx"].to_numpy().astype(int)
    S["penalty"] = pool["penalty"].to_numpy()[idx]
    S["stop"] = (S["penalty"].fillna(0) == 1) | ((S["code"] == 1) & (S["yards"] == 0)) | ((S["po"] + S["pdf"]) > 0) | S["flip"].astype(bool)
    S["otu"] = S["s_otu"]
    S["dtu"] = S["s_dtu"]
    S["next_code"] = S.groupby("g")["code"].shift(-1)
    S["next_qtr"] = S.groupby("g")["qtr"].shift(-1)
    oto = np.zeros(len(S))
    dto = np.zeros(len(S))
    g = S["g"].to_numpy()
    q = S["qtr"].to_numpy()
    oh = S["offhome"].to_numpy().astype(bool)
    otu = S["otu"].to_numpy()
    dtu = S["dtu"].to_numpy()
    home = away = 3
    ph = 0
    cg = None
    for i in range(len(S)):
        if g[i] != cg:
            cg = g[i]
            home = away = 3
            ph = 1 if q[i] <= 2 else 2
        h = 1 if q[i] <= 2 else 2
        if q[i] <= 4 and h != ph:
            home = away = 3
            ph = h
        if oh[i]:
            oto[i], dto[i] = home, away
            home = max(0, home - otu[i])
            away = max(0, away - dtu[i])
        else:
            oto[i], dto[i] = away, home
            away = max(0, away - otu[i])
            home = max(0, home - dtu[i])
    S["oto_sim"] = oto
    S["dto_sim"] = dto
    return S


def bucket(d):
    h = np.where(d["qtr"] <= 2, 1, 2)
    hs = np.where(d["qtr"] <= 2, d["gsr"] - 1800, d["gsr"]).clip(0, 1800)
    d = d.assign(half=h, hs=hs)
    d["tb"] = pd.cut(d["hs"], [-1, 60, 120, 240, 480, 900, 1801], labels=["0-1m", "1-2m", "2-4m", "4-8m", "8-15m", "15m+"])
    d["sb"] = pd.cut(d["sd"], [-100, -9, -1, 0, 8, 100], labels=["trail9+", "trail1-8", "tied", "lead1-8", "lead9+"], right=True)
    return d


def table(d, oto, dto):
    d = bucket(d)
    d = d.assign(oto=oto, dto=dto)
    res = {}
    n = d["g"].nunique()
    res["games"] = int(n)
    res["off_per_game"] = float(d["otu"].sum() / n)
    res["def_per_game"] = float(d["dtu"].sum() / n)
    for hv in (1, 2):
        m = d["half"] == hv
        res[f"off_h{hv}"] = float(d.loc[m, "otu"].sum() / n)
        res[f"def_h{hv}"] = float(d.loc[m, "dtu"].sum() / n)
    ph = ((d["otu"] > 0) & (d["oto"] == 0)).sum() + ((d["dtu"] > 0) & (d["dto"] == 0)).sum()
    res["phantom_uses_per_game"] = float(ph / n)
    for (hv, tb), g in d.groupby(["half", "tb"], observed=True):
        res[f"rate|h{hv}|{tb}|off"] = float((g["otu"] > 0).mean())
        res[f"rate|h{hv}|{tb}|def"] = float((g["dtu"] > 0).mean())
    for (tb, sb), g in d[d["tb"].isin(["0-1m", "1-2m", "2-4m"])].groupby(["tb", "sb"], observed=True):
        res[f"rate|{tb}|{sb}|off"] = float((g["otu"] > 0).mean())
        res[f"rate|{tb}|{sb}|def"] = float((g["dtu"] > 0).mean())
    for k in (0, 1, 2, 3):
        m = d["oto"] == k
        res[f"off_rate_left{k}"] = float((d.loc[m, "otu"] > 0).mean())
        m = d["dto"] == k
        res[f"def_rate_left{k}"] = float((d.loc[m, "dtu"] > 0).mean())
    for dn in (1, 2, 3, 4):
        m = d["down"] == dn
        res[f"rate|down{dn}|off"] = float((d.loc[m, "otu"] > 0).mean())
        res[f"rate|down{dn}|def"] = float((d.loc[m, "dtu"] > 0).mean())
    for st in (False, True):
        m = d["stop"] == st
        res[f"rate|stop{int(st)}|off"] = float((d.loc[m, "otu"] > 0).mean())
        res[f"rate|stop{int(st)}|def"] = float((d.loc[m, "dtu"] > 0).mean())
    ice = d["next_code"].isin([2, 3]) & (d["qtr"] == d["next_qtr"])
    res["def_to_before_kick_rate"] = float((d.loc[ice, "dtu"] > 0).mean())
    res["def_to_before_kick_n"] = int(ice.sum())
    res["off_to_before_kick_rate"] = float((d.loc[ice, "otu"] > 0).mean())
    res["off_to_per_game_end_h1_final2"] = float(d.loc[(d["half"] == 1) & (d["hs"] <= 120), "otu"].sum() / n)
    res["def_to_per_game_end_h1_final2"] = float(d.loc[(d["half"] == 1) & (d["hs"] <= 120), "dtu"].sum() / n)
    res["off_to_per_game_end_h2_final2"] = float(d.loc[(d["half"] == 2) & (d["hs"] <= 120), "otu"].sum() / n)
    res["def_to_per_game_end_h2_final2"] = float(d.loc[(d["half"] == 2) & (d["hs"] <= 120), "dtu"].sum() / n)
    post = (d["hs"] - d["el"]).clip(lower=0)
    for who, col in (("off", "otu"), ("def", "dtu")):
        m = (d[col] > 0) & (d["hs"] <= 300)
        res[f"clock_at_call_mean_last5|{who}"] = float(post[m].mean())
        res[f"clock_at_call_share_le30|{who}"] = float((post[m] <= 30).mean())
        res[f"clock_at_call_share_le60|{who}"] = float((post[m] <= 60).mean())
    return res


def cmd_desc(args):
    rf = pd.read_parquet(OUT / "real_fit.parquet")
    rh = pd.read_parquet(OUT / "real_held.parquet")
    S = sim_frame(args.sim)
    S = S[S["qtr"] <= 4]
    rf = rf[rf["qtr"] <= 4]
    rh = rh[rh["qtr"] <= 4]
    T = {"fit": table(rf, rf["oto"].to_numpy(), rf["dto"].to_numpy()), "held": table(rh, rh["oto"].to_numpy(), rh["dto"].to_numpy()), "sim": table(S, S["oto_sim"].to_numpy(), S["dto_sim"].to_numpy())}
    df = pd.DataFrame(T)
    pd.set_option("display.width", 200, "display.max_rows", 500)
    print(df.round(4).to_string())
    df.to_csv(OUT / "desc.csv")


def feats(down, dist, yl, sd, gsr, qtr, code, oto, dto, stop):
    qtr = np.asarray(qtr)
    gsr = np.asarray(gsr, dtype=float)
    ph = np.where(qtr <= 2, 0, np.where(qtr <= 4, 1, 2))
    hs = np.where(qtr <= 2, np.clip(gsr - 1800.0, 0, 1800), gsr)
    return np.column_stack([ph, hs, np.clip(sd, -24, 24), oto, dto, down, np.clip(dist, 0, 25), yl, code, np.asarray(stop).astype(float)]).astype(np.float64)


def frame_xy(d):
    m = d["code"].isin([0, 1]) & d["down"].isin([1, 2, 3, 4])
    d = d[m].reset_index(drop=True)
    X = feats(d["down"].to_numpy(), d["dist"].to_numpy(), d["yl"].to_numpy(), d["sd"].to_numpy(), d["gsr"].to_numpy(), d["qtr"].to_numpy(), d["code"].to_numpy(), d["oto"].to_numpy(), d["dto"].to_numpy(), d["stop"].to_numpy())
    y = ((d["otu"].to_numpy() > 0).astype(int) + 2 * (d["dtu"].to_numpy() > 0).astype(int))
    return d, X, y


def logloss(P, y):
    return float(-np.mean(np.log(np.clip(P[np.arange(len(y)), y], 1e-9, 1))))


def cmd_fit(args):
    import joblib
    from sklearn.ensemble import HistGradientBoostingClassifier

    rf = pd.read_parquet(OUT / "real_fit.parquet")
    rh = pd.read_parquet(OUT / "real_held.parquet")
    df, Xf, yf = frame_xy(rf)
    dh, Xh, yh = frame_xy(rh)
    clf = HistGradientBoostingClassifier(max_depth=4, max_iter=150, learning_rate=0.05, min_samples_leaf=300, l2_regularization=1.0, random_state=1).fit(Xf, yf)
    joblib.dump(clf, OUT / "policy.joblib")
    Ph = clf.predict_proba(Xh)
    Pf = clf.predict_proba(Xf)
    print("classes", np.bincount(yf) / len(yf), "held", np.bincount(yh) / len(yh))
    print("4-class log loss fit", round(logloss(Pf, yf), 5), "held", round(logloss(Ph, yh), 5))
    base = np.bincount(yf, minlength=4) / len(yf)
    print("base-rate held", round(logloss(np.tile(base, (len(yh), 1)), yh), 5))
    hf_ = Xf[:, 0].astype(int) * 100 + np.digitize(Xf[:, 1], [60, 120, 240, 480, 900])
    hh_ = Xh[:, 0].astype(int) * 100 + np.digitize(Xh[:, 1], [60, 120, 240, 480, 900])
    cell = {}
    for k in np.unique(hf_):
        cell[k] = np.bincount(yf[hf_ == k], minlength=4) / max((hf_ == k).sum(), 1)
    Pc = np.array([cell.get(k, base) for k in hh_])
    print("half x time cell held", round(logloss(Pc, yh), 5))
    z = np.load(REPO / "artifacts" / "mod25c" / "policies2_2009_2017.npz") if (REPO / "artifacts" / "mod25c" / "policies2_2009_2017.npz").exists() else None
    import mod25c_noise as c25

    f = c25.OUT / f"policies2_{c25.TRAIN[0]}_{c25.TRAIN[-1]}.npz"
    z = np.load(f)
    pt = z["ptout"]
    m25 = c25.m25
    dk = dh["down"].to_numpy().astype(int)
    sdc = np.clip(dh["sd"].to_numpy(), -24, 24)
    ph = np.where(dh["qtr"].to_numpy() >= 5, 1, 0)
    po = np.array([pt[dk[i] - 1, m25.nearest(m25.SD_AX, sdc[i]), m25.nearest(c25.TX_AX, dh["gsr"].to_numpy()[i]), ph[i], int(dh["code"].to_numpy()[i]), int(dh["oto"].to_numpy()[i] > 0), int(dh["dto"].to_numpy()[i] > 0)] for i in range(len(dh))])
    anyt = (yh > 0).astype(int)
    ll_old = float(-np.mean(anyt * np.log(np.clip(po, 1e-9, 1)) + (1 - anyt) * np.log(np.clip(1 - po, 1e-9, 1))))
    pn = 1 - Ph[:, 0]
    ll_new = float(-np.mean(anyt * np.log(np.clip(pn, 1e-9, 1)) + (1 - anyt) * np.log(np.clip(1 - pn, 1e-9, 1))))
    print("any-timeout binary log loss held: old grid", round(ll_old, 5), "new", round(ll_new, 5))
    for nm, mk in (("late-half hs<=300", Xh[:, 1] <= 300), ("hs>300", Xh[:, 1] > 300)):
        print(nm, "n", int(mk.sum()), "classes real", np.round(np.bincount(yh[mk], minlength=4) / mk.sum(), 4), "pred", np.round(Ph[mk].mean(0), 4), "old any pred", round(float(po[mk].mean()), 4), "real any", round(float(anyt[mk].mean()), 4))


def cmd_het(args):
    import joblib

    clf = joblib.load(OUT / "policy.joblib")
    rf = pd.read_parquet(OUT / "real_fit.parquet")
    nv = pd.concat([pd.read_parquet(f"{NV}/pbp_{y}.parquet", columns=["game_id", "play_id", "home_team", "home_coach", "away_coach"]) for y in FIT]).drop_duplicates(["game_id", "play_id"])
    rf = rf.merge(nv, on=["game_id", "play_id"], how="left", suffixes=("", "_nv"))
    rf["ocoach"] = np.where(rf["posteam"] == rf["home_team"], rf["home_coach"], rf["away_coach"])
    rf["dcoach"] = np.where(rf["posteam"] == rf["home_team"], rf["away_coach"], rf["home_coach"])
    d, X, y = frame_xy(rf)
    P = clf.predict_proba(X)
    d["po"] = P[:, 1] + P[:, 3]
    d["pdf"] = P[:, 2] + P[:, 3]
    d["ro"] = (d["otu"] > 0).astype(float) - d["po"]
    d["rd"] = (d["dtu"] > 0).astype(float) - d["pdf"]
    d["hs"] = X[:, 1]
    d["ph"] = X[:, 0]
    late = d[(d["hs"] <= args.hs) & d["ocoach"].notna()].copy()
    late["gi"] = late.groupby("ocoach")["g"].transform(lambda s: pd.factorize(s)[0])
    out = {}
    for who, co, r, pr, ob in (("off", "ocoach", "ro", "po", "otu"), ("def", "dcoach", "rd", "pdf", "dtu")):
        L = late.copy()
        L["gi"] = L.groupby(co)["g"].transform(lambda s: pd.factorize(s)[0])
        L["par"] = L["gi"] % 2
        L["early"] = L["season"] <= 2013
        res = {}
        for nm, key in (("parity", "par"), ("era", "early")):
            a = L[L[key].astype(bool)].groupby(co)[r].agg(["mean", "count"])
            b = L[~L[key].astype(bool)].groupby(co)[r].agg(["mean", "count"])
            j = a.join(b, lsuffix="_a", rsuffix="_b", how="inner")
            j = j[(j["count_a"] >= args.min_n) & (j["count_b"] >= args.min_n)]
            rr = float(np.corrcoef(j["mean_a"], j["mean_b"])[0, 1]) if len(j) > 3 else float("nan")
            bs = []
            rng = np.random.default_rng(5)
            for _ in range(2000):
                k = rng.integers(0, len(j), len(j))
                bs.append(np.corrcoef(j["mean_a"].to_numpy()[k], j["mean_b"].to_numpy()[k])[0, 1])
            bs = np.array(bs)
            res[nm] = {"coaches": int(len(j)), "r": rr, "spearman_brown": 2 * rr / (1 + rr) if nm == "parity" else None, "p_pos": float(np.mean(bs > 0))}
        c = L.groupby(co)[r].agg(["mean", "count"])
        c = c[c["count"] >= args.min_n]
        pbar = float(L[pr].mean())
        se2 = pbar * (1 - pbar) / c["count"]
        tau2 = float(c["mean"].var() - se2.mean())
        res["coaches_all"] = int(len(c))
        res["tau2"] = tau2
        res["tau"] = float(np.sqrt(max(tau2, 0)))
        res["mean_pred_rate"] = pbar
        res["rows"] = int(len(L))
        out[who] = res
    print(json.dumps(out, indent=1))
    (OUT / "het.json").write_text(json.dumps(out, indent=1))


dv.DV["f2"] = dict(dv.DV["u4g"], f2=1)
SALT = 4100
ORDER = (3, 2, 1, 0, 4)


def install_f2():
    import joblib
    import sim_fast

    sim = dv.sim
    ns = dv._G["ns"]
    orig = ns["DECIDE"]
    fg = sim_fast.FastGBM(joblib.load(OUT / "policy.joblib"))
    sim.PBP_SNAPSHOT_DIR = dv.m25.SNAP
    pbp = sim.load_reg_seasons(tuple(dv.TRAIN))
    tr = sim.build_transition_frame(pbp)
    a = dv._G["tables"]["arrays"]
    assert len(tr) == len(a["play_type_code"])
    ex = pd.concat([pd.read_parquet(f"{dv.m25.SNAP}/season={y}/plays.parquet", columns=["game_id", "play_id", "penalty"]) for y in dv.TRAIN]).drop_duplicates(["game_id", "play_id"])
    pen = tr[["game_id", "play_id"]].merge(ex, on=["game_id", "play_id"], how="left")["penalty"].fillna(0).to_numpy()
    code = tr["play_type_code"].to_numpy()
    stop = ((pen == 1) | ((code == 1) & (tr["yards_gained"].to_numpy() == 0)) | ((tr["points_off"].to_numpy() + tr["points_def"].to_numpy()) > 0) | tr["possession_flip"].to_numpy().astype(bool)).astype(int)
    cls = (tr["off_to_used"].to_numpy() > 0).astype(int) + 2 * (tr["def_to_used"].to_numpy() > 0).astype(int)
    tov = np.asarray(dv._G["tables"]["attrs"]["tov"]).astype(int)
    down_arr = tr["down_i"].to_numpy()
    phase_arr = tr["phase"].to_numpy()
    fm = sim.feature_matrix(tr["dist_raw"].to_numpy(), tr["fp_raw"].to_numpy(), tr["sc_raw"].to_numpy(), tr["time_raw"].to_numpy(), tr["off_to_raw"].to_numpy(), tr["def_to_raw"].to_numpy(), phase_arr)
    trees = {}
    cache = {}
    st = {"k": None, "rng": None}
    cseed = int(dv._G["cfg"].get("seed", 3))

    def tree(dk, p, c, stp, tv, cl):
        k = (dk, p, c, stp, tv, cl)
        if k not in trees:
            sub = np.flatnonzero((down_arr == dk) & sim.phase_pool_mask(phase_arr, p) & (code == c) & (stop == stp) & (tov == tv) & (cls == cl))
            trees[k] = (sim.KDTree(fm[sub]), sub) if len(sub) >= 8 else None
        return trees[k]

    def pick(rng, dk, ph, c, stp, tv, cl, dist, yl, sd, tf, ot_, dt_):
        for p in (ph,) + ORDER:
            ent = tree(dk, p, c, stp, tv, cl)
            if ent is None:
                continue
            key = (sim.round_state_key(dk, p, dist, yl, sd, tf, ot_, dt_), dk, c, stp, tv, cl)
            nb = cache.get(key)
            if nb is None:
                t, sub = ent
                f = sim.feature_matrix(np.array([dist]), np.array([yl]), np.array([sd]), np.array([tf]), np.array([ot_]), np.array([dt_]), np.array([p]))
                _, ind = t.query(f, k=min(sim.K_NEIGHBORS, len(sub)))
                nb = sub[ind[0]]
                cache[key] = nb
            return int(nb[rng.integers(len(nb))])
        return None

    def ctdraw(idx, qtr, clock_val, in_ot, down, distance, yardline, score_diff, off_to, def_to):
        if st["k"] != dv._G.get("task_key"):
            st["k"] = dv._G.get("task_key")
            st["rng"] = dv.task_rng(SALT, cseed)
        r = st["rng"]
        dk = down if down in (1, 2, 3, 4) else 4
        x = feats(np.array([dk]), np.array([distance]), np.array([yardline]), np.array([score_diff]), np.array([clock_val]), np.array([5 if in_ot else qtr]), np.array([int(code[idx])]), np.array([off_to]), np.array([def_to]), np.array([stop[idx]]))[0]
        raw = fg.raw(x)
        pr = np.exp(raw - raw.max())
        pr = pr / pr.sum()
        if off_to <= 0:
            pr[0] += pr[1]
            pr[2] += pr[3]
            pr[1] = pr[3] = 0.0
        if def_to <= 0:
            pr[0] += pr[2]
            pr[1] += pr[3]
            pr[2] = pr[3] = 0.0
        return min(int(np.searchsorted(np.cumsum(pr), r.random(), side="right")), 3)

    def tostep(idx, qtr, clock_val, in_ot, down, distance, yardline, score_diff, off_to, def_to, phase, time_feat):
        c0 = int(code[idx])
        if c0 not in (0, 1):
            return idx
        ct = ctdraw(idx, qtr, clock_val, in_ot, down, distance, yardline, score_diff, off_to, def_to)
        r = st["rng"]
        dk = down if down in (1, 2, 3, 4) else 4
        if ct == cls[idx]:
            return idx
        args = (dk, phase, c0, int(stop[idx]), int(tov[idx]))
        j = pick(r, *args, ct, distance, yardline, score_diff, time_feat, off_to, def_to)
        if j is None and ct > 0:
            j = pick(r, *args, 0, distance, yardline, score_diff, time_feat, off_to, def_to)
        return idx if j is None else j

    def wrap(idx, rng, tbl, qtr, clock_val, in_ot, down, distance, yardline, score_diff, off_to, def_to, phase, time_feat):
        idx = orig(idx, rng, tbl, qtr, clock_val, in_ot, down, distance, yardline, score_diff, off_to, def_to, phase, time_feat)
        return tostep(idx, qtr, clock_val, in_ot, down, distance, yardline, score_diff, off_to, def_to, phase, time_feat)

    dv._G["f2h"] = {"ctdraw": ctdraw, "tostep": tostep, "cls": cls, "stop": stop, "code": code}
    ns["DECIDE"] = wrap


def f2_init_budget(setting):
    u4g.u4g_init_budget(setting)
    if dv._G["cfg"].get("f2"):
        install_f2()


def f2_init_ss(setting):
    u4g.u4g_init_ss(setting)
    if dv._G["cfg"].get("f2"):
        install_f2()


def cmd_sim(args):
    u4g.hk.hk_init_budget = f2_init_budget
    u4g.hk.cmd_sim(args)


def cmd_e5(args):
    u4g.hk.hk_init_ss = f2_init_ss
    u4g.hk.cmd_e5(args)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("real")
    sub.add_parser("fit")
    sm = sub.add_parser("sim")
    sm.add_argument("--variant", default="f2")
    sm.add_argument("--out-dir", dest="out_dir", default=str(OUT / "play_f2"))
    sm.add_argument("--worlds", type=int, default=6)
    sm.add_argument("--seasons", type=int, default=8)
    sm.add_argument("--workers", type=int, default=3)
    sm.add_argument("--seed", type=int, default=21)
    e5 = sub.add_parser("e5")
    e5.add_argument("--variant", default="f2")
    e5.add_argument("--worlds", type=int, default=8)
    e5.add_argument("--seasons", type=int, default=8)
    e5.add_argument("--workers", type=int, default=3)
    e5.add_argument("--seed", type=int, default=11)
    h = sub.add_parser("het")
    h.add_argument("--hs", type=float, default=300.0)
    h.add_argument("--min-n", dest="min_n", type=int, default=150)
    d = sub.add_parser("desc")
    d.add_argument("--sim", default=str(u4g.OUT / "sim_crzhk.parquet"))
    args = ap.parse_args()
    {"real": cmd_real, "desc": cmd_desc, "fit": cmd_fit, "het": cmd_het, "sim": cmd_sim, "e5": cmd_e5}[args.cmd](args)


if __name__ == "__main__":
    main()
