import json
import os
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

FIT = REPO / "artifacts" / "mod25e3" / "geo" / "fit.json"
SEASONS = tuple(range(2009, 2018))
BANDS = (0, 1, 2, 3, 5, 8, 13, 21)
GAIN_EDGES = (0, 1, 2, 3, 4, 6, 10, 20)
NCLS = 2 + len(GAIN_EDGES) + 1
RUN_PASS = (0, 1)
SALT = 8131
YL_MAX = 40
PRIOR_BAND = 2


def enabled():
    return os.environ.get("GEO") in ("1", "2")


def classes(td, free, y):
    c = 2 + np.digitize(y, GAIN_EDGES)
    c = np.where(free, 1, c)
    return np.where(td, 0, c).astype(np.int64)


def consistent(td, free, y, fp, q):
    return free | (td & (fp >= q)) | (~td & ~free & (y < q))


def load_real():
    import sim04_engine as sim

    pbp = sim.load_reg_seasons(SEASONS)
    trans = sim.build_transition_frame(pbp).reset_index(drop=True)
    season = trans["game_id"].astype(str).str.slice(0, 4).astype(int).to_numpy()
    keep = np.isin(trans["play_type_code"].to_numpy(), RUN_PASS) & np.isin(trans["down_i"].to_numpy(), (1, 2, 3, 4))
    t = trans[keep]
    f = sim.feature_matrix(t["dist_raw"].to_numpy(), t["fp_raw"].to_numpy(), t["sc_raw"].to_numpy(), t["time_raw"].to_numpy(), t["off_to_raw"].to_numpy(), t["def_to_raw"].to_numpy(), t["phase"].to_numpy())
    flip = t["possession_flip"].to_numpy().astype(bool)
    pdf = t["points_def"].to_numpy() > 0
    td = t["points_off"].to_numpy() >= 6
    free = (flip | pdf) & ~td
    y = t["yards_gained"].to_numpy().astype(float)
    return {
        "season": season[keep],
        "game": t["game_id"].to_numpy(),
        "down": t["down_i"].to_numpy().astype(int),
        "code": t["play_type_code"].to_numpy().astype(int),
        "fp": t["fp_raw"].to_numpy().astype(float),
        "f4": f[:, :4],
        "f3": f[:, [0, 2, 3]],
        "td": td,
        "free": free,
        "y": y,
        "cls": classes(td, free, y),
    }


def nearest(feat_pool, feat_q, k):
    n = len(feat_pool)
    kk = min(k, n)
    d2 = ((feat_q[:, None, :] - feat_pool[None, :, :]) ** 2).sum(axis=2)
    if kk < n:
        return np.argpartition(d2, kk - 1, axis=1)[:, :kk], kk
    return np.tile(np.arange(n), (len(feat_q), 1)), kk


def prior_for(R, tr_idx, q):
    w = PRIOR_BAND
    while True:
        s = tr_idx[np.abs(R["fp"][tr_idx] - q) <= w]
        s = s[consistent(R["td"][s], R["free"][s], R["y"][s], R["fp"][s], q)]
        if len(s) > 0 or w >= 99:
            break
        w += 1
    if len(s) == 0:
        return np.full(NCLS, 1.0 / NCLS)
    c = np.bincount(R["cls"][s], minlength=NCLS).astype(float) + 1.0
    return c / c.sum()


def band_pool(R, tr_sorted, fp_sorted, q, b):
    w = b
    while True:
        lo = np.searchsorted(fp_sorted, q - w, side="left")
        hi = np.searchsorted(fp_sorted, q + w, side="right")
        s = tr_sorted[lo:hi]
        s = s[consistent(R["td"][s], R["free"][s], R["y"][s], R["fp"][s], q)]
        if len(s) > 0 or w >= 99:
            return s
        w += 1


def counts(cls_mat, nclass=NCLS):
    out = np.zeros((cls_mat.shape[0], nclass))
    for cc in range(nclass):
        out[:, cc] = (cls_mat == cc).sum(axis=1)
    return out


def evaluate(R, test_season, excl, k, bands=BANDS):
    from scipy.spatial import cKDTree

    tr_mask = ~np.isin(R["season"], list(excl))
    te_all = np.flatnonzero((R["season"] == test_season) & (R["fp"] <= YL_MAX))
    out = {"A": np.zeros(len(te_all)), "viol_A": np.zeros(len(te_all))}
    for b in bands:
        out[("B", b)] = np.zeros(len(te_all))
        out[("M", b)] = np.zeros(len(te_all))
    for d in (1, 2, 3, 4):
        for c in RUN_PASS:
            tr = np.flatnonzero(tr_mask & (R["down"] == d) & (R["code"] == c))
            te = te_all[(R["down"][te_all] == d) & (R["code"][te_all] == c)]
            if len(te) == 0:
                continue
            kk = min(k, len(tr))
            _, nb = cKDTree(R["f4"][tr]).query(R["f4"][te], k=kk)
            nb = tr[np.asarray(nb).reshape(len(te), kk)]
            order = np.argsort(R["fp"][tr], kind="stable")
            tr_sorted = tr[order]
            fp_sorted = R["fp"][tr_sorted]
            qs = np.round(R["fp"][te]).astype(int)
            for q in np.unique(qs):
                g = np.flatnonzero(qs == q)
                te_g = te[g]
                nb_g = nb[g]
                rows = np.arange(len(g))
                pos = np.searchsorted(te_all, te_g)
                cons = consistent(R["td"][nb_g], R["free"][nb_g], R["y"][nb_g], R["fp"][nb_g], float(q))
                cls_n = R["cls"][nb_g]
                pr = prior_for(R, tr, float(q))
                cnt_all = counts(cls_n)
                cnt_con = counts(np.where(cons, cls_n, -1))
                nviol = (~cons).sum(axis=1).astype(float)
                tc = R["cls"][te_g]
                out["A"][pos] = np.log(((cnt_all + pr[None, :]) / (kk + 1.0))[rows, tc])
                out["viol_A"][pos] = nviol / kk
                for b in bands:
                    cand = band_pool(R, tr_sorted, fp_sorted, float(q), b)
                    ix, kb = nearest(R["f3"][cand], R["f3"][te_g], k)
                    cnt_b = counts(R["cls"][cand][ix])
                    pB = (cnt_b + pr[None, :]) / (kb + 1.0)
                    pM = (cnt_con + nviol[:, None] * cnt_b / kb + pr[None, :]) / (kk + 1.0)
                    out[("B", b)][pos] = np.log(pB[rows, tc])
                    out[("M", b)][pos] = np.log(pM[rows, tc])
    return te_all, out


CACHE = REPO / "tests" / "scratch" / "e115g"


def stage(name, k=200):
    import pickle
    import time

    t0 = time.time()
    R = load_real()
    if name == "outer":
        out = {}
        for t in SEASONS:
            te, o = evaluate(R, t, {t}, k)
            out[t] = (R["game"][te], R["fp"][te], o)
        (CACHE / "outer.pkl").write_bytes(pickle.dumps(out))
    else:
        o_ = int(name)
        out = {}
        for t in SEASONS:
            if t == o_:
                continue
            _, o = evaluate(R, t, {t, o_}, k)
            out[t] = {b: float(o[("M", b)].sum()) for b in BANDS}
        (CACHE / f"inner_{o_}.pkl").write_bytes(pickle.dumps(out))
    print(name, round(time.time() - t0), flush=True)


def fit(k=200):
    import pickle
    import time

    t0 = time.time()
    seasons = list(SEASONS)
    outer_raw = pickle.loads((CACHE / "outer.pkl").read_bytes())
    outer = {}
    inner = {}
    games = {}
    viol_A = {}
    for t in seasons:
        game, fp, o = outer_raw[t]
        outer[t] = {kk_: float(v.sum()) for kk_, v in o.items() if kk_ != "viol_A"}
        outer[t]["n"] = int(len(game))
        games[t] = (game, {kk_: v for kk_, v in o.items() if kk_ != "viol_A"})
        viol_A[t] = (fp, o["viol_A"])
    for o_ in seasons:
        d = pickle.loads((CACHE / f"inner_{o_}.pkl").read_bytes())
        for t, v in d.items():
            inner[(o_, t)] = v
    res = {"per_fold": {}, "bands": list(BANDS), "k": k}
    for o_ in seasons:
        tot = {b: sum(inner[(o_, t)][b] for t in seasons if t != o_) for b in BANDS}
        best = max(tot, key=tot.get)
        n = outer[o_]["n"]
        res["per_fold"][str(o_)] = {
            "band": int(best),
            "n": n,
            "ll_A": outer[o_]["A"] / n,
            "ll_M_nested": outer[o_][("M", best)] / n,
            "ll_B_nested": outer[o_][("B", best)] / n,
            "gain_M": (outer[o_][("M", best)] - outer[o_]["A"]) / n,
            "gain_B": (outer[o_][("B", best)] - outer[o_]["A"]) / n,
            "ll_M_by_band": {str(b): outer[o_][("M", b)] / n for b in BANDS},
        }
    ntot = sum(outer[t]["n"] for t in seasons)
    tot_final = {b: sum(outer[t][("M", b)] for t in seasons) for b in BANDS}
    res["final_band"] = int(max(tot_final, key=tot_final.get))
    res["final_M_ll_by_band"] = {str(b): tot_final[b] / ntot for b in BANDS}
    res["final_A_ll"] = sum(outer[t]["A"] for t in seasons) / ntot
    res["final_B_ll_by_band"] = {str(b): sum(outer[t][("B", b)] for t in seasons) / ntot for b in BANDS}
    res["wins_M_of_9"] = int(sum(1 for t in seasons if res["per_fold"][str(t)]["gain_M"] > 0))
    res["wins_B_of_9"] = int(sum(1 for t in seasons if res["per_fold"][str(t)]["gain_B"] > 0))
    res["mean_gain_M"] = sum(res["per_fold"][str(t)]["gain_M"] * outer[t]["n"] for t in seasons) / ntot
    res["mean_gain_B"] = sum(res["per_fold"][str(t)]["gain_B"] * outer[t]["n"] for t in seasons) / ntot
    rng = np.random.default_rng(5)
    sums, nums = [], []
    for t in seasons:
        bnd = res["per_fold"][str(t)]["band"]
        te_g, lls = games[t]
        gk, gi = np.unique(te_g, return_inverse=True)
        sums.append(np.bincount(gi, weights=lls[("M", bnd)] - lls["A"], minlength=len(gk)))
        nums.append(np.bincount(gi, minlength=len(gk)).astype(float))
    cs, cn = np.concatenate(sums), np.concatenate(nums)
    reps = np.array([cs[ii].sum() / cn[ii].sum() for ii in (rng.integers(0, len(cs), len(cs)) for _ in range(300))])
    res["boot_game_mean_gain_M"] = {"p05": float(np.quantile(reps, 0.05)), "p95": float(np.quantile(reps, 0.95)), "probability_positive": float((reps > 0).mean())}
    lo = np.concatenate([viol_A[t][0] for t in seasons])
    va = np.concatenate([viol_A[t][1] for t in seasons])
    res["engine_like_violation_share_of_draws"] = {"yl2_5": float(va[(lo >= 2) & (lo <= 5)].mean()), "yl6_10": float(va[(lo >= 6) & (lo <= 10)].mean()), "yl11_40": float(va[lo > 10].mean())}
    res["looks"] = {"bands": len(BANDS), "variants": 3, "folds": 9, "inner_pairs": 72, "family": "geometry re-pick band choice on held-out log-lik of TD, free, gain bin"}
    res["seconds"] = round(time.time() - t0)
    FIT.parent.mkdir(parents=True, exist_ok=True)
    FIT.write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(json.dumps(res, indent=1))


REPICK_KEYS = ("points_off", "points_def", "clock_elapsed", "next_down", "next_distance", "next_yardline", "dist_gained", "off_to_used", "def_to_used", "play_type_code")


class Geo:
    def __init__(self, arrays, ids, dat, band, k_state, gain):
        self.a = arrays
        self.band = int(band)
        self.k = int(k_state)
        self.gain = float(gain)
        self.pools = {}
        td = arrays["points_off"] >= 6
        free = (arrays["possession_flip"].astype(bool) | (arrays["points_def"] > 0)) & ~td
        for d in (1, 2, 3, 4):
            for c in RUN_PASS:
                m = (arrays["down_i"][ids] == d) & (arrays["play_type_code"][ids] == c)
                sel = ids[m]
                o = np.argsort(arrays["fp_raw"][sel], kind="stable")
                sel = sel[o]
                self.pools[(d, c)] = {
                    "ids": sel,
                    "fp": arrays["fp_raw"][sel].astype(float),
                    "f3": dat[m][o][:, [0, 2, 3]],
                    "td": td[sel],
                    "free": free[sel],
                    "y": arrays["yards_gained"][sel].astype(float),
                    "net": (arrays["off_row"][sel] - arrays["def_row"][sel]).astype(float),
                }

    def violates(self, drawn, i, q):
        if float(drawn["points_off"]) >= 6:
            return float(self.a["fp_raw"][int(i)]) < q
        if bool(drawn["flip"]) or float(drawn["points_def"]) > 0:
            return False
        return float(drawn["yards_gained"]) >= q

    def candidates(self, P, q, net_sim):
        w = self.band
        while True:
            lo = np.searchsorted(P["fp"], q - w, side="left")
            hi = np.searchsorted(P["fp"], q + w, side="right")
            sl = slice(lo, hi)
            shift = self.gain * (net_sim - P["net"][sl])
            ok = P["free"][sl] | (P["td"][sl] & (P["fp"][sl] >= q)) | (~P["td"][sl] & ~P["free"][sl] & (P["y"][sl] + shift < q))
            pos = np.flatnonzero(ok) + lo
            if len(pos) > 0 or w >= 99:
                return pos
            w += 1

    def fix(self, drawn, i, down, q, f3q, net_sim, weight_fn, rng, force=False):
        if not force and not self.violates(drawn, i, q):
            return None
        d = down if down in (1, 2, 3, 4) else 4
        P = self.pools.get((d, int(drawn["play_type_code"])))
        if P is None:
            return None
        pos = self.candidates(P, q, net_sim)
        if len(pos) == 0:
            return None
        dd = ((P["f3"][pos] - f3q[None, :]) ** 2).sum(axis=1)
        k = min(self.k, len(pos))
        near = pos[np.argpartition(dd, k - 1)[:k]] if k < len(pos) else pos
        ids = P["ids"][near]
        w = weight_fn(ids)
        c = np.cumsum(w)
        if not np.isfinite(c[-1]) or c[-1] <= 0.0:
            j = int(ids[int(rng.integers(len(ids)))])
        else:
            j = int(ids[min(int(np.searchsorted(c, rng.random() * c[-1], side="right")), len(ids) - 1)])
        shift = self.gain * (net_sim - float(self.a["off_row"][j] - self.a["def_row"][j]))
        new = dict(drawn)
        for key in REPICK_KEYS:
            new[key] = self.a[key][j]
        new["flip"] = bool(self.a["possession_flip"][j])
        new["auto_first"] = bool(self.a["auto_first"][j])
        new["repeat_down"] = bool(self.a["repeat_down"][j])
        new["yards_gained"] = self.a["yards_gained"][j] + shift
        new["idx"] = j
        return new, j, shift


def install_geo():
    import mod25d_variance as dv
    import sim04_engine as sim

    spec = json.loads(FIT.read_text(encoding="utf-8"))
    ns = dv._G["ns"]
    tables = dv._G["tables"]
    arrays = tables["arrays"]
    seed = int(dv._G["cfg"].get("seed", 3))
    rows = {}
    for tree, sub in tables["nn_trees_cond"].values():
        rows.update(zip(sub.tolist(), tree.data))
    ids = np.array(sorted(rows), dtype=np.int64)
    dat = np.array([rows[i] for i in ids.tolist()])
    geo = Geo(arrays, ids, dat, spec["final_band"], sim.K_STATE, sim.TEAM_RATING_YARD_GAIN)
    st = {"k": None, "rng": None, "rep": None}
    always = os.environ.get("GEO") == "2"
    base = dv._G["pol"]

    def locals_of():
        fr = sys._getframe(2)
        while fr is not None and not ("offense" in fr.f_locals and "off_sim" in fr.f_locals):
            fr = fr.f_back
        return fr.f_locals

    def pol(down, distance, yardline, score_diff, qtr, clock_val, drawn):
        i = drawn.get("idx")
        if i is not None and int(drawn["play_type_code"]) in RUN_PASS and (always or geo.violates(drawn, i, float(yardline))):
            if st["k"] != dv._G.get("task_key"):
                st["k"] = dv._G.get("task_key")
                st["rng"] = dv.task_rng(SALT, seed)
            L = locals_of()
            o_rt, d_rt = L["off_sim"], L["def_sim"]
            h = tables["team_kernel_h"]

            def weight_fn(near):
                d2 = (arrays["off_row"][near] - o_rt) ** 2 + (arrays["def_row"][near] - d_rt) ** 2
                w = np.exp(-d2 / (2.0 * h * h)) * np.where(arrays["is_home_off"][near] == (1 if L["offense"] == "home" else 0), ns["TEAM_KERNEL_LAMBDA"], 1.0)
                if "IPW" in ns:
                    w = w * np.asarray(ns["IPW"])[near]
                if "PASS_W" in ns:
                    w = w * ns["PASS_W"](None, near, o_rt, d_rt)
                return w

            tf = sim.continuous_time_feature(qtr, clock_val)
            f = sim.feature_matrix(np.array([distance]), np.array([float(yardline)]), np.array([score_diff]), np.array([tf]), np.array([0.0]), np.array([0.0]), np.array([sim.compute_phase(qtr, clock_val)]))[0]
            out = geo.fix(drawn, i, down, float(yardline), f[[0, 2, 3]], float(o_rt - d_rt), weight_fn, st["rng"], force=always)
            if out is not None:
                drawn, j, shift = out
                st["rep"] = (j, shift)
        return base(down, distance, yardline, score_diff, qtr, clock_val, drawn)

    class PL(list):
        def append(self, v):
            rep = st.pop("rep", None)
            st["rep"] = None
            super().append(v if rep is None else (v[0], rep[0], rep[1]))

    ns["PLAYLOG"] = PL(ns["PLAYLOG"])
    dv._G["pol"] = pol


def check(n=6000):
    R = load_real()
    import sim04_engine as sim

    pbp = sim.load_reg_seasons(SEASONS)
    trans = sim.build_transition_frame(pbp).reset_index(drop=True)
    a = {k_: trans[k_].to_numpy() for k_ in ("down_i", "play_type_code", "fp_raw", "points_off", "points_def", "possession_flip", "yards_gained", "next_down", "next_distance", "next_yardline", "dist_gained", "clock_elapsed", "off_to_used", "def_to_used", "auto_first", "repeat_down", "dist_raw", "sc_raw", "time_raw", "off_to_raw", "def_to_raw", "phase")}
    a["off_row"] = np.zeros(len(trans))
    a["def_row"] = np.zeros(len(trans))
    f = sim.feature_matrix(a["dist_raw"], a["fp_raw"], a["sc_raw"], a["time_raw"], a["off_to_raw"], a["def_to_raw"], a["phase"])
    ids = np.flatnonzero(np.isin(a["down_i"], (1, 2, 3, 4)))
    spec = json.loads(FIT.read_text(encoding="utf-8"))
    geo = Geo(a, ids, f[ids], spec["final_band"], sim.K_STATE, sim.TEAM_RATING_YARD_GAIN)
    rng = np.random.default_rng(11)
    from scipy.spatial import cKDTree

    qs_rows = np.flatnonzero(np.isin(a["play_type_code"], RUN_PASS) & (a["fp_raw"] <= 99) & np.isin(a["down_i"], (1, 2, 3, 4)))
    pick = rng.choice(qs_rows, size=min(n * 8, len(qs_rows)), replace=False)
    ones = lambda near: np.ones(len(near))
    stats = {"pre": {}, "post_viol": {}, "post_all": {}}
    bins = ((1, 1), (2, 5), (6, 10), (11, 20), (21, 40), (41, 99))
    for d in (1, 2, 3, 4):
        for c in RUN_PASS:
            pool = ids[(a["down_i"][ids] == d) & (a["play_type_code"][ids] == c)]
            tree = cKDTree(f[pool][:, :4])
            qr = pick[(a["down_i"][pick] == d) & (a["play_type_code"][pick] == c)]
            _, nb = tree.query(f[qr][:, :4], k=sim.K_STATE)
            for n_, r in enumerate(qr.tolist()):
                j0 = int(pool[nb[n_][rng.integers(len(nb[n_]))]])
                q = float(a["fp_raw"][r])
                drawn = {k_: a[k_][j0] for k_ in REPICK_KEYS}
                drawn["flip"] = bool(a["possession_flip"][j0])
                drawn["auto_first"] = bool(a["auto_first"][j0])
                drawn["repeat_down"] = bool(a["repeat_down"][j0])
                drawn["yards_gained"] = float(a["yards_gained"][j0])
                drawn["idx"] = j0
                for tag, dr_ in (("pre", drawn), ("post_viol", None), ("post_all", None)):
                    if tag != "pre":
                        out = geo.fix(drawn, j0, d, q, f[r][[0, 2, 3]], 0.0, ones, rng, force=tag == "post_all")
                        dr_ = out[0] if out is not None else drawn
                    td = float(dr_["points_off"]) >= 6
                    bad = geo.violates(dr_, dr_["idx"], q)
                    b = next(i for i, (lo, hi) in enumerate(bins) if lo <= q <= hi)
                    s = stats[tag].setdefault(b, [0, 0, 0])
                    s[0] += 1
                    s[1] += int(bad)
                    s[2] += int(td)
    res = {}
    for tag in ("pre", "post_viol", "post_all"):
        res[tag] = {f"yl{bins[b][0]}-{bins[b][1]}": {"n": s[0], "violation_share": s[1] / s[0], "td_per_play": s[2] / s[0]} for b, s in sorted(stats[tag].items())}
    real = {}
    for b, (lo, hi) in enumerate(bins):
        m = (R["fp"] >= lo) & (R["fp"] <= hi)
        real[f"yl{lo}-{hi}"] = {"n": int(m.sum()), "td_per_play": float(R["td"][m].mean())}
    res["real"] = real
    print(json.dumps(res, indent=1))
    FIT.with_name("check.json").write_text(json.dumps(res, indent=1), encoding="utf-8")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "fit":
        fit()
    if len(sys.argv) > 2 and sys.argv[1] == "stage":
        stage(sys.argv[2])
    if len(sys.argv) > 1 and sys.argv[1] == "check":
        check()
