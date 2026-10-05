import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
ART = REPO / "artifacts" / "mod25e3"
OUTD = ART / "kos"
KICKS = ART / "kick" / "kicks_2009.parquet"
HK = {"raw": (3.0, 6.0, 12.0, 24.0, 48.0), "eng": (1.0, 2.0, 4.0, 8.0, 16.0)}
HT = {"raw": (150.0, 300.0, 600.0, 1200.0, 2400.0), "sqrt": (3.0, 6.0, 12.0, 24.0, 48.0)}
M_GRID = (50, 100, 200, 400)
WT_GRID = (0.25, 0.5, 1.0, 2.0, 4.0)
LAM = (2.0, 8.0, 32.0, 128.0, 512.0)
OUT = []


def enabled():
    return os.environ.get("KOS") == "1"


def say(s=""):
    print(s, flush=True)
    OUT.append(s)


def kd_feat(kd, kt):
    if kt == "raw":
        return np.asarray(kd, float)
    import sim04_engine as sim

    return np.asarray(sim.scaled_score_diff(np.asarray(kd, float)), float)


def t_feat(gs, tt):
    g = np.asarray(gs, float)
    return g if tt == "raw" else np.sqrt(np.maximum(g, 0.0))


def cells(K):
    r = K.retained.to_numpy(int)
    t = K.rtd.to_numpy(int)
    f = K.nfp.to_numpy(float)
    c = np.where(t == 1, 0, np.where(r == 1, 1000 + (f // 10).astype(int), f.astype(int)))
    return pd.factorize(c)[0]


def load():
    K = pd.read_parquet(KICKS).reset_index(drop=True)
    K["valid"] = ((K.retained == 0) & (K.rtd == 0)).astype(float)
    K["x"] = 2.0 - pd.cut(K.kd, [-np.inf, -8.5, -0.5, 0.5, 8.5, np.inf], labels=False).astype(float)
    return K


def fg_flag(K):
    import mod25e_kick as mk

    out = []
    for sn in sorted(K.season.unique()):
        p = pd.read_parquet(mk.PBP / f"season={sn}" / "plays.parquet", columns=["game_id", "play_id", "play_type", "season_type"])
        p = p[p.season_type == "REG"].sort_values(["game_id", "play_id"], kind="stable").reset_index(drop=True)
        p["prev"] = p.play_type.shift(1)
        p.loc[p.game_id != p.game_id.shift(1), "prev"] = None
        out.append(p[["game_id", "play_id", "prev"]])
    P = pd.concat(out, ignore_index=True)
    return (K[["game_id", "play_id"]].merge(P, on=["game_id", "play_id"], how="left").prev.eq("field_goal")).to_numpy()


def slope(x, y, q):
    ok = np.isfinite(x) & np.isfinite(y)
    x, y, q = x[ok], y[ok], q[ok]
    xd = x - pd.Series(x).groupby(q).transform("mean").to_numpy()
    yd = y - pd.Series(y).groupby(q).transform("mean").to_numpy()
    return float((xd * yd).sum() / (xd * xd).sum())


def fit_kick():
    K = load()
    N = len(K)
    y = cells(K)
    C = int(y.max()) + 1
    seasons = sorted(K.season.unique())
    kd, gs = K.kd.to_numpy(float), K.gsr.to_numpy(float)
    nfp, valid = K.nfp.to_numpy(float), K.valid.to_numpy(float)
    models = [("cur", None)]
    models += [("knn", (m, wt)) for m in M_GRID for wt in WT_GRID]
    models += [("gauss", (kt, tt, hk, ht)) for kt in HK for tt in HT for hk in HK[kt] for ht in HT[tt]]
    names = ["cur"] + [f"knn m{m} wt{wt:g}" for m in M_GRID for wt in WT_GRID] + [f"gauss kd:{kt} {hk:g} t:{tt} {ht:g}" for kt in HK for tt in HT for hk in HK[kt] for ht in HT[tt]]
    LL = np.zeros((len(models), len(LAM), len(seasons)))
    YH = np.full((len(models), N), np.nan)
    YA = np.full((len(models), N), np.nan)
    anyv = (K.rtd.to_numpy() == 0).astype(float)
    for si, s in enumerate(seasons):
        te = np.flatnonzero(K.season.to_numpy() == s)
        tr = np.flatnonzero(K.season.to_numpy() != s)
        cnt = np.bincount(y[tr], minlength=C) + 0.5
        g = cnt / cnt.sum()
        M = (y[te][:, None] == y[tr][None, :]).astype(np.float32)
        v = valid[tr].astype(np.float32)
        nv = (valid[tr] * nfp[tr]).astype(np.float32)
        va = anyv[tr].astype(np.float32)
        na = (anyv[tr] * nfp[tr]).astype(np.float32)
        dk = {kt: (kd_feat(kd[te], kt)[:, None] - kd_feat(kd[tr], kt)[None, :]).astype(np.float32) ** 2 for kt in HK}
        dt = {tt: (t_feat(gs[te], tt)[:, None] - t_feat(gs[tr], tt)[None, :]).astype(np.float32) ** 2 for tt in HT}
        Ek = {(kt, h): np.exp(-0.5 * dk[kt] / np.float32(h * h)) for kt in HK for h in HK[kt]}
        Et = {(tt, h): np.exp(-0.5 * dt[tt] / np.float32(h * h)) for tt in HT for h in HT[tt]}
        sk, st = kd[tr].std(), gs[tr].std()
        gy = g[y[te]]

        def score(mi, num, den):
            for li, lam in enumerate(LAM):
                LL[mi, li, si] = np.log((num + lam * gy) / (den + lam)).sum()

        def put(mi, w):
            YH[mi, te] = (w @ nv) / np.maximum(w @ v, 1e-12)
            YA[mi, te] = (w @ na) / np.maximum(w @ va, 1e-12)

        d0 = (kd[te][:, None] - kd[tr][None, :]) ** 2 / sk**2
        d1 = (gs[te][:, None] - gs[tr][None, :]) ** 2 / st**2
        mcur = int(np.sqrt(len(tr)))
        order_cache = {}
        for mi, (fam, p) in enumerate(models):
            if fam == "cur":
                wt, ms = 1.0, [mcur]
            elif fam == "knn":
                wt, ms = p[1], [p[0]]
            else:
                wt, ms = None, None
            if fam in ("cur", "knn"):
                if wt not in order_cache:
                    d = d0 + wt * wt * d1
                    top = np.argpartition(d, 399, axis=1)[:, :400]
                    o = np.take_along_axis(top, np.argsort(np.take_along_axis(d, top, 1), axis=1), 1)
                    order_cache[wt] = o
                o = order_cache[wt]
                m = ms[0]
                sel = o[:, :m]
                num = np.take_along_axis(M, sel, 1).sum(1)
                score(mi, num, np.full(len(te), float(m)))
                W = np.zeros((len(te), len(tr)), np.float32)
                np.put_along_axis(W, sel, 1.0, 1)
                put(mi, W)
            else:
                kt, tt, hk, ht = p
                W = Ek[(kt, hk)] * Et[(tt, ht)]
                score(mi, (W * M).sum(1), W.sum(1))
                put(mi, W)
        say(f"fold {s} done")
    tab = LL.sum(2)
    mean = LL.mean(2)
    best = {}
    for mi, (fam, p) in enumerate(models):
        li = int(mean[mi].argmax())
        best[mi] = (li, mean[mi, li])
    say(f"kicks {N} seasons {len(seasons)}; looks: {len(models)} models x {len(LAM)} lam = {len(models) * len(LAM)}; mean held-out log lik per kick")
    cur_li, cur_ll = best[0]
    say(f"current (sqrt(N) nearest, std scaling): lam {LAM[cur_li]:g} LL/kick {cur_ll / (N / len(seasons)):.5f}")
    rank = sorted(range(1, len(models)), key=lambda i: -best[i][1])[:8]
    per = N / len(seasons)
    for mi in rank:
        li, v = best[mi]
        wins = int((LL[mi, li] > LL[0, cur_li]).sum())
        say(f"  {names[mi]:42s} lam {LAM[li]:g} LL/kick {v / per:.5f} gain vs cur {(v - cur_ll) / per:+.5f} fold wins {wins}/{len(seasons)}")
    flat = [(mi, li) for mi in range(len(models)) for li in range(len(LAM))]
    wins_n, gains = 0, []
    for si, s in enumerate(seasons):
        oth = [j for j in range(len(seasons)) if j != si]
        mi, li = max(flat, key=lambda a: LL[a[0], a[1], oth].mean())
        cl = max(range(len(LAM)), key=lambda l: LL[0, l, oth].mean())
        gain = LL[mi, li, si] - LL[0, cl, si]
        gains.append(gain / (K.season == s).sum())
        wins_n += int(gain > 0)
        say(f"  nested-lite fold {s}: chose {names[mi]} lam {LAM[li]:g}; LL/kick gain vs cur {gains[-1]:+.5f}")
    say(f"nested-lite (selection on other seasons' LOSO scores) mean gain {np.mean(gains):+.5f} wins {wins_n}/{len(seasons)}")
    top = rank[0]
    li = best[top][0]
    q = K.qtr.to_numpy(float)
    x = K.x.to_numpy(float)
    vm = valid > 0
    say(f"start yardline slope per receiver lead-band step (quarter intercepts, nonreturned non-TD kicks, {int(vm.sum())}); positive x = receiver leads")
    say(f"  real {slope(x[vm], nfp[vm], q[vm]):+.3f}  current LOSO draw mean {slope(x[vm], YH[0][vm], q[vm]):+.3f}  best {names[top]} {slope(x[vm], YH[top][vm], q[vm]):+.3f}")
    for qq in (1, 2, 3, 4):
        mq = vm & (q == qq)
        say(f"  Q{qq} real {slope(x[mq], nfp[mq], q[mq]):+.3f} cur {slope(x[mq], YH[0][mq], q[mq]):+.3f} best {slope(x[mq], YH[top][mq], q[mq]):+.3f}")
    nr = anyv > 0
    say(f"same slope over all non-TD kicks including onside recoveries ({int(nr.sum())}), x = receiver lead band")
    say(f"  real {slope(x[nr], nfp[nr], q[nr]):+.3f}  current {slope(x[nr], YA[0][nr], q[nr]):+.3f}  best {slope(x[nr], YA[top][nr], q[nr]):+.3f}")
    fg = fg_flag(K)
    for lab, mk_ in (("after TD", nr & ~fg), ("after FG", nr & fg)):
        say(f"{lab} ({int(mk_.sum())}): slope real {slope(x[mk_], nfp[mk_], q[mk_]):+.3f} cur {slope(x[mk_], YA[0][mk_], q[mk_]):+.3f} best {slope(x[mk_], YA[top][mk_], q[mk_]):+.3f}")
        for b in (-2, -1, 0, 1, 2):
            mb = mk_ & (x == b)
            say(f"   receiver lead band {b:+d} n {int(mb.sum()):5d}: start yl0 real {nfp[mb].mean():.2f} cur {YA[0][mb].mean():.2f} best {YA[top][mb].mean():.2f}; onside-kept share real {K.retained.to_numpy()[mb].mean():.3f}")
    np.savez(OUTD / "kick_yh.npz", cur=YA[0], best=YA[top], fg=fg)
    fam, p = models[top]
    spec = {"name": names[top], "fam": fam, "p": list(p), "lam": LAM[li], "n_train": N, "nested_wins": wins_n}
    OUTD.mkdir(parents=True, exist_ok=True)
    (OUTD / "kick_spec.json").write_text(json.dumps(spec))
    (OUTD / "kick_fit.txt").write_text("\n".join(OUT))


def tov_frame():
    import sim04_engine as sim

    F = pd.read_parquet(ART / "desp" / "frame.parquet", columns=["game_id", "play_id", "down_i", "play_type_code", "dist_raw", "fp_raw", "sc_raw", "qtr_actual", "gsr_actual", "off_to_raw", "def_to_raw", "possession_flip", "points_off", "points_def", "yards_gained", "season"]).reset_index(drop=True)
    F["time_raw"] = sim.vectorized_time_raw(F.qtr_actual.to_numpy(), F.gsr_actual.to_numpy())
    F["phase"] = sim.vectorized_phase(F.qtr_actual.to_numpy(), F.gsr_actual.to_numpy())
    F["tov"] = (F.possession_flip.astype(bool) & (F.points_off == 0) & (F.points_def == 0) & F.play_type_code.isin([0, 1])).astype(int)
    F["r"] = 100.0 - 2.0 * F.fp_raw + F.yards_gained
    return F


def fit_tov():
    import sim04_engine as sim
    from scipy.spatial import cKDTree

    F = tov_frame()
    T = F[F.tov == 1].reset_index(drop=True)
    seasons = sorted(F.season.unique())
    cell = np.clip(np.floor(T.r.to_numpy() / 5.0), -6, 20).astype(int) + 6
    C = int(cell.max()) + 1
    full = sim.feature_matrix(F.dist_raw, F.fp_raw, F.sc_raw, F.time_raw, F.off_to_raw, F.def_to_raw, F.phase.to_numpy())
    tfull = sim.feature_matrix(T.dist_raw, T.fp_raw, T.sc_raw, T.time_raw, T.off_to_raw, T.def_to_raw, T.phase.to_numpy())
    ws_grid, k_grid = (0.0, 0.5, 1.0, 2.0, 4.0), (50, 100, 200, 400)
    names = ["cur (engine metric, k200 all rows, turnover share)"] + [f"tov-only knn ws{w:g} k{k}" for w in ws_grid for k in k_grid]
    LL = np.zeros((len(names), len(LAM), len(seasons)))
    YH = np.full((len(names), len(T)), np.nan)
    rr = T.r.to_numpy()
    for si, sn in enumerate(seasons):
        te = np.flatnonzero(T.season.to_numpy() == sn)
        trT = np.flatnonzero(T.season.to_numpy() != sn)
        cnt = np.bincount(cell[trT], minlength=C) + 0.5
        g = cnt / cnt.sum()
        gy = g[cell[te]]
        trF = F.season.to_numpy() != sn
        down_t, phase_t = T.down_i.to_numpy()[te], T.phase.to_numpy()[te]
        num = np.zeros(len(te))
        den = np.zeros(len(te))
        mean_num = np.zeros(len(te))
        for d in (1, 2, 3, 4):
            for ph in range(5):
                sel = np.flatnonzero((down_t == d) & (phase_t == ph))
                if len(sel) == 0:
                    continue
                mask = trF & (F.down_i.to_numpy() == d) & sim.phase_pool_mask(F.phase.to_numpy(), ph)
                idx = np.flatnonzero(mask)
                tree = cKDTree(full[idx])
                _, ind = tree.query(tfull[te[sel]], k=min(200, len(idx)))
                nb = idx[ind]
                isT = F.tov.to_numpy()[nb] == 1
                rF = F.r.to_numpy()[nb]
                cF = np.clip(np.floor(rF / 5.0), -6, 20).astype(int) + 6
                num[sel] = ((cF == cell[te[sel]][:, None]) & isT).sum(1)
                den[sel] = isT.sum(1)
                mean_num[sel] = np.where(isT, rF, 0.0).sum(1) / np.maximum(isT.sum(1), 1)
        for li, lam in enumerate(LAM):
            LL[0, li, si] = np.log((num + lam * gy) / (den + lam)).sum()
        YH[0, te] = mean_num
        dtr = T.down_i.to_numpy()[trT]
        for wi, w in enumerate(ws_grid):
            sc = np.array([1.0, 1.0, w, 1.0, 1.0, 1.0])
            for ki, k in enumerate(k_grid):
                mi = 1 + wi * len(k_grid) + ki
                nm = np.zeros(len(te))
                dn = np.zeros(len(te))
                mu = np.zeros(len(te))
                for d in (1, 2, 3, 4):
                    sel = np.flatnonzero(down_t == d)
                    ii = trT[dtr == d]
                    if len(sel) == 0:
                        continue
                    tree = cKDTree(tfull[ii] * sc)
                    _, ind = tree.query(tfull[te[sel]] * sc, k=min(k, len(ii)))
                    nb = ii[ind]
                    nm[sel] = (cell[nb] == cell[te[sel]][:, None]).sum(1)
                    dn[sel] = nb.shape[1]
                    mu[sel] = rr[nb].mean(1)
                for li, lam in enumerate(LAM):
                    LL[mi, li, si] = np.log((nm + lam * gy) / (dn + lam)).sum()
                YH[mi, te] = mu
        say(f"fold {sn} done")
    per = len(T) / len(seasons)
    mean = LL.mean(2)
    bl = mean.argmax(1)
    bestv = mean[np.arange(len(names)), bl]
    say(f"turnover plays {len(T)} (flip, no score, run/pass), seasons {len(seasons)}; looks {len(names)} models x {len(LAM)} lam = {len(names) * len(LAM)}")
    say(f"current: lam {LAM[bl[0]]:g} LL/play {bestv[0] / per:.5f}")
    rank = sorted(range(1, len(names)), key=lambda i: -bestv[i])[:6]
    for mi in rank:
        wins = int((LL[mi, bl[mi]] > LL[0, bl[0]]).sum())
        say(f"  {names[mi]:36s} lam {LAM[bl[mi]]:g} LL/play {bestv[mi] / per:.5f} gain {(bestv[mi] - bestv[0]) / per:+.5f} fold wins {wins}/{len(seasons)}")
    flat = [(mi, li) for mi in range(len(names)) for li in range(len(LAM))]
    gains, wn = [], 0
    for si in range(len(seasons)):
        oth = [j for j in range(len(seasons)) if j != si]
        mi, li = max(flat, key=lambda a: LL[a[0], a[1], oth].mean())
        cl = max(range(len(LAM)), key=lambda l: LL[0, l, oth].mean())
        gn = LL[mi, li, si] - LL[0, cl, si]
        n_s = (T.season == seasons[si]).sum()
        gains.append(gn / n_s)
        wn += int(gn > 0)
        say(f"  nested-lite {seasons[si]}: {names[mi]} lam {LAM[li]:g} gain {gains[-1]:+.5f}")
    say(f"nested-lite mean gain {np.mean(gains):+.5f} wins {wn}/{len(seasons)}")
    top = rank[0]
    x = pd.cut(T.sc_raw, [-np.inf, -8.5, -0.5, 0.5, 8.5, np.inf], labels=False).to_numpy(float) - 2.0
    q = T.qtr_actual.to_numpy(float)
    say("return advance r=(100-fp)-next_yardline vs committing-team lead band (x, positive = committing team leads), quarter intercepts")
    say(f"  slope real {slope(x, rr, q):+.3f} current {slope(x, YH[0], q):+.3f} best {slope(x, YH[top], q):+.3f}")
    for b in (-2, -1, 0, 1, 2):
        m_ = x == b
        say(f"   band {b:+d} n {int(m_.sum()):5d} mean r real {rr[m_].mean():.2f} cur {YH[0][m_].mean():.2f} best {YH[top][m_].mean():.2f}")
    spec = {"name": names[top], "lam": LAM[bl[top]], "wins": int((LL[top, bl[top]] > LL[0, bl[0]]).sum()), "nested_wins": wn}
    (OUTD / "tov_spec.json").write_text(json.dumps(spec))
    (OUTD / "tov_fit.txt").write_text(chr(10).join(OUT))


def install_kos():
    import mod25d_variance as dv
    import mod25e_kick as mk

    spec = json.loads((OUTD / "kick_spec.json").read_text(encoding="utf-8"))
    if spec["fam"] != "gauss":
        raise RuntimeError("kick_spec is not a kernel draw")
    kt, tt, hk, ht = spec["p"][0], spec["p"][1], float(spec["p"][2]), float(spec["p"][3])
    lam = float(spec["lam"])
    cfg = dv._G["cfg"]
    mode = int(cfg.get("kick", 1))
    seed = int(cfg.get("seed", 3))
    K = pd.read_parquet(mk.OUT / f"kicks_{mk.POOL[0]}.parquet")
    fk = kd_feat(K.kd.to_numpy(float), kt)
    ft = t_feat(K.gsr.to_numpy(float), tt)
    ret = K.retained.to_numpy(int)
    nfp = K.nfp.to_numpy(float)
    rtd = K.rtd.to_numpy(int)
    pe = K.pat_extra.dropna().to_numpy(int)
    ok_all = np.ones(len(K), dtype=bool)
    ok_nt = rtd == 0
    state = {"k": None, "rng": None}

    def draw(rng, kdv, tv, mask):
        w = np.exp(-0.5 * ((fk - kd_feat(np.array([kdv]), kt)[0]) / hk) ** 2 - 0.5 * ((ft - t_feat(np.array([tv]), tt)[0]) / ht) ** 2)
        w = np.where(mask, w, 0.0) + np.where(mask, lam / mask.sum(), 0.0)
        c = np.cumsum(w)
        return int(min(np.searchsorted(c, rng.random() * c[-1]), len(c) - 1))

    base = dv._G["pol"]

    def pol(down, distance, yardline, score_diff, qtr, clock_val, drawn):
        loc = sys._getframe(1).f_locals
        gsr, offense, off_to, def_to = loc.get("gsr"), loc.get("offense"), loc.get("off_to"), loc.get("def_to")
        po = float(drawn["points_off"])
        pdf = float(drawn["points_def"])
        if mode == 0 or qtr > 4 or not ((po >= 3 and pdf == 0) or (pdf >= 6 and po == 0)):
            return base(down, distance, yardline, score_diff, qtr, clock_val, drawn)
        if state["k"] != dv._G.get("task_key"):
            state["k"] = dv._G.get("task_key")
            state["rng"] = dv.task_rng(15485863, seed)
        rng = state["rng"]
        k_off = po > 0
        kdv = score_diff + po if k_off else pdf - score_diff
        t = max(float(clock_val) - float(drawn["clock_elapsed"]), 0.0)
        i = draw(rng, kdv, t, ok_all)
        new = dict(drawn)
        holder_k = bool(ret[i])
        fp = nfp[i]
        if rtd[i]:
            b = int(pe[int(rng.integers(0, len(pe)))]) if len(pe) else 1
            if k_off:
                new["points_def"] = 6.0 + b
            else:
                new["points_off"] = 6.0 + b
            j = draw(rng, 6.0 + b - kdv, t, ok_nt)
            holder_k = not bool(ret[j])
            fp = nfp[j]
        holder_off = holder_k == k_off
        new["flip"] = not holder_off
        new["next_down"] = 1.0
        new["next_distance"] = min(10.0, float(fp))
        new["next_yardline"] = float(fp)
        return base(down, distance, yardline, score_diff, qtr, clock_val, new)

    dv._G["pol"] = pol


def install_tovret():
    import mod25d_variance as dv
    import sim04_engine as sim

    ns = dv._G["ns"]
    base = ns["pick_index_nn_conditioned"]
    K_TOV = 200
    box = {}

    def build(tbl):
        a = tbl["arrays"]
        tov = np.asarray(a["possession_flip"]).astype(bool) & (np.asarray(a["points_off"]) == 0) & (np.asarray(a["points_def"]) == 0) & np.isin(np.asarray(a["play_type_code"]), (0, 1))
        out = {}
        for d in (1, 2, 3, 4):
            feats, idx = [], []
            seen = set()
            for ph in range(5):
                e = tbl["nn_trees_cond"].get((d, ph))
                if e is None:
                    continue
                tree, sub = e
                keep = np.flatnonzero(tov[sub] & ~np.isin(sub, list(seen)))
                feats.append(tree.data[keep])
                idx.append(sub[keep])
                seen.update(sub[keep].tolist())
            if idx:
                from scipy.spatial import cKDTree

                f = np.vstack(feats)
                f[:, 2] = 0.0
                out[d] = (cKDTree(f), np.concatenate(idx))
        box["t"] = (tov, out)

    def pick(rng, tbl, down, phase, dist, fp, score, time_raw, off_to, def_to, k_state, off_sim, def_sim, is_home_sim):
        loc = sys._getframe(1).f_locals
        qtr, gsr, offense = loc.get("qtr"), loc.get("gsr"), loc.get("offense")
        i = base(rng, tbl, down, phase, dist, fp, score, time_raw, off_to, def_to, k_state, off_sim, def_sim, is_home_sim)
        if "t" not in box:
            build(tbl)
        tov, trees = box["t"]
        if not tov[i]:
            return i
        d = down if down in (1, 2, 3, 4) else 4
        e = trees.get(d)
        if e is None:
            return i
        tree, sub = e
        f = sim.feature_matrix(np.array([dist]), np.array([fp]), np.array([score]), np.array([time_raw]), np.array([off_to]), np.array([def_to]), np.array([phase]))[0]
        f[2] = 0.0
        _, ind = tree.query(f, k=min(K_TOV, len(sub)))
        return int(sub[np.atleast_1d(ind)[int(rng.integers(0, min(K_TOV, len(sub))))]])

    ns["pick_index_nn_conditioned"] = pick


def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else "kick"
    {"kick": fit_kick, "tov": fit_tov}[cmd]()


if __name__ == "__main__":
    main()
