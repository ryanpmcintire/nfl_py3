import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
ART = REPO / "artifacts" / "mod25e3"
OUTD = ART / "adj"
FIT = OUTD / "fit.json"
RAW = REPO / "data" / "pbp" / "raw" / "20260925T202544Z"
BANDS = [-8.5, -0.5, 0.5, 8.5]
KAPPAS = (0.0, 10.0, 25.0, 50.0, 100.0, float("inf"))
BURN = 2
LABEL = os.environ.get("CU_LABEL", "crHpqokgndecsmfwtjo2")
SIM_SEEDS = tuple(int(x) for x in os.environ.get("ADJ_SEEDS", "11").split(","))
LAMBDA_MAX = 30.0
RP = (0, 1)


def enabled():
    return os.environ.get("ADJ") == "1"


def pool_epa():
    import sim09_f2 as f2

    dv = f2.dv
    dv.sim.PBP_SNAPSHOT_DIR = dv.m25.SNAP
    pbp = dv.sim.load_reg_seasons(tuple(dv.TRAIN))
    tr = dv.sim.build_transition_frame(pbp)
    ep = pd.concat([pd.read_parquet(f"{dv.m25.SNAP}/season={y}/plays.parquet", columns=["game_id", "play_id", "epa"]) for y in dv.TRAIN]).drop_duplicates(["game_id", "play_id"])
    m = tr[["game_id", "play_id"]].merge(ep, on=["game_id", "play_id"], how="left")
    assert len(m) == len(tr)
    return m["epa"].fillna(0.0).to_numpy(dtype=np.float64)


def real_frame():
    out = []
    for s in range(2009, 2018):
        f = pd.read_parquet(RAW / f"season={s}" / "plays.parquet", columns=["game_id", "play_id", "season_type", "home_team", "away_team", "posteam", "play_type", "epa", "qtr", "score_differential"])
        f = f[(f.season_type == "REG") & f.play_type.isin(["run", "pass"]) & (f.qtr <= 4) & f.posteam.notna()].sort_values(["game_id", "play_id"], kind="stable")
        home = (f.posteam == f.home_team).to_numpy()
        out.append(pd.DataFrame({"gk": str(s) + "_" + f.game_id.to_numpy(), "side": np.where(home, 0, 1), "q": f.qtr.astype(int).to_numpy() - 1, "epa": f.epa.fillna(0.0).to_numpy(),
                                 "sd": f.score_differential.fillna(0.0).to_numpy(), "tk": f.posteam.to_numpy(), "dk": np.where(home, f.away_team, f.home_team), "sk": str(s)}))
    return pd.concat(out, ignore_index=True)


def sim_frame():
    epa_by_idx = pool_epa()
    out = []
    for sd in SIM_SEEDS:
        root = ART / f"e5_{LABEL}_s{sd}"
        sg = pd.read_parquet(root / "sim_games.parquet", columns=["game_id", "home_team", "away_team"])
        sg["w"] = sg.game_id.str[1:5].astype(int)
        sg["s"] = sg.game_id.str[6:8].astype(int)
        sg["g"] = sg.game_id.str[-3:].astype(int)
        for (w, s), tm in sg.groupby(["w", "s"]):
            if s < BURN:
                continue
            d = pd.read_parquet(root / f"play_{w}_{s}.parquet", columns=["g", "qtr", "offhome", "code", "idx", "sd"])
            d = d[(d.qtr <= 4) & d.code.isin(RP)].sort_values("g", kind="stable")
            t = tm.set_index("g")
            gi = d.g.astype(int).to_numpy()
            ht = t.home_team.reindex(gi).to_numpy()
            at = t.away_team.reindex(gi).to_numpy()
            oh = d.offhome.to_numpy() == 1
            key = f"{sd}_{w}_{s}"
            out.append(pd.DataFrame({"gk": key + "_" + d.g.astype(int).astype(str).to_numpy(), "side": np.where(oh, 0, 1), "q": d.qtr.astype(int).to_numpy() - 1, "epa": epa_by_idx[d.idx.astype(int).to_numpy()],
                                     "sd": d.sd.to_numpy(float), "tk": np.where(oh, ht, at).astype(str), "dk": np.where(oh, at, ht).astype(str), "sk": key, "grp": w}))
    return pd.concat(out, ignore_index=True)


def lgo(df, col, key):
    gs = df.groupby(["sk", key])[col].transform("sum")
    gn = df.groupby(["sk", key])[col].transform("size")
    ggs = df.groupby(["gk", key])[col].transform("sum")
    ggn = df.groupby(["gk", key])[col].transform("size")
    n = gn - ggn
    return np.where(n > 0, (gs - ggs) / np.maximum(n, 1), 0.0)


def residual(df):
    cell = df.q.to_numpy() * 5 + np.digitize(df.sd.to_numpy(), BANDS)
    df = df.assign(cell=cell)
    df["e"] = df.epa - df.groupby("cell").epa.transform("mean")
    df["r"] = df.e - lgo(df, "e", "tk") - lgo(df, "e", "dk")
    return df


def history(df):
    df = df.reset_index(drop=True)
    r = df.r.to_numpy()
    side = df.side.to_numpy()
    gk = df.gk.to_numpy()
    new = np.r_[True, gk[1:] != gk[:-1]]
    gid = np.cumsum(new) - 1
    sums, cnts = [], []
    for s in (0, 1):
        c = np.cumsum(np.where(side == s, r, 0.0))
        n = np.cumsum((side == s).astype(float))
        base_c = np.r_[0.0, c][np.flatnonzero(new)][gid]
        base_n = np.r_[0.0, n][np.flatnonzero(new)][gid]
        sums.append(c - base_c - np.where(side == s, r, 0.0))
        cnts.append(n - base_n - (side == s))
    S = np.stack(sums, 1)
    N = np.stack(cnts, 1)
    ar = np.arange(len(df))
    df["sumB"], df["nB"] = S[ar, side], N[ar, side]
    df["sumA"], df["nA"] = S[ar, 1 - side], N[ar, 1 - side]
    return df


def prep(df):
    d = history(residual(df))
    return d[(d.q >= 1) & (d.nA >= 1)].reset_index(drop=True)


def zfeat(sm, n, kappa):
    return sm if np.isinf(kappa) else sm / (n + kappa)


def design(d, kappa, byq, own):
    q = d.q.to_numpy()
    cols = [np.ones(len(d))]
    for use, sm, n in ((True, d.sumA.to_numpy(), d.nA.to_numpy()), (own, d.sumB.to_numpy(), d.nB.to_numpy())):
        if not use:
            continue
        z = zfeat(sm, n, kappa)
        if byq:
            cols += [z * (q == j) for j in (1, 2, 3)]
        else:
            cols.append(z)
    return np.stack(cols, 1)


def ols(X, y):
    return np.linalg.lstsq(X, y, rcond=None)[0]


def loso(d, forms):
    seasons = sorted(d.sk.unique())
    y = d.r.to_numpy()
    sk = d.sk.to_numpy()
    res = {}
    for name, (kappa, byq, own) in forms.items():
        sse = 0.0
        for s in seasons:
            te = sk == s
            if name == "const":
                X = np.ones((len(d), 1))
            else:
                X = design(d, kappa, byq, own)
            b = ols(X[~te], y[~te])
            sse += float(((y[te] - X[te] @ b) ** 2).sum())
        res[name] = sse
    return res


def forms_grid():
    f = {"const": (0.0, False, False)}
    for k in KAPPAS:
        for byq in (False, True):
            for own in (False, True):
                f[f"k{k:g}|{'byq' if byq else 'pool'}|{'own' if own else 'opp'}"] = (k, byq, own)
    return f


def coef_dict(b, byq, own):
    names = []
    if byq:
        names = [("opp", j) for j in (1, 2, 3)] + ([("own", j) for j in (1, 2, 3)] if own else [])
    else:
        names = [("opp", 0)] + ([("own", 0)] if own else [])
    out = {"opp": {}, "own": {}}
    for (w, j), v in zip(names, b[1:]):
        if byq:
            out[w][str(j + 1)] = float(v)
        else:
            for qq in (2, 3, 4):
                out[w][str(qq)] = float(v)
    return out


def cmd_fit():
    OUTD.mkdir(parents=True, exist_ok=True)
    R = prep(real_frame())
    S = prep(sim_frame())
    forms = forms_grid()
    res = loso(R, forms)
    base = res["const"]
    lines = [f"E87 matchup adjustment fit; real plays {len(R)} (games {R.gk.nunique()}), sim plays {len(S)} (games {S.gk.nunique()}); looks {len(forms)}; LOSO 9 seasons; residual = EPA minus quarter x lead-band mean minus leave-game-out team-season off/def means"]
    for k, v in sorted(res.items(), key=lambda kv: kv[1]):
        lines.append(f"  {k:24s} held-out SSE {v:.3f}  gain vs const {base - v:+.3f} ({(base - v) / base * 1e4:+.2f} bp)")
    best = min(res, key=res.get)
    kappa, byq, own = forms[best]
    yR, XR = R.r.to_numpy(), design(R, kappa, byq, own)
    bR = ols(XR, yR)
    folds = []
    for s in sorted(R.sk.unique()):
        te = R.sk.to_numpy() == s
        folds.append(ols(XR[~te], yR[~te]))
    folds = np.array(folds)
    yS, XS = S.r.to_numpy(), design(S, kappa, byq, own)
    bS = ols(XS, yS)
    gS = S.grp.to_numpy()
    sf = np.array([ols(XS[gS != g], yS[gS != g]) for g in sorted(np.unique(gS))])
    cR, cS = coef_dict(bR, byq, own), coef_dict(bS, byq, own)
    delta = {w: {q: cR[w][q] - cS[w].get(q, 0.0) for q in cR[w]} for w in cR}
    lines.append(f"best {best}")
    lines.append(f"  real coef {np.round(bR[1:], 5).tolist()} per-fold sd {np.round(folds[:, 1:].std(0), 5).tolist()} (se of mean about {np.round(folds[:, 1:].std(0) / 3, 5).tolist()})")
    lines.append(f"  sim coef  {np.round(bS[1:], 5).tolist()} leave-world-out sd {np.round(sf[:, 1:].std(0), 5).tolist()}")
    lines.append(f"  delta (real - sim) {json.dumps(delta)}")
    spec = dict(form=best, kappa=None if np.isinf(kappa) else kappa, kappa_inf=bool(np.isinf(kappa)), byq=byq, own=own, real=cR, sim=cS, delta=delta, heldout_sse=res, looks=len(forms))
    FIT.write_text(json.dumps(spec), encoding="utf-8")
    (OUTD / "fit.txt").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


def load_spec():
    sp = json.loads(FIT.read_text(encoding="utf-8"))
    kappa = float("inf") if sp["kappa_inf"] else float(sp["kappa"])
    return sp, kappa


def solve_tilt(w, e, target):
    lw = np.log(np.maximum(w, 1e-300))

    def mean_at(lam):
        x = lw + lam * e
        x = x - x.max()
        p = np.exp(x)
        return float((p * e).sum() / p.sum())

    lo, hi = -LAMBDA_MAX, LAMBDA_MAX
    if target >= mean_at(hi):
        return hi
    if target <= mean_at(lo):
        return lo
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if mean_at(mid) < target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def tilt_pool(nb, w, epa_nb, code_nb, shift):
    rp = np.isin(code_nb, RP)
    if rp.sum() < 2:
        return None
    wr = w[rp]
    er = epa_nb[rp]
    m0 = float((wr * er).sum() / wr.sum())
    if shift == 0.0:
        return rp, m0, w, 0.0
    lam = solve_tilt(wr, er - m0, shift)
    x = np.log(np.maximum(wr, 1e-300)) + lam * (er - m0)
    p = np.exp(x - x.max())
    p = p * (wr.sum() / p.sum())
    w2 = w.copy()
    w2[rp] = p
    return rp, m0, w2, lam


def install_adj():
    import mod25d_variance as dv

    sp, kappa = load_spec()
    ns = dv._G["ns"]
    t = dv._G["tables"]
    epa = pool_epa()
    codes = np.asarray(t["arrays"]["play_type_code"])
    assert len(epa) == len(codes)
    delta = {w: {int(q): v for q, v in sp["delta"][w].items()} for w in ("opp", "own")}
    use_own = bool(sp["own"])
    base = ns["pick_index_nn_conditioned"]
    st = {"fr": None, "sum": np.zeros(2), "n": np.zeros(2)}
    stats = {"draws": 0, "tilted": 0, "capped": 0, "n": 0.0, "sx": 0.0, "sy": 0.0, "sxx": 0.0, "sxy": 0.0}
    stat_file = OUTD / "smoke" / f"adj_stats_{os.getpid()}.json"
    stat_file.parent.mkdir(parents=True, exist_ok=True)
    dv._G["adj_stats"] = stats

    def pick(rng, tables, down, phase, dist, fp, score, time_raw, off_to, def_to, k_state, off_sim, def_sim, is_home_sim):
        fr = sys._getframe(1)
        while fr is not None and "offense" not in fr.f_locals:
            fr = fr.f_back
        offense, qtr, gsr = fr.f_locals["offense"], fr.f_locals["qtr"], fr.f_locals.get("gsr")  # noqa: F841
        off_to_l, def_to_l, possessions = fr.f_locals.get("off_to"), fr.f_locals.get("def_to"), fr.f_locals.get("possessions")  # noqa: F841
        idx = base(rng, tables, down, phase, dist, fp, score, time_raw, off_to, def_to, k_state, off_sim, def_sim, is_home_sim)
        if int(down) > 3 or phase >= 4:
            return idx
        if st["fr"] is not fr:
            st["fr"] = fr
            st["sum"][:] = 0.0
            st["n"][:] = 0.0
        b = 0 if offense == "home" else 1
        a = 1 - b
        key = ns["round_state_key"](down if down in (1, 2, 3, 4) else 4, phase, dist, fp, score, time_raw, off_to, def_to)
        nb = tables["nn_cache_cond"].get(key)
        cw = tables["nn_weight_cache_cond"].get((key, off_sim, def_sim, is_home_sim))
        if nb is None or cw is None:
            return idx
        cdf, _ = cw
        w = np.diff(np.r_[0.0, cdf])
        shift = 0.0
        if 2 <= int(qtr) <= 4:
            if st["n"][a] >= 1:
                shift += delta["opp"][int(qtr)] * zfeat(st["sum"][a], st["n"][a], kappa)
            if use_own and st["n"][b] >= 1:
                shift += delta["own"][int(qtr)] * zfeat(st["sum"][b], st["n"][b], kappa)
        out = tilt_pool(nb, w, epa[nb], codes[nb], shift)
        if out is None:
            return idx
        rp, m0, w2, lam = out
        stats["draws"] += 1
        if shift != 0.0:
            stats["tilted"] += 1
            stats["capped"] += int(abs(lam) >= LAMBDA_MAX)
            c2 = np.cumsum(w2)
            idx = int(nb[min(int(np.searchsorted(c2, rng.random() * c2[-1], side="right")), len(nb) - 1)])
        if codes[idx] in RP:
            if 2 <= int(qtr) <= 4 and st["n"][a] >= 1:
                x, y = st["sum"][a], epa[idx] - m0
                stats["n"] += 1.0
                stats["sx"] += x
                stats["sy"] += y
                stats["sxx"] += x * x
                stats["sxy"] += x * y
            if stats["draws"] % 2000 == 0:
                stat_file.write_text(json.dumps(stats), encoding="utf-8")
            st["sum"][b] += epa[idx] - m0
            st["n"][b] += 1.0
        return idx

    ns["pick_index_nn_conditioned"] = pick


def cmd_val():
    import mod25d_variance as dv
    import mod25e_crH as H

    sp, kappa = load_spec()
    cfg = H.flags("")
    cfg.update(condition=1, def_sign=1.0, yard_bias=0.0)
    dv._d_init(dv.TRAIN, json.dumps(cfg))
    import sim04_engine as sim

    t = dv._G["tables"]
    ns = dv._G["ns"]
    a = t["arrays"]
    epa = pool_epa()
    codes = np.asarray(a["play_type_code"])
    pbp = sim.load_reg_seasons(tuple(dv.TRAIN))
    trans = sim.build_transition_frame(pbp)
    rng = np.random.default_rng(87)
    rows = np.flatnonzero((np.asarray(a["down_i"]) <= 3) & np.isin(codes, RP) & (trans["phase"].to_numpy() < 4) & (trans["qtr_actual"].to_numpy() >= 2) & (trans["qtr_actual"].to_numpy() <= 4))
    rows = rng.choice(rows, 300, replace=False)
    lines = [f"E87 offline replay on {len(rows)} real states (2009-17 pool rows, own ratings); target shift = delta_q x z at real z quantiles"]
    zs = np.array([-0.3, -0.1, 0.0, 0.1, 0.3])
    dq = np.mean([sp["delta"]["opp"][str(q)] for q in (2, 3, 4)])
    res = {z: [] for z in zs}
    for i in rows:
        down = int(a["down_i"][i])
        phase = int(trans["phase"].to_numpy()[i])
        args = (down, phase, float(a["dist_raw"][i]), float(a["fp_raw"][i]), float(trans["sc_raw"].to_numpy()[i]), float(trans["time_raw"].to_numpy()[i]), float(trans["off_to_raw"].to_numpy()[i]), float(trans["def_to_raw"].to_numpy()[i]))
        off_sim, def_sim, home = float(a["off_row"][i]), float(a["def_row"][i]), int(a["is_home_off"][i])
        ns["pick_index_nn_conditioned"](rng, t, *args, ns["K_STATE"], off_sim, def_sim, home)
        key = ns["round_state_key"](args[0], args[1], args[2], args[3], args[4], args[5], args[6], args[7])
        nb = t["nn_cache_cond"][key]
        cdf, _ = t["nn_weight_cache_cond"][(key, off_sim, def_sim, home)]
        w = np.diff(np.r_[0.0, cdf])
        for z in zs:
            tgt = dq * z
            out = tilt_pool(nb, w, epa[nb], codes[nb], tgt)
            if out is None:
                continue
            rp, m0, w2, lam = out
            m1 = float((w2[rp] * epa[nb][rp]).sum() / w2[rp].sum())
            mix0 = float(w[rp].sum() / w.sum())
            mix1 = float(w2[rp].sum() / w2.sum())
            res[z].append((tgt, m1 - m0, mix1 - mix0, abs(lam) >= LAMBDA_MAX))
    for z in zs:
        r = np.array(res[z], dtype=float)
        lines.append(f"  z {z:+.2f}: target shift {r[:, 0].mean():+.5f} achieved mean {r[:, 1].mean():+.5f} max abs err {np.abs(r[:, 1] - r[:, 0]).max():.2e}; capped {int(r[:, 3].sum())}/{len(r)}; run/pass share change {r[:, 2].mean():+.2e}")
    lines.append(f"  looks {len(zs)}")
    (OUTD / "val.txt").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


def adj_budget(setting):
    import mod25e_crH as H

    H.H_budget(setting)
    if enabled():
        install_adj()


def cmd_smoke():
    import mod25e_crG as crG
    import mod25e_crH as H

    orig = H.install

    def inst():
        orig()
        crG.crG_budget = adj_budget

    H.install = inst
    sys.argv = ["mod25e_crH.py", "sim", "--workers", "1", "--out-dir", str(OUTD / "smoke")]
    H.main()


if __name__ == "__main__":
    {"fit": cmd_fit, "val": cmd_val}[sys.argv[1]]()
