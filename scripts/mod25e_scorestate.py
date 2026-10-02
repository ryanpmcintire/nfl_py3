import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

import mod25d_variance as dv  # noqa: E402

OUT = REPO / "artifacts" / "mod25e3"
SIMDIR = OUT / "crj_play"
WINDOWS = [("Q1", 2700, 3600), ("Q2a", 1920, 2700), ("Q2end", 1800, 1920), ("Q3", 900, 1800), ("Q4a>10m", 600, 900), ("Q4b10-5", 300, 600), ("Q4c5-2", 120, 300), ("Q4d<2", 0, 120)]
LEADS = [("trail11+", -99, -11), ("trail4-10", -10, -4), ("close<=3", -3, 3), ("lead4-10", 4, 10), ("lead11+", 11, 99)]


def real_frame(seasons):
    dv.sim.PBP_SNAPSHOT_DIR = dv.m25.SNAP
    pbp = dv.sim.load_reg_seasons(tuple(seasons))
    tr = dv.sim.build_transition_frame(pbp).sort_values(["game_id", "play_id"]).reset_index(drop=True)
    g, _ = pd.factorize(tr["game_id"])
    return pd.DataFrame({
        "sid": tr["game_id"].str[:4].astype(int).to_numpy(), "g": g, "down": tr["down_i"].to_numpy(), "dist": tr["dist_raw"].to_numpy(float),
        "yl": tr["fp_raw"].to_numpy(float), "sd": tr["sc_raw"].to_numpy(float), "gsr": tr["gsr_actual"].to_numpy(float), "qtr": tr["qtr_actual"].to_numpy(),
        "code": tr["play_type_code"].to_numpy(), "po": tr["points_off"].to_numpy(float), "pdf": tr["points_def"].to_numpy(float),
        "flip": tr["possession_flip"].to_numpy().astype(bool), "el": tr["clock_elapsed"].to_numpy(float),
    })


def sim_frame(src, minsid=2):
    fr = []
    for f in sorted(Path(src).glob("play_*_*.parquet")):
        _, w, s = f.stem.split("_")
        w, s = int(w), int(s)
        if s < minsid:
            continue
        d = pd.read_parquet(f)
        d["sid"] = w * 1000 + s + 1
        d["g"] = d["g"].astype(np.int64) + (w * 10 + s) * 100000
        fr.append(d)
    T = pd.concat(fr, ignore_index=True)
    T["flip"] = T["flip"].astype(bool)
    T["code"] = T["code"].astype(int)
    T["down"] = T["down"].astype(int)
    return T


def add_drives(T):
    T = T[T["qtr"] <= 4].reset_index(drop=True)
    g = T["g"].to_numpy()
    end = (T["flip"].to_numpy() | (T["po"].to_numpy() > 0) | (T["pdf"].to_numpy() > 0))
    newg = np.r_[True, g[1:] != g[:-1]]
    q = T["qtr"].to_numpy()
    half = np.r_[False, (q[:-1] <= 2) & (q[1:] >= 3)]
    start = newg | half | np.r_[True, end[:-1]]
    T["drive"] = np.cumsum(start)
    T["wi"] = win_idx(T["gsr"].to_numpy())
    T["li"] = lead_idx(T["sd"].to_numpy())
    return T


def win_idx(gsr):
    out = np.full(len(gsr), -1)
    for i, (_, lo, hi) in enumerate(WINDOWS):
        out[(gsr > lo) & (gsr <= hi)] = i
    out[gsr == 0] = len(WINDOWS) - 1
    return out


def lead_idx(sd):
    out = np.full(len(sd), -1)
    for i, (_, lo, hi) in enumerate(LEADS):
        out[(sd >= lo) & (sd <= hi)] = i
    return out


def play_cells(T):
    live = T[T["code"].isin([0, 1])]
    rows = {}
    for (w, l), d in live.groupby(["wi", "li"]):
        rows[(w, l)] = dict(n=len(d), run=float((d["code"] == 0).mean()), sec=float(d["el"].mean()))
    f4 = T[(T["down"] == 4) & T["code"].isin([0, 1, 2, 3])]
    for (w, l), d in f4.groupby(["wi", "li"]):
        rows.setdefault((w, l), {}).update(n4=len(d), go=float(d["code"].isin([0, 1]).mean()))
    return rows


def drive_cells(T):
    g = T.groupby("drive")
    D = pd.DataFrame({"dur": g["el"].sum(), "plays": g.size(), "pts": g["po"].sum() - g["pdf"].sum(), "wi": g["wi"].first(), "li": g["li"].first()})
    rows = {}
    for (w, l), d in D.groupby(["wi", "li"]):
        rows[(w, l)] = dict(nd=len(d), dur=float(d["dur"].mean()), dplays=float(d["plays"].mean()), pts=float(d["pts"].mean()))
    return rows, D


def cmd_profile(args):
    OUT.mkdir(parents=True, exist_ok=True)
    R = add_drives(real_frame(dv.TRAIN))
    S = add_drives(sim_frame(args.simdir))
    out = []
    for nm, T in (("real", R), ("sim", S)):
        pc = play_cells(T)
        dc, D = drive_cells(T)
        for k in set(pc) | set(dc):
            if k[0] < 0 or k[1] < 0:
                continue
            r = dict(src=nm, win=WINDOWS[k[0]][0], lead=LEADS[k[1]][0], wi=k[0], li=k[1])
            r.update(pc.get(k, {}))
            r.update(dc.get(k, {}))
            out.append(r)
    A = pd.DataFrame(out)
    A.to_csv(OUT / f"profile_{args.tag}.csv", index=False)
    pd.set_option("display.width", 250)
    for met in ("run", "sec", "go", "dur", "dplays", "pts"):
        if met not in A:
            continue
        P = A.pivot_table(index=["wi", "li"], columns="src", values=met)
        P["diff"] = P["sim"] - P["real"]
        P.index = [f"{WINDOWS[a][0]:8s} {LEADS[b][0]}" for a, b in P.index]
        print(met)
        print(P.round(3).to_string())
    print("games real", R["g"].nunique(), "sim", S["g"].nunique())


GW_AX = np.r_[np.arange(0, 901, 30), np.arange(960, 3601, 60)].astype(float)
POLICY = OUT / "css_policy.npz"
CAP = {}


def ensure_css():
    if POLICY.exists():
        return
    from sklearn.ensemble import HistGradientBoostingClassifier

    m25 = dv.m25
    dv.sim.PBP_SNAPSHOT_DIR = m25.SNAP
    pbp = dv.sim.load_reg_seasons(tuple(dv.TRAIN))
    trans = dv.sim.build_transition_frame(pbp)
    sd = np.clip(trans["sc_raw"].to_numpy(), -24, 24)
    gsr = trans["gsr_actual"].to_numpy()
    qtr = trans["qtr_actual"].to_numpy()
    down = trans["down_i"].to_numpy()
    code = trans["play_type_code"].to_numpy()
    otf = (trans["off_to_raw"].to_numpy() > 0).astype(float)
    dtf = (trans["def_to_raw"].to_numpy() > 0).astype(float)
    late = ((qtr == 4) & (gsr <= 600)) | ((qtr == 2) & (gsr <= 1980))
    m = (down <= 3) & (qtr <= 4) & ~late & np.isin(code, (0, 1, 3, 4, 5))
    X = np.column_stack([down, trans["dist_raw"].to_numpy(), trans["fp_raw"].to_numpy(), sd, gsr, otf, dtf])[m]
    y = np.array([m25.CODE_L[int(c)] for c in code[m]])
    clf = HistGradientBoostingClassifier(max_depth=5, max_iter=200, learning_rate=0.08, min_samples_leaf=200, random_state=1).fit(X, y)
    G = m25.grid_rows([1.0, 2.0, 3.0], m25.DL_AX, m25.YL_AX, m25.SD_AX, GW_AX, [0.0, 1.0], [0.0, 1.0])
    Q = m25.batched_proba(clf, G)
    P5 = np.zeros((len(Q), 5), dtype=np.float32)
    P5[:, clf.classes_.astype(int)] = Q
    P = P5.reshape(3, len(m25.DL_AX), len(m25.YL_AX), len(m25.SD_AX), len(GW_AX), 2, 2, 5)
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez(POLICY, P=P.astype(np.float32))
    print("css policy fit rows", int(m.sum()), flush=True)


def install_css(cfg):
    m25 = dv.m25
    c25 = dv.c25
    P = np.load(POLICY)["P"]
    orig = m25.make_decide
    if cfg.get("css_cls") or cfg.get("css_cls2"):
        kern = bool(cfg.get("css_cls2"))

        class FCache(dict):
            def __init__(self, base, codes):
                super().__init__()
                self.base = base
                self.codes = codes

            def get(self, key, default=None):
                r = dict.get(self, key)
                if r is not None:
                    return r
                nb = self.base.get(key)
                if nb is None:
                    return default
                cm = np.isin(dv._G["tables"]["arrays"]["play_type_code"][nb], self.codes)
                r = nb[cm] if cm.sum() >= 5 else nb
                dict.__setitem__(self, key, r)
                return r

        def make_decide(tables, trans, pol4, poll, mech, seed):
            base = orig(tables, trans, pol4, poll, mech, seed)
            arrays = tables["arrays"]
            rng2 = np.random.default_rng(seed + 555)
            sim = dv.sim
            st = {}
            if kern:
                tcls = []
                for codes in m25.CODES_L:
                    t2 = dict(tables)
                    t2["nn_cache_cond"] = FCache(tables["nn_cache_cond"], list(codes))
                    t2["nn_weight_cache_cond"] = {}
                    tcls.append(t2)
                st["t"] = tcls
                st["wrapped"] = False
            else:
                trees = m25.build_class_trees(trans)
                cache = {}
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
                            f = sim.feature_matrix(np.array([dist]), np.array([yl]), np.array([sd]), np.array([tf]), np.array([ot_]), np.array([dt_]), np.array([p]))
                            _, ind = tree.query(f, k=min(sim.K_NEIGHBORS, len(sub)))
                            nb = sub[ind[0]]
                            cache[key] = nb
                        return int(nb[rng.integers(len(nb))])
                    return None

            def decide(idx, rng, tbl, qtr, clock_val, in_ot, down, distance, yardline, score_diff, off_to, def_to, phase, time_feat):
                idx = base(idx, rng, tbl, qtr, clock_val, in_ot, down, distance, yardline, score_diff, off_to, def_to, phase, time_feat)
                code0 = int(arrays["play_type_code"][idx])
                if code0 == 6 or in_ot or down not in (1, 2, 3) or m25.in_late(qtr, clock_val):
                    return idx
                sdc = min(max(score_diff, -24.0), 24.0)
                p = P[down - 1, m25.nearest(m25.DL_AX, distance), m25.nearest(m25.YL_AX, yardline), m25.nearest(m25.SD_AX, sdc), m25.nearest(GW_AX, clock_val), 1 if off_to > 0 else 0, 1 if def_to > 0 else 0].astype(np.float64)
                c = np.cumsum(p)
                ci = min(int(np.searchsorted(c, rng2.random() * c[-1])), 4)
                if kern:
                    ns = dv._G["ns"]
                    if not st["wrapped"]:
                        f0 = ns["pick_index_nn_conditioned"]

                        def rec(*a):
                            st["ctx"] = a[11:14]
                            return f0(*a)

                        ns["pick_index_nn_conditioned"] = rec
                        st["f0"] = f0
                        st["wrapped"] = True
                    if "ctx" not in st:
                        return idx
                    tc = st["t"][ci]
                    if len(tc["nn_weight_cache_cond"]) > 150000:
                        tc["nn_weight_cache_cond"].clear()
                    return int(st["f0"](rng, tc, down, phase, distance, yardline, score_diff, time_feat, off_to, def_to, sim.K_STATE, *st["ctx"]))
                if ci == m25.CODE_L.get(code0, -1):
                    return idx
                j = cpick(rng, down, phase, ci, distance, yardline, score_diff, time_feat, off_to, def_to)
                return idx if j is None else j

            return decide

        m25.make_decide = make_decide
    if cfg.get("css_clk"):
        orig_v, orig_s = c25.zone_v, c25.zone_s

        def zone_v(qtr, g):
            z = orig_v(qtr, g)
            return np.where(np.asarray(qtr) == 3, 7, z).astype(np.int64)

        def zone_s(qtr, g):
            return 7 if qtr == 3 else orig_s(qtr, g)

        c25.zone_v, c25.zone_s = zone_v, zone_s


ORIG_GEN_INIT = dv.d_gen_init


def s_gen_init(setting):
    install_css(json.loads(setting["mech"]))
    ORIG_GEN_INIT(setting)


def install_lattice():
    import os

    base_pol = dv._G["pol"]
    arrays = dv._G["tables"]["arrays"]
    rng = np.random.default_rng(int(dv._G["cfg"].get("seed", 3)) * 7919 + os.getpid())

    def pol(down, distance, yardline, score_diff, qtr, clock_val, drawn):
        i = drawn["idx"]
        b = float(arrays["yards_gained"][i])
        x = float(drawn["yards_gained"])
        if x != b:
            drawn = dict(drawn)
            if b == 0.0 and int(drawn["play_type_code"]) == 1:
                drawn["yards_gained"] = 0.0
            else:
                lo = np.floor(x)
                drawn["yards_gained"] = float(lo + (rng.random() < x - lo))
        return base_pol(down, distance, yardline, score_diff, qtr, clock_val, drawn)

    dv._G["pol"] = pol


def s_init(setting):
    s_gen_init(setting)
    if dv._G["cfg"].get("intyd"):
        install_lattice()
    ns = dv._G["ns"]
    orig = ns["run_one_game"]
    CAP["rows"] = []

    def wrapped(*a, **k):
        rec, cap = orig(*a, **k)
        lg = np.array(dv._G["log"], dtype=np.float64).reshape(-1, 12)
        pl = np.array(ns["PLAYLOG"], dtype=np.float64).reshape(-1, 3)
        CAP["rows"].append((lg, pl, float(rec["total"]), float(rec["margin"])))
        return rec, cap

    ns["run_one_game"] = wrapped


def s_play_season(task):
    CAP["rows"].clear()
    res = dv.d_play_season(task)
    world, sidx = task[0], task[1]
    frames = []
    for gi, (lg, pl, tot, mar) in enumerate(CAP["rows"]):
        n = min(len(lg), len(pl))
        frames.append(np.column_stack([np.full(n, gi), lg[:n, :11], pl[:n, 0], np.full(n, tot), np.full(n, mar)]))
    cols = ["g", "down", "dist", "yl", "sd", "gsr", "qtr", "code", "po", "pdf", "flip", "el", "offhome", "total", "margin"]
    outd = Path(dv._G["cfg"]["outdir"])
    pd.DataFrame(np.concatenate(frames), columns=cols).to_parquet(outd / f"play_{world}_{sidx}.parquet")
    return res


DV = dv.DV
DV["css"] = dict(DV["crj"], css_cls=1, css_clk=1)
DV["css2"] = dict(DV["crj"], css_cls2=1, css_clk=1)
DV["cf4s"] = dict(DV["crf4"], css_cls2=1, css_clk=1)
DV["cscl"] = dict(DV["crj"], css_cls=1)
DV["cscc"] = dict(DV["crj"], css_clk=1)
DV["crl"] = dict(DV["crf4"], intyd=1)


def cmd_sim(args):
    import mod25_generator as gen

    ensure_css()
    outd = OUT / f"sim_{args.variant}"
    outd.mkdir(parents=True, exist_ok=True)
    dv.c25.ensure_policies()
    cfgj = json.dumps(dict(DV[args.variant], seed=args.seed, name=f"{args.variant}_s{args.scale:g}", outdir=str(outd)))
    setting = dict(gen.SETTING_DEFAULTS)
    setting.update({"scale": args.scale, "yard_gain": 0.0, "def_sign": 1.0, "yard_bias": 0.0, "drift": 1.0, "mech": cfgj})
    gen.init_worker = s_init
    gen.play_season = s_play_season
    games, plays, latents, el = gen.run_generation(setting, args.worlds, args.seasons, args.workers, args.seed, progress=True)
    games.to_parquet(outd / "sim_games.parquet")
    print("done", len(games), el, flush=True)


def cmd_budget(args):
    import mod25e_budget as bg

    bg.OUT = OUT / f"sim_{args.variant}"
    bg.cmd_budget(args)


def budget_row(D, X, re_):
    from mod25e_analysis import stats

    o = stats(D, X, re_)
    m3 = X["m3"].to_numpy(float)
    mr = np.where(np.isnan(X["mreg"].to_numpy(float)), X["margin"].to_numpy(float), X["mreg"].to_numpy(float))
    m3 = np.where(np.isnan(m3), mr, m3)
    inc4 = mr - m3
    slope = float(np.cov(inc4, m3)[0, 1] / m3.var(ddof=1))
    r = {"margin_var": o["margin_var"], "strength_re_var": o["strength_diff_var_RE"], "noise_re_var": o["noise_var_RE"], "reg_margin_var": o["reg_margin_var"]}
    r["xq_cov_sum"] = o["reg_margin_var"] - sum(o[f"q{q}_inc_var"] for q in (1, 2, 3, 4))
    r["q4_slope"] = slope
    r["drives_g"] = o["nposs_mean"]
    return r


def cmd_e5real(args):
    import mod25e_budget as bg

    out = {}
    for nm, ss in (("real_eval", dv.EVAL), ("real_train", dv.TRAIN)):
        T, G = bg.build_real(ss)
        from mod25e_analysis import per_game, season_re

        D, X = per_game(T, G)
        out[nm] = budget_row(D, X, list(season_re(G).values()))
    (OUT / "e5_real.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


def cmd_e5(args):
    import mod25_generator as gen
    import mod25e_budget as bg
    from mod25e_analysis import per_game, season_re

    ensure_css()
    outd = OUT / f"e5_{args.variant}_s{args.seed}"
    outd.mkdir(parents=True, exist_ok=True)
    dv.c25.ensure_policies()
    cfgj = json.dumps(dict(DV[args.variant], seed=args.seed, name=f"{args.variant}_s{args.scale:g}", outdir=str(outd)))
    setting = dict(gen.SETTING_DEFAULTS)
    setting.update({"scale": args.scale, "yard_gain": 0.0, "def_sign": 1.0, "yard_bias": 0.0, "drift": 1.0, "mech": cfgj})
    gen.init_worker = s_init
    gen.play_season = s_play_season
    games, plays, latents, el = gen.run_generation(setting, args.worlds, args.seasons, args.workers, args.seed, progress=False)
    games.to_parquet(outd / "sim_games.parquet")
    ts = gen.team_stats_from_plays(plays)
    keep = [w * 1000 + s + 1 for w in range(args.worlds) for s in range(2, args.seasons)]
    m, ac, dd = gen.all_metrics(games, ts, keep)
    m["r2_pooled"] = float(np.corrcoef(dd["x"], dd["y"])[0, 1] ** 2)
    m["nonstrength_var"] = float(m["margin_sd"] ** 2 * (1 - m["r2_pooled"]))
    m["pts_game"] = float((games["home_score"] + games["away_score"]).mean())
    bg.OUT = outd
    T, G = bg.build_sim(2)
    D, X = per_game(T, G)
    b = budget_row(D, X, list(season_re(G).values()))
    (outd / "e5.json").write_text(json.dumps({"gate": m, "budget": b, "elapsed": el}, indent=1, default=float))
    print("done", args.variant, args.seed, el, flush=True)


def cmd_e5report(args):
    import glob

    real = json.loads((OUT / "e5_real.json").read_text())["real_eval"]
    rg = json.loads((OUT.parent / "mod25d" / "gate_e4a.json").read_text())["real_2018_2025"]
    rows = {}
    for v in args.variants.split(","):
        runs = [json.loads(Path(f).read_text()) for f in sorted(glob.glob(str(OUT / f"e5_{v}_s*" / "e5.json")))]
        rows[v] = runs
    keys = [("gate", k) for k in ("margin_sd", "nonstrength_var", "r2_w10_18", "r2_w5_9", "r2_w1_4", "epa_autocorr_lag1", "pts_game")] + [("budget", k) for k in real]
    print("metric real " + " ".join(f"{v}(mean sd n)" for v in rows))
    for sec, k in keys:
        rv = real.get(k) if sec == "budget" else rg.get(k)
        cells = []
        for v, runs in rows.items():
            x = np.array([r[sec][k] for r in runs if k in r[sec]], float)
            cells.append(f"{x.mean():.4f} {x.std(ddof=1) if len(x) > 1 else float('nan'):.4f} {len(x)}")
        print(sec, k, rv if rv is None else round(float(rv), 4), " | ".join(cells))


def cmd_gate(args):
    ensure_css()
    dv.c25.ensure_policies()
    dv.d_gen_init = s_gen_init
    dv.OUT = OUT
    dv.cmd_gate(args)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("e5real")
    r5 = sub.add_parser("e5report")
    r5.add_argument("--variants", default="crf4,cf4s")
    e5 = sub.add_parser("e5")
    e5.add_argument("--variant", default="cf4s")
    e5.add_argument("--scale", type=float, default=1.0)
    e5.add_argument("--worlds", type=int, default=8)
    e5.add_argument("--seasons", type=int, default=8)
    e5.add_argument("--workers", type=int, default=4)
    e5.add_argument("--seed", type=int, default=11)
    p = sub.add_parser("profile")
    p.add_argument("--simdir", default=str(SIMDIR))
    p.add_argument("--tag", default="crj")
    s = sub.add_parser("sim")
    s.add_argument("--variant", default="css")
    s.add_argument("--scale", type=float, default=1.0)
    s.add_argument("--worlds", type=int, default=6)
    s.add_argument("--seasons", type=int, default=8)
    s.add_argument("--workers", type=int, default=4)
    s.add_argument("--seed", type=int, default=9)
    b = sub.add_parser("budget")
    b.add_argument("--variant", default="css")
    b.add_argument("--boot", type=int, default=400)
    g = sub.add_parser("gate")
    g.add_argument("--variants", default="css")
    g.add_argument("--scale", type=float, default=1.0)
    g.add_argument("--worlds", type=int, default=8)
    g.add_argument("--seasons", type=int, default=8)
    g.add_argument("--workers", type=int, default=4)
    g.add_argument("--seed", type=int, default=9)
    g.add_argument("--yard-gain", dest="yard_gain", type=float, default=0.0)
    g.add_argument("--yard-bias", dest="yard_bias", type=float, default=0.0)
    g.add_argument("--tag", default="s1")
    args = ap.parse_args()
    {"profile": cmd_profile, "sim": cmd_sim, "budget": cmd_budget, "gate": cmd_gate, "e5": cmd_e5, "e5real": cmd_e5real, "e5report": cmd_e5report}[args.cmd](args)


if __name__ == "__main__":
    main()
