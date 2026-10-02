import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

SYN = REPO / "data" / "processed" / "synthetic" / "crp04"
OUT = REPO / "artifacts" / "mod25_pipeline"
VARIANT = "crp04"
SCALE = 0.75
YARD_GAIN = 2.0
YARD_BIAS = 0.75
GEN_FILES = ("mod25d_variance.py", "mod25c_noise.py", "mod25_mechanisms.py", "mod25_generator.py", "sim04_engine.py")


def gen_hashes():
    return {f: hashlib.sha256((REPO / "scripts" / f).read_bytes()).hexdigest()[:16] for f in GEN_FILES}


def produce_chunk(k, worlds, seasons, workers):
    import mod25d_variance as d
    import mod25c_noise as c25
    import mod25_generator as gen

    d.c25.ensure_policies()
    seed = 7000 + k
    cfgj = json.dumps(dict(d.DV[VARIANT], seed=seed, name=f"{VARIANT}_s{SCALE:g}_k{k}"))
    setting = dict(gen.SETTING_DEFAULTS)
    setting.update({"scale": SCALE, "yard_gain": YARD_GAIN, "def_sign": 1.0, "yard_bias": YARD_BIAS, "drift": 1.0, "mech": cfgj})
    gen.init_worker = d.d_gen_init
    gen.play_season = d.d_play_season
    games, plays, latents, elapsed = gen.run_generation(setting, worlds, seasons, workers, seed, progress=False)
    pre = f"c{k:02d}"
    games["game_id"] = pre + games["game_id"]
    games["home_team"] = pre + games["home_team"]
    games["away_team"] = pre + games["away_team"]
    plays["game_id"] = pre + plays["game_id"]
    plays["posteam"] = pre + plays["posteam"]
    ts = gen.team_stats_from_plays(plays)
    d_ = SYN / f"chunk_{k:02d}"
    d_.mkdir(parents=True, exist_ok=True)
    games["world"] = games["game_id"].str.slice(0, 8)
    games["sidx"] = games["season"] % 1000 - 1
    games["chunk"] = k
    games.to_parquet(d_ / "games.parquet")
    ts.to_parquet(d_ / "team_stats.parquet")
    plays.astype({"epa": "float32", "yards_gained": "float32"}).to_parquet(d_ / "plays.parquet")
    lat = {f"{pre}_w{w}_s{s}_{n}": a for (w, s), (wk, qb) in latents.items() for n, a in (("weekly", wk), ("qb_out", qb))}
    np.savez_compressed(d_ / "latents.npz", **lat)
    meta = dict(k=k, seed=seed, worlds=worlds, seasons=seasons, games=len(games), plays=len(plays), elapsed=elapsed, scale=SCALE, variant=VARIANT, gen_sha=gen_hashes(), mech=cfgj)
    (d_ / "meta.json").write_text(json.dumps(meta, indent=1))
    return meta


def cmd_produce(args):
    t0 = time.time()
    k = args.start
    metas = []
    per = None
    while True:
        spent = (time.time() - t0) / 60.0
        if args.minutes and k > args.start and per and spent + per > args.minutes:
            break
        if k - args.start >= args.max_chunks:
            break
        t1 = time.time()
        m = produce_chunk(k, args.worlds, args.seasons, args.workers)
        per = (time.time() - t1) / 60.0
        print(f"chunk {k} games {m['games']} {per:.1f} min", flush=True)
        metas.append(m)
        k += 1
    (OUT / "produce_log.json").write_text(json.dumps({"chunks": [m["k"] for m in metas], "minutes": (time.time() - t0) / 60.0}, indent=1))


ZERO_FILL = ("rest_diff", "neutral_site", "div_game", "temp", "wind", "home_off_cpoe", "away_off_cpoe", "diff_off_cpoe")
BOOK_PREFIXES = ("offense", "defense")


def load_chunks(ids=None):
    parts = []
    for d_ in sorted(SYN.glob("chunk_*")):
        k = int(d_.name.split("_")[1])
        if ids is not None and k not in ids:
            continue
        if not (d_ / "meta.json").exists():
            continue
        parts.append((k, pd.read_parquet(d_ / "games.parquet"), pd.read_parquet(d_ / "team_stats.parquet")))
    return parts


def schedule_frame(g):
    s = g[["game_id", "season", "week", "gameday", "home_team", "away_team", "home_score", "away_score"]].copy()
    s["season"] = 2000 + (s["season"] % 1000 - 1)
    s["game_type"] = "REG"
    s["result"] = s["home_score"] - s["away_score"]
    s["spread_line"] = np.nan
    s["total_line"] = np.nan
    s["gameday"] = pd.to_datetime(s["gameday"])
    return s


def book_columns(cols):
    from nfl_ats.margin import margin_feature_columns, resolve_feature_groups

    full = list(margin_feature_columns("market_residual", "weak_stack"))
    fam = dict(zip(full, resolve_feature_groups(full), strict=True))
    keep = [c for c in full if fam[c] in ("offense", "defense", "elo") and c not in ZERO_FILL and c != "elo_home_win_prob"]
    keep += ["home_point_diff", "away_point_diff", "diff_point_diff"]
    return [c for c in keep if c in cols]


def build_world(sched, stats):
    from nfl_ats.features import build_game_features

    return build_game_features(sched, stats)


FEAT_ID = ["game_id", "season", "week", "gameday", "home_team", "away_team", "home_score", "away_score", "result", "spread_line", "total_line", "ats_margin", "world", "chunk", "sidx"]


def real_frame():
    f = pd.read_parquet(REPO / "data" / "processed" / "game_features_weak_stack.parquet")
    f = f.loc[f["game_type"].eq("REG")].copy()
    f["gameday"] = pd.to_datetime(f["gameday"])
    return f


def make_pipe(alpha=10.0):
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    return Pipeline([("i", SimpleImputer(strategy="median")), ("s", StandardScaler()), ("r", Ridge(alpha=alpha))])


def fit_book():
    f = real_frame()
    sbr = pd.read_parquet(REPO / "data" / "processed" / "sbr_odds.parquet").dropna(subset=["game_id", "open_home_spread"]).drop_duplicates("game_id")
    d = f.merge(sbr[["game_id", "open_home_spread", "open_total"]], on="game_id", how="inner")
    d = d.loc[d["season"].between(2009, 2019)].reset_index(drop=True)
    cols = book_columns(d.columns)
    out = {"cols": cols, "n": len(d)}
    for name, tgt in (("spread", "open_home_spread"), ("total", "open_total")):
        y = d[tgt].to_numpy(float)
        ok = np.isfinite(y)
        X = d.loc[ok, cols]
        pipe = make_pipe().fit(X, y[ok])
        pred = pipe.predict(X)
        loso = np.full(ok.sum(), np.nan)
        se = d.loc[ok, "season"].to_numpy()
        for s_ in sorted(set(se)):
            tr = se != s_
            loso[~tr] = make_pipe().fit(X[tr], y[ok][tr]).predict(X[~tr])
        yy = y[ok]
        out[name] = {
            "pipe": pipe,
            "target_sd": float(yy.std()),
            "r2_in": float(1 - ((yy - pred) ** 2).sum() / ((yy - yy.mean()) ** 2).sum()),
            "r2_loso": float(1 - ((yy - loso) ** 2).sum() / ((yy - yy.mean()) ** 2).sum()),
            "resid_sd_loso": float((yy - loso).std()),
            "pred_sd": float(pred.std()),
        }
    return out


_W = {}


def _world_init(book):
    _W["book"] = book


def _world_pass1(task):
    k, wname, sched, stats = task
    book = _W["book"]
    f1 = build_world(sched, stats)
    X = f1[book["cols"]]
    return pd.DataFrame({"game_id": f1["game_id"], "sp": book["spread"]["pipe"].predict(X), "tt": book["total"]["pipe"].predict(X), "result": f1["result"], "pts": f1["home_score"] + f1["away_score"]})


def _world_pass2(task):
    k, wname, sched, stats, lines = task
    s2 = sched.copy()
    s2["spread_line"] = s2["game_id"].map(lines["sp"])
    s2["total_line"] = s2["game_id"].map(lines["tt"])
    f2 = build_world(s2, stats)
    f2["world"] = wname
    f2["chunk"] = k
    return f2


def cmd_features(args):
    from nfl_ats.margin import margin_feature_columns

    book = fit_book()
    full = list(margin_feature_columns("market_residual", "weak_stack"))
    tasks = []
    for k, g, ts in load_chunks():
        for wname, w in g.groupby("world"):
            tasks.append((k, wname, schedule_frame(w), ts[ts["game_id"].isin(set(w["game_id"]))]))
    import multiprocessing as mp

    t0 = time.time()
    with mp.get_context("spawn").Pool(args.workers, initializer=_world_init, initargs=(book,)) as pool:
        p1 = pd.concat(pool.map(_world_pass1, tasks, chunksize=1), ignore_index=True)
        off_sp = float(p1["result"].mean() - p1["sp"].mean())
        off_tt = float(p1["pts"].mean() - p1["tt"].mean())
        p1["sp"] = np.round((p1["sp"] + off_sp) * 2) / 2
        p1["tt"] = np.round((p1["tt"] + off_tt) * 2) / 2
        lines = {"sp": dict(zip(p1["game_id"], p1["sp"], strict=True)), "tt": dict(zip(p1["game_id"], p1["tt"], strict=True))}
        frames = pool.map(_world_pass2, [t + (lines,) for t in tasks], chunksize=1)
    f = pd.concat(frames, ignore_index=True)
    absent = [c for c in full if c not in f.columns]
    for c in absent:
        f[c] = 0.0
    zeroed = list(ZERO_FILL)
    for c in zeroed:
        f[c] = 0.0
    gnames = [g for k, g, ts in load_chunks()]
    sidx = pd.concat(gnames).set_index("game_id")["sidx"]
    f["sidx"] = f["game_id"].map(sidx)
    f["ats_margin"] = f["result"] - f["spread_line"]
    keep = [c for c in dict.fromkeys(FEAT_ID + full) if c in f.columns]
    f = f[keep]
    f.to_parquet(SYN / "features.parquet")
    na = f[full].isna().mean()
    rep = {
        "games": len(f), "worlds": int(f["world"].nunique()), "seconds": time.time() - t0,
        "absent_filled_zero": absent, "zero_fill_forced": zeroed,
        "still_nan_fraction": {c: float(v) for c, v in na.items() if v > 0},
        "book": {n: {kk: vv for kk, vv in book[n].items() if kk != "pipe"} for n in ("spread", "total")},
        "offset_spread": off_sp, "offset_total": off_tt, "book_cols": book["cols"], "book_n": book["n"],
        "synthetic": {"line_sd": float(f["spread_line"].std()), "result_sd": float(f["result"].std()), "ats_sd": float(f["ats_margin"].std()), "ats_mean": float(f["ats_margin"].mean()), "total_mean": float(f["total_line"].mean())},
    }
    r = real_frame()
    r = r.loc[r["season"].between(2009, 2019)]
    rep["real"] = {"close_line_sd": float(r["spread_line"].std()), "result_sd": float(r["result"].std()), "ats_sd": float(r["ats_margin"].std()), "ats_mean": float(r["ats_margin"].mean()), "total_mean": float(r["total_line"].mean())}
    (OUT / "features_report.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep, indent=1)[:3000])


LAMBDAS = (10.0, 30.0, 100.0, 300.0, 1000.0, 3000.0, 10000.0, 100000.0)
TEST_SEASONS = tuple(range(2020, 2026))
BASE_ALPHA = 10.0
OPENER = REPO / "artifacts" / "opener_evaluation" / "20260929T192743Z"
DRAWS = 10000
SEED = 20261002


class ShrinkRidge:
    def __init__(self, lam, prior_raw):
        self.lam = float(lam)
        self.prior_raw = prior_raw

    def get_params(self, deep=True):
        return {"lam": self.lam, "prior_raw": self.prior_raw}

    def fit(self, X, y):
        from sklearn.impute import SimpleImputer
        from sklearn.preprocessing import StandardScaler

        self.imp = SimpleImputer(strategy="median", add_indicator=True)
        Z = self.imp.fit_transform(X)
        names = list(self.imp.get_feature_names_out(list(X.columns)))
        self.sc = StandardScaler().fit(Z)
        Zs = self.sc.transform(Z)
        yv = np.asarray(y, dtype=float)
        self.ybar = float(yv.mean())
        pri = np.array([self.prior_raw.get(n, 0.0) for n in names]) * self.sc.scale_
        A = Zs.T @ Zs + self.lam * np.eye(Zs.shape[1])
        self.coef_ = np.linalg.solve(A, Zs.T @ (yv - self.ybar) + self.lam * pri)
        self.names_ = names
        return self

    def predict(self, X):
        Z = self.imp.transform(X)
        return self.sc.transform(Z) @ self.coef_ + self.ybar


def real_train_frame():
    from nfl_ats.modeling import regular_season_rows

    f = pd.read_parquet(REPO / "data" / "processed" / "game_features_weak_stack.parquet")
    f = regular_season_rows(f)
    return f.loc[f["ats_margin"].notna() & f["result"].notna()].copy()


def syn_frame():
    f = pd.read_parquet(SYN / "features.parquet")
    return f.loc[f["sidx"].ge(1) & f["ats_margin"].notna()].copy()


def raw_coefs(est, cols):
    imp = est.named_steps["imputer"]
    names = list(imp.get_feature_names_out(cols))
    sc = est.named_steps["scaler"]
    std = est.named_steps["regressor"].coef_
    return names, std, std / sc.scale_


def family_of(cols):
    from nfl_ats.margin import resolve_feature_groups

    return dict(zip(cols, resolve_feature_groups(cols), strict=True))


def agreement(names, syn_std_in_real_units, real_std, cols):
    fam = family_of(cols)
    rows = {}
    for n, a, b in zip(names, syn_std_in_real_units, real_std, strict=True):
        if n not in fam:
            continue
        r = rows.setdefault(fam[n], {"n": 0, "n_active": 0, "agree": 0, "w_agree": 0.0, "w_total": 0.0})
        r["n"] += 1
        if a == 0.0:
            continue
        r["n_active"] += 1
        r["w_total"] += abs(b)
        if np.sign(a) == np.sign(b):
            r["agree"] += 1
            r["w_agree"] += abs(b)
    for r in rows.values():
        r["agree_share"] = r["agree"] / r["n_active"] if r["n_active"] else None
        r["weighted_agree_share"] = r["w_agree"] / r["w_total"] if r["w_total"] else None
    act = [(a, b) for n, a, b in zip(names, syn_std_in_real_units, real_std, strict=True) if n in fam and a != 0.0]
    corr = float(np.corrcoef([x for x, _ in act], [y for _, y in act])[0, 1])
    return rows, corr


def inner_select(real, prior_raw, cols, season):
    from nfl_ats.margin import make_margin_estimator

    inner = list(range(max(2010, season - 4), season))
    scores = {}
    for lam in LAMBDAS:
        tot, cnt = 0.0, 0
        for t in inner:
            tr, te = real.loc[real["season"].lt(t)], real.loc[real["season"].eq(t)]
            m = ShrinkRidge(lam, prior_raw).fit(tr[cols], tr["ats_margin"])
            e = te["ats_margin"].to_numpy(float) - m.predict(te[cols])
            tot += float((e**2).sum())
            cnt += len(e)
        scores[lam] = tot / cnt
    tot, cnt = 0.0, 0
    for t in inner:
        tr, te = real.loc[real["season"].lt(t)], real.loc[real["season"].eq(t)]
        m = make_margin_estimator("ridge", 42, ridge_alpha=BASE_ALPHA).fit(tr[cols], tr["ats_margin"])
        e = te["ats_margin"].to_numpy(float) - np.asarray(m.predict(te[cols]), dtype=float)
        tot += float((e**2).sum())
        cnt += len(e)
    best = min(LAMBDAS, key=lambda x: (scores[x], -x))
    return {"inner_seasons": inner, "mse": {str(k): v for k, v in scores.items()}, "base_mse": tot / cnt, "lambda": best}


def grade_worker(task):
    prior_raw, lam = task
    features = pd.read_parquet(REPO / "data" / "processed" / "game_features_weak_stack.parquet")
    t0 = time.time()
    out = grade_lambda(features, prior_raw, lam)
    out.to_parquet(OUT / f"scored_lambda_{lam:g}.parquet")
    return lam, out, time.time() - t0


def grade_lambda(features, prior_raw, lam):
    import nfl_ats.margin as mg
    from nfl_ats import clv
    from nfl_ats.constants import DEFAULT_MIN_TRAIN_GAMES

    orig = mg.make_margin_estimator

    def patched(model_name, random_state=42, **kw):
        if model_name == "ridge":
            return ShrinkRidge(lam, prior_raw)
        return orig(model_name, random_state, **kw)

    tag = hashlib.sha256(json.dumps(sorted(prior_raw.items())).encode()).hexdigest()[:12]
    cfg = {"feature_profile": "weak_stack", "regressor": "ridge", "target": "market_residual", "probability_method": "gaussian_median", "calibration_method": "none", "ridge_alpha": BASE_ALPHA, "mod25_variant": f"{lam}-{tag}"}
    mg.make_margin_estimator = patched
    try:
        sc = clv.opener_pick_evaluation(REPO / "data" / "market" / "raw", features, active_model_config=cfg, min_train_games=DEFAULT_MIN_TRAIN_GAMES)
    finally:
        mg.make_margin_estimator = orig
    return sc.loc[sc["season"].between(2020, 2025)].copy()


def tfit(z, y):
    from scipy.optimize import minimize

    def nll(par):
        q = par[0] * z
        return float(np.mean(np.logaddexp(0, q) - y * q))

    return float(minimize(nll, [1.0], method="BFGS").x[0])


def temp_recal(f):
    from scipy.special import expit, logit

    d = f.loc[f["margin_vs_open"].ne(0) & f["correct_at_open_probability_rule"].notna()]
    z_all = logit(f["home_cover_probability_at_open"].clip(1e-6, 1 - 1e-6))
    out = pd.Series(np.nan, index=f.index)
    for s_ in sorted(f["season"].unique()):
        tr = d.loc[d["season"].ne(s_)]
        t = tfit(logit(tr["home_cover_probability_at_open"].clip(1e-6, 1 - 1e-6)).to_numpy(), (tr["margin_vs_open"] > 0).astype(float).to_numpy())
        m = f["season"].eq(s_)
        out[m] = expit(t * z_all[m])
    return out


def season_boot(diff, seasons):
    ss = np.array(sorted(seasons.unique()))
    g = diff.groupby(seasons).agg(["sum", "count"]).reindex(ss).fillna(0)
    num, den = g["sum"].to_numpy(), g["count"].to_numpy()
    rng = np.random.default_rng(SEED)
    idx = rng.integers(0, len(ss), size=(DRAWS, len(ss)))
    b = num[idx].sum(1) / den[idx].sum(1)
    return {"mean": float(diff.mean()), "lo": float(np.percentile(b, 2.5)), "hi": float(np.percentile(b, 97.5)), "probability_positive": float((b > 0).mean()), "seasons_positive": int(((g["sum"] / g["count"]).to_numpy() > 0).sum()), "n_seasons": len(ss)}


def arm_stats(arm, base):
    a = arm.set_index("game_id") if "game_id" in arm.columns else arm
    b = base.set_index("game_id") if "game_id" in base.columns else base
    da = a.loc[a["margin_vs_open"].ne(0) & a["correct_at_open_probability_rule"].notna()]
    db = b.loc[b["margin_vs_open"].ne(0) & b["correct_at_open_probability_rule"].notna()]
    idx = da.index.intersection(db.index)
    da, db = da.loc[idx], db.loc[idx]
    ca, cb = da["correct_at_open_probability_rule"].astype(float), db["correct_at_open_probability_rule"].astype(float)
    pa, pb = temp_recal(a), temp_recal(b)
    y = (da["margin_vs_open"] > 0).astype(float)

    def ll(p):
        p = p.loc[idx].clip(1e-6, 1 - 1e-6)
        return -(y * np.log(p) + (1 - y) * np.log(1 - p))

    def br(p):
        return (p.loc[idx] - y) ** 2

    fl = ca - cb
    return {
        "n": len(idx), "record": f"{int(ca.sum())}-{len(idx) - int(ca.sum())}", "base_record": f"{int(cb.sum())}-{len(idx) - int(cb.sum())}",
        "flips": int((da["pick_home_at_open_probability_rule"] != db["pick_home_at_open_probability_rule"]).sum()),
        "flip_arm_correct": int((fl > 0).sum()), "flip_base_correct": int((fl < 0).sum()),
        "acc_diff_points": season_boot(100.0 * fl, da["season"]),
        "temp_logloss_gain": season_boot(ll(pb) - ll(pa), da["season"]),
        "temp_brier_gain": season_boot(br(pb) - br(pa), da["season"]),
        "arm_temp_logloss": float(ll(pa).mean()), "base_temp_logloss": float(ll(pb).mean()),
        "by_season_acc": {int(s_): float(100 * fl[da["season"] == s_].mean()) for s_ in sorted(da["season"].unique())},
    }


def cmd_student(args):
    from nfl_ats.margin import make_margin_estimator, margin_feature_columns

    cols = list(margin_feature_columns("market_residual", "weak_stack"))
    syn = syn_frame()
    syn_est = make_margin_estimator("ridge", 42, ridge_alpha=BASE_ALPHA).fit(syn[cols], syn["ats_margin"])
    names, std, raw = raw_coefs(syn_est, cols)
    prior_raw = dict(zip(names, raw.tolist(), strict=True))
    last = syn["chunk"].max()
    ho, tr = syn.loc[syn["chunk"].eq(last)], syn.loc[syn["chunk"].ne(last)]
    syn_r2 = None
    if len(tr) and len(ho):
        m2 = make_margin_estimator("ridge", 42, ridge_alpha=BASE_ALPHA).fit(tr[cols], tr["ats_margin"])
        e = ho["ats_margin"].to_numpy() - np.asarray(m2.predict(ho[cols]), dtype=float)
        syn_r2 = float(1 - (e**2).sum() / ((ho["ats_margin"] - tr["ats_margin"].mean()) ** 2).sum())
    real = real_train_frame()
    rtr = real.loc[real["season"].lt(2020)]
    rest = make_margin_estimator("ridge", 42, ridge_alpha=BASE_ALPHA).fit(rtr[cols], rtr["ats_margin"])
    rnames, rstd, rraw = raw_coefs(rest, cols)
    rsc = rest.named_steps["scaler"].scale_
    prior_in_real = np.array([prior_raw.get(n, 0.0) for n in rnames]) * rsc
    fams, corr = agreement(rnames, prior_in_real, rstd, cols)
    sel = {str(s_): inner_select(real, prior_raw, cols, s_) for s_ in TEST_SEASONS}
    (OUT / "synthetic_coefs.json").write_text(json.dumps({"names": names, "std": std.tolist(), "raw": raw.tolist(), "synthetic_rows": len(syn), "holdout_chunk_r2": syn_r2}, indent=1))
    (OUT / "selection.json").write_text(json.dumps(sel, indent=1))
    (OUT / "sign_agreement.json").write_text(json.dumps({"by_family": fams, "coef_corr_active": corr, "real_fit": "ridge alpha 10 on real seasons before 2020", "prior_scaled_to_real_units": True}, indent=1))
    print("selected", {k: v["lambda"] for k, v in sel.items()}, "syn_r2", syn_r2, "corr", corr, flush=True)
    base = pd.read_parquet(OPENER / "per_game.parquet")
    base = base.loc[base["season"].between(2020, 2025)]
    need = sorted({v["lambda"] for v in sel.values()} | (set(LAMBDAS) if args.all_lambdas else set()))
    import multiprocessing as mp

    scored = {}
    with mp.get_context("spawn").Pool(min(args.workers, len(need))) as pool:
        for lam, out, secs in pool.imap_unordered(grade_worker, [(prior_raw, l) for l in need]):
            scored[lam] = out
            print("lambda", lam, "rows", len(out), f"{secs:.0f}s", flush=True)
    pieces = [scored[sel[str(s_)]["lambda"]].loc[lambda d, s_=s_: d["season"].eq(s_)] for s_ in TEST_SEASONS]
    chosen = pd.concat(pieces, ignore_index=True)
    res = {"base_reproduction": arm_stats(base, base)["record"], "chosen": arm_stats(chosen, base), "selection": {k: v["lambda"] for k, v in sel.items()}}
    if args.all_lambdas:
        res["fixed_lambda_diagnostic"] = {str(l): {k: v for k, v in arm_stats(scored[l], base).items() if k in ("record", "flips", "acc_diff_points", "temp_logloss_gain")} for l in LAMBDAS}
    res["syn_holdout_r2"] = syn_r2
    (OUT / "results.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1)[:4000])


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("produce")
    p.add_argument("--minutes", type=float, default=40)
    p.add_argument("--worlds", type=int, default=6)
    p.add_argument("--seasons", type=int, default=10)
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--max-chunks", dest="max_chunks", type=int, default=99)
    q = sub.add_parser("features")
    q.add_argument("--workers", type=int, default=6)
    st = sub.add_parser("student")
    st.add_argument("--all-lambdas", dest="all_lambdas", action="store_true")
    st.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    {"produce": cmd_produce, "features": cmd_features, "student": cmd_student}[args.cmd](args)


if __name__ == "__main__":
    main()
