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

SYN = REPO / "data" / "processed" / "synthetic" / "crj"
OUT = REPO / "artifacts" / "mod25_pipeline_v2"
VARIANT = "crj"
SCALE = 0.9
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



def v2_setting(cfg_extra=None):
    import mod25d_variance as d
    import mod25_generator as gen

    cfgj = json.dumps(dict(d.DV[VARIANT], seed=1, name=f"{VARIANT}_s{SCALE:g}_verify"))
    setting = dict(gen.SETTING_DEFAULTS)
    setting.update({"scale": SCALE, "yard_gain": YARD_GAIN, "def_sign": 1.0, "yard_bias": YARD_BIAS, "drift": 1.0, "mech": cfgj})
    return setting


def latent_x(weekly, qb, shock):
    return weekly[..., 0] + np.where(qb, shock, 0.0), weekly[..., 1]


def cmd_verify(args):
    import multiprocessing as mp

    import mod25d_variance as d
    import mod25_generator as gen

    d.c25.ensure_policies()
    setting = v2_setting()
    fit = gen.load_fit()
    shock = float(fit["qb"]["backup_off_epa_effect"])
    league_off, league_def = gen.league_means()
    rng = np.random.default_rng(args.seed)
    weekly, qb = gen.gen_world_latents(rng, 1, fit, setting)[0]
    ratings = gen.season_ratings(weekly, qb, fit, setting, league_off, league_def)
    matchups = []
    while len(matchups) < args.matchups:
        w = int(rng.integers(1, 19))
        h, a = (int(x) for x in rng.choice(32, 2, replace=False))
        matchups.append((w, h, a))
    reps = args.reps
    per_task = 2
    tasks = []
    for t in range(reps // per_task):
        sched = [m for m in matchups for _ in range(per_task)]
        tasks.append((0, t, int(rng.integers(1, 2**31 - 1)), sched, ratings))
    gen.init_worker = d.d_gen_init
    t0 = time.time()
    res = []
    with mp.get_context("spawn").Pool(args.workers, initializer=d.d_gen_init, initargs=(setting,)) as pool:
        for r in pool.imap_unordered(d.d_play_season, tasks, chunksize=1):
            res.append(r)
    mid = {m: i for i, m in enumerate(matchups)}
    rows = []
    for world, t, gm_rows, plays in res:
        sched = tasks[t][3]
        for gi, hs, as_, ot, cap in gm_rows:
            w, h, a = sched[gi]
            rows.append((mid[(w, h, a)], hs - as_, hs + as_))
    df = pd.DataFrame(rows, columns=["m", "margin", "total"])
    g = df.groupby("m").agg(em=("margin", "mean"), n=("margin", "size"), sd=("margin", "std"), tot=("total", "mean"))
    xo, xd = latent_x(weekly, qb, shock)
    feat = []
    for mi, (w, h, a) in enumerate(matchups):
        feat.append((xo[w - 1, h] - xo[w - 1, a], xd[w - 1, h] - xd[w - 1, a]))
    F = np.array(feat)
    g["dxo"], g["dxd"] = F[g.index.to_numpy(), 0], F[g.index.to_numpy(), 1]
    X = np.column_stack([np.ones(len(g)), g["dxo"], g["dxd"]])
    wts = g["n"].to_numpy(float)
    W = np.sqrt(wts)[:, None]
    beta, *_ = np.linalg.lstsq(X * W, g["em"].to_numpy() * W[:, 0], rcond=None)
    pred = X @ beta
    resid = g["em"].to_numpy() - pred
    se_m = g["sd"].to_numpy() / np.sqrt(wts)
    sig2 = float(np.sum(wts * resid**2) / (len(g) - 3))
    cov = np.linalg.inv((X * wts[:, None]).T @ X) * sig2
    zl = g["dxo"].to_numpy() - g["dxd"].to_numpy()
    b1 = np.polyfit(zl, g["em"].to_numpy(), 1)
    quad = np.polyfit(zl, g["em"].to_numpy(), 2)
    ss = float(((g["em"] - g["em"].mean()) ** 2).sum())
    out = {
        "matchups": len(g), "games": int(g["n"].sum()), "minutes": (time.time() - t0) / 60.0,
        "intercept_hfa": float(beta[0]), "coef_off_diff": float(beta[1]), "coef_def_diff": float(beta[2]),
        "se": [float(x) for x in np.sqrt(np.diag(cov))],
        "r2_between_matchups": float(1 - (resid**2).sum() / ss),
        "mean_sampling_se_per_matchup": float(se_m.mean()),
        "weighted_resid_sd_vs_sampling_se": float(np.sqrt(np.mean((resid / se_m) ** 2))),
        "single_slope_on_off_minus_def": [float(x) for x in b1], "quad_coef": [float(x) for x in quad],
        "latent_diff_sd": float(zl.std()), "em_sd": float(g["em"].std()), "game_margin_sd": float(df["margin"].std()),
        "mean_total": float(df["total"].mean()),
        "shock": shock, "scale": SCALE, "variant": VARIANT,
    }
    (OUT / "verify.json").write_text(json.dumps(out, indent=1))
    g.assign(pred=pred).to_csv(OUT / "verify_matchups.csv")
    print(json.dumps(out, indent=1))


def cmd_truth(args):
    import mod25_generator as gen

    fit = gen.load_fit()
    shock = float(fit["qb"]["backup_off_epa_effect"])
    v = json.loads((OUT / "verify.json").read_text())
    c0, a_o, a_d = v["intercept_hfa"], v["coef_off_diff"], v["coef_def_diff"]
    parts = []
    for k, g, ts in load_chunks():
        lat = np.load(SYN / f"chunk_{k:02d}" / "latents.npz")
        pre = f"c{k:02d}"
        ids = g["game_id"].to_numpy()
        w_ = g["game_id"].str.slice(4, 8).astype(int).to_numpy()
        s_ = g["game_id"].str.slice(9, 11).astype(int).to_numpy()
        h_ = g["home_team"].str.split("T").str[1].astype(int).to_numpy()
        a_ = g["away_team"].str.split("T").str[1].astype(int).to_numpy()
        wk = g["week"].to_numpy() - 1
        dxo = np.zeros(len(g))
        dxd = np.zeros(len(g))
        for (wi, si) in sorted(set(zip(w_.tolist(), s_.tolist()))):
            weekly = lat[f"{pre}_w{wi}_s{si}_weekly"]
            qb = lat[f"{pre}_w{wi}_s{si}_qb_out"]
            xo, xd = latent_x(weekly, qb, shock)
            m = (w_ == wi) & (s_ == si)
            dxo[m] = xo[wk[m], h_[m]] - xo[wk[m], a_[m]]
            dxd[m] = xd[wk[m], h_[m]] - xd[wk[m], a_[m]]
        parts.append(pd.DataFrame({"game_id": ids, "true_em": c0 + a_o * dxo + a_d * dxd, "dxo": dxo, "dxd": dxd}))
    t = pd.concat(parts, ignore_index=True)
    t.to_parquet(SYN / "truth.parquet")
    f = pd.read_parquet(SYN / "features.parquet", columns=["game_id", "result", "sidx"])
    j = f.merge(t, on="game_id")
    j = j.loc[j["sidx"].ge(1)]
    rep = {"games": len(t), "true_em_sd": float(t["true_em"].std()), "true_em_mean": float(t["true_em"].mean()), "result_sd": float(j["result"].std()), "r2_true_em_vs_result": float(1 - ((j["result"] - j["true_em"]) ** 2).sum() / ((j["result"] - j["result"].mean()) ** 2).sum()), "corr_true_em_result": float(np.corrcoef(j["true_em"], j["result"])[0, 1])}
    (OUT / "truth_report.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep, indent=1))


def v2_cols():
    from nfl_ats.margin import margin_feature_columns

    full = list(margin_feature_columns("market_residual", "weak_stack"))
    fam = family_of(full)
    return full, [c for c in full if fam[c] != "market"]


def z_stats(df, cols):
    mu = df[cols].mean()
    sd = df[cols].std(ddof=0)
    sd = sd.where(sd > 1e-9, 1.0)
    return mu, sd


def z_apply(df, cols, mu, sd):
    return ((df[cols] - mu) / sd).fillna(0.0).to_numpy(float)


def r2(y, p, ref=None):
    y = np.asarray(y, float)
    ref = y.mean() if ref is None else ref
    return float(1 - ((y - p) ** 2).sum() / ((y - ref) ** 2).sum())


def real_walk_z(real, cols):
    out = np.zeros((len(real), len(cols)))
    seasons = real["season"].to_numpy()
    for s_ in sorted(set(seasons.tolist())):
        prior = real.loc[real["season"].lt(s_)]
        if len(prior) < 200:
            prior = real.loc[real["season"].eq(s_)]
        mu, sd = z_stats(prior, cols)
        m = seasons == s_
        out[m] = z_apply(real.loc[m], cols, mu, sd)
    return out


def cmd_student2(args):
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.linear_model import Ridge

    full, nomkt = v2_cols()
    f = pd.read_parquet(SYN / "features.parquet")
    t = pd.read_parquet(SYN / "truth.parquet")
    syn = f.merge(t[["game_id", "true_em"]], on="game_id", how="inner")
    syn = syn.loc[syn["sidx"].ge(1)].reset_index(drop=True)
    worlds = sorted(syn["world"].unique())
    hold_w = set(worlds[::4])
    ho = syn["world"].isin(hold_w).to_numpy()
    rep = {"synthetic_rows": len(syn), "worlds": len(worlds), "holdout_worlds": len(hold_w), "holdout_rows": int(ho.sum())}
    rep["oracle_true_em_r2_vs_result"] = r2(syn["result"][ho], syn["true_em"][ho].to_numpy())
    real = real_frame()
    real = real.sort_values(["season", "week", "game_id"]).reset_index(drop=True)
    models = {}
    out = real[["game_id", "season"]].copy()
    for name, cols in (("nomkt", nomkt), ("all", full)):
        mu, sd = z_stats(syn, cols)
        Z = z_apply(syn, cols, mu, sd)
        y_te = syn["true_em"].to_numpy()
        y_re = syn["result"].to_numpy()
        res = {}
        for a in (1.0, 10.0, 100.0, 1000.0, 10000.0):
            m = Ridge(alpha=a).fit(Z[~ho], y_te[~ho])
            p = m.predict(Z[ho])
            m2 = Ridge(alpha=a).fit(Z[~ho], y_re[~ho])
            p2 = m2.predict(Z[ho])
            res[str(a)] = {"r2_vs_true": r2(y_te[ho], p), "r2_vs_realized": r2(y_re[ho], p), "realized_label_model_r2_vs_realized": r2(y_re[ho], p2), "realized_label_model_r2_vs_true": r2(y_te[ho], p2)}
        best = max(res, key=lambda k: res[k]["r2_vs_true"])
        h = HistGradientBoostingRegressor(max_iter=400, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=100, l2_regularization=1.0, random_state=0).fit(Z[~ho], y_te[~ho])
        ph = h.predict(Z[ho])
        res["hgb"] = {"r2_vs_true": r2(y_te[ho], ph), "r2_vs_realized": r2(y_re[ho], ph)}
        rep[name] = {"grid": res, "ridge_alpha_chosen_on_holdout_worlds": float(best)}
        rid = Ridge(alpha=float(best)).fit(Z, y_te)
        hg = HistGradientBoostingRegressor(max_iter=400, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=100, l2_regularization=1.0, random_state=0).fit(Z, y_te)
        Zr = real_walk_z(real, cols)
        out[f"ridge_{name}"] = rid.predict(Zr)
        out[f"hgb_{name}"] = hg.predict(Zr)
        models[name] = (rid, hg)
        rep[name]["ridge_coef_top"] = {c: float(v) for c, v in sorted(zip(cols, rid.coef_, strict=True), key=lambda x: -abs(x[1]))[:12]}
    out.to_parquet(OUT / "real_distilled.parquet")
    rep["real_distilled_summary"] = {c: {"mean": float(out[c].mean()), "sd": float(out[c].std())} for c in out.columns if c not in ("game_id", "season")}
    rep["real_result_mean_sd"] = [float(real["result"].mean()), float(real["result"].std())]
    (OUT / "student_report.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep, indent=1)[:6000])


def cmd_diag(args):
    d = pd.read_parquet(OUT / "real_distilled.parquet")
    real = real_frame()[["game_id", "season", "week", "result", "spread_line"]]
    pg = pd.read_parquet(OPENER / "per_game.parquet")[["game_id", "tue_open_home_spread", "residual_at_open", "result"]].rename(columns={"result": "result_pg"})
    sbr = pd.read_parquet(REPO / "data" / "processed" / "sbr_odds.parquet").dropna(subset=["game_id", "open_home_spread"]).drop_duplicates("game_id")[["game_id", "open_home_spread"]]
    j = real.merge(d.drop(columns=["season"]), on="game_id").merge(pg, on="game_id", how="left").merge(sbr, on="game_id", how="left")
    j = j.loc[j["result"].notna()]
    out = {}
    for arm in ("ridge_nomkt", "hgb_nomkt", "ridge_all"):
        for era, m, op in (("2020_2025", j["season"].between(2020, 2025) & j["tue_open_home_spread"].notna(), "tue_open_home_spread"), ("2011_2019", j["season"].between(2011, 2019) & j["open_home_spread"].notna(), "open_home_spread")):
            e = j.loc[m]
            op_ = e[op].to_numpy(float)
            dm = e[arm].to_numpy()
            q = dm - op_
            r_ = e["result"].to_numpy() - op_
            row = {"n": len(e), "corr_distilled_close": float(np.corrcoef(dm, e["spread_line"])[0, 1]), "corr_distilled_open": float(np.corrcoef(dm, op_)[0, 1]), "corr_distilled_result": float(np.corrcoef(dm, e["result"])[0, 1]), "corr_close_result": float(np.corrcoef(e["spread_line"], e["result"])[0, 1]), "corr_open_result": float(np.corrcoef(op_, e["result"])[0, 1]), "corr_q_vs_margin_vs_open": float(np.corrcoef(q, r_)[0, 1]), "sd_q": float(q.std()), "corr_q_vs_move_close_minus_open": float(np.corrcoef(q, e["spread_line"] - op_)[0, 1])}
            if era == "2020_2025":
                b = e["residual_at_open"].to_numpy(float)
                ok = np.isfinite(b)
                qq, rr, bb = q[ok], r_[ok], b[ok]
                A = np.column_stack([np.ones(ok.sum()), bb])
                rq = qq - A @ np.linalg.lstsq(A, qq, rcond=None)[0]
                rr_ = rr - A @ np.linalg.lstsq(A, rr, rcond=None)[0]
                row["partial_corr_q_resid_given_base"] = float(np.corrcoef(rq, rr_)[0, 1])
                row["corr_base_pred_vs_margin_vs_open"] = float(np.corrcoef(bb, rr)[0, 1])
                row["corr_q_vs_base_pred"] = float(np.corrcoef(qq, bb)[0, 1])
                row["n_partial"] = int(ok.sum())
                ss = sorted(e.loc[ok, "season"].unique())
                se = []
                for s_ in ss:
                    mm = (e.loc[ok, "season"] == s_).to_numpy()
                    A2 = A[mm]
                    a_ = qq[mm] - A2 @ np.linalg.lstsq(A2, qq[mm], rcond=None)[0]
                    b_ = rr[mm] - A2 @ np.linalg.lstsq(A2, rr[mm], rcond=None)[0]
                    se.append(float(np.corrcoef(a_, b_)[0, 1]))
                row["partial_corr_by_season"] = dict(zip([int(x) for x in ss], se, strict=True))
            out[f"{arm}_{era}"] = row
    (OUT / "diagnostics.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1)[:7000])


WLOG = []


class TwoStage:
    def __init__(self, base_cols, orig, kw):
        self.base_cols = list(base_cols)
        self.orig = orig
        self.kw = kw

    def get_params(self, deep=True):
        return {"base_cols": self.base_cols, "orig": self.orig, "kw": self.kw}

    def _base(self):
        return self.orig("ridge", 42, ridge_alpha=BASE_ALPHA, **self.kw)

    def fit(self, X, y):
        yv = np.asarray(y, dtype=float)
        season = X["season"].to_numpy(float)
        B = X[self.base_cols]
        self.base = self._base().fit(B, yv)
        b_oos = np.full(len(yv), np.nan)
        for s_ in np.unique(season):
            m = season == s_
            if (~m).sum() < 100:
                continue
            b_oos[m] = np.asarray(self._base().fit(B.iloc[~m], yv[~m]).predict(B.iloc[m]), dtype=float)
        ok = np.isfinite(b_oos)
        q = (X["distilled_margin"] - X["spread_line"]).to_numpy(float)
        self.qbar = float(q[ok].mean())
        self.bbar = float(b_oos[ok].mean())
        t = (q[ok] - self.qbar) - (b_oos[ok] - self.bbar)
        r = yv[ok] - b_oos[ok]
        r = r - r.mean()
        self.w = float((t @ r) / (t @ t))
        WLOG.append((int(season.max()), self.w))
        return self

    def predict(self, X):
        b = np.asarray(self.base.predict(X[self.base_cols]), dtype=float)
        q = (X["distilled_margin"] - X["spread_line"]).to_numpy(float)
        t = (q - self.qbar) - (b - self.bbar)
        return b + self.w * t


def two_stage_patch():
    import nfl_ats.margin as mg

    orig = mg.make_margin_estimator
    full, _ = v2_cols()

    def patched(model_name, random_state=42, **kw):
        if model_name == "ridge":
            return TwoStage(full, orig, {k: v for k, v in kw.items() if k == "suppressed_indicator_columns"})
        return orig(model_name, random_state, **kw)

    return mg, orig, patched


def frame_from_pergame(pg):
    d = pg.loc[pg["margin_vs_open"].ne(0) & pg["correct_at_open_probability_rule"].notna()]
    return pd.DataFrame({"season": d["season"].to_numpy(), "p": d["home_cover_probability_at_open"].to_numpy(), "pick_home": d["pick_home_at_open_probability_rule"].to_numpy(), "correct": d["correct_at_open_probability_rule"].astype(float).to_numpy()}, index=pd.Index(d["game_id"].to_numpy(), name="game_id"))


def frame_from_proxy(df):
    return pd.DataFrame({"season": df["season"].to_numpy(), "p": df["p"].to_numpy(), "pick_home": df["pick_home"].to_numpy(), "correct": df["correct"].astype(float).to_numpy()}, index=df.index)


def recal_series(fr):
    from scipy.special import expit, logit

    y = np.where(fr["pick_home"], fr["correct"], 1.0 - fr["correct"])
    z = logit(fr["p"].clip(1e-6, 1 - 1e-6).to_numpy(float))
    out = np.zeros(len(fr))
    se = fr["season"].to_numpy()
    for s_ in np.unique(se):
        m = se == s_
        t = tfit(z[~m], y[~m])
        out[m] = expit(t * z[m])
    return pd.Series(out, index=fr.index), y


def compare_frames(arm, base):
    idx = arm.index.intersection(base.index)
    a, b = arm.loc[idx], base.loc[idx]
    pa, ya = recal_series(a)
    pb, yb = recal_series(b)
    assert (ya == yb).all()

    def ll(p):
        p = p.clip(1e-6, 1 - 1e-6).to_numpy()
        return -(ya * np.log(p) + (1 - ya) * np.log(1 - p))

    def br(p):
        return (p.to_numpy() - ya) ** 2

    fl = a["correct"] - b["correct"]
    seasons = a["season"]
    gain = pd.Series(ll(pb) - ll(pa), index=idx)
    bg = pd.Series(br(pb) - br(pa), index=idx)
    wins = int(a["correct"].sum())
    bw = int(b["correct"].sum())
    flipped = a["pick_home"].to_numpy() != b["pick_home"].to_numpy()
    return {
        "n": len(idx), "record": f"{wins}-{len(idx) - wins}", "base_record": f"{bw}-{len(idx) - bw}",
        "flips": int(flipped.sum()), "flip_record_arm_W_L": f"{int((fl[flipped] > 0).sum())}-{int((fl[flipped] < 0).sum())}",
        "acc_diff_points": season_boot(100.0 * fl, seasons), "recal_logloss_gain": season_boot(gain, seasons), "recal_brier_gain": season_boot(bg, seasons),
        "arm_recal_logloss": float(ll(pa).mean()), "base_recal_logloss": float(ll(pb).mean()),
        "by_season_acc_diff": {int(s_): float(100 * fl[seasons == s_].mean()) for s_ in sorted(seasons.unique())},
        "by_season_logloss_gain": {int(s_): float(gain[seasons == s_].mean()) for s_ in sorted(seasons.unique())},
    }


def cmd_grade(args):
    import mod24_u1 as u1
    from nfl_ats import clv
    from nfl_ats.constants import DEFAULT_MIN_TRAIN_GAMES

    full, _ = v2_cols()
    d = pd.read_parquet(OUT / "real_distilled.parquet")[["game_id", args.arm]].rename(columns={args.arm: "distilled_margin"})
    feats = pd.read_parquet(REPO / "data" / "processed" / "game_features_weak_stack.parquet")
    feats = feats.merge(d, on="game_id", how="left")
    reg = feats.loc[feats["game_type"].eq("REG")]
    assert reg["distilled_margin"].notna().all()
    cols = full + ["distilled_margin", "season"]
    res = {"arm": args.arm}
    base_cache = OUT / "proxy_base.parquet"
    pop = u1.proxy_population(feats)
    if base_cache.exists():
        pbase = pd.read_parquet(base_cache)
    else:
        pbase = u1.proxy_eval(feats, pop, lambda s: (full, BASE_ALPHA))
        pbase.to_parquet(base_cache)
    mg, orig, patched = two_stage_patch()
    mg.make_margin_estimator = patched
    WLOG.clear()
    try:
        parm = u1.proxy_eval(feats, pop, lambda s: (cols, BASE_ALPHA))
        w_proxy = list(WLOG)
        WLOG.clear()
        cfg = {"feature_profile": "weak_stack", "regressor": "ridge", "target": "market_residual", "probability_method": "gaussian_median", "calibration_method": "none", "ridge_alpha": BASE_ALPHA, "mod25_variant": f"twostage-{args.arm}"}
        with u1.patched(cols):
            sc = clv.opener_pick_evaluation(REPO / "data" / "market" / "raw", feats, active_model_config=cfg, min_train_games=DEFAULT_MIN_TRAIN_GAMES)
        w_true = list(WLOG)
    finally:
        mg.make_margin_estimator = orig
    sc = sc.loc[sc["season"].between(2020, 2025)].copy()
    sc.to_parquet(OUT / f"scored_{args.arm}.parquet")
    parm.to_parquet(OUT / f"proxy_{args.arm}.parquet")
    base_pg = pd.read_parquet(OPENER / "per_game.parquet")
    base_pg = base_pg.loc[base_pg["season"].between(2020, 2025)]
    res["true_2020_2025"] = compare_frames(frame_from_pergame(sc), frame_from_pergame(base_pg))
    res["base_reproduction_2020_2025"] = frame_from_pergame(base_pg)["correct"].sum()
    res["proxy_2011_2019_sbr_open"] = compare_frames(frame_from_proxy(parm), frame_from_proxy(pbase))

    def wsum(w):
        df = pd.DataFrame(w, columns=["max_season", "w"])
        return {int(k): float(v) for k, v in df.groupby("max_season")["w"].mean().items()}

    res["weight_by_last_training_season_mean"] = {"proxy": wsum(w_proxy), "true": wsum(w_true)}
    (OUT / f"grade_{args.arm}.json").write_text(json.dumps(res, indent=1, default=float))
    print(json.dumps(res, indent=1, default=float)[:6000])

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
    vf = sub.add_parser("verify")
    vf.add_argument("--matchups", type=int, default=96)
    vf.add_argument("--reps", type=int, default=100)
    vf.add_argument("--workers", type=int, default=6)
    vf.add_argument("--seed", type=int, default=11)
    sub.add_parser("truth")
    sub.add_parser("student2")
    sub.add_parser("diag")
    gr = sub.add_parser("grade")
    gr.add_argument("--arm", required=True)
    st = sub.add_parser("student")
    st.add_argument("--all-lambdas", dest="all_lambdas", action="store_true")
    st.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    {"produce": cmd_produce, "features": cmd_features, "student": cmd_student, "verify": cmd_verify, "truth": cmd_truth, "student2": cmd_student2, "diag": cmd_diag, "grade": cmd_grade}[args.cmd](args)


if __name__ == "__main__":
    main()
