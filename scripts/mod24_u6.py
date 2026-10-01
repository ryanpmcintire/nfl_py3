import glob
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import brentq, minimize
from scipy.special import expit, logit
from sklearn.linear_model import Ridge

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import lead66_unit1 as l66

from nfl_ats import clv
from nfl_ats import margin as margin_mod
from nfl_ats.constants import DEFAULT_MIN_TRAIN_GAMES, FEATURE_SETS
from nfl_ats.margin import (
    _MARGIN_PROFILE_FEATURE_SETS,
    _PROFILE_SUPPRESSED_MISSING_INDICATORS,
    MarginModel,
    make_margin_estimator,
    margin_feature_columns,
)
from nfl_ats.mass_preserving_lattice import DiscretePushReader, prior_pool, prior_pool_for_week, serve_discrete_three_way
from nfl_ats.modeling import regular_season_rows

OUT = ROOT / "artifacts" / "mod24_u6"
FEATURES = ROOT / "data" / "processed" / "game_features_weak_stack.parquet"
PBP = ROOT / "data" / "pbp" / "raw" / "20260929T191306Z"
MARKET_ROOT = ROOT / "data" / "market" / "raw"
SBR = ROOT / "data" / "processed" / "sbr_odds.parquet"
OPENER = ROOT / "artifacts" / "opener_evaluation" / "20260929T192743Z"
U1 = ROOT / "artifacts" / "mod24_u1"
PROFILE = "weak_stack"
SET_NAME = _MARGIN_PROFILE_FEATURE_SETS[PROFILE][1]
SUPP = _PROFILE_SUPPRESSED_MISSING_INDICATORS.get(PROFILE, ())
FULL = tuple(margin_feature_columns("market_residual", PROFILE))
BASE_ALPHA = 10.0
ALPHA_GRID = (1.0, 3.0, 10.0, 30.0, 100.0)
PROXY_SEASONS = tuple(range(2011, 2020))
TRUE_SEASONS = tuple(range(2020, 2026))
INNER = 4
FIRST_INNER = 2010
PROB_METHOD = "gaussian_median"
BOOT = 20000
SEED = 20261002
ARMS = {"h1_stacked_halves": 2, "h2_multitask_halves": 2, "h3_quarters": 4}
STATE = {"arm": None, "halves": None}


def build_halves():
    rows = []
    for f in sorted(glob.glob(str(PBP / "season=*" / "plays.parquet"))):
        d = pd.read_parquet(
            f, columns=["game_id", "qtr", "score_differential", "posteam", "home_team", "play_id"]
        )
        d = d.loc[d["qtr"].between(1, 4) & d["posteam"].notna() & d["score_differential"].notna()].copy()
        d["hm"] = np.where(d["posteam"].eq(d["home_team"]), d["score_differential"], -d["score_differential"])
        d = d.sort_values(["game_id", "play_id"])
        first = d.groupby(["game_id", "qtr"])["hm"].first().unstack()
        first = first.reindex(columns=[2.0, 3.0, 4.0])
        first.columns = ["m_q2", "m_q3", "m_q4"]
        rows.append(first.reset_index())
    return pd.concat(rows, ignore_index=True)


def load_halves(features):
    path = OUT / "halves.parquet"
    if not path.exists():
        OUT.mkdir(parents=True, exist_ok=True)
        build_halves().to_parquet(path)
    h = pd.read_parquet(path)
    f = features[["game_id", "result", "spread_line"]].drop_duplicates("game_id")
    h = f.merge(h, on="game_id", how="left")
    return h.set_index("game_id")


class HalfRidge:
    def __init__(self, arm, alpha):
        self.arm = arm
        self.k = ARMS[arm]
        self.alpha = alpha
        self.pre = None
        self.coef_ = None

    def targets(self, frame):
        h = STATE["halves"].reindex(frame["game_id"].to_numpy())
        line = pd.to_numeric(frame["spread_line"], errors="coerce").to_numpy(dtype=float)
        res = h["result"].to_numpy(dtype=float)
        if self.k == 2:
            cum = np.column_stack([h["m_q3"].to_numpy(dtype=float), res])
        else:
            cum = np.column_stack([h["m_q2"], h["m_q3"], h["m_q4"], res]).astype(float)
        parts = np.diff(np.column_stack([np.zeros(len(cum)), cum]), axis=1)
        return parts - (line / self.k)[:, None]

    def fit(self, frame):
        est = make_margin_estimator("ridge", 42, ridge_alpha=self.alpha, suppressed_indicator_columns=SUPP)
        self.pre = est[:-1]
        X = frame.loc[:, list(FULL)]
        t = self.targets(frame)
        keep = np.isfinite(t).all(axis=1)
        Z = np.asarray(self.pre.fit_transform(X), dtype=float)
        Zk, tk = Z[keep], t[keep]
        self.zmean = Zk.mean(axis=0)
        zc = Zk - self.zmean
        stacked = np.vstack([zc] * self.k)
        y = np.concatenate([tk[:, j] - tk[:, j].mean() for j in range(self.k)])
        if self.arm == "h1_stacked_halves":
            pooled = np.concatenate([tk[:, j] for j in range(self.k)])
            r = Ridge(alpha=self.alpha, fit_intercept=True).fit(stacked, pooled)
            self.coef_ = r.coef_
            self.icpt = self.k * float(r.intercept_)
        else:
            r = Ridge(alpha=self.alpha, fit_intercept=False).fit(stacked, y)
            self.coef_ = r.coef_
            self.icpt = float(tk.sum(axis=1).mean())
        self.nkept = int(keep.sum())
        return self

    def predict(self, X):
        Z = np.asarray(self.pre.transform(X), dtype=float)
        return self.k * ((Z - self.zmean) @ self.coef_) + self.icpt


def fit_half_model(training, *, target="market_residual", model_name="ridge", feature_profile=PROFILE, ridge_alpha=10.0, distribution_fraction=0.20, **_):
    arm = STATE["arm"]
    tr = regular_season_rows(training)
    tr = tr.loc[pd.to_numeric(tr["ats_margin"], errors="coerce").notna()].copy()
    tr["gameday"] = pd.to_datetime(tr["gameday"], errors="raise")
    tr = tr.sort_values(["gameday", "game_id"]).reset_index(drop=True)
    n_dist = int(len(tr) * distribution_fraction)
    split = len(tr) - n_dist
    fit_part, dist_part = tr.iloc[:split], tr.iloc[split:]
    tmp = HalfRidge(arm, ridge_alpha).fit(fit_part)
    pred = tmp.predict(dist_part.loc[:, list(FULL)])
    resid = pd.to_numeric(dist_part["ats_margin"]).to_numpy(dtype=float) - pred
    resid = resid[np.isfinite(resid)]
    est = HalfRidge(arm, ridge_alpha).fit(tr)
    return MarginModel(
        estimator=est,
        residuals=np.asarray(resid, dtype=np.float64),
        model_name="ridge",
        ridge_alpha=ridge_alpha,
        target="market_residual",
        feature_columns=FULL,
        training_rows=len(tr),
        distribution_rows=len(resid),
        training_max_gameday=tr["gameday"].max().date().isoformat(),
    )


def inner_mse(frame, arm, alpha, seasons):
    tot, cnt = 0.0, 0
    for s in seasons:
        train = frame.loc[frame["season"].lt(s)]
        test = frame.loc[frame["season"].eq(s)]
        m = HalfRidge(arm, alpha).fit(train)
        err = test["ats_margin"].to_numpy(dtype=float) - m.predict(test.loc[:, list(FULL)])
        tot += float(np.sum(err**2))
        cnt += len(err)
    return tot / cnt


def select(frame):
    path = OUT / "selection.json"
    sel = json.loads(path.read_text()) if path.exists() else {}
    for arm in ARMS:
        sel.setdefault(arm, {})
        for s in PROXY_SEASONS + TRUE_SEASONS:
            if str(s) in sel[arm]:
                continue
            inner = list(range(max(FIRST_INNER, s - INNER), s))
            scores = {a: inner_mse(frame, arm, a, inner) for a in ALPHA_GRID}
            best = min(ALPHA_GRID, key=lambda a: (scores[a], abs(a - BASE_ALPHA)))
            sel[arm][str(s)] = {"alpha": best, "inner_mse": scores[best], "grid": {str(k): v for k, v in scores.items()}}
            print(arm, s, best, flush=True)
            path.write_text(json.dumps(sel, indent=1))
    return sel


class patched_cols:
    def __enter__(self):
        self.orig = FEATURE_SETS[SET_NAME]

    def __exit__(self, *a):
        FEATURE_SETS[SET_NAME] = self.orig


def proxy_population(features):
    reg = regular_season_rows(features)
    reg = reg.loc[reg["season"].between(PROXY_SEASONS[0], PROXY_SEASONS[-1])]
    sbr = pd.read_parquet(SBR).dropna(subset=["game_id", "open_home_spread"]).drop_duplicates("game_id")
    pop = reg[["game_id", "season", "week", "result"]].merge(sbr[["game_id", "open_home_spread"]], on="game_id", how="inner")
    pop = pop.loc[pd.to_numeric(pop["result"], errors="coerce").notna()].copy()
    pop["season"] = pop["season"].astype(int)
    pop["week"] = pop["week"].astype(int)
    return pop.rename(columns={"open_home_spread": "open_line"})


def finish(df, p):
    df = df.copy()
    df["pick_home"] = p.to_numpy() >= 0.5
    m = df["margin_vs_open"]
    df["correct"] = np.where(m.eq(0), np.nan, np.where(df["pick_home"], m.gt(0), ~m.gt(0)))
    df = df.loc[df["correct"].notna()].copy()
    df["y"] = (df["margin_vs_open"] > 0).astype(float)
    pc = df["p"].clip(1e-6, 1 - 1e-6)
    df["ll"] = -(df["y"] * np.log(pc) + (1 - df["y"]) * np.log(1 - pc))
    df["brier"] = (pc - df["y"]) ** 2
    df["open_line"] = df["open_line"] if "open_line" in df else np.nan
    return df.set_index("game_id")[["season", "week", "p", "pick_home", "correct", "ll", "brier", "margin_vs_open", "open_line"]]


def proxy_eval(features, pop, alpha, seasons):
    frame = regular_season_rows(features).copy()
    frame["gameday"] = pd.to_datetime(frame["gameday"])
    completed = frame.loc[frame["result"].notna()].copy()
    pool = prior_pool(frame, pop.set_index("game_id")["open_line"])
    rows = []
    for (season, week), group in pop.loc[pop["season"].isin(seasons)].groupby(["season", "week"], sort=True):
        week_rows = frame.loc[frame["game_id"].isin(set(group["game_id"]))]
        target = frame.loc[frame["season"].eq(season) & frame["week"].eq(week)]
        cutoff = target["gameday"].min()
        training = completed.loc[completed["gameday"].lt(cutoff)]
        if len(training) < DEFAULT_MIN_TRAIN_GAMES:
            continue
        model = fit_half_model(training, ridge_alpha=alpha)
        reader = DiscretePushReader.for_week(pool, season=int(season), week=int(week), cutoff=pd.Timestamp(cutoff), exclude_game_ids=set(week_rows["game_id"].astype(str)))
        scoring = week_rows.merge(group[["game_id", "open_line"]], on="game_id").copy()
        at_open = scoring.copy()
        at_open["spread_line"] = at_open["open_line"]
        pred = model.predict(at_open, probability_method=PROB_METHOD)
        disc = serve_discrete_three_way(pred, at_open, reader, residuals=model.residuals, probability_method=PROB_METHOD)
        out = scoring[["game_id", "result", "open_line"]].copy()
        out["season"] = int(season)
        out["week"] = int(week)
        out["p"] = disc["home_cover_probability"].to_numpy()
        rows.append(out)
    out = pd.concat(rows, ignore_index=True)
    out["margin_vs_open"] = out["result"] - out["open_line"]
    return finish(out, out["p"])


def true_eval(features, alpha):
    cfg = {
        "feature_profile": PROFILE,
        "regressor": "ridge",
        "target": "market_residual",
        "probability_method": PROB_METHOD,
        "calibration_method": "none",
        "ridge_alpha": alpha,
        "mod24_variant": json.dumps(["u6", STATE["arm"], alpha]),
    }
    scored = clv.opener_pick_evaluation(MARKET_ROOT, features, active_model_config=cfg, min_train_games=DEFAULT_MIN_TRAIN_GAMES)
    scored = scored.loc[scored["correct_at_open_probability_rule"].notna()].copy()
    scored["p"] = scored["home_cover_probability_at_open"]
    scored["open_line"] = scored["tue_open_home_spread"] if "tue_open_home_spread" in scored else np.nan
    return finish(scored[["game_id", "season", "week", "p", "margin_vs_open", "open_line"]], scored["p"])


def run_arm(arm, features, pop, sel):
    STATE["arm"] = arm
    clv.fit_margin_model = fit_half_model
    cache_t, cache_p, parts = {}, {}, []
    for s in TRUE_SEASONS:
        a = sel[arm][str(s)]["alpha"]
        if a not in cache_t:
            cp = OUT / f"cache_{arm}_true_{a}.parquet"
            if cp.exists():
                cache_t[a] = pd.read_parquet(cp)
            else:
                cache_t[a] = true_eval(features, a)
                cache_t[a].to_parquet(cp)
            print(arm, "true", a, flush=True)
        parts.append(cache_t[a].loc[cache_t[a]["season"].eq(s)])
    for a in sorted({sel[arm][str(s)]["alpha"] for s in PROXY_SEASONS}):
        seasons = [s for s in PROXY_SEASONS if sel[arm][str(s)]["alpha"] == a]
        cache_p[a] = proxy_eval(features, pop, a, seasons)
        print(arm, "proxy", a, seasons, flush=True)
        parts.append(cache_p[a])
    return pd.concat(parts)


def boot(df, col, block):
    keys = df[block].astype(str).agg("-".join, axis=1) if len(block) > 1 else df[block[0]]
    g = df.groupby(keys)[col].agg(["sum", "count"])
    sums, sizes = g["sum"].to_numpy(), g["count"].to_numpy()
    rng = np.random.default_rng(SEED)
    idx = rng.integers(0, len(sums), (BOOT, len(sums)))
    d = sums[idx].sum(axis=1) / sizes[idx].sum(axis=1)
    return {"low": float(np.percentile(d, 2.5)), "high": float(np.percentile(d, 97.5)), "pp": float((d > 0).mean()), "blocks": len(sums)}


def fit_temp(z, y):
    r = minimize(lambda q: float(np.mean(np.logaddexp(0, q[0] * z) - y * q[0] * z)), [1.0], method="BFGS")
    return float(r.x[0])


def recal(df):
    p = df["p"].clip(1e-6, 1 - 1e-6)
    z = logit(p).to_numpy()
    y = df["y"].to_numpy() if "y" in df else (df["margin_vs_open"] > 0).astype(float).to_numpy()
    out = np.zeros(len(df))
    seasons = df["season"].to_numpy()
    for s in np.unique(seasons):
        tr = seasons != s
        t = fit_temp(z[tr], y[tr])
        out[seasons == s] = expit(t * z[seasons == s])
    return pd.Series(out, index=df.index)


def metrics_recal(df):
    pr = recal(df)
    y = (df["margin_vs_open"] > 0).astype(float)
    pc = pr.clip(1e-6, 1 - 1e-6)
    return pd.DataFrame({"season": df["season"], "week": df["week"], "pr": pr, "ll": -(y * np.log(pc) + (1 - y) * np.log(1 - pc)), "brier": (pc - y) ** 2})


def theta_for(atoms, counts, line, target):
    home, away = atoms > line, atoms < line

    def diff(th):
        m = l66.mass_at_theta(atoms, counts, th)
        return float(m[home].sum() / m[home | away].sum() - target)

    if not 0 < target < 1:
        return None
    if diff(-2.0) > 0 or diff(2.0) < 0:
        return None
    return brentq(diff, -2.0, 2.0, xtol=1e-13)


def rps(atoms, mass, actual):
    grid = np.arange(min(atoms.min(), actual), max(atoms.max(), actual) + 1)
    cum = np.concatenate([[0.0], np.cumsum(mass)])
    cdf = cum[np.searchsorted(atoms, grid, side="right")]
    return float(np.square(cdf - (actual <= grid)).sum())


def rps_table(frames_recal):
    feats = pd.read_parquet(FEATURES, columns=["game_id", "season", "week", "gameday", "spread_line", "result"])
    opener = pd.read_parquet(OPENER / "per_game.parquet")
    opener = opener.loc[opener.season.between(2020, 2025)].copy()
    pool = prior_pool(feats, opener.set_index("game_id").tue_open_home_spread)
    result = feats.set_index("game_id").result
    gameday = feats.set_index("game_id").gameday
    line = opener.set_index("game_id").tue_open_home_spread
    games = opener.assign(gameday=pd.to_datetime(opener.game_id.map(gameday)))
    out = {k: {} for k in frames_recal}
    for (season, week), group in games.groupby(["season", "week"], sort=True):
        target = feats.loc[feats.season.eq(season) & feats.week.eq(week)]
        prior = prior_pool_for_week(pool, season=int(season), week=int(week), cutoff=pd.to_datetime(target.gameday).min(), exclude_game_ids=group.game_id)
        for gid in group.game_id:
            ln = float(line.loc[gid])
            selected, _ = l66.select_band(prior, ln)
            atoms, counts = l66.atomic_counts(selected)
            for k, pr in frames_recal.items():
                if gid not in pr.index:
                    continue
                t = theta_for(atoms, counts, ln, float(pr.loc[gid]))
                if t is not None:
                    out[k][gid] = rps(atoms, l66.mass_at_theta(atoms, counts, t), float(result.loc[gid]))
    return {k: pd.Series(v) for k, v in out.items()}


def compare(arm, base, with_rps):
    j = arm.join(base, rsuffix="_b", how="inner")
    j["d_acc"] = 100.0 * (j["correct"] - j["correct_b"])
    ra, rb = metrics_recal(arm), metrics_recal(base)
    j["d_ll"] = (rb["ll"] - ra["ll"]).reindex(j.index)
    j["d_brier"] = (rb["brier"] - ra["brier"]).reindex(j.index)
    if with_rps:
        j["d_rps"] = (with_rps["base"] - with_rps["arm"]).reindex(j.index)
    res = {}
    eras = {"true_2020_2025": j["season"].ge(2020), "proxy_2011_2019": j["season"].le(2019), "pooled_2011_2025": j["season"].ge(0)}
    for name, mask in eras.items():
        s = j.loc[mask]
        a = arm.loc[s.index]
        wins = int(a["correct"].sum())
        flip = s["pick_home"] != s["pick_home_b"]
        r = {
            "n": len(s),
            "record": f"{wins}-{len(s) - wins}",
            "base_record": f"{int(s['correct_b'].sum())}-{len(s) - int(s['correct_b'].sum())}",
            "flips": int(flip.sum()),
            "flip_wins_losses": f"{int(s.loc[flip, 'correct'].sum())}-{int(flip.sum() - s.loc[flip, 'correct'].sum())}",
            "d_acc": float(s["d_acc"].mean()),
            "seasons_pos": int((s.groupby("season")["d_acc"].mean() > 0).sum()),
            "seasons": int(s["season"].nunique()),
            "recal_gain_ll": float(s["d_ll"].mean()),
            "recal_gain_brier": float(s["d_brier"].mean()),
            "acc_season_boot": boot(s, "d_acc", ["season"]),
            "acc_week_boot": boot(s, "d_acc", ["season", "week"]),
            "ll_season_boot": boot(s, "d_ll", ["season"]),
            "brier_season_boot": boot(s, "d_brier", ["season"]),
        }
        if with_rps and name == "true_2020_2025":
            r["recal_gain_rps"] = float(s["d_rps"].mean())
            r["rps_season_boot"] = boot(s.dropna(subset=["d_rps"]), "d_rps", ["season"])
        res[name] = r
    return res


def stability(frame, sel):
    out = {}
    folds = list(PROXY_SEASONS + TRUE_SEASONS)
    for name in ("base",) + tuple(ARMS):
        vecs = {}
        for s in folds:
            train = frame.loc[frame["season"].lt(s)]
            if name == "base":
                est = make_margin_estimator("ridge", 42, ridge_alpha=BASE_ALPHA, suppressed_indicator_columns=SUPP)
                est.fit(train.loc[:, list(FULL)], train["ats_margin"])
                vecs[s] = np.asarray(est[-1].coef_, dtype=float)
            else:
                m = HalfRidge(name, sel[name][str(s)]["alpha"]).fit(train)
                vecs[s] = m.k * m.coef_
        def mean_corr(ss):
            ks = [k for k in ss]
            cs = [np.corrcoef(vecs[a], vecs[b])[0, 1] for i, a in enumerate(ks) for b in ks[i + 1:]]
            adj = [np.corrcoef(vecs[ks[i]], vecs[ks[i + 1]])[0, 1] for i in range(len(ks) - 1)]
            return {"pairwise": float(np.mean(cs)), "adjacent": float(np.mean(adj))}
        out[name] = {"true_2020_2025": mean_corr(TRUE_SEASONS), "all_2011_2025": mean_corr(folds)}
    return out


def equivalence(frame):
    s = 2020
    train = frame.loc[frame["season"].lt(s)]
    test = frame.loc[frame["season"].eq(s)]
    out = {}
    for arm, k in ARMS.items():
        h = HalfRidge(arm, 10.0).fit(train)
        stacked_pred = h.predict(test.loc[:, list(FULL)])
        est = make_margin_estimator("ridge", 42, ridge_alpha=10.0 / k, suppressed_indicator_columns=SUPP)
        keep = np.isfinite(h.targets(train)).all(axis=1)
        sub = train.loc[keep]
        est.fit(sub.loc[:, list(FULL)], sub["ats_margin"])
        base_pred = np.asarray(est.predict(test.loc[:, list(FULL)]), dtype=float)
        out[arm] = {"max_abs_diff_vs_base_alpha_over_k": float(np.max(np.abs(stacked_pred - base_pred))), "pred_sd": float(np.std(base_pred)), "rows_kept": int(keep.sum()), "rows": len(train)}
    return out


def main():
    stage = sys.argv[1] if len(sys.argv) > 1 else "all"
    OUT.mkdir(parents=True, exist_ok=True)
    features = pd.read_parquet(FEATURES)
    STATE["halves"] = load_halves(features)
    h = STATE["halves"]
    reg = features.loc[features["result"].notna(), ["game_id", "season"]]
    cover = h.reindex(reg["game_id"]).dropna(subset=["m_q2", "m_q3", "m_q4"])
    print("games with all quarter margins", len(cover), "of", len(reg), flush=True)
    frame = regular_season_rows(features)
    frame = frame.loc[frame["ats_margin"].notna() & frame["result"].notna()].copy()
    if stage == "check":
        print(json.dumps(equivalence(frame), indent=1))
        return
    sel = select(frame)
    if stage == "select":
        return
    pop = proxy_population(features)
    base = pd.read_parquet(U1 / "base_per_game.parquet")
    base["margin_vs_open"] = np.nan
    arms = {}
    for arm in ARMS:
        path = OUT / f"{arm}_per_game.parquet"
        if path.exists():
            arms[arm] = pd.read_parquet(path)
        elif arm == "h2_multitask_halves" and "h1_stacked_halves" in arms:
            arms[arm] = arms["h1_stacked_halves"].copy()
            arms[arm].to_parquet(path)
        else:
            arms[arm] = run_arm(arm, features, pop, sel)
            arms[arm].to_parquet(path)
    ref = arms["h1_stacked_halves"]
    base["margin_vs_open"] = ref["margin_vs_open"].reindex(base.index)
    base["open_line"] = ref["open_line"].reindex(base.index)
    base = base.dropna(subset=["margin_vs_open"])
    tb = base.loc[base["season"].ge(2020)]
    print("base true era", int(tb["correct"].sum()), len(tb), flush=True)
    rb = metrics_recal(base.loc[base["season"].ge(2020)])
    rps_in = {"base": rb["pr"]}
    for a, df in arms.items():
        rps_in[a] = metrics_recal(df.loc[df["season"].ge(2020)])["pr"]
    rp = rps_table(rps_in)
    report = {"equivalence": equivalence(frame), "selection": {a: {s: v["alpha"] for s, v in sel[a].items()} for a in ARMS}, "base_rps_true": float(rp["base"].mean()), "arms": {}}
    for a, df in arms.items():
        report["arms"][a] = compare(df, base, {"base": rp["base"], "arm": rp[a]})
        print(a, report["arms"][a]["pooled_2011_2025"]["d_acc"], flush=True)
    report["stability"] = stability(frame, sel)
    (OUT / "report.json").write_text(json.dumps(report, indent=1, default=str))
    print(json.dumps(report, indent=1, default=str))


main()
