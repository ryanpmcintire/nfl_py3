import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import brentq, minimize
from scipy.special import expit, logit
from scipy.stats import norm

sys.path.insert(0, str(Path(__file__).resolve().parent))

import mod23_unit3 as u3
from lead66_unit1 import atomic_counts, mass_at_theta, select_band

from nfl_ats import clv
from nfl_ats.constants import DEFAULT_MIN_TRAIN_GAMES
from nfl_ats.home_side_location import (
    HOME_SIDE_OFFSET_SERVED,
    fit_home_side_offsets,
    prior_rows_before,
)
from nfl_ats.margin import (
    _PROFILE_SUPPRESSED_MISSING_INDICATORS,
    fit_margin_model,
    make_margin_estimator,
    margin_feature_columns,
)
from nfl_ats.mass_preserving_lattice import (
    DiscretePushReader,
    prior_pool,
    prior_pool_for_week,
    serve_discrete_three_way,
)
from nfl_ats.modeling import regular_season_rows

ROOT = u3.ROOT
OUT = ROOT / "artifacts" / "mod24_u5"
PBP = ROOT / "data" / "pbp" / "raw" / "20260929T191306Z"
SBR = ROOT / "data" / "processed" / "sbr_odds.parquet"
OPENER = ROOT / "artifacts" / "opener_evaluation" / "20260929T192743Z"
PROFILE = "weak_stack"
BASE_ALPHA = 10.0
PROB_METHOD = "gaussian_median"
FULL = tuple(margin_feature_columns("market_residual", PROFILE))
SUPP = _PROFILE_SUPPRESSED_MISSING_INDICATORS.get(PROFILE, ())
PROXY = tuple(range(2011, 2020))
TRUE = tuple(range(2020, 2026))
LAMBDAS = (0.0, 0.25, 0.5, 0.75, 1.0)
INNER = 4
WP_LO, WP_HI = 0.05, 0.95
BOOT = 20000
SEED = 20261001
DOWNS = (1, 2, 3, 4)
TYPES = ("pass", "run")
ARMS = ("base", "t1_epa", "t2_deserved", "t3_blend")


def game_aggregates():
    parts = []
    for season in range(2009, 2026):
        p = pd.read_parquet(
            PBP / f"season={season}" / "plays.parquet",
            columns=[
                "game_id",
                "home_team",
                "away_team",
                "posteam",
                "play_type",
                "down",
                "epa",
                "success",
                "wp",
                "fumble_lost",
            ],
        )
        p = p.loc[p["epa"].notna() & p["posteam"].notna()].copy()
        p["side"] = np.where(p["posteam"].eq(p["home_team"]), 1.0, -1.0)
        p["season"] = season
        parts.append(p)
    plays = pd.concat(parts, ignore_index=True)
    d1 = plays.assign(v=plays["epa"] * plays["side"]).groupby("game_id")["v"].sum().rename("d1")
    el = plays.loc[
        plays["play_type"].isin(TYPES) & plays["wp"].between(WP_LO, WP_HI) & plays["success"].notna()
    ].copy()
    el["fl"] = el["fumble_lost"].fillna(0).eq(1)
    clean = el.loc[~el["fl"]]
    d2 = (
        clean.assign(v=clean["epa"] * clean["side"])
        .groupby("game_id")["v"]
        .sum()
        .rename("d2_nofumble")
    )
    s2 = (
        el.assign(v=el["success"].astype(float) * el["side"]).groupby("game_id")["v"].sum().rename("s2")
    )
    fum = el.loc[el["fl"]]
    cols = {}
    for t in TYPES:
        for dn in DOWNS:
            sel = fum.loc[fum["play_type"].eq(t) & fum["down"].eq(dn)]
            cols[f"f_{t}_{dn}"] = sel.groupby("game_id")["side"].sum()
    fdf = pd.DataFrame(cols)
    agg = pd.concat([d1, d2, s2, fdf], axis=1).fillna(0.0)
    tab = clean.groupby(["season", "play_type", "down"])["epa"].agg(["sum", "count"]).reset_index()
    return agg, tab


def fumble_means(tab, asof):
    sub = tab.loc[tab["season"].lt(asof)].groupby(["play_type", "down"])[["sum", "count"]].sum()
    return {f"f_{t}_{int(dn)}": float(r["sum"] / r["count"]) for (t, dn), r in sub.iterrows()}


class Targets:
    def __init__(self, base, agg, tab):
        self.base = base.set_index("game_id")[["season", "spread_line", "result"]].join(agg, how="left")
        self.tab = tab
        self.has = self.base["d1"].notna()
        self.cache = {}

    def fit(self, asof):
        if asof in self.cache:
            return self.cache[asof]
        b = self.base
        tr = b.loc[self.has & b["season"].lt(asof) & b["result"].notna()]
        fm = fumble_means(self.tab, asof)
        d2 = b["d2_nofumble"] + sum(b[k] * v for k, v in fm.items())
        x1 = np.column_stack([np.ones(len(tr)), tr["d1"]])
        c1 = np.linalg.lstsq(x1, tr["result"].to_numpy(float), rcond=None)[0]
        d2tr = d2.loc[tr.index]
        x2 = np.column_stack([np.ones(len(tr)), d2tr, tr["s2"]])
        c2 = np.linalg.lstsq(x2, tr["result"].to_numpy(float), rcond=None)[0]
        t1 = c1[0] + c1[1] * b["d1"]
        t2 = c2[0] + c2[1] * d2 + c2[2] * b["s2"]
        out = {
            "t1": t1,
            "t2": t2,
            "c1": c1.tolist(),
            "c2": c2.tolist(),
            "r2_1": float(1 - np.var(tr["result"] - x1 @ c1) / np.var(tr["result"])),
            "r2_2": float(1 - np.var(tr["result"] - x2 @ c2) / np.var(tr["result"])),
        }
        self.cache[asof] = out
        return out

    def margin(self, asof, arm, lam=None):
        f = self.fit(asof)
        real = self.base["result"]
        if arm == "t1_epa":
            m = f["t1"]
        elif arm == "t2_deserved":
            m = f["t2"]
        elif arm == "t3_blend":
            m = lam * real + (1 - lam) * f["t2"]
        else:
            m = real
        m = m.where(self.has & real.notna(), real)
        return m

    def ats(self, asof, arm, lam=None):
        return (self.margin(asof, arm, lam) - self.base["spread_line"]).where(self.base["result"].notna())


def fit_k(z, y):
    def nll(par):
        q = par[0] * z
        return float(np.mean(np.logaddexp(0, q) - y * q))

    return float(minimize(nll, [1.0], method="BFGS").x[0])


def inner_lambda(frame, tg, season, first=2010):
    inner = list(range(max(first, season - INNER), season))
    scores = {}
    for lam in LAMBDAS:
        preds, ys = [], []
        for t in inner:
            train = frame.loc[frame["season"].lt(t)]
            test = frame.loc[frame["season"].eq(t)]
            tgt = tg.ats(t, "t3_blend", lam)
            est = make_margin_estimator("ridge", 42, ridge_alpha=BASE_ALPHA, suppressed_indicator_columns=SUPP)
            est.fit(train.loc[:, list(FULL)], tgt.loc[train["game_id"]].to_numpy(float))
            pred = np.asarray(est.predict(test.loc[:, list(FULL)]), dtype=float)
            margin = (test["result"] - test["spread_line"]).to_numpy(float)
            keep = margin != 0
            preds.append(pred[keep])
            ys.append((margin[keep] > 0).astype(float))
        z, y = np.concatenate(preds), np.concatenate(ys)
        k = fit_k(z, y)
        q = k * z
        scores[lam] = float(np.mean(np.logaddexp(0, q) - y * q))
    best = min(LAMBDAS, key=lambda l: (scores[l], -l))
    return best, scores


def finish(df):
    df = df.copy()
    df["pick_home"] = df["p"].to_numpy() >= 0.5
    m = df["margin_vs_open"]
    df["correct"] = np.where(m.eq(0), np.nan, np.where(df["pick_home"], m.gt(0), ~m.gt(0)))
    df = df.loc[df["correct"].notna()].copy()
    df["y"] = (df["margin_vs_open"] > 0).astype(float)
    pc = df["p"].clip(1e-6, 1 - 1e-6)
    df["ll"] = -(df["y"] * np.log(pc) + (1 - df["y"]) * np.log(1 - pc))
    df["brier"] = (pc - df["y"]) ** 2
    return df.set_index("game_id")[["season", "week", "p", "pick_home", "correct", "y", "ll", "brier"]]


def true_eval(features, season, tag):
    cfg = {
        "feature_profile": PROFILE,
        "regressor": "ridge",
        "target": "market_residual",
        "probability_method": PROB_METHOD,
        "calibration_method": "none",
        "ridge_alpha": BASE_ALPHA,
        "mod24_u5_variant": tag,
    }
    f = features.loc[features["season"].le(season)]
    scored = clv.opener_pick_evaluation(
        u3.MARKET_ROOT, f, active_model_config=cfg, min_train_games=DEFAULT_MIN_TRAIN_GAMES
    )
    scored = scored.loc[scored["season"].eq(season)]
    scored = scored.loc[scored["correct_at_open_probability_rule"].notna()].copy()
    scored["p"] = scored["home_cover_probability_at_open"]
    return finish(scored[["game_id", "season", "week", "p", "margin_vs_open"]])


def proxy_population(features):
    reg = regular_season_rows(features)
    reg = reg.loc[reg["season"].between(PROXY[0], PROXY[-1])]
    sbr = pd.read_parquet(SBR).dropna(subset=["game_id", "open_home_spread"]).drop_duplicates("game_id")
    pop = reg[["game_id", "season", "week", "result"]].merge(
        sbr[["game_id", "open_home_spread"]], on="game_id", how="inner"
    )
    pop = pop.loc[pd.to_numeric(pop["result"], errors="coerce").notna()].copy()
    pop["season"] = pop["season"].astype(int)
    pop["week"] = pop["week"].astype(int)
    return pop.rename(columns={"open_home_spread": "open_line"})


def proxy_eval(features, pop, tmap_for_season):
    frame = regular_season_rows(features).copy()
    frame["gameday"] = pd.to_datetime(frame["gameday"])
    completed = frame.loc[frame["result"].notna()].copy()
    pool = prior_pool(frame, pop.set_index("game_id")["open_line"])
    rows = []
    stream = pd.DataFrame()
    for (season, week), group in pop.groupby(["season", "week"], sort=True):
        week_rows = frame.loc[frame["game_id"].isin(set(group["game_id"]))]
        target = frame.loc[frame["season"].eq(season) & frame["week"].eq(week)]
        cutoff = target["gameday"].min()
        training = completed.loc[completed["gameday"].lt(cutoff)].copy()
        if len(training) < DEFAULT_MIN_TRAIN_GAMES:
            continue
        tm = tmap_for_season(int(season))
        training["ats_margin"] = training["game_id"].map(tm).fillna(training["ats_margin"])
        model = fit_margin_model(
            training,
            target="market_residual",
            model_name="ridge",
            feature_profile=PROFILE,
            ridge_alpha=BASE_ALPHA,
        )
        reader = DiscretePushReader.for_week(
            pool,
            season=int(season),
            week=int(week),
            cutoff=pd.Timestamp(cutoff),
            exclude_game_ids=set(week_rows["game_id"].astype(str)),
        )
        scoring = week_rows.merge(group[["game_id", "open_line"]], on="game_id").copy()
        at_open = scoring.copy()
        at_open["spread_line"] = at_open["open_line"]
        pred = model.predict(at_open, probability_method=PROB_METHOD)
        if HOME_SIDE_OFFSET_SERVED and not stream.empty:
            fitted = fit_home_side_offsets(prior_rows_before(stream, int(season), int(week)))
            off = fitted.offset_for(at_open["spread_line"]).fillna(0.0).to_numpy(dtype=float)
        else:
            off = np.zeros(len(at_open))
        served = (
            model.predict(at_open, probability_method=PROB_METHOD, center_offset=off)
            if np.any(off != 0.0)
            else pred
        )
        disc = serve_discrete_three_way(
            served, at_open, reader, residuals=model.residuals, probability_method=PROB_METHOD
        )
        res = pred["predicted_market_residual"].to_numpy()
        out = scoring[["game_id", "result", "open_line"]].copy()
        out["season"] = int(season)
        out["week"] = int(week)
        out["p"] = disc["home_cover_probability"].to_numpy()
        rows.append(out)
        ws = pd.DataFrame(
            {
                "game_id": scoring["game_id"].astype(str).to_numpy(),
                "season": int(season),
                "week": int(week),
                "spread_line": scoring["open_line"].to_numpy(dtype=float),
                "point_incumbent": scoring["open_line"].to_numpy(dtype=float) + res,
                "result": pd.to_numeric(scoring["result"], errors="coerce").to_numpy(),
            }
        )
        stream = ws if stream.empty else pd.concat([stream, ws], ignore_index=True)
    out = pd.concat(rows, ignore_index=True)
    out["margin_vs_open"] = out["result"] - out["open_line"]
    return finish(out)


def recal_temp(df):
    out = pd.Series(np.nan, index=df.index)
    z = logit(df["p"].clip(1e-6, 1 - 1e-6))
    for s in sorted(df["season"].unique()):
        tr = df["season"].ne(s)
        k = fit_k(z[tr].to_numpy(), df.loc[tr, "y"].to_numpy())
        m = df["season"].eq(s)
        out[m] = expit(k * z[m])
    return out


def temp_scores(df):
    pt = recal_temp(df).clip(1e-6, 1 - 1e-6)
    y = df["y"]
    return pd.DataFrame(
        {
            "ll_t": -(y * np.log(pt) + (1 - y) * np.log(1 - pt)),
            "brier_t": (pt - y) ** 2,
        },
        index=df.index,
    )


def theta_for(atoms, counts, line, target):
    home, away = atoms > line, atoms < line

    def diff(theta):
        mass = mass_at_theta(atoms, counts, theta)
        return float(mass[home].sum() / mass[home | away].sum() - target)

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


def lattice_rps(arms_true, temp_p, features):
    f = features[["game_id", "season", "week", "gameday", "spread_line", "result"]]
    opener = pd.read_parquet(OPENER / "per_game.parquet")
    opener = opener.loc[opener["season"].between(2020, 2025)].copy()
    line = opener.set_index("game_id")["tue_open_home_spread"]
    pool = prior_pool(f, line)
    res = f.set_index("game_id")["result"]
    games = opener.assign(gameday=pd.to_datetime(opener["game_id"].map(f.set_index("game_id")["gameday"])))
    out = {a: {} for a in arms_true}
    for (season, week), group in games.groupby(["season", "week"], sort=True):
        target = f.loc[f["season"].eq(season) & f["week"].eq(week)]
        prior = prior_pool_for_week(
            pool,
            season=int(season),
            week=int(week),
            cutoff=pd.to_datetime(target["gameday"]).min(),
            exclude_game_ids=group["game_id"],
        )
        for gid in group["game_id"]:
            ln = float(line.loc[gid])
            actual = float(res.loc[gid])
            atoms, counts = atomic_counts(select_band(prior, ln)[0])
            for a in arms_true:
                if gid not in temp_p[a].index:
                    continue
                t = theta_for(atoms, counts, ln, float(temp_p[a].loc[gid]))
                if t is not None:
                    out[a][gid] = rps(atoms, mass_at_theta(atoms, counts, t), actual)
    return {a: pd.Series(v) for a, v in out.items()}


def boot_groups(df, col, keys, seed=SEED):
    k = df[keys].astype(str).agg("-".join, axis=1) if len(keys) > 1 else df[keys[0]]
    g = df.groupby(k)[col].agg(["sum", "count"])
    s, n = g["sum"].to_numpy(), g["count"].to_numpy()
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(s), (BOOT, len(s)))
    d = s[idx].sum(1) / n[idx].sum(1)
    return {"lo": float(np.percentile(d, 2.5)), "hi": float(np.percentile(d, 97.5)), "pp": float((d > 0).mean())}


def compare(arm, base, extra=None):
    j = arm.join(base, rsuffix="_b", how="inner")
    j["d_acc"] = 100.0 * (j["correct"] - j["correct_b"])
    j["d_ll"] = j["ll_b"] - j["ll"]
    j["d_brier"] = j["brier_b"] - j["brier"]
    j["d_ll_t"] = j["ll_t_b"] - j["ll_t"]
    j["d_brier_t"] = j["brier_t_b"] - j["brier_t"]
    if extra is not None:
        j["d_rps_t"] = extra
    out = {}
    for name, mask in (
        ("true_2020_2025", j["season"].ge(2020)),
        ("proxy_2011_2019", j["season"].le(2019)),
        ("pooled_2011_2025", j["season"].ge(0)),
    ):
        s = j.loc[mask]
        if len(s) == 0:
            continue
        w = int(s["correct"].sum())
        flips = s["pick_home"] != s["pick_home_b"]
        r = {
            "n": len(s),
            "record": f"{w}-{len(s) - w}",
            "record_base": f"{int(s['correct_b'].sum())}-{len(s) - int(s['correct_b'].sum())}",
            "flips": int(flips.sum()),
            "flips_W_L": f"{int((flips & s['correct'].eq(1.0)).sum())}-{int((flips & s['correct'].eq(0.0)).sum())}",
            "d_acc": float(s["d_acc"].mean()),
            "d_acc_season": boot_groups(s, "d_acc", ["season"]),
            "d_acc_week": boot_groups(s, "d_acc", ["season", "week"]),
            "seasons_pos": int((s.groupby("season")["d_acc"].mean() > 0).sum()),
            "seasons": int(s["season"].nunique()),
        }
        for m in ("d_ll", "d_brier", "d_ll_t", "d_brier_t"):
            r[m] = float(s[m].mean())
            r[m + "_season"] = boot_groups(s, m, ["season"])
        if "d_rps_t" in s and s["d_rps_t"].notna().any():
            sr = s.dropna(subset=["d_rps_t"])
            r["d_rps_t"] = float(sr["d_rps_t"].mean())
            r["d_rps_t_season"] = boot_groups(sr, "d_rps_t", ["season"])
        out[name] = r
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    stage = sys.argv[1] if len(sys.argv) > 1 else "all"
    features = pd.read_parquet(u3.FEATURES)
    frame = regular_season_rows(features)
    frame = frame.loc[frame["ats_margin"].notna() & frame["result"].notna()].copy()
    agg, tab = game_aggregates()
    tg = Targets(features, agg, tab)
    print("games with pbp aggregates", int(tg.has.sum()), "of", len(tg.has), flush=True)

    sd = {}
    f25 = tg.fit(2025)
    b = tg.base
    m = tg.has & b["season"].lt(2025) & b["result"].notna()
    for name, ser in (
        ("real", b["result"]),
        ("t1_epa", f25["t1"]),
        ("t2_deserved", f25["t2"]),
        ("t3_blend_0.5", 0.5 * b["result"] + 0.5 * f25["t2"]),
    ):
        sd[name] = {
            "sd_minus_line": float((ser - b["spread_line"])[m].std()),
            "sd_raw": float(ser[m].std()),
        }
    sd["fit_asof_2025"] = {k: f25[k] for k in ("c1", "c2", "r2_1", "r2_2")}
    lam = {}
    for s in PROXY + TRUE:
        best, sc = inner_lambda(frame, tg, s)
        lam[s] = {"lambda": best, "inner_log_loss": sc}
        print("lambda", s, best, {k: round(v, 5) for k, v in sc.items()}, flush=True)
    (OUT / "targets_sd_lambda.json").write_text(json.dumps({"sd": sd, "lambda": lam}, indent=2, default=str))
    print(json.dumps(sd, indent=1), flush=True)
    if stage == "targets":
        return

    pop = proxy_population(features)
    per_arm = {}
    for arm in ARMS:
        path = OUT / f"{arm}_per_game.parquet"
        if path.exists():
            per_arm[arm] = pd.read_parquet(path)
            continue

        def amap(s, arm=arm):
            lm = lam[s]["lambda"] if arm == "t3_blend" else None
            return tg.ats(s, arm, lm)

        parts = [proxy_eval(features, pop, lambda s, amap=amap: amap(s))]
        for s in TRUE:
            ext = features.copy()
            tgt = amap(s)
            ext["ats_margin"] = ext["game_id"].map(tgt).where(ext["ats_margin"].notna())
            ext["ats_margin"] = ext["ats_margin"].fillna(features["ats_margin"])
            parts.append(true_eval(ext, s, f"{arm}-{s}-{lam[s]['lambda'] if arm == 't3_blend' else ''}"))
            print(arm, s, int(parts[-1]["correct"].sum()), len(parts[-1]), flush=True)
        df = pd.concat(parts)
        df.to_parquet(path)
        per_arm[arm] = df
    base = per_arm["base"]
    for arm, df in per_arm.items():
        per_arm[arm] = df.join(
            pd.concat([temp_scores(df.loc[df["season"].le(2019)]), temp_scores(df.loc[df["season"].ge(2020)])])
        )
    p_t = {}
    for arm, df in per_arm.items():
        t = df.loc[df["season"].ge(2020)]
        p_t[arm] = recal_temp(t)
    rps_by = lattice_rps(list(per_arm), p_t, features)
    results = {}
    brps = rps_by["base"]
    for arm in ARMS:
        extra = (brps - rps_by[arm]).reindex(per_arm[arm].index)
        if arm == "base":
            tr = per_arm["base"].loc[per_arm["base"]["season"].ge(2020)]
            results["base"] = {
                "true_record": f"{int(tr['correct'].sum())}-{len(tr) - int(tr['correct'].sum())}",
                "proxy_record": f"{int(per_arm['base'].loc[per_arm['base']['season'].le(2019), 'correct'].sum())}-{int((per_arm['base']['season'].le(2019)).sum()) - int(per_arm['base'].loc[per_arm['base']['season'].le(2019), 'correct'].sum())}",
                "ll_t": float(tr["ll_t"].mean()),
                "brier_t": float(tr["brier_t"].mean()),
                "rps_t": float(brps.mean()),
            }
            continue
        results[arm] = compare(per_arm[arm], per_arm["base"], extra)
        results[arm]["arm_rps_t"] = float(rps_by[arm].mean())
        print(arm, results[arm]["pooled_2011_2025"]["record"], results[arm]["pooled_2011_2025"]["d_acc"], flush=True)
    (OUT / "report.json").write_text(json.dumps(results, indent=2, default=str))
    print(json.dumps(results, indent=1, default=str))


main()
