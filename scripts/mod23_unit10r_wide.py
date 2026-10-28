import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import HuberRegressor

sys.path.insert(0, str(Path(__file__).resolve().parent))

import mod23_unit3 as u3

from nfl_ats import clv
from nfl_ats import margin as M

HARNESS_PATH = Path(__file__).resolve().parent / "mod24_u1.py"
HARNESS = {"__file__": str(HARNESS_PATH), "__name__": "mod24_u1_harness"}
exec(compile(HARNESS_PATH.read_text().rsplit("\nmain()", 1)[0], str(HARNESS_PATH), "exec"), HARNESS)

ROOT = u3.ROOT
OUT = ROOT / "artifacts" / "mod23_unit10r_wide"
PBP = sorted((ROOT / "data" / "pbp" / "raw").glob("2*"))[-1]
BASE5 = ROOT / "artifacts" / "mod24_u5" / "base_per_game.parquet"
FULL = HARNESS["FULL"]
BASE_ALPHA = HARNESS["BASE_ALPHA"]
SUPP = HARNESS["SUPP"]
BOOT = 20000
SEED = 20261019
QGRID = (0.40, 0.50, 0.60, 0.70, 0.80, 1.00)
MIN_PRIOR_SEASONS = 3
MAD_TO_SIGMA = 1.4826
CTX = {"t_map": None, "arm": "W", "q_log": {}, "q_cache": {}, "cache_on": False}


def label(frame):
    if CTX["arm"] == "T":
        return frame["game_id"].map(CTX["t_map"]).to_numpy(dtype=float)
    return pd.to_numeric(frame["ats_margin"]).to_numpy(dtype=float)


def removed_points(season):
    p = pd.read_parquet(PBP / f"season={season}" / "plays.parquet")
    p = p.loc[p["season_type"].eq("REG")].sort_values(["game_id", "play_id"]).copy()
    home_pre = np.where(p["posteam"].eq(p["home_team"]), p["posteam_score"], p["posteam_score"] - p["score_differential"])
    away_pre = np.where(p["posteam"].eq(p["away_team"]), p["posteam_score"], p["posteam_score"] - p["score_differential"])
    p["h"] = home_pre
    p["a"] = away_pre
    p[["h", "a"]] = p.groupby("game_id")[["h", "a"]].ffill()
    p["dh"] = (p.groupby("game_id")["h"].shift(-1) - p["h"]).fillna(0.0).clip(lower=0.0)
    p["da"] = (p.groupby("game_id")["a"].shift(-1) - p["a"]).fillna(0.0).clip(lower=0.0)
    garbage = p["qtr"].eq(4) & (p["wp"].lt(0.05) | p["wp"].gt(0.95))
    nonoff = p["touchdown"].eq(1) & (~p["play_type"].isin(["pass", "run"]) | p["interception"].eq(1) | p["fumble_lost"].eq(1))
    extra = p["play_type_nfl"].isin(["XP_KICK", "PAT2"])
    prev_removed = (nonoff | garbage).groupby(p["game_id"]).shift(1).fillna(False).astype(bool)
    remove = garbage | nonoff | (extra & prev_removed)
    p["rh"] = np.where(remove, p["dh"], 0.0)
    p["ra"] = np.where(remove, p["da"], 0.0)
    g = p.groupby("game_id").agg(rh=("rh", "sum"), ra=("ra", "sum"), h_end=("h", "max"), a_end=("a", "max"))
    return g


def add_target_t(features):
    parts = [removed_points(s) for s in sorted(features["season"].unique()) if (PBP / f"season={s}").exists()]
    g = pd.concat(parts)
    f = features.merge(g[["rh", "ra"]], left_on="game_id", right_index=True, how="left")
    have = f["rh"].notna()
    print("pbp coverage", float(have.mean()), "mean removed home/away", float(f["rh"].mean()), float(f["ra"].mean()), flush=True)
    f["ats_margin_t"] = pd.to_numeric(f["ats_margin"]) - (f["rh"] - f["ra"]).fillna(0.0)
    f["adj_delta"] = (f["rh"] - f["ra"]).fillna(0.0)
    return f.drop(columns=["rh", "ra"])


def make_pipe(alpha):
    return M.make_margin_estimator("ridge", 42, ridge_alpha=alpha, suppressed_indicator_columns=SUPP)


def sigma_of(x, y):
    base = make_pipe(BASE_ALPHA)
    base.fit(x, y)
    r = y - np.asarray(base.predict(x), dtype=float)
    return MAD_TO_SIGMA * float(np.median(np.abs(r - np.median(r)))), r


def fit_robust(x, y, arm, q):
    if q >= 1.0:
        est = make_pipe(BASE_ALPHA)
        est.fit(x, y)
        return est
    if arm == "T":
        est = make_pipe(BASE_ALPHA)
        est.fit(x, y)
        return est
    if arm == "W":
        cut = float(np.quantile(np.abs(y), q))
        est = make_pipe(BASE_ALPHA)
        est.fit(x, np.clip(y, -cut, cut))
        return est
    sigma, r = sigma_of(x, y)
    eps = max(float(np.quantile(np.abs(r) / sigma, q)), 1.0 + 1e-6)
    est = make_pipe(BASE_ALPHA)
    est.steps[-1] = ("regressor", HuberRegressor(epsilon=eps, alpha=BASE_ALPHA / sigma, max_iter=1000))
    est.fit(x, y)
    return est


def select_q(training, arm):
    latest = int(training["season"].max())
    key = (latest, arm)
    if CTX["cache_on"] and key in CTX["q_cache"]:
        return CTX["q_cache"][key]
    prior = training.loc[training["season"].lt(latest)].reset_index(drop=True)
    seasons = sorted(prior["season"].unique())
    if len(seasons) < MIN_PRIOR_SEASONS:
        return 1.0
    y = label(prior)
    cols = list(FULL)
    mae = {}
    for q in QGRID:
        err = 0.0
        for s in seasons:
            te = prior["season"].to_numpy() == s
            est = fit_robust(prior.loc[~te, cols], y[~te], arm, q)
            err += float(np.sum(np.abs(y[te] - np.asarray(est.predict(prior.loc[te, cols]), dtype=float))))
        mae[q] = err / len(prior)
    chosen = min(mae, key=mae.get)
    if CTX["cache_on"]:
        CTX["q_cache"][key] = chosen
    return chosen


class Wrapped:
    def __init__(self, est, cols):
        self.est = est
        self.cols = cols

    def predict(self, x):
        return np.asarray(self.est.predict(x.loc[:, self.cols]), dtype=float)


def fit_part(part, arm, q):
    cols = list(FULL)
    return Wrapped(fit_robust(part.loc[:, cols], label(part), arm, q), cols)


def robust_fit(frame, *, target="margin", model_name="ridge", distribution_fraction=0.20,
               min_distribution_rows=10, random_state=42, feature_profile="base", ridge_alpha=10.0,
               column_penalties=None):
    feature_columns = M.margin_feature_columns(target, feature_profile)
    training = M.regular_season_rows(frame)
    training = training.loc[M._target_values(training, target).notna()].copy()
    training["gameday"] = pd.to_datetime(training["gameday"], errors="raise")
    training = training.sort_values(["gameday", "game_id"]).reset_index(drop=True)
    q = 1.0 if CTX["arm"] == "T" else select_q(training, CTX["arm"])
    CTX["q_log"].setdefault(int(training["season"].max()), q)
    n = len(training)
    split = n - int(n * distribution_fraction)
    fit_p = training.iloc[:split]
    dist_part = training.iloc[split:]
    tmp = fit_part(fit_p, CTX["arm"], q)
    cal = tmp.predict(dist_part)
    residuals = M._target_values(dist_part, target).to_numpy(dtype=float) - cal
    residuals = residuals[np.isfinite(residuals)]
    est = fit_part(training, CTX["arm"], q)
    return M.MarginModel(
        estimator=est, residuals=residuals, model_name=model_name, ridge_alpha=None, target=target,
        feature_columns=feature_columns, training_rows=len(training), distribution_rows=len(residuals),
        training_max_gameday=training["gameday"].max().date().isoformat(), column_penalties=None,
    )


def install():
    clv.fit_margin_model = robust_fit
    HARNESS["fit_margin_model"] = robust_fit


def boot_season(df, col):
    g = df.groupby("season")[col].agg(["sum", "count"])
    s, n = g["sum"].to_numpy(), g["count"].to_numpy()
    rng = np.random.default_rng(SEED)
    idx = rng.integers(0, len(s), (BOOT, len(s)))
    d = s[idx].sum(1) / n[idx].sum(1)
    return {
        "low": float(np.percentile(d, 2.5)),
        "high": float(np.percentile(d, 97.5)),
        "pp": float((d > 0).mean()),
        "blocks": len(s),
    }


def market_only(j):
    y = np.where(j["pick_home_b"], j["correct_b"], 1.0 - j["correct_b"])
    j = j.assign(y=y)
    ll, br = [], []
    for s, t in j.groupby("season"):
        prior = j.loc[j["season"].lt(s) & (j["season"].le(2019) == (s <= 2019)), "y"]
        p = float(prior.mean()) if len(prior) else 0.5
        p = min(max(p, 1e-6), 1 - 1e-6)
        ll.append(-(t["y"] * np.log(p) + (1 - t["y"]) * np.log(1 - p)))
        br.append((p - t["y"]) ** 2)
    return pd.concat(ll), pd.concat(br)


def compare(arm, base):
    j = arm.join(base, rsuffix="_b", how="inner")
    j["d_acc"] = 100.0 * (j["correct"] - j["correct_b"])
    mll, mbr = market_only(j)
    j["ll_m"] = mll
    j["brier_m"] = mbr
    out = {}
    for name, mask in (("sbr_2011_2019", j["season"].le(2019)), ("tuesday_2020_2025", j["season"].ge(2020))):
        s = j.loc[mask]
        w = int(s["correct"].sum())
        flips = s["pick_home"] != s["pick_home_b"]
        per = {}
        for season, t in s.groupby("season"):
            per[int(season)] = {
                "record": f"{int(t['correct'].sum())}-{len(t) - int(t['correct'].sum())}",
                "base": f"{int(t['correct_b'].sum())}-{len(t) - int(t['correct_b'].sum())}",
                "d_acc": float(t["d_acc"].mean()),
            }
        out[name] = {
            "n": len(s),
            "record": f"{w}-{len(s) - w}",
            "record_base": f"{int(s['correct_b'].sum())}-{len(s) - int(s['correct_b'].sum())}",
            "flips": int(flips.sum()),
            "flips_W_L": f"{int((flips & s['correct'].eq(1.0)).sum())}-{int((flips & s['correct'].eq(0.0)).sum())}",
            "d_acc": float(s["d_acc"].mean()),
            "d_acc_season_boot": boot_season(s, "d_acc"),
            "ll_arm": float(s["ll"].mean()),
            "ll_base": float(s["ll_b"].mean()),
            "ll_market": float(s["ll_m"].mean()),
            "brier_arm": float(s["brier"].mean()),
            "brier_base": float(s["brier_b"].mean()),
            "brier_market": float(s["brier_m"].mean()),
            "per_season": per,
        }
    return out


def reliability(run, base_idx):
    r = run.loc[run.index.isin(base_idx)].copy()
    r["y"] = np.where(r["pick_home"], r["correct"], 1.0 - r["correct"])
    out = {}
    for name, mask in (("sbr_2011_2019", r["season"].le(2019)), ("tuesday_2020_2025", r["season"].ge(2020))):
        s = r.loc[mask].copy()
        s["bin"] = pd.qcut(s["p"], 10, labels=False, duplicates="drop")
        g = s.groupby("bin").agg(n=("y", "size"), mean_p=("p", "mean"), mean_y=("y", "mean"))
        out[name] = g.round(4).reset_index().to_dict("records")
    return out


def checks(features):
    reg = M.regular_season_rows(features)
    reg = reg.loc[reg["result"].notna()].copy()
    reg["gameday"] = pd.to_datetime(reg["gameday"])
    reg = reg.sort_values(["gameday", "game_id"]).reset_index(drop=True)
    train = reg.loc[reg["season"].le(2022)]
    test = reg.loc[reg["season"].eq(2023)]
    base = make_pipe(BASE_ALPHA)
    base.fit(train.loc[:, list(FULL)], label(train))
    p0 = fit_part(train, "W", 1.0)
    diff = float(np.max(np.abs(np.asarray(base.predict(test.loc[:, list(FULL)])) - p0.predict(test))))
    print("q=1 identity max abs prediction diff vs base:", diff, flush=True)
    if diff > 1e-6:
        raise SystemExit("IDENTITY FAILED")
    for arm in ("W", "H"):
        CTX["arm"] = arm
        a1 = select_q(train, arm)
        bad = train.copy()
        last = bad["season"].eq(int(bad["season"].max()))
        bad.loc[last, ["home_score", "away_score", "result", "ats_margin", "ats_margin_t"]] = 999.0
        a2 = select_q(bad, arm)
        print(arm, "selection identical when latest-season outcomes corrupted:", a1 == a2, a1, flush=True)
        if a1 != a2:
            raise SystemExit("LEAK CHECK FAILED")
        older = train.loc[train["season"].lt(int(train["season"].max()) - 3)].copy()
        older2 = older.copy()
        older2["ats_margin"] = pd.to_numeric(older2["ats_margin"]) + np.arange(len(older2)) % 17 * 4.0 - 30.0
        print(arm, "positive control selection differs when prior labels change:", select_q(older, arm) != select_q(older2, arm), flush=True)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    features = add_target_t(pd.read_parquet(u3.FEATURES))
    CTX["t_map"] = features.set_index("game_id")["ats_margin_t"]
    pop = HARNESS["proxy_population"](features)
    base = pd.read_parquet(BASE5)
    tr = base.loc[base["season"].ge(2020)]
    print("base true", int(tr["correct"].sum()), len(tr), flush=True)
    if (int(tr["correct"].sum()), len(tr)) != (802, 1503):
        raise SystemExit("base reproduction failed")
    sb = base.loc[base["season"].le(2019)]
    print("base proxy", int(sb["correct"].sum()), len(sb), flush=True)
    if (int(sb["correct"].sum()), len(sb)) != (1142, 2231):
        raise SystemExit("proxy base reproduction failed")
    checks(features)
    install()
    CTX["cache_on"] = True
    os.environ[clv.CACHE_DISABLED_ENV] = "1"
    results = {}
    runs = {}
    for name in ("W", "H", "T"):
        path = OUT / f"{name}_run.parquet"
        if path.exists():
            run = pd.read_parquet(path)
        else:
            CTX["arm"] = name
            CTX["q_log"] = {}
            run = HARNESS["run_arm"](features, pop, lambda s: (FULL, BASE_ALPHA))
            run.to_parquet(path)
            (OUT / f"{name}_q.json").write_text(json.dumps({str(k): v for k, v in sorted(CTX["q_log"].items())}, indent=1))
        run = run.loc[run.index.isin(base.index)]
        runs[name] = run
        results[name] = compare(run, base)
        for era, r in results[name].items():
            print(name, era, r["record"], "base", r["record_base"], "d_acc", round(r["d_acc"], 3), r["d_acc_season_boot"], "flips", r["flips"], r["flips_W_L"], "ll", round(r["ll_arm"], 4), round(r["ll_base"], 4), round(r["ll_market"], 4), "brier", round(r["brier_arm"], 4), round(r["brier_base"], 4), round(r["brier_market"], 4), flush=True)
    better = max(("W", "H", "T"), key=lambda k: sum(r["d_acc"] * r["n"] for r in results[k].values()))
    results["better_arm"] = better
    results["reliability_better"] = reliability(runs[better], base.index)
    (OUT / "report_all.json").write_text(json.dumps(results, indent=1, default=str))
    print("better arm", better, json.dumps(results["reliability_better"]), flush=True)
    cells = []
    for name in ("W", "H", "T"):
        for era, r in results[name].items():
            lo, hi = (2011, 2019) if era.startswith("sbr") else (2020, 2025)
            kind = {"W": "winsorized labels", "H": "Huber loss", "T": "margin without non-offensive touchdowns and garbage-time scoring"}[name]
            cells.append(
                {
                    "source": "artifacts/mod23_unit10r_wide/report_all.json",
                    "effect_units": "accuracy_points",
                    "classification": "unresolved_below_power",
                    "league": "nfl",
                    "season_start": lo,
                    "season_end": hi,
                    "family": "mod23_unit10_robust_target_wide",
                    "category": "modeling",
                    "classification_evidence": "No positive control and no admissible closing ground; season-blocked bootstrap interval reported.",
                    "notes": f"Base fit with {kind} at a cut chosen leave-one-season-out on prior seasons, arm {name}; six looks in this family (three arms, two eras), ten cumulative with the four of mod23_unit10_robust_target.",
                    "name": f"mod23_unit10_robust_target_wide_{name}_{era}_accuracy_diff",
                    "description": f"Robust target arm {name} paired vs base, {lo}-{hi}; {r['record']} vs {r['record_base']}.",
                    "effect": r["d_acc"],
                    "interval_low": r["d_acc_season_boot"]["low"],
                    "interval_high": r["d_acc_season_boot"]["high"],
                    "probability_positive": r["d_acc_season_boot"]["pp"],
                    "sample_games": r["n"],
                    "sample_blocks": r["d_acc_season_boot"]["blocks"],
                    "plain_summary": "Capping extreme game margins in training, or down-weighting them with a robust loss, moved picks against the opener by an unresolved amount.",
                }
            )
    (OUT / "weak_signals_batch.json").write_text(json.dumps({"cells": cells}, indent=1))


main()
