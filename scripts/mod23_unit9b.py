import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import mod23_unit3 as u3

from nfl_ats import clv
from nfl_ats import margin as M

HARNESS_PATH = Path(__file__).resolve().parent / "mod24_u1.py"
HARNESS = {"__file__": str(HARNESS_PATH), "__name__": "mod24_u1_harness"}
exec(compile(HARNESS_PATH.read_text().rsplit("\nmain()", 1)[0], str(HARNESS_PATH), "exec"), HARNESS)

ROOT = u3.ROOT
OUT = ROOT / "artifacts" / "mod23_unit9"
BASE5 = ROOT / "artifacts" / "mod24_u5" / "base_per_game.parquet"
FULL = HARNESS["FULL"]
BASE_ALPHA = HARNESS["BASE_ALPHA"]
SUPP = HARNESS["SUPP"]
BOOT = 20000
SEED = 20261018
ALPHA_GRID = (1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1000.0)
MIN_PRIOR_SEASONS = 3
DROP_HOME = ("away_off_", "home_def_")
DROP_AWAY = ("home_off_", "away_def_")
CTX = {"arm": "P0", "alpha_log": {}, "alpha_cache": {}, "cache_on": False}


def side_columns(side, arm):
    cols = list(FULL)
    if arm != "P2":
        return cols
    drop = DROP_HOME if side == "home" else DROP_AWAY
    return [c for c in cols if not c.startswith(drop)]


def side_targets(frame):
    spread = pd.to_numeric(frame["spread_line"]).to_numpy(dtype=float)
    total = pd.to_numeric(frame["total_line"]).to_numpy(dtype=float)
    if "home_score" in frame.columns:
        hs = pd.to_numeric(frame["home_score"]).to_numpy(dtype=float)
        as_ = pd.to_numeric(frame["away_score"]).to_numpy(dtype=float)
    else:
        hs = frame["game_id"].map(CTX["home_score"]).to_numpy(dtype=float)
        as_ = frame["game_id"].map(CTX["away_score"]).to_numpy(dtype=float)
    h = hs - (total + spread) / 2.0
    a = as_ - (total - spread) / 2.0
    return {"home": h, "away": a}


def make_pipe(alpha):
    return M.make_margin_estimator("ridge", 42, ridge_alpha=alpha, suppressed_indicator_columns=SUPP)


class TwoSided:
    def __init__(self, cols, pipes):
        self.cols = cols
        self.pipes = pipes

    def predict(self, x):
        return np.asarray(self.pipes["home"].predict(x.loc[:, self.cols["home"]]), dtype=float) - np.asarray(
            self.pipes["away"].predict(x.loc[:, self.cols["away"]]), dtype=float
        )


def select_alphas(training, arm):
    latest = int(training["season"].max())
    key = (latest, arm)
    if CTX.get("cache_on") and key in CTX["alpha_cache"]:
        return CTX["alpha_cache"][key]
    prior = training.loc[training["season"].lt(latest)].reset_index(drop=True)
    seasons = sorted(prior["season"].unique())
    if len(seasons) < MIN_PRIOR_SEASONS:
        return {"home": BASE_ALPHA, "away": BASE_ALPHA}
    tg = side_targets(prior)
    chosen = {}
    for side in ("home", "away"):
        cols = side_columns(side, arm)
        mse = {}
        for alpha in ALPHA_GRID:
            err = 0.0
            for s in seasons:
                te = prior["season"].to_numpy() == s
                est = make_pipe(alpha)
                est.fit(prior.loc[~te, cols], tg[side][~te])
                err += float(np.sum((tg[side][te] - np.asarray(est.predict(prior.loc[te, cols]), dtype=float)) ** 2))
            mse[alpha] = err / len(prior)
        chosen[side] = min(mse, key=mse.get)
    if CTX.get("cache_on"):
        CTX["alpha_cache"][key] = chosen
    return chosen


def fit_two_sided(part, arm, alphas):
    tg = side_targets(part)
    pipes, cols = {}, {}
    for side in ("home", "away"):
        cols[side] = side_columns(side, arm)
        est = make_pipe(alphas[side])
        est.fit(part.loc[:, cols[side]], tg[side])
        pipes[side] = est
    return TwoSided(cols, pipes)


def two_sided_fit(frame, *, target="margin", model_name="ridge", distribution_fraction=0.20,
                  min_distribution_rows=10, random_state=42, feature_profile="base", ridge_alpha=10.0,
                  column_penalties=None):
    feature_columns = M.margin_feature_columns(target, feature_profile)
    training = M.regular_season_rows(frame)
    training = training.loc[M._target_values(training, target).notna()].copy()
    training["gameday"] = pd.to_datetime(training["gameday"], errors="raise")
    training = training.sort_values(["gameday", "game_id"]).reset_index(drop=True)
    arm = CTX["arm"]
    alphas = select_alphas(training, arm) if arm == "P1" else {"home": BASE_ALPHA, "away": BASE_ALPHA}
    CTX["alpha_log"][int(training["season"].max())] = alphas
    n = len(training)
    split = n - int(n * distribution_fraction)
    fit_part = training.iloc[:split]
    dist_part = training.iloc[split:]
    tmp = fit_two_sided(fit_part, arm, alphas)
    cal = np.asarray(tmp.predict(dist_part), dtype=float)
    residuals = M._target_values(dist_part, target).to_numpy(dtype=float) - cal
    residuals = residuals[np.isfinite(residuals)]
    est = fit_two_sided(training, arm, alphas)
    return M.MarginModel(
        estimator=est, residuals=residuals, model_name=model_name, ridge_alpha=None, target=target,
        feature_columns=feature_columns, training_rows=len(training), distribution_rows=len(residuals),
        training_max_gameday=training["gameday"].max().date().isoformat(), column_penalties=None,
    )


def install():
    clv.fit_margin_model = two_sided_fit
    HARNESS["fit_margin_model"] = two_sided_fit


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
    j["d_ll"] = j["ll"] - j["ll_b"]
    j["d_brier"] = j["brier"] - j["brier_b"]
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


def checks(features):
    reg = M.regular_season_rows(features)
    reg = reg.loc[reg["result"].notna()].copy()
    reg["gameday"] = pd.to_datetime(reg["gameday"])
    reg = reg.sort_values(["gameday", "game_id"]).reset_index(drop=True)
    train = reg.loc[reg["season"].le(2022)]
    test = reg.loc[reg["season"].eq(2023)]
    base = make_pipe(BASE_ALPHA)
    base.fit(train.loc[:, list(FULL)], pd.to_numeric(train["ats_margin"]))
    p0 = fit_two_sided(train, "P0", {"home": BASE_ALPHA, "away": BASE_ALPHA})
    diff = float(np.max(np.abs(np.asarray(base.predict(test.loc[:, list(FULL)])) - p0.predict(test))))
    print("P0 identity max abs prediction diff vs base:", diff, flush=True)
    if diff > 1e-6:
        raise SystemExit("P0 IDENTITY FAILED")
    a1 = select_alphas(train, "P1")
    bad = train.copy()
    last = bad["season"].eq(int(bad["season"].max()))
    bad.loc[last, ["home_score", "away_score", "result", "ats_margin"]] = 999.0
    a2 = select_alphas(bad, "P1")
    ok = a1 == a2
    print("alpha selection identical when latest-season outcomes corrupted:", ok, a1, flush=True)
    if not ok:
        raise SystemExit("LEAK CHECK FAILED")
    older = train.loc[train["season"].lt(int(train["season"].max()) - 3)].copy()
    older2 = older.copy()
    older2["home_score"] = older2["home_score"] + 3.0 * pd.to_numeric(older2["spread_line"])
    print("positive control selection differs when prior outcomes change:", select_alphas(older, "P1") != select_alphas(older2, "P1"), flush=True)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    features = pd.read_parquet(u3.FEATURES)
    once = features.drop_duplicates("game_id").set_index("game_id")
    CTX["home_score"] = once["home_score"]
    CTX["away_score"] = once["away_score"]
    pop = HARNESS["proxy_population"](features)
    base = pd.read_parquet(BASE5)
    tr = base.loc[base["season"].ge(2020)]
    print("base true", int(tr["correct"].sum()), len(tr), flush=True)
    if (int(tr["correct"].sum()), len(tr)) != (802, 1503):
        raise SystemExit("base reproduction failed")
    sb = base.loc[base["season"].le(2019)]
    print("base proxy", int(sb["correct"].sum()), len(sb), flush=True)
    checks(features)
    install()
    CTX["cache_on"] = True
    os.environ[clv.CACHE_DISABLED_ENV] = "1"
    results = {}
    for name in ("P1", "P2"):
        path = OUT / f"{name}_run.parquet"
        if path.exists():
            run = pd.read_parquet(path)
        else:
            CTX["arm"] = name
            run = HARNESS["run_arm"](features, pop, lambda s: (FULL, BASE_ALPHA))
            run.to_parquet(path)
            if name == "P1":
                (OUT / "P1_alphas.json").write_text(json.dumps({str(k): v for k, v in sorted(CTX["alpha_log"].items())}, indent=1))
        run = run.loc[run.index.isin(base.index)]
        results[name] = compare(run, base)
        for era, r in results[name].items():
            print(name, era, r["record"], "base", r["record_base"], "d_acc", round(r["d_acc"], 3), r["d_acc_season_boot"], "flips", r["flips"], r["flips_W_L"], "ll", round(r["ll_arm"], 4), round(r["ll_base"], 4), round(r["ll_market"], 4), flush=True)
    (OUT / "report_all.json").write_text(json.dumps(results, indent=1, default=str))
    cells = []
    for name, res in results.items():
        for era, r in res.items():
            lo, hi = (2011, 2019) if era.startswith("sbr") else (2020, 2025)
            cells.append(
                {
                    "source": "artifacts/mod23_unit9/report_all.json",
                    "effect_units": "accuracy_points",
                    "classification": "unresolved_below_power",
                    "league": "nfl",
                    "season_start": lo,
                    "season_end": hi,
                    "family": "mod23_unit9_two_sided",
                    "category": "modeling",
                    "classification_evidence": "No positive control and no admissible closing ground; season-blocked bootstrap interval reported.",
                    "notes": f"Home and away points fitted as separate ridge targets (residual to market-implied team points), margin by difference, arm {name}; four looks (two arms, two eras).",
                    "name": f"mod23_unit9_two_sided_{name}_{era}_accuracy_diff",
                    "description": f"Two-sided points labels arm {name} paired vs base, {lo}-{hi}; {r['record']} vs {r['record_base']}.",
                    "effect": r["d_acc"],
                    "interval_low": r["d_acc_season_boot"]["low"],
                    "interval_high": r["d_acc_season_boot"]["high"],
                    "probability_positive": r["d_acc_season_boot"]["pp"],
                    "sample_games": r["n"],
                    "sample_blocks": r["d_acc_season_boot"]["blocks"],
                    "plain_summary": "Predicting each side's points separately and subtracting moved picks against the opener by an unresolved amount.",
                }
            )
    (OUT / "weak_signals_batch.json").write_text(json.dumps({"cells": cells}, indent=1))


main()
