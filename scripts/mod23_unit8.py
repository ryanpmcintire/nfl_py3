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
from nfl_ats.constants import DEFAULT_MIN_TRAIN_GAMES

HARNESS_PATH = Path(__file__).resolve().parent / "mod24_u1.py"
HARNESS = {"__file__": str(HARNESS_PATH), "__name__": "mod24_u1_harness"}
exec(compile(HARNESS_PATH.read_text().rsplit("\nmain()", 1)[0], str(HARNESS_PATH), "exec"), HARNESS)

ROOT = u3.ROOT
OUT = ROOT / "artifacts" / "mod23_unit8"
BASE5 = ROOT / "artifacts" / "mod24_u5" / "base_per_game.parquet"
PBP = ROOT / "data" / "pbp" / "team_style" / "raw_pbp_narrow.parquet"
FULL = HARNESS["FULL"]
BASE_ALPHA = HARNESS["BASE_ALPHA"]
SUPP = HARNESS["SUPP"]
BOOT = 20000
SEED = 20261018
LEAK_SEASON = 2022
LEAK_WEEK = 10
STRATA_PARTS = ("qb", "band", "ntd")
NON_OFFENSIVE_KICK = ("kickoff", "punt")
CTX = {"mode": "raw", "parts": STRATA_PARTS, "oof": None, "strata": None, "log": []}


def game_strata():
    p = pd.read_parquet(
        PBP,
        columns=[
            "game_id", "posteam", "qtr", "score_differential", "qb_dropback", "qb_kneel", "qb_spike",
            "passer_player_id", "touchdown", "interception", "fumble_lost", "play_type",
        ],
    )
    drop = p["qb_dropback"].eq(1) & p["qb_kneel"].ne(1) & p["qb_spike"].ne(1) & p["passer_player_id"].notna()
    passers = p.loc[drop].groupby(["game_id", "posteam"])["passer_player_id"].nunique()
    qb = passers.groupby("game_id").max().ge(2).astype(int).rename("qb")
    q4 = p.loc[p["qtr"].eq(4) & p["score_differential"].notna()].drop_duplicates("game_id")
    lead = q4.set_index("game_id")["score_differential"].abs()
    band = pd.cut(lead, [-0.5, 7.5, 16.5, 1e9], labels=[0, 1, 2]).astype("float").rename("band")
    nonoff = p["touchdown"].eq(1) & (
        p["interception"].eq(1) | p["fumble_lost"].eq(1) | p["play_type"].isin(NON_OFFENSIVE_KICK)
    )
    ntd = nonoff.groupby(p["game_id"]).sum().clip(upper=1).rename("ntd")
    return pd.concat([qb, band, ntd], axis=1)


def oof_residuals(features):
    reg = M.regular_season_rows(features)
    reg = reg.loc[pd.to_numeric(reg["ats_margin"], errors="coerce").notna()]
    parts = []
    for s in sorted(reg["season"].unique()):
        train = reg.loc[reg["season"].lt(s)]
        test = reg.loc[reg["season"].eq(s)]
        if len(train) < DEFAULT_MIN_TRAIN_GAMES:
            continue
        est = M.make_margin_estimator("ridge", 42, ridge_alpha=BASE_ALPHA, suppressed_indicator_columns=SUPP)
        est.fit(train.loc[:, list(FULL)], pd.to_numeric(train["ats_margin"]))
        pred = np.asarray(est.predict(test.loc[:, list(FULL)]), dtype=float)
        parts.append(
            pd.DataFrame(
                {
                    "game_id": test["game_id"].to_numpy(),
                    "season": int(s),
                    "res": pd.to_numeric(test["ats_margin"]).to_numpy(dtype=float) - pred,
                }
            )
        )
    return pd.concat(parts, ignore_index=True)


def stratum_key(strata, parts):
    s = strata[list(parts)]
    key = s.astype("Int64").astype(str).agg("|".join, axis=1)
    return key.where(s.notna().all(axis=1), "unknown")


def variance_table(oof, strata, parts, max_season):
    use = oof.loc[oof["season"].lt(max_season)]
    if use.empty:
        return None
    key = stratum_key(strata, parts).reindex(use["game_id"]).to_numpy()
    r = use["res"].to_numpy()
    overall = float(np.var(r, ddof=1))
    tab = {}
    for k in set(key):
        v = r[key == k]
        tab[k] = float(np.var(v, ddof=1)) if len(v) >= 2 else overall
    return tab, overall


def weights_for(training, oof, strata, parts):
    max_season = int(training["season"].max())
    vt = variance_table(oof, strata, parts, max_season)
    if vt is None:
        return None
    tab, overall = vt
    key = stratum_key(strata, parts).reindex(training["game_id"]).to_numpy()
    s2 = np.array([tab.get(k, overall) if k != "unknown" else overall for k in key])
    w = 1.0 / s2
    return w / w.mean()


def weighted_fit(frame, *, target="margin", model_name="ridge", distribution_fraction=0.20,
                 min_distribution_rows=10, random_state=42, feature_profile="base", ridge_alpha=10.0,
                 column_penalties=None):
    feature_columns = M.margin_feature_columns(target, feature_profile)
    supp = M._PROFILE_SUPPRESSED_MISSING_INDICATORS.get(feature_profile, ())
    training = M.regular_season_rows(frame)
    training = training.loc[M._target_values(training, target).notna()].copy()
    training["gameday"] = pd.to_datetime(training["gameday"], errors="raise")
    training = training.sort_values(["gameday", "game_id"]).reset_index(drop=True)
    if CTX["mode"] == "raw":
        w = None
    else:
        w = weights_for(training, CTX["oof"], CTX["strata"], CTX["parts"])
    CTX["last_w"] = w
    CTX["last_ids"] = training["game_id"].to_numpy()
    n = len(training)
    split = n - int(n * distribution_fraction)
    fit_part = training.iloc[:split]
    dist_part = training.iloc[split:]

    def fit(est, part, wt):
        x = part.loc[:, list(feature_columns)]
        y = M._target_values(part, target)
        if wt is None:
            est.fit(x, y)
        else:
            est.fit(x, y, regressor__sample_weight=wt)
        return est

    mk = lambda: M.make_margin_estimator(
        model_name, random_state, ridge_alpha=ridge_alpha, column_penalties=column_penalties,
        suppressed_indicator_columns=supp,
    )
    tmp = fit(mk(), fit_part, None if w is None else w[:split])
    cal = np.asarray(tmp.predict(dist_part.loc[:, list(feature_columns)]), dtype=float)
    residuals = M._target_values(dist_part, target).to_numpy(dtype=float) - cal
    residuals = residuals[np.isfinite(residuals)]
    est = fit(mk(), training, w)
    return M.MarginModel(
        estimator=est, residuals=residuals, model_name=model_name,
        ridge_alpha=ridge_alpha if model_name == "ridge" else None, target=target,
        feature_columns=feature_columns, training_rows=len(training), distribution_rows=len(residuals),
        training_max_gameday=training["gameday"].max().date().isoformat(),
        column_penalties=None if column_penalties is None else dict(column_penalties),
    )


def install():
    clv.fit_margin_model = weighted_fit
    HARNESS["fit_margin_model"] = weighted_fit


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


def leak_check(features, oof, strata):
    reg = M.regular_season_rows(features)
    cut = reg.loc[reg["season"].eq(LEAK_SEASON) & reg["week"].eq(LEAK_WEEK), "gameday"].min()
    cut = pd.Timestamp(cut)
    reg = reg.assign(gameday=pd.to_datetime(reg["gameday"]))
    train = reg.loc[reg["gameday"].lt(cut) & reg["result"].notna()]
    if int(train["season"].max()) >= LEAK_SEASON + 1:
        raise SystemExit("LEAK CHECK FAILED: train season")
    w1 = weights_for(train.sort_values(["gameday", "game_id"]).reset_index(drop=True), oof, strata, STRATA_PARTS)
    late = reg.loc[reg["gameday"].ge(cut), "game_id"]
    features2 = features.copy()
    hit = features2["game_id"].isin(set(late))
    features2.loc[hit, ["ats_margin", "result"]] = 999.0
    oof2 = oof_residuals(features2)
    strata2 = strata.copy()
    strata2.loc[strata2.index.isin(set(late)), :] = 0
    w2 = weights_for(train.sort_values(["gameday", "game_id"]).reset_index(drop=True), oof2, strata2, STRATA_PARTS)
    same = bool(np.array_equal(w1, w2))
    train_ok = float(train["season"].max()) <= LEAK_SEASON
    train2 = reg.loc[reg["gameday"].lt(cut) & reg["result"].notna()].sort_values(["gameday", "game_id"]).reset_index(drop=True)
    w3 = weights_for(train2, oof2, strata2, STRATA_PARTS)
    late_season = M.regular_season_rows(features).assign(
        gameday=lambda d: pd.to_datetime(d["gameday"])
    )
    later = late_season.loc[late_season["gameday"].ge(cut)].copy()
    later_train = pd.concat([train2, later.loc[later["season"].eq(LEAK_SEASON + 1)].head(300)]).sort_values(["gameday", "game_id"]).reset_index(drop=True)
    a = weights_for(later_train, oof, strata, STRATA_PARTS)
    b = weights_for(later_train, oof2, strata2, STRATA_PARTS)
    control = not np.array_equal(a, b)
    print("leak check weights identical when outcomes at/after cutoff corrupted:", same, "train max season", int(train["season"].max()), "positive control differs:", control, flush=True)
    if not same or not train_ok or w3 is None or not control:
        raise SystemExit("LEAK CHECK FAILED")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    features = pd.read_parquet(u3.FEATURES)
    strata = game_strata()
    oof = oof_residuals(features)
    pop = HARNESS["proxy_population"](features)
    base = pd.read_parquet(BASE5)
    tr = base.loc[base["season"].ge(2020)]
    print("base true", int(tr["correct"].sum()), len(tr), flush=True)
    if (int(tr["correct"].sum()), len(tr)) != (802, 1503):
        raise SystemExit("base reproduction failed")
    print("strata coverage", {c: int(strata[c].notna().sum()) for c in strata.columns}, "means", strata.mean().round(3).to_dict(), flush=True)
    CTX["oof"] = oof
    CTX["strata"] = strata
    leak_check(features, oof, strata)
    install()
    arms = {"W": STRATA_PARTS, "W_qb": ("qb",), "W_band": ("band",), "W_ntd": ("ntd",)}
    stage = sys.argv[1] if len(sys.argv) > 1 else "all"
    todo = ["repro", *arms] if stage == "all" else [stage]
    results = {}
    for name in todo:
        path = OUT / f"{name}_run.parquet"
        if path.exists():
            run = pd.read_parquet(path)
        else:
            CTX["mode"] = "raw" if name == "repro" else "weighted"
            if name == "repro":
                os.environ.pop(clv.CACHE_DISABLED_ENV, None)
            else:
                os.environ[clv.CACHE_DISABLED_ENV] = "1"
            CTX["parts"] = arms.get(name, STRATA_PARTS)
            run = HARNESS["run_arm"](features, pop, lambda s: (FULL, BASE_ALPHA))
            run.to_parquet(path)
        run = run.loc[run.index.isin(base.index)]
        results[name] = compare(run, base)
        for era, r in results[name].items():
            print(name, era, r["record"], "base", r["record_base"], "d_acc", round(r["d_acc"], 3), r["d_acc_season_boot"], "flips", r["flips"], r["flips_W_L"], flush=True)
    (OUT / f"report_{stage}.json").write_text(json.dumps(results, indent=1, default=str))
    if "W" in results:
        cells = []
        for era, r in results["W"].items():
            lo, hi = (2011, 2019) if era.startswith("sbr") else (2020, 2025)
            cells.append(
                {
                    "source": "artifacts/mod23_unit8/report_all.json",
                    "effect_units": "accuracy_points",
                    "classification": "unresolved_below_power",
                    "league": "nfl",
                    "season_start": lo,
                    "season_end": hi,
                    "family": "mod23_unit8_label_weights",
                    "category": "modeling",
                    "classification_evidence": "No positive control and no admissible closing ground; season-blocked bootstrap interval reported.",
                    "notes": "Ridge alpha 10 trained with weights of one over the out-of-fold residual variance of each completed-game stratum (QB change, fourth-quarter margin band, non-offensive touchdown), variances from prior seasons only; two looks (two eras), single-stratum ablations unrecorded.",
                    "name": f"mod23_unit8_label_weights_{era}_accuracy_diff",
                    "description": f"Base model trained with game weights from measured label noise, paired vs base, {lo}-{hi}; {r['record']} vs {r['record_base']}.",
                    "effect": r["d_acc"],
                    "interval_low": r["d_acc_season_boot"]["low"],
                    "interval_high": r["d_acc_season_boot"]["high"],
                    "probability_positive": r["d_acc_season_boot"]["pp"],
                    "sample_games": r["n"],
                    "sample_blocks": r["d_acc_season_boot"]["blocks"],
                    "plain_summary": "Giving noisy-result games less weight in training moved picks against the opener by an unresolved amount.",
                }
            )
        (OUT / "weak_signals_batch_w.json").write_text(json.dumps({"cells": cells}, indent=1))


main()
