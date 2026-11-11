from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNetCV
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))

import mod57_p2n_margin_error_vs_line as p2n

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts/mod57_p2p"
HINDSIGHT = ROOT / "artifacts/mod57_p2o/per_game_ratings_LEAK_NOT_SERVABLE.parquet"
P2N_PREDICTIONS = ROOT / "artifacts/mod57_p2n/oos_predictions.parquet"
FAMILY = "mod57_how_wrong_is_the_line"
SERVED_OPEN_RECORD = (802, 701)
ARMS = ("EL", "GL")
REFERENCE_ARMS = ("R", "S", "G", "H", "E")
FIRST_TEST = p2n.FIRST_TEST
LAST_SEASON = p2n.LAST_SEASON
INNER_VALIDATION_SEASONS = p2n.INNER_VALIDATION_SEASONS
ENET_SPLITS = p2n.ENET_SPLITS
BOOTSTRAP_DRAWS = 4000
SEED = 20261113
PERMUTATION_REPEATS = 5
THIRD_EDGES = (0, 6, 12, 99)
THIRD_LABELS = ("weeks_1_6", "weeks_7_12", "weeks_13_18")
GAP_EDGES = (-1.0, 1.0, 3.0, 1e9)
GAP_LABELS = ("under_1", "1_to_3", "over_3")
TRUNCATION_SEASONS = (2012, 2016, 2019, 2022, 2024)
TRUNCATION_WEEKS = (6, 12)
TRUNCATION_MIN_GAMES = 2
TRUNCATION_TOLERANCE = 1e-9
BASE_TABLE = "game_features"
NOTES = (
    "Two models trained on the hindsight gap (own-game-removed season rating margin minus "
    "close line) from earlier completed seasons only, tested walk-forward 2015-25. "
    "Season-block bootstrap."
)
EVIDENCE = (
    "Walk-forward by season, probability_positive reported; no refuted mechanism "
    "and no closing ground."
)
SUMMARY = "Tests how much of the line's hindsight error can be forecast before kickoff."


def table_of_columns():
    seen = set()
    owner = {}
    for name in p2n.TABLES:
        frame = pd.read_parquet(p2n.PROCESSED / f"{name}.parquet")
        for c in frame.columns:
            if c in p2n.NON_FEATURE or c == "week" or c.endswith("_version"):
                continue
            if c not in seen:
                seen.add(c)
                owner[c] = name
    owner["spread_line"] = "line_itself"
    return owner


def load_data():
    data, counts, served, all_cols = p2n.load()
    h = pd.read_parquet(HINDSIGHT)[["game_id", "margin", "close_line", "R_H"]]
    data = data.merge(h, on="game_id", how="left", validate="one_to_one")
    assert data.R_H.notna().all()
    assert np.allclose(data.margin, data.result)
    assert np.allclose(data.close_line, data.spread_line)
    data["label"] = data.R_H - data.spread_line
    return data, counts, served, all_cols


def enet_model():
    return make_pipeline(
        SimpleImputer(strategy="median", add_indicator=True),
        StandardScaler(),
        ElasticNetCV(cv=TimeSeriesSplit(n_splits=ENET_SPLITS)),
    )


def gl_fit(xtr, ytr, n_inner):
    inner_cols = [c for c in xtr.columns if xtr.iloc[:n_inner][c].nunique() > 1]
    probe = HistGradientBoostingRegressor(early_stopping=False, max_iter=p2n.HGB_DEFAULT_MAX_ITER)
    probe.fit(xtr.iloc[:n_inner][inner_cols], ytr[:n_inner])
    errs = [
        float(np.mean((ytr[n_inner:] - p) ** 2))
        for p in probe.staged_predict(xtr.iloc[n_inner:][inner_cols])
    ]
    best = int(np.argmin(errs)) + 1
    final = HistGradientBoostingRegressor(early_stopping=False, max_iter=best)
    final.fit(xtr, ytr)
    return final, best


def walk_forward(data, all_cols, owner):
    preds = {a: [] for a in ARMS}
    scales = {}
    meta = []
    families = {"EL": [], "GL": []}
    rng = np.random.default_rng(SEED)
    for season in range(FIRST_TEST, LAST_SEASON + 1):
        train = data[data.season < season]
        test = data[data.season == season]
        assert train.season.max() < season
        n_inner = int((train.season < season - INNER_VALIDATION_SEASONS).sum())
        ytr = train.label.to_numpy()
        scale = float(np.cov(train.resid, train.label)[0, 1] / np.var(train.label, ddof=1))
        scales[season] = scale
        cols = [c for c in all_cols if train[c].nunique() > 1]
        observed = [c for c in all_cols if train[c].notna().any()]
        enet = enet_model().fit(train[observed], ytr)
        p_e = enet.predict(test[observed])
        gl, iters = gl_fit(train[cols], ytr, n_inner)
        p_g = gl.predict(test[cols])
        base = float(np.mean((test.label.to_numpy() - p_g) ** 2))
        fam = {}
        for table in sorted({owner[c] for c in cols}):
            members = [c for c in cols if owner[c] == table]
            rises = []
            for _ in range(PERMUTATION_REPEATS):
                shuffled = test[cols].copy()
                order = rng.permutation(len(shuffled))
                shuffled[members] = shuffled[members].to_numpy()[order]
                rises.append(
                    float(np.mean((test.label.to_numpy() - gl.predict(shuffled)) ** 2)) - base
                )
            fam[table] = float(np.mean(rises))
        families["GL"].append({"season": season, "mse_rise_by_table": fam})
        coef = enet[-1].coef_
        imputer = enet[0]
        names = list(observed) + [f"missing::{observed[i]}" for i in imputer.indicator_.features_]
        weight = {}
        nonzero = {}
        for name, value in zip(names, coef, strict=True):
            if value != 0:
                key = owner[name.removeprefix("missing::")]
                weight[key] = weight.get(key, 0.0) + abs(float(value))
                nonzero[key] = nonzero.get(key, 0) + 1
        families["EL"].append(
            {"season": season, "abs_coef_by_table": weight, "nonzero_by_table": nonzero}
        )
        for arm, p in zip(ARMS, (p_e, p_g), strict=True):
            preds[arm].append(pd.Series(p, index=test.index))
        meta.append(
            {
                "test_season": season,
                "max_label_season_in_training": int(train.season.max()),
                "train_games": len(train),
                "test_games": len(test),
                "label_to_result_slope_train": scale,
                "gl_iterations": iters,
                "enet_alpha": float(enet[-1].alpha_),
                "enet_l1_ratio": float(enet[-1].l1_ratio_),
                "enet_nonzero": int((coef != 0).sum()),
            }
        )
        print(meta[-1], flush=True)
    return {a: pd.concat(v).sort_index() for a, v in preds.items()}, scales, meta, families


def label_sums(frame, pred, label):
    rows = []
    for _, g in frame.groupby("season"):
        y = label.loc[g.index].to_numpy()
        p = pred.loc[g.index].to_numpy()
        rows.append(
            [
                len(g),
                ((y - p) ** 2).sum(),
                (y**2).sum(),
                p.sum(),
                y.sum(),
                (p * p).sum(),
                (p * y).sum(),
            ]
        )
    return np.array(rows)


def label_stats(s):
    n, sse, ssl, sp, sy, spp, spy = s.sum(axis=0)
    vp = spp - sp * sp / n
    vy = ssl - sy * sy / n
    cov = spy - sp * sy / n
    return {
        "r2_vs_zero": 1 - sse / ssl,
        "corr": cov / np.sqrt(vp * vy) if vp > 0 and vy > 0 else np.nan,
        "slope_label_on_pred": cov / vp if vp > 0 else np.nan,
    }


def label_block(frame, pred, rng):
    s = label_sums(frame, pred, frame.label)
    point = label_stats(s)
    draws = rng.integers(0, len(s), size=(BOOTSTRAP_DRAWS, len(s)))
    boots = [label_stats(s[d]) for d in draws]
    out = {}
    for k, v in point.items():
        b = np.array([x[k] for x in boots])
        out[k] = {
            "value": float(v),
            "interval": p2n.interval(b),
            "probability_positive": float((b[np.isfinite(b)] > 0).mean()),
        }
    return out


def tally(pick, result, line):
    w, lo = p2n.record_count(pick, result, line)
    return w, lo


def accuracy_block(frame, pick, line, season_col, rng):
    result = frame.result.to_numpy()
    win = ((pick > 0) & (result > line)) | ((pick < 0) & (result < line))
    decisive = (result != line) & (pick != 0)
    seasons = sorted(frame[season_col].unique())
    per = np.array(
        [
            [
                int((win & decisive & (frame[season_col].to_numpy() == s)).sum()),
                int((~win & decisive & (frame[season_col].to_numpy() == s)).sum()),
            ]
            for s in seasons
        ]
    )
    if len(per) == 0:
        return None
    total = per.sum(axis=0)
    draws = rng.integers(0, len(per), size=(BOOTSTRAP_DRAWS, len(per)))
    accs = np.array([per[d, 0].sum() / max(per[d].sum(), 1) for d in draws])
    return {
        "record": [int(total[0]), int(total[1])],
        "accuracy": float(total[0] / total.sum()),
        "interval": p2n.interval(accs),
        "probability_above_half": float((accs > 0.5).mean()),
    }


def splits(frame, pick, line, gap, rng):
    out = {}
    thirds = pd.cut(frame.week, bins=list(THIRD_EDGES), labels=THIRD_LABELS).to_numpy()
    for label in THIRD_LABELS:
        m = thirds == label
        out[label] = accuracy_block(frame[m], pick[m], line[m], "season", rng)
    bins = pd.cut(np.abs(gap), bins=list(GAP_EDGES), labels=GAP_LABELS, right=False).to_numpy()
    for label in GAP_LABELS:
        m = bins == label
        out[f"gap_{label}"] = accuracy_block(frame[m], pick[m], line[m], "season", rng)
    return out


def result_block(frame, resid_pred, gap, rng, open_frame):
    resid = frame.resid
    s = p2n.season_sums(frame, resid_pred, resid)
    b = p2n.boot(s, rng)
    n = s[:, 0].sum()
    e = resid.loc[frame.index]
    p = resid_pred.loc[frame.index]
    line = frame.spread_line.to_numpy()
    pick = np.sign(p.to_numpy())
    out = {
        "mae": float(np.mean(np.abs(e - p))),
        "rmse": float(np.sqrt(np.mean((e - p) ** 2))),
        "line_mae": float(np.mean(np.abs(e))),
        "line_rmse": float(np.sqrt(np.mean(e**2))),
        "abs_diff_vs_line": float(s[:, 1].sum() / n),
        "abs_diff_interval": p2n.interval(b["abs"]),
        "abs_diff_probability_positive": float((b["abs"] > 0).mean()),
        "sq_diff_vs_line": float(s[:, 2].sum() / n),
        "sq_diff_interval": p2n.interval(b["sq"]),
        "sq_diff_probability_positive": float((b["sq"] > 0).mean()),
        "slope": float(p2n.slope_from(s)),
        "slope_interval": p2n.interval(b["slope"]),
        "slope_probability_positive": float((b["slope"] > 0).mean()),
        "mean_abs_pred": float(np.mean(np.abs(p))),
        "close_accuracy": accuracy_block(frame, pick, line, "season", rng),
        "close_splits": splits(frame, pick, line, gap.loc[frame.index].to_numpy(), rng),
    }
    po = resid_pred.loc[open_frame.index]
    open_line = open_frame.open_line.to_numpy()
    pick_open = np.sign(po.to_numpy())
    out["open_accuracy"] = accuracy_block(open_frame, pick_open, open_line, "season", rng)
    out["open_splits"] = splits(
        open_frame, pick_open, open_line, gap.loc[open_frame.index].to_numpy(), rng
    )
    return out


def evaluate(data, preds, scales, ref):
    rng = np.random.default_rng(SEED)
    test = data[data.season >= FIRST_TEST].copy()
    opener = ref[["game_id", "tue_open_home_spread"]].dropna()
    open_frame = test.merge(opener, on="game_id", how="inner")
    open_frame = open_frame.assign(open_line=open_frame.tue_open_home_spread)
    open_frame.index = (
        test.reset_index().merge(opener[["game_id"]], on="game_id")["index"].to_numpy()
    )
    out = {
        "line": {
            "games": len(test),
            "open_games": len(open_frame),
            "served_open_record": list(SERVED_OPEN_RECORD),
        },
        "label": {},
        "result": {},
        "reference": {},
    }
    scale = pd.Series(test.season.map(scales).to_numpy(), index=test.index)
    for arm in ARMS:
        p = preds[arm].loc[test.index]
        out["label"][arm] = {
            "overall": label_block(test, p, rng),
            **{
                label: label_block(test[(test.week > lo) & (test.week <= hi)], p, rng)
                for label, lo, hi in zip(
                    THIRD_LABELS, THIRD_EDGES[:-1], THIRD_EDGES[1:], strict=True
                )
            },
        }
        out["result"][arm] = result_block(test, scale * p, p, rng, open_frame)
    refp = ref.set_index("game_id")
    for arm in REFERENCE_ARMS:
        p = pd.Series(refp.loc[test.game_id, f"pred_{arm}"].to_numpy(), index=test.index)
        out["reference"][arm] = result_block(test, p, p, rng, open_frame)
    return out


def truncation_check():
    from nfl_ats.pbp import enrich_with_pbp_features, latest_pbp_snapshot, load_pbp_snapshot

    manifest = json.loads(
        (p2n.PROCESSED / "game_features_pbp.manifest.json").read_text(encoding="utf-8")
    )
    games = pd.read_parquet(p2n.PROCESSED / f"{BASE_TABLE}.parquet")
    games["gameday"] = pd.to_datetime(games["gameday"])
    stored = pd.read_parquet(p2n.PROCESSED / "game_features_pbp.parquet").set_index("game_id")
    pbp = load_pbp_snapshot(latest_pbp_snapshot(ROOT / "data/pbp/raw"))
    day = games.set_index("game_id")["gameday"]
    pbp_day = pbp["game_id"].map(day)
    added = [c for c in stored.columns if c not in games.columns]
    rng = np.random.default_rng(SEED)
    reg = games[(games.game_type == "REG")]
    report = []
    for season in TRUNCATION_SEASONS:
        s = reg[(reg.season == season) & reg.week.between(*TRUNCATION_WEEKS)]
        counts = s.groupby("gameday").size()
        dates = counts[counts >= TRUNCATION_MIN_GAMES].index
        cutoff = pd.Timestamp(rng.choice(dates))
        trunc = pbp[pbp_day < cutoff]
        rebuilt = enrich_with_pbp_features(
            games,
            trunc,
            span=manifest["ewm_span"],
            min_periods=manifest["min_periods"],
            offseason_retention=manifest["offseason_retention"],
            opponent_half_life_weeks=manifest["opponent_half_life_weeks"],
            opponent_ridge_alpha=manifest["opponent_ridge_alpha"],
            opponent_min_team_games=manifest["opponent_min_team_games"],
        ).set_index("game_id")
        ids = games.loc[games.gameday == cutoff, "game_id"]
        a = rebuilt.loc[ids, added].apply(pd.to_numeric, errors="coerce").to_numpy(float)
        b = stored.loc[ids, added].apply(pd.to_numeric, errors="coerce").to_numpy(float)
        same_nan = np.isnan(a) == np.isnan(b)
        diff = np.where(np.isnan(a) | np.isnan(b), 0.0, np.abs(a - b))
        bad_cols = [added[j] for j in np.where((diff > TRUNCATION_TOLERANCE).any(axis=0))[0]]
        report.append(
            {
                "cutoff": str(cutoff.date()),
                "games_compared": len(ids),
                "pbp_rows_kept": len(trunc),
                "columns_compared": len(added),
                "max_abs_diff": float(diff.max()),
                "nan_pattern_mismatches": int((~same_nan).sum()),
                "columns_that_differ": bad_cols,
            }
        )
        print(report[-1], flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "truncation_pbp.json").write_text(json.dumps(report, indent=1), encoding="utf-8")


def cells(results):
    out = []
    window = (FIRST_TEST, LAST_SEASON)
    for arm in ARMS:
        lab = results["label"][arm]["overall"]["r2_vs_zero"]
        res = results["result"][arm]
        specs = (
            (
                "label_variance_share",
                lab["value"],
                lab["interval"],
                lab["probability_positive"],
                "variance_share",
            ),
            (
                "mae_diff_line_minus_model",
                res["abs_diff_vs_line"],
                res["abs_diff_interval"],
                res["abs_diff_probability_positive"],
                "margin_points",
            ),
            (
                "mse_diff_line_minus_model",
                res["sq_diff_vs_line"],
                res["sq_diff_interval"],
                res["sq_diff_probability_positive"],
                "squared_margin_points",
            ),
            (
                "result_calibration_slope",
                res["slope"],
                res["slope_interval"],
                res["slope_probability_positive"],
                "calibration_slope",
            ),
        )
        for name, effect, interval, pp, units in specs:
            out.append(
                {
                    "name": f"mod57_p2p_{arm}_{name}",
                    "description": f"Model {arm}, hindsight-gap label, walk-forward: {name}",
                    "effect": float(effect),
                    "effect_units": units,
                    "interval_low": interval[0],
                    "interval_high": interval[1],
                    "probability_positive": pp,
                    "season_start": window[0],
                    "season_end": window[1],
                }
            )
        for line, start in (("close", window[0]), ("open", 2020)):
            acc = res[f"{line}_accuracy"]
            draws_pp = acc["probability_above_half"]
            out.append(
                {
                    "name": f"mod57_p2p_{arm}_{line}_pick_accuracy",
                    "description": f"Model {arm} pick accuracy vs {line} line, decisive games",
                    "effect": 100 * (acc["accuracy"] - 0.5),
                    "effect_units": "accuracy_points",
                    "interval_low": 100 * (acc["interval"][0] - 0.5),
                    "interval_high": 100 * (acc["interval"][1] - 0.5),
                    "probability_positive": draws_pp,
                    "season_start": start,
                    "season_end": window[1],
                }
            )
    return out


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "truncate":
        truncation_check()
        return
    OUT.mkdir(parents=True, exist_ok=True)
    data, counts, _served, all_cols = load_data()
    owner = table_of_columns()
    print({"games": len(data), "columns": len(all_cols), "per_table": counts}, flush=True)
    preds, scales, meta, families = walk_forward(data, all_cols, owner)
    test = data[data.season >= FIRST_TEST]
    frame = test[["game_id", "season", "week", "result", "spread_line", "resid", "label"]].copy()
    for arm in ARMS:
        frame[f"pred_label_{arm}"] = preds[arm].loc[frame.index]
        frame[f"pred_resid_{arm}"] = frame[f"pred_label_{arm}"] * frame.season.map(scales)
    frame.to_parquet(OUT / "oos_predictions.parquet", index=False)
    ref = pd.read_parquet(P2N_PREDICTIONS)
    results = evaluate(data, preds, scales, ref)
    (OUT / "results.json").write_text(
        json.dumps({"folds": meta, "families": families, "results": results}, indent=1),
        encoding="utf-8",
    )
    batch = {
        "league": "nfl",
        "classification": "unresolved_below_power",
        "source": "artifacts/mod57_p2p/results.json",
        "category": "market",
        "notes": NOTES,
        "classification_evidence": EVIDENCE,
        "plain_summary": SUMMARY,
        "family": FAMILY,
        "cells": cells(results),
    }
    (OUT / "batch.json").write_text(json.dumps(batch, indent=1), encoding="utf-8")
    print(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
