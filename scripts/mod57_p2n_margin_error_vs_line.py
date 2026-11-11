from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNetCV, Ridge
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from nfl_ats.constants import FEATURE_SETS

ROOT = Path("F:/Repos/nfl_py3")
PROCESSED = ROOT / "data/processed"
PER_GAME = ROOT / "artifacts/pick_probability/20261111T011859Z/per_game.parquet"
OUT = ROOT / "artifacts/mod57_p2n"
TABLES = (
    "game_features",
    "game_features_pbp",
    "game_features_player",
    "game_features_player_value",
    "game_features_player_participation",
    "game_features_player_learned_availability",
    "game_features_qb",
    "game_features_weak_stack",
)
NON_FEATURE = {
    "game_id",
    "season",
    "gameday",
    "weekday",
    "gametime",
    "kickoff",
    "game_type",
    "location",
    "result",
    "ats_margin",
    "home_cover",
    "home_score",
    "away_score",
    "spread_line",
}
FIRST_SEASON = 2009
LAST_SEASON = 2025
FIRST_TEST = 2015
INNER_VALIDATION_SEASONS = 2
SHRINK_GRID = np.round(np.arange(0.0, 1.0001, 0.1), 1)
RIDGE_ALPHA = 10.0
HGB_DEFAULT_MAX_ITER = 100
ENET_SPLITS = 5
BOOTSTRAP_DRAWS = 2000
SEED = 20261111
ARMS = ("R", "S", "G", "H", "E")


def load() -> tuple[pd.DataFrame, dict[str, int], list[str], list[str]]:
    merged = None
    counts = {}
    seen = set()
    order = []
    for name in TABLES:
        frame = pd.read_parquet(PROCESSED / f"{name}.parquet")
        cols = [
            c
            for c in frame.columns
            if c not in NON_FEATURE and c != "week" and not c.endswith("_version")
        ]
        counts[name] = len(cols)
        new = [c for c in cols if c not in seen]
        seen.update(new)
        order.extend(new)
        keep = frame[["game_id", *new]]
        if merged is None:
            base = frame[
                [
                    "game_id",
                    "season",
                    "week",
                    "gameday",
                    "game_type",
                    "result",
                    "ats_margin",
                    "spread_line",
                ]
            ]
            merged = base.merge(keep, on="game_id", how="left")
        else:
            merged = merged.merge(keep, on="game_id", how="left")
    merged = merged[
        (merged.game_type == "REG")
        & merged.season.between(FIRST_SEASON, LAST_SEASON)
        & merged.result.notna()
        & merged.spread_line.notna()
    ].copy()
    merged = merged.sort_values(["gameday", "game_id"]).reset_index(drop=True)
    merged["resid"] = merged.result - merged.spread_line
    assert np.allclose(merged.resid, merged.ats_margin)
    served = [c for c in FEATURE_SETS["football_weak_stack"] if c in merged.columns]
    assert len(served) == len(FEATURE_SETS["football_weak_stack"])
    all_cols = ["spread_line", *[c for c in order if c in merged.columns]]
    merged[all_cols] = merged[all_cols].apply(pd.to_numeric, errors="coerce")
    all_cols = [c for c in all_cols if merged[c].notna().any()]
    return merged, counts, served, all_cols


def ridge_model():
    return make_pipeline(
        SimpleImputer(strategy="median", add_indicator=True),
        StandardScaler(),
        Ridge(alpha=RIDGE_ALPHA),
    )


def hgb_fit_predict(xtr, ytr, xte, n_inner):
    inner_train = slice(0, n_inner)
    inner_val = slice(n_inner, len(xtr))
    probe = HistGradientBoostingRegressor(early_stopping=False, max_iter=HGB_DEFAULT_MAX_ITER)
    inner_cols = [c for c in xtr.columns if xtr.iloc[inner_train][c].nunique() > 1]
    probe.fit(xtr.iloc[inner_train][inner_cols], ytr[inner_train])
    errs = [
        float(np.mean((ytr[inner_val] - p) ** 2))
        for p in probe.staged_predict(xtr.iloc[inner_val][inner_cols])
    ]
    best = int(np.argmin(errs)) + 1
    final = HistGradientBoostingRegressor(early_stopping=False, max_iter=best)
    final.fit(xtr, ytr)
    return final.predict(xte), best


def walk_forward(data, served, all_cols):
    preds = {a: [] for a in ARMS}
    meta = []
    for season in range(FIRST_TEST, LAST_SEASON + 1):
        train = data[data.season < season]
        test = data[data.season == season]
        n_inner = int((train.season < season - INNER_VALIDATION_SEASONS).sum())
        ytr = train.resid.to_numpy()
        ridge = ridge_model().fit(train[served], ytr)
        p_r = ridge.predict(test[served])
        inner = ridge_model().fit(train[served].iloc[:n_inner], ytr[:n_inner])
        p_val = inner.predict(train[served].iloc[n_inner:])
        y_val = ytr[n_inner:]
        grid_err = [float(np.mean(np.abs(y_val - f * p_val))) for f in SHRINK_GRID]
        factor = float(SHRINK_GRID[int(np.argmin(grid_err))])
        cols_g = [c for c in all_cols if train[c].nunique() > 1]
        cols_h = [c for c in served if train[c].nunique() > 1]
        p_g, iters_g = hgb_fit_predict(train[cols_g], ytr, test[cols_g], n_inner)
        p_h, iters_h = hgb_fit_predict(train[cols_h], ytr, test[cols_h], n_inner)
        enet = make_pipeline(
            SimpleImputer(strategy="median", add_indicator=True),
            StandardScaler(),
            ElasticNetCV(cv=TimeSeriesSplit(n_splits=ENET_SPLITS)),
        ).fit(train[all_cols], ytr)
        p_e = enet.predict(test[all_cols])
        for arm, p in zip(ARMS, (p_r, factor * p_r, p_g, p_h, p_e), strict=True):
            preds[arm].append(pd.Series(p, index=test.index))
        meta.append(
            {
                "season": season,
                "train_games": len(train),
                "test_games": len(test),
                "shrink_factor": factor,
                "hgb_all_iterations": iters_g,
                "hgb_served_iterations": iters_h,
                "enet_alpha": float(enet[-1].alpha_),
                "enet_l1_ratio": float(enet[-1].l1_ratio_),
                "enet_nonzero": int((enet[-1].coef_ != 0).sum()),
            }
        )
        print(meta[-1], flush=True)
    return {a: pd.concat(v).sort_index() for a, v in preds.items()}, meta


def season_sums(frame, pred, benchmark_resid):
    rows = []
    for _season, g in frame.groupby("season"):
        e = benchmark_resid.loc[g.index].to_numpy()
        p = pred.loc[g.index].to_numpy()
        rows.append(
            [
                len(g),
                np.sum(np.abs(e) - np.abs(e - p)),
                np.sum(e**2 - (e - p) ** 2),
                p.sum(),
                e.sum(),
                (p * p).sum(),
                (p * e).sum(),
                (e * e).sum(),
            ]
        )
    return np.array(rows)


def slope_from(s):
    n, _, _, sp, se, spp, spe, _ = s.sum(axis=0)
    var = spp - sp * sp / n
    return (spe - sp * se / n) / var if var > 0 else np.nan


def corr_from(s):
    n, _, _, sp, se, spp, spe, see = s.sum(axis=0)
    vp = spp - sp * sp / n
    ve = see - se * se / n
    return (spe - sp * se / n) / np.sqrt(vp * ve) if vp > 0 and ve > 0 else np.nan


def intercept_from(s):
    n, _, _, sp, se, *_ = s.sum(axis=0)
    return (se - slope_from(s) * sp) / n


def boot(s, rng):
    blocks = len(s)
    draws = rng.integers(0, blocks, size=(BOOTSTRAP_DRAWS, blocks))
    n_all = s[:, 0]
    out = {"abs": [], "sq": [], "slope": []}
    for d in draws:
        sub = s[d]
        n = n_all[d].sum()
        out["abs"].append(sub[:, 1].sum() / n)
        out["sq"].append(sub[:, 2].sum() / n)
        out["slope"].append(slope_from(sub))
    return {k: np.array(v) for k, v in out.items()}


def interval(x):
    x = x[np.isfinite(x)]
    return [float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))]


def record_count(pick_home, result, line):
    decisive = (result != line) & (pick_home != 0)
    win = ((np.sign(pick_home) > 0) & (result > line)) | (
        (np.sign(pick_home) < 0) & (result < line)
    )
    return int((win & decisive).sum()), int((~win & decisive).sum())


def evaluate(data, preds, per_game):
    rng = np.random.default_rng(SEED)
    resid = data.resid
    out = {}
    base_mae = float(np.mean(np.abs(resid)))
    base_rmse = float(np.sqrt(np.mean(resid**2)))
    test = data[data.season >= FIRST_TEST]
    out["line"] = {
        "games": len(test),
        "mae": float(np.mean(np.abs(resid.loc[test.index]))),
        "rmse": float(np.sqrt(np.mean(resid.loc[test.index] ** 2))),
        "full_history_mae": base_mae,
        "full_history_rmse": base_rmse,
    }
    opened = test.merge(
        per_game[["game_id", "tue_open_home_spread", "model_correct"]], on="game_id", how="inner"
    )
    opened_idx = test.reset_index().merge(opened[["game_id"]], on="game_id")["index"].to_numpy()
    open_line = pd.Series(opened.tue_open_home_spread.to_numpy(), index=opened_idx, name="open")
    open_resid = data.result.loc[open_line.index] - open_line
    out["open_reference"] = {
        "games": len(open_line),
        "open_mae": float(np.mean(np.abs(open_resid))),
        "open_rmse": float(np.sqrt(np.mean(open_resid**2))),
        "close_mae_same_games": float(np.mean(np.abs(resid.loc[open_line.index]))),
        "served_record": [
            int(opened.model_correct.sum()),
            int(len(opened) - opened.model_correct.sum()),
        ],
    }
    for arm in ARMS:
        p = preds[arm].loc[test.index]
        s = season_sums(test, p, resid)
        b = boot(s, rng)
        n = s[:, 0].sum()
        e = resid.loc[test.index]
        mae = float(np.mean(np.abs(e - p)))
        rmse = float(np.sqrt(np.mean((e - p) ** 2)))
        w, lo_ = record_count(p.to_numpy(), test.result.to_numpy(), test.spread_line.to_numpy())
        po = p.loc[open_line.index]
        wo, lo = record_count(
            po.to_numpy(), data.result.loc[open_line.index].to_numpy(), open_line.to_numpy()
        )
        eo = data.result.loc[open_line.index] - open_line - po
        d_abs_o = np.abs(open_resid) - np.abs(eo)
        d_sq_o = open_resid**2 - eo**2
        out[arm] = {
            "mae": mae,
            "rmse": rmse,
            "abs_diff_vs_line": float(s[:, 1].sum() / n),
            "abs_diff_interval": interval(b["abs"]),
            "abs_diff_probability_positive": float((b["abs"] > 0).mean()),
            "sq_diff_vs_line": float(s[:, 2].sum() / n),
            "sq_diff_interval": interval(b["sq"]),
            "sq_diff_probability_positive": float((b["sq"] > 0).mean()),
            "slope": float(slope_from(s)),
            "slope_interval": interval(b["slope"]),
            "slope_probability_positive": float((b["slope"] > 0).mean()),
            "slope_probability_ge_1": float((b["slope"] >= 1).mean()),
            "intercept": float(intercept_from(s)),
            "corr": float(corr_from(s)),
            "r2_oos": float(1 - ((e - p) ** 2).sum() / (e**2).sum()),
            "mean_abs_pred": float(np.mean(np.abs(p))),
            "sd_pred": float(p.std()),
            "close_record_decisive": [w, lo_],
            "close_accuracy": w / (w + lo_) if w + lo_ else None,
            "open_games": len(open_line),
            "open_mae": float(np.mean(np.abs(eo))),
            "open_rmse": float(np.sqrt(np.mean(eo**2))),
            "open_abs_diff_vs_open_line": float(d_abs_o.mean()),
            "open_sq_diff_vs_open_line": float(d_sq_o.mean()),
            "open_record_decisive": [wo, lo],
            "open_accuracy": wo / (wo + lo) if wo + lo else None,
            "by_season_abs_diff": {
                int(k): float(v)
                for k, v in zip(sorted(test.season.unique()), s[:, 1] / s[:, 0], strict=True)
            },
        }
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    data, counts, served, all_cols = load()
    per_game = pd.read_parquet(PER_GAME)
    info = {
        "games": len(data),
        "seasons": [FIRST_SEASON, LAST_SEASON],
        "table_feature_columns": counts,
        "served_inputs": len(served),
        "all_feature_columns_union": len(all_cols),
        "pushes_kept": int((data.resid == 0).sum()),
    }
    corrs = data[all_cols].corrwith(data.resid).abs().sort_values(ascending=False)
    info["leak_screen_top5_abs_corr_with_residual"] = {
        k: float(v) for k, v in corrs.head(5).items()
    }
    print(info, flush=True)
    preds, meta = walk_forward(data, served, all_cols)
    frame = data[["game_id", "season", "week", "result", "spread_line", "resid"]].copy()
    for arm in ARMS:
        frame[f"pred_{arm}"] = preds[arm]
    frame = frame[frame.season >= FIRST_TEST]
    frame = frame.merge(per_game[["game_id", "tue_open_home_spread"]], on="game_id", how="left")
    frame.to_parquet(OUT / "oos_predictions.parquet", index=False)
    results = evaluate(data, preds, per_game)
    (OUT / "results.json").write_text(
        json.dumps({"info": info, "folds": meta, "results": results}, indent=1), encoding="utf-8"
    )
    print(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
