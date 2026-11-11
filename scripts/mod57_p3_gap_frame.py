from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNetCV
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))

import mod57_p2n_margin_error_vs_line as p2n
import mod57_p2p_forecast_the_hindsight_gap as p2p

SEED = p2p.SEED
FIRST_TEST = p2p.FIRST_TEST
LAST_SEASON = p2p.LAST_SEASON
BOOTSTRAP_DRAWS = p2p.BOOTSTRAP_DRAWS
ENET_SPLITS = p2p.ENET_SPLITS
WINDOWS = {"weeks_1_6": (0, 6), "weeks_7_12": (6, 12)}
LABEL_COLUMN = {"gap": "label", "margin": "resid"}
DEFAULT_FAMILY = "mod57_team_change_gap_frame"
OPENER_FIRST_SEASON = 2020


def column_model():
    return make_pipeline(
        SimpleImputer(strategy="median", add_indicator=True),
        StandardScaler(),
        ElasticNetCV(cv=TimeSeriesSplit(n_splits=ENET_SPLITS)),
    )


def in_window(df, window):
    lo, hi = WINDOWS[window]
    return df.week.gt(lo) & df.week.le(hi)


def walk(slim, cols, label_name, window, intercept_only=False, with_coef=False):
    ycol = LABEL_COLUMN[label_name]
    preds, scales, coefs = [], [], []
    mask = in_window(slim, window)
    for season in range(FIRST_TEST, LAST_SEASON + 1):
        train = slim[(slim.season < season) & mask]
        test = slim[(slim.season == season) & mask]
        assert train.season.max() < season
        ytr = train[ycol].to_numpy()
        keep = [] if intercept_only else [c for c in cols if train[c].nunique() > 1]
        if keep:
            model = column_model().fit(train[keep], ytr)
            p = model.predict(test[keep])
            if with_coef:
                names = list(keep) + [f"missing::{keep[i]}" for i in model[0].indicator_.features_]
                coefs.append(
                    {
                        "season": season,
                        "alpha": float(model[-1].alpha_),
                        "coef_standardized": dict(
                            zip(names, [float(v) for v in model[-1].coef_], strict=True)
                        ),
                    }
                )
        else:
            p = np.full(len(test), float(ytr.mean()))
            if with_coef:
                coefs.append({"season": season, "alpha": None, "coef_standardized": {}})
        if label_name == "gap":
            scale = float(np.cov(train.resid, train.label)[0, 1] / np.var(train.label, ddof=1))
        else:
            scale = 1.0
        preds.append(pd.Series(p, index=test.index))
        scales.append(pd.Series(scale, index=test.index))
    return pd.concat(preds), pd.concat(scales), coefs


def light(slim, p, scale, label_name):
    y = slim.loc[p.index, LABEL_COLUMN[label_name]].to_numpy()
    e = slim.loc[p.index, "resid"].to_numpy()
    q = p.to_numpy()
    sp = q * scale.loc[p.index].to_numpy()
    var = np.var(sp)
    return {
        "share": float(1 - ((y - q) ** 2).sum() / (y**2).sum()),
        "abs_gain": float(np.mean(np.abs(e) - np.abs(e - sp))),
        "slope": float(np.cov(sp, e, bias=True)[0, 1] / var) if var > 0 else float("nan"),
    }


def permuted_slim(slim, cols, season_index, rng):
    out = slim.copy()
    block = slim[cols].to_numpy()
    shuffled = block.copy()
    for ix in season_index:
        shuffled[ix] = block[ix[rng.permutation(len(ix))]]
    out[cols] = shuffled
    return out


def full_block(data, p, scale, label_name, window, open_all, rng):
    frame = data.loc[p.index].copy()
    frame["label"] = frame[LABEL_COLUMN[label_name]]
    open_w = open_all[open_all.index.isin(p.index)]
    resid_pred = p * scale.loc[p.index]
    share = p2p.label_block(frame, p, rng)
    result = p2p.result_block(frame, resid_pred, p, rng, open_w)
    for key in ("close_splits", "open_splits"):
        result.pop(key, None)
    return {
        "games": len(frame),
        "open_games": len(open_w),
        "share_of_label": share["r2_vs_zero"],
        "corr_with_label": share["corr"],
        "mae_model": result["mae"],
        "mae_line": result["line_mae"],
        "mae_improvement": {
            "value": result["abs_diff_vs_line"],
            "interval": result["abs_diff_interval"],
            "probability_positive": result["abs_diff_probability_positive"],
        },
        "slope_on_margin_residual": {
            "value": result["slope"],
            "interval": result["slope_interval"],
            "probability_positive": result["slope_probability_positive"],
        },
        "mean_abs_prediction_points": result["mean_abs_pred"],
        "close_pick": result["close_accuracy"],
        "open_pick": result["open_accuracy"],
    }


def coefficient_summary(coefs):
    names = sorted({n for c in coefs for n in c["coef_standardized"]})
    out = {}
    for n in names:
        v = np.array([c["coef_standardized"].get(n, 0.0) for c in coefs])
        out[n] = {
            "positive_folds": int((v > 0).sum()),
            "negative_folds": int((v < 0).sum()),
            "zero_folds": int((v == 0).sum()),
            "mean_standardized": float(v.mean()),
        }
    return out


def stack_walk(data, base_cols, cols, label_name):
    ycol = LABEL_COLUMN[label_name]
    preds, scales = [], []
    for season in range(FIRST_TEST, LAST_SEASON + 1):
        train = data[data.season < season]
        test = data[data.season == season]
        assert train.season.max() < season
        observed = [c for c in [*base_cols, *cols] if train[c].notna().any()]
        model = p2p.enet_model().fit(train[observed], train[ycol].to_numpy())
        preds.append(pd.Series(model.predict(test[observed]), index=test.index))
        if label_name == "gap":
            scale = float(np.cov(train.resid, train.label)[0, 1] / np.var(train.label, ddof=1))
        else:
            scale = 1.0
        scales.append(pd.Series(scale, index=test.index))
        print({"stack_fold": season, "label": label_name, "columns": len(observed)}, flush=True)
    return pd.concat(preds), pd.concat(scales)


def increment(data, p_base, p_ext, scale, label_name, window, rng):
    ycol = LABEL_COLUMN[label_name]
    idx = p_ext.index[in_window(data.loc[p_ext.index], window).to_numpy()]
    rows = []
    for _, g in data.loc[idx].groupby("season"):
        y = g[ycol].to_numpy()
        e = g.resid.to_numpy()
        pb = p_base.loc[g.index].to_numpy()
        pe = p_ext.loc[g.index].to_numpy()
        s = scale.loc[g.index].to_numpy()
        rows.append(
            [
                len(g),
                ((y - pb) ** 2).sum(),
                ((y - pe) ** 2).sum(),
                (y**2).sum(),
                (np.abs(e) - np.abs(e - s * pb)).sum(),
                (np.abs(e) - np.abs(e - s * pe)).sum(),
            ]
        )
    s = np.array(rows)
    draws = rng.integers(0, len(s), size=(BOOTSTRAP_DRAWS, len(s)))

    def stat(a):
        return (
            (a[:, 1].sum() - a[:, 2].sum()) / a[:, 3].sum(),
            (a[:, 5].sum() - a[:, 4].sum()) / a[:, 0].sum(),
        )

    point = stat(s)
    boots = np.array([stat(s[d]) for d in draws])
    out = {}
    for j, name in enumerate(("share_of_label_added", "mae_improvement_added")):
        out[name] = {
            "value": float(point[j]),
            "interval": p2n.interval(boots[:, j]),
            "probability_positive": float((boots[:, j] > 0).mean()),
        }
    return out


def fmt_pp(block):
    return f"{block['value']:+.4f}[{block['probability_positive']:.2f}]"


def text_table(results):
    lines = [f"tag {results['tag']}  columns {','.join(results['columns'])}", ""]
    lines.append(
        "label  window      games  close_record  open_record  share_of_label  "
        "mae_gain[p+]  slope[p+]  mean|pred|"
    )
    for label, by_window in results["cells"].items():
        for window, c in by_window.items():
            m = c["column_model"]
            cp = m["close_pick"]
            op = m["open_pick"]
            close_rec = f"{cp['record'][0]}-{cp['record'][1]}"
            open_rec = f"{op['record'][0]}-{op['record'][1]}" if op else "NA"
            lines.append(
                f"{label:6s} {window:11s} {m['games']:5d}  {close_rec:12s}  {open_rec:11s}  "
                f"{m['share_of_label']['value']:+.5f}  {fmt_pp(m['mae_improvement'])}  "
                f"{fmt_pp(m['slope_on_margin_residual'])}  {m['mean_abs_prediction_points']:.3f}"
            )
            i = c["intercept_only"]
            lines.append(
                f"   intercept-only: share {i['share_of_label']['value']:+.5f}  "
                f"mae_gain {i['mae_improvement']['value']:+.4f}  "
                f"close {i['close_pick']['record'][0]}-{i['close_pick']['record'][1]}"
            )
            if "in_sample_share_of_label" in c:
                lines.append(
                    f"   in-sample share {c['in_sample_share_of_label']:+.5f}  "
                    f"gap to out-of-sample {c['in_sample_minus_out_of_sample_share']:+.5f}"
                )
            pm = c.get("permutation")
            if pm:
                lines.append(
                    f"   permutation n={pm['n']} real percentile: "
                    f"share {pm['share']['real_percentile']:.3f}  "
                    f"mae_gain {pm['abs_gain']['real_percentile']:.3f}  "
                    f"slope {pm['slope']['real_percentile']:.3f}"
                )
            st = c.get("el_stack")
            if st:
                inc = st["increment"]
                lines.append(
                    f"   on top of 295-col EL: share added {fmt_pp(inc['share_of_label_added'])}  "
                    f"mae_gain added {fmt_pp(inc['mae_improvement_added'])}"
                )
            signs = "; ".join(
                f"{k} +{v['positive_folds']}/-{v['negative_folds']}/0:{v['zero_folds']}"
                for k, v in c["coefficient_signs"].items()
            )
            lines.append(f"   fold signs: {signs}")
    return "\n".join(lines)


def cells_for_registry(results, family, source):
    window = next(iter(WINDOWS))
    start = FIRST_TEST
    out = []
    for label in ("gap", "margin"):
        m = results["cells"][label][window]["column_model"]["mae_improvement"]
        out.append(
            {
                "name": f"{results['tag']}_{label}_label_{window}_mae_improvement",
                "description": (
                    f"{results['tag']} input vs {label} label, {window}, walk-forward: "
                    "MAE improvement on real margin vs closing line"
                ),
                "effect": m["value"],
                "effect_units": "mae_improvement",
                "interval_low": m["interval"][0],
                "interval_high": m["interval"][1],
                "probability_positive": m["probability_positive"],
                "season_start": start,
                "season_end": LAST_SEASON,
            }
        )
    return {
        "league": "nfl",
        "classification": "unresolved_below_power",
        "source": source,
        "category": "market",
        "notes": (
            "Elastic net on the named input only, trained on earlier seasons in the same "
            "week window, "
            "tested walk-forward 2015-25. Season-block bootstrap. Placebo window and permutation "
            "controls are in the source file."
        ),
        "classification_evidence": (
            "Walk-forward by season, probability_positive reported; no refuted mechanism "
            "and no closing ground."
        ),
        "plain_summary": f"Tests whether {results['tag']} predicts early-season line error.",
        "family": family,
        "cells": out,
    }


def main():
    ap = argparse.ArgumentParser(
        description="Hindsight-gap weeks 1-6 test harness for one set of pregame input columns"
    )
    ap.add_argument(
        "--features", required=True, help="parquet keyed by game_id with the input columns"
    )
    ap.add_argument("--columns", required=True, help="comma-separated input column names")
    ap.add_argument("--tag", required=True, help="short name for the input family")
    ap.add_argument("--out", required=True, help="output directory")
    ap.add_argument("--permutations", type=int, default=1000, help="within-season shuffles")
    ap.add_argument("--family", default=DEFAULT_FAMILY, help="weak-signals family for batch.json")
    ap.add_argument("--skip-stack", action="store_true", help="skip the 295-column stacked run")
    ap.add_argument("--primary-weeks", default="1-6", help="first-last week of the recorded window")
    ap.add_argument("--placebo-weeks", default="7-12", help="first-last week of the placebo window")
    args = ap.parse_args()
    WINDOWS.clear()
    for span in (args.primary_weeks, args.placebo_weeks):
        first, last = (int(v) for v in span.split("-"))
        WINDOWS[f"weeks_{first}_{last}"] = (first - 1, last)
    cols = args.columns.split(",")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    data, _counts, _served, all_cols = p2p.load_data()
    feats = pd.read_parquet(args.features)
    assert not set(cols) & set(data.columns)
    data = data.merge(feats[["game_id", *cols]], on="game_id", how="left", validate="one_to_one")
    data[cols] = data[cols].apply(pd.to_numeric, errors="coerce")
    missing_rows = int(data[cols].isna().all(axis=1).sum())
    print({"games": len(data), "rows_without_input": missing_rows, "columns": cols}, flush=True)

    slim = data[["game_id", "season", "week", "result", "resid", "label", *cols]].copy()
    test = data[data.season >= FIRST_TEST]
    ref = pd.read_parquet(p2p.P2N_PREDICTIONS)
    opener = ref[["game_id", "tue_open_home_spread"]].dropna()
    opener = opener[opener.game_id.isin(test[test.season >= OPENER_FIRST_SEASON].game_id)]
    open_all = test.reset_index().merge(opener, on="game_id").set_index("index")
    open_all["open_line"] = open_all.tue_open_home_spread

    stored = pd.read_parquet(p2p.OUT / "oos_predictions.parquet").set_index("game_id")
    stored_margin = ref.set_index("game_id")
    baseline = {
        "gap": pd.Series(
            stored.pred_label_EL.reindex(data.game_id).to_numpy(), index=data.index
        ).dropna(),
        "margin": pd.Series(
            stored_margin.pred_E.reindex(data.game_id).to_numpy(), index=data.index
        ).dropna(),
    }

    rng = np.random.default_rng(SEED)
    season_index = [np.where(slim.season.to_numpy() == s)[0] for s in sorted(slim.season.unique())]
    results = {
        "tag": args.tag,
        "columns": cols,
        "rows_without_input": missing_rows,
        "cells": {},
    }
    real_light = {}
    for label in LABEL_COLUMN:
        results["cells"][label] = {}
        for window in WINDOWS:
            p, scale, coefs = walk(slim, cols, label, window, with_coef=True)
            p0, scale0, _ = walk(slim, cols, label, window, intercept_only=True)
            cell = {
                "column_model": full_block(data, p, scale, label, window, open_all, rng),
                "intercept_only": full_block(data, p0, scale0, label, window, open_all, rng),
                "fold_coefficients": coefs,
                "coefficient_signs": coefficient_summary(coefs),
            }
            real_light[(label, window)] = light(slim, p, scale, label)
            ycol = LABEL_COLUMN[label]
            rows = slim[in_window(slim, window)]
            fit_cols = [c for c in cols if rows[c].nunique() > 1]
            if fit_cols:
                model = column_model().fit(rows[fit_cols], rows[ycol].to_numpy())
                scored = rows[rows.season >= FIRST_TEST]
                yv = scored[ycol].to_numpy()
                fitted = model.predict(scored[fit_cols])
                in_sample = float(1 - ((yv - fitted) ** 2).sum() / (yv**2).sum())
                cell["in_sample_share_of_label"] = in_sample
                cell["in_sample_minus_out_of_sample_share"] = (
                    in_sample - cell["column_model"]["share_of_label"]["value"]
                )
            results["cells"][label][window] = cell
            print({"label": label, "window": window, "done": "column model"}, flush=True)

    for label in LABEL_COLUMN:
        for window in WINDOWS:
            draws = {"share": [], "abs_gain": [], "slope": []}
            for _ in range(args.permutations):
                shuf = permuted_slim(slim, cols, season_index, rng)
                p, scale, _ = walk(shuf, cols, label, window)
                stats = light(shuf, p, scale, label)
                for k, v in stats.items():
                    draws[k].append(v)
            pm = {"n": args.permutations}
            for k, v in draws.items():
                arr = np.array(v, dtype=float)
                real = real_light[(label, window)][k]
                ok = arr[np.isfinite(arr)]
                pm[k] = {
                    "real": real,
                    "permuted_mean": float(ok.mean()) if len(ok) else None,
                    "permuted_p2_5": float(np.percentile(ok, 2.5)) if len(ok) else None,
                    "permuted_p97_5": float(np.percentile(ok, 97.5)) if len(ok) else None,
                    "real_percentile": float(
                        ((ok < real).sum() + 0.5 * (ok == real).sum()) / len(ok)
                    )
                    if len(ok)
                    else None,
                }
            results["cells"][label][window]["permutation"] = pm
            print({"label": label, "window": window, "done": "permutation"}, flush=True)

    if not args.skip_stack:
        for label in LABEL_COLUMN:
            p_ext, scale = stack_walk(data, all_cols, cols, label)
            p_base = baseline[label].loc[p_ext.index]
            for window in WINDOWS:
                idx = p_ext.index[in_window(data.loc[p_ext.index], window).to_numpy()]
                results["cells"][label][window]["el_stack"] = {
                    "base_only": full_block(
                        data, p_base.loc[idx], scale, label, window, open_all, rng
                    ),
                    "with_input": full_block(
                        data, p_ext.loc[idx], scale, label, window, open_all, rng
                    ),
                    "increment": increment(data, p_base, p_ext, scale, label, window, rng),
                }

    (out / f"{args.tag}_results.json").write_text(json.dumps(results, indent=1), encoding="utf-8")
    target = out.resolve() / f"{args.tag}_results.json"
    source = (
        target.relative_to(p2p.ROOT).as_posix() if target.is_relative_to(p2p.ROOT) else str(target)
    )
    batch = cells_for_registry(results, args.family, source)
    (out / f"{args.tag}_batch.json").write_text(json.dumps(batch, indent=1), encoding="utf-8")
    table = text_table(results)
    (out / f"{args.tag}_table.txt").write_text(table, encoding="utf-8")
    print(table)


if __name__ == "__main__":
    main()
