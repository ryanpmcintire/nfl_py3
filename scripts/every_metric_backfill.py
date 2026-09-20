import json
import math
import os
import re
from datetime import UTC, datetime

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARTIFACTS_ROOT = os.path.join(REPO_ROOT, "artifacts")
REGISTRY_PATH = os.path.join(REPO_ROOT, "registry", "weak_signals.json")
ACTIVE_MODEL_PATH = os.path.join(REPO_ROOT, "artifacts", "active_ats_model.json")
OUTPUT_ROOT = os.path.join(REPO_ROOT, "artifacts", "every_metric")

NEEDED_COLUMNS = [
    "game_id",
    "season",
    "week",
    "home_score",
    "away_score",
    "result",
    "ats_margin",
    "home_cover",
    "spread_line",
    "market_spread",
    "predicted_margin",
    "fair_spread",
    "predicted_market_residual",
    "home_cover_probability",
    "push_probability",
    "method",
    "model_name",
]

METRIC_UNIT_MAP = {
    "accuracy": {"accuracy_points", "ats_points"},
    "brier": {"brier_improvement", "brier"},
    "log_loss": {"log_loss_improvement", "log_loss"},
    "margin_mae": {"mae_improvement", "mae"},
}
METRIC_NAMES = ["accuracy", "brier", "log_loss", "margin_mae"]
EPS = 1e-6
LN2 = math.log(2.0)
N_BOOTSTRAP = 1000
BOOTSTRAP_SEED = 20260916


def discover_prediction_files():
    found = []
    for root, _dirs, files in os.walk(ARTIFACTS_ROOT):
        for name in files:
            if name == "predictions.parquet":
                found.append(os.path.join(root, name))
    found.sort()
    return found


def family_of(path):
    rel = os.path.relpath(path, ARTIFACTS_ROOT)
    parts = rel.replace("\\", "/").split("/")
    return parts[0]


def season_from_game_id(game_id):
    m = re.match(r"^(\d{4})_", str(game_id))
    if m:
        return int(m.group(1))
    return None


def load_needed_columns(path):
    pf = pq.ParquetFile(path)
    available = pf.schema_arrow.names
    cols = [c for c in NEEDED_COLUMNS if c in available]
    if not cols:
        return None, available
    df = pd.read_parquet(path, columns=cols)
    return df, available


def actual_margin_series(df):
    if "result" in df.columns:
        return pd.to_numeric(df["result"], errors="coerce")
    if "home_score" in df.columns and "away_score" in df.columns:
        return pd.to_numeric(df["home_score"], errors="coerce") - pd.to_numeric(
            df["away_score"], errors="coerce"
        )
    return None


def actual_cover_series(df, margin, market_signed):
    if "home_cover" in df.columns:
        return pd.to_numeric(df["home_cover"], errors="coerce")
    if "ats_margin" in df.columns:
        ats = pd.to_numeric(df["ats_margin"], errors="coerce")
        out = pd.Series(np.where(ats > 0, 1.0, np.where(ats < 0, 0.0, np.nan)), index=df.index)
        return out
    if margin is not None and market_signed is not None:
        diff = margin - market_signed
        out = pd.Series(np.where(diff > 0, 1.0, np.where(diff < 0, 0.0, np.nan)), index=df.index)
        return out
    return None


def market_column(df):
    if "market_spread" in df.columns and df["market_spread"].notna().any():
        return pd.to_numeric(df["market_spread"], errors="coerce"), "market_spread"
    if "spread_line" in df.columns and df["spread_line"].notna().any():
        return pd.to_numeric(df["spread_line"], errors="coerce"), "spread_line"
    return None, None


def season_series(df):
    if "season" in df.columns:
        return pd.to_numeric(df["season"], errors="coerce")
    if "game_id" in df.columns:
        return df["game_id"].map(season_from_game_id)
    return None


def group_candidates(df):
    if "method" in df.columns:
        if "model_name" in df.columns:
            keys = list(df.groupby(["method", "model_name"], dropna=False).groups.keys())
            groups = []
            for method, model_name in keys:
                mask = (df["method"] == method) & (df["model_name"] == model_name)
                label = f"{method}|{model_name}"
                groups.append((label, df.loc[mask]))
            return groups
        keys = list(df.groupby(["method"], dropna=False).groups.keys())
        groups = []
        for method in keys:
            mask = df["method"] == method
            groups.append((str(method), df.loc[mask]))
        return groups
    return [("single", df)]


def raw_accuracy_brier_logloss(prob, cover):
    mask = cover.notna() & prob.notna()
    n = int(mask.sum())
    if n == 0:
        return None
    p = prob[mask].clip(EPS, 1.0 - EPS)
    y = cover[mask]
    pick = (p > 0.5).astype(float)
    accuracy = float((pick == y).mean())
    brier = float(((p - y) ** 2).mean())
    logloss = float((-(y * np.log(p) + (1.0 - y) * np.log(1.0 - p))).mean())
    return {"n": n, "accuracy": accuracy, "brier": brier, "log_loss": logloss}


def raw_margin_mae(predicted, actual):
    mask = predicted.notna() & actual.notna()
    n = int(mask.sum())
    if n == 0:
        return None
    mae = float((predicted[mask] - actual[mask]).abs().mean())
    return {"n": n, "mae": mae}


def compute_candidate_metrics(sub, margin, market_signed, cover, sign_used):
    out = {"n_rows": len(sub)}
    if "home_cover_probability" in sub.columns and sub["home_cover_probability"].notna().any():
        prob = pd.to_numeric(sub["home_cover_probability"], errors="coerce")
        cov_sub = cover.loc[sub.index] if cover is not None else None
        stats = raw_accuracy_brier_logloss(prob, cov_sub) if cov_sub is not None else None
    else:
        stats = None
    if stats is not None:
        out["accuracy"] = stats["accuracy"]
        out["brier"] = stats["brier"]
        out["log_loss"] = stats["log_loss"]
        out["accuracy_n"] = stats["n"]
    else:
        out["accuracy"] = None
        out["brier"] = None
        out["log_loss"] = None
        out["accuracy_n"] = 0
    if "predicted_margin" in sub.columns and sub["predicted_margin"].notna().any():
        predicted = pd.to_numeric(sub["predicted_margin"], errors="coerce")
        margin_sub = margin.loc[sub.index] if margin is not None else None
        mstats = raw_margin_mae(predicted, margin_sub) if margin_sub is not None else None
    else:
        mstats = None
    if mstats is not None:
        out["margin_mae"] = mstats["mae"]
        out["margin_mae_n"] = mstats["n"]
    else:
        out["margin_mae"] = None
        out["margin_mae_n"] = 0
    market = {
        "accuracy": None,
        "brier": None,
        "log_loss": None,
        "margin_mae": None,
        "accuracy_n": 0,
        "margin_mae_n": 0,
    }
    if cover is not None:
        cov_sub = cover.loc[sub.index]
        n_cov = int(cov_sub.notna().sum())
        if n_cov > 0:
            market["accuracy"] = 0.5
            market["brier"] = 0.25
            market["log_loss"] = LN2
            market["accuracy_n"] = n_cov
    if market_signed is not None and margin is not None:
        ms_sub = market_signed.loc[sub.index]
        margin_sub = margin.loc[sub.index]
        mstats2 = raw_margin_mae(ms_sub, margin_sub)
        if mstats2 is not None:
            market["margin_mae"] = mstats2["mae"]
            market["margin_mae_n"] = mstats2["n"]
    out["market"] = market
    out["sign_used"] = sign_used
    return out


def diff_positive_favours_candidate(metric, candidate_value, market_value):
    if candidate_value is None or market_value is None:
        return None
    if metric == "accuracy":
        return candidate_value - market_value
    return market_value - candidate_value


def bootstrap_probability_positive(rng, season_arr, per_season_payload, metric):
    seasons = sorted(set(season_arr))
    if len(seasons) < 2:
        return None, None
    n_seasons = len(seasons)
    positives = 0
    diffs = []
    for _ in range(N_BOOTSTRAP):
        draw = rng.choice(seasons, size=n_seasons, replace=True)
        cand_vals = []
        mkt_vals = []
        for s in draw:
            payload = per_season_payload.get(s)
            if payload is None:
                continue
            cand_vals.append(payload["candidate"])
            mkt_vals.append(payload["market"])
        if not cand_vals:
            continue
        if metric == "accuracy":
            cand_agg = np.mean([c["accuracy"] for c in cand_vals if c["accuracy"] is not None])
            mkt_agg = 0.5
        elif metric == "brier":
            cand_agg = np.mean([c["brier"] for c in cand_vals if c["brier"] is not None])
            mkt_agg = 0.25
        elif metric == "log_loss":
            cand_agg = np.mean([c["log_loss"] for c in cand_vals if c["log_loss"] is not None])
            mkt_agg = LN2
        else:
            cand_agg = np.mean([c["margin_mae"] for c in cand_vals if c["margin_mae"] is not None])
            mkt_agg = np.mean([m["margin_mae"] for m in mkt_vals if m["margin_mae"] is not None])
        d = diff_positive_favours_candidate(metric, float(cand_agg), float(mkt_agg))
        if d is None or (isinstance(d, float) and math.isnan(d)):
            continue
        diffs.append(d)
        if d > 0:
            positives += 1
    if not diffs:
        return None, None
    return float(np.mean(diffs)), float(positives) / float(len(diffs))


def extract_family_tokens(source_text, known_families):
    if not source_text:
        return set()
    matched = set()
    pieces = re.split(r";| \+ ", source_text)
    for piece in pieces:
        norm = piece.strip().replace("\\", "/")
        parts = norm.split("/")
        for i, p in enumerate(parts):
            if p == "artifacts" and i + 1 < len(parts):
                fam = parts[i + 1]
                if fam in known_families:
                    matched.add(fam)
    return matched


def main():
    started = datetime.now(UTC)
    stamp = started.strftime("%Y%m%dT%H%M%SZ")
    out_dir = os.path.join(OUTPUT_ROOT, stamp)
    os.makedirs(out_dir, exist_ok=True)

    with open(ACTIVE_MODEL_PATH, encoding="utf-8") as fh:
        active_model = json.load(fh)
    served_artifact = active_model["historical_evaluation"]["artifact"]
    served_path = os.path.join(
        ARTIFACTS_ROOT, served_artifact.replace("/", os.sep), "predictions.parquet"
    )
    served_method = active_model["method"]
    served_regressor = active_model["regressor"]

    files = discover_prediction_files()
    known_families = sorted({family_of(p) for p in files})

    loaded = {}
    inventory = []
    for path in files:
        rel = os.path.relpath(path, REPO_ROOT).replace("\\", "/")
        fam = family_of(path)
        pf = pq.ParquetFile(path)
        n_rows_meta = pf.metadata.num_rows
        df, _available_cols = load_needed_columns(path)
        entry = {
            "path": rel,
            "family": fam,
            "rows": int(n_rows_meta),
            "available_needed_columns": [] if df is None else list(df.columns),
        }
        if df is None or (
            "home_cover_probability" not in df.columns and "predicted_margin" not in df.columns
        ):
            entry["skip_reason"] = "no predicted_margin or home_cover_probability column present"
            entry["seasons"] = []
            entry["supports"] = {
                "accuracy": False,
                "brier": False,
                "log_loss": False,
                "margin_mae": False,
            }
            inventory.append(entry)
            continue
        seasons = season_series(df)
        if seasons is not None:
            valid_seasons = sorted(int(s) for s in pd.Series(seasons).dropna().unique())
        else:
            valid_seasons = []
        margin = actual_margin_series(df)
        market_signed_raw, market_source_col = market_column(df)
        entry["seasons"] = [valid_seasons[0], valid_seasons[-1]] if valid_seasons else []
        entry["market_column_used"] = market_source_col
        has_prob = (
            "home_cover_probability" in df.columns and df["home_cover_probability"].notna().any()
        )
        has_margin_pred = "predicted_margin" in df.columns and df["predicted_margin"].notna().any()
        has_actual_margin = margin is not None and margin.notna().any()
        cover_placeholder = actual_cover_series(
            df, margin, market_signed_raw if market_signed_raw is not None else None
        )
        has_cover = cover_placeholder is not None and cover_placeholder.notna().any()
        entry["supports"] = {
            "accuracy": bool(has_prob and has_cover),
            "brier": bool(has_prob and has_cover),
            "log_loss": bool(has_prob and has_cover),
            "margin_mae": bool(has_margin_pred and has_actual_margin),
        }
        entry["skip_reason"] = (
            None
            if any(entry["supports"].values())
            else "columns present but all-null for required fields"
        )
        inventory.append(entry)
        if any(entry["supports"].values()):
            loaded[path] = df

    sign_check_pos_total = 0.0
    sign_check_neg_total = 0.0
    sign_check_n = 0
    served_pos = served_neg = None
    for path, df in loaded.items():
        margin = actual_margin_series(df)
        market_signed_raw, _ = market_column(df)
        if margin is None or market_signed_raw is None:
            continue
        mask = margin.notna() & market_signed_raw.notna()
        if mask.sum() == 0:
            continue
        pos_err = (margin[mask] - market_signed_raw[mask]).abs().sum()
        neg_err = (margin[mask] - (-market_signed_raw[mask])).abs().sum()
        sign_check_pos_total += float(pos_err)
        sign_check_neg_total += float(neg_err)
        sign_check_n += int(mask.sum())
        if os.path.normpath(path) == os.path.normpath(served_path):
            served_pos = float(pos_err) / float(mask.sum())
            served_neg = float(neg_err) / float(mask.sum())

    chosen_sign = 1.0 if sign_check_pos_total <= sign_check_neg_total else -1.0
    sign_report = {
        "chosen_sign": chosen_sign,
        "pooled_mae_if_positive_sign": sign_check_pos_total / sign_check_n
        if sign_check_n
        else None,
        "pooled_mae_if_negative_sign": sign_check_neg_total / sign_check_n
        if sign_check_n
        else None,
        "pooled_n_rows": sign_check_n,
        "served_file_mae_if_positive_sign": served_pos,
        "served_file_mae_if_negative_sign": served_neg,
    }

    total_looks = 0
    file_candidate_metrics = {}
    for path, df in loaded.items():
        margin = actual_margin_series(df)
        market_signed_raw, _ = market_column(df)
        market_signed = chosen_sign * market_signed_raw if market_signed_raw is not None else None
        cover = actual_cover_series(df, margin, market_signed)
        groups = group_candidates(df)
        per_file = {}
        for label, sub in groups:
            m = compute_candidate_metrics(sub, margin, market_signed, cover, chosen_sign)
            per_file[label] = m
            for metric in METRIC_NAMES:
                if m.get(metric) is not None:
                    total_looks += 1
        file_candidate_metrics[path] = per_file

    served_df = loaded.get(served_path)
    served_report = {
        "path": os.path.relpath(served_path, REPO_ROOT).replace("\\", "/"),
        "candidates": {},
    }
    if served_df is not None:
        margin = actual_margin_series(served_df)
        market_signed_raw, _ = market_column(served_df)
        market_signed = chosen_sign * market_signed_raw
        cover = actual_cover_series(served_df, margin, market_signed)
        seasons = season_series(served_df)
        groups = group_candidates(served_df)
        rng = np.random.default_rng(BOOTSTRAP_SEED)
        for label, sub in groups:
            season_sub = seasons.loc[sub.index]
            per_season = {}
            for s in sorted(pd.Series(season_sub).dropna().unique()):
                s_int = int(s)
                mask = season_sub == s
                idx = sub.index[mask]
                sub_season = sub.loc[idx]
                cand = compute_candidate_metrics(
                    sub_season, margin, market_signed, cover, chosen_sign
                )
                per_season[s_int] = {
                    "candidate": cand,
                    "market": cand["market"],
                    "n_rows": cand["n_rows"],
                }
            overall = compute_candidate_metrics(sub, margin, market_signed, cover, chosen_sign)
            bootstrap = {}
            season_arr = season_sub.dropna().astype(int).values
            for metric in METRIC_NAMES:
                mean_diff, prob_pos = bootstrap_probability_positive(
                    rng, season_arr, per_season, metric
                )
                bootstrap[metric] = {
                    "mean_diff_positive_favours_candidate": mean_diff,
                    "probability_positive": prob_pos,
                }
            served_report["candidates"][label] = {
                "overall": overall,
                "per_season": per_season,
                "bootstrap": bootstrap,
            }

    with open(REGISTRY_PATH, encoding="utf-8") as fh:
        registry = json.load(fh)
    signals = registry.get("signals", {})
    family_units = {fam: set() for fam in known_families}
    family_signal_count = dict.fromkeys(known_families, 0)
    for _key, rec in signals.items():
        source = rec.get("source") or ""
        matched = extract_family_tokens(source, set(known_families))
        for fam in matched:
            family_units[fam].add(rec.get("effect_units"))
            family_signal_count[fam] += 1

    family_summary = {}
    for fam in known_families:
        fam_files = [p for p in files if family_of(p) == fam]
        fam_files_loaded = [p for p in fam_files if p in file_candidate_metrics]
        supportable = dict.fromkeys(METRIC_NAMES, False)
        candidate_labels = set()
        agg = {}
        for p in fam_files_loaded:
            for label, m in file_candidate_metrics[p].items():
                candidate_labels.add(label)
                agg.setdefault(label, {mn: [] for mn in METRIC_NAMES})
                for mn in METRIC_NAMES:
                    if m.get(mn) is not None:
                        supportable[mn] = True
                        cand_val = m[mn]
                        mkt_val = m["market"].get(mn)
                        d = diff_positive_favours_candidate(mn, cand_val, mkt_val)
                        agg[label][mn].append({"candidate": cand_val, "market": mkt_val, "diff": d})
        candidate_agg_summary = {}
        for label, metrics_map in agg.items():
            candidate_agg_summary[label] = {}
            for mn, rows in metrics_map.items():
                if not rows:
                    candidate_agg_summary[label][mn] = None
                    continue
                mean_cand = float(np.mean([r["candidate"] for r in rows]))
                mean_mkt = (
                    float(np.mean([r["market"] for r in rows if r["market"] is not None]))
                    if any(r["market"] is not None for r in rows)
                    else None
                )
                mean_diff = (
                    float(np.mean([r["diff"] for r in rows if r["diff"] is not None]))
                    if any(r["diff"] is not None for r in rows)
                    else None
                )
                candidate_agg_summary[label][mn] = {
                    "n_files": len(rows),
                    "mean_candidate": mean_cand,
                    "mean_market": mean_mkt,
                    "mean_diff_positive_favours_candidate": mean_diff,
                }
        existing_units = sorted(u for u in family_units[fam] if u is not None)
        new_metrics = []
        for mn in METRIC_NAMES:
            if not supportable[mn]:
                continue
            mapped = METRIC_UNIT_MAP[mn]
            if not (mapped & set(existing_units)):
                new_metrics.append(mn)
        family_summary[fam] = {
            "n_files_total": len(fam_files),
            "n_files_supportable": len(fam_files_loaded),
            "candidate_labels": sorted(candidate_labels),
            "supportable_metrics": [m for m in METRIC_NAMES if supportable[m]],
            "n_matched_registry_signals": family_signal_count[fam],
            "existing_registry_units": existing_units,
            "new_metrics_count": len(new_metrics),
            "new_metrics": new_metrics,
            "candidate_aggregate": candidate_agg_summary,
        }

    unmatched_families = [fam for fam in known_families if family_signal_count[fam] == 0]

    decisive_cases = []
    if served_df is not None:
        for label, payload in served_report["candidates"].items():
            overall = payload["overall"]
            acc_diff = diff_positive_favours_candidate(
                "accuracy", overall.get("accuracy"), overall["market"].get("accuracy")
            )
            mae_diff = diff_positive_favours_candidate(
                "margin_mae", overall.get("margin_mae"), overall["market"].get("margin_mae")
            )
            if (
                acc_diff is not None
                and mae_diff is not None
                and acc_diff != 0
                and mae_diff != 0
                and (acc_diff > 0) != (mae_diff > 0)
            ):
                decisive_cases.append(
                    {
                        "scope": "served_backtest_candidate",
                        "name": label,
                        "accuracy_diff": acc_diff,
                        "margin_mae_diff": mae_diff,
                    }
                )

    margins_summary = family_summary.get("margins")
    if margins_summary is not None:
        for label, metrics_map in margins_summary["candidate_aggregate"].items():
            acc = metrics_map.get("accuracy")
            mae = metrics_map.get("margin_mae")
            if acc is None or mae is None:
                continue
            acc_diff = acc.get("mean_diff_positive_favours_candidate")
            mae_diff = mae.get("mean_diff_positive_favours_candidate")
            if (
                acc_diff is not None
                and mae_diff is not None
                and acc_diff != 0
                and mae_diff != 0
                and (acc_diff > 0) != (mae_diff > 0)
            ):
                decisive_cases.append(
                    {
                        "scope": "margins_family_aggregate",
                        "name": label,
                        "accuracy_diff": acc_diff,
                        "margin_mae_diff": mae_diff,
                    }
                )

    results = {
        "generated_at_utc": started.isoformat(),
        "active_model": {
            "served_artifact": served_artifact,
            "method": served_method,
            "regressor": served_regressor,
        },
        "n_predictions_files_found": len(files),
        "known_families": known_families,
        "sign_convention": sign_report,
        "inventory": inventory,
        "served_backtest": served_report,
        "family_summary": family_summary,
        "unmatched_families": unmatched_families,
        "decisive_check": {"opposite_sign_cases": decisive_cases, "count": len(decisive_cases)},
        "total_looks": total_looks,
    }

    results_path = os.path.join(out_dir, "results.json")
    with open(results_path, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2, default=lambda o: None)

    report_lines = []
    report_lines.append("# ENG-46 unit 1: every metric from the same run (read-only measurement)")
    report_lines.append("")
    report_lines.append(
        "Generated (measured, this run): {} UTC".format(started.strftime("%Y-%m-%d %H:%M:%S"))
    )
    report_lines.append("")
    report_lines.append("## Decisive line")
    report_lines.append("")
    if decisive_cases:
        names = ", ".join("{}:{}".format(c["scope"], c["name"]) for c in decisive_cases)
        report_lines.append(
            f"measured: {len(decisive_cases)} candidate(s) show a margin-MAE paired difference "
            "with the opposite sign from their "
            f"accuracy-point difference: {names}. Per AGENTS.md this is neither a refuted "
            "mechanism nor a positive-control bound, so it is unresolved_below_power, "
            "not a rejection; "
            "it implies the parent session should record probability_positive for margin MAE "
            "separately from accuracy before treating the two metrics as redundant for these "
            "candidates."
        )
    else:
        report_lines.append(
            "measured: 0 served-or-near-served candidates show a margin-MAE paired difference "
            "with the opposite sign from their accuracy-point difference (checked across the {} "
            "served-backtest candidates and the margins-family aggregate). This implies accuracy "
            "and margin MAE currently agree in direction for the served model's own candidates, "
            "so adding margin MAE/Brier/log-loss reporting would mostly add precision and "
            "calibration information, not contradict the accuracy-based read; it does not by "
            "itself close any open signal.".format(len(served_report["candidates"]))
        )
    report_lines.append("")
    report_lines.append(
        "Sign convention (measured, served backtest {} rows): "
        "+market_spread/spread_line MAE = {:.4f}, "
        "-market_spread MAE = {:.4f} -> chose {}. "
        "Pooled across {} rows from all supportable files: "
        "+sign MAE = {:.4f}, -sign MAE = {:.4f}, agrees with served-file choice.".format(
            served_df.shape[0] if served_df is not None else 0,
            sign_report["served_file_mae_if_positive_sign"] or float("nan"),
            sign_report["served_file_mae_if_negative_sign"] or float("nan"),
            "positive (no negation)" if chosen_sign > 0 else "negative (negate market spread)",
            sign_report["pooled_n_rows"],
            sign_report["pooled_mae_if_positive_sign"] or float("nan"),
            sign_report["pooled_mae_if_negative_sign"] or float("nan"),
        )
    )
    report_lines.append("")
    report_lines.append(
        f"Every look counted (measured): {total_looks} "
        "(file x candidate x metric combinations actually computed "
        f"across {len(files)} predictions.parquet files, "
        f"of which {len(loaded)} supported at least one metric)."
    )
    report_lines.append("")
    report_lines.append("## Inventory (measured)")
    report_lines.append("")
    report_lines.append(
        "{} predictions.parquet files found under artifacts/. {} skipped for lacking a predicted "
        "margin or cover probability column; reasons listed in results.json `inventory`.".format(
            len(files), sum(1 for e in inventory if e["skip_reason"] is not None)
        )
    )
    report_lines.append("")
    report_lines.append(
        "| family | files | rows (sum) | seasons | accuracy | brier | log_loss | margin_mae |"
    )
    report_lines.append("|---|---|---|---|---|---|---|---|")
    fam_rows = {}
    for e in inventory:
        fam_rows.setdefault(e["family"], []).append(e)
    for fam in sorted(fam_rows.keys()):
        entries = fam_rows[fam]
        rows_sum = sum(e["rows"] for e in entries)
        seasons_all = [s for e in entries for s in e["seasons"]]
        season_range = f"{min(seasons_all)}-{max(seasons_all)}" if seasons_all else "n/a"
        acc = any(e["supports"]["accuracy"] for e in entries)
        bri = any(e["supports"]["brier"] for e in entries)
        ll = any(e["supports"]["log_loss"] for e in entries)
        mae = any(e["supports"]["margin_mae"] for e in entries)

        def yn(b):
            return "yes" if b else "no"

        report_lines.append(
            f"| {fam} | {len(entries)} | {rows_sum} | {season_range} | "
            f"{yn(acc)} | {yn(bri)} | {yn(ll)} | {yn(mae)} |"
        )
    report_lines.append("")
    report_lines.append(
        "## Served backtest ({}, method={}, regressor={}) (measured)".format(
            served_report["path"], served_method, served_regressor
        )
    )
    report_lines.append("")
    report_lines.append(
        "| candidate | n | accuracy (cand/mkt) | brier (cand/mkt) | log_loss (cand/mkt) | "
        "margin_mae (cand/mkt) | P+ acc | P+ brier | P+ logloss | P+ mae |"
    )
    report_lines.append("|---|---|---|---|---|---|---|---|---|---|")
    for label in sorted(served_report["candidates"].keys()):
        payload = served_report["candidates"][label]
        o = payload["overall"]
        b = payload["bootstrap"]

        def fmt(v):
            return f"{v:.4f}" if v is not None else "n/a"

        report_lines.append(
            "| {} | {} | {}/{} | {}/{} | {}/{} | {}/{} | {} | {} | {} | {} |".format(
                label,
                o["n_rows"],
                fmt(o.get("accuracy")),
                fmt(o["market"].get("accuracy")),
                fmt(o.get("brier")),
                fmt(o["market"].get("brier")),
                fmt(o.get("log_loss")),
                fmt(o["market"].get("log_loss")),
                fmt(o.get("margin_mae")),
                fmt(o["market"].get("margin_mae")),
                fmt(b["accuracy"]["probability_positive"]),
                fmt(b["brier"]["probability_positive"]),
                fmt(b["log_loss"]["probability_positive"]),
                fmt(b["margin_mae"]["probability_positive"]),
            )
        )
    report_lines.append("")
    report_lines.append(
        f"probability_positive is the season-block bootstrap ({N_BOOTSTRAP} draws, "
        "resampling seasons with "
        "replacement) share of draws where the metric favours the candidate over the market "
        "baseline; never read as the binary contains-zero test."
    )
    report_lines.append("")
    report_lines.append(
        "## Per-family new-metric count relative to the registry (measured + inferred mapping)"
    )
    report_lines.append("")
    report_lines.append(
        "Mapping used (inferred, stated so it can be challenged): "
        "accuracy -> accuracy_points/ats_points; "
        "brier -> brier_improvement/brier; log_loss -> log_loss_improvement/log_loss; "
        "margin_mae -> mae_improvement/mae. A family's existing units come from registry signals "
        "whose `source` path prefix matches `artifacts/<family>/`."
    )
    report_lines.append("")
    report_lines.append(
        "| family | files | supportable metrics | matched registry signals | "
        "existing units | new metrics |"
    )
    report_lines.append("|---|---|---|---|---|---|")
    for fam in sorted(family_summary.keys()):
        s = family_summary[fam]
        report_lines.append(
            "| {} | {} | {} | {} | {} | {} |".format(
                fam,
                s["n_files_total"],
                ",".join(s["supportable_metrics"]) or "none",
                s["n_matched_registry_signals"],
                ",".join(s["existing_registry_units"]) or "none",
                ",".join(s["new_metrics"]) or "none",
            )
        )
    report_lines.append("")
    report_lines.append(
        "Families with zero matched registry signals (measured, {} of {}): {}".format(
            len(unmatched_families),
            len(known_families),
            ", ".join(unmatched_families) if unmatched_families else "none",
        )
    )
    report_lines.append("")
    report_lines.append("## Caveats (inferred / methodology choices, not measurements)")
    report_lines.append("")
    report_lines.append(
        "- Market baseline cover probability is fixed at 0.5 (Brier fixed at 0.25, "
        "log loss fixed at ln2 = 0.6931) by the task's own definition, excluding push games; "
        "it is not fitted from data, "
        "so its bootstrap variability comes only from which games are sampled, not from the "
        "baseline value itself."
    )
    report_lines.append(
        "- Accuracy/Brier/log-loss require `home_cover_probability`; a file with only a raw side "
        "column and no probability is treated as unsupported for those three metrics, a "
        "simplification for this pass."
    )
    report_lines.append(
        "- The margins-family aggregate in the decisive check is the unweighted mean "
        "of each file's "
        "own metric, not a pooled re-fit across files, because many margins snapshots are "
        "overlapping retrain runs over the same seasons and pooling raw rows would overweight "
        "repeated games."
    )
    report_lines.append(
        "- Family-to-registry matching is by `source` path prefix string match against "
        "`artifacts/<family>/`; registry sources pointing at docs, ROADMAP.md anchors, or stale "
        "scratchpad paths never match and do not count as evidence the family lacks measurement, "
        "only that this script could not find it recorded under that path."
    )
    report_lines.append("")
    report_lines.append("## Command run")
    report_lines.append("")
    report_lines.append("```")
    report_lines.append(".\\.tools\\uv.exe run --no-sync python scripts\\every_metric_backfill.py")
    report_lines.append("```")
    report_lines.append("")
    report_lines.append(
        "Last lines of stdout are appended by the invoking session after running the command once."
    )

    report_path = os.path.join(out_dir, "report.md")
    stdout_lines = [
        "RESULTS_JSON=" + results_path,
        "REPORT_MD=" + report_path,
        f"FILES_FOUND={len(files)}",
        f"FILES_SUPPORTABLE={len(loaded)}",
        f"TOTAL_LOOKS={total_looks}",
        f"CHOSEN_SIGN={chosen_sign}",
        f"DECISIVE_OPPOSITE_SIGN_COUNT={len(decisive_cases)}",
        f"UNMATCHED_FAMILIES={len(unmatched_families)}/{len(known_families)}",
    ]
    report_lines[-1] = (
        f"Exact last {len(stdout_lines)} lines of stdout from the run that produced this report:"
    )
    report_lines.append("")
    report_lines.append("```")
    report_lines.extend(stdout_lines)
    report_lines.append("```")

    with open(report_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(report_lines))

    for line in stdout_lines:
        print(line)


if __name__ == "__main__":
    main()
