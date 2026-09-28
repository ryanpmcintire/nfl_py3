import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit, logit
from scipy.stats import binomtest

parser = argparse.ArgumentParser()
parser.add_argument("--start-season", type=int, default=2024)
parser.add_argument(
    "--output", type=Path, default=Path("artifacts/diagnostics/2026-three-weeks/recent-history")
)
args = parser.parse_args()
out = args.output
out.mkdir(parents=True, exist_ok=True)
live = pd.read_csv("artifacts/diagnostics/2026-three-weeks/published_grades.csv")
history_path = Path("artifacts/pick_probability/20260927T161329Z")
history = pd.read_parquet(history_path / "per_game.parquet")
metadata = json.loads((history_path / "metadata.json").read_text(encoding="utf-8"))
history = history[
    history.chronological_home_probability.notna() & history.season.ge(args.start_season)
].copy()
history["p"] = history.chronological_home_probability
history["raw_p"] = history.model_probability
history["y"] = history.home_covered.astype(float)
history["spread"] = history.tue_open_home_spread
history["picked_home"] = history.p.ge(0.5)
history["picked_probability"] = np.maximum(history.p, 1 - history.p)
history["hit"] = history.picked_home.eq(history.y)
history["raw_hit"] = history.raw_p.ge(0.5).eq(history.y)
history["disagree"] = history.picked_home.ne(history.raw_p.ge(0.5))
history["picked_line"] = history.spread.where(history.picked_home, -history.spread)


def distribution(ps):
    values = np.array([1.0])
    for p in ps:
        values = np.convolve(values, [1 - p, p])
    return values


def summarize(frame):
    rows = []
    for name, p in [
        ("combined", frame.p.to_numpy()),
        ("raw", frame.raw_p.to_numpy()),
        ("neutral_market", np.full(len(frame), 0.5)),
    ]:
        y = frame.y.to_numpy()
        clipped = np.clip(p, 1e-9, 1 - 1e-9)
        rows.append(
            {
                "arm": name,
                "games": len(frame),
                "wins": int(((p >= 0.5) == y).sum()),
                "brier": float(((p - y) ** 2).mean()),
                "log_loss": float((-y * np.log(clipped) - (1 - y) * np.log1p(-clipped)).mean()),
            }
        )
    return rows


live_rows = []
for week, frame in live[live.outcome.ne("pending")].groupby("week"):
    ps = frame.displayed_score.to_numpy()
    dist = distribution(ps)
    wins = int(frame.outcome.eq("won").sum())
    live_rows.append(
        {
            "week": int(week),
            "games": len(frame),
            "wins": wins,
            "expected": float(ps.sum()),
            "variance": float(np.sum(ps * (1 - ps))),
            "p_at_most": float(dist[: wins + 1].sum()),
            "p_at_least": float(dist[wins:].sum()),
            "central_95_win_range": np.searchsorted(np.cumsum(dist), [0.025, 0.975]).tolist(),
            "probabilities_verified": week in [2, 3],
            "halfpoint_wins": int(frame.ats_margin.eq(0.5).sum()),
            "halfpoint_losses": int(frame.ats_margin.eq(-0.5).sum()),
        }
    )

dispersion = {}
for included in [[2, 3], [1, 2, 3]]:
    groups = [live[live.week.eq(w) & live.outcome.ne("pending")] for w in included]
    distributions = [distribution(g.displayed_score.to_numpy()) for g in groups]
    expectations = [g.displayed_score.sum() for g in groups]
    variances = [(g.displayed_score * (1 - g.displayed_score)).sum() for g in groups]
    observed = sum(
        (g.outcome.eq("won").sum() - mu) ** 2 / var
        for g, mu, var in zip(groups, expectations, variances, strict=True)
    )
    tail = 0.0
    for wins in itertools.product(*(range(len(d)) for d in distributions)):
        stat = sum(
            (w - mu) ** 2 / var for w, mu, var in zip(wins, expectations, variances, strict=True)
        )
        if stat >= observed - 1e-12:
            tail += np.prod([d[w] for d, w in zip(distributions, wins, strict=True)])
    total = distribution(np.concatenate([g.displayed_score.to_numpy() for g in groups]))
    actual_total = sum(int(g.outcome.eq("won").sum()) for g in groups)
    dispersion["_".join(map(str, included))] = {
        "statistic": float(observed),
        "exact_independent_tail": float(tail),
        "games": sum(len(g) for g in groups),
        "wins": actual_total,
        "expected_wins": float(sum(expectations)),
        "total_wins_at_least": float(total[actual_total:].sum()),
        "total_wins_at_most": float(total[: actual_total + 1].sum()),
        "week1_scores_are_unverified_probabilities": 1 in included,
    }

weekly = []
for (season, week), frame in history.groupby(["season", "week"]):
    ps = frame.picked_probability.to_numpy()
    hits = int(frame.hit.sum())
    var = float(np.sum(ps * (1 - ps)))
    weekly.append(
        {
            "season": int(season),
            "week": int(week),
            "games": len(frame),
            "wins": hits,
            "accuracy": hits / len(frame),
            "expected": float(ps.sum()),
            "variance": var,
            "z": (hits - ps.sum()) / np.sqrt(var),
            "pearson": (hits - ps.sum()) ** 2 / var,
        }
    )
weekly = pd.DataFrame(weekly)
weekly.to_csv(out / "historical_weeks.csv", index=False)
windows = []
for season, frame in weekly.groupby("season"):
    records = frame.sort_values("week").to_dict("records")
    for length in [2, 3]:
        for start in range(len(records) - length + 1):
            group = records[start : start + length]
            if group[-1]["week"] - group[0]["week"] != length - 1:
                continue
            windows.append(
                {
                    "season": int(season),
                    "start_week": group[0]["week"],
                    "weeks": length,
                    "pearson": sum(r["pearson"] for r in group),
                    "worst_accuracy": min(r["accuracy"] for r in group),
                    "best_accuracy": max(r["accuracy"] for r in group),
                }
            )
windows = pd.DataFrame(windows)
windows.to_csv(out / "historical_windows.csv", index=False)

historical_summary = {
    "all": summarize(history),
    "per_season": {str(s): summarize(f) for s, f in history.groupby("season")},
    "week_count": len(weekly),
    "weekly_pearson_mean": float(weekly.pearson.mean()),
    "weeks_at_or_below_one_third": int(weekly.accuracy.le(1 / 3).sum()),
    "weeks_at_or_above_14_of_16": int(weekly.accuracy.ge(14 / 16).sum()),
    "window_tails": {
        str(n): {
            "windows": len(windows[windows.weeks.eq(n)]),
            "as_or_more_dispersed": int(
                windows.loc[windows.weeks.eq(n), "pearson"]
                .ge(dispersion["2_3" if n == 2 else "1_2_3"]["statistic"])
                .sum()
            ),
        }
        for n in [2, 3]
    },
}

blocks = history[["season", "week"]].drop_duplicates().reset_index(drop=True)
block_map = {tuple(row): i for i, row in enumerate(blocks.to_numpy())}
block_indices = np.array(
    [block_map[(s, w)] for s, w in zip(history.season, history.week, strict=True)]
)
rng = np.random.default_rng(20260928)
weights = rng.multinomial(len(blocks), np.full(len(blocks), 1 / len(blocks)), size=20000)


def estimate(values, mask):
    mask = np.asarray(mask, dtype=bool)
    values = np.asarray(values, dtype=float)
    if not mask.any():
        return None
    sums = np.bincount(block_indices[mask], weights=values[mask], minlength=len(blocks))
    counts = np.bincount(block_indices[mask], minlength=len(blocks))
    denominator = weights @ counts
    numerator = weights @ sums
    draws = numerator[denominator > 0] / denominator[denominator > 0]
    return {
        "estimate": float(values[mask].mean()),
        "lower": float(np.quantile(draws, 0.025)),
        "upper": float(np.quantile(draws, 0.975)),
        "probability_positive": float(np.mean(draws > 0) + 0.5 * np.mean(draws == 0)),
    }


groups = {
    "all": np.ones(len(history), dtype=bool),
    "raw_agree": ~history.disagree,
    "raw_disagree": history.disagree,
    "home": history.picked_home,
    "away": ~history.picked_home,
    "favorite": history.picked_line.gt(0),
    "underdog": history.picked_line.lt(0),
    "pickem": history.picked_line.eq(0),
    "confidence_below_55": history.picked_probability.lt(0.55),
    "confidence_55_to_60": history.picked_probability.ge(0.55)
    & history.picked_probability.lt(0.60),
    "confidence_at_least_60": history.picked_probability.ge(0.60),
    "spread_below_3": history.spread.abs().lt(3),
    "spread_3_through_7": history.spread.abs().ge(3) & history.spread.abs().le(7),
    "spread_above_7": history.spread.abs().gt(7),
    "weeks_1_to_3": history.week.le(3),
    "later": history.week.gt(3),
}
for flag in ["coach", "division", "arrests", "bye", "cold_visitor", "protection", "tank_zone"]:
    groups["flag_" + flag] = history["flag_" + flag].ne(0)
term_rows = []
for row in history.itertuples():
    coef = metadata["chronological_coefficients"][str(row.season)]
    term_rows.append(
        {
            "raw_model": coef["model_logit"] * row.model_logit,
            "situational_flags": coef["composition_flag_sum"] * row.composition_flag_sum,
            "market_movement": coef["market_move_toward_home"] * row.market_move_toward_home,
            "intercept_and_availability": coef["intercept"]
            + coef["market_move_available"] * row.market_move_available,
        }
    )
terms = pd.DataFrame(term_rows, index=history.index)
dominant = terms.abs().idxmax(axis=1)
reconstructed = expit(terms.sum(axis=1))
assert np.allclose(reconstructed, history.p, atol=1e-10)
for name in terms:
    groups["dominant_" + name] = dominant.eq(name)
history["dominant_term"] = dominant

group_results = {}
brier_gain = (history.raw_p - history.y) ** 2 - (history.p - history.y) ** 2
accuracy_gain = 100 * (history.hit.astype(float) - history.raw_hit.astype(float))
calibration_error = history.hit.astype(float) - history.picked_probability
for name, mask in groups.items():
    frame = history.loc[mask]
    result = {
        "games": len(frame),
        "wins": int(frame.hit.sum()),
        "expected_wins": float(frame.picked_probability.sum()),
        "brier_gain_over_raw": estimate(brier_gain, mask),
        "accuracy_gain_points": estimate(accuracy_gain, mask),
        "observed_minus_expected": estimate(calibration_error, mask),
        "arms": summarize(frame) if len(frame) else [],
        "per_season": {
            str(s): {
                "games": len(f),
                "wins": int(f.hit.sum()),
                "raw_wins": int(f.raw_hit.sum()),
                "expected_wins": float(f.picked_probability.sum()),
            }
            for s, f in frame.groupby("season")
        },
    }
    disagreement = frame[frame.disagree]
    result["decisive_disagreements"] = {
        "games": len(disagreement),
        "combined_wins": int(disagreement.hit.sum()),
        "exact_null_p": float(binomtest(int(disagreement.hit.sum()), len(disagreement), 0.5).pvalue)
        if len(disagreement)
        else None,
    }
    group_results[name] = result

reliability = []
for arm in ["p", "raw_p"]:
    for lo, hi in [(0, 0.4), (0.4, 0.5), (0.5, 0.6), (0.6, 1.00001)]:
        f = history[history[arm].ge(lo) & history[arm].lt(hi)]
        reliability.append(
            {
                "arm": arm,
                "lower": lo,
                "upper": min(hi, 1),
                "games": len(f),
                "mean_probability": float(f[arm].mean()) if len(f) else None,
                "home_cover_rate": float(f.y.mean()) if len(f) else None,
            }
        )
calibration_fits = {}
for name, p in [("combined", history.p), ("raw", history.raw_p)]:
    x = logit(np.clip(p.to_numpy(), 1e-6, 1 - 1e-6))
    y = history.y.to_numpy()
    fitted = minimize(
        lambda b, x=x, y=y: np.mean(np.logaddexp(0, b[0] + b[1] * x) - y * (b[0] + b[1] * x)),
        [0, 1],
        method="BFGS",
    )
    calibration_fits[name] = {
        "intercept": float(fitted.x[0]),
        "slope": float(fitted.x[1]),
        "success": bool(fitted.success),
        "scope": "descriptive fit to held-year predictions; not deployed",
    }

report = {
    "live_weeks": live_rows,
    "live_dispersion": dispersion,
    "historical": historical_summary,
    "historical_groups": group_results,
    "reliability": reliability,
    "calibration_fits": calibration_fits,
    "assumptions": [
        "Week-level Poisson-binomial probabilities assume calibrated probabilities and "
        "independent outcomes. They are not odds that the model is correct.",
        "Week 1 displayed scores were not produced by the current calibrated combination; "
        "Week 1 probability calculations are sensitivity-only.",
        "Historical recipes were designed using these seasons; chronological predictions "
        "remove fitting leakage but not research selection.",
    ],
    "look_counts": {
        "historical_groups_including_all": len(groups),
        "group_metric_intervals": 3 * len(groups),
        "reliability_cells": len(reliability),
        "calibration_fits": 2,
        "weekly_live_tails": 2 * len(live_rows),
        "joint_dispersion_tails": len(dispersion),
    },
}
history.to_parquet(out / "historical_diagnostic_rows.parquet", index=False)
(out / "statistical_diagnosis.json").write_text(
    json.dumps(report, indent=2, allow_nan=False), encoding="utf-8"
)
print("LIVE", json.dumps(live_rows))
print("DISPERSION", json.dumps(dispersion))
print("HISTORICAL", json.dumps(historical_summary))
for name in [
    "all",
    "raw_disagree",
    "weeks_1_to_3",
    "dominant_market_movement",
    "flag_coach",
    "confidence_at_least_60",
]:
    r = group_results[name]
    print(
        "GROUP",
        name,
        json.dumps(
            {
                k: r[k]
                for k in [
                    "games",
                    "wins",
                    "expected_wins",
                    "brier_gain_over_raw",
                    "accuracy_gain_points",
                    "observed_minus_expected",
                    "decisive_disagreements",
                ]
            }
        ),
    )
print("CALIBRATION", json.dumps(calibration_fits))
