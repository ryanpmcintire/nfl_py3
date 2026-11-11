from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from mod57_p2c_uncorrected_error_model import BOOTSTRAP_SAMPLES, BOOTSTRAP_SEED
from mod57_p2o_how_wrong_is_the_line import (
    THIRD_EDGES,
    THIRD_LABELS,
    design,
    fit_coef,
    record,
    rng_draws,
    team_index,
)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts/mod57_p4a"
RATINGS = ROOT / "artifacts/mod57_p2o/per_game_ratings_LEAK_NOT_SERVABLE.parquet"
FAMILY = "mod57_future_market_label"
PRIMARY = (3, 8)
ALTERNATIVE = (1, 4)
PLACEBO = (-8, -3)
MIN_WEEKS = 3
SHUFFLE_SEED = 20261113
FIRST_SEASON = 2009
LAST_SEASON = 2025
OPENER_FIRST = 2020
EARLY = "weeks_1_6"
SCOPES = ("overall", *THIRD_LABELS)


def build_label(reg, index, window, shuffle_rng=None):
    pred = np.full(len(reg), np.nan)
    lo, hi = window
    n = len(index)
    for _, ids in reg.groupby("season").indices.items():
        sdf = reg.iloc[ids]
        present = sorted(sdf["week"].unique())
        for w in present:
            wanted = [x for x in present if w + min(lo, hi) <= x <= w + max(lo, hi)]
            if len(wanted) < MIN_WEEKS:
                continue
            train = sdf.loc[sdf["week"].isin(wanted)]
            coef = fit_coef(design(train, index), train["close_line"].to_numpy(float))
            counts = np.bincount(train["home_team"].map(index), minlength=n) + np.bincount(
                train["away_team"].map(index), minlength=n
            )
            target = sdf.loc[sdf["week"] == w].copy()
            if shuffle_rng is not None:
                target["home_team"] = shuffle_rng.permutation(target["home_team"].to_numpy())
                target["away_team"] = shuffle_rng.permutation(target["away_team"].to_numpy())
            hi_i = target["home_team"].map(index).to_numpy()
            ai_i = target["away_team"].map(index).to_numpy()
            ok = (counts[hi_i] > 0) & (counts[ai_i] > 0)
            value = design(target, index) @ coef
            positions = ids[np.flatnonzero(sdf["week"].to_numpy() == w)]
            pred[positions[ok]] = value[ok]
    return pred


def slope_draws(x, y, block, idx):
    n_blocks = int(block.max()) + 1
    n = np.bincount(block, minlength=n_blocks).astype(float)
    sx = np.bincount(block, weights=x, minlength=n_blocks)
    sy = np.bincount(block, weights=y, minlength=n_blocks)
    sxx = np.bincount(block, weights=x * x, minlength=n_blocks)
    sxy = np.bincount(block, weights=x * y, minlength=n_blocks)

    def slope(sel):
        nn, a, b, c, d = (v[sel].sum(axis=-1) for v in (n, sx, sy, sxx, sxy))
        return (d - a * b / nn) / (c - a * a / nn)

    return float(slope(np.arange(n_blocks))), slope(idx)


def interval(point, draws):
    return {
        "mean": point,
        "ci95": [float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))],
        "probability_positive": float((draws > 0).mean()),
    }


def pick_block(d, pred, line, rng):
    dec = d.loc[(d["margin"] != d[line]) & (d[pred] != d[line]) & d[line].notna()]
    if len(dec) < 20:
        return None
    hit = ((dec[pred] > dec[line]) == (dec["margin"] > dec[line])).to_numpy()
    block = pd.factorize(dec["season"])[0]
    out = record(hit)
    out["points_over_50"] = float(100.0 * (hit.mean() - 0.5))
    if block.max() >= 1:
        draws = 100.0 * rng_draws(hit.astype(float) - 0.5, block, rng)
        out["ci95"] = [float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))]
        out["probability_positive"] = float((draws > 0).mean())
    return out


def scope_mask(frame, scope):
    if scope == "overall":
        return np.ones(len(frame), bool)
    cut = pd.cut(frame["week"], THIRD_EDGES, labels=THIRD_LABELS).astype(str)
    return (cut == scope).to_numpy()


def scope_block(frame, col, rng, hind_idx):
    ok = frame[col].notna() & frame["R_H"].notna()
    d = frame.loc[ok].reset_index(drop=True)
    if len(d) < 30:
        return None
    label = d[col] - d["close_line"]
    hind = d["R_H"] - d["close_line"]
    real = d["margin"] - d["close_line"]
    block = pd.factorize(d["season"])[0]
    idx = rng.integers(0, int(block.max()) + 1, (BOOTSTRAP_SAMPLES, int(block.max()) + 1))
    s_l, dr_l = slope_draws(label.to_numpy(), real.to_numpy(), block, idx)
    s_h, dr_h = slope_draws(hind.to_numpy(), real.to_numpy(), block, idx)
    out = {
        "games": len(d),
        "seasons": int(block.max()) + 1,
        "label_slope": interval(s_l, dr_l),
        "hindsight_slope": interval(s_h, dr_h),
        "slope_difference": interval(s_l - s_h, dr_l - dr_h),
        "mean_abs_label": float(label.abs().mean()),
        "mean_abs_hindsight": float(hind.abs().mean()),
        "corr_with_hindsight_gap": float(np.corrcoef(label, hind)[0, 1]),
        "picks_close": pick_block(d, col, "close_line", rng),
        "hindsight_picks_close": pick_block(d, "R_H", "close_line", rng),
        "picks_open": pick_block(d.loc[d["season"] >= OPENER_FIRST], col, "open_line", rng),
        "hindsight_picks_open": pick_block(
            d.loc[d["season"] >= OPENER_FIRST], "R_H", "open_line", rng
        ),
    }
    return out


def coverage(frame, col):
    ok = frame[col].notna()
    return {
        int(s): [int(ok[frame["season"] == s].sum()), int((frame["season"] == s).sum())]
        for s in sorted(frame["season"].unique())
    }


def evaluate(frame, col, rng):
    result = {}
    for scope in SCOPES:
        sub = frame.loc[scope_mask(frame, scope)]
        result[scope] = scope_block(sub, col, rng, None)
    early = frame.loc[frame["week"].between(1, 6)]
    result[EARLY] = scope_block(early, col, rng, None)
    return result


def cells(results):
    out = []
    for tag, window in (("primary", PRIMARY), ("alternative", ALTERNATIVE)):
        e = results[tag]["scopes"][EARLY]
        w = f"weeks {window[0]} to {window[1]} ahead"
        sd = e["slope_difference"]
        out.append(
            {
                "name": f"mod57_p4a_{tag}_weeks_1_6_slope_difference_vs_hindsight",
                "description": (
                    f"Calibration slope on the real result against the closing line, weeks 1-6, "
                    f"of the later-market label ({w}, same season, own week excluded) minus the "
                    "hindsight label slope on the same games; both are future-information "
                    "training labels, not servable; season-block bootstrap"
                ),
                "effect": sd["mean"],
                "effect_units": "slope_difference",
                "interval_low": sd["ci95"][0],
                "interval_high": sd["ci95"][1],
                "probability_positive": sd["probability_positive"],
                "season_start": FIRST_SEASON,
                "season_end": LAST_SEASON,
            }
        )
    for tag, window in (("primary", PRIMARY), ("alternative", ALTERNATIVE)):
        p = results[tag]["scopes"][EARLY]["picks_close"]
        w = f"weeks {window[0]} to {window[1]} ahead"
        out.append(
            {
                "name": f"mod57_p4a_{tag}_weeks_1_6_picks_close_accuracy_points",
                "description": (
                    f"Pick accuracy minus 50 against the close, weeks 1-6, side of the later-"
                    f"market label ({w}); future-information label, not servable; decisive "
                    "games; season-block bootstrap"
                ),
                "effect": p["points_over_50"],
                "effect_units": "accuracy_points",
                "interval_low": p["ci95"][0],
                "interval_high": p["ci95"][1],
                "probability_positive": p["probability_positive"],
                "season_start": FIRST_SEASON,
                "season_end": LAST_SEASON,
            }
        )
    return out


def line(tag, e):
    if e is None:
        return f"{tag} undefined"
    pc, po = e["picks_close"], e["picks_open"]
    hc, ho = e["hindsight_picks_close"], e["hindsight_picks_open"]

    def fmt(p):
        return "na" if p is None else f"{p['record']} {100 * p['accuracy']:.1f}%"

    return (
        f"{tag} n {e['games']} slope {e['label_slope']['mean']:+.3f} "
        f"[{e['label_slope']['ci95'][0]:+.3f},{e['label_slope']['ci95'][1]:+.3f}] | hind "
        f"{e['hindsight_slope']['mean']:+.3f} | diff {e['slope_difference']['mean']:+.3f} "
        f"[{e['slope_difference']['ci95'][0]:+.3f},{e['slope_difference']['ci95'][1]:+.3f}] "
        f"P+ {e['slope_difference']['probability_positive']:.3f} | |label| "
        f"{e['mean_abs_label']:.2f} hind {e['mean_abs_hindsight']:.2f} | corr "
        f"{e['corr_with_hindsight_gap']:.3f} | close {fmt(pc)} hind {fmt(hc)} | open "
        f"{fmt(po)} hind {fmt(ho)}"
    )


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    reg = pd.read_parquet(RATINGS)
    reg["gameday"] = pd.to_datetime(reg["gameday"])
    reg = reg.sort_values(["gameday", "game_id"]).reset_index(drop=True)
    index = team_index(reg)
    frame = reg.copy()
    frame["F_primary"] = build_label(reg, index, PRIMARY)
    frame["F_alternative"] = build_label(reg, index, ALTERNATIVE)
    frame["F_placebo_past"] = build_label(reg, index, PLACEBO)
    frame["F_shuffled_primary"] = build_label(
        reg, index, PRIMARY, np.random.default_rng(SHUFFLE_SEED)
    )
    frame["F_shuffled_alternative"] = build_label(
        reg, index, ALTERNATIVE, np.random.default_rng(SHUFFLE_SEED + 1)
    )
    for tag, col in (
        ("primary", "F_primary"),
        ("alternative", "F_alternative"),
    ):
        frame[f"label_{tag}"] = frame[col] - frame["close_line"]
    frame[
        [
            "game_id",
            "season",
            "week",
            "label_primary",
            "label_alternative",
            "F_primary",
            "F_alternative",
        ]
    ].to_parquet(OUT / "future_market_label.parquet", index=False)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    results = {}
    for tag, col in (
        ("primary", "F_primary"),
        ("alternative", "F_alternative"),
        ("placebo_past", "F_placebo_past"),
        ("shuffled_primary", "F_shuffled_primary"),
        ("shuffled_alternative", "F_shuffled_alternative"),
    ):
        results[tag] = {"coverage": coverage(frame, col), "scopes": evaluate(frame, col, rng)}
    results["windows"] = {"primary": PRIMARY, "alternative": ALTERNATIVE, "placebo": PLACEBO}
    (OUT / "results.json").write_text(json.dumps(results, indent=1, default=str), encoding="utf-8")
    batch = {
        "league": "nfl",
        "classification": "unresolved_below_power",
        "source": "artifacts/mod57_p4a/results.json",
        "category": "market",
        "family": FAMILY,
        "notes": (
            "Later-market rating label (closing lines of later weeks of the same season) versus "
            "the hindsight label, weeks 1-6; uses future lines, training label only. Later lines "
            "re-rate each team for its own game, so the label carries the game's own result "
            "(calibration slope above 1); a higher slope here is contamination, not quality."
        ),
        "classification_evidence": (
            "Season-block bootstrap, probability_positive reported; no refuted mechanism and "
            "no closing ground."
        ),
        "plain_summary": (
            "Whether what the market thinks of two teams a few weeks later is a cleaner target "
            "for how wrong the early line is than game results are."
        ),
        "cells": cells(results),
    }
    (OUT / "batch.json").write_text(json.dumps(batch, indent=1), encoding="utf-8")
    for tag in results:
        if tag == "windows":
            continue
        cov = results[tag]["coverage"]
        print(tag, "coverage", {k: v[0] for k, v in cov.items()})
        for scope in (EARLY, *SCOPES):
            print(" ", line(scope, results[tag]["scopes"][scope]))


if __name__ == "__main__":
    main()
