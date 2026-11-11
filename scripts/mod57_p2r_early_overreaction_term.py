from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from lead98_price_skew import BASE, fit, loso, metrics, predict, reliability, summary
from mod57_p2q_market_reaction_speed import (
    CHECK_GAMES,
    CHECK_SEED,
    EARLY,
    RATINGS,
    surprise_values,
)
from mod57_u3_deadline_fair_line import block_bootstrap, decisive
from per_flag_weights_eval import SEED

from nfl_ats.pick_probability_fit import FIT_RIDGE, _fit_logit
from nfl_ats.pool_hook import crossed_push_mass, tuesday_market_lines

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts/mod57_p2r"
ARTIFACT = ROOT / "artifacts/pick_probability/20261111T011859Z/per_game.parquet"
FAMILY = "mod57_how_wrong_is_the_line"
PLACEBO = (7, 11)
FIRST_TEST = 2015
LAST_SEASON = 2025
DRAW_COUNT = 10000
FORWARD_COEFFICIENTS = ROOT / "artifacts/pick_probability/20261111T011859Z/coefficients.json"
FORWARD_GAMES = ROOT / "tests/scratch/fixpicks/root_m2_games.csv"
SCHEDULES = ROOT / "data/raw_schedules"
TERM = "early_surprise"
TAPER = "early_surprise_taper"
PERM = "early_surprise_permuted"
PLACE = "placebo_surprise"
WEEKS = tuple(range(EARLY[0], EARLY[1] + 1))
TAPER_WEIGHTS = dict(zip(WEEKS, np.linspace(1.0, 0.2, len(WEEKS)), strict=True))
METRIC_UNITS = (
    ("accuracy_points", "accuracy_points"),
    ("log_loss", "log_loss_improvement"),
    ("brier", "brier_improvement"),
)


def load_reg():
    reg = pd.read_parquet(RATINGS)
    reg["gameday"] = pd.to_datetime(reg["gameday"])
    return reg.sort_values(["gameday", "game_id"]).reset_index(drop=True)


def truncation_diffs(reg, full):
    pool = reg.loc[reg["season"] > reg["season"].min()]
    sample = pool.loc[pool["week"] > 1].sample(CHECK_GAMES, random_state=CHECK_SEED)
    diffs = []
    for _, g in sample.iterrows():
        keep = reg["gameday"] < g["gameday"]
        same = (reg["gameday"] == g["gameday"]) & (reg["season"] == g["season"])
        cut = pd.concat(
            [reg.loc[keep], reg.loc[same].assign(margin=0.0, close_line=0.0)]
        ).reset_index(drop=True)
        again = surprise_values(cut)
        a, f = again.loc[g["game_id"]], full.loc[g["game_id"]]
        diffs.append(0.0 if (np.isnan(a) and np.isnan(f)) else float(abs(a - f)))
    return diffs


def build_terms(data, rng):
    d2 = data["D2"]
    week = data["week"]
    early = week.between(*EARLY)
    data[TERM] = d2.where(early, 0.0).fillna(0.0)
    data[TAPER] = (d2 * week.map(TAPER_WEIGHTS)).fillna(0.0)
    permuted = d2.copy()
    for _, idx in data.groupby(["season", "week"]).groups.items():
        permuted.loc[idx] = rng.permutation(d2.loc[idx].to_numpy())
    data[PERM] = permuted.where(early, 0.0).fillna(0.0)
    data[PLACE] = d2.where(week.between(*PLACEBO), 0.0).fillna(0.0)
    return data


def improvement(entry, name):
    d = entry[name]
    if name == "accuracy_points":
        return d["delta_mean"], d["ci95"][0], d["ci95"][1], d["probability_positive"]
    return -d["delta_mean"], -d["ci95"][1], -d["ci95"][0], d["probability_positive"]


def primary(data, rng):
    y = data["home_covered"].to_numpy(float)
    week = data["week"]
    pops = {
        "all_games": np.ones(len(data), bool),
        "weeks_2_6": week.between(*EARLY).to_numpy(),
        "weeks_7_11": week.between(*PLACEBO).to_numpy(),
    }
    variants = {
        "four_term": BASE,
        "plus_T": (*BASE, TERM),
        "plus_T2_taper": (*BASE, TAPER),
        "control_permuted": (*BASE, PERM),
        "control_placebo_weeks_7_11": (*BASE, PLACE),
        "control_intercept_only": (),
    }
    probs, folds, insample = {}, {}, {}
    for name, feats in variants.items():
        oos, fo = loso(data, feats)
        probs[name], folds[name] = oos, fo
        b, m, s = fit(data, feats)
        insample[name] = predict(data, feats, b, m, s)
    out = {"folds": {}, "fold_summary": {}, "pops": {}}
    for name in ("plus_T", "plus_T2_taper", "control_permuted", "control_placebo_weeks_7_11"):
        out["folds"][name] = folds[name]
        out["fold_summary"][name] = summary(folds[name])
    for pop, mask in pops.items():
        sub = data.loc[mask].reset_index(drop=True)
        ys = y[mask]
        entry = {"games": int(mask.sum()), "variants": {}, "paired": {}, "decisive": {}}
        for name in variants:
            oos = metrics(probs[name][mask], ys)
            ins = metrics(insample[name][mask], ys)
            entry["variants"][name] = {
                "oos": oos,
                "in_sample": ins,
                "gap_in_minus_oos_accuracy_points": 100.0 * (ins["accuracy"] - oos["accuracy"]),
            }
        for name in variants:
            if name == "four_term":
                continue
            entry["paired"][name] = block_bootstrap(
                sub, probs[name][mask], probs["four_term"][mask], rng
            )
            entry["decisive"][name] = decisive(probs[name][mask], probs["four_term"][mask], ys)
        entry["reliability"] = {
            name: reliability(probs[name][mask], ys) for name in ("four_term", "plus_T")
        }
        out["pops"][pop] = entry
    full = fit(data, (*BASE, TERM))
    out["frozen_T"] = {"beta": [float(x) for x in full[0]], "means": full[1], "stds": full[2]}
    full2 = fit(data, (*BASE, TAPER))
    out["frozen_T2"] = {"beta": [float(x) for x in full2[0]], "means": full2[1], "stds": full2[2]}
    return out


def season_block_stats(frame, rng):
    block = pd.factorize(frame["season"])[0]
    n_blocks = int(block.max()) + 1
    sizes = np.bincount(block, minlength=n_blocks).astype(float)
    idx = rng.integers(0, n_blocks, (DRAW_COUNT, n_blocks))
    out = {}
    for name, v, scale in (
        ("accuracy_points", frame["hit"].to_numpy(float) - 0.5, 100.0),
        ("log_loss_improvement", frame["ll_gain"].to_numpy(float), 1.0),
    ):
        sums = np.bincount(block, weights=v, minlength=n_blocks)
        d = scale * sums[idx].sum(axis=1) / sizes[idx].sum(axis=1)
        out[name] = {
            "effect": float(scale * v.mean()),
            "ci95": [float(np.quantile(d, 0.025)), float(np.quantile(d, 0.975))],
            "probability_positive": float((d > 0).mean()),
        }
    return out


def power(reg, d2, rng):
    f = reg[["game_id", "season", "week", "margin", "close_line"]].copy()
    f["D2"] = d2.reindex(f["game_id"]).to_numpy()
    f = f.loc[f["week"].between(*EARLY) & f["D2"].notna() & (f["margin"] != f["close_line"])]
    f = f.assign(y=(f["margin"] > f["close_line"]).astype(float))
    parts = []
    coefficients = {}
    for test in range(FIRST_TEST, LAST_SEASON + 1):
        tr, te = f.loc[f["season"] < test], f.loc[f["season"] == test]
        mu, sd = float(tr["D2"].mean()), float(tr["D2"].std(ddof=0))
        x = np.column_stack([np.ones(len(tr)), (tr["D2"] - mu) / sd])
        b = _fit_logit(x, tr["y"].to_numpy(float), FIT_RIDGE)
        coefficients[test] = float(b[1] / sd)
        z = b[0] + b[1] * (te["D2"] - mu) / sd
        p = np.clip(1.0 / (1.0 + np.exp(-z.to_numpy())), 1e-9, 1 - 1e-9)
        y = te["y"].to_numpy(float)
        ll = -(y * np.log(p) + (1 - y) * np.log1p(-p))
        parts.append(
            pd.DataFrame(
                {
                    "season": test,
                    "week": te["week"].to_numpy(),
                    "hit": ((p >= 0.5) == (y == 1)).astype(float),
                    "ll_gain": np.log(2.0) - ll,
                }
            )
        )
    rows = pd.concat(parts, ignore_index=True)
    out = {"coefficient_by_test_season": coefficients, "pooled": season_block_stats(rows, rng)}
    out["pooled"]["games"] = len(rows)
    out["pooled"]["record"] = f"{int(rows['hit'].sum())}-{int(len(rows) - rows['hit'].sum())}"
    out["by_season"] = {
        str(s): {
            "games": len(g),
            "accuracy_points": float(100 * (g["hit"].mean() - 0.5)),
            "log_loss_gain": float(g["ll_gain"].mean()),
        }
        for s, g in rows.groupby("season")
    }
    out["by_week"] = {
        str(w): {
            "games": len(g),
            "record": f"{int(g['hit'].sum())}-{int(len(g) - g['hit'].sum())}",
            "accuracy_points": float(100 * (g["hit"].mean() - 0.5)),
            "log_loss_gain": float(g["ll_gain"].mean()),
        }
        for w, g in rows.groupby("week")
    }
    return out


def forward(frozen):
    snapshot = sorted(p for p in SCHEDULES.iterdir() if (p / "schedules.parquet").is_file())[-1]
    sched = pd.read_parquet(snapshot / "schedules.parquet")
    played = sched.loc[
        (sched["season"] == 2026) & sched["result"].notna() & sched["spread_line"].notna()
    ]
    reg26 = pd.DataFrame(
        {
            "game_id": played["game_id"],
            "season": played["season"],
            "gameday": pd.to_datetime(played["gameday"]),
            "home_team": played["home_team"],
            "away_team": played["away_team"],
            "margin": played["result"].astype(float),
            "close_line": played["spread_line"].astype(float),
        }
    ).reset_index(drop=True)
    d2 = surprise_values(reg26)
    coefficients = json.loads(FORWARD_COEFFICIENTS.read_text(encoding="utf-8"))
    c = coefficients["coefficients"]
    push = {float(a): float(m) for a, m in coefficients["hook_push_mass"]}
    d = pd.read_csv(FORWARD_GAMES)
    anchor = {}
    for week, g in d.groupby("week"):
        anchor.update(tuesday_market_lines(ROOT / "data", 2026, int(week), g["game_id"].tolist()))
    d["anchor"] = d["game_id"].map(anchor)
    masses, dirs = [], []
    for r in d.itertuples():
        m = (
            0.0
            if pd.isna(r.anchor)
            else crossed_push_mass(float(r.anchor), float(r.pool_home_spread), push)
        )
        masses.append(m)
        dirs.append(0.0 if m <= 0 else float(np.sign(r.pool_home_spread - r.anchor)))
    half = [0.5 * (1 - x * m) for x, m in zip(dirs, masses, strict=True)]
    d["hook"] = [0.0 if x == 0 else np.log(h / (1 - h)) for x, h in zip(dirs, half, strict=True)]
    s = d["raw_model_home_prob_final"].clip(1e-6, 1 - 1e-6)
    base = s.clip(1e-6, 1 - 1e-6)
    mv = d["served_move"].fillna(0.0)
    av = d["served_move_available"].fillna(0.0).astype(float)
    z = (
        c["intercept"]
        + c["model_logit"] * np.log(base / (1 - base))
        + c["move_toward_home"] * mv
        + c["move_available"] * av
        + c["pool_hook"] * d["hook"]
    )
    current = 1.0 / (1.0 + np.exp(-z.to_numpy()))
    week = d["week"]
    d2g = d["game_id"].map(d2)
    terms = {
        "plus_T": (frozen["frozen_T"], d2g.where(week.between(*EARLY), 0.0).fillna(0.0), TERM),
        "plus_T2_taper": (frozen["frozen_T2"], (d2g * week.map(TAPER_WEIGHTS)).fillna(0.0), TAPER),
    }
    y = d["hc"].to_numpy(float)
    mask = week.between(*EARLY).to_numpy()
    probs = {"current": current}
    for name, (fz, term, col) in terms.items():
        position = list(BASE).index(BASE[-1]) + 2
        beta = fz["beta"][position]
        contribution = beta * term.to_numpy() / fz["stds"][col]
        probs[name] = 1.0 / (1.0 + np.exp(-(z.to_numpy() + contribution)))
    sub = d.loc[mask, ["week", "hc"]].assign(season=2026).reset_index(drop=True)
    sub["home_covered"] = sub["hc"]
    rng = np.random.default_rng(SEED)
    out = {
        "games_weeks_2_6": int(mask.sum()),
        "games_all": len(d),
        "term_nonzero_games": int((terms["plus_T"][1].to_numpy()[mask] != 0).sum()),
        "frozen_natural_coefficient": {
            k: float(v["beta"][len(BASE) + 1] / v["stds"][col])
            for k, (v, _, col) in ((n, (t[0], t[1], t[2])) for n, t in terms.items())
        },
        "metrics": {k: metrics(p[mask], y[mask]) for k, p in probs.items()},
        "paired": {},
        "decisive": {},
    }
    for name in terms:
        out["paired"][name] = block_bootstrap(sub, probs[name][mask], current[mask], rng)
        out["decisive"][name] = decisive(probs[name][mask], current[mask], y[mask])
    changed = (probs["plus_T"] >= 0.5) != (current >= 0.5)
    out["changed_games_plus_T"] = (
        d.loc[changed, ["game_id", "week", "hc"]]
        .assign(current=current[changed], plus_T=probs["plus_T"][changed])
        .to_dict("records")
    )
    return out


def cells(res, pw, fw):
    out = []
    for arm in ("plus_T", "plus_T2_taper"):
        for pop in ("all_games", "weeks_2_6"):
            entry = res["pops"][pop]
            for metric, units in METRIC_UNITS:
                eff, lo, hi, pp = improvement(entry["paired"][arm], metric)
                out.append(
                    {
                        "name": f"mod57_p2r_{arm}_{pop}_{metric}",
                        "description": (
                            f"Fifth fitted term {arm} (early-season surprise, weeks 2-6) added "
                            f"to the served four-term probability, leave-one-season-out, "
                            f"{pop}, {metric} versus the four-term, week-block bootstrap; "
                            "improvement positive"
                        ),
                        "effect": eff,
                        "effect_units": units,
                        "interval_low": lo,
                        "interval_high": hi,
                        "probability_positive": pp,
                        "season_start": 2020,
                        "season_end": 2025,
                        "sample_games": entry["games"],
                    }
                )
    for name in ("control_permuted", "control_placebo_weeks_7_11", "control_intercept_only"):
        eff, lo, hi, pp = improvement(res["pops"]["weeks_2_6"]["paired"][name], "accuracy_points")
        out.append(
            {
                "name": f"mod57_p2r_{name}_weeks_2_6_accuracy_points",
                "description": (
                    f"Control {name} through the same leave-one-season-out fit, weeks 2-6 "
                    "accuracy versus the four-term, week-block bootstrap"
                ),
                "effect": eff,
                "effect_units": "accuracy_points",
                "interval_low": lo,
                "interval_high": hi,
                "probability_positive": pp,
                "season_start": 2020,
                "season_end": 2025,
                "sample_games": res["pops"]["weeks_2_6"]["games"],
            }
        )
    for metric in ("accuracy_points", "log_loss_improvement"):
        e = pw["pooled"][metric]
        out.append(
            {
                "name": f"mod57_p2r_power_walkforward_{metric}",
                "description": (
                    "Walk-forward one-term logistic on early-season surprise, fitted on "
                    "all earlier seasons from 2009, weeks 2-6 games against the closing line, "
                    f"{metric} versus 0.5, season-block bootstrap"
                ),
                "effect": e["effect"],
                "effect_units": metric,
                "interval_low": e["ci95"][0],
                "interval_high": e["ci95"][1],
                "probability_positive": e["probability_positive"],
                "season_start": FIRST_TEST,
                "season_end": LAST_SEASON,
                "sample_games": pw["pooled"]["games"],
            }
        )
    for metric, units in METRIC_UNITS:
        eff, lo, hi, pp = improvement(fw["paired"]["plus_T"], metric)
        out.append(
            {
                "name": f"mod57_p2r_forward_2026_{metric}",
                "description": (
                    "Forward read: early-season surprise term frozen on 2020-25 added to current "
                    f"structure, 2026 weeks 2-6 at the pool line, {metric} versus current, "
                    "week-block bootstrap; improvement positive"
                ),
                "effect": eff,
                "effect_units": units,
                "interval_low": lo,
                "interval_high": hi,
                "probability_positive": pp,
                "season_start": 2026,
                "season_end": 2026,
                "sample_games": fw["games_weeks_2_6"],
            }
        )
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    reg = load_reg()
    d2 = surprise_values(reg)
    trunc = truncation_diffs(reg, d2)
    rng = np.random.default_rng(SEED)
    data = pd.read_parquet(ARTIFACT)
    data = data.loc[data["season"].between(2020, 2025)].reset_index(drop=True)
    data["D2"] = d2.reindex(data["game_id"]).to_numpy()
    data = build_terms(data, rng)
    res = primary(data, rng)
    pw = power(reg, d2, rng)
    fw = forward(res)
    results = {
        "truncation_max_abs_diff": max(trunc),
        "games": len(data),
        "early_games_missing_D2": int((data["week"].between(*EARLY) & data["D2"].isna()).sum()),
        "primary": res,
        "power": pw,
        "forward": fw,
    }
    (OUT / "results.json").write_text(json.dumps(results, indent=1, default=str), encoding="utf-8")
    batch = {
        "league": "nfl",
        "classification": "unresolved_below_power",
        "source": "artifacts/mod57_p2r/results.json",
        "category": "market",
        "family": FAMILY,
        "notes": (
            "Early-season surprise to date (home minus away mean of margin minus close) in weeks "
            "2-6 as a fifth fitted term in the four-term pick probability, leave-one-season-out, "
            "with a walk-forward power read and a frozen 2026 forward read."
        ),
        "classification_evidence": (
            "Week-block and season-block bootstrap, probability_positive reported; no refuted "
            "mechanism and no closing ground."
        ),
        "plain_summary": (
            "Whether a team beating or missing the line early in a season makes the next game's "
            "line too high or too low."
        ),
        "cells": cells(res, pw, fw),
    }
    (OUT / "batch.json").write_text(json.dumps(batch, indent=1), encoding="utf-8")
    print("truncation max abs diff", max(trunc), "games", len(data))
    print("early games missing D2", results["early_games_missing_D2"])
    for pop, e in res["pops"].items():
        print("POP", pop, e["games"])
        for name, v in e["variants"].items():
            print(
                f"  {name} oos {v['oos']['record']} acc {v['oos']['accuracy']:.4f} "
                f"ll {v['oos']['log_loss']:.5f} brier {v['oos']['brier']:.5f} "
                f"in {v['in_sample']['accuracy']:.4f} "
                f"gap {v['gap_in_minus_oos_accuracy_points']:+.2f}"
            )
        for name, dec in e["decisive"].items():
            p = e["paired"][name]
            print(
                f"  vs four {name} changed {dec['changed_pick_games']} new {dec['new_wins']} "
                f"base {dec['base_wins']} | "
                + " | ".join(
                    f"{m} {p[m]['delta_mean']:+.5f} "
                    f"[{p[m]['ci95'][0]:+.5f},{p[m]['ci95'][1]:+.5f}] "
                    f"P+ {p[m]['probability_positive']:.3f}"
                    for m in ("accuracy_points", "log_loss", "brier")
                )
            )
    for k, v in res["fold_summary"].items():
        print(
            "FOLD",
            k,
            {
                n: [round(x, 4) for x in (a["mean"], a["min"], a["max"])] + [a["positive_folds"]]
                for n, a in v.items()
                if n in (TERM, TAPER, PERM, PLACE)
            },
        )
    print("RELIABILITY weeks 2-6", json.dumps(res["pops"]["weeks_2_6"]["reliability"]))
    print("POWER", json.dumps(pw, indent=None))
    print("FORWARD", json.dumps(fw, default=str))


if __name__ == "__main__":
    main()
