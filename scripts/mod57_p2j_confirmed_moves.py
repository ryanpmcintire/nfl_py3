from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from lead98_price_skew import BASE, design, fit, loso, metrics, predict, reliability, summary
from mod57_p2j_consensus import build
from mod57_u3_deadline_fair_line import block_bootstrap, decisive
from per_flag_weights_eval import SEED

from nfl_ats.pick_probability import MOVE_AVAILABLE_COLUMN, MOVE_COLUMN

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "artifacts/pick_probability/20261111T011859Z/per_game.parquet"
GAMES_2026 = ROOT / "tests/scratch/fixpicks/root_m2_games.csv"
PAYLOADS = ROOT / "tests/scratch/mod57/p2e/payload_prices.parquet"
COEFFICIENTS = ROOT / "artifacts/pick_probability/20261111T011859Z/coefficients.json"
OUT = ROOT / "tests/scratch/mod57/p2j"
FAMILY = "mod57_confirmed_moves"
CONS_MOVE = "consensus_move"
CONS_AVAIL = "consensus_available"
MIN_BOOKS = 5
FIRST_WEEK_2026 = 3
VARIANTS = {
    "four_term": BASE,
    "arm_j1_both": (*BASE, CONS_MOVE, CONS_AVAIL),
    "arm_j2_replace": ("model_logit", BASE[1], CONS_MOVE, CONS_AVAIL),
    "model_only": ("model_logit",),
    "intercept_only": (),
    "consensus_only_control": (CONS_MOVE, CONS_AVAIL),
}
ARMS = ("arm_j1_both", "arm_j2_replace")
METRICS = (
    ("accuracy_points", "accuracy_points"),
    ("log_loss", "log_loss_improvement"),
    ("brier", "brier_improvement"),
)


def improvement(entry, name):
    d = entry[name]
    if name == "accuracy_points":
        return d["delta_mean"], d["ci95"][0], d["ci95"][1], d["probability_positive"]
    return -d["delta_mean"], -d["ci95"][1], -d["ci95"][0], d["probability_positive"]


def cell(arm, pop, metric, units, boot, games, start, end, what):
    eff, lo, hi, pp = improvement(boot, metric)
    return {
        "name": f"mod57_p2j_{arm}_{pop}_{metric}",
        "description": (
            f"MOD-57 P2-J: {arm} versus the current served structure, {what}, "
            "week-block bootstrap; improvement positive."
        ),
        "effect": eff,
        "effect_units": units,
        "interval_low": lo,
        "interval_high": hi,
        "probability_positive": pp,
        "season_start": start,
        "season_end": end,
        "sample_games": int(games),
    }


def batch(cells, source):
    return {
        "family": FAMILY,
        "league": "nfl",
        "category": "market",
        "classification": "unresolved_below_power",
        "source": source,
        "cells": cells,
    }


def load_history():
    data = pd.read_parquet(ARTIFACT)
    data = data.loc[data["season"].between(2020, 2025)].reset_index(drop=True)
    table = build()
    data = data.merge(
        table[
            ["consensus_line", "consensus_books", "minutes_before_deadline_newest", "admissible"]
        ],
        left_on="game_id",
        right_index=True,
        how="left",
    )
    data[CONS_AVAIL] = data["admissible"].fillna(False).astype(float)
    data[CONS_MOVE] = (data["consensus_line"] - data["tue_open_home_spread"]).where(
        data[CONS_AVAIL] > 0
    )
    return data


def coverage(data):
    rows = {}
    for season, g in data.groupby("season"):
        ok = g[CONS_AVAIL] > 0
        rows[int(season)] = {
            "games": len(g),
            "admissible": int(ok.sum()),
            "median_minutes_before_deadline": float(
                g.loc[ok, "minutes_before_deadline_newest"].median()
            ),
            "min_minutes_before_deadline": float(g.loc[ok, "minutes_before_deadline_newest"].min()),
            "median_books": float(g.loc[ok, "consensus_books"].median()),
            "leader_move_available": int((g[MOVE_AVAILABLE_COLUMN] > 0).sum()),
        }
    return rows


def sign_check(data):
    both = data.loc[(data[CONS_AVAIL] > 0) & (data[MOVE_AVAILABLE_COLUMN] > 0)]
    nz = both.loc[(both[CONS_MOVE] != 0) & (both[MOVE_COLUMN] != 0)]
    ok = data[CONS_AVAIL] > 0
    return {
        "games_both": len(both),
        "corr_consensus_move_vs_leader_move": float(both[CONS_MOVE].corr(both[MOVE_COLUMN])),
        "same_sign_share_when_both_nonzero": float(
            (np.sign(nz[CONS_MOVE]) == np.sign(nz[MOVE_COLUMN])).mean()
        ),
        "corr_consensus_move_vs_open_move": float(
            data.loc[ok, CONS_MOVE].corr(data.loc[ok, "open_move"])
        ),
        "examples": both[
            ["game_id", "tue_open_home_spread", "consensus_line", CONS_MOVE, MOVE_COLUMN]
        ]
        .head(4)
        .to_dict("records"),
    }


def run_history(data):
    y = data["home_covered"].to_numpy(float)
    result = {"games": len(data), "coverage": coverage(data), "sign_check": sign_check(data)}
    print(json.dumps(result, indent=1, default=str), flush=True)
    probs, folds, insample, frozen = {}, {}, {}, {}
    for name, feats in VARIANTS.items():
        oos, fo = loso(data, feats)
        probs[name], folds[name] = oos, fo
        b, m, s = fit(data, feats)
        insample[name] = predict(data, feats, b, m, s)
        frozen[name] = {
            "features": list(feats),
            "beta": [float(v) for v in b],
            "means": m,
            "stds": s,
        }
    probs["market_half"] = np.full(len(data), 0.5)
    result["fold_coefficients"] = {n: folds[n] for n in VARIANTS if VARIANTS[n]}
    result["fold_summary"] = {n: summary(folds[n]) for n in result["fold_coefficients"]}
    result["j1_leader_coefficient_by_fold"] = {
        k: v[MOVE_COLUMN] for k, v in folds["arm_j1_both"].items()
    }
    populations = {
        "all_2020_25": np.ones(len(data), dtype=bool),
        "seasons_2020_22": data["season"].between(2020, 2022).to_numpy(),
        "seasons_2023_25": data["season"].between(2023, 2025).to_numpy(),
    }
    rng = np.random.default_rng(SEED)
    result["populations"] = {}
    cells = []
    for pop, mask in populations.items():
        sub = data.loc[mask].reset_index(drop=True)
        ys = y[mask]
        entry = {"games": int(mask.sum()), "decisive": {}, "paired": {}, "variants": {}}
        for arm in ARMS:
            entry["decisive"][arm] = decisive(probs[arm][mask], probs["four_term"][mask], ys)
            boot = block_bootstrap(sub, probs[arm][mask], probs["four_term"][mask], rng)
            entry["paired"][arm] = boot
            if pop == "all_2020_25":
                for metric, units in METRICS:
                    cells.append(
                        cell(
                            arm,
                            "2020_25",
                            metric,
                            units,
                            boot,
                            mask.sum(),
                            2020,
                            2025,
                            "leave-one-season-out 2020-25",
                        )
                    )
        for name in probs:
            v = {"oos": metrics(probs[name][mask], ys)}
            if name in insample:
                v["in_sample"] = metrics(insample[name][mask], ys)
                v["accuracy_gap_in_minus_oos_points"] = 100.0 * (
                    v["in_sample"]["accuracy"] - v["oos"]["accuracy"]
                )
            entry["variants"][name] = v
        entry["reliability"] = {n: reliability(probs[n][mask], ys) for n in ("four_term", *ARMS)}
        result["populations"][pop] = entry
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "frozen.json").write_text(json.dumps(frozen, indent=1), encoding="utf-8")
    (OUT / "results_history.json").write_text(
        json.dumps(result, indent=1, default=str), encoding="utf-8"
    )
    (OUT / "batch_history.json").write_text(
        json.dumps(batch(cells, "tests/scratch/mod57/p2j/results_history.json"), indent=1),
        encoding="utf-8",
    )
    print(json.dumps(result, indent=1, default=str))
    return frozen


def consensus_2026(d):
    t = pd.read_parquet(PAYLOADS)
    t["scan"] = pd.to_datetime(t["scan"], utc=True)
    t = t.dropna(subset=["home_line"])
    deadline = d.set_index("game_id")["pick_deadline_utc"].map(pd.Timestamp)
    out = {}
    for gid in d["game_id"]:
        _, _, away, home = gid.split("_")
        g = t.loc[t["home"].eq(home) & t["away"].eq(away) & t["scan"].lt(deadline[gid])]
        counts = g.groupby("scan")["book"].nunique()
        counts = counts.loc[counts >= MIN_BOOKS]
        if counts.empty:
            continue
        scan = counts.index.max()
        lines = g.loc[g["scan"].eq(scan), "home_line"]
        out[gid] = {
            "line": float(lines.median()),
            "books": len(lines),
            "minutes_before_deadline": (deadline[gid] - scan).total_seconds() / 60,
        }
    return pd.DataFrame(out).T


def run_2026(frozen):
    from nfl_ats.pool_hook import crossed_push_mass, tuesday_market_lines

    d = pd.read_csv(GAMES_2026)
    d = d.loc[d["week"] >= FIRST_WEEK_2026].reset_index(drop=True)
    co = json.loads(COEFFICIENTS.read_text(encoding="utf-8"))
    c = co["coefficients"]
    push = {float(a): float(m) for a, m in co["hook_push_mass"]}
    anchor = {}
    for week, g in d.groupby("week"):
        anchor.update(tuesday_market_lines(ROOT / "data", 2026, int(week), g.game_id.tolist()))
    d["anchor"] = d.game_id.map(anchor)
    masses, dirs = [], []
    for r in d.itertuples():
        m = (
            0.0
            if pd.isna(r.anchor)
            else crossed_push_mass(float(r.anchor), float(r.pool_home_spread), push)
        )
        masses.append(m)
        dirs.append(0.0 if m <= 0 else float(np.sign(r.pool_home_spread - r.anchor)))
    d["mass"], d["dir"] = masses, dirs
    d["hook"] = [
        0.0 if x == 0 else np.log((0.5 * (1 - x * m)) / (1 - 0.5 * (1 - x * m)))
        for x, m in zip(d.dir, d.mass, strict=True)
    ]
    y = d.hc.to_numpy(float)
    s = d.raw_model_home_prob_final.clip(1e-6, 1 - 1e-6)
    d["model_logit"] = np.log(s / (1 - s))
    avail = d.served_move_available.fillna(0).astype(float)
    d[MOVE_AVAILABLE_COLUMN] = avail
    d[MOVE_COLUMN] = d.served_move.where(avail > 0)
    d["composition_flag_sum"] = np.nan
    cons = consensus_2026(d)
    d["cons_raw"] = d.game_id.map(cons["line"].astype(float))
    corr = float(d["cons_raw"].corr(d["tuesday_market_line"]))
    d["consensus_line"] = (1.0 if corr > 0 else -1.0) * d["cons_raw"]
    d[CONS_AVAIL] = d["consensus_line"].notna().astype(float)
    d[CONS_MOVE] = (d["consensus_line"] - d["tuesday_market_line"]).where(d[CONS_AVAIL] > 0)
    info = {
        "games": len(d),
        "consensus_available": int(d[CONS_AVAIL].sum()),
        "payload_sign_corr_with_tuesday_line": corr,
        "median_minutes_before_deadline": float(
            cons["minutes_before_deadline"].astype(float).median()
        ),
        "min_minutes_before_deadline": float(cons["minutes_before_deadline"].astype(float).min()),
        "median_books": float(cons["books"].astype(float).median()),
        "corr_consensus_move_vs_leader_move": float(d[CONS_MOVE].corr(d[MOVE_COLUMN])),
    }
    print(json.dumps(info, indent=1, default=str), flush=True)
    z = (
        c["intercept"]
        + c["model_logit"] * d.model_logit
        + c["move_toward_home"] * d[MOVE_COLUMN].fillna(0)
        + c["move_available"] * avail
        + c["pool_hook"] * d.hook
    )
    probs = {"current": 1 / (1 + np.exp(-z))}
    for name in ARMS:
        f = frozen[name]
        zz = (
            design(d, f["features"], f["means"], f["stds"]) @ np.array(f["beta"])
            + c["pool_hook"] * d.hook.to_numpy()
        )
        probs[name] = 1 / (1 + np.exp(-zz))
    res = {
        "info": info,
        "metrics": {k: metrics(p, y) for k, p in probs.items()},
        "decisive": {},
        "paired": {},
        "changed": {},
    }
    frame = pd.DataFrame({"season": 2026, "week": d.week, "home_covered": y})
    rng = np.random.default_rng(SEED)
    cells = []
    for name in ARMS:
        res["decisive"][name] = decisive(probs[name], probs["current"], y)
        boot = block_bootstrap(frame, probs[name], probs["current"], rng)
        res["paired"][name] = boot
        for metric, units in METRICS:
            cells.append(
                cell(
                    name,
                    "2026",
                    metric,
                    units,
                    boot,
                    len(d),
                    2026,
                    2026,
                    "coefficients frozen 2020-25, 2026 Weeks 3-9 at the pool line",
                )
            )
        dm = (probs[name] >= 0.5) != (probs["current"] >= 0.5)
        res["changed"][name] = (
            d.assign(cur=probs["current"], new=probs[name])
            .loc[dm, ["game_id", "pool_home_spread", "served_move", CONS_MOVE, "cur", "new", "hc"]]
            .round(3)
            .to_dict("records")
        )
    (OUT / "results_2026.json").write_text(json.dumps(res, indent=1, default=str), encoding="utf-8")
    (OUT / "batch_2026.json").write_text(
        json.dumps(batch(cells, "tests/scratch/mod57/p2j/results_2026.json"), indent=1),
        encoding="utf-8",
    )
    print(json.dumps(res, indent=1, default=str))


def main():
    stage = sys.argv[1] if len(sys.argv) > 1 else "all"
    if stage == "2026":
        run_2026(json.loads((OUT / "frozen.json").read_text(encoding="utf-8")))
        return
    frozen = run_history(load_history())
    if stage == "all":
        run_2026(frozen)


if __name__ == "__main__":
    main()
