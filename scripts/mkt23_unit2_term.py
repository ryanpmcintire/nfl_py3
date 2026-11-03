import json
import subprocess
import sys

import mkt23_move_decomposition as D
import numpy as np
import pandas as pd
import sky04_lead_lag as L

OUT = D.OUT
FAMILY = "mkt23_unit2_term"
SEASONS = (2023, 2024, 2025)
ARMS = {
    "a_intercept": [],
    "b2_base_refit": ["logit"],
    "c_direction": ["logit", "direction"],
    "d_one_team_direction": ["logit", "one_team_direction"],
    "e_both": ["logit", "direction", "one_team_direction"],
}
VS_B = ("c_direction", "d_one_team_direction", "e_both", "b2_base_refit")
LOOKS = len(VS_B) * 2 + 1
NEWTON_STEPS = 50
NEWTON_TOL = 1e-10


def design(x, n):
    return np.column_stack([np.ones(n), x]) if x.size else np.ones((n, 1))


def fit(x, y):
    x = design(x, len(y))
    w = np.zeros(x.shape[1])
    for _ in range(NEWTON_STEPS):
        p = 1 / (1 + np.exp(-x @ w))
        grad = x.T @ (y - p)
        hess = (x * (p * (1 - p))[:, None]).T @ x
        step = np.linalg.solve(hess, grad)
        w = w + step
        if np.abs(step).max() < NEWTON_TOL:
            break
    return w


def predict(w, x):
    return 1 / (1 + np.exp(-design(x, len(x)) @ w))


def loss(p, y):
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def population():
    g = pd.read_parquet(L.PERGAME)
    g = g[(g.game_type == "REG") & g.season.isin(SEASONS) & (g.margin_vs_open != 0)]
    g = g.dropna(subset=["out_of_season_home_probability", "home_covered"])
    mv = pd.read_parquet(OUT / "move_table.parquet")[["game_id", "d_hs", "cat"]]
    g = g.merge(mv, on="game_id", how="left").reset_index(drop=True)
    g["d_hs"] = g.d_hs.fillna(0.0)
    p = g.out_of_season_home_probability.astype(float)
    g["logit"] = np.log(p / (1 - p))
    g["direction"] = np.sign(g.d_hs)
    g["one_team_direction"] = g.direction * (g.cat == "one_team")
    return g


def interval(d, rng):
    m = d[rng.integers(0, len(d), size=(L.BOOT, len(d)))].mean(axis=1)
    return {
        "effect": float(d.mean()),
        "interval": [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))],
        "standard_error": float(m.std(ddof=1)),
        "probability_positive": float((m > 0).mean()),
    }


def run():
    g = population()
    y = g.home_covered.values.astype(float)
    base_p = g.out_of_season_home_probability.values.astype(float)
    season = g.season.values
    preds = {"b_served_base": base_p}
    coefs = {}
    ins = {}
    for arm, cols in ARMS.items():
        x = g[cols].values.astype(float) if cols else np.empty((len(g), 0))
        oos = np.zeros(len(g))
        coefs[arm] = {}
        for s in SEASONS:
            te = season == s
            w = fit(x[~te], y[~te])
            oos[te] = predict(w, x[te])
            coefs[arm][str(s)] = [float(v) for v in w]
        preds[arm] = oos
        wall = fit(x, y)
        ins[arm] = float(loss(predict(wall, x), y).mean())
        coefs[arm]["all"] = [float(v) for v in wall]
    rng = np.random.default_rng(L.SEED)
    res = {
        "family": FAMILY,
        "n": len(g),
        "n_moved_direction_nonzero": int((g.direction != 0).sum()),
        "n_one_team_direction_nonzero": int((g.one_team_direction != 0).sum()),
        "look_count": LOOKS,
        "arms": {},
        "coefficients": coefs,
    }
    base_side = base_p > 0.5
    for arm, p in preds.items():
        ll = loss(p, y)
        entry = {
            "log_loss": float(ll.mean()),
            "brier": float(((p - y) ** 2).mean()),
            "accuracy": float(((p > 0.5) == (y == 1)).mean()),
            "in_sample_log_loss": ins.get(arm),
        }
        if arm in ins:
            entry["oos_minus_in_sample_log_loss"] = entry["log_loss"] - ins[arm]
        if arm in VS_B:
            for ref in ("b_served_base", "b2_base_refit"):
                if ref == arm:
                    continue
                entry[f"log_loss_improvement_vs_{ref}"] = interval(loss(preds[ref], y) - ll, rng)
                entry[f"brier_improvement_vs_{ref}"] = interval(
                    ((preds[ref] - y) ** 2) - ((p - y) ** 2), rng
                )
            flip = (p > 0.5) != base_side
            entry["decisive_vs_served_base"] = {
                "flips": int(flip.sum()),
                "arm_right": int(((p > 0.5) == (y == 1))[flip].sum()),
                "base_right": int((base_side == (y == 1))[flip].sum()),
            }
            entry["by_season_log_loss_improvement_vs_b2"] = {
                str(s): float((loss(preds["b2_base_refit"], y) - ll)[season == s].mean())
                for s in SEASONS
            }
        res["arms"][arm] = entry
    (OUT / "unit2_term.json").write_text(json.dumps(res, indent=1))
    cells = []
    for arm in ("c_direction", "d_one_team_direction", "e_both"):
        e = res["arms"][arm]["log_loss_improvement_vs_b2_base_refit"]
        cells.append(
            {
                "name": f"mkt23_unit2_{arm}_vs_refit_base",
                "effect": e["effect"],
                "effect_units": "log_loss_improvement",
                "interval_low": e["interval"][0],
                "interval_high": e["interval"][1],
                "standard_error": e["standard_error"],
                "probability_positive": e["probability_positive"],
                "classification_evidence": (
                    f"LOSO 2023-25, n {len(g)}, {LOOKS} looks; no power control or "
                    "mechanism refutation; deadline-view term, unknown at the Tuesday pick"
                ),
            }
        )
    batch = {
        "classification": "unresolved_below_power",
        "league": "nfl",
        "season_start": 2023,
        "season_end": 2025,
        "family": FAMILY,
        "category": "market",
        "source": "artifacts/mkt23/unit2_term.json",
        "sample_games": len(g),
        "plain_summary": (
            "Whether the direction of the Tuesday-to-deadline spread move, alone or for "
            "one-team total moves, improves the out-of-season cover probability at the opener."
        ),
        "description": (
            f"LOSO logistic terms on base logit; log loss gain vs refit base; {LOOKS} looks"
        ),
        "cells": cells,
    }
    (OUT / "unit2_record_batch.json").write_text(json.dumps(batch, indent=1))
    print(json.dumps(res, indent=1))
    if "--record" in sys.argv:
        subprocess.run(
            [
                ".tools/uv.exe",
                "run",
                "--no-sync",
                "--no-cache",
                "nfl-ats",
                "weak-signals",
                "record",
                "--batch",
                str(OUT / "unit2_record_batch.json"),
            ],
            check=True,
            cwd=L.ROOT,
        )


if __name__ == "__main__":
    sys.exit(run())
