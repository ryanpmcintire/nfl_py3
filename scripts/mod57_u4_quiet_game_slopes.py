from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from lead98_price_skew import BASE, fit, loso, metrics, predict, reliability, summary
from mod57_u3_deadline_fair_line import block_bootstrap, decisive
from per_flag_weights_eval import SEED

from nfl_ats.pick_probability import MOVE_AVAILABLE_COLUMN, MOVE_COLUMN

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "artifacts/pick_probability/20261111T011859Z/per_game.parquet"
OUT = ROOT / "tests/scratch/mod57/u4"
SMALL_MOVE = 0.5
LATTICE = [0.0, 0.25, 0.5, 0.75, 1.0]
FAMILY = "mod57_quiet_game_slopes"
MOVE_SMALL = "move_small"
MOVE_LARGE = "move_large"
MODEL_QUIET = "model_quiet"
MODEL_LARGE = "model_large"
MODEL_NOMOVE = "model_nomove"


def lattice_table(absmove):
    rows = {}
    for edge in LATTICE:
        rows[f"{edge}"] = float(np.isclose(absmove, edge).mean())
    rows[f"more_than_{LATTICE[-1]}"] = float((absmove > LATTICE[-1] + 1e-9).mean())
    return rows


def add_regimes(data):
    avail = data[MOVE_AVAILABLE_COLUMN].astype(float) > 0
    absmove = data[MOVE_COLUMN].abs()
    small = avail & (absmove <= SMALL_MOVE + 1e-9)
    large = avail & (absmove > SMALL_MOVE + 1e-9)
    data = data.copy()
    data[MOVE_SMALL] = data[MOVE_COLUMN].where(small)
    data[MOVE_LARGE] = data[MOVE_COLUMN].where(large)
    data[MODEL_QUIET] = data["model_logit"].where(small)
    data[MODEL_LARGE] = data["model_logit"].where(large)
    data[MODEL_NOMOVE] = data["model_logit"].where(~avail)
    return data, avail, small, large


def to_improvement(entry, name):
    d = entry[name]
    if name == "accuracy_points":
        return d["delta_mean"], d["ci95"][0], d["ci95"][1], d["probability_positive"]
    return -d["delta_mean"], -d["ci95"][1], -d["ci95"][0], d["probability_positive"]


def main():
    data = pd.read_parquet(ARTIFACT)
    data = data.loc[data["season"].between(2020, 2025)].reset_index(drop=True)
    data, avail, small, large = add_regimes(data)
    y = data["home_covered"].to_numpy(float)
    result = {"games": len(data)}
    absmove = data.loc[avail, MOVE_COLUMN].abs()
    result["move_available_games"] = int(avail.sum())
    result["abs_move_lattice_share_of_available"] = lattice_table(absmove)
    result["small_games"] = int(small.sum())
    result["large_games"] = int(large.sum())
    sample = data.loc[avail & data["open_move"].notna(), [MOVE_COLUMN, "open_move"]]
    result["sign_check"] = {
        "corr_move_toward_home_vs_open_move": float(sample[MOVE_COLUMN].corr(sample["open_move"])),
        "same_sign_share_when_both_nonzero": float(
            (
                np.sign(sample[MOVE_COLUMN])[(sample != 0).all(axis=1)]
                == np.sign(sample["open_move"])[(sample != 0).all(axis=1)]
            ).mean()
        ),
        "examples": data.loc[avail, ["game_id", "tue_open_home_spread", "close_home_spread"]]
        .assign(move_toward_home=data[MOVE_COLUMN], open_move=data["open_move"])
        .head(4)
        .to_dict("records"),
    }
    print(json.dumps(result, indent=1, default=str), flush=True)
    variants = {
        "four_term": BASE,
        "arm_a_move_slope_by_size": (
            "model_logit",
            BASE[1],
            MOVE_SMALL,
            MOVE_LARGE,
            MOVE_AVAILABLE_COLUMN,
        ),
        "arm_m_model_slope_by_state": (
            MODEL_QUIET,
            MODEL_LARGE,
            MODEL_NOMOVE,
            *BASE[1:],
        ),
        "model_only": ("model_logit",),
        "intercept_only": (),
    }
    probs, folds, insample = {}, {}, {}
    for name, feats in variants.items():
        oos, fo = loso(data, feats)
        probs[name] = oos
        folds[name] = fo
        b, m, s = fit(data, feats)
        insample[name] = predict(data, feats, b, m, s)
    probs["market_half"] = np.full(len(data), 0.5)
    open_move = data["open_move"].abs()
    populations = {
        "all_games": np.ones(len(data), dtype=bool),
        "quiet_open_move_half_point_or_less": (open_move <= SMALL_MOVE + 1e-9).to_numpy(),
    }
    rng = np.random.default_rng(SEED)
    result["populations"] = {}
    result["fold_coefficients"] = {
        n: folds[n] for n in ("four_term", "arm_a_move_slope_by_size", "arm_m_model_slope_by_state")
    }
    result["fold_summary"] = {n: summary(folds[n]) for n in result["fold_coefficients"]}
    arms = ("arm_a_move_slope_by_size", "arm_m_model_slope_by_state")
    cells = []
    for pop, mask in populations.items():
        sub = data.loc[mask].reset_index(drop=True)
        ys = y[mask]
        entry = {"games": int(mask.sum()), "variants": {}, "paired": {}, "decisive": {}}
        for name in probs:
            entry["variants"][name] = {"oos": metrics(probs[name][mask], ys)}
            if name in insample:
                entry["variants"][name]["in_sample"] = metrics(insample[name][mask], ys)
                entry["variants"][name]["accuracy_gap_in_minus_oos_points"] = 100.0 * (
                    entry["variants"][name]["in_sample"]["accuracy"]
                    - entry["variants"][name]["oos"]["accuracy"]
                )
        for arm in arms:
            entry["decisive"][arm] = decisive(probs[arm][mask], probs["four_term"][mask], ys)
            boot = block_bootstrap(sub, probs[arm][mask], probs["four_term"][mask], rng)
            entry["paired"][arm] = boot
            for metric, units in (
                ("accuracy_points", "accuracy_points"),
                ("log_loss", "log_loss_improvement"),
                ("brier", "brier_improvement"),
            ):
                eff, lo, hi, pp = to_improvement(boot, metric)
                cells.append(
                    {
                        "name": f"mod57_u4_{arm}_{pop}_{metric}",
                        "description": (
                            f"MOD-57 U4: {arm} versus served four-term, leave-one-season-out "
                            f"2020-25, {pop}, week-block bootstrap; improvement positive."
                        ),
                        "effect": eff,
                        "effect_units": units,
                        "interval_low": lo,
                        "interval_high": hi,
                        "probability_positive": pp,
                        "season_start": 2020,
                        "season_end": 2025,
                        "sample_games": int(mask.sum()),
                    }
                )
        entry["reliability"] = {
            n: reliability(probs[n][mask], ys) for n in ("four_term", *arms, "model_only")
        }
        result["populations"][pop] = entry
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "results.json").write_text(json.dumps(result, indent=1, default=str), encoding="utf-8")
    batch = {
        "family": FAMILY,
        "league": "nfl",
        "category": "market",
        "classification": "unresolved_below_power",
        "source": "tests/scratch/mod57/u4/results.json",
        "cells": cells,
    }
    (OUT / "batch.json").write_text(json.dumps(batch, indent=1), encoding="utf-8")
    print(json.dumps(result["populations"], indent=1, default=str))
    print(json.dumps(result["fold_summary"], indent=1))


if __name__ == "__main__":
    main()
