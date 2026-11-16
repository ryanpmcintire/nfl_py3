from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import v4_candidate as v4
from mod27_fenced_repairs_replay import FOUR, Fit, log_loss, paired, record

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "artifacts/opener_parity"
OUT = ROOT / "artifacts/two_base_stage"
FAMILY = "two_base_stage_2026_11_16"
REFERENCE_RECORD = "836-667"
OP_TERM = "op_model_logit"
FIVE = ("model_logit", OP_TERM, *FOUR[1:])


def base_probability(name, ids):
    raw = pd.read_parquet(SOURCE / f"opener_{name}.parquet")
    return raw["home_cover_probability_at_open"].reindex(ids).to_numpy(float)


def two_base(frame):
    p = pd.Series(np.nan, index=frame.index)
    folds = {}
    for season in sorted(frame["season"].unique()):
        fit = Fit(FIVE, frame[frame["season"] != season])
        held = frame["season"] == season
        p.loc[held] = fit.predict(frame[held])
        folds[str(int(season))] = fit.natural()
    full = Fit(FIVE, frame)
    return p.to_numpy(float), folds, full, full.predict(frame)


def classify(gain):
    if gain["hi"] < 0:
        return {
            "classification": "refuted_mechanism",
            "closing_ground": "wrong_sign_resolved",
            "classification_evidence": "Season-block interval entirely below zero",
        }
    return {
        "classification": "unresolved_below_power",
        "classification_evidence": (
            "Interval and probability_positive reported; "
            "no resolved wrong sign and no positive control bound"
        ),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Grade the four-term stage with QW and OP base logits as separate terms."
    )
    parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    pop = pd.read_parquet(v4.PICK / "per_game.parquet")
    pop["game_id"] = pop["game_id"].astype(str)
    pop = pop.reset_index(drop=True)
    ids = pop["game_id"].to_numpy()
    y = pop["home_covered"].to_numpy(float)
    qw = base_probability("QW", ids)
    op = base_probability("OP", ids)
    ref, ref_folds, ref_full, ref_in = v4.stage_two(pop, qw)
    if record(ref, y) != REFERENCE_RECORD:
        print(json.dumps({"error": "reference not reproduced", "record": record(ref, y)}))
        sys.exit(2)
    frame = pop.copy()
    frame["model_logit"] = v4.logit(qw)
    frame[OP_TERM] = v4.logit(op)
    arm, folds, full, arm_in = two_base(frame)
    blocks = pop.assign(week_block=pop["season"] * 100 + pop["week"])
    season_gain = paired(blocks, ref, arm, "season")
    week_gain = paired(blocks, ref, arm, "week_block")
    changed = (arm >= 0.5) != (ref >= 0.5)
    lq, lo = frame["model_logit"].to_numpy(float), frame[OP_TERM].to_numpy(float)
    corr_season = {
        str(int(s)): float(np.corrcoef(lq[m], lo[m])[0, 1])
        for s in sorted(pop["season"].unique())
        for m in [(pop["season"] == s).to_numpy()]
    }
    ll_arm, ll_arm_in = float(log_loss(arm, y).mean()), float(log_loss(arm_in, y).mean())
    ll_ref, ll_ref_in = float(log_loss(ref, y).mean()), float(log_loss(ref_in, y).mean())
    result = {
        "games": len(pop),
        "missing_model": {"QW": int(np.isnan(qw).sum()), "OP": int(np.isnan(op).sum())},
        "model_only_record": {"QW": record(qw, y), "OP": record(op, y)},
        "logit_correlation": float(np.corrcoef(lq, lo)[0, 1]),
        "logit_correlation_by_season": corr_season,
        "QW": {
            "record": record(ref, y),
            "in_sample_record": record(ref_in, y),
            "log_loss": ll_ref,
            "in_sample_log_loss": ll_ref_in,
            "held_out_minus_in_sample_log_loss": ll_ref - ll_ref_in,
            "model_logit_by_fold": {k: v["model_logit"] for k, v in ref_folds.items()},
            "full_coefficients": ref_full.natural(),
            "per_season": v4.season_table(pop, ref),
        },
        "TB": {
            "record": record(arm, y),
            "in_sample_record": record(arm_in, y),
            "log_loss": ll_arm,
            "in_sample_log_loss": ll_arm_in,
            "held_out_minus_in_sample_log_loss": ll_arm - ll_arm_in,
            "brier": float(((arm - y) ** 2).mean()),
            "fold_coefficients": folds,
            "full_coefficients": full.natural(),
            "per_season": v4.season_table(pop, arm),
            "vs_QW": {
                "season_block": season_gain,
                "week_block": week_gain,
                "changed_picks": int(changed.sum()),
                "changed_record_arm": record(arm[changed], y[changed]) if changed.any() else "0-0",
                "changed_record_reference": record(ref[changed], y[changed])
                if changed.any()
                else "0-0",
            },
        },
    }
    (OUT / "results.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    vs = result["TB"]["vs_QW"]
    base_text = (
        "Four-term pick probability with both base logits as separate fitted terms "
        "(QW base trained at the close, OP base trained with spread and target "
        "at the as-of opener), minus the QW four-term stage; refit leave-one-season-out, "
        f"opener grade 2020-2025, {len(pop)} games, season-block bootstrap "
        f"(week-block P+ {{week:.3f}}); record {result['TB']['record']} vs "
        f"{result['QW']['record']}; changed picks {vs['changed_picks']} "
        f"(arm {vs['changed_record_arm']}); logit correlation {result['logit_correlation']:.3f}"
    )
    cells = []
    for units, key in (
        ("accuracy_points", "accuracy_points_gain"),
        ("log_loss_improvement", "log_loss_gain"),
    ):
        g = season_gain[key]
        cells.append(
            {
                "name": f"two_base_stage_tb_{units}",
                "description": base_text.format(week=week_gain[key]["probability_positive"]),
                "effect": g["point"],
                "effect_units": units,
                "interval_low": g["lo"],
                "interval_high": g["hi"],
                "probability_positive": g["probability_positive"],
                "plain_summary": (
                    "Two-base stage: the close-trained and opener-trained model readings "
                    "enter the pick probability as two separate fitted terms."
                ),
                **classify(g),
            }
        )
    batch = {
        "family": FAMILY,
        "source": "artifacts/two_base_stage/results.json",
        "sample_games": len(pop),
        "season_start": 2020,
        "season_end": 2025,
        "league": "nfl",
        "category": "modeling",
        "cells": cells,
    }
    (OUT / "cells.json").write_text(json.dumps(batch, indent=2), encoding="utf-8")
    summary = {
        k: result[k] for k in ("games", "missing_model", "model_only_record", "logit_correlation")
    }
    summary["QW"] = {
        k: result["QW"][k]
        for k in (
            "record",
            "in_sample_record",
            "log_loss",
            "in_sample_log_loss",
            "model_logit_by_fold",
        )
    }
    summary["TB"] = {
        k: result["TB"][k]
        for k in (
            "record",
            "in_sample_record",
            "log_loss",
            "in_sample_log_loss",
            "held_out_minus_in_sample_log_loss",
            "vs_QW",
        )
    }
    summary["TB"]["model_terms_by_fold"] = {
        s: {"QW": c["model_logit"], "OP": c[OP_TERM]} for s, c in folds.items()
    }
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
