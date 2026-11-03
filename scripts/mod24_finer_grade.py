from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

OPENER = Path("artifacts/opener_evaluation/20261101T151614Z/per_game.parquet")
OUT = Path("artifacts/mod24_finer_grade")
SEED = 20261103
DRAWS = 10_000
ARMS = {
    "u1_alpha_only": "mod23_unit1/alpha_only",
    "u1_greedy": "mod23_unit1/greedy",
    "u2_divergence": "mod23_unit2/divergence",
    "u2_full_strength": "mod23_unit2/full_strength",
    "u2_lineup_and_divergence": "mod23_unit2/lineup_and_divergence",
    "u2_lineup_total": "mod23_unit2/lineup_total",
    "u2_qb_out": "mod23_unit2/qb_out",
    "u2_qb_quality_loss": "mod23_unit2/qb_quality_loss",
    "u3_add_adjusted": "mod23_unit3/add_adjusted",
    "u3_compact_net": "mod23_unit3/compact_net",
    "u3_replace_adjusted": "mod23_unit3/replace_adjusted",
    "u4_blend_unit1_trimmed": "mod23_unit4/blend_unit1_trimmed",
    "u4_shrink_served": "mod23_unit4/shrink_served",
    "u5_man_zone": "mod23_unit5/arm_a_man_zone",
    "u5_coverage_type": "mod23_unit5/arm_b_coverage_type",
    "u5_both": "mod23_unit5/arm_c_both",
}


def frame(d):
    d = d.set_index("game_id") if "game_id" in d.columns else d
    d = d.loc[d.margin_vs_open.ne(0) & d.correct_at_open_probability_rule.notna()].copy()
    p = d.home_cover_probability_at_open.clip(1e-6, 1 - 1e-6)
    y = (d.margin_vs_open > 0).astype(float)
    err = d.margin_vs_open - d.residual_at_open
    return pd.DataFrame(
        {
            "season": d.season,
            "acc": d.correct_at_open_probability_rule,
            "ll": -(y * np.log(p) + (1 - y) * np.log(1 - p)),
            "sq": err**2,
            "ab": err.abs(),
        }
    )


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    base = frame(pd.read_parquet(OPENER).loc[lambda x: x.season.between(2020, 2025)])
    seasons = np.array(sorted(base.season.unique()))
    rng = np.random.default_rng(SEED)
    draws = rng.integers(0, len(seasons), size=(DRAWS, len(seasons)))
    rows = []
    for arm, path in ARMS.items():
        a = frame(pd.read_parquet(Path("artifacts") / f"{path}_per_game.parquet"))
        j = a.join(base, rsuffix="_b", how="inner")
        row = {"arm": arm, "n": len(j)}
        for m in ("acc", "ll", "sq", "ab"):
            diff = j[m + "_b"] - j[m] if m != "acc" else j[m] - j[m + "_b"]
            s = diff.groupby(j.season).agg(["sum", "count"]).reindex(seasons).fillna(0)
            num, den = s["sum"].to_numpy(), s["count"].to_numpy()
            boot = num[draws].sum(1) / den[draws].sum(1)
            row[m] = float(diff.mean())
            row[m + "_lo"], row[m + "_hi"] = np.percentile(boot, [2.5, 97.5]).tolist()
            row[m + "_pplus"] = float((boot > 0).mean())
            row[m + "_t"] = float(diff.mean() / (diff.std(ddof=1) / np.sqrt(len(diff))))
            row[m + "_seasons_pos"] = int(((s["sum"] / s["count"]) > 0).sum())
        dec = j.loc[j.acc.ne(j.acc_b)]
        row["decisive_w"] = int((dec.acc > dec.acc_b).sum())
        row["decisive_l"] = int((dec.acc < dec.acc_b).sum())
        rows.append(row)
    res = pd.DataFrame(rows)
    mult = {}
    for m in ("ll", "sq", "ab"):
        ratio = (res[m + "_t"] ** 2).mean() / (res["acc_t"] ** 2).mean()
        idx = np.random.default_rng(SEED).integers(0, len(res), size=(DRAWS, len(res)))
        t_f, t_b = (res[m + "_t"] ** 2).to_numpy(), (res["acc_t"] ** 2).to_numpy()
        boot = t_f[idx].mean(1) / t_b[idx].mean(1)
        mult[m] = {
            "multiplier_vs_accuracy": float(ratio),
            "arm_boot_lo": float(np.percentile(boot, 2.5)),
            "arm_boot_hi": float(np.percentile(boot, 97.5)),
            "sign_agree_with_acc": int((np.sign(res[m]) == np.sign(res["acc"])).sum()),
            "corr_t_with_acc_t": float(np.corrcoef(res[m + "_t"], res["acc_t"])[0, 1]),
        }
    res.to_csv(OUT / "results.csv", index=False)
    cols = ["acc", "ll", "sq", "ab"]
    payload = {"rows": rows, "multiplier": mult, "base_means": base[cols].mean().to_dict()}
    with open(OUT / "results.json", "w") as fh:
        json.dump(payload, fh, indent=1)
    cells = []
    for r in rows:
        cells.append(
            {
                "name": f"mod24_finer_{r['arm']}_margin_sq_error_improvement",
                "description": f"MOD-23 arm {r['arm']} vs served base at Tuesday opener 2020-2025: paired squared error of the arm's own margin forecast vs margin_vs_open (positive = arm better); season-blocked bootstrap.",  # noqa: E501
                "source": "artifacts/mod24_finer_grade/results.json",
                "effect": r["sq"],
                "effect_units": "margin_sq_error_improvement",
                "interval_low": r["sq_lo"],
                "interval_high": r["sq_hi"],
                "probability_positive": r["sq_pplus"],
                "sample_games": r["n"],
                "sample_blocks": 6,
                "classification": "unresolved_below_power",
                "classification_evidence": "Six season blocks; no positive control of this size; not a refuted mechanism.",  # noqa: E501
                "category": "modeling",
                "league": "nfl",
                "season_start": 2020,
                "season_end": 2025,
                "family": "mod24_finer_grade",
                "plain_summary": "Whether this variant forecasts the game margin more accurately than the base model at the opener.",  # noqa: E501
                "notes": "Look of 32 (16 arms x squared and absolute error); absolute error in results.json.",  # noqa: E501
            }
        )
    with open(OUT / "weak_signals_batch.json", "w") as fh:
        json.dump({"cells": cells}, fh, indent=1)
    pd.set_option("display.width", 250)
    print(json.dumps(mult, indent=1))


main()
