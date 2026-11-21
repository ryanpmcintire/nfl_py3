import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
A = ROOT / "artifacts" / "xlg15"
ERA = {
    "2014-2019": (2014, 2019),
    "2023-2025": (2023, 2025),
    "2009-2019": (2009, 2019),
    "2020-2025": (2020, 2025),
}

wr = json.loads((A / "wind_reliability.json").read_text())
nf = json.loads((A / "nfl_confirm.json").read_text())
cells = []
by_outcome = {}
for key, v in wr["cells"].items():
    by_outcome.setdefault(key.split("|")[0], []).append(v["powered_zero"])
closes = {o: all(z) for o, z in by_outcome.items()}
for key, v in wr["cells"].items():
    outcome, en, split = key.split("|")
    s0, s1 = ERA[en]
    c = {
        "league": "cfb",
        "category": "environment",
        "effect_units": "correlation",
        "name": f"xlg15_cfb_wind_elasticity_{outcome}_{split}_{en}",
        "description": f"CFB team wind elasticity of {outcome} per mph, split-half r ({split}), {en}.",
        "season_start": s0,
        "season_end": s1,
        "effect": v["r_half"],
        "interval_low": v["r_ci"][0],
        "interval_high": v["r_ci"][1],
        "probability_positive": v["p_pos"],
        "sample_blocks": v["n_teams"],
        "sample_games": v["team_games"],
        "reliability": v["sb"],
        "plain_summary": "Whether college teams keep the same response to wind from one half of their games to the other.",
        "notes": f"perm p {v['perm_p']:.3f}; r80 detectable {v['r80_detectable']}; realized station wind (no CFB forecast archive).",
    }
    if closes[outcome]:
        c["classification"] = "refuted_mechanism"
        c["closing_ground"] = "no_split_half_reliability"
        c["classification_evidence"] = (
            f"upper 97.5% r {v['r_ci'][1]:.3f} below r80 {v['r80_detectable']} (positive control, power 0.8) "
            "in all four split x era cells of this outcome"
        )
    else:
        c["classification_evidence"] = (
            f"r {v['r_half']:.3f} [{v['r_ci'][0]:.3f}, {v['r_ci'][1]:.3f}], r80 {v['r80_detectable']}; "
            "not every cell of this outcome excludes the detectable reliability"
        )
    cells.append(c)
for trait, eras in nf["nfl_reliability"].items():
    for en, v in eras.items():
        s0, s1 = ERA[en]
        cells.append(
            {
                "league": "nfl",
                "category": "onfield",
                "effect_units": "correlation",
                "name": f"xlg15_nfl_reliability_{trait}_{en}",
                "description": f"NFL split-half reliability (alternating games, team-season), {trait}, {en}; CFB construction.",
                "season_start": s0,
                "season_end": s1,
                "effect": v["r_half"],
                "interval_low": v["r_ci"][0],
                "interval_high": v["r_ci"][1],
                "probability_positive": v["p_pos"],
                "sample_blocks": v["n_units"],
                "reliability": v["sb"],
                "classification_evidence": f"reliable trait (perm p {v['perm_p']:.4f}); reliability is not a pick effect",
                "plain_summary": "Whether NFL teams keep this habit from one half of their games to the other.",
            }
        )
for trait, v in nf["nfl_opener"].items():
    g = v["ll_gain_vs_icpt"]
    cells.append(
        {
            "league": "nfl",
            "category": "onfield",
            "effect_units": "log_loss_improvement",
            "name": f"xlg15_nfl_opener_addon_{trait}",
            "description": f"NFL served-base LOSO add-on of as-of home-minus-away {trait}; log-loss gain vs intercept-only, 2020-2025.",
            "season_start": 2020,
            "season_end": 2025,
            "effect": g["mean"],
            "interval_low": g["ci"][0],
            "interval_high": g["ci"][1],
            "probability_positive": g["p_pos"],
            "sample_games": v["n_games"],
            "sample_blocks": len(v["coef_by_fold"]),
            "reliability": nf["nfl_reliability"][trait]["2020-2025"]["sb"],
            "classification_evidence": (
                f"matched-null percentile {v['null_percentile']:.1f}; decisive flips {v['flip_arm_correct']}-"
                f"{v['decisive_flips'] - v['flip_arm_correct']}; sign not pre-declared; below the add-one noise floor"
            ),
            "plain_summary": "Whether the habit adds anything to the served pick probability at the opener.",
            "notes": f"fold coefs {v['coef_by_fold']}; acc base {v['acc_base']:.4f} arm {v['acc_arm']:.4f}",
        }
    )
batch = {
    "source": "artifacts/xlg15/wind_reliability.json; artifacts/xlg15/nfl_confirm.json",
    "classification": "unresolved_below_power",
    "family": "xlg15_cfb_trait_reliability",
    "notes": "Unit 2: 14 looks (8 CFB wind, 4 NFL reliability, 2 NFL opener add-ons), declared in lane before signs.",
    "cells": cells,
}
(A / "batch_u2.json").write_text(json.dumps(batch, indent=1))
print("cells", len(cells), "closes", closes)
