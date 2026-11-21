import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
A = ROOT / "artifacts" / "xlg15"

wr = json.loads((A / "wind_reliability_enlarged.json").read_text())
eras = {k: tuple(v) for k, v in wr["eras"].items()}
by_outcome = {}
for key, v in wr["cells"].items():
    by_outcome.setdefault(key.split("|")[0], []).append(v["powered_zero"])
closes = {o: all(z) for o, z in by_outcome.items()}
cells = []
for key, v in wr["cells"].items():
    outcome, en, split = key.split("|")
    s0, s1 = eras[en]
    c = {
        "league": "cfb",
        "category": "environment",
        "effect_units": "correlation",
        "name": f"xlg15_cfb_wind_elasticity_{outcome}_{split}_{en}",
        "description": (
            f"CFB team wind elasticity of {outcome} per mph, "
            f"split-half r ({split}), enlarged era {en}."
        ),
        "season_start": s0,
        "season_end": s1,
        "effect": v["r_half"],
        "interval_low": v["r_ci"][0],
        "interval_high": v["r_ci"][1],
        "probability_positive": v["p_pos"],
        "sample_blocks": v["n_teams"],
        "sample_games": v["team_games"],
        "reliability": v["sb"],
        "plain_summary": (
            "Whether college teams keep the same response to wind "
            "from one half of their games to the other."
        ),
        "notes": (
            f"perm p {v['perm_p']:.3f}; r80 detectable {v['r80_detectable']}; "
            "realized station wind "
            "(no CFB forecast archive); Unit 3 amendment 1 enlarged eras."
        ),
    }
    if closes[outcome]:
        c["classification"] = "refuted_mechanism"
        c["closing_ground"] = "no_split_half_reliability"
        c["classification_evidence"] = (
            f"upper 97.5% r {v['r_ci'][1]:.3f} below r80 {v['r80_detectable']:.3f} "
            "(positive control, power 0.8) in all four split x era cells of this outcome "
            "(enlarged eras 2010-19, 2020-25)"
        )
    else:
        c["classification_evidence"] = (
            f"r {v['r_half']:.3f} [{v['r_ci'][0]:.3f}, {v['r_ci'][1]:.3f}], "
            f"r80 {v['r80_detectable']}; "
            "not every enlarged cell of this outcome excludes the detectable reliability"
        )
    cells.append(c)
batch = {
    "source": "artifacts/xlg15/wind_reliability_enlarged.json",
    "classification": "unresolved_below_power",
    "family": "xlg15_cfb_trait_reliability",
    "notes": (
        "Unit 3: 8 looks, CFB wind elasticity on enlarged eras "
        "(amendment 1 in lane, written before added seasons were fetched)."
    ),
    "cells": cells,
}
(A / "batch_u3.json").write_text(json.dumps(batch, indent=1))
print("cells", len(cells), "closes", closes)
for c in cells:
    print(c["name"], c.get("classification", batch["classification"]), round(c["reliability"], 3))
