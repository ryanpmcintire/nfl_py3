import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "artifacts/xlg15/report.json"
BATCH = ROOT / "artifacts/xlg15/batch.json"
ERA = {"2014-2019": (2014, 2019), "2023-2025": (2023, 2025)}

r = json.loads(REPORT.read_text())
cells = []
for trait, eras in r["reliability"].items():
    for en, v in eras.items():
        s0, s1 = ERA[en]
        cells.append(
            {
                "effect_units": "correlation",
                "name": f"xlg15_cfb_reliability_{trait}_{en}",
                "description": f"CFB split-half reliability, {trait}, {en}.",
                "season_start": s0,
                "season_end": s1,
                "effect": v["r_half"],
                "interval_low": None,
                "interval_high": None,
                "reliability": v["sb"],
                "sample_blocks": v["n_units"],
                "plain_summary": "Split-half reliability of the trait on college play-by-play.",
            }
        )
for trait, eras in r["pricing"].items():
    rel = r["reliability"][trait]
    for en, v in eras.items():
        s0, s1 = ERA[en]
        cells.append(
            {
                "effect_units": "ats_points",
                "name": f"xlg15_cfb_pricing_{trait}_{en}",
                "description": f"CFB opener ATS margin on as-of {trait}, {en}.",
                "season_start": s0,
                "season_end": s1,
                "effect": v["slope_ats_points_per_sd"],
                "interval_low": v["ci"][0],
                "interval_high": v["ci"][1],
                "standard_error": v["se"],
                "probability_positive": v["p_pos"],
                "sample_games": v["n_games"],
                "reliability": rel[en]["sb"],
                "plain_summary": "Whether the college opener already prices this trait.",
            }
        )
cells = [{k: x for k, x in c.items() if x is not None} for c in cells]
b = {
    "source": "artifacts/xlg15/report.json",
    "classification": "unresolved_below_power",
    "league": "cfb",
    "family": "xlg15_cfb_trait_reliability",
    "category": "onfield",
    "notes": ("3 traits x 2 eras x (reliability, pricing) = 12 cells; CPOE not buildable."),
    "cells": cells,
}
BATCH.write_text(json.dumps(b, indent=1))
