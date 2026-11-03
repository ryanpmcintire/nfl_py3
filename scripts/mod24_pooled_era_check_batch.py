import json
from pathlib import Path

ROOT = Path("F:/Repos/nfl_py3")
REPORT = ROOT / "artifacts" / "pooled_screen" / "mod24_era_check.json"
BATCH = ROOT / "artifacts" / "pooled_screen" / "mod24_era_check_batch.json"
LEAGUE_SEASONS = (("nfl", (2020, 2025)), ("cfb", (2023, 2025)))
METRICS = (("accuracy", "accuracy_points"), ("log_loss", "log_loss_improvement"))

r = json.loads(REPORT.read_text())
cells = []
for arm, v in r["arms"].items():
    for lg, (s0, s1) in LEAGUE_SEASONS:
        for m, u in METRICS:
            e = v[lg][m]
            cells.append(
                {
                    "effect_units": u,
                    "league": lg,
                    "season_start": s0,
                    "season_end": s1,
                    "name": f"mod24_era_check_{arm}_{lg}_{m}",
                    "description": (
                        f"Arm {arm} on 14 shared features vs ridge alpha 10, "
                        f"{lg} {s0}-{s1} opener; {m}."
                    ),
                    "effect": e["effect"],
                    "interval_low": e["low"],
                    "interval_high": e["high"],
                    "probability_positive": e["p_pos"],
                    "standard_error": e["se"],
                    "plain_summary": (
                        "A training-side change on the shared inputs moved results against "
                        "the opener by an unresolved amount."
                    ),
                }
            )
b = {
    "source": "artifacts/pooled_screen/mod24_era_check.json",
    "classification": "unresolved_below_power",
    "family": "mod24_pooled_era_check",
    "category": "modeling",
    "classification_evidence": (
        "No positive control and no admissible closing ground; "
        "season-week blocked bootstrap interval reported."
    ),
    "notes": (
        "CFB espn_book_era 2023-25 against NFL 2020-25 Tuesday opener; "
        "6 arms x 2 leagues x 2 metrics = 24 cells."
    ),
    "cells": cells,
}
BATCH.write_text(json.dumps(b, indent=1))
