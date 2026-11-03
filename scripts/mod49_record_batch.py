import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "artifacts" / "mod49" / "construction.json"
OUT = REPO / "artifacts" / "mod49" / "weak_signals_batch.json"
FAMILY = "mod49_mod41_construction"
PLAIN = {
    "V0_baseline": "Baseline construction replicated on the served base.",
    "W2": "Same term with a two-game recent window.",
    "W4": "Same term with a four-game recent window.",
    "W5": "Same term with a five-game recent window.",
    "OFF_only": "Same term counting only the team's own offensive plays.",
    "PRIOR_norm": "Same term normalised by last season's mean instead of the season so far.",
    "SEC": "Same term in seconds of possession instead of plays.",
    "PLACEBO_opp_opp": "Placebo: the load of the opponent's previous opponent.",
}


def main():
    d = json.loads(SRC.read_text())
    cells = []
    for name, r in d["variants"].items():
        ll = r["ll_gain"]
        a = r["acc_gain"]
        cells.append(
            {
                "name": f"mod49_{name}_log_loss_gain",
                "effect_units": "log_loss_improvement",
                "effect": ll["point"],
                "interval_low": ll["ci95"][0],
                "interval_high": ll["ci95"][1],
                "standard_error": (ll["ci95"][1] - ll["ci95"][0]) / 3.92,
                "probability_positive": ll["probability_positive"],
                "sample_games": r["usable"],
                "description": (
                    f"{PLAIN[name]} Served base, LOSO 2020-25, opener grade; decisive "
                    f"{r['decisive_term_correct']}-{r['decisive_base_correct']} "
                    f"of {r['decisive']}; "
                    f"accuracy gain {a['point']:.2f} pts, P+ {a['probability_positive']:.2f}; "
                    f"coef {r['coef_full'][-1]:.4f}; permutation p {r['null']['null_p_ge']:.3f}"
                ),
                "plain_summary": PLAIN[name],
            }
        )
    batch = {
        "league": "nfl",
        "season_start": 2020,
        "season_end": 2025,
        "category": "schedule",
        "sample_blocks": 6,
        "source": "artifacts/mod49/construction.json",
        "family": FAMILY,
        "classification": "unresolved_below_power",
        "classification_evidence": (
            "No positive control and no admissible closing ground; probability_positive from "
            "week-blocked bootstrap; 16 looks (8 log loss, 8 accuracy) in the construction family."
        ),
        "cells": cells,
    }
    OUT.write_text(json.dumps(batch, indent=2))


if __name__ == "__main__":
    main()
