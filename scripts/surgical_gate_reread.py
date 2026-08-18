"""Free re-read: measure `f` under the surgical gate, WITHOUT scoring accuracy.

Research item: ``docs/surgical_injury.md``. Why this is admissible without a
rotation window: ``[2020, 2021]`` is spent for ``mod07_weak_signal_stack``
(``registry/rotation_registry.json``), and reconstructing an already-taken
look is precedented as free re-read by ``scripts/availability_ablation.py``
and ``scripts/availability_mechanism_screen.py`` (both explicit: "attribution
on data already looked at costs no window", quoting ``docs/pool_edge_plan.md``).
This script follows the identical pattern to reconstruct arms A and D from
``docs/injury_value_lost.md`` section 4 (the cleanest, semantics-confound-free
isolation of the value-lost candidate) and asserts the reproduction matches
that document's recorded numbers byte-for-byte before computing anything new.

What is NOT admissible, and what this script structurally cannot produce: a
gated variant is a NEW variant (task brief, binding), so its ACCURACY may not
be measured on this spent window and reported as evidence -- that would be
indistinguishable from iterating a new candidate until it wins on a window
already used to select this family, which ``docs/mod07_stack.md`` and
``docs/availability_confirmation.md`` both already rule out. This script
therefore drops every correctness/outcome column from both arms IMMEDIATELY
after the byte-exact reproduction check (which only confirms already-published
numbers, nothing new) and never reloads them. Every computation after that
point uses ONLY: which side each arm picked (``pick_home_at_open``, a
pregame-timestamped boolean) and the pregame injury-magnitude covariate. No
accuracy, no `probability_positive`, no delta-points number for the gated
variant exists anywhere in this script's output, by construction, not by
omission.

What this measures instead: `f` (the fraction of the 456 games on which the
gated candidate's forced pick differs from the baseline's), before and after
gating, plus the implied MDE80 change -- purely mechanical properties of the
PICK SET, which is what the task brief asks this step to produce.

Usage::

    .\\.tools\\uv.exe run --no-sync python scripts/surgical_gate_reread.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

import availability_ablation as abl  # noqa: E402

from nfl_ats.estimation_variance import mde80, picks_differ_fraction  # noqa: E402
from nfl_ats.surgical_gating import (  # noqa: E402
    VALUE_LOST_MAGNITUDE_THRESHOLD,
    gate_active_fraction,
    gate_by_value_lost_magnitude,
    raw_value_magnitude,
)

# The reproduction target: docs/injury_value_lost.md section 4's D-A contrast,
# already independently reproduced once this session
# (scratchpad clean_value_lost_arm_report.json) and matching the document
# exactly. Re-asserted here so this script is self-checking on its own.
RECORDED = {
    "paired_games": 456,
    "weeks": 35,
    "left_accuracy": 0.5131578947368421,
    "right_accuracy": 0.5263157894736842,
    "delta_points": 1.3157894736842035,
    "probability_positive": 0.8875,
    "picks_disagreeing": 26,
}


def main() -> None:
    player_features = pd.read_parquet(REPO / "data/processed/game_features_player.parquet")
    player_value_features = pd.read_parquet(
        REPO / "data/processed/game_features_player_value.parquet"
    )
    market_root = REPO / "data/market/raw"
    seasons = (2020, 2021)
    min_train_games = 500
    samples = 2000
    seed = 20260818

    arm_a = abl.arm(
        player_features,
        seasons=seasons,
        profile="player",
        market_root=market_root,
        min_train_games=min_train_games,
    )
    arm_d = abl.arm(
        player_value_features,
        seasons=seasons,
        profile="player_value",
        market_root=market_root,
        min_train_games=min_train_games,
    )
    reproduction = abl.contrast(
        arm_a, arm_d, name="D_minus_A_value_lost_only", samples=samples, seed=seed
    )
    checks = {
        "paired_games_match": reproduction["paired_games"] == RECORDED["paired_games"],
        "weeks_match": reproduction["weeks"] == RECORDED["weeks"],
        "left_accuracy_match": abs(reproduction["left_accuracy"] - RECORDED["left_accuracy"])
        < 1e-12,
        "right_accuracy_match": abs(reproduction["right_accuracy"] - RECORDED["right_accuracy"])
        < 1e-12,
        "delta_match": abs(reproduction["delta_points"] - RECORDED["delta_points"]) < 1e-9,
        "probability_positive_match": abs(
            reproduction["probability_positive"] - RECORDED["probability_positive"]
        )
        < 1e-9,
        "disagreements_match": reproduction["picks_disagreeing"] == RECORDED["picks_disagreeing"],
    }
    if not all(checks.values()):
        raise SystemExit(f"Reproduction of docs/injury_value_lost.md section 4 FAILED: {checks}")

    # --- Reproduction confirmed against already-published numbers. Now drop
    # every correctness/outcome column and never touch one again: everything
    # below reads only picks (pregame-timestamped) and the injury covariate. --
    paired = abl.paired_frame(arm_a, arm_d)[["game_id", "left_pick_home", "right_pick_home"]].copy()
    del arm_a, arm_d  # both carried correct_at_open; do not let it leak into scope below

    magnitude = raw_value_magnitude(player_value_features.set_index("game_id"))
    paired = paired.merge(
        magnitude.rename("value_magnitude"), left_on="game_id", right_index=True, how="inner"
    )
    if len(paired) != RECORDED["paired_games"]:
        raise SystemExit("Magnitude join dropped games; window mismatch")

    baseline_pick = paired["left_pick_home"].to_numpy(dtype=float)
    candidate_pick = paired["right_pick_home"].to_numpy(dtype=float)
    mag = paired["value_magnitude"].to_numpy(dtype=float)

    f_ungated = picks_differ_fraction(baseline_pick, candidate_pick)
    gated_pick = gate_by_value_lost_magnitude(
        baseline_pick, candidate_pick, mag, threshold=VALUE_LOST_MAGNITUDE_THRESHOLD
    )
    f_gated = picks_differ_fraction(baseline_pick, gated_pick)
    n = len(paired)

    disagree_mask = paired["left_pick_home"].ne(paired["right_pick_home"])
    disagreements_retained = int((disagree_mask & (mag >= VALUE_LOST_MAGNITUDE_THRESHOLD)).sum())
    disagreements_dropped = int((disagree_mask & (mag < VALUE_LOST_MAGNITUDE_THRESHOLD)).sum())

    report: dict[str, Any] = {
        "note": (
            "Free re-read of the mod07_weak_signal_stack-spent [2020, 2021] window "
            "(docs/injury_value_lost.md section 4's D-A arms). Reproduction of the "
            "PUBLISHED accuracy numbers is asserted above and then every correctness "
            "column is discarded -- this report contains no accuracy, no delta-points, "
            "and no probability_positive for the GATED variant. It measures only which "
            "picks the gate changes, which is admissible per the task brief; scoring "
            "the gated variant's accuracy on this window is not, and is not done here."
        ),
        "window": list(seasons),
        "games": n,
        "reproduction_checks": checks,
        "threshold": VALUE_LOST_MAGNITUDE_THRESHOLD,
        "threshold_derivation": (
            "scripts/surgical_value_lost_distribution.py "
            "(conditional median of nonzero magnitude, full 2009-2025 history)"
        ),
        "gate_active_fraction_on_this_window": gate_active_fraction(
            mag, threshold=VALUE_LOST_MAGNITUDE_THRESHOLD
        ),
        "f_ungated": f_ungated,
        "f_gated": f_gated,
        "f_reduction_factor": (f_ungated / f_gated) if f_gated > 0 else float("inf"),
        "mde80_ungated": mde80(f_ungated, n),
        "mde80_gated": mde80(f_gated, n) if f_gated > 0 else 0.0,
        "disagreements_total": int(disagree_mask.sum()),
        "disagreements_retained_by_gate": disagreements_retained,
        "disagreements_dropped_by_gate": disagreements_dropped,
    }
    text = json.dumps(report, indent=2)
    print(text)


if __name__ == "__main__":
    main()
