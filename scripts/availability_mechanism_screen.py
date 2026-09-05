"""Does the MOD-07 availability advantage sit where the mechanism says it must?

The +1.75-point availability/value half (``scripts/availability_ablation.py``)
is 456 games wide and its interval crosses zero, so it cannot be confirmed by
size. It CAN be checked for mechanism, and that check is free: it re-reads a
window that is already spent for ``mod07_weak_signal_stack`` and takes no new
look.

The two arms differ in exactly two ways, and nothing else:

* the candidate table carries **learned** availability semantics in the shared
  ``*_injury_*`` / ``*_qb_start_probability`` columns where the baseline carries
  hand-authored fixed priors, and
* the candidate feature set adds exactly two columns,
  ``diff_injury_skill_epa_value_lost`` and
  ``diff_injury_defense_disruption_value_lost``.

So for every game there is a measurable quantity: **how far the learned
semantics actually moved the model's inputs**. If the +1.75 points is the
availability mechanism working, the advantage must concentrate in the games the
learned numbers actually changed, and games where the two tables agree to
within rounding must contribute nothing -- they are the arms' own internal
placebo. If instead the advantage is flat across that axis, or lives in the
games where the inputs barely moved, then the mechanism is not what produced it
and the +1.75 is noise wearing the family's name.

This is a falsification test, not a confirmation: passing it does not promote
anything, and the prospective 2026 ledger remains the only clean evidence.

Usage::

    .\\.tools\\uv.exe run --no-sync python scripts/availability_mechanism_screen.py
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from nfl_ats.clv import week_blocked_bootstrap

REPO = Path(__file__).resolve().parents[1]

PAIRED = (
    REPO
    / "artifacts/availability_experiments"
    / "mod07_ablation_2020_2021.B_minus_A_availability_and_value.parquet"
)

# The columns whose SEMANTICS change between the two tables and which both arms
# read (they are in the `player` feature set, so the baseline sees them too).
SHARED_AVAILABILITY_COLUMNS = (
    "diff_injury_offense_unavailability",
    "diff_injury_defense_unavailability",
    "diff_injury_special_teams_unavailability",
    "diff_injury_offensive_line_unavailability",
    "diff_injury_skill_unavailability",
    "diff_injury_front_unavailability",
    "diff_injury_secondary_unavailability",
    "diff_qb_start_probability",
    "diff_qb_expected_epa_per_dropback",
)

# The two columns the candidate feature set ADDS. Their magnitude is a second,
# independent axis: a game with no injury value lost cannot be moved by them.
ADDED_VALUE_COLUMNS = (
    "diff_injury_skill_epa_value_lost",
    "diff_injury_defense_disruption_value_lost",
)


def input_movement(baseline_table: Path, candidate_table: Path, game_ids: pd.Index) -> pd.DataFrame:
    """Per-game magnitude of the input change the learned semantics produced."""

    left = pd.read_parquet(baseline_table).drop_duplicates("game_id").set_index("game_id")
    right = pd.read_parquet(candidate_table).drop_duplicates("game_id").set_index("game_id")
    index = game_ids.intersection(left.index).intersection(right.index)

    # Standardise each column by its own spread over the scored games so that
    # one large-scale column cannot dominate the norm.
    parts = []
    for column in SHARED_AVAILABILITY_COLUMNS:
        a = pd.to_numeric(left.loc[index, column], errors="coerce")
        b = pd.to_numeric(right.loc[index, column], errors="coerce")
        scale = float(pd.concat([a, b]).std(ddof=0)) or 1.0
        parts.append(((b - a) / scale).fillna(0.0))
    semantics_shift = pd.concat(parts, axis=1).pow(2).sum(axis=1).pow(0.5)

    value_parts = []
    for column in ADDED_VALUE_COLUMNS:
        b = pd.to_numeric(right.loc[index, column], errors="coerce")
        scale = float(b.std(ddof=0)) or 1.0
        value_parts.append((b / scale).fillna(0.0))
    value_magnitude = pd.concat(value_parts, axis=1).abs().sum(axis=1)

    return pd.DataFrame(
        {"semantics_shift": semantics_shift, "value_magnitude": value_magnitude}, index=index
    )


def _metric_fn(frame: pd.DataFrame) -> dict[str, float]:
    return {"right_minus_left": float(frame["right_correct"].mean() - frame["left_correct"].mean())}


def stratify(paired: pd.DataFrame, axis: str, *, bins: int, samples: int, seed: int) -> list[dict]:
    """Split the scored games into equal-count strata and score each one."""

    ranked = paired.sort_values(axis).reset_index(drop=True)
    ranked["stratum"] = pd.qcut(ranked[axis].rank(method="first"), bins, labels=False)
    rows = []
    for stratum, block in ranked.groupby("stratum"):
        delta = float(block["right_correct"].mean() - block["left_correct"].mean())
        entry: dict[str, Any] = {
            "axis": axis,
            "stratum": int(stratum),
            "games": len(block),
            f"{axis}_min": float(block[axis].min()),
            f"{axis}_max": float(block[axis].max()),
            "left_accuracy": float(block["left_correct"].mean()),
            "right_accuracy": float(block["right_correct"].mean()),
            "delta_points": delta * 100.0,
            "picks_disagreeing": int(block["left_pick_home"].ne(block["right_pick_home"]).sum()),
        }
        try:
            boot = week_blocked_bootstrap(
                block, _metric_fn, block="week", samples=samples, seed=seed
            )
            entry["probability_positive"] = float(boot.iloc[0]["probability_positive"])
        except Exception:  # pragma: no cover - thin strata have too few weeks
            entry["probability_positive"] = None
        rows.append(entry)
    return rows


def disagreement_contrast(paired: pd.DataFrame, axis: str) -> dict[str, Any]:
    """Are the games the arms SPLIT on the games the inputs actually moved?

    Every point of the delta lives in the disagreement games -- an agreed pick
    contributes zero by construction. So the sharpest mechanism question is
    whether disagreement is predicted by input movement at all.
    """

    disagree = paired["left_pick_home"].ne(paired["right_pick_home"])
    moved = paired.loc[disagree, axis]
    same = paired.loc[~disagree, axis]
    # Rank-biserial correlation from the Mann-Whitney U statistic: distribution
    # free, and the paired frame is small enough that a normal approximation on
    # the raw values would be the weaker choice.
    ranks = paired[axis].rank()
    n_moved, n_same = len(moved), len(same)
    total = n_moved + n_same
    u = float(ranks.loc[disagree].sum()) - n_moved * (n_moved + 1) / 2.0
    rank_biserial = 2.0 * u / (n_moved * n_same) - 1.0
    # Var(U) = n1 n2 (N+1)/12 under the null, so Var(r) = (N+1)/(3 n1 n2).
    # This statistic uses all 456 games and a continuous covariate, which is why
    # it resolves where the 456-game ACCURACY delta cannot.
    standard_error = float(np.sqrt((total + 1.0) / (3.0 * n_moved * n_same)))
    z = rank_biserial / standard_error
    return {
        "axis": axis,
        "disagreement_games": int(n_moved),
        "agreement_games": int(n_same),
        "mean_on_disagreements": float(moved.mean()),
        "mean_on_agreements": float(same.mean()),
        "rank_biserial_correlation": float(rank_biserial),
        "standard_error": standard_error,
        "interval_95": [
            float(rank_biserial - 1.96 * standard_error),
            float(rank_biserial + 1.96 * standard_error),
        ],
        "z": float(z),
        "two_sided_p": float(math.erfc(abs(z) / math.sqrt(2.0))),
        "reading": (
            "positive means the arms split precisely on the games the learned "
            "semantics moved, which is what the mechanism predicts"
        ),
    }


READ_ONLY_SCRIPT = True
# ENG-29: read-only with respect to artifacts/ and registry/; the ENG-29 scanner confirms its only
# write sites resolve to a caller-supplied `--output`/`--out` path with no artifacts/ or registry/
# default, never a governed tree by default.


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--baseline-features",
        type=Path,
        default=REPO / "data/processed/game_features_player.parquet",
    )
    parser.add_argument(
        "--candidate-features",
        type=Path,
        default=REPO / "data/processed/game_features_weak_stack.parquet",
    )
    parser.add_argument("--bins", type=int, default=3)
    parser.add_argument("--samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260818)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    paired = pd.read_parquet(PAIRED)
    movement = input_movement(
        args.baseline_features, args.candidate_features, pd.Index(paired["game_id"])
    )
    paired = paired.merge(movement, left_on="game_id", right_index=True, how="inner")

    # Placebo axes: the same tercile machinery on quantities with NO availability
    # content. Tercile spread on 456 games is large by itself, so the strata
    # table below means nothing without these beside it.
    features = (
        pd.read_parquet(args.candidate_features).drop_duplicates("game_id").set_index("game_id")
    )
    paired["abs_spread"] = (
        pd.to_numeric(features.loc[paired["game_id"], "spread_line"], errors="coerce")
        .abs()
        .to_numpy()
    )
    paired["total"] = pd.to_numeric(
        features.loc[paired["game_id"], "total_line"], errors="coerce"
    ).to_numpy()

    report = {
        "source": str(PAIRED.relative_to(REPO)),
        "window": [2020, 2021],
        "grade": "opener",
        "games": len(paired),
        "overall_delta_points": float(
            (paired["right_correct"].mean() - paired["left_correct"].mean()) * 100.0
        ),
        "registry_note": (
            "Re-read of a window already spent for mod07_weak_signal_stack. "
            "No new look, no registry write."
        ),
        "strata": (
            stratify(
                paired, "semantics_shift", bins=args.bins, samples=args.samples, seed=args.seed
            )
            + stratify(
                paired, "value_magnitude", bins=args.bins, samples=args.samples, seed=args.seed
            )
        ),
        "placebo_strata": [
            row
            for axis in ("abs_spread", "total", "week")
            for row in stratify(paired, axis, bins=args.bins, samples=args.samples, seed=args.seed)
        ],
        "placebo_correlations_with_value_magnitude": {
            axis: {
                "pearson": float(paired["value_magnitude"].corr(paired[axis])),
                "spearman": float(paired["value_magnitude"].corr(paired[axis], method="spearman")),
            }
            for axis in ("abs_spread", "total", "week", "semantics_shift")
        },
        "disagreement_contrasts": [
            disagreement_contrast(paired, "semantics_shift"),
            disagreement_contrast(paired, "value_magnitude"),
        ],
    }
    print(json.dumps(report, indent=2))
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
