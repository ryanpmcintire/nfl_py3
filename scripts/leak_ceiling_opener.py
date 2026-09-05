"""How much room is left at the grade the pool actually settles on?

Predeclaration (written before this script produced any output)
--------------------------------------------------------------
``docs/leak_ceiling_control.md`` measured the deliberate-leak ceiling at
**55.57%** and drew the project's headline strategic conclusion from it: "the
gap between the honest 52.10% and the leaked 55.6% (~3.5 points) is the MAXIMUM
room any amount of additional pregame feature engineering could possibly buy."

That number is **close-graded**, on 2,075 games. The pool settles at the
**opener**, where the same model scores 53.36% and the played four-member
policy scores 55.42% in-sample. Reading the close-graded ceiling against an
opener-graded record is a grade mismatch -- exactly the error class that
produced a wrong recommendation earlier in this same session -- so the "~3.5
points of room" figure cannot be applied to what is actually played until the
ceiling is measured at the same grade.

This measures it. Same recipe as arm B of the close-graded control (full
weak_stack design, ridge alpha 10, in-sample fit on the ATS target, in-sample
residuals as the predictive distribution, forced pick at p >= 0.5), with one
substitution: the target and the grading line are the **Tuesday opener**
instead of the close.

Endpoints, frozen here:

1. the opener-graded leak ceiling on the paired opener population;
2. the same arm at alpha=1, matching the close-graded control's own
   shrinkage-sensitivity check;
3. the market-line-only leak arm, as the floor reference;
4. the honest references on the identical population -- the raw model's
   opener accuracy and the played four-member policy -- so the remaining
   headroom is a subtraction the reader can check rather than a claim.

**This tells us where to spend effort, not whether anything is real.** A small
remaining gap argues for spending on composition, late-week information and
Best-Pick selection rather than on more pregame features; a large one argues
the opposite. Nothing here is a signal, nothing is recorded as one, and no
rotation-registry window is spent.

The leak arms are contaminated BY CONSTRUCTION -- they fit on the outcomes they
are scored against. That is the point of a positive control, and they must
never be read as achievable.
"""

from __future__ import annotations

import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

from nfl_ats.margin import make_margin_estimator, margin_feature_columns  # noqa: E402
from nfl_ats.provenance import write_stamped_artifact  # noqa: E402

OPENER_ARCHIVE = REPO / "artifacts/opener_evaluation/20260819T174244Z/per_game.parquet"
FEATURE_TABLE = REPO / "data/processed/game_features_weak_stack.parquet"
OUTPUT_ROOT = REPO / "artifacts/leak_ceiling_opener"

RIDGE_ALPHA = 10.0
ALT_ALPHA = 1.0

#: Honest references on the SAME population, measured elsewhere this session.
RAW_MODEL_OPENER = 0.533599
PLAYED_UNION_OPENER = 0.554225


def leak_arm(frame: pd.DataFrame, columns: list[str], alpha: float) -> dict[str, Any]:
    """In-sample ridge fit on the OPENER-graded ATS target, then forced picks.

    Deliberately leaky: it is fitted on the very outcomes it is scored on, and
    its predictive distribution is the in-sample residual. This bounds the
    estimator class rather than describing anything achievable.
    """

    design = frame.loc[:, columns]
    target = frame["opener_ats_margin"].to_numpy(dtype=float)

    estimator = make_margin_estimator("ridge", ridge_alpha=alpha)
    estimator.fit(design, target)
    fitted = np.asarray(estimator.predict(design), dtype=float)
    residuals = target - fitted
    sigma = float(np.std(residuals[np.isfinite(residuals)], ddof=1))

    # Gaussian mapping, matching production's promoted probability read.
    from scipy.stats import norm

    probability = norm.cdf(fitted / sigma) if sigma > 0 else np.full(len(fitted), 0.5)
    actual = (target > 0).astype(int)
    correct = ((probability >= 0.5).astype(int) == actual).astype(float)

    per_season = [
        {
            "season": int(season),
            "games": len(group),
            "accuracy": float(group["_c"].mean()),
        }
        for season, group in frame.assign(_c=correct).groupby("season", sort=True)
    ]
    return {
        "alpha": alpha,
        "n_features": len(columns),
        "games": len(frame),
        "correct": int(correct.sum()),
        "accuracy": float(correct.mean()),
        "residual_sigma": sigma,
        "per_season": per_season,
    }


def main() -> None:
    archive = pd.read_parquet(OPENER_ARCHIVE)
    features = pd.read_parquet(FEATURE_TABLE)

    archive["game_id"] = archive["game_id"].astype(str)
    features["game_id"] = features["game_id"].astype(str)

    # The opener-graded ATS margin. `margin_vs_open` is the archive's own
    # field; recomputing it from result and the opener line and asserting they
    # agree means a schema change cannot silently redefine the target.
    archive["opener_ats_margin"] = archive["margin_vs_open"].astype(float)
    recomputed = archive["result"].astype(float) - archive["tue_open_home_spread"].astype(float)
    disagreement = float((archive["opener_ats_margin"] - recomputed).abs().max())
    if disagreement > 1e-9:
        raise SystemExit(f"margin_vs_open does not equal result - opener line (max {disagreement})")

    merged = archive.merge(features, on="game_id", how="inner", suffixes=("", "_feat"))
    # Pushes carry no forced-pick outcome and are excluded, matching every
    # other evaluation in this project.
    merged = merged.loc[merged["opener_ats_margin"] != 0.0].reset_index(drop=True)

    weak_stack_columns = [
        column
        for column in margin_feature_columns("market_residual", "weak_stack")
        if column in merged.columns
    ]
    market_columns = [c for c in ("spread_line", "total_line") if c in merged.columns]

    report: dict[str, Any] = {
        "computed_at_utc": datetime.now(UTC).isoformat(),
        "opener_archive": str(OPENER_ARCHIVE),
        "feature_table": str(FEATURE_TABLE),
        "games_after_push_drop": len(merged),
        "seasons": sorted(int(s) for s in merged["season"].unique()),
        "grade": "Tuesday opener (tue_open_home_spread), forced pick at p >= 0.5",
        "arms": {
            "market_line_leak": leak_arm(merged, market_columns, RIDGE_ALPHA),
            "weak_stack_leak": leak_arm(merged, weak_stack_columns, RIDGE_ALPHA),
            "weak_stack_leak_alpha1": leak_arm(merged, weak_stack_columns, ALT_ALPHA),
        },
        "honest_references_same_grade": {
            "raw_model_opener_probability_rule": RAW_MODEL_OPENER,
            "played_four_member_union_opener": PLAYED_UNION_OPENER,
            "note": (
                "Both measured elsewhere this session on the opener archive. The played "
                "union's 55.42% is itself selection-inflated (best of 127 subsets); its "
                "de-inflated planning estimate is about 54.6%."
            ),
        },
    }

    ceiling = report["arms"]["weak_stack_leak"]["accuracy"]
    report["headroom_accuracy_points"] = {
        "ceiling_minus_raw_model": (ceiling - RAW_MODEL_OPENER) * 100.0,
        "ceiling_minus_played_union_in_sample": (ceiling - PLAYED_UNION_OPENER) * 100.0,
        "ceiling_minus_played_union_deinflated": (ceiling - 0.546) * 100.0,
        "reading": (
            "The last figure is the honest one: it compares a leaked ceiling against a "
            "de-inflated expectation. Both are estimates and neither is a bound on a "
            "richer model class -- the close-graded control's own limitation note applies "
            "here unchanged (this bounds ridge on standardized linear designs, not "
            "nonlinear learners)."
        ),
    }

    out_dir = OUTPUT_ROOT / datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    out_dir.mkdir(parents=True, exist_ok=True)
    write_stamped_artifact(report, out_dir / "results.json")  # ENG-38

    print(json.dumps({k: v for k, v in report.items() if k != "arms"}, indent=2))
    for name, arm in report["arms"].items():
        print(
            f"{name:<26} alpha={arm['alpha']:<5} n_feat={arm['n_features']:<4} "
            f"accuracy={arm['accuracy']:.4%}"
        )
    print(f"\nWrote {out_dir}")


if __name__ == "__main__":
    main()
