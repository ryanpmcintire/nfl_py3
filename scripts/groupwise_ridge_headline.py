"""The one comparison the block sweep does not make: best arm vs the FROZEN config.

``groupwise_ridge_screen.py`` compares every block arm against ``uniform`` at the
SAME global alpha, which is what isolates the block allocation. But the
practically meaningful question for a deployment decision is different: how does
the best configuration compare against what is actually running today
(``uniform`` at the frozen alpha = 10)? That crosses both axes at once, so it is
reported separately rather than folded into the screen.

Free under rotation rule 8 (CFB benchmark, unreserved).

Run:  .\\.tools\\uv.exe run --no-sync python scripts/groupwise_ridge_headline.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

# Sibling module: Python puts this script's own directory on sys.path, so the
# screen imports by bare name rather than through a `scripts` package (there is
# none -- pyproject packages only `src/nfl_ats`).
from groupwise_ridge_screen import (
    BASELINE_ARM,
    FEATURES_PATH,
    arm_summary,
    paired_bootstrap,
    pick_flips,
    run_arm,
)

from nfl_ats.cfb_benchmark import CFB_BENCHMARK_RIDGE_ALPHA

# The best-performing block arm from the predeclared grid, at the penalty level
# where the penalty is not inert. Named here so the comparison is explicit about
# being a post-hoc selection out of 21 configurations.
CANDIDATE_ARM = "market_light_10"
CANDIDATE_ALPHA = 10_000.0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/groupwise_ridge"))
    args = parser.parse_args()

    features = pd.read_parquet(FEATURES_PATH)
    features["gameday"] = pd.to_datetime(features["gameday"], errors="raise")
    completed = features.loc[
        pd.to_numeric(features["result"], errors="coerce").notna()
        & pd.to_numeric(features["ats_margin"], errors="coerce").notna()
    ].copy()
    completed = completed.sort_values(["gameday", "game_id"]).reset_index(drop=True)

    frozen = run_arm(completed, BASELINE_ARM, CFB_BENCHMARK_RIDGE_ALPHA)
    candidate = run_arm(completed, CANDIDATE_ARM, CANDIDATE_ALPHA)

    rows = []
    for window in ("clean_core", "all"):
        rows.append(arm_summary(frozen, window))
        rows.append(arm_summary(candidate, window))
    summary = pd.DataFrame(rows)

    comparisons = []
    for window in ("clean_core", "all"):
        for block in ("week", "season"):
            comparisons.extend(paired_bootstrap(frozen, candidate, window=window, block=block))
    comparison = pd.DataFrame(comparisons)
    flips = pd.DataFrame([pick_flips(frozen, candidate)])

    pd.set_option("display.width", 220)
    print("=== frozen config vs best block arm ===")
    print(summary.to_string(index=False))
    print("\n=== paired blocked comparison (candidate minus frozen) ===")
    print(comparison.to_string(index=False))
    print("\n=== pick flips ===")
    print(flips.to_string(index=False))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.output_dir / "headline_summary.csv", index=False)
    comparison.to_csv(args.output_dir / "headline_comparison.csv", index=False)
    flips.to_csv(args.output_dir / "headline_flips.csv", index=False)
    frozen.to_parquet(args.output_dir / "predictions_frozen.parquet", index=False)
    candidate.to_parquet(args.output_dir / "predictions_candidate.parquet", index=False)
    print(f"\nWrote {args.output_dir}")


if __name__ == "__main__":
    main()
