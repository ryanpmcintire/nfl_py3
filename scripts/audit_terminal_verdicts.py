"""Re-read stored per-game artifacts and recompute paired statistics for
three TERMINAL verdicts flagged by the 2026-08-18 measurement-instrument
audit (``docs/revisit_list.md``, Tier 1).

This script performs NO new model fits, NO new windows, and NO calls into
``nfl_ats.rotation``. It only re-reads existing per-game prediction rows
already stored under ``artifacts/`` and recomputes, from first principles,
the paired statistics several registry entries never recorded:

- ``f`` -- the fraction of paired games on which the two compared arms
  picked different sides (the disagreement rate the project's own MDE80
  formula is built on, ``docs/estimation_variance.md`` sec 5:
  ``MDE80 = 280 * sqrt(f / n)``).
- The exact win/loss split on the disagreement games.
- The PAIRED standard error of the accuracy delta, ``100 * sqrt(f / n)``
  points -- not the UNPAIRED coin-flip sd ``100 * 0.5 / sqrt(n)`` that one
  registry entry's classification evidence used to dismiss a paired result.
- ``probability_positive`` via a normal approximation to the paired mean
  (``mean_delta / se``), for entries that never recorded one.
- The 95% paired-normal interval on the accuracy delta, as a cross-check
  against the block-bootstrap interval already stored in each artifact's
  own ``paired_comparisons.csv`` (which uses a *different*, resampling-based
  method; this script's interval is a closed-form Gaussian approximation
  using the same per-game disagreement structure MDE80 already assumes).

Three targets, each read from an existing artifact directory:

1. ``player_qb_continuity_matched_alpha`` / ``player_qb_continuity`` --
   ``artifacts/qb_continuity_replication/20260816T143913Z/predictions.parquet``.
   Three arms per game: ``candidate`` (qb-continuity features, ridge
   alpha=1), ``base_alpha1`` (no continuity features, ridge alpha=1),
   ``base_alpha10`` (no continuity features, ridge alpha=10, the production
   comparator). The MATCHED-ALPHA contrast is candidate vs base_alpha1; the
   ORIGINAL closed_negative contrast is candidate vs base_alpha10.
2. ``pbp_drive_bundle`` --
   ``artifacts/pbp_replication/20260816T142340Z/predictions.parquet``. Two
   arms per game distinguished by ``feature_profile`` (``base`` vs ``pbp``).

Usage::

    .tools/uv.exe run --no-sync python scripts/audit_terminal_verdicts.py
    .tools/uv.exe run --no-sync python scripts/audit_terminal_verdicts.py \
        --output <scratch>/audit_terminal_verdicts.json

Writes nothing under ``registry/`` or ``artifacts/``; only prints a report
table and optionally dumps the same numbers as JSON to ``--output``.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import pandas as pd
from scipy import stats

REPO_ROOT = Path(__file__).resolve().parent.parent

MDE80_Z = 2.8  # z_(1-alpha/2) + z_power = 1.96 + 0.84, two-sided alpha=0.05, 80% power


def paired_stats_from_picks(
    df: pd.DataFrame,
    game_col: str,
    outcome_col: str,
    prob_col: str,
    arm_col: str,
    arm_a: str,
    arm_b: str,
) -> dict[str, Any]:
    """Compute paired accuracy statistics between two arms on shared games.

    ``df`` must contain one row per (game, arm). Games with a missing
    ``outcome_col`` (pushes) are dropped before pairing, matching the
    project's existing ``paired_feature_comparisons`` convention.
    """
    a = df[df[arm_col] == arm_a].set_index(game_col)
    b = df[df[arm_col] == arm_b].set_index(game_col)
    shared = a.index.intersection(b.index)
    a = a.loc[shared]
    b = b.loc[shared]

    outcome = a[outcome_col]
    valid = outcome.notna() & b[outcome_col].notna()
    # Sanity: outcome must agree across arms for the same game_id.
    mismatched_outcome = int((a.loc[valid, outcome_col] != b.loc[valid, outcome_col]).sum())

    a = a.loc[valid]
    b = b.loc[valid]
    n = len(a)

    pick_a = (a[prob_col] >= 0.5).astype(int)  # 1 = pick home cover
    pick_b = (b[prob_col] >= 0.5).astype(int)
    y = a[outcome_col].astype(int)  # 1 = home covered

    correct_a = (pick_a == y).astype(int)
    correct_b = (pick_b == y).astype(int)
    delta = correct_a - correct_b  # +1: a right/b wrong, -1: a wrong/b right, 0: agree

    disagree = pick_a != pick_b
    f_count = int(disagree.sum())
    f = f_count / n if n else float("nan")

    a_wins = int(((delta == 1) & disagree).sum())
    b_wins = int(((delta == -1) & disagree).sum())

    mean_delta = float(delta.mean())  # fraction scale
    # Paired sd using the disagreement-only model MDE80 assumes: on agreeing
    # games delta==0 exactly, on disagreeing games delta==+-1.
    paired_sd_pred = math.sqrt(f) if n else float("nan")  # predicted from f alone
    paired_sd_emp = float(delta.std(ddof=1)) if n > 1 else float("nan")  # empirical, ddof=1
    se_emp = paired_sd_emp / math.sqrt(n) if n else float("nan")

    # Unpaired "coin-flip" sd some evidence strings compare against, for
    # explicit contrast -- NOT the right comparator for a paired delta.
    unpaired_coinflip_sd_pts = 100 * 0.5 / math.sqrt(n) if n else float("nan")

    z = mean_delta / se_emp if se_emp else float("nan")
    prob_positive = float(stats.norm.cdf(z)) if not math.isnan(z) else float("nan")

    ci95_lower = mean_delta - 1.96 * se_emp
    ci95_upper = mean_delta + 1.96 * se_emp

    mde80_pts = MDE80_Z * 100 * math.sqrt(f / n) if n else float("nan")

    return {
        "arm_a": arm_a,
        "arm_b": arm_b,
        "n_games": n,
        "mismatched_outcome_rows": mismatched_outcome,
        "disagreement_games": f_count,
        "f_disagreement_fraction": f,
        "a_wins_on_disagreement": a_wins,
        "b_wins_on_disagreement": b_wins,
        "mean_delta_accuracy_points": mean_delta * 100,
        "paired_sd_points_from_f": paired_sd_pred * 100,
        "paired_sd_points_empirical": paired_sd_emp * 100,
        "paired_se_points_empirical": se_emp * 100,
        "unpaired_coinflip_sd_points": unpaired_coinflip_sd_pts,
        "z_normal_approx": z,
        "probability_positive": prob_positive,
        "ci95_points": [ci95_lower * 100, ci95_upper * 100],
        "mde80_points": mde80_pts,
    }


def audit_qb_continuity(artifact_dir: Path) -> dict[str, dict[str, Any]]:
    source = str(artifact_dir / "predictions.parquet")
    df = pd.read_parquet(artifact_dir / "predictions.parquet")
    matched_alpha = paired_stats_from_picks(
        df,
        game_col="game_id",
        outcome_col="home_cover",
        prob_col="home_cover_probability",
        arm_col="arm",
        arm_a="candidate",
        arm_b="base_alpha1",
    )
    original_primary = paired_stats_from_picks(
        df,
        game_col="game_id",
        outcome_col="home_cover",
        prob_col="home_cover_probability",
        arm_col="arm",
        arm_a="candidate",
        arm_b="base_alpha10",
    )
    alpha_only = paired_stats_from_picks(
        df,
        game_col="game_id",
        outcome_col="home_cover",
        prob_col="home_cover_probability",
        arm_col="arm",
        arm_a="base_alpha1",
        arm_b="base_alpha10",
    )
    matched_alpha["source"] = source
    original_primary["source"] = source
    alpha_only["source"] = source
    return {
        # candidate=alpha1+continuity vs base_alpha1 (matched regularization)
        "player_qb_continuity_matched_alpha": matched_alpha,
        # candidate=alpha1+continuity vs base_alpha10, the ORIGINAL closed_negative pairing
        "player_qb_continuity": original_primary,
        # base_alpha1 vs base_alpha10, isolates the regularization change alone
        "alpha_only_control": alpha_only,
    }


def audit_pbp_drive_bundle(artifact_dir: Path) -> dict[str, dict[str, Any]]:
    source = str(artifact_dir / "predictions.parquet")
    df = pd.read_parquet(artifact_dir / "predictions.parquet")
    result = paired_stats_from_picks(
        df,
        game_col="game_id",
        outcome_col="home_cover",
        prob_col="home_cover_probability",
        arm_col="feature_profile",
        arm_a="pbp",
        arm_b="base",
    )
    result["source"] = source
    return {
        # candidate=full_pbp vs base
        "pbp_drive_bundle": result,
    }


def format_report(results: dict[str, dict[str, Any]]) -> str:
    lines: list[str] = []
    for family, stat in results.items():
        lines.append(f"\n=== {family} ===")
        for k, v in stat.items():
            lines.append(f"  {k}: {v}")
    return "\n".join(lines)


READ_ONLY_SCRIPT = True
# ENG-29: read-only with respect to artifacts/ and registry/; the ENG-29 scanner confirms its only
# write sites resolve to a caller-supplied `--output`/`--out` path with no artifacts/ or registry/
# default, never a governed tree by default.


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--qb-continuity-artifact",
        type=Path,
        default=REPO_ROOT / "artifacts/qb_continuity_replication/20260816T143913Z",
        help="Directory containing predictions.parquet for the QB continuity replication.",
    )
    parser.add_argument(
        "--pbp-artifact",
        type=Path,
        default=REPO_ROOT / "artifacts/pbp_replication/20260816T142340Z",
        help="Directory containing predictions.parquet for the PBP/drive bundle replication.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional path to dump the same numbers as JSON.",
    )
    args = parser.parse_args()

    results: dict[str, Any] = {}
    results.update(audit_qb_continuity(args.qb_continuity_artifact))
    results.update(audit_pbp_drive_bundle(args.pbp_artifact))

    print(format_report(results))

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(results, indent=2, default=str))
        print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()
