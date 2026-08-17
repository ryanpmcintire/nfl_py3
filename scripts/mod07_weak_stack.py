"""SPEC-4 MOD-07 weak-signal stack: the paired opener-graded look.

Two arms, one registry window, one look:

* **baseline** -- the frozen active config (``player`` profile, ridge 10, no
  calibration) on ``game_features_player.parquet``;
* **candidate** -- the ``weak_stack`` profile (player value composite + the
  three documented opener-bias families) on
  ``game_features_weak_stack.parquet``, which carries LEARNED availability
  semantics in its injury columns.

Both are graded by the existing :func:`clv.opener_pick_evaluation` machinery at
the Tuesday opener, restricted to the registry window via
:func:`rotation.confirmation_split` so the fit sees exactly the forward-chained
training the registry assigned. Pairing is per ``game_id``; the reported
evidence is the paired accuracy delta with ``probability_positive`` from a
week-blocked bootstrap.

Nothing here writes to the registry -- ``rotation record`` is a separate,
deliberate step, exactly as with the SPEC-5 screen.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from nfl_ats.clv import opener_pick_evaluation, week_blocked_bootstrap
from nfl_ats.rotation import confirmation_split, load_registry

REPO = Path(__file__).resolve().parents[1]

REGRESSOR = "ridge"
RIDGE_ALPHA = 10.0

# Predeclared in SPEC-4; fixed before any number below existed.
CONFIRM_AT = 0.90
CLOSE_NEGATIVE_AT = 0.10


def _config(profile: str) -> dict[str, Any]:
    return {
        "feature_profile": profile,
        "regressor": REGRESSOR,
        "ridge_alpha": RIDGE_ALPHA,
        "target": "market_residual",
    }


def arm(
    features: pd.DataFrame,
    *,
    registry: Any,
    family: str,
    profile: str,
    market_root: Path,
    min_train_games: int,
) -> pd.DataFrame:
    """Score one arm at the opener grade over the family's assigned window."""

    training, window = confirmation_split(features, registry, family)
    # The evaluator re-derives its own forward-chained cutoff per week; handing it
    # training+window means only window seasons can be scored (the opener archive
    # does not reach the training seasons) while every earlier completed game
    # remains available to the fit -- which is precisely the registry's split.
    scoped = pd.concat([training, window], ignore_index=True)
    scored = opener_pick_evaluation(
        market_root,
        scoped,
        active_model_config=_config(profile),
        min_train_games=min_train_games,
    )
    seasons = sorted(window["season"].astype(int).unique())
    scored = scored.loc[scored["season"].isin(seasons)]
    return scored.loc[scored["correct_at_open"].notna()].copy()


def paired_frame(baseline: pd.DataFrame, candidate: pd.DataFrame) -> pd.DataFrame:
    """One row per game scored by BOTH arms; anything else is not a paired test."""

    left = baseline[["game_id", "season", "week", "correct_at_open", "pick_home_at_open"]].rename(
        columns={"correct_at_open": "baseline_correct", "pick_home_at_open": "baseline_pick_home"}
    )
    right = candidate[["game_id", "correct_at_open", "pick_home_at_open"]].rename(
        columns={"correct_at_open": "candidate_correct", "pick_home_at_open": "candidate_pick_home"}
    )
    paired = left.merge(right, on="game_id", how="inner")
    paired["baseline_correct"] = paired["baseline_correct"].astype(float)
    paired["candidate_correct"] = paired["candidate_correct"].astype(float)
    return paired.sort_values(["season", "week", "game_id"]).reset_index(drop=True)


def _metric_fn(frame: pd.DataFrame) -> dict[str, float]:
    return {
        "candidate_minus_baseline": float(
            frame["candidate_correct"].mean() - frame["baseline_correct"].mean()
        )
    }


def verdict_for(probability_positive: float) -> str:
    if probability_positive >= CONFIRM_AT:
        return "confirmed"
    if probability_positive <= CLOSE_NEGATIVE_AT:
        return "closed_negative"
    return "unresolved"


def run(
    *,
    baseline_features: pd.DataFrame,
    candidate_features: pd.DataFrame,
    family: str,
    market_root: Path,
    registry_path: Path | None,
    min_train_games: int,
    samples: int,
    seed: int,
) -> tuple[dict[str, Any], pd.DataFrame]:
    registry = load_registry(registry_path)
    baseline = arm(
        baseline_features,
        registry=registry,
        family=family,
        profile="player",
        market_root=market_root,
        min_train_games=min_train_games,
    )
    candidate = arm(
        candidate_features,
        registry=registry,
        family=family,
        profile="weak_stack",
        market_root=market_root,
        min_train_games=min_train_games,
    )
    paired = paired_frame(baseline, candidate)
    if paired.empty:
        raise ValueError("No game was scored by both arms")

    bootstrap = week_blocked_bootstrap(paired, _metric_fn, block="week", samples=samples, seed=seed)
    row = bootstrap.iloc[0]
    probability_positive = float(row["probability_positive"])
    disagreements = paired.loc[paired["baseline_pick_home"].ne(paired["candidate_pick_home"])]
    summary: dict[str, Any] = {
        "family": family,
        "grade": "opener",
        "seasons": sorted(int(s) for s in paired["season"].unique()),
        "paired_games": len(paired),
        "weeks": int(paired.groupby(["season", "week"]).ngroups),
        "baseline_accuracy": float(paired["baseline_correct"].mean()),
        "candidate_accuracy": float(paired["candidate_correct"].mean()),
        "delta": float(row["estimate"]),
        "bootstrap_lower": float(row["lower"]),
        "bootstrap_upper": float(row["upper"]),
        "probability_positive": probability_positive,
        "picks_disagreeing": len(disagreements),
        "disagreement_baseline_correct": (
            float(disagreements["baseline_correct"].mean()) if len(disagreements) else float("nan")
        ),
        "disagreement_candidate_correct": (
            float(disagreements["candidate_correct"].mean()) if len(disagreements) else float("nan")
        ),
        "bootstrap_samples": samples,
        "bootstrap_seed": seed,
        "verdict": verdict_for(probability_positive),
        "verdict_thresholds": {"confirmed": CONFIRM_AT, "closed_negative": CLOSE_NEGATIVE_AT},
    }
    return summary, paired


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--family", default="mod07_weak_signal_stack")
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
    parser.add_argument("--market-root", type=Path, default=REPO / "data/market/raw")
    parser.add_argument("--registry", type=Path, default=None)
    parser.add_argument("--min-train-games", type=int, default=500)
    parser.add_argument("--samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260817)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    summary, paired = run(
        baseline_features=pd.read_parquet(args.baseline_features),
        candidate_features=pd.read_parquet(args.candidate_features),
        family=args.family,
        market_root=args.market_root,
        registry_path=args.registry,
        min_train_games=args.min_train_games,
        samples=args.samples,
        seed=args.seed,
    )
    print(json.dumps(summary, indent=2))
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        paired.to_parquet(args.out.with_suffix(".paired.parquet"))


if __name__ == "__main__":
    main()
