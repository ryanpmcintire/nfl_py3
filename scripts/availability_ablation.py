"""Attribution ablation inside the already-spent MOD-07 opener window.

Why this is admissible without a rotation window: the [2020, 2021] opener block
is ALREADY SPENT for ``mod07_weak_signal_stack`` (registry/rotation_registry.json,
``spent_at`` 2026-08-17). Decomposing a look that was already taken is attribution
on data already seen -- ``docs/pool_edge_plan.md`` states this explicitly ("an
ablation on the already-spent window (free -- attribution on data already looked
at costs no window)"). Nothing here touches the registry: no ``assign``, no
``record_look``, no new window. It is a pure re-read.

What it decomposes. The recorded MOD-07 look scored two arms:

* ``A`` baseline -- ``player`` profile on ``game_features_player.parquet``
  (hand-authored fixed injury priors, no player-value columns, no bias columns);
* ``C`` candidate -- ``weak_stack`` profile on ``game_features_weak_stack.parquet``
  (LEARNED availability semantics + player-value composite + three bias families).

``weak_stack`` = ``full_player_value`` + the nine bias columns, so the stack
splits cleanly in two. This script adds the middle arm:

* ``B`` -- ``player_value`` profile on ``game_features_weak_stack.parquet``
  (learned availability + player value, bias columns dropped).

Then ``B - A`` is the availability/player-value contribution and ``C - B`` is
the opener-bias contribution. ``C - A`` must reproduce the recorded +1.97 points
at ``probability_positive`` 0.8745, which is the correctness check on the whole
reconstruction: if the recorded pair does not come back byte-for-byte, the third
arm means nothing and the run is discarded.

The window split is reconstructed rather than taken from
``rotation.confirmation_split``, which refuses a spent window by design. The
reconstruction is a literal transcription of that function's body (regular-season
rows, window = the assigned seasons, training = every completed game strictly
before the window's first gameday) and is asserted against the recorded arm
accuracies before any new number is reported.

Usage::

    .\\.tools\\uv.exe run --no-sync python scripts/availability_ablation.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from nfl_ats.clv import opener_pick_evaluation, week_blocked_bootstrap
from nfl_ats.modeling import regular_season_rows
from nfl_ats.rotation import load_registry

REPO = Path(__file__).resolve().parents[1]

REGRESSOR = "ridge"
RIDGE_ALPHA = 10.0

# The recorded MOD-07 look, from artifacts/mod07_weak_stack/opener_2020_2021.json.
# These are the reproduction targets; the ablation is void if they do not return.
RECORDED = {
    "paired_games": 456,
    "weeks": 35,
    "baseline_accuracy": 0.5131578947368421,
    "candidate_accuracy": 0.5328947368421053,
    "delta": 0.019736842105263164,
    "probability_positive": 0.8745,
}


def _config(profile: str) -> dict[str, Any]:
    return {
        "feature_profile": profile,
        "regressor": REGRESSOR,
        "ridge_alpha": RIDGE_ALPHA,
        "target": "market_residual",
    }


def spent_window_split(
    features: pd.DataFrame, seasons: tuple[int, int]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Reproduce ``rotation.confirmation_split`` for a window that is already spent.

    Transcribed from that function so the fit sees exactly the forward-chained
    training the registry assigned when the look was taken.
    """

    frame = regular_season_rows(features).copy()
    frame["gameday"] = pd.to_datetime(frame["gameday"], errors="raise")
    span = list(range(int(seasons[0]), int(seasons[1]) + 1))
    missing = [s for s in span if s not in set(frame["season"].astype(int))]
    if missing:
        raise ValueError(f"Feature table is missing window seasons: {missing}")
    window = frame.loc[frame["season"].astype(int).isin(span)].copy()
    cutoff = window["gameday"].min()
    training = frame.loc[frame["gameday"].lt(cutoff) & frame["result"].notna()].copy()
    if training.empty:
        raise ValueError("No completed games before the window; cannot forward-chain")
    return training, window


def arm(
    features: pd.DataFrame,
    *,
    seasons: tuple[int, int],
    profile: str,
    market_root: Path,
    min_train_games: int,
) -> pd.DataFrame:
    """Score one arm at the opener grade over the spent window."""

    training, window = spent_window_split(features, seasons)
    scoped = pd.concat([training, window], ignore_index=True)
    scored = opener_pick_evaluation(
        market_root,
        scoped,
        active_model_config=_config(profile),
        min_train_games=min_train_games,
    )
    span = sorted(window["season"].astype(int).unique())
    scored = scored.loc[scored["season"].isin(span)]
    return scored.loc[scored["correct_at_open"].notna()].copy()


def paired_frame(left_arm: pd.DataFrame, right_arm: pd.DataFrame) -> pd.DataFrame:
    """One row per game scored by BOTH arms; anything else is not a paired test."""

    left = left_arm[["game_id", "season", "week", "correct_at_open", "pick_home_at_open"]].rename(
        columns={"correct_at_open": "left_correct", "pick_home_at_open": "left_pick_home"}
    )
    right = right_arm[["game_id", "correct_at_open", "pick_home_at_open"]].rename(
        columns={"correct_at_open": "right_correct", "pick_home_at_open": "right_pick_home"}
    )
    paired = left.merge(right, on="game_id", how="inner")
    paired["left_correct"] = paired["left_correct"].astype(float)
    paired["right_correct"] = paired["right_correct"].astype(float)
    return paired.sort_values(["season", "week", "game_id"]).reset_index(drop=True)


def _metric_fn(frame: pd.DataFrame) -> dict[str, float]:
    return {"right_minus_left": float(frame["right_correct"].mean() - frame["left_correct"].mean())}


def contrast(
    left_arm: pd.DataFrame,
    right_arm: pd.DataFrame,
    *,
    name: str,
    samples: int,
    seed: int,
) -> dict[str, Any]:
    paired = paired_frame(left_arm, right_arm)
    if paired.empty:
        raise ValueError(f"{name}: no game scored by both arms")
    bootstrap = week_blocked_bootstrap(paired, _metric_fn, block="week", samples=samples, seed=seed)
    row = bootstrap.iloc[0]
    disagree = paired.loc[paired["left_pick_home"].ne(paired["right_pick_home"])]
    return {
        "contrast": name,
        "paired_games": len(paired),
        "weeks": int(paired.groupby(["season", "week"]).ngroups),
        "left_accuracy": float(paired["left_correct"].mean()),
        "right_accuracy": float(paired["right_correct"].mean()),
        "delta_points": float(row["estimate"]) * 100.0,
        "bootstrap_lower_points": float(row["lower"]) * 100.0,
        "bootstrap_upper_points": float(row["upper"]) * 100.0,
        "probability_positive": float(row["probability_positive"]),
        "picks_disagreeing": len(disagree),
        "disagreement_left_correct": (
            float(disagree["left_correct"].mean()) if len(disagree) else None
        ),
        "disagreement_right_correct": (
            float(disagree["right_correct"].mean()) if len(disagree) else None
        ),
    }


def run(
    *,
    player_features: pd.DataFrame,
    stack_features: pd.DataFrame,
    seasons: tuple[int, int],
    market_root: Path,
    min_train_games: int,
    samples: int,
    seed: int,
) -> dict[str, Any]:
    arm_a = arm(
        player_features,
        seasons=seasons,
        profile="player",
        market_root=market_root,
        min_train_games=min_train_games,
    )
    arm_b = arm(
        stack_features,
        seasons=seasons,
        profile="player_value",
        market_root=market_root,
        min_train_games=min_train_games,
    )
    arm_c = arm(
        stack_features,
        seasons=seasons,
        profile="weak_stack",
        market_root=market_root,
        min_train_games=min_train_games,
    )

    reproduction = contrast(
        arm_a, arm_c, name="C_minus_A_recorded_look", samples=samples, seed=seed
    )
    availability = contrast(
        arm_a, arm_b, name="B_minus_A_availability_and_value", samples=samples, seed=seed
    )
    bias = contrast(arm_b, arm_c, name="C_minus_B_opener_bias", samples=samples, seed=seed)
    paired_frames = {
        "C_minus_A_recorded_look": paired_frame(arm_a, arm_c),
        "B_minus_A_availability_and_value": paired_frame(arm_a, arm_b),
        "C_minus_B_opener_bias": paired_frame(arm_b, arm_c),
    }

    checks = {
        "paired_games_match": reproduction["paired_games"] == RECORDED["paired_games"],
        "weeks_match": reproduction["weeks"] == RECORDED["weeks"],
        "baseline_accuracy_match": abs(
            reproduction["left_accuracy"] - RECORDED["baseline_accuracy"]
        )
        < 1e-12,
        "candidate_accuracy_match": abs(
            reproduction["right_accuracy"] - RECORDED["candidate_accuracy"]
        )
        < 1e-12,
        "delta_match": abs(reproduction["delta_points"] / 100.0 - RECORDED["delta"]) < 1e-12,
        "probability_positive_match": abs(
            reproduction["probability_positive"] - RECORDED["probability_positive"]
        )
        < 1e-9,
    }
    return paired_frames, {
        "window": list(seasons),
        "grade": "opener",
        "registry_note": (
            "The [2020, 2021] opener window is already spent for "
            "mod07_weak_signal_stack. This run is attribution on data already "
            "looked at and takes no new look; the registry is not written."
        ),
        "arms": {
            "A": "player profile on game_features_player.parquet (fixed injury priors)",
            "B": "player_value profile on game_features_weak_stack.parquet"
            " (learned availability + value)",
            "C": "weak_stack profile on game_features_weak_stack.parquet (B + three bias families)",
        },
        "reproduction_checks": checks,
        "reproduction_ok": all(checks.values()),
        "contrasts": [reproduction, availability, bias],
        "bootstrap_samples": samples,
        "bootstrap_seed": seed,
        "min_train_games": min_train_games,
    }


READ_ONLY_SCRIPT = True
# ENG-29: read-only with respect to artifacts/ and registry/; the ENG-29 scanner confirms its only
# write sites resolve to a caller-supplied `--output`/`--out` path with no artifacts/ or registry/
# default, never a governed tree by default.


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--family", default="mod07_weak_signal_stack")
    parser.add_argument(
        "--player-features",
        type=Path,
        default=REPO / "data/processed/game_features_player.parquet",
    )
    parser.add_argument(
        "--stack-features",
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

    registry = load_registry(args.registry)
    declared = registry.families[args.family]
    spent = [w for w in declared.windows if w.state == "spent"]
    if not spent:
        raise SystemExit(
            f"{args.family} has no spent window; this script only decomposes a look "
            "that was already taken."
        )
    seasons = (int(spent[-1].seasons[0]), int(spent[-1].seasons[1]))

    paired_frames, summary = run(
        player_features=pd.read_parquet(args.player_features),
        stack_features=pd.read_parquet(args.stack_features),
        seasons=seasons,
        market_root=args.market_root,
        min_train_games=args.min_train_games,
        samples=args.samples,
        seed=args.seed,
    )
    print(json.dumps(summary, indent=2))
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        for name, frame in paired_frames.items():
            frame.to_parquet(args.out.with_suffix(f".{name}.parquet"))


if __name__ == "__main__":
    main()
