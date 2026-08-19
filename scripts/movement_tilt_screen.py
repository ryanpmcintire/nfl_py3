"""MOV-01: predicted line-movement direction as a forced-pick tilt.

Never-tested use of the existing MKT-06 line-movement machinery: on games
where a leak-free predicted movement direction DISAGREES with the active
model's own opener pick, does flipping to the movement side improve
forced-pick accuracy, graded at the opener (primary) and the close
(secondary)?

Full predeclaration, written and frozen before this script scored anything:
``<scratchpad>/agent_movement/predeclaration.md``. This module implements
exactly what that document specifies.

**Leak-free walk-forward.** Reuses ``nfl_ats.clv.build_pilot_frame`` /
``fit_pilot_model`` / ``FROZEN_PILOT_FEATURES`` -- the frozen MKT-06 pilot's
own frame builder and ridge model -- with no reimplementation. The archive
(2020-2025) has no season before 2020 to train on, so 2020 is excluded.
2021-2023 get a fresh expanding-window walk-forward refit (train seasons
strictly before the target season). 2024 and 2025 reuse the frozen MKT-06
pilot's own model AS-IS (trained once on 2020-2023, exactly
``FROZEN_PILOT_PROTOCOL``) -- no training row ever comes from the season
being scored.

**Grading.** Joined against the active ``weak_stack`` model's own
opener-evaluation artifact (``artifacts/opener_evaluation/20260818T013115Z/
per_game.parquet``, 1,537 paired REG games 2020-2025), restricted to the
five leak-free seasons. Paired comparison (tilted arm vs. the active
model's own arm), ``nfl_ats.clv.week_blocked_bootstrap`` (week-blocked
primary, season-blocked secondary), effect in accuracy points,
``probability_positive``.

**Measure-only.** This script never writes to the weak-signal registry
(``registry/weak_signals.json``); it writes a JSON+CSV artifact under
``artifacts/movement_tilt_screen/<run_id>/``. Recording to the weak-signal
registry is a separate, explicit ``nfl-ats weak-signals record`` call per
rule, made after reading this script's output (per the repo's registry
write-lock protocol -- concurrent agents share ``registry/weak_signals.json``).
It DOES write an automatic, low-stakes experiment-provenance stamp to
``registry/experiments/`` via ``write_experiment_artifact`` (RWB-09) -- a
run log, not a verdict; it carries no closing-ground and asserts nothing
about the rules' status.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from nfl_ats.clv import (  # noqa: E402
    FROZEN_PILOT_FEATURES,
    FROZEN_PILOT_PROTOCOL,
    build_pilot_frame,
    fit_pilot_model,
    pick_correct,
    resolve_active_model_config,
    week_blocked_bootstrap,
)
from nfl_ats.io import atomic_csv, run_id  # noqa: E402
from nfl_ats.modeling import regular_season_rows  # noqa: E402
from nfl_ats.odds_backfill import HISTORICAL_CAPTURE_KIND  # noqa: E402
from nfl_ats.provenance import artifact_provenance, write_experiment_artifact  # noqa: E402

DEFAULT_FEATURES = REPO / "data" / "processed" / "game_features_weak_stack.parquet"
DEFAULT_MARKET_ROOT = REPO / "data" / "market" / "raw"
DEFAULT_ARTIFACTS_ROOT = REPO / "artifacts"
DEFAULT_OPENER_EVAL_ARTIFACT = (
    REPO / "artifacts" / "opener_evaluation" / "20260818T013115Z" / "per_game.parquet"
)

# 2020 has no earlier archived season to train a leak-free movement model on
# (BUILD note in the predeclaration). Screened seasons are exactly these five.
SCREENED_SEASONS: tuple[int, ...] = (2021, 2022, 2023, 2024, 2025)
EXPANDING_WALK_FORWARD_SEASONS: tuple[int, ...] = (2021, 2022, 2023)
FROZEN_REUSE_SEASONS: tuple[int, ...] = (2024, 2025)

BOOTSTRAP_SAMPLES = 10_000
BOOTSTRAP_SEED = 20260819

RULES: tuple[str, ...] = ("primary", "variant1_no_filter", "variant2_top_quartile")


def _predicted_moves_for_season(
    pilot_frame: pd.DataFrame, target_season: int
) -> tuple[pd.DataFrame, float, float]:
    """Leak-free predicted move + confidence thresholds for one target season.

    Returns (predictions, median_threshold, q75_threshold), where the
    thresholds are computed from the fitted model's OWN predictions on its
    training rows only (never on the target season).
    """

    if target_season in EXPANDING_WALK_FORWARD_SEASONS:
        train = pilot_frame.loc[pilot_frame["season"].lt(target_season)]
    elif target_season in FROZEN_REUSE_SEASONS:
        train = pilot_frame.loc[
            pilot_frame["season"].between(
                FROZEN_PILOT_PROTOCOL.train_start_season,
                FROZEN_PILOT_PROTOCOL.train_end_season,
            )
        ]
    else:
        raise ValueError(f"Season {target_season} is not in the screened protocol")

    if train.empty:
        raise ValueError(f"No leak-free training rows available before season {target_season}")

    target = pilot_frame.loc[pilot_frame["season"].eq(target_season)]
    if target.empty:
        raise ValueError(f"No archived rows for target season {target_season}")

    model = fit_pilot_model(train)
    in_sample = np.asarray(model.predict(train.loc[:, list(FROZEN_PILOT_FEATURES)]), dtype=float)
    median_threshold = float(np.median(np.abs(in_sample)))
    q75_threshold = float(np.quantile(np.abs(in_sample), 0.75))

    predicted = np.asarray(model.predict(target.loc[:, list(FROZEN_PILOT_FEATURES)]), dtype=float)
    out = pd.DataFrame(
        {
            "game_id": target["game_id"].to_numpy(),
            "season": target_season,
            "predicted_close_minus_open": predicted,
            "train_games": len(train),
        }
    )
    return out, median_threshold, q75_threshold


def build_predictions(pilot_frame: pd.DataFrame) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for season in SCREENED_SEASONS:
        preds, median_threshold, q75_threshold = _predicted_moves_for_season(pilot_frame, season)
        preds["confidence_threshold_median"] = median_threshold
        preds["confidence_threshold_q75"] = q75_threshold
        frames.append(preds)
    return pd.concat(frames, ignore_index=True)


def _metric_fn_factory(
    open_col: str, close_col: str, tilt_open_col: str, tilt_close_col: str
) -> Callable[[pd.DataFrame], dict[str, float]]:
    def metric_fn(frame: pd.DataFrame) -> dict[str, float]:
        def mean_or_nan(series: pd.Series) -> float:
            values = pd.to_numeric(series, errors="coerce").dropna()
            return float(values.mean()) if len(values) else float("nan")

        delta_open = pd.to_numeric(frame[tilt_open_col], errors="coerce") - pd.to_numeric(
            frame[open_col], errors="coerce"
        )
        delta_close = pd.to_numeric(frame[tilt_close_col], errors="coerce") - pd.to_numeric(
            frame[close_col], errors="coerce"
        )
        delta_open_mean = (
            float(delta_open.dropna().mean()) if delta_open.notna().any() else float("nan")
        )
        delta_close_mean = (
            float(delta_close.dropna().mean()) if delta_close.notna().any() else float("nan")
        )
        return {
            "tilt_accuracy_open": mean_or_nan(frame[tilt_open_col]),
            "active_accuracy_open": mean_or_nan(frame[open_col]),
            "delta_open": delta_open_mean,
            "tilt_accuracy_close": mean_or_nan(frame[tilt_close_col]),
            "active_accuracy_close": mean_or_nan(frame[close_col]),
            "delta_close": delta_close_mean,
        }

    return metric_fn


def score_rules(scored: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply the 3 predeclared rules; return (per-game frame, bootstrap table)."""

    scored = scored.copy()
    scored["movement_pick_home"] = scored["predicted_close_minus_open"].gt(0.0)
    scored["disagreement"] = scored["movement_pick_home"].ne(scored["pick_home_at_open"])
    scored["abs_predicted"] = scored["predicted_close_minus_open"].abs()

    # The active-model baseline at the close grade must be the SAME decision
    # the tilt rule modifies (the opener pick), settled at the close line --
    # NOT `correct_at_close` from the opener-evaluation artifact, which is the
    # active model independently RE-EVALUATED with the close-time spread as
    # input (a different decision that already "sees" close-time information
    # no tilt rule here ever gets). Using that column as the baseline would
    # let the baseline win on close-time information alone, confounding
    # whatever the movement tilt itself contributes. This isolates the tilt.
    scored["active_same_decision_correct_at_close"] = pick_correct(
        scored["pick_home_at_open"], scored["margin_vs_close"]
    )

    scored["high_confidence_median"] = scored["abs_predicted"].ge(
        scored["confidence_threshold_median"]
    )
    scored["high_confidence_q75"] = scored["abs_predicted"].ge(scored["confidence_threshold_q75"])

    rule_flip_masks = {
        "primary": scored["disagreement"] & scored["high_confidence_median"],
        "variant1_no_filter": scored["disagreement"],
        "variant2_top_quartile": scored["disagreement"] & scored["high_confidence_q75"],
    }

    bootstrap_rows: list[pd.DataFrame] = []
    for rule, flip_mask in rule_flip_masks.items():
        tilted_pick_home = scored["pick_home_at_open"].where(
            ~flip_mask, scored["movement_pick_home"]
        )
        open_col = f"tilt_correct_at_open__{rule}"
        close_col = f"tilt_correct_at_close__{rule}"
        scored[f"flip__{rule}"] = flip_mask
        scored[open_col] = pick_correct(tilted_pick_home, scored["margin_vs_open"])
        scored[close_col] = pick_correct(tilted_pick_home, scored["margin_vs_close"])

        metric_fn = _metric_fn_factory(
            "correct_at_open", "active_same_decision_correct_at_close", open_col, close_col
        )
        for block in ("week", "season"):
            table = week_blocked_bootstrap(
                scored,
                metric_fn,
                block=block,
                samples=BOOTSTRAP_SAMPLES,
                seed=BOOTSTRAP_SEED,
            )
            table.insert(0, "rule", rule)
            table.insert(1, "n_flips", int(flip_mask.sum()))
            bootstrap_rows.append(table)

    return scored, pd.concat(bootstrap_rows, ignore_index=True)


def disagreement_report(scored: pd.DataFrame) -> pd.DataFrame:
    """Per-season disagreement rate and unpaired descriptive accuracy on disagreements."""

    rows: list[dict[str, Any]] = []
    for season, group in scored.groupby("season", sort=True):
        disagreements = group.loc[group["disagreement"]]
        n_dis = len(disagreements)

        def acc(col: str, frame: pd.DataFrame = disagreements) -> float:
            values = pd.to_numeric(frame[col], errors="coerce").dropna()
            return float(values.mean()) if len(values) else float("nan")

        movement_correct_open = pick_correct(
            disagreements["movement_pick_home"], disagreements["margin_vs_open"]
        )
        movement_correct_close = pick_correct(
            disagreements["movement_pick_home"], disagreements["margin_vs_close"]
        )
        rows.append(
            {
                "season": int(str(season)),
                "n_games": len(group),
                "n_disagreements": n_dis,
                "disagreement_rate": n_dis / len(group) if len(group) else float("nan"),
                "active_accuracy_open_on_disagreements": acc("correct_at_open"),
                "active_accuracy_close_on_disagreements": acc(
                    "active_same_decision_correct_at_close"
                ),
                "movement_accuracy_open_on_disagreements": float(
                    movement_correct_open.dropna().mean()
                )
                if movement_correct_open.notna().any()
                else float("nan"),
                "movement_accuracy_close_on_disagreements": float(
                    movement_correct_close.dropna().mean()
                )
                if movement_correct_close.notna().any()
                else float("nan"),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, default=DEFAULT_FEATURES)
    parser.add_argument("--market-root", type=Path, default=DEFAULT_MARKET_ROOT)
    parser.add_argument("--artifacts-root", type=Path, default=DEFAULT_ARTIFACTS_ROOT)
    parser.add_argument("--opener-eval-artifact", type=Path, default=DEFAULT_OPENER_EVAL_ARTIFACT)
    parser.add_argument("--min-train-games", type=int, default=500)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    features = pd.read_parquet(args.features)
    features = regular_season_rows(features)

    active_model_config = resolve_active_model_config(args.artifacts_root)

    pilot_frame = build_pilot_frame(
        args.market_root,
        features,
        capture_kind=HISTORICAL_CAPTURE_KIND,
        active_model_config=active_model_config,
        min_train_games=args.min_train_games,
    )
    pilot_frame = pilot_frame.loc[pilot_frame["season"].isin((*SCREENED_SEASONS, 2020))].copy()

    predictions = build_predictions(pilot_frame)

    opener_eval = pd.read_parquet(args.opener_eval_artifact)
    required_cols = {
        "game_id",
        "season",
        "week",
        "pick_home_at_open",
        "correct_at_open",
        "correct_at_close",
        "margin_vs_open",
        "margin_vs_close",
    }
    missing = sorted(required_cols.difference(opener_eval.columns))
    if missing:
        raise SystemExit(f"opener-evaluation artifact is missing columns: {missing}")
    opener_eval = opener_eval.loc[opener_eval["season"].isin(SCREENED_SEASONS)][list(required_cols)]

    scored = predictions.merge(opener_eval, on="game_id", how="inner", validate="one_to_one")
    if "season_y" in scored.columns:
        assert (scored["season_x"] == scored["season_y"]).all()
        scored = scored.drop(columns=["season_y"]).rename(columns={"season_x": "season"})

    scored, bootstrap_table = score_rules(scored)
    disagreements = disagreement_report(scored)

    output = args.output or (REPO / "artifacts" / "movement_tilt_screen" / run_id())
    atomic_csv(bootstrap_table, output / "bootstrap.csv")
    atomic_csv(disagreements, output / "disagreement_report.csv")
    atomic_csv(
        scored.drop(columns=[c for c in scored.columns if c.startswith("flip__")]),
        output / "per_game.csv",
    )

    configuration = {
        "command": "movement_tilt_screen",
        "market_root": str(args.market_root),
        "artifacts_root": str(args.artifacts_root),
        "opener_eval_artifact": str(args.opener_eval_artifact),
        "min_train_games": args.min_train_games,
        "active_model_config": active_model_config,
        "screened_seasons": list(SCREENED_SEASONS),
        "bootstrap_samples": BOOTSTRAP_SAMPLES,
        "bootstrap_seed": BOOTSTRAP_SEED,
    }
    metadata = {
        "command": "movement_tilt_screen",
        "predeclaration": (
            "scratchpad/agent_movement/predeclaration.md (not tracked; see AGENTS.md taxonomy)"
        ),
        "active_model_config": active_model_config,
        "screened_seasons": list(SCREENED_SEASONS),
        "excluded_season_2020_reason": (
            "no earlier archived season to train a leak-free movement model on"
        ),
        "expanding_walk_forward_seasons": list(EXPANDING_WALK_FORWARD_SEASONS),
        "frozen_reuse_seasons": list(FROZEN_REUSE_SEASONS),
        "frozen_pilot_train_window": [
            FROZEN_PILOT_PROTOCOL.train_start_season,
            FROZEN_PILOT_PROTOCOL.train_end_season,
        ],
        "bootstrap_samples": BOOTSTRAP_SAMPLES,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "games": len(scored),
        "rules": list(RULES),
        "n_flips_per_rule": {rule: int(scored[f"flip__{rule}"].sum()) for rule in RULES},
        "bootstrap_summary": bootstrap_table.to_dict(orient="records"),
        "disagreement_report": disagreements.to_dict(orient="records"),
        "provenance": artifact_provenance(configuration, args.features, project_root=REPO),
    }
    write_experiment_artifact(
        output,
        "metadata.json",
        metadata,
        command="movement_tilt_screen",
        metrics=metadata,
        notes=(
            "MOV-01 measure-only screen; never writes to registry/weak_signals.json "
            "directly, recording is a separate nfl-ats weak-signals record call."
        ),
    )
    print(f"Wrote artifact to {output}")
    print(bootstrap_table.to_string(index=False))
    print(disagreements.to_string(index=False))


if __name__ == "__main__":
    main()
