"""Lane T promotion evaluation (docs/promotion_eval_20260905.md).

Screens ``weak_stack_qb_revenge``, ``weak_stack_deadline_drag``, and their
union (``weak_stack_qb_revenge_deadline_drag``) against production
``weak_stack``, on the rebuilt (ENG-39-fixed) production table, at the
OPENER, on the full REUSED 2020-2025 Tuesday-opener archive -- a promotion
look, not a fresh rotation confirmation (see the doc's "Multiplicity
disclosure" section). No rotation window is spent.

Reuses ``scripts/on_production_opener_confirmation.py``'s ``paired_frame``,
``summarize``, ``opener_evaluation_metrics`` verbatim (imported, never
reimplemented). Does NOT use that module's ``scoped_window_frame``/
``confirmation_split``, because this is not a rotation-assigned
confirmation -- ``opener_pick_evaluation`` itself already restricts to
completed games with a resolvable Tuesday-opener + close pairing, which the
market archive only carries for 2020-2025, so no explicit season filter is
needed to reproduce that population.

Two modes:

- ``--mode screen``: the single promotion look -- four arms
  (base/qb_revenge/deadline_drag/both), paired candidate-minus-base summary
  statistics, per-season breakdown, and the 2026 Week-1 card-impact read
  (fit-once-on-full-table, predict the current week's games, read-only
  against the newest already-published predictions.csv).
- ``--mode positive-control``: each candidate arm's own added column(s)
  replaced by the REALIZED ``ats_margin`` before fitting; base always
  stays clean. Must read a huge, unambiguous positive effect or the harness
  cannot be trusted.
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))
if str(REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "scripts"))

import on_production_opener_confirmation as confirmation  # noqa: E402

from nfl_ats.clv import opener_pick_evaluation  # noqa: E402
from nfl_ats.margin import fit_margin_model, margin_feature_columns  # noqa: E402
from nfl_ats.provenance import artifact_provenance, write_experiment_artifact  # noqa: E402
from nfl_ats.qb_identity_features import (  # noqa: E402
    QB_REVENGE_COLUMN,
    attach_qb_revenge_features,
)
from nfl_ats.transaction_flag_features import (  # noqa: E402
    DEADLINE_INTEGRATION_DRAG_COLUMN,
    attach_deadline_integration_drag_features,
)

BASELINE_PROFILE = confirmation.BASELINE_PROFILE
REGRESSOR = confirmation.REGRESSOR
RIDGE_ALPHA = confirmation.RIDGE_ALPHA
BOOTSTRAP_SAMPLES = confirmation.BOOTSTRAP_SAMPLES
BOOTSTRAP_SEED = confirmation.BOOTSTRAP_SEED
DEFAULT_MIN_TRAIN_GAMES = 500
DEFAULT_FEATURES = REPO_ROOT / "data/processed/game_features_weak_stack.parquet"
DEFAULT_MARKET_ROOT = confirmation.DEFAULT_MARKET_ROOT


@dataclass(frozen=True)
class Arm:
    name: str
    profile: str
    columns: tuple[str, ...]


ARMS: dict[str, Arm] = {
    "base": Arm("base", BASELINE_PROFILE, ()),
    "qb_revenge": Arm("qb_revenge", "weak_stack_qb_revenge", (QB_REVENGE_COLUMN,)),
    "deadline_drag": Arm(
        "deadline_drag", "weak_stack_deadline_drag", (DEADLINE_INTEGRATION_DRAG_COLUMN,)
    ),
    "both": Arm(
        "both",
        "weak_stack_qb_revenge_deadline_drag",
        (QB_REVENGE_COLUMN, DEADLINE_INTEGRATION_DRAG_COLUMN),
    ),
}
CANDIDATE_ARM_NAMES = ("qb_revenge", "deadline_drag", "both")


def model_config(profile: str) -> dict[str, Any]:
    return {
        "feature_profile": profile,
        "regressor": REGRESSOR,
        "ridge_alpha": RIDGE_ALPHA,
        "target": "market_residual",
    }


def build_combined_features(base_features: pd.DataFrame) -> pd.DataFrame:
    """Attach BOTH candidate columns onto the production table, exactly as
    each column's own lane built it (newest local schedule/roster/combine/
    transaction-wire/snap-count snapshots, no precomputed candidate-specific
    parquet)."""

    with_revenge = attach_qb_revenge_features(base_features)
    with_both = attach_deadline_integration_drag_features(with_revenge)
    return with_both


def check_profile_identity(arm: Arm, features: pd.DataFrame) -> dict[str, Any]:
    """Fail closed unless ``arm.profile`` is production ``weak_stack`` plus
    exactly ``arm.columns`` -- generalizes
    ``on_production_opener_confirmation.profile_identity`` to a possibly
    multi-column arm (the ``both`` arm adds two columns at once)."""

    baseline = set(margin_feature_columns("market_residual", BASELINE_PROFILE))
    treatment = set(margin_feature_columns("market_residual", arm.profile))
    if treatment - baseline != set(arm.columns) or baseline - treatment:
        raise ValueError(f"{arm.profile} is not {BASELINE_PROFILE} plus {arm.columns}")
    missing = sorted(treatment.difference(features.columns))
    if missing:
        raise ValueError(f"Feature table lacks required arm inputs: {missing}")
    return {
        "baseline_columns": len(baseline),
        "candidate_columns": len(treatment),
        "added_columns": list(arm.columns),
    }


def run_arm(
    features: pd.DataFrame,
    arm: Arm,
    *,
    market_root: Path,
    min_train_games: int,
    leak: bool,
) -> pd.DataFrame:
    source = features.copy() if leak else features
    if leak:
        # The only permitted treatment leak, used solely by
        # --mode positive-control. Replaces EVERY column this arm adds
        # (one for qb_revenge/deadline_drag, two for both) with the
        # realized ats_margin.
        for column in arm.columns:
            source[column] = pd.to_numeric(source["ats_margin"], errors="raise")
    return opener_pick_evaluation(
        market_root,
        source,
        active_model_config=model_config(arm.profile),
        min_train_games=min_train_games,
    )


def per_season_breakdown(paired: pd.DataFrame) -> list[dict[str, Any]]:
    rows = []
    for season, group in paired.groupby("season", sort=True):
        baseline = group["baseline_correct_open_pr"].dropna()
        candidate = group["candidate_correct_open_pr"].dropna()
        disagreeing = int((group["baseline_pick_home_pr"] != group["candidate_pick_home_pr"]).sum())
        rows.append(
            {
                "season": int(season),
                "n": len(group),
                "baseline_accuracy": float(baseline.mean()) if len(baseline) else float("nan"),
                "candidate_accuracy": float(candidate.mean()) if len(candidate) else float("nan"),
                "delta": (
                    float(candidate.mean() - baseline.mean())
                    if len(baseline) and len(candidate)
                    else float("nan")
                ),
                "disagreeing_picks": disagreeing,
            }
        )
    return rows


def card_impact(features: pd.DataFrame, published_predictions_path: Path) -> dict[str, Any]:
    """2026 Week-1 read-only card-impact diagnostic.

    Fits each arm ONCE on the full current table (all completed games, no
    window restriction -- the same internal training-row selection
    ``fit_margin_model``/``opener_pick_evaluation``/production's own weekly
    forecast all share) and predicts the 16 in-table 2026 Week-1 REG games
    with ``probability_method="gaussian"`` (matching
    ``nfl_ats.outcomes.score_outcome_week``'s own production call). Reports
    two DISTINCT comparisons, kept separate because they answer different
    questions: each candidate arm vs. the freshly-recomputed ``base`` arm
    (isolates the candidate's own effect), and each arm vs. the pick already
    recorded in the newest, already-published predictions.csv (read-only,
    never regenerated; conflates the candidate effect with the ENG-39 table
    fix itself, since that published card predates the rebuilt table --
    disclosed in docs/promotion_eval_20260905.md).
    """

    week1 = features.loc[
        (pd.to_numeric(features["season"], errors="coerce") == 2026)
        & (pd.to_numeric(features["week"], errors="coerce") == 1)
        & (features["game_type"] == "REG")
    ].copy()

    published = pd.read_csv(published_predictions_path)
    published_mr = published.loc[published["method"].eq("market_residual")][
        ["game_id", "home_cover_probability"]
    ].copy()
    published_mr["published_pick_home"] = published_mr["home_cover_probability"].ge(0.5)
    published_mr = published_mr.rename(
        columns={"home_cover_probability": "published_home_cover_probability"}
    )

    merged = week1[["game_id", "season", "week", "away_team", "home_team"]].copy()
    for name, arm in ARMS.items():
        model = fit_margin_model(
            features,
            target="market_residual",
            model_name="ridge",
            feature_profile=arm.profile,
            ridge_alpha=RIDGE_ALPHA,
        )
        predicted = model.predict(week1, probability_method="gaussian")
        merged[f"{name}_home_cover_probability"] = predicted["home_cover_probability"].to_numpy()
        merged[f"{name}_pick_home"] = predicted["home_cover_probability"].ge(0.5).to_numpy()
    merged = merged.merge(published_mr, on="game_id", how="left")

    result: dict[str, Any] = {
        "n_week1_games": len(merged),
        "published_predictions_path": str(published_predictions_path),
        "base_flips_vs_published_card": int(
            (merged["base_pick_home"] != merged["published_pick_home"]).sum()
        ),
    }
    for name in CANDIDATE_ARM_NAMES:
        result[name] = {
            "flips_vs_base_recomputed": int(
                (merged[f"{name}_pick_home"] != merged["base_pick_home"]).sum()
            ),
            "flips_vs_published_card": int(
                (merged[f"{name}_pick_home"] != merged["published_pick_home"]).sum()
            ),
        }
    result["per_game"] = merged.drop(
        columns=["published_home_cover_probability"], errors="ignore"
    ).to_dict(orient="records")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("screen", "positive-control"), required=True)
    parser.add_argument("--features", type=Path, default=DEFAULT_FEATURES)
    parser.add_argument("--market-root", type=Path, default=DEFAULT_MARKET_ROOT)
    parser.add_argument("--published-predictions", type=Path, default=None)
    parser.add_argument("--bootstrap-samples", type=int, default=BOOTSTRAP_SAMPLES)
    parser.add_argument("--seed", type=int, default=BOOTSTRAP_SEED)
    parser.add_argument("--min-train-games", type=int, default=DEFAULT_MIN_TRAIN_GAMES)
    args = parser.parse_args()

    started = time.time()
    base_features = pd.read_parquet(args.features)
    features = build_combined_features(base_features)

    identity = {name: check_profile_identity(arm, features) for name, arm in ARMS.items()}

    leak = args.mode == "positive-control"
    base_scored = run_arm(
        features,
        ARMS["base"],
        market_root=args.market_root,
        min_train_games=args.min_train_games,
        leak=False,
    )

    result: dict[str, Any] = {"status": "scored", "profile_identity": identity}
    arm_scored: dict[str, pd.DataFrame] = {"base": base_scored}
    for name in CANDIDATE_ARM_NAMES:
        arm_scored[name] = run_arm(
            features,
            ARMS[name],
            market_root=args.market_root,
            min_train_games=args.min_train_games,
            leak=leak,
        )

    per_game_frames: dict[str, pd.DataFrame] = {}
    for name in CANDIDATE_ARM_NAMES:
        paired = confirmation.paired_frame(base_scored, arm_scored[name])
        if paired.empty:
            raise RuntimeError(f"No paired opener-grade games were scored for arm {name!r}")
        per_game_frames[name] = paired
        arm_result: dict[str, Any] = {
            "paired_games": len(paired),
            "paired_weeks": int(paired.groupby(["season", "week"]).ngroups),
            "paired_seasons": sorted(int(s) for s in paired["season"].unique()),
        }
        for label, reference, treatment_col in (
            ("opener_production_rule", "baseline_correct_open_pr", "candidate_correct_open_pr"),
            ("opener_sign_rule", "baseline_correct_open", "candidate_correct_open"),
            ("close_production_rule", "baseline_correct_close_pr", "candidate_correct_close_pr"),
            ("close_sign_rule", "baseline_correct_close", "candidate_correct_close"),
        ):
            arm_result[label] = confirmation.summarize(
                paired, reference, treatment_col, args.bootstrap_samples, args.seed
            )
        arm_result["picks_disagreeing_production_rule"] = int(
            (paired.baseline_pick_home_pr != paired.candidate_pick_home_pr).sum()
        )
        arm_result["picks_disagreeing_sign_rule"] = int(
            (paired.baseline_pick_home != paired.candidate_pick_home).sum()
        )
        if args.mode == "screen":
            arm_result["per_season_opener_production_rule"] = per_season_breakdown(paired)
            arm_result["candidate_metrics"] = confirmation.opener_evaluation_metrics(
                arm_scored[name]
            )
        result[name] = arm_result
    result["base_metrics"] = confirmation.opener_evaluation_metrics(base_scored)

    if args.mode == "screen":
        published_path = (
            args.published_predictions
            or sorted(
                (REPO_ROOT / "artifacts" / "margin_predictions").glob(
                    "2026-week-01-*/predictions.csv"
                )
            )[-1]
        )
        result["card_impact_2026_week1"] = card_impact(features, published_path)

    configuration = {
        "mode": args.mode,
        "grade": "opener",
        "baseline_profile": BASELINE_PROFILE,
        "arms": {
            name: {"profile": arm.profile, "columns": list(arm.columns)}
            for name, arm in ARMS.items()
        },
        "features_path": str(args.features),
        "market_root": str(args.market_root),
        "bootstrap_samples": args.bootstrap_samples,
        "seed": args.seed,
        "min_train_games": args.min_train_games,
        "rotation_window": None,
        "population": "full reused 2020-2025 opener archive (no rotation window; promotion look)",
    }
    payload = {
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(started)),
        "elapsed_seconds": round(time.time() - started, 1),
        **configuration,
        "result": result,
        "provenance": artifact_provenance(configuration, args.features, project_root=REPO_ROOT),
    }
    output = (
        REPO_ROOT / "artifacts" / "promotion_eval" / time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    )
    write_experiment_artifact(
        output,
        "results.json",
        payload,
        command="promotion-eval-20260905",
        metrics={"mode": args.mode, "status": "scored"},
        notes=(
            "Lane T promotion evaluation, docs/promotion_eval_20260905.md: "
            "REUSED-population promotion look (no rotation window) for "
            "weak_stack_qb_revenge, weak_stack_deadline_drag, and their "
            "union, on the rebuilt (ENG-39) production table."
        ),
    )
    for name, paired in per_game_frames.items():
        paired.to_csv(output / f"paired_predictions_{name}.csv", index=False)
    print(f"wrote {output / 'results.json'}")
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
