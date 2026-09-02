"""PER-13 Stage 1: does a per-player durability prior sharpen P(plays)?

Runs the comparison frozen in ``docs/per13_durability_prior.md`` sections 4-8.
Stage 1 lives entirely on the player-level availability target; it spends no
ATS window and produces no ATS row.

    ./.tools/uv.exe run --no-sync python scripts/per13_durability_stage1.py \
        --mode positive-control
    ./.tools/uv.exe run --no-sync python scripts/per13_durability_stage1.py \
        --mode screen

Artifacts land in ``artifacts/per13_durability_stage1/<stamp>/``.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

from nfl_ats.availability import (  # noqa: E402
    build_availability_outcomes,
    build_season_lagged_availability_rates,
    position_group,
    score_availability_rates,
)
from nfl_ats.clv import week_blocked_bootstrap  # noqa: E402
from nfl_ats.durability_prior import (  # noqa: E402
    DURABILITY_COLUMNS,
    DURABILITY_PRIOR_VERSION,
    DurabilityHistory,
    durability_prior_columns,
    split_half_reliability,
)
from nfl_ats.players import (  # noqa: E402
    attach_snap_player_ids,
    canonicalize_injuries,
    canonicalize_rosters,
    canonicalize_snaps,
    latest_player_snapshot,
    load_player_snapshot,
)
from nfl_ats.provenance import write_experiment_artifact  # noqa: E402

DECISION_HOURS = 24
MIN_TRAIN_ROWS = 2_000
BOOTSTRAP_SAMPLES = 2_000
BOOTSTRAP_SEED = 20260901
POSITIVE_CONTROL_BRIER_CEILING = 0.02
PROBABILITY_FLOOR = 1e-6


# ---------------------------------------------------------------------------
# Data assembly
# ---------------------------------------------------------------------------


def load_frames() -> dict[str, Any]:
    """Reproduce the incumbent learned-availability pipeline exactly, then extend it."""

    features = pd.read_parquet(REPO_ROOT / "data/processed/game_features_pbp.parquet")
    snapshot = latest_player_snapshot(REPO_ROOT / "data/players/raw")
    injuries, rosters, snaps = load_player_snapshot(snapshot)
    canonical_injuries = canonicalize_injuries(injuries)
    canonical_rosters = canonicalize_rosters(rosters)
    canonical_snaps = attach_snap_player_ids(canonicalize_snaps(snaps), canonical_rosters)

    outcomes = build_availability_outcomes(
        canonical_injuries,
        canonical_snaps,
        features,
        decision_hours_before_kickoff=DECISION_HOURS,
    )
    rates = build_season_lagged_availability_rates(
        outcomes, target_seasons=sorted(features["season"].astype(int).unique())
    )
    scored = score_availability_rates(outcomes, rates)
    return {
        "snapshot_id": snapshot.snapshot_id,
        "features": features,
        "rosters": canonical_rosters,
        "snaps": canonical_snaps,
        "outcomes": outcomes,
        "rates": rates,
        "scored": scored,
    }


def build_history(frames: dict[str, Any]) -> DurabilityHistory:
    """Assemble the point-in-time outcome and roster history tables."""

    features = frames["features"]
    kickoffs = features.loc[features["game_id"].notna(), ["game_id", "kickoff"]].drop_duplicates(
        "game_id"
    )
    outcomes = frames["outcomes"].merge(kickoffs, on="game_id", how="left", validate="many_to_one")
    if outcomes["kickoff"].isna().any():
        raise RuntimeError("availability outcomes carry a game without a kickoff timestamp")
    key = ["season", "week", "team", "gsis_id"]
    cell = frames["scored"][[*key, "learned_unavailability"]].rename(
        columns={"learned_unavailability": "cell_probability"}
    )
    outcomes = outcomes.merge(cell, on=key, how="left", validate="one_to_one")

    snaps = frames["snaps"]
    snap_seasons = sorted({int(value) for value in snaps["season"].dropna().unique()})
    played = (
        snaps.loc[snaps["gsis_id"].notna()]
        .assign(
            total=lambda block: (
                block[["offense_snaps", "defense_snaps", "st_snaps"]].fillna(0.0).sum(axis=1)
            )
        )
        .groupby(["season", "week", "team", "gsis_id"], observed=True)["total"]
        .max()
        .gt(0.0)
        .rename("played")
        .reset_index()
    )
    rosters = frames["rosters"].merge(
        played, on=["season", "week", "team", "gsis_id"], how="left", validate="many_to_one"
    )
    rosters["played"] = rosters["played"].fillna(False).astype(bool)
    rosters["snap_covered"] = rosters["season"].isin(snap_seasons)
    rosters["position_group"] = rosters["position"].map(position_group)
    return DurabilityHistory(outcomes=outcomes, rosters=rosters)


def target_rows(frames: dict[str, Any]) -> pd.DataFrame:
    """The 57,294-style out-of-season evaluation frame, with its decision cutoffs."""

    features = frames["features"]
    kickoffs = features.loc[features["game_id"].notna(), ["game_id", "kickoff"]].drop_duplicates(
        "game_id"
    )
    rows = frames["scored"].merge(kickoffs, on="game_id", how="left", validate="many_to_one")
    rows["kickoff"] = pd.to_datetime(rows["kickoff"], errors="coerce", utc=True)
    rows["decision_cutoff"] = rows["kickoff"] - pd.Timedelta(hours=DECISION_HOURS)
    rows["gsis_id"] = rows["gsis_id"].astype(str)
    rows["position_group"] = rows["position_group"].astype(str)
    rows["unavailable"] = rows["unavailable"].astype(float)
    rows["cell_probability"] = rows["learned_unavailability"].astype(float)
    return rows.sort_values(["season", "week", "game_id", "team", "gsis_id"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def brier(probability: np.ndarray, outcome: np.ndarray) -> float:
    return float(np.square(probability - outcome).mean())


def log_loss(probability: np.ndarray, outcome: np.ndarray) -> float:
    clipped = np.clip(probability, PROBABILITY_FLOOR, 1.0 - PROBABILITY_FLOOR)
    return float(-(outcome * np.log(clipped) + (1.0 - outcome) * np.log(1.0 - clipped)).mean())


def paired_bootstrap(frame: pd.DataFrame, columns: list[str], block: str) -> pd.DataFrame:
    def metric(sample: pd.DataFrame) -> dict[str, float]:
        return {name: float(sample[name].mean()) for name in columns}

    return week_blocked_bootstrap(
        frame,
        metric,
        block=block,  # type: ignore[arg-type]
        samples=BOOTSTRAP_SAMPLES,
        seed=BOOTSTRAP_SEED,
    )


# ---------------------------------------------------------------------------
# The fold loop
# ---------------------------------------------------------------------------


def run_folds(
    rows: pd.DataFrame,
    aggregates: pd.DataFrame,
    history: DurabilityHistory,
    *,
    arms: dict[str, list[str]],
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """Expanding chronological folds: fit on ``season < S``, evaluate on ``season == S``."""

    seasons = sorted({int(value) for value in rows["season"].unique()})
    outcome = rows["unavailable"].to_numpy(dtype=float)
    base_logit = np.log(
        np.clip(
            rows["cell_probability"].to_numpy(dtype=float), PROBABILITY_FLOOR, 1 - PROBABILITY_FLOOR
        )
        / np.clip(
            1.0 - rows["cell_probability"].to_numpy(dtype=float),
            PROBABILITY_FLOOR,
            1 - PROBABILITY_FLOOR,
        )
    )
    predictions: list[pd.DataFrame] = []
    calibrations: list[dict[str, Any]] = []
    for season in seasons:
        train_mask = (rows["season"] < season).to_numpy()
        eval_mask = (rows["season"] == season).to_numpy()
        if int(train_mask.sum()) < MIN_TRAIN_ROWS:
            continue
        calibration = history.calibration(before_season=season)
        columns = durability_prior_columns(aggregates, calibration)
        design = pd.DataFrame({"base_logit": base_logit}, index=rows.index)
        design = pd.concat([design, columns], axis=1)
        design["leaked_played"] = 1.0 - outcome

        block = pd.DataFrame(
            {
                "season": rows.loc[eval_mask, "season"].to_numpy(),
                "week": rows.loc[eval_mask, "week"].to_numpy(),
                "gsis_id": rows.loc[eval_mask, "gsis_id"].to_numpy(),
                "position_group": rows.loc[eval_mask, "position_group"].to_numpy(),
                "unavailable": outcome[eval_mask],
                "probability_raw": rows.loc[eval_mask, "cell_probability"].to_numpy(),
            }
        )
        for arm, arm_columns in arms.items():
            scaler = StandardScaler()
            train_x = scaler.fit_transform(
                design.loc[train_mask, arm_columns].to_numpy(dtype=float)
            )
            model = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000)
            model.fit(train_x, outcome[train_mask])
            eval_x = scaler.transform(design.loc[eval_mask, arm_columns].to_numpy(dtype=float))
            block[f"probability_{arm}"] = model.predict_proba(eval_x)[:, 1]
        predictions.append(block)
        record = calibration.as_dict()
        record["train_rows"] = int(train_mask.sum())
        record["eval_rows"] = int(eval_mask.sum())
        calibrations.append(record)
    if not predictions:
        raise RuntimeError("no fold had enough training rows")
    return pd.concat(predictions, ignore_index=True), calibrations


# ---------------------------------------------------------------------------
# Feasibility (history depth and trait reliability)
# ---------------------------------------------------------------------------


def feasibility(
    frames: dict[str, Any], rows: pd.DataFrame, aggregates: pd.DataFrame
) -> dict[str, Any]:
    depth = aggregates["rate_n"].to_numpy(dtype=float)
    scored = rows.copy()
    scored["residual"] = scored["unavailable"] - scored["cell_probability"]
    ordered = scored.sort_values(["season", "week"]).reset_index(drop=True)
    return {
        "outcome_rows": len(frames["outcomes"]),
        "scored_rows": len(rows),
        "unique_players": int(frames["outcomes"]["gsis_id"].nunique()),
        "seasons": [int(value) for value in sorted(rows["season"].unique())],
        "prior_appearances": {
            "mean": float(depth.mean()),
            "median": float(np.median(depth)),
            "share_at_least_1": float((depth >= 1).mean()),
            "share_at_least_3": float((depth >= 3).mean()),
            "share_at_least_5": float((depth >= 5).mean()),
            "share_at_least_10": float((depth >= 10).mean()),
            "share_at_least_20": float((depth >= 20).mean()),
        },
        "prior_appearances_by_season": {
            str(int(season)): {
                "mean": float(block.mean()),
                "median": float(block.median()),
                "share_at_least_5": float((block >= 5).mean()),
            }
            for season, block in pd.Series(depth, index=rows.index).groupby(rows["season"])
        },
        "reliability_raw_rate": {
            str(minimum): split_half_reliability(ordered, "unavailable", minimum_per_half=minimum)
            for minimum in (5, 10, 20)
        },
        "reliability_residual": {
            str(minimum): split_half_reliability(ordered, "residual", minimum_per_half=minimum)
            for minimum in (5, 10, 20)
        },
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def summarize(predictions: pd.DataFrame, baseline: str, candidate: str) -> dict[str, Any]:
    outcome = predictions["unavailable"].to_numpy(dtype=float)
    base = predictions[f"probability_{baseline}"].to_numpy(dtype=float)
    cand = predictions[f"probability_{candidate}"].to_numpy(dtype=float)
    paired = predictions[["season", "week"]].copy()
    paired["brier_improvement"] = np.square(base - outcome) - np.square(cand - outcome)
    clipped_base = np.clip(base, PROBABILITY_FLOOR, 1 - PROBABILITY_FLOOR)
    clipped_cand = np.clip(cand, PROBABILITY_FLOOR, 1 - PROBABILITY_FLOOR)
    paired["log_loss_improvement"] = -(
        outcome * np.log(clipped_base) + (1 - outcome) * np.log(1 - clipped_base)
    ) + (outcome * np.log(clipped_cand) + (1 - outcome) * np.log(1 - clipped_cand))
    columns = ["brier_improvement", "log_loss_improvement"]
    week = paired_bootstrap(paired, columns, "week")
    season = paired_bootstrap(paired, columns, "season")
    return {
        "baseline_arm": baseline,
        "candidate_arm": candidate,
        "rows": len(predictions),
        "week_blocks": int(predictions.groupby(["season", "week"]).ngroups),
        "season_blocks": int(predictions["season"].nunique()),
        "baseline_brier": brier(base, outcome),
        "candidate_brier": brier(cand, outcome),
        "baseline_log_loss": log_loss(base, outcome),
        "candidate_log_loss": log_loss(cand, outcome),
        "week_blocked": week.to_dict(orient="records"),
        "season_blocked": season.to_dict(orient="records"),
    }


def breakdown(predictions: pd.DataFrame, baseline: str, candidate: str, key: str) -> pd.DataFrame:
    outcome = predictions["unavailable"].to_numpy(dtype=float)
    working = predictions.copy()
    working["brier_base"] = np.square(
        working[f"probability_{baseline}"].to_numpy(dtype=float) - outcome
    )
    working["brier_candidate"] = np.square(
        working[f"probability_{candidate}"].to_numpy(dtype=float) - outcome
    )
    grouped = working.groupby(key, observed=True).agg(
        rows=("unavailable", "size"),
        brier_base=("brier_base", "mean"),
        brier_candidate=("brier_candidate", "mean"),
    )
    grouped["improvement"] = grouped["brier_base"] - grouped["brier_candidate"]
    return grouped.reset_index()


def diagnostics(
    rows: pd.DataFrame, aggregates: pd.DataFrame, history: DurabilityHistory
) -> dict[str, Any]:
    """POST-HOC, and outside the frozen comparison. Three questions the size of
    the screen's result makes it irresponsible not to ask.

    1. *Placebo.* Permute the aggregate rows so every player is handed some
       other player's history. If the harness is crediting the candidate arm
       for merely having more columns, the improvement survives; if the gain is
       real row-level information, it vanishes.
    2. *Single-column and leave-one-out ablation.* Which of the six carries it,
       and how much is redundant.
    3. *Prior seasons only.* Re-derive every column with same-season history
       removed, so what is left is the multi-season durability trait PER-13
       actually names rather than a within-season absence streak.
    """

    base = ["base_logit"]
    evaluated: list[int] = []

    def arm_brier(columns: list[str], source: pd.DataFrame) -> float:
        predictions, _ = run_folds(rows, source, history, arms={"arm": columns})
        evaluated.append(len(predictions))
        return brier(
            predictions["probability_arm"].to_numpy(dtype=float),
            predictions["unavailable"].to_numpy(dtype=float),
        )

    baseline = arm_brier(base, aggregates)
    everything = arm_brier([*base, *DURABILITY_COLUMNS], aggregates)
    single = {name: baseline - arm_brier([*base, name], aggregates) for name in DURABILITY_COLUMNS}
    leave_one_out = {
        name: arm_brier(
            [*base, *[other for other in DURABILITY_COLUMNS if other != name]], aggregates
        )
        - everything
        for name in DURABILITY_COLUMNS
    }

    generator = np.random.default_rng(BOOTSTRAP_SEED)
    shuffled = aggregates.iloc[generator.permutation(len(aggregates))].set_axis(aggregates.index)
    placebo = arm_brier([*base, *DURABILITY_COLUMNS], shuffled)

    lagged = rows.copy()
    lagged["decision_cutoff"] = rows.groupby("season")["kickoff"].transform("min")
    lagged["week"] = 0
    prior_seasons = history.aggregates(lagged)
    prior_seasons_only = arm_brier([*base, *DURABILITY_COLUMNS], prior_seasons)

    return {
        "note": "post-hoc; not part of the frozen comparison in docs/per13_durability_prior.md",
        "target_rows_available": len(rows),
        "evaluated_rows": max(evaluated),
        "baseline_brier": baseline,
        "candidate_brier": everything,
        "single_column_improvement": single,
        "leave_one_out_brier_cost": leave_one_out,
        "placebo_permuted_brier": placebo,
        "placebo_improvement": baseline - placebo,
        "prior_seasons_only_brier": prior_seasons_only,
        "prior_seasons_only_improvement": baseline - prior_seasons_only,
        "prior_seasons_only_share_without_history": float(
            (prior_seasons["rate_n"].to_numpy(dtype=float) == 0).mean()
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode", choices=("positive-control", "screen", "diagnostics"), required=True
    )
    parser.add_argument("--output-root", type=Path, default=REPO_ROOT / "artifacts")
    args = parser.parse_args()

    started = datetime.now(UTC)
    stamp = started.strftime("%Y%m%dT%H%M%SZ")
    destination = args.output_root / "per13_durability_stage1" / stamp
    destination.mkdir(parents=True, exist_ok=True)

    frames = load_frames()
    history = build_history(frames)
    rows = target_rows(frames)
    aggregates = history.aggregates(rows)

    incumbent_brier = brier(
        rows["cell_probability"].to_numpy(dtype=float), rows["unavailable"].to_numpy(dtype=float)
    )

    if args.mode == "diagnostics":
        report = diagnostics(rows, aggregates, history)
        write_experiment_artifact(
            destination,
            "diagnostics.json",
            report,
            command="per13-durability-stage1",
            metrics={"mode": args.mode},
            notes="PER-13 stage 1 durability prior; see docs/per13_durability_prior.md.",
        )
        print(json.dumps(report, indent=2, sort_keys=True, default=float))
        print(f"\nartifacts: {destination}")
        return 0

    if args.mode == "positive-control":
        arms = {
            "baseline": ["base_logit"],
            "control": ["base_logit", "leaked_played"],
        }
        baseline_arm, candidate_arm = "baseline", "control"
    else:
        arms = {
            "baseline": ["base_logit"],
            "candidate": ["base_logit", *DURABILITY_COLUMNS],
        }
        baseline_arm, candidate_arm = "baseline", "candidate"

    predictions, calibrations = run_folds(rows, aggregates, history, arms=arms)
    outcome = predictions["unavailable"].to_numpy(dtype=float)
    summary = summarize(predictions, baseline_arm, candidate_arm)
    summary["incumbent_unfitted_brier_full_frame"] = incumbent_brier
    summary["incumbent_unfitted_brier_evaluated_rows"] = brier(
        predictions["probability_raw"].to_numpy(dtype=float), outcome
    )
    summary["incumbent_unfitted_log_loss_evaluated_rows"] = log_loss(
        predictions["probability_raw"].to_numpy(dtype=float), outcome
    )

    if args.mode == "positive-control":
        summary["control_ceiling"] = POSITIVE_CONTROL_BRIER_CEILING
        summary["control_passes"] = bool(
            summary["candidate_brier"] < POSITIVE_CONTROL_BRIER_CEILING
        )
    else:
        predictions_vs_raw = predictions.copy()
        summary["candidate_vs_raw_incumbent"] = summarize(
            predictions_vs_raw.rename(columns={"probability_raw": "probability_rawarm"}),
            "rawarm",
            candidate_arm,
        )
        breakdown(predictions, baseline_arm, candidate_arm, "season").to_csv(
            destination / "by_season.csv", index=False
        )
        breakdown(predictions, baseline_arm, candidate_arm, "position_group").to_csv(
            destination / "by_position_group.csv", index=False
        )
        write_experiment_artifact(
            destination,
            "feasibility.json",
            feasibility(frames, rows, aggregates),
            command="per13-durability-stage1",
            metrics={"mode": args.mode},
            notes="PER-13 stage 1 durability prior; see docs/per13_durability_prior.md.",
        )

    predictions.to_parquet(destination / "predictions.parquet", index=False)
    metadata = {
        "command": "per13-durability-stage1",
        "mode": args.mode,
        "created_at_utc": started.isoformat(),
        "durability_prior_version": DURABILITY_PRIOR_VERSION,
        "player_snapshot": frames["snapshot_id"],
        "decision_hours_before_kickoff": DECISION_HOURS,
        "min_train_rows": MIN_TRAIN_ROWS,
        "bootstrap_samples": BOOTSTRAP_SAMPLES,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "arms": {name: list(columns) for name, columns in arms.items()},
        "durability_columns": list(DURABILITY_COLUMNS),
        "scored_rows_available": len(rows),
        "hypothesis_frozen_before_scoring": True,
        "predeclaration": "docs/per13_durability_prior.md",
        "summary": summary,
        "fold_calibrations": calibrations,
    }
    write_experiment_artifact(
        destination,
        "metadata.json",
        metadata,
        command="per13-durability-stage1",
        metrics={"mode": args.mode},
        notes="PER-13 stage 1 durability prior; see docs/per13_durability_prior.md.",
    )
    print(json.dumps(summary, indent=2, sort_keys=True, default=float))
    print(f"\nartifacts: {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
