from __future__ import annotations

import json
from itertools import pairwise
from pathlib import Path

import numpy as np
import pandas as pd
from confidence_best_pick_sunday_matched import FEATURES, POPULATION, PREDICTIONS, push_candidates
from confidence_ranking_audit import blocked_mean, fit_predict, within_week_slope
from scipy.optimize import minimize
from scipy.special import expit, logit
from scipy.stats import binomtest

from nfl_ats.pick_probability_fit import FIT_FEATURES

OUTPUT = Path("artifacts/confidence_top_calibration/20260920_fixed")
ARMS = ("uncalibrated", "all_temperature", "nominee_temperature", "model", "market")
REPAIRS = ARMS[1:3]
BANDS = (0.5, 0.55, 0.60, 0.65, 1.000001)


def population() -> pd.DataFrame:
    current = pd.read_parquet(PREDICTIONS)
    frozen = pd.read_parquet(POPULATION)
    rows = frozen.merge(
        current[["game_id", "sunday_move_fitted"]], on="game_id", validate="one_to_one"
    )
    rows["market_move_toward_home"] = rows["sunday_move_fitted"]
    columns = ["game_id", "season", "week", "home_covered", "model_probability", *FIT_FEATURES]
    rows = pd.concat(
        [rows[columns], push_candidates(current, include_features=True)[columns]], ignore_index=True
    )
    metadata = pd.read_parquet(FEATURES, columns=["game_id", "game_type", "kickoff"])
    rows = rows.merge(metadata, on="game_id", validate="one_to_one")
    if len(rows) != 1537 or not rows["game_type"].eq("REG").all():
        raise ValueError("The declared full opener population changed")
    rows["kickoff"] = pd.to_datetime(rows["kickoff"], utc=True)
    first = rows.groupby(["season", "week"])["kickoff"].transform("min")
    local = first.dt.tz_convert("America/New_York").dt.tz_localize(None)
    sunday = local.dt.normalize() + pd.to_timedelta((6 - local.dt.weekday) % 7, unit="D")
    rows["as_of"] = (
        (sunday + pd.Timedelta(hours=12, minutes=45))
        .dt.tz_localize("America/New_York")
        .dt.tz_convert("UTC")
    )
    rows["eligible"] = rows["kickoff"].gt(rows["as_of"])
    return rows


def nominees(frame: pd.DataFrame, probability: str) -> pd.Index:
    confidence = np.maximum(frame[probability], 1.0 - frame[probability])
    ranked = frame.assign(confidence=confidence).sort_values(
        ["confidence", "game_id"], ascending=[False, True]
    )
    return ranked.groupby(["season", "week"], sort=True).head(1).index


def temperature(frame: pd.DataFrame) -> float:
    resolved = frame.dropna(subset=["home_covered"])
    x = logit(resolved["uncalibrated"].clip(1e-12, 1.0 - 1e-12).to_numpy())
    y = resolved["home_covered"].to_numpy(dtype=float)

    def objective(beta: np.ndarray) -> tuple[float, np.ndarray]:
        z = beta[0] * x
        loss = np.mean(np.logaddexp(0.0, z) - y * z)
        gradient = np.array([np.mean((expit(z) - y) * x)])
        return float(loss), gradient

    result = minimize(objective, np.array([1.0]), jac=True, bounds=[(1e-6, None)])
    if not result.success:
        raise ValueError(result.message)
    return float(result.x[0])


def build_predictions(rows: pd.DataFrame) -> tuple[pd.DataFrame, list[dict[str, object]]]:
    predictions = []
    coefficients = []
    for protocol in ("chronological", "leave_two_seasons_out"):
        for outer in (2023, 2024, 2025):
            calibration = outer - 1
            train_mask = (
                rows["season"].lt(calibration)
                if protocol == "chronological"
                else ~rows["season"].isin([calibration, outer])
            )
            train = rows.loc[train_mask].dropna(subset=["home_covered"])
            if train["season"].nunique() < 2:
                raise ValueError("Insufficient independent fitting seasons")
            evaluated = rows.loc[
                rows["eligible"] & rows["season"].isin([calibration, outer])
            ].copy()
            evaluated["uncalibrated"], natural = fit_predict(train, evaluated, FIT_FEATURES)
            cal = evaluated.loc[evaluated["season"].eq(calibration)]
            betas = {
                "all_temperature": temperature(cal),
                "nominee_temperature": temperature(cal.loc[nominees(cal, "uncalibrated")]),
            }
            for arm, beta in betas.items():
                evaluated[arm] = expit(
                    beta * logit(evaluated["uncalibrated"].clip(1e-12, 1.0 - 1e-12))
                )
            evaluated["model"] = evaluated["model_probability"]
            evaluated["market"] = 0.5
            evaluated["nominee"] = evaluated.index.isin(nominees(evaluated, "uncalibrated"))
            evaluated["model_nominee"] = evaluated.index.isin(nominees(evaluated, "model"))
            for arm in REPAIRS:
                if set(nominees(evaluated, arm)) != set(evaluated.index[evaluated["nominee"]]):
                    raise ValueError("Temperature changed the nominated game")
                if not evaluated[arm].ge(0.5).equals(evaluated["uncalibrated"].ge(0.5)):
                    raise ValueError("Temperature changed a side")
            evaluated["protocol"] = protocol
            evaluated["outer_season"] = outer
            evaluated["split"] = np.where(evaluated["season"].eq(outer), "outer", "calibration")
            predictions.append(evaluated)
            coefficients.append(
                {
                    "protocol": protocol,
                    "outer_season": outer,
                    "calibration_season": calibration,
                    "train_seasons": sorted(int(s) for s in train["season"].unique()),
                    "train_games": len(train),
                    "calibration_games": int(cal["home_covered"].notna().sum()),
                    "calibration_nominees": int(
                        cal.loc[nominees(cal, "uncalibrated"), "home_covered"].notna().sum()
                    ),
                    "combined_coefficients": natural,
                    "inverse_temperatures": betas,
                }
            )
    return pd.concat(predictions, ignore_index=True), coefficients


def score(frame: pd.DataFrame, arm: str) -> dict[str, object]:
    resolved = frame.dropna(subset=["home_covered"])
    p = resolved[arm].clip(1e-12, 1.0 - 1e-12).to_numpy()
    y = resolved["home_covered"].to_numpy(dtype=float)
    confidence = np.maximum(p, 1.0 - p)
    correct = (p >= 0.5) == y
    bands = []
    for lower, upper in pairwise(BANDS):
        mask = (confidence >= lower) & (confidence < upper)
        if mask.any():
            bands.append(
                {
                    "lower": lower,
                    "upper": min(upper, 1.0),
                    "games": int(mask.sum()),
                    "predicted": float(confidence[mask].mean()),
                    "actual": None if arm == "market" else float(correct[mask].mean()),
                    "gap_points": None
                    if arm == "market"
                    else blocked_mean(
                        resolved.loc[mask], 100.0 * (correct[mask] - confidence[mask])
                    ),
                }
            )
    return {
        "candidates": len(frame),
        "pushes": int(frame["home_covered"].isna().sum()),
        "games": len(resolved),
        "weeks": len(resolved.groupby(["season", "week"])),
        "brier": float(np.mean((p - y) ** 2)),
        "log_loss": float(np.mean(np.logaddexp(0.0, logit(p)) - y * logit(p))),
        "accuracy": None if arm == "market" else float(correct.mean()),
        "mean_confidence": float(confidence.mean()),
        "calibration_gap_points": None
        if arm == "market"
        else blocked_mean(resolved, 100.0 * (correct - confidence)),
        "reliability": bands,
    }


def comparison(frame: pd.DataFrame, arm: str) -> dict[str, object]:
    resolved = frame.dropna(subset=["home_covered"])
    p = resolved[arm].clip(1e-12, 1.0 - 1e-12).to_numpy()
    raw = resolved["uncalibrated"].clip(1e-12, 1.0 - 1e-12).to_numpy()
    y = resolved["home_covered"].to_numpy(dtype=float)
    correct = (p >= 0.5) == y
    raw_loss = np.logaddexp(0.0, logit(raw)) - y * logit(raw)
    loss = np.logaddexp(0.0, logit(p)) - y * logit(p)
    return {
        "games": len(resolved),
        "weeks": len(resolved.groupby(["season", "week"])),
        "brier_improvement": blocked_mean(resolved, (raw - y) ** 2 - (p - y) ** 2),
        "log_loss_improvement": blocked_mean(resolved, raw_loss - loss),
        "calibration_gap_points": blocked_mean(resolved, 100.0 * (correct - np.maximum(p, 1 - p))),
    }


def ranking(frame: pd.DataFrame) -> dict[str, object]:
    resolved = frame.dropna(subset=["home_covered"]).copy()
    resolved["raw_confidence"] = np.maximum(resolved["uncalibrated"], 1 - resolved["uncalibrated"])
    resolved["raw_correct"] = (
        resolved["uncalibrated"].ge(0.5).eq(resolved["home_covered"]).astype(float)
    )
    selected = frame.loc[frame["nominee"]].copy()
    model = frame.loc[frame["model_nominee"]].copy()
    selected["points"] = np.where(
        selected["home_covered"].isna(),
        0.5,
        selected["uncalibrated"].ge(0.5).eq(selected["home_covered"]).astype(float),
    )
    model["points"] = np.where(
        model["home_covered"].isna(),
        0.5,
        model["model"].ge(0.5).eq(model["home_covered"]).astype(float),
    )
    paired = selected.merge(model, on=["season", "week"], suffixes=("_raw", "_model"))
    delta = paired["points_raw"] - paired["points_model"]
    wins = int(delta.gt(0).sum())
    losses = int(delta.lt(0).sum())
    week_scores = resolved.groupby(["season", "week", "nominee"])["raw_correct"].mean().unstack()
    paired_weeks = week_scores.dropna().reset_index()
    return {
        "temperature_nominee_changes": 0,
        "temperature_side_changes": 0,
        "temperature_decisive_games": 0,
        "model_nominee_changes": int(paired["game_id_raw"].ne(paired["game_id_model"]).sum()),
        "model_decisive_weeks": wins + losses,
        "combined_wins": wins,
        "model_wins": losses,
        "decisive_exact_two_sided_p": float(binomtest(wins, wins + losses).pvalue)
        if wins + losses
        else None,
        "within_week_slope_accuracy_points_per_10_confidence_points": within_week_slope(
            resolved, "raw"
        ),
        "nominee_minus_rest_accuracy_points": blocked_mean(
            paired_weeks, 100.0 * (paired_weeks[True] - paired_weeks[False]).to_numpy()
        ),
        "model_own_nominees": score(model, "model"),
    }


def main() -> None:
    rows = population()
    predictions, coefficients = build_predictions(rows)
    summary: dict[str, object] = {
        "family": "confidence_top_calibration_20260920",
        "bootstrap_draws": 10000,
        "bootstrap_seed": 20260920,
        "look_count": {"paired_brier": 8, "paired_log_loss": 8, "calibration_gap": 8},
        "coefficients": coefficients,
        "protocols": {},
    }
    protocols = {}
    for protocol, protocol_frame in predictions.groupby("protocol"):
        outer = protocol_frame.loc[protocol_frame["split"].eq("outer")]
        scopes = {"all": outer, "nominees": outer.loc[outer["nominee"]]}
        result = {
            "ranking": ranking(outer),
            "scores": {
                scope: {arm: score(frame, arm) for arm in ARMS} for scope, frame in scopes.items()
            },
            "comparisons": {
                scope: {arm: comparison(frame, arm) for arm in REPAIRS}
                for scope, frame in scopes.items()
            },
            "folds": [],
        }
        for season, fold in protocol_frame.groupby("outer_season"):
            splits = {}
            for split, sample in fold.groupby("split"):
                splits[split] = {
                    scope: {arm: score(frame, arm) for arm in ARMS}
                    for scope, frame in {
                        "all": sample,
                        "nominees": sample.loc[sample["nominee"]],
                    }.items()
                }
            gaps = {
                scope: {
                    arm: {
                        metric: splits["outer"][scope][arm][metric]
                        - splits["calibration"][scope][arm][metric]
                        for metric in ("brier", "log_loss")
                    }
                    for arm in ARMS
                }
                for scope in scopes
            }
            result["folds"].append(
                {"outer_season": int(season), "splits": splits, "outer_minus_calibration": gaps}
            )
        protocols[str(protocol)] = result
    summary["protocols"] = protocols
    OUTPUT.mkdir(parents=True, exist_ok=True)
    predictions.to_parquet(OUTPUT / "per_game.parquet", index=False)
    (OUTPUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(OUTPUT),
                "comparisons": {p: v["comparisons"]["nominees"] for p, v in protocols.items()},
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
