from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from confidence_ranking_audit import blocked_mean, within_week_slope
from scipy.stats import binomtest
from sunday_market_probability_eval import QUOTE_CACHE, sunday_move

from nfl_ats.pick_probability import PROBABILITY_EPSILON, signed_composition_flags
from nfl_ats.pick_probability_fit import (
    SCHEDULE_COLUMNS,
    _arrest_incidents,
    _forecast_temperatures,
    _protection_back_side,
)
from nfl_ats.snapshots import latest_snapshot, load_snapshot

PREDICTIONS = Path("artifacts/sunday_market_probability/20260920_fixed/predictions.parquet")
ALPHA = Path("artifacts/ridge_alpha_promotion/20260818T221459Z/opener_paired.parquet")
FEATURES = Path("data/processed/game_features.parquet")
OPENER = Path("artifacts/opener_evaluation/20260920T135435Z/per_game.parquet")
POPULATION = Path("artifacts/confidence_ranking_audit/20260920_aligned/population.parquet")
COEFFICIENTS = Path("artifacts/sunday_market_probability/20260920_fixed/coefficients.csv")
DATA_ROOT = Path("data")
OUTPUT = Path("artifacts/confidence_best_pick_sunday_matched/20260920_fixed")
PROTOCOLS = ("leave_one_season_out", "chronological")


def push_candidates(current: pd.DataFrame, *, include_features: bool = False) -> pd.DataFrame:
    opener = pd.read_parquet(OPENER)
    population = pd.read_parquet(POPULATION)
    pushes = opener.loc[opener.margin_vs_open.eq(0.0)].copy()
    if len(pushes) != 34 or len(population) + len(pushes) != len(opener):
        raise ValueError("Full opener pool does not reconcile with frozen nonpush predictions")
    if set(current.game_id) != set(population.game_id):
        raise ValueError("Frozen Sunday predictions differ from aligned nonpush population")
    schedules, _ = load_snapshot(latest_snapshot(DATA_ROOT / "raw"))
    schedule_columns = [column for column in SCHEDULE_COLUMNS if column in schedules.columns]
    all_rows = opener.merge(
        schedules[schedule_columns].drop_duplicates("game_id"),
        on=["game_id", "season"],
        how="left",
        suffixes=("", "_schedule"),
    )
    if "week_schedule" in all_rows:
        all_rows = all_rows.drop(columns="week_schedule")
    if all_rows.home_team.isna().any():
        raise ValueError("Full opener pool is missing historical schedule data")
    flags = signed_composition_flags(
        all_rows,
        schedules,
        incidents=_arrest_incidents(DATA_ROOT),
        forecasts_tuesday_noon=_forecast_temperatures(DATA_ROOT),
        protection_back_side=_protection_back_side(schedules, DATA_ROOT),
    )
    all_rows = all_rows.merge(flags, on="game_id", validate="one_to_one")
    check = all_rows.merge(
        population[["game_id", "composition_flag_sum"]],
        on="game_id",
        how="inner",
        suffixes=("", "_frozen"),
        validate="one_to_one",
    )
    if (
        len(check) != len(population)
        or not check.composition_flag_sum.eq(check.composition_flag_sum_frozen).all()
    ):
        raise ValueError("Reconstructed historical composition flags fail nonpush parity")
    pushes = all_rows.loc[all_rows.margin_vs_open.eq(0.0)].copy()
    quote_cache = pd.read_parquet(QUOTE_CACHE)
    quote_cache = quote_cache.loc[
        quote_cache.decision_label.eq("intraday_hourly")
        & quote_cache.archive_season.isin((2023, 2024, 2025))
        & quote_cache.nflverse_game_id.isin(set(opener.game_id))
    ].copy()
    games = quote_cache.groupby("nflverse_game_id", as_index=False).agg(
        commence_time_utc=("commence_time_utc", "min")
    )
    games = games.rename(columns={"nflverse_game_id": "game_id"}).merge(
        opener[["game_id", "season", "week"]], on="game_id", validate="one_to_one"
    )
    games["week_first_commence_utc"] = games.groupby(
        ["season", "week"]
    ).commence_time_utc.transform("min")
    moves = sunday_move(quote_cache, games)
    pushes = pushes.merge(moves, on="game_id", how="left", validate="one_to_one")
    exposed = pushes.season.ge(2023)
    if pushes.loc[exposed, "sunday_move"].isna().any():
        raise ValueError("Archived Sunday movement is missing for push candidates")
    pushes["market_move_available"] = exposed.astype(float)
    pushes["market_move_toward_home"] = pushes.sunday_move.fillna(0.0)
    pushes["model_probability"] = pushes.home_cover_probability_at_open.clip(
        PROBABILITY_EPSILON, 1.0 - PROBABILITY_EPSILON
    )
    pushes["model_logit"] = np.log(pushes.model_probability / (1.0 - pushes.model_probability))
    coefficients = pd.read_csv(COEFFICIENTS)
    frozen_check = population.merge(
        current[["game_id", "sunday_move_fitted", *[f"{p}_sunday_p" for p in PROTOCOLS]]],
        on="game_id",
        validate="one_to_one",
    )
    for protocol in PROTOCOLS:
        pushes[f"{protocol}_sunday_p"] = np.nan
        for _, coef in coefficients.loc[
            coefficients.protocol.eq(protocol) & coefficients.arm.eq("sunday")
        ].iterrows():
            existing = frozen_check.season.eq(coef.held)
            z_existing = (
                coef.intercept
                + coef.model_logit * frozen_check.loc[existing, "model_logit"]
                + coef.composition_flag_sum * frozen_check.loc[existing, "composition_flag_sum"]
                + coef.market_move_toward_home * frozen_check.loc[existing, "sunday_move_fitted"]
                + coef.market_move_available * frozen_check.loc[existing, "market_move_available"]
            )
            predicted = 1.0 / (1.0 + np.exp(-np.clip(z_existing, -35.0, 35.0)))
            frozen = frozen_check.loc[existing, f"{protocol}_sunday_p"]
            if frozen.notna().any() and np.max(np.abs(predicted - frozen)) > 1e-9:
                raise ValueError("Frozen coefficients fail nonpush prediction parity")
            held = pushes.season.eq(coef.held)
            z = (
                coef.intercept
                + coef.model_logit * pushes.loc[held, "model_logit"]
                + coef.composition_flag_sum * pushes.loc[held, "composition_flag_sum"]
                + coef.market_move_toward_home * pushes.loc[held, "market_move_toward_home"]
                + coef.market_move_available * pushes.loc[held, "market_move_available"]
            )
            pushes.loc[held, f"{protocol}_sunday_p"] = 1.0 / (
                1.0 + np.exp(-np.clip(z, -35.0, 35.0))
            )
    feature_columns = (
        ["model_logit", "composition_flag_sum", "market_move_toward_home", "market_move_available"]
        if include_features
        else []
    )
    return pushes[
        [
            "game_id",
            "season",
            "week",
            "model_probability",
            "leave_one_season_out_sunday_p",
            "chronological_sunday_p",
            *feature_columns,
        ]
    ].assign(home_covered=np.nan)


def main() -> None:
    current = pd.read_parquet(PREDICTIONS)
    current = pd.concat([current, push_candidates(current)], ignore_index=True)
    alpha = pd.read_parquet(ALPHA, columns=["game_id", "candidate_prob_open"])
    features = pd.read_parquet(
        FEATURES, columns=["game_id", "season", "week", "game_type", "kickoff"]
    )
    for label, frame in (("current", current), ("alpha", alpha), ("features", features)):
        if frame.game_id.duplicated().any():
            raise ValueError(f"{label} has duplicate game IDs")
    frame = current.merge(alpha, on="game_id", how="left", validate="one_to_one").merge(
        features, on=["game_id", "season", "week"], how="left", validate="one_to_one"
    )
    if (
        len(frame) != len(current)
        or frame[["candidate_prob_open", "kickoff", "game_type"]].isna().any().any()
    ):
        raise ValueError("Matched ranking requires a complete full-game join")
    if not frame.game_type.eq("REG").all() or not frame.candidate_prob_open.between(0, 1).all():
        raise ValueError("Matched ranking requires REG games and finite alpha probabilities")
    frame["kickoff"] = pd.to_datetime(frame.kickoff, utc=True)
    first = (
        frame.groupby(["season", "week"]).kickoff.transform("min").dt.tz_convert("America/New_York")
    )
    sunday = first.dt.tz_localize(None).dt.normalize() + pd.to_timedelta(
        (6 - first.dt.weekday) % 7, unit="D"
    )
    frame["decision_as_of_utc"] = (
        (sunday + pd.Timedelta(hours=12, minutes=45))
        .dt.tz_localize("America/New_York")
        .dt.tz_convert("UTC")
    )
    frame["eligible"] = frame.kickoff.gt(frame.decision_as_of_utc)
    if not frame.groupby(["season", "week"]).eligible.any().all():
        raise ValueError("At least one historical week lacks a Sunday-eligible game")
    outputs = []
    weekly_rows = []
    summaries = {}
    for protocol in PROTOCOLS:
        p = f"{protocol}_sunday_p"
        rows = frame.loc[frame[p].notna()].copy()
        rows["current_pick_home"] = rows[p].ge(0.5)
        rows["current_confidence"] = np.maximum(rows[p], 1.0 - rows[p])
        rows["current_grade"] = np.where(
            rows.home_covered.isna(),
            "P",
            np.where(rows.current_pick_home.eq(rows.home_covered.eq(1.0)), "W", "L"),
        )
        rows["current_correct"] = rows.current_grade.map({"W": 1.0, "L": 0.0})
        rows["current_points"] = rows.current_grade.map({"W": 1.0, "P": 0.5, "L": 0.0})
        rows["alpha_current_side_p"] = rows.candidate_prob_open.where(
            rows.current_pick_home, 1.0 - rows.candidate_prob_open
        )
        rows["protocol"] = protocol
        outputs.append(rows)
        eligible = rows.loc[rows.eligible].copy()
        selected = []
        for (season, week), group in eligible.groupby(["season", "week"]):
            if len(group) < 2:
                raise ValueError("Within-week ranking requires at least two eligible games")
            by_current = group.sort_values(
                ["current_confidence", "game_id"], ascending=[False, True]
            ).iloc[0]
            by_alpha = (
                group.sort_values(
                    ["alpha_current_side_p", "game_id"], ascending=[False, True]
                ).iloc[0]
                if protocol == "chronological"
                else None
            )
            selected.append(
                {
                    "protocol": protocol,
                    "season": int(season),
                    "week": int(week),
                    "eligible_games": len(group),
                    "current_game_id": by_current.game_id,
                    "alpha_game_id": by_alpha.game_id if by_alpha is not None else None,
                    "current_grade": by_current.current_grade,
                    "alpha_grade": by_alpha.current_grade if by_alpha is not None else None,
                    "current_correct": by_current.current_correct,
                    "alpha_correct": by_alpha.current_correct if by_alpha is not None else None,
                    "current_points": float(by_current.current_points),
                    "alpha_points": float(by_alpha.current_points)
                    if by_alpha is not None
                    else None,
                    "current_confidence": float(by_current.current_confidence),
                    "alpha_selected_current_confidence": (
                        float(by_alpha.current_confidence) if by_alpha is not None else None
                    ),
                    "alpha_selected_alpha_probability": (
                        float(by_alpha.alpha_current_side_p) if by_alpha is not None else None
                    ),
                    "current_model_probability": float(
                        by_current.model_probability
                        if by_current.current_pick_home
                        else 1.0 - by_current.model_probability
                    ),
                    "alpha_selected_model_probability": (
                        float(
                            by_alpha.model_probability
                            if by_alpha.current_pick_home
                            else 1.0 - by_alpha.model_probability
                        )
                        if by_alpha is not None
                        else None
                    ),
                    "rest_accuracy": float(
                        group.loc[group.game_id.ne(by_current.game_id), "current_correct"].mean()
                    ),
                }
            )
        weeks = pd.DataFrame(selected)
        weekly_rows.append(weeks)
        if protocol == "chronological":
            weeks["nominee_differs"] = weeks.current_game_id.ne(weeks.alpha_game_id)
            decisive = weeks.loc[weeks.current_points.ne(weeks.alpha_points)]
            current_decisive_wins = int(decisive.current_points.gt(decisive.alpha_points).sum())
            alpha_decisive_wins = int(decisive.alpha_points.gt(decisive.current_points).sum())
            paired = blocked_mean(
                weeks,
                100.0
                * (
                    weeks.current_grade.eq("W").astype(float)
                    - weeks.alpha_grade.eq("W").astype(float)
                ).to_numpy(),
            )
            exact = (
                float(binomtest(current_decisive_wins, len(decisive), p=0.5).pvalue)
                if len(decisive)
                else 1.0
            )
        else:
            current_decisive_wins = None
            alpha_decisive_wins = None
            paired = None
            exact = None
        graded_eligible = eligible.loc[eligible.current_grade.ne("P")].copy()
        graded_eligible["sunday_confidence"] = graded_eligible.current_confidence
        graded_eligible["sunday_correct"] = graded_eligible.current_correct
        slope = within_week_slope(graded_eligible, "sunday")
        current_graded = weeks.loc[weeks.current_grade.ne("P")]
        current_gap = blocked_mean(
            current_graded,
            100.0 * (current_graded.current_correct - current_graded.current_confidence).to_numpy(),
        )
        alpha_selected_gap = (
            blocked_mean(
                weeks.loc[weeks.alpha_grade.ne("P")],
                100.0
                * (
                    weeks.loc[weeks.alpha_grade.ne("P"), "alpha_correct"]
                    - weeks.loc[weeks.alpha_grade.ne("P"), "alpha_selected_current_confidence"]
                ).to_numpy(),
            )
            if protocol == "chronological"
            else None
        )
        top_vs_rest = blocked_mean(
            current_graded,
            100.0 * (current_graded.current_correct - current_graded.rest_accuracy).to_numpy(),
        )
        loss = {}
        loss_arms = [
            ("current", "current_confidence", "current_correct", "current_model_probability"),
        ]
        if protocol == "chronological":
            loss_arms.append(
                (
                    "alpha_selected",
                    "alpha_selected_current_confidence",
                    "alpha_correct",
                    "alpha_selected_model_probability",
                )
            )
        for arm, confidence_column, correct_column, model_column in loss_arms:
            measured = weeks.loc[weeks[correct_column].notna()]
            actual = measured[correct_column].to_numpy(dtype=float)
            chance = measured[confidence_column].to_numpy(dtype=float).clip(1e-9, 1 - 1e-9)
            model = measured[model_column].to_numpy(dtype=float).clip(1e-9, 1 - 1e-9)
            loss[arm] = {
                "graded_nominees": len(measured),
                "brier": float(np.mean((chance - actual) ** 2)),
                "log_loss": float(
                    np.mean(-actual * np.log(chance) - (1 - actual) * np.log1p(-chance))
                ),
                "model_only_brier": float(np.mean((model - actual) ** 2)),
                "model_only_log_loss": float(
                    np.mean(-actual * np.log(model) - (1 - actual) * np.log1p(-model))
                ),
                "neutral_market_brier": 0.25,
                "neutral_market_log_loss": float(np.log(2)),
            }
        summaries[protocol] = {
            "weeks": len(weeks),
            "eligible_games": len(eligible),
            "excluded_games": len(rows) - len(eligible),
            "nominee_differs": (
                int(weeks.nominee_differs.sum()) if protocol == "chronological" else None
            ),
            "decisive_weeks": len(decisive) if protocol == "chronological" else None,
            "decisive_current_wins": current_decisive_wins,
            "decisive_alpha_wins": alpha_decisive_wins,
            "decisive_exact_two_sided_p": exact,
            "current_top_correct": int(weeks.current_correct.sum()),
            "current_top_pushes": int(weeks.current_grade.eq("P").sum()),
            "current_top_losses": int(weeks.current_grade.eq("L").sum()),
            "alpha_top_correct": (
                int(weeks.alpha_correct.sum()) if protocol == "chronological" else None
            ),
            "alpha_top_pushes": (
                int(weeks.alpha_grade.eq("P").sum()) if protocol == "chronological" else None
            ),
            "alpha_top_losses": (
                int(weeks.alpha_grade.eq("L").sum()) if protocol == "chronological" else None
            ),
            "current_minus_alpha_wins_per_nomination_points": paired,
            "current_top_vs_rest_accuracy_points": top_vs_rest,
            "current_within_week_slope_per_10_confidence_points": slope,
            "current_top_calibration_gap_points": current_gap,
            "alpha_selected_current_probability_calibration_gap_points": alpha_selected_gap,
            "current_top_mean_probability": float(current_graded.current_confidence.mean()),
            "alpha_selected_current_mean_probability": (
                float(
                    weeks.loc[weeks.alpha_grade.ne("P"), "alpha_selected_current_confidence"].mean()
                )
                if protocol == "chronological"
                else None
            ),
            "top_probability_loss": loss,
            "seasons": [
                {
                    "season": int(season),
                    "weeks": len(season_weeks),
                    "current_correct": int(season_weeks.current_correct.sum()),
                    "current_pushes": int(season_weeks.current_grade.eq("P").sum()),
                    "alpha_correct": (
                        int(season_weeks.alpha_correct.sum())
                        if protocol == "chronological"
                        else None
                    ),
                    "alpha_pushes": (
                        int(season_weeks.alpha_grade.eq("P").sum())
                        if protocol == "chronological"
                        else None
                    ),
                    "decisive": (
                        int(season_weeks.current_points.ne(season_weeks.alpha_points).sum())
                        if protocol == "chronological"
                        else None
                    ),
                }
                for season, season_weeks in weeks.groupby("season")
            ],
        }
    OUTPUT.mkdir(parents=True, exist_ok=True)
    pd.concat(outputs, ignore_index=True).to_parquet(OUTPUT / "per_game.parquet", index=False)
    pd.concat(weekly_rows, ignore_index=True).to_parquet(OUTPUT / "weekly.parquet", index=False)
    summary = {
        "plan": "docs/lanes/confidence-best-pick-unification.md",
        "source": {
            "sunday_predictions": str(PREDICTIONS),
            "full_opener": str(OPENER),
            "frozen_sunday_coefficients": str(COEFFICIENTS),
            "push_candidate_reconstruction": (
                "Frozen discrete opener probability, historical composition flags, "
                "pregame Sunday quote movement, and frozen protocol coefficients."
            ),
            "alpha_home_probability": str(ALPHA),
            "kickoffs": str(FEATURES),
            "alpha_producer": "git show 9f84d09:scripts/ridge_alpha_promotion_eval.py",
            "alpha_training": (
                "Completed games with gameday before each week; candidate alpha=2000."
            ),
            "alpha_score": "open_pred.home_cover_probability",
        },
        "eligibility": "REG kickoff strictly after Sunday 12:45 p.m. America/New_York",
        "grading": (
            "Nominate from all pregame eligible games. Grade W/L/P. Paired "
            "effect is wins per nomination with pushes retained as no win, "
            "not counted as losses; exact decisive sign compares ordered W>P>L. "
            "Conditional nonpush accuracy, calibration, and loss exclude push "
            "nominees without replacement, as in src/nfl_ats/backtest.py:126-147."
        ),
        "protocol_limit": (
            "Alpha=2000 has weekly strictly-earlier fits but no LOSO alpha archive; "
            "head-to-head is chronological only."
        ),
        "look_inventory": {
            "new_ranking_arms": 1,
            "conditional_nonpush_diagnostic_looks": 1,
            "corrected_full_pregame_pool_looks": 1,
            "matched_protocols": 1,
            "current_order_diagnostic_protocols": 2,
            "primary_paired_top_accuracy_cells": 1,
            "diagnostic_inferential_cells": 7,
            "top_loss_metric_cells": 6,
            "top_loss_baseline_cells": 12,
            "season_record_rows": 10,
            "exact_decisive_nulls": 1,
        },
        "results": summaries,
    }
    (OUTPUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                protocol: {
                    key: value
                    for key, value in summaries[protocol].items()
                    if key
                    in (
                        "weeks",
                        "eligible_games",
                        "decisive_weeks",
                        "decisive_current_wins",
                        "decisive_alpha_wins",
                        "current_top_correct",
                        "alpha_top_correct",
                        "current_minus_alpha_wins_per_nomination_points",
                        "current_within_week_slope_per_10_confidence_points",
                    )
                }
                for protocol in PROTOCOLS
            }
        )
    )


if __name__ == "__main__":
    main()
