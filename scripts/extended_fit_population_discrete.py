from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from nfl_ats.clv import DEFAULT_MIN_TRAIN_GAMES, resolve_active_model_config
from nfl_ats.home_side_location import (
    HOME_SIDE_OFFSET_SERVED,
    fit_home_side_offsets,
    prior_rows_before,
)
from nfl_ats.margin import fit_margin_model
from nfl_ats.mass_preserving_lattice import (
    BASE_PROBABILITY_POLICY,
    DiscretePushReader,
    prior_pool,
    serve_discrete_three_way,
)
from nfl_ats.modeling import regular_season_rows
from nfl_ats.pick_probability import (
    FLAG_SUM_COLUMN,
    MOVE_AVAILABLE_COLUMN,
    MOVE_COLUMN,
    PROBABILITY_EPSILON,
    signed_composition_flags,
)
from nfl_ats.pick_probability_fit import (
    FIT_RIDGE,
    SCHEDULE_COLUMNS,
    _arrest_incidents,
    _design,
    _fit_logit,
    _forecast_temperatures,
    _predict,
    _protection_back_side,
    _standardisers,
    build_fit_population,
)
from nfl_ats.public_board import find_matching_opener_evaluation
from nfl_ats.snapshots import latest_snapshot, load_snapshot

REPO = Path(__file__).resolve().parents[1]
ARTIFACTS_ROOT = REPO / "artifacts"
DATA_ROOT = REPO / "data"
FEATURES_PATH = DATA_ROOT / "processed" / "game_features_weak_stack.parquet"
SBR_SCORED_ARTIFACT = ARTIFACTS_ROOT / "sbr_era_opener_eval" / "20260819T233013Z" / "scored.parquet"
SBR_SEASON_END = 2019
OUT_ROOT = ARTIFACTS_ROOT / "extended_fit_population"
KEEP_COLUMNS = [
    "game_id",
    "season",
    "week",
    "home_covered",
    "model_logit",
    FLAG_SUM_COLUMN,
    MOVE_COLUMN,
    MOVE_AVAILABLE_COLUMN,
    "opener_source",
]


def _walk_forward_discrete_open(
    frame: pd.DataFrame,
    grading: pd.DataFrame,
    *,
    config: dict,
    probability_method: str,
    min_train_games: int,
    apply_offset: bool,
) -> pd.DataFrame:

    completed = frame.loc[frame["result"].notna()].copy()
    completed["gameday"] = pd.to_datetime(completed["gameday"], errors="raise")
    grading = grading.copy()
    grading["game_id"] = grading["game_id"].astype(str)
    grading["gameday"] = pd.to_datetime(grading["gameday"], errors="raise")
    opener_map = grading.set_index("game_id")["open_home_spread"]
    discrete_pool = prior_pool(frame, opener_map)
    archive_columns = ["game_id", "season", "week", "spread_line", "point_incumbent", "result"]
    archive_stream = pd.DataFrame(columns=archive_columns)
    rows: list[pd.DataFrame] = []
    for (season, week), group in grading.groupby(["season", "week"], sort=True):
        season_i, week_i = int(season), int(week)
        week_rows = frame.loc[frame["game_id"].astype(str).isin(set(group["game_id"]))]
        if week_rows.empty:
            continue
        cutoff = pd.Timestamp(group["gameday"].min())
        training = completed.loc[completed["gameday"].lt(cutoff)]
        if len(training) < min_train_games:
            continue
        model = fit_margin_model(
            training,
            target="market_residual",
            model_name=config["regressor"],
            feature_profile=config["feature_profile"],
            ridge_alpha=config["ridge_alpha"],
        )
        exclude_ids = set(group["game_id"].astype(str))
        reader = DiscretePushReader.for_week(
            discrete_pool,
            season=season_i,
            week=week_i,
            cutoff=cutoff,
            exclude_game_ids=exclude_ids,
        )
        scoring = week_rows.merge(
            group[["game_id", "open_home_spread"]], on="game_id", how="inner"
        ).copy()
        at_open = scoring.copy()
        at_open["spread_line"] = at_open["open_home_spread"]
        predicted_at_open = model.predict(at_open, probability_method=probability_method)
        if apply_offset and not archive_stream.empty:
            fitted = fit_home_side_offsets(prior_rows_before(archive_stream, season_i, week_i))
            offsets = fitted.offset_for(at_open["spread_line"]).fillna(0.0).to_numpy(dtype=float)
        else:
            offsets = np.zeros(len(at_open), dtype=float)
        if apply_offset and np.any(offsets != 0.0):
            served_at_open = model.predict(
                at_open, probability_method=probability_method, center_offset=offsets
            )
        else:
            served_at_open = predicted_at_open
        discrete = serve_discrete_three_way(
            served_at_open,
            at_open,
            reader,
            residuals=model.residuals,
            probability_method=probability_method,
        )
        out = scoring[["game_id"]].copy()
        out["season"] = season_i
        out["week"] = week_i
        out["gameday"] = at_open["gameday"].to_numpy()
        out["tue_open_home_spread"] = at_open["spread_line"].to_numpy()
        out["home_cover_probability_at_open"] = discrete["home_cover_probability"].to_numpy()
        out["home_side_offset_at_open"] = offsets
        out["base_probability_policy"] = BASE_PROBABILITY_POLICY
        result_values = pd.to_numeric(scoring["result"], errors="coerce").to_numpy()
        out["margin_vs_open"] = result_values - at_open["spread_line"].to_numpy()
        rows.append(out)
        week_stream = pd.DataFrame(
            {
                "game_id": scoring["game_id"].astype(str).to_numpy(),
                "season": season_i,
                "week": week_i,
                "spread_line": at_open["spread_line"].to_numpy(dtype=float),
                "point_incumbent": predicted_at_open["predicted_margin"].to_numpy(dtype=float),
                "result": result_values,
            }
        )
        archive_stream = (
            week_stream
            if archive_stream.empty
            else pd.concat([archive_stream, week_stream], ignore_index=True)
        )
    if not rows:
        return pd.DataFrame(
            columns=[
                "game_id",
                "season",
                "week",
                "tue_open_home_spread",
                "home_cover_probability_at_open",
                "home_side_offset_at_open",
                "base_probability_policy",
                "margin_vs_open",
            ]
        )
    return (
        pd.concat(rows, ignore_index=True)
        .sort_values(["season", "week", "game_id"])
        .reset_index(drop=True)
    )


def _reproduction_check(frame: pd.DataFrame, config: dict, probability_method: str) -> dict:

    matched = find_matching_opener_evaluation(ARTIFACTS_ROOT)
    if matched is None:
        raise RuntimeError("no opener evaluation matches the active model for the repro check")
    _metadata, directory = matched
    served = pd.read_parquet(directory / "per_game.parquet")
    served["game_id"] = served["game_id"].astype(str)
    schedule_slice = frame[["game_id", "season", "week", "gameday"]].drop_duplicates("game_id")
    grading = served[["game_id", "season", "week", "tue_open_home_spread"]].rename(
        columns={"tue_open_home_spread": "open_home_spread"}
    )
    grading = grading.merge(schedule_slice[["game_id", "gameday"]], on="game_id", how="inner")
    rebuilt = _walk_forward_discrete_open(
        frame,
        grading,
        config=config,
        probability_method=probability_method,
        min_train_games=DEFAULT_MIN_TRAIN_GAMES,
        apply_offset=HOME_SIDE_OFFSET_SERVED,
    )
    compared = served[["game_id", "home_cover_probability_at_open"]].merge(
        rebuilt[["game_id", "home_cover_probability_at_open"]],
        on="game_id",
        how="inner",
        suffixes=("_served", "_rebuilt"),
    )
    diff = (
        compared["home_cover_probability_at_open_served"]
        - compared["home_cover_probability_at_open_rebuilt"]
    ).abs()
    return {
        "served_games": len(served),
        "rebuilt_games": len(rebuilt),
        "compared_games": len(compared),
        "max_abs_diff": float(diff.max()) if len(diff) else None,
        "mean_abs_diff": float(diff.mean()) if len(diff) else None,
        "opener_evaluation_directory": str(directory.relative_to(ARTIFACTS_ROOT)).replace(
            "\\", "/"
        ),
    }


def _sbr_grading(frame: pd.DataFrame) -> pd.DataFrame:

    scored = pd.read_parquet(SBR_SCORED_ARTIFACT)
    scored = scored.loc[scored["season"].le(SBR_SEASON_END)].copy()
    scored["game_id"] = scored["game_id"].astype(str)
    known_ids = set(frame["game_id"].astype(str))
    scored = scored.loc[scored["game_id"].isin(known_ids)]
    grading = scored[["game_id", "season", "week", "gameday", "result"]].copy()
    grading["open_home_spread"] = pd.to_numeric(scored["proxy_open_home_spread"], errors="coerce")
    grading = grading.loc[grading["open_home_spread"].notna()].reset_index(drop=True)
    return grading


def _discrete_sbr_population(
    frame: pd.DataFrame, schedules: pd.DataFrame, config: dict, probability_method: str
) -> tuple[pd.DataFrame, int]:

    grading = _sbr_grading(frame)
    rebuilt = _walk_forward_discrete_open(
        frame,
        grading,
        config=config,
        probability_method=probability_method,
        min_train_games=DEFAULT_MIN_TRAIN_GAMES,
        apply_offset=HOME_SIDE_OFFSET_SERVED,
    )
    schedule_columns = [c for c in SCHEDULE_COLUMNS if c in schedules.columns]
    joined = rebuilt.merge(
        schedules[schedule_columns].drop_duplicates("game_id"),
        on=["game_id", "season"],
        how="left",
        suffixes=("", "_schedule"),
    )
    if "week_schedule" in joined.columns:
        joined = joined.drop(columns=["week_schedule"])
    before = len(joined)
    joined = joined.loc[joined["home_team"].notna()].copy()
    dropped_no_schedule = before - len(joined)

    margin = pd.to_numeric(joined["margin_vs_open"], errors="coerce")
    graded = joined.loc[margin.notna() & margin.ne(0.0)].reset_index(drop=True)
    graded["home_covered"] = (
        pd.to_numeric(graded["margin_vs_open"], errors="coerce").gt(0.0).astype(float)
    )
    stated = pd.to_numeric(graded["home_cover_probability_at_open"], errors="coerce").clip(
        PROBABILITY_EPSILON, 1.0 - PROBABILITY_EPSILON
    )
    graded["model_logit"] = np.log(stated / (1.0 - stated))

    incidents = _arrest_incidents(DATA_ROOT)
    forecasts = _forecast_temperatures(DATA_ROOT)
    protection = _protection_back_side(schedules, DATA_ROOT)
    flags = signed_composition_flags(
        graded,
        schedules,
        incidents=incidents,
        forecasts_tuesday_noon=forecasts,
        protection_back_side=protection,
    )
    graded = graded.merge(
        flags[["game_id", FLAG_SUM_COLUMN]], on="game_id", how="left", validate="one_to_one"
    )
    graded[FLAG_SUM_COLUMN] = graded[FLAG_SUM_COLUMN].fillna(0.0)
    graded[MOVE_COLUMN] = 0.0
    graded[MOVE_AVAILABLE_COLUMN] = 0.0
    graded["opener_source"] = "sbr_proxy_discrete"
    return graded[KEEP_COLUMNS].reset_index(drop=True), dropped_no_schedule


def _true_population() -> tuple[pd.DataFrame, dict]:
    graded, provenance = build_fit_population(ARTIFACTS_ROOT, DATA_ROOT)
    graded = graded.copy()
    graded["game_id"] = graded["game_id"].astype(str)
    graded["opener_source"] = "tue_open"
    return graded[KEEP_COLUMNS].reset_index(drop=True), provenance


def _loso_predictions(frame: pd.DataFrame) -> pd.DataFrame:
    seasons = sorted(frame["season"].unique())
    predictions: list[pd.DataFrame] = []
    for season in seasons:
        train = frame.loc[frame["season"].ne(season)]
        test = frame.loc[frame["season"].eq(season)]
        if train.empty or test.empty:
            continue
        means, stds = _standardisers(train)
        beta = _fit_logit(
            _design(train, means, stds),
            train["home_covered"].to_numpy(dtype=float),
            FIT_RIDGE,
        )
        probability = _predict(test, beta, means, stds)
        pick_home = probability >= 0.5
        out = test[["game_id", "season"]].copy()
        out["correct"] = pick_home.astype(float) == test["home_covered"].to_numpy(dtype=float)
        predictions.append(out)
    if not predictions:
        return pd.DataFrame(columns=["game_id", "season", "correct"])
    return pd.concat(predictions, ignore_index=True)


def main() -> None:
    frame = regular_season_rows(pd.read_parquet(FEATURES_PATH))
    frame["game_id"] = frame["game_id"].astype(str)
    schedules, _team_stats = load_snapshot(latest_snapshot(DATA_ROOT / "raw"))
    config = resolve_active_model_config(ARTIFACTS_ROOT)
    probability_method = config["probability_method"]

    repro = _reproduction_check(frame, config, probability_method)

    sbr_population, dropped_no_schedule = _discrete_sbr_population(
        frame, schedules, config, probability_method
    )
    true_population, provenance = _true_population()
    extended = (
        pd.concat([sbr_population, true_population], ignore_index=True)
        .sort_values(["season", "week", "game_id"])
        .reset_index(drop=True)
    )

    ts = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    out_dir = OUT_ROOT / ts
    out_dir.mkdir(parents=True, exist_ok=True)
    extended.to_parquet(out_dir / "population.parquet", index=False)

    games_per_season = {int(k): int(v) for k, v in extended.groupby("season").size().items()}
    source_counts = {str(k): int(v) for k, v in extended.groupby("opener_source").size().items()}
    flag_coverage_nonzero = float(extended[FLAG_SUM_COLUMN].ne(0.0).mean())

    extended_predictions = _loso_predictions(extended)
    true_only_predictions = _loso_predictions(true_population)
    extended_on_2020_2025 = extended_predictions.loc[extended_predictions["season"].ge(2020)]

    summary = {
        "generated_at_utc": ts,
        "games_total": len(extended),
        "games_per_season": games_per_season,
        "opener_source_counts": source_counts,
        "sbr_rows_dropped_no_schedule_match": int(dropped_no_schedule),
        "flag_coverage_nonzero_fraction": flag_coverage_nonzero,
        "sbr_model_logit_source": "walk_forward_discrete_rebuild_at_sbr_proxy_open",
        "sbr_model_logit_is_discrete": True,
        "sbr_model_logit_note": (
            "pre-2020 model_logit is rebuilt from scratch by this script's own walk-forward "
            "loop, mirroring opener_pick_evaluation's at-open branch (DiscretePushReader, "
            "serve_discrete_three_way, fit_home_side_offsets under HOME_SIDE_OFFSET_SERVED), "
            "settled at sbr_era_opener_eval's matched SBR proxy open instead of the archived "
            "Tuesday opener. 2020-2025 rows still reuse the served build_fit_population output "
            "unchanged."
        ),
        "reproduction_check_2020_2025": repro,
        "true_population_provenance": provenance,
        "sanity_look": {
            "note": "reported this run only, not recorded via weak-signals",
            "extended_2011_2025_loso_games": len(extended_predictions),
            "extended_2011_2025_loso_overall_accuracy": float(
                extended_predictions["correct"].mean()
            )
            if len(extended_predictions)
            else None,
            "extended_2011_2025_loso_accuracy_on_2020_2025_subset": float(
                extended_on_2020_2025["correct"].mean()
            )
            if len(extended_on_2020_2025)
            else None,
            "extended_2011_2025_loso_games_2020_2025_subset": len(extended_on_2020_2025),
            "true_2020_2025_only_loso_accuracy": float(true_only_predictions["correct"].mean())
            if len(true_only_predictions)
            else None,
            "true_2020_2025_only_loso_games": len(true_only_predictions),
        },
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
