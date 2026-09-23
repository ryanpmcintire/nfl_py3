from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from nfl_ats.clv import resolve_active_model_config
from nfl_ats.era_weighted_half_life_8_overlay import (
    _prepare_sorted_training,
    _target_values,
    fit_weighted_ridge_margin,
    half_life_weights,
)
from nfl_ats.margin import margin_feature_columns
from nfl_ats.mass_preserving_lattice import DiscretePushReader, prior_pool, serve_discrete_three_way
from nfl_ats.modeling import regular_season_rows
from nfl_ats.pick_probability_fit import (
    FIT_RIDGE,
    _design,
    _fit_logit,
    _predict,
    _standardisers,
    build_fit_population,
)
from nfl_ats.public_board import find_matching_opener_evaluation

REPO = Path(__file__).resolve().parents[1]
ARTIFACTS_ROOT = REPO / "artifacts"
DATA_ROOT = REPO / "data"
FEATURES_PATH = DATA_ROOT / "processed" / "game_features_weak_stack.parquet"

CANDIDATE_HALF_LIVES: list[float | None] = [2.0, 4.0, 8.0, None]
OUTER_SEASONS = [2020, 2021, 2022, 2023, 2024, 2025]
INNER_LOOKBACK = 3
MIN_TRAIN_GAMES = 500
RIDGE_ALPHA = 10.0
FEATURE_PROFILE = "weak_stack"
BOOTSTRAP_SAMPLES = 2000
BOOTSTRAP_SEED = 20260923


def _season_cutoff(features: pd.DataFrame, season: int) -> pd.Timestamp:
    rows = features.loc[features["season"].eq(season)]
    return pd.to_datetime(rows["gameday"]).min()


def _fit_for_season(
    features: pd.DataFrame, predict_season: int, half_life: float | None, cutoff: pd.Timestamp
) -> Any:
    completed = regular_season_rows(features)
    completed = completed.loc[
        completed["result"].notna() & pd.to_datetime(completed["gameday"]).lt(cutoff)
    ]
    if len(completed) < MIN_TRAIN_GAMES:
        return None
    sorted_frame = _prepare_sorted_training(completed)
    target_values = _target_values(sorted_frame).to_numpy(dtype=float)
    columns = margin_feature_columns("market_residual", FEATURE_PROFILE)
    seasons_arr = sorted_frame["season"].to_numpy(dtype=float)
    weights = (
        np.ones(len(sorted_frame), dtype=float)
        if half_life is None
        else half_life_weights(seasons_arr, predict_season=predict_season, half_life=half_life)
    )
    return fit_weighted_ridge_margin(
        sorted_frame,
        target=target_values,
        feature_columns=columns,
        weights=weights,
        ridge_alpha=RIDGE_ALPHA,
        model_name="ridge",
    )


def _predict_season_discrete(
    model: Any,
    features: pd.DataFrame,
    grading: pd.DataFrame,
    opener_map: pd.Series | None,
    probability_method: str,
) -> pd.DataFrame:
    discrete_pool = prior_pool(features, opener_map)
    rows: list[pd.DataFrame] = []
    weeks = sorted(int(w) for w in grading["week"].unique())
    season_i = int(grading["season"].iloc[0])
    for week in weeks:
        group = grading.loc[grading["week"].eq(week)]
        ids = set(group["game_id"].astype(str))
        at_open = features.loc[features["game_id"].astype(str).isin(ids)].copy()
        if at_open.empty:
            continue
        if opener_map is not None:
            line_map = group.set_index(group["game_id"].astype(str))["tue_open_home_spread"]
            at_open["spread_line"] = (
                at_open["game_id"].astype(str).map(line_map).astype(float).to_numpy()
            )
        cutoff = pd.to_datetime(group["gameday"]).min()
        try:
            reader = DiscretePushReader.for_week(
                discrete_pool, season=season_i, week=week, cutoff=cutoff, exclude_game_ids=ids
            )
        except ValueError:
            continue
        predicted = model.predict(at_open, probability_method=probability_method)
        discrete = serve_discrete_three_way(
            predicted,
            at_open,
            reader,
            residuals=model.residuals,
            probability_method=probability_method,
        )
        home_cover_probability = discrete["home_cover_probability"].to_numpy(dtype=float)
        rows.append(
            pd.DataFrame(
                {
                    "game_id": at_open["game_id"].astype(str).to_numpy(),
                    "predicted_margin": predicted["predicted_margin"].to_numpy(dtype=float),
                    "home_cover_probability": home_cover_probability,
                }
            )
        )
    if not rows:
        return pd.DataFrame(columns=["game_id", "predicted_margin", "home_cover_probability"])
    return pd.concat(rows, ignore_index=True)


def _inner_accuracy(
    features: pd.DataFrame, half_life: float | None, inner_season: int
) -> float | None:
    cutoff = _season_cutoff(features, inner_season)
    model = _fit_for_season(features, inner_season, half_life, cutoff)
    if model is None:
        return None
    grading = features.loc[
        features["season"].eq(inner_season) & features["result"].notna(),
        ["game_id", "season", "week", "gameday"],
    ].copy()
    if grading.empty:
        return None
    predicted = _predict_season_discrete(model, features, grading, None, "gaussian_median")
    if predicted.empty:
        return None
    joined = predicted.merge(
        features[["game_id", "ats_margin"]].assign(game_id=lambda d: d["game_id"].astype(str)),
        on="game_id",
        how="left",
    )
    joined = joined.loc[joined["ats_margin"].notna() & joined["ats_margin"].ne(0.0)]
    if joined.empty:
        return None
    pick_home = joined["home_cover_probability"].ge(0.5)
    actual_home = joined["ats_margin"].gt(0.0)
    return float(pick_home.eq(actual_home).mean())


def select_half_life(features: pd.DataFrame, outer_season: int) -> dict[str, Any]:
    inner_seasons = [outer_season - k for k in range(1, INNER_LOOKBACK + 1)]
    per_candidate: dict[str, Any] = {}
    for half_life in CANDIDATE_HALF_LIVES:
        label = "baseline" if half_life is None else f"half_life_{half_life:g}"
        accs = []
        for inner_season in inner_seasons:
            acc = _inner_accuracy(features, half_life, inner_season)
            if acc is not None:
                accs.append({"inner_season": inner_season, "accuracy": acc})
        per_candidate[label] = {
            "half_life": half_life,
            "inner_results": accs,
            "mean_accuracy": float(np.mean([a["accuracy"] for a in accs])) if accs else None,
        }
    scored = [
        (label, v["mean_accuracy"])
        for label, v in per_candidate.items()
        if v["mean_accuracy"] is not None
    ]
    if not scored:
        chosen_label, chosen_half_life = "baseline", None
    else:
        chosen_label = max(scored, key=lambda item: item[1])[0]
        chosen_half_life = per_candidate[chosen_label]["half_life"]
    return {
        "outer_season": outer_season,
        "inner_seasons_tried": inner_seasons,
        "candidates": per_candidate,
        "chosen_label": chosen_label,
        "chosen_half_life": chosen_half_life,
    }


def _week_blocked_bootstrap(
    df: pd.DataFrame, value_fn: Any, samples: int, seed: int
) -> dict[str, float]:
    blocks = list(df.groupby(["season", "week"]).groups.keys())
    rng = np.random.default_rng(seed)
    point = value_fn(df)
    draws = np.empty(samples, dtype=float)
    grouped = {key: frame for key, frame in df.groupby(["season", "week"])}  # noqa: C416
    n_blocks = len(blocks)
    for i in range(samples):
        picks = rng.integers(0, n_blocks, size=n_blocks)
        sample = pd.concat([grouped[blocks[p]] for p in picks], ignore_index=True)
        draws[i] = value_fn(sample)
    lower, upper = np.quantile(draws, [0.025, 0.975])
    probability_positive = float(np.mean(draws > 0.0))
    return {
        "estimate": float(point),
        "lower": float(lower),
        "upper": float(upper),
        "probability_positive": probability_positive,
    }


def _record(correct: pd.Series) -> str:
    wins = int(correct.sum())
    losses = int(len(correct) - wins)
    return f"{wins}-{losses}"


def _loso_predict(df: pd.DataFrame, logit_col: str) -> np.ndarray:
    frame = df.rename(columns={logit_col: "model_logit"})
    seasons = sorted(frame["season"].unique())
    oos = pd.Series(index=frame.index, dtype=float)
    for held in seasons:
        train = frame.loc[frame["season"].ne(held)]
        test = frame.loc[frame["season"].eq(held)]
        if train.empty or test.empty:
            continue
        means, stds = _standardisers(train)
        beta = _fit_logit(
            _design(train, means, stds), train["home_covered"].to_numpy(dtype=float), FIT_RIDGE
        )
        oos.loc[test.index] = _predict(test, beta, means, stds)
    return oos.to_numpy(dtype=float)


def main() -> None:
    features = pd.read_parquet(FEATURES_PATH)
    config = resolve_active_model_config(ARTIFACTS_ROOT)
    probability_method = str(config.get("probability_method", "gaussian_median"))

    matched = find_matching_opener_evaluation(ARTIFACTS_ROOT)
    if matched is None:
        raise RuntimeError("no opener evaluation matches the active model")
    _opener_meta, opener_dir = matched
    per_game = pd.read_parquet(opener_dir / "per_game.parquet")
    per_game["game_id"] = per_game["game_id"].astype(str)
    per_game = per_game.merge(
        features[["game_id", "gameday"]].assign(game_id=lambda d: d["game_id"].astype(str)),
        on="game_id",
        how="left",
    )
    opener_map = per_game.set_index("game_id")["tue_open_home_spread"]

    fold_selections: list[dict[str, Any]] = []
    recency_rows: list[pd.DataFrame] = []
    for season in OUTER_SEASONS:
        selection = select_half_life(features, season)
        fold_selections.append(selection)
        cutoff = _season_cutoff(features, season)
        model = _fit_for_season(features, season, selection["chosen_half_life"], cutoff)
        if model is None:
            continue
        grading = per_game.loc[
            per_game["season"].eq(season),
            ["game_id", "season", "week", "gameday", "tue_open_home_spread"],
        ].copy()
        predicted = _predict_season_discrete(
            model, features, grading, opener_map, probability_method
        )
        predicted["season"] = season
        recency_rows.append(predicted)

    recency = (
        pd.concat(recency_rows, ignore_index=True)
        if recency_rows
        else pd.DataFrame(
            columns=["game_id", "predicted_margin", "home_cover_probability", "season"]
        )
    )

    graded = per_game.merge(recency, on="game_id", how="inner", suffixes=("", "_recency"))
    margin_vs_open = pd.to_numeric(graded["margin_vs_open"], errors="coerce")
    graded = graded.loc[margin_vs_open.notna() & margin_vs_open.ne(0.0)].reset_index(drop=True)
    graded["home_covered"] = pd.to_numeric(graded["margin_vs_open"], errors="coerce").gt(0.0)
    graded["recency_pick_home"] = graded["home_cover_probability"].ge(0.5)
    graded["served_pick_home"] = pd.to_numeric(
        graded["home_cover_probability_at_open"], errors="coerce"
    ).ge(0.5)
    graded["recency_correct"] = (
        graded["recency_pick_home"].astype(float).eq(graded["home_covered"].astype(float))
    )
    graded["served_correct"] = (
        graded["served_pick_home"].astype(float).eq(graded["home_covered"].astype(float))
    )
    eps = 1e-6
    p_recency = graded["home_cover_probability"].astype(float).clip(eps, 1 - eps)
    p_served = pd.to_numeric(graded["home_cover_probability_at_open"], errors="coerce")
    p_served = p_served.clip(eps, 1 - eps)
    y = graded["home_covered"].astype(float)
    graded["recency_brier"] = (p_recency - y) ** 2
    graded["served_brier"] = (p_served - y) ** 2
    graded["recency_logloss"] = -(y * np.log(p_recency) + (1 - y) * np.log(1 - p_recency))
    graded["served_logloss"] = -(y * np.log(p_served) + (1 - y) * np.log(1 - p_served))
    graded["recency_abs_margin_error"] = (
        graded["predicted_margin"].astype(float) - graded["margin_vs_open"].astype(float)
    ).abs()
    graded["served_abs_margin_error"] = (
        pd.to_numeric(graded["residual_at_open"], errors="coerce")
        - graded["margin_vs_open"].astype(float)
    ).abs()
    graded["recency_model_logit"] = np.log(p_recency / (1 - p_recency))

    standalone_summary = {
        "n_games": len(graded),
        "recency_accuracy": float(graded["recency_correct"].mean()),
        "served_accuracy": float(graded["served_correct"].mean()),
        "recency_brier": float(graded["recency_brier"].mean()),
        "served_brier": float(graded["served_brier"].mean()),
        "recency_logloss": float(graded["recency_logloss"].mean()),
        "served_logloss": float(graded["served_logloss"].mean()),
        "recency_margin_mae": float(graded["recency_abs_margin_error"].mean()),
        "served_margin_mae": float(graded["served_abs_margin_error"].mean()),
    }

    def _standalone_accuracy_diff(d: pd.DataFrame) -> float:
        recency = d["recency_correct"].astype(float).mean()
        served = d["served_correct"].astype(float).mean()
        return float(recency - served) * 100.0

    standalone_accuracy_boot = _week_blocked_bootstrap(
        graded, _standalone_accuracy_diff, BOOTSTRAP_SAMPLES, BOOTSTRAP_SEED
    )
    standalone_brier_boot = _week_blocked_bootstrap(
        graded,
        lambda d: float((d["served_brier"] - d["recency_brier"]).mean()),
        BOOTSTRAP_SAMPLES,
        BOOTSTRAP_SEED,
    )
    standalone_logloss_boot = _week_blocked_bootstrap(
        graded,
        lambda d: float((d["served_logloss"] - d["recency_logloss"]).mean()),
        BOOTSTRAP_SAMPLES,
        BOOTSTRAP_SEED,
    )
    standalone_margin_boot = _week_blocked_bootstrap(
        graded,
        lambda d: float((d["served_abs_margin_error"] - d["recency_abs_margin_error"]).mean()),
        BOOTSTRAP_SAMPLES,
        BOOTSTRAP_SEED,
    )

    served_pop, fit_provenance = build_fit_population(ARTIFACTS_ROOT, DATA_ROOT)
    served_pop = served_pop.copy()
    served_pop["game_id"] = served_pop["game_id"].astype(str)
    combined = served_pop.merge(
        graded[["game_id", "recency_model_logit"]], on="game_id", how="inner"
    )
    combined["served_model_logit"] = combined["model_logit"].astype(float)

    served_oos_p = _loso_predict(
        combined.rename(columns={"served_model_logit": "model_logit_served"}).assign(
            model_logit=lambda d: d["model_logit_served"]
        ),
        "model_logit",
    )
    recency_oos_p = _loso_predict(
        combined.assign(model_logit=lambda d: d["recency_model_logit"]), "model_logit"
    )
    combined["served4_p"] = served_oos_p
    combined["recency4_p"] = recency_oos_p
    combined["home_covered_f"] = combined["home_covered"].astype(float)
    combined["served4_correct"] = (
        (combined["served4_p"] >= 0.5).astype(float).eq(combined["home_covered_f"])
    )
    combined["recency4_correct"] = (
        (combined["recency4_p"] >= 0.5).astype(float).eq(combined["home_covered_f"])
    )
    pc_s = combined["served4_p"].clip(eps, 1 - eps)
    pc_r = combined["recency4_p"].clip(eps, 1 - eps)
    yv = combined["home_covered_f"]
    combined["served4_brier"] = (pc_s - yv) ** 2
    combined["recency4_brier"] = (pc_r - yv) ** 2
    combined["served4_logloss"] = -(yv * np.log(pc_s) + (1 - yv) * np.log(1 - pc_s))
    combined["recency4_logloss"] = -(yv * np.log(pc_r) + (1 - yv) * np.log(1 - pc_r))

    combined_summary = {
        "n_games": len(combined),
        "served4_accuracy": float(combined["served4_correct"].mean()),
        "recency4_accuracy": float(combined["recency4_correct"].mean()),
        "served4_brier": float(combined["served4_brier"].mean()),
        "recency4_brier": float(combined["recency4_brier"].mean()),
        "served4_logloss": float(combined["served4_logloss"].mean()),
        "recency4_logloss": float(combined["recency4_logloss"].mean()),
        "served4_record": _record(combined["served4_correct"]),
        "recency4_record": _record(combined["recency4_correct"]),
    }
    decisive = combined.loc[(combined["served4_p"] >= 0.5) != (combined["recency4_p"] >= 0.5)]
    decisive_summary = {
        "n_decisive": len(decisive),
        "served_record_on_decisive": _record(decisive["served4_correct"]),
        "recency_record_on_decisive": _record(decisive["recency4_correct"]),
    }

    def _combined_accuracy_diff(d: pd.DataFrame) -> float:
        recency = d["recency4_correct"].astype(float).mean()
        served = d["served4_correct"].astype(float).mean()
        return float(recency - served) * 100.0

    combined_accuracy_boot = _week_blocked_bootstrap(
        combined, _combined_accuracy_diff, BOOTSTRAP_SAMPLES, BOOTSTRAP_SEED
    )
    combined_brier_boot = _week_blocked_bootstrap(
        combined,
        lambda d: float((d["served4_brier"] - d["recency4_brier"]).mean()),
        BOOTSTRAP_SAMPLES,
        BOOTSTRAP_SEED,
    )
    combined_logloss_boot = _week_blocked_bootstrap(
        combined,
        lambda d: float((d["served4_logloss"] - d["recency4_logloss"]).mean()),
        BOOTSTRAP_SAMPLES,
        BOOTSTRAP_SEED,
    )

    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    out_dir = ARTIFACTS_ROOT / "base_model_recency" / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    graded.to_parquet(out_dir / "standalone_graded.parquet")
    combined.to_parquet(out_dir / "combined_fourterm.parquet")
    payload = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "config": {
            "feature_profile": FEATURE_PROFILE,
            "ridge_alpha": RIDGE_ALPHA,
            "min_train_games": MIN_TRAIN_GAMES,
            "probability_method": probability_method,
            "candidate_half_lives": [None if h is None else h for h in CANDIDATE_HALF_LIVES],
            "outer_seasons": OUTER_SEASONS,
            "inner_lookback_seasons": INNER_LOOKBACK,
            "granularity": "season_boundary_refit_not_weekly",
            "inner_selection_grade": "close",
            "outer_evaluation_grade": "opener",
        },
        "fold_selections": fold_selections,
        "standalone_summary": standalone_summary,
        "standalone_bootstrap": {
            "accuracy_points": standalone_accuracy_boot,
            "brier_improvement": standalone_brier_boot,
            "logloss_improvement": standalone_logloss_boot,
            "margin_mae_improvement": standalone_margin_boot,
        },
        "fit_population_provenance": {k: str(v) for k, v in fit_provenance.items()},
        "combined_fourterm_summary": combined_summary,
        "combined_decisive": decisive_summary,
        "combined_bootstrap": {
            "accuracy_points": combined_accuracy_boot,
            "brier_improvement": combined_brier_boot,
            "logloss_improvement": combined_logloss_boot,
        },
    }
    metadata_text = json.dumps(payload, indent=2, default=str)
    (out_dir / "metadata.json").write_text(metadata_text, encoding="utf-8")
    print(json.dumps({"out_dir": str(out_dir)}, indent=2))
    print(json.dumps(payload, indent=2, default=str))


if __name__ == "__main__":
    main()
