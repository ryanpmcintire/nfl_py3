"""MOD-18 lane N: constrain the line column and change the training target."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))
if str(REPO / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO / "scripts"))

from spread_hole_arms import (  # noqa: E402
    COARSE,
    SAMPLES,
    SEED,
    card_correct,
    paired_stats,
)

from nfl_ats import margin as margin_module  # noqa: E402
from nfl_ats.clv import (  # noqa: E402
    CLOSE_LABEL_PRIORITY,
    HISTORICAL_CAPTURE_KIND,
    build_pairing_table,
    close_reference_table,
    pick_correct,
)
from nfl_ats.constants import FEATURE_SETS  # noqa: E402
from nfl_ats.home_side_location import fit_home_side_offsets, prior_rows_before  # noqa: E402
from nfl_ats.margin import (  # noqa: E402
    fit_margin_model,
    margin_feature_columns,
    margin_feature_groups,
    resolve_feature_groups,
)
from nfl_ats.modeling import regular_season_rows  # noqa: E402
from nfl_ats.overlay_composition import DEFAULT_INCIDENTS  # noqa: E402
from nfl_ats.spread_regime import spread_bucket  # noqa: E402

PROFILE = "weak_stack"
NO_LINE_PROFILE = "weak_stack_no_line"
NO_LINE_SET = "full_weak_stack_no_line"
TEAM_QUALITY_FAMILIES = ("results", "elo", "offense", "defense")
LINE_COLUMN = "spread_line"
INDICATOR = "missingindicator_"
WINSOR = 21.0
LOGISTIC_C = 0.1
ARMS = ("incumbent", "T1", "T2", "T3")
FORCED_BUCKET = "7.5-10"


def register_no_line_profile() -> tuple[str, ...]:
    """Register a runtime weak_stack profile with spread_line deleted."""

    served = FEATURE_SETS[margin_module.margin_feature_set("market_residual", PROFILE)]
    trimmed = tuple(column for column in served if column != LINE_COLUMN)
    FEATURE_SETS[NO_LINE_SET] = trimmed
    margin_module._MARGIN_PROFILE_FEATURE_SETS[NO_LINE_PROFILE] = (NO_LINE_SET, NO_LINE_SET)
    margin_module.MARGIN_FEATURE_PROFILES = (
        *margin_module.MARGIN_FEATURE_PROFILES,
        NO_LINE_PROFILE,
    )
    return trimmed


def team_quality_columns() -> tuple[str, ...]:
    """The served profile's results / elo / offense / defense columns."""

    columns = margin_feature_columns("market_residual", PROFILE)
    groups = margin_feature_groups("market_residual", PROFILE)
    return tuple(
        column
        for column, group in zip(columns, groups, strict=True)
        if group in TEAM_QUALITY_FAMILIES
    )


def orthogonalise_block(
    training: pd.DataFrame, scoring: pd.DataFrame, columns: tuple[str, ...]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Strip the line's linear share out of every team-quality column."""

    fitted_training = training.copy()
    fitted_scoring = scoring.copy()
    line_train = pd.to_numeric(training[LINE_COLUMN], errors="coerce")
    line_score = pd.to_numeric(scoring[LINE_COLUMN], errors="coerce")
    for column in columns:
        y = pd.to_numeric(training[column], errors="coerce")
        usable = y.notna() & line_train.notna()
        if int(usable.sum()) < 50:
            continue
        slope, intercept = np.polyfit(line_train[usable], y[usable], 1)
        fitted_training[column] = y - (intercept + slope * line_train)
        y_score = pd.to_numeric(scoring[column], errors="coerce")
        fitted_scoring[column] = y_score - (intercept + slope * line_score)
    return fitted_training, fitted_scoring


def logistic_pipeline() -> Pipeline:
    """The served ridge's preprocessing with a cover-side classifier on top."""

    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
            ("scaler", StandardScaler()),
            ("regressor", LogisticRegression(C=LOGISTIC_C, max_iter=2000)),
        ]
    )


def family_contributions(model: Any, frame: pd.DataFrame) -> pd.DataFrame:
    """Per-family ridge contributions in points for one weekly model."""

    pipeline = model.estimator
    columns = list(model.feature_columns)
    imputer = pipeline.named_steps["imputer"]
    scaler = pipeline.named_steps["scaler"]
    ridge = pipeline.named_steps["regressor"]
    imputed = np.asarray(imputer.transform(frame.loc[:, columns]), dtype=float)
    names = [str(name) for name in imputer.get_feature_names_out(columns)]
    scaled = (imputed - scaler.mean_) / scaler.scale_
    contribution = scaled * np.asarray(ridge.coef_, dtype=float)[np.newaxis, :]
    base = dict(zip(columns, resolve_feature_groups(columns), strict=True))
    out = pd.DataFrame(index=frame.index)
    for index, name in enumerate(names):
        source = name[len(INDICATOR) :] if name.startswith(INDICATOR) else name
        family = base[source]
        key = f"grp_{family}"
        values = contribution[:, index]
        out[key] = values if key not in out.columns else out[key].to_numpy() + values
    return out


def ridge_week(
    arm: str,
    training: pd.DataFrame,
    at_open: pd.DataFrame,
    at_close: pd.DataFrame,
    block: tuple[str, ...],
):
    """One weekly ridge fit and its opener / close predictions for one arm."""

    if arm == "incumbent":
        fit_rows, open_rows, close_rows = training, at_open, at_close
        profile = PROFILE
    else:
        fit_rows, open_rows = orthogonalise_block(training, at_open, block)
        _unused, close_rows = orthogonalise_block(training, at_close, block)
        profile = NO_LINE_PROFILE
    if arm == "T2":
        fit_rows = fit_rows.copy()
        fit_rows["ats_margin"] = pd.to_numeric(fit_rows["ats_margin"], errors="coerce").clip(
            -WINSOR, WINSOR
        )
    model = fit_margin_model(
        fit_rows,
        target="market_residual",
        model_name="ridge",
        feature_profile=profile,
        ridge_alpha=10.0,
    )
    return model, open_rows, close_rows


def logistic_week(
    training: pd.DataFrame,
    at_open: pd.DataFrame,
    at_close: pd.DataFrame,
    block: tuple[str, ...],
    columns: tuple[str, ...],
):
    """One weekly cover-side logistic fit and its opener / close probabilities."""

    fit_rows, open_rows = orthogonalise_block(training, at_open, block)
    _unused, close_rows = orthogonalise_block(training, at_close, block)
    fit_rows = regular_season_rows(fit_rows)
    target = pd.to_numeric(fit_rows["ats_margin"], errors="coerce")
    usable = target.notna() & target.ne(0.0)
    fit_rows = fit_rows.loc[usable]
    labels = (target.loc[usable] > 0.0).astype(int)
    pipeline = logistic_pipeline()
    pipeline.fit(fit_rows.loc[:, list(columns)], labels)
    open_probability = pipeline.predict_proba(open_rows.loc[:, list(columns)])[:, 1]
    close_probability = pipeline.predict_proba(close_rows.loc[:, list(columns)])[:, 1]
    return open_probability, close_probability, len(fit_rows)


def walk_forward(
    features: pd.DataFrame,
    market_root: Path,
    arms: tuple[str, ...],
    min_train: int,
    limit_weeks: int = 0,
):
    """One weekly refit per arm on identical training rows and identical weeks."""

    block = team_quality_columns()
    no_line_columns = margin_feature_columns("market_residual", NO_LINE_PROFILE)

    pairing = build_pairing_table(
        market_root,
        capture_kind=HISTORICAL_CAPTURE_KIND,
        labels=("tue_open", *CLOSE_LABEL_PRIORITY),
        schedule=features,
    )
    close = close_reference_table(pairing, features)
    tue_open = pairing.loc[pairing["decision_label"].eq("tue_open")][
        ["game_id", "season", "week", "home_spread", "spread_books"]
    ].rename(columns={"home_spread": "tue_open_home_spread", "spread_books": "opener_books"})
    paired = tue_open.merge(close, on="game_id", how="inner")
    outcomes = features[["game_id", "result"]].drop_duplicates("game_id")
    paired = paired.merge(outcomes, on="game_id", how="inner")
    paired = paired.loc[pd.to_numeric(paired["result"], errors="coerce").notna()].copy()

    frame = regular_season_rows(features).copy()
    frame["gameday"] = pd.to_datetime(frame["gameday"], errors="raise")
    completed = frame.loc[frame["result"].notna()].copy()

    stream_columns = ["game_id", "season", "week", "spread_line", "point_incumbent", "result"]
    streams = {arm: pd.DataFrame(columns=stream_columns) for arm in arms}
    scored_weeks: dict[str, list[pd.DataFrame]] = {arm: [] for arm in arms}

    for (season, week), group in paired.groupby(["season", "week"], sort=True):
        week_rows = frame.loc[frame["game_id"].isin(set(group["game_id"]))]
        if week_rows.empty:
            continue
        cutoff = week_rows["gameday"].min()
        training = completed.loc[completed["gameday"].lt(cutoff)]
        if len(training) < min_train:
            continue
        scoring = week_rows.merge(
            group[["game_id", "tue_open_home_spread", "close_home_spread"]],
            on="game_id",
            how="inner",
        ).copy()
        at_open = scoring.copy()
        at_open["spread_line"] = pd.to_numeric(at_open["tue_open_home_spread"], errors="raise")
        at_close = scoring.copy()
        at_close["spread_line"] = pd.to_numeric(at_close["close_home_spread"], errors="raise")

        for arm in arms:
            scored = scoring[["game_id"]].copy()
            scored["season"] = int(str(season))
            scored["week"] = int(str(week))
            if arm == "T3":
                open_probability, close_probability, rows_used = logistic_week(
                    training, at_open, at_close, block, no_line_columns
                )
                scored["residual_at_open"] = np.nan
                scored["residual_at_close"] = np.nan
                scored["home_cover_probability_at_open_raw"] = open_probability
                scored["home_cover_probability_at_close_raw"] = close_probability
                scored["home_cover_probability_at_open"] = open_probability
                scored["home_cover_probability_at_close"] = close_probability
                scored["home_side_offset_at_open"] = 0.0
                scored["training_rows"] = rows_used
                scored_weeks[arm].append(scored)
                continue
            model, open_rows, close_rows = ridge_week(arm, training, at_open, at_close, block)
            predicted_open = model.predict(open_rows, probability_method="gaussian_median")
            predicted_close = model.predict(close_rows, probability_method="gaussian_median")
            scored["residual_at_open"] = predicted_open["predicted_market_residual"].to_numpy()
            scored["residual_at_close"] = predicted_close["predicted_market_residual"].to_numpy()
            scored["home_cover_probability_at_open_raw"] = predicted_open[
                "home_cover_probability"
            ].to_numpy()
            scored["home_cover_probability_at_close_raw"] = predicted_close[
                "home_cover_probability"
            ].to_numpy()
            prior = prior_rows_before(streams[arm], int(str(season)), int(str(week)))
            if not streams[arm].empty:
                fitted = fit_home_side_offsets(prior)
                offsets = fitted.offset_for(open_rows["spread_line"]).fillna(0.0).to_numpy(float)
            else:
                offsets = np.zeros(len(open_rows), dtype=float)
            scored["home_side_offset_at_open"] = offsets
            if np.any(offsets != 0.0):
                served_open = model.predict(
                    open_rows, probability_method="gaussian_median", center_offset=offsets
                )
                served_close = model.predict(
                    close_rows, probability_method="gaussian_median", center_offset=offsets
                )
                scored["home_cover_probability_at_open"] = served_open[
                    "home_cover_probability"
                ].to_numpy()
                scored["home_cover_probability_at_close"] = served_close[
                    "home_cover_probability"
                ].to_numpy()
            else:
                scored["home_cover_probability_at_open"] = scored[
                    "home_cover_probability_at_open_raw"
                ]
                scored["home_cover_probability_at_close"] = scored[
                    "home_cover_probability_at_close_raw"
                ]
            contributions = family_contributions(model, open_rows.reset_index(drop=True))
            for column in contributions.columns:
                scored[column] = contributions[column].to_numpy()
            intercept = float(model.estimator.named_steps["regressor"].intercept_)
            scored["decomposition_gap"] = (
                contributions.to_numpy(dtype=float).sum(axis=1)
                + intercept
                - scored["residual_at_open"].to_numpy()
            )
            scored_weeks[arm].append(scored)
            week_stream = pd.DataFrame(
                {
                    "game_id": scoring["game_id"].astype(str).to_numpy(),
                    "season": int(str(season)),
                    "week": int(str(week)),
                    "spread_line": pd.to_numeric(
                        scoring["tue_open_home_spread"], errors="coerce"
                    ).to_numpy(),
                    "point_incumbent": (
                        pd.to_numeric(scoring["tue_open_home_spread"], errors="coerce").to_numpy()
                        + scored["residual_at_open"].to_numpy()
                    ),
                    "result": pd.to_numeric(scoring["result"], errors="coerce").to_numpy(),
                }
            )
            streams[arm] = (
                week_stream
                if streams[arm].empty
                else pd.concat([streams[arm], week_stream], ignore_index=True)
            )
        print(f"scored {season} week {week}", flush=True)
        if limit_weeks and len(scored_weeks[arms[0]]) >= limit_weeks:
            break

    output: dict[str, pd.DataFrame] = {}
    for arm in arms:
        residuals = pd.concat(scored_weeks[arm], ignore_index=True)
        result = paired.merge(residuals.drop(columns=["season", "week"]), on="game_id", how="inner")
        output[arm] = finalise(result)
    return output


def finalise(result: pd.DataFrame) -> pd.DataFrame:
    """Attach the margin, pick and correctness columns the harness expects."""

    result = result.copy()
    result["margin_vs_open"] = result["result"] - result["tue_open_home_spread"]
    result["margin_vs_close"] = result["result"] - result["close_home_spread"]
    result["open_move"] = result["close_home_spread"] - result["tue_open_home_spread"]
    result["pick_home_at_open_probability_rule"] = result["home_cover_probability_at_open"].ge(0.5)
    result["pick_home_at_close_probability_rule"] = result["home_cover_probability_at_close"].ge(
        0.5
    )
    result["correct_at_open_probability_rule"] = pick_correct(
        result["pick_home_at_open_probability_rule"], result["margin_vs_open"]
    )
    result["correct_at_close_probability_rule"] = pick_correct(
        result["pick_home_at_close_probability_rule"], result["margin_vs_close"]
    )
    result["pick_home_at_open_probability_rule_raw"] = result[
        "home_cover_probability_at_open_raw"
    ].ge(0.5)
    result["correct_at_open_probability_rule_raw"] = pick_correct(
        result["pick_home_at_open_probability_rule_raw"], result["margin_vs_open"]
    )
    return result.sort_values(["season", "week", "game_id"]).reset_index(drop=True)


def force_underdog(incumbent: pd.DataFrame) -> pd.DataFrame:
    """Positive control T4: the served pick forced to the underdog on 7.5-10."""

    forced = incumbent.copy()
    coarse = spread_bucket(forced["tue_open_home_spread"]).map(COARSE)
    line = pd.to_numeric(forced["tue_open_home_spread"], errors="coerce")
    target_home = line.lt(0.0).to_numpy()
    in_bucket = coarse.eq(FORCED_BUCKET).to_numpy()
    for column in (
        "home_cover_probability_at_open",
        "home_cover_probability_at_close",
        "home_cover_probability_at_open_raw",
        "home_cover_probability_at_close_raw",
    ):
        probability = forced[column].to_numpy(dtype=float)
        confidence = np.maximum(probability, 1.0 - probability)
        sided = np.where(target_home, confidence, 1.0 - confidence)
        forced[column] = np.where(in_bucket, sided, probability)
    mirror = in_bucket & (
        target_home != forced["pick_home_at_open_probability_rule"].to_numpy(dtype=bool)
    )
    forced["forced_games"] = mirror.astype(int)
    out = finalise(forced)
    out_bucket = spread_bucket(out["tue_open_home_spread"]).map(COARSE).eq(FORCED_BUCKET).to_numpy()
    out_home = pd.to_numeric(out["tue_open_home_spread"], errors="coerce").lt(0.0).to_numpy()
    check = out["pick_home_at_open_probability_rule"].to_numpy(dtype=bool)
    if not bool(np.all(check[out_bucket] == out_home[out_bucket])):
        raise ValueError("T4 did not force the underdog on every 7.5-10 game")
    return out


def mechanism_rows(arm: str, frame: pd.DataFrame) -> list[dict[str, Any]]:
    """Favourite share, team-quality lean and probability scores by bucket."""

    live = frame.copy()
    live["coarse"] = spread_bucket(live["tue_open_home_spread"]).map(COARSE)
    live["fav_sign"] = np.sign(live["tue_open_home_spread"])
    live["home_cover"] = np.where(
        live["margin_vs_open"].gt(0),
        1.0,
        np.where(live["margin_vs_open"].lt(0), 0.0, np.nan),
    )
    block_columns = [f"grp_{name}" for name in TEAM_QUALITY_FAMILIES if f"grp_{name}" in live]
    live["grp_team_quality"] = live[block_columns].sum(axis=1) if block_columns else np.nan
    rows: list[dict[str, Any]] = []
    scopes: list[tuple[str, pd.DataFrame]] = [("overall", live)]
    scopes.extend((str(key), b) for key, b in live.groupby("coarse", sort=False))
    for scope, subset in scopes:
        scored = subset.loc[subset["home_cover"].notna()]
        probability = scored["home_cover_probability_at_open"].to_numpy(dtype=float)
        outcome = scored["home_cover"].to_numpy(dtype=float)
        clipped = np.clip(probability, 1e-9, 1.0 - 1e-9)
        lined = subset.loc[subset["fav_sign"].ne(0)]
        picked_home = lined["pick_home_at_open_probability_rule"].astype(float).to_numpy() * 2 - 1
        rows.append(
            {
                "arm": arm,
                "scope": scope,
                "n": len(scored),
                "accuracy": float(scored["correct_at_open_probability_rule"].mean()),
                "accuracy_raw_pre_s3": float(scored["correct_at_open_probability_rule_raw"].mean()),
                "favourite_share": float(
                    (picked_home * lined["fav_sign"].to_numpy(dtype=float) > 0).mean()
                ),
                "results_toward_favourite": (
                    float((lined["grp_results"] * lined["fav_sign"]).mean())
                    if "grp_results" in lined
                    else float("nan")
                ),
                "market_toward_favourite": (
                    float((lined["grp_market"] * lined["fav_sign"]).mean())
                    if "grp_market" in lined
                    else float("nan")
                ),
                "team_quality_toward_favourite": (
                    float((lined["grp_team_quality"] * lined["fav_sign"]).mean())
                    if block_columns
                    else float("nan")
                ),
                "mean_residual_toward_favourite": float(
                    (lined["residual_at_open"] * lined["fav_sign"]).mean()
                ),
                "brier": float(np.mean((probability - outcome) ** 2)),
                "log_loss": float(
                    -np.mean(outcome * np.log(clipped) + (1.0 - outcome) * np.log(1.0 - clipped))
                ),
                "mean_stated_confidence": float(
                    np.mean(np.maximum(probability, 1.0 - probability))
                ),
            }
        )
    return rows


def card_brier_rows(arm: str, frame: pd.DataFrame, flips: set[str]) -> list[dict[str, Any]]:
    """Brier and log loss after the played card mirrors the flipped games."""

    live = frame.copy()
    live["coarse"] = spread_bucket(live["tue_open_home_spread"]).map(COARSE)
    live["home_cover"] = np.where(
        live["margin_vs_open"].gt(0),
        1.0,
        np.where(live["margin_vs_open"].lt(0), 0.0, np.nan),
    )
    flipped = live["game_id"].isin(flips)
    live["card_probability"] = np.where(
        flipped,
        1.0 - live["home_cover_probability_at_open"],
        live["home_cover_probability_at_open"],
    )
    rows: list[dict[str, Any]] = []
    scopes: list[tuple[str, pd.DataFrame]] = [("overall", live)]
    scopes.extend((str(key), b) for key, b in live.groupby("coarse", sort=False))
    for scope, subset in scopes:
        scored = subset.loc[subset["home_cover"].notna()]
        probability = scored["card_probability"].to_numpy(dtype=float)
        outcome = scored["home_cover"].to_numpy(dtype=float)
        clipped = np.clip(probability, 1e-9, 1.0 - 1e-9)
        rows.append(
            {
                "arm": arm,
                "scope": scope,
                "n": len(scored),
                "card_brier": float(np.mean((probability - outcome) ** 2)),
                "card_log_loss": float(
                    -np.mean(outcome * np.log(clipped) + (1.0 - outcome) * np.log(1.0 - clipped))
                ),
            }
        )
    return rows


def paired_rows(
    arm: str,
    surface: str,
    candidate: pd.Series,
    baseline: pd.Series,
    base_frame: pd.DataFrame,
    baseline_label: str,
) -> list[dict[str, Any]]:
    """Paired week- and season-blocked deltas overall and by bucket."""

    joined = pd.concat({"candidate": candidate, "baseline": baseline}, axis=1).join(
        base_frame, how="inner"
    )
    live = joined.loc[joined["candidate"].notna() & joined["baseline"].notna()]
    rows: list[dict[str, Any]] = []
    for scope, subset in (
        ("overall", live),
        *[(str(bucket), b) for bucket, b in live.groupby("coarse", sort=False)],
    ):
        if subset.empty:
            continue
        delta = (subset["candidate"] - subset["baseline"]).to_numpy(dtype=float)
        blocks = subset[["season", "week"]].reset_index(drop=True)
        row: dict[str, Any] = {
            "arm": arm,
            "surface": surface,
            "scope": scope,
            "baseline_label": baseline_label,
            "n": len(subset),
            "baseline_accuracy": float(subset["baseline"].mean()),
            "candidate_accuracy": float(subset["candidate"].mean()),
            "picks_changed": int((subset["candidate"] != subset["baseline"]).sum()),
        }
        row.update({f"week_{k}": v for k, v in paired_stats(delta, blocks, "week").items()})
        row.update({f"season_{k}": v for k, v in paired_stats(delta, blocks, "season").items()})
        rows.append(row)
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--market-root", type=Path, required=True)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--min-train-games", type=int, default=500)
    parser.add_argument("--arms", default=",".join(ARMS))
    parser.add_argument("--limit-weeks", type=int, default=0)
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    no_line_columns = register_no_line_profile()
    features = pd.read_parquet(args.features)
    arms = tuple(args.arms.split(","))
    cards = walk_forward(features, args.market_root, arms, args.min_train_games, args.limit_weeks)
    cards["T4"] = force_underdog(cards["incumbent"])
    all_arms = (*arms, "T4")

    reference = pd.read_parquet(args.archive / "per_game.parquet")
    incumbent = cards["incumbent"]
    check = reference[["game_id", "residual_at_open", "home_cover_probability_at_open"]].merge(
        incumbent[["game_id", "residual_at_open", "home_cover_probability_at_open"]],
        on="game_id",
        suffixes=("_archive", "_replay"),
    )
    reproduction = {
        "rows": len(check),
        "max_residual_gap": float(
            (check["residual_at_open_archive"] - check["residual_at_open_replay"]).abs().max()
        ),
        "max_probability_gap": float(
            (
                check["home_cover_probability_at_open_archive"]
                - check["home_cover_probability_at_open_replay"]
            )
            .abs()
            .max()
        ),
        "max_decomposition_gap": {
            arm: float(cards[arm]["decomposition_gap"].abs().max())
            for arm in all_arms
            if "decomposition_gap" in cards[arm]
        },
        "forced_games_t4": int(cards["T4"]["forced_games"].sum()),
    }
    print(json.dumps({"reproduction": reproduction}, indent=2), flush=True)

    features_path = args.data_root / "processed" / "game_features_pbp.parquet"
    incidents = REPO / DEFAULT_INCIDENTS
    if not incidents.is_file():
        incidents = args.data_root / DEFAULT_INCIDENTS.relative_to("data")

    card_series: dict[str, pd.Series] = {}
    flip_counts: dict[str, int] = {}
    mechanism: list[dict[str, Any]] = []
    probability_rows: list[dict[str, Any]] = []
    for arm in all_arms:
        frame = cards[arm]
        frame.to_parquet(args.out / f"per_game_{arm}.parquet", index=False)
        series, flips = card_correct(frame, args.data_root, features_path, incidents)
        card_series[arm] = series
        flip_counts[arm] = len(flips)
        mechanism.extend(mechanism_rows(arm, frame))
        probability_rows.extend(card_brier_rows(arm, frame, flips))
    pd.DataFrame(mechanism).to_csv(args.out / "mechanism_by_bucket.csv", index=False)
    pd.DataFrame(probability_rows).to_csv(args.out / "card_probability_scores.csv", index=False)

    base_frame = cards["incumbent"][["game_id", "season", "week", "tue_open_home_spread"]].copy()
    base_frame["coarse"] = spread_bucket(base_frame["tue_open_home_spread"]).map(COARSE)
    base_frame = base_frame.set_index("game_id")

    standalone = {
        name: cards[name].set_index("game_id")["correct_at_open_probability_rule"].astype(float)
        for name in all_arms
    }
    raw_incumbent = (
        cards["incumbent"]
        .set_index("game_id")["correct_at_open_probability_rule_raw"]
        .astype(float)
    )

    rows: list[dict[str, Any]] = []
    for arm in all_arms:
        if arm == "incumbent":
            continue
        rows.extend(
            paired_rows(
                arm, "card", card_series[arm], card_series["incumbent"], base_frame, "incumbent"
            )
        )
        rows.extend(
            paired_rows(
                arm,
                "standalone",
                standalone[arm],
                standalone["incumbent"],
                base_frame,
                "incumbent",
            )
        )
    rows.extend(
        paired_rows(
            "T3",
            "standalone_vs_raw",
            standalone["T3"],
            raw_incumbent,
            base_frame,
            "incumbent_raw_pre_s3",
        )
    )
    table = pd.DataFrame(rows)
    table.to_csv(args.out / "arm_results.csv", index=False)

    payload = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "archive": str(args.archive),
        "features": str(args.features),
        "arms": list(all_arms),
        "samples": SAMPLES,
        "seed": SEED,
        "winsor_points": WINSOR,
        "logistic_C": LOGISTIC_C,
        "forced_bucket": FORCED_BUCKET,
        "team_quality_columns": list(team_quality_columns()),
        "no_line_columns": list(no_line_columns),
        "card_flip_counts": flip_counts,
        "reproduction": reproduction,
        "results": rows,
        "mechanism": mechanism,
        "card_probability_scores": probability_rows,
    }
    (args.out / "arm_results.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(
        table[
            [
                "arm",
                "surface",
                "scope",
                "n",
                "baseline_accuracy",
                "candidate_accuracy",
                "picks_changed",
                "week_estimate_accuracy_points",
                "week_lower_accuracy_points",
                "week_upper_accuracy_points",
                "week_probability_positive",
                "season_lower_accuracy_points",
                "season_upper_accuracy_points",
                "season_probability_positive",
            ]
        ].to_string(index=False, float_format=lambda v: f"{v:.4f}")
    )
    print(
        pd.DataFrame(mechanism).to_string(index=False, float_format=lambda v: f"{v:.4f}"),
        flush=True,
    )
    print(
        pd.DataFrame(probability_rows).to_string(index=False, float_format=lambda v: f"{v:.4f}"),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
