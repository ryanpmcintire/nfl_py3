"""MOD-18 lane M: group-penalise or orthogonalise the whole team-quality block against the line."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

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

from nfl_ats.clv import (  # noqa: E402
    CLOSE_LABEL_PRIORITY,
    HISTORICAL_CAPTURE_KIND,
    build_pairing_table,
    close_reference_table,
    pick_correct,
)
from nfl_ats.home_side_location import fit_home_side_offsets, prior_rows_before  # noqa: E402
from nfl_ats.margin import (  # noqa: E402
    column_penalty_multipliers,
    fit_margin_model,
    margin_feature_columns,
    margin_feature_groups,
    resolve_feature_groups,
)
from nfl_ats.modeling import regular_season_rows  # noqa: E402
from nfl_ats.overlay_composition import DEFAULT_INCIDENTS  # noqa: E402
from nfl_ats.spread_regime import spread_bucket  # noqa: E402

PROFILE = "weak_stack"
TEAM_QUALITY_FAMILIES = ("results", "elo", "offense", "defense")
UNPENALISED_COLUMN = "spread_line"
UNPENALISED_MULTIPLIER = 1e-6
INDICATOR = "missingindicator_"
ARMS = ("incumbent", "G1", "G2", "G3")
BLOCK_MULTIPLIER = {"G1": 3.0, "G2": 10.0}


def team_quality_columns() -> tuple[str, ...]:
    """The served profile's results / elo / offense / defense columns."""

    columns = margin_feature_columns("market_residual", PROFILE)
    groups = margin_feature_groups("market_residual", PROFILE)
    return tuple(
        column
        for column, group in zip(columns, groups, strict=True)
        if group in TEAM_QUALITY_FAMILIES
    )


def penalty_map(multiplier: float) -> dict[str, float]:
    """Per-column ridge penalty multipliers with spread_line exempted."""

    columns = list(margin_feature_columns("market_residual", PROFILE))
    block = set(team_quality_columns())
    penalised = [column for column in columns if column != UNPENALISED_COLUMN]
    labels = ["team_quality" if column in block else "rest" for column in penalised]
    scaled = column_penalty_multipliers(
        penalised,
        labels,
        {"team_quality": multiplier, "rest": 1.0},
        normalize=True,
    )
    scaled[UNPENALISED_COLUMN] = UNPENALISED_MULTIPLIER
    return scaled


def orthogonalise_block(
    training: pd.DataFrame, scoring: pd.DataFrame, columns: tuple[str, ...]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Strip the line's linear share out of every team-quality column (arm G3)."""

    fitted_training = training.copy()
    fitted_scoring = scoring.copy()
    line_train = pd.to_numeric(training[UNPENALISED_COLUMN], errors="coerce")
    line_score = pd.to_numeric(scoring[UNPENALISED_COLUMN], errors="coerce")
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


def arm_inputs(arm: str, training: pd.DataFrame, scoring: pd.DataFrame, closing: pd.DataFrame):
    """Training and scoring frames plus the penalty map for one arm."""

    if arm == "incumbent":
        return training, scoring, closing, None
    if arm in BLOCK_MULTIPLIER:
        return training, scoring, closing, penalty_map(BLOCK_MULTIPLIER[arm])
    if arm == "G3":
        block = team_quality_columns()
        fitted_training, fitted_scoring = orthogonalise_block(training, scoring, block)
        _unused, fitted_closing = orthogonalise_block(training, closing, block)
        return fitted_training, fitted_scoring, fitted_closing, None
    raise ValueError(f"Unknown arm {arm!r}")


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
    group_penalty = pipeline.named_steps.get("group_penalty")
    if group_penalty is not None:
        scaled = scaled * np.asarray(group_penalty.scale_, dtype=float)[np.newaxis, :]
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


def walk_forward(features: pd.DataFrame, market_root: Path, arms: tuple[str, ...], min_train: int):
    """One weekly refit per arm on identical training rows and identical weeks."""

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
            fit_rows, open_rows, close_rows, penalties = arm_inputs(
                arm, training, at_open, at_close
            )
            model = fit_margin_model(
                fit_rows,
                target="market_residual",
                model_name="ridge",
                feature_profile=PROFILE,
                ridge_alpha=10.0,
                column_penalties=penalties,
            )
            scored = scoring[["game_id"]].copy()
            scored["season"] = int(str(season))
            scored["week"] = int(str(week))
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

    output: dict[str, pd.DataFrame] = {}
    for arm in arms:
        residuals = pd.concat(scored_weeks[arm], ignore_index=True)
        result = paired.merge(residuals.drop(columns=["season", "week"]), on="game_id", how="inner")
        result["margin_vs_open"] = result["result"] - result["tue_open_home_spread"]
        result["margin_vs_close"] = result["result"] - result["close_home_spread"]
        result["open_move"] = result["close_home_spread"] - result["tue_open_home_spread"]
        result["pick_home_at_open"] = result["residual_at_open"].gt(0.0)
        result["pick_home_at_close"] = result["residual_at_close"].gt(0.0)
        result["correct_at_open"] = pick_correct(
            result["pick_home_at_open"], result["margin_vs_open"]
        )
        result["correct_at_close"] = pick_correct(
            result["pick_home_at_close"], result["margin_vs_close"]
        )
        result["pick_home_at_open_probability_rule"] = result["home_cover_probability_at_open"].ge(
            0.5
        )
        result["pick_home_at_close_probability_rule"] = result[
            "home_cover_probability_at_close"
        ].ge(0.5)
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
        output[arm] = result.sort_values(["season", "week", "game_id"]).reset_index(drop=True)
    return output


def mechanism_rows(arm: str, frame: pd.DataFrame) -> list[dict[str, Any]]:
    """Favourite share, team-quality lean and probability scores by bucket."""

    live = frame.copy()
    live["bucket"] = spread_bucket(live["tue_open_home_spread"])
    live["coarse"] = live["bucket"].map(COARSE)
    live["fav_sign"] = np.sign(live["tue_open_home_spread"])
    live["home_cover"] = np.where(
        live["margin_vs_open"].gt(0),
        1.0,
        np.where(live["margin_vs_open"].lt(0), 0.0, np.nan),
    )
    block_columns = [f"grp_{name}" for name in TEAM_QUALITY_FAMILIES if f"grp_{name}" in live]
    live["grp_team_quality"] = live[block_columns].sum(axis=1)
    rows: list[dict[str, Any]] = []
    scopes: list[tuple[str, pd.DataFrame]] = [("overall", live)]
    scopes.extend((str(key), block) for key, block in live.groupby("coarse", sort=False))
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
                "favourite_share": float(
                    (picked_home * lined["fav_sign"].to_numpy(dtype=float) > 0).mean()
                ),
                "results_toward_favourite": float(
                    (lined["grp_results"] * lined["fav_sign"]).mean()
                ),
                "team_quality_toward_favourite": float(
                    (lined["grp_team_quality"] * lined["fav_sign"]).mean()
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
    live["bucket"] = spread_bucket(live["tue_open_home_spread"])
    live["coarse"] = live["bucket"].map(COARSE)
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
    scopes.extend((str(key), block) for key, block in live.groupby("coarse", sort=False))
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


def penalty_report(penalties: Mapping[str, float] | None) -> dict[str, float] | None:
    """The distinct multipliers actually handed to the ridge."""

    if penalties is None:
        return None
    block = set(team_quality_columns())
    return {
        "team_quality": float(next(v for k, v in penalties.items() if k in block)),
        "rest": float(
            next(v for k, v in penalties.items() if k not in block and k != UNPENALISED_COLUMN)
        ),
        "spread_line": float(penalties[UNPENALISED_COLUMN]),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--market-root", type=Path, required=True)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--min-train-games", type=int, default=500)
    parser.add_argument("--arms", default=",".join(ARMS))
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    features = pd.read_parquet(args.features)
    arms = tuple(args.arms.split(","))
    cards = walk_forward(features, args.market_root, arms, args.min_train_games)

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
    }
    reproduction["max_decomposition_gap"] = {
        arm: float(cards[arm]["decomposition_gap"].abs().max()) for arm in arms
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
    for arm in arms:
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
    base_frame["bucket"] = spread_bucket(base_frame["tue_open_home_spread"])
    base_frame["coarse"] = base_frame["bucket"].map(COARSE)
    base_frame = base_frame.set_index("game_id")

    rows: list[dict[str, Any]] = []
    for arm in arms:
        if arm == "incumbent":
            continue
        for surface, series_map in (
            ("card", card_series),
            (
                "standalone",
                {
                    name: cards[name]
                    .set_index("game_id")["correct_at_open_probability_rule"]
                    .astype(float)
                    for name in arms
                },
            ),
        ):
            candidate = series_map[arm]
            baseline = series_map["incumbent"]
            joined = pd.concat({"candidate": candidate, "baseline": baseline}, axis=1).join(
                base_frame, how="inner"
            )
            live = joined.loc[joined["candidate"].notna() & joined["baseline"].notna()]
            for scope, subset in (
                ("overall", live),
                *[(str(bucket), block) for bucket, block in live.groupby("coarse", sort=False)],
            ):
                if subset.empty:
                    continue
                delta = (subset["candidate"] - subset["baseline"]).to_numpy(dtype=float)
                blocks = subset[["season", "week"]].reset_index(drop=True)
                row: dict[str, Any] = {
                    "arm": arm,
                    "surface": surface,
                    "scope": scope,
                    "n": len(subset),
                    "baseline_accuracy": float(subset["baseline"].mean()),
                    "candidate_accuracy": float(subset["candidate"].mean()),
                    "picks_changed": int((subset["candidate"] != subset["baseline"]).sum()),
                }
                row.update({f"week_{k}": v for k, v in paired_stats(delta, blocks, "week").items()})
                row.update(
                    {f"season_{k}": v for k, v in paired_stats(delta, blocks, "season").items()}
                )
                rows.append(row)
    table = pd.DataFrame(rows)
    table.to_csv(args.out / "arm_results.csv", index=False)

    payload = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "archive": str(args.archive),
        "features": str(args.features),
        "arms": list(arms),
        "samples": SAMPLES,
        "seed": SEED,
        "team_quality_columns": list(team_quality_columns()),
        "penalties": {
            arm: penalty_report(arm_inputs(arm, features, features, features)[3])
            for arm in arms
            if arm in BLOCK_MULTIPLIER
        },
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
