"""MOD-18 lane F: explain the big-spread hole in the served weak_stack ridge."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

from nfl_ats.clv import (  # noqa: E402
    CLOSE_LABEL_PRIORITY,
    HISTORICAL_CAPTURE_KIND,
    build_pairing_table,
    close_reference_table,
    week_blocked_bootstrap,
)
from nfl_ats.margin import (  # noqa: E402
    fit_margin_model,
    margin_feature_columns,
    resolve_feature_groups,
)
from nfl_ats.modeling import regular_season_rows  # noqa: E402
from nfl_ats.spread_regime import spread_bucket  # noqa: E402

SEED = 20260817
DRAWS = 20_000
COARSE = {"0-3": "0-6.5", "3.5-6.5": "0-6.5", "7": "7", "7.5-10": "7.5-10", "10.5+": "10.5+"}
INDICATOR = "missingindicator_"


def contributions_for_week(model: Any, frame: pd.DataFrame) -> tuple[pd.DataFrame, float, float]:
    """Per-column ridge contributions, the residual median and its spread."""

    pipeline = model.estimator
    columns = list(model.feature_columns)
    imputer = pipeline.named_steps["imputer"]
    scaler = pipeline.named_steps["scaler"]
    ridge = pipeline.named_steps["regressor"]
    raw = frame.loc[:, columns]
    imputed = np.asarray(imputer.transform(raw), dtype=float)
    names = [str(name) for name in imputer.get_feature_names_out(columns)]
    scaled = (imputed - scaler.mean_) / scaler.scale_
    contribution = scaled * np.asarray(ridge.coef_, dtype=float)[np.newaxis, :]
    table = pd.DataFrame(contribution, columns=names, index=frame.index)
    return table, float(np.median(model.residuals)), float(np.std(model.residuals, ddof=1))


def group_map(columns: list[str], names: list[str]) -> dict[str, str]:
    """Label every transformed column with its declared feature family."""

    base = dict(zip(columns, resolve_feature_groups(columns), strict=True))
    mapping: dict[str, str] = {}
    for name in names:
        source = name[len(INDICATOR) :] if name.startswith(INDICATOR) else name
        mapping[name] = base[source]
    return mapping


def walk_forward(features: pd.DataFrame, market_root: Path, min_train: int) -> pd.DataFrame:
    """Reproduce opener_pick_evaluation's fits and record the decomposition."""

    profile = "weak_stack"
    columns = list(margin_feature_columns("market_residual", profile))
    pairing = build_pairing_table(
        market_root,
        capture_kind=HISTORICAL_CAPTURE_KIND,
        labels=("tue_open", *CLOSE_LABEL_PRIORITY),
        schedule=features,
    )
    close = close_reference_table(pairing, features)
    tue_open = pairing.loc[pairing["decision_label"].eq("tue_open")][
        ["game_id", "season", "week", "home_spread"]
    ].rename(columns={"home_spread": "tue_open_home_spread"})
    paired = tue_open.merge(close, on="game_id", how="inner")
    outcomes = features[["game_id", "result"]].drop_duplicates("game_id")
    paired = paired.merge(outcomes, on="game_id", how="inner")
    paired = paired.loc[pd.to_numeric(paired["result"], errors="coerce").notna()].copy()

    frame = regular_season_rows(features).copy()
    frame["gameday"] = pd.to_datetime(frame["gameday"], errors="raise")
    completed = frame.loc[frame["result"].notna()].copy()

    rows: list[pd.DataFrame] = []
    for (season, week), group in paired.groupby(["season", "week"], sort=True):
        week_rows = frame.loc[frame["game_id"].isin(set(group["game_id"]))]
        if week_rows.empty:
            continue
        cutoff = week_rows["gameday"].min()
        training = completed.loc[completed["gameday"].lt(cutoff)]
        if len(training) < min_train:
            continue
        model = fit_margin_model(
            training,
            target="market_residual",
            model_name="ridge",
            feature_profile=profile,
            ridge_alpha=10.0,
        )
        scoring = week_rows.merge(
            group[["game_id", "tue_open_home_spread"]], on="game_id", how="inner"
        ).copy()
        scoring["spread_line"] = pd.to_numeric(scoring["tue_open_home_spread"], errors="raise")
        table, location, scale = contributions_for_week(model, scoring)
        mapping = group_map(columns, list(table.columns))
        grouped: dict[str, Any] = {}
        for name in table.columns:
            family = mapping[name]
            running = grouped.get(family)
            values = table[name].to_numpy(dtype=float)
            grouped[family] = values if running is None else running + values
        out = scoring[["game_id", "season", "week", "spread_line", "result"]].copy()
        out["season"] = int(str(season))
        out["week"] = int(str(week))
        out["residual_raw"] = table.to_numpy(dtype=float).sum(axis=1) + float(
            ridge_intercept(model)
        )
        out["residual_location"] = location
        out["residual_scale"] = scale
        out["training_rows"] = model.training_rows
        for name, values in grouped.items():
            out[f"grp_{name}"] = values
        out["grp_intercept"] = float(ridge_intercept(model))
        rows.append(out)
    return pd.concat(rows, ignore_index=True)


def ridge_intercept(model: Any) -> float:
    """The fitted ridge intercept for one weekly model."""

    return float(model.estimator.named_steps["regressor"].intercept_)


def accuracy_metric(frame: pd.DataFrame) -> dict[str, float]:
    """Forced-pick accuracy in points for the bootstrap helper."""

    live = frame.loc[frame["home_cover"].notna()]
    if live.empty:
        return {"accuracy_points": float("nan")}
    return {"accuracy_points": 100.0 * float(live["correct"].mean())}


def interval(
    frame: pd.DataFrame, metric: Any, *, draws: int, seed: int, block: str = "week"
) -> dict[str, float]:
    """Point estimate, week-blocked interval and probability_positive."""

    summary = week_blocked_bootstrap(frame, metric, block=block, samples=draws, seed=seed)
    row = summary.iloc[0]
    return {
        "estimate": float(row["estimate"]),
        "lower": float(row["lower"]),
        "upper": float(row["upper"]),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--market-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--draws", type=int, default=DRAWS)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--min-train-games", type=int, default=500)
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    features = pd.read_parquet(args.features)
    served = pd.read_parquet(args.archive / "per_game.parquet")

    decomposition = walk_forward(features, args.market_root, args.min_train_games)
    merged = decomposition.merge(
        served[
            [
                "game_id",
                "tue_open_home_spread",
                "residual_at_open",
                "home_side_offset_at_open",
                "home_cover_probability_at_open",
                "pick_home_at_open_probability_rule",
                "correct_at_open_probability_rule",
                "pick_home_at_open_probability_rule_raw",
                "correct_at_open_probability_rule_raw",
            ]
        ],
        on="game_id",
        how="inner",
    )
    merged["reproduction_gap"] = (merged["residual_raw"] - merged["residual_at_open"]).abs()
    merged["margin_less_line"] = merged["result"] - merged["spread_line"]
    merged["home_cover"] = np.where(
        merged["margin_less_line"].gt(0),
        1.0,
        np.where(merged["margin_less_line"].lt(0), 0.0, np.nan),
    )
    merged["bucket"] = spread_bucket(merged["spread_line"])
    merged["coarse"] = merged["bucket"].map(COARSE)
    merged["fav_sign"] = np.sign(merged["spread_line"])
    merged["point_raw"] = merged["residual_raw"] + merged["residual_location"]
    merged["point_served"] = merged["point_raw"] + merged["home_side_offset_at_open"]
    merged["correct"] = merged["correct_at_open_probability_rule"].astype(float)
    merged["pick_favourite"] = np.where(
        merged["fav_sign"].eq(0),
        np.nan,
        (merged["pick_home_at_open_probability_rule"].astype(float) * 2 - 1)
        .mul(merged["fav_sign"])
        .gt(0)
        .astype(float),
    )

    groups = sorted(column[4:] for column in merged.columns if column.startswith("grp_"))
    merged.to_parquet(args.out / "per_game_decomposition.parquet", index=False)

    summary: dict[str, Any] = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "archive": str(args.archive),
        "features": str(args.features),
        "games": len(merged),
        "max_reproduction_gap": float(merged["reproduction_gap"].max()),
        "draws": args.draws,
        "seed": args.seed,
        "groups": groups,
    }

    lean_rows: list[dict[str, Any]] = []
    for bucket, block in merged.groupby("bucket", sort=False):
        live = block.loc[block["fav_sign"].ne(0)]
        for name in groups:
            values = live[f"grp_{name}"] * live["fav_sign"]
            lean_rows.append(
                {
                    "bucket": str(bucket),
                    "group": name,
                    "n": len(live),
                    "mean_toward_favourite": float(values.mean()),
                    "sd": float(values.std(ddof=1)),
                    "share_toward_favourite": float(values.gt(0).mean()),
                }
            )
    lean = pd.DataFrame(lean_rows)
    lean.to_csv(args.out / "group_lean_by_bucket.csv", index=False)

    drop_rows: list[dict[str, Any]] = []
    for bucket, block in merged.groupby("bucket", sort=False):
        live = block.loc[block["home_cover"].notna()].copy()
        base = live.copy()
        base["correct"] = (
            base["point_served"].gt(0).astype(float).eq(base["home_cover"]).astype(float)
        )
        base_estimate = interval(base, accuracy_metric, draws=args.draws, seed=args.seed)
        drop_rows.append(
            {
                "bucket": str(bucket),
                "group": "(full model)",
                "n": len(live),
                "accuracy_points": base_estimate["estimate"],
                "lower": base_estimate["lower"],
                "upper": base_estimate["upper"],
                "delta_points": 0.0,
                "picks_changed": 0,
            }
        )
        for name in groups:
            trial = live.copy()
            point = trial["point_served"] - trial[f"grp_{name}"]
            trial["correct"] = point.gt(0).astype(float).eq(trial["home_cover"]).astype(float)
            estimate = interval(trial, accuracy_metric, draws=args.draws, seed=args.seed)
            drop_rows.append(
                {
                    "bucket": str(bucket),
                    "group": name,
                    "n": len(live),
                    "accuracy_points": estimate["estimate"],
                    "lower": estimate["lower"],
                    "upper": estimate["upper"],
                    "delta_points": estimate["estimate"] - base_estimate["estimate"],
                    "picks_changed": int(point.gt(0).ne(trial["point_served"].gt(0)).sum()),
                }
            )
    drops = pd.DataFrame(drop_rows)
    drops.to_csv(args.out / "leave_one_group_out_by_bucket.csv", index=False)

    sign_rows: list[dict[str, Any]] = []
    for label, key in (("bucket", "bucket"), ("coarse", "coarse")):
        for bucket, block in merged.groupby(key, sort=False):
            live = block.loc[block["home_cover"].notna()].copy()
            for variant, column in (
                ("raw_point", "point_raw"),
                ("served_point", "point_served"),
                ("raw_residual_sign", "residual_raw"),
            ):
                trial = live.copy()
                trial["correct"] = (
                    trial[column].gt(0).astype(float).eq(trial["home_cover"]).astype(float)
                )
                estimate = interval(trial, accuracy_metric, draws=args.draws, seed=args.seed)
                sign_rows.append(
                    {
                        "scale": label,
                        "bucket": str(bucket),
                        "variant": variant,
                        "n": len(trial),
                        "accuracy_points": estimate["estimate"],
                        "lower": estimate["lower"],
                        "upper": estimate["upper"],
                        "favourite_share": float(
                            trial[column].gt(0).eq(trial["fav_sign"].gt(0)).mean()
                        ),
                    }
                )
    signs = pd.DataFrame(sign_rows)
    signs.to_csv(args.out / "sign_accuracy_by_bucket.csv", index=False)

    era = regular_season_rows(features).copy()
    era = era.loc[era["result"].notna() & era["spread_line"].notna()].copy()
    era["season"] = pd.to_numeric(era["season"], errors="raise")
    era = era.loc[era["season"].between(2009, 2025)].copy()
    era["bucket"] = spread_bucket(era["spread_line"])
    era["coarse"] = era["bucket"].map(COARSE)
    era["margin_less_line"] = era["result"] - era["spread_line"]
    era["home_cover"] = np.where(
        era["margin_less_line"].gt(0), 1.0, np.where(era["margin_less_line"].lt(0), 0.0, np.nan)
    )
    era["fav_cover"] = np.where(
        era["spread_line"].gt(0), era["home_cover"], 1.0 - era["home_cover"]
    )
    era.loc[era["spread_line"].eq(0), "fav_cover"] = np.nan
    era_rows: list[dict[str, Any]] = []
    for (season, bucket), block in era.groupby(["season", "coarse"], sort=True):
        live = block.loc[block["fav_cover"].notna()]
        if live.empty:
            continue
        era_rows.append(
            {
                "season": int(season),
                "bucket": str(bucket),
                "n": len(live),
                "favourite_cover_rate": float(live["fav_cover"].mean()),
                "underdog_cover_rate": 1.0 - float(live["fav_cover"].mean()),
            }
        )
    era_table = pd.DataFrame(era_rows)
    era_table.to_csv(args.out / "market_favourite_bias_by_season.csv", index=False)

    era_block_rows: list[dict[str, Any]] = []
    for name, lo, hi in (
        ("2009-2013", 2009, 2013),
        ("2014-2019", 2014, 2019),
        ("2020-2025", 2020, 2025),
    ):
        window = era.loc[era["season"].between(lo, hi)]
        for bucket, block in window.groupby("coarse", sort=False):
            live = block.loc[block["fav_cover"].notna()].copy()
            live["correct"] = 1.0 - live["fav_cover"]
            live["home_cover"] = live["fav_cover"]
            live["week"] = pd.to_numeric(live["week"], errors="coerce")
            estimate = interval(live, accuracy_metric, draws=args.draws, seed=args.seed)
            era_block_rows.append(
                {
                    "era": name,
                    "bucket": str(bucket),
                    "n": len(live),
                    "underdog_cover_points": estimate["estimate"],
                    "lower": estimate["lower"],
                    "upper": estimate["upper"],
                }
            )
    era_blocks = pd.DataFrame(era_block_rows)
    era_blocks.to_csv(args.out / "market_favourite_bias_by_era.csv", index=False)

    big = merged.loc[merged["coarse"].isin(["7.5-10", "10.5+"])].copy()
    big["signed_gap"] = (big["result"] - big["spread_line"]) * big["fav_sign"]
    mass_rows: list[dict[str, Any]] = []
    counts = big["signed_gap"].round().value_counts().sort_index()
    total = int(counts.sum())
    for value, count in counts.items():
        mass_rows.append(
            {
                "signed_margin_minus_line": float(value),
                "games": int(count),
                "share": float(count) / total,
            }
        )
    mass = pd.DataFrame(mass_rows)
    mass.to_csv(args.out / "big_spread_margin_mass.csv", index=False)

    key_rows: list[dict[str, Any]] = []
    for label, block in (
        ("7.5-10", merged.loc[merged["coarse"].eq("7.5-10")]),
        ("10.5+", merged.loc[merged["coarse"].eq("10.5+")]),
        ("0-6.5", merged.loc[merged["coarse"].eq("0-6.5")]),
    ):
        live = block.loc[block["home_cover"].notna()].copy()
        actual = (live["result"] - live["spread_line"]).round()
        scale_hat = float(live["residual_scale"].mean())
        centre = live["point_raw"] - live["spread_line"]
        for key in (-14, -10, -7, -3, 0, 3, 7, 10, 14):
            empirical = float(actual.eq(key).mean())
            modelled = float(
                np.mean(
                    stats.norm.cdf(key + 0.5, loc=centre, scale=live["residual_scale"])
                    - stats.norm.cdf(key - 0.5, loc=centre, scale=live["residual_scale"])
                )
            )
            key_rows.append(
                {
                    "bucket": label,
                    "margin_minus_line": key,
                    "n": len(live),
                    "empirical_share": empirical,
                    "gaussian_share": modelled,
                    "gap": empirical - modelled,
                    "mean_residual_scale": scale_hat,
                }
            )
    keys = pd.DataFrame(key_rows)
    keys.to_csv(args.out / "key_number_mass_vs_gaussian.csv", index=False)

    side_rows: list[dict[str, Any]] = []
    for bucket, block in merged.groupby("coarse", sort=False):
        live = block.loc[block["home_cover"].notna() & block["fav_sign"].ne(0)].copy()
        for side, mask in (
            ("favourite", live["pick_favourite"].eq(1)),
            ("underdog", live["pick_favourite"].eq(0)),
        ):
            chunk = live.loc[mask]
            if chunk.empty:
                continue
            estimate = interval(chunk, accuracy_metric, draws=args.draws, seed=args.seed)
            side_rows.append(
                {
                    "bucket": str(bucket),
                    "picked_side": side,
                    "n": len(chunk),
                    "accuracy_points": estimate["estimate"],
                    "lower": estimate["lower"],
                    "upper": estimate["upper"],
                    "mean_stated_confidence": float(
                        chunk["home_cover_probability_at_open"].sub(0.5).abs().add(0.5).mean()
                    ),
                }
            )
    sides = pd.DataFrame(side_rows)
    sides.to_csv(args.out / "accuracy_by_picked_side.csv", index=False)

    summary["reproduction_ok"] = bool(merged["reproduction_gap"].max() < 1e-6)
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps(summary, indent=2))
    print()
    print("== group lean toward the favourite, mean points ==")
    print(
        lean.pivot(index="group", columns="bucket", values="mean_toward_favourite").to_string(
            float_format=lambda v: f"{v:+.3f}"
        )
    )
    print()
    print("== leave-one-group-out accuracy delta, points ==")
    print(
        drops.pivot(index="group", columns="bucket", values="delta_points").to_string(
            float_format=lambda v: f"{v:+.2f}"
        )
    )
    print()
    print("== sign accuracy by bucket ==")
    print(signs.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print()
    print("== market favourite bias by era ==")
    print(era_blocks.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print()
    print("== key-number mass vs the served Gaussian ==")
    print(keys.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print()
    print("== accuracy by picked side ==")
    print(sides.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print()
    print("== mass of signed margin minus line, big spreads ==")
    print(mass.loc[mass["signed_margin_minus_line"].between(-25, 25)].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
