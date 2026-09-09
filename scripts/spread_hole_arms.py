"""MOD-18 lane F Part 2: score the C1 and R1 arms through the played card."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

from nfl_ats.clv import (  # noqa: E402
    CLOSE_LABEL_PRIORITY,
    HISTORICAL_CAPTURE_KIND,
    build_pairing_table,
    close_reference_table,
    pick_correct,
)
from nfl_ats.home_side_location import fit_home_side_offsets, prior_rows_before  # noqa: E402
from nfl_ats.margin import fit_margin_model  # noqa: E402
from nfl_ats.modeling import regular_season_rows  # noqa: E402
from nfl_ats.overlay_composition import (  # noqa: E402
    DEFAULT_INCIDENTS,
    blocked_bootstrap_matrix,
    build_predictions_frame,
    reconstruct_arrest_flip_set,
    run_overlays,
)
from nfl_ats.spread_regime import attach_spread_regime, spread_bucket  # noqa: E402
from nfl_ats.unserved_tilt_marginals import (  # noqa: E402
    CARD_CHOICES,
    SERVED_CARD_MEMBERS,
    served_card_flip_set,
)

RESULTS_COLUMNS = (
    "home_point_diff",
    "away_point_diff",
    "diff_point_diff",
    "home_ats_residual",
    "away_ats_residual",
    "diff_ats_residual",
)
CARD_MEMBERS = (
    "coach_fade_overlay",
    "division_revenge_tilt_overlay",
    "player_arrests_back_side_policy",
)
COARSE = {"0-3": "0-6.5", "3.5-6.5": "0-6.5", "7": "7", "7.5-10": "7.5-10", "10.5+": "10.5+"}
SAMPLES = 20_000
SEED = 20260821


def orthogonalise_results(
    training: pd.DataFrame, scoring: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Strip the line's linear share out of the season-form block (arm R1)."""

    fitted_training = training.copy()
    fitted_scoring = scoring.copy()
    line_train = pd.to_numeric(training["spread_line"], errors="coerce")
    line_score = pd.to_numeric(scoring["spread_line"], errors="coerce")
    for column in RESULTS_COLUMNS:
        y = pd.to_numeric(training[column], errors="coerce")
        usable = y.notna() & line_train.notna()
        if int(usable.sum()) < 50:
            continue
        slope, intercept = np.polyfit(line_train[usable], y[usable], 1)
        fitted_training[column] = y - (intercept + slope * line_train)
        y_score = pd.to_numeric(scoring[column], errors="coerce")
        fitted_scoring[column] = y_score - (intercept + slope * line_score)
    return fitted_training, fitted_scoring


def arm_frames(arm: str, training: pd.DataFrame, scoring: pd.DataFrame, closing: pd.DataFrame):
    """Training and scoring frames plus the feature profile for one arm."""

    if arm == "incumbent":
        return training, scoring, closing, "weak_stack"
    if arm == "C1":
        return (
            attach_spread_regime(training),
            attach_spread_regime(scoring),
            attach_spread_regime(closing),
            "weak_stack_spread_regime",
        )
    if arm == "R1":
        fitted_training, fitted_scoring = orthogonalise_results(training, scoring)
        _unused, fitted_closing = orthogonalise_results(training, closing)
        return fitted_training, fitted_scoring, fitted_closing, "weak_stack"
    raise ValueError(f"Unknown arm {arm!r}")


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
            fit_rows, open_rows, close_rows, profile = arm_frames(arm, training, at_open, at_close)
            model = fit_margin_model(
                fit_rows,
                target="market_residual",
                model_name="ridge",
                feature_profile=profile,
                ridge_alpha=10.0,
            )
            scored = scoring[["game_id"]].copy()
            scored["season"] = int(str(season))
            scored["week"] = int(str(week))
            scored["probability_method"] = "gaussian_median"
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
            scored["residual_at_open_served"] = scored["residual_at_open"] + offsets
            scored["residual_at_close_served"] = scored["residual_at_close"] + offsets
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
        result["pick_home_at_close_probability_rule_raw"] = result[
            "home_cover_probability_at_close_raw"
        ].ge(0.5)
        result["correct_at_open_probability_rule_raw"] = pick_correct(
            result["pick_home_at_open_probability_rule_raw"], result["margin_vs_open"]
        )
        result["correct_at_close_probability_rule_raw"] = pick_correct(
            result["pick_home_at_close_probability_rule_raw"], result["margin_vs_close"]
        )
        oracle = pick_correct(result["open_move"].gt(0.0), result["margin_vs_open"])
        result["oracle_correct_at_open"] = oracle.where(result["open_move"].ne(0.0))
        output[arm] = result.sort_values(["season", "week", "game_id"]).reset_index(drop=True)
    return output


def card_correct(
    per_game: pd.DataFrame,
    data_root: Path,
    features: Path,
    incidents: Path,
    *,
    card: str = "served",
    repo_root: Path = REPO,
):
    """The played card's OR union recomputed against one incoming card."""

    _unused, schedules, player_features, _name, _path = load_inputs_from_frame(per_game, data_root)
    if card == "served":
        flips, _members = served_card_flip_set(
            per_game,
            data_root=data_root,
            repo_root=repo_root,
            features=features,
            incidents=incidents,
            schedules=schedules,
        )
    else:
        predictions = build_predictions_frame(per_game, schedules)
        results = run_overlays(predictions, schedules, player_features)
        flips = set()
        for name in CARD_MEMBERS:
            if name == "player_arrests_back_side_policy":
                arrest_ids, _scored = reconstruct_arrest_flip_set(per_game, features, incidents)
                flips |= arrest_ids
            else:
                flips |= {flip.game_id for flip in results[name].flips}
    base = per_game.set_index("game_id")["correct_at_open_probability_rule"].astype(float)
    flipped = pd.Series(base.index.isin(flips), index=base.index)
    card = pd.Series(np.where(flipped, 1.0 - base, base), index=base.index)
    return card, flips


def load_inputs_from_frame(per_game: pd.DataFrame, data_root: Path):
    """load_inputs' schedule and player joins without re-reading a per_game file."""

    from nfl_ats.snapshots import latest_snapshot, load_snapshot
    from nfl_ats.surgical_gating import VALUE_LOST_DIFF_COLUMNS

    snapshot = latest_snapshot(data_root / "raw")
    schedules, _team_stats = load_snapshot(snapshot)
    player_feature_path = data_root / "processed" / "game_features_player.parquet"
    player_features = pd.read_parquet(
        player_feature_path, columns=["game_id", *VALUE_LOST_DIFF_COLUMNS]
    )
    return per_game, schedules, player_features, snapshot.root.name, player_feature_path


def paired_stats(delta: np.ndarray, blocks: pd.DataFrame, block: str) -> dict[str, float]:
    """overlay_composition's own blocked bootstrap on a paired accuracy delta."""

    stats = blocked_bootstrap_matrix(
        delta[:, np.newaxis], blocks, block=block, samples=SAMPLES, seed=SEED
    )
    return {
        "estimate_accuracy_points": float(stats["estimate"][0] * 100.0),
        "lower_accuracy_points": float(stats["lower"][0] * 100.0),
        "upper_accuracy_points": float(stats["upper"][0] * 100.0),
        "probability_positive": float(stats["probability_positive"][0]),
        "standard_error_accuracy_points": float(stats["standard_error"][0] * 100.0),
        "blocks": int(stats["block_count"]),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--market-root", type=Path, required=True)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--min-train-games", type=int, default=500)
    parser.add_argument("--arms", default="incumbent,C1,R1")
    parser.add_argument("--card", choices=CARD_CHOICES, default="served")
    parser.add_argument("--repo-root", type=Path, default=REPO)
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
    print(json.dumps({"reproduction": reproduction}, indent=2), flush=True)

    features_path = args.data_root / "processed" / "game_features_pbp.parquet"
    incidents = REPO / DEFAULT_INCIDENTS
    if not incidents.is_file():
        incidents = args.data_root / DEFAULT_INCIDENTS.relative_to("data")

    card_series: dict[str, pd.Series] = {}
    flip_counts: dict[str, int] = {}
    for arm in arms:
        frame = cards[arm]
        frame.to_parquet(args.out / f"per_game_{arm}.parquet", index=False)
        series, flips = card_correct(
            frame,
            args.data_root,
            features_path,
            incidents,
            card=args.card,
            repo_root=args.repo_root,
        )
        card_series[arm] = series
        flip_counts[arm] = len(flips)

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
        "card": args.card,
        "card_members": list(SERVED_CARD_MEMBERS if args.card == "served" else CARD_MEMBERS),
        "card_flip_counts": flip_counts,
        "reproduction": reproduction,
        "results": rows,
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
                "season_probability_positive",
            ]
        ].to_string(index=False, float_format=lambda v: f"{v:.4f}")
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
