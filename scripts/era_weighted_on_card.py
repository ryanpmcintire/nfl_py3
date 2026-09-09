"""Score the era-weighted refit arms through the served card at the Tuesday opener."""

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
from nfl_ats.era_weighted_half_life_8_overlay import (  # noqa: E402
    _prepare_sorted_training,
    _target_values,
    fit_weighted_ridge_margin,
    half_life_weights,
)
from nfl_ats.home_side_location import fit_home_side_offsets, prior_rows_before  # noqa: E402
from nfl_ats.margin import fit_margin_model, margin_feature_columns  # noqa: E402
from nfl_ats.modeling import regular_season_rows  # noqa: E402
from nfl_ats.overlay_composition import (  # noqa: E402
    DEFAULT_INCIDENTS,
    blocked_bootstrap_matrix,
    build_predictions_frame,
    reconstruct_arrest_flip_set,
    run_overlays,
)
from nfl_ats.spread_regime import spread_bucket  # noqa: E402

CARD_MEMBERS = (
    "coach_fade_overlay",
    "division_revenge_tilt_overlay",
    "player_arrests_back_side_policy",
)
COARSE = {"0-3": "0-6.5", "3.5-6.5": "0-6.5", "7": "7", "7.5-10": "7.5-10", "10.5+": "10.5+"}
HALF_LIVES = {"hl4": 4.0, "hl8": 8.0, "hl16": 16.0}
FEATURE_PROFILE = "weak_stack"
RIDGE_ALPHA = 10.0
SAMPLES = 20_000
SEED = 20260821
PROBABILITY_FLOOR = 1e-12


def fit_arm(arm: str, training: pd.DataFrame, season: int):
    """One weekly refit for one arm; only the sample-weight vector varies."""

    if arm == "incumbent":
        return fit_margin_model(
            training,
            target="market_residual",
            model_name="ridge",
            feature_profile=FEATURE_PROFILE,
            ridge_alpha=RIDGE_ALPHA,
        )
    sorted_frame = _prepare_sorted_training(regular_season_rows(training))
    target_values = _target_values(sorted_frame).to_numpy(dtype=float)
    columns = margin_feature_columns("market_residual", FEATURE_PROFILE)
    if arm == "uniform":
        weights = np.ones(len(sorted_frame), dtype=float)
    else:
        weights = half_life_weights(
            sorted_frame["season"].to_numpy(dtype=float),
            predict_season=season,
            half_life=HALF_LIVES[arm],
        )
    return fit_weighted_ridge_margin(
        sorted_frame,
        target=target_values,
        feature_columns=columns,
        weights=weights,
        ridge_alpha=RIDGE_ALPHA,
        model_name="ridge",
    )


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
            model = fit_arm(arm, training, int(str(season)))
            scored = scoring[["game_id"]].copy()
            scored["season"] = int(str(season))
            scored["week"] = int(str(week))
            scored["probability_method"] = "gaussian_median"
            predicted_open = model.predict(at_open, probability_method="gaussian_median")
            predicted_close = model.predict(at_close, probability_method="gaussian_median")
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
                offsets = fitted.offset_for(at_open["spread_line"]).fillna(0.0).to_numpy(float)
            else:
                offsets = np.zeros(len(at_open), dtype=float)
            scored["home_side_offset_at_open"] = offsets
            if np.any(offsets != 0.0):
                served_open = model.predict(
                    at_open, probability_method="gaussian_median", center_offset=offsets
                )
                served_close = model.predict(
                    at_close, probability_method="gaussian_median", center_offset=offsets
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
        result["pick_home_at_open_probability_rule"] = result["home_cover_probability_at_open"].ge(
            0.5
        )
        result["correct_at_open_probability_rule"] = pick_correct(
            result["pick_home_at_open_probability_rule"], result["margin_vs_open"]
        )
        result["pick_home_at_open_probability_rule_raw"] = result[
            "home_cover_probability_at_open_raw"
        ].ge(0.5)
        result["correct_at_open_probability_rule_raw"] = pick_correct(
            result["pick_home_at_open_probability_rule_raw"], result["margin_vs_open"]
        )
        output[arm] = result.sort_values(["season", "week", "game_id"]).reset_index(drop=True)
    return output


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


def card_flip_set(per_game: pd.DataFrame, data_root: Path, features: Path, incidents: Path):
    """Played three-member OR flip set recomputed against one incoming card."""

    _unused, schedules, player_features, _name, _path = load_inputs_from_frame(per_game, data_root)
    predictions = build_predictions_frame(per_game, schedules)
    results = run_overlays(predictions, schedules, player_features)
    flips: set[str] = set()
    for name in CARD_MEMBERS:
        if name == "player_arrests_back_side_policy":
            arrest_ids, _scored = reconstruct_arrest_flip_set(per_game, features, incidents)
            flips |= arrest_ids
        else:
            flips |= {flip.game_id for flip in results[name].flips}
    return flips


def surface_series(per_game: pd.DataFrame, flips: set[str]) -> pd.DataFrame:
    """Correctness and served home-cover probability on the standalone and card surfaces."""

    indexed = per_game.set_index("game_id")
    correct = indexed["correct_at_open_probability_rule"].astype(float)
    probability = indexed["home_cover_probability_at_open"].astype(float)
    flipped = pd.Series(indexed.index.isin(flips), index=indexed.index)
    return pd.DataFrame(
        {
            "standalone_correct": correct,
            "standalone_probability": probability,
            "card_correct": pd.Series(
                np.where(flipped, 1.0 - correct, correct), index=correct.index
            ),
            "card_probability": pd.Series(
                np.where(flipped, 1.0 - probability, probability), index=probability.index
            ),
        }
    )


def paired_stats(delta: np.ndarray, blocks: pd.DataFrame, block: str, scale: float):
    """The composition study's own blocked bootstrap on a paired per-game delta."""

    stats = blocked_bootstrap_matrix(
        delta[:, np.newaxis], blocks, block=block, samples=SAMPLES, seed=SEED
    )
    return {
        "estimate": float(stats["estimate"][0] * scale),
        "lower": float(stats["lower"][0] * scale),
        "upper": float(stats["upper"][0] * scale),
        "probability_positive": float(stats["probability_positive"][0]),
        "standard_error": float(stats["standard_error"][0] * scale),
        "blocks": int(stats["block_count"]),
    }


def brier_terms(probability: pd.Series, outcome: pd.Series) -> pd.Series:
    return (probability - outcome) ** 2


def log_loss_terms(probability: pd.Series, outcome: pd.Series) -> pd.Series:
    clipped = probability.clip(PROBABILITY_FLOOR, 1.0 - PROBABILITY_FLOOR)
    return -(outcome * np.log(clipped) + (1.0 - outcome) * np.log(1.0 - clipped))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--market-root", type=Path, required=True)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--min-train-games", type=int, default=500)
    parser.add_argument("--arms", default="incumbent,uniform,hl4,hl8,hl16")
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    arms = tuple(args.arms.split(","))
    cached = {arm: args.out / f"per_game_{arm}.parquet" for arm in arms}
    if all(path.is_file() for path in cached.values()):
        cards = {arm: pd.read_parquet(path) for arm, path in cached.items()}
        print("reusing cached per-arm walk-forward frames", flush=True)
    else:
        features = pd.read_parquet(args.features)
        cards = walk_forward(features, args.market_root, arms, args.min_train_games)
        for arm, path in cached.items():
            cards[arm].to_parquet(path, index=False)

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
    uniform_gate: dict[str, Any] = {}
    if "uniform" in arms:
        gate = incumbent[["game_id", "residual_at_open", "home_cover_probability_at_open"]].merge(
            cards["uniform"][["game_id", "residual_at_open", "home_cover_probability_at_open"]],
            on="game_id",
            suffixes=("_incumbent", "_uniform"),
        )
        residual_gap = float(
            (gate["residual_at_open_incumbent"] - gate["residual_at_open_uniform"]).abs().max()
        )
        probability_gap = float(
            (
                gate["home_cover_probability_at_open_incumbent"]
                - gate["home_cover_probability_at_open_uniform"]
            )
            .abs()
            .max()
        )
        uniform_gate = {
            "rows": len(gate),
            "max_residual_gap": residual_gap,
            "max_probability_gap": probability_gap,
            "passes_atol_1e_9": bool(residual_gap <= 1e-9 and probability_gap <= 1e-9),
        }
    print(
        json.dumps({"reproduction": reproduction, "uniform_gate": uniform_gate}, indent=2),
        flush=True,
    )
    if uniform_gate and not uniform_gate["passes_atol_1e_9"]:
        raise SystemExit("uniform-weight reproduction gate failed; no weighted arm is interpreted")

    features_path = args.data_root / "processed" / "game_features_pbp.parquet"
    incidents = REPO / DEFAULT_INCIDENTS
    if not incidents.is_file():
        incidents = args.data_root / DEFAULT_INCIDENTS.relative_to("data")

    surfaces: dict[str, pd.DataFrame] = {}
    flip_counts: dict[str, int] = {}
    for arm in arms:
        frame = cards[arm]
        flips = card_flip_set(frame, args.data_root, features_path, incidents)
        flip_counts[arm] = len(flips)
        surfaces[arm] = surface_series(frame, flips)

    base_frame = cards["incumbent"][["game_id", "season", "week", "tue_open_home_spread"]].copy()
    base_frame["bucket"] = spread_bucket(base_frame["tue_open_home_spread"])
    base_frame["coarse"] = base_frame["bucket"].map(COARSE)
    base_frame["home_cover_at_open"] = (
        cards["incumbent"]["margin_vs_open"].gt(0.0).astype(float).to_numpy()
    )
    base_frame["push_at_open"] = cards["incumbent"]["margin_vs_open"].eq(0.0).to_numpy()
    base_frame = base_frame.set_index("game_id")

    rows: list[dict[str, Any]] = []
    probability_rows: list[dict[str, Any]] = []
    for arm in arms:
        if arm in {"incumbent", "uniform"}:
            continue
        for surface in ("card", "standalone"):
            candidate = surfaces[arm][f"{surface}_correct"]
            baseline = surfaces["incumbent"][f"{surface}_correct"]
            candidate_probability = surfaces[arm][f"{surface}_probability"]
            baseline_probability = surfaces["incumbent"][f"{surface}_probability"]
            joined = pd.concat(
                {
                    "candidate": candidate,
                    "baseline": baseline,
                    "candidate_probability": candidate_probability,
                    "baseline_probability": baseline_probability,
                },
                axis=1,
            ).join(base_frame, how="inner")
            live = joined.loc[joined["candidate"].notna() & joined["baseline"].notna()]
            scopes: list[tuple[str, str, pd.DataFrame]] = [("overall", "overall", live)]
            scopes += [
                ("bucket", str(bucket), block)
                for bucket, block in live.groupby("coarse", sort=False)
            ]
            scopes += [
                ("season", str(int(season)), block)
                for season, block in live.groupby("season", sort=True)
            ]
            for scope_kind, scope, subset in scopes:
                if subset.empty:
                    continue
                delta = (subset["candidate"] - subset["baseline"]).to_numpy(dtype=float)
                blocks = subset[["season", "week"]].reset_index(drop=True)
                row: dict[str, Any] = {
                    "name": f"era_weighted_{arm}_{surface}_{scope}",
                    "arm": arm,
                    "half_life": HALF_LIVES[arm],
                    "surface": surface,
                    "scope_kind": scope_kind,
                    "scope": scope,
                    "n": len(subset),
                    "baseline_accuracy": float(subset["baseline"].mean()),
                    "candidate_accuracy": float(subset["candidate"].mean()),
                    "picks_changed": int((subset["candidate"] != subset["baseline"]).sum()),
                }
                week_stats = paired_stats(delta, blocks, "week", 100.0)
                season_stats = paired_stats(delta, blocks, "season", 100.0)
                row.update({f"week_{key}": value for key, value in week_stats.items()})
                row.update({f"season_{key}": value for key, value in season_stats.items()})
                row["season_degenerate"] = bool(season_stats["blocks"] < 2)
                rows.append(row)

                outcome = subset["home_cover_at_open"]
                brier_delta = (
                    brier_terms(subset["baseline_probability"], outcome)
                    - brier_terms(subset["candidate_probability"], outcome)
                ).to_numpy(dtype=float)
                log_delta = (
                    log_loss_terms(subset["baseline_probability"], outcome)
                    - log_loss_terms(subset["candidate_probability"], outcome)
                ).to_numpy(dtype=float)
                probability_row: dict[str, Any] = {
                    "name": f"era_weighted_{arm}_{surface}_{scope}",
                    "arm": arm,
                    "surface": surface,
                    "scope_kind": scope_kind,
                    "scope": scope,
                    "n": len(subset),
                    "baseline_brier": float(
                        brier_terms(subset["baseline_probability"], outcome).mean()
                    ),
                    "candidate_brier": float(
                        brier_terms(subset["candidate_probability"], outcome).mean()
                    ),
                    "baseline_log_loss": float(
                        log_loss_terms(subset["baseline_probability"], outcome).mean()
                    ),
                    "candidate_log_loss": float(
                        log_loss_terms(subset["candidate_probability"], outcome).mean()
                    ),
                }
                brier_stats = paired_stats(brier_delta, blocks, "week", 1.0)
                log_stats = paired_stats(log_delta, blocks, "week", 1.0)
                probability_row.update(
                    {f"brier_improvement_{key}": value for key, value in brier_stats.items()}
                )
                probability_row.update(
                    {f"log_loss_improvement_{key}": value for key, value in log_stats.items()}
                )
                probability_rows.append(probability_row)

    table = pd.DataFrame(rows)
    table.to_csv(args.out / "arm_results.csv", index=False)
    probability_table = pd.DataFrame(probability_rows)
    probability_table.to_csv(args.out / "probability_scores.csv", index=False)

    payload = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "archive": str(args.archive),
        "features": str(args.features),
        "arms": list(arms),
        "half_lives": HALF_LIVES,
        "samples": SAMPLES,
        "seed": SEED,
        "card_members": list(CARD_MEMBERS),
        "card_flip_counts": flip_counts,
        "reproduction": reproduction,
        "uniform_gate": uniform_gate,
        "results": rows,
        "probability_results": probability_rows,
    }
    (args.out / "arm_results.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(
        table.loc[table["scope_kind"].eq("overall")][
            [
                "name",
                "n",
                "baseline_accuracy",
                "candidate_accuracy",
                "picks_changed",
                "week_estimate",
                "week_lower",
                "week_upper",
                "week_probability_positive",
                "season_probability_positive",
            ]
        ].to_string(index=False, float_format=lambda v: f"{v:.4f}")
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
