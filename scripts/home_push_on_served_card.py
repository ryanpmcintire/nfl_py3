"""MOD-18 lane AW: four home-side push settings through the served nine-member card."""

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

from nfl_ats import unserved_tilt_marginals as utm  # noqa: E402
from nfl_ats.clv import (  # noqa: E402
    CLOSE_LABEL_PRIORITY,
    HISTORICAL_CAPTURE_KIND,
    build_pairing_table,
    close_reference_table,
    pick_correct,
)
from nfl_ats.home_side_location import (  # noqa: E402
    HOME_SIDE_OFFSET_BUCKETS,
    fit_home_side_offsets,
    prior_rows_before,
)
from nfl_ats.margin import fit_margin_model  # noqa: E402
from nfl_ats.modeling import regular_season_rows  # noqa: E402
from nfl_ats.overlay_composition import (  # noqa: E402
    DEFAULT_INCIDENTS,
    blocked_bootstrap_matrix,
)
from nfl_ats.snapshots import latest_snapshot, load_snapshot  # noqa: E402
from nfl_ats.spread_regime import BUCKETS, spread_bucket  # noqa: E402
from nfl_ats.unserved_tilt_marginals import SERVED_CARD_MEMBERS  # noqa: E402

SAMPLES = 20_000
SEED = 20260821
COARSE = {"0-3": "0-6.5", "3.5-6.5": "0-6.5", "7": "7", "7.5-10": "7.5-10", "10.5+": "10.5+"}
COARSE_ORDER = ("0-6.5", "7", "7.5-10", "10.5+")
ARM_BUCKETS: dict[str, tuple[str, ...]] = {
    "H0": tuple(HOME_SIDE_OFFSET_BUCKETS),
    "H1": (),
    "H2": tuple(BUCKETS),
    "H3": ("10.5+",),
}
ARMS = ("H0", "H1", "H2", "H3")
ERAS = {
    2020: "2020_2021",
    2021: "2020_2021",
    2022: "2022_2023",
    2023: "2022_2023",
    2024: "2024_2025",
    2025: "2024_2025",
}

_PBP08_CACHE: dict[str, tuple[pd.DataFrame, str]] = {}
_ORIGINAL_PBP08 = utm.build_pbp08_flag_table


def _cached_pbp08(data_root: Path) -> tuple[pd.DataFrame, str]:
    """Memoised PBP-08 flag table so four card rebuilds do not rebuild it four times."""

    key = str(data_root)
    if key not in _PBP08_CACHE:
        _PBP08_CACHE[key] = _ORIGINAL_PBP08(data_root)
    table, name = _PBP08_CACHE[key]
    return table.copy(), name


utm.build_pbp08_flag_table = _cached_pbp08


def walk_forward(
    features: pd.DataFrame, market_root: Path, min_train: int
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """One weekly refit, four push settings scored off the same fit and the same stream."""

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
    stream = pd.DataFrame(columns=stream_columns)
    scored_weeks: list[pd.DataFrame] = []
    offset_log: list[dict[str, Any]] = []

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

        model = fit_margin_model(
            training,
            target="market_residual",
            model_name="ridge",
            feature_profile="weak_stack",
            ridge_alpha=10.0,
        )
        predicted_open = model.predict(at_open, probability_method="gaussian_median")
        scored = scoring[["game_id"]].copy()
        scored["season"] = int(str(season))
        scored["week"] = int(str(week))
        scored["residual_at_open"] = predicted_open["predicted_market_residual"].to_numpy()
        raw_probability = predicted_open["home_cover_probability"].to_numpy()
        scored["home_cover_probability_at_open_raw"] = raw_probability

        row_buckets = spread_bucket(at_open["spread_line"])
        if stream.empty:
            fitted_all = dict.fromkeys(BUCKETS, 0.0)
            prior_games = dict.fromkeys(BUCKETS, 0)
        else:
            fit = fit_home_side_offsets(
                prior_rows_before(stream, int(str(season)), int(str(week))), all_buckets=True
            )
            fitted_all = dict(fit.offsets)
            prior_games = dict(fit.prior_games)
        offset_log.append(
            {
                "season": int(str(season)),
                "week": int(str(week)),
                "offsets": {k: float(v) for k, v in fitted_all.items()},
                "prior_games": {k: int(v) for k, v in prior_games.items()},
            }
        )

        for arm in ARMS:
            allowed = ARM_BUCKETS[arm]
            table = {b: (fitted_all[b] if b in allowed else 0.0) for b in BUCKETS}
            offsets = row_buckets.map(table).astype(float).fillna(0.0).to_numpy()
            scored[f"home_side_offset_{arm}"] = offsets
            if np.any(offsets != 0.0):
                served = model.predict(
                    at_open, probability_method="gaussian_median", center_offset=offsets
                )
                scored[f"home_cover_probability_at_open_{arm}"] = served[
                    "home_cover_probability"
                ].to_numpy()
            else:
                scored[f"home_cover_probability_at_open_{arm}"] = raw_probability

        scored_weeks.append(scored)
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
        stream = (
            week_stream if stream.empty else pd.concat([stream, week_stream], ignore_index=True)
        )
        print(f"scored {season} week {week}", flush=True)

    residuals = pd.concat(scored_weeks, ignore_index=True)
    result = paired.merge(residuals.drop(columns=["season", "week"]), on="game_id", how="inner")
    result["margin_vs_open"] = result["result"] - result["tue_open_home_spread"]
    result["bucket"] = spread_bucket(result["tue_open_home_spread"])
    result["coarse"] = result["bucket"].map(COARSE)
    result["era"] = result["season"].map(ERAS)
    for arm in ARMS:
        probability = result[f"home_cover_probability_at_open_{arm}"]
        result[f"pick_home_{arm}"] = probability.ge(0.5)
        result[f"correct_{arm}"] = pick_correct(
            result[f"pick_home_{arm}"], result["margin_vs_open"]
        )
    return result.sort_values(["season", "week", "game_id"]).reset_index(drop=True), offset_log


def per_game_for_arm(result: pd.DataFrame, arm: str) -> pd.DataFrame:
    """The opener-archive schema every overlay reads, seeded with one arm's probability."""

    frame = result.copy()
    frame["home_cover_probability_at_open"] = frame[f"home_cover_probability_at_open_{arm}"]
    frame["home_side_offset_at_open"] = frame[f"home_side_offset_{arm}"]
    frame["pick_home_at_open_probability_rule"] = frame[f"pick_home_{arm}"]
    frame["correct_at_open_probability_rule"] = frame[f"correct_{arm}"]
    return frame


def card_for_arm(
    result: pd.DataFrame,
    arm: str,
    data_root: Path,
    repo_root: Path,
    features_path: Path,
    incidents: Path,
    schedules: pd.DataFrame,
) -> tuple[pd.Series, set[str], dict[str, set[str]]]:
    """The nine-member OR union recomputed on this arm, plus its correctness series."""

    per_game = per_game_for_arm(result, arm)
    predictions = utm.build_predictions_frame(per_game, schedules)
    members = utm.build_served_card_flip_sets(
        predictions, schedules, per_game, data_root, repo_root, features_path, incidents
    )
    union: set[str] = set()
    for ids in members.values():
        union |= ids
    base = per_game.set_index("game_id")["correct_at_open_probability_rule"].astype(float)
    flipped = pd.Series(base.index.isin(union), index=base.index)
    return pd.Series(np.where(flipped, 1.0 - base, base), index=base.index), union, members


def paired_stats(delta: np.ndarray, blocks: pd.DataFrame, block: str) -> dict[str, float]:
    """The project's own blocked bootstrap on a paired accuracy delta."""

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


def brier_and_log_loss(probability: np.ndarray, outcome: np.ndarray) -> tuple[float, float]:
    """Brier and log loss of a home-cover probability against the realised cover."""

    clipped = np.clip(probability, 1e-12, 1.0 - 1e-12)
    brier = float(np.mean((clipped - outcome) ** 2))
    log_loss = float(-np.mean(outcome * np.log(clipped) + (1.0 - outcome) * np.log(1.0 - clipped)))
    return brier, log_loss


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--market-root", type=Path, required=True)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=REPO)
    parser.add_argument("--min-train-games", type=int, default=500)
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    features = pd.read_parquet(args.features)
    result, offset_log = walk_forward(features, args.market_root, args.min_train_games)

    reference = pd.read_parquet(args.archive / "per_game.parquet")
    check = reference[
        [
            "game_id",
            "tue_open_home_spread",
            "residual_at_open",
            "home_cover_probability_at_open",
            "home_side_offset_at_open",
        ]
    ].merge(
        result[
            [
                "game_id",
                "tue_open_home_spread",
                "residual_at_open",
                "home_cover_probability_at_open_H0",
                "home_side_offset_H0",
            ]
        ],
        on="game_id",
        suffixes=("_archive", "_replay"),
    )
    reproduction = {
        "rows": len(check),
        "max_line_gap": float(
            (check["tue_open_home_spread_archive"] - check["tue_open_home_spread_replay"])
            .abs()
            .max()
        ),
        "max_residual_gap": float(
            (check["residual_at_open_archive"] - check["residual_at_open_replay"]).abs().max()
        ),
        "max_offset_gap": float(
            (check["home_side_offset_at_open"] - check["home_side_offset_H0"]).abs().max()
        ),
        "max_probability_gap": float(
            (check["home_cover_probability_at_open"] - check["home_cover_probability_at_open_H0"])
            .abs()
            .max()
        ),
    }
    print(json.dumps({"reproduction": reproduction}, indent=2), flush=True)

    snapshot = latest_snapshot(args.data_root / "raw")
    schedules, _team_stats = load_snapshot(snapshot)
    features_path = args.data_root / "processed" / "game_features_pbp.parquet"
    incidents = REPO / DEFAULT_INCIDENTS
    if not incidents.is_file():
        incidents = args.data_root / DEFAULT_INCIDENTS.relative_to("data")

    cards: dict[str, pd.Series] = {}
    unions: dict[str, set[str]] = {}
    member_sets: dict[str, dict[str, set[str]]] = {}
    for arm in ARMS:
        series, union, members = card_for_arm(
            result, arm, args.data_root, args.repo_root, features_path, incidents, schedules
        )
        cards[arm] = series
        unions[arm] = union
        member_sets[arm] = members
        print(f"card built for {arm}: {len(union)} union flips", flush=True)

    indexed = result.set_index("game_id")
    frame = indexed[
        ["season", "week", "coarse", "era", "margin_vs_open", "tue_open_home_spread"]
    ].copy()
    for arm in ARMS:
        frame[f"standalone_{arm}"] = indexed[f"correct_{arm}"].astype(float)
        frame[f"card_{arm}"] = cards[arm].reindex(frame.index).astype(float)
        frame[f"pick_home_{arm}"] = indexed[f"pick_home_{arm}"]
        card_flipped = frame.index.isin(unions[arm])
        frame[f"card_pick_home_{arm}"] = np.where(
            card_flipped,
            ~indexed[f"pick_home_{arm}"].to_numpy(),
            indexed[f"pick_home_{arm}"].to_numpy(),
        )
        probability = indexed[f"home_cover_probability_at_open_{arm}"].to_numpy(dtype=float)
        frame[f"probability_{arm}"] = probability
        frame[f"card_probability_{arm}"] = np.where(card_flipped, 1.0 - probability, probability)
        frame[f"offset_{arm}"] = indexed[f"home_side_offset_{arm}"].to_numpy(dtype=float)

    live = frame.loc[frame["standalone_H0"].notna()].copy()
    live["home_cover"] = (live["margin_vs_open"] > 0.0).astype(float)
    live["favourite_home"] = live["tue_open_home_spread"] < 0.0

    rows: list[dict[str, Any]] = []
    scopes: list[tuple[str, str, pd.DataFrame]] = [("overall", "2020_2025", live)]
    for bucket in COARSE_ORDER:
        block = live.loc[live["coarse"].eq(bucket)]
        if not block.empty:
            scopes.append((bucket, "2020_2025", block))
    for era in ("2020_2021", "2022_2023", "2024_2025"):
        block = live.loc[live["era"].eq(era)]
        if not block.empty:
            scopes.append(("overall", era, block))
    for bucket in COARSE_ORDER:
        for era in ("2020_2021", "2022_2023", "2024_2025"):
            block = live.loc[live["coarse"].eq(bucket) & live["era"].eq(era)]
            if not block.empty:
                scopes.append((bucket, era, block))

    for surface in ("card", "standalone"):
        for scope, window, subset in scopes:
            baseline = subset[f"{surface}_H0"].to_numpy(dtype=float)
            blocks = subset[["season", "week"]].reset_index(drop=True)
            for arm in ("H1", "H2", "H3"):
                candidate = subset[f"{surface}_{arm}"].to_numpy(dtype=float)
                delta = candidate - baseline
                row: dict[str, Any] = {
                    "arm": arm,
                    "surface": surface,
                    "scope": scope,
                    "window": window,
                    "n": len(subset),
                    "baseline_accuracy": float(baseline.mean()),
                    "candidate_accuracy": float(candidate.mean()),
                    "picks_changed": int(np.count_nonzero(delta != 0.0)),
                }
                row.update({f"week_{k}": v for k, v in paired_stats(delta, blocks, "week").items()})
                row.update(
                    {f"season_{k}": v for k, v in paired_stats(delta, blocks, "season").items()}
                )
                rows.append(row)

    descriptive: list[dict[str, Any]] = []
    for scope, window, subset in scopes:
        for arm in ARMS:
            outcome = subset["home_cover"].to_numpy(dtype=float)
            model_brier, model_log = brier_and_log_loss(
                subset[f"probability_{arm}"].to_numpy(dtype=float), outcome
            )
            card_brier, card_log = brier_and_log_loss(
                subset[f"card_probability_{arm}"].to_numpy(dtype=float), outcome
            )
            picks_home = subset[f"pick_home_{arm}"].to_numpy(dtype=bool)
            card_picks_home = subset[f"card_pick_home_{arm}"].to_numpy(dtype=bool)
            favourite_home = subset["favourite_home"].to_numpy(dtype=bool)
            descriptive.append(
                {
                    "arm": arm,
                    "scope": scope,
                    "window": window,
                    "n": len(subset),
                    "standalone_accuracy": float(subset[f"standalone_{arm}"].mean()),
                    "card_accuracy": float(subset[f"card_{arm}"].mean()),
                    "home_pick_share": float(picks_home.mean()),
                    "card_home_pick_share": float(card_picks_home.mean()),
                    "favourite_pick_share": float((picks_home == favourite_home).mean()),
                    "card_favourite_pick_share": float((card_picks_home == favourite_home).mean()),
                    "mean_offset": float(subset[f"offset_{arm}"].mean()),
                    "model_brier": model_brier,
                    "model_log_loss": model_log,
                    "card_brier": card_brier,
                    "card_log_loss": card_log,
                }
            )

    push_flip_ids = set(
        live.index[live["pick_home_H0"].to_numpy() != live["pick_home_H1"].to_numpy()]
    )
    push_flip_mask = live.index.isin(push_flip_ids)
    overlap_members = {name: len(ids & push_flip_ids) for name, ids in member_sets["H0"].items()}
    final_same = int(
        np.count_nonzero(
            live.loc[push_flip_mask, "card_pick_home_H0"].to_numpy()
            == live.loc[push_flip_mask, "card_pick_home_H1"].to_numpy()
        )
    )
    push_subset = live.loc[push_flip_mask]
    overlap = {
        "push_flips_on_scored_games": int(push_flip_mask.sum()),
        "push_flips_by_bucket": {
            str(bucket): int(count)
            for bucket, count in push_subset["coarse"].value_counts().items()
        },
        "push_flips_also_in_nine_member_union": len(unions["H0"] & push_flip_ids),
        "push_flips_per_member": overlap_members,
        "card_cancels_the_push_same_final_pick": final_same,
        "push_survives_to_the_played_card": int(push_flip_mask.sum()) - final_same,
        "h0_card_accuracy_on_push_flips": float(push_subset["card_H0"].mean())
        if len(push_subset)
        else float("nan"),
        "h1_card_accuracy_on_push_flips": float(push_subset["card_H1"].mean())
        if len(push_subset)
        else float("nan"),
        "h0_standalone_accuracy_on_push_flips": float(push_subset["standalone_H0"].mean())
        if len(push_subset)
        else float("nan"),
        "union_flip_counts": {arm: len(unions[arm]) for arm in ARMS},
        "member_flip_counts": {
            arm: {name: len(ids) for name, ids in member_sets[arm].items()} for arm in ARMS
        },
    }

    control_rows: list[dict[str, Any]] = []
    for scope, window, subset in scopes:
        mask = subset.index.isin(push_flip_ids)
        if not mask.any():
            continue
        blocks = subset[["season", "week"]].reset_index(drop=True)
        for baseline_arm in ("H0", "H1"):
            baseline = subset[f"card_{baseline_arm}"].to_numpy(dtype=float)
            perfect = np.where(mask, 1.0, baseline)
            delta = perfect - baseline
            control_rows.append(
                {
                    "control": "perfect_foresight_on_push_flip_set",
                    "baseline_arm": baseline_arm,
                    "scope": scope,
                    "window": window,
                    "n": len(subset),
                    "touched": int(mask.sum()),
                    "baseline_accuracy": float(baseline.mean()),
                    "control_accuracy": float(perfect.mean()),
                    **{f"week_{k}": v for k, v in paired_stats(delta, blocks, "week").items()},
                    **{f"season_{k}": v for k, v in paired_stats(delta, blocks, "season").items()},
                }
            )

    table = pd.DataFrame(rows)
    table.to_csv(args.out / "arm_results.csv", index=False)
    pd.DataFrame(descriptive).to_csv(args.out / "descriptive.csv", index=False)
    pd.DataFrame(control_rows).to_csv(args.out / "positive_control.csv", index=False)
    live.reset_index().to_parquet(args.out / "per_game_arms.parquet", index=False)

    payload = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "archive": str(args.archive),
        "features": str(args.features),
        "schedule_snapshot": snapshot.root.name,
        "arms": {arm: list(ARM_BUCKETS[arm]) for arm in ARMS},
        "card_members": list(SERVED_CARD_MEMBERS),
        "samples": SAMPLES,
        "seed": SEED,
        "reproduction": reproduction,
        "n_archive_games": len(result),
        "n_scored_games": len(live),
        "overlap": overlap,
        "results": rows,
        "descriptive": descriptive,
        "positive_control": control_rows,
        "weekly_offsets": offset_log,
    }
    (args.out / "arm_results.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(
        table.loc[table["window"].eq("2020_2025")][
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
    print(json.dumps({"overlap": overlap}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
