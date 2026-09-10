"""Score the expected-lineup-loss refit through the served nine-member card at the opener."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO = Path(r"F:\Repos\nfl_py3")
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

from nfl_ats.clv import (  # noqa: E402
    CLOSE_LABEL_PRIORITY,
    HISTORICAL_CAPTURE_KIND,
    build_pairing_table,
    close_reference_table,
    pick_correct,
)
from nfl_ats.expected_lineup_loss_challenger import (  # noqa: E402
    CANDIDATE_FEATURE_PROFILE,
    candidate_profile,
)
from nfl_ats.expected_lineup_loss_features import (  # noqa: E402
    EXPECTED_LINEUP_LOSS_COLUMNS,
    LINEUP_GROUPS,
    attach_expected_lineup_loss_features,
    team_season_split_half_reliability,
)
from nfl_ats.home_side_location import fit_home_side_offsets, prior_rows_before  # noqa: E402
from nfl_ats.margin import fit_margin_model, margin_feature_columns  # noqa: E402
from nfl_ats.modeling import regular_season_rows  # noqa: E402
from nfl_ats.overlay_composition import DEFAULT_INCIDENTS, blocked_bootstrap_matrix  # noqa: E402
from nfl_ats.spread_regime import spread_bucket  # noqa: E402
from nfl_ats.unserved_tilt_marginals import SERVED_CARD_MEMBERS, served_card_flip_set  # noqa: E402

COARSE = {"0-3": "0-6.5", "3.5-6.5": "0-6.5", "7": "7", "7.5-10": "7.5-10", "10.5+": "10.5+"}
SAMPLES = 20_000
SEED = 20260821
PROBABILITY_FLOOR = 1e-12
REFRESH_SEASONS = (2023, 2024, 2025)


def build_augmented(base_path: Path, panel_path: Path, out: Path) -> tuple[Path, dict[str, Any]]:
    """The challenger's own feature build, written under this lane's artifact root."""

    from nfl_ats.provenance import sha256_file

    destination = out / "features_lineup_loss.parquet"
    build_path = out / "build.json"
    if destination.is_file() and build_path.is_file():
        return destination, json.loads(build_path.read_text(encoding="utf-8"))
    base = pd.read_parquet(base_path)
    manifest = json.loads(base_path.with_suffix(".manifest.json").read_text(encoding="utf-8"))
    snapshot = manifest["source_player_snapshot"]
    injuries_path = REPO / "data/players/raw" / snapshot / "injuries.parquet"
    panel = pd.read_parquet(panel_path)
    features = attach_expected_lineup_loss_features(
        base, panel=panel, injuries=pd.read_parquet(injuries_path)
    )
    teams = (
        pd.concat(
            [
                features[
                    [
                        "season",
                        "week",
                        f"{side}_team",
                        *[f"{side}_expected_lineup_loss_{group}" for group in LINEUP_GROUPS],
                    ]
                ].rename(
                    columns={
                        f"{side}_team": "team",
                        **{
                            f"{side}_expected_lineup_loss_{group}": f"expected_lineup_loss_{group}"
                            for group in LINEUP_GROUPS
                        },
                    }
                )
                for side in ("home", "away")
            ],
            ignore_index=True,
        )
        .dropna()
        .drop_duplicates(["season", "week", "team"])
    )
    coverage = {}
    for season, rows in features.groupby("season"):
        valid = rows[list(EXPECTED_LINEUP_LOSS_COLUMNS)].notna().all(axis=1)
        coverage[str(int(season))] = {"games": len(rows), "covered_games": int(valid.sum())}
    features.to_parquet(destination, index=False)
    teams.to_parquet(out / "team_week_loss.parquet", index=False)
    metadata = {
        "reliability": team_season_split_half_reliability(teams),
        "coverage": coverage,
        "panel_rows": len(panel),
        "panel_seasons": sorted(int(value) for value in panel["season"].unique()),
        "base_features": str(base_path),
        "base_sha256": sha256_file(base_path),
        "panel_sha256": sha256_file(panel_path),
        "injuries_path": str(injuries_path),
        "injuries_sha256": sha256_file(injuries_path),
        "player_snapshot": snapshot,
        "decision": "min(kickoff, Sunday 16:00 America/New_York)",
        "legacy_depth_assumption": "Pregame week proxy; exact observation times unavailable",
    }
    build_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2), flush=True)
    return destination, metadata


def profile_identity() -> dict[str, Any]:
    """Gate 2: the candidate tuple is weak_stack plus exactly the three frozen columns."""

    baseline = margin_feature_columns("market_residual", "weak_stack")
    candidate = margin_feature_columns("market_residual", CANDIDATE_FEATURE_PROFILE)
    if tuple(candidate) != (*baseline, *EXPECTED_LINEUP_LOSS_COLUMNS):
        raise SystemExit("candidate profile is not weak_stack plus exactly the three columns")
    return {
        "baseline_columns": len(baseline),
        "candidate_columns": len(candidate),
        "added_columns": list(EXPECTED_LINEUP_LOSS_COLUMNS),
    }


def fit_arm(arm: str, training: pd.DataFrame):
    """One weekly refit for one arm; only the feature profile varies."""

    profile = "weak_stack" if arm == "incumbent" else CANDIDATE_FEATURE_PROFILE
    return fit_margin_model(
        training,
        target="market_residual",
        model_name="ridge",
        feature_profile=profile,
        ridge_alpha=10.0,
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

        for arm in arms:
            model = fit_arm(arm, training)
            scored = scoring[["game_id"]].copy()
            scored["season"] = int(str(season))
            scored["week"] = int(str(week))
            scored["probability_method"] = "gaussian_median"
            predicted_open = model.predict(at_open, probability_method="gaussian_median")
            scored["residual_at_open"] = predicted_open["predicted_market_residual"].to_numpy()
            scored["home_cover_probability_at_open_raw"] = predicted_open[
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
                scored["home_cover_probability_at_open"] = served_open[
                    "home_cover_probability"
                ].to_numpy()
            else:
                scored["home_cover_probability_at_open"] = scored[
                    "home_cover_probability_at_open_raw"
                ]
            scored["residual_at_open_served"] = scored["residual_at_open"] + offsets
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
        result["spread_line"] = result["tue_open_home_spread"]
        result["home_cover_probability"] = result["home_cover_probability_at_open"]
        result["pick_home_at_open_probability_rule"] = result["home_cover_probability_at_open"].ge(
            0.5
        )
        result["correct_at_open_probability_rule"] = pick_correct(
            result["pick_home_at_open_probability_rule"], result["margin_vs_open"]
        )
        output[arm] = result.sort_values(["season", "week", "game_id"]).reset_index(drop=True)
    return output


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
    parser.add_argument(
        "--panel", type=Path, default=REPO / "data/processed/play_probability_panel.parquet"
    )
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--market-root", type=Path, required=True)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--min-train-games", type=int, default=500)
    parser.add_argument("--arms", default="incumbent,lineup_loss")
    parser.add_argument("--repo-root", type=Path, default=REPO)
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    arms = tuple(args.arms.split(","))
    augmented_path, build = build_augmented(args.features, args.panel, args.out)

    with candidate_profile():
        identity = profile_identity()
    print(json.dumps({"profile_identity": identity}, indent=2), flush=True)

    cached = {arm: args.out / f"per_game_{arm}.parquet" for arm in arms}
    if all(path.is_file() for path in cached.values()):
        cards = {arm: pd.read_parquet(path) for arm, path in cached.items()}
        print("reusing cached per-arm walk-forward frames", flush=True)
    else:
        features = pd.read_parquet(augmented_path)
        with candidate_profile():
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
    print(json.dumps({"reproduction": reproduction}, indent=2), flush=True)

    augmented = pd.read_parquet(augmented_path, columns=["game_id", *EXPECTED_LINEUP_LOSS_COLUMNS])
    covered = set(
        augmented.loc[
            augmented[list(EXPECTED_LINEUP_LOSS_COLUMNS)].notna().all(axis=1), "game_id"
        ].astype(str)
    )
    archive_ids = set(incumbent["game_id"].astype(str))
    uncovered = sorted(archive_ids - covered)
    print(json.dumps({"archive_games_without_loss_features": len(uncovered)}), flush=True)

    features_path = args.data_root / "processed" / "game_features_pbp.parquet"
    incidents = REPO / DEFAULT_INCIDENTS
    if not incidents.is_file():
        incidents = args.data_root / DEFAULT_INCIDENTS.relative_to("data")

    surfaces: dict[str, pd.DataFrame] = {}
    flip_counts: dict[str, int] = {}
    member_counts: dict[str, dict[str, int]] = {}
    for arm in arms:
        frame = cards[arm]
        if uncovered:
            frame = frame.loc[~frame["game_id"].astype(str).isin(set(uncovered))].copy()
        flips, members = served_card_flip_set(
            frame,
            data_root=args.data_root,
            repo_root=args.repo_root,
            features=features_path,
            incidents=incidents,
            card="served",
        )
        flip_counts[arm] = len(flips)
        member_counts[arm] = {name: len(ids) for name, ids in members.items()}
        surfaces[arm] = surface_series(frame, flips)

    base = cards["incumbent"]
    if uncovered:
        base = base.loc[~base["game_id"].astype(str).isin(set(uncovered))].copy()
    base_frame = base[["game_id", "season", "week", "tue_open_home_spread"]].copy()
    base_frame["bucket"] = spread_bucket(base_frame["tue_open_home_spread"])
    base_frame["coarse"] = base_frame["bucket"].map(COARSE)
    base_frame["home_cover_at_open"] = base["margin_vs_open"].gt(0.0).astype(float).to_numpy()
    base_frame["push_at_open"] = base["margin_vs_open"].eq(0.0).to_numpy()
    base_frame = base_frame.set_index("game_id")

    rows: list[dict[str, Any]] = []
    probability_rows: list[dict[str, Any]] = []
    for arm in arms:
        if arm == "incumbent":
            continue
        for surface in ("card", "standalone"):
            joined = pd.concat(
                {
                    "candidate": surfaces[arm][f"{surface}_correct"],
                    "baseline": surfaces["incumbent"][f"{surface}_correct"],
                    "candidate_probability": surfaces[arm][f"{surface}_probability"],
                    "baseline_probability": surfaces["incumbent"][f"{surface}_probability"],
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
            scopes.append(
                (
                    "refresh",
                    "refresh_2023_2025",
                    live.loc[live["season"].isin(REFRESH_SEASONS)],
                )
            )
            for scope_kind, scope, subset in scopes:
                if subset.empty:
                    continue
                delta = (subset["candidate"] - subset["baseline"]).to_numpy(dtype=float)
                blocks = subset[["season", "week"]].reset_index(drop=True)
                row: dict[str, Any] = {
                    "name": f"lineup_loss_on_card_{surface}_{scope}",
                    "arm": arm,
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
                    "name": f"lineup_loss_on_card_{surface}_{scope}",
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
                probability_row.update(
                    {
                        f"brier_improvement_{key}": value
                        for key, value in paired_stats(brier_delta, blocks, "week", 1.0).items()
                    }
                )
                probability_row.update(
                    {
                        f"log_loss_improvement_{key}": value
                        for key, value in paired_stats(log_delta, blocks, "week", 1.0).items()
                    }
                )
                probability_rows.append(probability_row)

    table = pd.DataFrame(rows)
    table.to_csv(args.out / "arm_results.csv", index=False)
    probability_table = pd.DataFrame(probability_rows)
    probability_table.to_csv(args.out / "probability_scores.csv", index=False)

    payload = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "archive": str(args.archive),
        "features": str(augmented_path),
        "arms": list(arms),
        "samples": SAMPLES,
        "seed": SEED,
        "card": "served",
        "card_members": list(SERVED_CARD_MEMBERS),
        "card_flip_counts": flip_counts,
        "card_member_flip_counts": member_counts,
        "profile_identity": identity,
        "reproduction": reproduction,
        "archive_games_without_loss_features": uncovered,
        "build": build,
        "results": rows,
        "probability_results": probability_rows,
    }
    (args.out / "arm_results.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(
        table.loc[table["scope_kind"].isin(("overall", "refresh"))][
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
