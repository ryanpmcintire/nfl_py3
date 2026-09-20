from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from nfl_ats.pick_probability import MARKET_MOVE_FEATURE_SUNDAY, MOVE_COLUMN
from nfl_ats.pick_probability_fit import FIT_FEATURES, _design, _fit_logit, _predict, _standardisers
from nfl_ats.sharp_book_movement_features import LEADER_BOOKS, sharp_book_movement_features

QUOTE_CACHE = Path("artifacts/sharp_book_weighted_movement/spread_quotes.parquet")
POPULATION = Path("artifacts/confidence_ranking_audit/20260920_aligned/population.parquet")
OUTPUT = Path("artifacts/sunday_market_probability/20260920_fixed")
SEED = 20260920
BOOTSTRAPS = 10000


def sunday_move(quotes: pd.DataFrame, games: pd.DataFrame) -> pd.DataFrame:
    local = pd.to_datetime(games.week_first_commence_utc, utc=True).dt.tz_convert(
        "America/New_York"
    )
    sunday = local.dt.tz_localize(None).dt.normalize() + pd.to_timedelta(
        (6 - local.dt.weekday) % 7, unit="D"
    )
    monday = (sunday - pd.Timedelta(days=6)).dt.tz_localize("America/New_York").dt.tz_convert("UTC")
    wednesday = (
        (sunday - pd.Timedelta(days=4)).dt.tz_localize("America/New_York").dt.tz_convert("UTC")
    )
    as_of = (
        (sunday + pd.Timedelta(hours=12, minutes=45))
        .dt.tz_localize("America/New_York")
        .dt.tz_convert("UTC")
    )
    bounds = games[["game_id", "commence_time_utc"]].copy()
    bounds["monday"] = monday.to_numpy()
    bounds["wednesday"] = wednesday.to_numpy()
    bounds["cutoff"] = (
        pd.concat([pd.to_datetime(games.commence_time_utc, utc=True), as_of], axis=1)
        .min(axis=1)
        .to_numpy()
    )
    q = quotes.loc[
        quotes.bookmaker_key.isin(LEADER_BOOKS),
        [
            "nflverse_game_id",
            "bookmaker_key",
            "home_spread_line",
            "observed_at_utc",
            "bookmaker_last_update_utc",
        ],
    ].rename(columns={"nflverse_game_id": "game_id"})
    q = q.merge(bounds, on="game_id", validate="many_to_one")
    q = q.loc[
        q.observed_at_utc.ge(q.monday)
        & q.observed_at_utc.lt(q.cutoff)
        & q.bookmaker_last_update_utc.le(q.observed_at_utc)
        & np.isfinite(q.home_spread_line)
    ].copy()
    keys = ["game_id", "bookmaker_key", "observed_at_utc"]
    if q.groupby(keys).home_spread_line.nunique().gt(1).any():
        raise ValueError("Conflicting book lines at one observed timestamp")
    q = q.sort_values(keys).drop_duplicates(keys)
    q["move"] = q.groupby(["game_id", "bookmaker_key"]).home_spread_line.diff()
    q = q.loc[q.observed_at_utc.ge(q.wednesday) & q.move.notna()]
    books = q.groupby(["game_id", "bookmaker_key"], as_index=False).move.sum()
    return (
        books.groupby("game_id")
        .agg(sunday_move=("move", "median"), sunday_books=("move", "size"))
        .reset_index()
    )


def fit(
    frame: pd.DataFrame, train: pd.DataFrame, test: pd.DataFrame
) -> tuple[np.ndarray, dict[str, float]]:
    means, stds = _standardisers(train)
    beta = _fit_logit(_design(train, means, stds), train.home_covered.to_numpy(dtype=float), 1e-3)
    natural = {name: float(beta[i + 1] / stds[name]) for i, name in enumerate(FIT_FEATURES)}
    natural["intercept"] = float(
        beta[0] - sum(beta[i + 1] * means[name] / stds[name] for i, name in enumerate(FIT_FEATURES))
    )
    return _predict(test, beta, means, stds), natural


def metrics(frame: pd.DataFrame, a: str, b: str) -> dict[str, object]:
    target = frame.home_covered.to_numpy(dtype=float)
    p = frame[a].to_numpy(dtype=float)
    q = frame[b].to_numpy(dtype=float)
    outcome = {}
    for name, prob in (
        ("current", p),
        ("sunday", q),
        ("model_only", frame.model_probability.to_numpy(dtype=float)),
        ("market", np.full(len(frame), 0.5)),
    ):
        prob = np.clip(prob, 1e-9, 1 - 1e-9)
        outcome[name] = {
            "games": len(frame),
            "correct": int(((prob >= 0.5) == target).sum()),
            "brier": float(np.mean((prob - target) ** 2)),
            "log_loss": float(np.mean(-target * np.log(prob) - (1 - target) * np.log1p(-prob))),
        }
    week = frame[["season", "week"]].copy()
    week["brier"] = (p - target) ** 2 - (q - target) ** 2
    week["log_loss"] = (
        -target * np.log(np.clip(p, 1e-9, 1 - 1e-9))
        - (1 - target) * np.log1p(-np.clip(p, 1e-9, 1 - 1e-9))
        + target * np.log(np.clip(q, 1e-9, 1 - 1e-9))
        + (1 - target) * np.log1p(-np.clip(q, 1e-9, 1 - 1e-9))
    )
    week["accuracy_points"] = 100 * (
        ((q >= 0.5) == target).astype(float) - ((p >= 0.5) == target).astype(float)
    )
    groups = week.groupby(["season", "week"])[["brier", "log_loss", "accuracy_points"]].agg(
        ["sum", "count"]
    )
    rng = np.random.default_rng(SEED)
    index = rng.integers(0, len(groups), (BOOTSTRAPS, len(groups)))
    deltas = {}
    for metric in ("brier", "log_loss", "accuracy_points"):
        totals = groups[(metric, "sum")].to_numpy()
        counts = groups[(metric, "count")].to_numpy()
        draws = totals[index].sum(axis=1) / counts[index].sum(axis=1)
        deltas[metric] = {
            "effect": float(week[metric].mean()),
            "interval_low": float(np.quantile(draws, 0.025)),
            "interval_high": float(np.quantile(draws, 0.975)),
            "probability_positive": float(np.mean(draws > 0) + 0.5 * np.mean(draws == 0)),
        }
    different = (p >= 0.5) != (q >= 0.5)
    decisive = {
        "games": int(different.sum()),
        "current_correct": int(((p[different] >= 0.5) == target[different]).sum()),
        "sunday_correct": int(((q[different] >= 0.5) == target[different]).sum()),
    }
    reliability = {}
    for name, prob in (("current", p), ("sunday", q)):
        confidence = np.maximum(prob, 1 - prob)
        correct = ((prob >= 0.5) == target).astype(float)
        cells = []
        for low, high in zip(
            (0.5, 0.52, 0.55, 0.58, 0.62), (0.52, 0.55, 0.58, 0.62, 1.0), strict=True
        ):
            select = (confidence >= low) & (confidence < high)
            if select.any():
                cells.append(
                    {
                        "low": low,
                        "high": high,
                        "games": int(select.sum()),
                        "predicted": float(confidence[select].mean()),
                        "actual": float(correct[select].mean()),
                    }
                )
        reliability[name] = cells
    seasons = []
    for season, rows in frame.groupby("season"):
        y = rows.home_covered.to_numpy(dtype=float)
        current = rows[a].to_numpy(dtype=float)
        sunday = rows[b].to_numpy(dtype=float)
        seasons.append(
            {
                "season": int(season),
                "games": len(rows),
                "current_correct": int(((current >= 0.5) == y).sum()),
                "sunday_correct": int(((sunday >= 0.5) == y).sum()),
                "brier_gain": float(np.mean((current - y) ** 2 - (sunday - y) ** 2)),
            }
        )
    return {
        "arms": outcome,
        "sunday_minus_current": deltas,
        "decisive": decisive,
        "reliability": reliability,
        "seasons": seasons,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    population = pd.read_parquet(POPULATION)
    quotes = pd.read_parquet(QUOTE_CACHE)
    quotes = quotes.loc[
        quotes.decision_label.eq("intraday_hourly")
        & quotes.archive_season.isin((2023, 2024, 2025))
        & quotes.nflverse_game_id.isin(set(population.game_id))
    ].copy()
    games = quotes.groupby("nflverse_game_id", as_index=False).agg(
        commence_time_utc=("commence_time_utc", "min")
    )
    games = games.rename(columns={"nflverse_game_id": "game_id"}).merge(
        population[["game_id", "season", "week"]], on="game_id", validate="one_to_one"
    )
    games["week_first_commence_utc"] = games.groupby(
        ["season", "week"]
    ).commence_time_utc.transform("min")
    baseline = sharp_book_movement_features(quotes, games)[
        ["game_id", "leader_median_net_move", "leader_books"]
    ]
    joined = population.merge(baseline, on="game_id", how="left", validate="one_to_one")
    exposed = joined.market_move_available.eq(1.0)
    parity_error = (joined["leader_median_net_move"] - joined[MOVE_COLUMN]).abs()
    parity = float(parity_error.loc[exposed].max())
    parity_mismatch = exposed & parity_error.gt(1e-9)
    eligible = exposed & ~parity_mismatch
    if joined.loc[eligible, "leader_books"].isna().any():
        raise ValueError("Current input reconstruction lacks book counts")
    new = sunday_move(quotes, games)
    joined = joined.merge(new, on="game_id", how="left", validate="one_to_one")
    missing = int(joined.loc[eligible, "sunday_move"].isna().sum())
    if missing:
        raise ValueError(f"Sunday input absent for {missing} currently exposed games")
    joined["current_move"] = joined[MOVE_COLUMN]
    joined["sunday_move_fitted"] = joined[MOVE_COLUMN].where(~eligible, joined.sunday_move)
    change = eligible & joined.current_move.ne(joined.sunday_move_fitted)
    predictions = joined[
        [
            "game_id",
            "season",
            "week",
            "home_covered",
            "model_probability",
            "current_move",
            "sunday_move_fitted",
        ]
    ].copy()
    coefficients = []
    results = {}
    for protocol in ("in_sample", "leave_one_season_out", "chronological"):
        scored = joined.copy()
        for arm, move in (("current", "current_move"), ("sunday", "sunday_move_fitted")):
            scored[MOVE_COLUMN] = scored[move]
            scored[f"{arm}_p"] = np.nan
            for held in sorted(scored.season.unique()):
                train = (
                    scored.loc[
                        scored.season.lt(held)
                        if protocol == "chronological"
                        else scored.season.ne(held)
                    ]
                    if protocol != "in_sample"
                    else scored
                )
                test = scored.loc[scored.season.eq(held)]
                if train.season.nunique() < (2 if protocol == "chronological" else 1):
                    continue
                values, fitted = fit(scored, train, test)
                scored.loc[test.index, f"{arm}_p"] = values
                coefficients.append({"protocol": protocol, "arm": arm, "held": int(held), **fitted})
        valid = scored.dropna(subset=["current_p", "sunday_p"])
        results[protocol] = metrics(valid, "current_p", "sunday_p")
        predictions[f"{protocol}_current_p"] = scored.current_p
        predictions[f"{protocol}_sunday_p"] = scored.sunday_p
    args.output.mkdir(parents=True, exist_ok=True)
    market_move = joined.loc[
        exposed,
        ["game_id", "season", "week", "sunday_move_fitted", "leader_books", "sunday_books"],
    ].rename(columns={"sunday_move_fitted": "leader_median_net"})
    market_move_path = args.output / "market_move.parquet"
    market_move.to_parquet(market_move_path, index=False)
    predictions.to_parquet(args.output / "predictions.parquet", index=False)
    pd.DataFrame(coefficients).to_csv(args.output / "coefficients.csv", index=False)
    summary = {
        "family": "sunday_market_probability_fixed_v1",
        "arms": ["current Sunday blackout", "include Sunday through 12:45 ET or kickoff"],
        "look_count": {
            "structural_arms": 1,
            "protocols": 3,
            "metric_families": 3,
            "reliability_bands_per_arm": 5,
            "coefficient_fits": len(coefficients),
        },
        "source": {
            "population": str(POPULATION),
            "quotes": str(QUOTE_CACHE),
            "parity_max_abs": parity,
            "parity_mismatch_held": joined.loc[parity_mismatch, "game_id"].tolist(),
            "current_exposed": int(exposed.sum()),
            "eligible_exposed": int(eligible.sum()),
            "changed_move": int(change.sum()),
            "sunday_observed": int(joined.loc[eligible, "sunday_books"].gt(0).sum()),
        },
        "market_move_feature_version": MARKET_MOVE_FEATURE_SUNDAY,
        "market_move_artifact": "market_move.parquet",
        "market_move_sha256": hashlib.sha256(market_move_path.read_bytes()).hexdigest(),
        "source_sha256": {
            "population": hashlib.sha256(POPULATION.read_bytes()).hexdigest(),
            "quotes": hashlib.sha256(QUOTE_CACHE.read_bytes()).hexdigest(),
        },
        "results": results,
        "limitations": [
            "Same aligned discrete population and existing composition features; "
            "historical feature choice already used these seasons.",
            "New input is evaluated only where incumbent move exposure exists; "
            "other rows retain their existing values and availability.",
            "Week-block uncertainty conditions on fitted predictions and does "
            "not undo earlier research selection.",
        ],
    }
    (args.output / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "exposures": summary["source"]["current_exposed"],
                "changed_move": summary["source"]["changed_move"],
                "loso_brier_gain": results["leave_one_season_out"]["sunday_minus_current"]["brier"],
                "chronological_brier_gain": results["chronological"]["sunday_minus_current"][
                    "brier"
                ],
            }
        )
    )


if __name__ == "__main__":
    main()
