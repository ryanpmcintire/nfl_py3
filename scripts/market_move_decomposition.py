from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from nfl_ats.pick_probability import FLAG_SUM_COLUMN, MOVE_AVAILABLE_COLUMN, MOVE_COLUMN
from nfl_ats.pick_probability_fit import (
    FIT_RIDGE,
    MARKET_MOVE_COLUMN,
    _design,
    _fit_logit,
    _natural_coefficients,
    _predict,
    _standardisers,
    build_fit_population,
)
import nfl_ats.pick_probability_fit as ppf
from nfl_ats.sharp_book_movement_features import LEADER_BOOKS, LEADERSHIP_WEIGHTS

REPO = Path(__file__).resolve().parents[1]
ARTIFACTS_ROOT = REPO / "artifacts"
DATA_ROOT = REPO / "data"
QUOTES_CACHE = ARTIFACTS_ROOT / "sharp_book_weighted_movement" / "spread_quotes.parquet"
SERVED_PER_GAME = ARTIFACTS_ROOT / "sharp_weighted_follow" / "20260909T233611Z" / "per_game.parquet"

OUTER_SEASONS = [2020, 2021, 2022, 2023, 2024, 2025]
ALL_BOOKS = tuple(sorted(LEADERSHIP_WEIGHTS))
KEY_NUMBERS = (3.0, 7.0, 10.0, 14.0, 17.0)
BOOTSTRAP_SAMPLES = 2000
BOOTSTRAP_SEED = 20260923
EPS = 1e-6


def _game_anchors(quotes: pd.DataFrame, population: pd.DataFrame) -> pd.DataFrame:
    ids = set(population.game_id.astype(str))
    q = quotes.loc[quotes.nflverse_game_id.astype(str).isin(ids)]
    commence = pd.to_datetime(
        q.groupby("nflverse_game_id").commence_time_utc.min(), utc=True
    )
    games = population[["game_id", "season", "week"]].drop_duplicates("game_id").copy()
    games["game_id"] = games["game_id"].astype(str)
    games["commence_time_utc"] = games.game_id.map(commence)
    games = games.loc[games.commence_time_utc.notna()].copy()
    games["week_first_commence_utc"] = games.groupby(
        ["season", "week"]
    ).commence_time_utc.transform("min")
    anchor = pd.to_datetime(games.week_first_commence_utc, utc=True).dt.tz_convert(
        "America/New_York"
    )
    local = anchor.dt.tz_localize(None).dt.normalize()
    sunday = local + pd.to_timedelta((6 - anchor.dt.weekday) % 7, unit="D")

    def _to_utc(naive: pd.Series, hours: float = 0.0) -> pd.Series:
        return (naive + pd.Timedelta(hours=hours)).dt.tz_localize(
            "America/New_York"
        ).dt.tz_convert("UTC")

    games["_monday"] = _to_utc(sunday - pd.Timedelta(days=6))
    games["_tuesday_noon"] = _to_utc(sunday - pd.Timedelta(days=5), hours=12.0)
    games["_wednesday"] = _to_utc(sunday - pd.Timedelta(days=4))
    games["_thursday"] = _to_utc(sunday - pd.Timedelta(days=3))
    deadline = _to_utc(sunday, hours=16.0)
    games["cutoff_utc"] = pd.concat([games.commence_time_utc, deadline], axis=1).min(axis=1)
    return games


def _eligible_quotes(quotes: pd.DataFrame, games: pd.DataFrame) -> pd.DataFrame:
    needed = {
        "nflverse_game_id",
        "bookmaker_key",
        "market",
        "home_spread_line",
        "observed_at_utc",
        "bookmaker_last_update_utc",
    }
    q = quotes.loc[
        quotes.market.eq("spreads") & quotes.bookmaker_key.isin(ALL_BOOKS), sorted(needed)
    ].rename(columns={"nflverse_game_id": "game_id"})
    q["game_id"] = q["game_id"].astype(str)
    q = q.merge(
        games[["game_id", "cutoff_utc", "_monday"]], on="game_id", how="inner"
    )
    for column in ("observed_at_utc", "bookmaker_last_update_utc", "cutoff_utc", "_monday"):
        q[column] = pd.to_datetime(q[column], utc=True, errors="coerce")
    q["home_spread_line"] = pd.to_numeric(q.home_spread_line, errors="coerce")
    q = q.loc[
        q.observed_at_utc.lt(q.cutoff_utc)
        & q.observed_at_utc.ge(q._monday)
        & q.bookmaker_last_update_utc.le(q.observed_at_utc)
        & np.isfinite(q.home_spread_line)
    ].copy()
    keys = ["game_id", "bookmaker_key", "observed_at_utc"]
    q = q.sort_values(keys).drop_duplicates(keys, keep="last")
    return q.sort_values("observed_at_utc")


def _asof_lines(eligible: pd.DataFrame, games: pd.DataFrame) -> pd.DataFrame:
    boundaries = games.melt(
        id_vars=["game_id"],
        value_vars=["_tuesday_noon", "_wednesday", "_thursday", "cutoff_utc"],
        var_name="boundary",
        value_name="query_time",
    )
    left = boundaries.merge(
        pd.DataFrame({"bookmaker_key": ALL_BOOKS}), how="cross"
    )
    left["query_time"] = pd.to_datetime(left["query_time"], utc=True)
    left = left.sort_values("query_time")
    right = eligible.sort_values("observed_at_utc")
    merged = pd.merge_asof(
        left,
        right[["game_id", "bookmaker_key", "observed_at_utc", "home_spread_line"]],
        left_on="query_time",
        right_on="observed_at_utc",
        by=["game_id", "bookmaker_key"],
        direction="backward",
    )
    return merged.pivot_table(
        index=["game_id", "bookmaker_key"],
        columns="boundary",
        values="home_spread_line",
        aggfunc="last",
    ).reset_index()


def _median_points(lines: pd.DataFrame, books: tuple[str, ...], start: str, end: str) -> pd.Series:
    subset = lines.loc[lines.bookmaker_key.isin(books)].copy()
    subset["move"] = subset[end] - subset[start]
    subset = subset.loc[subset.move.notna()]
    return subset.groupby("game_id").move.median()


def _crossing(start: pd.Series, end: pd.Series, keys: tuple[float, ...]) -> pd.Series:
    lo = np.minimum(start, end)
    hi = np.maximum(start, end)
    sign = np.sign(end - start)
    count = np.zeros(len(start), dtype=float)
    for k in keys:
        for kk in (k, -k):
            count += ((lo < kk) & (kk < hi)).astype(float)
    return pd.Series(count * sign, index=start.index)


def _median_crossing(
    lines: pd.DataFrame, books: tuple[str, ...], start: str, end: str, keys: tuple[float, ...], bucket: bool
) -> pd.Series:
    subset = lines.loc[lines.bookmaker_key.isin(books)].copy()
    subset = subset.loc[subset[start].notna() & subset[end].notna()]
    crossing = _crossing(subset[start], subset[end], keys)
    if bucket:
        crossing = np.sign(crossing) * crossing.abs().clip(upper=1.0)
    subset["value"] = crossing
    return subset.groupby("game_id").value.median()


def _week_blocked_bootstrap(df: pd.DataFrame, value_fn: Any, samples: int, seed: int) -> dict[str, float]:
    blocks = list(df.groupby(["season", "week"]).groups.keys())
    rng = np.random.default_rng(seed)
    point = value_fn(df)
    draws = np.empty(samples, dtype=float)
    grouped = {k: v for k, v in df.groupby(["season", "week"])}
    n_blocks = len(blocks)
    for i in range(samples):
        picks = rng.integers(0, n_blocks, size=n_blocks)
        sample = pd.concat([grouped[blocks[p]] for p in picks], ignore_index=True)
        draws[i] = value_fn(sample)
    lower, upper = np.quantile(draws, [0.025, 0.975])
    return {
        "estimate": float(point),
        "lower": float(lower),
        "upper": float(upper),
        "probability_positive": float(np.mean(draws > 0.0)),
    }


def _loso_predict(frame: pd.DataFrame, features: tuple[str, ...]) -> np.ndarray:
    previous = ppf.FIT_FEATURES
    ppf.FIT_FEATURES = features
    try:
        oos = pd.Series(index=frame.index, dtype=float)
        for held in OUTER_SEASONS:
            train = frame.loc[frame.season.ne(held)]
            test = frame.loc[frame.season.eq(held)]
            if train.empty or test.empty:
                continue
            means, stds = _standardisers(train)
            beta = _fit_logit(
                _design(train, means, stds), train.home_covered.to_numpy(dtype=float), FIT_RIDGE
            )
            oos.loc[test.index] = _predict(test, beta, means, stds)
        return oos.to_numpy(dtype=float)
    finally:
        ppf.FIT_FEATURES = previous


def _fold_coefficients(frame: pd.DataFrame, features: tuple[str, ...]) -> list[dict[str, Any]]:
    previous = ppf.FIT_FEATURES
    ppf.FIT_FEATURES = features
    rows: list[dict[str, Any]] = []
    try:
        for held in OUTER_SEASONS:
            train = frame.loc[frame.season.ne(held)]
            if train.empty:
                continue
            means, stds = _standardisers(train)
            beta = _fit_logit(
                _design(train, means, stds), train.home_covered.to_numpy(dtype=float), FIT_RIDGE
            )
            natural = _natural_coefficients(beta, means, stds)
            rows.append({"held_out_season": held, "coefficients": natural})
    finally:
        ppf.FIT_FEATURES = previous
    return rows


def _score_arm(
    name: str, population: pd.DataFrame, features: tuple[str, ...], served_p: np.ndarray
) -> dict[str, Any]:
    p = _loso_predict(population, features)
    valid = ~np.isnan(p) & ~np.isnan(served_p)
    frame = population.loc[valid].copy()
    p = p[valid]
    sp = served_p[valid]
    y = frame.home_covered.to_numpy(dtype=float)
    pc = np.clip(p, EPS, 1 - EPS)
    spc = np.clip(sp, EPS, 1 - EPS)
    frame["arm_p"] = p
    frame["served_p"] = sp
    frame["arm_correct"] = (p >= 0.5).astype(float) == y
    frame["served_correct"] = (sp >= 0.5).astype(float) == y
    frame["arm_brier"] = (pc - y) ** 2
    frame["served_brier"] = (spc - y) ** 2
    frame["arm_logloss"] = -(y * np.log(pc) + (1 - y) * np.log(1 - pc))
    frame["served_logloss"] = -(y * np.log(spc) + (1 - y) * np.log(1 - spc))
    decisive = frame.loc[(frame.arm_p >= 0.5) != (frame.served_p >= 0.5)]
    accuracy_boot = _week_blocked_bootstrap(
        frame,
        lambda d: float((d.arm_correct.astype(float) - d.served_correct.astype(float)).mean()) * 100.0,
        BOOTSTRAP_SAMPLES,
        BOOTSTRAP_SEED,
    )
    brier_boot = _week_blocked_bootstrap(
        frame,
        lambda d: float((d.served_brier - d.arm_brier).mean()),
        BOOTSTRAP_SAMPLES,
        BOOTSTRAP_SEED,
    )
    logloss_boot = _week_blocked_bootstrap(
        frame,
        lambda d: float((d.served_logloss - d.arm_logloss).mean()),
        BOOTSTRAP_SAMPLES,
        BOOTSTRAP_SEED,
    )
    return {
        "name": name,
        "features": list(features),
        "n_games": int(len(frame)),
        "arm_accuracy": float(frame.arm_correct.mean()),
        "served_accuracy": float(frame.served_correct.mean()),
        "arm_record": f"{int(frame.arm_correct.sum())}-{int((~frame.arm_correct).sum())}",
        "served_record": f"{int(frame.served_correct.sum())}-{int((~frame.served_correct).sum())}",
        "arm_brier": float(frame.arm_brier.mean()),
        "served_brier": float(frame.served_brier.mean()),
        "arm_logloss": float(frame.arm_logloss.mean()),
        "served_logloss": float(frame.served_logloss.mean()),
        "n_decisive": int(len(decisive)),
        "arm_decisive_record": f"{int(decisive.arm_correct.sum())}-{int((~decisive.arm_correct).sum())}",
        "served_decisive_record": f"{int(decisive.served_correct.sum())}-{int((~decisive.served_correct).sum())}",
        "accuracy_points_bootstrap": accuracy_boot,
        "brier_improvement_bootstrap": brier_boot,
        "logloss_improvement_bootstrap": logloss_boot,
        "fold_coefficients": _fold_coefficients(population, features),
    }


def main() -> None:
    population, provenance = build_fit_population(ARTIFACTS_ROOT, DATA_ROOT)
    population = population.copy()
    population["game_id"] = population["game_id"].astype(str)

    quotes = pd.read_parquet(QUOTES_CACHE)
    games = _game_anchors(quotes, population)
    eligible = _eligible_quotes(quotes, games)
    lines = _asof_lines(eligible, games)

    served_ref = pd.read_parquet(SERVED_PER_GAME, columns=["game_id", MARKET_MOVE_COLUMN])
    served_ref["game_id"] = served_ref["game_id"].astype(str)
    served_leader = _median_points(lines, LEADER_BOOKS, "_wednesday", "cutoff_utc").rename(
        "recomputed_served"
    )
    parity = served_ref.merge(served_leader, on="game_id", how="inner")
    parity_max_abs_diff = float(
        (parity[MARKET_MOVE_COLUMN] - parity["recomputed_served"]).abs().max()
    )
    parity_n = int(len(parity))

    move_all_books = _median_points(lines, ALL_BOOKS, "_wednesday", "cutoff_utc")
    move_early = _median_points(lines, LEADER_BOOKS, "_tuesday_noon", "_thursday")
    move_late = _median_points(lines, LEADER_BOOKS, "_thursday", "cutoff_utc")
    move_keynum = _median_crossing(
        lines, LEADER_BOOKS, "_wednesday", "cutoff_utc", KEY_NUMBERS, bucket=False
    )
    move_across_3 = _median_crossing(
        lines, LEADER_BOOKS, "_wednesday", "cutoff_utc", (3.0,), bucket=True
    )
    move_across_7 = _median_crossing(
        lines, LEADER_BOOKS, "_wednesday", "cutoff_utc", (7.0, 10.0, 14.0, 17.0), bucket=True
    )

    def _with_move(series: pd.Series) -> pd.DataFrame:
        frame = population.copy()
        mapped = frame.game_id.map(series)
        frame[MOVE_AVAILABLE_COLUMN] = mapped.notna().astype(float)
        frame[MOVE_COLUMN] = mapped.fillna(0.0)
        return frame

    served_pop = population
    served_pop[MOVE_AVAILABLE_COLUMN] = pd.to_numeric(
        served_pop[MOVE_AVAILABLE_COLUMN], errors="coerce"
    ).fillna(0.0)
    served_pop[MOVE_COLUMN] = pd.to_numeric(served_pop[MOVE_COLUMN], errors="coerce").fillna(0.0)
    served_p = _loso_predict(
        served_pop, ("model_logit", FLAG_SUM_COLUMN, MOVE_COLUMN, MOVE_AVAILABLE_COLUMN)
    )

    arms: list[dict[str, Any]] = []
    arms.append(
        _score_arm(
            "a_all_books_median",
            _with_move(move_all_books),
            ("model_logit", FLAG_SUM_COLUMN, MOVE_COLUMN, MOVE_AVAILABLE_COLUMN),
            served_p,
        )
    )
    arms.append(
        _score_arm(
            "b_early_window_tue_noon_thu",
            _with_move(move_early),
            ("model_logit", FLAG_SUM_COLUMN, MOVE_COLUMN, MOVE_AVAILABLE_COLUMN),
            served_p,
        )
    )
    arms.append(
        _score_arm(
            "b_late_window_thu_cutoff",
            _with_move(move_late),
            ("model_logit", FLAG_SUM_COLUMN, MOVE_COLUMN, MOVE_AVAILABLE_COLUMN),
            served_p,
        )
    )
    arms.append(
        _score_arm(
            "c_key_number_crossing_count",
            _with_move(move_keynum),
            ("model_logit", FLAG_SUM_COLUMN, MOVE_COLUMN, MOVE_AVAILABLE_COLUMN),
            served_p,
        )
    )
    d_frame = population.copy()
    d_frame["move_across_3"] = d_frame.game_id.map(move_across_3).fillna(0.0)
    d_frame["move_across_7"] = d_frame.game_id.map(move_across_7).fillna(0.0)
    d_frame[MOVE_AVAILABLE_COLUMN] = (
        d_frame.game_id.map(move_across_3).notna() | d_frame.game_id.map(move_across_7).notna()
    ).astype(float)
    arms.append(
        _score_arm(
            "d_move_across_3_and_7_separate",
            d_frame,
            ("model_logit", FLAG_SUM_COLUMN, "move_across_3", "move_across_7", MOVE_AVAILABLE_COLUMN),
            served_p,
        )
    )

    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    out_dir = ARTIFACTS_ROOT / "market_move_decomposition" / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    lines.to_parquet(out_dir / "asof_lines.parquet")
    payload = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "served_move_source": {
            "table_function": "nfl_ats.pick_probability_fit._market_move_table",
            "artifact_root": "sharp_weighted_follow",
            "move_column": MARKET_MOVE_COLUMN,
            "leader_books": list(LEADER_BOOKS),
            "window": "_wednesday..cutoff_utc (kickoff or Sunday 16:00 ET, whichever earlier)",
        },
        "parity_check": {
            "n_games_compared": parity_n,
            "max_abs_diff_points": parity_max_abs_diff,
        },
        "fit_population_provenance": {k: str(v) for k, v in provenance.items()},
        "outer_seasons": OUTER_SEASONS,
        "n_fit_population": int(len(population)),
        "arms": arms,
    }
    (out_dir / "metadata.json").write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    print(json.dumps({"out_dir": str(out_dir)}, indent=2))
    print(
        json.dumps(
            {
                "parity_check": payload["parity_check"],
                "arms_summary": [
                    {
                        "name": arm["name"],
                        "n_games": arm["n_games"],
                        "arm_accuracy": arm["arm_accuracy"],
                        "served_accuracy": arm["served_accuracy"],
                        "arm_brier": arm["arm_brier"],
                        "served_brier": arm["served_brier"],
                        "arm_logloss": arm["arm_logloss"],
                        "served_logloss": arm["served_logloss"],
                        "n_decisive": arm["n_decisive"],
                        "arm_decisive_record": arm["arm_decisive_record"],
                        "served_decisive_record": arm["served_decisive_record"],
                        "accuracy_points_bootstrap": arm["accuracy_points_bootstrap"],
                        "brier_improvement_bootstrap": arm["brier_improvement_bootstrap"],
                        "logloss_improvement_bootstrap": arm["logloss_improvement_bootstrap"],
                    }
                    for arm in arms
                ],
            },
            indent=2,
            default=str,
        )
    )


if __name__ == "__main__":
    main()
