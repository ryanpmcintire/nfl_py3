from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from nfl_ats.clv import load_snapshot_manifest_index  # noqa: E402
from nfl_ats.io import atomic_csv, atomic_parquet, run_id  # noqa: E402
from nfl_ats.overlay_composition import (  # noqa: E402
    DEFAULT_FEATURES,
    DEFAULT_INCIDENTS,
    blocked_bootstrap_matrix,
    load_inputs,
)
from nfl_ats.provenance import sha256_file, write_stamped_artifact  # noqa: E402
from nfl_ats.sharp_book_movement_features import (  # noqa: E402
    LEADER_BOOKS,
    LEADERSHIP_WEIGHTS,
    late_week_follow_frame,
)
from nfl_ats.unserved_tilt_marginals import served_card_flip_set  # noqa: E402

DATA = REPO / "data"
OUTPUT = REPO / "artifacts/sharp_book_weighted_movement"
OPENER_ARCHIVE = REPO / "artifacts/opener_evaluation/20260910T211255Z/per_game.parquet"
SAMPLES = 20_000
SEED = 20260821
SCORED_SEASONS = (2023, 2024, 2025)
LEADERSHIP_FLOOR_SEASON = 2020
MIN_PARTICIPATIONS = 200
BOOKS = tuple(sorted(LEADERSHIP_WEIGHTS))
FLAT_FULL = 1.0
FLAT_HALF = 0.5
ARM_ORDER = ("b0", "b1", "b1f", "w1s", "w2s", "w1f", "w2f", "w1h", "w2h", "pc")
SURFACES = ("card", "raw")


def build_quote_cache(destination: Path) -> Path:
    columns = [
        "nflverse_game_id",
        "bookmaker_key",
        "observed_at_utc",
        "bookmaker_last_update_utc",
        "home_spread_line",
        "commence_time_utc",
        "market",
        "sport_key",
    ]
    index = load_snapshot_manifest_index(DATA / "market/raw")
    index = index.loc[index.capture_kind.eq("historical_backfill")]
    frames: list[pd.DataFrame] = []
    for row in index.itertuples(index=False):
        path = Path(str(row.dir)) / "quotes.parquet"
        if not path.is_file():
            continue
        quotes = pd.read_parquet(path, columns=columns)
        quotes = quotes.loc[
            quotes.sport_key.eq("americanfootball_nfl")
            & quotes.market.eq("spreads")
            & quotes.bookmaker_key.isin(LEADERSHIP_WEIGHTS)
            & quotes.nflverse_game_id.notna()
        ]
        if quotes.empty:
            continue
        quotes = quotes.drop(columns=["sport_key"])
        quotes["snapshot_timestamp_utc"] = row.snapshot_timestamp_utc
        quotes["decision_label"] = row.decision_label
        frames.append(quotes)
    cache = pd.concat(frames, ignore_index=True)
    cache["nflverse_game_id"] = cache.nflverse_game_id.astype(str)
    for column in ("observed_at_utc", "bookmaker_last_update_utc", "commence_time_utc"):
        cache[column] = pd.to_datetime(cache[column], utc=True)
    cache["home_spread_line"] = pd.to_numeric(cache.home_spread_line, errors="coerce")
    cache["archive_season"] = cache.nflverse_game_id.str[:4].astype(int)
    destination.parent.mkdir(parents=True, exist_ok=True)
    cache.to_parquet(destination, index=False)
    return destination


def leadership_counts(quotes: pd.DataFrame) -> pd.DataFrame:
    usable = quotes.loc[
        quotes.bookmaker_last_update_utc.le(quotes.observed_at_utc)
        & quotes.observed_at_utc.lt(quotes.commence_time_utc)
        & np.isfinite(quotes.home_spread_line)
    ]
    usable = usable.sort_values(
        ["nflverse_game_id", "observed_at_utc", "bookmaker_key"]
    ).drop_duplicates(["nflverse_game_id", "bookmaker_key", "observed_at_utc"])
    leads = dict.fromkeys(BOOKS, 0)
    participations = dict.fromkeys(BOOKS, 0)
    for _game_id, group in usable.groupby("nflverse_game_id", sort=False):
        wide = group.pivot(
            index="observed_at_utc", columns="bookmaker_key", values="home_spread_line"
        ).sort_index()
        if len(wide) < 2 or wide.shape[1] < 3:
            continue
        matrix = wide.ffill().to_numpy(dtype=float)
        names = list(wide.columns)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            for position, book in enumerate(names):
                others = np.delete(matrix, position, axis=1)
                consensus = np.nanmedian(others, axis=1)
                own = matrix[:, position]
                previous_consensus = consensus[:-1]
                current_consensus = consensus[1:]
                previous_own = own[:-1]
                current_own = own[1:]
                valid = (
                    np.isfinite(previous_consensus)
                    & np.isfinite(current_consensus)
                    & np.isfinite(previous_own)
                    & np.isfinite(current_own)
                    & (current_consensus != previous_consensus)
                )
                if not valid.any():
                    continue
                direction = np.sign(current_consensus - previous_consensus)
                early = valid & (direction * (previous_own - previous_consensus) > 0)
                participations[book] += int(valid.sum())
                leads[book] += int(early.sum())
    return pd.DataFrame(
        {
            "bookmaker_key": list(BOOKS),
            "leads": [leads[book] for book in BOOKS],
            "participations": [participations[book] for book in BOOKS],
        }
    )


def scores_from_counts(counts: pd.DataFrame) -> pd.DataFrame:
    frame = counts.copy()
    frame["raw_score"] = np.where(
        frame.participations.gt(0), frame.leads / frame.participations.replace(0, np.nan), np.nan
    )
    estimated = frame.participations.ge(MIN_PARTICIPATIONS)
    if estimated.any():
        prior = float(
            frame.loc[estimated, "leads"].sum() / frame.loc[estimated, "participations"].sum()
        )
    else:
        prior = float(np.nanmean(frame.raw_score)) if frame.raw_score.notna().any() else 0.5
    frame["estimated"] = estimated
    frame["score"] = np.where(estimated, frame.raw_score, prior)
    frame["neutral_prior"] = prior
    return frame


def per_book_net_moves(quotes: pd.DataFrame, games: pd.DataFrame) -> pd.DataFrame:
    result = games.copy()
    kickoff = pd.to_datetime(result.commence_time_utc, utc=True)
    anchor = pd.to_datetime(result.week_first_commence_utc, utc=True).dt.tz_convert(
        "America/New_York"
    )
    local = anchor.dt.tz_localize(None).dt.normalize()
    sunday = local + pd.to_timedelta((6 - anchor.dt.weekday) % 7, unit="D")
    deadline = (
        (sunday + pd.Timedelta(hours=16)).dt.tz_localize("America/New_York").dt.tz_convert("UTC")
    )
    cutoffs = [kickoff, deadline]
    if "cutoff_utc" in result:
        cutoffs.append(pd.to_datetime(result.cutoff_utc, utc=True))
    result["cutoff_utc"] = pd.concat(cutoffs, axis=1).min(axis=1)
    result["_monday"] = (
        (sunday - pd.Timedelta(days=6)).dt.tz_localize("America/New_York").dt.tz_convert("UTC")
    )
    result["_wednesday"] = (
        (sunday - pd.Timedelta(days=4)).dt.tz_localize("America/New_York").dt.tz_convert("UTC")
    )
    result["_sunday"] = sunday.dt.tz_localize("America/New_York").dt.tz_convert("UTC")
    frame = quotes.loc[
        quotes.market.eq("spreads") & quotes.bookmaker_key.isin(LEADERSHIP_WEIGHTS),
        [
            "nflverse_game_id",
            "bookmaker_key",
            "observed_at_utc",
            "bookmaker_last_update_utc",
            "home_spread_line",
        ],
    ].rename(columns={"nflverse_game_id": "game_id"})
    frame = frame.merge(
        result[["game_id", "cutoff_utc", "_monday", "_wednesday", "_sunday"]], on="game_id"
    )
    for column in ("observed_at_utc", "bookmaker_last_update_utc"):
        frame[column] = pd.to_datetime(frame[column], utc=True, errors="coerce")
    frame["home_spread_line"] = pd.to_numeric(frame.home_spread_line, errors="coerce")
    frame = frame.loc[
        frame.observed_at_utc.lt(frame.cutoff_utc)
        & frame.observed_at_utc.ge(frame._monday)
        & frame.observed_at_utc.lt(frame._sunday)
        & frame.bookmaker_last_update_utc.le(frame.observed_at_utc)
        & np.isfinite(frame.home_spread_line)
    ].copy()
    keys = ["game_id", "bookmaker_key", "observed_at_utc"]
    frame = frame.sort_values(keys).drop_duplicates(keys)
    frame["move"] = frame.groupby(["game_id", "bookmaker_key"]).home_spread_line.diff()
    frame = frame.loc[frame.observed_at_utc.ge(frame._wednesday) & frame.move.notna()].copy()
    return frame.groupby(["game_id", "bookmaker_key"], as_index=False).move.sum()


def weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    order = np.argsort(values, kind="stable")
    values = values[order]
    weights = weights[order]
    total = weights.sum()
    if total <= 0:
        return float("nan")
    cumulative = np.cumsum(weights)
    half = total / 2.0
    index = int(np.searchsorted(cumulative, half, side="left"))
    index = min(index, len(values) - 1)
    if np.isclose(cumulative[index], half) and index + 1 < len(values):
        return float((values[index] + values[index + 1]) / 2.0)
    return float(values[index])


def weighted_aggregates(
    books: pd.DataFrame, seasons: pd.Series, weights_by_season: dict[int, dict[str, float]]
) -> pd.DataFrame:
    season_of = seasons.to_dict()
    rows: list[dict[str, Any]] = []
    for game_id, group in books.groupby("game_id", sort=False):
        season = season_of.get(str(game_id))
        if season is None:
            continue
        table = weights_by_season[int(season)]
        values = group.move.to_numpy(dtype=float)
        weight = np.array([table[key] for key in group.bookmaker_key], dtype=float)
        rows.append(
            {
                "game_id": str(game_id),
                "weighted_mean_net": float((values * weight).sum() / weight.sum()),
                "weighted_median_net": weighted_median(values, weight),
                "weighted_books": len(values),
            }
        )
    return pd.DataFrame(rows)


def paired(delta: np.ndarray, blocks: pd.DataFrame, block: str) -> dict[str, float]:
    stats = blocked_bootstrap_matrix(
        delta[:, np.newaxis], blocks.reset_index(drop=True), block=block, samples=SAMPLES, seed=SEED
    )
    return {
        "estimate_accuracy_points": float(stats["estimate"][0] * 100.0),
        "lower_accuracy_points": float(stats["lower"][0] * 100.0),
        "upper_accuracy_points": float(stats["upper"][0] * 100.0),
        "probability_positive": float(stats["probability_positive"][0]),
        "standard_error_accuracy_points": float(stats["standard_error"][0] * 100.0),
        "blocks": int(stats["block_count"]),
    }


def cell(
    label: str, candidate: pd.Series, baseline: pd.Series, meta: pd.DataFrame
) -> dict[str, Any]:
    frame = pd.concat({"candidate": candidate, "baseline": baseline}, axis=1).join(
        meta, how="inner"
    )
    live = frame.loc[frame.candidate.notna() & frame.baseline.notna()]
    delta = (live.candidate - live.baseline).to_numpy(dtype=float)
    row: dict[str, Any] = {
        "label": label,
        "n": len(live),
        "baseline_accuracy": float(live.baseline.mean()),
        "candidate_accuracy": float(live.candidate.mean()),
        "delta_accuracy_points": float(delta.mean() * 100.0) if len(delta) else 0.0,
        "week_blocked": paired(delta, live[["season", "week"]], "week"),
        "season_blocked": paired(delta, live[["season", "week"]], "season"),
        "season_deltas": {
            str(int(season)): float((group.candidate - group.baseline).mean() * 100.0)
            for season, group in live.groupby("season")
        },
    }
    return row


def reliability(pairs: dict[str, pd.Series]) -> dict[str, Any]:
    first, second = pairs["first"], pairs["second"]
    joined = pd.concat({"first": first, "second": second}, axis=1).dropna()
    if len(joined) < 3:
        return {"books": len(joined), "pearson": None, "spearman": None}
    return {
        "books": len(joined),
        "pearson": float(joined.first.corr(joined.second)),
        "spearman": float(joined.first.corr(joined.second, method="spearman")),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "MKT-15: chronologically estimated per-book leadership scores, spent as weights on "
            "the late-week follow's net move, against the served three-book leader median."
        )
    )
    parser.add_argument(
        "--quote-cache",
        type=Path,
        default=OUTPUT / "spread_quotes.parquet",
        help="per-book NFL spread capture cache; built from data/market/raw when absent",
    )
    parser.add_argument("--opener-archive", type=Path, default=OPENER_ARCHIVE)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    output = args.output or (OUTPUT / run_id())
    if not args.quote_cache.is_file():
        build_quote_cache(args.quote_cache)
    cache = pd.read_parquet(args.quote_cache)
    cache["archive_season"] = cache.nflverse_game_id.str[:4].astype(int)
    cache["archive_week"] = cache.nflverse_game_id.str[5:7].astype(int)

    pools: dict[int, tuple[int, ...]] = {
        season: tuple(range(LEADERSHIP_FLOOR_SEASON, season)) for season in SCORED_SEASONS
    }
    leadership: dict[str, Any] = {}
    weights_by_season: dict[int, dict[str, float]] = {}
    for season, pool in pools.items():
        counts = leadership_counts(cache.loc[cache.archive_season.isin(pool)])
        table = scores_from_counts(counts)
        weights_by_season[season] = dict(zip(table.bookmaker_key, table.score, strict=True))
        leadership[str(season)] = {
            "pool_seasons": list(pool),
            "books": table.to_dict(orient="records"),
        }

    era_tables: dict[str, pd.DataFrame] = {}
    for name, pool in (
        ("coarse_2020_2022", (2020, 2021, 2022)),
        ("fine_2023_2025", SCORED_SEASONS),
    ):
        era_tables[name] = scores_from_counts(
            leadership_counts(cache.loc[cache.archive_season.isin(pool)])
        ).set_index("bookmaker_key")
    parity_tables: dict[str, pd.DataFrame] = {}
    for name, keep in (
        ("odd_weeks", cache.archive_week.mod(2).eq(1)),
        ("even_weeks", cache.archive_week.mod(2).eq(0)),
    ):
        parity_tables[name] = scores_from_counts(leadership_counts(cache.loc[keep])).set_index(
            "bookmaker_key"
        )
    season_tables = {
        season: scores_from_counts(
            leadership_counts(cache.loc[cache.archive_season.eq(season)])
        ).set_index("bookmaker_key")
        for season in range(LEADERSHIP_FLOOR_SEASON, 2026)
    }

    def estimated_scores(table: pd.DataFrame) -> pd.Series:
        return table.raw_score.where(table.participations.ge(MIN_PARTICIPATIONS))

    reliabilities = {
        "odd_even_weeks_all_seasons": reliability(
            {
                "first": estimated_scores(parity_tables["odd_weeks"]),
                "second": estimated_scores(parity_tables["even_weeks"]),
            }
        ),
        "coarse_era_vs_fine_era": reliability(
            {
                "first": estimated_scores(era_tables["coarse_2020_2022"]),
                "second": estimated_scores(era_tables["fine_2023_2025"]),
            }
        ),
    }
    season_pairs: list[dict[str, Any]] = []
    seasons_available = sorted(season_tables)
    for i, left in enumerate(seasons_available):
        for right in seasons_available[i + 1 :]:
            value = reliability(
                {
                    "first": estimated_scores(season_tables[left]),
                    "second": estimated_scores(season_tables[right]),
                }
            )
            season_pairs.append({"left": left, "right": right, **value})
    pearsons = [row["pearson"] for row in season_pairs if row["pearson"] is not None]
    reliabilities["season_pairs"] = {
        "pairs": season_pairs,
        "mean_pearson": float(np.mean(pearsons)) if pearsons else None,
        "min_pearson": float(np.min(pearsons)) if pearsons else None,
        "max_pearson": float(np.max(pearsons)) if pearsons else None,
    }

    incumbent, schedules, _player, snapshot_name, _path = load_inputs(args.opener_archive, DATA)
    ids = incumbent.game_id.astype(str)
    seasons_series = pd.to_numeric(incumbent.season, errors="coerce")
    in_window = set(ids[seasons_series.isin(list(SCORED_SEASONS)).to_numpy()])
    served_flips, members = served_card_flip_set(
        incumbent,
        data_root=DATA,
        repo_root=REPO,
        features=REPO / DEFAULT_FEATURES,
        incidents=REPO / DEFAULT_INCIDENTS,
        schedules=schedules,
        card="served",
    )
    raw_home = pd.to_numeric(incumbent.home_cover_probability_at_open, errors="coerce").ge(0.5)
    raw_home.index = ids
    card_home = raw_home.ne(pd.Series(ids.isin(served_flips).to_numpy(), index=ids))
    margin = pd.to_numeric(incumbent.margin_vs_open, errors="coerce")
    margin.index = ids
    opener_line = pd.to_numeric(incumbent.tue_open_home_spread, errors="coerce")
    opener_line.index = ids
    meta = incumbent.set_index(ids)[["season", "week"]]

    replay = cache.loc[
        cache.decision_label.eq("intraday_hourly")
        & cache.archive_season.isin(list(SCORED_SEASONS))
        & cache.nflverse_game_id.isin(in_window)
    ].copy()
    schedule_weeks = schedules[["game_id", "season", "week"]].drop_duplicates("game_id")
    commence = pd.to_datetime(replay.groupby("nflverse_game_id").commence_time_utc.min(), utc=True)
    games = pd.DataFrame(
        {"game_id": commence.index.astype(str), "commence_time_utc": commence.to_numpy()}
    ).merge(schedule_weeks, on="game_id", how="inner")
    games["week_first_commence_utc"] = games.groupby(
        ["season", "week"]
    ).commence_time_utc.transform("min")
    now = pd.Timestamp.now(tz="UTC")
    games["cutoff_utc"] = now
    games["decision_home_spread"] = games.game_id.map(opener_line)

    sides = card_home.loc[games.game_id].map({True: "HOME", False: "AWAY"})
    sides.index = games.game_id.to_numpy()
    exposure, refused = late_week_follow_frame(replay, games, now=now, tuesday_pick_side=sides)
    exposure = exposure.set_index(exposure.game_id.astype(str))

    books = per_book_net_moves(replay, games.copy())
    reconstructed = books.groupby("game_id").move.agg(["mean", "size"])
    leader_side = books.loc[books.bookmaker_key.isin(LEADER_BOOKS)]
    reconstructed_leader = leader_side.groupby("game_id").move.agg(["median", "size"])
    parity = {
        "equal_net_max_abs_diff": float(
            (
                exposure.equal_net_move
                - exposure.index.map(reconstructed["mean"])
                .to_series(index=exposure.index)
                .fillna(0.0)
            )
            .abs()
            .max()
        ),
        "equal_books_mismatches": int(
            (
                exposure.eligible_books.astype(int)
                != exposure.index.map(reconstructed["size"])
                .to_series(index=exposure.index)
                .fillna(0)
                .astype(int)
            ).sum()
        ),
        "leader_median_max_abs_diff": float(
            (
                exposure.leader_median_net_move
                - exposure.index.map(reconstructed_leader["median"])
                .to_series(index=exposure.index)
                .fillna(0.0)
            )
            .abs()
            .max()
        ),
        "leader_books_mismatches": int(
            (
                exposure.leader_books.astype(int)
                != exposure.index.map(reconstructed_leader["size"])
                .to_series(index=exposure.index)
                .fillna(0)
                .astype(int)
            ).sum()
        ),
        "refused_quote_rows": int(refused),
    }
    if (
        parity["equal_net_max_abs_diff"] > 1e-9
        or parity["equal_books_mismatches"]
        or parity["leader_median_max_abs_diff"] > 1e-9
        or parity["leader_books_mismatches"]
    ):
        raise ValueError(
            f"Per-book reconstruction does not reproduce the served aggregate: {parity}"
        )

    index = pd.Index(sorted(set(exposure.index) & in_window), name="game_id")
    scoped_meta = meta.reindex(index)
    weighted = weighted_aggregates(
        books.loc[books.game_id.isin(set(index))], scoped_meta.season, weights_by_season
    ).set_index("game_id")

    gate = pd.to_numeric(exposure.late_week_threshold_applied, errors="coerce").reindex(index)
    leader_net = pd.to_numeric(exposure.leader_median_net_move, errors="coerce").reindex(index)
    leader_books_count = (
        pd.to_numeric(exposure.leader_books, errors="coerce").reindex(index).fillna(0)
    )
    weighted_mean_net = weighted.weighted_mean_net.reindex(index)
    weighted_median_net = weighted.weighted_median_net.reindex(index)
    weighted_books = weighted.weighted_books.reindex(index).fillna(0)

    leader_reach = leader_books_count.gt(0)
    weighted_reach = weighted_books.gt(0)
    margin_scoped = margin.reindex(index)
    scoreable = margin_scoped.notna() & margin_scoped.ne(0.0)
    covered_home = margin_scoped.gt(0.0)

    fires = {
        "b1": leader_net.abs().ge(gate).fillna(False) & leader_reach,
        "b1f": leader_net.abs().ge(FLAT_FULL).fillna(False) & leader_reach,
        "w1s": weighted_mean_net.abs().ge(gate).fillna(False) & weighted_reach,
        "w2s": weighted_median_net.abs().ge(gate).fillna(False) & weighted_reach,
        "w1f": weighted_mean_net.abs().ge(FLAT_FULL).fillna(False) & weighted_reach,
        "w2f": weighted_median_net.abs().ge(FLAT_FULL).fillna(False) & weighted_reach,
        "w1h": weighted_mean_net.abs().ge(FLAT_HALF).fillna(False) & weighted_reach,
        "w2h": weighted_median_net.abs().ge(FLAT_HALF).fillna(False) & weighted_reach,
    }
    market_home = {
        "b1": leader_net.gt(0),
        "b1f": leader_net.gt(0),
        "w1s": weighted_mean_net.gt(0),
        "w2s": weighted_median_net.gt(0),
        "w1f": weighted_mean_net.gt(0),
        "w2f": weighted_median_net.gt(0),
        "w1h": weighted_mean_net.gt(0),
        "w2h": weighted_median_net.gt(0),
    }

    surfaces = {"card": card_home.reindex(index), "raw": raw_home.reindex(index)}
    picks: dict[str, dict[str, pd.Series]] = {}
    correct: dict[str, dict[str, pd.Series]] = {}
    for surface, base in surfaces.items():
        arms = {"b0": base}
        for name in fires:
            arms[name] = base.mask(fires[name], market_home[name])
        arms["pc"] = base.mask(weighted_reach | leader_reach, covered_home)
        picks[surface] = arms
        correct[surface] = {
            name: side.eq(covered_home).astype(float).where(scoreable)
            for name, side in arms.items()
        }

    counts: dict[str, Any] = {
        "archive_games_in_window": len(in_window),
        "games_with_replayable_exposure": len(exposure),
        "games_in_scope": len(index),
        "games_scored": int(scoreable.sum()),
        "opener_pushes": int((~scoreable).sum()),
        "week_blocks": int(scoped_meta.loc[scoreable].groupby(["season", "week"]).ngroups),
        "games_with_leader_book_evidence": int(leader_reach.sum()),
        "games_with_any_book_evidence": int(weighted_reach.sum()),
        "nine_member_flips_in_window": int(pd.Series(index.isin(served_flips), index=index).sum()),
        "served_card_members": {name: len(ids_) for name, ids_ in members.items()},
        "fires": {name: int(flag.sum()) for name, flag in fires.items()},
        "refused_quote_rows": int(refused),
    }
    for surface in surfaces:
        counts[f"picks_changed_{surface}"] = {
            name: {
                "vs_b0": int(picks[surface][name].ne(picks[surface]["b0"]).sum()),
                "vs_b1": int(picks[surface][name].ne(picks[surface]["b1"]).sum()),
                "vs_b1_scored": int(
                    (picks[surface][name].ne(picks[surface]["b1"]) & scoreable).sum()
                ),
            }
            for name in ARM_ORDER
        }

    accuracies = {
        surface: {
            name: {
                "overall": float(series.dropna().mean()),
                **{
                    str(season): float(series.loc[scoped_meta.season.eq(season)].dropna().mean())
                    for season in SCORED_SEASONS
                },
            }
            for name, series in correct[surface].items()
        }
        for surface in surfaces
    }

    cells: list[dict[str, Any]] = []
    for surface in SURFACES:
        for name in ARM_ORDER:
            for baseline in ("b0", "b1"):
                if name == baseline:
                    continue
                cells.append(
                    {
                        "surface": surface,
                        "arm": name,
                        "baseline": baseline,
                        **cell(
                            f"{surface}_{name}_vs_{baseline}",
                            correct[surface][name],
                            correct[surface][baseline],
                            scoped_meta,
                        ),
                        "picks_changed": counts[f"picks_changed_{surface}"][name][f"vs_{baseline}"],
                    }
                )

    per_game = pd.DataFrame(
        {
            "game_id": index,
            "season": scoped_meta.season.to_numpy(),
            "week": scoped_meta.week.to_numpy(),
            "opener_home_spread": opener_line.reindex(index).to_numpy(),
            "gate": gate.to_numpy(),
            "leader_median_net": leader_net.to_numpy(),
            "leader_books": leader_books_count.to_numpy(),
            "weighted_mean_net": weighted_mean_net.to_numpy(),
            "weighted_median_net": weighted_median_net.to_numpy(),
            "weighted_books": weighted_books.to_numpy(),
            "margin_vs_open": margin_scoped.to_numpy(),
            "scoreable": scoreable.to_numpy(),
            "covered_home": covered_home.to_numpy(),
        }
    )
    for surface in SURFACES:
        for name in ARM_ORDER:
            per_game[f"{surface}_{name}_pick_home"] = picks[surface][name].to_numpy()
            per_game[f"{surface}_{name}_correct"] = correct[surface][name].to_numpy()

    output.mkdir(parents=True, exist_ok=True)
    atomic_parquet(per_game, output / "per_game.parquet")
    atomic_csv(
        pd.DataFrame(
            [
                {
                    "surface": row["surface"],
                    "arm": row["arm"],
                    "baseline": row["baseline"],
                    "n": row["n"],
                    "picks_changed": row["picks_changed"],
                    "candidate_accuracy": row["candidate_accuracy"],
                    "baseline_accuracy": row["baseline_accuracy"],
                    "delta_accuracy_points": row["delta_accuracy_points"],
                    "week_low": row["week_blocked"]["lower_accuracy_points"],
                    "week_high": row["week_blocked"]["upper_accuracy_points"],
                    "week_probability_positive": row["week_blocked"]["probability_positive"],
                    "season_low": row["season_blocked"]["lower_accuracy_points"],
                    "season_high": row["season_blocked"]["upper_accuracy_points"],
                    "season_probability_positive": row["season_blocked"]["probability_positive"],
                }
                for row in cells
            ]
        ),
        output / "cells.csv",
    )
    atomic_csv(
        pd.DataFrame(
            [
                {
                    "scored_season": season,
                    "pool": ",".join(str(s) for s in body["pool_seasons"]),
                    **book,
                }
                for season, body in leadership.items()
                for book in body["books"]
            ]
        ),
        output / "leadership.csv",
    )
    results = {
        "leadership": leadership,
        "reliability": reliabilities,
        "counts": counts,
        "accuracies": accuracies,
        "parity": parity,
        "cells": cells,
        "configuration": {
            "scored_seasons": list(SCORED_SEASONS),
            "leadership_floor_season": LEADERSHIP_FLOOR_SEASON,
            "min_participations": MIN_PARTICIPATIONS,
            "flat_full_threshold": FLAT_FULL,
            "flat_half_threshold": FLAT_HALF,
            "bootstrap_samples": SAMPLES,
            "bootstrap_seed": SEED,
            "opener_archive": str(args.opener_archive),
            "schedule_snapshot": snapshot_name,
            "quote_cache": str(args.quote_cache),
            "predeclaration_sha256": sha256_file(REPO / "docs/sharp_book_weighted_movement.md"),
            "script_sha256": sha256_file(Path(__file__)),
            "feature_module_sha256": sha256_file(
                REPO / "src/nfl_ats/sharp_book_movement_features.py"
            ),
        },
    }
    write_stamped_artifact(json.loads(json.dumps(results, default=str)), output / "results.json")
    print(
        json.dumps(
            {k: results[k] for k in ("counts", "parity", "reliability")}, indent=2, default=str
        ),
        flush=True,
    )
    print(json.dumps(leadership, indent=2, default=str), flush=True)
    print(json.dumps(accuracies, indent=2, default=str), flush=True)
    print(
        pd.read_csv(output / "cells.csv").to_string(index=False),
        flush=True,
    )
    print(f"artifacts: {output}", flush=True)


if __name__ == "__main__":
    main()
