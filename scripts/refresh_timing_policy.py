from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime, time
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

from nfl_ats.clv import load_snapshot_manifest_index  # noqa: E402
from nfl_ats.nfl_week import pool_decision_cutoff, week_cycle_sunday  # noqa: E402
from nfl_ats.overlay_composition import (  # noqa: E402
    DEFAULT_FEATURES,
    DEFAULT_INCIDENTS,
    blocked_bootstrap_matrix,
    load_inputs,
)
from nfl_ats.sharp_book_movement_features import (  # noqa: E402
    LEADER_BOOKS,
    LEADERSHIP_WEIGHTS,
    leader_follow_threshold,
    refresh_pick,
    sharp_book_movement_features,
)
from nfl_ats.unserved_tilt_marginals import served_card_flip_set  # noqa: E402

EASTERN = ZoneInfo("America/New_York")
HISTORICAL = "historical_backfill"
SAMPLES = 20_000
SEED = 20260911

SERVED_PASSES: tuple[tuple[str, int, time], ...] = (
    ("refresh_wed", -4, time(18, 15)),
    ("refresh_wed_inactives_primetime", -4, time(19, 15)),
    ("refresh_thu_inactives_early", -3, time(11, 55)),
    ("refresh_thu", -3, time(15, 0)),
    ("refresh_thu_inactives_late", -3, time(15, 25)),
    ("refresh_thu_inactives_primetime", -3, time(19, 15)),
    ("refresh_sat", -1, time(10, 30)),
    ("refresh_sat_inactives_early", -1, time(15, 50)),
    ("refresh_sat_inactives_late", -1, time(19, 15)),
    ("refresh_sun", 0, time(10, 0)),
    ("refresh_sun_inactives_early", 0, time(11, 55)),
    ("refresh_sun_inactives_late", 0, time(15, 0)),
)

TRIGGER_THRESHOLDS = (0.5, 1.0)
DECAY_LEAD_HOURS = (0.0, 3.0, 6.0, 12.0, 18.0, 24.0, 36.0, 48.0, 60.0, 72.0, 96.0, 120.0)
WINDOWS: tuple[tuple[str, tuple[int, ...]], ...] = (
    ("2020_2021", (2020, 2021)),
    ("2022_2023", (2022, 2023)),
    ("2024_2025", (2024, 2025)),
    ("2020_2025", (2020, 2021, 2022, 2023, 2024, 2025)),
)


def build_quote_cache(market_root: Path, cache_path: Path, seasons: tuple[int, ...]) -> None:
    columns = [
        "nflverse_game_id",
        "bookmaker_key",
        "market",
        "home_spread_line",
        "observed_at_utc",
        "bookmaker_last_update_utc",
    ]
    index = load_snapshot_manifest_index(market_root)
    index = index.loc[index["capture_kind"].eq(HISTORICAL) & index["week"].notna()]
    index = index.loc[index["season"].isin(set(seasons))]
    frames: list[pd.DataFrame] = []
    for row in index.itertuples(index=False):
        path = Path(str(row.dir)) / "quotes.parquet"
        if not path.is_file():
            continue
        quotes = pd.read_parquet(path, columns=columns)
        quotes = quotes.loc[
            quotes["market"].eq("spreads")
            & quotes["bookmaker_key"].isin(LEADERSHIP_WEIGHTS)
            & quotes["nflverse_game_id"].notna()
            & quotes["home_spread_line"].notna()
        ]
        if quotes.empty:
            continue
        quotes = quotes.assign(
            season=int(row.season), week=int(row.week), decision_label=str(row.decision_label)
        )
        frames.append(quotes)
    combined = pd.concat(frames, ignore_index=True)
    combined["nflverse_game_id"] = combined["nflverse_game_id"].astype(str)
    combined = combined.drop_duplicates(
        ["nflverse_game_id", "bookmaker_key", "observed_at_utc", "home_spread_line"]
    )
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(cache_path, index=False)


def capture_cadence(market_root: Path) -> list[dict[str, Any]]:
    index = load_snapshot_manifest_index(market_root)
    index = index.loc[index["capture_kind"].eq(HISTORICAL) & index["week"].notna()]
    rows: list[dict[str, Any]] = []
    for entry in index.itertuples(index=False):
        manifest = json.loads((Path(str(entry.dir)) / "manifest.json").read_text(encoding="utf-8"))
        request = manifest.get("request") or {}
        if str(request.get("sport")) != "americanfootball_nfl":
            continue
        rows.append(
            {
                "season": int(entry.season),
                "week": int(entry.week),
                "label": str(entry.decision_label),
                "observed": entry.snapshot_timestamp_utc,
            }
        )
    frame = pd.DataFrame(rows)
    frame["et"] = frame["observed"].dt.tz_convert(EASTERN)
    frame["dow"] = frame["et"].dt.day_name().str[:3]
    out: list[dict[str, Any]] = []
    for season, group in frame.groupby("season"):
        per_week = group.groupby("week")["observed"].nunique()
        gaps: list[float] = []
        for _week, block in group.groupby("week"):
            stamps = np.sort(block["observed"].unique())
            if len(stamps) > 1:
                gaps.extend(np.diff(stamps).astype("timedelta64[m]").astype(float))
        labels = sorted(group["label"].unique())
        weekday_counts = group.groupby("dow")["observed"].nunique().to_dict()
        sundays = group.loc[group["dow"].eq("Sun")].copy()
        sundays["clock"] = sundays["et"].dt.strftime("%H:%M")
        late = sundays.loc[sundays["clock"].between("11:00", "15:59")]
        before_lock = sundays.loc[sundays["clock"].lt("16:00"), "clock"]
        out.append(
            {
                "season": int(season),
                "weeks": int(group["week"].nunique()),
                "snapshots": int(group["observed"].nunique()),
                "median_snapshots_per_week": float(per_week.median()),
                "min_snapshots_per_week": int(per_week.min()),
                "max_snapshots_per_week": int(per_week.max()),
                "median_gap_minutes": float(np.median(gaps)) if gaps else float("nan"),
                "labels": labels,
                "snapshots_by_weekday": {
                    day: int(weekday_counts.get(day, 0))
                    for day in ("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun")
                },
                "sunday_captures_between_1100_and_1600_et": int(late["observed"].nunique()),
                "latest_sunday_capture_clock_before_1600_et": (
                    str(before_lock.max()) if not before_lock.empty else None
                ),
            }
        )
    return out


def build_games(per_game_path: Path, data_root: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    per_game, schedules, _players, snapshot_name, _path = load_inputs(per_game_path, data_root)
    flips, members = served_card_flip_set(
        per_game,
        data_root=data_root,
        repo_root=REPO,
        features=REPO / DEFAULT_FEATURES,
        incidents=REPO / DEFAULT_INCIDENTS,
        schedules=schedules,
        card="served",
    )
    kickoffs = (
        schedules.loc[:, ["game_id", "gameday", "gametime"]]
        .drop_duplicates("game_id")
        .set_index("game_id")
        .reindex(per_game["game_id"].astype(str))
    )
    ids = per_game["game_id"].astype(str)
    local = pd.Series(
        pd.to_datetime(
            kickoffs["gameday"].astype(str).to_numpy()
            + " "
            + kickoffs["gametime"].astype(str).to_numpy()
        )
    )
    kickoff = (
        local.dt.tz_localize(EASTERN, nonexistent="shift_forward", ambiguous=True)
        .dt.tz_convert(UTC)
        .to_numpy()
    )
    games = pd.DataFrame(
        {
            "game_id": ids.to_numpy(),
            "season": pd.to_numeric(per_game["season"], errors="coerce").astype(int).to_numpy(),
            "week": pd.to_numeric(per_game["week"], errors="coerce").astype(int).to_numpy(),
            "commence_time_utc": kickoff,
            "decision_home_spread": pd.to_numeric(
                per_game["tue_open_home_spread"], errors="coerce"
            ).to_numpy(),
            "margin_vs_open": pd.to_numeric(per_game["margin_vs_open"], errors="coerce").to_numpy(),
            "raw_pick_home": pd.to_numeric(
                per_game["home_cover_probability_at_open"], errors="coerce"
            )
            .ge(0.5)
            .to_numpy(),
            "composition_flip": ids.isin(flips).to_numpy(),
        }
    )
    if games["commence_time_utc"].isna().any():
        raise ValueError("every archive game needs a kickoff from the schedule snapshot")
    games["card_pick_home"] = games["raw_pick_home"] ^ games["composition_flip"]
    kickoff_series = pd.to_datetime(games["commence_time_utc"], utc=True)
    games["deadline_utc"] = [
        pool_decision_cutoff(value.to_pydatetime()) for value in kickoff_series
    ]
    games["deadline_utc"] = pd.to_datetime(games["deadline_utc"], utc=True)
    sunday_dates = [week_cycle_sunday(value.tz_convert(EASTERN).date()) for value in kickoff_series]
    games["week_sunday_utc"] = pd.to_datetime(
        [datetime.combine(day, time(0, 0), tzinfo=EASTERN).astimezone(UTC) for day in sunday_dates],
        utc=True,
    )
    games["week_first_commence_utc"] = games["week_sunday_utc"] + pd.Timedelta(hours=12)
    games["wednesday_utc"] = games["week_sunday_utc"] - pd.Timedelta(days=4)
    games["threshold"] = [leader_follow_threshold(value) for value in games["decision_home_spread"]]
    meta = {
        "schedule_snapshot": snapshot_name,
        "archive_games": len(games),
        "composition_flips": int(games["composition_flip"].sum()),
        "member_flip_counts": {name: len(members[name]) for name in members},
        "opener_push_games": int((games["margin_vs_open"] == 0.0).sum()),
    }
    return games, meta


def frozen_net_move(quotes: pd.DataFrame, games: pd.DataFrame) -> pd.DataFrame:
    frame = games.loc[
        :,
        [
            "game_id",
            "commence_time_utc",
            "week_first_commence_utc",
            "cutoff_utc",
            "decision_home_spread",
        ],
    ].copy()
    exposure = sharp_book_movement_features(quotes, frame)
    return exposure.loc[:, ["game_id", "leader_median_net_move", "leader_books"]]


def replica_net_move(
    quotes: pd.DataFrame, games: pd.DataFrame, *, include_sunday: bool
) -> pd.DataFrame:
    frame = games.loc[
        :,
        [
            "game_id",
            "commence_time_utc",
            "week_first_commence_utc",
            "cutoff_utc",
            "decision_home_spread",
            "week_sunday_utc",
            "wednesday_utc",
        ],
    ].copy()
    quote = quotes.loc[
        quotes["market"].eq("spreads") & quotes["bookmaker_key"].isin(LEADERSHIP_WEIGHTS),
        [
            "nflverse_game_id",
            "bookmaker_key",
            "home_spread_line",
            "observed_at_utc",
            "bookmaker_last_update_utc",
        ],
    ].rename(columns={"nflverse_game_id": "game_id"})
    quote = quote.merge(
        frame[["game_id", "cutoff_utc", "week_sunday_utc", "wednesday_utc"]], on="game_id"
    )
    monday = quote["week_sunday_utc"] - pd.Timedelta(days=6)
    keep = (
        quote["observed_at_utc"].lt(quote["cutoff_utc"])
        & quote["observed_at_utc"].ge(monday)
        & quote["bookmaker_last_update_utc"].le(quote["observed_at_utc"])
        & np.isfinite(quote["home_spread_line"])
    )
    if not include_sunday:
        keep = keep & quote["observed_at_utc"].lt(quote["week_sunday_utc"])
    quote = quote.loc[keep].copy()
    keys = ["game_id", "bookmaker_key", "observed_at_utc"]
    quote = quote.sort_values(keys).drop_duplicates(keys)
    quote["move"] = quote.groupby(["game_id", "bookmaker_key"])["home_spread_line"].diff()
    quote = quote.loc[quote["observed_at_utc"].ge(quote["wednesday_utc"]) & quote["move"].notna()]
    books = quote.groupby(["game_id", "bookmaker_key"], as_index=False)["move"].sum()
    leaders = books.loc[books["bookmaker_key"].isin(LEADER_BOOKS)]
    summary = leaders.groupby("game_id").agg(
        leader_median_net_move=("move", "median"), leader_books=("move", "size")
    )
    out = frame[["game_id"]].copy()
    out["leader_median_net_move"] = (
        out["game_id"].map(summary["leader_median_net_move"]).fillna(0.0)
    )
    out["leader_books"] = out["game_id"].map(summary["leader_books"]).fillna(0).astype(int)
    return out


def policy_pick(
    quotes: pd.DataFrame, games: pd.DataFrame, cutoff: pd.Series, *, include_sunday: bool
) -> pd.Series:
    frame = games.copy()
    frame["cutoff_utc"] = pd.to_datetime(cutoff, utc=True).to_numpy()
    if include_sunday:
        moves = replica_net_move(quotes, frame, include_sunday=True)
    else:
        moves = frozen_net_move(quotes, frame)
    merged = frame.merge(moves, on="game_id", how="left", validate="one_to_one")
    fires = merged["leader_books"].fillna(0).gt(0)
    move = merged["leader_median_net_move"].fillna(0.0).where(fires, 0.0)
    picked = refresh_pick(merged["card_pick_home"], move, threshold=merged["threshold"])
    return pd.Series(picked.to_numpy(bool), index=merged["game_id"].to_numpy())


def served_pass_cutoff(games: pd.DataFrame) -> pd.Series:
    sunday_local = pd.to_datetime(games["week_sunday_utc"], utc=True).dt.tz_convert(EASTERN)
    deadline = pd.to_datetime(games["deadline_utc"], utc=True)
    best = pd.Series(pd.NaT, index=games.index, dtype="datetime64[ns, UTC]")
    for _name, offset_days, clock in SERVED_PASSES:
        day = sunday_local.dt.normalize() + pd.Timedelta(days=offset_days)
        instant = (
            (day + pd.Timedelta(hours=clock.hour, minutes=clock.minute))
            .dt.tz_localize(None)
            .dt.tz_localize(EASTERN, nonexistent="shift_forward", ambiguous=True)
            .dt.tz_convert(UTC)
        )
        usable = instant.lt(deadline)
        best = best.where(~(usable & (best.isna() | instant.gt(best))), instant)
    return best.fillna(pd.to_datetime(games["wednesday_utc"], utc=True))


def served_pass_count(games: pd.DataFrame) -> pd.DataFrame:
    sunday_local = pd.to_datetime(games["week_sunday_utc"], utc=True).dt.tz_convert(EASTERN)
    deadline = pd.to_datetime(games["deadline_utc"], utc=True)
    rows: list[pd.DataFrame] = []
    for name, offset_days, clock in SERVED_PASSES:
        day = sunday_local.dt.normalize() + pd.Timedelta(days=offset_days)
        instant = (
            (day + pd.Timedelta(hours=clock.hour, minutes=clock.minute))
            .dt.tz_localize(None)
            .dt.tz_localize(EASTERN, nonexistent="shift_forward", ambiguous=True)
            .dt.tz_convert(UTC)
        )
        rows.append(
            pd.DataFrame(
                {
                    "season": games["season"].to_numpy(),
                    "week": games["week"].to_numpy(),
                    "instant": instant.to_numpy(),
                    "name": name,
                    "reaches_a_game": instant.lt(deadline).to_numpy(),
                }
            )
        )
    passes = pd.concat(rows, ignore_index=True)
    passes = passes.loc[passes["reaches_a_game"]]
    return passes.groupby(["season", "week"])["name"].nunique().rename("refreshes").reset_index()


def leader_median_series(quotes: pd.DataFrame, games: pd.DataFrame) -> pd.DataFrame:
    quote = quotes.loc[
        quotes["bookmaker_key"].isin(LEADER_BOOKS),
        ["nflverse_game_id", "bookmaker_key", "home_spread_line", "observed_at_utc"],
    ].rename(columns={"nflverse_game_id": "game_id"})
    median = (
        quote.groupby(["game_id", "observed_at_utc"], as_index=False)["home_spread_line"]
        .median()
        .rename(columns={"home_spread_line": "median_home_spread"})
    )
    anchors = games.loc[
        :, ["game_id", "season", "week", "wednesday_utc", "deadline_utc", "decision_home_spread"]
    ]
    median = median.merge(anchors, on="game_id", how="inner")
    median = median.loc[
        median["observed_at_utc"].ge(median["wednesday_utc"])
        & median["observed_at_utc"].lt(median["deadline_utc"])
    ]
    return median.sort_values(["game_id", "observed_at_utc"]).reset_index(drop=True)


def trigger_walk(series: pd.DataFrame, threshold: float) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for game_id, block in series.groupby("game_id", sort=False):
        reference = float(block["decision_home_spread"].iat[0])
        last_trigger: pd.Timestamp | None = None
        triggers = 0
        for stamp, line in zip(
            block["observed_at_utc"].to_numpy(), block["median_home_spread"].to_numpy(), strict=True
        ):
            if not np.isfinite(line):
                continue
            if abs(float(line) - reference) >= threshold:
                reference = float(line)
                last_trigger = pd.Timestamp(stamp)
                triggers += 1
        rows.append(
            {
                "game_id": game_id,
                "season": int(block["season"].iat[0]),
                "week": int(block["week"].iat[0]),
                "last_trigger_utc": last_trigger,
                "triggers": triggers,
            }
        )
    return pd.DataFrame(rows)


def paired_stats(delta: np.ndarray, blocks: pd.DataFrame, block: str) -> dict[str, float]:
    stats = blocked_bootstrap_matrix(
        delta[:, np.newaxis],
        blocks.reset_index(drop=True),
        block=block,
        samples=SAMPLES,
        seed=SEED,
    )
    return {
        "estimate_accuracy_points": float(stats["estimate"][0] * 100.0),
        "lower_accuracy_points": float(stats["lower"][0] * 100.0),
        "upper_accuracy_points": float(stats["upper"][0] * 100.0),
        "probability_positive": float(stats["probability_positive"][0]),
        "standard_error_accuracy_points": float(stats["standard_error"][0] * 100.0),
        "blocks": int(stats["block_count"]),
    }


def correctness(pick_home: pd.Series, margin: pd.Series) -> pd.Series:
    home_covers = margin.gt(0.0)
    return (pick_home.astype(bool) == home_covers).astype(float)


def cell(
    label: str,
    candidate: pd.Series,
    baseline: pd.Series,
    scored: pd.DataFrame,
    refreshes: dict[str, float] | None,
) -> dict[str, Any]:
    delta = (candidate - baseline).to_numpy(dtype=float)
    row: dict[str, Any] = {
        "label": label,
        "n": len(scored),
        "baseline_accuracy_pct": float(baseline.mean() * 100.0),
        "candidate_accuracy_pct": float(candidate.mean() * 100.0),
        "delta_accuracy_points": float(delta.mean() * 100.0),
        "week_blocked": paired_stats(delta, scored[["season", "week"]], "week"),
        "season_blocked": paired_stats(delta, scored[["season", "week"]], "season"),
    }
    if refreshes is not None:
        row["refreshes_per_week"] = refreshes
    return row


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "MKT-08: compare refresh-timing policies for the served late-week follow rule "
            "on the historical market archive."
        )
    )
    parser.add_argument(
        "--per-game", default="artifacts/opener_evaluation/20260910T211255Z/per_game.parquet"
    )
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--market-root", default="data/market/raw")
    parser.add_argument("--out", required=True)
    parser.add_argument("--quote-cache", default=None)
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    data_root = Path(args.data_root)
    market_root = Path(args.market_root)
    cache_path = Path(args.quote_cache) if args.quote_cache else out / "leader_quotes.parquet"
    seasons = tuple(range(2020, 2026))
    if not cache_path.is_file():
        build_quote_cache(market_root, cache_path, seasons)
    quotes = pd.read_parquet(cache_path)
    quotes["observed_at_utc"] = pd.to_datetime(quotes["observed_at_utc"], utc=True)
    quotes["bookmaker_last_update_utc"] = pd.to_datetime(
        quotes["bookmaker_last_update_utc"], utc=True
    )

    cadence = capture_cadence(market_root)
    (out / "capture_cadence.json").write_text(json.dumps(cadence, indent=2), encoding="utf-8")

    games, meta = build_games(Path(args.per_game), data_root)
    games = games.loc[games["season"].isin(seasons)].reset_index(drop=True)

    check = games.copy()
    check["cutoff_utc"] = pd.to_datetime(check["deadline_utc"], utc=True).to_numpy()
    frozen = frozen_net_move(quotes, check).set_index("game_id")
    replica = replica_net_move(quotes, check, include_sunday=False).set_index("game_id")
    replica_max_gap = float(
        (frozen["leader_median_net_move"] - replica["leader_median_net_move"]).abs().max()
    )
    if replica_max_gap > 1e-9:
        raise ValueError(
            "the include-Sunday replica does not reproduce the frozen served rule "
            f"when the Sunday filter is restored (max gap {replica_max_gap})"
        )

    fixed_cutoff = served_pass_cutoff(games)
    gap_hours = (
        pd.to_datetime(games["deadline_utc"], utc=True) - pd.to_datetime(fixed_cutoff, utc=True)
    ).dt.total_seconds() / 3600.0
    kick_et = pd.to_datetime(games["commence_time_utc"], utc=True).dt.tz_convert(EASTERN)
    slot = kick_et.dt.day_name().str[:3] + " " + kick_et.dt.strftime("%H:%M")
    gap_frame = pd.DataFrame({"slot": slot.to_numpy(), "gap_hours": gap_hours.to_numpy()})
    served_gap = {
        "all_games": {
            "median_hours": float(gap_frame["gap_hours"].median()),
            "mean_hours": float(gap_frame["gap_hours"].mean()),
            "max_hours": float(gap_frame["gap_hours"].max()),
        },
        "by_kickoff_slot": [
            {
                "slot": str(name),
                "games": len(block),
                "median_hours": float(block["gap_hours"].median()),
                "max_hours": float(block["gap_hours"].max()),
            }
            for name, block in gap_frame.groupby("slot")
        ],
    }

    picks: dict[str, pd.Series] = {}
    picks["tuesday_only"] = pd.Series(
        games["card_pick_home"].to_numpy(bool), index=games["game_id"].to_numpy()
    )
    picks["fixed_passes"] = policy_pick(quotes, games, fixed_cutoff, include_sunday=False)
    picks["as_late_as_allowed"] = policy_pick(
        quotes, games, games["deadline_utc"], include_sunday=False
    )
    picks["as_late_as_allowed_incl_sunday"] = policy_pick(
        quotes, games, games["deadline_utc"], include_sunday=True
    )

    series = leader_median_series(quotes, games)
    trigger_cost: dict[str, pd.DataFrame] = {}
    for threshold in TRIGGER_THRESHOLDS:
        walk = trigger_walk(series, threshold)
        name = f"news_triggered_{str(threshold).replace('.', '_')}"
        cutoff = (
            games["game_id"]
            .map(walk.set_index("game_id")["last_trigger_utc"])
            .fillna(pd.to_datetime(games["wednesday_utc"], utc=True))
        )
        picks[name] = policy_pick(quotes, games, cutoff, include_sunday=False)
        trigger_cost[name] = walk

    frame = games.copy()
    for name, side in picks.items():
        frame[f"pick_{name}"] = frame["game_id"].map(side).astype(bool)
    scored_all = frame.loc[frame["margin_vs_open"].ne(0.0)].reset_index(drop=True)
    for name in picks:
        scored_all[f"correct_{name}"] = correctness(
            scored_all[f"pick_{name}"], scored_all["margin_vs_open"]
        )
    scored_all.to_parquet(out / "per_game.parquet", index=False)

    fixed_counts = served_pass_count(games)
    results: dict[str, Any] = {
        "generated_at_utc": datetime.now(tz=UTC).isoformat(),
        "per_game_archive": str(args.per_game),
        "samples": SAMPLES,
        "seed": SEED,
        "coverage": {**meta, "replica_max_gap_vs_frozen_rule": replica_max_gap},
        "capture_cadence": cadence,
        "served_passes": [
            {"job": name, "days_from_week_sunday": offset, "eastern": clock.strftime("%H:%M")}
            for name, offset, clock in SERVED_PASSES
        ],
        "served_pass_gap_to_deadline": served_gap,
        "windows": {},
        "decay": {},
    }

    for window_name, window_seasons in WINDOWS:
        scored = scored_all.loc[scored_all["season"].isin(window_seasons)].reset_index(drop=True)
        baseline = scored["correct_tuesday_only"]
        window_rows: list[dict[str, Any]] = []
        for name in picks:
            if name == "tuesday_only":
                cost = {"mean": 0.0, "median": 0.0, "max": 0.0}
            elif name == "fixed_passes":
                counts = fixed_counts.loc[fixed_counts["season"].isin(window_seasons), "refreshes"]
                cost = {
                    "mean": float(counts.mean()),
                    "median": float(counts.median()),
                    "max": float(counts.max()),
                }
            elif name.startswith("news_triggered"):
                walk = trigger_cost[name]
                walk = walk.loc[walk["season"].isin(window_seasons)]
                per_week = walk.groupby(["season", "week"])["triggers"].sum()
                distinct = (
                    walk.loc[walk["last_trigger_utc"].notna()]
                    .groupby(["season", "week"])["last_trigger_utc"]
                    .nunique()
                )
                cost = {
                    "mean": float(distinct.reindex(per_week.index).fillna(0).mean()),
                    "median": float(distinct.reindex(per_week.index).fillna(0).median()),
                    "max": float(distinct.reindex(per_week.index).fillna(0).max()),
                    "game_triggers_per_week_mean": float(per_week.mean()),
                }
            else:
                per_week = scored.groupby(["season", "week"])["deadline_utc"].nunique()
                cost = {
                    "mean": float(per_week.mean()),
                    "median": float(per_week.median()),
                    "max": float(per_week.max()),
                }
            row = cell(name, scored[f"correct_{name}"], baseline, scored, cost)
            row["picks_changed_vs_tuesday"] = int(
                (scored[f"pick_{name}"] != scored["pick_tuesday_only"]).sum()
            )
            row["seasons"] = list(window_seasons)
            window_rows.append(row)
        per_season: dict[str, list[dict[str, Any]]] = {}
        for season in window_seasons:
            block = scored.loc[scored["season"].eq(season)]
            if block.empty:
                continue
            per_season[str(season)] = [
                {
                    "label": name,
                    "n": len(block),
                    "accuracy_pct": float(block[f"correct_{name}"].mean() * 100.0),
                    "delta_accuracy_points": float(
                        (block[f"correct_{name}"] - block["correct_tuesday_only"]).mean() * 100.0
                    ),
                    "picks_changed_vs_tuesday": int(
                        (block[f"pick_{name}"] != block["pick_tuesday_only"]).sum()
                    ),
                }
                for name in picks
            ]
        head_to_head: list[dict[str, Any]] = []
        incumbent = scored["correct_fixed_passes"]
        for name in picks:
            if name == "fixed_passes":
                continue
            row = cell(
                f"{name}_vs_fixed_passes", scored[f"correct_{name}"], incumbent, scored, None
            )
            row["picks_changed_vs_fixed_passes"] = int(
                (scored[f"pick_{name}"] != scored["pick_fixed_passes"]).sum()
            )
            head_to_head.append(row)
        changed_rows: list[dict[str, Any]] = []
        for name in picks:
            if name == "tuesday_only":
                continue
            changed = scored.loc[scored[f"pick_{name}"] != scored["pick_tuesday_only"]]
            changed_rows.append(
                {
                    "label": name,
                    "changed_picks": len(changed),
                    "changed_new_accuracy_pct": (
                        float(changed[f"correct_{name}"].mean() * 100.0) if len(changed) else None
                    ),
                    "changed_tuesday_accuracy_pct": (
                        float(changed["correct_tuesday_only"].mean() * 100.0)
                        if len(changed)
                        else None
                    ),
                }
            )
        results["windows"][window_name] = {
            "cells": window_rows,
            "per_season": per_season,
            "head_to_head_vs_fixed_passes": head_to_head,
            "changed_pick_hit_rate": changed_rows,
        }

    decay_rows: list[dict[str, Any]] = []
    for lead in DECAY_LEAD_HOURS:
        cutoff = pd.to_datetime(games["deadline_utc"], utc=True) - pd.Timedelta(hours=lead)
        for arm, include_sunday in (("served_rule", False), ("incl_sunday", True)):
            side = policy_pick(quotes, games, cutoff, include_sunday=include_sunday)
            column = frame["game_id"].map(side).astype(bool)
            block = frame.assign(decay_pick=column.to_numpy())
            block = block.loc[block["margin_vs_open"].ne(0.0)].reset_index(drop=True)
            block["decay_correct"] = correctness(block["decay_pick"], block["margin_vs_open"])
            base = correctness(block["pick_tuesday_only"], block["margin_vs_open"])
            for window_name, window_seasons in WINDOWS:
                sub = block.loc[block["season"].isin(window_seasons)].reset_index(drop=True)
                if sub.empty:
                    continue
                sub_base = base.loc[block["season"].isin(window_seasons)].reset_index(drop=True)
                delta = (sub["decay_correct"] - sub_base).to_numpy(dtype=float)
                stats = paired_stats(delta, sub[["season", "week"]], "week")
                decay_rows.append(
                    {
                        "arm": arm,
                        "window": window_name,
                        "hours_before_deadline": lead,
                        "n": len(sub),
                        "picks_changed_vs_tuesday": int(
                            (sub["decay_pick"] != sub["pick_tuesday_only"]).sum()
                        ),
                        "accuracy_pct": float(sub["decay_correct"].mean() * 100.0),
                        "baseline_accuracy_pct": float(sub_base.mean() * 100.0),
                        "delta_accuracy_points": float(delta.mean() * 100.0),
                        "week_blocked": stats,
                    }
                )
    results["decay"] = decay_rows
    pd.DataFrame(
        [
            {
                "arm": row["arm"],
                "window": row["window"],
                "hours_before_deadline": row["hours_before_deadline"],
                "n": row["n"],
                "picks_changed_vs_tuesday": row["picks_changed_vs_tuesday"],
                "accuracy_pct": row["accuracy_pct"],
                "delta_accuracy_points": row["delta_accuracy_points"],
                "lower": row["week_blocked"]["lower_accuracy_points"],
                "upper": row["week_blocked"]["upper_accuracy_points"],
                "probability_positive": row["week_blocked"]["probability_positive"],
            }
            for row in decay_rows
        ]
    ).to_csv(out / "decay_curve.csv", index=False)

    (out / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps({"out": str(out), "games": len(scored_all)}, indent=2))


if __name__ == "__main__":
    main()
