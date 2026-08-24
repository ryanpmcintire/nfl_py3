"""VegasInsider multi-book dispersion screen: cross-book spread/total
disagreement on Tuesday/Wednesday Wayback boards (2005-2016 backfill,
artifacts/vegasinsider_backfill/20260822T033952Z) as a residual-uncertainty
signal, joined to REG 2009-2016 ATS outcomes.

Batch 1: four predeclared cells (directions frozen in
docs/vi_dispersion_screen.md BEFORE any outcome was computed; only feature
distributions were inspected): top-tercile spread-SD underdog (+1),
bottom-tercile underdog control (+1), era split of the top-tercile underdog
cell, top-tercile favorite (-1). NOTE 2026-08-24 (measured): the VI parquet
cannot encode favorite side -- its spread_line is the displayed favorite-side
quote (~97% negative), so those flags actually scored away-side/home-side
contrasts; numbers stand, side labels do not.

Batch 2 (docs/vi_dispersion_screen.md, predeclared before these outcomes):
three orientation-free within-season-tercile cells -- P: top-vs-bottom
spread-SD vs home_cover; S2: dispersion vs absolute close-minus-opener line
movement; S3: total-SD vs actual-total overshoot of the opener total.
Week-blocked bootstrap primary (20k, seed 20260823), season-blocked
secondary, full-slate-scaled accuracy_points for P and native points for
S2/S3. Measure-only: never writes either registry JSON; stamps a run log to
registry/experiments/vi-dispersion-screen/.
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from nfl_ats.features import add_ats_outcomes  # noqa: E402
from nfl_ats.provenance import artifact_provenance, write_experiment_artifact  # noqa: E402

BOOTSTRAP_SAMPLES = 20_000
BOOTSTRAP_SEED = 20260823
SEASON_START = 2009
SEASON_END = 2016
BOARD_SEASONS = tuple(range(2005, 2017))
EXCLUDED_REDUCED_CONFIDENCE_SEASONS = frozenset({2006})
TUESDAY_WEDNESDAY = frozenset({1, 2})
MIN_BOOKS = 3
RANGE_CAP = 10.0
ERA_SPLITS: tuple[tuple[str, int, int], ...] = (
    ("2009_2012", 2009, 2012),
    ("2013_2016", 2013, 2016),
)
RELIABILITY_DRAWS = 250

VI_CODE_TO_SCHEDULE_DEFAULTS = {"LAR": "STL", "LAC": "SD", "LV": "OAK"}
VI_CODE_TO_SCHEDULE_OVERRIDES = {"LAR": {2016: "LA"}}

DEFAULT_BACKFILL = REPO / "artifacts/vegasinsider_backfill/20260822T033952Z"
SCHEDULE_COLUMNS = [
    "game_id",
    "season",
    "week",
    "game_type",
    "gameday",
    "home_team",
    "away_team",
    "result",
    "spread_line",
    "home_score",
    "away_score",
]


def _latest_schedules() -> Path:
    candidates = sorted((REPO / "data/raw").glob("*/schedules.parquet"))
    if not candidates:
        raise FileNotFoundError("no data/raw/*/schedules.parquet snapshot found")
    return candidates[-1]


DEFAULT_SCHEDULES = _latest_schedules()


def vi_to_sched(code: str, season: int) -> str:
    default = VI_CODE_TO_SCHEDULE_DEFAULTS.get(code)
    if default is None:
        return code
    return VI_CODE_TO_SCHEDULE_OVERRIDES.get(code, {}).get(season, default)


def season_of(day: pd.Timestamp) -> int:
    return int(day.year) if day.month >= 6 else int(day.year) - 1


def load_board_instances(backfill_dir: Path) -> pd.DataFrame:
    frames = [
        pd.read_parquet(path)
        for path in sorted(backfill_dir.glob("season_*.parquet"))
        if int(path.stem.split("_")[1]) in BOARD_SEASONS
    ]
    tidy = pd.concat(frames, ignore_index=True)
    tidy["capture_dt"] = pd.to_datetime(tidy["capture_ts"], format="%Y%m%d%H%M%S")
    tidy["game_day"] = pd.to_datetime(tidy["game_date"])
    tidy["capture_weekday"] = tidy["capture_dt"].dt.weekday
    tidy["season"] = np.where(
        tidy["game_day"].dt.month >= 6, tidy["game_day"].dt.year, tidy["game_day"].dt.year - 1
    )
    tidy["season"] = tidy["season"].astype(int)
    tidy = tidy.loc[~tidy["season"].isin(EXCLUDED_REDUCED_CONFIDENCE_SEASONS)]
    tw = tidy.loc[tidy["capture_weekday"].isin(TUESDAY_WEDNESDAY)].copy()

    def _kickoff_et(row: pd.Series) -> pd.Timestamp | None:
        try:
            local = datetime.strptime(str(row["kickoff_time"]), "%I:%M %p")
        except ValueError:
            return None
        day = row["game_day"]
        return pd.Timestamp(
            year=day.year,
            month=day.month,
            day=day.day,
            hour=local.hour,
            minute=local.minute,
            tz="America/New_York",
        )

    rows: list[dict[str, Any]] = []
    n_leak_dropped = 0
    n_range_capped = 0
    grouped = tw.groupby(["capture_ts", "game_day", "away", "home"], sort=False)
    for (cap, gday, away, home), grp in grouped:
        sp = (
            grp.dropna(subset=["book"])
            .dropna(subset=["spread_line"])
            .drop_duplicates("book")
            .sort_values("book")
        )
        tt = (
            grp.dropna(subset=["book"])
            .dropna(subset=["total_line"])
            .drop_duplicates("book")
            .sort_values("book")
        )
        n_books_spread = len(sp)
        n_books_total = len(tt)
        spread_sd = float(sp["spread_line"].std(ddof=1)) if n_books_spread >= 2 else np.nan
        total_sd = float(tt["total_line"].std(ddof=1)) if n_books_total >= 2 else np.nan
        spread_range = (
            float(sp["spread_line"].max() - sp["spread_line"].min())
            if n_books_spread >= 2
            else np.nan
        )
        total_range = (
            float(tt["total_line"].max() - tt["total_line"].min()) if n_books_total >= 2 else np.nan
        )
        capture_dt = grp["capture_dt"].iloc[0]
        kickoff = _kickoff_et(grp.iloc[0])
        capture_et = pd.Timestamp(capture_dt).tz_localize("UTC").tz_convert("America/New_York")
        if kickoff is None or not (capture_et < kickoff):
            n_leak_dropped += 1
            continue
        if (
            np.isfinite(spread_range)
            and np.isfinite(total_range)
            and (spread_range > RANGE_CAP or total_range > RANGE_CAP)
        ):
            n_range_capped += 1
            continue
        rows.append(
            {
                "capture_ts": cap,
                "capture_dt": capture_dt,
                "game_day": gday,
                "inferred_season": season_of(gday),
                "away": away,
                "home": home,
                "spread_sd": spread_sd,
                "total_sd": total_sd,
                "spread_range": spread_range,
                "total_range": total_range,
                "n_books_spread": n_books_spread,
                "n_books_total": n_books_total,
                "median_spread": float(sp["spread_line"].median()) if n_books_spread else np.nan,
                "mean_total": float(tt["total_line"].mean()) if n_books_total else np.nan,
                "books": sorted(sp["book"].tolist()),
                "spread_lines": sp["spread_line"].to_numpy(dtype=float).tolist(),
                "total_lines": tt["total_line"].to_numpy(dtype=float).tolist(),
            }
        )
    instances = pd.DataFrame(rows)
    instances.attrs["leak_guard_dropped"] = n_leak_dropped
    instances.attrs["range_cap_dropped"] = n_range_capped
    return instances


def join_schedule(instances: pd.DataFrame, schedules_path: Path) -> pd.DataFrame:
    sched = pd.read_parquet(schedules_path, columns=SCHEDULE_COLUMNS)
    sched = sched.loc[sched["game_type"] == "REG"].copy()
    sched["gameday"] = pd.to_datetime(sched["gameday"])
    sched["season"] = pd.to_numeric(sched["season"], errors="raise").astype(int)

    index: dict[tuple[str, str], list[tuple[pd.Timestamp, int, str]]] = {}
    for row in sched.itertuples(index=False):
        gameday = pd.Timestamp(row.gameday)
        index.setdefault((str(row.away_team), str(row.home_team)), []).append(
            (gameday, int(row.season), str(row.game_id))
        )

    matched_ids: list[str | None] = []
    ambiguous = 0
    for inst in instances.itertuples(index=False):
        candidates: set[str] = set()
        season = int(inst.inferred_season)
        for alt_season in {season - 1, season, season + 1}:
            alt_pair = (
                vi_to_sched(str(inst.away), alt_season),
                vi_to_sched(str(inst.home), alt_season),
            )
            for gameday, entry_season, game_id in index.get(alt_pair, []):
                game_day = pd.Timestamp(inst.game_day)
                if abs((game_day.normalize() - pd.Timestamp(gameday).normalize()).days) <= 1:
                    candidates.add(f"{entry_season}:{game_id}")
        if len(candidates) == 1:
            matched_ids.append(next(iter(candidates)))
        else:
            if len(candidates) > 1:
                ambiguous += 1
            matched_ids.append(None)
    instances = instances.copy()
    instances["match_key"] = matched_ids
    instances.attrs["ambiguous_matches"] = ambiguous
    return instances


def game_level_features(instances: pd.DataFrame) -> pd.DataFrame:
    usable = instances.loc[instances["match_key"].notna()].copy()
    usable = usable.sort_values("capture_dt").drop_duplicates("match_key", keep="first")
    usable = usable.rename(columns={"match_key": "game_key"})
    return usable.reset_index(drop=True)


def build_scored_frame(
    games: pd.DataFrame, schedules_path: Path
) -> tuple[pd.DataFrame, float, float]:
    sched = pd.read_parquet(schedules_path, columns=SCHEDULE_COLUMNS)
    sched = sched.loc[sched["game_type"] == "REG"].copy()
    sched["key"] = sched["season"].astype(int).astype(str) + ":" + sched["game_id"].astype(str)
    scored_games = sched.loc[sched["key"].isin(set(games["game_key"]))].merge(
        games, left_on="key", right_on="game_key", how="inner"
    )
    eligible = scored_games.loc[
        (scored_games["n_books_spread"] >= MIN_BOOKS) & scored_games["spread_sd"].notna()
    ].copy()
    q_low = float(eligible["spread_sd"].quantile(1 / 3))
    q_high = float(eligible["spread_sd"].quantile(2 / 3))
    eligible["disp_tercile"] = np.select(
        [eligible["spread_sd"] <= q_low, eligible["spread_sd"] >= q_high],
        ["bottom", "top"],
        default="middle",
    )
    eligible["home_is_favorite"] = eligible["median_spread"] < 0
    eligible = add_ats_outcomes(eligible)
    eligible = eligible.loc[eligible["home_cover"].notna()].reset_index(drop=True)

    rows = []
    for is_home in (True, False):
        side = pd.DataFrame(
            {
                "game_id": eligible["game_id"],
                "season": eligible["season"],
                "week": eligible["week"],
                "team_covered": eligible["home_cover"] if is_home else 1.0 - eligible["home_cover"],
                "is_home": is_home,
                "disp_top": eligible["disp_tercile"] == "top",
                "disp_bottom": eligible["disp_tercile"] == "bottom",
                "home_is_favorite": eligible["home_is_favorite"],
            }
        )
        rows.append(side)
    long_df = pd.concat(rows, ignore_index=True)
    long_df["week_block"] = long_df["season"] * 100 + long_df["week"]
    long_df["side_is_underdog"] = np.where(
        long_df["home_is_favorite"], ~long_df["is_home"], long_df["is_home"]
    ).astype(bool)
    long_df["side_is_favorite"] = ~long_df["side_is_underdog"] & long_df["home_is_favorite"].notna()
    long_df.loc[~long_df["home_is_favorite"] & ~long_df["side_is_underdog"], "side_is_favorite"] = (
        False
    )
    long_df["flag_top_dog"] = long_df["disp_top"] & long_df["side_is_underdog"]
    long_df["flag_bottom_dog"] = long_df["disp_bottom"] & long_df["side_is_underdog"]
    long_df["flag_top_fav"] = long_df["disp_top"] & long_df["side_is_favorite"]
    return long_df, q_low, q_high


def block_bootstrap_two_group(
    df: pd.DataFrame,
    *,
    flag_col: str,
    value_col: str,
    block_col: str,
    samples: int,
    seed: int,
) -> np.ndarray:
    blocks, block_index = np.unique(df[block_col].to_numpy(), return_inverse=True)
    block_index = np.asarray(block_index).reshape(-1)
    block_count = len(blocks)
    values = df[value_col].to_numpy(dtype=np.float64)
    flag = df[flag_col].to_numpy(dtype=bool)

    sums: dict[bool, np.ndarray] = {}
    counts: dict[bool, np.ndarray] = {}
    for group in (True, False):
        mask = flag == group
        sums[group] = np.bincount(
            block_index[mask], weights=values[mask], minlength=block_count
        ).astype(np.float64)
        counts[group] = np.bincount(block_index[mask], minlength=block_count).astype(np.float64)

    rng = np.random.default_rng(seed)
    drawn = rng.multinomial(block_count, np.full(block_count, 1.0 / block_count), size=samples)
    subset_count = drawn @ counts[True]
    complement_count = drawn @ counts[False]
    with np.errstate(invalid="ignore", divide="ignore"):
        mean_subset = (drawn @ sums[True]) / subset_count
        mean_complement = (drawn @ sums[False]) / complement_count
    gap = (mean_subset - mean_complement) * 100.0
    valid = (subset_count > 0) & (complement_count > 0)
    return gap[valid]


def summarize(
    df: pd.DataFrame,
    *,
    flag: pd.Series,
    sign: int,
    block_col: str,
    samples: int,
    seed: int,
) -> dict[str, Any]:
    n_total = len(df)
    n_flag = int(flag.sum())
    n_complement = n_total - n_flag
    if n_flag == 0 or n_complement == 0:
        return {
            "n_total": n_total,
            "n_flag": n_flag,
            "n_complement": n_complement,
            "insufficient_data": True,
        }

    work = df.copy()
    work["_flag"] = flag.to_numpy()
    subset_cover = float(work.loc[work["_flag"], "team_covered"].mean())
    complement_cover = float(work.loc[~work["_flag"], "team_covered"].mean())
    raw_gap_pts = (subset_cover - complement_cover) * 100.0
    fraction_of_slate = n_flag / n_total
    full_slate_effect_pts = sign * raw_gap_pts * fraction_of_slate

    draws = block_bootstrap_two_group(
        work,
        flag_col="_flag",
        value_col="team_covered",
        block_col=block_col,
        samples=samples,
        seed=seed,
    )
    signed_draws = sign * draws
    scaled_draws = signed_draws * fraction_of_slate
    lower, upper = (
        np.quantile(scaled_draws, [0.025, 0.975]) if len(scaled_draws) else (np.nan, np.nan)
    )

    return {
        "n_total": n_total,
        "n_flag": n_flag,
        "n_complement": n_complement,
        "n_blocks": int(work[block_col].nunique()),
        "subset_cover": subset_cover,
        "complement_cover": complement_cover,
        "raw_gap_pts": raw_gap_pts,
        "fraction_of_slate": fraction_of_slate,
        "full_slate_effect_pts": full_slate_effect_pts,
        "ci95_scaled": [float(lower), float(upper)],
        "probability_positive": float(np.mean(signed_draws > 0)) if len(signed_draws) else np.nan,
        "bootstrap_samples": samples,
        "dropped_draws": int(samples - len(draws)),
        "insufficient_data": False,
    }


def split_half_reliability(
    games: pd.DataFrame,
    seed: int,
    *,
    lines_col: str = "spread_lines",
    books_col: str = "n_books_spread",
) -> dict[str, Any]:
    pool = games.loc[games[books_col] >= 6]
    n_games_used = len(pool)
    if n_games_used < 30:
        return {
            "n_games_eligible": int(n_games_used),
            "reliability": None,
            "note": "too few >=6-book games for a stable split-half estimate",
        }
    rng = np.random.default_rng(seed)
    correlations: list[float] = []
    game_lines = [np.asarray(lines, dtype=np.float64) for lines in pool[lines_col]]
    for _draw in range(RELIABILITY_DRAWS):
        first: list[float] = []
        second: list[float] = []
        for lines in game_lines:
            order = rng.permutation(len(lines))
            mid = len(lines) // 2
            part_a = lines[order[:mid]]
            part_b = lines[order[mid:]]
            first.append(float(np.std(part_a, ddof=1)))
            second.append(float(np.std(part_b, ddof=1)))
        if np.std(first) == 0 or np.std(second) == 0:
            continue
        correlations.append(float(np.corrcoef(first, second)[0, 1]))
    mean_r = float(np.mean(correlations)) if correlations else 0.0
    spearman_brown = 2 * mean_r / (1 + mean_r) if (1 + mean_r) != 0 else 0.0
    return {
        "n_games_eligible": int(n_games_used),
        "draws": RELIABILITY_DRAWS,
        "usable_draws": len(correlations),
        "mean_split_correlation": mean_r,
        "reliability": spearman_brown,
    }


# ---------------------------------------------------------------------------
# Batch 2 (2026-08-24): orientation-free cells, predeclared in
# docs/vi_dispersion_screen.md BEFORE these outcomes were computed.
# Within-season terciles; top-vs-bottom arms; week-blocked primary and
# season-blocked secondary bootstraps.
# ---------------------------------------------------------------------------


def assign_within_season_terciles(frame: pd.DataFrame, value_col: str) -> pd.Series:
    labels = pd.Series("middle", index=frame.index, dtype="object")
    for _season, grp in frame.groupby("season", sort=True):
        values = grp[value_col]
        q_low = float(values.quantile(1 / 3))
        q_high = float(values.quantile(2 / 3))
        labels.loc[values.index[values <= q_low]] = "bottom"
        labels.loc[values.index[values >= q_high]] = "top"
    return labels


def bootstrap_two_arm_gap(
    df: pd.DataFrame,
    *,
    flag_top: pd.Series,
    flag_bottom: pd.Series,
    value_col: str,
    block_col: str,
    samples: int,
    seed: int,
) -> tuple[np.ndarray, int]:
    blocks, block_index = np.unique(df[block_col].to_numpy(), return_inverse=True)
    block_index = np.asarray(block_index).reshape(-1)
    block_count = len(blocks)
    values = df[value_col].to_numpy(dtype=np.float64)
    top = flag_top.to_numpy(dtype=bool)
    bottom = flag_bottom.to_numpy(dtype=bool)

    top_sums = np.bincount(block_index[top], weights=values[top], minlength=block_count).astype(
        np.float64
    )
    top_counts = np.bincount(block_index[top], minlength=block_count).astype(np.float64)
    bottom_sums = np.bincount(
        block_index[bottom], weights=values[bottom], minlength=block_count
    ).astype(np.float64)
    bottom_counts = np.bincount(block_index[bottom], minlength=block_count).astype(np.float64)

    rng = np.random.default_rng(seed)
    drawn = rng.multinomial(block_count, np.full(block_count, 1.0 / block_count), size=samples)
    top_n = drawn @ top_counts
    bottom_n = drawn @ bottom_counts
    with np.errstate(invalid="ignore", divide="ignore"):
        gap = (drawn @ top_sums) / top_n - (drawn @ bottom_sums) / bottom_n
    valid = (top_n > 0) & (bottom_n > 0)
    return gap[valid], int(samples - int(valid.sum()))


def summarize_two_arm(
    frame: pd.DataFrame,
    *,
    value_col: str,
    arm_col: str,
    block_col: str,
    samples: int,
    seed: int,
    gap_multiplier: float = 1.0,
    slate_scale: float = 1.0,
) -> dict[str, Any]:
    flag_top = frame[arm_col] == "top"
    flag_bottom = frame[arm_col] == "bottom"
    n_top = int(flag_top.sum())
    n_bottom = int(flag_bottom.sum())
    n_middle = len(frame) - n_top - n_bottom
    if n_top == 0 or n_bottom == 0:
        return {
            "n_top": n_top,
            "n_bottom": n_bottom,
            "n_middle": n_middle,
            "insufficient_data": True,
        }
    mean_top = float(frame.loc[flag_top, value_col].mean())
    mean_bottom = float(frame.loc[flag_bottom, value_col].mean())
    raw_gap = mean_top - mean_bottom

    draws, dropped = bootstrap_two_arm_gap(
        frame,
        flag_top=flag_top,
        flag_bottom=flag_bottom,
        value_col=value_col,
        block_col=block_col,
        samples=samples,
        seed=seed,
    )
    scaled_draws = draws * gap_multiplier * slate_scale
    lower, upper = (
        np.quantile(scaled_draws, [0.025, 0.975]) if len(scaled_draws) else (np.nan, np.nan)
    )
    return {
        "n_top": n_top,
        "n_bottom": n_bottom,
        "n_middle_excluded": n_middle,
        "n_blocks": int(frame[block_col].nunique()),
        "mean_top": mean_top,
        "mean_bottom": mean_bottom,
        "raw_gap": raw_gap,
        "gap_multiplier": gap_multiplier,
        "slate_scale": slate_scale,
        "effect": raw_gap * gap_multiplier * slate_scale,
        "ci95": [float(lower), float(upper)],
        "probability_positive": float(np.mean(scaled_draws > 0)) if len(scaled_draws) else np.nan,
        "bootstrap_samples": samples,
        "dropped_draws": dropped,
        "insufficient_data": False,
    }


def per_season_stability(
    frame: pd.DataFrame, *, value_col: str, arm_col: str
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for season, grp in frame.groupby("season", sort=True):
        top = grp.loc[grp[arm_col] == "top", value_col]
        bottom = grp.loc[grp[arm_col] == "bottom", value_col]
        rows.append(
            {
                "season": int(season),
                "n_top": len(top),
                "n_bottom": len(bottom),
                "mean_top": float(top.mean()) if len(top) else None,
                "mean_bottom": float(bottom.mean()) if len(bottom) else None,
                "raw_gap": float(top.mean() - bottom.mean()) if len(top) and len(bottom) else None,
            }
        )
    return rows


def build_batch2_frames(
    games: pd.DataFrame, schedules_path: Path
) -> dict[str, pd.DataFrame | dict[str, int]]:
    sched = pd.read_parquet(schedules_path, columns=SCHEDULE_COLUMNS)
    sched = sched.loc[sched["game_type"] == "REG"].copy()
    sched["key"] = sched["season"].astype(int).astype(str) + ":" + sched["game_id"].astype(str)
    merged = games.merge(
        sched[["key", "season", "week", "result", "spread_line", "home_score", "away_score"]],
        left_on="game_key",
        right_on="key",
        how="inner",
    )
    merged = merged.loc[merged["season"].between(SEASON_START, SEASON_END)].reset_index(drop=True)
    merged = add_ats_outcomes(merged)
    merged["week_block"] = merged["season"].astype(int) * 100 + merged["week"].astype(int)

    spread_gate = (merged["n_books_spread"] >= MIN_BOOKS) & merged["spread_sd"].notna()
    total_gate = (merged["n_books_total"] >= MIN_BOOKS) & merged["total_sd"].notna()

    p_frame = merged.loc[spread_gate & merged["home_cover"].notna()].copy()
    p_frame["arm"] = assign_within_season_terciles(p_frame, "spread_sd")

    s2_frame = p_frame.loc[p_frame["spread_line"].notna()].copy()
    s2_frame["movement"] = (s2_frame["spread_line"].abs() - s2_frame["median_spread"].abs()).abs()
    s2_frame["arm"] = assign_within_season_terciles(s2_frame, "spread_sd")

    s3_frame = merged.loc[
        total_gate & merged["home_score"].notna() & merged["away_score"].notna()
    ].copy()
    s3_frame["overshoot"] = (s3_frame["home_score"] + s3_frame["away_score"]) - s3_frame[
        "mean_total"
    ]
    s3_frame["arm"] = assign_within_season_terciles(s3_frame, "total_sd")

    scoreable = merged.loc[merged["season"].between(SEASON_START, SEASON_END)]
    coverage: dict[str, int] = {
        "matched_games_all_seasons": len(games),
        "matched_games_lt2_named_book_spreads": int((games["n_books_spread"] < 2).sum()),
        "matched_games_2book_band_dropped_by_min3": int(
            ((games["n_books_spread"] >= 2) & (games["n_books_spread"] < MIN_BOOKS)).sum()
        ),
        "scoreable_window_games": len(scoreable),
        "scoreable_lt2_named_book_spreads": int((scoreable["n_books_spread"] < 2).sum()),
        "scoreable_2book_band_dropped_by_min3": int(
            ((scoreable["n_books_spread"] >= 2) & (scoreable["n_books_spread"] < MIN_BOOKS)).sum()
        ),
        "cell_S2_missing_close_line_dropped": int(len(p_frame) - len(s2_frame)),
        "cell_S3_totals_gate_failures_among_spread_scored": int(
            (spread_gate & ~total_gate)[merged["home_cover"].notna()].sum()
        ),
    }
    return {"P": p_frame, "S2": s2_frame, "S3": s3_frame, "coverage_counts": coverage}


def score_batch2(
    frames: dict[str, pd.DataFrame], samples: int, seed: int
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    p_frame = frames["P"]
    frac_arms_p = ((p_frame["arm"] != "middle").mean()) if len(p_frame) else 0.0

    specs: list[tuple[str, str, str, str, float]] = [
        (
            "vi_disp_homecover_top_vs_bottom_tercile",
            "P",
            "home_cover",
            "Within-season spread-SD terciles, TOP vs BOTTOM, outcome home_cover; "
            "+1 predeclared (informed by batch-1's measured HOME-side contrast)",
            100.0,
        ),
        (
            "vi_disp_movement_top_vs_bottom_tercile",
            "S2",
            "movement",
            "Within-season spread-SD terciles, TOP vs BOTTOM, outcome "
            "|close|-|opener| absolute line movement in points; +1 predeclared",
            1.0,
        ),
        (
            "vi_totaldisp_overshoot_top_vs_bottom_tercile",
            "S3",
            "overshoot",
            "Within-season total-SD terciles, TOP vs BOTTOM, outcome actual total "
            "minus opener mean posted total (points); +1 predeclared",
            1.0,
        ),
    ]

    results: list[dict[str, Any]] = []
    for name, key, value_col, description, multiplier in specs:
        frame = frames[key]
        slate_scale = frac_arms_p if key == "P" else 1.0
        week_blocked = summarize_two_arm(
            frame,
            value_col=value_col,
            arm_col="arm",
            block_col="week_block",
            samples=samples,
            seed=seed,
            gap_multiplier=multiplier,
            slate_scale=slate_scale,
        )
        season_blocked = summarize_two_arm(
            frame,
            value_col=value_col,
            arm_col="arm",
            block_col="season",
            samples=samples,
            seed=seed,
            gap_multiplier=multiplier,
            slate_scale=slate_scale,
        )
        results.append(
            {
                "name": name,
                "population_key": key,
                "direction": 1,
                "description": description,
                "value_units": (
                    "accuracy_points_full_slate_scaled"
                    if key == "P"
                    else ("line_movement_points" if key == "S2" else "total_points")
                ),
                "n_games": len(frame),
                "week_blocked_primary": week_blocked,
                "season_blocked_secondary": season_blocked,
                "per_season_stability": per_season_stability(
                    frame, value_col=value_col, arm_col="arm"
                ),
            }
        )
    coverage = frames["coverage_counts"]
    assert isinstance(coverage, dict)
    return results, coverage


def coverage_table(
    instances: pd.DataFrame, games: pd.DataFrame, scored: pd.DataFrame
) -> pd.DataFrame:
    rows = []
    for season in BOARD_SEASONS:
        inst_s = (
            instances.loc[instances["inferred_season"] == season] if len(instances) else instances
        )
        games_s = games.loc[games["inferred_season"] == season] if len(games) else games
        scored_s = scored.loc[scored["season"] == season] if len(scored) else scored
        rows.append(
            {
                "season": season,
                "board_instances_clean": len(inst_s),
                "matched_unique_games": int(games_s["game_key"].nunique()) if len(games_s) else 0,
                "scored_games_min3books": int(scored_s["game_id"].nunique())
                if len(scored_s)
                else 0,
            }
        )
    return pd.DataFrame(rows)


def score_all_cells(long_df: pd.DataFrame, samples: int, seed: int) -> list[dict[str, Any]]:
    specs: list[tuple[str, pd.Series, int, str]] = [
        (
            "vi_dispersion_top_tercile_underdog",
            long_df["flag_top_dog"],
            1,
            "Top-tercile cross-book spread SD, UNDERDOG side -- stale unresolved quotes "
            "overstate the favorite; +1 frozen before scoring",
        ),
        (
            "vi_dispersion_bottom_tercile_underdog",
            long_df["flag_bottom_dog"],
            1,
            "Bottom-tercile spread SD (sharp-consensus control), UNDERDOG side -- "
            "mechanism predicts near-null relative to the top tercile; +1 frozen",
        ),
        (
            "vi_dispersion_top_tercile_favorite",
            long_df["flag_top_fav"],
            -1,
            "Top-tercile spread SD, FAVORITE side -- opposite-side interaction implied "
            "by the same stale-premium mechanism; -1 frozen",
        ),
    ]
    results: list[dict[str, Any]] = []
    for name, flag, sign, description in specs:
        week_blocked = summarize(
            long_df,
            flag=flag.fillna(False).astype(bool),
            sign=sign,
            block_col="week_block",
            samples=samples,
            seed=seed,
        )
        season_blocked = summarize(
            long_df,
            flag=flag.fillna(False).astype(bool),
            sign=sign,
            block_col="season",
            samples=samples,
            seed=seed,
        )
        era_results: dict[str, Any] = {}
        for era_label, start, end in ERA_SPLITS:
            if name == "vi_dispersion_top_tercile_underdog":
                era_mask = long_df["season"].between(start, end)
                era_results[era_label] = summarize(
                    long_df.loc[era_mask].reset_index(drop=True),
                    flag=flag.loc[era_mask].reset_index(drop=True),
                    sign=sign,
                    block_col="week_block",
                    samples=samples,
                    seed=seed,
                )
        results.append(
            {
                "name": name,
                "sign_dir": sign,
                "description": description,
                "n_flag": int(flag.fillna(False).sum()),
                "week_blocked_primary": week_blocked,
                "season_blocked_secondary": season_blocked,
                "era_split": era_results,
            }
        )
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backfill", type=Path, default=DEFAULT_BACKFILL)
    parser.add_argument("--schedules", type=Path, default=DEFAULT_SCHEDULES)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--samples", type=int, default=BOOTSTRAP_SAMPLES)
    parser.add_argument("--seed", type=int, default=BOOTSTRAP_SEED)
    args = parser.parse_args()

    started = time.time()
    timestamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    output_dir: Path = args.output or (REPO / "artifacts" / "vi_dispersion" / timestamp)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== boards: {args.backfill} ===")
    instances = load_board_instances(args.backfill)
    print(
        f"T/W instances kept: {len(instances)} "
        f"(leak-guard dropped: {instances.attrs['leak_guard_dropped']}, "
        f"range-cap dropped: {instances.attrs['range_cap_dropped']})"
    )
    instances = join_schedule(instances, args.schedules)
    print(f"ambiguous matches dropped: {instances.attrs['ambiguous_matches']}")
    games = game_level_features(instances)
    print(f"unique matched games (earliest T/W capture): {len(games)}")

    long_df, q_low, q_high = build_scored_frame(games, args.schedules)
    scored_game_count = long_df["game_id"].nunique()
    print(f"scored games ({SEASON_START}-{SEASON_END}, >={MIN_BOOKS} books): {scored_game_count}")
    print(f"tercile cuts on scored spread_sd: low={q_low:.6f} high={q_high:.6f}")

    coverage = coverage_table(instances, games, long_df)
    print(coverage.to_string(index=False))

    reliability = split_half_reliability(games, args.seed)
    print(f"split-half reliability of spread_sd: {reliability}")
    total_reliability = split_half_reliability(
        games,
        args.seed,
        lines_col="total_lines",
        books_col="n_books_total",
    )
    print(f"split-half reliability of total_sd: {total_reliability}")

    cells = score_all_cells(long_df, samples=args.samples, seed=args.seed)
    for cell in cells:
        wb = cell["week_blocked_primary"]
        sb = cell["season_blocked_secondary"]
        print(f"\n=== {cell['name']} (sign {cell['sign_dir']:+d}) ===")
        if wb.get("insufficient_data"):
            print("  insufficient data")
            continue
        print(
            f"  n_flag={cell['n_flag']} subset={wb['subset_cover']:.4f} "
            f"complement={wb['complement_cover']:.4f} raw_gap={wb['raw_gap_pts']:+.3f}pts"
        )
        print(
            f"  full-slate effect {wb['full_slate_effect_pts']:+.4f}pts "
            f"week-blocked 95% [{wb['ci95_scaled'][0]:+.4f}, {wb['ci95_scaled'][1]:+.4f}] "
            f"P+={wb['probability_positive']:.4f}"
        )
        if not sb.get("insufficient_data"):
            print(
                f"  [season secondary] {sb['full_slate_effect_pts']:+.4f}pts "
                f"P+={sb['probability_positive']:.4f}"
            )
        for era_label, result in cell["era_split"].items():
            if result.get("insufficient_data"):
                print(f"  [{era_label}] insufficient data")
                continue
            print(
                f"  [{era_label}] n_flag={result['n_flag']} "
                f"{result['full_slate_effect_pts']:+.4f}pts P+={result['probability_positive']:.4f}"
            )

    frames = build_batch2_frames(games, args.schedules)
    coverage_counts = frames["coverage_counts"]
    assert isinstance(coverage_counts, dict)
    print("\n=== batch-2 coverage honesty (measured) ===")
    for key_, value_ in coverage_counts.items():
        print(f"  {key_}: {value_}")
    batch2_cells, coverage_counts = score_batch2(frames, samples=args.samples, seed=args.seed)
    for cell in batch2_cells:
        wb = cell["week_blocked_primary"]
        sb = cell["season_blocked_secondary"]
        print(f"\n=== {cell['name']} (direction +{cell['direction']}) ===")
        if wb.get("insufficient_data") or sb.get("insufficient_data"):
            print("  insufficient data")
            continue
        print(
            f"  n_games={cell['n_games']} (top {wb['n_top']} / bottom {wb['n_bottom']} / "
            f"middle excluded {wb['n_middle_excluded']}) mean_top={wb['mean_top']:.4f} "
            f"mean_bottom={wb['mean_bottom']:.4f}"
        )
        print(
            f"  effect {wb['effect']:+.4f} ({cell['value_units']}) "
            f"week-blocked 95% [{wb['ci95'][0]:+.4f}, {wb['ci95'][1]:+.4f}] "
            f"P+={wb['probability_positive']:.4f} blocks={wb['n_blocks']}"
        )
        print(
            f"  [season secondary] effect {sb['effect']:+.4f} "
            f"95% [{sb['ci95'][0]:+.4f}, {sb['ci95'][1]:+.4f}] "
            f"P+={sb['probability_positive']:.4f} blocks={sb['n_blocks']}"
        )
        stability_rows = []
        for row in cell["per_season_stability"]:
            gap = row["raw_gap"]
            stability_rows.append(
                f"{row['season']}:{gap:+.3f}" if gap is not None else f"{row['season']}:n/a"
            )
        print(f"  per-season gaps: {' '.join(stability_rows)}")

    configuration = {
        "command": "vi-dispersion-screen",
        "backfill_dir": str(args.backfill),
        "schedules": str(args.schedules),
        "bootstrap_samples": args.samples,
        "bootstrap_seed": args.seed,
        "board_seasons": list(BOARD_SEASONS),
        "scored_seasons": [SEASON_START, SEASON_END],
        "min_books": MIN_BOOKS,
        "range_cap": RANGE_CAP,
        "tuesday_wednesday_capture_weekdays_only": True,
        "reduced_confidence_excluded_seasons": sorted(EXCLUDED_REDUCED_CONFIDENCE_SEASONS),
    }
    payload = {
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(started)),
        "elapsed_seconds": time.time() - started,
        "bootstrap_samples": args.samples,
        "bootstrap_seed": args.seed,
        "n_cells": len(cells),
        "n_board_instances_clean": len(instances),
        "n_leak_guard_dropped": int(instances.attrs["leak_guard_dropped"]),
        "n_range_cap_dropped": int(instances.attrs["range_cap_dropped"]),
        "n_matched_unique_games": len(games),
        "n_scored_games": int(scored_game_count),
        "tercile_cut_low": q_low,
        "tercile_cut_high": q_high,
        "coverage_by_season": coverage.to_dict(orient="records"),
        "split_half_reliability": reliability,
        "split_half_reliability_total_sd": total_reliability,
        "batch2_coverage_counts": coverage_counts,
        "batch2_results": batch2_cells,
        "instrument_note": (
            "VI spread_line is the displayed favorite-side quote with no "
            "home/away orientation (97% negative; sign agreement vs nflverse "
            "close at base rates), so batch-1 underdog/favorite flags actually "
            "scored away/home-side contrasts. Batch-2 cells are "
            "orientation-free."
        ),
        "predeclaration": "docs/vi_dispersion_screen.md (frozen before scoring)",
        "results": cells,
        "provenance": artifact_provenance(configuration, args.schedules, project_root=REPO),
    }
    write_experiment_artifact(
        output_dir,
        "results.json",
        payload,
        command="vi-dispersion-screen",
        metrics=payload,
        notes=(
            "Measure-only lead-generation screen. Batch 1: 4 predeclared "
            "multi-book dispersion cells mined from one feature family and "
            "one window (tercile cut chosen after inspecting FEATURE "
            "distributions only). Batch 2 (2026-08-24): 3 further "
            "orientation-free cells predeclared in the doc before their "
            "outcomes were computed; direction of cell P is informed by "
            "batch-1 outputs -- same family/window, never pool as "
            "independent. Instrument note: VI spread_line carries no "
            "favorite-side orientation, so batch-1 dog/favorite labels are "
            "actually away/home contrasts. Every scoreable cell records via "
            "nfl-ats weak-signals record as unresolved_below_power "
            "regardless of interval shape (AGENTS.md taxonomy); season 2006 "
            "excluded entirely (reduced-confidence book-identity fallback "
            "0.64); seasons 2005-2008 unscorable (local schedules start "
            "2009)."
        ),
        source="scripts/vi_dispersion_screen.py",
        project_root=REPO,
    )
    print(f"\nwrote {output_dir / 'results.json'}")


if __name__ == "__main__":
    main()
