"""Within-season divisional REMATCH dynamics screen: 5 predeclared cells on
REG 2009-2025 NFL team-games, week-blocked bootstrap primary, season-blocked
secondary, full-slate-scaled accuracy_points effects, plus one descriptive
(no ATS claim) total/margin regression readout.

Strictly distinct from ``bias_battery_division_revenge_game`` /
``division_revenge_tilt_overlay`` (unsplit revenge side, no venue/margin/
timing conditioning): this screen scores the blowout-winner fade in the
rematch, the revenge side split by the first meeting's venue, and the
revenge side split by rematch timing. Cell C is descriptive only.
Predeclaration frozen in ``docs/divisional_rematch_screen.md`` before any
cover rate was computed. Measure-only: never writes registry JSON; stamps a
run log to ``registry/experiments/divisional-rematch-screen/``.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from nfl_ats.constants import TEAM_ABBREVIATION_ALIASES  # noqa: E402
from nfl_ats.features import add_ats_outcomes  # noqa: E402
from nfl_ats.provenance import artifact_provenance, write_experiment_artifact  # noqa: E402

BOOTSTRAP_SAMPLES = 20_000
BOOTSTRAP_SEED = 20260821
SEASON_START = 2009
SEASON_END = 2025
BLOWOUT_MARGIN = 14
EARLY_WEEK_MAX = 6
LATE_WEEK_MIN = 12
ERA_SPLITS: tuple[tuple[str, int, int], ...] = (
    ("2009_2017", 2009, 2017),
    ("2018_2025", 2018, 2025),
)

SCHEDULE_COLUMNS = [
    "game_id",
    "season",
    "week",
    "game_type",
    "gameday",
    "home_team",
    "away_team",
    "result",
    "total",
    "spread_line",
    "div_game",
]


def _latest_schedules() -> Path:
    candidates = sorted((REPO / "data/raw").glob("*/schedules.parquet"))
    if not candidates:
        raise FileNotFoundError("no data/raw/*/schedules.parquet snapshot found")
    return candidates[-1]


def default_schedules() -> Path:
    """Resolve lazily so importing this module never requires local data."""
    return _latest_schedules()


def load_population(schedules_path: Path) -> pd.DataFrame:
    available = [c for c in SCHEDULE_COLUMNS if c in pd.read_parquet(schedules_path).columns]
    df = pd.read_parquet(schedules_path, columns=available)
    df = df.loc[df["game_type"] == "REG"].copy()
    df["season"] = pd.to_numeric(df["season"], errors="raise").astype(int)
    df["week"] = pd.to_numeric(df["week"], errors="raise").astype(int)
    df = df.loc[df["season"].between(SEASON_START, SEASON_END)].reset_index(drop=True)
    df = add_ats_outcomes(df)
    n_before_push_drop = len(df)
    df = df.loc[df["home_cover"].notna()].reset_index(drop=True)
    for column in ("home_team", "away_team"):
        df[column] = df[column].replace(TEAM_ABBREVIATION_ALIASES)
    df["gameday"] = pd.to_datetime(df["gameday"], errors="raise")
    df.attrs["n_before_push_drop"] = n_before_push_drop
    df.attrs["pushes_or_missing"] = n_before_push_drop - len(df)
    return df


def build_long_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for is_home in (True, False):
        side = pd.DataFrame(
            {
                "game_id": df["game_id"],
                "season": df["season"],
                "week": df["week"],
                "gameday": df["gameday"],
                "team": df["home_team"] if is_home else df["away_team"],
                "opponent": df["away_team"] if is_home else df["home_team"],
                "is_home": is_home,
                "team_covered": df["home_cover"] if is_home else 1.0 - df["home_cover"],
                "score_margin": df["result"] if is_home else -df["result"],
                "total": df["total"],
                "div_game": df["div_game"],
            }
        )
        rows.append(side)
    long_df = pd.concat(rows, ignore_index=True)
    long_df = long_df.sort_values(["team", "opponent", "season", "gameday"]).reset_index(drop=True)
    grouped = long_df.groupby(["team", "opponent", "season"], sort=False)
    long_df["meeting_rank"] = grouped.cumcount()
    first = grouped[["score_margin", "is_home", "week", "total"]].first()
    first.columns = ["first_margin", "first_is_home", "first_week", "first_total"]
    long_df = long_df.merge(first.reset_index(), on=["team", "opponent", "season"], how="left")
    long_df["week_block"] = long_df["season"] * 100 + long_df["week"]
    return long_df


def build_cells(long_df: pd.DataFrame) -> dict[str, dict[str, Any]]:
    rematch = long_df["meeting_rank"] >= 1
    revenge = rematch & (long_df["first_margin"] < 0)
    blowout_winner = rematch & (long_df["first_margin"] >= BLOWOUT_MARGIN)

    specs: list[tuple[str, pd.Series, int, str]] = [
        (
            "rematch_blowout_winner_fade",
            blowout_winner,
            -1,
            f"Rematch side that WON the first meeting by >= {BLOWOUT_MARGIN} raw points "
            "(film + motivation reversal: the loser game-plans the fix, the winner "
            "coasts) -- predicted NEGATIVE team-cover edge (fade the blowout winner "
            "in the rematch)",
        ),
        (
            "rematch_revenge_road_loser",
            revenge & (~long_df["first_is_home"]),
            1,
            "Revenge side (lost the first meeting this season) whose game-1 loss came "
            "ON THE ROAD -- predicted positive team-cover edge (revenge spot with a "
            "venue flip coming home is the classic construction)",
        ),
        (
            "rematch_revenge_home_loser",
            revenge & (long_df["first_is_home"]),
            1,
            "Revenge side (lost the first meeting this season) whose game-1 loss came "
            "AT HOME -- predicted positive team-cover edge (weaker form of the "
            "revenge spot: no venue flip to lean on)",
        ),
        (
            "rematch_revenge_early_w1to6",
            revenge & (long_df["week"] <= EARLY_WEEK_MAX),
            1,
            f"Revenge side in an EARLY-season rematch (rematch week <= {EARLY_WEEK_MAX}) "
            "-- predicted positive team-cover edge (grudge is fresh, first-meeting "
            "film still current)",
        ),
        (
            "rematch_revenge_late_w12plus",
            revenge & (long_df["week"] >= LATE_WEEK_MIN),
            1,
            f"Revenge side in a LATE-season rematch (rematch week >= {LATE_WEEK_MIN}) "
            "-- predicted positive team-cover edge (playoff-seeding stakes amplify "
            "the revenge spot)",
        ),
    ]

    cells: dict[str, dict[str, Any]] = {}
    for name, flag, sign, description in specs:
        cells[name] = {
            "flag": flag.fillna(False).astype(bool),
            "sign": sign,
            "description": f"{description} (pregame-safe schedule facts only, no leakage caveat).",
        }
    expected = 5
    assert len(cells) == expected, f"expected {expected} predeclared cells, got {len(cells)}"
    return cells


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
    dropped = samples - len(draws)
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
        "dropped_draws": int(dropped),
        "insufficient_data": False,
    }


def score_cell(
    long_df: pd.DataFrame, name: str, spec: dict[str, Any], *, samples: int, seed: int
) -> dict[str, Any]:
    population = long_df
    flag = spec["flag"]
    week_blocked = summarize(
        population, flag=flag, sign=spec["sign"], block_col="week_block", samples=samples, seed=seed
    )
    season_blocked = summarize(
        population, flag=flag, sign=spec["sign"], block_col="season", samples=samples, seed=seed
    )
    era_results = {}
    for era_label, start, end in ERA_SPLITS:
        era_mask = population["season"].between(start, end)
        era_results[era_label] = summarize(
            population.loc[era_mask].reset_index(drop=True),
            flag=flag.loc[era_mask].reset_index(drop=True),
            sign=spec["sign"],
            block_col="week_block",
            samples=samples,
            seed=seed,
        )
    return {
        "name": name,
        "sign_dir": spec["sign"],
        "description": spec["description"],
        "n_flag": int(flag.sum()),
        "week_blocked": week_blocked,
        "season_blocked_secondary": season_blocked,
        "era_split": era_results,
    }


def descriptive_regression(schedule_df: pd.DataFrame) -> dict[str, Any]:
    df = schedule_df.copy()
    df["pair"] = df.apply(lambda r: tuple(sorted((r["home_team"], r["away_team"]))), axis=1)
    df = df.sort_values(["season", "pair", "gameday"]).reset_index(drop=True)
    df["pair_rank"] = df.groupby(["season", "pair"]).cumcount()
    firsts = (
        df.loc[df["pair_rank"] == 0, ["season", "pair", "total", "result"]]
        .drop_duplicates(["season", "pair"])
        .rename(columns={"total": "first_total", "result": "first_result"})
    )
    rematches = df.loc[df["pair_rank"] >= 1].merge(firsts, on=["season", "pair"], how="inner")
    rematches["first_abs_margin"] = rematches["first_result"].abs()
    rematches = rematches.loc[
        rematches[["total", "result", "first_total", "first_abs_margin"]].notna().all(axis=1)
    ]

    def _fit(x: np.ndarray, y: np.ndarray) -> dict[str, float]:
        slope, intercept = np.polyfit(x, y, 1)
        r = float(np.corrcoef(x, y)[0, 1])
        return {
            "slope": float(slope),
            "intercept": float(intercept),
            "pearson_r": r,
            "r_squared": r * r,
        }

    x_margin = rematches["first_abs_margin"].to_numpy(dtype=np.float64)
    y_margin = rematches["result"].abs().to_numpy(dtype=np.float64)
    x_total = rematches["first_total"].to_numpy(dtype=np.float64)
    y_total = rematches["total"].to_numpy(dtype=np.float64)
    return {
        "n_rematch_games": len(rematches),
        "margin_regression_first_abs_margin_to_rematch_abs_margin": _fit(x_margin, y_margin),
        "total_regression_first_total_to_rematch_total": _fit(x_total, y_total),
        "mean_first_abs_margin": float(x_margin.mean()),
        "mean_rematch_abs_margin": float(y_margin.mean()),
        "mean_first_total": float(x_total.mean()),
        "mean_rematch_total": float(y_total.mean()),
        "no_ats_claim": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--schedules", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--samples", type=int, default=BOOTSTRAP_SAMPLES)
    parser.add_argument("--seed", type=int, default=BOOTSTRAP_SEED)
    parser.add_argument("--counts-only", action="store_true")
    args = parser.parse_args()
    if args.schedules is None:
        args.schedules = default_schedules()

    started = time.time()
    timestamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    output_dir: Path = args.output or (REPO / "artifacts" / "divisional_rematch_screen" / timestamp)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== loading {args.schedules} ===")
    df = load_population(args.schedules)
    print(
        f"REG {SEASON_START}-{SEASON_END} games: {df.attrs['n_before_push_drop']}, "
        f"pushes/missing dropped: {df.attrs['pushes_or_missing']}"
    )
    long_df = build_long_table(df)
    print(f"team-game rows (pushes dropped): {len(long_df)}")

    rematch_rows = long_df.loc[long_df["meeting_rank"] >= 1]
    rematch_games = rematch_rows["game_id"].nunique()
    non_div_rematches = int((rematch_rows["div_game"] != 1).sum())
    tied_first_meetings = int(
        ((long_df["meeting_rank"] >= 1) & (long_df["first_margin"] == 0)).sum()
    )
    print(f"rematch team-game rows: {len(rematch_rows)} across {rematch_games} games")
    print(f"rematch games with div_game != 1: {non_div_rematches}")
    print(f"rematch team-game rows after a tied first meeting: {tied_first_meetings}")

    cells = build_cells(long_df)
    counts = {name: int(spec["flag"].sum()) for name, spec in cells.items()}
    for name, count in counts.items():
        print(f"  {name}: n_flag={count}")

    if args.counts_only:
        return

    results = []
    for name, spec in cells.items():
        print(f"\n=== {name} ===")
        cell = score_cell(long_df, name, spec, samples=args.samples, seed=args.seed)
        results.append(cell)
        wb = cell["week_blocked"]
        sb = cell["season_blocked_secondary"]
        if wb.get("insufficient_data"):
            print("  insufficient data (empty subset or complement)")
            continue
        print(
            f"  n_flag={cell['n_flag']} n_total={wb['n_total']} "
            f"subset_cover={wb['subset_cover']:.4f} complement_cover={wb['complement_cover']:.4f}"
        )
        print(
            f"  raw_gap={wb['raw_gap_pts']:+.3f}pts frac_of_slate={wb['fraction_of_slate']:.4f} "
            f"full_slate_effect={wb['full_slate_effect_pts']:+.4f}pts"
        )
        print(
            f"  week-blocked 95% [{wb['ci95_scaled'][0]:+.4f}, {wb['ci95_scaled'][1]:+.4f}] "
            f"P+={wb['probability_positive']:.4f} n_week_blocks={wb['n_blocks']}"
        )
        if not sb.get("insufficient_data"):
            print(
                f"  [season-blocked secondary] 95% [{sb['ci95_scaled'][0]:+.4f}, "
                f"{sb['ci95_scaled'][1]:+.4f}] P+={sb['probability_positive']:.4f} "
                f"n_seasons={sb['n_blocks']}"
            )
        for era_label, _, _ in ERA_SPLITS:
            era = cell["era_split"][era_label]
            if era.get("insufficient_data"):
                print(f"  [{era_label}] insufficient data")
                continue
            print(
                f"  [{era_label}] n_flag={era['n_flag']} "
                f"full_slate_effect={era['full_slate_effect_pts']:+.4f}pts "
                f"P+={era['probability_positive']:.4f}"
            )

    print("\n=== cell C: descriptive regression (no ATS claim) ===")
    regression = descriptive_regression(df)
    print(
        f"  n_rematch_games={regression['n_rematch_games']} "
        f"margin~first_margin slope="
        f"{regression['margin_regression_first_abs_margin_to_rematch_abs_margin']['slope']:.4f} "
        f"r={regression['margin_regression_first_abs_margin_to_rematch_abs_margin']['pearson_r']:.4f}"
    )
    print(
        f"  total~first_total slope="
        f"{regression['total_regression_first_total_to_rematch_total']['slope']:.4f} "
        f"r={regression['total_regression_first_total_to_rematch_total']['pearson_r']:.4f}"
    )

    early = next(c for c in results if c["name"] == "rematch_revenge_early_w1to6")["week_blocked"]
    late = next(c for c in results if c["name"] == "rematch_revenge_late_w12plus")["week_blocked"]
    timing_contrast = {
        "early_raw_gap_pts": None if early.get("insufficient_data") else early["raw_gap_pts"],
        "late_raw_gap_pts": None if late.get("insufficient_data") else late["raw_gap_pts"],
        "descriptive_only": True,
    }

    configuration = {
        "command": "divisional-rematch-screen",
        "schedules": str(args.schedules),
        "bootstrap_samples": args.samples,
        "bootstrap_seed": args.seed,
        "season_start": SEASON_START,
        "season_end": SEASON_END,
        "blowout_margin": BLOWOUT_MARGIN,
        "early_week_max": EARLY_WEEK_MAX,
        "late_week_min": LATE_WEEK_MIN,
    }
    payload = {
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(started)),
        "elapsed_seconds": time.time() - started,
        "bootstrap_samples": args.samples,
        "bootstrap_seed": args.seed,
        "n_cells": len(cells),
        "season_start": SEASON_START,
        "season_end": SEASON_END,
        "n_reg_games_before_push_drop": df.attrs["n_before_push_drop"],
        "n_pushes_or_missing_dropped": df.attrs["pushes_or_missing"],
        "n_team_game_rows": len(long_df),
        "n_rematch_team_game_rows": len(rematch_rows),
        "n_rematch_games": rematch_games,
        "n_rematch_games_div_game_not_1": non_div_rematches,
        "n_rematch_rows_after_tied_first_meeting": tied_first_meetings,
        "cell_counts_predeclared": counts,
        "predeclaration": "docs/divisional_rematch_screen.md (frozen before scoring)",
        "results": results,
        "descriptive_regression_no_ats_claim": regression,
        "revenge_timing_contrast_descriptive": timing_contrast,
        "provenance": artifact_provenance(configuration, args.schedules, project_root=REPO),
    }
    write_experiment_artifact(
        output_dir,
        "results.json",
        payload,
        command="divisional-rematch-screen",
        metrics=payload,
        notes=(
            "Measure-only lead-generation screen (5 predeclared within-season "
            "divisional-rematch cells + descriptive total/margin regression with no "
            "ATS claim); every cell predeclared to record unresolved_below_power via "
            "nfl-ats weak-signals record regardless of interval shape (AGENTS.md). "
            "Overlap: the unsplit revenge construct is already recorded "
            "(bias_battery_division_revenge_game/_opener, wired as "
            "division_revenge_tilt_overlay) and is deliberately NOT re-recorded here; "
            "the venue/timing splits and the blowout-winner fade are distinct cells, "
            "correlated with the parent construct -- never pool them as independent."
        ),
    )
    print(f"\nwrote {output_dir / 'results.json'}")


if __name__ == "__main__":
    main()
