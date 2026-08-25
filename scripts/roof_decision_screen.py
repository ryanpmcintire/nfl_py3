"""Retractable-roof open/closed decision screen: a predeclared 5-cell mined
battery testing whether a retractable venue's roof status (open vs closed)
correlates with ATS/totals outcomes.

Source: ``docs/archive/data_source_scout_v4.md`` rank #6 ("Retractable-roof
open/closed decisions"), build-next rank 5 in that doc's "Top 5 for
immediate build" list. The `roof` field already exists in the project's
ingested nflverse schedule data; this script is the backtest half of that
rank's effort estimate (a live T-90 prospective scrape is a separate,
unbuilt follow-up).

**Measure-only.** This script never writes to ``registry/`` and never edits
a tracked doc. All 5 predeclared cells are scored and reported, including
the ones whose interval crosses zero -- per ``AGENTS.md``, an interval
containing zero is never a rejection.

The full predeclaration (population, exact cell definitions, mechanism,
direction, the untuned "benign weather" threshold) is frozen in
``<scratchpad>/roof_decision_screen/predeclaration.md`` and was written
before this script scored anything.

**Method**, reused from ``scripts/nfl_bias_battery_screen.py`` (team-game
long table, one row per side per game, week-blocked joint block bootstrap
on cover rate, subset-vs-whole-league-complement gap scaled by the fraction
of the slate the subset represents, so the recorded ``accuracy_points``
figure stays commensurable with every other registry entry). The bootstrap
helper is duplicated here (not imported) to keep this script self-contained,
per this project's mined-battery precedent of one script per battery.

**Opener-grade re-screen** for the two ATS cells (home-cover and
visiting-dome-team) uses
``nfl_ats.experiment_runner._opener_graded_features`` -- imported, not
modified, per this task's constraint against touching ``src/nfl_ats``. That
function restricts to the paired 2020-2025 Tuesday-opener/close archive and
overwrites ``spread_line``/``home_cover``/``ats_margin`` with the OPENER
line, with zero other changes needed since roof/stadium columns are simple
passthroughs.

Data:
- ``data/processed/game_features.parquet`` inner-joined on ``game_id`` with
  the newest ``data/raw/*/schedules.parquet`` snapshot for ``roof``/
  ``stadium`` (neither is in the feature table).
- ``data/raw/forecast_archive/full_2020_2025/forecasts.parquet`` for
  ``forecast_temp_f``/``forecast_wind_mph`` (Cell 4 only, 2020-2025) --
  the only local source with a pregame forecast at the stadium's own
  coordinates independent of roof state; ``schedules.parquet``'s own
  ``temp``/``wind`` fields are ~100% NaN for retractable-venue games
  regardless of roof state (measured).

Writes JSON to ``artifacts/roof_decision_screen/<UTC timestamp>/results.json``.
Proposes (does not execute) ``nfl-ats weak-signals record`` commands for
every cell at the end of stdout.
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
from nfl_ats.experiment_runner import _opener_graded_features  # noqa: E402
from nfl_ats.provenance import artifact_provenance, write_experiment_artifact  # noqa: E402

DEFAULT_FEATURES = REPO / "data/processed/game_features.parquet"
DEFAULT_FORECAST = REPO / "data/raw/forecast_archive/full_2020_2025/forecasts.parquet"

RETRACTABLE_TEAMS = frozenset({"ARI", "ATL", "DAL", "HOU", "IND"})
BENIGN_TEMP_MIN_F = 50.0
BENIGN_WIND_MAX_MPH = 15.0
ERA_SPLITS: tuple[tuple[str, int, int], ...] = (
    ("2009_2016", 2009, 2016),
    ("2017_2025", 2017, 2025),
)
BOOTSTRAP_SAMPLES = 20_000
BOOTSTRAP_SEED = 20260820


def _latest_schedules() -> Path:
    candidates = sorted((REPO / "data/raw").glob("*/schedules.parquet"))
    if not candidates:
        raise FileNotFoundError("no data/raw/*/schedules.parquet snapshot found")
    return candidates[-1]


def default_schedules() -> Path:
    """Resolve lazily so importing this module never requires local data."""
    return _latest_schedules()


def _canonical(team: pd.Series) -> pd.Series:
    return team.map(lambda code: TEAM_ABBREVIATION_ALIASES.get(code, code))


def load_merged(features_path: Path, schedules_path: Path) -> pd.DataFrame:
    features = pd.read_parquet(features_path)
    schedules = pd.read_parquet(schedules_path).loc[:, ["game_id", "roof", "stadium"]]
    merged = features.merge(schedules, on="game_id", how="inner", validate="one_to_one")
    merged = merged.loc[merged["game_type"] == "REG"].copy()
    merged["home_team"] = _canonical(merged["home_team"])
    merged["away_team"] = _canonical(merged["away_team"])
    merged["season"] = merged["season"].astype(int)
    merged["week"] = merged["week"].astype(int)
    return merged


def fixed_dome_team_seasons(schedules_path: Path) -> frozenset[tuple[str, int]]:
    """(team, season) pairs where every home-game roof value that season is
    exactly 'dome' (measured, data-driven -- not an assumed team list).
    """

    sched = pd.read_parquet(schedules_path)
    sched = sched.loc[sched["game_type"] == "REG"].copy()
    sched["home_team"] = _canonical(sched["home_team"])
    grouped = sched.groupby(["home_team", "season"])["roof"].apply(lambda s: frozenset(s.dropna()))
    pairs: list[tuple[str, int]] = []
    for key, roofs in grouped.items():
        team, season = key  # type: ignore[misc]
        if roofs == frozenset({"dome"}):
            pairs.append((str(team), int(season)))
    return frozenset(pairs)


_LONG_BASE_HOME = {"team": "home_team", "opponent": "away_team"}
_LONG_BASE_AWAY = {"team": "away_team", "opponent": "home_team"}


def build_long_table(
    merged: pd.DataFrame, dome_team_seasons: frozenset[tuple[str, int]]
) -> pd.DataFrame:
    """One row per (game, side); pushes (``team_covered`` NaN) dropped."""

    merged = merged.copy()
    merged["gameday"] = pd.to_datetime(merged["gameday"], errors="raise")

    sides = []
    for is_home, cols in ((True, _LONG_BASE_HOME), (False, _LONG_BASE_AWAY)):
        side = pd.DataFrame(
            {
                "game_id": merged["game_id"],
                "season": merged["season"],
                "week": merged["week"],
                "gameday": merged["gameday"],
                "team": merged[cols["team"]],
                "opponent": merged[cols["opponent"]],
                "is_home": is_home,
                "home_team": merged["home_team"],
                "roof": merged["roof"],
                "team_covered": merged["home_cover"] if is_home else 1.0 - merged["home_cover"],
            }
        )
        sides.append(side)

    long_df = pd.concat(sides, ignore_index=True)
    long_df = long_df.loc[long_df["team_covered"].notna()].copy()
    long_df["week_block"] = long_df["season"] * 100 + long_df["week"]

    long_df["retract_open"] = (
        long_df["is_home"]
        & long_df["home_team"].isin(RETRACTABLE_TEAMS)
        & (long_df["roof"] == "open")
    )
    long_df["retract_closed"] = (
        long_df["is_home"]
        & long_df["home_team"].isin(RETRACTABLE_TEAMS)
        & (long_df["roof"] == "closed")
    )
    team_season_pairs = list(zip(long_df["team"], long_df["season"], strict=True))
    is_dome_visitor = pd.Series(
        [pair in dome_team_seasons for pair in team_season_pairs], index=long_df.index
    )
    long_df["visiting_dome_open"] = (
        (~long_df["is_home"])
        & long_df["home_team"].isin(RETRACTABLE_TEAMS)
        & (long_df["roof"] == "open")
        & is_dome_visitor
    )
    return long_df.sort_values(["team", "season", "gameday"]).reset_index(drop=True)


def build_game_table(merged: pd.DataFrame) -> pd.DataFrame:
    """One row per game, for the totals-market (game-level) cell."""

    game = merged.copy()
    game["gameday"] = pd.to_datetime(game["gameday"], errors="raise")
    game["total_actual"] = pd.to_numeric(game["home_score"], errors="coerce") + pd.to_numeric(
        game["away_score"], errors="coerce"
    )
    game["total_line"] = pd.to_numeric(game["total_line"], errors="coerce")
    diff = game["total_actual"] - game["total_line"]
    game["total_covered"] = np.select([diff > 0, diff < 0], [1.0, 0.0], default=np.nan)
    game = game.loc[game["total_covered"].notna()].copy()
    game["week_block"] = game["season"] * 100 + game["week"]
    game["retract_open"] = game["home_team"].isin(RETRACTABLE_TEAMS) & (game["roof"] == "open")
    game["retract_closed"] = game["home_team"].isin(RETRACTABLE_TEAMS) & (game["roof"] == "closed")
    return game.reset_index(drop=True)


def merge_forecast(long_df: pd.DataFrame, forecast_path: Path) -> pd.DataFrame:
    forecast = pd.read_parquet(forecast_path).loc[
        :, ["game_id", "forecast_temp_f", "forecast_wind_mph"]
    ]
    out = long_df.merge(forecast, on="game_id", how="left")
    benign = (out["forecast_temp_f"] >= BENIGN_TEMP_MIN_F) & (
        out["forecast_wind_mph"] <= BENIGN_WIND_MAX_MPH
    )
    out["benign_forecast_closed"] = out["retract_closed"] & benign
    out["non_benign_forecast_open"] = (
        out["retract_open"] & (~benign) & out["forecast_temp_f"].notna()
    )
    return out


def block_bootstrap_two_group(
    df: pd.DataFrame,
    *,
    flag_col: str,
    value_col: str,
    block_col: str,
    samples: int,
    seed: int,
) -> np.ndarray:
    """Vectorized week-blocked bootstrap of ``100*(subset_mean-complement_mean)``.

    Reused verbatim (method, not import) from
    ``scripts/nfl_bias_battery_screen.py::block_bootstrap_two_group``.
    """

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


def summarize_population(
    df: pd.DataFrame, *, flag: pd.Series, value_col: str, sign: int, samples: int, seed: int
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
    subset_cover = float(work.loc[work["_flag"], value_col].mean())
    complement_cover = float(work.loc[~work["_flag"], value_col].mean())
    raw_gap_pts = sign * (subset_cover - complement_cover) * 100.0
    fraction_of_slate = n_flag / n_total
    full_slate_effect_pts = raw_gap_pts * fraction_of_slate

    draws = block_bootstrap_two_group(
        work,
        flag_col="_flag",
        value_col=value_col,
        block_col="week_block",
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
        "n_week_blocks": int(work["week_block"].nunique()),
        "subset_cover": subset_cover,
        "complement_cover": complement_cover,
        "raw_gap_pts": raw_gap_pts,
        "fraction_of_slate": fraction_of_slate,
        "full_slate_effect_pts": full_slate_effect_pts,
        "week_blocked_ci95_scaled": [float(lower), float(upper)],
        "probability_positive": float(np.mean(signed_draws > 0)) if len(signed_draws) else np.nan,
        "bootstrap_samples": samples,
        "dropped_draws": int(dropped),
        "insufficient_data": False,
    }


def score_cell(
    population: pd.DataFrame,
    *,
    name: str,
    flag_col: str,
    value_col: str,
    sign: int,
    mechanism: str,
    era_splits: tuple[tuple[str, int, int], ...],
    samples: int,
    seed: int,
) -> dict[str, Any]:
    flag = population[flag_col]
    full = summarize_population(
        population, flag=flag, value_col=value_col, sign=sign, samples=samples, seed=seed
    )
    era_results = {}
    for era_label, start, end in era_splits:
        era_mask = population["season"].between(start, end)
        era_pop = population.loc[era_mask].reset_index(drop=True)
        era_flag = flag.loc[era_mask].reset_index(drop=True)
        era_results[era_label] = summarize_population(
            era_pop, flag=era_flag, value_col=value_col, sign=sign, samples=samples, seed=seed
        )
    return {
        "name": name,
        "mechanism": mechanism,
        "sign_dir": sign,
        "full_period": full,
        "era_split": era_results,
    }


def _print_cell(cell: dict[str, Any]) -> None:
    print(f"\n=== {cell['name']} ===")
    full = cell["full_period"]
    if full.get("insufficient_data"):
        print(f"  insufficient data: n_flag={full['n_flag']} n_complement={full['n_complement']}")
        return
    print(
        f"  n_flag={full['n_flag']} n_total={full['n_total']} "
        f"subset_cover={full['subset_cover']:.4f} complement_cover={full['complement_cover']:.4f}"
    )
    print(
        f"  raw_gap={full['raw_gap_pts']:+.3f}pts frac_of_slate={full['fraction_of_slate']:.4f} "
        f"full_slate_effect={full['full_slate_effect_pts']:+.4f}pts"
    )
    print(
        f"  week-blocked 95% scaled [{full['week_blocked_ci95_scaled'][0]:+.4f}, "
        f"{full['week_blocked_ci95_scaled'][1]:+.4f}] P+={full['probability_positive']:.4f}"
    )
    for era_label, era in cell["era_split"].items():
        if era.get("insufficient_data"):
            print(f"  [{era_label}] insufficient data n_flag={era['n_flag']}")
            continue
        print(
            f"  [{era_label}] n_flag={era['n_flag']} cover={era['subset_cover']:.4f} "
            f"vs {era['complement_cover']:.4f} "
            f"full_slate_effect={era['full_slate_effect_pts']:+.4f}pts "
            f"P+={era['probability_positive']:.4f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, default=DEFAULT_FEATURES)
    parser.add_argument("--schedules", type=Path, default=None)
    parser.add_argument("--forecast", type=Path, default=DEFAULT_FORECAST)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--samples", type=int, default=BOOTSTRAP_SAMPLES)
    parser.add_argument("--seed", type=int, default=BOOTSTRAP_SEED)
    args = parser.parse_args()
    if args.schedules is None:
        args.schedules = default_schedules()

    started = time.time()
    timestamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    output_dir: Path = args.output or (REPO / "artifacts" / "roof_decision_screen" / timestamp)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== loading {args.features} + {args.schedules} ===")
    merged = load_merged(args.features, args.schedules)
    print(f"REG games (close grade): {len(merged)}")

    dome_seasons = fixed_dome_team_seasons(args.schedules)
    print(f"fixed-dome (team,season) pairs: {len(dome_seasons)}")

    long_df = build_long_table(merged, dome_seasons)
    print(f"team-game rows (pushes dropped): {len(long_df)}")
    game_df = build_game_table(merged)
    print(f"game-level rows (totals pushes dropped): {len(game_df)}")

    results: list[dict[str, Any]] = []

    cell1 = score_cell(
        long_df,
        name="roof_battery_home_cover_open_vs_closed",
        flag_col="retract_open",
        value_col="team_covered",
        sign=-1,
        mechanism=(
            "Open roof strips the home team's climate-controlled identity; predicts home cover "
            "rate LOWER when open vs closed at a retractable venue."
        ),
        era_splits=ERA_SPLITS,
        samples=args.samples,
        seed=args.seed,
    )
    results.append(cell1)
    _print_cell(cell1)

    cell2 = score_cell(
        long_df,
        name="roof_battery_visiting_dome_open_vs_closed",
        flag_col="visiting_dome_open",
        value_col="team_covered",
        sign=-1,
        mechanism=(
            "A fixed-dome visiting team is hypothesized more disrupted by genuine outdoor "
            "conditions than an already-acclimated outdoor team; predicts the VISITING dome "
            "team's own cover rate LOWER when the retractable venue's roof is open."
        ),
        era_splits=ERA_SPLITS,
        samples=args.samples,
        seed=args.seed,
    )
    results.append(cell2)
    _print_cell(cell2)

    cell3 = score_cell(
        game_df,
        name="roof_battery_total_covered_open_vs_closed",
        flag_col="retract_open",
        value_col="total_covered",
        sign=-1,
        mechanism=(
            "TOTALS market, not ATS. Open roof exposes the game to real wind/cold/precip a "
            "total line set for a controlled environment may not price; predicts the over-rate "
            "LOWER (unders cover more) when open."
        ),
        era_splits=ERA_SPLITS,
        samples=args.samples,
        seed=args.seed,
    )
    results.append(cell3)
    _print_cell(cell3)

    print(f"\n=== loading {args.forecast} for cell 4 (2020-2025 only) ===")
    long_df_fc = merge_forecast(
        long_df.loc[long_df["season"].between(2020, 2025)].copy(), args.forecast
    )
    cell4 = score_cell(
        long_df_fc,
        name="roof_battery_closed_benign_forecast_vs_open",
        flag_col="benign_forecast_closed",
        value_col="team_covered",
        sign=1,
        mechanism=(
            "No predeclared directional hypothesis (exploratory). Subset = closed roof despite "
            "a benign NDFD/GFS-MOS forecast (>=50F, <=15mph, thresholds fixed before scoring); "
            "complement = everyone else in the league. sign=+1, raw gap reported as-is."
        ),
        era_splits=(),
        samples=args.samples,
        seed=args.seed,
    )
    results.append(cell4)
    _print_cell(cell4)
    non_benign_open_count = int(long_df_fc["non_benign_forecast_open"].sum())
    print(
        "  side count (anecdotal, not bootstrapped): open despite non-benign forecast = "
        f"{non_benign_open_count}"
    )

    print("\n=== opener-grade re-screen (2020-2025 Tuesday-open/close-paired archive) ===")
    opener_merged, opener_note = _opener_graded_features(merged, repo_root=REPO, market_root=None)
    print(f"  {opener_note}")
    opener_long = build_long_table(opener_merged, dome_seasons)
    print(f"  opener-grade team-game rows: {len(opener_long)}")

    cell1_opener = score_cell(
        opener_long,
        name="roof_battery_home_cover_open_vs_closed_opener",
        flag_col="retract_open",
        value_col="team_covered",
        sign=-1,
        mechanism=cell1["mechanism"] + " OPENER grade (2020-2025 archive intersection).",
        era_splits=(),
        samples=args.samples,
        seed=args.seed,
    )
    results.append(cell1_opener)
    _print_cell(cell1_opener)

    cell2_opener = score_cell(
        opener_long,
        name="roof_battery_visiting_dome_open_vs_closed_opener",
        flag_col="visiting_dome_open",
        value_col="team_covered",
        sign=-1,
        mechanism=cell2["mechanism"] + " OPENER grade (2020-2025 archive intersection).",
        era_splits=(),
        samples=args.samples,
        seed=args.seed,
    )
    results.append(cell2_opener)
    _print_cell(cell2_opener)

    configuration = {
        "command": "roof-decision-screen",
        "features": str(args.features),
        "schedules": str(args.schedules),
        "forecast": str(args.forecast),
        "bootstrap_samples": args.samples,
        "bootstrap_seed": args.seed,
        "retractable_teams": sorted(RETRACTABLE_TEAMS),
        "benign_temp_min_f": BENIGN_TEMP_MIN_F,
        "benign_wind_max_mph": BENIGN_WIND_MAX_MPH,
        "era_splits": ERA_SPLITS,
    }
    payload = {
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(started)),
        "elapsed_seconds": time.time() - started,
        "bootstrap_samples": args.samples,
        "bootstrap_seed": args.seed,
        "n_reg_games_close_grade": len(merged),
        "n_team_game_rows_close_grade": len(long_df),
        "n_game_rows_totals": len(game_df),
        "non_benign_forecast_open_count": non_benign_open_count,
        "opener_population_note": opener_note,
        "retractable_teams": sorted(RETRACTABLE_TEAMS),
        "benign_temp_min_f": BENIGN_TEMP_MIN_F,
        "benign_wind_max_mph": BENIGN_WIND_MAX_MPH,
        "predeclaration": (
            "<scratchpad>/roof_decision_screen/predeclaration.md (frozen before scoring)"
        ),
        "results": results,
        "provenance": artifact_provenance(configuration, args.features, project_root=REPO),
    }
    write_experiment_artifact(
        output_dir,
        "results.json",
        payload,
        command="roof-decision-screen",
        metrics=payload,
        notes=(
            "Measure-only battery, 5 predeclared mined cells (home-cover, visiting-dome-team, "
            "totals, benign-forecast-closed, plus a 2020-2025 opener-grade re-screen of the "
            "two ATS cells); every cell scored and reported regardless of interval shape "
            "(AGENTS.md). Never writes to registry/ or a tracked doc itself -- proposes "
            "nfl-ats weak-signals record commands on stdout."
        ),
    )
    print(f"\nwrote {output_dir / 'results.json'}")


if __name__ == "__main__":
    main()
