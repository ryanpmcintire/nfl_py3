"""Lead-generation attention bias battery (PILOT): 6 predeclared checks of a
brand-new, free, point-in-time-safe instrument -- Wikipedia pageviews as a
public-attention proxy -- screened against the spread on NFL REG 2016-2025.

**Pilot / measure-only.** This script never writes to the weak-signal
registry (``registry/weak_signals.json``); recording a finding there happens
via a separate, explicit ``nfl-ats weak-signals record`` invocation against
this script's output. It DOES write an automatic, low-stakes
experiment-provenance stamp to ``registry/experiments/`` via
``write_experiment_artifact`` (RWB-09) -- a run log, not a verdict; it
carries no closing-ground and asserts nothing about any cell's status. Per
AGENTS.md, an interval containing zero is never a rejection -- this
battery's purpose is to surface category-3 leads, not to prove any one of
them, and every cell here is predeclared to record
``unresolved_below_power`` regardless of shape (mined, uncorrected
multiplicity, 6 cells; precedent: ``cfb_bias_battery_rivalry_finale_proxy``
stayed unresolved even fully resolved-shaped, for exactly this reason).

The full predeclaration (attention window rule, normalization, exact cell
definitions, hypothesized mechanism + sign per cell) is frozen in
``<scratchpad>/agent_attention/predeclaration.md`` and was written before this
script scored anything. This module implements exactly what that document
specifies.

**Data sources**:

- Raw Wikimedia pageviews JSON + fetch manifest:
  ``<scratchpad>/agent_attention/raw/*.json`` and ``fetch_manifest.json``
  (fetched by ``<scratchpad>/agent_attention/fetch_pageviews.py``, daily
  granularity, 2015-07-01 through 2026-01-01, 32 teams incl. renamed-franchise
  predecessor articles summed per date).
- Newest ``data/raw/*/schedules.parquet`` snapshot for games, teams,
  ``spread_line``, ``result``. REG season only, 2016-2025 (2015 excluded: the
  pageviews API starts 2015-07, so 2015 would have an unstable partial-season
  baseline).

**Method**, reused from ``scripts/nfl_bias_battery_screen.py`` /
``scripts/nfl_weather_battery_screen.py``: the same
``block_bootstrap_two_group`` joint week-blocked bootstrap generalized to an
arbitrary boolean subset flag vs. its complement, the same full-slate effect
scaling (raw gap x fraction-of-eligible-population), and the same
``probability_positive`` definition. A season-blocked secondary bootstrap is
also reported (robustness check, console/notes only, not the registry
interval).

Cover computation reuses ``nfl_ats.features.add_ats_outcomes`` verbatim
(``ats_margin = result - spread_line``; ``home_cover`` = 1.0/0.0/NaN-on-push).

Writes JSON to ``artifacts/attention_battery/<UTC timestamp>/results.json``
and prints a summary table (including the reliability check) to stdout.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

sys.path.append(str(REPO / "scripts"))

from _common import block_bootstrap_two_group, latest_schedules  # noqa: E402

from nfl_ats.constants import TEAM_ABBREVIATION_ALIASES  # noqa: E402
from nfl_ats.features import add_ats_outcomes  # noqa: E402
from nfl_ats.provenance import artifact_provenance, write_experiment_artifact  # noqa: E402

DEFAULT_SCRATCH = Path(
    "C:/Users/Ryan/AppData/Local/Temp/claude/F--Repos-nfl-py3/"
    "26042060-ffd8-45a7-b2e7-a9b30b87bd34/scratchpad/agent_attention"
)

TEAM_ARTICLES: dict[str, list[str]] = {
    "ARI": ["Arizona_Cardinals"],
    "ATL": ["Atlanta_Falcons"],
    "BAL": ["Baltimore_Ravens"],
    "BUF": ["Buffalo_Bills"],
    "CAR": ["Carolina_Panthers"],
    "CHI": ["Chicago_Bears"],
    "CIN": ["Cincinnati_Bengals"],
    "CLE": ["Cleveland_Browns"],
    "DAL": ["Dallas_Cowboys"],
    "DEN": ["Denver_Broncos"],
    "DET": ["Detroit_Lions"],
    "GB": ["Green_Bay_Packers"],
    "HOU": ["Houston_Texans"],
    "IND": ["Indianapolis_Colts"],
    "JAX": ["Jacksonville_Jaguars"],
    "KC": ["Kansas_City_Chiefs"],
    "LA": ["St._Louis_Rams", "Los_Angeles_Rams"],
    "LAC": ["San_Diego_Chargers", "Los_Angeles_Chargers"],
    "LV": ["Oakland_Raiders", "Las_Vegas_Raiders"],
    "MIA": ["Miami_Dolphins"],
    "MIN": ["Minnesota_Vikings"],
    "NE": ["New_England_Patriots"],
    "NO": ["New_Orleans_Saints"],
    "NYG": ["New_York_Giants"],
    "NYJ": ["New_York_Jets"],
    "PHI": ["Philadelphia_Eagles"],
    "PIT": ["Pittsburgh_Steelers"],
    "SEA": ["Seattle_Seahawks"],
    "SF": ["San_Francisco_49ers"],
    "TB": ["Tampa_Bay_Buccaneers"],
    "TEN": ["Tennessee_Titans"],
    "WAS": ["Washington_Redskins", "Washington_Football_Team", "Washington_Commanders"],
}

BOOTSTRAP_SAMPLES = 20_000
BOOTSTRAP_SEED = 20260819
SEASON_START = 2016
SEASON_END = 2025
TRAILING_WINDOW_GAMES = 8
TRAILING_MIN_GAMES = 2

DESCRIPTION_SUFFIX = (
    " mined pilot battery, uncorrected multiplicity; Wikipedia pageview "
    "attention proxy, window ends Tuesday of game week (point-in-time safe)."
)


def _canonical(team: pd.Series) -> pd.Series:
    return team.map(lambda code: TEAM_ABBREVIATION_ALIASES.get(code, code))


# --------------------------------------------------------------------------
# Pageview loading
# --------------------------------------------------------------------------


def load_team_daily_views(scratch: Path) -> dict[str, pd.Series]:
    """Return {canonical_team_code: pd.Series indexed by date (daily views,
    summed across alias articles, 0 where missing)}."""

    raw_dir = scratch / "raw"
    article_series: dict[str, pd.Series] = {}
    for team, articles in TEAM_ARTICLES.items():
        summed: pd.Series | None = None
        for article in articles:
            path = raw_dir / f"{article}.json"
            body = json.loads(path.read_text())
            items = body["items"]
            dates = pd.to_datetime([it["timestamp"][:8] for it in items], format="%Y%m%d")
            views = pd.Series([float(it["views"]) for it in items], index=dates, dtype="float64")
            views = views.groupby(views.index).sum()
            summed = views if summed is None else summed.add(views, fill_value=0.0)
        assert summed is not None
        article_series[team] = summed.sort_index()
    return article_series


def attention_for_window(
    series: pd.Series, window_start: pd.Timestamp, window_end: pd.Timestamp
) -> float:
    sliced = series.loc[(series.index >= window_start) & (series.index <= window_end)]
    return float(sliced.sum())


# --------------------------------------------------------------------------
# Game/team-game table construction
# --------------------------------------------------------------------------


def load_games(schedules_path: Path) -> pd.DataFrame:
    schedules = pd.read_parquet(schedules_path)
    games = schedules.loc[
        (schedules["game_type"] == "REG")
        & (schedules["season"] >= SEASON_START)
        & (schedules["season"] <= SEASON_END)
    ].copy()
    games["home_team"] = _canonical(games["home_team"])
    games["away_team"] = _canonical(games["away_team"])
    games["gameday"] = pd.to_datetime(games["gameday"], errors="raise")
    games["spread_line"] = pd.to_numeric(games["spread_line"], errors="coerce")
    games = games.loc[games["spread_line"].notna()].copy()
    games = add_ats_outcomes(games)
    games = games.loc[games["home_cover"].notna()].copy()
    games["week"] = games["week"].astype(int)
    games["season"] = games["season"].astype(int)
    games["week_block"] = games["season"] * 100 + games["week"]
    return games.reset_index(drop=True)


def build_team_game_long(games: pd.DataFrame, team_views: dict[str, pd.Series]) -> pd.DataFrame:
    """One row per (game, side): team, gameday, attention window value."""

    sides = []
    for is_home, team_col in ((True, "home_team"), (False, "away_team")):
        side = pd.DataFrame(
            {
                "game_id": games["game_id"],
                "season": games["season"],
                "week": games["week"],
                "gameday": games["gameday"],
                "team": games[team_col],
                "is_home": is_home,
            }
        )
        sides.append(side)
    long_df = pd.concat(sides, ignore_index=True)

    weekday = long_df["gameday"].dt.weekday  # Monday=0 ... Sunday=6, Tuesday=1
    tuesday_offset = (weekday - 1) % 7
    window_end = long_df["gameday"] - pd.to_timedelta(tuesday_offset, unit="D")
    window_start = window_end - pd.Timedelta(days=6)
    long_df["window_start"] = window_start
    long_df["window_end"] = window_end

    attention = np.empty(len(long_df), dtype="float64")
    for team, group in long_df.groupby("team", sort=False):
        series = team_views[team]
        vals = [
            attention_for_window(series, ws, we)
            for ws, we in zip(group["window_start"], group["window_end"], strict=True)
        ]
        attention[group.index.to_numpy()] = vals
    long_df["attention"] = attention

    long_df = long_df.sort_values(["team", "season", "gameday"]).reset_index(drop=True)
    grouped = long_df.groupby(["team", "season"], sort=False)["attention"]
    trailing_mean = grouped.transform(
        lambda s: (
            s.shift(1).rolling(window=TRAILING_WINDOW_GAMES, min_periods=TRAILING_MIN_GAMES).mean()
        )
    )
    trailing_std = grouped.transform(
        lambda s: (
            s.shift(1).rolling(window=TRAILING_WINDOW_GAMES, min_periods=TRAILING_MIN_GAMES).std()
        )
    )
    long_df["trailing_mean"] = trailing_mean
    long_df["trailing_std"] = trailing_std
    long_df["has_baseline"] = trailing_mean.notna() & trailing_std.notna() & (trailing_std > 0)
    with np.errstate(invalid="ignore", divide="ignore"):
        long_df["attention_z"] = (long_df["attention"] - trailing_mean) / trailing_std
    long_df.loc[~long_df["has_baseline"], "attention_z"] = np.nan

    # prior in-season game's score margin (needed for hot_after_loss)
    result = pd.to_numeric(
        games.set_index("game_id").reindex(long_df["game_id"])["result"], errors="coerce"
    ).to_numpy()
    long_df["team_score_margin"] = np.where(long_df["is_home"], result, -result)
    long_df = long_df.sort_values(["team", "season", "gameday"]).reset_index(drop=True)
    long_df["prior_score_margin"] = long_df.groupby(["team", "season"], sort=False)[
        "team_score_margin"
    ].shift(1)

    return long_df


def attach_game_level(games: pd.DataFrame, long_df: pd.DataFrame) -> pd.DataFrame:
    home_side = long_df.loc[long_df["is_home"]].set_index("game_id")
    away_side = long_df.loc[~long_df["is_home"]].set_index("game_id")

    out = games.set_index("game_id").copy()
    out["home_z"] = home_side["attention_z"]
    out["home_has_baseline"] = home_side["has_baseline"]
    out["home_prior_score_margin"] = home_side["prior_score_margin"]
    out["away_z"] = away_side["attention_z"]
    out["away_has_baseline"] = away_side["has_baseline"]

    is_home_favorite = out["spread_line"] > 0
    is_away_favorite = out["spread_line"] < 0
    out["favorite_z"] = np.where(is_home_favorite, out["home_z"], out["away_z"])
    out["dog_z"] = np.where(is_home_favorite, out["away_z"], out["home_z"])
    out["favorite_covered"] = np.select(
        [is_home_favorite, is_away_favorite],
        [out["home_cover"], 1.0 - out["home_cover"]],
        default=np.nan,
    )
    out["favorite_defined"] = is_home_favorite | is_away_favorite

    return out.reset_index()


# --------------------------------------------------------------------------
# Bootstrap (verbatim pattern from nfl_bias_battery_screen.py)
# --------------------------------------------------------------------------


def summarize_population(
    df: pd.DataFrame,
    *,
    flag: pd.Series,
    value_col: str,
    block_col: str,
    sign: int,
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
    subset_cover = float(work.loc[work["_flag"], value_col].mean())
    complement_cover = float(work.loc[~work["_flag"], value_col].mean())
    raw_gap_pts = sign * (subset_cover - complement_cover) * 100.0
    fraction_of_slate = n_flag / n_total
    full_slate_effect_pts = raw_gap_pts * fraction_of_slate

    draws = block_bootstrap_two_group(
        work,
        flag_col="_flag",
        value_col=value_col,
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
        "week_blocked_ci95_scaled": [float(lower), float(upper)],
        "probability_positive": float(np.mean(signed_draws > 0)) if len(signed_draws) else np.nan,
        "bootstrap_samples": samples,
        "dropped_draws": int(dropped),
        "insufficient_data": False,
    }


# --------------------------------------------------------------------------
# Predeclared cells
# --------------------------------------------------------------------------


def build_cells(game_df: pd.DataFrame) -> dict[str, dict[str, Any]]:
    cells: dict[str, dict[str, Any]] = {}

    eligible = game_df["home_has_baseline"].fillna(False).to_numpy(dtype=bool)
    cells["hot_team_fade"] = {
        "eligible": eligible,
        "flag": (game_df["home_z"] >= 2.0).to_numpy(dtype=bool),
        "value_col": "home_cover",
        "sign": -1,
        "description": "Home team attention z >= +2 (subset) vs complement, response "
        "home_cover; hypothesized fade of an overhyped hot home team." + DESCRIPTION_SUFFIX,
    }

    eligible = game_df["away_has_baseline"].fillna(False).to_numpy(dtype=bool)
    cells["away_hot"] = {
        "eligible": eligible,
        "flag": (game_df["away_z"] >= 2.0).to_numpy(dtype=bool),
        "value_col": "home_cover",
        "sign": 1,
        "description": "Away team attention z >= +2 (subset) vs complement, response "
        "home_cover; hypothesized home-side value when the away team is the hot story."
        + DESCRIPTION_SUFFIX,
    }

    both_baseline = (
        game_df["home_has_baseline"].fillna(False) & game_df["away_has_baseline"].fillna(False)
    ).to_numpy(dtype=bool)
    fav_eligible = both_baseline & game_df["favorite_defined"].fillna(False).to_numpy(dtype=bool)
    gap_fav_dog = (game_df["favorite_z"] - game_df["dog_z"]).to_numpy(dtype=float)
    gap_dog_fav = (game_df["dog_z"] - game_df["favorite_z"]).to_numpy(dtype=float)
    cells["attention_gap_favorite"] = {
        "eligible": fav_eligible,
        "flag": gap_fav_dog >= 1.5,
        "value_col": "favorite_covered",
        "sign": -1,
        "description": "Favorite's attention z minus dog's z >= +1.5 (subset) vs "
        "complement, response favorite_covered; hypothesized fade of an overhyped "
        "favorite." + DESCRIPTION_SUFFIX,
    }
    cells["attention_gap_dog"] = {
        "eligible": fav_eligible,
        "flag": gap_dog_fav >= 1.5,
        "value_col": "favorite_covered",
        "sign": 1,
        "description": "Dog's attention z minus favorite's z >= +1.5 (subset) vs "
        "complement, response favorite_covered; hypothesized favorite value when the "
        "underdog is the hot story." + DESCRIPTION_SUFFIX,
    }

    cells["both_cold"] = {
        "eligible": both_baseline,
        "flag": ((game_df["home_z"] <= -0.5) & (game_df["away_z"] <= -0.5)).to_numpy(dtype=bool),
        "value_col": "home_cover",
        "sign": -1,
        "description": "Both teams attention z <= -0.5, a low-visibility-game proxy "
        "(subset) vs complement, response home_cover; DIFFERENT instrument than prior "
        "null book-count-based low-visibility proxies." + DESCRIPTION_SUFFIX,
    }

    hot_after_loss_eligible = (
        game_df["home_has_baseline"].fillna(False) & game_df["home_prior_score_margin"].notna()
    ).to_numpy(dtype=bool)
    cells["hot_after_loss"] = {
        "eligible": hot_after_loss_eligible,
        "flag": ((game_df["home_z"] >= 2.0) & (game_df["home_prior_score_margin"] < 0)).to_numpy(
            dtype=bool
        ),
        "value_col": "home_cover",
        "sign": 1,
        "description": "Home team attention z >= +2 AND lost its immediately preceding "
        "in-season game (subset) vs complement, response home_cover; hypothesized "
        "market overreaction to negative-recency news." + DESCRIPTION_SUFFIX,
    }

    return cells


# --------------------------------------------------------------------------
# Reliability
# --------------------------------------------------------------------------


def split_half_reliability(long_df: pd.DataFrame) -> dict[str, Any]:
    base = long_df.loc[long_df["has_baseline"]].copy()
    base["half"] = np.where(base["week"] % 2 == 0, "even", "odd")
    team_half = base.groupby(["team", "half"])["attention_z"].mean().unstack("half")
    team_half = team_half.dropna()
    n_teams = len(team_half)
    if n_teams < 3:
        return {"n_teams": n_teams, "correlation": None, "note": "too few teams with both halves"}
    corr = float(team_half["odd"].corr(team_half["even"]))
    return {
        "n_teams": n_teams,
        "correlation": corr,
        "odd_half_mean_z_by_team": team_half["odd"].round(4).to_dict(),
        "even_half_mean_z_by_team": team_half["even"].round(4).to_dict(),
    }


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scratch", type=Path, default=DEFAULT_SCRATCH)
    parser.add_argument("--schedules", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--samples", type=int, default=BOOTSTRAP_SAMPLES)
    parser.add_argument("--seed", type=int, default=BOOTSTRAP_SEED)
    args = parser.parse_args()

    schedules_path = args.schedules or latest_schedules()
    started = time.time()
    timestamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    output_dir: Path = args.output or (REPO / "artifacts" / "attention_battery" / timestamp)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== loading pageviews from {args.scratch} + {schedules_path} ===")
    team_views = load_team_daily_views(args.scratch)
    for team, series in team_views.items():
        print(
            f"  {team}: {len(series)} distinct dates, {series.index.min()} - {series.index.max()}"
        )

    games = load_games(schedules_path)
    print(f"\nREG {SEASON_START}-{SEASON_END} games with spread_line: {len(games)}")

    long_df = build_team_game_long(games, team_views)
    n_no_baseline = int((~long_df["has_baseline"]).sum())
    week1_2_no_baseline = int(
        (~long_df["has_baseline"] & long_df.groupby(["team", "season"]).cumcount().lt(2)).sum()
    )
    print(
        f"team-game rows: {len(long_df)}; no baseline: {n_no_baseline} "
        f"(of which week1/2-of-season: {week1_2_no_baseline})"
    )

    game_df = attach_game_level(games, long_df)

    reliability = split_half_reliability(long_df)
    print(
        f"\n=== split-half reliability (odd/even week, team-level mean attention z) ==="
        f"\n  n_teams={reliability['n_teams']} correlation={reliability.get('correlation')}"
    )

    cells = build_cells(game_df)
    results = []
    for name, spec in cells.items():
        print(f"\n=== {name} ===")
        value_col = spec["value_col"]
        full_eligible_df = game_df.loc[spec["eligible"]].copy()
        full_eligible_df["_flag"] = spec["flag"][spec["eligible"]]
        full_eligible_df = full_eligible_df.loc[full_eligible_df[value_col].notna()].reset_index(
            drop=True
        )
        flag = full_eligible_df["_flag"]

        primary = summarize_population(
            full_eligible_df,
            flag=flag,
            value_col=value_col,
            block_col="week_block",
            sign=spec["sign"],
            samples=args.samples,
            seed=args.seed,
        )
        secondary = summarize_population(
            full_eligible_df,
            flag=flag,
            value_col=value_col,
            block_col="season",
            sign=spec["sign"],
            samples=args.samples,
            seed=args.seed,
        )
        results.append(
            {
                "name": f"attention_battery_{name}",
                "description": spec["description"],
                "sign_dir": spec["sign"],
                "value_col": value_col,
                "week_blocked_primary": primary,
                "season_blocked_secondary": secondary,
            }
        )
        if primary.get("insufficient_data"):
            print("  insufficient data (empty subset or complement)")
            continue
        print(
            f"  n_flag={primary['n_flag']} n_total={primary['n_total']} "
            f"subset_cover={primary['subset_cover']:.4f} "
            f"complement_cover={primary['complement_cover']:.4f}"
        )
        print(
            f"  raw_gap={primary['raw_gap_pts']:+.3f}pts "
            f"frac_of_slate={primary['fraction_of_slate']:.4f} "
            f"full_slate_effect={primary['full_slate_effect_pts']:+.4f}pts"
        )
        print(
            f"  week-blocked 95% scaled [{primary['week_blocked_ci95_scaled'][0]:+.4f}, "
            f"{primary['week_blocked_ci95_scaled'][1]:+.4f}] "
            f"P+={primary['probability_positive']:.4f}"
        )
        if not secondary.get("insufficient_data"):
            print(
                f"  season-blocked (secondary) 95% scaled "
                f"[{secondary['week_blocked_ci95_scaled'][0]:+.4f}, "
                f"{secondary['week_blocked_ci95_scaled'][1]:+.4f}] "
                f"P+={secondary['probability_positive']:.4f}"
            )

    configuration = {
        "command": "attention-battery-screen",
        "schedules": str(schedules_path),
        "scratch": str(args.scratch),
        "bootstrap_samples": args.samples,
        "bootstrap_seed": args.seed,
        "season_start": SEASON_START,
        "season_end": SEASON_END,
    }
    payload = {
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(started)),
        "elapsed_seconds": time.time() - started,
        "bootstrap_samples": args.samples,
        "bootstrap_seed": args.seed,
        "season_start": SEASON_START,
        "season_end": SEASON_END,
        "n_cells": len(cells),
        "n_reg_games": len(games),
        "n_team_game_rows": len(long_df),
        "n_no_baseline": n_no_baseline,
        "n_week1_2_no_baseline": week1_2_no_baseline,
        "predeclaration": "<scratchpad>/agent_attention/predeclaration.md (frozen before scoring)",
        "reliability": reliability,
        "results": results,
        "provenance": artifact_provenance(configuration, schedules_path, project_root=REPO),
    }
    write_experiment_artifact(
        output_dir,
        "results.json",
        payload,
        command="attention-battery-screen",
        metrics=payload,
        notes=(
            "Pilot/measure-only lead-generation battery (6 predeclared cells); mined "
            "family, every cell predeclared to record unresolved_below_power via a "
            "separate nfl-ats weak-signals record call regardless of interval shape "
            "(AGENTS.md)."
        ),
    )
    print(f"\nwrote {output_dir / 'results.json'}")


if __name__ == "__main__":
    main()
