"""Venue milestone states screen: 5 predeclared cells (home opener,
new-stadium/relocation debut, former-stadium swing, post-bye home/road)
screened against the spread on 2009-2025 NFL REG games with a week-blocked
bootstrap (season-blocked secondary), full-slate scaled accuracy_points
effects, seeded and deterministic.

Predeclaration frozen in ``docs/venue_milestone_screen.md`` BEFORE this
script scored anything. Machinery reused from
``scripts/nfl_travel_rest_battery_screen.py``. Measure-only: never writes
either registry JSON; every flag is a schedule fact derived solely from
schedules.parquet columns known at schedule release plus static reference
mappings, so all cells are point-in-time safe by construction.

Writes JSON to ``artifacts/venue_milestone_screen/<UTC timestamp>/results.json``
and stamps ``registry/experiments/venue-milestone-screen/``.
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

from nfl_ats.features import add_ats_outcomes  # noqa: E402
from nfl_ats.provenance import artifact_provenance, write_experiment_artifact  # noqa: E402

DEFAULT_SCHEDULE_COLUMNS = [
    "game_id",
    "season",
    "week",
    "game_type",
    "gameday",
    "home_team",
    "away_team",
    "result",
    "spread_line",
    "location",
    "stadium",
]

BOOTSTRAP_SAMPLES = 20_000
BOOTSTRAP_SEED = 20260821
SEASON_START = 2009
SEASON_END = 2025

POST_BYE_GAP_DAYS = 12

VENUE_ALIASES: dict[str, tuple[str, ...]] = {
    "AT&T Stadium": ("Cowboys Stadium", "AT&T Stadium"),
    "MetLife Stadium": ("New Meadowlands Stadium", "MetLife Stadium"),
    "Metrodome": ("Hubert H. Humphrey Metrodome", "Mall of America Field"),
    "TCF Bank Stadium": ("TCF Bank Stadium",),
    "U.S. Bank Stadium": ("U.S. Bank Stadium",),
    "State Farm Stadium": ("University of Phoenix Stadium", "State Farm Stadium"),
    "NRG Stadium": ("Reliant Stadium", "NRG Stadium"),
    "GEHA Field at Arrowhead Stadium": ("Arrowhead Stadium", "GEHA Field at Arrowhead Stadium"),
    "Paycor Stadium": ("Paul Brown Stadium", "Paycor Stadium"),
    "Jacksonville Jaguars Venues": (
        "Jacksonville Municipal Stadium",
        "EverBank Field",
        "TIAA Bank Stadium",
        "EverBank Stadium",
    ),
    "Hard Rock Stadium": ("Dolphin Stadium", "Sun Life Stadium", "Hard Rock Stadium"),
    "Nissan Stadium": ("LP Field", "Nissan Stadium"),
    "New Era Field": ("Ralph Wilson Stadium", "New Era Field", "Highmark Stadium"),
    "Empower Field at Mile High": (
        "Invesco Field at Mile High",
        "Sports Authority Field at Mile High",
        "Empower Field at Mile High",
    ),
    "Lumen Field": ("Qwest Field", "CenturyLink Field", "Lumen Field"),
    "Oakland Coliseum": (
        "Oakland-Alameda County Coliseum",
        "O.co Coliseum",
        "Ring Central Coliseum",
    ),
    "Mercedes-Benz Superdome": ("Louisiana Superdome", "Mercedes-Benz Superdome"),
    "Mercedes-Benz Stadium": ("Mercedes-Benz Stadium",),
    "Acrisure Stadium": ("Heinz Field", "Acrisure Stadium"),
    "FirstEnergy Stadium": ("Cleveland Browns Stadium", "FirstEnergy Stadium"),
    "Candlestick Park": ("Candlestick Park",),
    "Levi's Stadium": ("Levi's Stadium",),
    "Los Angeles Memorial Coliseum": ("Los Angeles Memorial Coliseum",),
    "SoFi Stadium": ("SoFi Stadium",),
    "StubHub Center": ("StubHub Center",),
    "Allegiant Stadium": ("Allegiant Stadium",),
    "Gillette Stadium": ("Gillette Stadium",),
    "M&T Bank Stadium": ("M&T Bank Stadium",),
    "Soldier Field": ("Soldier Field",),
    "Bank of America Stadium": ("Bank of America Stadium",),
    "Ford Field": ("Ford Field",),
    "Lambeau Field": ("Lambeau Field",),
    "Lucas Oil Stadium": ("Lucas Oil Stadium",),
    "Lincoln Financial Field": ("Lincoln Financial Field",),
    "FedExField": ("FedExField",),
    "Raymond James Stadium": ("Raymond James Stadium",),
    "Georgia Dome": ("Georgia Dome",),
    "Giants Stadium": ("Giants Stadium",),
    "Edward Jones Dome": ("Edward Jones Dome",),
    "Qualcomm Stadium": ("Qualcomm Stadium",),
    "Rogers Centre": ("Rogers Centre",),
    "Wembley Stadium": ("Wembley Stadium",),
    "Tottenham Stadium": ("Tottenham Stadium",),
    "Twickenham Stadium": ("Twickenham Stadium",),
    "Azteca Stadium": ("Azteca Stadium",),
    "Allianz Arena": ("Allianz Arena",),
    "Arena Corinthians": ("Arena Corinthians",),
    "Deutsche Bank Park": ("Deutsche Bank Park",),
}

NEW_VENUE_DEBUTS: tuple[dict[str, Any], ...] = (
    {"team": "DAL", "debut_season": 2009, "canonical_venue": "AT&T Stadium"},
    {"team": "NYG", "debut_season": 2010, "canonical_venue": "MetLife Stadium"},
    {"team": "NYJ", "debut_season": 2010, "canonical_venue": "MetLife Stadium"},
    {"team": "SF", "debut_season": 2014, "canonical_venue": "Levi's Stadium"},
    {"team": "MIN", "debut_season": 2014, "canonical_venue": "TCF Bank Stadium"},
    {
        "team": "LA",
        "debut_season": 2016,
        "canonical_venue": "Los Angeles Memorial Coliseum",
    },
    {"team": "MIN", "debut_season": 2016, "canonical_venue": "U.S. Bank Stadium"},
    {"team": "LAC", "debut_season": 2017, "canonical_venue": "StubHub Center"},
    {"team": "ATL", "debut_season": 2017, "canonical_venue": "Mercedes-Benz Stadium"},
    {"team": "LA", "debut_season": 2020, "canonical_venue": "SoFi Stadium"},
    {"team": "LAC", "debut_season": 2020, "canonical_venue": "SoFi Stadium"},
    {"team": "LV", "debut_season": 2020, "canonical_venue": "Allegiant Stadium"},
)


def _latest_schedules() -> Path:
    candidates = sorted((REPO / "data/raw").glob("*/schedules.parquet"))
    if not candidates:
        raise FileNotFoundError("no data/raw/*/schedules.parquet snapshot found")
    return candidates[-1]


def default_schedules() -> Path:
    """Resolve lazily so importing this module never requires local data."""
    return _latest_schedules()


_CANONICAL: dict[str, str] = {raw: canon for canon, names in VENUE_ALIASES.items() for raw in names}


def canonical_venue(name: object) -> str | None:
    if not isinstance(name, str):
        return None
    return _CANONICAL.get(name)


def load_population(schedules_path: Path) -> pd.DataFrame:
    raw = pd.read_parquet(schedules_path)
    available = [c for c in DEFAULT_SCHEDULE_COLUMNS if c in raw.columns]
    df = raw.loc[:, available].copy()
    df = df.loc[df["game_type"] == "REG"].copy()
    df["season"] = pd.to_numeric(df["season"], errors="raise").astype(int)
    df["week"] = pd.to_numeric(df["week"], errors="raise").astype(int)
    df = df.loc[df["season"].between(SEASON_START, SEASON_END)].reset_index(drop=True)

    df = add_ats_outcomes(df)
    df = df.loc[df["home_cover"].notna()].reset_index(drop=True)

    df["week_block"] = df["season"] * 100 + df["week"]
    df["gameday_dt"] = pd.to_datetime(df["gameday"], errors="coerce")
    df["canonical_venue"] = df["stadium"].map(canonical_venue)
    return df


def build_flags(df: pd.DataFrame) -> tuple[dict[str, pd.Series], dict[str, Any]]:
    diagnostics: dict[str, Any] = {}

    home_rows = df.loc[(df["location"] == "Home") & df["canonical_venue"].notna()]
    ordered = home_rows.sort_values(["home_team", "season", "gameday_dt"])
    first_home_game_ids = set(ordered.drop_duplicates(subset=["home_team", "season"])["game_id"])
    home_opener_flag = df["game_id"].isin(first_home_game_ids)

    debut_flag = pd.Series(False, index=df.index)
    debut_cases: list[dict[str, Any]] = []
    for spec in NEW_VENUE_DEBUTS:
        cand = home_rows.loc[
            (home_rows["home_team"] == spec["team"])
            & (home_rows["canonical_venue"] == spec["canonical_venue"])
            & (home_rows["season"] == spec["debut_season"])
        ].sort_values("gameday_dt")
        if cand.empty:
            raise ValueError(f"no games found for debut spec {spec}")
        first_row = cand.iloc[0]
        if int(first_row["season"]) != spec["debut_season"]:
            raise ValueError(
                f"debut spec {spec} mismatch: first game found in season {int(first_row['season'])}"
            )
        debut_cases.append(
            {
                "team": spec["team"],
                "debut_season": spec["debut_season"],
                "canonical_venue": spec["canonical_venue"],
                "raw_stadium": str(first_row["stadium"]),
                "game_id": str(first_row["game_id"]),
                "gameday": str(first_row["gameday"]),
            }
        )
        debut_flag |= df["game_id"] == first_row["game_id"]

    team_home_venues = (
        home_rows.groupby(["home_team", "season"])["canonical_venue"]
        .agg(lambda s: s.mode().iat[0])
        .reset_index()
    )
    swing_flag = pd.Series(False, index=df.index)
    swing_cases: list[dict[str, Any]] = []
    away_rows = df.loc[
        (df["location"] == "Home") & df["canonical_venue"].notna() & df["away_team"].notna()
    ]
    for _, row in away_rows.iterrows():
        current = team_home_venues.loc[
            (team_home_venues["home_team"] == row["away_team"])
            & (team_home_venues["season"] == row["season"]),
            "canonical_venue",
        ]
        prior = team_home_venues.loc[
            (team_home_venues["home_team"] == row["away_team"])
            & (team_home_venues["season"] < row["season"]),
            "canonical_venue",
        ]
        if row["canonical_venue"] in set(prior) - set(current):
            swing_flag |= df["game_id"] == row["game_id"]
            swing_cases.append(
                {
                    "game_id": str(row["game_id"]),
                    "season": int(row["season"]),
                    "week": int(row["week"]),
                    "visiting_team": str(row["away_team"]),
                    "host_team": str(row["home_team"]),
                    "venue": str(row["stadium"]),
                }
            )

    long_rows = []
    for _, g in df.iterrows():
        for side, team in (("home", g["home_team"]), ("away", g["away_team"])):
            long_rows.append(
                {
                    "game_id": g["game_id"],
                    "season": g["season"],
                    "team": team,
                    "side": side,
                    "gameday_dt": g["gameday_dt"],
                }
            )
    long_df = pd.DataFrame(long_rows).sort_values(["team", "season", "gameday_dt"])
    long_df["gap_days"] = long_df.groupby(["team", "season"])["gameday_dt"].diff().dt.days
    long_df["post_bye"] = long_df["gap_days"] >= POST_BYE_GAP_DAYS

    def side_map(side: str) -> pd.Series:
        joined = df[["game_id"]].merge(
            long_df.loc[long_df["side"] == side, ["game_id", "post_bye"]],
            on="game_id",
            how="left",
        )
        return joined["post_bye"].fillna(False).astype(bool)

    post_bye_home_flag = side_map("home")
    post_bye_road_flag = side_map("away")

    flags = {
        "venue_milestone_home_opener": home_opener_flag,
        "venue_milestone_new_stadium_debut": debut_flag,
        "venue_milestone_former_stadium_swing": swing_flag,
        "venue_milestone_post_bye_home": post_bye_home_flag,
        "venue_milestone_post_bye_road": post_bye_road_flag,
    }
    assert len(flags) == 5, f"expected 5 predeclared cells, got {len(flags)}"
    diagnostics["new_stadium_debut_cases"] = debut_cases
    diagnostics["former_stadium_swing_cases"] = swing_cases
    return flags, diagnostics


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
    subset_cover = float(work.loc[work["_flag"], "home_cover"].mean())
    complement_cover = float(work.loc[~work["_flag"], "home_cover"].mean())
    raw_gap_pts = (subset_cover - complement_cover) * 100.0
    fraction_of_slate = n_flag / n_total
    full_slate_effect_pts = raw_gap_pts * fraction_of_slate

    draws = block_bootstrap_two_group(
        work,
        flag_col="_flag",
        value_col="home_cover",
        block_col=block_col,
        samples=samples,
        seed=seed,
    )
    dropped = samples - len(draws)
    scaled_draws = draws * fraction_of_slate
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
        "probability_positive": float(np.mean(draws > 0)) if len(draws) else np.nan,
        "bootstrap_samples": samples,
        "dropped_draws": int(dropped),
        "insufficient_data": False,
    }


CELL_DESCRIPTIONS = {
    "venue_milestone_home_opener": (
        "HOME team's first home game of its season (crowd-energy mechanism); "
        "predeclared direction positive."
    ),
    "venue_milestone_new_stadium_debut": (
        "Franchise's first regular-season home game in a venue new to it "
        "(brand-new opening or relocation destination; neutral-site "
        "internationals excluded); unfamiliarity mechanism; predeclared "
        "direction negative."
    ),
    "venue_milestone_former_stadium_swing": (
        "Team playing a regular-season road game at a physical venue that was "
        "its own home in an earlier season (relocation revenge); predeclared "
        "direction negative."
    ),
    "venue_milestone_post_bye_home": (
        "HOME team's first game after a bye (>=12-day gap to its immediately "
        "preceding game; strict bye definition excluding primetime extra "
        "rest); venue-conditioned rest variant disclosed vs "
        "travel_rest_home_off_bye; predeclared direction positive."
    ),
    "venue_milestone_post_bye_road": (
        "AWAY team's first game after a bye (same strict definition); mirror "
        "of the home cell; predeclared direction negative."
    ),
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--schedules", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--samples", type=int, default=BOOTSTRAP_SAMPLES)
    parser.add_argument("--seed", type=int, default=BOOTSTRAP_SEED)
    args = parser.parse_args()
    if args.schedules is None:
        args.schedules = default_schedules()

    started = time.time()
    timestamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    output_dir: Path = args.output or (REPO / "artifacts" / "venue_milestone_screen" / timestamp)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== loading {args.schedules} ===")
    df = load_population(args.schedules)
    print(
        f"REG {SEASON_START}-{SEASON_END} games: scored population {len(df)} "
        f"(pushes/missing dropped upstream of add_ats_outcomes filter)"
    )

    flags, diagnostics = build_flags(df)
    print(f"new-stadium debut cases enumerated: {len(diagnostics['new_stadium_debut_cases'])}")
    print(
        f"former-stadium swing cases enumerated: "
        f"{len(diagnostics['former_stadium_swing_cases'])} "
        f"{diagnostics['former_stadium_swing_cases']}"
    )

    results = []
    for name, flag in flags.items():
        print(f"\n=== {name} ===")
        week_blocked = summarize(
            df, flag=flag, block_col="week_block", samples=args.samples, seed=args.seed
        )
        season_blocked = summarize(
            df, flag=flag, block_col="season", samples=args.samples, seed=args.seed
        )
        cell = {
            "name": name,
            "description": CELL_DESCRIPTIONS[name],
            "n_flag": int(flag.sum()),
            "week_blocked": week_blocked,
            "season_blocked_secondary": season_blocked,
        }
        results.append(cell)
        wb = cell["week_blocked"]
        sb = cell["season_blocked_secondary"]
        if wb.get("insufficient_data"):
            print(f"  insufficient data (n_flag={wb['n_flag']}, n_complement={wb['n_complement']})")
            continue
        print(
            f"  n_flag={cell['n_flag']} n_total={wb['n_total']} "
            f"subset_cover={wb['subset_cover']:.4f} "
            f"complement_cover={wb['complement_cover']:.4f} "
            f"raw_gap={wb['raw_gap_pts']:+.3f}pts frac_of_slate={wb['fraction_of_slate']:.4f}"
        )
        print(
            f"  full_slate_effect={wb['full_slate_effect_pts']:+.4f}pts "
            f"week-blocked 95% [{wb['ci95_scaled'][0]:+.4f}, {wb['ci95_scaled'][1]:+.4f}] "
            f"P+={wb['probability_positive']:.4f} n_week_blocks={wb['n_blocks']}"
        )
        if not sb.get("insufficient_data"):
            print(
                f"  [season-blocked secondary] 95% [{sb['ci95_scaled'][0]:+.4f}, "
                f"{sb['ci95_scaled'][1]:+.4f}] P+={sb['probability_positive']:.4f} "
                f"n_seasons={sb['n_blocks']}"
            )

    configuration = {
        "command": "venue-milestone-screen",
        "schedules": str(args.schedules),
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
        "n_cells": len(flags),
        "season_start": SEASON_START,
        "season_end": SEASON_END,
        "n_scored_population": len(df),
        "predeclaration": "docs/venue_milestone_screen.md (frozen before scoring)",
        "point_in_time_safety": (
            "every flag is a schedule fact derived solely from schedules.parquet "
            "columns known at schedule release plus static reference mappings"
        ),
        "new_stadium_debut_cases": diagnostics["new_stadium_debut_cases"],
        "former_stadium_swing_cases": diagnostics["former_stadium_swing_cases"],
        "results": results,
        "provenance": artifact_provenance(configuration, args.schedules, project_root=REPO),
    }
    write_experiment_artifact(
        output_dir,
        "results.json",
        payload,
        command="venue-milestone-screen",
        metrics=payload,
        notes=(
            "Measure-only lead-generation screen (5 predeclared venue-milestone "
            "cells); mined family, every scoreable cell predeclared to record "
            "unresolved_below_power via a separate nfl-ats weak-signals record "
            "call regardless of interval shape (AGENTS.md)."
        ),
    )
    print(f"\nwrote {output_dir / 'results.json'}")


if __name__ == "__main__":
    main()
