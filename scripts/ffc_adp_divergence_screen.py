"""Predeclared FFC ADP divergence screen (docs/ffc_adp_divergence_screen.md).

Implements EXACTLY the freeze written into that doc before any scoring ran:
mechanism = crowd enthusiasm (dated pre-Week-1 FantasyFootballCalculator ADP)
diverging from market price in Weeks 1-4 where team-quality priors are
thinnest. ADP is a preseason covariate with no in-season refresh -- disclosed,
not discovered.

Six frozen cells, each with ONE direction chosen before scoring (see the doc's
section 4 table): back-the-crowd on high-ADP-roster underdogs and on lone
|z|>1 positive ADP-vs-prior-wins residual teams, weeks 1-4 primary, weeks 1-2
thin-info replications, standard-scoring robustness replications.

Standard battery: week-blocked primary bootstrap (block = season*100+week),
20k draws, seed 20260822; identical season-blocked secondary; effect =
(mean(forced pick cover) - 0.5) * 100 accuracy points on the full qualifying
slate. Vig note is part of the predeclaration (~+2.4 accuracy points breakeven
at -110), not an afterthought.

Classification policy is the AGENTS.md binding taxonomy, decided before
scoring: only whole-interval-below-zero wrong-sign or positive-control closes;
everything else records unresolved_below_power via explicit `nfl-ats
weak-signals record` calls made by the caller (NOT by this script -- neither
registry JSON is touched here). This script writes its results artifact plus
an automatic low-stakes experiment-provenance stamp to
`registry/experiments/ffc-adp-divergence-screen/`.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

sys.path.append(str(REPO / "scripts"))

from _common import latest_schedules  # noqa: E402

from nfl_ats.constants import TEAM_ABBREVIATION_ALIASES  # noqa: E402
from nfl_ats.features import add_ats_outcomes  # noqa: E402
from nfl_ats.provenance import artifact_provenance, write_experiment_artifact  # noqa: E402

BOOTSTRAP_SAMPLES = 20_000
BOOTSTRAP_SEED = 20260822
SEASON_START = 2010
SEASON_END = 2025
WEEK_BLOCK_MAX = 4
WEEK_THIN_MAX = 2
Z_THRESHOLD = 1.0
DEFAULT_ADP_SNAPSHOT = "20260822T004750Z"

PREDECLARED_CELL_NAMES = frozenset(
    {
        "ffc_adp_cellA_highadp_underdog_back_ppr_w14",
        "ffc_adp_cellB_adpwins_residual_pos_back_ppr_w14",
        "ffc_adp_cellC_highadp_underdog_back_ppr_w12",
        "ffc_adp_cellD_adpwins_residual_pos_back_ppr_w12",
        "ffc_adp_robust_std_cellA_highadp_underdog_back_w14",
        "ffc_adp_robust_std_cellB_adpwins_residual_pos_back_w14",
    }
)

VIG_NOTE = (
    "predeclared vig framing: ATS breakeven at -110 is ~52.4%, so forced-pick "
    "effects clear ~+2.4 accuracy points before being wagering-grade at all; "
    "historical accuracy is never read as a stable profit claim"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------


def load_schedule(schedules_path: Path) -> pd.DataFrame:
    raw = pd.read_parquet(schedules_path)
    df = raw.loc[raw["game_type"] == "REG"].copy()
    df["season"] = pd.to_numeric(df["season"], errors="raise").astype(int)
    df["week"] = pd.to_numeric(df["week"], errors="raise").astype(int)
    for column in ("home_team", "away_team"):
        df[column] = df[column].replace(TEAM_ABBREVIATION_ALIASES)
    return df


def prior_season_wins(schedule: pd.DataFrame) -> pd.Series:
    """REG wins per (season, team), tie = 0.5, indexed for season-1 lookup."""

    reg = schedule.loc[schedule["season"].between(SEASON_START - 1, SEASON_END)]
    home = pd.DataFrame(
        {
            "season": reg["season"],
            "team": reg["home_team"],
            "win": np.select(
                [reg["home_score"] > reg["away_score"], reg["home_score"] < reg["away_score"]],
                [1.0, 0.0],
                default=0.5,
            ),
        }
    )
    away = home.assign(team=reg["away_team"])
    wins = pd.concat([home, away]).groupby(["season", "team"])["win"].sum()
    prior = wins.copy()
    prior.index = pd.MultiIndex.from_arrays(
        [prior.index.get_level_values("season") + 1, prior.index.get_level_values("team")]
    )
    return prior


def load_adp(adp_root: Path, scoring: str) -> pd.DataFrame:
    table = pd.read_parquet(
        adp_root / "team_top8_feasibility.parquet",
        columns=["year", "scoring", "franchise_code", "n_top8", "mean_adp_top8"],
    )
    agg = table.loc[table["scoring"] == scoring].copy()
    agg = agg.rename(columns={"year": "season"})
    agg["franchise_code"] = agg["franchise_code"].replace({"LAR": "LA"})
    return agg


def add_quality_ranks(agg: pd.DataFrame) -> pd.DataFrame:
    work = agg.sort_values(["season", "mean_adp_top8"]).copy()
    ranks = work.groupby("season")["mean_adp_top8"].rank(method="first")
    sizes = work.groupby("season")["mean_adp_top8"].transform("size")
    work["top_tercile"] = ranks <= np.ceil(sizes / 3.0)
    return work


def add_residual_z(agg: pd.DataFrame, prior_wins: pd.Series) -> pd.DataFrame:
    """Standardized OLS residuals of adp rank on prior-season wins, per season.

    Sign convention (frozen): positive z = crowd ranks the team BETTER than its
    prior-season wins justify (enthusiasm premium over the wins baseline).
    """

    work = add_quality_ranks(agg)
    keys = list(zip(work["season"].astype(int), work["franchise_code"], strict=True))
    work["prior_wins"] = [
        float(prior_wins.get((season - 1, team), np.nan)) for season, team in keys
    ]
    work["adp_rank"] = work.groupby("season")["mean_adp_top8"].rank(method="first")

    pieces = []
    for _, frame in work.groupby("season"):
        frame = frame.copy()
        fit = frame.dropna(subset=["prior_wins", "adp_rank"])
        z = pd.Series(np.nan, index=frame.index)
        if len(fit) >= 3 and fit["prior_wins"].std(ddof=1) > 0:
            x = fit["prior_wins"].to_numpy(dtype=np.float64)
            y = fit["adp_rank"].to_numpy(dtype=np.float64)
            if np.var(x, ddof=1) > 0 and y.std() > 0:
                slope = np.cov(x, y, ddof=1)[0, 1] / np.var(x, ddof=1)
                residual = y - (y.mean() + slope * (x - x.mean()))
                sd = residual.std(ddof=1)
                if sd > 0:
                    z.loc[fit.index] = -(residual - residual.mean()) / sd
        frame["residual_z"] = z
        pieces.append(frame)
    return pd.concat(pieces).sort_index()


def attach_adp(games: pd.DataFrame, adp_with_features: pd.DataFrame) -> pd.DataFrame:
    home = adp_with_features.add_prefix("home_")
    away = adp_with_features.add_prefix("away_")
    joined = games.merge(
        home,
        left_on=["season", "home_team"],
        right_on=["home_season", "home_franchise_code"],
        how="left",
    ).merge(
        away,
        left_on=["season", "away_team"],
        right_on=["away_season", "away_franchise_code"],
        how="left",
    )
    joined["has_adp"] = joined["home_mean_adp_top8"].notna() & joined["away_mean_adp_top8"].notna()
    return joined


# ---------------------------------------------------------------------------
# Cell construction (exactly the doc's section 4 table)
# ---------------------------------------------------------------------------


def cell_a_values(pop: pd.DataFrame) -> pd.Series:
    top_dog_home = pop["home_top_tercile"].fillna(False) & (pop["spread_line"] < 0)
    top_dog_away = pop["away_top_tercile"].fillna(False) & (pop["spread_line"] > 0)
    qualifies = (top_dog_home ^ top_dog_away) & (pop["spread_line"] != 0)
    value = np.where(top_dog_home, pop["home_cover"], 1.0 - pop["home_cover"])
    return pd.Series(np.where(qualifies, value, np.nan), index=pop.index)


def cell_b_values(pop: pd.DataFrame) -> pd.Series:
    extreme_home = (pop["home_residual_z"].abs() > Z_THRESHOLD).fillna(False)
    extreme_away = (pop["away_residual_z"].abs() > Z_THRESHOLD).fillna(False)
    lone_extreme = extreme_home ^ extreme_away
    z_home = pop["home_residual_z"].fillna(0.0)
    z_away = pop["away_residual_z"].fillna(0.0)
    positive_side = lone_extreme & np.where(extreme_home, z_home > 0, z_away > 0)
    pick_home = (extreme_home & positive_side).to_numpy()
    pick_away = (extreme_away & positive_side).to_numpy()
    value = np.where(pick_home, pop["home_cover"], 1.0 - pop["home_cover"])
    return pd.Series(np.where(pick_home | pick_away, value, np.nan), index=pop.index)


# ---------------------------------------------------------------------------
# Standard battery: week-blocked primary, season-blocked secondary
# ---------------------------------------------------------------------------


def block_bootstrap_single(
    df: pd.DataFrame, *, block_col: str, samples: int, seed: int
) -> np.ndarray:
    """Vectorized blocked bootstrap of ``(mean(value_col) - 0.5) * 100``."""

    blocks, block_index = np.unique(df[block_col].to_numpy(), return_inverse=True)
    block_index = np.asarray(block_index).reshape(-1)
    block_count = len(blocks)
    values = df["value"].to_numpy(dtype=np.float64)

    sums = np.bincount(block_index, weights=values, minlength=block_count).astype(np.float64)
    counts = np.bincount(block_index, minlength=block_count).astype(np.float64)

    rng = np.random.default_rng(seed)
    drawn = rng.multinomial(block_count, np.full(block_count, 1.0 / block_count), size=samples)
    resampled_count = drawn @ counts
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = (drawn @ sums) / resampled_count
    valid = resampled_count > 0
    return (mean[valid] - 0.5) * 100.0


def summarize(df: pd.DataFrame, *, samples: int, seed: int) -> dict[str, Any]:
    n_total = int(df["value"].notna().sum())
    scored = df.loc[df["value"].notna()]
    if n_total == 0:
        return {"n_total": 0, "n_week_blocks": 0, "insufficient_data": True}

    week_draws = block_bootstrap_single(scored, block_col="week_block", samples=samples, seed=seed)
    season_draws = block_bootstrap_single(
        scored, block_col="season_block", samples=samples, seed=seed
    )

    def _summary(draws: np.ndarray, blocks: int) -> dict[str, Any]:
        lower, upper = np.quantile(draws, [0.025, 0.975])
        return {
            "effect_pts": float((scored["value"].mean() - 0.5) * 100.0),
            "ci95": [float(lower), float(upper)],
            "probability_positive": float(np.mean(draws > 0)),
            "n_blocks": blocks,
            "dropped_draws": int(samples - len(draws)),
        }

    primary = _summary(week_draws, int(scored["week_block"].nunique()))
    secondary = _summary(season_draws, int(scored["season_block"].nunique()))
    return {
        "n_total": n_total,
        "cover_rate": float(scored["value"].mean()),
        **primary,
        "season_secondary": secondary,
        "bootstrap_samples": samples,
        "bootstrap_seed": seed,
        "insufficient_data": False,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--schedules", type=Path, default=latest_schedules())
    parser.add_argument(
        "--adp-root",
        type=Path,
        default=REPO / "artifacts" / "ffc_adp" / DEFAULT_ADP_SNAPSHOT,
    )
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--samples", type=int, default=BOOTSTRAP_SAMPLES)
    parser.add_argument("--seed", type=int, default=BOOTSTRAP_SEED)
    args = parser.parse_args()

    started = time.time()
    timestamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    output_dir = args.output or REPO / "artifacts" / "ffc_adp_divergence_screen" / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== loading schedules {args.schedules} + ADP {args.adp_root} ===")
    schedule = load_schedule(args.schedules)
    prior_wins = prior_season_wins(schedule)

    games = schedule.loc[
        schedule["season"].between(SEASON_START, SEASON_END)
        & (schedule["week"] <= WEEK_BLOCK_MAX)
        & schedule["spread_line"].notna()
    ].copy()
    games = add_ats_outcomes(games)
    games = games.loc[games["home_cover"].notna()].copy()
    games["week_block"] = games["season"] * 100 + games["week"]
    games["season_block"] = games["season"]

    cells: list[dict[str, Any]] = []
    coverage_rows: list[dict[str, Any]] = []

    for scoring in ("ppr", "standard"):
        adp = load_adp(args.adp_root, scoring)
        featured = add_residual_z(adp, prior_wins)
        pop = attach_adp(games, featured)
        base = pop.loc[pop["has_adp"]]

        per_season = base.groupby("season").size().reset_index(name="games_in_scope")
        for record in per_season.to_dict(orient="records"):
            coverage_rows.append(
                {
                    "scoring": scoring,
                    "season": int(record["season"]),
                    "games_weeks_1_4_both_adp_scored": int(record["games_in_scope"]),
                }
            )

        week_masks = {
            "w14": (base["week"] <= WEEK_BLOCK_MAX, "A", "B"),
            "w12": (base["week"] <= WEEK_THIN_MAX, "C", "D"),
        }
        for mask_label, (mask, a_letter, b_letter) in week_masks.items():
            if scoring != "ppr" and mask_label != "w14":
                continue
            subset = base.loc[mask].copy()
            week_label = "weeks 1-2" if mask_label == "w12" else "weeks 1-4"
            if scoring == "ppr":
                names = (
                    f"ffc_adp_cell{a_letter}_highadp_underdog_back_ppr_{mask_label}",
                    f"ffc_adp_cell{b_letter}_adpwins_residual_pos_back_ppr_{mask_label}",
                )
            else:
                names = (
                    f"ffc_adp_robust_std_cell{a_letter}_highadp_underdog_back_{mask_label}",
                    f"ffc_adp_robust_std_cell{b_letter}_adpwins_residual_pos_back_{mask_label}",
                )
            cell_values = (cell_a_values(subset), cell_b_values(subset))
            mechanisms = (
                "back the top-tercile mean_adp_top8 roster priced as underdog "
                "(frozen direction: crowd side covers)",
                "back the lone |z|>1 positive ADP-vs-prior-wins residual team "
                "(frozen direction: crowd-hot side covers)",
            )
            for name, values, mechanism in zip(names, cell_values, mechanisms, strict=True):
                work = subset.assign(value=values)
                summary = summarize(work, samples=args.samples, seed=args.seed)
                description = f"{week_label}, {scoring} scoring: {mechanism}"
                cells.append(
                    {
                        "name": name,
                        "description": description,
                        "frozen_direction": "back_the_crowd",
                        "season_start": int(subset["season"].min()),
                        "season_end": int(subset["season"].max()),
                        **summary,
                    }
                )
                print(
                    f"  {name}: n={summary.get('n_total')} "
                    f"effect={summary.get('effect_pts', float('nan')):+.4f}pts "
                    f"P+={summary.get('probability_positive', float('nan')):.4f}"
                )

    actual_names = {cell["name"] for cell in cells}
    missing = PREDECLARED_CELL_NAMES - actual_names
    extra = actual_names - PREDECLARED_CELL_NAMES
    assert not missing and not extra, (
        f"cell name mismatch vs predeclaration: missing={missing} extra={extra}"
    )

    print("\n=== cells (frozen directions, doc section 4) ===")
    for cell in cells:
        if cell.get("insufficient_data"):
            print(f"  {cell['name']}: insufficient data")
            continue
        sec = cell["season_secondary"]
        print(
            f"  {cell['name']}: n={cell['n_total']} "
            f"effect={cell['effect_pts']:+.4f}pts "
            f"week-blocked 95% [{cell['ci95'][0]:+.4f},{cell['ci95'][1]:+.4f}] "
            f"P+={cell['probability_positive']:.4f} | season-blocked "
            f"{sec['effect_pts']:+.4f}pts P+={sec['probability_positive']:.4f}"
        )

    configuration = {
        "command": "ffc-adp-divergence-screen",
        "schedules": str(args.schedules),
        "adp_root": str(args.adp_root),
        "bootstrap_samples": args.samples,
        "bootstrap_seed": args.seed,
        "season_start": SEASON_START,
        "season_end": SEASON_END,
        "z_threshold": Z_THRESHOLD,
        "week_blocks": [WEEK_BLOCK_MAX, WEEK_THIN_MAX],
    }
    payload = {
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(started)),
        "elapsed_seconds": time.time() - started,
        "predeclaration": "docs/ffc_adp_divergence_screen.md sections 1-6 (frozen before scoring)",
        "vig_note": VIG_NOTE,
        "adp_covariate_disclosure": (
            "ADP is a preseason covariate with NO in-season refresh; every aggregate is "
            "formed in late August / early September of its own season (latest observed "
            "window end 2011-09-09, which may include mocks run during that season's Week 1)"
        ),
        "known_gaps": (
            "BUF absent from ADP aggregates 2010/2011/2013/2014 both formats; LAR/NYJ/SF "
            "missing 2012 ppr; affected games drop out of the join"
        ),
        "coverage_by_season": coverage_rows,
        "results": cells,
        "provenance": artifact_provenance(
            configuration, args.adp_root / "team_top8_feasibility.parquet", project_root=REPO
        ),
        "input_sha256": {
            "schedules": _sha256(args.schedules),
            "adp_team_top8": _sha256(args.adp_root / "team_top8_feasibility.parquet"),
            "adp_metadata": _sha256(args.adp_root / "metadata.json"),
        },
    }
    write_experiment_artifact(
        output_dir,
        "results.json",
        payload,
        command="ffc-adp-divergence-screen",
        metrics={
            "cells": {
                cell["name"]: {
                    "effect_pts": cell.get("effect_pts"),
                    "ci95": cell.get("ci95"),
                    "probability_positive": cell.get("probability_positive"),
                    "n_total": cell.get("n_total"),
                }
                for cell in cells
            }
        },
        notes=(
            "Measure-only predeclared FFC ADP divergence screen (6 frozen cells, one "
            "direction each, frozen in docs/ffc_adp_divergence_screen.md before scoring). "
            "Every cell records unresolved_below_power unless the whole interval sits below "
            "zero against its frozen direction; recording happens via explicit nfl-ats "
            "weak-signals record lines returned to the owner, never inside this script."
        ),
        source=(
            "scripts/ffc_adp_divergence_screen.py; docs/ffc_adp_divergence_screen.md; "
            "artifacts/ffc_adp_divergence_screen/"
        ),
        project_root=REPO,
    )
    print(f"\nwrote {output_dir / 'results.json'}")


if __name__ == "__main__":
    main()
