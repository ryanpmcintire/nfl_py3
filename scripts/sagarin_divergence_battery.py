"""Predeclared Sagarin-divergence screen (docs/sagarin_backfill.md section 6).

**Ceiling caveat, stated up front per the project's own measured finding**:
a feature that only measures team quality better than the market is bounded
near zero (``team-quality-is-already-priced``, project MEMORY.md). Sagarin
divergence from the market line -- an independent power-rating's
disagreement with the market's own consensus price -- is exactly that
family. This screen's job is to locate WHERE in the divergence distribution
(tails vs. bulk, era, model-agreement) any residual signal lives, not to
assume an effect exists going in.

**Predeclaration.** Every cell below, the exact 3-point threshold, the
top-decile definition, the era split, and the RATING/home_edge_rating family
choice were frozen in
``<scratchpad>/sagarin_divergence/predeclaration.json`` BEFORE this script
computed anything. This module implements exactly what that document
specifies -- see it for the full text; this docstring only summarizes.

**Construction.** ``sagarin_implied_spread_home = home_rating - away_rating
+ home_edge_rating`` (home-positive convention, matching nflverse
``spread_line``'s own home-favored-positive convention). Two grades:

- CLOSE grade (``divergence_close``): ``sagarin_implied_spread_home -
  spread_line`` (the schedule's own recorded line), REG 2010-2025, wherever
  a Tuesday-asof Sagarin snapshot with a non-null ``home_edge_rating``
  exists.
- OPENER grade (``divergence_open``): ``sagarin_implied_spread_home -
  tue_open_home_spread`` (the project's own historical odds-snapshot
  archive), REG 2020-2025, restricted to
  ``nfl_ats.experiment_runner._opener_graded_features``'s own paired
  tue_open+close population (the same 1,537-game, 2020-2025 archive
  ``docs/opener_evaluation.md`` documents) intersected with Sagarin
  Tuesday-asof coverage.

**7 predeclared cells**, every one scored as ``(mean(sagarin_side_cover) -
0.5) * 100`` accuracy points via a week-blocked bootstrap (block =
season*100+week), except ``sagarin_battery_model_agreement_close`` which is
a two-group gap (agree accuracy - disagree accuracy). None of the 7 cells
carries a predeclared sign -- "does the Sagarin side cover" is genuinely
two-sided a priori -- so none can be ``wrong_sign_resolved``; no positive
control was run, so none can be ``positive_control_bound`` either. Per
AGENTS.md, every cell is expected to record ``unresolved_below_power``
regardless of interval shape.

**Measure-only.** This script never writes to the weak-signal registry
(``registry/weak_signals.json``); recording happens via a separate,
explicit ``nfl-ats weak-signals record`` invocation per cell (see
``scripts/record_sagarin_divergence_battery.py``). It DOES write an
automatic, low-stakes experiment-provenance stamp to ``registry/experiments/``
via ``write_experiment_artifact`` -- a run log, not a verdict.

Data:
- ``data/raw/sagarin/<snapshot>/asof_tuesday_view.parquet`` (default: the
  ``20260820T112501Z`` snapshot docs/sagarin_backfill.md documents).
- newest ``data/raw/*/schedules.parquet``.
- ``data/market/raw/`` historical-backfill odds archive (opener grade only).
- ``artifacts/active_ats_model.json`` + its named ``historical_evaluation``
  margin-backtest artifact (model-agreement cell only; skipped cleanly if
  either is absent, since generated artifacts are local-only per AGENTS.md).

Writes JSON to ``artifacts/sagarin_divergence_battery/<UTC timestamp>/results.json``
and prints a summary table to stdout.
"""

from __future__ import annotations

import argparse
import hashlib
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

from nfl_ats.active_model import active_artifact_path, load_active_ats_model  # noqa: E402
from nfl_ats.constants import TEAM_ABBREVIATION_ALIASES  # noqa: E402
from nfl_ats.experiment_runner import ExperimentRunnerError, _opener_graded_features  # noqa: E402
from nfl_ats.features import add_ats_outcomes  # noqa: E402
from nfl_ats.provenance import artifact_provenance, write_experiment_artifact  # noqa: E402

BOOTSTRAP_SAMPLES = 20_000
BOOTSTRAP_SEED = 20260820
SEASON_START = 2010
SEASON_END = 2025
LARGE_DIVERGENCE_THRESHOLD = 3.0
ERA_SPLITS: tuple[tuple[str, int, int], ...] = (
    ("2010_2016", 2010, 2016),
    ("2017_2025", 2017, 2025),
)
DEFAULT_SAGARIN_SNAPSHOT = "20260820T112501Z"

CEILING_CAVEAT = (
    "team-quality-is-already-priced ceiling: Sagarin divergence from the market line is a "
    "power-rating-vs-market-consensus disagreement, exactly the family the project already "
    "measured is bounded near zero -- this screen locates where in the divergence "
    "distribution any residual signal lives, it does not assume one exists"
)


def default_schedules() -> Path:
    """Resolve lazily so importing this module never requires local data."""
    return latest_schedules()


# ---------------------------------------------------------------------------
# Loading + join
# ---------------------------------------------------------------------------


def load_raw_schedule(schedules_path: Path) -> pd.DataFrame:
    """REG-only schedule, team codes normalized to current nflverse codes."""

    raw = pd.read_parquet(schedules_path)
    df = raw.loc[raw["game_type"] == "REG"].copy()
    df["season"] = pd.to_numeric(df["season"], errors="raise").astype(int)
    df["week"] = pd.to_numeric(df["week"], errors="raise").astype(int)
    df = df.loc[df["season"].between(SEASON_START, SEASON_END)].reset_index(drop=True)
    for column in ("home_team", "away_team"):
        df[column] = df[column].replace(TEAM_ABBREVIATION_ALIASES)
    return df


def load_sagarin_ratings(sagarin_root: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return (home_ratings, away_ratings, home_edge) keyed on (season, week[, team])."""

    av = pd.read_parquet(
        sagarin_root / "asof_tuesday_view.parquet",
        columns=[
            "season",
            "week",
            "team_code",
            "rating",
            "home_edge_rating",
            "has_tuesday_snapshot",
        ],
    )
    sag = av.loc[av["has_tuesday_snapshot"]].copy()
    sag["season"] = sag["season"].astype(int)
    sag["week"] = sag["week"].astype(int)

    home = sag[["season", "week", "team_code", "rating"]].rename(
        columns={"team_code": "home_team", "rating": "home_rating"}
    )
    away = sag[["season", "week", "team_code", "rating"]].rename(
        columns={"team_code": "away_team", "rating": "away_rating"}
    )
    home_edge = sag.groupby(["season", "week"])["home_edge_rating"].first().reset_index()
    return home, away, home_edge


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sagarin_source_provenance(sagarin_root: Path) -> dict[str, Any]:
    """Hash the exact consolidated Sagarin inputs used by a screen."""

    required = {
        "manifest": sagarin_root / "manifest.json",
        "asof_tuesday_view": sagarin_root / "asof_tuesday_view.parquet",
        "index": sagarin_root / "index.parquet",
        "captures_log": sagarin_root / "captures_log.parquet",
    }
    missing = [str(path) for path in required.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"incomplete Sagarin source snapshot; missing {missing}")
    manifest = json.loads(required["manifest"].read_text(encoding="utf-8"))
    return {
        "snapshot_id": sagarin_root.name,
        "fetched_at_utc": manifest.get("fetched_at_utc"),
        "captures_attempted": manifest.get("captures_attempted"),
        "captures_fetch_ok": manifest.get("captures_fetch_ok"),
        "captures_fetch_failed": manifest.get("captures_fetch_failed"),
        "captures_parse_ok": manifest.get("captures_parse_ok"),
        "index_rows": manifest.get("index_rows"),
        "sha256": {name: _sha256(path) for name, path in required.items()},
    }


def attach_sagarin(schedule: pd.DataFrame, sagarin_root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Join Sagarin ratings onto a REG schedule; return (joined, coverage_by_season).

    ``joined`` keeps every input row (left join) with home_rating/away_rating/
    home_edge_rating possibly null -- callers drop nulls explicitly and the
    coverage table below counts the drop per season honestly rather than
    silently trimming it.
    """

    home, away, home_edge = load_sagarin_ratings(sagarin_root)
    joined = schedule.merge(home, on=["season", "week", "home_team"], how="left")
    joined = joined.merge(away, on=["season", "week", "away_team"], how="left")
    joined = joined.merge(home_edge, on=["season", "week"], how="left")

    has_ratings = (
        joined["home_rating"].notna()
        & joined["away_rating"].notna()
        & joined["home_edge_rating"].notna()
    )
    joined["has_sagarin"] = has_ratings

    coverage = (
        joined.groupby("season")
        .agg(games=("game_id", "size"), games_with_sagarin=("has_sagarin", "sum"))
        .reset_index()
    )
    coverage["coverage_pct"] = (100.0 * coverage["games_with_sagarin"] / coverage["games"]).round(1)
    return joined, coverage


def add_divergence(df: pd.DataFrame, *, market_col: str, out_col: str) -> pd.DataFrame:
    work = df.copy()
    work["sagarin_implied_spread_home"] = (
        work["home_rating"] - work["away_rating"] + work["home_edge_rating"]
    )
    work[out_col] = work["sagarin_implied_spread_home"] - work[market_col]
    return work


def add_sagarin_side_cover(
    df: pd.DataFrame, *, divergence_col: str, home_cover_col: str
) -> pd.Series:
    divergence = df[divergence_col]
    home_cover = df[home_cover_col]
    return np.select(
        [divergence > 0.0, divergence < 0.0],
        [home_cover, 1.0 - home_cover],
        default=np.nan,
    )


# ---------------------------------------------------------------------------
# Populations
# ---------------------------------------------------------------------------


def build_close_population(
    schedules_path: Path, sagarin_root: Path
) -> tuple[pd.DataFrame, pd.DataFrame]:
    schedule = load_raw_schedule(schedules_path)
    schedule = add_ats_outcomes(schedule)
    joined, coverage = attach_sagarin(schedule, sagarin_root)

    pop = joined.loc[joined["has_sagarin"] & joined["home_cover"].notna()].copy()
    pop = add_divergence(pop, market_col="spread_line", out_col="divergence_close")
    pop["sagarin_side_cover"] = add_sagarin_side_cover(
        pop, divergence_col="divergence_close", home_cover_col="home_cover"
    )
    pop = pop.loc[pop["divergence_close"].notna() & pop["sagarin_side_cover"].notna()].copy()
    pop["week_block"] = pop["season"] * 100 + pop["week"]
    return pop.reset_index(drop=True), coverage


def build_open_population(
    schedules_path: Path, sagarin_root: Path, *, repo_root: Path
) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    raw = pd.read_parquet(schedules_path)
    raw["season"] = pd.to_numeric(raw["season"], errors="raise").astype(int)
    raw["week"] = pd.to_numeric(raw["week"], errors="raise").astype(int)
    for column in ("home_team", "away_team"):
        raw[column] = raw[column].replace(TEAM_ABBREVIATION_ALIASES)

    try:
        opener_features, note = _opener_graded_features(raw, repo_root=repo_root, market_root=None)
    except ExperimentRunnerError as error:
        empty = pd.DataFrame()
        return empty, empty, f"opener-grade population unavailable: {error}"

    joined, coverage = attach_sagarin(opener_features, sagarin_root)
    pop = joined.loc[joined["has_sagarin"] & joined["home_cover"].notna()].copy()
    pop = add_divergence(pop, market_col="spread_line", out_col="divergence_open")
    pop["sagarin_side_cover"] = add_sagarin_side_cover(
        pop, divergence_col="divergence_open", home_cover_col="home_cover"
    )
    pop = pop.loc[pop["divergence_open"].notna() & pop["sagarin_side_cover"].notna()].copy()
    pop["week_block"] = pop["season"] * 100 + pop["week"]
    return pop.reset_index(drop=True), coverage, note


def build_model_agreement_population(
    close_population: pd.DataFrame, *, artifacts_root: Path
) -> tuple[pd.DataFrame, str]:
    manifest = load_active_ats_model(artifacts_root)
    if manifest is None:
        return pd.DataFrame(), "artifacts/active_ats_model.json absent this session; cell skipped"

    margin_dir = active_artifact_path(artifacts_root, manifest, "historical_evaluation")
    if margin_dir is None:
        return (
            pd.DataFrame(),
            "active_ats_model.json has no historical_evaluation.artifact; cell skipped",
        )

    predictions_path = margin_dir / "predictions.parquet"
    if not predictions_path.is_file():
        return pd.DataFrame(), f"{predictions_path} not present locally; cell skipped"

    predictions = pd.read_parquet(predictions_path)
    active_method = manifest.get("method", "market_residual")
    model_rows = predictions.loc[predictions["method"] == active_method].copy()
    if model_rows.empty:
        return (
            pd.DataFrame(),
            f"no predictions.parquet rows for method={active_method!r}; cell skipped",
        )
    model_rows = model_rows[["game_id", "season", "week", "home_cover_probability"]].dropna(
        subset=["home_cover_probability"]
    )
    model_rows["model_pick_home"] = model_rows["home_cover_probability"] > 0.5

    merged = close_population.merge(model_rows, on="game_id", how="inner", suffixes=("", "_model"))
    merged["sagarin_side_home"] = merged["divergence_close"] > 0.0
    merged["agree"] = merged["sagarin_side_home"] == merged["model_pick_home"]
    merged["model_correct"] = np.where(
        merged["model_pick_home"], merged["home_cover"], 1.0 - merged["home_cover"]
    )
    note = (
        f"active model method={active_method!r}, walk-forward artifact={margin_dir.name}, "
        f"season range {int(model_rows['season'].min())}-{int(model_rows['season'].max())} "
        "(narrower than the 2010-2025 close population -- the walk-forward evaluation's own "
        "min_train_games cutoff, not a choice made in this script)"
    )
    return merged.reset_index(drop=True), note


# ---------------------------------------------------------------------------
# Week-blocked bootstrap
# ---------------------------------------------------------------------------


def block_bootstrap_single(
    df: pd.DataFrame, *, value_col: str, block_col: str, samples: int, seed: int
) -> np.ndarray:
    """Vectorized week-blocked bootstrap of ``(mean(value_col) - 0.5) * 100``."""

    blocks, block_index = np.unique(df[block_col].to_numpy(), return_inverse=True)
    block_index = np.asarray(block_index).reshape(-1)
    block_count = len(blocks)
    values = df[value_col].to_numpy(dtype=np.float64)

    sums = np.bincount(block_index, weights=values, minlength=block_count).astype(np.float64)
    counts = np.bincount(block_index, minlength=block_count).astype(np.float64)

    rng = np.random.default_rng(seed)
    drawn = rng.multinomial(block_count, np.full(block_count, 1.0 / block_count), size=samples)
    resampled_count = drawn @ counts
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = (drawn @ sums) / resampled_count
    valid = resampled_count > 0
    return (mean[valid] - 0.5) * 100.0


def summarize_single(df: pd.DataFrame, *, samples: int, seed: int) -> dict[str, Any]:
    n_total = len(df)
    if n_total == 0:
        return {"n_total": 0, "insufficient_data": True}
    cover_rate = float(df["sagarin_side_cover"].mean())
    draws = block_bootstrap_single(
        df, value_col="sagarin_side_cover", block_col="week_block", samples=samples, seed=seed
    )
    dropped = samples - len(draws)
    lower, upper = np.quantile(draws, [0.025, 0.975]) if len(draws) else (np.nan, np.nan)
    return {
        "n_total": n_total,
        "n_blocks": int(df["week_block"].nunique()),
        "sagarin_side_cover_rate": cover_rate,
        "effect_pts": (cover_rate - 0.5) * 100.0,
        "ci95": [float(lower), float(upper)],
        "probability_positive": float(np.mean(draws > 0)) if len(draws) else float("nan"),
        "bootstrap_samples": samples,
        "dropped_draws": int(dropped),
        "insufficient_data": False,
    }


def summarize_two_group(
    df: pd.DataFrame, *, flag_col: str, value_col: str, samples: int, seed: int
) -> dict[str, Any]:
    n_total = len(df)
    n_flag = int(df[flag_col].sum())
    n_complement = n_total - n_flag
    if n_flag == 0 or n_complement == 0:
        return {
            "n_total": n_total,
            "n_flag": n_flag,
            "n_complement": n_complement,
            "insufficient_data": True,
        }

    subset_mean = float(df.loc[df[flag_col], value_col].mean())
    complement_mean = float(df.loc[~df[flag_col], value_col].mean())
    gap_pts = (subset_mean - complement_mean) * 100.0

    draws = block_bootstrap_two_group(
        df,
        flag_col=flag_col,
        value_col=value_col,
        block_col="week_block",
        samples=samples,
        seed=seed,
    )
    dropped = samples - len(draws)
    lower, upper = np.quantile(draws, [0.025, 0.975]) if len(draws) else (np.nan, np.nan)
    return {
        "n_total": n_total,
        "n_flag": n_flag,
        "n_complement": n_complement,
        "n_blocks": int(df["week_block"].nunique()),
        "agree_accuracy": subset_mean,
        "disagree_accuracy": complement_mean,
        "effect_pts": gap_pts,
        "ci95": [float(lower), float(upper)],
        "probability_positive": float(np.mean(draws > 0)) if len(draws) else float("nan"),
        "bootstrap_samples": samples,
        "dropped_draws": int(dropped),
        "insufficient_data": False,
    }


# ---------------------------------------------------------------------------
# Cells
# ---------------------------------------------------------------------------


def build_cells(
    close_pop: pd.DataFrame,
    open_pop: pd.DataFrame,
    agreement_pop: pd.DataFrame,
    *,
    samples: int,
    seed: int,
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []

    def _season_range(subset: pd.DataFrame) -> tuple[int | None, int | None]:
        if subset.empty:
            return None, None
        return int(subset["season"].min()), int(subset["season"].max())

    def add_single(name: str, description: str, subset: pd.DataFrame) -> None:
        summary = summarize_single(subset, samples=samples, seed=seed)
        season_start, season_end = _season_range(subset)
        results.append(
            {
                "name": name,
                "description": description,
                "kind": "single",
                "season_start": season_start,
                "season_end": season_end,
                **summary,
            }
        )

    # (a) large divergence, close + open grade
    large_close = close_pop.loc[close_pop["divergence_close"].abs() >= LARGE_DIVERGENCE_THRESHOLD]
    add_single(
        "sagarin_battery_large_divergence_close",
        f"|divergence_close| >= {LARGE_DIVERGENCE_THRESHOLD} pts, close grade, REG 2010-2025",
        large_close,
    )
    large_open = open_pop.loc[open_pop["divergence_open"].abs() >= LARGE_DIVERGENCE_THRESHOLD]
    add_single(
        "sagarin_battery_large_divergence_open",
        f"|divergence_open| >= {LARGE_DIVERGENCE_THRESHOLD} pts, opener grade, "
        "REG 2020-2025 paired subset",
        large_open,
    )

    # (c) top-decile sub-cell, threshold computed within each population separately
    if len(close_pop):
        close_decile_threshold = float(close_pop["divergence_close"].abs().quantile(0.90))
        top_close = close_pop.loc[close_pop["divergence_close"].abs() >= close_decile_threshold]
        add_single(
            "sagarin_battery_top_decile_close",
            f"|divergence_close| >= its own 90th pct "
            f"({close_decile_threshold:.3f} pts), close grade",
            top_close,
        )
    if len(open_pop):
        open_decile_threshold = float(open_pop["divergence_open"].abs().quantile(0.90))
        top_open = open_pop.loc[open_pop["divergence_open"].abs() >= open_decile_threshold]
        add_single(
            "sagarin_battery_top_decile_open",
            f"|divergence_open| >= its own 90th pct "
            f"({open_decile_threshold:.3f} pts), opener grade",
            top_open,
        )

    # (d) era splits of the close-grade large-divergence cell
    for label, lo, hi in ERA_SPLITS:
        subset = large_close.loc[large_close["season"].between(lo, hi)]
        add_single(
            f"sagarin_battery_large_divergence_era_{label}",
            f"sagarin_battery_large_divergence_close restricted to season {lo}-{hi}",
            subset,
        )

    # (b) divergence-sign agreement with the active model's forced pick
    if len(agreement_pop):
        summary = summarize_two_group(
            agreement_pop, flag_col="agree", value_col="model_correct", samples=samples, seed=seed
        )
        season_start, season_end = _season_range(agreement_pop)
        results.append(
            {
                "name": "sagarin_battery_model_agreement_close",
                "description": (
                    "active-model forced-pick accuracy when Sagarin's side agrees with the "
                    "model's pick minus when it disagrees (close grade)"
                ),
                "kind": "two_group",
                "season_start": season_start,
                "season_end": season_end,
                **summary,
            }
        )
    else:
        results.append(
            {
                "name": "sagarin_battery_model_agreement_close",
                "description": "active-model agreement cell",
                "kind": "two_group",
                "season_start": None,
                "season_end": None,
                "n_total": 0,
                "insufficient_data": True,
            }
        )

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--schedules", type=Path, default=None)
    parser.add_argument(
        "--sagarin-root",
        type=Path,
        default=REPO / "data" / "raw" / "sagarin" / DEFAULT_SAGARIN_SNAPSHOT,
    )
    parser.add_argument("--artifacts-root", type=Path, default=REPO / "artifacts")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--samples", type=int, default=BOOTSTRAP_SAMPLES)
    parser.add_argument("--seed", type=int, default=BOOTSTRAP_SEED)
    args = parser.parse_args()
    if args.schedules is None:
        args.schedules = default_schedules()

    started = time.time()
    timestamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    output_dir: Path = args.output or (
        REPO / "artifacts" / "sagarin_divergence_battery" / timestamp
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== ceiling caveat: {CEILING_CAVEAT} ===\n")
    print(f"=== loading schedule {args.schedules} + Sagarin {args.sagarin_root} ===")

    close_pop, close_coverage = build_close_population(args.schedules, args.sagarin_root)
    print(f"close-grade population: {len(close_pop)} REG games, 2010-2025")
    print("\njoin coverage by season (close-grade schedule x Sagarin Tuesday-asof):")
    print(close_coverage.to_string(index=False))

    open_pop, open_coverage, open_note = build_open_population(
        args.schedules, args.sagarin_root, repo_root=REPO
    )
    print(f"\nopener-grade population: {len(open_pop)} REG games -- {open_note}")
    if len(open_coverage):
        print("\njoin coverage by season (opener-grade paired schedule x Sagarin Tuesday-asof):")
        print(open_coverage.to_string(index=False))

    agreement_pop, agreement_note = build_model_agreement_population(
        close_pop, artifacts_root=args.artifacts_root
    )
    print(f"\nmodel-agreement population: {len(agreement_pop)} games -- {agreement_note}")

    cells = build_cells(close_pop, open_pop, agreement_pop, samples=args.samples, seed=args.seed)

    expected_names = {
        "sagarin_battery_large_divergence_close",
        "sagarin_battery_large_divergence_open",
        "sagarin_battery_top_decile_close",
        "sagarin_battery_top_decile_open",
        "sagarin_battery_large_divergence_era_2010_2016",
        "sagarin_battery_large_divergence_era_2017_2025",
        "sagarin_battery_model_agreement_close",
    }
    actual_names = {cell["name"] for cell in cells}
    assert actual_names == expected_names, (
        f"cell name mismatch vs predeclaration: missing={expected_names - actual_names} "
        f"extra={actual_names - expected_names}"
    )

    print("\n=== cells ===")
    for cell in cells:
        if cell.get("insufficient_data"):
            print(f"  {cell['name']}: insufficient data (n_total={cell.get('n_total', 0)})")
            continue
        print(
            f"  {cell['name']}: n={cell['n_total']} n_blocks={cell['n_blocks']} "
            f"effect={cell['effect_pts']:+.4f}pts 95%[{cell['ci95'][0]:+.4f},"
            f"{cell['ci95'][1]:+.4f}] P+={cell['probability_positive']:.4f}"
        )

    configuration = {
        "command": "sagarin-divergence-battery",
        "schedules": str(args.schedules),
        "sagarin_root": str(args.sagarin_root),
        "bootstrap_samples": args.samples,
        "bootstrap_seed": args.seed,
        "season_start": SEASON_START,
        "season_end": SEASON_END,
        "large_divergence_threshold": LARGE_DIVERGENCE_THRESHOLD,
    }
    payload = {
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(started)),
        "elapsed_seconds": time.time() - started,
        "ceiling_caveat": CEILING_CAVEAT,
        "predeclaration": (
            "scratchpad/sagarin_divergence/predeclaration.json (frozen before scoring)"
        ),
        "bootstrap_samples": args.samples,
        "bootstrap_seed": args.seed,
        "close_population_n": len(close_pop),
        "close_join_coverage_by_season": close_coverage.to_dict(orient="records"),
        "open_population_n": len(open_pop),
        "open_population_note": open_note,
        "open_join_coverage_by_season": open_coverage.to_dict(orient="records")
        if len(open_coverage)
        else [],
        "model_agreement_population_n": len(agreement_pop),
        "model_agreement_note": agreement_note,
        "sagarin_source": sagarin_source_provenance(args.sagarin_root),
        "results": cells,
        "provenance": artifact_provenance(configuration, args.schedules, project_root=REPO),
    }
    write_experiment_artifact(
        output_dir,
        "results.json",
        payload,
        command="sagarin-divergence-battery",
        metrics=payload,
        notes=(
            "Measure-only predeclared Sagarin-divergence screen (7 cells); mined/first "
            "measurement, every cell predeclared to record unresolved_below_power via a "
            "separate nfl-ats weak-signals record call regardless of interval shape "
            "(AGENTS.md). team-quality-is-already-priced ceiling applies."
        ),
    )
    print(f"\nwrote {output_dir / 'results.json'}")


if __name__ == "__main__":
    main()
