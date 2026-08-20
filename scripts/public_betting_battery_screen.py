"""Public-betting fade/follow battery (measure-only, mined battery on a sparse archive).

Predeclared in full in `docs/public_betting_battery_predeclaration.md`
**before** this script was run or any effect was computed -- read that
document for the frozen population construction, cell definitions, and
bootstrap spec; this docstring summarizes rather than repeats.

**Mined battery, not a confirmation look.** Coverage of the underlying
archive (`docs/public_betting_sourcing.md` section 5) tops out around 34%
of REG games in the best-covered season, so every cell here is
exploratory/backfill-quality by construction. Nothing here calls
`nfl_ats.rotation.assign_window`/`record_look`; no `registry/rotation*`
file is touched. `nfl-ats weak-signals record` commands are PROPOSED
(printed + written to metadata.json) here; this script does not execute
them -- a separate pass runs the proposed commands, matching
`scripts/odds_microstructure_battery.py`'s own propose-don't-execute
convention.

Per AGENTS.md's binding closing-grounds taxonomy (restated in the
predeclaration doc verbatim): an interval containing zero is never grounds
to reject a cell; every cell defaults to `unresolved_below_power` unless
BOTH the week- and season-blocked 95% intervals sit entirely below zero (a
genuinely resolved wrong sign), in which case `refuted_mechanism` /
`wrong_sign_resolved` is proposed instead. No cell here has a positive
control, so `bounded_by_control` is never proposed.

Writes `*.parquet` / `cells_summary.csv` / `metadata.json` to
`artifacts/public_betting_battery/<run_id>/`.

Run:  .\\.tools\\uv.exe run --no-sync python scripts/public_betting_battery_screen.py
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from nfl_ats.clv import build_pairing_table, pick_correct, week_blocked_bootstrap
from nfl_ats.io import atomic_json, atomic_parquet, run_id
from nfl_ats.modeling import regular_season_rows
from nfl_ats.odds_backfill import HISTORICAL_CAPTURE_KIND

REPO = Path(__file__).resolve().parents[1]
DEFAULT_ARCHIVE = REPO / "data/raw/public_betting/20260820T111148Z/actionnetwork/index.parquet"
DEFAULT_FEATURES = REPO / "data/processed/game_features.parquet"
DEFAULT_MARKET_ROOT = REPO / "data/market/raw"
DEFAULT_PREDICTIONS = REPO / "artifacts/margins/20260820T004951Z/predictions.parquet"
DEFAULT_OUTPUT_ROOT = REPO / "artifacts/public_betting_battery"

BOOTSTRAP_SAMPLES = 20_000
BOOTSTRAP_SEED = 20260818

FADE_THRESHOLD = 70.0
DIVERGENCE_THRESHOLD = 15.0
KICKOFF_MATCH_HOURS = 72.0
BASE_SEASON_START = 2018
BASE_SEASON_END = 2025
OPENER_SEASON_START = 2020
OPENER_SEASON_END = 2025

TEAM_ALIASES = {
    "OAK": "LV",
    "SD": "LAC",
    "STL": "LA",
    "LAR": "LA",
    "JAC": "JAX",
    "WSH": "WAS",
}


def _git_revision() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return "unknown"


def _normalize_team(abbr: str) -> str:
    return TEAM_ALIASES.get(abbr, abbr)


def _match_to_schedule(archive: pd.DataFrame, schedule: pd.DataFrame) -> pd.DataFrame:
    """Row-level match of archive captures to REG schedule games.

    Identical method to `scripts/ingest_public_betting.py`'s
    `build_coverage_report`: team-pair merge, then restrict to a kickoff
    within `KICKOFF_MATCH_HOURS` of the site's own `start_time_utc`,
    keeping the closest match per (capture_ts, site_game_id) if more than
    one schedule row is within range.
    """

    working = archive.copy()
    working["away_team"] = working["away_team"].map(_normalize_team)
    working["home_team"] = working["home_team"].map(_normalize_team)
    working["start_time_utc"] = pd.to_datetime(working["start_time_utc"], utc=True, errors="coerce")

    sched = schedule.loc[schedule["game_type"].eq("REG")][
        [
            "game_id",
            "season",
            "week",
            "away_team",
            "home_team",
            "kickoff",
            "spread_line",
            "result",
            "ats_margin",
            "home_cover",
        ]
    ].copy()
    sched["kickoff"] = pd.to_datetime(sched["kickoff"], utc=True)

    merged = working.merge(
        sched, on=["away_team", "home_team"], how="inner", suffixes=("", "_sched")
    )
    merged["kickoff_delta_hours"] = (
        merged["kickoff"] - merged["start_time_utc"]
    ).dt.total_seconds().abs() / 3600.0
    matched = merged.loc[merged["kickoff_delta_hours"] <= KICKOFF_MATCH_HOURS].copy()
    matched = matched.sort_values("kickoff_delta_hours").drop_duplicates(
        subset=["capture_ts", "site_game_id", "game_id"]
    )
    return matched


def _latest_pregame_capture(matched: pd.DataFrame) -> pd.DataFrame:
    """One row per game_id: the pregame capture with the max capture_ts."""

    pregame = matched.loc[matched["capture_ts"] < matched["kickoff"]].copy()
    if pregame.empty:
        return pregame
    latest = (
        pregame.sort_values("capture_ts")
        .groupby("game_id", as_index=False)
        .tail(1)
        .reset_index(drop=True)
    )
    latest["staleness_hours"] = (
        latest["kickoff"] - latest["capture_ts"]
    ).dt.total_seconds() / 3600.0
    return latest


def _staleness_summary(latest: pd.DataFrame) -> dict[str, float]:
    hours = pd.to_numeric(latest["staleness_hours"], errors="coerce").dropna()
    if hours.empty:
        return {}
    quantiles = hours.quantile([0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0])
    return {
        "n_games": len(hours),
        "min_hours": float(quantiles.loc[0.0]),
        "p10_hours": float(quantiles.loc[0.10]),
        "p25_hours": float(quantiles.loc[0.25]),
        "median_hours": float(quantiles.loc[0.5]),
        "p75_hours": float(quantiles.loc[0.75]),
        "p90_hours": float(quantiles.loc[0.90]),
        "max_hours": float(quantiles.loc[1.0]),
        "mean_hours": float(hours.mean()),
    }


def _accuracy_metric_fn(correct_col: str):
    def _metric(rows: pd.DataFrame) -> dict[str, float]:
        values = pd.to_numeric(rows[correct_col], errors="coerce").dropna()
        accuracy = float(values.mean()) if len(values) else float("nan")
        return {"accuracy_minus_half": accuracy - 0.5}

    return _metric


def _group_diff_metric_fn(correct_col: str, group_col: str):
    def _metric(rows: pd.DataFrame) -> dict[str, float]:
        values = pd.to_numeric(rows[correct_col], errors="coerce")
        group = rows[group_col].astype(bool)
        against = values.loc[group].dropna()
        withh = values.loc[~group].dropna()
        against_mean = float(against.mean()) if len(against) else float("nan")
        with_mean = float(withh.mean()) if len(withh) else float("nan")
        return {"against_minus_with": against_mean - with_mean}

    return _metric


def _both_blocks(frame: pd.DataFrame, metric_fn) -> dict[str, pd.DataFrame]:
    return {
        "week": week_blocked_bootstrap(
            frame, metric_fn, block="week", samples=BOOTSTRAP_SAMPLES, seed=BOOTSTRAP_SEED
        ),
        "season": week_blocked_bootstrap(
            frame, metric_fn, block="season", samples=BOOTSTRAP_SAMPLES, seed=BOOTSTRAP_SEED
        ),
    }


def _row(ci: dict[str, pd.DataFrame], metric: str, n: int) -> dict[str, Any]:
    week_row = ci["week"].loc[ci["week"]["metric"].eq(metric)].iloc[0]
    season_row = ci["season"].loc[ci["season"]["metric"].eq(metric)].iloc[0]
    week_resolved_wrong_sign = bool(week_row["upper"] < 0.0)
    season_resolved_wrong_sign = bool(season_row["upper"] < 0.0)
    return {
        "n": n,
        "estimate": float(week_row["estimate"]),
        "week_lower": float(week_row["lower"]),
        "week_upper": float(week_row["upper"]),
        "week_probability_positive": float(week_row["probability_positive"]),
        "season_lower": float(season_row["lower"]),
        "season_upper": float(season_row["upper"]),
        "season_probability_positive": float(season_row["probability_positive"]),
        # Only true when BOTH blocks' whole interval sits below zero -- the
        # sole condition (AGENTS.md) under which `refuted_mechanism` /
        # `wrong_sign_resolved` may be proposed instead of
        # `unresolved_below_power`. An interval crossing zero, in either
        # block, is never proposed as a closure.
        "resolved_wrong_sign": week_resolved_wrong_sign and season_resolved_wrong_sign,
    }


def _print_accuracy_row(row: dict[str, Any]) -> None:
    accuracy = 0.5 + row["estimate"]
    p_plus = row["week_probability_positive"]
    print(
        f"  accuracy={accuracy:.4f} week_P+={p_plus:.3f} "
        f"[{row['week_lower']:.4f}, {row['week_upper']:.4f}]"
    )


def _cell_record(
    cells: list[dict[str, Any]],
    *,
    cell_name: str,
    row: dict[str, Any],
    season_start: int,
    season_end: int,
    description: str,
) -> None:
    p_plus = row["week_probability_positive"]
    classification = "unresolved_below_power"
    closing_ground = None
    if row["resolved_wrong_sign"]:
        classification = "refuted_mechanism"
        closing_ground = "wrong_sign_resolved"
    effect_points = row["estimate"] * 100.0
    interval_low_points = row["week_lower"] * 100.0
    interval_high_points = row["week_upper"] * 100.0
    cells.append(
        {
            "cell": cell_name,
            "description": description,
            "season_start": season_start,
            "season_end": season_end,
            **row,
            "classification": classification,
            "closing_ground": closing_ground,
            "proposed_command": (
                "nfl-ats weak-signals record "
                f"--name {cell_name} "
                f'--description "{description}" '
                "--source artifacts/public_betting_battery/<run_id>/cells_summary.csv "
                f"--effect {effect_points:.6f} --effect-units accuracy_points "
                f"--classification {classification} --league nfl "
                f"--season-start {season_start} --season-end {season_end} "
                f"--interval-low {interval_low_points:.6f} "
                f"--interval-high {interval_high_points:.6f} "
                f"--probability-positive {p_plus:.6f} --sample-games {row['n']} "
                + (f"--closing-ground {closing_ground} " if closing_ground else "")
                + '--notes "public_betting fade/follow battery, mined/exploratory, '
                'sparse archive coverage"'
            ),
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument("--features", type=Path, default=DEFAULT_FEATURES)
    parser.add_argument("--market-root", type=Path, default=DEFAULT_MARKET_ROOT)
    parser.add_argument("--predictions", type=Path, default=DEFAULT_PREDICTIONS)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    args = parser.parse_args()

    cells: list[dict[str, Any]] = []

    print(f"Loading archive: {args.archive}")
    archive = pd.read_parquet(args.archive)
    print(f"Loading features/schedule: {args.features}")
    features = pd.read_parquet(args.features)
    regular = regular_season_rows(features)

    matched = _match_to_schedule(archive, regular)
    print(f"Matched archive rows (REG, <=72h): {len(matched)}")
    latest = _latest_pregame_capture(matched)
    latest = latest.loc[
        latest["season"].between(BASE_SEASON_START, BASE_SEASON_END)
        & latest["result"].notna()
        & latest["spread_line"].notna()
    ].copy()
    print(f"Base population (one row per game, latest pregame capture): {len(latest)}")

    staleness = _staleness_summary(latest)
    print(f"Staleness (hours before kickoff) summary: {staleness}")

    # ==================================================================
    # Cell A: fade-heavy-public
    # ==================================================================
    print("\n=== Cell A: fade-heavy-public ===")
    both_pct = latest["spread_home_bet_pct"].notna() & latest["spread_away_bet_pct"].notna()
    heavy = both_pct & (
        latest["spread_home_bet_pct"].ge(FADE_THRESHOLD)
        | latest["spread_away_bet_pct"].ge(FADE_THRESHOLD)
    )
    a_pop = latest.loc[heavy].copy()
    a_pop["public_side_home"] = a_pop["spread_home_bet_pct"].gt(a_pop["spread_away_bet_pct"])
    a_pop["fade_side_home"] = ~a_pop["public_side_home"]
    a_pop["correct"] = pick_correct(a_pop["fade_side_home"], a_pop["ats_margin"])
    a_pop = a_pop.loc[a_pop["ats_margin"].ne(0.0)].dropna(subset=["correct"])
    print(f"A.1 close-grade population: {len(a_pop)}")
    if len(a_pop):
        ci = _both_blocks(a_pop, _accuracy_metric_fn("correct"))
        row = _row(ci, "accuracy_minus_half", len(a_pop))
        _print_accuracy_row(row)
        _cell_record(
            cells,
            cell_name="public_betting_battery_fade_heavy_public_close",
            row=row,
            season_start=BASE_SEASON_START,
            season_end=BASE_SEASON_END,
            description=(
                "public betting battery: fade the side with >=70% bet share, "
                "graded at close, 2018-2025"
            ),
        )

    # Cell A.2: opener-grade variant, 2020-2025 subset
    print("Building tue_open pairing for the opener-grade variant")
    tue_pairing = build_pairing_table(
        args.market_root,
        capture_kind=HISTORICAL_CAPTURE_KIND,
        labels=("tue_open",),
        schedule=regular[["game_id", "season", "week"]].drop_duplicates("game_id"),
    )
    tue_open = tue_pairing[["game_id", "home_spread"]].rename(
        columns={"home_spread": "tue_open_home_spread"}
    )
    a_opener = a_pop.merge(tue_open, on="game_id", how="inner")
    a_opener = a_opener.loc[
        a_opener["season"].between(OPENER_SEASON_START, OPENER_SEASON_END)
        & a_opener["tue_open_home_spread"].notna()
    ].copy()
    a_opener["margin_vs_open"] = a_opener["result"] - a_opener["tue_open_home_spread"]
    a_opener["correct_opener"] = pick_correct(
        a_opener["fade_side_home"], a_opener["margin_vs_open"]
    )
    a_opener = a_opener.loc[a_opener["margin_vs_open"].ne(0.0)].dropna(subset=["correct_opener"])
    print(f"A.2 opener-grade population (2020-2025): {len(a_opener)}")
    if len(a_opener):
        ci = _both_blocks(a_opener, _accuracy_metric_fn("correct_opener"))
        row = _row(ci, "accuracy_minus_half", len(a_opener))
        _print_accuracy_row(row)
        _cell_record(
            cells,
            cell_name="public_betting_battery_fade_heavy_public_opener",
            row=row,
            season_start=OPENER_SEASON_START,
            season_end=OPENER_SEASON_END,
            description=(
                "public betting battery: fade the side with >=70% bet share, "
                "graded at tue_open, 2020-2025"
            ),
        )
    else:
        print("  A.2: empty population (no overlap between archive readings and tue_open archive)")

    # ==================================================================
    # Cell B: follow-sharp-divergence (era2 only)
    # ==================================================================
    print("\n=== Cell B: follow-sharp-divergence (era2) ===")
    era2 = latest.loc[latest["era"].eq("era2_scoreboard_response")].copy()
    era2 = era2.loc[
        era2["spread_home_bet_pct"].notna()
        & era2["spread_away_bet_pct"].notna()
        & era2["spread_home_money_pct"].notna()
        & era2["spread_away_money_pct"].notna()
    ].copy()
    era2["gap_home"] = era2["spread_home_money_pct"] - era2["spread_home_bet_pct"]
    era2["gap_away"] = era2["spread_away_money_pct"] - era2["spread_away_bet_pct"]
    money_side_home = np.select(
        [era2["gap_home"].ge(DIVERGENCE_THRESHOLD), era2["gap_away"].ge(DIVERGENCE_THRESHOLD)],
        [True, False],
        default=np.nan,
    )
    era2["money_side_home"] = money_side_home
    b_pop = era2.loc[era2["money_side_home"].notna()].copy()
    b_pop["money_side_home"] = b_pop["money_side_home"].astype(bool)
    b_pop["correct"] = pick_correct(b_pop["money_side_home"], b_pop["ats_margin"])
    b_pop = b_pop.loc[b_pop["ats_margin"].ne(0.0)].dropna(subset=["correct"])
    print(f"B.1 population: {len(b_pop)}")
    if len(b_pop):
        ci = _both_blocks(b_pop, _accuracy_metric_fn("correct"))
        row = _row(ci, "accuracy_minus_half", len(b_pop))
        _print_accuracy_row(row)
        _cell_record(
            cells,
            cell_name="public_betting_battery_sharp_divergence_close",
            row=row,
            season_start=BASE_SEASON_START,
            season_end=BASE_SEASON_END,
            description=(
                "public betting battery: follow the side with money%-bet% gap "
                ">=15pts, era2 only, graded at close"
            ),
        )
    else:
        print("  B.1: empty population")

    # ==================================================================
    # Cell C: public-vs-our-model interaction
    # ==================================================================
    print("\n=== Cell C: public-vs-our-model interaction ===")
    predictions = pd.read_parquet(args.predictions)
    model = predictions.loc[
        predictions["method"].eq("market_residual") & predictions["model_name"].eq("ridge")
    ][["game_id", "home_cover_probability", "ats_margin"]].rename(
        columns={"ats_margin": "model_ats_margin"}
    )
    c_pop = a_pop.merge(model, on="game_id", how="inner")
    print(f"C population (fade-heavy games with a production model pick): {len(c_pop)}")
    if len(c_pop):
        c_pop["model_pick_home"] = c_pop["home_cover_probability"].ge(0.5)
        c_pop["model_correct"] = pick_correct(c_pop["model_pick_home"], c_pop["model_ats_margin"])
        c_pop["against"] = c_pop["public_side_home"].ne(c_pop["model_pick_home"])
        c_pop = c_pop.loc[c_pop["model_ats_margin"].ne(0.0)].dropna(subset=["model_correct"])
        n_against = int(c_pop["against"].sum())
        n_with = int((~c_pop["against"]).sum())
        print(f"  against={n_against} with={n_with}")

        against_pop = c_pop.loc[c_pop["against"]]
        if len(against_pop):
            ci = _both_blocks(against_pop, _accuracy_metric_fn("model_correct"))
            row = _row(ci, "accuracy_minus_half", len(against_pop))
            print(
                f"  C.1 against-subset accuracy={0.5 + row['estimate']:.4f} "
                f"week_P+={row['week_probability_positive']:.3f}"
            )
            _cell_record(
                cells,
                cell_name="public_betting_battery_model_interaction_against",
                row=row,
                season_start=BASE_SEASON_START,
                season_end=BASE_SEASON_END,
                description=(
                    "public betting battery: production model's forced-pick "
                    "accuracy when public is heavy (>=70%) on the side the "
                    "model did NOT pick, graded at close"
                ),
            )
        else:
            print("  C.1: empty against-subset")

        if len(c_pop) and n_against and n_with:
            ci = _both_blocks(c_pop, _group_diff_metric_fn("model_correct", "against"))
            row = _row(ci, "against_minus_with", len(c_pop))
            print(
                f"  C.2 against-minus-with diff={row['estimate']:.4f} "
                f"week_P+={row['week_probability_positive']:.3f}"
            )
            _cell_record(
                cells,
                cell_name="public_betting_battery_model_interaction_diff",
                row=row,
                season_start=BASE_SEASON_START,
                season_end=BASE_SEASON_END,
                description=(
                    "public betting battery: production model's forced-pick "
                    "accuracy when public is heavy against the model's pick "
                    "minus accuracy when public is heavy WITH it, graded at close"
                ),
            )
        else:
            print("  C.2: cannot form a paired diff (one side empty)")
    else:
        print("  C: empty population")

    # ------------------------------------------------------------------
    # Write artifacts
    # ------------------------------------------------------------------
    output_dir = args.output_root / run_id()
    atomic_parquet(latest, output_dir / "base_population.parquet")
    atomic_parquet(a_pop, output_dir / "cell_a_close.parquet")
    if len(a_opener):
        atomic_parquet(a_opener, output_dir / "cell_a_opener.parquet")
    atomic_parquet(b_pop, output_dir / "cell_b.parquet")
    atomic_parquet(c_pop, output_dir / "cell_c.parquet")
    cells_frame = pd.DataFrame(cells)
    atomic_parquet(cells_frame, output_dir / "cells_summary.parquet")
    (output_dir / "cells_summary.csv").parent.mkdir(parents=True, exist_ok=True)
    cells_frame.to_csv(output_dir / "cells_summary.csv", index=False)

    metadata: dict[str, Any] = {
        "predeclaration": "docs/public_betting_battery_predeclaration.md",
        "diagnostic_not_confirmation": True,
        "rotation_registry_touched": False,
        "registry_files_written": [],
        "git_revision": _git_revision(),
        "bootstrap_samples": BOOTSTRAP_SAMPLES,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "matched_archive_rows": len(matched),
        "base_population_games": len(latest),
        "staleness_hours_summary": staleness,
        "cells": cells,
    }
    atomic_json(metadata, output_dir / "metadata.json")
    print(f"\nartifacts: {output_dir}")
    print(f"\n{len(cells)} cells computed; proposed weak-signals record commands:")
    for cell in cells:
        p_plus = cell["week_probability_positive"]
        print(f"  [{cell['classification']}, P+={p_plus:.3f}] {cell['cell']}")
        print(f"    {cell['proposed_command']}")


if __name__ == "__main__":
    main()
