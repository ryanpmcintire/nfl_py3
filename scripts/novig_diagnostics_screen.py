"""MKT-03 no-vig market probability diagnostics.

Research item: ``docs/novig_diagnostics.md``. This is a DIAGNOSTIC, not a
rotation-family confirmation look: it produces no pick, gates no modeling
decision, and never calls ``nfl_ats.rotation.assign_window``/``record_look``
(see ``docs/novig_diagnostics.md`` section "What this is and is not" and
rule 8 of ``docs/rotation_registry.md``, "CFB and non-reserved seasons stay
free... the registry governs NFL confirmation looks only"). Consuming any of
this output as a model feature is explicitly out of scope -- see
``nfl_ats.novig``'s module docstring.

Reads only committed local snapshots under ``data/market/raw/`` via
``nfl_ats.clv``'s loaders -- never the Odds API (the paid, owner-managed
quota is never touched by this script).

Computes:
  1. Per-(game, decision label) no-vig ATS (spread) probability and hold at
     the Tuesday opener (primary) and the two store close labels
     (comparison arm), from ``nfl_ats.clv.spread_price_consensus_table``'s
     consensus spread PRICE joined to ``nfl_ats.clv.build_pairing_table``'s
     consensus spread LINE.
  2. Per-(game, decision label) no-vig moneyline (SU) probability and hold,
     from ``build_pairing_table``'s existing ``home_moneyline``/
     ``away_moneyline`` columns, at every decision label the archive has
     moneyline coverage for.
  3. A favourite-longshot calibration bucket table for both the ATS arm
     (primary-goal-relevant) and the SU arm (secondary), each at the
     Tuesday opener, with week- and season-blocked bootstrap intervals on
     each bucket's calibration gap.
  4. Market hold trended by decision label, as a liquidity/agreement
     diagnostic (thin coverage at a label is a caution flag on that
     read, not itself accuracy-moving).

Writes ``spread_novig.parquet``, ``moneyline_novig.parquet``,
``calibration_buckets.parquet``, and ``metadata.json`` to
``artifacts/novig_diagnostics/<run_id>/``, matching the project's existing
artifact convention (e.g. ``artifacts/opener_evaluation/<run_id>/``).
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from nfl_ats.clv import build_pairing_table, load_decision_quotes, spread_price_consensus_table
from nfl_ats.io import atomic_json, atomic_parquet, run_id
from nfl_ats.modeling import regular_season_rows
from nfl_ats.novig import (
    bootstrap_calibration_gaps,
    calibration_bucket_edges,
    favourite_longshot_calibration,
    moneyline_novig_probabilities,
    spread_novig_probabilities,
)
from nfl_ats.odds_backfill import HISTORICAL_CAPTURE_KIND

REPO = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = REPO / "data/market/raw"
DEFAULT_FEATURES = REPO / "data/processed/game_features_player.parquet"
DEFAULT_OUTPUT_ROOT = REPO / "artifacts/novig_diagnostics"

# The 6 historical decision labels the archive carries h2h (moneyline) rows
# at (measured against the local archive; see docs/novig_diagnostics.md).
# Restricting to these -- rather than passing labels=None -- keeps this
# script from also loading the archive's 6,966 intraday_hourly snapshots,
# which no diagnostic here needs.
ALL_LABELS: tuple[str, ...] = (
    "tue_open",
    "thu_pre_tnf",
    "sat_midday",
    "sun_early_close",
    "sun_late_close",
    "mon_pre_mnf",
)
SPREAD_ARM_LABELS: tuple[str, ...] = ("tue_open", "sun_late_close", "sun_early_close")
CALIBRATION_LABEL = "tue_open"


def _git_revision() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return "unknown"


def _bucket_ci_frame(ci: pd.DataFrame, prefix: str) -> pd.DataFrame:
    columns = ["bucket", f"{prefix}_lower", f"{prefix}_upper", f"{prefix}_probability_positive"]
    rows = ci.loc[ci["metric"].str.startswith("bucket_")].copy()
    if rows.empty:
        return pd.DataFrame(columns=columns)
    rows["bucket"] = rows["metric"].str.extract(r"bucket_(\d+)_gap")[0].astype(int)
    return rows.rename(
        columns={
            "lower": f"{prefix}_lower",
            "upper": f"{prefix}_upper",
            "probability_positive": f"{prefix}_probability_positive",
        }
    )[columns]


def _overall_gap_summary(ci: pd.DataFrame) -> dict[str, float]:
    row = ci.loc[ci["metric"].eq("mean_abs_calibration_gap")]
    if row.empty:
        return {}
    record = row.iloc[0]
    return {
        "estimate": float(record["estimate"]),
        "lower": float(record["lower"]),
        "upper": float(record["upper"]),
        "probability_positive": float(record["probability_positive"]),
    }


def build_calibration_arm(
    frame: pd.DataFrame,
    *,
    arm: str,
    probability_col: str,
    outcome_col: str,
    buckets: int,
    samples: int,
    seed: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """One arm's bucket table (with week/season CIs attached) plus its summary dict."""

    working = frame.loc[frame[probability_col].notna() & frame[outcome_col].notna()].copy()
    if working.empty:
        raise ValueError(f"No usable rows for calibration arm {arm!r}")
    edges = calibration_bucket_edges(working[probability_col], buckets=buckets)
    table = favourite_longshot_calibration(working, probability_col, outcome_col, edges=edges)
    week_ci = bootstrap_calibration_gaps(
        working, probability_col, outcome_col, edges, block="week", samples=samples, seed=seed
    )
    season_ci = bootstrap_calibration_gaps(
        working, probability_col, outcome_col, edges, block="season", samples=samples, seed=seed
    )
    table = table.merge(_bucket_ci_frame(week_ci, "week"), on="bucket", how="left")
    table = table.merge(_bucket_ci_frame(season_ci, "season"), on="bucket", how="left")
    table.insert(0, "arm", arm)
    summary = {
        "arm": arm,
        "n_games": len(working),
        "bucket_edges": [float(x) for x in edges],
        "mean_abs_calibration_gap_week_blocked": _overall_gap_summary(week_ci),
        "mean_abs_calibration_gap_season_blocked": _overall_gap_summary(season_ci),
    }
    return table, summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--features", type=Path, default=DEFAULT_FEATURES)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--bootstrap-samples", type=int, default=2_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260818)
    parser.add_argument("--calibration-buckets", type=int, default=5)
    args = parser.parse_args()

    print(f"Loading features: {args.features}")
    features = pd.read_parquet(args.features)
    regular = regular_season_rows(features)
    schedule = regular[["game_id", "season", "week"]].drop_duplicates("game_id")
    outcomes = regular[["game_id", "result"]].drop_duplicates("game_id")

    print(f"Loading raw quotes for labels {ALL_LABELS} from {args.root}")
    quotes = load_decision_quotes(
        args.root, capture_kind=HISTORICAL_CAPTURE_KIND, labels=ALL_LABELS
    )
    spread_quotes = quotes.loc[quotes["market"].eq("spreads") & quotes["price"].notna()]
    not_minus_110_share = (
        float((spread_quotes["price"] != -110).mean()) if len(spread_quotes) else float("nan")
    )
    print(
        f"Raw spread quote rows: {len(spread_quotes)}, share not exactly -110: "
        f"{not_minus_110_share:.4f}"
    )

    print("Building pairing table (line) and spread price consensus table (price)")
    pairing = build_pairing_table(
        args.root, capture_kind=HISTORICAL_CAPTURE_KIND, labels=ALL_LABELS, schedule=schedule
    )
    prices = spread_price_consensus_table(
        args.root, capture_kind=HISTORICAL_CAPTURE_KIND, labels=ALL_LABELS, schedule=schedule
    ).drop(columns=["snapshot_timestamp_utc"])

    join_keys = ["game_id", "season", "week", "decision_label", "capture_kind"]
    spread_source = pairing.merge(prices, on=join_keys, how="inner")
    spread_source = spread_source.loc[
        spread_source["decision_label"].isin(SPREAD_ARM_LABELS)
    ].copy()
    spread_novig = spread_novig_probabilities(spread_source)
    spread_novig = spread_novig.merge(outcomes, on="game_id", how="inner")
    ats_margin = pd.to_numeric(spread_novig["result"], errors="coerce") - pd.to_numeric(
        spread_novig["home_spread"], errors="coerce"
    )
    spread_novig["home_cover_at_label"] = np.select(
        [ats_margin.gt(0.0), ats_margin.lt(0.0)], [1.0, 0.0], default=np.nan
    )

    moneyline_source = pairing.loc[
        pairing["home_moneyline"].notna() & pairing["away_moneyline"].notna()
    ].copy()
    moneyline_novig = moneyline_novig_probabilities(moneyline_source)
    moneyline_novig = moneyline_novig.merge(outcomes, on="game_id", how="inner")
    result_numeric = pd.to_numeric(moneyline_novig["result"], errors="coerce")
    moneyline_novig["home_win"] = np.select(
        [result_numeric.gt(0.0), result_numeric.lt(0.0)], [1.0, 0.0], default=np.nan
    )

    print(f"spread_novig rows: {len(spread_novig)}; moneyline_novig rows: {len(moneyline_novig)}")

    ats_arm = spread_novig.loc[spread_novig["decision_label"].eq(CALIBRATION_LABEL)]
    su_arm = moneyline_novig.loc[moneyline_novig["decision_label"].eq(CALIBRATION_LABEL)]

    ats_table, ats_summary = build_calibration_arm(
        ats_arm,
        arm="ats_tue_open",
        probability_col="no_vig_home_cover_probability",
        outcome_col="home_cover_at_label",
        buckets=args.calibration_buckets,
        samples=args.bootstrap_samples,
        seed=args.bootstrap_seed,
    )
    su_table, su_summary = build_calibration_arm(
        su_arm,
        arm="su_tue_open",
        probability_col="no_vig_home_win_probability",
        outcome_col="home_win",
        buckets=args.calibration_buckets,
        samples=args.bootstrap_samples,
        seed=args.bootstrap_seed,
    )
    calibration_buckets = pd.concat([ats_table, su_table], ignore_index=True)

    print("\n=== ATS arm (no_vig_home_cover_probability @ tue_open) ===")
    print(ats_table.to_string(index=False))
    ats_week_gap = ats_summary["mean_abs_calibration_gap_week_blocked"]
    ats_season_gap = ats_summary["mean_abs_calibration_gap_season_blocked"]
    print(f"mean_abs_calibration_gap (week-blocked): {ats_week_gap}")
    print(f"mean_abs_calibration_gap (season-blocked): {ats_season_gap}")

    print("\n=== SU arm (no_vig_home_win_probability @ tue_open) ===")
    print(su_table.to_string(index=False))
    su_week_gap = su_summary["mean_abs_calibration_gap_week_blocked"]
    su_season_gap = su_summary["mean_abs_calibration_gap_season_blocked"]
    print(f"mean_abs_calibration_gap (week-blocked): {su_week_gap}")
    print(f"mean_abs_calibration_gap (season-blocked): {su_season_gap}")

    hold_by_label = (
        spread_novig.groupby("decision_label")["spread_hold"]
        .agg(mean_hold="mean", n="count")
        .reset_index()
    )
    print("\n=== Spread hold by decision label ===")
    print(hold_by_label.to_string(index=False))

    moneyline_hold_by_label = (
        moneyline_novig.groupby("decision_label")["moneyline_hold"]
        .agg(mean_hold="mean", n="count")
        .reset_index()
    )
    print("\n=== Moneyline hold by decision label ===")
    print(moneyline_hold_by_label.to_string(index=False))

    coverage = {
        "spread_rows_by_label": spread_novig.groupby("decision_label").size().to_dict(),
        "moneyline_rows_by_label": moneyline_novig.groupby("decision_label").size().to_dict(),
        "spread_seasons": sorted(int(s) for s in spread_novig["season"].unique()),
        "moneyline_seasons": sorted(int(s) for s in moneyline_novig["season"].unique()),
    }

    output_dir = args.output_root / run_id()
    atomic_parquet(spread_novig, output_dir / "spread_novig.parquet")
    atomic_parquet(moneyline_novig, output_dir / "moneyline_novig.parquet")
    atomic_parquet(calibration_buckets, output_dir / "calibration_buckets.parquet")

    metadata: dict[str, Any] = {
        "predeclaration": "docs/novig_diagnostics.md",
        "diagnostic_not_confirmation": True,
        "rotation_registry_touched": False,
        "git_revision": _git_revision(),
        "bootstrap_samples": args.bootstrap_samples,
        "bootstrap_seed": args.bootstrap_seed,
        "calibration_buckets_requested": args.calibration_buckets,
        "calibration_label": CALIBRATION_LABEL,
        "spread_arm_labels": list(SPREAD_ARM_LABELS),
        "moneyline_labels_loaded": list(ALL_LABELS),
        "raw_spread_quote_rows": len(spread_quotes),
        "raw_spread_quote_share_not_minus_110": not_minus_110_share,
        "coverage": coverage,
        "ats_arm_summary": ats_summary,
        "su_arm_summary": su_summary,
        "spread_hold_by_label": hold_by_label.to_dict(orient="records"),
        "moneyline_hold_by_label": moneyline_hold_by_label.to_dict(orient="records"),
    }
    atomic_json(metadata, output_dir / "metadata.json")
    print(f"\nartifacts: {output_dir}")


if __name__ == "__main__":
    main()
