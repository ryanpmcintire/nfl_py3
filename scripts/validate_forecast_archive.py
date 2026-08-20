"""ENV-01 validation: forecast temp/wind vs game-time actuals, outdoor games.

Joins a ``data/raw/forecast_archive/<timestamp>/forecasts.parquet`` snapshot
(built by ``scripts/ingest_forecast_archive.py``) against its own carried
``actual_temp_f``/``actual_wind_mph`` columns (sourced from the same
schedules parquet the ingestion script read), restricted to outdoor games
(``roof in {outdoors, open}``) with a successful forecast fetch. Reports
Pearson correlation, mean absolute error, and coverage -- a genuine 5-day-out
forecast should correlate strongly with the actual on the seasonal
temperature swing even though day-to-day forecast error is real and
expected; a near-zero or negative correlation would mean something is
misjoined (wrong station, wrong valid-time, wrong day), not just "weather is
hard to forecast."

Usage:
    .\\.tools\\uv.exe run --no-sync python scripts/validate_forecast_archive.py \\
        --forecast-archive data/raw/forecast_archive/<timestamp>/forecasts.parquet
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
OUTDOOR_ROOFS = {"outdoors", "open"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--forecast-archive", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    df = pd.read_parquet(args.forecast_archive)
    n_total = len(df)
    n_unmappable = int((df["fetch_status"] == "unmappable_international_stadium").sum())
    n_domestic = n_total - n_unmappable
    n_ok = int((df["fetch_status"] == "ok").sum())
    coverage_of_domestic = n_ok / n_domestic if n_domestic else float("nan")
    coverage_of_all = n_ok / n_total if n_total else float("nan")

    outdoor = df.loc[(df["fetch_status"] == "ok") & (df["roof"].isin(OUTDOOR_ROOFS))].copy()
    n_outdoor_ok = len(outdoor)

    both = outdoor.dropna(subset=["forecast_temp_f", "actual_temp_f"])
    n_temp_pairs = len(both)
    if n_temp_pairs >= 2:
        temp_corr = float(np.corrcoef(both["forecast_temp_f"], both["actual_temp_f"])[0, 1])
        temp_mae = float((both["forecast_temp_f"] - both["actual_temp_f"]).abs().mean())
        temp_bias = float((both["forecast_temp_f"] - both["actual_temp_f"]).mean())
    else:
        temp_corr = temp_mae = temp_bias = float("nan")

    both_wind = outdoor.dropna(subset=["forecast_wind_mph", "actual_wind_mph"])
    n_wind_pairs = len(both_wind)
    if n_wind_pairs >= 2:
        wind_corr = float(
            np.corrcoef(both_wind["forecast_wind_mph"], both_wind["actual_wind_mph"])[0, 1]
        )
        wind_mae = float(
            (both_wind["forecast_wind_mph"] - both_wind["actual_wind_mph"]).abs().mean()
        )
        wind_bias = float((both_wind["forecast_wind_mph"] - both_wind["actual_wind_mph"]).mean())
    else:
        wind_corr = wind_mae = wind_bias = float("nan")

    lookback_dist = (
        outdoor["lookback_steps_used"].value_counts().sort_index().to_dict()
        if "lookback_steps_used" in outdoor.columns
        else {}
    )

    result = {
        "forecast_archive": str(args.forecast_archive),
        "n_total_games": n_total,
        "n_unmappable_international": n_unmappable,
        "n_domestic": n_domestic,
        "n_fetch_ok": n_ok,
        "coverage_of_domestic": coverage_of_domestic,
        "coverage_of_all_reg_games": coverage_of_all,
        "n_outdoor_fetch_ok": n_outdoor_ok,
        "temp": {
            "n_pairs": n_temp_pairs,
            "pearson_r": temp_corr,
            "mae_f": temp_mae,
            "mean_bias_forecast_minus_actual_f": temp_bias,
        },
        "wind": {
            "n_pairs": n_wind_pairs,
            "pearson_r": wind_corr,
            "mae_mph": wind_mae,
            "mean_bias_forecast_minus_actual_mph": wind_bias,
        },
        "lookback_steps_used_distribution": {str(k): int(v) for k, v in lookback_dist.items()},
        "fetch_status_counts": {k: int(v) for k, v in df["fetch_status"].value_counts().items()},
        "games_per_season": {str(k): int(v) for k, v in df.groupby("season").size().items()},
    }

    print(json.dumps(result, indent=2))

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()
