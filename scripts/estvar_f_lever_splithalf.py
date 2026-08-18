"""Out-of-sample check for the f-lever tau sweep (estvar).

``scripts/estvar_real_cfb_audit.py``'s f-lever sweep on comparison B (thin vs
full) found tau=0.05 lifted ``probability_positive`` from 0.264 (ungated) to
0.9505 while cutting f from 19.8% to 2.06%. That tau was chosen by looking at
a 9-point sweep AFTER seeing the results -- exactly the kind of researcher
degree of freedom AGENTS.md's pooling discipline warns about ("the family
must be declared before the signs are seen"). This script checks whether the
same tau region replicates out of sample: split the clean-core seasons into
two halves by alternating season order (avoids one era owning one half),
fit thin/full POINT models only (no bootstrap refit -- this is a cheap
robustness check, not another interval), and sweep tau independently on each
half.

Not a rotation-registry look (CFB only, no picks moved). Writes JSON to the
scratchpad.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from nfl_ats.cfb_benchmark import (
    CFB_BENCHMARK_MIN_TRAIN_GAMES,
    CFB_BENCHMARK_RIDGE_ALPHA,
    fit_cfb_residual_model,
)
from nfl_ats.cfb_features import (
    CFB_CONTEXT_FEATURES,
    CFB_EXPERIENCE_FEATURES,
    CFB_MARKET_FEATURES,
    CFB_MODEL_FEATURE_COLUMNS,
)
from nfl_ats.estimation_variance import (
    gate_by_disagreement,
    mde80,
    naive_block_bootstrap_interval,
    picks_differ_fraction,
)

REPO = Path(__file__).resolve().parents[1]
DEFAULT_CFB_FEATURES = REPO / "data/processed/cfb_game_features.parquet"
OUT_DIR = Path(
    r"C:\Users\Ryan\AppData\Local\Temp\claude\F--Repos-nfl-py3"
    r"\56edf890-1650-456a-b560-8d8b00b374b6\scratchpad\estvar"
)

THIN_COLUMNS = (*CFB_MARKET_FEATURES, *CFB_CONTEXT_FEATURES, *CFB_EXPERIENCE_FEATURES)
FULL_COLUMNS = CFB_MODEL_FEATURE_COLUMNS
CLEAN_CORE_SEASONS = tuple(range(2012, 2020)) + tuple(range(2021, 2026))
TAU_GRID = (0.0, 0.01, 0.02, 0.03, 0.05, 0.08, 0.12, 0.18)


def _load_completed() -> pd.DataFrame:
    frame = pd.read_parquet(DEFAULT_CFB_FEATURES)
    frame["gameday"] = pd.to_datetime(frame["gameday"], errors="raise")
    completed = frame.loc[
        pd.to_numeric(frame["result"], errors="coerce").notna()
        & pd.to_numeric(frame["ats_margin"], errors="coerce").notna()
    ].copy()
    return completed.sort_values(["gameday", "game_id"]).reset_index(drop=True)


def pointwise_predictions(completed: pd.DataFrame, seasons: tuple[int, ...]) -> pd.DataFrame:
    rows = []
    for season in seasons:
        test = completed.loc[completed["season"].eq(season)]
        if test.empty:
            continue
        cutoff = test["gameday"].min()
        training = completed.loc[completed["gameday"].lt(cutoff)]
        if len(training) < CFB_BENCHMARK_MIN_TRAIN_GAMES:
            continue
        test = test.sort_values(["gameday", "game_id"]).reset_index(drop=True)
        keep = pd.to_numeric(test["home_cover"], errors="coerce").notna()
        test = test.loc[keep].reset_index(drop=True)

        thin_model = fit_cfb_residual_model(
            training, ridge_alpha=CFB_BENCHMARK_RIDGE_ALPHA, feature_columns=THIN_COLUMNS
        )
        full_model = fit_cfb_residual_model(
            training, ridge_alpha=CFB_BENCHMARK_RIDGE_ALPHA, feature_columns=FULL_COLUMNS
        )
        thin_prob = thin_model.predict(test)["home_cover_probability"].to_numpy(dtype=float)
        full_prob = full_model.predict(test)["home_cover_probability"].to_numpy(dtype=float)
        rows.append(
            pd.DataFrame(
                {
                    "season": season,
                    "week": test["week"].to_numpy(),
                    "actual": pd.to_numeric(test["home_cover"], errors="raise").to_numpy(
                        dtype=float
                    ),
                    "thin_prob": thin_prob,
                    "full_prob": full_prob,
                }
            )
        )
        print(f"  season {season}: train={thin_model.training_rows} test={len(test)}", flush=True)
    return pd.concat(rows, ignore_index=True)


def sweep(predictions: pd.DataFrame, *, label: str) -> list[dict[str, Any]]:
    actual = predictions["actual"].to_numpy()
    baseline = predictions["thin_prob"].to_numpy()
    candidate = predictions["full_prob"].to_numpy()
    block_ids = predictions["season"].to_numpy(dtype=np.int64) * 100 + predictions["week"].to_numpy(
        dtype=np.int64
    )
    n = len(actual)
    rows = []
    for tau in TAU_GRID:
        gated = gate_by_disagreement(baseline, candidate, threshold=tau)
        f = picks_differ_fraction(baseline, gated)
        interval = naive_block_bootstrap_interval(
            actual, baseline, gated, block_ids, samples=2_000, seed=20260818
        )
        rows.append(
            {
                "half": label,
                "tau": tau,
                "n_games": n,
                "f": f,
                "estimate": interval.estimate,
                "probability_positive": interval.probability_positive,
                "mde80": mde80(max(f, 1e-9), n),
            }
        )
    return rows


def main() -> None:
    completed = _load_completed()
    seasons = sorted(CLEAN_CORE_SEASONS)
    half_a = tuple(seasons[0::2])  # alternate seasons -> both halves span the full era
    half_b = tuple(seasons[1::2])
    print(f"Half A seasons: {half_a}", flush=True)
    print(f"Half B seasons: {half_b}", flush=True)

    print("=== Half A ===", flush=True)
    predictions_a = pointwise_predictions(completed, half_a)
    rows_a = sweep(predictions_a, label="A")

    print("=== Half B ===", flush=True)
    predictions_b = pointwise_predictions(completed, half_b)
    rows_b = sweep(predictions_b, label="B")

    all_rows = rows_a + rows_b
    for row in all_rows:
        print(json.dumps(row), flush=True)

    out_path = OUT_DIR / "f_lever_splithalf_results.json"
    out_path.write_text(json.dumps(all_rows, indent=2), encoding="utf-8")
    print(f"\nWrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
