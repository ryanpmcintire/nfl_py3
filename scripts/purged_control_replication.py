"""Replicate ``scripts/purged_validate.py``'s positive control across many seeds.

``docs/calibration_distortion.md`` S8 ("One methodological change that should
stick") concludes the positive control in ``scripts/purged_validate.py``
should be replicated before it is cited again: ``positive_control()`` calls
``inject_synthetic_signal(df, target_accuracy=..., seed=42)`` exactly ONCE per
magnitude, so its readout (planted 51.3% came back 49.33%, wrong sign;
planted 53.0% came back 51.14%) has no measured dispersion attached to it.
Worse, because ``inject_synthetic_signal`` draws BOTH its ``{-1,+1}`` signal
and its noise from one ``np.random.default_rng(seed)``
(``src/nfl_ats/purged_cv.py:652-655``), the two magnitudes at seed 42 share
the identical signal and noise vectors -- only ``beta`` differs -- so the
recorded control is one unlucky draw reported as two independent readings.

This script reruns ``positive_control()``'s own construction verbatim -- same
``n_blocks=40``, same ``DEFAULT_PURGE_WEEKS``/``DEFAULT_EMBARGO_WEEKS``, same
``CFB_MODEL_FEATURE_COLUMNS + synthetic_signal`` feature set, same
``_accuracy_and_ci`` bootstrap (samples=500) -- at the same two magnitudes
``scripts/purged_validate.py::main`` tests (0.513, 0.53), across 20
INDEPENDENT seeds per magnitude (a per-magnitude generator, not one shared
generator reused across magnitudes, which is what fixes the "one draw counted
twice" defect above). One run measured ~4.6s locally, so 20 seeds x 2
magnitudes is ~3 minutes -- comfortably inside the 10-20 seed band this
replication was scoped to. Seed 42 is included first in the seed list so the
recorded reading's position in the fresh distribution is directly readable,
not re-derived.

Reuses ``scripts/purged_validate.py``'s own ``_load``/``_accuracy_and_ci`` by
import rather than copying them -- this script must be invoked as
``python scripts/purged_control_replication.py`` so ``scripts/`` is
``sys.path[0]`` and ``import purged_validate`` resolves -- and
``nfl_ats.purged_cv``'s ``inject_synthetic_signal``/``purged_cv_backtest``,
the exact functions ``positive_control()`` calls. Does not modify
``scripts/purged_validate.py``.

CFB only, read-only against ``data/processed/cfb_game_features.parquet``
(rotation-registry rule 8: CFB costs no NFL window). No ``registry/*.json``
or tracked doc is written. Writes per-seed raw rows and a summary to the
scratchpad as JSON/CSV, not to ``artifacts/``.

    ./.venv/Scripts/python.exe scripts/purged_control_replication.py
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import purged_validate as pv  # scripts/purged_validate.py -- reused by import, not modified
from scipy import stats

from nfl_ats.cfb_benchmark import CFB_BENCHMARK_MIN_TRAIN_GAMES, CFB_BENCHMARK_RIDGE_ALPHA
from nfl_ats.cfb_features import CFB_MODEL_FEATURE_COLUMNS
from nfl_ats.provenance import stamp_sidecar, write_stamped_artifact
from nfl_ats.purged_cv import (
    DEFAULT_EMBARGO_WEEKS,
    DEFAULT_PURGE_WEEKS,
    inject_synthetic_signal,
    purged_cv_backtest,
)

SCRATCH = Path(
    r"C:\Users\Ryan\AppData\Local\Temp\claude\F--Repos-nfl-py3"
    r"\c8c5fbdd-027f-438d-b992-979e83a91c2e\scratchpad\purged_control"
)

#: The exact two magnitudes ``scripts/purged_validate.py::main`` tests.
MAGNITUDES: tuple[float, ...] = (0.513, 0.53)

#: n_blocks=40 is what the recorded ``positive_control()`` calls use.
N_BLOCKS = 40

#: Independent seeds per magnitude. 42 -- the recorded control's own seed --
#: is listed first so its position in the fresh distribution is directly
#: readable rather than re-derived. Deliberately NOT shared between
#: magnitudes: each (magnitude, seed) pair below draws its own
#: ``inject_synthetic_signal`` generator, unlike the recorded script where
#: both magnitudes reused seed 42's single generator.
SEEDS: tuple[int, ...] = (42, *range(1, 20))


def run_one(df: pd.DataFrame, *, target_accuracy: float, seed: int) -> dict[str, Any]:
    """One independent draw of ``positive_control()``'s exact construction."""

    injected = inject_synthetic_signal(df, target_accuracy=target_accuracy, seed=seed)
    population_accuracy = float(
        (np.sign(injected["synthetic_signal"]) == np.sign(injected["ats_margin"])).mean()
    )
    feature_columns = (*CFB_MODEL_FEATURE_COLUMNS, "synthetic_signal")
    result = purged_cv_backtest(
        injected,
        n_blocks=N_BLOCKS,
        purge_weeks=DEFAULT_PURGE_WEEKS,
        embargo_weeks=DEFAULT_EMBARGO_WEEKS,
        min_train_games=CFB_BENCHMARK_MIN_TRAIN_GAMES,
        feature_columns=feature_columns,
        include_market_baseline=False,
    )
    stats_row = pv._accuracy_and_ci(result.predictions, samples=500, seed=9999)
    return {
        "target_accuracy": target_accuracy,
        "seed": seed,
        "population_accuracy_realized": population_accuracy,
        "truth_points": 100.0 * (population_accuracy - 0.5),
        "n_blocks": N_BLOCKS,
        "ridge_alpha": CFB_BENCHMARK_RIDGE_ALPHA,
        "n": stats_row["n"],
        "accuracy": stats_row["accuracy"],
        "ci_lower": stats_row["ci_lower"],
        "ci_upper": stats_row["ci_upper"],
        "raw_sign_accuracy": stats_row["raw_sign_accuracy"],
        "recovered_points": 100.0 * (stats_row["accuracy"] - 0.5),
        "sign_only_points": 100.0 * (stats_row["raw_sign_accuracy"] - 0.5),
    }


def _mean_ci(values: np.ndarray) -> dict[str, float]:
    """Mean, sd, 95% Student-t interval, min/max and ``probability_positive`` across seeds.

    Mirrors ``calibration_distortion_screen._mean_ci``: the spread ACROSS
    independent plant draws is the replication unit the doc's concern is
    about ("would a fresh plant of this size read the same way"), not a
    within-run bootstrap. ``probability_positive`` is ``P(mean > 0)`` under
    Student-t at ``n - 1`` df, per AGENTS.md's binding rule against bare
    pass/fail on an interval that contains zero.
    """

    values = np.asarray(values, dtype=float)
    n = int(values.size)
    mean = float(values.mean())
    if n < 2:
        return {
            "n": n,
            "mean": mean,
            "sd": float("nan"),
            "lower": mean,
            "upper": mean,
            "min": mean,
            "max": mean,
            "probability_positive": float("nan"),
        }
    sd = float(values.std(ddof=1))
    se = sd / np.sqrt(n)
    if se <= 0.0:
        probability = 1.0 if mean > 0 else (0.0 if mean < 0 else 0.5)
        return {
            "n": n,
            "mean": mean,
            "sd": sd,
            "lower": mean,
            "upper": mean,
            "min": float(values.min()),
            "max": float(values.max()),
            "probability_positive": probability,
        }
    half = float(stats.t.ppf(0.975, n - 1)) * se
    return {
        "n": n,
        "mean": mean,
        "sd": sd,
        "lower": mean - half,
        "upper": mean + half,
        "min": float(values.min()),
        "max": float(values.max()),
        "probability_positive": float(stats.t.cdf(mean / se, n - 1)),
    }


def _seed_42_summary(group: pd.DataFrame) -> dict[str, Any]:
    recorded = group.loc[group["seed"] == 42]
    others = group.loc[group["seed"] != 42, "recovered_points"]
    seed_42_recovered = (
        float(recorded["recovered_points"].iloc[0]) if len(recorded) else float("nan")
    )
    others_sd = float(others.std(ddof=1)) if len(others) > 1 else float("nan")
    z = (
        (seed_42_recovered - float(others.mean())) / others_sd
        if len(others) > 1 and others_sd > 0.0
        else float("nan")
    )
    return {
        "seed_42_recovered_points": seed_42_recovered,
        "other_seeds_mean_recovered_points": float(others.mean()) if len(others) else float("nan"),
        "other_seeds_sd_recovered_points": others_sd,
        "seed_42_z_vs_other_seeds": z,
    }


def main() -> None:
    SCRATCH.mkdir(parents=True, exist_ok=True)
    df = pv._load()
    rows: list[dict[str, Any]] = []
    t0 = time.time()
    for target_accuracy in MAGNITUDES:
        for seed in SEEDS:
            row = run_one(df, target_accuracy=target_accuracy, seed=seed)
            rows.append(row)
            print(
                f"target={target_accuracy:.3f} seed={seed:>3d} "
                f"truth={row['truth_points']:+.2f} "
                f"recovered={row['recovered_points']:+.2f} "
                f"sign_only={row['sign_only_points']:+.2f}",
                flush=True,
            )
    elapsed = time.time() - t0

    table = pd.DataFrame(rows)
    table.to_csv(SCRATCH / "per_seed_raw.csv", index=False)
    stamp_sidecar(SCRATCH / "per_seed_raw.csv")  # ENG-38
    with (SCRATCH / "per_seed_raw.json").open("w", encoding="utf-8") as handle:
        json.dump(rows, handle, indent=2, default=str)
    # ENG-38: rows is a list, not a dict -- write_stamped_artifact requires a
    # dict payload, so this list-shaped file is stamped via a sidecar instead
    # of changing its top-level JSON shape.
    stamp_sidecar(SCRATCH / "per_seed_raw.json")

    by_magnitude: dict[str, Any] = {}
    for target_accuracy, group in table.groupby("target_accuracy"):
        recovered = _mean_ci(group["recovered_points"].to_numpy())
        sign_only = _mean_ci(group["sign_only_points"].to_numpy())
        n_recover_sign = int((group["recovered_points"] > 0).sum())
        by_magnitude[f"{target_accuracy:.3f}"] = {
            "truth_points_mean": float(group["truth_points"].mean()),
            "recovered": recovered,
            "sign_only": sign_only,
            "seeds_recovering_planted_sign": n_recover_sign,
            "seeds_total": len(group),
            **_seed_42_summary(group),
        }

    summary: dict[str, Any] = {
        "elapsed_seconds": elapsed,
        "n_blocks": N_BLOCKS,
        "seeds": list(SEEDS),
        "magnitudes": list(MAGNITUDES),
        "by_magnitude": by_magnitude,
    }
    write_stamped_artifact(summary, SCRATCH / "summary.json")  # ENG-38
    print(json.dumps(summary, indent=2, default=str))
    print(f"\nwrote {SCRATCH / 'summary.json'}", flush=True)


if __name__ == "__main__":
    main()
