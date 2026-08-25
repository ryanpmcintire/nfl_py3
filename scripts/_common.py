"""Shared boilerplate for ``scripts/`` research screens and utilities.

Extracted by task ``ref-scripts-common`` (wave 1) from the verbatim-identical
function clusters quantified in ``reports/wave1/hyg-scripts-audit.md``
(§4a: 29 duplicate clusters / ~2k recoverable lines). Behavior-preserving
only: every helper here is byte-for-byte the algorithm it replaces (docstrings
and formatting aside), so each converted script's CLI interface, printed
output, and written artifacts are unchanged.

Importing this module makes the repo's ``src/`` tree importable (the same
two-line header every script previously repeated) and exposes:

- ``REPO`` — repository root (parents[1] of this file).
- ``latest_schedules()`` / ``default_schedules()`` — newest
  ``data/raw/*/schedules.parquet`` snapshot resolution (was per-file
  ``_latest_schedules`` in 28 files + ``default_schedules`` in 17).
- ``block_bootstrap_two_group(...)`` — vectorized joint week/season-blocked
  bootstrap of a subset-vs-complement gap (was copy-pasted in ~30 screens).
- ``summarize(...)`` — home-cover cell summary with slate-scaled effect,
  quantile CI, and ``probability_positive`` (verbatim cluster of 6 weather/
  travel-family screens).
- ``bootstrap_pearson_ci(...)`` — paired-resample Pearson CI (cluster of 4).

Measure-only posture is unchanged: nothing here writes to
``registry/weak_signals.json``, and no evaluation window is opened by this
refactor (AGENTS.md binding rule 3; static code movement only).
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]

_SRC = REPO / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))


def latest_schedules() -> Path:
    """Newest ``data/raw/*/schedules.parquet`` snapshot path."""
    candidates = sorted((REPO / "data/raw").glob("*/schedules.parquet"))
    if not candidates:
        raise FileNotFoundError("no data/raw/*/schedules.parquet snapshot found")
    return candidates[-1]


def default_schedules() -> Path:
    """Resolve lazily so importing a screen module never requires local data."""
    return latest_schedules()


def block_bootstrap_two_group(
    df: pd.DataFrame,
    *,
    flag_col: str,
    value_col: str,
    block_col: str,
    samples: int,
    seed: int,
) -> np.ndarray:
    """Vectorized joint block bootstrap of ``100*(subset_mean-complement_mean)``.

    One multinomial draw over block ids per replicate feeds both groups'
    means, so a resample never mixes blocks across groups. Invalid replicates
    (empty subset or complement) are dropped.
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


def summarize(
    df: pd.DataFrame,
    *,
    flag: pd.Series,
    block_col: str,
    samples: int,
    seed: int,
) -> dict[str, Any]:
    """Home-cover cell summary: raw gap, slate-scaled effect, CI95, P+."""
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


def bootstrap_pearson_ci(x: np.ndarray, y: np.ndarray, *, samples: int, seed: int) -> list[float]:
    """Paired-index bootstrap 95% CI for a Pearson correlation."""
    n = len(x)
    rng = np.random.default_rng(seed)
    draws = np.empty(samples, dtype=float)
    for i in range(samples):
        idx = rng.integers(0, n, size=n)
        xi, yi = x[idx], y[idx]
        if np.std(xi) == 0 or np.std(yi) == 0:
            draws[i] = np.nan
            continue
        draws[i] = float(np.corrcoef(xi, yi)[0, 1])
    valid = draws[~np.isnan(draws)]
    lower, upper = np.quantile(valid, [0.025, 0.975])
    return [float(lower), float(upper)]
