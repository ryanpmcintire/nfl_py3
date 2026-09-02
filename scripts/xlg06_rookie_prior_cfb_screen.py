"""XLG-06 Stage 1 (only): pure-CFB screen -- no crosswalk, no NFL data, no rosters.

Executes the "Screen 1" step of the rookie/young-player-prior design at
``C:\\Users\\Ryan\\AppData\\Local\\Temp\\claude\\F--Repos-nfl-py3
\\c8c5fbdd-027f-438d-b992-979e83a91c2e\\scratchpad\\xlg06_scope\\design.md``,
as adjudicated by the reviewing orchestrator for this stage only: follow the
design's stated defaults, with four hard constraints -- (1) no external API
calls, locally-ingested CFB data only; (2) no portal/transfer data, no
name-based joining; (3) QB-first population phasing; (4) stages 2-3 (the
crosswalk, NFL rosters, any registry write) are explicitly out of scope for
this script. Stages 2-3 are NOT built here: no ``build_cfb_nfl_player_
crosswalk``, no NFL roster read, no ``registry/*.json`` write.

Question: does recruiting pedigree (known before a single college snap)
predict a true freshman's realized CFB usage in their first meaningful
season? Reliability of the outcome construct is measured before any
predictive claim, per AGENTS.md's binding "split-half reliability before
any predictive claim" rule.

**A restriction the design did not anticipate, measured this session**: the
design's Screen 1 (``design.md`` lines 192-206) calls for a cross-cohort
reliability read and cites the same odd/even-*week* split-half recipe
``docs/cfb_role_features.md`` Sec.6 and ``docs/injury_value_lost.md`` Sec.3.1
already used. Neither is executable as literally specified from local data
alone:

1. **Cohort depth.** ``recruiting_players`` is ingested locally for seasons
   2024-2026 only (``docs/cfb_data.md``: "recent classes; older classes await
   a future backfill"); ``usage`` is ingested locally for 2023-2025 only
   (``docs/cfb_data.md``: "2023+ ingested; 2013-2022 awaits a future
   backfill" -- note this directly contradicts this task's own framing of
   "usage 2013+" as locally present; the true local floor is 2023, measured
   this session via ``ls``/season-partition enumeration, not read from any
   doc). A "true freshman" (recruit class year == usage season) can only be
   matched where BOTH sources cover the same season, which leaves exactly
   **two** usable cohort-seasons: 2024 and 2025 (2026 recruits exist locally
   but 2026 usage does not). "Across cohorts" as the design phrases it
   plausibly intends many recruiting-class years; two is not a base a
   cohort-blocked bootstrap can stand on (this project's own measured floor,
   ``MIN_BLOCKS_FOR_INTERVAL = 10`` in ``estimation_variance.py``, is applied
   below and would reject a 2-block interval outright).
2. **Grain.** ``usage`` is a season-aggregate table -- one row per
   player-season, no week column, confirmed by reading its schema this
   session (14 columns, no ``week``). The temporal odd/even-week split-half
   recipe requires *repeated within-season measurement occasions*, which do
   not exist in this source. Reconstructing a per-week usage measure from
   ``participants`` (which does carry ``game_id``/``week`` and ESPN athlete
   ids in the same id space) was considered and rejected for Stage 1: it
   would be a *different, unvalidated construct* -- CFBD's own proprietary
   ``usage.*`` share formula is not disclosed, so a participants-rebuilt
   proxy would not be a second measurement of the same quantity, it would be
   a new pipeline entirely, which is out of Stage-1 scope (pure-CFB *screen*,
   cheapest possible test, not a new feature build).

**Largest honest subset actually run**, in place of the inexecutable pieces:

- The predictive question itself (recruiting rating/stars vs. realized
  freshman-season usage) *is* fully executable from local data with no
  crosswalk and no NFL rows, exactly as Screen 1 specifies, pooling the two
  available cohort-seasons (2024, 2025) and reporting each cohort
  separately as a coherence check (not a formal blocked interval -- two
  cohorts is below the block-count floor).
- In place of the (inexecutable) temporal split-half, this script reports a
  **construct-facet split-half**: CFBD's own ``usage`` table already
  partitions each player's plays into two disjoint down-type buckets,
  ``usage.standardDowns`` and ``usage.passingDowns``, which jointly cover
  essentially all of a player's snaps. Correlating these two already-computed
  facets across freshmen, Spearman-Brown corrected, is the nearest available
  analogue to a split-half reliability check that stays inside the same
  measurement pipeline (no new construct invented) -- explicitly NOT the
  canonical temporal recipe, and a result here does not by itself license
  invoking ``no_split_half_reliability`` as a registry closing ground
  (``weak_signals.validate_closure`` only requires *a* reliability number be
  cited, but AGENTS.md's intent is the genuine repeated-occasion reliability,
  which remains unmeasured here and is flagged as an open gap).

QB-first phasing (hard constraint 3): QB is the primary/headline population.
RB/WR/TE are reported as secondary, position-separated coherence checks --
never pooled with QB or with each other, because ``usage.overall`` is on a
structurally different scale by position (measured this session: QB mean
0.279, RB 0.113, WR 0.043, TE 0.029) and a naive pooled correlation would be
confounded by "is this a QB" rather than testing "does recruiting pedigree
predict usage within a position." ``usage``'s position vocabulary is itself
restricted to {QB, RB, WR, TE, FB} (measured, CFBD does not compute a usage
share for defense/OL/ST) -- so "all positions" was never on the table for
this specific outcome construct, independent of the QB-first phasing choice;
flagging this as a previously-undocumented restriction.

CFB-only, rotation-registry rule 8 (``docs/rotation_registry.md:58-60``):
CFB screens are free, no window assigned or needed. Nothing here writes to
``registry/*.json``; a ``weak-signals record`` command is proposed in the
results write-up, never executed by this script.

**2026-08-18 addendum (second run): the cohort-depth restriction above is
lifted, the grain restriction is not.** A backfill agent ingested
``usage`` 2013-2022 and ``recruiting_players`` 2013-2023 via this repo's own
``cfb-ingest`` CLI (no ad hoc API calls from this script -- confirmed by
inspection: ``load_sources`` only calls ``latest_cfb_snapshot``/
``load_cfb_snapshot``, which read on-disk parquet, and automatically picked
up the new consolidated snapshots since both are the newest manifest per
source, verified this session) and consolidated each source into one
full-range snapshot (``usage``: 13 seasons, 46,854 rows; ``recruiting_
players``: 14 seasons, 56,924 rows). The recruiting-years/usage-seasons
intersection is now **2013-2025, 13 usable cohorts** (measured; 2026
recruits still have no matching 2026 usage row, so 2026 stays excluded for
the same reason as before) -- above ``MIN_BLOCKS_FOR_INTERVAL = 10``, so a
genuine cohort-blocked bootstrap (point 1 above) is now computable and is
reported as **primary**; the original player-row bootstrap is kept as a
secondary coherence check. Point 2 (no week grain in ``usage``) is
unchanged and still blocks the canonical temporal split-half recipe --
the construct-facet substitute is re-run on the full history instead, same
caveats as before.

    ./.tools/uv.exe run --no-sync python scripts/xlg06_rookie_prior_cfb_screen.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import rankdata

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from nfl_ats.cfb import latest_cfb_snapshot, load_cfb_snapshot  # noqa: E402
from nfl_ats.data import DataContractError  # noqa: E402
from nfl_ats.estimation_variance import (  # noqa: E402
    MIN_BLOCKS_FOR_INTERVAL,
    BootstrapDegeneracyError,
    guard_block_count,
)
from nfl_ats.io import atomic_json, run_id  # noqa: E402

OUT_ROOT = REPO / "artifacts" / "xlg06_rookie_prior_cfb"
CFB_ROOT = REPO / "data" / "cfb"

# 20,000 per this task's binding instruction (mirrors experiments.py's own D3
# rationale: 2,000 leaves ~6-7% Monte-Carlo jitter near a gate). Seed frozen
# before any correlation or reliability number in this script was computed --
# only join/coverage diagnostics (counts, null rates) had been measured first.
BOOTSTRAP_SAMPLES = 20_000
BOOTSTRAP_SEED = 20260818

# QB-first (hard constraint 3): primary population. RB/WR/TE reported
# separately as secondary coherence checks, never pooled (see module
# docstring -- usage.overall's scale differs structurally by position).
PRIMARY_POSITION = "QB"
SECONDARY_POSITIONS = ("RB", "WR", "TE")

PREDICTOR_COLUMNS = ("rating", "stars")
OUTCOME_COLUMNS = ("usage.overall", "usage.pass", "usage.rush")


def load_sources() -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Load the two locally-ingested CFB sources this screen needs. No crosswalk."""

    recruiting_snapshot = latest_cfb_snapshot(CFB_ROOT, "recruiting_players")
    usage_snapshot = latest_cfb_snapshot(CFB_ROOT, "usage")
    recruiting = load_cfb_snapshot(recruiting_snapshot)
    usage = load_cfb_snapshot(usage_snapshot)
    provenance = {
        "recruiting_players_snapshot": recruiting_snapshot.snapshot_id,
        "recruiting_players_seasons_local": list(recruiting_snapshot.seasons),
        "usage_snapshot": usage_snapshot.snapshot_id,
        "usage_seasons_local": list(usage_snapshot.seasons),
    }
    return recruiting, usage, provenance


def build_true_freshman_population(
    recruiting: pd.DataFrame, usage: pd.DataFrame
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Join recruiting pedigree to same-season usage, entirely inside CFBD's own id space.

    A "true freshman" is operationalized as recruiting class year == usage
    season year (the standard reading of the term); this repository has no
    separate enrollment-year source to confirm redshirt/grayshirt status, so
    this is a pregame-knowable approximation, not a verified enrollment date.
    """

    recruiting_years = sorted(int(year) for year in recruiting["year"].unique())
    usage_seasons = sorted(int(season) for season in usage["season"].unique())
    cohort_years = sorted(set(recruiting_years) & set(usage_seasons))

    diagnostics: dict[str, Any] = {
        "recruiting_players_years_local": recruiting_years,
        "usage_seasons_local": usage_seasons,
        "cohort_years_usable": cohort_years,
        "recruiting_years_with_no_usage_overlap": sorted(set(recruiting_years) - set(cohort_years)),
        "usage_position_vocabulary": sorted(usage["position"].dropna().unique().tolist()),
        "per_cohort": {},
    }

    frames: list[pd.DataFrame] = []
    for year in cohort_years:
        recruits = recruiting.loc[recruiting["year"].eq(year)]
        recruits_with_id = recruits.loc[recruits["athleteId"].notna()]
        season_usage = usage.loc[usage["season"].eq(year)]
        joined = recruits_with_id.merge(
            season_usage,
            left_on="athleteId",
            right_on="id",
            how="inner",
            suffixes=("_recruit", "_usage"),
        )
        if joined["athleteId"].duplicated().any():
            raise DataContractError(
                f"XLG-06 stage-1 join produced duplicate athleteId matches for cohort {year}"
            )
        joined = joined.copy()
        joined["cohort_year"] = year
        frames.append(joined)
        diagnostics["per_cohort"][str(year)] = {
            "recruits_total": len(recruits),
            "recruits_with_athlete_id": len(recruits_with_id),
            "recruits_athlete_id_null_rate": float(recruits["athleteId"].isna().mean()),
            "usage_rows_total": len(season_usage),
            "matched_true_freshmen": len(joined),
            "matched_by_usage_position": {
                str(key): int(value)
                for key, value in joined["position_usage"].value_counts().items()
            },
        }

    matched = (
        pd.concat(frames, ignore_index=True)
        if frames
        else pd.DataFrame(columns=[*recruiting.columns, *usage.columns, "cohort_year"])
    )
    diagnostics["matched_total"] = len(matched)
    diagnostics["matched_rating_null"] = int(matched["rating"].isna().sum()) if len(matched) else 0
    diagnostics["matched_stars_null"] = int(matched["stars"].isna().sum()) if len(matched) else 0
    return matched, diagnostics


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    """Pearson correlation of ranks -- identical to scipy's Spearman rho, faster in a loop."""

    return float(np.corrcoef(rankdata(x), rankdata(y))[0, 1])


def _spearman_brown(r: float) -> float:
    return (2.0 * r) / (1.0 + r) if r > -1.0 else float("nan")


def _rowwise_pearson(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Vectorized equivalent of one ``np.corrcoef`` call per row pair."""
    x_centered = x - x.mean(axis=1, keepdims=True)
    y_centered = y - y.mean(axis=1, keepdims=True)
    numerator = np.sum(x_centered * y_centered, axis=1)
    denominator = np.sqrt(np.sum(x_centered**2, axis=1) * np.sum(y_centered**2, axis=1))
    with np.errstate(divide="ignore", invalid="ignore"):
        return numerator / denominator


def _rowwise_weighted_pearson(x: np.ndarray, y: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Correlation equivalent to expanding columns by integer ``weights``."""
    totals = weights.sum(axis=1)
    x_centered = x - (np.sum(weights * x, axis=1) / totals)[:, None]
    y_centered = y - (np.sum(weights * y, axis=1) / totals)[:, None]
    numerator = np.sum(weights * x_centered * y_centered, axis=1)
    sum_xx = np.sum(weights * x_centered**2, axis=1)
    sum_yy = np.sum(weights * y_centered**2, axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.asarray(numerator / np.sqrt(sum_xx * sum_yy))


def _batched_bootstrap_correlations(
    x: np.ndarray,
    y: np.ndarray,
    *,
    block_groups: list[np.ndarray],
    rng: np.random.Generator,
    samples: int,
    batch_size: int = 256,
) -> tuple[np.ndarray, np.ndarray]:
    """Bootstrap correlations without Python work per draw."""

    block_count = len(block_groups)
    if block_count == len(x):
        pearson = np.empty(samples)
        spearman = np.empty(samples)
        for start in range(0, samples, batch_size):
            stop = min(start + batch_size, samples)
            chosen = rng.integers(0, block_count, size=(stop - start, block_count))
            xs, ys = x[chosen], y[chosen]
            pearson[start:stop] = _rowwise_pearson(xs, ys)
            spearman[start:stop] = _rowwise_pearson(rankdata(xs, axis=1), rankdata(ys, axis=1))
        return pearson, spearman

    row_blocks = np.empty(len(x), dtype=np.intp)
    for block_index, group in enumerate(block_groups):
        row_blocks[group] = block_index
    rank_plans = []
    for values in (x, y):
        _, value_groups = np.unique(values, return_inverse=True)
        value_count = int(value_groups.max()) + 1
        counts = np.vstack(
            [np.bincount(value_groups[group], minlength=value_count) for group in block_groups]
        )
        rank_plans.append((value_groups, counts))

    pearson = np.empty(samples)
    spearman = np.empty(samples)
    for start in range(0, samples, batch_size):
        stop = min(start + batch_size, samples)
        batch = stop - start
        chosen = rng.integers(0, block_count, size=(batch, block_count))
        block_counts = np.sum(chosen[:, :, None] == np.arange(block_count), axis=1)
        weights = block_counts[:, row_blocks]
        ranks = []
        for value_groups, counts in rank_plans:
            group_weights = block_counts @ counts
            group_ranks = np.cumsum(group_weights, axis=1) - 0.5 * group_weights + 0.5
            ranks.append(group_ranks[:, value_groups])
        pearson[start:stop] = _rowwise_weighted_pearson(x, y, weights)
        spearman[start:stop] = _rowwise_weighted_pearson(ranks[0], ranks[1], weights)
    return pearson, spearman


def bootstrap_correlation(
    x: np.ndarray,
    y: np.ndarray,
    *,
    seed: int,
    samples: int,
    context: str,
    block: str = "player",
    block_ids: np.ndarray | None = None,
    min_blocks: int = MIN_BLOCKS_FOR_INTERVAL,
) -> dict[str, Any]:
    """Percentile bootstrap of Pearson r and Spearman rho.

    ``block="player"``: each matched player-row is its own block (no "week"
    dimension exists at this grain -- see module docstring point 2 -- so this
    is the degenerate, block-size-1 case of the project's block bootstrap,
    the same recipe ``split_half_reliability`` in ``scripts/cfb_value_
    weighted_continuity_screen.py`` and ``scripts/cfb_role_continuity_
    remeasurement.py`` already use to resample team-season rows directly).

    ``block="cohort"``: resamples whole recruiting-class cohorts (2026-08-18
    re-run, once the backfill made >= 10 cohorts available) -- the direct
    analogue of this project's standard "week" block for a source whose
    finest available temporal grain is a season, not a week. Requires
    ``block_ids`` (one cohort-year label per row of ``x``/``y``). This is now
    reported as PRIMARY (see module docstring's 2026-08-18 addendum): it
    resamples whole cohorts rather than treating within-cohort rows as fully
    independent, so it is the more conservative, more defensible interval
    once the block count supports one at all.

    ``on_degenerate="raise"`` in both cases, per this task's binding
    instruction: a caller about to report a verdict-grade number must not
    silently hand back an interval too coarse to be one.
    """

    if len(x) != len(y):
        raise ValueError("x and y must be the same length")
    n = len(x)
    if block == "player":
        block_groups = [np.array([index]) for index in range(n)]
    elif block == "cohort":
        if block_ids is None or len(block_ids) != n:
            raise ValueError("block_ids (one per row) is required for block='cohort'")
        unique_blocks = np.unique(block_ids)
        block_groups = [np.where(block_ids == label)[0] for label in unique_blocks]
    else:
        raise ValueError(f"Unknown block type: {block!r}")
    block_count = len(block_groups)
    verdict = guard_block_count(
        block_count, min_blocks=min_blocks, on_degenerate="raise", context=context
    )
    pearson_r = float(np.corrcoef(x, y)[0, 1])
    spearman_rho = _spearman(x, y)
    rng = np.random.default_rng(seed)
    pearson_draws, spearman_draws = _batched_bootstrap_correlations(
        x, y, block_groups=block_groups, rng=rng, samples=samples
    )
    return {
        "context": context,
        "block_type": block,
        "n": n,
        "blocks": verdict.block_count,
        "degenerate": verdict.degenerate,
        "marginal": verdict.marginal,
        "block_message": verdict.message,
        "pearson_r": pearson_r,
        "pearson_r_ci95": [
            float(np.nanquantile(pearson_draws, 0.025)),
            float(np.nanquantile(pearson_draws, 0.975)),
        ],
        "pearson_probability_positive": float(np.mean(pearson_draws > 0.0)),
        "spearman_rho": spearman_rho,
        "spearman_rho_ci95": [
            float(np.nanquantile(spearman_draws, 0.025)),
            float(np.nanquantile(spearman_draws, 0.975)),
        ],
        "spearman_probability_positive": float(np.mean(spearman_draws > 0.0)),
        "samples": samples,
        "seed": seed,
    }


def _safe_run(fn: Any, *args: Any, **kwargs: Any) -> dict[str, Any]:
    """Let the degeneracy guard raise, but do not let one degenerate subgroup kill the run."""

    try:
        result: dict[str, Any] = fn(*args, **kwargs)
        return result
    except BootstrapDegeneracyError as error:
        return {"degenerate_error": str(error)}


def construct_facet_reliability(
    matched: pd.DataFrame, *, position: str, seed: int, samples: int, block: str = "cohort"
) -> dict[str, Any]:
    """Substitute for the (inexecutable) temporal split-half -- see module docstring.

    ``block="cohort"`` (default since the 2026-08-18 backfill made >= 10
    cohorts available) resamples whole recruiting-class cohorts; pass
    ``block="player"`` for the original (2-cohort-era) per-row bootstrap,
    reported as a secondary coherence check in the second run.
    """

    subset = matched.loc[matched["position_usage"].eq(position)].dropna(
        subset=["usage.standardDowns", "usage.passingDowns"]
    )
    x = subset["usage.standardDowns"].to_numpy(dtype=float)
    y = subset["usage.passingDowns"].to_numpy(dtype=float)
    block_ids = subset["cohort_year"].to_numpy() if block == "cohort" else None
    result = _safe_run(
        bootstrap_correlation,
        x,
        y,
        seed=seed,
        samples=samples,
        context=f"construct_facet_reliability[{position}:{block}]",
        block=block,
        block_ids=block_ids,
    )
    result["position"] = position
    result["recipe"] = "construct_facet_substitute_not_temporal"
    if "pearson_r" in result:
        result["spearman_brown_full_length_reliability_pearson"] = _spearman_brown(
            result["pearson_r"]
        )
        result["spearman_brown_full_length_reliability_spearman"] = _spearman_brown(
            result["spearman_rho"]
        )
    return result


def predictor_outcome_correlation(
    matched: pd.DataFrame,
    *,
    position: str,
    predictor: str,
    outcome: str,
    seed: int,
    samples: int,
    block: str = "cohort",
    cohort_year: int | None = None,
) -> dict[str, Any]:
    """``block="cohort"`` (default, primary since the full-history backfill) resamples whole
    recruiting-class cohorts; ``block="player"`` is the original per-row bootstrap, reported as
    a secondary coherence check. ``cohort_year`` restricts to a single cohort (descriptive only,
    not itself blocked -- used for the per-cohort trend diagnostic).
    """

    subset = matched.loc[matched["position_usage"].eq(position)]
    if cohort_year is not None:
        subset = subset.loc[subset["cohort_year"].eq(cohort_year)]
    subset = subset.dropna(subset=[predictor, outcome])
    x = subset[predictor].to_numpy(dtype=float)
    y = subset[outcome].to_numpy(dtype=float)
    effective_block = "player" if cohort_year is not None else block
    block_ids = subset["cohort_year"].to_numpy() if effective_block == "cohort" else None
    label = f"{position}:{predictor}->{outcome}:{effective_block}" + (
        f":cohort{cohort_year}" if cohort_year is not None else ":pooled"
    )
    result = _safe_run(
        bootstrap_correlation,
        x,
        y,
        seed=seed,
        samples=samples,
        context=label,
        block=effective_block,
        block_ids=block_ids,
    )
    result.update(
        {
            "position": position,
            "predictor": predictor,
            "outcome": outcome,
            "cohort_year": cohort_year,
        }
    )
    return result


def main() -> None:
    started = time.perf_counter()
    timings: dict[str, float] = {}
    output = OUT_ROOT / run_id()
    print(f"Artifact directory: {output}", flush=True)

    print("=== Step 1: load recruiting_players + usage (no crosswalk, no NFL data) ===", flush=True)
    t0 = time.perf_counter()
    recruiting, usage, provenance = load_sources()
    timings["step1_load_sources_seconds"] = time.perf_counter() - t0
    print(json.dumps(provenance, indent=2), flush=True)

    print(
        "\n=== Step 2: build true-freshman population (recruit class year == usage season) ===",
        flush=True,
    )
    t0 = time.perf_counter()
    matched, join_diagnostics = build_true_freshman_population(recruiting, usage)
    timings["step2_build_population_seconds"] = time.perf_counter() - t0
    print(json.dumps(join_diagnostics, indent=2), flush=True)

    n_cohorts = len(join_diagnostics["cohort_years_usable"])
    print(f"\nUsable cohort count: {n_cohorts} (measured)", flush=True)

    print(
        "\n=== Step 3: construct-facet reliability of the freshman-usage construct "
        "(substitute for the inexecutable temporal split-half; BEFORE any predictive claim). "
        "cohort-blocked = primary, player-blocked = secondary coherence check ===",
        flush=True,
    )
    t0 = time.perf_counter()
    reliability_cohort = {
        position: construct_facet_reliability(
            matched, position=position, seed=1000 + index, samples=BOOTSTRAP_SAMPLES, block="cohort"
        )
        for index, position in enumerate((PRIMARY_POSITION, *SECONDARY_POSITIONS))
    }
    reliability_player = {
        position: construct_facet_reliability(
            matched, position=position, seed=1100 + index, samples=BOOTSTRAP_SAMPLES, block="player"
        )
        for index, position in enumerate((PRIMARY_POSITION, *SECONDARY_POSITIONS))
    }
    timings["step3_construct_reliability_seconds"] = time.perf_counter() - t0
    print(json.dumps({"cohort_blocked": reliability_cohort}, indent=2, default=str), flush=True)
    print(json.dumps({"player_blocked": reliability_player}, indent=2, default=str), flush=True)

    print(
        "\n=== Step 4: predictive correlation, QB-first (primary), pooled cohorts. "
        "cohort-blocked = primary, player-blocked = secondary coherence check ===",
        flush=True,
    )
    t0 = time.perf_counter()
    primary_cohort = {
        f"{predictor}->{outcome}": predictor_outcome_correlation(
            matched,
            position=PRIMARY_POSITION,
            predictor=predictor,
            outcome=outcome,
            seed=2000 + column_index,
            samples=BOOTSTRAP_SAMPLES,
            block="cohort",
        )
        for column_index, (predictor, outcome) in enumerate(
            (p, o) for p in PREDICTOR_COLUMNS for o in OUTCOME_COLUMNS
        )
    }
    primary_player = {
        f"{predictor}->{outcome}": predictor_outcome_correlation(
            matched,
            position=PRIMARY_POSITION,
            predictor=predictor,
            outcome=outcome,
            seed=2100 + column_index,
            samples=BOOTSTRAP_SAMPLES,
            block="player",
        )
        for column_index, (predictor, outcome) in enumerate(
            (p, o) for p in PREDICTOR_COLUMNS for o in OUTCOME_COLUMNS
        )
    }
    timings["step4_primary_correlation_seconds"] = time.perf_counter() - t0
    print(json.dumps({"cohort_blocked": primary_cohort}, indent=2, default=str), flush=True)
    print(json.dumps({"player_blocked": primary_player}, indent=2, default=str), flush=True)

    print(
        "\n=== Step 5: per-cohort trend diagnostic, QB, rating -> usage.overall "
        "(descriptive point estimates per cohort, not itself a blocked interval) ===",
        flush=True,
    )
    t0 = time.perf_counter()
    cohort_trend = {
        str(year): predictor_outcome_correlation(
            matched,
            position=PRIMARY_POSITION,
            predictor="rating",
            outcome="usage.overall",
            seed=3000 + year,
            samples=BOOTSTRAP_SAMPLES,
            cohort_year=year,
        )
        for year in join_diagnostics["cohort_years_usable"]
    }
    timings["step5_cohort_trend_seconds"] = time.perf_counter() - t0
    print(json.dumps(cohort_trend, indent=2, default=str), flush=True)

    print(
        "\n=== Step 6: secondary positions (RB/WR/TE), rating -> usage.overall, pooled cohorts. "
        "cohort-blocked = primary, player-blocked = secondary coherence check ===",
        flush=True,
    )
    t0 = time.perf_counter()
    secondary_cohort = {
        position: predictor_outcome_correlation(
            matched,
            position=position,
            predictor="rating",
            outcome="usage.overall",
            seed=4000 + index,
            samples=BOOTSTRAP_SAMPLES,
            block="cohort",
        )
        for index, position in enumerate(SECONDARY_POSITIONS)
    }
    secondary_player = {
        position: predictor_outcome_correlation(
            matched,
            position=position,
            predictor="rating",
            outcome="usage.overall",
            seed=4100 + index,
            samples=BOOTSTRAP_SAMPLES,
            block="player",
        )
        for index, position in enumerate(SECONDARY_POSITIONS)
    }
    timings["step6_secondary_positions_seconds"] = time.perf_counter() - t0
    print(json.dumps({"cohort_blocked": secondary_cohort}, indent=2, default=str), flush=True)
    print(json.dumps({"player_blocked": secondary_player}, indent=2, default=str), flush=True)

    timings["total_seconds"] = time.perf_counter() - started
    payload = {
        "design_path": (
            "C:/Users/Ryan/AppData/Local/Temp/claude/F--Repos-nfl-py3/"
            "c8c5fbdd-027f-438d-b992-979e83a91c2e/scratchpad/xlg06_scope/design.md"
        ),
        "stage": "1_pure_cfb_screen_only",
        "run": "2_full_history_backfill_2026-08-18",
        "n_cohorts": n_cohorts,
        "bootstrap_samples": BOOTSTRAP_SAMPLES,
        "bootstrap_seed_base": BOOTSTRAP_SEED,
        "min_blocks_for_interval": MIN_BLOCKS_FOR_INTERVAL,
        "provenance": provenance,
        "join_diagnostics": join_diagnostics,
        "construct_facet_reliability_cohort_blocked_primary": reliability_cohort,
        "construct_facet_reliability_player_blocked_secondary": reliability_player,
        "primary_qb_correlations_cohort_blocked_primary": primary_cohort,
        "primary_qb_correlations_player_blocked_secondary": primary_player,
        "primary_qb_per_cohort_trend": cohort_trend,
        "secondary_position_correlations_cohort_blocked_primary": secondary_cohort,
        "secondary_position_correlations_player_blocked_secondary": secondary_player,
        "timings": timings,
    }
    atomic_json(payload, output / "results.json")
    print(f"\nWrote {output / 'results.json'}", flush=True)
    print(json.dumps(timings, indent=2), flush=True)


if __name__ == "__main__":
    main()
