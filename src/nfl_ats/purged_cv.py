"""Purged, embargoed cross-validation for the leak-safe walk-forward pipeline.

``nfl_ats.backtest.walk_forward_backtest`` and
``nfl_ats.cfb_benchmark.cfb_walk_forward_benchmark`` each extract exactly ONE
test observation per game, in exactly ONE ordering of history (predict week
``t`` using only games before it, refit, advance). That is leak-safe and it
is the evaluator of record for every pick this project makes -- nothing here
replaces it. But it also means every early season is spent purely as
training with no test credit, and there is no way to ask "how would this
model have scored a 2008 week if it could also learn from 2015-2025" without
literally reusing a game as both a training input for other folds and a test
point in this one.

This module adds that second, supplementary lens: **purged, embargoed
cross-validation** (Lopez de Prado's combinatorial purged CV, standard in
quantitative finance for exactly this problem). Instead of one growing
training window, it partitions the timeline into many blocks, tests each
combination of blocks once, and -- the part that makes it leak-safe rather
than a random k-fold -- deletes ("purges") every training observation whose
feature window could overlap the test block, plus an additional forward-only
buffer ("embargo") past the test block's end.

Leakage channel this module exists to neutralise
--------------------------------------------------
The frozen CFB benchmark's only per-game feature family with a rolling
window is the span-8 EWMA team-state (``cfb_features.build_cfb_team_states``,
NFL parameters taken verbatim). A team's state after game ``k`` is a weighted
average of its own last several games, so a "training" row for team T that
sits a few games away from a test-block row for the same team T carries a
feature vector that already reflects T's real, realized performance in nearby
games -- including, if purge is too narrow, games inside the test block
itself. ``ewma_contamination_games`` below turns the EWMA's own decay formula
into a materiality-threshold games count, and that count is what the module
purges and embargoes by default. Two other rolling-window families exist in
the wider pipeline (weighted-ridge opponent adjustment, half-life 16 weeks;
schedule-strength graph ratings, half-life 8 weeks) but neither is part of
the frozen CFB base-benchmark feature contract this module validates
against, so they are declared, not defaulted against -- see
``docs/purged_cv.md``.

Scope discipline
-----------------
Every empirical validation in this module and its companion scripts
(``scripts/purged_*.py``) runs on the CFB benchmark only (free, ~12,500
games, no NFL rotation cost -- rotation registry rule 8). Nothing here reads
or writes ``registry/rotation_registry.json``, changes a feature profile, or
moves a pick. See ``docs/purged_cv.md`` for what a result from this module
does and does not license.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy import stats

from nfl_ats.cfb_benchmark import (
    CFB_BENCHMARK_MIN_TRAIN_GAMES,
    CFB_BENCHMARK_RIDGE_ALPHA,
    cfb_evaluation_window,
    fit_cfb_residual_model,
)
from nfl_ats.cfb_features import CFB_MODEL_FEATURE_COLUMNS
from nfl_ats.data import DataContractError, require_columns
from nfl_ats.margin import MarginModel, fit_market_baseline

# ---------------------------------------------------------------------------
# Contamination-span measurement (EWMA team-state decay)
# ---------------------------------------------------------------------------

#: The frozen CFB base benchmark's only rolling-window feature family. Pinned
#: here as a constant instead of re-deriving it, and guarded by
#: ``tests/test_purged_cv.py::test_team_state_span_matches_source`` against
#: ``cfb_features.build_cfb_team_states``'s own default so the two cannot
#: silently drift apart.
TEAM_STATE_SPAN = 8

#: Declared spans for the OTHER rolling-window families in the wider
#: pipeline. Neither is part of the frozen CFB base-benchmark feature
#: contract (``CFB_MODEL_FEATURE_COLUMNS``) this module validates against, so
#: they do not feed ``DEFAULT_PURGE_WEEKS``/``DEFAULT_EMBARGO_WEEKS`` below --
#: they are recorded so a future purged-CV run against the "graph" or
#: "player_*" margin feature profiles knows to widen the window, and guarded
#: against drift the same way as ``TEAM_STATE_SPAN``.
OPPONENT_ADJUSTMENT_HALF_LIFE_WEEKS = 16.0
GRAPH_RATING_HALF_LIFE_WEEKS = 8.0


def half_life_contamination_weeks(half_life_weeks: float, weight_threshold: float) -> float:
    """Weeks until a ``0.5 ** (age / half_life)`` decay drops below ``weight_threshold``.

    Same materiality-threshold idea as ``ewma_contamination_games``, for the
    OTHER declared spans above: opponent adjustment and graph ratings decay
    continuously in calendar weeks (``fit_opponent_effects``,
    ``GraphRatingConfig``) rather than in a team's own game count, so their
    purge width is not directly comparable to ``ewma_contamination_games``'s
    game-count answer without this conversion.
    """

    if half_life_weeks <= 0.0:
        raise ValueError("half_life_weeks must be positive")
    if not 0.0 < weight_threshold < 1.0:
        raise ValueError("weight_threshold must be strictly between 0 and 1")
    return half_life_weeks * math.log2(1.0 / weight_threshold)


#: Materiality thresholds for "how much of an EWMA's weight must remain
#: before we call a game contaminated". Purge removes training rows where a
#: shared game could still hold >=5% of an EWMA update's weight; embargo
#: extends that to the 1% tail as an extra margin of safety, matching this
#: project's convention of separating a primary control from a safety buffer
#: (compare ``DEFAULT_OFFSEASON_RETENTION``'s own worked derivation).
PURGE_WEIGHT_THRESHOLD = 0.05
EMBARGO_WEIGHT_THRESHOLD = 0.01


def ewma_retained_weight(games_elapsed: int, span: int) -> float:
    """Fraction of a span-``span`` EWMA weight still attributable to an observation this far back.

    An EWM with span ``s`` has smoothing factor ``alpha = 2 / (s + 1)``; the
    weight on the observation ``k`` games before the most recent one is
    ``alpha * (1 - alpha) ** k`` before renormalisation, so the *relative*
    weight retained after ``k`` further observations (what matters for "has
    this decayed away") is ``(1 - alpha) ** k``, independent of the
    renormalising constant.
    """

    if games_elapsed < 0:
        raise ValueError("games_elapsed cannot be negative")
    if span < 2:
        raise ValueError("span must be at least 2")
    alpha = 2.0 / (span + 1.0)
    return float((1.0 - alpha) ** games_elapsed)


def ewma_contamination_games(span: int, weight_threshold: float) -> int:
    """Smallest game count ``k`` such that ``ewma_retained_weight(k, span) <= weight_threshold``.

    This is the measured (not guessed) contamination span: the number of a
    team's own games that must separate two dates before an EWMA state built
    from one date can no longer carry a materially-weighted trace of the
    other.
    """

    if not 0.0 < weight_threshold < 1.0:
        raise ValueError("weight_threshold must be strictly between 0 and 1")
    alpha = 2.0 / (span + 1.0)
    games = math.log(weight_threshold) / math.log(1.0 - alpha)
    return math.ceil(games)


#: Measured, not guessed: 12 weeks is where a span-8 EWMA's per-game weight
#: falls below 5%; purge removes training rows within that many weeks of a
#: test block on EITHER side (a shared-team feature window can reach forward
#: or backward).
DEFAULT_PURGE_WEEKS = ewma_contamination_games(TEAM_STATE_SPAN, PURGE_WEIGHT_THRESHOLD)

#: The additional forward-only buffer needed to reach the stricter 1% tail,
#: applied only AFTER a test block (Lopez de Prado's embargo is one-sided by
#: convention -- purge already covers both sides at the primary threshold).
DEFAULT_EMBARGO_WEEKS = (
    ewma_contamination_games(TEAM_STATE_SPAN, EMBARGO_WEIGHT_THRESHOLD) - DEFAULT_PURGE_WEEKS
)


# ---------------------------------------------------------------------------
# Chronological block partitioning
# ---------------------------------------------------------------------------


def assign_week_order(frame: pd.DataFrame) -> pd.DataFrame:
    """Attach a 0-based ``week_order``: the global chronological rank of each row's (season, week).

    Distinct from ``season``/``week`` alone because CFB week numbers reset
    every season; ``week_order`` is monotonic across the whole 2006-2025
    span, which is what purge/embargo distances are measured in. Idempotent:
    a pre-existing ``week_order`` column is dropped and recomputed rather
    than colliding with the merge.
    """

    require_columns(frame, ("season", "week", "gameday"), "purged CV frame")
    working = frame.drop(columns=["week_order"], errors="ignore").copy()
    working["gameday"] = pd.to_datetime(working["gameday"], errors="raise")
    week_starts = (
        working.groupby(["season", "week"], sort=False)["gameday"]
        .min()
        .reset_index()
        .sort_values("gameday")
        .reset_index(drop=True)
    )
    week_starts["week_order"] = np.arange(len(week_starts), dtype="int64")
    working = working.merge(
        week_starts[["season", "week", "week_order"]],
        on=["season", "week"],
        how="left",
        validate="many_to_one",
    )
    return working


def partition_week_blocks(n_weeks: int, n_blocks: int) -> npt.NDArray[np.int64]:
    """Assign each week_order to one of ``n_blocks`` contiguous, near-equal chronological groups."""

    if n_weeks < 1:
        raise ValueError("n_weeks must be positive")
    if not 1 <= n_blocks <= n_weeks:
        raise ValueError("n_blocks must be between 1 and n_weeks")
    chunks = np.array_split(np.arange(n_weeks, dtype="int64"), n_blocks)
    block_of_week = np.empty(n_weeks, dtype="int64")
    for block_id, chunk in enumerate(chunks):
        block_of_week[chunk] = block_id
    return block_of_week


# ---------------------------------------------------------------------------
# Purged + embargoed fold generation
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PurgedFold:
    """One train/test partition of a combinatorial purged CV run."""

    path_id: int
    test_blocks: tuple[int, ...]
    test_week_range: tuple[int, int]
    train_index: npt.NDArray[np.int64]
    test_index: npt.NDArray[np.int64]


def purged_embargoed_folds(
    frame: pd.DataFrame,
    *,
    n_blocks: int,
    test_group_size: int = 1,
    purge_weeks: int = DEFAULT_PURGE_WEEKS,
    embargo_weeks: int = DEFAULT_EMBARGO_WEEKS,
    max_paths: int | None = None,
    path_seed: int = 20260818,
) -> list[PurgedFold]:
    """Combinatorial purged, embargoed train/test folds over ``frame``'s chronological weeks.

    ``frame`` is partitioned into ``n_blocks`` contiguous week-blocks. Every
    ``test_group_size``-combination of blocks (``itertools.combinations``) is
    one path's test set; a path's training set is every row NOT in the test
    blocks, NOT within ``purge_weeks`` of either edge of any selected block
    (the feature-window-overlap purge), and NOT within an additional
    ``embargo_weeks`` immediately after any selected block's end (the
    one-sided safety buffer). ``test_group_size=1`` (the default) yields one
    path per block -- a purged, embargoed leave-one-block-out scheme, which
    is what the module's headline validations use, since it partitions every
    row into a test set exactly once. ``test_group_size>1`` generates the
    full combinatorial family (multiple overlapping paths per game), capped
    at ``max_paths`` by uniform subsampling when the exact combination count
    is intractable.
    """

    if purge_weeks < 0:
        raise ValueError("purge_weeks cannot be negative")
    if embargo_weeks < 0:
        raise ValueError("embargo_weeks cannot be negative")
    if test_group_size < 1:
        raise ValueError("test_group_size must be positive")
    if test_group_size > n_blocks:
        raise ValueError("test_group_size cannot exceed n_blocks")

    working = assign_week_order(frame)
    n_weeks = int(working["week_order"].max()) + 1
    block_of_week = partition_week_blocks(n_weeks, n_blocks)
    row_week = working["week_order"].to_numpy(dtype="int64")
    row_block = block_of_week[row_week]

    combos = list(itertools.combinations(range(n_blocks), test_group_size))
    if max_paths is not None and len(combos) > max_paths:
        rng = np.random.default_rng(path_seed)
        keep = np.sort(rng.choice(len(combos), size=max_paths, replace=False))
        combos = [combos[i] for i in keep]

    folds: list[PurgedFold] = []
    for path_id, combo in enumerate(combos):
        test_mask = np.isin(row_block, combo)
        block_ranges = [
            (
                int(np.flatnonzero(block_of_week == block_id).min()),
                int(np.flatnonzero(block_of_week == block_id).max()),
            )
            for block_id in combo
        ]
        purge_mask = np.zeros(len(working), dtype=bool)
        embargo_mask = np.zeros(len(working), dtype=bool)
        for lo, hi in block_ranges:
            purge_mask |= (row_week >= lo - purge_weeks) & (row_week <= hi + purge_weeks)
            embargo_mask |= (row_week > hi + purge_weeks) & (
                row_week <= hi + purge_weeks + embargo_weeks
            )
        train_mask = (~test_mask) & (~purge_mask) & (~embargo_mask)
        folds.append(
            PurgedFold(
                path_id=path_id,
                test_blocks=combo,
                test_week_range=(
                    min(lo for lo, _ in block_ranges),
                    max(hi for _, hi in block_ranges),
                ),
                train_index=np.flatnonzero(train_mask),
                test_index=np.flatnonzero(test_mask),
            )
        )
    return folds


# ---------------------------------------------------------------------------
# Fit + score each fold, reusing the frozen CFB benchmark's own estimator
# ---------------------------------------------------------------------------

_PASSTHROUGH_COLUMNS = (
    "game_id",
    "season",
    "week",
    "gameday",
    "spread_line",
    "home_spread_odds",
    "away_spread_odds",
    "result",
    "ats_margin",
    "home_cover",
)


@dataclass(frozen=True)
class PurgedCVResult:
    predictions: pd.DataFrame
    fold_summary: pd.DataFrame
    config: dict[str, Any] = field(default_factory=dict)


def _score_fold(
    test: pd.DataFrame,
    models: dict[str, MarginModel],
    fold: PurgedFold,
) -> list[pd.DataFrame]:
    batches: list[pd.DataFrame] = []
    for method, model in models.items():
        batch = test.loc[:, list(_PASSTHROUGH_COLUMNS)].copy()
        forecasts = model.predict(test)
        for column in forecasts:
            batch[column] = forecasts[column].to_numpy()
        batch["method"] = method
        batch["model_name"] = model.model_name
        batch["train_rows"] = model.training_rows
        batch["distribution_rows"] = model.distribution_rows
        batch["train_max_gameday"] = model.training_max_gameday
        batch["path_id"] = fold.path_id
        batch["test_blocks"] = str(fold.test_blocks)
        # A forced-pick instrument, not a wagering policy -- mirrors
        # cfb_benchmark._score_week so downstream summaries are compatible.
        batch["bet_side"] = "PASS"
        batch["bet_odds"] = np.nan
        batches.append(batch)
    return batches


def purged_cv_backtest(
    features: pd.DataFrame,
    *,
    n_blocks: int | None = None,
    test_group_size: int = 1,
    purge_weeks: int = DEFAULT_PURGE_WEEKS,
    embargo_weeks: int = DEFAULT_EMBARGO_WEEKS,
    min_train_games: int = CFB_BENCHMARK_MIN_TRAIN_GAMES,
    ridge_alpha: float = CFB_BENCHMARK_RIDGE_ALPHA,
    feature_columns: tuple[str, ...] = CFB_MODEL_FEATURE_COLUMNS,
    include_market_baseline: bool = True,
    max_paths: int | None = None,
    path_seed: int = 20260818,
) -> PurgedCVResult:
    """Purged, embargoed cross-validation over the CFB benchmark's own frozen estimator.

    Refits ``fit_cfb_residual_model``/``fit_market_baseline`` (unmodified,
    imported from ``nfl_ats.cfb_benchmark``/``nfl_ats.margin``) on each
    fold's purged, embargoed training set and scores its held-out test
    block(s). Supplementary to, and never a replacement for,
    ``cfb_walk_forward_benchmark``: this function is willing to train on
    games chronologically AFTER a test block (purged/embargoed by distance,
    not by direction), which the primary walk-forward evaluator never does.
    """

    required = {*_PASSTHROUGH_COLUMNS, *feature_columns}
    missing = sorted(required.difference(features.columns))
    if missing:
        raise DataContractError(f"purged CV features are missing columns: {', '.join(missing)}")

    frame = features.copy()
    frame["gameday"] = pd.to_datetime(frame["gameday"], errors="raise")
    completed = frame.loc[
        pd.to_numeric(frame["result"], errors="coerce").notna()
        & pd.to_numeric(frame["ats_margin"], errors="coerce").notna()
    ].copy()
    completed = (
        assign_week_order(completed).sort_values(["week_order", "game_id"]).reset_index(drop=True)
    )
    if completed.empty:
        raise ValueError("No completed games available for purged CV")

    n_weeks = int(completed["week_order"].max()) + 1
    resolved_n_blocks = n_weeks if n_blocks is None else n_blocks
    folds = purged_embargoed_folds(
        completed,
        n_blocks=resolved_n_blocks,
        test_group_size=test_group_size,
        purge_weeks=purge_weeks,
        embargo_weeks=embargo_weeks,
        max_paths=max_paths,
        path_seed=path_seed,
    )

    batches: list[pd.DataFrame] = []
    fold_rows: list[dict[str, Any]] = []
    skipped = 0
    for fold in folds:
        train = completed.iloc[fold.train_index]
        test = completed.iloc[fold.test_index]
        if len(train) < min_train_games or test.empty:
            skipped += 1
            continue
        models: dict[str, MarginModel] = {}
        if include_market_baseline:
            models["market"] = fit_market_baseline(train)
        models["market_residual"] = fit_cfb_residual_model(
            train, ridge_alpha=ridge_alpha, feature_columns=feature_columns
        )
        fold_batches = _score_fold(test, models, fold)
        batches.extend(fold_batches)
        for batch in fold_batches:
            fold_rows.append(
                {
                    "path_id": fold.path_id,
                    "test_blocks": str(fold.test_blocks),
                    "method": str(batch["method"].iloc[0]),
                    "train_games": len(train),
                    "test_games": len(test),
                    "accuracy": float(
                        (
                            batch["home_cover_probability"].ge(0.5).astype(float)
                            == batch["home_cover"]
                        )
                        .loc[batch["home_cover"].notna()]
                        .mean()
                    ),
                }
            )

    if not batches:
        raise ValueError("No purged CV fold had enough purged/embargoed training games")

    predictions = pd.concat(batches, ignore_index=True)
    predictions["evaluation_window"] = predictions["season"].map(
        lambda season: cfb_evaluation_window(int(season))
    )
    predictions = predictions.sort_values(["path_id", "gameday", "game_id", "method"]).reset_index(
        drop=True
    )
    fold_summary = pd.DataFrame(fold_rows).sort_values(["method", "path_id"]).reset_index(drop=True)
    config = {
        "n_blocks": resolved_n_blocks,
        "test_group_size": test_group_size,
        "purge_weeks": purge_weeks,
        "embargo_weeks": embargo_weeks,
        "min_train_games": min_train_games,
        "ridge_alpha": ridge_alpha,
        "feature_columns": list(feature_columns),
        "folds_run": len(folds) - skipped,
        "folds_skipped_insufficient_training": skipped,
    }
    return PurgedCVResult(predictions=predictions, fold_summary=fold_summary, config=config)


# ---------------------------------------------------------------------------
# Negative and positive controls
# ---------------------------------------------------------------------------


def permute_target(frame: pd.DataFrame, *, seed: int) -> pd.DataFrame:
    """Negative control: globally permute ``ats_margin`` (and re-derive ``result``/``home_cover``).

    Every feature -- including each team's real, causally-computed EWMA
    state trajectory, AND the real quoted ``spread_line`` -- is left
    untouched; only the row-to-outcome pairing is randomised. This is a valid
    "is there ANY signal at all" null, but it is NOT, on its own, a test of
    purge WIDTH specifically: a global i.i.d. permutation destroys every
    team-persistent structure in the target, and the shared-team
    feature-proximity channel this module exists to purge can only inflate
    results when the target has real team-persistent structure to leak
    (see ``team_persistent_null`` for the control that does carry that
    structure). ``docs/purged_cv.md`` reports both and is explicit about
    which one actually moved under a purge-width change.

    Permutes ``ats_margin`` itself, not ``result``: subtracting the row's OWN
    (real, unpermuted) ``spread_line`` from a permuted ``result`` would leave
    a deterministic ``-spread_line`` term baked into the "permuted" target --
    and ``spread_line`` is itself a model feature, so that construction would
    hand the model a real, exploitable relationship instead of noise. An
    earlier draft of this function had exactly that bug; a smoke run showed
    ~68% forced-pick accuracy on "permuted" data, which is what caught it.
    Permuting the market-relative residual directly has no such leak: a
    permuted ``ats_margin`` drawn from another game carries no relationship
    to this row's own ``spread_line`` or team-state features.
    """

    require_columns(frame, ("ats_margin", "spread_line"), "purged CV frame")
    working = frame.copy()
    rng = np.random.default_rng(seed)
    permuted_ats_margin = rng.permutation(
        pd.to_numeric(working["ats_margin"], errors="raise").to_numpy(dtype=float)
    )
    working["ats_margin"] = permuted_ats_margin
    spread = pd.to_numeric(working["spread_line"], errors="raise").to_numpy(dtype=float)
    working["result"] = spread + permuted_ats_margin
    working["home_cover"] = np.select(
        [permuted_ats_margin > 0, permuted_ats_margin < 0], [1.0, 0.0], default=np.nan
    )
    return working


def team_persistent_null(
    frame: pd.DataFrame,
    *,
    team_sigma: float,
    noise_sigma: float,
    seed: int,
) -> pd.DataFrame:
    """Negative control WITH team-persistent structure, but none of it visible to the model.

    Draws one latent "quality" per (team, season) -- ``N(0, team_sigma)``,
    independent across teams and seasons and independent of every real
    feature -- and sets ``ats_margin = quality[home] - quality[away] +
    N(0, noise_sigma)``. This mirrors the one property real team strength has
    that ``permute_target`` destroys: the SAME team's games share a
    persistent, unobserved component across a season. No feature in
    ``CFB_MODEL_FEATURE_COLUMNS`` reveals which team is playing or its
    latent quality, so the true population accuracy is still exactly 50% --
    but the shared-team feature-proximity channel now has real persistent
    structure sitting next to it in time, which is what purge/embargo widths
    are supposed to keep from being exploitable. A purge-width effect that
    fails to show up under ``permute_target`` but does show up here would
    confirm the channel is specifically about persistence, not raw signal.
    """

    require_columns(frame, ("home_id", "away_id", "season", "spread_line"), "purged CV frame")
    working = frame.copy()
    rng = np.random.default_rng(seed)
    team_seasons = pd.concat(
        [
            working[["home_id", "season"]].rename(columns={"home_id": "team_id"}),
            working[["away_id", "season"]].rename(columns={"away_id": "team_id"}),
        ],
        ignore_index=True,
    ).drop_duplicates()
    quality = pd.Series(
        rng.normal(0.0, team_sigma, size=len(team_seasons)),
        index=pd.MultiIndex.from_frame(team_seasons),
    )
    home_quality = quality.reindex(
        pd.MultiIndex.from_frame(
            working[["home_id", "season"]].rename(columns={"home_id": "team_id"})
        )
    ).to_numpy()
    away_quality = quality.reindex(
        pd.MultiIndex.from_frame(
            working[["away_id", "season"]].rename(columns={"away_id": "team_id"})
        )
    ).to_numpy()
    noise = rng.normal(0.0, noise_sigma, size=len(working))
    fake_ats_margin = home_quality - away_quality + noise

    spread = pd.to_numeric(working["spread_line"], errors="raise").to_numpy(dtype=float)
    working["ats_margin"] = fake_ats_margin
    working["result"] = spread + fake_ats_margin
    working["home_cover"] = np.select(
        [fake_ats_margin > 0, fake_ats_margin < 0], [1.0, 0.0], default=np.nan
    )
    return working


def synthetic_signal_accuracy(beta: float, noise_std: float) -> float:
    """True population forced-pick accuracy of the ``inject_synthetic_signal`` generative model.

    ``synthetic_margin = beta * signal + Normal(0, noise_std)`` with
    ``signal in {-1, +1}``; by symmetry the sign matches ``signal`` with
    probability ``Phi(beta / noise_std)``.
    """

    if noise_std <= 0.0:
        raise ValueError("noise_std must be positive")
    return float(stats.norm.cdf(beta / noise_std))


def synthetic_signal_beta(target_accuracy: float, noise_std: float) -> float:
    """Inverse of ``synthetic_signal_accuracy``: the ``beta`` that produces ``target_accuracy``."""

    if not 0.5 < target_accuracy < 1.0:
        raise ValueError("target_accuracy must be strictly between 0.5 and 1.0")
    if noise_std <= 0.0:
        raise ValueError("noise_std must be positive")
    return float(noise_std * stats.norm.ppf(target_accuracy))


def inject_synthetic_signal(
    frame: pd.DataFrame,
    *,
    target_accuracy: float,
    noise_std: float | None = None,
    seed: int,
    column: str = "synthetic_signal",
) -> pd.DataFrame:
    """Positive control: plant a new feature with a KNOWN population forced-pick accuracy edge.

    Adds an iid ``{-1, +1}`` column and overwrites ``result``/``ats_margin``/
    ``home_cover`` with a generative process whose only informative
    ingredient is that column, calibrated so the TRUE population accuracy of
    "always pick the side ``column`` favours" is exactly ``target_accuracy``
    (see ``synthetic_signal_accuracy``). A CV scheme that recovers
    approximately ``target_accuracy`` -- not inflated, not collapsed to 50%
    -- has proven it can both detect and correctly SIZE a real effect this
    small.

    ``noise_std`` defaults to the frame's OWN empirical ``ats_margin``
    standard deviation (measured, not guessed) rather than an arbitrary
    constant. That match matters: ``fit_cfb_residual_model``'s ridge penalty
    (``ridge_alpha=10``) is tuned for the real residual scale (~15.4 points
    of ATS margin std). An earlier version of this function defaulted
    ``noise_std=6.0`` -- a made-up constant a third the real scale -- which
    under-regularizes relative to what the frozen ridge alpha expects: the
    33 real-but-irrelevant CFB features soak up enough of the artificially
    low residual variance that their spurious fit swamps the deliberately
    tiny planted signal, and the recovered accuracy came out BELOW 50% (the
    wrong sign) instead of near the target. Matching the real noise scale
    removes that self-inflicted confound so this control tests the CV
    scheme, not an arbitrary hyperparameter mismatch.
    """

    require_columns(frame, ("spread_line", "ats_margin"), "purged CV frame")
    if noise_std is None:
        noise_std = float(pd.to_numeric(frame["ats_margin"], errors="coerce").std())
    beta = synthetic_signal_beta(target_accuracy, noise_std)
    rng = np.random.default_rng(seed)
    n = len(frame)
    signal = rng.choice(np.array([-1.0, 1.0]), size=n)
    noise = rng.normal(0.0, noise_std, size=n)
    synthetic_margin = beta * signal + noise

    working = frame.copy()
    working[column] = signal
    spread = pd.to_numeric(working["spread_line"], errors="raise").to_numpy(dtype=float)
    working["result"] = spread + synthetic_margin
    working["ats_margin"] = synthetic_margin
    working["home_cover"] = np.select(
        [synthetic_margin > 0, synthetic_margin < 0], [1.0, 0.0], default=np.nan
    )
    return working
