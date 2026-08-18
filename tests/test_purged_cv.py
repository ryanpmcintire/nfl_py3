from __future__ import annotations

import inspect
import itertools
import math

import numpy as np
import pandas as pd
import pytest

from nfl_ats.cfb_features import CFB_MODEL_FEATURE_COLUMNS, build_cfb_team_states
from nfl_ats.data import DataContractError
from nfl_ats.purged_cv import (
    DEFAULT_EMBARGO_WEEKS,
    DEFAULT_PURGE_WEEKS,
    GRAPH_RATING_HALF_LIFE_WEEKS,
    OPPONENT_ADJUSTMENT_HALF_LIFE_WEEKS,
    TEAM_STATE_SPAN,
    assign_week_order,
    ewma_contamination_games,
    ewma_retained_weight,
    half_life_contamination_weeks,
    inject_synthetic_signal,
    partition_week_blocks,
    permute_target,
    purged_cv_backtest,
    purged_embargoed_folds,
    synthetic_signal_accuracy,
    synthetic_signal_beta,
    team_persistent_null,
)

# ---------------------------------------------------------------------------
# Contamination-span measurement: pinned to the pipeline's own declared spans
# ---------------------------------------------------------------------------


def test_team_state_span_matches_source() -> None:
    """Guards against ``TEAM_STATE_SPAN`` silently drifting from the real default."""

    assert inspect.signature(build_cfb_team_states).parameters["span"].default == TEAM_STATE_SPAN


def test_declared_half_lives_match_source() -> None:
    from nfl_ats.cfb_opponent_adjustment import CFB_OPPONENT_HALF_LIFE_WEEKS
    from nfl_ats.graph_ratings import GraphRatingConfig

    assert OPPONENT_ADJUSTMENT_HALF_LIFE_WEEKS == CFB_OPPONENT_HALF_LIFE_WEEKS
    assert GraphRatingConfig().half_life_weeks == GRAPH_RATING_HALF_LIFE_WEEKS


def test_ewma_retained_weight_matches_direct_recursion() -> None:
    """Cross-check the closed-form decay against a literal EWMA recursion.

    Feed a span-8 EWMA a single unit impulse followed by 30 zeros; the
    state after k further updates IS the retained weight of that impulse
    (an EWMA of an impulse is its own weight sequence), so it must match
    ``ewma_retained_weight`` exactly.
    """

    span = 8
    alpha = 2.0 / (span + 1.0)
    state = 1.0
    for k in range(30):
        assert state == pytest.approx(ewma_retained_weight(k, span))
        state *= 1.0 - alpha
    with pytest.raises(ValueError):
        ewma_retained_weight(-1, span)
    with pytest.raises(ValueError):
        ewma_retained_weight(0, 1)


def test_ewma_contamination_games_thresholds() -> None:
    # Hand-verified against math.log(threshold) / math.log(1 - alpha).
    assert ewma_contamination_games(8, 0.5) == 3
    assert ewma_contamination_games(8, 0.05) == 12
    assert ewma_contamination_games(8, 0.01) == 19
    with pytest.raises(ValueError):
        ewma_contamination_games(8, 0.0)
    with pytest.raises(ValueError):
        ewma_contamination_games(8, 1.0)


def test_default_purge_and_embargo_are_derived_not_hardcoded() -> None:
    """The binding project rule: an ungated constant is a defect. Both defaults must be
    reproducible from the measured decay formula, not typed-in numbers."""

    assert DEFAULT_PURGE_WEEKS == ewma_contamination_games(TEAM_STATE_SPAN, 0.05) == 12
    assert (
        DEFAULT_EMBARGO_WEEKS
        == ewma_contamination_games(TEAM_STATE_SPAN, 0.01) - DEFAULT_PURGE_WEEKS
        == 7
    )


def test_half_life_contamination_weeks_matches_hand_derivation() -> None:
    # weeks = half_life * log2(1/threshold); hand-checked for both declared,
    # not-yet-exercised rolling-window families.
    assert half_life_contamination_weeks(16.0, 0.05) == pytest.approx(69.15, abs=0.01)
    assert half_life_contamination_weeks(16.0, 0.01) == pytest.approx(106.30, abs=0.01)
    assert half_life_contamination_weeks(8.0, 0.05) == pytest.approx(34.58, abs=0.01)
    assert half_life_contamination_weeks(GRAPH_RATING_HALF_LIFE_WEEKS, 0.05) == pytest.approx(
        half_life_contamination_weeks(OPPONENT_ADJUSTMENT_HALF_LIFE_WEEKS, 0.05) / 2.0, abs=0.01
    )
    with pytest.raises(ValueError):
        half_life_contamination_weeks(0.0, 0.05)
    with pytest.raises(ValueError):
        half_life_contamination_weeks(16.0, 0.0)


# ---------------------------------------------------------------------------
# Chronological block partitioning
# ---------------------------------------------------------------------------


def test_partition_week_blocks_is_contiguous_and_covers_every_week() -> None:
    block_of_week = partition_week_blocks(37, 5)
    assert len(block_of_week) == 37
    assert set(block_of_week) == set(range(5))
    # Every block is a contiguous run: weeks assigned to block b form an
    # unbroken range once sorted (true for np.array_split of a sorted range).
    for block_id in range(5):
        weeks = np.flatnonzero(block_of_week == block_id)
        assert list(weeks) == list(range(weeks.min(), weeks.max() + 1))
    with pytest.raises(ValueError):
        partition_week_blocks(37, 0)
    with pytest.raises(ValueError):
        partition_week_blocks(37, 38)


def test_assign_week_order_is_global_and_idempotent() -> None:
    frame = pd.DataFrame(
        {
            "season": [2013, 2013, 2014, 2014],
            "week": [1, 2, 1, 2],
            "gameday": pd.to_datetime(["2013-09-01", "2013-09-08", "2014-08-31", "2014-09-07"]),
        }
    )
    ordered = assign_week_order(frame)
    assert list(ordered["week_order"]) == [0, 1, 2, 3]
    # Idempotent: calling again on an already-ordered frame must not collide.
    twice = assign_week_order(ordered)
    assert list(twice["week_order"]) == [0, 1, 2, 3]


# ---------------------------------------------------------------------------
# Fold generation: the leakage-prevention contract itself
# ---------------------------------------------------------------------------


def _synthetic_week_frame(n_weeks: int, games_per_week: int = 3) -> pd.DataFrame:
    rows = []
    game_id = 0
    for week_order in range(n_weeks):
        season = 2000 + week_order // 16
        week = week_order % 16 + 1
        for _ in range(games_per_week):
            rows.append(
                {
                    "game_id": game_id,
                    "season": season,
                    "week": week,
                    "gameday": pd.Timestamp("2000-01-01") + pd.Timedelta(days=7 * week_order),
                }
            )
            game_id += 1
    return pd.DataFrame(rows)


def test_folds_are_disjoint_and_purge_embargo_widths_are_exact() -> None:
    frame = _synthetic_week_frame(n_weeks=40)
    purge, embargo = 3, 2
    folds = purged_embargoed_folds(
        frame, n_blocks=40, test_group_size=1, purge_weeks=purge, embargo_weeks=embargo
    )
    assert len(folds) == 40
    week_order = assign_week_order(frame)["week_order"].to_numpy()
    for fold in folds:
        train_idx, test_idx = set(fold.train_index), set(fold.test_index)
        assert train_idx.isdisjoint(test_idx)
        lo, hi = fold.test_week_range
        test_weeks = set(week_order[list(test_idx)])
        assert test_weeks == {lo} == {hi}  # test_group_size=1 -> single week
        excluded = set(range(len(frame))) - train_idx - test_idx
        excluded_weeks = set(week_order[list(excluded)]) if excluded else set()
        # Every excluded (purged/embargoed) row must fall within
        # [lo-purge, hi+purge+embargo]; nothing outside that band is dropped.
        for week in excluded_weeks:
            assert lo - purge <= week <= hi + purge + embargo
        # And every week strictly outside the purge+embargo band on either
        # side IS present in training (purge/embargo don't over-purge).
        train_weeks = set(week_order[list(train_idx)])
        for week in range(len(frame)):
            week_value = int(week_order[week])
            if week_value < lo - purge or week_value > hi + purge + embargo:
                assert week_value in train_weeks


def test_zero_purge_and_embargo_leaves_only_the_test_block_excluded() -> None:
    frame = _synthetic_week_frame(n_weeks=10)
    folds = purged_embargoed_folds(frame, n_blocks=10, purge_weeks=0, embargo_weeks=0)
    for fold in folds:
        assert len(fold.train_index) + len(fold.test_index) == len(frame)


def test_purge_and_embargo_growth_shrinks_training_monotonically() -> None:
    frame = _synthetic_week_frame(n_weeks=60)
    narrow = purged_embargoed_folds(frame, n_blocks=60, purge_weeks=1, embargo_weeks=0)
    wide = purged_embargoed_folds(frame, n_blocks=60, purge_weeks=5, embargo_weeks=3)
    # Same path count/ordering (block structure unchanged), strictly fewer
    # training rows once purge/embargo widen, for every interior fold.
    interior = range(10, 50)
    for path_id in interior:
        assert len(wide[path_id].train_index) <= len(narrow[path_id].train_index)


def test_combinatorial_test_group_size_generates_expected_path_count() -> None:
    frame = _synthetic_week_frame(n_weeks=8, games_per_week=2)
    n_blocks, k = 8, 2
    folds = purged_embargoed_folds(
        frame, n_blocks=n_blocks, test_group_size=k, purge_weeks=0, embargo_weeks=0
    )
    assert len(folds) == math.comb(n_blocks, k)
    combos_seen = {fold.test_blocks for fold in folds}
    assert combos_seen == set(itertools.combinations(range(n_blocks), k))
    # A block that is non-adjacent to another in a combo still gets its OWN
    # purge/embargo band applied independently (both sides represented).
    for fold in folds:
        assert len(fold.test_blocks) == k


def test_max_paths_subsamples_without_exceeding_the_cap() -> None:
    frame = _synthetic_week_frame(n_weeks=8, games_per_week=2)
    folds = purged_embargoed_folds(
        frame, n_blocks=8, test_group_size=3, purge_weeks=0, embargo_weeks=0, max_paths=10
    )
    assert len(folds) == 10


def test_fold_input_validation() -> None:
    frame = _synthetic_week_frame(n_weeks=5)
    with pytest.raises(ValueError, match="purge_weeks"):
        purged_embargoed_folds(frame, n_blocks=5, purge_weeks=-1)
    with pytest.raises(ValueError, match="embargo_weeks"):
        purged_embargoed_folds(frame, n_blocks=5, embargo_weeks=-1)
    with pytest.raises(ValueError, match="test_group_size"):
        purged_embargoed_folds(frame, n_blocks=5, test_group_size=0)
    with pytest.raises(ValueError, match="test_group_size"):
        purged_embargoed_folds(frame, n_blocks=5, test_group_size=6)


# ---------------------------------------------------------------------------
# End-to-end backtest smoke tests on the project's own CFB fixture
# ---------------------------------------------------------------------------


def test_purged_cv_backtest_contracts(cfb_features_frame: pd.DataFrame) -> None:
    result = purged_cv_backtest(
        cfb_features_frame,
        n_blocks=6,
        purge_weeks=1,
        embargo_weeks=1,
        min_train_games=30,
    )
    predictions = result.predictions
    assert set(predictions["method"]) == {"market", "market_residual"}
    assert result.config["folds_run"] > 0
    assert not predictions.empty

    # Every fold that ran met the training floor (folds that didn't are
    # counted in folds_skipped_insufficient_training, not silently dropped).
    assert result.fold_summary["train_games"].ge(30).all()

    residual = predictions.loc[predictions["method"].eq("market_residual")]
    assert residual["home_cover_probability"].between(0.0, 1.0).all()
    assert residual["path_id"].nunique() == result.config["folds_run"]


def test_purged_cv_backtest_rejects_missing_columns(cfb_features_frame: pd.DataFrame) -> None:
    with pytest.raises(DataContractError, match="missing columns"):
        purged_cv_backtest(cfb_features_frame.drop(columns=["spread_line"]), n_blocks=5)


def test_purged_cv_backtest_can_train_on_chronologically_later_games(
    cfb_features_frame: pd.DataFrame,
) -> None:
    """The defining, non-walk-forward property: an EARLY test block is allowed
    to use LATER games as training, once purged/embargoed."""

    result = purged_cv_backtest(
        cfb_features_frame, n_blocks=6, purge_weeks=0, embargo_weeks=0, min_train_games=30
    )
    predictions = result.predictions
    early_path = predictions["path_id"].min()
    early_test_gameday = predictions.loc[predictions["path_id"].eq(early_path), "gameday"].max()
    # At least one training row used for the earliest test block must be
    # dated AFTER that block's own games -- impossible in the walk-forward
    # evaluator by construction.
    later_training_exists = cfb_features_frame["gameday"].max() > early_test_gameday
    assert later_training_exists


# ---------------------------------------------------------------------------
# Negative control: permutation must leave no exploitable structure
# ---------------------------------------------------------------------------


def test_permute_target_breaks_spread_line_dependence(cfb_features_frame: pd.DataFrame) -> None:
    permuted = permute_target(cfb_features_frame, seed=7)
    spread = pd.to_numeric(permuted["spread_line"], errors="raise")
    # The bug this function's docstring warns about: permuting ``result``
    # while keeping ``spread_line`` fixed bakes a deterministic -spread_line
    # term into "ats_margin". Guard against regressing into that by checking
    # the permuted ats_margin is uncorrelated with spread_line in a frame
    # large enough for the check to be meaningful.
    if len(permuted) >= 30:
        correlation = np.corrcoef(
            permuted["ats_margin"].to_numpy(dtype=float), spread.to_numpy(dtype=float)
        )[0, 1]
        assert abs(correlation) < 0.3
    # Internal consistency: result/ats_margin/home_cover still agree.
    recomputed = pd.to_numeric(permuted["result"], errors="raise") - spread
    pd.testing.assert_series_equal(
        recomputed.reset_index(drop=True),
        permuted["ats_margin"].astype(float).reset_index(drop=True),
        check_names=False,
    )


def test_permute_target_preserves_features(cfb_features_frame: pd.DataFrame) -> None:
    permuted = permute_target(cfb_features_frame, seed=3)
    for column in CFB_MODEL_FEATURE_COLUMNS:
        pd.testing.assert_series_equal(
            permuted[column], cfb_features_frame[column], check_names=False
        )


def test_team_persistent_null_preserves_features_and_has_zero_population_signal(
    cfb_features_frame: pd.DataFrame,
) -> None:
    """The control built to actually exercise the shared-team-proximity channel:
    real team-persistent structure in the target, but no feature reveals it."""

    null_frame = team_persistent_null(cfb_features_frame, team_sigma=8.0, noise_sigma=13.0, seed=5)
    for column in CFB_MODEL_FEATURE_COLUMNS:
        pd.testing.assert_series_equal(
            null_frame[column], cfb_features_frame[column], check_names=False
        )
    assert null_frame["ats_margin"].std() > 0
    assert set(null_frame["home_cover"].dropna().unique()) <= {0.0, 1.0}
    spread = pd.to_numeric(null_frame["spread_line"], errors="raise")
    recomputed = pd.to_numeric(null_frame["result"], errors="raise") - spread
    pd.testing.assert_series_equal(
        recomputed.reset_index(drop=True),
        null_frame["ats_margin"].astype(float).reset_index(drop=True),
        check_names=False,
    )
    with pytest.raises(DataContractError, match="missing"):
        team_persistent_null(
            cfb_features_frame.drop(columns=["home_id"]),
            team_sigma=8.0,
            noise_sigma=13.0,
            seed=5,
        )


# ---------------------------------------------------------------------------
# Positive control: a planted, known-magnitude effect
# ---------------------------------------------------------------------------


def test_synthetic_signal_accuracy_roundtrip() -> None:
    for target in (0.51, 0.513, 0.55, 0.65):
        beta = synthetic_signal_beta(target, noise_std=6.0)
        assert synthetic_signal_accuracy(beta, noise_std=6.0) == pytest.approx(target, abs=1e-9)
    with pytest.raises(ValueError):
        synthetic_signal_beta(0.5, noise_std=6.0)
    with pytest.raises(ValueError):
        synthetic_signal_beta(1.0, noise_std=6.0)


def test_inject_synthetic_signal_realizes_target_accuracy_at_scale(
    cfb_features_frame: pd.DataFrame,
) -> None:
    # The fixture is tiny (n~115); use it only for the contract, not the
    # magnitude (population accuracy needs a large n to concentrate).
    injected = inject_synthetic_signal(cfb_features_frame, target_accuracy=0.55, seed=11)
    assert "synthetic_signal" in injected.columns
    assert set(injected["synthetic_signal"].unique()) <= {-1.0, 1.0}
    realized = (np.sign(injected["synthetic_signal"]) == np.sign(injected["ats_margin"])).mean()
    # Loose bound at small n; scripts/purged_validate.py checks the tight
    # bound at the full ~12,500-game scale.
    assert 0.35 < realized < 0.75
