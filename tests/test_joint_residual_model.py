"""Tests for the MOD-17 joint residual model (``docs/mod17_joint_residual_model.md``).

Four things are pinned here, matching the task's own required coverage:
(1) the two-target ridge fit shape, and the predeclared mathematical fact
that it is column-independent (identical to two single-target fits);
(2) a walk-forward cutoff leakage test built so that VIOLATING the guard
changes the answer, mirroring ``tests/test_totals.py``'s own flip-week
pattern; (3) the correlation math on a small hand-computable frame; and
(4) enough of the realised-residual-frame / opener-evaluation plumbing to
catch a wiring regression without re-deriving the real odds archive.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nfl_ats.data import DataContractError
from nfl_ats.joint_residual_model import (
    MARGIN_BASELINE_FEATURES,
    UNION_FEATURES,
    blocked_correlation,
    joint_opener_pick_evaluation,
    leak_target_into_feature,
    make_joint_estimator,
    opener_accuracy_bootstrap,
    out_of_sample_r2,
    paired_opener_accuracy,
    pearson_correlation,
    per_season_correlation,
    realised_residual_frame,
    second_stage_predictions,
    totals_shaped_predictions,
    walk_forward_joint_predictions,
)
from nfl_ats.totals_wave2 import WAVE2_DRIVE_FEATURES

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_FEATURES = ("wind", "temp")


def _synthetic_features(
    *,
    weeks: int = 8,
    games_per_week: int = 40,
    flip_week: int = 5,
    season: int = 2000,
) -> pd.DataFrame:
    """A game table whose margin/total residuals REVERSE at ``flip_week``.

    ``wind`` drives ``margin_residual`` with slope +6 before the flip week
    and -6 from it onward (mirrors ``tests/test_totals.py``'s
    ``_synthetic_population`` flip-week trick); ``temp`` drives
    ``total_residual`` the same way. A walk-forward fit that honours the
    guard when predicting the flip week has seen only the pre-flip regime;
    one that leaked even a single later row has seen both, and the two give
    visibly different predictions.
    """

    generator = np.random.default_rng(20260905)
    rows = []
    gameday = pd.Timestamp("2000-09-01")
    for week in range(1, weeks + 1):
        margin_slope = 6.0 if week < flip_week else -6.0
        total_slope = 6.0 if week < flip_week else -6.0
        for game in range(games_per_week):
            wind = float(generator.uniform(-1.0, 1.0))
            temp = float(generator.uniform(-1.0, 1.0))
            spread_line = 0.0
            total_line = 44.0
            margin_residual = margin_slope * wind
            total_residual = total_slope * temp
            result = spread_line + margin_residual
            actual_total = total_line + total_residual
            home_score = (actual_total + result) / 2.0
            away_score = (actual_total - result) / 2.0
            rows.append(
                {
                    "game_id": f"{season}_{week:02d}_{game:02d}",
                    "season": season,
                    "week": week,
                    "game_type": "REG",
                    "gameday": gameday + pd.Timedelta(days=7 * (week - 1)),
                    "result": result,
                    "ats_margin": margin_residual,
                    "spread_line": spread_line,
                    "total_line": total_line,
                    "home_score": home_score,
                    "away_score": away_score,
                    "wind": wind,
                    "temp": temp,
                }
            )
    return pd.DataFrame(rows)


def _postseason_row(features: pd.DataFrame) -> pd.DataFrame:
    extra = features.iloc[[0]].copy()
    extra["game_id"] = "playoff_row"
    extra["game_type"] = "WC"
    return pd.concat([features, extra], ignore_index=True)


# ---------------------------------------------------------------------------
# 1. Union feature set
# ---------------------------------------------------------------------------


def test_union_features_are_the_margin_baseline_plus_wave2_drive_with_no_overlap() -> None:
    assert len(UNION_FEATURES) == len(MARGIN_BASELINE_FEATURES) + len(WAVE2_DRIVE_FEATURES)
    assert set(MARGIN_BASELINE_FEATURES).isdisjoint(WAVE2_DRIVE_FEATURES)
    assert tuple(MARGIN_BASELINE_FEATURES) + tuple(WAVE2_DRIVE_FEATURES) == UNION_FEATURES
    assert len(set(UNION_FEATURES)) == len(UNION_FEATURES)


# ---------------------------------------------------------------------------
# 2. realised_residual_frame: targets and the regular-season filter
# ---------------------------------------------------------------------------


def test_realised_residual_frame_computes_both_targets_and_drops_postseason() -> None:
    features = _synthetic_features(weeks=3, games_per_week=5)
    contaminated = _postseason_row(features)

    frame = realised_residual_frame(contaminated, feature_columns=_FEATURES)

    assert (frame["game_type"] == "REG").all()
    assert "playoff_row" not in set(frame["game_id"])
    assert frame["margin_residual"].to_numpy() == pytest.approx(frame["ats_margin"].to_numpy())
    expected_total_residual = (
        frame["home_score"].to_numpy()
        + frame["away_score"].to_numpy()
        - frame["total_line"].to_numpy()
    )
    assert frame["total_residual"].to_numpy() == pytest.approx(expected_total_residual)
    assert frame["market_total"].to_numpy() == pytest.approx(frame["total_line"].to_numpy())
    assert frame["actual_total"].to_numpy() == pytest.approx(
        frame["home_score"].to_numpy() + frame["away_score"].to_numpy()
    )
    assert frame["market_error"].to_numpy() == pytest.approx(
        frame["market_total"].to_numpy() - frame["actual_total"].to_numpy()
    )


def test_realised_residual_frame_requires_declared_columns() -> None:
    features = _synthetic_features(weeks=2, games_per_week=3).drop(columns=["ats_margin"])
    with pytest.raises(DataContractError, match="ats_margin"):
        realised_residual_frame(features, feature_columns=_FEATURES)


# ---------------------------------------------------------------------------
# 3. Multi-output ridge is column-independent (the predeclared fact this
#    module is built around).
# ---------------------------------------------------------------------------


def test_multi_output_ridge_matches_two_independent_single_target_fits() -> None:
    features = _synthetic_features(weeks=8, games_per_week=40)
    population = realised_residual_frame(features, feature_columns=_FEATURES)

    joint = walk_forward_joint_predictions(
        population,
        feature_columns=_FEATURES,
        target_columns=("margin_residual", "total_residual"),
        min_train_games=40,
    )
    margin_only = walk_forward_joint_predictions(
        population,
        feature_columns=_FEATURES,
        target_columns=("margin_residual",),
        min_train_games=40,
    )
    total_only = walk_forward_joint_predictions(
        population,
        feature_columns=_FEATURES,
        target_columns=("total_residual",),
        min_train_games=40,
    )

    joint_by_id = joint.set_index("game_id")
    margin_by_id = margin_only.set_index("game_id")
    total_by_id = total_only.set_index("game_id")

    np.testing.assert_allclose(
        joint_by_id.loc[margin_by_id.index, "predicted_margin_residual"].to_numpy(),
        margin_by_id["predicted_margin_residual"].to_numpy(),
        atol=1e-9,
    )
    np.testing.assert_allclose(
        joint_by_id.loc[total_by_id.index, "predicted_total_residual"].to_numpy(),
        total_by_id["predicted_total_residual"].to_numpy(),
        atol=1e-9,
    )


# ---------------------------------------------------------------------------
# 4. Walk-forward cutoff leakage guard
# ---------------------------------------------------------------------------


def test_walk_forward_joint_predictions_trains_only_on_strictly_earlier_weeks() -> None:
    features = _synthetic_features(weeks=8, games_per_week=40, flip_week=5)
    population = realised_residual_frame(features, feature_columns=_FEATURES)

    predictions = walk_forward_joint_predictions(
        population,
        feature_columns=_FEATURES,
        target_columns=("margin_residual", "total_residual"),
        min_train_games=40,
    )

    scored_weeks = sorted(predictions["week"].unique())
    assert scored_weeks == [2, 3, 4, 5, 6, 7, 8]
    for week in scored_weeks:
        block = predictions.loc[predictions["week"] == week]
        assert int(block["train_games"].iloc[0]) == 40 * (week - 1)

    target_week = 5  # the flip week: honest and leaky training disagree here.
    honest_train = population.loc[population["week"] < target_week]
    leaky_train = population.loc[population["week"] <= target_week]
    target_rows = population.loc[population["week"] == target_week]

    honest = make_joint_estimator()
    honest.fit(
        honest_train.loc[:, list(_FEATURES)],
        honest_train.loc[:, ["margin_residual", "total_residual"]].to_numpy(),
    )
    leaky = make_joint_estimator()
    leaky.fit(
        leaky_train.loc[:, list(_FEATURES)],
        leaky_train.loc[:, ["margin_residual", "total_residual"]].to_numpy(),
    )
    honest_prediction = np.asarray(honest.predict(target_rows.loc[:, list(_FEATURES)]), dtype=float)
    leaky_prediction = np.asarray(leaky.predict(target_rows.loc[:, list(_FEATURES)]), dtype=float)
    walked = predictions.loc[
        predictions["week"] == target_week,
        ["predicted_margin_residual", "predicted_total_residual"],
    ].to_numpy()

    np.testing.assert_allclose(walked, honest_prediction, atol=1e-9)
    assert np.abs(leaky_prediction - honest_prediction).max() > 1.0
    assert not np.allclose(walked, leaky_prediction)


def test_walk_forward_joint_predictions_respects_the_warm_up_floor() -> None:
    features = _synthetic_features(weeks=6, games_per_week=40)
    population = realised_residual_frame(features, feature_columns=_FEATURES)

    predictions = walk_forward_joint_predictions(
        population,
        feature_columns=_FEATURES,
        target_columns=("margin_residual", "total_residual"),
        min_train_games=100,
    )
    assert sorted(predictions["week"].unique()) == [4, 5, 6]
    assert int(predictions["train_games"].min()) >= 100

    with pytest.raises(ValueError, match="min_train_games"):
        walk_forward_joint_predictions(
            population,
            feature_columns=_FEATURES,
            target_columns=("margin_residual", "total_residual"),
            min_train_games=10_000,
        )


def test_second_stage_predictions_uses_only_strictly_earlier_stage1_blocks() -> None:
    features = _synthetic_features(weeks=10, games_per_week=40, flip_week=6)
    population = realised_residual_frame(features, feature_columns=_FEATURES)
    stage1 = walk_forward_joint_predictions(
        population,
        feature_columns=_FEATURES,
        target_columns=("margin_residual", "total_residual"),
        min_train_games=40,
    )

    stage2 = second_stage_predictions(stage1, min_train_games=80)
    scored_weeks = sorted(stage2["week"].unique())
    assert scored_weeks, "stage 2 produced no predictions"
    for week in scored_weeks:
        block = stage2.loc[stage2["week"] == week]
        prior_rows = int((stage1["week"] < week).sum())
        assert int(block["stage2_train_games"].iloc[0]) == prior_rows
        assert prior_rows >= 80


# ---------------------------------------------------------------------------
# 5. totals_shaped_predictions and out_of_sample_r2
# ---------------------------------------------------------------------------


def test_totals_shaped_predictions_aliases_the_requested_target() -> None:
    frame = pd.DataFrame({"predicted_total_residual": [1.0, 2.0], "other": [9.0, 9.0]})
    shaped = totals_shaped_predictions(frame, target_column="total_residual")
    assert shaped["predicted_residual"].tolist() == [1.0, 2.0]
    assert "predicted_total_residual" in shaped.columns  # original column preserved

    with pytest.raises(DataContractError):
        totals_shaped_predictions(frame, target_column="margin_residual")


def test_out_of_sample_r2_known_values() -> None:
    actual = pd.Series([2.0, -3.0, 4.0, -1.0])
    assert out_of_sample_r2(actual, actual) == pytest.approx(1.0)
    assert out_of_sample_r2(actual, pd.Series([0.0, 0.0, 0.0, 0.0])) == pytest.approx(0.0)
    # A prediction that overshoots to 3x the true magnitude is worse than the
    # zero baseline: the residual left behind (2x actual) has 4x actual's
    # own sum of squares, so SS_res > SS_tot and R2 < 0.
    assert out_of_sample_r2(actual, actual * 3.0) < 0.0

    with pytest.raises(ValueError):
        out_of_sample_r2(pd.Series([0.0, 0.0]), pd.Series([1.0, 1.0]))


# ---------------------------------------------------------------------------
# 6. Correlation math on a hand-computable frame
# ---------------------------------------------------------------------------


def test_pearson_correlation_known_values() -> None:
    a = pd.Series([1.0, 2.0, 3.0, 4.0])
    assert pearson_correlation(a, a) == pytest.approx(1.0)
    assert pearson_correlation(a, -a) == pytest.approx(-1.0)
    # Zero variance in one column is a documented guard, not an exception.
    assert pearson_correlation(a, pd.Series([5.0, 5.0, 5.0, 5.0])) == 0.0

    with pytest.raises(ValueError):
        pearson_correlation(pd.Series([1.0]), pd.Series([1.0]))


def test_per_season_correlation_matches_manual_groupby() -> None:
    frame = pd.DataFrame(
        {
            "season": [2020, 2020, 2020, 2021, 2021, 2021],
            "a": [1.0, 2.0, 3.0, 1.0, 2.0, 3.0],
            "b": [1.0, 2.0, 3.0, 3.0, 2.0, 1.0],
        }
    )
    result = per_season_correlation(frame, "a", "b").set_index("season")
    assert result.loc[2020, "correlation"] == pytest.approx(1.0)
    assert result.loc[2021, "correlation"] == pytest.approx(-1.0)
    assert result.loc[2020, "games"] == 3


def test_blocked_correlation_reports_probability_positive() -> None:
    generator = np.random.default_rng(20260905)
    seasons = np.repeat(np.arange(2018, 2024), 40)
    a = generator.normal(size=len(seasons))
    b = a * 0.6 + generator.normal(scale=0.3, size=len(seasons))
    frame = pd.DataFrame({"season": seasons, "week": np.tile(np.arange(1, 41), 6), "a": a, "b": b})

    result = blocked_correlation(frame, "a", "b", block="season", samples=500, seed=1)
    assert result["estimate"] > 0.0
    assert result["lower"] < result["estimate"] < result["upper"]
    assert 0.0 <= result["probability_positive"] <= 1.0
    assert result["blocks"] == 6
    assert result["games"] == len(frame)


# ---------------------------------------------------------------------------
# 7. Positive-control contamination helper
# ---------------------------------------------------------------------------


def test_leak_target_into_feature_replaces_only_the_named_column() -> None:
    frame = pd.DataFrame({"margin_residual": [1.0, 2.0, 3.0], "home_point_diff": [9.0, 9.0, 9.0]})
    leaked = leak_target_into_feature(
        frame, feature_column="home_point_diff", target_column="margin_residual"
    )
    assert leaked["home_point_diff"].tolist() == [1.0, 2.0, 3.0]
    assert frame["home_point_diff"].tolist() == [9.0, 9.0, 9.0]  # input untouched

    with pytest.raises(DataContractError):
        leak_target_into_feature(frame, feature_column="missing", target_column="margin_residual")


# ---------------------------------------------------------------------------
# 8. Opener-archive plumbing (synthetic baseline; no real odds archive needed)
# ---------------------------------------------------------------------------


def _synthetic_baseline(features: pd.DataFrame) -> pd.DataFrame:
    """A hand-built stand-in for ``nfl_ats.clv.opener_pick_evaluation``'s output.

    Only the columns :func:`joint_opener_pick_evaluation` and
    :func:`paired_opener_accuracy` actually read. The reference "model" is
    the simplest possible one -- always pick home to cover -- so
    ``correct_at_open`` is just whether home actually covered; on the
    zero-mean random ``margin_residual`` in :func:`_synthetic_features` that
    reference sits near 50%, a legitimate (if unsophisticated) comparator.
    """

    rows = []
    for _, row in features.iterrows():
        tue_open = float(row["spread_line"]) - 1.0
        close = float(row["spread_line"])
        margin_vs_open = float(row["result"]) - tue_open
        margin_vs_close = float(row["result"]) - close
        rows.append(
            {
                "game_id": row["game_id"],
                "season": int(row["season"]),
                "week": int(row["week"]),
                "tue_open_home_spread": tue_open,
                "close_home_spread": close,
                "margin_vs_open": margin_vs_open,
                "margin_vs_close": margin_vs_close,
                "correct_at_open": 1.0 if margin_vs_open > 0 else 0.0,
            }
        )
    return pd.DataFrame(rows)


def test_joint_opener_pick_evaluation_scores_every_baseline_week_and_agrees_on_the_sign_rule() -> (
    None
):
    features = _synthetic_features(weeks=8, games_per_week=40)
    baseline = _synthetic_baseline(features)

    joint = joint_opener_pick_evaluation(
        baseline, features, feature_columns=_FEATURES, min_train_games=40
    )
    assert set(joint["game_id"]) == set(
        baseline.loc[baseline["week"] > 1, "game_id"]
    )  # week 1 has no prior training rows
    assert (joint["pick_home_at_open"] == joint["predicted_margin_residual_open"].gt(0.0)).all()

    paired = paired_opener_accuracy(baseline, joint)
    assert set(paired.columns) >= {
        "game_id",
        "baseline_correct_open",
        "candidate_correct_open",
        "delta",
    }
    np.testing.assert_allclose(
        paired["delta"].to_numpy(),
        (paired["candidate_correct_open"] - paired["baseline_correct_open"]).to_numpy(),
    )

    bootstrap = opener_accuracy_bootstrap(paired, samples=500, seed=1)
    assert bootstrap["games"] == len(paired)
    assert 0.0 <= bootstrap["probability_positive"] <= 1.0


def test_joint_opener_pick_evaluation_positive_control_reads_hugely_positive() -> None:
    """Leaking margin truth into a feature must make the sign-rule pick hugely accurate.

    Not exactly 1.0: the pipeline's ridge penalty (alpha=10) shrinks even a
    unit-slope, zero-noise relationship, and the earliest scored weeks train
    on very few prior games. The real, full-scale positive control in
    ``docs/mod17_joint_residual_model.md`` reads 96.9%, not 100%, for the
    same reason; this synthetic fixture is smaller still, so the bar here is
    looser than "nearly perfect" but still far above chance (baseline sign
    accuracy on random noise is ~50%).
    """

    features = _synthetic_features(weeks=8, games_per_week=40)
    baseline = _synthetic_baseline(features)
    contaminated = leak_target_into_feature(
        features.assign(margin_residual=features["ats_margin"]),
        feature_column="wind",
        target_column="margin_residual",
    )

    joint = joint_opener_pick_evaluation(
        baseline, contaminated, feature_columns=_FEATURES, min_train_games=40
    )
    paired = paired_opener_accuracy(baseline, joint)
    assert paired["candidate_correct_open"].mean() > 0.85
