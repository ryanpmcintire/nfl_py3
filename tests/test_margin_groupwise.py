"""Group-wise (block-wise) ridge penalties.

Three things are pinned here, in order of importance:

1. **The frozen path is untouched when penalties are unused.** Group-wise
   penalties are opt-in; the default estimator must be the same three-step
   pipeline it has always been, and must predict bit-identically.
2. **The column-scaling trick is exact.** Scaling column ``j`` by
   ``1/sqrt(m_j)`` under a plain ``Ridge(alpha)`` is generalized ridge with
   penalty ``alpha * m_j``, not an approximation of it.
3. **It escapes the MOD-06 corollary.** Differential shrinkage changes the
   direction of the coefficient vector, so ``sign(prediction)`` can flip --
   unlike a positive rescale, which never can.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from nfl_ats.margin import (
    MARGIN_FEATURE_PROFILES,
    MARGIN_TARGETS,
    GroupPenaltyScaler,
    column_penalty_multipliers,
    fit_margin_model,
    make_margin_estimator,
    margin_feature_columns,
    margin_feature_groups,
    margin_model_metadata,
    resolve_feature_groups,
)


def _design(
    rows: int = 400, *, seed: int = 11, missing: bool = True
) -> tuple[pd.DataFrame, np.ndarray]:
    generator = np.random.default_rng(seed)
    columns = [f"c{index}" for index in range(6)]
    frame = pd.DataFrame(generator.normal(size=(rows, len(columns))), columns=columns)
    if missing:
        frame.loc[frame.index[:25], "c2"] = np.nan
    coefficients = np.array([2.0, -1.0, 0.5, 0.0, 1.5, -0.7])
    target = frame.fillna(0.0).to_numpy() @ coefficients + generator.normal(scale=2.0, size=rows)
    return frame, target


# ---------------------------------------------------------------------------
# 1. The frozen path
# ---------------------------------------------------------------------------


def test_default_estimator_is_the_unchanged_three_step_pipeline() -> None:
    estimator = make_margin_estimator("ridge", ridge_alpha=10.0)
    assert list(estimator.named_steps) == ["imputer", "scaler", "regressor"]
    assert estimator.named_steps["regressor"].alpha == 10.0
    # No output container was reconfigured on the frozen path.
    assert estimator.get_params()["imputer"].get_params().get("add_indicator") is True


def test_default_estimator_matches_an_independently_built_reference() -> None:
    frame, target = _design()
    reference = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
            ("scaler", StandardScaler()),
            ("regressor", Ridge(alpha=10.0)),
        ]
    )
    reference.fit(frame, target)
    frozen = make_margin_estimator("ridge", ridge_alpha=10.0)
    frozen.fit(frame, target)
    np.testing.assert_array_equal(frozen.predict(frame), reference.predict(frame))


def test_uniform_multipliers_reproduce_the_frozen_path_exactly() -> None:
    frame, target = _design()
    frozen = make_margin_estimator("ridge", ridge_alpha=10.0)
    frozen.fit(frame, target)
    grouped = make_margin_estimator(
        "ridge", ridge_alpha=10.0, column_penalties=dict.fromkeys(frame.columns, 1.0)
    )
    grouped.fit(frame, target)
    np.testing.assert_array_equal(grouped.predict(frame), frozen.predict(frame))


def test_fit_margin_model_default_carries_no_penalties(model_frame: pd.DataFrame) -> None:
    model = fit_margin_model(model_frame, target="market_residual")
    assert model.column_penalties is None
    # A frozen run's metadata payload must not gain a key.
    assert "column_penalties" not in margin_model_metadata(model)
    assert list(model.estimator.named_steps) == ["imputer", "scaler", "regressor"]


def test_fit_margin_model_records_penalties_when_used(model_frame: pd.DataFrame) -> None:
    columns = margin_feature_columns("market_residual")
    groups = margin_feature_groups("market_residual")
    penalties = column_penalty_multipliers(columns, groups, {"market": 0.1}, normalize=False)
    model = fit_margin_model(model_frame, target="market_residual", column_penalties=penalties)
    assert model.column_penalties is not None
    metadata = margin_model_metadata(model)
    assert metadata["column_penalties"]["spread_line"] == pytest.approx(0.1)
    assert metadata["column_penalties"]["elo_diff"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# 2. Exactness of the column-scaling implementation
# ---------------------------------------------------------------------------


def test_column_scaling_equals_closed_form_generalized_ridge() -> None:
    frame, target = _design()
    multipliers = {"c0": 0.1, "c1": 0.1, "c2": 1.0, "c3": 1.0, "c4": 10.0, "c5": 10.0}
    alpha = 10.0
    pipeline = make_margin_estimator("ridge", ridge_alpha=alpha, column_penalties=multipliers)
    pipeline.fit(frame, target)

    # Rebuild the transformed design the ridge actually saw, then solve the
    # generalized normal equations directly.
    transformed = pipeline.named_steps["scaler"].transform(
        pipeline.named_steps["imputer"].transform(frame)
    )
    penalties = pipeline.named_steps["group_penalty"].penalty_multipliers_
    centred = transformed - transformed.mean(axis=0)
    coefficients = np.linalg.solve(
        centred.T @ centred + alpha * np.diag(penalties), centred.T @ (target - target.mean())
    )
    expected = centred.to_numpy() @ coefficients + target.mean()
    np.testing.assert_allclose(pipeline.predict(frame), expected, atol=1e-9)


def test_missing_indicator_columns_inherit_their_source_block() -> None:
    frame, target = _design()
    multipliers = {"c0": 1.0, "c1": 1.0, "c2": 7.0, "c3": 1.0, "c4": 1.0, "c5": 1.0}
    pipeline = make_margin_estimator("ridge", ridge_alpha=10.0, column_penalties=multipliers)
    pipeline.fit(frame, target)
    scaler = pipeline.named_steps["group_penalty"]
    names = list(scaler.feature_names_in_)
    assert names[-1] == "missingindicator_c2"
    assert scaler.penalty_multipliers_[-1] == pytest.approx(7.0)


def test_scaler_rejects_an_undeclared_column() -> None:
    frame, target = _design(missing=False)
    pipeline = make_margin_estimator("ridge", ridge_alpha=10.0, column_penalties={"c0": 1.0})
    with pytest.raises(ValueError, match="No penalty multiplier declared"):
        pipeline.fit(frame, target)


def test_scaler_requires_named_columns() -> None:
    scaler = GroupPenaltyScaler({"c0": 1.0})
    with pytest.raises(TypeError, match="named columns"):
        scaler.fit(np.zeros((4, 1)))


def test_group_penalties_are_rejected_for_the_boosted_model() -> None:
    with pytest.raises(ValueError, match="only to the ridge"):
        make_margin_estimator("hgb", column_penalties={"c0": 1.0})


# ---------------------------------------------------------------------------
# 3. Block resolution and normalisation
# ---------------------------------------------------------------------------


def test_feature_groups_cover_the_active_model_contract() -> None:
    columns = margin_feature_columns("market_residual", "player")
    groups = margin_feature_groups("market_residual", "player")
    assert len(groups) == len(columns)
    assert set(groups) == {
        "market",
        "context",
        "elo",
        "experience",
        "offense",
        "results",
        "defense",
        "player_qb",
        "player_injuries",
        "player_continuity",
    }
    assert dict(zip(columns, groups, strict=True))["spread_line"] == "market"


@pytest.mark.parametrize("target", MARGIN_TARGETS)
@pytest.mark.parametrize("profile", MARGIN_FEATURE_PROFILES)
def test_every_margin_profile_resolves_to_blocks(target: str, profile: str) -> None:
    """No profile may hit the raise; a new feature family must declare a block."""

    columns = margin_feature_columns(target, profile)  # type: ignore[arg-type]
    groups = margin_feature_groups(target, profile)  # type: ignore[arg-type]
    assert len(groups) == len(columns)
    assert all(groups)


def test_unknown_columns_raise_rather_than_defaulting() -> None:
    with pytest.raises(ValueError, match="No declared feature family"):
        resolve_feature_groups(("spread_line", "not_a_real_feature"))


def test_normalisation_holds_the_average_penalty_fixed() -> None:
    columns = ("a", "b", "c", "d")
    groups = ("light", "light", "heavy", "heavy")
    multipliers = column_penalty_multipliers(
        columns, groups, {"light": 0.25, "heavy": 4.0}, normalize=True
    )
    values = np.array([multipliers[column] for column in columns])
    assert float(np.exp(np.mean(np.log(values)))) == pytest.approx(1.0)
    assert multipliers["c"] / multipliers["a"] == pytest.approx(16.0)


def test_unknown_block_names_are_rejected() -> None:
    with pytest.raises(ValueError, match="Unknown penalty blocks"):
        column_penalty_multipliers(("a",), ("light",), {"heavey": 2.0})


def test_non_positive_multipliers_are_rejected() -> None:
    with pytest.raises(ValueError, match="finite and positive"):
        column_penalty_multipliers(("a",), ("light",), {"light": 0.0})


# ---------------------------------------------------------------------------
# 4. The structural claim: picks can flip
# ---------------------------------------------------------------------------


def test_differential_penalties_flip_prediction_signs() -> None:
    """A positive rescale can never flip a sign; differential shrinkage can.

    Two blocks that disagree about the answer. Penalising one of them harder
    moves the crossing point, so a set of rows changes side -- the property
    MOD-06's corollary denies to any pure-rescale method.
    """

    generator = np.random.default_rng(3)
    rows = 800
    frame = pd.DataFrame(
        {
            "market_a": generator.normal(size=rows),
            "market_b": generator.normal(size=rows),
            "noisy_a": generator.normal(size=rows),
            "noisy_b": generator.normal(size=rows),
        }
    )
    target = (
        1.0 * frame["market_a"] - 1.0 * frame["noisy_a"] + generator.normal(scale=1.0, size=rows)
    ).to_numpy()

    light = make_margin_estimator(
        "ridge",
        ridge_alpha=500.0,
        column_penalties={"market_a": 0.01, "market_b": 0.01, "noisy_a": 100.0, "noisy_b": 100.0},
    ).fit(frame, target)
    uniform = make_margin_estimator(
        "ridge", ridge_alpha=500.0, column_penalties=dict.fromkeys(frame.columns, 1.0)
    ).fit(frame, target)

    flips = int(np.count_nonzero(np.sign(light.predict(frame)) != np.sign(uniform.predict(frame))))
    assert flips > 0
    # And the two predictions are genuinely not proportional, which is the
    # algebraic form of the same statement.
    ratio = light.predict(frame) / uniform.predict(frame)
    assert float(np.nanstd(ratio)) > 1e-6


def test_a_positive_rescale_flips_nothing() -> None:
    """The control the claim above is measured against."""

    frame, target = _design()
    fitted = make_margin_estimator("ridge", ridge_alpha=10.0).fit(frame, target)
    prediction = fitted.predict(frame)
    for factor in (0.01, 0.5, 3.0, 100.0):
        assert np.array_equal(np.sign(prediction * factor), np.sign(prediction))
