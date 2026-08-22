import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from nfl_ats.attribution_waterfall import (
    KIND_FAMILY,
    KIND_FINAL,
    KIND_MARKET,
    KIND_OVERLAY,
    KIND_PROBABILITY_RULE,
    WATERFALL_RECONCILIATION_ATOL,
    GameWaterfall,
    WaterfallInputError,
    build_game_waterfall,
    family_contributions_from_ridge,
    key_number_distance,
    probability_rule_offset,
    read_waterfall_artifact,
    reject_outcome_columns,
    write_waterfall_artifact,
)
from nfl_ats.calibration_distortion import implied_pick_threshold
from nfl_ats.constants import OUTCOME_COLUMNS
from nfl_ats.key_numbers import DEFAULT_KEY_NUMBERS
from nfl_ats.margin import make_margin_estimator

SYNTHETIC_FAMILIES = {"alpha": ("a1", "a2"), "beta": ("b1",)}

TWO_FAMILY_CONTRIBUTIONS = {"alpha": 1.5, "beta": -0.75, "intercept": 0.25}


def _base_overrides() -> dict[str, object]:
    return {
        "game_id": "2026_01_ARI_LAC",
        "market_line": -3.0,
        "predicted_residual": 1.0,
        "family_contributions": dict(TWO_FAMILY_CONTRIBUTIONS),
        "probability_rule_offset_points": 0.4,
        "raw_home_cover_probability": 0.62,
        "families": SYNTHETIC_FAMILIES,
    }


def test_synthetic_two_family_reconciles_exactly() -> None:
    waterfall = build_game_waterfall(
        **_base_overrides(), overlays=[{"overlay": "coach_fade", "fires": True}]
    )
    kinds = [step.kind for step in waterfall.steps]
    assert kinds == [
        KIND_MARKET,
        KIND_FAMILY,
        KIND_FAMILY,
        KIND_FAMILY,
        KIND_PROBABILITY_RULE,
        KIND_OVERLAY,
        KIND_FINAL,
    ]
    assert [step.family for step in waterfall.steps[:4]] == ["market", "alpha", "beta", "intercept"]
    running = 0.0
    for step in waterfall.steps:
        running += step.delta_points
        assert step.cumulative_points == pytest.approx(running, abs=WATERFALL_RECONCILIATION_ATOL)
    assert sum(step.delta_points for step in waterfall.steps) == pytest.approx(
        waterfall.steps[-1].cumulative_points, abs=WATERFALL_RECONCILIATION_ATOL
    )
    assert waterfall.steps[0].delta_points == pytest.approx(3.0)
    assert waterfall.steps[-1].cumulative_points == pytest.approx(-4.4)
    assert waterfall.picked_side == "AWAY"
    assert waterfall.final_probability == pytest.approx(0.38)
    assert waterfall.edge_vs_spread == pytest.approx(4.4)
    assert waterfall.key_number_distance == pytest.approx(0.0)
    assert len(waterfall.flip_events) == 1
    assert waterfall.flip_events[0].overlay == "coach_fade"
    assert waterfall.flip_events[0].would_flip_alone is True


def test_no_firing_overlay_keeps_raw_side() -> None:
    waterfall = build_game_waterfall(
        **_base_overrides(),
        overlays=[{"overlay": "coach_fade", "fires": False}],
    )
    overlay_steps = [step for step in waterfall.steps if step.kind == KIND_OVERLAY]
    assert overlay_steps == []
    assert waterfall.flip_events == ()
    assert waterfall.picked_side == "HOME"
    assert waterfall.final_probability == pytest.approx(0.62)
    assert waterfall.steps[-1].cumulative_points == pytest.approx(4.4)


def test_overlay_flip_ordering_deterministic() -> None:
    scrambled_a = [
        {"overlay": "zeta_overlay", "fires": True},
        {"overlay": "coach_fade", "fires": True},
        {"overlay": "arrest_back_side", "fires": False},
    ]
    scrambled_b = [
        {"overlay": "coach_fade", "fires": True},
        {"overlay": "zeta_overlay", "fires": True},
    ]
    first = build_game_waterfall(**_base_overrides(), overlays=scrambled_a)
    second = build_game_waterfall(**_base_overrides(), overlays=scrambled_b)
    assert first.steps == second.steps
    assert first.flip_events == second.flip_events
    firing = [event.overlay for event in first.flip_events]
    assert firing == ["coach_fade", "zeta_overlay"]
    overlay_deltas = {s.step_id: s.delta_points for s in first.steps if s.kind == KIND_OVERLAY}
    assert overlay_deltas["overlay:coach_fade"] == pytest.approx(-8.8)
    assert overlay_deltas["overlay:zeta_overlay"] == 0.0


def test_unknown_family_and_inconsistent_inputs_rejected() -> None:
    overrides = _base_overrides()
    with pytest.raises(WaterfallInputError, match="Unknown feature families"):
        build_game_waterfall(
            **{**overrides, "family_contributions": {"gamma_not_in_registry": 1.0}},  # type: ignore[arg-type]
        )
    bad_sum = dict(TWO_FAMILY_CONTRIBUTIONS)
    bad_sum["alpha"] = 9.9
    with pytest.raises(WaterfallInputError, match="sum to"):
        build_game_waterfall(**{**_base_overrides(), "family_contributions": bad_sum})  # type: ignore[arg-type]
    overrides["raw_home_cover_probability"] = 0.45
    with pytest.raises(WaterfallInputError, match="deployed prediction row"):
        build_game_waterfall(**overrides)  # type: ignore[arg-type]


def test_key_number_distance_matches_key_numbers_semantics() -> None:
    cases = {
        -6.6: 7.0,
        3.49: 3.0,
        2.5: 2.0,
        10.4: 10.0,
        20.3: 20.0,
        -8.5: 8.0,
    }
    for projected, expected_magnitude in cases.items():
        expected = min(abs(expected_magnitude - k) for k in DEFAULT_KEY_NUMBERS)
        assert key_number_distance(projected) == pytest.approx(expected), projected
    waterfall = build_game_waterfall(**_base_overrides())
    projected_margin = waterfall.market_line + waterfall.predicted_residual + 0.4
    assert waterfall.key_number_distance == pytest.approx(key_number_distance(projected_margin))
    with pytest.raises(ValueError, match="key number"):
        key_number_distance(3.0, key_numbers=[])


def test_probability_rule_offset_matches_implied_threshold() -> None:
    residuals = np.random.default_rng(11).normal(scale=13.0, size=257)
    offset = probability_rule_offset(residuals)
    assert offset == pytest.approx(-implied_pick_threshold(residuals))
    home_pick = implied_pick_threshold(residuals) < 0.75
    assert (0.75 + offset > 0.0) is bool(home_pick)


def _fitted_ridge(frame: pd.DataFrame) -> object:
    y = pd.Series(np.random.default_rng(5).normal(size=len(frame)))
    estimator = make_margin_estimator("ridge", ridge_alpha=5.0)
    estimator.fit(frame.loc[:, ["a1", "a2", "b1"]], y)
    return estimator


def test_family_contributions_from_ridge_reconciles_with_prediction() -> None:
    rng = np.random.default_rng(3)
    frame = pd.DataFrame(
        {
            "a1": rng.normal(size=24),
            "a2": rng.normal(size=24),
            "b1": 2.5,
        }
    )
    estimator = _fitted_ridge(frame)
    totals = family_contributions_from_ridge(
        estimator,
        frame,
        feature_columns=("a1", "a2", "b1"),
        families=SYNTHETIC_FAMILIES,
    )
    assert set(totals[0]) == {"weekly_context", "alpha", "intercept"}
    predicted = np.asarray(estimator.predict(frame.loc[:, ["a1", "a2", "b1"]]), dtype=float)
    for row_totals, prediction in zip(totals, predicted, strict=True):
        assert sum(row_totals.values()) == pytest.approx(float(prediction), abs=1e-6)

    single = family_contributions_from_ridge(
        estimator,
        frame.iloc[[0]],
        feature_columns=("a1", "a2", "b1"),
        families=SYNTHETIC_FAMILIES,
    )
    assert set(single[0]) == {"beta", "alpha", "intercept"}
    single_prediction = float(estimator.predict(frame.iloc[[0]][["a1", "a2", "b1"]])[0])
    assert sum(single[0].values()) == pytest.approx(single_prediction, abs=1e-6)


def test_family_contributions_reject_outcome_columns() -> None:
    frame = pd.DataFrame({"a1": [1.0], "result": [7.0]})
    with pytest.raises(WaterfallInputError, match="outcome columns"):
        family_contributions_from_ridge(
            object(),
            frame,
            feature_columns=("a1",),
            families=SYNTHETIC_FAMILIES,
        )


def test_reject_outcome_columns_covers_registry() -> None:
    for column in OUTCOME_COLUMNS:
        with pytest.raises(WaterfallInputError, match="outcome columns"):
            reject_outcome_columns([column])
    reject_outcome_columns(["spread_line", "elo_diff"])


def test_serialized_schema_carries_no_result_fields() -> None:
    waterfall = build_game_waterfall(**_base_overrides())  # type: ignore[arg-type]
    payload = waterfall.to_dict()
    assert isinstance(payload["steps"], tuple)
    forbidden = set(OUTCOME_COLUMNS)
    assert not forbidden.intersection(payload)
    for step in payload["steps"]:
        assert not forbidden.intersection(step)


def test_artifact_round_trip(tmp_path: Path) -> None:
    out_dir = tmp_path
    games = [
        build_game_waterfall(**_base_overrides()),  # type: ignore[arg-type]
        build_game_waterfall(
            game_id="2026_01_KC_BAL",
            market_line=1.5,
            predicted_residual=-2.25,
            family_contributions={"context": -2.0, "elo": -0.25, "intercept": 0.0},
            probability_rule_offset_points=-0.1,
            raw_home_cover_probability=0.35,
        ),
    ]
    directory = write_waterfall_artifact(games, out_dir)
    assert (directory / "manifest.json").is_file()
    loaded = read_waterfall_artifact(directory)
    expected = json.loads(json.dumps([game.to_dict() for game in games]))
    assert loaded == expected
    manifest = (directory / "manifest.json").read_text(encoding="utf-8")
    assert '"waterfalls.json"' in manifest

    data_path = directory / "waterfalls.json"
    original = data_path.read_text(encoding="utf-8")
    data_path.write_text(original.replace("HOME", "AWAY"), encoding="utf-8")
    with pytest.raises(WaterfallInputError, match="manifest hash"):
        read_waterfall_artifact(directory)


def test_second_game_waterfall_values(tmp_path: object) -> None:
    waterfall: GameWaterfall = build_game_waterfall(
        game_id="2026_01_KC_BAL",
        market_line=1.5,
        predicted_residual=-2.25,
        family_contributions={"context": -2.0, "elo": -0.25, "intercept": 0.0},
        probability_rule_offset_points=-0.1,
        raw_home_cover_probability=0.35,
    )
    assert waterfall.steps[-1].cumulative_points == pytest.approx(-3.85)
    assert waterfall.picked_side == "AWAY"
    assert waterfall.final_probability == pytest.approx(0.35)
    assert waterfall.edge_vs_spread == pytest.approx(3.85)
    assert waterfall.key_number_distance == pytest.approx(0.0)
