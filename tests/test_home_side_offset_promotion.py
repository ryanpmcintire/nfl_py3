"""MOD-18 lane S promotion: the served home-side offset and its paired challenger.

Covers the production fit path (history precedence, target-week exclusion,
leakage), the point-shift contract on ``MarginModel.predict`` /
``line_sweep``, the sidecar-driven challenger recorder, and the served /
challenger parity that the promotion rests on.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from _overlay_test_kit import write_active_model_and_card, write_challenger_registry

from nfl_ats.data import DataContractError
from nfl_ats.home_side_location import (
    HOME_SIDE_OFFSET_FILENAME,
    HOME_SIDE_OFFSET_POLICY,
    PRIOR_WEIGHT_GAMES,
    archive_prior_stream,
    fit_production_home_side_offsets,
    load_forecast_home_side_offsets,
    prior_rows_before,
)
from nfl_ats.home_side_offset_incumbent_overlay import (
    CHALLENGER_ID,
    record_home_side_offset_incumbent_challenger_decisions,
    uncorrected_card,
)
from nfl_ats.margin import MarginModel
from nfl_ats.outcomes import center_offset_for_games
from nfl_ats.prospective_scoring import load_challenger_decisions


def _archive(rows: int = 240, seed: int = 7) -> pd.DataFrame:
    """A synthetic opener archive: big home favourites under-located by +3."""

    rng = np.random.default_rng(seed)
    seasons = np.repeat([2021, 2022, 2023, 2024, 2025], rows // 5)
    weeks = np.tile(np.arange(1, rows // 5 + 1), 5)
    line = rng.choice([-12.0, -3.0, 2.5, 7.5, 11.0, 13.5], size=rows)
    residual = rng.normal(0.0, 1.0, size=rows)
    true_error = np.where(np.abs(line) > 10.0, 3.0, 0.0)
    result = line + residual + true_error + rng.normal(0.0, 2.0, size=rows)
    return pd.DataFrame(
        {
            "game_id": [
                f"{s}_{w:02d}_A_B{i}" for i, (s, w) in enumerate(zip(seasons, weeks, strict=True))
            ],
            "season": seasons,
            "week": weeks,
            "tue_open_home_spread": line,
            "residual_at_open": residual,
            "result": result,
        }
    )


def _write_evaluation(root: Path, name: str, per_game: pd.DataFrame, model_id: str) -> Path:
    evaluation = root / "opener_evaluation" / name
    evaluation.mkdir(parents=True)
    per_game.to_parquet(evaluation / "per_game.parquet")
    (evaluation / "metadata.json").write_text(
        json.dumps({"active_model_id": model_id}), encoding="utf-8"
    )
    return evaluation


def test_archive_prior_stream_uses_the_out_of_time_point() -> None:
    stream = archive_prior_stream(_archive(10))
    expected = stream["spread_line"] + _archive(10)["residual_at_open"]
    assert np.allclose(stream["point_incumbent"], expected)


def test_prior_rows_exclude_target_week_and_later_and_old_seasons() -> None:
    stream = archive_prior_stream(_archive())
    stream.loc[stream.index[:3], "result"] = np.nan
    prior = prior_rows_before(stream, 2025, 10)
    assert prior["season"].max() == 2025
    assert prior.loc[prior["season"].eq(2025), "week"].max() < 10
    assert not (prior["season"].eq(2025) & prior["week"].ge(10)).any()
    assert prior["season"].min() >= 2020
    assert prior["result"].notna().all()


def test_fit_learns_the_big_spread_home_error_and_nothing_elsewhere(tmp_path: Path) -> None:
    _write_evaluation(tmp_path, "run-a", _archive(), "model-a")
    fitted = fit_production_home_side_offsets(
        tmp_path, {"model_id": "model-a"}, season=2026, week=1
    )
    assert fitted.policy == HOME_SIDE_OFFSET_POLICY
    n = fitted.prior_games["10.5+"]
    assert n > 0
    shrunk = 3.0 * n / (n + PRIOR_WEIGHT_GAMES)
    assert shrunk - 0.6 < fitted.offsets["10.5+"] < shrunk + 0.6
    assert abs(fitted.offsets["0-3"]) < 1.0
    assert fitted.prior_rows == 240


def test_fit_falls_back_to_newest_evaluation_with_a_warning(tmp_path: Path) -> None:
    _write_evaluation(tmp_path, "20260101T000000Z", _archive(seed=1), "model-old")
    _write_evaluation(tmp_path, "20260201T000000Z", _archive(seed=2), "model-older-id-but-newer")
    fitted = fit_production_home_side_offsets(
        tmp_path, {"model_id": "model-new"}, season=2026, week=1
    )
    assert fitted.source_path == "opener_evaluation/20260201T000000Z"
    assert any("newest evaluation" in warning for warning in fitted.warnings)
    assert any("active model is model-new" in warning for warning in fitted.warnings)


def test_fit_serves_zero_offsets_when_no_archive_exists(tmp_path: Path) -> None:
    fitted = fit_production_home_side_offsets(tmp_path, None, season=2026, week=1)
    assert all(value == 0.0 for value in fitted.offsets.values())
    assert fitted.source_path is None
    assert any("zero offsets" in warning for warning in fitted.warnings)


def test_fit_is_leak_safe_against_later_games_and_result_mutations(tmp_path: Path) -> None:
    archive = _archive()
    _write_evaluation(tmp_path, "run-a", archive, "model-a")
    before = fit_production_home_side_offsets(
        tmp_path, {"model_id": "model-a"}, season=2024, week=5
    )
    later = archive.copy()
    mask = later["season"].gt(2024) | (later["season"].eq(2024) & later["week"].ge(5))
    later.loc[mask, "result"] = later.loc[mask, "result"] + 40.0
    extra = later.iloc[:5].copy()
    extra["season"] = 2026
    extra["result"] = extra["result"] + 60.0
    root = tmp_path / "later"
    _write_evaluation(root, "run-a", pd.concat([later, extra], ignore_index=True), "model-a")
    after = fit_production_home_side_offsets(root, {"model_id": "model-a"}, season=2024, week=5)
    assert after.offsets == before.offsets
    assert after.prior_games == before.prior_games


def _model_frame(rows: int = 320, seed: int = 3) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    kickoff = pd.date_range("2020-09-10", periods=rows, freq="12h", tz="UTC")
    spread = rng.normal(0.0, 6.0, size=rows).round(1)
    x = rng.normal(0.0, 1.0, size=rows)
    result = spread + 1.5 * x + rng.normal(0.0, 8.0, size=rows)
    return pd.DataFrame(
        {
            "game_id": [f"G{i}" for i in range(rows)],
            "season": 2020,
            "week": (np.arange(rows) // 16) + 1,
            "kickoff": kickoff,
            "home_team": "HM",
            "away_team": "AW",
            "spread_line": spread,
            "result": result,
            "feature_x": x,
        }
    )


def _fitted_model() -> tuple[MarginModel, pd.DataFrame]:
    """A real ``MarginModel`` on one synthetic feature, built directly so the
    test does not depend on any named feature profile."""

    from sklearn.linear_model import Ridge

    frame = _model_frame()
    train = frame.iloc[:280]
    residual_target = (train["result"] - train["spread_line"]).to_numpy(dtype=float)
    estimator = Ridge(alpha=10.0).fit(train[["feature_x"]], residual_target)
    fitted = estimator.predict(train[["feature_x"]])
    model = MarginModel(
        estimator=estimator,
        residuals=np.asarray(residual_target - fitted, dtype=float),
        model_name="ridge",
        ridge_alpha=10.0,
        target="market_residual",
        feature_columns=("feature_x",),
        training_rows=len(train),
        distribution_rows=len(train),
        training_max_gameday="2020-11-01",
    )
    return model, frame.iloc[280:].reset_index(drop=True)


def test_center_offset_moves_point_residual_and_probability_together() -> None:
    model, target = _fitted_model()
    base = model.predict(target, probability_method="gaussian_median")
    shift = np.full(len(target), 2.0)
    moved = model.predict(target, probability_method="gaussian_median", center_offset=shift)
    assert np.allclose(moved["predicted_margin"], base["predicted_margin"] + 2.0)
    assert np.allclose(moved["fair_spread"], base["fair_spread"] + 2.0)
    assert np.allclose(moved["predicted_market_residual"], base["predicted_market_residual"] + 2.0)
    assert (moved["home_cover_probability"] >= base["home_cover_probability"] - 1e-12).all()
    assert (moved["home_cover_probability"] > base["home_cover_probability"]).any()
    zero = model.predict(
        target, probability_method="gaussian_median", center_offset=np.zeros(len(target))
    )
    pd.testing.assert_frame_equal(zero, base)


def test_center_offset_refuses_misaligned_or_non_finite_shifts() -> None:
    model, target = _fitted_model()
    with pytest.raises(ValueError, match="one value per row"):
        model.predict(target, center_offset=np.zeros(len(target) - 1))
    bad = np.zeros(len(target))
    bad[0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        model.predict(target, center_offset=bad)


def test_line_sweep_carries_the_same_shift() -> None:
    model, target = _fitted_model()
    base = model.line_sweep(target, offsets=(0.0,), probability_method="gaussian_median")
    moved = model.line_sweep(
        target,
        offsets=(0.0,),
        probability_method="gaussian_median",
        center_offset=np.full(len(target), 3.0),
    )
    assert (moved["home_cover_probability"] >= base["home_cover_probability"] - 1e-12).all()
    assert (moved["home_cover_probability"] > base["home_cover_probability"]).any()


def test_center_offset_for_games_defaults_missing_games_to_zero() -> None:
    games = pd.DataFrame({"game_id": ["A", "B", "C"]})
    shift = center_offset_for_games(games, {"A": 1.5, "C": -0.5})
    assert shift.tolist() == [1.5, 0.0, -0.5]


_SEASON, _WEEK = 2026, 1
_FORECAST_DIR = "2026-week-01-forecast"
_MODEL_CONFIG = {
    "method": "market_residual",
    "target": "market_residual",
    "regressor": "ridge",
    "ridge_alpha": 10.0,
    "calibration_method": "none",
    "feature_profile": "weak_stack",
    "min_edge": 0.02,
    "min_train_games": 500,
    "feature_table": "data/processed/game_features_weak_stack.parquet",
}


def _card() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "game_id": ["2026_01_A_B", "2026_01_C_D", "2026_01_E_F"],
            "season": _SEASON,
            "week": _WEEK,
            "kickoff": "2026-09-13T17:00:00+00:00",
            "away_team": ["A", "C", "E"],
            "home_team": ["B", "D", "F"],
            "spread_line": [-3.5, 11.0, 7.5],
            "home_cover_probability": [0.48, 0.52, 0.55],
        }
    )


def _sidecar(card: pd.DataFrame) -> dict[str, object]:
    uncorrected = {"2026_01_A_B": 0.48, "2026_01_C_D": 0.47, "2026_01_E_F": 0.51}
    return {
        "schema": "home_side_offset/1",
        "served": True,
        "fit": {"policy": HOME_SIDE_OFFSET_POLICY, "offsets": {"10.5+": 1.9}},
        "error": None,
        "games": [
            {
                "game_id": row.game_id,
                "home_cover_probability": row.home_cover_probability,
                "home_cover_probability_uncorrected": uncorrected[row.game_id],
                "home_side_offset": 1.9 if row.spread_line > 10 else 0.0,
            }
            for row in card.itertuples()
        ],
    }


def _write_forecast(artifacts: Path, *, with_sidecar: bool = True) -> Path:
    card = _card()
    write_challenger_registry(artifacts, challenger_id=CHALLENGER_ID, model_config=_MODEL_CONFIG)
    write_active_model_and_card(
        artifacts,
        season=_SEASON,
        week=_WEEK,
        created_at_utc="2026-09-08T13:15:00+00:00",
        forecast_dir=_FORECAST_DIR,
        recommendations=card,
        probability_method="gaussian_median",
    )
    forecast = artifacts / "margin_predictions" / _FORECAST_DIR
    if with_sidecar:
        (forecast / HOME_SIDE_OFFSET_FILENAME).write_text(
            json.dumps(_sidecar(card)), encoding="utf-8"
        )
    return forecast


def test_uncorrected_card_swaps_only_the_probability() -> None:
    card = _card()
    paired = uncorrected_card(card, _sidecar(card))
    assert paired["home_cover_probability"].tolist() == [0.48, 0.47, 0.51]
    assert paired.drop(columns="home_cover_probability").equals(
        card.drop(columns="home_cover_probability")
    )
    with pytest.raises(DataContractError, match="lacks the uncorrected read"):
        uncorrected_card(card, {"games": []})


def test_recorder_writes_the_uncorrected_picks_and_reports_the_flips(tmp_path: Path) -> None:
    artifacts = tmp_path / "artifacts"
    forecast = _write_forecast(artifacts)
    assert load_forecast_home_side_offsets(forecast) is not None
    result = record_home_side_offset_incumbent_challenger_decisions(
        artifacts, tmp_path / "data", now=datetime(2026, 9, 8, 14, 0, tzinfo=UTC)
    )
    assert result["recorded"] == 3
    assert result["flip_count"] == 1
    assert result["flipped_game_ids"] == ["2026_01_C_D"]
    ledger = load_challenger_decisions(artifacts)
    mine = ledger.loc[ledger["challenger_id"].eq(CHALLENGER_ID)].set_index("game_id")
    assert mine.loc["2026_01_C_D", "pick_side"] == "AWAY"
    assert mine.loc["2026_01_E_F", "pick_side"] == "HOME"
    assert (mine["bet_side"] == "PASS").all()
    again = record_home_side_offset_incumbent_challenger_decisions(
        artifacts, tmp_path / "data", now=datetime(2026, 9, 8, 15, 0, tzinfo=UTC)
    )
    assert again["recorded"] == 0
    assert again["already_recorded"] == 3


def test_recorder_refuses_a_card_without_the_served_sidecar(tmp_path: Path) -> None:
    artifacts = tmp_path / "artifacts"
    _write_forecast(artifacts, with_sidecar=False)
    with pytest.raises(DataContractError, match="no served"):
        record_home_side_offset_incumbent_challenger_decisions(
            artifacts, tmp_path / "data", now=datetime(2026, 9, 8, 14, 0, tzinfo=UTC)
        )


def test_recorder_refuses_a_fingerprint_drift(tmp_path: Path) -> None:
    artifacts = tmp_path / "artifacts"
    _write_forecast(artifacts)
    write_challenger_registry(
        artifacts, challenger_id=CHALLENGER_ID, model_config={**_MODEL_CONFIG, "ridge_alpha": 3.0}
    )
    with pytest.raises(DataContractError, match="fingerprint"):
        record_home_side_offset_incumbent_challenger_decisions(
            artifacts, tmp_path / "data", now=datetime(2026, 9, 8, 14, 0, tzinfo=UTC)
        )
