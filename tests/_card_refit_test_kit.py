"""Shared card/sidecar fixtures for own-arm recorder regression tests."""

import json

import numpy as np
from sklearn.dummy import DummyRegressor

from nfl_ats.home_side_location import center_offset_for_frame, center_offsets_from_metadata
from nfl_ats.margin import MarginModel
from nfl_ats.spread_regime import BUCKETS


def corrected_card(forecast, card, mode, *, historical_method="gaussian"):
    card = card.copy()
    card["spread_line"] = np.resize([1.5, 7.5], len(card))
    estimator = DummyRegressor(strategy="constant", constant=-0.4)
    estimator.fit(card[["spread_line"]], np.zeros(len(card)))
    model = MarginModel(
        estimator=estimator,
        residuals=np.array(
            [-17.0, -14.0, -10.0, -7.0, -3.0, 0.0, 3.0, 7.0, 10.0, 14.0, 17.0, 21.0]
        ),
        model_name="ridge",
        ridge_alpha=10.0,
        target="market_residual",
        feature_columns=("spread_line",),
        training_rows=100,
        distribution_rows=12,
        training_max_gameday="2025-12-01",
    )
    metadata_path = forecast / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    offsets = None
    method = historical_method
    if mode != "legacy":
        method = "gaussian_median"
        metadata["probability_method"] = method
        metadata["home_side_offset"] = {
            "served": True,
            "offsets": {bucket: (i + 1) * 1.25 for i, bucket in enumerate(BUCKETS)},
        }
        offsets = center_offsets_from_metadata(metadata, card)
        (forecast / "home_side_offset.json").write_text(
            json.dumps(
                {
                    "served": True,
                    "games": [
                        {"game_id": game_id, "home_side_offset": offset}
                        for game_id, offset in offsets.items()
                    ],
                }
            ),
            encoding="utf-8",
        )
        if mode == "sidecar":
            del metadata["home_side_offset"]
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    expected = model.predict(
        card,
        probability_method=method,
        center_offset=center_offset_for_frame(card, offsets),
    )
    card["home_cover_probability"] = expected["home_cover_probability"].to_numpy()
    card.to_csv(forecast / "recommendations.csv", index=False)
    return model, card


def assert_stack_refit(
    tmp_path, monkeypatch, mode, module, write_registry, write_card, patch_fit, record, now
):
    artifacts = tmp_path / "artifacts"
    write_registry(artifacts)
    forecast, card = write_card(artifacts, tmp_path)
    model, card = corrected_card(forecast, card, mode)
    patch_fit(monkeypatch, card, dict.fromkeys(card.game_id, 0.5))
    target = card.iloc[::-1].copy()
    monkeypatch.setattr(
        module, "fit_margin_models_for_week", lambda *a, **k: (target, {"market_residual": model})
    )
    original_predict = model.predict
    observed = []

    def predict(frame, **kwargs):
        result = original_predict(frame, **kwargs)
        expected = card.set_index("game_id").loc[frame.game_id, "home_cover_probability"]
        np.testing.assert_allclose(result.home_cover_probability, expected, rtol=0, atol=1e-12)
        observed.append(result)
        return result

    monkeypatch.setattr(model, "predict", predict)
    result = record(artifacts, tmp_path, now=now)
    assert len(observed) == 1
    assert result["recorded"] == len(card)
    assert result["picks_differing_from_active"] == 0
    assert bool(result["warnings"]) == (mode == "legacy")
