from __future__ import annotations

import json
import re
import sys
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import scripts.waterfall_feed as wff
from nfl_ats.active_model import ACTIVE_ATS_MODEL_VERSION
from nfl_ats.attribution_waterfall import (
    KIND_FAMILY,
    KIND_FINAL,
    KIND_MARKET,
    KIND_OVERLAY,
    KIND_PROBABILITY_RULE,
)
from nfl_ats.card_view import BestPickNomination, CardView
from nfl_ats.coach_fade_overlay import OverlayResult
from nfl_ats.four_overlay_composition import (
    FourOverlayCompositionResult,
    GameProvenance,
    MemberProvenance,
)
from nfl_ats.margin import MarginModel, make_margin_estimator
from nfl_ats.player_arrests_back_side_overlay import ArrestOverlayResult
from nfl_ats.provenance import sha256_file

FEATURE_COLUMNS = ("elo_diff", "diff_def_epa_per_play", "spread_line")

G1 = "2026_01_GGG_HHH"
G2 = "2026_01_III_JJJ"


def _fitted_model() -> MarginModel:
    rng = np.random.default_rng(11)
    train = pd.DataFrame(
        {
            "elo_diff": rng.normal(size=80),
            "diff_def_epa_per_play": rng.normal(size=80),
            "spread_line": rng.normal(scale=5.0, size=80),
        }
    )
    y = pd.Series(rng.normal(scale=13.0, size=80))
    estimator = make_margin_estimator("ridge", ridge_alpha=1.0)
    estimator.fit(train.loc[:, list(FEATURE_COLUMNS)], y)
    residuals = rng.normal(scale=13.0, size=40)
    return MarginModel(
        estimator=estimator,
        residuals=residuals,
        model_name="ridge",
        ridge_alpha=1.0,
        target="market_residual",
        feature_columns=FEATURE_COLUMNS,
        training_rows=80,
        distribution_rows=40,
        training_max_gameday="2026-01-01",
    )


def _target_frame() -> pd.DataFrame:
    rng = np.random.default_rng(5)
    return pd.DataFrame(
        {
            "game_id": [G1, G2],
            "season": [2026, 2026],
            "week": [1, 1],
            "gameday": ["2026-09-13", "2026-09-13"],
            "home_team": ["HHH", "JJJ"],
            "away_team": ["GGG", "III"],
            "kickoff": [
                "2026-09-13 17:00:00+00:00",
                "2026-09-14 01:15:00+00:00",
            ],
            "spread_line": [-3.0, 4.5],
            "elo_diff": rng.normal(scale=100.0, size=2),
            "diff_def_epa_per_play": rng.normal(size=2),
        }
    )


def _fake_fit_fn(model: MarginModel, target: pd.DataFrame) -> Any:
    def _fit(features: pd.DataFrame, **kwargs: Any) -> tuple[pd.DataFrame, dict[str, Any]]:
        assert kwargs["season"] == 2026 and kwargs["week"] == 1
        assert kwargs["methods"] == ("market_residual",)
        return target.copy(), {"market_residual": model}

    return _fit


def _fake_resolver(raw_probabilities: dict[str, float]) -> Any:
    def _resolve(
        card: pd.DataFrame, sweep: pd.DataFrame, metadata: dict[str, Any], *, data_root: Path
    ) -> Any:
        flipped = card.copy()
        first = flipped.index[0]
        flipped.loc[first, "home_cover_probability"] = (
            1.0 - flipped.loc[first, "home_cover_probability"]
        )
        flip_gid = str(card.loc[first, "game_id"])
        flags = pd.Series(False, index=flipped.index)
        composition = FourOverlayCompositionResult(
            overlaid_predictions=flipped,
            policy_id="test_union",
            policy_fingerprint="fp",
            composition_order=("coach_fade",),
            members=(
                MemberProvenance(
                    member_id="coach_fade",
                    order=0,
                    implementation="test",
                    enabled=True,
                    status="applied",
                    flipped_game_ids=(flip_gid,),
                ),
            ),
            games=(
                GameProvenance(
                    game_id=flip_gid,
                    member_ids=("coach_fade",),
                    raw_home_cover_probability=float(raw_probabilities[flip_gid]),
                    final_home_cover_probability=float(
                        flipped.loc[first, "home_cover_probability"]
                    ),
                ),
            ),
            union_flipped_game_ids=(flip_gid,),
            overlapping_game_ids=(),
            arrest_snapshot_id="snap",
            arrest_snapshot_fetched_at_utc=pd.Timestamp("2026-08-20T00:00:00Z"),
            arrest_safe_index_sha256="abc",
        )
        return CardView(
            flipped,
            OverlayResult(card.copy(), (), (), 2, False),
            ArrestOverlayResult(card.copy(), (), flags.copy(), flags.copy(), False),
            BestPickNomination(None, "", None, "v1", None, "", ""),
            composition,
        )

    return _resolve


def _write_world(tmp_path: Path) -> SimpleNamespace:
    artifacts_root = tmp_path / "artifacts"
    forecast_dir = artifacts_root / "margin_predictions" / "fake"
    forecast_dir.mkdir(parents=True)
    processed_dir = tmp_path / "data" / "processed"
    processed_dir.mkdir(parents=True)

    rng = np.random.default_rng(3)
    features = pd.DataFrame(
        {
            "elo_diff": rng.normal(size=10),
            "diff_def_epa_per_play": rng.normal(size=10),
            "spread_line": rng.normal(scale=4.0, size=10),
        }
    )
    features_path = processed_dir / "game_features_player.parquet"
    features.to_parquet(features_path, index=False)

    built_at = datetime(2026, 8, 19, tzinfo=UTC)
    created_at = datetime(2026, 8, 20, tzinfo=UTC)

    manifest = {
        "version": ACTIVE_ATS_MODEL_VERSION,
        "status": "SYNCHRONIZED",
        "model_id": "test-model",
        "method": "market_residual",
        "feature_profile": "player",
        "regressor": "ridge",
        "ridge_alpha": 10.0,
        "weekly_forecast": {"artifact": "margin_predictions/fake"},
    }
    (artifacts_root / "active_ats_model.json").write_text(json.dumps(manifest), encoding="utf-8")
    metadata = {
        "season": 2026,
        "week": 1,
        "created_at_utc": created_at.isoformat(),
        "active_model_id": "test-model",
        "synchronization_status": "SYNCHRONIZED",
        "min_train_games": 20,
        "provenance": {
            "feature_table": {
                "path": str(features_path),
                "sha256": sha256_file(features_path),
                "manifest": {"built_at_utc": built_at.isoformat()},
            }
        },
    }
    (forecast_dir / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")

    model = _fitted_model()
    target = _target_frame()
    scored = model.predict(target).assign(game_id=target["game_id"].to_numpy()).set_index("game_id")
    offset = wff.probability_rule_offset(model.residuals)
    raw_probabilities: dict[str, float] = {}
    rows: list[dict[str, Any]] = []
    for _, row in target.iterrows():
        residual = float(scored.loc[row["game_id"], "predicted_market_residual"])
        raw_probability = 0.62 if residual + offset >= 0.0 else 0.38
        raw_probabilities[str(row["game_id"])] = raw_probability
        rows.append(
            {
                "game_id": row["game_id"],
                "season": 2026,
                "week": 1,
                "gameday": row["gameday"],
                "home_team": row["home_team"],
                "away_team": row["away_team"],
                "kickoff": row["kickoff"],
                "spread_line": float(row["spread_line"]),
                "predicted_market_residual": residual,
                "home_cover_probability": raw_probability,
                "method": "market_residual",
            }
        )
    recommendations = pd.DataFrame(rows)
    predictions = recommendations.assign(predicted_margin=scored["predicted_margin"].to_numpy())
    predictions.to_csv(forecast_dir / "predictions.csv", index=False)
    recommendations.to_csv(forecast_dir / "recommendations.csv", index=False)
    return SimpleNamespace(artifacts_root=artifacts_root, raw_probabilities=raw_probabilities)


def _build_kwargs(world: SimpleNamespace) -> dict[str, Any]:
    return {
        "now": world.now,
        "fit_fn": world.fit_fn,
        "resolve_card_fn": world.resolve_fn,
    }


@pytest.fixture()
def world(tmp_path: Path) -> SimpleNamespace:
    written = _write_world(tmp_path)
    model = _fitted_model()
    target = _target_frame()
    context = wff.load_feed_context(written.artifacts_root, tmp_path / "data")
    return SimpleNamespace(
        context=context,
        fit_fn=_fake_fit_fn(model, target),
        resolve_fn=_fake_resolver(written.raw_probabilities),
        raw_probabilities=written.raw_probabilities,
        now=datetime(2026, 8, 21, tzinfo=UTC),
        registry_root=tmp_path / "registry",
    )


def test_synthetic_end_to_end_build_and_write(world: SimpleNamespace) -> None:
    feed = wff.build_feed(world.context, **_build_kwargs(world))
    assert feed["schema_version"] == wff.WATERFALL_FEED_SCHEMA_VERSION
    assert len(feed["games"]) == 2
    kickoffs = [game["kickoff"] for game in feed["games"]]
    assert kickoffs == sorted(kickoffs)

    offset = feed["probability_rule_offset_points"]
    by_id = {game["game_id"]: game for game in feed["games"]}
    for game in by_id.values():
        kinds = [step["kind"] for step in game["steps"]]
        assert kinds[0] == KIND_MARKET and kinds[-1] == KIND_FINAL
        assert KIND_PROBABILITY_RULE in kinds and KIND_FAMILY in kinds
        running = 0.0
        for step in game["steps"]:
            running += step["delta_points"]
            assert step["cumulative_points"] == pytest.approx(running, abs=1e-9)
        flip_sign = -1.0 if game["flip_events"] else 1.0
        assert game["steps"][-1]["cumulative_points"] == pytest.approx(
            flip_sign * (game["predicted_residual"] + offset), abs=1e-9
        )
        assert game["edge_vs_spread"] == pytest.approx(abs(game["predicted_residual"]))
        expected_side = "HOME" if game["steps"][-1]["cumulative_points"] >= 0.0 else "AWAY"
        assert game["picked_side"] == expected_side
        assert len(game["rationale_sentences"]) >= 2

    flipped = by_id[G1]
    steady = by_id[G2]
    assert [event["overlay"] for event in flipped["flip_events"]] == ["coach_fade"]
    raw_side = "HOME" if world.raw_probabilities[G1] >= 0.5 else "AWAY"
    assert flipped["picked_side"] != raw_side
    assert flipped["final_probability"] == pytest.approx(
        1.0 - world.raw_probabilities[G1], abs=1e-9
    )
    assert any(sentence.endswith("flips this pick") for sentence in flipped["rationale_sentences"])
    assert len([step for step in flipped["steps"] if step["kind"] == KIND_OVERLAY]) == 1
    assert steady["flip_events"] == []
    assert steady["picked_side"] == ("HOME" if world.raw_probabilities[G2] >= 0.5 else "AWAY")
    assert not any(
        sentence.endswith("flips this pick") for sentence in steady["rationale_sentences"]
    )

    directory = wff.write_feed(
        feed,
        world.context.artifacts_root,
        now=world.now,
        features_path=world.context.features_path,
        registry_root=world.registry_root,
    )
    assert (world.registry_root / "experiments" / "waterfall-feed").is_dir()
    feed_path = directory / "feed.json"
    assert feed_path.is_file() and (directory / "manifest.json").is_file()
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["files"][0]["sha256"] == sha256_file(feed_path)
    assert manifest["games"] == 2
    assert json.loads(feed_path.read_text(encoding="utf-8")) == feed
    pointer = json.loads(
        (world.context.artifacts_root / wff.ARTIFACT_DIRNAME / "latest.json").read_text(
            encoding="utf-8"
        )
    )
    assert pointer["latest"] == directory.name
    assert pointer["manifest_sha256"] == sha256_file(directory / "manifest.json")


def test_feed_is_deterministic_for_fixed_now(world: SimpleNamespace) -> None:
    first = wff.build_feed(world.context, **_build_kwargs(world))
    second = wff.build_feed(world.context, **_build_kwargs(world))
    assert first == second


def test_fail_closed_on_stale_unverifiable_feature_table(tmp_path: Path) -> None:
    written = _write_world(tmp_path)
    metadata_path = written.artifacts_root / "margin_predictions" / "fake" / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    del metadata["provenance"]["feature_table"]["sha256"]
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(wff.WaterfallFeedError, match="freshness"):
        wff.load_feed_context(
            written.artifacts_root,
            tmp_path / "data",
            max_staleness_hours=12.0,
        )


def test_hash_verified_table_age_is_disclosed_not_blocking(tmp_path: Path) -> None:
    written = _write_world(tmp_path)
    context = wff.load_feed_context(
        written.artifacts_root,
        tmp_path / "data",
        max_staleness_hours=12.0,
    )
    assert context.feature_table_hash_verified is True
    assert context.feature_table_age_hours == pytest.approx(24.0, abs=0.1)


def test_fail_closed_on_feature_hash_mismatch(tmp_path: Path) -> None:
    written = _write_world(tmp_path)
    context = wff.load_feed_context(written.artifacts_root, tmp_path / "data")
    pd.read_parquet(context.features_path).assign(extra=1.0).to_parquet(
        context.features_path, index=False
    )
    with pytest.raises(wff.WaterfallFeedError, match="sha256"):
        wff.load_feed_context(written.artifacts_root, tmp_path / "data")


def test_reconciliation_inherited_from_builder(
    world: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[str] = []
    real_builder = wff.build_game_waterfall

    def spy(**kwargs: Any) -> Any:
        calls.append(str(kwargs["game_id"]))
        return real_builder(**kwargs)

    monkeypatch.setattr(wff, "build_game_waterfall", spy)
    feed = wff.build_feed(world.context, **_build_kwargs(world))
    assert sorted(calls) == sorted(game["game_id"] for game in feed["games"])

    def _bad_contributions(estimator: Any, frame: Any, **kwargs: Any) -> list[dict[str, float]]:
        return [{"elo_diff": 99.0}] * len(frame)

    monkeypatch.setattr(wff, "family_contributions_from_ridge", _bad_contributions)
    with pytest.raises(wff.WaterfallFeedError, match=r"[Rr]econcil"):
        wff.build_feed(world.context, **_build_kwargs(world))


def test_rationale_numbers_are_field_sourced_only(world: SimpleNamespace) -> None:
    feed = wff.build_feed(world.context, **_build_kwargs(world))
    wff.audit_rationale_numbers(feed)
    pattern = r"-?\d+\.\d+"
    all_sentences: list[str] = []
    for game in feed["games"]:
        allowed = wff.allowed_rationale_numbers(game, feed["probability_rule_offset_points"])
        assert allowed
        for sentence in game["rationale_sentences"]:
            all_sentences.append(sentence)
            for token in re.findall(pattern, sentence):
                assert token.lstrip("+-") in allowed, (sentence, token)
    joined = "\n".join(all_sentences)
    assert re.findall(pattern, joined)


def test_audit_rejects_unsourced_number(world: SimpleNamespace) -> None:
    feed = wff.build_feed(world.context, **_build_kwargs(world))
    feed["games"][0]["rationale_sentences"].append("A unicorn adds 7.77 points here")
    with pytest.raises(wff.WaterfallFeedError, match="field-sourced"):
        wff.audit_rationale_numbers(feed)
