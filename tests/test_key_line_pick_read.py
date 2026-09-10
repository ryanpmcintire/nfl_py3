from __future__ import annotations

import ast
import json
import os
import time
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from _overlay_test_kit import write_active_model_and_card, write_challenger_registry
from test_discrete_push_read import (
    allow_whole_number_pool_lines,
    integer_line_week,
    reader_for_2020_week_1,
    synthetic_pool,
)
from test_home_side_offset_promotion import _fitted_model

from nfl_ats import board_content, board_terminal
from nfl_ats.card_explanation import explain_card, explain_pick
from nfl_ats.card_refit import CardRefit, load_card_refit
from nfl_ats.cli_commands import prediction as prediction_cli
from nfl_ats.data import DataContractError
from nfl_ats.ecdf_mapping_incumbent_overlay import apply_ecdf_mapping_incumbent_overlay
from nfl_ats.gaussian_mean_mapping_incumbent_overlay import (
    apply_gaussian_mean_mapping_incumbent_overlay,
)
from nfl_ats.home_side_location import ProductionHomeSideOffsets
from nfl_ats.key_line_pick_read import (
    KEY_LINE_ATOMS,
    KEY_LINE_PICK_READ_FILENAME,
    KEY_LINE_PICK_READ_POLICY,
    KEY_LINE_STATUS_INAPPLICABLE,
    KEY_LINE_STATUS_NOT_RUN,
    KEY_LINE_STATUS_SERVED,
    KeyLinePickRead,
    ServedKeyLineRead,
    apply_key_line_pick_read,
    apply_pick_overrides,
    is_half_point_line,
    key_line_applicability,
    key_line_atom,
    key_line_decision_probability,
    key_line_mask,
    key_line_metadata_block,
    key_line_read_applicable,
    key_line_sidecar,
    key_line_touched_games,
    load_pick_overrides,
    pick_overrides_from_metadata,
    served_pick_overrides,
)
from nfl_ats.key_line_pick_read_incumbent_overlay import (
    CHALLENGER_ID,
    record_key_line_pick_read_incumbent_challenger_decisions,
    smooth_card,
)
from nfl_ats.margin import MarginModel
from nfl_ats.mass_preserving_lattice import (
    DISCRETE_PUSH_READ_FILENAME,
    THREE_WAY_COLUMNS,
    DiscretePushReader,
    ProductionDiscretePushRead,
    ServedPushRead,
    discrete_read,
    prior_pool,
    residual_location,
    serve_discrete_three_way,
)
from nfl_ats.outcomes import (
    fit_margin_models_for_week,
    score_outcome_week,
    score_outcome_week_line_sweep,
)
from nfl_ats.prediction_safety import validate_three_way_split
from nfl_ats.prospective_scoring import load_challenger_decisions
from nfl_ats.public_board import assert_spread_explorer_matches_card
from nfl_ats.spread_explorer import (
    SpreadExplorerGameParams,
    compute_spread_explorer_distribution,
    compute_spread_explorer_params,
    spread_explorer_payload,
)

REPO = Path(__file__).resolve().parents[1]


def test_atoms_are_declared_once_as_three_and_seven() -> None:
    assert KEY_LINE_ATOMS == (3.0, 7.0)
    assert prediction_cli.KEY_LINE_ATOMS is KEY_LINE_ATOMS
    assert KEY_LINE_PICK_READ_POLICY == "key_line_pick_read_v1"
    assert KeyLinePickRead(reader=reader_for_2020_week_1(synthetic_pool())).atoms is KEY_LINE_ATOMS


def test_key_line_mask_selects_exactly_three_and_seven_either_sign() -> None:
    lines = [3.0, -3.0, 7.0, -7.0, 3.5, -3.5, 6.5, 7.5, 6.75, 7.25, 10.0, 14.0, 0.0, 2.5, np.nan]
    mask = key_line_mask(lines)
    assert mask.tolist() == [True, True, True, True] + [False] * 11
    assert key_line_mask(pd.Series(lines)).tolist() == mask.tolist()
    assert not key_line_mask([6.75, 7.25]).any()
    assert key_line_atom(-7.0) == 7.0 and key_line_atom(3.0) == 3.0
    assert key_line_atom(7.5) is None and key_line_atom(float("nan")) is None
    assert key_line_mask([10.0, 14.0], atoms=(3.0, 7.0, 10.0, 14.0)).tolist() == [True, True]
    assert key_line_mask([10.0], atoms=()).tolist() == [False]


def test_applicability_predicate_is_the_vectorised_mask_and_names_half_points() -> None:

    lines = [3.0, -3.0, 7.0, -7.0, 3.5, -2.5, 6.5, 9.5, 6.75, 10.0, 0.0, float("nan")]
    assert [key_line_read_applicable(line) for line in lines] == key_line_mask(lines).tolist()
    assert key_line_read_applicable(None) is False
    assert [is_half_point_line(line) for line in lines] == [
        False, False, False, False, True, True, True, True, False, False, False, False
    ]  # fmt: skip
    assert is_half_point_line(None) is False and is_half_point_line(float("inf")) is False


def test_pool_week_of_half_points_is_inapplicable_not_a_read_that_did_not_run() -> None:

    splash = json.loads(
        (REPO / "data" / "splash" / "2026_week01_20260908_noon.json").read_text(encoding="utf-8")
    )
    pool_lines = [float(game["home_spread"]) for game in splash["games"]]
    assert pool_lines and all(is_half_point_line(line) for line in pool_lines)
    assert not key_line_mask(pool_lines).any()

    applicability = key_line_applicability(pool_lines)
    assert applicability.applicable is False
    assert applicability.games == len(pool_lines) == applicability.half_point_lines
    assert applicability.lines_on_an_atom == 0 and applicability.whole_number_lines == 0
    assert applicability.status == "inapplicable"
    reason = applicability.reason
    assert reason is not None and "half points" in reason and "3 or 7" in reason

    on_atom = key_line_applicability([3.0, 3.5, -7.0])
    assert on_atom.applicable is True and on_atom.reason is None
    assert on_atom.lines_on_an_atom == 2 and on_atom.half_point_lines == 1
    assert on_atom.status == "served"


def test_decision_number_is_lane_t_cover_plus_half_push() -> None:
    reader = reader_for_2020_week_1(synthetic_pool(push_share=0.10))
    read = reader.read(3.0, 2.0)
    assert read.push > 0.05
    assert key_line_decision_probability(read) == read.cover + 0.5 * read.push
    assert key_line_decision_probability(read) == read.home_cover_probability
    assert key_line_decision_probability(read) != pytest.approx(
        read.cover / (read.cover + read.loss), abs=1e-6
    )


def _policy() -> KeyLinePickRead:
    return KeyLinePickRead(reader=reader_for_2020_week_1(synthetic_pool(push_share=0.10)))


def test_override_touches_only_key_line_two_way_after_the_offset(
    model_frame: pd.DataFrame,
) -> None:
    features = integer_line_week(model_frame)
    policy = _policy()
    target = features.loc[features["season"].eq(2020) & features["week"].eq(1)]
    offsets = {str(g): 1.5 for g in target["game_id"]}
    base = score_outcome_week(
        features,
        season=2020,
        week=1,
        min_train_games=80,
        center_offsets=offsets,
        discrete_read=policy.reader,
    )
    log: dict[str, ServedKeyLineRead] = {}
    push_log: dict[str, ServedPushRead] = {}
    served = score_outcome_week(
        features,
        season=2020,
        week=1,
        min_train_games=80,
        center_offsets=offsets,
        discrete_read=policy.reader,
        discrete_read_log=push_log,
        key_line_pick_read=policy,
        key_line_pick_read_log=log,
    )
    unshifted_log: dict[str, ServedKeyLineRead] = {}
    score_outcome_week(
        features,
        season=2020,
        week=1,
        min_train_games=80,
        discrete_read=policy.reader,
        key_line_pick_read=policy,
        key_line_pick_read_log=unshifted_log,
    )
    decision_columns = [
        "home_cover_probability",
        "pick",
        "bet_side",
        "edge",
        "bet_odds",
        "break_even_probability",
    ]
    untouched_columns = [c for c in base.columns if c not in decision_columns]
    pd.testing.assert_frame_equal(base[untouched_columns], served[untouched_columns])
    others = base["method"].ne("market_residual") | base["spread_line"].ne(3.0)
    pd.testing.assert_frame_equal(base.loc[others], served.loc[others])
    ats = served.loc[served["method"].eq("market_residual")].set_index(
        served.loc[served["method"].eq("market_residual"), "game_id"].astype(str)
    )
    touched = ats.loc[ats["spread_line"].eq(3.0)]
    assert not touched.empty
    assert set(log) == set(ats.index)
    for game_id, row in touched.iterrows():
        record = log[str(game_id)]
        assert record.touched and record.atom == 3.0
        assert record.point == push_log[str(game_id)].point
        assert record.point - unshifted_log[str(game_id)].point == pytest.approx(1.5)
        read = policy.reader.read(3.0, record.point)
        assert row["home_cover_probability"] == key_line_decision_probability(read)
        assert record.served == row["home_cover_probability"]
        assert record.push == row["push_probability"]
        assert unshifted_log[str(game_id)].served != record.served
        if "pick" in row:
            assert row["pick"] == ("HOME" if row["home_cover_probability"] >= 0.5 else "AWAY")
    for game_id, row in ats.loc[ats["spread_line"].ne(3.0)].iterrows():
        record = log[str(game_id)]
        assert not record.touched and record.atom is None
        assert record.served == record.smooth == row["home_cover_probability"]


def test_no_policy_is_byte_identical_and_offset_applies_first_in_the_pure_function(
    model_frame: pd.DataFrame,
) -> None:
    features = integer_line_week(model_frame)
    baseline = score_outcome_week(features, season=2020, week=1, min_train_games=80)
    again = score_outcome_week(
        features, season=2020, week=1, min_train_games=80, key_line_pick_read=None
    )
    pd.testing.assert_frame_equal(baseline, again)
    target, models = fit_margin_models_for_week(
        features, season=2020, week=1, min_train_games=80, methods=("market_residual",)
    )
    model = models["market_residual"]
    shifted = model.predict(
        target, probability_method="gaussian_median", center_offset=np.full(len(target), 2.0)
    )
    policy = _policy()
    out = apply_key_line_pick_read(
        shifted, target, policy, residuals=model.residuals, probability_method="gaussian_median"
    )
    location = residual_location(model.residuals, "gaussian_median")
    on_atom = target["spread_line"].eq(3.0).to_numpy()
    expected = shifted["predicted_margin"].to_numpy() + location
    for index in np.flatnonzero(on_atom):
        assert out["home_cover_probability"].iloc[index] == key_line_decision_probability(
            policy.reader.read(3.0, float(expected[index]))
        )
    assert np.array_equal(
        out.loc[~on_atom, "home_cover_probability"], shifted.loc[~on_atom, "home_cover_probability"]
    )
    with pytest.raises(ValueError, match="row-aligned"):
        apply_key_line_pick_read(
            shifted.iloc[:1], target, policy, residuals=model.residuals, probability_method="ecdf"
        )


def test_sweep_reads_the_lattice_at_every_alternative_atom_line(model_frame: pd.DataFrame) -> None:
    features = integer_line_week(model_frame)
    policy = _policy()
    plain = score_outcome_week_line_sweep(
        features, season=2020, week=1, min_train_games=80, discrete_read=policy.reader
    )
    served = score_outcome_week_line_sweep(
        features,
        season=2020,
        week=1,
        min_train_games=80,
        discrete_read=policy.reader,
        key_line_pick_read=policy,
    )
    card = score_outcome_week(
        features,
        season=2020,
        week=1,
        min_train_games=80,
        discrete_read=policy.reader,
        key_line_pick_read=policy,
    )
    ats_card = card.loc[card["method"].eq("market_residual")].set_index(
        card.loc[card["method"].eq("market_residual"), "game_id"].astype(str)
    )
    on_atom = key_line_mask(served["alternative_line"]) & served["method"].eq("market_residual")
    assert on_atom.any()
    off = ~on_atom
    pd.testing.assert_frame_equal(
        plain.loc[off].reset_index(drop=True), served.loc[off].reset_index(drop=True)
    )
    rows = served.loc[on_atom]
    assert (
        rows["pick_probability"]
        == np.where(
            rows["home_cover_probability"] >= 0.5,
            rows["home_cover_probability"],
            1 - rows["home_cover_probability"],
        )
    ).all()
    zero = rows.loc[rows["line_offset"].eq(0.0)]
    assert not zero.empty
    for _, row in zero.iterrows():
        assert (
            row["home_cover_probability"]
            == ats_card.loc[str(row["game_id"]), "home_cover_probability"]
        )


def _served_week(model_frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, ServedKeyLineRead]]:
    features = integer_line_week(model_frame)
    log: dict[str, ServedKeyLineRead] = {}
    predictions = score_outcome_week(
        features,
        season=2020,
        week=1,
        min_train_games=80,
        key_line_pick_read=_policy(),
        key_line_pick_read_log=log,
    )
    return predictions, log


def test_sidecar_and_metadata_carry_both_reads_and_rebuild_the_overrides(
    model_frame: pd.DataFrame, tmp_path: Path
) -> None:
    predictions, log = _served_week(model_frame)
    ats = predictions.loc[predictions["method"].eq("market_residual")]
    ids = sorted(ats["game_id"].astype(str))
    sidecar = key_line_sidecar(_policy(), log, ids)
    json.dumps(sidecar)
    assert sidecar["served"] is True and sidecar["error"] is None
    assert sidecar["policy"] == KEY_LINE_PICK_READ_POLICY and sidecar["atoms"] == [3.0, 7.0]
    assert sidecar["fit"]["decision_number"] == "cover + push / 2"
    assert [g["game_id"] for g in sidecar["games"]] == ids
    touched = [g for g in sidecar["games"] if g["touched"]]
    assert touched and all(g["atom"] == 3.0 for g in touched)
    for game in sidecar["games"]:
        assert {
            "home_cover_probability_smooth",
            "home_cover_probability_discrete",
            "home_cover_probability",
            "side_changed",
            "cover",
            "push",
            "loss",
        } <= set(game)
        assert game["home_cover_probability"] == (
            game["home_cover_probability_discrete"]
            if game["touched"]
            else game["home_cover_probability_smooth"]
        )
        assert game["side_changed"] == (
            (game["home_cover_probability"] >= 0.5)
            != (game["home_cover_probability_smooth"] >= 0.5)
        )
    block = key_line_metadata_block(sidecar)
    assert block["served"] is True and block["games"] == len(ids)
    assert [g["game_id"] for g in block["touched"]] == [g["game_id"] for g in touched]
    assert block["sides_changed"] == [g["game_id"] for g in touched if g["side_changed"]]
    metadata = {"key_line_pick_read": block}
    expected = {g["game_id"]: g["home_cover_probability"] for g in touched}
    assert pick_overrides_from_metadata(metadata) == expected
    assert key_line_touched_games(metadata) == frozenset(expected)
    (tmp_path / KEY_LINE_PICK_READ_FILENAME).write_text(json.dumps(sidecar), encoding="utf-8")
    assert served_pick_overrides(tmp_path) == expected
    assert load_pick_overrides({}, tmp_path) == expected
    assert load_pick_overrides(metadata, tmp_path / "missing") == expected
    assert pick_overrides_from_metadata({}) is None
    assert served_pick_overrides(tmp_path / "missing") is None
    assert served_pick_overrides(None) is None
    assert load_pick_overrides({}, None) is None
    assert key_line_touched_games({}) == frozenset()
    degraded = key_line_sidecar(None, {}, ids, error="no lattice")
    assert degraded["served"] is False and degraded["error"] == "no lattice"
    assert (
        pick_overrides_from_metadata({"key_line_pick_read": key_line_metadata_block(degraded)})
        is None
    )
    (tmp_path / KEY_LINE_PICK_READ_FILENAME).write_text(json.dumps(degraded), encoding="utf-8")
    assert served_pick_overrides(tmp_path) is None
    values = apply_pick_overrides([0.1, 0.2, 0.3], ["a", "b", "c"], {"b": 0.9})
    assert values.tolist() == [0.1, 0.9, 0.3]
    assert apply_pick_overrides(pd.Series([0.1]), ["a"], None).tolist() == [0.1]


def test_sidecar_tells_did_not_apply_apart_from_did_not_run(model_frame: pd.DataFrame) -> None:

    predictions, log = _served_week(model_frame)
    ats = predictions.loc[predictions["method"].eq("market_residual")]
    ids = sorted(ats["game_id"].astype(str))
    policy = _policy()

    served = key_line_sidecar(policy, log, ids)
    assert served["served"] is True and served["status"] == KEY_LINE_STATUS_SERVED
    assert served["applicability"]["applicable"] is True
    assert served["applicability"]["reason"] is None
    assert served["applicability"]["lines_on_an_atom"] > 0
    assert any(game["touched"] for game in served["games"])

    half_point_log = {
        game_id: replace(
            read,
            line=read.line + (0.5 if float(read.line).is_integer() else 0.0),
            atom=None,
            touched=False,
            served=read.smooth,
        )
        for game_id, read in log.items()
    }
    inapplicable = key_line_sidecar(policy, half_point_log, ids)
    assert inapplicable["served"] is True, "the lattice was fitted; only the lines were not atoms"
    assert inapplicable["status"] == KEY_LINE_STATUS_INAPPLICABLE
    assert inapplicable["error"] is None and inapplicable["fit"] is not None
    assert not any(game["touched"] for game in inapplicable["games"])
    reason = inapplicable["applicability"]["reason"]
    assert reason is not None and "half points" in reason
    assert inapplicable["applicability"]["half_point_lines"] == len(ids)
    for game in inapplicable["games"]:
        assert game["inapplicable_reason"] is not None
        assert "half-point line" in game["inapplicable_reason"]
    assert _overrides(inapplicable) == {}

    not_run = key_line_sidecar(None, {}, ids, error="no lattice")
    assert not_run["served"] is False and not_run["status"] == KEY_LINE_STATUS_NOT_RUN
    assert not_run["applicability"] is None
    assert not_run["error"] == "no lattice" and not_run["fit"] is None

    assert key_line_metadata_block(served)["status"] == KEY_LINE_STATUS_SERVED
    block = key_line_metadata_block(inapplicable)
    assert block["status"] == KEY_LINE_STATUS_INAPPLICABLE and block["touched"] == []
    assert block["applicability"]["reason"] == reason
    assert key_line_metadata_block(not_run)["status"] == KEY_LINE_STATUS_NOT_RUN
    assert key_line_metadata_block(not_run)["applicability"] is None
    assert key_line_metadata_block({"served": True})["status"] is None


def _overrides(sidecar: dict[str, object]) -> dict[str, float]:
    return {
        str(game["game_id"]): float(game["home_cover_probability"])
        for game in sidecar["games"]  # type: ignore[index,union-attr]
        if game["touched"]
    }


def test_served_policy_needs_the_lattice_and_never_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    reader = reader_for_2020_week_1(synthetic_pool())
    production = ProductionDiscretePushRead(
        policy="p",
        reader=reader,
        source_path=None,
        source_model_id=None,
        active_model_id=None,
        opener_lines_matched=0,
        warnings=(),
    )
    policy, error = prediction_cli._served_key_line_pick_read(production)
    assert policy is not None and error is None and policy.reader is reader
    policy, error = prediction_cli._served_key_line_pick_read(None)
    assert policy is None and "needs its lattice" in str(error)
    policy, error = prediction_cli._served_key_line_pick_read(
        replace(production, reader=None, error="boom")
    )
    assert policy is None and error is not None and "boom" in error
    monkeypatch.setattr(prediction_cli, "KEY_LINE_PICK_READ_SERVED", False)
    assert prediction_cli._served_key_line_pick_read(production) == (None, None)


def test_margin_predict_serves_the_key_line_read_and_writes_the_sidecar(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, model_frame: pd.DataFrame
) -> None:
    data_root = tmp_path / "data"
    artifacts_root = tmp_path / "artifacts"
    monkeypatch.setenv("NFL_ATS_DATA_DIR", str(data_root))
    monkeypatch.setenv("NFL_ATS_ARTIFACTS_DIR", str(artifacts_root))
    monkeypatch.setenv("NFL_ATS_REGISTRY_DIR", str(tmp_path / "registry"))
    features_path = data_root / "processed" / "game_features.parquet"
    features_path.parent.mkdir(parents=True)
    integer_line_week(model_frame).to_parquet(features_path, index=False)
    allow_whole_number_pool_lines(monkeypatch)
    monkeypatch.setattr(
        prediction_cli,
        "fit_production_home_side_offsets",
        lambda *a, **k: ProductionHomeSideOffsets(
            policy="test",
            offsets={"0-3": 1.25, "3.5-6.5": 0.0, "7": 0.0, "7.5-10": 0.0, "10.5+": 0.0},
            prior_games={},
            source_path=None,
            source_model_id=None,
            active_model_id=None,
            prior_rows=0,
            warnings=(),
        ),
    )
    request = prediction_cli.MarginPredictRequest(
        features=features_path,
        season=2020,
        week=1,
        regressor="ridge",
        min_edge=0.02,
        min_train_games=80,
        feature_profile="base",
        ridge_alpha=10.0,
        probability_method="gaussian_median",
        line_sweep=True,
    )
    result = prediction_cli.orchestrate_margin_predict(request)
    sidecar = json.loads((result.output / KEY_LINE_PICK_READ_FILENAME).read_text(encoding="utf-8"))
    assert sidecar["served"] is True and sidecar["error"] is None
    block = result.metadata["key_line_pick_read"]
    assert block["served"] is True and block["policy"] == KEY_LINE_PICK_READ_POLICY
    assert result.metadata["line_sweep"]["pick_read"] == KEY_LINE_PICK_READ_POLICY
    card = pd.read_csv(result.output / "recommendations.csv")
    by_game = {g["game_id"]: g for g in sidecar["games"]}
    push_by_game = {
        g["game_id"]: g
        for g in json.loads(
            (result.output / "discrete_push_read.json").read_text(encoding="utf-8")
        )["games"]
    }
    offsets_by_game = {
        g["game_id"]: g
        for g in json.loads((result.output / "home_side_offset.json").read_text(encoding="utf-8"))[
            "games"
        ]
    }
    features = pd.read_parquet(features_path)
    pool = prior_pool(features)
    cutoff = pd.Timestamp(
        pd.to_datetime(
            features.loc[features["season"].eq(2020) & features["week"].eq(1), "gameday"]
        ).min()
    )
    touched_ids = []
    for _, row in card.iterrows():
        game = by_game[str(row["game_id"])]
        assert row["home_cover_probability"] == pytest.approx(game["home_cover_probability"])
        assert push_by_game[str(row["game_id"])]["home_cover_probability"] == pytest.approx(
            row["home_cover_probability"]
        )
        assert game["point"] == pytest.approx(push_by_game[str(row["game_id"])]["point"])
        if float(row["spread_line"]) == 3.0:
            touched_ids.append(str(row["game_id"]))
            assert game["touched"] and game["atom"] == 3.0
            expected = discrete_read(
                pool,
                3.0,
                float(game["point"]),
                season=2020,
                week=1,
                cutoff=cutoff,
                game_id=str(row["game_id"]),
            )
            assert row["home_cover_probability"] == pytest.approx(
                expected.cover + 0.5 * expected.push
            )
            assert offsets_by_game[str(row["game_id"])]["home_side_offset"] == 1.25
            uncorrected_point = float(game["point"]) - 1.25
            expected_off = discrete_read(
                pool,
                3.0,
                uncorrected_point,
                season=2020,
                week=1,
                cutoff=cutoff,
                game_id=str(row["game_id"]),
            )
            assert offsets_by_game[str(row["game_id"])][
                "home_cover_probability_uncorrected"
            ] == pytest.approx(expected_off.cover + 0.5 * expected_off.push)
        else:
            assert not game["touched"]
            assert row["home_cover_probability"] == pytest.approx(
                game["home_cover_probability_smooth"]
            )
    assert touched_ids
    assert [g["game_id"] for g in block["touched"]] == touched_ids
    sweep = pd.read_parquet(result.output / "line_sweep.parquet")
    zero = sweep.loc[
        sweep["method"].eq("market_residual") & sweep["line_offset"].eq(0.0)
    ].set_index(
        sweep.loc[
            sweep["method"].eq("market_residual") & sweep["line_offset"].eq(0.0), "game_id"
        ].astype(str)
    )
    for game_id in touched_ids:
        assert zero.loc[game_id, "home_cover_probability"] == pytest.approx(
            by_game[game_id]["home_cover_probability"]
        )
    monkeypatch.setattr(prediction_cli, "KEY_LINE_PICK_READ_SERVED", False)
    time.sleep(1.1)
    off = prediction_cli.orchestrate_margin_predict(request)
    assert off.output != result.output
    assert not (off.output / KEY_LINE_PICK_READ_FILENAME).exists()
    assert "key_line_pick_read" not in off.metadata
    off_card = pd.read_csv(off.output / "recommendations.csv").set_index("game_id")
    for game_id in touched_ids:
        assert off_card.loc[game_id, "home_cover_probability"] == pytest.approx(
            by_game[game_id]["home_cover_probability_smooth"]
        )


def test_card_refit_applies_the_override_after_predict(tmp_path: Path) -> None:
    model, target = _fitted_model()
    base = model.predict(target, probability_method="gaussian_median")
    game_id = str(target["game_id"].iloc[2])
    refit = CardRefit(None, "gaussian_median", (), {game_id: 0.123})
    out = refit.predict(model, target)
    assert out["home_cover_probability"].iloc[2] == 0.123
    others = [i for i in range(len(target)) if i != 2]
    assert np.array_equal(
        out["home_cover_probability"].iloc[others], base["home_cover_probability"].iloc[others]
    )
    pd.testing.assert_frame_equal(
        out.drop(columns="home_cover_probability"), base.drop(columns="home_cover_probability")
    )
    pd.testing.assert_frame_equal(CardRefit(None, "gaussian_median").predict(model, target), base)
    forecast = tmp_path / "forecast"
    forecast.mkdir()
    card = target.copy()
    assert load_card_refit({}, card, forecast).pick_overrides is None
    metadata = {
        "key_line_pick_read": {
            "served": True,
            "touched": [{"game_id": game_id, "home_cover_probability": 0.44}],
        }
    }
    assert load_card_refit(metadata, card, forecast).pick_overrides == {game_id: 0.44}
    (forecast / KEY_LINE_PICK_READ_FILENAME).write_text(
        json.dumps(
            {
                "served": True,
                "games": [{"game_id": game_id, "touched": True, "home_cover_probability": 0.45}],
            }
        ),
        encoding="utf-8",
    )
    assert load_card_refit({}, card, forecast).pick_overrides == {game_id: 0.45}
    assert load_card_refit(metadata, card, forecast).pick_overrides == {game_id: 0.44}


_FEATURE_PROFILE = "base"
_RIDGE_ALPHA = 10.0
_MIN_TRAIN_GAMES = 100
_SEASON, _WEEK = 2020, 4


def _median_card(model_frame: pd.DataFrame, method: str = "gaussian_median") -> pd.DataFrame:
    target, margin_models = fit_margin_models_for_week(
        model_frame,
        season=_SEASON,
        week=_WEEK,
        regressor="ridge",
        min_train_games=_MIN_TRAIN_GAMES,
        feature_profile=_FEATURE_PROFILE,
        ridge_alpha=_RIDGE_ALPHA,
        methods=("market_residual",),
    )
    predicted = margin_models["market_residual"].predict(target, probability_method=method)  # type: ignore[arg-type]
    card = target.copy()
    card["game_id"] = card["game_id"].astype(str)
    card["home_cover_probability"] = predicted["home_cover_probability"].to_numpy()
    return card


def _touched_card(card: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, float]]:

    touched = card.copy()
    game_id = str(touched["game_id"].iloc[0])
    override = 1.0 - float(touched["home_cover_probability"].iloc[0])
    touched.iloc[0, touched.columns.get_loc("home_cover_probability")] = override
    return touched, {game_id: override}


def test_spread_explorer_reproduces_a_pinned_game_and_is_unchanged_without_one(
    model_frame: pd.DataFrame,
) -> None:
    card = _median_card(model_frame)
    kwargs: dict[str, object] = {
        "regressor": "ridge",
        "ridge_alpha": _RIDGE_ALPHA,
        "feature_profile": _FEATURE_PROFILE,
        "min_train_games": _MIN_TRAIN_GAMES,
        "probability_method": "gaussian_median",
    }
    plain = compute_spread_explorer_params(card, model_frame, **kwargs)
    assert all(not p.key_line_pinned for p in plain.values())
    assert all("pinned" not in v for v in spread_explorer_payload(plain).values())
    touched, overrides = _touched_card(card)
    ((game_id, override),) = overrides.items()
    with pytest.raises(DataContractError, match="do not"):
        compute_spread_explorer_params(touched, model_frame, **kwargs)
    params = compute_spread_explorer_params(
        touched, model_frame, pick_overrides=overrides, **kwargs
    )
    assert (
        params[game_id].key_line_pinned and params[game_id].card_home_cover_probability == override
    )
    assert params[game_id].center == plain[game_id].center
    for other in set(params) - {game_id}:
        assert params[other] == plain[other]
    payload = spread_explorer_payload(params)
    assert payload[game_id]["pinned"] == round(override, 6)
    assert_spread_explorer_matches_card(params, touched)
    with pytest.raises(DataContractError):
        assert_spread_explorer_matches_card(plain, touched)
    with pytest.raises(DataContractError, match="does not reproduce"):
        compute_spread_explorer_distribution(touched, model_frame, game_id=game_id, **kwargs)
    distribution = compute_spread_explorer_distribution(
        touched, model_frame, game_id=game_id, pick_overrides=overrides, **kwargs
    )
    assert distribution.key_line_pinned and distribution.card_home_cover_probability == override
    assert not compute_spread_explorer_distribution(
        card, model_frame, game_id=game_id, **kwargs
    ).key_line_pinned


def test_mapping_incumbent_recorders_reproduce_a_touched_card(model_frame: pd.DataFrame) -> None:
    card = _median_card(model_frame)
    touched, overrides = _touched_card(card)
    kwargs: dict[str, object] = {
        "regressor": "ridge",
        "ridge_alpha": _RIDGE_ALPHA,
        "feature_profile": _FEATURE_PROFILE,
        "min_train_games": _MIN_TRAIN_GAMES,
    }
    with pytest.raises(DataContractError, match="do not"):
        apply_gaussian_mean_mapping_incumbent_overlay(touched, model_frame, **kwargs)
    with pytest.raises(DataContractError, match="do not"):
        apply_ecdf_mapping_incumbent_overlay(
            touched, model_frame, probability_method="gaussian_median", **kwargs
        )
    mean_plain = apply_gaussian_mean_mapping_incumbent_overlay(card, model_frame, **kwargs)
    mean_touched = apply_gaussian_mean_mapping_incumbent_overlay(
        touched, model_frame, pick_overrides=overrides, **kwargs
    )
    assert np.array_equal(
        mean_plain.overlaid_predictions["home_cover_probability"],
        mean_touched.overlaid_predictions["home_cover_probability"],
    )
    ecdf_touched = apply_ecdf_mapping_incumbent_overlay(
        touched,
        model_frame,
        probability_method="gaussian_median",
        pick_overrides=overrides,
        **kwargs,
    )
    ecdf_plain = apply_ecdf_mapping_incumbent_overlay(
        card, model_frame, probability_method="gaussian_median", **kwargs
    )
    assert np.array_equal(
        ecdf_plain.overlaid_predictions["home_cover_probability"],
        ecdf_touched.overlaid_predictions["home_cover_probability"],
    )


def test_board_curve_adjuster_and_widget_carry_the_pinned_number() -> None:
    from nfl_ats.spread_explorer import widget_home_cover_probability

    smooth = widget_home_cover_probability(7.0, 6.2, 0.4, 13.0)
    params = SpreadExplorerGameParams(
        game_id="2026_01_NO_DET",
        home_team="DET",
        away_team="NO",
        center=6.2,
        residual_mean=0.4,
        residual_std=13.0,
        card_line=7.0,
        card_home_cover_probability=0.472,
        key_line_pinned=True,
    )
    assert abs(smooth - 0.472) > 0.01
    game = board_content.GameRow(
        game_id="2026_01_NO_DET",
        gameday=datetime(2026, 9, 13).date(),
        weekday_name="Sunday",
        home="DET",
        away="NO",
        market_spread=7.0,
        pick_team="NO",
        pick_probability=0.528,
        confidence_word="lean",
        is_best=False,
        is_flipped=False,
    )
    curve = board_content._build_cover_curve(pd.DataFrame(), game, {params.game_id: params})
    zero = next(point for point in curve if point.offset == 0.0)
    assert zero.probability == pytest.approx(1.0 - 0.472)
    half = next(point for point in curve if point.offset == 0.5)
    assert half.probability == pytest.approx(
        1.0 - widget_home_cover_probability(7.5, 6.2, 0.4, 13.0)
    )
    assert board_content._cover_curve_offset_zero_note(curve, game) is None
    adjuster = board_content._build_adjuster(game, {params.game_id: params})
    assert adjuster is not None and adjuster.pinned_home_cover_probability == 0.472
    unpinned = board_content._build_adjuster(
        game, {params.game_id: replace(params, key_line_pinned=False)}
    )
    assert unpinned is not None and unpinned.pinned_home_cover_probability is None
    dive = SimpleNamespace(
        adjuster=adjuster, pick_team="NO", pick_spread_text="+7", probability_text="52.8%"
    )
    html = board_terminal._adjuster_html(dive, x_min=-4, x_max=4, y_min=0.3, y_max=0.7)
    assert 'data-pinned-p="0.472000"' in html
    plain_html = board_terminal._adjuster_html(
        SimpleNamespace(
            adjuster=unpinned, pick_team="NO", pick_spread_text="+7", probability_text="52.8%"
        ),
        x_min=-4,
        x_max=4,
        y_min=0.3,
        y_max=0.7,
    )
    assert "data-pinned-p" not in plain_html
    assert (
        "pinnedP" in board_terminal._DIVE_SCRIPT
        and "offset === 0 && !isNaN(pinnedP)" in board_terminal._DIVE_SCRIPT
    )
    predictions = pd.DataFrame({"game_id": [params.game_id], "home_cover_probability": [0.472]})
    assert_spread_explorer_matches_card({params.game_id: params}, predictions)
    with pytest.raises(DataContractError):
        assert_spread_explorer_matches_card(
            {params.game_id: replace(params, key_line_pinned=False)}, predictions
        )


_TUESDAY_PUSH_SPLITS: dict[str, tuple[float, float, float]] = {
    "2026_02_AAA_BBB": (0.47, 0.0, 0.53),
    "2026_02_CCC_DDD": (0.45, 0.06, 0.49),
    "2026_02_EEE_FFF": (0.52, 0.0, 0.48),
    "2026_02_GGG_HHH": (0.40, 0.11, 0.49),
}
_TUESDAY_KEY_LINE_SPLIT = (0.38, 0.12, 0.50)


def _refresh_setup(
    tmp_path: Path,
    model_frame: pd.DataFrame,
    *,
    with_sidecar: bool,
    served_value: float = 0.31,
    with_push_sidecar: bool = False,
    key_line_split: tuple[float, float, float] | None = None,
    key_served: bool = True,
    push_served: bool = True,
) -> tuple[Path, Path, Path, dict[str, float]]:
    from test_pick_refresh import (
        GAMES,
        ORIGINAL_LINES,
        _original_rows,
        _reference_probability,
        _target_frame,
        _write_original_card,
    )

    from nfl_ats.active_model import ACTIVE_ATS_MODEL_VERSION
    from nfl_ats.io import atomic_json, atomic_parquet

    artifacts_root = tmp_path / "artifacts"
    data_root = tmp_path / "data"
    forecast = artifacts_root / "margin_predictions" / "2026-week-02-forecast"
    forecast.mkdir(parents=True)
    atomic_json(
        {
            "version": ACTIVE_ATS_MODEL_VERSION,
            "status": "SYNCHRONIZED",
            "method": "market_residual",
            "feature_profile": "base",
            "regressor": "ridge",
            "ridge_alpha": 10.0,
            "probability_method": "ecdf",
            "model_id": "model-1",
            "weekly_forecast": {
                "artifact": "margin_predictions/2026-week-02-forecast",
                "season": 2026,
                "week": 2,
            },
        },
        artifacts_root / "active_ats_model.json",
    )
    reference = _reference_probability(model_frame, GAMES, ORIGINAL_LINES, season=2026, week=2)
    _write_original_card(artifacts_root, _original_rows(reference, flip=False))
    features_path = data_root / "processed" / "game_features.parquet"
    atomic_parquet(_target_frame(model_frame, GAMES), features_path)
    touched = {g: ORIGINAL_LINES[g] for g in ORIGINAL_LINES if abs(ORIGINAL_LINES[g]) == 3.0}
    assert touched == {"2026_02_GGG_HHH": 3.0}
    if with_sidecar:
        games = []
        for game_id, line in ORIGINAL_LINES.items():
            is_touched = game_id in touched
            row: dict[str, object] = {
                "game_id": game_id,
                "spread_line": line,
                "touched": is_touched,
                "atom": 3.0 if is_touched else None,
                "home_cover_probability_smooth": reference[game_id],
                "home_cover_probability": served_value if is_touched else reference[game_id],
            }
            if is_touched and key_line_split is not None:
                row["cover"], row["push"], row["loss"] = key_line_split
            games.append(row)
        (forecast / KEY_LINE_PICK_READ_FILENAME).write_text(
            json.dumps(
                {
                    "served": key_served,
                    "policy": KEY_LINE_PICK_READ_POLICY,
                    "atoms": [3.0, 7.0],
                    "games": games,
                }
            ),
            encoding="utf-8",
        )
    if with_push_sidecar:
        (forecast / DISCRETE_PUSH_READ_FILENAME).write_text(
            json.dumps(
                {
                    "schema": "discrete_push_read/1",
                    "served": push_served,
                    "policy": "discrete_push_read_v1",
                    "games": [
                        {
                            "game_id": game_id,
                            "spread_line": ORIGINAL_LINES[game_id],
                            "home_cover_probability": reference[game_id],
                            "served": {"cover": cover, "push": push, "loss": loss},
                            "smooth": {"cover": 0.5, "push": 0.0, "loss": 0.5},
                        }
                        for game_id, (cover, push, loss) in _TUESDAY_PUSH_SPLITS.items()
                    ],
                }
            ),
            encoding="utf-8",
        )
    return artifacts_root, data_root, features_path, reference


def _frozen_line_refit(features: pd.DataFrame) -> tuple[MarginModel, pd.DataFrame, pd.DataFrame]:

    from test_pick_refresh import MIN_TRAIN_GAMES, ORIGINAL_LINES

    from nfl_ats.lines import apply_external_lines

    target, models = fit_margin_models_for_week(
        features,
        season=2026,
        week=2,
        regressor="ridge",
        min_train_games=MIN_TRAIN_GAMES,
        feature_profile="base",
        ridge_alpha=10.0,
        methods=("market_residual",),
    )
    model = models["market_residual"]
    target = target.copy()
    target["game_id"] = target["game_id"].astype(str)
    overridden = apply_external_lines(
        target,
        pd.DataFrame(
            {"game_id": list(ORIGINAL_LINES), "home_spread": list(ORIGINAL_LINES.values())}
        ),
    )
    return model, overridden, model.predict(overridden, probability_method="ecdf")


def _stand_in_lattice(monkeypatch: pytest.MonkeyPatch) -> DiscretePushReader:

    from nfl_ats import mass_preserving_lattice

    reader = DiscretePushReader.for_week(
        synthetic_pool(), season=2020, week=1, cutoff=pd.Timestamp("2020-09-10")
    )
    monkeypatch.setattr(
        mass_preserving_lattice,
        "fit_production_discrete_push_reader",
        lambda *a, **k: ProductionDiscretePushRead(
            policy="p",
            reader=reader,
            source_path=None,
            source_model_id=None,
            active_model_id=None,
            opener_lines_matched=0,
            warnings=(),
        ),
    )
    return reader


def _refresh_lattice_reads(
    artifacts_root: Path, features_path: Path
) -> tuple[MarginModel, pd.DataFrame, pd.DataFrame, pd.DataFrame]:

    from nfl_ats.pick_refresh import _served_lattice_reads

    active = json.loads((artifacts_root / "active_ats_model.json").read_text(encoding="utf-8"))
    features = pd.read_parquet(features_path)
    model, overridden, forecasts = _frozen_line_refit(features)
    result = _served_lattice_reads(
        artifacts_root,
        active,
        features,
        overridden,
        forecasts,
        residuals=model.residuals,
        probability_method="ecdf",
        season=2026,
        week=2,
    )
    return model, overridden, forecasts, result


def _split(frame: pd.DataFrame, position: int) -> tuple[float, float, float]:
    cover, push, loss = (float(frame[column].iloc[position]) for column in THREE_WAY_COLUMNS)
    return cover, push, loss


def test_refresh_reproduces_the_served_number_and_reapplies_the_policy(
    tmp_path: Path, model_frame: pd.DataFrame, monkeypatch: pytest.MonkeyPatch
) -> None:
    from test_pick_refresh import MIN_TRAIN_GAMES

    from nfl_ats import mass_preserving_lattice
    from nfl_ats.pick_refresh import plan_refresh

    artifacts_root, data_root, features_path, reference = _refresh_setup(
        tmp_path, model_frame, with_sidecar=True
    )
    now = datetime(2026, 9, 16, tzinfo=UTC)
    plan = plan_refresh(
        artifacts_root,
        data_root,
        season=2026,
        week=2,
        features_path=features_path,
        min_train_games=MIN_TRAIN_GAMES,
        now=now,
    )
    by_id = {game.game_id: game for game in plan.games}
    assert by_id["2026_02_GGG_HHH"].new_home_cover_probability == 0.31
    for game_id, value in reference.items():
        if game_id != "2026_02_GGG_HHH":
            assert by_id[game_id].new_home_cover_probability == pytest.approx(value)
    reader = DiscretePushReader.for_week(
        synthetic_pool(), season=2020, week=1, cutoff=pd.Timestamp("2020-09-10")
    )
    monkeypatch.setattr(
        mass_preserving_lattice,
        "fit_production_discrete_push_reader",
        lambda *a, **k: ProductionDiscretePushRead(
            policy="p",
            reader=reader,
            source_path=None,
            source_model_id=None,
            active_model_id=None,
            opener_lines_matched=0,
            warnings=(),
        ),
    )
    plan = plan_refresh(
        artifacts_root,
        data_root,
        season=2026,
        week=2,
        features_path=features_path,
        min_train_games=MIN_TRAIN_GAMES,
        now=now,
    )
    by_id = {game.game_id: game for game in plan.games}
    from test_pick_refresh import GAMES, _target_frame

    from nfl_ats.lines import apply_external_lines

    features = _target_frame(model_frame, GAMES)
    target, models = fit_margin_models_for_week(
        features,
        season=2026,
        week=2,
        regressor="ridge",
        min_train_games=MIN_TRAIN_GAMES,
        feature_profile="base",
        ridge_alpha=10.0,
        methods=("market_residual",),
    )
    model = models["market_residual"]
    from test_pick_refresh import ORIGINAL_LINES

    overridden = apply_external_lines(
        target,
        pd.DataFrame(
            {"game_id": list(ORIGINAL_LINES), "home_spread": list(ORIGINAL_LINES.values())}
        ),
    )
    row = overridden.loc[overridden["game_id"].astype(str).eq("2026_02_GGG_HHH")]
    point = float(
        model.predict(row, probability_method="ecdf")["predicted_margin"].iloc[0]
    ) + residual_location(model.residuals, "ecdf")
    assert by_id["2026_02_GGG_HHH"].new_home_cover_probability == key_line_decision_probability(
        reader.read(3.0, point)
    )
    for game_id, value in reference.items():
        if game_id != "2026_02_GGG_HHH":
            assert by_id[game_id].new_home_cover_probability == pytest.approx(value)


def test_refresh_without_a_sidecar_is_the_pre_promotion_refit(
    tmp_path: Path, model_frame: pd.DataFrame
) -> None:
    from test_pick_refresh import MIN_TRAIN_GAMES

    from nfl_ats.pick_refresh import plan_refresh

    artifacts_root, data_root, features_path, reference = _refresh_setup(
        tmp_path, model_frame, with_sidecar=False
    )
    plan = plan_refresh(
        artifacts_root,
        data_root,
        season=2026,
        week=2,
        features_path=features_path,
        min_train_games=MIN_TRAIN_GAMES,
        now=datetime(2026, 9, 16, tzinfo=UTC),
    )
    for game in plan.games:
        assert game.new_home_cover_probability == pytest.approx(reference[game.game_id])


def test_refresh_frame_split_is_the_lattice_split_beside_the_key_line_pick(
    tmp_path: Path, model_frame: pd.DataFrame, monkeypatch: pytest.MonkeyPatch
) -> None:
    from test_pick_refresh import MIN_TRAIN_GAMES

    from nfl_ats.pick_refresh import plan_refresh

    artifacts_root, data_root, features_path, reference = _refresh_setup(
        tmp_path, model_frame, with_sidecar=True, with_push_sidecar=True
    )
    reader = _stand_in_lattice(monkeypatch)
    model, overridden, forecasts, result = _refresh_lattice_reads(artifacts_root, features_path)
    expected = serve_discrete_three_way(
        forecasts, overridden, reader, residuals=model.residuals, probability_method="ecdf"
    )
    expected = apply_key_line_pick_read(
        expected, overridden, _policy(), residuals=model.residuals, probability_method="ecdf"
    )
    pd.testing.assert_frame_equal(result, expected)
    ids = overridden["game_id"].astype(str).to_list()
    location = residual_location(model.residuals, "ecdf")
    for position, game_id in enumerate(ids):
        line = float(overridden["spread_line"].iloc[position])
        point = float(forecasts["predicted_margin"].iloc[position]) + location
        assert _split(result, position) == reader.read(line, point).three_way()
        if game_id == "2026_02_GGG_HHH":
            cover, push, _ = _split(result, position)
            assert result["home_cover_probability"].iloc[position] == cover + 0.5 * push
            assert result["home_cover_probability"].iloc[position] == key_line_decision_probability(
                reader.read(3.0, point)
            )
        else:
            assert result["home_cover_probability"].iloc[position] == float(
                forecasts["home_cover_probability"].iloc[position]
            )
    touched = ids.index("2026_02_GGG_HHH")
    assert _split(result, touched) != _split(forecasts, touched)
    scored = pd.concat(
        [
            overridden[["game_id", "spread_line"]].reset_index(drop=True),
            result.reset_index(drop=True),
        ],
        axis=1,
    )
    validate_three_way_split(scored, line_column="spread_line")
    plan = plan_refresh(
        artifacts_root,
        data_root,
        season=2026,
        week=2,
        features_path=features_path,
        min_train_games=MIN_TRAIN_GAMES,
        now=datetime(2026, 9, 16, tzinfo=UTC),
    )
    by_id = {game.game_id: game for game in plan.games}
    for position, game_id in enumerate(ids):
        assert by_id[game_id].new_home_cover_probability == float(
            result["home_cover_probability"].iloc[position]
        )
    assert by_id["2026_02_GGG_HHH"].new_home_cover_probability != pytest.approx(
        reference["2026_02_GGG_HHH"]
    )


def test_refresh_serves_the_discrete_split_without_a_key_line_sidecar(
    tmp_path: Path, model_frame: pd.DataFrame, monkeypatch: pytest.MonkeyPatch
) -> None:
    artifacts_root, _, features_path, _ = _refresh_setup(
        tmp_path, model_frame, with_sidecar=False, with_push_sidecar=True
    )
    reader = _stand_in_lattice(monkeypatch)
    model, overridden, forecasts, result = _refresh_lattice_reads(artifacts_root, features_path)
    expected = serve_discrete_three_way(
        forecasts, overridden, reader, residuals=model.residuals, probability_method="ecdf"
    )
    pd.testing.assert_frame_equal(result, expected)
    assert np.array_equal(
        result["home_cover_probability"].to_numpy(dtype=float),
        forecasts["home_cover_probability"].to_numpy(dtype=float),
    )


def test_refresh_fallback_restores_the_split_and_the_pick_from_the_sidecars(
    tmp_path: Path, model_frame: pd.DataFrame
) -> None:
    from test_pick_refresh import MIN_TRAIN_GAMES

    from nfl_ats.pick_refresh import plan_refresh

    artifacts_root, data_root, features_path, reference = _refresh_setup(
        tmp_path,
        model_frame,
        with_sidecar=True,
        with_push_sidecar=True,
        key_line_split=_TUESDAY_KEY_LINE_SPLIT,
    )
    _, overridden, forecasts, result = _refresh_lattice_reads(artifacts_root, features_path)
    ids = overridden["game_id"].astype(str).to_list()
    for position, game_id in enumerate(ids):
        if game_id == "2026_02_GGG_HHH":
            assert _split(result, position) == _TUESDAY_KEY_LINE_SPLIT
            assert result["home_cover_probability"].iloc[position] == 0.31
        else:
            assert _split(result, position) == _TUESDAY_PUSH_SPLITS[game_id]
            assert result["home_cover_probability"].iloc[position] == float(
                forecasts["home_cover_probability"].iloc[position]
            )
    untouched = [
        c for c in forecasts.columns if c not in (*THREE_WAY_COLUMNS, "home_cover_probability")
    ]
    pd.testing.assert_frame_equal(result[untouched], forecasts[untouched])
    plan = plan_refresh(
        artifacts_root,
        data_root,
        season=2026,
        week=2,
        features_path=features_path,
        min_train_games=MIN_TRAIN_GAMES,
        now=datetime(2026, 9, 16, tzinfo=UTC),
    )
    by_id = {game.game_id: game for game in plan.games}
    assert by_id["2026_02_GGG_HHH"].new_home_cover_probability == 0.31
    for game_id, value in reference.items():
        if game_id != "2026_02_GGG_HHH":
            assert by_id[game_id].new_home_cover_probability == pytest.approx(value)


def test_refresh_fallback_restores_from_whichever_sidecar_served(
    tmp_path: Path, model_frame: pd.DataFrame
) -> None:
    artifacts_root, _, features_path, _ = _refresh_setup(
        tmp_path / "push", model_frame, with_sidecar=False, with_push_sidecar=True
    )
    _, overridden, forecasts, result = _refresh_lattice_reads(artifacts_root, features_path)
    ids = overridden["game_id"].astype(str).to_list()
    for position, game_id in enumerate(ids):
        assert _split(result, position) == _TUESDAY_PUSH_SPLITS[game_id]
    assert np.array_equal(
        result["home_cover_probability"].to_numpy(dtype=float),
        forecasts["home_cover_probability"].to_numpy(dtype=float),
    )
    artifacts_root, _, features_path, _ = _refresh_setup(
        tmp_path / "key",
        model_frame,
        with_sidecar=True,
        key_line_split=_TUESDAY_KEY_LINE_SPLIT,
    )
    _, overridden, forecasts, result = _refresh_lattice_reads(artifacts_root, features_path)
    for position, game_id in enumerate(ids):
        if game_id == "2026_02_GGG_HHH":
            assert _split(result, position) == _TUESDAY_KEY_LINE_SPLIT
            assert result["home_cover_probability"].iloc[position] == 0.31
        else:
            assert _split(result, position) == _split(forecasts, position)
    artifacts_root, _, features_path, _ = _refresh_setup(
        tmp_path / "old", model_frame, with_sidecar=True
    )
    _, overridden, forecasts, result = _refresh_lattice_reads(artifacts_root, features_path)
    touched = ids.index("2026_02_GGG_HHH")
    assert result["home_cover_probability"].iloc[touched] == 0.31
    assert _split(result, touched) == _split(forecasts, touched)
    artifacts_root, _, features_path, _ = _refresh_setup(
        tmp_path / "off",
        model_frame,
        with_sidecar=True,
        with_push_sidecar=True,
        key_line_split=_TUESDAY_KEY_LINE_SPLIT,
        key_served=False,
        push_served=False,
    )
    _, _, forecasts, result = _refresh_lattice_reads(artifacts_root, features_path)
    assert result is forecasts


def test_refresh_without_any_sidecar_keeps_the_smooth_split_bit_for_bit(
    tmp_path: Path, model_frame: pd.DataFrame, monkeypatch: pytest.MonkeyPatch
) -> None:
    artifacts_root, _, features_path, _ = _refresh_setup(
        tmp_path, model_frame, with_sidecar=False, with_push_sidecar=False
    )
    _stand_in_lattice(monkeypatch)
    _, _, forecasts, result = _refresh_lattice_reads(artifacts_root, features_path)
    assert result is forecasts
    pd.testing.assert_frame_equal(result, forecasts)


def test_refresh_atom_test_is_keyed_to_the_frozen_tuesday_line(
    tmp_path: Path, model_frame: pd.DataFrame, monkeypatch: pytest.MonkeyPatch
) -> None:
    import copy

    from test_pick_refresh import GAMES, MIN_TRAIN_GAMES, ORIGINAL_LINES, _target_frame

    from nfl_ats.io import atomic_parquet
    from nfl_ats.pick_refresh import plan_refresh

    artifacts_root, data_root, features_path, reference = _refresh_setup(
        tmp_path, model_frame, with_sidecar=True, with_push_sidecar=True
    )
    moved = copy.deepcopy(GAMES)
    for game in moved:
        if game["game_id"] == "2026_02_CCC_DDD":
            game["spread_line"] = 7.0
    assert next(g for g in moved if g["game_id"] == "2026_02_GGG_HHH")["spread_line"] == -3.5
    assert ORIGINAL_LINES["2026_02_GGG_HHH"] == 3.0 and ORIGINAL_LINES["2026_02_CCC_DDD"] == -1.0
    atomic_parquet(_target_frame(model_frame, moved), features_path)
    reader = _stand_in_lattice(monkeypatch)
    model, overridden, forecasts, result = _refresh_lattice_reads(artifacts_root, features_path)
    ids = overridden["game_id"].astype(str).to_list()
    assert dict(zip(ids, overridden["spread_line"].astype(float), strict=True)) == ORIGINAL_LINES
    assert key_line_mask(overridden["spread_line"]).tolist() == [
        game_id == "2026_02_GGG_HHH" for game_id in ids
    ]
    plan = plan_refresh(
        artifacts_root,
        data_root,
        season=2026,
        week=2,
        features_path=features_path,
        min_train_games=MIN_TRAIN_GAMES,
        now=datetime(2026, 9, 16, tzinfo=UTC),
    )
    by_id = {game.game_id: game for game in plan.games}
    touched = ids.index("2026_02_GGG_HHH")
    point = float(forecasts["predicted_margin"].iloc[touched]) + residual_location(
        model.residuals, "ecdf"
    )
    assert by_id["2026_02_GGG_HHH"].new_home_cover_probability == key_line_decision_probability(
        reader.read(3.0, point)
    )
    assert by_id["2026_02_GGG_HHH"].new_home_cover_probability == float(
        result["home_cover_probability"].iloc[touched]
    )
    assert by_id["2026_02_CCC_DDD"].new_home_cover_probability == pytest.approx(
        reference["2026_02_CCC_DDD"]
    )
    drifted = ids.index("2026_02_CCC_DDD")
    assert result["home_cover_probability"].iloc[drifted] == float(
        forecasts["home_cover_probability"].iloc[drifted]
    )
    assert (
        _split(result, drifted)
        == reader.read(
            -1.0,
            float(forecasts["predicted_margin"].iloc[drifted])
            + residual_location(model.residuals, "ecdf"),
        ).three_way()
    )


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
            "game_id": ["2026_01_NO_DET", "2026_01_DEN_KC", "2026_01_NE_SEA"],
            "season": 2026,
            "week": 1,
            "kickoff": "2026-09-13T17:00:00+00:00",
            "away_team": ["NO", "DEN", "NE"],
            "home_team": ["DET", "KC", "SEA"],
            "spread_line": [7.0, 3.0, 3.5],
            "home_cover_probability": [0.472, 0.515, 0.498],
        }
    )


def _sidecar(card: pd.DataFrame, *, served: bool = True) -> dict[str, object]:
    smooth = {"2026_01_NO_DET": 0.516, "2026_01_DEN_KC": 0.518, "2026_01_NE_SEA": 0.498}
    return {
        "schema": "key_line_pick_read/1",
        "served": served,
        "policy": KEY_LINE_PICK_READ_POLICY,
        "atoms": [3.0, 7.0],
        "games": [
            {
                "game_id": row.game_id,
                "spread_line": row.spread_line,
                "touched": row.spread_line in (3.0, 7.0),
                "home_cover_probability_smooth": smooth[row.game_id],
                "home_cover_probability": row.home_cover_probability,
            }
            for row in card.itertuples()
        ],
    }


def _write_forecast(artifacts: Path, *, with_sidecar: bool = True, served: bool = True) -> Path:
    card = _card()
    write_challenger_registry(artifacts, challenger_id=CHALLENGER_ID, model_config=_MODEL_CONFIG)
    write_active_model_and_card(
        artifacts,
        season=2026,
        week=1,
        created_at_utc="2026-09-08T13:15:00+00:00",
        forecast_dir="2026-week-01-forecast",
        recommendations=card,
        probability_method="gaussian_median",
    )
    forecast = artifacts / "margin_predictions" / "2026-week-01-forecast"
    if with_sidecar:
        (forecast / KEY_LINE_PICK_READ_FILENAME).write_text(
            json.dumps(_sidecar(card, served=served)), encoding="utf-8"
        )
    return forecast


def test_smooth_card_swaps_only_the_probability() -> None:
    card = _card()
    paired = smooth_card(card, _sidecar(card))
    assert paired["home_cover_probability"].tolist() == [0.516, 0.518, 0.498]
    assert paired.drop(columns="home_cover_probability").equals(
        card.drop(columns="home_cover_probability")
    )
    with pytest.raises(DataContractError, match="lacks the smooth read"):
        smooth_card(card, {"games": []})


def test_recorder_writes_the_smooth_picks_verbatim_and_reports_the_flip(tmp_path: Path) -> None:
    artifacts = tmp_path / "artifacts"
    _write_forecast(artifacts)
    result = record_key_line_pick_read_incumbent_challenger_decisions(
        artifacts, tmp_path / "data", now=datetime(2026, 9, 8, 14, 0, tzinfo=UTC)
    )
    assert result["recorded"] == 3
    assert result["flip_count"] == 1 and result["flipped_game_ids"] == ["2026_01_NO_DET"]
    ledger = load_challenger_decisions(artifacts)
    mine = ledger.loc[ledger["challenger_id"].eq(CHALLENGER_ID)].set_index("game_id")
    assert mine.loc["2026_01_NO_DET", "pick_side"] == "HOME"
    assert mine.loc["2026_01_DEN_KC", "pick_side"] == "HOME"
    assert mine.loc["2026_01_NE_SEA", "pick_side"] == "AWAY"
    assert (mine["bet_side"] == "PASS").all()
    again = record_key_line_pick_read_incumbent_challenger_decisions(
        artifacts, tmp_path / "data", now=datetime(2026, 9, 8, 15, 0, tzinfo=UTC)
    )
    assert again["recorded"] == 0 and again["already_recorded"] == 3


def test_recorder_refuses_without_a_served_sidecar_or_on_fingerprint_drift(tmp_path: Path) -> None:
    artifacts = tmp_path / "artifacts"
    _write_forecast(artifacts, with_sidecar=False)
    with pytest.raises(DataContractError, match="no served"):
        record_key_line_pick_read_incumbent_challenger_decisions(
            artifacts, tmp_path / "data", now=datetime(2026, 9, 8, 14, 0, tzinfo=UTC)
        )
    _write_forecast(artifacts, served=False)
    with pytest.raises(DataContractError, match="no served"):
        record_key_line_pick_read_incumbent_challenger_decisions(
            artifacts, tmp_path / "data", now=datetime(2026, 9, 8, 14, 0, tzinfo=UTC)
        )
    _write_forecast(artifacts)
    write_challenger_registry(
        artifacts, challenger_id=CHALLENGER_ID, model_config={**_MODEL_CONFIG, "ridge_alpha": 3.0}
    )
    with pytest.raises(DataContractError, match="fingerprint"):
        record_key_line_pick_read_incumbent_challenger_decisions(
            artifacts, tmp_path / "data", now=datetime(2026, 9, 8, 14, 0, tzinfo=UTC)
        )


def test_explanation_says_the_line_sits_on_the_number_in_pool_player_words() -> None:
    base = {
        "game_id": "2026_01_NO_DET",
        "home_team": "DET",
        "away_team": "NO",
        "spread_line": 7.0,
        "home_cover_probability": 0.472,
        "gameday": "2026-09-13",
    }
    text = explain_pick({**base, "push_probability": 0.035}, key_line_read=True).text
    sentence = (
        "The line sits right on 7, a number games land on a lot, so this pick is read off "
        "how games with lines like this actually finished"
    )
    assert sentence in text
    assert "roughly 4 in 100 that ended exactly there, a push, are counted in that chance" in text
    assert text.count("The line sits") == 1
    bare = explain_pick(base, key_line_read=True).text
    assert "actually finished." in bare and "push" not in bare
    plain = explain_pick({**base, "push_probability": 0.035}).text
    assert "read off how games" not in plain and "a push, and that chance is counted here" in plain
    for banned in ("lattice", "discrete", "policy", "atom", "smooth", "KL1", "sidecar"):
        assert banned not in text.lower()
    routed = explain_card(
        [base, {**base, "game_id": "2026_01_NE_SEA", "spread_line": 3.5}],
        key_line_games={"2026_01_NO_DET"},
    )
    assert "read off how games" in routed[0].text and "read off how games" not in routed[1].text


def test_half_point_game_says_there_is_no_tie_rather_than_a_zero_push_chance() -> None:

    base = {
        "game_id": "2026_01_CHI_CAR",
        "home_team": "CAR",
        "away_team": "CHI",
        "spread_line": -2.5,
        "home_cover_probability": 0.488,
        "gameday": "2026-09-13",
    }
    text = explain_pick({**base, "push_probability": 0.0}).text
    assert "The line is a half point, so the game cannot finish exactly on it" in text
    assert "there are no ties here" in text
    assert "in 100" not in text and "0%" not in text and "0.0" not in text
    assert "The line sits" not in text

    whole = explain_pick(
        {**base, "spread_line": 3.0, "push_probability": 0.091},
    ).text
    assert (
        "The line sits right on 3" in whole and "a push, and that chance is counted here" in whole
    )
    assert "cannot finish exactly on it" not in whole

    flagged = explain_pick({**base, "push_probability": 0.0}, key_line_read=True).text
    assert "read off how games" not in flagged
    assert "there are no ties here" in flagged

    for banned in ("lattice", "discrete", "policy", "atom", "smooth", "push chance", "sidecar"):
        assert banned not in text.lower()


def test_publish_and_rehearsal_wire_the_recorder() -> None:
    publishing = (REPO / "src" / "nfl_ats" / "cli_commands" / "publishing.py").read_text(
        encoding="utf-8"
    )
    assert publishing.count('result["key_line_pick_read_off_incumbent_challenger_ledger"]') >= 2
    assert "record_key_line_pick_read_incumbent_challenger_decisions(" in publishing
    rehearsal = (REPO / "scripts" / "lockday_rehearsal.py").read_text(encoding="utf-8")
    tree = ast.parse(rehearsal)
    bindings = next(
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "bindings" for t in node.targets)
    )
    keys = {ast.literal_eval(k) for k in bindings.keys}  # type: ignore[union-attr]
    assert "record_key_line_pick_read_incumbent_challenger_decisions" in keys
    simple = next(
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.AnnAssign)
        and isinstance(node.target, ast.Name)
        and node.target.id == "simple"
    )
    labels = [ast.literal_eval(element.elts[0]) for element in simple.elts]  # type: ignore[attr-defined]
    assert "key_line_pick_read_off_incumbent" in labels
    assert (
        labels.index("key_line_pick_read_off_incumbent")
        == labels.index("home_side_offset_off_incumbent") + 1
    )
    from nfl_ats.cli_commands.publishing import PUBLISH_CHALLENGER_RESULT_KEYS

    registry_path = REPO / "artifacts" / "prospective" / "challengers.json"
    if registry_path.is_file():
        registered = {
            e["challenger_id"]
            for e in json.loads(registry_path.read_text(encoding="utf-8"))["challengers"]
            if e.get("status") == "ACTIVE_PROSPECTIVE"
        }
        assert (CHALLENGER_ID in PUBLISH_CHALLENGER_RESULT_KEYS) == (CHALLENGER_ID in registered)


def _lane_t_root() -> Path:
    override = os.environ.get("NFL_ATS_ARTIFACTS_DIR")
    root = Path(override) if override else REPO / "artifacts"
    return root / "research" / "laneT"


@pytest.mark.skipif(
    not (_lane_t_root() / "scored.parquet").is_file()
    or not (_lane_t_root() / "pool.parquet").is_file(),
    reason="lane T's research artifacts are local-only and absent here",
)
def test_lane_t_kl1b_replays_bit_for_bit_through_the_served_read() -> None:
    root = _lane_t_root()
    pool = pd.read_parquet(root / "pool.parquet")
    scored = pd.read_parquet(root / "scored.parquet")
    gameday = pool.set_index("game_id")["gameday"].reindex(scored["game_id"]).to_numpy()
    assert not pd.isna(gameday).any()
    served = np.full(len(scored), np.nan)
    touched = np.zeros(len(scored), dtype=bool)
    for (season, week), group in scored.assign(gameday=gameday).groupby(
        ["season", "week"], sort=True
    ):
        reader = DiscretePushReader.for_week(
            pool,
            season=int(season),
            week=int(week),
            cutoff=pd.Timestamp(group["gameday"].min()),
            exclude_game_ids=set(group["game_id"].astype(str)),
        )
        games = pd.DataFrame(
            {
                "game_id": group["game_id"].astype(str).to_numpy(),
                "spread_line": group["tue_open_home_spread"].to_numpy(dtype=float),
            }
        )
        forecasts = pd.DataFrame(
            {
                "predicted_margin": group["center_S3"].to_numpy(dtype=float),
                "home_cover_probability": group["p_S3"].to_numpy(dtype=float),
            }
        )
        log: dict[str, ServedKeyLineRead] = {}
        out = apply_key_line_pick_read(
            forecasts,
            games,
            KeyLinePickRead(reader=reader),
            residuals=np.array([0.0]),
            probability_method="gaussian_median",
            log=log,
        )
        served[group.index] = out["home_cover_probability"].to_numpy()
        touched[group.index] = [log[g].touched for g in games["game_id"]]
    assert np.array_equal(touched, scored["touched_KL1b"].to_numpy(dtype=bool))
    assert int(touched.sum()) == 272
    assert np.array_equal(served, scored["p_KL1b"].to_numpy(dtype=float))
    assert np.array_equal(served[~touched], scored.loc[~touched, "p_S3"].to_numpy(dtype=float))
    week1 = json.loads((root / "week1.json").read_text(encoding="utf-8"))
    changed = week1["card_changes"]["KL1b"]
    assert [row["game_id"] for row in changed] == ["2026_01_NO_DET"]
    assert changed[0]["p_S3"] == pytest.approx(0.5158, abs=5e-5)
    assert changed[0]["p_KL1b"] == pytest.approx(0.4720, abs=5e-5)
    assert changed[0]["card_home_S3"] is True and changed[0]["card_home_KL1b"] is False
    rows = pd.read_parquet(root / "week1.parquet")
    assert key_line_mask(rows["spread_line"]).tolist() == rows["touched_KL1b"].tolist()
    assert rows.loc[rows["touched_KL1b"], "game_id"].tolist() == [
        "2026_01_DEN_KC",
        "2026_01_NO_DET",
    ]


def test_discrete_push_log_two_way_is_the_served_number(model_frame: pd.DataFrame) -> None:
    features = integer_line_week(model_frame)
    policy = _policy()
    push_log: dict[str, ServedPushRead] = {}
    predictions = score_outcome_week(
        features,
        season=2020,
        week=1,
        min_train_games=80,
        discrete_read=policy.reader,
        discrete_read_log=push_log,
        key_line_pick_read=policy,
    )
    ats = predictions.loc[predictions["method"].eq("market_residual")]
    for _, row in ats.iterrows():
        assert push_log[str(row["game_id"])].home_cover_probability == row["home_cover_probability"]
