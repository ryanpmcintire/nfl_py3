from __future__ import annotations

import json
import os
import time
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

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

from nfl_ats.card_explanation import explain_pick
from nfl_ats.cli_commands import prediction as prediction_cli
from nfl_ats.data import DataContractError
from nfl_ats.home_side_location import ProductionHomeSideOffsets
from nfl_ats.key_line_pick_read import (
    KEY_LINE_ATOMS,
    KEY_LINE_PICK_READ_FILENAME,
    KEY_LINE_PICK_READ_POLICY,
    KeyLinePickRead,
    ServedKeyLineRead,
    key_line_atom,
    key_line_decision_probability,
    key_line_mask,
)
from nfl_ats.key_line_pick_read_incumbent_overlay import (
    CHALLENGER_ID,
    record_key_line_pick_read_incumbent_challenger_decisions,
)
from nfl_ats.margin import MarginModel
from nfl_ats.mass_preserving_lattice import (
    DISCRETE_PUSH_READ_FILENAME,
    THREE_WAY_COLUMNS,
    DiscretePushReader,
    ProductionDiscretePushRead,
    discrete_read,
    prior_pool,
)
from nfl_ats.outcomes import (
    fit_margin_models_for_week,
    score_outcome_week,
)

REPO = Path(__file__).resolve().parents[1]


def test_atoms_are_declared_once_as_three_and_seven() -> None:
    assert KEY_LINE_ATOMS == (3.0, 7.0)
    assert prediction_cli.KEY_LINE_ATOMS is KEY_LINE_ATOMS
    assert KEY_LINE_PICK_READ_POLICY == "key_line_pick_read_v2"
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


def test_decision_number_is_lane_t_cover_plus_half_push() -> None:
    reader = reader_for_2020_week_1(synthetic_pool(push_share=0.10))
    read = reader.read(3.0, 2.0)
    assert read.push > 0.05
    assert key_line_decision_probability(read) == read.cover / (read.cover + read.loss)
    assert key_line_decision_probability(read) == read.home_cover_probability
    assert key_line_decision_probability(read) != pytest.approx(
        read.cover + 0.5 * read.push, abs=1e-6
    )


def _policy() -> KeyLinePickRead:
    return KeyLinePickRead(reader=reader_for_2020_week_1(synthetic_pool(push_share=0.10)))


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
                expected.conditional_cover_probability
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
            ] == pytest.approx(expected_off.conditional_cover_probability)
        else:
            assert not game["touched"]
            assert row["home_cover_probability"] == pytest.approx(
                game["home_cover_probability_discrete"]
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
            by_game[game_id]["home_cover_probability_discrete"]
        )


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


def _lane_t_root() -> Path:
    override = os.environ.get("NFL_ATS_ARTIFACTS_DIR")
    root = Path(override) if override else REPO / "artifacts"
    return root / "research" / "laneT"
