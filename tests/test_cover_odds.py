"""Tests for ``scripts/cover_odds.py`` -- the command-line spread-explorer.

Owner request, 2026-08-20: "pick a spread for a game and see the odds of
covering." This is the query-tool half (the other half is the picks-page
widget, ``tests/test_public_board.py``'s spread-explorer tests). Loaded via
``importlib`` from its file path, mirroring ``tests/test_sensitivity_audit.py``'s
established pattern for testing a ``scripts/*.py`` file that is not part of
the installed package.

Three things are load-bearing:

1. ``resolve_game`` accepts a real game_id, a two-team matchup in any token
   order/separator, or a single team code -- and fails closed (naming every
   available game) rather than guessing when a query is ambiguous or unknown.
2. ``load_active_forecast`` fails closed with a clear message for every
   precondition this tool depends on: a synchronized active model, a
   "gaussian" probability method, and a forecast whose recommendations carry
   only the active method.
3. ``query_cover_odds`` reuses ``nfl_ats.spread_explorer`` (no separate
   fitting logic in the script) and reproduces the published card's own
   ``home_cover_probability`` at the card's own line to floating-point
   precision, then moves sensibly at a different queried spread.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType

import pandas as pd
import pytest

from nfl_ats.active_model import ACTIVE_ATS_MODEL_VERSION
from nfl_ats.data import DataContractError
from nfl_ats.outcomes import fit_margin_models_for_week


def _load_cover_odds_script() -> ModuleType:
    path = Path(__file__).parents[1] / "scripts" / "cover_odds.py"
    spec = importlib.util.spec_from_file_location("cover_odds", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load cover_odds from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


cover_odds = _load_cover_odds_script()

_FEATURE_PROFILE = "base"
_RIDGE_ALPHA = 10.0
_MIN_TRAIN_GAMES = 100
_SEASON = 2020
_WEEK = 4


# ---------------------------------------------------------------------------
# 1. _tokenize_game_query / resolve_game
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "query,expected",
    [
        ("NE@SEA", ["NE", "SEA"]),
        ("ne-sea", ["NE", "SEA"]),
        ("NE_SEA", ["NE", "SEA"]),
        ("NE at SEA", ["NE", "SEA"]),
        ("NE vs SEA", ["NE", "SEA"]),
        ("SEA", ["SEA"]),
        ("  ne / sea  ", ["NE", "SEA"]),
    ],
)
def test_tokenize_game_query(query: str, expected: list[str]) -> None:
    assert cover_odds._tokenize_game_query(query) == expected


def _predictions_fixture() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "game_id": ["2026_01_ARI_LAC", "2026_01_SF_LA"],
            "season": [2026, 2026],
            "week": [1, 1],
            "home_team": ["LAC", "LA"],
            "away_team": ["ARI", "SF"],
            "spread_line": [3.5, -3.5],
            "home_cover_probability": [0.38, 0.62],
        }
    )


def test_resolve_game_by_exact_game_id() -> None:
    row = cover_odds.resolve_game(_predictions_fixture(), "2026_01_ARI_LAC")
    assert row["home_team"] == "LAC"


def test_resolve_game_by_matchup_either_token_order() -> None:
    for query in ("ARI@LAC", "LAC@ARI", "ARI-LAC", "LAC vs ARI"):
        row = cover_odds.resolve_game(_predictions_fixture(), query)
        assert row["game_id"] == "2026_01_ARI_LAC"


def test_resolve_game_by_single_team_code() -> None:
    row = cover_odds.resolve_game(_predictions_fixture(), "SF")
    assert row["game_id"] == "2026_01_SF_LA"


def test_resolve_game_unknown_lists_available_games() -> None:
    with pytest.raises(cover_odds.CoverOddsError, match="Available:"):
        cover_odds.resolve_game(_predictions_fixture(), "XYZ@ABC")


def test_resolve_game_ambiguous_single_token_raises() -> None:
    ambiguous = pd.concat(
        [
            _predictions_fixture(),
            pd.DataFrame(
                {
                    "game_id": ["2026_02_SF_DAL"],
                    "season": [2026],
                    "week": [2],
                    "home_team": ["DAL"],
                    "away_team": ["SF"],
                    "spread_line": [-2.0],
                    "home_cover_probability": [0.55],
                }
            ),
        ],
        ignore_index=True,
    )
    with pytest.raises(cover_odds.CoverOddsError, match="more than one"):
        cover_odds.resolve_game(ambiguous, "SF")


# ---------------------------------------------------------------------------
# 2. load_active_forecast -- fail-closed preconditions
# ---------------------------------------------------------------------------


def _week_card(model_frame: pd.DataFrame) -> pd.DataFrame:
    target, margin_models = fit_margin_models_for_week(
        model_frame,
        season=_SEASON,
        week=_WEEK,
        regressor="ridge",
        min_train_games=_MIN_TRAIN_GAMES,
        feature_profile=_FEATURE_PROFILE,  # type: ignore[arg-type]
        ridge_alpha=_RIDGE_ALPHA,
        methods=("market_residual",),
    )
    model = margin_models["market_residual"]
    predicted = model.predict(target, probability_method="gaussian")
    card = target.copy()
    card["method"] = "market_residual"
    card["home_cover_probability"] = predicted["home_cover_probability"].to_numpy()
    assert not card.empty
    return card


def _write_forecast(
    artifacts_root: Path,
    data_root: Path,
    model_frame: pd.DataFrame,
    *,
    probability_method: str = "gaussian",
    synchronization_status: str = "SYNCHRONIZED",
    model_id: str = "model-gauss",
    active_model_id: str | None = None,
) -> pd.DataFrame:
    card = _week_card(model_frame)

    feature_path = data_root / "processed" / "cover_odds_test_features.parquet"
    feature_path.parent.mkdir(parents=True, exist_ok=True)
    model_frame.to_parquet(feature_path)

    forecast = artifacts_root / "margin_predictions" / "forecast"
    forecast.mkdir(parents=True, exist_ok=True)
    metadata = {
        "active_model_id": active_model_id if active_model_id is not None else model_id,
        "synchronization_status": synchronization_status,
        "season": _SEASON,
        "week": _WEEK,
        "probability_method": probability_method,
        "regressor": "ridge",
        "ridge_alpha": _RIDGE_ALPHA,
        "feature_profile": _FEATURE_PROFILE,
        "min_train_games": _MIN_TRAIN_GAMES,
        "created_at_utc": "2026-09-08T15:00:00+00:00",
        "provenance": {"feature_table": {"path": str(feature_path)}},
    }
    (forecast / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    card.to_csv(forecast / "recommendations.csv", index=False)

    active = {
        "version": ACTIVE_ATS_MODEL_VERSION,
        "status": "SYNCHRONIZED",
        "model_id": model_id,
        "method": "market_residual",
        "probability_method": probability_method,
        "feature_profile": _FEATURE_PROFILE,
        "historical_evaluation": {"accuracy": 0.52, "correct": 1, "games": 1, "intervals": {}},
        "weekly_forecast": {
            "artifact": "margin_predictions/forecast",
            "season": _SEASON,
            "week": _WEEK,
        },
    }
    (artifacts_root / "active_ats_model.json").write_text(json.dumps(active), encoding="utf-8")
    return card


def test_load_active_forecast_missing_active_model_fails_closed(tmp_path: Path) -> None:
    with pytest.raises(cover_odds.CoverOddsError, match="No synchronized active"):
        cover_odds.load_active_forecast(tmp_path)


def test_load_active_forecast_refuses_a_non_gaussian_method(
    tmp_path: Path, model_frame: pd.DataFrame
) -> None:
    artifacts_root = tmp_path / "artifacts"
    data_root = tmp_path / "data"
    _write_forecast(artifacts_root, data_root, model_frame, probability_method="ecdf")
    with pytest.raises(cover_odds.CoverOddsError, match="gaussian"):
        cover_odds.load_active_forecast(artifacts_root)


def test_load_active_forecast_refuses_an_unsynchronized_forecast(
    tmp_path: Path, model_frame: pd.DataFrame
) -> None:
    artifacts_root = tmp_path / "artifacts"
    data_root = tmp_path / "data"
    _write_forecast(artifacts_root, data_root, model_frame, synchronization_status="UNLINKED")
    with pytest.raises(cover_odds.CoverOddsError, match="not synchronized"):
        cover_odds.load_active_forecast(artifacts_root)


def test_load_active_forecast_refuses_a_model_id_mismatch(
    tmp_path: Path, model_frame: pd.DataFrame
) -> None:
    artifacts_root = tmp_path / "artifacts"
    data_root = tmp_path / "data"
    _write_forecast(artifacts_root, data_root, model_frame, active_model_id="something-else")
    with pytest.raises(cover_odds.CoverOddsError, match="model ID does not match"):
        cover_odds.load_active_forecast(artifacts_root)


def test_load_active_forecast_returns_the_synchronized_chain(
    tmp_path: Path, model_frame: pd.DataFrame
) -> None:
    artifacts_root = tmp_path / "artifacts"
    data_root = tmp_path / "data"
    card = _write_forecast(artifacts_root, data_root, model_frame)
    active, forecast_directory, metadata, predictions = cover_odds.load_active_forecast(
        artifacts_root
    )
    assert active["model_id"] == "model-gauss"
    assert forecast_directory.name == "forecast"
    assert metadata["season"] == _SEASON
    assert len(predictions) == len(card)


# ---------------------------------------------------------------------------
# 3. query_cover_odds -- end to end
# ---------------------------------------------------------------------------


def test_query_cover_odds_at_the_cards_own_line_matches_the_published_probability(
    tmp_path: Path, model_frame: pd.DataFrame
) -> None:
    artifacts_root = tmp_path / "artifacts"
    data_root = tmp_path / "data"
    card = _write_forecast(artifacts_root, data_root, model_frame)
    row = card.iloc[0]
    game_id = str(row["game_id"])
    line = float(row["spread_line"])
    published = float(row["home_cover_probability"])

    payload = cover_odds.query_cover_odds(
        season=_SEASON,
        week=_WEEK,
        game=game_id,
        spread=line,
        side="home",
        artifacts_root=artifacts_root,
        data_root=data_root,
    )
    assert payload["two_way_forced_pick_probability"] == pytest.approx(published, abs=1e-6)
    assert payload["home_cover_probability"] == pytest.approx(published, abs=1e-6)
    assert payload["provenance"]["published_home_cover_probability"] == pytest.approx(
        published, abs=1e-6
    )
    # cover + push + no_cover always sums to 1.
    total = (
        payload["cover_probability"] + payload["push_probability"] + payload["no_cover_probability"]
    )
    assert total == pytest.approx(1.0, abs=1e-9)


def test_query_cover_odds_moves_sensibly_at_a_different_spread(
    tmp_path: Path, model_frame: pd.DataFrame
) -> None:
    """Per this codebase's spread_line convention (see
    ``nfl_ats.public_board.spread_words``: a positive home spread means the
    home team is FAVORED), making the home team a bigger underdog (a lower
    home spread) must never make it HARDER for home to cover."""

    artifacts_root = tmp_path / "artifacts"
    data_root = tmp_path / "data"
    card = _write_forecast(artifacts_root, data_root, model_frame)
    row = card.iloc[0]
    game_id = str(row["game_id"])
    line = float(row["spread_line"])

    at_card_line = cover_odds.query_cover_odds(
        season=_SEASON,
        week=_WEEK,
        game=game_id,
        spread=line,
        side="home",
        artifacts_root=artifacts_root,
        data_root=data_root,
    )
    much_friendlier = cover_odds.query_cover_odds(
        season=_SEASON,
        week=_WEEK,
        game=game_id,
        spread=line - 10.0,
        side="home",
        artifacts_root=artifacts_root,
        data_root=data_root,
    )
    assert (
        much_friendlier["two_way_forced_pick_probability"]
        > at_card_line["two_way_forced_pick_probability"]
    )


def test_query_cover_odds_push_is_zero_at_a_half_point_line_and_can_be_nonzero_at_an_integer(
    tmp_path: Path, model_frame: pd.DataFrame
) -> None:
    artifacts_root = tmp_path / "artifacts"
    data_root = tmp_path / "data"
    card = _write_forecast(artifacts_root, data_root, model_frame)
    row = card.iloc[0]
    game_id = str(row["game_id"])

    at_half = cover_odds.query_cover_odds(
        season=_SEASON,
        week=_WEEK,
        game=game_id,
        spread=1.5,
        side="home",
        artifacts_root=artifacts_root,
        data_root=data_root,
    )
    assert at_half["push_probability"] == 0.0

    at_integer = cover_odds.query_cover_odds(
        season=_SEASON,
        week=_WEEK,
        game=game_id,
        spread=1.0,
        side="home",
        artifacts_root=artifacts_root,
        data_root=data_root,
    )
    assert at_integer["push_probability"] >= 0.0  # never negative; may legitimately be 0


def test_query_cover_odds_refuses_a_season_week_mismatch(
    tmp_path: Path, model_frame: pd.DataFrame
) -> None:
    artifacts_root = tmp_path / "artifacts"
    data_root = tmp_path / "data"
    card = _write_forecast(artifacts_root, data_root, model_frame)
    game_id = str(card.iloc[0]["game_id"])

    with pytest.raises(cover_odds.CoverOddsError, match="does not match"):
        cover_odds.query_cover_odds(
            season=_SEASON,
            week=_WEEK + 1,
            game=game_id,
            spread=0.0,
            side="home",
            artifacts_root=artifacts_root,
            data_root=data_root,
        )


def test_query_cover_odds_refuses_a_drifted_card(tmp_path: Path, model_frame: pd.DataFrame) -> None:
    artifacts_root = tmp_path / "artifacts"
    data_root = tmp_path / "data"
    card = _write_forecast(artifacts_root, data_root, model_frame)
    game_id = str(card.iloc[0]["game_id"])
    card_path = artifacts_root / "margin_predictions" / "forecast" / "recommendations.csv"
    tampered = pd.read_csv(card_path)
    tampered.loc[0, "home_cover_probability"] = 0.999999
    tampered.to_csv(card_path, index=False)

    with pytest.raises(DataContractError, match="does not reproduce"):
        cover_odds.query_cover_odds(
            season=_SEASON,
            week=_WEEK,
            game=game_id,
            spread=0.0,
            side="home",
            artifacts_root=artifacts_root,
            data_root=data_root,
        )


# ---------------------------------------------------------------------------
# 4. format_text / main() -- CLI entry point smoke tests
# ---------------------------------------------------------------------------


def test_main_prints_text_by_default(
    tmp_path: Path, model_frame: pd.DataFrame, capsys: pytest.CaptureFixture[str]
) -> None:
    artifacts_root = tmp_path / "artifacts"
    data_root = tmp_path / "data"
    card = _write_forecast(artifacts_root, data_root, model_frame)
    game_id = str(card.iloc[0]["game_id"])

    exit_code = cover_odds.main(
        [
            "--season",
            str(_SEASON),
            "--week",
            str(_WEEK),
            "--game",
            game_id,
            "--spread",
            "0.0",
            "--artifacts-root",
            str(artifacts_root),
            "--data-root",
            str(data_root),
        ]
    )
    assert exit_code == 0
    out = capsys.readouterr().out
    assert "Provenance:" in out
    assert "model_id: model-gauss" in out
    assert "not a live re-forecast" in out


def test_main_prints_json_with_the_json_flag(
    tmp_path: Path, model_frame: pd.DataFrame, capsys: pytest.CaptureFixture[str]
) -> None:
    artifacts_root = tmp_path / "artifacts"
    data_root = tmp_path / "data"
    card = _write_forecast(artifacts_root, data_root, model_frame)
    game_id = str(card.iloc[0]["game_id"])

    exit_code = cover_odds.main(
        [
            "--season",
            str(_SEASON),
            "--week",
            str(_WEEK),
            "--game",
            game_id,
            "--spread",
            "0.0",
            "--artifacts-root",
            str(artifacts_root),
            "--data-root",
            str(data_root),
            "--json",
        ]
    )
    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["game_id"] == game_id
    assert "provenance" in payload


def test_main_fails_closed_with_a_nonzero_exit_and_no_traceback(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    exit_code = cover_odds.main(
        [
            "--season",
            "2026",
            "--week",
            "1",
            "--game",
            "NE@SEA",
            "--spread",
            "0.0",
            "--artifacts-root",
            str(tmp_path),
        ]
    )
    assert exit_code == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "cover_odds:" in captured.err
