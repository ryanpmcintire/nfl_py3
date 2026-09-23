from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import nfl_ats.best_pick_nomination as bpn
from nfl_ats.best_pick_nomination import (
    CHALLENGER_ID,
    CHALLENGER_ID_V3,
    DispersionPool,
    NominationV2Result,
    NominationV3Result,
    fit_candidate_probabilities,
    nominate_v2,
    record_nomination_challenger_decisions,
    record_nomination_v3_challenger_decisions,
    select_nominee,
)
from nfl_ats.constants import GRAPH_FEATURE_COLUMNS, MODEL_FEATURE_COLUMNS
from nfl_ats.data import DataContractError
from nfl_ats.prospective_scoring import load_challenger_decisions

TUESDAY = pd.Timestamp("2026-08-18T13:00:00Z")
KICKOFF = TUESDAY + pd.Timedelta(days=4)

_MODEL_CONFIG = {
    "method": "market_residual",
    "target": "market_residual",
    "regressor": "ridge",
    "ridge_alpha": 10.0,
    "calibration_method": "none",
    "feature_profile": "base",
    "min_edge": 0.02,
    "min_train_games": 100,
    "feature_table": "features.parquet",
}


_QUOTE_COLUMNS = [
    "nflverse_game_id",
    "bookmaker_key",
    "market",
    "outcome_side",
    "home_spread_line",
    "observed_at_utc",
    "commence_time_utc",
]


def _quotes(per_game: dict[str, list[float]]) -> pd.DataFrame:

    rows = []
    for game_id, lines in per_game.items():
        for book_index, line in enumerate(lines):
            rows.append(
                {
                    "nflverse_game_id": game_id,
                    "bookmaker_key": f"book{book_index}",
                    "market": "spreads",
                    "outcome_side": "HOME",
                    "home_spread_line": line,
                    "observed_at_utc": TUESDAY,
                    "commence_time_utc": KICKOFF,
                }
            )
    return (
        pd.DataFrame(rows, columns=_QUOTE_COLUMNS) if rows else pd.DataFrame(columns=_QUOTE_COLUMNS)
    )


def _candidates(**rows: tuple[float, float | None]) -> pd.DataFrame:

    return pd.DataFrame(
        [
            {"game_id": game_id, "candidate_dist": dist, "spread_std": std}
            for game_id, (dist, std) in rows.items()
        ]
    )


def test_select_nominee_takes_the_unambiguous_max() -> None:
    candidates = _candidates(low=(0.02, 1.0), high=(0.20, 1.0), mid=(0.10, 1.0))
    game_id, n_tied, tie_break = select_nominee(candidates)
    assert (game_id, n_tied, tie_break) == ("high", 1, "none")


def test_select_nominee_breaks_a_tie_with_lower_dispersion() -> None:
    candidates = _candidates(noisy=(0.20, 3.0), quiet=(0.20, 1.0), also_noisy=(0.20, 5.0))
    game_id, n_tied, tie_break = select_nominee(candidates)
    assert (game_id, n_tied, tie_break) == ("quiet", 3, "dispersion")


def test_select_nominee_requires_at_least_one_candidate() -> None:
    with pytest.raises(ValueError, match="at least one candidate"):
        select_nominee(pd.DataFrame(columns=["game_id", "candidate_dist", "spread_std"]))


def _result(
    *, n_tied: int = 1, tie_break: str = "none", fallback: bool = False, reason: str | None = None
) -> NominationV2Result:
    dispersion = DispersionPool(
        frame=pd.DataFrame({"game_id": ["g"], "spread_std": [1.0], "pool_pass": [True]}),
        fallback=fallback,
        fallback_reason=reason,
        n_games=1,
        n_missing=0,
        n_pool_pass=1,
    )
    return NominationV2Result(
        game_id="g",
        n_tied_at_max=n_tied,
        tie_break=tie_break,
        probability_table=pd.DataFrame(),
        dispersion=dispersion,
    )


def _v3_result(*, n_tied: int = 1, tie_break: str = "none") -> NominationV3Result:
    dispersion = DispersionPool(
        frame=pd.DataFrame({"game_id": ["g"], "spread_std": [1.0], "pool_pass": [True]}),
        fallback=False,
        fallback_reason=None,
        n_games=1,
        n_missing=0,
        n_pool_pass=1,
    )
    return NominationV3Result(
        game_id="g",
        n_tied_at_max=n_tied,
        tie_break=tie_break,
        probability_table=pd.DataFrame(),
        dispersion=dispersion,
    )


def _fake_probabilities(dist_by_game: dict[str, float]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "game_id": list(dist_by_game),
            "candidate_home_cover_probability": [0.5 + d for d in dist_by_game.values()],
            "candidate_dist": list(dist_by_game.values()),
        }
    )


def _predictions(game_ids: list[str], *, game_type: str = "REG") -> pd.DataFrame:
    return pd.DataFrame({"game_id": game_ids, "game_type": [game_type] * len(game_ids)})


def test_nominate_v2_raises_when_a_carded_game_has_no_candidate_probability(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        bpn, "fit_candidate_probabilities", lambda *a, **k: _fake_probabilities({"g1": 0.1})
    )
    with pytest.raises(DataContractError, match="missing games"):
        nominate_v2(
            _predictions(["g1", "g2"]),
            pd.DataFrame(),
            market_root=Path("unused"),
            season=2026,
            week=1,
            regressor="ridge",
            feature_profile="base",
        )


def _walk_forward_features(train_rows: int = 150, target_rows: int = 6) -> pd.DataFrame:
    total = train_rows + target_rows
    start = date(2019, 9, 1)
    index = np.arange(total)
    frame = pd.DataFrame(
        {
            "game_id": [f"train_{v:03d}" for v in range(train_rows)]
            + [f"target_{v}" for v in range(target_rows)],
            "season": np.where(index < train_rows, 2019, 2020),
            "week": np.where(index < train_rows, (index // 15) + 1, 1),
            "gameday": [start + timedelta(days=int(v)) for v in range(train_rows)]
            + [date(2020, 9, 10)] * target_rows,
            "away_team": "AWY",
            "home_team": "HME",
        }
    )
    all_features = (*MODEL_FEATURE_COLUMNS, *GRAPH_FEATURE_COLUMNS)
    for feature_index, column in enumerate(all_features, start=1):
        frame[column] = np.sin(index / feature_index) + (index % 5) / 10.0
    frame["spread_line"] = np.where(index % 2 == 0, 2.5, -2.5)
    rng = np.random.default_rng(20260818)
    frame["ats_margin"] = rng.normal(loc=0.0, scale=8.0, size=total)
    frame["home_cover"] = (frame["ats_margin"] > 0).astype(float)
    frame["result"] = frame["spread_line"] + frame["ats_margin"]
    frame.loc[index >= train_rows, ["home_cover", "ats_margin", "result"]] = np.nan
    return frame


def test_fit_candidate_probabilities_never_leaks_the_target_weeks_own_outcome() -> None:

    baseline = _walk_forward_features()
    leaked = baseline.copy()
    is_target = leaked["game_id"].str.startswith("target_")
    leaked.loc[is_target, "result"] = 999.0
    leaked.loc[is_target, "ats_margin"] = 999.0
    leaked.loc[is_target, "home_cover"] = 1.0

    before = fit_candidate_probabilities(
        baseline,
        season=2020,
        week=1,
        regressor="ridge",
        feature_profile="base",
        min_train_games=100,
    )
    after = fit_candidate_probabilities(
        leaked, season=2020, week=1, regressor="ridge", feature_profile="base", min_train_games=100
    )
    pd.testing.assert_series_equal(
        before.sort_values("game_id").reset_index(drop=True)["candidate_home_cover_probability"],
        after.sort_values("game_id").reset_index(drop=True)["candidate_home_cover_probability"],
    )


def _write_registry(
    artifacts: Path, *, status: str = "ACTIVE_PROSPECTIVE", challenger_id: str = CHALLENGER_ID
) -> None:
    payload = {
        "ledger": "prospective_challengers",
        "schema_version": 1,
        "challengers": [
            {"challenger_id": challenger_id, "status": status, "model": dict(_MODEL_CONFIG)}
        ],
    }
    path = artifacts / "prospective" / "challengers.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_active_model_and_card(
    artifacts: Path, tmp_path: Path, *, ridge_alpha: float = 10.0, n_games: int = 2
) -> Path:
    forecast = artifacts / "margin_predictions" / "2026-week-01-forecast"
    forecast.mkdir(parents=True, exist_ok=True)
    metadata = {
        "active_model_id": "model-xyz",
        "synchronization_status": "SYNCHRONIZED",
        "season": 2026,
        "week": 1,
        "created_at_utc": "2026-09-08T15:00:00+00:00",
        "ats_method": "market_residual",
        "regressor": "ridge",
        "ridge_alpha": ridge_alpha,
        "calibration_method": "none",
        "feature_profile": "base",
        "min_edge": 0.02,
        "min_train_games": 100,
        "provenance": {"feature_table": {"path": "features.parquet", "sha256": "abc123"}},
    }
    (forecast / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")

    kickoffs = [KICKOFF + pd.Timedelta(hours=i) for i in range(n_games)]
    card = pd.DataFrame(
        {
            "game_id": [f"2026_01_G{i}" for i in range(n_games)],
            "season": 2026,
            "week": 1,
            "kickoff": [ts.isoformat() for ts in kickoffs],
            "away_team": [f"AWY{i}" for i in range(n_games)],
            "home_team": [f"HME{i}" for i in range(n_games)],
            "spread_line": [1.5 + i for i in range(n_games)],
            "home_cover_probability": [0.55 for _ in range(n_games)],
        }
    )
    card.to_csv(forecast / "recommendations.csv", index=False)

    active = {
        "version": 1,
        "status": "SYNCHRONIZED",
        "model_id": "model-xyz",
        "method": "market_residual",
        "feature_profile": "base",
        "historical_evaluation": {"accuracy": 0.52, "correct": 1, "games": 1, "intervals": {}},
        "weekly_forecast": {
            "artifact": "margin_predictions/2026-week-01-forecast",
            "season": 2026,
            "week": 1,
        },
    }
    (artifacts / "active_ats_model.json").write_text(json.dumps(active), encoding="utf-8")

    features_path = tmp_path / "features.parquet"
    metadata["provenance"]["feature_table"]["path"] = str(features_path)
    (forecast / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    pd.DataFrame({"game_id": ["placeholder"]}).to_parquet(features_path)
    return forecast


def test_record_nomination_challenger_decisions_requires_the_whole_week_pre_kickoff(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:

    artifacts = tmp_path / "artifacts"
    _write_registry(artifacts)
    _write_active_model_and_card(artifacts, tmp_path, n_games=2)
    fake = NominationV2Result(
        game_id="2026_01_G1",
        n_tied_at_max=1,
        tie_break="none",
        probability_table=pd.DataFrame(),
        dispersion=DispersionPool(pd.DataFrame(), False, None, 2, 0, 2),
    )
    monkeypatch.setattr(bpn, "nominate_v2", lambda *a, **k: fake)
    now = KICKOFF + pd.Timedelta(minutes=30)

    result = record_nomination_challenger_decisions(artifacts, tmp_path, now=now)

    assert result["recorded"] == 0
    assert result["post_kickoff_skipped"] == 1
    assert load_challenger_decisions(artifacts).empty


def test_record_nomination_challenger_refuses_a_fingerprint_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    artifacts = tmp_path / "artifacts"
    _write_registry(artifacts)
    _write_active_model_and_card(artifacts, tmp_path, ridge_alpha=1.0)
    monkeypatch.setattr(bpn, "nominate_v2", lambda *a, **k: None)

    with pytest.raises(DataContractError, match="configuration fingerprint"):
        record_nomination_challenger_decisions(
            artifacts, tmp_path, now=KICKOFF - pd.Timedelta(days=3)
        )
    assert load_challenger_decisions(artifacts).empty


def test_record_nomination_v3_and_v2_ledger_rows_coexist(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:

    artifacts = tmp_path / "artifacts"
    payload = {
        "ledger": "prospective_challengers",
        "schema_version": 1,
        "challengers": [
            {
                "challenger_id": CHALLENGER_ID,
                "status": "ACTIVE_PROSPECTIVE",
                "model": dict(_MODEL_CONFIG),
            },
            {
                "challenger_id": CHALLENGER_ID_V3,
                "status": "ACTIVE_PROSPECTIVE",
                "model": dict(_MODEL_CONFIG),
            },
        ],
    }
    path = artifacts / "prospective" / "challengers.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    _write_active_model_and_card(artifacts, tmp_path, n_games=2)

    v2_fake = NominationV2Result(
        game_id="2026_01_G0",
        n_tied_at_max=1,
        tie_break="none",
        probability_table=pd.DataFrame(),
        dispersion=DispersionPool(pd.DataFrame(), False, None, 2, 0, 2),
    )
    v3_fake = NominationV3Result(
        game_id="2026_01_G1",
        n_tied_at_max=1,
        tie_break="none",
        probability_table=pd.DataFrame(),
        dispersion=DispersionPool(pd.DataFrame(), False, None, 2, 0, 2),
    )
    monkeypatch.setattr(bpn, "nominate_v2", lambda *a, **k: v2_fake)
    monkeypatch.setattr(bpn, "nominate_v3", lambda *a, **k: v3_fake)
    now = KICKOFF - pd.Timedelta(days=3)

    record_nomination_challenger_decisions(artifacts, tmp_path, now=now)
    record_nomination_v3_challenger_decisions(artifacts, tmp_path, now=now)

    ledger = load_challenger_decisions(artifacts)
    assert len(ledger) == 2
    by_challenger = dict(zip(ledger["challenger_id"], ledger["game_id"], strict=True))
    assert by_challenger == {CHALLENGER_ID: "2026_01_G0", CHALLENGER_ID_V3: "2026_01_G1"}
