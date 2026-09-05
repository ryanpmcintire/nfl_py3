"""Tests for the MOD-17 served-total provider (``nfl_ats.served_total``).

Covers the task's required surface: both named methods on a synthetic frame,
the ``blend_k01`` hash pin, the ``joint_residual`` walk-forward cutoff
leakage guard (built, like ``tests/test_totals.py`` and
``tests/test_joint_residual_model.py``, so VIOLATING the guard changes the
answer), the dispatcher's fall-back contract, and that ``tiebreaker.json``
carries both totals plus the method that served.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from nfl_ats.joint_residual_model import make_joint_estimator, realised_residual_frame
from nfl_ats.served_total import (
    BLEND_K01_WEIGHT,
    JOINT_TOTAL_BLEND_WEIGHT,
    SERVED_TOTAL_METHOD,
    apply_blend,
    joint_residual_total_view,
    served_total,
    served_total_blend_k01,
    served_total_joint_residual,
)
from nfl_ats.tiebreaker import TOTALS_RESIDUAL_WEIGHT, MarketConsensus, build_report, lined_finals
from nfl_ats.totals import TotalsView, design_matrix

_FEATURES = ("wind", "temp")


# ---------------------------------------------------------------------------
# 0. The module default is what production is meant to serve
# ---------------------------------------------------------------------------


def test_served_total_method_defaults_to_joint_residual() -> None:
    assert SERVED_TOTAL_METHOD == "joint_residual"


def test_blend_k01_weight_matches_tiebreakers_totals_residual_weight() -> None:
    """The two constants are DUPLICATED (not imported, to avoid a circular
    import -- see ``served_total.py``'s module docstring) and must never
    silently diverge."""

    assert pytest.approx(TOTALS_RESIDUAL_WEIGHT) == BLEND_K01_WEIGHT


# ---------------------------------------------------------------------------
# 1. served_total_blend_k01: today's rule, unchanged
# ---------------------------------------------------------------------------


def test_served_total_blend_k01_matches_todays_formula() -> None:
    view = TotalsView(
        predicted_total=43.42, market_total=43.0, residual=0.42, train_games=4_630, source="x"
    )
    assert served_total_blend_k01(43.0, view) == pytest.approx(43.0 + 0.1 * 0.42)
    assert served_total_blend_k01(43.0, view, weight=0.2) == pytest.approx(43.0 + 0.2 * 0.42)
    # No view: the market total alone, exactly like build_report's own
    # "market-only" degrade.
    assert served_total_blend_k01(43.0, None) == pytest.approx(43.0)


def test_apply_blend_is_the_one_formula_both_named_methods_share() -> None:
    view = TotalsView(
        predicted_total=44.0, market_total=44.0, residual=-1.5, train_games=500, source="x"
    )
    assert apply_blend(44.0, view, weight=0.1) == pytest.approx(43.85)
    assert apply_blend(44.0, None, weight=0.1) == pytest.approx(44.0)


def test_served_total_blend_k01_is_hash_pinned() -> None:
    """Guards against silent arithmetic drift in the arm that is now the
    COMPARISON side rather than the default -- a formula this simple should
    never need to change, and if it ever does this test must fail loudly."""

    fixture: list[tuple[float, float | None]] = [
        (43.0, None),
        (43.0, 0.42),
        (44.5, -1.2),
        (37.25, 0.0),
        (51.0, 3.333333),
    ]
    values: list[float] = []
    for market_total, residual in fixture:
        view = (
            None
            if residual is None
            else TotalsView(
                predicted_total=market_total + residual,
                market_total=market_total,
                residual=residual,
                train_games=500,
                source="x",
            )
        )
        values.append(served_total_blend_k01(market_total, view))

    payload = ",".join(f"{value:.10f}" for value in values)
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    assert digest == "7c2a59c26bc5ad64a500f3ddc56c4f09e0f0bdd9616ef6d445ff685f13dde4ed"


# ---------------------------------------------------------------------------
# 2. served_total_joint_residual: the pure blend half
# ---------------------------------------------------------------------------


def test_served_total_joint_residual_blends_at_its_own_weight() -> None:
    view = TotalsView(
        predicted_total=44.6, market_total=44.0, residual=0.6, train_games=3_919, source="x"
    )
    assert served_total_joint_residual(44.0, view) == pytest.approx(
        44.0 + JOINT_TOTAL_BLEND_WEIGHT * 0.6
    )
    assert served_total_joint_residual(44.0, None) is None


# ---------------------------------------------------------------------------
# 3. joint_residual_total_view: I/O + fit, walk-forward cutoff leakage guard
# ---------------------------------------------------------------------------


def _synthetic_features(
    *, weeks: int = 8, games_per_week: int = 40, flip_week: int = 5, season: int = 2000
) -> pd.DataFrame:
    """Duplicated (not imported) from ``tests/test_joint_residual_model.py``'s
    own fixture of the same name -- the same cross-file duplication
    convention every copy of a synthetic fixture already follows in this
    repository. A game table whose margin/total residuals REVERSE at
    ``flip_week``, so a walk-forward fit that honours the cutoff when
    predicting the flip week has seen only the pre-flip regime, and one that
    leaked even a single later row gives a visibly different prediction."""

    generator = np.random.default_rng(20260905)
    rows = []
    gameday = pd.Timestamp("2000-09-01")
    for week in range(1, weeks + 1):
        slope = 6.0 if week < flip_week else -6.0
        for game in range(games_per_week):
            wind = float(generator.uniform(-1.0, 1.0))
            temp = float(generator.uniform(-1.0, 1.0))
            spread_line = 0.0
            total_line = 44.0
            margin_residual = slope * wind
            total_residual = slope * temp
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


def _write_joint_features_fixture(
    tmp_path: Path, *, target_id: str, weeks: int = 8, games_per_week: int = 40, flip_week: int = 5
) -> pd.DataFrame:
    """The full synthetic table, with ``target_id`` stripped of its outcome
    (unplayed -- the game ``joint_residual_total_view`` is asked to price),
    written to the default weak-stack path under ``tmp_path``."""

    features = _synthetic_features(weeks=weeks, games_per_week=games_per_week, flip_week=flip_week)
    served = features.copy()
    mask = served["game_id"] == target_id
    assert mask.any(), f"{target_id} not present in the synthetic fixture"
    served.loc[mask, ["home_score", "away_score", "result", "ats_margin"]] = np.nan

    processed = tmp_path / "processed"
    processed.mkdir(parents=True, exist_ok=True)
    served.to_parquet(processed / "game_features_weak_stack.parquet")
    return features  # the FULLY PLAYED table, for building the honest/leaky comparators


def test_joint_residual_total_view_trains_only_on_strictly_earlier_weeks(tmp_path: Path) -> None:
    target_week = 5
    target_id = f"2000_{target_week:02d}_00"
    full_features = _write_joint_features_fixture(
        tmp_path, target_id=target_id, flip_week=target_week
    )

    view = joint_residual_total_view(
        target_id, tmp_path, feature_columns=_FEATURES, min_train_games=40
    )
    assert view is not None
    assert view.train_games == 40 * (target_week - 1)  # weeks 1..4, none from week 5 onward
    assert view.market_total == pytest.approx(44.0)
    assert view.predicted_total == pytest.approx(view.market_total + view.residual)

    # The decisive comparison: the walked residual must equal a model fit on
    # STRICTLY earlier weeks only, and must DIFFER from one that also saw the
    # target week (or later weeks) -- the same "honest vs leaky" pattern
    # tests/test_totals.py and tests/test_joint_residual_model.py both use.
    population = realised_residual_frame(full_features, feature_columns=_FEATURES)
    honest_train = population.loc[population["week"] < target_week]
    leaky_train = population.loc[population["week"] <= target_week]
    target_row = full_features.loc[full_features["game_id"] == target_id]
    target_design = design_matrix(target_row, _FEATURES)

    honest = make_joint_estimator()
    honest.fit(
        honest_train.loc[:, list(_FEATURES)],
        honest_train.loc[:, ["margin_residual", "total_residual"]].to_numpy(),
    )
    honest_prediction = np.asarray(honest.predict(target_design), dtype=float).reshape(-1)

    leaky = make_joint_estimator()
    leaky.fit(
        leaky_train.loc[:, list(_FEATURES)],
        leaky_train.loc[:, ["margin_residual", "total_residual"]].to_numpy(),
    )
    leaky_prediction = np.asarray(leaky.predict(target_design), dtype=float).reshape(-1)

    assert view.residual == pytest.approx(honest_prediction[1], abs=1e-9)
    assert abs(leaky_prediction[1] - honest_prediction[1]) > 0.05
    assert view.residual != pytest.approx(leaky_prediction[1])


def test_joint_residual_total_view_declines_below_the_training_floor(tmp_path: Path) -> None:
    target_week = 5
    target_id = f"2000_{target_week:02d}_00"
    _write_joint_features_fixture(tmp_path, target_id=target_id, flip_week=target_week)

    assert (
        joint_residual_total_view(
            target_id, tmp_path, feature_columns=_FEATURES, min_train_games=10_000
        )
        is None
    )


def test_joint_residual_total_view_returns_none_on_missing_inputs(tmp_path: Path) -> None:
    target_week = 5
    target_id = f"2000_{target_week:02d}_00"
    _write_joint_features_fixture(tmp_path, target_id=target_id, flip_week=target_week)

    # No table at all under the default path.
    empty_root = tmp_path / "empty"
    assert (
        joint_residual_total_view(
            target_id, empty_root, feature_columns=_FEATURES, min_train_games=40
        )
        is None
    )
    # A game_id the table does not carry.
    assert (
        joint_residual_total_view(
            "no_such_game", tmp_path, feature_columns=_FEATURES, min_train_games=40
        )
        is None
    )


# ---------------------------------------------------------------------------
# 4. served_total(): the dispatcher's fall-back contract
# ---------------------------------------------------------------------------


def test_served_total_dispatch_prefers_joint_residual_when_a_view_exists() -> None:
    blend_view = TotalsView(
        predicted_total=43.4, market_total=43.0, residual=0.4, train_games=500, source="blend"
    )
    joint_view = TotalsView(
        predicted_total=43.6, market_total=43.0, residual=0.6, train_games=3_919, source="joint"
    )

    value, method = served_total(
        "joint_residual", market_total=43.0, blend_view=blend_view, joint_view=joint_view
    )
    assert method == "joint_residual"
    assert value == pytest.approx(43.0 + JOINT_TOTAL_BLEND_WEIGHT * 0.6)


def test_served_total_dispatch_falls_back_to_blend_when_no_joint_view_exists() -> None:
    blend_view = TotalsView(
        predicted_total=43.4, market_total=43.0, residual=0.4, train_games=500, source="blend"
    )

    value, method = served_total(
        "joint_residual", market_total=43.0, blend_view=blend_view, joint_view=None
    )
    assert method == "blend_k01"
    assert value == pytest.approx(43.0 + BLEND_K01_WEIGHT * 0.4)


def test_served_total_dispatch_blend_k01_ignores_any_joint_view() -> None:
    blend_view = TotalsView(
        predicted_total=43.4, market_total=43.0, residual=0.4, train_games=500, source="blend"
    )
    joint_view = TotalsView(
        predicted_total=43.9, market_total=43.0, residual=0.9, train_games=3_919, source="joint"
    )

    value, method = served_total(
        "blend_k01", market_total=43.0, blend_view=blend_view, joint_view=joint_view
    )
    assert method == "blend_k01"
    assert value == pytest.approx(43.0 + BLEND_K01_WEIGHT * 0.4)


def test_served_total_dispatch_rejects_an_unknown_method() -> None:
    with pytest.raises(ValueError, match="Unknown served-total method"):
        served_total(
            "not_a_method",  # type: ignore[arg-type]
            market_total=43.0,
            blend_view=None,
            joint_view=None,
        )


# ---------------------------------------------------------------------------
# 5. build_report wiring: served_total_method / comparison_total_blend_k01
# ---------------------------------------------------------------------------


def _den_kc_fixture() -> tuple[pd.Series, MarketConsensus, pd.DataFrame]:
    schedules = pd.DataFrame(
        {
            "game_id": ["2024_01_A_B", "2024_01_C_D", "2026_01_DEN_KC"],
            "season": [2024, 2024, 2026],
            "week": [1, 1, 1],
            "game_type": ["REG"] * 3,
            "gameday": ["2024-09-08", "2024-09-08", "2026-09-14"],
            "gametime": ["13:00", "16:25", "20:15"],
            "home_team": ["B", "D", "KC"],
            "away_team": ["A", "C", "DEN"],
            "home_score": [24.0, 20.0, None],
            "away_score": [20.0, 23.0, None],
            "spread_line": [3.0, 2.5, 2.5],
            "total_line": [43.5, 44.0, 43.0],
        }
    )
    game = schedules.iloc[2]
    consensus = MarketConsensus(
        game_id="2026_01_DEN_KC", home_expected_margin=2.5, total_line=43.0, source="test"
    )
    return game, consensus, lined_finals(schedules)


def test_build_report_serves_joint_residual_when_both_views_exist() -> None:
    game, consensus, finals = _den_kc_fixture()
    blend_view = TotalsView(
        predicted_total=43.4, market_total=43.0, residual=0.4, train_games=500, source="blend"
    )
    joint_view = TotalsView(
        predicted_total=43.6, market_total=43.0, residual=0.6, train_games=3_919, source="joint"
    )

    report = build_report(game, consensus, finals, None, blend_view, joint_view)

    assert report.served_total_method == "joint_residual"
    assert report.served_total == pytest.approx(43.0 + JOINT_TOTAL_BLEND_WEIGHT * 0.6)
    assert report.guess_total_line == pytest.approx(report.served_total)
    assert report.comparison_total_blend_k01 == pytest.approx(43.0 + TOTALS_RESIDUAL_WEIGHT * 0.4)
    # The two arms genuinely differ here -- otherwise this test could pass by
    # accident even with a broken dispatcher.
    assert report.served_total != pytest.approx(report.comparison_total_blend_k01)


def test_build_report_falls_back_to_blend_when_no_joint_view_is_supplied() -> None:
    game, consensus, finals = _den_kc_fixture()
    blend_view = TotalsView(
        predicted_total=43.4, market_total=43.0, residual=0.4, train_games=500, source="blend"
    )

    report = build_report(game, consensus, finals, None, blend_view, None)

    assert report.served_total_method == "blend_k01"
    assert report.served_total == pytest.approx(43.0 + TOTALS_RESIDUAL_WEIGHT * 0.4)
    assert report.served_total == pytest.approx(report.comparison_total_blend_k01)


# ---------------------------------------------------------------------------
# 6. tiebreaker.json carries both totals and the serving method
# ---------------------------------------------------------------------------


def test_tiebreaker_json_payload_carries_served_total_method_and_both_totals() -> None:
    from nfl_ats.publishing import _tiebreaker_json_payload

    game, consensus, finals = _den_kc_fixture()
    blend_view = TotalsView(
        predicted_total=43.4, market_total=43.0, residual=0.4, train_games=500, source="blend"
    )
    joint_view = TotalsView(
        predicted_total=43.6, market_total=43.0, residual=0.6, train_games=3_919, source="joint"
    )
    report = build_report(game, consensus, finals, None, blend_view, joint_view)

    payload = _tiebreaker_json_payload(
        report,
        generated_at=pd.Timestamp("2026-09-05T12:00:00Z").to_pydatetime(),
        model_id="test-model",
    )

    assert payload["served_total"] == pytest.approx(report.served_total)
    assert payload["served_total_method"] == "joint_residual"
    assert payload["comparison_total_blend_k01"] == pytest.approx(report.comparison_total_blend_k01)
    # blended_total is retained, unchanged, for readers that predate this
    # switch -- it always equals the served total.
    assert payload["blended_total"] == pytest.approx(payload["served_total"])
