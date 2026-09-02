"""Tests for the over/under regime (``docs/totals_model.md``, run 2026-09-01).

The four the frozen contract names, in its order: the walk-forward guard, the
feature allowlist, the blend math, and the tiebreaker wiring. The guard test
is deliberately built so that VIOLATING it changes the answer -- a guard test
that passes whether or not the guard exists is not a test.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from nfl_ats.totals import (
    BLEND_WEIGHTS,
    TOTALS_FEATURES,
    TOTALS_RIDGE_ALPHA,
    TotalsDataError,
    TotalsView,
    blend_sweep,
    blend_total,
    choose_weight,
    chronological_blocks,
    design_matrix,
    load_population,
    make_totals_estimator,
    model_total_view,
    paired_error_frame,
    per_season_deltas,
    walk_forward_predictions,
)

_TARGET = "total_residual"


def _synthetic_population(
    *,
    weeks: int = 6,
    games_per_week: int = 40,
    flip_week: int = 4,
    season: int = 2000,
) -> pd.DataFrame:
    """A population whose signal REVERSES at ``flip_week``.

    ``wind`` drives the residual with slope +5 before the flip week and -5
    from it onward. A model that honours the walk-forward guard when
    predicting the flip week has seen only the +5 regime; a model that leaked
    even one row from that week onward has seen both. The two therefore give
    visibly different predictions, which is what makes the guard testable.
    """

    rows = []
    generator = np.random.default_rng(20260901)
    for week in range(1, weeks + 1):
        slope = 5.0 if week < flip_week else -5.0
        for game in range(games_per_week):
            wind = float(generator.uniform(-1.0, 1.0))
            market_total = 44.0
            residual = slope * wind
            rows.append(
                {
                    "game_id": f"{season}_{week:02d}_{game:02d}",
                    "season": season,
                    "week": week,
                    "game_type": "REG",
                    "market_total": market_total,
                    "actual_total": market_total + residual,
                    _TARGET: residual,
                    "wind": wind,
                }
            )
    frame = pd.DataFrame(rows)
    for column in TOTALS_FEATURES:
        if column not in frame.columns:
            frame[column] = 0.0
    return frame


# ---------------------------------------------------------------------------
# 1. The walk-forward guard
# ---------------------------------------------------------------------------


def test_walk_forward_trains_only_on_strictly_earlier_weeks() -> None:
    population = _synthetic_population()
    predictions = walk_forward_predictions(population, min_train_games=40)

    # Weeks 2..6 are scored (week 1 has no prior games); every block reports a
    # training pool equal to the count of STRICTLY earlier games.
    scored_weeks = sorted(predictions["week"].unique())
    assert scored_weeks == [2, 3, 4, 5, 6]
    for week in scored_weeks:
        block = predictions.loc[predictions["week"] == week]
        assert int(block["train_games"].iloc[0]) == 40 * (week - 1)

    # The decisive comparison: at the flip week the guarded prediction must
    # equal a model fit on weeks 1-3 only, and must DIFFER from one that also
    # saw week 4. If the guard were "<=" instead of "<", the two would agree.
    target_week = 4
    honest_train = population.loc[population["week"] < target_week]
    leaky_train = population.loc[population["week"] <= target_week]
    target_rows = population.loc[population["week"] == target_week]

    honest = make_totals_estimator(ridge_alpha=TOTALS_RIDGE_ALPHA)
    honest.fit(design_matrix(honest_train), honest_train[_TARGET].to_numpy())
    leaky = make_totals_estimator(ridge_alpha=TOTALS_RIDGE_ALPHA)
    leaky.fit(design_matrix(leaky_train), leaky_train[_TARGET].to_numpy())

    honest_prediction = np.asarray(honest.predict(design_matrix(target_rows)), dtype=float)
    leaky_prediction = np.asarray(leaky.predict(design_matrix(target_rows)), dtype=float)
    walked = predictions.loc[predictions["week"] == target_week, "predicted_residual"].to_numpy()

    assert walked == pytest.approx(honest_prediction)
    # The leak is not a rounding difference: it moves the answer by points.
    assert np.abs(leaky_prediction - honest_prediction).max() > 1.0
    assert not np.allclose(walked, leaky_prediction)


def test_walk_forward_respects_the_warm_up_floor() -> None:
    population = _synthetic_population()
    predictions = walk_forward_predictions(population, min_train_games=100)
    # 100 games are not banked until week 4 (weeks 1-3 supply 120).
    assert sorted(predictions["week"].unique()) == [4, 5, 6]
    assert int(predictions["train_games"].min()) >= 100

    with pytest.raises(TotalsDataError, match="min_train_games"):
        walk_forward_predictions(population, min_train_games=10_000)


def test_chronological_blocks_are_sorted_and_unique() -> None:
    population = _synthetic_population(weeks=3, games_per_week=5)
    shuffled = population.sample(frac=1.0, random_state=7)
    assert chronological_blocks(shuffled) == [(2000, 1), (2000, 2), (2000, 3)]


# ---------------------------------------------------------------------------
# 2. Allowlist enforcement
# ---------------------------------------------------------------------------


def test_design_matrix_is_exactly_the_allowlist_in_order() -> None:
    population = _synthetic_population(weeks=2, games_per_week=5)
    matrix = design_matrix(population)
    assert list(matrix.columns) == list(TOTALS_FEATURES)
    # Outcome and identifier columns are present in the source frame and
    # still never reach the matrix.
    for banned in ("actual_total", _TARGET, "game_id", "season", "week", "game_type"):
        assert banned in population.columns
        assert banned not in matrix.columns


def test_an_extra_column_never_enters_the_design_matrix_or_the_fit() -> None:
    population = _synthetic_population(weeks=4, games_per_week=30)
    # A column that would be overwhelmingly predictive if it ever leaked in.
    contaminated = population.assign(
        leaked_actual_total=population["actual_total"] * 1_000.0,
        home_off_epa_per_play_v2=population["wind"] * 99.0,
    )
    assert "leaked_actual_total" not in design_matrix(contaminated).columns
    assert "home_off_epa_per_play_v2" not in design_matrix(contaminated).columns

    clean = walk_forward_predictions(population, min_train_games=30)
    dirty = walk_forward_predictions(contaminated, min_train_games=30)
    assert dirty["predicted_residual"].to_numpy() == pytest.approx(
        clean["predicted_residual"].to_numpy()
    )


def test_a_renamed_allowlist_column_is_a_hard_error_not_a_substitution() -> None:
    population = _synthetic_population(weeks=2, games_per_week=5)
    renamed = population.rename(columns={"home_off_cpoe": "home_offense_cpoe"})
    with pytest.raises(TotalsDataError, match="home_off_cpoe"):
        design_matrix(renamed)
    with pytest.raises(TotalsDataError, match="home_off_cpoe"):
        walk_forward_predictions(renamed, min_train_games=5)


# ---------------------------------------------------------------------------
# 3. Blend math
# ---------------------------------------------------------------------------


def test_blend_total_endpoints_are_the_market_and_the_raw_model() -> None:
    market = pd.Series([44.0, 41.5, 50.0])
    residual = pd.Series([1.0, -2.0, 0.5])
    assert blend_total(market, residual, 0.0).tolist() == market.tolist()
    assert blend_total(market, residual, 1.0).tolist() == (market + residual).tolist()
    assert blend_total(market, residual, 0.3).tolist() == pytest.approx([44.3, 41.5 - 0.6, 50.15])


def test_blend_sweep_covers_the_declared_grid_and_agrees_at_the_endpoints() -> None:
    predictions = pd.DataFrame(
        {
            "market_total": [44.0, 44.0, 44.0, 44.0],
            "predicted_residual": [2.0, -2.0, 4.0, -4.0],
            "actual_total": [46.0, 42.0, 45.0, 43.0],
            "market_error": [-2.0, 2.0, -1.0, 1.0],
        }
    )
    sweep = blend_sweep(predictions)
    assert sweep["k"].tolist() == pytest.approx(list(BLEND_WEIGHTS))
    assert sweep.loc[sweep["k"] == 0.0, "mae"].iloc[0] == pytest.approx(1.5)
    # k=1 is the raw model: predicted totals 46, 42, 48, 40 vs actuals -> errors
    # 0, 0, 3, -3, so MAE 1.5 as well, and the improvement column is signed
    # market-minus-blend.
    assert sweep.loc[sweep["k"] == 1.0, "mae"].iloc[0] == pytest.approx(1.5)
    assert sweep["mae_improvement_vs_market"].iloc[0] == pytest.approx(0.0)


def test_choose_weight_takes_the_mae_minimum_and_breaks_ties_low() -> None:
    sweep = pd.DataFrame({"k": [0.0, 0.1, 0.2], "mae": [10.0, 9.5, 9.5]})
    assert choose_weight(sweep) == pytest.approx(0.1)
    flat = pd.DataFrame({"k": [0.0, 0.1, 0.2], "mae": [9.5, 9.5, 9.5]})
    assert choose_weight(flat) == pytest.approx(0.0)


def test_paired_improvement_is_positive_when_the_blend_is_closer() -> None:
    predictions = pd.DataFrame(
        {
            "game_id": ["a", "b"],
            "season": [2020, 2020],
            "week": [1, 1],
            "market_total": [44.0, 44.0],
            "predicted_residual": [2.0, 2.0],
            "actual_total": [46.0, 42.0],
            "market_error": [-2.0, 2.0],
        }
    )
    paired = paired_error_frame(predictions, 1.0)
    # Game a: the blend nails it (improvement +2). Game b: the blend moves the
    # wrong way (improvement -2).
    assert paired["abs_error_improvement"].tolist() == pytest.approx([2.0, -2.0])

    seasons = per_season_deltas(predictions, 1.0)
    assert seasons["mae_improvement"].iloc[0] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Population assembly (the contract's population definition)
# ---------------------------------------------------------------------------


def _write_population_fixture(root: Path) -> Path:
    schedules = pd.DataFrame(
        {
            "game_id": ["2020_01_A_B", "2020_01_C_D", "2020_02_E_F", "2026_01_X_Y"],
            "season": [2020, 2020, 2020, 2026],
            "week": [1, 1, 2, 1],
            "game_type": ["REG", "REG", "REG", "REG"],
            "gameday": ["2020-09-10", "2020-09-13", "2020-09-20", "2026-09-13"],
            "home_team": ["B", "D", "F", "Y"],
            "away_team": ["A", "C", "E", "X"],
            "home_score": [24.0, 20.0, 30.0, None],  # the 2026 game is unplayed
            "away_score": [20.0, 23.0, 13.0, None],
            "total_line": [43.5, 44.0, 41.0, 44.5],
        }
    )
    raw = root / "raw" / "20260901T000000Z"
    raw.mkdir(parents=True)
    schedules.to_parquet(raw / "schedules.parquet")

    features = pd.DataFrame({"game_id": schedules["game_id"]})
    for column in TOTALS_FEATURES:
        features[column] = 0.5
    features["season"] = schedules["season"]
    features["week"] = schedules["week"]
    features["total_line"] = schedules["total_line"]
    features["an_unlisted_column"] = 999.0
    features_path = root / "processed" / "game_features.parquet"
    features_path.parent.mkdir(parents=True)
    features.to_parquet(features_path)
    return features_path


def test_load_population_keeps_only_lined_finals_and_computes_the_target(tmp_path: Path) -> None:
    features_path = _write_population_fixture(tmp_path)
    population = load_population(tmp_path, features_path)

    assert len(population) == 3  # the unplayed 2026 game is excluded
    assert population["game_id"].tolist() == ["2020_01_A_B", "2020_01_C_D", "2020_02_E_F"]
    assert population["actual_total"].tolist() == pytest.approx([44.0, 43.0, 43.0])
    assert population[_TARGET].tolist() == pytest.approx([0.5, -1.0, 2.0])
    assert "an_unlisted_column" not in population.columns


def test_model_total_view_trains_only_on_games_before_the_target_week(tmp_path: Path) -> None:
    features_path = _write_population_fixture(tmp_path)
    # Three prior games is under any realistic floor, so the view declines.
    assert model_total_view("2026_01_X_Y", tmp_path, features_path, min_train_games=500) is None

    view = model_total_view("2026_01_X_Y", tmp_path, features_path, min_train_games=3)
    assert view is not None
    assert view.train_games == 3  # the three 2020 finals, none from 2026
    assert view.market_total == pytest.approx(44.5)
    assert view.predicted_total == pytest.approx(view.market_total + view.residual)

    # A game the feature table does not price gets no view rather than a guess.
    assert model_total_view("2026_01_NO_SUCH", tmp_path, features_path, min_train_games=3) is None
    assert model_total_view("2026_01_X_Y", tmp_path, tmp_path / "absent.parquet") is None


# ---------------------------------------------------------------------------
# 4. Tiebreaker wiring
# ---------------------------------------------------------------------------


def test_tiebreaker_blends_the_totals_residual_at_the_measured_weight() -> None:
    from nfl_ats.tiebreaker import (
        TOTALS_RESIDUAL_WEIGHT,
        MarketConsensus,
        build_report,
        lined_finals,
    )

    schedules = pd.DataFrame(
        {
            "game_id": ["2024_01_A_B", "2024_01_C_D", "2024_02_E_F", "2026_01_DEN_KC"],
            "season": [2024, 2024, 2024, 2026],
            "week": [1, 1, 2, 1],
            "game_type": ["REG"] * 4,
            "gameday": ["2024-09-08", "2024-09-08", "2024-09-15", "2026-09-14"],
            "gametime": ["13:00", "16:25", "13:00", "20:15"],
            "home_team": ["B", "D", "F", "KC"],
            "away_team": ["A", "C", "E", "DEN"],
            "home_score": [24.0, 20.0, 30.0, None],
            "away_score": [20.0, 23.0, 13.0, None],
            "spread_line": [3.0, 2.5, 7.0, 2.5],
            "total_line": [43.5, 44.0, 41.0, 43.0],
        }
    )
    finals = lined_finals(schedules)
    game = schedules.iloc[3]
    consensus = MarketConsensus(
        game_id="2026_01_DEN_KC", home_expected_margin=2.5, total_line=43.0, source="test"
    )
    view = TotalsView(
        predicted_total=43.42,
        market_total=43.0,
        residual=0.42,
        train_games=4_630,
        source="test",
    )

    blended = build_report(game, consensus, finals, None, view)
    assert blended.totals_view is view
    assert blended.guess_total_line == pytest.approx(43.0 + TOTALS_RESIDUAL_WEIGHT * 0.42)
    # The implied scores are built from the BLENDED total, not the market one.
    assert blended.implied_home + blended.implied_away == pytest.approx(blended.guess_total_line)

    # Without a totals view the guess total is the market's alone -- the
    # pre-regime behaviour, preserved exactly.
    market_only = build_report(game, consensus, finals)
    assert market_only.totals_view is None
    assert market_only.guess_total_line == pytest.approx(43.0)


def test_tiebreaker_report_line_names_the_totals_disagreement_and_the_weight() -> None:
    from nfl_ats.tiebreaker import (
        TOTALS_RESIDUAL_WEIGHT,
        MarketConsensus,
        build_report,
        format_report,
        lined_finals,
    )

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
    consensus = MarketConsensus(
        game_id="2026_01_DEN_KC", home_expected_margin=2.5, total_line=43.0, source="test"
    )
    view = TotalsView(
        predicted_total=43.42, market_total=43.0, residual=0.42, train_games=4_630, source="src"
    )
    text = format_report(
        build_report(schedules.iloc[2], consensus, lined_finals(schedules), None, view)
    )
    assert "model total view" in text
    assert "+0.42" in text
    assert f"weight {TOTALS_RESIDUAL_WEIGHT:g}" in text
    # No totals view -> no totals lines at all.
    market_only = format_report(build_report(schedules.iloc[2], consensus, lined_finals(schedules)))
    assert "model total view" not in market_only


def test_totals_blend_cannot_move_the_neighborhood_across_a_line_bucket() -> None:
    """The totals blend must never move the guess by moving a WINDOW EDGE.

    This replaces a test that pinned the opposite (WP14, 2026-09-01). The old
    ``_neighborhood`` used a HARD +/-1.5-point total window, and quoted totals
    are quantized to half points, so a blend nudge smaller than the quantum
    dropped or added a whole bucket of comparable games. Measured on the live
    board 2026-09-01: with the market total 43.0 the window held 259 games
    (buckets 41.5 through 44.5); at the blended 43.0421 it held 221 (the 41.5
    bucket fell outside), the median actual total moved 43 -> 41, and the
    published guess moved DOWN from KC 23 - DEN 20 to KC 22 - DEN 19 while the
    totals model was arguing the total should be HIGHER (+0.42). A displayed
    number moving the wrong way off a mechanical edge is a defect. The
    neighborhood is now kernel-weighted and continuous in its centre, so the
    0.042-point nudge moves the LINE (as the model said) and leaves the
    comparable-game weighting, the median and the guess where they were.
    """

    from nfl_ats.tiebreaker import MarketConsensus, build_report, lined_finals

    # Three totals buckets around a 43.0 centre. The 41.5 one sits EXACTLY on
    # the retired hard window's edge -- the position that used to be worth a
    # full vote on one side of the nudge and nothing on the other.
    rows = []
    index = 0
    for total_line, actual_total in ((41.5, 41), (43.0, 43), (44.0, 47)):
        for _ in range(160):
            rows.append(
                {
                    "game_id": f"2020_01_{index:05d}",
                    "season": 2020,
                    "week": 1,
                    "game_type": "REG",
                    "gameday": "2020-09-10",
                    "gametime": "13:00",
                    "home_team": "H",
                    "away_team": "A",
                    "home_score": float(actual_total - 10),
                    "away_score": 10.0,
                    "spread_line": 2.5,
                    "total_line": total_line,
                }
            )
            index += 1
    rows.append(
        {
            "game_id": "2026_01_DEN_KC",
            "season": 2026,
            "week": 1,
            "game_type": "REG",
            "gameday": "2026-09-14",
            "gametime": "20:15",
            "home_team": "KC",
            "away_team": "DEN",
            "home_score": None,
            "away_score": None,
            "spread_line": 2.5,
            "total_line": 43.0,
        }
    )
    schedules = pd.DataFrame(rows)
    finals = lined_finals(schedules)
    game = schedules.iloc[-1]
    consensus = MarketConsensus(
        game_id="2026_01_DEN_KC", home_expected_margin=2.5, total_line=43.0, source="test"
    )
    view = TotalsView(
        predicted_total=43.42, market_total=43.0, residual=0.42, train_games=4_630, source="test"
    )

    market_only = build_report(game, consensus, finals)
    blended = build_report(game, consensus, finals, None, view)

    # The blended line moves UP, exactly as the model argued.
    assert blended.guess_total_line > market_only.guess_total_line
    # ... and nothing the guess is built from flips underneath it.
    assert blended.median_total == market_only.median_total
    assert blended.median_home_margin == market_only.median_home_margin
    assert (blended.guess_home, blended.guess_away) == (
        market_only.guess_home,
        market_only.guess_away,
    )
    # The guess total may not move AGAINST an upward push on the line.
    assert (blended.guess_home + blended.guess_away) >= (
        market_only.guess_home + market_only.guess_away
    )
    # The effective neighborhood is essentially unchanged -- no bucket-sized
    # cliff (it used to halve, 320 -> 160, on this very history).
    assert abs(blended.neighborhood_games - market_only.neighborhood_games) <= 10
