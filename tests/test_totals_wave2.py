"""Tests for the wave-2 totals screen (``docs/totals_model_wave2.md``, WP18).

Four groups, mirroring ``tests/test_totals.py``'s structure: allowlist
enforcement for the extended 65-column list, the point-in-time join, the
wave-vs-wave paired comparison math, and a light-weight positive-control
shape check. The PBP drive family's own leakage safety is already covered by
``tests/test_pbp.py::test_current_game_plays_cannot_change_current_pregame_features``
(exercises ``enrich_with_pbp_features`` generically over
``PBP_ENRICHMENT_STATE_METRICS``, which includes every ``DRIVE_STATE_METRICS``
entry) -- not re-tested here, per the predeclaration.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from nfl_ats.constants import DRIVE_STATE_METRICS
from nfl_ats.totals import (
    TOTALS_FEATURES,
    TOTALS_RIDGE_ALPHA,
    TotalsDataError,
    blend_sweep,
    choose_weight,
    design_matrix,
    make_totals_estimator,
    walk_forward_predictions,
)
from nfl_ats.totals_wave2 import (
    WAVE1_CHOSEN_K,
    WAVE2_DRIVE_FEATURES,
    WAVE2_FEATURES,
    bootstrap_wave_vs_wave,
    load_population_wave2,
    model_total_view_wave2,
    wave_vs_wave_paired_frame,
)

_TARGET = "total_residual"


# ---------------------------------------------------------------------------
# 1. Allowlist enforcement for the extended list
# ---------------------------------------------------------------------------


def test_wave2_drive_features_is_exactly_home_away_cross_drive_state_metrics() -> None:
    assert len(WAVE2_DRIVE_FEATURES) == 24
    expected = tuple(
        f"{side}_{metric}" for metric in DRIVE_STATE_METRICS for side in ("home", "away")
    )
    assert expected == WAVE2_DRIVE_FEATURES
    # No diff_* and no bare pbp_drives count column snuck in.
    for column in WAVE2_DRIVE_FEATURES:
        assert not column.startswith("diff_")
        assert "pbp_drives" not in column


def test_wave2_features_is_wave1_plus_the_24_drive_columns_nothing_else() -> None:
    assert tuple(TOTALS_FEATURES) + WAVE2_DRIVE_FEATURES == WAVE2_FEATURES
    assert len(WAVE2_FEATURES) == 41 + 24 == 65
    # Every wave-1 column survives unchanged and in its original relative order.
    assert WAVE2_FEATURES[:41] == tuple(TOTALS_FEATURES)
    # No duplicates between the two halves.
    assert len(set(WAVE2_FEATURES)) == len(WAVE2_FEATURES)


def _wave2_synthetic_population(
    *, weeks: int = 6, games_per_week: int = 40, flip_week: int = 4, season: int = 2000
) -> pd.DataFrame:
    """Like ``test_totals.py``'s fixture, but the signal rides a DRIVE column
    instead of ``wind`` -- proves the guard holds for the extended list too."""

    rows = []
    generator = np.random.default_rng(20260901)
    driver_column = "home_drive_points_per_drive"
    for week in range(1, weeks + 1):
        slope = 5.0 if week < flip_week else -5.0
        for game in range(games_per_week):
            driver = float(generator.uniform(-1.0, 1.0))
            market_total = 44.0
            residual = slope * driver
            rows.append(
                {
                    "game_id": f"{season}_{week:02d}_{game:02d}",
                    "season": season,
                    "week": week,
                    "game_type": "REG",
                    "market_total": market_total,
                    "actual_total": market_total + residual,
                    _TARGET: residual,
                    driver_column: driver,
                }
            )
    frame = pd.DataFrame(rows)
    for column in WAVE2_FEATURES:
        if column not in frame.columns:
            frame[column] = 0.0
    return frame


def test_design_matrix_selects_exactly_the_65_column_allowlist() -> None:
    population = _wave2_synthetic_population(weeks=2, games_per_week=5)
    matrix = design_matrix(population, WAVE2_FEATURES)
    assert list(matrix.columns) == list(WAVE2_FEATURES)
    for banned in ("actual_total", _TARGET, "game_id", "season", "week", "game_type"):
        assert banned not in matrix.columns


def test_a_renamed_drive_column_is_a_hard_error_not_a_substitution() -> None:
    population = _wave2_synthetic_population(weeks=2, games_per_week=5)
    renamed = population.rename(columns={"home_drive_points_per_drive": "home_drive_pts_per_drive"})
    with pytest.raises(TotalsDataError, match="home_drive_points_per_drive"):
        design_matrix(renamed, WAVE2_FEATURES)


# ---------------------------------------------------------------------------
# 2. Join point-in-time proof
# ---------------------------------------------------------------------------


def test_walk_forward_guard_holds_when_a_drive_column_drives_the_signal() -> None:
    """The decisive comparison from ``test_totals.py``, replayed with a drive
    column as the flip-week driver and the full 65-column allowlist: the
    guarded prediction at the flip week must equal a model trained on weeks
    1-3 only, and must DIFFER from one that also saw week 4."""

    population = _wave2_synthetic_population()
    predictions = walk_forward_predictions(population, min_train_games=40, features=WAVE2_FEATURES)
    scored_weeks = sorted(predictions["week"].unique())
    assert scored_weeks == [2, 3, 4, 5, 6]

    target_week = 4
    honest_train = population.loc[population["week"] < target_week]
    leaky_train = population.loc[population["week"] <= target_week]
    target_rows = population.loc[population["week"] == target_week]

    honest = make_totals_estimator(ridge_alpha=TOTALS_RIDGE_ALPHA)
    honest.fit(design_matrix(honest_train, WAVE2_FEATURES), honest_train[_TARGET].to_numpy())
    leaky = make_totals_estimator(ridge_alpha=TOTALS_RIDGE_ALPHA)
    leaky.fit(design_matrix(leaky_train, WAVE2_FEATURES), leaky_train[_TARGET].to_numpy())

    honest_prediction = np.asarray(
        honest.predict(design_matrix(target_rows, WAVE2_FEATURES)), dtype=float
    )
    leaky_prediction = np.asarray(
        leaky.predict(design_matrix(target_rows, WAVE2_FEATURES)), dtype=float
    )
    walked = predictions.loc[predictions["week"] == target_week, "predicted_residual"].to_numpy()

    assert walked == pytest.approx(honest_prediction)
    assert np.abs(leaky_prediction - honest_prediction).max() > 1.0
    assert not np.allclose(walked, leaky_prediction)


def _write_wave2_population_fixture(root: Path) -> tuple[Path, Path]:
    schedules = pd.DataFrame(
        {
            "game_id": ["2020_01_A_B", "2020_01_C_D", "2020_02_E_F", "2026_01_X_Y"],
            "season": [2020, 2020, 2020, 2026],
            "week": [1, 1, 2, 1],
            "game_type": ["REG", "REG", "REG", "REG"],
            "gameday": ["2020-09-10", "2020-09-13", "2020-09-20", "2026-09-13"],
            "home_team": ["B", "D", "F", "Y"],
            "away_team": ["A", "C", "E", "X"],
            "home_score": [24.0, 20.0, 30.0, None],
            "away_score": [20.0, 23.0, 13.0, None],
            "total_line": [43.5, 44.0, 41.0, 44.5],
        }
    )
    raw = root / "raw" / "20260901T000000Z"
    raw.mkdir(parents=True)
    schedules.to_parquet(raw / "schedules.parquet")

    # wave-1 feature table: only the 41-column allowlist.
    wave1_features = pd.DataFrame({"game_id": schedules["game_id"]})
    for column in TOTALS_FEATURES:
        wave1_features[column] = 0.5
    wave1_features["season"] = schedules["season"]
    wave1_features["week"] = schedules["week"]
    wave1_features["total_line"] = schedules["total_line"]
    wave1_path = root / "processed" / "game_features.parquet"
    wave1_path.parent.mkdir(parents=True)
    wave1_features.to_parquet(wave1_path)

    # wave-2 feature table: the 41-column allowlist PLUS the 24 drive columns,
    # with values distinguishable from wave 1's (0.5) to prove they actually
    # get selected, plus a diff_* decoy that must never enter the join.
    wave2_features = wave1_features.copy()
    for index, column in enumerate(WAVE2_DRIVE_FEATURES):
        wave2_features[column] = 1.0 + 0.01 * index
    wave2_features["diff_drive_points_per_drive"] = 999.0
    wave2_features["home_pbp_drives"] = 777.0
    wave2_path = root / "processed" / "game_features_pbp.parquet"
    wave2_features.to_parquet(wave2_path)
    return wave1_path, wave2_path


def test_load_population_wave2_matches_wave1s_game_set_and_pulls_drive_values(
    tmp_path: Path,
) -> None:
    from nfl_ats.totals import load_population as load_population_wave1

    wave1_path, wave2_path = _write_wave2_population_fixture(tmp_path)
    wave1_population = load_population_wave1(tmp_path, wave1_path)
    wave2_population = load_population_wave2(tmp_path, wave2_path)

    # Identical scored game set -- the wider feature table adds columns, not
    # rows, and drops none of wave 1's.
    assert set(wave2_population["game_id"]) == set(wave1_population["game_id"])
    assert len(wave2_population) == len(wave1_population) == 3

    # Drive-column values came through the join unmolested, and the diff_*
    # decoy / pbp_drives count column never entered the population frame.
    for index, column in enumerate(WAVE2_DRIVE_FEATURES):
        assert wave2_population[column].tolist() == pytest.approx([1.0 + 0.01 * index] * 3)
    assert "diff_drive_points_per_drive" not in wave2_population.columns
    assert "home_pbp_drives" not in wave2_population.columns

    # The target and market total are computed identically to wave 1's.
    assert wave2_population[_TARGET].tolist() == pytest.approx(wave1_population[_TARGET].tolist())


def test_load_population_wave2_on_real_data_matches_wave1_game_set() -> None:
    """Real-file join integrity: the production ``game_features_pbp.parquet``
    scores the identical population wave 1's ``game_features.parquet`` does."""

    from nfl_ats.totals import load_population as load_population_wave1

    data_root = Path("data")
    wave1_path = data_root / "processed" / "game_features.parquet"
    wave2_path = data_root / "processed" / "game_features_pbp.parquet"
    if not wave1_path.is_file() or not wave2_path.is_file():
        pytest.skip("production feature tables not present in this checkout")

    wave1_population = load_population_wave1(data_root, wave1_path)
    wave2_population = load_population_wave2(data_root, wave2_path)
    assert set(wave2_population["game_id"]) == set(wave1_population["game_id"])

    raw_wave2_table = pd.read_parquet(wave2_path)
    sample = wave2_population.sample(n=min(25, len(wave2_population)), random_state=20260901)
    lookup = raw_wave2_table.set_index("game_id")
    for _, row in sample.iterrows():
        source_row = lookup.loc[row["game_id"]]
        for column in WAVE2_DRIVE_FEATURES:
            a, b = row[column], source_row[column]
            if pd.isna(a) and pd.isna(b):
                continue
            assert a == pytest.approx(float(b))


# ---------------------------------------------------------------------------
# 3. Wave-vs-wave paired comparison math
# ---------------------------------------------------------------------------


def test_wave_vs_wave_paired_frame_sign_convention_is_wave1_minus_wave2() -> None:
    wave1_predictions = pd.DataFrame(
        {
            "game_id": ["a", "b"],
            "season": [2020, 2020],
            "week": [1, 1],
            "market_total": [44.0, 44.0],
            "predicted_residual": [0.0, 0.0],
            "actual_total": [46.0, 42.0],
        }
    )
    # wave1 blend at k=0.1 with predicted_residual 0 == market alone -> |error| 2,2
    wave2_predictions = pd.DataFrame(
        {
            "game_id": ["a", "b"],
            "season": [2020, 2020],
            "week": [1, 1],
            "market_total": [44.0, 44.0],
            "predicted_residual": [2.0, 2.0],
            "actual_total": [46.0, 42.0],
        }
    )
    # wave2 blend at k=1.0: predicted totals 46, 46 -> |error| 0, 4
    paired = wave_vs_wave_paired_frame(wave1_predictions, 0.1, wave2_predictions, 1.0)
    assert paired["wave1_abs_error"].tolist() == pytest.approx([2.0, 2.0])
    assert paired["wave2_abs_error"].tolist() == pytest.approx([0.0, 4.0])
    assert paired["abs_error_improvement"].tolist() == pytest.approx([2.0, -2.0])


def test_wave_vs_wave_paired_frame_rejects_mismatched_game_sets() -> None:
    wave1_predictions = pd.DataFrame(
        {
            "game_id": ["a", "b"],
            "season": [2020, 2020],
            "week": [1, 1],
            "market_total": [44.0, 44.0],
            "predicted_residual": [0.0, 0.0],
            "actual_total": [46.0, 42.0],
        }
    )
    wave2_predictions = pd.DataFrame(
        {
            "game_id": ["a", "c"],
            "season": [2020, 2020],
            "week": [1, 1],
            "market_total": [44.0, 44.0],
            "predicted_residual": [0.0, 0.0],
            "actual_total": [46.0, 42.0],
        }
    )
    with pytest.raises(TotalsDataError, match="scored game sets differ"):
        wave_vs_wave_paired_frame(wave1_predictions, 0.1, wave2_predictions, 0.1)


def test_bootstrap_wave_vs_wave_reports_probability_positive_not_a_binary_read() -> None:
    generator = np.random.default_rng(20260901)
    rows = []
    for week in range(1, 11):
        for game in range(20):
            rows.append(
                {
                    "game_id": f"g_{week}_{game}",
                    "season": 2020,
                    "week": week,
                    "abs_error_improvement": float(generator.normal(0.05, 1.0)),
                }
            )
    paired = pd.DataFrame(rows)
    result = bootstrap_wave_vs_wave(paired, samples=500, seed=20260901)
    assert 0.0 <= result["probability_positive"] <= 1.0
    assert result["lower"] <= result["estimate"] <= result["upper"]
    assert result["games"] == len(paired)
    assert result["blocks"] == 10


def test_wave1_chosen_k_is_frozen_at_wave1s_own_operating_point() -> None:
    # Not re-derived: the paired comparison must grade wave 1 at its own
    # already-chosen k, never a k re-swept on wave 2's run.
    assert pytest.approx(0.1) == WAVE1_CHOSEN_K


# ---------------------------------------------------------------------------
# 4. Positive-control shape check (light-weight; the full-data run is
#    produced by scripts/totals_wave2_backtest.py --mode positive-control)
# ---------------------------------------------------------------------------


def test_injecting_the_target_into_a_drive_column_drives_k_toward_one() -> None:
    generator = np.random.default_rng(20260901)
    rows = []
    for week in range(1, 9):
        for game in range(40):
            market_total = 44.0
            residual = float(generator.normal(0.0, 6.0))
            rows.append(
                {
                    "game_id": f"2000_{week:02d}_{game:03d}",
                    "season": 2000,
                    "week": week,
                    "game_type": "REG",
                    "market_total": market_total,
                    "actual_total": market_total + residual,
                    _TARGET: residual,
                }
            )
    population = pd.DataFrame(rows)
    for column in WAVE2_FEATURES:
        if column not in population.columns:
            population[column] = 0.0
    # The positive control: one drive column IS the target.
    contaminated = population.copy()
    contaminated["home_drive_points_per_drive"] = contaminated[_TARGET].astype(float)

    predictions = walk_forward_predictions(
        contaminated, min_train_games=40, features=WAVE2_FEATURES
    )
    sweep = blend_sweep(predictions)
    chosen = choose_weight(sweep)

    # A clean run (no injection) should NOT push k toward 1 the same way.
    clean_predictions = walk_forward_predictions(
        population, min_train_games=40, features=WAVE2_FEATURES
    )
    clean_sweep = blend_sweep(clean_predictions)
    clean_chosen = choose_weight(clean_sweep)

    assert chosen >= 0.8
    assert chosen > clean_chosen
    contaminated_mae_improvement = sweep.loc[
        np.isclose(sweep["k"], chosen), "mae_improvement_vs_market"
    ].iloc[0]
    assert contaminated_mae_improvement > 1.0


# ---------------------------------------------------------------------------
# 5. model_total_view_wave2 serving (WP27, 2026-09-01)
#
# Mirrors nfl_ats.totals.model_total_view's own test coverage plus the two
# behaviours specific to wave 2: allowlist enforcement in serving, and the
# "no PBP row for this game" case. Design choice, stated in the function's
# own docstring and pinned here: a missing single-game PBP row falls back to
# MARKET-ONLY (returns None), mirroring model_total_view's exact contract --
# it does NOT reach across to wave 1 internally. The wave-1-VIEW fallback is
# a decision made one level up, in nfl_ats.tiebreaker.tiebreaker_report, and
# fires only when the whole PBP table file is absent, never when a single
# game's row is merely missing from an existing table.
# ---------------------------------------------------------------------------


def test_model_total_view_wave2_trains_only_on_games_before_the_target_week(
    tmp_path: Path,
) -> None:
    _wave1_path, wave2_path = _write_wave2_population_fixture(tmp_path)
    # Under any realistic floor the three 2020 finals are not enough.
    assert model_total_view_wave2("2026_01_X_Y", tmp_path, wave2_path, min_train_games=500) is None

    view = model_total_view_wave2("2026_01_X_Y", tmp_path, wave2_path, min_train_games=3)
    assert view is not None
    assert view.train_games == 3  # the three 2020 finals, none from 2026
    assert view.market_total == pytest.approx(44.5)
    assert view.predicted_total == pytest.approx(view.market_total + view.residual)
    # The report line must say which wave served the number.
    assert "wave 2" in view.source
    assert "65 cols" in view.source
    assert "drive pace" in view.source

    # A game the feature table does not price gets no view rather than a guess.
    assert (
        model_total_view_wave2("2026_01_NO_SUCH", tmp_path, wave2_path, min_train_games=3) is None
    )
    # No PBP table at all -> None, the same contract wave 1's own
    # model_total_view has for a missing table.
    assert model_total_view_wave2("2026_01_X_Y", tmp_path, tmp_path / "absent.parquet") is None


def test_model_total_view_wave2_with_no_pbp_row_falls_back_to_market_only(
    tmp_path: Path,
) -> None:
    """The design choice this test pins: a PBP table that EXISTS but carries
    no row for this particular game_id (a game the PBP pipeline has not
    enriched) yields None -- market-only -- rather than an internal
    substitution of wave 1's number. See the module docstring for why that
    substitution belongs one level up instead."""

    _wave1_path, wave2_path = _write_wave2_population_fixture(tmp_path)
    assert (
        model_total_view_wave2("2026_01_NO_SUCH_GAME", tmp_path, wave2_path, min_train_games=3)
        is None
    )


def test_model_total_view_wave2_serving_enforces_the_65_column_allowlist(
    tmp_path: Path,
) -> None:
    _wave1_path, wave2_path = _write_wave2_population_fixture(tmp_path)
    baseline = model_total_view_wave2("2026_01_X_Y", tmp_path, wave2_path, min_train_games=3)
    assert baseline is not None

    # Decoy columns already in the fixture (diff_drive_points_per_drive,
    # home_pbp_drives) never enter the fit -- inflating them a thousandfold
    # must not move the served residual at all.
    doctored = pd.read_parquet(wave2_path)
    doctored["diff_drive_points_per_drive"] *= 1_000.0
    doctored["home_pbp_drives"] *= 1_000.0
    doctored_path = tmp_path / "processed" / "game_features_pbp_doctored.parquet"
    doctored.to_parquet(doctored_path)
    doctored_view = model_total_view_wave2(
        "2026_01_X_Y", tmp_path, doctored_path, min_train_games=3
    )
    assert doctored_view is not None
    assert doctored_view.residual == pytest.approx(baseline.residual)

    # A renamed allowlist column is a hard error, never a silent substitution
    # (design_matrix's own guarantee, exercised here through the serving path).
    renamed = pd.read_parquet(wave2_path).rename(columns={"home_off_cpoe": "home_offense_cpoe"})
    renamed_path = tmp_path / "processed" / "game_features_pbp_renamed.parquet"
    renamed.to_parquet(renamed_path)
    with pytest.raises(TotalsDataError, match="home_off_cpoe"):
        model_total_view_wave2("2026_01_X_Y", tmp_path, renamed_path, min_train_games=3)


def test_model_total_view_wave2_fails_closed_for_a_stale_or_misaligned_table(
    tmp_path: Path,
) -> None:
    """A table from a different schedule snapshot must not serve a residual.

    The target row can still be present in a stale build, so checking only for
    that row is insufficient.  The serving guard compares the complete game
    identity and market-line columns before fitting anything.
    """

    _wave1_path, wave2_path = _write_wave2_population_fixture(tmp_path)
    baseline = model_total_view_wave2("2026_01_X_Y", tmp_path, wave2_path, min_train_games=3)
    assert baseline is not None

    stale = pd.read_parquet(wave2_path)
    stale.loc[stale["game_id"] == "2026_01_X_Y", "total_line"] = 99.5
    stale_path = tmp_path / "processed" / "game_features_pbp_stale.parquet"
    stale.to_parquet(stale_path)
    assert model_total_view_wave2("2026_01_X_Y", tmp_path, stale_path, min_train_games=3) is None

    partial = pd.read_parquet(wave2_path).iloc[:-1]
    partial_path = tmp_path / "processed" / "game_features_pbp_partial.parquet"
    partial.to_parquet(partial_path)
    assert model_total_view_wave2("2026_01_X_Y", tmp_path, partial_path, min_train_games=3) is None

    duplicate = pd.concat([pd.read_parquet(wave2_path), pd.read_parquet(wave2_path).iloc[[0]]])
    duplicate_path = tmp_path / "processed" / "game_features_pbp_duplicate.parquet"
    duplicate.to_parquet(duplicate_path)
    assert (
        model_total_view_wave2("2026_01_X_Y", tmp_path, duplicate_path, min_train_games=3) is None
    )


def _write_walk_forward_guard_fixture(root: Path) -> Path:
    """20 no-signal week-1 games, 20 STRONG-signal week-3 games, and one
    unplayed week-3 TARGET game. If ``model_total_view_wave2``'s own prior
    filter used ``<=`` instead of ``<`` on the target's week, the week-3
    non-target games (strong signal, slope 20 on ``home_drive_points_per_drive``)
    would leak into the target's training pool and move its prediction by a
    large, unambiguous margin relative to a fit that honestly saw only the
    no-signal week-1 games."""

    generator = np.random.default_rng(20260901)
    driver_week1 = generator.uniform(-1.0, 1.0, size=20)
    driver_week3 = generator.uniform(-1.0, 1.0, size=20)
    residual_week1 = np.zeros(20)  # no signal
    residual_week3 = 20.0 * driver_week3  # strong signal

    rows = []
    for index in range(20):
        rows.append(
            {
                "game_id": f"2020_01_{index:02d}",
                "season": 2020,
                "week": 1,
                "game_type": "REG",
                "home_score": 30.0 + residual_week1[index],
                "away_score": 14.0,
                "total_line": 44.0,
                "driver": driver_week1[index],
            }
        )
    for index in range(20):
        rows.append(
            {
                "game_id": f"2020_03_{index:02d}",
                "season": 2020,
                "week": 3,
                "game_type": "REG",
                "home_score": 30.0 + residual_week3[index],
                "away_score": 14.0,
                "total_line": 44.0,
                "driver": driver_week3[index],
            }
        )
    rows.append(
        {
            "game_id": "2020_03_TARGET",
            "season": 2020,
            "week": 3,
            "game_type": "REG",
            "home_score": None,
            "away_score": None,
            "total_line": 44.0,
            "driver": 1.0,
        }
    )
    schedules = pd.DataFrame(rows)
    raw = root / "raw" / "20260901T000000Z"
    raw.mkdir(parents=True)
    schedules.to_parquet(raw / "schedules.parquet")

    features = pd.DataFrame({"game_id": schedules["game_id"]})
    for column in TOTALS_FEATURES:
        features[column] = 0.5
    features["total_line"] = schedules["total_line"]
    features["season"] = schedules["season"]
    features["week"] = schedules["week"]
    for column in WAVE2_DRIVE_FEATURES:
        features[column] = 0.0
    features["home_drive_points_per_drive"] = schedules["driver"]
    features_path = root / "processed" / "game_features_pbp.parquet"
    features_path.parent.mkdir(parents=True)
    features.to_parquet(features_path)
    return features_path


def test_model_total_view_wave2_walk_forward_guard_excludes_the_target_week(
    tmp_path: Path,
) -> None:
    features_path = _write_walk_forward_guard_fixture(tmp_path)
    population = load_population_wave2(tmp_path, features_path)

    target_season, target_week = 2020, 3
    honest_train = population.loc[
        (population["season"] < target_season)
        | ((population["season"] == target_season) & (population["week"] < target_week))
    ]
    leaky_train = population.loc[
        (population["season"] < target_season)
        | ((population["season"] == target_season) & (population["week"] <= target_week))
    ]
    assert len(honest_train) == 20  # week 1 only
    assert len(leaky_train) == 40  # week 1 + the 20 non-target week-3 games

    raw_features = pd.read_parquet(features_path)
    target_row = raw_features.loc[raw_features["game_id"] == "2020_03_TARGET"]

    honest = make_totals_estimator(ridge_alpha=TOTALS_RIDGE_ALPHA)
    honest.fit(
        design_matrix(honest_train, WAVE2_FEATURES), honest_train["total_residual"].to_numpy()
    )
    leaky = make_totals_estimator(ridge_alpha=TOTALS_RIDGE_ALPHA)
    leaky.fit(design_matrix(leaky_train, WAVE2_FEATURES), leaky_train["total_residual"].to_numpy())

    honest_prediction = float(
        np.asarray(honest.predict(design_matrix(target_row, WAVE2_FEATURES)), dtype=float)[0]
    )
    leaky_prediction = float(
        np.asarray(leaky.predict(design_matrix(target_row, WAVE2_FEATURES)), dtype=float)[0]
    )
    # The leak is not a rounding difference: the strong week-3 signal moves
    # the answer by several points.
    assert abs(leaky_prediction - honest_prediction) > 1.0

    view = model_total_view_wave2("2020_03_TARGET", tmp_path, features_path, min_train_games=15)
    assert view is not None
    assert view.train_games == 20
    assert view.residual == pytest.approx(honest_prediction)
    assert view.residual != pytest.approx(leaky_prediction)
