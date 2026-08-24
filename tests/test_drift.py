"""RWB-12 drift monitoring: signals, thresholds, artifacts, and pipeline wiring.

Drift reports are operational telemetry, not evidence. These tests pin the
signals (feature shift, PSI, missingness, probability drift, calibration
drift), the fail-loud behavior on missing inputs, the artifact format, and
the weekly-pipeline hook -- including that the hook is optional and never on
the card's critical path.

Constructions are deterministic wherever a threshold sits nearby: random
draws at n=16 sit close enough to the warn/alert boundaries that a seeded
RNG would still be flaky under ``-k`` selection.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from nfl_ats.cli import build_parser
from nfl_ats.constants import FEATURE_FAMILIES
from nfl_ats.drift import (
    _ece,
    build_drift_report,
    calibration_drift_summary,
    feature_drift_table,
    probability_drift_summary,
    psi,
    reference_window,
    registered_feature_columns,
    summarize_feature_drift,
    worst_status,
    write_drift_artifacts,
)
from nfl_ats.weekly import plan_weekly_run

# ---------------------------------------------------------------------------
# Synthetic frames. Registered columns only, so the monitored set is stable.
# ---------------------------------------------------------------------------


def _features_frame(weeks: list[tuple[int, int]], games_per_week: int = 16) -> pd.DataFrame:
    """Sixteen prior weeks plus whatever target week the caller asks about."""

    rng = np.random.default_rng(20260825)
    rows = []
    game_index = 0
    for season, week in weeks:
        for _ in range(games_per_week):
            game_index += 1
            rows.append(
                {
                    "game_id": f"{season}_{week:02d}_g{game_index}",
                    "season": season,
                    "week": week,
                    "gameday": pd.Timestamp(f"{season}-01-01") + pd.Timedelta(days=7 * week),
                    "spread_line": float(rng.normal(-2.5, 6.0)),
                    "elo_diff": float(rng.normal(0.0, 40.0)),
                    "rest_diff": float(rng.choice([-7.0, 0.0, 3.0, 7.0])),
                    "result": float(rng.normal(0.0, 14.0)),
                }
            )
    return pd.DataFrame(rows)


def _predictions_frame(
    season: int,
    week: int,
    features: pd.DataFrame,
    *,
    method: str = "market_residual",
    probabilities: np.ndarray | None = None,
    outcomes: np.ndarray | None = None,
) -> pd.DataFrame:
    target = features.loc[features["season"].eq(season) & features["week"].eq(week)]
    if probabilities is None:
        # Symmetric around 0.5 and entirely inside any plausible reference
        # band, so the default never trips the probability-drift thresholds.
        probabilities = np.linspace(0.30, 0.70, len(target))
    if outcomes is None:
        outcomes = np.tile([1.0, 0.0], ceil_div(len(target), 2))[: len(target)]
    return pd.DataFrame(
        {
            "game_id": target["game_id"].to_numpy(),
            "season": season,
            "week": week,
            "gameday": target["gameday"].to_numpy(),
            "method": method,
            "home_cover_probability": probabilities,
            "home_cover": outcomes,
        }
    )


def ceil_div(a: int, b: int) -> int:
    return -(-a // b)


def _deterministically_calibrated_outcomes(probabilities: np.ndarray, seed: int = 0) -> np.ndarray:
    """Outcomes drawn as ``Bernoulli(p)`` against a stratified uniform draw.

    The uniforms are a fixed grid permuted by a seeded RNG, so the run is
    deterministic while the empirical cover rate in every probability region
    tracks that region's mean predicted probability -- a genuinely
    well-calibrated baseline without RNG flake at the thresholds.
    """

    n = len(probabilities)
    uniforms = np.random.default_rng(seed).permutation((np.arange(n) + 0.5) / n)
    return (uniforms < probabilities).astype(float)


def _history_predictions(features: pd.DataFrame, weeks: list[tuple[int, int]]) -> pd.DataFrame:
    """Enough settled history to clear the calibration floors (32 recent / 200 prior).

    Each week is repeated across twelve synthetic cards with distinct game ids,
    so the most recent four weeks hold 4 x 16 x 12 = 768 settled games and the
    prior holds the rest.
    """

    frames = []
    for repetition in range(12):
        chunk = features.copy()
        chunk["game_id"] = [f"hist_{repetition}_{i}" for i in range(len(chunk))]
        rng = np.random.default_rng(900 + repetition)
        for season, week in weeks:
            target = chunk.loc[chunk["season"].eq(season) & chunk["week"].eq(week)]
            probs = np.clip(rng.normal(0.5, 0.15, len(target)), 0.05, 0.95)
            frames.append(
                pd.DataFrame(
                    {
                        "game_id": target["game_id"].to_numpy(),
                        "season": season,
                        "week": week,
                        "gameday": target["gameday"].to_numpy(),
                        "method": "market_residual",
                        "home_cover_probability": probs,
                        "home_cover": _deterministically_calibrated_outcomes(
                            probs, seed=900 + repetition
                        ),
                    }
                )
            )
    return pd.concat(frames, ignore_index=True)


@pytest.fixture()
def season_frame() -> pd.DataFrame:
    return _features_frame([(2025, week) for week in range(1, 18)])


# ---------------------------------------------------------------------------
# Status folding and the raw PSI signal
# ---------------------------------------------------------------------------


def test_worst_status_folds_in_severity_order() -> None:
    assert worst_status(["ok", "ok"]) == "ok"
    assert worst_status(["ok", "warn", "insufficient_history"]) == "warn"
    assert worst_status(["insufficient_history", "alert", "ok"]) == "alert"
    assert worst_status([]) == "ok"


def test_psi_is_near_zero_for_identical_distributions_and_large_for_a_shift() -> None:
    local = np.random.default_rng(7)
    reference = pd.Series(local.normal(0.0, 1.0, 5_000))
    same = pd.Series(local.normal(0.0, 1.0, 500))
    shifted = pd.Series(local.normal(1.0, 1.0, 500))
    assert psi(same, reference) < 0.05
    assert psi(shifted, reference) > 0.5


def test_psi_catches_a_constant_column_going_nonconstant() -> None:
    constant_reference = pd.Series(np.zeros(500))
    moved = pd.Series(np.ones(100))
    assert psi(moved, constant_reference) > 1.0
    assert psi(pd.Series(np.zeros(100)), constant_reference) == pytest.approx(0.0, abs=1e-9)


# ---------------------------------------------------------------------------
# Feature and missingness drift table
# ---------------------------------------------------------------------------


def test_feature_table_reports_no_alerts_on_a_stable_week(
    season_frame: pd.DataFrame,
) -> None:
    current = season_frame.loc[season_frame["week"].eq(17)]
    reference, keys = reference_window(season_frame, season=2025, week=17, reference_weeks=8)
    assert len(keys) == 8
    table = feature_drift_table(current, reference)
    summary = summarize_feature_drift(table)
    # A 16-game window cannot score PSI (below the noise floor), and a mean
    # shift of a 16-game sample sits within a hair of the warn tier, so the
    # binding assertions here are "no alert, nothing silently dropped".
    assert summary["status"] != "alert"
    assert summary["alerts"] == []
    row = table.loc[table["column"].eq("spread_line")].iloc[0]
    assert np.isfinite(row["psi"])
    assert row["psi_status"] == "insufficient_history"


def test_feature_table_alerts_on_a_wholesale_level_shift(
    season_frame: pd.DataFrame,
) -> None:
    shifted = season_frame.copy()
    mask = shifted["week"].eq(17)
    shifted.loc[mask, "elo_diff"] = shifted.loc[mask, "elo_diff"] + 200.0
    current = shifted.loc[shifted["week"].eq(17)]
    reference, _ = reference_window(shifted, season=2025, week=17, reference_weeks=8)
    table = feature_drift_table(current, reference)
    row = table.loc[table["column"].eq("elo_diff")].iloc[0]
    assert row["shift_status"] == "alert"
    summary = summarize_feature_drift(table)
    assert summary["status"] == "alert"
    assert "elo_diff" in summary["alerts"]


def test_vanishing_column_is_reported_as_full_missingness(
    season_frame: pd.DataFrame,
) -> None:
    broken = season_frame.drop(columns=["rest_diff"])
    current = broken.loc[broken["week"].eq(17)]
    reference, _ = reference_window(season_frame, season=2025, week=17, reference_weeks=8)
    table = feature_drift_table(current, reference, columns=["rest_diff"])
    row = table.iloc[0]
    assert row["missingness_current_pct"] == 100.0
    assert row["missingness_delta_pp"] == 100.0
    assert row["missingness_status"] == "alert"
    assert row["psi_status"] == "insufficient_history"


def test_half_null_column_alerts_on_missingness(season_frame: pd.DataFrame) -> None:
    degraded = season_frame.copy()
    mask = degraded["week"].eq(17)
    degraded.loc[mask & (np.arange(len(degraded)) % 2 == 0), "rest_diff"] = np.nan
    current = degraded.loc[degraded["week"].eq(17)]
    reference, _ = reference_window(degraded, season=2025, week=17, reference_weeks=8)
    table = feature_drift_table(current, reference)
    row = table.loc[table["column"].eq("rest_diff")].iloc[0]
    assert row["missingness_delta_pp"] == pytest.approx(50.0)
    assert row["missingness_status"] == "alert"


def test_non_numeric_garbage_counts_as_missing(season_frame: pd.DataFrame) -> None:
    corrupted = season_frame.copy()
    corrupted["elo_diff"] = corrupted["elo_diff"].astype(object)
    corrupted.loc[corrupted["week"].eq(17), "elo_diff"] = "n/a"
    current = corrupted.loc[corrupted["week"].eq(17)]
    reference, _ = reference_window(corrupted, season=2025, week=17, reference_weeks=8)
    table = feature_drift_table(current, reference)
    row = table.loc[table["column"].eq("elo_diff")].iloc[0]
    assert row["missingness_current_pct"] == 100.0


# ---------------------------------------------------------------------------
# Probability and calibration drift
# ---------------------------------------------------------------------------


def test_probability_drift_flags_a_mean_shift_and_passes_a_matched_one() -> None:
    local = np.random.default_rng(11)
    reference = pd.Series(np.clip(local.normal(0.5, 0.15, 400), 0.02, 0.98))
    drifted = pd.Series(np.full(16, 0.62))
    summary = probability_drift_summary(drifted, reference)
    assert summary["status"] == "warn"
    assert summary["delta_mean"] > 0.05

    # Same spread, same size, centered exactly on the reference mean.
    matched = pd.Series(np.linspace(0.42, 0.58, 16))
    assert probability_drift_summary(matched, reference)["status"] == "ok"


def test_probability_drift_is_insufficient_without_history() -> None:
    summary = probability_drift_summary(pd.Series([0.6] * 4), pd.Series([0.5] * 400))
    assert summary["status"] == "insufficient_history"


def test_calibration_drift_insufficient_below_the_game_floors() -> None:
    settled = pd.DataFrame(
        {
            "season": [2025] * 40 + [2026] * 40,
            "week": [1] * 40 + [1] * 40,
            "gameday": pd.to_datetime(["2025-09-01"] * 40 + ["2026-09-01"] * 40),
            "home_cover_probability": np.full(80, 0.5),
            "home_cover": np.zeros(80),
        }
    )
    summary = calibration_drift_summary(settled, recent_weeks=4)
    assert summary["status"] == "insufficient_history"


def test_calibration_drift_detects_recent_miscalibration() -> None:
    def calibrated_block(
        n: int, *, season: int, week_start: int, games_per_week: int
    ) -> pd.DataFrame:
        rng = np.random.default_rng(1000 + season * 10 + week_start)
        probs = np.clip(rng.normal(0.5, 0.15, n), 0.05, 0.95)
        weeks = [week_start + i // games_per_week for i in range(n)]
        return pd.DataFrame(
            {
                "season": [season] * n,
                "week": weeks,
                "gameday": pd.date_range(f"{season}-01-01", periods=n, freq="D")
                + pd.to_timedelta([w * 7 for w in weeks], unit="D"),
                "home_cover_probability": probs,
                "home_cover": _deterministically_calibrated_outcomes(
                    probs, seed=season * 100 + week_start
                ),
            }
        )

    # Prior history: 500 settled games in weeks 1-32; recent: 96 in weeks 40-45.
    # The most recent four distinct weeks therefore hold >=32 games, the prior
    # holds well over 200, and the calibrated recent window sits far enough
    # from the warn boundary that the comparison is not RNG-flaky.
    prior = pd.concat(
        [
            calibrated_block(250, season=2024, week_start=1, games_per_week=16),
            calibrated_block(250, season=2024, week_start=17, games_per_week=16),
        ],
        ignore_index=True,
    )
    recent_ok = calibrated_block(96, season=2025, week_start=40, games_per_week=16)
    ok_summary = calibration_drift_summary(
        pd.concat([prior, recent_ok], ignore_index=True), recent_weeks=4
    )
    assert ok_summary["status"] == "ok"

    # Recent window: probabilities say 0.35, outcomes hit 100% -- Brier blows up.
    n_recent = 48
    miscalibrated = pd.DataFrame(
        {
            "season": [2025] * n_recent,
            "week": [60 + i // 16 for i in range(n_recent)],
            "gameday": pd.date_range("2025-06-01", periods=n_recent, freq="D"),
            "home_cover_probability": np.full(n_recent, 0.35),
            "home_cover": np.ones(n_recent),
        }
    )
    bad_summary = calibration_drift_summary(
        pd.concat([prior, miscalibrated], ignore_index=True), recent_weeks=4
    )
    assert bad_summary["status"] == "alert"
    assert bad_summary["delta_brier"] > 0.04


def test_ece_is_zero_for_a_perfectly_calibrated_constant_and_positive_otherwise() -> None:
    assert _ece(np.array([0.5, 0.5]), np.array([1.0, 0.0])) == pytest.approx(0.0)
    assert _ece(np.array([0.9, 0.9]), np.array([0.0, 0.0])) == pytest.approx(0.9)


# ---------------------------------------------------------------------------
# Full report assembly and artifact writing
# ---------------------------------------------------------------------------


def test_build_drift_report_end_to_end(season_frame: pd.DataFrame) -> None:
    history_weeks = [(2025, week) for week in range(1, 17)]
    history_features = season_frame.loc[~season_frame["week"].eq(17)]
    history = _history_predictions(history_features, history_weeks)
    current = _predictions_frame(2025, 17, season_frame)
    report, table = build_drift_report(
        season_frame,
        current,
        history,
        season=2025,
        week=17,
        feature_profile="player",
    )
    assert report["command"] == "drift-report"
    assert report["overall_status"] in {"ok", "warn"}
    assert report["sections"]["probability_drift"]["status"] == "ok"
    assert report["sections"]["calibration_drift"]["status"] == "ok"
    assert any("never be cited as evidence" in note for note in report["notes"])
    assert not table.empty
    known_families = set(FEATURE_FAMILIES) | {"unregistered"}
    assert set(table["feature_family"]).issubset(known_families)


def test_build_drift_report_insufficient_without_history(
    season_frame: pd.DataFrame,
) -> None:
    current = _predictions_frame(2025, 17, season_frame)
    report, _ = build_drift_report(
        season_frame,
        current,
        None,
        season=2025,
        week=17,
        feature_profile="player",
    )
    assert report["sections"]["probability_drift"]["status"] == "insufficient_history"
    assert report["sections"]["calibration_drift"]["status"] == "insufficient_history"


def test_reference_window_refuses_an_empty_or_unpreceded_target(
    season_frame: pd.DataFrame,
) -> None:
    with pytest.raises(ValueError, match="No games found"):
        reference_window(season_frame, season=2025, week=22)
    lonely = season_frame.loc[season_frame["week"].eq(1)].copy()
    with pytest.raises(ValueError, match="No completed games precede"):
        reference_window(lonely, season=2025, week=1)


def test_write_drift_artifacts_creates_json_and_csv(tmp_path: Path) -> None:
    small = _features_frame([(2025, 1), (2025, 2)])
    current = small.loc[small["week"].eq(2)]
    reference, _ = reference_window(small, season=2025, week=2, reference_weeks=1)
    table = feature_drift_table(current, reference)
    report = {"season": 2025, "week": 2, "created_at_utc": "2025-01-01T00:00:00+00:00"}
    output = write_drift_artifacts(report, table, tmp_path / "drift")
    assert output.is_dir()
    written = json.loads((output / "drift_report.json").read_text(encoding="utf-8"))
    assert written["season"] == 2025
    reloaded = pd.read_csv(output / "feature_drift.csv")
    assert not reloaded.empty


# ---------------------------------------------------------------------------
# Weekly-pipeline hook
# ---------------------------------------------------------------------------


def test_weekly_plan_includes_optional_drift_step_after_publish(tmp_path: Path) -> None:
    data_root = _write_weekly_data_root(tmp_path)
    steps = plan_weekly_run(season=2026, week=1, data_root=data_root, skip_prospective=True)
    names = [step.name for step in steps]
    assert names[-1] == "drift-report"
    drift_step = steps[-1]
    assert drift_step.number == 13
    assert drift_step.optional is True
    assert drift_step.skipped is False
    assert drift_step.command[0] == "drift-report"
    assert "--feature-profile" in drift_step.command
    # Read-only monitoring sits strictly after the publish, never on its path.
    assert names.index("publish-predictions") < names.index("drift-report")


def test_weekly_plan_can_skip_drift(tmp_path: Path) -> None:
    data_root = _write_weekly_data_root(tmp_path)
    steps = plan_weekly_run(
        season=2026, week=1, data_root=data_root, skip_prospective=True, skip_drift=True
    )
    assert all(step.name != "drift-report" for step in steps)


def _write_weekly_data_root(tmp_path: Path) -> Path:
    """Minimal data root so plan_weekly_run can resolve production manifests."""

    from nfl_ats.io import atomic_json

    raw = tmp_path / "raw" / "20260812T130036Z"
    atomic_json(
        {
            "snapshot_id": "20260812T130036Z",
            "seasons": list(range(2009, 2027)),
            "team_stat_seasons": list(range(2009, 2026)),
        },
        raw / "manifest.json",
    )
    (raw / "schedules.parquet").write_bytes(b"")
    processed = tmp_path / "processed"
    atomic_json(
        {"source_pbp_snapshot": "20260812T142851Z"},
        processed / "game_features_pbp.manifest.json",
    )
    atomic_json(
        {
            "source_pbp_snapshot": "20260812T142851Z",
            "source_player_snapshot": "20260812T200527Z",
            "source_player_value_snapshot": "20260813T121050Z",
        },
        processed / "game_features_player.manifest.json",
    )
    return tmp_path


def test_cli_parses_drift_report_command() -> None:
    args = build_parser().parse_args(["drift-report", "--season", "2026", "--week", "3"])
    assert args.season == 2026
    assert args.week == 3
    assert args.feature_profile == "player"
    assert args.probability_method == "gaussian"


def test_registered_columns_come_from_the_registry() -> None:
    registered = set(registered_feature_columns())
    expected = {column for columns in FEATURE_FAMILIES.values() for column in columns}
    assert registered == expected
