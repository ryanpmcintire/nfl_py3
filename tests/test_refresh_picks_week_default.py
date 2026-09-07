"""`refresh-picks` defaults --season/--week to the active model's linked forecast.

Measured 2026-09-06 (data/scheduler_log.txt): every scheduled Sunday refresh
pass failed on the argparse usage line because the pair was required and the
schedule never passed it. The week a refresh operates on is not a choice --
it is the week ``publish-predictions`` locked, recorded on the active
manifest -- so the CLI reads it from there when the flags are omitted.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pytest

from nfl_ats.active_model import ACTIVE_ATS_MODEL_VERSION, active_forecast_season_week
from nfl_ats.cli_common import (
    _add_active_forecast_season_week_args,
    _resolve_active_forecast_season_week,
)
from nfl_ats.io import atomic_json


def _write_manifest(artifacts: Path, *, forecast: dict[str, object] | None) -> None:
    manifest: dict[str, object] = {
        "version": ACTIVE_ATS_MODEL_VERSION,
        "status": "SYNCHRONIZED",
        "method": "market_residual",
        "feature_profile": "weak_stack",
        "regressor": "ridge",
        "ridge_alpha": 10.0,
        "probability_method": "gaussian",
        "model_id": "model-1",
    }
    if forecast is not None:
        manifest["weekly_forecast"] = forecast
    atomic_json(manifest, artifacts / "active_ats_model.json")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    _add_active_forecast_season_week_args(parser)
    return parser


def test_active_forecast_season_week_reads_the_linked_forecast(tmp_path: Path) -> None:
    _write_manifest(
        tmp_path,
        forecast={
            "artifact": "margin_predictions/2026-week-03-x",
            "season": 2026,
            "week": 3,
            "game_type": "REG",
        },
    )
    assert active_forecast_season_week(tmp_path) == (2026, 3)


def test_active_forecast_season_week_is_none_without_a_manifest_or_forecast(
    tmp_path: Path,
) -> None:
    assert active_forecast_season_week(tmp_path) is None
    _write_manifest(tmp_path, forecast=None)
    assert active_forecast_season_week(tmp_path) is None


def test_omitting_both_flags_resolves_to_the_active_forecast(tmp_path: Path) -> None:
    _write_manifest(tmp_path, forecast={"artifact": "x", "season": 2026, "week": 7})
    args = _parser().parse_args([])
    assert (args.season, args.week) == (None, None)
    assert _resolve_active_forecast_season_week(args, tmp_path) == (2026, 7)


def test_explicit_flags_win_over_the_active_forecast(tmp_path: Path) -> None:
    _write_manifest(tmp_path, forecast={"artifact": "x", "season": 2026, "week": 7})
    args = _parser().parse_args(["--season", "2025", "--week", "12"])
    assert _resolve_active_forecast_season_week(args, tmp_path) == (2025, 12)


@pytest.mark.parametrize("argv", [["--season", "2026"], ["--week", "2"]])
def test_half_a_pair_is_rejected_with_a_named_reason(tmp_path: Path, argv: list[str]) -> None:
    _write_manifest(tmp_path, forecast={"artifact": "x", "season": 2026, "week": 7})
    args = _parser().parse_args(argv)
    with pytest.raises(ValueError, match="both --season and --week"):
        _resolve_active_forecast_season_week(args, tmp_path)


def test_no_flags_and_no_forecast_names_the_missing_manifest(tmp_path: Path) -> None:
    args = _parser().parse_args([])
    with pytest.raises(ValueError, match=r"active_ats_model.json"):
        _resolve_active_forecast_season_week(args, tmp_path)
