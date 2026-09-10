from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

from nfl_ats.backup_qb_fade_overlay import record_backup_qb_fade_challenger_decisions
from nfl_ats.best_pick_big_spread_challenger import (
    record_big_spread_nomination_challenger_decisions,
)
from nfl_ats.best_pick_nomination import (
    record_nomination_challenger_decisions,
    record_nomination_v3_challenger_decisions,
)
from nfl_ats.best_pick_refresh_prospective import record_best_pick_refresh, record_best_pick_tuesday
from nfl_ats.board_content import verify_number_provenance
from nfl_ats.board_site import build_site
from nfl_ats.bye_edge_fade_overlay import record_bye_edge_fade_challenger_decisions
from nfl_ats.cli_common import (
    _add_active_forecast_season_week_args,
    _add_board_destination_args,
    _artifacts_root,
    _data_root,
    _print_json,
    _registry_root,
    _resolve_active_forecast_season_week,
)
from nfl_ats.clv import record_paper_decisions
from nfl_ats.coach_fade_overlay import record_overlay_challenger_decisions
from nfl_ats.consensus_movement_refresh_overlay import (
    record_consensus_movement_refresh_overlay,
)
from nfl_ats.constants import DEFAULT_MIN_TRAIN_GAMES
from nfl_ats.crew_tilt_refresh_overlay import record_crew_tilt_refresh_overlay
from nfl_ats.deadline_drag_challenger import record_deadline_drag_challenger_decisions
from nfl_ats.division_revenge_tilt_overlay import record_division_revenge_tilt_challenger_decisions
from nfl_ats.ecdf_mapping_incumbent_overlay import (
    record_ecdf_mapping_incumbent_challenger_decisions,
)
from nfl_ats.era_weighted_half_life_8_overlay import (
    record_era_weighted_half_life_8_challenger_decisions,
)
from nfl_ats.expected_lineup_loss_challenger import record_expected_lineup_loss_challenger_decisions
from nfl_ats.forecast_cold_visitor_tilt_overlay import (
    record_forecast_cold_visitor_tilt_challenger_decisions,
)
from nfl_ats.forecast_weather_kn_precip_high_total_tilt_overlay import (
    record_forecast_weather_kn_precip_high_total_tilt_challenger_decisions,
)
from nfl_ats.forecast_weather_kn_warm_team_cold_late_tilt_overlay import (
    fetch_shared_kickoff_nearest_forecasts_fail_open,
    record_forecast_weather_kn_warm_team_cold_late_tilt_challenger_decisions,
)
from nfl_ats.four_overlay_incumbent import record_former_production_incumbent_decisions
from nfl_ats.gaussian_mean_mapping_incumbent_overlay import (
    record_gaussian_mean_mapping_incumbent_challenger_decisions,
)
from nfl_ats.handle_follow_refresh_overlay import record_handle_follow_refresh_overlay
from nfl_ats.home_side_offset_incumbent_overlay import (
    record_home_side_offset_incumbent_challenger_decisions,
)
from nfl_ats.inactives_refresh_overlay import record_inactives_refresh_overlay
from nfl_ats.injury_signal_refresh_tilt import record_injury_signal_refresh_tilt
from nfl_ats.injury_value_tilt_overlay import record_injury_value_tilt_challenger_decisions
from nfl_ats.interim_hc_first_game_tilt_overlay import (
    record_interim_hc_first_game_tilt_challenger_decisions,
)
from nfl_ats.io import atomic_text
from nfl_ats.key_line_pick_read_incumbent_overlay import (
    record_key_line_pick_read_incumbent_challenger_decisions,
)
from nfl_ats.late_week_move_follow_refresh_overlay import (
    record_late_week_move_follow_refresh_overlay,
)
from nfl_ats.low_total_div_home_dog_challenger import (
    record_low_total_div_home_dog_challenger_decisions,
)
from nfl_ats.nflcom_refresh_overlay import record_nflcom_refresh_overlay
from nfl_ats.pace_mismatch_dog_tilt_overlay import (
    record_pace_mismatch_dog_tilt_challenger_decisions,
)
from nfl_ats.pbp08_protection_mismatch_tilt_overlay import (
    record_pbp08_protection_mismatch_tilt_challenger_decisions,
)
from nfl_ats.pick_refresh import append_refresh_to_card, plan_refresh, record_plan, refresh_summary
from nfl_ats.prospective import (
    record_movement_rule_composed_challenger_decisions,
    record_nflcom_refresh_out2_starters_challenger_decisions,
)
from nfl_ats.publishing import publish_active_predictions
from nfl_ats.qb_revenge_deadline_drag_stack_challenger import (
    record_qb_revenge_deadline_drag_stack_challenger_decisions,
)
from nfl_ats.rain_on_grass_dog_challenger import record_rain_on_grass_dog_challenger_decisions
from nfl_ats.retired_four_member_union import record_retired_four_member_union_decisions
from nfl_ats.retired_three_member_union import record_retired_three_member_union_decisions
from nfl_ats.served_total_challenger import record_totals_served_method_decisions
from nfl_ats.special_teams_return_tilt_overlay import (
    record_special_teams_return_tilt_challenger_decisions,
)
from nfl_ats.specialist_absence_fade_refresh_overlay import (
    record_specialist_absence_fade_refresh_overlay,
)
from nfl_ats.spread_gap_zone_fade_overlay import record_spread_gap_zone_fade_challenger_decisions
from nfl_ats.surface_switch_tilt_overlay import record_surface_switch_tilt_challenger_decisions
from nfl_ats.tank_zone_fade_tilt_overlay import record_tank_zone_fade_tilt_challenger_decisions
from nfl_ats.third_down_reversion_fade_overlay import (
    record_third_down_reversion_fade_challenger_decisions,
)
from nfl_ats.tiebreaker_shade_prospective import record_tiebreaker_shade_decisions
from nfl_ats.turnover_luck_rebound_tilt_overlay import (
    record_turnover_luck_rebound_tilt_challenger_decisions,
)

PUBLISH_CHALLENGER_RESULT_KEYS: dict[str, str] = {
    "tiebreaker_low_side_shade": "tiebreaker_shade_ledger",
    "weak_stack_deadline_drag": "deadline_drag_challenger_ledger",
    "weak_stack_expected_lineup_loss": "expected_lineup_loss_challenger_ledger",
    "hc_year_one_fade_overlay": "overlay_challenger_ledger",
    "bye_edge_fade_overlay": "bye_edge_fade_challenger_ledger",
    "best_pick_nomination_v2": "nomination_challenger_ledger",
    "best_pick_nomination_v3": "nomination_v3_challenger_ledger",
    "best_pick_big_spread_eligibility": "big_spread_nomination_challenger_ledger",
    "injury_value_lost_tilt_overlay": "injury_value_tilt_challenger_ledger",
    "division_revenge_tilt_overlay": "division_revenge_tilt_challenger_ledger",
    "surface_switch_tilt_overlay": "surface_switch_tilt_challenger_ledger",
    "overlay_four_member_union_retired_20260907": "retired_four_member_union_challenger_ledger",
    "overlay_three_member_union_retired_20260909": "retired_three_member_union_challenger_ledger",
    "spread_gap_zone_fade_overlay": "spread_gap_zone_fade_challenger_ledger",
    "overlay_production_chain_coach_arrest_incumbent": ("four_overlay_incumbent_challenger_ledger"),
    "ecdf_mapping_incumbent": "ecdf_mapping_incumbent_challenger_ledger",
    "gaussian_mean_mapping_incumbent": "gaussian_mean_mapping_incumbent_challenger_ledger",
    "home_side_offset_off_incumbent": "home_side_offset_off_incumbent_challenger_ledger",
    "key_line_pick_read_off_incumbent": "key_line_pick_read_off_incumbent_challenger_ledger",
    "era_weighted_half_life_8": "era_weighted_half_life_8_challenger_ledger",
    "forecast_cold_visitor_tilt": "forecast_cold_visitor_tilt_challenger_ledger",
    "interim_hc_first_game_tilt_overlay": "interim_hc_first_game_tilt_challenger_ledger",
    "forecast_weather_kn_warm_team_cold_late_tilt": (
        "forecast_weather_kn_warm_team_cold_late_tilt_challenger_ledger"
    ),
    "forecast_weather_kn_precip_high_total_tilt": (
        "forecast_weather_kn_precip_high_total_tilt_challenger_ledger"
    ),
    "movement_rule_composed_v1": "movement_rule_composed_challenger_ledger",
    "nflcom_friday_refresh_out2_starters_v1": "nflcom_refresh_out2_starters_challenger_ledger",
    "pbp08_protection_mismatch_tilt_overlay": ("pbp08_protection_mismatch_tilt_challenger_ledger"),
    "tank_zone_fade_tilt_overlay": "tank_zone_fade_tilt_challenger_ledger",
    "third_down_reversion_fade_overlay": ("third_down_reversion_fade_challenger_ledger"),
    "turnover_luck_rebound_tilt_overlay": ("turnover_luck_rebound_tilt_challenger_ledger"),
    "special_teams_return_tilt_overlay": ("special_teams_return_tilt_challenger_ledger"),
    "pace_mismatch_dog_tilt_overlay": "pace_mismatch_dog_tilt_challenger_ledger",
    "weak_stack_qb_revenge_deadline_drag": "qb_revenge_deadline_drag_stack_challenger_ledger",
    "totals_served_method": "totals_served_method_challenger_ledger",
    "low_total_div_home_dog_challenger": "low_total_div_home_dog_challenger_ledger",
    "rain_on_grass_dog_challenger": "rain_on_grass_dog_challenger_ledger",
}

REFRESH_CHALLENGER_RESULT_KEYS: dict[str, str] = {
    "best_pick_sunday_renomination": "best_pick_refresh_ledger",
    "model_only_refresh_incumbent": "ledger",
    "injury_signal_refresh_tilt": "injury_signal_refresh_tilt",
    "nflcom_friday_refresh_out2_starters_v1": "nflcom_refresh_out2_starters_overlay",
    "inactives_refresh_v1": "inactives_refresh_overlay",
    "crew_tilt_refresh_v1": "crew_tilt_refresh_overlay",
    "specialist_absence_fade_refresh_v1": "specialist_absence_fade_refresh_overlay",
    "late_week_move_follow_refresh_v1": "late_week_move_follow_refresh_overlay",
    "late_week_leader_median_follow_v1": "late_week_move_follow_refresh_overlay",
    "late_week_leader_median_follow_0_5_off_incumbent": "late_week_move_follow_refresh_overlay",
    "late_week_leader_median_follow_flat_1_0_off_incumbent": (
        "late_week_move_follow_refresh_overlay"
    ),
    "late_week_follow_no_news_veto_off_incumbent": "late_week_move_follow_refresh_overlay",
    "handle_follow_refresh_off_incumbent": "handle_follow_refresh_overlay",
    "rookie_crew_underdog_off_incumbent": "ledger",
    "consensus_movement_1_0_off_incumbent": "consensus_movement_refresh_overlay",
}


REPLACE_WEEK_FROZEN_ARMS: dict[str, str] = {
    "low_total_div_home_dog_challenger": (
        "recorded 2026-09-05 from its own card 2026-week-01-20260905T141453Z; --replace-week "
        "leaves it on that card"
    ),
    "rain_on_grass_dog_challenger": (
        "recorded 2026-09-05 from its own card 2026-week-01-20260905T141453Z on that day's "
        "live weather read; --replace-week leaves it on that card"
    ),
    "weak_stack_qb_revenge_deadline_drag": (
        "recorded 2026-09-05 from its own card 2026-week-01-20260905T141453Z; --replace-week "
        "leaves it on that card"
    ),
}


def replace_week_for(request: PublishPredictionsRequest, challenger_id: str) -> bool:

    return request.replace_week and challenger_id not in REPLACE_WEEK_FROZEN_ARMS


def collect_replacement_report(result: dict[str, Any]) -> dict[str, Any]:

    keys = set(PUBLISH_CHALLENGER_RESULT_KEYS.values())
    keys.update({"clv_ledger", "best_pick_tuesday_ledger"})
    ledgers: dict[str, dict[str, int]] = {}
    for result_key in sorted(keys):
        entry = result.get(result_key)
        if not isinstance(entry, dict):
            continue
        replaced = int(entry.get("replaced_rows", 0) or 0)
        left = int(entry.get("left_post_kickoff", 0) or 0)
        if replaced or left:
            ledgers[result_key] = {"replaced": replaced, "left_post_kickoff": left}
    return {
        "ledgers": ledgers,
        "total_replaced": sum(row["replaced"] for row in ledgers.values()),
        "frozen_arms_left_alone": sorted(REPLACE_WEEK_FROZEN_ARMS),
    }


def collect_failed_recorders(
    result: dict[str, Any], result_keys: dict[str, str]
) -> list[dict[str, str]]:

    failures: list[dict[str, str]] = []
    for challenger_id, result_key in result_keys.items():
        entry = result.get(result_key)
        if not isinstance(entry, dict):
            continue
        error = entry.get("error")
        if not error:
            continue
        failures.append(
            {"challenger_id": challenger_id, "result_key": result_key, "error": str(error)}
        )
    return sorted(failures, key=lambda failure: failure["challenger_id"])


def _site_directory(destination: Path) -> Path:

    return destination.parent if destination.suffix else destination


def _write_public_site(destination: Path) -> dict[str, Any]:

    directory = _site_directory(destination)
    verify_number_provenance(_artifacts_root())
    pages = build_site(_artifacts_root(), require_fresh_arrest_overlay=True)
    written = []
    for relative_path, html in pages.items():
        path = directory / relative_path
        atomic_text(html, path)
        written.append(str(path))
    nojekyll = directory / ".nojekyll"
    if not nojekyll.is_file():
        atomic_text("", nojekyll)
    return {
        "site_destination": str(directory),
        "pages_written": written,
        "board_destination": str(directory / "index.html"),
        "nojekyll": str(nojekyll),
    }


@dataclass(frozen=True)
class PublishPredictionsRequest:
    destination: Path
    readme: Path
    with_board: bool
    site_destination: Path | None
    board_destination: Path | None
    record_decisions: bool
    record_from_forecast: str | None = None
    replace_week: bool = False


def parse_publish_predictions_request(args: argparse.Namespace) -> PublishPredictionsRequest:

    return PublishPredictionsRequest(
        destination=args.destination,
        readme=args.readme,
        with_board=bool(args.with_board),
        site_destination=args.site_destination,
        board_destination=args.board_destination,
        record_decisions=bool(args.record_decisions),
        record_from_forecast=getattr(args, "record_from_forecast", None),
        replace_week=bool(getattr(args, "replace_week", False)),
    )


def orchestrate_publish_predictions(request: PublishPredictionsRequest) -> dict[str, Any]:

    publish_instant = datetime.now(UTC)
    verify_number_provenance(_artifacts_root())
    result = publish_active_predictions(
        _artifacts_root(),
        destination=request.destination,
        readme_path=request.readme,
        data_root=_data_root(),
        published_at=publish_instant,
        registry_root=_registry_root(),
    )
    if request.with_board:
        try:
            site_destination = cast(Path, request.site_destination or request.board_destination)
            result.update(_write_public_site(site_destination))
        except (ValueError, FileNotFoundError) as error:
            result["public_site"] = {"written": False, "error": str(error)}
    if request.record_decisions:
        try:
            result["clv_ledger"] = record_paper_decisions(
                _artifacts_root(),
                data_root=_data_root(),
                now=publish_instant,
                require_fresh_arrest_overlay=True,
                forecast_artifact=request.record_from_forecast,
                replace_week=request.replace_week,
            )
        except Exception as error:
            result["clv_ledger"] = {"recorded": 0, "error": str(error)}
        try:
            result["best_pick_tuesday_ledger"] = record_best_pick_tuesday(
                _artifacts_root(),
                _data_root(),
                result,
                now=publish_instant,
                replace_week=request.replace_week,
            )
        except Exception as error:
            result["best_pick_tuesday_ledger"] = {"recorded": 0, "error": str(error)}
        try:
            result["tiebreaker_shade_ledger"] = record_tiebreaker_shade_decisions(
                _artifacts_root(),
                _data_root(),
                published_path=(
                    Path(result["tiebreaker_json_path"])
                    if result.get("tiebreaker_json_path")
                    else None
                ),
                now=publish_instant,
                forecast_artifact=request.record_from_forecast,
                replace_week=request.replace_week,
            )
        except Exception as error:
            result["tiebreaker_shade_ledger"] = {"recorded": 0, "error": str(error)}
        try:
            result["overlay_challenger_ledger"] = record_overlay_challenger_decisions(
                _artifacts_root(),
                _data_root(),
                forecast_artifact=request.record_from_forecast,
                replace_week=request.replace_week,
            )
        except Exception as error:
            result["overlay_challenger_ledger"] = {"recorded": 0, "error": str(error)}
        try:
            result["nomination_challenger_ledger"] = record_nomination_challenger_decisions(
                _artifacts_root(),
                _data_root(),
                forecast_artifact=request.record_from_forecast,
                replace_week=request.replace_week,
            )
        except Exception as error:
            result["nomination_challenger_ledger"] = {"recorded": 0, "error": str(error)}
        try:
            result["nomination_v3_challenger_ledger"] = record_nomination_v3_challenger_decisions(
                _artifacts_root(),
                _data_root(),
                forecast_artifact=request.record_from_forecast,
                replace_week=request.replace_week,
            )
        except Exception as error:
            result["nomination_v3_challenger_ledger"] = {"recorded": 0, "error": str(error)}
        try:
            result["big_spread_nomination_challenger_ledger"] = (
                record_big_spread_nomination_challenger_decisions(
                    _artifacts_root(),
                    _data_root(),
                    forecast_artifact=request.record_from_forecast,
                    replace_week=request.replace_week,
                )
            )
        except Exception as error:
            result["big_spread_nomination_challenger_ledger"] = {
                "recorded": 0,
                "error": str(error),
            }
        try:
            result["injury_value_tilt_challenger_ledger"] = (
                record_injury_value_tilt_challenger_decisions(
                    _artifacts_root(),
                    _data_root(),
                    forecast_artifact=request.record_from_forecast,
                    replace_week=request.replace_week,
                )
            )
        except Exception as error:
            result["injury_value_tilt_challenger_ledger"] = {"recorded": 0, "error": str(error)}
        try:
            result["division_revenge_tilt_challenger_ledger"] = (
                record_division_revenge_tilt_challenger_decisions(
                    _artifacts_root(),
                    _data_root(),
                    forecast_artifact=request.record_from_forecast,
                    replace_week=request.replace_week,
                )
            )
        except Exception as error:
            result["division_revenge_tilt_challenger_ledger"] = {
                "recorded": 0,
                "error": str(error),
            }
        try:
            result["backup_qb_fade_challenger_ledger"] = record_backup_qb_fade_challenger_decisions(
                _artifacts_root(),
                _data_root(),
                forecast_artifact=request.record_from_forecast,
                replace_week=request.replace_week,
            )
        except Exception as error:
            result["backup_qb_fade_challenger_ledger"] = {"recorded": 0, "error": str(error)}
        try:
            result["surface_switch_tilt_challenger_ledger"] = (
                record_surface_switch_tilt_challenger_decisions(
                    _artifacts_root(),
                    _data_root(),
                    forecast_artifact=request.record_from_forecast,
                    replace_week=request.replace_week,
                )
            )
        except Exception as error:
            result["surface_switch_tilt_challenger_ledger"] = {"recorded": 0, "error": str(error)}
        try:
            result["spread_gap_zone_fade_challenger_ledger"] = (
                record_spread_gap_zone_fade_challenger_decisions(
                    _artifacts_root(),
                    _data_root(),
                    forecast_artifact=request.record_from_forecast,
                    replace_week=request.replace_week,
                )
            )
        except Exception as error:
            result["spread_gap_zone_fade_challenger_ledger"] = {
                "recorded": 0,
                "error": str(error),
            }
        try:
            result["expected_lineup_loss_challenger_ledger"] = (
                record_expected_lineup_loss_challenger_decisions(
                    _artifacts_root(),
                    _data_root(),
                    forecast_artifact=request.record_from_forecast,
                    replace_week=request.replace_week,
                )
            )
        except Exception as error:
            result["expected_lineup_loss_challenger_ledger"] = {"recorded": 0, "error": str(error)}
        try:
            result["deadline_drag_challenger_ledger"] = record_deadline_drag_challenger_decisions(
                _artifacts_root(),
                _data_root(),
                forecast_artifact=request.record_from_forecast,
                replace_week=request.replace_week,
            )
        except Exception as error:
            result["deadline_drag_challenger_ledger"] = {"recorded": 0, "error": str(error)}
        try:
            result["low_total_div_home_dog_challenger_ledger"] = (
                record_low_total_div_home_dog_challenger_decisions(
                    _artifacts_root(),
                    _data_root(),
                    forecast_artifact=request.record_from_forecast,
                    replace_week=replace_week_for(request, "low_total_div_home_dog_challenger"),
                )
            )
        except Exception as error:
            result["low_total_div_home_dog_challenger_ledger"] = {
                "recorded": 0,
                "error": str(error),
            }
        try:
            result["bye_edge_fade_challenger_ledger"] = record_bye_edge_fade_challenger_decisions(
                _artifacts_root(),
                _data_root(),
                forecast_artifact=request.record_from_forecast,
                replace_week=request.replace_week,
            )
        except Exception as error:
            result["bye_edge_fade_challenger_ledger"] = {"recorded": 0, "error": str(error)}
        try:
            result["tank_zone_fade_tilt_challenger_ledger"] = (
                record_tank_zone_fade_tilt_challenger_decisions(
                    _artifacts_root(),
                    _data_root(),
                    forecast_artifact=request.record_from_forecast,
                    replace_week=request.replace_week,
                )
            )
        except Exception as error:
            result["tank_zone_fade_tilt_challenger_ledger"] = {
                "recorded": 0,
                "error": str(error),
            }
        try:
            result["third_down_reversion_fade_challenger_ledger"] = (
                record_third_down_reversion_fade_challenger_decisions(
                    _artifacts_root(),
                    _data_root(),
                    forecast_artifact=request.record_from_forecast,
                    replace_week=request.replace_week,
                )
            )
        except Exception as error:
            result["third_down_reversion_fade_challenger_ledger"] = {
                "recorded": 0,
                "error": str(error),
            }
        try:
            result["turnover_luck_rebound_tilt_challenger_ledger"] = (
                record_turnover_luck_rebound_tilt_challenger_decisions(
                    _artifacts_root(),
                    _data_root(),
                    forecast_artifact=request.record_from_forecast,
                    replace_week=request.replace_week,
                )
            )
        except Exception as error:
            result["turnover_luck_rebound_tilt_challenger_ledger"] = {
                "recorded": 0,
                "error": str(error),
            }
        try:
            result["special_teams_return_tilt_challenger_ledger"] = (
                record_special_teams_return_tilt_challenger_decisions(
                    _artifacts_root(),
                    _data_root(),
                    forecast_artifact=request.record_from_forecast,
                    replace_week=request.replace_week,
                )
            )
        except Exception as error:
            result["special_teams_return_tilt_challenger_ledger"] = {
                "recorded": 0,
                "error": str(error),
            }
        try:
            result["pace_mismatch_dog_tilt_challenger_ledger"] = (
                record_pace_mismatch_dog_tilt_challenger_decisions(
                    _artifacts_root(),
                    _data_root(),
                    forecast_artifact=request.record_from_forecast,
                    replace_week=request.replace_week,
                )
            )
        except Exception as error:
            result["pace_mismatch_dog_tilt_challenger_ledger"] = {
                "recorded": 0,
                "error": str(error),
            }
        try:
            result["pbp08_protection_mismatch_tilt_challenger_ledger"] = (
                record_pbp08_protection_mismatch_tilt_challenger_decisions(
                    _artifacts_root(),
                    _data_root(),
                    now=publish_instant,
                    forecast_artifact=request.record_from_forecast,
                    replace_week=request.replace_week,
                )
            )
        except Exception as error:
            result["pbp08_protection_mismatch_tilt_challenger_ledger"] = {
                "recorded": 0,
                "error": str(error),
            }
        try:
            result["four_overlay_incumbent_challenger_ledger"] = (
                record_former_production_incumbent_decisions(
                    _artifacts_root(),
                    _data_root(),
                    now=publish_instant,
                    forecast_artifact=request.record_from_forecast,
                    replace_week=request.replace_week,
                )
            )
        except Exception as error:
            result["four_overlay_incumbent_challenger_ledger"] = {
                "recorded": 0,
                "error": str(error),
            }
        try:
            result["retired_four_member_union_challenger_ledger"] = (
                record_retired_four_member_union_decisions(
                    _artifacts_root(),
                    _data_root(),
                    now=publish_instant,
                    forecast_artifact=request.record_from_forecast,
                    replace_week=request.replace_week,
                )
            )
        except Exception as error:
            result["retired_four_member_union_challenger_ledger"] = {
                "recorded": 0,
                "error": str(error),
            }
        try:
            result["retired_three_member_union_challenger_ledger"] = (
                record_retired_three_member_union_decisions(
                    _artifacts_root(),
                    _data_root(),
                    now=publish_instant,
                    forecast_artifact=request.record_from_forecast,
                    replace_week=request.replace_week,
                )
            )
        except Exception as error:
            result["retired_three_member_union_challenger_ledger"] = {
                "recorded": 0,
                "error": str(error),
            }
        try:
            result["ecdf_mapping_incumbent_challenger_ledger"] = (
                record_ecdf_mapping_incumbent_challenger_decisions(
                    _artifacts_root(),
                    _data_root(),
                    forecast_artifact=request.record_from_forecast,
                    replace_week=request.replace_week,
                )
            )
        except Exception as error:
            result["ecdf_mapping_incumbent_challenger_ledger"] = {
                "recorded": 0,
                "error": str(error),
            }
        try:
            result["gaussian_mean_mapping_incumbent_challenger_ledger"] = (
                record_gaussian_mean_mapping_incumbent_challenger_decisions(
                    _artifacts_root(),
                    _data_root(),
                    forecast_artifact=request.record_from_forecast,
                    replace_week=request.replace_week,
                )
            )
        except Exception as error:
            result["gaussian_mean_mapping_incumbent_challenger_ledger"] = {
                "recorded": 0,
                "error": str(error),
            }
        try:
            result["home_side_offset_off_incumbent_challenger_ledger"] = (
                record_home_side_offset_incumbent_challenger_decisions(
                    _artifacts_root(),
                    _data_root(),
                    forecast_artifact=request.record_from_forecast,
                    replace_week=request.replace_week,
                )
            )
        except Exception as error:
            result["home_side_offset_off_incumbent_challenger_ledger"] = {
                "recorded": 0,
                "error": str(error),
            }
        try:
            result["key_line_pick_read_off_incumbent_challenger_ledger"] = (
                record_key_line_pick_read_incumbent_challenger_decisions(
                    _artifacts_root(),
                    _data_root(),
                    forecast_artifact=request.record_from_forecast,
                    replace_week=request.replace_week,
                )
            )
        except Exception as error:
            result["key_line_pick_read_off_incumbent_challenger_ledger"] = {
                "recorded": 0,
                "error": str(error),
            }
        try:
            result["era_weighted_half_life_8_challenger_ledger"] = (
                record_era_weighted_half_life_8_challenger_decisions(
                    _artifacts_root(),
                    _data_root(),
                    forecast_artifact=request.record_from_forecast,
                    replace_week=request.replace_week,
                )
            )
        except Exception as error:
            result["era_weighted_half_life_8_challenger_ledger"] = {
                "recorded": 0,
                "error": str(error),
            }
        try:
            result["forecast_cold_visitor_tilt_challenger_ledger"] = (
                record_forecast_cold_visitor_tilt_challenger_decisions(
                    _artifacts_root(),
                    _data_root(),
                    _registry_root(),
                    forecast_artifact=request.record_from_forecast,
                    replace_week=request.replace_week,
                )
            )
        except Exception as error:
            result["forecast_cold_visitor_tilt_challenger_ledger"] = {
                "recorded": 0,
                "error": str(error),
            }
        try:
            result["interim_hc_first_game_tilt_challenger_ledger"] = (
                record_interim_hc_first_game_tilt_challenger_decisions(
                    _artifacts_root(),
                    _data_root(),
                    forecast_artifact=request.record_from_forecast,
                    replace_week=request.replace_week,
                )
            )
        except Exception as error:
            result["interim_hc_first_game_tilt_challenger_ledger"] = {
                "recorded": 0,
                "error": str(error),
            }
        shared_kn_forecasts = fetch_shared_kickoff_nearest_forecasts_fail_open(
            _artifacts_root(),
            _data_root(),
            _registry_root(),
            forecast_artifact=request.record_from_forecast,
        )
        try:
            result["forecast_weather_kn_warm_team_cold_late_tilt_challenger_ledger"] = (
                record_forecast_weather_kn_warm_team_cold_late_tilt_challenger_decisions(
                    _artifacts_root(),
                    _data_root(),
                    _registry_root(),
                    forecasts=shared_kn_forecasts,
                    forecast_artifact=request.record_from_forecast,
                    replace_week=request.replace_week,
                )
            )
        except Exception as error:
            result["forecast_weather_kn_warm_team_cold_late_tilt_challenger_ledger"] = {
                "recorded": 0,
                "error": str(error),
            }
        try:
            result["forecast_weather_kn_precip_high_total_tilt_challenger_ledger"] = (
                record_forecast_weather_kn_precip_high_total_tilt_challenger_decisions(
                    _artifacts_root(),
                    _data_root(),
                    _registry_root(),
                    forecasts=shared_kn_forecasts,
                    forecast_artifact=request.record_from_forecast,
                    replace_week=request.replace_week,
                )
            )
        except Exception as error:
            result["forecast_weather_kn_precip_high_total_tilt_challenger_ledger"] = {
                "recorded": 0,
                "error": str(error),
            }
        try:
            result["rain_on_grass_dog_challenger_ledger"] = (
                record_rain_on_grass_dog_challenger_decisions(
                    _artifacts_root(),
                    _data_root(),
                    _registry_root(),
                    forecasts=shared_kn_forecasts,
                    forecast_artifact=request.record_from_forecast,
                    replace_week=replace_week_for(request, "rain_on_grass_dog_challenger"),
                )
            )
        except Exception as error:
            result["rain_on_grass_dog_challenger_ledger"] = {
                "recorded": 0,
                "error": str(error),
            }
        try:
            result["movement_rule_composed_challenger_ledger"] = (
                record_movement_rule_composed_challenger_decisions(
                    _artifacts_root(),
                    _data_root(),
                    now=publish_instant,
                    forecast_artifact=request.record_from_forecast,
                    replace_week=request.replace_week,
                )
            )
        except Exception as error:
            result["movement_rule_composed_challenger_ledger"] = {
                "recorded": 0,
                "error": str(error),
            }
        try:
            result["nflcom_refresh_out2_starters_challenger_ledger"] = (
                record_nflcom_refresh_out2_starters_challenger_decisions(
                    _artifacts_root(),
                    _data_root(),
                    now=publish_instant,
                    forecast_artifact=request.record_from_forecast,
                    replace_week=request.replace_week,
                )
            )
        except Exception as error:
            result["nflcom_refresh_out2_starters_challenger_ledger"] = {
                "recorded": 0,
                "error": str(error),
            }
        try:
            result["qb_revenge_deadline_drag_stack_challenger_ledger"] = (
                record_qb_revenge_deadline_drag_stack_challenger_decisions(
                    _artifacts_root(),
                    _data_root(),
                    now=publish_instant,
                    forecast_artifact=request.record_from_forecast,
                    replace_week=replace_week_for(request, "weak_stack_qb_revenge_deadline_drag"),
                )
            )
        except Exception as error:
            result["qb_revenge_deadline_drag_stack_challenger_ledger"] = {
                "recorded": 0,
                "error": str(error),
            }
        try:
            result["totals_served_method_challenger_ledger"] = (
                record_totals_served_method_decisions(
                    _artifacts_root(),
                    _data_root(),
                    now=publish_instant,
                    replace_week=request.replace_week,
                )
            )
        except Exception as error:
            result["totals_served_method_challenger_ledger"] = {
                "recorded": 0,
                "error": str(error),
            }
        result["failed_recorders"] = collect_failed_recorders(
            result, PUBLISH_CHALLENGER_RESULT_KEYS
        )
        if request.replace_week:
            result["replaced_week"] = collect_replacement_report(result)
    else:
        result["best_pick_tuesday_ledger"] = {
            "recorded": 0,
            "skipped": True,
            "reason": "pass --record-decisions",
        }
        result["tiebreaker_shade_ledger"] = {
            "recorded": 0,
            "skipped": True,
            "reason": "pass --record-decisions",
        }
        result["clv_ledger"] = {
            "recorded": 0,
            "skipped": True,
            "reason": "pass --record-decisions to append this card's picks to the "
            "paper-decision ledger",
        }
        result["overlay_challenger_ledger"] = {
            "recorded": 0,
            "skipped": True,
            "reason": "pass --record-decisions to append the overlay's picks to the "
            "prospective challenger ledger",
        }
        result["nomination_challenger_ledger"] = {
            "recorded": 0,
            "skipped": True,
            "reason": "pass --record-decisions to append the v2 Best Pick nomination to "
            "the prospective challenger ledger",
        }
        result["nomination_v3_challenger_ledger"] = {
            "recorded": 0,
            "skipped": True,
            "reason": "pass --record-decisions to append the v3 Best Pick nomination to "
            "the prospective challenger ledger",
        }
        result["big_spread_nomination_challenger_ledger"] = {
            "recorded": 0,
            "skipped": True,
            "reason": "pass --record-decisions to append the big-spread-screened "
            "Best Pick nomination to the prospective challenger ledger",
        }
        result["injury_value_tilt_challenger_ledger"] = {
            "recorded": 0,
            "skipped": True,
            "reason": "pass --record-decisions to append the injury value-lost tilt's "
            "picks to the prospective challenger ledger",
        }
        result["division_revenge_tilt_challenger_ledger"] = {
            "recorded": 0,
            "skipped": True,
            "reason": "pass --record-decisions to append the division-revenge tilt's "
            "picks to the prospective challenger ledger",
        }
        result["backup_qb_fade_challenger_ledger"] = {
            "recorded": 0,
            "skipped": True,
            "reason": "pass --record-decisions to append the backup-QB fade's picks to "
            "the prospective challenger ledger",
        }
        result["surface_switch_tilt_challenger_ledger"] = {
            "recorded": 0,
            "skipped": True,
            "reason": "pass --record-decisions to append the surface-switch tilt's "
            "picks to the prospective challenger ledger",
        }
        result["spread_gap_zone_fade_challenger_ledger"] = {
            "recorded": 0,
            "skipped": True,
            "reason": "pass --record-decisions to append the spread-gap-zone fade's "
            "picks to the prospective challenger ledger",
        }
        result["expected_lineup_loss_challenger_ledger"] = {
            "recorded": 0,
            "skipped": True,
            "reason": "Decision recording was not requested",
        }
        result["deadline_drag_challenger_ledger"] = {
            "recorded": 0,
            "skipped": True,
            "reason": "Decision recording was not requested",
        }
        result["low_total_div_home_dog_challenger_ledger"] = {
            "recorded": 0,
            "skipped": True,
            "reason": "pass --record-decisions to append the low-total divisional "
            "home-dog challenger's picks to the prospective challenger ledger",
        }
        result["rain_on_grass_dog_challenger_ledger"] = {
            "recorded": 0,
            "skipped": True,
            "reason": "pass --record-decisions to append the rain-on-grass underdog "
            "challenger's picks to the prospective challenger ledger",
        }
        result["bye_edge_fade_challenger_ledger"] = {
            "recorded": 0,
            "skipped": True,
            "reason": "pass --record-decisions to append the bye-edge fade's picks to the "
            "prospective challenger ledger",
        }
        result["tank_zone_fade_tilt_challenger_ledger"] = {
            "recorded": 0,
            "skipped": True,
            "reason": "pass --record-decisions to append the tank-zone fade tilt's picks to the "
            "prospective challenger ledger",
        }
        result["third_down_reversion_fade_challenger_ledger"] = {
            "recorded": 0,
            "skipped": True,
            "reason": "pass --record-decisions to append the third-down reversion fade's "
            "picks to the prospective challenger ledger",
        }
        result["turnover_luck_rebound_tilt_challenger_ledger"] = {
            "recorded": 0,
            "skipped": True,
            "reason": "pass --record-decisions to append the turnover-luck rebound tilt's "
            "picks to the prospective challenger ledger",
        }
        result["special_teams_return_tilt_challenger_ledger"] = {
            "recorded": 0,
            "skipped": True,
            "reason": "pass --record-decisions to append the special-teams return tilt's "
            "picks to the prospective challenger ledger",
        }
        result["pace_mismatch_dog_tilt_challenger_ledger"] = {
            "recorded": 0,
            "skipped": True,
            "reason": "pass --record-decisions to append the pace-mismatch dog tilt's picks to the "
            "prospective challenger ledger",
        }
        result["pbp08_protection_mismatch_tilt_challenger_ledger"] = {
            "recorded": 0,
            "skipped": True,
            "reason": "pass --record-decisions to append the PBP-08 protection-mismatch "
            "tilt's picks to the prospective challenger ledger",
        }
        result["retired_four_member_union_challenger_ledger"] = {
            "recorded": 0,
            "skipped": True,
            "reason": "pass --record-decisions to append the retired four-member card",
        }
        result["retired_three_member_union_challenger_ledger"] = {
            "recorded": 0,
            "skipped": True,
            "reason": "pass --record-decisions to append the retired three-member card",
        }
        result["four_overlay_incumbent_challenger_ledger"] = {
            "recorded": 0,
            "skipped": True,
            "reason": "pass --record-decisions to append the former coach-to-arrests "
            "incumbent's picks to the prospective challenger ledger",
        }
        result["ecdf_mapping_incumbent_challenger_ledger"] = {
            "recorded": 0,
            "skipped": True,
            "reason": "pass --record-decisions to append the ECDF-mapping-incumbent "
            "overlay's picks to the prospective challenger ledger",
        }
        result["gaussian_mean_mapping_incumbent_challenger_ledger"] = {
            "recorded": 0,
            "skipped": True,
            "reason": "pass --record-decisions to append the former mean mapping "
            "overlay's picks to the prospective challenger ledger",
        }
        result["home_side_offset_off_incumbent_challenger_ledger"] = {
            "recorded": 0,
            "skipped": True,
            "reason": "pass --record-decisions to append the uncorrected point read's "
            "picks to the prospective challenger ledger",
        }
        result["key_line_pick_read_off_incumbent_challenger_ledger"] = {
            "recorded": 0,
            "skipped": True,
            "reason": "pass --record-decisions to append the smooth-everywhere read's "
            "picks to the prospective challenger ledger",
        }
        result["era_weighted_half_life_8_challenger_ledger"] = {
            "recorded": 0,
            "skipped": True,
            "reason": "pass --record-decisions to append the era-weighted (half-life 8) "
            "refit's picks to the prospective challenger ledger",
        }
        result["forecast_cold_visitor_tilt_challenger_ledger"] = {
            "recorded": 0,
            "skipped": True,
            "reason": "pass --record-decisions to append the forecast cold-visitor "
            "tilt's picks to the prospective challenger ledger",
        }
        result["interim_hc_first_game_tilt_challenger_ledger"] = {
            "recorded": 0,
            "skipped": True,
            "reason": "pass --record-decisions to append the interim head-coach "
            "first-game tilt's picks to the prospective challenger ledger",
        }
        result["forecast_weather_kn_warm_team_cold_late_tilt_challenger_ledger"] = {
            "recorded": 0,
            "skipped": True,
            "reason": "pass --record-decisions to append the forecast (kickoff-nearest) "
            "warm-team-cold-late tilt's picks to the prospective challenger ledger",
        }
        result["forecast_weather_kn_precip_high_total_tilt_challenger_ledger"] = {
            "recorded": 0,
            "skipped": True,
            "reason": "pass --record-decisions to append the forecast (kickoff-nearest) "
            "precip-high-total tilt's picks to the prospective challenger ledger",
        }
        result["movement_rule_composed_challenger_ledger"] = {
            "recorded": 0,
            "skipped": True,
            "reason": "pass --record-decisions to append the movement-rule-on-chain "
            "challenger's picks to the prospective challenger ledger",
        }
        result["nflcom_refresh_out2_starters_challenger_ledger"] = {
            "recorded": 0,
            "skipped": True,
            "reason": "pass --record-decisions to append the NFL.com Friday out>=2-starters "
            "refresh fade's picks to the prospective challenger ledger",
        }
        result["qb_revenge_deadline_drag_stack_challenger_ledger"] = {
            "recorded": 0,
            "skipped": True,
            "reason": "pass --record-decisions to append the qb_revenge/deadline_drag "
            "stacked candidate's picks to the prospective challenger ledger",
        }
        result["totals_served_method_challenger_ledger"] = {
            "recorded": 0,
            "skipped": True,
            "reason": "pass --record-decisions to append this week's tiebreaker game under "
            "both served-total methods to the prospective challenger ledger",
        }
    return result


def _cmd_publish_predictions(args: argparse.Namespace) -> None:
    result = orchestrate_publish_predictions(parse_publish_predictions_request(args))
    _print_json(result)


def _cmd_publish_board(args: argparse.Namespace) -> None:
    _print_json(_write_public_site(args.site_destination or args.destination))


def _cmd_refresh_picks(args: argparse.Namespace) -> None:
    season, week = _resolve_active_forecast_season_week(args, _artifacts_root())
    plan = plan_refresh(
        _artifacts_root(),
        _data_root(),
        season=season,
        week=week,
        features_path=args.features,
        min_train_games=args.min_train_games,
    )
    result = refresh_summary(plan, record_decisions=args.record_decisions)
    try:
        result["best_pick_refresh_ledger"] = record_best_pick_refresh(
            _artifacts_root(), _data_root(), plan, record_decisions=args.record_decisions
        )
    except Exception as error:
        result["best_pick_refresh_ledger"] = {"recorded": 0, "error": str(error)}
    try:
        result["ledger"] = record_plan(
            _artifacts_root(),
            plan,
            note=args.note,
            record_decisions=args.record_decisions,
            trigger_type=getattr(args, "trigger_type", "clock_dispatch"),
            trigger_source=getattr(args, "trigger_source", ""),
        )
    except Exception as error:
        result["ledger"] = {"recorded": 0, "error": str(error)}
    try:
        result["injury_signal_refresh_tilt"] = record_injury_signal_refresh_tilt(
            _artifacts_root(), _data_root(), plan, record_decisions=args.record_decisions
        )
    except Exception as error:
        result["injury_signal_refresh_tilt"] = {"recorded": 0, "error": str(error)}
    try:
        result["nflcom_refresh_out2_starters_overlay"] = record_nflcom_refresh_overlay(
            _artifacts_root(), _data_root(), plan, record_decisions=args.record_decisions
        )
    except Exception as error:
        result["nflcom_refresh_out2_starters_overlay"] = {
            "recorded": 0,
            "error": str(error),
        }
    try:
        result["inactives_refresh_overlay"] = record_inactives_refresh_overlay(
            _artifacts_root(), _data_root(), plan, record_decisions=args.record_decisions
        )
    except Exception as error:
        result["inactives_refresh_overlay"] = {"recorded": 0, "error": str(error)}
    try:
        result["crew_tilt_refresh_overlay"] = record_crew_tilt_refresh_overlay(
            _artifacts_root(),
            _data_root(),
            plan,
            repo_root=Path.cwd(),
            record_decisions=args.record_decisions,
        )
    except Exception as error:
        result["crew_tilt_refresh_overlay"] = {"recorded": 0, "error": str(error)}
    try:
        result["specialist_absence_fade_refresh_overlay"] = (
            record_specialist_absence_fade_refresh_overlay(
                _artifacts_root(),
                _data_root(),
                plan,
                record_decisions=args.record_decisions,
            )
        )
    except Exception as error:
        result["specialist_absence_fade_refresh_overlay"] = {"recorded": 0, "error": str(error)}
    try:
        result["late_week_move_follow_refresh_overlay"] = (
            record_late_week_move_follow_refresh_overlay(
                _artifacts_root(), _data_root(), plan, record_decisions=args.record_decisions
            )
        )
    except Exception as error:
        result["late_week_move_follow_refresh_overlay"] = {"recorded": 0, "error": str(error)}
    try:
        result["handle_follow_refresh_overlay"] = record_handle_follow_refresh_overlay(
            _artifacts_root(), plan, record_decisions=args.record_decisions
        )
    except Exception as error:
        result["handle_follow_refresh_overlay"] = {"recorded": 0, "error": str(error)}
    try:
        result["consensus_movement_refresh_overlay"] = record_consensus_movement_refresh_overlay(
            _artifacts_root(), plan, record_decisions=args.record_decisions
        )
    except Exception as error:
        result["consensus_movement_refresh_overlay"] = {"recorded": 0, "error": str(error)}
    result["failed_recorders"] = collect_failed_recorders(result, REFRESH_CHALLENGER_RESULT_KEYS)
    if args.publish_card:
        if not plan.changed_games:
            result["card"] = {
                "written": False,
                "reason": "no eligible picks changed; nothing to append",
            }
        else:
            try:
                append_refresh_to_card(args.destination, plan, note=args.note)
                result["card"] = {"written": True, "destination": str(args.destination)}
            except (ValueError, FileNotFoundError) as error:
                result["card"] = {"written": False, "error": str(error)}
    _print_json(result)


def register(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
    current_year: int,
) -> None:

    publish = subparsers.add_parser(
        "publish-predictions",
        help="write the synchronized active weekly ATS card into GitHub Markdown",
    )
    publish.add_argument("--destination", type=Path, default=Path("CURRENT_PREDICTIONS.md"))
    publish.add_argument("--readme", type=Path, default=Path("README.md"))
    publish.add_argument(
        "--with-board",
        action="store_true",
        default=True,
        help=(
            "regenerate the public GitHub Pages site into docs/. ON by default since "
            "2026-08-19 so the "
            "served site can never lag the published card; retained as an explicit "
            "flag only so existing invocations keep working"
        ),
    )
    publish.add_argument(
        "--no-board",
        dest="with_board",
        action="store_false",
        help="skip regenerating the public site (rehearsal publishes that must not touch docs/)",
    )
    _add_board_destination_args(publish, legacy_flag="--board-destination")
    publish.add_argument(
        "--record-decisions",
        action="store_true",
        help=(
            "append this card's pre-kickoff picks to the paper-decision CLV ledger. Off "
            "by default -- recording is a deliberate act for the real weekly lock, not "
            "something an ordinary/rehearsal publish should do. record_paper_decisions "
            "also refuses to write when this week's earliest kickoff is more than "
            "RECORDING_LOCK_WINDOW away, so passing this flag outside the real lock "
            "week still does not reach the ledger."
        ),
    )
    publish.add_argument(
        "--record-from-forecast",
        default=None,
        metavar="ARTIFACT",
        help=(
            "operator override (owner, 2026-09-09): record every ledger this run writes "
            "-- the paper-decision ledger, each overlay and Best Pick challenger arm, and "
            "the tiebreaker shade ledger -- from this margin_predictions/... artifact "
            "instead of the active model's linked forecast, so they all record the card "
            "that was actually played when the linked forecast has since moved on. Only "
            "meaningful with --record-decisions."
        ),
    )
    publish.add_argument(
        "--replace-week",
        action="store_true",
        help=(
            "operator override (owner, 2026-09-09): drop the recorded week's existing "
            "rows first in every ledger this run writes -- paper decisions, each overlay "
            "and Best Pick challenger arm, the Best Pick Tuesday ledger and the "
            "tiebreaker shade ledger (each prior ledger is kept as a timestamped .bak "
            "beside it) -- so a missed lock, or one recorded on the wrong lines, can be "
            "re-recorded everywhere rather than only in the paper ledger. Only rows for "
            "games that are still before kickoff are replaced; a row for a game already "
            "under way is left exactly as it is, because the append step would not "
            "re-create it. Pre-kickoff and recording-window guards still apply. The arms "
            "in REPLACE_WEEK_FROZEN_ARMS are left on the card they were frozen from, and "
            "the result's replaced_week block reports what every ledger replaced and "
            "left. Only meaningful with --record-decisions."
        ),
    )
    publish.set_defaults(handler=_cmd_publish_predictions)

    refresh_picks = subparsers.add_parser(
        "refresh-picks",
        help=(
            "recompute one week's picks with current data at the frozen Tuesday grading "
            "lines (POL-11, docs/late_week_refresh.md); a second, opt-in step run any "
            "time between the Tuesday publish and each game's own deadline"
        ),
    )
    _add_active_forecast_season_week_args(refresh_picks)
    refresh_picks.add_argument(
        "--features",
        type=Path,
        default=None,
        help=(
            "current-week feature table (default: the active model's own card-path "
            "table under data/processed/, matching weekly-run's card path)"
        ),
    )
    refresh_picks.add_argument("--min-train-games", type=int, default=DEFAULT_MIN_TRAIN_GAMES)
    refresh_picks.add_argument(
        "--record-decisions",
        action="store_true",
        help=(
            "append this pass's changed, eligible picks to the append-only "
            "pick-revision ledger, AND every eligible game's injury_signal_refresh_tilt "
            "challenger reading (both arms) to its own ledger. Off by default, exactly "
            "like publish-predictions --record-decisions. Both recorders also refuse to "
            "write when this week's earliest kickoff is more than RECORDING_LOCK_WINDOW "
            "away, and record_plan never revises a game whose own deadline (its kickoff, "
            "or the week's Sunday 4:00 PM ET cap if earlier) has already passed."
        ),
    )
    refresh_picks.add_argument(
        "--publish-card",
        action="store_true",
        help=(
            "additively label a 'Late-week refresh' section onto the published card "
            "(--destination) listing only changed picks; never rewrites the Tuesday "
            "section publish-predictions wrote"
        ),
    )
    refresh_picks.add_argument("--destination", type=Path, default=Path("CURRENT_PREDICTIONS.md"))
    refresh_picks.add_argument(
        "--note",
        type=str,
        default="",
        help=(
            "free-text label for which weekly pass this is (e.g. 'thursday_afternoon', "
            "'sunday_morning_final'), stored in each revision's reason field and shown "
            "in the card section"
        ),
    )
    refresh_picks.add_argument(
        "--trigger-type",
        type=str,
        default="clock_dispatch",
        help=(
            "MKT-08 refresh provenance: 'clock_dispatch' for the scheduled "
            "passes, 'news_event' for a future news-driven pass. Stored on "
            "every appended pick-revision row."
        ),
    )
    refresh_picks.add_argument(
        "--trigger-source",
        type=str,
        default="",
        help=(
            "MKT-08 refresh provenance: the scheduler job id or invoking "
            "context (e.g. 'refresh_thu'). Stored on every appended "
            "pick-revision row."
        ),
    )
    refresh_picks.set_defaults(handler=_cmd_refresh_picks)

    publish_board = subparsers.add_parser(
        "publish-board",
        help=(
            "render the public GitHub Pages site (ATS Terminal): index.html, model.html, "
            "and findings.html into docs/"
        ),
    )
    _add_board_destination_args(publish_board, legacy_flag="--destination")
    publish_board.set_defaults(handler=_cmd_publish_board)
