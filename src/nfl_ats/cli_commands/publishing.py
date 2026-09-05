"""Publishing commands: the weekly card, the public site and pick refresh."""

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
from nfl_ats.board_content import verify_number_provenance
from nfl_ats.board_site import build_site
from nfl_ats.bye_edge_fade_overlay import record_bye_edge_fade_challenger_decisions
from nfl_ats.cli_common import (
    _add_board_destination_args,
    _add_season_week_args,
    _artifacts_root,
    _data_root,
    _print_json,
    _registry_root,
)
from nfl_ats.clv import record_paper_decisions
from nfl_ats.coach_fade_overlay import record_overlay_challenger_decisions
from nfl_ats.constants import DEFAULT_MIN_TRAIN_GAMES
from nfl_ats.crew_tilt_refresh_overlay import record_crew_tilt_refresh_overlay
from nfl_ats.data import DataContractError
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
from nfl_ats.inactives_refresh_overlay import record_inactives_refresh_overlay
from nfl_ats.injury_signal_refresh_tilt import record_injury_signal_refresh_tilt
from nfl_ats.injury_value_tilt_overlay import record_injury_value_tilt_challenger_decisions
from nfl_ats.interim_hc_first_game_tilt_overlay import (
    record_interim_hc_first_game_tilt_challenger_decisions,
)
from nfl_ats.io import atomic_text
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
from nfl_ats.turnover_luck_rebound_tilt_overlay import (
    record_turnover_luck_rebound_tilt_challenger_decisions,
)

PUBLISH_CHALLENGER_RESULT_KEYS: dict[str, str] = {
    "weak_stack_expected_lineup_loss": "expected_lineup_loss_challenger_ledger",
    "hc_year_one_fade_overlay": "overlay_challenger_ledger",
    "bye_edge_fade_overlay": "bye_edge_fade_challenger_ledger",
    "best_pick_nomination_v2": "nomination_challenger_ledger",
    "best_pick_nomination_v3": "nomination_v3_challenger_ledger",
    "best_pick_big_spread_eligibility": "big_spread_nomination_challenger_ledger",
    "injury_value_lost_tilt_overlay": "injury_value_tilt_challenger_ledger",
    "division_revenge_tilt_overlay": "division_revenge_tilt_challenger_ledger",
    "surface_switch_tilt_overlay": "surface_switch_tilt_challenger_ledger",
    "spread_gap_zone_fade_overlay": "spread_gap_zone_fade_challenger_ledger",
    "overlay_production_chain_coach_arrest_incumbent": ("four_overlay_incumbent_challenger_ledger"),
    "ecdf_mapping_incumbent": "ecdf_mapping_incumbent_challenger_ledger",
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


def _site_directory(destination: Path) -> Path:
    """The directory a public-site flag points at.

    ``--destination``/``--board-destination`` historically named the single
    board FILE (``docs/index.html``); the site is now three pages, so a path
    that looks like a file is reduced to its parent directory. That keeps every
    existing invocation working while ``--site-destination docs`` says what is
    actually meant.
    """

    return destination.parent if destination.suffix else destination


def _write_public_site(destination: Path) -> dict[str, Any]:
    """Write the real ATS Terminal site (:func:`nfl_ats.board_site.build_site`)
    to ``destination``'s directory.

    2026-08-31 full-site conversion: this used to call
    ``public_board.build_public_site`` (a single skin, one file per the old
    seven-entry ``SITE_PAGES``, written flat into ``directory``). It briefly
    called a two-skin ``build_two_skin_site`` (a ``terminal/``/``desk/``
    directory split behind a top-level redirect) before the owner dropped
    the Cover Desk skin entirely. It now calls
    :func:`~nfl_ats.board_site.build_site`, which returns exactly THREE
    pages -- ``"index.html"``, ``"model.html"``, ``"findings.html"`` -- each
    a bare, site-root relative path, same flat layout as the original
    single-skin site. Nothing else about this function's contract (loaders,
    guards, fail-open behavior -- all owned by
    ``build_site``/``board_site_content.load_site_content``) changed.
    """

    directory = _site_directory(destination)
    # Fail closed, not degraded (owner, 2026-09-05, verbatim: "please do not
    # let those percentages get out of date anymore") -- raises
    # NumberProvenanceError, uncaught here on purpose, naming exactly which
    # artifact needs recomputing, before a single page is written.
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
        # Retained for callers that parsed the single-page output -- the
        # redirect page still lives at exactly this path.
        "board_destination": str(directory / "index.html"),
        "nojekyll": str(nojekyll),
    }


@dataclass(frozen=True)
class PublishPredictionsRequest:
    """Everything ``nfl-ats publish-predictions`` needs from the command line."""

    destination: Path
    readme: Path
    with_board: bool
    site_destination: Path | None
    board_destination: Path | None
    record_decisions: bool


def parse_publish_predictions_request(args: argparse.Namespace) -> PublishPredictionsRequest:
    """Validate the parsed namespace into a PublishPredictionsRequest.

    Pure: reads only ``args`` and raises exactly what reading a missing or
    ill-typed attribute raises today."""

    return PublishPredictionsRequest(
        destination=args.destination,
        readme=args.readme,
        with_board=bool(args.with_board),
        site_destination=args.site_destination,
        board_destination=args.board_destination,
        record_decisions=bool(args.record_decisions),
    )


def orchestrate_publish_predictions(request: PublishPredictionsRequest) -> dict[str, Any]:
    """Publish the active card, optionally the site, and the opt-in recorders.

    Returns the result document the handler prints. Every recorder stays
    fail-open here exactly as before: a recorder error lands in the result
    and never un-publishes the card."""

    publish_instant = datetime.now(UTC)
    # Fail closed, not degraded (owner, 2026-09-05, verbatim: "please do not
    # let those percentages get out of date anymore") -- runs even when
    # ``--no-board`` skips the site build below, since the published card's
    # own headline prose (``publishing._composition_note``) quotes the same
    # played-policy figures. Raises NumberProvenanceError, uncaught here on
    # purpose, naming exactly which artifact needs recomputing.
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
        # Default-on since 2026-08-19: the public site is THE dashboard, and a
        # publish that skips regeneration is how docs/ served picks that
        # disagreed with the published card (owner-observed: the site showed
        # the pre-overlay BAL pick and the v1 ARI Best Pick for hours). A
        # rehearsal publish that must not touch docs/ passes --no-board.
        # Fail-open like the ledger recorders below: a site-build failure must
        # stay visible in the result but never un-publish the card.
        try:
            # cast only: --board-destination always has a Path default, so the
            # `or` can never actually yield None on the real CLI path.
            site_destination = cast(Path, request.site_destination or request.board_destination)
            result.update(_write_public_site(site_destination))
        except (ValueError, FileNotFoundError) as error:
            result["public_site"] = {"written": False, "error": str(error)}
    if request.record_decisions:
        # MKT-04 routine wiring: every published card's pre-kickoff picks are
        # appended to the paper-decision CLV ledger. A failure here must stay
        # visible but not un-publish the files already written above.
        # ``record_paper_decisions`` itself refuses to write when this week's
        # earliest kickoff is more than RECORDING_LOCK_WINDOW away, so passing
        # this flag on a rehearsal run still does not reach the ledger.
        try:
            result["clv_ledger"] = record_paper_decisions(
                _artifacts_root(),
                data_root=_data_root(),
                now=publish_instant,
                require_fresh_arrest_overlay=True,
            )
        except (ValueError, FileNotFoundError) as error:
            result["clv_ledger"] = {"recorded": 0, "error": str(error)}
        # The year-1-coach fade overlay arm (PER-07,
        # docs/coach_fade_overlay.md): the paper ledger stores both the raw
        # model side and the final played policy side; this appends the
        # coach-only arm for every game to the SEPARATE prospective ledger.
        # ``prospective-score`` derives the raw-model control from the frozen
        # ``model_pick_side`` column. A failure here must not un-publish the
        # card either.
        try:
            result["overlay_challenger_ledger"] = record_overlay_challenger_decisions(
                _artifacts_root(), _data_root()
            )
        except (ValueError, FileNotFoundError, DataContractError) as error:
            result["overlay_challenger_ledger"] = {"recorded": 0, "error": str(error)}
        # POL-09's v2 Best Pick nomination rule (nfl_ats.best_pick_nomination):
        # v1's nomination is already in clv_ledger above (unchanged, via the
        # active model's own is_best_pick flag); this appends v2's weekly
        # nominee to the SEPARATE prospective challenger ledger, so the
        # season scores v1 against v2 weekly. A failure here must not
        # un-publish the card either.
        try:
            result["nomination_challenger_ledger"] = record_nomination_challenger_decisions(
                _artifacts_root(), _data_root()
            )
        except (ValueError, FileNotFoundError, DataContractError) as error:
            result["nomination_challenger_ledger"] = {"recorded": 0, "error": str(error)}
        # POL-09's v3 Best Pick nomination challenger (docs/best_pick_ranker.md
        # "v3 audit", 2026-08-19): SIDE-LEDGER-ONLY, never read by publishing.py
        # and never touches NOMINATION_V2_ENABLED or is_best_pick. Same filter
        # and primary ranking as v2, but ties break on game_id alone (no
        # dispersion tie-break) -- the historical head-to-head against v2 leaned
        # positive (P+ 0.631) but traced to a single diverging week of 103, so
        # this accrues independent 2026 evidence rather than resting on that.
        # A failure here must not un-publish the card either.
        try:
            result["nomination_v3_challenger_ledger"] = record_nomination_v3_challenger_decisions(
                _artifacts_root(), _data_root()
            )
        except (ValueError, FileNotFoundError, DataContractError) as error:
            result["nomination_v3_challenger_ledger"] = {"recorded": 0, "error": str(error)}
        # Prospective-only 10+ point spread eligibility screen for Best Pick
        # (docs/best_pick_big_spread_challenger.md). It composes with v2 and
        # records one alternative nominee; publishing.py never imports it, so
        # the played/published Best Pick remains untouched. A failure here
        # must not un-publish the card either.
        try:
            result["big_spread_nomination_challenger_ledger"] = (
                record_big_spread_nomination_challenger_decisions(_artifacts_root(), _data_root())
            )
        except (ValueError, FileNotFoundError, DataContractError) as error:
            result["big_spread_nomination_challenger_ledger"] = {
                "recorded": 0,
                "error": str(error),
            }
        # Injury value-lost tilt overlay (docs/injury_value_lost_tilt_overlay.md):
        # a parameter-free pick-level nudge, dual-tracked against the active
        # model in the SEPARATE prospective challenger ledger only -- it is
        # never applied to the published card. A failure here must not
        # un-publish the card either.
        try:
            result["injury_value_tilt_challenger_ledger"] = (
                record_injury_value_tilt_challenger_decisions(_artifacts_root(), _data_root())
            )
        except (ValueError, FileNotFoundError, DataContractError) as error:
            result["injury_value_tilt_challenger_ledger"] = {"recorded": 0, "error": str(error)}
        # Division-revenge tilt overlay (docs/division_revenge_tilt_overlay.md):
        # a parameter-free pick-level nudge, dual-tracked against the active
        # model in the SEPARATE prospective challenger ledger only -- it is
        # never applied to the published card. A failure here must not
        # un-publish the card either.
        try:
            result["division_revenge_tilt_challenger_ledger"] = (
                record_division_revenge_tilt_challenger_decisions(_artifacts_root(), _data_root())
            )
        except (ValueError, FileNotFoundError, DataContractError) as error:
            result["division_revenge_tilt_challenger_ledger"] = {
                "recorded": 0,
                "error": str(error),
            }
        # Backup-QB fade overlay (docs/backup_qb_fade_overlay.md): a
        # parameter-free pick-level nudge, dual-tracked against the active
        # model in the SEPARATE prospective challenger ledger only -- it is
        # never applied to the published card. A failure here must not
        # un-publish the card either.
        try:
            result["backup_qb_fade_challenger_ledger"] = record_backup_qb_fade_challenger_decisions(
                _artifacts_root(), _data_root()
            )
        except (ValueError, FileNotFoundError, DataContractError) as error:
            result["backup_qb_fade_challenger_ledger"] = {"recorded": 0, "error": str(error)}
        # Surface-switch tilt overlay (docs/surface_switch_tilt_overlay.md): a
        # parameter-free pick-level nudge, dual-tracked against the active
        # model in the SEPARATE prospective challenger ledger only -- it is
        # never applied to the published card. A failure here must not
        # un-publish the card either.
        try:
            result["surface_switch_tilt_challenger_ledger"] = (
                record_surface_switch_tilt_challenger_decisions(_artifacts_root(), _data_root())
            )
        except (ValueError, FileNotFoundError, DataContractError) as error:
            result["surface_switch_tilt_challenger_ledger"] = {"recorded": 0, "error": str(error)}
        # Spread-gap-zone fade overlay (docs/spread_gap_zone_fade_overlay.md):
        # a parameter-free pick-level nudge, dual-tracked against the active
        # model in the SEPARATE prospective challenger ledger only -- it is
        # never applied to the published card. A failure here must not
        # un-publish the card either.
        try:
            result["spread_gap_zone_fade_challenger_ledger"] = (
                record_spread_gap_zone_fade_challenger_decisions(_artifacts_root(), _data_root())
            )
        except (ValueError, FileNotFoundError, DataContractError) as error:
            result["spread_gap_zone_fade_challenger_ledger"] = {
                "recorded": 0,
                "error": str(error),
            }
        try:
            result["expected_lineup_loss_challenger_ledger"] = (
                record_expected_lineup_loss_challenger_decisions(_artifacts_root(), _data_root())
            )
        except (ValueError, FileNotFoundError) as error:
            result["expected_lineup_loss_challenger_ledger"] = {"recorded": 0, "error": str(error)}
        # Low-total divisional home-dog challenger (LEAD-42,
        # docs/schedule_flag_battery.md Wave 2): a parameter-free pick-level
        # nudge, dual-tracked against the active model in the SEPARATE
        # prospective challenger ledger only -- it is never applied to the
        # published card. Reads only the card's own div_game/total_line/
        # spread_line columns, no external data source. A failure here must
        # not un-publish the card either.
        try:
            result["low_total_div_home_dog_challenger_ledger"] = (
                record_low_total_div_home_dog_challenger_decisions(_artifacts_root(), _data_root())
            )
        except (ValueError, FileNotFoundError, DataContractError) as error:
            result["low_total_div_home_dog_challenger_ledger"] = {
                "recorded": 0,
                "error": str(error),
            }
        # The six 2026-09-01 parameter-free overlays are prospective-only.
        # Each records its own forced-pick arm and cannot affect the published
        # card; preserve an individual failure in the result without undoing
        # the publish that already completed above.
        try:
            result["bye_edge_fade_challenger_ledger"] = record_bye_edge_fade_challenger_decisions(
                _artifacts_root(), _data_root()
            )
        except (ValueError, FileNotFoundError, DataContractError) as error:
            result["bye_edge_fade_challenger_ledger"] = {"recorded": 0, "error": str(error)}
        try:
            result["tank_zone_fade_tilt_challenger_ledger"] = (
                record_tank_zone_fade_tilt_challenger_decisions(_artifacts_root(), _data_root())
            )
        except (ValueError, FileNotFoundError, DataContractError) as error:
            result["tank_zone_fade_tilt_challenger_ledger"] = {
                "recorded": 0,
                "error": str(error),
            }
        try:
            result["third_down_reversion_fade_challenger_ledger"] = (
                record_third_down_reversion_fade_challenger_decisions(
                    _artifacts_root(), _data_root()
                )
            )
        except (ValueError, FileNotFoundError, DataContractError) as error:
            result["third_down_reversion_fade_challenger_ledger"] = {
                "recorded": 0,
                "error": str(error),
            }
        try:
            result["turnover_luck_rebound_tilt_challenger_ledger"] = (
                record_turnover_luck_rebound_tilt_challenger_decisions(
                    _artifacts_root(), _data_root()
                )
            )
        except (ValueError, FileNotFoundError, DataContractError) as error:
            result["turnover_luck_rebound_tilt_challenger_ledger"] = {
                "recorded": 0,
                "error": str(error),
            }
        try:
            result["special_teams_return_tilt_challenger_ledger"] = (
                record_special_teams_return_tilt_challenger_decisions(
                    _artifacts_root(), _data_root()
                )
            )
        except (ValueError, FileNotFoundError, DataContractError) as error:
            result["special_teams_return_tilt_challenger_ledger"] = {
                "recorded": 0,
                "error": str(error),
            }
        try:
            result["pace_mismatch_dog_tilt_challenger_ledger"] = (
                record_pace_mismatch_dog_tilt_challenger_decisions(_artifacts_root(), _data_root())
            )
        except (ValueError, FileNotFoundError, DataContractError) as error:
            result["pace_mismatch_dog_tilt_challenger_ledger"] = {
                "recorded": 0,
                "error": str(error),
            }
        # PBP-08 protection-mismatch tilt (docs/pbp08_matchup_screen.md): back
        # the defense when one side's offense carries a top-quartile four-game
        # pressure-allowed window against a top-quartile pressure-generating
        # defense. The strongest mined mean-edge cell in the project (+0.336
        # points, both blockings excluding zero, mirror controls clean), wired
        # as a dual-tracked challenger only -- never applied to the published
        # card, and costing no rotation-registry window. The flag build is
        # FAIL-OPEN (absent snapshot -> zero flags), but this outer try/except
        # still guards every other failure mode so a failure here must not
        # un-publish the card either.
        try:
            result["pbp08_protection_mismatch_tilt_challenger_ledger"] = (
                record_pbp08_protection_mismatch_tilt_challenger_decisions(
                    _artifacts_root(), _data_root(), now=publish_instant
                )
            )
        except (ValueError, FileNotFoundError, DataContractError) as error:
            result["pbp08_protection_mismatch_tilt_challenger_ledger"] = {
                "recorded": 0,
                "error": str(error),
            }
        # Paired incumbent for the four-member production policy: record the
        # exact former coach->arrests chain frozen by the primary ledger.
        try:
            result["four_overlay_incumbent_challenger_ledger"] = (
                record_former_production_incumbent_decisions(
                    _artifacts_root(), _data_root(), now=publish_instant
                )
            )
        except (ValueError, FileNotFoundError, DataContractError) as error:
            result["four_overlay_incumbent_challenger_ledger"] = {
                "recorded": 0,
                "error": str(error),
            }
        # ECDF-mapping-incumbent overlay (docs/smooth_cdf_mapping.md, MOD-08
        # promotion, 2026-08-19): the published card's own probability read
        # IS now the Gaussian mapping (score_outcome_week's promoted
        # default); this challenger tracks the FORMER production ECDF read
        # off the SAME out-of-time residual sample, dual-tracked against the
        # active model in the SEPARATE prospective challenger ledger only --
        # it is never applied to the published card. Supersedes the retired
        # smooth_cdf_mapping challenger (artifacts/prospective/challengers.json),
        # which tracked the mapping in the opposite direction while the
        # published card was still ECDF-native. A failure here must not
        # un-publish the card either.
        try:
            result["ecdf_mapping_incumbent_challenger_ledger"] = (
                record_ecdf_mapping_incumbent_challenger_decisions(_artifacts_root(), _data_root())
            )
        except (ValueError, FileNotFoundError, DataContractError) as error:
            result["ecdf_mapping_incumbent_challenger_ledger"] = {
                "recorded": 0,
                "error": str(error),
            }
        # Era-weighted (half-life 8) challenger (docs/era_weighting_screen.md,
        # MOD-14): refits the active recipe weekly with exponential
        # season-decay sample weights, dual-tracked against the active model
        # in the SEPARATE prospective challenger ledger only -- it is never
        # applied to the published card. A failure here must not un-publish
        # the card either.
        try:
            result["era_weighted_half_life_8_challenger_ledger"] = (
                record_era_weighted_half_life_8_challenger_decisions(
                    _artifacts_root(), _data_root()
                )
            )
        except (ValueError, FileNotFoundError, DataContractError) as error:
            result["era_weighted_half_life_8_challenger_ledger"] = {
                "recorded": 0,
                "error": str(error),
            }
        # Forecast cold-visitor tilt overlay (docs/forecast_weather_screen.md,
        # ENV-01): a parameter-free pick-level nudge using the live
        # Tuesday-noon GFS-MOS forecast, dual-tracked against the active
        # model in the SEPARATE prospective challenger ledger only -- it is
        # never applied to the published card. The live forecast fetch is
        # FAIL-OPEN (network/station-mapping failures fold into zero flags,
        # never an exception), but this outer try/except still guards
        # against every other failure mode (registry/fingerprint/ledger
        # errors) so a failure here must not un-publish the card either.
        try:
            result["forecast_cold_visitor_tilt_challenger_ledger"] = (
                record_forecast_cold_visitor_tilt_challenger_decisions(
                    _artifacts_root(), _data_root(), _registry_root()
                )
            )
        except (ValueError, FileNotFoundError, DataContractError) as error:
            result["forecast_cold_visitor_tilt_challenger_ledger"] = {
                "recorded": 0,
                "error": str(error),
            }
        # Interim head-coach first-game tilt overlay (docs/interim_coach_screen.md):
        # a parameter-free pick-level nudge, dual-tracked against the active
        # model in the SEPARATE prospective challenger ledger only -- it is
        # never applied to the published card. The interim-coach join is
        # FAIL-OPEN (missing/unavailable source data folds into zero flags,
        # never an exception), but this outer try/except still guards against
        # every other failure mode (registry/fingerprint/ledger errors) so a
        # failure here must not un-publish the card either.
        try:
            result["interim_hc_first_game_tilt_challenger_ledger"] = (
                record_interim_hc_first_game_tilt_challenger_decisions(
                    _artifacts_root(), _data_root()
                )
            )
        except (ValueError, FileNotFoundError, DataContractError) as error:
            result["interim_hc_first_game_tilt_challenger_ledger"] = {
                "recorded": 0,
                "error": str(error),
            }
        # Shared live kickoff-nearest GFS-MOS fetch (docs/forecast_weather_screen.md,
        # "Wiring recommendations": "one fetch, several consumers") for the two
        # challengers below -- fetched ONCE here and passed to both via their
        # forecasts= parameter, rather than each making its own outbound
        # network call for the identical (game, station, cutoff) set. Never
        # raises (see its own docstring); None just means each recorder falls
        # back to fetching for itself.
        shared_kn_forecasts = fetch_shared_kickoff_nearest_forecasts_fail_open(
            _artifacts_root(), _data_root(), _registry_root()
        )
        # Forecast (kickoff-nearest) warm-team-cold-late tilt overlay
        # (docs/forecast_weather_screen.md, highest-EV wiring recommendation
        # after the archive's 2009-2025 fetch completed -- both windows'
        # registered intervals exclude zero): a parameter-free pick-level
        # nudge using a LIVE kickoff-nearest GFS-MOS forecast, dual-tracked
        # against the active model in the SEPARATE prospective challenger
        # ledger only -- it is never applied to the published card. The live
        # forecast fetch is FAIL-OPEN, but this outer try/except still guards
        # against every other failure mode so a failure here must not
        # un-publish the card either.
        try:
            result["forecast_weather_kn_warm_team_cold_late_tilt_challenger_ledger"] = (
                record_forecast_weather_kn_warm_team_cold_late_tilt_challenger_decisions(
                    _artifacts_root(),
                    _data_root(),
                    _registry_root(),
                    forecasts=shared_kn_forecasts,
                )
            )
        except (ValueError, FileNotFoundError, DataContractError) as error:
            result["forecast_weather_kn_warm_team_cold_late_tilt_challenger_ledger"] = {
                "recorded": 0,
                "error": str(error),
            }
        # Forecast (kickoff-nearest) precip-high-total tilt overlay
        # (docs/forecast_weather_screen.md): a parameter-free pick-level
        # nudge sharing the SAME live kickoff-nearest GFS-MOS fetch as the
        # warm-team-cold-late challenger above (one fetch, several
        # consumers), dual-tracked against the active model in the SEPARATE
        # prospective challenger ledger only -- it is never applied to the
        # published card. The live forecast fetch is FAIL-OPEN, but this
        # outer try/except still guards against every other failure mode so
        # a failure here must not un-publish the card either.
        try:
            result["forecast_weather_kn_precip_high_total_tilt_challenger_ledger"] = (
                record_forecast_weather_kn_precip_high_total_tilt_challenger_decisions(
                    _artifacts_root(),
                    _data_root(),
                    _registry_root(),
                    forecasts=shared_kn_forecasts,
                )
            )
        except (ValueError, FileNotFoundError, DataContractError) as error:
            result["forecast_weather_kn_precip_high_total_tilt_challenger_ledger"] = {
                "recorded": 0,
                "error": str(error),
            }
        # Rain-on-grass underdog challenger (LEAD-37, docs/weather_venue_leads.md):
        # a parameter-free pick-level nudge sharing the SAME live
        # kickoff-nearest GFS-MOS fetch as the two challengers above (one
        # fetch, several consumers), dual-tracked against the active model in
        # the SEPARATE prospective challenger ledger only -- it is never
        # applied to the published card. The live forecast fetch is FAIL-OPEN,
        # but this outer try/except still guards against every other failure
        # mode so a failure here must not un-publish the card either.
        try:
            result["rain_on_grass_dog_challenger_ledger"] = (
                record_rain_on_grass_dog_challenger_decisions(
                    _artifacts_root(),
                    _data_root(),
                    _registry_root(),
                    forecasts=shared_kn_forecasts,
                )
            )
        except (ValueError, FileNotFoundError, DataContractError) as error:
            result["rain_on_grass_dog_challenger_ledger"] = {
                "recorded": 0,
                "error": str(error),
            }
        # Movement-rule-on-composed-chain challenger (2026-08-22 registration,
        # docs/movement_composition_eval.md): flips the PLAYED chain pick to the
        # market side whenever the latest captured line moved >=1.0 pt off the
        # frozen Tuesday line, reusing nfl_ats.pick_refresh's own read-only
        # captured-line read. Dual-tracked only -- never applied to the
        # published card. Runs after record_paper_decisions above, whose rows
        # are its base card. A failure here must not un-publish the card.
        try:
            result["movement_rule_composed_challenger_ledger"] = (
                record_movement_rule_composed_challenger_decisions(
                    _artifacts_root(), _data_root(), now=publish_instant
                )
            )
        except (ValueError, FileNotFoundError, DataContractError) as error:
            result["movement_rule_composed_challenger_ledger"] = {
                "recorded": 0,
                "error": str(error),
            }
        # NFL.com Friday out>=2-starters refresh fade on the chain (2026-08-22
        # registration, docs/nflcom_friday_refresh.md frozen rule text):
        # freshness-gated injury-page flags flip the PLAYED chain pick,
        # dual-tracked only. FAIL-OPEN by design: absent/stale inputs come back
        # as a skipped week, and any other failure is caught here so it must
        # not un-publish the card either.
        try:
            result["nflcom_refresh_out2_starters_challenger_ledger"] = (
                record_nflcom_refresh_out2_starters_challenger_decisions(
                    _artifacts_root(), _data_root(), now=publish_instant
                )
            )
        except (ValueError, FileNotFoundError, DataContractError) as error:
            result["nflcom_refresh_out2_starters_challenger_ledger"] = {
                "recorded": 0,
                "error": str(error),
            }
        # weak_stack_qb_revenge_deadline_drag prospective challenger (lane T,
        # docs/promotion_eval_20260905.md): NOT a pick-level tilt -- genuinely
        # refits its own weak_stack_qb_revenge_deadline_drag margin model each
        # week, walk-forward, attaching qb_revenge_flag and
        # deadline_integration_drag_flag onto the active model's own base
        # feature table at record time. The archive read is confounded by
        # multiplicity (best of three correlated arms on one reused window,
        # both components read AGAINST the candidate on that same
        # population), so the coordinator decision is do-not-promote on
        # SELECTION grounds, not a threshold; this SEPARATE prospective
        # challenger ledger is the no-window-cost way to keep testing it.
        # Never applied to the published card. A failure here must not
        # un-publish the card either.
        try:
            result["qb_revenge_deadline_drag_stack_challenger_ledger"] = (
                record_qb_revenge_deadline_drag_stack_challenger_decisions(
                    _artifacts_root(), _data_root(), now=publish_instant
                )
            )
        except (ValueError, FileNotFoundError, DataContractError) as error:
            result["qb_revenge_deadline_drag_stack_challenger_ledger"] = {
                "recorded": 0,
                "error": str(error),
            }
        # MOD-17 served-total side-ledger challenger (docs/mod17_joint_residual_model.md,
        # docs/tiebreaker.md "one lattice, one margin, one total"): records the
        # week's tiebreaker game under BOTH served-total methods
        # (nfl_ats.served_total.served_total_blend_k01 and
        # served_total_joint_residual) plus which one actually served, so the
        # 2026-09-05 EV promotion of the joint model's total output keeps
        # accruing paired prospective evidence at no rotation-registry cost.
        # Never affects the published tiebreaker guess itself -- that already
        # reads nfl_ats.served_total.SERVED_TOTAL_METHOD directly. A failure
        # here must not un-publish the card either.
        try:
            result["totals_served_method_challenger_ledger"] = (
                record_totals_served_method_decisions(
                    _artifacts_root(), _data_root(), now=publish_instant
                )
            )
        except (ValueError, FileNotFoundError, DataContractError) as error:
            result["totals_served_method_challenger_ledger"] = {
                "recorded": 0,
                "error": str(error),
            }
    else:
        # Safe by default: an ordinary publish does not touch the ledger.
        # Recording is a deliberate act (--record-decisions), because an
        # ordinary command silently reaching the real ledger during
        # rehearsal/testing is exactly how it was contaminated on 2026-08-18
        # (docs/prospective_evidence.md, "Known divergence").
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
    plan = plan_refresh(
        _artifacts_root(),
        _data_root(),
        season=args.season,
        week=args.week,
        features_path=args.features,
        min_train_games=args.min_train_games,
    )
    result = refresh_summary(plan, record_decisions=args.record_decisions)
    result["ledger"] = record_plan(
        _artifacts_root(),
        plan,
        note=args.note,
        record_decisions=args.record_decisions,
        trigger_type=getattr(args, "trigger_type", "clock_dispatch"),
        trigger_source=getattr(args, "trigger_source", ""),
    )
    result["injury_signal_refresh_tilt"] = record_injury_signal_refresh_tilt(
        _artifacts_root(), _data_root(), plan, record_decisions=args.record_decisions
    )
    # NFL.com Friday out>=2-starters fade (docs/nflcom_friday_refresh.md frozen
    # rule), CHALLENGER-TRACKED at refresh time: computes the WOULD-BE pick on
    # top of the post-market-follow played side and records it to a SEPARATE
    # append-only ledger. It can never alter the played pick -- plan is
    # consumed read-only -- and every absent-input path inside the recorder is
    # a documented no-op; this guard additionally keeps any unexpected failure
    # here from breaking the production refresh pass.
    try:
        result["nflcom_refresh_out2_starters_overlay"] = record_nflcom_refresh_overlay(
            _artifacts_root(), _data_root(), plan, record_decisions=args.record_decisions
        )
    except (ValueError, FileNotFoundError, DataContractError) as error:
        result["nflcom_refresh_out2_starters_overlay"] = {
            "recorded": 0,
            "error": str(error),
        }
    # Official T-90 inactives are a prospective challenger only. This recorder
    # consumes `plan` read-only and writes its separate ledger; it cannot alter
    # the refresh ledger, published card, or played pick.
    try:
        result["inactives_refresh_overlay"] = record_inactives_refresh_overlay(
            _artifacts_root(), _data_root(), plan, record_decisions=args.record_decisions
        )
    except (ValueError, FileNotFoundError, DataContractError) as error:
        result["inactives_refresh_overlay"] = {"recorded": 0, "error": str(error)}
    # Officiating crews are published after Tuesday, so this prospective arm
    # belongs to each late refresh rather than the Tuesday publish. It consumes
    # the plan read-only and writes only its own ledger; unexpected recorder
    # failures remain visible but cannot break a production refresh or card
    # append.
    try:
        result["crew_tilt_refresh_overlay"] = record_crew_tilt_refresh_overlay(
            _artifacts_root(),
            _data_root(),
            plan,
            repo_root=Path.cwd(),
            record_decisions=args.record_decisions,
        )
    except (ValueError, FileNotFoundError, DataContractError) as error:
        result["crew_tilt_refresh_overlay"] = {"recorded": 0, "error": str(error)}
    # Specialist (long-snapper/punter) absence fade (LEAD-17,
    # docs/schedule_flag_battery.md "Wave 7"): the weekly injury report is
    # filed Wednesday-Friday, strictly after the Tuesday lock, so this
    # prospective arm belongs to each late refresh rather than the Tuesday
    # publish. It consumes the plan read-only and writes only its own
    # ledger, graded at the frozen Tuesday line; unexpected recorder
    # failures remain visible but cannot break a production refresh or card
    # append.
    try:
        result["specialist_absence_fade_refresh_overlay"] = (
            record_specialist_absence_fade_refresh_overlay(
                _artifacts_root(),
                _data_root(),
                plan,
                record_decisions=args.record_decisions,
            )
        )
    except (ValueError, FileNotFoundError, DataContractError) as error:
        result["specialist_absence_fade_refresh_overlay"] = {"recorded": 0, "error": str(error)}
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
    """Register the publishing commands."""

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
    publish.set_defaults(handler=_cmd_publish_predictions)

    refresh_picks = subparsers.add_parser(
        "refresh-picks",
        help=(
            "recompute one week's picks with current data at the frozen Tuesday grading "
            "lines (POL-11, docs/late_week_refresh.md); a second, opt-in step run any "
            "time between the Tuesday publish and each game's own deadline"
        ),
    )
    _add_season_week_args(refresh_picks, required=True)
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
