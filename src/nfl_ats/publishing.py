from __future__ import annotations

import json
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import pandas as pd

from nfl_ats.active_model import active_artifact_path, load_active_ats_model
from nfl_ats.artifact_contracts import KIND_CARD, check_compatible, stamp
from nfl_ats.best_pick_nomination import nominate_v2_small_spread
from nfl_ats.card_explanation import (
    OverlayFiring,
    RefreshChangeInput,
    explain_card,
    explanations_to_dict,
    overlay_firing_from_arrest_flip,
    overlay_firing_from_coach_fade_flip,
    overlay_firings_from_composition,
    refresh_change_from_pick_revision,
)
from nfl_ats.card_explanation import (
    render_markdown as render_explanations_markdown,
)
from nfl_ats.card_view import BestPickNomination, resolve_card_view
from nfl_ats.coach_fade_overlay import OverlayResult, overlay_disclosure_note
from nfl_ats.dashboard.findings_content import PLAYED_CARD_EXPECTATION_HERO
from nfl_ats.displayed_confidence import (
    DISPLAYED_CONFIDENCE_FILENAME,
    DISPLAYED_PICK_PROBABILITY_COLUMN,
    ProductionDisplayedConfidence,
    attach_displayed_confidence,
    fit_production_displayed_confidence,
)
from nfl_ats.four_overlay_composition import FourOverlayCompositionResult
from nfl_ats.io import atomic_json, atomic_text
from nfl_ats.key_line_pick_read import key_line_touched_games
from nfl_ats.lineage import (
    LINEAGE_FILENAME,
    PUBLISHED_DISPLAY_FIELDS,
    CardLineage,
    OverlaySource,
    TiebreakerSource,
    build_card_lineage,
    extend_card_lineage_for_publication,
    feature_table_manifest,
    overlay_sources_from_composition,
    read_card_lineage,
    validate_card_lineage,
    write_card_lineage,
)
from nfl_ats.margin import margin_feature_columns
from nfl_ats.pick_refresh import load_pick_revisions, served_best_pick
from nfl_ats.player_arrests_back_side_overlay import (
    ArrestOverlayResult,
    arrest_overlay_disclosure_note,
)
from nfl_ats.public_board import humanize_identifier, load_waterfall_feed
from nfl_ats.published_picks import FrozenPick, frozen_picks
from nfl_ats.readme_state import apply_generated_state_blocks
from nfl_ats.source_freshness_policy import (
    BLOCKED as SOURCE_STATE_BLOCKED,
)
from nfl_ats.source_freshness_policy import (
    SourceFreshnessError,
    report_for_publication,
)
from nfl_ats.tiebreaker import (
    TiebreakerConsistencyError,
    TiebreakerReport,
    last_game_of_week,
    newest_schedules_path,
    tiebreaker_lineage_sources,
    tiebreaker_report,
)

TIEBREAKER_ARTIFACT_FILENAME = "tiebreaker.json"

README_PREDICTIONS_START = "<!-- CURRENT_PREDICTIONS:START -->"
README_PREDICTIONS_END = "<!-- CURRENT_PREDICTIONS:END -->"

BEST_PICK_MARK = "★ "


def _line(value: float) -> str:
    return "PK" if value == 0.0 else f"{value:+g}"


def _published_card(
    predictions: pd.DataFrame,
    best_pick_id: str | None = None,
    frozen: Mapping[str, FrozenPick] | None = None,
) -> pd.DataFrame:
    required = {
        "game_id",
        "gameday",
        "away_team",
        "home_team",
        "spread_line",
        "home_cover_probability",
    }
    missing = sorted(required.difference(predictions.columns))
    if missing:
        raise ValueError(f"Active forecast is missing publish columns: {', '.join(missing)}")
    card = predictions.copy()
    home_pick = card["home_cover_probability"].ge(0.5)
    card["Pick"] = card["home_team"].where(home_pick, card["away_team"])
    pick_line = (-card["spread_line"]).where(home_pick, card["spread_line"])
    card["ATS prediction"] = card["Pick"] + " " + pick_line.map(_line)
    if best_pick_id is not None:
        best = card["game_id"].astype(str).eq(best_pick_id)
        card.loc[best, "ATS prediction"] = BEST_PICK_MARK + card.loc[best, "ATS prediction"]
    stated = card["home_cover_probability"].where(home_pick, 1.0 - card["home_cover_probability"])
    card["Decision score"] = (
        pd.to_numeric(card[DISPLAYED_PICK_PROBABILITY_COLUMN], errors="coerce").fillna(stated)
        if DISPLAYED_PICK_PROBABILITY_COLUMN in card
        else stated
    )
    for game_id, pick in (frozen or {}).items():
        mask = card["game_id"].astype(str).eq(game_id)
        if not mask.any():
            continue
        frozen_home = card.loc[mask, "home_team"].eq(pick.pick_team)
        line = (-pick.market_spread) if bool(frozen_home.iloc[0]) else pick.market_spread
        prefix = BEST_PICK_MARK if best_pick_id == game_id else ""
        card.loc[mask, "ATS prediction"] = f"{prefix}{pick.pick_team} {_line(line)}"
        card.loc[mask, "Decision score"] = pick.displayed_score
    card["Matchup"] = card["away_team"] + " at " + card["home_team"]
    card["_gameday"] = pd.to_datetime(card["gameday"], errors="raise")
    card["Date"] = card["_gameday"].dt.strftime("%a, %b %d")
    card = card.sort_values(["_gameday", "game_id"], kind="stable")
    published = card[["Date", "Matchup", "ATS prediction", "Decision score"]].copy()
    published["Decision score"] = published["Decision score"].map(lambda value: f"{value:.1%}")
    return published


def _decision_score_note(displayed_confidence: ProductionDisplayedConfidence) -> str:

    if not displayed_confidence.served:
        return (
            "`Decision score` is the computer's own chance that this side covers, "
            "oriented to the final pick. On a flip it is a mirrored decision-strength "
            "score; it is also not historical accuracy.\n"
        )
    return (
        "`Decision score` is the computer's own chance that this side covers, adjusted "
        "for how the computer has actually done on spreads this size. Big favourites and "
        "big underdogs have been its weak spot, so a very confident-looking number there "
        "is pulled back toward what it has really hit, and it is never shown below 50% on a "
        "side this card is picking. It is a per-game chance, not historical accuracy.\n"
    )


def _publication_context(
    artifacts_root: Path,
    data_root: Path | None = None,
    *,
    published_at: datetime | None = None,
    require_fresh_arrest_overlay: bool = True,
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    pd.DataFrame,
    BestPickNomination,
    OverlayResult,
    ArrestOverlayResult,
    FourOverlayCompositionResult | None,
    pd.DataFrame,
    ProductionDisplayedConfidence,
]:
    active = load_active_ats_model(artifacts_root)
    if active is None:
        raise ValueError("No synchronized active ATS model is available to publish")
    forecast = active_artifact_path(artifacts_root, active, "weekly_forecast")
    if forecast is None:
        raise ValueError("Active ATS model has no linked weekly forecast")
    metadata_path = forecast / "metadata.json"
    recommendations_path = forecast / "recommendations.csv"
    if not metadata_path.is_file() or not recommendations_path.is_file():
        raise ValueError("Linked weekly forecast is missing metadata or recommendations")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("active_model_id") != active.get("model_id"):
        raise ValueError("Weekly forecast model ID does not match the active model")
    if metadata.get("synchronization_status") != "SYNCHRONIZED":
        raise ValueError("Weekly forecast is not synchronized with an evaluation")
    predictions = pd.read_csv(recommendations_path)
    method = str(active.get("method"))
    if "method" in predictions and not predictions["method"].eq(method).all():
        raise ValueError("Weekly recommendations contain a method other than the active method")
    sweep_path = forecast / "line_sweep.parquet"
    sweep = pd.read_parquet(sweep_path) if sweep_path.is_file() else pd.DataFrame()
    view = resolve_card_view(
        predictions,
        sweep,
        metadata,
        data_root=data_root,
        now=published_at,
        require_fresh_arrest_overlay=require_fresh_arrest_overlay,
        nominate_v2_fn=nominate_v2_small_spread,
        renominated_game_id=served_best_pick(
            artifacts_root, season=int(metadata["season"]), week=int(metadata["week"])
        ),
    )
    displayed_confidence = fit_production_displayed_confidence(
        artifacts_root,
        active,
        season=int(metadata["season"]),
        week=int(metadata["week"]),
    )
    served = attach_displayed_confidence(view.predictions, displayed_confidence)
    card = _published_card(
        served,
        view.nomination.active_game_id,
        frozen_picks(
            artifacts_root,
            now=published_at or datetime.now(UTC),
            season=int(metadata["season"]),
            week=int(metadata["week"]),
        ),
    )
    return (
        active,
        metadata,
        card,
        view.nomination,
        view.overlay,
        view.arrest_overlay,
        view.production_overlay,
        served,
        displayed_confidence,
    )


def _best_pick_note(card: pd.DataFrame, nomination: BestPickNomination) -> str:
    marked = card.loc[card["ATS prediction"].str.startswith(BEST_PICK_MARK)]
    if marked.empty:
        return ""
    row = marked.iloc[0]
    pick_text = row["ATS prediction"].removeprefix(BEST_PICK_MARK)
    lead = (
        f"**Best Pick of the week ({BEST_PICK_MARK.strip()}):** {pick_text} in {row['Matchup']}. "
        "The pool scores one Best Pick per regular-season week"
    )
    if nomination.active_rule == "v2":
        return f"{lead}. This pick was {nomination.method_note}\n\n"
    disclosure = f" {nomination.active_tie_note}" if nomination.active_tie_note else ""
    return (
        f"{lead}; this is the pick whose edge survives the widest range of "
        f"line movement.{disclosure}\n\n"
    )


def _overlay_note(overlay: OverlayResult) -> str:
    note = overlay_disclosure_note(overlay)
    return f"{note}\n\n" if note else ""


def _arrest_overlay_note(overlay: ArrestOverlayResult) -> str:
    note = arrest_overlay_disclosure_note(overlay)
    return f"{note}\n\n" if note else ""


def _composition_note(composition: FourOverlayCompositionResult) -> str:
    members = ", ".join(humanize_identifier(member.member_id) for member in composition.members)
    plural = "" if composition.flip_count == 1 else "s"
    return (
        "**Production policy active:** three situational rules run independently against "
        "the computer's first pick and flip it once when any one of them fires: coach fade, "
        "division revenge, and player arrests. "
        f"This week they changed {composition.flip_count} pick{plural}. "
        "The spread-only threshold adjustment is retired because it has no explained "
        "mechanism. Its archive comparison reuses 127 similar combinations scored on "
        "the same games; it is not independent evidence of future accuracy. "
        f"The planning estimate remains {PLAYED_CARD_EXPECTATION_HERO}. "
        "Paired prospective tracking against the former four-adjustment card begins "
        f"at the Week 1 lock. Rules: {members}. See docs/spread_gap_zone_retired.md.\n\n"
    )


def _tiebreaker_json_payload(
    guess: TiebreakerReport,
    *,
    generated_at: datetime,
    model_id: str | None,
    season: int,
    week: int,
    forecast_artifact: str | None,
) -> dict[str, Any]:

    return {
        "schema_version": 1,
        "game_id": guess.game_id,
        "season": season,
        "week": week,
        "forecast_artifact": forecast_artifact,
        "home": guess.home,
        "away": guess.away,
        "guess_home": guess.guess_home,
        "guess_away": guess.guess_away,
        "projected_total": guess.guess_home + guess.guess_away,
        "market_total": guess.consensus.total_line,
        "blended_total": guess.guess_total_line,
        "served_total": guess.served_total,
        "served_total_method": guess.served_total_method,
        "comparison_total_blend_k01": guess.comparison_total_blend_k01,
        "total_low_side_shade_points": guess.low_side_shade_points,
        "total_low_side_shade_source": guess.low_side_shade_source,
        "implied_margin": guess.guess_home - guess.guess_away,
        "pick_side": guess.pick_side,
        "lattice_centre_margin": (
            guess.model_view.predicted_margin
            if guess.model_view is not None
            else guess.guess_margin
        ),
        "pick_spread_line": guess.pick_spread_line,
        "pick_cover_probability": guess.pick_cover_probability,
        "pick_push_probability": guess.pick_push_probability,
        "consistency_note": guess.consistency_note,
        "method_note": "one lattice, one margin, one total -- see docs/tiebreaker.md",
        "generated_at_utc": generated_at.astimezone(UTC).isoformat(),
        "model_id": model_id,
    }


def _tiebreaker_card_line(guess: TiebreakerReport) -> str:

    total = guess.guess_home + guess.guess_away
    line = (
        f"**Tiebreaker (last game, {guess.away} at {guess.home}):** "
        f"{guess.home} {guess.guess_home} - {guess.away} {guess.guess_away}, "
        f"total {total} (market total {guess.consensus.total_line:g})"
    )
    if guess.consistency_note:
        line += f" -- {guess.consistency_note}"
    return line + ".\n\n"


def _publication_header(
    active: dict[str, Any],
    metadata: dict[str, Any],
    card: pd.DataFrame,
    nomination: BestPickNomination,
    overlay: OverlayResult | None = None,
    arrest_overlay: ArrestOverlayResult | None = None,
    production_overlay: FourOverlayCompositionResult | None = None,
) -> str:
    historical = active["historical_evaluation"]
    intervals = historical.get("intervals", {})
    week = intervals.get("week", {})
    season = int(metadata["season"])
    nfl_week = int(metadata["week"])
    method_label = (
        f"{humanize_identifier(str(active['feature_profile']))} "
        f"({humanize_identifier(str(active['method']))})"
    )
    return (
        f"## Current ATS forecast: {season} Week {nfl_week}\n\n"
        "> **Lines, injuries, depth charts, and model inputs may change before kickoff.** "
        "Regenerate and republish this card as the week approaches.\n\n"
        f"Active model: {method_label}. Its distinct close-graded chronological 2018-2025 "
        "evaluation classified "
        f"**{historical['correct']:,} of {historical['games']:,} non-push games correctly "
        f"({historical['accuracy']:.2%})**. The 95% range was "
        f"{week.get('lower', float('nan')):.2%}-{week.get('upper', float('nan')):.2%}. "
        "The model's baseline comparison is the separate opener-graded accuracy rule "
        "documented in `docs/opener_evaluation.md`.\n\n"
        + (
            _composition_note(production_overlay)
            if production_overlay is not None
            else (_overlay_note(overlay) if overlay is not None else "")
            + (_arrest_overlay_note(arrest_overlay) if arrest_overlay is not None else "")
        )
        + _best_pick_note(card, nomination)
    )


def _replace_readme_section(readme: str, section: str) -> str:
    block = f"{README_PREDICTIONS_START}\n{section.rstrip()}\n{README_PREDICTIONS_END}"
    if README_PREDICTIONS_START in readme or README_PREDICTIONS_END in readme:
        if readme.count(README_PREDICTIONS_START) != 1 or readme.count(README_PREDICTIONS_END) != 1:
            raise ValueError("README prediction markers must appear exactly once as a pair")
        before, remainder = readme.split(README_PREDICTIONS_START, maxsplit=1)
        _, after = remainder.split(README_PREDICTIONS_END, maxsplit=1)
        return before.rstrip() + "\n\n" + block + after
    paragraphs = readme.split("\n\n", maxsplit=2)
    if len(paragraphs) < 3:
        raise ValueError("README is too short to insert the current predictions section")
    return "\n\n".join((paragraphs[0], paragraphs[1], block, paragraphs[2]))


def published_tiebreaker_guess(
    data_root: Path,
    *,
    artifacts_root: Path,
    active: dict[str, Any],
    metadata: dict[str, Any],
    predictions: pd.DataFrame,
) -> TiebreakerReport:
    game = last_game_of_week(
        pd.read_parquet(newest_schedules_path(data_root)),
        int(metadata["season"]),
        int(metadata["week"]),
    )
    rows = predictions.loc[predictions["game_id"].eq(str(game["game_id"]))]
    if len(rows) != 1:
        raise TiebreakerConsistencyError("Published card must contain exactly one tiebreaker row")
    row = rows.iloc[0]
    return tiebreaker_report(
        data_root,
        artifacts_root=artifacts_root,
        game_id=str(game["game_id"]),
        forecast_row=row,
        forecast_model_id=metadata.get("active_model_id"),
        forecast_artifact=active.get("weekly_forecast", {}).get("artifact"),
        model_id=active.get("model_id"),
        published_pick_side="HOME" if float(row["home_cover_probability"]) >= 0.5 else "AWAY",
        frozen_spread=float(row["spread_line"]),
    )


def publish_active_predictions(
    artifacts_root: Path,
    *,
    destination: Path,
    readme_path: Path,
    data_root: Path | None = None,
    published_at: datetime | None = None,
    registry_root: Path | None = None,
    include_pick_explanation_lines: bool = False,
) -> dict[str, Any]:

    publish_instant = published_at or datetime.now(UTC)
    (
        active,
        metadata,
        card,
        nomination,
        overlay,
        arrest_overlay,
        production_overlay,
        raw_predictions,
        displayed_confidence,
    ) = _publication_context(
        artifacts_root,
        data_root,
        published_at=publish_instant,
        require_fresh_arrest_overlay=True,
    )
    source_report = report_for_publication(
        data_root=data_root,
        artifacts_root=artifacts_root,
        now=publish_instant,
        arrest_snapshot_at=arrest_overlay.snapshot_fetched_at_utc,
        arrest_snapshot_id=arrest_overlay.snapshot_id,
    )
    if source_report.state == SOURCE_STATE_BLOCKED:
        raise SourceFreshnessError(source_report.block_message())
    publish_compatibility = check_compatible(
        active, feature_table_manifest(metadata), forecast_metadata=metadata
    )
    publish_compatibility.refuse_if_incompatible(action="publish this card")
    timestamp = publish_instant.astimezone(UTC).isoformat()

    tiebreaker_guess: TiebreakerReport | None = None
    tiebreaker_skip_reason: str | None = None
    if data_root is not None:
        try:
            tiebreaker_guess = published_tiebreaker_guess(
                data_root,
                artifacts_root=artifacts_root,
                active=active,
                metadata=metadata,
                predictions=raw_predictions,
            )
        except TiebreakerConsistencyError as error:
            tiebreaker_skip_reason = f"consistency check refused: {error}"
        except (FileNotFoundError, OSError, ValueError, KeyError) as error:
            tiebreaker_skip_reason = str(error) or "tiebreaker guess unavailable"
    else:
        tiebreaker_skip_reason = "no data_root supplied"
    if tiebreaker_guess is None:
        (destination.parent / TIEBREAKER_ARTIFACT_FILENAME).unlink(missing_ok=True)
        linked_forecast = active_artifact_path(artifacts_root, active, "weekly_forecast")
        if linked_forecast is not None:
            (linked_forecast / TIEBREAKER_ARTIFACT_FILENAME).unlink(missing_ok=True)
    tiebreaker_card_line = (
        _tiebreaker_card_line(tiebreaker_guess) if tiebreaker_guess is not None else ""
    )
    tiebreaker_json_path: str | None = None
    if tiebreaker_guess is not None:
        tiebreaker_payload = _tiebreaker_json_payload(
            tiebreaker_guess,
            generated_at=publish_instant,
            model_id=active.get("model_id"),
            season=int(metadata["season"]),
            week=int(metadata["week"]),
            forecast_artifact=active.get("weekly_forecast", {}).get("artifact"),
        )
        atomic_json(tiebreaker_payload, destination.parent / TIEBREAKER_ARTIFACT_FILENAME)
        tiebreaker_json_path = str(destination.parent / TIEBREAKER_ARTIFACT_FILENAME)

    header = _publication_header(
        active,
        metadata,
        card,
        nomination,
        overlay,
        arrest_overlay,
        production_overlay,
    )
    table = card.to_markdown(index=False)
    heading = f"## Current ATS forecast: {metadata['season']} Week {metadata['week']}\n\n"
    published_at_text = publish_instant.astimezone(UTC).strftime("%Y-%m-%d %H:%M UTC")
    detail = (
        f"# NFL ATS predictions: {metadata['season']} Week {metadata['week']}\n\n"
        f"Published from the synchronized "
        f"{humanize_identifier(str(active['feature_profile']))} model, "
        f"{published_at_text}.\n\n"
        f"<!-- publication: model_id={active['model_id']} "
        f"published_at_utc={publish_instant.astimezone(UTC).isoformat()} -->\n\n"
        + header.removeprefix(heading)
        + table
        + "\n\n"
        + tiebreaker_card_line
        + source_report.summary_line()
        + "\n\n"
        + _decision_score_note(displayed_confidence)
    )

    played_card_lineage_path: str | None = None
    played_card_lineage_checks: tuple[str, ...] = ()
    played_card_overlay_lineage_count = 0
    played_card_tiebreaker_lineage_count = 0

    forecast_dir = active_artifact_path(artifacts_root, active, "weekly_forecast")
    if forecast_dir is not None:
        if tiebreaker_guess is not None:
            atomic_json(
                _tiebreaker_json_payload(
                    tiebreaker_guess,
                    generated_at=publish_instant,
                    model_id=active.get("model_id"),
                    season=int(metadata["season"]),
                    week=int(metadata["week"]),
                    forecast_artifact=active.get("weekly_forecast", {}).get("artifact"),
                ),
                forecast_dir / TIEBREAKER_ARTIFACT_FILENAME,
            )
        try:
            lineage_obj: CardLineage | None = read_card_lineage(forecast_dir)
        except (FileNotFoundError, OSError, ValueError, KeyError):
            lineage_obj = None

        played_base_lineage = lineage_obj
        if played_base_lineage is None:
            feature_profile = metadata.get("feature_profile") or active.get("feature_profile")
            if feature_profile:
                try:
                    played_base_lineage = build_card_lineage(
                        raw_predictions,
                        metadata,
                        active_model=active,
                        feature_columns=margin_feature_columns("market_residual", feature_profile),
                        display_fields=PUBLISHED_DISPLAY_FIELDS,
                    )
                except (KeyError, ValueError):
                    played_base_lineage = None

        if played_base_lineage is not None:
            overlay_sources: tuple[OverlaySource, ...] = ()
            if production_overlay is not None:
                overlay_sources = overlay_sources_from_composition(
                    production_overlay,
                    fallback_effective_timestamp=played_base_lineage.prediction_timestamp,
                )

            tiebreaker_sources: tuple[TiebreakerSource, ...] = (
                tiebreaker_lineage_sources(
                    tiebreaker_guess,
                    fallback_effective_timestamp=played_base_lineage.prediction_timestamp,
                )
                if tiebreaker_guess is not None
                else ()
            )

            played_card_lineage = extend_card_lineage_for_publication(
                played_base_lineage,
                overlay_sources=overlay_sources,
                tiebreaker_sources=tiebreaker_sources,
                prediction_timestamp=publish_instant,
                generated_at=publish_instant,
            )
            played_card_lineage_checks = validate_card_lineage(played_card_lineage)
            write_card_lineage(played_card_lineage, destination.parent)
            played_card_lineage_path = str(destination.parent / LINEAGE_FILENAME)
            played_card_overlay_lineage_count = len(overlay_sources)
            played_card_tiebreaker_lineage_count = len(tiebreaker_sources)

        all_game_ids = raw_predictions["game_id"].astype(str).tolist()
        overlays_by_game: dict[str, tuple[OverlayFiring, ...]] = dict.fromkeys(all_game_ids, ())
        if production_overlay is not None:
            for game_id in all_game_ids:
                overlays_by_game[game_id] = overlay_firings_from_composition(
                    production_overlay, game_id
                )
        else:
            for coach_flip in overlay.flips:
                key = str(coach_flip.game_id)
                overlays_by_game[key] = (
                    *overlays_by_game.get(key, ()),
                    overlay_firing_from_coach_fade_flip(coach_flip),
                )
            for arrest_flip in arrest_overlay.flips:
                key = str(arrest_flip.game_id)
                overlays_by_game[key] = (
                    *overlays_by_game.get(key, ()),
                    overlay_firing_from_arrest_flip(arrest_flip),
                )

        revisions = load_pick_revisions(artifacts_root)
        week_revisions = revisions.loc[
            revisions["season"].astype(int).eq(int(metadata["season"]))
            & revisions["week"].astype(int).eq(int(metadata["week"]))
        ]
        refresh_changes_by_game: dict[str, RefreshChangeInput] = {}
        if not week_revisions.empty:
            latest_revisions = (
                week_revisions.sort_values("revision_recorded_at_utc")
                .groupby("game_id", as_index=False)
                .tail(1)
            )
            for _, revision_row in latest_revisions.iterrows():
                adapted = refresh_change_from_pick_revision(
                    cast(dict[str, Any], revision_row.to_dict())
                )
                if adapted is not None:
                    refresh_changes_by_game[str(revision_row["game_id"])] = adapted

        explanations = explain_card(
            cast(list[dict[str, Any]], raw_predictions.to_dict("records")),
            lineage=lineage_obj,
            source_report=source_report,
            overlays_by_game=overlays_by_game,
            refresh_changes_by_game=refresh_changes_by_game,
            waterfall_by_game=load_waterfall_feed(artifacts_root),
            key_line_games=key_line_touched_games(metadata),
        )
        atomic_json(explanations_to_dict(explanations), forecast_dir / "explanations.json")
        if include_pick_explanation_lines:
            detail = detail + "\n\n" + render_explanations_markdown(explanations)
        atomic_json(source_report.to_metadata(), forecast_dir / "source_policy.json")
        atomic_json(displayed_confidence.to_dict(), forecast_dir / DISPLAYED_CONFIDENCE_FILENAME)

    atomic_json(source_report.to_metadata(), destination.parent / "source_policy.json")

    atomic_text(detail, destination)
    readme_section = (
        header
        + table
        + f"\n\n[Open the standalone card]({destination.as_posix()}) for provenance and "
        "interpretation.\n"
    )
    current_readme = readme_path.read_text(encoding="utf-8")
    updated_readme = _replace_readme_section(current_readme, readme_section)
    updated_readme = apply_generated_state_blocks(
        updated_readme, artifacts_root=artifacts_root, registry_root=registry_root
    )
    atomic_text(updated_readme, readme_path)
    return {
        "model_id": active["model_id"],
        "season": int(metadata["season"]),
        "week": int(metadata["week"]),
        "games": len(card),
        "best_pick_game_id": nomination.active_game_id,
        "best_pick_prospective_input": {
            "predictions": raw_predictions[["game_id", "home_cover_probability"]].to_dict(
                orient="records"
            ),
            "pool": (
                nomination.v2_result.dispersion.frame.to_dict(orient="records")
                if nomination.v2_result is not None
                else [
                    {"game_id": str(game_id), "pool_pass": True, "spread_std": None}
                    for game_id in raw_predictions["game_id"]
                ]
            ),
        },
        "best_pick_tied": bool(nomination.active_tie_note),
        "best_pick_nomination_rule": nomination.active_rule,
        "best_pick_nomination_v1_game_id": nomination.v1_game_id,
        "best_pick_nomination_v2_game_id": (
            nomination.v2_result.game_id if nomination.v2_result is not None else None
        ),
        "best_pick_nomination_unrestricted_game_id": (
            nomination.v2_result.base_game_id if nomination.v2_result is not None else None
        ),
        "best_pick_nomination_spread_threshold": (
            nomination.v2_result.spread_threshold if nomination.v2_result is not None else None
        ),
        "best_pick_nomination_spread_fallback": (
            nomination.v2_result.spread_fallback if nomination.v2_result is not None else False
        ),
        "best_pick_nomination_v2_available": nomination.v2_result is not None,
        "historical_accuracy": active["historical_evaluation"]["accuracy"],
        "destination": str(destination),
        "readme": str(readme_path),
        "published_at_utc": timestamp,
        "source_policy": source_report.to_metadata(),
        "overlay_enabled": overlay.enabled,
        "overlay_flip_count": overlay.flip_count,
        "overlay_flipped_game_ids": [flip.game_id for flip in overlay.flips],
        "overlay_both_year_one_game_ids": list(overlay.both_year_one_games),
        "player_arrests_overlay_enabled": arrest_overlay.enabled,
        "player_arrests_overlay_flip_count": arrest_overlay.flip_count,
        "player_arrests_overlay_flipped_game_ids": [flip.game_id for flip in arrest_overlay.flips],
        "decision_policy_id": production_overlay.policy_id if production_overlay else None,
        "decision_policy_fingerprint": (
            production_overlay.policy_fingerprint if production_overlay else None
        ),
        "production_overlay_flip_count": (
            production_overlay.flip_count if production_overlay else 0
        ),
        "production_overlay_flipped_game_ids": (
            list(production_overlay.union_flipped_game_ids) if production_overlay else []
        ),
        "production_overlay_overlap_game_ids": (
            list(production_overlay.overlapping_game_ids) if production_overlay else []
        ),
        **stamp(KIND_CARD, {}),
        "artifact_contract_compatibility": publish_compatibility.to_dict(),
        "pick_explanations_path": (
            str(forecast_dir / "explanations.json") if forecast_dir is not None else None
        ),
        "played_card_lineage_path": played_card_lineage_path,
        "played_card_lineage_checks_passed": list(played_card_lineage_checks),
        "tiebreaker_json_path": tiebreaker_json_path,
        "tiebreaker_skip_reason": tiebreaker_skip_reason,
        "card_metadata": {
            "played_card_lineage_path": played_card_lineage_path,
            "played_card_lineage_checks_passed": list(played_card_lineage_checks),
            "played_card_overlay_lineage_count": played_card_overlay_lineage_count,
            "played_card_tiebreaker_lineage_count": played_card_tiebreaker_lineage_count,
        },
    }
