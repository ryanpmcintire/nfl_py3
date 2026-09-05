"""Publish a synchronized weekly ATS card as tracked GitHub-friendly Markdown."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import pandas as pd

from nfl_ats.active_model import active_artifact_path, load_active_ats_model
from nfl_ats.artifact_contracts import KIND_CARD, check_compatible, stamp
from nfl_ats.best_pick_nomination import nominate_v2
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
from nfl_ats.four_overlay_composition import FourOverlayCompositionResult
from nfl_ats.io import atomic_json, atomic_text
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
from nfl_ats.pick_refresh import load_pick_revisions
from nfl_ats.player_arrests_back_side_overlay import (
    ArrestOverlayResult,
    arrest_overlay_disclosure_note,
)
from nfl_ats.public_board import humanize_identifier, load_waterfall_feed
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

#: Filename for the persisted tiebreaker guess -- read by
#: ``nfl_ats.board_content._load_tiebreaker_view`` (UI-20(g)) and by the
#: board assistant's tiebreaker intent. Written BOTH beside the linked
#: forecast artifact (``forecast_dir``, matching ``explanations.json`` /
#: ``source_policy.json``) and beside the published card
#: (``destination.parent``, matching ``lineage.json``), so a reader of
#: either location finds the SAME number.
TIEBREAKER_ARTIFACT_FILENAME = "tiebreaker.json"

README_PREDICTIONS_START = "<!-- CURRENT_PREDICTIONS:START -->"
README_PREDICTIONS_END = "<!-- CURRENT_PREDICTIONS:END -->"

#: Marks the one game per regular-season week the pool scores as the Best Pick.
#: The card is what the user reads at pick time, so the nomination has to be
#: visible on it -- persisting it in the ledger (POL-10) answers "what did we
#: choose?" months later, but only this answers "what do I enter today?".
BEST_PICK_MARK = "★ "


def _line(value: float) -> str:
    return "PK" if value == 0.0 else f"{value:+g}"


def _published_card(predictions: pd.DataFrame, best_pick_id: str | None = None) -> pd.DataFrame:
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
    card["Decision score"] = card["home_cover_probability"].where(
        home_pick, 1.0 - card["home_cover_probability"]
    )
    card["Matchup"] = card["away_team"] + " at " + card["home_team"]
    card["_gameday"] = pd.to_datetime(card["gameday"], errors="raise")
    card["Date"] = card["_gameday"].dt.strftime("%a, %b %d")
    card = card.sort_values(["_gameday", "game_id"], kind="stable")
    published = card[["Date", "Matchup", "ATS prediction", "Decision score"]].copy()
    published["Decision score"] = published["Decision score"].map(lambda value: f"{value:.1%}")
    return published


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
    # Best Pick is selected on the UN-overlaid predictions (both rules) -- the
    # overlay must never influence which game is nominated, only which side a
    # game's forced pick lands on. Both levers are resolved through the one
    # shared implementation every surface uses (see nfl_ats.card_view);
    # ``nominate_v2_fn`` threads THIS module's own (patchable) ``nominate_v2``
    # name through rather than card_view's, so tests that monkeypatch
    # ``publishing.nominate_v2`` keep working unchanged.
    view = resolve_card_view(
        predictions,
        sweep,
        metadata,
        data_root=data_root,
        now=published_at,
        require_fresh_arrest_overlay=require_fresh_arrest_overlay,
        nominate_v2_fn=nominate_v2,
    )
    card = _published_card(view.predictions, view.nomination.active_game_id)
    return (
        active,
        metadata,
        card,
        view.nomination,
        view.overlay,
        view.arrest_overlay,
        view.production_overlay,
        # ENG-12: the raw (overlay-applied but display-unformatted) per-game
        # frame -- publish_active_predictions needs game_id/home_team/
        # away_team/spread_line/home_cover_probability to build each pick's
        # explanation, none of which survive _published_card's own
        # Date/Matchup/ATS-prediction/Decision-score projection.
        view.predictions,
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
        "**Production policy active:** four situational rules run independently against "
        "the computer's first pick and flip it once when any one of them fires: coach fade, "
        "division revenge, player arrests, and the spread-gap zone. "
        f"This week they changed {composition.flip_count} pick{plural}. "
        "Its archive score (see the This Week board's headline) is the best of 127 similar "
        "combinations scored on the very "
        "games that chose it, so treat it as a ceiling, never an expectation. The de-inflated "
        f"planning estimate for the played card is {PLAYED_CARD_EXPECTATION_HERO}: four real "
        "out-of-sample split-half selections average +1.30 accuracy points, and shrinking the "
        "raw archive gain by the measured 0.59-0.64 selection-shrinkage factor lands "
        "in the same place. A separate re-check of the selection step "
        "itself measured 0.00 points, so treat the estimate as an upper-middle read, not a "
        "floor. Paired prospective tracking against the prior coach-to-arrests chain begins at "
        f"the Week 1 lock. Rules: {members}. See docs/overlay_subset_holdout_v2.md.\n\n"
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
    """The persisted tiebreaker artifact (UI-20(g) extension, 2026-09-05):
    read by ``nfl_ats.board_content._load_tiebreaker_view`` and the board
    assistant's tiebreaker intent -- never recomputed by either reader.

    ``implied_margin`` is deliberately ``guess_home - guess_away`` (the
    margin the PUBLISHED SCORE actually carries), not ``guess.guess_margin``
    (the pre-lattice blended figure) -- the whole point of the one-lattice
    fix is that the displayed score and the displayed margin can never
    disagree.
    """

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
        # MOD-17 (docs/tiebreaker.md "one lattice, one margin, one total";
        # nfl_ats.served_total): "blended_total" above IS the served total
        # (same value, kept for readers that predate this switch);
        # "served_total"/"served_total_method" name it explicitly, and
        # "comparison_total_blend_k01" always reports today's blend
        # arithmetic regardless of which method served, so a report never
        # hides the arm it did not serve.
        "served_total": guess.served_total,
        "served_total_method": guess.served_total_method,
        "comparison_total_blend_k01": guess.comparison_total_blend_k01,
        "implied_margin": guess.guess_home - guess.guess_away,
        "pick_side": guess.pick_side,
        "pick_spread_line": guess.pick_spread_line,
        "pick_cover_probability": guess.pick_cover_probability,
        "pick_push_probability": guess.pick_push_probability,
        "consistency_note": guess.consistency_note,
        "method_note": "one lattice, one margin, one total -- see docs/tiebreaker.md",
        "generated_at_utc": generated_at.astimezone(UTC).isoformat(),
        "model_id": model_id,
    }


def _tiebreaker_card_line(guess: TiebreakerReport) -> str:
    """The ONE tiebreaker line added under the picks table (UI-20(g)
    extension). Never independently rounds or recomputes anything --
    every number here is read straight off ``guess``."""

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
    """Compute without publishing, using the resolved card and verified forecast identity."""
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
    """Publish the active card and update the README from the same rendered table.

    ``data_root`` locates the local nflverse schedule snapshot the year-1-coach
    fade overlay (``docs/coach_fade_overlay.md``) is derived from, AND the
    local market snapshot store the v2 Best Pick nomination rule
    (``nfl_ats.best_pick_nomination``, POL-09) reads its cross-book opener
    dispersion from. Coach and nomination inputs retain their documented
    fallbacks, but publication always requires a current, complete,
    hash-verified player-arrest snapshot. There is no public fail-open switch
    for the production card.

    ``registry_root`` locates ``weak_signals.json`` / ``rotation_registry.json``
    for the README's generated research-state block (see
    ``nfl_ats.readme_state``); this publisher also refreshes that block and the
    generated active-model-state block in the same README write, alongside the
    ``CURRENT_PREDICTIONS`` card table this function has always owned.
    ``registry_root=None`` (the default for direct callers/tests) renders that
    block as "not available" rather than reading an ambient path.

    ENG-12: a per-pick ``explanations.json`` (market line, this game's own
    model probability, fired overlays, source freshness, and any recorded
    Tuesday-to-refresh change -- see ``nfl_ats.card_explanation``) is always
    written beside the linked forecast artifact. ``include_pick_explanation_lines``
    (default ``False``) additionally gates a short explanation line per pick
    appended to the tracked Markdown card itself; it defaults off so the
    existing card-writer tests need no changes.
    """

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
    ) = _publication_context(
        artifacts_root,
        data_root,
        published_at=publish_instant,
        require_fresh_arrest_overlay=True,
    )
    # ENG-14 source outage / degraded-mode policy (docs/source_freshness_policy.md).
    # Read-only over the local tree, evaluated AFTER `_publication_context` so the
    # arrest gate that already refuses a missing/stale/unverified snapshot has run
    # first and this layer reports its verified instant rather than re-deriving one.
    # It refuses only for a source whose consumer is ALREADY fail-closed, so no
    # currently-permitted publish path becomes newly blockable here.
    source_report = report_for_publication(
        data_root=data_root,
        artifacts_root=artifacts_root,
        now=publish_instant,
        arrest_snapshot_at=arrest_overlay.snapshot_fetched_at_utc,
        arrest_snapshot_id=arrest_overlay.snapshot_id,
    )
    if source_report.state == SOURCE_STATE_BLOCKED:
        raise SourceFreshnessError(source_report.block_message())
    # ENG-09: refuse to publish a card built on a feature table whose stamped
    # version contradicts what the active model was fit on, or on a forecast
    # carrying an unrecognized schema version. Evaluated after the arrest and
    # source gates above so an already-blocked publish still reports THAT
    # reason first. legacy_unversioned (either artifact predates this
    # contract layer) stays a warning, not a refusal -- see
    # nfl_ats.artifact_contracts.
    publish_compatibility = check_compatible(
        active, feature_table_manifest(metadata), forecast_metadata=metadata
    )
    publish_compatibility.refuse_if_incompatible(action="publish this card")
    timestamp = publish_instant.astimezone(UTC).isoformat()

    # POL-12 (2026-09-05 owner mandate: "our project over/under total needs
    # to line up with our spread prediction"). Computed HERE, before the
    # card markdown is assembled, so the SAME guess object backs the card's
    # tiebreaker line, `tiebreaker.json`, and the lineage records further
    # below -- never three independently-computed numbers that could
    # silently disagree (`docs/tiebreaker.md`'s "one lattice, one margin,
    # one total"). A ``TiebreakerConsistencyError`` -- the projected margin
    # contradicts the card's own pick, or the projected total drifted more
    # than a point from the served total -- degrades to "not published"
    # for the tiebreaker ONLY, the same fail-open contract every other
    # optional artifact on this publish path already follows; it never
    # blocks the pool's card itself, which must publish regardless.
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
    # Beside the card (destination.parent, matching lineage.json's own
    # location) -- written unconditionally on a real guess, whether or not
    # a forecast directory resolves below, so the reader closest to the
    # published card (the This Week panel / the board assistant) never has
    # to know which forecast produced it.
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
        # Machine-readable publication record for ``nfl_ats.handoff`` (an HTML
        # comment: invisible when the Markdown renders, so the reader-facing
        # sentence above stays free of ids and ISO timestamps).
        f"<!-- publication: model_id={active['model_id']} "
        f"published_at_utc={publish_instant.astimezone(UTC).isoformat()} -->\n\n"
        + header.removeprefix(heading)
        + table
        + "\n\n"
        + tiebreaker_card_line
        + source_report.summary_line()
        + "\n\n"
        "`Decision score` is the computer's own probability, oriented to the final pick. "
        "On a flip it is a mirrored decision-strength score, not a newly calibrated "
        "probability for that side; it is also not historical accuracy.\n"
    )

    # ENG-12: card-level explanation contract (nfl_ats.card_explanation).
    # Additive only -- explanations.json is a NEW file written beside the
    # forecast artifact; the inline card line is gated behind
    # ``include_pick_explanation_lines`` (default False) so the existing
    # card-writer tests need no changes. ``lineage.json`` and the
    # pick-revision ledger are both OPTIONAL artifacts read read-only here
    # and degraded from when absent, matching every other optional artifact
    # already on this publish path.
    # ENG-24: the played card's own lineage -- the forecast's lineage.json (or,
    # when absent, a fresh equivalent built from the same inputs margin-predict/
    # predict would have used) extended with the overlay and tiebreaker records
    # that only exist at publish time. Declared here (rather than only inside
    # the ``forecast_dir is not None`` block below) so the publish summary can
    # report it either way -- ``None``/``()`` on the unreachable path where no
    # forecast artifact resolves, matching ``pick_explanations_path`` below.
    played_card_lineage_path: str | None = None
    played_card_lineage_checks: tuple[str, ...] = ()
    played_card_overlay_lineage_count = 0
    played_card_tiebreaker_lineage_count = 0

    forecast_dir = active_artifact_path(artifacts_root, active, "weekly_forecast")
    if forecast_dir is not None:
        # Beside the forecast too (matching explanations.json / source_policy.json),
        # so nfl_ats.board_content._load_tiebreaker_view's first, preferred
        # read location has the SAME payload just written beside the card.
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

        # ENG-24: fall back to a freshly built lineage when the forecast never
        # got one (an artifact predating the ENG-16 wiring, or a legacy path) --
        # same inputs and feature contract margin-predict/predict use, so the
        # played card is never left with NO base lineage to extend just because
        # its forecast file happens to lack one.
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

            # The pool's tiebreaker game is identifiable from data alone (the
            # week's last REG kickoff -- nfl_ats.tiebreaker.last_game_of_week,
            # used by tiebreaker_report whenever season/week are given). Reuses
            # the SAME ``tiebreaker_guess`` already computed above (never a
            # second, independently-timed computation that could disagree
            # with the card's own tiebreaker line / ``tiebreaker.json``); a
            # ``None`` guess (see ``tiebreaker_skip_reason`` above) degrades
            # to no tiebreaker lineage records, same fail-open contract as
            # the coach-fade snapshot fallback above.
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
            # Fail closed: a played card whose overlay/tiebreaker records are
            # malformed, or whose decision-bearing fields are incomplete, must
            # never reach the artifact directory (mirrors the artifact-contract
            # and source-freshness gates already enforced above in this
            # function).
            played_card_lineage_checks = validate_card_lineage(played_card_lineage)
            write_card_lineage(played_card_lineage, destination.parent)
            played_card_lineage_path = str(destination.parent / LINEAGE_FILENAME)
            played_card_overlay_lineage_count = len(overlay_sources)
            played_card_tiebreaker_lineage_count = len(tiebreaker_sources)

        # Every game gets an entry -- not only the flipped ones -- so an
        # unflipped pick reads as "evaluated, none fired" (measured, an
        # empty tuple) rather than "overlay evaluation not supplied" (no_data).
        # ``production_overlay.games``/``*_overlay.flips`` list ONLY the
        # flipped subset by construction (see nfl_ats.four_overlay_composition
        # / nfl_ats.coach_fade_overlay), so seeding every game id first is
        # required, not defensive padding.
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

        # UI-20 explanation rewrite (2026-09-05): the "what tips it" sentence
        # names the biggest football-terms factors off the SAME real
        # attribution-waterfall feed the site's own dive panels read
        # (public_board.load_waterfall_feed) -- fail-open to an empty map
        # (every game then simply omits that one sentence) like every other
        # optional artifact on this publish path.
        explanations = explain_card(
            cast(list[dict[str, Any]], raw_predictions.to_dict("records")),
            lineage=lineage_obj,
            source_report=source_report,
            overlays_by_game=overlays_by_game,
            refresh_changes_by_game=refresh_changes_by_game,
            waterfall_by_game=load_waterfall_feed(artifacts_root),
        )
        atomic_json(explanations_to_dict(explanations), forecast_dir / "explanations.json")
        if include_pick_explanation_lines:
            detail = detail + "\n\n" + render_explanations_markdown(explanations)
        # ENG-34: persist the ENG-14 source-policy block beside the forecast
        # artifact, additively -- `source_report.to_metadata()` was already
        # computed above and embedded in `source_report.summary_line()`
        # further down; this is the SAME object, written verbatim so a
        # later reader (nfl_ats.board_content._load_source_policy_view)
        # never has to re-derive it or re-run publish. Never touches
        # metadata.json: that file's digest is recorded by the lock-day
        # package and replay, so it must stay exactly as margin-predict
        # wrote it.
        atomic_json(source_report.to_metadata(), forecast_dir / "source_policy.json")

    # ENG-34: the same block, also written beside the PUBLISHED card next to
    # lineage.json (ENG-24, destination.parent) -- unconditional on
    # forecast_dir/lineage above, since source_report is computed
    # unconditionally near the top of this function and this copy is what a
    # reader of the published card (not the forecast artifact) opens.
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
        "best_pick_tied": bool(nomination.active_tie_note),
        # POL-09 2026-08-18: both rules' nominations, so the season can be
        # audited old-vs-new even though only `best_pick_nomination_rule`'s
        # rule is actually marked on the card. v2 is pinned/tracked
        # separately, in full, via the challenger ledger (see
        # nfl_ats.best_pick_nomination.record_nomination_challenger_decisions).
        "best_pick_nomination_rule": nomination.active_rule,
        "best_pick_nomination_v1_game_id": nomination.v1_game_id,
        "best_pick_nomination_v2_game_id": (
            nomination.v2_result.game_id if nomination.v2_result is not None else None
        ),
        "best_pick_nomination_v2_available": nomination.v2_result is not None,
        "historical_accuracy": active["historical_evaluation"]["accuracy"],
        "destination": str(destination),
        "readme": str(readme_path),
        "published_at_utc": timestamp,
        # ENG-14: which sources this card was built from, and in what state.
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
        # ENG-09: this publish summary's own schema/builder-version contract,
        # plus the compatibility report the publish path already refused on
        # above -- surfaced here (rather than only raising) so a caller can
        # see legacy_unversioned warnings without them ever blocking a publish.
        **stamp(KIND_CARD, {}),
        "artifact_contract_compatibility": publish_compatibility.to_dict(),
        # ENG-12: where the per-pick explanation contract was written, or
        # None on the (validation-blocked) path where no forecast artifact
        # could be resolved.
        "pick_explanations_path": (
            str(forecast_dir / "explanations.json") if forecast_dir is not None else None
        ),
        # ENG-24: the PLAYED card's own lineage.json -- fired overlays and
        # tiebreaker inputs on top of the forecast's own decision-bearing
        # fields -- written beside this publish's own ``destination``, never
        # overwriting the forecast's file. ``None``/``()``/``0`` on the
        # (validation-blocked) path where no forecast artifact resolves, or
        # where neither the forecast's own lineage.json nor a fresh equivalent
        # could be built.
        "played_card_lineage_path": played_card_lineage_path,
        "played_card_lineage_checks_passed": list(played_card_lineage_checks),
        # POL-12 (2026-09-05): the persisted tiebreaker guess -- ``None``
        # path / a non-``None`` ``tiebreaker_skip_reason`` means the
        # consistency check refused (or the guess was otherwise
        # unavailable) and neither ``tiebreaker.json`` nor the card line
        # was written; the pool's card itself still published regardless.
        "tiebreaker_json_path": tiebreaker_json_path,
        "tiebreaker_skip_reason": tiebreaker_skip_reason,
        "card_metadata": {
            "played_card_lineage_path": played_card_lineage_path,
            "played_card_lineage_checks_passed": list(played_card_lineage_checks),
            "played_card_overlay_lineage_count": played_card_overlay_lineage_count,
            "played_card_tiebreaker_lineage_count": played_card_tiebreaker_lineage_count,
        },
    }
