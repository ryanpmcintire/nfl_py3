"""Publish a synchronized weekly ATS card as tracked GitHub-friendly Markdown."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from nfl_ats.active_model import active_artifact_path, load_active_ats_model
from nfl_ats.best_pick_nomination import nominate_v2
from nfl_ats.card_view import BestPickNomination, resolve_card_view
from nfl_ats.coach_fade_overlay import OverlayResult, overlay_disclosure_note
from nfl_ats.dashboard.findings_content import PLAYED_CARD_EXPECTATION_HERO
from nfl_ats.four_overlay_composition import FourOverlayCompositionResult
from nfl_ats.io import atomic_text
from nfl_ats.player_arrests_back_side_overlay import (
    ArrestOverlayResult,
    arrest_overlay_disclosure_note,
)

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
    members = ", ".join(member.member_id for member in composition.members)
    plural = "" if composition.flip_count == 1 else "s"
    return (
        "**Production policy active:** the frozen four-member policy evaluates coach fade, "
        "division revenge, player arrests, and the spread-gap zone independently against "
        "the raw model pick, then flips once when any member fires. "
        f"This week it changed {composition.flip_count} pick{plural}; policy "
        f"`{composition.policy_id}` (`{composition.policy_fingerprint[:16]}`). "
        "Its 55.42% archive score is the best of 127 correlated subsets scored on the very "
        "games that chose it, so it is a ceiling and never an expectation. The de-inflated "
        f"planning estimate for the played card is {PLAYED_CARD_EXPECTATION_HERO}: four real "
        "out-of-sample split-half selections average +1.30 accuracy points, and shrinking the "
        "+2.06-point archive gain by the measured 0.59-0.64 selection-shrinkage factor lands "
        "in the same place. A separate leave-one-season-out re-check of the selection step "
        "itself measured 0.00 points, so treat the estimate as an upper-middle read, not a "
        "floor. Paired prospective tracking against the prior coach-to-arrests chain begins at "
        f"the Week 1 lock. Members: {members}. See docs/overlay_subset_holdout_v2.md.\n\n"
    )


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
    return (
        f"## Current ATS forecast: {season} Week {nfl_week}\n\n"
        "> **Early, mutable research preview.** Lines, injuries, depth charts, and model "
        "inputs may change before kickoff. Regenerate and republish this card as the week "
        "approaches.\n\n"
        f"Active model: `{active['method']}` with `{active['feature_profile']}` features "
        f"(`{active['model_id']}`). Its distinct close-graded chronological 2018-2025 "
        "evaluation classified "
        f"**{historical['correct']:,} of {historical['games']:,} non-push games correctly "
        f"({historical['accuracy']:.2%})**. The week-blocked 95% interval was "
        f"{week.get('lower', float('nan')):.2%}-{week.get('upper', float('nan')):.2%}. "
        "The model baseline is the separate opener-graded probability rule "
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


def publish_active_predictions(
    artifacts_root: Path,
    *,
    destination: Path,
    readme_path: Path,
    data_root: Path | None = None,
    published_at: datetime | None = None,
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
    """

    publish_instant = published_at or datetime.now(UTC)
    active, metadata, card, nomination, overlay, arrest_overlay, production_overlay = (
        _publication_context(
            artifacts_root,
            data_root,
            published_at=publish_instant,
            require_fresh_arrest_overlay=True,
        )
    )
    timestamp = publish_instant.astimezone(UTC).isoformat()
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
    detail = (
        f"# NFL ATS predictions: {metadata['season']} Week {metadata['week']}\n\n"
        f"Published from synchronized model `{active['model_id']}` at `{timestamp}`.\n\n"
        + header.removeprefix(heading)
        + table
        + "\n\n"
        "`Decision score` is the raw model probability oriented to the final policy side. "
        "On a policy flip it is a mirrored decision-strength score, not a newly calibrated "
        "probability for that side; it is also not historical accuracy. This is research "
        "output, not a wagering recommendation.\n"
    )
    atomic_text(detail, destination)
    readme_section = (
        header
        + table
        + f"\n\n[Open the standalone card]({destination.as_posix()}) for provenance and "
        "interpretation.\n"
    )
    current_readme = readme_path.read_text(encoding="utf-8")
    atomic_text(_replace_readme_section(current_readme, readme_section), readme_path)
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
    }
