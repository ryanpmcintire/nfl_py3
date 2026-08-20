"""Year-1 head-coach fade overlay: a pick-level flip on top of the active card.

Owner decision 2026-08-18 (``docs/coach_fade_overlay.md``): weeks 1-8 of the
2026 season, the published pool card flips a game's forced pick from the
model's own choice to the opposite side whenever the model sides WITH a team
whose head coach is in year 1 of their tenure AND the opponent's coach is NOT
(the "clean case" -- see ``docs/hc_year_one_fade.md`` and ROADMAP PER-07).
Games where BOTH coaches are year-1 (no measured direction) and weeks 9+
(outside the registered window) are left untouched.

This module is intentionally **pick-level, not feature-level**: it transforms
the model's already-computed ``home_cover_probability`` for a handful of
games, after prediction, rather than adding a training-time column. That
sidesteps the frozen-inputs invariant entirely -- see the module docstring's
"Design choice" note in ``docs/coach_fade_overlay.md`` -- because it never
touches a feature table, a training frame, or a stored model artifact; the
active model's own historical evaluation is unaffected by this file existing.

Two things live here:

1. :func:`year_one_by_game` -- the pregame-safe, DATA-DERIVED year-1 flag,
   built straight from ``schedules.parquet``'s ``home_coach``/``away_coach``
   columns (the same source ``scripts/hc_year_one_fade.py`` and the Week 1
   2026 decision brief used), never a hardcoded team list.
2. :func:`apply_coach_fade_overlay` -- the pick-level transform, plus
   :func:`overlay_disclosure_note` for the plain-English provenance sentence
   the published card discloses, mirroring
   ``nfl_ats.best_pick.best_pick_tie_note``.

:func:`record_overlay_challenger_decisions` writes the coach-only arm to the
prospective challenger ledger (``nfl_ats.prospective_scoring``). The primary
paper ledger freezes both the raw ``model_pick_side`` and final played side;
``prospective-score`` derives the raw-model control from that frozen column.
Together those rows score the coach transform cleanly even after a later
production overlay is composed on top of it.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from nfl_ats.active_model import active_artifact_path, load_active_ats_model
from nfl_ats.clv import refuse_if_outside_recording_lock_window
from nfl_ats.constants import TEAM_ABBREVIATION_ALIASES
from nfl_ats.data import DataContractError
from nfl_ats.io import atomic_parquet
from nfl_ats.prospective_scoring import (
    ACTIVE_CHALLENGER_STATUS,
    CHALLENGER_DECISION_COLUMNS,
    artifact_model_config,
    challenger_ledger_path,
    config_fingerprint,
    find_challenger,
    load_challenger_decisions,
)
from nfl_ats.provenance import sha256_file
from nfl_ats.snapshots import latest_snapshot, load_snapshot

#: Owner decision 2026-08-18 (docs/coach_fade_overlay.md): play the clean-case
#: overlay for real. Flipping this off restores the un-overlaid card wherever
#: the overlay is wired in; the challenger ledger still records both arms
#: regardless of this switch, since it reads the flag itself, not this
#: constant's effect on the published card.
OVERLAY_ENABLED = True

#: The registered measurement (ROADMAP PER-07, docs/hc_year_one_fade.md) only
#: covers weeks 1-8; the overlay has no claim outside that window.
OVERLAY_WEEK_MAX = 8

#: Registered in artifacts/prospective/challengers.json.
CHALLENGER_ID = "hc_year_one_fade_overlay"


def _canonical_team(team: pd.Series) -> pd.Series:
    return team.astype(str).map(lambda code: TEAM_ABBREVIATION_ALIASES.get(code, code))


def team_season_primary_coach(schedules: pd.DataFrame) -> pd.DataFrame:
    """The modal credited REG-season coach for every (team, season) pair.

    Pregame-safe by construction: this aggregates a season taken as a WHOLE,
    and :func:`year_one_by_game` only ever looks it up for a season STRICTLY
    BEFORE the game being flagged. It is never used to describe the CURRENT
    season a game belongs to -- that would reintroduce the within-season
    lookahead ``docs/hc_year_one_fade.md`` flags as the reason a per-game
    field is required instead of a same-season aggregate.
    """

    required = {"home_team", "away_team", "home_coach", "away_coach", "season", "game_type"}
    missing = sorted(required.difference(schedules.columns))
    if missing:
        raise DataContractError(
            f"schedules is missing columns for coach tenure: {', '.join(missing)}"
        )
    reg = schedules.loc[schedules["game_type"].astype(str).eq("REG")]
    home = pd.DataFrame(
        {
            "team": _canonical_team(reg["home_team"]),
            "season": reg["season"].astype(int),
            "coach": reg["home_coach"],
        }
    )
    away = pd.DataFrame(
        {
            "team": _canonical_team(reg["away_team"]),
            "season": reg["season"].astype(int),
            "coach": reg["away_coach"],
        }
    )
    long = pd.concat([home, away], ignore_index=True)
    long = long.loc[long["coach"].notna()]
    primary = (
        long.groupby(["team", "season"])["coach"]
        .agg(lambda values: values.value_counts().idxmax())
        .reset_index()
        .rename(columns={"coach": "primary_coach"})
    )
    return primary


def year_one_by_game(schedules: pd.DataFrame) -> pd.DataFrame:
    """One row per ``game_id``: ``year_one_home``/``year_one_away``, pregame-safe.

    A side is "year 1" when both hold:

    1. its franchise has an OBSERVED, immediately-prior REG season in
       ``schedules`` (``season - 1``, looked up directly rather than via a
       shift, so a franchise's first season in the data, an expansion team, or
       any gap year is excluded rather than guessed at); and
    2. THIS game's own credited coach (``home_coach``/``away_coach`` -- a
       pregame-known fact once the schedule snapshot carries it) differs from
       that PRIOR season's modal coach.

    Using the game's own coach field (never a same-season aggregate) is what
    keeps this leak-safe *within* a season: the spec in
    ``docs/hc_year_one_fade.md`` ("What needs to be built") calls this out
    explicitly as the production-safe design, distinct from the season-mode
    shortcut the research script and the Week 1 brief used.
    """

    required = {
        "game_id",
        "season",
        "game_type",
        "home_team",
        "away_team",
        "home_coach",
        "away_coach",
    }
    missing = sorted(required.difference(schedules.columns))
    if missing:
        raise DataContractError(
            f"schedules is missing columns for coach tenure: {', '.join(missing)}"
        )

    primary = team_season_primary_coach(schedules)

    frame = pd.DataFrame(
        {
            "game_id": schedules["game_id"].astype(str),
            "season": schedules["season"].astype(int),
            "home_team": _canonical_team(schedules["home_team"]),
            "away_team": _canonical_team(schedules["away_team"]),
            "home_coach": schedules["home_coach"],
            "away_coach": schedules["away_coach"],
        }
    )
    frame["prior_season"] = frame["season"] - 1

    home_prior = primary.rename(
        columns={"team": "home_team", "season": "prior_season", "primary_coach": "home_prior_coach"}
    )
    frame = frame.merge(home_prior, on=["home_team", "prior_season"], how="left")
    away_prior = primary.rename(
        columns={"team": "away_team", "season": "prior_season", "primary_coach": "away_prior_coach"}
    )
    frame = frame.merge(away_prior, on=["away_team", "prior_season"], how="left")

    frame["year_one_home"] = (
        frame["home_prior_coach"].notna()
        & frame["home_coach"].notna()
        & frame["home_coach"].ne(frame["home_prior_coach"])
    )
    frame["year_one_away"] = (
        frame["away_prior_coach"].notna()
        & frame["away_coach"].notna()
        & frame["away_coach"].ne(frame["away_prior_coach"])
    )
    return frame[["game_id", "season", "year_one_home", "year_one_away"]]


@dataclass(frozen=True)
class OverlayFlip:
    """One game the overlay flipped, for provenance and ledger recording."""

    game_id: str
    matchup: str
    year_one_team: str
    opponent_team: str


@dataclass(frozen=True)
class OverlayResult:
    """The overlay's effect on one week's card.

    ``overlaid_predictions`` is ``predictions`` unchanged except for
    ``home_cover_probability`` on flipped rows -- every other column,
    including every model feature, is byte-identical to the input, which is
    what lets ``publishing._published_card`` and the challenger recorder
    consume it exactly like an ordinary forecast with no overlay-specific
    branching downstream.
    """

    overlaid_predictions: pd.DataFrame
    flips: tuple[OverlayFlip, ...]
    both_year_one_games: tuple[str, ...]
    week_max: int
    enabled: bool

    @property
    def flip_count(self) -> int:
        return len(self.flips)


def apply_coach_fade_overlay(
    predictions: pd.DataFrame,
    schedules: pd.DataFrame,
    *,
    week_max: int = OVERLAY_WEEK_MAX,
    enabled: bool = OVERLAY_ENABLED,
) -> OverlayResult:
    """Flip the forced pick on every "clean case" year-1-coach game.

    A game flips only when ALL hold:

    * ``week <= week_max`` (weeks 9+ are always left untouched -- the
      registered measurement makes no claim there);
    * ``game_type == "REG"`` when that column is present (the effect is a
      regular-season measurement);
    * the model's own pick (``home_cover_probability >= 0.5`` picks home)
      lands on a year-1 team; and
    * the OPPONENT is not also year-1 (the "clean case" -- a year-1-vs-year-1
      game has no measured direction to fade, per
      ``docs/hc_year_one_fade.md`` Section 2, and is reported in
      ``both_year_one_games`` instead of flipped).

    Flipping sets ``home_cover_probability`` to its complement (``1 - p``),
    which is exactly the model's own probability the OTHER side covers
    (modulo the push probability the pool card already ignores) -- so the
    result stays a coherent probability for whichever side ends up picked,
    and every existing reader of ``home_cover_probability`` needs no
    overlay-aware branch.
    """

    required = {"game_id", "season", "week", "home_team", "away_team", "home_cover_probability"}
    missing = sorted(required.difference(predictions.columns))
    if missing:
        raise DataContractError(f"predictions is missing overlay columns: {', '.join(missing)}")

    base = predictions.reset_index(drop=True).copy()
    if not enabled:
        return OverlayResult(base, (), (), week_max, enabled)

    flags = year_one_by_game(schedules)
    merged = base.merge(
        flags,
        on=["game_id", "season"],
        how="left",
        validate="one_to_one",
    )
    merged["year_one_home"] = merged["year_one_home"].fillna(False).astype(bool)
    merged["year_one_away"] = merged["year_one_away"].fillna(False).astype(bool)

    eligible = merged["week"].astype(int).le(week_max)
    if "game_type" in merged.columns:
        eligible &= merged["game_type"].astype(str).eq("REG")

    home_pick = merged["home_cover_probability"].ge(0.5)
    picked_is_year_one = merged["year_one_home"].where(home_pick, merged["year_one_away"])
    opponent_is_year_one = merged["year_one_away"].where(home_pick, merged["year_one_home"])
    both_year_one = merged["year_one_home"] & merged["year_one_away"]

    flip_mask = eligible & picked_is_year_one & ~opponent_is_year_one

    overlaid = base.copy()
    overlaid.loc[flip_mask, "home_cover_probability"] = (
        1.0 - overlaid.loc[flip_mask, "home_cover_probability"]
    )

    flips: list[OverlayFlip] = []
    for _, row in merged.loc[flip_mask].iterrows():
        row_home_pick = bool(row["home_cover_probability"] >= 0.5)
        year_one_team = str(row["home_team"] if row_home_pick else row["away_team"])
        opponent_team = str(row["away_team"] if row_home_pick else row["home_team"])
        flips.append(
            OverlayFlip(
                game_id=str(row["game_id"]),
                matchup=f"{row['away_team']} at {row['home_team']}",
                year_one_team=year_one_team,
                opponent_team=opponent_team,
            )
        )

    both_ids = tuple(merged.loc[eligible & both_year_one, "game_id"].astype(str))
    return OverlayResult(overlaid, tuple(flips), both_ids, week_max, enabled)


def overlay_disclosure_note(result: OverlayResult) -> str:
    """Plain-language provenance sentence, mirroring ``best_pick_tie_note``.

    Empty when the overlay is off or changed nothing this week, so a week
    with no flip candidate reads exactly like a card with no overlay at all
    -- the same "silent when inapplicable" contract
    ``publishing._best_pick_note`` already uses.
    """

    if not result.enabled or result.flip_count == 0:
        return ""
    plural = "" if result.flip_count == 1 else "s"
    detail = "; ".join(
        f"{flip.matchup}: {flip.year_one_team} -> {flip.opponent_team}" for flip in result.flips
    )
    return (
        f"**Overlay applied: {result.flip_count} pick{plural} flipped** by the year-1 "
        f"head-coach fade (weeks 1-{result.week_max}, clean case only: the model sided "
        "with a first-year coach's team against a coach the opponent KEPT). "
        f"{detail}. See docs/coach_fade_overlay.md."
    )


def _record_instant(now: datetime | None) -> pd.Timestamp:
    instant = pd.Timestamp(now if now is not None else datetime.now(UTC))
    return instant.tz_localize("UTC") if instant.tzinfo is None else instant.tz_convert("UTC")


def record_overlay_challenger_decisions(
    artifacts_root: Path,
    data_root: Path,
    *,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Append the coach-only picks to the prospective challenger ledger.

    Unlike an ordinary registered challenger, this is not a retrained model
    with its own ``margin-predict`` artifact: its "model" IS the active
    model, transformed post-prediction. So this reads the active model's OWN
    synchronized weekly forecast (the same artifact
    ``nfl_ats.clv.record_paper_decisions`` records from) rather than
    searching ``artifacts/margin_predictions/`` by configuration fingerprint
    -- ``nfl_ats.prospective_scoring.find_challenger_artifact`` would find
    that SAME artifact directory and hand back its UN-flipped picks, which is
    exactly the wrong ledger row for this challenger to record.

    ``bet_side`` is always recorded as ``"PASS"`` and ``edge`` as NaN: this
    challenger tracks the overlay's FORCED-PICK accuracy only (the
    ``decision_line`` grade in ``prospective_scoring.prospective_accuracy``),
    not a paper-bet edge against the market for the post-flip side, which was
    never computed and would be an invented number if fabricated here.

    Every write-refusal guarantee of the ordinary challenger path still
    applies: pre-kickoff only, never rewrites an existing row, and the
    ``RECORDING_LOCK_WINDOW`` rehearsal guard
    (``nfl_ats.clv.refuse_if_outside_recording_lock_window``).
    """

    entry = find_challenger(artifacts_root, CHALLENGER_ID)
    status = str(entry.get("status"))
    if status != ACTIVE_CHALLENGER_STATUS:
        raise ValueError(
            f"Challenger {CHALLENGER_ID!r} is registered as {status!r}; only "
            f"{ACTIVE_CHALLENGER_STATUS} challengers have picks recorded"
        )

    active = load_active_ats_model(artifacts_root)
    if active is None:
        raise ValueError(
            "No synchronized active ATS model is available to record overlay decisions from"
        )
    forecast = active_artifact_path(artifacts_root, active, "weekly_forecast")
    if forecast is None:
        raise ValueError("Active ATS model has no linked weekly forecast")
    metadata_path = forecast / "metadata.json"
    card_path = forecast / "recommendations.csv"
    if not metadata_path.is_file() or not card_path.is_file():
        raise ValueError(f"Linked weekly forecast is incomplete: {forecast}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("active_model_id") != active.get("model_id"):
        raise ValueError("Weekly forecast model ID does not match the active model")
    if metadata.get("synchronization_status") != "SYNCHRONIZED":
        raise ValueError("Weekly forecast is not synchronized with an evaluation")

    observed_config = artifact_model_config(metadata)
    declared_fingerprint = config_fingerprint(entry.get("model", {}))
    observed_fingerprint = config_fingerprint(observed_config)
    if declared_fingerprint != observed_fingerprint:
        raise DataContractError(
            f"Challenger {CHALLENGER_ID!r} is registered pinned to configuration "
            f"fingerprint {declared_fingerprint}, but the current active forecast "
            f"{forecast} was produced with {observed_fingerprint}; the active model "
            "changed underneath this overlay -- re-register before recording"
        )

    card = pd.read_csv(card_path)
    required = {
        "game_id",
        "season",
        "week",
        "kickoff",
        "away_team",
        "home_team",
        "spread_line",
        "home_cover_probability",
    }
    missing = sorted(required.difference(card.columns))
    if missing:
        raise DataContractError(f"Active forecast card is missing columns: {', '.join(missing)}")
    if card["game_id"].duplicated().any():
        raise DataContractError("Active forecast card contains duplicate games")
    spreads = pd.to_numeric(card["spread_line"], errors="coerce")
    if not np.isfinite(spreads.to_numpy(dtype=float)).all():
        raise DataContractError("Active forecast card has games without a decision spread")
    kickoffs = pd.to_datetime(card["kickoff"], errors="coerce", utc=True)
    if kickoffs.isna().any():
        raise DataContractError("Active forecast card has games without a kickoff timestamp")

    schedules, _team_stats = load_snapshot(latest_snapshot(data_root / "raw"))
    overlay = apply_coach_fade_overlay(card, schedules)
    overlaid_card = overlay.overlaid_predictions

    recorded_at = _record_instant(now)
    refuse_if_outside_recording_lock_window(kickoffs, recorded_at, ledger="challenger")
    pre_kickoff = kickoffs.gt(recorded_at)
    existing = load_challenger_decisions(artifacts_root)
    mine = existing.loc[existing["challenger_id"].astype(str).eq(CHALLENGER_ID)]
    already = card["game_id"].astype(str).isin(set(mine["game_id"].astype(str)))
    keep = pre_kickoff & ~already
    fresh = overlaid_card.loc[keep]

    decisions = pd.DataFrame(
        {
            "recorded_at_utc": recorded_at,
            "challenger_id": CHALLENGER_ID,
            "config_fingerprint": observed_fingerprint,
            "source_artifact": forecast.name,
            "source_sha256": sha256_file(card_path),
            "forecast_created_at_utc": pd.to_datetime(
                metadata.get("created_at_utc"), utc=True, errors="coerce"
            ),
            "feature_profile": str(metadata.get("feature_profile")),
            "feature_table_sha256": str(observed_config.get("feature_table_sha256")),
            "game_id": fresh["game_id"].astype(str),
            "season": fresh["season"].astype(int),
            "week": fresh["week"].astype(int),
            "kickoff": kickoffs.loc[fresh.index],
            "away_team": fresh["away_team"].astype(str),
            "home_team": fresh["home_team"].astype(str),
            "pick_side": np.where(
                pd.to_numeric(fresh["home_cover_probability"], errors="coerce").ge(0.5),
                "HOME",
                "AWAY",
            ).astype(str),
            "bet_side": "PASS",
            "decision_home_spread": spreads.loc[fresh.index].astype(float),
            "edge": np.nan,
        }
    )
    if not decisions.empty:
        combined = (
            decisions if existing.empty else pd.concat([existing, decisions], ignore_index=True)
        )
        atomic_parquet(
            combined[list(CHALLENGER_DECISION_COLUMNS)], challenger_ledger_path(artifacts_root)
        )
        ledger_rows = len(combined)
    else:
        ledger_rows = len(existing)

    return {
        "challenger_id": CHALLENGER_ID,
        "season": int(card["season"].iloc[0]),
        "week": int(card["week"].iloc[0]),
        "source_artifact": forecast.name,
        "config_fingerprint": observed_fingerprint,
        "recorded": len(decisions),
        "already_recorded": int(already.sum()),
        "post_kickoff_skipped": int((~pre_kickoff & ~already).sum()),
        "ledger_rows": int(ledger_rows),
        "flip_count": overlay.flip_count,
        "flipped_game_ids": [flip.game_id for flip in overlay.flips],
    }
