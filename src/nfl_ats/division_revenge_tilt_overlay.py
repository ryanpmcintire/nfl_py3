"""Division-revenge tilt overlay: a parameter-free pick-level nudge.

Research chain: ``scripts/nfl_bias_battery_screen.py``'s ``division_revenge_game``
cell -- "2nd meeting this season vs. same opponent; team lost the 1st meeting"
-- is one of 17 predeclared situational cells in the NFL bias battery,
close-graded (2009-2025, 8,634 team-game rows) at +0.1907 accuracy points,
``probability_positive`` 0.8825 (``registry/weak_signals.json:
bias_battery_division_revenge_game``, ``unresolved_below_power``). The same
construct, ported into ``nfl_ats.experiment_runner`` and re-screened at the
opener grade (2020-2025, 3,006 team-games), reads +0.2911 accuracy points,
``probability_positive`` 0.8642 (``registry/weak_signals.json:
bias_battery_division_revenge_game_opener``, also ``unresolved_below_power``).
Both grades sit on the SAME side (positive -- the revenge side outperforms)
even though the two windows overlap and the leans are correlated, not
independent confirmations; per AGENTS.md an interval crossing zero at this
evaluator's ~2-point resolution is the EXPECTED shape for a real small
signal, never grounds to close the line.

This module is the no-window-cost path, built on the exact pattern of
``injury_value_tilt_overlay.py`` and ``coach_fade_overlay.py`` (the original
precedent): a **pick-level, post-prediction transform** of the active model's
own forced pick, dual-tracked against that same active model in the
prospective challenger ledger (``nfl_ats.prospective_scoring``), at no
rotation-registry window cost and with zero training-time feature changes.
**Nothing in this module is wired into ``publishing.py`` or the production
pick path** -- like the injury tilt, and unlike the coach-fade overlay, no
owner decision to play this on the real card has been made; it is dual-
tracked only.

**The rule is parameter-free** -- no threshold, no tuning, nothing derived
from 2018-2025 outcomes. It reads straight from the newest local schedule
snapshot (``data/raw/<snapshot>/schedules.parquet``): a game is a "division
revenge game" for one specific side when it is the SECOND (or later) meeting
between the same two teams in the same regular season and that side LOST the
first meeting. Under the current NFL scheduling formula, two regular-season
meetings between the same two teams are effectively always division games (a
non-division rematch practically cannot happen), so this module -- exactly
like the bias-battery construct it ports -- never adds an explicit
``div_game`` filter; the meeting-count logic alone reproduces the "division
opponents" framing.

The loser of the first meeting is unique (score margins are zero-sum, modulo
an exact tie), so it is IMPOSSIBLE for both teams in a game to qualify as the
revenge side simultaneously -- a tied first meeting simply produces no
revenge side for either team, and the game is left untouched.

Two things live here, mirroring ``coach_fade_overlay.py`` exactly:

1. :func:`division_revenge_side_by_game` -- the pregame-safe, DATA-DERIVED
   signal, ported verbatim from ``nfl_ats.experiment_runner._flag_division_revenge_game``
   / ``scripts/nfl_bias_battery_screen.py``'s ``revenge_flag`` construct
   (same masks, same ``meeting_rank >= 1 and first_margin < 0`` logic), read
   straight from the schedule snapshot, never hand-typed.
2. :func:`apply_division_revenge_tilt_overlay` -- the pick-level transform,
   plus :func:`overlay_disclosure_note` for the plain-English provenance
   sentence.

:func:`record_division_revenge_tilt_challenger_decisions` writes the
overlay's own arm to the prospective challenger ledger so 2026 scores it
cleanly, independent of whether it is ever played on the real card.
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

#: Registered in artifacts/prospective/challengers.json.
CHALLENGER_ID = "division_revenge_tilt_overlay"


def _canonical_team(team: pd.Series) -> pd.Series:
    return team.astype(str).map(lambda code: TEAM_ABBREVIATION_ALIASES.get(code, code))


def division_revenge_side_by_game(schedules: pd.DataFrame) -> pd.DataFrame:
    """One row per REG-season ``game_id``: ``revenge_home``/``revenge_away``.

    Ported verbatim from ``nfl_ats.experiment_runner._flag_division_revenge_game``
    (in turn ported from ``scripts/nfl_bias_battery_screen.py``'s
    ``revenge_flag`` construct): for every (team, opponent, season) triple,
    sort that team's meetings against that specific opponent by ``gameday``;
    ``meeting_rank`` is the 0-indexed occurrence count and ``first_margin`` is
    the team's own score margin (``result`` for the home side, ``-result``
    for the away side -- ``features.py``'s ``home_score - away_score``
    convention) in the FIRST such meeting. ``revenge_flag = meeting_rank >= 1
    and first_margin < 0`` -- the team lost its earlier meeting against this
    same opponent this season.

    Pregame-safe by construction: a game's own flag depends only on strictly
    EARLIER meetings between the same two teams in the same season (the
    first meeting, which by definition happened before any later one), never
    on the current game's own result or any later meeting. The first meeting
    itself always has ``meeting_rank == 0`` and is therefore never flagged.
    A push (``first_margin == 0``, an exact tie) flags neither side.

    Rows without a resolvable first-meeting result (a future or otherwise
    incomplete earlier game -- data missing) get ``first_margin`` of NaN,
    which compares False against ``< 0``, so those rows are simply not
    flagged rather than raising -- "missing data means no flip", matching
    the overlay's frozen rule.
    """

    required = {"game_id", "season", "game_type", "gameday", "home_team", "away_team", "result"}
    missing = sorted(required.difference(schedules.columns))
    if missing:
        raise DataContractError(
            f"schedules is missing columns for division-revenge tracking: {', '.join(missing)}"
        )

    reg = schedules.loc[schedules["game_type"].astype(str).eq("REG")].copy()
    reg["home_team"] = _canonical_team(reg["home_team"])
    reg["away_team"] = _canonical_team(reg["away_team"])
    reg["gameday"] = pd.to_datetime(reg["gameday"], errors="raise")
    reg["result"] = pd.to_numeric(reg["result"], errors="coerce")
    reg["season"] = reg["season"].astype(int)

    home_side = pd.DataFrame(
        {
            "game_id": reg["game_id"].astype(str),
            "season": reg["season"],
            "team": reg["home_team"],
            "opponent": reg["away_team"],
            "gameday": reg["gameday"],
            "team_margin": reg["result"],
            "is_home": True,
        }
    )
    away_side = pd.DataFrame(
        {
            "game_id": reg["game_id"].astype(str),
            "season": reg["season"],
            "team": reg["away_team"],
            "opponent": reg["home_team"],
            "gameday": reg["gameday"],
            "team_margin": -reg["result"],
            "is_home": False,
        }
    )
    long_df = pd.concat([home_side, away_side], ignore_index=True)
    long_df = long_df.sort_values(["team", "opponent", "season", "gameday"]).reset_index(drop=True)

    grouped = long_df.groupby(["team", "opponent", "season"], sort=False)
    meeting_rank = grouped.cumcount()
    first_margin = grouped["team_margin"].transform("first")
    long_df["revenge_flag"] = (meeting_rank >= 1) & (first_margin < 0)

    home_rows = long_df.loc[long_df["is_home"], ["game_id", "revenge_flag"]].rename(
        columns={"revenge_flag": "revenge_home"}
    )
    away_rows = long_df.loc[~long_df["is_home"], ["game_id", "revenge_flag"]].rename(
        columns={"revenge_flag": "revenge_away"}
    )
    flags = home_rows.merge(away_rows, on="game_id", how="outer")
    flags["revenge_home"] = flags["revenge_home"].fillna(False).astype(bool)
    flags["revenge_away"] = flags["revenge_away"].fillna(False).astype(bool)

    seasons = reg[["game_id", "season"]].drop_duplicates(subset="game_id")
    return seasons.merge(flags, on="game_id", how="left")


@dataclass(frozen=True)
class TiltFlip:
    """One game the overlay flipped, for provenance and ledger recording."""

    game_id: str
    matchup: str
    revenge_team: str
    opponent_team: str


@dataclass(frozen=True)
class TiltResult:
    """The overlay's effect on one week's card.

    ``overlaid_predictions`` is ``predictions`` unchanged except for
    ``home_cover_probability`` on flipped rows -- every other column stays
    byte-identical, mirroring ``coach_fade_overlay.OverlayResult``.
    """

    overlaid_predictions: pd.DataFrame
    flips: tuple[TiltFlip, ...]
    enabled: bool

    @property
    def flip_count(self) -> int:
        return len(self.flips)


def apply_division_revenge_tilt_overlay(
    predictions: pd.DataFrame,
    schedules: pd.DataFrame,
    *,
    enabled: bool = True,
) -> TiltResult:
    """Flip the forced pick to the revenge side wherever the pick is against it.

    A game flips only when ALL hold:

    * ``game_type == "REG"`` when that column is present (the construct's
      close- and opener-graded measurements were both scored on regular-
      season games only);
    * exactly one side is flagged as the revenge side (never both -- the
      loser of the first meeting is unique by construction); and
    * the model's own pick (``home_cover_probability >= 0.5`` picks home)
      lands on the OTHER side, i.e. against the revenge side.

    Flipping sets ``home_cover_probability`` to its complement, exactly as
    ``coach_fade_overlay.apply_coach_fade_overlay`` and
    ``injury_value_tilt_overlay.apply_injury_value_tilt_overlay`` do, so
    every existing reader of the column needs no overlay-aware branch.
    """

    required = {"game_id", "season", "home_team", "away_team", "home_cover_probability"}
    missing = sorted(required.difference(predictions.columns))
    if missing:
        raise DataContractError(f"predictions is missing overlay columns: {', '.join(missing)}")

    base = predictions.reset_index(drop=True).copy()
    if not enabled:
        return TiltResult(base, (), enabled)

    flags = division_revenge_side_by_game(schedules)
    merged = base.merge(flags, on=["game_id", "season"], how="left", validate="one_to_one")
    merged["revenge_home"] = merged["revenge_home"].fillna(False).astype(bool)
    merged["revenge_away"] = merged["revenge_away"].fillna(False).astype(bool)

    eligible = pd.Series(True, index=merged.index)
    if "game_type" in merged.columns:
        eligible &= merged["game_type"].astype(str).eq("REG")

    home_pick = merged["home_cover_probability"].ge(0.5)
    flip_mask = eligible & (
        (merged["revenge_home"] & ~home_pick) | (merged["revenge_away"] & home_pick)
    )

    overlaid = base.copy()
    overlaid.loc[flip_mask, "home_cover_probability"] = (
        1.0 - overlaid.loc[flip_mask, "home_cover_probability"]
    )

    flips: list[TiltFlip] = []
    for _, row in merged.loc[flip_mask].iterrows():
        revenge_is_home = bool(row["revenge_home"])
        revenge_team = str(row["home_team"] if revenge_is_home else row["away_team"])
        opponent_team = str(row["away_team"] if revenge_is_home else row["home_team"])
        flips.append(
            TiltFlip(
                game_id=str(row["game_id"]),
                matchup=f"{row['away_team']} at {row['home_team']}",
                revenge_team=revenge_team,
                opponent_team=opponent_team,
            )
        )

    return TiltResult(overlaid, tuple(flips), enabled)


def overlay_disclosure_note(result: TiltResult) -> str:
    """Plain-language provenance sentence, mirroring
    ``injury_value_tilt_overlay.overlay_disclosure_note``.

    Empty when the overlay is off or changed nothing this week. Not currently
    surfaced on the published card -- this overlay is dual-tracked only.
    """

    if not result.enabled or result.flip_count == 0:
        return ""
    plural = "" if result.flip_count == 1 else "s"
    detail = "; ".join(
        f"{flip.matchup}: {flip.opponent_team} -> {flip.revenge_team}" for flip in result.flips
    )
    return (
        f"**Tilt applied: {result.flip_count} pick{plural} flipped** by the division "
        "revenge tilt (the model's pick sat against the team that lost the first "
        f"meeting this season). {detail}. See docs/division_revenge_tilt_overlay.md. "
        "Prospective evidence only -- not applied to the published card."
    )


def _record_instant(now: datetime | None) -> pd.Timestamp:
    instant = pd.Timestamp(now if now is not None else datetime.now(UTC))
    return instant.tz_localize("UTC") if instant.tzinfo is None else instant.tz_convert("UTC")


def record_division_revenge_tilt_challenger_decisions(
    artifacts_root: Path,
    data_root: Path,
    *,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Append the tilt overlay's picks to the prospective challenger ledger.

    Mirrors ``injury_value_tilt_overlay.record_injury_value_tilt_challenger_decisions``
    and ``coach_fade_overlay.record_overlay_challenger_decisions`` exactly:
    this is not a retrained model with its own ``margin-predict`` artifact --
    its "model" IS the active model, transformed post-prediction -- so it
    reads the active model's own synchronized weekly forecast rather than
    searching ``artifacts/margin_predictions/`` by fingerprint, and it
    refuses to record if the active model's live fingerprint no longer
    matches the snapshot this challenger was registered against.

    ``bet_side`` is always ``"PASS"`` and ``edge`` is always NaN: this
    challenger tracks the tilt's forced-pick (``decision_line``) accuracy
    only, never a fabricated paper-bet edge for the post-tilt side.
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
            "No synchronized active ATS model is available to record tilt decisions from"
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
            "changed underneath this tilt -- re-register before recording"
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
    tilt = apply_division_revenge_tilt_overlay(card, schedules)
    tilted_card = tilt.overlaid_predictions

    recorded_at = _record_instant(now)
    refuse_if_outside_recording_lock_window(kickoffs, recorded_at, ledger="challenger")
    pre_kickoff = kickoffs.gt(recorded_at)
    existing = load_challenger_decisions(artifacts_root)
    mine = existing.loc[existing["challenger_id"].astype(str).eq(CHALLENGER_ID)]
    already = card["game_id"].astype(str).isin(set(mine["game_id"].astype(str)))
    keep = pre_kickoff & ~already
    fresh = tilted_card.loc[keep]

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
        "flip_count": tilt.flip_count,
        "flipped_game_ids": [flip.game_id for flip in tilt.flips],
    }
