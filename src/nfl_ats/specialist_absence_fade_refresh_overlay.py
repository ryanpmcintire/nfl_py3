"""Specialist (long-snapper/punter) absence fade as a refresh-path
prospective challenger (LEAD-17).

**Binding closing-grounds taxonomy (AGENTS.md), restated verbatim per this
project's rule for any module that scores or adjudicates an experiment:** an
interval or CI that contains zero is NEVER grounds to reject, fail, or close
an experiment. At this evaluator's ~2-point resolution, "contains zero" is
the EXPECTED outcome for a real small signal. Only two grounds ever close a
line of work: (1) refuted mechanism -- a RESOLVED wrong sign (whole interval
on the wrong side of zero) or zero split-half reliability; (2) bounded by a
positive control proven able to detect an effect that size. Everything else
is ``unresolved_below_power``: record it with ``nfl-ats weak-signals
record``, report ``probability_positive``, never the binary "contains zero."

What this is
-------------

``docs/schedule_flag_battery.md`` "Wave 7" measured
``specialist_absence_fade_on_production`` stacked on PRODUCTION at the
opener: **+0.877 accuracy points, week-blocked 95% [-0.436, +2.242], P+
0.879** (ROADMAP LEAD-17). Neither admissible closing ground applies (the
interval is not entirely below zero, and no positive control was run for
this specific screen), so it stays ``unresolved_below_power`` -- and per
AGENTS.md's "a promotion bar is not a decision bar," a ``probability_positive``
this far above 0.5 favours dual-tracking it, not shelving it.

That measurement's own source
(``src/nfl_ats/roster_availability_flag_features.py``) is built from a
PINNED historical injury snapshot and caps at
``SPECIALIST_INJURY_SEASON_END = 2024`` -- a population choice matching the
Wave-7 measurement window, disclosed there as excluding "the disclosed 2025
``date_modified`` schema break" (``docs/injury_timestamp_fallback.md``). That
cap makes ``derive_specialist_absence_features``/``attach_specialist_absence_features``
permanently return an all-zero flag for any 2025+ season regardless of which
``injuries`` frame is supplied -- reusing those two functions unmodified
would make this challenger a structural no-op for the entire 2026 season,
which is not what a live prospective challenger is for. This module therefore
reuses what IS reusable unmodified
(:data:`nfl_ats.roster_availability_flag_features.SPECIALIST_POSITIONS`,
:func:`nfl_ats.roster_availability_flag_features.specialist_player_slugs`) and
duplicates, rather than imports, the one piece that needs a different
population: :func:`live_specialist_out_qualifying` is
``weekly_specialist_out_qualifying`` minus the season cap -- a disclosed,
deliberate deviation, not a silent reuse, matching this repository's own
established convention for exactly this situation
(``nfl_ats.weather_venue_flag_features`` duplicates its schedule/opener-line
loaders from ``nfl_ats.schedule_flag_features`` rather than importing them,
for the identical reason: several lanes edit the shared module concurrently
and the borrowing module needs only a narrow slice of it).

The wire-based IR-placement coverage extension
(``roster_availability_flag_features._specialist_wire_window_qualifying``) is
NOT reused here -- a disclosed simplification, given this task's scope. The
live signal below is the weekly injury report's ``report_status == "Out"``
flag alone (position LS/P), which is the dominant, most auditable half of the
construct and matches the plain-English mechanism ("a team whose long
snapper or punter is Out").

Why this is a REFRESH-path challenger, not a Tuesday one
-----------------------------------------------------------

The NFL's official weekly injury report is filed Wednesday-Friday
(``docs/injury_news_sourcing.md``), which is strictly AFTER the pool's
Tuesday lock. Recording this construct at the Tuesday publish would always
see an empty report and record nothing useful; the late-week refresh pass is
the first point at which the report can genuinely exist. Mirrors the
established refresh-path precedent
(``nfl_ats.crew_tilt_refresh_overlay``, ``docs/officials_crew_leads.md``):
**graded at the FROZEN Tuesday line** (``decision_home_spread`` read from the
week's original published card, never re-formed at a later line), never
altering the played pick, never touching ``pick_revisions.parquet`` or the
published card -- the would-be pick lives only in this module's own
append-only ledger.

Data source and its expected live state
-----------------------------------------

The newest local ``data/raw/nflverse_injuries/*/injuries.parquet`` snapshot
(:func:`latest_nflverse_injuries_snapshot`), NOT the pinned Wave-7 path. As of
this module's creation (2026-09-05) that newest snapshot is still the pinned
``20260826T122850Z`` one, which carries **zero 2026 rows**
(``docs/injury_timestamp_fallback.md`` M1, measured that session) -- there is
no scheduled capture job that refreshes this specific nflverse archive
(distinct from the separately-scheduled ``ingest_nflcom_injuries.py``
Wed/Thu/Fri/Sat jobs that feed ``nflcom_refresh_overlay``); a fresh snapshot
requires an owner-triggered, network ``player-ingest`` run. Until then, this
challenger's own build correctly finds no injury row for the current season
and reports a documented, FAIL-OPEN skip (never an exception, never a flip)
-- exactly the "skips cleanly with a placeholder result when its source is
absent" contract this challenger family requires. Should a future snapshot
directory appear under that same path (an owner-triggered ingest, or this
source becoming scheduled), this module picks it up automatically via the
newest-snapshot convention, with no code change needed here.

Both arms, one row per game: this module never touches the published card or
``pick_revisions.parquet``; it records the PLAYED pick side
(``game.new_pick_side``, the production arm) alongside the specialist-fade
would-be pick side in the SAME ledger row, so both arms are always paired.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from nfl_ats.clv import refuse_if_outside_recording_lock_window
from nfl_ats.data import DataContractError
from nfl_ats.io import atomic_parquet
from nfl_ats.pick_refresh import RefreshResult, original_card
from nfl_ats.roster_availability_flag_features import (
    SPECIALIST_POSITIONS,
    specialist_player_slugs,
)
from nfl_ats.transaction_wire_features import canonical_team

#: Registered in artifacts/prospective/challengers.json.
CHALLENGER_ID = "specialist_absence_fade_refresh_v1"

OVERLAY_STATUS_APPLIED = "specialist_resolved"
OVERLAY_STATUS_NO_SNAPSHOT = "no_injury_snapshot_available"
OVERLAY_STATUS_NO_REPORT_FOR_WEEK = "no_lsp_out_report_for_week"
OVERLAY_STATUS_BOTH_OUT = "both_teams_missing_a_specialist"
OVERLAY_STATUS_NEITHER_OUT = "neither_team_missing_a_specialist"

SPECIALIST_ABSENCE_REFRESH_COLUMNS: tuple[str, ...] = (
    "revision_recorded_at_utc",
    "refresh_run_id",
    "season",
    "week",
    "game_id",
    "home_team",
    "away_team",
    "kickoff",
    "deadline",
    "decision_home_spread",
    "played_pick_side",
    "production_home_cover_probability",
    "injury_snapshot_id",
    "home_specialist_out",
    "away_specialist_out",
    "specialist_would_be_pick_side",
    "specialist_fade_flip",
    "overlay_status",
    "model_id",
    "feature_table_sha256",
)


def specialist_absence_fade_refresh_ledger_path(artifacts_root: Path) -> Path:
    return artifacts_root / "prospective" / "specialist_absence_fade_refresh_decisions.parquet"


def load_specialist_absence_fade_refresh_decisions(artifacts_root: Path) -> pd.DataFrame:
    """The append-only refresh-time overlay ledger (empty frame when none)."""

    path = specialist_absence_fade_refresh_ledger_path(artifacts_root)
    if not path.is_file():
        return pd.DataFrame(columns=list(SPECIALIST_ABSENCE_REFRESH_COLUMNS))
    ledger = pd.read_parquet(path)
    missing = sorted(set(SPECIALIST_ABSENCE_REFRESH_COLUMNS).difference(ledger.columns))
    if missing:
        raise DataContractError(
            f"Specialist-absence-fade refresh ledger is missing columns: {', '.join(missing)}"
        )
    return ledger[list(SPECIALIST_ABSENCE_REFRESH_COLUMNS)]


def latest_nflverse_injuries_snapshot(
    data_root: Path, *, as_of: pd.Timestamp | None = None
) -> Path | None:
    """Newest ``data_root/raw/nflverse_injuries/*/injuries.parquet`` snapshot,
    or ``None`` if the directory or every snapshot is absent.

    Deliberately NOT the pinned Wave-7 path
    (``roster_availability_flag_features.DEFAULT_INJURIES_PATH``) -- a live
    refresh pass must pick up whatever the newest capture holds, so a future
    owner-triggered ``player-ingest`` run is consumed automatically with no
    code change here.
    """

    root = data_root / "raw" / "nflverse_injuries"
    if not root.is_dir():
        return None
    candidates = sorted(root.glob("*/injuries.parquet"))
    if as_of is not None:
        candidates = [
            path
            for path in candidates
            if pd.to_datetime(path.parent.name, format="%Y%m%dT%H%M%SZ", utc=True, errors="coerce")
            <= as_of
        ]
    return candidates[-1] if candidates else None


_REQUIRED_INJURY_COLUMNS = {"season", "week", "team", "position", "report_status", "game_type"}


def live_specialist_out_qualifying(injuries: pd.DataFrame) -> pd.DataFrame:
    """``(season, week, team)`` rows where a LS/P is on the weekly injury
    report as ``report_status == "Out"``, REG season only.

    Deliberate, disclosed DEVIATION from
    ``roster_availability_flag_features.weekly_specialist_out_qualifying``:
    no ``SPECIALIST_INJURY_SEASON_END`` cap. That cap exists ONLY to match
    the historical Wave-7 on-production measurement's declared "full
    2009-2024 depth" population; a live refresh pass instead reads whatever
    the newest capture holds for the CURRENT season, whichever season that
    is (see module docstring).
    """

    missing = sorted(_REQUIRED_INJURY_COLUMNS.difference(injuries.columns))
    if missing:
        raise DataContractError(f"injuries is missing columns: {', '.join(missing)}")
    frame = injuries.copy()
    frame["team"] = frame["team"].astype(str).map(canonical_team)
    frame["season"] = pd.to_numeric(frame["season"], errors="coerce")
    frame["week"] = pd.to_numeric(frame["week"], errors="coerce")
    mask = (
        frame["position"].isin(SPECIALIST_POSITIONS)
        & frame["report_status"].astype(str).eq("Out")
        & frame["game_type"].astype(str).eq("REG")
        & frame["season"].notna()
        & frame["week"].notna()
    )
    rows = frame.loc[mask, ["season", "week", "team"]].copy()
    rows["season"] = rows["season"].astype(int)
    rows["week"] = rows["week"].astype(int)
    return rows.drop_duplicates().reset_index(drop=True)


def _opposite(side: str) -> str:
    return "AWAY" if side == "HOME" else "HOME"


@dataclass(frozen=True)
class _InjurySnapshot:
    snapshot_id: str
    out_teams_by_week: pd.DataFrame  # columns: season, week, team


def build_specialist_absence_fade_refresh_rows(
    plan: RefreshResult, *, data_root: Path
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Pure computation: one row per ELIGIBLE game in ``plan``.

    FAIL-OPEN everywhere: no injury snapshot at all, or a snapshot with no
    report for this exact (season, week), returns an
    EMPTY frame plus ``{"skipped": True, "reason": ...}`` -- a documented
    NO-OP, never an exception and never a flip. Never writes anything.
    """

    empty = pd.DataFrame(columns=list(SPECIALIST_ABSENCE_REFRESH_COLUMNS))
    eligible_games = [game for game in plan.games if game.eligible]
    if not eligible_games:
        return empty, {"skipped": True, "reason": "no eligible games in this refresh pass"}

    snapshot_path = latest_nflverse_injuries_snapshot(
        data_root, as_of=pd.Timestamp(plan.computed_at_utc)
    )
    if snapshot_path is None:
        return empty, {
            "skipped": True,
            "reason": OVERLAY_STATUS_NO_SNAPSHOT,
            "detail": (
                f"no {data_root / 'raw' / 'nflverse_injuries'}/*/injuries.parquet snapshot found"
            ),
        }

    try:
        injuries = pd.read_parquet(snapshot_path)
        if "date_modified" in injuries.columns:
            modified = pd.to_datetime(injuries["date_modified"], utc=True, errors="coerce")
            injuries = injuries.loc[modified.le(pd.Timestamp(plan.computed_at_utc))].copy()
        # specialist_player_slugs is imported, not re-derived, so this module
        # and roster_availability_flag_features can never disagree on which
        # positions count as "specialist" -- called here purely to prove the
        # snapshot has a resolvable player-name universe before it is trusted
        # (a malformed snapshot raises inside this call, caught below).
        specialist_player_slugs(injuries)
        qualifying = live_specialist_out_qualifying(injuries)
    except (DataContractError, KeyError, ValueError) as error:
        return empty, {
            "skipped": True,
            "reason": OVERLAY_STATUS_NO_SNAPSHOT,
            "detail": f"{type(error).__name__}: {error}",
            "injury_snapshot_id": snapshot_path.parent.name,
        }

    week_qualifying = qualifying.loc[
        qualifying["season"].eq(int(plan.season)) & qualifying["week"].eq(int(plan.week))
    ]
    report_for_week = (
        pd.to_numeric(injuries["season"], errors="coerce").eq(int(plan.season))
        & pd.to_numeric(injuries["week"], errors="coerce").eq(int(plan.week))
        & injuries["game_type"].astype(str).eq("REG")
    )
    if not report_for_week.any():
        return empty, {
            "skipped": True,
            "reason": OVERLAY_STATUS_NO_REPORT_FOR_WEEK,
            "detail": (
                f"no injury report resolves for season {plan.season} week {plan.week} in "
                f"{snapshot_path}"
            ),
            "injury_snapshot_id": snapshot_path.parent.name,
        }

    out_teams = set(week_qualifying["team"].astype(str))
    rows: list[dict[str, Any]] = []
    for game in eligible_games:
        home_out = canonical_team(str(game.home_team)) in out_teams
        away_out = canonical_team(str(game.away_team)) in out_teams
        if away_out and not home_out:
            would_be_side = "HOME"
            status = OVERLAY_STATUS_APPLIED
        elif home_out and not away_out:
            would_be_side = "AWAY"
            status = OVERLAY_STATUS_APPLIED
        elif home_out and away_out:
            would_be_side = game.new_pick_side
            status = OVERLAY_STATUS_BOTH_OUT
        else:
            would_be_side = game.new_pick_side
            status = OVERLAY_STATUS_NEITHER_OUT
        rows.append(
            {
                "revision_recorded_at_utc": plan.computed_at_utc,
                "refresh_run_id": plan.refresh_run_id,
                "season": plan.season,
                "week": plan.week,
                "game_id": str(game.game_id),
                "home_team": game.home_team,
                "away_team": game.away_team,
                "kickoff": game.kickoff,
                "deadline": game.deadline,
                "decision_home_spread": game.decision_home_spread,
                "played_pick_side": game.new_pick_side,
                "production_home_cover_probability": float(game.new_home_cover_probability),
                "injury_snapshot_id": snapshot_path.parent.name,
                "home_specialist_out": bool(home_out),
                "away_specialist_out": bool(away_out),
                "specialist_would_be_pick_side": would_be_side,
                "specialist_fade_flip": bool(would_be_side != game.new_pick_side),
                "overlay_status": status,
                "model_id": plan.model_id,
                "feature_table_sha256": plan.feature_table_sha256,
            }
        )

    frame = pd.DataFrame(rows, columns=list(SPECIALIST_ABSENCE_REFRESH_COLUMNS))
    diagnostics = {
        "skipped": False,
        "injury_snapshot_id": snapshot_path.parent.name,
        "games_considered": len(frame),
        "home_specialist_out_game_ids": frame.loc[frame["home_specialist_out"], "game_id"].tolist(),
        "away_specialist_out_game_ids": frame.loc[frame["away_specialist_out"], "game_id"].tolist(),
        "would_flip_game_ids": frame.loc[frame["specialist_fade_flip"], "game_id"].tolist(),
        "status_counts": frame["overlay_status"].value_counts().to_dict(),
    }
    return frame, diagnostics


def record_specialist_absence_fade_refresh_overlay(
    artifacts_root: Path,
    data_root: Path,
    plan: RefreshResult,
    *,
    record_decisions: bool = False,
) -> dict[str, Any]:
    """Append this pass's would-be picks to the specialist-fade overlay ledger.

    Mirrors ``nfl_ats.crew_tilt_refresh_overlay.record_crew_tilt_refresh_overlay``'s
    opt-in ``record_decisions`` contract and reuses
    ``refuse_if_outside_recording_lock_window`` against the week's ORIGINAL
    card kickoffs unchanged. The PLAYED pipeline cannot see this function's
    output: it writes only its own separate ledger, and the ``RefreshResult``
    handed in is consumed strictly read-only.

    Repeated passes across a week legitimately append MULTIPLE rows per game
    (not deduped), mirroring the sibling refresh ledgers: how the flag
    evolves across passes (e.g. an "Out" designation appearing Friday that
    was absent Wednesday) is part of what prospective scoring reads. Scoring
    consumes the LATEST pre-kickoff row per game.
    """

    if not record_decisions:
        return {
            "challenger_id": CHALLENGER_ID,
            "recorded": 0,
            "skipped": True,
            "reason": (
                "pass --record-decisions to append this pass's would-be picks to the "
                "specialist-absence-fade refresh ledger"
            ),
        }

    original = original_card(artifacts_root, season=plan.season, week=plan.week)
    refuse_if_outside_recording_lock_window(
        original["kickoff"], plan.computed_at_utc, ledger="specialist-absence-fade-refresh"
    )

    rows, diagnostics = build_specialist_absence_fade_refresh_rows(plan, data_root=data_root)
    existing = load_specialist_absence_fade_refresh_decisions(artifacts_root)
    if rows.empty:
        return {
            "challenger_id": CHALLENGER_ID,
            "recorded": 0,
            "ledger_rows": len(existing),
            **diagnostics,
        }

    combined = pd.concat([existing, rows], ignore_index=True) if not existing.empty else rows
    atomic_parquet(
        combined[list(SPECIALIST_ABSENCE_REFRESH_COLUMNS)],
        specialist_absence_fade_refresh_ledger_path(artifacts_root),
    )
    return {
        "challenger_id": CHALLENGER_ID,
        "recorded": len(rows),
        "ledger_rows": len(combined),
        **diagnostics,
    }


__all__ = [
    "CHALLENGER_ID",
    "OVERLAY_STATUS_APPLIED",
    "OVERLAY_STATUS_BOTH_OUT",
    "OVERLAY_STATUS_NEITHER_OUT",
    "OVERLAY_STATUS_NO_REPORT_FOR_WEEK",
    "OVERLAY_STATUS_NO_SNAPSHOT",
    "SPECIALIST_ABSENCE_REFRESH_COLUMNS",
    "build_specialist_absence_fade_refresh_rows",
    "latest_nflverse_injuries_snapshot",
    "live_specialist_out_qualifying",
    "load_specialist_absence_fade_refresh_decisions",
    "record_specialist_absence_fade_refresh_overlay",
    "specialist_absence_fade_refresh_ledger_path",
]
