"""MOD-17 side-ledger challenger: track both served-total methods weekly.

Lane AC's research half promoted the joint residual model's total output to
:data:`nfl_ats.served_total.SERVED_TOTAL_METHOD` on a single, already-spent
Tuesday-opener/full-population screen (``docs/mod17_joint_residual_model.md``,
``probability_positive`` 0.791). That measurement does not need a second
opener window to keep accruing evidence -- the pool computes a tiebreaker
guess every week regardless, so this challenger records BOTH served-total
candidates (:func:`nfl_ats.served_total.served_total_blend_k01` and
:func:`nfl_ats.served_total.served_total_joint_residual`) for the week's
tiebreaker game, pre-kickoff, and backfills the realised total once the game
finishes -- one row per week, no rotation-registry cost, mirroring
``nfl_ats.best_pick_nomination``'s v3 side-ledger-only registration pattern
(``docs/best_pick_ranker.md`` "v3 audit"): registered ``ACTIVE_PROSPECTIVE``
in ``artifacts/prospective/challengers.json``, recorded from
``publish-predictions --record-decisions``, never read back into the
published card.

Unlike every other challenger ledger in this project, the two arms compared
here are NUMERIC TOTALS, not side picks, so this uses its OWN small ledger
(:data:`LEDGER_COLUMNS`) rather than
``nfl_ats.prospective_scoring.CHALLENGER_DECISION_COLUMNS`` (which is shaped
around ``pick_side``/``decision_home_spread`` and has no field for a
continuous prediction). The registry-membership check
(:func:`nfl_ats.prospective_scoring.find_challenger`) and the anti-backdating
recording-lock-window guard (:func:`nfl_ats.clv.refuse_if_outside_recording_lock_window`)
are reused unmodified -- only the row shape differs.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from nfl_ats.clv import refuse_if_outside_recording_lock_window
from nfl_ats.data import DataContractError
from nfl_ats.io import atomic_parquet
from nfl_ats.prospective_scoring import ACTIVE_CHALLENGER_STATUS, find_challenger
from nfl_ats.tiebreaker import newest_schedules_path, tiebreaker_report

#: Registered in ``artifacts/prospective/challengers.json``.
CHALLENGER_ID = "totals_served_method"

#: One row per (challenger, week): the tiebreaker game's market total, both
#: served-total candidates, which one actually served, and the realised
#: total once the game is final (``NaN`` while pending).
LEDGER_COLUMNS: tuple[str, ...] = (
    "recorded_at_utc",
    "challenger_id",
    "served_total_method",
    "game_id",
    "season",
    "week",
    "kickoff",
    "home_team",
    "away_team",
    "market_total",
    "served_total_blend_k01",
    "served_total_joint_residual",
    "realised_total",
)


def ledger_path(artifacts_root: Path) -> Path:
    return artifacts_root / "prospective" / "totals_served_method_decisions.parquet"


def load_decisions(artifacts_root: Path) -> pd.DataFrame:
    """The append-only ledger (empty frame when none exists)."""

    path = ledger_path(artifacts_root)
    if not path.is_file():
        return pd.DataFrame(columns=list(LEDGER_COLUMNS))
    ledger = pd.read_parquet(path)
    missing = sorted(set(LEDGER_COLUMNS).difference(ledger.columns))
    if missing:
        raise DataContractError(
            f"totals_served_method ledger is missing columns: {', '.join(missing)}"
        )
    if ledger["game_id"].duplicated().any():
        raise DataContractError(f"totals_served_method ledger contains duplicate rows: {path}")
    return ledger[list(LEDGER_COLUMNS)]


def _record_instant(now: datetime | None) -> pd.Timestamp:
    instant = pd.Timestamp(now if now is not None else datetime.now(UTC))
    return instant.tz_localize("UTC") if instant.tzinfo is None else instant.tz_convert("UTC")


def _schedule_kickoff_utc(schedules: pd.DataFrame) -> pd.Series:
    """Combine nflverse ``gameday`` + Eastern ``gametime`` into UTC.

    Duplicated (not imported) from ``nfl_ats.play_probability._schedule_kickoff_utc``
    / ``nfl_ats.players._schedule_kickoff_utc`` -- the same cross-module
    duplication convention every copy of this helper already follows in this
    repository.
    """

    if "gametime" not in schedules:
        return pd.Series(pd.NaT, index=schedules.index, dtype="datetime64[ns, UTC]")
    date_text = pd.to_datetime(schedules["gameday"], errors="coerce").dt.strftime("%Y-%m-%d")
    time_text = schedules["gametime"].astype("string")
    local = pd.to_datetime(date_text + " " + time_text, errors="coerce")
    return local.dt.tz_localize(
        "America/New_York", ambiguous="NaT", nonexistent="shift_forward"
    ).dt.tz_convert("UTC")


def settle_realised_totals(decisions: pd.DataFrame, schedules: pd.DataFrame) -> pd.DataFrame:
    """Fill in ``realised_total`` for any pending row whose game has a final score.

    Never overwrites an already-settled value (an existing non-NaN
    ``realised_total`` is left exactly as recorded) and never touches any
    other column -- this is purely a backfill, not a re-grade.
    """

    if decisions.empty:
        return decisions
    scores = schedules.loc[:, ["game_id", "home_score", "away_score"]].copy()
    scores["game_id"] = scores["game_id"].astype(str)
    scores["_realised_total"] = pd.to_numeric(
        scores["home_score"], errors="coerce"
    ) + pd.to_numeric(scores["away_score"], errors="coerce")
    scores = scores.drop_duplicates("game_id")[["game_id", "_realised_total"]]

    updated = decisions.copy()
    updated["game_id"] = updated["game_id"].astype(str)
    updated = updated.merge(scores, on="game_id", how="left")
    updated["realised_total"] = updated["realised_total"].where(
        updated["realised_total"].notna(), updated["_realised_total"]
    )
    return updated.drop(columns=["_realised_total"])[list(LEDGER_COLUMNS)]


def record_totals_served_method_decisions(
    artifacts_root: Path,
    data_root: Path,
    *,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Record this week's tiebreaker game under both served-total methods.

    Computes the SAME :class:`~nfl_ats.tiebreaker.TiebreakerReport`
    ``nfl-ats tiebreaker`` would (:func:`nfl_ats.tiebreaker.tiebreaker_report`),
    so both candidates are read straight off the production pipeline rather
    than re-derived. Records exactly one row per week -- the tiebreaker
    game, never every game on the card, since only that one game's total is
    served anywhere. Whole-game-pre-kickoff anti-backdating: refuses via
    :func:`nfl_ats.clv.refuse_if_outside_recording_lock_window` when called
    more than ``RECORDING_LOCK_WINDOW`` before kickoff, and silently skips
    (rather than recording a hindsight row) once kickoff has passed. Every
    call also backfills ``realised_total`` on any prior pending rows whose
    games have since finished, so evidence keeps accruing without a second
    recording pass.
    """

    entry = find_challenger(artifacts_root, CHALLENGER_ID)
    status = str(entry.get("status"))
    if status != ACTIVE_CHALLENGER_STATUS:
        raise ValueError(
            f"Challenger {CHALLENGER_ID!r} is registered as {status!r}; only "
            f"{ACTIVE_CHALLENGER_STATUS} challengers have picks recorded"
        )

    recorded_at = _record_instant(now)
    report = tiebreaker_report(data_root, artifacts_root=artifacts_root, today=recorded_at.date())
    schedules = pd.read_parquet(newest_schedules_path(data_root))
    game_row = schedules.loc[schedules["game_id"].astype(str).eq(report.game_id)]
    if game_row.empty:
        raise DataContractError(
            f"{report.game_id} (this week's tiebreaker game) is not in the newest schedules"
        )
    game_row = game_row.reset_index(drop=True)
    kickoff = _schedule_kickoff_utc(game_row).iloc[0]
    game = game_row.iloc[0]
    season = int(game["season"])
    week = int(game["week"])

    existing = load_decisions(artifacts_root)
    settled = settle_realised_totals(existing, schedules)
    settled_changed = not settled.equals(existing)

    already = report.game_id in set(settled["game_id"].astype(str))
    if already:
        if settled_changed:
            atomic_parquet(settled, ledger_path(artifacts_root))
        return {
            "challenger_id": CHALLENGER_ID,
            "recorded": 0,
            "already_recorded": 1,
            "post_kickoff_skipped": 0,
            "game_id": report.game_id,
            "ledger_rows": len(settled),
            "settled_rows_updated": bool(settled_changed),
        }

    kickoffs = pd.Series([kickoff])
    refuse_if_outside_recording_lock_window(kickoffs, recorded_at, ledger="challenger")
    if pd.isna(kickoff) or kickoff <= recorded_at:
        if settled_changed:
            atomic_parquet(settled, ledger_path(artifacts_root))
        return {
            "challenger_id": CHALLENGER_ID,
            "recorded": 0,
            "already_recorded": 0,
            "post_kickoff_skipped": 1,
            "game_id": report.game_id,
            "ledger_rows": len(settled),
            "settled_rows_updated": bool(settled_changed),
        }

    joint_value = (
        report.served_total if report.served_total_method == "joint_residual" else float("nan")
    )
    row = pd.DataFrame(
        [
            {
                "recorded_at_utc": recorded_at,
                "challenger_id": CHALLENGER_ID,
                "served_total_method": report.served_total_method,
                "game_id": report.game_id,
                "season": season,
                "week": week,
                "kickoff": kickoff,
                "home_team": report.home,
                "away_team": report.away,
                "market_total": float(report.consensus.total_line),
                "served_total_blend_k01": float(report.comparison_total_blend_k01),
                "served_total_joint_residual": (
                    float(joint_value) if pd.notna(joint_value) else float("nan")
                ),
                "realised_total": float("nan"),
            }
        ]
    )
    combined = pd.concat([settled, row], ignore_index=True) if not settled.empty else row
    atomic_parquet(combined[list(LEDGER_COLUMNS)], ledger_path(artifacts_root))
    return {
        "challenger_id": CHALLENGER_ID,
        "recorded": 1,
        "already_recorded": 0,
        "post_kickoff_skipped": 0,
        "game_id": report.game_id,
        "season": season,
        "week": week,
        "served_total_method": report.served_total_method,
        "market_total": float(report.consensus.total_line),
        "served_total_blend_k01": float(report.comparison_total_blend_k01),
        "served_total_joint_residual": (float(joint_value) if pd.notna(joint_value) else None),
        "ledger_rows": len(combined),
    }


__all__ = [
    "CHALLENGER_ID",
    "LEDGER_COLUMNS",
    "ledger_path",
    "load_decisions",
    "record_totals_served_method_decisions",
    "settle_realised_totals",
]
