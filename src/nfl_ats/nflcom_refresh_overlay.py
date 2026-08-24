"""NFL.com Friday out>=2-starters fade as an OPTIONAL refresh-path overlay.

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

What this module is, and what it deliberately is not
----------------------------------------------------

The challenger ``nflcom_friday_refresh_out2_starters_v1`` is ALREADY tracked
prospectively at PUBLISH time: ``nfl_ats.prospective.record_nflcom_refresh_
out2_starters_challenger_decisions`` records its picks into the shared
challenger ledger every ``publish-predictions --record-decisions`` run,
against the Tuesday chain pick. What was missing is the OTHER half of the
frozen spec in ``docs/nflcom_friday_refresh.md``'s "Refresh-path integration
contract": computing the flag AT REFRESH TIME -- after the observed-movement
policy has had its say -- so the 2026 season can be scored on what the
refresh machinery WOULD have picked under this rule, alongside the played
market-follow card.

This module adds exactly that hook to ``nfl-ats refresh-picks`` and NOTHING
else:

* **It never alters the played pick.** The market-follow decision in
  ``nfl_ats.pick_refresh.plan_refresh`` stands; every ``RefreshedGame`` is
  consumed read-only here. The would-be pick exists only as a column of a
  SEPARATE append-only ledger
  (``artifacts/prospective/nflcom_friday_refresh_decisions.parquet``), never
  in ``pick_revisions.parquet``, never in the published card.
* **Challenger-tracked only, by measurement.** The max-EV composition study
  (`scripts/nflcom_friday_refresh_feature.py`, docs/nflcom_friday_refresh.md)
  showed adding the overlay to the PLAYED chain LOWERS composed accuracy;
  arm b's +2.1795 pts standalone composition is attribution on
  already-looked-at data. So this stays a recorded counterfactual until a
  promotion decision says otherwise -- wiring it for SCORING is not wiring
  it for PLAYING.

Signal construction -- reused VERBATIM, not reimplemented
---------------------------------------------------------

Every input is the frozen machinery itself:
``nfl_ats.prospective.nflcom_team_starter_out_counts`` (extracted verbatim
from the publish-time recorder's body, which itself ports
``scripts/nflcom_friday_designation_screen.py``'s normalization/starter-proxy
machinery -- the identical machinery registry/weak_signals.json:
nflcom_refresh_out2_starters_on_chain measured with) and
``nfl_ats.prospective.nflcom_out2_starters_flip`` (the frozen flip/tie rule).
There is no third copy of the normalization or starter-proxy logic anywhere.

Leakage discipline (pinned in tests)
------------------------------------

The flag may consume ONLY the week's FINAL Friday league injury page --
manifest-gated to fetched >= Friday 16:00 ET of the game week AND < the
week's earliest kickoff (the identical gate the publish-time recorder runs),
plus prior-week snap shares for the starter proxy. A page absent, stale, or
post-kickoff is a DOCUMENTED NO-OP: the pass records nothing and flips
nothing -- never an error, never a fallback to a different information set.
Week 1's starter proxy is unavailable by construction -> counts 0 -> keep.

Base pick: the WOULD-BE pick composes on top of the post-market-follow
PLAYED side (``RefreshedGame.new_pick_side``), mirroring how the composition
study applied the overlay to whatever the chain backs at refresh time.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import pandas as pd

from nfl_ats.clv import refuse_if_outside_recording_lock_window
from nfl_ats.constants import TEAM_ABBREVIATION_ALIASES
from nfl_ats.data import DataContractError
from nfl_ats.io import atomic_parquet
from nfl_ats.pick_refresh import RefreshResult, original_card, sunday_pick_lock
from nfl_ats.prospective import (
    NFLCOM_STARTER_OUT_THRESHOLD,
    latest_nflcom_injuries_snapshot,
    nflcom_out2_starters_flip,
    nflcom_team_starter_out_counts,
)

#: Same registered challenger as publish-time tracking -- this ledger is the
#: refresh-time VIEW of that one challenger's decisions, not a new challenger.
CHALLENGER_ID = "nflcom_friday_refresh_out2_starters_v1"

NFLCOM_REFRESH_OVERLAY_COLUMNS: tuple[str, ...] = (
    "revision_recorded_at_utc",
    "refresh_run_id",
    "season",
    "week",
    "game_id",
    "home_team",
    "away_team",
    "kickoff",
    "decision_home_spread",
    "played_pick_side",
    "nflcom_would_be_pick_side",
    "nflcom_flip",
    "picked_team",
    "opponent_team",
    "picked_starter_out",
    "opponent_starter_out",
    "picked_flag_ge_threshold",
    "opponent_flag_ge_threshold",
    "injury_page_snapshot",
    "injury_page_fetched_at_utc",
    "model_id",
    "feature_table_sha256",
)


def nflcom_refresh_overlay_ledger_path(artifacts_root: Path) -> Path:
    return artifacts_root / "prospective" / "nflcom_friday_refresh_decisions.parquet"


def load_nflcom_refresh_overlay_decisions(artifacts_root: Path) -> pd.DataFrame:
    """The append-only refresh-time overlay ledger (empty frame when none)."""

    path = nflcom_refresh_overlay_ledger_path(artifacts_root)
    if not path.is_file():
        return pd.DataFrame(columns=list(NFLCOM_REFRESH_OVERLAY_COLUMNS))
    ledger = pd.read_parquet(path)
    missing = sorted(set(NFLCOM_REFRESH_OVERLAY_COLUMNS).difference(ledger.columns))
    if missing:
        raise DataContractError(
            f"NFL.com refresh-overlay ledger is missing columns: {', '.join(missing)}"
        )
    return ledger[list(NFLCOM_REFRESH_OVERLAY_COLUMNS)]


def _friday_gate(kickoffs: pd.Series) -> pd.Timestamp:
    """Friday 16:00 ET of the game week == the week-wide Sunday pick lock
    minus two days -- the IDENTICAL freshness gate the publish-time recorder
    derives from ``nfl_ats.pick_refresh.sunday_pick_lock``."""

    return sunday_pick_lock(kickoffs) - pd.Timedelta(days=2)


def build_nflcom_refresh_overlay_rows(
    plan: RefreshResult, *, data_root: Path
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Pure computation: one row per ELIGIBLE game in ``plan``, carrying the
    played side and the rule's WOULD-BE side, plus gate diagnostics.

    FAIL-OPEN everywhere per the frozen rule text: no snapshot, no snap-counts
    table, or the week's page failing the freshness gate returns an EMPTY
    frame plus ``{"skipped": True, "reason": ...}`` -- a documented NO-OP,
    never an exception and never a flip. Never writes anything -- see
    :func:`record_nflcom_refresh_overlay` for the append-only write.
    """

    empty = pd.DataFrame(columns=list(NFLCOM_REFRESH_OVERLAY_COLUMNS))

    eligible_games = [game for game in plan.games if game.eligible]
    kickoffs = pd.Series(pd.to_datetime([game.kickoff for game in plan.games], utc=True))
    snapshot = latest_nflcom_injuries_snapshot(data_root)
    if snapshot is None:
        return empty, {
            "skipped": True,
            "reason": "no_nflcom_injuries_snapshot",
            "detail": "no data/raw/nflcom_injuries/*/manifest.json snapshot exists yet",
        }
    snaps_candidates = sorted((data_root / "players" / "raw").glob("*/snap_counts.parquet"))
    if not snaps_candidates:
        return empty, {
            "skipped": True,
            "reason": "no_snap_counts_snapshot",
            "detail": "no data/players/raw/*/snap_counts.parquet snapshot exists yet",
        }
    if not eligible_games:
        return empty, {"skipped": True, "reason": "no eligible games in this refresh pass"}
    snapshot_dir, fetched_by_week = snapshot

    fetched_raw = fetched_by_week.get((plan.season, plan.week))
    friday_gate = _friday_gate(kickoffs)
    earliest_kickoff = kickoffs.min()
    gate_reason = ""
    if fetched_raw is None:
        gate_reason = f"page ({plan.season}, week {plan.week}) absent from snapshot manifest"
    else:
        fetched = pd.Timestamp(fetched_raw)
        fetched = (
            fetched.tz_localize("UTC") if fetched.tzinfo is None else fetched.tz_convert("UTC")
        )
        if fetched < friday_gate:
            gate_reason = (
                f"page fetched {fetched.isoformat()} is before Friday 16:00 ET of the "
                f"game week ({friday_gate.isoformat()})"
            )
        elif fetched >= earliest_kickoff:
            # Leakage regression guard, pinned in tests: a page dated at or
            # after kickoff carries post-kickoff information and must NEVER
            # be consumed -- documented no-op, not a fallback read.
            gate_reason = (
                f"page fetched {fetched.isoformat()} is at or after the week's earliest "
                f"kickoff ({earliest_kickoff.isoformat()})"
            )
    if gate_reason:
        return empty, {
            "skipped": True,
            "reason": f"freshness gate failed: {gate_reason}",
        }

    starter_out = nflcom_team_starter_out_counts(snapshot_dir, snaps_candidates[-1])
    fetched_at = pd.Timestamp(fetched_by_week[(plan.season, plan.week)])
    fetched_at = fetched_at.tz_localize("UTC") if fetched_at.tzinfo is None else fetched_at

    rows: list[dict[str, Any]] = []
    for game in eligible_games:
        played_side = game.new_pick_side
        picked_is_home = played_side == "HOME"
        picked_raw = game.home_team if picked_is_home else game.away_team
        opponent_raw = game.away_team if picked_is_home else game.home_team
        picked_team = TEAM_ABBREVIATION_ALIASES.get(str(picked_raw), str(picked_raw))
        opponent_team = TEAM_ABBREVIATION_ALIASES.get(str(opponent_raw), str(opponent_raw))
        picked_count = starter_out.get((plan.season, plan.week, picked_team), 0)
        opponent_count = starter_out.get((plan.season, plan.week, opponent_team), 0)
        would_be_home = nflcom_out2_starters_flip(
            picked_is_home,
            float(picked_count),
            float(opponent_count),
        )
        would_be_side = "HOME" if would_be_home else "AWAY"
        rows.append(
            {
                "revision_recorded_at_utc": plan.computed_at_utc,
                "refresh_run_id": plan.refresh_run_id,
                "season": plan.season,
                "week": plan.week,
                "game_id": game.game_id,
                "home_team": game.home_team,
                "away_team": game.away_team,
                "kickoff": game.kickoff,
                "decision_home_spread": game.decision_home_spread,
                "played_pick_side": played_side,
                "nflcom_would_be_pick_side": would_be_side,
                "nflcom_flip": would_be_side != played_side,
                "picked_team": picked_team,
                "opponent_team": opponent_team,
                "picked_starter_out": int(picked_count),
                "opponent_starter_out": int(opponent_count),
                "picked_flag_ge_threshold": bool(picked_count >= NFLCOM_STARTER_OUT_THRESHOLD),
                "opponent_flag_ge_threshold": bool(opponent_count >= NFLCOM_STARTER_OUT_THRESHOLD),
                "injury_page_snapshot": snapshot_dir.name,
                "injury_page_fetched_at_utc": fetched_at,
                "model_id": plan.model_id,
                "feature_table_sha256": plan.feature_table_sha256,
            }
        )

    frame = pd.DataFrame(rows, columns=list(NFLCOM_REFRESH_OVERLAY_COLUMNS))
    diagnostics = {
        "skipped": False,
        "snapshot_dir": snapshot_dir.name,
        "page_fetched_at_utc": cast(pd.Timestamp, frame["injury_page_fetched_at_utc"].iloc[0]),
        "games_considered": len(frame),
        "would_flip_game_ids": frame.loc[frame["nflcom_flip"], "game_id"].astype(str).tolist(),
        "both_flagged_kept_game_ids": frame.loc[
            frame["picked_flag_ge_threshold"] & frame["opponent_flag_ge_threshold"],
            "game_id",
        ]
        .astype(str)
        .tolist(),
    }
    return frame, diagnostics


def record_nflcom_refresh_overlay(
    artifacts_root: Path,
    data_root: Path,
    plan: RefreshResult,
    *,
    record_decisions: bool = False,
) -> dict[str, Any]:
    """Append this pass's would-be picks to the refresh-time overlay ledger.

    Mirrors ``pick_refresh.record_plan``'s opt-in ``record_decisions``
    contract and reuses ``refuse_if_outside_recording_lock_window`` against
    the week's ORIGINAL card kickoffs unchanged, exactly like
    ``record_injury_signal_refresh_tilt``. The PLAYED pipeline cannot see
    this function's output: it writes only its own separate ledger, and the
    ``RefreshResult`` handed in is consumed strictly read-only -- the pinned
    played-pick-invariance guarantee.

    Repeated passes across a week legitimately append MULTIPLE rows per game
    (not deduped), mirroring the injury-signal ledger: how the flag evolves
    across Thursday/Saturday/Sunday passes is part of what prospective
    scoring reads. Scoring consumes the LATEST pre-kickoff row per game.
    """

    if not record_decisions:
        return {
            "challenger_id": CHALLENGER_ID,
            "recorded": 0,
            "skipped": True,
            "reason": (
                "pass --record-decisions to append this pass's would-be picks to the "
                "NFL.com refresh-overlay ledger"
            ),
        }

    original = original_card(artifacts_root, season=plan.season, week=plan.week)
    refuse_if_outside_recording_lock_window(
        original["kickoff"], plan.computed_at_utc, ledger="nflcom-refresh-overlay"
    )

    rows, diagnostics = build_nflcom_refresh_overlay_rows(plan, data_root=data_root)
    existing = load_nflcom_refresh_overlay_decisions(artifacts_root)
    if rows.empty:
        return {
            "challenger_id": CHALLENGER_ID,
            "recorded": 0,
            "ledger_rows": len(existing),
            **diagnostics,
        }

    combined = pd.concat([existing, rows], ignore_index=True) if not existing.empty else rows
    atomic_parquet(
        combined[list(NFLCOM_REFRESH_OVERLAY_COLUMNS)],
        nflcom_refresh_overlay_ledger_path(artifacts_root),
    )
    return {
        "challenger_id": CHALLENGER_ID,
        "recorded": len(rows),
        "ledger_rows": len(combined),
        **diagnostics,
    }
