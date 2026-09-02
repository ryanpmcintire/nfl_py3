"""Late-week officiating-crew tilt as a refresh-path prospective challenger.

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

What this module is
-------------------

The frozen rule text lives in ``docs/referee_assignments_capture.md`` section
"Late-week crew-tilt challenger predeclaration (2026-09-01, WP47)", written
BEFORE this file existed and before any number was computed. This module
implements exactly that rule and nothing else.

``docs/referee_assignments_capture.md`` section 2 measured that Football
Zebras never publishes a week's crew assignments before Tuesday afternoon, so
the capture (``referee_assignments_wed``) can only ever feed a LATE-WEEK
refresh -- never the Tuesday-lock card. Section 5 of that document listed
what was still missing to make the two ``penalty_crew_tendencies`` cells
prospectively playable. This module supplies items 1, 2 and 4 of that list;
item 3 (the family declaration) is the predeclaration section itself.

* **It never alters the played pick.** Every ``RefreshedGame`` is consumed
  read-only. The would-be pick exists only in a SEPARATE append-only ledger
  (``artifacts/prospective/crew_tilt_refresh_decisions.parquet``), never in
  ``pick_revisions.parquet``, never in the published card.
* **A prospective challenger is paper evidence at zero window cost**, not a
  promotion and not a claim that either cell is resolved. Both remain
  ``unresolved_below_power`` in ``registry/weak_signals.json``.

Signal construction -- imported, not reimplemented
--------------------------------------------------

The two cells' flags come from the screen's own builders
(``nfl_ats.experiment_runner._build_referee_type_trait_data``,
``._build_referee_trait_data``, ``._merge_home_pass_rate_quartile``,
``._HEAVY_UNDERDOG_THRESHOLD_DEFAULT``). Those builders key entirely off
completed games' ``officials.parquet`` join, so they are structurally
incapable of scoring a game that has not been played. The ONE thing added
here is that forward hop: :func:`build_crew_trait_lookup` re-derives the
per-(referee, season) mean rate and the lagged population's own qcut
cutpoints so a referee's PRIOR completed season can be bucketed for a future
game. ``tests/test_crew_tilt_refresh_overlay.py`` MEASURES that this adapter
reproduces the builders' own ``lag_type_quartile`` /
``lag_penalty_rate_quartile`` exactly on every historical (referee, season)
pair -- a pinned second path, never a second definition.

Leakage discipline (pinned in tests)
------------------------------------

A crew snapshot may only be consumed for a game whose own pick deadline
``min(kickoff, Sunday 16:00 ET)`` (``nfl_ats.pick_refresh.pick_deadline``) is
strictly AFTER the snapshot's ``captured_at_utc``. A snapshot at or after a
game's kickoff, or at or after that week's Sunday 16:00 ET lock, can never
apply to it. A Wednesday capture IS before the Sunday lock, so SNF and MNF
are playable for this channel -- verified per game against ``pick_deadline``,
never assumed. A missing, stale, or post-deadline snapshot is a DOCUMENTED
NO-OP: zero tilt, the incumbent Tuesday pick stands, the row is tagged.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd

from nfl_ats.clv import refuse_if_outside_recording_lock_window
from nfl_ats.data import DataContractError
from nfl_ats.experiment_runner import (
    _HEAVY_UNDERDOG_THRESHOLD_DEFAULT,
    _HOLDING_PENALTY_TYPE,
    _REFEREE_POSITION,
    _REFEREE_SEASON_TYPE,
    _build_referee_trait_data,
    _build_referee_type_trait_data,
    _latest_officials_snapshot,
    _latest_penalty_type_snapshot,
    _latest_schedules_snapshot,
    _merge_home_pass_rate_quartile,
)
from nfl_ats.io import atomic_parquet
from nfl_ats.pick_refresh import RefreshResult, original_card

#: Registered in artifacts/prospective/challengers.json.
CHALLENGER_ID = "crew_tilt_refresh_v1"

#: Registry family for the stacked back-test verdict (predeclared in
#: docs/referee_assignments_capture.md, WP47 section 5).
STACKED_SIGNAL_NAME = "crew_tilt_stacked_on_production"

# ---------------------------------------------------------------------------
# The two cells' own measured per-game gaps. NOT chosen here.
#
# ``raw_gap_pct`` is stored already multiplied by the construct's sign
# (src/nfl_ats/experiment_runner.py:3629-3630 --
# ``raw_gap_pct = construct.sign * (subset_cover - complement_cover) * 100``),
# so the SIGNED home-cover gap in percentage points is ``sign * raw_gap_pct``
# and the probability tilt is that divided by 100. Both values are read from
# the experiments' own artifacts and pinned by
# ``tests/test_crew_tilt_refresh_overlay.py`` against BOTH that artifact and
# ``registry/weak_signals.json``'s ``classification_evidence`` text.
# ---------------------------------------------------------------------------

#: registry/weak_signals.json:penalty_crew_holding_tilt_run_heavy
HOLDING_RUN_HEAVY_SIGNAL = "penalty_crew_holding_tilt_run_heavy"
#: artifacts/experiment_runner/20260820T113432Z/metadata.json result.raw_gap_pct
HOLDING_RUN_HEAVY_ARTIFACT = "artifacts/experiment_runner/20260820T113432Z/metadata.json"
HOLDING_RUN_HEAVY_RAW_GAP_POINTS = 5.994893289010933
#: classification_evidence: "sign=-1 (positive flag=True favours the hypothesis)"
HOLDING_RUN_HEAVY_SIGN = -1
#: Signed tilt applied to P(home cover) when cell C's flag fires.
HOLDING_RUN_HEAVY_TILT = HOLDING_RUN_HEAVY_SIGN * HOLDING_RUN_HEAVY_RAW_GAP_POINTS / 100.0

#: registry/weak_signals.json:penalty_crew_high_flag_heavy_underdog_opener
HIGH_FLAG_UNDERDOG_SIGNAL = "penalty_crew_high_flag_heavy_underdog_opener"
#: artifacts/experiment_runner/20260820T113443Z/metadata.json result.raw_gap_pct
HIGH_FLAG_UNDERDOG_ARTIFACT = "artifacts/experiment_runner/20260820T113443Z/metadata.json"
HIGH_FLAG_UNDERDOG_RAW_GAP_POINTS = 16.772226131832042
#: classification_evidence: "sign=+1 (positive flag=True favours the hypothesis)"
HIGH_FLAG_UNDERDOG_SIGN = 1
#: Signed tilt applied to P(home cover) when cell A's flag fires.
HIGH_FLAG_UNDERDOG_TILT = HIGH_FLAG_UNDERDOG_SIGN * HIGH_FLAG_UNDERDOG_RAW_GAP_POINTS / 100.0

#: The heavy-underdog cut is the screen's own default, imported rather than
#: retyped (src/nfl_ats/experiment_runner.py:1614). Home is a heavy underdog
#: when the frozen decision line is <= -7.0 in the nflverse convention
#: (positive = home favored), i.e. home is getting >= 7 points.
HEAVY_UNDERDOG_THRESHOLD = _HEAVY_UNDERDOG_THRESHOLD_DEFAULT

#: Quartile buckets, matching the builders' own ``pd.qcut(..., 4,
#: labels=[1, 2, 3, 4])``.
TOP_QUARTILE = 4
BOTTOM_QUARTILE = 1

OVERLAY_STATUS_APPLIED = "crew_resolved"
OVERLAY_STATUS_NO_SNAPSHOT = "no_crew_snapshot_for_week"
OVERLAY_STATUS_SNAPSHOT_AFTER_DEADLINE = "crew_snapshot_at_or_after_pick_deadline"
OVERLAY_STATUS_NO_ROW = "game_absent_from_crew_snapshot"
OVERLAY_STATUS_UNKNOWN_REFEREE = "referee_has_no_prior_season_trait"

CREW_TILT_REFRESH_COLUMNS: tuple[str, ...] = (
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
    "crew_snapshot_id",
    "crew_captured_at_utc",
    "referee",
    "holding_crew_top_quartile",
    "home_run_heavy_bottom_quartile",
    "holding_tilt_flag",
    "flag_rate_crew_top_quartile",
    "home_heavy_underdog",
    "high_flag_underdog_flag",
    "tilt_points",
    "tilted_home_cover_probability",
    "crew_would_be_pick_side",
    "crew_tilt_flip",
    "overlay_status",
    "model_id",
    "feature_table_sha256",
)


def crew_tilt_refresh_ledger_path(artifacts_root: Path) -> Path:
    return artifacts_root / "prospective" / "crew_tilt_refresh_decisions.parquet"


def load_crew_tilt_refresh_decisions(artifacts_root: Path) -> pd.DataFrame:
    """The append-only refresh-time overlay ledger (empty frame when none)."""

    path = crew_tilt_refresh_ledger_path(artifacts_root)
    if not path.is_file():
        return pd.DataFrame(columns=list(CREW_TILT_REFRESH_COLUMNS))
    ledger = pd.read_parquet(path)
    missing = sorted(set(CREW_TILT_REFRESH_COLUMNS).difference(ledger.columns))
    if missing:
        raise DataContractError(
            f"Crew-tilt refresh ledger is missing columns: {', '.join(missing)}"
        )
    return ledger[list(CREW_TILT_REFRESH_COLUMNS)]


# ---------------------------------------------------------------------------
# The forward hop: a referee's PRIOR completed season, bucketed against the
# screen builders' own frozen quartile cutpoints.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _LaggedTrait:
    """One trait's per-(referee, season) mean plus the frozen lagged cutpoints.

    ``name_season`` has columns ``official_name``/``season``/``mean_total``.
    ``interior_cutpoints`` are the three interior boundaries of the builder's
    own ``pd.qcut(lagged["prev_total"], 4)`` over every (official, season)
    pair carrying a valid one-season lag -- the population Section 5 item 2
    of ``docs/referee_assignments_capture.md`` requires be FROZEN rather than
    recomputed for a future game.
    """

    name_season: pd.DataFrame
    interior_cutpoints: tuple[float, float, float]
    historical_quartile: dict[tuple[str, int], int]


def _lagged_quartile_bucket(value: float, cutpoints: tuple[float, float, float]) -> int:
    """``pd.qcut``'s own right-closed bucketing, applied to one value.

    ``pd.qcut`` produces right-closed intervals ``(a, b]``, so a value equal
    to an interior cutpoint belongs to the LOWER bucket. ``searchsorted``
    with ``side="left"`` reproduces exactly that.
    """

    index = int(np.searchsorted(np.asarray(cutpoints, dtype=float), float(value), side="left"))
    return index + 1


def _build_lagged_trait(name_season: pd.DataFrame) -> _LaggedTrait:
    """The builders' own lag/qcut, with the cutpoints kept instead of discarded.

    This repeats the three lines
    ``nfl_ats.experiment_runner._build_referee_trait_data`` /
    ``._build_referee_type_trait_data`` run internally (shift by one season,
    keep pairs exactly one season apart, ``pd.qcut(prev_total, 4)``). It is
    repeated rather than imported ONLY because those functions return the
    per-game join and discard both ``prev_total`` and the bin edges, which is
    precisely what a not-yet-played game needs. Equality with the builders is
    MEASURED, not assumed -- see the module docstring.
    """

    lag = name_season.sort_values(["official_name", "season"]).copy()
    lag["prev_total"] = lag.groupby("official_name")["mean_total"].shift(1)
    lag["prev_season"] = lag.groupby("official_name")["season"].shift(1)
    lagged = lag.loc[lag["season"] - lag["prev_season"] == 1].copy()
    buckets, bins = pd.qcut(lagged["prev_total"], 4, labels=[1, 2, 3, 4], retbins=True)
    lagged["quartile"] = buckets.astype(int)
    interior = (float(bins[1]), float(bins[2]), float(bins[3]))
    historical = {
        (str(row.official_name), int(cast(Any, row.season))): int(cast(Any, row.quartile))
        for row in lagged.itertuples(index=False)
    }
    return _LaggedTrait(
        name_season=name_season.loc[:, ["official_name", "season", "mean_total"]].copy(),
        interior_cutpoints=interior,
        historical_quartile=historical,
    )


@dataclass(frozen=True)
class CrewTraitLookup:
    """Both cells' season-lagged crew traits, usable for a FUTURE game.

    ``holding`` is the Offensive-Holding-rate trait cell C keys on;
    ``flag_rate`` is the overall ``mean_total`` penalty-rate trait cell A
    reuses from ``docs/referee_battery.md``.
    """

    holding: _LaggedTrait
    flag_rate: _LaggedTrait
    officials_snapshot_id: str
    penalty_type_snapshot_id: str

    def _quartile(self, trait: _LaggedTrait, referee: str, season: int) -> int | None:
        historical = trait.historical_quartile.get((referee, season))
        if historical is not None:
            return historical
        prior = trait.name_season.loc[
            (trait.name_season["official_name"] == referee)
            & (trait.name_season["season"] == season - 1),
            "mean_total",
        ]
        if prior.empty:
            return None
        return _lagged_quartile_bucket(float(prior.iloc[0]), trait.interior_cutpoints)

    def holding_quartile(self, referee: str, season: int) -> int | None:
        """Quartile of ``referee``'s PRIOR-season Offensive Holding rate."""

        return self._quartile(self.holding, referee, season)

    def flag_rate_quartile(self, referee: str, season: int) -> int | None:
        """Quartile of ``referee``'s PRIOR-season overall ``mean_total`` rate."""

        return self._quartile(self.flag_rate, referee, season)


def _referee_name_season(repo_root: Path, *, penalty_type: str | None) -> pd.DataFrame:
    """Per-(referee, season) mean penalty rate, the builders' own aggregation.

    ``penalty_type=None`` reproduces ``_build_referee_trait_data``'s
    ``mean_total`` (every penalty, from ``game_penalties.parquet``); a
    ``penalty_type`` string reproduces ``_build_referee_type_trait_data``'s
    single-type rate (from ``game_penalty_types.parquet``, absent games
    filled to 0.0 because a game with none of that type is a genuine zero).
    """

    officials_path, game_penalties_path, _snapshot = _latest_officials_snapshot(repo_root)
    officials = pd.read_parquet(officials_path)
    refs = officials.loc[
        (officials["position"] == _REFEREE_POSITION)
        & (officials["season_type"] == _REFEREE_SEASON_TYPE)
    ].copy()
    schedules = pd.read_parquet(_latest_schedules_snapshot(repo_root)).loc[
        :, ["game_id", "old_game_id"]
    ]
    refs = refs.merge(
        schedules, left_on="game_id", right_on="old_game_id", how="inner", suffixes=("_legacy", "")
    )
    refs = refs.loc[:, ["game_id", "official_name", "season"]]

    if penalty_type is None:
        game_penalties = pd.read_parquet(game_penalties_path)
        merged = refs.merge(game_penalties, on="game_id", how="inner", suffixes=("", "_gp"))
    else:
        penalty_type_path, _ = _latest_penalty_type_snapshot(repo_root)
        game_penalty_types = pd.read_parquet(penalty_type_path)
        type_counts = game_penalty_types.loc[
            game_penalty_types["penalty_type"] == penalty_type, ["game_id", "penalties_total"]
        ]
        merged = refs.merge(type_counts, on="game_id", how="left")
        merged["penalties_total"] = merged["penalties_total"].fillna(0.0)

    return (
        merged.groupby(["official_name", "season"])
        .agg(mean_total=("penalties_total", "mean"))
        .reset_index()
    )


def build_crew_trait_lookup(repo_root: Path) -> CrewTraitLookup:
    """Both season-lagged crew traits, with frozen cutpoints for a future game."""

    _officials_path, _penalties_path, officials_snapshot_id = _latest_officials_snapshot(repo_root)
    _penalty_type_path, penalty_type_snapshot_id = _latest_penalty_type_snapshot(repo_root)
    return CrewTraitLookup(
        holding=_build_lagged_trait(
            _referee_name_season(repo_root, penalty_type=_HOLDING_PENALTY_TYPE)
        ),
        flag_rate=_build_lagged_trait(_referee_name_season(repo_root, penalty_type=None)),
        officials_snapshot_id=officials_snapshot_id,
        penalty_type_snapshot_id=penalty_type_snapshot_id,
    )


# ---------------------------------------------------------------------------
# The two cells' flags and the additive tilt
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CrewTiltFlags:
    """One game's two cell flags and the additive tilt they compose to."""

    referee: str | None
    holding_crew_top_quartile: bool
    home_run_heavy_bottom_quartile: bool
    holding_tilt_flag: bool
    flag_rate_crew_top_quartile: bool
    home_heavy_underdog: bool
    high_flag_underdog_flag: bool
    tilt_points: float
    status: str


def crew_tilt_flags(
    *,
    referee: str | None,
    season: int,
    home_pass_rate_quartile: int | None,
    decision_home_spread: float | None,
    lookup: CrewTraitLookup,
) -> CrewTiltFlags:
    """The frozen per-game rule (docs/referee_assignments_capture.md, WP47 §2.3).

    Cell C fires when the home team's prior-rolling pass-rate quartile is the
    BOTTOM one (run-heavy) AND the referee's PRIOR-season Offensive Holding
    rate is in the TOP quartile. Cell A fires when the referee's PRIOR-season
    ``mean_total`` rate is in the TOP quartile AND the home team is getting
    >= 7 points at the FROZEN Tuesday line. They compose ADDITIVELY -- a
    composition that was never measured, disclosed in the predeclaration.
    """

    if referee is None or not str(referee).strip():
        return CrewTiltFlags(
            referee=None,
            holding_crew_top_quartile=False,
            home_run_heavy_bottom_quartile=False,
            holding_tilt_flag=False,
            flag_rate_crew_top_quartile=False,
            home_heavy_underdog=False,
            high_flag_underdog_flag=False,
            tilt_points=0.0,
            status=OVERLAY_STATUS_NO_ROW,
        )

    name = str(referee)
    holding_quartile = lookup.holding_quartile(name, season)
    flag_rate_quartile = lookup.flag_rate_quartile(name, season)
    if holding_quartile is None and flag_rate_quartile is None:
        return CrewTiltFlags(
            referee=name,
            holding_crew_top_quartile=False,
            home_run_heavy_bottom_quartile=False,
            holding_tilt_flag=False,
            flag_rate_crew_top_quartile=False,
            home_heavy_underdog=False,
            high_flag_underdog_flag=False,
            tilt_points=0.0,
            status=OVERLAY_STATUS_UNKNOWN_REFEREE,
        )

    holding_top = holding_quartile == TOP_QUARTILE
    run_heavy = home_pass_rate_quartile == BOTTOM_QUARTILE
    holding_flag = bool(holding_top and run_heavy)

    flag_rate_top = flag_rate_quartile == TOP_QUARTILE
    heavy_underdog = (
        decision_home_spread is not None
        and not pd.isna(decision_home_spread)
        and float(decision_home_spread) <= -HEAVY_UNDERDOG_THRESHOLD
    )
    underdog_flag = bool(flag_rate_top and heavy_underdog)

    tilt = 0.0
    if holding_flag:
        tilt += HOLDING_RUN_HEAVY_TILT
    if underdog_flag:
        tilt += HIGH_FLAG_UNDERDOG_TILT

    return CrewTiltFlags(
        referee=name,
        holding_crew_top_quartile=bool(holding_top),
        home_run_heavy_bottom_quartile=bool(run_heavy),
        holding_tilt_flag=holding_flag,
        flag_rate_crew_top_quartile=bool(flag_rate_top),
        home_heavy_underdog=bool(heavy_underdog),
        high_flag_underdog_flag=underdog_flag,
        tilt_points=tilt,
        status=OVERLAY_STATUS_APPLIED,
    )


def tilted_probability(production_probability: float, tilt_points: float) -> float:
    """``clip(p + tilt, 0, 1)`` -- the frozen additive composition."""

    return float(min(1.0, max(0.0, float(production_probability) + float(tilt_points))))


def _side(probability: float) -> str:
    return "HOME" if probability >= 0.5 else "AWAY"


def _opposite(side: str) -> str:
    return "AWAY" if side == "HOME" else "HOME"


# ---------------------------------------------------------------------------
# The in-window crew snapshot
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CrewSnapshot:
    """One captured crew-assignment snapshot, already filtered to a week."""

    snapshot_id: str
    captured_at_utc: pd.Timestamp
    referee_by_game_id: dict[str, str]
    row_count: int
    empty_reason: str | None


def latest_crew_snapshot(
    data_root: Path, *, season: int, week: int, before: pd.Timestamp | None = None
) -> CrewSnapshot | None:
    """The newest ``referee_assignments`` snapshot covering ``(season, week)``.

    ``before``, when given, restricts to snapshots captured strictly before
    that instant -- the week-wide pre-filter. The binding per-GAME check is
    still made against each game's own ``pick_deadline`` by the caller; this
    argument only avoids loading a snapshot no game could use.
    """

    root = data_root / "players" / "referee_assignments"
    if not root.is_dir():
        return None
    best: CrewSnapshot | None = None
    for manifest_path in sorted(root.glob("*/manifest.json")):
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if int(manifest.get("season", -1)) != int(season):
            continue
        if int(manifest.get("week", -1)) != int(week):
            continue
        captured_raw = manifest.get("captured_at_utc")
        if captured_raw is None:
            continue
        captured = pd.Timestamp(captured_raw)
        captured = captured.tz_localize("UTC") if captured.tzinfo is None else captured
        if before is not None and captured >= pd.Timestamp(before):
            continue
        referee_by_game: dict[str, str] = {}
        assignments_path = manifest_path.with_name("assignments.parquet")
        if assignments_path.is_file():
            frame = pd.read_parquet(assignments_path)
            for row in frame.itertuples(index=False):
                game_id = getattr(row, "game_id", None)
                referee = getattr(row, "referee", None)
                if game_id is None or referee is None:
                    continue
                if pd.isna(game_id) or pd.isna(referee):
                    continue
                referee_by_game[str(game_id)] = str(referee)
        candidate = CrewSnapshot(
            snapshot_id=manifest_path.parent.name,
            captured_at_utc=captured,
            referee_by_game_id=referee_by_game,
            row_count=int(manifest.get("row_count", len(referee_by_game))),
            empty_reason=manifest.get("empty_reason"),
        )
        if best is None or candidate.captured_at_utc > best.captured_at_utc:
            best = candidate
    return best


def home_pass_rate_quartiles(repo_root: Path, game_ids: list[str]) -> dict[str, int]:
    """The screen's own ``home_pbp_off_pass_rate`` quartile, per game_id.

    Calls ``_merge_home_pass_rate_quartile`` verbatim (the same helper cell C
    uses), so the cutpoint population is identical to the screen's. A game
    absent from ``game_features_pbp.parquet`` -- or carrying a null pass rate
    -- simply has no entry, which the caller treats as "no flag", never as an
    error.
    """

    if not game_ids:
        return {}
    frame = pd.DataFrame({"game_id": [str(game_id) for game_id in game_ids]})
    try:
        merged = _merge_home_pass_rate_quartile(frame, repo_root)
    except Exception:
        return {}
    return {
        str(row.game_id): int(cast(Any, row.home_pass_rate_quartile))
        for row in merged.itertuples(index=False)
    }


# ---------------------------------------------------------------------------
# The refresh-time rows
# ---------------------------------------------------------------------------


def build_crew_tilt_refresh_rows(
    plan: RefreshResult, *, data_root: Path, repo_root: Path
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Pure computation: one row per ELIGIBLE game in ``plan``.

    FAIL-OPEN everywhere: no crew snapshot for the week, or a snapshot no
    game's deadline can accept, returns an EMPTY frame plus
    ``{"skipped": True, "reason": ...}`` -- a documented NO-OP, never an
    exception and never a flip. Never writes anything.
    """

    empty = pd.DataFrame(columns=list(CREW_TILT_REFRESH_COLUMNS))
    eligible_games = [game for game in plan.games if game.eligible]
    if not eligible_games:
        return empty, {"skipped": True, "reason": "no eligible games in this refresh pass"}

    snapshot = latest_crew_snapshot(data_root, season=plan.season, week=plan.week)
    if snapshot is None:
        return empty, {
            "skipped": True,
            "reason": OVERLAY_STATUS_NO_SNAPSHOT,
            "detail": (
                "no data/players/referee_assignments/*/manifest.json snapshot covers "
                f"season {plan.season} week {plan.week}"
            ),
        }

    in_window = [
        game for game in eligible_games if snapshot.captured_at_utc < pd.Timestamp(game.deadline)
    ]
    if not in_window:
        return empty, {
            "skipped": True,
            "reason": OVERLAY_STATUS_SNAPSHOT_AFTER_DEADLINE,
            "detail": (
                f"crew snapshot {snapshot.snapshot_id} captured "
                f"{snapshot.captured_at_utc.isoformat()} is at or after the pick deadline of "
                "every eligible game in this pass"
            ),
            "crew_snapshot_id": snapshot.snapshot_id,
        }

    lookup = build_crew_trait_lookup(repo_root)
    quartiles = home_pass_rate_quartiles(repo_root, [str(game.game_id) for game in in_window])

    rows: list[dict[str, Any]] = []
    out_of_window: list[str] = []
    for game in eligible_games:
        if snapshot.captured_at_utc >= pd.Timestamp(game.deadline):
            out_of_window.append(str(game.game_id))
            continue
        game_id = str(game.game_id)
        referee = snapshot.referee_by_game_id.get(game_id)
        flags = crew_tilt_flags(
            referee=referee,
            season=int(plan.season),
            home_pass_rate_quartile=quartiles.get(game_id),
            decision_home_spread=game.decision_home_spread,
            lookup=lookup,
        )
        production_probability = float(game.new_home_cover_probability)
        tilted = tilted_probability(production_probability, flags.tilt_points)
        crossed = _side(tilted) != _side(production_probability)
        would_be_side = _opposite(game.new_pick_side) if crossed else game.new_pick_side
        rows.append(
            {
                "revision_recorded_at_utc": plan.computed_at_utc,
                "refresh_run_id": plan.refresh_run_id,
                "season": plan.season,
                "week": plan.week,
                "game_id": game_id,
                "home_team": game.home_team,
                "away_team": game.away_team,
                "kickoff": game.kickoff,
                "deadline": game.deadline,
                "decision_home_spread": game.decision_home_spread,
                "played_pick_side": game.new_pick_side,
                "production_home_cover_probability": production_probability,
                "crew_snapshot_id": snapshot.snapshot_id,
                "crew_captured_at_utc": snapshot.captured_at_utc,
                "referee": flags.referee,
                "holding_crew_top_quartile": flags.holding_crew_top_quartile,
                "home_run_heavy_bottom_quartile": flags.home_run_heavy_bottom_quartile,
                "holding_tilt_flag": flags.holding_tilt_flag,
                "flag_rate_crew_top_quartile": flags.flag_rate_crew_top_quartile,
                "home_heavy_underdog": flags.home_heavy_underdog,
                "high_flag_underdog_flag": flags.high_flag_underdog_flag,
                "tilt_points": flags.tilt_points,
                "tilted_home_cover_probability": tilted,
                "crew_would_be_pick_side": would_be_side,
                "crew_tilt_flip": bool(crossed),
                "overlay_status": flags.status,
                "model_id": plan.model_id,
                "feature_table_sha256": plan.feature_table_sha256,
            }
        )

    frame = pd.DataFrame(rows, columns=list(CREW_TILT_REFRESH_COLUMNS))
    diagnostics = {
        "skipped": False,
        "crew_snapshot_id": snapshot.snapshot_id,
        "crew_captured_at_utc": snapshot.captured_at_utc,
        "crew_snapshot_empty_reason": snapshot.empty_reason,
        "crew_snapshot_row_count": snapshot.row_count,
        "officials_snapshot_id": lookup.officials_snapshot_id,
        "penalty_type_snapshot_id": lookup.penalty_type_snapshot_id,
        "games_considered": len(frame),
        "snapshot_after_deadline_skipped_game_ids": out_of_window,
        "holding_tilt_flag_game_ids": frame.loc[frame["holding_tilt_flag"], "game_id"]
        .astype(str)
        .tolist(),
        "high_flag_underdog_flag_game_ids": frame.loc[frame["high_flag_underdog_flag"], "game_id"]
        .astype(str)
        .tolist(),
        "both_cells_flagged_game_ids": frame.loc[
            frame["holding_tilt_flag"] & frame["high_flag_underdog_flag"], "game_id"
        ]
        .astype(str)
        .tolist(),
        "would_flip_game_ids": frame.loc[frame["crew_tilt_flip"], "game_id"].astype(str).tolist(),
        "status_counts": frame["overlay_status"].value_counts().to_dict(),
    }
    return frame, diagnostics


def record_crew_tilt_refresh_overlay(
    artifacts_root: Path,
    data_root: Path,
    plan: RefreshResult,
    *,
    repo_root: Path,
    record_decisions: bool = False,
) -> dict[str, Any]:
    """Append this pass's would-be picks to the crew-tilt overlay ledger.

    Mirrors ``nfl_ats.nflcom_refresh_overlay.record_nflcom_refresh_overlay``'s
    opt-in ``record_decisions`` contract and reuses
    ``refuse_if_outside_recording_lock_window`` against the week's ORIGINAL
    card kickoffs unchanged. The PLAYED pipeline cannot see this function's
    output: it writes only its own separate ledger, and the ``RefreshResult``
    handed in is consumed strictly read-only.

    Repeated passes across a week legitimately append MULTIPLE rows per game
    (not deduped), mirroring the sibling refresh ledgers: how the flag
    evolves across passes is part of what prospective scoring reads. Scoring
    consumes the LATEST pre-kickoff row per game.
    """

    if not record_decisions:
        return {
            "challenger_id": CHALLENGER_ID,
            "recorded": 0,
            "skipped": True,
            "reason": (
                "pass --record-decisions to append this pass's would-be picks to the "
                "crew-tilt refresh ledger"
            ),
        }

    original = original_card(artifacts_root, season=plan.season, week=plan.week)
    refuse_if_outside_recording_lock_window(
        original["kickoff"], plan.computed_at_utc, ledger="crew-tilt-refresh-overlay"
    )

    rows, diagnostics = build_crew_tilt_refresh_rows(plan, data_root=data_root, repo_root=repo_root)
    existing = load_crew_tilt_refresh_decisions(artifacts_root)
    if rows.empty:
        return {
            "challenger_id": CHALLENGER_ID,
            "recorded": 0,
            "ledger_rows": len(existing),
            **diagnostics,
        }

    combined = pd.concat([existing, rows], ignore_index=True) if not existing.empty else rows
    atomic_parquet(
        combined[list(CREW_TILT_REFRESH_COLUMNS)],
        crew_tilt_refresh_ledger_path(artifacts_root),
    )
    return {
        "challenger_id": CHALLENGER_ID,
        "recorded": len(rows),
        "ledger_rows": len(combined),
        **diagnostics,
    }


# ---------------------------------------------------------------------------
# Card-level application (the historical / back-test path)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CrewTiltOverlayResult:
    """The overlay applied to a card-shaped predictions frame."""

    predictions: pd.DataFrame
    flip_game_ids: tuple[str, ...]
    holding_flag_game_ids: tuple[str, ...]
    underdog_flag_game_ids: tuple[str, ...]
    both_flag_game_ids: tuple[str, ...]
    n_with_referee: int


def historical_crew_by_game(repo_root: Path) -> pd.DataFrame:
    """The screen's own per-game referee/quartile table, from the builders.

    Point-in-time-EQUIVALENT stand-in for the capture: a week's crew
    assignment is public by Wednesday (docs/referee_assignments_capture.md
    section 2), so knowing who refereed a completed game is not knowing
    anything a Wednesday forecaster could not have known. It is used ONLY by
    the back-test path; the live path reads the captured snapshot.
    """

    holding = _build_referee_type_trait_data(repo_root, _HOLDING_PENALTY_TYPE).game_trait
    flag_rate = _build_referee_trait_data(repo_root).game_trait
    holding = holding.loc[:, ["game_id", "official_name", "season", "lag_type_quartile"]]
    flag_rate = flag_rate.loc[:, ["game_id", "lag_penalty_rate_quartile"]]
    merged = holding.merge(flag_rate, on="game_id", how="outer")
    merged["game_id"] = merged["game_id"].astype(str)
    return merged.drop_duplicates("game_id").reset_index(drop=True)


def apply_crew_tilt_refresh_overlay(
    predictions: pd.DataFrame, repo_root: Path
) -> CrewTiltOverlayResult:
    """Apply the frozen tilt to a card-shaped frame, historical-crew path.

    ``predictions`` must carry ``game_id``/``season``/``spread_line``/
    ``home_cover_probability`` (the schema every sibling overlay's ``apply_*``
    consumes). The returned frame's ``home_cover_probability`` is the TILTED
    probability; ``flip_game_ids`` are the games whose 0.5 side changed.
    """

    required = {"game_id", "season", "spread_line", "home_cover_probability"}
    missing = sorted(required.difference(predictions.columns))
    if missing:
        raise DataContractError(f"crew-tilt overlay needs columns: {', '.join(missing)}")

    frame = predictions.copy()
    frame["game_id"] = frame["game_id"].astype(str)
    crew = historical_crew_by_game(repo_root)
    frame = frame.merge(crew, on="game_id", how="left")
    quartiles = home_pass_rate_quartiles(repo_root, frame["game_id"].tolist())
    frame["home_pass_rate_quartile"] = frame["game_id"].map(quartiles)

    holding_top = frame["lag_type_quartile"].eq(TOP_QUARTILE)
    run_heavy = frame["home_pass_rate_quartile"].eq(BOTTOM_QUARTILE)
    holding_flag = holding_top & run_heavy
    flag_rate_top = frame["lag_penalty_rate_quartile"].eq(TOP_QUARTILE)
    heavy_underdog = pd.to_numeric(frame["spread_line"], errors="coerce").le(
        -HEAVY_UNDERDOG_THRESHOLD
    )
    underdog_flag = flag_rate_top & heavy_underdog

    tilt = holding_flag.astype(float) * HOLDING_RUN_HEAVY_TILT
    tilt = tilt + underdog_flag.astype(float) * HIGH_FLAG_UNDERDOG_TILT
    base = pd.to_numeric(frame["home_cover_probability"], errors="coerce")
    tilted = (base + tilt).clip(lower=0.0, upper=1.0)
    flipped = base.ge(0.5).ne(tilted.ge(0.5))

    out = predictions.copy()
    out["game_id"] = out["game_id"].astype(str)
    out["home_cover_probability"] = tilted.to_numpy()
    return CrewTiltOverlayResult(
        predictions=out,
        flip_game_ids=tuple(frame.loc[flipped, "game_id"].astype(str)),
        holding_flag_game_ids=tuple(frame.loc[holding_flag, "game_id"].astype(str)),
        underdog_flag_game_ids=tuple(frame.loc[underdog_flag, "game_id"].astype(str)),
        both_flag_game_ids=tuple(frame.loc[holding_flag & underdog_flag, "game_id"].astype(str)),
        n_with_referee=int(frame["official_name"].notna().sum()),
    )


# ---------------------------------------------------------------------------
# Week preview (never a ledger write)
# ---------------------------------------------------------------------------


PREVIEW_COLUMNS: tuple[str, ...] = (
    "season",
    "week",
    "game_id",
    "home_team",
    "away_team",
    "kickoff",
    "deadline",
    "crew_snapshot_id",
    "crew_captured_at_utc",
    "snapshot_in_window",
    "referee",
    "decision_home_spread",
    "holding_tilt_flag",
    "high_flag_underdog_flag",
    "tilt_points",
    "keeps_incumbent_pick",
    "overlay_status",
)


def preview_week(
    data_root: Path,
    repo_root: Path,
    *,
    season: int,
    week: int,
    now: pd.Timestamp | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """What the overlay WOULD do for one week, read-only, no ledger write.

    Works without a published card: the incumbent pick is whatever the
    Tuesday card says, and this preview reports only whether the overlay
    would leave it alone (``tilt_points == 0`` -> ``keeps_incumbent_pick``).
    It exists so a lock-day rehearsal can show the challenger equals the
    incumbent before any real row is ever written.
    """

    del now
    schedules = pd.read_parquet(_latest_schedules_snapshot(repo_root))
    week_games = schedules.loc[
        (schedules["season"].astype(int) == int(season))
        & (schedules["week"].astype(int) == int(week))
        & (schedules["game_type"] == "REG")
    ].copy()
    if week_games.empty:
        return pd.DataFrame(columns=list(PREVIEW_COLUMNS)), {
            "reason": "no REG games in the newest schedule snapshot for this week",
            "season": season,
            "week": week,
        }

    kickoffs = pd.to_datetime(week_games["gameday"].astype(str), errors="coerce")
    if "gametime" in week_games.columns:
        combined = week_games["gameday"].astype(str) + " " + week_games["gametime"].astype(str)
        parsed = pd.to_datetime(combined, errors="coerce")
        kickoffs = parsed.fillna(kickoffs)
    kickoffs = kickoffs.dt.tz_localize(
        "America/New_York", ambiguous=True, nonexistent="shift_forward"
    )
    kickoffs_utc = kickoffs.dt.tz_convert("UTC")
    week_games = week_games.assign(crew_kickoff_utc=kickoffs_utc.to_numpy())

    from nfl_ats.pick_refresh import pick_deadline, sunday_pick_lock

    lock = sunday_pick_lock(pd.Series(week_games["crew_kickoff_utc"]))
    snapshot = latest_crew_snapshot(data_root, season=season, week=week)
    lookup: CrewTraitLookup | None = None
    quartiles: dict[str, int] = {}
    if snapshot is not None and snapshot.referee_by_game_id:
        lookup = build_crew_trait_lookup(repo_root)
        quartiles = home_pass_rate_quartiles(
            repo_root, [str(g) for g in week_games["game_id"].astype(str)]
        )

    rows: list[dict[str, Any]] = []
    for row in week_games.itertuples(index=False):
        game_id = str(row.game_id)
        kickoff = pd.Timestamp(cast(Any, row.crew_kickoff_utc))
        deadline = pick_deadline(kickoff, lock)
        in_window = snapshot is not None and snapshot.captured_at_utc < deadline
        referee = (
            snapshot.referee_by_game_id.get(game_id)
            if (snapshot is not None and in_window)
            else None
        )
        spread = getattr(row, "spread_line", None)
        if lookup is not None and referee is not None:
            flags = crew_tilt_flags(
                referee=referee,
                season=int(season),
                home_pass_rate_quartile=quartiles.get(game_id),
                decision_home_spread=None if spread is None or pd.isna(spread) else float(spread),
                lookup=lookup,
            )
            status = flags.status
            holding_flag = flags.holding_tilt_flag
            underdog_flag = flags.high_flag_underdog_flag
            tilt = flags.tilt_points
        else:
            holding_flag = False
            underdog_flag = False
            tilt = 0.0
            if snapshot is None:
                status = OVERLAY_STATUS_NO_SNAPSHOT
            elif not in_window:
                status = OVERLAY_STATUS_SNAPSHOT_AFTER_DEADLINE
            else:
                status = OVERLAY_STATUS_NO_ROW
        rows.append(
            {
                "season": int(season),
                "week": int(week),
                "game_id": game_id,
                "home_team": str(row.home_team),
                "away_team": str(row.away_team),
                "kickoff": kickoff,
                "deadline": deadline,
                "crew_snapshot_id": None if snapshot is None else snapshot.snapshot_id,
                "crew_captured_at_utc": None if snapshot is None else snapshot.captured_at_utc,
                "snapshot_in_window": bool(in_window),
                "referee": referee,
                "decision_home_spread": None
                if spread is None or pd.isna(spread)
                else float(spread),
                "holding_tilt_flag": bool(holding_flag),
                "high_flag_underdog_flag": bool(underdog_flag),
                "tilt_points": float(tilt),
                "keeps_incumbent_pick": tilt == 0.0,
                "overlay_status": status,
            }
        )

    frame = pd.DataFrame(rows, columns=list(PREVIEW_COLUMNS))
    summary = {
        "challenger_id": CHALLENGER_ID,
        "season": int(season),
        "week": int(week),
        "games": len(frame),
        "crew_snapshot_id": None if snapshot is None else snapshot.snapshot_id,
        "crew_snapshot_empty_reason": None if snapshot is None else snapshot.empty_reason,
        "crew_snapshot_row_count": None if snapshot is None else snapshot.row_count,
        "sunday_pick_lock_utc": lock.isoformat(),
        "games_keeping_incumbent_pick": int(frame["keeps_incumbent_pick"].sum()),
        "games_tilted": int((~frame["keeps_incumbent_pick"]).sum()),
        "status_counts": frame["overlay_status"].value_counts().to_dict(),
        "ledger_write": "none -- preview only",
    }
    return frame, summary


# ---------------------------------------------------------------------------
# Stacked-on-production back-test (context, never a gate)
# ---------------------------------------------------------------------------


def _load_script_module(repo_root: Path, name: str) -> Any:
    """Load a ``scripts/*.py`` helper by path.

    Loaded dynamically rather than imported so ``mypy src`` is not dragged
    into the ``scripts`` package (which it was never configured to gate) --
    the same isolation ``pyproject.toml``'s per-script ``ignore_errors``
    overrides buy for the modules that DO import scripts statically.
    """

    path = repo_root / "scripts" / f"{name}.py"
    scripts_dir = str(repo_root / "scripts")
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:  # pragma: no cover - defensive
        raise DataContractError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _stacked_metric(frame: pd.DataFrame) -> dict[str, float]:
    valid = frame.dropna(subset=["correct_baseline"])
    if valid.empty:  # pragma: no cover - defensive
        return {
            "baseline_accuracy": float("nan"),
            "production_accuracy": float("nan"),
            "candidate_accuracy": float("nan"),
            "candidate_minus_production": float("nan"),
            "production_minus_baseline": float("nan"),
        }
    baseline = float(valid["correct_baseline"].mean())
    production = float(valid["correct_production"].mean())
    candidate = float(valid["correct_candidate"].mean())
    return {
        "baseline_accuracy": baseline,
        "production_accuracy": production,
        "candidate_accuracy": candidate,
        "candidate_minus_production": candidate - production,
        "production_minus_baseline": production - baseline,
    }


def _solo_metric(frame: pd.DataFrame) -> dict[str, float]:
    valid = frame.dropna(subset=["correct_baseline"])
    if valid.empty:  # pragma: no cover - defensive
        return {"baseline_accuracy": float("nan"), "solo_minus_baseline": float("nan")}
    baseline = float(valid["correct_baseline"].mean())
    solo = float(valid["correct_solo_crew_tilt"].mean())
    return {
        "baseline_accuracy": baseline,
        "solo_accuracy": solo,
        "solo_minus_baseline": solo - baseline,
    }


def run_stacked_backtest(
    repo_root: Path,
    *,
    per_game_artifact: Path,
    data_root: Path,
    features_path: Path,
    incidents_path: Path,
    samples: int,
    seed: int,
) -> dict[str, Any]:
    """The overlay on top of the played four-overlay chain, opener grade.

    MINED-SEASONS context, declared NOT a gate in
    ``docs/referee_assignments_capture.md`` (WP47 section 5) before it was
    ever run. Spends no rotation-registry window.
    """

    stack = _load_script_module(repo_root, "overlay_stack_backtest")
    composition = _load_script_module(repo_root, "overlay_subset_composition")
    from nfl_ats.four_overlay_composition import (
        COACH_FADE,
        COMPOSITION_ORDER,
        DIVISION_REVENGE_TILT,
        PLAYER_ARRESTS_BACK_SIDE_POLICY,
        POLICY_ID,
        SPREAD_GAP_ZONE_FADE,
    )

    per_game, schedules, player_features, snapshot_name, player_feature_path = stack.load_inputs(
        per_game_artifact, data_root
    )
    predictions = stack.build_predictions_frame(per_game, schedules)

    overlay_results = stack.run_overlays(predictions, schedules, player_features)
    overlay_flip_sets = {
        name: {flip.game_id for flip in result.flips} for name, result in overlay_results.items()
    }
    stack.verify_no_direction_conflicts(predictions, overlay_results, overlay_flip_sets)
    arrest_ids, _scored = composition.reconstruct_arrest_flip_set(
        per_game, features_path, incidents_path
    )
    members: dict[str, set[str]] = {
        COACH_FADE: overlay_flip_sets["coach_fade_overlay"],
        DIVISION_REVENGE_TILT: overlay_flip_sets["division_revenge_tilt_overlay"],
        SPREAD_GAP_ZONE_FADE: overlay_flip_sets["spread_gap_zone_fade_overlay"],
        PLAYER_ARRESTS_BACK_SIDE_POLICY: arrest_ids,
    }
    missing = [member for member in COMPOSITION_ORDER if member not in members]
    if missing:  # pragma: no cover - defensive
        raise DataContractError(f"production chain members not reconstructed: {missing}")
    production_ids: set[str] = set().union(*members.values())

    # The production probability IS the raw probability complemented on every
    # game the four-overlay union flips -- the same transform every member
    # overlay applies (see division_revenge_tilt_overlay's own flip path).
    production_predictions = predictions.copy()
    production_predictions["game_id"] = production_predictions["game_id"].astype(str)
    flipped_mask = production_predictions["game_id"].isin(production_ids)
    production_predictions.loc[flipped_mask, "home_cover_probability"] = (
        1.0 - production_predictions.loc[flipped_mask, "home_cover_probability"]
    )

    stacked = apply_crew_tilt_refresh_overlay(production_predictions, repo_root)
    solo = apply_crew_tilt_refresh_overlay(predictions, repo_root)

    empty_flip_sets: dict[str, set[str]] = {name: set() for name in stack.OVERLAY_NAMES}
    eval_frame = stack.build_eval_frame(predictions, per_game, empty_flip_sets)
    eval_frame = eval_frame[["game_id", "season", "week", "correct_baseline"]].copy()
    eval_frame["game_id"] = eval_frame["game_id"].astype(str)

    base = eval_frame["correct_baseline"]
    eval_frame["in_production"] = eval_frame["game_id"].isin(production_ids)
    eval_frame["correct_production"] = np.where(eval_frame["in_production"], 1.0 - base, base)
    stacked_flips = set(stacked.flip_game_ids)
    eval_frame["in_crew_tilt"] = eval_frame["game_id"].isin(stacked_flips)
    eval_frame["correct_candidate"] = np.where(
        eval_frame["in_crew_tilt"],
        1.0 - eval_frame["correct_production"],
        eval_frame["correct_production"],
    )
    solo_flips = set(solo.flip_game_ids)
    eval_frame["correct_solo_crew_tilt"] = np.where(
        eval_frame["game_id"].isin(solo_flips), 1.0 - base, base
    )

    scored = eval_frame.dropna(subset=["correct_baseline"])
    stacked_stats = stack.run_both_blockings(
        eval_frame, _stacked_metric, samples=samples, seed=seed
    )
    solo_stats = stack.run_both_blockings(eval_frame, _solo_metric, samples=samples, seed=seed)

    per_season = (
        eval_frame.loc[eval_frame["in_crew_tilt"]].groupby("season").size().astype(int).to_dict()
    )
    return {
        "computed_at_utc": datetime.now(UTC).isoformat(),
        "challenger_id": CHALLENGER_ID,
        "signal_name": STACKED_SIGNAL_NAME,
        "rule": (
            "At a late-week refresh pass, tilt the production home-cover probability by each "
            "flagged cell's own measured per-game gap -- cell C "
            f"({HOLDING_RUN_HEAVY_SIGNAL}) {HOLDING_RUN_HEAVY_TILT:+.8f} when the home team is "
            "run-heavy (bottom pass-rate quartile) and the referee's prior-season Offensive "
            f"Holding rate is top-quartile; cell A ({HIGH_FLAG_UNDERDOG_SIGNAL}) "
            f"{HIGH_FLAG_UNDERDOG_TILT:+.8f} when the referee's prior-season mean_total rate is "
            "top-quartile and the home team is getting >= 7 points at the frozen Tuesday line. "
            "Additive composition (never measured); the pick flips only when the tilt crosses "
            "0.5."
        ),
        "read_kind": (
            "MINED-SEASONS read on an already-looked-at archive; CONTEXT, not a gate. The "
            "historical crew from the officials/PBP join stands in for the Wednesday capture, "
            "which is point-in-time-EQUIVALENT for crew identity only. No rotation-registry "
            "window is spent."
        ),
        "closing_grounds_taxonomy": (
            "An interval containing zero is NEVER grounds to reject, fail or close an "
            "experiment. Only (1) a refuted mechanism -- a RESOLVED wrong sign (whole "
            "interval on the wrong side of zero) or zero split-half reliability -- or "
            "(2) bounded by a positive control proven able to detect an effect that size "
            "closes a line of work. Everything else is unresolved_below_power: report "
            "probability_positive, never the binary 'contains zero'."
        ),
        "source_artifact": str(per_game_artifact),
        "schedule_snapshot": snapshot_name,
        "player_feature_table": str(player_feature_path),
        "arrest_features": str(features_path),
        "arrest_incidents": str(incidents_path),
        "incumbent_policy_id": POLICY_ID,
        "incumbent_members": list(COMPOSITION_ORDER),
        "incumbent_member_flip_counts": {member: len(ids) for member, ids in members.items()},
        "production_union_flip_count": len(production_ids),
        "grading_rule": (
            "production probability rule (home_cover_probability >= 0.5) at the "
            "Tuesday-opener decision line"
        ),
        "tilt_points": {
            HOLDING_RUN_HEAVY_SIGNAL: HOLDING_RUN_HEAVY_TILT,
            HIGH_FLAG_UNDERDOG_SIGNAL: HIGH_FLAG_UNDERDOG_TILT,
        },
        "seasons": [int(eval_frame["season"].min()), int(eval_frame["season"].max())],
        "n_games": len(eval_frame),
        "n_pushes": int(eval_frame["correct_baseline"].isna().sum()),
        "n_scored_games": len(scored),
        "week_block_count": int(eval_frame[["season", "week"]].drop_duplicates().shape[0]),
        "season_block_count": int(eval_frame["season"].nunique()),
        "bootstrap_samples": samples,
        "bootstrap_seed": seed,
        "n_games_with_referee": stacked.n_with_referee,
        "n_holding_flag": len(stacked.holding_flag_game_ids),
        "n_underdog_flag": len(stacked.underdog_flag_game_ids),
        "n_both_flag": len(stacked.both_flag_game_ids),
        "n_flipped_stacked": len(stacked_flips),
        "n_flipped_solo": len(solo_flips),
        "flipped_stacked_game_ids": sorted(stacked_flips),
        "flipped_solo_game_ids": sorted(solo_flips),
        "flips_per_season": {str(k): int(v) for k, v in per_season.items()},
        "stacked_on_production": stacked_stats,
        "solo_vs_bare_baseline": solo_stats,
    }


DEFAULT_PER_GAME_ARTIFACT = Path("artifacts/opener_evaluation/20260819T174244Z/per_game.parquet")
DEFAULT_ARREST_INCIDENTS = Path(
    "data/raw/player_arrests/20260820T153000Z/incidents_point_in_time.parquet"
)
DEFAULT_ARREST_FEATURES = Path("data/processed/game_features_pbp.parquet")
DEFAULT_OUTPUT_ROOT = Path("artifacts/crew_tilt_refresh")
DEFAULT_SAMPLES = 20_000
#: Fixed, recorded seed -- this session's UTC date, chosen before any result
#: was seen, matching every sibling back-test script's convention.
DEFAULT_SEED = 20260901


def main(argv: list[str] | None = None) -> int:
    """``python -m nfl_ats.crew_tilt_refresh_overlay`` entry point.

    Two modes: ``backtest`` (the stacked-on-production context run) and
    ``preview`` (one week, read-only, never a ledger write).
    """

    repo_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)

    backtest = sub.add_parser("backtest")
    backtest.add_argument("--per-game-artifact", type=Path, default=DEFAULT_PER_GAME_ARTIFACT)
    backtest.add_argument("--data-root", type=Path, default=Path("data"))
    backtest.add_argument("--features", type=Path, default=DEFAULT_ARREST_FEATURES)
    backtest.add_argument("--incidents", type=Path, default=DEFAULT_ARREST_INCIDENTS)
    backtest.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    backtest.add_argument("--samples", type=int, default=DEFAULT_SAMPLES)
    backtest.add_argument("--seed", type=int, default=DEFAULT_SEED)

    preview = sub.add_parser("preview")
    preview.add_argument("--data-root", type=Path, default=Path("data"))
    preview.add_argument("--season", type=int, required=True)
    preview.add_argument("--week", type=int, required=True)

    args = parser.parse_args(argv)

    if args.mode == "preview":
        frame, summary = preview_week(args.data_root, repo_root, season=args.season, week=args.week)
        print(json.dumps(summary, indent=2, sort_keys=True, default=str))
        if not frame.empty:
            columns = [
                "game_id",
                "home_team",
                "away_team",
                "snapshot_in_window",
                "referee",
                "tilt_points",
                "keeps_incumbent_pick",
                "overlay_status",
            ]
            print(frame[columns].to_string(index=False))
        return 0

    from nfl_ats.provenance import artifact_provenance, write_experiment_artifact

    result = run_stacked_backtest(
        repo_root,
        per_game_artifact=args.per_game_artifact,
        data_root=args.data_root,
        features_path=args.features,
        incidents_path=args.incidents,
        samples=args.samples,
        seed=args.seed,
    )
    configuration = {
        "command": "crew-tilt-refresh-stacked-backtest",
        "per_game_artifact": str(args.per_game_artifact),
        "data_root": str(args.data_root),
        "bootstrap_samples": args.samples,
        "bootstrap_seed": args.seed,
        "challenger_id": CHALLENGER_ID,
        "predeclaration": "docs/referee_assignments_capture.md (WP47 section)",
    }
    payload = {
        **result,
        "provenance": artifact_provenance(
            configuration, args.per_game_artifact, project_root=repo_root
        ),
    }
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    output_dir = args.output_root / timestamp
    stacked = payload["stacked_on_production"]["week_blocked"]["candidate_minus_production"]
    write_experiment_artifact(
        output_dir,
        "results.json",
        payload,
        command="crew-tilt-refresh-stacked-backtest",
        metrics={
            "candidate_minus_production_accuracy_points": stacked["estimate"] * 100.0,
            "probability_positive": stacked["probability_positive"],
            "n_flipped_stacked": payload["n_flipped_stacked"],
            "n_scored_games": payload["n_scored_games"],
        },
        notes=(
            "Late-week officiating-crew tilt (registry cells "
            f"{HOLDING_RUN_HEAVY_SIGNAL} and {HIGH_FLAG_UNDERDOG_SIGNAL}) stacked on the played "
            "four-overlay production chain at the Tuesday opener grade. MINED-SEASONS context, "
            "not a gate; spends no rotation-registry window. See "
            "docs/referee_assignments_capture.md (WP47 section)."
        ),
        project_root=repo_root,
    )
    print(f"Wrote {output_dir / 'results.json'}")
    print(
        f"games with a resolvable crew={payload['n_games_with_referee']} "
        f"holding_flag={payload['n_holding_flag']} underdog_flag={payload['n_underdog_flag']} "
        f"both={payload['n_both_flag']} flips_stacked={payload['n_flipped_stacked']} "
        f"scored={payload['n_scored_games']}"
    )
    for block in ("week_blocked", "season_blocked"):
        row = payload["stacked_on_production"][block]["candidate_minus_production"]
        print(
            f"candidate - production ({block}): {row['estimate'] * 100:+.4f} pts "
            f"95% [{row['lower'] * 100:+.4f}, {row['upper'] * 100:+.4f}] "
            f"P+ {row['probability_positive']:.4f}"
        )
    accuracy = payload["stacked_on_production"]["week_blocked"]
    print(
        "accuracies: baseline "
        f"{accuracy['baseline_accuracy']['estimate'] * 100:.4f}%, production "
        f"{accuracy['production_accuracy']['estimate'] * 100:.4f}%, candidate "
        f"{accuracy['candidate_accuracy']['estimate'] * 100:.4f}%"
    )
    solo_row = payload["solo_vs_bare_baseline"]["week_blocked"]["solo_minus_baseline"]
    print(
        f"context -- solo vs bare baseline (week-blocked): {solo_row['estimate'] * 100:+.4f} pts "
        f"95% [{solo_row['lower'] * 100:+.4f}, {solo_row['upper'] * 100:+.4f}] "
        f"P+ {solo_row['probability_positive']:.4f}"
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
