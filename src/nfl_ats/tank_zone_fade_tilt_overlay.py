"""Tank-zone fade tilt overlay: a parameter-free pick-level flip, weeks 14-18.

Research chain (measured 2026-08-21 by ``scripts/motivation_ladder_screen.py``,
predeclared in ``docs/motivation_ladder_screen.md`` cell M4 BEFORE the screen
scored anything, and read out of
``artifacts/motivation_ladder_screen/20260821T182643Z/results.json`` and
``registry/weak_signals.json`` before this module was built):

``motivation_ladder_tank_zone_wk14_18`` flags a team whose record places it in
the league's BOTTOM TWO league-wide (the "#1-overall-pick tank zone"), in weeks
14-18 only. Population: NFL REG close-graded slate 2009-2025, team-perspective
long table, weeks 13-18, n=2,768 team-games, n_flag=144 (5.20% of the slate).
Week-blocked primary: full-slate effect **+0.3049 accuracy points**, 95%
**[-0.0794, +0.6966]**, ``probability_positive`` **0.9334**, 90 week blocks.
Season-blocked secondary: **+0.3049**, 95% **[+0.0371, +0.6118]**,
``probability_positive`` **0.9856**, 17 season blocks.

**The interval crosses zero at the primary blocking.** Per AGENTS.md, at this
evaluator's ~2-point resolution that is the EXPECTED shape for a real-but-small
signal and is NEVER grounds to decline building a no-window-cost prospective
challenger. Neither admissible closing ground applies (no resolved wrong sign,
no positive-control bound), so the cell stays ``unresolved_below_power`` in the
registry; wiring it here is an EV-positive dual-tracked play (P+ 0.9334 > 0.5),
not a claim of a proven edge.

**Registry-description correction (carry this forward).** The registry entry's
``description`` field reads "leans OPPOSITE tank-fade prediction (tank teams
over-cover)", and ``docs/motivation_ladder_screen.md``'s M4 classification
bullet repeats it. **Both are wrong about the direction, and the artifact says
so.** Measured, from
``artifacts/motivation_ladder_screen/20260821T182643Z/results.json`` cell
``tank_zone_wk14_18``: ``sign_dir`` is ``-1`` (the predeclared direction is
NEGATIVE on ``team_covered`` -- i.e. FADE the tank-zone team),
``subset_mean`` is **0.4444** and ``complement_mean`` is **0.5030**. Tank-zone
teams covered **44.4%** against a **50.3%** complement: they UNDER-covered, and
the screen's own sign convention (``full_slate_effect_pts`` positive =
prediction CONFIRMED, ``scripts/motivation_ladder_screen.py:355-361``) is why
the recorded effect is **+0.3049** rather than negative. The predeclared FADE
direction is the direction the data shows. The season-blocked secondary
excludes zero on the CONFIRMING side, not the opposite side. This module uses
the predeclared fade direction; the registry ``description`` string is
misleading and is flagged for correction rather than edited here (registry JSON
is written only through the CLI).

**The rule is parameter-free and frozen** -- no threshold, no tuning, nothing
fitted to outcomes. REG season, weeks 14-18 only. Build the tank-zone flag for
both teams. If EXACTLY ONE of the two teams is flagged AND the active model's
own forced pick IS that team, flip the pick to the other side. Both-flagged
games are never touched (the same clean-case handling
``coach_fade_overlay``/``interim_hc_first_game_tilt_overlay`` use: no measured
direction when both sides carry the flag). Never flip in any other situation.
The two constants -- weeks **14-18** and **bottom two** league-wide -- are the
registry cell's own flag definition, not choices made here: see
``scripts/motivation_ladder_screen.py:532-550`` (``population["week"].between(14, 18)``
inside the M4 cell) and ``scripts/motivation_ladder_screen.py:164-165``
(``league_ordered = sorted(DIVISIONS, key=lambda t: (tallies[t][0], -tallies[t][1], t))``
then ``tank_zone = set(league_ordered[:2])``).

**Standings-convention disclosure -- a deliberate, disclosed adaptation.** The
screen's standings snapshots are taken per ``gameday``: for each distinct
gameday in a season, the state is computed from every game on STRICTLY EARLIER
gamedays (``scripts/motivation_ladder_screen.py:192-199``). That is
point-in-time safe for a historical replay, but it is NOT available at this
project's Tuesday recording lock, because a Sunday game's snapshot under that
convention already includes that same week's Thursday-night result. This live
overlay therefore computes the standings from every completed game in STRICTLY
PRIOR WEEKS of the same season -- the repo's standard "prior games only"
convention (``coach_fade_overlay``, ``backup_qb_fade_overlay``,
``division_revenge_tilt_overlay``, ``forecast_cold_visitor_tilt_overlay``'s
``climatology_deviation_disclosure``), and exactly what a Tuesday-lock snapshot
can actually see, since the current week's games have no ``result`` yet.
Measured cost of the adaptation on the registry cell's own population
(``data/raw/20260817T235649Z/schedules.parquet``, the snapshot the screen ran
against): the week-granular flag fires on **143** of the screen's **144**
flagged team-games -- 99.96% agreement, one team-game differs, none added --
and the flagged cover rate moves from 44.44% to 44.76% against a 50.29%
complement. The registered **+0.3049 / P+ 0.9334** figures therefore do NOT
transfer exactly to this live arm; the 2026 prospective ledger accrues fresh
evidence for THIS construction.

Two further verbatim-port notes, so the construct is auditable:

* **Push-dropped tallies.** The screen builds its standings timeline from the
  frame ``load_schedules`` already filtered to ``home_cover.notna()``
  (``scripts/motivation_ladder_screen.py:92-98`` then ``:453``), so a prior game
  that PUSHED against the spread contributes nothing to wins/losses. Kept
  verbatim here, because it is part of the construct that produced the measured
  numbers, and it is pregame-known for prior games. A game with no ``result``
  yet is excluded by the same filter, which is exactly the behaviour a live
  Tuesday run needs.
* **League membership.** The screen ranks the hardcoded 32-team ``DIVISIONS``
  map (``scripts/motivation_ladder_screen.py:53-66``). This module derives the
  ranked set from the season's own schedule instead, which is measured to be
  EXACTLY equivalent on every season 2009-2025 (each season's team set equals
  ``set(DIVISIONS)``, 32 teams, verified this session) and additionally works on
  a test fixture with synthetic team codes.

This module is the no-window-cost path, built on the exact pattern of
``surface_switch_tilt_overlay.py``, ``interim_hc_first_game_tilt_overlay.py``
and ``coach_fade_overlay.py``: a **pick-level, post-prediction transform** of
the active model's own forced pick, dual-tracked against that same active model
in the prospective challenger ledger (``nfl_ats.prospective_scoring``), at no
rotation-registry window cost and with zero training-time feature changes.
**Nothing in this module is wired into ``publishing.py`` or the production pick
path** -- no owner decision to play this on the real card has been made; it is
dual-tracked only.

Two things live here, mirroring the sibling overlays exactly:

1. :func:`tank_zone_flag_by_game` -- the pregame-safe, DATA-DERIVED signal,
   ported from ``scripts/motivation_ladder_screen.py``'s own tally loop and
   league ordering, read straight from the newest local schedule snapshot
   (``data/raw/<snapshot>/schedules.parquet``), never hand-typed.
2. :func:`apply_tank_zone_fade_tilt_overlay` -- the pick-level transform, plus
   :func:`overlay_disclosure_note` for the plain-English provenance sentence.

:func:`record_tank_zone_fade_tilt_challenger_decisions` writes the overlay's own
arm to the prospective challenger ledger so 2026 scores it cleanly, independent
of whether it is ever played on the real card.
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
from nfl_ats.features import add_ats_outcomes
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
CHALLENGER_ID = "tank_zone_fade_tilt_overlay"

#: The registered cell's own week window (scripts/motivation_ladder_screen.py:532-550,
#: ``population["week"].between(14, 18)`` inside the M4 flag). Not a tuned choice.
OVERLAY_WEEK_MIN = 14
OVERLAY_WEEK_MAX = 18

#: "bottom two league-wide records" -- scripts/motivation_ladder_screen.py:164-165's
#: ``tank_zone = set(league_ordered[:2])``. Not a tuned choice.
TANK_ZONE_SIZE = 2

#: The merged-in flags travel under module-private names so a predictions frame
#: that already carries same-named columns collides with neither (the same
#: defensive naming ``surface_switch_tilt_overlay.OVERLAY_FLAG_COLUMN`` adopted
#: after a 2026-08-24 KeyError rehearsal).
HOME_FLAG_COLUMN = "_tank_zone_tilt_home"
AWAY_FLAG_COLUMN = "_tank_zone_tilt_away"

_REQUIRED_SCHEDULE_COLUMNS = frozenset(
    {"game_id", "season", "week", "game_type", "home_team", "away_team", "result", "spread_line"}
)


def _canonical_team(team: pd.Series) -> pd.Series:
    return team.astype(str).map(lambda code: TEAM_ABBREVIATION_ALIASES.get(code, code))


def _season_tank_zone_by_week(season_games: pd.DataFrame) -> dict[int, frozenset[str]]:
    """``{week: the two worst records entering that week}`` for ONE season.

    Ported from ``scripts/motivation_ladder_screen.py``'s tally loop
    (``build_state_timeline``, lines 190-241) and its league ordering
    (``compute_day_states``, lines 164-166), with the snapshot granularity
    moved from gameday to strictly-prior-week (see the module docstring's
    standings-convention disclosure). A ``result`` of exactly 0 is a tie and
    increments neither wins nor losses, matching the screen's separate ties
    counter.
    """

    teams = sorted(set(season_games["home_team"]) | set(season_games["away_team"]))
    wins: dict[str, int] = dict.fromkeys(teams, 0)
    losses: dict[str, int] = dict.fromkeys(teams, 0)
    settled = season_games.loc[season_games["_standings_eligible"]]
    by_week: dict[int, frozenset[str]] = {}
    for week in sorted(int(value) for value in season_games["week"].unique()):
        ordered = sorted(teams, key=lambda team: (wins[team], -losses[team], team))
        by_week[week] = frozenset(ordered[:TANK_ZONE_SIZE])
        this_week = settled.loc[settled["week"].eq(week), ["home_team", "away_team", "result"]]
        for home, away, result in this_week.to_numpy():
            margin = float(result)
            if margin > 0:
                wins[str(home)] += 1
                losses[str(away)] += 1
            elif margin < 0:
                wins[str(away)] += 1
                losses[str(home)] += 1
    return by_week


def tank_zone_flag_by_game(schedules: pd.DataFrame) -> pd.DataFrame:
    """One row per REG ``game_id``: ``tank_zone_home`` / ``tank_zone_away``.

    A side is in the tank zone when its record places it in the league's
    BOTTOM TWO, ordered by wins ascending, then losses DESCENDING, then team
    abbreviation ascending for determinism -- the screen's own
    ``sorted(DIVISIONS, key=lambda t: (tallies[t][0], -tallies[t][1], t))[:2]``
    (``scripts/motivation_ladder_screen.py:164-165``), transcribed rather than
    re-derived. Ties in the ordering are broken exactly the way the screen
    breaks them: more losses ranks WORSE at equal wins, then alphabetical team
    code. Actual tie GAMES (``result == 0``) increment neither counter.

    **Pregame-safe by construction.** The standings entering week *W* of a
    season are built only from that season's completed games in weeks strictly
    less than *W*; this function never reads the flagged game's own ``result``
    for its own flag, and a later week's results can never change an earlier
    week's flags. Two leakage regression tests in
    ``tests/test_tank_zone_fade_tilt_overlay.py`` prove both properties
    empirically.

    Only games whose ATS outcome is settled (``home_cover`` not NaN, i.e. the
    screen's own ``load_schedules`` filter) contribute to the tallies -- so a
    not-yet-played game contributes nothing, which is what makes a live
    Tuesday-lock run behave identically to the historical replay.
    """

    missing = sorted(_REQUIRED_SCHEDULE_COLUMNS.difference(schedules.columns))
    if missing:
        raise DataContractError(
            f"schedules is missing columns for tank-zone tracking: {', '.join(missing)}"
        )

    reg = schedules.loc[schedules["game_type"].astype(str).eq("REG")].copy()
    reg["home_team"] = _canonical_team(reg["home_team"])
    reg["away_team"] = _canonical_team(reg["away_team"])
    reg["season"] = pd.to_numeric(reg["season"], errors="coerce").astype(int)
    reg["week"] = pd.to_numeric(reg["week"], errors="coerce").astype(int)
    reg["_standings_eligible"] = add_ats_outcomes(reg)["home_cover"].notna().to_numpy()

    if reg.empty:
        return pd.DataFrame(
            {
                "game_id": pd.Series([], dtype=object),
                "season": pd.Series([], dtype=int),
                "tank_zone_home": pd.Series([], dtype=bool),
                "tank_zone_away": pd.Series([], dtype=bool),
            }
        )

    frames: list[pd.DataFrame] = []
    for _season, season_games in reg.groupby("season", sort=True):
        by_week = _season_tank_zone_by_week(season_games)
        home_flags = [
            str(team) in by_week[int(week)]
            for team, week in season_games[["home_team", "week"]].to_numpy()
        ]
        away_flags = [
            str(team) in by_week[int(week)]
            for team, week in season_games[["away_team", "week"]].to_numpy()
        ]
        frames.append(
            pd.DataFrame(
                {
                    "game_id": season_games["game_id"].astype(str).to_numpy(),
                    "season": season_games["season"].to_numpy(),
                    "tank_zone_home": home_flags,
                    "tank_zone_away": away_flags,
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


@dataclass(frozen=True)
class TiltFlip:
    """One game the overlay flipped, for provenance and ledger recording."""

    game_id: str
    matchup: str
    tank_zone_team: str
    opponent_team: str


@dataclass(frozen=True)
class TiltResult:
    """The overlay's effect on one week's card.

    ``overlaid_predictions`` is ``predictions`` unchanged except for
    ``home_cover_probability`` on flipped rows -- every other column stays
    byte-identical, mirroring ``surface_switch_tilt_overlay.TiltResult``.
    ``both_tank_zone_games`` lists eligible games where BOTH sides carry the
    tank-zone flag; there is no measured direction for that case (mirroring
    ``coach_fade_overlay``'s ``both_year_one_games``), so those games are
    reported, never flipped.
    """

    overlaid_predictions: pd.DataFrame
    flips: tuple[TiltFlip, ...]
    both_tank_zone_games: tuple[str, ...]
    week_min: int
    week_max: int
    enabled: bool

    @property
    def flip_count(self) -> int:
        return len(self.flips)


def apply_tank_zone_fade_tilt_overlay(
    predictions: pd.DataFrame,
    schedules: pd.DataFrame,
    *,
    week_min: int = OVERLAY_WEEK_MIN,
    week_max: int = OVERLAY_WEEK_MAX,
    enabled: bool = True,
) -> TiltResult:
    """Fade the tank-zone team: flip the forced pick OFF it, weeks 14-18 only.

    A game flips only when ALL hold:

    * ``week_min <= week <= week_max`` (weeks 1-13 are ALWAYS left untouched --
      the registered cell's flag carries no claim there, and this is why a
      Week 1 card can never be moved by this overlay);
    * ``game_type == "REG"`` when that column is present (the registered
      measurement is a regular-season, close-graded read);
    * EXACTLY ONE side of the game carries the tank-zone flag (a both-flagged
      game has no measured direction and is reported in
      ``both_tank_zone_games`` instead of flipped); and
    * the model's own pick (``home_cover_probability >= 0.5`` picks home) IS
      that tank-zone side.

    Deliberately one-directional: the overlay never flips a pick TOWARD a
    tank-zone team, because the measured evidence is a fade -- flagged teams
    covered 44.4% against a 50.3% complement (see the module docstring's
    registry-description correction).

    Flipping sets ``home_cover_probability`` to its complement, exactly as the
    sibling overlays do, so every existing reader of the column needs no
    overlay-aware branch. Games with no schedule row, or with no flag after the
    merge, are the documented no-op -- zero flips, never a ``KeyError``.
    """

    required = {"game_id", "season", "week", "home_team", "away_team", "home_cover_probability"}
    missing = sorted(required.difference(predictions.columns))
    if missing:
        raise DataContractError(f"predictions is missing overlay columns: {', '.join(missing)}")

    base = predictions.reset_index(drop=True).copy()
    base["game_id"] = base["game_id"].astype(str)
    if not enabled:
        return TiltResult(base, (), (), week_min, week_max, enabled)

    flags = tank_zone_flag_by_game(schedules).rename(
        columns={"tank_zone_home": HOME_FLAG_COLUMN, "tank_zone_away": AWAY_FLAG_COLUMN}
    )
    merged = base.merge(
        flags,
        on=["game_id", "season"],
        how="left",
        validate="one_to_one",
    )
    for column in (HOME_FLAG_COLUMN, AWAY_FLAG_COLUMN):
        if column not in merged.columns:
            merged[column] = False
        merged[column] = merged[column].fillna(False).astype(bool)

    weeks = pd.to_numeric(merged["week"], errors="coerce")
    eligible = weeks.ge(week_min) & weeks.le(week_max)
    if "game_type" in merged.columns:
        eligible &= merged["game_type"].astype(str).eq("REG")

    home_pick = pd.to_numeric(merged["home_cover_probability"], errors="coerce").ge(0.5)
    both_tank = merged[HOME_FLAG_COLUMN] & merged[AWAY_FLAG_COLUMN]
    picked_is_tank = merged[HOME_FLAG_COLUMN].where(home_pick, merged[AWAY_FLAG_COLUMN])

    flip_mask = eligible & picked_is_tank & ~both_tank

    overlaid = base.copy()
    overlaid.loc[flip_mask, "home_cover_probability"] = (
        1.0 - overlaid.loc[flip_mask, "home_cover_probability"]
    )

    flips: list[TiltFlip] = []
    for _, row in merged.loc[flip_mask].iterrows():
        row_home_pick = bool(float(row["home_cover_probability"]) >= 0.5)
        tank_team = str(row["home_team"] if row_home_pick else row["away_team"])
        opponent = str(row["away_team"] if row_home_pick else row["home_team"])
        flips.append(
            TiltFlip(
                game_id=str(row["game_id"]),
                matchup=f"{row['away_team']} at {row['home_team']}",
                tank_zone_team=tank_team,
                opponent_team=opponent,
            )
        )

    both_ids = tuple(merged.loc[eligible & both_tank, "game_id"].astype(str))
    return TiltResult(overlaid, tuple(flips), both_ids, week_min, week_max, enabled)


def overlay_disclosure_note(result: TiltResult) -> str:
    """Plain-language provenance sentence, mirroring the sibling overlays'.

    Empty when the overlay is off or changed nothing this week. Not currently
    surfaced on the published card -- this overlay is dual-tracked only.
    """

    if not result.enabled or result.flip_count == 0:
        return ""
    plural = "" if result.flip_count == 1 else "s"
    detail = "; ".join(
        f"{flip.matchup}: {flip.tank_zone_team} -> {flip.opponent_team}" for flip in result.flips
    )
    return (
        f"**Tilt applied: {result.flip_count} pick{plural} flipped** by the tank-zone fade "
        f"(weeks {result.week_min}-{result.week_max}, clean case only: the model sided with a "
        "team holding one of the league's two worst records entering the week, against an "
        f"opponent that does not). {detail}. See docs/tank_zone_fade_tilt_overlay.md. "
        "Prospective evidence only -- not applied to the published card."
    )


def _record_instant(now: datetime | None) -> pd.Timestamp:
    instant = pd.Timestamp(now if now is not None else datetime.now(UTC))
    return instant.tz_localize("UTC") if instant.tzinfo is None else instant.tz_convert("UTC")


def record_tank_zone_fade_tilt_challenger_decisions(
    artifacts_root: Path,
    data_root: Path,
    *,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Append the tilt overlay's picks to the prospective challenger ledger.

    Mirrors ``surface_switch_tilt_overlay.record_surface_switch_tilt_challenger_decisions``
    exactly: this is not a retrained model with its own ``margin-predict``
    artifact -- its "model" IS the active model, transformed post-prediction --
    so it reads the active model's own synchronized weekly forecast rather than
    searching ``artifacts/margin_predictions/`` by fingerprint, and it refuses to
    record if the active model's live fingerprint no longer matches the snapshot
    this challenger was registered against.

    ``bet_side`` is always ``"PASS"`` and ``edge`` is always NaN: this challenger
    tracks the tilt's forced-pick (``decision_line``) accuracy only, never a
    fabricated paper-bet edge for the post-tilt side.
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
    tilt = apply_tank_zone_fade_tilt_overlay(card, schedules)
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
        "both_tank_zone_games": list(tilt.both_tank_zone_games),
    }
