"""Official T-90 inactives as a refresh-time, CHALLENGER-ONLY overlay (WP41).

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
``docs/inactives_channel.md`` Section 1 found the official game-day inactive
list to be the last unclaimed slice of the injury timeline (Tue/Fri/Sat are
all already tested), Section 2 MEASURED 238/272 (87.5%) of 2026 REG games as
deadline-eligible for it, and WP17 built the capture
(``nfl_ats.inactives_capture``, snapshots under
``data/players/inactives/<UTC stamp>/``). What was missing is the wiring: no
code read a captured snapshot back into a pick. This module is that wiring,
and NOTHING else. Its rule is frozen in that document's
"Prospective wiring predeclaration (2026-09-01, WP41)" section, written
before this file existed.

Per game, at every ``nfl-ats refresh-picks`` pass:

* **SNF/MNF** -- the official inactives instant (``kickoff - 90 minutes``,
  the league convention reported in that document's Section 3 and used
  unchanged throughout its Section 2 arithmetic) falls at or after the game's
  own pick deadline, so the channel can never act. Keep the Tuesday pick,
  tagged :data:`SOURCE_STRUCTURALLY_EXCLUDED`.
* **No in-window snapshot** -- no snapshot for this (season, week) was
  captured strictly before this game's deadline AND actually reported rows.
  Keep the Tuesday pick, tagged :data:`SOURCE_NO_SNAPSHOT`. A zero-row
  snapshot (the off-season placeholder case, and any future
  ``empty_reason``) counts as "no report yet", never as "nobody is
  inactive" -- the same fail-open convention
  ``nfl_ats.nflcom_refresh_overlay`` already uses for an absent or stale
  NFL.com page. "Strictly before the deadline" is also the anti-backdating
  guard: the deadline is at most the kickoff, so a snapshot captured at or
  after kickoff can never apply.
* **In-window snapshot** -- recompute the injury construct with every listed
  player at **P(plays) = 0**, everyone else exactly as the injury report
  already has them, and re-run the production pick at the frozen Tuesday
  line. Tagged ``inactives_snapshot <stamp>``.

No constant is invented
-----------------------
``P(plays) = 0`` is unavailability ``1.0`` by definition, and that is
bit-identical to the weight production already assigns a player ruled Out
(``nfl_ats.availability.fixed_unavailability`` maps ``"out" -> 1.0``). The
overlay therefore applies the INCREMENT ``1.0 - fixed_unavailability(newest
visible report row)`` -- 0.0 for a player already Out (no double-count), 1.0
for a genuine surprise absence -- and folds it through
``nfl_ats.players._injury_features``, PRODUCTION'S OWN aggregation function,
imported rather than reimplemented, via the ``_unavailability`` hook that
function already reads. Its ``severity x share / 11`` and per-group ``/5``,
``/6``, ``/7`` normalizers are production's, not a copy. Role shares come
from the player's most recent strictly-earlier same-season snap-count row --
the same table and the same prior-game-share proxy
``nfl_ats.nflcom_refresh_overlay`` uses for its starter proxy; a player with
no prior snap row scores share 0 and contributes nothing, identical to
production's own ``roles.get(gsis_id, {})`` default. Every threshold
downstream is ``nfl_ats.pick_refresh``'s own, reused and never re-tuned
here (notably ``MOVEMENT_POLICY_THRESHOLD = 1.0``, frozen by
``docs/observed_movement_channel.md``'s predeclared grid).

Scope boundary, disclosed rather than discovered
------------------------------------------------
The active ``weak_stack`` profile consumes NINE injury columns: seven
``diff_injury_*_unavailability`` and two ``diff_injury_*_value_lost``. This
module adjusts **the seven unavailability columns only**. The two
value-lost columns need a per-player value rate drawn from a span-16 EWMA
state ``players.enrich_with_player_features`` builds transiently and never
persists, and rebuilding it here would be a reimplementation of production's
aggregation -- exactly what this design refuses to do. The candidate arm
therefore moves LESS than a full feature-table rebuild would, so a small or
null reading from it bounds the channel from below, not above.

Never touches the played pick
-----------------------------
The ``RefreshResult`` handed in is consumed strictly read-only. The
recomputed pick exists only as a column of a SEPARATE append-only ledger
(``artifacts/prospective/inactives_refresh_decisions.parquet``), never in
``pick_revisions.parquet`` and never on the published card. A prospective
registration is paper evidence at zero window cost, not a promotion.
"""

from __future__ import annotations

import json
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd

from nfl_ats.availability import fixed_unavailability
from nfl_ats.clv import refuse_if_outside_recording_lock_window
from nfl_ats.constants import DEFAULT_MIN_TRAIN_GAMES, TEAM_ABBREVIATION_ALIASES
from nfl_ats.data import DataContractError
from nfl_ats.io import atomic_parquet
from nfl_ats.pick_refresh import RefreshResult, original_card, plan_refresh
from nfl_ats.players import (
    PLAYER_INJURY_STATE_METRICS,
    _injury_features,
    _normalized_player_name,
    _position_group,
    attach_snap_player_ids,
    canonicalize_injuries,
    canonicalize_rosters,
    canonicalize_snaps,
    latest_player_snapshot,
    load_player_snapshot,
)

#: Registered in artifacts/prospective/challengers.json.
CHALLENGER_ID = "inactives_refresh_v1"

#: The league's published inactives lead, REPORTED in
#: ``docs/inactives_channel.md`` Section 3 (RotoWire's own FAQ: "NFL
#: inactives are released 90 minutes before kickoff for every game") and used
#: unchanged throughout that document's Section 2 deadline arithmetic, which
#: MEASURED SNF at -170 minutes of slack and MNF at -1,605 against their own
#: pick deadlines while every other slot stayed positive. Not a tuned
#: parameter and not re-derived here: it is the source's own convention, and
#: it is what makes the SNF/MNF exclusion structural rather than a filter.
INACTIVES_LEAD_MINUTES = 90

SOURCE_NO_SNAPSHOT = "tuesday_card (no in-window snapshot)"
SOURCE_STRUCTURALLY_EXCLUDED = "tuesday_card (SNF/MNF excluded)"
ET = ZoneInfo("America/New_York")
_SNAPSHOT_SCHEMA = "nflcom_inactives_snapshot/1"
_RECOGNIZED_SOURCES = frozenset({"primary", "fallback"})


def snapshot_source_tag(snapshot_id: str) -> str:
    """The per-game ``source`` tag for a game an in-window snapshot reached."""

    return f"inactives_snapshot {snapshot_id}"


INACTIVES_REFRESH_OVERLAY_COLUMNS: tuple[str, ...] = (
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
    "source",
    "inactives_snapshot_id",
    "inactives_captured_at_utc",
    "home_inactives_listed",
    "away_inactives_listed",
    "home_unavailability_increment",
    "away_unavailability_increment",
    "tuesday_pick_side",
    "played_pick_side",
    "inactives_pick_side",
    "inactives_flip_vs_tuesday",
    "inactives_flip_vs_played",
    "played_home_cover_probability",
    "inactives_home_cover_probability",
    "model_id",
    "feature_table_sha256",
)


def inactives_refresh_overlay_ledger_path(artifacts_root: Path) -> Path:
    return artifacts_root / "prospective" / "inactives_refresh_decisions.parquet"


def load_inactives_refresh_overlay_decisions(artifacts_root: Path) -> pd.DataFrame:
    """The append-only inactives refresh-overlay ledger (empty frame when none)."""

    path = inactives_refresh_overlay_ledger_path(artifacts_root)
    if not path.is_file():
        return pd.DataFrame(columns=list(INACTIVES_REFRESH_OVERLAY_COLUMNS))
    ledger = pd.read_parquet(path)
    missing = sorted(set(INACTIVES_REFRESH_OVERLAY_COLUMNS).difference(ledger.columns))
    if missing:
        raise DataContractError(
            f"Inactives refresh-overlay ledger is missing columns: {', '.join(missing)}"
        )
    return ledger[list(INACTIVES_REFRESH_OVERLAY_COLUMNS)]


# ---------------------------------------------------------------------------
# Reading captured inactives snapshots (WP17's writer, read back)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class InactivesSnapshot:
    """One ``nfl_ats.inactives_capture`` snapshot, as its manifest describes it."""

    snapshot_id: str
    root: Path
    captured_at_utc: pd.Timestamp
    season: int | None
    week: int | None
    row_count: int
    empty_reason: str | None
    source_used: str

    @property
    def parquet_path(self) -> Path:
        return self.root / "inactives.parquet"

    @property
    def reported_inactives(self) -> bool:
        """Did this snapshot actually carry an inactive list?

        A zero-row snapshot means "no report yet" (the off-season placeholder,
        an unreachable source, an unrecognized page), never "nobody is
        inactive" -- see the module docstring's fail-open note.
        """

        return (
            self.row_count > 0
            and self.empty_reason is None
            and self.source_used in _RECOGNIZED_SOURCES
        )


def _as_utc(value: Any) -> pd.Timestamp | None:
    try:
        stamp = pd.Timestamp(value) if value is not None else None
    except (TypeError, ValueError):
        return None
    if stamp is None or pd.isna(stamp):
        return None
    return stamp.tz_localize("UTC") if stamp.tzinfo is None else stamp.tz_convert("UTC")


def load_inactives_snapshots(data_root: Path) -> tuple[InactivesSnapshot, ...]:
    """Every readable inactives snapshot, oldest capture instant first.

    FAIL-OPEN: a snapshot whose manifest is missing, malformed, or carries no
    usable ``captured_at_utc`` is skipped, not raised on -- one bad directory
    must never take down a refresh pass.
    """

    root = data_root / "players" / "inactives"
    found: list[InactivesSnapshot] = []
    for manifest_path in sorted(root.glob("*/manifest.json")):
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if not isinstance(manifest, dict):
            continue
        captured = _as_utc(manifest.get("captured_at_utc"))
        if captured is None:
            continue
        if manifest.get("schema") != _SNAPSHOT_SCHEMA or manifest.get("ok") is not True:
            continue
        try:
            season = manifest.get("season")
            week = manifest.get("week")
            found.append(
                InactivesSnapshot(
                    snapshot_id=str(manifest.get("snapshot_id") or manifest_path.parent.name),
                    root=manifest_path.parent,
                    captured_at_utc=captured,
                    season=None if season is None else int(season),
                    week=None if week is None else int(week),
                    row_count=int(manifest.get("row_count") or 0),
                    empty_reason=(
                        None
                        if manifest.get("empty_reason") is None
                        else str(manifest["empty_reason"])
                    ),
                    source_used=str(manifest.get("source_used") or "none"),
                )
            )
        except (TypeError, ValueError):
            continue
    return tuple(sorted(found, key=lambda snapshot: snapshot.captured_at_utc))


def newest_snapshot_before(
    snapshots: tuple[InactivesSnapshot, ...],
    deadline: pd.Timestamp,
    *,
    season: int,
    week: int,
    now: pd.Timestamp | None = None,
    game_day: pd.Timestamp | None = None,
) -> InactivesSnapshot | None:
    """The newest snapshot for this week captured STRICTLY before ``deadline``.

    Strictness is the anti-backdating guard: a game's deadline is at most its
    own kickoff (``pick_refresh.pick_deadline``), so a snapshot captured at or
    after kickoff can never be returned here for that game.
    """

    decision_instant = _as_utc(now) if now is not None else None
    game_instant = _as_utc(game_day) if game_day is not None else None
    expected_game_day = game_instant.tz_convert(ET).date() if game_instant is not None else None
    usable = [
        snapshot
        for snapshot in snapshots
        if snapshot.reported_inactives
        and snapshot.season == season
        and snapshot.week == week
        and snapshot.captured_at_utc < deadline
        and (decision_instant is None or snapshot.captured_at_utc <= decision_instant)
        and (
            expected_game_day is None
            or snapshot.captured_at_utc.tz_convert(ET).date() == expected_game_day
        )
    ]
    return usable[-1] if usable else None


def read_inactives_rows(snapshot: InactivesSnapshot) -> pd.DataFrame:
    """One snapshot's parquet, team codes canonicalized. Empty frame on failure."""

    try:
        rows = pd.read_parquet(snapshot.parquet_path)
    except (OSError, ValueError):
        return pd.DataFrame(columns=["team", "player_name", "position"])
    if rows.empty:
        return rows
    rows = rows.copy()
    rows["team"] = (
        rows["team"].astype(str).map(lambda code: TEAM_ABBREVIATION_ALIASES.get(code, code))
    )
    return rows


def inactives_rows_for_game(
    snapshot: InactivesSnapshot,
    *,
    season: int,
    week: int,
    game_id: str,
    home_team: str,
    away_team: str,
) -> pd.DataFrame:
    """Return one safely aligned game's rows, otherwise an empty frame.

    Captures are slate-wide, but an inactive list can only affect the game it
    explicitly names. A malformed, partial, or schedule-misaligned snapshot
    is therefore indistinguishable from no report for this challenger.
    """

    rows = read_inactives_rows(snapshot)
    required = {
        "captured_at_utc",
        "season",
        "week",
        "game_id",
        "home_team",
        "away_team",
        "team",
        "player_name",
        "position",
    }
    if rows.empty or not required.issubset(rows.columns):
        return pd.DataFrame(columns=list(required))
    rows = rows.copy()
    for column in ("home_team", "away_team", "team"):
        rows[column] = (
            rows[column].astype(str).map(lambda code: TEAM_ABBREVIATION_ALIASES.get(code, code))
        )
    game_rows = rows.loc[rows["game_id"].astype(str).eq(str(game_id))].copy()
    if game_rows.empty:
        return pd.DataFrame(columns=list(required))
    if not (
        game_rows["season"].eq(season).all()
        and game_rows["week"].eq(week).all()
        and game_rows["home_team"].eq(home_team).all()
        and game_rows["away_team"].eq(away_team).all()
        and game_rows["team"].isin({home_team, away_team}).all()
    ):
        return pd.DataFrame(columns=list(required))
    captured = pd.to_datetime(game_rows["captured_at_utc"], utc=True, errors="coerce")
    if captured.isna().any() or not captured.eq(snapshot.captured_at_utc).all():
        return pd.DataFrame(columns=list(required))
    return game_rows


# ---------------------------------------------------------------------------
# The P(plays) = 0 override, folded through production's own aggregation
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PlayerContext:
    """Everything needed to turn one inactives list into injury increments."""

    #: (season, team, normalized player name) -> gsis_id, plus a globally
    #: unique-name fallback under key ``(season, "", name)``.
    identities: dict[tuple[int, str, str], str]
    #: gsis_id -> the newest injury-report unavailability already credited.
    credited: dict[tuple[int, int, str, str], float]
    #: gsis_id -> production-shaped role dict for _injury_features.
    roles: dict[str, dict[str, float | str]]


def _share(value: Any) -> float:
    """A snap share as a float, with a missing/unparseable share meaning 0.0.

    Production's own default for a player it has no role state for is an empty
    role dict, which ``_injury_features`` reads as ``0.0`` for every share
    (``float(role.get("offense_pct", 0.0))``); this keeps a NaN snap
    percentage on the same side of that convention instead of poisoning the
    whole team's total with NaN.
    """

    if value is None or pd.isna(value):
        return 0.0
    return float(value)


def _identity_lookup(rosters: pd.DataFrame) -> dict[tuple[int, str, str], str]:
    frame = rosters.loc[rosters["gsis_id"].notna(), ["season", "team", "full_name", "gsis_id"]]
    frame = frame.copy()
    frame["norm"] = frame["full_name"].map(_normalized_player_name)
    lookup: dict[tuple[int, str, str], str] = {}
    # Season-wide unique-name fallback, mirroring attach_snap_player_ids'
    # own "accepted only when that normalized name refers to exactly one
    # GSIS identity" rule rather than inventing a looser match.
    ambiguous: set[tuple[int, str]] = set()
    unique: dict[tuple[int, str], str] = {}
    for record in frame.to_dict("records"):
        season = int(record["season"])
        norm = str(record["norm"])
        gsis = str(record["gsis_id"])
        lookup[(season, str(record["team"]), norm)] = gsis
        seen = unique.get((season, norm))
        if seen is None:
            unique[(season, norm)] = gsis
        elif seen != gsis:
            ambiguous.add((season, norm))
    for (season, norm), gsis in unique.items():
        if (season, norm) not in ambiguous:
            lookup.setdefault((season, "", norm), gsis)
    return lookup


def _credited_unavailability(
    injuries: pd.DataFrame, *, cutoff: pd.Timestamp
) -> dict[tuple[int, int, str, str], float]:
    """The unavailability the injury report ALREADY credits, per player-week.

    Keyed ``(season, week, team, gsis_id)``, taken from the newest report row
    visible at ``cutoff`` (the snapshot's own capture instant -- rows filed
    after it were not knowable when the inactive list posted). Uses
    production's ``fixed_unavailability`` verbatim, so a player already ruled
    Out credits 1.0 and the override's increment for them is exactly zero.
    """

    if injuries.empty:
        return {}
    visible = injuries.loc[injuries["date_modified"].le(cutoff)]
    if visible.empty:
        return {}
    latest = visible.sort_values("date_modified").drop_duplicates(
        ["season", "week", "team", "gsis_id"], keep="last"
    )
    credited: dict[tuple[int, int, str, str], float] = {}
    for record in latest.to_dict("records"):
        key = (
            int(record["season"]),
            int(record["week"]),
            str(record["team"]),
            str(record["gsis_id"]),
        )
        credited[key] = float(
            fixed_unavailability(record["report_status"], record["practice_status"])
        )
    return credited


def _prior_snap_roles(
    snaps: pd.DataFrame, *, season: int, week: int
) -> dict[str, dict[str, float | str]]:
    """gsis_id -> role shares from that player's most recent EARLIER game.

    Same-season, strictly-earlier week, latest row wins: the identical
    prior-game snap-share proxy ``nfl_ats.nflcom_refresh_overlay`` already
    uses for its starter proxy. Week 1 has no prior game, so every share is
    absent and the override contributes nothing that week -- the same
    documented Week-1 behaviour that overlay's frozen rule text carries.
    """

    scoped = snaps.loc[snaps["season"].eq(season) & snaps["week"].lt(week)]
    scoped = scoped.loc[scoped["gsis_id"].notna()]
    if scoped.empty:
        return {}
    latest = scoped.sort_values("week").drop_duplicates("gsis_id", keep="last")
    roles: dict[str, dict[str, float | str]] = {}
    for record in latest.to_dict("records"):
        roles[str(record["gsis_id"])] = {
            "offense_pct": _share(record["offense_pct"]),
            "defense_pct": _share(record["defense_pct"]),
            "st_pct": _share(record["st_pct"]),
            "position_group": _position_group(record["position"]),
        }
    return roles


def load_player_context(
    data_root: Path, *, season: int, week: int, cutoff: pd.Timestamp
) -> PlayerContext | None:
    """Assemble the identity/credit/role inputs, or ``None`` on ANY failure.

    FAIL-OPEN by design, mirroring
    ``injury_signal_refresh_tilt._latest_official_injuries_fail_open``: a
    missing or malformed player snapshot means the override contributes
    nothing this pass, never an exception into ``refresh-picks``.
    """

    try:
        snapshot = latest_player_snapshot(data_root / "players" / "raw")
        injuries_raw, rosters_raw, snaps_raw = load_player_snapshot(snapshot)
        injuries = canonicalize_injuries(injuries_raw)
        rosters = canonicalize_rosters(rosters_raw)
        snaps = attach_snap_player_ids(canonicalize_snaps(snaps_raw), rosters)
    except Exception:  # deliberate fail-open, see docstring
        return None
    return PlayerContext(
        identities=_identity_lookup(rosters),
        credited=_credited_unavailability(injuries, cutoff=cutoff),
        roles=_prior_snap_roles(snaps, season=season, week=week),
    )


def team_unavailability_increments(
    inactives: pd.DataFrame,
    context: PlayerContext,
    *,
    season: int,
    week: int,
    team: str,
) -> tuple[dict[str, float], int]:
    """One team's inactives folded through production's own aggregation.

    Returns ``(increments, listed_count)`` where ``increments`` maps each of
    ``players.PLAYER_INJURY_STATE_METRICS`` to the amount this team's listed
    inactives ADD on top of what the injury report already credited. Computed
    by calling ``players._injury_features`` -- the exact function
    ``enrich_with_player_features`` calls -- so the normalizers are
    production's, never a copy.
    """

    zero = dict.fromkeys(PLAYER_INJURY_STATE_METRICS, 0.0)
    if inactives.empty:
        return zero, 0
    listed = inactives.loc[inactives["team"].astype(str).eq(team)]
    if listed.empty:
        return zero, 0

    rows: list[dict[str, Any]] = []
    roles: dict[str, dict[str, float | str]] = {}
    for record in listed.to_dict("records"):
        norm = _normalized_player_name(record["player_name"])
        gsis = context.identities.get((season, team, norm)) or context.identities.get(
            (season, "", norm)
        )
        key = gsis if gsis is not None else f"name:{team}:{norm}"
        credited = (
            context.credited.get((season, week, team, gsis), 0.0) if gsis is not None else 0.0
        )
        increment = 1.0 - credited
        if increment <= 0.0:
            continue
        role = context.roles.get(key)
        if role is None:
            # No prior snap row: production's own roles.get(id, {}) default is
            # an empty role, i.e. every share 0.0. Kept explicit so the
            # position still reaches _injury_features' group fallback.
            role = {"position_group": _position_group(record["position"])}
        roles[key] = role
        rows.append({"gsis_id": key, "_unavailability": increment, "position": record["position"]})

    if not rows:
        return zero, len(listed)
    frame = pd.DataFrame(rows, columns=["gsis_id", "_unavailability", "position"])
    return _injury_features(frame, roles), len(listed)


def apply_inactives_increments(
    features: pd.DataFrame, increments: dict[str, dict[str, dict[str, float]]]
) -> pd.DataFrame:
    """A COPY of ``features`` with only the named games' injury columns moved.

    ``increments`` maps ``game_id -> {"home": {...}, "away": {...}}``. Every
    other row and every other column is byte-identical, which is what keeps
    ``fit_margin_models_for_week``'s training set -- and therefore the fitted
    model -- unchanged when the candidate plan is computed on this table.
    ``diff_`` is recomputed as ``home - away``, ``features.py``'s convention.
    A NaN column (a game production could not build an injury state for) stays
    NaN: adding to it changes nothing, which is the correct no-op.
    """

    adjusted = features.copy()
    if not increments:
        return adjusted
    game_ids = adjusted["game_id"].astype(str)
    for game_id, sides in increments.items():
        positions = adjusted.index[game_ids.eq(str(game_id))]
        if len(positions) == 0:
            continue
        for metric in PLAYER_INJURY_STATE_METRICS:
            home_column = f"home_{metric}"
            away_column = f"away_{metric}"
            diff_column = f"diff_{metric}"
            if home_column not in adjusted.columns or away_column not in adjusted.columns:
                continue
            adjusted.loc[positions, home_column] = adjusted.loc[positions, home_column] + float(
                sides["home"].get(metric, 0.0)
            )
            adjusted.loc[positions, away_column] = adjusted.loc[positions, away_column] + float(
                sides["away"].get(metric, 0.0)
            )
            if diff_column in adjusted.columns:
                adjusted.loc[positions, diff_column] = (
                    adjusted.loc[positions, home_column] - adjusted.loc[positions, away_column]
                )
    return adjusted


# ---------------------------------------------------------------------------
# The overlay pass
# ---------------------------------------------------------------------------


def _inactives_instant(kickoff: pd.Timestamp) -> pd.Timestamp:
    return pd.Timestamp(kickoff) - pd.Timedelta(minutes=INACTIVES_LEAD_MINUTES)


def structurally_excluded(kickoff: pd.Timestamp, deadline: pd.Timestamp) -> bool:
    """SNF/MNF: the inactives instant lands at or after the pick deadline.

    NOT ``deadline < kickoff`` -- that naive test would also exclude the
    Sunday 16:05-17:00 ET slot, which ``docs/inactives_channel.md`` Section 2
    MEASURED as playable at +65 to +85 minutes of slack.
    """

    return _inactives_instant(kickoff) >= pd.Timestamp(deadline)


def build_inactives_refresh_overlay_rows(
    plan: RefreshResult,
    *,
    artifacts_root: Path,
    data_root: Path,
    min_train_games: int = DEFAULT_MIN_TRAIN_GAMES,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Pure computation: one row per ELIGIBLE game in ``plan``, three arms each.

    Every eligible game gets a row whether or not an inactives snapshot
    reached it -- the paired comparator is the whole Tuesday card, so games
    the channel could not touch are part of the population, not missing from
    it. FAIL-OPEN throughout: an absent snapshot store, an unreadable player
    snapshot, or an all-zero increment leaves the candidate arm equal to the
    played pick and says so in the diagnostics, never raises and never flips.
    Never writes anything -- see :func:`record_inactives_refresh_overlay`.
    """

    empty = pd.DataFrame(columns=list(INACTIVES_REFRESH_OVERLAY_COLUMNS))
    eligible = [game for game in plan.games if game.eligible]
    if not eligible:
        return empty, {"skipped": True, "reason": "no eligible games in this refresh pass"}

    original = original_card(artifacts_root, season=plan.season, week=plan.week)
    tuesday_side = (
        original.set_index("game_id")["pick_side"].astype(str).to_dict()
        if not original.empty
        else {}
    )

    snapshots = load_inactives_snapshots(data_root)
    sources: dict[str, str] = {}
    matched: dict[str, InactivesSnapshot] = {}
    for game in eligible:
        game_id = str(game.game_id)
        if structurally_excluded(game.kickoff, game.deadline):
            sources[game_id] = SOURCE_STRUCTURALLY_EXCLUDED
            continue
        snapshot = newest_snapshot_before(
            snapshots,
            pd.Timestamp(game.deadline),
            season=plan.season,
            week=plan.week,
            now=plan.computed_at_utc,
            game_day=pd.Timestamp(game.kickoff),
        )
        if snapshot is None:
            sources[game_id] = SOURCE_NO_SNAPSHOT
            continue
        aligned = inactives_rows_for_game(
            snapshot,
            season=plan.season,
            week=plan.week,
            game_id=game_id,
            home_team=TEAM_ABBREVIATION_ALIASES.get(str(game.home_team), str(game.home_team)),
            away_team=TEAM_ABBREVIATION_ALIASES.get(str(game.away_team), str(game.away_team)),
        )
        if aligned.empty:
            sources[game_id] = SOURCE_NO_SNAPSHOT
            continue
        sources[game_id] = snapshot_source_tag(snapshot.snapshot_id)
        matched[game_id] = snapshot

    increments: dict[str, dict[str, dict[str, float]]] = {}
    listed: dict[str, tuple[int, int]] = {}
    no_adjustment_reason = ""
    if matched:
        cutoff = max(snapshot.captured_at_utc for snapshot in matched.values())
        context = load_player_context(data_root, season=plan.season, week=plan.week, cutoff=cutoff)
        if context is None:
            no_adjustment_reason = "no_player_snapshot"
        else:
            by_game = {str(game.game_id): game for game in eligible}
            for game_id, snapshot in matched.items():
                game = by_game[game_id]
                rows = inactives_rows_for_game(
                    snapshot,
                    season=plan.season,
                    week=plan.week,
                    game_id=game_id,
                    home_team=TEAM_ABBREVIATION_ALIASES.get(
                        str(game.home_team), str(game.home_team)
                    ),
                    away_team=TEAM_ABBREVIATION_ALIASES.get(
                        str(game.away_team), str(game.away_team)
                    ),
                )
                if rows.empty:
                    continue
                home_increment, home_listed = team_unavailability_increments(
                    rows,
                    context,
                    season=plan.season,
                    week=plan.week,
                    team=TEAM_ABBREVIATION_ALIASES.get(str(game.home_team), str(game.home_team)),
                )
                away_increment, away_listed = team_unavailability_increments(
                    rows,
                    context,
                    season=plan.season,
                    week=plan.week,
                    team=TEAM_ABBREVIATION_ALIASES.get(str(game.away_team), str(game.away_team)),
                )
                listed[game_id] = (home_listed, away_listed)
                if any(home_increment.values()) or any(away_increment.values()):
                    increments[game_id] = {"home": home_increment, "away": away_increment}
            if not increments:
                no_adjustment_reason = "no_nonzero_increment"

    candidate_sides: dict[str, str] = {}
    candidate_probabilities: dict[str, float] = {}
    candidate_feature_sha = ""
    if increments:
        features = pd.read_parquet(plan.feature_table_path)
        adjusted = apply_inactives_increments(features, increments)
        with tempfile.TemporaryDirectory(prefix="inactives_refresh_") as work:
            adjusted_path = Path(work) / "features_inactives_adjusted.parquet"
            atomic_parquet(adjusted, adjusted_path)
            candidate = plan_refresh(
                artifacts_root,
                data_root,
                season=plan.season,
                week=plan.week,
                features_path=adjusted_path,
                min_train_games=min_train_games,
                now=plan.computed_at_utc.to_pydatetime(),
            )
        candidate_feature_sha = candidate.feature_table_sha256
        for game in candidate.games:
            candidate_sides[str(game.game_id)] = game.new_pick_side
            candidate_probabilities[str(game.game_id)] = game.new_home_cover_probability

    rows_out: list[dict[str, Any]] = []
    for game in eligible:
        game_id = str(game.game_id)
        snapshot = matched.get(game_id)
        home_listed, away_listed = listed.get(game_id, (0, 0))
        side = candidate_sides.get(game_id, game.new_pick_side)
        tuesday = tuesday_side.get(game_id, game.previous_pick_side)
        rows_out.append(
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
                "source": sources[game_id],
                "inactives_snapshot_id": "" if snapshot is None else snapshot.snapshot_id,
                "inactives_captured_at_utc": (
                    pd.NaT if snapshot is None else snapshot.captured_at_utc
                ),
                "home_inactives_listed": home_listed,
                "away_inactives_listed": away_listed,
                "home_unavailability_increment": float(
                    sum(increments.get(game_id, {}).get("home", {}).values())
                ),
                "away_unavailability_increment": float(
                    sum(increments.get(game_id, {}).get("away", {}).values())
                ),
                "tuesday_pick_side": tuesday,
                "played_pick_side": game.new_pick_side,
                "inactives_pick_side": side,
                "inactives_flip_vs_tuesday": side != tuesday,
                "inactives_flip_vs_played": side != game.new_pick_side,
                "played_home_cover_probability": game.new_home_cover_probability,
                "inactives_home_cover_probability": candidate_probabilities.get(
                    game_id, game.new_home_cover_probability
                ),
                "model_id": plan.model_id,
                "feature_table_sha256": plan.feature_table_sha256,
            }
        )

    frame = pd.DataFrame(rows_out, columns=list(INACTIVES_REFRESH_OVERLAY_COLUMNS))
    diagnostics: dict[str, Any] = {
        "skipped": False,
        "games_considered": len(frame),
        "snapshots_available": len(snapshots),
        "source_counts": frame["source"].value_counts().to_dict(),
        "games_with_in_window_snapshot": sorted(matched),
        "games_adjusted": sorted(increments),
        "adjusted_feature_table_sha256": candidate_feature_sha,
        "would_flip_vs_played_game_ids": frame.loc[frame["inactives_flip_vs_played"], "game_id"]
        .astype(str)
        .tolist(),
        "would_flip_vs_tuesday_game_ids": frame.loc[frame["inactives_flip_vs_tuesday"], "game_id"]
        .astype(str)
        .tolist(),
    }
    if no_adjustment_reason:
        diagnostics["no_adjustment_reason"] = no_adjustment_reason
    return frame, diagnostics


def record_inactives_refresh_overlay(
    artifacts_root: Path,
    data_root: Path,
    plan: RefreshResult,
    *,
    record_decisions: bool = False,
    min_train_games: int = DEFAULT_MIN_TRAIN_GAMES,
) -> dict[str, Any]:
    """Append this pass's inactives-refreshed picks to the overlay ledger.

    Mirrors ``nflcom_refresh_overlay.record_nflcom_refresh_overlay`` exactly:
    the same opt-in ``record_decisions`` contract, the same
    ``refuse_if_outside_recording_lock_window`` guard against the week's
    ORIGINAL card kickoffs, and the same guarantee that the played pipeline
    cannot see this function's output -- it writes only its own separate
    ledger and consumes the ``RefreshResult`` strictly read-only.

    Repeated passes across a week legitimately append MULTIPLE rows per game
    (not deduped): a Thursday pass sees no Sunday inactives, and how the
    channel's reach grows across a week is part of what prospective scoring
    reads. Scoring consumes the LATEST pre-kickoff row per game, mirroring
    ``pick_refresh.final_pick_per_game``.
    """

    if not record_decisions:
        return {
            "challenger_id": CHALLENGER_ID,
            "recorded": 0,
            "skipped": True,
            "reason": (
                "pass --record-decisions to append this pass's inactives-refreshed picks "
                "to the inactives refresh-overlay ledger"
            ),
        }

    original = original_card(artifacts_root, season=plan.season, week=plan.week)
    refuse_if_outside_recording_lock_window(
        original["kickoff"], plan.computed_at_utc, ledger="inactives-refresh-overlay"
    )

    rows, diagnostics = build_inactives_refresh_overlay_rows(
        plan,
        artifacts_root=artifacts_root,
        data_root=data_root,
        min_train_games=min_train_games,
    )
    existing = load_inactives_refresh_overlay_decisions(artifacts_root)
    if rows.empty:
        return {
            "challenger_id": CHALLENGER_ID,
            "recorded": 0,
            "ledger_rows": len(existing),
            **diagnostics,
        }

    combined = pd.concat([existing, rows], ignore_index=True) if not existing.empty else rows
    atomic_parquet(
        combined[list(INACTIVES_REFRESH_OVERLAY_COLUMNS)],
        inactives_refresh_overlay_ledger_path(artifacts_root),
    )
    return {
        "challenger_id": CHALLENGER_ID,
        "recorded": len(rows),
        "ledger_rows": len(combined),
        **diagnostics,
    }
