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

CHALLENGER_ID = "inactives_refresh_v1"

INACTIVES_LEAD_MINUTES = 90

SOURCE_NO_SNAPSHOT = "tuesday_card (no in-window snapshot)"
SOURCE_STRUCTURALLY_EXCLUDED = "tuesday_card (SNF/MNF excluded)"
ET = ZoneInfo("America/New_York")
_SNAPSHOT_SCHEMA = "nflcom_inactives_snapshot/1"
_RECOGNIZED_SOURCES = frozenset({"primary", "fallback"})


def snapshot_source_tag(snapshot_id: str) -> str:

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


@dataclass(frozen=True)
class InactivesSnapshot:
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


@dataclass(frozen=True)
class PlayerContext:
    identities: dict[tuple[int, str, str], str]
    credited: dict[tuple[int, int, str, str], float]
    roles: dict[str, dict[str, float | str]]


def _share(value: Any) -> float:

    if value is None or pd.isna(value):
        return 0.0
    return float(value)


def _identity_lookup(rosters: pd.DataFrame) -> dict[tuple[int, str, str], str]:
    frame = rosters.loc[rosters["gsis_id"].notna(), ["season", "team", "full_name", "gsis_id"]]
    frame = frame.copy()
    frame["norm"] = frame["full_name"].map(_normalized_player_name)
    lookup: dict[tuple[int, str, str], str] = {}
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

    try:
        snapshot = latest_player_snapshot(data_root / "players" / "raw")
        injuries_raw, rosters_raw, snaps_raw = load_player_snapshot(snapshot)
        injuries = canonicalize_injuries(injuries_raw)
        rosters = canonicalize_rosters(rosters_raw)
        snaps = attach_snap_player_ids(canonicalize_snaps(snaps_raw), rosters)
    except Exception:
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


def _inactives_instant(kickoff: pd.Timestamp) -> pd.Timestamp:
    return pd.Timestamp(kickoff) - pd.Timedelta(minutes=INACTIVES_LEAD_MINUTES)


def structurally_excluded(kickoff: pd.Timestamp, deadline: pd.Timestamp) -> bool:

    return _inactives_instant(kickoff) >= pd.Timestamp(deadline)


def build_inactives_refresh_overlay_rows(
    plan: RefreshResult,
    *,
    artifacts_root: Path,
    data_root: Path,
    min_train_games: int = DEFAULT_MIN_TRAIN_GAMES,
) -> tuple[pd.DataFrame, dict[str, Any]]:

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
