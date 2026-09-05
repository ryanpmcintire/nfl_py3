"""Build the optional, ignored lineup artifact consumed by This Week.

This is deliberately a separate refresh step: GitHub Pages can only serve
static JSON, and the renderer must never reach out to a live roster or injury
provider.

UI-20 lineup probabilities (2026-09-05): every listed player with a
``gsis_id`` now carries a ``play_probability``, sourced from the SAME
learned-then-fixed availability machinery
(``nfl_ats.availability.resolve_unavailability``) that already produces
``{side}_qb_start_probability`` for the one QB the forecast consumes
(``probability_source: "base_model_qb"``, kept bit-identical). Every other
player either matches this week's own visible injury-report row
(``"availability_model"``, same learned/fixed rule, applied per-player
instead of only to the model's QB) or -- when no injury designation exists
for that player this week -- the position's historical no-designation base
rate (``nfl_ats.lineup_availability``, also ``"availability_model"``).
``"unavailable"`` is reserved for a depth-chart row this machinery genuinely
cannot score (no ``gsis_id``, or no rate available for that position).
Nothing is ever invented; see ``probability_reason`` on each player and
``docs/lineups.md``.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import nflreadpy as nfl
import pandas as pd

from nfl_ats.availability import availability_rate_lookup, canonicalize_availability_rates
from nfl_ats.lineup_availability import (
    RECENT_ROLE_UNKNOWN_NO_HISTORY,
    build_no_designation_outcomes,
    build_no_designation_rates,
    latest_recent_roles,
    no_designation_rate_lookup,
    resolve_play_probability,
)
from nfl_ats.lineup_view import STABLE_LINEUP_PATH
from nfl_ats.players import canonicalize_injuries, latest_player_snapshot, load_player_snapshot
from nfl_ats.provenance import stamp_sidecar
from nfl_ats.public_board import load_public_board_artifacts
from nfl_ats.quarterbacks import write_depth_snapshot

#: Where the active model's already-fitted learned availability rates live
#: (the SAME table ``nfl_ats.players.enrich_with_player_features`` reads to
#: compute ``{side}_qb_start_probability``). Read-only here; never rebuilt
#: or overwritten by this script.
WEAK_STACK_AVAILABILITY_RATES_PATH = (
    Path("data") / "processed" / "weak_stack_availability_rates.parquet"
)

#: Local player snapshots (injuries/weekly rosters/snap counts) live here;
#: the no-designation base rate is derived from whichever snapshot is
#: newest, entirely offline (no network fetch of its own).
PLAYER_SNAPSHOT_ROOT = Path("data") / "players" / "raw"


def _number(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if pd.notna(result) else None


def _learned_availability_lookup(
    path: Path = WEAK_STACK_AVAILABILITY_RATES_PATH,
) -> tuple[dict[tuple[int, str, str, str], float] | None, str]:
    """The active model's own learned availability rates, read-only.

    Falls back to ``None`` (the per-player resolver then uses the fixed,
    hand-authored prior -- the same fallback ``resolve_unavailability``
    already provides in production) when the table is absent; never
    rebuilds or writes it.
    """

    if not path.is_file():
        return None, f"no learned availability rate table at {path}; using the fixed prior"
    rates = canonicalize_availability_rates(pd.read_parquet(path))
    return availability_rate_lookup(
        rates
    ), f"{path} (max target_season {int(rates['target_season'].max())})"


def _no_designation_lookup(
    season: int, root: Path = PLAYER_SNAPSHOT_ROOT
) -> tuple[dict[tuple[int, str, str], float] | None, dict[str, str], str]:
    """The position's no-designation base rate plus each current player's
    ``recent_role``, derived from the newest local player snapshot -- no
    network fetch. See ``nfl_ats.lineup_availability`` for the derivation
    and why a bare position average would understate a starter's true
    probability.
    """

    try:
        snapshot = latest_player_snapshot(root)
    except FileNotFoundError as exc:
        return None, {}, f"no local player snapshot under {root}: {exc}"
    injuries, rosters, snaps = load_player_snapshot(snapshot, include_postseason=False)
    outcomes = build_no_designation_outcomes(injuries, rosters, snaps)
    rates = build_no_designation_rates(outcomes, target_seasons=[season])
    lookup = no_designation_rate_lookup(rates)
    roles = latest_recent_roles(rosters, snaps, before_season=season)
    provenance = (
        f"player snapshot {snapshot.snapshot_id} (seasons "
        f"{min(snapshot.roster_seasons)}-{max(snapshot.roster_seasons)})"
    )
    return lookup, roles, provenance


def _fetch_current_week_injuries(
    season: int, week: int, schedule: pd.DataFrame, generated_at: datetime
) -> tuple[pd.DataFrame, str]:
    """This week's live nflverse injury report, filtered to rows observed
    strictly before ``generated_at`` -- the same leakage-safe ``week_proxy``
    fallback ENG-39 built for the historical feature table
    (``nfl_ats.players.canonicalize_injuries``), since a current in-season
    release may omit ``date_modified`` entirely (docs/injury_timestamp_fallback.md).

    Returns an empty frame with a note, never an exception, when nflverse
    has not published this season yet (a live ``ValueError`` before kickoff
    week 1 -- measured this session) or has no rows for this week.
    """

    try:
        raw = nfl.load_injuries(seasons=[season]).to_pandas()
    except ValueError as exc:
        return pd.DataFrame(), f"nflverse has not published season {season} injuries yet ({exc})"
    raw = raw.loc[pd.to_numeric(raw["week"], errors="coerce") == week].copy()
    if raw.empty:
        return raw, f"nflverse has season {season} but no injury rows yet for week {week}"
    canonical = canonicalize_injuries(
        raw, include_postseason=False, timestamp_fallback="week_proxy", schedule=schedule
    )
    generated_ts = pd.Timestamp(generated_at)
    if generated_ts.tzinfo is None:
        generated_ts = generated_ts.tz_localize("UTC")
    visible = canonical.loc[canonical["effective_observed_at"] <= generated_ts].copy()
    return visible, (
        "live nflverse injury report, week_proxy fallback for a missing date_modified, "
        f"{len(visible)}/{len(canonical)} rows visible by {generated_ts.isoformat()}"
    )


def _visible_injuries_by_team(visible: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Latest visible revision per (team, gsis_id), keyed by team then
    indexed by ``gsis_id`` for O(1) per-player lookup."""

    if visible.empty:
        return {}
    deduped = visible.sort_values("effective_observed_at").drop_duplicates(
        ["team", "gsis_id"], keep="last"
    )
    return {
        str(team): group.set_index("gsis_id", drop=False)
        for team, group in deduped.groupby("team", sort=False)
    }


def _team_payload(
    depth: pd.DataFrame,
    team: str,
    model_qb_id: str | None,
    qb_probability: float | None,
    *,
    target_season: int,
    current_injuries: pd.DataFrame | None,
    learned_lookup: dict[tuple[int, str, str, str], float] | None,
    no_designation_lookup: dict[tuple[int, str, str], float] | None,
    recent_roles: dict[str, str],
) -> dict[str, Any]:
    rows = depth[depth["team"] == team].copy()
    # nflverse retains a row for each historical depth-chart update. Keep the
    # complete latest snapshot, including backups, rather than only starters.
    time_column = "observed_at_utc" if "observed_at_utc" in rows else "dt"
    rows["_dt"] = pd.to_datetime(rows[time_column], errors="coerce", utc=True)
    latest = rows["_dt"].max()
    if pd.notna(latest):
        rows = rows[rows["_dt"] == latest]
    rows = rows.drop_duplicates(subset=["pos_abb", "pos_rank", "player_name"], keep="last")
    unit_order = {"offense": 0, "defense": 1, "special_teams": 2}
    position_order = {
        "QB": 0,
        "RB": 1,
        "FB": 2,
        "WR": 3,
        "TE": 4,
        "LT": 5,
        "LG": 6,
        "C": 7,
        "RG": 8,
        "RT": 9,
        "LDE": 0,
        "LDT": 1,
        "NT": 2,
        "RDT": 3,
        "RDE": 4,
        "WLB": 5,
        "LILB": 6,
        "MLB": 7,
        "RILB": 8,
        "SLB": 9,
        "LCB": 10,
        "SS": 11,
        "FS": 12,
        "RCB": 13,
        "NB": 14,
        "PK": 0,
        "P": 1,
        "H": 2,
        "LS": 3,
        "PR": 4,
        "KR": 5,
    }

    def unit(position: str) -> str:
        if position in {"PK", "P", "H", "LS", "PR", "KR"}:
            return "special_teams"
        if position in {"QB", "RB", "FB", "WR", "TE", "LT", "LG", "C", "RG", "RT"}:
            return "offense"
        return "defense"

    rows["_unit"] = rows["pos_abb"].fillna("").map(lambda value: unit(str(value)))
    rows["_rank"] = pd.to_numeric(rows["pos_rank"], errors="coerce").fillna(99)
    rows["_position_order"] = rows["pos_abb"].map(position_order).fillna(99)
    rows = rows.sort_values(
        ["_unit", "_position_order", "_rank", "player_name"],
        key=lambda values: values.map(unit_order) if values.name == "_unit" else values,
        na_position="last",
    )
    team_injuries = current_injuries if current_injuries is not None else pd.DataFrame()
    players: list[dict[str, Any]] = []
    for _, row in rows.iterrows():
        position = str(row.get("pos_abb") or row.get("pos_name") or "")
        rank = int(row["_rank"]) if row["_rank"] < 99 else 1
        gsis_id = str(row["gsis_id"]) if pd.notna(row.get("gsis_id")) else None
        is_base_model_qb = position == "QB" and gsis_id == model_qb_id
        if is_base_model_qb:
            # Bit-identical to the pre-UI-20 behaviour: the model's own QB
            # probability comes from the forecast input, never recomputed
            # here.
            probability = qb_probability
            probability_source = "base_model_qb"
            probability_reason = (
                "forecast input (home/away_qb_start_probability); this is the QB the active "
                "model actually consumed"
                if qb_probability is not None
                else "forecast input unavailable for this QB (no home/away_qb_start_probability "
                "on the weekly forecast row)"
            )
        else:
            current_injury = None
            if gsis_id is not None and not team_injuries.empty and gsis_id in team_injuries.index:
                current_injury = team_injuries.loc[gsis_id]
                if isinstance(current_injury, pd.DataFrame):
                    # Defensive: set_index should already guarantee one row
                    # per gsis_id after _visible_injuries_by_team's own
                    # drop_duplicates, but never silently pick one of many.
                    current_injury = current_injury.iloc[-1]
            probability, probability_source, probability_reason = resolve_play_probability(
                gsis_id=gsis_id,
                position=position,
                target_season=target_season,
                current_injury=current_injury,
                learned_lookup=learned_lookup,
                no_designation_lookup=no_designation_lookup,
                recent_role=recent_roles.get(gsis_id, RECENT_ROLE_UNKNOWN_NO_HISTORY)
                if gsis_id is not None
                else RECENT_ROLE_UNKNOWN_NO_HISTORY,
            )
        players.append(
            {
                "name": str(row.get("player_name") or "Unknown player"),
                "position": position,
                "slot": f"{position}{rank}",
                "depth": rank,
                "unit": str(row["_unit"]),
                "gsis_id": gsis_id,
                "play_probability": probability,
                "probability_source": probability_source,
                "probability_reason": probability_reason,
                "model_role": "base_model" if gsis_id == model_qb_id else "context_only",
            }
        )
    current_qb = next((player for player in players if player["position"] == "QB"), None)
    note = None
    if model_qb_id and (current_qb is None or current_qb["gsis_id"] != model_qb_id):
        note = (
            "Current depth chart QB differs from forecast input; rerun forecast before treating "
            "this as a model update."
        )
    if team_injuries.empty:
        injury_status = (
            "no players listed on this week's injury report (or the report is not yet "
            "published); per-player probabilities use the position's no-designation base rate"
        )
    else:
        injury_status = (
            f"nflverse injury report attached ({len(team_injuries)} player(s) listed); "
            "per-player probabilities from the availability model"
        )
    return {
        "team": team,
        "players": players,
        "as_of": str(rows[time_column].iloc[0]) if not rows.empty else None,
        "source": "nflverse depth charts",
        "injury_status": injury_status,
        "note": note,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifacts-root", type=Path, default=Path("artifacts"))
    parser.add_argument("--season", type=int, default=2026)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    artifacts = load_public_board_artifacts(args.artifacts_root)
    season = int(artifacts.metadata.get("season", args.season))
    week = int(artifacts.metadata.get("week", 1))
    # Captured once, up front: both the point-in-time cutoff for this
    # week's live injury feed AND the payload's own `generated_at` stamp
    # must be the exact same instant (UI-20's leakage discipline -- a
    # revision observed after this instant must never move a player's
    # number).
    generated_at = datetime.now(UTC)
    display_depth = nfl.load_depth_charts(season).to_pandas()
    depth_snapshot = write_depth_snapshot(
        display_depth, Path("data") / "quarterbacks" / "depth" / "raw", [season]
    )
    schedule = (
        artifacts.predictions.loc[:, ["season", "week", "home_team", "away_team", "kickoff"]]
        .drop_duplicates()
        .reset_index(drop=True)
    )
    current_injuries, injury_feed_note = _fetch_current_week_injuries(
        season, week, schedule, generated_at
    )
    current_injuries_by_team = _visible_injuries_by_team(current_injuries)
    learned_lookup, learned_lookup_note = _learned_availability_lookup()
    no_designation_lookup, recent_roles, no_designation_note = _no_designation_lookup(season)
    games: dict[str, Any] = {}
    for _, row in artifacts.predictions.iterrows():
        game_id = str(row["game_id"])
        home_qb = (
            str(row["home_projected_qb_id"]) if pd.notna(row.get("home_projected_qb_id")) else None
        )
        away_qb = (
            str(row["away_projected_qb_id"]) if pd.notna(row.get("away_projected_qb_id")) else None
        )
        home_team = str(row["home_team"])
        away_team = str(row["away_team"])
        games[game_id] = {
            "home": _team_payload(
                display_depth,
                home_team,
                home_qb,
                _number(row.get("home_qb_start_probability")),
                target_season=season,
                current_injuries=current_injuries_by_team.get(home_team),
                learned_lookup=learned_lookup,
                no_designation_lookup=no_designation_lookup,
                recent_roles=recent_roles,
            ),
            "away": _team_payload(
                display_depth,
                away_team,
                away_qb,
                _number(row.get("away_qb_start_probability")),
                target_season=season,
                current_injuries=current_injuries_by_team.get(away_team),
                learned_lookup=learned_lookup,
                no_designation_lookup=no_designation_lookup,
                recent_roles=recent_roles,
            ),
        }
    stamp = generated_at.strftime("%Y%m%dT%H%M%SZ")
    # Replacement, not accumulation: the default target is one stable path that
    # every refresh overwrites. An explicit --output still writes exactly there
    # (and skips legacy cleanup, so ad-hoc exports never delete the live file).
    explicit_output = args.output is not None
    output = args.output or args.artifacts_root / STABLE_LINEUP_PATH
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = (
        json.dumps(
            {
                "season": season,
                "week": week,
                "generated_at": stamp,
                "model_id": artifacts.active.get("model_id"),
                "forecast_artifact": artifacts.active.get("weekly_forecast", {}).get("artifact"),
                "depth_snapshot": depth_snapshot.snapshot_id,
                "probability_provenance": {
                    "learned_availability_rate_table": learned_lookup_note,
                    "no_designation_base_rate": no_designation_note,
                    "current_injury_feed": injury_feed_note,
                },
                "games": games,
            },
            indent=2,
        )
        + "\n"
    )
    staging = output.with_name(f".{output.name}.{stamp}.tmp")
    staging.write_text(payload, encoding="utf-8")
    os.replace(staging, output)
    # ENG-38: stamp via a sidecar rather than rerouting the write through
    # write_stamped_artifact -- this file's own staging/os.replace atomicity
    # and immediately-following size check are production-sensitive (the
    # live public board reads STABLE_LINEUP_PATH); the sidecar adds
    # provenance without touching that path at all.
    stamp_sidecar(output)
    _check_artifact_size(output)
    if not explicit_output:
        _remove_legacy_stamped_runs(args.artifacts_root / "lineups", keep=output)
    print(output)


#: Fail-closed ceiling for the display artifact: 16 games of ~140 small
#: player dicts should stay well under a megabyte (measured 674 KB,
#: 2026-09-03); a 2026-09-03 stamped run once reached 37 MB and was deleted
#: unexamined under the replacement policy, so the builder now refuses to
#: publish a bloated artifact silently instead of hoping it was a one-off.
MAX_LINEUP_BYTES = 5 * 1024 * 1024


def _check_artifact_size(path: Path, *, limit: int = MAX_LINEUP_BYTES) -> None:
    size = path.stat().st_size
    if size > limit:
        raise SystemExit(
            f"Refusing to publish {path} at {size} bytes (limit {limit}): "
            "the lineup artifact should stay near a megabyte; inspect the "
            "payload before overriding this guard."
        )


def _remove_legacy_stamped_runs(lineups_root: Path, *, keep: Path) -> None:
    """Delete pre-replacement-policy stamped `*/lineups.json` runs.

    Each stamped run is a ~37 MB display copy superseded by the stable path;
    provenance (model, forecast, depth snapshot) lives inside the payload, and
    the underlying depth snapshots remain in `data/quarterbacks/depth/raw`.
    Only directories directly under the lineups root holding a `lineups.json`
    are touched; anything else is left alone.
    """
    if not lineups_root.is_dir():
        return
    for child in sorted(lineups_root.iterdir()):
        if not child.is_dir() or child.name == keep.parent.name:
            continue
        artifact = child / "lineups.json"
        if not artifact.is_file():
            continue
        shutil.rmtree(child, ignore_errors=True)


if __name__ == "__main__":
    main()
