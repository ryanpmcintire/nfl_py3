"""Static, source-aware projected lineup view models for the public board.

The board is a static GitHub Pages site, so this module deliberately reads a
pre-built JSON artifact.  It never calls a live provider while rendering and
it keeps freshness and model/data mismatches visible to the reader.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ProjectedPlayer:
    name: str
    position: str
    slot: str
    depth: int
    gsis_id: str | None = None
    play_probability: float | None = None
    injury_status: str | None = None
    model_role: str = "context_only"
    model_impact_points: float | None = None
    model_impact_note: str | None = None


@dataclass(frozen=True)
class TeamLineup:
    team: str
    players: tuple[ProjectedPlayer, ...]
    as_of: str | None
    source: str | None
    injury_status: str
    note: str | None = None

    def with_model_impact(
        self, *, family_points: float | None, model_qb_id: str | None
    ) -> TeamLineup:
        """Attach the team-level QB family contribution without inventing a
        player coefficient.  The waterfall is family-level, not player-level.
        """
        players = list(self.players)
        for index, player in enumerate(players):
            if player.position != "QB":
                continue
            role = (
                "base_model"
                if player.gsis_id and player.gsis_id == model_qb_id
                else player.model_role
            )
            note = None
            if family_points is not None and role == "base_model":
                direction = "supports" if family_points >= 0 else "works against"
                note = f"QB family {direction} the pick ({family_points:+.2f} pts)"
            players[index] = replace(
                player,
                model_role=role,
                model_impact_points=family_points if role == "base_model" else None,
                model_impact_note=note,
            )
        return replace(self, players=tuple(players))


def _player(raw: Mapping[str, Any]) -> ProjectedPlayer:
    probability = raw.get("play_probability")
    return ProjectedPlayer(
        name=str(raw.get("name") or "Unknown player"),
        position=str(raw.get("position") or ""),
        slot=str(raw.get("slot") or raw.get("position") or ""),
        depth=int(raw.get("depth") or 1),
        gsis_id=str(raw["gsis_id"]) if raw.get("gsis_id") else None,
        play_probability=float(probability) if probability is not None else None,
        injury_status=str(raw["injury_status"]) if raw.get("injury_status") else None,
        model_role=str(raw.get("model_role") or "context_only"),
    )


def team_lineup(raw: Mapping[str, Any]) -> TeamLineup:
    return TeamLineup(
        team=str(raw.get("team") or ""),
        players=tuple(
            _player(player) for player in raw.get("players", []) if isinstance(player, Mapping)
        ),
        as_of=str(raw["as_of"]) if raw.get("as_of") else None,
        source=str(raw["source"]) if raw.get("source") else None,
        injury_status=str(raw.get("injury_status") or "unavailable"),
        note=str(raw["note"]) if raw.get("note") else None,
    )


def load_lineups(artifacts_root: Path) -> dict[str, tuple[TeamLineup, TeamLineup]]:
    """Load the newest optional lineups artifact, failing open when absent."""
    root = artifacts_root / "lineups"
    candidates = sorted(root.glob("*/lineups.json"), reverse=True)
    if not candidates:
        return {}
    try:
        payload = json.loads(candidates[0].read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    # Older/manual presentation snapshots are not eligible to gate or feed a
    # forecast. Only artifacts explicitly tied to the active forecast enter
    # the public board path.
    if not payload.get("model_id") or not payload.get("forecast_artifact"):
        return {}
    result: dict[str, tuple[TeamLineup, TeamLineup]] = {}
    for game_id, raw in (payload.get("games") or {}).items():
        if not isinstance(raw, Mapping):
            continue
        home, away = raw.get("home"), raw.get("away")
        if isinstance(home, Mapping) and isinstance(away, Mapping):
            result[str(game_id)] = (team_lineup(home), team_lineup(away))
    return result


def validate_lineup_model_sync(
    lineups: Mapping[str, tuple[TeamLineup, TeamLineup]], predictions: Any
) -> None:
    """Fail closed rather than publish a model beside a different QB lineup."""
    if not lineups or not hasattr(predictions, "iterrows"):
        return
    missing: list[str] = []
    for _, row in predictions.iterrows():
        game_id = str(row.get("game_id"))
        teams = lineups.get(game_id)
        if teams is None:
            missing.append(game_id)
            continue
        for side, team in zip(("home", "away"), teams, strict=True):
            model_id = row.get(f"{side}_projected_qb_id")
            model_id = str(model_id) if model_id and str(model_id) != "nan" else None
            lineup_ids = {player.gsis_id for player in team.players if player.position == "QB"}
            if model_id is not None and model_id not in lineup_ids:
                missing.append(f"{game_id}:{side}")
    if missing:
        raise ValueError(
            "Lineup/model mismatch; refusing to render model suggestions until the forecast "
            f"is regenerated from the current lineup snapshot ({', '.join(missing[:8])})."
        )
