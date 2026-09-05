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
    unit: str = "offense"
    gsis_id: str | None = None
    play_probability: float | None = None
    #: UI-20-AB (2026-09-05): the availability model's own P(starts) for
    #: this player -- populated for every scored player, but only ever
    #: RENDERED for the QB slot (a second, smaller "start" number next to
    #: the main "plays" percentage; see ``board_terminal._lineup_team_html``).
    #: Distinct from ``model_qb_start_probability`` below, which is the
    #: active margin model's own forecast input for the one QB it consumed.
    start_probability: float | None = None
    injury_status: str | None = None
    model_role: str = "context_only"
    model_impact_points: float | None = None
    model_impact_note: str | None = None
    #: UI-20-AB: how ``play_probability`` was produced -- ``"play_probability_model"``
    #: (``nfl_ats.play_probability``'s walk-forward, calibrated model, for
    #: every player with a ``gsis_id``) or ``"unavailable"`` (no ``gsis_id``,
    #: or no predictor was available this run -- ``play_probability`` stays
    #: ``None``). See ``probability_reason`` for the human-readable "why".
    probability_source: str | None = None
    probability_reason: str | None = None
    #: Whether THIS player carries a visible injury-report row this week
    #: (observed strictly before the artifact's own ``generated_at``) --
    #: purely informational now (rendered next to the player's name/injury
    #: status), not a gate on whether the percentage is shown. UI-20 legibility
    #: fix (2026-09-05) briefly used this to hide the percentage for
    #: undesignated players; that stopgap is retired now that
    #: ``play_probability`` is a real per-player forecast for everyone
    #: (UI-20-AB, ``docs/play_probability_model.md``), not a position-level
    #: base rate -- see ``docs/projected_lineups.md``.
    has_injury_designation: bool = False


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
    start_probability = raw.get("start_probability")
    return ProjectedPlayer(
        name=str(raw.get("name") or "Unknown player"),
        position=str(raw.get("position") or ""),
        slot=str(raw.get("slot") or raw.get("position") or ""),
        depth=int(raw.get("depth") or 1),
        unit=str(raw.get("unit") or "offense"),
        gsis_id=str(raw["gsis_id"]) if raw.get("gsis_id") else None,
        play_probability=float(probability) if probability is not None else None,
        start_probability=float(start_probability) if start_probability is not None else None,
        injury_status=str(raw["injury_status"]) if raw.get("injury_status") else None,
        model_role=str(raw.get("model_role") or "context_only"),
        probability_source=str(raw["probability_source"])
        if raw.get("probability_source")
        else None,
        probability_reason=str(raw["probability_reason"])
        if raw.get("probability_reason")
        else None,
        has_injury_designation=bool(raw.get("has_injury_designation", False)),
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


#: The lineup artifact is REPLACED on every refresh, not accumulated: the
#: builder always overwrites this stable path, so at most one lineup snapshot
#: is ever on disk. History lives in the depth-chart snapshots the payload
#: cites (`depth_snapshot`), not in stamped display copies.
STABLE_LINEUP_PATH = Path("lineups") / "current" / "lineups.json"


def load_lineups(artifacts_root: Path) -> dict[str, tuple[TeamLineup, TeamLineup]]:
    """Load the newest optional lineups artifact, failing open when absent."""
    root = artifacts_root / "lineups"
    stable = artifacts_root / STABLE_LINEUP_PATH
    candidates = [stable] if stable.is_file() else []
    # Legacy stamped runs predate the stable-path replacement policy; prefer
    # the stable artifact but still honor a surviving stamped copy.
    candidates.extend(
        candidate
        for candidate in sorted(root.glob("*/lineups.json"), reverse=True)
        if candidate != stable
    )
    payload: dict[str, Any] | None = None
    for candidate in candidates:
        try:
            parsed = json.loads(candidate.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(parsed, dict):
            payload = parsed
            break
    if payload is None:
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
            # The predictions artifact also contains historical rows. Only
            # the current lineup artifact can be checked against the board.
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
