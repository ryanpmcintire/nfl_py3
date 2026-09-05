"""Lineup-aware assistant intents (ENG-04 / UI-18).

Extends the board assistant's retrieval engine (:mod:`nfl_ats.board_assistant`)
with intents that answer projected-QB and availability questions from the
published ``lineups.json`` artifact ONLY -- via
:class:`nfl_ats.lineup_view.TeamLineup`, the exact same structured data
``board_content.py`` already loads with :func:`nfl_ats.lineup_view.load_lineups`
and attaches to each :class:`nfl_ats.board_content.GameDive`
(``home_lineup``/``away_lineup``). This module never opens an artifact itself
and never calls a live provider; it only composes text from data the caller
already loaded, exactly like every other entry in
:func:`nfl_ats.board_assistant.build_knowledge`.

Every SUPPORTED answer (a resolved starter, a player's availability, a
team's injury notes, or a specific backup-QB game list) names its source and
source-capture time inline, ``"as of <time> from <source>"`` -- see
:data:`_ANCHOR_TEMPLATE`. Two documented fallbacks never guess:

* **Absent** -- no lineup entry exists for the requested team/game (the
  artifact was never published, or does not cover this week's game). See
  :data:`_UNPUBLISHED_TEXT`.
* **Stale** -- a lineup entry exists but its own ``as_of`` timestamp is older
  than :data:`LINEUP_STALE_BUDGET_HOURS` relative to the page's own build
  time (``BoardContent.generated_at``). See :data:`_STALE_TEXT`. Staleness is
  computed ONCE, at corpus-build time (:func:`build_lineup_knowledge`), and
  baked into the corpus as a plain boolean -- the same "precompute at publish
  time, never at query time" discipline every other entry in this corpus
  already follows.

**Fail-closed forecast/lineup consistency rule.** ``scripts/build_week_lineups.py``
already stamps ``TeamLineup.note`` whenever the current depth chart's QB1
disagrees with the forecast's own assumed QB (``model_role == "base_model"``
on a different player, or missing from the roster snapshot entirely). This
module treats a non-``None`` ``note`` as the SAME signal :func:`docs
projected_lineups.md` describes: when it fires, :func:`qb_starter_answer` and
:func:`backup_qb_games_answer` name BOTH the forecast's assumed QB and the
depth chart's current QB1 and explicitly refuse to state a single starter,
rather than picking one silently.

Depends on :mod:`nfl_ats.board_assistant` only through deferred (in-function)
imports, so ``board_assistant`` can import this module at its own top level
without a circular-import failure (``board_assistant`` -> this module is the
only import edge that exists at module-load time).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime
from typing import TYPE_CHECKING, Any

import pandas as pd

from nfl_ats.lineup_view import TeamLineup

if TYPE_CHECKING:
    from nfl_ats.board_assistant import AssistantAnswer

#: How long a published lineups.json snapshot is treated as current. The
#: scheduled refresh runs once a day (docs/projected_lineups.md,
#: scripts/refresh_lineup_forecast.py: noon Eastern, every day of the week),
#: so one missed day of slack (48h) is the line between "normal daily
#: cadence" and "the scheduler stopped running" -- mirrors the
#: MAX_SNAPSHOT_AGE pattern in player_arrests_back_side_overlay.py (36h
#: there, for a source with a different documented cadence).
LINEUP_STALE_BUDGET_HOURS = 48.0

#: Words that route a team question to the QB-starter intent rather than the
#: existing team-pick/confidence/schedule branches in board_assistant.answer.
QB_WORDS: frozenset[str] = frozenset(
    {"qb", "quarterback", "quarterbacks", "starter", "starters", "starting"}
)

#: Words that route a QB question to the "which games have a backup QB"
#: intent instead of a single-team QB-starter question.
BACKUP_WORDS: frozenset[str] = frozenset({"backup", "backups"})

#: Cue words required (alongside a resolved player name) before a bare name
#: mention is treated as an availability question -- keeps a stray shared
#: token from a player's name ("Cook", "Hill", ...) from hijacking an
#: unrelated question that happens to contain it.
AVAILABILITY_WORDS: frozenset[str] = frozenset(
    {
        "playing",
        "play",
        "plays",
        "available",
        "availability",
        "active",
        "inactive",
        "inactives",
        "status",
        "out",
        "questionable",
        "doubtful",
        "injured",
        "injury",
        "hurt",
        "healthy",
        "starting",
    }
)

_ANCHOR_TEMPLATE = "as of {as_of} from {source}"

_UNPUBLISHED_TEXT = (
    "No projected-lineup artifact is published for {team} this week -- the "
    "This Week page's lineup panel only appears once "
    "scripts/build_week_lineups.py has run for this game "
    "(docs/projected_lineups.md); I won't guess."
)

_STALE_TEXT = (
    "The newest projected-lineup snapshot for {team} is {anchor}, which is "
    "older than the {budget:.0f}-hour freshness budget this assistant "
    "enforces -- I won't guess a current starter or availability from a "
    "stale snapshot. Check the live This Week page for a fresher one."
)


def _make_answer(topic: str, text: str, anchors: Sequence[str]) -> AssistantAnswer:
    from nfl_ats.board_assistant import AssistantAnswer as _AssistantAnswer

    return _AssistantAnswer(topic=topic, text=text, anchors=tuple(anchors))


def _hours_since(as_of: str | None, reference: datetime) -> float | None:
    """Hours between ``as_of`` and ``reference``, or ``None`` when ``as_of``
    is absent or unparseable -- callers must treat ``None`` as "cannot
    verify freshness", never as "fresh", per the fail-closed rule."""

    if not as_of:
        return None
    parsed = pd.to_datetime(as_of, utc=True, errors="coerce")
    if pd.isna(parsed):
        return None
    ref = pd.Timestamp(reference)
    ref = ref.tz_localize("UTC") if ref.tzinfo is None else ref.tz_convert("UTC")
    delta = ref - pd.Timestamp(parsed)
    return delta.total_seconds() / 3600.0


def _anchor_text(entry: Mapping[str, Any]) -> str:
    return _ANCHOR_TEMPLATE.format(
        as_of=entry.get("as_of") or "an unrecorded time",
        source=entry.get("source") or "an unrecorded source",
    )


def _team_entry(
    lineup: TeamLineup | None,
    *,
    game_id: str,
    side: str,
    reference: datetime,
    budget_hours: float,
) -> dict[str, Any] | None:
    if lineup is None:
        return None
    age_hours = _hours_since(lineup.as_of, reference)
    stale = age_hours is None or age_hours > budget_hours
    current_qb = min(
        (p for p in lineup.players if p.position == "QB"),
        key=lambda p: p.depth,
        default=None,
    )
    model_qb = next(
        (p for p in lineup.players if p.position == "QB" and p.model_role == "base_model"),
        None,
    )
    players = [
        {
            "name": p.name,
            "position": p.position,
            "slot": p.slot,
            "depth": p.depth,
            "unit": p.unit,
            "gsis_id": p.gsis_id,
            "play_probability": p.play_probability,
            "injury_status": p.injury_status,
            "model_role": p.model_role,
            "probability_source": p.probability_source,
        }
        for p in lineup.players
    ]
    return {
        "game_id": game_id,
        "side": side,
        "team": lineup.team,
        "source": lineup.source,
        "as_of": lineup.as_of,
        "stale": stale,
        "note": lineup.note,
        "injury_status": lineup.injury_status,
        "current_qb_name": current_qb.name if current_qb is not None else None,
        "model_qb_name": model_qb.name if model_qb is not None else None,
        "players": players,
    }


def build_lineup_knowledge(
    lineups: Mapping[str, tuple[TeamLineup, TeamLineup]],
    *,
    reference: datetime,
    budget_hours: float = LINEUP_STALE_BUDGET_HOURS,
) -> dict[str, Any]:
    """The precomputed, retrieval-only lineup block merged into the
    assistant corpus by ``board_assistant.build_knowledge_for_board``.

    ``lineups`` is exactly what :func:`nfl_ats.lineup_view.load_lineups`
    returns (``{game_id: (home, away)}``); ``reference`` is the page's own
    build time (``BoardContent.generated_at``), so staleness is computed
    once, at publish time -- never re-derived at query time in Python or JS.
    """

    games: dict[str, Any] = {}
    players: list[dict[str, Any]] = []
    for game_id, (home, away) in lineups.items():
        home_entry = _team_entry(
            home, game_id=game_id, side="home", reference=reference, budget_hours=budget_hours
        )
        away_entry = _team_entry(
            away, game_id=game_id, side="away", reference=reference, budget_hours=budget_hours
        )
        if home_entry is None and away_entry is None:
            continue
        games[game_id] = {"home": home_entry, "away": away_entry}
        for entry in (home_entry, away_entry):
            if entry is None:
                continue
            for player in entry["players"]:
                players.append(
                    {
                        **player,
                        "team": entry["team"],
                        "game_id": entry["game_id"],
                        "side": entry["side"],
                        "as_of": entry["as_of"],
                        "source": entry["source"],
                        "stale": entry["stale"],
                    }
                )
    return {
        "games": games,
        "players": players,
        "stale_budget_hours": budget_hours,
    }


def _team_lookup(
    lineup_knowledge: Mapping[str, Any], team_code: str
) -> tuple[str, Mapping[str, Any]] | None:
    code = team_code.upper()
    for game_id, sides in lineup_knowledge.get("games", {}).items():
        for side in ("home", "away"):
            entry = sides.get(side)
            if entry is not None and str(entry.get("team", "")).upper() == code:
                return str(game_id), entry
    return None


def qb_starter_answer(
    teams: Sequence[str], lineup_knowledge: Mapping[str, Any] | None
) -> AssistantAnswer | None:
    """ "Who is starting at QB for <team>" -- refuses to name a single
    starter when the fail-closed consistency rule fires (``entry["note"]``)
    or the snapshot is stale/absent, per the module docstring."""

    if not teams:
        return None
    if lineup_knowledge is None:
        lineup_knowledge = {"games": {}}
    parts: list[str] = []
    anchors: list[str] = []
    for code in teams:
        found = _team_lookup(lineup_knowledge, code)
        if found is None:
            parts.append(_UNPUBLISHED_TEXT.format(team=code.upper()))
            continue
        game_id, entry = found
        anchors.append(f"index.html#{game_id}")
        if entry["stale"]:
            parts.append(
                _STALE_TEXT.format(
                    team=entry["team"],
                    anchor=_anchor_text(entry),
                    budget=lineup_knowledge.get("stale_budget_hours", LINEUP_STALE_BUDGET_HOURS),
                )
            )
            continue
        anchor = _anchor_text(entry)
        if entry["note"]:
            model_name = entry["model_qb_name"] or "a QB not on the current roster snapshot"
            current_name = entry["current_qb_name"] or "no QB listed on the current snapshot"
            parts.append(
                f"{entry['team']}: the published forecast assumed {model_name} at QB, but the "
                f"current depth-chart snapshot ({anchor}) lists {current_name} at QB1 instead -- "
                "I can't state a single starter until the forecast is regenerated from this "
                "snapshot."
            )
        else:
            name = entry["current_qb_name"] or "no QB listed on the current snapshot"
            parts.append(f"{entry['team']} starting QB: {name} ({anchor}).")
    if not parts:
        return None
    return _make_answer("lineup:qb", " ".join(parts), anchors or ("index.html",))


def team_injuries_answer(
    teams: Sequence[str], lineup_knowledge: Mapping[str, Any] | None
) -> AssistantAnswer | None:
    """ "Any injuries for <team>" -- reports only what the lineup artifact
    itself carries; never infers a status the artifact doesn't publish."""

    if not teams:
        return None
    if lineup_knowledge is None:
        lineup_knowledge = {"games": {}}
    parts: list[str] = []
    anchors: list[str] = []
    for code in teams:
        found = _team_lookup(lineup_knowledge, code)
        if found is None:
            parts.append(_UNPUBLISHED_TEXT.format(team=code.upper()))
            continue
        game_id, entry = found
        anchors.append(f"index.html#{game_id}")
        if entry["stale"]:
            parts.append(
                _STALE_TEXT.format(
                    team=entry["team"],
                    anchor=_anchor_text(entry),
                    budget=lineup_knowledge.get("stale_budget_hours", LINEUP_STALE_BUDGET_HOURS),
                )
            )
            continue
        anchor = _anchor_text(entry)
        flagged = [p for p in entry["players"] if p.get("injury_status")]
        if flagged:
            listing = "; ".join(f"{p['name']} ({p['injury_status']})" for p in flagged)
            parts.append(f"{entry['team']} injury notes ({anchor}): {listing}.")
        else:
            status = entry["injury_status"] or "unavailable"
            parts.append(
                f"{entry['team']}: no per-player injury designation in the lineup snapshot "
                f"({anchor}); team-level injury feed status: {status}."
            )
    if not parts:
        return None
    return _make_answer("lineup:injuries", " ".join(parts), anchors or ("index.html",))


def _dedupe_players(players: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    seen: dict[Any, Mapping[str, Any]] = {}
    for player in players:
        key = player.get("gsis_id") or (player.get("name"), player.get("team"))
        seen.setdefault(key, player)
    return list(seen.values())


def _resolve_players(
    tokens: frozenset[str], players: Sequence[Mapping[str, Any]]
) -> list[Mapping[str, Any]]:
    from nfl_ats.board_assistant import _tokens as _tokenize

    _SUFFIXES = {"jr", "sr", "ii", "iii", "iv"}
    exact: list[Mapping[str, Any]] = []
    for player in players:
        name_tokens = [t for t in _tokenize(str(player["name"])) if t not in _SUFFIXES]
        if name_tokens and all(t in tokens for t in name_tokens):
            exact.append(player)
    if exact:
        return exact
    last_name_hits: list[Mapping[str, Any]] = []
    for player in players:
        name_tokens = [t for t in _tokenize(str(player["name"])) if t not in _SUFFIXES]
        if name_tokens and name_tokens[-1] in tokens:
            last_name_hits.append(player)
    return last_name_hits


def player_availability_answer(
    tokens: frozenset[str], lineup_knowledge: Mapping[str, Any] | None
) -> AssistantAnswer | None:
    """ "Is <player> playing / available" -- only fires when a player name
    resolves AND the question carries an availability/status cue word, so a
    shared surname token can never hijack an unrelated question."""

    if lineup_knowledge is None or not (tokens & AVAILABILITY_WORDS):
        return None
    players = lineup_knowledge.get("players", ())
    if not players:
        return None
    matches = _resolve_players(tokens, players)
    if not matches:
        return None
    distinct = _dedupe_players(matches)
    if len(distinct) > 1:
        names = ", ".join(sorted(f"{p['name']} ({p['team']})" for p in distinct))
        return _make_answer(
            "lineup:availability",
            f"More than one player in this week's published lineups matches that name: "
            f"{names}. Ask again naming the team.",
            ("index.html",),
        )
    parts: list[str] = []
    anchors: list[str] = []
    for player in distinct:
        anchors.append(f"index.html#{player['game_id']}")
        if player.get("stale"):
            parts.append(
                _STALE_TEXT.format(
                    team=player["team"],
                    anchor=_anchor_text(player),
                    budget=lineup_knowledge.get("stale_budget_hours", LINEUP_STALE_BUDGET_HOURS),
                )
            )
            continue
        anchor = _anchor_text(player)
        probability = player.get("play_probability")
        probability_text = f"{probability:.0%}" if probability is not None else "not published"
        source = player.get("probability_source")
        source_note = {
            "base_model_qb": ", from the active model's own forecast input",
            "availability_model": ", from the availability model (this week's injury "
            "designation, or the position's no-designation base rate when unlisted)",
            "unavailable": " (no gsis_id or rate available for this player)",
        }.get(str(source), "")
        injury = player.get("injury_status") or "no report"
        role_note = (
            "the forecast's assumed starter"
            if player.get("model_role") == "base_model"
            else "context only -- not the model's scored player"
        )
        parts.append(
            f"{player['name']} ({player['team']}, {player['slot']}): play probability "
            f"{probability_text}{source_note}, injury status {injury}, {role_note} ({anchor})."
        )
    return _make_answer("lineup:availability", " ".join(parts), tuple(anchors))


def backup_qb_games_answer(lineup_knowledge: Mapping[str, Any] | None) -> AssistantAnswer:
    """ "Which games have a backup QB" -- reads this as the same fail-closed
    signal ``qb_starter_answer`` refuses a single starter over: a team whose
    current depth-chart QB1 disagrees with the forecast's assumed QB. Games
    with a stale snapshot are excluded (never guessed) and named as such."""

    if lineup_knowledge is None or not lineup_knowledge.get("games"):
        return _make_answer(
            "lineup:backup_qb",
            "No projected-lineup artifact is published this week, so I can't compare current "
            "depth-chart starters to the forecast.",
            ("index.html",),
        )
    hits: list[str] = []
    anchors: list[str] = []
    stale_any = False
    for game_id, sides in lineup_knowledge["games"].items():
        for side in ("home", "away"):
            entry = sides.get(side)
            if entry is None:
                continue
            if entry["stale"]:
                stale_any = True
                continue
            if entry["note"]:
                model_name = entry["model_qb_name"] or "a QB not on the current roster snapshot"
                current_name = entry["current_qb_name"] or "no QB listed"
                hits.append(
                    f"{entry['team']} ({_anchor_text(entry)}): forecast assumed {model_name}, "
                    f"current snapshot lists {current_name}"
                )
                anchors.append(f"index.html#{game_id}")
    tail = (
        " (at least one team's snapshot is stale and was excluded from this answer, never guessed)"
        if stale_any
        else ""
    )
    if not hits:
        return _make_answer(
            "lineup:backup_qb",
            "No team's current depth-chart QB1 disagrees with its forecast-assumed QB in the "
            f"published lineup snapshot{tail}.",
            ("index.html",),
        )
    return _make_answer(
        "lineup:backup_qb",
        "Depth chart lists a different QB than the forecast assumed for: "
        + "; ".join(hits)
        + f"{tail}."
        + ' ("Backup QB" here means the current depth chart disagrees with the forecast\'s '
        "assumed starter, not merely that a backup is on the roster.)",
        tuple(anchors),
    )


__all__ = [
    "AVAILABILITY_WORDS",
    "BACKUP_WORDS",
    "LINEUP_STALE_BUDGET_HOURS",
    "QB_WORDS",
    "backup_qb_games_answer",
    "build_lineup_knowledge",
    "player_availability_answer",
    "qb_starter_answer",
    "team_injuries_answer",
]
