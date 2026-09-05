"""Static guided board assistant (UI-16).

A chat panel over publish-time-generated knowledge: every answer is
RETRIEVED from the page's own embedded corpus, never generated. There is
no LLM, no backend, no network call -- the corpus ships inline in each
page as ``<script type="application/json">`` (a separate
``assistant_knowledge.json`` file is NOT written: the hosted dashboard's
nginx allowlist serves only the four HTML pages, and its
``connect-src 'none'`` CSP bans ``fetch``, so a sidecar file would be
unreachable exactly where it matters; see ``docs/board_assistant_scout.md``).

The ranking rule below is the port contract shared by the Python
reference matcher (:func:`answer`, fully tested) and the thin inline-JS
port in :data:`ASSISTANT_SCRIPT` (which only ranks the same embedded
entries with the same embedded synonym table and returns the winning
entry's own body -- it cannot compose new text, so by construction it
cannot emit a probability, pick, or record absent from the corpus).
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from html import escape
from typing import Any

from nfl_ats.board_assistant_lineups import AVAILABILITY_WORDS as _LINEUP_AVAILABILITY_WORDS
from nfl_ats.board_assistant_lineups import BACKUP_WORDS as _LINEUP_BACKUP_WORDS
from nfl_ats.board_assistant_lineups import QB_WORDS as _LINEUP_QB_WORDS
from nfl_ats.board_assistant_lineups import backup_qb_games_answer as _lineup_backup_qb_games_answer
from nfl_ats.board_assistant_lineups import build_lineup_knowledge as _build_lineup_knowledge
from nfl_ats.board_assistant_lineups import (
    player_availability_answer as _lineup_player_availability_answer,
)
from nfl_ats.board_assistant_lineups import qb_starter_answer as _lineup_qb_starter_answer
from nfl_ats.board_assistant_lineups import team_injuries_answer as _lineup_team_injuries_answer
from nfl_ats.board_content import (
    BoardContent,
    GameRow,
    SourcePolicyView,
    TiebreakerView,
    human_update_time,
)
from nfl_ats.board_site_content import (
    FindingsPageContent,
    HistoryPageContent,
    ModelPageContent,
)
from nfl_ats.market_data import NFL_TEAM_NAMES
from nfl_ats.public_board import humanize_identifier

#: Corpus schema version, stamped into every payload's provenance block.
ASSISTANT_VERSION = 1

#: Minimum query-token length admitted to scoring. Team codes are two
#: characters (``NE``, ``LA``); single characters only add noise.
_MIN_TOKEN_LENGTH = 2

#: Filler words dropped before scoring (never before deflect matching,
#: which runs on the raw query). Without these, ``the`` matches the
#: ``panthers`` keyword and every ``pick`` question ties every game.
STOPWORDS = frozenset(
    {
        "the",
        "a",
        "an",
        "of",
        "to",
        "in",
        "on",
        "at",
        "for",
        "with",
        "from",
        "by",
        "and",
        "or",
        "if",
        "then",
        "than",
        "so",
        "as",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "do",
        "does",
        "did",
        "will",
        "would",
        "can",
        "could",
        "should",
        "shall",
        "i",
        "me",
        "my",
        "we",
        "us",
        "you",
        "your",
        "it",
        "its",
        "this",
        "that",
        "there",
        "their",
    }
)


@dataclass(frozen=True)
class GlossaryEntry:
    term: str
    definition: str


#: Evergreen definitions only -- no number, no record, no claim that can
#: go stale. (A stale number here would need the findings-page curation
#: contract; plain football vocabulary does not.)
GLOSSARY: tuple[GlossaryEntry, ...] = (
    GlossaryEntry(
        "ATS",
        "Against the spread: picking the side that covers the point "
        "spread, not the side that wins the game.",
    ),
    GlossaryEntry(
        "cover",
        "A team covers when its final margin beats the spread: a "
        "favorite by more than the points it lays, an underdog by losing "
        "by less than the points it gets (or winning outright).",
    ),
    GlossaryEntry(
        "spread",
        "The point handicap the market sets: positive favors the home "
        "team. This board restates it from the picked team's side, so a "
        "plus sign means points received and a minus sign means points "
        "laid.",
    ),
    GlossaryEntry(
        "push",
        "A tie against the spread: the final margin lands exactly on "
        "the line, so neither side covers.",
    ),
    GlossaryEntry(
        "opener",
        "The Tuesday line this pool grades against. Picks lock against "
        "a frozen Tuesday line; later line moves do not change the grade.",
    ),
    GlossaryEntry(
        "closing line",
        "The market's final line before kickoff -- the sharpest read, "
        "reported on this site only as a secondary grade, never the one "
        "the pool settles on.",
    ),
    GlossaryEntry(
        "cover probability",
        "The model's estimated chance the picked side covers, from a "
        "smooth Gaussian read of its out-of-time residual distribution. "
        "A decision-strength score, not a historical win rate.",
    ),
    GlossaryEntry(
        "Best Pick",
        "The pool scores one Best Pick per week. This board nominates "
        "it by calibrated probability among low-disagreement games.",
    ),
    GlossaryEntry(
        "confidence",
        "The board's three-band vocabulary for a pick's cover "
        "probability: slight, lean, or strong.",
    ),
    GlossaryEntry(
        "vig",
        "The bookmaker's commission baked into betting prices. The "
        "forced-pick pool card ignores prices; vig appears only in "
        "secondary paper-betting diagnostics, never in a pick.",
    ),
)

#: Extra lookup words per glossary term. Kept minimal on purpose: every
#: alias is a word that can only mean its term here ("line" is NOT an
#: alias of anything — it belongs to movement questions).
GLOSSARY_ALIASES: dict[str, tuple[str, ...]] = {
    "vig": ("juice", "odds"),
    "ATS": ("against the spread",),
}


def _team_synonyms() -> dict[str, tuple[str, ...]]:
    """Canonical team code -> query aliases (code, city, nickname)."""

    synonyms: dict[str, tuple[str, ...]] = {}
    for full_name, code in NFL_TEAM_NAMES.items():
        parts = full_name.split()
        aliases = {code.lower(), full_name.lower(), parts[-1].lower()}
        if len(parts) > 1:
            aliases.add(" ".join(parts[:-1]).lower())
            aliases.add(parts[0].lower())
        # "New York" is shared by two teams; keep it off both alias sets
        # so it can never route a question to the wrong game.
        aliases.discard("new york")
        aliases.discard("new")
        aliases.discard("los angeles")
        aliases.discard("los")
        synonyms[code.lower()] = tuple(sorted(aliases))
    return synonyms


_TEAM_SYNONYMS = _team_synonyms()

#: Canonical topic -> query aliases. Rendered into every page from this
#: one constant, so the inline-JS port can never drift from the tested
#: Python matcher (pinned by
#: ``tests/test_board_assistant.py::test_rendered_synonyms_match_python``).
SYNONYMS: dict[str, tuple[str, ...]] = {
    "best_pick": (
        "best pick",
        "best-pick",
        "bestpick",
        "best bet",
        "star",
        "pick of the week",
        "lock",
        "locks",
        "mortal lock",
    ),
    "record": (
        "record",
        "accuracy",
        "how good",
        "track record",
        "profitable",
        "profit",
        "edge",
        "win rate",
        "history of the model",
        "past performance",
    ),
    "policy": (
        "policy",
        "overlay",
        "flip",
        "flipped",
        "changed",
        "coach fade",
        "revenge",
        "arrests",
        "spread gap",
        "why",
    ),
    "findings": ("finding", "findings", "learned", "lesson", "research", "study", "signal"),
    "leads": ("watching", "lead", "leads", "prospect", "challenger", "working on"),
    "glossary": ("mean", "means", "definition", "glossary", "explain", "what is", "what does"),
    "timing": (
        "when",
        "lock",
        "tuesday",
        "deadline",
        "update",
        "refresh",
        "kickoff",
        "publish",
        "change since",
        "movement",
        "move",
        "moves",
        "moved",
    ),
}


@dataclass(frozen=True)
class _Entry:
    """One fixed answer: a precomputed body plus its anchor. Routing no
    longer scores keywords — the intent parser in :func:`answer` decides,
    and these entries are looked up by id."""

    entry_id: str
    body: str
    anchor: str


@dataclass(frozen=True)
class AssistantAnswer:
    topic: str
    text: str
    anchors: tuple[str, ...]


def _tokens(text: str) -> tuple[str, ...]:
    return tuple(
        token
        for token in "".join(char.lower() if char.isalnum() else " " for char in text).split()
        if len(token) >= _MIN_TOKEN_LENGTH and token not in STOPWORDS
    )


def _team_hits(tokens: frozenset[str]) -> tuple[str, ...]:
    """Team codes whose every alias-word is present as a query token.

    Multi-word aliases ("new england") match word-wise, so word order
    and extra words never matter; shared fragments ("la") can only
    match the literal token, never a substring of another word.
    """

    hits = []
    for code, aliases in _TEAM_SYNONYMS.items():
        for alias in aliases:
            if all(word in tokens for word in alias.split()):
                hits.append(code)
                break
    # Sorted, not detection order: the JS port reads the alphabetically
    # sorted teams table, and multi-team answers must come out identical.
    return tuple(sorted(hits))


_DAY_NAMES = (
    "Sunday",
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
)

_DAY_TOKENS: dict[str, tuple[str, ...]] = {
    "Sunday": ("sunday",),
    "Monday": ("monday",),
    "Tuesday": ("tuesday",),
    "Wednesday": ("wednesday", "wednesdays"),
    "Thursday": ("thursday", "thursdays", "tnf"),
    "Friday": ("friday",),
    "Saturday": ("saturday",),
}

_EXPLAIN_WORDS = frozenset(
    {"mean", "means", "meaning", "define", "definition", "explain", "explains", "stands"}
)
_CONFIDENCE_WORDS = frozenset(
    {"confident", "confidence", "sure", "strong", "weak", "certain", "nervous"}
)
_SCHEDULE_WORDS = frozenset({"when", "play", "playing", "kickoff", "date", "schedule", "slot"})
_CHANGE_WORDS = frozenset(
    {"change", "changed", "changes", "update", "updated", "refresh", "refreshed"}
)
_RANK_WORDS = frozenset(
    {
        "rank",
        "ranked",
        "ranking",
        "rankings",
        "top",
        "bottom",
        "least",
        "most",
        "weakest",
        "strongest",
        "surest",
    }
)
_DOG_WORDS = frozenset({"underdog", "underdogs", "dog", "dogs", "upset", "upsets"})
_FAVORITE_WORDS = frozenset(
    {"favorite", "favorites", "favourite", "favourites", "fav", "favs", "chalk"}
)
_RECORD_WORDS = frozenset(
    {
        "record",
        "accuracy",
        "accurate",
        "historically",
        "history",
        "past",
        "profitable",
        "profit",
        "edge",
        "track",
        "good",
    }
)
_POLICY_WORDS = frozenset(
    {
        "policy",
        "policies",
        "overlay",
        "overlays",
        "flip",
        "flipped",
        "flips",
        "fade",
        "revenge",
        "arrest",
        "arrests",
    }
)
_FINDINGS_WORDS = frozenset(
    {
        "finding",
        "findings",
        "learned",
        "lesson",
        "lessons",
        "research",
        "study",
        "signal",
        "interesting",
    }
)
_TIMING_WORDS = frozenset(
    {
        "when",
        "lock",
        "locked",
        "locks",
        "tuesday",
        "deadline",
        "update",
        "refresh",
        "kickoff",
        "publish",
        "published",
        "movement",
        "move",
        "moves",
        "moved",
    }
)
_WINNERS_WORDS = frozenset({"win", "wins", "winner", "winners", "victory", "moneyline", "outright"})
_INJURY_WORDS = frozenset(
    {
        "injury",
        "injuries",
        "injured",
        "hurt",
        "doubtful",
        "questionable",
        "inactive",
        "inactives",
        "health",
        "concussion",
        "illness",
        "limited",
    }
)
_WEATHER_WORDS = frozenset(
    {
        "weather",
        "wind",
        "windy",
        "rain",
        "rainy",
        "snow",
        "snowy",
        "cold",
        "heat",
        "temperature",
        "dome",
        "forecast",
    }
)
#: ENG-34: "were the sources complete this week" and siblings -- routes to
#: the single "sources" corpus entry :func:`build_knowledge_for_board`
#: appends from ``board.source_policy`` (ENG-14's card state). No golden
#: question collides: the two existing "...-blocked?" accuracy questions
#: are caught by ``_RECORD_WORDS`` earlier in :func:`answer`'s dispatch, and
#: the one "...what's your source?" lineup question is caught by the
#: earlier ``parsed.teams`` block.
_SOURCE_POLICY_WORDS = frozenset(
    {
        "source",
        "sources",
        "freshness",
        "fresh",
        "snapshot",
        "snapshots",
        "stale",
        "degraded",
        "blocked",
    }
)
#: UI-20(g) extension (2026-09-05): "what's the tiebreaker" and siblings --
#: routes to the single "tiebreaker" corpus entry :func:`build_knowledge_for_board`
#: appends from ``board.tiebreaker``, read straight off ``nfl_ats.publishing``'s
#: persisted ``tiebreaker.json``. Checked in :func:`answer` AFTER the
#: deflect set, which no longer catches bare "tiebreaker" (see
#: :func:`_deflect_rule_sets`'s comment) but still catches an UNRELATED
#: "guess the exact/final score" question about a regular game.
_TIEBREAKER_WORDS = frozenset({"tiebreaker", "tiebreak"})


@dataclass(frozen=True)
class _Parsed:
    tokens: frozenset[str]
    teams: tuple[str, ...]
    days: tuple[str, ...]
    term: str | None
    number: int | None
    least: bool


def _glossary_term_match(ordered: tuple[str, ...], glossary_terms: Any) -> str | None:
    """Longest-match-first phrase lookup for :func:`_parse`'s ``term``
    field (ENG-36): every glossary term name and alias is re-tokenised
    with :func:`_tokens` -- the SAME normalisation already applied to the
    query -- into an n-gram of words, then the longest n-gram of
    ``ordered`` equal to a candidate's word tuple wins. This makes
    multi-word terms ("cover probability", "closing line", "Best Pick")
    and multi-word aliases ("against the spread") reachable, while a
    single-word term still matches exactly as the old set-membership
    check did (a 1-token "n-gram" is just token-in-``ordered``).
    """

    candidates: list[tuple[tuple[str, ...], str]] = []
    longest = 0
    for item in glossary_terms:
        for name in (item["term"], *item.get("aliases", ())):
            name_tokens = _tokens(name)
            if name_tokens:
                candidates.append((name_tokens, item["term"]))
                longest = max(longest, len(name_tokens))
    for length in range(min(longest, len(ordered)), 0, -1):
        for start in range(len(ordered) - length + 1):
            gram = ordered[start : start + length]
            for name_tokens, term in candidates:
                if name_tokens == gram:
                    return term
    return None


def _parse(question: str, knowledge: Mapping[str, Any]) -> _Parsed:
    """Parse meaning, not keywords: entities (teams, days, glossary
    terms, counts) plus intent signals. Pure function of the query and
    the corpus team/glossary tables."""

    ordered = _tokens(question)
    tokens = frozenset(ordered)
    teams = _team_hits(tokens)
    # Uppercase codes disambiguate WAS-the-team from was-the-verb: only
    # exact all-caps tokens count here, so lowercase prose never matches.
    raw_codes = set(re.findall(r"\b[A-Z]{2,3}\b", question))
    valid = {code for code in _TEAM_SYNONYMS if code.upper() in raw_codes}
    teams = tuple(sorted(set(teams) | set(valid)))
    days: list[str] = []
    for day, words in _DAY_TOKENS.items():
        if any(word in tokens for word in words):
            days.append(day)
    if "weekend" in tokens:
        for day in ("Saturday", "Sunday"):
            if day not in days:
                days.append(day)
    term = _glossary_term_match(ordered, knowledge.get("glossary_terms", ()))
    number: int | None = None
    # Digit scan ignores the token length floor: "top 5" must read 5.
    raw_words = "".join(char.lower() if char.isalnum() else " " for char in question).split()
    for token in raw_words:
        if token.isdigit():
            number = int(token)
            break
    least = bool(tokens & {"least", "bottom", "worst", "weakest"})
    return _Parsed(
        tokens=tokens, teams=teams, days=tuple(days), term=term, number=number, least=least
    )


def _deflect_entries(season: int, week: int) -> tuple[_Entry, ...]:
    """The must-deflect set, in the exact order of
    :func:`_deflect_rule_sets` (rule *i* fires entry *i*): each rule is
    an AND of OR-groups, so ``best bet`` still routes to the Best Pick
    while ``should I bet`` deflects."""

    wager = _Entry(
        entry_id="deflect:wager",
        body=(
            "This board doesn't give betting advice: every pick here is a "
            "paper pick for testing a forecasting model, and a small "
            "historical edge doesn't tell you it holds up. The card itself "
            "is on This week."
        ),
        anchor="index.html",
    )
    future = _Entry(
        entry_id="deflect:future",
        body=(
            f"This published card covers {season} Week {week} only. "
            "Future weeks have no published picks, and pending games "
            "have no outcomes yet -- see History once results settle."
        ),
        anchor="history.html",
    )
    ownership = _Entry(
        entry_id="deflect:ownership",
        body=(
            "No pick-popularity feed exists on this board: the pool's "
            "pick distribution unlocks game by game and is unavailable "
            "before each deadline, so there is nothing here about who "
            "everyone else is picking."
        ),
        anchor="findings.html",
    )
    score = _Entry(
        entry_id="deflect:score",
        body=(
            "The published card covers sides against the spread only, "
            "not exact final scores. Tiebreak guesses are a separate "
            "weekly exercise, not part of this card."
        ),
        anchor="index.html",
    )
    # Rule i fires entries[i]: teaser/buy-points ride the wager entry,
    # over-under rides score, fade-the-public rides ownership.
    return (
        wager,
        wager,
        future,
        ownership,
        ownership,
        score,
        score,
        wager,
        wager,
        score,
        ownership,
    )


def _deflect_rule_sets() -> tuple[tuple[frozenset[str], ...], ...]:
    return (
        (
            frozenset({"bet", "wager", "gamble", "parlay", "kelly", "bankroll", "stake"}),
            frozenset({"should", "much", "place", "put", "make", "advice", "recommend"}),
        ),
        (frozenset({"wager", "parlay", "kelly", "bankroll", "gamble"}),),
        (
            frozenset(
                {
                    "next",
                    "2",
                    "3",
                    "4",
                    "5",
                    "6",
                    "7",
                    "8",
                    "9",
                    "10",
                    "playoff",
                    "super",
                    "rest",
                    "future",
                }
            ),
            frozenset({"week", "weeks", "playoff", "playoffs", "bowl", "season", "games"}),
        ),
        (frozenset({"ownership", "consensus"}),),
        (
            frozenset({"public", "popular", "everyone", "squares", "sharps"}),
            frozenset({"picking", "picks", "betting", "money", "side", "consensus"}),
        ),
        # "tiebreaker" itself is a real intent now (UI-20(g) extension,
        # 2026-09-05: it answers from the published nfl_ats.publishing
        # tiebreaker.json, never a wagering deflection) -- see
        # _TIEBREAKER_WORDS below. "exact score"/"final score" alone (a
        # random game's score, not the pool's tiebreaker) stay deflected.
        (frozenset({"exact score", "final score"}),),
        (
            frozenset({"score", "scoreline"}),
            frozenset({"predict", "prediction", "guess", "exact", "forecast", "be"}),
        ),
        (frozenset({"teaser", "teasers"}),),
        (frozenset({"buy points", "buying points"}),),
        (frozenset({"over under", "over/under", "totals", "total points"}),),
        (frozenset({"fade the public", "fade public"}),),
    )


def _deflect_words(query: str) -> tuple[set[str], str]:
    """Word set plus order-preserving flat string, punctuation stripped
    (so a trailing ``?`` cannot defeat the must-deflect set). No length
    filter here: single-digit weeks (``week 2``) must stay matchable."""

    flat = " ".join("".join(char.lower() if char.isalnum() else " " for char in query).split())
    return set(flat.split()), flat


def _deflect_fires(rule: tuple[frozenset[str], ...], query: str) -> bool:
    words, flat = _deflect_words(query)
    padded = f" {flat} "
    return all(
        any((f" {alias} " in padded) if " " in alias else (alias in words) for alias in group)
        for group in rule
    )


def _game_body(game: GameRow, *, best_pick_note: str | None) -> str:
    kickoff = f"{game.weekday_name[:3]} {game.gameday.strftime('%b %d')}"
    parts = [
        f"{game.away} at {game.home} ({kickoff}): "
        f"pick {game.pick_team} {game.pick_spread_text} -- "
        f"{game.probability_text} cover probability ({game.confidence_word}). "
        f"Line: {game.spread_text}."
    ]
    if game.is_best and best_pick_note:
        parts.append(f"Best Pick of the week. {best_pick_note}")
    elif game.is_best:
        parts.append("Best Pick of the week.")
    if game.is_flipped and game.flip_member_labels:
        parts.append(f"Policy flip: {' + '.join(game.flip_member_labels)}.")
    if game.final and game.cover_result_label:
        tail = f"Final: {game.cover_result_label}."
        if game.final_score_text:
            tail += f" {game.final_score_text}."
        parts.append(tail)
    return " ".join(parts)


def build_knowledge(
    *,
    page: str,
    season: int | None,
    week: int | None,
    generated_at_text: str,
    model_id: str | None,
    method_label: str,
    games: tuple[GameRow, ...],
    best_pick_game_id: str | None,
    best_pick_note: str | None,
    policy_text: str | None,
    record_lines: tuple[str, ...],
    finding_items: tuple[tuple[str, str], ...],
    watching_items: tuple[tuple[str, str, str, float], ...],
    refresh_lines: tuple[str, ...] = (),
) -> dict[str, Any]:
    """Build the deterministic retrieval corpus for one page.

    Every answer body is composed verbatim from the inputs -- this
    function invents no number and no claim, so the numeric-guard test
    (every number in an answer also occurs in the corpus) holds by
    construction.
    """

    entries: list[_Entry] = list(_deflect_entries(int(season or 0), int(week or 0)))

    # The refresh entry sits ahead of the games so a change-question
    # with a team name in it still routes to the refresh diff; pure
    # team questions score zero here and fall through to their game.
    # It is emitted ONLY when a refresh actually ran — pre-lock there
    # is nothing to report and clock questions stay with timing.
    if refresh_lines:
        entries.append(
            _Entry(
                entry_id="refresh",
                body="Late-week refresh: " + " ".join(refresh_lines),
                anchor="index.html",
            )
        )

    games_data: list[dict[str, Any]] = [
        {
            "game_id": game.game_id,
            "away": game.away,
            "home": game.home,
            "kickoff": f"{game.weekday_name[:3]} {game.gameday.strftime('%b %d')}",
            "day": game.weekday_name,
            "spread": game.spread_text,
            "pick": game.pick_team,
            "pick_spread": game.pick_spread_text,
            "probability": round(game.pick_probability, 4),
            "probability_text": game.probability_text,
            "confidence": game.confidence_word,
            "best": bool(game.is_best),
            "flips": list(game.flip_member_labels),
            "final": game.cover_result_label or None,
            "final_score": game.final_score_text,
            "why": _game_body(game, best_pick_note=best_pick_note),
            "anchor": f"index.html#{game.game_id}",
        }
        for game in games
    ]
    by_probability = sorted(games_data, key=lambda item: (-item["probability"], item["game_id"]))
    ranked_ids = [item["game_id"] for item in by_probability]
    for position, item in enumerate(by_probability, start=1):
        item["rank"] = position
    dog_ids = [item["game_id"] for item in games_data if item["pick_spread"].startswith("+")]
    favorite_ids = [item["game_id"] for item in games_data if item["pick_spread"].startswith("-")]
    flat_pick_ids = [
        item["game_id"] for item in games_data if not item["pick_spread"].startswith(("+", "-"))
    ]
    day_order: list[str] = []
    day_map: dict[str, list[str]] = {}
    for item in games_data:
        if item["day"] not in day_map:
            day_map[item["day"]] = []
            day_order.append(item["day"])
        day_map[item["day"]].append(item["game_id"])

    if best_pick_game_id is not None:
        best = next((game for game in games if game.game_id == best_pick_game_id), None)
        label = (
            f"{best.pick_team} {best.pick_spread_text}" if best is not None else best_pick_game_id
        )
        body = f"Best Pick of the week: {label}."
        if best_pick_note:
            body += f" {best_pick_note}"
        entries.append(
            _Entry(
                entry_id="best_pick",
                body=body,
                anchor=f"index.html#{best_pick_game_id}",
            )
        )

    if record_lines:
        entries.append(
            _Entry(
                entry_id="record",
                body=" ".join(record_lines),
                anchor="model.html",
            )
        )

    if policy_text:
        entries.append(
            _Entry(
                entry_id="policy",
                body=(f"{policy_text} Games the policy flipped are marked on the board."),
                anchor="index.html",
            )
        )

    for tag, text in finding_items:
        entries.append(
            _Entry(
                entry_id=f"finding:{tag}",
                body=text,
                anchor="findings.html",
            )
        )

    for name, description, effect_text, probability_positive in watching_items:
        # ``description`` is already plain English (a curated blurb, a
        # recorded plain_summary, or the "Plain-English summary pending"
        # placeholder -- see board_site_content._watching_lead_view); this
        # must never hand-build its own sentence out of the raw jargon
        # fields the way an earlier version of this loop did (2026-09-05
        # fix, dashboard humanising follow-up to lane AH's audit -- that
        # jargon reaches a reader the moment the assistant answers a
        # question about this lead, even though it never appears in the
        # page's static HTML).
        entries.append(
            _Entry(
                entry_id=f"watching:{name}",
                body=(
                    f"{humanize_identifier(name)}: {description} "
                    f"({effect_text}, about {probability_positive:.0%} likely real -- "
                    "an open lead, not a settled verdict)."
                ),
                anchor="findings.html",
            )
        )

    for glossary_item in GLOSSARY:
        entries.append(
            _Entry(
                entry_id=f"glossary:{glossary_item.term}",
                body=f"{glossary_item.term}: {glossary_item.definition}",
                anchor="index.html",
            )
        )

    entries.append(
        _Entry(
            entry_id="timing",
            body=(
                "Picks can be updated until each game's own kickoff; the "
                "pool's lines freeze Tuesday. If a captured line moves at "
                "least a point from the frozen Tuesday line, the refreshed "
                "pick follows the market side. This card was generated "
                f"{human_update_time(generated_at_text)}."
            ),
            anchor="index.html",
        )
    )

    payload = {
        "assistant_version": ASSISTANT_VERSION,
        "generated_at": human_update_time(generated_at_text),
        "season": season,
        "week": week,
        "page": page,
        "model": {
            "model_id": model_id,
            "method_label": method_label,
        },
        "synonyms": {key: list(value) for key, value in SYNONYMS.items()},
        "glossary_aliases": {term: list(aliases) for term, aliases in GLOSSARY_ALIASES.items()},
        "glossary_terms": [
            {"term": item.term, "aliases": list(GLOSSARY_ALIASES.get(item.term, ()))}
            for item in GLOSSARY
        ],
        "games": games_data,
        "ranked": ranked_ids,
        "dogs": dog_ids,
        "favorites": favorite_ids,
        "flat_picks": flat_pick_ids,
        "days": day_map,
        "day_order": day_order,
        "refresh_lines": list(refresh_lines),
        "counts": {
            "games": len(games_data),
            "dogs": len(dog_ids),
            "favorites": len(favorite_ids),
        },
        "best_pick": {
            "game_id": best_pick_game_id,
            "note": best_pick_note,
            "label": (
                next(
                    (
                        f"{item['pick']} {item['pick_spread']}"
                        for item in games_data
                        if item["game_id"] == best_pick_game_id
                    ),
                    best_pick_game_id,
                )
                if best_pick_game_id is not None
                else None
            ),
        },
        "scope": {
            "winners": (
                "This card picks sides against the spread, never "
                "straight-up winners: a team can win the game and still "
                "not cover, so there is no 'who wins' pick here. Ask "
                "about a team for its ATS pick."
            ),
            "injury": (
                "No injury table is published on this board. Availability "
                "reaches the picks two ways: the preseason availability "
                "model baked into the raw pick, and late-week refreshes "
                "once Friday designations land (ask what changed after a "
                "refresh runs). The player-arrests overlay is the only "
                "person-level policy member and is named on the card when "
                "it flips a pick."
            ),
            "weather": (
                "No weather table is published on this board, and the "
                "played Tuesday card never adjusts for weather. "
                "Forecast-weather reads are tracked as prospective "
                "challengers, not wired into any pick."
            ),
        },
        "fallback": {
            "body": (
                "That is not in this week's published card. I answer only "
                "from the picks, the model's measured record, the policy "
                "note, the findings, and football vocabulary -- try asking "
                "about a team, the Best Pick, the record, or what a term "
                "means."
            ),
            "anchors": ("index.html", "model.html", "findings.html"),
        },
        "entries": [
            {"id": entry.entry_id, "body": entry.body, "anchor": entry.anchor} for entry in entries
        ],
        "provenance": {
            "builder": "nfl_ats.board_assistant.build_knowledge",
            "assistant_version": ASSISTANT_VERSION,
            "season": season,
            "week": week,
            "model_id": model_id,
            "game_count": len(games),
            "note": (
                "Retrieval only: every entry body is composed verbatim "
                "from the published card and registry-backed page content."
            ),
        },
    }
    # Deterministic key order so the golden test is stable.
    reloaded: dict[str, Any] = json.loads(json.dumps(payload, sort_keys=True))
    return reloaded


def _source_policy_body(view: SourcePolicyView) -> str:
    """ENG-34: one retrievable sentence for the "sources" corpus entry --
    verbatim off :class:`~nfl_ats.board_content.SourcePolicyView`, never a
    second source-freshness read."""

    if not view.recorded:
        return (
            "Source freshness is not recorded for this forecast -- an older artifact that "
            "predates the ENG-14 policy being persisted to metadata.json."
        )
    if not view.rows:
        return f"Source freshness this week: card state {view.card_state_label}."
    per_source = "; ".join(f"{humanize_identifier(row.source_id)} {row.state}" for row in view.rows)
    return (
        f"Source freshness this week: card state {view.card_state_label} ({per_source}). "
        "Complete means every source was inside its freshness budget, degraded means a "
        "source used its documented fallback, and blocked means a fail-closed source "
        "breached and the card would not have published."
    )


def _tiebreaker_body(view: TiebreakerView) -> str:
    """UI-20(g) extension (2026-09-05): one retrievable sentence for the
    "tiebreaker" corpus entry -- verbatim off :class:`~nfl_ats.board_content
    .TiebreakerView`, itself read straight from ``nfl_ats.publishing``'s
    persisted ``tiebreaker.json``. Never recomputes the guess."""

    if not view.recorded:
        return view.note
    guess_line = f", guess {view.guess_score_text}" if view.guess_score_text else ""
    return (
        f"Tiebreaker (the pool's last game, {view.matchup_text}): market total "
        f"{view.market_total_text}, blended total {view.blended_total_text}, "
        f"implied margin {view.implied_margin_text}{guess_line}. {view.note}"
    )


def build_knowledge_for_board(
    board: BoardContent,
    *,
    findings_page: FindingsPageContent | None = None,
    page: str = "index.html",
) -> dict[str, Any]:
    """Full corpus for the This Week page."""

    headline = board.headline
    record_lines = (
        headline.played_card_value_text + ": " + headline.played_card_caption,
        headline.raw_model_value_text + ": " + headline.raw_model_caption,
        headline.close_grade_value_text + ": " + headline.close_grade_caption,
        headline.selection_caveat_text,
        headline.prospective_scoreboard.headline_text,
    )
    policy_text = board.policy.rich_narrative or board.policy.composition_text
    finding_items = tuple((finding.tag, finding.text) for finding in board.findings)
    watching_items: tuple[tuple[str, str, str, float], ...] = ()
    if findings_page is not None:
        watching_items = tuple(
            (lead.name, lead.description, lead.effect_text, lead.probability_positive)
            for lead in findings_page.watching_leads
        )
    knowledge = build_knowledge(
        refresh_lines=board.refresh_lines,
        page=page,
        season=board.season,
        week=board.week,
        generated_at_text=board.generated_at_text,
        model_id=headline.model_id,
        method_label=headline.model_method_label,
        games=board.games,
        best_pick_game_id=board.best_pick_game_id,
        best_pick_note=board.best_pick_note,
        policy_text=policy_text,
        record_lines=record_lines,
        finding_items=finding_items,
        watching_items=watching_items,
    )
    # ENG-04/UI-18: the lineups.json-derived block feeding the QB-starter,
    # availability, team-injury, and backup-QB intents in answer(). Built
    # from the SAME per-game TeamLineup objects board_content.py already
    # attached to each dive -- this never opens an artifact itself. Absent
    # entirely (empty dict) whenever no game carries both a home and away
    # lineup, which the lineup intents already treat as "not published".
    lineups_by_game = {
        dive.game_id: (dive.home_lineup, dive.away_lineup)
        for dive in board.dives
        if dive.home_lineup is not None and dive.away_lineup is not None
    }
    knowledge["lineups"] = _build_lineup_knowledge(lineups_by_game, reference=board.generated_at)
    # ENG-34: the single "sources" entry answering "were the sources
    # complete this week" -- generic retrieval (_entry_answer/entry()), no
    # dedicated intent handler needed, same pattern as "record"/"policy".
    knowledge["entries"] = [
        *knowledge["entries"],
        {"id": "sources", "body": _source_policy_body(board.source_policy), "anchor": page},
        # UI-20(g) extension: the single "tiebreaker" entry, verbatim off
        # board.tiebreaker (itself read straight from nfl_ats.publishing's
        # persisted tiebreaker.json -- see nfl_ats.board_content
        # ._load_tiebreaker_view). Never recomputed here.
        {"id": "tiebreaker", "body": _tiebreaker_body(board.tiebreaker), "anchor": page},
    ]
    # Re-sort after the merge -- build_knowledge already returns its payload
    # sorted (top-level AND every nested level) via the same round trip, and
    # the golden determinism test checks that property on the WHOLE corpus,
    # so the merged "lineups" block needs the identical treatment.
    resorted: dict[str, Any] = json.loads(json.dumps(knowledge, sort_keys=True))
    return resorted


def _record_lines_for_headline(headline: Any) -> tuple[str, ...]:
    return (
        headline.played_card_value_text + ": " + headline.played_card_caption,
        headline.raw_model_value_text + ": " + headline.raw_model_caption,
        headline.close_grade_value_text + ": " + headline.close_grade_caption,
        headline.selection_caveat_text,
    )


def build_knowledge_for_model(model: ModelPageContent) -> dict[str, Any]:
    """Corpus for The Model page: games (via the ticker), the shared
    headline record, and vocabulary -- no policy or findings, which live
    on their own pages."""

    return build_knowledge(
        page="model.html",
        season=model.ticker_chrome.season,
        week=model.ticker_chrome.week,
        generated_at_text=model.generated_at_text,
        model_id=model.headline.model_id,
        method_label=model.ticker_chrome.model_method_label,
        games=model.ticker_chrome.games,
        best_pick_game_id=model.ticker_chrome.best_pick_game_id,
        best_pick_note=None,
        policy_text=None,
        record_lines=_record_lines_for_headline(model.headline),
        finding_items=(),
        watching_items=(),
    )


def build_knowledge_for_history(history: HistoryPageContent) -> dict[str, Any]:
    """Corpus for the History page: games (via the ticker), timing, and
    vocabulary. Settled rows already render on the page itself; the
    assistant only routes to them."""

    return build_knowledge(
        page="history.html",
        season=history.ticker_chrome.season,
        week=history.ticker_chrome.week,
        generated_at_text=history.generated_at_text,
        model_id=None,
        method_label=history.ticker_chrome.model_method_label,
        games=history.ticker_chrome.games,
        best_pick_game_id=history.ticker_chrome.best_pick_game_id,
        best_pick_note=None,
        policy_text=None,
        record_lines=(),
        finding_items=(),
        watching_items=(),
    )


def build_knowledge_for_findings(findings: FindingsPageContent) -> dict[str, Any]:
    """Corpus for What We've Learned: verdict groups, open leads, and
    the honesty rules -- the page's own prose, retrievable."""

    finding_items = tuple(
        (f"{group.verdict}: {item.question}", f"{item.plain_answer} {item.detail}".strip())
        for group in findings.groups
        for item in group.findings
    )
    watching_items = tuple(
        (lead.name, lead.description, lead.effect_text, lead.probability_positive)
        for lead in findings.watching_leads
    )
    honesty_items = tuple((f"honesty: {rule.title}", rule.body) for rule in findings.honesty_rules)
    return build_knowledge(
        page="findings.html",
        season=findings.ticker_chrome.season,
        week=findings.ticker_chrome.week,
        generated_at_text=findings.generated_at_text,
        model_id=None,
        method_label=findings.ticker_chrome.model_method_label,
        games=findings.ticker_chrome.games,
        best_pick_game_id=findings.ticker_chrome.best_pick_game_id,
        best_pick_note=None,
        policy_text=None,
        record_lines=(),
        finding_items=finding_items + honesty_items,
        watching_items=watching_items,
    )


def _ordinal(n: int) -> str:
    suffix = "th" if 10 <= n % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


#: Liking verbs only count as Best-Pick phrasing beside a pick/game
#: noun -- "teams like KC" is answered by the teams branch long before
#: this is reached, and "what's the line like" must not route here.
_LIKE_NOUNS = frozenset({"pick", "picks", "game", "games", "call", "team", "teams"})


def _best_words(tokens: frozenset[str]) -> bool:
    """Best-pick phrasing: every word of a known alias is present, or a
    liking verb beside a pick/game noun."""

    for alias in SYNONYMS["best_pick"]:
        if all(word in tokens for word in alias.replace("-", " ").split()):
            return True
    return bool(tokens & {"like", "love", "fancy"} and tokens & _LIKE_NOUNS)


def _rankings_answer(parsed: _Parsed, knowledge: Mapping[str, Any]) -> AssistantAnswer | None:
    ranked = [str(game_id) for game_id in knowledge.get("ranked", ())]
    if not ranked:
        return None
    count = len(ranked)
    size = parsed.number if parsed.number else 3
    size = max(1, min(size, count))
    lookup = _game_lookup(knowledge)
    if parsed.least:
        chosen = ranked[-size:][::-1]
        head = "Least confident"
    else:
        chosen = ranked[:size]
        head = "Most confident"
    items = [
        f"{lookup[game_id]['pick']} {lookup[game_id]['pick_spread']} "
        f"({lookup[game_id]['probability_text']})"
        for game_id in chosen
    ]
    tail_id = ranked[-1]
    tail = (
        f"Least confident: {lookup[tail_id]['pick']} "
        f"{lookup[tail_id]['pick_spread']} ({lookup[tail_id]['probability_text']})."
    )
    text = f"{head}: " + ", ".join(items) + f". {tail}"
    return AssistantAnswer(
        topic="rankings",
        text=text,
        anchors=tuple(str(lookup[game_id]["anchor"]) for game_id in chosen[:3]),
    )


def _dog_favorite_answer(
    parsed: _Parsed, knowledge: Mapping[str, Any], key: str, label: str
) -> AssistantAnswer:
    ids = [str(game_id) for game_id in knowledge.get(key, ())]
    total = _count_games(knowledge)
    lookup = _game_lookup(knowledge)
    flat = [str(game_id) for game_id in knowledge.get("flat_picks", ())]
    flat_text = ""
    if key == "favorites" and flat:
        flat_text = (
            " Pick'em (no points either way): "
            + ", ".join(
                f"{lookup[game_id]['away']} at {lookup[game_id]['home']}" for game_id in flat
            )
            + "."
        )
    if not ids:
        return AssistantAnswer(
            topic=key,
            text=(
                f"No {label.lower()} picks this week -- every pick lays "
                f"points across all {total} games.{flat_text}"
            ),
            anchors=("index.html",),
        )
    items = [
        f"{lookup[game_id]['pick']} {lookup[game_id]['pick_spread']} "
        f"({lookup[game_id]['probability_text']})"
        for game_id in ids
    ]
    return AssistantAnswer(
        topic=key,
        text=(f"{label} picks ({len(ids)} of {total}): " + ", ".join(items) + "." + flat_text),
        anchors=tuple(str(lookup[game_id]["anchor"]) for game_id in ids),
    )


def _days_answer(parsed: _Parsed, knowledge: Mapping[str, Any]) -> AssistantAnswer:
    lookup = _game_lookup(knowledge)
    day_map = knowledge.get("days", {})
    parts: list[str] = []
    anchors: list[str] = []
    for day in parsed.days:
        ids = [str(game_id) for game_id in day_map.get(day, ())]
        if not ids:
            parts.append(f"No {day} game this week.")
            continue
        matchups = "; ".join(
            f"{lookup[game_id]['away']} at {lookup[game_id]['home']}" for game_id in ids
        )
        parts.append(f"{day} ({len(ids)}): {matchups}.")
        anchors.extend(str(lookup[game_id]["anchor"]) for game_id in ids)
    parts.append("Ask about a team for its pick.")
    return AssistantAnswer(
        topic="slots", text=" ".join(parts), anchors=tuple(anchors) or ("index.html",)
    )


def _findings_answer(knowledge: Mapping[str, Any]) -> AssistantAnswer | None:
    bodies = []
    for entry in knowledge["entries"]:
        if str(entry["id"]).startswith("finding:"):
            bodies.append(str(entry["body"]))
        if len(bodies) >= 2:
            break
    leads = []
    for entry in knowledge["entries"]:
        if str(entry["id"]).startswith("watching:"):
            leads.append(str(entry["body"]))
        if len(leads) >= 2:
            break
    text = " ".join(bodies + leads)
    if not text:
        return None
    return AssistantAnswer(
        topic="findings",
        text=text + " More on What we've learned.",
        anchors=("findings.html",),
    )


def _refresh_lines_for_teams(knowledge: Mapping[str, Any], teams: tuple[str, ...]) -> list[str]:
    codes = {code.upper() for code in teams}
    return [
        str(line)
        for line in knowledge.get("refresh_lines", ())
        if any(code in str(line) for code in codes)
    ]


def _entry_by_id(knowledge: Mapping[str, Any], entry_id: str) -> Mapping[str, Any] | None:
    entries: list[Mapping[str, Any]] = list(knowledge["entries"])
    for entry in entries:
        if str(entry["id"]) == entry_id:
            return entry
    return None


def _entry_answer(
    knowledge: Mapping[str, Any], entry_id: str, *, topic: str | None = None
) -> AssistantAnswer | None:
    entry = _entry_by_id(knowledge, entry_id)
    if entry is None:
        return None
    return AssistantAnswer(
        topic=topic or str(entry["id"]),
        text=str(entry["body"]),
        anchors=(str(entry["anchor"]),),
    )


def _fallback_answer(knowledge: Mapping[str, Any]) -> AssistantAnswer:
    fallback = knowledge["fallback"]
    anchors = [str(anchor) for anchor in fallback["anchors"]]
    return AssistantAnswer(
        topic="fallback",
        text=str(fallback["body"]),
        anchors=(anchors[0],) if anchors else ("index.html",),
    )


def _game_lookup(knowledge: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    return {str(game["game_id"]): game for game in knowledge.get("games", ())}


def _team_games(code: str, knowledge: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    code = code.upper()
    return [
        game
        for game in knowledge.get("games", ())
        if str(game["away"]).upper() == code or str(game["home"]).upper() == code
    ]


def _short_label(game: Mapping[str, Any]) -> str:
    return f"{game['pick']} {game['pick_spread']} ({game['probability_text']})"


def _rank_of(game_id: str, knowledge: Mapping[str, Any]) -> int:
    return list(knowledge.get("ranked", ())).index(game_id) + 1


def _count_games(knowledge: Mapping[str, Any]) -> int:
    return len(list(knowledge.get("games", ())))


def answer(question: str, knowledge: Mapping[str, Any]) -> AssistantAnswer:
    """Reference engine: deflect set first, then intent parse with
    composed answers, else the fallback. Pure function of the query and
    the corpus — every number returned already occurs in the corpus."""

    query = question.strip()
    if not query:
        return _fallback_answer(knowledge)
    entries = knowledge["entries"]
    for index, rule in enumerate(_deflect_rule_sets()):
        if _deflect_fires(rule, query):
            entry = entries[index]
            assert str(entry["id"]).startswith("deflect:")
            return AssistantAnswer(
                topic=str(entry["id"]),
                text=str(entry["body"]),
                anchors=(str(entry["anchor"]),),
            )
    parsed = _parse(question, knowledge)
    tokens = parsed.tokens

    # ENG-04/UI-18: "is <player> playing/available" -- checked ahead of the
    # team block since a resolved player name is a stronger, more specific
    # signal than any team-code match, and player names never collide with
    # this module's existing team/topic vocabulary. Only fires beside an
    # availability/status cue word (see AVAILABILITY_WORDS), and returns
    # None (falls through) whenever no lineup artifact is published or no
    # player name resolves, so every other route is unaffected.
    lineup_knowledge = knowledge.get("lineups")
    player_resolved = _lineup_player_availability_answer(tokens, lineup_knowledge)
    if player_resolved is not None:
        return player_resolved

    # Glossary: an explained term, or a bare term on its own.
    if parsed.term is not None and (
        tokens & _EXPLAIN_WORDS or len(tokens) == 1 or "what is" in query.lower()
    ):
        for entry in knowledge["entries"]:
            if str(entry["id"]) == f"glossary:{parsed.term}":
                return AssistantAnswer(
                    topic=str(entry["id"]),
                    text=str(entry["body"]),
                    anchors=(str(entry["anchor"]),),
                )

    # Team questions: confidence, schedule, refresh, or the pick itself.
    if parsed.teams:
        lookup = _game_lookup(knowledge)
        if tokens & _CONFIDENCE_WORDS:
            parts = []
            for code in parsed.teams:
                for game in _team_games(code, knowledge):
                    rank = _ordinal(_rank_of(str(game["game_id"]), knowledge))
                    total = _count_games(knowledge)
                    parts.append(f"{_short_label(game)} -- {rank} most confident of {total}.")
            if parts:
                return AssistantAnswer(
                    topic="team_confidence",
                    text=" ".join(parts),
                    anchors=tuple(
                        str(lookup[game["game_id"]]["anchor"])
                        for game in _team_games(parsed.teams[0], knowledge)
                    ),
                )
        if tokens & _SCHEDULE_WORDS:
            parts = []
            anchors = []
            for code in parsed.teams:
                for game in _team_games(code, knowledge):
                    parts.append(
                        f"{game['away']} at {game['home']} plays {game['kickoff']} "
                        f"(pick {game['pick']} {game['pick_spread']})."
                    )
                    anchors.append(str(game["anchor"]))
            if parts:
                return AssistantAnswer(
                    topic="team_schedule", text=" ".join(parts), anchors=tuple(anchors)
                )
        if tokens & _CHANGE_WORDS:
            lines = [str(line) for line in _refresh_lines_for_teams(knowledge, parsed.teams)]
            if lines:
                return AssistantAnswer(
                    topic="refresh", text=" ".join(lines), anchors=("index.html",)
                )
            names = ", ".join(code.upper() for code in parsed.teams)
            return AssistantAnswer(
                topic="refresh",
                text=(
                    f"No refresh recorded for {names} yet -- the Tuesday "
                    "card stands. Ask again after a late-week refresh runs."
                ),
                anchors=("index.html",),
            )
        # ENG-04/UI-18: QB-starter and team-injury questions read ONLY the
        # published lineups.json block (never guessing), applying the
        # existing fail-closed forecast/lineup consistency rule -- see
        # nfl_ats.board_assistant_lineups. Placed ahead of the generic
        # team-pick blurb below so a team+QB or team+injury question gets
        # the lineup-specific answer instead of the plain pick summary.
        if tokens & _LINEUP_QB_WORDS and not (tokens & _LINEUP_BACKUP_WORDS):
            qb_resolved = _lineup_qb_starter_answer(parsed.teams, lineup_knowledge)
            if qb_resolved is not None:
                return qb_resolved
        if tokens & _INJURY_WORDS:
            injury_resolved = _lineup_team_injuries_answer(parsed.teams, lineup_knowledge)
            if injury_resolved is not None:
                return injury_resolved
        game_ids: list[str] = []
        for code in parsed.teams:
            for game in _team_games(code, knowledge):
                if str(game["game_id"]) not in game_ids:
                    game_ids.append(str(game["game_id"]))
        if game_ids:
            if len(game_ids) > 3:
                game_ids = game_ids[:3]
            bodies = [str(lookup[game_id]["why"]) for game_id in game_ids]
            if len(game_ids) > 1:
                ordered = sorted(
                    (lookup[game_id] for game_id in game_ids),
                    key=lambda item: -float(item["probability"]),
                )
                top = ordered[0]
                bodies.append(f"{top['pick']} rates highest at {top['probability_text']}.")
            return AssistantAnswer(
                topic="team_pick",
                text=" ".join(bodies),
                anchors=tuple(str(lookup[game_id]["anchor"]) for game_id in game_ids),
            )

    # Composed lists: rankings, dogs, favorites, day schedules. These
    # run before the Best Pick so a list question with a liking verb
    # ("which underdogs do you like") lists instead of nominating.
    if tokens & _RANK_WORDS:
        resolved = _rankings_answer(parsed, knowledge)
        if resolved is not None:
            return resolved
    if tokens & _DOG_WORDS:
        return _dog_favorite_answer(parsed, knowledge, "dogs", "Underdog")
    if tokens & _FAVORITE_WORDS:
        return _dog_favorite_answer(parsed, knowledge, "favorites", "Favorite")
    # ENG-04/UI-18: "which games have a backup QB" -- a composed list over
    # every published lineup entry, same fail-closed rule as the per-team
    # QB question. No team code required, so this runs alongside the other
    # composed-list intents rather than inside the `if parsed.teams:` block.
    if tokens & _LINEUP_BACKUP_WORDS and tokens & _LINEUP_QB_WORDS:
        return _lineup_backup_qb_games_answer(lineup_knowledge)
    # Day schedules yield to change- and clock-questions ("what
    # changed since Tuesday" is about the refresh, not the weekday).
    if parsed.days and not (
        tokens & _CHANGE_WORDS or tokens & {"when", "deadline", "lock", "locked", "locks"}
    ):
        return _days_answer(parsed, knowledge)

    # Best Pick nomination ("lock" belongs here, not to clock
    # questions: a when/deadline word keeps the query with timing).
    if _best_words(tokens) and not (tokens & {"when", "deadline"}):
        resolved = _entry_answer(knowledge, "best_pick")
        if resolved is not None:
            return resolved

    # Scope answers run ahead of the record: a question naming wins,
    # injuries, or weather is about board coverage, not the track
    # record, even when it shares a word like "edge" or "win".
    scope = knowledge.get("scope", {})
    if tokens & _WINNERS_WORDS and "winners" in scope:
        return AssistantAnswer(
            topic="scope:winners", text=str(scope["winners"]), anchors=("index.html",)
        )
    if tokens & _INJURY_WORDS and "injury" in scope:
        return AssistantAnswer(
            topic="scope:injury", text=str(scope["injury"]), anchors=("index.html",)
        )
    if tokens & _WEATHER_WORDS and "weather" in scope:
        return AssistantAnswer(
            topic="scope:weather", text=str(scope["weather"]), anchors=("index.html",)
        )
    # Single answers from named entries.
    if tokens & _RECORD_WORDS:
        resolved = _entry_answer(knowledge, "record")
        if resolved is not None:
            return resolved
    if tokens & _POLICY_WORDS:
        resolved = _entry_answer(knowledge, "policy")
        if resolved is not None:
            return resolved
    # ENG-34: "were the sources complete this week" -- the single "sources"
    # entry build_knowledge_for_board appends from board.source_policy.
    if tokens & _SOURCE_POLICY_WORDS:
        resolved = _entry_answer(knowledge, "sources")
        if resolved is not None:
            return resolved
    # UI-20(g) extension: "what's the tiebreaker" -- the single
    # "tiebreaker" entry build_knowledge_for_board appends from
    # board.tiebreaker.
    if tokens & _TIEBREAKER_WORDS:
        resolved = _entry_answer(knowledge, "tiebreaker")
        if resolved is not None:
            return resolved
    if tokens & _FINDINGS_WORDS:
        resolved = _findings_answer(knowledge)
        if resolved is not None:
            return resolved
    refresh_entry = _entry_by_id(knowledge, "refresh")
    if tokens & _CHANGE_WORDS and refresh_entry is not None:
        return AssistantAnswer(
            topic="refresh",
            text=str(refresh_entry["body"]),
            anchors=(str(refresh_entry["anchor"]),),
        )
    if tokens & _TIMING_WORDS:
        resolved = _entry_answer(knowledge, "timing")
        if resolved is not None:
            return resolved
    return _fallback_answer(knowledge)


def _js_string_array(items: tuple[str, ...], *, per_line: int = 8) -> str:
    """A wrapped JS string-array literal, aliases sorted for determinism."""

    quoted = [f'"{item}"' for item in sorted(items)]
    lines = [
        ", ".join(quoted[index : index + per_line]) for index in range(0, len(quoted), per_line)
    ]
    return "[" + ",\n      ".join(lines) + "]"


#: Intent vocabulary shared with the JS port, generated into the page
#: so the two engines can never drift. Keys mirror the ``_X_WORDS``
#: sets used by :func:`answer`. ``lineup_qb``/``lineup_backup``/
#: ``lineup_availability`` (ENG-25) are the ``nfl_ats.board_assistant_lineups``
#: word sets that gate the four ENG-04 lineup intents (``injury`` already
#: covers team-injury questions -- see :data:`_INJURY_WORDS`).
_INTENT_WORDS: dict[str, frozenset[str]] = {
    "explain": _EXPLAIN_WORDS,
    "confidence": _CONFIDENCE_WORDS,
    "schedule": _SCHEDULE_WORDS,
    "change": _CHANGE_WORDS,
    "rank": _RANK_WORDS,
    "dog": _DOG_WORDS,
    "favorite": _FAVORITE_WORDS,
    "record": _RECORD_WORDS,
    "policy": _POLICY_WORDS,
    "findings": _FINDINGS_WORDS,
    "timing": _TIMING_WORDS,
    "winners": _WINNERS_WORDS,
    "injury": _INJURY_WORDS,
    "weather": _WEATHER_WORDS,
    "sources": _SOURCE_POLICY_WORDS,
    "tiebreaker": _TIEBREAKER_WORDS,
    "lineup_qb": _LINEUP_QB_WORDS,
    "lineup_backup": _LINEUP_BACKUP_WORDS,
    "lineup_availability": _LINEUP_AVAILABILITY_WORDS,
}


def assistant_script() -> str:
    """Render the inline-JS port of :func:`answer`.

    STOP, the deflect rules, and the intent vocabulary are GENERATED
    from :data:`STOPWORDS`, :func:`_deflect_rule_sets`, and
    :data:`_INTENT_WORDS` -- the JS can never drift from the tested
    Python engine on any of them. The port only ever returns corpus
    strings composed exactly the way Python composes them (verified by
    executing the shipped script against the full question battery). Ports
    the four ENG-04 lineup intents too (``lineupQbStarterAnswer`` and
    siblings), reading the 48h staleness budget from the corpus's own
    ``lineups.stale_budget_hours`` rather than a second hardcoded constant.

    The IIFE exposes a pure, DOM-free ``answerQuestion(question, corpus)``
    via a guarded ``module.exports`` (ENG-25) so
    ``tests/parity/assistant_parity.mjs`` can evaluate the SAME engine
    under Node -- a no-op in the browser, where ``module`` is undefined.
    """

    stop_js = _js_string_array(tuple(STOPWORDS))
    rule_js_lines = []
    for position, rule in enumerate(_deflect_rule_sets()):
        groups = "[" + ", ".join(_js_string_array(tuple(group)) for group in rule) + "]"
        rule_js_lines.append(f"      {{ groups: {groups}, entry: {position} }},")
    rules_js = "[\n" + "\n".join(rule_js_lines) + "\n    ]"
    intent_js = (
        "{\n"
        + ",\n".join(
            f"      {name}: {_js_string_array(tuple(words))}"
            for name, words in sorted(_INTENT_WORDS.items())
        )
        + "\n    }"
    )
    return (
        _ASSISTANT_SCRIPT_TEMPLATE.replace("/*__STOP__*/", stop_js)
        .replace("/*__RULES__*/", rules_js)
        .replace("/*__INTENT__*/", intent_js)
    )


_ASSISTANT_SCRIPT_TEMPLATE = """
<script>
(function () {
  var STOP = /*__STOP__*/;
  function tokens(text) {
    return text.toLowerCase().split(/[^a-z0-9]+/).filter(function (t) {
      return t.length >= 2 && STOP.indexOf(t) === -1;
    });
  }
  var INTENT = /*__INTENT__*/;
  var DAYWORDS = {sunday:"Sunday",monday:"Monday",tuesday:"Tuesday",
    wednesday:"Wednesday",wednesdays:"Wednesday",thursday:"Thursday",
    thursdays:"Thursday",tnf:"Thursday",friday:"Friday",saturday:"Saturday"};
  var DAYORDER = ["Sunday","Monday","Tuesday","Wednesday","Thursday","Friday","Saturday"];
  function teamHits(tokArray, teams) {
    var hits = [];
    Object.keys(teams).sort().forEach(function (code) {
      var found = teams[code].some(function (alias) {
        return alias.split(" ").every(function (w) { return tokArray.indexOf(w) !== -1; });
      });
      if (found) { hits.push(code); }
    });
    return hits;
  }
  function hasAny(tokArray, words) {
    return words.some(function (w) { return tokArray.indexOf(w) !== -1; });
  }
  function bestPhrased(tokArray, synonyms) {
    var aliases = synonyms.best_pick || [];
    var aliasHit = aliases.some(function (alias) {
      return alias.replace(/-/g, " ").split(" ").every(function (w) {
        return tokArray.indexOf(w) !== -1;
      });
    });
    if (aliasHit) return true;
    var nouns = ["pick", "picks", "game", "games", "call", "team", "teams"];
    var liking = ["like", "love", "fancy"].some(function (w) {
      return tokArray.indexOf(w) !== -1;
    });
    return liking && nouns.some(function (w) { return tokArray.indexOf(w) !== -1; });
  }
  // ENG-36: longest-match-first n-gram phrase lookup, mirroring Python's
  // _glossary_term_match exactly -- every term name/alias is re-tokenised
  // with tokens() (same normalisation as the query) so multi-word terms
  // ("cover probability", "closing line", "Best Pick") and multi-word
  // aliases ("against the spread") are reachable, while a single-word
  // term still matches exactly as a plain token-in-array check would.
  function matchGlossaryTerm(toks, glossaryTerms) {
    var candidates = [];
    var longest = 0;
    (glossaryTerms || []).forEach(function (item) {
      [item.term].concat(item.aliases || []).forEach(function (name) {
        var nameToks = tokens(name);
        if (nameToks.length) {
          candidates.push({ key: nameToks.join(" "), len: nameToks.length, term: item.term });
          longest = Math.max(longest, nameToks.length);
        }
      });
    });
    for (var length = Math.min(longest, toks.length); length > 0; length--) {
      for (var start = 0; start <= toks.length - length; start++) {
        var key = toks.slice(start, start + length).join(" ");
        var found = candidates.filter(function (c) {
          return c.len === length && c.key === key;
        })[0];
        if (found) { return found.term; }
      }
    }
    return null;
  }
  function parse(q, corpus) {
    var toks = tokens(q);
    var teamMap = {};
    (corpus.teams || []).forEach(function (t) { teamMap[t.code] = t.aliases; });
    var teams = teamHits(toks, teamMap);
    var rawCodes = q.match(/\\b[A-Z]{2,3}\\b/g) || [];
    Object.keys(teamMap).forEach(function (code) {
      if (rawCodes.indexOf(code.toUpperCase()) !== -1 && teams.indexOf(code) === -1) {
        teams.push(code);
      }
    });
    teams.sort();
    var days = [];
    DAYORDER.forEach(function (day) {
      var hit = Object.keys(DAYWORDS).some(function (w) {
        return DAYWORDS[w] === day && toks.indexOf(w) !== -1;
      });
      if (hit) { days.push(day); }
    });
    if (toks.indexOf("weekend") !== -1) {
      ["Saturday", "Sunday"].forEach(function (day) {
        if (days.indexOf(day) === -1) { days.push(day); }
      });
    }
    var term = matchGlossaryTerm(toks, corpus.glossary_terms);
    var rawWords = q.toLowerCase().replace(/[^a-z0-9]+/g, " ").split(" ").filter(Boolean);
    var number = null;
    rawWords.forEach(function (t) {
      if (number === null && /^[0-9]+$/.test(t)) { number = parseInt(t, 10); }
    });
    var least = hasAny(toks, ["least", "bottom", "worst", "weakest"]);
    return {tokens: toks, teams: teams, days: days, term: term, number: number, least: least};
  }
  function ordinal(n) {
    var suffix = "th";
    if (n % 100 < 10 || n % 100 > 20) {
      if (n % 10 === 1) suffix = "st";
      else if (n % 10 === 2) suffix = "nd";
      else if (n % 10 === 3) suffix = "rd";
    }
    return n + suffix;
  }
  function gameById(corpus) {
    var map = {};
    (corpus.games || []).forEach(function (g) { map[g.game_id] = g; });
    return map;
  }
  function shortLabel(g) {
    return g.pick + " " + g.pick_spread + " (" + g.probability_text + ")";
  }
  function teamGames(code, corpus) {
    code = code.toUpperCase();
    return (corpus.games || []).filter(function (g) {
      return g.away.toUpperCase() === code || g.home.toUpperCase() === code;
    });
  }
  function entryById(corpus, id) {
    var found = null;
    (corpus.entries || []).forEach(function (e) {
      if (e.id === id) { found = e; }
    });
    return found;
  }
  function asAnswer(topic, text, anchors) {
    return {topic: topic, text: text, anchors: anchors};
  }
  function rankingsAnswer(parsed, corpus) {
    var ranked = corpus.ranked || [];
    if (!ranked.length) return null;
    var count = ranked.length;
    var size = parsed.number || 3;
    size = Math.max(1, Math.min(size, count));
    var byId = gameById(corpus);
    var chosen, head;
    if (parsed.least) {
      chosen = ranked.slice(-size).reverse();
      head = "Least confident";
    } else {
      chosen = ranked.slice(0, size);
      head = "Most confident";
    }
    var items = chosen.map(function (id) {
      var g = byId[id];
      return g.pick + " " + g.pick_spread + " (" + g.probability_text + ")";
    });
    var tailId = ranked[ranked.length - 1];
    var tail = byId[tailId];
    var text = head + ": " + items.join(", ") + ". Least confident: " +
      tail.pick + " " + tail.pick_spread + " (" + tail.probability_text + ").";
    return asAnswer("rankings", text, chosen.slice(0, 3).map(function (id) {
      return byId[id].anchor;
    }));
  }
  function listAnswer(parsed, corpus, key, label) {
    var ids = corpus[key] || [];
    var total = (corpus.games || []).length;
    var byId = gameById(corpus);
    var flat = "";
    if (key === "favorites" && (corpus.flat_picks || []).length) {
      flat = " Pick'em (no points either way): " + corpus.flat_picks.map(function (id) {
        return byId[id].away + " at " + byId[id].home;
      }).join(", ") + ".";
    }
    if (!ids.length) {
      return asAnswer(key, "No " + label.toLowerCase() + " picks this week -- every pick lays " +
        "points across all " + total + " games." + flat, ["index.html"]);
    }
    var items = ids.map(function (id) {
      var g = byId[id];
      return g.pick + " " + g.pick_spread + " (" + g.probability_text + ")";
    });
    return asAnswer(key, label + " picks (" + ids.length + " of " + total + "): " +
      items.join(", ") + "." + flat,
      ids.map(function (id) { return byId[id].anchor; }));
  }
  function daysAnswer(parsed, corpus) {
    var byId = gameById(corpus);
    var parts = [];
    var anchors = [];
    parsed.days.forEach(function (day) {
      var ids = (corpus.days || {})[day] || [];
      if (!ids.length) {
        parts.push("No " + day + " game this week.");
        return;
      }
      var matchups = ids.map(function (id) {
        return byId[id].away + " at " + byId[id].home;
      }).join("; ");
      parts.push(day + " (" + ids.length + "): " + matchups + ".");
      ids.forEach(function (id) { anchors.push(byId[id].anchor); });
    });
    parts.push("Ask about a team for its pick.");
    return asAnswer("slots", parts.join(" "), anchors.length ? anchors : ["index.html"]);
  }
  function findingsAnswer(corpus) {
    var bodies = [];
    (corpus.entries || []).forEach(function (e) {
      if (bodies.length < 2 && e.id.indexOf("finding:") === 0) { bodies.push(e.body); }
    });
    (corpus.entries || []).forEach(function (e) {
      if (bodies.length < 4 && e.id.indexOf("watching:") === 0) { bodies.push(e.body); }
    });
    if (!bodies.length) return null;
    return asAnswer("findings", bodies.join(" ") + " More on What we've learned.",
      ["findings.html"]);
  }
  function gameWhy(g, corpus) {
    var bestNote = null;
    if (g.best) {
      if (corpus.best_pick && corpus.best_pick.note) { bestNote = corpus.best_pick.note; }
    }
    var parts = [g.away + " at " + g.home + " (" + g.kickoff + "): pick " +
      g.pick + " " + g.pick_spread + " -- " + g.probability_text +
      " cover probability (" + g.confidence + "). Line: " + g.spread + "."];
    if (g.best) {
      parts.push(bestNote ? "Best Pick of the week. " + bestNote : "Best Pick of the week.");
    }
    if (g.flips && g.flips.length) {
      parts.push("Policy flip: " + g.flips.join(" + ") + ".");
    }
    if (g.final) {
      parts.push("Final: " + g.final + "." + (g.final_score ? " " + g.final_score + "." : ""));
    }
    return parts.join(" ");
  }
  function refreshLinesForTeams(corpus, teams) {
    var codes = teams.map(function (c) { return c.toUpperCase(); });
    return (corpus.refresh_lines || []).filter(function (line) {
      return codes.some(function (code) { return String(line).indexOf(code) !== -1; });
    });
  }
  // ENG-25: lineup intents (ENG-04/UI-18) ported from
  // nfl_ats.board_assistant_lineups -- mirrors that module's fail-closed
  // consistency refusal, stale/absent fallbacks, and source-time anchor
  // text verbatim; the 48h staleness budget is read from
  // corpus.lineups.stale_budget_hours (never a second hardcoded copy).
  function lineupAnchorText(entry) {
    return "as of " + (entry.as_of || "an unrecorded time") + " from " +
      (entry.source || "an unrecorded source");
  }
  function lineupBudgetOf(lineupKnowledge) {
    return (lineupKnowledge && lineupKnowledge.stale_budget_hours != null) ?
      lineupKnowledge.stale_budget_hours : 48.0;
  }
  function lineupStaleText(team, anchor, budget) {
    return "The newest projected-lineup snapshot for " + team + " is " + anchor +
      ", which is older than the " + Math.round(budget) + "-hour freshness budget this " +
      "assistant enforces -- I won't guess a current starter or availability from a stale " +
      "snapshot. Check the live This Week page for a fresher one.";
  }
  function lineupUnpublishedText(team) {
    return "No projected-lineup artifact is published for " + team + " this week -- the " +
      "This Week page's lineup panel only appears once scripts/build_week_lineups.py has run " +
      "for this game (docs/projected_lineups.md); I won't guess.";
  }
  function lineupTeamLookup(lineupKnowledge, teamCode) {
    var code = teamCode.toUpperCase();
    var games = (lineupKnowledge && lineupKnowledge.games) || {};
    var found = null;
    Object.keys(games).forEach(function (gameId) {
      if (found) return;
      var sides = games[gameId];
      ["home", "away"].forEach(function (side) {
        if (found) return;
        var entry = sides[side];
        if (entry && String(entry.team || "").toUpperCase() === code) {
          found = [gameId, entry];
        }
      });
    });
    return found;
  }
  function lineupQbStarterAnswer(teamsList, lineupKnowledge) {
    if (!teamsList.length) return null;
    var lk = lineupKnowledge || {games: {}};
    var parts = [];
    var anchors = [];
    teamsList.forEach(function (code) {
      var found = lineupTeamLookup(lk, code);
      if (!found) { parts.push(lineupUnpublishedText(code.toUpperCase())); return; }
      var gameId = found[0], entry = found[1];
      anchors.push("index.html#" + gameId);
      if (entry.stale) {
        parts.push(lineupStaleText(entry.team, lineupAnchorText(entry), lineupBudgetOf(lk)));
        return;
      }
      var anchor = lineupAnchorText(entry);
      if (entry.note) {
        var modelName = entry.model_qb_name || "a QB not on the current roster snapshot";
        var currentName = entry.current_qb_name || "no QB listed on the current snapshot";
        parts.push(entry.team + ": the published forecast assumed " + modelName + " at QB, " +
          "but the current depth-chart snapshot (" + anchor + ") lists " + currentName +
          " at QB1 instead -- I can't state a single starter until the forecast is " +
          "regenerated from this snapshot.");
      } else {
        var name = entry.current_qb_name || "no QB listed on the current snapshot";
        parts.push(entry.team + " starting QB: " + name + " (" + anchor + ").");
      }
    });
    if (!parts.length) return null;
    return asAnswer("lineup:qb", parts.join(" "), anchors.length ? anchors : ["index.html"]);
  }
  function lineupTeamInjuriesAnswer(teamsList, lineupKnowledge) {
    if (!teamsList.length) return null;
    var lk = lineupKnowledge || {games: {}};
    var parts = [];
    var anchors = [];
    teamsList.forEach(function (code) {
      var found = lineupTeamLookup(lk, code);
      if (!found) { parts.push(lineupUnpublishedText(code.toUpperCase())); return; }
      var gameId = found[0], entry = found[1];
      anchors.push("index.html#" + gameId);
      if (entry.stale) {
        parts.push(lineupStaleText(entry.team, lineupAnchorText(entry), lineupBudgetOf(lk)));
        return;
      }
      var anchor = lineupAnchorText(entry);
      var flagged = (entry.players || []).filter(function (p) { return p.injury_status; });
      if (flagged.length) {
        var listing = flagged.map(function (p) {
          return p.name + " (" + p.injury_status + ")";
        }).join("; ");
        parts.push(entry.team + " injury notes (" + anchor + "): " + listing + ".");
      } else {
        var status = entry.injury_status || "unavailable";
        parts.push(entry.team + ": no per-player injury designation in the lineup snapshot (" +
          anchor + "); team-level injury feed status: " + status + ".");
      }
    });
    if (!parts.length) return null;
    return asAnswer("lineup:injuries", parts.join(" "), anchors.length ? anchors : ["index.html"]);
  }
  function lineupDedupePlayers(players) {
    var seen = {};
    var order = [];
    players.forEach(function (p) {
      var key = p.gsis_id || (p.name + "|" + p.team);
      if (!(key in seen)) { seen[key] = p; order.push(key); }
    });
    return order.map(function (key) { return seen[key]; });
  }
  function lineupResolvePlayers(toks, players) {
    var SUFFIXES = ["jr", "sr", "ii", "iii", "iv"];
    function nameTokens(name) {
      return tokens(String(name)).filter(function (t) { return SUFFIXES.indexOf(t) === -1; });
    }
    var exact = players.filter(function (p) {
      var nt = nameTokens(p.name);
      return nt.length > 0 && nt.every(function (t) { return toks.indexOf(t) !== -1; });
    });
    if (exact.length) return exact;
    return players.filter(function (p) {
      var nt = nameTokens(p.name);
      return nt.length > 0 && toks.indexOf(nt[nt.length - 1]) !== -1;
    });
  }
  function lineupPlayerAvailabilityAnswer(toks, lineupKnowledge) {
    if (!lineupKnowledge || !hasAny(toks, INTENT.lineup_availability)) return null;
    var players = lineupKnowledge.players || [];
    if (!players.length) return null;
    var matches = lineupResolvePlayers(toks, players);
    if (!matches.length) return null;
    var distinct = lineupDedupePlayers(matches);
    if (distinct.length > 1) {
      var names = distinct.map(function (p) { return p.name + " (" + p.team + ")"; }).sort();
      return asAnswer("lineup:availability",
        "More than one player in this week's published lineups matches that name: " +
        names.join(", ") + ". Ask again naming the team.", ["index.html"]);
    }
    var parts = [];
    var anchors = [];
    distinct.forEach(function (player) {
      anchors.push("index.html#" + player.game_id);
      if (player.stale) {
        parts.push(lineupStaleText(
          player.team, lineupAnchorText(player), lineupBudgetOf(lineupKnowledge)
        ));
        return;
      }
      var anchor = lineupAnchorText(player);
      var probability = player.play_probability;
      // UI-20-AB (2026-09-05): every player's percentage is now a real
      // per-player, per-game forecast from the availability model (depth
      // chart + injury report + recent snaps), designated or not, so it is
      // always quoted when present (mirrors
      // nfl_ats.board_assistant_lineups.player_availability_answer
      // exactly; retires the 2026-09-05 "no designation" stopgap).
      var probabilityText;
      if (probability === null || probability === undefined) {
        probabilityText = "not published";
      } else {
        probabilityText = Math.round(probability * 100) + "% chance of taking the field";
      }
      var injury = player.injury_status || "no report";
      var roleNote = player.model_role === "base_model" ?
        "the model's starter" : "context only -- not the model's scored player";
      parts.push(player.name + " (" + player.team + ", " + player.slot + "): " +
        probabilityText + ", injury status " + injury + ", " + roleNote + " (" + anchor + ").");
    });
    return asAnswer("lineup:availability", parts.join(" "), anchors);
  }
  function lineupBackupQbGamesAnswer(lineupKnowledge) {
    if (!lineupKnowledge || !Object.keys(lineupKnowledge.games || {}).length) {
      return asAnswer("lineup:backup_qb",
        "No projected-lineup artifact is published this week, so I can't compare current " +
        "depth-chart starters to the forecast.", ["index.html"]);
    }
    var hits = [];
    var anchors = [];
    var staleAny = false;
    Object.keys(lineupKnowledge.games).forEach(function (gameId) {
      var sides = lineupKnowledge.games[gameId];
      ["home", "away"].forEach(function (side) {
        var entry = sides[side];
        if (!entry) return;
        if (entry.stale) { staleAny = true; return; }
        if (entry.note) {
          var modelName = entry.model_qb_name || "a QB not on the current roster snapshot";
          var currentName = entry.current_qb_name || "no QB listed";
          hits.push(entry.team + " (" + lineupAnchorText(entry) + "): forecast assumed " +
            modelName + ", current snapshot lists " + currentName);
          anchors.push("index.html#" + gameId);
        }
      });
    });
    var tail = staleAny ?
      " (at least one team's snapshot is stale and was excluded from this answer, never guessed)" :
      "";
    if (!hits.length) {
      return asAnswer("lineup:backup_qb",
        "No team's current depth-chart QB1 disagrees with its forecast-assumed QB in the " +
        "published lineup snapshot" + tail + ".", ["index.html"]);
    }
    return asAnswer("lineup:backup_qb",
      "Depth chart lists a different QB than the forecast assumed for: " + hits.join("; ") +
      tail + "." + ' ("Backup QB" here means the current depth chart disagrees with the ' +
      "forecast's assumed starter, not merely that a backup is on the roster.)",
      anchors);
  }
  function answerParsed(parsed, q, corpus) {
    var toks = parsed.tokens;
    var byId = gameById(corpus);
    // ENG-25: "is <player> playing/available" -- checked ahead of the
    // glossary/team blocks, same placement as _lineup_player_availability_answer
    // in nfl_ats.board_assistant.answer.
    var lineupKnowledge = corpus.lineups || null;
    var playerResolved = lineupPlayerAvailabilityAnswer(toks, lineupKnowledge);
    if (playerResolved) return playerResolved;
    function entry(id) {
      var found = entryById(corpus, id);
      return found ? asAnswer(found.id, found.body, [found.anchor]) : null;
    }
    var glossaryHit = null;
    var uniqueCount = toks.filter(function (t, i) { return toks.indexOf(t) === i; }).length;
    if (parsed.term && (hasAny(toks, INTENT.explain) || uniqueCount === 1 ||
        q.toLowerCase().indexOf("what is") !== -1)) {
      (corpus.entries || []).forEach(function (e) {
        if (e.id === "glossary:" + parsed.term) {
          glossaryHit = asAnswer(e.id, e.body, [e.anchor]);
        }
      });
      if (glossaryHit) return glossaryHit;
    }
    if (parsed.teams.length) {
      if (hasAny(toks, INTENT.confidence)) {
        var confParts = [];
        parsed.teams.forEach(function (code) {
          teamGames(code, corpus).forEach(function (g) {
            var rank = corpus.ranked.indexOf(g.game_id) + 1;
            confParts.push(shortLabel(g) + " -- " + ordinal(rank) +
              " most confident of " + corpus.games.length + ".");
          });
        });
        if (confParts.length) {
          return asAnswer("team_confidence", confParts.join(" "),
            teamGames(parsed.teams[0], corpus).map(function (g) { return g.anchor; }));
        }
      }
      if (hasAny(toks, INTENT.schedule)) {
        var schedParts = [];
        var schedAnchors = [];
        parsed.teams.forEach(function (code) {
          teamGames(code, corpus).forEach(function (g) {
            schedParts.push(g.away + " at " + g.home + " plays " + g.kickoff +
              " (pick " + g.pick + " " + g.pick_spread + ").");
            schedAnchors.push(g.anchor);
          });
        });
        if (schedParts.length) {
          return asAnswer("team_schedule", schedParts.join(" "), schedAnchors);
        }
      }
      if (hasAny(toks, INTENT.change)) {
        var lines = refreshLinesForTeams(corpus, parsed.teams);
        if (lines.length) {
          return asAnswer("refresh", lines.join(" "), ["index.html"]);
        }
        var names = parsed.teams.map(function (c) { return c.toUpperCase(); }).join(", ");
        return asAnswer("refresh", "No refresh recorded for " + names + " yet -- the Tuesday " +
          "card stands. Ask again after a late-week refresh runs.", ["index.html"]);
      }
      // ENG-25: QB-starter and team-injury lineup intents, ahead of the
      // generic team-pick blurb -- same placement/precedence as
      // nfl_ats.board_assistant.answer.
      if (hasAny(toks, INTENT.lineup_qb) && !hasAny(toks, INTENT.lineup_backup)) {
        var qbResolved = lineupQbStarterAnswer(parsed.teams, lineupKnowledge);
        if (qbResolved) return qbResolved;
      }
      if (hasAny(toks, INTENT.injury)) {
        var injuryResolved = lineupTeamInjuriesAnswer(parsed.teams, lineupKnowledge);
        if (injuryResolved) return injuryResolved;
      }
      var gameIds = [];
      parsed.teams.forEach(function (code) {
        teamGames(code, corpus).forEach(function (g) {
          if (gameIds.indexOf(g.game_id) === -1) { gameIds.push(g.game_id); }
        });
      });
      if (gameIds.length) {
        if (gameIds.length > 3) { gameIds = gameIds.slice(0, 3); }
        var bodies = gameIds.map(function (id) { return gameWhy(byId[id], corpus); });
        if (gameIds.length > 1) {
          var ordered = gameIds.map(function (id) { return byId[id]; }).sort(function (a, b) {
            return b.probability - a.probability;
          });
          bodies.push(ordered[0].pick + " rates highest at " +
            ordered[0].probability_text + ".");
        }
        return asAnswer("team_pick", bodies.join(" "), gameIds.map(function (id) {
          return byId[id].anchor;
        }));
      }
    }
    if (hasAny(toks, INTENT.rank)) {
      var ranked = rankingsAnswer(parsed, corpus);
      if (ranked) return ranked;
    }
    if (hasAny(toks, INTENT.dog)) return listAnswer(parsed, corpus, "dogs", "Underdog");
    if (hasAny(toks, INTENT.favorite)) {
      return listAnswer(parsed, corpus, "favorites", "Favorite");
    }
    // ENG-25: "which games have a backup QB" -- no team code required, same
    // placement as nfl_ats.board_assistant.answer (alongside the other
    // composed-list intents, ahead of the day-schedule branch).
    if (hasAny(toks, INTENT.lineup_backup) && hasAny(toks, INTENT.lineup_qb)) {
      return lineupBackupQbGamesAnswer(lineupKnowledge);
    }
    var clockGuard = ["when", "deadline", "lock", "locked", "locks"].some(function (w) {
      return toks.indexOf(w) !== -1;
    });
    if (parsed.days.length && !hasAny(toks, INTENT.change) && !clockGuard) {
      return daysAnswer(parsed, corpus);
    }
    var noClock = ["when", "deadline"].some(function (w) { return toks.indexOf(w) !== -1; });
    if (bestPhrased(toks, corpus.synonyms || {}) && !noClock) {
      var best = entry("best_pick");
      if (best) return best;
    }
    if (hasAny(toks, INTENT.winners) && corpus.scope && corpus.scope.winners) {
      return asAnswer("scope:winners", corpus.scope.winners, ["index.html"]);
    }
    if (hasAny(toks, INTENT.injury) && corpus.scope && corpus.scope.injury) {
      return asAnswer("scope:injury", corpus.scope.injury, ["index.html"]);
    }
    if (hasAny(toks, INTENT.weather) && corpus.scope && corpus.scope.weather) {
      return asAnswer("scope:weather", corpus.scope.weather, ["index.html"]);
    }
    if (hasAny(toks, INTENT.record)) {
      var record = entry("record");
      if (record) return record;
    }
    if (hasAny(toks, INTENT.policy)) {
      var policy = entry("policy");
      if (policy) return policy;
    }
    // ENG-34: mirrors nfl_ats.board_assistant.answer's _SOURCE_POLICY_WORDS
    // branch -- the single "sources" entry built from board.source_policy.
    if (hasAny(toks, INTENT.sources)) {
      var sources = entry("sources");
      if (sources) return sources;
    }
    // UI-20(g) extension: mirrors nfl_ats.board_assistant.answer's
    // _TIEBREAKER_WORDS branch -- the single "tiebreaker" entry built from
    // board.tiebreaker.
    if (hasAny(toks, INTENT.tiebreaker)) {
      var tiebreaker = entry("tiebreaker");
      if (tiebreaker) return tiebreaker;
    }
    if (hasAny(toks, INTENT.findings)) {
      var found = findingsAnswer(corpus);
      if (found) return found;
    }
    var refreshEntry = entryById(corpus, "refresh");
    if (hasAny(toks, INTENT.change) && refreshEntry) {
      return asAnswer("refresh", refreshEntry.body, [refreshEntry.anchor]);
    }
    if (hasAny(toks, INTENT.timing)) {
      var timing = entry("timing");
      if (timing) return timing;
    }
    return asAnswer("fallback", corpus.fallback.body, [corpus.fallback.anchors[0]]);
  }
  // Deflect rules mirror _deflect_rule_sets: AND of OR-groups. Hoisted to
  // module scope (ENG-25) so the pure answerQuestion() below -- the Node
  // parity harness's entry point -- can use the same table without a DOM
  // element to read it from.
  var RULES = /*__RULES__*/;
  function answerQuestion(question, corpus) {
    var q = String(question || '').trim();
    if (!q) {
      var emptyFallback = corpus.fallback;
      return {topic: 'fallback', text: emptyFallback.body, anchors: [emptyFallback.anchors[0]]};
    }
    var flat = q.toLowerCase().replace(/[^a-z0-9]+/g, ' ').replace(/ +/g, ' ').trim();
    var words = flat ? flat.split(' ') : [];
    var padded = ' ' + flat + ' ';
    var fired = null;
    RULES.forEach(function (rule) {
      if (fired) return;
      var ok = rule.groups.every(function (group) {
        return group.some(function (alias) {
          if (alias.indexOf(' ') !== -1) { return padded.indexOf(' ' + alias + ' ') !== -1; }
          return words.indexOf(alias) !== -1;
        });
      });
      if (ok) {
        var hit = corpus.entries[rule.entry];
        fired = {topic: hit.id, text: hit.body, anchors: [hit.anchor]};
      }
    });
    if (fired) return fired;
    var parsed = parse(q, corpus);
    var result = answerParsed(parsed, q, corpus);
    if (result) return result;
    var fb = corpus.fallback;
    return {topic: 'fallback', text: fb.body, anchors: [fb.anchors[0]]};
  }
  if (typeof document !== 'undefined') {
    document.querySelectorAll('.assistant-box').forEach(function (box) {
      var dataEl = box.querySelector('.assistant-data');
      var form = box.querySelector('.assistant-form');
      var input = box.querySelector('.assistant-input');
      var log = box.querySelector('.assistant-log');
      if (!dataEl || !form || !input || !log) return;
      var corpus = JSON.parse(dataEl.textContent);
      var teams = {};
      (corpus.teams || []).forEach(function (t) { teams[t.code] = t.aliases; });
      function addEntry(kind, topic, text, anchor) {
        var item = document.createElement('div');
        item.className = 'assistant-' + kind;
        var p = document.createElement('p');
        p.textContent = text;
        item.appendChild(p);
        if (anchor) {
          var a = document.createElement('a');
          a.href = anchor;
          a.textContent = 'Open on the board';
          item.appendChild(a);
        }
        log.appendChild(item);
        log.scrollTop = log.scrollHeight;
      }
      form.addEventListener('submit', function (evt) {
        evt.preventDefault();
        var q = input.value.trim();
        if (!q) return;
        addEntry('q', 'question', q, null);
        var result = answerQuestion(q, corpus);
        addEntry('a', result.topic, result.text,
          result.anchors && result.anchors.length ? result.anchors[0] : null);
        input.value = '';
        input.focus();
      });
    });
  }
  // ENG-25: expose the pure engine so the Node parity harness
  // (tests/parity/assistant_parity.mjs) can evaluate it without a DOM.
  // Guarded so this is a no-op in the browser, where `module` is undefined.
  if (typeof module !== 'undefined' && module.exports) {
    module.exports = { answer: answerQuestion };
  }
})();
</script>
"""


def _corpus_json(corpus: Mapping[str, Any]) -> str:
    """Corpus JSON safe to inline: ``<``, ``>``, and ``&`` ride as
    unicode escapes (plain ``json.loads`` decodes them back), so no
    payload text can break out of its ``<script
    type="application/json">`` block or read as markup."""

    return (
        json.dumps(corpus, sort_keys=True)
        .replace("<", "\\u003c")
        .replace(">", "\\u003e")
        .replace("&", "\\u0026")
    )


def assistant_section(corpus: Mapping[str, Any]) -> str:
    """The chat panel for one page: ``<details>``-native (keyboard
    operable, mobile-collapsed by default), honest without JS (topic
    links to the existing anchor-linked sections), answering from the
    embedded corpus with JS. No animation anywhere, so
    ``prefers-reduced-motion`` is honored by having nothing to reduce."""

    teams = [
        {"code": code, "aliases": list(aliases)} for code, aliases in sorted(_TEAM_SYNONYMS.items())
    ]
    with_teams = dict(corpus)
    with_teams["teams"] = teams
    blob = _corpus_json(with_teams)
    topics = [
        ("This week's picks", "index.html"),
        ("Best Pick", "index.html"),
        ("Model record", "model.html"),
        ("What we've learned", "findings.html"),
        ("History", "history.html"),
    ]
    links = " ".join(f'<a href="{escape(href)}">{escape(label)}</a>' for label, href in topics)
    # ENG-05 accessibility contract: the golden-question suite's no-JS check
    # requires the <noscript> fallback to SAY it needs JavaScript, not just
    # list links -- the rest of the page (including its own picks table)
    # already renders unconditionally, so only the assistant itself needs
    # the callout.
    noscript_note = (
        "This assistant needs JavaScript to answer questions live -- the "
        "rest of this page (including its picks table) works without it. "
        "Jump straight to a section:"
    )
    return (
        '<section class="assistant" aria-label="Board assistant">'
        '<details class="assistant-box">'
        '<summary><span class="micro">Board assistant</span> '
        'Ask the board <span class="assistant-hint">retrieval only -- '
        "answers come from this page's published card</span></summary>"
        f'<script type="application/json" class="assistant-data">{blob}</script>'
        '<form class="assistant-form">'
        '<label class="assistant-label" for="assistant-q">Question</label>'
        '<input class="assistant-input" id="assistant-q" name="q" type="text" '
        'autocomplete="off" placeholder="Why MIA +3.5? What is the record?">'
        '<button class="assistant-ask" type="submit">Ask</button>'
        "</form>"
        '<div class="assistant-log" aria-live="polite"></div>'
        f"<noscript><p>{escape(noscript_note)}</p>"
        f'<div class="assistant-topics">{links}</div></noscript>'
        "</details>"
        "</section>"
    )


__all__ = [
    "ASSISTANT_VERSION",
    "GLOSSARY",
    "STOPWORDS",
    "SYNONYMS",
    "AssistantAnswer",
    "answer",
    "assistant_script",
    "assistant_section",
    "build_knowledge",
    "build_knowledge_for_board",
    "build_knowledge_for_findings",
    "build_knowledge_for_history",
    "build_knowledge_for_model",
]
