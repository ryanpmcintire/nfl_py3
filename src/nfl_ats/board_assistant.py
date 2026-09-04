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
from collections.abc import Mapping
from dataclasses import dataclass
from html import escape
from typing import Any

from nfl_ats.board_content import BoardContent, GameRow
from nfl_ats.board_site_content import (
    FindingsPageContent,
    HistoryPageContent,
    ModelPageContent,
)
from nfl_ats.market_data import NFL_TEAM_NAMES

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
    "best_pick": ("best pick", "best-pick", "bestpick", "best bet", "star", "pick of the week"),
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
    ),
}


@dataclass(frozen=True)
class _Entry:
    """One retrievable answer: precomputed body, match keywords, anchor."""

    entry_id: str
    keywords: tuple[str, ...]
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


def _expand(token: str) -> frozenset[str]:
    """A query token plus its whole synonym group (team or topic)."""

    group: set[str] = {token}
    for aliases in _TEAM_SYNONYMS.values():
        if token in aliases:
            group.update(aliases)
    for canonical, aliases in SYNONYMS.items():
        if token == canonical or token in aliases:
            group.add(canonical)
            group.update(aliases)
    return frozenset(group)


def _score(query_tokens: frozenset[str], keywords: tuple[str, ...]) -> tuple[int, int, int, int]:
    """Shared rank rule (port contract), most-specific first: per
    distinct query token, a raw touch outranks any synonym-expansion
    touch. The raw token matches either direction (a three-plus
    character keyword also matches inside a longer token, so
    ``seahawks`` finds the ``sea`` keyword); synonym EXPANSIONS match
    forward only -- otherwise ``lock`` expands to ``timing``, which
    happens to contain the ``min`` keyword, and every clock question
    routes to Minnesota. Returns ``(raw_hits, exact_hits,
    expanded_hits, raw_best_length)``: whole-token equality outranks a
    longer partial touch, so ``cover`` reaches the ``cover`` definition
    rather than ``cover probability``, while an expansion hit on a long
    finding tag can never outrank a team's own keyword."""

    raw_hits = 0
    exact_hits = 0
    expanded_hits = 0
    raw_best_length = 0
    lowered = [keyword.lower() for keyword in keywords]
    for token in query_tokens:
        raw_lengths = [
            len(keyword)
            for keyword in lowered
            if token in keyword or (len(keyword) >= 3 and keyword in token)
        ]
        if raw_lengths:
            raw_hits += 1
            exact_hits += sum(1 for keyword in lowered if token == keyword)
            raw_best_length = max(raw_best_length, max(raw_lengths))
            continue
        if any(
            variant in keyword
            for variant in _expand(token)
            if variant != token
            for keyword in lowered
        ):
            expanded_hits += 1
    return raw_hits, exact_hits, expanded_hits, raw_best_length


def _deflect_entries(season: int, week: int) -> tuple[_Entry, ...]:
    """The must-deflect set, in the exact order of
    :func:`_deflect_rule_sets` (rule *i* fires entry *i*): each rule is
    an AND of OR-groups, so ``best bet`` still routes to the Best Pick
    while ``should I bet`` deflects."""

    wager = _Entry(
        entry_id="deflect:wager",
        keywords=("wager:should bet",),
        body=(
            "This board never advises wagers: every pick shown is a "
            "simulated, paper pick for evaluating a forecasting model, "
            "and a small historical edge is not proof of a profitable "
            "one. The card itself is on This week."
        ),
        anchor="index.html",
    )
    future = _Entry(
        entry_id="deflect:future",
        keywords=("future:week ahead",),
        body=(
            f"This published card covers {season} Week {week} only. "
            "Future weeks have no published picks, and pending games "
            "have no outcomes yet -- see History once results settle."
        ),
        anchor="history.html",
    )
    ownership = _Entry(
        entry_id="deflect:ownership",
        keywords=("ownership:consensus data",),
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
        keywords=("score:exact final",),
        body=(
            "The published card covers sides against the spread only, "
            "not exact final scores. Tiebreak guesses are a separate "
            "weekly exercise, not part of this card."
        ),
        anchor="index.html",
    )
    return (wager, wager, future, ownership, ownership, score, score)


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
        (frozenset({"tiebreaker", "exact score", "final score"}),),
        (
            frozenset({"score", "scoreline"}),
            frozenset({"predict", "prediction", "guess", "exact", "forecast", "be"}),
        ),
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
        tail = f" Final: {game.cover_result_label}."
        if game.final_score_text:
            tail += f" {game.final_score_text}."
        parts.append(tail)
    return " ".join(parts)


def _game_keywords(game: GameRow) -> tuple[str, ...]:
    keywords = list(_TEAM_SYNONYMS.get(game.home.lower(), (game.home.lower(),)))
    keywords.extend(_TEAM_SYNONYMS.get(game.away.lower(), (game.away.lower(),)))
    keywords.append(game.pick_team.lower())
    keywords.append("pick")
    return tuple(keywords)


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
    watching_items: tuple[tuple[str, str, float], ...],
) -> dict[str, Any]:
    """Build the deterministic retrieval corpus for one page.

    Every answer body is composed verbatim from the inputs -- this
    function invents no number and no claim, so the numeric-guard test
    (every number in an answer also occurs in the corpus) holds by
    construction.
    """

    entries: list[_Entry] = list(_deflect_entries(int(season or 0), int(week or 0)))

    for game in games:
        entries.append(
            _Entry(
                entry_id=f"game:{game.game_id}",
                keywords=_game_keywords(game),
                body=_game_body(game, best_pick_note=best_pick_note),
                anchor=f"index.html#{game.game_id}",
            )
        )

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
                keywords=SYNONYMS["best_pick"],
                body=body,
                anchor=f"index.html#{best_pick_game_id}",
            )
        )

    if record_lines:
        entries.append(
            _Entry(
                entry_id="record",
                keywords=SYNONYMS["record"],
                body=" ".join(record_lines),
                anchor="model.html",
            )
        )

    if policy_text:
        entries.append(
            _Entry(
                entry_id="policy",
                keywords=SYNONYMS["policy"],
                body=(f"{policy_text} Games the policy flipped are marked on the board."),
                anchor="index.html",
            )
        )

    for tag, text in finding_items:
        entries.append(
            _Entry(
                entry_id=f"finding:{tag}",
                keywords=SYNONYMS["findings"] + (tag.lower(),),
                body=text,
                anchor="findings.html",
            )
        )

    for name, effect_text, probability_positive in watching_items:
        entries.append(
            _Entry(
                entry_id=f"watching:{name}",
                keywords=SYNONYMS["leads"] + (name.lower(),),
                body=(
                    f"{name}: {effect_text} "
                    f"(probability positive {probability_positive:.4f}; "
                    "unresolved below power -- an open lead, not a verdict)."
                ),
                anchor="findings.html",
            )
        )

    for item in GLOSSARY:
        entries.append(
            _Entry(
                entry_id=f"glossary:{item.term}",
                keywords=(item.term.lower(),),
                body=f"{item.term}: {item.definition}",
                anchor="index.html",
            )
        )

    entries.append(
        _Entry(
            entry_id="timing",
            keywords=SYNONYMS["timing"],
            body=(
                "Picks can be updated until each game's own kickoff; the "
                "pool's lines freeze Tuesday. This card was generated "
                f"{generated_at_text}."
            ),
            anchor="index.html",
        )
    )

    payload = {
        "assistant_version": ASSISTANT_VERSION,
        "generated_at": generated_at_text,
        "season": season,
        "week": week,
        "page": page,
        "model": {
            "model_id": model_id,
            "method_label": method_label,
        },
        "synonyms": {key: list(value) for key, value in SYNONYMS.items()},
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
            {
                "id": entry.entry_id,
                "keywords": list(entry.keywords),
                "body": entry.body,
                "anchor": entry.anchor,
            }
            for entry in entries
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
    watching_items: tuple[tuple[str, str, float], ...] = ()
    if findings_page is not None:
        watching_items = tuple(
            (lead.name, lead.effect_text, lead.probability_positive)
            for lead in findings_page.watching_leads
        )
    return build_knowledge(
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
        (lead.name, lead.effect_text, lead.probability_positive) for lead in findings.watching_leads
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


def answer(question: str, knowledge: Mapping[str, Any]) -> AssistantAnswer:
    """Reference matcher: deflect set first, then ranked entries, else
    the fallback. Pure function of (question, corpus)."""

    query = question.strip()
    if not query:
        fallback = knowledge["fallback"]
        return AssistantAnswer(
            topic="fallback",
            text=str(fallback["body"]),
            anchors=tuple(str(anchor) for anchor in fallback["anchors"]),
        )
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
    query_tokens = frozenset(_tokens(query))
    scored = [
        (_score(query_tokens, tuple(entry["keywords"])), position, entry)
        for position, entry in enumerate(entries)
        if not str(entry["id"]).startswith("deflect:")
    ]
    (best_raw, _, _, _), _, best = max(scored, key=lambda row: (row[0], -row[1]))
    if best_raw < 1 and not any(row[0][2] for row in scored):
        fallback = knowledge["fallback"]
        return AssistantAnswer(
            topic="fallback",
            text=str(fallback["body"]),
            anchors=tuple(str(anchor) for anchor in fallback["anchors"]),
        )
    return AssistantAnswer(
        topic=str(best["id"]), text=str(best["body"]), anchors=(str(best["anchor"]),)
    )


def _js_string_array(items: tuple[str, ...], *, per_line: int = 8) -> str:
    """A wrapped JS string-array literal, aliases sorted for determinism."""

    quoted = [f'"{item}"' for item in sorted(items)]
    lines = [
        ", ".join(quoted[index : index + per_line]) for index in range(0, len(quoted), per_line)
    ]
    return "[" + ",\n      ".join(lines) + "]"


def assistant_script() -> str:
    """Render the thin inline-JS port of :func:`answer`'s rank rule.

    The STOP table and the deflect rules are GENERATED from
    :data:`STOPWORDS` and :func:`_deflect_rule_sets` -- the JS can never
    drift from the tested Python matcher on either. The port only ever
    returns an entry's own body: no string it did not find in the
    corpus.
    """

    stop_js = _js_string_array(tuple(STOPWORDS))
    rule_js_lines = []
    for position, rule in enumerate(_deflect_rule_sets()):
        groups = "[" + ", ".join(_js_string_array(tuple(group)) for group in rule) + "]"
        rule_js_lines.append(f"      {{ groups: {groups}, entry: {position} }},")
    rules_js = "[\n" + "\n".join(rule_js_lines) + "\n    ]"
    return _ASSISTANT_SCRIPT_TEMPLATE.replace("/*__STOP__*/", stop_js).replace(
        "/*__RULES__*/", rules_js
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
  function expand(token, synonyms, teams) {
    var group = [token];
    Object.keys(teams).forEach(function (code) {
      if (teams[code].indexOf(token) !== -1) { group = group.concat(teams[code]); }
    });
    Object.keys(synonyms).forEach(function (canon) {
      if (canon === token || synonyms[canon].indexOf(token) !== -1) {
        group.push(canon);
        group = group.concat(synonyms[canon]);
      }
    });
    return group;
  }
  function score(qtokens, keywords, synonyms, teams) {
    var raw = 0, exact = 0, expanded = 0, rawBest = 0;
    var lowered = keywords.map(function (k) { return k.toLowerCase(); });
    qtokens.forEach(function (token) {
      var rawLengths = lowered.filter(function (kw) {
        return kw.indexOf(token) !== -1 || (kw.length >= 3 && token.indexOf(kw) !== -1);
      }).map(function (kw) { return kw.length; });
      if (rawLengths.length) {
        raw += 1;
        exact += lowered.filter(function (kw) { return kw === token; }).length;
        rawBest = Math.max.apply(null, [rawBest].concat(rawLengths));
        return;
      }
      var variants = expand(token, synonyms, teams).filter(function (v) { return v !== token; });
      var expandedHit = variants.some(function (v) {
        return lowered.some(function (kw) { return kw.indexOf(v) !== -1; });
      });
      if (expandedHit) { expanded += 1; }
    });
    return [raw, exact, expanded, rawBest];
  }
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
    // Deflect rules mirror _deflect_rule_sets: AND of OR-groups.
    var rules = /*__RULES__*/;
    form.addEventListener('submit', function (evt) {
      evt.preventDefault();
      var q = input.value.trim();
      if (!q) return;
      addEntry('q', 'question', q, null);
      var seen = {};
      var qtokens = tokens(q).filter(function (t) {
        if (seen[t]) return false;
        seen[t] = true;
        return true;
      });
      var flat = q.toLowerCase().replace(/[^a-z0-9]+/g, ' ').replace(/ +/g, ' ').trim();
      var words = flat ? flat.split(' ') : [];
      var padded = ' ' + flat + ' ';
      var fired = null;
      rules.forEach(function (rule) {
        if (fired) return;
        var ok = rule.groups.every(function (group) {
          return group.some(function (alias) {
            if (alias.indexOf(' ') !== -1) { return padded.indexOf(' ' + alias + ' ') !== -1; }
            return words.indexOf(alias) !== -1;
          });
        });
        if (ok) { fired = corpus.entries[rule.entry]; }
      });
      var result = fired;
      if (!result && qtokens.length) {
        var best = null, bestScore = [0, 0, 0, 0], anyExpanded = false;
        corpus.entries.forEach(function (entry) {
          if (entry.id.indexOf('deflect:') === 0) return;
          var s = score(qtokens, entry.keywords, corpus.synonyms, teams);
          if (s[2] > 0) { anyExpanded = true; }
          var better = s[0] > bestScore[0] ||
              (s[0] === bestScore[0] && s[1] > bestScore[1]) ||
              (s[0] === bestScore[0] && s[1] === bestScore[1] &&
                s[2] > bestScore[2]) ||
              (s[0] === bestScore[0] && s[1] === bestScore[1] &&
                s[2] === bestScore[2] && s[3] > bestScore[3]);
          if (better) { bestScore = s; best = entry; }
        });
        result = (bestScore[0] >= 1 || anyExpanded) ? best : null;
      }
      if (result) {
        addEntry('a', result.id, result.body, result.anchor);
      } else {
        var fb = corpus.fallback;
        addEntry('a', 'fallback', fb.body, fb.anchors[0]);
      }
      input.value = '';
      input.focus();
    });
  });
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
        f'<noscript><div class="assistant-topics">{links}</div></noscript>'
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
