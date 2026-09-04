"""Anticipated-question battery for the board assistant (static path).

Every row is a question type a reader could plausibly ask, written BEFORE
looking at engine output: (question, expected topic, required substring).
The engine parses meaning (entities + intent) and composes answers from
corpus data -- it retrieves no canned rows. The same battery runs against
the shipped inline JS for parity (see parity harness, outside the repo).
"""

from __future__ import annotations

import json
import re
from dataclasses import replace

from _board_content_fixtures import build_fixture_content

from nfl_ats.board_assistant import answer, build_knowledge_for_board

BATTERY: tuple[tuple[str, str, str], ...] = (
    # Team picks: codes, nicknames, cities, full names.
    ("Why MIA?", "team_pick", "MIA +3.5"),
    ("dolphins", "team_pick", "MIA +3.5"),
    ("Miami Dolphins", "team_pick", "MIA +3.5"),
    ("Seahawks Patriots", "team_pick", "SEA -3.5"),
    ("New England", "team_pick", "NE at SEA"),
    ("Patriots Seahawks", "team_pick", "SEA -3.5"),
    ("Chiefs?", "team_pick", "DEN at KC"),
    ("Kansas City", "team_pick", "DEN at KC"),
    ("Chiefs Broncos?", "team_pick", "DEN at KC"),
    ("MIA or WAS?", "team_pick", "rates highest"),
    ("compare MIA and Buffalo", "team_pick", "rates highest"),
    ("which is more confident, MIA or KC?", "team_confidence", "3rd most confident"),
    ("why the Colts?", "team_pick", "coach fade"),
    ("tell me about the Dolphins game", "team_pick", "MIA +3.5"),
    ("how about the Jets", "team_pick", "NYJ"),
    ("Bears", "team_pick", "CHI at CAR"),
    ("Green Bay", "team_pick", "GB at MIN"),
    ("Tampa Bay", "team_pick", "TB at CIN"),
    ("49ers", "team_pick", "SF at LA"),
    ("MIA vs LV", "team_pick", "MIA +3.5"),
    ("is MIA the best pick?", "team_pick", "Best Pick"),
    # Team confidence (computed rank included).
    ("how confident are you in Miami?", "team_confidence", "5th most confident of 16"),
    ("how sure are you about the Chiefs pick?", "team_confidence", "3rd most confident"),
    ("confidence in the Seahawks game", "team_confidence", "13th most confident"),
    ("is the MIA pick strong?", "team_confidence", "MIA +3.5 (54.4%"),
    ("how confident is the ARI call?", "team_confidence", "1st most confident"),
    # Team schedules.
    ("when do the Dolphins play?", "team_schedule", "Sun Sep 13"),
    ("when do the Chiefs play?", "team_schedule", "Mon Sep 14"),
    ("where are the Patriots playing?", "team_schedule", "NE at SEA"),
    ("when is the Denver game?", "team_schedule", "Mon Sep 14"),
    # Comparisons.
    ("MIA or Buffalo?", "team_pick", "rates highest"),
    ("who is better, KC or ARI?", "team_pick", "rates highest"),
    # Best Pick phrasings.
    ("best pick", "best_pick", "MIA +3.5"),
    ("best bet", "best_pick", "MIA +3.5"),
    ("lock of the week", "best_pick", "MIA +3.5"),
    ("give me your mortal lock", "best_pick", "MIA +3.5"),
    ("star pick", "best_pick", "MIA +3.5"),
    ("who is the best pick?", "best_pick", "MIA +3.5"),
    # Rankings (composed top-N).
    ("which games are you most confident in?", "rankings", "Most confident: ARI +10.5 (63.8%)"),
    ("rank the games by confidence", "rankings", "WAS +4.5 (61.7%)"),
    ("top 3 picks", "rankings", "KC -3 (57.1%)"),
    ("top 5 picks", "rankings", "MIA +3.5 (54.4%)"),
    ("least confident", "rankings", "Least confident: NYJ +3 (50.5%)"),
    ("what is your weakest pick?", "rankings", "Least confident:"),
    ("surest picks", "rankings", "Most confident:"),
    ("top 1", "rankings", "Most confident: ARI +10.5 (63.8%)"),
    # Dogs and favorites (composed lists).
    ("any upset picks?", "dogs", "Underdog picks (9 of 16)"),
    ("which underdogs do you like?", "dogs", "Underdog picks"),
    ("show me the dogs", "dogs", "MIA +3.5"),
    ("which favorites?", "favorites", "Favorite picks (7 of 16)"),
    ("any chalk?", "favorites", "KC -3 (57.1%)"),
    ("list the underdog picks", "dogs", "ARI +10.5"),
    # Day schedules (composed).
    ("what games are on Sunday?", "slots", "Sunday (13)"),
    ("Monday night game", "slots", "Monday (1): DEN at KC"),
    ("who plays Thursday?", "slots", "Thursday (1): SF at LA"),
    ("any Saturday games?", "slots", "No Saturday game this week"),
    ("weekend games", "slots", "Sunday (13)"),
    ("Wednesday game", "slots", "NE at SEA"),
    ("what plays Monday?", "slots", "DEN at KC"),
    # Record.
    ("how good is the model?", "record", "55.4%"),
    ("is this profitable?", "record", "prospective expectation"),
    ("how has the model done historically?", "record", "55.4%"),
    ("what is the track record?", "record", "55.4%"),
    ("past performance", "record", "55.4%"),
    ("how often does the model win?", "scope:winners", "never straight-up winners"),
    # Policy.
    ("why did the pick flip?", "policy", "Flipped 2 picks"),
    ("tell me about the coach fade", "policy", "coach fade"),
    ("what overlays are active?", "policy", "coach fade"),
    ("explain the arrests policy", "policy", "arrests"),
    ("did anything flip?", "policy", "Flipped 2 picks"),
    # Findings.
    ("what have you learned?", "findings", "63.8%"),
    ("any research on this?", "findings", "63.8%"),
    ("tell me something interesting", "findings", "63.8%"),
    # Timing.
    ("when do picks lock?", "timing", "freeze Tuesday"),
    ("what changed since Tuesday?", "timing", "freeze Tuesday"),
    ("Tuesday deadline", "timing", "freeze Tuesday"),
    ("when was this card generated?", "timing", "generated"),
    ("what happens if the line moves?", "timing", "follows the market side"),
    # Scope: winners, injuries, weather.
    ("who will win it all?", "scope:winners", "never straight-up winners"),
    ("who wins?", "scope:winners", "never straight-up winners"),
    ("straight up winner picks?", "scope:winners", "against the spread"),
    ("any injury news?", "scope:injury", "No injury table"),
    ("is anyone hurt?", "scope:injury", "No injury table"),
    ("any inactives?", "scope:injury", "No injury table"),
    ("what about the weather?", "scope:weather", "No weather table"),
    ("windy games?", "scope:weather", "challengers"),
    ("dome games?", "scope:weather", "challengers"),
    ("cold weather edge?", "scope:weather", "No weather table"),
    # Glossary.
    ("what does push mean?", "glossary:push", "tie against the spread"),
    ("what does cover mean?", "glossary:cover", "beats the spread"),
    ("what does ATS mean?", "glossary:ATS", "Against the spread"),
    ("define vig", "glossary:vig", "commission"),
    ("what is the spread?", "glossary:spread", "point handicap"),
    ("odds?", "glossary:vig", "commission"),
    ("juice", "glossary:vig", "commission"),
    ("what is a push?", "glossary:push", "tie against the spread"),
    ("explain the odds", "glossary:vig", "commission"),
    ("what is the vig?", "glossary:vig", "commission"),
    # Deflects.
    ("should I bet on the Chiefs?", "deflect:wager", "never advises wagers"),
    ("best teaser legs?", "deflect:wager", "never advises wagers"),
    ("should I buy points?", "deflect:wager", "never advises wagers"),
    ("who wins week 2?", "deflect:future", "Week 1 only"),
    ("who wins the Super Bowl?", "deflect:future", "Week 1 only"),
    ("who is everyone picking?", "deflect:ownership", "No pick-popularity"),
    ("fade the public?", "deflect:ownership", "No pick-popularity"),
    ("what will the exact score be?", "deflect:score", "sides against the spread only"),
    ("over under total for Monday?", "deflect:score", "sides against the spread only"),
    ("predict the final score", "deflect:score", "sides against the spread only"),
    # Fallback and edges.
    ("asdkjfh qzx", "fallback", "not in this week's published card"),
    ("", "fallback", "not in this week's published card"),
    ("how are you?", "fallback", "not in this week's published card"),
    ("who are you?", "fallback", "not in this week's published card"),
    ("is there a game you really like?", "best_pick", "MIA +3.5"),
    ("which team do you love this week?", "best_pick", "MIA +3.5"),
    ("thanks", "fallback", "not in this week's published card"),
    ("what time do games start?", "fallback", "not in this week's published card"),
)

BATTERY_REFRESH: tuple[tuple[str, str, str], ...] = (
    ("what changed since Tuesday?", "refresh", "pick now MIA"),
    ("has the MIA pick been refreshed?", "refresh", "pick now MIA"),
    ("did the Sunday update change anything?", "refresh", "pick now MIA"),
    ("did anything change for Kansas City?", "refresh", "No refresh recorded for KC"),
    ("when do picks lock?", "timing", "freeze Tuesday"),
)

_REFRESH_LINES = (
    "MIA at LV refresh (refresh_sat): pick now MIA (Tuesday card: LV); "
    "frozen Tuesday line (home +3.5); line moved +1.5 points.",
)


def _knowledge():
    return build_knowledge_for_board(build_fixture_content())


def _refreshed_knowledge():
    content = replace(build_fixture_content(), refresh_lines=_REFRESH_LINES)
    return build_knowledge_for_board(content)


def test_battery() -> None:
    knowledge = _knowledge()
    for question, topic, substring in BATTERY:
        resolved = answer(question, knowledge)
        assert resolved.topic == topic, f"{question!r} -> {resolved.topic!r}"
        assert substring in resolved.text, f"{question!r} missing {substring!r}"
        assert resolved.anchors, f"{question!r} carries no anchor"


def test_refresh_battery() -> None:
    knowledge = _refreshed_knowledge()
    for question, topic, substring in BATTERY_REFRESH:
        resolved = answer(question, knowledge)
        assert resolved.topic == topic, f"{question!r} -> {resolved.topic!r}"
        assert substring in resolved.text, f"{question!r} missing {substring!r}"


def _plain_numbers(text: str) -> list[str]:
    # Ordinals ("5th") refer to ranks already in the corpus; strip the
    # suffix so the guard checks the number, not the formatting.
    return re.findall(r"\d+(?:\.\d+)?%?", re.sub(r"(\d+)(st|nd|rd|th)\b", r"\1", text))


def test_battery_numbers_occur_in_the_corpus() -> None:
    for knowledge in (_knowledge(), _refreshed_knowledge()):
        allowed = set(re.findall(r"\d+(?:\.\d+)?%?", json.dumps(knowledge)))
        for question, _topic, _substring in BATTERY + BATTERY_REFRESH:
            text = answer(question, knowledge).text
            for number in _plain_numbers(text):
                assert number in allowed, f"{number!r} from {question!r} not in corpus"
