"""Backfill ``plain_summary`` onto specific ``registry/weak_signals.json`` rows
that the public findings page renders today but that either had no
plain-English summary at all, or had one that still leaked the research
machinery's own vocabulary (a literal ``P+`` probability notation, or a bare
snake_case field/identifier name -- e.g. ``probability_positive=0.0418``).

Lane AQ, 2026-09-05 (dashboard humanising follow-up to lane AH's audit):
``nfl_ats.findings_registry.top_open_leads``/``recent_registry_activity`` and
``nfl_ats.signal_ledger.build_ledger_rows`` were enumerated against the LIVE
site build (``nfl-ats publish-board``) to find exactly which registry rows
reach a reader today. All but twelve already carried a genuinely plain
``plain_summary``; this script corrects those twelve, and only those twelve --
it is not a general-purpose bulk-editor and intentionally does not scan the
rest of the registry (475+ rows never rendered on the current findings page
are out of this lane's scope, same discipline
``nfl_ats.signal_ledger``/``docs/ledger.html`` already applies elsewhere).

**Deliberately goes through ``nfl_ats.cli.main`` (the same code path
``nfl-ats weak-signals record --replace`` runs), in-process rather than via a
subprocess** -- so ``record_signal``'s ``validate_closure``/
``validate_coherence`` checks execute exactly as they would for a human
typing the command, and a session that widens this script's mapping later
gets the same protection this project's binding closing-grounds rule
requires. In-process (not ``subprocess.run``) only to sidestep Windows
command-line quoting of the long, newline-bearing ``notes``/
``classification_evidence`` fields these rows carry -- the argv list handed
to ``main()`` is identical in shape to what a shell invocation would parse.

Every field below except ``plain_summary`` is read STRAIGHT from the live
registry immediately before re-recording and passed back unchanged
(``--replace``); nothing here hand-transcribes effect, interval,
classification, seasons, notes, or closing_ground -- see the
``_record_args`` docstring for exactly which registry field maps to which
CLI flag. Confirm-after-the-fact: this script's own ``--verify`` pass (also
run automatically at the end of a live run) diffs the registry before/after
and asserts the ONLY JSON key that differs anywhere in the file is
``plain_summary``.

Usage::

    .tools\\uv.exe run --no-sync python scripts\\backfill_plain_summaries.py
    .tools\\uv.exe run --no-sync python scripts\\backfill_plain_summaries.py --dry-run

**2026-09-08 addition (lane Q):** a second, TEMPLATE-driven mode for large
batteries of near-identical cells that a name/field grammar can describe
mechanically, rather than one hand-written sentence per row. A research lane
recorded several hundred ``mod18_home_side_location_cfb_v1_*`` rows (a
college-football replication of the MOD-18 home-side push) with no
``--plain-summary``, which is far too many for the ``PLAIN_SUMMARIES``
hand-written mapping above -- and, being ``recorded_at`` today, they are
exactly what trips
``tests/test_board_humanised.py::test_recent_activity_weak_signal_entries_have_no_plain_summary_backlog``.
``--prefix``/``--family`` select the rows, ``--template-set`` names the
grammar to read them with (see :data:`TEMPLATE_SETS`), and every row with a
name shape the template does not recognise is SKIPPED with a reason rather
than guessed at -- a bad guess in reader-facing text is worse than a visible
gap. Every generated sentence is checked against :func:`banned_tokens_in`
(the same render-contract patterns ``tests/test_board_humanised.py`` scans
for) before it is ever considered for writing. Unlike the mapping mode above,
this one previews by default and writes ONLY with ``--apply``::

    .tools\\uv.exe run --no-sync python scripts\\backfill_plain_summaries.py \\
        --prefix mod18_home_side_location_cfb_v1_ --template-set cfb_home_side_location_v1
    .tools\\uv.exe run --no-sync python scripts\\backfill_plain_summaries.py \\
        --prefix mod18_home_side_location_cfb_v1_ --template-set cfb_home_side_location_v1 --apply
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import json
import os
import re
from collections.abc import Callable, Iterator
from datetime import UTC, datetime
from pathlib import Path

from nfl_ats import cli as nfl_ats_cli
from nfl_ats.weak_signals import Registry, WeakSignal, default_registry_path, load_registry

try:
    from nfl_ats.card_explanation import BANNED_BOILERPLATE
except ImportError:  # pragma: no cover - defensive fallback if the module ever moves
    BANNED_BOILERPLATE = (
        "wagering recommendation",
        "descriptive research summary",
        "research preview",
        "not proof of a profitable",
        "not proof of a stable",
        "gambling problem",
        "1-800-gambler",
    )

REPO = Path(__file__).resolve().parents[1]

#: signal name -> the new, hand-written plain-English replacement. Each is
#: one or two sentences a football fan with no statistics background can
#: read on its own, naming the situation and what the rule found -- and
#: none contain "P+", "week-blocked", a raw identifier, or any other token
#: ``tests/test_board_humanised.py`` bans from reader-visible text.
PLAIN_SUMMARIES: dict[str, str] = {
    # 2026-09-07 lane K (MOD-18): conditional integer-margin mapping, 21 cells.
    "mod18_conditional_margin_v1_K1_2020_2021": (
        "Reading the cover chance off whole-number final margins near the model's predicted "
        "margin did not beat the current model against the opening line in 2020-21: about 29% "
        "likely to help, and the range crosses zero, so nothing changes on the card."
    ),
    "mod18_conditional_margin_v1_K1_2020_2025": (
        "Reading the cover chance off whole-number final margins near the model's predicted "
        "margin did not beat the current model against the opening line in 2020-25: about 11% "
        "likely to help, and the range crosses zero, so nothing changes on the card."
    ),
    "mod18_conditional_margin_v1_K1_2022_2023": (
        "Reading the cover chance off whole-number final margins near the model's predicted "
        "margin did not beat the current model against the opening line in 2022-23: about 37% "
        "likely to help, and the range crosses zero, so nothing changes on the card."
    ),
    "mod18_conditional_margin_v1_K1_2024_2025": (
        "Reading the cover chance off whole-number final margins near the model's predicted "
        "margin did not beat the current model against the opening line in 2024-25: about 8% "
        "likely to help, and the range crosses zero, so nothing changes on the card."
    ),
    "mod18_conditional_margin_v1_K1_brier_2020_2025": (
        "Reading the cover chance off whole-number final margins near the model's predicted "
        "margin gave slightly less accurate probabilities than the current model in 2020-25 "
        "(about 29% likely to be better on that score); nothing changes on the card."
    ),
    "mod18_conditional_margin_v1_K1_log_loss_2020_2025": (
        "Reading the cover chance off whole-number final margins near the model's predicted "
        "margin scored slightly worse than the current model on how well its probabilities "
        "matched results in 2020-25 (about 28% likely to be better); nothing changes on the "
        "card."
    ),
    "mod18_conditional_margin_v1_K1_three_member_card_2020_2025": (
        "Reading the cover chance off whole-number final margins near the model's predicted "
        "margin did not beat the current model on the card with its three fix-up rules in "
        "2020-25: about 11% likely to help, and the range crosses zero, so nothing changes on "
        "the card."
    ),
    "mod18_conditional_margin_v1_K2_2020_2021": (
        "Sliding the historical margin pattern so it centres on the model's prediction did "
        "not beat the current model against the opening line in 2020-21: about 57% likely to "
        "help, and the range crosses zero, so nothing changes on the card."
    ),
    "mod18_conditional_margin_v1_K2_2020_2025": (
        "Sliding the historical margin pattern so it centres on the model's prediction did "
        "not beat the current model against the opening line in 2020-25: about 27% likely to "
        "help, and the range crosses zero, so nothing changes on the card."
    ),
    "mod18_conditional_margin_v1_K2_2022_2023": (
        "Sliding the historical margin pattern so it centres on the model's prediction did "
        "not beat the current model against the opening line in 2022-23: about 11% likely to "
        "help, and the range crosses zero, so nothing changes on the card."
    ),
    "mod18_conditional_margin_v1_K2_2024_2025": (
        "Sliding the historical margin pattern so it centres on the model's prediction did "
        "not beat the current model against the opening line in 2024-25: about 39% likely to "
        "help, and the range crosses zero, so nothing changes on the card."
    ),
    "mod18_conditional_margin_v1_K2_brier_2020_2025": (
        "Sliding the historical margin pattern so it centres on the model's prediction gave "
        "slightly less accurate probabilities than the current model in 2020-25 (about 0% "
        "likely to be better on that score); nothing changes on the card."
    ),
    "mod18_conditional_margin_v1_K2_log_loss_2020_2025": (
        "Sliding the historical margin pattern so it centres on the model's prediction scored "
        "slightly worse than the current model on how well its probabilities matched results "
        "in 2020-25 (about 0% likely to be better); nothing changes on the card."
    ),
    "mod18_conditional_margin_v1_K2_three_member_card_2020_2025": (
        "Sliding the historical margin pattern so it centres on the model's prediction did "
        "not beat the current model on the card with its three fix-up rules in 2020-25: about "
        "28% likely to help, and the range crosses zero, so nothing changes on the card."
    ),
    "mod18_conditional_margin_v1_K3_2020_2021": (
        "The whole-number margin read that also notes which side of a key number the line "
        "sits on did not beat the current model against the opening line in 2020-21: about 9% "
        "likely to help, and the range crosses zero, so nothing changes on the card."
    ),
    "mod18_conditional_margin_v1_K3_2020_2025": (
        "The whole-number margin read that also notes which side of a key number the line "
        "sits on did not beat the current model against the opening line in 2020-25: about 5% "
        "likely to help, and the range crosses zero, so nothing changes on the card."
    ),
    "mod18_conditional_margin_v1_K3_2022_2023": (
        "The whole-number margin read that also notes which side of a key number the line "
        "sits on did not beat the current model against the opening line in 2022-23: about "
        "20% likely to help, and the range crosses zero, so nothing changes on the card."
    ),
    "mod18_conditional_margin_v1_K3_2024_2025": (
        "The whole-number margin read that also notes which side of a key number the line "
        "sits on did not beat the current model against the opening line in 2024-25: about "
        "23% likely to help, and the range crosses zero, so nothing changes on the card."
    ),
    "mod18_conditional_margin_v1_K3_brier_2020_2025": (
        "The whole-number margin read that also notes which side of a key number the line "
        "sits on gave slightly less accurate probabilities than the current model in 2020-25 "
        "(about 16% likely to be better on that score); nothing changes on the card."
    ),
    "mod18_conditional_margin_v1_K3_log_loss_2020_2025": (
        "The whole-number margin read that also notes which side of a key number the line "
        "sits on scored slightly worse than the current model on how well its probabilities "
        "matched results in 2020-25 (about 15% likely to be better); nothing changes on the "
        "card."
    ),
    "mod18_conditional_margin_v1_K3_three_member_card_2020_2025": (
        "The whole-number margin read that also notes which side of a key number the line "
        "sits on did not beat the current model on the card with its three fix-up rules in "
        "2020-25: about 9% likely to help, and the range crosses zero, so nothing changes on "
        "the card."
    ),
    # 2026-09-07 lane F (PER-09): season-lagged play-level unit ratings on top
    # of the played model, opener-graded, three sequential windows.
    "apm_unit_on_production_2020_2021": (
        "Adding each side's offensive and defensive unit strength from play-level ratings "
        "to the model, checked against the opening line in 2020-21: it leaned against the "
        "idea (about 5% likely to help) but the range crosses zero, so it isn't settled."
    ),
    "apm_unit_on_production_2022_2023": (
        "The same unit-strength check against the opening line in 2022-23: essentially a "
        "coin flip (about 53% likely to help); nothing changes on the card."
    ),
    "apm_unit_on_production_2024_2025": (
        "The same unit-strength check against the opening line in 2024-25: a slight lean "
        "in its favour (about 60% likely to help), still inside a range that crosses zero."
    ),
    "fluview_home_market_elevated_opener_confirmation_2022_2023": (
        "A second real check of the home team's rising-illness signal, this time against "
        "the actual opening line in 2022-23: it leans slightly against backing it (about "
        "9% likely to help) but the range still crosses zero, so it isn't settled."
    ),
    "fluview_home_market_elevated_opener_confirmation": (
        "The first real check of the home team's rising-illness signal against the actual "
        "opening line, back in 2020-21: it leans mildly against backing it (about 34% "
        "likely to help), but the range still crosses zero, so this alone doesn't rule it out."
    ),
    "cfb_option_side_on_benchmark": (
        "Backing the triple-option team against an opponent with less time to prepare for "
        "it leans slightly the wrong way in this college-football test (about 38% likely to "
        "help), but the range still crosses zero, so it isn't settled."
    ),
    "sept_heat_home_on_production": (
        "Backing the heat-acclimated home team against a cold-weather visitor in September "
        "leans slightly against it under the rule we actually play (about 8% likely to "
        "help), but leans for it under the simpler sign-only rule (about 78% likely) -- the "
        "two disagree, so neither is proven nor ruled out."
    ),
    "inactives_channel_historical_proxy_v1": (
        "Trying to spot last-minute surprise inactives before kickoff using only past "
        "patterns did not beat the existing Tuesday card in this test (about 4% likely to "
        "help); a control check confirmed the test itself can find real effects, it just "
        "did not find one here."
    ),
    "fluview_home_market_elevated_on_production": (
        "Testing the home team's rising-illness signal on top of the model we actually "
        "play, not just a bare market line, leans positive (about 79% likely to help), but "
        "the range still crosses zero, so it isn't settled."
    ),
    "cfb_true_freshman_road_qb_on_benchmark_era_2021_2025": (
        "In 2021-25, fading a true-freshman quarterback making a road start leaned "
        "strongly the expected way (close to 90% likely to help) -- the closest this lead "
        "has come to resolving, but the range still just touches zero."
    ),
    "rookie_qb_debut_fade_on_production": (
        "Betting against a rookie quarterback in his very first career start did not "
        "clearly help or hurt picks in this test window; it leans slightly against the "
        "fade (about 40% likely to help), but the sample is far too small to call either way."
    ),
    "low_total_div_home_dog_on_production": (
        "Backing the home underdog in a low-scoring division game leans positive so far "
        "(about 69% likely to help), but the range still crosses zero -- not proven, not "
        "ruled out."
    ),
    "new_stadium_home_on_production": (
        "Backing the home team during a brand-new stadium's first two seasons reads as "
        "close to a coin flip so far (about 35% likely to help) -- not proven, not ruled out."
    ),
    "dome_shootout_favorite_on_production": (
        "Backing the favorite in a high-scoring dome game with a tight spread reads as "
        "close to a coin flip so far (about 44% likely to help) -- not proven, not ruled out."
    ),
    "graph_team_stat_active_roster_continuity": (
        "Tests whether adjusting a team's roster-continuity number for strength of "
        "schedule beats simply using the raw number as-is."
    ),
}


def _record_args(signal: WeakSignal, *, plain_summary: str) -> list[str]:
    """The ``weak-signals record --replace`` argv for ``signal``, with every
    field read back off the record itself except ``plain_summary`` -- the
    one field this script is allowed to change."""

    args: list[str] = [
        "weak-signals",
        "record",
        "--name",
        signal.name,
        "--description",
        signal.description,
        "--source",
        signal.source,
        # ``--flag=value`` form: argparse treats a bare "-1e-05" as an option.
        f"--effect={signal.effect!r}",
        "--effect-units",
        signal.effect_units,
        "--classification",
        signal.classification,
        "--league",
        signal.league,
        "--season-start",
        str(signal.seasons[0]),
        "--season-end",
        str(signal.seasons[1]),
    ]
    if signal.standard_error is not None:
        args += [f"--standard-error={signal.standard_error!r}"]
    if signal.interval is not None:
        args += [
            f"--interval-low={signal.interval[0]!r}",
            f"--interval-high={signal.interval[1]!r}",
        ]
    if signal.probability_positive is not None:
        args += [f"--probability-positive={signal.probability_positive!r}"]
    if signal.sample_games is not None:
        args += ["--sample-games", str(signal.sample_games)]
    if signal.sample_blocks is not None:
        args += ["--sample-blocks", str(signal.sample_blocks)]
    if signal.reliability is not None:
        args += [f"--reliability={signal.reliability!r}"]
    if signal.family is not None:
        args += ["--family", signal.family]
    args += ["--classification-evidence", signal.classification_evidence]
    if signal.closing_ground is not None:
        args += ["--closing-ground", signal.closing_ground]
    args += ["--plain-summary", plain_summary]
    if signal.category is not None:
        args += ["--category", signal.category]
    args += ["--notes", signal.notes]
    args += ["--recorded-at", signal.recorded_at]
    args += ["--replace"]
    return args


def _diff_keys(before: dict, after: dict, *, path: str = "") -> set[str]:
    """Every JSON-path key whose value differs between ``before`` and
    ``after`` -- used to assert the ONLY thing this script changed anywhere
    in the registry file is a ``plain_summary`` leaf."""

    changed: set[str] = set()
    if isinstance(before, dict) and isinstance(after, dict):
        for key in sorted(set(before) | set(after)):
            child = f"{path}.{key}" if path else key
            if key not in before or key not in after:
                changed.add(child)
                continue
            changed |= _diff_keys(before[key], after[key], path=child)
    elif before != after:
        changed.add(path)
    return changed


# ---------------------------------------------------------------------------
# Render-contract safety net (mirrors tests/test_board_humanised.py's scan)
# ---------------------------------------------------------------------------

_HEX_RE = re.compile(r"(?<![a-z0-9])[0-9a-f]{8,}(?![a-z0-9])", re.IGNORECASE)
_VERSIONED_SLUG_RE = re.compile(r"\b[a-z][a-z0-9]*(?:_[a-z0-9]+)*_v\d\b")
_SNAKE_CASE_RE = re.compile(r"\b[a-z][a-z0-9]*(?:_[a-z0-9]+)+\b")
_SNAPSHOT_STAMP_RE = re.compile(r"\b\d{8}T\d{6}Z\b")
_ISO_TIMESTAMP_RE = re.compile(r"\b\d{4}-\d{2}-\d{2}T\d{2}:\d{2}")
_P_PLUS_RE = re.compile(r"\bP\+")
_WEEK_BLOCKED_RE = re.compile(r"week-blocked", re.IGNORECASE)


def banned_tokens_in(text: str) -> list[str]:
    """Every render-contract violation ``text`` would trip on
    ``tests/test_board_humanised.py``'s scan -- a hex-looking id, a
    ``..._v1``-style slug, a raw artifact/ISO stamp, the literal "P+" or
    "week-blocked", a bare snake_case identifier, or a banned boilerplate
    phrase. Mirrors that test's patterns directly (not a full
    reimplementation -- a generated ``plain_summary`` is always a bare
    sentence, never markup, so there is no HTML/backtick/file-path stripping
    to do first). Used to check a generated summary BEFORE it is ever
    written, not after the next test run finds it.
    """

    violations: list[str] = []
    lowered = text.lower()
    for phrase in BANNED_BOILERPLATE:
        if phrase in lowered:
            violations.append(f"banned boilerplate phrase {phrase!r}")
    hex_hit = _HEX_RE.search(text)
    if hex_hit is not None and not hex_hit.group(0).isdigit():
        violations.append(f"hex-looking token {hex_hit.group(0)!r}")
    slug_hit = _VERSIONED_SLUG_RE.search(text)
    if slug_hit is not None:
        violations.append(f"versioned slug {slug_hit.group(0)!r}")
    stamp_hit = _SNAPSHOT_STAMP_RE.search(text)
    if stamp_hit is not None:
        violations.append(f"raw artifact stamp {stamp_hit.group(0)!r}")
    iso_hit = _ISO_TIMESTAMP_RE.search(text)
    if iso_hit is not None:
        violations.append(f"raw ISO timestamp {iso_hit.group(0)!r}")
    if _P_PLUS_RE.search(text) is not None:
        violations.append('literal "P+" notation')
    if _WEEK_BLOCKED_RE.search(text) is not None:
        violations.append('literal "week-blocked" phrase')
    snake_hit = _SNAKE_CASE_RE.search(text)
    if snake_hit is not None:
        violations.append(f"bare snake_case identifier {snake_hit.group(0)!r}")
    return violations


# ---------------------------------------------------------------------------
# Small formatting helpers shared by both template sets
# ---------------------------------------------------------------------------


def _format_points(value: float) -> str:
    """A point-scale number (``ats_points``/``accuracy_points``) as reader
    text: one decimal, with a trailing ``.0`` dropped."""

    rounded = round(value, 1)
    if rounded == int(rounded):
        return str(int(rounded))
    return f"{rounded:.1f}"


def _format_small(value: float) -> str:
    """A probability-scale number (``brier_improvement``/
    ``log_loss_improvement``, typically 0.0001-0.01) as reader text: up to
    four decimals, trailing zeros dropped."""

    text = f"{round(value, 4):.4f}".rstrip("0").rstrip(".")
    return text if text else "0"


def _noun_for(value_text: str, plural_noun: str) -> str:
    """``plural_noun`` singularised when ``value_text`` (already formatted by
    :func:`_format_points`) reads exactly "1" -- "1 percentage point", never
    "1 percentage points"."""

    return plural_noun[:-1] if value_text == "1" and plural_noun.endswith("s") else plural_noun


def _rate_clause(effect: float, effect_units: str) -> str:
    """Plain-English read of an ``accuracy_points``/``brier_improvement``/
    ``log_loss_improvement`` effect, always in the module-wide "positive
    favours the candidate" convention (``nfl_ats.weak_signals``): positive
    reads better than the comparison, negative reads worse. Point-scale and
    probability-scale units get different number formatting
    (:func:`_format_points` vs :func:`_format_small`) because a raw Brier/
    log-loss difference is two more orders of magnitude smaller."""

    if effect_units == "accuracy_points":
        if abs(effect) < 0.05:
            return "reads about the same as not applying it"
        direction = "better" if effect > 0 else "worse"
        value_text = _format_points(abs(effect))
        noun = _noun_for(value_text, "percentage points")
        return f"reads about {value_text} {noun} {direction} than not applying it"
    if effect_units in ("brier_improvement", "log_loss_improvement"):
        if abs(effect) < 0.00005:
            return (
                "makes no measurable difference to how well the model's confidence "
                "numbers matched what happened"
            )
        direction = "closer to" if effect > 0 else "further from"
        return (
            f"the model's confidence numbers land about {_format_small(abs(effect))} "
            f"points {direction} what actually happened than not applying it"
        )
    raise ValueError(f"effect_units {effect_units!r} has no rate-clause phrasing")


# ---------------------------------------------------------------------------
# Template set 1: the CFB home-side-location replication
# (mod18_home_side_location_cfb_v1_{arm}_{era}_{spread}_{home}_{units})
# ---------------------------------------------------------------------------

_CFB_DESCRIPTION_RE = re.compile(r"^CFB (?P<arm>\S+) (?P<era>\S+) (?P<spread>\S+) (?P<home>\S+)$")

_CFB_HOME_PHRASE = {
    "home_favourite": "a home favourite",
    "home_underdog": "a home underdog",
}

_CFB_ARM_CLAUSE = {
    "s2": "applying it across every spread size",
    "s3": "applying it only on spreads of seven points or more",
}


def describe_cfb_home_side_location_cell(signal: WeakSignal) -> str:
    """Plain-English summary for one ``mod18_home_side_location_cfb_v1_*``
    row (docs/home_side_offset_promotion.md's S2/S3 home-side push,
    independently replicated on college football; family
    ``mod18_home_side_location_cfb_v1``, notes "Independent CFB replication
    of frozen NFL shape").

    Reads the ``arm``/``era``/``spread``/``home`` breakdown from the row's
    own ``description`` field (``"CFB {arm} {era} {spread} {home}"``, e.g.
    ``"CFB diagnosis era_2006_2011 7.5-10 home_underdog"``) rather than
    re-parsing the name -- the description already tokenises cleanly on
    spaces, while the name packs the same pieces into one underscore run
    that is ambiguous to split back apart (``era_2006_2011`` vs
    ``7_5_10``). The year range itself always comes from the row's own
    ``seasons`` field, never the era token, since that is the one place a
    measurement, not a label, records the window. Raises ``ValueError`` for
    any row whose description does not match this shape or names an arm
    outside ``{diagnosis, s2, s3}`` -- the caller skips it rather than
    guessing.
    """

    match = _CFB_DESCRIPTION_RE.match(signal.description)
    if match is None:
        raise ValueError(
            f"signal {signal.name!r}: description {signal.description!r} does not match "
            "the CFB home-side-location cell shape 'CFB {arm} {era} {spread} {home}'"
        )
    arm = match.group("arm")
    era_token = match.group("era")
    spread_token = match.group("spread")
    home_token = match.group("home")
    if arm not in ("diagnosis", "s2", "s3"):
        raise ValueError(f"signal {signal.name!r}: unrecognised CFB arm {arm!r}")

    start, end = signal.seasons
    year_phrase = f"{start}" if start == end else f"{start}-{end}"
    if era_token == "clean_core":
        year_phrase += " (a cleaner sample of games)"

    if arm == "diagnosis":
        if signal.effect_units != "ats_points":
            raise ValueError(
                f"signal {signal.name!r}: diagnosis arm expected ats_points, got "
                f"{signal.effect_units!r}"
            )
        descriptors = []
        if spread_token != "all":
            descriptors.append(f"on {spread_token} point spreads")
        home_phrase = _CFB_HOME_PHRASE.get(home_token)
        if home_phrase:
            descriptors.append(f"with {home_phrase}")
        where_clause = " ".join(descriptors) if descriptors else "across every spread size"
        value = signal.effect
        if abs(value) < 0.05:
            outcome = "the home team landed almost exactly on the model's number"
        else:
            direction = "beat" if value > 0 else "missed"
            value_text = _format_points(abs(value))
            noun = _noun_for(value_text, "points")
            outcome = f"the home team {direction} the model's number by about {value_text} {noun}"
        return (
            f"College football check of the home-team push: {where_clause}, "
            f"{year_phrase}, {outcome}."
        )

    if arm not in _CFB_ARM_CLAUSE:
        raise ValueError(f"signal {signal.name!r}: no serving-rule clause for arm {arm!r}")
    descriptors = [_CFB_ARM_CLAUSE[arm]]
    if spread_token != "all":
        descriptors.append(f"looking only at {spread_token} point spreads")
    lead_clause = ", ".join(descriptors)
    reads_clause = _rate_clause(signal.effect, signal.effect_units)
    return (
        f"College football check of the home-team push: {lead_clause}, "
        f"{year_phrase}, {reads_clause}."
    )


# ---------------------------------------------------------------------------
# Template set 2: the NFL home-side-location / conditional-margin cell
# grammar shared by mod18_home_side_location_v1_{s3,s4,s5}_* and
# mod18_conditional_margin_v1_{m1,mp1}_*
# ---------------------------------------------------------------------------

_NFL_CELL_FAMILIES = ("mod18_home_side_location_v1", "mod18_conditional_margin_v1")

_NFL_BUCKET_DISPLAY = {
    "0-3": "0-3",
    "3p5-6p5": "3.5-6.5",
    "7": "7",
    "7p5-10": "7.5-10",
    "10p5plus": "10.5+",
}

_NFL_HOMESIDE_DISPLAY = {
    "home_favourite": "games with the home team favoured",
    "home_underdog": "games with the home team an underdog",
    "pickem": "pick'em games with no favourite",
}

_NFL_METRIC = r"(?:conditional_log_loss|standalone|card|log_loss|brier)"
_NFL_BUCKET = r"(?:0-3|3p5-6p5|7p5-10|10p5plus|7)"
_NFL_HOMESIDE = r"(?:home_favourite|home_underdog|pickem)"

_NFL_CELL_BUCKET_RE = re.compile(
    rf"^cell_bucket_(?P<bucket>{_NFL_BUCKET})(?:_(?P<home>{_NFL_HOMESIDE}|all))?"
    rf"_(?P<metric>{_NFL_METRIC})_(?P<start>\d{{4}})_(?P<end>\d{{4}})$"
)
_NFL_OVERALL_RE = re.compile(
    rf"^overall_(?P<metric>{_NFL_METRIC})_(?P<start>\d{{4}})_(?P<end>\d{{4}})$"
)
_NFL_SEASON_RE = re.compile(
    rf"^season_(?P<year>\d{{4}})_(?P<metric>{_NFL_METRIC})_(?P<start>\d{{4}})_(?P<end>\d{{4}})$"
)
_NFL_PUSH_AT_3_RE = re.compile(
    rf"^push_at_3_(?P<metric>{_NFL_METRIC})_(?P<start>\d{{4}})_(?P<end>\d{{4}})$"
)
_NFL_S3_VS_S2_RE = re.compile(r"^s3_(?P<year>\d{4})_vs_s2$")

#: (family, sub-arm GROUP) -> what the arm does, in plain English. Read from
#: docs/home_side_offset_promotion.md ("S3 played" section), docs/home_side_side_aware.md
#: (lane G, S4), docs/home_side_prior.md (lane I, S5a/b/c), docs/big_spread_lattice.md
#: (lane H, M1/M1b/MP1).
_NFL_LEAD = {
    ("mod18_home_side_location_v1", "s3"): (
        "the home-team push in the NFL, applied only on spreads of seven points or more"
    ),
    ("mod18_home_side_location_v1", "s4"): (
        "the home-team push in the NFL, fit separately for home favourites, home "
        "underdogs, and pick'em games"
    ),
    ("mod18_home_side_location_v1", "s5"): (
        "the home-team push in the NFL, using a different-sized sample of past games "
        "to size the correction"
    ),
    ("mod18_conditional_margin_v1", "m1"): (
        "reading the home team's win chance from nearby past final scores, on big NFL spreads"
    ),
    ("mod18_conditional_margin_v1", "mp1"): (
        "reading the home team's win chance from nearby past final scores weighted "
        "toward the model's own pick, on big NFL spreads"
    ),
}

#: (GROUP, VARIANT) -> an extra clause for a lettered sensitivity variant of
#: the base arm; a variant with no entry here (including every arm's own
#: base case, VARIANT == GROUP) gets no extra clause. Not exhaustive of every
#: possible future variant letter -- an unlisted variant still gets the base
#: arm's sentence, just without its own distinguishing note; see the
#: docstring below.
_NFL_VARIANT_NOTE = {
    ("s4", "s4b"): "using a smaller, 50-game sample to fit it",
    ("s5", "s5a"): "using a 50-game sample to fit it",
    ("s5", "s5b"): "using a 25-game sample to fit it",
    ("s5", "s5c"): "using a 200-game sample to fit it",
    ("m1", "m1b"): (
        "using a version that also checks which side of the nearest key number the line sits on"
    ),
}


def _nfl_confidence_clause(effect: float, metric: str) -> str:
    note = " (weighted toward the closest calls)" if metric == "conditional_log_loss" else ""
    if abs(effect) < 0.00005:
        return (
            "makes no measurable difference to how well the model's confidence numbers "
            f"matched what happened{note}"
        )
    direction = "closer to" if effect > 0 else "further from"
    return (
        f"the model's confidence numbers land about {_format_small(abs(effect))} points "
        f"{direction} what actually happened than not using it{note}"
    )


def _nfl_s3_vs_s2_sentence(signal: WeakSignal, year: str) -> str:
    if abs(signal.effect) < 0.05:
        outcome = "reads about the same as applying it on every spread"
    else:
        direction = "better" if signal.effect > 0 else "worse"
        value_text = _format_points(abs(signal.effect))
        noun = _noun_for(value_text, "percentage points")
        outcome = f"reads about {value_text} {noun} {direction} than applying it on every spread"
    return (
        "Comparing the home-team push applied only on spreads of seven points or more against "
        f"applying it everywhere, for the {year} NFL season: the seven-plus-only version {outcome}."
    )


def describe_nfl_home_side_cell(signal: WeakSignal) -> str:
    """Plain-English summary for one row of the cell grammar shared by
    ``mod18_home_side_location_v1_{s3,s4,s5}_*`` (docs/home_side_offset_promotion.md,
    docs/home_side_side_aware.md, docs/home_side_prior.md) and
    ``mod18_conditional_margin_v1_{m1,mp1}_*`` (docs/big_spread_lattice.md).

    Reads the row's own ``family`` field to know which lead sentence applies,
    then parses the NAME (this grammar has no informative ``description``
    text to fall back on -- every row's description is the boilerplate
    ``"Home-side location arm {name-suffix}, positive favours the
    candidate"``) against the shared cell shapes: ``{group}_{variant}_cell_bucket_
    {bucket}[_{home}]_{metric}_{start}_{end}``, ``..._overall_{metric}_{start}_{end}``,
    ``..._season_{year}_{metric}_{start}_{end}``, ``..._push_at_3_{metric}_{start}_{end}``,
    plus S3's own ``s3_{year}_vs_s2`` comparison shape. Raises ``ValueError`` for
    any family, sub-arm group, or cell shape this grammar does not recognise --
    the caller skips it rather than guessing (see the ``_NFL_VARIANT_NOTE``
    docstring: an unrecognised lettered VARIANT of a known GROUP still
    produces a sentence, just without its own distinguishing clause; only an
    unrecognised GROUP or cell shape is skipped outright).
    """

    family = signal.family or ""
    if family not in _NFL_CELL_FAMILIES:
        raise ValueError(
            f"signal {signal.name!r}: family {family!r} is not one of {_NFL_CELL_FAMILIES}"
        )
    prefix = family + "_"
    if not signal.name.startswith(prefix):
        raise ValueError(
            f"signal {signal.name!r}: name does not start with its own family {prefix!r}"
        )
    rest = signal.name[len(prefix) :]

    vs_match = _NFL_S3_VS_S2_RE.match(rest)
    if vs_match is not None:
        return _nfl_s3_vs_s2_sentence(signal, vs_match.group("year"))

    tokens = rest.split("_", 2)
    if len(tokens) < 3:
        raise ValueError(f"signal {signal.name!r}: unrecognised NFL cell shape {rest!r}")
    group, variant, cell_rest = tokens

    lead_key = (family, group)
    if lead_key not in _NFL_LEAD:
        raise ValueError(
            f"signal {signal.name!r}: unrecognised sub-arm group {group!r} for family {family!r}"
        )
    lead = f"Check of {_NFL_LEAD[lead_key]}"
    variant_note = _NFL_VARIANT_NOTE.get((group, variant))
    if variant_note:
        lead = f"{lead}, {variant_note}"

    cell_match = (
        _NFL_CELL_BUCKET_RE.match(cell_rest)
        or _NFL_OVERALL_RE.match(cell_rest)
        or _NFL_SEASON_RE.match(cell_rest)
        or _NFL_PUSH_AT_3_RE.match(cell_rest)
    )
    if cell_match is None:
        raise ValueError(f"signal {signal.name!r}: unrecognised NFL cell shape {cell_rest!r}")

    gd = cell_match.groupdict()
    metric = gd["metric"]
    start, end = int(gd["start"]), int(gd["end"])
    year_phrase = f"{start}" if start == end else f"{start}-{end}"

    if cell_match.re is _NFL_CELL_BUCKET_RE:
        bucket_display = _NFL_BUCKET_DISPLAY.get(gd["bucket"], gd["bucket"])
        parts = [f"on {bucket_display} point spreads"]
        home = gd.get("home")
        if home and home != "all":
            parts.append(f"in {_NFL_HOMESIDE_DISPLAY.get(home, home.replace('_', ' '))}")
        population = " ".join(parts)
    elif cell_match.re is _NFL_SEASON_RE:
        population = f"in the {gd['year']} season"
    elif cell_match.re is _NFL_PUSH_AT_3_RE:
        population = "specifically on games that landed exactly on a 3-point push"
    else:
        population = "across the whole sample"

    if signal.effect_units == "accuracy_points":
        reads_clause = _rate_clause(signal.effect, signal.effect_units)
    elif signal.effect_units in ("brier_improvement", "log_loss_improvement"):
        reads_clause = _nfl_confidence_clause(signal.effect, metric)
    else:
        raise ValueError(
            f"signal {signal.name!r}: unsupported effect_units {signal.effect_units!r} for "
            "the NFL home-side cell template set"
        )

    return f"{lead}: {population}, {year_phrase}, {reads_clause}."


#: Every template set this script knows, by the ``--template-set`` name a
#: caller passes on the command line. Add a new entry here (and a new
#: ``describe_*`` function above) rather than growing either function to
#: cover a second, unrelated cell grammar.
TEMPLATE_SETS: dict[str, Callable[[WeakSignal], str]] = {
    "cfb_home_side_location_v1": describe_cfb_home_side_location_cell,
    "nfl_home_side_cell_v1": describe_nfl_home_side_cell,
}


def _select_candidates(
    registry: Registry, *, prefix: str | None, family: str | None
) -> list[WeakSignal]:
    """Every signal matching ``prefix``/``family`` (either or both; a
    signal must satisfy every filter given) that has no ``plain_summary``
    yet -- the exact backlog a ``--template-set`` run would fill in."""

    chosen = []
    for signal in registry.signals.values():
        if prefix is not None and not signal.name.startswith(prefix):
            continue
        if family is not None and signal.family != family:
            continue
        if (signal.plain_summary or "").strip():
            continue
        chosen.append(signal)
    return sorted(chosen, key=lambda s: s.name)


@contextlib.contextmanager
def _forced_registry_dir(registry_path: Path) -> Iterator[None]:
    """Force ``nfl_ats.weak_signals.default_registry_path()`` -- and every
    CLI handler that calls it, ``nfl_ats_cli.main`` included -- to resolve
    to ``registry_path`` for the duration of the block, regardless of the
    ambient ``NFL_ATS_REGISTRY_DIR``.

    Exists because of a real incident (2026-09-08, lane Q): ``main`` here
    reads/writes ``registry_path`` directly, but ``nfl_ats_cli.main(argv)``
    -- the ``weak-signals record --replace`` path this function uses to
    write -- has no way to be TOLD which registry to use; it always resolves
    its own path via ``default_registry_path()``, which reads
    ``NFL_ATS_REGISTRY_DIR`` from the process environment. A test that
    forgot to set that variable had its own reads/candidate-selection target
    a scratch fixture while its writes silently landed on the real,
    concurrently-written ``registry/weak_signals.json`` -- fabricating field
    values on three live rows before the bug was caught. This guard makes
    that class of mismatch structurally impossible: every write in
    :func:`run_template_backfill` now happens with the environment forced to
    agree with the path this function was actually given, and the function
    additionally refuses to run at all if, even after forcing it, the two
    paths still disagree (a custom filename, say) rather than writing to a
    file it was not told to.
    """

    target = str(registry_path.parent)
    previous = os.environ.get("NFL_ATS_REGISTRY_DIR")
    os.environ["NFL_ATS_REGISTRY_DIR"] = target
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop("NFL_ATS_REGISTRY_DIR", None)
        else:
            os.environ["NFL_ATS_REGISTRY_DIR"] = previous


def run_template_backfill(
    registry_path: Path,
    *,
    prefix: str | None,
    family: str | None,
    template_set: str,
    apply: bool,
) -> dict[str, object]:
    """Preview (``apply=False``, the default) or write (``apply=True``) a
    template-generated ``plain_summary`` for every row matching
    ``prefix``/``family`` that has none yet.

    Writes go through :func:`_record_args` + ``nfl_ats_cli.main`` -- the
    exact same ``weak-signals record --replace`` path the hand-written
    ``PLAIN_SUMMARIES`` mode above uses -- so ``record_signal``'s
    ``validate_closure``/``validate_coherence`` run on every write and every
    field except ``plain_summary`` is read back off the live record, never
    hand-transcribed. The before/after ``plain_summary``-only diff check
    below is the same assertion :func:`main`'s legacy path already makes.
    A row whose generated text fails :func:`banned_tokens_in` is treated
    exactly like a row the template could not parse: skipped, with the
    violation as the reason, never written. Every write happens inside
    :func:`_forced_registry_dir` -- see that function's docstring for why.
    """

    describe = TEMPLATE_SETS[template_set]
    registry = load_registry(registry_path)
    candidates = _select_candidates(registry, prefix=prefix, family=family)

    generated: dict[str, str] = {}
    skipped: dict[str, str] = {}
    for signal in candidates:
        try:
            summary = describe(signal)
        except ValueError as exc:
            skipped[signal.name] = str(exc)
            continue
        violations = banned_tokens_in(summary)
        if violations:
            skipped[signal.name] = (
                f"generated summary failed the banned-token check: {violations!r}"
            )
            continue
        generated[signal.name] = summary

    report: dict[str, object] = {
        "registry": str(registry_path),
        "template_set": template_set,
        "prefix": prefix,
        "family": family,
        "candidates": len(candidates),
        "generated": generated,
        "generated_count": len(generated),
        "skipped": skipped,
        "skipped_count": len(skipped),
        "applied": False,
    }

    if not apply or not generated:
        return report

    before_payload = json.loads(registry_path.read_text(encoding="utf-8"))
    before_snapshot = copy.deepcopy(before_payload)

    recorded: list[str] = []
    with _forced_registry_dir(registry_path):
        resolved = default_registry_path()
        if resolved != registry_path:
            raise ValueError(
                f"registry_path {registry_path} does not resolve back through "
                f"default_registry_path() ({resolved}) even after forcing "
                "NFL_ATS_REGISTRY_DIR; refusing to write via nfl_ats_cli.main to avoid "
                "writing to the wrong file (see _forced_registry_dir's docstring)"
            )
        for name, summary in sorted(generated.items()):
            live_registry = load_registry(registry_path)  # previous writes land here
            signal = live_registry.signals[name]
            argv = _record_args(signal, plain_summary=summary)
            nfl_ats_cli.main(argv)  # runs record_signal -> validate_closure/validate_coherence
            recorded.append(name)

    after_payload = json.loads(registry_path.read_text(encoding="utf-8"))
    changed_keys = _diff_keys(before_snapshot, after_payload)
    non_plain_summary_changes = sorted(k for k in changed_keys if not k.endswith(".plain_summary"))
    report["applied"] = True
    report["recorded"] = recorded
    report["recorded_count"] = len(recorded)
    report["changed_json_keys"] = sorted(changed_keys)
    report["non_plain_summary_changes"] = non_plain_summary_changes
    if non_plain_summary_changes:
        raise SystemExit(
            "template backfill changed field(s) other than plain_summary: "
            f"{non_plain_summary_changes!r}"
        )
    return report


def find_missing_plain_summaries(*, days: int = 7) -> dict[str, object]:
    """Everything the LIVE findings page renders today (What we're watching,
    Research this week, Signal registry -- the same three surfaces this
    script's own ``PLAIN_SUMMARIES`` mapping worked through on 2026-09-05)
    that still has no genuine ``plain_summary`` and therefore renders the
    "Plain-English summary pending" placeholder. A future session's backlog
    view: ``--missing-plain-summary`` (task item 4's "``nfl-ats weak-signals
    status --missing-plain-summary`` style listing (or a script flag)" --
    ``cli_commands/registry.py`` is out of this lane's editable files, so
    this is that flag, here).

    Deliberately scoped to what is actually RENDERED, not the whole
    registry: 700+ rows never reach a reader and backfilling every one of
    them is a different, much larger task than this lane's."""

    registry = load_registry(default_registry_path())
    from nfl_ats.findings_registry import (
        load_rotation_registry,
        recent_registry_activity,
        top_open_leads,
    )
    from nfl_ats.signal_ledger import build_ledger_rows

    rotation_registry = load_rotation_registry()

    watching_missing = sorted(
        lead.name
        for lead in top_open_leads(registry)
        if not (registry.signals[lead.name].plain_summary or "").strip()
    )

    activity = recent_registry_activity(registry, rotation_registry, datetime.now(UTC), days=days)
    recent_missing_weak_signal: set[str] = set()
    recent_missing_rotation: set[str] = set()
    for _category, entries in activity.entries_by_category:
        for entry in entries:
            if entry.plain_summary:
                continue
            store, name = entry.key.split(":", 1)
            (recent_missing_weak_signal if store == "weak_signal" else recent_missing_rotation).add(
                name
            )

    rows = build_ledger_rows(registry)
    rows.sort(key=lambda r: r.get("pp") if r.get("pp") is not None else -1, reverse=True)
    _notable_signal_limit = 8  # mirrors board_site_content._NOTABLE_SIGNAL_LIMIT
    notable_missing = sorted(
        str(r["name"]) for r in rows[:_notable_signal_limit] if r.get("fallback")
    )

    return {
        "watching_leads_missing_plain_summary": watching_missing,
        "recent_activity_weak_signal_missing_plain_summary": sorted(recent_missing_weak_signal),
        "recent_activity_rotation_missing_plain_summary": sorted(recent_missing_rotation),
        "signal_registry_notable_missing_plain_summary": notable_missing,
        "note": (
            "Rotation entries are ALWAYS listed here: rotation.Family has no "
            "plain_summary field in its schema at all (only a research-prose "
            "description), so filling this in is a schema change, out of this "
            "script's scope -- listed for visibility, not as a per-row backlog "
            "item the way the weak-signal names are. Each weak-signal name "
            "above is rendered TODAY with the 'Plain-English summary pending' "
            "placeholder; fix with `nfl-ats weak-signals record --replace "
            '--plain-summary "..."` (passing every other field unchanged) or '
            "add it to this script's own PLAIN_SUMMARIES mapping."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print the CLI argv each row would run, without writing the registry",
    )
    parser.add_argument(
        "--missing-plain-summary",
        action="store_true",
        help=(
            "report the live findings page's current plain_summary backlog as JSON "
            "and exit, without recording anything (this script's PLAIN_SUMMARIES "
            "mapping is not consulted)"
        ),
    )
    parser.add_argument(
        "--prefix",
        default=None,
        help="template mode: only rows whose name starts with this string",
    )
    parser.add_argument(
        "--family",
        default=None,
        help="template mode: only rows whose recorded 'family' field equals this",
    )
    parser.add_argument(
        "--template-set",
        choices=sorted(TEMPLATE_SETS),
        default=None,
        help=(
            "derive plain_summary from each matching row's own name/fields using this "
            "template set, for every row that has none yet; requires --prefix and/or "
            "--family; PREVIEWS by default, writes only with --apply (unlike the legacy "
            "PLAIN_SUMMARIES mode below, which is the reverse: it writes by default and "
            "--dry-run opts out)"
        ),
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="with --template-set: actually write the generated summaries (default: preview only)",
    )
    args = parser.parse_args()

    if args.missing_plain_summary:
        print(json.dumps(find_missing_plain_summaries(), indent=2, sort_keys=True))
        return

    if args.template_set is not None:
        if args.prefix is None and args.family is None:
            raise SystemExit("--template-set requires --prefix and/or --family")
        report = run_template_backfill(
            default_registry_path(),
            prefix=args.prefix,
            family=args.family,
            template_set=args.template_set,
            apply=args.apply,
        )
        generated = report["generated"]
        skipped = report["skipped"]
        assert isinstance(generated, dict)
        assert isinstance(skipped, dict)
        for name, summary in sorted(generated.items()):
            print(f"{name} -> {summary}")
        for name, reason in sorted(skipped.items()):
            print(f"{name} -> SKIPPED: {reason}")
        summary_report = {k: v for k, v in report.items() if k not in ("generated", "skipped")}
        print(json.dumps(summary_report, indent=2, sort_keys=True))
        return

    if args.prefix is not None or args.family is not None:
        raise SystemExit("--prefix/--family require --template-set")

    registry_path = default_registry_path()
    before_payload = json.loads(registry_path.read_text(encoding="utf-8"))
    before_snapshot = copy.deepcopy(before_payload)

    missing_names = [
        name for name in PLAIN_SUMMARIES if name not in load_registry(registry_path).signals
    ]
    if missing_names:
        raise SystemExit(f"registry has no signal(s) named {missing_names!r}; nothing recorded")

    recorded: list[str] = []
    for name, plain_summary in sorted(PLAIN_SUMMARIES.items()):
        registry = load_registry(registry_path)  # re-read each time: previous writes land here
        signal = registry.signals[name]
        argv = _record_args(signal, plain_summary=plain_summary)
        if args.dry_run:
            print(f"[dry-run] nfl-ats {' '.join(argv)}")
            continue
        nfl_ats_cli.main(argv)  # runs record_signal -> validate_closure/validate_coherence
        recorded.append(name)

    if args.dry_run:
        return

    after_payload = json.loads(registry_path.read_text(encoding="utf-8"))
    changed_keys = _diff_keys(before_snapshot, after_payload)
    non_plain_summary_changes = sorted(k for k in changed_keys if not k.endswith(".plain_summary"))
    report = {
        "registry": str(registry_path),
        "recorded": recorded,
        "recorded_count": len(recorded),
        "changed_json_keys": sorted(changed_keys),
        "non_plain_summary_changes": non_plain_summary_changes,
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    if non_plain_summary_changes:
        raise SystemExit(
            f"backfill changed field(s) other than plain_summary: {non_plain_summary_changes!r}"
        )


if __name__ == "__main__":
    main()
