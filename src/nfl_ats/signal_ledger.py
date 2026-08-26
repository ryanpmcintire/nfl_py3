"""The public Signal Ledger page: every recorded weak-signal experiment, in
plain language, sortable/filterable/searchable, regenerated fresh from
``registry/weak_signals.json`` on every ``publish-board``.

ROADMAP item (Signal Ledger). This module owns the SCHEMA-to-page mapping and
the rendering; a separate process owns the WORDS
(``registry/reference/signal_summaries.json``, applied onto the registry by
``scripts/apply_signal_summaries.py`` -- see that script's docstring). Nothing
here writes to the registry.

Status derivation (see :func:`_status`): three things are DERIVED from the
record, in precedence order, never guessed:

* ``classification`` of ``refuted_mechanism`` or ``bounded_by_control`` ->
  ``closed`` (an admissible closing ground is already enforced at record
  time, see :mod:`nfl_ats.weak_signals`).
* ``category == "control"`` -> ``control`` (a deliberately unplayable
  instrument check, oracle, or placebo).
* ``name`` in :func:`nfl_ats.four_overlay_composition.on_the_card_registry_names`
  -> ``on_the_card`` (the live production policy).
* Everything else -> ``recorded``.

**"On the card" (2026-08-26 fix).** The obvious source for this was
``artifacts/prospective/challengers.json``'s ``evidence.registry_source``
field -- the exact mechanism :mod:`nfl_ats.model_ledger` already uses to link
a challenger to its registry evidence. Measured: it is ``None`` on EVERY
player-arrests-related entry in that artifact (the promoted overlay, its
retired coach-only incumbent, the union policy itself, and its own
incumbent), so that mechanism resolved nothing for this policy and the page
originally shipped with "On the card" and "Candidates" as two dead filter
chips. The fix is
:data:`nfl_ats.four_overlay_composition.MEMBER_REGISTRY_EVIDENCE` -- a
mapping in CODE, next to the policy it describes, covering all four members
(one of them, the arrest policy, resolved by an exact NUMBER MATCH against
``HANDOFF.md`` rather than a declared source; see that module's docstring for
the full chain and why). ``tests/test_four_overlay_composition.py`` fails the
build if a member has no mapping or a mapped name is missing from the live
registry, so this can't go silently empty again.

**"Candidate" was dropped, not filled in.** There is no code-derivable
source distinguishing "under active consideration for promotion" from
"merely recorded" -- unlike on-the-card, nothing in this codebase declares
that distinction anywhere. A five-chip status row that all work beats a
six-chip row where one is decorative.
"""

from __future__ import annotations

import json
from html import escape
from typing import Any

from nfl_ats.four_overlay_composition import on_the_card_registry_names
from nfl_ats.weak_signals import (
    CATEGORIES,
    TERMINAL_CLASSIFICATIONS,
    Registry,
    WeakSignal,
    family_overlap_warnings,
)

PAGE_FILENAME = "ledger.html"
PAGE_TITLE = "Signal ledger"

STATUS_ON_CARD = "on_the_card"
STATUS_RECORDED = "recorded"
STATUS_CONTROL = "control"
STATUS_CLOSED = "closed"

#: Reader-facing label + pill tone (see ``public_board.pill_html``) per
#: status, in filter-chip order: "All / On the card / Recorded / Controls /
#: Closed" -- five chips, all of which can match a row (see the module
#: docstring for why "Candidate" was dropped instead of shipped empty). The
#: per-row PILL text differs slightly from the per-chip text
#: (:data:`_STATUS_CHIP_LABELS`) only for grammatical number.
_STATUS_META: dict[str, tuple[str, str]] = {
    STATUS_ON_CARD: ("On the card", "live"),
    STATUS_RECORDED: ("Recorded", "idle"),
    STATUS_CONTROL: ("Control arm", "control"),
    STATUS_CLOSED: ("Closed", "bad"),
}

#: Filter-chip label text, plural where the owner's brief used the plural
#: ("Controls") -- the per-row pill above stays singular.
_STATUS_CHIP_LABELS: dict[str, str] = {
    STATUS_ON_CARD: "On the card",
    STATUS_RECORDED: "Recorded",
    STATUS_CONTROL: "Controls",
    STATUS_CLOSED: "Closed",
}

#: Reader-facing names for the fixed :data:`nfl_ats.weak_signals.CATEGORIES`
#: vocabulary, in the module's own declared order.
CATEGORY_LABELS: dict[str, str] = {
    "market": "Market",
    "onfield": "On-field play",
    "health": "Health & availability",
    "schedule": "Schedule & travel",
    "environment": "Weather & venue",
    "attention": "Public attention",
    "offfield": "Off-field",
    "modeling": "Modeling",
    "control": "Control arm",
}
UNCATEGORISED = "uncategorised"
UNCATEGORISED_LABEL = "Uncategorised"

#: Decimal precision + unit words per ``effect_units``, matching the scale
#: comment at the top of ``weak_signals.py``.
_UNIT_META: dict[str, tuple[int, str]] = {
    "accuracy_points": (2, "accuracy pts"),
    "ats_points": (3, "ATS pts"),
    "brier": (4, "Brier"),
    "log_loss": (4, "log-loss"),
    "mae": (3, "MAE"),
}
_ACCURACY_UNIT = "accuracy_points"

#: Free-text markers this project's registry entries already use to disclose
#: a mined/multi-cell battery or a correlated, non-independent measurement
#: (see AGENTS.md's "pooled inputs must be commensurable" discipline and
#: ``weak_signals.family_overlap_warnings``). Substring, case-insensitive.
_MINED_MARKERS = ("mined", "multiplicit", "predeclared cell", "uncorrected multiplicity")
_CORRELATED_MARKERS = ("correlated decomposition", "not independent", "correlated with")


#: Computed once at import time, not per-row: a frozenset lookup is cheap
#: enough to call for all 480+ rows, but there is no reason to recompute the
#: SAME five-name set that many times over one page build.
_ON_THE_CARD_NAMES = on_the_card_registry_names()


def _status(signal: WeakSignal) -> str:
    # Precedence: a definitive registry verdict (closed) or a declared
    # control arm always wins over "happens to also be a live policy
    # member" -- neither co-occurs in the current registry, but a closed or
    # control-labelled row would be a more important fact than its card
    # membership if it ever did.
    if signal.classification in TERMINAL_CLASSIFICATIONS:
        return STATUS_CLOSED
    if signal.category == "control":
        return STATUS_CONTROL
    if signal.name in _ON_THE_CARD_NAMES:
        return STATUS_ON_CARD
    return STATUS_RECORDED


def _idea_text(signal: WeakSignal) -> tuple[str, bool]:
    """(text, is_fallback) -- the plain summary, or the raw description."""

    if signal.plain_summary and signal.plain_summary.strip():
        return signal.plain_summary.strip(), False
    return signal.description, True


def _caveat_flags(
    signal: WeakSignal, overlapping_names: frozenset[str], duplicate_names: frozenset[str]
) -> list[str]:
    text = " ".join(
        (signal.notes or "", signal.classification_evidence or "", signal.description or "")
    ).lower()
    flags: list[str] = []
    if any(marker in text for marker in _MINED_MARKERS):
        flags.append("mined / uncorrected multiplicity, not a single predeclared test")
    if signal.name in overlapping_names or any(marker in text for marker in _CORRELATED_MARKERS):
        flags.append("shares a measurement window with other recorded signals, not independent")
    if signal.effect_units != _ACCURACY_UNIT:
        _, unit_words = _UNIT_META.get(signal.effect_units, (2, signal.effect_units))
        flags.append(f"measured in {unit_words}, not comparable to accuracy-point rows")
    if signal.name in duplicate_names:
        flags.append(
            "appears twice in the registry under two names, same effect/interval/P+/sample"
        )
    return flags


def _overlapping_names(registry: Registry) -> frozenset[str]:
    """Signals whose measurement window overlaps another in the same
    inferred family -- computed live from the registry (never hardcoded),
    the same accounting :func:`nfl_ats.weak_signals.family_overlap_warnings`
    already does for the ``weak-signals pool`` command."""

    report = family_overlap_warnings(list(registry.signals.values()))
    names: set[str] = set()
    for family in report["within_family"]:
        names.update(family["member_names"])
    return frozenset(names)


def _duplicate_names(registry: Registry) -> frozenset[str]:
    """Signals that are byte-for-byte identical to another recorded signal on
    every quantitative field (league, effect, interval, probability_positive,
    sample_games, sample_blocks) -- e.g. ``body_clock_night_dose_ge2000`` and
    ``body_clock_night_west_road_ge2000et`` (owner, measured 2026-08-26: a
    single genuine pair in 480 rows). This is a REGISTRY DATA QUESTION, not a
    rendering decision: both rows still render, this only adds a caveat flag
    so a reader is not misled into treating them as two independent
    confirmations. Requires interval/probability_positive/sample_games all be
    present so two sparse rows that both happen to be missing the same fields
    never false-positive as "duplicates" of each other.
    """

    fingerprints: dict[tuple[Any, ...], list[str]] = {}
    for signal in registry.signals.values():
        if signal.interval is None or signal.probability_positive is None:
            continue
        if signal.sample_games is None:
            continue
        key = (
            signal.league,
            round(signal.effect, 6),
            round(signal.interval[0], 6),
            round(signal.interval[1], 6),
            round(signal.probability_positive, 6),
            signal.sample_games,
            signal.sample_blocks,
        )
        fingerprints.setdefault(key, []).append(signal.name)
    return frozenset(name for names in fingerprints.values() if len(names) > 1 for name in names)


# ---------------------------------------------------------------------------
# Evidence chip group (owner spec, 2026-08-26): a third filter axis over
# whether a row has actually been CHECKED for repeatability, not just
# whether the effect looks big. Bands are independently derived per row
# (a row can carry more than one -- "never checked" and "found by sweeping"
# are unrelated axes), so the JS filter matches by set-membership, not
# equality, unlike the single-valued status/category groups.
# ---------------------------------------------------------------------------

EVIDENCE_REPEATS_WELL = "repeats_well"
EVIDENCE_DOESNT_REPEAT = "doesnt_repeat"
EVIDENCE_NEVER_CHECKED = "never_checked"
EVIDENCE_FOUND_BY_SWEEPING = "found_by_sweeping"

_EVIDENCE_LABELS: dict[str, str] = {
    EVIDENCE_REPEATS_WELL: "Repeats well",
    EVIDENCE_DOESNT_REPEAT: "Doesn't repeat",
    EVIDENCE_NEVER_CHECKED: "Never checked",
    EVIDENCE_FOUND_BY_SWEEPING: "Found by sweeping",
}

_REPEATS_WELL_MIN = 0.60
_DOESNT_REPEAT_MAX = 0.20

#: Owner-specified, case-insensitive substrings over ``notes``/``description``
#: that flag a mined/multi-cell battery or a non-independent result -- the
#: SAME kind of disclosure :func:`_caveat_flags` looks for on the idea cell,
#: kept as an independently-named list here since the owner gave an exact set
#: (measured against the live registry, 2026-08-26) rather than the looser
#: markers that function already used.
_SWEEP_MARKERS = (
    "mined",
    "multiplicity",
    "selection-inflated",
    "correlated decomposition",
    "not an independent",
    "do not pool",
)


def _evidence_flags(signal: WeakSignal) -> list[str]:
    flags: list[str] = []
    if signal.reliability is None:
        flags.append(EVIDENCE_NEVER_CHECKED)
    else:
        if signal.reliability >= _REPEATS_WELL_MIN:
            flags.append(EVIDENCE_REPEATS_WELL)
        if signal.reliability <= _DOESNT_REPEAT_MAX:
            flags.append(EVIDENCE_DOESNT_REPEAT)
    text = f"{signal.notes or ''} {signal.description or ''}".lower()
    if any(marker in text for marker in _SWEEP_MARKERS):
        flags.append(EVIDENCE_FOUND_BY_SWEEPING)
    return flags


def _row_payload(
    signal: WeakSignal, overlapping_names: frozenset[str], duplicate_names: frozenset[str]
) -> dict[str, Any]:
    idea, is_fallback = _idea_text(signal)
    status = _status(signal)
    category = signal.category if signal.category in CATEGORIES else None
    digits, unit_words = _UNIT_META.get(signal.effect_units, (2, signal.effect_units))
    # Escaped HERE, not in the JS: these two fields are free text out of the
    # registry (a plain summary, a raw description, an internal name), and
    # the client-side renderer concatenates them straight into innerHTML
    # (see ideaCell() in _JS). Every OTHER string the JS renders is either a
    # developer-authored constant (flags, unit words) or a value already
    # constrained to a fixed vocabulary (status, category, league).
    return {
        "name": escape(signal.name),
        "idea": escape(idea),
        "fallback": is_fallback,
        "flags": _caveat_flags(signal, overlapping_names, duplicate_names),
        "effect": signal.effect,
        "units": signal.effect_units,
        "unit_words": unit_words,
        "digits": digits,
        "is_accuracy": signal.effect_units == _ACCURACY_UNIT,
        "interval": None if signal.interval is None else list(signal.interval),
        "pp": signal.probability_positive,
        "rel": signal.reliability,
        "games": signal.sample_games,
        "seasons": f"{signal.seasons[0]}-{signal.seasons[1]}",
        "league": signal.league,
        "status": status,
        "category": category or UNCATEGORISED,
        "evidence": _evidence_flags(signal),
    }


def build_ledger_rows(registry: Registry) -> list[dict[str, Any]]:
    """Every signal in ``registry``, as a JSON-ready row for the page's
    client-side table. Sorted by name for determinism (the page's own JS
    controls the visible order)."""

    overlapping = _overlapping_names(registry)
    duplicates = _duplicate_names(registry)
    return [
        _row_payload(signal, overlapping, duplicates)
        for _, signal in sorted(registry.signals.items())
    ]


def _counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    return {
        "total": len(rows),
        "on_the_card": sum(1 for row in rows if row["status"] == STATUS_ON_CARD),
        "control": sum(1 for row in rows if row["status"] == STATUS_CONTROL),
        "closed": sum(1 for row in rows if row["status"] == STATUS_CLOSED),
        "recorded": sum(1 for row in rows if row["status"] == STATUS_RECORDED),
        "plain_summary": sum(1 for row in rows if not row["fallback"]),
        "categorised": sum(1 for row in rows if row["category"] != UNCATEGORISED),
        "never_checked": sum(1 for row in rows if EVIDENCE_NEVER_CHECKED in row["evidence"]),
        "found_by_sweeping": sum(
            1 for row in rows if EVIDENCE_FOUND_BY_SWEEPING in row["evidence"]
        ),
        "nfl": sum(1 for row in rows if row["league"] == "nfl"),
        "cfb": sum(1 for row in rows if row["league"] == "cfb"),
    }


def _guide_section(counts: dict[str, int]) -> str:
    cards = [
        (
            "Effect on accuracy",
            "Percentage points of forced-pick accuracy across a season, unless the "
            "unit label says otherwise. The bar is centred on zero; the thin "
            "whisker is the 95% interval. Most whiskers cross zero &mdash; at this "
            "evaluator's resolution that is the expected shape for a real small "
            "signal, not a failure.",
        ),
        (
            "Chance it helps",
            "How likely the true effect favours the idea. Not a p-value. Since "
            "every game in this pool must be picked one way or another, anything "
            "above 50% is the side worth taking.",
        ),
        (
            "Repeats?",
            "Split-half reliability: measure the idea on one half of its games, "
            "then the other, and see if they agree. Zero or below means whatever "
            "showed up reverses on the other half.",
        ),
        (
            "What closes an idea",
            "Only two things ever close a line of work here: the effect resolving "
            "to the wrong direction, or zero repeatability. A margin of error "
            "that happens to include zero never does.",
        ),
        (
            '"Never checked"',
            f"{counts['never_checked']:,} of {counts['total']:,} rows below have never had a "
            "split-half repeatability measurement run at all -- the one check that can "
            "legitimately close a line of work. This is a standing work queue, not a "
            "verdict: an unchecked row is not weaker evidence than a checked one, it is "
            "simply not yet checked. Use the Evidence chips to see the pile directly.",
        ),
    ]
    body = "".join(
        '<div class="card"><p class="title" style="font-size:14px;margin:0 0 4px;">'
        f'{escape(title)}</p><p class="fine">{text}</p></div>'
        for title, text in cards
    )
    return f'<div class="row" style="margin-top:8px;">{body}</div>'


def _filter_chip(*, group: str, value: str, label: str, pressed: bool) -> str:
    state = "true" if pressed else "false"
    return (
        f'<button type="button" class="chip" data-group="{escape(group)}" '
        f'data-value="{escape(value)}" aria-pressed="{state}">{escape(label)}</button>'
    )


def _controls_html(counts: dict[str, int]) -> str:
    status_chips = [_filter_chip(group="status", value="all", label="All", pressed=True)]
    status_chips += [
        _filter_chip(group="status", value=value, label=_STATUS_CHIP_LABELS[value], pressed=False)
        for value in _STATUS_META
    ]
    # Labelled "Subject" in the UI (the owner's term); the group key stays
    # "category" internally, matching WeakSignal.category and CATEGORY_LABELS.
    subject_chips = [_filter_chip(group="category", value="all", label="All", pressed=True)]
    subject_chips += [
        _filter_chip(group="category", value=value, label=label, pressed=False)
        for value, label in CATEGORY_LABELS.items()
    ]
    subject_chips.append(
        _filter_chip(
            group="category", value=UNCATEGORISED, label=UNCATEGORISED_LABEL, pressed=False
        )
    )
    evidence_chips = [_filter_chip(group="evidence", value="all", label="All", pressed=True)]
    evidence_chips += [
        _filter_chip(group="evidence", value=value, label=label, pressed=False)
        for value, label in _EVIDENCE_LABELS.items()
    ]
    return (
        '<div class="ledger-controls" style="margin-top:18px;">'
        '<span class="lbl">Status</span>' + "".join(status_chips) + "</div>"
        '<div class="ledger-controls" style="margin-top:8px;">'
        '<span class="lbl">Subject</span>' + "".join(subject_chips) + "</div>"
        '<div class="ledger-controls" style="margin-top:8px;">'
        '<span class="lbl">Evidence</span>' + "".join(evidence_chips) + "</div>"
        '<div class="ledger-controls" style="margin-top:12px;">'
        '<label class="fine" for="ledger-search">Search '
        '<input type="search" id="ledger-search" class="ledger-search" '
        'placeholder="idea text or internal name" style="margin-left:6px;"></label>'
        '<span class="fine" id="ledger-count" style="margin-left:auto;">'
        f"Showing {counts['total']} of {counts['total']}</span></div>"
    )


_COLUMNS = (
    ("idea", "The idea", True),
    ("effect", "Effect on accuracy", True),
    ("pp", "Chance it helps", True),
    ("rel", "Repeats?", True),
    ("games", "Tested on", True),
    ("status", "Status", False),
)


def _table_html() -> str:
    head_cells = []
    for key, label, sortable in _COLUMNS:
        if sortable:
            head_cells.append(
                f'<th scope="col" class="sortable" data-key="{key}" tabindex="0" '
                f'role="columnheader">{escape(label)}<span class="arrow"></span></th>'
            )
        else:
            head_cells.append(f'<th scope="col">{escape(label)}</th>')
    caption_style = (
        "position:absolute;width:1px;height:1px;padding:0;margin:-1px;overflow:hidden;"
        "clip:rect(0,0,0,0);white-space:nowrap;border:0;"
    )
    return (
        '<div style="overflow-x:auto;margin-top:10px;">'
        '<table class="data ledger" id="signal-ledger">'
        f'<caption style="{caption_style}">Every recorded weak-signal experiment: '
        "its plain-English idea, effect on accuracy, chance it helps, "
        "repeatability, sample, and status. Column headers sort the table; the "
        "chips above filter it.</caption>"
        f"<thead><tr>{''.join(head_cells)}</tr></thead>"
        '<tbody id="ledger-body"></tbody></table></div>'
    )


def _script(rows: list[dict[str, Any]]) -> str:
    status_labels_json = json.dumps(
        {value: label for value, (label, _tone) in _STATUS_META.items()},
        separators=(",", ":"),
    )
    status_tones_json = json.dumps(
        {value: tone for value, (_label, tone) in _STATUS_META.items()},
        separators=(",", ":"),
    )
    rows_json = json.dumps(rows, separators=(",", ":"))
    return (
        '<script type="application/json" id="ledger-data">' + rows_json + "</script>\n"
        '<script type="application/json" id="ledger-status-labels">'
        + status_labels_json
        + "</script>\n"
        '<script type="application/json" id="ledger-status-tones">'
        + status_tones_json
        + "</script>\n"
        "<script>\n" + _JS + "</script>\n"
    )


# Vanilla JS, delegated once per page (same pattern as
# ``nfl_ats.dashboard.viz.cover_curve_script`` and the team-explorer matchup
# script): embedded JSON is the single source of truth, sort/filter/search
# rebuild the visible <tbody> from it. No external requests, no libraries.
_JS = """
(function () {
  var dataEl = document.getElementById('ledger-data');
  var labelsEl = document.getElementById('ledger-status-labels');
  var tonesEl = document.getElementById('ledger-status-tones');
  var body = document.getElementById('ledger-body');
  if (!dataEl || !body) { return; }
  var rows, statusLabels, statusTones;
  try {
    rows = JSON.parse(dataEl.textContent);
    statusLabels = labelsEl ? JSON.parse(labelsEl.textContent) : {};
    statusTones = tonesEl ? JSON.parse(tonesEl.textContent) : {};
  } catch (e) { return; }

  // Half-width of the zero-centred effect gauge, in accuracy points. A
  // value outside this clamp still renders its full number (never
  // truncated), with an overflow arrow on the bar -- see effectCell().
  var SCALE = 4.0;
  var state = {
    status: 'all', category: 'all', evidence: 'all', query: '', key: 'effect', dir: -1
  };

  function clamp(v) { return Math.max(-SCALE, Math.min(SCALE, v)); }
  function toPct(v) { return 50 + (clamp(v) / SCALE) * 50; }

  function pillHtml(status) {
    var label = statusLabels[status] || status;
    var tone = statusTones[status] || 'idle';
    return '<span class="pill is-' + tone + '">' + label + '</span>';
  }

  function fmtNum(value, digits) {
    var sign = value >= 0 ? '+' : '-';
    return sign + Math.abs(value).toFixed(digits);
  }

  function effectCell(row) {
    var num = fmtNum(row.effect, row.digits) + ' ' + row.unit_words;
    var tone = row.effect > 0 ? 'pos' : (row.effect < 0 ? 'neg' : 'zero');
    var html = '<span class="ledger-effect"><span class="num delta ' + tone + '">' +
      num + '</span>';
    if (row.is_accuracy) {
      var whisk = '';
      if (row.interval) {
        var a = toPct(row.interval[0]), b = toPct(row.interval[1]);
        var lo = Math.min(a, b), hi = Math.max(a, b);
        whisk = '<span class="whisk" style="left:' + lo.toFixed(1) + '%;width:' +
          (hi - lo).toFixed(1) + '%"></span>';
      }
      var up = row.effect >= 0;
      var edge = toPct(row.effect);
      var barLeft = up ? 50 : edge;
      var barWidth = Math.abs(edge - 50);
      var bar = '<span class="bar ' + (up ? 'up' : 'dn') + '" style="left:' +
        barLeft.toFixed(1) + '%;width:' + barWidth.toFixed(1) + '%"></span>';
      var overflow = '';
      if (Math.abs(row.effect) > SCALE) {
        overflow = '<span class="over" style="' + (up ? 'right:0;' : 'left:0;') + '">' +
          (up ? '→' : '←') + '</span>';
      }
      html += '<span class="ledger-gauge"><span class="rail"></span><span class="zero"></span>' +
        whisk + bar + overflow + '</span>';
    }
    var rangeText = row.interval
      ? fmtNum(row.interval[0], row.digits) + ' to ' + fmtNum(row.interval[1], row.digits)
      : 'no interval recorded';
    html += '<span class="fine">' + rangeText + '</span></span>';
    return html;
  }

  function ppCell(row) {
    if (row.pp === null || row.pp === undefined) {
      return '<span class="fine" style="font-style:italic;">not measured</span>';
    }
    var pct = Math.round(row.pp * 100) + '%';
    var tone = row.pp > 0.5 ? 'pos' : (row.pp < 0.5 ? 'neg' : 'zero');
    var cue = row.pp > 0.5 ? 'favours it' : (row.pp < 0.5 ? 'leans against' : 'coin flip');
    return '<span class="num delta ' + tone + '">' + pct + '</span>' +
      '<span class="fine">' + cue + '</span>';
  }

  function relCell(row) {
    var rel = row.rel;
    if (rel === null || rel === undefined) {
      return '<span class="fine" style="font-style:italic;">not measured</span>';
    }
    var width = Math.max(0, Math.min(1, rel)) * 100;
    var tone = rel <= 0 ? 'bad' : (rel < 0.4 ? 'weak' : '');
    var cue = rel <= 0 ? 'reverses' : (rel < 0.4 ? 'weak' : (rel < 0.75 ? 'holds' : 'strong'));
    return '<span class="ledger-rel"><span class="num">' + rel.toFixed(2) + '</span>' +
      '<span class="track"><span class="fill ' + tone + '" style="width:' +
      width + '%"></span></span>' +
      '<span class="fine">' + cue + '</span></span>';
  }

  function ideaCell(row) {
    var flagsHtml = '';
    if (row.flags && row.flags.length) {
      flagsHtml = '<span class="ledger-flags">' + row.flags.map(function (f) {
        return '<span class="ledger-flag">' + f + '</span>';
      }).join('') + '</span>';
    }
    var ideaClass = row.fallback ? 'fallback' : '';
    var prefix = row.fallback ? '<span class="fine">Raw description &mdash; </span>' : '';
    return '<span class="ledger-idea"><span class="' + ideaClass + '">' +
      prefix + row.idea + '</span>' + flagsHtml +
      '<span class="sub">' + row.name + ' &middot; ' + row.league.toUpperCase() +
      '</span></span>';
  }

  function gamesCell(row) {
    if (row.games === null || row.games === undefined) {
      return '<span class="fine">not recorded</span><span class="fine">' + row.seasons + '</span>';
    }
    return '<span class="num">' + row.games.toLocaleString() + '</span>' +
      '<span class="fine">games &middot; ' + row.seasons + '</span>';
  }

  function render(list) {
    var html = list.map(function (row) {
      return '<tr>' +
        '<td>' + ideaCell(row) + '</td>' +
        '<td>' + effectCell(row) + '</td>' +
        '<td>' + ppCell(row) + '</td>' +
        '<td>' + relCell(row) + '</td>' +
        '<td>' + gamesCell(row) + '</td>' +
        '<td>' + pillHtml(row.status) + '</td>' +
        '</tr>';
    }).join('');
    body.innerHTML = html;
    var countEl = document.getElementById('ledger-count');
    if (countEl) {
      countEl.textContent = 'Showing ' + list.length + ' of ' + rows.length;
    }
  }

  function apply() {
    var q = state.query.trim().toLowerCase();
    var filtered = rows.filter(function (row) {
      if (state.status !== 'all' && row.status !== state.status) { return false; }
      if (state.category !== 'all' && row.category !== state.category) { return false; }
      // Evidence is set-membership, not equality: a row can be BOTH
      // "never checked" (no reliability recorded) and "found by sweeping"
      // (a mining/multiplicity disclosure), which are unrelated axes.
      if (state.evidence !== 'all' && row.evidence.indexOf(state.evidence) === -1) {
        return false;
      }
      if (
        q && row.idea.toLowerCase().indexOf(q) === -1 &&
        row.name.toLowerCase().indexOf(q) === -1
      ) {
        return false;
      }
      return true;
    });
    filtered.sort(function (a, b) {
      var k = state.key;
      if (k === 'idea') { return state.dir * a.idea.localeCompare(b.idea); }
      var x = a[k], y = b[k];
      if (x === null || x === undefined) { return 1; }
      if (y === null || y === undefined) { return -1; }
      if (x < y) { return -state.dir; }
      if (x > y) { return state.dir; }
      return 0;
    });
    render(filtered);
  }

  var buttons = document.querySelectorAll('.ledger-controls .chip');
  for (var i = 0; i < buttons.length; i++) {
    (function (btn) {
      btn.addEventListener('click', function () {
        var group = btn.getAttribute('data-group');
        var value = btn.getAttribute('data-value');
        var siblings = document.querySelectorAll(
          '.ledger-controls .chip[data-group="' + group + '"]'
        );
        for (var j = 0; j < siblings.length; j++) {
          siblings[j].setAttribute('aria-pressed', 'false');
        }
        btn.setAttribute('aria-pressed', 'true');
        state[group] = value;
        apply();
      });
    })(buttons[i]);
  }

  var search = document.getElementById('ledger-search');
  if (search) {
    search.addEventListener('input', function () {
      state.query = search.value;
      apply();
    });
  }

  var headers = document.querySelectorAll('table.ledger th.sortable');
  for (var h = 0; h < headers.length; h++) {
    (function (th) {
      function toggle() {
        var key = th.getAttribute('data-key');
        state.dir = state.key === key ? -state.dir : (key === 'idea' ? 1 : -1);
        state.key = key;
        for (var k = 0; k < headers.length; k++) { headers[k].removeAttribute('aria-sort'); }
        th.setAttribute('aria-sort', state.dir === 1 ? 'ascending' : 'descending');
        apply();
      }
      th.addEventListener('click', toggle);
      th.addEventListener('keydown', function (e) {
        if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); toggle(); }
      });
    })(headers[h]);
  }

  apply();
})();
"""


def build_signal_ledger_body(registry: Registry) -> tuple[str, str]:
    """Compose the ledger page body and its trailing ``<script>`` block.

    Returns ``(body_html, script_html)`` -- the same shape
    ``public_board._team_explorer_matchup`` returns -- so the caller wraps it
    with ``public_board._page(..., scripts=script_html)``.
    """

    from nfl_ats.dashboard import viz

    rows = build_ledger_rows(registry)
    counts = _counts(rows)

    sub = (
        "Every idea this project has tested against the spread, what it was "
        "worth, and how much to trust it. Nothing here is a proven edge -- "
        "results are kept whether they worked or not, which is the point of a "
        "ledger. Regenerated automatically from the live registry on every "
        "site build."
    )
    header = viz.page_header("Research", "Signal ledger", sub=sub)

    tiles = "".join(
        [
            viz.stat_tile(
                "Ideas recorded",
                f"{counts['total']:,}",
                f"{counts['nfl']:,} NFL, {counts['cfb']:,} college -- leagues are never mixed",
            ),
            viz.stat_tile(
                "Plain-English",
                f"{counts['plain_summary']:,} of {counts['total']:,}",
                "carry a written plain summary; the rest show their raw description",
            ),
            viz.stat_tile(
                "Categorised",
                f"{counts['categorised']:,} of {counts['total']:,}",
                "the rest collect under Uncategorised",
            ),
            viz.stat_tile(
                "On the card",
                f"{counts['on_the_card']:,}",
                "evidence behind the live played policy's members",
            ),
            viz.stat_tile(
                "Control arms", f"{counts['control']:,}", "deliberately unplayable checks"
            ),
            viz.stat_tile("Closed", f"{counts['closed']:,}", "refuted or control-bounded"),
        ]
    )
    tile_grid = (
        '<div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));'
        f'gap:14px;margin-top:8px;">{tiles}</div>'
    )

    status_note = (
        '<p class="fine" style="margin-top:10px;max-width:78ch;">'
        f"<b>{counts['on_the_card']}</b> rows below are marked "
        '<span class="pill is-live">On the card</span>: the weak-signal evidence behind each of '
        "the live played policy's four members (<code>coach_fade</code>, "
        "<code>division_revenge_tilt</code>, <code>player_arrests_back_side_policy</code>, "
        "<code>spread_gap_zone_fade</code>; see <code>CURRENT_PREDICTIONS.md</code>), mapped in "
        "code next to the policy definition "
        "(<code>nfl_ats.four_overlay_composition.MEMBER_REGISTRY_EVIDENCE</code>) rather than "
        "read from an artifact field that was null for every arrest-related entry. The "
        "arrest-policy link was verified by an exact match on effect, probability_positive, "
        "seasons and sample size against <code>HANDOFF.md</code>'s promoted-component line, not "
        "a declared source -- see that mapping's own comment for the full chain. There is no "
        "&ldquo;Candidate&rdquo; status: nothing in this codebase distinguishes a candidate from "
        "a merely recorded row, so that chip was dropped rather than shipped empty.</p>"
    )

    guide = _guide_section(counts)
    controls = _controls_html(counts)
    table = _table_html()

    body = "\n".join([header, tile_grid, status_note, guide, controls, table])
    script = _script(rows)
    return body, script


__all__ = [
    "CATEGORY_LABELS",
    "PAGE_FILENAME",
    "PAGE_TITLE",
    "STATUS_CLOSED",
    "STATUS_CONTROL",
    "STATUS_ON_CARD",
    "STATUS_RECORDED",
    "UNCATEGORISED",
    "UNCATEGORISED_LABEL",
    "build_ledger_rows",
    "build_signal_ledger_body",
]
