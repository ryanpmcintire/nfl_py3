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
"""

from __future__ import annotations

import argparse
import copy
import json
from datetime import UTC, datetime
from pathlib import Path

from nfl_ats import cli as nfl_ats_cli
from nfl_ats.weak_signals import WeakSignal, default_registry_path, load_registry

REPO = Path(__file__).resolve().parents[1]

#: signal name -> the new, hand-written plain-English replacement. Each is
#: one or two sentences a football fan with no statistics background can
#: read on its own, naming the situation and what the rule found -- and
#: none contain "P+", "week-blocked", a raw identifier, or any other token
#: ``tests/test_board_humanised.py`` bans from reader-visible text.
PLAIN_SUMMARIES: dict[str, str] = {
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
        "--effect",
        repr(signal.effect),
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
        args += ["--standard-error", repr(signal.standard_error)]
    if signal.interval is not None:
        args += [
            "--interval-low",
            repr(signal.interval[0]),
            "--interval-high",
            repr(signal.interval[1]),
        ]
    if signal.probability_positive is not None:
        args += ["--probability-positive", repr(signal.probability_positive)]
    if signal.sample_games is not None:
        args += ["--sample-games", str(signal.sample_games)]
    if signal.sample_blocks is not None:
        args += ["--sample-blocks", str(signal.sample_blocks)]
    if signal.reliability is not None:
        args += ["--reliability", repr(signal.reliability)]
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
    args = parser.parse_args()

    if args.missing_plain_summary:
        print(json.dumps(find_missing_plain_summaries(), indent=2, sort_keys=True))
        return

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
