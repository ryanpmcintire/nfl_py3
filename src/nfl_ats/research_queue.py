"""ENG-20: the research-queue evidence ledger (ROADMAP.md Phase 13).

Every unresolved roadmap experiment used to live only in prose: its
predeclaration (if any), its required data source, its rotation-registry
window, its linked weak-signal IDs, and what happened last time were spread
across `ROADMAP.md`'s Phase 12 lead queue, the "Recommended execution order"
narrative, and the "Sensitivity-aware review" table -- three places a session
had to read in full, by hand, before it could tell whether a lead was fresh,
already spent, or quietly circular. This module builds one row per tracked
experiment ID by joining those sources programmatically, so "what should run
next" is a table lookup instead of a prose archaeology exercise.

BINDING (AGENTS.md, pasted through, restated at every call site that can
reject or close a line of work): an interval or CI that contains zero is
NEVER grounds to reject, fail, or close an experiment. Only a RESOLVED wrong
sign (whole interval on the wrong side of zero), zero split-half reliability,
or a positive control proven able to detect an effect that size can close a
line; everything else is `unresolved_below_power`. This module therefore:

* never computes or reports "games needed" -- `next_admissible_action` is
  drawn from a fixed six-item vocabulary that does not include waiting;
* never derives a row's classification from whether an interval crosses
  zero -- it reads the classification/verdict the registries themselves
  already recorded (`weak_signals.WeakSignal.classification`,
  `rotation.Window.verdict`), which are validated at write time by
  `weak_signals.signal_from_payload` / `rotation._validate_closing_ground`;
* treats "windows retire PER-FAMILY, not globally; a reused window carries a
  stated discount, not a ban" (`rotation.py`, rule 4) as the correct reading
  of reuse -- `is_circular`/`cross_family_reuse` below flag UNDISCLOSED
  reuse, not reuse itself.

Nothing here scores a model, spends a rotation window, or writes to either
registry. It only reads `ROADMAP.md`, `registry/rotation_registry.json`,
`registry/weak_signals.json`, `registry/experiment_specs/*.json`, and
`scripts/capture_scheduler.py`'s `SCHEDULE`, and reports what it finds.
"""

from __future__ import annotations

import json
import re
import sys
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from nfl_ats import registry_explorer, rotation, weak_signals

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.capture_scheduler import SCHEDULE, Job  # noqa: E402
from scripts.roadmap_inventory import (  # noqa: E402
    STATUS_LABELS,
    RoadmapItem,
    parse_roadmap,
)

RESEARCH_QUEUE_VERSION = 1
DEFAULT_EXPERIMENT_SPECS_DIR = REPO / "registry" / "experiment_specs"

PHASE_12 = "Phase 12 — open lead queue"

#: The fixed vocabulary AGENTS.md requires: "'Next admissible action' must
#: never be 'wait for more data' -- it is one of" these six. Each row's
#: `next_admissible_action` is always exactly one of these literal strings;
#: the specifics ("which window", "which family") live in the paired
#: `next_admissible_action_detail` string so the controlled vocabulary itself
#: never grows free text.
ACTION_RUN_UNSPENT_WINDOW = "run_unspent_window"
ACTION_RUN_REUSED_WINDOW_WITH_DISCOUNT = "run_reused_window_with_discount"
ACTION_TEST_ON_TOP_OF_PRODUCTION = "test_on_top_of_production"
ACTION_RUN_POSITIVE_CONTROL = "run_positive_control"
ACTION_RECORD_PENDING_LOOK = "record_pending_look"
ACTION_CLOSED = "closed"

NEXT_ACTIONS = (
    ACTION_RUN_UNSPENT_WINDOW,
    ACTION_RUN_REUSED_WINDOW_WITH_DISCOUNT,
    ACTION_TEST_ON_TOP_OF_PRODUCTION,
    ACTION_RUN_POSITIVE_CONTROL,
    ACTION_RECORD_PENDING_LOOK,
    ACTION_CLOSED,
)

#: Phrases this ledger must never emit (AGENTS.md, binding). Enforced by
#: `tests/test_research_queue.py::test_generated_output_never_contains_banned_phrases`
#: over the persisted JSON and Markdown, not just this module's own strings.
BANNED_PHRASES = ("more data", "needs n", "contains zero", "failed")

_BACKTICK_TOKEN_RE = re.compile(r"`([a-zA-Z][a-zA-Z0-9_]*)`")
_FULL_ROW_RE = re.compile(
    r"^\| (?P<item_id>[A-Z]+-\d+) \| (?P<status>✅|🚧|⬜|🔬|🌙|❌) \| [^|]+ \| (?P<dod>.*) \|$"
)

# --------------------------------------------------------------------------
# Roadmap row text (reuses scripts/roadmap_inventory.py's item/status/phase
# parser for the fields it already extracts; adds only the "definition of
# done" column text that parser intentionally does not capture).
# --------------------------------------------------------------------------


def dod_text_by_item(roadmap_text: str) -> dict[str, str]:
    """Map each roadmap item ID to its full "definition of done" column text.

    `roadmap_inventory.ROW_RE` deliberately stops after the title column; the
    family names, artifact paths, and registry IDs this ledger joins against
    live in the DoD column past that point, so this re-scans the same lines
    with the DoD column included rather than modifying the shared parser.
    """

    result: dict[str, str] = {}
    for line in roadmap_text.splitlines():
        match = _FULL_ROW_RE.match(line)
        if match is not None:
            result[match.group("item_id")] = match.group("dod").strip()
    return result


def backtick_tokens(*texts: str) -> tuple[str, ...]:
    """Every `snake_case`-style identifier quoted in backticks, in order of
    first appearance, deduplicated. This is how a row's own prose links
    itself to a rotation family, a weak-signal name, or a predeclaration
    spec: every example in the live roadmap already writes those names in
    backticks (e.g. LEAD-45: "Recorded `cfb_option_side_on_benchmark`").
    """

    seen: dict[str, None] = {}
    for text in texts:
        for token in _BACKTICK_TOKEN_RE.findall(text):
            seen.setdefault(token, None)
    return tuple(seen)


# --------------------------------------------------------------------------
# Capture-source mapping (scripts/capture_scheduler.py SCHEDULE).
# --------------------------------------------------------------------------

_DAY_TOKENS = frozenset({"mon", "tue", "wed", "thu", "fri", "sat", "sun"})
_SLOT_TOKENS = frozenset(
    {"open", "close", "late", "early", "checkpoint", "primetime", "afternoon", "tnf", "mnf"}
)
# weekly_lock replays the paper-forecast pipeline on top of already-captured
# data, and every refresh_* job re-scores picks with `refresh-picks`; neither
# captures a raw external source, so both are excluded from the source-family
# vocabulary a roadmap row's "required source" can resolve to.
_NON_SOURCE_JOB_NAMES = frozenset({"weekly_lock"})
_NON_SOURCE_JOB_PREFIXES = ("refresh_",)


def _job_family(name: str) -> str:
    tokens = name.split("_")
    while len(tokens) > 1 and (tokens[-1] in _DAY_TOKENS or tokens[-1] in _SLOT_TOKENS):
        tokens.pop()
    return "_".join(tokens)


def capture_job_families(jobs: Iterable[Job]) -> dict[str, bool]:
    """``{source_family: any_job_in_that_family_enabled}`` from SCHEDULE.

    Job names encode a day/slot suffix (``odds_sun_close``, ``inactives_thu_
    afternoon_early``); this collapses them to the source family the row-level
    keyword matcher below resolves to (``odds``, ``inactives``, ...).
    """

    families: dict[str, bool] = {}
    for job in jobs:
        if job.name in _NON_SOURCE_JOB_NAMES or any(
            job.name.startswith(prefix) for prefix in _NON_SOURCE_JOB_PREFIXES
        ):
            continue
        family = _job_family(job.name)
        families[family] = families.get(family, False) or job.enabled
    return families


# Ordered, most-specific-first: a row mentioning both "spread" and "arrest"
# language is vanishingly unlikely, but "designation"/"practice status" (an
# injuries-report term) must not fall through to the generic "odds" bucket
# just because a row also mentions the market. Best-effort keyword classifier
# -- unmapped text resolves to `None` ("unknown"), never a guess dressed up as
# a fact.
SOURCE_KEYWORDS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("player_arrests", ("arrest",)),
    (
        "referee_assignments",
        ("referee", "officiating crew", "crew chief", "penalty crew", "umpire"),
    ),
    (
        "pfr_transactions",
        (
            "transaction wire",
            "trade deadline",
            "waiver",
            "acquisition",
            "holdout",
            "suspension",
            "roster move",
            "pfr transaction",
        ),
    ),
    ("inactives", ("inactive list", "t-90", "inactives")),
    (
        "injuries",
        (
            "injury report",
            "designation",
            "practice status",
            "dnp",
            "questionable",
            "concussion",
            "illness",
            "beat-writer",
            "coach-speak",
        ),
    ),
    ("public_betting", ("bet%", "money%", "public betting", "handle")),
    ("lineups", ("depth chart", "starting qb", "lineup")),
    ("airnow", ("air quality", "aqi")),
    (
        "odds",
        (
            "spread",
            "moneyline",
            "line movement",
            "closing line",
            "opener",
            "book",
            "steam",
            "quote",
            "hourly archive",
            "total line",
        ),
    ),
)


def guess_required_source(text: str) -> str | None:
    """Best-effort keyword match from a row's own text to a capture-job family.

    Returns ``None`` -- rendered ``"unknown"`` in the table -- when nothing
    matches, per ENG-20's explicit instruction: "unknown if unmappable"
    rather than a fabricated guess.
    """

    lowered = text.lower()
    for family, keywords in SOURCE_KEYWORDS:
        if any(keyword in lowered for keyword in keywords):
            return family
    return None


def registry_explorer_source_citation(family_name: str | None) -> str:
    """Citation-backed source note from ``registry_explorer.FAMILY_SOURCE_RULES``.

    ENG-07's table is hand-verified (each row cites the exact file it was
    read from) and keyed to already-declared families, which is a strictly
    higher-confidence source than this module's own keyword guesser
    (``guess_required_source``) -- but it only covers families that already
    exist in a registry, so it can never replace the guesser for a brand-new,
    not-yet-declared Phase 12 lead. Used as a supplementary citation
    alongside, not instead of, ``required_source``/``source_captured_today``.
    Returns ``"none"`` when ``family_name`` is unset or no rule's prefix
    matches, mirroring ``registry_explorer._source_for_family``'s own
    longest-prefix-wins rule without importing that private helper.
    """

    if not family_name:
        return "none"
    matches = [
        rule
        for rule in registry_explorer.FAMILY_SOURCE_RULES
        if family_name == rule.prefix or family_name.startswith(rule.prefix)
    ]
    if not matches:
        return "none"
    best = max(matches, key=lambda rule: len(rule.prefix))
    return f"{best.status}: {best.citation}"


def guess_grade(text: str) -> str:
    """Which rotation grade pool a not-yet-declared family would draw from.

    Defaults to ``close`` (the broad 2009-2025 pool) unless the row's own
    text names the ``opener`` grade explicitly, matching
    ``rotation.GRADE_POOLS``'s two commonly-used pools.
    """

    lowered = text.lower()
    if "opener" in lowered:
        return "opener"
    if "nflverse" in lowered:
        return "nflverse_spread"
    return "close"


# --------------------------------------------------------------------------
# Circular-run guard (rotation.py rule 4 / AGENTS.md).
# --------------------------------------------------------------------------

_DISCLOSURE_MARKERS = ("disclosed", "discount", "acknowledg")


def _notes_disclose_reuse(*notes: str) -> bool:
    lowered = " ".join(notes).lower()
    return any(marker in lowered for marker in _DISCLOSURE_MARKERS)


def is_circular(family: rotation.Family, window: rotation.Window) -> bool:
    """True when ``family`` has already looked at ``window``'s seasons itself.

    ``rotation._validate`` already refuses to let a *loaded* ledger hold two
    overlapping windows for one family (the "has overlapping windows"
    ``RegistryError``) -- this is the same rule expressed as a pure predicate
    a caller can check on a *candidate* window before ever calling
    ``assign_window``. Unlike cross-family reuse (see ``cross_family_reuse``
    below), a family re-scoring seasons it has already personally spent is
    always circular; AGENTS.md's "a reused window carries a stated discount,
    not a ban" is about a DIFFERENT family drawing an already-touched season
    pool, never about one family looking at its own history twice. A
    disclosure note attached to either window is still honoured, so a
    deliberately-documented re-read (e.g. a correction) is not flagged.
    """

    for existing in family.windows:
        if existing is window:
            continue
        if set(window.covered_seasons) & set(existing.covered_seasons):
            return not _notes_disclose_reuse(window.notes, existing.notes)
    return False


def cross_family_reuse(
    registry: rotation.Registry, family_name: str, window: rotation.Window
) -> bool:
    """True when ``window`` reuses seasons a DIFFERENT family already spent,
    with no disclosure of the overlap in ``window.notes``.

    rotation.py rule 4 / AGENTS.md: cross-family reuse is explicitly
    permitted ("windows retire PER-FAMILY, not globally") -- this is not a
    violation by itself, and callers should read it as "reuse, undisclosed"
    rather than "circular". It exists so a reused block is visible in the
    table exactly the way ``best_pick_ranker``'s own window notes already
    disclose one by hand ("Seasons previously mined by pbp_drive_bundle
    [2013, 2017] (rule 4 overlap, disclosed)").
    """

    if family_name not in registry.families:
        raise rotation.RegistryError(f"Unknown family: {family_name!r}")
    chain = {family_name}
    pending = list(registry.families[family_name].inherits)
    while pending:
        parent = pending.pop()
        if parent in chain:
            continue
        chain.add(parent)
        if parent in registry.families:
            pending.extend(registry.families[parent].inherits)

    touched = set(window.covered_seasons)
    for other_name, other_family in registry.families.items():
        if other_name in chain:
            continue
        for other_window in other_family.windows:
            if other_window.state != "spent":
                continue
            if touched & set(other_window.covered_seasons):
                return not _notes_disclose_reuse(window.notes)
    return False


def _reuse_flag(registry: rotation.Registry, family_name: str, family: rotation.Family) -> bool:
    return any(
        is_circular(family, window) or cross_family_reuse(registry, family_name, window)
        for window in family.windows
    )


# --------------------------------------------------------------------------
# Next admissible action.
# --------------------------------------------------------------------------


def _admissible_closed_window(family: rotation.Family) -> rotation.Window | None:
    for window in family.windows:
        if (
            window.state == "spent"
            and window.verdict == "closed_negative"
            and window.closing_ground
        ):
            return window
    return None


def _latest_spent_window(family: rotation.Family) -> rotation.Window | None:
    spent = [window for window in family.windows if window.state == "spent" and window.spent_at]
    if not spent:
        return None
    return max(spent, key=lambda window: window.spent_at or "")


def _has_recorded_positive_control(family: rotation.Family) -> bool:
    return any(
        "positive control" in window.notes.lower()
        or "positive_control" in (window.artifact or "").lower()
        for window in family.windows
    )


def next_admissible_action(
    registry: rotation.Registry,
    *,
    family_name: str | None,
    grade_guess: str,
    weak_signal: weak_signals.WeakSignal | None,
) -> tuple[str, str]:
    """Return ``(action, detail)`` where ``action`` is one of ``NEXT_ACTIONS``.

    Decision order, all drawn from the registries actually loaded (never from
    a games-needed calculation, per AGENTS.md):

    1. An admissible ``closed_negative`` verdict already recorded -> ``closed``.
    2. A currently-assigned, not-yet-recorded window -> ``record_pending_look``.
    3. A fresh block still eligible in this family's grade pool ->
       ``run_unspent_window``.
    4. No fresh block, and this family has never been measured on top of
       production -> ``test_on_top_of_production``.
    5. No fresh block, already on-production, unresolved, and no positive
       control has been sized yet -> ``run_positive_control``.
    6. No fresh block and everything above is exhausted ->
       ``run_reused_window_with_discount`` (rule 4: reuse is permitted with a
       stated discount, never a ban).

    When no rotation family has been declared for the row yet, the same
    six-item vocabulary still applies: a terminal weak-signal classification
    with an admissible closing ground is ``closed``; otherwise the row's
    admissible next step is to declare a family and draw the earliest
    globally unspent block in its guessed grade pool (``run_unspent_window``),
    or, if the whole pool is already spent, to reuse one with the discount.
    """

    if family_name is not None and family_name in registry.families:
        family = registry.families[family_name]
        closed = _admissible_closed_window(family)
        if closed is not None:
            return ACTION_CLOSED, (
                f"{family_name} closed_negative on {list(closed.seasons)} "
                f"({closed.closing_ground}; rotation.py CLOSING_GROUNDS taxonomy)"
            )
        if family.assigned_window is not None:
            window = family.assigned_window
            return ACTION_RECORD_PENDING_LOOK, (
                f"{family_name} holds an unspent assigned window {list(window.seasons)}; "
                "run it and call rotation.record_look"
            )
        blocks = rotation.eligible_blocks(registry, family_name)
        if blocks:
            return ACTION_RUN_UNSPENT_WINDOW, (
                f"{family_name}: draw the earliest eligible {family.grade} block {list(blocks[0])}"
            )
        already_on_production = "on_production" in family_name or "production" in (
            family.description.lower()
        )
        if not already_on_production:
            return ACTION_TEST_ON_TOP_OF_PRODUCTION, (
                f"{family_name}: has a screened/close-graded result but no look measured "
                "on top of the production `weak_stack` chain yet"
            )
        latest = _latest_spent_window(family)
        if (
            latest is not None
            and latest.verdict == "unresolved"
            and not _has_recorded_positive_control(family)
        ):
            return ACTION_RUN_POSITIVE_CONTROL, (
                f"{family_name}: {family.grade} pool exhausted, on-production look "
                "unresolved, no positive control sized for this family yet -- run a "
                "candidate-sized one before any closure is admissible"
            )
        return ACTION_RUN_REUSED_WINDOW_WITH_DISCOUNT, (
            f"{family_name}: {family.grade} pool exhausted (remaining_eligible_windows=0); "
            "reuse an already-spent block with the AGENTS.md-mandated stated discount"
        )

    if (
        weak_signal is not None
        and weak_signal.classification in weak_signals.TERMINAL_CLASSIFICATIONS
        and weak_signal.closing_ground is not None
    ):
        return ACTION_CLOSED, (
            f"{weak_signal.name} closed on {weak_signal.classification} "
            f"({weak_signal.closing_ground})"
        )

    capacity = rotation.grade_pool_capacity(registry)
    unspent_blocks = capacity.get(grade_guess, {}).get("unspent_blocks", [])
    if unspent_blocks:
        return ACTION_RUN_UNSPENT_WINDOW, (
            f"declare a rotation family and draw the earliest unspent {grade_guess} block "
            f"{unspent_blocks[0]}"
        )
    return ACTION_RUN_REUSED_WINDOW_WITH_DISCOUNT, (
        f"no fresh {grade_guess} block remains globally; declare the family and reuse an "
        "already-spent block, stating the discount"
    )


# --------------------------------------------------------------------------
# Row assembly.
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class QueueRow:
    """One roadmap experiment's predeclaration, source, window, and next step."""

    item_id: str
    status: str
    status_label: str
    title: str
    phase: str
    predeclaration_spec: str
    required_source: str
    source_captured_today: str
    source_citation: str
    rotation_family: str
    rotation_grade: str
    windows_used: tuple[str, ...]
    windows_unspent: int
    weak_signal_ids: tuple[str, ...]
    last_attempt: str
    last_attempt_source: str
    last_attempt_classification: str
    reuse_flag: bool
    next_admissible_action: str
    next_admissible_action_detail: str


def _match_rotation_family(tokens: Sequence[str], registry: rotation.Registry) -> str | None:
    for token in tokens:
        if token in registry.families:
            return token
    return None


def _match_weak_signals(tokens: Sequence[str], registry: weak_signals.Registry) -> tuple[str, ...]:
    return tuple(token for token in tokens if token in registry.signals)


def _match_predeclaration_spec(tokens: Sequence[str], spec_names: frozenset[str]) -> str:
    for token in tokens:
        if token in spec_names:
            return f"registry/experiment_specs/{token}.json"
    return "none"


def _last_attempt(
    family: rotation.Family | None,
    matched_signals: Sequence[weak_signals.WeakSignal],
) -> tuple[str, str, str]:
    """Return ``(last_attempt, source, classification)``.

    ``last_attempt`` is a registry timestamp only -- never derived from
    roadmap prose -- so ``"never"`` here is a true statement: neither
    registry has recorded a look for this row yet, not merely "not found by
    this script".
    """

    candidates: list[tuple[str, str, str]] = []
    if family is not None:
        window = _latest_spent_window(family)
        if window is not None and window.spent_at is not None:
            candidates.append((window.spent_at, "rotation_family", window.verdict or "unresolved"))
    if matched_signals:
        latest_signal = max(matched_signals, key=lambda signal: signal.recorded_at)
        candidates.append((latest_signal.recorded_at, "weak_signal", latest_signal.classification))
    if not candidates:
        return "never", "none", "not_yet_run"
    return max(candidates, key=lambda candidate: candidate[0])


def _select_rows(
    items: Sequence[RoadmapItem],
    registry: rotation.Registry,
    dod_by_id: dict[str, str],
) -> list[RoadmapItem]:
    selected: list[RoadmapItem] = []
    for item in items:
        if item.phase == PHASE_12:
            selected.append(item)
            continue
        if item.status not in {"🚧", "⬜", "🔬"}:
            continue
        tokens = backtick_tokens(item.title, dod_by_id.get(item.item_id, ""))
        if any(token in registry.families for token in tokens):
            selected.append(item)
    return selected


def build_queue(
    roadmap_text: str,
    rotation_registry: rotation.Registry,
    weak_signal_registry: weak_signals.Registry,
    *,
    experiment_spec_names: frozenset[str] = frozenset(),
    jobs: Sequence[Job] = SCHEDULE,
) -> list[QueueRow]:
    """Build one :class:`QueueRow` per selected roadmap experiment ID.

    Selected rows are every Phase 12 lead (regardless of status) plus every
    other 🚧/⬜/🔬 row that names a declared rotation family in its own text
    (ENG-20's definition of done). ``experiment_spec_names`` and ``jobs`` are
    injectable so tests never touch the real filesystem or the real schedule.
    """

    items = parse_roadmap(roadmap_text)
    dod_by_id = dod_text_by_item(roadmap_text)
    capture_families = capture_job_families(jobs)
    selected = _select_rows(items, rotation_registry, dod_by_id)

    rows: list[QueueRow] = []
    for item in selected:
        dod_text = dod_by_id.get(item.item_id, "")
        combined_text = f"{item.title} {dod_text}"
        tokens = backtick_tokens(item.title, dod_text)

        family_name = _match_rotation_family(tokens, rotation_registry)
        family = rotation_registry.families.get(family_name) if family_name else None

        weak_signal_names = _match_weak_signals(tokens, weak_signal_registry)
        matched_signals = [weak_signal_registry.signals[name] for name in weak_signal_names]
        representative_signal = (
            max(matched_signals, key=lambda signal: signal.recorded_at) if matched_signals else None
        )

        predeclaration_spec = _match_predeclaration_spec(tokens, experiment_spec_names)

        required_source = guess_required_source(combined_text)
        if required_source is None:
            source_captured_today = "unknown"
            required_source_label = "unknown"
        else:
            required_source_label = required_source
            source_captured_today = "yes" if capture_families.get(required_source) else "no"
            if required_source not in capture_families:
                source_captured_today = "unknown"

        source_citation = registry_explorer_source_citation(family_name)

        grade_guess = family.grade if family is not None else guess_grade(combined_text)

        if family is not None and family_name is not None:
            windows_used = tuple(
                f"{list(window.seasons)}:{window.verdict or window.state}"
                for window in family.windows
            )
            windows_unspent = len(rotation.eligible_blocks(rotation_registry, family_name))
            reuse = _reuse_flag(rotation_registry, family_name, family)
        else:
            windows_used = ()
            capacity = rotation.grade_pool_capacity(rotation_registry)
            windows_unspent = len(capacity.get(grade_guess, {}).get("unspent_blocks", []))
            reuse = False

        last_attempt, last_attempt_source, last_attempt_classification = _last_attempt(
            family, matched_signals
        )

        action, action_detail = next_admissible_action(
            rotation_registry,
            family_name=family_name,
            grade_guess=grade_guess,
            weak_signal=representative_signal,
        )

        rows.append(
            QueueRow(
                item_id=item.item_id,
                status=item.status,
                status_label=STATUS_LABELS[item.status],
                title=item.title,
                phase=item.phase,
                predeclaration_spec=predeclaration_spec,
                required_source=required_source_label,
                source_captured_today=source_captured_today,
                source_citation=source_citation,
                rotation_family=family_name or "none",
                rotation_grade=grade_guess,
                windows_used=windows_used,
                windows_unspent=windows_unspent,
                weak_signal_ids=weak_signal_names,
                last_attempt=last_attempt,
                last_attempt_source=last_attempt_source,
                last_attempt_classification=last_attempt_classification,
                reuse_flag=reuse,
                next_admissible_action=action,
                next_admissible_action_detail=action_detail,
            )
        )
    return rows


def load_experiment_spec_names(specs_dir: Path = DEFAULT_EXPERIMENT_SPECS_DIR) -> frozenset[str]:
    """Every predeclared experiment spec's file stem (its declared ``name``)."""

    if not specs_dir.is_dir():
        return frozenset()
    return frozenset(path.stem for path in specs_dir.glob("*.json"))


# --------------------------------------------------------------------------
# Persistence (registry/research_queue.json, docs/research_queue.md).
# --------------------------------------------------------------------------

MARKDOWN_HEADER = """# Research queue evidence ledger

**Generated by `scripts/research_queue.py`. Do not hand-edit; run the script
to refresh this file.** One row per tracked roadmap experiment ID (every
Phase 12 lead, plus every other 🚧/⬜/🔬 row that names a declared rotation
family), joining its predeclaration spec, required data source and whether
it is captured today, rotation-registry window history, linked weak-signal
IDs, its last attempt and classification, an undisclosed-reuse flag, and its
next admissible action -- the fixed vocabulary AGENTS.md requires instead of
an instruction to keep waiting: `run_unspent_window`,
`run_reused_window_with_discount`, `test_on_top_of_production`,
`run_positive_control`, `record_pending_look`, or `closed` (only with an
admissible closing ground already on record).

An interval or CI crossing zero is never grounds to reject, fail, or close a
row here (AGENTS.md, binding); classifications below are read verbatim from
`registry/rotation_registry.json` / `registry/weak_signals.json`, never
re-derived from an interval's sign.
"""


def queue_payload(rows: Sequence[QueueRow]) -> dict[str, Any]:
    """The deterministic JSON payload for ``registry/research_queue.json``.

    Carries no wall-clock timestamp: the payload is a pure function of
    `ROADMAP.md` plus the two registries plus the capture schedule, so
    `scripts/research_queue.py --check` can compare a fresh build to the
    committed file byte-for-byte without every run reporting stale on clock
    drift alone.
    """

    raw = {
        "version": RESEARCH_QUEUE_VERSION,
        "row_count": len(rows),
        "rows": [asdict(row) for row in rows],
    }
    # `asdict` preserves each field's own container type, so tuple fields
    # (windows_used, weak_signal_ids) stay tuples here. A round-trip through
    # JSON normalises them to lists -- exactly what `json.loads` produces
    # when `--check` reads the committed file back -- so the two are directly
    # comparable by `==` without a tuple/list mismatch reporting spurious
    # staleness on every single run.
    normalized: dict[str, Any] = json.loads(json.dumps(raw))
    return normalized


def _markdown_row(row: QueueRow) -> str:
    windows = "; ".join(row.windows_used) if row.windows_used else "none"
    signals = ", ".join(f"`{name}`" for name in row.weak_signal_ids) or "none"
    source = f"{row.required_source} (captured: {row.source_captured_today}; {row.source_citation})"
    last_attempt = (
        f"{row.last_attempt} [{row.last_attempt_source}] -> {row.last_attempt_classification}"
    )
    action = f"**{row.next_admissible_action}** -- {row.next_admissible_action_detail}"
    return (
        f"| {row.item_id} | {row.status} {row.status_label} | {row.title} | {row.phase} | "
        f"{row.predeclaration_spec} | {source} | {row.rotation_family} ({row.rotation_grade}) | "
        f"{windows} | unspent={row.windows_unspent} | {signals} | {last_attempt} | "
        f"{row.reuse_flag} | {action} |"
    )


def queue_markdown(rows: Sequence[QueueRow]) -> str:
    """Render the tracked, generated Markdown view of ``rows``."""

    lines = [
        MARKDOWN_HEADER,
        "| ID | Status | Title | Phase | Predeclaration | Required source | Rotation family "
        "(grade) | Windows used | Windows unspent | Weak-signal IDs | Last attempt | Reuse flag "
        "| Next admissible action |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    lines.extend(_markdown_row(row) for row in rows)
    lines.append("")
    return "\n".join(lines)
