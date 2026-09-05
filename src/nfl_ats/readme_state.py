"""Generated README sections for state that goes stale between sessions.

``nfl_ats.publishing`` already owns one self-updating README region -- the
``<!-- CURRENT_PREDICTIONS:START -->`` / ``:END`` block that
``publish_active_predictions`` rewrites from the active weekly forecast. This
module extends that exact marked-block pattern to two more numbers that were
observed drifting silently in hand-written prose: which model is active and
what it actually grades at, and how big the weak-signal/rotation/prospective
research registries are. Both blocks are rendered from the artifacts and
registries that are already each subsystem's single source of truth, so the
README can never again quote a number that contradicts them.

``artifacts/`` is gitignored (only ``active_ats_model.json`` and the linked
``opener_evaluation/`` run back the active-model block), so a fresh clone has
none of it. Every render function here is required to degrade to an honest
"not built in this clone" sentence rather than raise or print a stale number;
only a genuinely malformed manifest (already-active-in-production behaviour
in :mod:`nfl_ats.handoff`) is allowed to propagate as an error.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

from nfl_ats import rotation, weak_signals
from nfl_ats.active_model import load_active_ats_model
from nfl_ats.io import atomic_text
from nfl_ats.prospective_scoring import ACTIVE_CHALLENGER_STATUS, load_challenger_registry
from nfl_ats.public_board import load_baseline_measurement

README_ACTIVE_MODEL_START = "<!-- ACTIVE_MODEL_STATE:START -->"
README_ACTIVE_MODEL_END = "<!-- ACTIVE_MODEL_STATE:END -->"
README_RESEARCH_STATE_START = "<!-- RESEARCH_STATE:START -->"
README_RESEARCH_STATE_END = "<!-- RESEARCH_STATE:END -->"


def _uncertainty_interval(
    metadata: dict[str, Any], metric: str, *, block: str = "week"
) -> tuple[float, float] | None:
    for row in metadata.get("uncertainty", []):
        if not isinstance(row, dict):
            continue
        if row.get("metric") != metric or row.get("block") != block:
            continue
        lower, upper = row.get("lower"), row.get("upper")
        if isinstance(lower, (int, float)) and isinstance(upper, (int, float)):
            return float(lower), float(upper)
    return None


def render_active_model_block(artifacts_root: Path) -> str:
    """The active model's identity and its two accuracy grades.

    A manifest that fails to load (unsupported version, ``status`` not
    ``SYNCHRONIZED``) raises -- matching the existing, unguarded
    ``load_active_ats_model`` call in ``nfl_ats.handoff._model_markdown`` --
    because that is a real data-integrity defect, not the expected
    fresh-clone absence.
    """

    active = load_active_ats_model(artifacts_root)
    if active is None:
        return (
            "Local model artifacts are not built in this clone (`artifacts/` is "
            "gitignored and starts empty). Run the reproduction commands under "
            "Quick start, then `nfl-ats publish-predictions`, to populate "
            "`artifacts/active_ats_model.json` and regenerate this section."
        )
    historical = active["historical_evaluation"]
    week_interval = historical.get("intervals", {}).get("week", {})
    lines = [
        f"Active model: `{active['method']}` with `{active['feature_profile']}` "
        f"features (`{active['model_id']}`), regressor `{active['regressor']}`, "
        f"ridge alpha `{active.get('ridge_alpha', 10.0)}`, calibration "
        f"`{active.get('calibration_method', 'none')}`.",
        "",
    ]
    try:
        opener = load_baseline_measurement(artifacts_root, active)
    except ValueError:
        opener = None
    if opener is None:
        lines.append(
            "- **Opener-graded, probability-rule accuracy (the pool-relevant "
            "grade -- picks lock Tuesday against a frozen line):** "
            "**unavailable in local artifacts** (no `artifacts/opener_evaluation/` "
            "run matches this active model's recipe; run `nfl-ats "
            "opener-evaluation` to produce one)."
        )
    else:
        opener_accuracy = opener.accuracy
        opener_games = opener.games
        opener_interval = opener.week_interval
        interval_text = (
            f", week-blocked 95% interval [{opener_interval[0]:.2%}, {opener_interval[1]:.2%}]"
            if opener_interval is not None
            else ""
        )
        lines.append(
            "- **Opener-graded, probability-rule accuracy (the pool-relevant "
            f"grade -- picks lock Tuesday against a frozen line):** "
            f"**{opener_accuracy:.2%}** on **{opener_games:,} paired games**"
            f"{interval_text}."
        )
    week_interval_text = (
        f", week-blocked 95% interval [{week_interval['lower']:.2%}, {week_interval['upper']:.2%}]"
        if isinstance(week_interval.get("lower"), (int, float))
        and isinstance(week_interval.get("upper"), (int, float))
        else ""
    )
    lines.append(
        "- Close-graded accuracy (secondary -- the market's sharpest, and "
        "least representative, decision point): "
        f"**{historical['accuracy']:.2%}** ({historical['correct']:,} of "
        f"{historical['games']:,} non-push games){week_interval_text}."
    )
    lines.append("")
    lines.append(
        "Neither figure is a game-specific probability, and neither is proof of "
        "a profitable or stable market edge (see `AGENTS.md`)."
    )
    return "\n".join(lines)


def _weak_signal_summary_line(registry_root: Path) -> str:
    path = weak_signals.default_registry_path(registry_root)
    try:
        registry = weak_signals.load_registry(path)
    except (OSError, ValueError):
        return (
            "- **Weak-signal registry:** not available in this clone "
            "(`registry/weak_signals.json` missing or unreadable)."
        )
    total = len(registry.signals)
    if total == 0:
        return "- **Weak-signal registry:** 0 results recorded yet."
    unresolved = sum(
        1
        for signal in registry.signals.values()
        if signal.classification == weak_signals.POOLABLE_CLASSIFICATION
    )
    terminal_counts = Counter(
        signal.classification
        for signal in registry.signals.values()
        if signal.classification in weak_signals.TERMINAL_CLASSIFICATIONS
    )
    closed = total - unresolved
    breakdown = ", ".join(
        f"{terminal_counts.get(name, 0)} {name}" for name in weak_signals.TERMINAL_CLASSIFICATIONS
    )
    return (
        f"- **Weak-signal registry:** {total:,} results recorded -- {unresolved:,} "
        f"unresolved_below_power, {closed:,} closed ({breakdown}). An interval "
        "crossing zero is never by itself grounds to close a line of work; see "
        "`AGENTS.md`."
    )


def _rotation_summary_line(registry_root: Path) -> str:
    path = registry_root / rotation.ROTATION_REGISTRY_FILENAME
    try:
        registry = rotation.load_registry(path)
    except (OSError, ValueError):
        return (
            "- **Rotation registry:** not available in this clone "
            "(`registry/rotation_registry.json` missing or unreadable)."
        )
    total = len(registry.families)
    if total == 0:
        return "- **Rotation registry:** 0 declared research families yet."
    counts: dict[str, int] = {}
    for family in registry.families.values():
        counts[family.status] = counts.get(family.status, 0) + 1
    open_count = counts.get("open", 0)
    stub_count = counts.get(rotation.COVERAGE_STUB_STATUS, 0)
    # Count the terminal statuses explicitly rather than as "everything that is
    # not open": coverage stubs are neither open nor closed (ENG-37).
    terminal_count = sum(
        counts.get(status, 0) for status in ("confirmed", "closed_negative", "retired")
    )
    line = (
        f"- **Rotation registry:** {total:,} declared research families -- "
        f"{open_count:,} open, {terminal_count:,} confirmed/closed/retired"
    )
    if stub_count:
        line += f", {stub_count:,} declared for coverage only (no window yet)"
    return line + "."


def _challenger_summary_line(artifacts_root: Path) -> str:
    try:
        payload = load_challenger_registry(artifacts_root)
    except (OSError, ValueError):
        return (
            "- **Prospective challengers:** not available in this clone "
            "(`artifacts/prospective/challengers.json` missing or unreadable)."
        )
    entries = [entry for entry in payload.get("challengers", []) if isinstance(entry, dict)]
    total = len(entries)
    active = sum(1 for entry in entries if entry.get("status") == ACTIVE_CHALLENGER_STATUS)
    return (
        f"- **Prospective challengers:** {active:,} of {total:,} registered "
        "challengers are actively tracked prospectively "
        "(`artifacts/prospective/challengers.json`)."
    )


def render_research_state_block(registry_root: Path | None, artifacts_root: Path) -> str:
    """A few-line summary of how large the research registries are right now."""

    if registry_root is None:
        return (
            "Research registry state is not available in this render (no "
            "registry root was configured)."
        )
    return "\n".join(
        (
            _weak_signal_summary_line(registry_root),
            _rotation_summary_line(registry_root),
            _challenger_summary_line(artifacts_root),
        )
    )


def _replace_marked_section(text: str, start: str, end: str, content: str) -> str:
    """Rewrite the text between ``start``/``end``, matching ``publishing``'s pattern.

    Mirrors ``nfl_ats.publishing._replace_readme_section``: an existing pair is
    replaced in place; a first-time-setup README missing the pair gets the
    section appended rather than the render failing outright. A malformed
    README (the pair present more than once, or only one of the two markers)
    is a real structural defect and raises.
    """

    block = f"{start}\n{content.rstrip()}\n{end}"
    if start in text or end in text:
        if text.count(start) != 1 or text.count(end) != 1:
            raise ValueError(f"README markers {start!r}/{end!r} must appear exactly once as a pair")
        before, remainder = text.split(start, maxsplit=1)
        _, after = remainder.split(end, maxsplit=1)
        return before.rstrip() + "\n\n" + block + after
    return text.rstrip() + "\n\n" + block + "\n"


def _extract_marked_section(text: str, start: str, end: str) -> str | None:
    if text.count(start) != 1 or text.count(end) != 1:
        return None
    _, remainder = text.split(start, maxsplit=1)
    section, _ = remainder.split(end, maxsplit=1)
    return section.strip("\n")


def apply_generated_state_blocks(
    text: str,
    *,
    artifacts_root: Path,
    registry_root: Path | None,
) -> str:
    """Return ``text`` with both generated blocks replaced by fresh renders."""

    text = _replace_marked_section(
        text,
        README_ACTIVE_MODEL_START,
        README_ACTIVE_MODEL_END,
        render_active_model_block(artifacts_root),
    )
    text = _replace_marked_section(
        text,
        README_RESEARCH_STATE_START,
        README_RESEARCH_STATE_END,
        render_research_state_block(registry_root, artifacts_root),
    )
    return text


def regenerate_readme_state(
    artifacts_root: Path,
    registry_root: Path | None,
    readme_path: Path,
) -> dict[str, Any]:
    """Rewrite the README's two generated blocks in place from current state."""

    current = readme_path.read_text(encoding="utf-8")
    updated = apply_generated_state_blocks(
        current, artifacts_root=artifacts_root, registry_root=registry_root
    )
    changed = updated != current
    if changed:
        atomic_text(updated, readme_path)
    return {"readme": str(readme_path), "changed": changed}


def readme_state_failures(
    readme_text: str,
    *,
    artifacts_root: Path,
    registry_root: Path | None,
) -> list[str]:
    """Return human-readable descriptions of any stale/missing generated block.

    Used by ``nfl_ats.handoff.check_session_handoff`` to report README drift
    the same way it already reports a stale ``HANDOFF.md``.
    """

    failures: list[str] = []
    active_current = _extract_marked_section(
        readme_text, README_ACTIVE_MODEL_START, README_ACTIVE_MODEL_END
    )
    if active_current is None:
        failures.append("README is missing the generated active-model-state block")
    elif active_current != render_active_model_block(artifacts_root).rstrip():
        failures.append("README active-model-state block is stale relative to local artifacts")

    research_current = _extract_marked_section(
        readme_text, README_RESEARCH_STATE_START, README_RESEARCH_STATE_END
    )
    if research_current is None:
        failures.append("README is missing the generated research-state block")
    elif research_current != render_research_state_block(registry_root, artifacts_root).rstrip():
        failures.append("README research-state block is stale relative to the registries")

    return failures
