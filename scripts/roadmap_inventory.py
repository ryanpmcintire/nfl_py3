"""Summarize the tracked backlog in ``ROADMAP.md``.

The roadmap intentionally mixes executable work, prerequisite-gated research,
and moonshots.  This small parser keeps those categories separate so a raw
"not done" count is not mistaken for an immediately actionable queue.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_ROADMAP = REPO_ROOT / "ROADMAP.md"

STATUS_LABELS = {
    "✅": "done",
    "🚧": "in_progress",
    "⬜": "ready_or_planned",
    "🔬": "research_question",
    "🌙": "moonshot",
    "❌": "declined_or_blocked",
}
ACTIONABLE_STATUSES = frozenset({"🚧", "⬜"})

PHASE_RE = re.compile(r"^## (?P<phase>Phase \d+ .+|Cross-league evidence and transfer.+)$")
ROW_RE = re.compile(
    r"^\| (?P<item_id>[A-Z]+-\d+) \| (?P<status>✅|🚧|⬜|🔬|🌙|❌) "
    r"\| (?P<title>[^|]+) \|"
)


@dataclass(frozen=True)
class RoadmapItem:
    item_id: str
    status: str
    title: str
    phase: str


@dataclass(frozen=True)
class PhaseSummary:
    phase: str
    total: int
    done: int
    remaining: int
    actionable: int


@dataclass(frozen=True)
class Inventory:
    total: int
    done: int
    remaining: int
    actionable: int
    status_counts: dict[str, int]
    phases: list[PhaseSummary]


def parse_roadmap(text: str) -> list[RoadmapItem]:
    """Parse roadmap rows while retaining the nearest phase heading."""

    phase = "Unassigned"
    items: list[RoadmapItem] = []
    seen: set[str] = set()
    for line_number, line in enumerate(text.splitlines(), start=1):
        if phase_match := PHASE_RE.match(line):
            phase = phase_match.group("phase")
            continue
        row_match = ROW_RE.match(line)
        if row_match is None:
            continue
        item_id = row_match.group("item_id")
        if item_id in seen:
            raise ValueError(f"duplicate roadmap item {item_id!r} at line {line_number}")
        seen.add(item_id)
        items.append(
            RoadmapItem(
                item_id=item_id,
                status=row_match.group("status"),
                title=row_match.group("title").strip(),
                phase=phase,
            )
        )
    if not items:
        raise ValueError("no roadmap items found")
    return items


def summarize(items: list[RoadmapItem]) -> Inventory:
    """Return overall and per-phase counts with gated work kept separate."""

    counts = Counter(item.status for item in items)
    unknown = set(counts) - STATUS_LABELS.keys()
    if unknown:
        raise ValueError(f"unknown roadmap statuses: {sorted(unknown)!r}")

    phase_order = list(dict.fromkeys(item.phase for item in items))
    phases: list[PhaseSummary] = []
    for phase in phase_order:
        phase_items = [item for item in items if item.phase == phase]
        done = sum(item.status == "✅" for item in phase_items)
        phases.append(
            PhaseSummary(
                phase=phase,
                total=len(phase_items),
                done=done,
                remaining=len(phase_items) - done,
                actionable=sum(item.status in ACTIONABLE_STATUSES for item in phase_items),
            )
        )

    done = counts["✅"]
    return Inventory(
        total=len(items),
        done=done,
        remaining=len(items) - done,
        actionable=sum(counts[status] for status in ACTIONABLE_STATUSES),
        status_counts={STATUS_LABELS[status]: counts[status] for status in STATUS_LABELS},
        phases=phases,
    )


def render_text(inventory: Inventory) -> str:
    """Render a compact human-readable inventory."""

    lines = [
        (
            f"roadmap: {inventory.done}/{inventory.total} done; "
            f"{inventory.remaining} remaining; {inventory.actionable} active/planned"
        ),
        "status: "
        + ", ".join(f"{name}={count}" for name, count in inventory.status_counts.items()),
        "phases:",
    ]
    lines.extend(
        (
            f"  {phase.phase}: {phase.done}/{phase.total} done, "
            f"{phase.remaining} remaining, {phase.actionable} active/planned"
        )
        for phase in inventory.phases
    )
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--roadmap", type=Path, default=DEFAULT_ROADMAP)
    parser.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    inventory = summarize(parse_roadmap(args.roadmap.read_text(encoding="utf-8")))
    if args.json:
        print(json.dumps(asdict(inventory), indent=2))
    else:
        print(render_text(inventory))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
