from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import scripts.roadmap_inventory as roadmap_inventory

SAMPLE = """\
## Phase 0 — foundation

| ID | Status | Item | Definition of done |
|---|---|---|---|
| FND-01 | ✅ | Finished | Complete |
| FND-02 | 🚧 | Active | Partial |

## Phase 1 — next

| ID | Status | Item | Definition of done |
|---|---|---|---|
| MOD-01 | ⬜ | Planned | Ready |
| MOD-02 | 🔬 | Research | Gated |
| MOD-03 | 🌙 | Moonshot | Someday |
| MOD-04 | ❌ | Declined | Unavailable |
"""


def test_parse_and_summarize_keeps_non_actionable_work_separate() -> None:
    inventory = roadmap_inventory.summarize(roadmap_inventory.parse_roadmap(SAMPLE))

    assert inventory.total == 6
    assert inventory.done == 1
    assert inventory.remaining == 5
    assert inventory.actionable == 2
    assert inventory.status_counts == {
        "done": 1,
        "in_progress": 1,
        "ready_or_planned": 1,
        "research_question": 1,
        "moonshot": 1,
        "declined_or_blocked": 1,
    }
    assert inventory.phases[0] == roadmap_inventory.PhaseSummary(
        phase="Phase 0 — foundation", total=2, done=1, remaining=1, actionable=1
    )


def test_duplicate_item_ids_fail_closed() -> None:
    duplicate = SAMPLE + "\n| FND-01 | ⬜ | Duplicate | No |\n"

    with pytest.raises(ValueError, match="duplicate roadmap item"):
        roadmap_inventory.parse_roadmap(duplicate)


def test_empty_document_fails_closed() -> None:
    with pytest.raises(ValueError, match="no roadmap items found"):
        roadmap_inventory.parse_roadmap("# Nothing here")


def test_live_roadmap_has_unique_assignable_items() -> None:
    root = Path(__file__).resolve().parents[1]
    items = roadmap_inventory.parse_roadmap((root / "ROADMAP.md").read_text(encoding="utf-8"))

    assert len(items) == 262
    assert all(item.phase != "Unassigned" for item in items)


def test_cli_json_matches_library_summary(tmp_path: Path) -> None:
    roadmap = tmp_path / "ROADMAP.md"
    roadmap.write_text(SAMPLE, encoding="utf-8")

    completed = subprocess.run(
        [
            sys.executable,
            str(Path(roadmap_inventory.__file__).resolve()),
            "--roadmap",
            str(roadmap),
            "--json",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    payload = json.loads(completed.stdout)
    assert payload["total"] == 6
    assert payload["actionable"] == 2
