from __future__ import annotations

import argparse
import json
import os
import re
import sys
from datetime import date
from pathlib import Path
from typing import Any

from nfl_ats.cli_common import _print_json
from nfl_ats.weak_signals import default_registry_path as weak_signal_registry_path
from nfl_ats.weak_signals import load_registry as load_weak_signals

CLOSED_ROW_MARKERS = ("✅", "❌")
SAFE_FAMILY = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*")


def proposals_dir() -> Path:
    return Path(os.environ.get("NFL_ATS_REGISTRY_DIR", "registry")) / "proposals"


def proposal_path(family: str) -> Path:
    return proposals_dir() / f"{family}.json"


def _roadmap_row_status(row_id: str, roadmap: Path) -> str | None:
    prefix = f"| {row_id} |"
    for line in roadmap.read_text(encoding="utf-8").splitlines():
        if line.startswith(prefix):
            cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
            return cells[1] if len(cells) > 1 else ""
    return None


def _family_looks(family: str) -> int:
    registry = load_weak_signals(weak_signal_registry_path())
    return sum(1 for signal in registry.signals.values() if signal.family == family)


def _cmd_propose(args: argparse.Namespace) -> None:
    if SAFE_FAMILY.fullmatch(args.family) is None:
        raise ValueError(f"family {args.family!r} is not a safe file stem")
    if args.look_budget < 1:
        raise ValueError("--look-budget must be at least 1")
    if not args.approval_reference.strip():
        raise ValueError("--approval-reference must name the owner instruction")
    status = _roadmap_row_status(args.row, Path(args.roadmap))
    if status is None:
        raise ValueError(f"ROADMAP row {args.row} does not exist in {args.roadmap}")
    if any(marker in status for marker in CLOSED_ROW_MARKERS):
        raise ValueError(f"ROADMAP row {args.row} is closed ({status})")
    declaration = Path(args.declaration)
    if not declaration.is_file():
        raise ValueError(f"declaration file {declaration} does not exist")
    path = proposal_path(args.family)
    if path.exists():
        raise ValueError(f"a proposal for family {args.family} already exists at {path}")
    payload = {
        "family": args.family,
        "roadmap_row": args.row,
        "roadmap_status": status,
        "look_budget": args.look_budget,
        "declaration": args.declaration,
        "approval_reference": args.approval_reference,
        "proposed_on": date.today().isoformat(),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    _print_json({"proposed": args.family, "path": str(path), **payload})


def _cmd_status(args: argparse.Namespace) -> None:
    path = proposal_path(args.family)
    if not path.exists():
        _print_json({"family": args.family, "proposal": None, "verdict": "no proposal"})
        sys.exit(2)
    proposal: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    looks = _family_looks(args.family)
    budget = int(proposal["look_budget"])
    exceeded = looks > budget
    _print_json(
        {
            "family": args.family,
            "roadmap_row": proposal["roadmap_row"],
            "approval_reference": proposal["approval_reference"],
            "looks_recorded": looks,
            "look_budget": budget,
            "looks_remaining": max(budget - looks, 0),
            "verdict": "budget exceeded" if exceeded else "within budget",
        }
    )
    if exceeded:
        sys.exit(1)


def register(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    research = subparsers.add_parser(
        "research",
        help="record an approved, fixed-budget research proposal per weak-signals family "
        "and check recorded looks against it; does not block weak-signals record",
    )
    commands = research.add_subparsers(dest="research_command", required=True)

    propose = commands.add_parser(
        "propose",
        help="write registry/proposals/<family>.json; one proposal per family",
    )
    propose.add_argument("--family", required=True)
    propose.add_argument("--row", required=True, help="open ROADMAP row id, e.g. SIM-08")
    propose.add_argument("--look-budget", type=int, required=True)
    propose.add_argument("--declaration", required=True, help="path to the declaration file")
    propose.add_argument(
        "--approval-reference",
        required=True,
        help="reference to the owner instruction that authorizes the row's work",
    )
    propose.add_argument("--roadmap", default="ROADMAP.md")
    propose.set_defaults(handler=_cmd_propose)

    status = commands.add_parser(
        "status",
        help="looks recorded in a family against its proposal budget; exit 1 when exceeded, "
        "2 when no proposal exists",
    )
    status.add_argument("--family", required=True)
    status.set_defaults(handler=_cmd_status)
