"""Build the ENG-20 research-queue evidence ledger.

Writes ``registry/research_queue.json`` (machine-readable) and
``docs/research_queue.md`` (generated Markdown view) from a fresh join of
``ROADMAP.md``, ``registry/rotation_registry.json``,
``registry/weak_signals.json``, ``registry/experiment_specs/*.json``, and
``scripts/capture_scheduler.py``'s ``SCHEDULE``. See
``src/nfl_ats/research_queue.py`` for the join logic and the binding
crossing-zero / next-admissible-action rules this ledger encodes.

Usage
-----
    python scripts/research_queue.py            # write the tracked files
    python scripts/research_queue.py --check    # exit 1 if they are stale

``--check`` mirrors ``nfl-ats handoff --check``: it rebuilds the payload from
the live sources and compares it to what is committed, without writing
anything. The JSON payload carries no wall-clock timestamp, so a clean
`--check` run reports "up to date" on every invocation until an actual input
(the roadmap, a registry, the capture schedule, or a spec file) changes.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
for _path in (REPO_ROOT / "src", REPO_ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from nfl_ats import research_queue, rotation, weak_signals  # noqa: E402
from nfl_ats.io import atomic_json, atomic_text  # noqa: E402

DEFAULT_ROADMAP = REPO_ROOT / "ROADMAP.md"
DEFAULT_ROTATION_REGISTRY = REPO_ROOT / "registry" / "rotation_registry.json"
DEFAULT_WEAK_SIGNALS = REPO_ROOT / "registry" / "weak_signals.json"
DEFAULT_EXPERIMENT_SPECS_DIR = REPO_ROOT / "registry" / "experiment_specs"
DEFAULT_OUTPUT_JSON = REPO_ROOT / "registry" / "research_queue.json"
DEFAULT_OUTPUT_MD = REPO_ROOT / "docs" / "research_queue.md"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--roadmap", type=Path, default=DEFAULT_ROADMAP)
    parser.add_argument("--rotation-registry", type=Path, default=DEFAULT_ROTATION_REGISTRY)
    parser.add_argument("--weak-signals", type=Path, default=DEFAULT_WEAK_SIGNALS)
    parser.add_argument("--experiment-specs", type=Path, default=DEFAULT_EXPERIMENT_SPECS_DIR)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--output-md", type=Path, default=DEFAULT_OUTPUT_MD)
    parser.add_argument(
        "--check",
        action="store_true",
        help="exit 1 if the committed files are stale versus a fresh build; write nothing",
    )
    return parser


def _build(args: argparse.Namespace) -> tuple[dict, str]:
    roadmap_text = args.roadmap.read_text(encoding="utf-8")
    rotation_registry = rotation.load_registry(args.rotation_registry)
    weak_signal_registry = weak_signals.load_registry(args.weak_signals)
    spec_names = research_queue.load_experiment_spec_names(args.experiment_specs)

    rows = research_queue.build_queue(
        roadmap_text,
        rotation_registry,
        weak_signal_registry,
        experiment_spec_names=spec_names,
    )
    return research_queue.queue_payload(rows), research_queue.queue_markdown(rows)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    fresh_json, fresh_md = _build(args)

    if args.check:
        stale: list[str] = []
        if not args.output_json.is_file():
            stale.append(str(args.output_json))
        else:
            committed_json = json.loads(args.output_json.read_text(encoding="utf-8"))
            if committed_json != fresh_json:
                stale.append(str(args.output_json))
        if not args.output_md.is_file() or args.output_md.read_text(encoding="utf-8") != fresh_md:
            stale.append(str(args.output_md))

        if stale:
            print(f"research queue is stale: {', '.join(stale)}", file=sys.stderr)
            return 1
        print("research queue is up to date")
        return 0

    atomic_json(fresh_json, args.output_json)
    atomic_text(fresh_md, args.output_md)
    print(f"wrote {args.output_json} and {args.output_md} ({fresh_json['row_count']} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
