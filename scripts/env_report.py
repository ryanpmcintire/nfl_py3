"""ENG-21: print (or compare) the deterministic environment lock report.

    uv run --no-sync python scripts/env_report.py
    uv run --no-sync python scripts/env_report.py --json > env_a.json
    uv run --no-sync python scripts/env_report.py --compare env_a.json

Implemented as a standalone script rather than a ``nfl-ats`` subcommand:
``src/nfl_ats/cli.py`` is a single very large file under concurrent edit by
other in-flight work (ENG-01 ``lockday_package.py``, ENG-02 ``preflight.py``,
both of which already registered their own subcommands there while this
script was being written), and CLI registration needs three separate edit
locations in that file (an import, a subparser block, a handler function).
This command is read-only and has no need to live on the console-script
surface to be useful, so it ships here instead -- avoiding any collision risk
in the shared file for a command that does not require one. The underlying
function (:func:`nfl_ats.environment_report.environment_report`) is what
``nfl_ats.provenance.artifact_provenance`` and
``nfl_ats.lockday_package.build_manifest`` actually call; this script is a
thin, independent way to print or diff its output on demand.

``--compare`` prints a fresh report, diffs it against the given prior report
(a JSON file, e.g. one this script wrote earlier with ``--json``), and
classifies every differing field as ``reproducibility_affecting`` or
``cosmetic`` (see ``nfl_ats.environment_report``'s module docstring for the
classification rationale). Per this repository's binding research invariant,
an environment difference is reporting context, never itself grounds to
reject a result -- this command never exits nonzero on a diff.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from nfl_ats.environment_report import compare_environment, environment_report  # noqa: E402


def _render_report(report: dict[str, object]) -> str:
    if "error" in report and len(report) == 1:
        return f"environment report FAILED: {report['error']}"
    python_info = report.get("python", {})
    uv_info = report.get("uv", {})
    platform_info = report.get("platform", {})
    packages = report.get("packages", {})
    git_info = report.get("git", {})
    lock_info = report.get("uv_lock", {})
    lines = [
        f"generated_at_utc : {report.get('generated_at_utc')}",
        f"python           : {python_info.get('version')} ({python_info.get('implementation')})",
        f"uv               : {uv_info.get('raw') or '(unavailable)'}",
        f"platform         : {platform_info.get('system')} {platform_info.get('release')} "
        f"{platform_info.get('machine')}",
        f"git              : {git_info.get('revision')}"
        f"{'  DIRTY' if git_info.get('dirty') else ''}",
        f"uv.lock sha256   : {lock_info.get('sha256')}",
        "packages         :",
    ]
    for name in sorted(packages) if isinstance(packages, dict) else []:
        lines.append(f"  {name:<14} {packages[name]}")
    secrets = report.get("secrets_detected", {})
    if isinstance(secrets, dict) and secrets:
        present = sorted(name for name, is_present in secrets.items() if is_present)
        absent = sorted(name for name, is_present in secrets.items() if not is_present)
        lines.append(f"secrets present  : {', '.join(present) if present else '(none)'}")
        if absent:
            lines.append(f"secrets absent   : {', '.join(absent)}")
    return "\n".join(lines)


def _render_comparison(comparison: dict[str, object]) -> str:
    if not comparison.get("differs"):
        return "no differences"
    lines = [
        f"{len(comparison.get('reproducibility_affecting_fields', []))} "
        "reproducibility-affecting difference(s), "
        f"{len(comparison.get('cosmetic_fields', []))} cosmetic difference(s)",
    ]
    fields = comparison.get("fields", {})
    if isinstance(fields, dict):
        for path in sorted(fields):
            row = fields[path]
            lines.append(f"  [{row['classification']}] {path}: {row['a']!r} -> {row['b']!r}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="emit the report as JSON")
    parser.add_argument(
        "--compare",
        type=Path,
        default=None,
        help="diff the current environment against a prior report JSON file",
    )
    args = parser.parse_args(argv)

    report = environment_report()

    if args.compare is not None:
        prior = json.loads(args.compare.read_text(encoding="utf-8"))
        comparison = compare_environment(prior, report)
        if args.json:
            print(
                json.dumps({"current": report, "comparison": comparison}, indent=2, sort_keys=True)
            )
        else:
            print(_render_report(report))
            print()
            print(f"compared against {args.compare}:")
            print(_render_comparison(comparison))
        return 0

    print(json.dumps(report, indent=2, sort_keys=True) if args.json else _render_report(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
