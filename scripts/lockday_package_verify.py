from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from nfl_ats.lockday_package import (  # noqa: E402
    load_package,
    summarise_package,
    verify_package,
)


def render(report: dict[str, object]) -> str:
    status = "PASS" if report["ok"] else "FAIL"
    lines = [
        f"lock-day package verification: {status}"
        f"{'  [REHEARSAL]' if report.get('rehearsal') else ''}",
        f"  package        : {report['package_directory']}",
        f"  season/week    : {report.get('season')} week {report.get('week')}",
        f"  manifest digest: {'ok' if report['manifest_sha256_ok'] else 'MISMATCH'} "
        f"({str(report.get('manifest_sha256_actual'))[:16]}...)",
        f"  files          : {report['files_verified']} verified of "
        f"{report['files_checked']} checked",
    ]
    for label, key in (
        ("CHANGED", "changed"),
        ("missing", "missing"),
        ("mutable changed (expected for append-only ledgers)", "mutable_changed"),
        ("not hashed (nothing was claimed, so nothing to check)", "unhashed"),
    ):
        items = report.get(key) or []
        if not isinstance(items, list) or not items:
            continue
        lines.append(f"  {label}: {len(items)}")
        for item in items[:20]:
            if isinstance(item, dict):
                lines.append(f"    {item.get('role')}  {item.get('path')}")
    build_errors = report.get("build_errors") or []
    if isinstance(build_errors, list) and build_errors:
        lines.append(f"  build errors recorded in the package: {len(build_errors)}")
        for item in build_errors:
            if isinstance(item, dict):
                lines.append(f"    {item.get('component')}: {item.get('error')}")
    return "\n".join(lines)


READ_ONLY_SCRIPT = True


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "package",
        type=Path,
        help="the package directory, or its manifest.json",
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=REPO_ROOT,
        help=(
            "resolve each hashed file's recorded repo-relative path against this root "
            "when its absolute path no longer exists (a package copied to another machine)"
        ),
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help=(
            "also fail when a file that WAS hashed at write time no longer exists "
            "on disk. Entries with no digest at all (a ledger this lock never wrote, "
            "an over-size file) are never fatal in either mode"
        ),
    )
    parser.add_argument("--json", action="store_true", help="emit the report as JSON")
    parser.add_argument(
        "--summary",
        action="store_true",
        help="print the package's human-readable summary instead of verifying",
    )
    args = parser.parse_args(argv)

    if args.summary:
        print(summarise_package(load_package(args.package)))
        return 0

    report = verify_package(args.package, repo_root=args.repo_root, strict=args.strict)
    print(json.dumps(report, indent=2, sort_keys=True) if args.json else render(report))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
