"""ENG-13: replay a recorded run and verify it reproduces (read-only).

Consumes either an ENG-01 lock-day package (``manifest.json``) or a bare
forecast artifact's ``metadata.json`` (the file ``margin-predict`` writes
beside ``predictions.csv``) and checks, without refetching or rewriting a
production artifact:

* every source/feature-table/forecast/card digest the manifest actually
  recorded, against disk (reusing ``nfl_ats.lockday_package``'s own
  verifier);
* the recorded environment block against ``environment_report()`` run now,
  split into reproducibility-affecting vs. cosmetic differences;
* the recorded git revision and dirty flag against ``git rev-parse HEAD``
  now;
* (unless ``--no-recompute``) a fresh, in-process regeneration of the
  forecast for the manifest's own season/week, written only under
  ``--output-root`` (a fresh temp directory by default, NEVER a production
  artifact directory), compared column-by-column against the recorded
  predictions.

Usage::

    .\\.tools\\uv.exe run --no-sync python scripts/replay_run.py <manifest>
    .\\.tools\\uv.exe run --no-sync python scripts/replay_run.py <manifest> --no-recompute
    .\\.tools\\uv.exe run --no-sync python scripts/replay_run.py <manifest> --json `
        --output-root $env:TEMP\\eng13_replay

Exit code is 0 only when every recorded digest verifies, no
reproducibility-affecting environment difference exists, and (if recompute
ran) the regenerated outputs match. A git revision mismatch is reported but
does not by itself fail replay -- replaying an old run from a newer checkout
is expected use, not an error. Per this repository's binding research
invariant this command reports differences; it never adjudicates them.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from nfl_ats.run_replay import replay_manifest  # noqa: E402


def _line(label: str, value: object) -> str:
    return f"  {label:<22}: {value}"


def render(report: dict[str, object]) -> str:
    status = "PASS" if report.get("ok") else "FAIL"
    lines = [f"run replay: {status}", _line("manifest", report.get("manifest_path"))]
    lines.append(_line("kind", report.get("manifest_kind")))
    lines.append(_line("season/week", f"{report.get('season')} week {report.get('week')}"))

    digests = report.get("digest_verification")
    if isinstance(digests, dict):
        lines.append(
            _line(
                "digests",
                f"{'ok' if digests.get('ok') else 'FAIL'}  "
                f"{digests.get('files_verified', 0)} verified of "
                f"{digests.get('files_checked', 0)} checked",
            )
        )
        for label, key in (("changed", "changed"), ("missing", "missing")):
            items = digests.get(key)
            if isinstance(items, list) and items:
                lines.append(f"    {label}: {len(items)}")
                for item in items[:10]:
                    if isinstance(item, dict):
                        lines.append(f"      {item.get('role')}  {item.get('path')}")

    revision = report.get("git_revision")
    if isinstance(revision, dict):
        match = "match" if revision.get("revision_match") else "MISMATCH"
        lines.append(
            _line(
                "git revision",
                f"{match}  recorded={str(revision.get('recorded_revision'))[:12]} "
                f"current={str(revision.get('current_revision'))[:12]}",
            )
        )

    environment = report.get("environment_comparison")
    if isinstance(environment, dict):
        if not environment.get("available"):
            lines.append(_line("environment", "not recorded in this manifest"))
        else:
            repro = environment.get("reproducibility_affecting_fields") or []
            cosmetic = environment.get("cosmetic_fields") or []
            lines.append(
                _line(
                    "environment",
                    f"{len(repro)} reproducibility-affecting diff(s), "
                    f"{len(cosmetic)} cosmetic diff(s)",
                )
            )
            for path in repro:
                fields = environment.get("fields") or {}
                detail = fields.get(path) if isinstance(fields, dict) else None
                if isinstance(detail, dict):
                    lines.append(
                        f"    REPRODUCIBILITY-AFFECTING  {path}: "
                        f"{detail.get('a')!r} -> {detail.get('b')!r}"
                    )

    recompute = report.get("recompute")
    if recompute is None:
        lines.append(_line("recompute", "not requested (--no-recompute)"))
    elif isinstance(recompute, dict):
        if not recompute.get("attempted"):
            lines.append(_line("recompute", f"skipped: {recompute.get('reason')}"))
        else:
            predictions = recompute.get("predictions_comparison") or {}
            metadata = recompute.get("metadata_comparison") or {}
            lines.append(
                _line(
                    "recompute",
                    f"{'match' if recompute.get('match') else 'DRIFT'}  "
                    f"predictions {'match' if predictions.get('match') else 'DRIFT'}, "
                    f"metadata {'match' if metadata.get('match') else 'DRIFT'}",
                )
            )
            columns = predictions.get("columns") if isinstance(predictions, dict) else None
            if isinstance(columns, dict):
                for column, detail in sorted(columns.items()):
                    if isinstance(detail, dict) and not detail.get("equal"):
                        lines.append(
                            f"    DRIFT  {column}: {detail.get('mismatched_rows')} row(s), "
                            f"max_abs_diff={detail.get('max_abs_diff')}"
                        )
            for key in ("columns_only_in_recorded", "columns_only_in_regenerated"):
                extra = predictions.get(key) if isinstance(predictions, dict) else None
                if extra:
                    lines.append(f"    {key}: {extra}")
            output_root = recompute.get("output_root")
            if output_root:
                lines.append(_line("recompute output", output_root))

    notes = report.get("notes")
    if isinstance(notes, list) and notes:
        lines.append("  notes:")
        lines.extend(f"    - {note}" for note in notes)

    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "manifest",
        type=Path,
        help=(
            "a lock-day package folder/manifest.json, or a forecast artifact folder/metadata.json"
        ),
    )
    parser.add_argument(
        "--no-recompute",
        action="store_true",
        help="skip in-process forecast regeneration; only verify digests/environment/revision",
    )
    parser.add_argument("--json", action="store_true", help="emit the report as JSON")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help=(
            "TEMP directory recompute writes into (never a production artifact directory); "
            "defaults to a fresh directory under the OS temp root"
        ),
    )
    args = parser.parse_args(argv)

    output_root = args.output_root
    if output_root is None:
        output_root = Path(tempfile.mkdtemp(prefix="nfl_ats_replay_"))

    report = replay_manifest(
        args.manifest,
        output_root=output_root,
        recompute=not args.no_recompute,
        repo_root=REPO_ROOT,
    )
    payload = report.to_dict()
    print(json.dumps(payload, indent=2, sort_keys=True) if args.json else render(payload))
    return 0 if report.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
