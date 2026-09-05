"""ENG-15: CLI for ``nfl_ats.ledger_reconcile`` -- ledger reconciliation and recovery.

Read-only, idempotent. Joins recorder result summaries, each append-only
ledger's own rows, and the published card's picks for one season/week, then
classifies every registered recorder into one of six buckets (see
``src/nfl_ats/ledger_reconcile.py``'s module docstring for the full
contract) and prints a report-only recovery plan for anything that is not
``consistent``.

Usage::

    .\\.tools\\uv.exe run --no-sync python scripts/ledger_reconcile.py --season 2026 --week 1
    .\\.tools\\uv.exe run --no-sync python scripts/ledger_reconcile.py --season 2026 --week 1 --json
    .\\.tools\\uv.exe run --no-sync python scripts/ledger_reconcile.py --season 2026 --week 1 `
        --package artifacts/lockday_packages/2026_wk01_<stamp>

Exit code is 0 only when every recorder classifies ``consistent``.

This is deliberately a thin wrapper: all of the logic lives in
``nfl_ats.ledger_reconcile`` so it is importable and unit-testable without
going through argv. If you also want the existing wiring-level audit (does
every ``ACTIVE_PROSPECTIVE`` challenger have a CLI path at all), run
``scripts/lockday_verify.py`` separately -- this script already imports it
internally to resolve the dedicated-ledger map, and folds one summary line
of its report into this script's human output below, but it does not
replace running it on its own for the full wiring detail.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from nfl_ats.ledger_reconcile import load_lockday_verify_module, reconcile, render  # noqa: E402

READ_ONLY_SCRIPT = True
# ENG-29: read-only; the ENG-29 scanner confirms zero write sites -- it joins recorder summaries,
# each ledger's own rows, and the published card, and prints the result (--json prints to stdout),
# creating nothing under artifacts/.


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--season", type=int, required=True)
    parser.add_argument("--week", type=int, required=True)
    parser.add_argument(
        "--run-id", type=str, default=None, help="restrict to one forecast/source/refresh run id"
    )
    parser.add_argument("--artifacts", type=Path, default=REPO_ROOT / "artifacts")
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument(
        "--card",
        type=Path,
        default=None,
        help="published card path (default: <repo-root>/CURRENT_PREDICTIONS.md)",
    )
    parser.add_argument(
        "--run-summary",
        type=Path,
        default=None,
        help="a saved weekly-run/publish-predictions JSON summary (same shape lockday_verify.py "
        "already consumes), used to detect missing_rows/not_run gate reasons",
    )
    parser.add_argument(
        "--package",
        type=Path,
        default=None,
        help="an ENG-01 lock-day decision package directory or manifest.json; optional, the "
        "reconciler works fully without it",
    )
    parser.add_argument("--json", action="store_true", help="emit the report as JSON")
    args = parser.parse_args(argv)

    report = reconcile(
        args.artifacts,
        season=args.season,
        week=args.week,
        run_id=args.run_id,
        repo_root=args.repo_root,
        card_path=args.card,
        run_summary_path=args.run_summary,
        package_path=args.package,
    )

    if args.json:
        import json

        print(json.dumps(report, indent=2, sort_keys=True, default=str))
    else:
        print(render(report))
        if report["lockday_verify_available"]:
            try:
                verify_module = load_lockday_verify_module(args.repo_root)
                verify_report = verify_module.verify(
                    args.artifacts, season=args.season, week=args.week
                )
                print("")
                print("  lockday_verify (wiring-level audit; run it directly for full detail):")
                print(
                    f"    {verify_report['recorded']} recorded, {verify_report['skipped']} "
                    f"skipped, {len(verify_report['missing'])} MISSING, "
                    f"{len(verify_report.get('pending_wiring', []))} pending wiring of "
                    f"{verify_report['active_registered']} active"
                )
            except Exception as error:  # never let this additive check break the main report
                print(f"\n  lockday_verify: could not run ({type(error).__name__}: {error})")

    return 0 if report["all_consistent"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
