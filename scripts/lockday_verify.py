"""Did every registered challenger actually record this week? One answer, one line each.

Why this exists
---------------
The prospective evidence for a week is spread across FOUR append-only
ledgers, not one, and the recorders that fill them are deliberately
fail-open: ``nfl_ats.cli._cmd_publish_predictions`` wraps seventeen of them in
``try/except -> {"recorded": 0, "error": ...}`` so that a broken challenger can
never un-publish the card. That is the right trade for the card and the wrong
one for the evidence -- a challenger that records nothing produces a run that
still reports success, with the failure buried in one of twenty nested JSON
keys nobody reads.

This is the aggregate nobody had. It answers, per registered
``ACTIVE_PROSPECTIVE`` challenger, one of three things:

* **recorded** -- rows landed in that challenger's own ledger.
* **skipped** -- zero rows AND a documented gate said so (no fresh market
  capture yet, no Friday injury page yet, no pick actually changed). Correct
  behaviour, not a defect, but it must be visible rather than assumed.
* **MISSING** -- zero rows and nothing explaining why. This is the failure the
  file exists to catch.

Run it immediately after the Tuesday lock command, and again after each
late-week refresh pass::

    uv run --no-sync python scripts/lockday_verify.py --season 2026 --week 1

Exit code is 0 only when nothing is MISSING.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

import pandas as pd  # noqa: E402

from nfl_ats.clv import load_paper_decisions  # noqa: E402
from nfl_ats.injury_signal_refresh_tilt import load_injury_signal_decisions  # noqa: E402
from nfl_ats.nflcom_refresh_overlay import load_nflcom_refresh_overlay_decisions  # noqa: E402
from nfl_ats.pick_refresh import load_pick_revisions  # noqa: E402
from nfl_ats.prospective_scoring import (  # noqa: E402
    active_challenger_ids,
    load_challenger_decisions,
)

#: Challengers whose rows do NOT live in the shared challenger ledger. Each
#: maps to the loader for its own ledger and a note on when it legitimately
#: records nothing, so a zero can be read as "not yet" rather than "broken".
DEDICATED_LEDGERS: dict[str, dict[str, Any]] = {
    "injury_signal_refresh_tilt": {
        "ledger": "prospective/injury_signal_refresh_decisions.parquet",
        "loader": load_injury_signal_decisions,
        "written_by": "refresh-picks --record-decisions",
        "legitimately_empty": (
            "records only on a late-week refresh pass; zero at the Tuesday lock is expected"
        ),
    },
    "nflcom_friday_refresh_out2_starters_v1": {
        "ledger": "prospective/nflcom_friday_refresh_decisions.parquet",
        "loader": load_nflcom_refresh_overlay_decisions,
        "written_by": "refresh-picks --record-decisions",
        "legitimately_empty": (
            "its freshness gate needs a page fetched at or after Friday 16:00 ET, so it "
            "cannot record at the Tuesday lock -- only on a Saturday/Sunday refresh pass"
        ),
    },
    "model_only_refresh_incumbent": {
        "ledger": "prospective/pick_revisions.parquet",
        "loader": load_pick_revisions,
        "written_by": "refresh-picks --record-decisions",
        "legitimately_empty": (
            "the pick-revision ledger records only games whose pick CHANGED; a week where "
            "no pick moved legitimately writes nothing"
        ),
    },
}

#: Challengers that record one row per WEEK rather than one per game.
WEEKLY_SINGLE_ROW = ("best_pick_nomination_v2", "best_pick_nomination_v3")


def _week_rows(frame: pd.DataFrame, *, season: int, week: int) -> pd.DataFrame:
    if frame.empty or "season" not in frame.columns or "week" not in frame.columns:
        return frame.iloc[0:0]
    return frame.loc[
        (pd.to_numeric(frame["season"], errors="coerce") == season)
        & (pd.to_numeric(frame["week"], errors="coerce") == week)
    ]


def gated_skips(run_summary: dict[str, Any] | None) -> dict[str, str]:
    """Challenger -> the gate reason its recorder reported, from a run summary.

    A recorder that returns ``{"skipped": true, "reason": ...}`` did its job:
    the market line was not captured yet, the injury page did not exist yet.
    That is invisible in the ledgers -- zero rows look identical either way --
    so without the run's own JSON a correct skip reads as a defect. Nested
    dicts are walked because ``weekly-run`` embeds each step's output.
    """

    reasons: dict[str, str] = {}
    if not run_summary:
        return reasons

    def walk(node: Any) -> None:
        if isinstance(node, list):
            for item in node:
                walk(item)
            return
        if not isinstance(node, dict):
            return
        challenger_id = node.get("challenger_id")
        reason = node.get("reason") or node.get("error")
        if isinstance(challenger_id, str) and isinstance(reason, str) and not node.get("recorded"):
            reasons.setdefault(challenger_id, reason)
        for value in node.values():
            walk(value)

    walk(run_summary)
    return reasons


def verify(
    artifacts_root: Path,
    *,
    season: int,
    week: int,
    run_summary: dict[str, Any] | None = None,
) -> dict[str, Any]:
    active = active_challenger_ids(artifacts_root)
    reported_gates = gated_skips(run_summary)

    shared = _week_rows(load_challenger_decisions(artifacts_root), season=season, week=week)
    shared_counts: dict[str, int] = {}
    if not shared.empty and "challenger_id" in shared.columns:
        shared_counts = shared["challenger_id"].astype(str).value_counts().to_dict()

    paper = _week_rows(load_paper_decisions(artifacts_root), season=season, week=week)

    rows: list[dict[str, Any]] = []
    for challenger_id in active:
        dedicated = DEDICATED_LEDGERS.get(challenger_id)
        if dedicated is None:
            count = int(shared_counts.get(challenger_id, 0))
            gate = reported_gates.get(challenger_id, "")
            if count:
                status, note = "recorded", ""
            elif gate:
                status, note = "skipped", gate
            else:
                status, note = "MISSING", "no rows and no gate explaining why"
            rows.append(
                {
                    "challenger_id": challenger_id,
                    "rows": count,
                    "ledger": "prospective/challenger_decisions.parquet",
                    "written_by": "publish-predictions --record-decisions (via weekly-run)",
                    "status": status,
                    "note": note,
                }
            )
            continue

        loader = dedicated["loader"]
        count = len(_week_rows(loader(artifacts_root), season=season, week=week))
        rows.append(
            {
                "challenger_id": challenger_id,
                "rows": count,
                "ledger": dedicated["ledger"],
                "written_by": dedicated["written_by"],
                "status": "recorded" if count else "skipped",
                "note": "" if count else dedicated["legitimately_empty"],
            }
        )

    missing = sorted(row["challenger_id"] for row in rows if row["status"] == "MISSING")
    return {
        "season": season,
        "week": week,
        "artifacts_root": str(artifacts_root),
        "verified_at_utc": datetime.now(UTC).isoformat(),
        "paper_ledger_rows": len(paper),
        "paper_best_pick": (
            str(paper.loc[paper["is_best_pick"].fillna(False), "game_id"].iloc[0])
            if not paper.empty
            and "is_best_pick" in paper.columns
            and bool(paper["is_best_pick"].fillna(False).any())
            else None
        ),
        "active_registered": len(active),
        "recorded": sum(1 for row in rows if row["status"] == "recorded"),
        "skipped": sum(1 for row in rows if row["status"] == "skipped"),
        "missing": missing,
        "challengers": sorted(
            rows, key=lambda row: (row["status"] != "MISSING", row["challenger_id"])
        ),
    }


def render(report: dict[str, Any]) -> str:
    lines = [
        f"lock-day verification  {report['season']} week {report['week']}",
        f"  artifacts root : {report['artifacts_root']}",
        f"  paper ledger   : {report['paper_ledger_rows']} rows"
        f"   best pick: {report['paper_best_pick'] or '(none)'}",
        f"  challengers    : {report['recorded']} recorded, {report['skipped']} skipped, "
        f"{len(report['missing'])} MISSING  of {report['active_registered']} active",
        "",
    ]
    width = max((len(row["challenger_id"]) for row in report["challengers"]), default=10)
    for row in report["challengers"]:
        marker = {"recorded": "ok ", "skipped": "-- ", "MISSING": "!! "}[row["status"]]
        line = f"  {marker}{row['challenger_id']:<{width}}  {row['rows']:>3} rows"
        if row["note"]:
            line += f"   ({row['note']})"
        lines.append(line)
    if report["missing"]:
        lines += [
            "",
            "  MISSING means the recorder produced nothing and named no gate. Read the",
            "  run's JSON summary for that challenger's key before the week's games start;",
            "  once a game kicks off its pick can no longer be recorded.",
        ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--season", type=int, required=True)
    parser.add_argument("--week", type=int, required=True)
    parser.add_argument("--artifacts", type=Path, default=REPO_ROOT / "artifacts")
    parser.add_argument(
        "--run-summary",
        type=Path,
        default=None,
        help=(
            "the JSON summary weekly-run/publish-predictions printed, so a recorder that "
            "reported a gate reason is scored 'skipped' rather than MISSING"
        ),
    )
    parser.add_argument("--json", action="store_true", help="emit the report as JSON")
    args = parser.parse_args(argv)

    run_summary: dict[str, Any] | None = None
    if args.run_summary is not None:
        run_summary = json.loads(args.run_summary.read_text(encoding="utf-8"))

    report = verify(args.artifacts, season=args.season, week=args.week, run_summary=run_summary)
    print(json.dumps(report, indent=2, sort_keys=True) if args.json else render(report))
    return 1 if report["missing"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
