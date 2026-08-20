"""Record the GDELT-source replication of ``attention_battery_both_cold``
into ``registry/weak_signals.json`` via ``nfl-ats weak-signals record``.

This is a same-definition replication attempt on a second, independent
attention source (GDELT DOC 2.0 timelinevol), not a new mined cell -- the
threshold (home_z<=-0.5 AND away_z<=-0.5), eligibility, value_col, and sign
are IDENTICAL to ``attention_battery_both_cold``
(``scripts/attention_battery_screen.py``). Every numeric field is read
directly from ``scripts/gdelt_attention_screen.py``'s ``results.json`` and
passed through as a CLI argument -- no hand-typed numbers.

Closing-grounds taxonomy (restated verbatim, AGENTS.md): An interval or CI
that contains zero is NEVER grounds to reject, fail, or close an experiment.
At this evaluator's ~2-point resolution, "contains zero" is the EXPECTED
outcome for a real small signal. Only two grounds ever close a line of work:
(1) refuted mechanism -- a RESOLVED wrong sign (whole interval on the wrong
side of zero) or zero split-half reliability; (2) bounded by a positive
control proven able to detect an effect that size. Everything else is
``unresolved_below_power``: record it with ``nfl-ats weak-signals record``,
report ``probability_positive``, never the binary "contains zero." The
registry code hard-rejects inadmissible closures; if a record command
errors, the verdict is wrong, not the validator.

Usage::

    uv run python scripts/record_gdelt_attention.py --results <path> [--replace]
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]

LEGACY_SOURCE = (
    "scripts/ingest_gdelt_attention.py; scripts/gdelt_attention_screen.py; "
    "artifacts/gdelt_attention/{timestamp}/results.json; "
    "docs/attention_followup.md (parent cell definition)"
)
WEEKLY_SOURCE = (
    "scripts/ingest_gdelt_backfill.py; scripts/build_gdelt_weekly_features.py; "
    "scripts/gdelt_attention_screen.py; "
    "artifacts/gdelt_attention/{timestamp}/results.json; "
    "docs/gdelt_backfill.md section 7 (frozen cell definition)"
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--replace", action="store_true")
    parser.add_argument("--recorded-at", default=None)
    args = parser.parse_args()

    payload = json.loads(args.results.read_text(encoding="utf-8"))
    timestamp = args.results.parent.name
    input_kind = payload.get("gdelt_input", {}).get("kind", "legacy_timelinevol")
    is_weekly_volume = input_kind == "timelinevolraw_tuesday_z"
    source = (WEEKLY_SOURCE if is_weekly_volume else LEGACY_SOURCE).format(timestamp=timestamp)
    instrument = (
        "GDELT DOC 2.0 timelinevolraw Tuesday raw-count z-score"
        if is_weekly_volume
        else "GDELT DOC 2.0 timelinevol"
    )

    cell: dict[str, Any] = payload["both_cold_gdelt_replication"]
    wb = cell["week_blocked_primary"]
    sb = cell["season_blocked_secondary"]
    corr = payload.get("cross_source_correlation", {})

    if wb.get("insufficient_data"):
        raise SystemExit("both_cold_gdelt_replication has insufficient_data -- nothing to record")

    corr_note = (
        f"Cross-source correlation (Wikipedia attention_z vs GDELT attention_z, "
        f"overlapping team-weeks): "
        f"n={corr.get('n_overlapping_team_weeks')}, r={corr.get('correlation')}, "
        f"n_teams={corr.get('n_teams')}, season_range={corr.get('season_range')}. "
        "A positive correlation is split-half-style evidence the two independent "
        "instruments measure the same underlying attention construct."
    )

    lower, upper = wb["week_blocked_ci95_scaled"]
    wrong_sign_resolved = float(upper) < 0.0
    classification = "refuted_mechanism" if wrong_sign_resolved else "unresolved_below_power"
    classification_evidence = (
        "Same-definition replication of attention_battery_both_cold "
        "(identical threshold/eligibility/sign/value_col) on a second, "
        f"independent attention source ({instrument}, domain-filtered to "
        "sports/news outlets). Week-blocked 95% interval "
        f"[{lower:+.4f}, {upper:+.4f}], probability_positive="
        f"{wb['probability_positive']:.4f}. "
    )
    if wrong_sign_resolved:
        classification_evidence += (
            "The whole interval is below zero after applying the predeclared "
            "sign=-1 transform, satisfying wrong_sign_resolved. "
        )
    else:
        classification_evidence += (
            "Wrong sign is not resolved and no positive-control bound was run, "
            "so the admissible classification is unresolved_below_power. "
        )
    classification_evidence += corr_note

    notes = (
        f"GDELT ingest: n_requests={payload['gdelt_ingest_manifest_summary'].get('n_requests')}, "
        f"n_parse_failures={payload['gdelt_ingest_manifest_summary'].get('n_parse_failures')}, "
        f"n_teams_covered={payload['gdelt_ingest_manifest_summary'].get('n_teams_covered')}/"
        f"{payload['gdelt_ingest_manifest_summary'].get('n_teams_total')}, "
        f"years {payload['gdelt_ingest_manifest_summary'].get('start_year')}-"
        f"{payload['gdelt_ingest_manifest_summary'].get('end_year')}, domain allowlist "
        f"{payload['gdelt_ingest_manifest_summary'].get('domain_allowlist')}. "
        f"gdelt_n_has_baseline={payload.get('gdelt_n_has_baseline')} / "
        f"gdelt_n_team_game_rows={payload.get('gdelt_n_team_game_rows')}. "
        f"Season-blocked secondary bootstrap (block=season, n={sb.get('n_blocks')} seasons): "
        f"95% [{sb['week_blocked_ci95_scaled'][0]:+.4f}, "
        f"{sb['week_blocked_ci95_scaled'][1]:+.4f}] P+={sb['probability_positive']:.4f} "
        "(robustness check, not the registry interval). Historical outcome is "
        "close-graded; this replication cannot veto an opener-graded play. " + corr_note
    )

    cmd = [
        sys.executable,
        "-m",
        "nfl_ats.cli",
        "weak-signals",
        "record",
        "--name",
        "attention_battery_both_cold_gdelt_replication",
        "--description",
        (
            "Same-definition replication of attention_battery_both_cold "
            "(both teams' trailing attention z <= -0.5, response home_cover, "
            f"sign=-1) on {instrument} instead of Wikipedia "
            "pageviews -- independent second attention source."
        ),
        "--source",
        source,
        "--effect",
        f"{wb['full_slate_effect_pts']:.10f}",
        "--effect-units",
        "accuracy_points",
        "--classification",
        classification,
        "--league",
        "nfl",
        "--season-start",
        str(payload["season_start"]),
        "--season-end",
        str(payload["season_end"]),
        "--interval-low",
        f"{wb['week_blocked_ci95_scaled'][0]:.10f}",
        "--interval-high",
        f"{wb['week_blocked_ci95_scaled'][1]:.10f}",
        "--probability-positive",
        f"{wb['probability_positive']:.10f}",
        "--sample-games",
        str(wb["n_total"]),
        "--sample-blocks",
        str(wb.get("n_blocks")),
        "--classification-evidence",
        classification_evidence,
        "--notes",
        notes,
    ]
    if corr.get("correlation") is not None:
        cmd += ["--reliability", f"{corr['correlation']:.10f}"]
    if wrong_sign_resolved:
        cmd += ["--closing-ground", "wrong_sign_resolved"]
    if args.recorded_at:
        cmd += ["--recorded-at", args.recorded_at]
    if args.replace:
        cmd.append("--replace")

    print("=== recording attention_battery_both_cold_gdelt_replication ===")
    result = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True)
    print(result.stdout)
    if result.returncode != 0:
        print(result.stderr, file=sys.stderr)
        raise SystemExit(
            f"weak-signals record failed (exit {result.returncode}); per AGENTS.md "
            "'if a record command errors, the verdict is wrong, not the validator' -- "
            "fix the invocation, do not weaken the classification to force it through."
        )


if __name__ == "__main__":
    main()
