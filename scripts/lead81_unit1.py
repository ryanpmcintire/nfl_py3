import csv
import gzip
import json
import os
import re
import sys
from collections import Counter
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

REPORT = Path("docs/lead81_unit1.md")
INVENTORY = Path("tests/scratch/codex/lead81_unit1_inventory.json")
LANE = Path("docs/lanes/lead81.md")
SOURCE_WORDS = re.compile(r"arrival|flight|rerout|cancel|itinerar|disrupt|travel", re.I)
ARRIVAL_WORDS = re.compile(r"arrival|arriv(?:ed|ing)|flight|rerout|itinerar", re.I)
SCHEMA_GROUPS = {
    "planned_arrival": r"(?:planned|scheduled).*arrival|arrival.*(?:planned|scheduled)",
    "actual_arrival": r"actual.*arrival|arrival.*actual",
    "publication_time": r"published|publication|issued|reported_at",
    "ingestion_time": r"ingested|captured|retrieved|fetched",
    "explicit_status": r"arrival_status|on_time|ontime|interruption_status",
    "source_reference": r"source_url|source_ref|report_url|evidence_url",
    "team": r"(?:^|[_.])team(?:$|[_.])|team_id|team_abbr",
    "game": r"game_id|game_key|event_id",
}


def table_columns(path):
    name = path.name.lower()
    if name.endswith(".parquet"):
        return pq.read_schema(path).names
    if name.endswith((".csv", ".csv.gz", ".tsv", ".tsv.gz")):
        opener = gzip.open if name.endswith(".gz") else open
        delimiter = "\t" if ".tsv" in name else ","
        with opener(path, "rt", encoding="utf-8-sig", errors="strict") as handle:
            header = handle.readline(1_048_577)
        if len(header) > 1_048_576:
            raise ValueError("header exceeds metadata size limit")
        return next(csv.reader([header], delimiter=delimiter), [])
    return None


def inventory():
    counts = Counter()
    extensions = Counter()
    candidates = []
    errors = []
    source_like_paths = []
    roots = []

    def walk_error(error):
        errors.append({"path": str(error.filename), "error": str(error)})

    for root in (Path("data"), Path("artifacts")):
        roots.append({"path": root.as_posix(), "exists": root.is_dir()})
        if not root.is_dir():
            continue
        for folder, directories, filenames in os.walk(root, onerror=walk_error):
            directories[:] = sorted(
                name for name in directories if not (Path(folder) / name).is_symlink()
            )
            for filename in sorted(filenames):
                path = Path(folder) / filename
                if path.is_symlink():
                    counts["symlinks_skipped"] += 1
                    continue
                counts[f"{root.name}_files"] += 1
                extensions[f"{root.name}:{''.join(path.suffixes[-2:])}"] += 1
                named_candidate = bool(SOURCE_WORDS.search(path.as_posix()))
                if named_candidate:
                    source_like_paths.append(path.as_posix())
                if root.name == "artifacts" and not named_candidate:
                    continue
                try:
                    columns = table_columns(path)
                except Exception as error:
                    errors.append({"path": path.as_posix(), "error": str(error)})
                    continue
                if columns is None:
                    counts[f"{root.name}_non_tabular_metadata_only"] += 1
                    continue
                counts[f"{root.name}_schemas"] += 1
                matches = [name for name in columns if SOURCE_WORDS.search(name)]
                if not named_candidate and not matches:
                    continue
                groups = {
                    group: [name for name in columns if re.search(pattern, name, re.I)]
                    for group, pattern in SCHEMA_GROUPS.items()
                }
                candidates.append(
                    {
                        "path": path.as_posix(),
                        "matching_columns": matches,
                        "arrival_or_flight_columns": [
                            name for name in columns if ARRIVAL_WORDS.search(name)
                        ],
                        "schema_groups": groups,
                        "missing_groups": [group for group, names in groups.items() if not names],
                        "column_count": len(columns),
                    }
                )

    with Path("config/source_policies.json").open(encoding="utf-8-sig") as handle:
        sources = json.load(handle)["sources"]
    complete = [item["path"] for item in candidates if not item["missing_groups"]]
    return {
        "protocol": {
            "lead": "LEAD-81",
            "seasons": list(range(2020, 2026)),
            "season_type": "REG",
            "B": 2,
            "F": 6,
            "E": 0,
            "planned_looks": 291,
            "executed_statistical_looks": 0,
        },
        "status": "metadata_candidate_requires_provenance_review" if complete else "source_gap",
        "roots": roots,
        "counts": dict(sorted(counts.items())),
        "extensions": dict(sorted(extensions.items())),
        "source_policy_count": len(sources),
        "source_policy_name_matches": [name for name in sources if SOURCE_WORDS.search(name)],
        "source_like_paths": source_like_paths,
        "candidate_schemas": candidates,
        "complete_schema_candidates": complete,
        "metadata_errors": errors,
        "verified_eligible_game_pairs": 0,
        "scored_games": 0,
        "probability_positive": None,
        "confidence_interval": None,
    }


def report(result):
    counts = result["counts"]
    schemas = result["candidate_schemas"]
    complete = result["complete_schema_candidates"]
    arrival_schemas = [item for item in schemas if item["arrival_or_flight_columns"]]
    rows = [
        "# LEAD-81 unit 1: arrival-source inventory",
        "",
        (
            "**Measured:** decisive-game record is unavailable: 0 games scored; no win/loss/push "
            "record was computed."
        ),
        "Effect, 95% interval and `probability_positive` are unestimated, not zero or 0.5.",
        "",
        "**Read:** the protocol was copied into `docs/lanes/lead81.md` before this run from",
        "`ROADMAP.md:864` and `docs/lanes/ideation-2026-09-29b.md:11-27`.",
        "It requires 2020-2025 REG games, verified planned/actual arrivals for both teams,",
        "explicit on-time controls and public issuance/ingestion before the pick deadline.",
        "",
        "## Inventory result",
        "",
        f"**Measured:** `{result['status']}`; {len(complete)} complete-schema candidates;",
        (
            f"{result['verified_eligible_game_pairs']} eligible game pairs established by this "
            f"inventory."
        ),
        "This is a source-availability finding; the mechanism remains untested.",
        "",
        "| Metadata check | Measured result |",
        "|---|---:|",
        f"| Local data filenames, including ignored files | {counts.get('data_files', 0)} |",
        (
            f"| Local artifact filenames, including ignored files | "
            f"{counts.get('artifacts_files', 0)} |"
        ),
        f"| Data table schemas inspected | {counts.get('data_schemas', 0)} |",
        f"| Travel-named artifact table schemas inspected | {counts.get('artifacts_schemas', 0)} |",
        f"| Source-like filenames across both roots | {len(result['source_like_paths'])} |",
        f"| Tables with source-like filenames or columns | {len(schemas)} |",
        f"| Tables with arrival/flight/reroute/itinerary columns | {len(arrival_schemas)} |",
        f"| Registered source policies | {result['source_policy_count']} |",
        (
            f"| Source-policy names matching travel/arrival terms | "
            f"{len(result['source_policy_name_matches'])} |"
        ),
        f"| Metadata read errors | {len(result['metadata_errors'])} |",
        "",
        "**Measured:** Parquet reads use schemas only; CSV/TSV reads use headers only.",
        (
            "Data files in other formats and unrelated artifact files were catalogued by "
            "filename only."
        ),
        "No outcome rows, archives, model fits, network requests or source acquisitions were used.",
        (
            "Split-table sources, generic JSON/news archives and symlink targets are not "
            "certified absent."
        ),
        (
            "A complete schema still requires row-level timing and control-coverage review "
            "before a replay."
        ),
        "Missing roots and read errors are recorded; neither proves source absence.",
        "",
        "**Read:** `docs/travel_geometry_features.md:21-32` describes scheduled distance,",
        (
            "venue timezone and prior scheduled travel. Those inputs do not establish realized "
            "arrival delays."
        ),
    ]
    if complete:
        rows += [
            (
                "**Inferred:** inspect candidate provenance before deciding whether a replay is "
                "possible."
            ),
            *[f"- `{path}`" for path in complete],
        ]
    else:
        rows += [
            (
                "**Inferred:** no qualifying arrival archive was identified in this inventory; "
                "retain the source gap."
            ),
            (
                "Remembered cancellations or distance proxies cannot substitute for the declared "
                "population."
            ),
        ]
    rows += [
        "",
        "## Evaluation availability",
        "",
        "**Measured:** IS/OOS accuracy, Brier, log loss, margin MAE and their gaps: unavailable.",
        (
            "Combined-model, model-only, market-only and Elo comparisons and five-band "
            "reliability: unavailable."
        ),
        "For each scheduled fold (2020, 2021, 2022, 2023, 2024, 2025), coefficients and season",
        (
            "intervals are unavailable because no arrival dataset was certified. No weights were "
            "fitted."
        ),
        "Planned looks: B=2, F=6, E=0; (2 + 36)(6 + 1) + 25 = 291. Executed statistical looks: 0.",
        (
            "**Read:** `AGENTS.md:65-83` permits no closure from insufficient evidence; no "
            "terminal verdict"
        ),
        "or research effect is asserted here. There is no numeric result to record yet.",
        "",
        "## Prospective collection specification",
        "",
        (
            "**Inferred collection requirements, without changing the registered historical "
            "protocol:**"
        ),
        "",
        "- Declare the entire team-game roster before collecting reports, including home teams.",
        "  Keep missing reports as missing; collect explicit on-time confirmations for controls.",
        (
            "- Preserve game/team/opponent/venue, season/type, frozen Tuesday-noon opener and "
            "its source,"
        ),
        "  pick deadline, kickoff and observed market movement with issuance and ingestion times.",
        "- Preserve separate planned and actual arrival times with timezone, report URLs, public",
        "  publication times, local first-ingestion times and immutable evidence references.",
        "  Retain original reports and revisions; never overwrite the as-of record.",
        "- Capture explicit arrival status and documented cancellation/reroute cause, interruption",
        "  time and report provenance. Confirm interruption occurred after the frozen opener.",
        "- Require both teams' timing and evidence before the pick deadline and actual arrival",
        "  no later than its report. Reject retrospective and outcome-selected reports.",
        "- Derive arrival delay in hours and the away-minus-home term only from certified reports.",
        (
            "  Missing evidence never becomes zero. Reconstruct authentic frozen openers; never "
            "use closes."
        ),
        "- Before replay, tabulate both-team/control coverage by scheduled season without scoring.",
        "  Use certified predictions and earlier-only training, selection and calibration folds.",
        "  One calibrated discrete-margin distribution fits delay conditional on market movement.",
        (
            "  Preserve the declared metrics, baselines, season intervals, coefficients and "
            "291-look budget."
        ),
        "- New prospective seasons require a separate predeclared protocol before outcomes; they",
        "  cannot silently replace the fixed 2020-2025 historical population.",
        "",
        "## Reproduction and handoff",
        "",
        "Command: `.tools/uv.exe run --no-sync python scripts/lead81_unit1.py`.",
        "Metadata details: `tests/scratch/codex/lead81_unit1_inventory.json` (local, ignored).",
        (
            "Next unit: identify a qualifying existing archive or arrange authorized prospective "
            "collection."
        ),
        (
            "No registry command was run; there is no estimable effect or valid numeric record "
            "command yet."
        ),
    ]
    return "\n".join(rows) + "\n"


def main():
    if sys.version_info[:2] != (3, 12):
        raise RuntimeError("LEAD-81 requires the locked Python 3.12 environment")
    declaration = LANE.read_text(encoding="utf-8-sig")
    for required in ("2020-2025 REG", "explicit on-time controls", "291", "Protocol"):
        if required not in declaration:
            raise RuntimeError(f"Missing predeclared protocol item: {required}")
    pa.set_cpu_count(1)
    pa.set_io_thread_count(1)
    result = inventory()
    INVENTORY.parent.mkdir(parents=True, exist_ok=True)
    INVENTORY.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    REPORT.write_text(report(result), encoding="utf-8")
    summary = {
        "status": result["status"],
        "counts": result["counts"],
        "complete_schema_candidates": len(result["complete_schema_candidates"]),
        "candidate_schemas": len(result["candidate_schemas"]),
        "source_policy_count": result["source_policy_count"],
        "source_policy_name_matches": result["source_policy_name_matches"],
        "metadata_errors": len(result["metadata_errors"]),
        "scored_games": 0,
        "planned_looks": 291,
        "executed_statistical_looks": 0,
        "report": REPORT.as_posix(),
        "inventory": INVENTORY.as_posix(),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
