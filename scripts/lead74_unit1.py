from __future__ import annotations

import json
import os
import re
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

SEASONS = (2023, 2024, 2025)
INGESTION = re.compile(
    r"ingest|retriev|receiv|fetch.*(?:at|time)|collect.*(?:at|time)|"
    r"download.*(?:at|time)|captured_at|stored_at|created_at",
    re.IGNORECASE,
)
PRICE_COLUMNS = {
    "bookmaker_key",
    "market",
    "outcome_side",
    "line",
    "price",
    "observed_at_utc",
    "commence_time_utc",
}


def field_names(value: Any, prefix: str = "") -> set[str]:
    if not isinstance(value, dict):
        return set()
    found = set()
    for key, child in value.items():
        name = f"{prefix}.{key}" if prefix else str(key)
        found.add(name)
        found.update(field_names(child, name))
    return found


def timestamp_season(value: Any) -> int | None:
    try:
        instant = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
    return instant.year - int(instant.month < 7)


def inventory(root: Path) -> dict[str, Any]:
    counts = {season: Counter() for season in SEASONS}
    labels = {season: Counter() for season in SEASONS}
    captures: Counter[str] = Counter()
    schemas: Counter[tuple[str, ...]] = Counter()
    ingestion_names: set[str] = set()
    errors: Counter[str] = Counter()
    loaded = excluded = inferred = candidate_errors = 0
    with os.scandir(root) as entries:
        paths = sorted(Path(entry.path) for entry in entries if entry.is_dir())
    for snapshot in paths:
        try:
            manifest = json.loads((snapshot / "manifest.json").read_text(encoding="utf-8"))
            if not isinstance(manifest, dict):
                raise ValueError("Manifest must be an object")
        except (OSError, ValueError) as error:
            errors[f"manifest:{type(error).__name__}"] += 1
            try:
                directory_season = timestamp_season(
                    datetime.strptime(snapshot.name[:16], "%Y%m%dT%H%M%SZ").isoformat()
                )
            except ValueError:
                directory_season = None
            candidate_errors += int(directory_season in SEASONS or directory_season is None)
            continue
        loaded += 1
        request = manifest.get("request") or {}
        if not isinstance(request, dict):
            request = {}
        try:
            season = int(request["season"])
        except (KeyError, ValueError, TypeError):
            season = timestamp_season(manifest.get("observed_at_utc"))
            inferred += int(season in SEASONS)
        if season not in SEASONS:
            excluded += 1
            continue
        row = counts[season]
        row["manifests"] += 1
        labels[season][str(request.get("decision_label", "unspecified"))] += 1
        kind = str(manifest.get("capture_kind", "unspecified"))
        captures[kind] += 1
        row["non_backfill"] += int(kind != "historical_backfill")
        names = {name for name in field_names(manifest) if INGESTION.search(name)}
        ingestion_names.update(names)
        row["manifest_ingestion_fields"] += int(bool(names))
        try:
            metadata = pq.read_metadata(snapshot / "quotes.parquet")
            columns = set(metadata.schema.to_arrow_schema().names)
        except (OSError, ValueError) as error:
            errors[f"quote_metadata:{type(error).__name__}"] += 1
            candidate_errors += 1
            continue
        schemas[tuple(sorted(columns))] += 1
        row["quote_files"] += 1
        row["quote_rows"] += metadata.num_rows
        row["price_schema_files"] += int(PRICE_COLUMNS.issubset(columns))
        quote_names = {name for name in columns if INGESTION.search(name)}
        ingestion_names.update(quote_names)
        row["ingestion_candidates"] += int(bool(names or quote_names))
    totals = sum(counts.values(), Counter())
    if candidate_errors:
        status = "inventory_incomplete"
    elif not totals["manifests"]:
        status = "source_gated_no_target_season_archives"
    elif not totals["ingestion_candidates"] and not totals["non_backfill"]:
        status = "source_gated_missing_ingestion_provenance"
    else:
        status = "source_metadata_requires_further_audit"
    return {
        "root": root.as_posix(),
        "status": status,
        "directories": len(paths),
        "loaded": loaded,
        "excluded": excluded,
        "inferred_seasons": inferred,
        "counts": counts,
        "labels": labels,
        "captures": captures,
        "schemas": schemas,
        "ingestion_names": sorted(ingestion_names),
        "errors": errors,
        "candidate_errors": candidate_errors,
        "totals": totals,
    }


def write_report(result: dict[str, Any], output: Path) -> None:
    totals = result["totals"]
    lines = [
        "# LEAD-74 unit 1: local quote provenance inventory",
        "",
        f"**Measured:** {result['status']}. Metadata only; no outcomes were read.",
        "The frozen declaration is in docs/lanes/lead74.md:9.",
        "",
        f"**Measured:** {result['root']} contains {result['directories']:,} snapshot "
        f"directories; {result['loaded']:,} manifests were readable.",
        f"There are {totals['manifests']:,} target-season manifests, "
        f"{totals['quote_files']:,} readable quote-file footers, and "
        f"{totals['quote_rows']:,} quote rows across all markets and game types.",
        "Source counts are not paired quotes, REG games, independent observations, or picks.",
        "",
        "| Season | Manifests | Quote files | Quote rows | Price-schema files | "
        "Ingestion-field candidates |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for season in SEASONS:
        row = result["counts"][season]
        lines.append(
            f"| {season} | {row['manifests']:,} | {row['quote_files']:,} | "
            f"{row['quote_rows']:,} | {row['price_schema_files']:,} | "
            f"{row['ingestion_candidates']:,} |"
        )
    lines += [
        "",
        "**Measured:** capture kinds: "
        + ", ".join(f"{key} = {value:,}" for key, value in sorted(result["captures"].items()))
        + ".",
        f"Target-season metadata errors: {result['candidate_errors']}; "
        f"season labels inferred from timestamps: {result['inferred_seasons']}.",
        "All-archive access/parse errors: "
        + (
            ", ".join(f"{key} = {value}" for key, value in sorted(result["errors"].items()))
            or "none"
        )
        + ".",
        "",
        "**Measured:** decision-label counts follow; labels alone do not prove pool deadlines.",
        "",
        "| Decision label | 2023 | 2024 | 2025 |",
        "|---|---:|---:|---:|",
    ]
    for label in sorted({label for labels in result["labels"].values() for label in labels}):
        lines.append(
            f"| {label} | "
            + " | ".join(str(result["labels"][season][label]) for season in SEASONS)
            + " |"
        )
    lines += [
        "",
        f"**Measured:** {len(result['schemas'])} normalized quote schema(s). "
        "Explicit ingestion/retrieval field candidates: "
        + (", ".join(result["ingestion_names"]) or "none")
        + ".",
        "",
        "**Read:** Protocol B requires issuance/ingestion before the pool deadline "
        "(docs/lanes/ideation-2026-09-29b.md:12). The backfill writer stores requested "
        "historical time and provider snapshot time without a retrieval timestamp "
        "(src/nfl_ats/odds_backfill.py:258). Those times cannot stand in for ingestion; "
        "file modification times are also not proof.",
        "",
    ]
    if result["status"] == "source_gated_missing_ingestion_provenance":
        lines += [
            "**Inferred:** the audited archive lacks the required predeadline-ingestion "
            "evidence. The declared replay is source-gated. This is a provenance gap, not "
            "absence of historical prices or evidence against the proposed mechanism.",
            "Same-book, two-sided, unchanged-spread matching; REG-only filtering; authentic "
            "frozen-opener and pool-deadline matching; and certified prediction coverage "
            "remain unaudited because the prerequisite failed.",
        ]
    else:
        lines.append(
            "**Inferred:** metadata has not certified the required quote pairs. Resolve "
            "the stated source/audit gaps before any outcome inspection or replay."
        )
    lines += [
        "",
        "| Requested result | Availability |",
        "|---|---|",
        "| Decisive-game record | Unscored; not a 0-0 result |",
        "| IS/OOS accuracy, Brier, log loss, margin MAE, and gap | Not estimated |",
        "| Season-block 95% intervals and probability_positive | Not estimated |",
        "| 2023, 2024, 2025 fold coefficients and stability | No folds fitted |",
        "| Five training-quantile reliability bands | Not estimated |",
        "| Looks | 177 predeclared; 0 fitting/outcome-reporting looks used |",
        "",
        "**Inferred:** descriptive inventory counts have no sampling interval. No effect, "
        "closing ground, promotion verdict, or registry record is justified. An interval "
        "crossing zero would not close this signal; one fitted calibrated discrete-margin "
        "probability would select the side in a future eligible replay.",
        "",
        "Next: supply verifiable predeadline ingestion provenance, then audit "
        "Tuesday/deadline quote pairs and frozen openers without changing the declaration.",
        "",
        "Command: .tools/uv.exe run --no-sync python scripts/lead74_unit1.py",
    ]
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    root = Path(os.environ.get("NFL_ATS_DATA_DIR", "data")) / "market" / "raw"
    result = inventory(root)
    output = Path("docs/lead74_unit1_inventory.md")
    write_report(result, output)
    print(
        json.dumps(
            {
                "status": result["status"],
                "target_manifests": result["totals"]["manifests"],
                "quote_rows": result["totals"]["quote_rows"],
                "ingestion_field_candidates": result["totals"]["ingestion_candidates"],
                "candidate_errors": result["candidate_errors"],
                "planned_looks": 177,
                "realized_outcome_looks": 0,
                "report": output.as_posix(),
            }
        )
    )


if __name__ == "__main__":
    main()
