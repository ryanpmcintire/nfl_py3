from __future__ import annotations

import json
import os
from collections import Counter, defaultdict
from itertools import pairwise
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

SEASONS = (2023, 2024, 2025)
SPORTS = {"americanfootball_nfl": "NFL", "americanfootball_ncaaf": "CFB"}
QUOTE_FIELDS = {
    "provider_event_id",
    "bookmaker_key",
    "market",
    "outcome_side",
    "price",
    "line",
    "observed_at_utc",
    "commence_time_utc",
}
INGESTION_FIELDS = {
    "ingested_at_utc",
    "ingestion_timestamp_utc",
    "retrieved_at_utc",
    "received_at_utc",
    "fetched_at_utc",
    "captured_at_utc",
    "collected_at_utc",
    "stored_at_utc",
    "downloaded_at_utc",
}
CLOCK_FIELDS = {"bookmaker_last_update_utc", "market_last_update_utc"}
READ_FIELDS = (
    {
        "sport_key",
        "provider_event_id",
        "bookmaker_key",
        "market",
        "outcome_side",
        "observed_at_utc",
        "commence_time_utc",
        "source_scan_at_utc",
        "quote_timestamp_basis",
    }
    | CLOCK_FIELDS
    | INGESTION_FIELDS
)


def leaf_names(value: Any) -> set[str]:
    if not isinstance(value, dict):
        return set()
    found = set(value)
    for child in value.values():
        found.update(leaf_names(child))
    return found


def table(headers: list[str], rows: list[list[Any]]) -> list[str]:
    return [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
        *["| " + " | ".join(str(cell).replace("|", "/") for cell in row) + " |" for row in rows],
        "",
    ]


def main() -> None:
    pa.set_cpu_count(2)
    pa.set_io_thread_count(2)
    root = Path(os.environ.get("NFL_ATS_DATA_DIR", "data"))
    roots = {"market archive": root / "market", "college archive": root / "cfb"}
    families: dict[str, Counter[str]] = defaultdict(Counter)
    schemas: dict[str, Counter[tuple[str, ...]]] = defaultdict(Counter)
    manifest_names: dict[str, set[str]] = defaultdict(set)
    captures: Counter[str] = Counter()
    counts = {(league, season): Counter() for league in SPORTS.values() for season in SEASONS}
    events: dict[tuple[str, int], set[str]] = defaultdict(set)
    observations: dict[tuple[str, int, str], set[pd.Timestamp]] = defaultdict(set)
    cadence: dict[tuple[str, int, str, str], set[pd.Timestamp]] = defaultdict(set)
    errors: list[str] = []
    inputs: list[tuple[str, Path]] = []
    for family, directory in roots.items():
        families[family]["root_exists"] = int(directory.is_dir())
        inputs.extend((family, path) for path in sorted(directory.rglob("*.parquet")))
    legacy = root / "processed" / "sbr_odds.parquet"
    families["legacy SBR"]["root_exists"] = int(legacy.is_file())
    if legacy.is_file():
        inputs.append(("legacy SBR", legacy))
    for family, path in inputs:
        summary = families[family]
        summary["files"] += 1
        try:
            parquet = pq.ParquetFile(path)
            names = set(parquet.schema_arrow.names)
            summary["rows"] += parquet.metadata.num_rows
            schemas[family][tuple(sorted(names))] += 1
            manifest_path = path.parent / "manifest.json"
            manifest = {}
            if manifest_path.is_file():
                manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
                if not isinstance(manifest, dict):
                    raise ValueError("Manifest is not an object")
            manifest_names[family].update(leaf_names(manifest))
            if not QUOTE_FIELDS.issubset(names):
                summary["non_panel_files"] += 1
                continue
            summary["quote_files"] += 1
            summary["quote_rows"] += parquet.metadata.num_rows
            ingestion = INGESTION_FIELDS.intersection(names | leaf_names(manifest))
            summary["quote_files_with_ingestion_field"] += bool(ingestion)
            captures[str(manifest.get("capture_kind", "unspecified"))] += 1
            request = manifest.get("request") or {}
            sport = request.get("sport", "") if isinstance(request, dict) else ""
            frame = parquet.read(columns=sorted(names & READ_FIELDS), use_threads=False).to_pandas()
            if "sport_key" not in frame:
                frame["sport_key"] = sport
            starts = pd.to_datetime(frame.commence_time_utc, utc=True, errors="coerce")
            observed = pd.to_datetime(frame.observed_at_utc, utc=True, errors="coerce")
            season_values = starts.dt.year - (starts.dt.month < 7).astype(int)
            summary["rows_without_event_date"] += int(starts.isna().sum())
            summary["rows_unknown_sport"] += int((~frame.sport_key.isin(SPORTS)).sum())
            for sport_key, league in SPORTS.items():
                for season in SEASONS:
                    selected = frame.sport_key.eq(sport_key) & season_values.eq(season)
                    if not selected.any():
                        continue
                    group = frame.loc[selected]
                    key = (league, season)
                    result = counts[key]
                    result["files"] += 1
                    result["rows"] += len(group)
                    result["files_with_ingestion_field"] += bool(ingestion)
                    result["observed_rows"] += int(observed[selected].notna().sum())
                    result["pregame_rows"] += int((observed[selected] < starts[selected]).sum())
                    provider = pd.Series(pd.NaT, index=group.index, dtype="datetime64[ns, UTC]")
                    for field in ("market_last_update_utc", "bookmaker_last_update_utc"):
                        if field in group:
                            parsed = pd.to_datetime(group[field], utc=True, errors="coerce")
                            result[field] += int(parsed.notna().sum())
                            provider = provider.fillna(parsed)
                    result["provider_rows"] += int(provider.notna().sum())
                    result["nonnegative_age_rows"] += int((provider <= observed[selected]).sum())
                    for event, book, instant in zip(
                        group.provider_event_id.astype("string"),
                        group.bookmaker_key,
                        observed[selected],
                        strict=True,
                    ):
                        if pd.isna(event) or not event:
                            result["missing_event_id_rows"] += 1
                            continue
                        events[key].add(str(event))
                        if pd.notna(instant):
                            observations[(*key, str(event))].add(instant)
                            cadence[(*key, str(event), str(book))].add(instant)
        except (OSError, ValueError, TypeError, KeyError, pa.ArrowException) as error:
            errors.append(f"{path.as_posix()}: {type(error).__name__}: {error}")
    totals = {
        league: sum(counts[league, year]["rows"] for year in SEASONS) for league in SPORTS.values()
    }
    missing = [league for league, total in totals.items() if total == 0]
    state = (
        "inventory_incomplete"
        if errors
        else "source_gap"
        if missing
        else "timestamp_audit_required"
    )
    report = [
        "# LEAD-77 unit 1 - local quote timestamp inventory",
        "",
        f"**Measured:** `{state}`. Only schemas, capture metadata and quote clocks "
        "were read; no game outcome or price-response target was read or "
        "calculated.",
        "",
        "**Read:** ROADMAP.md:860 and docs/lanes/ideation-2026-09-29b.md:11-25 "
        "define the 2023-2025 NFL opener games and NFL/CFB quote panels. The "
        "declaration was copied to docs/lanes/lead77.md before execution. The "
        "partially pooled response remains the declared challenger.",
        "",
        "## Local source coverage",
        "",
        f"**Measured:** data root `{root.as_posix()}`; recursive Parquet metadata "
        "inventory covers `market/`, `cfb/` and `processed/sbr_odds.parquet`. Counts "
        "include out-of-population seasons. Quote panels require event, book, "
        "market, side, price, line, observation and event-time fields. File/row "
        "counts are not independent games.",
        "",
    ]
    report += table(
        [
            "Source",
            "Present",
            "Parquet files",
            "Rows",
            "Quote files",
            "Quote rows",
            "Quote files with ingestion field",
        ],
        [
            [
                family,
                bool(value["root_exists"]),
                value["files"],
                value["rows"],
                value["quote_files"],
                value["quote_rows"],
                value["quote_files_with_ingestion_field"],
            ]
            for family, value in families.items()
        ],
    )
    report += [
        "## Target-season quote clocks",
        "",
        "**Measured:** event dates assign football seasons (January-June use the "
        "preceding year). These are inventory labels, not folds. Provider event IDs "
        "are not certified opener-game counts.",
        "",
    ]
    report += table(
        [
            "League",
            "Season",
            "Files",
            "Rows",
            "Event IDs",
            "Events with 2+ observations",
            "Provider-clock rows",
            "Nonnegative-age rows",
            "Pregame rows",
            "Files with ingestion field",
        ],
        [
            [
                league,
                season,
                counts[league, season]["files"],
                counts[league, season]["rows"],
                len(events[league, season]),
                sum(
                    len(times) >= 2
                    for key, times in observations.items()
                    if key[:2] == (league, season)
                ),
                counts[league, season]["provider_rows"],
                counts[league, season]["nonnegative_age_rows"],
                counts[league, season]["pregame_rows"],
                counts[league, season]["files_with_ingestion_field"],
            ]
            for league in SPORTS.values()
            for season in SEASONS
        ],
    )
    report += [
        "**Measured:** cadence uses unique observations per event/book, pooling "
        "market/side duplicates. It does not estimate synchronized-consensus "
        "response.",
        "",
    ]
    cadence_rows = []
    for league in SPORTS.values():
        for season in SEASONS:
            seconds = []
            for key, instants in cadence.items():
                if key[:2] == (league, season):
                    ordered = sorted(instants)
                    seconds.extend(
                        (right - left).total_seconds() for left, right in pairwise(ordered)
                    )
            gaps = pd.Series(seconds, dtype=float)
            cadence_rows.append(
                [
                    league,
                    season,
                    len(gaps),
                    "NA" if gaps.empty else f"{gaps.median():.1f}",
                    "NA" if gaps.empty else f"{gaps.min():.1f}",
                    "NA" if gaps.empty else f"{gaps.max():.1f}",
                ]
            )
    report += table(
        ["League", "Season", "Cadence intervals", "Median seconds", "Min seconds", "Max seconds"],
        cadence_rows,
    )
    report += ["## Source limitations", ""]
    for family in families:
        combined = set().union(*(set(schema) for schema in schemas[family]))
        relevant = sorted(
            field
            for field in combined
            if field in READ_FIELDS
            or any(
                word in field.lower()
                for word in (
                    "odds",
                    "line",
                    "spread",
                    "time",
                    "date",
                    "season",
                    "year",
                    "book",
                    "provider",
                    "ingest",
                    "captur",
                    "retriev",
                    "fetch",
                )
            )
        )
        report.append(
            f"- **Measured:** {family}: {len(schemas[family])} schema variants; "
            "relevant field names: "
            f"{', '.join(relevant) or 'none'}. Manifest ingestion names: "
            f"{', '.join(sorted(manifest_names[family] & INGESTION_FIELDS)) or 'none'}."
        )
        report.append(
            f"- **Measured:** {family}: {families[family]['rows_without_event_date']} quote rows "
            f"without event dates; {families[family]['rows_unknown_sport']} quote rows "
            "with unrecognized sport keys."
        )
    report += [
        f"- **Measured:** quote capture kinds: {dict(sorted(captures.items()))}.",
        f"- **Measured:** inventory errors: {len(errors)}.",
        "- **Read:** src/nfl_ats/odds_backfill.py:242-264 uses provider snapshot "
        "time as historical observed time. This does not establish local ingestion "
        "time. src/nfl_ats/market_data.py:309-331 stores observed time and permits "
        "extra manifest fields.",
        "- **Inferred:** historical observed time, file modification time or "
        "pre-kickoff availability cannot substitute for issuance and ingestion "
        "before the authentic pool deadline. No deadline join or synchronization "
        "target has been certified.",
        "",
        "## Research reporting status",
        "",
        f"**Measured:** target quote rows NFL={totals['NFL']}, CFB={totals['CFB']}. "
        "Missing quote-panel populations: "
        f"{', '.join(missing) or 'none identified by schema'}. Fitted/scored games=0; "
        "consumed outcome looks=0. The declared family remains B=4, F=3, E=1, L=217. "
        "Inventory tables are metadata diagnostics, not outcome looks.",
        "",
        "**Measured:** decisive-game record is unavailable (no games scored), not "
        "evidence of a 0-0 record. IS/OOS accuracy, Brier, log loss, margin MAE, "
        "response MAE, gaps, five-band reliability, season-block 95% intervals and "
        "probability_positive are unestimated. No effect estimate or closure verdict "
        "is supported.",
        "",
    ]
    report += table(
        [
            "Held-out season",
            "IS",
            "OOS",
            "Gap",
            "Coefficients",
            "95% interval",
            "probability_positive",
        ],
        [
            [season, "NA", "NA", "NA", "not fitted", "not estimated", "not estimated"]
            for season in SEASONS
        ],
    )
    report += [
        "**Inferred:** remain at the inventory unit. Obtain documented pre-deadline "
        "NFL/CFB panels with provider clocks and ingestion provenance, then certify "
        "authentic opener/deadline joins and enough separate earlier training, "
        "selection and calibration seasons. Keep the pooled arm fixed, exclude "
        "target/later seasons from both leagues, normalize on training only and "
        "weight games equally. Missing earlier folds remain unavailable. Missing "
        "sources neither refute nor close the mechanism (AGENTS.md:65-85).",
        "",
        "## Reproduction",
        "",
        "```bash",
        ".tools/uv.exe run --no-sync python scripts/lead77_unit1.py",
        "```",
        "",
        "No registry command is supplied: no effect, interval, probability_positive "
        "or research verdict was computed. A placeholder estimate would not be a "
        "valid measurement.",
        "",
    ]
    if errors:
        report += ["## Inventory errors", "", *[f"- {error}" for error in errors[:10]], ""]
    destination = Path("docs/lead77_unit1_inventory.md")
    destination.write_text("\n".join(report), encoding="utf-8")
    print(
        json.dumps(
            {
                "status": state,
                "target_quote_rows": totals,
                "inventory_errors": len(errors),
                "outcome_looks": 0,
                "declared_looks": 217,
                "report": destination.as_posix(),
            },
            sort_keys=True,
        )
    )
    if errors:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
