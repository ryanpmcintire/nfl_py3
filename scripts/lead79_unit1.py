from __future__ import annotations

import argparse
import json
import os
import re
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

SEASONS = tuple(range(2020, 2026))
LOOKAHEAD_NAME = re.compile(r"look[_-]?ahead", re.IGNORECASE)
CAPTURE_NAME = re.compile(r"^(\d{4})_week(\d{2})_.+\.json$")
QUOTE_FIELDS = (
    "nflverse_game_id",
    "commence_time_utc",
    "home_team",
    "away_team",
    "source_scan_at_utc",
)
SCHEDULE_FIELDS = ("game_id", "season", "game_type", "gameday", "home_team", "away_team")
REPORT = Path("docs/lead79_inventory.md")


def iso_date(value: Any) -> str | None:
    if value is None:
        return None
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00")).date().isoformat()
    except ValueError:
        return None


def schedule_inventory(data_root: Path) -> tuple[dict, dict, Path | None]:
    paths = sorted((data_root / "raw").glob("*/schedules.parquet"))
    if not paths:
        return {}, {}, None
    path = paths[-1]
    rows = pq.read_table(path, columns=list(SCHEDULE_FIELDS), use_threads=False).to_pylist()
    games = {}
    matchups = {}
    for row in rows:
        if row["game_type"] == "REG" and row["season"] in SEASONS:
            games[row["game_id"]] = row["season"]
            matchups[(iso_date(row["gameday"]), row["home_team"], row["away_team"])] = (
                row["game_id"],
                row["season"],
            )
    return games, matchups, path


def inventory(data_root: Path, artifacts_root: Path) -> dict[str, Any]:
    games, matchups, schedule = schedule_inventory(data_root)
    market_root = data_root / "market"
    paths = sorted(market_root.rglob("*.parquet"))
    result: dict[str, Any] = {
        "data_root": data_root.as_posix(),
        "artifacts_root": artifacts_root.as_posix(),
        "schedule": schedule.as_posix() if schedule else None,
        "schedule_games": Counter(games.values()),
        "market_files": len(paths),
        "market_rows": 0,
        "quote_files": 0,
        "quote_rows": 0,
        "target_calendar_rows": Counter(),
        "target_reg_games": {season: set() for season in SEASONS},
        "capture_files": Counter(),
        "capture_weeks": {season: set() for season in SEASONS},
        "manifest_files": 0,
        "manifest_kinds": Counter(),
        "manifest_ingestion_fields": Counter(),
        "quote_dates": Counter(),
        "source_scan_rows": 0,
        "missing_quote_identity_files": 0,
        "errors": [],
        "named_candidates": [],
    }
    for path in paths:
        try:
            parquet = pq.ParquetFile(path)
            result["market_rows"] += parquet.metadata.num_rows
            fields = set(parquet.schema_arrow.names)
            if "commence_time_utc" not in fields:
                result["missing_quote_identity_files"] += 1
                continue
            result["quote_files"] += 1
            columns = [field for field in QUOTE_FIELDS if field in fields]
            for batch in parquet.iter_batches(columns=columns, use_threads=False):
                for row in batch.to_pylist():
                    result["quote_rows"] += 1
                    stamp = iso_date(row.get("commence_time_utc"))
                    if stamp:
                        result["quote_dates"][stamp] += 1
                        year, month = (int(part) for part in stamp.split("-")[:2])
                        season = year - (month <= 2)
                        if season in SEASONS:
                            result["target_calendar_rows"][season] += 1
                    game_id = row.get("nflverse_game_id")
                    matchup = (stamp, row.get("home_team"), row.get("away_team"))
                    if game_id in games:
                        reg_season = games[game_id]
                    elif matchup in matchups:
                        game_id, reg_season = matchups[matchup]
                    else:
                        reg_season = None
                    if reg_season in SEASONS:
                        result["target_reg_games"][reg_season].add(game_id)
                    if iso_date(row.get("source_scan_at_utc")):
                        result["source_scan_rows"] += 1
        except (OSError, ValueError, pa.ArrowException) as error:
            result["errors"].append(f"{path.as_posix()}: {type(error).__name__}")
    for path in sorted(market_root.glob("raw/*/manifest.json")):
        try:
            manifest = json.loads(path.read_text(encoding="utf-8-sig"))
            result["manifest_files"] += 1
            kind = manifest.get("capture_kind", manifest.get("provider", "unspecified"))
            result["manifest_kinds"][str(kind)] += 1
            for field in ("ingested_at_utc", "captured_at_utc", "retrieved_at_utc"):
                if iso_date(manifest.get(field)):
                    result["manifest_ingestion_fields"][field] += 1
        except (OSError, ValueError, AttributeError) as error:
            result["errors"].append(f"{path.as_posix()}: {type(error).__name__}")
    for path in sorted((data_root / "splash").glob("*.json")):
        match = CAPTURE_NAME.match(path.name)
        if match:
            season, week = map(int, match.groups())
            result["capture_files"][season] += 1
            if season in SEASONS:
                result["capture_weeks"][season].add(week)
    for root in (data_root, artifacts_root):
        if root.is_dir():
            result["named_candidates"].extend(
                path.as_posix()
                for path in root.rglob("*")
                if path.is_file() and LOOKAHEAD_NAME.search(path.relative_to(root).as_posix())
            )
    return result


def render(result: dict[str, Any]) -> str:
    dates = result["quote_dates"]
    target_captures = sum(result["capture_files"][season] for season in SEASONS)
    target_quotes = sum(result["target_calendar_rows"].values())
    if result["errors"] or not result["schedule"]:
        state = "inventory_incomplete"
    elif target_captures == 0 or target_quotes == 0:
        state = "source_gap"
    else:
        state = "provenance_review_required"
    rows = [
        "# LEAD-79 unit 1 — local source inventory",
        "",
        f"**Measured:** `{state}`. No outcomes, fits, scores, or registry writes.",
        "The source gate does not close or reject the proposed mechanism.",
        "",
        "## Declared population and scope",
        "",
        "**Read:** `ROADMAP.md:862`; `docs/lanes/lead79.md` preserves Protocol B before execution.",
        "2020-2025 REG; same-fixture lookahead issued before either intervening game,",
        "issuance/ingestion before the pool deadline, and an authentic frozen Tuesday pool line.",
        "B=2, F=6, E=0: 291 planned looks; **measured:** 0 outcome looks used.",
        "",
        f"**Measured:** data root `{result['data_root']}`; schedule `{result['schedule']}`.",
        "Read only schedule identity fields and market identity/timestamp metadata;",
        "Parquet enumeration includes ignored local files. No scores or prediction rows loaded.",
        "The scan covers every Parquet under the data root's `market/`,",
        "`market/raw/*/manifest.json`, `splash/*.json` filenames, and names containing",
        "lookahead/look_ahead/look-ahead under the configured data and artifact roots.",
        f"Artifact root: `{result['artifacts_root']}`.",
        "Other unnamed/custom stores and external sources are outside this inventory.",
        "",
        "## Coverage",
        "",
        "**Measured:** counts are inventory totals, without a sampling interval.",
        "Quote rows repeat bookmakers, sides and markets; they are not independent games.",
        "A calendar-season row is a date candidate, not a certified lookahead quote.",
        "Pool file/week counts are filename coverage, not issuance or ingestion certification.",
        "",
        "| Season | REG schedule games | Quote rows by NFL calendar | "
        "Matched REG games | Pool files | Pool weeks |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for season in SEASONS:
        rows.append(
            f"| {season} | {result['schedule_games'][season]} | "
            f"{result['target_calendar_rows'][season]} | "
            f"{len(result['target_reg_games'][season])} | "
            f"{result['capture_files'][season]} | {len(result['capture_weeks'][season])} |"
        )
    rows.extend(
        [
            "",
            f"**Measured:** {result['market_files']} market Parquet files / "
            f"{result['market_rows']} stored rows; {result['quote_files']} files / "
            f"{result['quote_rows']} rows expose fixture kickoff metadata.",
            f"Quote kickoff range: {min(dates) if dates else 'unavailable'} to "
            f"{max(dates) if dates else 'unavailable'}.",
            f"Pool filename season counts: {dict(sorted(result['capture_files'].items()))}.",
            f"Files without fixture kickoff metadata: {result['missing_quote_identity_files']}.",
            f"Manifest files: {result['manifest_files']}; kinds: {dict(result['manifest_kinds'])}.",
            f"Explicit manifest ingestion fields: {dict(result['manifest_ingestion_fields'])}; "
            f"quote rows with a source-scan date: {result['source_scan_rows']}.",
            "",
            "**Read:** `src/nfl_ats/odds_backfill.py:242` writes historical snapshot time as",
            "`observed_at`; that value alone does not prove contemporaneous ingestion.",
            "`src/nfl_ats/pool_decision_lines.py:27` identifies authentic pool capture storage.",
            "No closing-line, retrospective-news, or motivational-lookahead substitute was used.",
            "",
            f"**Measured:** {len(result['named_candidates'])} lookahead-named candidate files.",
        ]
    )
    rows.extend(f"- `{path}`" for path in result["named_candidates"][:12])
    if len(result["named_candidates"]) > 12:
        rows.append("Only the first 12 names are shown; none is certified by its name alone.")
    rows.extend(
        [
            "",
            f"**Measured:** {len(result['errors'])} inventory read errors.",
            *result["errors"][:12],
            "",
            "## Statistical outputs and next unit",
            "",
            "**Measured:** decisive-game record unavailable (0 games scored); IS/OOS opener",
            "accuracy, Brier, log loss, margin MAE and their gaps unavailable; season-block",
            "95% intervals and `probability_positive` unavailable; calibration bands unavailable.",
            "2020, 2021, 2022, 2023, 2024 and 2025 fold coefficients are all unavailable",
            "because no fold was fitted. No effect estimate or terminal verdict exists to record.",
            "",
            "**Inferred:** continue only after an authentic same-fixture lookahead archive and",
            "Tuesday pool captures pass the declared timestamp gates. Then execute the fixed",
            "chronological LOSO protocol with cached predictions and four paired baselines.",
            "No evidence here refutes the mechanism or establishes serving eligibility.",
            "",
            "Reproduce: `.tools/uv.exe run --no-sync python scripts/lead79_unit1.py`.",
            "",
        ]
    )
    return "\n".join(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description="Inventory LEAD-79 local source provenance only.")
    parser.add_argument(
        "--data-root", type=Path, default=Path(os.getenv("NFL_ATS_DATA_DIR", "data"))
    )
    parser.add_argument(
        "--artifacts-root", type=Path, default=Path(os.getenv("NFL_ATS_ARTIFACTS_DIR", "artifacts"))
    )
    args = parser.parse_args()
    pa.set_cpu_count(2)
    pa.set_io_thread_count(2)
    result = inventory(args.data_root, args.artifacts_root)
    REPORT.write_text(render(result), encoding="utf-8")
    print(
        json.dumps(
            {
                "report": REPORT.as_posix(),
                "quote_files": result["quote_files"],
                "quote_rows": result["quote_rows"],
                "target_season_quote_rows": sum(result["target_calendar_rows"].values()),
                "target_season_pool_files": sum(result["capture_files"][s] for s in SEASONS),
                "lookahead_named_files": len(result["named_candidates"]),
                "errors": len(result["errors"]),
                "outcome_looks": 0,
            },
            sort_keys=True,
        )
    )
    return 1 if result["errors"] or not result["schedule"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
