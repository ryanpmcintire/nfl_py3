from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from textwrap import fill

import pandas as pd
import pyarrow.parquet as pq

ARCHIVE_ROOT = Path("data/raw/forecast_archive")
DEFAULT_OUTPUT = Path("docs/lead75_inventory.md")
METADATA_COLUMNS = (
    "game_id",
    "kickoff_utc",
    "tuesday_cutoff_utc",
    "decision_cutoff_utc",
    "issuance_runtime_utc",
    "forecast_valid_utc",
    "forecast_wind_mph",
    "fetch_status",
    "icao_station",
    "cutoff_mode",
)
INGESTION_COLUMNS = (
    "ingested_at_utc",
    "ingestion_utc",
    "fetched_at_utc",
    "collected_at_utc",
    "retrieved_at_utc",
)


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def load_population() -> tuple[pd.DataFrame, Path]:
    manifest = read_json(ARCHIVE_ROOT / "pool_decision_2009_2025/manifest.json")
    path = Path(manifest["inputs"]["schedules"]["path"])
    if path.is_absolute():
        path = path.resolve().relative_to(Path.cwd().resolve())
    frame = pd.read_parquet(
        path, columns=["game_id", "season", "game_type", "roof"], use_threads=False
    )
    frame = frame.loc[
        frame["season"].between(2020, 2025)
        & frame["game_type"].eq("REG")
        & frame["roof"].eq("outdoors"),
        ["game_id", "season"],
    ].copy()
    if frame["game_id"].duplicated().any():
        raise ValueError("Schedule inventory requires unique game identifiers")
    return frame, path


def inspect_archive(path: Path, population: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    manifest = read_json(path.with_name("manifest.json"))
    schema = pq.ParquetFile(path).schema_arrow.names
    columns = [name for name in (*METADATA_COLUMNS, *INGESTION_COLUMNS) if name in schema]
    frame = pd.read_parquet(path, columns=columns, use_threads=False).merge(
        population, on="game_id", how="inner", validate="many_to_one"
    )
    mode = manifest.get("cutoff_mode")
    if not mode and "cutoff_mode" in frame:
        modes = frame["cutoff_mode"].dropna().unique()
        mode = str(modes[0]) if len(modes) == 1 else None
    if not mode and "tuesday_cutoff_utc" in frame and "decision_cutoff_utc" not in frame:
        mode = "tuesday_noon"
    product = manifest.get("mos_model")
    if not product:
        match = re.search(r"\bmodel=([A-Za-z0-9_-]+)", manifest.get("source", ""))
        product = match.group(1) if match else "unknown"
    cutoff_column = "tuesday_cutoff_utc" if mode == "tuesday_noon" else "decision_cutoff_utc"
    for name in ("kickoff_utc", cutoff_column, "issuance_runtime_utc", "forecast_valid_utc"):
        frame[name] = pd.to_datetime(frame.get(name), utc=True, errors="coerce")
    valid = (
        frame["fetch_status"].eq("ok")
        & pd.to_numeric(frame["forecast_wind_mph"], errors="coerce").notna()
        & frame["forecast_valid_utc"].notna()
        & frame["kickoff_utc"].notna()
        & frame["issuance_runtime_utc"].le(frame[cutoff_column])
    )
    ingestion_columns = [name for name in INGESTION_COLUMNS if name in schema]
    frame["ingestion_ok"] = False
    for name in ingestion_columns:
        ingestion = pd.to_datetime(frame[name], utc=True, errors="coerce")
        frame["ingestion_ok"] |= ingestion.ge(frame["issuance_runtime_utc"]) & ingestion.le(
            frame[cutoff_column]
        )
    frame["product"] = product
    frame["mode"] = mode or "unknown"
    frame["source"] = str(manifest.get("source", ""))
    summary = {
        "archive": path.parent.name,
        "product": product,
        "mode": mode or "unknown",
        "rows": len(frame),
        "valid_rows": int(valid.sum()),
        "ingestion_rows": int((valid & frame["ingestion_ok"]).sum()),
        "ingestion_fields": len(ingestion_columns),
    }
    return summary, frame.loc[valid].copy()


def inventory() -> tuple[str, dict]:
    population, schedules_path = load_population()
    paths = sorted(ARCHIVE_ROOT.rglob("forecasts.parquet"))
    if not paths:
        raise FileNotFoundError("No local forecast parquet archives")
    summaries = []
    frames = []
    for path in paths:
        summary, frame = inspect_archive(path, population)
        summaries.append(summary)
        frames.append(frame)
    all_rows = pd.concat(frames, ignore_index=True)
    keys = [
        "game_id",
        "kickoff_utc",
        "icao_station",
        "forecast_valid_utc",
        "issuance_runtime_utc",
        "product",
        "source",
        "ingestion_ok",
    ]
    early = all_rows.loc[all_rows["mode"].eq("tuesday_noon"), keys].drop_duplicates()
    late = all_rows.loc[all_rows["mode"].eq("pool_decision"), keys].drop_duplicates()
    paired = early.merge(late, on="game_id", suffixes=("_tuesday", "_deadline"))
    same_product = (
        paired["product_tuesday"].ne("unknown")
        & paired["product_tuesday"].eq(paired["product_deadline"])
        & paired["source_tuesday"].ne("")
        & paired["source_tuesday"].eq(paired["source_deadline"])
    )
    matched_target = (
        paired["kickoff_utc_tuesday"].eq(paired["kickoff_utc_deadline"])
        & paired["icao_station_tuesday"].eq(paired["icao_station_deadline"])
        & paired["forecast_valid_utc_tuesday"].eq(paired["forecast_valid_utc_deadline"])
        & paired["issuance_runtime_utc_tuesday"].lt(paired["issuance_runtime_utc_deadline"])
    )
    eligible = (
        same_product
        & matched_target
        & paired["ingestion_ok_tuesday"]
        & paired["ingestion_ok_deadline"]
    )
    result = {
        "status": "source_pairs_available" if eligible.any() else "data_gap",
        "archives": len(paths),
        "population_games": len(population),
        "tuesday_games": int(early["game_id"].nunique()),
        "deadline_games": int(late["game_id"].nunique()),
        "overlap_games": int(paired["game_id"].nunique()),
        "same_product_games": int(paired.loc[same_product, "game_id"].nunique()),
        "same_product_target_games": int(
            paired.loc[same_product & matched_target, "game_id"].nunique()
        ),
        "eligible_games": int(paired.loc[eligible, "game_id"].nunique()),
        "archives_with_ingestion_fields": sum(row["ingestion_fields"] > 0 for row in summaries),
        "planned_looks": 291,
        "executed_outcome_looks": 0,
    }
    partial = {
        path.parent
        for pattern in ("run_config.json", "results.jsonl")
        for path in ARCHIVE_ROOT.rglob(pattern)
        if not path.with_name("forecasts.parquet").is_file()
    }
    lines = [
        "# LEAD-75 unit 1 — local forecast inventory",
        "",
        f"**Measured:** source gate `{result['status']}`; "
        f"{result['eligible_games']} eligible matched forecast pairs. No fit or score ran.",
        "",
        "Command: `.tools/uv.exe run --no-sync python scripts/lead75_unit1.py`.",
        "Protocol: `docs/lanes/lead75.md`, saved before outcomes; "
        "ROADMAP.md:858 and Protocol B at docs/lanes/ideation-2026-09-29b.md:11-23.",
        "",
        f"**Measured:** scanned {len(paths)} parquet archives under `{ARCHIVE_ROOT.as_posix()}`; "
        f"{len(partial)} additional partial archive directories. "
        f"Population: {len(population)} REG open-air games in 2020-2025, "
        f"from `{schedules_path.as_posix()}` using only game identity, season, game type and roof. "
        "Forecast reads project only forecast values and provenance; no realized weather, "
        "scores, margins, picks or prediction outcomes were loaded.",
        "",
        "All counts below are **measured** inventory counts, not effect estimates.",
        "",
        "| Archive | Product | Cutoff | Rows | Usable wind/issuance | Timely ingestion |",
        "|---|---|---|---:|---:|---:|",
    ]
    for row in summaries:
        lines.append(
            f"| {row['archive']} | {row['product']} | {row['mode']} | {row['rows']} | "
            f"{row['valid_rows']} | {row['ingestion_rows']} |"
        )
    lines += [
        "",
        "Usable wind/issuance requires fetch success, a nonmissing wind forecast and valid time, "
        "and issuance no later than the archive cutoff. It is not certification for replay. "
        "Rows from spot checks and checkpoints overlap; archive row counts are not additive.",
        "",
        f"**Measured:** unique Tuesday games {result['tuesday_games']}; "
        f"unique pool-deadline games {result['deadline_games']}; "
        f"overlap {result['overlap_games']}; same-product overlap {result['same_product_games']}; "
        f"same-product/station/kickoff/forecast-valid-time pairs "
        f"{result['same_product_target_games']}; "
        f"pairs also satisfying issuance order and ingestion cutoffs {result['eligible_games']}.",
        "",
        f"**Measured:** {result['archives_with_ingestion_fields']}/{len(paths)} archives carry "
        "a recognized row-level ingestion timestamp. Archive build timestamps are collection "
        "metadata, not evidence of ingestion before historical pool deadlines. "
        "The legacy Tuesday product is identified from its manifest's explicit `model=` source "
        "and Tuesday cutoff column; an unlabelled checkpoint remains unknown.",
        "",
        "**Read:** scripts/ingest_forecast_archive.py:23-28 selects Tuesday MEX and deadline GFS; "
        "lines 176-187 select each bulletin's nearest valid time, so a shared game identifier "
        "alone cannot certify an identical forecast target. "
        "The inventory requires matching product/source, station, kickoff and forecast valid "
        "time before certifying a revision.",
        "",
        "| Season fold | Population games | IS / OOS / gap | Coefficients |",
        "|---|---:|---|---|",
    ]
    for season in range(2020, 2026):
        count = int(population["season"].eq(season).sum())
        lines.append(f"| {season} | {count} | Not estimated: source gate | Not fitted |")
    lines += [
        "",
        "**Measured:** decisive games scored: 0; win-loss record not estimated. "
        "IS/OOS opener accuracy, Brier, log loss, margin MAE, gaps, fold coefficients, "
        "calibration bands, season-block 95% intervals and `probability_positive` are "
        "not estimated because the declared source gate blocks replay. "
        "An inventory count has no sampling interval. "
        "Planned looks: (2 + 36 + 8*0)*(6 + 1) + 25 = 291; executed outcome looks: 0.",
        "",
        "**Inferred:** the existing cross-product pair cannot identify forecast innovation "
        "separately from product differences. This is a data gap, not a refuted mechanism "
        "or a negative effect. No research closure or registry record is supported. "
        "The roadmap source gate requires stopping here. Next: obtain comparable paired "
        "forecasts with identical valid times and historical issuance/ingestion provenance, "
        "then re-audit before the predeclared replay. "
        "No external source access or serving change ran.",
        "",
    ]
    report = "\n".join(
        line if line.startswith("|") else fill(line, width=100, break_long_words=False)
        for line in lines
    )
    return report, result


def main() -> int:
    parser = argparse.ArgumentParser(description="Inventory LEAD-75 local wind revision sources.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    report, result = inventory()
    args.output.write_text(report, encoding="utf-8")
    print(json.dumps({**result, "report": args.output.as_posix()}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
