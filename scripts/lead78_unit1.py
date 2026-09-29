from __future__ import annotations

import argparse
import csv
import json
import os
import re
import textwrap
from collections import Counter
from datetime import datetime
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

SEASONS = tuple(range(2020, 2026))
LANE = Path("docs/lanes/lead78.md")
REPORT = Path("docs/lead78_unit1.md")
BASELINE = Path("artifacts/four_term_probability/20260914T222345Z/per_game.csv")
CALENDAR_FIELDS = ["game_id", "season", "week", "game_type", "gameday"]
FINAL_FIELDS = {
    "published_final_at",
    "published_final_at_utc",
    "final_published_at",
    "completed_at",
    "completed_at_utc",
    "game_end_time",
    "game_end_timestamp",
    "end_time",
    "final_at",
    "final_timestamp",
}


def verify_declaration() -> None:
    declaration = " ".join(LANE.read_text(encoding="utf-8-sig").split())
    row = next(
        line
        for line in Path("ROADMAP.md").read_text(encoding="utf-8-sig").splitlines()
        if line.startswith("| LEAD-78 ")
    )
    context = Path("docs/lanes/ideation-2026-09-29b.md").read_text(encoding="utf-8-sig")
    paragraphs = [
        line
        for line in context.splitlines()
        if line.startswith(
            (
                "Rows fix population",
                "Chronology-purged LOSO",
                "Pair combined-model",
                "Count reporting cells",
            )
        )
    ]
    if len(paragraphs) != 4 or any(
        " ".join(value.split()) not in declaration for value in [row, *paragraphs]
    ):
        raise ValueError("Save the unchanged LEAD-78 row and Protocol B before inventory")


def table(headers: list[str], rows: list[list[object]]) -> str:
    return "\n".join(
        [
            "| " + " | ".join(headers) + " |",
            "| " + " | ".join("---" for _ in headers) + " |",
            *("| " + " | ".join(str(value) for value in row) + " |" for row in rows),
        ]
    )


def aware_timestamp(value: object) -> bool:
    if not isinstance(value, str):
        return False
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return False
    return parsed.tzinfo is not None and parsed.utcoffset() is not None


def run(data_root: Path) -> None:
    verify_declaration()
    pa.set_cpu_count(2)
    paths = sorted((data_root / "raw").glob("*/schedules.parquet"))
    schemas: Counter = Counter()
    completion_files = 0
    fetched: list[str] = []
    calendars = []
    for path in paths:
        columns = set(pq.ParquetFile(path).schema_arrow.names)
        schemas[
            tuple(
                sorted(
                    c
                    for c in columns
                    if re.search(r"time|date|day|final|complet|publish|ingest", c)
                )
            )
        ] += 1
        completion_files += int(bool(columns & FINAL_FIELDS))
        manifest = path.with_name("manifest.json")
        if manifest.is_file():
            metadata = json.loads(manifest.read_text(encoding="utf-8-sig"))
            if aware_timestamp(metadata.get("fetched_at_utc")):
                fetched.append(str(metadata["fetched_at_utc"]))
        if set(CALENDAR_FIELDS).issubset(columns):
            frame = pq.read_table(path, columns=CALENDAR_FIELDS, use_threads=False).to_pandas()
            frame = frame.loc[frame.season.isin(SEASONS) & frame.game_type.eq("REG")].copy()
            if frame.game_id.duplicated().any():
                raise ValueError(f"Duplicate calendar game IDs in {path}")
            frame["calendar_source"] = path.as_posix()
            calendars.append(frame)
    calendar = (
        pd.concat(calendars, ignore_index=True).drop_duplicates("game_id", keep="last")
        if calendars
        else pd.DataFrame(columns=[*CALENDAR_FIELDS, "calendar_source"])
    )
    calendar["weekday"] = pd.to_datetime(calendar.gameday, errors="raise").dt.dayofweek
    capture_paths = sorted(
        (data_root / "splash").glob("[0-9][0-9][0-9][0-9]_week[0-9][0-9]_*.json")
    )
    captures: Counter = Counter()
    timestamped: Counter = Counter()
    locks: Counter = Counter()
    weeks: dict[int, set[int]] = {season: set() for season in SEASONS}
    for path in capture_paths:
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
        season = int(payload["season"])
        if season in SEASONS:
            captures[season] += 1
            weeks[season].add(int(payload["week"]))
            timestamped[season] += int(aware_timestamp(payload.get("captured_at_et")))
            locks[season] += int(aware_timestamp(payload.get("picks_lock_et")))
    baseline_columns: list[str] = []
    if BASELINE.is_file():
        with BASELINE.open(encoding="utf-8-sig", newline="") as handle:
            baseline_columns = next(csv.reader(handle))
    baseline_time_fields = [
        c for c in baseline_columns if re.search(r"timestamp|issued|ingested|deadline|captured", c)
    ]
    missing = []
    if completion_files == 0:
        missing.append(
            "No inspected schedule archive exposes a recognized final/completion timestamp field."
        )
    absent = ", ".join(str(s) for s in SEASONS if not captures[s])
    if absent:
        missing.append(f"No local Splash capture files for scheduled seasons {absent}.")
    if not all(locks[s] for s in SEASONS):
        missing.append("Timezone-aware pool lock timestamps do not cover all scheduled seasons.")
    status = "source_gap" if missing else "metadata_requires_validation"
    season_rows = []
    for season in SEASONS:
        frame = calendar.loc[calendar.season.eq(season)]
        season_rows.append(
            [
                season,
                len(frame),
                int(frame.weekday.isin([3, 5]).sum()),
                int(frame.weekday.eq(6).sum()),
                captures[season],
                len(weeks[season]),
                locks[season],
            ]
        )
    sections = [
        "# LEAD-78 unit one: completion and state-update source audit",
        f"**Measured:** `{status}`; zero fits, scored games and outcome looks; 298 looks "
        f"preregistered.",
        "**Inferred:** this source gap leaves the effect unmeasured. It neither refutes the "
        "mechanism nor closes research.",
        "## Declaration and scope",
        "**Read:** the unchanged roadmap row and Protocol B were copied to "
        "`docs/lanes/lead78.md` before this command. "
        "Population: 2020-2025 REG games with a later pick deadline; retain every eligible game. "
        "Target: authentic Tuesday frozen-opener cover. Terms: fixed score-only "
        "opponent-graph state, "
        "updated-minus-Tuesday implied home-margin delta, market move, and their joint "
        "fitted discrete-margin probability. "
        "Chronology-purged LOSO excludes target/later seasons and separates earlier "
        "training, selection and calibration. "
        "B=3, F=6, E=0; L=(3+36)(6+1)+25=298.",
        "**Measured:** inspected schedule parquet schemas and calendar columns, adjacent "
        "snapshot manifest metadata, "
        "Splash capture metadata and the referenced baseline CSV header. No score, margin, "
        "cover label, probability "
        "or market-line value was used. No source fetch, registry write, rebuild or "
        "served-card change ran.",
        f"Schedule scope: `{data_root.as_posix()}/raw/*/schedules.parquet`; "
        f"capture scope: `{data_root.as_posix()}/splash/YYYY_weekWW_*.json`; "
        f"baseline header: `{BASELINE.as_posix()}`.",
        "## Source counts",
        "**Measured:** exact inventory counts; statistical intervals do not apply.",
        table(
            ["Inventory", "Count / value"],
            [
                ["Schedule schemas inspected", len(paths)],
                ["Schemas with recognized completion fields", completion_files],
                ["Manifests with timezone-aware snapshot fetch times", len(fetched)],
                ["Splash capture files, all seasons", len(capture_paths)],
                ["Splash capture files, 2020-2025", sum(captures.values())],
                ["2020-2025 captures with timezone-aware capture times", sum(timestamped.values())],
                ["2020-2025 captures with timezone-aware pool lock times", sum(locks.values())],
                [
                    "Referenced baseline exists / column count",
                    f"{BASELINE.is_file()} / {len(baseline_columns)}",
                ],
            ],
        ),
        table(
            ["Snapshot count", "Temporal column names"],
            [[count, ", ".join(fields) or "none"] for fields, count in sorted(schemas.items())],
        ),
        "Snapshot fetch-time range: "
        + (f"`{min(fetched)}` to `{max(fetched)}`." if fetched else "unavailable."),
        "Baseline timestamp-header matches: " + (", ".join(baseline_time_fields) or "none") + ".",
        "## Calendar and capture coverage",
        "**Measured:** calendar counts are not certified eligible games; Thu/Sat kickoff "
        "dates do not establish final "
        "publication. Capture counts can include duplicates and do not establish Tuesday "
        "issuance or deadline validity. "
        "For each game, calendar metadata comes from the latest lexically named inspected "
        "snapshot containing it.",
        table(
            ["Season", "REG games", "Thu/Sat", "Sunday", "Capture files", "Weeks", "Lock times"],
            season_rows,
        ),
        table(
            ["Calendar source", "Games"],
            [
                [source, int(count)]
                for source, count in calendar.calendar_source.value_counts().sort_index().items()
            ],
        ),
        "## Gate and requested evaluation",
        "**Measured:** " + " ".join(missing or ["Source semantics remain unvalidated."]),
        "**Read:** LEAD-78 requires published-final timestamps and excludes unfinished "
        "games and later corrections. "
        "Protocol B requires issuance/ingestion before the pool deadline and authentic "
        "frozen openers. "
        "Snapshot fetch times and kickoff dates cannot substitute for historical "
        "final-publication times.",
        "**Inferred:** without that provenance no eligible population or state replay is "
        "certified. "
        "The baseline CSV header alone also cannot certify chronology-purged folds or a "
        "calibrated discrete-margin distribution.",
        "Decisive-game W-L-P record; IS/OOS opener accuracy, Brier, log loss, margin MAE and gaps; "
        "five-band reliability; season-block 95% intervals; and `probability_positive`: "
        "unavailable, not zero. "
        "Zero games scored; zero folds fitted. Calendar/capture counts above are not "
        "performance looks.",
        table(
            ["Scheduled fold", "IS / OOS / gap", "Coefficients", "95% CI / probability_positive"],
            [[s, "unavailable", "not fitted", "unavailable"] for s in SEASONS],
        ),
        "## Next unit",
        "Recover deadline-valid historical frozen openers and lock metadata, and score-only "
        "histories with "
        "published-final and ingestion timestamps preserving the version available then. "
        "Verify the fixed earlier-trained "
        "rating architecture and certified baseline folds before replay. Retain all "
        "eligible games, including zero-delta weeks. "
        "No registry command is supplied because no effect estimate or research verdict exists.",
        "Verification command: `.tools/uv.exe run --no-sync python scripts/lead78_unit1.py`.",
    ]
    rendered = [
        section
        if section.startswith("|")
        else textwrap.fill(section, width=180, break_long_words=False, break_on_hyphens=False)
        for section in sections
    ]
    REPORT.write_text("\n\n".join(rendered) + "\n", encoding="utf-8")
    print(
        f"status={status}; schedule_snapshots={len(paths)}; completion_fields={completion_files}; "
        f"historical_captures={sum(captures.values())}; "
        f"historical_lock_times={sum(locks.values())}; "
        f"calendar_games={len(calendar)}; fits=0; scored_games=0; outcome_looks=0/298; "
        f"report={REPORT.as_posix()}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Audit LEAD-78 local sources without fitting.")
    parser.add_argument(
        "--data-root", type=Path, default=Path(os.environ.get("NFL_ATS_DATA_DIR", "data"))
    )
    run(parser.parse_args().data_root)
