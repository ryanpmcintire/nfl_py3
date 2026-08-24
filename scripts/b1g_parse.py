"""Parse Big Ten Football Availability Report PDFs into a tidy per-week table.

Input is a snapshot directory produced by ``scripts/pilot_bigten_availability.py``
(default: the newest run under ``data/raw/bigten_availability/``), containing one
weekly availability-report PDF per Big Ten week plus its SHA-256 ``manifest.json``.
Each PDF carries one page per Big Ten school listing that school's unavailable
players grouped under designation headers (OUT, QUESTIONABLE).

Observed layout generations, all handled here (see docs/b1g_parse.md):

1. Standard game page: report header, "Week N: <dates>" line, an all-caps B1G
   matchup line ("ILLINOIS vs. Toledo" / "NEBRASKA at Minnesota"), a kickoff/
   broadcast line, then designation sections of "<number> <name>" player lines,
   each section possibly reading "None" when empty. Weeks 4+ append "(season)"/
   "(Season)" annotations to some player names.
2. Bye page (weeks 5+): team name alone on the matchup line, a "BYE" line, and
   the two designation headers collapsed onto one empty "OUT QUESTIONABLE" line.

Extraction quirks handled explicitly: pypdf splits a leading capital T/Y from the
rest of the word ("T yson Rooks"), which is repaired conservatively (only for the
measured artifact letters T and Y); curly apostrophes (U+2019) are normalized;
the raw extracted name is always preserved alongside the cleaned form.

Every page must parse structurally: unparseable pages are collected as failures,
written to the run report, and make the script exit nonzero -- never silently
dropped. Team names are mapped to stable conference codes with a documented
crosswalk to the canonical CFB display names used by ``data/processed/
cfb_game_features.parquet``.

Source ingestion + validation ONLY: no model features, no ATS screen, no
tracked-registry write (the experiment stamp is redirected inside the gitignored
artifact directory).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import pypdf
from pypdf import PdfReader

from nfl_ats.provenance import artifact_provenance, sha256_file, write_experiment_artifact

REPORT_HEADER = "BIG TEN FOOTBALL AVAILABILITY REPORT"
WEEK_LINE_RE = re.compile(r"^Week (?P<week>\d+):\s*(?P<dates>.+?)\s*$")
MATCHUP_RE = re.compile(
    r"^(?P<team>[A-Z][A-Z .&'\x2019-]*?)\s+(?:vs\.?|at\.?)\s+(?P<opponent>.+?)\s*$"
)
PLAYER_LINE_RE = re.compile(r"^(?P<number>\d{1,2})\s+(?P<name>\S.*?)\s*$")
ANNOTATION_RE = re.compile(r"\s*\((?P<annotation>[^()]+)\)\s*$")
# Measured across all six snapshot PDFs: pypdf inserts a spurious space after a
# capital T (53 occurrences) or Y (3 occurrences); no other letter is affected.
SPLIT_ARTIFACT_LETTERS = "TY"
SPLIT_ARTIFACT_RE = re.compile(rf"\b([{SPLIT_ARTIFACT_LETTERS}]) (?=[a-z.])")

SEASON_ANNOTATION = "season"

TEAM_CODES: dict[str, tuple[str, str]] = {
    # raw all-caps PDF name -> (stable code, canonical CFB display name as used by
    # data/processed/cfb_game_features.parquet)
    "ILLINOIS": ("ILL", "Illinois"),
    "INDIANA": ("IND", "Indiana"),
    "IOWA": ("IOWA", "Iowa"),
    "MARYLAND": ("MD", "Maryland"),
    "MICHIGAN": ("MICH", "Michigan"),
    "MICHIGAN STATE": ("MSU", "Michigan State"),
    "MINNESOTA": ("MINN", "Minnesota"),
    "NEBRASKA": ("NEB", "Nebraska"),
    "NORTHWESTERN": ("NU", "Northwestern"),
    "OHIO STATE": ("OSU", "Ohio State"),
    "PENN STATE": ("PSU", "Penn State"),
    "PURDUE": ("PUR", "Purdue"),
    "RUTGERS": ("RUTG", "Rutgers"),
    "WISCONSIN": ("WISC", "Wisconsin"),
}

TIDY_COLUMNS = [
    "season",
    "week",
    "team_code",
    "cfb_display_name",
    "team_raw",
    "opponent_raw",
    "venue_side",
    "player_number",
    "player_raw",
    "player",
    "annotation",
    "designation_raw",
    "designation_norm",
    "source_file",
    "source_sha256",
    "source_url",
    "page_index",
    "parsed_at_utc",
]

REPO = Path(__file__).resolve().parents[1]
RAW_ROOT = REPO / "data" / "raw" / "bigten_availability"
ARTIFACT_ROOT = REPO / "artifacts" / "b1g_parse"


class PageParseError(ValueError):
    """A PDF page did not match any known availability-report layout."""


def utc_now_iso() -> str:
    return datetime.now(UTC).isoformat()


@dataclass(frozen=True)
class PlayerEntry:
    number: int
    name_raw: str
    name: str
    annotation: str | None
    designation_raw: str


@dataclass
class ParsedPage:
    page_index: int
    week: int
    dates_raw: str
    team_raw: str
    team_code: str
    cfb_display_name: str
    opponent_raw: str | None
    venue_side: str | None
    is_bye: bool
    entries: list[PlayerEntry] = field(default_factory=list)


def clean_player_name(name_raw: str) -> tuple[str, str | None]:
    """Repair extraction artifacts; return (clean name, trailing annotation)."""

    annotation: str | None = None
    name = name_raw
    match = ANNOTATION_RE.search(name)
    if match is not None:
        candidate = match.group("annotation").strip().casefold()
        if candidate == SEASON_ANNOTATION:
            annotation = SEASON_ANNOTATION
            name = name[: match.start()].rstrip()
    name = name.replace("\u2019", "'")
    name = SPLIT_ARTIFACT_RE.sub(r"\1", name)
    name = re.sub(r"\s+", " ", name).strip()
    return name, annotation


def _require_team(team_raw: str) -> tuple[str, str]:
    entry = TEAM_CODES.get(re.sub(r"\s+", " ", team_raw).strip())
    if entry is None:
        raise PageParseError(f"unmapped Big Ten team name: {team_raw!r}")
    return entry


def parse_page_text(text: str, page_index: int, *, context: str) -> ParsedPage:
    """Parse one page of a weekly availability report into structured entries."""

    lines = [line.strip() for line in text.splitlines()]
    lines = [line for line in lines if line]
    if not lines or lines[0] != REPORT_HEADER:
        raise PageParseError(
            f"{context}: missing report header; first line was {lines[0] if lines else None!r}"
        )

    week_match = WEEK_LINE_RE.match(lines[1]) if len(lines) > 1 else None
    if week_match is None:
        raise PageParseError(f"{context}: second line is not a Week line: {lines[1]!r}")
    week = int(week_match.group("week"))
    dates_raw = week_match.group("dates")

    matchup_match = MATCHUP_RE.match(lines[2]) if len(lines) > 2 else None
    is_bye = False
    if matchup_match is not None:
        team_raw = matchup_match.group("team").strip()
        opponent_raw = matchup_match.group("opponent").strip()
        separator = lines[2][len(team_raw) :].lstrip()
        venue_side = "away" if separator.startswith("at") else "home"
    else:
        # Bye generation: the school's name stands alone, followed by a BYE line
        # and both designation headers collapsed onto one empty line.
        team_raw = lines[2]
        if len(lines) < 4 or lines[3] != "BYE":
            raise PageParseError(
                f"{context}: third line matches neither a matchup nor a bye page: {lines[2:]!r}"
            )
        opponent_raw = None
        venue_side = None
        is_bye = True
    team_code, cfb_display_name = _require_team(team_raw)

    page = ParsedPage(
        page_index=page_index,
        week=week,
        dates_raw=dates_raw,
        team_raw=team_raw,
        team_code=team_code,
        cfb_display_name=cfb_display_name,
        opponent_raw=opponent_raw,
        venue_side=venue_side,
        is_bye=is_bye,
    )
    if is_bye:
        tail = lines[4:]
        if tail != ["OUT QUESTIONABLE"]:
            raise PageParseError(f"{context}: unexpected content on bye page: {tail!r}")
        return page

    current_designation: str | None = None
    seen_sections: set[str] = set()
    for line in lines[3:]:
        if line in ("OUT", "QUESTIONABLE"):
            current_designation = line
            seen_sections.add(line)
            continue
        if current_designation is None:
            # Kickoff/broadcast lines precede the first designation header.
            if "|" in line or re.match(r"^\d{1,2}:\d{2}", line):
                continue
            raise PageParseError(f"{context}: line before any designation header: {line!r}")
        if line == "None":
            continue
        player_match = PLAYER_LINE_RE.match(line)
        if player_match is None:
            raise PageParseError(
                f"{context}: unparseable player line under {current_designation}: {line!r}"
            )
        name_raw = player_match.group("name")
        name, annotation = clean_player_name(name_raw)
        page.entries.append(
            PlayerEntry(
                number=int(player_match.group("number")),
                name_raw=name_raw,
                name=name,
                annotation=annotation,
                designation_raw=current_designation,
            )
        )
    if not seen_sections:
        raise PageParseError(f"{context}: no designation sections found")
    missing = {"OUT", "QUESTIONABLE"} - seen_sections
    if missing:
        raise PageParseError(f"{context}: missing designation sections: {sorted(missing)}")
    return page


def assert_week_matches(page: ParsedPage, expected_week: int, context: str) -> None:
    if page.week != expected_week:
        raise PageParseError(
            f"{context}: page says Week {page.week}, manifest says Week {expected_week}"
        )


def parse_snapshot_pdf(
    pdf_path: Path,
    *,
    expected_week: int,
    source_url: str,
    source_sha256: str,
    parsed_at_utc: str,
) -> tuple[list[dict[str, object]], list[ParsedPage]]:
    """Parse one weekly PDF; return (tidy rows, parsed pages). Raises on failure."""

    reader = PdfReader(pdf_path)
    pages: list[ParsedPage] = []
    for index, pdf_page in enumerate(reader.pages):
        text = pdf_page.extract_text()
        page = parse_page_text(text, index, context=f"{pdf_path.name} page {index}")
        assert_week_matches(page, expected_week, f"{pdf_path.name} page {index}")
        pages.append(page)

    rows: list[dict[str, object]] = []
    for page in pages:
        for entry in page.entries:
            rows.append(
                {
                    "season": None,  # filled by caller (the snapshot's season)
                    "week": page.week,
                    "team_code": page.team_code,
                    "cfb_display_name": page.cfb_display_name,
                    "team_raw": page.team_raw,
                    "opponent_raw": page.opponent_raw,
                    "venue_side": page.venue_side,
                    "player_number": entry.number,
                    "player_raw": entry.name_raw,
                    "player": entry.name,
                    "annotation": entry.annotation,
                    "designation_raw": entry.designation_raw,
                    "designation_norm": entry.designation_raw.casefold(),
                    "source_file": pdf_path.name,
                    "source_sha256": source_sha256,
                    "source_url": source_url,
                    "page_index": page.page_index,
                    "parsed_at_utc": parsed_at_utc,
                }
            )
    return rows, pages


def latest_raw_run(root: Path) -> Path:
    candidates = sorted(path for path in root.iterdir() if path.is_dir())
    if not candidates:
        raise FileNotFoundError(f"no snapshot runs under {root}")
    return candidates[-1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--raw-run",
        type=Path,
        default=None,
        help="Snapshot directory with weekly PDFs + manifest.json "
        "(default: newest under data/raw/bigten_availability)",
    )
    parser.add_argument("--season", type=int, default=2023, help="Season of the snapshot")
    parser.add_argument("--artifact-dir", type=Path, default=None, help=argparse.SUPPRESS)
    args = parser.parse_args()

    raw_dir = (args.raw_run or latest_raw_run(RAW_ROOT)).resolve()
    manifest_path = raw_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    artifact_dir = args.artifact_dir or (ARTIFACT_ROOT / run_id)
    artifact_dir.mkdir(parents=True, exist_ok=True)

    parsed_at_utc = utc_now_iso()
    rows: list[dict[str, object]] = []
    failures: list[dict[str, str]] = []
    file_reports: list[dict[str, object]] = []
    all_pages: list[ParsedPage] = []

    for entry in manifest["entries"]:
        filename = entry["filename"]
        pdf_path = raw_dir / filename
        file_report: dict[str, object] = {
            "week": entry["week"],
            "filename": filename,
            "source_url": entry["source_url"],
            "sha256_expected": entry["sha256"],
        }
        actual_sha = None
        if pdf_path.is_file():
            actual_sha = sha256_file(pdf_path)
        file_report["sha256_actual"] = actual_sha
        if actual_sha != entry["sha256"]:
            failures.append(
                {
                    "file": filename,
                    "error": (
                        "sha256 mismatch"
                        if actual_sha is not None
                        else "file missing from snapshot"
                    ),
                }
            )
            file_report["outcome"] = "hash_mismatch_or_missing"
            file_reports.append(file_report)
            continue
        try:
            file_rows, pages = parse_snapshot_pdf(
                pdf_path,
                expected_week=int(entry["week"]),
                source_url=entry["source_url"],
                source_sha256=actual_sha or "",
                parsed_at_utc=parsed_at_utc,
            )
        except (PageParseError, ValueError, OSError) as error:
            failures.append({"file": filename, "error": str(error)})
            file_report["outcome"] = "parse_failed"
            file_reports.append(file_report)
            continue
        for row in file_rows:
            row["season"] = args.season
        rows.extend(file_rows)
        all_pages.extend(pages)
        file_report["outcome"] = "parsed"
        file_report["pages"] = len(pages)
        file_report["bye_pages"] = sum(1 for page in pages if page.is_bye)
        file_report["rows"] = len(file_rows)
        file_reports.append(file_report)

    frame = pd.DataFrame(rows, columns=TIDY_COLUMNS)
    frame.to_csv(artifact_dir / "tidy_rows.csv", index=False)
    frame.to_parquet(artifact_dir / "tidy_rows.parquet", index=False)

    weeks = sorted({int(page.week) for page in all_pages})
    teams_per_week = {
        str(week): sorted({page.team_code for page in all_pages if page.week == week})
        for week in weeks
    }
    designation_counts = Counter(str(value) for value in frame["designation_norm"])
    annotation_counts = Counter(str(value) for value in frame["annotation"].dropna())
    files_parsed = sum(1 for report in file_reports if report["outcome"] == "parsed")
    files_failed = len(file_reports) - files_parsed
    pages_parsed = sum(int(report.get("pages", 0)) for report in file_reports)
    bye_pages = sum(int(report.get("bye_pages", 0)) for report in file_reports)
    yield_stats = {
        "files_total": len(manifest["entries"]),
        "files_parsed": files_parsed,
        "files_failed": files_failed,
        "pages_total": pages_parsed,
        "bye_pages": bye_pages,
        "player_rows": len(frame),
        "weeks": weeks,
        "teams_per_week": teams_per_week,
        "designations": dict(sorted(designation_counts.items())),
        "annotations": dict(sorted(annotation_counts.items())),
    }
    status: dict[str, object] = {
        "run_id": run_id,
        "created_at_utc": utc_now_iso(),
        "raw_dir": str(raw_dir),
        "raw_manifest_created_at_utc": manifest.get("created_at_utc"),
        "hub_url": manifest.get("hub_url"),
        "season": args.season,
        "parser": f"pypdf {pypdf.__version__}",
        "yield": yield_stats,
        "files": file_reports,
        "failures": failures,
        "schema": {
            "grain": "one row per (week, B1G team, listed player, designation)",
            "columns": TIDY_COLUMNS,
            "team_key": "team_code (stable conference code; see docs/b1g_parse.md)",
        },
    }

    input_manifest = {
        "run_id": run_id,
        "created_at_utc": status["created_at_utc"],
        "raw_dir": str(raw_dir),
        "season": args.season,
        "entries": [
            {
                "week": report["week"],
                "filename": report["filename"],
                "source_url": report["source_url"],
                "sha256": report["sha256_expected"],
            }
            for report in file_reports
        ],
    }
    # Named so artifact_provenance() picks it up as the feature table's manifest.
    (artifact_dir / "tidy_rows.manifest.json").write_text(
        json.dumps(input_manifest, indent=2), encoding="utf-8"
    )
    (artifact_dir / "parse_report.json").write_text(json.dumps(status, indent=2), encoding="utf-8")

    status["provenance"] = artifact_provenance(
        {
            "source": "bigten.org weekly Football Availability Report PDFs",
            "raw_dir": str(raw_dir),
            "season": args.season,
            "parser": f"pypdf {pypdf.__version__}",
        },
        artifact_dir / "tidy_rows.csv",
        project_root=REPO,
    )
    write_experiment_artifact(
        artifact_dir,
        "metadata.json",
        status,
        command="b1g_parse",
        metrics={
            "files_parsed": files_parsed,
            "files_failed": files_failed,
            "pages_parsed": pages_parsed,
            "bye_pages": bye_pages,
            "player_rows": len(frame),
            "weeks_count": len(weeks),
        },
        notes=(
            "B1G availability PDF ingest+parse only: tidy rows, yield stats and a "
            "loud failure ledger; no model features, no ATS screen, no XLG-03 join, "
            "no tracked-registry write (registry redirected inside this gitignored "
            "artifact snapshot)."
        ),
        source="scripts/b1g_parse.py",
        registry_root=artifact_dir / "experiment_registry",
        project_root=REPO,
    )

    print(
        json.dumps(
            {
                "rows": len(frame),
                "weeks": weeks,
                "files_parsed": files_parsed,
                "files_failed": files_failed,
                "pages_parsed": pages_parsed,
                "bye_pages": bye_pages,
                "designations": dict(sorted(designation_counts.items())),
                "failures": failures,
                "artifact_dir": str(artifact_dir),
            },
            indent=2,
        )
    )

    if failures:
        print(
            f"FAILING LOUDLY: {len(failures)} file-level failure(s); see "
            f"{artifact_dir / 'parse_report.json'}",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
