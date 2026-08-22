"""Ingest FantasyFootballCalculator historical ADP snapshots (2010-2025).

Source: docs/data_source_scout_v5.md Section C rank 2. The FFC ADP REST API
(fantasyfootballcalculator.com/api/v1/adp/{scoring}) is free, keyless, and the
vendor requests attribution for use. Endpoint verified live this session for
both `ppr` and `standard` scoring at teams=12, position=all: each response is a
JSON object with a `meta` block carrying total_drafts, start_date, end_date
(the EXACT mock-draft window the aggregate was computed over -- this is what
earns the point-in-time grade A) plus a `players` list with name, position,
team, adp, times_drafted. Measured live: 2024 ppr 12-team returned
total_drafts=1371, window 2024-08-31..2024-09-01; 2010 standard 12-team
returned total_drafts=1535, window 2010-09-06..2010-09-08.

Team-code normalization: measured across eras, FFC already emits CURRENT
franchise codes even in archived seasons (2010 rows carry LAC/LAR/LV, not
SD/STL/OAK), so mapping to the repo's nflverse-style franchise identity is a
32-code passthrough plus a small alias table for observed variant spellings.
Any raw team value that survives neither becomes an ambiguity row recorded in
the normalization report (rate reported honestly in docs/ffc_adp_sourcing.md);
blank/FA/free-agent strings are expected ambiguities.

Rate limiting: self-imposed >=1.5s delay between requests (task requirement;
no published Crawl-delay was found for this host this session). Fetches go
through a curl subprocess, matching this project's existing precedent
(scripts/ingest_sagarin_ratings.py) where the local Python HTTP stack hung on
long-running fetch loops.

Usage:
    .\\.tools\\uv.exe run --no-sync python scripts/ingest_ffc_adp.py

    # Resume an interrupted run into the same snapshot dir:
    .\\.tools\\uv.exe run --no-sync python scripts/ingest_ffc_adp.py `
        --snapshot <YYYYMMDDTHHMMSSZ>

Writes:
    data/raw/ffc_adp/<ts>/responses/{scoring}_{year}.json   raw response bytes
    data/raw/ffc_adp/<ts>/manifest.json                     per-response sha256
                                                            + meta verbatim +
                                                            attribution string
    artifacts/ffc_adp/<ts>/adp_tidy.parquet                 one row per
                                                            player-year-scoring
    artifacts/ffc_adp/<ts>/team_top8_feasibility.parquet    per year/scoring/
                                                            franchise top-8
                                                            mean/min ADP
    artifacts/ffc_adp/<ts>/normalization_report.json        team-name mapping
                                                            audit + ambiguity
                                                            counts
    artifacts/ffc_adp/<ts>/metadata.json                    provenance-stamped
                                                            build manifest via
                                                            nfl_ats.provenance

The repo's release-blocking provenance contract
(tests/test_experiment_registry.py) requires every artifacts/-JSON-writing
script to go through ``write_experiment_artifact``. This is an ingestion build,
not an adjudicated screen, so the stamp records coverage metrics only -- and
``registry_root`` is pointed INSIDE the gitignored artifacts snapshot rather
than at the tracked ``registry/experiments/`` tree, because this task's brief
forbids writing any tracked registry JSON. The helper's ``registry_root``
parameter exists exactly for such redirected records.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from nfl_ats.provenance import artifact_provenance, write_experiment_artifact  # noqa: E402

BASE_URL = "https://fantasyfootballcalculator.com/api/v1/adp/{scoring}"
ATTRIBUTION = (
    "ADP data provided by FantasyFootballCalculator.com "
    "(https://fantasyfootballcalculator.com/adp), free public API; the vendor "
    "requests attribution for use of its data."
)
USER_AGENT = "nfl-ats-research/0.1 (private research; contact ryanpmcintire@gmail.com)"
RATE_LIMIT_SECONDS = 1.5
SNAPSHOT_DIR_RE = re.compile(r"^\d{8}T\d{6}Z$")
DEFAULT_SCORINGS = ("ppr", "standard")
DEFAULT_START_YEAR = 2010
DEFAULT_END_YEAR = 2025

FRANCHISE_CODES = frozenset(
    {
        "ARI",
        "ATL",
        "BAL",
        "BUF",
        "CAR",
        "CHI",
        "CIN",
        "CLE",
        "DAL",
        "DEN",
        "DET",
        "GB",
        "HOU",
        "IND",
        "JAX",
        "KC",
        "LAC",
        "LAR",
        "LV",
        "MIA",
        "MIN",
        "NE",
        "NO",
        "NYG",
        "NYJ",
        "PHI",
        "PIT",
        "SEA",
        "SF",
        "TB",
        "TEN",
        "WAS",
    }
)

TEAM_CODE_ALIASES = {
    "JAC": "JAX",
    "ARZ": "ARI",
    "OAK": "LV",
    "SD": "LAC",
    "STL": "LA",
    "WSH": "WAS",
}


def normalize_team_code(raw: str | None) -> str | None:
    if raw is None:
        return None
    key = raw.strip().upper()
    if key in FRANCHISE_CODES:
        return key
    return TEAM_CODE_ALIASES.get(key)


@dataclass
class RateLimiter:
    delay_seconds: float
    _last_request: float | None = field(default=None, init=False)

    def wait(self) -> None:
        if self._last_request is not None:
            elapsed = time.monotonic() - self._last_request
            remaining = max(self.delay_seconds - elapsed, 0.0)
            if remaining > 0:
                time.sleep(remaining)
        self._last_request = time.monotonic()


def _fetch(
    url: str,
    limiter: RateLimiter,
    *,
    timeout: int = 30,
    retries: int = 4,
) -> tuple[bytes, str]:
    last_error: Exception | None = None
    for attempt in range(retries):
        limiter.wait()
        try:
            completed = subprocess.run(
                [
                    "curl",
                    "-s",
                    "-S",
                    "-L",
                    "--compressed",
                    "--max-time",
                    str(timeout),
                    "-A",
                    USER_AGENT,
                    "-w",
                    "\n__CURL_HTTP_CODE__%{http_code}",
                    url,
                ],
                capture_output=True,
                timeout=timeout + 5,
                check=False,
            )
        except subprocess.TimeoutExpired as error:
            last_error = error
            print(f"    curl timeout ({attempt + 1}/{retries}) {url}", file=sys.stderr)
            time.sleep(3.0 * (attempt + 1))
            continue
        if completed.returncode != 0:
            stderr_text = completed.stderr.decode(errors="replace")[:300]
            last_error = RuntimeError(f"curl exit {completed.returncode}: {stderr_text}")
            print(
                f"    fetch failed ({attempt + 1}/{retries}) {url}: {last_error}", file=sys.stderr
            )
            time.sleep(3.0 * (attempt + 1))
            continue
        marker = b"\n__CURL_HTTP_CODE__"
        idx = completed.stdout.rfind(marker)
        if idx == -1:
            last_error = RuntimeError("curl output missing http-code marker")
            time.sleep(3.0 * (attempt + 1))
            continue
        body = completed.stdout[:idx]
        http_code = completed.stdout[idx + len(marker) :].decode(errors="replace").strip()
        if http_code != "200":
            last_error = RuntimeError(f"http status {http_code}")
            print(
                f"    fetch failed ({attempt + 1}/{retries}) {url}: status {http_code}",
                file=sys.stderr,
            )
            if http_code in ("404", "400"):
                break
            time.sleep(3.0 * (attempt + 1))
            continue
        if not body:
            last_error = RuntimeError("empty response body")
            time.sleep(3.0 * (attempt + 1))
            continue
        return body, http_code
    assert last_error is not None
    raise last_error


def resolve_snapshot_dir(out_dir: Path, snapshot: str | None) -> Path:
    if snapshot is not None:
        snapshot_dir = out_dir / snapshot
        snapshot_dir.mkdir(parents=True, exist_ok=True)
        return snapshot_dir
    new_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    snapshot_dir = out_dir / new_id
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    return snapshot_dir


@dataclass
class ResponseRecord:
    scoring: str
    year: int
    url: str
    http_status: str
    sha256: str
    size_bytes: int
    source: str
    meta: dict[str, Any] | None
    n_players: int | None
    error: str | None


def ingest(
    snapshot_dir: Path,
    scorings: list[str],
    start_year: int,
    end_year: int,
    limiter: RateLimiter,
) -> list[ResponseRecord]:
    responses_dir = snapshot_dir / "responses"
    responses_dir.mkdir(parents=True, exist_ok=True)

    records: list[ResponseRecord] = []
    combos = [(scoring, year) for scoring in scorings for year in range(start_year, end_year + 1)]
    print(f"{len(combos)} scoring-year combinations to ensure")

    for i, (scoring, year) in enumerate(combos):
        url = f"{BASE_URL.format(scoring=scoring)}?teams=12&year={year}&position=all"
        file_path = responses_dir / f"{scoring}_{year}.json"

        if file_path.exists():
            raw = file_path.read_bytes()
            source = "cached"
            http_status = "200"
            print(f"  [{i + 1}/{len(combos)}] cached {scoring} {year}")
        else:
            try:
                raw, http_status = _fetch(url, limiter)
            except Exception as error:
                records.append(
                    ResponseRecord(
                        scoring=scoring,
                        year=year,
                        url=url,
                        http_status="error",
                        sha256="",
                        size_bytes=0,
                        source="failed",
                        meta=None,
                        n_players=None,
                        error=str(error),
                    )
                )
                print(f"  [{i + 1}/{len(combos)}] FAILED {scoring} {year}: {error}")
                continue
            file_path.write_bytes(raw)
            source = "fetched"

        sha256 = hashlib.sha256(raw).hexdigest()
        meta: dict[str, Any] | None = None
        n_players: int | None = None
        error: str | None = None
        try:
            payload = json.loads(raw.decode("utf-8"))
            if payload.get("status") != "Success":
                error = f"api_status_{payload.get('status')!r}"
            else:
                meta = payload.get("meta")
                players = payload.get("players", [])
                n_players = len(players)
        except Exception as parse_error:
            error = f"parse_exception: {parse_error}"

        records.append(
            ResponseRecord(
                scoring=scoring,
                year=year,
                url=url,
                http_status=http_status,
                sha256=sha256,
                size_bytes=len(raw),
                source=source,
                meta=meta,
                n_players=n_players,
                error=error,
            )
        )
        if i % 8 == 7 or i == len(combos) - 1:
            print(
                f"  [{i + 1}/{len(combos)}] {scoring} {year}: drafts="
                f"{meta['total_drafts'] if meta else '?'} players={n_players}"
            )

    return records


def write_raw_manifest(snapshot_dir: Path, records: list[ResponseRecord]) -> Path:
    manifest_path = snapshot_dir / "manifest.json"
    manifest = {
        "source": "FantasyFootballCalculator historical ADP REST API",
        "endpoint_pattern": BASE_URL + "?teams=12&year={year}&position=all",
        "attribution": ATTRIBUTION,
        "fetched_at_utc": datetime.now(UTC).isoformat(),
        "user_agent": USER_AGENT,
        "rate_limit_seconds": RATE_LIMIT_SECONDS,
        "responses": [
            {
                "scoring": r.scoring,
                "year": r.year,
                "url": r.url,
                "file": f"responses/{r.scoring}_{r.year}.json" if r.source != "failed" else None,
                "sha256": r.sha256 or None,
                "size_bytes": r.size_bytes,
                "http_status": r.http_status,
                "source": r.source,
                "meta": r.meta,
                "n_players": r.n_players,
                "error": r.error,
            }
            for r in records
        ],
        "usage_note": (
            "Private research caching only; never republish raw rows. Attribution "
            "string above satisfies the vendor's stated request."
        ),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest_path


def build_tidy(records: list[ResponseRecord], snapshot_dir: Path) -> pd.DataFrame:
    responses_dir = snapshot_dir / "responses"
    rows: list[dict[str, Any]] = []
    for record in records:
        if record.source == "failed":
            continue
        file_path = responses_dir / f"{record.scoring}_{record.year}.json"
        if not file_path.exists():
            continue
        payload = json.loads(file_path.read_bytes().decode("utf-8"))
        if payload.get("status") != "Success":
            continue
        meta = payload["meta"]
        window_start = meta.get("start_date")
        window_end = meta.get("end_date")
        for player in payload.get("players", []):
            rows.append(
                {
                    "year": record.year,
                    "scoring": record.scoring,
                    "player": player.get("name"),
                    "position": player.get("position"),
                    "team": player.get("team"),
                    "adp": player.get("adp"),
                    "times_drafted": player.get("times_drafted"),
                    "window_start": window_start,
                    "window_end": window_end,
                    "team_code": normalize_team_code(player.get("team")),
                }
            )
    frame = pd.DataFrame(rows)
    if not frame.empty:
        frame = frame.sort_values(["year", "scoring", "adp"]).reset_index(drop=True)
    return frame


def build_team_top8(tidy: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    mapped = tidy.dropna(subset=["team_code", "adp"]).copy()
    mapped = mapped[mapped["times_drafted"] > 0]
    mapped = mapped.sort_values(["year", "scoring", "team_code", "adp"])
    groups: list[pd.DataFrame] = []
    for _, group in mapped.groupby(["year", "scoring", "team_code"], sort=True):
        groups.append(group.head(8))
    top8 = pd.concat(groups) if groups else pd.DataFrame()

    agg_rows: list[dict[str, Any]] = []
    for (year, scoring, team_code), group in top8.groupby(
        ["year", "scoring", "team_code"], sort=True
    ):
        agg_rows.append(
            {
                "year": int(year),
                "scoring": scoring,
                "franchise_code": team_code,
                "n_top8": len(group),
                "mean_adp_top8": round(float(group["adp"].mean()), 3),
                "min_adp_top8": round(float(group["adp"].min()), 3),
                "mean_times_drafted_top8": round(float(group["times_drafted"].mean()), 1),
            }
        )
    aggregate = pd.DataFrame(agg_rows)

    unmapped_mask = tidy["team_code"].isna()
    unmapped_values = tidy.loc[unmapped_mask, "team"].fillna("<missing>").value_counts().to_dict()
    report = {
        "rows_total": len(tidy),
        "rows_mapped_to_franchise": int((~unmapped_mask).sum()),
        "rows_unmapped_ambiguous": int(unmapped_mask.sum()),
        "ambiguity_rate": (round(float(unmapped_mask.mean()), 6) if len(tidy) else 0.0),
        "unmapped_raw_values": {str(k): int(v) for k, v in unmapped_values.items()},
        "alias_table_applied": TEAM_CODE_ALIASES,
        "note": (
            "FFC already emits current franchise codes (LAC/LAR/LV) even in "
            "archived seasons, so mapping is a 32-code passthrough plus aliases; "
            "unmapped values are blank/FA/free-agent strings."
        ),
    }
    return aggregate, report


def write_artifacts(
    artifacts_dir: Path,
    snapshot_dir: Path,
    tidy: pd.DataFrame,
    aggregate: pd.DataFrame,
    report: dict[str, Any],
    records: list[ResponseRecord],
) -> None:
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    tidy.to_parquet(artifacts_dir / "adp_tidy.parquet", index=False)
    aggregate.to_parquet(artifacts_dir / "team_top8_feasibility.parquet", index=False)
    (artifacts_dir / "normalization_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )

    configuration = {
        "source": "FantasyFootballCalculator historical ADP REST API",
        "scorings": sorted({r.scoring for r in records}),
        "start_year": min(r.year for r in records),
        "end_year": max(r.year for r in records),
        "teams": 12,
        "position": "all",
        "raw_snapshot": str(snapshot_dir.resolve()),
    }
    metadata = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "attribution": ATTRIBUTION,
        "provenance": artifact_provenance(
            configuration, artifacts_dir / "adp_tidy.parquet", project_root=REPO
        ),
    }
    metrics = {
        "tidy_rows": len(tidy),
        "team_aggregate_rows": len(aggregate),
        "responses_ok": sum(1 for r in records if r.error is None),
        "responses_failed": sum(1 for r in records if r.error is not None),
        "total_drafts_captured": sum((r.meta or {}).get("total_drafts", 0) for r in records),
        "ambiguous_team_rows": report["rows_unmapped_ambiguous"],
        "ambiguity_rate": report["ambiguity_rate"],
    }
    write_experiment_artifact(
        artifacts_dir,
        "metadata.json",
        metadata,
        command="ingest_ffc_adp",
        metrics=metrics,
        notes=(
            "Ingestion + feasibility build manifest (raw ADP snapshots and "
            "derived tidy/team-top8 tables); NOT an adjudicated screen -- no "
            "ATS evaluation was run. Registry row kept inside the gitignored "
            "snapshot via registry_root because tracked-registry writes are "
            "out of this ingest's scope."
        ),
        source="scripts/ingest_ffc_adp.py",
        registry_root=artifacts_dir / "experiment_registry",
        project_root=REPO,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--raw-out", type=Path, default=Path("data/raw/ffc_adp"))
    parser.add_argument("--artifacts-out", type=Path, default=Path("artifacts/ffc_adp"))
    parser.add_argument("--snapshot", default=None, metavar="YYYYMMDDTHHMMSSZ")
    parser.add_argument("--start-year", type=int, default=DEFAULT_START_YEAR)
    parser.add_argument("--end-year", type=int, default=DEFAULT_END_YEAR)
    parser.add_argument("--scorings", nargs="+", default=list(DEFAULT_SCORINGS))
    parser.add_argument("--delay", type=float, default=RATE_LIMIT_SECONDS)
    args = parser.parse_args()

    args.raw_out.mkdir(parents=True, exist_ok=True)
    snapshot_dir = resolve_snapshot_dir(args.raw_out, args.snapshot)
    print(f"Snapshot directory: {snapshot_dir}")

    limiter = RateLimiter(max(args.delay, RATE_LIMIT_SECONDS))
    records = ingest(snapshot_dir, args.scorings, args.start_year, args.end_year, limiter)
    manifest_path = write_raw_manifest(snapshot_dir, records)
    ok_count = sum(1 for r in records if r.error is None)
    fail_count = sum(1 for r in records if r.error is not None)
    total_drafts = sum((r.meta or {}).get("total_drafts", 0) for r in records)
    print(f"Ingest done: ok={ok_count} failed={fail_count} total_drafts={total_drafts}")
    print(f"Manifest: {manifest_path}")

    artifacts_dir = args.artifacts_out / snapshot_dir.name
    tidy = build_tidy(records, snapshot_dir)
    aggregate, report = build_team_top8(tidy)
    write_artifacts(artifacts_dir, snapshot_dir, tidy, aggregate, report, records)
    print(f"Tidy rows: {len(tidy)}; team-aggregate rows: {len(aggregate)}")
    print(f"Ambiguous team rows: {report['rows_unmapped_ambiguous']} / {report['rows_total']}")
    print(f"Artifacts: {artifacts_dir}")


if __name__ == "__main__":
    main()
