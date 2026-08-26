"""Bulk-ingest the FULL-column nflverse injuries release, season by season,
into an immutable snapshot -- fixing a real column loss in the repo's
existing player pipeline.

**Measured, this session**: the repo's only local injury sources before this
script are (1) ``data/raw/nflcom_injuries/20260821T222602Z/injuries.parquet``,
an NFL.com scrape covering only seasons 2022-2024 (17,483 rows, no revision
history, one ``fetched_at_utc`` per week not per player), and (2)
``nfl_ats.players.canonicalize_injuries`` (``src/nfl_ats/players.py``), which
already calls ``nflreadpy.load_injuries(seasons=...)`` but immediately
subsets the result to ``INJURY_REQUIRED_COLUMNS`` -- 9 columns -- dropping
``report_primary_injury``, ``report_secondary_injury``,
``practice_primary_injury``, and ``practice_secondary_injury`` on ingest.
Those description columns are exactly what any illness/designation-reason
feature (this script exists to support ``docs/illness_battery.md``) needs,
and they are NOT recoverable from the existing snapshot -- they were never
written to disk in the first place.

**Loader choice (measured, this session)**: ``nflreadpy`` (an existing repo
dependency) exposes ``load_injuries(seasons=...)`` -- confirmed by reading
its source, ``nflreadpy/load_injuries.py``: for each requested season it
calls ``downloader.download("nflverse-data", f"injuries/injuries_{season}")``,
which resolves to exactly
``https://github.com/nflverse/nflverse-data/releases/download/injuries/injuries_{season}.parquet``
(``NflverseDownloader.BASE_URLS["nflverse-data"]``, read from
``nflreadpy/downloader.py``) -- the same release-asset pattern
``docs/new_lead_classes_20260826.md`` reported from raw URL probing. This
script therefore uses ``nflreadpy.load_injuries`` (the maintained library
path) rather than fetching those URLs directly, per repo instruction to
prefer the library when it exists. It calls the loader ONE SEASON AT A TIME
(rather than the single bulk ``seasons=True`` call) so the manifest can carry
one row -- source URL, per-season row count, per-season SHA-256 -- per
release asset, matching this repo's other per-source ingest manifests
(``scripts/fluview_battery_ingest.py``, ``scripts/ingest_nflcom_injuries.py``).

**Season coverage, measured this session** (``get_current_season()`` from
``nflreadpy.utils_date`` resolves to 2025 as of 2026-08-26, so
``seasons=True`` would already stop at 2025; confirmed directly): seasons
2009-2025 all return real data (2009: 4,821 rows; 2025: 6,068 rows).
``injuries_2026.parquet`` returns HTTP 404 (checked directly) -- the 2026
season has not been published yet, consistent with it being the offseason.

**Critical point-in-time finding, measured this session**: the ``date_modified``
column -- the per-row revision timestamp this whole battery's as-of
construction depends on -- is NOT uniformly populated:
  - 2011-2024 (14 seasons): 0 nulls.
  - 2010: 62 nulls out of 4,491 rows (~1.4%).
  - 2009: 4,804 nulls out of 4,821 rows (~99.6%) -- effectively unusable
    for point-in-time construction.
  - **2025: 6,068 nulls out of 6,068 rows -- ENTIRELY missing.** The 2025
    release has no ``date_modified`` column at all; it instead carries a
    ``season_type`` column (REG/POST) that 2009-2024 lack. This is a
    genuine upstream schema change, not a parsing bug on this side (checked
    directly against the raw per-season frame's own column list). Any
    as-of/checkpoint construction built on ``date_modified`` will correctly
    treat every 2025 row as unresolvable-as-of -- i.e. missing, never a
    leaked final value -- by the same "no checkpoint row qualifies -> missing"
    construction ``docs/fluview_battery.md`` section 3 already uses for its
    own pre-2017 gap. This is disclosed here, before any scoring, exactly as
    that precedent requires.

Because 2025 lacks ``date_modified`` and 2009 is 99.6% missing it, this
ingest still pulls the full 2009-2025 range (per repo convention: report the
gap, do not silently truncate the ingest around it), but the
point-in-time-recoverable window for any downstream battery is realistically
2010-2024.

Output: ``data/raw/nflverse_injuries/<UTC timestamp>/injuries.parquet`` (one
combined table, all 17 seasons, union of every season's columns -- some
older/newer columns are season-specific, see the finding above) plus
``manifest.json`` recording, per season: the exact source URL, fetch
timestamp, row count, column list, and a SHA-256 of that season's own
re-serialized parquet bytes (labelled honestly as a fingerprint of what this
ingest consumed and stored, not a byte-identical copy of the upstream HTTP
response -- ``nflreadpy`` parses the response through polars before this
script ever sees bytes, so a byte-identical hash of the original file is not
recoverable from the library path). The combined output file's own SHA-256
(via ``nfl_ats.provenance.sha256_file``, the repo's existing convention for
hashing a written artifact) is also recorded. Gitignored, per repo convention
(``data/raw`` is never committed).
"""

from __future__ import annotations

import hashlib
import io
import sys
import time
from pathlib import Path
from typing import Any

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from nfl_ats.io import atomic_json, atomic_parquet  # noqa: E402
from nfl_ats.provenance import sha256_file  # noqa: E402

RELEASE_URL_TEMPLATE = (
    "https://github.com/nflverse/nflverse-data/releases/download/injuries/injuries_{season}.parquet"
)
SEASON_START = 2009
# One past the last season nflreadpy's own get_current_season() resolved to,
# measured this session (2026-08-26) -- kept as a literal upper bound (not a
# live call to get_current_season() at import time) so a snapshot's season
# range is reproducible from this script's own text, matching every other
# ingest script's SEASON_START/SEASON_END convention in this repo.
SEASON_END = 2025


def _to_pandas(frame: Any) -> pd.DataFrame:
    """Polars -> pandas, tolerant of nflreadpy returning either (matches the
    ``_to_pandas`` helper already used by ``src/nfl_ats/participation.py`` /
    ``src/nfl_ats/pbp.py`` for the identical nflreadpy-return-type concern)."""

    if isinstance(frame, pd.DataFrame):
        return frame
    to_pandas = getattr(frame, "to_pandas", None)
    if to_pandas is None:
        raise TypeError(f"Unexpected nflreadpy return type: {type(frame)!r}")
    return to_pandas()


def fetch_season(season: int) -> dict[str, Any]:
    import nflreadpy as nfl

    t0 = time.time()
    url = RELEASE_URL_TEMPLATE.format(season=season)
    try:
        polars_frame = nfl.load_injuries(seasons=[season])
    except Exception as exc:  # record the failure, keep going
        return {
            "season": season,
            "url": url,
            "ok": False,
            "error": f"{type(exc).__name__}: {exc}",
            "elapsed_seconds": time.time() - t0,
            "frame": None,
        }
    frame = _to_pandas(polars_frame)
    elapsed = time.time() - t0
    buffer = io.BytesIO()
    frame.to_parquet(buffer, index=False)
    season_sha256 = hashlib.sha256(buffer.getvalue()).hexdigest()
    return {
        "season": season,
        "url": url,
        "ok": True,
        "loader": "nflreadpy.load_injuries",
        "n_rows": len(frame),
        "columns": frame.columns.tolist(),
        "n_null_date_modified": (
            int(frame["date_modified"].isna().sum()) if "date_modified" in frame.columns else None
        ),
        "sha256_of_reserialized_parquet": season_sha256,
        "elapsed_seconds": elapsed,
        "frame": frame,
    }


def run_ingest(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_entries: list[dict[str, Any]] = []
    frames: list[pd.DataFrame] = []

    seasons = list(range(SEASON_START, SEASON_END + 1))
    for i, season in enumerate(seasons, start=1):
        print(f"[{i}/{len(seasons)}] fetching {season} via nflreadpy.load_injuries ...")
        result = fetch_season(season)
        frame = result.pop("frame")
        manifest_entries.append(result)
        if not result["ok"]:
            print(f"  FAILED: {result['error']}")
            continue
        print(
            f"  rows={result['n_rows']} columns={len(result['columns'])} "
            f"null_date_modified={result['n_null_date_modified']} "
            f"elapsed={result['elapsed_seconds']:.2f}s"
        )
        assert frame is not None
        frames.append(frame)

    if not frames:
        raise SystemExit(
            "no seasons fetched successfully -- aborting, not writing an empty snapshot"
        )

    # Union of columns across seasons (2025 lacks date_modified but has
    # season_type; 2009-2024 is the reverse) -- outer-join the column sets so
    # every season's real columns survive; missing columns for a given
    # season's rows become NaN, not silently dropped.
    combined = pd.concat(frames, ignore_index=True, sort=False)
    combined["season"] = pd.to_numeric(combined["season"], errors="raise").astype(int)

    out_path = output_dir / "injuries.parquet"
    atomic_parquet(combined, out_path)

    ok_entries = [e for e in manifest_entries if e["ok"]]
    failed_entries = [e for e in manifest_entries if not e["ok"]]
    per_season_row_counts = {str(e["season"]): e["n_rows"] for e in ok_entries}
    per_season_null_date_modified = {
        str(e["season"]): e["n_null_date_modified"] for e in ok_entries
    }

    manifest = {
        "schema": "nflverse_injuries_snapshot/1",
        "source": "nflverse injuries release, per-season parquet, via nflreadpy.load_injuries",
        "release_url_template": RELEASE_URL_TEMPLATE,
        "loader": "nflreadpy.load_injuries(seasons=[season]), one season per call",
        "fetched_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "season_start_requested": SEASON_START,
        "season_end_requested": SEASON_END,
        "seasons_ok": sorted(e["season"] for e in ok_entries),
        "seasons_failed": sorted(e["season"] for e in failed_entries),
        "n_rows_total": len(combined),
        "n_rows_per_season": per_season_row_counts,
        "n_null_date_modified_per_season": per_season_null_date_modified,
        "columns_union": combined.columns.tolist(),
        "point_in_time_note": (
            "date_modified is 0-null for seasons 2011-2024, 62/4491 null for 2010, "
            "4804/4821 null (~99.6%) for 2009, and 6068/6068 (100%, entirely absent -- "
            "replaced by a season_type column instead) for 2025. Any as-of/checkpoint "
            "feature keyed on date_modified will correctly resolve every 2025 row (and "
            "nearly every 2009 row) to missing, not a leaked final value, by construction. "
            "The point-in-time-recoverable window is therefore 2010-2024, not the nominal "
            "2009-2025 ingest range."
        ),
        "requests": manifest_entries,
        "output_parquet": str(out_path.relative_to(REPO)),
        "output_parquet_sha256": sha256_file(out_path),
    }
    manifest_path = output_dir / "manifest.json"
    atomic_json(manifest, manifest_path)

    print(f"\nwrote {out_path} ({len(combined)} rows, {out_path.stat().st_size} bytes)")
    print(f"wrote {manifest_path}")
    if failed_entries:
        print(
            f"WARNING: {len(failed_entries)} season(s) failed to fetch: "
            f"{[e['season'] for e in failed_entries]}",
            file=sys.stderr,
        )


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    output_dir = args.output or (
        REPO / "data" / "raw" / "nflverse_injuries" / time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    )
    run_ingest(output_dir)


if __name__ == "__main__":
    main()
