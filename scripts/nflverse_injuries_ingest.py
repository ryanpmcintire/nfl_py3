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
from nfl_ats.nflverse_current_season import load_season_frame  # noqa: E402
from nfl_ats.provenance import sha256_file  # noqa: E402

RELEASE_URL_TEMPLATE = (
    "https://github.com/nflverse/nflverse-data/releases/download/injuries/injuries_{season}.parquet"
)
SEASON_START = 2009
SEASON_END = 2026


def _to_pandas(frame: Any) -> pd.DataFrame:

    if isinstance(frame, pd.DataFrame):
        return frame
    to_pandas = getattr(frame, "to_pandas", None)
    if to_pandas is None:
        raise TypeError(f"Unexpected nflreadpy return type: {type(frame)!r}")
    return to_pandas()


def fetch_season(season: int) -> dict[str, Any]:

    t0 = time.time()
    url = RELEASE_URL_TEMPLATE.format(season=season)
    try:
        polars_frame = load_season_frame("injuries", season)
    except Exception as exc:
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
