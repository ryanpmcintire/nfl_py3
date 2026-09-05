"""Preserve the full 2025 nflverse daily depth release in an immutable snapshot."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from nfl_ats.io import atomic_json, atomic_parquet
from nfl_ats.provenance import sha256_file

RAW_ROOT = Path("data/raw/depth_charts")


def run_ingest(output_dir: Path) -> dict[str, object]:
    import nflreadpy as nfl
    from nflreadpy.downloader import NflverseDownloader

    if output_dir.exists():
        raise FileExistsError(f"Snapshot already exists: {output_dir}")
    fetched_at = datetime.now(UTC).isoformat()
    raw = nfl.load_depth_charts(seasons=[2025])
    frame = raw if isinstance(raw, pd.DataFrame) else raw.to_pandas()
    dates = pd.to_datetime(frame["dt"], utc=True, errors="coerce")
    if frame.empty or not dates.notna().any():
        raise ValueError("The 2025 release has no usable daily observations")
    output_dir.mkdir(parents=True, exist_ok=False)
    path = output_dir / "depth_charts.parquet"
    atomic_parquet(frame, path)
    manifest: dict[str, object] = {
        "schema": "nflverse_depth_charts_snapshot/1",
        "loader": "nflreadpy.load_depth_charts(seasons=[2025])",
        "source_url": NflverseDownloader.BASE_URLS["nflverse-data"].rstrip("/")
        + "/depth_charts/depth_charts_2025.parquet",
        "fetched_at_utc": fetched_at,
        "completed_at_utc": datetime.now(UTC).isoformat(),
        "requested_seasons": [2025],
        "rows": len(frame),
        "columns": frame.columns.tolist(),
        "teams": int(frame["team"].nunique()),
        "usable_dt_rows": int(dates.notna().sum()),
        "dt_min": dates.min().isoformat(),
        "dt_max": dates.max().isoformat(),
        "output_parquet_sha256": sha256_file(path),
        "hash_note": "Fingerprint of reserialized library output, not original HTTP bytes",
    }
    atomic_json(manifest, output_dir / "manifest.json")
    return manifest


def main() -> None:
    destination = RAW_ROOT / datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    print(json.dumps({"snapshot": str(destination), **run_ingest(destination)}, indent=2))


if __name__ == "__main__":
    main()
