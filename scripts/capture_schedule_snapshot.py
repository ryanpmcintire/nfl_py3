import hashlib
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from nfl_ats.data import _to_pandas, validate_schedules
from nfl_ats.nflverse_current_season import nflreadpy_current_season
from nfl_ats.snapshots import utc_snapshot_id

ROOT = Path(__file__).resolve().parents[1]
SCHEDULE_ROOT = ROOT / "data" / "raw_schedules"
HASH_BLOCK_BYTES = 1024 * 1024


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(HASH_BLOCK_BYTES), b""):
            digest.update(block)
    return digest.hexdigest()


def capture(root=SCHEDULE_ROOT):
    import nflreadpy as nfl

    season = nflreadpy_current_season()
    fetched = datetime.now(UTC)
    schedules = _to_pandas(nfl.load_schedules(seasons=[season]))
    validate_schedules(schedules)
    if schedules.empty:
        raise ValueError(f"nflverse schedules returned no rows for {season}")
    directory = root / utc_snapshot_id(fetched)
    directory.mkdir(parents=True)
    path = directory / "schedules.parquet"
    temporary = path.with_suffix(".parquet.tmp")
    pd.DataFrame(schedules).to_parquet(temporary, index=False)
    temporary.replace(path)
    manifest = {
        "snapshot_id": directory.name,
        "fetched_at_utc": fetched.isoformat(),
        "schedule_seasons": [season],
        "sources": {
            "loader": "nflreadpy",
            "repository": "https://github.com/nflverse/nflverse-data",
            "schedules": "nflreadpy.load_schedules",
        },
        "files": {"schedules.parquet": {"rows": len(schedules), "sha256": sha256(path)}},
    }
    (directory / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return directory


def main():
    directory = capture()
    print(directory)
    return 0


if __name__ == "__main__":
    sys.exit(main())
