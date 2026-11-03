import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import nflreadpy as nfl

ROOT = Path(__file__).resolve().parents[1]
SEASONS = list(range(1999, 2009))


def to_pandas(frame):
    return frame.to_pandas() if hasattr(frame, "to_pandas") else frame


def main():
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    out = ROOT / "data" / "raw" / "nflverse_pbp_full" / stamp
    out.mkdir(parents=True)
    files = []
    schedules = to_pandas(nfl.load_schedules(seasons=SEASONS))
    schedules.to_parquet(out / "schedules_1999_2008.parquet")
    team_stats = to_pandas(nfl.load_team_stats(seasons=SEASONS, summary_level="week"))
    team_stats.to_parquet(out / "team_stats_1999_2008.parquet")
    for name in ("schedules_1999_2008.parquet", "team_stats_1999_2008.parquet"):
        files.append(
            {"file": name, "sha256": hashlib.sha256((out / name).read_bytes()).hexdigest()}
        )
    for season in SEASONS:
        pbp = to_pandas(nfl.load_pbp(seasons=[season]))
        name = f"pbp_{season}.parquet"
        pbp.to_parquet(out / name)
        files.append(
            {
                "file": name,
                "rows": len(pbp),
                "sha256": hashlib.sha256((out / name).read_bytes()).hexdigest(),
            }
        )
        print(season, len(pbp), flush=True)
    manifest = {"captured_utc": stamp, "loader": "nflreadpy", "files": files}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print(out)


main()
