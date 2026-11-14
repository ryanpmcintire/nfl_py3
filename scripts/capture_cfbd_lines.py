import argparse
import hashlib
import json
import sys
import time
import urllib.error
import urllib.parse
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nfl_ats import cfb  # noqa: E402

OUT = ROOT / "data" / "raw" / "cfbd_lines"
PATH = "/lines"
MIN_GAP_SECONDS = 2.0


def cached(season, week, season_type):
    for manifest in sorted(OUT.glob("*/manifest.json")):
        doc = json.loads(manifest.read_text())
        p = doc["request_parameters"]
        if p == {"year": season, "week": week, "seasonType": season_type}:
            return manifest.parent
    return None


def main():
    ap = argparse.ArgumentParser(
        description="Fetch CFBD /lines weeks with recorded observation time."
    )
    ap.add_argument("--season", type=int, required=True)
    ap.add_argument("--weeks", type=int, nargs="+", required=True)
    ap.add_argument("--season-type", default="regular")
    ap.add_argument(
        "--refresh",
        action="store_true",
        help="fetch again even when an earlier capture of the week exists",
    )
    args = ap.parse_args()
    key = cfb.cfbd_api_key()
    first = True
    for week in args.weeks:
        hit = None if args.refresh else cached(args.season, week, args.season_type)
        if hit is not None:
            print("cached", week, hit)
            continue
        if not first:
            time.sleep(MIN_GAP_SECONDS)
        first = False
        params = {"year": args.season, "week": week, "seasonType": args.season_type}
        url = f"{cfb.CFBD_API_ROOT}{PATH}?{urllib.parse.urlencode(params)}"
        try:
            payload, _quota = cfb._cfbd_http(url, key)
        except urllib.error.HTTPError as error:
            print("stop http", error.code, "week", week)
            return 1
        observed = datetime.now(UTC)
        records = cfb._cfbd_records(payload, "CFBD lines")
        stamp = observed.strftime("%Y%m%dT%H%M%SZ")
        folder = OUT / stamp
        folder.mkdir(parents=True, exist_ok=False)
        (folder / f"lines_{args.season}_week{week:02d}.json").write_bytes(payload)
        manifest = {
            "url": url,
            "observed_at_utc": observed.isoformat(),
            "sha256": hashlib.sha256(payload).hexdigest(),
            "request_parameters": params,
            "records": len(records),
            "file": f"lines_{args.season}_week{week:02d}.json",
        }
        (folder / "manifest.json").write_text(json.dumps(manifest, indent=1))
        print("saved", week, len(records), folder)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
