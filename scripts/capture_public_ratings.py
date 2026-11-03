import argparse
import hashlib
import json
import sys
import time
import urllib.error
import urllib.request
from datetime import UTC, datetime
from pathlib import Path

SOURCE_ID = "github_nfelo_public_ratings"
REPO = "greerreNFL/nfelo"
BRANCH = "main"
FILES = (
    "output_data/nfelo_games.csv",
    "output_data/elo_snapshot.csv",
)
RAW_BASE = f"https://raw.githubusercontent.com/{REPO}/{BRANCH}/"
COMMITS_API = f"https://api.github.com/repos/{REPO}/commits"
USER_AGENT = "nfl-ats-research (ryanpmcintire@gmail.com)"
REQUEST_DELAY_SECONDS = 2.0
POLICY_PATH = Path(__file__).resolve().parents[1] / "config" / "source_policies.json"
OUT_ROOT = Path(__file__).resolve().parents[1] / "data" / "raw" / "public_ratings"
STOP_STATUSES = (403, 429)


def build_requests():
    urls = [RAW_BASE + path for path in FILES]
    urls.append(f"{COMMITS_API}?path={FILES[0]}&per_page=1")
    return urls


def policy_allows():
    if not POLICY_PATH.exists():
        return False
    sources = json.loads(POLICY_PATH.read_text(encoding="utf-8")).get("sources", {})
    entry = sources.get(SOURCE_ID)
    return bool(entry and entry.get("acquisition_allowed") is True)


def fetch(url):
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=60) as response:
        return response.status, response.read()


def capture():
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    out_dir = OUT_ROOT / stamp
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = {"source_id": SOURCE_ID, "observed_at_utc": stamp, "items": []}
    for index, url in enumerate(build_requests()):
        if index:
            time.sleep(REQUEST_DELAY_SECONDS)
        try:
            status, body = fetch(url)
        except urllib.error.HTTPError as error:
            manifest["items"].append({"url": url, "status": error.code})
            if error.code in STOP_STATUSES or error.code >= 500:
                break
            continue
        name = f"{index}_{url.rsplit('/', 1)[-1].split('?')[0]}"
        (out_dir / name).write_bytes(body)
        manifest["items"].append(
            {
                "url": url,
                "status": status,
                "file": name,
                "sha256": hashlib.sha256(body).hexdigest(),
                "bytes": len(body),
            }
        )
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return out_dir


def main():
    parser = argparse.ArgumentParser(description="Capture public rating files from a GitHub repo")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.dry_run:
        for url in build_requests():
            print(url)
        return 0
    if not policy_allows():
        print(f"policy entry {SOURCE_ID} not allowed", file=sys.stderr)
        return 2
    print(capture())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
