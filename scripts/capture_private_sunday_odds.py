from __future__ import annotations

import argparse
import json
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
from capture_bovada_private import REPO
from capture_bovada_private import capture as capture_bovada
from capture_odds_gap_private import capture as capture_odds_gap

ROOT = REPO / "data" / "market" / "raw"


def _latest_age(source: str, now: datetime) -> float | None:
    latest: pd.Timestamp | None = None
    for path in ROOT.glob("*/manifest.json"):
        manifest = json.loads(path.read_text(encoding="utf-8"))
        if manifest.get("provider") != source:
            continue
        instant = pd.to_datetime(
            manifest.get("retrieved_at_utc") or manifest.get("observed_at_utc"),
            utc=True,
            errors="coerce",
        )
        if pd.notna(instant) and (latest is None or instant > latest):
            latest = instant
    return None if latest is None else (pd.Timestamp(now) - latest).total_seconds() / 60


def capture(features_path: Path, max_age_minutes: int) -> dict[str, Any]:
    now = datetime.now(UTC)
    features = pd.read_parquet(features_path, columns=["kickoff"])
    kickoff = pd.to_datetime(features["kickoff"], utc=True, errors="coerce")
    if not kickoff.between(pd.Timestamp(now), pd.Timestamp(now) + pd.Timedelta(days=2)).any():
        return {"captured": False, "reason": "no_upcoming_nfl_games"}
    results: dict[str, Any] = {}
    for source, runner in (
        ("the_odds_gap_lineshop_private", capture_odds_gap),
        ("bovada_public_nfl", capture_bovada),
    ):
        blocked = REPO / "data" / "market" / f"{source}_blocked.json"
        if blocked.is_file():
            results[source] = {
                "captured": False,
                "reason": "source_http_block",
                "marker": str(blocked),
            }
            continue
        age = _latest_age(source, now)
        if age is not None and age < max_age_minutes:
            results[source] = {
                "captured": False,
                "reason": "recent_private_capture",
                "age_minutes": round(age, 1),
            }
            continue
        try:
            results[source] = runner(features_path)
        except (OSError, ValueError) as error:
            message = str(error)
            match = re.search(r"HTTP (\d{3})", message)
            status = int(match.group(1)) if match else 0
            if status in (403, 429) or status >= 500:
                blocked.parent.mkdir(parents=True, exist_ok=True)
                blocked.write_text(
                    json.dumps(
                        {
                            "source": source,
                            "blocked_at_utc": datetime.now(UTC).isoformat(),
                            "reason": message,
                        }
                    ),
                    encoding="utf-8",
                )
            results[source] = {"captured": False, "reason": message}
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Capture bounded private Sunday NFL odds")
    parser.add_argument(
        "--features", type=Path, default=REPO / "data" / "processed" / "game_features.parquet"
    )
    parser.add_argument("--max-age-minutes", type=int, default=30)
    args = parser.parse_args()
    if args.max_age_minutes < 1:
        parser.error("max-age-minutes must be positive")
    print(json.dumps(capture(args.features, args.max_age_minutes), sort_keys=True))


if __name__ == "__main__":
    main()
