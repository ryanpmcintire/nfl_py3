from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

from nfl_ats.market_data import NFL_TEAM_NAMES, ODDS_API_PROVIDER, ODDS_API_SPORT
from nfl_ats.odds_backfill import plan_backfill

SEASONS = tuple(range(2020, 2026))
MARKETS = frozenset({"h2h", "spreads", "totals"})


def instant(value: object) -> pd.Timestamp:
    parsed = pd.Timestamp(value)
    if pd.isna(parsed) or parsed.tzinfo is None:
        raise ValueError("Missing or timezone-naive source timestamp")
    return parsed.tz_convert("UTC")


def number(value: object) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def paired_market(market: dict, home: str, away: str) -> bool:
    key = market.get("key")
    expected = {"Over", "Under"} if key == "totals" else {home, away}
    outcomes = [row for row in market.get("outcomes", []) if row.get("name") in expected]
    if len(outcomes) != 2 or {row.get("name") for row in outcomes} != expected:
        return False
    prices = [number(row.get("price")) for row in outcomes]
    if any(price is None or abs(price) < 100 for price in prices):
        return False
    if key == "h2h":
        return True
    points = [number(row.get("point")) for row in outcomes]
    if any(point is None for point in points):
        return False
    if key == "spreads":
        return math.isclose(points[0] + points[1], 0.0, abs_tol=1e-9)
    return points[0] > 0 and math.isclose(points[0], points[1], abs_tol=1e-9)


def load_schedule(path: Path) -> pd.DataFrame:
    source = pq.ParquetFile(path)
    columns = ["game_id", "season", "week", "gameday", "kickoff", "home_team", "away_team"]
    if "game_type" in source.schema_arrow.names:
        columns.append("game_type")
    frame = source.read(columns=columns, use_threads=False).to_pandas()
    frame = frame.loc[frame["season"].isin(SEASONS)].copy()
    if frame.empty or frame["game_id"].isna().any() or frame["game_id"].duplicated().any():
        raise ValueError("Inventory requires a nonempty schedule with unique game IDs")
    if set(frame["season"].unique()) != set(SEASONS):
        raise ValueError("Schedule must include all six declared seasons")
    frame["kickoff"] = pd.to_datetime(frame["kickoff"], utc=True, errors="raise")
    if frame["kickoff"].isna().any():
        raise ValueError("Schedule contains missing kickoff timestamps")
    return frame


def inventory(root: Path, schedule: pd.DataFrame) -> dict:
    targets = {
        (target.season, target.week): pd.Timestamp(target.requested_at_utc)
        for target in plan_backfill(schedule, 2020, 2025, labels=["tue_open"])
    }
    pairs = defaultdict(list)
    for row in schedule.itertuples(index=False):
        pairs[(row.home_team, row.away_team)].append(row)
    coverage = {season: defaultdict(set) for season in SEASONS}
    counts = Counter()
    rejected = Counter()
    errors = []
    manifests = sorted(root.glob("*/manifest.json"))
    if not manifests:
        errors.append("No manifests found at the configured local archive root")
    for path in manifests:
        try:
            manifest = json.loads(path.read_text(encoding="utf-8-sig"))
            request = manifest.get("request", {})
            if manifest.get("provider") != ODDS_API_PROVIDER:
                continue
            if request.get("sport") != ODDS_API_SPORT:
                continue
            if request.get("decision_label") != "tue_open":
                continue
            season = int(request["season"])
            if season not in SEASONS:
                continue
            week = int(request["week"])
            counts[(season, "manifests")] += 1
            key = (season, week)
            if key not in targets:
                rejected["snapshot outside scheduled season/week"] += 1
                continue
            cutoff = targets[key]
            requested = instant(manifest.get("requested_at_utc"))
            snapshot = instant(manifest.get("snapshot_timestamp_utc"))
            if requested != cutoff:
                rejected["request differs from canonical Tuesday 09:00 Eastern"] += 1
                continue
            if snapshot > cutoff or instant(manifest.get("observed_at_utc")) != snapshot:
                rejected["snapshot timestamp is late or inconsistent"] += 1
                continue
            if manifest.get("capture_kind") != "historical_backfill":
                rejected["capture is not a historical backfill"] += 1
                continue
            if request.get("odds_format") != "american":
                raise ValueError("Unsupported archived odds format")
            payload = (path.parent / "response.json").read_bytes()
            expected_hash = manifest["files"]["response.json"]["sha256"]
            if hashlib.sha256(payload).hexdigest() != expected_hash:
                raise ValueError("Raw response hash differs from manifest")
            decoded = json.loads(payload)
            if not isinstance(decoded, dict) or not isinstance(decoded.get("data"), list):
                raise ValueError("Historical response must contain an event list")
            if instant(decoded.get("timestamp")) != snapshot:
                raise ValueError("Raw response timestamp differs from manifest")
            counts[(season, "valid_snapshots")] += 1
            coverage[season]["weeks"].add(week)
            for event in decoded["data"]:
                if event.get("sport_key") != ODDS_API_SPORT:
                    continue
                event_time = instant(event.get("commence_time"))
                home, away = str(event.get("home_team")), str(event.get("away_team"))
                team_pair = (NFL_TEAM_NAMES.get(home), NFL_TEAM_NAMES.get(away))
                games = [
                    game
                    for game in pairs.get(team_pair, [])
                    if abs(game.kickoff - event_time) <= pd.Timedelta(hours=12)
                ]
                if len(games) != 1:
                    rejected["event lacks a unique schedule match"] += 1
                    continue
                game = games[0]
                if (int(game.season), int(game.week)) != key:
                    rejected["event belongs to another scheduled week"] += 1
                    continue
                if cutoff >= game.kickoff or cutoff >= event_time:
                    rejected["event is not pregame at the Tuesday cutoff"] += 1
                    continue
                game_id = str(game.game_id)
                seen = set()
                for book in event.get("bookmakers", []):
                    if not book.get("key"):
                        continue
                    book_time = instant(book.get("last_update"))
                    if book_time > snapshot:
                        rejected["bookmaker timestamp follows the source snapshot"] += 1
                        continue
                    book_markets = set()
                    for market in book.get("markets", []):
                        market_key = market.get("key")
                        if market_key not in MARKETS:
                            continue
                        counts[(season, f"raw_{market_key}")] += 1
                        market_time = instant(market.get("last_update") or book_time)
                        if market_time > snapshot:
                            rejected["market timestamp follows the source snapshot"] += 1
                            continue
                        if not paired_market(market, home, away):
                            rejected["market lacks a valid two-sided American quote"] += 1
                            continue
                        book_markets.add(market_key)
                        coverage[season][market_key].add(game_id)
                    seen.update(book_markets)
                    if book_markets >= MARKETS:
                        coverage[season]["same_book"].add(game_id)
                if seen >= MARKETS:
                    coverage[season]["all_three"].add(game_id)
        except (OSError, ValueError, KeyError, TypeError, AttributeError) as exc:
            errors.append(f"{path.as_posix()}: {type(exc).__name__}: {exc}")
    rows = []
    for season in SEASONS:
        games = int(schedule["season"].eq(season).sum())
        values = coverage[season]
        rows.append(
            {
                "season": season,
                "games": games,
                "manifests": counts[(season, "manifests")],
                "snapshots": counts[(season, "valid_snapshots")],
                "weeks": len(values["weeks"]),
                **{market: len(values[market]) for market in sorted(MARKETS)},
                "all_three": len(values["all_three"]),
                "same_book": len(values["same_book"]),
                "raw_h2h": counts[(season, "raw_h2h")],
            }
        )
    total_games = sum(row["games"] for row in rows)
    covered = sum(row["all_three"] for row in rows)
    return {
        "rows": rows,
        "total_games": total_games,
        "covered": covered,
        "coverage": covered / total_games,
        "manifests": len(manifests),
        "errors": errors,
        "rejected": dict(sorted(rejected.items())),
        "gate_passed": not errors and covered * 5 > total_games * 4,
    }


def render(result: dict, args: argparse.Namespace, schedule: pd.DataFrame) -> str:
    if result["errors"]:
        verdict = (
            "INCOMPLETE: resolve archive read or integrity errors before deciding the source gate."
        )
    elif result["gate_passed"]:
        verdict = (
            "SOURCE GATE MET: a separately predeclared unit 2 may proceed. No model was fitted."
        )
    else:
        verdict = (
            "SOURCE GATE NOT MET: unit 2 remains source-gated; this does not reject the mechanism."
        )
    population = (
        ", ".join(
            f"{kind}: {count}"
            for kind, count in schedule["game_type"].value_counts().sort_index().items()
        )
        if "game_type" in schedule
        else "all game IDs in the local feature schedule"
    )
    lines = [
        "# LEAD-71 unit 1: local Tuesday-open market inventory",
        "",
        f"**Measured:** {result['covered']}/{result['total_games']} games "
        f"({result['coverage']:.2%}) have valid paired h2h, spread and "
        f"total quotes in the same snapshot.",
        f"**Measured:** {verdict}",
        "**Read:** ROADMAP.md:854 requires strictly greater than 80% "
        "coverage in 2020\u20132025 before unit 2.",
        "",
        "## Reproduction and scope",
        f"Command: `.tools/uv.exe run --no-sync python scripts/lead71_unit1.py "
        f"--archive {args.archive.as_posix()} --features {args.features.as_posix()} "
        f"--report {args.report.as_posix()}`.",
        f"**Measured:** inspected {result['manifests']} local archive "
        f"manifests; population: {population}.",
        "Only schedule identifiers/timing and raw market quotes were read; "
        "no final scores, margins or picks.",
        "**Read:** `src/nfl_ats/odds_backfill.py:40` defines tue_open as "
        "Tuesday 09:00 America/New_York.",
        "The canonical season/week cutoff comes from that module's schedule planner.",
        "Historical response hashes and source timestamps must match their "
        "manifests; snapshots and quote",
        "updates must precede the cutoff and kickoff. Games must match "
        "teams and kickoff within 12 hours.",
        "Paired quotes require both home/away or over/under prices, "
        "opposing spread lines and equal totals.",
        "All-three coverage permits different books within one snapshot; "
        "same-book coverage is also shown.",
        "The denominator includes every locally scheduled 2020\u20132025 "
        "game, including games without quotes.",
        "Coverage is an exact local source census, not a sampled "
        "performance estimate; no confidence interval applies.",
        "",
        "## Coverage by season",
        "",
        "| Season | Games | Snapshots | Weeks | h2h | Spread | Total | All "
        "three | Same book | Coverage |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in result["rows"]:
        lines.append(
            f"| {row['season']} | {row['games']} | {row['snapshots']} | {row['weeks']} | "
            f"{row['h2h']} | {row['spreads']} | {row['totals']} | {row['all_three']} | "
            f"{row['same_book']} | {row['all_three'] / row['games']:.2%} |"
        )
    lines.extend(
        [
            "",
            "## Inventory diagnostics",
            "",
            f"**Measured:** {sum(row['manifests'] for row in result['rows'])} "
            f"target Tuesday-open manifests; "
            f"{sum(row['raw_h2h'] for row in result['rows'])} matching pregame "
            f"raw h2h market blocks; "
            f"{len(result['errors'])} archive read/integrity errors.",
        ]
    )
    lines.extend(f"- **Measured:** {name}: {count}." for name, count in result["rejected"].items())
    lines.extend(f"- **Measured:** {error}" for error in result["errors"][:8])
    lines.extend(
        [
            "",
            "## Research status",
            "",
            "Predictive looks: **0** executed; the roadmap reserves **2** for unit 2.",
            "IS/OOS performance, their gap, per-fold coefficients, "
            "probability_positive and decisive-game",
            "record: **not estimated**. There were no fits, LOSO scores, "
            "outcome reads or prediction rows.",
            "This inventory establishes source availability only, with no "
            "claim of predictive gain or its absence.",
            "Zero crossing never closes a signal. A later model must select "
            "sides through one fitted calibrated probability.",
            "No registry write was performed; a source census cannot supply an "
            "effect estimate for a signal record.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Inventory LEAD-71 Tuesday-open quotes without scoring outcomes."
    )
    parser.add_argument("--archive", type=Path, default=Path("data/market/raw"))
    parser.add_argument(
        "--features", type=Path, default=Path("data/processed/game_features.parquet")
    )
    parser.add_argument("--report", type=Path, default=Path("docs/lead71_unit1_inventory.md"))
    args = parser.parse_args()
    schedule = load_schedule(args.features)
    result = inventory(args.archive, schedule)
    args.report.write_text(render(result, args, schedule), encoding="utf-8")
    summary = {key: result[key] for key in ("total_games", "covered", "coverage", "gate_passed")}
    summary["errors"] = len(result["errors"])
    summary["report"] = args.report.as_posix()
    summary["predictive_looks"] = 0
    print(json.dumps(summary, sort_keys=True))
    return 1 if result["errors"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
