from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from textwrap import fill
from urllib.parse import urlsplit

import pandas as pd
import pyarrow.parquet as pq

ZONE = "America/New_York"
VI_RAW = Path("data/raw/vegasinsider")
VI_TABLES = Path("artifacts/vegasinsider_backfill")
MARKET_RAW = Path("data/market/raw")
LEADERS = {"bovada", "williamhill_us", "mybookieag"}
KEYS = ["game_id", "book", "capture"]


def instant(value: object) -> pd.Timestamp:
    text = str(value)
    if len(text) == 14 and text.isdigit():
        return pd.to_datetime(text, format="%Y%m%d%H%M%S", utc=True, errors="coerce")
    return pd.to_datetime(value, utc=True, errors="coerce")


def load_games(path: Path) -> pd.DataFrame:
    columns = [
        "game_id",
        "season",
        "game_type",
        "week",
        "gameday",
        "gametime",
        "home_team",
        "away_team",
    ]
    games = pq.read_table(path, columns=columns, use_threads=False).to_pandas(use_threads=False)
    games = games.loc[games.season.between(2009, 2022) & games.game_type.eq("REG")].copy()
    games["gameday"] = pd.to_datetime(games.gameday)
    local = pd.to_datetime(
        games.gameday.dt.strftime("%Y-%m-%d") + " " + games.gametime.astype(str), errors="coerce"
    )
    games["kickoff"] = local.dt.tz_localize(ZONE, ambiguous="NaT", nonexistent="NaT").dt.tz_convert(
        "UTC"
    )
    anchor = games.groupby(["season", "week"]).gameday.transform("min")
    sunday = anchor + pd.to_timedelta((6 - anchor.dt.dayofweek) % 7, unit="D")
    for name, offset in {
        "monday": pd.Timedelta(days=-6),
        "freeze": pd.Timedelta(days=-5, hours=12),
        "wednesday": pd.Timedelta(days=-4),
        "deadline": pd.Timedelta(hours=12, minutes=45),
    }.items():
        games[name] = (sunday + offset).dt.tz_localize(ZONE).dt.tz_convert("UTC")
    games["deadline"] = games[["kickoff", "deadline"]].min(axis=1)
    return games.dropna(subset=["kickoff"]).reset_index(drop=True)


def probe_wayback(policy: dict) -> tuple[set[str], dict]:
    required = {"capture_timestamp_required", "no_direct_vegasinsider_fetch"}
    if not required.issubset(policy["conditions"]):
        raise ValueError("Wayback policy no longer matches the declared inventory")
    manifests = sorted(VI_RAW.glob("*/manifest*.json"))
    verified = set()
    rejected = 0
    entries = 0
    for path in manifests:
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
        for row in payload.get("snapshots", []):
            entries += 1
            stamp = str(row.get("capture_timestamp", ""))
            capture = instant(stamp)
            relative = row.get("file")
            raw = path.parent / relative if isinstance(relative, str) else path.parent
            inside = raw.resolve().is_relative_to(path.parent.resolve())
            if (
                pd.isna(capture)
                or not inside
                or not raw.is_file()
                or row.get("error")
                or urlsplit(str(row.get("original_url", ""))).hostname
                not in {"vegasinsider.com", "www.vegasinsider.com"}
                or f"web.archive.org/web/{stamp}" not in str(row.get("wayback_url", ""))
                or not row.get("sha256")
            ):
                rejected += 1
                continue
            if hashlib.sha256(raw.read_bytes()).hexdigest() != row["sha256"]:
                rejected += 1
                continue
            verified.add(stamp)
    dates = [instant(stamp) for stamp in verified]
    return verified, {
        "manifest_files": len(manifests),
        "manifest_entries": entries,
        "verified_captures": len(verified),
        "rejected_entries": rejected,
        "earliest_capture": min(dates).isoformat() if dates else "unavailable",
    }


def probe_vegas(policy: dict, verified: set[str], games: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    if policy["acquisition_allowed"] or "no_direct_fetch" not in policy["conditions"]:
        raise ValueError("VegasInsider policy no longer matches the declared inventory")
    paths = sorted(VI_TABLES.glob("*/season_*.parquet"))
    selected = [path for path in paths if 2009 <= int(path.stem.removeprefix("season_")) <= 2022]
    frames = []
    columns = ["capture_ts", "game_date", "away", "home", "book", "spread_line"]
    for path in selected:
        frame = pq.read_table(path, columns=columns, use_threads=False).to_pandas(use_threads=False)
        frame["capture_ts"] = frame.capture_ts.astype(str)
        frames.append(frame)
    if not frames:
        return pd.DataFrame(), {
            "table_files": 0,
            "table_rows": 0,
            "matched_rows": 0,
            "unverified_rows": 0,
        }
    rows = pd.concat(frames, ignore_index=True).drop_duplicates()
    rows["capture"] = pd.to_datetime(
        rows.capture_ts, format="%Y%m%d%H%M%S", utc=True, errors="coerce"
    )
    rows["gameday"] = pd.to_datetime(rows.game_date, errors="coerce")
    rows["timestamp_verified"] = rows.capture_ts.isin(verified)
    rows["line_present"] = rows.spread_line.notna() & rows.spread_line.astype(str).str.strip().ne(
        ""
    )
    rows = rows.rename(columns={"home": "home_team", "away": "away_team"})
    matched = rows.merge(
        games, on=["gameday", "home_team", "away_team"], how="inner", validate="many_to_one"
    )
    return matched.loc[matched.timestamp_verified & matched.line_present].copy(), {
        "table_files": len(selected),
        "table_rows": len(rows),
        "matched_rows": len(matched),
        "unverified_rows": int((~rows.timestamp_verified).sum()),
    }


def probe_odds(policy: dict, games: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    if "quota_headers_required" not in policy["conditions"]:
        raise ValueError("Odds API policy no longer matches the declared inventory")
    paths = sorted(MARKET_RAW.glob("*/manifest.json"))
    selected = []
    labels: dict[str, int] = {}
    rejected = 0
    frames = []
    columns = [
        "nflverse_game_id",
        "bookmaker_key",
        "market",
        "home_spread_line",
        "observed_at_utc",
        "bookmaker_last_update_utc",
    ]
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
        request = payload.get("request", {})
        stamp = instant(payload.get("snapshot_timestamp_utc"))
        if (
            payload.get("provider") != "the-odds-api"
            or pd.isna(stamp)
            or not 2009 <= stamp.year <= 2023
        ):
            continue
        if request.get("sport") != "americanfootball_nfl" or "spreads" not in str(
            request.get("markets", "")
        ).split(","):
            continue
        if payload.get("capture_kind") != "historical_backfill":
            continue
        if int(request.get("season", 0)) not in range(2009, 2023):
            continue
        selected.append(path)
        label = str(request.get("decision_label", "missing"))
        labels[label] = labels.get(label, 0) + 1
        quotes = path.parent / "quotes.parquet"
        metadata = payload.get("files", {}).get("quotes.parquet", {})
        if (
            not quotes.is_file()
            or not metadata.get("sha256")
            or hashlib.sha256(quotes.read_bytes()).hexdigest() != metadata["sha256"]
        ):
            rejected += 1
            continue
        frame = pq.read_table(quotes, columns=columns, use_threads=False).to_pandas(
            use_threads=False
        )
        frame["capture"] = pd.to_datetime(
            frame.observed_at_utc, utc=True, errors="coerce", format="mixed"
        )
        update = pd.to_datetime(
            frame.bookmaker_last_update_utc, utc=True, errors="coerce", format="mixed"
        )
        valid = frame.market.eq("spreads") & frame.capture.eq(stamp) & update.le(frame.capture)
        valid &= pd.to_numeric(frame.home_spread_line, errors="coerce").notna()
        frame = frame.loc[valid, ["nflverse_game_id", "bookmaker_key", "capture"]]
        frames.append(
            frame.rename(columns={"nflverse_game_id": "game_id", "bookmaker_key": "book"})
        )
    rows = (
        pd.concat(frames, ignore_index=True).drop_duplicates()
        if frames
        else pd.DataFrame(columns=KEYS)
    )
    matched = rows.merge(games, on="game_id", how="inner", validate="many_to_one")
    return matched, {
        "manifest_files": len(paths),
        "selected_files": len(selected),
        "rejected_files": rejected,
        "matched_rows": len(matched),
        "decision_labels": labels,
    }


def summarize(rows: pd.DataFrame) -> dict:
    empty = {
        "games": 0,
        "paired_games": 0,
        "freeze_pair_games": 0,
        "leader_pair_games": 0,
        "earliest": "unavailable",
        "timing": "unavailable",
        "seasons": [],
    }
    if rows.empty:
        return empty
    eligible = rows.loc[
        rows.capture.ge(rows.monday) & rows.capture.lt(rows.deadline)
    ].drop_duplicates(KEYS)
    after = eligible.loc[eligible.capture.gt(eligible.freeze)]
    if after.empty:
        return empty
    groups = eligible.groupby(["game_id", "book"])
    counts = groups.capture.nunique()
    latest = groups.capture.max()
    bounds = eligible.groupby(["game_id", "book"])[["freeze", "wednesday"]].first()
    pairs = counts.ge(2) & latest.ge(bounds.wednesday)
    before_freeze = groups.capture.min().le(bounds.freeze)
    crosses_freeze = pairs & before_freeze & latest.gt(bounds.freeze)
    pair_ids = set(pairs.loc[pairs].index.get_level_values("game_id"))
    freeze_ids = set(crosses_freeze.loc[crosses_freeze].index.get_level_values("game_id"))
    leader_ids = set(
        pairs.loc[
            pairs & pairs.index.get_level_values("book").isin(LEADERS)
        ].index.get_level_values("game_id")
    )
    first = after.sort_values(["capture", "game_id"]).iloc[0]
    since = (first.capture - first.freeze).total_seconds() / 3600
    until = (first.deadline - first.capture).total_seconds() / 3600
    seasons = []
    for season, frame in after.groupby("season"):
        ids = set(frame.game_id)
        seasons.append(
            (
                int(season),
                len(ids),
                len(ids & pair_ids),
                len(ids & freeze_ids),
                len(ids & leader_ids),
            )
        )
    return {
        "games": after.game_id.nunique(),
        "paired_games": len(pair_ids),
        "freeze_pair_games": len(freeze_ids),
        "leader_pair_games": len(leader_ids),
        "earliest": first.capture.isoformat(),
        ("timing"): (
            f"{since:.3f} hours after freeze; {until:.3f} hours before deadline (freeze "
            f"{first.freeze.isoformat()}, deadline {first.deadline.isoformat()})"
        ),
        "seasons": seasons,
    }


def render(archive: dict, vegas: dict, odds: dict, vi: dict, api: dict) -> str:
    paragraphs = [
        "# LEAD-73 unit 1: local pre-2023 source inventory",
        (
            "**Measured:** one offline probe per declared candidate; three inventory probes and "
            "zero statistical looks. No outcomes loaded, fits, scoring, network requests, or "
            "registry writes."
        ),
        (
            "Protocol was saved in `docs/lanes/lead73.md` before this run. Source policies were "
            "read before the probes. Regular-season NFL games, 2009-2022; timestamp bounds "
            "follow `scripts/sunday_market_probability_eval.py:22-68`: Monday history, "
            "Wednesday-or-later second capture, deadline at the earlier of kickoff and Sunday "
            "12:45 ET. Freeze is Tuesday noon ET. Equality with deadline is excluded. File "
            "modification time and retrieval time never substitute for capture time."
        ),
        "## Candidate inventory",
        (
            f"**Measured, Wayback:** {archive['manifest_files']} manifests, "
            f"{archive['manifest_entries']} entries, {archive['verified_captures']} distinct "
            f"captures with existing SHA-256-verified raw files; {archive['rejected_entries']} "
            f"entries rejected. Earliest raw capture: {archive['earliest_capture']}. The locally "
            f"identified archive is VegasInsider; its game-level timing is below. No claim about "
            f"archives absent from these local roots."
        ),
        (
            f"**Measured, VegasInsider:** {vegas['table_files']} pre-2023 season tables, "
            f"{vegas['table_rows']} distinct rows, {vegas['matched_rows']} exact "
            f"schedule-matched rows; {vegas['unverified_rows']} rows lack verified capture "
            f"provenance. There are {vi['games']} games with a verified spread cell after freeze "
            f"and before deadline; {vi['paired_games']} have two same-book captures with the "
            f"second on/after Wednesday. **Inferred:** timing evidence only; signed home-spread "
            f"orientation and book comparability still need verification before reuse in the "
            f"four-term fit."
        ),
        (
            f"**Measured, cached Odds API:** {odds['selected_files']} selected pre-2023 NFL "
            f"spread manifests from {odds['manifest_files']} local manifests; "
            f"{odds['rejected_files']} quote files rejected by integrity check; "
            f"{odds['matched_rows']} matched spread observations. There are {api['games']} games "
            f"with post-freeze, pre-deadline quotes; {api['paired_games']} same-book pair games; "
            f"{api['leader_pair_games']} pair games from incumbent leaders. **Read:** new paid "
            f"acquisition remains cancelled (`docs/lanes/done/odds-api-key-deactivated.md:3`)."
        ),
        (
            "Wayback and VegasInsider share raw evidence; their coverage must not be added. The "
            "VegasInsider probe inspects board tables, not an open/close label as timestamp "
            "proof. The Odds API probe excludes futures, period markets, unverified quote files, "
            "missing book-update instants, and book updates after observation."
        ),
        "## Earliest usable capture relative to both gates",
        f"**Measured, Wayback/VegasInsider:** {vi['earliest']}; {vi['timing']}.",
        f"**Measured, Odds API:** {api['earliest']}; {api['timing']}.",
        "## Coverage by season",
        (
            "These are source counts, not decisive-game or accuracy results. A pair means two "
            "distinct timestamps for one game/book before deadline, with the later timestamp "
            "on/after Wednesday. A freeze pair also brackets Tuesday noon. No requirement that "
            "the line changes is imposed. Incumbent leader keys are read from "
            "`src/nfl_ats/sharp_book_movement_features.py:24`; zero matching VegasInsider keys "
            "does not establish that its books have no information."
        ),
    ]
    sections = [
        fill(text, width=108, break_long_words=False, break_on_hyphens=False) for text in paragraphs
    ]
    table = [
        (
            "| Source | Season | Post-freeze games | Pair games | Freeze pair games | Leader "
            "pair games |"
        ),
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, result in [("VegasInsider", vi), ("Odds API", api)]:
        for row in result["seasons"]:
            table.append(f"| {name} | " + " | ".join(str(value) for value in row) + " |")
    sections.append("\n".join(table))
    paragraphs = [
        (
            f"**Measured:** cached Odds API decision-label counts: "
            f"`{json.dumps(odds['decision_labels'], sort_keys=True)}`."
        ),
        "## Decision and limits",
        (
            f"**Measured:** {api['leader_pair_games']} pre-2023 games have timestamp-admissible "
            f"cached pairs from incumbent leaders. **Inferred:** "
        )
        + (
            (
                "the local timestamp source gate clears for unit-2 preparation; "
                "model-logit/composition coverage, exact move reconstruction, and the frozen "
                "four-term LOSO protocol still need verification before fitting."
            )
            if api["leader_pair_games"]
            else (
                "the incumbent-leader move extension has no demonstrated local pairs; remain at "
                "inventory until a compatible source is established."
            )
        ),
        (
            "**Read:** the four terms are model logit, composition sum, market move, and move "
            "availability (`docs/lanes/done/market-move-decomposition.md:30-33`). This assigned "
            "unit implements source inventory only. No model has been refitted. No extension "
            "size or detection power is inferred from source counts."
        ),
        (
            "In-sample, out-of-sample, their gap, per-fold coefficients, calibration, "
            "decisive-game record, effect intervals, and probability_positive: **not "
            "estimated**; no outcome read or statistical look occurred. Intervals do not apply "
            "to this finite local-file census. This source result neither closes nor promotes a "
            "signal. AGENTS.md:65-85 requires admissible evidence for closure; AGENTS.md:87-105 "
            "requires one fitted probability and out-of-season parameter choice before scoring."
        ),
        (
            "Registry commands: none. Source inventory supplies no signal estimate to record; "
            "fabricating a neutral effect or probability_positive would misstate the evidence. "
            "The orchestrator owns any later record command and unit-2 fit."
        ),
        "Reproduce: `.tools/uv.exe run --no-sync --no-cache python scripts/lead73_unit1.py`.",
    ]
    sections.extend(
        fill(text, width=108, break_long_words=False, break_on_hyphens=False) for text in paragraphs
    )
    return "\n\n".join(sections) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Inventory local pre-2023 dated odds without loading outcomes or acquiring sources."
        )
    )
    parser.add_argument(
        "--schedule", type=Path, default=Path("data/raw/20260908T162105Z/schedules.parquet")
    )
    args = parser.parse_args()
    policies = json.loads(Path("config/source_policies.json").read_text(encoding="utf-8-sig"))[
        "sources"
    ]
    games = load_games(args.schedule)
    verified, archive = probe_wayback(policies["internet_archive_vegasinsider"])
    vi_rows, vegas = probe_vegas(policies["vegasinsider_content"], verified, games)
    api_rows, odds = probe_odds(policies["the_odds_api"], games)
    vi = summarize(vi_rows)
    api = summarize(api_rows)
    output = Path("docs/lead73_unit1.md")
    output.write_text(render(archive, vegas, odds, vi, api), encoding="utf-8")
    print(f"LEAD-73: 3 source probes; 0 outcome looks; {len(games)} schedule games")
    print(
        f"Wayback: {archive['verified_captures']} verified captures; "
        f"{archive['rejected_entries']} rejected entries"
    )
    print(f"VegasInsider: {vi['games']} post-freeze games; {vi['paired_games']} pair games")
    print(
        f"Odds API: {api['games']} post-freeze games; {api['paired_games']} pair games; "
        f"{api['leader_pair_games']} incumbent-leader pair games"
    )
    print(f"Saved {output.as_posix()}; no fit or registry write")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
