"""Odds ingest/summary and the market backfill commands."""

from __future__ import annotations

import argparse
import json
import os
from datetime import UTC, datetime

import pandas as pd

from nfl_ats.cli_common import _add_features_arg, _data_root, _load_features, _print_json
from nfl_ats.historical_market import fetch_historical_market_snapshot
from nfl_ats.market_data import (
    attach_nflverse_game_ids,
    fetch_odds_api_from_environment,
    load_quote_history,
    parse_odds_api_response,
    spread_consensus,
    write_market_snapshot,
)
from nfl_ats.odds_backfill import (
    DECISION_LABELS,
    DEFAULT_QUOTA_FLOOR,
    DEFAULT_SLEEP_SECONDS,
    execute_backfill,
    plan_backfill,
    summarize_backfill_plan,
)
from nfl_ats.open_close_market import fetch_open_close_snapshot
from nfl_ats.source_policy import require_private_raw_destination


def _cmd_odds_ingest(args: argparse.Namespace) -> None:
    market_root = _data_root() / "market" / "raw"
    require_private_raw_destination("the_odds_api", market_root)
    features = _load_features(args.features)
    payload, quota = fetch_odds_api_from_environment(
        regions=args.regions,
        markets=args.markets,
        bookmakers=args.bookmakers,
    )
    observed_at = datetime.now(UTC)
    quotes = parse_odds_api_response(payload, observed_at=observed_at)
    quotes = attach_nflverse_game_ids(quotes, features)
    request_metadata = {
        "sport": "americanfootball_nfl",
        "regions": args.regions,
        "markets": args.markets,
        "bookmakers": args.bookmakers,
        "odds_format": "american",
    }
    snapshot = write_market_snapshot(
        payload,
        quotes,
        market_root,
        observed_at=observed_at,
        request_metadata=request_metadata,
        quota=quota,
    )
    _print_json(
        {
            "snapshot_id": snapshot.snapshot_id,
            "directory": str(snapshot.root),
            "quotes": len(quotes),
            "events": int(quotes["provider_event_id"].nunique()),
            "matched_events": int(
                quotes.loc[quotes["nflverse_game_id"].notna(), "provider_event_id"].nunique()
            ),
            "quota": quota,
        }
    )


def _cmd_odds_backfill(args: argparse.Namespace) -> None:
    features = _load_features(args.features)
    weeks = [int(week) for week in args.weeks.split(",")] if args.weeks else None
    labels = args.labels.split(",") if args.labels else None
    targets = plan_backfill(
        features,
        args.start_season,
        args.end_season,
        regions=args.regions,
        weeks=weeks,
        labels=labels,
    )
    plan = summarize_backfill_plan(targets)
    if args.dry_run:
        _print_json({"dry_run": True, "regions": args.regions, "plan": plan})
        return
    market_root = _data_root() / "market" / "raw"
    require_private_raw_destination("the_odds_api", market_root)
    api_key = os.environ.get("THE_ODDS_API_KEY")
    if not api_key:
        raise ValueError("Set THE_ODDS_API_KEY before running the historical backfill")
    result = execute_backfill(
        targets,
        market_root,
        features,
        api_key=api_key,
        budget=args.budget,
        quota_floor=args.quota_floor,
        resume=args.resume,
        sleep_seconds=args.sleep_seconds,
    )
    _print_json({"regions": args.regions, "plan": plan, "result": result})


def _cmd_odds_summary(_: argparse.Namespace) -> None:
    history = load_quote_history(_data_root() / "market" / "raw")
    if history.empty:
        raise ValueError("No point-in-time market snapshots are available")
    consensus = spread_consensus(history)
    _print_json(
        {
            "quote_rows": len(history),
            "snapshots": int(history["observed_at_utc"].nunique()),
            "events": int(history["provider_event_id"].nunique()),
            "bookmakers": int(history["bookmaker_key"].nunique()),
            "matched_events": int(history["nflverse_game_id"].nunique()),
            "consensus_games": len(consensus),
            "first_observation": pd.to_datetime(history["observed_at_utc"], utc=True)
            .min()
            .isoformat(),
            "last_observation": pd.to_datetime(history["observed_at_utc"], utc=True)
            .max()
            .isoformat(),
        }
    )


def _cmd_market_backfill(args: argparse.Namespace) -> None:
    reference_games = _load_features(args.features)
    snapshot = fetch_historical_market_snapshot(
        _data_root() / "market" / "historical" / "raw",
        reference_games=reference_games,
    )
    manifest = json.loads(snapshot.manifest_path.read_text(encoding="utf-8"))
    _print_json(
        {
            "snapshot_id": snapshot.snapshot_id,
            "directory": str(snapshot.root),
            "source": manifest["source"],
            "rows": manifest["rows"],
            "semantics": manifest["semantics"],
            "audit": manifest["audit"],
        }
    )


def _cmd_market_open_close_backfill(args: argparse.Namespace) -> None:
    reference_games = _load_features(args.features)
    snapshot = fetch_open_close_snapshot(
        _data_root() / "market" / "historical" / "open_close" / "raw",
        reference_games=reference_games,
    )
    manifest = json.loads(snapshot.manifest_path.read_text(encoding="utf-8"))
    _print_json(
        {
            "snapshot_id": snapshot.snapshot_id,
            "directory": str(snapshot.root),
            "source": manifest["source"],
            "rows": manifest["rows"],
            "semantics": manifest["semantics"],
            "audit": manifest["audit"],
        }
    )


def register_odds(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
    current_year: int,
) -> None:
    """Register the point-in-time odds commands."""

    odds_ingest = subparsers.add_parser(
        "odds-ingest", help="archive timestamped NFL quotes from The Odds API"
    )
    _add_features_arg(odds_ingest)
    odds_ingest.add_argument("--regions", default="us")
    odds_ingest.add_argument("--markets", default="spreads,h2h")
    odds_ingest.add_argument("--bookmakers")
    odds_ingest.set_defaults(handler=_cmd_odds_ingest)

    odds_summary = subparsers.add_parser(
        "odds-summary", help="summarize locally archived point-in-time quotes"
    )
    odds_summary.set_defaults(handler=_cmd_odds_summary)


def register_backfill(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
    current_year: int,
) -> None:
    """Register the historical market backfill commands."""

    odds_backfill = subparsers.add_parser(
        "odds-backfill",
        help="backfill historical point-in-time NFL snapshots from The Odds API",
    )
    _add_features_arg(odds_backfill)
    odds_backfill.add_argument("--start-season", type=int, required=True)
    odds_backfill.add_argument("--end-season", type=int, required=True)
    odds_backfill.add_argument("--regions", default="us")
    odds_backfill.add_argument(
        "--weeks", help="comma-separated week filter, e.g. 1,2 (default: every scheduled week)"
    )
    odds_backfill.add_argument(
        "--labels",
        help=f"comma-separated decision labels from: {', '.join(DECISION_LABELS)}",
    )
    odds_backfill.add_argument(
        "--dry-run",
        action="store_true",
        help="print the exact call and credit plan without spending any credits",
    )
    odds_backfill.add_argument(
        "--budget",
        type=int,
        help="refuse to start when the planned cost exceeds this many credits",
    )
    odds_backfill.add_argument(
        "--quota-floor",
        type=int,
        default=DEFAULT_QUOTA_FLOOR,
        help="stop before any call that would leave fewer provider credits than this",
    )
    odds_backfill.add_argument(
        "--resume",
        action="store_true",
        help="skip planned snapshots already present in the store and continue",
    )
    odds_backfill.add_argument("--sleep-seconds", type=float, default=DEFAULT_SLEEP_SECONDS)
    odds_backfill.set_defaults(handler=_cmd_odds_backfill)

    market_backfill = subparsers.add_parser(
        "market-backfill",
        help="download and audit the free historical NFL closing-line archive",
    )
    _add_features_arg(market_backfill)
    market_backfill.set_defaults(handler=_cmd_market_backfill)

    open_close_backfill = subparsers.add_parser(
        "market-open-close-backfill",
        help="download the free 2025 NFL opener and multi-book closing sample",
    )
    _add_features_arg(open_close_backfill)
    open_close_backfill.set_defaults(handler=_cmd_market_open_close_backfill)
