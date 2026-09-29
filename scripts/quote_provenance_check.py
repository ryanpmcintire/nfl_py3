from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

pa.set_cpu_count(2)
pa.set_io_thread_count(1)

REQUIRED = {
    "observed_at_utc",
    "provider",
    "provider_event_id",
    "sport_key",
    "commence_time_utc",
    "bookmaker_key",
    "bookmaker_last_update_utc",
    "market_last_update_utc",
    "market",
    "outcome_side",
    "line",
    "price",
    "raw_response_sha256",
}


def instant(value: Any) -> pd.Timestamp:
    stamp = pd.Timestamp(value)
    if pd.isna(stamp) or stamp.tzinfo is None:
        raise ValueError("Timestamp must include a timezone")
    return stamp.tz_convert("UTC")


def validate_frame(
    quotes: pd.DataFrame,
    manifest: dict[str, Any],
    *,
    cutoff: pd.Timestamp | None = None,
) -> pd.DataFrame:
    missing = REQUIRED.difference(quotes.columns)
    if missing:
        raise ValueError(f"Missing quote columns: {', '.join(sorted(missing))}")
    observed = instant(manifest.get("observed_at_utc"))
    historical = manifest.get("capture_kind") == "historical_backfill"
    request = manifest.get("request", {})
    provider = manifest.get("provider")
    if historical:
        if provider != "the-odds-api" or request.get("endpoint") != "historical":
            raise ValueError("Unsupported historical source")
        snapshot = instant(manifest.get("snapshot_timestamp_utc"))
        requested = instant(manifest.get("requested_at_utc"))
        if snapshot != observed or snapshot > requested:
            raise ValueError("Historical snapshot/request clocks disagree")
        source_proven = False
        limit = requested if cutoff is None else instant(cutoff)
    else:
        source_proven = (
            provider == "bovada_public_nfl"
            and manifest.get("capture_kind") == "live"
            and request.get("source") == provider
            and manifest.get("quote_timestamp_semantics")
            == "book_update_unknown_capture_time_observed"
        ) or (
            provider == "the-odds-api"
            and manifest.get("capture_kind") in (None, "live")
            and request.get("sport") == "americanfootball_nfl"
            and request.get("endpoint") is None
            and "requested_at_utc" not in manifest
            and "snapshot_timestamp_utc" not in manifest
        )
        limit = observed if cutoff is None else instant(cutoff)
    result = quotes.copy()
    result["clock_consistent"] = (
        pd.to_datetime(quotes["observed_at_utc"], utc=True, errors="coerce").eq(observed)
        & quotes["provider"].eq(provider)
        & quotes["raw_response_sha256"].eq(manifest["files"]["response.json"]["sha256"])
        & quotes["provider_event_id"].notna()
        & quotes["bookmaker_key"].notna()
        & quotes["sport_key"].eq(request.get("sport", "americanfootball_nfl"))
    )
    if historical:
        if "capture_kind" not in quotes:
            raise ValueError("Historical quote capture kind is missing")
        result["clock_consistent"] &= quotes["capture_kind"].eq("historical_backfill")
    for column in ("bookmaker_last_update_utc", "market_last_update_utc"):
        stamps = pd.to_datetime(quotes[column], utc=True, errors="coerce")
        result["clock_consistent"] &= quotes[column].isna() | (stamps.notna() & stamps.le(observed))
    result["before_kickoff"] = pd.to_datetime(
        quotes["commence_time_utc"], utc=True, errors="coerce"
    ).gt(observed)
    result["asof_consistent"] = (
        result["clock_consistent"] & result["before_kickoff"] & (observed <= limit)
    )
    result["availability_proven"] = result["asof_consistent"] & source_proven
    return result


def pair_counts(frame: pd.DataFrame) -> Counter:
    counts = Counter(quote_rows=len(frame))
    for column in ("clock_consistent", "before_kickoff", "asof_consistent", "availability_proven"):
        counts[f"{column}_rows"] = int(frame[column].sum())
    spreads = frame.loc[
        frame["market"].eq("spreads") & frame["outcome_side"].isin(["HOME", "AWAY"])
    ].copy()
    line = pd.to_numeric(spreads["line"], errors="coerce")
    price = pd.to_numeric(spreads["price"], errors="coerce")
    spreads["home_line"] = line.where(spreads["outcome_side"].eq("HOME"), -line)
    spreads["price_valid"] = np.isfinite(price) & price.abs().ge(100)
    spreads["book_clock_known"] = pd.to_datetime(
        spreads["bookmaker_last_update_utc"], utc=True, errors="coerce"
    ).notna()
    spreads = spreads.loc[np.isfinite(line)]
    pairs = spreads.groupby(
        ["sport_key", "provider_event_id", "bookmaker_key", "home_line"], dropna=False
    ).agg(
        rows=("outcome_side", "size"),
        sides=("outcome_side", "nunique"),
        price_valid=("price_valid", "all"),
        asof_consistent=("asof_consistent", "all"),
        availability_proven=("availability_proven", "all"),
        book_clock_known=("book_clock_known", "all"),
    )
    complete = pairs["rows"].eq(2) & pairs["sides"].eq(2) & pairs["price_valid"]
    counts["complete_spread_pairs"] = int(complete.sum())
    counts["asof_consistent_pairs"] = int((complete & pairs["asof_consistent"]).sum())
    counts["availability_proven_pairs"] = int((complete & pairs["availability_proven"]).sum())
    counts["asof_pairs_with_book_clock"] = int(
        (complete & pairs["asof_consistent"] & pairs["book_clock_known"]).sum()
    )
    return counts


def check_snapshot(path: Path, manifest: dict[str, Any], cutoff: pd.Timestamp | None) -> Counter:
    if manifest.get("schema_version") != 1 or manifest.get("snapshot_id") != path.name:
        raise ValueError("Snapshot identity mismatch")
    observed = instant(manifest.get("observed_at_utc"))
    if not path.name.startswith(observed.strftime("%Y%m%dT%H%M%SZ")):
        raise ValueError("Directory clock disagrees with manifest")
    raw = (path / "response.json").read_bytes()
    quote_bytes = (path / "quotes.parquet").read_bytes()
    for name, payload in (("response.json", raw), ("quotes.parquet", quote_bytes)):
        if hashlib.sha256(payload).hexdigest() != manifest["files"][name]["sha256"]:
            raise ValueError(f"Hash mismatch: {name}")
    if len(raw) != manifest["files"]["response.json"]["bytes"]:
        raise ValueError("Response byte count mismatch")
    decoded = json.loads(raw)
    if manifest.get("capture_kind") == "historical_backfill":
        if not isinstance(decoded, dict) or not isinstance(decoded.get("data"), list):
            raise ValueError("Historical response envelope missing")
        for raw_key, manifest_key in (
            ("timestamp", "snapshot_timestamp_utc"),
            ("previous_timestamp", "previous_snapshot_timestamp_utc"),
            ("next_timestamp", "next_snapshot_timestamp_utc"),
        ):
            raw_value, recorded = decoded.get(raw_key), manifest.get(manifest_key)
            if raw_value is None and recorded is None and raw_key != "timestamp":
                continue
            if instant(raw_value) != instant(recorded):
                raise ValueError(f"Response/manifest clock mismatch: {raw_key}")
        previous, following = decoded.get("previous_timestamp"), decoded.get("next_timestamp")
        requested = instant(manifest.get("requested_at_utc"))
        if previous is not None and instant(previous) >= observed:
            raise ValueError("Previous snapshot is not earlier")
        if following is not None and instant(following) <= requested:
            raise ValueError("Returned snapshot does not bracket requested time")
    elif not isinstance(decoded, list):
        raise ValueError("Unsupported live response envelope")
    source = pa.BufferReader(quote_bytes)
    names = set(pq.read_schema(source).names)
    source.seek(0)
    columns = sorted((REQUIRED | {"capture_kind"}).intersection(names))
    quotes = pq.read_table(source, columns=columns, use_threads=False).to_pandas(use_threads=False)
    if len(quotes) != manifest["files"]["quotes.parquet"]["rows"]:
        raise ValueError("Quote row count mismatch")
    return pair_counts(validate_frame(quotes, manifest, cutoff=cutoff))


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit quote clocks without reading game outcomes")
    parser.add_argument("--root", type=Path, default=Path("data/market/raw"))
    parser.add_argument("--seasons", nargs="+", type=int, default=[2023, 2024, 2025])
    parser.add_argument(
        "--snapshot", type=Path, help="Audit one snapshot instead of the season archive"
    )
    parser.add_argument("--include-snapshot", type=Path, action="append", default=[])
    parser.add_argument("--deadline", help="Inclusive UTC cutoff; default is each request/capture")
    args = parser.parse_args()
    cutoff = instant(args.deadline) if args.deadline else None
    if args.snapshot is None and not args.root.is_dir():
        parser.error("Market root does not exist")
    paths = (
        [args.snapshot]
        if args.snapshot
        else sorted(p.parent for p in args.root.glob("*/manifest.json"))
    )
    extras = {p.resolve() for p in args.include_snapshot}
    paths = list(dict.fromkeys([*paths, *args.include_snapshot]))
    groups: dict[str, Counter] = defaultdict(Counter)
    errors: Counter = Counter()
    examples = []
    for path in paths:
        try:
            manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
            request = manifest.get("request", {})
            season = request.get("season")
            forced = args.snapshot is not None or path.resolve() in extras
            if not forced and season not in args.seasons:
                continue
            label = (
                f"{season}:{request.get('sport', request.get('source', manifest.get('provider')))}"
            )
            group = groups[label]
            group["snapshots"] += 1
            group["historical_snapshots"] += int(
                manifest.get("capture_kind") == "historical_backfill"
            )
            group.update(check_snapshot(path, manifest, cutoff))
            group["integrity_pass_snapshots"] += 1
        except (OSError, ValueError, KeyError, TypeError, pa.ArrowException) as error:
            errors[f"{type(error).__name__}: {error}"] += 1
            if len(examples) < 5:
                examples.append({"snapshot": path.as_posix(), "reason": str(error)})
    print(
        json.dumps(
            {
                "rule": "docs/quote_provenance.md",
                "cutoff_basis": args.deadline
                or "per-snapshot requested_at_utc or live observed_at_utc",
                "pair_unit": "one snapshot/event/book/opposing spread line, two prices",
                "historical_revision_guarantee": "not established by local evidence",
                "outcome_looks": 0,
                "groups": dict(sorted(groups.items())),
                "errors": dict(errors),
                "error_examples": examples,
            },
            indent=2,
        )
    )
    raise SystemExit(1 if errors or not groups else 0)


if __name__ == "__main__":
    main()
