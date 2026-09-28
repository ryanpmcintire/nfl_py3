from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from nfl_ats.data import DataContractError
from nfl_ats.provenance import sha256_file


@dataclass(frozen=True)
class _Snapshot:
    quotes: pd.DataFrame
    provider: str
    quote_path: Path
    manifest_path: Path
    response_path: Path
    hashes: dict[Path, str]


@dataclass(frozen=True)
class _Match:
    snapshot: _Snapshot
    bookmaker: str
    home_line: float
    away_line: float
    home_price: float
    away_price: float


PROOF_COLUMNS = (
    "game_id",
    "exact_market_status",
    "exact_market_reason",
    "exact_market_observed_at_utc",
    "exact_market_bookmaker",
    "exact_market_provider",
    "exact_market_home_line",
    "exact_market_away_line",
    "exact_market_home_price",
    "exact_market_away_price",
    "exact_market_quote_artifact",
    "exact_market_quote_sha256",
    "exact_market_manifest_artifact",
    "exact_market_manifest_sha256",
    "exact_market_response_artifact",
    "exact_market_response_sha256",
    "exact_market_inspected_source_hashes",
)


def _path_text(path: Path) -> str:
    return str(path).replace("\\", "/")


def _empty_proof(game_id: str) -> dict[str, object]:
    return {
        "game_id": game_id,
        "exact_market_status": "unavailable",
        "exact_market_reason": "unresolved",
        "exact_market_observed_at_utc": pd.NaT,
        "exact_market_bookmaker": None,
        "exact_market_provider": None,
        "exact_market_home_line": np.nan,
        "exact_market_away_line": np.nan,
        "exact_market_home_price": np.nan,
        "exact_market_away_price": np.nan,
        "exact_market_quote_artifact": None,
        "exact_market_quote_sha256": None,
        "exact_market_manifest_artifact": None,
        "exact_market_manifest_sha256": None,
        "exact_market_response_artifact": None,
        "exact_market_response_sha256": None,
        "exact_market_inspected_source_hashes": "{}",
    }


def _validated_snapshot(
    manifest_path: Path, observed_at: pd.Timestamp
) -> tuple[_Snapshot | None, str | None]:
    quote_path = manifest_path.parent / "quotes.parquet"
    response_path = manifest_path.parent / "response.json"
    if not quote_path.is_file() or not response_path.is_file():
        return None, "snapshot_files_missing"
    before = {
        manifest_path: sha256_file(manifest_path),
        quote_path: sha256_file(quote_path),
        response_path: sha256_file(response_path),
    }
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return None, "snapshot_manifest_invalid"
    if not isinstance(manifest, dict):
        return None, "snapshot_manifest_invalid"
    files = manifest.get("files")
    quote_entry = files.get("quotes.parquet") if isinstance(files, dict) else None
    response_entry = files.get("response.json") if isinstance(files, dict) else None
    manifest_observed_value = manifest.get("observed_at_utc")
    manifest_observed = (
        pd.to_datetime(manifest_observed_value, utc=True, errors="coerce")
        if isinstance(manifest_observed_value, str)
        else pd.NaT
    )
    valid = (
        manifest.get("schema_version") == 1
        and manifest.get("snapshot_id") == manifest_path.parent.name
        and not pd.isna(manifest_observed)
        and manifest_observed == observed_at
        and isinstance(quote_entry, dict)
        and isinstance(response_entry, dict)
        and quote_entry.get("sha256") == before[quote_path]
        and response_entry.get("sha256") == before[response_path]
        and response_entry.get("bytes") == response_path.stat().st_size
    )
    if not valid:
        return None, "snapshot_manifest_invalid"
    required = {
        "observed_at_utc",
        "nflverse_game_id",
        "bookmaker_key",
        "market",
        "outcome_side",
        "home_spread_line",
        "line",
        "price",
    }
    try:
        quotes = pd.read_parquet(quote_path, columns=sorted(required))
    except (OSError, ValueError):
        return None, "snapshot_quotes_invalid"
    quote_rows = quote_entry.get("rows") if isinstance(quote_entry, dict) else None
    if not required.issubset(quotes.columns) or quote_rows != len(quotes):
        return None, "snapshot_quotes_invalid"
    if before != {path: sha256_file(path) for path in before}:
        raise DataContractError(f"Market quote source changed during read: {manifest_path.parent}")
    return (
        _Snapshot(
            quotes=quotes,
            provider=str(manifest.get("provider") or "").strip(),
            quote_path=quote_path,
            manifest_path=manifest_path,
            response_path=response_path,
            hashes=before,
        ),
        None,
    )


def resolve_exact_market_quote_proofs(
    market_raw_root: Path,
    frame: pd.DataFrame,
    *,
    game_id_column: str = "game_id",
    observed_at_column: str = "market_observed_at_utc",
    home_spread_column: str = "decision_home_spread",
    home_odds_column: str = "home_spread_odds",
    away_odds_column: str = "away_spread_odds",
    forecast_created_column: str = "forecast_created_at_utc",
    published_at_column: str = "published_at_utc",
) -> pd.DataFrame:
    required = {
        game_id_column,
        observed_at_column,
        home_spread_column,
        home_odds_column,
        away_odds_column,
        forecast_created_column,
        published_at_column,
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise DataContractError(
            f"Exact market quote proof is missing columns: {', '.join(missing)}"
        )
    if frame[game_id_column].astype(str).duplicated().any():
        raise DataContractError("Exact market quote proof requires unique game IDs")
    snapshot_cache: dict[int, tuple[list[_Snapshot], str | None]] = {}
    results: list[dict[str, object]] = []
    for _, source in frame.iterrows():
        game_id = str(source[game_id_column])
        output = _empty_proof(game_id)
        observed_at = pd.to_datetime(source[observed_at_column], utc=True, errors="coerce")
        forecast_created = pd.to_datetime(
            source[forecast_created_column], utc=True, errors="coerce"
        )
        published_at = pd.to_datetime(source[published_at_column], utc=True, errors="coerce")
        output["exact_market_observed_at_utc"] = observed_at
        if pd.isna(observed_at):
            output["exact_market_reason"] = "quote_time_missing"
            results.append(output)
            continue
        if pd.isna(forecast_created) or pd.isna(published_at):
            output["exact_market_reason"] = "quote_cutoff_missing"
            results.append(output)
            continue
        if observed_at > published_at:
            output["exact_market_reason"] = "quote_after_publication"
            results.append(output)
            continue
        if observed_at > forecast_created:
            output["exact_market_reason"] = "quote_after_forecast_creation_potential_leakage"
            results.append(output)
            continue
        home_spread = pd.to_numeric(source[home_spread_column], errors="coerce")
        home_odds = pd.to_numeric(source[home_odds_column], errors="coerce")
        away_odds = pd.to_numeric(source[away_odds_column], errors="coerce")
        if not np.isfinite([home_spread, home_odds, away_odds]).all():
            output["exact_market_reason"] = "quote_values_invalid"
            results.append(output)
            continue
        cache_key = int(observed_at.value)
        if cache_key not in snapshot_cache:
            prefix = observed_at.strftime("%Y%m%dT%H%M%SZ")
            manifest_paths = sorted(market_raw_root.glob(f"{prefix}*/manifest.json"))
            snapshots: list[_Snapshot] = []
            invalid_reasons: list[str] = []
            for manifest_path in manifest_paths:
                snapshot, invalid_reason = _validated_snapshot(manifest_path, observed_at)
                if snapshot is not None:
                    snapshots.append(snapshot)
                elif invalid_reason is not None:
                    invalid_reasons.append(invalid_reason)
            if snapshots:
                snapshot_cache[cache_key] = (snapshots, None)
            elif invalid_reasons:
                snapshot_cache[cache_key] = ([], sorted(set(invalid_reasons))[0])
            else:
                snapshot_cache[cache_key] = ([], "snapshot_not_found")
        snapshots, unavailable_reason = snapshot_cache[cache_key]
        if not snapshots:
            output["exact_market_reason"] = unavailable_reason or "snapshot_not_found"
            results.append(output)
            continue
        inspected: dict[str, str] = {}
        matches: list[_Match] = []
        for snapshot in snapshots:
            inspected.update({_path_text(path): digest for path, digest in snapshot.hashes.items()})
            observed = pd.to_datetime(snapshot.quotes["observed_at_utc"], utc=True, errors="coerce")
            candidates = snapshot.quotes.loc[
                snapshot.quotes["nflverse_game_id"].astype(str).eq(game_id)
                & snapshot.quotes["market"].eq("spreads")
                & observed.eq(observed_at)
                & np.isclose(
                    pd.to_numeric(snapshot.quotes["home_spread_line"], errors="coerce"),
                    float(home_spread),
                    rtol=0,
                    atol=1e-10,
                )
            ]
            for bookmaker, group in candidates.groupby("bookmaker_key", dropna=False):
                home = group.loc[group["outcome_side"].eq("HOME")]
                away = group.loc[group["outcome_side"].eq("AWAY")]
                if len(home) != 1 or len(away) != 1 or pd.isna(bookmaker):
                    continue
                home_line = pd.to_numeric(home.iloc[0]["line"], errors="coerce")
                away_line = pd.to_numeric(away.iloc[0]["line"], errors="coerce")
                home_price = pd.to_numeric(home.iloc[0]["price"], errors="coerce")
                away_price = pd.to_numeric(away.iloc[0]["price"], errors="coerce")
                exact = bool(np.isfinite([home_line, away_line, home_price, away_price]).all())
                exact = exact and bool(
                    np.isclose(home_line, -float(home_spread), rtol=0, atol=1e-10)
                )
                exact = exact and bool(
                    np.isclose(away_line, float(home_spread), rtol=0, atol=1e-10)
                )
                exact = exact and bool(np.isclose(home_price, float(home_odds), rtol=0, atol=1e-10))
                exact = exact and bool(np.isclose(away_price, float(away_odds), rtol=0, atol=1e-10))
                if exact:
                    matches.append(
                        _Match(
                            snapshot=snapshot,
                            bookmaker=str(bookmaker),
                            home_line=float(home_line),
                            away_line=float(away_line),
                            home_price=float(home_price),
                            away_price=float(away_price),
                        )
                    )
        output["exact_market_inspected_source_hashes"] = json.dumps(
            dict(sorted(inspected.items())), separators=(",", ":")
        )
        matched_snapshots = {_path_text(match.snapshot.quote_path) for match in matches}
        if not matches:
            output["exact_market_reason"] = "exact_same_book_quote_pair_not_found"
            results.append(output)
            continue
        if len(matched_snapshots) != 1:
            output["exact_market_reason"] = "ambiguous_exact_quote_snapshots"
            results.append(output)
            continue
        match = sorted(matches, key=lambda value: value.bookmaker)[0]
        snapshot = match.snapshot
        output.update(
            {
                "exact_market_status": "ready",
                "exact_market_reason": "ready_exact_same_book_quote_pair",
                "exact_market_bookmaker": match.bookmaker,
                "exact_market_provider": snapshot.provider,
                "exact_market_home_line": match.home_line,
                "exact_market_away_line": match.away_line,
                "exact_market_home_price": match.home_price,
                "exact_market_away_price": match.away_price,
                "exact_market_quote_artifact": _path_text(snapshot.quote_path),
                "exact_market_quote_sha256": snapshot.hashes[snapshot.quote_path],
                "exact_market_manifest_artifact": _path_text(snapshot.manifest_path),
                "exact_market_manifest_sha256": snapshot.hashes[snapshot.manifest_path],
                "exact_market_response_artifact": _path_text(snapshot.response_path),
                "exact_market_response_sha256": snapshot.hashes[snapshot.response_path],
            }
        )
        results.append(output)
    return pd.DataFrame(results, columns=list(PROOF_COLUMNS))


def verify_exact_market_quote_sources(proofs: pd.DataFrame) -> None:
    if "exact_market_inspected_source_hashes" not in proofs:
        raise DataContractError("Exact market quote proof has no inspected source hashes")
    expected: dict[Path, str] = {}
    for payload in proofs["exact_market_inspected_source_hashes"].dropna():
        try:
            hashes = json.loads(str(payload))
        except (TypeError, ValueError) as error:
            raise DataContractError("Exact market quote proof source hashes are invalid") from error
        if not isinstance(hashes, dict):
            raise DataContractError("Exact market quote proof source hashes are invalid")
        for path_text, digest in hashes.items():
            path = Path(str(path_text))
            previous = expected.get(path)
            if previous is not None and previous != str(digest):
                raise DataContractError(f"Conflicting exact market quote hashes: {path}")
            expected[path] = str(digest)
    for path, digest in expected.items():
        if not path.is_file() or sha256_file(path) != digest:
            raise DataContractError(f"Exact market quote source changed during measurement: {path}")


__all__ = [
    "PROOF_COLUMNS",
    "resolve_exact_market_quote_proofs",
    "verify_exact_market_quote_sources",
]
