"""Backfill point-in-time publication evidence for Sports Media Watch ratings.

The seasonal archive is a living revision and is never treated as historical
publication evidence.  This command queries Sports Media Watch's own WordPress
API, caches every response, and admits a structured row only when a dated post
published after the game contains both the exact audience and the matchup.  It
admits an image only when the media API resolves its exact source filename.

The source snapshot is immutable.  Enriched parquet files and their evidence
indexes are written to a separate output directory.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import re
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable, Iterable
from datetime import UTC, datetime
from html.parser import HTMLParser
from pathlib import Path
from typing import Any

import pandas as pd

if __package__:
    from scripts.ingest_sports_media_watch import TEAM_CODE_ALIASES, TEAM_NAMES, USER_AGENT
else:
    from ingest_sports_media_watch import TEAM_CODE_ALIASES, TEAM_NAMES, USER_AGENT

API_ROOT = "https://www.sportsmediawatch.com/wp-json/wp/v2"
BACKFILL_SCHEMA = "sports_media_watch_publication_backfill/2"
MATCHING_RULE_VERSION = "exact_identity_after_conservative_completion_v3"
GAME_COMPLETION_BUFFER = pd.Timedelta(hours=6)
REQUEST_DELAY_SECONDS = 1.5
POST_PAGE_SIZE = 100
ADDED_COLUMNS = (
    "source_modified_at",
    "timestamp_source_url",
    "timestamp_source_id",
    "timestamp_match_method",
)


class PublicationBackfillError(ValueError):
    """Raised when publication evidence is missing or internally inconsistent."""


class _TextParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []

    def handle_data(self, data: str) -> None:
        self.parts.append(data)


def _plain_text(value: str) -> str:
    parser = _TextParser()
    parser.feed(html.unescape(value))
    return re.sub(r"\s+", " ", " ".join(parser.parts)).strip()


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _utc(value: str) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if timestamp.tz is None:
        timestamp = timestamp.tz_localize("UTC")
    return timestamp.tz_convert("UTC")


def _iso_utc(value: str) -> str:
    return _utc(value).isoformat().replace("+00:00", "Z")


def _fetch(url: str, *, timeout: int = 45, retries: int = 3) -> bytes:
    last_error: Exception | None = None
    for attempt in range(retries):
        request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return response.read()
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError) as error:
            last_error = error
            time.sleep(3 * (attempt + 1))
    assert last_error is not None
    raise last_error


def _load_json(payload: bytes, *, url: str) -> list[dict[str, Any]]:
    try:
        value = json.loads(payload)
    except json.JSONDecodeError as error:
        raise PublicationBackfillError(f"invalid JSON from {url}: {error}") from error
    if not isinstance(value, list) or any(not isinstance(item, dict) for item in value):
        raise PublicationBackfillError(f"unexpected WordPress response shape from {url}")
    return value


def _cached_query(
    url: str,
    path: Path,
    *,
    fetcher: Callable[[str], bytes],
    sleeper: Callable[[float], None],
) -> tuple[list[dict[str, Any]], str]:
    if path.exists():
        payload = path.read_bytes()
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        sleeper(REQUEST_DELAY_SECONDS)
        payload = fetcher(url)
        _atomic_write(path, payload, replace=False)
    return _load_json(payload, url=url), _sha256(payload)


def _atomic_write(path: Path, payload: bytes, *, replace: bool) -> None:
    """Atomically write bytes, refusing to replace immutable snapshot members."""

    if path.exists() and not replace:
        raise PublicationBackfillError(f"snapshot member already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=path.parent, prefix=f".{path.name}.", delete=False
    ) as file:
        temporary = Path(file.name)
        file.write(payload)
    try:
        if path.exists() and not replace:
            raise PublicationBackfillError(f"snapshot member already exists: {path}")
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _json_payload(value: dict[str, Any]) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()


def _atomic_json(path: Path, value: dict[str, Any], *, replace: bool) -> None:
    _atomic_write(path, _json_payload(value), replace=replace)


def fetch_posts(
    output: Path,
    seasons: Iterable[int],
    *,
    fetcher: Callable[[str], bytes] = _fetch,
    sleeper: Callable[[float], None] = time.sleep,
) -> pd.DataFrame:
    """Cache bounded official post searches for each NFL season."""

    rows: list[dict[str, Any]] = []
    for season in sorted({int(value) for value in seasons}):
        page = 1
        while True:
            params = urllib.parse.urlencode(
                {
                    "after": f"{season}-08-01T00:00:00",
                    "before": f"{season + 1}-03-01T00:00:00",
                    "search": "NFL",
                    "per_page": POST_PAGE_SIZE,
                    "page": page,
                    "_fields": "id,date_gmt,modified_gmt,link,title,content",
                }
            )
            url = f"{API_ROOT}/posts?{params}"
            cache_path = output / "raw" / "posts" / f"{season}-page-{page:02d}.json"
            items, digest = _cached_query(url, cache_path, fetcher=fetcher, sleeper=sleeper)
            for item in items:
                title = item.get("title", {})
                content = item.get("content", {})
                rendered_title = title.get("rendered", "") if isinstance(title, dict) else ""
                rendered_content = content.get("rendered", "") if isinstance(content, dict) else ""
                rows.append(
                    {
                        "season_query": season,
                        "post_id": int(item["id"]),
                        "source_published_at": _iso_utc(str(item["date_gmt"])),
                        "source_modified_at": _iso_utc(str(item["modified_gmt"])),
                        "source_url": str(item["link"]),
                        "title": _plain_text(str(rendered_title)),
                        "content_text": _plain_text(str(rendered_content)),
                        "response_path": str(cache_path.relative_to(output)).replace("\\", "/"),
                        "response_sha256": digest,
                    }
                )
            if len(items) < POST_PAGE_SIZE:
                break
            page += 1
    if not rows:
        raise PublicationBackfillError("official post search returned no rows")
    result = pd.DataFrame(rows).drop_duplicates("post_id")
    return result.sort_values(["source_published_at", "post_id"]).reset_index(drop=True)


def _asset_key(url: str) -> str:
    name = urllib.parse.unquote(Path(urllib.parse.urlparse(url).path).name).lower()
    stem = Path(name).stem
    return re.sub(r"-\d+x\d+$", "", stem)


def _asset_identity(url: str) -> str:
    """Return the exact upload path while ignoring WordPress resize suffixes."""

    path = Path(urllib.parse.unquote(urllib.parse.urlparse(url).path).lower())
    stem = re.sub(r"-\d+x\d+$", "", path.stem)
    return (path.parent / f"{stem}{path.suffix}").as_posix()


def fetch_media(
    output: Path,
    assets: pd.DataFrame,
    *,
    fetcher: Callable[[str], bytes] = _fetch,
    sleeper: Callable[[float], None] = time.sleep,
) -> pd.DataFrame:
    """Resolve every ratings image against the primary WordPress media index."""

    rows: list[dict[str, Any]] = []
    for key in sorted({_asset_key(url) for url in assets["asset_url"].dropna()}):
        params = urllib.parse.urlencode(
            {
                "search": key,
                "per_page": 100,
                "_fields": "id,date_gmt,modified_gmt,source_url,link,slug",
            }
        )
        url = f"{API_ROOT}/media?{params}"
        cache_path = output / "raw" / "media" / f"{key}.json"
        items, digest = _cached_query(url, cache_path, fetcher=fetcher, sleeper=sleeper)
        exact = [item for item in items if _asset_key(str(item.get("source_url", ""))) == key]
        for item in exact:
            rows.append(
                {
                    "asset_key": key,
                    "asset_identity": _asset_identity(str(item["source_url"])),
                    "media_id": int(item["id"]),
                    "source_published_at": _iso_utc(str(item["date_gmt"])),
                    "source_modified_at": _iso_utc(str(item["modified_gmt"])),
                    "source_url": str(item["source_url"]),
                    "attachment_url": str(item["link"]),
                    "response_path": str(cache_path.relative_to(output)).replace("\\", "/"),
                    "response_sha256": digest,
                }
            )
    if not rows:
        return pd.DataFrame(
            columns=[
                "asset_key",
                "asset_identity",
                "media_id",
                "source_published_at",
                "source_modified_at",
                "source_url",
                "attachment_url",
                "response_path",
                "response_sha256",
            ]
        )
    return (
        pd.DataFrame(rows).sort_values(["asset_key", "source_published_at"]).reset_index(drop=True)
    )


def _audience_values(text: str) -> set[int]:
    values: set[int] = set()
    for number in re.findall(r"(?<![\w.])(\d+(?:\.\d+)?)\s*(?:million|m\b)", text, re.I):
        values.add(round(float(number) * 1_000_000))
    for number in re.findall(r"(?<![\d,])(\d{1,3}(?:,\d{3})+)(?![\d,])", text):
        values.add(int(number.replace(",", "")))
    return values


_CODE_TO_NAME = {code: name.lower() for name, code in TEAM_NAMES.items()}


def _team_tokens(away_team: Any, home_team: Any) -> tuple[str, str] | None:
    if pd.isna(away_team) or pd.isna(home_team):
        return None
    away = TEAM_CODE_ALIASES.get(str(away_team), str(away_team))
    home_team_code = TEAM_CODE_ALIASES.get(str(home_team), str(home_team))
    away_name = _CODE_TO_NAME.get(away)
    home_name = _CODE_TO_NAME.get(home_team_code)
    return (away_name, home_name) if away_name and home_name else None


def _schedule_dates(
    schedules: pd.DataFrame,
) -> tuple[
    dict[tuple[int, int, str, str], pd.Timestamp],
    dict[tuple[int, int], tuple[pd.Timestamp, pd.Timestamp]],
]:
    required = {"season", "week", "gameday", "away_team", "home_team", "game_type"}
    missing = required.difference(schedules.columns)
    if missing:
        raise PublicationBackfillError(f"schedule missing columns: {sorted(missing)}")
    columns = ["season", "week", "gameday", "away_team", "home_team", "game_type"]
    if "gametime" in schedules.columns:
        columns.append("gametime")
    games = schedules.loc[schedules["game_type"].eq("REG"), columns].copy()
    games["gameday"] = pd.to_datetime(games["gameday"], utc=True, errors="raise")
    exact: dict[tuple[int, int, str, str], pd.Timestamp] = {}
    for row in games.itertuples(index=False):
        away = TEAM_CODE_ALIASES.get(str(row.away_team), str(row.away_team))
        home_code = TEAM_CODE_ALIASES.get(str(row.home_team), str(row.home_team))
        gametime = getattr(row, "gametime", None)
        if gametime is not None and pd.notna(gametime):
            kickoff = pd.Timestamp(
                f"{pd.Timestamp(row.gameday).date()} {gametime}", tz="America/New_York"
            ).tz_convert("UTC")
            available_after = kickoff + GAME_COMPLETION_BUFFER
        else:
            available_after = pd.Timestamp(row.gameday).normalize() + pd.Timedelta(days=1)
        exact[(int(row.season), int(row.week), away, home_code)] = available_after
    bounds: dict[tuple[int, int], tuple[pd.Timestamp, pd.Timestamp]] = {}
    for (season, week), group in games.groupby(["season", "week"]):
        bounds[(int(season), int(week))] = (
            max(
                exact[
                    (
                        int(season),
                        int(week),
                        TEAM_CODE_ALIASES.get(str(row.away_team), str(row.away_team)),
                        TEAM_CODE_ALIASES.get(str(row.home_team), str(row.home_team)),
                    )
                ]
                for row in group.itertuples(index=False)
            ),
            pd.Timestamp(group["gameday"].max()).normalize() + pd.Timedelta(days=6),
        )
    return exact, bounds


def enrich_rows(rows: pd.DataFrame, posts: pd.DataFrame, schedules: pd.DataFrame) -> pd.DataFrame:
    """Attach only matchup-and-audience-verified dated post evidence."""

    result = rows.copy()
    for column in ADDED_COLUMNS:
        result[column] = None
    result["source_published_at"] = None
    result["point_in_time_usable"] = False
    exact_dates, week_bounds = _schedule_dates(schedules)
    post_records: list[dict[str, Any]] = []
    for post in posts.to_dict("records"):
        text = f"{post['title']} {post['content_text']}".lower()
        post_records.append(
            {
                **post,
                "published": _utc(str(post["source_published_at"])),
                "text_lower": text,
                "audiences": _audience_values(text),
            }
        )

    for index, row in result.iterrows():
        if pd.isna(row.get("week")) or pd.isna(row.get("viewers")):
            continue
        season, week = int(row["season"]), int(row["week"])
        bounds = week_bounds.get((season, week))
        if bounds is None:
            continue
        teams = _team_tokens(row.get("away_team"), row.get("home_team"))
        event_at: pd.Timestamp | None = None
        if teams is not None:
            away = TEAM_CODE_ALIASES.get(str(row["away_team"]), str(row["away_team"]))
            home_code = TEAM_CODE_ALIASES.get(str(row["home_team"]), str(row["home_team"]))
            event_at = exact_dates.get((season, week, away, home_code))
        if event_at is None and pd.notna(row.get("event_date")):
            event_at = pd.Timestamp(row["event_date"], tz="UTC").normalize() + pd.Timedelta(days=1)
        earliest = event_at if event_at is not None else bounds[0]
        latest = bounds[1]
        viewers = int(row["viewers"])
        candidates: list[dict[str, Any]] = []
        for post in post_records:
            if not earliest <= post["published"] <= latest or viewers not in post["audiences"]:
                continue
            if teams is not None and not all(token in post["text_lower"] for token in teams):
                continue
            if teams is None:
                network = str(row.get("network") or "").lower()
                if not network or network not in post["text_lower"]:
                    continue
            candidates.append(post)
        if not candidates:
            continue
        chosen = min(candidates, key=lambda item: (item["published"], item["post_id"]))
        result.at[index, "source_published_at"] = chosen["source_published_at"]
        result.at[index, "source_modified_at"] = chosen["source_modified_at"]
        result.at[index, "timestamp_source_url"] = chosen["source_url"]
        result.at[index, "timestamp_source_id"] = str(chosen["post_id"])
        result.at[index, "timestamp_match_method"] = (
            "exact_audience_and_matchup_in_dated_primary_post"
            if teams is not None
            else "exact_audience_and_network_in_dated_primary_post"
        )
        result.at[index, "point_in_time_usable"] = True
    return result


def enrich_assets(assets: pd.DataFrame, media: pd.DataFrame) -> pd.DataFrame:
    """Attach media timestamps only on a unique exact source-filename match."""

    result = assets.copy()
    for column in ADDED_COLUMNS:
        result[column] = None
    result["source_published_at"] = None
    result["point_in_time_usable"] = False
    by_identity = dict(iter(media.groupby("asset_identity")))
    for index, row in result.iterrows():
        if pd.isna(row.get("week")):
            continue
        matches = by_identity.get(_asset_identity(str(row["asset_url"])))
        if matches is None or len(matches) != 1:
            continue
        match = matches.iloc[0]
        result.at[index, "source_published_at"] = match["source_published_at"]
        result.at[index, "source_modified_at"] = match["source_modified_at"]
        result.at[index, "timestamp_source_url"] = match["attachment_url"]
        result.at[index, "timestamp_source_id"] = str(match["media_id"])
        result.at[index, "timestamp_match_method"] = "exact_asset_upload_path_in_primary_media_api"
        result.at[index, "point_in_time_usable"] = True
    return result


def _verify_source_hashes(source: Path) -> dict[str, str]:
    manifest = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
    expected = manifest.get("output_sha256", {})
    actual: dict[str, str] = {}
    for name in ("ratings_rows.parquet", "source_index.parquet"):
        path = source / name
        digest = _sha256(path.read_bytes())
        actual[name] = digest
        if expected.get(name) != digest:
            raise PublicationBackfillError(f"source hash mismatch for {path}")
    return actual


def _run_config(
    source: Path,
    schedules_path: Path,
    source_hashes: dict[str, str],
    seasons: list[int],
) -> dict[str, Any]:
    return {
        "schema": BACKFILL_SCHEMA,
        "source_snapshot": str(source.resolve()),
        "source_sha256": source_hashes,
        "schedules_path": str(schedules_path.resolve()),
        "schedules_sha256": _sha256(schedules_path.read_bytes()),
        "official_api_root": API_ROOT,
        "seasons": seasons,
        "post_query": {
            "after": "{season}-08-01T00:00:00",
            "before": "{season+1}-03-01T00:00:00",
            "search": "NFL",
            "page_size": POST_PAGE_SIZE,
        },
        "media_query": {
            "search": "normalized asset filename stem",
            "page_size": 100,
        },
        "matching_rule_version": MATCHING_RULE_VERSION,
    }


def _verify_output_hashes(output: Path, manifest: dict[str, Any]) -> None:
    expected = manifest.get("output_sha256")
    if not isinstance(expected, dict) or not expected:
        raise PublicationBackfillError("completed/finalizing manifest has no output hashes")
    for name, digest in expected.items():
        path = output / str(name)
        if not path.is_file() or _sha256(path.read_bytes()) != digest:
            raise PublicationBackfillError(f"output hash mismatch for {path}")


def _prepare_snapshot(output: Path, config: dict[str, Any]) -> dict[str, Any]:
    """Create or validate a resumable output snapshot before any source request."""

    config_path = output / "run_config.json"
    manifest_path = output / "manifest.json"
    config_sha256 = _sha256(_json_payload(config))
    if not output.exists():
        output.mkdir(parents=True)
        _atomic_json(config_path, config, replace=False)
        manifest = {
            "schema": BACKFILL_SCHEMA,
            "status": "IN_PROGRESS",
            "started_at_utc": datetime.now(UTC).isoformat(),
            "run_config_sha256": config_sha256,
        }
        _atomic_json(manifest_path, manifest, replace=False)
        return manifest
    if not config_path.is_file() or not manifest_path.is_file():
        raise PublicationBackfillError(
            f"existing output is not a resumable {BACKFILL_SCHEMA} snapshot: {output}"
        )
    existing_config = json.loads(config_path.read_text(encoding="utf-8"))
    if existing_config != config:
        raise PublicationBackfillError("existing output run configuration is incompatible")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("run_config_sha256") != config_sha256:
        raise PublicationBackfillError("manifest run-config hash is incompatible")
    status = manifest.get("status")
    if status not in {"IN_PROGRESS", "FINALIZING", "COMPLETE"}:
        raise PublicationBackfillError(f"unsupported snapshot status: {status!r}")
    if status == "COMPLETE":
        _verify_output_hashes(output, manifest)
    return manifest


def _stage_parquet_once(path: Path, frame: pd.DataFrame) -> str:
    """Write a staging member once; a compatible resume reuses it by hash."""

    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            dir=path.parent, prefix=f".{path.name}.", suffix=".parquet", delete=False
        ) as file:
            temporary = Path(file.name)
        try:
            frame.to_parquet(temporary, index=False)
            if path.exists():
                raise PublicationBackfillError(f"staging member already exists: {path}")
            temporary.replace(path)
        finally:
            if temporary.exists():
                temporary.unlink()
    return _sha256(path.read_bytes())


def _finish_snapshot(output: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    """Promote hash-pinned staged outputs and seal the snapshot."""

    expected = manifest.get("output_sha256")
    if not isinstance(expected, dict) or not expected:
        raise PublicationBackfillError("finalizing manifest has no output hashes")
    for name, digest in expected.items():
        target = output / str(name)
        staged = output / "staging" / str(name)
        if target.exists():
            if _sha256(target.read_bytes()) != digest:
                raise PublicationBackfillError(f"output hash mismatch for {target}")
            continue
        if not staged.is_file() or _sha256(staged.read_bytes()) != digest:
            raise PublicationBackfillError(f"staging hash mismatch for {staged}")
        staged.replace(target)
    completed = {
        **manifest,
        "status": "COMPLETE",
        "completed_at_utc": manifest.get("completed_at_utc", datetime.now(UTC).isoformat()),
    }
    _atomic_json(output / "manifest.json", completed, replace=True)
    _verify_output_hashes(output, completed)
    return completed


def backfill(
    source: Path,
    output: Path,
    schedules_path: Path,
    *,
    fetcher: Callable[[str], bytes] = _fetch,
    sleeper: Callable[[float], None] = time.sleep,
) -> dict[str, Any]:
    """Build a separate immutable publication-evidence snapshot."""

    if output.resolve() == source.resolve():
        raise PublicationBackfillError("publication backfill output must differ from source")
    source_hashes = _verify_source_hashes(source)
    schedules = pd.read_parquet(schedules_path)
    rows = pd.read_parquet(source / "ratings_rows.parquet")
    assets = pd.read_parquet(source / "source_index.parquet")
    seasons = sorted(set(rows["season"].astype(int)).union(assets["season"].astype(int)))
    config = _run_config(source, schedules_path, source_hashes, seasons)
    snapshot = _prepare_snapshot(output, config)
    if snapshot["status"] == "COMPLETE":
        return snapshot
    if snapshot["status"] == "FINALIZING":
        return _finish_snapshot(output, snapshot)
    posts = fetch_posts(output, seasons, fetcher=fetcher, sleeper=sleeper)
    media = fetch_media(output, assets, fetcher=fetcher, sleeper=sleeper)
    enriched_rows = enrich_rows(rows, posts, schedules)
    enriched_assets = enrich_assets(assets, media)
    outputs = {
        "ratings_rows.parquet": enriched_rows,
        "source_index.parquet": enriched_assets,
        "post_index.parquet": posts,
        "media_index.parquet": media,
    }
    output_hashes: dict[str, str] = {}
    for name, frame in outputs.items():
        output_hashes[name] = _stage_parquet_once(output / "staging" / name, frame)
    eligible_rows = (
        enriched_rows["week"].notna()
        & enriched_rows["away_team"].notna()
        & enriched_rows["home_team"].notna()
        & enriched_rows["viewers"].notna()
    )
    eligible_assets = enriched_assets["week"].notna()
    manifest = {
        **snapshot,
        "status": "FINALIZING",
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "official_api_root": API_ROOT,
        "posts_cached": len(posts),
        "media_matches_cached": len(media),
        "structured_rows_total": len(enriched_rows),
        "structured_rows_feature_eligible": int(eligible_rows.sum()),
        "structured_rows_timestamped": int(
            enriched_rows.loc[eligible_rows, "point_in_time_usable"].sum()
        ),
        "ratings_assets_feature_eligible": int(eligible_assets.sum()),
        "ratings_assets_timestamped": int(
            enriched_assets.loc[eligible_assets, "point_in_time_usable"].sum()
        ),
        "output_sha256": output_hashes,
        "point_in_time_contract": {
            "seasonal_page_timestamp_is_evidence": False,
            "same_game_viewership_allowed": False,
            "feature_scope": "prior-game or season-to-date lag only",
            "rows_require_exact_audience_and_identity": True,
            "assets_require_unique_exact_upload_path": True,
            "unmatched_or_ambiguous_rows_usable": False,
            "publication_and_modification_preserved_separately": True,
        },
    }
    _atomic_json(output / "manifest.json", manifest, replace=True)
    return _finish_snapshot(output, manifest)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--schedules", type=Path, required=True)
    args = parser.parse_args()
    manifest = backfill(args.source, args.output, args.schedules)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
