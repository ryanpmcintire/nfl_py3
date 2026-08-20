r"""Archive Sports Media Watch NFL ratings without admitting hindsight leakage.

The seasonal pages are living documents.  Older seasons contain HTML tables;
newer seasons contain table images.  This ingester preserves the page, parses
the structured rows that are present, and indexes relevant image assets.  It
does *not* pretend the current revision was available historically.

Only a separately verified publication timestamp can make a row eligible for
a downstream lagged feature.  Same-game viewership is always excluded.

Usage::

    .\.tools\uv.exe run --no-sync python scripts/ingest_sports_media_watch.py \
        --output data/raw/sports_media_watch/20260820T170000Z \
        --seasons 2014-2023 --max-assets 20
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable, Iterable
from datetime import UTC, date, datetime
from html.parser import HTMLParser
from pathlib import Path
from typing import Any

import pandas as pd

BASE_URL = "https://www.sportsmediawatch.com/nfl-tv-ratings-viewership/"
USER_AGENT = "nfl-ats-research/0.1 (private research; polite archival fetch)"
REQUEST_DELAY_SECONDS = 1.5
WEEK_WORDS = {
    "ONE": 1,
    "TWO": 2,
    "THREE": 3,
    "FOUR": 4,
    "FIVE": 5,
    "SIX": 6,
    "SEVEN": 7,
    "EIGHT": 8,
    "NINE": 9,
    "TEN": 10,
    "ELEVEN": 11,
    "TWELVE": 12,
    "THIRTEEN": 13,
    "FOURTEEN": 14,
    "FIFTEEN": 15,
    "SIXTEEN": 16,
    "SEVENTEEN": 17,
    "EIGHTEEN": 18,
}
TEAM_NAMES = {
    "49ers": "SF",
    "Bears": "CHI",
    "Bengals": "CIN",
    "Bills": "BUF",
    "Broncos": "DEN",
    "Browns": "CLE",
    "Buccaneers": "TB",
    "Cardinals": "ARI",
    "Chargers": "LAC",
    "Chiefs": "KC",
    "Colts": "IND",
    "Commanders": "WAS",
    "Cowboys": "DAL",
    "Dolphins": "MIA",
    "Eagles": "PHI",
    "Falcons": "ATL",
    "Giants": "NYG",
    "Jaguars": "JAX",
    "Jets": "NYJ",
    "Lions": "DET",
    "Packers": "GB",
    "Panthers": "CAR",
    "Patriots": "NE",
    "Raiders": "LV",
    "Rams": "LA",
    "Ravens": "BAL",
    "Redskins": "WAS",
    "Saints": "NO",
    "Seahawks": "SEA",
    "Steelers": "PIT",
    "Texans": "HOU",
    "Titans": "TEN",
    "Vikings": "MIN",
    "Washington": "WAS",
}
TEAM_CODES = {
    "ARI",
    "ATL",
    "BAL",
    "BUF",
    "CAR",
    "CHI",
    "CIN",
    "CLE",
    "DAL",
    "DEN",
    "DET",
    "GB",
    "HOU",
    "IND",
    "JAX",
    "KC",
    "LA",
    "LAC",
    "LV",
    "MIA",
    "MIN",
    "NE",
    "NO",
    "NYG",
    "NYJ",
    "PHI",
    "PIT",
    "SEA",
    "SF",
    "TB",
    "TEN",
    "WAS",
}
TEAM_CODE_ALIASES = {"JAC": "JAX", "OAK": "LV", "SD": "LAC", "STL": "LA"}


class SportsMediaWatchIngestError(ValueError):
    """Raised when a source or point-in-time contract changes."""


class _ArchiveParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.in_ratings_table = False
        self.in_row = False
        self.in_cell = False
        self.heading_end_tag: str | None = None
        self.cell_parts: list[str] = []
        self.row: list[str] = []
        self.rows: list[list[str]] = []
        self.heading_parts: list[str] = []
        self.current_heading: str | None = None
        self.row_headings: list[str | None] = []
        self.assets: list[dict[str, str | None]] = []

    @staticmethod
    def _attrs(attrs: list[tuple[str, str | None]]) -> dict[str, str]:
        return {key: value or "" for key, value in attrs}

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        values = self._attrs(attrs)
        classes = set(values.get("class", "").split())
        if tag == "table":
            self.in_ratings_table = True
        elif tag == "tr" and self.in_ratings_table:
            self.in_row = True
            self.row = []
        elif tag in {"td", "th"} and self.in_row:
            self.in_cell = True
            self.cell_parts = []
        if tag in {"h2", "h3", "h4"} or (
            tag == "span" and classes.intersection({"sectionhed", "subhed"})
        ):
            self.heading_end_tag = tag
            self.heading_parts = []
        if tag == "img":
            source = values.get("data-src") or values.get("src")
            alt = values.get("alt", "")
            if source and _is_relevant_asset(source, alt):
                self.assets.append(
                    {"section_label": self.current_heading, "asset_url": source, "alt": alt}
                )

    def handle_endtag(self, tag: str) -> None:
        if tag in {"td", "th"} and self.in_cell:
            self.row.append(_clean_text(" ".join(self.cell_parts)))
            self.in_cell = False
        elif tag == "tr" and self.in_row:
            if any(self.row):
                self.rows.append(self.row)
                self.row_headings.append(self.current_heading)
            self.in_row = False
        elif tag == "table" and self.in_ratings_table:
            self.in_ratings_table = False
        if tag == self.heading_end_tag:
            heading = _clean_text(" ".join(self.heading_parts))
            if heading:
                self.current_heading = heading
            self.heading_end_tag = None

    def handle_data(self, data: str) -> None:
        if self.in_cell:
            self.cell_parts.append(data)
        if self.heading_end_tag is not None:
            self.heading_parts.append(data)


def _clean_text(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def _is_relevant_asset(url: str, alt: str) -> bool:
    parsed = urllib.parse.urlparse(url)
    if parsed.netloc not in {"sportsmediawatch.com", "www.sportsmediawatch.com"}:
        return False
    if "/wp-content/uploads/" not in parsed.path:
        return False
    label = f"{parsed.path} {alt}".lower()
    return any(token in label for token in ("nfl", "rating", "viewership", "week"))


def archive_url(season: int) -> str:
    if season == 2023:
        return BASE_URL
    return f"{BASE_URL}{season}-season/"


def parse_seasons(value: str) -> list[int]:
    seasons: set[int] = set()
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start_text, end_text = part.split("-", maxsplit=1)
            seasons.update(range(int(start_text), int(end_text) + 1))
        else:
            seasons.add(int(part))
    if not seasons:
        raise argparse.ArgumentTypeError("at least one season is required")
    return sorted(seasons)


def _parse_week(value: str) -> int | None:
    match = re.search(r"\bWEEK\s+([A-Z]+|\d{1,2})\b", value.upper())
    if not match:
        return None
    token = match.group(1)
    return int(token) if token.isdigit() else WEEK_WORDS.get(token)


def _parse_event_date(value: str, season: int) -> date | None:
    match = re.fullmatch(
        r"(?:Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday),\s+"
        r"([A-Za-z]+)\s+(\d{1,2})",
        value,
    )
    if not match:
        return None
    month_name, day_text = match.groups()
    month = datetime.strptime(month_name, "%B").month
    year = season + 1 if month <= 3 else season
    return date(year, month, int(day_text))


def _parse_number(value: str) -> float | None:
    match = re.search(r"\d+(?:\.\d+)?", value.replace(",", ""))
    return float(match.group()) if match else None


def _parse_viewers(value: str) -> int | None:
    number = _parse_number(value)
    if number is None:
        return None
    upper = value.upper()
    if "M" in upper:
        return round(number * 1_000_000)
    if "K" in upper:
        return round(number * 1_000)
    return round(number)


def _parse_teams(featured_game: str) -> tuple[str | None, str | None]:
    clean = re.sub(r"\s*\([^)]*\)", "", featured_game).strip()
    code_match = re.match(r"^([A-Z]{2,3})[/-]([A-Z]{2,3})(?:\s|$)", clean)
    if code_match:
        teams = tuple(TEAM_CODE_ALIASES.get(code, code) for code in code_match.groups())
        if teams[0] in TEAM_CODES and teams[1] in TEAM_CODES:
            return teams
    name_match = re.match(r"^([A-Za-z0-9]+)\s*/\s*([A-Za-z0-9]+)(?:\s|$)", clean)
    if name_match:
        return TEAM_NAMES.get(name_match.group(1)), TEAM_NAMES.get(name_match.group(2))
    parts = re.split(r"\s*(?:\bat\b|\bvs\.?\b)\s*", clean, maxsplit=1, flags=re.I)
    if len(parts) != 2:
        return None, None
    return TEAM_NAMES.get(parts[0]), TEAM_NAMES.get(parts[1])


def parse_archive_html(
    payload: bytes, *, season: int, page_url: str, observed_at: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    parser = _ArchiveParser()
    parser.feed(payload.decode("utf-8", errors="replace"))
    week: int | None = None
    event_date: date | None = None
    previous_heading: str | None = None
    records: list[dict[str, Any]] = []
    for cells, heading in zip(parser.rows, parser.row_headings, strict=True):
        joined = " ".join(cells)
        if heading != previous_heading:
            week = _parse_week(heading or "")
            previous_heading = heading
        parsed_week = _parse_week(joined) or _parse_week(heading or "")
        if parsed_week is not None:
            week = parsed_week
        if parsed_week is not None and len(cells) <= 2:
            continue
        if len(cells) <= 2 and re.search(
            r"\b(?:PRESEASON|WILD CARD|DIVISIONAL|CONFERENCE|SUPER BOWL|PRO BOWL)\b",
            joined.upper(),
        ):
            week = None
            continue
        parsed_date = _parse_event_date(joined, season)
        if parsed_date is not None and len(cells) <= 2:
            event_date = parsed_date
            continue
        if len(cells) not in {6, 7, 8, 9, 10} or cells[0].lower() in {
            "window",
            "window/game",
            "date/time",
        }:
            continue
        if len(cells) == 6:
            featured_game = cells[0]
            window = ""
            network, rating, rating_yoy, viewers, viewers_yoy = cells[1:]
        else:
            window, featured_game, network, rating, rating_yoy, viewers, viewers_yoy = cells[:7]
        away_team, home_team = _parse_teams(featured_game)
        records.append(
            {
                "season": season,
                "week": week,
                "event_date": event_date.isoformat() if event_date is not None else None,
                "window": window,
                "featured_game": featured_game,
                "away_team": away_team,
                "home_team": home_team,
                "network": network,
                "household_rating": _parse_number(rating),
                "rating_yoy_text": rating_yoy,
                "viewers": _parse_viewers(viewers),
                "viewers_yoy_text": viewers_yoy,
                "source_page_url": page_url,
                "source_observed_at": observed_at,
                # A current living-page revision does not reveal when this
                # row first became available.  Null is deliberate and is
                # enforced by point_in_time_view below.
                "source_published_at": None,
                "point_in_time_usable": False,
            }
        )
    asset_rows = [
        {
            "season": season,
            "section_label": row["section_label"],
            "week": _parse_week(row["section_label"] or ""),
            "asset_url": row["asset_url"],
            "alt": row["alt"],
            "source_page_url": page_url,
            "source_observed_at": observed_at,
            "source_published_at": None,
            "point_in_time_usable": False,
        }
        for row in parser.assets
    ]
    return pd.DataFrame(records), pd.DataFrame(asset_rows)


def point_in_time_view(rows: pd.DataFrame, *, decision_at: pd.Timestamp) -> pd.DataFrame:
    """Return strictly prior, already-published rows or fail on missing identity."""
    required = {"event_date", "source_published_at"}
    missing = required - set(rows.columns)
    if missing:
        raise SportsMediaWatchIngestError(f"missing point-in-time columns: {sorted(missing)}")
    if rows["source_published_at"].isna().any():
        raise SportsMediaWatchIngestError(
            "ratings lack a verified publication timestamp; current archive revision is not "
            "historically point-in-time safe"
        )
    result = rows.copy()
    event_at = pd.to_datetime(result["event_date"], utc=True)
    published_at = pd.to_datetime(result["source_published_at"], utc=True)
    decision_at = pd.Timestamp(decision_at)
    decision_at = (
        decision_at.tz_localize("UTC") if decision_at.tz is None else decision_at.tz_convert("UTC")
    )
    return result.loc[(event_at < decision_at.normalize()) & (published_at <= decision_at)].copy()


def _fetch(url: str, *, timeout: int = 45, retries: int = 3) -> bytes:
    last_error: Exception | None = None
    for attempt in range(retries):
        request = urllib.request.Request(
            url,
            headers={"User-Agent": USER_AGENT, "Accept": "text/html,image/*;q=0.9,*/*;q=0.1"},
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return response.read()
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError) as error:
            last_error = error
            print(f"fetch failed ({attempt + 1}/{retries}) {url}: {error}", file=sys.stderr)
            time.sleep(3 * (attempt + 1))
    assert last_error is not None
    raise last_error


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def ingest(
    output: Path,
    *,
    seasons: Iterable[int],
    max_assets: int | None,
    fetcher: Callable[[str], bytes] = _fetch,
    sleeper: Callable[[float], None] = time.sleep,
) -> dict[str, Any]:
    seasons = list(seasons)
    output.mkdir(parents=True, exist_ok=True)
    pages_dir = output / "pages"
    assets_dir = output / "assets"
    pages_dir.mkdir(exist_ok=True)
    assets_dir.mkdir(exist_ok=True)
    observed_at = datetime.now(UTC).isoformat()
    manifest_path = output / "manifest.json"
    prior_manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    prior_page_observed = prior_manifest.get("page_observed_at", {})
    fallback_observed = prior_manifest.get("observed_at")
    page_observed_at: dict[str, str] = {}
    all_records: list[pd.DataFrame] = []
    all_assets: list[pd.DataFrame] = []
    fetched_pages: list[int] = []
    cached_pages: list[int] = []
    page_hashes: dict[str, str] = {}

    for index, season in enumerate(seasons):
        page_path = pages_dir / f"{season}.html"
        url = archive_url(season)
        if page_path.exists():
            payload = page_path.read_bytes()
            cached_pages.append(season)
            page_observed_at[page_path.name] = prior_page_observed.get(
                page_path.name, fallback_observed or observed_at
            )
        else:
            if index > 0:
                sleeper(REQUEST_DELAY_SECONDS)
            payload = fetcher(url)
            page_path.write_bytes(payload)
            fetched_pages.append(season)
            page_observed_at[page_path.name] = observed_at
        page_hashes[page_path.name] = _sha256(payload)
        records, assets = parse_archive_html(
            payload,
            season=season,
            page_url=url,
            observed_at=page_observed_at[page_path.name],
        )
        all_records.append(records)
        all_assets.append(assets)

    records = pd.concat(all_records, ignore_index=True) if all_records else pd.DataFrame()
    assets = pd.concat(all_assets, ignore_index=True) if all_assets else pd.DataFrame()
    downloaded = 0
    cached_assets = 0
    if len(assets):
        local_paths: list[str | None] = []
        hashes: list[str | None] = []
        for row in assets.itertuples(index=False):
            parsed = urllib.parse.urlparse(row.asset_url)
            filename = Path(parsed.path).name
            local_path = assets_dir / str(row.season) / filename
            if local_path.exists():
                payload = local_path.read_bytes()
                cached_assets += 1
            elif max_assets is None or downloaded < max_assets:
                local_path.parent.mkdir(parents=True, exist_ok=True)
                sleeper(REQUEST_DELAY_SECONDS)
                payload = fetcher(row.asset_url)
                local_path.write_bytes(payload)
                downloaded += 1
            else:
                local_paths.append(None)
                hashes.append(None)
                continue
            local_paths.append(str(local_path.relative_to(output)).replace("\\", "/"))
            hashes.append(_sha256(payload))
        assets["local_path"] = local_paths
        assets["sha256"] = hashes

    output_hashes: dict[str, str] = {}
    if len(records):
        ratings_path = output / "ratings_rows.parquet"
        records.to_parquet(ratings_path, index=False)
        output_hashes[ratings_path.name] = _sha256(ratings_path.read_bytes())
    if len(assets):
        index_path = output / "source_index.parquet"
        assets.to_parquet(index_path, index=False)
        output_hashes[index_path.name] = _sha256(index_path.read_bytes())
    remaining_assets = int(assets.get("local_path", pd.Series(dtype="object")).isna().sum())
    manifest = {
        "source": "Sports Media Watch NFL TV ratings seasonal archive (primary pages)",
        "source_base_url": BASE_URL,
        "observed_at": observed_at,
        "seasons_requested": list(seasons),
        "pages_fetched_this_run": fetched_pages,
        "pages_cached_before_run": cached_pages,
        "page_observed_at": page_observed_at,
        "page_sha256": page_hashes,
        "output_sha256": output_hashes,
        "structured_rows": len(records),
        "structured_team_identified_rows": (
            int((records["away_team"].notna() & records["home_team"].notna()).sum())
            if len(records)
            else 0
        ),
        "indexed_assets": len(assets),
        "assets_downloaded_this_run": downloaded,
        "assets_cached_before_run": cached_assets,
        "assets_remaining": remaining_assets,
        "point_in_time_contract": {
            "same_game_viewership_allowed": False,
            "feature_scope": "prior-game or season-to-date lag only",
            "archive_rows_usable": False,
            "reason": (
                "living seasonal pages do not expose row-level first-publication timestamps; "
                "source_published_at remains null until linked dated articles are verified"
            ),
        },
        "resume_command": (
            f".\\.tools\\uv.exe run --no-sync python "
            f"scripts\\ingest_sports_media_watch.py --output {output} "
            f"--seasons {min(seasons)}-{max(seasons)}"
            if remaining_assets
            else None
        ),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seasons", type=parse_seasons, default=parse_seasons("2014-2023"))
    parser.add_argument(
        "--max-assets",
        type=int,
        default=None,
        help="maximum missing image assets to download this run (pages are always completed)",
    )
    args = parser.parse_args()
    manifest = ingest(args.output, seasons=args.seasons, max_assets=args.max_assets)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
