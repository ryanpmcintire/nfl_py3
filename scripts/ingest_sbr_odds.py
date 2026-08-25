"""Ingest + validate the sportsbookreviewsonline.com historical NFL odds archive.

Source: https://www.sportsbookreviewsonline.com/scoresoddsarchives/nfl-odds-<season>
(index: .../scoresoddsarchives/nfl/nfloddsarchives.htm). A classic academic-
backtest odds source, now sitting behind an offshore-book affiliate-marketing
redesign (docs/archive/data_source_scout_v2.md, candidate #1). **Measured** this
session: the index page lists exactly 15 season pages, 2007-08 through
2021-22 (no xlsx links found anywhere on the index or any season page --
checked via a plain substring scan of every fetched HTML file); 2022-23
onward return HTTP 200 with an empty table (0 rows), confirming the scout's
"archive stopped being updated after 2021-22" finding.

Every populated season page (**measured**, all 15) is a single HTML
``<table>`` with one header row plus two data rows per game (``VH`` column:
``V``=away, ``H``=home, ``N``=neutral -- Super Bowl only, one game/season) and
columns ``Date, Rot, VH, Team, 1st, 2nd, 3rd, 4th, Final, Open, Close, ML,
2H``. Row counts reproduce known NFL scheduling history exactly for every
season fetched (256 regular-season + 11 playoff games = 267 for 13 of the 15
seasons; 269 for 2020-21's expanded 14-team playoff field; 285 for 2021-22's
first 17-game regular season) -- a structural cross-check that the parser
below is reading the table correctly, not just avoiding exceptions.

Format quirks decoded here (**measured** against this fetch, not assumed from
memory of the classic SBR format):
  * No dependency added for parsing -- ``lxml``/``bs4``/``html5lib``/
    ``openpyxl`` are all absent from the locked uv environment (checked with
    a plain ``importlib.import_module`` probe), so this uses only
    ``html.parser.HTMLParser`` (stdlib) plus ``requests`` (present, though not
    in ``pyproject.toml``'s explicit dependency list -- pulled in transitively).
  * The Open/Close columns interleave point-spread and total across a game's
    two rows: the SMALLER of the two same-column values is the point spread
    (with "pk" parsed as 0 for pick'em), sitting on the favored team's row;
    the LARGER is the total. This is the classic, widely-documented SBR
    parsing convention (**inferred** -- not independently re-derived from a
    site FAQ this session) and it reproduces plausible favorites/totals on
    every hand-checked example below.
  * Team name tokens are inconsistent within and across seasons: relocations
    (SanDiego/LAChargers, St.Louis/LARams, Oakland/LasVegas/LVRaiders),
    outright typos ("Washingtom", one occurrence, 2020-21), truncations
    ("Kansas", "Tampa", both 2020-21 only), and one ambiguous bare "NewYork"
    (2013-14, single occurrence) -- **measured**, all 44 unique raw tokens
    enumerated from the actual fetched files, not guessed. The "NewYork" case
    is a cautionary tale for this whole exercise: an initial resolution by
    unverified memory ("Jets beat Raiders 24-20 that week") was wrong -- the
    Jets had a bye that week -- and only checking game_features.parquet
    directly (game_id "2013_10_OAK_NYG") caught it. See TEAM_MAP below.
  * No missing/blank Open or Close cells were found in this fetch (measured:
    every one of 8,050 data-row Open/Close/ML cells is either numeric or
    "pk" -- zero blanks, zero "NL" markers). The task brief's "occasional
    missing opens" caveat did not reproduce in this snapshot; reported as
    measured zero, not assumed absent.

Repo team codes: this project's ``data/processed/game_features.parquet`` uses
CURRENT franchise codes retroactively for the whole history (**measured**:
LA/LAC/LV appear for every season 2009-2026; STL/SD/OAK never appear at all),
so St.Louis/SanDiego/Oakland all map to LA/LAC/LV below, matching that
convention rather than the SBR-era city name.

``spread_line`` sign convention in ``game_features.parquet`` (**measured**,
not assumed): POSITIVE means the HOME team is favored by that many points
(``ats_margin = (home_score - away_score) - spread_line`` reproduces the
stored ``ats_margin`` column exactly on inspected rows; a spread_line=14.0
2009 week-1 NO(home)-vs-DET game, DET's real historical ~13.5-14 point
underdog, confirms the direction). This script's own ``open_home_spread``/
``close_home_spread`` columns are built to the same convention: positive =
home favored.

Usage (from repo root, locked env):
    .\\.tools\\uv.exe run --no-sync python scripts/ingest_sbr_odds.py \\
        --raw-root data/raw/sbr_odds --out data/processed/sbr_odds.parquet

    # Reuse an already-fetched raw snapshot instead of re-hitting the site:
    .\\.tools\\uv.exe run --no-sync python scripts/ingest_sbr_odds.py \\
        --skip-fetch --snapshot data/raw/sbr_odds/<timestamp> \\
        --out data/processed/sbr_odds.parquet

    # Fetch + parse + run the three validation checks and print a summary:
    .\\.tools\\uv.exe run --no-sync python scripts/ingest_sbr_odds.py --validate \\
        --game-features data/processed/game_features.parquet \\
        --market-root data/market/raw
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from datetime import UTC, datetime
from html.parser import HTMLParser
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from nfl_ats.io import atomic_json, atomic_parquet  # noqa: E402

SEASON_SLUGS: tuple[str, ...] = (
    "2007-08",
    "2008-09",
    "2009-10",
    "2010-11",
    "2011-12",
    "2012-13",
    "2013-14",
    "2014-15",
    "2015-16",
    "2016-17",
    "2017-18",
    "2018-19",
    "2019-20",
    "2020-21",
    "2021-22",
)
INDEX_URL = "https://www.sportsbookreviewsonline.com/scoresoddsarchives/nfl/nfloddsarchives.htm"
SEASON_URL_TMPL = "https://www.sportsbookreviewsonline.com/scoresoddsarchives/nfl-odds-{slug}"
ROBOTS_URL = "https://www.sportsbookreviewsonline.com/robots.txt"
USER_AGENT = "nfl-ats-research/0.1 (private research; contact ryanpmcintire@gmail.com)"
DELAY_SECONDS = 1.5

# Every one of the 44 unique raw "Team" tokens actually observed across all
# 15 fetched season files (see module docstring). Exact-match dict, not a
# fuzzy/substring matcher -- the token set is finite and known.
TEAM_MAP: dict[str, str] = {
    "Arizona": "ARI",
    "Atlanta": "ATL",
    "Baltimore": "BAL",
    "Buffalo": "BUF",
    "BuffaloBills": "BUF",
    "Carolina": "CAR",
    "Chicago": "CHI",
    "Cincinnati": "CIN",
    "Cleveland": "CLE",
    "Dallas": "DAL",
    "Denver": "DEN",
    "Detroit": "DET",
    "GreenBay": "GB",
    "Houston": "HOU",
    "HoustonTexans": "HOU",
    "Indianapolis": "IND",
    "Jacksonville": "JAX",
    "KCChiefs": "KC",
    "Kansas": "KC",
    "KansasCity": "KC",
    "LAChargers": "LAC",
    "LARams": "LA",
    "LVRaiders": "LV",
    "LasVegas": "LV",
    # Only occurs 2016-17 (n=1 season); unambiguous that year since the
    # Chargers were still tokenized "SanDiego" (they did not move to LA
    # until 2017-18) -- measured from the same token scan.
    "LosAngeles": "LA",
    "Miami": "MIA",
    "Minnesota": "MIN",
    "NYGiants": "NYG",
    "NYJets": "NYJ",
    "NewEngland": "NE",
    "NewOrleans": "NO",
    # Single occurrence, 2013-14, Rot 210, home team beat visiting Oakland
    # 24-20 on date "1110" (Nov 10 2013). **Measured** against
    # game_features.parquet (game_id "2013_10_OAK_NYG", home_team=NYG,
    # home_score=24, away_score=20, gameday 2013-11-10): this is the
    # Giants, not the Jets -- an initial guess of NYJ (from an unverified
    # memory of a Jets-Raiders score that week) was wrong; the Jets actually
    # had a Week 10 bye that season. Corrected after checking the source.
    "NewYork": "NYG",
    "Oakland": "LV",
    "Philadelphia": "PHI",
    "Pittsburgh": "PIT",
    "SanDiego": "LAC",
    "SanFrancisco": "SF",
    "Seattle": "SEA",
    "St.Louis": "LA",
    "Tampa": "TB",
    "TampaBay": "TB",
    "Tennessee": "TEN",
    "Washingtom": "WAS",  # typo, 2020-21 only
    "Washington": "WAS",
}

RAW_ROW_COLUMNS = [
    "Date",
    "Rot",
    "VH",
    "Team",
    "1st",
    "2nd",
    "3rd",
    "4th",
    "Final",
    "Open",
    "Close",
    "ML",
    "2H",
]


# ---------------------------------------------------------------------------
# 1. Fetch
# ---------------------------------------------------------------------------


def fetch_all(raw_root: Path) -> Path:
    """Fetch the index page, robots.txt, and all 15 populated season pages.

    Writes into ``raw_root/<UTC timestamp>/`` with a ``manifest.json``
    recording every URL, fetch time, HTTP status, byte length, and SHA-256.
    """

    import requests

    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    out_dir = raw_root / stamp
    out_dir.mkdir(parents=True, exist_ok=True)

    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})

    files: dict[str, Any] = {}

    def fetch_one(url: str, filename: str) -> None:
        response = session.get(url, timeout=30)
        destination = out_dir / filename
        destination.write_bytes(response.content)
        files[filename] = {
            "url": url,
            "status_code": response.status_code,
            "bytes": len(response.content),
            "sha256": hashlib.sha256(response.content).hexdigest(),
            "fetched_at_utc": datetime.now(UTC).isoformat(),
        }
        print(f"  {filename}: HTTP {response.status_code}, {len(response.content)} bytes")
        time.sleep(DELAY_SECONDS)

    print(f"Fetching into {out_dir}")
    fetch_one(INDEX_URL, "nfloddsarchives_index.html")
    fetch_one(ROBOTS_URL, "robots.txt")
    for slug in SEASON_SLUGS:
        fetch_one(SEASON_URL_TMPL.format(slug=slug), f"nfl-odds-{slug}.html")

    manifest = {
        "source": (
            "https://www.sportsbookreviewsonline.com/scoresoddsarchives/ "
            "(NFL historical odds archive)"
        ),
        "access_date_utc": datetime.now(UTC).isoformat(),
        "user_agent": USER_AGENT,
        "delay_seconds_between_requests": DELAY_SECONDS,
        "robots_txt_result": (
            "Allow: / (Disallow only covers /go/); no Crawl-delay directive found -- "
            "measured this fetch, see robots.txt in this same directory"
        ),
        "seasons_requested": list(SEASON_SLUGS),
        "files": files,
        "note": (
            "Public archive page, fetched for private research use only (matching this "
            "project's existing CFBD/nflverse precedent, docs/data_feasibility.md). No "
            "xlsx download links were found on the index page or any season page (plain "
            "substring scan of the raw HTML) -- the redesigned site serves HTML tables "
            "only; parsed from HTML accordingly, no new dependency added."
        ),
    }
    atomic_json(manifest, out_dir / "manifest.json")
    print(f"Wrote manifest: {out_dir / 'manifest.json'}")
    return out_dir


def latest_snapshot(raw_root: Path) -> Path:
    candidates = sorted(p for p in raw_root.glob("*") if (p / "manifest.json").is_file())
    if not candidates:
        raise FileNotFoundError(f"No sbr_odds snapshots with a manifest found under {raw_root}")
    return candidates[-1]


# ---------------------------------------------------------------------------
# 2. Parse
# ---------------------------------------------------------------------------


class _OddsTableParser(HTMLParser):
    """Minimal stdlib HTML table extractor (no lxml/bs4 in the locked env)."""

    def __init__(self) -> None:
        super().__init__()
        self.rows: list[list[str]] = []
        self._cur_row: list[str] | None = None
        self._cur_cell: list[str] | None = None
        self._in_table = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "table":
            self._in_table = True
        elif tag == "tr" and self._in_table:
            self._cur_row = []
        elif tag == "td" and self._in_table:
            self._cur_cell = []

    def handle_endtag(self, tag: str) -> None:
        if tag == "table":
            self._in_table = False
        elif tag == "tr" and self._cur_row is not None:
            self.rows.append(self._cur_row)
            self._cur_row = None
        elif tag == "td" and self._cur_cell is not None:
            assert self._cur_row is not None
            self._cur_row.append("".join(self._cur_cell).strip())
            self._cur_cell = None

    def handle_data(self, data: str) -> None:
        if self._cur_cell is not None:
            self._cur_cell.append(data)


def parse_raw_table(html: str) -> pd.DataFrame:
    """Parse one season page's single table into a long DataFrame (one row/team/game)."""

    parser = _OddsTableParser()
    parser.feed(html)
    if not parser.rows:
        return pd.DataFrame(columns=RAW_ROW_COLUMNS)
    header, *data_rows = parser.rows
    if header != RAW_ROW_COLUMNS:
        raise ValueError(f"Unexpected table header: {header}")
    bad = [row for row in data_rows if len(row) != len(RAW_ROW_COLUMNS)]
    if bad:
        raise ValueError(f"{len(bad)} data row(s) do not have {len(RAW_ROW_COLUMNS)} cells")
    return pd.DataFrame(data_rows, columns=RAW_ROW_COLUMNS)


def _to_number(value: str) -> float:
    if value.strip().lower() == "pk":
        return 0.0
    return float(value)


def _parse_date(raw: str, season_start_year: int) -> pd.Timestamp:
    digits = raw.strip()
    if len(digits) == 3:
        month, day = int(digits[0]), int(digits[1:3])
    elif len(digits) == 4:
        month, day = int(digits[0:2]), int(digits[2:4])
    else:
        raise ValueError(f"Unrecognized SBR date token: {raw!r}")
    year = season_start_year if month >= 7 else season_start_year + 1
    return pd.Timestamp(year=year, month=month, day=day)


def _map_team(raw_token: str) -> str:
    code = TEAM_MAP.get(raw_token)
    if code is None:
        raise KeyError(f"Unmapped SBR team token: {raw_token!r} (add it to TEAM_MAP)")
    return code


def _disambiguate_spread_total(away_value: float, home_value: float) -> tuple[float, float, bool]:
    """Return (home_spread, total, ambiguous) from a game's two same-column values.

    Convention (measured/documented in the module docstring): the value with
    the SMALLER MAGNITUDE is the point spread, sitting on the favored team's
    row; the LARGER-magnitude value is the total. ``home_spread`` follows
    this repo's ``spread_line`` convention: positive means the home team is
    favored.

    Magnitude, not the raw signed value, decides which row is the spread:
    every season 2007-08 through 2020-21 stores both numbers unsigned
    (unfavored team above ~30, favored team below ~26), but **measured**
    against the actual fetch, the 2021-22 Open column switches to an
    explicit signed convention for an entire week (11 games dated "1114",
    e.g. raw Pittsburgh Open="-9" vs Detroit Open="44") plus two isolated
    rows elsewhere -- 13 rows total out of 570, all in 2021-22, zero in every
    other season. Comparing signed values directly there would have flipped
    the spread's sign (min(44, -9) = -9, not +9); comparing absolute values
    reproduces the correct favorite in both the signed and unsigned rows,
    since the explicit minus sign in the signed rows is always attached to
    whichever row already has the smaller magnitude.
    """

    away_magnitude, home_magnitude = abs(away_value), abs(home_value)
    spread_magnitude = min(away_magnitude, home_magnitude)
    total = max(away_magnitude, home_magnitude)
    home_favored = home_magnitude <= away_magnitude
    home_spread = spread_magnitude if home_favored else -spread_magnitude
    # Flag cases where the min/max heuristic is questionable: a "spread" over
    # 26 points has essentially never been posted in the modern NFL, and a
    # "total" under 33 is very rare -- either signals the two numbers may not
    # actually split cleanly into (spread, total) this way.
    ambiguous = spread_magnitude > 26.0 or total < 33.0
    return home_spread, total, ambiguous


def build_games(raw_long: pd.DataFrame, season: int) -> pd.DataFrame:
    """Pair consecutive away/home rows into one row per game."""

    if len(raw_long) % 2 != 0:
        raise ValueError(f"Season {season}: odd number of rows ({len(raw_long)}), cannot pair")
    records: list[dict[str, Any]] = []
    rows = raw_long.to_dict("records")
    for i in range(0, len(rows), 2):
        first, second = rows[i], rows[i + 1]
        vh_pair = {first["VH"], second["VH"]}
        if vh_pair not in ({"V", "H"}, {"N"}):
            raise ValueError(f"Season {season} row {i}: unexpected VH pair {vh_pair}")
        # SBR row-order convention: first row = away position, second row =
        # home position -- literal V/H when present; for neutral-site games
        # ("N", Super Bowl only, ~1/season) this is a positional ASSUMPTION,
        # not a verified home designation (flagged via neutral_site below).
        away, home = first, second
        neutral_site = first["VH"] == "N"

        away_date = _parse_date(away["Date"], season)
        home_date = _parse_date(home["Date"], season)
        if away_date != home_date:
            raise ValueError(
                f"Season {season} row {i}: mismatched dates {away_date} vs {home_date}"
            )

        away_open, home_open = _to_number(away["Open"]), _to_number(home["Open"])
        away_close, home_close = _to_number(away["Close"]), _to_number(home["Close"])
        open_home_spread, open_total, open_ambiguous = _disambiguate_spread_total(
            away_open, home_open
        )
        close_home_spread, close_total, close_ambiguous = _disambiguate_spread_total(
            away_close, home_close
        )

        records.append(
            {
                "season": season,
                "game_date": away_date,
                "sbr_date_raw": away["Date"],
                "neutral_site": neutral_site,
                "away_team_raw": away["Team"],
                "home_team_raw": home["Team"],
                "away_team": _map_team(away["Team"]),
                "home_team": _map_team(home["Team"]),
                "away_rot": int(away["Rot"]),
                "home_rot": int(home["Rot"]),
                "away_score": float(away["Final"]),
                "home_score": float(home["Final"]),
                "open_home_spread": open_home_spread,
                "close_home_spread": close_home_spread,
                "open_total": open_total,
                "close_total": close_total,
                "open_ambiguous": open_ambiguous,
                "close_ambiguous": close_ambiguous,
                "away_moneyline": float(away["ML"]),
                "home_moneyline": float(home["ML"]),
            }
        )
    return pd.DataFrame.from_records(records)


def parse_snapshot(snapshot_dir: Path) -> pd.DataFrame:
    """Parse every season file in a fetched raw snapshot into one long table."""

    frames: list[pd.DataFrame] = []
    for slug in SEASON_SLUGS:
        path = snapshot_dir / f"nfl-odds-{slug}.html"
        if not path.is_file():
            print(f"  WARNING: {path} not found, skipping season {slug}", file=sys.stderr)
            continue
        html = path.read_text(encoding="utf-8")
        raw_long = parse_raw_table(html)
        season = int(slug.split("-")[0])
        games = build_games(raw_long, season)
        games["sbr_season_slug"] = slug
        frames.append(games)
        print(f"  parsed {slug}: {len(games)} games")
    if not frames:
        raise ValueError(f"No season files parsed from {snapshot_dir}")
    combined = pd.concat(frames, ignore_index=True)
    combined["game_date"] = pd.to_datetime(combined["game_date"])
    return combined


def attach_game_features(sbr: pd.DataFrame, game_features_path: Path) -> pd.DataFrame:
    """Left-join week/game_id from game_features.parquet on (season, home, away, date).

    Two passes: an exact-date join first, then -- for rows that still miss --
    a same-(season, home, away) join to the nearest game_features date within
    1 day. **Measured**: a handful of games (2010, 2014, and a whole 2015
    week-6 Sunday slate + that season's conference championship Sunday, 14
    games total) have an SBR ``Date`` token exactly one calendar day off from
    nflverse's ``gameday`` -- both directions occur (SBR one day after real
    for a Monday-night game; SBR one day before real for two full Sunday
    slates), so this is a genuine SBR-side date quirk, not a single
    correctable offset; the tolerant pass is recorded via
    ``week_match_date_diff_days`` (0 for an exact match, 1 for a
    tolerance-recovered match, null when neither found a match) rather than
    silently overwriting ``game_date``.

    Unmatched games in 2007-2008 stay null in both passes -- game_features.parquet's
    own coverage starts at season 2009 (measured, see docs/sbr_odds_archive.md).
    """

    features = pd.read_parquet(
        game_features_path,
        columns=["game_id", "season", "week", "home_team", "away_team", "gameday"],
    )
    features = features.copy()
    features["game_date"] = pd.to_datetime(features["gameday"]).dt.normalize()
    exact_lookup = features[
        ["game_id", "season", "week", "home_team", "away_team", "game_date"]
    ].drop_duplicates(subset=["season", "home_team", "away_team", "game_date"])
    merged = sbr.merge(
        exact_lookup,
        on=["season", "home_team", "away_team", "game_date"],
        how="left",
        validate="m:1",
    )
    merged["week_match_date_diff_days"] = np.where(merged["game_id"].notna(), 0, np.nan)

    unmatched_mask = merged["game_id"].isna()
    if unmatched_mask.any():
        pair_lookup = features[["game_id", "season", "week", "home_team", "away_team", "game_date"]]
        for idx in merged.loc[unmatched_mask].index:
            row = merged.loc[idx]
            candidates = pair_lookup.loc[
                pair_lookup["season"].eq(row["season"])
                & pair_lookup["home_team"].eq(row["home_team"])
                & pair_lookup["away_team"].eq(row["away_team"])
            ]
            if candidates.empty:
                continue
            gaps = (candidates["game_date"] - row["game_date"]).abs()
            best = gaps.idxmin()
            if gaps.loc[best] <= pd.Timedelta(days=1):
                merged.loc[idx, "game_id"] = candidates.loc[best, "game_id"]
                merged.loc[idx, "week"] = candidates.loc[best, "week"]
                merged.loc[idx, "week_match_date_diff_days"] = int(gaps.loc[best].days)
    return merged


def write_processed(sbr: pd.DataFrame, out_path: Path, *, raw_snapshot_dir: Path) -> None:
    atomic_parquet(sbr, out_path)
    manifest = {
        "built_at_utc": datetime.now(UTC).isoformat(),
        "source_raw_snapshot": str(raw_snapshot_dir),
        "rows": len(sbr),
        "seasons": sorted(int(s) for s in sbr["season"].unique()),
        "games_per_season": sbr.groupby("season").size().to_dict(),
        "matched_to_game_features_rows": (
            int(sbr["game_id"].notna().sum()) if "game_id" in sbr.columns else None
        ),
        "columns": list(sbr.columns),
        "provenance_note": (
            "home_spread positive = home favored (this repo's game_features.parquet "
            "spread_line convention, measured via exact ats_margin reproduction, see "
            "module docstring). Spread/total split via the min-value=spread heuristic; "
            "see open_ambiguous/close_ambiguous flags for rows where that heuristic is "
            "questionable (spread candidate > 26 or total candidate < 33)."
        ),
    }
    manifest_path = out_path.with_suffix(".manifest.json")
    atomic_json(manifest, manifest_path)
    print(f"Wrote {out_path} ({len(sbr)} rows) and {manifest_path}")


# ---------------------------------------------------------------------------
# 3. Validation
# ---------------------------------------------------------------------------


def close_check(sbr: pd.DataFrame, game_features_path: Path) -> pd.DataFrame:
    """Per-season CLOSE validation: match rate, mean |diff|, share within 0.5pt.

    ``sbr`` must already carry ``game_id`` (see :func:`attach_game_features`,
    called in ``main()`` before this runs) -- joins on that key rather than
    re-deriving (season, home, away, date), so the exact-date-or-1-day-
    tolerant matching already done there is reused, not redone differently.
    """

    features = pd.read_parquet(game_features_path, columns=["game_id", "spread_line"])
    joined = sbr.loc[sbr["game_id"].notna()].merge(features, on="game_id", how="inner")
    joined["diff"] = (joined["close_home_spread"] - joined["spread_line"]).abs()

    rows = []
    for season, group in sbr.groupby("season"):
        n_sbr = len(group)
        matched = joined.loc[joined["season"].eq(season)]
        n_matched = len(matched)
        rows.append(
            {
                "season": int(season),
                "sbr_games": n_sbr,
                "matched_games": n_matched,
                "match_rate": n_matched / n_sbr if n_sbr else float("nan"),
                "mean_abs_diff": float(matched["diff"].mean()) if n_matched else float("nan"),
                "median_abs_diff": float(matched["diff"].median()) if n_matched else float("nan"),
                "share_within_0.5pt": (
                    float((matched["diff"] <= 0.5).mean()) if n_matched else float("nan")
                ),
                "share_exact": float((matched["diff"] == 0.0).mean())
                if n_matched
                else float("nan"),
            }
        )
    return pd.DataFrame(rows).sort_values("season").reset_index(drop=True)


def opener_check(sbr: pd.DataFrame, market_root: Path, game_features_path: Path) -> dict[str, Any]:
    """OPENER validation: SBR's Open vs the store's Tuesday-opener consensus.

    Restricted to seasons where BOTH archives have coverage (measured:
    ``tue_open`` decision-label snapshots exist in ``data/market/raw`` for
    seasons 2020-2025; SBR's populated range tops out at 2021-22 -> overlap
    is seasons 2020 and 2021).
    """

    from nfl_ats.clv import CLOSE_LABEL_PRIORITY, build_pairing_table  # noqa: F401
    from nfl_ats.odds_backfill import HISTORICAL_CAPTURE_KIND

    features = pd.read_parquet(game_features_path)
    overlap_seasons = sorted(set(sbr["season"].unique()) & {2020, 2021})
    if not overlap_seasons:
        return {
            "overlap_seasons": [],
            "note": "No season overlap between SBR and the tue_open store.",
        }

    pairing = build_pairing_table(
        market_root,
        capture_kind=HISTORICAL_CAPTURE_KIND,
        labels=("tue_open",),
        seasons=overlap_seasons,
        schedule=features,
    )
    tue_open = pairing.loc[pairing["decision_label"].eq("tue_open")][
        ["game_id", "season", "home_spread"]
    ].rename(columns={"home_spread": "tue_open_home_spread"})

    joined = sbr.loc[sbr["game_id"].notna()].merge(
        tue_open.drop(columns=["season"]), on="game_id", how="inner"
    )
    joined["diff"] = (joined["open_home_spread"] - joined["tue_open_home_spread"]).abs()

    per_season = []
    for season in overlap_seasons:
        n_sbr = int((sbr["season"] == season).sum())
        n_tue_open_available = int((tue_open["season"] == season).sum())
        matched = joined.loc[joined["season"].eq(season)]
        n_matched = len(matched)
        per_season.append(
            {
                "season": season,
                "sbr_games": n_sbr,
                "tue_open_games_available": n_tue_open_available,
                "matched_games": n_matched,
                "mean_abs_diff": float(matched["diff"].mean()) if n_matched else float("nan"),
                "median_abs_diff": float(matched["diff"].median()) if n_matched else float("nan"),
                "share_within_0.5pt": (
                    float((matched["diff"] <= 0.5).mean()) if n_matched else float("nan")
                ),
                "share_exact": float((matched["diff"] == 0.0).mean())
                if n_matched
                else float("nan"),
            }
        )
    overall_n = len(joined)
    return {
        "overlap_seasons": overlap_seasons,
        "per_season": per_season,
        "overall_matched_games": overall_n,
        "overall_mean_abs_diff": float(joined["diff"].mean()) if overall_n else float("nan"),
        "overall_share_within_0.5pt": float((joined["diff"] <= 0.5).mean())
        if overall_n
        else float("nan"),
    }


def coverage_check(
    sbr: pd.DataFrame, game_features_path: Path, *, n_examples: int = 10
) -> dict[str, Any]:
    """Per-season games matched vs total in game_features; unmatched examples.

    ``sbr`` must already carry ``game_id``/``week_match_date_diff_days``
    (see :func:`attach_game_features`) -- "matched" here means an exact-date
    OR 1-day-tolerant match found one, and ``matched_exact_date`` is broken
    out separately so the tolerant recoveries stay visible, not folded in
    silently.
    """

    features = pd.read_parquet(game_features_path, columns=["season"])
    gf_counts = features.groupby("season").size().to_dict()

    rows = []
    unmatched_examples: list[dict[str, Any]] = []
    for season, group in sbr.groupby("season"):
        n_sbr = len(group)
        n_matched = int(group["game_id"].notna().sum())
        n_matched_exact = int(group["week_match_date_diff_days"].eq(0).sum())
        n_gf = int(gf_counts.get(season, 0))
        rows.append(
            {
                "season": int(season),
                "sbr_games": n_sbr,
                "matched_games": n_matched,
                "matched_exact_date": n_matched_exact,
                "matched_1day_tolerance": n_matched - n_matched_exact,
                "game_features_games": n_gf,
                "match_rate_of_sbr": n_matched / n_sbr if n_sbr else float("nan"),
            }
        )
        unmatched = group.loc[group["game_id"].isna()]
        for _, row in unmatched.head(3).iterrows():
            if len(unmatched_examples) >= n_examples:
                continue
            unmatched_examples.append(
                {
                    "season": int(row["season"]),
                    "game_date": str(row["game_date"].date()),
                    "away_team": row["away_team"],
                    "home_team": row["home_team"],
                    "away_team_raw": row["away_team_raw"],
                    "home_team_raw": row["home_team_raw"],
                }
            )
    return {
        "per_season": pd.DataFrame(rows).sort_values("season").reset_index(drop=True),
        "unmatched_examples": unmatched_examples,
    }


# ---------------------------------------------------------------------------
# 4. CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--raw-root", type=Path, default=REPO_ROOT / "data" / "raw" / "sbr_odds")
    parser.add_argument(
        "--out", type=Path, default=REPO_ROOT / "data" / "processed" / "sbr_odds.parquet"
    )
    parser.add_argument(
        "--game-features",
        type=Path,
        default=REPO_ROOT / "data" / "processed" / "game_features.parquet",
    )
    parser.add_argument("--market-root", type=Path, default=REPO_ROOT / "data" / "market" / "raw")
    parser.add_argument("--skip-fetch", action="store_true", help="Reuse an existing raw snapshot")
    parser.add_argument(
        "--snapshot",
        type=Path,
        default=None,
        help="Explicit raw snapshot dir (implies --skip-fetch)",
    )
    parser.add_argument(
        "--validate", action="store_true", help="Run and print the three validation checks"
    )
    args = parser.parse_args()

    if args.snapshot is not None:
        snapshot_dir = args.snapshot
    elif args.skip_fetch:
        snapshot_dir = latest_snapshot(args.raw_root)
        print(f"Reusing latest snapshot: {snapshot_dir}")
    else:
        snapshot_dir = fetch_all(args.raw_root)

    print("\nParsing...")
    sbr = parse_snapshot(snapshot_dir)
    sbr = attach_game_features(sbr, args.game_features)
    write_processed(sbr, args.out, raw_snapshot_dir=snapshot_dir)

    if args.validate:
        print("\n=== CLOSE check (vs game_features.parquet spread_line) ===")
        close_result = close_check(sbr, args.game_features)
        print(close_result.to_string(index=False))

        print("\n=== OPENER check (vs data/market/raw tue_open consensus) ===")
        opener_result = opener_check(sbr, args.market_root, args.game_features)
        print(json.dumps(opener_result, indent=2, default=str))

        print("\n=== COVERAGE check ===")
        coverage_result = coverage_check(sbr, args.game_features)
        print(coverage_result["per_season"].to_string(index=False))
        print("\nUnmatched examples:")
        print(json.dumps(coverage_result["unmatched_examples"], indent=2, default=str))


if __name__ == "__main__":
    main()
