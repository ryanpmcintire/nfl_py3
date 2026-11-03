import argparse
import hashlib
import json
import re
import sys
import xml.etree.ElementTree as ET
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
BEAT = ROOT / "data" / "raw" / "beat_reporters"
OUT = ROOT / "data" / "raw" / "travel_interruptions"
HANDLES = ROOT / "data" / "processed" / "beat_reporter_handles.csv"
GAMES = ROOT / "data" / "processed" / "game_features.parquet"
FAA = "https://nasstatus.faa.gov/api/airport-status-information"
UA = "nfl-ats-research/0.1 (public unauthenticated read-only, rate limited)"
DAYS_PER_SCHEDULE_WEEK = 7
SCHEMA_VERSION = "travel-interruptions-v2-2026-11-03"

AIRPORT = {
    "ARI": "PHX",
    "ATL": "ATL",
    "BAL": "BWI",
    "BUF": "BUF",
    "CAR": "CLT",
    "CHI": "ORD",
    "CIN": "CVG",
    "CLE": "CLE",
    "DAL": "DFW",
    "DEN": "DEN",
    "DET": "DTW",
    "GB": "GRB",
    "HOU": "IAH",
    "IND": "IND",
    "JAX": "JAX",
    "KC": "MCI",
    "LV": "LAS",
    "LAC": "LAX",
    "LA": "LAX",
    "MIA": "MIA",
    "MIN": "MSP",
    "NE": "BOS",
    "NO": "MSY",
    "NYG": "EWR",
    "NYJ": "EWR",
    "PHI": "PHL",
    "PIT": "PIT",
    "SF": "SFO",
    "SEA": "SEA",
    "TB": "TPA",
    "TEN": "BNA",
    "WAS": "IAD",
}

NFL_NICKNAMES = (
    "Cardinals|Falcons|Ravens|Bills|Panthers|Bears|Bengals|Browns|Cowboys|Broncos|Lions|Packers|"
    "Texans|Colts|Jaguars|Chiefs|Raiders|Chargers|Rams|Dolphins|Vikings|Patriots|Saints|Giants|"
    "Jets|Eagles|Steelers|49ers|Seahawks|Buccaneers|Bucs|Titans|Commanders"
)
TEAM_CONTEXT = re.compile(
    r"#\w+|\b(?:" + NFL_NICKNAMES + r")\b|"
    r"\bthe (?:[A-Z][\w.'\u2019-]*\s){0,2}[A-Z][\w.'\u2019-]*s\b"
)
TRAVEL_NOUN = re.compile(
    r"\b(flights?|plane|planes|charter|airport|tarmac|takeoff|departure|leaving|departing|"
    r"(?<!struggle )bus|"
    r"travel (?:issues|woes|delays|difficulties|problems))\b",
    re.I,
)
NOT_A_REPORT = re.compile(
    r"[?]$|"
    r"^[^\u201c\u201d\"]*\b(?:I|I'm|I\u2019m|I\u2019ve|I've|my|me|we|us|our)\b|"
    r"\b(?:because of|due to)\b[^.]*\btravel\b",
    re.I,
)
SENTENCE_BREAK = re.compile(r"(?<=[.!?])\s+")
INTERRUPTION_RULES = (
    ("flight_cancelled", re.compile(r"\b(cancel(?:led|ed|ation|lation))\b", re.I)),
    (
        "diverted_or_rerouted",
        re.compile(r"\b(divert(?:ed|ing)?|re-?rout(?:ed|ing|e)|alternate airport)\b", re.I),
    ),
    ("mechanical_or_grounded", re.compile(r"\b(mechanical|grounded|maintenance issue)\b", re.I)),
    (
        "weather_disruption",
        re.compile(
            r"\b(?:[Ss]now|[Ss]torm|[Ff]og|icy|ice|[Bb]lizzard|[Hh]urricane|[Tt]ornado|"
            r"[Tt]hunderstorm|[Ww]eather)\b"
        ),
    ),
    (
        "delayed_or_late_arrival",
        re.compile(
            r"\b(delay(?!\s+of\s+game)(?:ed|s)?|hours late|behind schedule|"
            r"(?:stuck|stranded)\s+(?:at|in|on|due|for|because|waiting|overnight))\b",
            re.I,
        ),
    ),
    (
        "flight_problem",
        re.compile(
            r"\bflight (?:issues|problems|trouble)\b|"
            r"\bawaiting the arrival of (?:a|the) (?:second|replacement|new) plane\b",
            re.I,
        ),
    ),
)
SMALL_HOURS_ARRIVAL = re.compile(
    r"\b(?:arrived|landed|got in)\b.*\b(?:hotel|airport)\b.*\b(?:12|[1-5])(?::\d\d)?\s?a\.?m\b"
    r"|\b(?:arrived|landed|got in)\b.*\b(?:12|[1-5])(?::\d\d)?\s?a\.?m\b.*\b(?:hotel|airport)\b",
    re.I,
)
ARRIVAL_ONLY = re.compile(r"\b(landed|arrived|arrives|touched down|arrival)\b", re.I)
FAA_RULES = (
    ("faa_ground_stop_or_closure", re.compile(r"ground stop|closure|closed", re.I)),
    ("faa_ground_delay_program", re.compile(r"ground delay", re.I)),
)


def classify_post(text):
    sentences = SENTENCE_BREAK.split(text)
    arrival = False
    for i, sent in enumerate(sentences):
        window = " ".join(sentences[max(0, i - 1) : i + 1])

        if TEAM_CONTEXT.search(window) is None:
            continue
        if NOT_A_REPORT.search(sent.strip()):
            continue
        travel = TRAVEL_NOUN.search(sent) is not None
        if travel:
            for name, rx in INTERRUPTION_RULES:
                if rx.search(sent):
                    return name
        if SMALL_HOURS_ARRIVAL.search(sent):
            return "late_hour_arrival"
        if travel and ARRIVAL_ONLY.search(sent):
            arrival = True
    if arrival:
        return "arrival_reported_no_interruption_term"
    return None


def classify_faa(name):
    for label, rx in FAA_RULES:
        if rx.search(name):
            return label
    return "faa_arrival_departure_delay"


def digest(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def load_games():
    g = pd.read_parquet(
        GAMES, columns=["game_id", "season", "week", "gameday", "away_team", "home_team"]
    )
    g["gameday"] = pd.to_datetime(g["gameday"], utc=True)
    return g


def next_game(games, team, when):
    m = games[
        ((games.away_team == team) | (games.home_team == team))
        & (games.gameday >= when.normalize())
    ]
    if m.empty:
        return None
    return m.sort_values("gameday").iloc[0]


def seen_keys(out_root):
    keys = set()
    if out_root.exists():
        for p in out_root.glob("*/rows.jsonl"):
            for line in p.read_text(encoding="utf-8").split(chr(10)):
                if not line:
                    continue
                r = json.loads(line)
                if r["source"] == "bluesky_beat":
                    keys.add((r["source_uri"], r["team"], r["game_id"]))
    return keys


def beat_rows(now, games, seen):
    handles = pd.read_csv(HANDLES, usecols=["team", "handle"])
    team_of = dict(zip(handles.handle, handles.team, strict=True))
    since = now - timedelta(days=DAYS_PER_SCHEDULE_WEEK)
    rows, scanned, done = [], 0, set()
    for stamp in sorted(BEAT.iterdir()) if BEAT.exists() else []:
        try:
            st = datetime.strptime(stamp.name, "%Y%m%dT%H%M%SZ").replace(tzinfo=UTC)
        except ValueError:
            continue
        if st < since:
            continue
        for f in stamp.glob("*.jsonl"):
            team = team_of.get(f.stem)
            if team is None:
                continue
            for line in f.read_text(encoding="utf-8").split(chr(10)):
                if not line:
                    continue
                scanned += 1
                p = json.loads(line)["post"]
                created = pd.Timestamp(p["record"]["createdAt"])
                if created.tzinfo is None:
                    created = created.tz_localize("UTC")
                if created < pd.Timestamp(since) or p["uri"] in done:
                    continue
                done.add(p["uri"])
                text = p["record"].get("text", "")
                cls = classify_post(text)
                if cls is None:
                    continue
                g = next_game(games, team, created)
                if g is None or (p["uri"], team, g.game_id) in seen:
                    continue
                rows.append(
                    {
                        "game_id": g.game_id,
                        "team": team,
                        "source": "bluesky_beat",
                        "source_uri": p["uri"],
                        "author": f.stem,
                        "post_created_at_utc": created.isoformat(),
                        "observed_at_utc": p.get("indexedAt"),
                        "text_hash": digest(text),
                        "text_snippet": text[:160],
                        "classification": cls,
                        "rule_version": SCHEMA_VERSION,
                    }
                )
    return rows, scanned


def faa_rows(now, games):
    r = requests.get(FAA, headers={"User-Agent": UA}, timeout=30)
    r.raise_for_status()
    root = ET.fromstring(r.content)
    flagged = {}
    for dtype in root.iter("Delay_type"):
        name = (dtype.findtext("Name") or "").strip()
        for arpt in dtype.iter("ARPT"):
            flagged.setdefault((arpt.text or "").strip(), []).append(name)
    fetched = now.isoformat()
    day0 = pd.Timestamp(now).normalize()
    week = games[
        (games.gameday >= day0) & (games.gameday < day0 + timedelta(days=DAYS_PER_SCHEDULE_WEEK))
    ]
    rows = []
    for g in week.itertuples():
        for team in (g.away_team, g.home_team):
            for role, airport in (("destination", AIRPORT[g.home_team]), ("origin", AIRPORT[team])):
                names = flagged.get(airport, [])
                cls = classify_faa(" ".join(names)) if names else "faa_no_advisory_control"
                txt = f"{role} {airport}: " + ("; ".join(names) if names else "no advisory listed")
                rows.append(
                    {
                        "game_id": g.game_id,
                        "team": team,
                        "source": "faa_nas",
                        "source_uri": FAA,
                        "airport": airport,
                        "airport_role": role,
                        "observed_at_utc": fetched,
                        "text_hash": digest(txt),
                        "text_snippet": txt,
                        "classification": cls,
                        "rule_version": SCHEMA_VERSION,
                    }
                )
    return rows


def main():
    ap = argparse.ArgumentParser(
        description="LEAD-81 forward capture of travel interruption observations"
    )
    ap.add_argument("--skip-faa", action="store_true")
    ap.add_argument("--out-root", default=str(OUT))
    a = ap.parse_args()
    out_root = Path(a.out_root)
    now = datetime.now(UTC)
    stamp = now.strftime("%Y%m%dT%H%M%SZ")
    games = load_games()
    rows, scanned = beat_rows(now, games, seen_keys(out_root))
    faa_status = "skipped"
    if not a.skip_faa:
        try:
            rows += faa_rows(now, games)
            faa_status = "ok"
        except Exception as e:
            faa_status = f"failed: {type(e).__name__}: {e}"
    d = out_root / stamp
    d.mkdir(parents=True, exist_ok=False)
    rp = d / "rows.jsonl"
    rp.write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in rows), encoding="utf-8")
    counts = pd.Series([r["classification"] for r in rows], dtype=object).value_counts().to_dict()
    manifest = {
        "stamp": stamp,
        "schema_version": SCHEMA_VERSION,
        "rows": len(rows),
        "rows_sha256": hashlib.sha256(rp.read_bytes()).hexdigest(),
        "beat_posts_scanned": scanned,
        "faa_status": faa_status,
        "classification_counts": counts,
        "sources": {
            "bluesky_beat": "data/raw/beat_reporters (public unauthenticated)",
            "faa_nas": FAA,
        },
        "silence_policy": (
            "no row means missing, not on-time; "
            "faa_no_advisory_control rows are the only explicit controls"
        ),
    }
    (d / "manifest.json").write_text(
        json.dumps(manifest, indent=1, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(manifest, indent=1, sort_keys=True))


if __name__ == "__main__":
    sys.exit(main())
