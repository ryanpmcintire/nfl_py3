import argparse
import hashlib
import json
import re
import shutil
import subprocess
from datetime import UTC, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts" / "lead99_llm_read"
PROMPT = ROOT / "scripts" / "lead99_prompt.md"
FROZEN = ART / "frozen.json"
ET = ZoneInfo("America/New_York")
UTC = UTC
MODEL = "claude-opus-5-5"
TOOLS = "WebSearch,WebFetch"
SEASON = 2026
INJURY_STATUSES = ("Out", "Doubtful", "Questionable")
PACKET_FIELDS = [
    "game_id",
    "away",
    "home",
    "kickoff_et",
    "deadline_et",
    "pool_line",
    "market",
    "injuries",
    "injury_headlines",
    "inactives",
    "beat_posts",
    "weather",
    "rest",
]
STAMP = re.compile(r"\d{8}T\d{6}Z")


def sha256_bytes(b):
    return hashlib.sha256(b).hexdigest()


def prompt_sha():
    return sha256_bytes(PROMPT.read_bytes())


def iso(ts):
    return pd.Timestamp(ts).tz_convert("UTC").strftime("%Y-%m-%dT%H:%M:%SZ")


def dir_time(name):
    return datetime.strptime(name[:16], "%Y%m%dT%H%M%SZ").replace(tzinfo=UTC)


def stamp_dirs(base, run_at):
    return sorted(
        d for d in base.iterdir() if STAMP.fullmatch(d.name[:16]) and dir_time(d.name) <= run_at
    )


def splash_files(week):
    return sorted((ROOT / "data" / "splash").glob(f"{SEASON}_week{week:02d}_*.json"))


def load_splash(week, run_at):
    chosen = None
    for f in splash_files(week):
        d = json.loads(f.read_text())
        if datetime.fromisoformat(d["captured_at_et"]) <= run_at:
            chosen = (f, d)
    if chosen is None:
        raise SystemExit("no splash capture before run instant")
    return chosen


def current_week(run_at):
    weeks = sorted(
        {
            int(f.name[len(f"{SEASON}_week") :][:2])
            for f in (ROOT / "data" / "splash").glob(f"{SEASON}_week*.json")
        }
    )
    for week in weeks:
        _, d = load_splash(week, run_at)
        if any(datetime.fromisoformat(g["kickoff_et"]) > run_at for g in d["games"]):
            return week
    raise SystemExit("no captured pool week has a game still to kick off")


def previous_kickoffs(week, run_at):
    prev = {}
    for w in range(1, week):
        fs = splash_files(w)
        if not fs:
            continue
        d = json.loads(fs[-1].read_text())
        for g in d["games"]:
            k = datetime.fromisoformat(g["kickoff_et"])
            if k <= run_at:
                for t in (g["home"], g["away"]):
                    prev[t] = max(prev.get(t, k), k)
    return prev


def deadline_for(kick, anchor):
    a = anchor.astimezone(ET)
    days = (6 - a.weekday()) % 7
    sun = (a + timedelta(days=days)).replace(hour=16, minute=0, second=0, microsecond=0)
    return min(kick, sun)


def market_block(run_at, week_start):
    frames = []
    for d in stamp_dirs(ROOT / "data" / "market" / "raw", run_at):
        if dir_time(d.name) < week_start:
            continue
        q = pd.read_parquet(d / "quotes.parquet")
        q = q[(q["market"] == "spreads") & (q["outcome_side"] == "HOME")]
        frames.append(q[["observed_at_utc", "nflverse_game_id", "bookmaker_key", "line"]])
    out = {}
    if not frames:
        return out
    q = pd.concat(frames)
    q = q[q["observed_at_utc"] <= pd.Timestamp(run_at)]
    for gid, g in q.groupby("nflverse_game_id"):
        g = g.sort_values("observed_at_utc")
        first, last = g.iloc[0], g.iloc[-1]
        out[gid] = {
            "home_spread_first_capture_of_week": -float(first["line"]),
            "first_capture_at_utc": iso(first["observed_at_utc"]),
            "home_spread_latest": -float(last["line"]),
            "latest_capture_at_utc": iso(last["observed_at_utc"]),
            "book": last["bookmaker_key"],
            "move_toward_home": float(first["line"] - last["line"]),
            "captures": len(g),
        }
    return out


def injuries_block(teams, run_at):
    d = stamp_dirs(ROOT / "data" / "raw" / "nflverse_injuries", run_at)[-1]
    x = pd.read_parquet(d / "injuries.parquet")
    x = x[(x["season"] == SEASON) & x["report_status"].isin(INJURY_STATUSES)]
    wk = int(x["week"].max())
    x = x[x["week"] == wk]
    out = {}
    for t in teams:
        r = x[x["team"] == t]
        out[t] = {
            "source": "nflverse_injuries",
            "snapshot_at_utc": iso(pd.Timestamp(dir_time(d.name))),
            "report_week": wk,
            "rows": [
                {
                    "player": a["full_name"],
                    "position": a["position"],
                    "status": a["report_status"],
                    "injury": a["report_primary_injury"],
                    "practice_status": a["practice_status"],
                    "date_modified_utc": None
                    if pd.isna(a["date_modified"])
                    else iso(a["date_modified"]),
                }
                for _, a in r.iterrows()
            ],
        }
    return out


def headlines_block(names, run_at, since):
    d = stamp_dirs(ROOT / "data" / "raw" / "injury_news", run_at)[-1]
    c = pd.read_parquet(d / "current.parquet")
    c = c[
        c["injury_relevant"]
        & (c["lastmod"] <= pd.Timestamp(run_at))
        & (c["lastmod"] >= pd.Timestamp(since))
    ]
    out = {}
    for t, nick in names.items():
        key = nick.lower().split()[-1]
        r = c[c["slug"].str.contains(key, regex=False)].sort_values("lastmod")
        out[t] = [
            {
                "headline": a["headline_guess"],
                "url": a["url"],
                "published_at_utc": iso(a["lastmod"]),
                "keywords": a["matched_keywords"],
            }
            for _, a in r.iterrows()
        ]
    return out, iso(pd.Timestamp(dir_time(d.name)))


def inactives_block(week, run_at):
    rows = []
    for d in stamp_dirs(ROOT / "data" / "players" / "inactives", run_at):
        x = pd.read_parquet(d / "inactives.parquet")
        x = x[(x["season"] == SEASON) & (x["week"] == week)]
        if len(x):
            rows.append(x)
    if not rows:
        return {}
    x = (
        pd.concat(rows)
        .sort_values("captured_at_utc")
        .drop_duplicates(["game_id", "team", "player_name"], keep="last")
    )
    out = {}
    for gid, g in x.groupby("game_id"):
        out[gid] = [
            {
                "team": a["team"],
                "player": a["player_name"],
                "position": a["position"],
                "status": a["status"],
                "captured_at_utc": a["captured_at_utc"],
            }
            for _, a in g.iterrows()
        ]
    return out


def weather_block(week, run_at):
    base = ROOT / "data" / "raw" / "paired_wind_forecast"
    best = {}
    for p in sorted(base.glob("*/*/*/forecasts.parquet")):
        f = pd.read_parquet(p)
        f = f[(f["week"] == week) & (f["fetch_status"] == "ok")]
        f = f[pd.to_datetime(f["captured_at_utc"], utc=True) <= pd.Timestamp(run_at)]
        for _, a in f.iterrows():
            k = (a["game_id"], a["product_id"])
            if k not in best or a["captured_at_utc"] > best[k]["captured_at_utc"]:
                best[k] = a
    out = {}
    for (gid, prod), a in best.items():
        out.setdefault(gid, []).append(
            {
                "stadium": a["stadium"],
                "roof": a["roof"],
                "model": prod,
                "forecast_valid_utc": a["forecast_valid_utc"],
                "forecast_wind_mph": float(a["forecast_wind_mph"]),
                "captured_at_utc": a["captured_at_utc"],
            }
        )
    return out


def beat_block(prev, teams, run_at):
    b = pd.read_parquet(
        ROOT / "data" / "processed" / "beat_reporter_posts.parquet",
        columns=["team", "author_handle", "text", "created_at", "season"],
    )
    b = b[(b["season"] == SEASON) & (b["created_at"] <= pd.Timestamp(run_at))]
    out = {}
    for t in teams:
        since = prev.get(t)
        r = b[b["team"] == t]
        if since is not None:
            r = r[r["created_at"] > pd.Timestamp(since)]
        r = r.sort_values("created_at")
        out[t] = {
            "since_utc": None if since is None else iso(pd.Timestamp(since)),
            "n_posts": len(r),
            "posts": [
                {"at_utc": iso(a["created_at"]), "author": a["author_handle"], "text": a["text"]}
                for _, a in r.iterrows()
            ],
        }
    return out


def rest_block(prev, home, away, kick):
    def days(t):
        p = prev.get(t)
        return None if p is None else round((kick - p).total_seconds() / 86400, 2)

    return {"home_days_since_last_game": days(home), "away_days_since_last_game": days(away)}


def team_names(run_at):
    names = {}
    for d in stamp_dirs(ROOT / "data" / "market" / "raw", run_at)[-3:]:
        q = pd.read_parquet(
            d / "quotes.parquet",
            columns=["home_team", "home_team_name", "away_team", "away_team_name"],
        )
        for _, a in q.drop_duplicates().iterrows():
            names[a["home_team"]] = a["home_team_name"]
            names[a["away_team"]] = a["away_team_name"]
    return names


def build_packet(week, run_at):
    f, spl = load_splash(week, run_at)
    anchor = datetime.fromisoformat(spl["captured_at_et"])
    games = spl["games"]
    prev = previous_kickoffs(week, run_at)
    week_start = (
        anchor.astimezone(ET).replace(hour=0, minute=0, second=0, microsecond=0).astimezone(UTC)
    )
    teams = sorted({g["home"] for g in games} | {g["away"] for g in games})
    mk = market_block(run_at, week_start)
    inj = injuries_block(teams, run_at)
    nicks = team_names(run_at)
    hl, hl_at = headlines_block(
        {t: nicks[t] for t in teams}, run_at, week_start - timedelta(days=7)
    )
    ina = inactives_block(week, run_at)
    wx = weather_block(week, run_at)
    beat = beat_block(prev, teams, run_at)
    packet_games = []
    for g in games:
        kick = datetime.fromisoformat(g["kickoff_et"])
        gid = g["game_id"]
        h, a = g["home"], g["away"]
        packet_games.append(
            {
                "game_id": gid,
                "away": a,
                "home": h,
                "kickoff_et": g["kickoff_et"],
                "deadline_et": deadline_for(kick, anchor).isoformat(),
                "pool_line": {
                    "home_spread": g["home_spread"],
                    "source": f.name,
                    "captured_at_et": spl["captured_at_et"],
                },
                "market": mk.get(gid),
                "injuries": {h: inj[h], a: inj[a]},
                "injury_headlines": {"snapshot_at_utc": hl_at, h: hl[h], a: hl[a]},
                "inactives": ina.get(gid, []),
                "beat_posts": {h: beat[h], a: beat[a]},
                "weather": wx.get(gid, []),
                "rest": rest_block(prev, h, a, kick),
            }
        )
    packet = {
        "week": week,
        "season": SEASON,
        "run_at_utc": run_at.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "convention": "home_spread positive means the home team is favored by that many points",
        "games": packet_games,
    }
    assert_before(packet, run_at)
    return packet


def assert_before(obj, run_at):
    pat = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}")
    suffixes = ("at_utc", "captured_at_et", "date_modified_utc", "since_utc", "captured_at")
    seen = [0]

    def walk(o, k=""):
        if isinstance(o, dict):
            for kk, v in o.items():
                walk(v, kk)
        elif isinstance(o, list):
            for v in o:
                walk(v, k)
        elif isinstance(o, str) and k.endswith(suffixes) and k != "run_at_utc" and pat.match(o):
            t = datetime.fromisoformat(o.replace("Z", "+00:00"))
            if t.tzinfo is None:
                t = t.replace(tzinfo=UTC)
            assert t <= run_at, (k, o)
            seen[0] += 1

    walk(obj)
    assert seen[0] > 0


def extract_json(text):
    s = text.strip()
    m = re.search(r"\{.*\}", s, re.S)
    return json.loads(m.group(0) if m else s)


def validate(parsed, game_ids):
    rows = parsed["games"]
    seen = [r["game_id"] for r in rows]
    assert sorted(seen) == sorted(game_ids), "game set mismatch"
    for r in rows:
        p = r["home_cover_probability"]
        assert isinstance(p, (int, float)) and 0 < p < 1, f"bad probability {r['game_id']}"
        assert isinstance(r["mechanism"], str) and r["mechanism"].strip()
        assert isinstance(r["key_facts"], list) and isinstance(r["sources"], list)
    return rows


def call_claude(prompt_text, packet_text, cwd):
    exe = shutil.which("claude")
    cmd = [
        exe,
        "-p",
        "--output-format",
        "json",
        "--model",
        MODEL,
        "--tools",
        TOOLS,
        "--allowedTools",
        TOOLS,
        "--no-session-persistence",
        "--setting-sources",
        "local",
        "--permission-mode",
        "dontAsk",
    ]
    p = subprocess.run(
        cmd,
        input=prompt_text + "\n\nPACKET\n" + packet_text,
        capture_output=True,
        text=True,
        encoding="utf-8",
        cwd=cwd,
        creationflags=subprocess.CREATE_NO_WINDOW,
        timeout=3000,
    )
    if p.returncode != 0:
        raise SystemExit(f"claude failed rc={p.returncode}: {p.stderr[-800:]} {p.stdout[-800:]}")
    env = json.loads(p.stdout)
    return env, env.get("result", "")


def cmd_enroll(args):
    ART.mkdir(parents=True, exist_ok=True)
    FROZEN.write_text(
        json.dumps(
            {
                "prompt_sha256": prompt_sha(),
                "model": MODEL,
                "tools": TOOLS,
                "packet_fields": PACKET_FIELDS,
                "enrolled_at_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
            },
            indent=2,
        )
    )
    print(FROZEN.read_text())


def cmd_run(args):
    fz = json.loads(FROZEN.read_text())
    if fz["prompt_sha256"] != prompt_sha() or fz["model"] != MODEL:
        raise SystemExit("prompt or model differs from frozen enrollment")
    run_at = datetime.now(UTC).replace(microsecond=0)
    packet = build_packet(args.week, run_at)
    packet_text = json.dumps(packet, ensure_ascii=False, sort_keys=True)
    prompt_text = PROMPT.read_text(encoding="utf-8")
    ids = [g["game_id"] for g in packet["games"]]
    cwd = ART / "cwd"
    cwd.mkdir(parents=True, exist_ok=True)
    attempts = []
    rows = None
    for attempt in (1, 2):
        env, raw = call_claude(prompt_text, packet_text, str(cwd))
        rec = {"attempt": attempt, "raw_response": raw, "envelope": env}
        try:
            rows = validate(extract_json(raw), ids)
            rec["valid"] = True
        except Exception as e:
            rec["valid"] = False
            rec["error"] = repr(e)
        attempts.append(rec)
        if rows is not None:
            break
    recorded = datetime.now(UTC)
    dl = {g["game_id"]: datetime.fromisoformat(g["deadline_et"]) for g in packet["games"]}
    kept, dropped = [], []
    for r in rows or []:
        (kept if recorded < dl[r["game_id"]] else dropped).append(r)
    out = {
        "week": args.week,
        "view": args.view,
        "packet": packet,
        "packet_sha256": sha256_bytes(packet_text.encode("utf-8")),
        "prompt_sha256": prompt_sha(),
        "model": MODEL,
        "tools": TOOLS,
        "attempts": attempts,
        "retried": len(attempts) > 1,
        "parsed_rows": kept,
        "dropped_past_deadline": [r["game_id"] for r in dropped],
        "dropped_rows": dropped,
        "deadlines_et": {k: v.isoformat() for k, v in dl.items()},
        "run_at_utc": packet["run_at_utc"],
        "recorded_at_utc": recorded.strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    runs = ART / "runs"
    runs.mkdir(parents=True, exist_ok=True)
    path = (
        runs
        / f"{SEASON}-week-{args.week:02d}-{args.view}-{recorded.strftime('%Y%m%dT%H%M%SZ')}.json"
    )
    path.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    ps = [r["home_cover_probability"] for r in kept]
    print(path)
    print(
        len(kept),
        "rows, dropped",
        len(dropped),
        "range",
        (min(ps), max(ps)) if ps else None,
        "retried",
        out["retried"],
    )
    if rows is None:
        raise SystemExit("no valid output after retry")


def cmd_packet(args):
    run_at = datetime.now(UTC).replace(microsecond=0)
    p = build_packet(args.week, run_at)
    s = json.dumps(p, ensure_ascii=False)
    print(len(s), "chars", len(p["games"]), "games")
    for g in p["games"]:
        print(
            g["game_id"],
            {t: v["n_posts"] for t, v in g["beat_posts"].items()},
            bool(g["market"]),
            len(g["weather"]),
            len(g["inactives"]),
        )


def main():
    ap = argparse.ArgumentParser(description="LEAD-99 LLM read runner")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("enroll").set_defaults(fn=cmd_enroll)
    for n, fn in (("run", cmd_run), ("packet", cmd_packet)):
        s = sub.add_parser(n)
        s.add_argument(
            "--week",
            type=int,
            help="pool week; defaults to the captured week with a game still to kick off",
        )
        if n == "run":
            s.add_argument("--view", choices=["tuesday", "deadline"], required=True)
        s.set_defaults(fn=fn)
    a = ap.parse_args()
    if getattr(a, "week", 0) is None:
        a.week = current_week(datetime.now(UTC).replace(microsecond=0))
    a.fn(a)


if __name__ == "__main__":
    main()
