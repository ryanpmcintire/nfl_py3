import glob
import hashlib
import json
import sys
import time
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

SNAP_GLOB = str(ROOT / "data" / "players" / "raw" / "*" / "snap_counts.parquet")
SCHED_GLOB = str(ROOT / "data" / "raw" / "*" / "schedules.parquet")
OUT_RAW = ROOT / "data" / "raw" / "wiki_pageviews"
OUT_PROC = ROOT / "data" / "processed" / "wiki_pageviews"
UA = "nfl-ats-research/1.0 (https://github.com/ryanpmcintire/nfl_py3; personal non-commercial research)"
FIRST_SEASON = 2015
PAGEVIEW_START = "20150701"
PAGEVIEW_END = "20251231"
SKILL_POS = {"RB": "running back", "HB": "running back", "FB": "fullback", "WR": "wide receiver", "TE": "tight end"}
DEF_POS = {
    "LB": "linebacker", "OLB": "linebacker", "ILB": "linebacker", "MLB": "linebacker",
    "DE": "defensive end", "DT": "defensive tackle", "NT": "defensive tackle", "DL": "defensive lineman",
    "CB": "cornerback", "FS": "safety", "SS": "safety", "S": "safety", "DB": "defensive back",
}
SKILL_N = 3
DEF_N = 3
REQUEST_GAP = 0.12


def http_json(url):
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    for attempt in range(5):
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                return r.status, r.read()
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return 404, b""
            time.sleep(2.0 * (attempt + 1))
        except Exception:
            time.sleep(2.0 * (attempt + 1))
    return 0, b""


def starters():
    snaps = pd.read_parquet(sorted(glob.glob(SNAP_GLOB))[-1])
    snaps = snaps.loc[snaps["season"].ge(FIRST_SEASON)].copy()
    snaps["offense_snaps"] = pd.to_numeric(snaps["offense_snaps"], errors="coerce").fillna(0.0)
    snaps["defense_snaps"] = pd.to_numeric(snaps["defense_snaps"], errors="coerce").fillna(0.0)
    qb = snaps.loc[snaps["position"].eq("QB")].sort_values(["game_id", "team", "offense_snaps", "player"], ascending=[True, True, False, True])
    qb = qb.groupby(["game_id", "team"]).head(1).assign(role="qb")
    sk = snaps.loc[snaps["position"].isin(SKILL_POS)].sort_values(["game_id", "team", "offense_snaps", "player"], ascending=[True, True, False, True])
    sk = sk.groupby(["game_id", "team"]).head(SKILL_N).assign(role="skill")
    df = snaps.loc[snaps["position"].isin(DEF_POS)].sort_values(["game_id", "team", "defense_snaps", "player"], ascending=[True, True, False, True])
    df = df.groupby(["game_id", "team"]).head(DEF_N).assign(role="defense")
    out = pd.concat([qb, sk, df])
    return out[["game_id", "season", "week", "game_type", "team", "player", "pfr_player_id", "position", "role"]]


def candidates(name, pos):
    word = SKILL_POS.get(pos) or DEF_POS.get(pos) or ("quarterback" if pos == "QB" else None)
    c = []
    if word:
        c.append(f"{name} ({word})")
    c += [f"{name} (American football)", name]
    return c


def resolve(players):
    cache_path = OUT_PROC / "title_resolution.json"
    cache = json.loads(cache_path.read_text()) if cache_path.exists() else {}
    todo = [(p, pos) for (p, pos) in players if f"{p}|{pos}" not in cache]
    cand_map = {}
    for p, pos in todo:
        for t in candidates(p, pos):
            cand_map[t] = None
    titles = sorted(cand_map)
    info = {}
    for i in range(0, len(titles), 50):
        chunk = titles[i : i + 50]
        q = urllib.parse.urlencode({"action": "query", "format": "json", "redirects": 1, "prop": "pageprops", "ppprop": "wikibase-shortdesc|disambiguation", "titles": "|".join(chunk)})
        st, body = http_json("https://en.wikipedia.org/w/api.php?" + q)
        time.sleep(REQUEST_GAP)
        if st != 200:
            continue
        j = json.loads(body)["query"]
        norm = {n["from"]: n["to"] for n in j.get("normalized", [])}
        red = {r["from"]: r["to"] for r in j.get("redirects", [])}
        pages = {p["title"]: p for p in j["pages"].values()}
        for t in chunk:
            t1 = norm.get(t, t)
            t2 = red.get(t1, t1)
            pg = pages.get(t2)
            if pg is None or "missing" in pg:
                continue
            pp = pg.get("pageprops", {})
            info[t] = (t2, pp.get("wikibase-shortdesc", ""), "disambiguation" in pp)
    for p, pos in todo:
        pick = None
        for t in candidates(p, pos):
            r = info.get(t)
            if r and not r[2] and "football" in r[1].lower() and "canadian" not in r[1].lower():
                pick = r[0]
                break
        cache[f"{p}|{pos}"] = pick
    cache_path.write_text(json.dumps(cache, indent=1))
    return cache


def fetch(titles, run_id):
    run_dir = OUT_RAW / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest = []
    for k, t in enumerate(titles):
        path = run_dir / (hashlib.sha1(t.encode()).hexdigest()[:16] + ".json")
        if path.exists():
            body = path.read_bytes()
            st = 200
        else:
            art = urllib.parse.quote(t.replace(" ", "_"), safe="")
            url = f"https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/en.wikipedia/all-access/user/{art}/daily/{PAGEVIEW_START}/{PAGEVIEW_END}"
            st, body = http_json(url)
            time.sleep(REQUEST_GAP)
            if st == 200:
                path.write_bytes(body)
        manifest.append({"title": t, "status": st, "file": path.name if st == 200 else None, "sha256": hashlib.sha256(body).hexdigest() if st == 200 else None, "bytes": len(body)})
        if k % 200 == 0:
            print("fetched", k, "of", len(titles), flush=True)
    (run_dir / "manifest.json").write_text(json.dumps({"run_id": run_id, "user_agent": UA, "start": PAGEVIEW_START, "end": PAGEVIEW_END, "retrieved_utc": datetime.now(timezone.utc).isoformat(), "articles": manifest}, indent=1))
    return run_dir, manifest


def main():
    OUT_PROC.mkdir(parents=True, exist_ok=True)
    st = starters()
    st.to_parquet(OUT_PROC / "starters.parquet")
    players = sorted(set(zip(st["player"], st["position"], strict=True)))
    print("starter rows", len(st), "unique player-positions", len(players), flush=True)
    cache = resolve(players)
    got = sum(1 for v in cache.values() if v)
    print("resolved", got, "of", len(cache), flush=True)
    titles = sorted({v for v in cache.values() if v})
    print("unique articles", len(titles), flush=True)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir, manifest = fetch(titles, run_id)
    print("run", run_id, "ok", sum(1 for m in manifest if m["status"] == 200), "missing", sum(1 for m in manifest if m["status"] != 200), flush=True)
    (OUT_PROC / "latest_run.txt").write_text(run_id)


main()
