import argparse
import hashlib
import json
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "beat_reporters"
HANDLES = ROOT / "data" / "processed" / "beat_reporter_handles.csv"
POSTS = ROOT / "data" / "processed" / "beat_reporter_posts.parquet"
LEADTIME = ROOT / "data" / "processed" / "beat_reporter_leadtime_2025.csv"
API = "https://public.api.bsky.app/xrpc/"
UA = "nfl-ats-research/0.1 (public unauthenticated read-only, rate limited)"
SLEEP = 0.3
LEXICON_VERSION = "u9-2026-10-01"

SEED = """ARI|daniel-guerrero.bsky.social|St. Louis Post-Dispatch
ARI|tdrake4sports.bsky.social|Arizona Sports
ARI|theomackie.bsky.social|Arizona Republic
ARI|kyleodegard.bsky.social|independent
ARI|doughaller.bsky.social|The Athletic
ATL|marcraimondi.bsky.social|ESPN
ATL|falcoholickevin.bsky.social|The Falcoholic
ATL|thefalcoholic.bsky.social|The Falcoholic
ATL|thediaryoff.bsky.social|independent
BAL|jamisonhensley.bsky.social|ESPN
BAL|giana-jade.bsky.social|Baltimore Banner
BUF|joebuscaglia.bsky.social|The Athletic
BUF|mattparrino.bsky.social|Syracuse.com
BUF|salsports.bsky.social|WGR 550
BUF|ryantalbotbills.bsky.social|NYUP Syracuse.com
BUF|nicksabatognn.bsky.social|Niagara Gazette
CAR|mikekayefootball.bsky.social|ESPN
CAR|sheenaquick.bsky.social|1340 AM Fox Sports
CHI|danwiederer.bsky.social|The Athletic
CHI|philthompsontrib.bsky.social|Chicago Tribune
CHI|seanhammond.bsky.social|Chicago Tribune
CHI|bradbiggs.bsky.social|Chicago Tribune
CHI|adamjahns.bsky.social|CHGO
CHI|kfishbain.bsky.social|The Athletic
CHI|mdwojak94.bsky.social|Shaw Local
CIN|pbrennanenq.bsky.social|Cincinnati Enquirer
CIN|laurelpfahler.bsky.social|Dayton Daily News
CIN|bbaby41.bsky.social|ESPN
CIN|mtoscano1.bsky.social|Bengals Talk SI
CLE|ashleybastock42.bsky.social|Cleveland.com
CLE|spencito.bsky.social|SI Browns
CLE|marykaycabot.bsky.social|Cleveland.com
CLE|ceasterlingabj.bsky.social|Akron Beacon Journal
CLE|tonygrossi.bsky.social|The Land On Demand
DAL|joejhoyt.bsky.social|Dallas Morning News
DAL|jctsports.bsky.social|LoneStarLive
DAL|calvinwatkins.bsky.social|Dallas Morning News
DAL|saadyousuf126.bsky.social|The Athletic
DEN|parkerjgabriel.bsky.social|Denver Post
DEN|nickkosmider.bsky.social|The Athletic
DEN|codyroarknfl.bsky.social|Mile High Sports
DEN|zacstevens.bsky.social|DNVR
DEN|henrychisholm.bsky.social|DNVR
DEN|troyrenck.bsky.social|Denver Post
DET|detroitfootball.net|Detroit Football Network
DET|davebirkett.bsky.social|Detroit Free Press
DET|nolanbianchi.bsky.social|Detroit News
DET|korywoods.bsky.social|MLive
DET|coltonpouncy.bsky.social|The Athletic
GB|mattschneidman.bsky.social|The Athletic
GB|tspoon62.bsky.social|Milwaukee Journal Sentinel
GB|zachjacobson.bsky.social|PackerDispatch
GB|jacobwestendorf.bsky.social|SI Packers
GB|billhubernfl.bsky.social|SI Packers
GB|zachkruse.bsky.social|Packers Wire
HOU|jonmalexander.bsky.social|Houston Chronicle
HOU|aaronwilsonnfl.bsky.social|KPRC 2
IND|stephenholder-nfl.bsky.social|ESPN
IND|joelaerickson.bsky.social|IndyStar
IND|zakkeefer.bsky.social|The Athletic
IND|jakearthurnfl.bsky.social|Roundtable Sports
JAX|demetrius.bsky.social|Florida Times-Union
KC|jessenewell.bsky.social|The Athletic
KC|bynatetaylor.bsky.social|ESPN
KC|edeastonjr.bsky.social|independent
LV|paulhgutierrez.bsky.social|Raiders.com
LV|raidersbeat.bsky.social|Raiders Beat
LAC|realframirez.bsky.social|Sporting Tribune
LAC|elliottteaford.bsky.social|OC Register
LA|adamgrosbard.bsky.social|LA Daily News
LA|nateatkins.bsky.social|The Athletic
MIA|ml-j.bsky.social|ESPN
MIA|schadjoe.bsky.social|Palm Beach Post
MIA|thefinsider.bsky.social|The Finsider
MIN|matthewcoller.bsky.social|Purple Insider
MIN|aleclewis.bsky.social|The Athletic
MIN|bengoessling.bsky.social|Star Tribune
MIN|therealforno.bsky.social|A to Z Sports
MIN|danemizutani.bsky.social|Pioneer Press
MIN|kevinseifert.bsky.social|ESPN
NE|mikereiss.bsky.social|ESPN
NE|andrewcallahan.bsky.social|Boston Herald
NE|dougkyed.bsky.social|Boston Herald
NE|tkyles39.bsky.social|Patriots CLNS
NE|mark-daniels.bsky.social|MassLive
NE|chadgraff.bsky.social|The Athletic
NE|sophieewellerr.bsky.social|A to Z Sports
NO|matthewparas.bsky.social|Times-Picayune Advocate
NO|rossjacksonnola.bsky.social|LouisianaSports.Net
NO|nickunderhill.bsky.social|NewOrleans.Football
NO|zewing.bsky.social|Times-Picayune Advocate
NO|kat-terrell.bsky.social|independent
NYG|evanbarnes.bsky.social|Newsday
NYG|jordanraanan.bsky.social|ESPN
NYG|plonnfl.bsky.social|NY Daily News
NYG|charlottecrrll.bsky.social|The Athletic
NYG|edvalentine.bsky.social|BigBlueView
NYJ|antwanstaley.bsky.social|NY Daily News
NYJ|zackblatt.bsky.social|The Athletic
PHI|ej-smith.bsky.social|AllPHLY
PHI|jeff-mclane.bsky.social|Philadelphia Inquirer
PHI|timmcmanus42.bsky.social|ESPN
PHI|eliotshorrparks.bsky.social|94WIP
PHI|oliviareiner.bsky.social|Philadelphia Inquirer
PHI|bkubena.bsky.social|The Athletic
PHI|caydensteele.bsky.social|NJ.com
PHI|jeff-neiburg.bsky.social|Philadelphia Inquirer
PHI|zberm.bsky.social|The Athletic
PHI|dzangaro.bsky.social|NBC Sports Philadelphia
PHI|jimmykempski.bsky.social|PhillyVoice
PIT|nickfarabaugh.bsky.social|PennLive
PIT|mikedefabo.bsky.social|The Athletic
PIT|asaunderspgh.bsky.social|Pittsburgh Steelers Now
PIT|chrishalicke.bsky.social|DK Pittsburgh Sports
PIT|bepryor.bsky.social|ESPN
PIT|teresava.bsky.social|Steelers.com
SF|caminman.bsky.social|Bay Area News Group
SF|mattbarrows.bsky.social|The Athletic
SF|victafur.bsky.social|The Athletic
SEA|johnpboyle.bsky.social|Seahawks.com
SEA|mookiealexander.bsky.social|Field Gulls
SEA|fieldgulls.bsky.social|Field Gulls
SEA|hawkblogger.com|Hawk Blogger
TB|jennalaine.bsky.social|ESPN
TB|tylervasallo.bsky.social|RBLR Bucs
TB|rblrbucs.rblrsports.com|RBLR Bucs
TEN|teresamwalker.bsky.social|Associated Press
WAS|tashanreed.bsky.social|Washington Post
WAS|benstandig.bsky.social|independent
WAS|dwharrison.bsky.social|Locked On Commanders
WAS|nickijhabvala.bsky.social|Washington Post
WAS|lakelewisjr.bsky.social|independent"""

NICK = {
    "ARI": ["cardinals"], "ATL": ["falcons"], "BAL": ["ravens"], "BUF": ["bills"],
    "CAR": ["panthers"], "CHI": ["bears"], "CIN": ["bengals"], "CLE": ["browns"],
    "DAL": ["cowboys"], "DEN": ["broncos"], "DET": ["lions"], "GB": ["packers"],
    "HOU": ["texans"], "IND": ["colts"], "JAX": ["jaguars", "jags"], "KC": ["chiefs"],
    "LV": ["raiders"], "LAC": ["chargers", "bolts"], "LA": ["rams"], "MIA": ["dolphins"],
    "MIN": ["vikings", "vikes"], "NE": ["patriots", "pats"], "NO": ["saints"], "NYG": ["giants"],
    "NYJ": ["jets"], "PHI": ["eagles"], "PIT": ["steelers"], "SF": ["49ers", "niners"],
    "SEA": ["seahawks"], "TB": ["buccaneers", "bucs"], "TEN": ["titans"],
    "WAS": ["commanders"],
}
NICK_TO_TEAM = {n: t for t, ns in NICK.items() for n in ns}
TEAM_ALIAS = {"LAR": "LA", "LVR": "LV", "OAK": "LV", "SD": "LAC", "STL": "LA", "WSH": "WAS",
              "JAC": "JAX", "ARZ": "ARI", "BLT": "BAL", "CLV": "CLE", "HST": "HOU", "SL": "LA"}

LEXICON = {
    "did_not_practice": r"did not practice|didn'?t practice|\bdnp\b|not practicing|sat out (?:of )?practice|no practice|did not participate|didn'?t participate|missed practice|not at practice",
    "limited": r"\blimited (?:in|at|during|participation|today|with|for)\b|\bwas limited\b|\bis limited\b|\blp\b|\bon a limited\b|\blimited participant",
    "full": r"full participant|practiced fully|full practice|fully participated|\bfp\b|full participation|\bpracticed in full\b|full-go|full go",
    "absent": r"\babsent\b|not (?:present )?at practice|not spotted|not seen|did not see|didn'?t see|not on the field|missing from practice|not in attendance|not out there",
    "boot_brace": r"walking boot|\bboot\b|\bbrace\b|\bcrutches\b|\bsling\b|knee scooter|\bscooter\b",
    "first_team_reps": r"first[- ]team|with the (?:1s|ones|starters|first)|\b1s\b|first[- ]string|first unit|ran with the (?:ones|starters)|starting unit",
    "ruled_out": r"ruled out|will not play|won'?t play|\bis out\b|\bare out\b|\bwill be out\b|out for (?:the )?(?:game|season|week|year)|\binactive\b|injured reserve|\bplaced on ir\b|season-ending|won'?t suit up|\bout (?:vs|against|sunday|monday|thursday|saturday|friday)\b",
    "questionable": r"questionable|doubtful|game[- ]time decision|\bgtd\b|day[- ]to[- ]day",
    "expected_to_play": r"expected to play|expected to start|expected back|plans to play|\bwill play\b|should play|good to go|expected to suit up|cleared to play|returned to practice|returns to practice|back at practice|back in practice|is a go\b",
}
LEX_RE = {k: re.compile(v, re.I) for k, v in LEXICON.items()}
OUT_CLASSES = ("did_not_practice", "limited", "ruled_out", "absent", "boot_brace")
TOKEN = re.compile(r"[A-Za-z][A-Za-z'.\-]*")


def season_of(ts):
    return ts.year if ts.month >= 3 else ts.year - 1


def http_get(path, params, tries=6):
    for i in range(tries):
        try:
            r = requests.get(API + path, params=params, headers={"User-Agent": UA}, timeout=40)
        except requests.RequestException:
            time.sleep(2 * (i + 1))
            continue
        if r.status_code == 200:
            return r.json()
        if r.status_code in (429, 500, 502, 503, 504):
            try:
                wait = float(r.headers.get("Retry-After", 5 * (i + 1)))
            except ValueError:
                wait = 5 * (i + 1)
            time.sleep(min(wait, 120))
            continue
        return None
    return None


def run_id():
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def seed_rows():
    return [line.split("|") for line in SEED.split("\n")]


def fetch_profiles(handles):
    out = {}
    hs = list(handles)
    for i in range(0, len(hs), 25):
        j = http_get("app.bsky.actor.getProfiles", [("actors", h) for h in hs[i:i + 25]])
        time.sleep(SLEEP)
        for p in (j or {}).get("profiles", []):
            out[p["handle"].lower()] = p
    return out


def fetch_feed(handle, out_path, stop_uris=None, max_pages=1500):
    cursor = None
    n = 0
    stop_uris = stop_uris or set()
    with open(out_path, "w", encoding="utf-8") as f:
        for _ in range(max_pages):
            params = {"actor": handle, "limit": 100, "filter": "posts_with_replies"}
            if cursor:
                params["cursor"] = cursor
            j = http_get("app.bsky.feed.getAuthorFeed", params)
            time.sleep(SLEEP)
            if not j:
                break
            feed = j.get("feed", [])
            hit = False
            for it in feed:
                if not it.get("reason") and it["post"]["uri"] in stop_uris:
                    hit = True
                f.write(json.dumps(it, ensure_ascii=False) + "\n")
                n += 1
            cursor = j.get("cursor")
            if not cursor or not feed or hit:
                break
    return n


def write_manifest(rdir, mode, extra):
    files = {}
    for p in sorted(rdir.glob("*.jsonl")):
        files[p.name] = {"sha256": sha256(p), "bytes": p.stat().st_size}
    m = {"run_id": rdir.name, "mode": mode, "created_utc": datetime.now(timezone.utc).isoformat(),
         "endpoint": API + "app.bsky.feed.getAuthorFeed", "files": files, **extra}
    (rdir / "manifest.json").write_text(json.dumps(m, indent=1), encoding="utf-8")


def latest_roster_path():
    dirs = sorted((ROOT / "data" / "players" / "raw").glob("2*Z"))
    for d in reversed(dirs):
        p = d / "weekly_rosters.parquet"
        if p.exists():
            return p
    raise SystemExit("no weekly_rosters on disk")


def norm_name(s):
    s = re.sub(r"[^a-z' ]", " ", s.lower().replace(".", ""))
    s = re.sub(r"\b(jr|sr|ii|iii|iv|v)\b", " ", s)
    return " ".join(s.split())


def load_rosters():
    r = pd.read_parquet(latest_roster_path(), columns=["season", "team", "full_name", "gsis_id", "pfr_id", "position"])
    r = r.dropna(subset=["full_name", "gsis_id"]).drop_duplicates(["season", "team", "gsis_id"])
    r["team"] = r["team"].replace(TEAM_ALIAS)
    r["nn"] = r["full_name"].map(norm_name)
    r["last"] = r["nn"].map(lambda s: s.split()[-1] if s else "")
    idx = {}
    for s, g in r.groupby("season"):
        full = {}
        for row in g.itertuples():
            full.setdefault(row.nn, []).append((row.gsis_id, row.team))
        lastteam = {}
        for (t, last), gg in g.groupby(["team", "last"]):
            if len(gg) == 1 and len(last) >= 4:
                lastteam[(t, last)] = gg.iloc[0]["gsis_id"]
        idx[int(s)] = (full, lastteam)
    return idx, r


def tag_post(text, team, season, idx):
    full, lastteam = idx.get(season, ({}, {}))
    toks = TOKEN.findall(text)
    low = [norm_name(t) for t in toks]
    found = {}
    for i in range(len(low) - 1):
        key = (low[i] + " " + low[i + 1]).strip()
        if key in full:
            for g, t in full[key]:
                if t == team or len(full[key]) == 1:
                    found[g] = "full"
    for tk, lw in zip(toks, low):
        if tk[0].isupper() and len(lw) >= 4 and (team, lw) in lastteam:
            found.setdefault(lastteam[(team, lw)], "last")
    words = {w.lower().strip(".'") for w in toks}
    teams = sorted({NICK_TO_TEAM[w] for w in words if w in NICK_TO_TEAM})
    classes = [k for k, rx in LEX_RE.items() if rx.search(text)]
    return found, teams, classes


def build_posts():
    idx, _ = load_rosters()
    teams = {r[1].lower(): r[0] for r in seed_rows()}
    rows = {}
    for rdir in sorted(RAW.glob("2*Z")):
        for p in sorted(rdir.glob("*.jsonl")):
            handle = p.stem
            with open(p, encoding="utf-8") as f:
                for line in f:
                    it = json.loads(line)
                    po = it["post"]
                    rec = po.get("record", {})
                    is_rp = bool(it.get("reason"))
                    key = (handle, po["uri"], is_rp)
                    if key in rows:
                        continue
                    rows[key] = (handle, teams.get(handle.lower()), po["uri"], po["author"]["handle"],
                                 rec.get("createdAt"), po.get("indexedAt"), rec.get("text", ""),
                                 "reply" in rec, is_rp, rdir.name)
    cols = ["handle", "team", "uri", "author_handle", "created_raw", "indexed_raw", "text", "is_reply", "is_repost", "run_id"]
    df = pd.DataFrame(list(rows.values()), columns=cols)
    df["created_at"] = pd.to_datetime(df["created_raw"], utc=True, errors="coerce", format="ISO8601")
    df["indexed_at"] = pd.to_datetime(df["indexed_raw"], utc=True, errors="coerce", format="ISO8601")
    df = df.drop(columns=["created_raw", "indexed_raw"]).dropna(subset=["created_at"])
    df = df[df.created_at >= pd.Timestamp("2022-01-01", tz="UTC")]
    df["season"] = df["created_at"].map(season_of)
    pl, pm, tm, cl = [], [], [], []
    for r in df.itertuples():
        if r.is_repost:
            pl.append([]); pm.append([]); tm.append([]); cl.append([])
            continue
        f, t, c = tag_post(r.text, r.team, r.season, idx)
        ks = sorted(f)
        pl.append(ks); pm.append([f[g] for g in ks]); tm.append(t); cl.append(c)
    df["player_gsis"] = pl
    df["player_match"] = pm
    df["mentioned_teams"] = tm
    df["avail_classes"] = cl
    df["lexicon_version"] = LEXICON_VERSION
    df = df.sort_values(["created_at", "uri"]).reset_index(drop=True)
    POSTS.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(POSTS, index=False)
    return df


def finish_handles(profs):
    df = None
    if POSTS.exists():
        df = pd.read_parquet(POSTS, columns=["handle", "created_at", "is_repost", "mentioned_teams", "player_gsis"])
    rows = []
    for t, h, o in seed_rows():
        p = profs.get(h.lower(), {})
        n = rel = 0
        first = None
        if df is not None:
            g = df[(df.handle == h) & (~df.is_repost)]
            n = len(g)
            if n:
                first = g.created_at.min()
                rel = sum(1 for a, b in zip(g.mentioned_teams, g.player_gsis) if t in a or len(b) > 0)
        share = rel / n if n else 0.0
        rows.append({"team": t, "handle": h, "outlet": o, "did": p.get("did"), "display_name": p.get("displayName"),
                     "profile_created_at": p.get("createdAt"), "profile_posts_count": p.get("postsCount"),
                     "own_posts": n, "first_post_at": first, "team_relevant_share": round(share, 3),
                     "predates_2025": bool(first is not None and first < pd.Timestamp("2025-01-01", tz="UTC")),
                     "verified": bool(p) and n >= 20 and share >= 0.15})
    HANDLES.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(HANDLES, index=False)


def profs_from_csv():
    if not HANDLES.exists():
        return {}
    return {r.handle.lower(): {"did": r.did, "displayName": r.display_name, "createdAt": r.profile_created_at,
                               "postsCount": r.profile_posts_count} for r in pd.read_csv(HANDLES).itertuples()}


def cmd_backfill(args):
    rdir = RAW / run_id()
    rdir.mkdir(parents=True)
    seed = seed_rows()
    profs = fetch_profiles([s[1] for s in seed])
    counts = {}
    for t, h, o in seed:
        if h.lower() not in profs:
            print("missing", h, flush=True)
            continue
        counts[h] = fetch_feed(h, rdir / f"{h}.jsonl")
        print(h, counts[h], flush=True)
    write_manifest(rdir, "backfill", {"counts": counts})
    df = build_posts()
    finish_handles(profs)
    print("posts", len(df))


def cmd_retag(args):
    df = build_posts()
    finish_handles(profs_from_csv())
    print("posts", len(df))


def cmd_capture(args):
    h = pd.read_csv(HANDLES)
    h = h[h.verified]
    df = pd.read_parquet(POSTS, columns=["handle", "uri", "is_repost"])
    known = df[~df.is_repost].groupby("handle")["uri"].agg(set).to_dict()
    rdir = RAW / run_id()
    rdir.mkdir(parents=True)
    counts = {}
    for hh in h.handle:
        counts[hh] = fetch_feed(hh, rdir / f"{hh}.jsonl", stop_uris=known.get(hh, set()), max_pages=20)
    write_manifest(rdir, "capture", {"counts": counts})
    out = build_posts()
    finish_handles(profs_from_csv())
    print("run", rdir.name, "fetched", sum(counts.values()), "posts_before", len(df), "posts_after", len(out))


def cmd_report(args):
    df = pd.read_parquet(POSTS)
    d = df[~df.is_repost]
    print("own posts", len(d), "reposts", int(df.is_repost.sum()))
    print(d.groupby("season").size().to_string())
    print(d.groupby("team").size().sort_values().to_string())
    tagged = d.avail_classes.map(len) > 0
    print("availability-tagged share", round(tagged.mean(), 4), "n", int(tagged.sum()))
    print("player-tagged share", round((d.player_gsis.map(len) > 0).mean(), 4))
    print(pd.Series([c for lst in d.avail_classes for c in lst]).value_counts().to_string())
    print(d.assign(t=tagged).groupby("season")["t"].mean().round(4).to_string())
    h = pd.read_csv(HANDLES)
    v = h[h.verified]
    cov = v.groupby("team").size().reindex(sorted(NICK), fill_value=0)
    print("verified handles", len(v), "of", len(h), "teams>=2:", int((cov >= 2).sum()), "teams>=1:", int((cov >= 1).sum()))
    print(cov[cov < 2].to_string())
    print("predates 2025 share", round(v.predates_2025.mean(), 3), int(v.predates_2025.sum()))
    print(h[~h.verified][["team", "handle", "own_posts", "team_relevant_share"]].to_string())


def cmd_leadtime(args):
    posts = pd.read_parquet(POSTS)
    posts = posts[(~posts.is_repost) & (posts.avail_classes.map(lambda c: any(x in OUT_CLASSES for x in c)))]
    h = pd.read_csv(HANDLES)
    posts = posts[posts.handle.isin(set(h[h.verified].handle))]
    base = latest_roster_path().parent
    inj = pd.read_parquet(base / "injuries.parquet")
    inj = inj[(inj.season == 2025) & (inj.game_type == "REG")].copy()
    inj["team"] = inj["team"].replace(TEAM_ALIAS)
    snaps = pd.read_parquet(base / "snap_counts.parquet")
    snaps = snaps[(snaps.season == 2025) & (snaps.game_type == "REG")].copy()
    snaps["team"] = snaps["team"].replace(TEAM_ALIAS)
    snaps["pct"] = snaps[["offense_pct", "defense_pct"]].max(axis=1)
    _, ros = load_rosters()
    ros = ros[ros.season == 2025][["team", "gsis_id", "pfr_id"]].drop_duplicates()
    snaps = snaps.merge(ros, left_on=["team", "pfr_player_id"], right_on=["team", "pfr_id"], how="left")
    prior = snaps.dropna(subset=["gsis_id"]).groupby(["team", "gsis_id", "week"])["pct"].max().reset_index()
    prior_map = {k: g.sort_values("week")[["week", "pct"]].values for k, g in prior.groupby(["team", "gsis_id"])}
    flag = inj.report_status.isin(["Out", "Doubtful"]) | inj.practice_status.str.contains("Did Not|Limited", na=False)
    posts_by = {k: g for k, g in posts.groupby("team")}
    rows = []
    for r in inj[flag].itertuples():
        arr = prior_map.get((r.team, r.gsis_id))
        if arr is None:
            continue
        before = arr[arr[:, 0] < r.week]
        if not len(before) or before[-1][1] < 0.5:
            continue
        t_off = r.effective_observed_at
        g = posts_by.get(r.team)
        p = g[(g.created_at >= t_off - pd.Timedelta(days=6)) & (g.created_at < t_off)] if g is not None else None
        if p is not None and len(p):
            p = p[p.player_gsis.map(lambda lst, x=r.gsis_id: x in lst)]
        rows.append({"team": r.team, "week": r.week, "gsis_id": r.gsis_id, "report_status": r.report_status,
                     "practice_status": r.practice_status, "official_proxy_ts": t_off,
                     "first_beat_ts": p.created_at.min() if p is not None and len(p) else pd.NaT,
                     "n_beat_posts": 0 if p is None else len(p)})
    res = pd.DataFrame(rows)
    res["lead_hours"] = (res.official_proxy_ts - res.first_beat_ts).dt.total_seconds() / 3600
    wed = res.official_proxy_ts.dt.normalize() - pd.Timedelta(days=3) + pd.Timedelta(hours=18)
    res["beat_before_wed_18z"] = res.first_beat_ts < wed
    both = res.dropna(subset=["lead_hours"])
    print("starter flagged team-weeks 2025 REG:", len(res), "teams", res.team.nunique(), "weeks", res.week.nunique())
    print("with at least one beat availability post naming the player:", len(both))
    if len(both):
        print("median lead hours vs weekly proxy:", round(both.lead_hours.median(), 2),
              "IQR", round(both.lead_hours.quantile(.25), 2), round(both.lead_hours.quantile(.75), 2))
        print("share beat post before Wed 18:00Z of game week:", round(both.beat_before_wed_18z.mean(), 3))
        sev = both[both.report_status.isin(["Out", "Doubtful"])]
        print("Out/Doubtful only n", len(sev), "median lead", round(sev.lead_hours.median(), 2))
    res.to_csv(LEADTIME, index=False)


def main():
    ap = argparse.ArgumentParser(description="MOD-24 U9 beat reporter Bluesky acquisition")
    ap.add_argument("mode", choices=["backfill", "retag", "capture", "report", "leadtime"])
    a = ap.parse_args()
    {"backfill": cmd_backfill, "retag": cmd_retag, "capture": cmd_capture, "report": cmd_report, "leadtime": cmd_leadtime}[a.mode](a)


if __name__ == "__main__":
    sys.exit(main())
