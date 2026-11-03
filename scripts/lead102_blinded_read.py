import argparse
import glob
import json
import re
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import lead99_llm_read as L

BLIND = L.ART / "blinded"
SOURCE = L.ART / "runs" / "2026-week-09-tuesday-20261103T183210Z.json"
REPEATS = L.ART / "repeats"
SEASONS_MIN = 2020
BOOT = 10000
SEED = 102
LABELS = ("Team A", "Team B")
ABBR_ALIASES = ("JAC", "LA", "WSH", "OAK", "SD", "STL", "ARZ", "BLT", "CLV", "HST", "SL")
LAST_MIN_LEN = 3
REDACT_KEYS = ("author", "url", "source", "book")
URL = re.compile(r"https?://\S+|www\.\S+|\b[\w-]+\.(?:com|org|net|io)\b\S*")
HANDLE = re.compile(r"@[\w.]+")


def logit(p):
    return float(np.log(p / (1 - p)))


def team_table():
    rows = {}
    for d in sorted(glob.glob(str(L.ROOT / "data" / "market" / "raw" / "*")), reverse=True):
        f = Path(d) / "quotes.parquet"
        if not f.exists():
            continue
        q = pd.read_parquet(
            f, columns=["home_team", "home_team_name", "away_team", "away_team_name"]
        )
        for _, a in q.drop_duplicates().iterrows():
            rows.setdefault(a["home_team"], a["home_team_name"])
            rows.setdefault(a["away_team"], a["away_team_name"])
        if len(rows) >= 32:
            break
    return rows


def team_strings(abbr, full):
    words = full.split()
    k = 2 if full.startswith(("New York", "Los Angeles")) else len(words) - 1
    return {"abbr": abbr, "full": full, "nick": " ".join(words[k:]), "city": " ".join(words[:k])}


def roster_names():
    fs = sorted(
        glob.glob(str(L.ROOT / "data" / "players" / "raw" / "*" / "weekly_rosters.parquet"))
    )
    d = pd.read_parquet(fs[-1], columns=["season", "team", "full_name"])
    d = d[d["season"] >= SEASONS_MIN].dropna(subset=["full_name"])
    return d.groupby("team")["full_name"].agg(lambda s: sorted(set(s))).to_dict(), set(
        d["full_name"]
    )


def stadiums(packet):
    return {
        r["stadium"]
        for g in packet["games"]
        for r in (g.get("weather") or [])
        if isinstance(g.get("weather"), list) and r.get("stadium")
    } | {
        r["stadium"]
        for g in packet["games"]
        if isinstance(g.get("weather"), dict)
        for v in g["weather"].values()
        for r in (v if isinstance(v, list) else [v])
        if isinstance(r, dict) and r.get("stadium")
    }


def pat(words, ci):
    ws = sorted({w for w in words if w}, key=len, reverse=True)
    if not ws:
        return None
    return re.compile(
        r"(?<![\w])(?:" + "|".join(re.escape(w) for w in ws) + r")(?![\w])", re.I if ci else 0
    )


def surname(name):
    t = name.replace(".", " ").split()
    return t[-1] if len(t) > 1 and len(t[-1]) >= LAST_MIN_LEN else None


class Blinder:
    def __init__(self, packet):
        self.teams = {a: team_strings(a, n) for a, n in team_table().items()}
        self.roster, self.all_players = roster_names()
        self.stadiums = stadiums(packet)
        self.anyone = pat([n for n in self.all_players if len(n.split()) > 1], True)
        self.packet = packet

    def players_for(self, abbr, game):
        names = set(self.roster.get(abbr, []))
        side = game.get("injuries", {}).get(abbr, {})
        names |= {r["player"] for r in side.get("rows", []) if r.get("player")}
        return sorted(names, key=len, reverse=True)

    def blind_game(self, game, idx):
        away, home = game["away"], game["home"]
        label = {away: LABELS[0], home: LABELS[1]}
        pnum = {away: {}, home: {}}
        ts = self.teams
        other = [a for a in ts if a not in label]

        def plabel(abbr, name):
            m = pnum[abbr]
            if name not in m:
                m[name] = f"{label[abbr]} Player {len(m) + 1}"
            return m[name]

        fulls, lasts = {}, {}
        for abbr in (away, home):
            names = self.players_for(abbr, game)
            fulls[abbr] = [(n, pat([n], True)) for n in names]
            ln = {}
            for n in names:
                s = surname(n)
                if s:
                    ln.setdefault(s, n)
            inj = {r["player"] for r in game.get("injuries", {}).get(abbr, {}).get("rows", [])}
            lasts[abbr] = [(n, pat([s], n in inj)) for s, n in ln.items()]
        other_pat = pat(
            [w for a in other for w in (ts[a]["full"], ts[a]["nick"], ts[a]["city"])], True
        )
        other_abbr = pat(other + list(ABBR_ALIASES), False)
        own = {
            a: (pat([ts[a]["full"], ts[a]["nick"], ts[a]["city"]], True), pat([a], False))
            for a in (away, home)
        }
        stad = pat(self.stadiums, True)
        anyone = self.anyone

        def scrub(text):
            t = URL.sub("[link]", text)
            t = HANDLE.sub("[handle]", t)
            for abbr in (away, home):
                for n, p in fulls[abbr]:
                    if p.search(t):
                        t = p.sub(plabel(abbr, n), t)
            for abbr in (away, home):
                for n, p in lasts[abbr]:
                    if p.search(t):
                        t = p.sub(plabel(abbr, n), t)
            if stad:
                t = stad.sub("the stadium", t)
            for abbr in (away, home):
                fp, ap = own[abbr]
                t = ap.sub(label[abbr], fp.sub(label[abbr], t))
            if other_pat:
                t = other_pat.sub("another team", t)
            t = other_abbr.sub("another team", t)
            return anyone.sub("another player", t)

        def walk(o, key=None):
            if isinstance(o, dict):
                return {(label.get(k, k)): walk(v, k) for k, v in o.items()}
            if isinstance(o, list):
                return [walk(v, key) for v in o]
            if isinstance(o, str):
                if key in REDACT_KEYS:
                    return "[redacted]"
                if key == "player":
                    for abbr in (away, home):
                        if any(o == n for n, _ in fulls[abbr]):
                            return plabel(abbr, o)
                return scrub(o)
            return o

        g = {k: v for k, v in game.items() if k not in ("game_id", "away", "home")}
        out = walk(g)
        out["game_id"] = f"game_{idx + 1:02d}"
        out["away"] = LABELS[0]
        out["home"] = LABELS[1]
        return out

    def blind(self):
        games, mapping = [], {}
        for i, g in enumerate(self.packet["games"]):
            b = self.blind_game(g, i)
            games.append(b)
            mapping[b["game_id"]] = g["game_id"]
        p = {k: v for k, v in self.packet.items() if k != "games"}
        p["games"] = games
        return p, mapping

    def leaks(self, blinded, mapping):
        text = json.dumps(blinded, ensure_ascii=False)
        by_id = {g["game_id"]: g for g in self.packet["games"]}
        words = [w for s in self.teams.values() for w in (s["full"], s["nick"], s["city"])]
        found = [("team", m.group(0)) for m in pat(words, True).finditer(text)]
        found += [
            ("abbr", m.group(0))
            for m in pat(list(self.teams) + list(ABBR_ALIASES), False).finditer(text)
        ]
        sp = pat(self.stadiums, True)
        if sp:
            found += [("stadium", m.group(0)) for m in sp.finditer(text)]
        pp = pat([n for n in self.all_players if len(n.split()) > 1], True)
        found += [("player", m.group(0)) for m in pp.finditer(text)]
        for bg in blinded["games"]:
            g = by_id[mapping[bg["game_id"]]]
            gt = json.dumps(bg, ensure_ascii=False)
            for abbr in (g["away"], g["home"]):
                for s in {surname(n) for n in self.players_for(abbr, g)} - {None}:
                    if re.search(r"(?<![\w])" + re.escape(s) + r"(?![\w])", gt):
                        found.append(("lastname", s))
        return found


def load_source():
    d = json.loads(SOURCE.read_text(encoding="utf-8"))
    fz = json.loads(L.FROZEN.read_text())
    if fz["prompt_sha256"] != L.prompt_sha() or fz["model"] != L.MODEL:
        raise SystemExit("prompt or model differs from frozen enrollment")
    if d["prompt_sha256"] != L.prompt_sha():
        raise SystemExit("stored run used a different prompt")
    pt = json.dumps(d["packet"], ensure_ascii=False, sort_keys=True)
    if L.sha256_bytes(pt.encode("utf-8")) != d["packet_sha256"]:
        raise SystemExit("stored packet hash mismatch")
    return d


def cmd_blind(a):
    d = load_source()
    b = Blinder(d["packet"])
    packet, mapping = b.blind()
    leaks = b.leaks(packet, mapping)
    BLIND.mkdir(parents=True, exist_ok=True)
    (BLIND / "blinded_packet.json").write_text(
        json.dumps(
            {"packet": packet, "mapping": mapping, "source_packet_sha256": d["packet_sha256"]},
            ensure_ascii=False,
            indent=1,
        ),
        encoding="utf-8",
    )
    print("leaks", len(leaks))
    for k, v in sorted(set(leaks)):
        print(" ", k, v)
    print("prompt_sha256", L.prompt_sha(), "matches frozen")


def cmd_run(a):
    d = load_source()
    store = json.loads((BLIND / "blinded_packet.json").read_text(encoding="utf-8"))
    packet = store["packet"]
    pt = json.dumps(packet, ensure_ascii=False, sort_keys=True)
    ids = [g["game_id"] for g in packet["games"]]
    prompt_text = L.PROMPT.read_text(encoding="utf-8")
    cwd = BLIND / "cwd"
    cwd.mkdir(parents=True, exist_ok=True)
    out = BLIND / "reads"
    out.mkdir(parents=True, exist_ok=True)
    for _ in range(a.k):
        env, raw = L.call_claude(prompt_text, pt, str(cwd))
        rows, err = None, None
        try:
            rows = L.validate(L.extract_json(raw), ids)
        except Exception as e:
            err = repr(e)
        t = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
        path = out / f"blinded-{t}.json"
        path.write_text(
            json.dumps(
                {
                    "source_packet_sha256": d["packet_sha256"],
                    "blinded_packet_sha256": L.sha256_bytes(pt.encode("utf-8")),
                    "prompt_sha256": L.prompt_sha(),
                    "model": L.MODEL,
                    "raw_response": raw,
                    "envelope": env,
                    "valid": rows is not None,
                    "error": err,
                    "parsed_rows": rows,
                    "recorded_at_utc": t,
                },
                ensure_ascii=False,
                indent=1,
            ),
            encoding="utf-8",
        )
        print(path, rows is not None, flush=True)


def reads(files, mapping=None):
    out = []
    for f in files:
        j = json.loads(Path(f).read_text(encoding="utf-8"))
        if not j.get("parsed_rows"):
            continue
        out.append(
            {
                (mapping[r["game_id"]] if mapping else r["game_id"]): logit(
                    r["home_cover_probability"]
                )
                for r in j["parsed_rows"]
            }
        )
    return out


def cmd_analyze(a):
    store = json.loads((BLIND / "blinded_packet.json").read_text(encoding="utf-8"))
    mapping = store["mapping"]
    named = reads([SOURCE, *sorted(glob.glob(str(REPEATS / "2026-week-09-tuesday-*.json")))])
    blind = reads(sorted(glob.glob(str(BLIND / "reads" / "*.json"))), mapping)
    ids = list(mapping.values())
    N = np.array([[r[i] for i in ids] for r in named if all(i in r for i in ids)])
    B = np.array([[r[i] for i in ids] for r in blind if all(i in r for i in ids)])
    nm, bm = N.mean(0), B.mean(0)
    sdn = float(np.sqrt(N.var(0, ddof=1).mean()))
    sdb = float(np.sqrt(B.var(0, ddof=1).mean()))
    pooled_se = a.noise * np.sqrt(1 / len(N) + 1 / len(B))
    diff = nm - bm
    print(f"named reads {len(N)} blinded reads {len(B)} games {len(ids)}")
    print(
        f"within-read SD named {sdn:.3f} blinded {sdb:.3f}; noise {a.noise};"
        f" SE of a game diff {pooled_se:.3f}"
    )
    for i in np.argsort(-np.abs(diff)):
        print(
            f"{ids[i]} named {nm[i]:+.3f} blinded {bm[i]:+.3f}"
            f" diff {diff[i]:+.3f} z {diff[i] / pooled_se:+.1f}"
        )
    rng = np.random.default_rng(SEED)
    n = len(ids)
    cs, ms = [], []
    for _ in range(BOOT):
        ix = rng.integers(0, n, n)
        if np.std(nm[ix]) > 0 and np.std(bm[ix]) > 0:
            cs.append(np.corrcoef(nm[ix], bm[ix])[0, 1])
        ms.append(diff[ix].mean())
    cs, ms = np.array(cs), np.array(ms)
    print(
        f"corr {np.corrcoef(nm, bm)[0, 1]:.3f}"
        f" boot95 [{np.percentile(cs, 2.5):.3f}, {np.percentile(cs, 97.5):.3f}]"
    )
    print(
        f"mean diff {diff.mean():+.3f}"
        f" boot95 [{np.percentile(ms, 2.5):+.3f}, {np.percentile(ms, 97.5):+.3f}]"
        f" probability_positive {(ms > 0).mean():.3f}"
    )
    print(
        f"SD of game diffs {diff.std(ddof=1):.3f} vs noise SE {pooled_se:.3f};"
        f" games |z|>2: {int((np.abs(diff / pooled_se) > 2).sum())}"
    )


def main():
    ap = argparse.ArgumentParser(description="LEAD-102 name-blinded control read")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("blind").set_defaults(f=cmd_blind)
    r = sub.add_parser("run")
    r.add_argument("--k", type=int, required=True)
    r.set_defaults(f=cmd_run)
    z = sub.add_parser("analyze")
    z.add_argument("--noise", type=float, required=True)
    z.set_defaults(f=cmd_analyze)
    a = ap.parse_args()
    a.f(a)


main()
