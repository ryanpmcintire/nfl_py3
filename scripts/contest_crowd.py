from __future__ import annotations

import argparse
import html
import json
import re
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

RAW = ROOT / "data/external/contest_crowd/raw"
OUT = ROOT / "artifacts/contest_crowd"
HUB = "https://www.thespread.com/nfl-super-contest-picks-from-the-las-vegas-hilton/"
CDX = "https://web.archive.org/cdx/search/cdx?url=" + HUB.split("://")[1] + "&output=json&fl=timestamp,statuscode,digest&filter=statuscode:200&limit=5000"
UA = "Mozilla/5.0 (research script)"
DELAY = 4.0
FIRST_CAPTURE = "20190901"
SCHEDULE = sorted((ROOT / "data/raw").glob("2026*/schedules.parquet"))[-1]
PICK = ROOT / "artifacts/pick_probability/20261116T171555Z/per_game.parquet"
OPENER_QW = ROOT / "artifacts/v4_candidate/opener_QW.parquet"
NULL_DRAWS = 200
BOOT_DRAWS = 2000
MIN_TEAMS = 20
POOL_CUTOFF_HOUR_ET = 16
SEED = 20261116

CODES = {
    "ARI", "ARZ", "ATL", "BAL", "BUF", "CAR", "CHI", "CIN", "CLE", "DAL", "DEN", "DET", "GB", "HOU",
    "IND", "JAX", "JAC", "KC", "LV", "OAK", "LAC", "LAR", "LA", "MIA", "MIN", "NE", "NO", "NYG",
    "NYJ", "PHI", "PIT", "SEA", "SF", "TB", "TEN", "WAS", "WSH",
}
NICK = {
    "cardinals": "ARI", "falcons": "ATL", "ravens": "BAL", "bills": "BUF", "panthers": "CAR",
    "bears": "CHI", "bengals": "CIN", "browns": "CLE", "cowboys": "DAL", "broncos": "DEN",
    "lions": "DET", "packers": "GB", "texans": "HOU", "colts": "IND", "jaguars": "JAX",
    "chiefs": "KC", "raiders": "LV", "chargers": "LAC", "rams": "LA", "dolphins": "MIA",
    "vikings": "MIN", "patriots": "NE", "saints": "NO", "giants": "NYG", "jets": "NYJ",
    "eagles": "PHI", "steelers": "PIT", "seahawks": "SEA", "49ers": "SF", "buccaneers": "TB",
    "titans": "TEN", "washington": "WAS", "commanders": "WAS", "redskins": "WAS", "team": "WAS",
}


def norm_code(code, season):
    code = {"ARZ": "ARI", "JAC": "JAX", "WSH": "WAS", "LAR": "LA"}.get(code, code)
    if code in ("LV", "OAK"):
        return "OAK" if season < 2020 else "LV"
    return code


def nick_code(name, season):
    words = re.findall(r"[a-z0-9]+", name.lower())
    for w in reversed(words):
        if w in NICK:
            return norm_code(NICK[w], season)
    return None


def get(url):
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=180) as f:
        return f.read().decode("utf-8", "replace")


def totext(t):
    t = re.sub(r"(?s)<script.*?</script>|<style.*?</style>|<!--.*?-->", " ", t)
    t = html.unescape(re.sub(r"<.*?>", " | ", t))
    t = re.sub(r"\s+", " ", t)
    return re.sub(r"(\s*\|\s*)+", " | ", t)


def fetch(_args):
    RAW.mkdir(parents=True, exist_ok=True)
    rows = json.loads(get(CDX))[1:]
    (RAW / "cdx_thespread_hub.json").write_text(json.dumps(rows), encoding="utf-8")
    seen = set()
    done = 0
    for ts, _, digest in rows:
        if ts < FIRST_CAPTURE or digest in seen:
            continue
        seen.add(digest)
        path = RAW / f"thespread_hub_{ts}.html"
        if path.exists():
            continue
        try:
            path.write_text(get(f"https://web.archive.org/web/{ts}id_/{HUB}"), encoding="utf-8")
            done += 1
        except Exception as exc:
            print("fail", ts, exc, flush=True)
        time.sleep(DELAY)
    print(json.dumps({"captures_listed": len(rows), "distinct_since": len(seen), "downloaded": done}))


def season_from(datestr):
    m = re.match(r"(\d{1,2})/(\d{1,2})/(\d{2,4})", datestr)
    if not m:
        return None
    y = int(m.group(3))
    y = y + 2000 if y < 100 else y
    return y if int(m.group(1)) >= 8 else y - 1


def parse_matchups(block, season):
    out = {}
    pat = re.compile(r"^(.+?)\s*(-\d+(?:\.\d)?|PK|Pk|pk)?\s+vs\.?\s+(.+?)\s*(-\d+(?:\.\d)?|PK|Pk|pk)?$")
    for seg in block.split("|"):
        seg = seg.strip()
        m = pat.match(seg)
        if not m:
            continue
        a, la, b, lb = m.group(1), m.group(2), m.group(3), m.group(4)
        ca, cb = nick_code(a, season), nick_code(b, season)
        if not ca or not cb or ca == cb:
            continue
        a_home = re.sub(r"[^A-Za-z]", "", a).isupper() and not re.sub(r"[^A-Za-z]", "", b).isupper()
        b_home = re.sub(r"[^A-Za-z]", "", b).isupper() and not re.sub(r"[^A-Za-z]", "", a).isupper()
        if a_home == b_home:
            continue
        home, away = (ca, cb) if a_home else (cb, ca)
        num = la or lb
        x = 0.0 if num is None or num.lower() == "pk" else abs(float(num))
        fav = ca if la else cb
        home_line = 0.0 if x == 0 else (-x if fav == home else x)
        out[frozenset((home, away))] = (home, away, home_line)
    return out


def parse_old(text, ts):
    blocks = []
    heads = list(re.finditer(r"SUPER ?CONTEST WEEK (\d+)\s*:\s*([0-9/]+)", text))
    for i, h in enumerate(heads):
        end = heads[i + 1].start() if i + 1 < len(heads) else len(text)
        block = text[h.start():end]
        season = season_from(h.group(2))
        if season is None:
            continue
        ks = [j for j in (block.find("PICKS BY"), block.find("BY # OF TEAMS")) if j >= 0]
        k = min(ks) if ks else -1
        stops = [j for j in (block.find("TOP 10"), block.find("LEADERS")) if j > k]
        cblock = block[k: min(stops) if stops else len(block)] if k >= 0 else ""
        counts = {}
        for code, n in re.findall(r"\b([A-Z]{2,3})\s?(\d{1,4})\b", cblock):
            if code in CODES:
                counts.setdefault(norm_code(code, season), int(n))
        flat = re.sub(r"\s*\|\s*(-\d+(?:\.\d)?|PK)(?=\s*\|)", r" \1", cblock or block)
        flat = re.sub(r"\s*\|\s*vs\.?\s*\|\s*", " vs. ", flat)
        blocks.append({"season": season, "week": int(h.group(1)), "capture": ts, "counts": counts,
                       "matchups": parse_matchups(flat, season)})
    return blocks


def parse_live(text, ts):
    blocks = []
    m0 = re.search(r"(20\d\d) Westgate NFL SuperContest", text)
    season = int(m0.group(1)) if m0 else None
    heads = list(re.finditer(r"Westgate SuperContest Week (\d+) \|", text))
    for i, h in enumerate(heads):
        end = heads[i + 1].start() if i + 1 < len(heads) else len(text)
        block = text[h.start():end]
        k = block.find("Consensus Picks")
        if k < 0 or season is None:
            continue
        counts = {}
        for name, n in re.findall(r"\| \d+ \| ([A-Za-z0-9 ]+?) \| (\d+) \| [\d.]+%", block[k:]):
            c = nick_code(name, season)
            if c:
                counts.setdefault(c, int(n))
        blocks.append({"season": season, "week": int(h.group(1)), "capture": ts, "counts": counts, "matchups": {}})
    return blocks


def parse(_args):
    blocks = []
    for path in sorted(RAW.glob("thespread_hub_*.html")):
        ts = path.stem.split("_")[-1]
        blocks += parse_old(totext(path.read_text(encoding="utf-8")), ts)
    live = RAW / "thespread_10356.html"
    if live.exists():
        blocks += parse_live(totext(live.read_text(encoding="utf-8")), "live_20261116")
    good = [b for b in blocks if len(b["counts"]) >= MIN_TEAMS]
    sched = pd.read_parquet(SCHEDULE)
    sched = sched[sched["game_type"].eq("REG")].copy()
    sched["kick"] = pd.to_datetime(sched["gameday"].astype(str) + " " + sched["gametime"].fillna("13:00").astype(str)).dt.tz_localize("America/New_York")
    day = pd.to_datetime(sched["gameday"])
    shift = np.where(day.dt.dayofweek.eq(0), -1, (6 - day.dt.dayofweek) % 7)
    sunday = (day + pd.to_timedelta(shift, unit="D")).dt.tz_localize("America/New_York")
    sched["deadline"] = np.minimum(sched["kick"], sunday + pd.Timedelta(hours=POOL_CUTOFF_HOUR_ET))
    sched["sunday_1pm"] = sunday + pd.Timedelta(hours=13)
    rows = []
    by_key = {}
    for b in good:
        by_key.setdefault((b["season"], b["week"]), []).append(b)
    for (season, week), bs in by_key.items():
        final = max(bs, key=lambda b: (b["capture"] != "live_20261116", b["capture"]))
        first = min(b["capture"] for b in bs)
        lines = {}
        for b in sorted(bs, key=lambda b: b["capture"]):
            lines.update(b["matchups"])
        g = sched[(sched["season"] == season) & (sched["week"] == week)]
        for _, r in g.iterrows():
            h, a = final["counts"].get(r["home_team"]), final["counts"].get(r["away_team"])
            mu = lines.get(frozenset((r["home_team"], r["away_team"])))
            first_utc = None if first.startswith("live") else pd.Timestamp(first, tz="UTC")
            rows.append({
                "game_id": r["game_id"], "season": season, "week": week, "home_team": r["home_team"],
                "away_team": r["away_team"], "home_count": h, "away_count": a,
                "contest_home_line": mu[2] if mu and mu[0] == r["home_team"] else np.nan,
                "result": r["result"], "kick_et": str(r["kick"]), "deadline_et": str(r["deadline"]),
                "first_capture_utc": str(first_utc) if first_utc is not None else "live",
                "capture_before_deadline": bool(first_utc is not None and first_utc < r["deadline"].tz_convert("UTC")),
                "sunday_or_monday": bool(r["kick"] >= r["sunday_1pm"]),
                "n_captures": len(bs),
            })
    df = pd.DataFrame(rows)
    df["share_home"] = df["home_count"] / (df["home_count"] + df["away_count"])
    df["term"] = df["share_home"] - 0.5
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT / "contest_shares.parquet", index=False)
    cov = df.assign(has=df["term"].notna(), line=df["contest_home_line"].notna()).groupby("season").agg(
        weeks=("week", "nunique"), games=("game_id", "size"), with_counts=("has", "sum"),
        with_contest_line=("line", "sum"), capture_before_deadline=("capture_before_deadline", "sum"))
    sched_reg = sched.groupby("season")["game_id"].size()
    cov["season_games"] = sched_reg.reindex(cov.index).to_numpy()
    print(cov.to_string())
    (OUT / "coverage.json").write_text(cov.reset_index().to_json(orient="records"), encoding="utf-8")


def boot_weeks(df, stat, rng):
    keys = (df["season"] * 100 + df["week"]).to_numpy()
    uk = np.unique(keys)
    idx = {k: np.flatnonzero(keys == k) for k in uk}
    out = []
    for _ in range(BOOT_DRAWS):
        pick = rng.choice(uk, size=len(uk), replace=True)
        out.append(stat(df.iloc[np.concatenate([idx[k] for k in pick])]))
    return np.array(out)


def contest_line_cell(df, rng):
    m = df["result"] + df["contest_home_line"]
    d = df.assign(hc=(m > 0).astype(float))[(m != 0) & df["term"].notna() & (df["term"] != 0) & m.notna()].reset_index(drop=True)

    def acc(x):
        return float(((x["term"] > 0) == (x["hc"] == 1)).mean())

    obs = acc(d)
    null = []
    for _ in range(NULL_DRAWS):
        t = d["term"].to_numpy().copy()
        for s in d["season"].unique():
            k = np.flatnonzero(d["season"].to_numpy() == s)
            t[k] = rng.permutation(t[k])
        null.append(float(((t > 0) == (d["hc"].to_numpy() == 1)).mean()))
    bs = (boot_weeks(d, acc, rng) - 0.5) * 100
    wins = int(((d["term"] > 0) == (d["hc"] == 1)).sum())
    per = {str(s): f"{int(((g['term'] > 0) == (g['hc'] == 1)).sum())}-{int(((g['term'] > 0) != (g['hc'] == 1)).sum())}" for s, g in d.groupby("season")}
    strong = d[d["term"].abs() >= d["term"].abs().median()]
    return {
        "games": len(d), "record": f"{wins}-{len(d) - wins}", "effect_accuracy_points": (obs - 0.5) * 100,
        "interval": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
        "probability_positive": float((bs > 0).mean()),
        "null_percentile": float((np.array(null) < obs).mean() * 100),
        "null_p95_points": float((np.percentile(null, 95) - 0.5) * 100),
        "per_season": per, "weeks": int((d["season"] * 100 + d["week"]).nunique()),
        "above_median_share_record": f"{int(((strong['term'] > 0) == (strong['hc'] == 1)).sum())}-{int(((strong['term'] > 0) != (strong['hc'] == 1)).sum())}",
    }


def fitted_cell(df, rng):
    from mod27_fenced_repairs_replay import FOUR, Fit, log_loss, paired, record

    pop = pd.read_parquet(PICK)
    pop["game_id"] = pop["game_id"].astype(str)
    pop = pop.reset_index(drop=True)
    qw = pd.read_parquet(OPENER_QW)["home_cover_probability_at_open"]
    pop["model_logit"] = np.log(np.clip(qw.reindex(pop["game_id"]).to_numpy(float), 1e-9, 1 - 1e-9) / np.clip(1 - qw.reindex(pop["game_id"]).to_numpy(float), 1e-9, 1))
    term = df.set_index("game_id")["term"]
    pop["contest_term"] = term.reindex(pop["game_id"]).fillna(0.0).to_numpy(float)
    y = pop["home_covered"].to_numpy(float)
    cols5 = tuple(FOUR) + ("contest_term",)

    def loso(frame, cols):
        p = np.full(len(frame), np.nan)
        coefs = {}
        for s in sorted(frame["season"].unique()):
            held = (frame["season"] == s).to_numpy()
            fit = Fit(cols, frame[~held])
            p[held] = fit.predict(frame[held])
            coefs[str(int(s))] = fit.natural()
        return p, coefs

    p4, _ = loso(pop, tuple(FOUR))
    p5, coefs = loso(pop, cols5)
    ins = Fit(cols5, pop).predict(pop)
    acc4, acc5 = float(((p4 >= 0.5) == (y == 1)).mean()), float(((p5 >= 0.5) == (y == 1)).mean())
    null = []
    for _ in range(NULL_DRAWS):
        f = pop.copy()
        t = f["contest_term"].to_numpy().copy()
        for s in f["season"].unique():
            k = np.flatnonzero(f["season"].to_numpy() == s)
            t[k] = rng.permutation(t[k])
        f["contest_term"] = t
        pn, _ = loso(f, cols5)
        null.append(float(((pn >= 0.5) == (y == 1)).mean()))
    frame = pop.assign(week_block=pop["season"] * 100 + pop["week"])
    changed = (p4 >= 0.5) != (p5 >= 0.5)
    nz = pop["contest_term"].ne(0).to_numpy()
    return {
        "games": len(pop), "nonzero_term_games": int(nz.sum()),
        "base_record": record(p4, y), "arm_record": record(p5, y),
        "effect_accuracy_points": (acc5 - acc4) * 100,
        "log_loss_base": float(log_loss(p4, y).mean()), "log_loss_arm": float(log_loss(p5, y).mean()),
        "in_sample_log_loss_arm": float(log_loss(ins, y).mean()),
        "null_percentile": float((np.array(null) < acc5).mean() * 100),
        "null_p95_points": float((np.percentile(null, 95) - acc4) * 100),
        "week_block": paired(frame, p4, p5, "week_block"),
        "changed_picks": int(changed.sum()),
        "changed_arm": record(p5[changed], y[changed]) if changed.any() else "0-0",
        "changed_base": record(p4[changed], y[changed]) if changed.any() else "0-0",
        "fold_contest_coefficient": {k: v.get("contest_term") if isinstance(v, dict) else None for k, v in coefs.items()},
    }


def grade(args):
    rng = np.random.default_rng(SEED)
    df = pd.read_parquet(OUT / "contest_shares.parquet")
    df = df[df["season"].between(args.start, args.end)]
    res = {"declared_at": args.declared_at}
    res["contest_line_all"] = contest_line_cell(df, rng)
    res["contest_line_sunday_monday"] = contest_line_cell(df[df["sunday_or_monday"]], rng)
    res["stage_qw_2020_25"] = fitted_cell(df[df["season"].between(2020, 2025)], rng)
    (OUT / "results.json").write_text(json.dumps(res, indent=2, default=str), encoding="utf-8")
    print(json.dumps(res, default=str))


def main():
    ap = argparse.ArgumentParser(description="Handicapping-contest crowd: SuperContest consensus counts from archived TheSpread hub captures.")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("fetch").set_defaults(fn=fetch)
    sub.add_parser("parse").set_defaults(fn=parse)
    g = sub.add_parser("grade")
    g.add_argument("--declared-at", required=True)
    g.add_argument("--start", type=int, default=2019)
    g.add_argument("--end", type=int, default=2025)
    g.set_defaults(fn=grade)
    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
