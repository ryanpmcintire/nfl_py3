import json
import re
import sys
import time
import urllib.error
import urllib.request
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))
sys.path.insert(0, str(ROOT / "src"))

DATA = ROOT / "data" / "external" / "preseason_futures"
OUT = ROOT / "artifacts" / "futures_anchor"
LANE = "docs/lanes/futures-anchor-2026-11-16.md"
FAMILY = "futures_anchor_family_2026_11_16"
STREAM = 11
UA = "Mozilla/5.0 (compatible; research-script/1.0)"
DELAY_SECONDS = 3
NFL_SEASONS = range(2009, 2027)
CFB_SEASONS = range(2012, 2026)
NFL_URL = "https://www.covers.com/sportsoddshistory/nfl-reg/?y={y}&sa=nfl&a=sb"
CFB_URL = "https://www.covers.com/sportsoddshistory/cfb-main/?y={y}&sa=cfb&a=nc"
CFB_ALIAS = {
    "Miami (FL)": "Miami",
    "Miami (Fla.)": "Miami",
    "Miami (Ohio)": "Miami (OH)",
    "Central Florida": "UCF",
    "Sourth Florida": "South Florida",
    "USF": "South Florida",
    "Appalachian State": "App State",
    "Hawaii": "Hawai'i",
    "Connecticut": "UConn",
    "Massachusetts": "UMass",
    "Mississippi": "Ole Miss",
    "Southern Mississippi": "Southern Miss",
    "Southern Cal": "USC",
    "Southern California": "USC",
    "North Carolina State": "NC State",
    "Brigham Young": "BYU",
    "Texas Christian": "TCU",
    "Southern Methodist": "SMU",
    "Louisiana State": "LSU",
    "Pitt": "Pittsburgh",
    "Cal": "California",
    "Louisiana-Lafayette": "Louisiana",
    "UL Lafayette": "Louisiana",
    "Texas-San Antonio": "UTSA",
    "San Jose State": "San José State",
    "Louisiana Lafayette": "Louisiana",
    "Middle Tennessee State": "Middle Tennessee",
    "Sam Houston State": "Sam Houston",
    "Texas San Antonio": "UTSA",
}
NOT_TEAMS = {"FIELD", "Field", "The Field"}
CANDIDATES = {
    "futures_gap": {
        "hypothesis": "negative",
        "what": (
            "the team's preseason championship-futures strength percentile among listed teams "
            "minus its current opener-implied strength percentile among the same teams, where "
            "opener strength is a least-squares rating fitted to the season's strictly earlier "
            "week openers with a home-field column."
        ),
        "plain": "were rated higher by preseason title odds than by this season's spreads",
    },
    "futures_gap_resid_ats": {
        "hypothesis": "negative",
        "what": (
            "the same gap residualised (pooled least squares, no outcome of the graded game) on "
            "the home-minus-away share of strictly earlier games covered against the opener."
        ),
        "plain": "were rated higher by preseason title odds than by spreads, after their cover record",
    },
}
NAMES = list(CANDIDATES)
REQUESTS = {"count": 0}


def fetch(url):
    for attempt in range(3):
        time.sleep(DELAY_SECONDS)
        REQUESTS["count"] += 1
        req = urllib.request.Request(url, headers={"User-Agent": UA, "Accept": "text/html"})
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                return resp.read().decode("utf-8", "replace")
        except urllib.error.HTTPError as exc:
            if exc.code in (403, 429) or exc.code >= 500:
                raise SystemExit(f"stop on {exc.code} {url}") from exc
            raise
        except urllib.error.URLError:
            if attempt == 2:
                raise
    return None


def text(c):
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", c)).replace("&nbsp;", " ").replace("&amp;", "&").strip()


def implied(odds):
    m = re.fullmatch(r"([+-])(\d+)", odds.replace(" ", ""))
    if not m:
        return np.nan
    v = float(m.group(2))
    return 100 / (v + 100) if m.group(1) == "+" else v / (v + 100)


def parse(html, league, season, url):
    tables = re.findall(r"<table.*?</table>", html, re.S)
    if not tables:
        return [], None
    trs = re.findall(r"<tr.*?</tr>", tables[0], re.S)
    groups = []
    for th in re.findall(r"<th([^>]*)>(.*?)</th>", trs[0], re.S):
        span = re.search(r"colspan=['\"]?(\d+)", th[0])
        if "rowspan" in th[0]:
            continue
        groups += [text(th[1])] * (int(span.group(1)) if span else 1)
    labels = [text(c) for c in re.findall(r"<th[^>]*>(.*?)</th>", trs[1], re.S)]
    rows = []
    for tr in trs[2:]:
        tds = [text(c) for c in re.findall(r"<td[^>]*>(.*?)</td>", tr, re.S)]
        if len(tds) < 2:
            continue
        for i, (g, lab) in enumerate(zip(groups, labels, strict=False)):
            if g.lower().startswith("result") or i + 1 >= len(tds):
                continue
            rows.append(
                {
                    "league": league,
                    "season": season,
                    "team_name": tds[0],
                    "column_index": i,
                    "group": g,
                    "label": lab,
                    "american": tds[i + 1],
                    "implied_probability": implied(tds[i + 1]),
                    "source_url": url,
                }
            )
    return rows, groups


def pull():
    DATA.mkdir(parents=True, exist_ok=True)
    (DATA / ".gitignore").write_text("*\n", encoding="utf-8")
    rows, pages = [], {}
    for league, seasons, url_t in (("cfb", CFB_SEASONS, CFB_URL), ("nfl", NFL_SEASONS, NFL_URL)):
        for y in seasons:
            url = url_t.format(y=y)
            r, groups = parse(fetch(url), league, y, url)
            pages[f"{league}_{y}"] = {"rows": len(r), "groups": sorted(set(groups or []))}
            rows += r
    df = pd.DataFrame(rows)
    df.to_parquet(DATA / "futures_long.parquet", index=False)
    write_joins(df, pages)


def rejoin():
    inv = json.loads((DATA / "inventory.json").read_text(encoding="utf-8"))
    REQUESTS["count"] = inv["requests_made"]
    write_joins(pd.read_parquet(DATA / "futures_long.parquet"), inv["pages"])


def write_joins(df, pages):
    from source_pull_2026_11_16 import team_from_name

    df = df[~df["team_name"].isin(NOT_TEAMS)].copy()
    nfl_mask = df["league"].eq("nfl")
    df.loc[nfl_mask, "team"] = df.loc[nfl_mask, "team_name"].map(team_from_name)
    df.loc[~nfl_mask, "team"] = df.loc[~nfl_mask, "team_name"].map(lambda n: CFB_ALIAS.get(n, n))
    df.to_parquet(DATA / "futures_long.parquet", index=False)
    pre = preseason(df)
    pre.to_parquet(DATA / "preseason_close.parquet", index=False)
    inv = {"pages": pages, "requests_made": REQUESTS["count"], "joins": joins(pre)}
    inv["written_utc"] = datetime.now(UTC).isoformat(timespec="seconds")
    (DATA / "inventory.json").write_text(json.dumps(inv, indent=1, default=str), encoding="utf-8")
    print(json.dumps(inv["joins"], indent=1, default=str))


def preseason(df):
    out = []
    for (league, season), g in df.dropna(subset=["implied_probability"]).groupby(["league", "season"]):
        if league == "cfb":
            pre = g[g["group"].str.lower().str.startswith("preseason")]
            close_idx, open_idx = pre["column_index"].max(), pre["column_index"].min()
        else:
            reg = g[g["label"].str.fullmatch(r"Wk 1")]
            close_idx = open_idx = reg["column_index"].min()
        c = g[g["column_index"].eq(close_idx)][["team_name", "team", "label", "implied_probability"]]
        o = g[g["column_index"].eq(open_idx)][["team_name", "label", "implied_probability"]]
        c = c.merge(o, on="team_name", how="left", suffixes=("", "_open"))
        c["league"], c["season"] = league, season
        c["strength_percentile"] = c["implied_probability"].rank(pct=True)
        out.append(c)
    return pd.concat(out, ignore_index=True)


def joins(pre):
    res = {}
    nfl = pd.read_parquet(ROOT / "data" / "processed" / "game_features.parquet", columns=["season", "home_team", "away_team"])
    cfb = pd.read_parquet(ROOT / "data" / "processed" / "cfb_game_features.parquet", columns=["season", "home_team", "away_team"])
    for league, g in (("nfl", nfl), ("cfb", cfb)):
        teams = set(g["home_team"]) | set(g["away_team"])
        p = pre[pre["league"].eq(league)]
        listed = p.dropna(subset=["team"])
        hit = listed["team"].isin(teams)
        res[league] = {
            "seasons": sorted(int(s) for s in p["season"].unique()),
            "listed_by_season": p.groupby("season").size().to_dict(),
            "publication_point": p.groupby("season")["label"].first().to_dict(),
            "opening_point": p.groupby("season")["label_open"].first().to_dict(),
            "listed_teams_joined_to_repo_names": float(hit.sum() / max(len(p), 1)),
            "unmatched_names": sorted(set(p.loc[~p["team"].isin(teams), "team_name"])),
        }
        if league == "nfl":
            ts = pd.concat([g[["season", "home_team"]].rename(columns={"home_team": "team"}), g[["season", "away_team"]].rename(columns={"away_team": "team"})]).drop_duplicates()
            ts = ts[ts["season"].isin(p["season"].unique())]
            j = ts.merge(p[["season", "team"]].assign(hit=1), on=["season", "team"], how="left")
            res[league]["repo_team_seasons_with_futures"] = float(j["hit"].notna().mean())
    return res


def opener_ratings(f):
    out = []
    for season, s in f.groupby("season"):
        weeks = sorted(s["week"].unique())
        for w in weeks[1:]:
            e = s[(s["week"] < w) & s["spread_open"].notna()]
            teams = sorted(set(e["home_team"]) | set(e["away_team"]))
            idx = {t: i for i, t in enumerate(teams)}
            x = np.zeros((len(e), len(teams) + 1))
            x[np.arange(len(e)), e["home_team"].map(idx)] = 1.0
            x[np.arange(len(e)), e["away_team"].map(idx)] = -1.0
            x[:, -1] = 1.0 - e["neutral_site"].fillna(False).astype(float).to_numpy()
            coef = np.linalg.lstsq(x, e["spread_open"].to_numpy(float), rcond=None)[0]
            out.append(pd.DataFrame({"season": season, "week": w, "team": teams, "rating": coef[:-1]}))
    return pd.concat(out, ignore_index=True)


def ats_to_date(f):
    g = f.dropna(subset=["spread_open", "result"])
    g = g[g["result"].ne(g["spread_open"])]
    home = (g["result"] > g["spread_open"]).astype(float)
    long = pd.concat(
        [
            pd.DataFrame({"season": g["season"], "week": g["week"], "team": g["home_team"], "c": home}),
            pd.DataFrame({"season": g["season"], "week": g["week"], "team": g["away_team"], "c": 1 - home}),
        ]
    )
    return long


def team_terms(f, pre, ratings, long):
    rows = []
    for (season, week), r in ratings.groupby(["season", "week"]):
        p = pre[pre["season"].eq(season)][["team", "implied_probability"]].dropna()
        m = r.merge(p, on="team")
        if len(m) < 2:
            continue
        m["gap"] = m["implied_probability"].rank(pct=True) - m["rating"].rank(pct=True)
        prior = long[(long["season"] == season) & (long["week"] < week)].groupby("team")["c"].mean()
        m["ats"] = m["team"].map(prior)
        m["season"], m["week"] = season, week
        rows.append(m[["season", "week", "team", "gap", "ats"]])
    return pd.concat(rows, ignore_index=True)


def game_terms(d, tt):
    look = tt.set_index(["season", "week", "team"])
    side = {}
    for s in ("home", "away"):
        key = pd.MultiIndex.from_arrays([d["season"], d["week"], d[f"{s}_team"]])
        side[s] = look.reindex(key).reset_index(drop=True)
    both = side["home"]["gap"].notna() & side["away"]["gap"].notna()
    gap = np.where(both, side["home"]["gap"] - side["away"]["gap"], 0.0)
    ats = (side["home"]["ats"] - side["away"]["ats"]).to_numpy()
    ok = both.to_numpy() & np.isfinite(ats)
    resid = np.zeros(len(d))
    xa = np.column_stack([np.ones(ok.sum()), ats[ok]])
    b = np.linalg.lstsq(xa, gap[ok], rcond=None)[0]
    resid[ok] = gap[ok] - xa @ b
    return {"futures_gap": gap, "futures_gap_resid_ats": resid}, int(both.sum()), int(ok.sum()), b.tolist()


def cells(m36, name, res, seasons, draws, power_lane):
    spec = CANDIDATES[name]
    held = res["held_out"]
    early = res["halves"]["earlier"]["coefficient_per_sd"]
    late = res["halves"]["later"]["coefficient_per_sd"]
    verdict = (
        "It stood out from chance and goes to one pro-football check."
        if res["survivor"]
        else "No effect large enough for this test to see showed up."
    )
    out = []
    for k, units in zip(m36.SHORT, m36.METRICS, strict=True):
        out.append(
            {
                "name": f"futures_anchor_cfb_{name}_{units}",
                "description": (
                    f"College-first preseason futures anchor (Fodor, Patterson and Shank 2025), "
                    f"{name}: {spec['what']} Home minus away; declared sign {spec['hypothesis']}; "
                    "one offset-logistic coefficient on the leave-one-season-out college ridge "
                    f"cover probability graded at the opener, {len(seasons)} seasons. Changed picks "
                    f"{held['changed']}, record {held['arm_right']}-{held['ref_right']}; coefficient "
                    f"per SD {res['coefficient_per_sd']:+.4f}; season-half coefficients {early:+.4f} "
                    f"and {late:+.4f}; percentile {res['null_percentile'][k]:.1f} of {draws} "
                    "within-season shuffles; in-sample minus held-out "
                    f"{res['in_sample_minus_held_out'][k]:+.6f}."
                ),
                "source": LANE,
                "effect": round(held[k]["effect"], 6),
                "effect_units": units,
                "classification": "unresolved_below_power",
                "league": "cfb",
                "season_start": int(seasons[0]),
                "season_end": int(seasons[-1]),
                "family": FAMILY,
                "category": "market",
                "classification_evidence": (
                    "No resolved wrong sign with the whole interval on the wrong side, split-half "
                    "reliability not measured, and no positive control bounds an effect this "
                    f"small on the college add-one harness ({power_lane})."
                ),
                "notes": (
                    f"Family {FAMILY}: 2 candidates x 3 metrics = 6 recorded cells; 2 gating reads "
                    "(log-loss percentile in the matched null plus same sign in both season halves), "
                    "declared in the lane before outcomes were joined; survivor: "
                    f"{'yes' if res['survivor'] else 'no'}."
                ),
                "interval_low": round(held[k]["low"], 6),
                "interval_high": round(held[k]["high"], 6),
                "probability_positive": round(held[k]["pp"], 4),
                "sample_games": int(res["games"]),
                "sample_blocks": len(seasons),
                "plain_summary": (
                    f"Checks whether college teams that {spec['plain']} go on to beat the opening "
                    f"spread less often. {verdict}"
                ),
            }
        )
    return out


def screen():
    import xlg14_luck_family as u3

    started = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    u3.STREAM_LUCK = STREAM
    hp = u3.load_power()
    m36 = hp.M36
    d = hp.x14.cfb_frame()
    full = pd.read_parquet(ROOT / "data" / "processed" / "cfb_game_features.parquet")
    pre = pd.read_parquet(DATA / "preseason_close.parquet")
    pre = pre[pre["league"].eq("cfb")]
    ratings = opener_ratings(full)
    tt = team_terms(full, pre, ratings, ats_to_date(full))
    terms, both, ok, b = game_terms(d, tt)
    frame = pd.DataFrame(
        {
            "game_id": d["game_id"].to_numpy(),
            "season": d["season"].to_numpy(int),
            "home_covered": d["y"].to_numpy(float),
            m36.SERVED_COLUMN: expit(d["logit_p"].to_numpy(float)),
        }
    )
    for n in NAMES:
        frame[n] = terms[n]
    frame = frame[frame["season"].isin(list(CFB_SEASONS))].reset_index(drop=True)
    seasons = sorted(int(s) for s in frame["season"].unique())
    out = {"family": FAMILY, "games": len(frame), "both_teams_listed_and_rated": both, "with_ats": ok, "resid_coefficients": b, "seasons": seasons, "draws": hp.DRAWS, "seed": [hp.SEED, STREAM], "candidates": {}}
    batch, lines = [], []
    for index, n in enumerate(NAMES):
        res, st = u3.read_candidate(hp, frame, n, index)
        if index == 0:
            out["controls"] = u3.controls(m36, st)
        del st
        out["candidates"][n] = res
        batch.extend(cells(m36, n, res, seasons, hp.DRAWS, u3.POWER_LANE))
        h = res["held_out"]
        lines.append(
            f"{n}: games {res['games']} nonzero {res['nonzero_games']}; changed {h['changed']} "
            f"record {h['arm_right']}-{h['ref_right']}; acc {h['acc']['effect']:+.3f} "
            f"[{h['acc']['low']:+.3f},{h['acc']['high']:+.3f}] pp {h['acc']['pp']:.3f} "
            f"pct {res['null_percentile']['acc']:.1f}; ll {h['ll']['effect']:+.6f} "
            f"[{h['ll']['low']:+.6f},{h['ll']['high']:+.6f}] pp {h['ll']['pp']:.3f} "
            f"pct {res['null_percentile']['ll']:.1f}; in-sample gap "
            f"{res['in_sample_minus_held_out']['ll']:+.6f}; coef {res['coefficient_per_sd']:+.4f} "
            f"folds {res['folds_with_full_sign']}/{len(seasons)}; halves "
            f"{res['halves']['earlier']['coefficient_per_sd']:+.4f} / "
            f"{res['halves']['later']['coefficient_per_sd']:+.4f}; survivor {res['survivor']}"
        )
        print(lines[-1], f"{time.time() - started:.0f}s", flush=True)
    out["survivors"] = [n for n in NAMES if out["candidates"][n]["survivor"]]
    lines.append("controls: " + "; ".join(f"{k} acc {v['accuracy']:.3f} ll {v['log_loss']:.6f}" for k, v in out["controls"].items()))
    lines.append(f"survivors: {out['survivors']}; both listed+rated {both}, with ats {ok} of {len(frame)}")
    print(lines[-2], lines[-1], sep="\n", flush=True)
    (OUT / "summary.json").write_text(json.dumps(out, indent=1, default=str), encoding="utf-8")
    (OUT / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (OUT / "weak_signals_batch.json").write_text(json.dumps({"cells": batch}, indent=1), encoding="utf-8")


if __name__ == "__main__":
    {"pull": pull, "rejoin": rejoin, "screen": screen}[sys.argv[1]]()
