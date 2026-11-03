import argparse
import hashlib
import json
import re
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import forward_terms_week as FTW
import lead99_llm_read as L99

ROOT = L99.ROOT
ART = ROOT / "artifacts" / "lead106_injury_language"
PROMPT = ROOT / "scripts" / "lead106_prompt.md"
SEASON = 2026
FIRST_WEEK = 1
LAST_WEEK = 8
STRATA = ("Out", "Doubtful", "Questionable", "none")
PER_STRATUM_BUDGET = 100
BATCH_ROWS = 40
BOOTSTRAP_REPS = 2000
HEADLINE_WINDOW_DAYS = 7
THURSDAY = 3
STAMP = "%Y%m%dT%H%M%SZ"
FORWARD = ART / "forward"


def sha(b):
    return hashlib.sha256(b).hexdigest()


def seed_value():
    return int(sha(PROMPT.read_bytes())[:8], 16)


def norm_name(s):
    v = re.sub(r"[^a-z ]", "", str(s).lower())
    v = re.sub(r"(?<![a-z])(jr|sr|ii|iii|iv|v)(?![a-z])", "", v)
    return re.sub(r"\s+", "-", v.strip())


def build_population():
    d = sorted((ROOT / "data" / "raw" / "nflverse_injuries").iterdir())[-1]
    x = pd.read_parquet(d / "injuries.parquet")
    x = x[(x["season"] == SEASON) & x["week"].between(FIRST_WEEK, LAST_WEEK)].copy()
    x["week"] = x["week"].astype(int)
    x = x.drop_duplicates(["gsis_id", "week"], keep="last")
    x["designation"] = x["report_status"].fillna("none")
    s = pd.read_parquet(
        ROOT / "data" / "players" / "forward_snaps" / "snap_counts_2026_before_week_09.parquet"
    )
    s = s[(s["season"] == SEASON) & s["week"].between(FIRST_WEEK, LAST_WEEK)].copy()
    s["key"] = s["player"].map(norm_name)
    s["total"] = s[["offense_snaps", "defense_snaps", "st_snaps"]].fillna(0).sum(axis=1)
    s["share"] = s[["offense_pct", "defense_pct"]].fillna(0).max(axis=1)
    s = s.groupby(["key", "team", "week"], as_index=False).agg(
        total=("total", "sum"), share=("share", "max")
    )
    x["key"] = x["full_name"].map(norm_name)
    x = x.merge(s, on=["key", "team", "week"], how="left")
    x["matched"] = x["total"].notna()
    x["played"] = (x["total"].fillna(0) > 0).astype(int)
    x["snap_share"] = x["share"].fillna(0.0)
    g = pd.read_parquet(
        ROOT / "data" / "processed" / "game_features.parquet",
        columns=["season", "week", "home_team", "away_team", "kickoff"],
    )
    g = g[g["season"] == SEASON]
    k = pd.concat(
        [
            g[["week", "home_team", "kickoff"]].rename(columns={"home_team": "team"}),
            g[["week", "away_team", "kickoff"]].rename(columns={"away_team": "team"}),
        ]
    )
    k = k.groupby(["team", "week"], as_index=False)["kickoff"].min()
    x = x.merge(k, on=["team", "week"], how="left")
    nd = sorted((ROOT / "data" / "raw" / "injury_news").iterdir())[-1]
    c = pd.read_parquet(nd / "current.parquet")
    c = c[c["injury_relevant"]]
    heads = []
    for _, a in x.iterrows():
        if pd.isna(a["kickoff"]) or not a["key"]:
            heads.append([])
            continue
        lo = a["kickoff"] - pd.Timedelta(days=HEADLINE_WINDOW_DAYS)
        r = c[
            (c["lastmod"] < a["kickoff"])
            & (c["lastmod"] >= lo)
            & c["slug"].str.contains(a["key"], regex=False)
        ]
        heads.append(
            [
                str(h).replace(a["key"].replace("-", " "), "the player")
                for h in r.sort_values("lastmod")["headline_guess"]
            ]
        )
    x["headlines"] = heads
    x["has_headline"] = x["headlines"].map(len) > 0
    return x.reset_index(drop=True)


def row_text(a):
    inj = a["report_primary_injury"]
    if pd.isna(inj):
        inj = a["practice_primary_injury"]
    parts = [
        f"position {a['position']}",
        f"official game designation: {a['designation']}",
        f"injury: {inj if pd.notna(inj) else 'not listed'}",
    ]
    if pd.notna(a["report_secondary_injury"]):
        parts.append(f"also: {a['report_secondary_injury']}")
    ps = a["practice_status"]
    parts.append(f"practice: {ps if pd.notna(ps) else 'not listed'}")
    if a["headlines"]:
        parts.append("headlines: " + " | ".join(a["headlines"]))
    return "; ".join(parts)


def draw_sample(x):
    rng = np.random.default_rng(seed_value())
    pool = x[x["week"] > FIRST_WEEK]
    take = set(pool.index[pool["has_headline"]])
    for st in STRATA:
        idx = [i for i in pool.index[pool["designation"] == st] if i not in take]
        have = len([i for i in take if pool.at[i, "designation"] == st])
        n = max(0, PER_STRATUM_BUDGET - have)
        take.update(rng.choice(idx, size=min(n, len(idx)), replace=False).tolist())
    s = x.loc[sorted(take)].copy()
    s = s.iloc[rng.permutation(len(s))].reset_index(drop=True)
    s["rid"] = [f"r{i:04d}" for i in range(len(s))]
    s["text"] = s.apply(row_text, axis=1)
    return s


def cmd_freeze(_):
    ART.mkdir(parents=True, exist_ok=True)
    x = build_population()
    s = draw_sample(x)
    s.drop(columns=["headlines"]).to_parquet(ART / "sample.parquet")
    x.drop(columns=["headlines"]).to_parquet(ART / "population.parquet")
    fz = {
        "prompt_sha256": sha(PROMPT.read_bytes()),
        "model": L99.MODEL,
        "tools": "",
        "seed": seed_value(),
        "per_stratum_budget": PER_STRATUM_BUDGET,
        "batch_rows": BATCH_ROWS,
        "population_rows": len(x),
        "sample_rows": len(s),
        "sample_text_sha256": sha("\n".join(s["rid"] + " " + s["text"]).encode()),
        "matched_snap_rate": float(x["matched"].mean()),
        "headline_rows": int(x["has_headline"].sum()),
    }
    (ART / "frozen.json").write_text(json.dumps(fz, indent=1))
    print(json.dumps(fz, indent=1))
    print(s["designation"].value_counts().to_dict())
    print(x.groupby("designation")[["matched", "played", "snap_share"]].mean())


def cmd_run(_):
    L99.TOOLS = ""
    fz = json.loads((ART / "frozen.json").read_text())
    assert fz["prompt_sha256"] == sha(PROMPT.read_bytes())
    s = pd.read_parquet(ART / "sample.parquet")
    (ART / "raw").mkdir(exist_ok=True)
    cwd = ART / "cwd"
    cwd.mkdir(exist_ok=True)
    prompt = PROMPT.read_text()
    for b in range(0, len(s), BATCH_ROWS):
        out = ART / "raw" / f"batch_{b // BATCH_ROWS:03d}.json"
        if out.exists():
            continue
        chunk = s.iloc[b : b + BATCH_ROWS]
        packet = json.dumps([{"id": r.rid, "text": r.text} for r in chunk.itertuples()], indent=1)
        env, text = L99.call_claude(prompt, packet, cwd)
        out.write_text(json.dumps({"envelope": env, "text": text}))
        print("done", out.name, flush=True)


def parse_reads():
    rows = {}
    for f in sorted((ART / "raw").glob("batch_*.json")):
        t = json.loads(f.read_text())["text"]
        for o in json.loads(t[t.index("[") : t.rindex("]") + 1]):
            rows[o["id"]] = (float(o["p_plays"]), float(o["snap_share"]))
    return rows


def ladder_tables(pop, keys):
    out = {}
    for w in range(FIRST_WEEK + 1, LAST_WEEK + 1):
        prior = pop[pop["week"] < w]
        agg = {"n": ("played", "size"), "k": ("played", "sum"), "m": ("snap_share", "mean")}
        out[w] = (prior.groupby(keys).agg(**agg), prior.groupby("designation").agg(**agg))
    return out


def ladder_pred(row, tab, keys):
    g, gd = tab[row["week"]]
    key = tuple(row[k] for k in keys) if len(keys) > 1 else row[keys[0]]
    a = g.loc[key] if key in g.index else gd.loc[row["designation"]]
    return (a["k"] + 1) / (a["n"] + 2), a["m"]


def wmean(v, w):
    return float(np.sum(v * w) / np.sum(w))


def boot(delta, w, cl, rng):
    ids = pd.factorize(cl)[0]
    n = ids.max() + 1
    by = [np.where(ids == i)[0] for i in range(n)]
    est = []
    for _ in range(BOOTSTRAP_REPS):
        pick = rng.integers(0, n, n)
        ix = np.concatenate([by[i] for i in pick])
        est.append(wmean(delta[ix], w[ix]))
    return np.array(est)


def summarize(name, delta, w, cl, rng):
    est = boot(delta, w, cl, rng)
    return {
        "metric": name,
        "estimate": wmean(delta, w),
        "unweighted": float(delta.mean()),
        "low": float(np.quantile(est, 0.025)),
        "high": float(np.quantile(est, 0.975)),
        "probability_positive": float((est > 0).mean()),
    }


def cmd_score(_):
    pop = pd.read_parquet(ART / "population.parquet")
    s = pd.read_parquet(ART / "sample.parquet")
    reads = parse_reads()
    s = s[s["rid"].isin(reads)].copy()
    n = len(s)
    clip = 1.0 / (n + 2)
    s["rp"] = s["rid"].map(lambda r: min(max(reads[r][0], clip), 1 - clip))
    s["rs"] = s["rid"].map(lambda r: min(max(reads[r][1], 0.0), 1.0))
    ladders = {
        "designation": ["designation"],
        "designation_x_practice": ["designation", "practice_status"],
    }
    pop = pop.copy()
    pop["practice_status"] = pop["practice_status"].fillna("none")
    s["practice_status"] = s["practice_status"].fillna("none")
    popn = pop[pop["week"] > FIRST_WEEK].groupby("designation").size()
    samn = s.groupby("designation").size()
    s["w"] = s["designation"].map(lambda d: popn[d] / samn[d])
    y = s["played"].to_numpy()
    sh = s["snap_share"].to_numpy()
    res = {"n": n, "clip": clip}
    for nm, keys in ladders.items():
        tab = ladder_tables(pop, keys)
        pr = [ladder_pred(r, tab, keys) for _, r in s.iterrows()]
        s[f"lp_{nm}"] = [min(max(p[0], clip), 1 - clip) for p in pr]
        s[f"ls_{nm}"] = [p[1] for p in pr]
    s.to_parquet(ART / "scored.parquet")

    def ll(p):
        return -(y * np.log(p) + (1 - y) * np.log(1 - p))

    def br(p):
        return (p - y) ** 2

    w = s["w"].to_numpy()
    cl = (s["team"] + s["week"].astype(str)).to_numpy()
    out = []
    for nm in ladders:
        rng = np.random.default_rng(seed_value())
        lp = s[f"lp_{nm}"].to_numpy()
        ls = s[f"ls_{nm}"].to_numpy()
        rp = s["rp"].to_numpy()
        rs = s["rs"].to_numpy()
        subsets = (("all", np.ones(n, bool)), ("headline_rows", s["has_headline"].to_numpy()))
        for sub, mask in subsets:
            if mask.sum() < 2:
                continue
            cells = (
                ("log_loss_improvement", ll(lp) - ll(rp)),
                ("brier_improvement", br(lp) - br(rp)),
                ("mae_improvement", np.abs(ls - sh) - np.abs(rs - sh)),
            )
            for metric, d in cells:
                r = summarize(metric, d[mask], w[mask], cl[mask], rng)
                r.update(baseline=nm, subset=sub, rows=int(mask.sum()))
                out.append(r)
        even = (s["week"] % 2 == 0).to_numpy()
        for hv, mk in (("even_weeks", even), ("odd_weeks", ~even)):
            d = (ll(lp) - ll(rp))[mk]
            out.append(
                {
                    "metric": "log_loss_improvement",
                    "baseline": nm,
                    "subset": hv,
                    "rows": int(mk.sum()),
                    "estimate": wmean(d, w[mk]),
                }
            )
    res["cells"] = out
    res["by_designation"] = {
        d: {
            "rows": int((s["designation"] == d).sum()),
            "played_rate": float(s.loc[s["designation"] == d, "played"].mean()),
            "mean_read_p": float(s.loc[s["designation"] == d, "rp"].mean()),
        }
        for d in STRATA
    }
    (ART / "score.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


def roster_map():
    f = sorted((ROOT / "data" / "players" / "raw").glob("*/weekly_rosters.parquet"))[-1]
    r = pd.read_parquet(f, columns=["season", "week", "team", "status", "gsis_id", "pfr_id"])
    r = r[r["season"] == SEASON]
    ids = r.dropna(subset=["pfr_id"]).drop_duplicates("gsis_id", keep="last")
    return r, dict(zip(ids["gsis_id"], ids["pfr_id"], strict=True))


def id_outcomes(pop):
    r, pmap = roster_map()
    s = pd.read_parquet(
        ROOT / "data" / "players" / "forward_snaps" / "snap_counts_2026_before_week_09.parquet"
    )
    s = s[(s["season"] == SEASON) & s["week"].between(FIRST_WEEK, LAST_WEEK)].copy()
    s["total"] = s[["offense_snaps", "defense_snaps", "st_snaps"]].fillna(0).sum(axis=1)
    s["share"] = s[["offense_pct", "defense_pct"]].fillna(0).max(axis=1)
    s = s.groupby(["pfr_player_id", "week"], as_index=False).agg(
        total=("total", "sum"), share=("share", "max"), st=("st_snaps", "sum")
    )
    teamweeks = set(
        map(
            tuple,
            pd.read_parquet(
                ROOT
                / "data"
                / "players"
                / "forward_snaps"
                / "snap_counts_2026_before_week_09.parquet",
                columns=["team", "week", "season"],
            )
            .query("season == @SEASON")[["team", "week"]]
            .drop_duplicates()
            .to_numpy(),
        )
    )
    rw = r.drop_duplicates(["gsis_id", "week"], keep="last").set_index(["gsis_id", "week"])[
        "status"
    ]
    x = pop.drop(columns=["total", "share", "matched", "played", "snap_share"]).copy()
    x["pfr"] = x["gsis_id"].map(pmap)
    x = x.merge(s, left_on=["pfr", "week"], right_on=["pfr_player_id", "week"], how="left")
    x["rost"] = [rw.get((g, w)) for g, w in zip(x["gsis_id"], x["week"], strict=True)]
    cat = []
    for a in x.itertuples():
        if (a.team, a.week) not in teamweeks:
            cat.append("unverified_team_week_not_in_snaps")
        elif pd.notna(a.total) and a.total > 0:
            cat.append("played")
        elif pd.notna(a.total):
            cat.append("in_snaps_zero_snaps")
        elif a.designation == "Out":
            cat.append("out_absent")
        elif pd.isna(a.pfr):
            cat.append("unverified_no_pfr_id")
        elif pd.notna(a.rost):
            cat.append("absent_rostered_" + str(a.rost))
        else:
            cat.append("unverified_absent_not_on_roster_week")
    x["category"] = cat
    x["verified"] = ~x["category"].str.startswith("unverified")
    x["played"] = (x["category"] == "played").astype(int)
    x["snap_share"] = np.where(x["played"] == 1, x["share"].fillna(0.0), 0.0)
    return x


def cmd_ids(_):
    pop = pd.read_parquet(ART / "population.parquet")
    pop = id_outcomes(pop)
    s0 = pd.read_parquet(ART / "sample.parquet")[["rid", "gsis_id", "week"]]
    rep = {
        "overall_verified_rate": float(pop["verified"].mean()),
        "by_designation_verified_rate": pop.groupby("designation")["verified"].mean().to_dict(),
        "category_by_designation": {
            d: g["category"].value_counts().to_dict() for d, g in pop.groupby("designation")
        },
    }
    ver = pop[pop["verified"]].copy()
    ver["practice_status"] = ver["practice_status"].fillna("none")
    s = s0.merge(ver.drop(columns=["rid"], errors="ignore"), on=["gsis_id", "week"], how="inner")
    reads = parse_reads()
    s = s[s["rid"].isin(reads)].copy()
    n = len(s)
    clip = 1.0 / (n + 2)
    s["rp"] = s["rid"].map(lambda r: min(max(reads[r][0], clip), 1 - clip))
    s["rs"] = s["rid"].map(lambda r: min(max(reads[r][1], 0.0), 1.0))
    popn = ver[ver["week"] > FIRST_WEEK].groupby("designation").size()
    samn = s.groupby("designation").size()
    s["w"] = s["designation"].map(lambda d: popn[d] / samn[d])
    y = s["played"].to_numpy()
    sh = s["snap_share"].to_numpy()
    w = s["w"].to_numpy()
    cl = (s["team"] + s["week"].astype(str)).to_numpy()
    for nm, keys in (
        ("designation", ["designation"]),
        ("designation_x_practice", ["designation", "practice_status"]),
    ):
        tab = ladder_tables(ver, keys)
        pr = [ladder_pred(r, tab, keys) for _, r in s.iterrows()]
        s[f"lp_{nm}"] = [min(max(p[0], clip), 1 - clip) for p in pr]
        s[f"ls_{nm}"] = [p[1] for p in pr]

    def ll(p):
        return -(y * np.log(p) + (1 - y) * np.log(1 - p))

    def br(p):
        return (p - y) ** 2

    rp = s["rp"].to_numpy()
    rs = s["rs"].to_numpy()
    out = []
    even = (s["week"] % 2 == 0).to_numpy()
    designs = s["designation"].to_numpy()
    for nm in ("designation", "designation_x_practice"):
        lp = s[f"lp_{nm}"].to_numpy()
        ls = s[f"ls_{nm}"].to_numpy()
        rng = np.random.default_rng(seed_value())
        subsets = [("all", np.ones(n, bool)), ("headline_rows", s["has_headline"].to_numpy())]
        subsets += [(d, designs == d) for d in STRATA]
        for sub, mask in subsets:
            if mask.sum() < 2:
                continue
            cells = (
                ("log_loss_improvement", ll(lp) - ll(rp)),
                ("brier_improvement", br(lp) - br(rp)),
                ("mae_improvement", np.abs(ls - sh) - np.abs(rs - sh)),
            )
            for metric, d in cells:
                r = summarize(metric, d[mask], w[mask], cl[mask], rng)
                r.update(baseline=nm, subset=sub, rows=int(mask.sum()))
                out.append(r)
        for hv, mk in (("even_weeks", even), ("odd_weeks", ~even)):
            out.append(
                {
                    "metric": "log_loss_improvement",
                    "baseline": nm,
                    "subset": hv,
                    "rows": int(mk.sum()),
                    "estimate": wmean((ll(lp) - ll(rp))[mk], w[mk]),
                }
            )
    rep["sample_verified_rows"] = n
    rep["sample_rows_total"] = len(s0)
    rep["sample_by_designation"] = {
        d: {
            "rows": int((s["designation"] == d).sum()),
            "played_rate": float(s.loc[s["designation"] == d, "played"].mean()),
            "mean_read_p": float(s.loc[s["designation"] == d, "rp"].mean()),
        }
        for d in STRATA
    }
    rep["cells"] = out
    (ART / "score_ids.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep, indent=1))


def live_inputs(week, view):
    inj_dir = sorted((ROOT / "data" / "raw" / "nflverse_injuries").iterdir())[-1]
    capture = pd.Timestamp(datetime.strptime(inj_dir.name, "%Y%m%dT%H%M%SZ").replace(tzinfo=UTC))
    x = pd.read_parquet(inj_dir / "injuries.parquet")
    x = x[(x["season"] == SEASON) & (x["week"] == week)].copy()
    x["week"] = x["week"].astype(int)
    x = x.drop_duplicates(["gsis_id", "week"], keep="last")
    x["designation"] = x["report_status"].fillna("none")
    g = pd.read_parquet(
        ROOT / "data" / "processed" / "game_features.parquet",
        columns=["season", "week", "home_team", "away_team", "kickoff"],
    )
    g = g[(g["season"] == SEASON) & (g["week"] == week)].copy()
    g["kickoff"] = pd.to_datetime(g["kickoff"], utc=True)
    g["thursday"] = g["kickoff"].dt.tz_convert("America/New_York").dt.weekday == THURSDAY
    g = g[g["thursday"] if view == "thursday" else ~g["thursday"]]
    g["game_id"] = g["away_team"] + "@" + g["home_team"]
    k = pd.concat(
        [
            g[["home_team", "game_id", "kickoff"]].rename(columns={"home_team": "team"}),
            g[["away_team", "game_id", "kickoff"]].rename(columns={"away_team": "team"}),
        ]
    )
    x = x.merge(k, on="team", how="inner")
    x["key"] = x["full_name"].map(norm_name)
    nd = sorted((ROOT / "data" / "raw" / "injury_news").iterdir())[-1]
    c = pd.read_parquet(nd / "current.parquet")
    c = c[c["injury_relevant"] & (c["lastmod"] <= capture.tz_convert(None).tz_localize("UTC"))]
    heads = []
    for _, a in x.iterrows():
        lo = a["kickoff"] - pd.Timedelta(days=HEADLINE_WINDOW_DAYS)
        r = c[
            (c["lastmod"] < a["kickoff"])
            & (c["lastmod"] >= lo)
            & bool(a["key"])
            & c["slug"].str.contains(a["key"], regex=False)
        ]
        heads.append(
            [
                str(h).replace(a["key"].replace("-", " "), "the player")
                for h in r.sort_values("lastmod")["headline_guess"]
            ]
        )
    x["headlines"] = heads
    x = x.reset_index(drop=True)
    x["rid"] = [f"f{i:04d}" for i in range(len(x))]
    x["text"] = [row_text(r) for _, r in x.iterrows()]
    return x, capture, g


def cmd_forward(a):
    fz = json.loads((ART / "frozen.json").read_text())
    if fz["prompt_sha256"] != sha(PROMPT.read_bytes()) or fz["model"] != L99.MODEL:
        raise SystemExit("prompt or model differs from frozen enrollment")
    week = FTW.resolve_week(a.week)
    x, capture, g = live_inputs(week, a.view)
    now = datetime.now(UTC)
    started = x["kickoff"] <= pd.Timestamp(now)
    counts = x.groupby("game_id").size().reindex(g["game_id"], fill_value=0)
    kick = g.set_index("game_id")["kickoff"]
    print(
        json.dumps(
            {
                "week": week,
                "view": a.view,
                "capture_utc": capture.strftime(STAMP),
                "players": len(x),
                "per_game": {
                    gid: {
                        "players": int(n),
                        "kickoff_utc": kick[gid].strftime(STAMP),
                        "kicked_off": bool(kick[gid] <= pd.Timestamp(now)),
                    }
                    for gid, n in counts.items()
                },
            },
            indent=1,
        )
    )
    if a.dry:
        return
    x = x[~started].reset_index(drop=True)
    out = FORWARD / f"{SEASON}_week{week:02d}_{a.view}_{capture.strftime(STAMP)}.json"
    if out.exists():
        print("exists", out)
        return
    if x.empty:
        print("no players before kickoff")
        return
    FORWARD.mkdir(parents=True, exist_ok=True)
    L99.TOOLS = ""
    cwd = ART / "cwd"
    cwd.mkdir(exist_ok=True)
    prompt = PROMPT.read_text()
    reads = {}
    for b in range(0, len(x), BATCH_ROWS):
        chunk = x.iloc[b : b + BATCH_ROWS]
        packet = json.dumps([{"id": r.rid, "text": r.text} for r in chunk.itertuples()], indent=1)
        _, text = L99.call_claude(prompt, packet, cwd)
        for o in json.loads(text[text.index("[") : text.rindex("]") + 1]):
            reads[o["id"]] = (float(o["p_plays"]), float(o["snap_share"]))
    recorded = datetime.now(UTC)
    rows = []
    for r in x.itertuples():
        if r.rid not in reads or r.kickoff <= pd.Timestamp(recorded):
            continue
        rows.append(
            {
                "gsis_id": r.gsis_id,
                "player": r.full_name,
                "team": r.team,
                "game_id": r.game_id,
                "designation": r.designation,
                "p_plays": reads[r.rid][0],
                "snap_share": reads[r.rid][1],
                "capture_utc": capture.strftime(STAMP),
                "kickoff_utc": r.kickoff.strftime(STAMP),
            }
        )
    rec = {
        "season": SEASON,
        "week": week,
        "view": a.view,
        "capture_utc": capture.strftime(STAMP),
        "recorded_utc": recorded.strftime(STAMP),
        "prompt_sha256": fz["prompt_sha256"],
        "model": L99.MODEL,
        "rows": rows,
    }
    out.write_text(json.dumps(rec, indent=1))
    print(out, len(rows))


def main():
    p = argparse.ArgumentParser()
    sp = p.add_subparsers(dest="cmd", required=True)
    for n, f in (("freeze", cmd_freeze), ("run", cmd_run), ("score", cmd_score), ("ids", cmd_ids)):
        sp.add_parser(n).set_defaults(fn=f)
    fp = sp.add_parser("forward")
    fp.add_argument("--week", default="auto")
    fp.add_argument("--view", choices=["thursday", "sunday"], required=True)
    fp.add_argument("--dry", action="store_true")
    fp.set_defaults(fn=cmd_forward)
    a = p.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
