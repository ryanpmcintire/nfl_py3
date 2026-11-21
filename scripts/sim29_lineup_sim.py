import argparse
import json
import multiprocessing as mp
import re
import sys
import time
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "src"))
LEAD = REPO / "artifacts" / "lead106_injury_language"
OUT = REPO / "artifacts" / "sim29"
SNAPS = REPO / "data" / "players" / "forward_snaps" / "snap_counts_2026_before_week_11.parquet"
FIT = REPO / "artifacts" / "sim08_market_anchored" / "main" / "fit_2025.json"
OL_POS = {"T", "G", "C", "OT", "OG", "OL", "LT", "RT", "LG", "RG"}
OL_SLOTS = json.loads((REPO / "artifacts" / "sim12" / "ol_exit.json").read_text())["slots"]
CONTEXT_SEASON = 2025
BOOT = 2000
PLAYERS_RAW = REPO / "data" / "players" / "raw"
ABSENT_STATUS = {"RES", "CUT", "RET", "EXE", "TRD", "TRT", "TRC"}
ET = ZoneInfo("America/New_York")

_E = {}


def norm(s):
    return re.sub(r"[^a-z]", "", str(s).lower())


def opener_utc(first_gameday):
    d = pd.Timestamp(first_gameday).normalize()
    tue = d - pd.Timedelta(days=(d.dayofweek - 1) % 7)
    return (tue + pd.Timedelta(hours=12)).tz_localize(ET).tz_convert("UTC")


def roster_before(t_open):
    caps = [
        p
        for p in PLAYERS_RAW.iterdir()
        if p.name[:1].isdigit() and (p / "weekly_rosters.parquet").exists()
    ]
    caps = [
        p for p in caps if pd.to_datetime(p.name[:15], format="%Y%m%dT%H%M%S", utc=True) < t_open
    ]
    snap = max(caps, key=lambda p: p.name)
    w = pd.read_parquet(snap / "weekly_rosters.parquet")
    w = w[(w["season"] == 2026) & (w["game_type"] == "REG")].copy()
    last = w.groupby("team")["week"].transform("max")
    w = w[w["week"] == last]
    keys = w["full_name"].map(name_key)
    w["key"] = keys.map(lambda k: "".join(k))
    w["last"] = keys.map(lambda k: k[-1] if k else "")
    w["fi"] = keys.map(lambda k: k[0][:1] if k else "")
    return snap.name, w


SUFFIX = {"jr", "sr", "ii", "iii", "iv", "v"}


def name_key(s):
    toks = [re.sub(r"[^a-z]", "", t) for t in str(s).lower().split()]
    return [t for t in toks if t and t not in SUFFIX]


def roster_state(ros, team, pfr, gsis_ids, name):
    k = name_key(name)
    hit = ros[ros["pfr_id"] == pfr]
    if hit.empty and gsis_ids:
        hit = ros[ros["gsis_id"].isin(gsis_ids)]
    if hit.empty:
        hit = ros[ros["key"] == "".join(k)]
    if hit.empty and k:
        cand = ros[(ros["team"] == team) & (ros["last"] == k[-1])]
        if len(cand) > 1:
            cand = cand[cand["fi"] == k[0][:1]]
        if len(cand) == 1:
            hit = cand
    if hit.empty:
        return "unmatched"
    own = hit[hit["team"] == team]
    if own.empty:
        return "other_team"
    st = str(own["status"].iloc[0])
    return st if st in ABSENT_STATUS else "present"


def costs():
    lat = json.loads((REPO / "artifacts" / "sim10" / "latent_fit_2009_25.json").read_text())
    ol = json.loads((REPO / "artifacts" / "sim12" / "ol_exit.json").read_text())
    return float(lat["qb"]["backup_off_epa_effect"]), float(ol["epa_per_play_shift"])


def load_reads():
    s = pd.read_parquet(LEAD / "scored.parquet")
    s = s[s["season"] == 2026][["week", "team", "gsis_id", "full_name", "rp"]]
    s = s.rename(columns={"full_name": "player"})
    s["src"] = "sample"
    f9 = pd.read_parquet(LEAD / "forward_score" / "2026_week09_scored.parquet")
    f9 = f9.sort_values("minutes_before_kickoff").drop_duplicates(["gsis_id", "week"])
    f9 = f9[["week", "team", "gsis_id", "player", "rp"]].assign(src="forward")
    rows = []
    for p in sorted((LEAD / "forward").glob("2026_week10_*.json")):
        d = json.loads(p.read_text())
        for r in d["rows"]:
            if r["capture_utc"] <= r["kickoff_utc"]:
                rows.append(
                    (
                        int(d["week"]),
                        r["team"],
                        r["gsis_id"],
                        r["player"],
                        float(r["p_plays"]),
                        r["capture_utc"],
                    )
                )
    f10 = pd.DataFrame(rows, columns=["week", "team", "gsis_id", "player", "rp", "cap"])
    f10 = (
        f10.sort_values("cap").drop_duplicates(["gsis_id", "week"], keep="last").drop(columns="cap")
    )
    f10["src"] = "forward"
    return pd.concat([s, f9, f10], ignore_index=True)


def crosswalk():
    pg = pd.read_parquet(REPO / "artifacts" / "per03" / "played_by_gsis.parquet")
    a = pg[["gsis_id", "pfr_id"]].dropna()
    f9 = pd.read_parquet(LEAD / "forward_score" / "2026_week09_scored.parquet")
    b = f9[["gsis_id", "pfr_player_id"]].dropna().rename(columns={"pfr_player_id": "pfr_id"})
    return pd.concat([a, b]).drop_duplicates("gsis_id").set_index("gsis_id")["pfr_id"].to_dict()


def games_2026():
    src = sorted((REPO / "data" / "raw_schedules").glob("*/schedules.parquet"))[-1]
    g = pd.read_parquet(src)
    g = g[(g["season"] == 2026) & (g["game_type"] == "REG") & g["home_score"].notna()]
    g = g[
        [
            "game_id",
            "season",
            "week",
            "gameday",
            "home_team",
            "away_team",
            "home_score",
            "away_score",
            "spread_line",
        ]
    ].copy()
    led = pd.read_parquet(REPO / "artifacts" / "forward_ledger" / "ledger.parquet")[
        ["game_id", "card_line"]
    ]
    g = g.merge(led.drop_duplicates("game_id"), on="game_id", how="left")
    g["line"] = g["spread_line"].astype(float)
    g["open_line"] = -g["card_line"].astype(float)
    g["margin"] = g["home_score"].astype(float) - g["away_score"].astype(float)
    return g.dropna(subset=["line"])


def build(n_draws, weeks):
    qb_cost, ol_cost = costs()
    sn = pd.read_parquet(SNAPS)
    sn = sn[(sn["season"] == 2026) & (sn["game_type"] == "REG")].copy()
    sn["grp"] = np.where(
        sn["position"] == "QB", "QB", np.where(sn["position"].isin(OL_POS), "OL", "")
    )
    sn = sn[sn["grp"] != ""]
    reads = load_reads()
    pop = pd.read_parquet(LEAD / "population.parquet")
    pop = pop[pop["season"] == 2026][["week", "team", "gsis_id"]]
    xw = crosswalk()
    rev = {}
    for k, v in xw.items():
        rev.setdefault(v, []).append(k)
    reads["pfr_id"] = reads["gsis_id"].map(xw)
    reads["nm"] = reads["player"].map(norm)
    pop["pfr_id"] = pop["gsis_id"].map(xw)
    g = games_2026()
    opens = {int(w): opener_utc(d) for w, d in g.groupby("week")["gameday"].min().items()}
    g = g[g["week"].isin(weeks)]
    rosters = {w: roster_before(opens[w]) for w in sorted(g["week"].unique())}
    status_counts = {}
    out = []
    for gi, r in enumerate(g.itertuples()):
        side = {}
        complete = True
        ros = rosters[r.week][1]
        for t in (r.home_team, r.away_team):
            prior = sn[(sn["team"] == t) & (sn["week"] < r.week)]
            cur = sn[(sn["team"] == t) & (sn["week"] == r.week)]
            if prior.empty or cur.empty:
                complete = False
                break
            tot = (
                prior.groupby(["grp", "pfr_player_id", "player"])["offense_snaps"]
                .sum()
                .reset_index()
            )
            st = pd.concat(
                [
                    tot[tot["grp"] == "QB"].nlargest(1, "offense_snaps"),
                    tot[tot["grp"] == "OL"].nlargest(OL_SLOTS, "offense_snaps"),
                ]
            )
            played = cur.groupby("pfr_player_id")["offense_snaps"].sum()
            rw = reads[(reads["week"] == r.week) & (reads["team"] == t)]
            listed = set(pop[(pop["week"] == r.week) & (pop["team"] == t)]["pfr_id"].dropna())
            recs = []
            for p in st.itertuples():
                hit = rw[rw["pfr_id"] == p.pfr_player_id]
                if hit.empty:
                    hit = rw[rw["nm"] == norm(p.player)]
                gs = rev.get(p.pfr_player_id, [])
                state = roster_state(ros, t, p.pfr_player_id, gs, p.player)
                known_absent = state in ABSENT_STATUS or state == "other_team"
                if not hit.empty:
                    pr = float(hit["rp"].iloc[-1])
                elif known_absent:
                    pr = 0.0
                elif r.week <= 8 and p.pfr_player_id in listed:
                    complete = False
                    pr = np.nan
                else:
                    pr = 1.0
                out_real = float(played.get(p.pfr_player_id, 0.0)) <= 0.0
                recs.append((p.grp, p.player, pr, out_real, not hit.empty, known_absent, state))
            side[t] = recs
        if not complete:
            continue
        for t in side:
            for x in side[t]:
                key = f"{x[6]}|{'out' if x[3] else 'played'}|{'read' if x[4] else 'noread'}"
                status_counts[key] = status_counts.get(key, 0) + 1
        rng = np.random.default_rng(29_000 + gi)
        cfg = {}
        actual = []
        n_listed_read = 0
        out_vs_read = 0
        for t in (r.home_team, r.away_team):
            recs = side[t]
            pr = np.array([x[2] for x in recs])
            isqb = np.array([x[0] == "QB" for x in recs])
            dr = rng.random((n_draws, len(recs))) >= pr
            shift = qb_cost * (dr & isqb).sum(1) + ol_cost * (dr & ~isqb).sum(1)
            cfg[t] = np.round(shift, 10)
            ro = np.array([x[3] for x in recs])
            actual.append(round(qb_cost * (ro & isqb).sum() + ol_cost * (ro & ~isqb).sum(), 10))
            n_listed_read += sum(x[4] for x in recs)
            ka = np.array([x[5] for x in recs])
            out_vs_read += int(((pr >= 0.5) & ro & ~ka).sum())
        pairs = pd.Series(list(zip(cfg[r.home_team], cfg[r.away_team], strict=True))).value_counts()
        out.append(
            {
                "game_id": r.game_id,
                "week": int(r.week),
                "line": float(r.line),
                "open_line": float(r.open_line),
                "margin": float(r.margin),
                "configs": [list(k) for k in pairs.index],
                "freq": (pairs.to_numpy() / n_draws).tolist(),
                "actual": actual,
                "qb_out_raw": int(sum(x[3] for t in side for x in side[t] if x[0] == "QB")),
                "ol_out_raw": int(sum(x[3] for t in side for x in side[t] if x[0] == "OL")),
                "qb_out": int(
                    sum(x[3] and not x[5] for t in side for x in side[t] if x[0] == "QB")
                ),
                "ol_out": int(
                    sum(x[3] and not x[5] for t in side for x in side[t] if x[0] == "OL")
                ),
                "known_absent": int(sum(x[5] for t in side for x in side[t])),
                "known_absent_with_read": int(sum(x[5] and x[4] for t in side for x in side[t])),
                "out_vs_read": out_vs_read,
                "starters_with_read": n_listed_read,
                "seed": 290_000 + gi,
            }
        )
    meta = {
        "qb_cost": qb_cost,
        "ol_cost": ol_cost,
        "games_considered": len(g),
        "opener_utc": {int(w): str(opens[w]) for w in rosters},
        "roster_snapshot": {int(w): rosters[w][0] for w in rosters},
        "starter_status_counts": dict(sorted(status_counts.items())),
    }
    return out, meta


def init_worker():
    import sim08_market_anchored as ma

    _E["eng"] = ma.engine()
    _E["fit"] = json.loads(FIT.read_text())


def sim_game(task):
    import sim08_market_anchored as ma

    row, m = task
    eng, fit = _E["eng"], _E["fit"]
    xd = ma.solve_diff_only(fit, row["line"])
    s = eng.s
    todo = [(0.0, 0.0), tuple(row["actual"])] + [tuple(c) for c in row["configs"]]
    res = {}
    for c in todo:
        if c in res:
            continue
        rng = np.random.default_rng(row["seed"])
        xh = 0.5 * xd + c[0] / s
        xa = -0.5 * xd + c[1] / s
        rh = (0.5 * s * xh, 0.5 * s * xa)
        ra = (0.5 * s * xa, 0.5 * s * xh)
        v = np.empty(m)
        for k in range(m):
            hs, as_, _, _, _, _ = eng.game(rng, (rh, ra), CONTEXT_SEASON, float(row["week"]))
            v[k] = hs - as_
        res[c] = float(v.mean())
    e_read = float(sum(f * res[tuple(c)] for c, f in zip(row["configs"], row["freq"], strict=True)))
    return {
        "game_id": row["game_id"],
        "sim_base": res[(0.0, 0.0)],
        "sim_actual": res[tuple(row["actual"])],
        "sim_read": e_read,
        "s": float(s),
        "x_diff": float(xd),
        "n_configs": len(res),
    }


def boot(df, fn, rng):
    n = len(df)
    if n < 3:
        return None
    pt = fn(df)
    v = np.array([fn(df.iloc[rng.integers(0, n, n)]) for _ in range(BOOT)])
    v = v[np.isfinite(v)]
    return {
        "point": float(pt),
        "ci": [float(np.quantile(v, 0.025)), float(np.quantile(v, 0.975))],
        "probability_positive": float((v > 0).mean()),
        "n": int(n),
    }


def slope(d):
    x = d["S"].to_numpy()
    if x.std() == 0:
        return np.nan
    return float(np.polyfit(x, d["R"].to_numpy(), 1)[0])


def score(df):
    rng = np.random.default_rng(29)
    df["S"] = df["sim_actual"] - df["sim_read"]
    df["R_close"] = df["margin"] - df["line"]
    df["R"] = df["margin"] - df["open_line"]
    df["D"] = df["R"] - df["S"]
    allg = df
    close = df.assign(R=df["R_close"])
    df = df[df["open_line"].notna()]
    b1 = df[(df["qb_out"] + df["ol_out"]) > 0]
    b2 = df[df["out_vs_read"] > 0]
    buckets = {
        "games_all": len(allg),
        "by_week_all": allg.groupby("week").size().to_dict(),
        "starter_out_all": int(((allg["qb_out"] + allg["ol_out"]) > 0).sum()),
        "games": len(df),
        "by_week": df.groupby("week").size().to_dict(),
        "no_starter_out": int(((df["qb_out"] + df["ol_out"]) == 0).sum()),
        "starter_out": len(b1),
        "starter_out_raw": int(((df["qb_out_raw"] + df["ol_out_raw"]) > 0).sum()),
        "games_with_known_absent": int((df["known_absent"] > 0).sum()),
        "qb_out_games": int((df["qb_out"] > 0).sum()),
        "ol_out_games": int((df["ol_out"] > 0).sum()),
        "out_against_read": len(b2),
        "mean_abs_S": float(df["S"].abs().mean()),
        "sd_S": float(df["S"].std()),
        "sd_R": float(df["R"].std()),
        "corr_line_margin": float(np.corrcoef(df["line"], df["margin"])[0, 1]),
    }
    looks = {
        "slope_R_on_S": boot(df, slope, rng),
        "slope_Rclose_on_S_all_weeks": boot(close, slope, rng),
        "starter_out_R_minus_S": boot(b1, lambda d: d["D"].mean(), rng) if len(b1) > 1 else None,
        "out_against_read_R_minus_S": boot(b2, lambda d: d["D"].mean(), rng)
        if len(b2) > 1
        else None,
    }
    side = {
        "starter_out_mean_S": float(b1["S"].mean()) if len(b1) else None,
        "starter_out_mean_R": float(b1["R"].mean()) if len(b1) else None,
        "out_against_read_mean_S": float(b2["S"].mean()) if len(b2) else None,
        "out_against_read_mean_R": float(b2["R"].mean()) if len(b2) else None,
        "mean_sim_actual_minus_base": float((df["sim_actual"] - df["sim_base"]).mean()),
    }
    return buckets, looks, side


def main():
    p = argparse.ArgumentParser(
        description="SIM-29 simulated lineup draws from the stored wording reads"
    )
    p.add_argument("--weeks", default="2-10")
    p.add_argument("--draws", type=int, default=200)
    p.add_argument("--sims", type=int, default=200)
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--prep-only", action="store_true", dest="prep_only")
    p.add_argument("--tag", default="lineup_sim")
    a = p.parse_args()
    lo, hi = (int(v) for v in a.weeks.split("-"))
    rows, meta = build(a.draws, list(range(lo, hi + 1)))
    if a.limit:
        rows = rows[: a.limit]
    print(json.dumps({**meta, "games_complete": len(rows), "free_gb": None}), flush=True)
    if a.prep_only:
        print(json.dumps(rows[:3], indent=1))
        return
    t0 = time.time()
    ctx = mp.get_context("spawn")
    with ctx.Pool(a.workers, initializer=init_worker) as pool:
        res = pool.map(sim_game, [(r, a.sims) for r in rows], chunksize=1)
    df = (
        pd.DataFrame(rows)
        .drop(columns=["configs", "freq", "actual"])
        .merge(pd.DataFrame(res), on="game_id")
    )
    buckets, looks, side = score(df)
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT / f"{a.tag}_games.parquet", index=False)
    out = {
        **meta,
        "draws_per_game": a.draws,
        "sims_per_lineup": a.sims,
        "context_season": CONTEXT_SEASON,
        "seconds": time.time() - t0,
        "buckets": buckets,
        "looks": looks,
        "side": side,
    }
    (OUT / f"{a.tag}.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
