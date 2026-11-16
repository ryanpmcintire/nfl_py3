import argparse
import hashlib
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import forward_ledger as FL
import mod25_wf_forward as mwf
import prediction_market_gap as PMG

from nfl_ats.clv import LIVE_CAPTURE_KIND, load_decision_quotes
from nfl_ats.pick_probability import MARKET_MOVE_FEATURE_SUNDAY, market_move_toward_home
from nfl_ats.sharp_book_movement_features import sharp_book_movement_features

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts" / "exchange_move"
FROZEN = OUT / "frozen.json"
DATA = ROOT / "data"
LANE = "docs/lanes/exchange-move-2026-11-16.md"
ARMS = ("beside", "replace")
MOVE = "market_move_toward_home"
SEASON = PMG.SEASON
FIRST_WEEK, LAST_WEEK = PMG.FIRST_WEEK, PMG.LAST_WEEK
STAMP = PMG.STAMP
sig, logit, ll = PMG.sig, PMG.logit, PMG.ll


def active():
    act = json.loads(PMG.ACTIVE.read_text())
    art = ROOT / "artifacts" / act["artifact"]
    meta = json.loads((art / "metadata.json").read_text())
    coef = json.loads((art / "coefficients.json").read_text())["coefficients"]
    folds = {int(k): float(v[MOVE]) for k, v in meta["fold_coefficients"].items()}
    return act, art, folds, float(coef["move_toward_home"])


def exchange_moves(g):
    r = PMG.residuals()
    g = g.reset_index(drop=True)
    prices = PMG.venue_prices(dict(zip(g["game_id"], g["home_team"], strict=True)))
    tue = PMG.price_at(prices, g, "T_tue")
    dl = PMG.price_at(prices, g, "T_deadline")
    res = pd.DataFrame(index=pd.Index(g["game_id"], name="game_id"))
    for v in PMG.VENUES:
        a, b = tue[v].reindex(res.index), dl[v].reindex(res.index)
        ok = (a.notna() & b.notna()).to_numpy()
        mv = PMG.spread_equiv(b.fillna(0.5), r) - PMG.spread_equiv(a.fillna(0.5), r)
        res[f"x_{v}"] = np.where(ok, mv, np.nan)
    res["x"] = res[[f"x_{v}" for v in PMG.VENUES]].mean(axis=1)
    return res


def history():
    act, art, folds, _ = active()
    pg = pd.read_parquet(art / "per_game.parquet")
    pg = pg[pg["season"].isin(PMG.FIT_SEASONS)]
    gf = pd.read_parquet(
        DATA / "processed" / "game_features.parquet", columns=["game_id", "kickoff"]
    )
    g = PMG.stamps(pg.merge(gf, on="game_id"))
    h = g.set_index("game_id").join(exchange_moves(g))
    h["book_move"] = h[MOVE].astype(float)
    h["book_available"] = h["market_move_available"].astype(float)
    h["c_move"] = h["season"].map(folds)
    h["off_beside"] = logit(h["out_of_season_home_probability"].to_numpy(float))
    h["off_replace"] = h["off_beside"] - h["c_move"] * h["book_move"]
    return h, act


def summary(h, label):
    out = {}
    for s, grp in h.groupby("season"):
        out[int(s)] = {"games": len(grp)} | {
            c: int(grp[c].notna().sum()) for c in ("x_kalshi", "x_polymarket", "x")
        }
    d = h.dropna(subset=["x"])
    d = d[d["book_available"] > 0]
    dist = {
        "mean_x": float(h["x"].mean()),
        "sd_x": float(h["x"].std()),
        "share_x_nonzero": float((h["x"].dropna().abs() > 0).mean()),
        "corr_with_book_move": float(np.corrcoef(d["x"], d["book_move"])[0, 1]),
        "corr_games": len(d),
        "share_book_zero": float((d["book_move"] == 0).mean()),
    }
    print(label, json.dumps(out), json.dumps(dist))
    return {"coverage": out, "distribution": dist}


def grade(ps, pt, y, blk):
    d = ll(ps, y) - ll(pt, y)
    bs = PMG.boot(d, blk, np.random.default_rng(PMG.SEED))
    return {
        "games": len(y),
        "served_side_record": PMG.side_record(ps, y),
        "term_side_record": PMG.side_record(pt, y),
        "changed_pick_record": PMG.record(pt, ps, y),
        "mean_log_loss": {"served": float(ll(ps, y).mean()), "term": float(ll(pt, y).mean())},
        "log_loss_gain": float(d.mean()),
        "ci": [float(np.quantile(bs, 0.025)), float(np.quantile(bs, 0.975))],
        "probability_positive": float((bs > 0).mean()),
        "reliability_term": PMG.reliability(pt, y),
    }


def zero_book(d, y, ps, pt):
    z = ((d["book_move"] == 0) & (d["x"].abs() > 0)).to_numpy()
    if not z.any():
        return {"games": 0}
    return {
        "games": int(z.sum()),
        "exchange_direction_record": PMG.side_record(
            np.where(d["x"].to_numpy()[z] > 0, 1.0, 0.0), y[z]
        ),
        "served_record": PMG.side_record(ps[z], y[z]),
        "beside_record": PMG.side_record(pt[z], y[z]),
        "beside_changed_pick_record": PMG.record(pt[z], ps[z], y[z]),
    }


def cmd_coverage(a):
    h, _ = history()
    rep = summary(h, "fit seasons 2024-2025 (no outcomes)")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "coverage_fit.json").write_text(json.dumps(rep, indent=1))


def cmd_fit(a):
    if FROZEN.exists():
        raise SystemExit(f"already frozen: {FROZEN}")
    h, act = history()
    h = h[h["result"] != h["tue_open_home_spread"]].dropna(
        subset=["x", "out_of_season_home_probability"]
    )
    y = (h["result"] > h["tue_open_home_spread"]).astype(float).to_numpy()
    x = h["x"].to_numpy(float)
    s = h["season"].to_numpy()
    blk = s * 100 + h["week"].to_numpy()
    ps = sig(h["off_beside"].to_numpy(float))
    rep = {"active_artifact": act["artifact"], "games": len(y), "arms": {}}
    frozen = {}
    held = {}
    for arm in ARMS:
        off = h[f"off_{arm}"].to_numpy(float)
        b, se = PMG.fit(x, y, off)
        ins = grade(ps, sig(off + b * x), y, blk)
        p_oos = ps.copy()
        folds = {}
        for k in PMG.FIT_SEASONS:
            tr = s != k
            folds[int(k)] = PMG.fit(x[tr], y[tr], off[tr])[0]
            p_oos[~tr] = sig(off[~tr] + folds[int(k)] * x[~tr])
        held[arm] = p_oos
        out = grade(ps, p_oos, y, blk)
        rep["arms"][arm] = {
            "coef_logit_per_point": b,
            "se": se,
            "coef_per_held_out_season": folds,
            "in_sample_log_loss_gain": ins["log_loss_gain"],
            "in_sample_changed_pick_record": ins["changed_pick_record"],
            "loso": out,
            "in_minus_held_out_gain": ins["log_loss_gain"] - out["log_loss_gain"],
        }
        frozen[arm] = b
        print(arm, json.dumps({k: v for k, v in rep["arms"][arm].items() if k != "loso"}))
        print(" loso", json.dumps({k: v for k, v in out.items() if k != "reliability_term"}))
    rep["zero_book_move_held_out"] = zero_book(h, y, ps, held["beside"])
    rep["distribution"] = summary(h, "fit games")["distribution"]
    print("zero_book", json.dumps(rep["zero_book_move_held_out"]))
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "fit_2024_2025.json").write_text(json.dumps(rep, indent=1, default=float))
    _, _, _, c26 = active()
    body = {
        "family": "exchange_move",
        "coefficient_logit_per_point": frozen,
        "served_move_coefficient_2026": c26,
        "rule": "offset-logistic slope on the exchange move (venue mean of L*(deadline price) "
        "- L*(Tuesday-noon price)); beside offset logit(served), replace offset subtracts the "
        "served move coefficient times the book move; label home cover vs Tuesday opener; "
        "fitted on 2024-2025",
        "market_move_feature_version": act["market_move_feature_version"],
        "map_residual_seasons_before": PMG.MAP_BEFORE,
        "window_hours": PMG.WINDOW / pd.Timedelta(hours=1),
        "lane": LANE,
        "created_at_utc": datetime.now(UTC).strftime(STAMP),
    }
    FROZEN.write_text(json.dumps(body, indent=1))
    print(json.dumps(body, indent=1))


def book_moves(g, version):
    quotes = load_decision_quotes(DATA / "market" / "raw", capture_kind=LIVE_CAPTURE_KIND)
    rows = []
    for _, grp in g.groupby("week"):
        games = pd.DataFrame(
            {
                "game_id": grp["game_id"].astype(str),
                "commence_time_utc": grp["kickoff_utc"],
                "week_first_commence_utc": grp["kickoff_utc"].min(),
                "cutoff_utc": grp["T_deadline"],
            }
        ).drop_duplicates("game_id")
        e = sharp_book_movement_features(
            quotes, games, include_sunday=version == MARKET_MOVE_FEATURE_SUNDAY
        )
        mv = pd.to_numeric(e["leader_median_net_move"], errors="coerce").fillna(0.0)
        av = pd.to_numeric(e["leader_books"], errors="coerce").fillna(0.0).gt(0)
        rows.append(
            pd.DataFrame(
                {
                    "game_id": e["game_id"].astype(str),
                    "book_move": mv.where(av, 0.0),
                    "book_available": av.astype(float),
                }
            )
        )
    return pd.concat(rows).set_index("game_id")


def cmd_score2026(a):
    path = OUT / "score_2026_w1_10.json"
    if path.exists():
        raise SystemExit(f"already scored once: {path}")
    fz = json.loads(FROZEN.read_text())
    g = PMG.season_games()
    g = g[g["week"].between(*PMG.SCORED_WEEKS) & g["result"].notna()].copy()
    op = mwf.tuesday_open(g)
    g["open_line"] = g["game_id"].map(lambda i: op.get(i, (np.nan,))[0])
    g = PMG.stamps(g.merge(FL._played_card(), on="game_id", how="inner")).reset_index(drop=True)
    h = g.set_index("game_id").join(exchange_moves(g))
    h = h.join(book_moves(g, fz["market_move_feature_version"]))
    h[["book_move", "book_available"]] = h[["book_move", "book_available"]].fillna(0.0)
    rep = {
        "frozen_sha256": hashlib.sha256(FROZEN.read_bytes()).hexdigest(),
        "summary": summary(h, "2026 W1-10 (no outcomes)"),
    }
    h = h.dropna(subset=["open_line", "served_p", "x"])
    h = h[h["result"] != h["open_line"]]
    y = (h["result"] > h["open_line"]).astype(float).to_numpy()
    x = h["x"].to_numpy(float)
    off_b = logit(h["served_p"].to_numpy(float))
    offs = {
        "beside": off_b,
        "replace": off_b - fz["served_move_coefficient_2026"] * h["book_move"].to_numpy(float),
    }
    ps = sig(off_b)
    pts = {}
    for arm in ARMS:
        pts[arm] = sig(offs[arm] + fz["coefficient_logit_per_point"][arm] * x)
        rep[arm] = grade(ps, pts[arm], y, h["week"].to_numpy())
        print(arm, json.dumps({k: v for k, v in rep[arm].items() if k != "reliability_term"}))
    rep["zero_book_move"] = zero_book(h, y, ps, pts["beside"])
    print("zero_book", json.dumps(rep["zero_book_move"]))
    path.write_text(json.dumps(rep, indent=1))


def cmd_pregame(a):
    now = pd.Timestamp(datetime.now(UTC))
    fz = json.loads(FROZEN.read_text())
    week = PMG.resolve(a.week, now)
    if not a.dry and not FIRST_WEEK <= week <= LAST_WEEK:
        raise SystemExit(f"week {week} outside the declared forward window")
    sched = PMG.schedule(week)
    sched = sched if a.dry else sched[sched["cutoff_utc"] > now]
    t_tue = sched["T_tue"].min()
    if sched.empty or t_tue >= now:
        print("no game after its Tuesday-noon stamp and before its cutoff")
        return
    px_t = PMG.live_prices(sched, t_tue)
    px_n = PMG.live_prices(sched, now)
    bm = market_move_toward_home(
        sched, DATA, now=now.to_pydatetime(), feature_version=fz["market_move_feature_version"]
    ).set_index("game_id")
    pub = pd.read_parquet(ROOT / "artifacts" / "clv_ledger" / "published_picks.parquet")
    pub = pub[(pub["season"] == SEASON) & (pub["week"] == week)]
    pub = pub[pd.to_datetime(pub["published_at_utc"], utc=True) <= now]
    base = {
        r.game_id: r.displayed_score if r.pick_team == r.home_team else 1 - r.displayed_score
        for r in pub.sort_values("published_at_utc").groupby("game_id").tail(1).itertuples()
    }
    res = PMG.residuals()
    b = fz["coefficient_logit_per_point"]
    games = []
    for r in sched.itertuples():
        mv = {}
        for v in PMG.VENUES:
            pt = px_t[(px_t["game_id"] == r.game_id) & (px_t["venue"] == v)]["p"]
            pn = px_n[(px_n["game_id"] == r.game_id) & (px_n["venue"] == v)]["p"]
            if len(pt) and len(pn):
                mv[v] = float(PMG.spread_equiv(pn.iloc[0], res) - PMG.spread_equiv(pt.iloc[0], res))
        if not mv or r.game_id not in base:
            continue
        x = float(np.mean(list(mv.values())))
        book = float(bm[MOVE].get(r.game_id, 0.0))
        z0 = float(logit(float(base[r.game_id])))
        pb = float(sig(z0 + b["beside"] * x))
        pr = float(sig(z0 - fz["served_move_coefficient_2026"] * book + b["replace"] * x))
        games.append(
            {
                "game_id": r.game_id,
                "kickoff_utc": r.kickoff_utc.isoformat(),
                "pick_deadline_utc": r.cutoff_utc.isoformat(),
                "venue_move": mv,
                "term_value": x,
                "book_move": book,
                "base_home_probability": float(base[r.game_id]),
                "beside_home_probability": pb,
                "replace_home_probability": pr,
                "beside_side_changed": (base[r.game_id] >= 0.5) != (pb >= 0.5),
                "replace_side_changed": (base[r.game_id] >= 0.5) != (pr >= 0.5),
            }
        )
    doc = {
        "season": SEASON,
        "week": week,
        "tuesday_noon_utc": t_tue.isoformat(),
        "created_at_utc": now.strftime(STAMP),
        "frozen_sha256": hashlib.sha256(FROZEN.read_bytes()).hexdigest(),
        "prices": {"tuesday": len(px_t), "now": len(px_n)},
        "games": games,
    }
    if a.dry:
        print(json.dumps({k: v for k, v in doc.items() if k != "games"}, default=str))
        for gm in games:
            print(
                gm["game_id"],
                gm["venue_move"],
                gm["book_move"],
                round(gm["base_home_probability"], 4),
                round(gm["beside_home_probability"], 4),
                round(gm["replace_home_probability"], 4),
            )
        print("DRY: nothing written")
        return
    if not games:
        print("no game with both exchange stamps and a published card before its cutoff")
        return
    out = OUT / f"{SEASON}_week{week:02d}_deadline_{now.strftime(STAMP)}.json"
    out.write_text(json.dumps(doc, indent=1, default=str))
    print(out, len(games), "games")


def cmd_score(a):
    fz = json.loads(FROZEN.read_text())
    sha = hashlib.sha256(FROZEN.read_bytes()).hexdigest()
    rows = []
    for f in sorted(OUT.glob(f"{SEASON}_week*_deadline_*.json")):
        d = json.loads(f.read_text())
        if d["frozen_sha256"] != sha or not FIRST_WEEK <= d["week"] <= LAST_WEEK:
            continue
        mtime = pd.Timestamp(f.stat().st_mtime, unit="s", tz="UTC")
        made = pd.Timestamp(datetime.strptime(d["created_at_utc"], STAMP).replace(tzinfo=UTC))
        for gm in d["games"]:
            cut = pd.Timestamp(gm["pick_deadline_utc"])
            if made < cut and mtime < cut:
                rows.append(
                    {
                        "game_id": gm["game_id"],
                        "x": gm["term_value"],
                        "book_move": gm["book_move"],
                        "created": d["created_at_utc"],
                    }
                )
    if not rows:
        print("no admissible forward rows yet; nothing graded")
        return
    t = pd.DataFrame(rows).sort_values("created").drop_duplicates("game_id", keep="last")
    g = PMG.season_games()
    g = g[g["game_id"].isin(t["game_id"]) & g["result"].notna()].copy()
    op = mwf.tuesday_open(g)
    g["open_line"] = g["game_id"].map(lambda i: op.get(i, (np.nan,))[0])
    g = g.merge(FL._played_card(), on="game_id").dropna(subset=["open_line", "served_p"])
    d = g[g["result"] != g["open_line"]].merge(t, on="game_id")
    if d.empty:
        print("no graded forward games yet")
        return
    y = (d["result"] > d["open_line"]).astype(float).to_numpy()
    x = d["x"].to_numpy(float)
    off_b = logit(d["served_p"].to_numpy(float))
    offs = {
        "beside": off_b,
        "replace": off_b - fz["served_move_coefficient_2026"] * d["book_move"].to_numpy(float),
    }
    rep = {}
    pts = {}
    for arm in ARMS:
        pts[arm] = sig(offs[arm] + fz["coefficient_logit_per_point"][arm] * x)
        rep[arm] = grade(sig(off_b), pts[arm], y, d["week"].to_numpy())
    rep["zero_book_move"] = zero_book(d, y, sig(off_b), pts["beside"])
    (OUT / "score_forward.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep, indent=1))


def cmd_pergame(a):
    fz = json.loads(FROZEN.read_text())
    cols = ["season", "week", "x", "x_kalshi", "x_polymarket", "book_move", "y", "served"]
    h, _ = history()
    h = h[h["result"] != h["tue_open_home_spread"]].dropna(
        subset=["x", "out_of_season_home_probability"]
    )
    h["y"] = (h["result"] > h["tue_open_home_spread"]).astype(float)
    h["served"] = sig(h["off_beside"].to_numpy(float))
    s, x, y = h["season"].to_numpy(), h["x"].to_numpy(float), h["y"].to_numpy()
    for arm in ARMS:
        off = h[f"off_{arm}"].to_numpy(float)
        p = np.empty(len(h))
        for k in PMG.FIT_SEASONS:
            tr = s != k
            p[~tr] = sig(off[~tr] + PMG.fit(x[tr], y[tr], off[tr])[0] * x[~tr])
        h[arm] = p
    h["grade"] = "loso_2024_2025"
    g = PMG.season_games()
    g = g[g["week"].between(*PMG.SCORED_WEEKS) & g["result"].notna()].copy()
    op = mwf.tuesday_open(g)
    g["open_line"] = g["game_id"].map(lambda i: op.get(i, (np.nan,))[0])
    g = PMG.stamps(g.merge(FL._played_card(), on="game_id", how="inner")).reset_index(drop=True)
    n = g.set_index("game_id").join(exchange_moves(g))
    n = n.join(book_moves(g, fz["market_move_feature_version"]))
    n["book_move"] = n["book_move"].fillna(0.0)
    n = n.dropna(subset=["open_line", "served_p", "x"])
    n = n[n["result"] != n["open_line"]].copy()
    n["y"] = (n["result"] > n["open_line"]).astype(float)
    n["served"] = n["served_p"].astype(float)
    off_b = logit(n["served"].to_numpy(float))
    b = fz["coefficient_logit_per_point"]
    n["beside"] = sig(off_b + b["beside"] * n["x"].to_numpy(float))
    n["replace"] = sig(
        off_b
        - fz["served_move_coefficient_2026"] * n["book_move"].to_numpy(float)
        + b["replace"] * n["x"].to_numpy(float)
    )
    n["grade"] = "frozen_2026_w1_10"
    pg = pd.concat([h[[*cols, *ARMS, "grade"]], n[[*cols, *ARMS, "grade"]]]).reset_index()
    pg.to_parquet(OUT / "per_game.parquet")
    rep = {}
    for label in ("loso_2024_2025", "frozen_2026_w1_10", "pooled"):
        d = pg if label == "pooled" else pg[pg["grade"] == label]
        z = d[(d["book_move"] == 0) & (d["x"].abs() > 0)]
        hit = ((z["x"] > 0) == (z["y"] == 1)).to_numpy()
        bs = PMG.boot(
            hit * 100.0 - 50.0,
            (z["season"] * 100 + z["week"]).to_numpy(),
            np.random.default_rng(PMG.SEED),
        )
        rep[label] = {
            "games": len(z),
            "exchange_direction_record": f"{int(hit.sum())}-{int((~hit).sum())}",
            "rate_pp_over_50": float(hit.mean() * 100 - 50),
            "ci": [float(np.quantile(bs, 0.025)), float(np.quantile(bs, 0.975))],
            "probability_positive": float((bs > 0).mean()),
        }
    (OUT / "zero_book_direction.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep))


def cmd_argv(a):
    base = [".tools/uv.exe", "run", "--no-sync", "python", "scripts/exchange_move.py"]
    print(
        json.dumps(
            {
                "pregame": {
                    "argv": [*base, "pregame"],
                    "slots": [
                        "Thursday 19:45 ET",
                        "Saturday 15:45 ET (when games)",
                        "Sunday 12:30 ET",
                        "Sunday 15:45 ET",
                    ],
                },
                "score": {"argv": [*base, "score"], "slot": "Tuesday 09:00 ET"},
            },
            indent=1,
        )
    )


def main():
    p = argparse.ArgumentParser(
        description=(
            "Exchange-confirmed line move: Kalshi and Polymarket win-price change from Tuesday "
            "noon ET to each game's pick deadline, mapped to points with the market residual "
            "distribution, as one fitted term beside (or replacing) the served book move. "
            "Research only; serves nothing."
        )
    )
    sp = p.add_subparsers(dest="cmd", required=True)
    sp.add_parser("coverage").set_defaults(fn=cmd_coverage)
    sp.add_parser("fit").set_defaults(fn=cmd_fit)
    sp.add_parser("score2026").set_defaults(fn=cmd_score2026)
    pg = sp.add_parser("pregame")
    pg.add_argument("--week", default="auto")
    pg.add_argument("--dry", action="store_true")
    pg.set_defaults(fn=cmd_pregame)
    sp.add_parser("score").set_defaults(fn=cmd_score)
    sp.add_parser("pergame").set_defaults(fn=cmd_pergame)
    sp.add_parser("argv").set_defaults(fn=cmd_argv)
    a = p.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
