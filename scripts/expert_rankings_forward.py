from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit

sys.path.insert(0, str(Path(__file__).resolve().parent))

import expert_rankings as E
import pooled_stack as PS
from mod27_fenced_repairs_replay import FOUR, Fit

ROOT = E.ROOT
OUT = ROOT / "artifacts/expert_rankings_forward"
FROZEN = OUT / "frozen.json"
INPUTS = OUT / "inputs"
SCREEN = E.OUT / "screen_games.parquet"
PUBLISHED = ROOT / "artifacts/clv_ledger/published_picks.parquet"
TERM = "gap"
HISTORY = (2020, 2025)
SEASON = 2026
LIVE_WEEKS = (2, 10)
FIRST_WEEK = 11
LAST_WEEK = 18
HALF_SPLIT = 2023
PUSH = 0.5
BOOT = 2000
SEED = 20261116
NEWTON_ITER = 100
TOL = 1e-12
STAMP = "%Y%m%dT%H%M%SZ"
FAMILY = "expert_rankings_served_2026_11_16"
LANE = "docs/lanes/expert-rankings-2026-11-16.md"
SLOTS = {"pregame": ("Fri 20:45", "Sat 12:05", "Sun 09:20"), "score": ("Tue 09:20",)}


def sha_file(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def offset_fit(x, y, off):
    b = 0.0
    for _ in range(NEWTON_ITER):
        p = expit(off + b * x)
        h = float(np.sum(x * x * p * (1 - p)))
        if h <= 0:
            break
        step = float(np.sum(x * (y - p))) / h
        b += step
        if abs(step) < TOL:
            break
    return b


def standardise(raw, rows):
    v = raw[rows & ~np.isnan(raw)]
    mu, sd = float(v.mean()), float(v.std(ddof=0))
    return np.nan_to_num((raw - mu) / sd, nan=0.0), mu, sd


def rankings(ecr):
    d = ecr.loc[ecr["page_type"].isin(E.PAGES)].copy()
    d["pos_page"] = d["page_type"].map(E.PAGES)
    d["scrape"] = pd.to_datetime(d["scrape_date"]).dt.normalize()
    d["pos_rank"] = d.groupby(["scrape", "pos_page"])["ecr"].rank(method="first")
    d["team"] = d["tm"].replace(E.TEAM_CODES)
    d["next_sunday"] = d["scrape"] + pd.to_timedelta(
        (E.SUNDAY - d["scrape"].dt.dayofweek) % 7, unit="D"
    )
    ids = pd.read_csv(E.RAW / "db_playerids.csv", usecols=["fantasypros_id", "gsis_id"], dtype=str)
    ids = ids.dropna().drop_duplicates("fantasypros_id", keep=False)
    return d.merge(ids, left_on="id", right_on="fantasypros_id", how="left")


def points(stats):
    s = stats.loc[stats["season_type"].eq("REG")]
    return s.groupby(["player_id", "season", "week"])["fantasy_points_ppr"].sum()


def history_terms():
    from hindsight_audit_team import tuesday_quotes

    table, snap, raw = E.schedules()
    d = E.rankings()
    pts = E.actual_points()
    chosen = E.assign_scrapes(table, pd.Series(sorted(d["scrape"].unique())))
    _, spread = tuesday_quotes(raw)
    spread.index = spread.index.astype(str)
    have = table["game_id"].astype(str).isin(spread.dropna().index)
    corr = float(
        np.corrcoef(
            spread.reindex(table.loc[have, "game_id"]).to_numpy(float),
            table.loc[have, "spread_line"],
        )[0, 1]
    )
    terms, _ = E.build_terms(table, d, pts, chosen, spread * np.sign(corr))
    terms["game_id"] = terms["game_id"].astype(str)
    return terms.drop_duplicates("game_id").set_index("game_id"), snap


def live_terms(ecr, stats, line):
    saved = E.SEASONS
    E.SEASONS = (SEASON, SEASON)
    try:
        table, snap, _ = E.schedules()
    finally:
        E.SEASONS = saved
    import v4_candidate as v4

    kick, cut = v4.kickoff_and_cutoff(table)
    table = table.assign(kickoff_utc=kick, cutoff_utc=cut)
    d = rankings(ecr)
    chosen = E.assign_scrapes(table, pd.Series(sorted(d["scrape"].unique())))
    terms, meta = E.build_terms(table, d, points(stats), chosen, line)
    terms["game_id"] = terms["game_id"].astype(str)
    terms = terms.drop_duplicates("game_id").set_index("game_id")
    terms["scrape"] = pd.Series(chosen).reindex(terms.index).astype(str)
    table["game_id"] = table["game_id"].astype(str)
    return terms, table.set_index("game_id"), snap, meta


def card(now=None):
    pub = pd.read_parquet(PUBLISHED)
    pub = pub.loc[pub["season"].eq(SEASON)]
    pub = pub.loc[pub["published_at_utc"].le(pub["pick_deadline_utc"])]
    if now is not None:
        pub = pub.loc[pd.to_datetime(pub["published_at_utc"], utc=True).le(pd.Timestamp(now))]
    pub = pub.sort_values("published_at_utc").drop_duplicates("game_id", keep="last").copy()
    home = pub["pick_team"].eq(pub["home_team"])
    pub["served_p"] = np.where(home, pub["displayed_score"], 1 - pub["displayed_score"])
    pub["card_line"] = pub["market_spread"].astype(float)
    pub["game_id"] = pub["game_id"].astype(str)
    return pub.set_index("game_id")[["week", "served_p", "card_line", "published_at_utc"]]


def evaluate(p, q, ylab, blocks, rng):
    decided = ylab != PUSH
    cp = ((p >= 0.5) == (ylab == 1)) & decided
    cq = ((q >= 0.5) == (ylab == 1)) & decided
    changed = (p >= 0.5) != (q >= 0.5)
    ones = np.ones(len(p))
    acc_game = 100 * (cp.astype(float) - cq.astype(float))
    ll_game = PS.ll(q, ylab) - PS.ll(p, ylab)
    br_game = (q - ylab) ** 2 - (p - ylab) ** 2
    ct, cb = changed & decided & ~cp, changed & decided & ~cq
    return {
        "games": len(p),
        "decided": int(decided.sum()),
        "pushes": int((~decided).sum()),
        "record_term": f"{int(cp.sum())}-{int((decided & ~cp).sum())}",
        "record_base": f"{int(cq.sum())}-{int((decided & ~cq).sum())}",
        "changed": int(changed.sum()),
        "changed_pushes": int((changed & ~decided).sum()),
        "changed_record_term": f"{int((changed & cp).sum())}-{int(ct.sum())}",
        "changed_record_base": f"{int((changed & cq).sum())}-{int(cb.sum())}",
        "accuracy_points": PS.block_boot(blocks, acc_game, decided.astype(float), rng),
        "log_loss_gain": PS.block_boot(blocks, ll_game, ones, rng),
        "brier_gain": PS.block_boot(blocks, br_game, ones, rng),
        "mean_log_loss": {
            "base": float(PS.ll(q, ylab).mean()),
            "term": float(PS.ll(p, ylab).mean()),
        },
        "reliability_term": PS.reliability(p[decided], ylab[decided]),
        "reliability_base": PS.reliability(q[decided], ylab[decided]),
    }


def cmd_fit(a):
    if FROZEN.exists():
        raise SystemExit(f"already frozen: {FROZEN}")
    OUT.mkdir(parents=True, exist_ok=True)
    terms, snap = history_terms()
    scr = pd.read_parquet(SCREEN)
    scr["game_id"] = scr["game_id"].astype(str)
    both = scr.set_index("game_id")[TERM].dropna()
    identity = float((terms[TERM].reindex(both.index) - both).abs().max())
    pop = PS.population()
    pop["game_id"] = pop["game_id"].astype(str)
    season = pop["season"].to_numpy(int)
    decided = ~pop["push"].to_numpy(bool)
    ylab = pop["y"].to_numpy(float)
    seasons = sorted(set(season.tolist()))
    raw = terms[TERM].reindex(pop["game_id"]).to_numpy(float)
    cache = {}

    def off_for(excl, rows):
        key = frozenset(excl)
        if key not in cache:
            cache[key] = Fit(FOUR, pop[decided & ~np.isin(season, list(key))])
        return PS.logit(cache[key].predict(pop[rows]))

    off1 = np.full(len(pop), np.nan)
    for s in seasons:
        off1[season == s] = off_for({s}, season == s)
    qw = expit(off1)
    p = np.full(len(pop), np.nan)
    p_int = np.full(len(pop), np.nan)
    folds = {}
    for s in seasons:
        tr = season != s
        off_tr = np.full(len(pop), np.nan)
        for t in seasons:
            if t != s:
                off_tr[season == t] = off_for({s, t}, season == t)
        xs, mu, sd = standardise(raw, tr)
        b = offset_fit(xs[tr], ylab[tr], off_tr[tr])
        bi = offset_fit(np.ones(int(tr.sum())), ylab[tr], off_tr[tr])
        held = season == s
        p[held] = expit(off1[held] + b * xs[held])
        p_int[held] = expit(off1[held] + bi)
        folds[str(s)] = {"coefficient_per_sd": b, "mean": mu, "sd": sd, "intercept_control": bi}
    xs, mu, sd = standardise(raw, np.ones(len(pop), bool))
    b_full = offset_fit(xs, ylab, off1)
    p_in = expit(off1 + b_full * xs)
    halves = {}
    for name, m in (("2020-22", season < HALF_SPLIT), ("2023-25", season >= HALF_SPLIT)):
        halves[name] = offset_fit(xs[m], ylab[m], off1[m])
    rng = np.random.default_rng(SEED)
    wk = season * 100 + pop["week"].to_numpy(int)
    res = {
        "schedule_snapshot": snap,
        "history_term_identity_vs_screen_max_abs": identity,
        "history_terms": int(terms[TERM].notna().sum()),
        "games": len(pop),
        "pushes": int((~decided).sum()),
        "nonzero_term_games": int((xs != 0).sum()),
        "qw_record": PS.record(qw[decided], ylab[decided]),
        "coefficient_per_sd_full": b_full,
        "term_mean": mu,
        "term_sd": sd,
        "fold": folds,
        "folds_positive": int(sum(f["coefficient_per_sd"] > 0 for f in folds.values())),
        "halves_coefficient_per_sd": halves,
        "week_block": evaluate(p, qw, ylab, wk, rng),
        "season_block": evaluate(p, qw, ylab, season, rng),
        "in_sample_log_loss_gain": float((PS.ll(qw, ylab) - PS.ll(p_in, ylab)).mean()),
        "intercept_only_log_loss_gain": float((PS.ll(qw, ylab) - PS.ll(p_int, ylab)).mean()),
        "corr_term_qw_logit": float(np.corrcoef(xs[xs != 0], off1[xs != 0])[0, 1]),
    }
    res["in_sample_minus_held_out_log_loss_gain"] = (
        res["in_sample_log_loss_gain"] - res["week_block"]["log_loss_gain"]["effect"]
    )
    pd.DataFrame(
        {"game_id": pop["game_id"], "season": season, "y": ylab, "qw": qw, "term": p, "x": xs}
    ).to_parquet(OUT / "history_games.parquet", index=False)
    (OUT / "fit.json").write_text(json.dumps(res, indent=1))
    body = {
        "family": FAMILY,
        "term": TERM,
        "sign": "positive on home cover",
        "coefficient_logit_per_sd": b_full,
        "term_mean_points": mu,
        "term_sd_points": sd,
        "fit": "offset logistic, no intercept, QW served stage offset (LOSO), 2020-25, pushes 0.5",
        "fit_games": len(pop),
        "fit_sha256": sha_file(OUT / "fit.json"),
        "first_week": FIRST_WEEK,
        "last_week": LAST_WEEK,
        "lane": LANE,
        "no_2026_outcomes": True,
        "created_at_utc": datetime.now(UTC).strftime(STAMP),
    }
    FROZEN.write_text(json.dumps(body, indent=1))
    print(json.dumps(res, indent=1))
    print(json.dumps(body, indent=1))


def apply_frozen(fz, gap, base):
    x = 0.0 if not np.isfinite(gap) else (gap - fz["term_mean_points"]) / fz["term_sd_points"]
    dl = fz["coefficient_logit_per_sd"] * x
    return x, dl, float(expit(PS.logit(base) + dl))


def cmd_live(a):
    fz = json.loads(FROZEN.read_text())
    ecr = pd.read_parquet(E.RAW / "db_fpecr.parquet")
    stats = load_stats()
    cd = card()
    terms, table, snap, meta = live_terms(ecr, stats, cd["card_line"])
    ids = cd.index[cd["week"].between(*LIVE_WEEKS)]
    rows = []
    for gid in ids:
        if gid not in table.index or pd.isna(table.loc[gid, "result"]):
            continue
        gap = float(terms[TERM].get(gid, np.nan))
        base = float(cd.loc[gid, "served_p"])
        x, _, p = apply_frozen(fz, gap, base)
        m = float(table.loc[gid, "result"]) - float(cd.loc[gid, "card_line"])
        rows.append(
            {
                "game_id": gid,
                "week": int(cd.loc[gid, "week"]),
                "scrape": terms["scrape"].get(gid),
                "gap": gap,
                "x": x,
                "base": base,
                "term": p,
                "y": PUSH if m == 0 else float(m > 0),
            }
        )
    g = pd.DataFrame(rows)
    g.to_parquet(OUT / "live_games.parquet", index=False)
    rng = np.random.default_rng(SEED)
    res = {
        "schedule_snapshot": snap,
        "frozen_sha256": sha_file(FROZEN),
        "games_with_term": int(g["gap"].notna().sum()),
        "weeks": sorted(int(w) for w in g["week"].unique()),
        "week_meta": meta,
        "per_week_changed": g.assign(c=(g["term"] >= 0.5) != (g["base"] >= 0.5))
        .groupby("week")["c"]
        .sum()
        .astype(int)
        .to_dict(),
        **evaluate(
            g["term"].to_numpy(float),
            g["base"].to_numpy(float),
            g["y"].to_numpy(float),
            g["week"].to_numpy(int),
            rng,
        ),
    }
    res["per_week_changed"] = {str(k): int(v) for k, v in res["per_week_changed"].items()}
    (OUT / "live.json").write_text(json.dumps(res, indent=1, default=str))
    print(json.dumps({k: v for k, v in res.items() if k != "week_meta"}, indent=1, default=str))


def load_stats(fetch=False):
    path = INPUTS / f"stats_player_week_{SEASON}.parquet"
    if fetch or not path.exists():
        import nflreadpy

        stats = nflreadpy.load_player_stats([SEASON], summary_level="week").to_pandas()
        stats = stats[["player_id", "season", "week", "season_type", "fantasy_points_ppr"]]
        INPUTS.mkdir(parents=True, exist_ok=True)
        stats.to_parquet(path, index=False)
    return pd.read_parquet(path)


def resolve(arg, table, now):
    if arg != "auto":
        return int(arg)
    ahead = table.loc[table["kickoff_utc"] > pd.Timestamp(now), "week"]
    if ahead.empty:
        raise SystemExit("no unplayed regular-season week")
    return int(ahead.min())


def cmd_pregame(a):
    import nflreadpy

    now = datetime.now(UTC)
    fz = json.loads(FROZEN.read_text())
    ecr = nflreadpy.load_ff_rankings("all").to_pandas()
    stats = load_stats(fetch=True)
    cd = card(now)
    terms, table, snap, _ = live_terms(ecr, stats, cd["card_line"])
    week = resolve(a.week, table, now)
    if not a.dry and not FIRST_WEEK <= week <= LAST_WEEK:
        raise SystemExit(
            f"week {week} outside the declared forward window {FIRST_WEEK}-{LAST_WEEK}"
        )
    newest = str(pd.to_datetime(ecr["scrape_date"]).max().date())
    games, skipped = [], {}
    for gid, r in table.loc[table["week"].eq(week)].iterrows():
        if not a.dry and r["cutoff_utc"] <= pd.Timestamp(now):
            skipped[gid] = "past_cutoff"
            continue
        if gid not in cd.index:
            skipped[gid] = "no_published_card"
            continue
        gap = float(terms[TERM].get(gid, np.nan))
        if not np.isfinite(gap):
            skipped[gid] = "no_pregame_scrape"
            continue
        base = float(cd.loc[gid, "served_p"])
        x, dl, p = apply_frozen(fz, gap, base)
        games.append(
            {
                "game_id": gid,
                "away_team": r["away_team"],
                "home_team": r["home_team"],
                "kickoff_utc": r["kickoff_utc"].isoformat(),
                "cutoff_utc": r["cutoff_utc"].isoformat(),
                "scrape": terms["scrape"].get(gid),
                "e_diff": float(terms["e_diff"].get(gid)),
                "card_line": float(cd.loc[gid, "card_line"]),
                "term_value": gap,
                "x": x,
                "base_home_probability": base,
                "base_published_at_utc": str(cd.loc[gid, "published_at_utc"]),
                "delta_logit": dl,
                "combined_home_probability": p,
                "side_changed": bool((base >= 0.5) != (p >= 0.5)),
            }
        )
    doc = {
        "season": SEASON,
        "week": week,
        "created_at_utc": now.strftime(STAMP),
        "schedule_snapshot": snap,
        "newest_ecr_scrape": newest,
        "frozen_sha256": sha_file(FROZEN),
        "coefficient_logit_per_sd": fz["coefficient_logit_per_sd"],
        "skipped": skipped,
        "games": games,
    }
    if a.dry:
        live = OUT / "live_games.parquet"
        if live.exists() and games:
            lg = pd.read_parquet(live).set_index("game_id")["gap"]
            mine = pd.Series({gm["game_id"]: gm["term_value"] for gm in games})
            both = pd.concat([mine, lg], axis=1, join="inner").dropna()
            doc["live_check"] = {
                "games": len(both),
                "max_abs_diff": float((both.iloc[:, 0] - both.iloc[:, 1]).abs().max())
                if len(both)
                else None,
            }
        print(json.dumps({k: v for k, v in doc.items() if k != "games"}, indent=1, default=str))
        for gm in games:
            print(
                f"{gm['game_id']:<18} gap {gm['term_value']:+.2f} base "
                f"{gm['base_home_probability']:.4f} term {gm['combined_home_probability']:.4f} "
                f"changed {gm['side_changed']}"
            )
        print("DRY: nothing written")
        return
    if not games:
        print(json.dumps({k: v for k, v in doc.items() if k != "games"}, indent=1, default=str))
        print("no game with a pregame scrape and a card before its cutoff; nothing written")
        return
    out = OUT / f"{SEASON}_week{week:02d}_{now.strftime(STAMP)}.json"
    out.write_text(json.dumps(doc, indent=1, default=str))
    print(out, len(games), "games")


def cmd_score(a):
    fz = json.loads(FROZEN.read_text())
    sha = sha_file(FROZEN)
    rows = []
    for f in sorted(OUT.glob(f"{SEASON}_week*_*.json")):
        d = json.loads(f.read_text())
        if d["frozen_sha256"] != sha or not FIRST_WEEK <= d["week"] <= LAST_WEEK:
            continue
        mtime = pd.Timestamp(f.stat().st_mtime, unit="s", tz="UTC")
        made = pd.Timestamp(datetime.strptime(d["created_at_utc"], STAMP).replace(tzinfo=UTC))
        for gm in d["games"]:
            cut = pd.Timestamp(gm["cutoff_utc"])
            if made < cut and mtime < cut:
                rows.append({**gm, "week": d["week"], "created": d["created_at_utc"]})
    if not rows:
        print("no admissible forward rows yet; nothing graded")
        return
    t = pd.DataFrame(rows).sort_values("created").drop_duplicates("game_id", keep="last")
    saved = E.SEASONS
    E.SEASONS = (SEASON, SEASON)
    try:
        table, _, _ = E.schedules()
    finally:
        E.SEASONS = saved
    result = table.assign(game_id=table["game_id"].astype(str)).set_index("game_id")["result"]
    cd = card()
    t = t.loc[t["game_id"].isin(cd.index)].copy()
    t["result"] = t["game_id"].map(result)
    t = t.dropna(subset=["result"])
    if t.empty:
        print("no settled forward game yet; nothing graded")
        return
    t["base"] = t["game_id"].map(cd["served_p"]).astype(float)
    m = t["result"].astype(float) - t["game_id"].map(cd["card_line"]).astype(float)
    t["y"] = np.where(m == 0, PUSH, (m > 0).astype(float))
    t["term"] = [apply_frozen(fz, g, b)[2] for g, b in zip(t["term_value"], t["base"], strict=True)]
    rep = {"weeks": sorted(int(w) for w in t["week"].unique())}
    rep.update(
        evaluate(
            t["term"].to_numpy(float),
            t["base"].to_numpy(float),
            t["y"].to_numpy(float),
            t["week"].to_numpy(int),
            np.random.default_rng(SEED),
        )
    )
    (OUT / "score.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep, indent=1))


def cmd_argv(a):
    base = [".tools/uv.exe", "run", "--no-sync", "python", "scripts/expert_rankings_forward.py"]
    for mode, slots in SLOTS.items():
        print(json.dumps({"argv": [*base, mode], "slots_et": list(slots)}))


def cell(name, look, metric, units, start, end, text, summary):
    b = look[metric]
    return {
        "name": name,
        "description": text,
        "source": LANE,
        "effect": round(b["effect"], 6),
        "effect_units": units,
        "classification": "unresolved_below_power",
        "league": "nfl",
        "season_start": start,
        "season_end": end,
        "family": FAMILY,
        "category": "market",
        "classification_evidence": (
            "Declared sign; week-block interval spans both signs at this sample; no refuted "
            "mechanism and no positive control bounds an effect this size on the served stage."
        ),
        "notes": (
            f"Family {FAMILY}: one served look at the opener 2020-25 (2 metrics) plus one frozen "
            "2026 Weeks 2-10 score (2 metrics), declared in the lane before either ran; follows "
            "the close screen family expert_rankings_2026_11_16. Not served."
        ),
        "interval_low": round(b["interval_low"], 6),
        "interval_high": round(b["interval_high"], 6),
        "probability_positive": round(b["probability_positive"], 4),
        "sample_games": int(look["decided"] if units == "accuracy_points" else look["games"]),
        "sample_blocks": int(b["blocks"]),
        "plain_summary": summary,
    }


def cmd_batch(a):
    fit = json.loads((OUT / "fit.json").read_text())
    live = json.loads((OUT / "live.json").read_text())
    w = fit["week_block"]
    h = fit["halves_coefficient_per_sd"]
    t1 = (
        "FantasyPros ECR team expectation minus the as-of spread (gap as built in "
        "expert_rankings.py), one offset-logistic coefficient per SD on the QW served stage at "
        "the opener 2020-25, pushes kept at 0.5, leave-one-season-out with nested offsets. "
        f"Full coefficient {fit['coefficient_per_sd_full']:+.4f}/SD, folds positive "
        f"{fit['folds_positive']}/6, halves {h['2020-22']:+.4f}/{h['2023-25']:+.4f}; record "
        f"{w['record_term']} v QW {w['record_base']}; changed {w['changed']} "
        f"({w['changed_record_term']} v {w['changed_record_base']}); in-sample minus held-out "
        f"log loss {fit['in_sample_minus_held_out_log_loss_gain']:+.5f}; intercept-only "
        f"{fit['intercept_only_log_loss_gain']:+.5f}."
    )
    t2 = (
        "Frozen 2020-25 coefficient applied once to the played 2026 card, Weeks 2-10, at the "
        "card's pool line; ECR scrapes dated before each deadline, rank map and k from 2026 "
        f"earlier weeks. {live['games_with_term']} of {live['games']} card games carry the term; "
        f"record {live['record_term']} v card {live['record_base']}; changed {live['changed']} "
        f"({live['changed_record_term']} v {live['changed_record_base']})."
    )
    s1 = (
        "Adds the fantasy experts' view of each offense to the model's own pick probability "
        "for 2020-25 and checks the picks it would have changed."
    )
    s2 = "The same frozen expert adjustment applied to this season's played card through Week 10."
    cells = [
        cell(
            "expert_rankings_gap_served_opener_accuracy",
            w,
            "accuracy_points",
            "accuracy_points",
            *HISTORY,
            t1,
            s1,
        ),
        cell(
            "expert_rankings_gap_served_opener_log_loss",
            w,
            "log_loss_gain",
            "log_loss_improvement",
            *HISTORY,
            t1,
            s1,
        ),
        cell(
            "expert_rankings_gap_frozen_2026_w2_10_accuracy",
            live,
            "accuracy_points",
            "accuracy_points",
            SEASON,
            SEASON,
            t2,
            s2,
        ),
        cell(
            "expert_rankings_gap_frozen_2026_w2_10_log_loss",
            live,
            "log_loss_gain",
            "log_loss_improvement",
            SEASON,
            SEASON,
            t2,
            s2,
        ),
    ]
    for c in cells:
        if c["interval_high"] < 0:
            c["classification"] = "refuted_mechanism"
            c["closing_ground"] = "wrong_sign_resolved"
            c["classification_evidence"] = (
                "Declared positive; the whole week-block interval (and the season-block interval) "
                "lies below zero: the served-stage term changes picks that lose."
            )
    path = OUT / "weak_signals_batch.json"
    path.write_text(json.dumps({"cells": cells}, indent=1), encoding="utf-8")
    print(path, len(cells))


def main():
    p = argparse.ArgumentParser(
        description=(
            "Expert consensus rankings gap: frozen served-stage fit 2020-25, one 2026 Weeks 2-10 "
            "score, and the Weeks 11-18 forward log. Research only; serves nothing."
        )
    )
    sp = p.add_subparsers(dest="cmd", required=True)
    sp.add_parser("fit").set_defaults(fn=cmd_fit)
    sp.add_parser("live").set_defaults(fn=cmd_live)
    pg = sp.add_parser("pregame")
    pg.add_argument("--week", default="auto")
    pg.add_argument("--dry", action="store_true")
    pg.set_defaults(fn=cmd_pregame)
    sp.add_parser("score").set_defaults(fn=cmd_score)
    sp.add_parser("argv").set_defaults(fn=cmd_argv)
    sp.add_parser("batch").set_defaults(fn=cmd_batch)
    a = p.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
