import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

sys.path.insert(0, str(Path(__file__).resolve().parent))
import lead110_absorption as U1
import mkt21_line_path as M

ROOT = Path(__file__).resolve().parents[1]
L6ART = ROOT / "artifacts" / "lead106_injury_language"
OUT = ROOT / "artifacts" / "lead110"
WEIGHTS = ROOT / "artifacts" / "lead110_absorption" / "report.json"
UNIT1_GAMES = ROOT / "artifacts" / "lead110_absorption" / "games.parquet"
MARKET = ROOT / "data" / "market" / "raw"
SPLASH = ROOT / "data" / "splash"
PUBLISHED = ROOT / "artifacts" / "clv_ledger" / "published_picks.parquet"
FEATURES = ROOT / "data" / "processed" / "game_features.parquet"
BOOK_CACHE = ROOT / "artifacts" / "sky04" / "book_series.parquet"
SEASON = 2026
REPS = 2000
SEED = 110
INTERVAL = (0.025, 0.975)
STAMP = "%Y%m%dT%H%M%SZ"
DAY = "%Y%m%d"
SECONDS_PER_MINUTE = 60.0
FAMILY = "lead110_market_absorption"
AFTER_DEADLINE = "read recorded after the deadline line"
NO_LADDER = "no scored ladder table for the week"
NO_ROWS = "no scored rows for the game and view"


def to_utc(text):
    return pd.to_datetime(text, format=STAMP, utc=True)


def forward_games():
    rows = []
    for f in sorted((L6ART / "forward").glob(f"{SEASON}_week*.json")):
        d = json.loads(f.read_text(encoding="utf-8"))
        for pair in sorted({r["game_id"] for r in d["rows"]}):
            away, home = pair.split("@")
            rows.append(
                {
                    "game_id": f"{SEASON}_{int(d['week']):02d}_{away}_{home}",
                    "week": int(d["week"]),
                    "view": d["view"],
                    "pair": pair,
                    "read_recorded_utc": to_utc(d["recorded_utc"]),
                    "read_source": f.name,
                }
            )
    return pd.DataFrame(rows)


def retro_games():
    ids = pd.read_parquet(UNIT1_GAMES, columns=["game_id", "week"])
    first = min(p.stat().st_mtime for p in (L6ART / "raw").glob("batch_*.json"))
    ids["week"] = ids["week"].astype(int)
    ids["view"] = "retrospective"
    ids["pair"] = ""
    ids["read_recorded_utc"] = pd.Timestamp(first, unit="s", tz="UTC")
    ids["read_source"] = "raw/batch_*.json"
    return ids


def read_terms(week):
    path = L6ART / "forward_score" / f"{SEASON}_week{week:02d}_scored.parquet"
    if not path.exists():
        return None, {}
    s = pd.read_parquet(path)
    weights = json.loads(WEIGHTS.read_text(encoding="utf-8"))["weights_spread_pts_per_out"]
    s["wt"] = s["position"].map(U1.POS_TO_GROUP).map(weights)
    bad = s["wt"].isna() | s["rp_raw"].isna() | s["lp_designation"].isna()
    dropped = s.loc[bad, "position"].fillna("missing").value_counts()
    total = len(s)
    s = s[~bad].copy()
    side = np.where(s["team"] == s["game_id"].str.split("@").str[1], 1.0, -1.0)
    s["v_read"] = side * s["wt"] * (1.0 - s["rp_raw"])
    s["v_ladder"] = side * s["wt"] * (1.0 - s["lp_designation"])
    agg = s.groupby(["game_id", "view"], as_index=False).agg(
        x_read=("v_read", "sum"),
        x_ladder=("v_ladder", "sum"),
        n_reads=("v_read", "size"),
        status_filed=("status_filed", "all"),
    )
    agg = agg.rename(columns={"game_id": "pair"})
    agg["x_extra"] = agg["x_read"] - agg["x_ladder"]
    info = {
        "read_rows": int(total),
        "rows_used": len(s),
        "rows_dropped": int(bad.sum()),
        "dropped_positions": {str(k): int(v) for k, v in dropped.items()},
        "weights": weights,
    }
    return agg, info


def load_quotes(start, end):
    cols = ["observed_at_utc", "nflverse_game_id", "bookmaker_key", "market", "home_spread_line"]
    lo, hi = start.strftime(DAY), end.strftime(DAY)
    frames = []
    for name in sorted(os.listdir(MARKET)):
        if not (lo <= name[:8] <= hi):
            continue
        f = MARKET / name / "quotes.parquet"
        if not f.exists():
            continue
        q = pd.read_parquet(f, columns=cols)
        keep = (
            (q["market"] == "spreads")
            & q["nflverse_game_id"].notna()
            & q["home_spread_line"].notna()
        )
        frames.append(q[keep].drop(columns=["market"]).drop_duplicates())
    q = pd.concat(frames, ignore_index=True)
    q["observed_at_utc"] = pd.to_datetime(q["observed_at_utc"], utc=True)
    return q, len(frames)


def minutes(delta):
    return float((delta.dt.total_seconds() / SECONDS_PER_MINUTE).median())


def game_lines(q, gid, t_pool, t_read, t_dead):
    g = q[
        (q["nflverse_game_id"] == gid)
        & (q["observed_at_utc"] >= t_pool)
        & (q["observed_at_utc"] <= t_dead)
    ].sort_values("observed_at_utc")
    pre = g[g["observed_at_utc"] <= t_read].groupby("bookmaker_key").tail(1)
    post = g[g["observed_at_utc"] > t_read].groupby("bookmaker_key").tail(1)
    pre = pre.set_index("bookmaker_key")
    post = post.set_index("bookmaker_key")
    both = pre.index.intersection(post.index)
    out = {
        "books_at_read": len(pre),
        "books_after_read": len(post),
        "books_paired": len(both),
        "line_at_read_all_books": float(pre["home_spread_line"].median()) if len(pre) else np.nan,
    }
    if len(both):
        a, b = pre.loc[both], post.loc[both]
        out["line_at_read"] = float(a["home_spread_line"].median())
        out["line_at_deadline"] = float(b["home_spread_line"].median())
        out["read_quote_age_min"] = minutes(t_read - a["observed_at_utc"])
        out["deadline_quote_lead_min"] = minutes(t_dead - b["observed_at_utc"])
    return out


def pool_week(week):
    files = sorted(SPLASH.glob(f"{SEASON}_week{week:02d}_*.json"))
    if not files:
        return {}, pd.NaT
    d = json.loads(files[-1].read_text(encoding="utf-8"))
    cap = pd.Timestamp(d["captured_at_et"]).tz_convert("UTC")
    return {g["game_id"]: float(g["home_spread"]) for g in d["games"]}, cap


def published(week):
    pub = pd.read_parquet(PUBLISHED)
    pub = pub[(pub["season"] == SEASON) & (pub["week"] == week)].copy()
    pub["t"] = pd.to_datetime(pub["published_at_utc"], utc=True)
    pub["p_home"] = np.where(
        pub["pick_team"] == pub["home_team"], pub["displayed_score"], 1.0 - pub["displayed_score"]
    )
    return pub.sort_values("t")


def served_at(pub, gid, t):
    rows = pub[(pub["game_id"] == gid) & (pub["t"] <= t)]
    if rows.empty:
        return np.nan, pd.NaT
    return float(rows["p_home"].iloc[-1]), rows["t"].iloc[-1]


def build():
    gf = pd.read_parquet(
        FEATURES,
        columns=[
            "game_id",
            "season",
            "week",
            "game_type",
            "gameday",
            "kickoff",
            "result",
            "spread_line",
        ],
    )
    gf = gf[(gf["season"] == SEASON) & (gf["game_type"] == "REG")].set_index("game_id")
    games = pd.concat([retro_games(), forward_games()], ignore_index=True)
    games = games[games["game_id"].isin(gf.index)].reset_index(drop=True)
    deadline = {
        gid: pd.Timestamp(
            M.deadline_ts(gf.at[gid, "gameday"], gf.at[gid, "kickoff"]), unit="s", tz="UTC"
        )
        for gid in games["game_id"]
    }
    weeks = sorted(games["week"].unique())
    terms = {w: read_terms(w) for w in weeks}
    pools = {w: pool_week(w) for w in weeks}
    pubs = {w: published(w) for w in weeks}
    live = [w for w in weeks if terms[w][0] is not None]
    caps = [pools[w][1] for w in live if pd.notna(pools[w][1])]
    ends = [deadline[g] for g, w in zip(games["game_id"], games["week"], strict=True) if w in live]
    q, snapshots = load_quotes(min(caps), max(ends))
    rows = []
    for g in games.itertuples():
        t_dead = deadline[g.game_id]
        row = {
            "game_id": g.game_id,
            "week": g.week,
            "view": g.view,
            "read_source": g.read_source,
            "read_recorded_utc": g.read_recorded_utc,
            "deadline_utc": t_dead,
            "kickoff_utc": gf.at[g.game_id, "kickoff"],
            "result": gf.at[g.game_id, "result"],
            "excluded_reason": None,
        }
        pool, cap = pools[g.week]
        row["pool_line"] = pool.get(g.game_id, np.nan)
        row["pool_captured_utc"] = cap
        table = terms[g.week][0]
        hit = None
        if g.read_recorded_utc >= t_dead:
            row["excluded_reason"] = AFTER_DEADLINE
        elif table is None:
            row["excluded_reason"] = NO_LADDER
        else:
            hit = table[(table["pair"] == g.pair) & (table["view"] == g.view)]
            if hit.empty:
                row["excluded_reason"] = NO_ROWS
        if row["excluded_reason"] is None:
            h = hit.iloc[0]
            row.update(
                {
                    "x_read": float(h["x_read"]),
                    "x_ladder": float(h["x_ladder"]),
                    "x_extra": float(h["x_extra"]),
                    "n_reads": int(h["n_reads"]),
                    "status_filed": float(bool(h["status_filed"])),
                }
            )
            row.update(game_lines(q, g.game_id, cap, g.read_recorded_utc, t_dead))
            p, when = served_at(pubs[g.week], g.game_id, t_dead)
            row["served_p_home"] = p
            row["served_published_utc"] = when
            row["served_p_home_at_read"] = served_at(pubs[g.week], g.game_id, g.read_recorded_utc)[
                0
            ]
        rows.append(row)
    t = pd.DataFrame(rows)
    for col in ("line_at_read", "line_at_deadline"):
        if col not in t:
            t[col] = np.nan
    t["move_post"] = t["line_at_deadline"] - t["line_at_read"]
    t["move_pre"] = t["line_at_read"] - t["pool_line"]
    t["move_total"] = t["line_at_deadline"] - t["pool_line"]
    t["cover_margin"] = t["result"] - t["pool_line"]
    decided = t["cover_margin"].notna() & (t["cover_margin"] != 0)
    t["home_cover"] = np.where(decided, (t["cover_margin"] > 0).astype(float), np.nan)
    info = {
        "terms": {int(w): terms[w][1] for w in live},
        "quote_snapshots_loaded": snapshots,
        "quote_rows": len(q),
        "published_score_range": {
            int(w): [
                float(pubs[w]["displayed_score"].min()),
                float(pubs[w]["displayed_score"].max()),
            ]
            for w in live
            if len(pubs[w])
        },
    }
    return t, gf, info


def coef(d, ycol, xcols):
    x = np.column_stack([np.ones(len(d)), d[xcols].to_numpy(float)])
    return float(np.linalg.lstsq(x, d[ycol].to_numpy(float), rcond=None)[0][1])


def permuted(d, col, rng):
    values = d[col].to_numpy(float).copy()
    for idx in d.groupby("week").indices.values():
        values[idx] = rng.permutation(values[idx])
    return values


def summarise(est, draws, null=None):
    draws = np.asarray(draws, float)
    out = {
        "effect": float(est),
        "interval": [
            float(np.quantile(draws, INTERVAL[0])),
            float(np.quantile(draws, INTERVAL[1])),
        ],
        "standard_error": float(draws.std(ddof=1)),
        "probability_positive": float((draws > 0).mean() + 0.5 * (draws == 0).mean()),
        "bootstrap_draws": len(draws),
    }
    if null is not None:
        null = np.asarray(null, float)
        out["perm_p_two_sided"] = float((np.abs(null) >= abs(est)).mean())
        out["perm_p_one_sided_positive"] = float((null >= est).mean())
    return out


def slope_look(d, ycol, xcols, rng):
    d = d.dropna(subset=[ycol, *xcols]).reset_index(drop=True)
    n = len(d)
    est = coef(d, ycol, xcols)
    draws = []
    for _ in range(REPS):
        b = d.iloc[rng.integers(0, n, n)]
        x = np.column_stack([np.ones(n), b[xcols].to_numpy(float)])
        if np.linalg.matrix_rank(x) < x.shape[1]:
            continue
        draws.append(coef(b, ycol, xcols))
    null = []
    for _ in range(REPS):
        p = d.copy()
        p[xcols[0]] = permuted(d, xcols[0], rng)
        null.append(coef(p, ycol, xcols))
    out = summarise(est, draws, null)
    out.update(
        {
            "n_games": int(n),
            "n_weeks": int(d["week"].nunique()),
            "y": ycol,
            "x": xcols,
            "y_sd": float(d[ycol].std()),
            "x_sd": float(d[xcols[0]].std()),
            "y_nonzero_games": int((d[ycol] != 0).sum()),
        }
    )
    return out


def sigmoid(v):
    return 1.0 / (1.0 + np.exp(-v))


def nll(b, z, x, y):
    eta = z + b * x
    v = float(np.mean(np.logaddexp(0.0, eta) - y * eta))
    return v if np.isfinite(v) else np.inf


def fit(z, x, y):
    return float(minimize_scalar(nll, args=(z, x, y)).x)


def loss(p, y):
    return -(y * np.log(p) + (1.0 - y) * np.log(1.0 - p))


def held_out(z, x, y, folds):
    p = np.empty(len(y))
    coefs = []
    for f in np.unique(folds):
        m = folds == f
        b = fit(z[~m], x[~m], y[~m])
        coefs.append(b)
        p[m] = sigmoid(z[m] + b * x[m])
    return p, coefs


def record(pick_home, y):
    wins = int((pick_home == (y == 1)).sum())
    return [wins, int(len(y) - wins)]


def added_term(d, xcol, rng):
    need = ["served_p_home", "home_cover"] + ([xcol] if xcol else [])
    d = d.dropna(subset=need).reset_index(drop=True)
    n = len(d)
    y = d["home_cover"].to_numpy(float)
    ps = d["served_p_home"].to_numpy(float)
    z = np.log(ps / (1.0 - ps))
    x = d[xcol].to_numpy(float) if xcol else np.ones(n)
    weeks = d["week"].to_numpy()
    by_week = len(np.unique(weeks)) > 1
    folds = weeks if by_week else np.arange(n)
    base = loss(ps, y)
    b_in = fit(z, x, y)
    p_in = sigmoid(z + b_in * x)
    p_out, coefs = held_out(z, x, y, folds)
    diff = base - loss(p_out, y)
    draws = [float(diff[rng.integers(0, n, n)].mean()) for _ in range(REPS)]
    null = None
    if xcol:
        null = []
        for _ in range(REPS):
            pp, _ = held_out(z, permuted(d, xcol, rng), y, folds)
            null.append(float(np.mean(base - loss(pp, y))))
    out = summarise(float(diff.mean()), draws, null)
    base_home = ps >= 0.5
    new_home = p_out >= 0.5
    changed = base_home != new_home
    out.update(
        {
            "n_games": int(n),
            "n_weeks": len(np.unique(weeks)),
            "term": xcol or "intercept",
            "out_of_sample_scheme": "leave_one_week_out" if by_week else "leave_one_game_out",
            "decisive_games": int(changed.sum()),
            "decisive_record_with_term": record(new_home[changed], y[changed]),
            "record_served": record(base_home, y),
            "record_with_term": record(new_home, y),
            "log_loss_served": float(base.mean()),
            "log_loss_coin": float(np.log(2.0)),
            "log_loss_with_term_out_of_sample": float(loss(p_out, y).mean()),
            "log_loss_with_term_in_sample": float(loss(p_in, y).mean()),
            "in_sample_gain": float(np.mean(base - loss(p_in, y))),
            "in_minus_out_gap": float(np.mean(base - loss(p_in, y)) - diff.mean()),
            "brier_served": float(np.mean((ps - y) ** 2)),
            "brier_with_term_out_of_sample": float(np.mean((p_out - y) ** 2)),
            "coefficient_in_sample": float(b_in),
            "coefficient_folds": {
                "mean": float(np.mean(coefs)),
                "sd": float(np.std(coefs, ddof=1)),
                "min": float(np.min(coefs)),
                "max": float(np.max(coefs)),
            },
            "served_pick_probability_mean": float(np.maximum(ps, 1.0 - ps).mean()),
            "served_pick_hit_rate": float((base_home == (y == 1)).mean()),
            "term_pick_probability_mean": float(np.maximum(p_out, 1.0 - p_out).mean()),
            "term_pick_hit_rate": float((new_home == (y == 1)).mean()),
        }
    )
    return out


def convention(t, gf):
    used = t.dropna(subset=["line_at_read_all_books", "pool_line"])
    out = {
        "store_vs_pool_games": len(used),
        "store_vs_pool_corr": float(
            np.corrcoef(used["line_at_read_all_books"], used["pool_line"])[0, 1]
        ),
        "store_minus_pool_mean": float((used["line_at_read_all_books"] - used["pool_line"]).mean()),
        "store_plus_pool_mean_abs": float(
            (used["line_at_read_all_books"] + used["pool_line"]).abs().mean()
        ),
        "store_minus_pool_mean_abs": float(
            (used["line_at_read_all_books"] - used["pool_line"]).abs().mean()
        ),
    }
    if BOOK_CACHE.exists():
        b = pd.read_parquet(BOOK_CACHE)
        b = b[b["nflverse_game_id"].str[:4] == str(SEASON)].sort_values("observed_at_utc")
        last = b.groupby(["nflverse_game_id", "bookmaker_key"])["home_spread_line"].last()
        last = last.groupby(level=0).median().rename("store_line")
        last.index = last.index.astype(str)
        j = gf.join(last, how="inner").dropna(subset=["spread_line", "store_line"])
        out["cache_vs_schedule_games"] = len(j)
        out["cache_vs_schedule_corr"] = float(np.corrcoef(j["spread_line"], j["store_line"])[0, 1])
    return out


def cell(name, description, units, r, plain, note):
    return {
        "name": f"lead110_fwd_{name}",
        "description": description,
        "effect": round(r["effect"], 4),
        "effect_units": units,
        "standard_error": round(r["standard_error"], 4),
        "interval_low": round(r["interval"][0], 4),
        "interval_high": round(r["interval"][1], 4),
        "probability_positive": round(r["probability_positive"], 4),
        "sample_games": r["n_games"],
        "sample_blocks": r["n_weeks"],
        "notes": note,
        "plain_summary": plain,
    }


def slope_note(r, looks):
    return (
        f"within-week permutation p two-sided {r['perm_p_two_sided']:.3f}; "
        f"{r['y_nonzero_games']} of {r['n_games']} games have a nonzero move; "
        f"{looks} looks in the family"
    )


def term_note(r, looks):
    perm = r.get("perm_p_one_sided_positive")
    perm_text = ""
    if perm is not None:
        perm_text = (
            f"within-week permutation p one-sided {perm:.3f}, "
            f"two-sided {r['perm_p_two_sided']:.3f}; "
        )
    w, lo = r["decisive_record_with_term"]
    return (
        f"{r['out_of_sample_scheme']}; decisive games {w}-{lo}; "
        f"in-sample gain {r['in_sample_gain']:.4f}; "
        f"coefficient {r['coefficient_in_sample']:.3f}; {perm_text}{looks} looks in the family"
    )


def batch(rep, looks):
    weeks = rep["weeks_used"]
    span = f"2026 forward weeks {weeks}, read recorded before the deadline"
    move = "spread points of line move per spread point of expected availability difference"
    cells = [
        cell(
            "move_after_read_on_read",
            f"Slope of the read-to-deadline spread move on the read availability gap; {span}",
            "slope_difference",
            rep["a1"],
            "After the injury wording was read, did the betting line then move toward the team the "
            "read called healthier? Too few games to say yet.",
            slope_note(rep["a1"], looks),
        ),
        cell(
            "move_after_read_on_read_minus_ladder",
            f"Slope of the read-to-deadline move on read-minus-ladder availability, "
            f"ladder held; {span}",
            "slope_difference",
            rep["a2"],
            "Did the line later move toward what the wording read saw beyond the official injury "
            "tags? Too few games to say yet.",
            slope_note(rep["a2"], looks),
        ),
        cell(
            "move_after_read_on_ladder_control",
            f"Control: slope of the read-to-deadline move on the designation-ladder gap; {span}",
            "slope_difference",
            rep["a3"],
            "Control check: did the line move after the read toward the team the official injury "
            "tags alone called healthier?",
            slope_note(rep["a3"], looks),
        ),
        cell(
            "cover_pool_line_read_term",
            f"Held-out log-loss gain from the read difference as one added term to the served "
            f"probability, cover at the Tuesday pool line; {span}",
            "log_loss_improvement",
            rep["b1"],
            "Adding the wording read's health gap to the card's own probability: did it predict "
            "covers at the frozen pool line any better? Too few games to say yet.",
            term_note(rep["b1"], looks),
        ),
        cell(
            "cover_pool_line_intercept_control",
            f"Control: held-out log-loss gain from an intercept-only shift of the served "
            f"probability; {span}",
            "log_loss_improvement",
            rep["b0"],
            "Control check: a plain home-or-away tilt added to the card's probability, with no "
            "injury information at all.",
            term_note(rep["b0"], looks),
        ),
        cell(
            "cover_pool_line_ladder_term_control",
            f"Control: held-out log-loss gain from the designation-ladder difference as the "
            f"added term; {span}",
            "log_loss_improvement",
            rep["b2"],
            "Control check: the official injury tags' health gap added to the card's probability "
            "in place of the wording read.",
            term_note(rep["b2"], looks),
        ),
        cell(
            "absorbed_share_pool_line_to_deadline",
            f"Slope of the pool-line-to-deadline line gap on the read difference ({move}); {span}",
            "slope_difference",
            rep["c1"],
            "How much of the health gap the wording read saw is already in the gap between the "
            "frozen pool line and the line at the pick deadline.",
            slope_note(rep["c1"], looks),
        ),
        cell(
            "absorbed_share_pool_line_to_read",
            f"Slope of the pool-line-to-read-time line gap on the read difference ({move}); {span}",
            "slope_difference",
            rep["c2"],
            "How much of the health gap the wording read saw was already in the betting line at "
            "the moment the read was made.",
            slope_note(rep["c2"], looks),
        ),
    ]
    return {
        "family": FAMILY,
        "league": "nfl",
        "season_start": SEASON,
        "season_end": SEASON,
        "classification": "unresolved_below_power",
        "classification_evidence": (
            f"{rep['games_used']} games in {len(weeks)} forward week(s), {looks} looks; one slate "
            "cannot resolve a sign, the held-out cover cells refit a coefficient with one game "
            "left out and sit inside their permutation null (see notes); no reliability estimate, "
            "no positive control"
        ),
        "source": "artifacts/lead110/report.json",
        "category": "market",
        "cells": cells,
    }


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    t, gf, info = build()
    used = t[t["excluded_reason"].isna()].reset_index(drop=True)
    excluded = t[t["excluded_reason"].notna()]
    moves = used.dropna(subset=["move_post"])
    covers = used.dropna(subset=["served_p_home", "home_cover"])
    rep = {
        "run_at_utc": str(pd.Timestamp.now(tz="UTC")),
        "run_at_et": str(pd.Timestamp.now(tz=M.L.ET)),
        "family": FAMILY,
        "games_with_reads": len(t),
        "games_used": len(used),
        "weeks_used": sorted(int(w) for w in used["week"].unique()),
        "excluded_by_reason": {
            str(k): int(v) for k, v in excluded["excluded_reason"].value_counts().items()
        },
        "excluded_by_week": {
            str(k): int(v) for k, v in excluded.groupby("week")["game_id"].size().items()
        },
        "excluded_games": [
            {
                "game_id": r.game_id,
                "reason": r.excluded_reason,
                "read_recorded_utc": str(r.read_recorded_utc),
                "deadline_utc": str(r.deadline_utc),
            }
            for r in excluded.itertuples()
        ],
        "move_games": len(moves),
        "no_quote_between_read_and_deadline": sorted(set(used["game_id"]) - set(moves["game_id"])),
        "cover_games": len(covers),
        "no_result_or_no_served_probability": sorted(set(used["game_id"]) - set(covers["game_id"])),
        "served_side_changed_after_read": int(
            ((covers["served_p_home_at_read"] >= 0.5) != (covers["served_p_home"] >= 0.5)).sum()
        ),
        "served_probability_missing_at_read": int(covers["served_p_home_at_read"].isna().sum()),
        "inputs": info,
        "convention": convention(used, gf),
    }
    looks = 8
    rep["a1"] = slope_look(moves, "move_post", ["x_read"], rng)
    rep["a2"] = slope_look(moves, "move_post", ["x_extra", "x_ladder"], rng)
    rep["a3"] = slope_look(moves, "move_post", ["x_ladder"], rng)
    rep["c1"] = slope_look(moves, "move_total", ["x_read"], rng)
    rep["c2"] = slope_look(moves, "move_pre", ["x_read"], rng)
    rep["b0"] = added_term(covers, None, rng)
    rep["b1"] = added_term(covers, "x_read", rng)
    rep["b2"] = added_term(covers, "x_ladder", rng)
    rep["looks"] = looks
    t.to_csv(OUT / "games.csv", index=False)
    t.to_parquet(OUT / "games.parquet")
    (OUT / "report.json").write_text(json.dumps(rep, indent=1, default=str), encoding="utf-8")
    (OUT / "registry_batch.json").write_text(
        json.dumps(batch(rep, looks), indent=1), encoding="utf-8"
    )
    show = [
        "game_id",
        "view",
        "read_recorded_utc",
        "deadline_utc",
        "pool_line",
        "line_at_read",
        "line_at_deadline",
        "books_paired",
        "read_quote_age_min",
        "deadline_quote_lead_min",
        "x_read",
        "x_ladder",
        "result",
        "home_cover",
        "served_p_home_at_read",
        "served_p_home",
        "status_filed",
    ]
    with pd.option_context("display.width", 320, "display.max_columns", 40):
        print(used[show].to_string(float_format=lambda v: f"{v:.3f}"))
    slim = {k: v for k, v in rep.items() if k != "excluded_games"}
    print(json.dumps(slim, indent=1, default=str))


if __name__ == "__main__":
    main()
