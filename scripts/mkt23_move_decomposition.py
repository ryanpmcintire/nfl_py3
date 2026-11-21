import glob
import json
import sys
from pathlib import Path

import mkt21_line_path as M
import numpy as np
import pandas as pd
import sky04_lead_lag as L

SEASONS = M.SEASONS
OUT = L.OUT.parent / "mkt23"
FAMILY = "mkt23_move_decomposition"
CATEGORIES = ("pure_side", "one_team", "two_team_same", "two_team_opposite")
KEYS = ["nflverse_game_id", "bookmaker_key", "observed_at_utc"]
LOOKS = len(CATEGORIES) + 1


def total_series():
    rows = []
    cols = ["observed_at_utc", "nflverse_game_id", "bookmaker_key", "market", "line"]
    for d in sorted(glob.glob(str(L.ROOT / "data/market/raw/*"))):
        name = Path(d).name
        if name[:4] not in [str(x) for x in SEASONS] or name.endswith("futures"):
            continue
        f = Path(d) / "quotes.parquet"
        if not f.exists():
            continue
        q = pd.read_parquet(f, columns=cols)
        q = q[(q.market == "totals") & q.nflverse_game_id.notna()]
        if len(q):
            rows.append(
                q.groupby(
                    ["nflverse_game_id", "bookmaker_key", "observed_at_utc"], as_index=False
                ).agg(total_line=("line", "median"))
            )
    t = pd.concat(rows, ignore_index=True)
    t["observed_at_utc"] = pd.to_datetime(t.observed_at_utc, utc=True)
    return t.drop_duplicates(["nflverse_game_id", "bookmaker_key", "observed_at_utc"])


def consensus_at(frame, value, ts_by_game):
    out = {}
    f = frame.sort_values("observed_at_utc")
    for gid, g in f.groupby("nflverse_game_id"):
        if gid not in ts_by_game:
            continue
        t0, t1 = ts_by_game[gid]
        res = []
        for t in (t0, t1):
            gg = g[g.observed_at_utc <= pd.Timestamp(t, unit="s", tz="UTC")]
            if gg.empty:
                res.append((np.nan, 0))
                continue
            last = gg.groupby("bookmaker_key")[value].last()
            res.append((float(last.median()), len(last)))
        out[gid] = res
    return out


def rate_diff(a, b, rng):
    ma = a[rng.integers(0, len(a), size=(L.BOOT, len(a)))].mean(axis=1)
    mb = b[rng.integers(0, len(b), size=(L.BOOT, len(b)))].mean(axis=1)
    d = (ma - mb) * 100
    return [
        float((a.mean() - b.mean()) * 100),
        float(np.percentile(d, 2.5)),
        float(np.percentile(d, 97.5)),
        float((d > 0).mean()),
    ]


def main():
    OUT.mkdir(exist_ok=True)
    s = L.book_series()
    s = s[s.nflverse_game_id.str[:4].isin([str(x) for x in SEASONS])]
    t = total_series()
    g = pd.read_parquet(L.PERGAME)
    g = g[(g.game_type == "REG") & g.season.isin(SEASONS)]
    gf = pd.read_parquet(
        L.ROOT / "data/processed/game_features.parquet", columns=["game_id", "kickoff"]
    )
    g = g.merge(gf, on="game_id", how="left")
    g = g[g.kickoff.notna() & (g.margin_vs_open != 0)].reset_index(drop=True)
    ts = {
        gid: (L.decision_ts(gd), M.deadline_ts(gd, ko))
        for gid, gd, ko in zip(g.game_id, g.gameday, g.kickoff)  # noqa: B905
    }
    sp = consensus_at(s, "home_spread_line", ts)
    tt = consensus_at(t, "total_line", ts)
    ids = [x for x in g.game_id if x in sp and x in tt]
    g = g[g.game_id.isin(ids)].reset_index(drop=True)
    g["hs0"] = [sp[x][0][0] for x in g.game_id]
    g["hs1"] = [sp[x][1][0] for x in g.game_id]
    g["t0"] = [tt[x][0][0] for x in g.game_id]
    g["t1"] = [tt[x][1][0] for x in g.game_id]
    g = g.dropna(subset=["hs0", "hs1", "t0", "t1"]).reset_index(drop=True)
    d_hs = (g.hs1 - g.hs0).round(4)
    d_t = (g.t1 - g.t0).round(4)
    nz = np.concatenate([np.abs(d_hs[d_hs != 0].values), np.abs(d_t[d_t != 0].values)])
    step = float(nz.min())
    resolvable = (d_hs.abs() >= step).values
    g["d_hs"], g["d_t"] = d_hs, d_t
    g["d_tt_home"] = d_t / 2 + d_hs / 2
    g["d_tt_away"] = d_t / 2 - d_hs / 2
    mv = g[resolvable].copy()
    ah, aa = mv.d_tt_home.abs().values, mv.d_tt_away.abs().values
    tol = step / 2 / 2
    cat = np.where(
        mv.d_t.abs().values < step,
        "pure_side",
        np.where(
            (ah < tol) | (aa < tol),
            "one_team",
            np.where(
                np.sign(mv.d_tt_home.values) == np.sign(mv.d_tt_away.values),
                "two_team_same",
                "two_team_opposite",
            ),
        ),
    )
    mv["cat"] = cat
    toward_home = np.sign(mv.d_hs.values)
    mv["follow"] = np.where(toward_home > 0, mv.home_covered, 1 - mv.home_covered).astype(float)
    same_dir = (np.sign(mv.d_t.values) == np.sign(mv.d_hs.values)) & (mv.d_t.abs().values >= step)
    rng = np.random.default_rng(L.SEED)
    res = {
        "family": FAMILY,
        "quote_step_measured": step,
        "games_matched": len(g),
        "games_by_season": {
            str(k): int(v) for k, v in g.season.value_counts().sort_index().items()
        },
        "spread_moved_games": len(mv),
        "spread_unmoved_total_moved": int(((~resolvable) & (d_t.abs() >= step).values).sum()),
        "share_spread_moves_with_same_direction_total_move": float(same_dir.mean()),
        "share_spread_moves_with_opposite_direction_total_move": float(
            (
                (np.sign(mv.d_t.values) == -np.sign(mv.d_hs.values)) & (mv.d_t.abs().values >= step)
            ).mean()
        ),
        "share_spread_moves_total_flat": float((mv.d_t.abs().values < step).mean()),
        "categories": {},
        "look_count": LOOKS,
    }
    for c in CATEGORIES:
        sub = mv[mv.cat == c]
        v = sub.follow.values
        if len(v) == 0:
            continue
        ent = {"n": len(v), "share": float(len(v) / len(mv)), "follow_rate": float(v.mean())}
        ent["follow_vs_50_points"] = L.boot((v - 0.5) * 100, rng)
        ent["by_season"] = {
            str(k): [len(x), float(x.mean())] for k, x in sub.groupby("season").follow
        }
        ent["season_halves"] = {
            "even_game_ids": float(v[::2].mean()),
            "odd_game_ids": float(v[1::2].mean()),
        }
        res["categories"][c] = ent
    a = mv[mv.cat == "one_team"].follow.values
    b = mv[mv.cat == "two_team_same"].follow.values
    res["one_team_minus_two_team_same_points"] = rate_diff(a, b, rng)
    res["one_team_minus_two_team_all_points"] = rate_diff(
        a, mv[mv.cat.isin(["two_team_same", "two_team_opposite"])].follow.values, rng
    )
    by_s = {}
    for sn in SEASONS:
        aa_ = mv[(mv.cat == "one_team") & (mv.season == sn)].follow.values
        bb_ = mv[(mv.cat == "two_team_same") & (mv.season == sn)].follow.values
        by_s[str(sn)] = float((aa_.mean() - bb_.mean()) * 100) if len(aa_) and len(bb_) else None
    res["one_minus_two_same_by_season_points"] = by_s
    mv.to_parquet(OUT / "move_table.parquet")
    (OUT / "decomposition.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


LIVE_SEASON = 2026
LIVE_WEEKS = tuple(range(1, 11))
PICK_CYCLE_DAYS = 7
SKIP_TAGS = ("reparse", "halves", "probe", "futures", "ncaaf", "h2h", "dup")
QCOLS = [
    "observed_at_utc",
    "nflverse_game_id",
    "bookmaker_key",
    "market",
    "home_spread_line",
    "line",
]
GAME_COLS = ["game_id", "season", "week", "gameday", "kickoff", "game_type", "home_cover"]


def live_quotes():
    parts = {"spreads": [], "totals": []}
    col = {"spreads": "home_spread_line", "totals": "line"}
    for d in sorted(glob.glob(str(L.ROOT / f"data/market/raw/{LIVE_SEASON}*"))):
        name = Path(d).name
        f = Path(d) / "quotes.parquet"
        if any(t in name for t in SKIP_TAGS) or not f.exists():
            continue
        q = pd.read_parquet(f, columns=QCOLS)
        q = q[q.nflverse_game_id.notna()]
        q = q[q.nflverse_game_id.str.startswith(f"{LIVE_SEASON}_")]
        for m in parts:
            x = q[q.market == m]
            if len(x):
                parts[m].append(
                    x.groupby(KEYS, as_index=False).agg(v=(col[m], "median")).assign(capture=name)
                )
    out = {}
    for m, p in parts.items():
        x = pd.concat(p, ignore_index=True)
        x["observed_at_utc"] = pd.to_datetime(x.observed_at_utc, utc=True)
        x = x.dropna(subset=["v"]).sort_values("observed_at_utc")
        out[m] = x.drop_duplicates(KEYS, keep="last")
    return out


def matched_anchor(g, t0, t1):
    t0 = pd.Timestamp(t0, unit="s", tz="UTC")
    t1 = pd.Timestamp(t1, unit="s", tz="UTC")
    rows = []
    g = g[g.observed_at_utc > t0 - pd.Timedelta(days=PICK_CYCLE_DAYS)]
    for book, b in g.groupby("bookmaker_key"):
        end = b[b.observed_at_utc <= t1]
        if end.empty:
            continue
        pre = end[end.observed_at_utc <= t0]
        start = pre.iloc[-1] if len(pre) else end.iloc[0]
        stop = end.iloc[-1]
        if stop.observed_at_utc <= start.observed_at_utc:
            continue
        rows.append(
            (book, start.v, stop.v, start.observed_at_utc, stop.observed_at_utc, len(pre) > 0)
        )
    if not rows:
        return None
    r = pd.DataFrame(rows, columns=["book", "v0", "v1", "at0", "at1", "pre_noon"])
    return {
        "v0": float(r.v0.median()),
        "v1": float(r.v1.median()),
        "books": len(r),
        "book_names": ",".join(sorted(r.book)),
        "at0": r.at0.max(),
        "at1": r.at1.max(),
        "pre_noon_share": float(r.pre_noon.mean()),
    }


def categorize(mv, step):
    mv["d_tt_home"] = mv.d_t / 2 + mv.d_hs / 2
    mv["d_tt_away"] = mv.d_t / 2 - mv.d_hs / 2
    ah, aa = mv.d_tt_home.abs().values, mv.d_tt_away.abs().values
    tol = step / 2 / 2
    cat = np.where(
        mv.d_t.abs().values < step,
        "pure_side",
        np.where(
            (ah < tol) | (aa < tol),
            "one_team",
            np.where(
                np.sign(mv.d_tt_home.values) == np.sign(mv.d_tt_away.values),
                "two_team_same",
                "two_team_opposite",
            ),
        ),
    )
    mv["cat"] = np.where(mv.d_hs.abs().values >= step, cat, None)
    return mv


def history_all(step):
    s = L.book_series()
    s = s[s.nflverse_game_id.str[:4].isin([str(x) for x in SEASONS])]
    t = total_series()
    g = pd.read_parquet(L.ROOT / "data/processed/game_features.parquet", columns=GAME_COLS)
    g = g[(g.game_type == "REG") & g.season.isin(SEASONS) & g.kickoff.notna()].reset_index(
        drop=True
    )
    ts = {
        gid: (L.decision_ts(gd), M.deadline_ts(gd, ko))
        for gid, gd, ko in zip(g.game_id, g.gameday, g.kickoff)  # noqa: B905
    }
    sp = consensus_at(s, "home_spread_line", ts)
    tt = consensus_at(t, "total_line", ts)
    g = g[g.game_id.isin([x for x in g.game_id if x in sp and x in tt])].reset_index(drop=True)
    g["hs0"] = [sp[x][0][0] for x in g.game_id]
    g["hs1"] = [sp[x][1][0] for x in g.game_id]
    g["t0"] = [tt[x][0][0] for x in g.game_id]
    g["t1"] = [tt[x][1][0] for x in g.game_id]
    g = g.dropna(subset=["hs0", "hs1", "t0", "t1"]).reset_index(drop=True)
    g["d_hs"] = (g.hs1 - g.hs0).round(4)
    g["d_t"] = (g.t1 - g.t0).round(4)
    return categorize(g, step)


def live_table(step):
    q = live_quotes()
    by = {m: dict(tuple(x.groupby("nflverse_game_id"))) for m, x in q.items()}
    g = pd.read_parquet(L.ROOT / "data/processed/game_features.parquet", columns=GAME_COLS)
    g = g[(g.game_type == "REG") & (g.season == LIVE_SEASON) & g.week.isin(LIVE_WEEKS)]
    rows = []
    for r in g.itertuples():
        t0, t1 = L.decision_ts(r.gameday), M.deadline_ts(r.gameday, r.kickoff)
        row = {"game_id": r.game_id, "season": r.season, "week": r.week, "home_cover": r.home_cover}
        for m, tag in (("spreads", "s"), ("totals", "t")):
            a = matched_anchor(by[m][r.game_id], t0, t1) if r.game_id in by[m] else None
            for k in ("v0", "v1", "books", "book_names", "at0", "at1", "pre_noon_share"):
                row[f"{tag}_{k}"] = a[k] if a else None
        rows.append(row)
    d = pd.DataFrame(rows)
    d["d_hs"] = (d.s_v1.astype(float) - d.s_v0.astype(float)).round(4)
    d["d_t"] = (d.t_v1.astype(float) - d.t_v0.astype(float)).round(4)
    both = d.d_hs.notna() & d.d_t.notna()
    d = d.join(
        categorize(d.loc[both, ["d_hs", "d_t"]].copy(), step)[["d_tt_home", "d_tt_away", "cat"]]
    )
    return d


def et_times(x):
    v = pd.to_datetime(x.dropna(), utc=True).dt.tz_convert(L.ET)
    return sorted(set(v.dt.strftime("%a %m-%d %H:%M")))


def coverage(d):
    out = {}
    for w, x in d.groupby("week"):
        out[str(w)] = {
            "games": len(x),
            "spread_move_measurable": int(x.d_hs.notna().sum()),
            "spread_books_median": float(x.s_books.astype(float).median()),
            "spread_books_seen": sorted({b for v in x.s_book_names.dropna() for b in v.split(",")}),
            "spread_start_at_or_before_tuesday_noon_share": float(
                x.s_pre_noon_share.astype(float).mean()
            ),
            "total_move_measurable": int(x.d_t.notna().sum()),
            "total_books_median": float(x.t_books.astype(float).median())
            if x.t_books.notna().any()
            else 0.0,
            "total_books_seen": sorted({b for v in x.t_book_names.dropna() for b in v.split(",")}),
            "total_start_at_or_before_tuesday_noon_share": float(
                x.t_pre_noon_share.astype(float).mean()
            )
            if x.t_pre_noon_share.notna().any()
            else None,
            "total_start_captures_et": et_times(x.t_at0),
            "total_end_captures_et": et_times(x.t_at1),
            "spread_start_captures_et": et_times(x.s_at0),
            "spread_end_captures_et": et_times(x.s_at1),
            "both_measurable": int((x.d_hs.notna() & x.d_t.notna()).sum()),
            "spread_moved": int((x.d_hs.abs() > 0).sum()),
            "total_moved": int((x.d_t.abs() > 0).sum()),
        }
    return out


def extend():
    unit1 = json.loads((OUT / "decomposition.json").read_text())
    step = unit1["quote_step_measured"]
    h = history_all(step)
    old = pd.read_parquet(OUT / "move_table.parquet")[["game_id", "d_hs", "d_t", "cat"]]
    chk = h.merge(old, on="game_id", suffixes=("", "_u1"))
    h.to_parquet(OUT / "move_table_history_all.parquet")
    d = live_table(step)
    d.to_parquet(OUT / "move_table_2026.parquet")
    mv = d[d.cat.notna()].copy()
    mv["follow"] = np.where(np.sign(mv.d_hs) > 0, mv.home_cover, 1 - mv.home_cover).astype(float)
    graded = mv[mv.home_cover.notna()]
    rng = np.random.default_rng(L.SEED)
    res = {
        "quote_step_from_unit1": step,
        "history_all_games": len(h),
        "history_all_by_season": {
            str(k): int(v) for k, v in h.season.value_counts().sort_index().items()
        },
        "history_unit1_rows_reproduced": len(chk),
        "history_unit1_d_hs_equal": int((chk.d_hs == chk.d_hs_u1).sum()),
        "history_unit1_cat_equal": int((chk.cat == chk.cat_u1).sum()),
        "live_games": len(d),
        "live_both_measurable": int((d.d_hs.notna() & d.d_t.notna()).sum()),
        "live_spread_moved_with_total": len(mv),
        "live_graded_spread_moved": len(graded),
        "live_coverage_by_week": coverage(d),
        "live_categories_at_pool_line": {},
        "look_count": LOOKS,
    }
    for c in CATEGORIES:
        v = graded[graded.cat == c].follow.values
        if len(v):
            res["live_categories_at_pool_line"][c] = {
                "n": len(v),
                "share": float(len(v) / len(graded)),
                "follow_rate": float(v.mean()),
                "follow_vs_50_points": L.boot((v - 0.5) * 100, rng),
            }
    (OUT / "decomposition_2026.json").write_text(json.dumps(res, indent=1, default=str))
    print(json.dumps(res, indent=1, default=str))


if __name__ == "__main__":
    sys.exit(extend() if "--extend" in sys.argv else main())
