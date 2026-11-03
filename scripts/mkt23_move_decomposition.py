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


if __name__ == "__main__":
    sys.exit(main())
