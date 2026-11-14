import glob
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import lead106_injury_language as L6
import mkt21_line_path as M
import mkt25_opener_revision as O
import sky04_lead_lag as L

ROOT = L6.ROOT
OUT = ROOT / "artifacts" / "lead110_absorption"
SEASON = 2026
FIRST_WEEK = 2
LAST_WEEK = 8
WEIGHT_FIRST_SEASON = 2011
WEIGHT_LAST_SEASON = 2025
WINDOW = O.WINDOW_WEEKS
BOOT = O.BOOT
SEED = 110
DECISIVE_FRACTION = 0.5
GROUPS = {
    "QB": ["QB"],
    "OL": ["T", "G", "C"],
    "RECV": ["WR", "TE"],
    "RB": ["RB", "FB"],
    "DL": ["DE", "DT"],
    "LB": ["LB"],
    "DB": ["CB", "S"],
    "ST": ["K", "P", "LS"],
}
POS_TO_GROUP = {p: k for k, v in GROUPS.items() for p in v}


def group_counts():
    path = sorted(glob.glob(str(ROOT / "data/raw/nflverse_injuries/*/injuries.parquet")))[-1]
    inj = pd.read_parquet(path)
    inj = inj[(inj.game_type == "REG") & inj.report_status.isin(O.OUT_STATUS)].copy()
    inj["grp"] = inj.position.map(POS_TO_GROUP)
    inj = inj.dropna(subset=["grp"])
    pc = inj.groupby(["season", "week", "team", "grp"]).size().unstack(fill_value=0)
    return pc.reindex(columns=list(GROUPS), fill_value=0)


def implied_table(g):
    rows = []
    sub = g[g.season.between(WEIGHT_FIRST_SEASON, WEIGHT_LAST_SEASON) & (g.week >= 2)]
    for (s, w), gw in sub.groupby(["season", "week"]):
        win = g[(g.slot < O.slot(s, w)) & (g.slot >= O.slot(s, w) - WINDOW)]
        hfa, r, _ = O.fit_ratings(win)
        for x in gw.itertuples():
            if x.home_team in r and x.away_team in r:
                rows.append((x.game_id, hfa + r[x.home_team] - r[x.away_team]))
    return pd.DataFrame(rows, columns=["game_id", "implied"]).set_index("game_id").implied


def position_weights(g, pc):
    imp = implied_table(g)
    gg = g[g.game_id.isin(imp.index)].copy()
    gg["implied"] = gg.game_id.map(imp)
    cols = []
    for k in GROUPS:
        keys_h = list(zip(gg.season, gg.week, gg.home_team, strict=True))
        keys_a = list(zip(gg.season, gg.week, gg.away_team, strict=True))
        h = np.array([pc[k].get(key, 0) for key in keys_h])
        a = np.array([pc[k].get(key, 0) for key in keys_a])
        gg["d_" + k] = h - a
        cols.append("d_" + k)
    y = gg.spread_line - gg.implied
    b = O.ols(gg[cols].to_numpy(float), y.to_numpy(float))
    return dict(zip(GROUPS, b[1:], strict=True)), len(gg)


def sample_rows():
    pop = pd.read_parquet(L6.ART / "population.parquet")
    pop = L6.id_outcomes(pop)
    s0 = pd.read_parquet(L6.ART / "sample.parquet")[["rid", "gsis_id", "week"]]
    ver = pop[pop["verified"]].copy()
    ver["practice_status"] = ver["practice_status"].fillna("none")
    s = s0.merge(ver.drop(columns=["rid"], errors="ignore"), on=["gsis_id", "week"], how="inner")
    reads = L6.parse_reads()
    s = s[s["rid"].isin(reads)].copy()
    n = len(s)
    clip = 1.0 / (n + 2)
    s["rp"] = s["rid"].map(lambda r: min(max(reads[r][0], clip), 1 - clip))
    out = {}
    for nm, keys in (
        ("des", ["designation"]),
        ("desprac", ["designation", "practice_status"]),
    ):
        tab = L6.ladder_tables(ver, keys)
        out[nm] = [L6.ladder_pred(r, tab, keys)[0] for _, r in s.iterrows()]
    s["lp_des"] = out["des"]
    s["lp_desprac"] = out["desprac"]
    return s[s["week"].between(FIRST_WEEK, LAST_WEEK)].reset_index(drop=True)


def consensus(frame, ts):
    f = frame.sort_values("observed_at_utc")
    res = {}
    for gid, grp in f.groupby("nflverse_game_id"):
        if gid not in ts:
            continue
        pair = []
        for t in ts[gid]:
            gg = grp[grp.observed_at_utc <= pd.Timestamp(t, unit="s", tz="UTC")]
            pair.append(
                np.nan
                if gg.empty
                else float(gg.groupby("bookmaker_key").home_spread_line.last().median())
            )
        res[gid] = pair
    return res


def games_frame(s, w):
    gf = pd.read_parquet(ROOT / "data/processed/game_features.parquet")
    gf = gf[
        (gf.season == SEASON) & gf.week.between(FIRST_WEEK, LAST_WEEK) & (gf.game_type == "REG")
    ]
    rows = []
    for x in gf.itertuples():
        for side, team in ((1, x.home_team), (-1, x.away_team)):
            sub = s[(s.team == team) & (s.week == x.week)]
            if sub.empty:
                continue
            wt = sub.position.map(POS_TO_GROUP).map(w)
            ok = wt.notna()
            wt = wt[ok]
            sub = sub[ok]
            rows.append(
                {
                    "game_id": x.game_id,
                    "n": len(sub),
                    "v_read_extra": side * float((wt * (sub.lp_desprac - sub.rp)).sum()),
                    "v_read_extra_des": side * float((wt * (sub.lp_des - sub.rp)).sum()),
                    "v_ladder": side * float((wt * (1 - sub.lp_desprac)).sum()),
                    "v_ladder_des": side * float((wt * (1 - sub.lp_des)).sum()),
                }
            )
    a = (
        pd.DataFrame(rows)
        .groupby("game_id")
        .agg(
            n=("n", "sum"),
            v_read_extra=("v_read_extra", "sum"),
            v_read_extra_des=("v_read_extra_des", "sum"),
            v_ladder=("v_ladder", "sum"),
            v_ladder_des=("v_ladder_des", "sum"),
        )
    )
    g = gf.merge(a, left_on="game_id", right_index=True)
    series = L.book_series()
    series = series[series.nflverse_game_id.str[:4] == str(SEASON)]
    ts = {
        gid: (L.decision_ts(gd), M.deadline_ts(gd, ko))
        for gid, gd, ko in zip(g.game_id, g.gameday, g.kickoff, strict=True)
    }
    sp = consensus(series, ts)
    g["hs0"] = g.game_id.map(lambda i: sp.get(i, [np.nan, np.nan])[0])
    g["hs1"] = g.game_id.map(lambda i: sp.get(i, [np.nan, np.nan])[1])
    g["move"] = g.hs1 - g.hs0
    return g.dropna(subset=["move"]).reset_index(drop=True)


def ols_fit(X, y):
    X1 = np.column_stack([np.ones(len(y)), X])
    return np.linalg.lstsq(X1, y, rcond=None)[0]


def loading(d, xcol, ccol):
    X = d[[xcol, ccol]].to_numpy(float)
    return ols_fit(X, d.move.to_numpy(float))[1]


def boot_perm(d, xcol, ccol, rng):
    est = loading(d, xcol, ccol)
    n = len(d)
    bs = []
    for _ in range(BOOT):
        b = d.iloc[rng.integers(0, n, n)]
        if b[xcol].std() == 0:
            continue
        bs.append(loading(b, xcol, ccol))
    bs = np.array(bs)
    pm = []
    for _ in range(BOOT):
        p = d.copy()
        p[xcol] = rng.permutation(p[xcol].to_numpy())
        pm.append(loading(p, xcol, ccol))
    pm = np.array(pm)
    return {
        "slope": float(est),
        "ci": [float(np.quantile(bs, 0.025)), float(np.quantile(bs, 0.975))],
        "p_positive": float((bs > 0).mean()),
        "perm_p_two_sided": float((np.abs(pm) >= abs(est)).mean()),
        "perm_p_one_sided_pos": float((pm >= est).mean()),
    }


def residual_look(d, xcol, ccol, rng):
    X = np.column_stack([d.move.to_numpy(float), d[ccol].to_numpy(float)])
    b = ols_fit(X, d[xcol].to_numpy(float))
    r = d[xcol].to_numpy(float) - np.column_stack([np.ones(len(d)), X]) @ b
    cov = (d.result - d.hs1).to_numpy(float)
    side = np.sign(r)
    live = (cov != 0) & (side != 0)
    out = {}
    for name, mask in (
        ("all", live),
        ("top_half_abs", live & (np.abs(r) >= np.quantile(np.abs(r), DECISIVE_FRACTION))),
    ):
        w = int(((np.sign(cov) == side) & mask).sum())
        n = int(mask.sum())
        mm = (side * cov)[mask]
        bs = np.array([rng.choice(mm, len(mm)).mean() for _ in range(BOOT)])
        sgn = rng.choice([-1.0, 1.0], (BOOT, len(mm)))
        pn = (sgn * np.abs(mm)).mean(axis=1)
        out[name] = {
            "wins": w,
            "losses": n - w,
            "mean_cover_pts": float(mm.mean()),
            "ci": [float(np.quantile(bs, 0.025)), float(np.quantile(bs, 0.975))],
            "p_positive": float((bs > 0).mean()),
            "sign_null_p_one_sided": float((pn >= mm.mean()).mean()),
        }
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    g = pd.read_parquet(ROOT / "data/processed/game_features.parquet")
    g = g[(g.game_type == "REG") & g.spread_line.notna() & g.result.notna()].copy()
    g["slot"] = [O.slot(s, w) for s, w in zip(g.season, g.week, strict=True)]
    w, nw = position_weights(g, group_counts())
    s = sample_rows()
    d = games_frame(s, w)
    rep = {
        "weights_spread_pts_per_out": w,
        "weight_games": nw,
        "sample_players": len(s),
        "games": len(d),
        "games_with_result": int(d.result.notna().sum()),
        "move_sd": float(d.move.std()),
        "x_sd": float(d.v_read_extra.std()),
    }
    rep["loading_vs_desprac"] = boot_perm(d, "v_read_extra", "v_ladder", rng)
    rep["loading_vs_des"] = boot_perm(d, "v_read_extra_des", "v_ladder_des", rng)
    rep["ladder_loading"] = boot_perm(d.assign(x=d.v_ladder, c=d.v_read_extra), "x", "c", rng)
    rep["residual_cover"] = residual_look(d, "v_read_extra", "v_ladder", rng)
    (OUT / "report.json").write_text(json.dumps(rep, indent=1))
    d.to_parquet(OUT / "games.parquet")
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
