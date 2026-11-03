import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for sub in ("scripts", "src"):
    if str(ROOT / sub) not in sys.path:
        sys.path.insert(0, str(ROOT / sub))

OUTD = ROOT / "artifacts" / "mod07_joint_stack"
SERVED = ROOT / "artifacts" / "pick_probability" / "20261103T195411Z" / "per_game.parquet"
FROZEN = ROOT / "artifacts" / "forward_terms"
SEASONS = (2020, 2021, 2022, 2023, 2024, 2025)
TERMS = (
    "sky03",
    "sky03_separated",
    "sky04",
    "env02",
    "sky01",
    "sky01snap",
    "surface",
    "mod35_cold_exposure",
    "mod41_avgwin_served",
)
PENALTIES = (1.0, 10.0, 100.0, 1000.0, 10000.0)
SEED = 20261103
DRAWS = 2000
CLIP = 1e-6
NEWTON_STEPS = 50
NEWTON_TOL = 1e-9


def logit(p):
    p = np.clip(np.asarray(p, dtype=float), CLIP, 1 - CLIP)
    return np.log(p / (1 - p))


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def served_games():
    g = pd.read_parquet(SERVED)
    g = g[(g.game_type == "REG") & g.season.isin(SEASONS)]
    g = g[g.margin_vs_open != 0].copy()
    g["served"] = g.out_of_season_home_probability.clip(CLIP, 1 - CLIP)
    g["y"] = g.home_covered.astype(float)
    return g.reset_index(drop=True)


def sky03_x(g):
    import sky03_def_traits as d
    import sky03_matchups as m

    t, _ = d.load()
    rr = d.build(t)
    rows = []
    for s in SEASONS:
        for w in sorted(rr[rr.season == s].week.unique()):
            hist = rr[rr.key < s * 100 + w]
            cur = rr[(rr.season == s) & (rr.week == w)]
            c = m.fit_c(hist, "zBc", "rB", "n_ns")
            rows.append(
                cur.assign(pts=cur.db_exp * c * cur.zBc * cur.ns_share)[["game_id", "off", "pts"]]
            )
    gx = pd.concat(rows).merge(g[["game_id", "home_team"]], on="game_id")
    gx["sp"] = np.where(gx.off == gx.home_team, 1.0, -1.0) * gx.pts
    return g.game_id.map(gx.groupby("game_id").sp.sum())


def sky03_separated_x(g):
    import sky03_matchup as sm
    import sky03_matchups as m

    t = m.load_games().merge(sm.load_script(), on=["game_id", "off"], how="left")
    t["sc"] = t.sc.fillna(0.5)
    rr = sm.build_rows(t, tuple(range(m.TRAIT_START + 1, max(SEASONS) + 1)))
    gx = m.game_x(rr, SEASONS).merge(g[["game_id", "home_team"]], on="game_id")
    gx["sp"] = np.where(gx.off == gx.home_team, 1.0, -1.0) * gx.pts
    return g.game_id.map(gx.groupby("game_id").sp.sum())


def sky04_x(g):
    import sky04_lead_lag as lead

    fr = json.loads((FROZEN / "sky04" / "frozen_coefficients.json").read_text())
    books = fr["books"]
    weights = np.array([fr["weights"][b] for b in books])
    s = lead.book_series()
    s["upd"] = s.upd.fillna(s.observed_at_utc)
    lead.BOOKS_REF = books
    mats = lead.game_mats(s, books)
    sub = g[g.season >= 2023]
    out = pd.Series(np.nan, index=g.index)
    for i, gid, gd in zip(sub.index, sub.game_id, sub.gameday, strict=True):
        if gid not in mats or pd.isna(gd):
            continue
        o, mat, _ = mats[gid]
        k = np.searchsorted(o, lead.decision_ts(gd), side="right") - 1
        if k < 0:
            continue
        v = mat[k]
        if (~np.isnan(v)).sum() < 3:
            continue
        ww = np.where(np.isnan(v), 0.0, weights)
        out[i] = float((np.nan_to_num(v) * ww).sum() / ww.sum() - np.nanmedian(v))
    return out


def env02_x(g):
    import env02_roof_fitted_term as e
    import roof_state_screen as rs

    flags = e.roof_open_probability(e.AWARE / "forecasts.parquet", e.AWARE / "manifest.json")
    p = g.game_id.map(flags.set_index("game_id")[e.NEW_TERM])
    retract = g.home_team.isin(rs.RETRACT_TEAMS)
    return p.where(retract)


def sky01_x(g, weighted):
    import sky01_drift_weighted as dw
    import sky01_news_availability as base

    sched = sorted(glob.glob(str(ROOT / "data/raw/*/schedules.parquet")))[-1]
    base.SCHED = Path(sched)
    s = base.schedule()
    p = base.beat_rows(base.team_games(s))
    q = g[["game_id", "home_team", "away_team"]].copy()
    q["home_team"] = q.home_team.map(base.alias)
    q["away_team"] = q.away_team.map(base.alias)
    if not weighted:
        return pd.Series(np.asarray(base.feature(p, q), dtype=float), index=g.index)
    snap_dir = Path(
        sorted(glob.glob(str(ROOT / "data/players/raw/*/snap_counts.parquet")))[-1]
    ).parent
    dw.SNAPS = snap_dir / "snap_counts.parquet"
    dw.ROSTERS = snap_dir / "weekly_rosters.parquet"
    ps, sm = dw.prior_share()
    v, _, _ = dw.weighted_feature(p, q, ps, sm)
    return pd.Series(np.asarray(v, dtype=float), index=g.index)


def surface_x(g):
    import surface_switch_fitted_term as sw

    from nfl_ats.snapshots import latest_snapshot, load_snapshot

    schedules, _ = load_snapshot(latest_snapshot(ROOT / "data" / "raw"))
    flags = sw.point_in_time_surface_switch_flags(schedules)
    return g.game_id.map(flags.set_index("game_id")[sw.NEW_TERM]).fillna(0.0)


def mod35_x(g):
    import mod35_apparent_temp as m35

    fc = pd.read_parquet(m35.FC)
    fc = fc.loc[fc.fetch_status == "ok"].copy()
    fc["home"] = fc.game_id.str.split("_").str[3]
    fc["fc_at"] = m35.apparent(fc.forecast_temp_f.to_numpy(), fc.forecast_wind_mph.to_numpy())
    rec = fc.loc[
        fc.roof.isin(m35.OUTDOOR) & fc.actual_temp_f.notna() & fc.actual_wind_mph.notna()
    ].copy()
    rec["app"] = m35.apparent(rec.actual_temp_f.to_numpy(), rec.actual_wind_mph.to_numpy())
    rec = rec[["home", "season", "week", "app"]]
    chosen = m35.select_half_widths(rec)
    b = g[["game_id", "season", "week", "away_team"]].merge(
        fc[["game_id", "roof", "fc_at"]], on="game_id", how="left"
    )
    nrm = np.full(len(b), np.nan)
    for s in SEASONS:
        m = (b.season == s).to_numpy()
        keys = list(zip(b.loc[m, "away_team"], b.loc[m, "season"], b.loc[m, "week"], strict=True))
        nrm[m] = m35.normals(rec, keys, chosen[s][0])
    x = np.maximum(b.fc_at.to_numpy() - nrm, 0.0)
    x = np.where(b.roof.isin(m35.OUTDOOR).to_numpy() & ~np.isnan(nrm), x, np.nan)
    return pd.Series(x, index=g.index)


def avgwin_x(g):
    import mod41_freeze_residual_load as f41

    fr = json.loads((FROZEN / "mod41_avgwin_served" / "frozen_coefficients.json").read_text())
    pbp = f41.latest_pbp(ROOT / "data" / "pbp" / "raw")
    out = pd.Series(np.nan, index=g.index)
    for season in SEASONS:
        t = f41.season_team_games(pbp, season)
        grp = t.groupby("team")
        state = pd.DataFrame({"team": t.team, "week": t.week, "n_played": grp.cumcount() + 1})
        state["own_td"] = grp["run"].transform(lambda s: s.expanding().mean())
        norm = grp["total"].transform(lambda s: s.expanding().mean())
        for w in fr["windows"]:
            state[f"load{w}"] = grp["total"].transform(lambda s, w=w: s.rolling(w).mean()) - norm
            state[f"opp{w}"] = grp["defended"].transform(lambda s, w=w: s.rolling(w).mean())
        for r in g[g.season == season].itertuples():
            prior = state[state.week < r.week]
            last = prior.sort_values("week").groupby("team").tail(1).set_index("team")
            if r.home_team not in last.index or r.away_team not in last.index:
                continue
            h, a = last.loc[r.home_team], last.loc[r.away_team]
            if min(h.n_played, a.n_played) < fr["min_prior_games"]:
                continue
            s_own = float(h.own_td - a.own_td)
            z = []
            for w in fr["windows"]:
                b0, b_own, b_opp = fr["decomposition_by_window"][str(w)]
                s_total = float(h[f"load{w}"] - a[f"load{w}"])
                s_opp = float(h[f"opp{w}"] - a[f"opp{w}"])
                z.append(
                    (s_total - (b0 + b_own * s_own + b_opp * s_opp)) / fr["sd_by_window"][str(w)]
                )
            out[r.Index] = float(np.mean(z))
    return out


def build_features():
    g = served_games()
    builders = {
        "sky03": lambda: sky03_x(g),
        "sky03_separated": lambda: sky03_separated_x(g),
        "sky04": lambda: sky04_x(g),
        "env02": lambda: env02_x(g),
        "sky01": lambda: sky01_x(g, False),
        "sky01snap": lambda: sky01_x(g, True),
        "surface": lambda: surface_x(g),
        "mod35_cold_exposure": lambda: mod35_x(g),
        "mod41_avgwin_served": lambda: avgwin_x(g),
    }
    cover = {}
    failed = {}
    for name, fn in builders.items():
        try:
            g[name] = np.asarray(fn(), dtype=float)
            cover[name] = int(g[name].notna().sum())
        except Exception as exc:
            failed[name] = f"{type(exc).__name__}: {exc}"
            print("FAILED", name, failed[name], flush=True)
    OUTD.mkdir(parents=True, exist_ok=True)
    keep = ["game_id", "season", "week", "served", "y", "result", "margin_vs_open", *cover]
    g[keep].to_parquet(OUTD / "features.parquet")
    (OUTD / "feature_coverage.json").write_text(
        json.dumps({"coverage": cover, "failed": failed, "games": len(g)}, indent=1)
    )
    print(json.dumps({"coverage": cover, "failed": failed, "games": len(g)}, indent=1))


def fit_ridge(offset, X, y, lam):
    n, k = X.shape
    A = np.column_stack([np.ones(n), X])
    pen = np.diag([0.0] + [lam] * k)
    beta = np.zeros(k + 1)
    for _ in range(NEWTON_STEPS):
        p = sigmoid(offset + A @ beta)
        grad = A.T @ (p - y) + pen @ beta
        hess = (A * (p * (1 - p))[:, None]).T @ A + pen
        step = np.linalg.solve(hess, grad)
        beta -= step
        if np.abs(step).max() < NEWTON_TOL:
            break
    return beta


def standardise(train, test):
    mu = np.nanmean(train, axis=0)
    sd = np.nanstd(train, axis=0)
    sd = np.where(sd > 0, sd, 1.0)
    f = lambda a: np.nan_to_num((a - mu) / sd, nan=0.0)  # noqa: E731
    return f(train), f(test), sd


def ll(p, y):
    p = np.clip(p, CLIP, 1 - CLIP)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def choose_penalty(off, X, y, seasons, cols):
    best, best_ll = None, np.inf
    table = {}
    for lam in PENALTIES:
        loss = 0.0
        for s in np.unique(seasons):
            te = seasons == s
            tr, tt, _ = standardise(X[~te][:, cols], X[te][:, cols])
            beta = fit_ridge(off[~te], tr, y[~te], lam)
            loss += ll(sigmoid(off[te] + beta[0] + tt @ beta[1:]), y[te]).sum()
        table[lam] = float(loss / len(y))
        if loss < best_ll:
            best, best_ll = lam, loss
    return best, table


def loso(off, X, y, seasons, cols):
    pred = np.zeros(len(y))
    betas, chosen = {}, {}
    for s in np.unique(seasons):
        te = seasons == s
        if cols:
            lam, _ = choose_penalty(off[~te], X[~te], y[~te], seasons[~te], cols)
        else:
            lam = PENALTIES[0]
        tr, tt, _ = standardise(X[~te][:, cols], X[te][:, cols])
        beta = fit_ridge(off[~te], tr, y[~te], lam)
        pred[te] = sigmoid(off[te] + beta[0] + tt @ beta[1:])
        betas[int(s)] = [float(b) for b in beta]
        chosen[int(s)] = lam
    return pred, betas, chosen


def block_boot(delta, blocks, rng):
    keys = pd.factorize(blocks)[0]
    sums = np.bincount(keys, weights=delta)
    cnt = np.bincount(keys)
    idx = rng.integers(0, len(sums), size=(DRAWS, len(sums)))
    means = sums[idx].sum(axis=1) / cnt[idx].sum(axis=1)
    return {
        "mean": float(delta.mean()),
        "low": float(np.quantile(means, 0.025)),
        "high": float(np.quantile(means, 0.975)),
        "probability_positive": float((means > 0).mean()),
    }


def record_vs(p_new, p_base, y):
    new_home = p_new > 0.5
    base_home = p_base > 0.5
    split = new_home != base_home
    win_new = int(((new_home == (y == 1)) & split).sum())
    win_base = int(((base_home == (y == 1)) & split).sum())
    return {
        "decisive_games": int(split.sum()),
        "new_wins": win_new,
        "served_wins": win_base,
        "new_correct": int((new_home == (y == 1)).sum()),
        "served_correct": int((base_home == (y == 1)).sum()),
        "games": len(y),
    }


def stack():
    d = pd.read_parquet(OUTD / "features.parquet")
    terms = [t for t in TERMS if t in d.columns]
    y = d.y.to_numpy()
    off = logit(d.served.to_numpy())
    seasons = d.season.to_numpy()
    X = d[terms].to_numpy(dtype=float)
    blocks = (d.season * 100 + d.week).to_numpy()
    rng = np.random.default_rng(SEED)
    base_ll = ll(d.served.to_numpy(), y)
    out = {"terms": terms, "games": len(d), "coverage": {t: int(d[t].notna().sum()) for t in terms}}
    cells = {}

    p0, b0, _ = loso(off, X, y, seasons, [])
    cells["intercept_only_control"] = {
        **block_boot(base_ll - ll(p0, y), blocks, rng),
        "intercept_by_fold": {k: v[0] for k, v in b0.items()},
    }

    allcols = list(range(len(terms)))
    pj, bj, cj = loso(off, X, y, seasons, allcols)
    joint_delta = base_ll - ll(pj, y)
    cells["joint"] = {
        **block_boot(joint_delta, blocks, rng),
        "log_loss_served": float(base_ll.mean()),
        "log_loss_joint": float(ll(pj, y).mean()),
        "record": record_vs(pj, d.served.to_numpy(), y),
        "per_season_gain": {
            int(s): float(joint_delta[seasons == s].mean()) for s in np.unique(seasons)
        },
        "penalty_by_fold": cj,
        "vs_intercept_control": block_boot(ll(p0, y) - ll(pj, y), blocks, rng),
    }
    coef = {t: [bj[s][i + 1] for s in sorted(bj)] for i, t in enumerate(terms)}
    out["joint_coefficients_per_fold_standardised"] = {
        s: dict(zip(["intercept", *terms], bj[s], strict=True)) for s in sorted(bj)
    }
    out["sign_stability"] = {
        t: {
            "positive_folds": int(sum(c > 0 for c in v)),
            "negative_folds": int(sum(c < 0 for c in v)),
            "mean": float(np.mean(v)),
            "sd": float(np.std(v)),
        }
        for t, v in coef.items()
    }

    lam_all, lam_table = choose_penalty(off, X, y, seasons, allcols)
    tr, _, _ = standardise(X, X)
    beta_all = fit_ridge(off, tr, y, lam_all)
    p_in = sigmoid(off + beta_all[0] + tr @ beta_all[1:])
    in_delta = base_ll - ll(p_in, y)
    cells["joint_in_sample"] = {
        "penalty_chosen_all_seasons": lam_all,
        "loso_log_loss_by_penalty": lam_table,
        "mean": float(in_delta.mean()),
        "oos_minus_in_sample_gap": float(joint_delta.mean() - in_delta.mean()),
        "coefficients_all_seasons_standardised": dict(
            zip(["intercept", *terms], map(float, beta_all), strict=True)
        ),
    }

    singles = {}
    for i, t in enumerate(terms):
        pt, bt, ct = loso(off, X, y, seasons, [i])
        delta = base_ll - ll(pt, y)
        singles[t] = {
            **block_boot(delta, blocks, rng),
            "record": record_vs(pt, d.served.to_numpy(), y),
            "coefficient_signs_by_fold": {k: float(np.sign(v[1])) for k, v in bt.items()},
            "penalty_by_fold": ct,
        }
    cells["individual"] = singles
    sum_single = float(sum(v["mean"] for v in singles.values()))
    cells["additivity"] = {
        "sum_of_individual_gains": sum_single,
        "joint_gain": cells["joint"]["mean"],
        "joint_minus_sum": cells["joint"]["mean"] - sum_single,
    }
    out["cells"] = cells
    out["looks"] = {
        "gain_cells": 2 + len(terms),
        "outer_folds": len(np.unique(seasons)),
        "penalty_grid": list(PENALTIES),
        "family": "mod07_joint_stack",
    }
    (OUTD / "stack_result.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["features", "stack"], required=True)
    a = ap.parse_args()
    if a.stage == "features":
        build_features()
    else:
        stack()
    return 0


if __name__ == "__main__":
    sys.exit(main())
