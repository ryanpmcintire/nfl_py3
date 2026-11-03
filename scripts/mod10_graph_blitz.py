import json
import sys

import mod10_graph_term as g
import mod10_graph_term_warm as w
import mod24_ftn_turnover_luck as m
import numpy as np
import pandas as pd

COLUMN = "ftn_graph_blitz_diff"
DEF_MIN_HALF = 1
QB_TOTAL_HALF_MIN = 100
QB_BLITZ_HALF_MIN = 30
DRAWS = 2000
SEED = 20261103


def load():
    ftn = pd.concat(
        [
            pd.read_parquet(m.FTN_HIST / f"ftn_charting_{s}.parquet")
            for s in (2022, 2023, 2024, 2025)
        ]
        + [pd.read_parquet(m.FTN_2026)],
        ignore_index=True,
    ).rename(columns={"nflverse_game_id": "game_id", "nflverse_play_id": "play_id"})
    cols = [
        "game_id",
        "play_id",
        "season",
        "season_type",
        "week",
        "posteam",
        "defteam",
        "qb_dropback",
        "epa",
        "passer_player_id",
    ]
    pbp = pd.concat(
        [pd.read_parquet(m.PBP_ROOT / f"season={s}", columns=cols) for s in range(2022, 2027)],
        ignore_index=True,
    )
    pbp = pbp[
        (pbp.season_type == "REG")
        & (pbp.qb_dropback == 1)
        & pbp.passer_player_id.notna()
        & pbp.epa.notna()
    ]
    pbp = pbp.assign(play_id=pbp.play_id.astype(float))
    ftn = ftn.assign(play_id=ftn.play_id.astype(float))
    t = pbp.merge(
        ftn[["game_id", "play_id", "n_blitzers", "is_catchable_ball"]],
        on=["game_id", "play_id"],
        how="inner",
    ).dropna(subset=["n_blitzers"])
    t["blitz"] = (t.n_blitzers >= 1).astype(float)
    t["cb"] = t.is_catchable_ball.fillna(False).astype(float)
    return t


def sb(a, b):
    rho = np.corrcoef(a, b)[0, 1]
    return 2 * rho / (1 + rho), rho


def boot_sb(a, b, rng):
    n = len(a)
    out = []
    for _ in range(DRAWS):
        i = rng.integers(0, n, n)
        out.append(sb(a[i], b[i])[0])
    d = np.array(out)
    return [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))], float((d > 0.436).mean())


def reliability(t):
    t = t[t.season <= 2025].assign(half=lambda d: np.where(d.week % 2 == 1, "odd", "even"))
    rng = np.random.default_rng(SEED)
    d = (
        t.groupby(["season", "defteam", "half"])
        .blitz.agg(["mean", "size"])
        .unstack("half")
        .dropna()
    )
    a, b = d[("mean", "odd")].to_numpy(), d[("mean", "even")].to_numpy()
    r_def, rho = sb(a, b)
    ci_def, p_gate_def = boot_sb(a, b, rng)
    n_def = float((d[("size", "odd")] + d[("size", "even")]).median())
    bl = t.groupby(["season", "passer_player_id", "half", "blitz"]).epa.agg(["mean", "size"])
    bl = bl.unstack(["half", "blitz"])
    tot = {
        h: bl[("size", h, 0.0)].fillna(0) + bl[("size", h, 1.0)].fillna(0) for h in ("odd", "even")
    }
    ok = (
        (tot["odd"] >= QB_TOTAL_HALF_MIN)
        & (tot["even"] >= QB_TOTAL_HALF_MIN)
        & (bl[("size", "odd", 1.0)] >= QB_BLITZ_HALF_MIN)
        & (bl[("size", "even", 1.0)] >= QB_BLITZ_HALF_MIN)
        & (bl[("size", "odd", 0.0)] >= QB_BLITZ_HALF_MIN)
        & (bl[("size", "even", 0.0)] >= QB_BLITZ_HALF_MIN)
    )
    q = bl[ok]
    go = (q[("mean", "odd", 1.0)] - q[("mean", "odd", 0.0)]).to_numpy()
    ge = (q[("mean", "even", 1.0)] - q[("mean", "even", 0.0)]).to_numpy()
    r_qb, rho_q = sb(go, ge)
    ci_qb, p_gate_qb = boot_sb(go, ge, rng)
    n_qb = float((tot["odd"][ok] + tot["even"][ok]).median())
    qh = (
        t.groupby(["season", "passer_player_id", "half"])
        .cb.agg(["mean", "size"])
        .unstack("half")
        .dropna()
    )
    qh = qh[
        (qh[("size", "odd")] >= QB_TOTAL_HALF_MIN) & (qh[("size", "even")] >= QB_TOTAL_HALF_MIN)
    ]
    ca, cb_ = qh[("mean", "odd")].to_numpy(), qh[("mean", "even")].to_numpy()
    r_cb, rho_cb = sb(ca, cb_)
    ci_cb, p_cb = boot_sb(ca, cb_, rng)
    return {
        "qb_catchable": {
            "sb": float(r_cb),
            "rho": float(rho_cb),
            "ci": ci_cb,
            "p_above_gate_436": p_cb,
            "n_qb_seasons": len(ca),
            "n_full": float((qh[("size", "odd")] + qh[("size", "even")]).median()),
        },
        "def_blitz_rate": {
            "sb": float(r_def),
            "rho": float(rho),
            "ci": ci_def,
            "p_above_gate_436": p_gate_def,
            "n_team_seasons": len(a),
            "n_full": n_def,
        },
        "qb_blitz_gap": {
            "sb": float(r_qb),
            "rho": float(rho_q),
            "ci": ci_qb,
            "p_above_gate_436": p_gate_qb,
            "n_qb_seasons": len(go),
            "n_full": n_qb,
        },
    }


def kfun(rel):
    r = rel["sb"]
    return rel["n_full"] * (1 - r) / r if r > 0 else np.inf


def shr(x, n, prior, k):
    if not np.isfinite(k):
        return prior
    return (x + k * prior) / (n + k)


def team_values(t, games, k_def, k_qb):
    dates = dict(zip(games.game_id, pd.to_datetime(games.gameday), strict=True))
    t = t.assign(gameday=t.game_id.map(dates)).dropna(subset=["gameday"])
    t = t.assign(nb=t.blitz, nn=1 - t.blitz, cs=t.cb, cn=1.0)
    sched = games.assign(gameday=pd.to_datetime(games.gameday))
    sched = sched[sched.season >= w.WARM]
    rows = []
    for day in sorted(sched.gameday.unique()):
        prior = t[t.gameday < day]
        if prior.empty:
            continue
        lg = prior[["cs", "cn", "nb", "nn"]].sum()
        l_rate = lg.nb / (lg.nb + lg.nn)
        l_cb = lg.cs / lg.cn
        dd = prior.groupby("defteam")[["nb", "nn"]].sum()
        drate = shr(dd.nb, dd.nb + dd.nn, l_rate, k_def) - l_rate
        qq = prior.groupby("passer_player_id")[["cs", "cn"]].sum()
        gap = shr(qq.cs, qq.cn, l_cb, k_qb) - l_cb
        today = sched[sched.gameday == day]
        for gid, h, a, hq, aq in zip(
            today.game_id,
            today.home_team,
            today.away_team,
            today.home_projected_qb_id,
            today.away_projected_qb_id,
            strict=True,
        ):
            rows.append((gid, h, gap.get(hq, 0.0) * drate.get(a, 0.0)))
            rows.append((gid, a, gap.get(aq, 0.0) * drate.get(h, 0.0)))
    return pd.DataFrame(rows, columns=["game_id", "team", "val"])


def attach(features, tv):
    home = tv.rename(columns={"team": "home_team", "val": "hv"})
    away = tv.rename(columns={"team": "away_team", "val": "av"})
    out = features.merge(home, on=["game_id", "home_team"], how="left")
    out = out.merge(away, on=["game_id", "away_team"], how="left")
    out[COLUMN] = (out.hv.fillna(0.0) - out.av.fillna(0.0)).where(out.season.gt(w.WARM), 0.0)
    return out.drop(columns=["hv", "av"])


def stage_run():
    g.OUT_DIR.mkdir(parents=True, exist_ok=True)
    served = pd.read_parquet(m.SERVED_FEATURES)
    t = load()
    rel = reliability(t)
    print(json.dumps(rel, indent=1), flush=True)
    k_def, k_qb = kfun(rel["def_blitz_rate"]), kfun(rel["qb_catchable"])
    tv = team_values(t, served, k_def, k_qb)
    table = attach(served, tv)
    cols = (*m.BASE_COLUMNS, COLUMN)
    base = m.decisive(m.run_eval(table, m.BASE_COLUMNS, "base"))
    arm = m.decisive(m.run_eval(table, cols, "blitz"))
    a, b = arm[arm.season.isin(w.SCORE)], base[base.season.isin(w.SCORE)]
    res = {
        "reliability": rel,
        "k_def": None if not np.isfinite(k_def) else k_def,
        "k_qb": None if not np.isfinite(k_qb) else k_qb,
        "base_record": m.record(b),
        "compare": m.compare("blitz matchup v base 2023-25", a, b),
        "nonzero_rows_scored": int((table[table.season.isin(w.SCORE)][COLUMN] != 0).sum()),
        "market_ll_half": float(np.log(2)),
        "mean_ll_base": float(b.ll.mean()),
        "mean_ll_arm": float(a.ll.mean()),
        "mean_brier_base": float(b.brier.mean()),
        "mean_brier_arm": float(a.brier.mean()),
        "fold_coefficients": {},
    }
    for season in w.SCORE:
        model, _ = g.fit_frame(table, cols, season)
        names = list(model[:-1].get_feature_names_out())
        coef = np.ravel(model[-1].coef_)
        res["fold_coefficients"][season] = {
            n: float(c) for n, c in zip(names, coef, strict=False) if "graph" in n
        }
    res["coefficient_bootstrap"] = w.coef_draws(table, cols)
    (g.OUT_DIR / "report_blitz.json").write_text(json.dumps(res, indent=2, default=str))
    print(json.dumps(res, indent=1, default=str), flush=True)


if __name__ == "__main__":
    if sys.argv[1:] == ["run"]:
        stage_run()
    else:
        raise SystemExit("usage: mod10_graph_blitz.py run")
