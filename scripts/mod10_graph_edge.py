import json
import sys

import mod10_graph_term as g
import mod10_graph_term_warm as w
import mod24_ftn_turnover_luck as m
import numpy as np
import pandas as pd

QB_TRAIT = "qb_catchable_per_attempt"
QB_HALF_MIN = 100


def load_pairs(games):
    ftn = pd.concat(
        [
            pd.read_parquet(m.FTN_HIST / f"ftn_charting_{s}.parquet")
            for s in (2022, 2023, 2024, 2025)
        ]
        + [pd.read_parquet(m.FTN_2026)],
        ignore_index=True,
    ).rename(columns={"nflverse_game_id": "game_id", "nflverse_play_id": "play_id"})
    cols = [*g.PBP_COLUMNS, "passer_player_id"]
    pbp = pd.concat(
        [pd.read_parquet(m.PBP_ROOT / f"season={s}", columns=cols) for s in range(2022, 2027)],
        ignore_index=True,
    )
    pbp = pbp[
        (pbp.season_type == "REG")
        & pbp.receiver_player_id.notna()
        & pbp.passer_player_id.notna()
        & (pbp.pass_attempt == 1)
    ]
    pbp = pbp.assign(play_id=pbp.play_id.astype(float))
    ftn = ftn.assign(play_id=ftn.play_id.astype(float))
    t = pbp.merge(ftn.drop(columns=["season", "week"]), on=["game_id", "play_id"], how="inner")
    for c in ("is_catchable_ball", "is_contested_ball", "is_created_reception"):
        t[c] = t[c].fillna(False).astype(bool)
    t["tg"] = 1.0
    t["c"] = t.is_catchable_ball.astype(float)
    t["cr"] = (t.is_created_reception & t.is_catchable_ball).astype(float)
    t["ct"] = (t.is_contested_ball & t.is_catchable_ball).astype(float)
    agg = t.groupby(
        ["game_id", "posteam", "passer_player_id", "receiver_player_id", "week", "season"],
        as_index=False,
    )[["tg", "c", "cr", "ct"]].sum()
    dates = dict(zip(games.game_id, pd.to_datetime(games.gameday), strict=True))
    agg["gameday"] = agg.game_id.map(dates)
    return agg.dropna(subset=["gameday"]).rename(
        columns={"receiver_player_id": "pid", "passer_player_id": "qb"}
    )


def qb_k(agg):
    results = json.loads(g.TRAITS.read_text())["results"]
    r = next(x["sb_pooled"]["point"] for x in results if x["trait"] == QB_TRAIT)
    h = agg[agg.season <= 2025].assign(half=lambda d: np.where(d.week % 2 == 1, "odd", "even"))
    q = h.groupby(["season", "qb", "half"]).tg.sum().unstack("half").dropna()
    q = q[(q.odd >= QB_HALF_MIN) & (q.even >= QB_HALF_MIN)]
    n_full = float((q.odd + q.even).median())
    return n_full * (1 - r) / r, n_full, r


def shrunk(x, n, prior, k):
    return (x + k * prior) / (n + k)


def qb_values(prior, k, kq):
    cols = ["tg", "c", "cr", "ct"]
    pair = prior.groupby(["qb", "pid"])[cols].sum().reset_index()
    qbt = prior.groupby("qb")[cols].sum()
    rec = prior.groupby("pid")[cols].sum()
    lg = prior[cols].sum()
    l_c, l_cr, l_ct = lg.c / lg.tg, lg.cr / lg.c, lg.ct / lg.c
    qb_c = shrunk(qbt.c, qbt.tg, l_c, kq)
    rec_cr = shrunk(rec.cr, rec.c, l_cr, k["created"])
    rec_ct = shrunk(rec.ct, rec.c, l_ct, k["contested"])
    pair["rc"] = pair.pid.map(rec.c)
    pc = shrunk(pair.c, pair.tg, pair.qb.map(qb_c), kq)
    pcr = shrunk(pair.cr, pair.c, pair.pid.map(rec_cr), k["created"])
    pct = shrunk(pair.ct, pair.c, pair.pid.map(rec_ct), k["contested"])
    edge = (pc - l_c) + (pcr - l_cr) + (pct - l_ct)
    pair["edge"] = edge.where(pair.rc >= g.HALF_MIN, 0.0)
    pair["share"] = pair.tg / pair.qb.map(qbt.tg)
    pair["val"] = pair.share * pair.edge
    return pair.groupby("qb").val.sum()


def team_values(agg, games, k, kq):
    rows = []
    sched = games.assign(gameday=pd.to_datetime(games.gameday))
    sched = sched[sched.season >= w.WARM]
    for day in sorted(sched.gameday.unique()):
        prior = agg[agg.gameday < day]
        if prior.empty:
            continue
        qv = qb_values(prior, k, kq)
        today = sched[sched.gameday == day]
        for gid, h, a, hq, aq in zip(
            today.game_id,
            today.home_team,
            today.away_team,
            today.home_projected_qb_id,
            today.away_projected_qb_id,
            strict=True,
        ):
            rows.append((gid, h, qv.get(hq, np.nan)))
            rows.append((gid, a, qv.get(aq, np.nan)))
    return pd.DataFrame(rows, columns=["game_id", "team", "val"])


def qb_change_ids(games):
    d = games.assign(gameday=pd.to_datetime(games.gameday))
    names = ["game_id", "gameday", "team", "qb"]
    long = pd.concat(
        [
            d[["game_id", "gameday", "home_team", "home_projected_qb_id"]].set_axis(names, axis=1),
            d[["game_id", "gameday", "away_team", "away_projected_qb_id"]].set_axis(names, axis=1),
        ]
    ).sort_values(["team", "gameday"])
    long["prev"] = long.groupby("team").qb.shift()
    changed = long[long.prev.notna() & (long.prev != long.qb)]
    return set(changed.game_id)


def stage_run():
    g.OUT_DIR.mkdir(parents=True, exist_ok=True)
    served = pd.read_parquet(m.SERVED_FEATURES)
    agg = load_pairs(served)
    k, _, reliability = g.shrink_k(agg)
    kq, n_full_q, r_q = qb_k(agg)
    tv = team_values(agg, served, k, kq)
    table = g.attach(served, tv)
    table[g.COLUMN] = table[g.COLUMN].where(table.season.gt(w.WARM), 0.0)
    cols = (*m.BASE_COLUMNS, g.COLUMN)
    base = m.decisive(m.run_eval(table, m.BASE_COLUMNS, "base"))
    arm = m.decisive(m.run_eval(table, cols, "graph_edge"))
    a, b = arm[arm.season.isin(w.SCORE)], base[base.season.isin(w.SCORE)]
    changed = qb_change_ids(served)
    ca, cb = a[a.index.isin(changed)], b[b.index.isin(changed)]
    res = {
        "k_receiver": k,
        "k_qb": kq,
        "n_full_qb": n_full_q,
        "r_qb": r_q,
        "reliability": reliability,
        "base_record": m.record(b),
        "all_games": m.compare("edge v base 2023-25", a, b),
        "qb_change_games": m.compare("edge v base, QB change", ca, cb),
        "nonzero_term_rows_scored": int((table[table.season.isin(w.SCORE)][g.COLUMN] != 0).sum()),
    }
    for name, f in (("base", b), ("arm", a), ("base_qbc", cb), ("arm_qbc", ca)):
        res[f"mean_ll_{name}"] = float(f.ll.mean())
        res[f"mean_brier_{name}"] = float(f.brier.mean())
    res["fold_coefficients"] = {}
    for season in w.SCORE:
        model, _ = g.fit_frame(table, cols, season)
        names = list(model[:-1].get_feature_names_out())
        coef = np.ravel(model[-1].coef_)
        res["fold_coefficients"][season] = {
            n: float(c) for n, c in zip(names, coef, strict=False) if "graph" in n
        }
    res["coefficient_bootstrap"] = w.coef_draws(table, cols)
    (g.OUT_DIR / "report_edge.json").write_text(json.dumps(res, indent=2, default=str))
    print(json.dumps(res, indent=1, default=str), flush=True)


if __name__ == "__main__":
    if sys.argv[1:] == ["run"]:
        stage_run()
    else:
        raise SystemExit("usage: mod10_graph_edge.py run")
