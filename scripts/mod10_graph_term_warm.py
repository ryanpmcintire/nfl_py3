import json
import sys

import mod10_graph_term as g
import mod24_ftn_turnover_luck as m
import numpy as np
import pandas as pd

OUT_DIR = g.OUT_DIR
SCORE = (2023, 2024, 2025)
WARM = 2022
DRAWS = 500
SEED = 20261103


def team_values(agg, games, k):
    rows = []
    sched = games.assign(gameday=pd.to_datetime(games.gameday))
    sched = sched[sched.season >= WARM]
    for day in sorted(sched.gameday.unique()):
        prior = agg[agg.gameday < day]
        if prior.empty:
            continue
        pid = prior.groupby("pid")[["c", "cr", "ct"]].sum()
        league = prior[["c", "cr", "ct"]].sum()
        safe = pid.c.where(pid.c > 0)
        dev_cr = pid.c * (pid.cr / safe - league.cr / league.c).fillna(0.0) / (pid.c + k["created"])
        dev_ct = (
            pid.c * (pid.ct / safe - league.ct / league.c).fillna(0.0) / (pid.c + k["contested"])
        )
        node = (dev_cr + dev_ct).where(pid.c >= g.HALF_MIN, 0.0)
        tp = prior.groupby(["posteam", "pid"]).tg.sum().reset_index()
        tp["share"] = tp.tg / tp.groupby("posteam").tg.transform("sum")
        tp["val"] = tp.share * tp.pid.map(node)
        team = tp.groupby("posteam").val.sum()
        today = sched[sched.gameday == day]
        for gid, h, a in zip(today.game_id, today.home_team, today.away_team, strict=True):
            for tm in (h, a):
                rows.append((gid, tm, team.get(tm, np.nan)))
    return pd.DataFrame(rows, columns=["game_id", "team", "val"])


def coef_draws(table, cols):
    frame = table.loc[table.game_type.eq("REG") & table.ats_margin.notna()]
    train = frame.loc[frame.season.between(SCORE[0], SCORE[-1])]
    blocks = [b for _, b in train.groupby(["season", "week"])]
    rng = np.random.default_rng(SEED)
    out = []
    for _ in range(DRAWS):
        idx = rng.integers(0, len(blocks), len(blocks))
        s = pd.concat([blocks[i] for i in idx])
        model, _ = g.fit_frame(s.assign(season=0), cols, 1)
        names = list(model[:-1].get_feature_names_out())
        coef = np.ravel(model[-1].coef_)
        out.append(float(coef[next(i for i, n in enumerate(names) if "graph" in n)]))
    d = np.array(out)
    return {
        "mean": float(d.mean()),
        "low": float(np.percentile(d, 2.5)),
        "high": float(np.percentile(d, 97.5)),
        "probability_positive": float((d > 0).mean()),
    }


def stage_run():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    served = pd.read_parquet(m.SERVED_FEATURES)
    agg = g.load_agg(served)
    k, n_full, reliability = g.shrink_k(agg)
    tv = team_values(agg, served, k)
    table = g.attach(served, tv)
    table[g.COLUMN] = table[g.COLUMN].where(table.season.gt(WARM), 0.0)
    arm_cols = (*m.BASE_COLUMNS, g.COLUMN)
    base = m.decisive(m.run_eval(table, m.BASE_COLUMNS, "base"))
    arm = m.decisive(m.run_eval(table, arm_cols, "graph"))
    a, b = arm[arm.season.isin(SCORE)], base[base.season.isin(SCORE)]
    results = {"k": k, "n_full": n_full, "reliability": reliability}
    results["base_record_2023_2025"] = m.record(b)
    results["graph_v_base_2023_2025"] = m.compare("graph v base 2023-25", a, b)
    results["fold_coefficients"] = {}
    for season in (2023, 2024, 2025):
        model, _ = g.fit_frame(table, arm_cols, season)
        names = list(model[:-1].get_feature_names_out())
        coef = np.ravel(model[-1].coef_)
        results["fold_coefficients"][season] = {
            n: float(c) for n, c in zip(names, coef, strict=False) if "graph" in n
        }
    results["coefficient_bootstrap_2023_2025_training"] = coef_draws(table, arm_cols)
    (OUT_DIR / "report_warm.json").write_text(json.dumps(results, indent=2, default=str))
    print(json.dumps(results, indent=1, default=str), flush=True)


if __name__ == "__main__":
    if sys.argv[1:] == ["run"]:
        stage_run()
    else:
        raise SystemExit("usage: mod10_graph_term_warm.py run")
