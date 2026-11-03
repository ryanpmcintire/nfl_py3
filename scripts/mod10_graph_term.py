import json
import sys

import mod24_ftn_turnover_luck as m
import numpy as np
import pandas as pd

from nfl_ats.margin import (
    _PROFILE_SUPPRESSED_MISSING_INDICATORS,
    _target_values,
    make_margin_estimator,
)

OUT_DIR = m.ROOT / "artifacts" / "mod10_graph_term"
TRAITS = m.ROOT / "artifacts" / "mod10_ftn_traits" / "result.json"
PBP_COLUMNS = [
    "game_id",
    "play_id",
    "season",
    "season_type",
    "week",
    "posteam",
    "receiver_player_id",
    "pass_attempt",
]
HALF_MIN = 25
COLUMN = "ftn_graph_diff"
CUTOFF = pd.Timestamp("2024-11-10")
TRAIT_KEYS = {"created": "rec_created_reception_rate", "contested": "rec_contested_rate"}


def load_agg(games):
    ftn = pd.concat(
        [
            pd.read_parquet(m.FTN_HIST / f"ftn_charting_{s}.parquet")
            for s in (2022, 2023, 2024, 2025)
        ]
        + [pd.read_parquet(m.FTN_2026)],
        ignore_index=True,
    ).rename(columns={"nflverse_game_id": "game_id", "nflverse_play_id": "play_id"})
    pbp = pd.concat(
        [
            pd.read_parquet(m.PBP_ROOT / f"season={s}", columns=PBP_COLUMNS)
            for s in range(2022, 2027)
        ],
        ignore_index=True,
    )
    pbp = pbp[(pbp.season_type == "REG") & pbp.receiver_player_id.notna() & (pbp.pass_attempt == 1)]
    pbp = pbp.assign(play_id=pbp.play_id.astype(float))
    ftn = ftn.assign(play_id=ftn.play_id.astype(float))
    t = pbp.merge(ftn.drop(columns=["season", "week"]), on=["game_id", "play_id"], how="inner")
    for c in ("is_catchable_ball", "is_contested_ball", "is_created_reception"):
        t[c] = t[c].fillna(False).astype(bool)
    t["tg"] = 1.0
    t["c"] = t.is_catchable_ball.astype(float)
    t["cr"] = (t.is_created_reception & t.is_catchable_ball).astype(float)
    t["ct"] = (t.is_contested_ball & t.is_catchable_ball).astype(float)
    agg = t.groupby(["game_id", "posteam", "receiver_player_id", "week", "season"], as_index=False)[
        ["tg", "c", "cr", "ct"]
    ].sum()
    dates = dict(zip(games.game_id, pd.to_datetime(games.gameday), strict=True))
    agg["gameday"] = agg.game_id.map(dates)
    return agg.dropna(subset=["gameday"]).rename(columns={"receiver_player_id": "pid"})


def shrink_k(agg):
    reliability = {
        r["trait"]: r["sb_pooled"]["point"]
        for r in json.loads(TRAITS.read_text())["results"]
        if not r["team_context_removed"]
    }
    h = agg[agg.season <= 2025].assign(half=lambda d: np.where(d.week % 2 == 1, "odd", "even"))
    g = h.groupby(["season", "pid", "half"]).c.sum().unstack("half").dropna()
    g = g[(g.odd >= HALF_MIN) & (g.even >= HALF_MIN)]
    n_full = float((g.odd + g.even).median())
    k = {key: n_full * (1 - reliability[v]) / reliability[v] for key, v in TRAIT_KEYS.items()}
    return k, n_full, {key: reliability[v] for key, v in TRAIT_KEYS.items()}


def team_values(agg, games, k):
    rows = []
    sched = games.assign(gameday=pd.to_datetime(games.gameday))
    sched = sched[sched.season >= 2022]
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
        node = dev_cr + dev_ct
        tp = prior.groupby(["posteam", "pid"]).tg.sum().reset_index()
        tp["share"] = tp.tg / tp.groupby("posteam").tg.transform("sum")
        tp["val"] = tp.share * tp.pid.map(node)
        team = tp.groupby("posteam").val.sum()
        today = sched[sched.gameday == day]
        for gid, h, a in zip(today.game_id, today.home_team, today.away_team, strict=True):
            for tm in (h, a):
                rows.append((gid, tm, team.get(tm, np.nan)))
    return pd.DataFrame(rows, columns=["game_id", "team", "val"])


def attach(features, tv):
    home = tv.rename(columns={"team": "home_team", "val": "hv"})
    away = tv.rename(columns={"team": "away_team", "val": "av"})
    out = features.merge(home, on=["game_id", "home_team"], how="left")
    out = out.merge(away, on=["game_id", "away_team"], how="left")
    out[COLUMN] = (out.hv.fillna(0.0) - out.av.fillna(0.0)).where(out.season.ge(2022), 0.0)
    return out.drop(columns=["hv", "av"])


def assert_pregame(agg, games, k, tv):
    days = pd.to_datetime(games.gameday)
    truncated = team_values(agg[agg.gameday < CUTOFF], games[days <= CUTOFF], k)
    ids = set(games.game_id[days == CUTOFF])
    a = tv[tv.game_id.isin(ids)].set_index(["game_id", "team"]).val
    b = truncated[truncated.game_id.isin(ids)].set_index(["game_id", "team"]).val
    both = a.index.intersection(b.index)
    if len(both) == 0 or not np.allclose(a[both], b[both], equal_nan=True, rtol=0, atol=1e-12):
        raise AssertionError("future information reaches the graph term")
    print("pregame truncation check passed", len(both), flush=True)


def fit_frame(table, columns, train_before):
    suppressed = tuple(
        c for c in _PROFILE_SUPPRESSED_MISSING_INDICATORS.get(m.PROFILE, ()) if c in columns
    )
    frame = table.loc[table.game_type.eq("REG") & table.ats_margin.notna()]
    train = frame.loc[frame.season.lt(train_before)]
    model = make_margin_estimator(
        "ridge", ridge_alpha=m.BASE_ALPHA, suppressed_indicator_columns=suppressed
    )
    model.fit(train.loc[:, list(columns)], _target_values(train, "market_residual"))
    return model, frame


def sign_acc(model, rows, columns):
    rows = rows[rows.ats_margin != 0]
    pred = np.ravel(model.predict(rows.loc[:, list(columns)]))
    return float(((pred > 0) == (rows.ats_margin > 0)).mean()), len(rows)


def diagnostics(table):
    arm_cols = (*m.BASE_COLUMNS, COLUMN)
    out = {"fold_coefficients": {}, "frozen_before_2026": {}}
    for season in (2023, 2024, 2025, 2026):
        model, _ = fit_frame(table, arm_cols, season)
        names = list(model[:-1].get_feature_names_out())
        coef = np.ravel(model[-1].coef_)
        out["fold_coefficients"][season] = {
            n: float(c) for n, c in zip(names, coef, strict=False) if "graph" in n
        }
    for label, cols in (("base", m.BASE_COLUMNS), ("arm", arm_cols)):
        model, frame = fit_frame(table, cols, 2026)
        ins, n_in = sign_acc(model, frame[frame.season.between(2022, 2025)], cols)
        fwd, n_fw = sign_acc(model, frame[frame.season.eq(2026)], cols)
        out["frozen_before_2026"][label] = {
            "in_sample_2022_2025": ins,
            "n_in": n_in,
            "frozen_2026": fwd,
            "n_2026": n_fw,
        }
    return out


def stage_run():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    served = pd.read_parquet(m.SERVED_FEATURES)
    agg = load_agg(served)
    k, n_full, reliability = shrink_k(agg)
    print("k", k, "n_full", n_full, flush=True)
    tv = team_values(agg, served, k)
    assert_pregame(agg, served, k, tv)
    table = attach(served, tv)
    print(table.loc[table.season.ge(2022), COLUMN].describe(), flush=True)
    base = m.decisive(m.run_eval(table, m.BASE_COLUMNS, "base"))
    arm = m.decisive(m.run_eval(table, (*m.BASE_COLUMNS, COLUMN), "graph"))
    results = {"k": k, "n_full": n_full, "reliability": reliability}
    seasons = m.SCORE_SEASONS
    a, b = arm[arm.season.isin(seasons)], base[base.season.isin(seasons)]
    results["base_record_2022_2025"] = m.record(b)
    results["graph_v_base_2022_2025"] = m.compare("graph v base 2022-25", a, b)
    a26, b26 = arm[arm.season.eq(2026)], base[base.season.eq(2026)]
    if len(a26):
        results["forward_2026_weekly_refit"] = m.compare("graph v base 2026", a26, b26)
    results.update(diagnostics(table))
    (OUT_DIR / "report.json").write_text(json.dumps(results, indent=2, default=str))
    print(json.dumps(results, indent=1, default=str), flush=True)


if __name__ == "__main__":
    if sys.argv[1:] == ["run"]:
        stage_run()
    else:
        raise SystemExit("usage: mod10_graph_term.py run")
