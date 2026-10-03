import argparse
import inspect
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "src"))

PBP_DIR = REPO / "data" / "pbp" / "raw" / "20260925T202544Z"
OUT_DIR = REPO / "artifacts" / "mod25_generator"
SYN_DIR = REPO / "data" / "processed" / "synthetic"
KEYS = (3, 7, 10, 14, 17)
BUCKETS = ((1, 4), (5, 9), (10, 18))
REAL_SEASONS = tuple(range(2011, 2026))
PBP_COLS = [
    "game_id", "season_type", "week", "home_team", "away_team", "posteam", "play_type",
    "epa", "yards_gained", "sack", "interception", "fumble_lost", "passer_player_id",
]


def load_real_plays(seasons):
    frames = []
    for s in seasons:
        f = pd.read_parquet(PBP_DIR / f"season={s}" / "plays.parquet", columns=PBP_COLS + ["season"])
        f = f[f["season_type"] == "REG"]
        frames.append(f)
    return pd.concat(frames, ignore_index=True)


def real_games(seasons):
    g = pd.read_parquet(REPO / "data" / "processed" / "game_features_pbp.parquet")
    g = g[(g["game_type"] == "REG") & g["season"].isin(seasons)]
    g = g[["game_id", "season", "week", "gameday", "home_team", "away_team", "home_score", "away_score"]]
    return g.dropna(subset=["home_score", "away_score"]).reset_index(drop=True)


STAT_COLS = [
    "attempts", "carries", "sacks_suffered", "passing_epa", "rushing_epa", "passing_yards",
    "rushing_yards", "passing_interceptions", "sack_fumbles_lost", "rushing_fumbles_lost",
    "receiving_fumbles_lost",
]


def team_stats_from_plays(plays):
    p = plays[plays["play_type"].isin(["run", "pass"])].copy()
    p["epa"] = p["epa"].fillna(0.0)
    is_pass = p["play_type"] == "pass"
    sack = (p["sack"] == 1) & is_pass
    att = is_pass & ~sack
    p["attempts"] = att.astype(float)
    p["carries"] = (~is_pass).astype(float)
    p["sacks_suffered"] = sack.astype(float)
    p["passing_epa"] = np.where(is_pass, p["epa"], 0.0)
    p["rushing_epa"] = np.where(~is_pass, p["epa"], 0.0)
    p["passing_yards"] = np.where(att, p["yards_gained"], 0.0)
    p["rushing_yards"] = np.where(~is_pass, p["yards_gained"], 0.0)
    p["passing_interceptions"] = np.where(is_pass, p["interception"].fillna(0.0), 0.0)
    fl = p["fumble_lost"].fillna(0.0)
    p["sack_fumbles_lost"] = np.where(sack, fl, 0.0)
    p["rushing_fumbles_lost"] = np.where(~is_pass, fl, 0.0)
    p["receiving_fumbles_lost"] = np.where(att, fl, 0.0)
    return (
        p.groupby(["game_id", "posteam"], as_index=False)[STAT_COLS]
        .sum()
        .rename(columns={"posteam": "team"})
    )


def schedules_from_games(games):
    s = games.copy()
    s["game_type"] = "REG"
    s["result"] = s["home_score"] - s["away_score"]
    s["spread_line"] = np.nan
    s["gameday"] = pd.to_datetime(s["gameday"])
    return s


def margin_metrics(games):
    m = (games["home_score"] - games["away_score"]).to_numpy(dtype=float)
    a = np.abs(m)
    out = {f"mass_{k}": float(np.mean(a == k)) for k in KEYS}
    out["margin_sd"] = float(np.std(m, ddof=1))
    out["home_edge"] = float(np.mean(m))
    out["share_le3"] = float(np.mean(a <= 3))
    out["n_games"] = int(len(m))
    return out


def autocorr_and_r2(games, team_stats, season_filter=None):
    from nfl_ats.constants import DEFAULT_OFFSEASON_RETENTION
    from nfl_ats.features import build_team_game_metrics, build_team_states

    sched = schedules_from_games(games)
    tg = build_team_game_metrics(sched, team_stats)
    wk = games.set_index("game_id")["week"]
    tg = tg.assign(week=tg["game_id"].map(wk))
    tg = tg.sort_values(["team", "season", "week"])
    tg["dev"] = tg["off_epa_per_play"] - tg.groupby("season")["off_epa_per_play"].transform("mean")
    tg["lag"] = tg.groupby(["team", "season"])["dev"].shift(1)
    pairs = tg.dropna(subset=["dev", "lag"])
    if season_filter is not None:
        pairs = pairs[pairs["season"].isin(season_filter)]
    ac_by_season = {
        int(s): (
            float(np.sum(d["dev"] * d["lag"])),
            float(np.sum(d["dev"] ** 2)),
            float(np.sum(d["lag"] ** 2)),
        )
        for s, d in pairs.groupby("season")
    }
    states = build_team_states(tg.drop(columns=["week"]))
    states = states.sort_values(["team", "gameday", "game_id"]).reset_index(drop=True)
    grp = states.groupby("team", sort=False)
    prev_season = grp["season"].shift(1)
    gap = (states["season"] - prev_season).to_numpy(dtype=float)
    pre = {"game_id": states["game_id"], "team": states["team"]}
    for metric in ("off_epa_per_play", "def_epa_per_play"):
        value = grp[f"state_{metric}"].shift(1).to_numpy(dtype=float)
        lm = grp[f"league_mean_{metric}"].shift(1).to_numpy(dtype=float)
        ret = DEFAULT_OFFSEASON_RETENTION ** np.maximum(1.0, np.nan_to_num(gap, nan=1.0))
        adj = np.where(gap > 0, lm + ret * (value - lm), value)
        pre[metric] = adj
    st = pd.DataFrame(pre).set_index(["game_id", "team"])
    g = games.copy()
    for side in ("home", "away"):
        idx = pd.MultiIndex.from_arrays([g["game_id"], g[f"{side}_team"]])
        for c in ("off_epa_per_play", "def_epa_per_play"):
            g[f"{side}_state_{c}"] = st[c].reindex(idx).to_numpy()
    g["x"] = (g["home_state_off_epa_per_play"] - g["home_state_def_epa_per_play"]) - (
        g["away_state_off_epa_per_play"] - g["away_state_def_epa_per_play"]
    )
    g["y"] = g["home_score"] - g["away_score"]
    g = g.dropna(subset=["x", "y"])
    if season_filter is not None:
        g = g[g["season"].isin(season_filter)]
    return ac_by_season, g[["season", "week", "x", "y"]]


def r2_by_bucket(d):
    out = {}
    for lo, hi in BUCKETS:
        b = d[(d["week"] >= lo) & (d["week"] <= hi)]
        if len(b) < 10 or b["x"].std() == 0:
            out[f"r2_w{lo}_{hi}"] = float("nan")
            continue
        out[f"r2_w{lo}_{hi}"] = float(np.corrcoef(b["x"], b["y"])[0, 1] ** 2)
    return out


def autocorr_value(ac_by_season, seasons):
    sxy = sum(ac_by_season[s][0] for s in seasons if s in ac_by_season)
    sxx = sum(ac_by_season[s][1] for s in seasons if s in ac_by_season)
    syy = sum(ac_by_season[s][2] for s in seasons if s in ac_by_season)
    return float(sxy / math.sqrt(sxx * syy))


def all_metrics(games, team_stats, seasons):
    ac, d = autocorr_and_r2(games, team_stats, season_filter=set(seasons))
    out = margin_metrics(games[games["season"].isin(seasons)])
    out["epa_autocorr_lag1"] = autocorr_value(ac, seasons)
    out.update(r2_by_bucket(d))
    return out, ac, d


GATE_NAMES = [f"mass_{k}" for k in KEYS] + [
    "margin_sd", "home_edge", "share_le3", "epa_autocorr_lag1", "r2_w1_4", "r2_w5_9", "r2_w10_18",
]


def real_targets(n_boot=2000, seed=7):
    games = real_games(tuple(range(2009, 2026)))
    plays = load_real_plays(range(2009, 2026))
    ts = team_stats_from_plays(plays)
    games = games[games["game_id"].isin(ts["game_id"])].reset_index(drop=True)
    seasons = list(REAL_SEASONS)
    point, ac, d = all_metrics(games, ts, seasons)
    rng = np.random.default_rng(seed)
    gs = {s: games[games["season"] == s] for s in seasons}
    ds = {s: d[d["season"] == s] for s in seasons}
    boots = {k: [] for k in GATE_NAMES}
    for _ in range(n_boot):
        pick = rng.choice(seasons, size=len(seasons), replace=True)
        mm = margin_metrics(pd.concat([gs[s] for s in pick]))
        mm.update(r2_by_bucket(pd.concat([ds[s] for s in pick])))
        sxy = sum(ac[s][0] for s in pick)
        sxx = sum(ac[s][1] for s in pick)
        syy = sum(ac[s][2] for s in pick)
        mm["epa_autocorr_lag1"] = sxy / math.sqrt(sxx * syy)
        for k in GATE_NAMES:
            boots[k].append(mm[k])
    gates = {}
    for k in GATE_NAMES:
        arr = np.array(boots[k], dtype=float)
        gates[k] = {
            "real": point[k],
            "lo": float(np.nanpercentile(arr, 2.5)),
            "hi": float(np.nanpercentile(arr, 97.5)),
        }
    payload = {"seasons": [seasons[0], seasons[-1]], "n_games": point["n_games"], "gates": gates}
    return payload, games, ts


def cmd_real(args):
    t = time.time()
    payload, games, ts = real_targets()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "real_targets.json").write_text(json.dumps(payload, indent=1))
    for k, v in payload["gates"].items():
        print(f"{k:20s} real {v['real']:.4f}  95% [{v['lo']:.4f}, {v['hi']:.4f}]")
    print("n_games", payload["n_games"], "seconds", round(time.time() - t, 1))


def team_game_epa(plays):
    p = plays[plays["play_type"].isin(["run", "pass"])].copy()
    p["epa"] = p["epa"].fillna(0.0)
    off = p.groupby(["game_id", "posteam"]).agg(off=("epa", "mean"), n=("epa", "size")).reset_index()
    p["defteam"] = np.where(p["posteam"] == p["home_team"], p["away_team"], p["home_team"])
    de = p.groupby(["game_id", "defteam"]).agg(dfn=("epa", "mean")).reset_index().rename(columns={"defteam": "posteam"})
    meta = p.groupby("game_id").agg(season=("season", "first"), week=("week", "first")).reset_index()
    out = off.merge(de, on=["game_id", "posteam"]).merge(meta, on="game_id")
    return out.sort_values(["posteam", "season", "week"]).reset_index(drop=True)


def fit_ar1(tg, col, max_lag=6):
    d = tg.copy()
    d["dev"] = d[col] - d.groupby("season")[col].transform("mean")
    var_total = float(d["dev"].var())
    gam = []
    for k in range(1, max_lag + 1):
        lagged = d.groupby(["posteam", "season"])["dev"].shift(k)
        ok = lagged.notna()
        gam.append(float(np.mean(d.loc[ok, "dev"].to_numpy() * lagged[ok].to_numpy())))
    ks = np.arange(1, max_lag + 1)
    g = np.array(gam)
    good = g > 0
    slope, intercept = np.polyfit(ks[good], np.log(g[good]), 1)
    phi = float(np.exp(slope))
    var_mu = float(np.exp(intercept))
    return {"var_total": var_total, "autocov": gam, "phi": phi, "var_mu": var_mu}


def fit_offseason(tg, col, phi, var_mu, n_games=17, seed=3):
    d = tg.copy()
    d["dev"] = d[col] - d.groupby("season")[col].transform("mean")
    sm = d.groupby(["posteam", "season"])["dev"].mean().reset_index()
    nxt = sm.copy()
    nxt["season"] = nxt["season"] - 1
    m = sm.merge(nxt, on=["posteam", "season"], suffixes=("", "_next"))
    obs_cov = float(np.mean(m["dev"] * m["dev_next"]))
    rng = np.random.default_rng(seed)
    best = None
    for rho in np.linspace(0.0, 1.0, 101):
        n = 4000
        z = rng.standard_normal((n, n_games)) * math.sqrt(var_mu * (1 - phi * phi))
        lat = np.zeros((n, n_games))
        lat[:, 0] = rng.standard_normal(n) * math.sqrt(var_mu)
        for t in range(1, n_games):
            lat[:, t] = phi * lat[:, t - 1] + z[:, t]
        start = rho * lat[:, -1] + math.sqrt(max(0.0, 1 - rho * rho)) * rng.standard_normal(n) * math.sqrt(var_mu)
        nxt_lat = np.zeros((n, n_games))
        nxt_lat[:, 0] = start
        z2 = rng.standard_normal((n, n_games)) * math.sqrt(var_mu * (1 - phi * phi))
        for t in range(1, n_games):
            nxt_lat[:, t] = phi * nxt_lat[:, t - 1] + z2[:, t]
        c = float(np.mean(lat.mean(axis=1) * nxt_lat.mean(axis=1)))
        if best is None or abs(c - obs_cov) < best[0]:
            best = (abs(c - obs_cov), float(rho), c)
    return {"rho_offseason": best[1], "obs_cov_season_means": obs_cov, "model_cov": best[2]}


def fit_qb(plays, tg):
    p = plays[(plays["play_type"] == "pass") & plays["passer_player_id"].notna()]
    c = p.groupby(["game_id", "posteam", "passer_player_id"]).size().reset_index(name="n")
    top = c.sort_values("n", ascending=False).drop_duplicates(["game_id", "posteam"])
    meta = tg[["game_id", "posteam", "season", "week", "off"]]
    top = top.merge(meta, on=["game_id", "posteam"])
    seas = c.merge(meta[["game_id", "posteam", "season"]], on=["game_id", "posteam"])
    tot = seas.groupby(["posteam", "season", "passer_player_id"])["n"].sum().reset_index()
    starter = tot.sort_values("n", ascending=False).drop_duplicates(["posteam", "season"])
    starter = starter.rename(columns={"passer_player_id": "starter"})[["posteam", "season", "starter"]]
    top = top.merge(starter, on=["posteam", "season"])
    top["out"] = (top["passer_player_id"] != top["starter"]).astype(int)
    top = top.sort_values(["posteam", "season", "week"])
    top["prev_out"] = top.groupby(["posteam", "season"])["out"].shift(1)
    rate = float(top["out"].mean())
    p_start = float(top.loc[top["prev_out"] == 0, "out"].mean())
    p_stay = float(top.loc[top["prev_out"] == 1, "out"].mean())
    top["dev"] = top["off"] - top.groupby("season")["off"].transform("mean")
    top["team_mean_starter"] = top.groupby(["posteam", "season"])["dev"].transform(
        lambda x: x[top.loc[x.index, "out"] == 0].mean()
    )
    eff = top[top["out"] == 1]
    effect = float((eff["dev"] - eff["team_mean_starter"]).mean())
    se = float((eff["dev"] - eff["team_mean_starter"]).std() / math.sqrt(len(eff)))
    return {
        "backup_rate": rate, "p_start_out": p_start, "p_stay_out": p_stay,
        "backup_off_epa_effect": effect, "backup_effect_se": se, "n_backup_games": int(len(eff)),
        "n_team_games": int(len(top)),
    }


def cmd_fit(args):
    plays = load_real_plays(REAL_SEASONS)
    tg = team_game_epa(plays)
    res = {"off": fit_ar1(tg, "off"), "def": fit_ar1(tg, "dfn")}
    res["off"].update(fit_offseason(tg, "off", res["off"]["phi"], res["off"]["var_mu"]))
    res["def"].update(fit_offseason(tg, "dfn", res["def"]["phi"], res["def"]["var_mu"]))
    sm = tg.copy()
    for c in ("off", "dfn"):
        sm[c] = sm[c] - sm.groupby("season")[c].transform("mean")
    sm = sm.groupby(["posteam", "season"])[["off", "dfn"]].mean()
    res["season_mean_corr_off_def"] = float(sm["off"].corr(sm["dfn"]))
    res["season_mean_sd"] = {"off": float(sm["off"].std()), "def": float(sm["dfn"].std())}
    res["qb"] = fit_qb(plays, tg)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "fit_params.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


ENGINE_SEASONS = tuple(range(2009, 2020))
SETTING_DEFAULTS = {
    "scale": 1.0, "drift": 1.0, "yard_gain": 0.6425, "def_sign": -1.0, "epa_per_yard": 0.07,
    "yard_bias": 0.0,
}
N_TEAMS = 32
BYE_WEEKS = (6, 7, 8, 9, 10, 11, 12, 13)
RATING_GRID = 0.0025
_G = {}


def load_fit():
    return json.loads((OUT_DIR / "fit_params.json").read_text())


def build_engine(seasons=ENGINE_SEASONS):
    import sim04_engine as sim

    cap = {}
    orig_trans = sim.build_transition_frame
    orig_load = sim.load_reg_seasons

    def load_hook(seasons_):
        pbp = orig_load(seasons_)
        cap["pbp"] = pbp
        return pbp

    def trans_hook(pbp, team_ratings=None):
        tr = orig_trans(pbp, team_ratings=team_ratings)
        cap["trans"] = tr
        return tr

    sim.load_reg_seasons = load_hook
    sim.build_transition_frame = trans_hook
    try:
        tables = sim.build_tables(tuple(seasons), condition_on_team=True)
    finally:
        sim.load_reg_seasons = orig_load
        sim.build_transition_frame = orig_trans
    trans = cap["trans"]
    pbp = cap["pbp"]
    cols = ["game_id", "play_id", "epa", "sack", "interception", "fumble_lost", "yards_gained"]
    right = pbp[cols].drop_duplicates(["game_id", "play_id"])
    attrs = trans[["game_id", "play_id"]].merge(right, on=["game_id", "play_id"], how="left")
    code = tables["arrays"]["play_type_code"]
    tables["attrs"] = {
        "epa": attrs["epa"].fillna(0.0).to_numpy(dtype=np.float32),
        "sack": attrs["sack"].fillna(0.0).to_numpy(dtype=np.int8),
        "int": attrs["interception"].fillna(0.0).to_numpy(dtype=np.int8),
        "fl": attrs["fumble_lost"].fillna(0.0).to_numpy(dtype=np.int8),
        "yards": attrs["yards_gained"].fillna(0.0).to_numpy(dtype=np.float32),
        "code": code,
    }
    return sim, tables


def make_runner(sim, setting):
    src = inspect.getsource(sim.run_one_game)
    a = 'drawn_net = float(arrays["off_row"][idx] - arrays["def_row"][idx])'
    b = "yard_shift = TEAM_RATING_YARD_GAIN * ((off_sim - def_sim) - drawn_net)"
    c = "drawn = policy(down, distance, yardline, score_diff, qtr, clock_val, drawn)"
    assert a in src and b in src and c in src
    src = src.replace(a, 'drawn_net = float(arrays["off_row"][idx] + DEF_SIGN * arrays["def_row"][idx])')
    src = src.replace(b, "yard_shift = YARD_BIAS + YARD_GAIN * ((off_sim + DEF_SIGN * def_sim) - drawn_net)")
    log_line = (
        "\n        PLAYLOG.append((1 if offense == 'home' else 0, int(idx),"
        " float(drawn['yards_gained']) - float(arrays['yards_gained'][idx])))"
    )
    src = src.replace(c, c + log_line)
    ns = dict(sim.__dict__)
    ns["PLAYLOG"] = []
    ns["YARD_GAIN"] = float(setting["yard_gain"])
    ns["DEF_SIGN"] = float(setting["def_sign"])
    ns["YARD_BIAS"] = float(setting["yard_bias"])
    exec(src, ns)
    return ns


def init_worker(setting):
    sim, tables = build_engine()
    _G["sim"] = sim
    _G["tables"] = tables
    _G["ns"] = make_runner(sim, setting)
    _G["setting"] = setting


def quant(x):
    return round(float(x) / RATING_GRID) * RATING_GRID


def play_season(task):
    sim = _G["sim"]
    tables = _G["tables"]
    ns = _G["ns"]
    setting = _G["setting"]
    arr = tables["attrs"]
    world, sidx, seed, sched, ratings = task
    rng = np.random.default_rng(seed)
    gm_rows = []
    play_chunks = []
    for gi, (week, h, a) in enumerate(sched):
        state = sim.initial_kickoff_state(rng, tables["opening_pool"])
        ns["PLAYLOG"].clear()
        hr = {"off": quant(ratings[week][h][0]), "def": quant(ratings[week][h][1])}
        ar = {"off": quant(ratings[week][a][0]), "def": quant(ratings[week][a][1])}
        rec, cap_hit = ns["run_one_game"](
            state, tables, rng, 600.0, sim.default_policy, sim.K_NEIGHBORS, sim.MAX_PLAYS_PER_GAME, hr, ar
        )
        tables["nn_weight_cache_cond"].clear()
        log = np.array(ns["PLAYLOG"], dtype=np.float64).reshape(-1, 3)
        idx = log[:, 1].astype(np.int64)
        code = arr["code"][idx]
        keep = (code == 0) | (code == 1)
        idx = idx[keep]
        shift = log[keep, 2]
        offhome = log[keep, 0].astype(np.int8)
        kind = np.where(code[keep] == 0, 0, np.where(arr["sack"][idx] == 1, 2, 1)).astype(np.int8)
        epa = arr["epa"][idx] + setting["epa_per_yard"] * shift
        yards = arr["yards"][idx] + shift
        turn = np.where(arr["int"][idx] == 1, 1, np.where(arr["fl"][idx] == 1, 2, 0)).astype(np.int8)
        play_chunks.append(
            np.rec.fromarrays(
                [np.full(len(idx), gi, dtype=np.int16), offhome, kind, epa.astype(np.float32), yards.astype(np.float32), turn],
                names="g,offhome,kind,epa,yards,turn",
            )
        )
        total = float(rec["total"])
        margin = float(rec["margin"])
        gm_rows.append((gi, (total + margin) / 2.0, (total - margin) / 2.0, bool(rec["went_ot"]), bool(cap_hit)))
    plays = np.concatenate(play_chunks) if play_chunks else np.empty(0)
    return world, sidx, gm_rows, plays


def make_schedule(rng):
    byes = np.array(list(BYE_WEEKS) * 4)
    team_perm = rng.permutation(N_TEAMS)
    bye_of = {int(team_perm[i]): int(byes[i]) for i in range(N_TEAMS)}
    sched = []
    for week in range(1, 19):
        teams = [t for t in range(N_TEAMS) if bye_of[t] != week]
        teams = list(rng.permutation(teams))
        for i in range(0, len(teams), 2):
            sched.append((week, int(teams[i]), int(teams[i + 1])))
    return sched


def gen_world_latents(rng, n_seasons, fit, setting):
    qb = fit["qb"]
    out = {}
    var_off = max(1e-6, fit["off"]["var_mu"] - qb["backup_rate"] * (1 - qb["backup_rate"]) * qb["backup_off_epa_effect"] ** 2)
    var_def = fit["def"]["var_mu"]
    phi_o = fit["off"]["phi"] ** setting["drift"]
    phi_d = fit["def"]["phi"] ** setting["drift"]
    rho_o = fit["off"]["rho_offseason"]
    rho_d = fit["def"]["rho_offseason"]
    corr = fit["season_mean_corr_off_def"]
    L = np.linalg.cholesky(np.array([[1.0, corr], [corr, 1.0]]))
    sd = np.array([math.sqrt(var_off), math.sqrt(var_def)])
    lat = (rng.standard_normal((N_TEAMS, 2)) @ L.T) * sd
    seasons = []
    for s in range(n_seasons):
        if s > 0:
            fresh = (rng.standard_normal((N_TEAMS, 2)) @ L.T) * sd
            rho = np.array([rho_o, rho_d])
            lat = rho * lat + np.sqrt(1 - rho**2) * fresh
        weekly = np.zeros((18, N_TEAMS, 2))
        qb_out = np.zeros((18, N_TEAMS), dtype=bool)
        state = np.zeros(N_TEAMS, dtype=bool)
        cur = lat.copy()
        for w in range(18):
            if w > 0:
                phi = np.array([phi_o, phi_d])
                innov = (rng.standard_normal((N_TEAMS, 2)) @ L.T) * sd * np.sqrt(1 - phi**2)
                cur = phi * cur + innov
            u = rng.random(N_TEAMS)
            state = np.where(state, u < qb["p_stay_out"], u < qb["p_start_out"])
            weekly[w] = cur
            qb_out[w] = state
        lat = cur
        seasons.append((weekly, qb_out))
    return seasons


def season_ratings(weekly, qb_out, fit, setting, league_off, league_def):
    shock = fit["qb"]["backup_off_epa_effect"]
    scale = setting["scale"]
    ratings = {}
    centre = np.zeros(18)
    if os.environ.get("QBC") == "1":
        qb = fit["qb"]
        p = 0.0
        for w in range(18):
            p = p * qb["p_stay_out"] + (1.0 - p) * qb["p_start_out"]
            centre[w] = p * shock
    for w in range(18):
        per = []
        for t in range(N_TEAMS):
            off = weekly[w, t, 0] + (shock if qb_out[w, t] else 0.0) - centre[w]
            per.append((league_off + scale * off, league_def + scale * weekly[w, t, 1]))
        ratings[w + 1] = per
    return ratings


def league_means():
    g = pd.read_parquet(REPO / "data" / "processed" / "game_features_pbp.parquet")
    g = g[(g["game_type"] == "REG") & g["season"].isin(ENGINE_SEASONS)]
    off = pd.concat([g["home_off_epa_per_play"], g["away_off_epa_per_play"]]).mean()
    de = pd.concat([g["home_def_epa_per_play"], g["away_def_epa_per_play"]]).mean()
    return float(off), float(de)


def run_generation(setting, n_worlds, n_seasons, workers, seed, keep_plays=False, progress=True):
    import multiprocessing as mp

    fit = load_fit()
    league_off, league_def = league_means()
    root = np.random.SeedSequence(seed)
    world_seeds = root.spawn(n_worlds)
    tasks = []
    latents = {}
    scheds = {}
    for w in range(n_worlds):
        rng = np.random.default_rng(world_seeds[w])
        lat = gen_world_latents(rng, n_seasons, fit, setting)
        game_seeds = world_seeds[w].spawn(n_seasons)
        for sidx, (weekly, qb_out) in enumerate(lat):
            sched = make_schedule(rng)
            ratings = season_ratings(weekly, qb_out, fit, setting, league_off, league_def)
            scheds[(w, sidx)] = sched
            latents[(w, sidx)] = (weekly.astype(np.float32), qb_out)
            tasks.append((w, sidx, int(game_seeds[sidx].generate_state(1)[0]), sched, ratings))
    t0 = time.time()
    results = []
    for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[k] = "1"
    ctx = mp.get_context("spawn")
    with ctx.Pool(workers, initializer=init_worker, initargs=(setting,)) as pool:
        for i, r in enumerate(pool.imap_unordered(play_season, tasks, chunksize=1)):
            results.append(r)
            if progress and (i + 1) % 10 == 0:
                print(f"  seasons {i + 1}/{len(tasks)} {time.time() - t0:.0f}s", flush=True)
    elapsed = time.time() - t0
    game_rows = []
    play_frames = []
    for world, sidx, gm_rows, plays in results:
        sched = scheds[(world, sidx)]
        season_id = world * 1000 + sidx + 1
        base = pd.Timestamp("2000-09-05") + pd.Timedelta(days=365 * sidx)
        for gi, hs, as_, went_ot, cap_hit in gm_rows:
            week, h, a = sched[gi]
            game_rows.append(
                (f"w{world:04d}s{sidx:02d}g{gi:03d}", season_id, week, base + pd.Timedelta(days=7 * (week - 1)),
                 f"W{world}T{h}", f"W{world}T{a}", hs, as_, went_ot, cap_hit)
            )
        if len(plays):
            gid = np.array([f"w{world:04d}s{sidx:02d}g{g:03d}" for g in range(len(sched))])[plays["g"]]
            home_t = np.array([f"W{world}T{x[1]}" for x in sched])[plays["g"]]
            away_t = np.array([f"W{world}T{x[2]}" for x in sched])[plays["g"]]
            pf = pd.DataFrame({
                "game_id": gid,
                "posteam": np.where(plays["offhome"] == 1, home_t, away_t),
                "play_type": np.where(plays["kind"] == 0, "run", "pass"),
                "sack": (plays["kind"] == 2).astype(np.int8),
                "interception": (plays["turn"] == 1).astype(np.int8),
                "fumble_lost": (plays["turn"] == 2).astype(np.int8),
                "epa": plays["epa"],
                "yards_gained": plays["yards"],
            })
            play_frames.append(pf)
    games = pd.DataFrame(
        game_rows,
        columns=["game_id", "season", "week", "gameday", "home_team", "away_team", "home_score", "away_score", "went_ot", "cap_hit"],
    )
    plays_all = pd.concat(play_frames, ignore_index=True)
    return games, plays_all, latents, elapsed


def evaluate(games, plays, n_seasons, n_worlds, burn=2):
    ts = team_stats_from_plays(plays)
    keep = [w * 1000 + s + 1 for w in range(n_worlds) for s in range(burn, n_seasons)]
    metrics, ac, d = all_metrics(games, ts, keep)
    return metrics, ts


def judge(metrics, targets):
    rows = {}
    for k in GATE_NAMES:
        t = targets["gates"][k]
        v = metrics[k]
        rows[k] = {"synthetic": v, "real": t["real"], "lo": t["lo"], "hi": t["hi"], "pass": bool(t["lo"] <= v <= t["hi"])}
    return rows


def parse_setting(args):
    st = dict(SETTING_DEFAULTS)
    for k in ("scale", "drift", "yard_gain", "def_sign", "epa_per_yard", "yard_bias"):
        v = getattr(args, k, None)
        if v is not None:
            st[k] = v
    return st


def cmd_gen(args):
    setting = parse_setting(args)
    targets = json.loads((OUT_DIR / "real_targets.json").read_text())
    games, plays, latents, elapsed = run_generation(setting, args.worlds, args.seasons, args.workers, args.seed)
    n_games = len(games)
    metrics, ts = evaluate(games, plays, args.seasons, args.worlds)
    rows = judge(metrics, targets)
    res = {
        "setting": setting, "worlds": args.worlds, "seasons": args.seasons, "games": n_games,
        "seconds": elapsed, "games_per_sec": n_games / elapsed, "gates": rows,
        "all_pass": all(r["pass"] for r in rows.values()), "cap_hits": int(games["cap_hit"].sum()),
        "ot_share": float(games["went_ot"].mean()),
        "mean_total_points": float((games["home_score"] + games["away_score"]).mean()),
        "plays_per_game": float(len(plays) / n_games),
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / f"gen_{args.tag}.json").write_text(json.dumps(res, indent=1))
    print(f"games {n_games} {elapsed:.0f}s {n_games / elapsed:.1f} games/s all_pass={res['all_pass']} total_pts {res['mean_total_points']:.1f} runpass/g {res['plays_per_game']:.0f}")
    for k, r in rows.items():
        print(f"{k:20s} syn {r['synthetic']:.4f} real {r['real']:.4f} [{r['lo']:.4f},{r['hi']:.4f}] {'PASS' if r['pass'] else 'FAIL'}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("real")
    sub.add_parser("fit")
    g = sub.add_parser("gen")
    g.add_argument("--tag", default="pilot")
    g.add_argument("--worlds", type=int, default=1)
    g.add_argument("--seasons", type=int, default=4)
    g.add_argument("--workers", type=int, default=3)
    g.add_argument("--seed", type=int, default=11)
    for k in ("scale", "drift", "yard_gain", "def_sign", "epa_per_yard", "yard_bias"):
        g.add_argument("--" + k.replace("_", "-"), dest=k, type=float, default=None)
    args = ap.parse_args()
    {"real": cmd_real, "fit": cmd_fit, "gen": cmd_gen}[args.cmd](args)


if __name__ == "__main__":
    main()
