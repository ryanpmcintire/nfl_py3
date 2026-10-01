import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import mod23_unit3 as u3

from nfl_ats import clv
from nfl_ats.constants import DEFAULT_MIN_TRAIN_GAMES, FEATURE_SETS
from nfl_ats.home_side_location import (
    HOME_SIDE_OFFSET_SERVED,
    fit_home_side_offsets,
    prior_rows_before,
)
from nfl_ats.margin import (
    _MARGIN_PROFILE_FEATURE_SETS,
    _PROFILE_SUPPRESSED_MISSING_INDICATORS,
    fit_margin_model,
    make_margin_estimator,
    margin_feature_columns,
    resolve_feature_groups,
)
from nfl_ats.mass_preserving_lattice import (
    DiscretePushReader,
    prior_pool,
    serve_discrete_three_way,
)
from nfl_ats.modeling import regular_season_rows

ROOT = u3.ROOT
OUT = ROOT / "artifacts" / "mod24_u1"
SBR = ROOT / "data" / "processed" / "sbr_odds.parquet"
SEL_TRUE = ROOT / "artifacts" / "mod23_unit1" / "selection.json"
SEL_PROXY = OUT / "selection_proxy.json"
TERMS5 = ROOT / "artifacts" / "mod23_unit5" / "terms.parquet"
PROFILE = "weak_stack"
SET_NAME = _MARGIN_PROFILE_FEATURE_SETS[PROFILE][1]
BASE_ALPHA = 10.0
ALPHA_GRID = (1.0, 3.0, 10.0, 30.0, 100.0)
PROXY_SEASONS = tuple(range(2011, 2020))
TRUE_SEASONS = tuple(range(2020, 2026))
INNER_SEASONS = 4
FIRST_INNER = 2010
MIN_RELATIVE_GAIN = 0.0005
PINNED = "market"
PROB_METHOD = "gaussian_median"
BOOT = 20000
SEED = 20260817
FULL = tuple(margin_feature_columns("market_residual", PROFILE))
COLFAM = dict(zip(FULL, resolve_feature_groups(FULL), strict=True))
FAMILIES = sorted({f for f in COLFAM.values() if f != PINNED})
SUPP = _PROFILE_SUPPRESSED_MISSING_INDICATORS.get(PROFILE, ())


def cols_for(dropped):
    return tuple(c for c in FULL if COLFAM[c] not in dropped)


def inner_mse(frame, columns, alpha, seasons):
    total = 0.0
    count = 0
    for s in seasons:
        train = frame.loc[frame["season"].lt(s)]
        test = frame.loc[frame["season"].eq(s)]
        est = make_margin_estimator("ridge", 42, ridge_alpha=alpha, suppressed_indicator_columns=SUPP)
        est.fit(train.loc[:, list(columns)], train["ats_margin"])
        pred = np.asarray(est.predict(test.loc[:, list(columns)]), dtype=float)
        err = test["ats_margin"].to_numpy(dtype=float) - pred
        total += float(np.sum(err**2))
        count += len(err)
    return total / count


def best_alpha(frame, columns, inner):
    scores = {a: inner_mse(frame, columns, a, inner) for a in ALPHA_GRID}
    a = min(ALPHA_GRID, key=lambda x: (scores[x], abs(x - BASE_ALPHA)))
    return a, scores[a]


def select_alpha_only(frame, season):
    inner = list(range(max(FIRST_INNER, season - INNER_SEASONS), season))
    a, s = best_alpha(frame, FULL, inner)
    return {"alpha": a, "dropped": [], "inner_mse": s}


def select_greedy(frame, season):
    inner = list(range(max(FIRST_INNER, season - INNER_SEASONS), season))
    dropped = frozenset()
    alpha, current = best_alpha(frame, cols_for(dropped), inner)
    while True:
        cands = []
        for fam in FAMILIES:
            if fam in dropped:
                continue
            trial = dropped | {fam}
            if not cols_for(trial):
                continue
            a, s = best_alpha(frame, cols_for(trial), inner)
            cands.append((s, fam, a))
        if not cands:
            break
        s, fam, a = min(cands)
        if s < current * (1.0 - MIN_RELATIVE_GAIN):
            dropped = dropped | {fam}
            alpha, current = a, s
        else:
            break
    return {"alpha": alpha, "dropped": sorted(dropped), "inner_mse": current}


def patched(columns):
    class Ctx:
        def __enter__(self):
            self.orig = FEATURE_SETS[SET_NAME]
            FEATURE_SETS[SET_NAME] = tuple(columns)

        def __exit__(self, *a):
            FEATURE_SETS[SET_NAME] = self.orig

    return Ctx()


def proxy_population(features):
    reg = regular_season_rows(features)
    reg = reg.loc[reg["season"].between(PROXY_SEASONS[0], PROXY_SEASONS[-1])]
    sbr = pd.read_parquet(SBR).dropna(subset=["game_id", "open_home_spread"])
    sbr = sbr.drop_duplicates("game_id")
    pop = reg[["game_id", "season", "week", "result"]].merge(
        sbr[["game_id", "open_home_spread"]], on="game_id", how="inner"
    )
    pop = pop.loc[pd.to_numeric(pop["result"], errors="coerce").notna()].copy()
    pop["season"] = pop["season"].astype(int)
    pop["week"] = pop["week"].astype(int)
    return pop.rename(columns={"open_home_spread": "open_line"})


def proxy_eval(features, pop, spec):
    frame = regular_season_rows(features).copy()
    frame["gameday"] = pd.to_datetime(frame["gameday"])
    completed = frame.loc[frame["result"].notna()].copy()
    pool = prior_pool(frame, pop.set_index("game_id")["open_line"])
    rows = []
    stream = pd.DataFrame()
    for (season, week), group in pop.groupby(["season", "week"], sort=True):
        week_rows = frame.loc[frame["game_id"].isin(set(group["game_id"]))]
        target = frame.loc[frame["season"].eq(season) & frame["week"].eq(week)]
        cutoff = target["gameday"].min()
        training = completed.loc[completed["gameday"].lt(cutoff)]
        if len(training) < DEFAULT_MIN_TRAIN_GAMES:
            continue
        columns, alpha = spec(season)
        with patched(columns):
            model = fit_margin_model(
                training,
                target="market_residual",
                model_name="ridge",
                feature_profile=PROFILE,
                ridge_alpha=alpha,
            )
            reader = DiscretePushReader.for_week(
                pool,
                season=int(season),
                week=int(week),
                cutoff=pd.Timestamp(cutoff),
                exclude_game_ids=set(week_rows["game_id"].astype(str)),
            )
            scoring = week_rows.merge(group[["game_id", "open_line"]], on="game_id").copy()
            at_open = scoring.copy()
            at_open["spread_line"] = at_open["open_line"]
            pred = model.predict(at_open, probability_method=PROB_METHOD)
            if HOME_SIDE_OFFSET_SERVED and not stream.empty:
                fitted = fit_home_side_offsets(prior_rows_before(stream, int(season), int(week)))
                off = fitted.offset_for(at_open["spread_line"]).fillna(0.0).to_numpy(dtype=float)
            else:
                off = np.zeros(len(at_open))
            served = (
                model.predict(at_open, probability_method=PROB_METHOD, center_offset=off)
                if np.any(off != 0.0)
                else pred
            )
            disc = serve_discrete_three_way(
                served, at_open, reader, residuals=model.residuals, probability_method=PROB_METHOD
            )
        res = pred["predicted_market_residual"].to_numpy()
        out = scoring[["game_id", "result", "open_line"]].copy()
        out["season"] = int(season)
        out["week"] = int(week)
        out["p"] = disc["home_cover_probability"].to_numpy()
        rows.append(out)
        ws = pd.DataFrame(
            {
                "game_id": scoring["game_id"].astype(str).to_numpy(),
                "season": int(season),
                "week": int(week),
                "spread_line": scoring["open_line"].to_numpy(dtype=float),
                "point_incumbent": scoring["open_line"].to_numpy(dtype=float) + res,
                "result": pd.to_numeric(scoring["result"], errors="coerce").to_numpy(),
            }
        )
        stream = ws if stream.empty else pd.concat([stream, ws], ignore_index=True)
    out = pd.concat(rows, ignore_index=True)
    out["margin_vs_open"] = out["result"] - out["open_line"]
    return finish(out, out["p"])


def finish(df, p):
    df = df.copy()
    df["pick_home"] = p.to_numpy() >= 0.5
    m = df["margin_vs_open"]
    df["correct"] = np.where(m.eq(0), np.nan, np.where(df["pick_home"], m.gt(0), ~m.gt(0)))
    df = df.loc[df["correct"].notna()].copy()
    df["y"] = (df["margin_vs_open"] > 0).astype(float)
    pc = df["p"].clip(1e-6, 1 - 1e-6)
    df["ll"] = -(df["y"] * np.log(pc) + (1 - df["y"]) * np.log(1 - pc))
    df["brier"] = (pc - df["y"]) ** 2
    return df.set_index("game_id")[["season", "week", "p", "pick_home", "correct", "ll", "brier"]]


def true_eval(features, columns, alpha):
    cfg = {
        "feature_profile": PROFILE,
        "regressor": "ridge",
        "target": "market_residual",
        "probability_method": PROB_METHOD,
        "calibration_method": "none",
        "ridge_alpha": alpha,
        "mod24_variant": json.dumps([list(columns), alpha]),
    }
    with patched(columns):
        scored = clv.opener_pick_evaluation(
            u3.MARKET_ROOT, features, active_model_config=cfg, min_train_games=DEFAULT_MIN_TRAIN_GAMES
        )
    scored = scored.loc[scored["correct_at_open_probability_rule"].notna()].copy()
    scored["p"] = scored["home_cover_probability_at_open"]
    return finish(scored[["game_id", "season", "week", "p", "margin_vs_open"]], scored["p"])


def run_arm(features, pop, spec):
    proxy = proxy_eval(features, pop, spec)
    cache = {}
    parts = []
    for s in TRUE_SEASONS:
        columns, alpha = spec(s)
        key = (tuple(columns), alpha)
        if key not in cache:
            cache[key] = true_eval(features, columns, alpha)
        parts.append(cache[key].loc[cache[key]["season"].eq(s)])
    return pd.concat([proxy, *parts])


def boot(df, col, block, rng_seed=SEED):
    keys = df[block].astype(str).agg("-".join, axis=1) if len(block) > 1 else df[block[0]]
    g = df.groupby(keys)[col].agg(["sum", "count"])
    sums = g["sum"].to_numpy()
    sizes = g["count"].to_numpy()
    rng = np.random.default_rng(rng_seed)
    idx = rng.integers(0, len(sums), (BOOT, len(sums)))
    draws = sums[idx].sum(axis=1) / sizes[idx].sum(axis=1)
    return {
        "low": float(np.percentile(draws, 2.5)),
        "high": float(np.percentile(draws, 97.5)),
        "pp": float((draws > 0).mean()),
        "blocks": len(sums),
    }


def compare(arm, base):
    j = arm.join(base, rsuffix="_b", how="inner")
    j["d_acc"] = 100.0 * (j["correct"] - j["correct_b"])
    j["d_ll"] = j["ll"] - j["ll_b"]
    j["d_brier"] = j["brier"] - j["brier_b"]
    eras = {
        "proxy_2011_2019": j["season"].le(2019),
        "true_2020_2025": j["season"].ge(2020),
        "pooled": j["season"].ge(0),
        "from_2018": j["season"].ge(2018),
    }
    out = {}
    for name, mask in eras.items():
        s = j.loc[mask]
        a = arm.loc[s.index]
        wins = int(a["correct"].sum())
        r = {
            "n": len(s),
            "record": f"{wins}-{len(s) - wins}",
            "flips": int((s["pick_home"] != s["pick_home_b"]).sum()),
            "acc_arm": float(a["correct"].mean()),
            "acc_base": float(s["correct_b"].mean()),
            "d_acc": float(s["d_acc"].mean()),
            "d_ll": float(s["d_ll"].mean()),
            "d_brier": float(s["d_brier"].mean()),
        }
        for blockname, block in (("season", ["season"]), ("week", ["season", "week"])):
            r[f"acc_{blockname}"] = boot(s, "d_acc", block)
            r[f"ll_{blockname}"] = boot(s, "d_ll", block)
            r[f"brier_{blockname}"] = boot(s, "d_brier", block)
        out[name] = r
    return out



from scipy.optimize import brentq, minimize
from scipy.special import expit, logit
from lead66_unit1 import atomic_counts, mass_at_theta, select_band
from nfl_ats.mass_preserving_lattice import prior_pool_for_week
from nfl_ats import features as F
from nfl_ats.constants import TEAM_ABBREVIATION_ALIASES

OUT = ROOT / "artifacts" / "mod24_u7b"
PBP = ROOT / "data" / "pbp" / "raw" / "20260929T191306Z"
OPENER = ROOT / "artifacts" / "opener_evaluation" / "20260929T192743Z"
BASE5 = ROOT / "artifacts" / "mod24_u5" / "base_per_game.parquet"
SEED = 20261001

def fit_k(z, y):
    def nll(par):
        q = par[0] * z
        return float(np.mean(np.logaddexp(0, q) - y * q))

    return float(minimize(nll, [1.0], method="BFGS").x[0])




def recal_temp(df):
    out = pd.Series(np.nan, index=df.index)
    z = logit(df["p"].clip(1e-6, 1 - 1e-6))
    for s in sorted(df["season"].unique()):
        tr = df["season"].ne(s)
        k = fit_k(z[tr].to_numpy(), df.loc[tr, "y"].to_numpy())
        m = df["season"].eq(s)
        out[m] = expit(k * z[m])
    return out


def temp_scores(df):
    pt = recal_temp(df).clip(1e-6, 1 - 1e-6)
    y = df["y"]
    return pd.DataFrame(
        {
            "ll_t": -(y * np.log(pt) + (1 - y) * np.log(1 - pt)),
            "brier_t": (pt - y) ** 2,
        },
        index=df.index,
    )


def theta_for(atoms, counts, line, target):
    home, away = atoms > line, atoms < line

    def diff(theta):
        mass = mass_at_theta(atoms, counts, theta)
        return float(mass[home].sum() / mass[home | away].sum() - target)

    if not 0 < target < 1:
        return None
    if diff(-2.0) > 0 or diff(2.0) < 0:
        return None
    return brentq(diff, -2.0, 2.0, xtol=1e-13)


def rps(atoms, mass, actual):
    grid = np.arange(min(atoms.min(), actual), max(atoms.max(), actual) + 1)
    cum = np.concatenate([[0.0], np.cumsum(mass)])
    cdf = cum[np.searchsorted(atoms, grid, side="right")]
    return float(np.square(cdf - (actual <= grid)).sum())


def lattice_rps(arms_true, temp_p, features):
    f = features[["game_id", "season", "week", "gameday", "spread_line", "result"]]
    opener = pd.read_parquet(OPENER / "per_game.parquet")
    opener = opener.loc[opener["season"].between(2020, 2025)].copy()
    line = opener.set_index("game_id")["tue_open_home_spread"]
    pool = prior_pool(f, line)
    res = f.set_index("game_id")["result"]
    games = opener.assign(gameday=pd.to_datetime(opener["game_id"].map(f.set_index("game_id")["gameday"])))
    out = {a: {} for a in arms_true}
    for (season, week), group in games.groupby(["season", "week"], sort=True):
        target = f.loc[f["season"].eq(season) & f["week"].eq(week)]
        prior = prior_pool_for_week(
            pool,
            season=int(season),
            week=int(week),
            cutoff=pd.to_datetime(target["gameday"]).min(),
            exclude_game_ids=group["game_id"],
        )
        for gid in group["game_id"]:
            ln = float(line.loc[gid])
            actual = float(res.loc[gid])
            atoms, counts = atomic_counts(select_band(prior, ln)[0])
            for a in arms_true:
                if gid not in temp_p[a].index:
                    continue
                t = theta_for(atoms, counts, ln, float(temp_p[a].loc[gid]))
                if t is not None:
                    out[a][gid] = rps(atoms, mass_at_theta(atoms, counts, t), actual)
    return {a: pd.Series(v) for a, v in out.items()}


def boot_groups(df, col, keys, seed=SEED):
    k = df[keys].astype(str).agg("-".join, axis=1) if len(keys) > 1 else df[keys[0]]
    g = df.groupby(k)[col].agg(["sum", "count"])
    s, n = g["sum"].to_numpy(), g["count"].to_numpy()
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(s), (BOOT, len(s)))
    d = s[idx].sum(1) / n[idx].sum(1)
    return {"lo": float(np.percentile(d, 2.5)), "hi": float(np.percentile(d, 97.5)), "pp": float((d > 0).mean())}


def compare5(arm, base, extra=None):
    j = arm.join(base, rsuffix="_b", how="inner")
    j["d_acc"] = 100.0 * (j["correct"] - j["correct_b"])
    j["d_ll"] = j["ll_b"] - j["ll"]
    j["d_brier"] = j["brier_b"] - j["brier"]
    j["d_ll_t"] = j["ll_t_b"] - j["ll_t"]
    j["d_brier_t"] = j["brier_t_b"] - j["brier_t"]
    if extra is not None:
        j["d_rps_t"] = extra
    out = {}
    for name, mask in (
        ("true_2020_2025", j["season"].ge(2020)),
        ("proxy_2011_2019", j["season"].le(2019)),
        ("pooled_2011_2025", j["season"].ge(0)),
    ):
        s = j.loc[mask]
        if len(s) == 0:
            continue
        w = int(s["correct"].sum())
        flips = s["pick_home"] != s["pick_home_b"]
        r = {
            "n": len(s),
            "record": f"{w}-{len(s) - w}",
            "record_base": f"{int(s['correct_b'].sum())}-{len(s) - int(s['correct_b'].sum())}",
            "flips": int(flips.sum()),
            "flips_W_L": f"{int((flips & s['correct'].eq(1.0)).sum())}-{int((flips & s['correct'].eq(0.0)).sum())}",
            "d_acc": float(s["d_acc"].mean()),
            "d_acc_season": boot_groups(s, "d_acc", ["season"]),
            "d_acc_week": boot_groups(s, "d_acc", ["season", "week"]),
            "seasons_pos": int((s.groupby("season")["d_acc"].mean() > 0).sum()),
            "seasons": int(s["season"].nunique()),
        }
        for m in ("d_ll", "d_brier", "d_ll_t", "d_brier_t"):
            r[m] = float(s[m].mean())
            r[m + "_season"] = boot_groups(s, m, ["season"])
        if "d_rps_t" in s and s["d_rps_t"].notna().any():
            sr = s.dropna(subset=["d_rps_t"])
            r["d_rps_t"] = float(sr["d_rps_t"].mean())
            r["d_rps_t_season"] = boot_groups(sr, "d_rps_t", ["season"])
        out[name] = r
    return out



ALIAS = TEAM_ABBREVIATION_ALIASES
POINT_M = ("point_diff", "ats_residual")
EPA_M = (
    "off_epa_per_play",
    "off_pass_epa_per_play",
    "off_rush_epa_per_play",
    "def_epa_per_play",
    "def_pass_epa_per_play",
    "def_rush_epa_per_play",
)
SIDES = ("home", "away", "diff")
W_GRID = (0.0, 0.25, 0.5, 0.75, 1.0)
CLEAN_COLS = [f"{s}_{m}" for m in POINT_M + EPA_M for s in SIDES]
EPA_COLS = [f"{s}_{m}" for m in EPA_M for s in SIDES]
POINT_COLS = [f"{s}_{m}" for m in POINT_M for s in SIDES]
CTX = {"adj": None, "mode": "raw"}


def src_seasons(s):
    return list(range(2009, s)) if s >= 2011 else [2009, 2010]


def pbp_adjust():
    parts = []
    for s in range(2009, 2026):
        p = pd.read_parquet(
            PBP / f"season={s}" / "plays.parquet",
            columns=[
                "game_id", "posteam", "defteam", "play_type", "down", "epa",
                "touchdown", "interception", "fumble_lost", "posteam_score", "posteam_score_post",
            ],
        )
        p["season"] = s
        parts.append(p)
    pl = pd.concat(parts, ignore_index=True)
    pl = pl.loc[pl["posteam"].notna() & pl["defteam"].notna()].copy()
    pl["posteam"] = pl["posteam"].replace(ALIAS)
    pl["defteam"] = pl["defteam"].replace(ALIAS)
    rp = pl["play_type"].isin(("pass", "run")) & pl["epa"].notna()
    pl["typ"] = np.where(pl["play_type"].eq("pass"), "pass", "run")
    pl["dn"] = pl["down"].fillna(1).clip(1, 4).astype(int)
    fl = pl["fumble_lost"].fillna(0).eq(1)
    intr = pl["interception"].fillna(0).eq(1)
    td = pl["touchdown"].fillna(0).eq(1) & (pl["posteam_score_post"] <= pl["posteam_score"])
    clean = pl.loc[rp & ~fl].groupby(["season", "typ", "dn"])["epa"].agg(["sum", "count"])
    intn = pl.loc[rp & intr & ~fl & ~td].groupby(["season", "dn"])["epa"].agg(["sum", "count"])
    pl["delta"] = 0.0
    for s in range(2009, 2026):
        src = src_seasons(s)
        cm = clean.loc[clean.index.get_level_values(0).isin(src)].groupby(level=[1, 2]).sum()
        cm = cm["sum"] / cm["count"]
        im = intn.loc[intn.index.get_level_values(0).isin(src)].groupby(level=1).sum()
        im = im["sum"] / im["count"]
        m = pl["season"].eq(s)
        key = pd.MultiIndex.from_arrays([pl.loc[m, "typ"], pl.loc[m, "dn"]])
        e_clean = pd.Series(cm.reindex(key).to_numpy(), index=pl.index[m])
        e_int = pl.loc[m, "dn"].map(im)
        a = rp[m] & fl[m]
        b = rp[m] & intr[m] & td[m] & ~fl[m]
        pl.loc[m, "delta"] = np.where(a, e_clean - pl.loc[m, "epa"], np.where(b, e_int - pl.loc[m, "epa"], 0.0))
    pl["fd"] = np.where(rp & fl, pl["delta"], 0.0)
    pl["dpass"] = np.where(pl["typ"].eq("pass"), pl["delta"], 0.0)
    pl["drush"] = np.where(pl["typ"].eq("run"), pl["delta"], 0.0)
    off = pl.groupby(["game_id", "posteam"])[["fd", "dpass", "drush"]].sum()
    off.index.names = ["game_id", "team"]
    tdp = pl.loc[td & ~(rp & fl)].groupby(["game_id", "defteam"]).size().mul(6.0).rename("tdpts")
    tdp.index.names = ["game_id", "team"]
    adj = off.join(tdp, how="outer").fillna(0.0).reset_index()
    cnt = {
        "def_td_non_fumble": int((td & ~(rp & fl)).sum()),
        "int_td": int((rp & intr & td & ~fl).sum()),
        "fumble_lost_rp": int((rp & fl).sum()),
    }
    print("adjust counts", cnt, flush=True)
    return adj


def wrapped_metrics(orig):
    def fn(schedules, team_stats, game_types=("REG",)):
        out = orig(schedules, team_stats, game_types)
        if CTX["mode"] == "raw":
            return out
        st = team_stats.copy()
        st["team"] = st["team"].replace(ALIAS)

        def num(c):
            if c in st:
                return pd.to_numeric(st[c], errors="coerce").astype(float)
            return pd.Series(0.0, index=st.index)

        pp = (num("attempts") + num("sacks_suffered").fillna(0.0)).replace(0.0, np.nan)
        rr = num("carries").replace(0.0, np.nan)
        tt = (pp.fillna(0.0) + rr.fillna(0.0)).replace(0.0, np.nan)
        dens = pd.DataFrame(
            {"game_id": st["game_id"], "team": st["team"], "pp": pp, "rr": rr, "tt": tt}
        ).drop_duplicates(["game_id", "team"])
        o = out.merge(CTX["adj"], on=["game_id", "team"], how="left").merge(
            dens, on=["game_id", "team"], how="left"
        )
        for c in ("fd", "dpass", "drush", "tdpts"):
            o[c] = o[c].fillna(0.0)
        o["d_off"] = ((o["dpass"] + o["drush"]) / o["tt"]).fillna(0.0)
        o["d_pass"] = (o["dpass"] / o["pp"]).fillna(0.0)
        o["d_rush"] = (o["drush"] / o["rr"]).fillna(0.0)
        pair = o[["game_id", "team", "fd", "tdpts", "d_off", "d_pass", "d_rush"]]
        opp = pair.rename(columns={c: c + "_o" for c in pair.columns if c != "game_id"}).rename(
            columns={"team_o": "opp"}
        )
        o = o.merge(opp, on="game_id")
        o = o.loc[o["team"].ne(o["opp"])].copy()
        padj = o["fd"] - o["fd_o"] - o["tdpts"] + o["tdpts_o"]
        o["point_diff"] = o["point_diff"] + padj
        o["ats_residual"] = o["ats_residual"] + padj
        if CTX["mode"] == "b2":
            o["off_epa_per_play"] = o["off_epa_per_play"] + o["d_off"]
            o["off_pass_epa_per_play"] = o["off_pass_epa_per_play"] + o["d_pass"]
            o["off_rush_epa_per_play"] = o["off_rush_epa_per_play"] + o["d_rush"]
            o["def_epa_per_play"] = o["def_epa_per_play"] + o["d_off_o"]
            o["def_pass_epa_per_play"] = o["def_pass_epa_per_play"] + o["d_pass_o"]
            o["def_rush_epa_per_play"] = o["def_rush_epa_per_play"] + o["d_rush_o"]
        if len(o) != len(out):
            raise RuntimeError("row mismatch")
        return o[list(out.columns)].sort_values(["gameday", "game_id", "team"])

    return fn


def prep():
    schedules = pd.read_parquet(u3.SNAPSHOT / "schedules.parquet")
    team_stats = pd.read_parquet(u3.SNAPSHOT / "team_stats.parquet")
    CTX["adj"] = pbp_adjust()
    orig = F.build_team_game_metrics
    F.build_team_game_metrics = wrapped_metrics(orig)
    res = {}
    for mode in ("raw", "b2"):
        CTX["mode"] = mode
        g = F._build_features_pass(schedules, team_stats, ("REG",))
        res[mode] = g.set_index("game_id")[CLEAN_COLS]
        print("built", mode, len(g), flush=True)
    feats = pd.read_parquet(u3.FEATURES).set_index("game_id")
    common = feats.index.intersection(res["raw"].index)
    d = (res["raw"].loc[common] - feats.loc[common, CLEAN_COLS]).abs()
    print("raw repro max abs diff", float(d.max().max()), "common", len(common), "of", len(feats), flush=True)
    print("nan mismatch", int((res["raw"].loc[common].isna() != feats.loc[common, CLEAN_COLS].isna()).sum().sum()))
    delta = (res["b2"].loc[common] - res["raw"].loc[common]).astype(float)
    delta.to_parquet(OUT / "clean_minus_raw.parquet")
    print({k: round(v, 5) for k, v in delta.abs().mean().to_dict().items()})
    print({c: round(float(feats.loc[common, c].std()), 5) for c in POINT_COLS + EPA_COLS[:3]})


def variant(features, delta, cols, w):
    f = features.copy()
    idx = f["game_id"].to_numpy()
    for c in cols:
        d = pd.Series(delta[c]).reindex(idx).fillna(0.0).to_numpy()
        f[c] = f[c] + w * d
    return f


def select_w(features, delta):
    sel = {}
    for s in PROXY_SEASONS + TRUE_SEASONS:
        inner = list(range(max(FIRST_INNER, s - INNER_SEASONS), s))
        sc = {}
        for w in W_GRID:
            fr = regular_season_rows(variant(features, delta, CLEAN_COLS, w))
            fr = fr.loc[fr["ats_margin"].notna() & fr["result"].notna()]
            sc[w] = inner_mse(fr, FULL, BASE_ALPHA, inner)
        best = min(W_GRID, key=lambda w: (sc[w], w))
        sel[str(s)] = {"w": best, "inner_mse": {str(k): v for k, v in sc.items()}}
        print("w", s, best, {k: round(v, 4) for k, v in sc.items()}, flush=True)
    return sel


def finish(df, p=None):
    df = df.copy()
    if p is not None:
        df["p"] = np.asarray(p)
    df["pick_home"] = df["p"].to_numpy() >= 0.5
    m = df["margin_vs_open"]
    df["correct"] = np.where(m.eq(0), np.nan, np.where(df["pick_home"], m.gt(0), ~m.gt(0)))
    df = df.loc[df["correct"].notna()].copy()
    df["y"] = (df["margin_vs_open"] > 0).astype(float)
    pc = df["p"].clip(1e-6, 1 - 1e-6)
    df["ll"] = -(df["y"] * np.log(pc) + (1 - df["y"]) * np.log(1 - pc))
    df["brier"] = (pc - df["y"]) ** 2
    return df.set_index("game_id")[["season", "week", "p", "pick_home", "correct", "y", "ll", "brier"]]


def run_variant(features, pop, tag):
    path = OUT / f"{tag}_run.parquet"
    if path.exists():
        return pd.read_parquet(path)
    df = run_arm(features, pop, lambda s: (FULL, BASE_ALPHA))
    df.to_parquet(path)
    print("ran", tag, int(df["correct"].sum()), len(df), flush=True)
    return df


def grade():
    features = pd.read_parquet(u3.FEATURES)
    delta = pd.read_parquet(OUT / "clean_minus_raw.parquet")
    pop = proxy_population(features)
    base = pd.read_parquet(BASE5)
    tr = base.loc[base["season"].ge(2020)]
    print("base", int(tr["correct"].sum()), len(tr), int(base.loc[base["season"].le(2019), "correct"].sum()), flush=True)
    runs = {}
    runs["b1"] = run_variant(variant(features, delta, POINT_COLS, 1.0), pop, "b1")
    runs["b2"] = run_variant(variant(features, delta, CLEAN_COLS, 1.0), pop, "b2")
    selpath = OUT / "b3_selection.json"
    if selpath.exists():
        sel = json.loads(selpath.read_text())
    else:
        sel = select_w(features, delta)
        selpath.write_text(json.dumps(sel, indent=2))
    runs_w = {1.0: runs["b2"], 0.0: base}
    for w in sorted({v["w"] for v in sel.values()}):
        if w not in runs_w:
            runs_w[w] = run_variant(variant(features, delta, CLEAN_COLS, w), pop, f"b3_w{w}")
    parts = []
    for s in PROXY_SEASONS + TRUE_SEASONS:
        r = runs_w[sel[str(s)]["w"]]
        parts.append(r.loc[r["season"].eq(s)])
    runs["b3"] = pd.concat(parts)
    runs["b3"].to_parquet(OUT / "b3_per_game.parquet")
    per_arm = {"base": base, **runs}
    for arm in per_arm:
        per_arm[arm] = per_arm[arm].loc[per_arm[arm].index.isin(base.index)]
    for arm, df in per_arm.items():
        per_arm[arm] = df.join(
            pd.concat([temp_scores(df.loc[df["season"].le(2019)]), temp_scores(df.loc[df["season"].ge(2020)])])
        )
    p_t = {a: recal_temp(df.loc[df["season"].ge(2020)]) for a, df in per_arm.items()}
    rps_by = lattice_rps(list(per_arm), p_t, features)
    brps = rps_by["base"]
    results = {}
    for arm in ("b1", "b2", "b3"):
        extra = (brps - rps_by[arm]).reindex(per_arm[arm].index)
        results[arm] = compare5(per_arm[arm], per_arm["base"], extra)
        results[arm]["arm_rps_t"] = float(rps_by[arm].mean())
        print(arm, results[arm]["pooled_2011_2025"]["record"], results[arm]["pooled_2011_2025"]["d_acc"], flush=True)
    results["base_rps_t"] = float(brps.mean())
    (OUT / "report.json").write_text(json.dumps(results, indent=2, default=str))
    print(json.dumps(results, indent=1, default=str))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    stage = sys.argv[1] if len(sys.argv) > 1 else "grade"
    if stage == "prep":
        prep()
    else:
        grade()


main()
