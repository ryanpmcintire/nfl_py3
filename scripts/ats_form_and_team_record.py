from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from team_mispricing import openers, schedules

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts/ats_form"
PER_GAME = ROOT / "artifacts/opener_evaluation/20261122T171458Z/per_game.parquet"
SCREEN_SEASONS = (2011, 2025)
HALVES = ((2011, 2018), (2019, 2025))
MODEL_HALVES = ((2020, 2022), (2023, 2025))
EARLY_WEEKS = (2, 8)
SHUFFLES = 200
BOOTSTRAPS = 2000
SEED = 20261123
FRANCHISE = {"OAK": "LV", "SD": "LAC", "STL": "LA"}


def slope(y, x):
    xc = x - x.mean()
    return float((xc * (y - y.mean())).sum() / (xc * xc).sum())


def groups_of(seasons):
    return [np.flatnonzero(seasons == s) for s in np.unique(seasons)]


def shuffle_within(x, groups, rng):
    out = x.copy()
    for idx in groups:
        out[idx] = rng.permutation(out[idx])
    return out


def season_boot(seasons, rng):
    uniq = np.unique(seasons)
    pick = rng.choice(uniq, size=len(uniq), replace=True)
    return np.concatenate([np.flatnonzero(seasons == s) for s in pick])


def team_long(games, line_col):
    rows = []
    for side, sign in (("home", 1.0), ("away", -1.0)):
        margin = sign * (games["result"].astype(float) - games[line_col].astype(float))
        rows.append(
            pd.DataFrame(
                {
                    "game_id": games["game_id"].astype(str),
                    "season": games["season"].astype(int),
                    "week": games["week"].astype(int),
                    "team": games[f"{side}_team"],
                    "side": side,
                    "ats_margin": margin,
                }
            )
        )
    long = pd.concat(rows, ignore_index=True).sort_values(["team", "season", "week"])
    return long


def prior_form(long):
    out = []
    for _, grp in long.groupby(["team", "season"], sort=False):
        covers = 0
        decided = 0
        margins = []
        streak = 0
        recs = []
        for m in grp["ats_margin"].to_numpy(float):
            recs.append(
                (
                    covers / decided if decided else np.nan,
                    float(streak),
                    float(np.mean(margins)) if margins else np.nan,
                    decided,
                    margins[-1] if margins else np.nan,
                )
            )
            if np.isnan(m):
                continue
            margins.append(m)
            if m > 0:
                covers += 1
                decided += 1
                streak = streak + 1 if streak > 0 else 1
            elif m < 0:
                decided += 1
                streak = streak - 1 if streak < 0 else -1
        part = grp.copy()
        arr = np.array(recs, dtype=float)
        part["ats_pct"] = arr[:, 0]
        part["ats_streak"] = arr[:, 1]
        part["ats_mean_margin"] = arr[:, 2]
        part["ats_decided"] = arr[:, 3]
        part["prev_margin"] = arr[:, 4]
        out.append(part)
    return pd.concat(out, ignore_index=True)


def build_games():
    table, snap, raw = schedules()
    lo, hi = SCREEN_SEASONS
    games = table.loc[table["season"].between(lo, hi)].dropna(subset=["result"]).copy()
    games["game_id"] = games["game_id"].astype(str)
    spread, sources = openers(raw)
    games["opener"] = spread.reindex(games["game_id"]).to_numpy(float)
    games["form_line"] = games["opener"].combine_first(games["spread_line"].astype(float))
    long = prior_form(team_long(games, "form_line"))
    wide = games[["game_id", "season", "week", "home_team", "away_team", "result", "opener"]]
    for side in ("home", "away"):
        part = long.loc[long["side"].eq(side)].set_index("game_id")
        for col in ("ats_pct", "ats_streak", "ats_mean_margin", "ats_decided", "prev_margin"):
            wide = wide.assign(**{f"{side}_{col}": part[col].reindex(wide["game_id"]).to_numpy()})
    wide = wide.copy()
    wide["ats_pct_diff"] = wide["home_ats_pct"] - wide["away_ats_pct"]
    wide["ats_streak_diff"] = wide["home_ats_streak"] - wide["away_ats_streak"]
    wide["ats_margin_diff"] = wide["home_ats_mean_margin"] - wide["away_ats_mean_margin"]
    wide["home_ats_pct_c"] = wide["home_ats_pct"] - 0.5
    margin = wide["result"] - wide["opener"]
    wide = wide.loc[wide["opener"].notna() & margin.ne(0)].copy()
    wide["home_cover"] = (wide["result"] - wide["opener"] > 0).astype(float)
    meta = {"schedule_snapshot": snap, "opener_sources": sources, "graded_games": len(wide)}
    return wide, long, meta


def screen_term(frame, term, rng, weeks=None):
    data = frame.dropna(subset=[term])
    if weeks is not None:
        data = data.loc[data["week"].between(*weeks)]
    raw = data[term].to_numpy(float)
    x = (raw - raw.mean()) / raw.std(ddof=0)
    y = data["home_cover"].to_numpy(float)
    seasons = data["season"].to_numpy(int)
    grp = groups_of(seasons)
    obs = slope(y, x) * 100.0
    null = np.array([slope(y, shuffle_within(x, grp, rng)) * 100.0 for _ in range(SHUFFLES)])
    boots = np.array(
        [slope(y[i], x[i]) * 100.0 for i in (season_boot(seasons, rng) for _ in range(BOOTSTRAPS))]
    )
    halves = {
        f"{a}-{b}": slope(y[(seasons >= a) & (seasons <= b)], x[(seasons >= a) & (seasons <= b)])
        * 100.0
        for a, b in HALVES
    }
    per_season = {
        int(s): slope(y[seasons == s], x[seasons == s]) * 100.0 for s in np.unique(seasons)
    }
    decisive = raw != 0
    follow_home = raw > 0
    wins = int(((follow_home & (y == 1)) | (~follow_home & (y == 0)))[decisive].sum())
    return {
        "n": len(data),
        "effect_follow_form_acc_points_per_sd": obs,
        "interval_95": [float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))],
        "probability_positive": float((boots > 0).mean()),
        "null_percentile": float((null < obs).mean() * 100.0),
        "null_p95": float(np.quantile(null, 0.95)),
        "null_p05": float(np.quantile(null, 0.05)),
        "halves": halves,
        "seasons_positive": int(sum(v > 0 for v in per_season.values())),
        "seasons": len(per_season),
        "follow_form_record": [wins, int(decisive.sum()) - wins],
    }


def lag_autocorr(long, rng):
    data = long.dropna(subset=["ats_margin", "prev_margin"])
    y = data["ats_margin"].to_numpy(float)
    x = data["prev_margin"].to_numpy(float)
    seasons = data["season"].to_numpy(int)
    grp = groups_of(seasons)
    r = float(np.corrcoef(x, y)[0, 1])
    null = np.array([np.corrcoef(shuffle_within(x, grp, rng), y)[0, 1] for _ in range(SHUFFLES)])
    boots = []
    for _ in range(BOOTSTRAPS):
        i = season_boot(seasons, rng)
        boots.append(np.corrcoef(x[i], y[i])[0, 1])
    boots = np.array(boots)
    cum = long.dropna(subset=["ats_margin", "ats_mean_margin"])
    r_cum = float(np.corrcoef(cum["ats_mean_margin"], cum["ats_margin"])[0, 1])
    b_cum = slope(cum["ats_margin"].to_numpy(float), cum["ats_mean_margin"].to_numpy(float))
    return {
        "pairs": len(data),
        "lag1_r": r,
        "interval_95": [float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))],
        "probability_positive": float((boots > 0).mean()),
        "null_percentile": float((null < r).mean() * 100.0),
        "season_to_date_r": r_cum,
        "season_to_date_slope_points_per_point": b_cum,
        "season_to_date_pairs": len(cum),
    }


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def fit_logistic(X, y, iters=50):
    w = np.zeros(X.shape[1])
    for _ in range(iters):
        p = 1.0 / (1.0 + np.exp(-(X @ w)))
        g = X.T @ (y - p)
        h = (X * (p * (1 - p))[:, None]).T @ X
        w = w + np.linalg.solve(h + 1e-9 * np.eye(len(w)), g)
    return w


def loso(base, term, y, seasons):
    pred = np.empty_like(y)
    coefs = {}
    for s in np.unique(seasons):
        tr = seasons != s
        cols = [np.ones(len(y)), base] + ([term] if term is not None else [])
        X = np.column_stack(cols)
        w = fit_logistic(X[tr], y[tr])
        pred[~tr] = 1.0 / (1.0 + np.exp(-(X[~tr] @ w)))
        coefs[int(s)] = [float(v) for v in w]
    return pred, coefs


def score(pred, y):
    pick = pred > 0.5
    correct = (pick & (y == 1)) | (~pick & (y == 0))
    ll = -np.mean(y * np.log(pred) + (1 - y) * np.log(1 - pred))
    return correct.astype(float), float(ll)


def model_frame():
    pg = pd.read_parquet(PER_GAME)
    pg = pg.loc[pg["correct_at_open_probability_rule"].notna()].copy()
    parts = pg["game_id"].str.split("_", expand=True)
    pg["away_team"] = parts[2].replace(FRANCHISE)
    pg["home_team"] = parts[3].replace(FRANCHISE)
    cover = pg["home_cover_probability_at_open"]
    loss = pg["home_loss_probability_at_open"]
    pg["p_model"] = cover / (cover + loss)
    pg["home_cover"] = (pg["margin_vs_open"] > 0).astype(float)
    pg["pick_home"] = pg["pick_home_at_open_probability_rule"].astype(bool)
    pg["correct"] = pg["correct_at_open_probability_rule"].astype(float)
    pg["picked"] = np.where(pg["pick_home"], pg["home_team"], pg["away_team"])
    pg["faded"] = np.where(pg["pick_home"], pg["away_team"], pg["home_team"])
    pg["err_home"] = pg["margin_vs_open"] - pg["residual_at_open"]
    return pg.sort_values(["season", "week", "game_id"]).reset_index(drop=True)


def model_bias_term(pg):
    hist: dict[str, list[float]] = {}
    picked_hist: dict[str, list[float]] = {}
    overall: list[float] = []
    bias_diff = []
    toward_home = []
    picked_rate = []
    for _, grp in pg.groupby(["season", "week"], sort=True):
        center = float(np.mean(overall)) if overall else np.nan
        for _, row in grp.iterrows():
            h = hist.get(row["home_team"], [])
            a = hist.get(row["away_team"], [])
            bias_diff.append((np.mean(h) if h else 0.0) - (np.mean(a) if a else 0.0))
            ph = picked_hist.get(row["picked"], [])
            rate = float(np.mean(ph)) if ph else np.nan
            picked_rate.append(rate)
            dev = rate - center if ph and overall else 0.0
            toward_home.append(dev if row["pick_home"] else -dev)
        for _, row in grp.iterrows():
            hist.setdefault(row["home_team"], []).append(row["err_home"])
            hist.setdefault(row["away_team"], []).append(-row["err_home"])
            picked_hist.setdefault(row["picked"], []).append(row["correct"])
            overall.append(row["correct"])
    out = pg.copy()
    out["model_team_bias_diff"] = bias_diff
    out["model_pick_rate_toward_home"] = toward_home
    out["picked_team_prior_rate"] = picked_rate
    return out


def on_model(pg, terms, rng):
    y = pg["home_cover"].to_numpy(float)
    seasons = pg["season"].to_numpy(int)
    base = logit(pg["p_model"].to_numpy(float))
    raw_correct = pg["correct"].to_numpy(float)
    base_pred, _ = loso(base, None, y, seasons)
    base_correct, base_ll = score(base_pred, y)
    res = {
        "games": len(y),
        "served_rule_record": [int(raw_correct.sum()), int(len(y) - raw_correct.sum())],
        "loso_base_record": [int(base_correct.sum()), int(len(y) - base_correct.sum())],
        "loso_base_log_loss": base_ll,
        "base_pick_matches_rule": float(((base > 0) == pg["pick_home"].to_numpy()).mean()),
        "terms": {},
    }
    grp = groups_of(seasons)
    for term in terms:
        raw = pg[term].to_numpy(float)
        sd = raw.std(ddof=0)
        x = (raw - raw.mean()) / sd if sd > 0 else raw
        pred, coefs = loso(base, x, y, seasons)
        corr, ll = score(pred, y)
        gain = (corr.mean() - base_correct.mean()) * 100.0
        null = []
        for _ in range(SHUFFLES):
            xs = shuffle_within(x, grp, rng)
            pn, _ = loso(base, xs, y, seasons)
            cn, _ = score(pn, y)
            null.append((cn.mean() - base_correct.mean()) * 100.0)
        null = np.array(null)
        diff = corr - base_correct
        boots = np.array(
            [diff[season_boot(seasons, rng)].mean() * 100.0 for _ in range(BOOTSTRAPS)]
        )
        halves = {
            f"{a}-{b}": float(diff[(seasons >= a) & (seasons <= b)].mean() * 100.0)
            for a, b in MODEL_HALVES
        }
        changed = (pred > 0.5) != (base_pred > 0.5)
        res["terms"][term] = {
            "record": [int(corr.sum()), int(len(y) - corr.sum())],
            "gain_acc_points_vs_loso_base": gain,
            "interval_95": [float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))],
            "probability_positive": float((boots > 0).mean()),
            "null_percentile": float((null < gain).mean() * 100.0),
            "log_loss": ll,
            "log_loss_base": base_ll,
            "halves": halves,
            "changed_sides": int(changed.sum()),
            "changed_record_term": [
                int(corr[changed].sum()),
                int(changed.sum() - corr[changed].sum()),
            ],
            "term_coef_per_fold": {k: v[2] for k, v in coefs.items()},
        }
    return res


def team_breakdown(pg, rng):
    picked = pg.groupby("picked")["correct"].agg(["sum", "count"])
    faded = pg.groupby("faded")["correct"].agg(["sum", "count"])
    teams = sorted(set(pg["home_team"]) | set(pg["away_team"]))
    rows = []
    for t in teams:
        pw, pn = (picked.loc[t, "sum"], picked.loc[t, "count"]) if t in picked.index else (0, 0)
        fw, fn = (faded.loc[t, "sum"], faded.loc[t, "count"]) if t in faded.index else (0, 0)
        rows.append(
            {
                "team": t,
                "picked_w": int(pw),
                "picked_l": int(pn - pw),
                "picked_pct": pw / pn if pn else np.nan,
                "against_w": int(fw),
                "against_l": int(fn - fw),
                "against_pct": fw / fn if fn else np.nan,
                "all_w": int(pw + fw),
                "all_l": int(pn + fn - pw - fw),
            }
        )
    table = pd.DataFrame(rows).sort_values("picked_pct")
    corr = pg["correct"].to_numpy(float)
    team_codes = pd.Categorical(pg["picked"], categories=teams).codes
    seasons = pg["season"].to_numpy(int)
    grp = groups_of(seasons)

    def spread_stat(c):
        w = np.bincount(team_codes, weights=c, minlength=len(teams))
        n = np.bincount(team_codes, minlength=len(teams))
        p0 = c.mean()
        return float((((w - n * p0) ** 2) / np.maximum(n * p0 * (1 - p0), 1e-9)).sum())

    obs = spread_stat(corr)
    null = np.array([spread_stat(shuffle_within(corr, grp, rng)) for _ in range(2000)])

    def half_rates(lo, hi):
        m = (seasons >= lo) & (seasons <= hi)
        w = np.bincount(team_codes[m], weights=corr[m], minlength=len(teams))
        n = np.bincount(team_codes[m], minlength=len(teams))
        return w / np.maximum(n, 1), n

    def half_corr(c):
        rates = []
        for lo, hi in MODEL_HALVES:
            m = (seasons >= lo) & (seasons <= hi)
            w = np.bincount(team_codes[m], weights=c[m], minlength=len(teams))
            n = np.bincount(team_codes[m], minlength=len(teams))
            rates.append(w / np.maximum(n, 1))
        return float(np.corrcoef(rates[0], rates[1])[0, 1])

    r_half = half_corr(corr)
    null_half = np.array([half_corr(shuffle_within(corr, grp, rng)) for _ in range(2000)])
    season_pairs = []
    for s in range(2020, 2025):
        a = pg.loc[pg["season"].eq(s)].groupby("picked")["correct"].mean()
        b = pg.loc[pg["season"].eq(s + 1)].groupby("picked")["correct"].mean()
        both = a.index.intersection(b.index)
        season_pairs.append(float(np.corrcoef(a[both], b[both])[0, 1]))
    worst = table.head(8)["team"].tolist()
    best = table.tail(8)["team"].tolist()
    r1, n1 = half_rates(*MODEL_HALVES[0])
    r2, n2 = half_rates(*MODEL_HALVES[1])
    halves_tbl = pd.DataFrame(
        {"team": teams, "rate_2020_22": r1, "n_2020_22": n1, "rate_2023_25": r2, "n_2023_25": n2}
    )
    return (
        table,
        {
            "overall_rate": float(corr.mean()),
            "chi_square_obs": obs,
            "chi_square_null_mean": float(null.mean()),
            "chi_square_null_percentile": float((null < obs).mean() * 100.0),
            "split_half_r": r_half,
            "split_half_null_p95": float(np.quantile(null_half, 0.95)),
            "split_half_null_percentile": float((null_half < r_half).mean() * 100.0),
            "season_to_next_r": season_pairs,
            "worst8": worst,
            "best8": best,
        },
        halves_tbl,
    )


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    wide, long, meta = build_games()
    screen = {}
    for term in ("ats_pct_diff", "ats_streak_diff", "ats_margin_diff", "home_ats_pct_c"):
        screen[term] = screen_term(wide, term, rng)
    screen["ats_margin_diff_weeks_2_8"] = screen_term(wide, "ats_margin_diff", rng, EARLY_WEEKS)
    screen["ats_margin_diff_weeks_9_plus"] = screen_term(
        wide, "ats_margin_diff", rng, (EARLY_WEEKS[1] + 1, 22)
    )
    lag = lag_autocorr(long.loc[long["season"].between(*SCREEN_SEASONS)], rng)
    pg = model_bias_term(model_frame())
    pg = pg.merge(
        wide[["game_id", "ats_pct_diff", "ats_streak_diff", "ats_margin_diff", "home_ats_pct_c"]],
        on="game_id",
        how="left",
    )
    for col in ("ats_pct_diff", "ats_streak_diff", "ats_margin_diff", "home_ats_pct_c"):
        pg[col] = pg[col].fillna(0.0)
    model = on_model(
        pg,
        (
            "ats_pct_diff",
            "ats_streak_diff",
            "ats_margin_diff",
            "home_ats_pct_c",
            "model_team_bias_diff",
            "model_pick_rate_toward_home",
        ),
        rng,
    )
    fade_rule = pg["picked_team_prior_rate"] < 0.5
    decided = pg["picked_team_prior_rate"].notna() & pg["picked_team_prior_rate"].ne(0.5)
    flip_correct = np.where(fade_rule, 1.0 - pg["correct"], pg["correct"])
    model["fade_when_picked_team_below_500_rule"] = {
        "record": [
            int(flip_correct[decided].sum()),
            int(decided.sum() - flip_correct[decided].sum()),
        ],
        "base_on_same_games": [
            int(pg.loc[decided, "correct"].sum()),
            int(decided.sum() - pg.loc[decided, "correct"].sum()),
        ],
        "flipped_games": int((fade_rule & decided).sum()),
    }
    table, team_stats, halves_tbl = team_breakdown(pg, rng)
    table.to_csv(OUT / "team_pick_record_2020_2025.csv", index=False)
    halves_tbl.to_csv(OUT / "team_pick_record_halves.csv", index=False)
    summary = {
        "meta": meta,
        "screen_vs_opener_2011_2025": screen,
        "market_lag_autocorrelation": lag,
        "on_model_2020_2025": model,
        "team_breakdown": team_stats,
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, default=float))
    print(json.dumps(summary, indent=2, default=float))
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
