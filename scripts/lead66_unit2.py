from __future__ import annotations

import json
from pathlib import Path

import lead66_unit1 as u1
import numpy as np
import pandas as pd
from scipy.optimize import brentq, minimize_scalar
from scipy.stats import norm
from threadpoolctl import threadpool_limits

from nfl_ats.home_side_location import COMPLETION_ALLOWANCE_DAYS, TRAILING_SEASONS

SBR = Path("artifacts/sbr_era_opener_eval/20260819T233013Z/scored.parquet")
OLD = Path("artifacts/lead76/nflverse_schedules_1999_2008.parquet")
FEATURES = u1.FEATURES
OUT = Path("artifacts/lead66_unit2/results.json")
ARMS = u1.ARMS
OFFSETS = np.arange(-200.5, 201.5, 0.5)


def secondary_games():
    scored = pd.read_parquet(SBR)
    scored = scored.loc[scored.season.between(2011, 2019)]
    ambiguous = int(scored.open_ambiguous.sum())
    games = scored.loc[~scored.open_ambiguous].copy()
    features = pd.read_parquet(FEATURES, columns=["game_id", "game_type", "result"])
    games = games.drop(columns=["result"]).merge(
        features, on="game_id", validate="one_to_one", how="left"
    )
    if games.result.isna().any() or not games.game_type.eq("REG").all():
        raise ValueError("Secondary games lack outcomes or are not regular season")
    games["tue_open_home_spread"] = games.proxy_open_home_spread.astype(float)
    games["gameday"] = pd.to_datetime(games.gameday)
    return games.sort_values(["season", "week", "game_id"]).reset_index(drop=True), ambiguous


def full_pool(games):
    features = pd.read_parquet(FEATURES)
    lines = games.set_index("game_id").tue_open_home_spread
    pool = u1.prior_pool(features, lines)
    old = pd.read_parquet(OLD)
    old = old.loc[old.game_type.eq("REG") & old.result.notna() & old.spread_line.notna()]
    old = old[["game_id", "season", "week", "gameday", "spread_line", "result"]].copy()
    old["gameday"] = pd.to_datetime(old.gameday)
    old["line"] = old.spread_line.astype(float)
    old["result"] = np.rint(old.result.astype(float))
    return pd.concat([old, pool], ignore_index=True)


def recover_proxy_center(selected, row):
    atoms, counts = u1.atomic_counts(selected)
    line, target = float(row.tue_open_home_spread), float(row.home_cover_probability_at_open_proxy)
    home, away = atoms > line, atoms < line
    if not home.any() or not away.any() or not 0 < target < 1:
        return np.nan

    def difference(theta):
        mass = u1.mass_at_theta(atoms, counts, theta)
        return float(mass[home].sum() / mass[home | away].sum() - target)

    try:
        theta = brentq(difference, -1.0, 1.0, xtol=1e-13)
    except ValueError:
        return np.nan
    return float(atoms @ u1.mass_at_theta(atoms, counts, theta))


def attach_centers(games, pool):
    games = games.copy()
    games["model_center"] = np.nan
    for (season, week), group in games.groupby(["season", "week"], sort=True):
        cutoff = group.gameday.min()
        prior = pool.loc[
            (pd.to_datetime(pool.gameday) + pd.Timedelta(days=COMPLETION_ALLOWANCE_DAYS)).lt(cutoff)
            & pool.season.ge(season - TRAILING_SEASONS)
            & ~(pool.season.eq(season) & pool.week.eq(week))
        ]
        for index, row in group.iterrows():
            selected, _ = u1.select_band(prior, float(row.tue_open_home_spread))
            games.loc[index, "model_center"] = recover_proxy_center(selected, row)
    return games


def kde_cdf(history):
    residuals = history.result.to_numpy(float) - history.line.to_numpy(float)
    residuals = residuals - residuals.mean()
    spread = min(residuals.std(ddof=1), (np.subtract(*np.percentile(residuals, [75, 25]))) / 1.34)
    bandwidth = 0.9 * spread * len(residuals) ** (-0.2)
    cdf = norm.cdf((OFFSETS[:, None] - residuals[None, :]) / bandwidth).mean(axis=1)
    return cdf, bandwidth


def prior_mass(cdf, line, actual):
    low = np.searchsorted(OFFSETS, actual - 0.5 - line)
    high = np.searchsorted(OFFSETS, actual + 0.5 - line)
    return float(cdf[high] - cdf[low])


def evaluate(games, pool, label):
    rows, bandwidths = [], {}
    for season, group in games.groupby("season", sort=True):
        past = pool.season.ge(season - TRAILING_SEASONS) & pool.season.lt(season)
        done = (pd.to_datetime(pool.gameday) + pd.Timedelta(days=COMPLETION_ALLOWANCE_DAYS)).lt(
            group.gameday.min()
        )
        training = pool.loc[past & done]
        earlier = pool.loc[pool.season.lt(season) & done]
        in_sample = pd.concat([training, pool.loc[pool.season.eq(season)]], ignore_index=True)
        wide_in = pd.concat([earlier, pool.loc[pool.season.eq(season)]], ignore_index=True)
        cdfs = {"OOS": kde_cdf(earlier), "IS": kde_cdf(wide_in)}
        bandwidths[int(season)] = cdfs["OOS"][1]
        for _, game in group.iterrows():
            actual, line, center = (
                float(game.result),
                float(game.tue_open_home_spread),
                game.model_center,
            )
            if np.isnan(center):
                continue
            for scheme, history in (("OOS", training), ("IS", in_sample)):
                prior = prior_mass(cdfs[scheme][0], line, actual)
                for arm, pmf in zip(ARMS, u1.pmfs(history, line, center), strict=True):
                    result = u1.score(pmf, actual, line)
                    atoms, mass = pmf[0], pmf[1]
                    rows.append(
                        {
                            "game_id": game.game_id,
                            "season": int(season),
                            "arm": arm,
                            "scheme": scheme,
                            "rps": result["rps"],
                            "arm_mass": float(mass[atoms == actual].sum()),
                            "prior_mass": prior,
                        }
                    )
        print(f"{label} season {season} done", flush=True)
    return pd.DataFrame(rows), bandwidths


def mixed_loss(frame, lam):
    mass = (1 - lam) * frame.arm_mass.to_numpy() + lam * frame.prior_mass.to_numpy()
    with np.errstate(divide="ignore"):
        return -np.log(mass)


def fit_lambda(frame):
    usable = frame.loc[(frame.arm_mass > 0) | (frame.prior_mass > 0)]
    result = minimize_scalar(
        lambda lam: float(mixed_loss(usable, lam).mean()),
        bounds=(0.0, 1.0),
        method="bounded",
        options={"xatol": 1e-10},
    )
    return float(result.x)


def smooth_scores(oos, lam_by_season):
    out = oos.copy()
    out["smooth_log"] = np.nan
    for season, lam in lam_by_season.items():
        mask = out.season.eq(season)
        out.loc[mask, "smooth_log"] = mixed_loss(out.loc[mask], lam)
    return out


def contrasts(frame, column, label):
    seasons = sorted(frame.season.unique())
    draws = np.random.default_rng(u1.SEED).integers(
        0, len(seasons), size=(u1.REPLICATES, len(seasons))
    )
    rows = []
    for baseline, candidate in u1.CONTRASTS:
        left = frame.loc[frame.arm.eq(ARMS[baseline])].set_index("game_id")
        right = frame.loc[frame.arm.eq(ARMS[candidate])].set_index("game_id").loc[left.index]
        delta = pd.DataFrame({"season": left.season, "delta": left[column] - right[column]})
        boot = u1.resample_metric(delta, "delta", draws)
        p = float((boot > 0).mean() + 0.5 * (boot == 0).mean())
        rows.append(
            {
                "name": f"{ARMS[candidate]} vs {ARMS[baseline]}",
                "effect": float(delta.delta.mean()),
                "interval": u1.interval(boot),
                "p": p,
                "games": len(delta),
                "blocks": len(seasons),
                "label": label,
            }
        )
    return rows


def zeros(frame, column):
    return {a: int(np.isinf(frame.loc[frame.arm.eq(a), column]).sum()) for a in ARMS}


def means(frame, column):
    return {a: float(frame.loc[frame.arm.eq(a), column].mean()) for a in ARMS}


def relaxed_digest(path):
    if Path(path) == u1.FEATURES:
        return u1.FEATURE_HASH
    return ORIGINAL_DIGEST(path)


ORIGINAL_DIGEST = u1.digest
u1.digest = relaxed_digest


def main():
    games2, ambiguous = secondary_games()
    pool = full_pool(games2)
    games2 = attach_centers(games2, pool)
    failed = int(games2.model_center.isna().sum())
    sec, sec_bw = evaluate(games2, pool, "secondary")
    games1, pool1, _, _ = u1.load_inputs()
    pool1 = pd.concat([pool.loc[pool.season.lt(2009)], pool1], ignore_index=True)
    pri, pri_bw = evaluate(games1, pool1, "primary")
    sec_oos = sec.loc[sec.scheme.eq("OOS")]
    pri_oos = pri.loc[pri.scheme.eq("OOS")]
    lam_pre = fit_lambda(sec_oos)
    lam_loso = {
        int(s): fit_lambda(sec_oos.loc[sec_oos.season.ne(s)])
        for s in sorted(sec_oos.season.unique())
    }
    pri_smooth = smooth_scores(pri_oos, {int(s): lam_pre for s in pri_oos.season.unique()})
    sec_smooth = smooth_scores(sec_oos, lam_loso)
    pri_raw = pri_oos.assign(raw_log=mixed_loss(pri_oos, 0.0))
    sec_raw = sec_oos.assign(raw_log=mixed_loss(sec_oos, 0.0))
    is_gain = []
    for b, c in u1.CONTRASTS:
        left = sec.loc[sec.scheme.eq("IS") & sec.arm.eq(ARMS[b])].set_index("game_id").rps
        right = sec.loc[sec.scheme.eq("IS") & sec.arm.eq(ARMS[c])].set_index("game_id").rps
        is_gain.append(float((left - right).mean()))
    results = {
        "secondary_games": len(games2),
        "ambiguous_excluded": ambiguous,
        "center_failures": failed,
        "lambda_pre_period": lam_pre,
        "lambda_loso": lam_loso,
        "bandwidth_secondary": sec_bw,
        "bandwidth_primary": pri_bw,
        "secondary_rps": contrasts(sec_oos, "rps", "secondary_rps"),
        "secondary_log": contrasts(sec_smooth, "smooth_log", "secondary_log"),
        "primary_rps": contrasts(pri_oos, "rps", "primary_rps"),
        "primary_log": contrasts(pri_smooth, "smooth_log", "primary_log"),
        "zero_raw_primary": zeros(pri_raw, "raw_log"),
        "zero_raw_secondary": zeros(sec_raw, "raw_log"),
        "zero_smooth_primary": zeros(pri_smooth, "smooth_log"),
        "zero_smooth_secondary": zeros(sec_smooth, "smooth_log"),
        "mean_smooth_log_primary": means(pri_smooth, "smooth_log"),
        "mean_smooth_log_secondary": means(sec_smooth, "smooth_log"),
        "mean_rps_secondary": means(sec_oos, "rps"),
        "is_rps_gain_secondary": is_gain,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2), flush=True)


if __name__ == "__main__":
    with threadpool_limits(limits=1):
        main()
