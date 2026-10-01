from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import brentq

sys.path.insert(0, str(Path(__file__).parent))
from lead66_unit1 import FEATURES, OPENER, mass_at_theta, select_band, atomic_counts
from nfl_ats.mass_preserving_lattice import prior_pool, prior_pool_for_week

OUT = Path("artifacts/mod24_u2b")
SEED = 20261001
DRAWS = 10_000
ARMS = {
    "u1_alpha_only": "mod23_unit1/alpha_only",
    "u1_greedy": "mod23_unit1/greedy",
    "u2_divergence": "mod23_unit2/divergence",
    "u2_full_strength": "mod23_unit2/full_strength",
    "u2_lineup_and_divergence": "mod23_unit2/lineup_and_divergence",
    "u2_lineup_total": "mod23_unit2/lineup_total",
    "u2_qb_out": "mod23_unit2/qb_out",
    "u2_qb_quality_loss": "mod23_unit2/qb_quality_loss",
    "u3_add_adjusted": "mod23_unit3/add_adjusted",
    "u3_compact_net": "mod23_unit3/compact_net",
    "u3_replace_adjusted": "mod23_unit3/replace_adjusted",
    "u4_blend_compact_net": "mod23_unit4/blend_compact_net",
    "u4_blend_unit1_trimmed": "mod23_unit4/blend_unit1_trimmed",
    "u4_shrink_served": "mod23_unit4/shrink_served",
    "u5_man_zone": "mod23_unit5/arm_a_man_zone",
    "u5_coverage_type": "mod23_unit5/arm_b_coverage_type",
    "u5_both": "mod23_unit5/arm_c_both",
}
METRICS = ("log_loss", "brier", "rps")


def theta_for(atoms, counts, line, target):
    home, away = atoms > line, atoms < line

    def diff(theta):
        mass = mass_at_theta(atoms, counts, theta)
        return float(mass[home].sum() / mass[home | away].sum() - target)

    if not 0 < target < 1:
        return None
    lo, hi = -2.0, 2.0
    if diff(lo) > 0 or diff(hi) < 0:
        return None
    return brentq(diff, lo, hi, xtol=1e-13)


def rps(atoms, mass, actual):
    grid = np.arange(min(atoms.min(), actual), max(atoms.max(), actual) + 1)
    cum = np.concatenate([[0.0], np.cumsum(mass)])
    cdf = cum[np.searchsorted(atoms, grid, side="right")]
    return float(np.square(cdf - (actual <= grid)).sum())


def load(name):
    return pd.read_parquet(Path("artifacts") / f"{name}_per_game.parquet")



from scipy.optimize import minimize
from scipy.special import expit, logit


def fit(z, y, kind):
    def nll(par):
        q = par[0] * z + (par[1] if kind == "platt" else 0.0)
        return float(np.mean(np.logaddexp(0, q) - y * q))
    x0 = [1.0, 0.0] if kind == "platt" else [1.0]
    return minimize(nll, x0, method="BFGS").x


def recal(f, kind):
    d = f.loc[f.margin_vs_open.ne(0) & f.correct_at_open_probability_rule.notna()]
    seasons_all = f.season
    p_all = f.home_cover_probability_at_open.clip(1e-6, 1 - 1e-6)
    z_all = logit(p_all)
    out = pd.Series(np.nan, index=f.index)
    for s in sorted(f.season.unique()):
        tr = d.loc[d.season.ne(s)]
        par = fit(logit(tr.home_cover_probability_at_open.clip(1e-6, 1 - 1e-6)).to_numpy(), (tr.margin_vs_open > 0).astype(float).to_numpy(), kind)
        m = seasons_all.eq(s)
        out[m] = expit(par[0] * z_all[m] + (par[1] if kind == "platt" else 0.0))
    return out


def main():
    features = pd.read_parquet(FEATURES, columns=["game_id", "season", "week", "gameday", "spread_line", "result"])
    opener = pd.read_parquet(OPENER / "per_game.parquet")
    opener = opener.loc[opener.season.between(2020, 2025)].copy()
    pool = prior_pool(features, pd.read_parquet(OPENER / "per_game.parquet").set_index("game_id").tue_open_home_spread)
    frames = {"base": opener}
    for arm, path in ARMS.items():
        frames[arm] = load(path)
    for arm, f in frames.items():
        frames[arm] = f if f.index.name == "game_id" else f.set_index("game_id")
    kinds = ("raw", "temp", "platt")
    P = {}
    params = []
    for arm, f in frames.items():
        d = f.loc[f.margin_vs_open.ne(0) & f.correct_at_open_probability_rule.notna()]
        zt = logit(d.home_cover_probability_at_open.clip(1e-6, 1 - 1e-6)).to_numpy()
        yt = (d.margin_vs_open > 0).astype(float).to_numpy()
        params.append({"arm": arm, "temp_all": float(fit(zt, yt, "temp")[0]), "platt_all": fit(zt, yt, "platt").tolist()})
        P[arm] = {"raw": f.home_cover_probability_at_open, "temp": recal(f, "temp"), "platt": recal(f, "platt")}
    result = features.set_index("game_id").result
    gameday = features.set_index("game_id").gameday
    games = opener.assign(gameday=pd.to_datetime(opener.game_id.map(gameday)))
    rps_rows = {(a, k): {} for a in frames for k in kinds}
    for (season, week), group in games.groupby(["season", "week"], sort=True):
        target = features.loc[features.season.eq(season) & features.week.eq(week)]
        prior = prior_pool_for_week(pool, season=int(season), week=int(week), cutoff=pd.to_datetime(target.gameday).min(), exclude_game_ids=group.game_id)
        for gid in group.game_id:
            line = float(frames["base"].loc[gid].tue_open_home_spread)
            actual = float(result.loc[gid])
            selected, _ = select_band(prior, line)
            atoms, counts = atomic_counts(selected)
            for arm, f in frames.items():
                if gid not in f.index:
                    continue
                for k in kinds:
                    t = theta_for(atoms, counts, line, float(P[arm][k].loc[gid]))
                    if t is not None:
                        rps_rows[(arm, k)][gid] = rps(atoms, mass_at_theta(atoms, counts, t), actual)
    tables = {}
    for arm, f in frames.items():
        d = f.loc[f.margin_vs_open.ne(0) & f.correct_at_open_probability_rule.notna()].copy()
        y = (d.margin_vs_open > 0).astype(float)
        for k in kinds:
            p = P[arm][k].loc[d.index].clip(1e-6, 1 - 1e-6)
            tables[(arm, k)] = pd.DataFrame({"season": d.season, "log_loss": -(y * np.log(p) + (1 - y) * np.log(1 - p)), "brier": (p - y) ** 2, "rps": pd.Series(rps_rows[(arm, k)])})
    rng = np.random.default_rng(SEED)
    seasons = np.array(sorted(tables[("base", "raw")].season.unique()))
    draws = rng.integers(0, len(seasons), size=(DRAWS, len(seasons)))
    out = []
    base_levels = []
    for k in kinds:
        b = tables[("base", k)]
        base_levels.append({"kind": k, **{m: float(b[m].mean()) for m in METRICS}, "season_log_loss": b.groupby("season").log_loss.mean().to_dict()})
        for arm in ARMS:
            j = tables[(arm, k)].join(b, rsuffix="_b", how="inner")
            row = {"kind": k, "arm": arm, "n": len(j), "arm_log_loss": float(tables[(arm, k)].log_loss.mean())}
            for m in METRICS:
                diff = (j[m + "_b"] - j[m]).dropna()
                s = diff.groupby(j.season).agg(["sum", "count"]).reindex(seasons).fillna(0)
                num, den = s["sum"].to_numpy(), s["count"].to_numpy()
                boot = num[draws].sum(1) / den[draws].sum(1)
                row[m] = float(diff.mean())
                row[m + "_lo"], row[m + "_hi"] = np.percentile(boot, [2.5, 97.5]).tolist()
                row[m + "_pplus"] = float((boot > 0).mean())
                row[m + "_seasons_pos"] = int((((s["sum"] / s["count"]).to_numpy()) > 0).sum())
            out.append(row)
    res = pd.DataFrame(out)
    res.to_csv(OUT / "results.csv", index=False)
    json.dump({"base_levels": base_levels, "params": params, "rows": out}, open(OUT / "results.json", "w"), indent=1)
    picks = pd.DataFrame({a: frames[a].pick_home_at_open_probability_rule for a in frames})
    same = {}
    for a in ("u3_compact_net", "u4_blend_compact_net", "u4_blend_unit1_trimmed", "u4_shrink_served"):
        same[a] = {b: int((picks[a].fillna(-1) != picks[b].fillna(-1)).sum()) for b in ("base", "u3_compact_net", "u4_blend_compact_net", "u4_blend_unit1_trimmed", "u4_shrink_served")}
    pp = pd.DataFrame({a: frames[a].home_cover_probability_at_open for a in ("u3_compact_net", "u4_blend_compact_net", "u4_blend_unit1_trimmed", "u4_shrink_served")})
    same["p_maxabsdiff_vs_u3"] = {a: float((pp[a] - pp.u3_compact_net).abs().max()) for a in pp}
    json.dump(same, open(OUT / "pick_identity.json", "w"), indent=1)
    pd.set_option("display.width", 250)
    print(json.dumps(base_levels, indent=0)[:1500])
    print(json.dumps(params)[:1500])
    print(json.dumps(same))
    for k in ("temp", "platt"):
        r = res.loc[res.kind.eq(k)]
        print(k)
        print(r[["arm", "log_loss", "log_loss_lo", "log_loss_hi", "log_loss_pplus", "log_loss_seasons_pos", "brier_pplus", "rps_pplus", "rps_seasons_pos"]].round(5).to_string())


main()
