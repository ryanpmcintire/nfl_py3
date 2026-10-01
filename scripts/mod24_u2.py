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

OUT = Path("artifacts/mod24_u2")
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


def main():
    features = pd.read_parquet(
        FEATURES, columns=["game_id", "season", "week", "gameday", "spread_line", "result"]
    )
    opener = pd.read_parquet(OPENER / "per_game.parquet")
    opener = opener.loc[opener.season.between(2020, 2025)].copy()
    pool = prior_pool(features, pd.read_parquet(OPENER / "per_game.parquet").set_index("game_id").tue_open_home_spread)
    frames = {"base": opener}
    for arm, path in ARMS.items():
        frames[arm] = load(path)
    for arm, f in frames.items():
        frames[arm] = f if f.index.name == "game_id" else f.set_index("game_id")
    ids = opener.game_id.tolist()
    result = features.set_index("game_id").result
    rps_rows = {arm: {} for arm in frames}
    skipped = {arm: 0 for arm in frames}
    gameday = features.set_index("game_id").gameday
    games = opener.assign(gameday=pd.to_datetime(opener.game_id.map(gameday)))
    for (season, week), group in games.groupby(["season", "week"], sort=True):
        target = features.loc[features.season.eq(season) & features.week.eq(week)]
        prior = prior_pool_for_week(
            pool,
            season=int(season),
            week=int(week),
            cutoff=pd.to_datetime(target.gameday).min(),
            exclude_game_ids=group.game_id,
        )
        for gid in group.game_id:
            base_row = frames["base"].loc[gid]
            line = float(base_row.tue_open_home_spread)
            actual = float(result.loc[gid])
            selected, _ = select_band(prior, line)
            atoms, counts = atomic_counts(selected)
            for arm, f in frames.items():
                if gid not in f.index:
                    skipped[arm] += 1
                    continue
                theta = theta_for(atoms, counts, line, float(f.loc[gid].home_cover_probability_at_open))
                if theta is None:
                    skipped[arm] += 1
                    continue
                rps_rows[arm][gid] = rps(atoms, mass_at_theta(atoms, counts, theta), actual)
    table = {}
    for arm, f in frames.items():
        d = f.loc[f.margin_vs_open.ne(0) & f.correct_at_open_probability_rule.notna()].copy()
        d["y"] = (d.margin_vs_open > 0).astype(float)
        d["p"] = d.home_cover_probability_at_open.clip(1e-6, 1 - 1e-6)
        d["log_loss"] = -(d.y * np.log(d.p) + (1 - d.y) * np.log(1 - d.p))
        d["brier"] = (d.p - d.y) ** 2
        d["acc"] = d.correct_at_open_probability_rule
        d["pick"] = d.pick_home_at_open_probability_rule
        d["rps"] = pd.Series(rps_rows[arm])
        table[arm] = d[["season", "log_loss", "brier", "rps", "acc", "pick"]]
    base = table["base"]
    rng = np.random.default_rng(SEED)
    seasons = np.array(sorted(base.season.unique()))
    draws = rng.integers(0, len(seasons), size=(DRAWS, len(seasons)))
    out = []
    for arm in ARMS:
        j = table[arm].join(base, rsuffix="_b", how="inner")
        row = {"arm": arm, "n": len(j), "rps_skipped": skipped[arm]}
        for m in METRICS:
            diff = (j[m + "_b"] - j[m]).dropna()
            s = diff.groupby(j.season).agg(["sum", "count"]).reindex(seasons).fillna(0)
            num, den = s["sum"].to_numpy(), s["count"].to_numpy()
            boot = num[draws].sum(1) / den[draws].sum(1)
            by = (s["sum"] / s["count"]).to_numpy()
            row[m] = float(diff.mean())
            row[m + "_lo"], row[m + "_hi"] = np.percentile(boot, [2.5, 97.5]).tolist()
            row[m + "_pplus"] = float((boot > 0).mean())
            row[m + "_seasons_pos"] = int((by > 0).sum())
        row["acc_diff"] = float((j.acc - j.acc_b).mean())
        dec = j.loc[j.pick.ne(j.pick_b)]
        row["flips"] = len(dec)
        row["flip_w"] = int((dec.acc > dec.acc_b).sum())
        row["flip_l"] = int((dec.acc < dec.acc_b).sum())
        s = (j.acc - j.acc_b).groupby(j.season).sum() / j.groupby("season").size()
        row["acc_seasons_pos"] = int((s > 0).sum())
        out.append(row)
    res = pd.DataFrame(out)
    res.to_csv(OUT / "results.csv", index=False)
    base_means = {m: float(base[m].mean()) for m in METRICS}
    json.dump({"base_means": base_means, "rows": out}, open(OUT / "results.json", "w"), indent=1)
    pd.set_option("display.width", 250)
    print(base_means, skipped["base"], len(base))
    print(res[["arm", "n", "rps_skipped", "log_loss", "log_loss_lo", "log_loss_hi", "log_loss_pplus", "log_loss_seasons_pos"]].round(5).to_string())
    print(res[["arm", "brier", "brier_pplus", "brier_seasons_pos", "rps", "rps_pplus", "rps_seasons_pos", "acc_diff", "flips", "flip_w", "flip_l", "acc_seasons_pos"]].round(5).to_string())


main()
