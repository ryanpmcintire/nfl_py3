import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

RAW = Path("data/splash/field/2026_weekly_points_w01-08_raw.txt")
OUT = Path("data/splash/field/2026_weekly_points_w01-08.parquet")
GAMES = Path("data/splash/field/2026_field_best_picks_w01-08.parquet")
RES = Path("artifacts/pool_rank_replay/weekly_points_field.json")
WEEKS = 8
OUR_TOTAL = 65
OUR_RANK = 19


def load():
    text = "\n".join(ln for ln in RAW.read_text().splitlines() if not ln.startswith("#"))
    nums = [int(t) for t in re.findall(r"\d+", text)]
    arr = np.array(nums).reshape(-1, WEEKS)
    return arr


def games_per_week():
    g = pd.read_parquet(GAMES)
    return g.groupby("week")["game_id"].nunique().reindex(range(1, WEEKS + 1)).to_numpy()


def split_half(rate):
    odd, even = rate[:, 0::2].mean(axis=1), rate[:, 1::2].mean(axis=1)
    r = float(np.corrcoef(odd, even)[0, 1])
    sb = 2 * r / (1 + r)
    return r, sb


def main():
    arr = load()
    n = games_per_week()
    entries = arr.shape[0]
    pd.DataFrame(arr, columns=[f"w{w}" for w in range(1, WEEKS + 1)]).assign(
        entry=np.arange(entries), total=arr.sum(axis=1)
    ).to_parquet(OUT)
    rate = arr / n
    p_bar = float(arr.sum() / (entries * n.sum()))
    weeks = []
    for w in range(WEEKS):
        weeks.append(
            {
                "week": w + 1,
                "games": int(n[w]),
                "mean": float(arr[:, w].mean()),
                "sd": float(arr[:, w].std(ddof=1)),
                "binomial_sd_at_field_mean": float(np.sqrt(n[w] * p_bar * (1 - p_bar))),
            }
        )
    tot = arr.sum(axis=1)
    bin_sd_total = float(np.sqrt(n.sum() * p_bar * (1 - p_bar)))
    rng = np.random.default_rng(0)
    r, sb = split_half(rate)
    reps = []
    for _ in range(4000):
        i = rng.integers(0, entries, entries)
        reps.append(split_half(rate[i])[1])
    ours = np.where(tot == OUR_TOTAL)[0]
    out = {
        "entries_read": int(entries),
        "games_per_week": n.tolist(),
        "field_mean_hit_rate": p_bar,
        "weeks": weeks,
        "total_mean": float(tot.mean()),
        "total_sd": float(tot.std(ddof=1)),
        "binomial_sd_total": bin_sd_total,
        "overdispersion_ratio_total_sd": float(tot.std(ddof=1) / bin_sd_total),
        "split_half_odd_even_r": r,
        "split_half_spearman_brown": sb,
        "split_half_sb_interval_95": [
            float(np.quantile(reps, 0.025)),
            float(np.quantile(reps, 0.975)),
        ],
        "total_quantiles": {
            str(q): float(np.quantile(tot, q)) for q in (0.05, 0.25, 0.5, 0.75, 0.95)
        },
        "entries_with_our_total": len(ours),
        "entries_above_our_total": int((tot > OUR_TOTAL).sum()),
        "our_percentile_of_read_entries": float(
            (tot < OUR_TOTAL).mean() + 0.5 * (tot == OUR_TOTAL).mean()
        ),
        "our_z": float((OUR_TOTAL - tot.mean()) / tot.std(ddof=1)),
    }
    RES.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
