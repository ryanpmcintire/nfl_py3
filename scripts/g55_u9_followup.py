import argparse

import numpy as np
import pandas as pd


def load(path, min_share):
    frame = pd.read_csv(path).drop_duplicates(["season", "expert"])
    if min_share > 0 and "picks" in frame:
        top = frame.groupby("season")["picks"].transform("max")
        frame = frame[frame["picks"] >= min_share * top]
    frame = frame.assign(n=frame["w"] + frame["l"])
    frame = frame[frame["n"] > 0].reset_index(drop=True)
    sums = frame.groupby("season")[["w", "n"]].sum()
    frame["p_season"] = (sums["w"] / sums["n"]).reindex(frame["season"]).to_numpy()
    return frame


def permute(frame, rng):
    base = np.lexsort((frame["n"].to_numpy(), frame["season"].to_numpy()))
    order = np.lexsort((rng.random(len(frame)), frame["n"].to_numpy(), frame["season"].to_numpy()))
    out = np.empty(len(frame), dtype=int)
    out[base] = frame["w"].to_numpy()[order]
    return out


def tail_counts(frame, wins, bar, min_seasons):
    hit = pd.Series((wins / frame["n"].to_numpy()) >= bar)
    per = hit.groupby(frame["expert"].to_numpy()).sum()
    seasons = frame.groupby("expert").size().reindex(per.index)
    keep = seasons >= min_seasons
    return int((per[keep] >= min_seasons).sum())


def best_career(frame, wins, min_seasons):
    tmp = pd.DataFrame({"expert": frame["expert"], "w": wins, "n": frame["n"]})
    car = tmp.groupby("expert")[["w", "n"]].sum()
    car = car[frame.groupby("expert").size().reindex(car.index) >= min_seasons]
    return float((car["w"] / car["n"]).max())


def forward(frame, wins, bar):
    rate = wins / frame["n"].to_numpy()
    keyed = frame[["expert", "season"]].assign(hot=rate >= bar)
    nxt = frame[["expert", "season", "n", "p_season"]].assign(w=wins, season=frame["season"] - 1)
    pairs = keyed[keyed["hot"]].merge(nxt, on=["expert", "season"])
    return pairs


def main():
    parser = argparse.ArgumentParser(description="Tail repeat and follow-forward tests.")
    parser.add_argument("--csv", required=True)
    parser.add_argument("--min-share", type=float, default=1.0)
    parser.add_argument("--bar", type=float, default=0.55)
    parser.add_argument("--sims", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20261123)
    args = parser.parse_args()
    frame = load(args.csv, args.min_share)
    rng = np.random.default_rng(args.seed)
    wins = frame["w"].to_numpy()
    for k in (3, 4):
        obs = tail_counts(frame, wins, args.bar, k)
        sims = range(args.sims)
        null = np.array([tail_counts(frame, permute(frame, rng), args.bar, k) for _ in sims])
        print(
            f"all_{k}_at_bar entries_with_{k}plus_seasons "
            f"{int((frame.groupby('expert').size() >= k).sum())} obs {obs}"
            f" perm_null_mean {null.mean():.2f} null_ge {np.mean(null >= obs):.3f}"
        )
        obs_b = best_career(frame, wins, k)
        null_b = np.array([best_career(frame, permute(frame, rng), k) for _ in range(args.sims)])
        print(
            f"best_career_{k}plus obs {obs_b:.4f} perm_null_mean {null_b.mean():.4f}"
            f" null_ge {np.mean(null_b >= obs_b):.3f}"
        )
    pairs = forward(frame, wins, args.bar)
    rate_next = pairs["w"].sum() / pairs["n"].sum()
    base_next = np.average(pairs["p_season"], weights=pairs["n"])
    experts = pairs["expert"].unique()
    codes = pd.factorize(pairs["expert"])[0]
    diffs = []
    for _ in range(2000):
        mult = np.bincount(rng.integers(0, len(experts), len(experts)), minlength=len(experts))
        wt = mult[codes]
        diffs.append(
            (wt * pairs["w"]).sum() / (wt * pairs["n"]).sum()
            - np.average(pairs["p_season"], weights=wt * pairs["n"])
        )
    diffs = np.asarray(diffs)
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    print(
        f"follow_hot pairs {len(pairs)} picks {int(pairs['n'].sum())} next_rate {rate_next:.4f}"
        f" season_mean {base_next:.4f} diff {rate_next - base_next:.4f}"
        f" boot_95 [{lo:.4f}, {hi:.4f}] probability_positive {np.mean(diffs > 0):.3f}"
    )
    by = pairs.groupby("season")[["w", "n"]].sum()
    by["rate_next"] = by["w"] / by["n"]
    by["season_mean_next"] = pairs.groupby("season")["p_season"].first()
    print(by.to_string())


if __name__ == "__main__":
    main()
