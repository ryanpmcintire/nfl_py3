import argparse

import numpy as np
import pandas as pd


def pair_corr(frame, wins):
    rate = wins / frame["n"].to_numpy()
    keyed = pd.DataFrame(
        {"expert": frame["expert"].to_numpy(), "season": frame["season"].to_numpy(), "rate": rate}
    )
    nxt = keyed.assign(season=keyed["season"] - 1)
    pairs = keyed.merge(nxt, on=["expert", "season"], suffixes=("_a", "_b"))
    if len(pairs) < 3:
        return np.nan, len(pairs)
    return float(np.corrcoef(pairs["rate_a"], pairs["rate_b"])[0, 1]), len(pairs)


def excess_var(frame, wins, pooled):
    n = frame["n"].to_numpy()
    return float(np.var(wins / n, ddof=1) - np.mean(pooled * (1 - pooled) / n))


def repeaters(frame, wins, bar):
    hit = pd.Series((wins / frame["n"].to_numpy()) >= bar, index=frame.index)
    per = hit.groupby(frame["expert"]).sum()
    return int((per >= 2).sum())


def bootstrap_corr(frame, rng, reps):
    experts = frame["expert"].unique()
    out = []
    for _ in range(reps):
        pick = rng.choice(experts, size=len(experts), replace=True)
        parts = [frame[frame["expert"] == e].assign(expert=f"{e}#{k}") for k, e in enumerate(pick)]
        sample = pd.concat(parts, ignore_index=True)
        out.append(pair_corr(sample, sample["w"].to_numpy())[0])
    return np.asarray([b for b in out if np.isfinite(b)])


def share_ge(null, obs):
    return float(np.nanmean(null >= obs))


def main():
    parser = argparse.ArgumentParser(description="ATS repeatability v binomial luck.")
    parser.add_argument("--csv", required=True)
    parser.add_argument("--sims", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=20261122)
    parser.add_argument("--bar", type=float, default=0.55)
    args = parser.parse_args()
    frame = pd.read_csv(args.csv).drop_duplicates(["season", "expert"]).reset_index(drop=True)
    frame["n"] = frame["w"] + frame["l"]
    wins = frame["w"].to_numpy()
    n = frame["n"].to_numpy()
    pooled = wins.sum() / n.sum()
    rng = np.random.default_rng(args.seed)
    obs_corr, npairs = pair_corr(frame, wins)
    obs_ex = excess_var(frame, wins, pooled)
    obs_rep = repeaters(frame, wins, args.bar)
    obs_single = int(((wins / n) >= args.bar).sum())
    nc, ne, nr, ns = [], [], [], []
    for _ in range(args.sims):
        sim = rng.binomial(n, pooled)
        nc.append(pair_corr(frame, sim)[0])
        ne.append(excess_var(frame, sim, pooled))
        nr.append(repeaters(frame, sim, args.bar))
        ns.append(int(((sim / n) >= args.bar).sum()))
    nc, ne, nr, ns = map(np.asarray, (nc, ne, nr, ns))
    boot = bootstrap_corr(frame, rng, 2000)
    seasons = sorted(int(s) for s in frame["season"].unique())
    print(f"person_seasons {len(frame)} experts {frame['expert'].nunique()} seasons {seasons}")
    print(f"pooled_rate {pooled:.4f} picks {int(n.sum())}")
    lo, hi = np.nanpercentile(nc, [2.5, 97.5])
    print(f"pairs {npairs} corr {obs_corr:.3f} null_mean {np.nanmean(nc):.3f}")
    print(f"null_95 [{lo:.3f}, {hi:.3f}] null_share_ge_obs {share_ge(nc, obs_corr):.3f}")
    blo, bhi = np.percentile(boot, [2.5, 97.5])
    print(f"boot_95 [{blo:.3f}, {bhi:.3f}] probability_positive {np.mean(boot > 0):.3f}")
    skill_sd = np.sqrt(max(obs_ex, 0))
    print(f"excess_var {obs_ex:.6f} skill_sd {skill_sd:.4f} null_ge {share_ge(ne, obs_ex):.3f}")
    print(f"at_bar {obs_single} null_mean {ns.mean():.2f} null_ge {share_ge(ns, obs_single):.3f}")
    print(f"at_bar_2plus {obs_rep} null_mean {nr.mean():.2f} null_ge {share_ge(nr, obs_rep):.3f}")
    career = frame.groupby("expert")[["w", "l"]].sum()
    career["rate"] = career["w"] / (career["w"] + career["l"])
    career["seasons"] = frame.groupby("expert").size()
    print(career.sort_values("rate", ascending=False).to_string())


if __name__ == "__main__":
    main()
