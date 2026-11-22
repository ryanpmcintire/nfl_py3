import argparse

import numpy as np
import pandas as pd


def load(args):
    frame = pd.read_csv(args.csv).drop_duplicates(["season", "expert"])
    if args.first_entry:
        frame = frame[frame["expert"].str.endswith("-1")].copy()
        frame["expert"] = frame["expert"].str.replace(r"-1$", "", regex=True)
    if args.min_share > 0 and "picks" in frame:
        top = frame.groupby("season")["picks"].transform("max")
        frame = frame[frame["picks"] >= args.min_share * top]
    frame = frame.assign(n=frame["w"] + frame["l"])
    return frame[frame["n"] > 0].reset_index(drop=True)


class Panel:
    def __init__(self, frame, bar):
        self.n = frame["n"].to_numpy()
        self.w = frame["w"].to_numpy()
        self.season = frame["season"].to_numpy()
        self.codes, self.names = pd.factorize(frame["expert"])
        sums = frame.groupby("season")[["w", "n"]].sum()
        self.p_season = (sums["w"] / sums["n"]).reindex(self.season).to_numpy()
        rows = np.arange(len(frame))
        keyed = pd.DataFrame({"code": self.codes, "season": self.season, "row": rows})
        nxt = keyed.assign(season=keyed["season"] - 1)
        pairs = keyed.merge(nxt, on=["code", "season"], suffixes=("_a", "_b"))
        self.ia, self.ib = pairs["row_a"].to_numpy(), pairs["row_b"].to_numpy()
        self.pair_code = pairs["code"].to_numpy()
        self.bar = bar
        self.base = np.lexsort((self.n, self.season))

    def dev(self, wins):
        return wins / self.n - self.p_season

    def corr(self, wins):
        d = self.dev(wins)
        return float(np.corrcoef(d[self.ia], d[self.ib])[0, 1])

    def excess(self, wins):
        d = self.dev(wins)
        return float(np.var(d, ddof=1) - np.mean(self.p_season * (1 - self.p_season) / self.n))

    def at_bar(self, wins):
        return (wins / self.n) >= self.bar

    def repeat(self, wins):
        hits = np.bincount(self.codes, weights=self.at_bar(wins), minlength=len(self.names))
        return int((hits >= 2).sum())

    def permuted(self, rng):
        order = np.lexsort((rng.random(len(self.n)), self.n, self.season))
        out = np.empty_like(self.w)
        out[self.base] = self.w[order]
        return out

    def stats(self, wins):
        return self.corr(wins), self.excess(wins), int(self.at_bar(wins).sum()), self.repeat(wins)


def boot_corr(panel, rng, reps):
    d = panel.dev(panel.w)
    a, b = d[panel.ia], d[panel.ib]
    clusters, inverse = np.unique(panel.pair_code, return_inverse=True)
    out = np.empty(reps)
    for k in range(reps):
        mult = np.bincount(rng.integers(0, len(clusters), len(clusters)), minlength=len(clusters))
        wt = mult[inverse].astype(float)
        ma, mb = np.average(a, weights=wt), np.average(b, weights=wt)
        cov = np.average((a - ma) * (b - mb), weights=wt)
        va, vb = np.average((a - ma) ** 2, weights=wt), np.average((b - mb) ** 2, weights=wt)
        out[k] = cov / np.sqrt(va * vb)
    return out


def report(label, obs, null):
    names = ["corr", "excess_var", "at_bar", "at_bar_2plus"]
    for i, name in enumerate(names):
        col = null[:, i]
        lo, hi = np.percentile(col, [2.5, 97.5])
        print(
            f"{label} {name} obs {obs[i]:.4f} null_mean {col.mean():.4f}"
            f" null_95 [{lo:.4f}, {hi:.4f}] null_ge {np.mean(col >= obs[i]):.3f}"
        )


def main():
    parser = argparse.ArgumentParser(description="ATS repeatability v luck nulls.")
    parser.add_argument("--csv", required=True)
    parser.add_argument("--sims", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=20261122)
    parser.add_argument("--bar", type=float, default=0.55)
    parser.add_argument("--min-share", type=float, default=0.0)
    parser.add_argument("--first-entry", action="store_true")
    parser.add_argument("--top", type=int, default=15)
    args = parser.parse_args()
    frame = load(args)
    panel = Panel(frame, args.bar)
    rng = np.random.default_rng(args.seed)
    obs = panel.stats(panel.w)
    binom = np.array(
        [panel.stats(rng.binomial(panel.n, panel.p_season)) for _ in range(args.sims)]
    )
    perm = np.array([panel.stats(panel.permuted(rng)) for _ in range(args.sims)])
    boot = boot_corr(panel, rng, 2000)
    seasons = sorted(int(s) for s in frame["season"].unique())
    print(f"person_seasons {len(frame)} experts {len(panel.names)} seasons {seasons}")
    rates = frame.groupby("season")["w"].sum() / frame.groupby("season")["n"].sum()
    print(f"pooled_rate {frame['w'].sum() / frame['n'].sum():.4f} picks {int(frame['n'].sum())}")
    print("season_rates " + " ".join(f"{s}:{r:.4f}" for s, r in rates.items()))
    print(f"pairs {len(panel.ia)} pair_experts {len(np.unique(panel.pair_code))}")
    report("binomial", obs, binom)
    report("permute", obs, perm)
    blo, bhi = np.percentile(boot, [2.5, 97.5])
    print(f"boot_95 [{blo:.4f}, {bhi:.4f}] probability_positive {np.mean(boot > 0):.3f}")
    skill_sd = np.sqrt(max(obs[1], 0))
    print(f"skill_sd {skill_sd:.4f}")
    career = frame.groupby("expert")[["w", "l"]].sum()
    career["rate"] = career["w"] / (career["w"] + career["l"])
    career["seasons"] = frame.groupby("expert").size()
    career["at_bar"] = frame.assign(h=panel.at_bar(panel.w)).groupby("expert")["h"].sum()
    multi = career[career["seasons"] >= 2].sort_values("rate", ascending=False)
    print(f"experts_2plus_seasons {len(multi)}")
    print(multi.head(args.top).to_string())


if __name__ == "__main__":
    main()
