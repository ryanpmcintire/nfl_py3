import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
import mod41_unit2_components as u2  # noqa: E402
from playcaller_change_screen import BOOTSTRAP_SEED  # noqa: E402

PICK = REPO / "artifacts" / "pick_probability" / "20261103T195411Z" / "per_game.parquet"
OUT = REPO / "artifacts" / "mod52" / "unit1_stability.json"
FAMILY = "mod52_mod41_stability_u1"
DRAWS = 2000
PERM_SEED = 5201
WEEK_BLOCKS = [(5, 8), (9, 12), (13, 18)]
TERCILES = 3


def tercile(s):
    return pd.qcut(s.rank(method="first"), TERCILES, labels=False)


def boot(df, rng):
    groups = [g["diff"].to_numpy() for _, g in df.groupby("block")]
    sums = np.array([g.sum() for g in groups])
    cnts = np.array([len(g) for g in groups])
    idx = rng.integers(0, len(groups), (DRAWS, len(groups)))
    return sums[idx].sum(1) / cnts[idx].sum(1)


def cell(df, rng):
    dec = df.loc[(df["p_term"] >= 0.5) != (df["p_base"] >= 0.5)]
    tc = int(((dec["p_term"] >= 0.5).astype(int) == dec["y"]).sum())
    draws = boot(df, rng)
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return {
        "n": len(df),
        "decisive": len(dec),
        "decisive_term_correct": tc,
        "decisive_base_correct": len(dec) - tc,
        "gain": float(df["diff"].mean()),
        "ci95": [float(lo), float(hi)],
        "probability_positive": float((draws > 0).mean()),
    }


def homogeneity(df, label, rng):
    codes = df[label].to_numpy()
    x = df["diff"].to_numpy()
    cats = np.unique(codes)

    def stat(c):
        m = x.mean()
        return sum(((c == k).sum()) * (x[c == k].mean() - m) ** 2 for k in cats)

    obs = stat(codes)
    null = np.array([stat(rng.permutation(codes)) for _ in range(DRAWS)])
    return {"statistic": float(obs), "p": float((1 + (null >= obs).sum()) / (DRAWS + 1))}


def slope(df, label, rng):
    def fit(frame):
        return np.polyfit(frame[label].astype(float), frame["diff"], 1)[0]

    groups = [g for _, g in df.groupby("block")]
    draws = np.empty(DRAWS)
    for i in range(DRAWS):
        idx = rng.integers(0, len(groups), len(groups))
        draws[i] = fit(pd.concat([groups[j] for j in idx]))
    lo, hi = np.percentile(draws, [2.5, 97.5])
    codes = df[label].to_numpy()
    obs = fit(df)
    null = np.array([fit(df.assign(**{label: rng.permutation(codes)})) for _ in range(DRAWS)])
    return {
        "slope_per_step": float(obs),
        "ci95": [float(lo), float(hi)],
        "probability_positive": float((draws > 0).mean()),
        "perm_p_two_sided": float((1 + (np.abs(null) >= abs(obs)).sum()) / (DRAWS + 1)),
    }


def main():
    d, _ = u2.build()
    pk = pd.read_parquet(PICK, columns=["game_id", "out_of_season_home_probability"])
    d = d.merge(pk, on="game_id", how="left")
    ps = d["out_of_season_home_probability"].clip(1e-6, 1 - 1e-6)
    d["logit_s"] = np.log(ps / (1 - ps))
    bp, _ = u2.loso(d, ["logit_s"])
    tp, _ = u2.loso(d, ["logit_s", "s_resid"])
    d["p_base"] = bp
    d["p_term"] = tp
    y = d["y"]

    def ll(p):
        p = p.clip(1e-9, 1 - 1e-9)
        return -(y * np.log(p) + (1 - y) * np.log(1 - p))

    d["diff"] = ll(d["p_base"]) - ll(d["p_term"])
    u = d.loc[d["usable"]].copy()
    u["season_cell"] = u["season"]
    u["week_block"] = -1
    for i, (a, b) in enumerate(WEEK_BLOCKS):
        u.loc[u["week"].between(a, b), "week_block"] = i
    u["spread_t"] = tercile(u["tue_open_home_spread"].abs())
    u["direction"] = (u["p_term"] > u["p_base"]).astype(int)
    u["term_t"] = tercile(u["s_resid"].abs())
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    prng = np.random.default_rng(PERM_SEED)
    out = {
        "family": FAMILY,
        "usable_games": len(u),
        "week_min": int(u["week"].min()),
        "week_max": int(u["week"].max()),
        "week_blocks": WEEK_BLOCKS,
        "overall": cell(u, rng),
        "profiles": {},
        "spread_tercile_edges_abs": [
            float(u.loc[u["spread_t"] == k, "tue_open_home_spread"].abs().max())
            for k in range(TERCILES)
        ],
        "term_tercile_edges_abs": [
            float(u.loc[u["term_t"] == k, "s_resid"].abs().max()) for k in range(TERCILES)
        ],
    }
    for label, ordered in (
        ("season_cell", False),
        ("week_block", True),
        ("spread_t", False),
        ("direction", False),
        ("term_t", True),
    ):
        cells = {str(k): cell(g, rng) for k, g in u.groupby(label)}
        prof = {"cells": cells, "homogeneity": homogeneity(u, label, prng)}
        if ordered:
            prof["slope"] = slope(u, label, prng)
        out["profiles"][label] = prof
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2, default=str))
    print(
        json.dumps(out["overall"]), out["spread_tercile_edges_abs"], out["term_tercile_edges_abs"]
    )
    for label, prof in out["profiles"].items():
        print(label, "homog", prof["homogeneity"], "slope", prof.get("slope"))
        for k, c in prof["cells"].items():
            print(
                " ",
                k,
                c["n"],
                f"dec {c['decisive_term_correct']}-{c['decisive_base_correct']}",
                f"gain {c['gain']:+.4f} [{c['ci95'][0]:+.4f},{c['ci95'][1]:+.4f}]",
                f"P+ {c['probability_positive']:.2f}",
            )


if __name__ == "__main__":
    main()
