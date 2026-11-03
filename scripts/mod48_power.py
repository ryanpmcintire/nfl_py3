import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
import forward_monitor as fm  # noqa: E402
import mod41_unit2_components as u2  # noqa: E402

PICK = REPO / "artifacts" / "pick_probability" / "20261103T195411Z" / "per_game.parquet"
OUT = REPO / "artifacts" / "mod48" / "power.json"
HORIZON_SEASONS = 1
FIRST_FORWARD_WEEK = 9
LAST_REGULAR_WEEK = 18
QUANTILES = (0.1, 0.5, 0.9)
SEARCH_STEPS = 60
SHRINK_CEILING = 8.0


def load():
    d, _beta = u2.build()
    pk = pd.read_parquet(PICK, columns=["game_id", "out_of_season_home_probability"])
    d = d.merge(pk, on="game_id", how="left")
    d["logit_s"] = np.log(
        d["out_of_season_home_probability"].clip(1e-6, 1 - 1e-6)
        / (1 - d["out_of_season_home_probability"].clip(1e-6, 1 - 1e-6))
    )
    bp, _ = u2.loso(d, ["logit_s"])
    tp, _ = u2.loso(d, ["logit_s", "s_resid"])
    d = d.assign(p_base=bp, p_term=tp)
    d = d[d["usable"]].dropna(subset=["p_base", "p_term", "y"])
    z = np.log(d["p_term"] / (1 - d["p_term"])) - np.log(d["p_base"] / (1 - d["p_base"]))
    return d, d["p_base"].to_numpy(), z.to_numpy(), d["y"].to_numpy()


def terms(ps, pc, y):
    up = np.log(pc / ps)
    down = np.log((1 - pc) / (1 - ps))
    diff = np.where(y == 1.0, up, down)
    worst = np.maximum(np.maximum(-up, -down), 0.0)
    scale = 0.5 * (up * up + down * down)
    return diff, worst, scale


def log_e_paths(diff, worst, scale, alpha):
    n = diff.shape[1]
    t = np.arange(1, n + 1, dtype=float)
    s1 = np.cumsum(diff, axis=1) - diff
    s2 = np.cumsum(diff * diff, axis=1) - diff * diff
    mu = s1 / t
    var = (scale + s2 - 2.0 * mu * s1 + (t - 1.0) * mu * mu) / t
    lam = np.sqrt(2.0 * math.log(1.0 / alpha) / (var * t * np.log1p(t)))
    with np.errstate(divide="ignore"):
        cap = np.where(worst > 0, fm.TRUNCATION / worst, np.inf)
    lam = np.minimum(lam, cap)
    return np.cumsum(np.log1p(lam * diff), axis=1)


def sample_gain(ps, z, y, shrink):
    lz = np.log(ps / (1 - ps)) + shrink * z
    pc = 1.0 / (1.0 + np.exp(-lz))
    return float(terms(ps, pc, y)[0].mean())


def peak_shrink(ps, z, y, hi):
    lo = 0.0
    for _ in range(SEARCH_STEPS):
        a = lo + (hi - lo) / 3.0
        b = hi - (hi - lo) / 3.0
        if sample_gain(ps, z, y, a) < sample_gain(ps, z, y, b):
            lo = a
        else:
            hi = b
    return 0.5 * (lo + hi)


def solve_shrink(ps, z, y, target, lo, hi, rising):
    for _ in range(SEARCH_STEPS):
        mid = 0.5 * (lo + hi)
        below = sample_gain(ps, z, y, mid) < target
        if below == rising:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def simulate(ps, z, y, shrink, mode, draws, n, alpha, rng):
    m = len(ps)
    gi = rng.integers(0, m, size=(draws, n))
    zi = gi if mode == "paired" else rng.integers(0, m, size=(draws, n))
    p0 = ps[gi]
    lz = np.log(p0 / (1 - p0)) + shrink * z[zi]
    pc = 1.0 / (1.0 + np.exp(-lz))
    diff, worst, scale = terms(p0, pc, y[gi])
    le = log_e_paths(diff, worst, scale, alpha)
    hit = le >= math.log(1.0 / alpha)
    first = np.where(hit.any(axis=1), hit.argmax(axis=1) + 1, 0)
    return first, float(diff.mean()), float(diff.var(ddof=1))


def summarize(first, n, gpw, horizon_games):
    crossed = first > 0
    q = {}
    if crossed.any():
        for p in QUANTILES:
            g = float(np.quantile(first[crossed], p))
            q[f"q{int(p * 100)}_games"] = g
            q[f"q{int(p * 100)}_weeks"] = g / gpw
    inside = crossed & (first <= horizon_games)
    return {
        "crossed_within_horizon_games": float(crossed.mean()),
        "crossed_within_2026_remaining": float(inside.mean()),
        "conditional_on_crossing": q,
        "draws": len(first),
        "horizon_games": int(n),
    }


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Power and false-crossing simulation of the forward_monitor e-process for the MOD-41 "
            "served-scale residual-load term, resampling 2020-25 per-game served-base paired gains"
        )
    )
    parser.add_argument("--draws", type=int, default=2000)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=u2.PERM_SEED)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    d, ps, z, y = load()
    wk = d.groupby(["season", "week"]).size()
    gpw = float(wk.mean())
    n = len(d)
    remaining = (LAST_REGULAR_WEEK - FIRST_FORWARD_WEEK + 1) * gpw
    rng = np.random.default_rng(args.seed)
    out = {
        "alpha": args.alpha,
        "threshold": 1.0 / args.alpha,
        "usable_games": n,
        "games_per_week": gpw,
        "remaining_2026_games": remaining,
        "arms": {},
    }
    peak = peak_shrink(ps, z, y, SHRINK_CEILING)
    full = sample_gain(ps, z, y, 1.0)
    half_s = solve_shrink(ps, z, y, 0.5 * full, 0.0, min(1.0, peak), True)
    null_s = solve_shrink(ps, z, y, 0.0, peak, SHRINK_CEILING, False)
    out["shrink"] = {"peak": peak, "full_gain": full, "half": half_s, "boundary_null": null_s}
    for name, shrink, mode in (
        ("full_effect", 1.0, "paired"),
        ("half_effect", half_s, "paired"),
        ("boundary_null_zero_gain", null_s, "paired"),
        ("permuted_null", 1.0, "permuted"),
    ):
        first, mean_gain, var_gain = simulate(
            ps, z, y, shrink, mode, args.draws, n, args.alpha, rng
        )
        res = summarize(first, n, gpw, remaining)
        res["mean_gain_per_game"] = mean_gain
        res["per_game_sd"] = math.sqrt(var_gain)
        rate = mean_gain**2 / (2.0 * var_gain)
        res["analytic_games_to_threshold"] = (
            math.log(1.0 / args.alpha) / rate if mean_gain > 0 else None
        )
        res["analytic_weeks_to_threshold"] = (
            None if mean_gain <= 0 else math.log(1.0 / args.alpha) / rate / gpw
        )
        out["arms"][name] = res
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
