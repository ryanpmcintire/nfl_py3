import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from nfl_ats.evidence_conventions import probability_positive_from_draws
from nfl_ats.pool_hook import POOL_OFFSET, is_whole, tuesday_market_lines

FAMILY = "pool10_best_pick_hooks"
OUT_DIR = Path("artifacts/pool_rank_replay")
SCORING = Path("artifacts/prospective_scoring")
SEASON = 2026
FIELD_DIR = Path("data/splash/field")
DATA_ROOT = Path("data")
BOOT_DRAWS = 4000
PERM_DRAWS = 20000
SEED = 10
ADJUSTED_SEED = 1010
INTERVAL = (2.5, 97.5)
IRLS_MAX_ITER = 50
IRLS_TOL = 1e-9
SPECS = {
    "favourite_line": ("favourite", "favoured_by"),
    "favourite_only": ("favourite",),
}
PRIMARY = "favourite_line"
PLACEMENT_LOOKS = 11
OUTCOME_LOOKS = 1


def hook_games(raw):
    ids = raw["game_id"].unique().tolist()
    market = {}
    for w in sorted(int(v) for v in raw["week"].unique()):
        market.update(
            tuesday_market_lines(
                DATA_ROOT, SEASON, w, [g for g in ids if g.split("_")[1] == f"{w:02d}"]
            )
        )
    home = raw[raw["home"]].copy()
    home["market"] = home["game_id"].map(market)
    off = -home["line"] - home["market"]
    whole = home["market"].map(lambda v: isinstance(v, float) and is_whole(v))
    home["hook_home"] = np.where(whole & np.isclose(off.abs(), POOL_OFFSET), -np.sign(off), 0.0)
    return home[["game_id", "hook_home"]]


def team_rows(last_week):
    raw = pd.read_parquet(FIELD_DIR / f"{SEASON}_field_best_picks_w01-{last_week:02d}.parquet")
    raw = raw.merge(hook_games(raw), on="game_id")
    raw["hook_game"] = raw["hook_home"] != 0
    sign = np.where(raw["home"], 1.0, -1.0)
    raw["hook_side"] = raw["hook_game"] & (raw["hook_home"] * sign > 0)
    raw["favourite"] = raw["line"] < 0
    raw["favoured_by"] = -raw["line"]
    raw["size"] = raw["line"].abs()
    return raw


def poisson_fit(X, y, off, w, start):
    b = start.copy()
    for _ in range(IRLS_MAX_ITER):
        mu = np.exp(off + X @ b)
        hess = (X * (w * mu)[:, None]).T @ X
        step = np.linalg.lstsq(hess, X.T @ (w * (y - mu)), rcond=None)[0]
        b += step
        if np.abs(step).max() < IRLS_TOL:
            break
    return b


def side_flags(game_hook):
    flags = np.zeros(2 * len(game_hook))
    flags[0::2] = game_hook < 0
    flags[1::2] = game_hook > 0
    return flags


def pct(draws):
    lo, hi = np.percentile(draws, INTERVAL)
    return [float(lo), float(hi)]


def adjusted(frame, spec, rng, permute):
    f = frame.sort_values(["week", "game_id", "home"]).reset_index(drop=True)
    if (f["picks"] <= 0).any() or not f["home"].to_numpy()[1::2].all():
        raise ValueError("every game needs two picked sides")
    weeks = sorted(int(v) for v in f["week"].unique())
    wk = f["week"].to_numpy()
    y = f["best_picks"].to_numpy().astype(float)
    off = np.log(f["picks"].to_numpy().astype(float))
    game_hook = f["hook_home"].to_numpy()[1::2]
    game_week = wk[1::2]
    names = [*SPECS[spec], "hook_side"]
    cols = [(wk == w).astype(float) for w in weeks]
    cols += [f[c].to_numpy().astype(float) for c in SPECS[spec]]
    X = np.column_stack([*cols, side_flags(game_hook)])
    start = np.zeros(X.shape[1])
    for i, w in enumerate(weeks):
        start[i] = np.log(y[wk == w].sum() / f.loc[wk == w, "picks"].sum())
    ones = np.ones(len(f))
    b = poisson_fit(X, y, off, ones, start)
    k = len(weeks)
    game_idx = [np.where(game_week == w)[0] for w in weeks]
    game_draws = np.zeros((BOOT_DRAWS, len(names)))
    week_draws = np.zeros((BOOT_DRAWS, len(names)))
    for d in range(BOOT_DRAWS):
        gw = np.zeros(len(game_hook))
        for idx in game_idx:
            gw[idx] = rng.multinomial(len(idx), np.full(len(idx), 1.0 / len(idx)))
        game_draws[d] = poisson_fit(X, y, off, np.repeat(gw, 2), b)[k:]
        if k > 1:
            ww = rng.multinomial(k, np.full(k, 1.0 / k))
            week_draws[d] = poisson_fit(X, y, off, ww[np.searchsorted(weeks, wk)], b)[k:]
    ok = np.isfinite(game_draws).all(1)
    okw = np.isfinite(week_draws).all(1)
    out = {
        "spec": spec,
        "weeks": weeks,
        "games": len(game_hook),
        "hook_games": int((game_hook != 0).sum()),
        "hook_games_per_week": {
            w: int((game_hook[i] != 0).sum()) for w, i in zip(weeks, game_idx, strict=True)
        },
        "best_picks": int(y.sum()),
        "best_picks_on_hook_side": int(y[X[:, -1] > 0].sum()),
        "coefficients": {n: float(v) for n, v in zip(names, b[k:], strict=True)},
        "intervals_game_bootstrap": {n: pct(game_draws[ok, j]) for j, n in enumerate(names)},
        "hook_coefficient": float(b[-1]),
        "hook_rate_ratio": float(np.exp(b[-1])),
        "hook_interval_game_bootstrap": pct(game_draws[ok, -1]),
        "hook_probability_positive": float(probability_positive_from_draws(game_draws[ok, -1])),
        "failed_game_draws": int((~ok).sum()),
    }
    if k > 1:
        out["hook_interval_week_bootstrap"] = pct(week_draws[okw, -1])
        out["hook_probability_positive_weeks"] = float(
            probability_positive_from_draws(week_draws[okw, -1])
        )
        loo = {}
        for w in weeks:
            loo[w] = float(poisson_fit(X, y, off, (wk != w).astype(float), b)[-1])
        out["hook_coefficient_leaving_out_week"] = loo
    if permute:
        Xp = X.copy()
        null = np.zeros(PERM_DRAWS)
        for d in range(PERM_DRAWS):
            gp = game_hook.copy()
            for idx in game_idx:
                gp[idx] = rng.permutation(game_hook[idx])
            Xp[:, -1] = side_flags(gp)
            null[d] = poisson_fit(Xp, y, off, ones, b)[-1]
        out["hook_permutation_p_two_sided"] = float((np.abs(null) >= abs(b[-1])).mean())
        out["hook_permutation_null_interval"] = pct(null)
        out["permutation"] = "game hook status shuffled across games within week"
    return out


def cluster(raw, mask_fn, rng):
    weeks = sorted(raw["week"].unique())
    tot = np.array([raw.loc[raw["week"] == w, "best_picks"].sum() for w in weeks], float)
    tpk = np.array([raw.loc[raw["week"] == w, "picks"].sum() for w in weeks], float)
    m = mask_fn(raw)
    sb = np.array([raw.loc[(raw["week"] == w) & m, "best_picks"].sum() for w in weeks], float)
    sp = np.array([raw.loc[(raw["week"] == w) & m, "picks"].sum() for w in weeks], float)
    point = (sb.sum() / tot.sum(), sp.sum() / tpk.sum())
    draws = []
    for _ in range(BOOT_DRAWS):
        i = rng.integers(0, len(weeks), len(weeks))
        draws.append(sb[i].sum() / tot[i].sum() - sp[i].sum() / tpk[i].sum())
    draws = np.array(draws)
    lo, hi = np.percentile(draws, INTERVAL)
    return {
        "best_pick_share": point[0],
        "pick_share": point[1],
        "excess": point[0] - point[1],
        "excess_interval": [lo, hi],
        "probability_positive": probability_positive_from_draws(draws),
        "weeks_positive": int((sb / tot > sp / tpk).sum()),
        "best_picks": int(sb.sum()),
    }


def record(raw):
    rng = np.random.default_rng(SEED)
    out = {}
    cats = {
        "hook_game": lambda f: f["hook_game"],
        "non_hook_game": lambda f: ~f["hook_game"],
        "hook_side": lambda f: f["hook_side"],
        "other_side_of_hook_game": lambda f: f["hook_game"] & ~f["hook_side"],
        "favourite": lambda f: f["favourite"],
        "underdog": lambda f: ~f["favourite"],
        "hook_side_favourite": lambda f: f["hook_side"] & f["favourite"],
        "hook_side_underdog": lambda f: f["hook_side"] & ~f["favourite"],
    }
    for lo, hi in [(0, 3), (3, 7), (7, 100)]:
        cats[f"size_{lo}_to_{hi}"] = lambda f, lo=lo, hi=hi: (f["size"] >= lo) & (f["size"] < hi)
    for k, fn in cats.items():
        out[k] = cluster(raw, fn, rng)
    return out


def outcome(raw):
    hs = raw[raw["hook_side"] & (raw["best_picks"] > 0)]
    ot = raw[~raw["hook_side"] & (raw["best_picks"] > 0)]

    def wl(f):
        w = int(f.loc[f["covered"], "best_picks"].sum())
        n = int(f["best_picks"].sum())
        return {"best_pick_wins": w, "best_picks": n, "rate": w / n if n else None}

    a, b = wl(hs), wl(ot)
    rng = np.random.default_rng(SEED)
    cov = raw["covered"].to_numpy()
    bp = raw["best_picks"].to_numpy()
    hside = raw["hook_side"].to_numpy()
    wk = raw["week"].to_numpy()
    diffs = []
    for _ in range(PERM_DRAWS):
        pm = hside.copy()
        for w in np.unique(wk):
            idx = np.where(wk == w)[0]
            pm[idx] = rng.permutation(hside[idx])
        n1 = bp[pm].sum()
        n0 = bp[~pm].sum()
        diffs.append((bp[pm & cov].sum() / n1) - (bp[~pm & cov].sum() / n0))
    obs = a["rate"] - b["rate"]
    diffs = np.array(diffs)
    return {
        "hook_side": a,
        "other": b,
        "difference": obs,
        "permutation_p_two_sided": float((np.abs(diffs) >= abs(obs)).mean()),
        "decisive_games_hook_side_best_pick_covered": int(hs["covered"].sum()),
        "games_hook_side_best_pick": len(hs),
    }


def ours(raw):
    files = sorted(SCORING.glob("*/settled_decisions.parquet"))
    d = pd.read_parquet(files[-1])
    d = d[(d["season"] == SEASON) & d["is_best_pick"].fillna(False).astype(bool)]
    d = d.sort_values("recorded_at_utc").groupby("week").tail(1).sort_values("week")
    wk_bp = raw.groupby("week")["best_picks"].sum()
    rows = []
    for _, r in d.iterrows():
        g = raw[raw["game_id"] == r["game_id"]]
        side_home = r["pick_side"] == "HOME"
        row = {
            "week": int(r["week"]),
            "game_id": r["game_id"],
            "side": r["pick_side"],
            "decision_home_spread": float(r["decision_home_spread"]),
        }
        if g.empty:
            row["hook_game"] = None
        else:
            t = g[g["home"] == side_home].iloc[0]
            row.update(
                hook_game=bool(t["hook_game"]),
                hook_side=bool(t["hook_side"]),
                field_best_pick_share_on_game=float(g["best_picks"].sum() / wk_bp[int(t["week"])]),
                field_best_picks_on_our_team=int(t["best_picks"]),
                field_best_picks_on_game=int(g["best_picks"].sum()),
            )
        rows.append(row)
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--last-week", type=int, required=True)
    parser.add_argument("--subset", action="append", default=[])
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    out_path = args.out or OUT_DIR / f"pool10_best_pick_hooks_w{args.last_week:02d}.json"
    raw = team_rows(args.last_week)
    rng = np.random.default_rng(ADJUSTED_SEED)
    fitted = {spec: adjusted(raw, spec, rng, spec == PRIMARY) for spec in SPECS}
    subsets = []
    for s in args.subset:
        lo, hi = (int(v) for v in s.split("-"))
        sub = raw[(raw["week"] >= lo) & (raw["week"] <= hi)]
        cell = adjusted(sub, PRIMARY, rng, True)
        cell["subset"] = s
        if lo < hi:
            cell["hook_side_excess"] = cluster(sub, lambda f: f["hook_side"], rng)
        subsets.append(cell)
    result = {
        "family": FAMILY,
        "last_week": args.last_week,
        "anchor": "served multi-book Tuesday line from tuesday_market_lines",
        "games": int(raw["game_id"].nunique()),
        "hook_games": int(raw.loc[raw["hook_game"], "game_id"].nunique()),
        "total_best_picks": int(raw["best_picks"].sum()),
        "looks": PLACEMENT_LOOKS + OUTCOME_LOOKS + len(SPECS) + 2 * len(subsets),
        "placement": record(raw),
        "outcome": outcome(raw),
        "favourite_adjusted_model": {
            "definition": "Poisson rate of Best Picks per pick on a side, week fixed effects, "
            "offset log picks; equivalent to a within-week multinomial of Best Pick share",
            "primary": PRIMARY,
            "specs": fitted,
            "subsets": subsets,
        },
        "ours": ours(raw),
    }
    out_path.write_text(json.dumps(result, indent=2, default=float) + "\n")
    brief = {k: v for k, v in result.items() if k not in ("placement", "ours")}
    print(json.dumps(brief, indent=1, default=float))
    for k, v in result["placement"].items():
        lo, hi = v["excess_interval"]
        print(
            k,
            f"bp {v['best_pick_share']:.4f} pk {v['pick_share']:.4f} ex {v['excess']:+.4f}",
            f"[{lo:+.4f},{hi:+.4f}] P+ {v['probability_positive']:.3f}",
            f"wk+ {v['weeks_positive']} n {v['best_picks']}",
        )
    print(json.dumps(result["ours"], default=float))


main()
