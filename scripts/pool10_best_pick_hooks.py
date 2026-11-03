import json
from pathlib import Path

import numpy as np
import pandas as pd

from nfl_ats.evidence_conventions import probability_positive_from_draws
from nfl_ats.pool_hook import POOL_OFFSET, is_whole, tuesday_market_lines

FAMILY = "pool10_best_pick_hooks"
OUT = Path("artifacts/pool_rank_replay/pool10_best_pick_hooks.json")
SCORING = Path("artifacts/prospective_scoring")
SEASON = 2026
WEEKS = range(1, 9)
FIELD_FILE = Path("data/splash/field/2026_field_best_picks_w01-08.parquet")
DATA_ROOT = Path("data")
BOOT_DRAWS = 4000
PERM_DRAWS = 20000
SEED = 10
INTERVAL = (2.5, 97.5)


def hook_games(raw):
    ids = raw["game_id"].unique().tolist()
    market = {}
    for w in WEEKS:
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


def team_rows():
    raw = pd.read_parquet(FIELD_FILE)
    raw = raw.merge(hook_games(raw), on="game_id")
    raw["hook_game"] = raw["hook_home"] != 0
    sign = np.where(raw["home"], 1.0, -1.0)
    raw["hook_side"] = raw["hook_game"] & (raw["hook_home"] * sign > 0)
    raw["favourite"] = raw["line"] < 0
    raw["size"] = raw["line"].abs()
    return raw


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
    raw = team_rows()
    result = {
        "family": FAMILY,
        "games": int(raw["game_id"].nunique()),
        "hook_games": int(raw.loc[raw["hook_game"], "game_id"].nunique()),
        "total_best_picks": int(raw["best_picks"].sum()),
        "looks": 11 + 1,
        "placement": record(raw),
        "outcome": outcome(raw),
        "ours": ours(raw),
    }
    OUT.write_text(json.dumps(result, indent=2, default=float) + "\n")
    print(json.dumps(result, indent=1, default=float))


main()
