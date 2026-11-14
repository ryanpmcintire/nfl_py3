import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from pool01_forward_sim import WEEKLY_FILE, draw_week, load_games, share_given_key

OUT = Path("artifacts/pool_rank_replay")
LAST_WEEK = int(re.search(r"w01-(\d+)", WEEKLY_FILE.name).group(1))
KEYS = ("p", "share", "u_o", "u_t")
MODES = ("panel", "panel_realised_cover", "actual_slate", "actual_results")
QUANTITIES = (
    "our_points",
    "leader_score",
    "our_rank",
    "gap_to_leader",
    "field_mean",
    "field_sd",
    "tied_at_top",
)


def band(sim, obs):
    lo, hi = np.quantile(sim, [0.05, 0.95])
    sd = float(sim.std(ddof=1))
    return {
        "observed": float(obs),
        "sim_mean": float(sim.mean()),
        "sim_sd": sd,
        "band_90": [float(lo), float(hi)],
        "inside_90": bool(lo <= obs <= hi),
        "percentile": float((sim < obs).mean() + 0.5 * (sim == obs).mean()),
        "z": float((obs - sim.mean()) / sd) if sd > 0 else None,
    }


def measures(op, rp):
    lead = np.maximum(rp.max(axis=-1), op)
    return {
        "our_points": op,
        "leader_score": lead,
        "our_rank": (rp > op[..., None]).sum(axis=-1) + 1.0,
        "gap_to_leader": lead - op,
        "field_mean": rp.mean(axis=-1),
        "field_sd": rp.std(axis=-1, ddof=1),
        "tied_at_top": (rp == rp.max(axis=-1, keepdims=True)).sum(axis=-1).astype(float),
    }


def slates(df, start, last, mode, S):
    past = df[df["week"] <= start]
    out = []
    for w in range(start + 1, last + 1):
        pool = past if mode.startswith("panel") else df[df["week"] == w]
        games = {k: pool[k].to_numpy() for k in KEYS}
        idx = None if mode.startswith("panel") else np.tile(np.arange(len(pool)), (S, 1))
        if mode == "panel_realised_cover":
            games["pc"] = np.full(len(pool), float(past["cov_o"].mean()))
        if mode == "actual_results":
            games["pc"] = pool["cov_o"].to_numpy(float)
        out.append((games, idx))
    return out


def main():
    ap = argparse.ArgumentParser(
        description="Backtest of the forward pool simulation from each earlier week's standings"
    )
    ap.add_argument("--sims", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=20261114)
    ap.add_argument("--weeks", type=int, default=LAST_WEEK, help="last observed week")
    ap.add_argument("--points", type=int, required=True, help="our season points on the page")
    ap.add_argument("--out", default="pool04_forward_backtest.json", help="results file name")
    a = ap.parse_args()
    df = load_games()
    w = pd.read_parquet(WEEKLY_FILE)
    field = w[[f"w{k}" for k in range(1, a.weeks + 1)]].to_numpy(float)
    mine = int(np.where(w["total"].to_numpy() == a.points)[0][0])
    cum = np.delete(field, mine, axis=0).cumsum(axis=1)
    by_week = df[df["week"] <= a.weeks].groupby("week")
    ours = by_week["cov_o"].sum().cumsum().to_numpy(float)
    n_games = by_week.size().to_numpy()
    E = cum.shape[0]
    obs = measures(np.array(ours[-1]), cum[:, -1])
    rng = np.random.default_rng(a.seed)
    starts = list(range(1, a.weeks))
    res = {
        "weekly_file": WEEKLY_FILE.name,
        "last_week": a.weeks,
        "rivals": E,
        "sims": a.sims,
        "start_weeks": starts,
        "looks": len(starts) * len(MODES) * len(QUANTITIES),
        "our_points_by_week_model_sides": ours.tolist(),
        "our_points_on_page": a.points,
        "games_per_week": n_games.tolist(),
        "observed": {q: float(obs[q]) for q in QUANTITIES},
        "modes": {},
    }
    for mode in MODES:
        per_start = {}
        for start in starts:
            past = df[df["week"] <= start]
            rp = np.tile(cum[:, start - 1], (a.sims, 1))
            op = np.full(a.sims, ours[start - 1])
            sim_games = 0
            for games, idx in slates(df, start, a.weeks, mode, a.sims):
                d = draw_week(games, rng, a.sims, E, idx=idx)
                rp = rp + d["pts"]
                op = op + d["cover"].sum(axis=1)
                sim_games += d["cover"].shape[1]
            sim = measures(op, rp)
            cells = {q: band(sim[q], obs[q]) for q in QUANTITIES}
            cells["p_first_points_only"] = float(share_given_key(rp, op).mean())
            cells["simulated_games"] = int(sim_games)
            cells["actual_games"] = int(n_games[start:].sum())
            cells["stated_cover_chance_to_date"] = float(past["p"].mean())
            cells["realised_cover_rate_to_date"] = float(past["cov_o"].mean())
            per_start[str(start)] = cells
        summary = {}
        for q in QUANTITIES:
            z = [per_start[str(s)][q]["z"] for s in starts]
            zz = np.array([v for v in z if v is not None])
            summary[q] = {
                "inside_90": int(sum(per_start[str(s)][q]["inside_90"] for s in starts)),
                "starts": len(starts),
                "observed_above_sim_mean": int((zz > 0).sum()),
                "mean_z": float(zz.mean()) if len(zz) else None,
                "rms_z": float(np.sqrt((zz**2).mean())) if len(zz) else None,
                "z_by_start": z,
            }
        res["modes"][mode] = {"by_start_week": per_start, "summary": summary}
    (OUT / a.out).write_text(json.dumps(res, indent=1) + "\n")
    print("observed", res["observed"], "ours", res["our_points_by_week_model_sides"])
    for mode in MODES:
        print(mode)
        for q, s in res["modes"][mode]["summary"].items():
            zs = [None if v is None else round(v, 2) for v in s["z_by_start"]]
            mz = None if s["mean_z"] is None else round(s["mean_z"], 2)
            print(f"  {q}: inside {s['inside_90']}/{s['starts']} mean_z {mz} z {zs}")
        b = res["modes"][mode]["by_start_week"]
        print("  p_first", [round(b[str(s)]["p_first_points_only"], 4) for s in starts])
        for q in ("our_points", "leader_score", "our_rank", "field_sd"):
            rows = [
                (round(b[str(s)][q]["sim_mean"], 1), [round(v, 1) for v in b[str(s)][q]["band_90"]])
                for s in starts
            ]
            print(f"  {q} sim mean and band by start", rows)
        print(
            "  games sim/actual",
            [(b[str(s)]["simulated_games"], b[str(s)]["actual_games"]) for s in starts],
        )


if __name__ == "__main__":
    main()
