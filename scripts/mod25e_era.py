import glob
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mod25_generator as gen

REPO = Path(__file__).resolve().parents[1]
E3 = REPO / "artifacts" / "mod25e3"
OUT = E3 / os.environ.get("ERA_OUT", "era")
VARIANTS = os.environ.get("ERA_VARIANTS", "crf4,crzk,crzhk,u4g,f2,crI").split(",")
GATE_KEYS = ["margin_sd", "nonstrength_var", "r2_w1_4", "r2_w5_9", "r2_w10_18", "mass_3", "mass_7", "mass_10", "mass_14", "mass_17", "pts_game"]
BUDGET_KEYS = ["margin_var", "strength_re_var", "noise_re_var", "xq_cov_sum", "q4_slope", "drives_g"]
POOL = range(2009, 2018)
HELD = range(2018, 2026)
PLAY_COLS = ["game_id", "season", "season_type", "posteam", "play_type", "yards_gained", "down", "yardline_100", "penalty", "fixed_drive", "fixed_drive_result", "qb_kneel", "qb_spike"]
EXPLOSIVE_YARDS = 20


def sim_means(v):
    runs = [json.loads(Path(f).read_text()) for f in sorted(glob.glob(str(E3 / f"e5_{v}_s*" / "e5.json")))]
    out = {}
    for k in GATE_KEYS:
        out[k] = float(np.mean([r["gate"][k] for r in runs]))
    for k in BUDGET_KEYS:
        out[k] = float(np.mean([r["budget"][k] for r in runs]))
    out["n"] = len(runs)
    return out


def season_rates(s):
    f = pd.read_parquet(gen.PBP_DIR / f"season={s}" / "plays.parquet", columns=[c for c in PLAY_COLS if c != "season"])
    f = f[f["season_type"] == "REG"]
    sc = f[f["play_type"].isin(["run", "pass"]) & (f["qb_kneel"].fillna(0) == 0) & (f["qb_spike"].fillna(0) == 0)]
    r = {"pass_rate": float((sc["play_type"] == "pass").mean())}
    r["explosive_rate"] = float((sc["yards_gained"] >= EXPLOSIVE_YARDS).mean())
    r["explosive_pass"] = float((sc.loc[sc["play_type"] == "pass", "yards_gained"] >= EXPLOSIVE_YARDS).mean())
    r["explosive_run"] = float((sc.loc[sc["play_type"] == "run", "yards_gained"] >= EXPLOSIVE_YARDS).mean())
    f4 = f[(f["down"] == 4) & f["play_type"].isin(["run", "pass", "punt", "field_goal"])]
    r["go_rate_4th"] = float(f4["play_type"].isin(["run", "pass"]).mean())
    f4s = f4[f4["yardline_100"] > 50]
    r["go_rate_4th_own_half"] = float(f4s["play_type"].isin(["run", "pass"]).mean())
    d = f[f["posteam"].notna() & f["fixed_drive"].notna()]
    g = d.groupby(["game_id", "posteam", "fixed_drive"]).agg(rz=("yardline_100", lambda x: float((x <= 20).any())), res=("fixed_drive_result", "first")).reset_index()
    rz = g[g["rz"] == 1]
    r["rz_td_rate"] = float((rz["res"] == "Touchdown").mean())
    r["rz_drives_per_game"] = float(len(rz) / f["game_id"].nunique())
    pl = f[f["play_type"].isin(["run", "pass", "punt", "field_goal", "kickoff"])]
    r["penalty_rate_all"] = float(pl["penalty"].fillna(0).mean())
    r["penalty_rate_scrimmage"] = float(sc["penalty"].fillna(0).mean())
    ap = f[f["play_type"].isin(["run", "pass"])]
    r["penalty_plays_per_game"] = float(f["penalty"].fillna(0).sum() / f["game_id"].nunique())
    r["scrimmage_per_game"] = float(len(ap) / f["game_id"].nunique())
    return r


def boot_diff(rows, a, b, k, n=4000, seed=0):
    rng = np.random.default_rng(seed)
    xa = np.array([rows[s][k] for s in a])
    xb = np.array([rows[s][k] for s in b])
    d = np.array([rng.choice(xb, len(xb)).mean() - rng.choice(xa, len(xa)).mean() for _ in range(n)])
    return float(xb.mean() - xa.mean()), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5)), float((d > 0).mean()), float(xa.mean()), float(xb.mean())


def main():
    rm = json.loads((REPO / "artifacts" / "mod25c" / "real_metrics.json").read_text())
    rb = json.loads((E3 / "e5_real.json").read_text())
    real = {"pool": dict(rm["2011_2017"], **rb["real_train"]), "held": dict(rm["2018_2025"], **rb["real_eval"])}
    rg = gen.real_games(tuple(range(2011, 2026)))
    tot = rg["home_score"] + rg["away_score"]
    real["pool"]["pts_game"] = float(tot[rg["season"] <= 2017].mean())
    real["held"]["pts_game"] = float(tot[rg["season"] >= 2018].mean())
    keys = GATE_KEYS + BUDGET_KEYS
    sims = {v: sim_means(v) for v in VARIANTS}
    lines = [f"{'metric':16s} {'pool':>8s} {'held':>8s} " + " ".join(f"{v:>14s}" for v in VARIANTS)]
    for k in keys:
        cells = []
        for v in VARIANTS:
            x = sims[v][k]
            cells.append(f"{x:8.3f}{x - real['pool'][k]:+6.2f}")
        lines.append(f"{k:16s} {real['pool'][k]:8.3f} {real['held'][k]:8.3f} " + " ".join(f"{c:>14s}" for c in cells))
    lines.append("n seeds " + " ".join(f"{v}:{sims[v]['n']}" for v in VARIANTS))
    lines.append("cell = sim mean, then sim minus pool-era real; pool gate seasons 2011-17 (2009-10 warm-up as in sim), pool budget 2009-17, held 2018-25")
    drift = {}
    for k in keys:
        drift[k] = {"pool": real["pool"][k], "held": real["held"][k], "diff": real["held"][k] - real["pool"][k]}
    rows = {s: season_rates(s) for s in list(POOL) + list(HELD)}
    pk = list(rows[2009])
    lines.append("")
    lines.append("era drift (2018-25 minus 2009-17), season bootstrap")
    for k in pk:
        d, lo, hi, pp, a, b = boot_diff(rows, list(POOL), list(HELD), k)
        lines.append(f"{k:24s} {a:9.4f} {b:9.4f} diff {d:+.4f} [{lo:+.4f},{hi:+.4f}] p_pos {pp:.3f}")
        drift["play_" + k] = {"pool": a, "held": b, "diff": d, "lo": lo, "hi": hi, "p_pos": pp}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "era.json").write_text(json.dumps({"real": real, "sims": sims, "drift": drift, "season_rates": {str(s): v for s, v in rows.items()}}, indent=1))
    txt = "\n".join(lines)
    (OUT / "era.txt").write_text(txt)
    print(txt)


if __name__ == "__main__":
    main()
