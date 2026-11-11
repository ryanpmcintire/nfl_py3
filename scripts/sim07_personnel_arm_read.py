import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
WF = REPO / "artifacts" / "sim10_wf" / "T2026"
OUT = REPO / "artifacts" / "sim07" / "personnel_arm"
TD_POINTS = 7.0
FG_POINTS = 3.0
SEEDS = (11, 12, 13)


def slot_table(drives):
    r = pd.read_parquet(WF / "real_games.parquet")
    t = pd.read_parquet(REPO / "artifacts" / "sim07" / "matchup_table_loso.parquet")
    sh = {
        (int(s), int(w), a, b): (x, y)
        for s, w, a, b, x, y in zip(
            t["season"], t["week"], t["team"], t["opp"], t["td_shift"], t["fg_shift"], strict=True
        )
    }
    rows = {}
    for s, d in r.groupby("season"):
        d = d.sort_values(["week", "game_id"]).reset_index(drop=True)
        for gi, x in d.iterrows():
            w = int(x["week"])
            h, a = x["home_team"], x["away_team"]
            z = (0.0, 0.0)
            hs = sh.get((int(s), w, h, a), z)
            as_ = sh.get((int(s), w, a, h), z)
            rows[(int(s), gi)] = (
                w,
                drives * (TD_POINTS * hs[0] + FG_POINTS * hs[1]),
                drives * (TD_POINTS * as_[0] + FG_POINTS * as_[1]),
            )
    return rows


def frame(label, seed, slots):
    g = pd.read_parquet(WF / f"{label}_s{seed}" / "games.parquet")
    gi = g["game_id"].str.extract(r"g(\d+)$")[0].astype(int).to_numpy()
    rs = g["real_season"].astype(int).to_numpy()
    look = [slots[(s, i)] for s, i in zip(rs, gi, strict=True)]
    return pd.DataFrame(
        {
            "game_id": g["game_id"].to_numpy(),
            "week": g["week"].to_numpy(),
            "slot_week": [x[0] for x in look],
            "home_x": [x[1] for x in look],
            "away_x": [x[2] for x in look],
            "home_y": g["home_score"].to_numpy(dtype=float),
            "away_y": g["away_score"].to_numpy(dtype=float),
        }
    )


def stats(x, y):
    return np.array([len(x), x.sum(), y.sum(), (x * x).sum(), (x * y).sum()])


def slope_from(s):
    n, sx, sy, sxx, sxy = s
    return (sxy - sx * sy / n) / (sxx - sx * sx / n)


def cluster_slope(per_game, draws, rng):
    point = slope_from(per_game.sum(axis=0))
    k = len(per_game)
    boot = np.empty(draws)
    for i in range(draws):
        idx = rng.integers(0, k, k)
        boot[i] = slope_from(per_game[idx].sum(axis=0))
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return {
        "slope": float(point),
        "lo": float(lo),
        "hi": float(hi),
        "probability_positive": float((boot > 0).mean()),
        "probability_above_expected": float((boot > 1.0).mean()),
        "boot_sd": float(boot.std()),
    }


def per_game_stats(f, ycol_h, ycol_a):
    xh, xa = f["home_x"].to_numpy(), f["away_x"].to_numpy()
    yh, ya = f[ycol_h].to_numpy(), f[ycol_a].to_numpy()
    return np.column_stack(
        [
            np.full(len(f), 2.0),
            xh + xa,
            yh + ya,
            xh * xh + xa * xa,
            xh * yh + xa * ya,
        ]
    )


def main():
    ap = argparse.ArgumentParser(description="Regress simulated score on the personnel table shift")
    ap.add_argument("--drives", type=float, default=11.0, help="drives per team-game from the lane")
    ap.add_argument("--draws", type=int, default=2000, help="bootstrap draws")
    ap.add_argument("--rng", type=int, default=7, help="bootstrap generator seed")
    a = ap.parse_args()
    slots = slot_table(a.drives)
    rng = np.random.default_rng(a.rng)
    on_all, off_all, pair_all = [], [], []
    checks = {}
    for seed in SEEDS:
        on = frame("pers_on", seed, slots)
        off = frame("pers_off", seed, slots)
        assert (on["game_id"] == off["game_id"]).all()
        assert (on["week"] == on["slot_week"]).all()
        d = on.copy()
        d["hd"] = on["home_y"] - off["home_y"]
        d["ad"] = on["away_y"] - off["away_y"]
        pair_all.append(per_game_stats(d, "hd", "ad"))
        on_all.append(per_game_stats(on, "home_y", "away_y"))
        off_all.append(
            per_game_stats(off.assign(home_x=on["home_x"], away_x=on["away_x"]), "home_y", "away_y")
        )
        gj = {
            k: json.loads((WF / f"{k}_s{seed}" / "gates.json").read_text())["sim"]
            for k in ("pers_on", "pers_off")
        }
        checks[seed] = {
            "games": len(on),
            "games_differ": int(((d["hd"] != 0) | (d["ad"] != 0)).sum()),
            "total_on": float((on["home_y"] + on["away_y"]).mean()),
            "total_off": float((off["home_y"] + off["away_y"]).mean()),
            "margin_sd_on": float((on["home_y"] - on["away_y"]).std()),
            "margin_sd_off": float((off["home_y"] - off["away_y"]).std()),
            "gates_on": gj["pers_on"],
            "gates_off": gj["pers_off"],
        }
    x = np.concatenate(
        [
            np.column_stack([f["home_x"], f["away_x"]]).ravel()
            for f in [frame("pers_on", s, slots) for s in SEEDS]
        ]
    )
    nz = x[x != 0]
    res = {
        "expected_slope": 1.0,
        "drives": a.drives,
        "shift_points_mean": float(x.mean()),
        "shift_points_sd": float(x.std()),
        "shift_points_sd_nonzero": float(nz.std()),
        "nonzero_share": float((x != 0).mean()),
        "on": cluster_slope(np.vstack(on_all), a.draws, rng),
        "off": cluster_slope(np.vstack(off_all), a.draws, rng),
        "paired_on_minus_off": cluster_slope(np.vstack(pair_all), a.draws, rng),
        "checks": checks,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "personnel_arm_read.json").write_text(json.dumps(res, indent=1))
    for k in ("on", "off", "paired_on_minus_off"):
        print(k, {q: round(v, 4) for q, v in res[k].items()})
    print({k: round(v, 4) for k, v in res.items() if k.startswith("shift") or k == "nonzero_share"})
    for s, c in checks.items():
        print(
            s,
            {
                k: (round(v, 3) if not isinstance(v, dict) else None)
                for k, v in c.items()
                if not isinstance(v, dict)
            },
        )
        for g in c["gates_on"]:
            if c["gates_on"][g] != c["gates_off"][g]:
                print("  ", g, round(c["gates_off"][g], 4), "->", round(c["gates_on"][g], 4))


if __name__ == "__main__":
    main()
