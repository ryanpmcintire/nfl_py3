import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(r"F:\Repos\nfl_py3")
SIM = REPO / "artifacts" / "sim10"
DRPLAYS = SIM / "instrument" / "drplays"
OUT = REPO / "artifacts" / "sim22"
SEASONS = (2009, 2017)
STATES = ("trail9", "trail1_8", "tied", "lead1_8", "lead9")
BANDS = ("h1", "q3", "q4", "last5")
GROUPS = {"trailing": ("trail9", "trail1_8"), "leading": ("lead1_8", "lead9")}
Q4_START = 900.0
LAST_FIVE = 300.0


def state_of(m):
    m = np.asarray(m, dtype=float)
    return np.select(
        [m <= -9, m < 0, m == 0, m < 9],
        ["trail9", "trail1_8", "tied", "lead1_8"],
        "lead9",
    )


def band_of(half, secs):
    half, secs = np.asarray(half), np.asarray(secs)
    return np.select(
        [half == 1, secs >= Q4_START, secs >= LAST_FIVE],
        ["h1", "q3", "q4"],
        "last5",
    )


def real_drives():
    seg = pd.read_parquet(SIM / "segments.parquet")
    c = np.load(SIM / "csr.npz")
    seg["plays"] = np.diff(c["pl_ptr"])
    seg = seg[
        (seg["bad"] == 0) & (seg["half"] < 3) & (seg["he"] == 0) & seg["season"].between(*SEASONS)
    ]
    return pd.DataFrame(
        {
            "game": seg["game_id"].to_numpy(),
            "state": state_of(seg["sd"]),
            "band": band_of(seg["half"], seg["secs"]),
            "plays": seg["plays"].to_numpy(float),
            "secs": seg["el"].to_numpy(float),
        }
    )


def sim_drives():
    parts = []
    for f in sorted(DRPLAYS.glob("*.jsonl")):
        for line in f.read_text().splitlines():
            r = json.loads(line)
            dr = np.array(r["dr"], dtype=float)
            dp = np.array(r["dp"], dtype=float)
            if len(dr) != len(dp):
                raise SystemExit(f"misaligned dr/dp in {r['id']}")
            k = (dp[:, 0] < 3) & (dp[:, 8] == 0)
            dp, dr = dp[k], dr[k]
            parts.append(
                pd.DataFrame(
                    {
                        "game": r["id"],
                        "state": state_of(dp[:, 4]),
                        "band": band_of(dp[:, 0], dp[:, 3]),
                        "plays": dp[:, 5],
                        "secs": dr[:, 6],
                    }
                )
            )
    return pd.concat(parts, ignore_index=True)


def cell_masks(d):
    out = {}
    for s in STATES:
        out[(s, "all")] = (d["state"] == s).to_numpy()
        for b in BANDS:
            out[(s, b)] = ((d["state"] == s) & (d["band"] == b)).to_numpy()
    for g, ss in GROUPS.items():
        out[(g, "all")] = d["state"].isin(ss).to_numpy()
        for b in BANDS:
            out[(g, b)] = (d["state"].isin(ss) & (d["band"] == b)).to_numpy()
    out[("all", "all")] = np.ones(len(d), bool)
    return out


def game_table(d, keys):
    codes, games = pd.factorize(d["game"])
    masks = cell_masks(d)
    arr = np.zeros((len(games), len(keys), 3))
    for i, key in enumerate(keys):
        m = masks[key]
        arr[:, i, 0] = np.bincount(codes[m], minlength=len(games))
        arr[:, i, 1] = np.bincount(codes[m], weights=d["plays"].to_numpy()[m], minlength=len(games))
        arr[:, i, 2] = np.bincount(codes[m], weights=d["secs"].to_numpy()[m], minlength=len(games))
    return arr


def quantities(t):
    n, p, s = t[..., 0], t[..., 1], t[..., 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.stack([p / n, s / n, s / p], axis=-1), n


def boot(arr, reps, rng):
    g = len(arr)
    res = np.empty((reps, *arr.shape[1:2], 3))
    for i in range(reps):
        res[i] = quantities(arr[rng.integers(0, g, g)].sum(axis=0))[0]
    return res


def lo(x):
    return np.nanquantile(x, 0.025, axis=0)


def hi(x):
    return np.nanquantile(x, 0.975, axis=0)


def main():
    ap = argparse.ArgumentParser(
        description="Plays per drive, seconds per drive and seconds per play by score state and "
        "half-time band, instrumented simulation against real 2009-2017 regular season"
    )
    ap.add_argument("--reps", type=int, default=1000, help="game-cluster bootstrap replicates")
    ap.add_argument("--seed", type=int, default=22, help="bootstrap seed")
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    real, sim = real_drives(), sim_drives()
    keys = list(cell_masks(real).keys())
    ra, sa = game_table(real, keys), game_table(sim, keys)
    rq, rn = quantities(ra.sum(axis=0))
    sq, sn = quantities(sa.sum(axis=0))
    rb, sb = boot(ra, a.reps, rng), boot(sa, a.reps, rng)
    names = ["plays_per_drive", "secs_per_drive", "secs_per_play"]
    diff = sb - rb
    lg = np.log(sb) - np.log(rb)
    rows = []
    for i, (st, bd) in enumerate(keys):
        row = {"state": st, "band": bd, "real_drives": float(rn[i]), "sim_drives": float(sn[i])}
        for j, q in enumerate(names):
            row[q] = {
                "real": float(rq[i, j]),
                "real_ci": [float(lo(rb)[i, j]), float(hi(rb)[i, j])],
                "sim": float(sq[i, j]),
                "diff": float(sq[i, j] - rq[i, j]),
                "diff_ci": [float(lo(diff)[i, j]), float(hi(diff)[i, j])],
                "p_diff_positive": float((diff[:, i, j] > 0).mean()),
            }
        d_ppd = lg[:, i, 0]
        d_spd = lg[:, i, 1]
        d_spp = lg[:, i, 2]
        row["log_decomp"] = {
            "spp_log_diff": [
                float(np.log(sq[i, 2] / rq[i, 2])),
                float(lo(d_spp[:, None])[0]),
                float(hi(d_spp[:, None])[0]),
            ],
            "from_secs_per_drive": [
                float(np.log(sq[i, 1] / rq[i, 1])),
                float(lo(d_spd[:, None])[0]),
                float(hi(d_spd[:, None])[0]),
            ],
            "from_plays_per_drive_neg": [
                float(-np.log(sq[i, 0] / rq[i, 0])),
                float(lo(-d_ppd[:, None])[0]),
                float(hi(-d_ppd[:, None])[0]),
            ],
        }
        rows.append(row)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "drive_play_decomposition.json").write_text(
        json.dumps({"reps": a.reps, "seed": a.seed, "rows": rows}, indent=1)
    )
    for r in rows:
        print(
            f"{r['state']:9s} {r['band']:6s} nR {r['real_drives']:6.0f} nS {r['sim_drives']:6.0f}",
            " | ".join(
                f"{q[:3]} R {r[q]['real']:.2f} S {r[q]['sim']:.2f} "
                f"d {r[q]['diff']:+.2f} [{r[q]['diff_ci'][0]:+.2f},{r[q]['diff_ci'][1]:+.2f}]"
                for q in names
            ),
        )


if __name__ == "__main__":
    main()
