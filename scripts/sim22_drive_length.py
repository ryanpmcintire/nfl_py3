import json
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(r"F:\Repos\nfl_py3")
SIM = REPO / "artifacts" / "sim10"
OUT = REPO / "artifacts" / "sim22"
LABEL = "instr_s11_s11"
INSTR = SIM / "instrument" / "instr_s11"
SEASONS = (2009, 2017)
BOOT = 2000
STATES = ("trailing", "tied", "leading")


def state_of(sd):
    return np.where(sd < 0, "trailing", np.where(sd > 0, "leading", "tied"))


def real_drives():
    seg = pd.read_parquet(SIM / "segments.parquet")
    c = np.load(SIM / "csr.npz")
    seg["plays"] = np.diff(c["pl_ptr"])
    ev, ptr = c["ev"], c["ev_ptr"]
    k = np.repeat(np.arange(len(seg)), np.diff(ptr))
    seg["pts"] = np.bincount(k, weights=np.maximum(ev[:, 1], 0), minlength=len(seg))
    seg = seg[
        (seg["bad"] == 0) & (seg["half"] < 3) & (seg["he"] == 0) & seg["season"].between(*SEASONS)
    ]
    return pd.DataFrame(
        {
            "game": seg["game_id"].to_numpy(),
            "state": state_of(seg["sd"].to_numpy()),
            "plays": seg["plays"].to_numpy(float),
            "secs": seg["el"].to_numpy(float),
            "scored": (seg["pts"].to_numpy() > 0).astype(float),
            "pts": seg["pts"].to_numpy(float),
        }
    )


def sim_drives():
    pl = pd.read_parquet(SIM / LABEL / "plays.parquet", columns=["game_id", "posteam"])
    g = pl["game_id"].to_numpy()
    p = pl["posteam"].to_numpy()
    chg = np.r_[True, (g[1:] != g[:-1]) | (p[1:] != p[:-1])]
    run_id = np.cumsum(chg) - 1
    runs = pd.DataFrame({"game": g[chg], "team": p[chg], "n": np.bincount(run_id)})
    runs["i"] = runs.groupby("game").cumcount()
    nruns = runs.groupby("game").size()
    rows = []
    nlog = {}
    for f in sorted(INSTR.glob("*.jsonl")):
        for line in f.read_text().splitlines():
            r = json.loads(line)
            dr = np.array(r["dr"], dtype=float)
            nlog[r["id"]] = len(dr)
            rows.append((r["id"], dr))
    rg = runs.groupby("game")
    out = []
    stats = {"games": len(rows), "equal": 0, "unequal": 0}
    for gid, dr in rows:
        dr = dr[dr[:, 0] < 3]
        full = nlog[gid]
        if gid not in nruns.index:
            continue
        r = rg.get_group(gid)
        if len(r) != full:
            stats["unequal"] += 1
            n = np.full(len(dr), np.nan)
        else:
            stats["equal"] += 1
            n = r["n"].to_numpy(float)[: len(dr)]
        lead = np.zeros(len(dr))
        marg = 0.0
        for i in range(len(dr)):
            lead[i] = marg if dr[i, 1] == 0 else -marg
            marg += (dr[i, 4] - dr[i, 5]) * (1.0 if dr[i, 1] == 0 else -1.0)
        out.append(
            pd.DataFrame(
                {
                    "game": gid,
                    "world": gid[:5],
                    "state": state_of(lead),
                    "plays": n,
                    "secs": dr[:, 6],
                    "scored": (dr[:, 4] > 0).astype(float),
                    "pts": dr[:, 4],
                    "trunc": dr[:, 7],
                }
            )
        )
    return pd.concat(out, ignore_index=True), stats


def ratio(d, num, den):
    return d[num].sum() / d[den].sum()


def metrics(d):
    m = d.dropna(subset=["plays"])
    return {
        "drives": float(len(d)),
        "matched_drives": float(len(m)),
        "plays_per_drive": m["plays"].mean(),
        "secs_per_play": ratio(m, "secs", "plays"),
        "secs_per_drive": d["secs"].mean(),
        "scoring_rate": d["scored"].mean(),
        "points_per_drive": d["pts"].mean(),
    }


def cluster_ci(d, rng):
    m = d.dropna(subset=["plays"])
    gm = m.groupby("game").agg(n=("plays", "size"), plays=("plays", "sum"), secs=("secs", "sum"))
    ga = d.groupby("game").agg(
        n=("secs", "size"), secs=("secs", "sum"), scored=("scored", "sum"), pts=("pts", "sum")
    )
    am, aa = gm.to_numpy(float), ga.to_numpy(float)
    res = []
    for _ in range(BOOT):
        x = am[rng.integers(0, len(am), len(am))].sum(axis=0)
        y = aa[rng.integers(0, len(aa), len(aa))].sum(axis=0)
        res.append([x[1] / x[0], x[2] / x[1], y[1] / y[0], y[2] / y[0], y[3] / y[0]])
    res = np.array(res)
    q = np.quantile(res, [0.025, 0.975], axis=0)
    return {
        k: [float(q[0, i]), float(q[1, i])]
        for i, k in enumerate(
            [
                "plays_per_drive",
                "secs_per_play",
                "secs_per_drive",
                "scoring_rate",
                "points_per_drive",
            ]
        )
    }


def main():
    rng = np.random.default_rng(22)
    real = real_drives()
    sim, stats = sim_drives()
    result = {
        "real_zero_play_share": float((real["plays"] == 0).mean()),
        "label": LABEL,
        "seasons": list(SEASONS),
        "match": stats,
        "cells": {},
    }
    for st in STATES:
        r = real[real["state"] == st]
        s = sim[(sim["state"] == st) & (sim["trunc"] == 0)]
        rm, sm = metrics(r), metrics(s)
        rc, sc = cluster_ci(r, rng), cluster_ci(s, rng)
        per_world = {w: metrics(d) for w, d in s.groupby("world")}
        keys = ["plays_per_drive", "secs_per_play", "scoring_rate", "points_per_drive"]
        wsd = {k: float(np.std([v[k] for v in per_world.values()], ddof=1)) for k in keys}
        result["cells"][st] = {
            "real": rm,
            "real_ci": rc,
            "sim": sm,
            "sim_ci": sc,
            "sim_world_sd": wsd,
            "worlds": len(per_world),
        }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "drive_length.json").write_text(json.dumps(result, indent=1))
    for st, c in result["cells"].items():
        for k in ("plays_per_drive", "secs_per_play", "scoring_rate", "points_per_drive"):
            print(
                f"{st:9s} {k:16s} real {c['real'][k]:.4f}"
                f" {c['real_ci'][k][0]:.4f}-{c['real_ci'][k][1]:.4f}"
                f" | sim {c['sim'][k]:.4f} {c['sim_ci'][k][0]:.4f}-{c['sim_ci'][k][1]:.4f}"
                f" worldsd {c['sim_world_sd'][k]:.4f}"
            )
        print(
            st,
            "drives real",
            c["real"]["drives"],
            "sim",
            c["sim"]["drives"],
            "sim matched",
            c["sim"]["matched_drives"],
        )
    print(stats, result["real_zero_play_share"])


if __name__ == "__main__":
    main()
