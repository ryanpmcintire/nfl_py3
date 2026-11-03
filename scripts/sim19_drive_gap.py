import glob
import json
import os

os.environ["OMP_NUM_THREADS"] = "1"
import numpy as np
import pandas as pd

N_BOOT = 300
HALF_SECS = 1800.0
rng = np.random.default_rng(1902)
with open("artifacts/sim19/trailing_risk.json") as fh0:
    edges = json.load(fh0)
DE = np.array(edges["deficit_edges"])
TE = np.array(edges["secs_edges"])
NCOL = len(TE) + 1

rows = []
for p in sorted(glob.glob("artifacts/sim10/instrument/instr_s11/*.jsonl")):
    with open(p) as fh:
        for ln in fh:
            r = json.loads(ln)
            sc = [0.0, 0.0]
            for hf, pos, _yl, secs, po, pd_, el, _tr in r["dr"]:
                if hf < 3:
                    rows.append(
                        (
                            r["id"],
                            sc[1 - int(pos)] - sc[int(pos)],
                            secs + (HALF_SECS if hf == 1 else 0.0),
                            float(po > 0),
                            po,
                            pd_,
                            el,
                        )
                    )
                sc[int(pos)] += po
                sc[1 - int(pos)] += pd_
sim = pd.DataFrame(rows, columns=["gid", "deficit", "gsecs", "score", "pts", "pts_opp", "el"])

d = pd.read_parquet(
    "data/pbp/team_style/raw_pbp_narrow.parquet",
    columns=[
        "game_id",
        "season_type",
        "qtr",
        "fixed_drive",
        "fixed_drive_result",
        "posteam",
        "game_seconds_remaining",
        "score_differential",
    ],
)
d = d[d.season_type.eq("REG") & d.qtr.between(1, 4) & d.posteam.notna() & d.fixed_drive.notna()]
d = d.dropna(subset=["game_seconds_remaining", "score_differential"])
dr = (
    d.sort_values(
        ["game_id", "fixed_drive", "game_seconds_remaining"], ascending=[True, True, False]
    )
    .groupby(["game_id", "fixed_drive"])
    .agg(
        gsecs=("game_seconds_remaining", "first"),
        sd=("score_differential", "first"),
        res=("fixed_drive_result", "last"),
    )
    .reset_index()
    .sort_values(["game_id", "fixed_drive"])
)
nxt = dr.groupby("game_id").gsecs.shift(-1).fillna(0.0)
half_end = np.where(dr.gsecs > HALF_SECS, HALF_SECS, 0.0)
dr["el"] = dr.gsecs - np.maximum(nxt, half_end)
dr["deficit"] = -dr.sd
dr["score"] = dr.res.isin(["Touchdown", "Field goal"]).astype(float)
dr["turnover"] = dr.res.isin(["Turnover", "Opp touchdown"]).astype(float)
dr = dr.rename(columns={"game_id": "gid"})


def table(df, cols):
    df = df[df.deficit > 0].copy()
    df["cell"] = np.digitize(df.deficit, DE) * NCOL + np.digitize(df.gsecs, TE)
    g = pd.factorize(df.gid)[0]
    G = g.max() + 1
    nc = (len(DE) + 1) * NCOL
    cnt = rng.multinomial(G, np.full(G, 1.0 / G), size=N_BOOT).astype(float)
    sx = np.zeros((G, nc))
    np.add.at(sx, (g, df.cell.values), 1.0)
    out = {}
    for c in cols:
        sy = np.zeros((G, nc))
        np.add.at(sy, (g, df.cell.values), df[c].values)
        est = sy.sum(0) / np.maximum(sx.sum(0), 1)
        bs = (cnt @ sy) / np.maximum(cnt @ sx, 1)
        out[c] = (est, bs)
    return sx.sum(0), out


ns, sm = table(sim, ["score", "pts", "el"])
nr, rl = table(dr, ["score", "turnover", "el"])
res = {}
for c in ("score", "el"):
    es, bs_ = sm[c]
    er, br_ = rl[c]
    gap = bs_ - br_
    res[c] = {}
    for i in range(len(es)):
        di, ti = divmod(i, NCOL)
        res[c][f"deficit bin {di} | secs bin {ti}"] = {
            "n_real": int(nr[i]),
            "n_sim": int(ns[i]),
            "real": float(er[i]),
            "sim": float(es[i]),
            "gap_sim_minus_real": float(es[i] - er[i]),
            "gap_ci95": np.percentile(gap[:, i], [2.5, 97.5]).tolist(),
        }
with open("artifacts/sim19/sim_vs_real_drives.json", "w") as fh:
    json.dump(res, fh, indent=1)
for c in res:
    print(c)
    for k, v in res[c].items():
        print(
            " ",
            k,
            v["n_real"],
            v["n_sim"],
            round(v["real"], 3),
            round(v["sim"], 3),
            round(v["gap_sim_minus_real"], 3),
            [round(x, 3) for x in v["gap_ci95"]],
        )
