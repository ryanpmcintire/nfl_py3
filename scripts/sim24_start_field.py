# ruff: noqa: E501, SIM115, E712
import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
SIGNS = ["trail", "lead", "tied"]
CLASSES = ["half", "kick", "nonscore"]
REPS = 500
SEED = 24


def sign_of(sd):
    return np.where(sd < 0, 0, np.where(sd > 0, 1, 2))


def real_drives():
    p = pd.read_parquet(REPO / "data" / "pbp" / "team_style" / "raw_pbp_narrow.parquet")
    p = p[(p["qtr"] <= 4) & p["posteam"].notna() & p["fixed_drive"].notna()].copy()
    p["half"] = np.where(p["qtr"] <= 2, 1, 2)
    p["int1"] = p["interception"].fillna(0) == 1
    p["fum1"] = p["fumble_lost"].fillna(0) == 1
    p = p.sort_values(["game_id", "play_id"])
    sc = p[p["down"].notna()]
    first = sc.groupby(["game_id", "fixed_drive"], sort=False).first()
    d = pd.DataFrame(
        {
            "half": first["half"],
            "sd": first["score_differential"],
            "gsr": first["game_seconds_remaining"],
            "yl": first["yardline_100"],
            "season": first["season"],
        }
    )
    allp = p.groupby(["game_id", "fixed_drive"], sort=False)
    d["res"] = allp["fixed_drive_result"].last()
    d["int"] = allp["int1"].any()
    d["fum"] = allp["fum1"].any()
    d = d.reset_index().sort_values(["game_id", "fixed_drive"]).reset_index(drop=True)
    d = d[d["sd"].notna() & d["res"].notna() & d["yl"].notna()].reset_index(drop=True)
    d["secs"] = np.where(d["half"] == 1, d["gsr"] - 1800.0, d["gsr"])
    d = d[(d["secs"] > 0) & (d["secs"] <= 1800)].reset_index(drop=True)
    lastd = d.groupby(["game_id", "half"])["fixed_drive"].transform("max")
    d["trunc"] = lastd == d["fixed_drive"]
    d["scored"] = d["res"].isin(["Touchdown", "Field goal"])
    g = d.groupby(["game_id", "half"], sort=False)
    d["pres"] = g["res"].shift(1)
    d["pint"] = g["int"].shift(1)
    d["pfum"] = g["fum"].shift(1)
    kick = d["pres"].isin(["Touchdown", "Field goal", "Safety", "Opp touchdown"])
    d["cls"] = np.where(d["pres"].isna(), "half", np.where(kick, "kick", "nonscore"))
    det = d["pres"].astype("object").copy()
    tov = d["pres"] == "Turnover"
    det[tov & (d["pint"] == True)] = "Interception"
    det[tov & (d["pint"] != True) & (d["pfum"] == True)] = "Fumble"
    det[d["pres"].isna()] = "Half start"
    d["det"] = det
    d["sg"] = sign_of(d["sd"].to_numpy())
    d["gid"] = pd.factorize(d["game_id"])[0]
    return d


def sim_drives(label):
    rows = []
    gi = 0
    for f in sorted(
        glob.glob(str(REPO / "artifacts" / "sim10" / "instrument" / label / "*.jsonl"))
    ):
        for ln in open(f, encoding="utf-8"):
            ln = ln.strip()
            if not ln:
                continue
            g = json.loads(ln)
            dr = [x for x in g["dr"] if x[0] < 3]
            tl = g.get("tl", [])
            if len(dr) != len(tl):
                continue
            prev = None
            for x, t in zip(dr, tl, strict=True):
                if prev is None or prev[0] != x[0]:
                    c = 0
                elif prev[4] + prev[5] > 0:
                    c = 1
                else:
                    c = 2
                rows.append((gi, t[2], x[3], x[2], x[4], x[7], c))
                prev = x
            gi += 1
    d = pd.DataFrame(
        np.array(rows, dtype=float), columns=["gid", "sd", "secs", "yl", "pts", "trunc", "ci"]
    )
    d["scored"] = d["pts"] > 0
    d["trunc"] = d["trunc"] == 1
    d["cls"] = np.array(CLASSES)[d["ci"].astype(int)]
    d["sg"] = sign_of(d["sd"].to_numpy())
    return d


def ci(x):
    x = np.asarray(x, float)
    if len(x) < 2:
        return [None, None]
    m = x.mean()
    se = x.std(ddof=1) / np.sqrt(len(x))
    return [float(m - 1.96 * se), float(m + 1.96 * se)]


def cell_rows(d, keys):
    out = {}
    for k, g in d.groupby(keys):
        out["|".join(str(v) for v in (k if isinstance(k, tuple) else (k,)))] = {
            "n": len(g),
            "mean_yl": float(g["yl"].mean()),
            "ci": ci(g["yl"]),
            "score": float(g["scored"].mean()),
        }
    return out


def one(r, s, slopes):
    tot = {}
    for sg in range(3):
        rr, ss = r[r["sg"] == sg], s[s["sg"] == sg]
        if len(rr) == 0 or len(ss) == 0:
            continue
        keyr = rr["cls"] + "|" + rr["tb"].astype(str)
        keys = ss["cls"] + "|" + ss["tb"].astype(str)
        w = keyr.value_counts(normalize=True)
        mr = rr.groupby(keyr)["yl"].mean()
        ms = ss.groupby(keys)["yl"].mean()
        idx = [c for c in w.index if c in ms.index]
        ww = w[idx] / w[idx].sum()
        dy = float((ww * (ms[idx] - mr[idx])).sum())
        tot[sg] = (dy, dy * slopes[sg])
    return tot


def gap_contrib(R, S, slopes, rng, reps):
    base = one(R, S, slopes)
    rg = R.groupby("gid").indices
    sgp = S.groupby("gid").indices
    rk, sk = np.array(list(rg)), np.array(list(sgp))
    bs = {sg: [] for sg in range(3)}
    for _ in range(reps):
        ri = np.concatenate([rg[k] for k in rng.choice(rk, len(rk))])
        si = np.concatenate([sgp[k] for k in rng.choice(sk, len(sk) // 8)])
        for sg, v in one(R.iloc[ri], S.iloc[si], slopes).items():
            bs[sg].append(v)
    res = {}
    for sg in range(3):
        a = np.array(bs[sg])
        res[SIGNS[sg]] = {
            "dy_start": base[sg][0],
            "dy_ci": [float(np.percentile(a[:, 0], 2.5)), float(np.percentile(a[:, 0], 97.5))],
            "gap_explained": base[sg][1],
            "gap_ci": [float(np.percentile(a[:, 1], 2.5)), float(np.percentile(a[:, 1], 97.5))],
            "p_positive": float((a[:, 1] > 0).mean()),
        }
    return res


def main():
    rng = np.random.default_rng(SEED)
    R = real_drives()
    R = R[R["season"] <= 2017].reset_index(drop=True)
    S = sim_drives("tl_s11")
    Rn = R[~R["trunc"]].copy()
    Sn = S[~S["trunc"]].copy()
    edges = np.quantile(Rn["secs"], [0.0, 0.25, 0.5, 0.75, 1.0])
    edges[0], edges[-1] = 0.0, 1800.0
    for d in (Rn, Sn):
        d["tb"] = np.clip(np.searchsorted(edges, d["secs"], side="left") - 1, 0, 3)
        d["sgn"] = np.array(SIGNS)[d["sg"].to_numpy()]
    slopes = {}
    for sg in range(3):
        g = Rn[Rn["sg"] == sg]
        slopes[sg] = float(np.polyfit(g["yl"], g["scored"].astype(float), 1)[0])
    out = {
        "looks": "real detail: 10 prev types x 3 signs; matched: 3 classes x 3 signs x 4 time bins x (real,sim); slopes 3; contribution bootstrap 3 signs",
        "real_n": len(R),
        "sim_n": len(S),
        "time_edges_secs": edges.tolist(),
        "score_slope_per_yard_yardline100": {SIGNS[k]: v for k, v in slopes.items()},
        "real_detail_by_prev_type": cell_rows(Rn, ["det"]),
        "real_detail_by_prev_type_sign": cell_rows(Rn, ["det", "sgn"]),
        "real_class_sign_time": cell_rows(Rn, ["cls", "sgn", "tb"]),
        "sim_class_sign_time": cell_rows(Sn, ["cls", "sgn", "tb"]),
        "real_class_sign": cell_rows(Rn, ["cls", "sgn"]),
        "sim_class_sign": cell_rows(Sn, ["cls", "sgn"]),
        "contribution": gap_contrib(Rn, Sn, slopes, rng, REPS),
    }
    (REPO / "artifacts" / "sim24").mkdir(parents=True, exist_ok=True)
    (REPO / "artifacts" / "sim24" / "start_field.json").write_text(json.dumps(out, indent=1))
    keys = [
        "real_n",
        "sim_n",
        "time_edges_secs",
        "score_slope_per_yard_yardline100",
        "contribution",
    ]
    print(json.dumps({k: out[k] for k in keys}, indent=1))
    for k in ["real_detail_by_prev_type", "real_class_sign", "sim_class_sign"]:
        print(k)
        for c, v in out[k].items():
            print(
                " ",
                c,
                v["n"],
                round(v["mean_yl"], 2),
                [round(x, 2) for x in v["ci"]],
                round(v["score"], 3),
            )


main()
