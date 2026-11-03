# ruff: noqa: E501, SIM115
import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
EDGES = [0.0, 120.0, 300.0, 900.0, 1800.0]
SIGNS = ["trail", "lead", "tied"]
NB = len(EDGES) - 1
NC = 3 * NB
REPS = 1000
SEED = 23


def sign_of(sd):
    return np.where(sd < 0, 0, np.where(sd > 0, 1, 2))


def bin_of(secs):
    return np.clip(np.searchsorted(EDGES, secs, side="left") - 1, 0, NB - 1)


def real_drives():
    p = pd.read_parquet(REPO / "data" / "pbp" / "team_style" / "raw_pbp_narrow.parquet")
    p = p[(p["qtr"] <= 4) & p["posteam"].notna() & p["fixed_drive"].notna() & p["down"].notna()]
    p = p.copy()
    p["half"] = np.where(p["qtr"] <= 2, 1, 2)
    p["kn"] = p["qb_kneel"].fillna(0) == 1
    d = (
        p.groupby(["game_id", "fixed_drive"], sort=False)
        .agg(
            half=("half", "first"),
            sd=("score_differential", "first"),
            gsr=("game_seconds_remaining", "first"),
            res=("fixed_drive_result", "last"),
            kneel=("kn", "any"),
        )
        .reset_index()
    )
    d = d[d["sd"].notna() & d["res"].notna()]
    d["secs"] = np.where(d["half"] == 1, d["gsr"] - 1800.0, d["gsr"])
    d = d[(d["secs"] > 0) & (d["secs"] <= 1800)].sort_values(["game_id", "fixed_drive"])
    last = d.groupby(["game_id", "half"])["fixed_drive"].transform("max")
    d["trunc_sim"] = last == d["fixed_drive"]
    d["eoh"] = d["res"] == "End of half"
    d["scored"] = d["res"].isin(["Touchdown", "Field goal"])
    d["trunc_strict"] = d["eoh"] | d["kneel"]
    d["sg"] = sign_of(d["sd"].to_numpy())
    d["bn"] = bin_of(d["secs"].to_numpy())
    d["cell"] = d["sg"] * NB + d["bn"]
    return d.reset_index(drop=True)


def sim_drives(label):
    rows = []
    gi = 0
    pat = str(REPO / "artifacts" / "sim10" / "instrument" / label / "*.jsonl")
    for f in sorted(glob.glob(pat)):
        for ln in open(f, encoding="utf-8"):
            ln = ln.strip()
            if not ln:
                continue
            g = json.loads(ln)
            dr = [x for x in g["dr"] if x[0] < 3]
            tl = g.get("tl", [])
            if len(dr) != len(tl):
                continue
            for x, t in zip(dr, tl, strict=True):
                rows.append((gi, t[2], x[3], x[4], x[6], x[7]))
            gi += 1
    d = pd.DataFrame(np.array(rows, dtype=float), columns=["gi", "sd", "secs", "pts", "el", "tr"])
    d["scored"] = d["pts"] > 0
    d["trunc_sim"] = d["tr"] == 1
    d["trunc_strict"] = d["trunc_sim"]
    d["eoh"] = False
    d["kneel"] = False
    d["sg"] = sign_of(d["sd"].to_numpy())
    d["bn"] = bin_of(d["secs"].to_numpy())
    d["cell"] = d["sg"] * NB + d["bn"]
    return d


def cell_tab(gid, cell, mask, vals):
    codes, gidx = np.unique(gid, return_inverse=True)
    n = np.zeros((len(codes), NC))
    s = np.zeros((len(codes), NC))
    np.add.at(n, (gidx[mask], cell[mask]), 1.0)
    np.add.at(s, (gidx[mask], cell[mask]), vals[mask])
    return n, s


def rate(m, d, col):
    return float(d.loc[m, col].mean()) if m.any() else None


def summarize(d):
    res = {}
    for si, sn in enumerate(SIGNS):
        m = d["sg"] == si
        row = {"n": int(m.sum()), "score_all": rate(m, d, "scored")}
        for tn in ("trunc_strict", "trunc_sim", "eoh", "kneel"):
            row[f"share_{tn}"] = rate(m, d, tn)
        for tn in ("trunc_strict", "trunc_sim"):
            row[f"score_excl_{tn}"] = rate(m & ~d[tn], d, "scored")
            row[f"score_in_{tn}"] = rate(m & d[tn], d, "scored")
        res[sn] = row
        for bi in range(NB):
            mb = m & (d["bn"] == bi)
            c = {"n": int(mb.sum()), "score_all": rate(mb, d, "scored")}
            for tn in ("trunc_strict", "trunc_sim"):
                c[f"share_{tn}"] = rate(mb, d, tn)
                c[f"score_excl_{tn}"] = rate(mb & ~d[tn], d, "scored")
            res[f"{sn}|{EDGES[bi]:.0f}-{EDGES[bi + 1]:.0f}"] = c
    return res


def ci(a):
    return [float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))]


def main():
    real = real_drives()
    sim = sim_drives("tl_s11")
    out = {
        "looks": "12 cells x (all, excl strict, excl sim-def) + 3 pooled rows + gap bootstrap 3 signs x 3 outputs",
        "real_n": len(real),
        "sim_n": len(sim),
        "real": summarize(real),
        "sim": summarize(sim),
    }
    rg = real["game_id"].to_numpy()
    sg = sim["gi"].to_numpy()
    cr, cs = real["cell"].to_numpy(), sim["cell"].to_numpy()
    rsc, ssc = real["scored"].to_numpy(float), sim["scored"].to_numpy(float)
    keep = {
        "all": (np.ones(len(real), bool), np.ones(len(sim), bool)),
        "excl": (~real["trunc_sim"].to_numpy(), ~sim["trunc_sim"].to_numpy()),
    }
    tabs = {k: (cell_tab(rg, cr, a, rsc), cell_tab(sg, cs, b, ssc)) for k, (a, b) in keep.items()}
    nrg, nsg = tabs["all"][0][0].shape[0], tabs["all"][1][0].shape[0]
    rng = np.random.default_rng(SEED)

    def gaps(ri, si):
        g = {}
        wall = tabs["all"][0][0][ri].sum(0)
        for k, ((rn, rs), (sn_, ss)) in tabs.items():
            rn2, rs2 = rn[ri].sum(0), rs[ri].sum(0)
            sn2, ss2 = sn_[si].sum(0), ss[si].sum(0)
            with np.errstate(invalid="ignore", divide="ignore"):
                rr, sr = rs2 / rn2, ss2 / sn2
            for s_i, name in enumerate(SIGNS):
                sl = slice(s_i * NB, (s_i + 1) * NB)
                ok = np.isfinite(rr[sl]) & np.isfinite(sr[sl]) & (rn2[sl] > 0)
                w = rn2[sl][ok]
                g[(k, name)] = float(np.sum(w * (sr[sl][ok] - rr[sl][ok])) / w.sum())
                g[(k, name, "w")] = float(w.sum() / wall[sl].sum())
        return g

    base = gaps(np.arange(nrg), np.arange(nsg))
    boots = [gaps(rng.integers(0, nrg, nrg), rng.integers(0, nsg, nsg)) for _ in range(REPS)]
    gap = {}
    for name in SIGNS:
        ga = np.array([b[("all", name)] for b in boots])
        gx = np.array([b[("excl", name)] for b in boots])
        wx = np.array([b[("excl", name, "w")] for b in boots])
        part = ga - gx * wx
        fr = part / ga
        b0 = base[("all", name)] - base[("excl", name)] * base[("excl", name, "w")]
        gap[name] = {
            "gap_all": [base[("all", name)], *ci(ga)],
            "gap_excl_trunc": [base[("excl", name)], *ci(gx)],
            "nontrunc_weight": [base[("excl", name, "w")], *ci(wx)],
            "gap_from_trunc_drives": [b0, *ci(part)],
            "frac_explained": [b0 / base[("all", name)], *ci(fr)],
            "prob_gap_all_negative": float(np.mean(ga < 0)),
            "prob_gap_excl_negative": float(np.mean(gx < 0)),
        }
    out["gap"] = gap
    (REPO / "artifacts" / "sim23").mkdir(parents=True, exist_ok=True)
    (REPO / "artifacts" / "sim23" / "truncated_drives.json").write_text(json.dumps(out, indent=1))
    for nm in ("real", "sim"):
        for name in SIGNS:
            print(nm, name, json.dumps(out[nm][name]))
    print(json.dumps(gap, indent=1))


if __name__ == "__main__":
    main()
