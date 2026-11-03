import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import sky03_matchups as m

OUTD = m.OUT / "separated"
SCRIPT_CACHE = OUTD / "script.parquet"
ITERS = 8
OUTCOME = {"oE": "epa", "dE": "epa", "oN": "epa_ns", "dN": "epa_ns", "oP": "press", "dP": "press"}
NN = {"epa": "n_db", "epa_ns": "n_ns", "press": "n_db"}
SCRIPT_TRAITS = {"epa", "epa_ns"}


def load_script():
    if SCRIPT_CACHE.exists():
        return pd.read_parquet(SCRIPT_CACHE)
    rows = []
    for s in range(m.WARMUP_START, max(m.EVAL_SEASONS) + 1):
        f = sorted(glob.glob(m.PBP_GLOB.format(s=s)))[-1]
        cols = ["game_id", "season_type", "posteam", "qb_dropback"]
        cols += ["qb_kneel", "qb_spike", "epa", "wp"]
        d = pd.read_parquet(f, columns=cols)
        keep = (d.season_type == "REG") & (d.qb_dropback == 1)
        keep &= (d.qb_kneel != 1) & (d.qb_spike != 1)
        d = d[keep & d.epa.notna() & d.posteam.notna() & d.wp.notna()]
        g = d.groupby(["game_id", "posteam"], as_index=False).wp.mean()
        rows.append(g.rename(columns={"posteam": "off", "wp": "sc"}))
    OUTD.mkdir(parents=True, exist_ok=True)
    out = pd.concat(rows, ignore_index=True)
    out.to_parquet(SCRIPT_CACHE)
    return out


def crossed(pool, val, nn, ko, kd, use_script):
    o, ou = pd.factorize(pool.off)
    d, du = pd.factorize(pool.deff)
    w = pool[nn].values.astype(float)
    v = pool[val].values
    mu = float((w * v).sum() / w.sum())
    sc = pool.sc.values
    scc = sc - float((w * sc).sum() / w.sum())
    eo = np.zeros(len(ou))
    ed = np.zeros(len(du))
    gam = 0.0
    for _ in range(ITERS):
        r = v - mu - ed[d] - gam * scc
        eo = np.bincount(o, w * r, len(ou)) / (np.bincount(o, w, len(ou)) + ko)
        r = v - mu - eo[o] - gam * scc
        ed = np.bincount(d, w * r, len(du)) / (np.bincount(d, w, len(du)) + kd)
        if use_script:
            r = v - mu - eo[o] - ed[d]
            gam = float((w * scc * r).sum() / (w * scc**2).sum())
    return pd.Series(eo, ou), pd.Series(ed, du), mu, gam, float((w * sc).sum() / w.sum())


def build_rows(t, seasons):
    parts = []
    for s in seasons:
        k = m.variance_components(t, s)
        for w in sorted(t[t.season == s].week.unique()):
            cur = t[(t.season == s) & (t.week == w)].copy()
            pool = t[((t.season == s) & (t.week < w)) | (t.season == s - 1)]
            assert pool.empty or pool.key.max() < s * 100 + w
            for val in ("epa", "epa_ns", "press"):
                nn = NN[val]
                names = [n for n, v in OUTCOME.items() if v == val]
                on = next(n for n in names if n[0] == "o")
                dn = next(n for n in names if n[0] == "d")
                eo, ed, mu, gam, scm = crossed(pool, val, nn, k[on], k[dn], val in SCRIPT_TRAITS)
                cur[on] = cur.off.map(eo).fillna(0.0)
                cur[dn] = cur.deff.map(ed).fillna(0.0)
                cur["mu_" + val] = mu
                cur["gam_" + val] = gam
                cur["scm_" + val] = scm
            dbm = pool.groupby("off").n_db.mean()
            cur["db_exp"] = cur.off.map(dbm).fillna(dbm.mean())
            cur["ns_share"] = pool.n_ns.sum() / pool.n_db.sum()
            parts.append(cur)
    r = pd.concat(parts, ignore_index=True)
    r["zA"] = r.oP * r.dP
    r["zB"] = r.oN * r.dN
    r["rA"] = r.epa - r.mu_epa - r.oE - r.dE - r.gam_epa * (r.sc - r.scm_epa)
    r["rB"] = r.epa_ns - r.mu_epa_ns - r.oN - r.dN - r.gam_epa_ns * (r.sc - r.scm_epa_ns)
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["rel", "term", "all"], default="all")
    a = ap.parse_args()
    seasons = m.EVAL_SEASONS
    OUTD.mkdir(parents=True, exist_ok=True)
    t = m.load_games().merge(load_script(), on=["game_id", "off"], how="left")
    t["sc"] = t.sc.fillna(0.5)
    rr = build_rows(t, tuple(range(m.TRAIT_START + 1, max(seasons) + 1)))
    res = {
        "rows": len(rr),
        "gamma_epa": float(rr.gam_epa.iloc[-1]),
        "gamma_epa_ns": float(rr.gam_epa_ns.iloc[-1]),
    }
    if a.stage in ("rel", "all"):
        res["reliability"] = m.reliability(rr, tuple(range(m.TRAIT_START + 2, max(seasons) + 1)))
        (OUTD / "rel.json").write_text(json.dumps(res, indent=1))
        print(json.dumps(res, indent=1), flush=True)
    if a.stage in ("term", "all"):
        r = m.fitted_term(rr, seasons)
        (OUTD / "term.json").write_text(json.dumps(r, indent=1))
        print(json.dumps(r, indent=1), flush=True)


if __name__ == "__main__":
    sys.exit(main())
