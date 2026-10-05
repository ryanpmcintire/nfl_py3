import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
NB = 200
LABEL = "crHpqokgndecsmfwtjo2"
SEEDS = (11, 12, 13)
ART = REPO / "artifacts" / "mod25e3"
OUT = []
rng = np.random.default_rng(85)


def say(s=""):
    print(s)
    OUT.append(s)


def real_plays():
    fr = []
    for s in range(2009, 2018):
        f = pd.read_parquet(REPO / "data" / "pbp" / "raw" / "20260925T202544Z" / f"season={s}" / "plays.parquet", columns=["game_id", "play_id", "season_type", "week", "home_team", "away_team", "posteam", "play_type", "epa", "qtr"])
        f = f[(f.season_type == "REG") & f.play_type.isin(["run", "pass"]) & (f.qtr <= 4)].copy()
        f["season"] = s
        fr.append(f)
    d = pd.concat(fr, ignore_index=True)
    d["epa"] = d["epa"].fillna(0.0)
    d["dfn"] = np.where(d.posteam == d.home_team, d.away_team, d.home_team)
    d["sk"] = d["season"].astype(str)
    d["gk"] = d["game_id"]
    d["off"] = d["posteam"]
    d["tk"] = d["posteam"]
    d["dk"] = d["dfn"]
    return d.sort_values(["gk", "play_id"], kind="stable")[["gk", "sk", "week", "off", "dfn", "epa"]].rename(columns={"off": "tk", "dfn": "dk"}).reset_index(drop=True)


def sim_plays(sd, epa_by_idx):
    root = ART / f"e5_{LABEL}_s{sd}"
    sg = pd.read_parquet(root / "sim_games.parquet", columns=["game_id", "week", "home_team", "away_team"])
    sg["w"] = sg.game_id.str[1:5].astype(int)
    sg["s"] = sg.game_id.str[6:8].astype(int)
    sg["g"] = sg.game_id.str[-3:].astype(int)
    fr = []
    for (w, s), tm in sg.groupby(["w", "s"]):
        d = pd.read_parquet(root / f"play_{w}_{s}.parquet", columns=["g", "qtr", "offhome", "code", "idx"])
        d = d[(d.qtr <= 4) & d.code.isin([0, 1])].copy()
        t = tm.set_index("g")
        gi = d.g.astype(int).to_numpy()
        ht = t["home_team"].reindex(gi).to_numpy()
        at = t["away_team"].reindex(gi).to_numpy()
        oh = d.offhome.to_numpy() == 1
        out = pd.DataFrame({
            "gk": f"{sd}_{w}_{s}_" + d.g.astype(int).astype(str),
            "sk": f"{sd}_{w}_{s}",
            "week": t["week"].reindex(gi).to_numpy(),
            "tk": np.where(oh, ht, at),
            "dk": np.where(oh, at, ht),
            "epa": epa_by_idx[d.idx.astype(int).to_numpy()],
        })
        out["tk"] = f"{sd}_{w}_" + out["tk"].astype(str)
        out["dk"] = f"{sd}_{w}_" + out["dk"].astype(str)
        fr.append(out)
    return pd.concat(fr, ignore_index=True)


def side_table(d):
    d = d.copy()
    d["ord"] = d.groupby(["gk", "tk"]).cumcount()
    d["odd"] = d["ord"] % 2
    d["a"] = d["epa"] * (d["odd"] == 0)
    d["b"] = d["epa"] * (d["odd"] == 1)
    d["na"] = (d["odd"] == 0).astype(int)
    d["nb"] = (d["odd"] == 1).astype(int)
    g = d.groupby(["gk", "sk", "tk", "dk", "week"], sort=False).agg(sa=("a", "sum"), sb=("b", "sum"), na=("na", "sum"), nb=("nb", "sum")).reset_index()
    g["m"] = (g.sa + g.sb) / (g.na + g.nb)
    g["ma"] = g.sa / g.na.clip(lower=1)
    g["mb"] = g.sb / g.nb.clip(lower=1)
    g = g[(g.na >= 10) & (g.nb >= 10)].reset_index(drop=True)
    lm = g.groupby("sk")["m"].transform("mean")
    for c in ("m", "ma", "mb"):
        g[c] = g[c] - lm
    return g


def loo(g, key, col):
    s = g.groupby(["sk", key])[col].transform("sum")
    k = g.groupby(["sk", key])[col].transform("size")
    return (s - g[col]) / (k - 1).clip(lower=1), k


def adjust(g):
    o_loo, k1 = loo(g, "tk", "m")
    d_loo, k2 = loo(g, "dk", "m")
    od = g[["gk", "tk", "m"]].rename(columns={"tk": "dk", "m": "m_opp"})
    g = g.copy()
    g["o"] = o_loo
    g["d"] = d_loo
    ok = (k1 >= 8) & (k2 >= 8)
    g = g[ok].reset_index(drop=True)
    o_by = g.set_index(["gk", "tk"])["o"]
    opp_o = o_by.reindex(pd.MultiIndex.from_arrays([g["gk"], g["dk"]])).to_numpy()
    for c in ("m", "ma", "mb"):
        g["x_" + c] = g[c] - g["o"] - g["d"]
    g["opp_o"] = opp_o
    return g


def stats(g, idx):
    xa = g["x_ma"].to_numpy()[idx]
    xb = g["x_mb"].to_numpy()[idx]
    x = g["x_m"].to_numpy()[idx]
    return float(np.mean(x * x)), float(np.mean(xa * xb)), float(np.mean(xa * xa)), float(np.mean(xb * xb))


def lag1(g, idx):
    d = g.iloc[idx].sort_values(["sk", "tk", "week"])
    nxt = d.groupby(["sk", "tk"])["x_m"].shift(1)
    ok = nxt.notna().to_numpy()
    return float(np.mean(d["x_m"].to_numpy()[ok] * nxt.to_numpy()[ok])), int(ok.sum())


def report(name, g):
    n = len(g)
    base = stats(g, np.arange(n))
    l1 = lag1(g, np.arange(n))
    skeys = g.sk.to_numpy()
    uniq = np.unique(skeys)
    groups = {u: np.flatnonzero(skeys == u) for u in uniq}
    b = []
    for _ in range(NB):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([groups[u] for u in pick])
        t = stats(g, idx)
        b.append((t[0], t[1], lag1(g, idx)[0]))
    b = np.array(b)
    lo = np.percentile(b, 5, axis=0)
    hi = np.percentile(b, 95, axis=0)
    say(f"{name}: n side-games {n}; var(x) {base[0]:.5f} [{lo[0]:.5f},{hi[0]:.5f}]; true game form cov(odd,even) {base[1]:.5f} [{lo[1]:.5f},{hi[1]:.5f}] = {base[1] / base[0]:.3f} of var; split-half var odd {base[2]:.5f} even {base[3]:.5f}; lag1 autocov {l1[0]:.5f} [{lo[2]:.5f},{hi[2]:.5f}] corr {l1[0] / base[0]:.3f}")
    return base, b


def latent_stats():
    import mod25_generator as gen
    import mod25e_cov as cv

    cv.patch_generator()
    fit = gen.load_fit()
    setting = dict(gen.SETTING_DEFAULTS)
    setting.update({"scale": 1.0, "drift": 1.0})
    shock = fit["qb"]["backup_off_epa_effect"]
    vo, vd, vq, vl = [], [], [], []
    for sd in SEEDS:
        ws = np.random.SeedSequence(sd).spawn(8)
        for w in range(8):
            lat = gen.gen_world_latents(np.random.default_rng(ws[w]), 8, fit, setting)
            for weekly, qb in lat:
                off = weekly[:, :, 0] + shock * qb
                dfn = weekly[:, :, 1]
                od = off - off.mean(axis=0, keepdims=True)
                dd = dfn - dfn.mean(axis=0, keepdims=True)
                od0 = weekly[:, :, 0] - weekly[:, :, 0].mean(axis=0, keepdims=True)
                vo.append(float(np.mean(od ** 2)))
                vd.append(float(np.mean(dd ** 2)))
                vq.append(float(np.mean((od - od0) ** 2)))
                vl.append(float(np.mean(od0 ** 2)))
    return float(np.mean(vo)), float(np.mean(vd)), float(np.mean(vq)), float(np.mean(vl))


def main():
    say("E85 team-form variance, real vs sim (scripts/mod25e_form.py; label %s seeds %s; no sim run)" % (LABEL, SEEDS))
    import sim09_f2 as f2

    dv = f2.dv
    dv.sim.PBP_SNAPSHOT_DIR = dv.m25.SNAP
    pbp = dv.sim.load_reg_seasons(tuple(dv.TRAIN))
    tr = dv.sim.build_transition_frame(pbp)
    ep = pd.concat([pd.read_parquet(f"{dv.m25.SNAP}/season={y}/plays.parquet", columns=["game_id", "play_id", "epa"]) for y in dv.TRAIN]).drop_duplicates(["game_id", "play_id"])
    m = tr[["game_id", "play_id"]].merge(ep, on=["game_id", "play_id"], how="left")
    assert len(m) == len(tr)
    epa_by_idx = m["epa"].fillna(0.0).to_numpy()
    real = adjust(side_table(real_plays()))
    rb, _ = report("real 2009-17", real)
    sims = []
    for sd in SEEDS:
        sims.append(side_table(sim_plays(sd, epa_by_idx)))
    sim = adjust(pd.concat(sims, ignore_index=True))
    sb, _ = report("sim 3 seeds", sim)
    lo, ld, lq, ll = latent_stats()
    say(f"sim latents (EPA/play, per team-season around its own mean, weekly sd): offence dev var {lo:.5f} (AR part {ll:.5f}, backup-QB part {lq:.5f}); defence dev var {ld:.5f}; off+def {lo + ld:.5f}")
    say(f"ratio sim/real true game form {sb[1] / rb[1]:.2f}; excess {sb[1] - rb[1]:+.5f}")
    (ART / "form").mkdir(exist_ok=True)
    (ART / "form" / "form.txt").write_text("\n".join(OUT))


main()
