import inspect
import os
import sys
from pathlib import Path

os.environ.setdefault("A2_LABEL", "crHpqokgndecsmfwtjo2as2")
os.environ.setdefault("A2_SEEDS", "11,12,13")

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
import mod25e_adj2 as a2  # noqa: E402
import mod25e_late as late  # noqa: E402

OUTD = REPO / "artifacts" / "mod25e3" / "e92"
NB = 300
OUT = []
LOOKS = [0]
rng = np.random.default_rng(92)
UNITS = ["off", "def", "cov", "ret"]
PC = ["off_y", "def_y", "cov_y", "ret_y"]
NL = chr(10)


def say(s=""):
    print(s)
    OUT.append(s)


def patch(fn, ns, pairs):
    src = inspect.getsource(fn)
    for a, b in pairs:
        assert src.count(a) == 1, a
        src = src.replace(a, b)
    exec(compile(src, fn.__name__ + "_e92", "exec"), ns)
    return ns[fn.__name__]


RD_INJECT = NL.join([
    'p["lyl"] = p.yardline_100',
    '        pm = np.where(p.penalty == 1, p.penalty_yards.fillna(0.0), 0.0)',
    '        bo = (p.penalty_team == p.posteam).to_numpy()',
    '        bd = (p.penalty_team == p.defteam).to_numpy()',
    '        kk = (p.play_type == "punt").to_numpy()',
    '        p["off_y"] = pm * bo * ~kk',
    '        p["def_y"] = pm * bd * ~kk',
    '        p["cov_y"] = pm * bo * kk',
    '        p["ret_y"] = pm * bd * kk',
])
YCOLS = '"off_y": gp.off_y.sum(), "def_y": gp.def_y.sum(), "cov_y": gp.cov_y.sum(), "ret_y": gp.ret_y.sum()})'

REAL_DRIVES = patch(late.real_play_drives, late.__dict__, [
    ('"yards_gained"]', '"yards_gained", "penalty", "penalty_team", "penalty_yards"]'),
    ('p["lyl"] = p.yardline_100', RD_INJECT),
    ('"yds": gp.yards_gained.sum()})', '"yds": gp.yards_gained.sum(), ' + YCOLS),
])

SD_INJECT = NL.join([
    'd["pp"] = d.po - d.pdf',
    '        ix = d.idx.fillna(-1).astype(int).to_numpy()',
    '        okx = ix >= 0',
    '        ixc = np.where(okx, ix, 0)',
    '        pm = PEN["pm"][ixc] * okx',
    '        bo, bd, kk = PEN["bo"][ixc], PEN["bd"][ixc], PEN["kk"][ixc]',
    '        d["off_y"] = pm * bo * ~kk',
    '        d["def_y"] = pm * bd * ~kk',
    '        d["cov_y"] = pm * bo * kk',
    '        d["ret_y"] = pm * bd * kk',
])

SIM_DRIVES = patch(a2.sim_drives, a2.__dict__, [
    ('"flip"])', '"flip", "idx"])'),
    ('d["pp"] = d.po - d.pdf', SD_INJECT),
    ('"p": gp.pp.sum()})', '"p": gp.pp.sum(), ' + YCOLS),
])


def pool_pen():
    import sim09_f2 as f2

    dv = f2.dv
    dv.sim.PBP_SNAPSHOT_DIR = dv.m25.SNAP
    pbp = dv.sim.load_reg_seasons(tuple(dv.TRAIN))
    tr = dv.sim.build_transition_frame(pbp)
    ep = pd.concat([pd.read_parquet(f"{dv.m25.SNAP}/season={y}/plays.parquet", columns=["game_id", "play_id", "posteam", "defteam", "play_type", "penalty", "penalty_team", "penalty_yards"]) for y in dv.TRAIN]).drop_duplicates(["game_id", "play_id"])
    m = tr[["game_id", "play_id"]].merge(ep, on=["game_id", "play_id"], how="left")
    assert len(m) == len(tr)
    pen = (m.penalty == 1).to_numpy()
    return {"pm": np.where(pen, m.penalty_yards.fillna(0.0), 0.0), "bo": (m.penalty_team == m.posteam).to_numpy(), "bd": (m.penalty_team == m.defteam).to_numpy(), "kk": (m.play_type == "punt").to_numpy(), "pen": pen}


def unit_rows(D, ns):
    H = D[D.h == 1]
    parts = []
    for aoff in (True, False):
        S = (H.s < 0).to_numpy() if aoff else (H.s > 0).to_numpy()
        lead = H.sd0.to_numpy() if aoff else -H.sd0.to_numpy()
        R = pd.DataFrame({"gk": H.gk.to_numpy(), "S": S.astype(int), "q0": H.q0.to_numpy(), "bA": pd.cut(lead, ns["LEAD_EDGES"], labels=False).astype(int)})
        z = np.zeros(len(H))
        o, d_, c, r = (H[k].to_numpy() for k in PC)
        R["A_off"], R["A_def"], R["A_cov"], R["A_ret"] = (-o, z, -c, z) if aoff else (z, -d_, z, -r)
        R["B_off"], R["B_def"], R["B_cov"], R["B_ret"] = (z, d_, z, r) if aoff else (o, z, c, z)
        parts.append(R)
    R = pd.concat(parts, ignore_index=True)
    cols = [f"{a}_{u}" for a in "AB" for u in UNITS]
    R["A_tot"] = R[[f"A_{u}" for u in UNITS]].sum(axis=1)
    R["B_tot"] = R[[f"B_{u}" for u in UNITS]].sum(axis=1)
    R["net"] = R.A_tot + R.B_tot
    cols += ["A_tot", "B_tot", "net"]
    for c in cols:
        R[c] = R[c] - R.groupby(["q0", "bA"])[c].transform("mean")
    return R.groupby(["gk", "S"])[cols].sum().reset_index(), cols


def start_channel(D):
    D = D.copy()
    tow = -D.off_y - D.cov_y + D.def_y + D.ret_y
    tows = -D.off_y + D.def_y
    towk = -D.cov_y + D.ret_y
    gkv = D.gk.to_numpy()
    same = gkv == np.r_[[""], gkv[:-1]]
    pend = pd.Series(np.r_[["x"], D.end.to_numpy()[:-1]]).isin(["punt", "to", "downs", "fgmiss"]).to_numpy()
    ph = np.r_[[0], D.h.to_numpy()[:-1]]
    ok = same & pend & (ph == 1) & (D.h.to_numpy() == 1)
    for nm, v in (("pst", tow), ("psc", tows), ("pkk", towk)):
        D[nm] = np.where(ok, np.r_[[0.0], v.to_numpy()[:-1]], 0.0)
        D[nm] = D[nm] - D.groupby([D.q0.to_numpy(), D.bi.to_numpy()])[nm].transform("mean").to_numpy()
    return D


def boots(G):
    return np.vstack([np.ones((1, G)), rng.multinomial(G, np.ones(G) / G, size=NB).astype(float)])


def main():
    OUTD.mkdir(parents=True, exist_ok=True)
    ns = a2.late.load_xq()
    a2.PEN = pool_pen()
    real = REAL_DRIVES(ns["POOL"])
    sim = pd.concat([SIM_DRIVES(sd) for sd in a2.SEEDS], ignore_index=True)
    rc, _ = ns["components"](real)
    fpmap = pd.Series(rc.p.to_numpy() - rc.p.mean() - rc.c_strength.to_numpy()).groupby(pd.cut(rc.yl0, ns["YL_BINS"], right=False).to_numpy(), observed=True).mean()
    Dr = a2.decorate(ns, real, None, fpmap, None)
    Ds = a2.decorate(ns, sim, None, fpmap, None)
    say(f"E92 penalty channel; label {a2.LABEL} seeds {a2.SEEDS}; real games {Dr.gk.nunique()} sim games {Ds.gk.nunique()}; boots {NB}; no sim run")
    say(f"pool rows with accepted penalty {a2.PEN['pen'].mean():.4f}; y = H2 penalty yards toward A, cells quarter x A-lead-band demeaned, summed per side-game; x = A H1 residual from E88")
    for nm, D in (("real", Dr), ("sim", Ds)):
        say(f"{nm}: H2 penalty yards per drive off {D[D.h == 1].off_y.mean():.3f} def {D[D.h == 1].def_y.mean():.3f} cov {D[D.h == 1].cov_y.mean():.3f} ret {D[D.h == 1].ret_y.mean():.3f}")
    res = {}
    for nm, D in (("real", Dr), ("sim", Ds)):
        U, cols = unit_rows(D, ns)
        T = U.merge(a2.xtable(D), on=["gk", "S"], how="inner")
        gk = pd.Index(sorted(T.gk.unique()))
        res[nm] = a2.slopes(T, cols, boots(len(gk)), gk)
        say(f"{nm}: side-games {len(T)} sd(x) {T.x.std():.2f}")
    R, S = res["real"], res["sim"]
    say("slope of H2 penalty yards toward A per pt of A H1 residual; real | sim | sim-real; pp = P(sim>real); ppos = P(slope>0)")
    for c in R:
        d = S[c] - R[c]
        LOOKS[0] += 1
        say(f"  {c:6s} real {a2.fmt(R[c])} ppos {np.mean(R[c][1:] > 0):.2f} | sim {a2.fmt(S[c])} ppos {np.mean(S[c][1:] > 0):.2f} | s-r {a2.fmt(d)} pp {np.mean(d[1:] > 0):.2f}")
    sc = {}
    for nm, D in (("real", Dr), ("sim", Ds)):
        Dc = start_channel(D)
        H = a2.drive_x(Dc).rename(columns={"xo": "x"})
        gk = pd.Index(sorted(H.gk.unique()))
        sc[nm] = a2.slopes(H, ["ylc", "pst", "psc", "pkk"], boots(len(gk)), gk)
        say(f"{nm}: B H2 drives {len(H)}; share with nonzero prior-drive penalty displacement {(H.pst.abs() > 1e-9).mean():.3f}")
    R, S = sc["real"], sc["sim"]
    say("B H2 start-yardline slope on A H1 residual (yd/pt) and penalty parts: pst = prior A drive penalty yards toward A (spot shift 1:1 for punt, turnover, downs, fg-miss endings), psc scrimmage part, pkk kick-play part")
    for c in R:
        d = S[c] - R[c]
        LOOKS[0] += 1
        say(f"  {c:4s} real {a2.fmt(R[c])} ppos {np.mean(R[c][1:] > 0):.2f} | sim {a2.fmt(S[c])} | s-r {a2.fmt(d)} pp {np.mean(d[1:] > 0):.2f}")
    gap = S["ylc"][0] - R["ylc"][0]
    pg = S["pst"][0] - R["pst"][0]
    say(f"gap sim-real total {gap:+.4f}; through penalties {pg:+.4f} ({100 * pg / gap:.1f}% of gap, point estimate)")
    say(f"looks {LOOKS[0]}")
    (OUTD / "e92.txt").write_text(NL.join(OUT))


if __name__ == "__main__":
    main()
