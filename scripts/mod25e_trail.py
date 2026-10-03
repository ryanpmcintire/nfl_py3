import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mod25e_late as late  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
BASE = REPO / "artifacts" / "mod25e3" / "crH"
PLAY = BASE / "play6"
OUT = BASE / "trail"
NB = 300
TB = [0, 30, 60, 120, 180, 240, 300]
FB = [0, 40, 60, 80, 101]
SB = [-100, -9, -4, -1, 1]


def prep(D):
    g = D.gk.to_numpy()
    same = np.r_[False, g[1:] == g[:-1]]
    D = D.copy()
    D["prev_end"] = np.where(same, np.r_[["none"], D.end.to_numpy()[:-1]], "start")
    D["prev_burn"] = np.where(same, np.r_[[np.nan], D.burn.to_numpy()[:-1]], np.nan)
    D["prev_n"] = np.where(same, np.r_[[np.nan], D.n.to_numpy()[:-1]], np.nan)
    D["prev_yl"] = np.where(same, np.r_[[np.nan], D.ylast.to_numpy()[:-1]], np.nan)
    return D


def sel(D):
    m = (D.ql.to_numpy() == 4) & (D.gsr0.to_numpy() < 300) & (D.sd0.to_numpy() <= 0)
    R = D[m].copy()
    R["tb"] = np.digitize(R.gsr0.to_numpy(), TB[1:-1])
    R["fb"] = np.digitize(R.yl0.to_numpy(), FB[1:-1])
    R["sb"] = np.digitize(R.sd0.to_numpy(), SB[1:])
    R["sc"] = (R.p > 0).astype(float)
    return R


def mats(R, keys, cells):
    cid = np.array([cells[tuple(r)] for r in R[keys].to_numpy().tolist()])
    gk = pd.factorize(R.gk)[0]
    n = gk.max() + 1
    C = np.zeros((n, len(cells)))
    S = np.zeros((n, len(cells)))
    return gk, cid, C, S, n


def cellsums(R, keys, col, cells):
    gk, cid, C, S, n = mats(R, keys, cells)
    np.add.at(C, (gk, cid), 1.0)
    np.add.at(S, (gk, cid), R[col].to_numpy())
    return C, S


def parts(Cr, Sr, Cs, Ss):
    nr = Cr.sum(0)
    ns = Cs.sum(0)
    ok = (nr > 0) & (ns > 0)
    wr = np.where(ok, nr, 0.0)
    ws = np.where(ok, ns, 0.0)
    wr = wr / wr.sum()
    ws = ws / ws.sum()
    mr = np.where(nr > 0, Sr.sum(0) / np.maximum(nr, 1), 0.0)
    ms = np.where(ns > 0, Ss.sum(0) / np.maximum(ns, 1), 0.0)
    return np.array([(wr * mr).sum() - (ws * ms).sum(), ((wr - ws) * ms).sum(), (wr * (mr - ms)).sum()])


def decomp_boot(Rr, Rs, keys, col, rng):
    cells = {k: i for i, k in enumerate(sorted(set(map(tuple, Rr[keys].to_numpy().tolist())) | set(map(tuple, Rs[keys].to_numpy().tolist()))))}
    Cr, Sr = cellsums(Rr, keys, col, cells)
    Cs, Ss = cellsums(Rs, keys, col, cells)
    est = parts(Cr, Sr, Cs, Ss)
    out = []
    for _ in range(NB):
        ir = rng.integers(0, len(Cr), len(Cr))
        is_ = rng.integers(0, len(Cs), len(Cs))
        out.append(parts(Cr[ir], Sr[ir], Cs[is_], Ss[is_]))
    return est, np.percentile(np.array(out), [2.5, 97.5], axis=0)


def real_plays():
    out = []
    for y in late.POOL:
        p = pd.read_parquet(late.PBP / f"season={y}" / "plays.parquet", columns=["season_type", "qtr", "game_seconds_remaining", "score_differential", "play_type", "yards_gained", "yardline_100", "down", "qb_kneel", "qb_spike", "posteam", "touchdown", "interception", "fumble_lost"])
        p = p[(p.season_type == "REG") & p.posteam.notna() & (p.qtr == 4) & p.play_type.isin(["run", "pass"]) & (p.qb_kneel.fillna(0) == 0) & (p.qb_spike.fillna(0) == 0)]
        out.append(pd.DataFrame({"gsr": p.game_seconds_remaining, "sd": p.score_differential, "pass": (p.play_type == "pass").astype(int), "yd": p.yards_gained, "yl": p.yardline_100, "down": p.down, "td": ((p.touchdown == 1) & (p.interception != 1) & (p.fumble_lost != 1)).astype(int)}))
    return pd.concat(out, ignore_index=True)


def sim_plays():
    out = []
    for f in sorted(PLAY.glob("play_*_*.parquet")):
        w, s = (int(x) for x in f.stem.split("_")[1:])
        if s < late.BURN:
            continue
        d = pd.read_parquet(f, columns=["qtr", "gsr", "sd", "code", "yards", "yl", "down", "po"])
        d = d[(d.qtr == 4) & d.code.isin([0, 1])]
        out.append(pd.DataFrame({"gsr": d.gsr, "sd": d.sd, "pass": d.code, "yd": d.yards.fillna(0), "yl": d.yl, "down": d.down, "td": (d.po >= 6).astype(int)}))
    return pd.concat(out, ignore_index=True)


def play_table(L):
    R = real_plays()
    S = sim_plays()
    L.append("play level Q4 trailing 1-8 or tied, run/pass, by seconds left: n, pass share, ypp, share>=20 yd, share>=40 yd, TD share, TD share when yl>=40 (real/sim)")
    for lo, hi in ((0, 30), (30, 60), (60, 120), (120, 300)):
        row = []
        for D in (R, S):
            d = D[(D.gsr >= lo) & (D.gsr < hi) & (D.sd <= 0) & (D.sd >= -8)]
            far = d[d.yl >= 40]
            row.append(f"n {len(d)} pass {d['pass'].mean():.3f} ypp {d.yd.mean():.2f} g20 {(d.yd >= 20).mean():.3f} g40 {(d.yd >= 40).mean():.3f} td {d.td.mean():.4f} tdfar {far.td.mean():.4f}")
        L.append(f"  [{lo},{hi}) real {row[0]} | sim {row[1]}")


def play_yl(L):
    R = real_plays()
    S = sim_plays()
    bins = [0, 10, 20, 30, 40, 60, 80, 101]
    L.append("snap share by yardline bin 0,10,20,30,40,60,80,100 and TD per play (real/sim), Q4 trailing 1-8 or tied")
    for lo, hi in ((0, 60), (60, 300)):
        for nm, D in (("real", R), ("sim", S)):
            d = D[(D.gsr >= lo) & (D.gsr < hi) & (D.sd <= 0) & (D.sd >= -8)]
            g = d.groupby(pd.cut(d.yl, bins, right=False), observed=False)
            L.append(f"  gsr [{lo},{hi}) {nm} n {len(d)} share " + " ".join(f"{v:.3f}" for v in g.size() / len(d)) + " | td " + " ".join(f"{v:.3f}" for v in g.td.mean()))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    late.SIMD = PLAY
    Dr = prep(late.real_play_drives(late.POOL, regulation=True))
    Ds = prep(late.sim_play_drives())
    Rr = sel(Dr)
    Rs = sel(Ds)
    L = [f"trailer/tied Q4 drives starting last 5 min: real n {len(Rr)} sim n {len(Rs)}"]
    L.append(f"pts/drive real {Rr.p.mean():.3f} sim {Rs.p.mean():.3f}; P(score) real {Rr.sc.mean():.3f} sim {Rs.sc.mean():.3f}")
    rng = np.random.default_rng(0)
    for col in ("p", "sc"):
        L.append(f"-- {col}: total gap (real-sim), composition (start mix, sim outcomes), within (real mix, outcome diff)")
        for nm, keys in (("time", ["tb"]), ("time+score", ["tb", "sb"]), ("time+score+field", ["tb", "sb", "fb"])):
            (t, c, w), ci = decomp_boot(Rr, Rs, keys, col, rng)
            L.append(f"  {nm:18s} total {t:+.3f} comp {c:+.3f} [{ci[0][1]:+.3f},{ci[1][1]:+.3f}] within {w:+.3f} [{ci[0][2]:+.3f},{ci[1][2]:+.3f}]")
    L.append("start mix (share) real vs sim")
    for nm, k in (("time bin", "tb"), ("score bin", "sb"), ("field bin", "fb")):
        a = Rr[k].value_counts(normalize=True).sort_index()
        b = Rs[k].value_counts(normalize=True).sort_index()
        L.append(f"  {nm}: " + " ".join(f"{i}:{a.get(i, 0):.3f}/{b.get(i, 0):.3f}" for i in sorted(set(a.index) | set(b.index))))
    L.append(f"  mean gsr0 real {Rr.gsr0.mean():.1f} sim {Rs.gsr0.mean():.1f}; mean yl0 {Rr.yl0.mean():.1f} {Rs.yl0.mean():.1f}")
    L.append("previous (opponent) drive: end type share, burn, plays, by real/sim")
    for nm, R in (("real", Rr), ("sim", Rs)):
        s = R.prev_end.value_counts(normalize=True)
        L.append(f"  {nm}: " + " ".join(f"{k} {v:.3f}" for k, v in s.items()) + f" | prev burn {R.prev_burn.mean():.1f} prev plays {R.prev_n.mean():.2f} drive burn {R.burn.mean():.1f} plays {R.n.mean():.2f}")
    L.append("within time bin: pts/drive real vs sim, drive burn real vs sim, plays")
    for b in range(len(TB) - 1):
        a = Rr[Rr.tb == b]
        c = Rs[Rs.tb == b]
        L.append(f"  [{TB[b]},{TB[b + 1]}) n {len(a)}/{len(c)} p {a.p.mean():.3f}/{c.p.mean():.3f} sc {a.sc.mean():.3f}/{c.sc.mean():.3f} td {(a.end == 'td').mean():.3f}/{(c.end == 'td').mean():.3f} fg {(a.end == 'fg').mean():.3f}/{(c.end == 'fg').mean():.3f} punt {(a.end == 'punt').mean():.3f}/{(c.end == 'punt').mean():.3f} to+dn {a.end.isin(['to', 'downs']).mean():.3f}/{c.end.isin(['to', 'downs']).mean():.3f} end {(a.end == 'end').mean():.3f}/{(c.end == 'end').mean():.3f} burn {a.burn.mean():.0f}/{c.burn.mean():.0f} plays {a.n.mean():.1f}/{c.n.mean():.1f}")
    play_table(L)
    play_yl(L)
    txt = "\n".join(L)
    (OUT / "trail.txt").write_text(txt)
    print(txt)


if __name__ == "__main__":
    main()
