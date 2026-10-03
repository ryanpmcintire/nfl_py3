import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

import mod25e_drill as dr  # noqa: E402

OUT = REPO / "artifacts" / "mod25e3" / "hurry"
QS = [0.1, 0.25, 0.5, 0.75, 0.9]


def frames(sim_dir):
    R = dr.real_df()
    S = dr.sim_df(Path(sim_dir))
    for D in (R, S):
        D["oc"] = dr.outcome(D)
    return R, S


def window(D, q, lo, hi, trail):
    m = (D.qtr == q) & (D.gsr > lo) & (D.gsr <= hi) & D.oc.isin(["run", "pass_comp", "pass_inc", "sack", "spike", "score_play"])
    if trail:
        m &= D.sd <= 0
    return D[m]


def cmd_el(a):
    OUT.mkdir(parents=True, exist_ok=True)
    R, S = frames(a.sim_dir)
    L = [f"real games {R.g.nunique()} sim games {S.g.nunique()}"]
    for nm, q, lo, hi in (("Q4 last 2:00", 4, 0, 120), ("Q2 last 2:00", 2, 1800, 1920)):
        L.append(f"== {nm}, offence trailing/tied")
        for who, D in (("real", R), ("sim", S)):
            w = window(D, q, lo, hi, True)
            L.append(f"  {who}: n {len(w)} snaps/game {len(w) / D.g.nunique():.3f} yards/snap {w.yards.mean():.2f} el/snap {w.el.mean():.2f} yards/sec {w.yards.sum() / w.el.sum():.4f}")
            for oc, g in w.groupby("oc"):
                qv = " ".join(f"{v:.1f}" for v in g.el.quantile(QS))
                L.append(f"     {oc:10s} share {len(g) / len(w):.3f} yds {g.yards.mean():.2f} el {g.el.mean():.2f} el q10-90 [{qv}] to-used {((g.otu > 0).mean()):.3f}")
    txt = "\n".join(L)
    (OUT / "el.txt").write_text(txt)
    print(txt)


def cmd_cov(a):
    R, S = frames(a.sim_dir)
    cut = float(window(R, 4, 0, 120, True).query("oc == 'pass_inc'").el.quantile(0.9))
    L = [f"stop cut from real incomplete el q90: {cut:.1f}"]
    YB = [-100, 0, 5, 10, 20, 101]
    for nm, q, lo, hi in (("Q4 last 2:00", 4, 0, 120), ("Q2 last 2:00", 2, 1800, 1920)):
        L.append(f"== {nm} trailing/tied: completions and runs, stop share (el<=cut) by yards bin {YB}, corr(yards, el)")
        for oc in ("pass_comp", "run"):
            for who, D in (("real", R), ("sim", S)):
                w = window(D, q, lo, hi, True)
                w = w[(w.oc == oc) & (w.otu == 0) & (w.dtu == 0)]
                st = (w.el <= cut).astype(float)
                gb = w.groupby(pd.cut(w.yards, YB, right=False), observed=False)
                L.append(f"  {oc:9s} {who}: n {len(w)} stop {st.mean():.3f} corr {np.corrcoef(w.yards, w.el)[0, 1]:+.3f} by yards: " + " ".join(f"{v:.3f}/{n}" for v, n in zip(st.groupby(pd.cut(w.yards, YB, right=False), observed=False).mean(), gb.size())))
    L.append("== Q4 trailing/tied by seconds-left band: completions (no timeout) stop share, el when running, el when stopped")
    for lo, hi in ((0, 30), (30, 60), (60, 120)):
        for who, D in (("real", R), ("sim", S)):
            w = window(D, 4, lo, hi, True)
            c = w[(w.oc == "pass_comp") & (w.otu == 0) & (w.dtu == 0)]
            r = w[(w.oc == "run") & (w.otu == 0) & (w.dtu == 0)]
            L.append(f"  [{lo},{hi}) {who}: comp n {len(c)} stop {(c.el <= cut).mean():.3f} el_run {c[c.el > cut].el.mean():.1f} el_stop {c[c.el <= cut].el.mean():.1f} | run n {len(r)} stop {(r.el <= cut).mean():.3f} el_run {r[r.el > cut].el.mean():.1f} | pass share {(w.oc.isin(['pass_comp', 'pass_inc', 'sack'])).mean():.3f} first-down-ish yards>=10 {(c.yards >= 10).mean():.3f}")
    txt = chr(10).join(L)
    (OUT / "cov.txt").write_text(txt)
    print(txt)


def cmd_band(a):
    R, S = frames(a.sim_dir)
    L = ["Q4 trailing/tied snaps per game by seconds-left band and outcome (real/sim); any score diff <=0"]
    for lo, hi in ((0, 15), (15, 30), (30, 60), (60, 120)):
        row = []
        for D in (R, S):
            w = window(D, 4, lo, hi, True)
            n = D.g.nunique()
            row.append((len(w) / n, {k: len(w[w.oc == k]) / n for k in ("pass_comp", "pass_inc", "run", "sack", "spike", "score_play")}, (w.otu > 0).sum() / n))
        L.append(f"  [{lo},{hi}) total {row[0][0]:.4f}/{row[1][0]:.4f} to {row[0][2]:.4f}/{row[1][2]:.4f} " + " ".join(f"{k} {row[0][1][k]:.4f}/{row[1][1][k]:.4f}" for k in row[0][1]))
    txt = chr(10).join(L)
    (OUT / "band.txt").write_text(txt)
    print(txt)


def cmd_fine(a):
    R, S = frames(a.sim_dir)
    L = ["Q4 snaps per game, seconds-left 5 s bins, by score band (real/sim), outcomes: comp inc run sack spike"]
    for sl, slo, shi in (("tied", 0, 0), ("trail1-8", -8, -1), ("trail>8", -99, -9)):
        L.append(f" {sl}")
        for lo in range(0, 30, 5):
            row = []
            for D in (R, S):
                w = window(D, 4, lo, lo + 5, False)
                w = w[(w.sd >= slo) & (w.sd <= shi)]
                n = D.g.nunique()
                row.append(" ".join(f"{len(w[w.oc == k]) / n:.4f}" for k in ("pass_comp", "pass_inc", "run", "sack", "spike")) + f" tot {len(w) / n:.4f}")
            L.append(f"  [{lo},{lo + 5}) real {row[0]} | sim {row[1]}")
    txt = chr(10).join(L)
    (OUT / "fine.txt").write_text(txt)
    print(txt)


def cmd_val(a):
    import mod25e_trail as tr
    R, S = frames(a.sim_dir)
    cut = float(window(R, 4, 0, 120, True).query("oc == 'pass_inc'").el.quantile(0.9))
    L = [f"== spike share by previous-play clock state and offence timeouts (snaps down<=3, trailing/tied, run/pass/spike); previous clock running = el > {cut:.0f}"]
    def prep(D, srt):
        D = D.sort_values(srt, kind="stable").reset_index(drop=True)
        g = D.g.to_numpy()
        same = np.r_[False, g[1:] == g[:-1]]
        D["pel"] = np.where(same, np.r_[np.nan, D.el.to_numpy()[:-1]], np.nan)
        D["prun"] = D.pel > cut
        return D
    R = prep(R, ["g", "play_id"])
    S = prep(S, ["g"])
    for lo, hi in ((15, 30), (30, 60), (60, 120)):
        for nm, D in (("real", R), ("sim", S)):
            w = D[(D.qtr == 4) & (D.gsr > lo) & (D.gsr <= hi) & (D.sd <= 0) & (D.sd >= -24) & (D.down <= 3) & D.code.isin([0, 1, 5]) & D.pel.notna()]
            t = []
            for run in (True, False):
                for to in (True, False):
                    m = w[(w.prun == run) & ((w.oto_ > 0) == to)]
                    t.append(f"run{int(run)} to{int(to)} n/g {len(m) / D.g.nunique():.4f} spike {(m.code == 5).mean():.3f}")
            L.append(f"  [{lo},{hi}) {nm}: " + " | ".join(t))
    tr.PLAY = Path(a.sim_dir)
    tr.OUT = Path(a.sim_dir).parent / (Path(a.sim_dir).name + "_trail")
    tr.late.SIMD = tr.PLAY
    import contextlib
    import io
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        tr.main()
    T = buf.getvalue().splitlines()
    keep = [x for x in T if x.startswith("trailer") or x.startswith("pts/drive") or x.startswith("  [") ]
    L += ["== trail.py drive conditionals (within time bin: real/sim)"] + keep
    txt = chr(10).join(L)
    (OUT / f"val_{Path(a.sim_dir).name}.txt").write_text(txt)
    print(txt)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("el")
    e.add_argument("--sim-dir", dest="sim_dir", default=str(REPO / "artifacts" / "mod25e3" / "crH" / "play6"))
    c = sub.add_parser("cov")
    c.add_argument("--sim-dir", dest="sim_dir", default=str(REPO / "artifacts" / "mod25e3" / "crH" / "play6"))
    b = sub.add_parser("band")
    b.add_argument("--sim-dir", dest="sim_dir", default=str(REPO / "artifacts" / "mod25e3" / "crH" / "play6"))
    f = sub.add_parser("fine")
    f.add_argument("--sim-dir", dest="sim_dir", default=str(REPO / "artifacts" / "mod25e3" / "crH" / "play6"))
    v = sub.add_parser("val")
    v.add_argument("--sim-dir", dest="sim_dir", required=True)
    a = ap.parse_args()
    {"val": cmd_val, "fine": cmd_fine, "el": cmd_el, "cov": cmd_cov, "band": cmd_band}[a.cmd](a)


if __name__ == "__main__":
    main()
