import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

import mod25e_endgame as eg  # noqa: E402

FG = 3
WIN = 30.0
YL = 40.0
LO, HI = -3.0, 0.0
HURRY = 120.0


def sel(d):
    return d[(d.qtr == 4) & (d.sd >= LO) & (d.sd <= HI) & (d.yl <= YL) & (d.hs <= WIN) & (d.hs > 0) & (d.down <= 3)]


def summ(name, d):
    print(f"{name}: n {len(d)} hs mean {d.hs.mean():.1f} med {d.hs.median():.0f} off_to>0 {np.mean(d.oto > 0):.3f} def_to>0 {np.mean(d.dto > 0):.3f} down {[round(float(np.mean(d.down == k)), 3) for k in (1, 2, 3)]} dist {d.dist.mean():.1f} yl {d.yl.mean():.1f} pstop {d.ps.mean():.3f}")


def main():
    T = eg.train_frame()
    R = T[(T.down <= 3) & (T.qtr == 4) & (T.hs <= HURRY) & (T.hs > 0)].copy()
    R["ps"] = R.pstop
    R["cc"] = R.code
    L = R[R.code.isin(eg.LATE_CODES)]
    clf = eg.fit_hgb(eg.dec_feats(L.down, L.dist, L.yl, L.sd, L.gsr, L.hs, np.ones(len(L)), L.oto, L.dto, L.pstop), L.code.map(eg.CLS).to_numpy())
    fgi = list(clf.classes_).index(eg.CLS[FG])

    def pfg(d, ps):
        X = eg.dec_feats(d.down, d.dist, d.yl, d.sd, d.hs, d.hs, np.ones(len(d)), d.oto, d.dto, ps)
        return clf.predict_proba(X)[:, fgi]

    r = sel(R)
    for tag in ("log2", "log3"):
        S = pd.concat([pd.read_parquet(p) for p in sorted((REPO / "artifacts" / "mod25e3" / "q4" / tag).glob("q4log_*.parquet"))])
        S["ps"] = S["ps"].fillna(1.0)
        s = sel(S)
        print(f"== {tag}")
        summ("real", r)
        summ("sim ", s)
        pr = pfg(r, r.ps.to_numpy())
        ps_ = pfg(s, s.ps.to_numpy())
        er = float(np.mean(r.cc == FG))
        es = float(np.mean(s.c_eg == FG))
        print(f"fg share real {er:.3f}  clf on real states {pr.mean():.3f}  clf on sim states {ps_.mean():.3f}  sim empirical c_eg {es:.3f} c0 {np.mean(s.c0 == FG):.3f} c_f3o {np.mean(s.c_f3o == FG):.3f}")
        print(f"composition {ps_.mean() - pr.mean():+.3f} decision {es - ps_.mean():+.3f} total {es - er:+.3f}")
        for k in ("hs", "oto", "dto", "dist", "yl", "ps"):
            a, b = r[k].to_numpy(float), s[k].to_numpy(float)
            pool = np.concatenate([a, b])
            print(f"  {k}: real {a.mean():.3f} sim {b.mean():.3f} sd-units {(b.mean() - a.mean()) / pool.std():+.3f}")
        for lo, hi in ((0, 6), (6, 12), (12, 20), (20, 30)):
            rr, ss = r[(r.hs > lo) & (r.hs <= hi)], s[(s.hs > lo) & (s.hs <= hi)]
            print(f"  hs ({lo},{hi}] share real {len(rr) / len(r):.3f} sim {len(ss) / len(s):.3f} fg real {np.mean(rr.cc == FG):.3f} sim {np.mean(ss.c_eg == FG):.3f} clf(sim) {pfg(ss, ss.ps.to_numpy()).mean():.3f} clf(real) {pfg(rr, rr.ps.to_numpy()).mean():.3f}")
        for lab, m in (("to>0 off", s.oto > 0), ("to=0 off", s.oto == 0)):
            mr = r.oto > 0 if lab == "to>0 off" else r.oto == 0
            print(f"  {lab}: share real {mr.mean():.3f} sim {m.mean():.3f} fg real {np.mean(r[mr].cc == FG):.3f} sim {np.mean(s[m].c_eg == FG):.3f}")


if __name__ == "__main__":
    main()


def real_pbp(seasons):
    out = []
    for s in seasons:
        p = pd.read_parquet(REPO / "data" / "pbp" / "raw" / "20260925T202544Z" / f"season={s}" / "plays.parquet", columns=["season_type", "qtr", "game_seconds_remaining", "yardline_100", "play_type", "score_differential", "down", "game_id", "play_id", "posteam_timeouts_remaining", "defteam_timeouts_remaining"])
        p = p[(p.season_type == "REG") & p.score_differential.notna() & (p.qtr == 4) & p.play_type.isin(["run", "pass", "punt", "field_goal", "qb_kneel", "qb_spike"])].sort_values(["game_id", "play_id"])
        out.append(p)
    return pd.concat(out, ignore_index=True)


def ending(d, gcol, gsr, code, sd, yl, down, tag):
    d = d.copy()
    d["last"] = d.groupby(gcol).cumcount(ascending=False) == 0
    w = d[(d[gsr] <= WIN) & (d[sd] >= LO) & (d[sd] <= HI) & (d[yl] <= YL) & (d[down] <= 3)]
    n = w.groupby(gcol).ngroups
    g = w.groupby(gcol)
    first = g.head(1)
    lastsnap = g.tail(1)
    print(f"{tag}: games with a window snap {n}; entry gsr mean {first[gsr].mean():.1f}; last-window-snap gsr mean {lastsnap[gsr].mean():.1f} share gsr<=6 {np.mean(lastsnap[gsr] <= 6):.3f} share gsr<2 {np.mean(lastsnap[gsr] < 2):.3f}")
    return w


def cmd_end():
    import glob
    P = pd.concat([pd.read_parquet(f).assign(f=f) for f in sorted(glob.glob(str(REPO / "artifacts" / "mod25e3" / "hurry" / "play_h" / "play_*_*.parquet")))])
    P["gid"] = P.f + P.g.astype(str)
    P = P[P.qtr == 4]
    P["nx"] = P.groupby("gid").gsr.shift(-1)
    P["rem"] = P.gsr - P.el
    w = ending(P, "gid", "gsr", "code", "sd", "yl", "down", "sim")
    fgw = w[w.code == FG]
    print(f"sim window FG snaps {len(fgw)} gsr<2 {np.sum(fgw.gsr < 2)}; window snaps with gsr<2 {np.sum(w.gsr < 2)}; of all Q4 snaps gsr<2: {np.sum(P.gsr < 2)}; el>gsr snaps in window {np.mean(w.el > w.gsr):.3f}")
    print("sim window snaps by gsr bin and code share:")
    for lo, hi in ((0, 2), (2, 6), (6, 12)):
        b = w[(w.gsr > lo) & (w.gsr <= hi)] if lo else w[w.gsr <= hi]
        print(f"  gsr ({lo},{hi}] n {len(b)} fg {np.mean(b.code == FG):.3f} spike {np.mean(b.code == 5):.3f} kneel {np.mean(b.code == 4):.3f}")
    last = P[P.groupby("gid").cumcount(ascending=False) == 0]
    print(f"sim last Q4 snap: gsr<2 {np.mean(last.gsr < 2):.4f}, gsr mean {last.gsr.mean():.1f}")
    R = real_pbp(range(2009, 2018))
    R["gsr"] = R.game_seconds_remaining
    rw = ending(R, "game_id", "gsr", "play_type", "score_differential", "yardline_100", "down", "real")
    rfg = rw[rw.play_type == "field_goal"]
    print(f"real window FG snaps {len(rfg)} gsr<2 {np.sum(rfg.gsr < 2)}; window snaps gsr<2 {np.sum(rw.gsr < 2)}")
    for lo, hi in ((0, 2), (2, 6), (6, 12)):
        b = rw[(rw.gsr > lo) & (rw.gsr <= hi)] if lo else rw[rw.gsr <= hi]
        print(f"  gsr ({lo},{hi}] n {len(b)} fg {np.mean(b.play_type == 'field_goal'):.3f}")
    ent = lambda d, g, gsr: d.groupby(g).head(1)
    print("entry timeouts real (first window snap):", rw.groupby("game_id").head(1).posteam_timeouts_remaining.mean())
