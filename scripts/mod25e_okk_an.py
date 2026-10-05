import glob
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
ART = REPO / "artifacts" / "mod25e3"
PBP = REPO / "data" / "pbp" / "raw" / "20260925T202544Z"
OUTD = ART / "okk"
LABEL = os.environ.get("OKK_LABEL", "crHpqokgndecsmfwtj")
EDGES = [-np.inf, -8.5, -0.5, 0.5, 8.5, np.inf]
BANDS = ["trail>8", "trail1-8", "tied", "lead1-8", "lead>8"]
TB = ["<=450", "<=900", "rest"]
OUT = []


def say(s=""):
    OUT.append(s)
    print(s)


def tbin(g):
    g = np.asarray(g, float)
    return np.where(g <= 450, 0, np.where(g <= 900, 1, 2))


def cell(kd, g):
    return pd.cut(np.asarray(kd, float), EDGES, labels=False).astype(int) * 3 + tbin(g)


def real_pool():
    K = pd.read_parquet(ART / "kick" / "kicks_2009.parquet").reset_index(drop=True)
    typ = np.full(len(K), "other", dtype=object)
    for s in sorted(K.season.unique()):
        p = pd.read_parquet(PBP / f"season={s}" / "plays.parquet", columns=["game_id", "play_id", "season_type", "touchdown", "play_type"])
        p = p[p.season_type == "REG"].sort_values(["game_id", "play_id"], kind="stable").reset_index(drop=True)
        g = p.game_id.to_numpy()
        pt = p.play_type.to_numpy()
        prev_td = np.r_[False, ((p.touchdown.to_numpy()[:-1] == 1) | (pt[:-1] == "extra_point")) & (g[1:] == g[:-1])]
        prev_fg = np.r_[False, (p.play_type.to_numpy()[:-1] == "field_goal") & (g[1:] == g[:-1])]
        key = pd.DataFrame({"game_id": g, "play_id": p.play_id.to_numpy(), "t": np.where(prev_td, "td", np.where(prev_fg, "fg", "other"))})
        m = K.season == s
        j = K[m][["game_id", "play_id"]].merge(key, on=["game_id", "play_id"], how="left")
        typ[m.to_numpy()] = j.t.fillna("other").to_numpy()
    K["typ"] = typ
    K["cross"] = ((K.gsr > 1800) & (K.ngsr <= 1800)).to_numpy()
    return K


def sim_states():
    rows = []
    for sd in (11, 12, 13):
        for f in sorted(glob.glob(str(ART / f"e5_{LABEL}_s{sd}" / "play_*_*.parquet"))):
            s = int(Path(f).stem.split("_")[2])
            if s < 2:
                continue
            d = pd.read_parquet(f, columns=["g", "qtr", "offhome", "yl", "sd", "po", "pdf", "gsr", "el"]).sort_values("g", kind="stable").reset_index(drop=True)
            g = d.g.to_numpy()
            po, pdf = d.po.to_numpy(), d.pdf.to_numpy()
            sc = ((po >= 3) & (pdf == 0)) | ((pdf >= 6) & (po == 0))
            nxt = np.r_[g[1:] == g[:-1], False]
            idx = np.flatnonzero(sc & nxt & (d.qtr.to_numpy() <= 4))
            j = idx + 1
            off = d.offhome.to_numpy()
            k_off = po[idx] > 0
            kick_home = np.where(k_off, off[idx], 1 - off[idx])
            rows.append(pd.DataFrame({"td": np.maximum(po[idx], pdf[idx]) >= 6, "kdv": np.where(k_off, d.sd.to_numpy()[idx] + po[idx], pdf[idx] - d.sd.to_numpy()[idx]),
                                      "t": np.maximum(d.gsr.to_numpy()[idx] - d.el.to_numpy()[idx], 0.0), "kept": off[j] == kick_home, "nyl": d.yl.to_numpy()[j], "nq": d.qtr.to_numpy()[j], "q": d.qtr.to_numpy()[idx], "cross": (d.qtr.to_numpy()[idx] <= 2) & (d.qtr.to_numpy()[j] >= 3)}))
    return pd.concat(rows, ignore_index=True)


def expected(K, S, mask, step):
    kd, gs = K.kd.to_numpy(float), K.gsr.to_numpy(float)
    sk, st = kd.std(), gs.std()
    ret, nfp = K.retained.to_numpy(float), K.nfp.to_numpy(float)
    idx = np.flatnonzero(mask)
    m = int(np.sqrt(len(idx)))
    S = S.iloc[::step].reset_index(drop=True)
    er, en = np.empty(len(S)), np.empty(len(S))
    for i in range(len(S)):
        d = ((kd[idx] - S.kdv.iat[i]) / sk) ** 2 + ((gs[idx] - S.t.iat[i]) / st) ** 2
        near = idx[np.argpartition(d, m - 1)[:m]]
        er[i], en[i] = ret[near].mean(), nfp[near].mean()
    S["er"], S["en"] = er, en
    return S


def main():
    K = real_pool()
    K["cell"] = cell(K.kd, K.gsr)
    say(f"real pool {len(K)} kicks 2009-17; typ {K.typ.value_counts().to_dict()}; half-crossing (kick gsr>1800, next live play gsr<=1800) {int(K.cross.sum())} of which flagged retained {int(K[K.cross].retained.sum())} ({K[K.cross].retained.mean():.3f}); non-crossing retained rate {K[~K.cross].retained.mean():.4f}; all {K.retained.mean():.4f}")
    nc = K[~K.cross]
    r = nc[(nc.retained == 1) & (nc.rtd == 0)]
    say(f"non-crossing non-return-TD kicker-kept {len(r)}: nfp<40 (muff-type, inferred) {int((r.nfp < 40).sum())}, nfp>=40 (onside-type, inferred) {int((r.nfp >= 40).sum())}; return-TD kicks {int(nc.rtd.sum())} with retained flag {int(nc[nc.rtd == 1].retained.sum())}")
    for t in ("td", "fg", "other"):
        a = K[K.typ == t]
        b = a[~a.cross]
        say(f"  after {t}: n {len(a)} kept raw {a.retained.mean():.4f}; non-crossing n {len(b)} kept {b.retained.mean():.4f}; of kept nfp<40 {int(((b.retained == 1) & (b.rtd == 0) & (b.nfp < 40)).sum())} nfp>=40 {int(((b.retained == 1) & (b.rtd == 0) & (b.nfp >= 40)).sum())}")
    S = sim_states()
    S["cell"] = cell(S.kdv, S.t)
    say(f"sim score states s11-13 with next row {len(S)}; TD {int(S.td.sum())}, FG {int((~S.td).sum())}; half-crossing {int(S.cross.sum())} flagged kept {S[S.cross].kept.mean():.3f}; kept raw TD {S[S.td].kept.mean():.4f} FG {S[~S.td].kept.mean():.4f}; non-crossing TD {S[S.td & ~S.cross].kept.mean():.4f} FG {S[~S.td & ~S.cross].kept.mean():.4f}")
    S0 = S
    S = S[~S.cross].reset_index(drop=True)
    old = expected(K, S, np.ones(len(K), bool), 2)
    fix = expected(K, S, ~K.cross.to_numpy(), 2)
    say("cell: kicker band x time | real kept raw (TD,FG) | real non-crossing (TD,FG) n | sim actual kept (TD,FG) n | draw expected old (TD,FG) | draw expected fixed (TD,FG)")
    for b in range(5):
        for t in range(3):
            c = b * 3 + t
            f = []
            for tp in ("td", "fg"):
                a = K[(K.typ == tp) & (K.cell == c)]
                f.append(a)
            sm = [S[(S.td == v) & (S.cell == c)] for v in (True, False)]
            oc = [old[(old.td == v) & (old.cell == c)] for v in (True, False)] if "td" in old else None
            if min(len(f[0]), len(sm[0])) < 20:
                continue
            fmt = lambda x, col: f"{x[col].mean():.3f}" if len(x) else "  -  "
            say(f"  {BANDS[b]:9s} {TB[t]:6s} | {fmt(f[0], 'retained')} {fmt(f[1], 'retained')} | {fmt(f[0][~f[0].cross], 'retained')} {fmt(f[1][~f[1].cross], 'retained')} {int((~f[0].cross).sum())},{int((~f[1].cross).sum())} | {fmt(sm[0], 'kept')} {fmt(sm[1], 'kept')} {len(sm[0])},{len(sm[1])}")
    for nm, E in (("old", old), ("fixed", fix)):
        say(f"draw expected kept share ({nm}) on sim states: TD {E[E.td].er.mean():.4f} FG {E[~E.td].er.mean():.4f}; mean drawn nfp TD {E[E.td].en.mean():.2f}")
    for b in range(5):
        m = (pd.cut(old.kdv, EDGES, labels=False) == b)
        mf = (pd.cut(fix.kdv, EDGES, labels=False) == b)
        rb = nc[(pd.cut(nc.kd, EDGES, labels=False) == b)]
        sb = S[(pd.cut(S.kdv, EDGES, labels=False) == b)]
        say(f"  band {BANDS[b]:9s} kept: real non-cross all {rb.retained.mean():.4f} | sim actual {sb.kept.mean():.4f} | draw old {old[m].er.mean():.4f} fixed {fix[mf].er.mean():.4f}; mean nfp real {rb.nfp.mean():.2f} draw old {old[m].en.mean():.2f} fixed {fix[mf].en.mean():.2f} sim actual next yl {sb.nyl.mean():.2f}")
    OUTD.mkdir(parents=True, exist_ok=True)
    (OUTD / "okk_an.txt").write_text("\n".join(OUT), encoding="utf-8")


if __name__ == "__main__":
    main()
