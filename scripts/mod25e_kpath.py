import glob
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
ART = REPO / "artifacts" / "mod25e3"
OUTD = ART / "e82"
EDGES = [-np.inf, -8.5, -0.5, 0.5, 8.5, np.inf]
LABEL = "crHpqokgndecsmfwt"
OUT = []


def say(s):
    OUT.append(s)
    print(s)


def band(v):
    return pd.cut(np.asarray(v, float), EDGES, labels=False).astype(float)


def slope(x, y, q):
    num = den = 0.0
    for k in range(1, 5):
        m = q == k
        if m.sum() < 3:
            continue
        xm, ym = x[m] - x[m].mean(), y[m] - y[m].mean()
        num += (xm * ym).sum()
        den += (xm * xm).sum()
    return num / den


def collect():
    rows = []
    for sd in (11, 12, 13):
        for f in sorted(glob.glob(str(ART / f"e5_{LABEL}_s{sd}" / "play_*_*.parquet"))):
            w, s = (int(v) for v in Path(f).stem.split("_")[1:])
            if s < 2:
                continue
            d = pd.read_parquet(f, columns=["g", "qtr", "offhome", "yl", "sd", "po", "pdf", "code", "gsr", "flip", "el", "down", "dist"])
            d = d.sort_values("g", kind="stable").reset_index(drop=True)
            n = len(d)
            g = d.g.to_numpy()
            po, pdf = d.po.to_numpy(), d.pdf.to_numpy()
            sc = ((po >= 3) & (pdf == 0)) | ((pdf >= 6) & (po == 0))
            nxt = np.r_[g[1:] == g[:-1], False]
            idx = np.flatnonzero(sc & nxt & (d.qtr.to_numpy() <= 4))
            j = idx + 1
            e = pd.DataFrame({"seed": sd, "po": po[idx], "pdf": pdf[idx], "sd0": d.sd.to_numpy()[idx], "q": d.qtr.to_numpy()[idx], "gsr0": d.gsr.to_numpy()[idx], "el": d.el.to_numpy()[idx],
                              "flip": d.flip.to_numpy()[idx], "code": d.code.to_numpy()[idx], "ndown": d.down.to_numpy()[j], "ndist": d.dist.to_numpy()[j], "nyl": d.yl.to_numpy()[j],
                              "nsd": d.sd.to_numpy()[j], "nq": d.qtr.to_numpy()[j], "ngsr": d.gsr.to_numpy()[j], "noff": d.offhome.to_numpy()[j], "off": d.offhome.to_numpy()[idx]})
            rows.append(e)
    return pd.concat(rows, ignore_index=True)


def main():
    E = collect()
    K = pd.read_parquet(ART / "kick" / "kicks_2009.parquet").reset_index(drop=True)
    kd, gs, nfp = K.kd.to_numpy(float), K.gsr.to_numpy(float), K.nfp.to_numpy(float)
    ret, rtd = K.retained.to_numpy(int), K.rtd.to_numpy(int)
    sk, st = kd.std(), gs.std()
    k_off = E.po.to_numpy() > 0
    E["kdv"] = np.where(k_off, E.sd0 + E.po, E.pdf - E.sd0)
    E["t"] = np.maximum(E.gsr0 - E.el, 0.0)
    td = (np.where(k_off, E.po, E.pdf) >= 6)
    say(f"label {LABEL} s11-13 score rows with a next row: {len(E)}; TD {int(td.sum())} FG {int((~td).sum())}")
    say("po/pdf value counts on TD rows: " + ", ".join(f"{k}:{v}" for k, v in pd.Series(np.where(k_off, E.po, E.pdf)[td]).value_counts().sort_index().items()))
    E["td"] = td
    T = E[E.td].iloc[::4].reset_index(drop=True)
    k_off = T.po.to_numpy() > 0
    kdv, t = T.kdv.to_numpy(), T.t.to_numpy()
    m = len(kd)
    mm = int(np.sqrt(m))
    ex = np.empty(len(T))
    exr = np.empty(len(T))
    insup = np.empty(len(T))
    for i in range(len(T)):
        d = ((kd - kdv[i]) / sk) ** 2 + ((gs - t[i]) / st) ** 2
        near = np.argpartition(d, mm - 1)[:mm]
        ex[i] = nfp[near].mean()
        exr[i] = nfp[near][(rtd[near] == 0) & (ret[near] == 0)].mean()
        insup[i] = float(np.any(nfp[near] == T.nyl.iloc[i]))
    T["ex"], T["exr"], T["insup"] = ex, exr, insup
    flip = T.flip.to_numpy() == 1
    xk = 2.0 - band(T.kdv.to_numpy())
    xn = band(T.nsd.to_numpy()) - 2.0
    kicker_home = np.where(k_off, T.off.to_numpy(), 1 - T.off.to_numpy())
    recv_is_off_next = T.noff.to_numpy() != kicker_home
    say(f"TD kicks {len(T)}; next-row offense is receiver {int(recv_is_off_next.sum())} ({recv_is_off_next.mean():.4f}); onside kicker-kept {int((~recv_is_off_next).sum())}; ndown!=1 {int((T.ndown != 1).sum())}; quarter change into q>4 excluded")
    say(f"next sd vs kdv: receiver-next rows with nsd == -kdv: {float((T.nsd.to_numpy()[recv_is_off_next] == -T.kdv.to_numpy()[recv_is_off_next]).mean()):.4f}")
    q = T.nq.to_numpy().astype(int)
    qs = T.q.to_numpy().astype(int)
    say("slope of next start yl on receiver lead band step (quarter-centred), after TD")
    for lab, mask in (("all next rows, x from kdv", np.ones(len(T), bool)), ("receiver-next only, x from kdv", recv_is_off_next), ("receiver-next, x from next sd (E80)", recv_is_off_next)):
        x = xn if "next sd" in lab else xk
        say(f"  {lab}: n {int(mask.sum())} actual {slope(x[mask], T.nyl.to_numpy()[mask], q[mask]):+.3f} expected-draw-mean(all) {slope(x[mask], T.ex.to_numpy()[mask], q[mask]):+.3f} expected(nonret,nontd) {slope(x[mask], T.exr.to_numpy()[mask], q[mask]):+.3f}")
    say("  quarter keyed by score row qtr: actual " + f"{slope(xk, T.nyl.to_numpy(), qs):+.3f} expected {slope(xk, T.ex.to_numpy(), qs):+.3f}")
    say(f"actual next yl mean {T.nyl.mean():.2f} expected draw mean {T.ex.mean():.2f}; share of next yl inside nearest-set support {T.insup.mean():.4f}")
    for b in range(5):
        mb = (xk == b - 2) & recv_is_off_next
        mb2 = (xk == b - 2)
        say(f"  receiver band {b - 2:+d} n {int(mb2.sum())}: actual yl {T.nyl.to_numpy()[mb2].mean():.2f} expected {T.ex.to_numpy()[mb2].mean():.2f}; kicker-kept {int((mb2 & ~recv_is_off_next).sum())}")
    say("per quarter (score-row qtr) actual|expected slope: " + ", ".join(f"Q{k} {slope(xk[qs == k], T.nyl.to_numpy()[qs == k], np.ones((qs == k).sum(), int)):+.2f}|{slope(xk[qs == k], T.ex.to_numpy()[qs == k], np.ones((qs == k).sum(), int)):+.2f}" for k in range(1, 5)))
    say("band +2 receiver-next by t bin (kdv>8 for kicker trailing >8? receiver leads): n, actual yl, expected, in-support share, mean ndist, share ndown==1")
    tb = pd.cut(T.t, [-1, 60, 300, 900, 1800, 3600], labels=False).to_numpy()
    for b in range(5):
        mb = (xk == 2) & recv_is_off_next & (tb == b)
        if mb.sum():
            say(f"  tbin {b} n {int(mb.sum())} actual {T.nyl.to_numpy()[mb].mean():.2f} expected {T.ex.to_numpy()[mb].mean():.2f} insup {T.insup.to_numpy()[mb].mean():.3f} share nyl<=60 actual {float((T.nyl.to_numpy()[mb] <= 60).mean()):.3f} expected-draw share(<=60) n/a")
    mb = (xk == 2) & recv_is_off_next
    say("  band+2 next-yl histogram (10yd bins) actual: " + str(np.histogram(T.nyl.to_numpy()[mb], bins=[0, 20, 40, 60, 70, 75, 79, 80.5, 101])[0] / mb.sum()))
    say(f"  band+2 t quantiles 5/25/50/75/95: {np.percentile(T.t.to_numpy()[mb], [5, 25, 50, 75, 95])}; el of score row quantiles {np.percentile(T.el.to_numpy()[mb], [5, 50, 95]) if 'el' in T else ''}")
    ex_ret = np.empty(len(T))
    for i in range(len(T)):
        d = ((kd - kdv[i]) / sk) ** 2 + ((gs - t[i]) / st) ** 2
        near = np.argpartition(d, mm - 1)[:mm]
        ex_ret[i] = ret[near].mean()
    T["exret"] = ex_ret
    say("onside/kicker-kept share: sim actual vs draw expectation (receiver band, kicker-kept = next offense is kicker)")
    for b in range(5):
        mb2 = (xk == b - 2)
        say(f"  band {b - 2:+d} actual {float((~recv_is_off_next)[mb2].mean()):.4f} draw {T.exret.to_numpy()[mb2].mean():.4f}")
    xe = np.where(recv_is_off_next, xk, band(T.nsd.to_numpy()) - 2.0)
    qr = K.qtr.to_numpy().astype(int)
    xr_k = 2.0 - band(K.kd.to_numpy())
    xr_e80 = np.where(ret == 1, band(K.kd.to_numpy()) - 2.0, xr_k)
    say("E80-style x (band of NEXT offense lead: kicker-kept flips sign) vs E81-style x (receiver always), TD kicks:")
    say(f"  sim actual: E81x {slope(xk, T.nyl.to_numpy(), q):+.3f} E80x {slope(xe, T.nyl.to_numpy(), q):+.3f}; sim draw-expected(all) E81x {slope(xk, T.ex.to_numpy(), q):+.3f}")
    say(f"  real all kicks (E81 pool incl FG): E81x {slope(xr_k, nfp, qr):+.3f} E80x {slope(xr_e80, nfp, qr):+.3f}; retained share {ret.mean():.4f}")
    mo = (xk == 2) & recv_is_off_next & (T.t.to_numpy() < 300) & (T.insup.to_numpy() == 0)
    say(f"band+2 receiver-next t<300 outside draw support: n {int(mo.sum())}; by score-row po/pdf TD points " + str(pd.Series(np.where(k_off, T.po, T.pdf)[mo]).value_counts().to_dict()) + "; yl quantiles 5/25/50/75/95 " + str(np.percentile(T.nyl.to_numpy()[mo], [5, 25, 50, 75, 95])) + "; ndist==10 share " + f"{float((T.ndist.to_numpy()[mo] == 10).mean()):.3f}; ndown1 {float((T.ndown.to_numpy()[mo] == 1).mean()):.3f}; score-row el median {np.median(T.el.to_numpy()[mo]):.1f} vs insup {np.median(T.el.to_numpy()[(xk == 2) & recv_is_off_next & (T.t.to_numpy() < 300) & (T.insup.to_numpy() == 1)]):.1f}")
    mi = (xk == 2) & recv_is_off_next & (T.t.to_numpy() < 300)
    say(f"  same cell all: n {int(mi.sum())} yl<=60 share {float((T.nyl.to_numpy()[mi] <= 60).mean()):.3f}; real kicks with kd<-8.5 (receiver leads >8), gsr<300: n {int(((K.kd < -8.5) & (K.gsr < 300)).sum())} nfp mean {K.nfp[(K.kd < -8.5) & (K.gsr < 300)].mean():.2f} retained {K.retained[(K.kd < -8.5) & (K.gsr < 300)].mean():.3f} share nfp<=60 {float((K.nfp[(K.kd < -8.5) & (K.gsr < 300)] <= 60).mean()):.3f}; sim mean {T.nyl.to_numpy()[mi].mean():.2f}")
    T["diff"] = T.nyl - T.ex
    big = (T["diff"].abs() > 15) & (~(T.nyl == 80))
    say(f"TD rows whose next yl is >15 from the draw expectation: {int(big.sum())} share {big.mean():.4f}")
    say("next yl value shares: 75:{:.3f} 80:{:.3f} <60:{:.3f}".format((T.nyl == 75).mean(), (T.nyl == 80).mean(), (T.nyl < 60).mean()))
    say(f"real kicks nfp: 75 {float((K.nfp == 75).mean()):.3f} 80 {float((K.nfp == 80).mean()):.3f} <60 {float((K.nfp < 60).mean()):.3f}; mean {K.nfp.mean():.2f}")
    (OUTD / "kpath.txt").write_text("\n".join(OUT))


main()
