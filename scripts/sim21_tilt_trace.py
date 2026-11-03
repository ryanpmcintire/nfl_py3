# ruff: noqa: E501, E731, B023, SIM115, RUF005
import glob
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
NAMES = ["el", "td", "fg", "ds"]


def load(label):
    rows, dr = [], []
    for f in glob.glob(str(REPO / "artifacts" / "sim10" / "instrument" / label / "*.jsonl")):
        for ln in open(f, encoding="utf-8"):
            ln = ln.strip()
            if not ln:
                continue
            g = json.loads(ln)
            d = [x for x in g["dr"] if x[0] < 3]
            t = g.get("tl", [])
            if len(d) != len(t):
                continue
            rows.extend(t)
            dr.extend(d)
    return np.array(rows, dtype=float), np.array(dr, dtype=float)


def main(label):
    T, D = load(label)
    sd, secs, hf = T[:, 2], T[:, 3], T[:, 0]
    mk, mh, tg0, tg, ach = (T[:, 6 + 4 * i : 10 + 4 * i] for i in range(5))
    clip = T[:, 26:29]
    real_pts = D[:, 4]
    bins = [("H1", hf == 1)]
    for lo, hi in ((0, 300), (300, 900), (900, 1800)):
        bins.append((f"H2 {lo}-{hi}", (hf == 2) & (secs > lo) & (secs <= hi)))
    signs = [("trail", sd < 0), ("lead", sd > 0), ("tied", sd == 0)]
    out = {"n": len(T)}
    print("n drives", len(T))
    print(
        "sign time n | scoring rate(td+fg): kern head tg0 tg ach realized | el: kern tg0 tg ach real | clip td fg el | ess pool"
    )
    for sn, sm in signs:
        for bn, bm in bins + [("all", np.ones(len(T), bool))]:
            m = sm & bm
            if m.sum() < 200:
                continue
            sr = lambda a: float((a[m, 1] + a[m, 2]).mean())
            print(
                f"{sn} {bn} {int(m.sum())} | {sr(mk):.4f} {sr(mh):.4f} {sr(tg0):.4f} {sr(tg):.4f} {sr(ach):.4f} {float((real_pts[m] > 0).mean()):.4f}"
                f" | {mk[m, 0].mean():.1f} {tg0[m, 0].mean():.1f} {tg[m, 0].mean():.1f} {ach[m, 0].mean():.1f} {D[m, 6].mean():.1f}"
                f" | {clip[m, 1].mean():.4f} {clip[m, 2].mean():.4f} {clip[m, 0].mean():.4f}"
                f" | {T[m, 29].mean():.1f} {np.median(T[m, 5]):.0f}"
            )
            out[f"{sn}|{bn}"] = {
                "n": int(m.sum()),
                "score": {
                    k: sr(a)
                    for k, a in (("kern", mk), ("head", mh), ("tg0", tg0), ("tg", tg), ("ach", ach))
                },
                "real_scored": float((real_pts[m] > 0).mean()),
                "clip": clip[m].mean(axis=0).tolist(),
            }
    for sn, sm in signs:
        m = sm
        sr = lambda a: float((a[m, 1] + a[m, 2]).mean())
        print(
            f"STAGES {sn}: head-kern {sr(mh) - sr(mk):+.4f} | tg0-head {sr(tg0) - sr(mh):+.4f} | dradj tg-tg0 {sr(tg) - sr(tg0):+.4f} | tilt shortfall ach-tg {sr(ach) - sr(tg):+.4f} | total ach-tg0 {sr(ach) - sr(tg0):+.4f}"
            f" | el: head-kern {mh[m, 0].mean() - mk[m, 0].mean():+.2f} tg-tg0 {tg[m, 0].mean() - tg0[m, 0].mean():+.2f} ach-tg {ach[m, 0].mean() - tg[m, 0].mean():+.2f} ach-tg0 {ach[m, 0].mean() - tg0[m, 0].mean():+.2f}"
        )
    (REPO / "artifacts" / "sim21").mkdir(parents=True, exist_ok=True)
    (REPO / "artifacts" / "sim21" / "tilt_trace.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main(sys.argv[1])
