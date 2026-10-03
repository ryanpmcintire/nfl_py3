import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

os.environ["PASS_MODE"] = "epa"

import mod25d_variance as dv  # noqa: E402
import mod25e_crG as crG  # noqa: E402
import mod25e_crH as crH  # noqa: E402
import mod25e_endgame as eg  # noqa: E402
import mod25e_pass as ps  # noqa: E402
import sim09_f2 as f2  # noqa: E402
import sim09_f3 as f3  # noqa: E402

OUT = REPO / "artifacts" / "mod25e3" / "q4"
HURRY = 120.0
FG_CODE = 3
SPIKE_CODE = 5
STAGES = ("c0", "c_base", "c_f2", "c_eg", "c_f3o")
REC = {}
ROWS = []
DONE = {"x": False}


def flush():
    if "down" in REC:
        ROWS.append(dict(REC))
    REC.clear()


def wrap_f2():
    orig_install = f2.install_f2

    def install():
        ns = dv._G["ns"]
        code = np.asarray(dv._G["tables"]["arrays"]["play_type_code"])
        base = ns["DECIDE"]

        def inner(idx, *a, **k):
            r = base(idx, *a, **k)
            REC["c_base"] = int(code[r])
            return r

        ns["DECIDE"] = inner
        orig_install()
        outer = ns["DECIDE"]

        def after(idx, *a, **k):
            r = outer(idx, *a, **k)
            REC["c_f2"] = int(code[r])
            return r

        ns["DECIDE"] = after

    f2.install_f2 = install


def wrap_eg():
    orig_install = eg.install_eg

    def install():
        orig_install()
        ns = dv._G["ns"]
        code = np.asarray(dv._G["tables"]["arrays"]["play_type_code"])
        dec = ns["DECIDE"]

        def top(idx, rng, tbl, qtr, clock_val, in_ot, down, distance, yardline, score_diff, off_to, def_to, *rest):
            flush()
            hs = clock_val - 1800.0 if qtr == 2 else clock_val
            live = (not in_ot) and down in (1, 2, 3) and qtr in (2, 4) and hs <= HURRY
            if live:
                REC.update(down=int(down), dist=float(distance), yl=float(yardline), sd=float(score_diff), qtr=int(qtr), hs=float(hs), oto=float(off_to), dto=float(def_to), c0=int(code[idx]))
            r = dec(idx, rng, tbl, qtr, clock_val, in_ot, down, distance, yardline, score_diff, off_to, def_to, *rest)
            if live:
                REC["c_eg"] = int(code[r])
                e = dv._G.get("egps")
                if e:
                    REC["ps"] = e[0]
                    REC["psd"] = np.nan if e[1] is None else e[1]
                    dv._G["egps"] = None
            return r

        ns["DECIDE"] = top

    eg.install_eg = install


def wrap_overlay():
    orig = f3.Overlay.apply

    def apply(self, d, st):
        out = orig(self, d, st)
        if "down" in REC:
            REC["c_f3o"] = int(out["play_type_code"])
        return out

    f3.Overlay.apply = apply


def Q_budget(setting):
    crH.weight_once()
    if not DONE["x"]:
        wrap_f2()
        wrap_eg()
        wrap_overlay()
        DONE["x"] = True
    ps.run_init(eg.eg_init_budget, setting)
    if dv._G["cfg"].get("f3"):
        f3.f3_install()


def Q_play_season(task):
    ROWS.clear()
    REC.clear()
    res = dv.d_play_season(task)
    flush()
    pd.DataFrame(ROWS).to_parquet(Path(os.environ["BUD_OUT"]) / f"q4log_{task[0]}_{task[1]}.parquet")
    return res


def cmd_sim(a):
    crH.install()
    crG.crG_budget = Q_budget
    crG.crG_play_season = Q_play_season
    crG.cmd_sim(a)


def window(d, qtr=4, lo=-3.0, hi=0.0, yl=40.0):
    return d[(d.qtr == qtr) & (d.sd >= lo) & (d.sd <= hi) & (d.yl <= yl) & (d.hs <= HURRY)]


def cmd_read(a):
    S = pd.concat([pd.read_parquet(p) for p in sorted(Path(a.out_dir).glob("q4log_*.parquet"))])
    T = eg.train_frame()
    R = T[(T.down <= 3) & (T.qtr.isin((2, 4))) & (T.hs <= HURRY) & (T.hs > 0)].rename(columns={"code": "c_real"})
    L = [f"sim logged snaps {len(S)}; real snaps {len(R)}"]
    for dk in (1, 2, 3):
        s = window(S[S.down == dk])
        r = window(R[R.down == dk])
        L.append(f"down {dk} trailing 0-3 Q4 yl<=40: sim n {len(s)} real n {len(r)}; real fg {np.mean(r.c_real == FG_CODE):.3f} spike {np.mean(r.c_real == SPIKE_CODE):.3f}")
        for c in STAGES:
            if c in s:
                v = s[c].dropna()
                L.append(f"   {c:7s} n {len(v)} fg {np.mean(v == FG_CODE):.3f} spike {np.mean(v == SPIKE_CODE):.3f}")
    txt = chr(10).join(L)
    print(txt)
    (Path(a.out_dir) / "read.txt").write_text(txt)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("sim")
    s.add_argument("--off", default="")
    s.add_argument("--worlds", type=int, default=1)
    s.add_argument("--seasons", type=int, default=2)
    s.add_argument("--workers", type=int, default=2)
    s.add_argument("--seed", type=int, default=31)
    s.add_argument("--out-dir", dest="out_dir", required=True)
    r = sub.add_parser("read")
    r.add_argument("--out-dir", dest="out_dir", required=True)
    a = ap.parse_args()
    {"sim": cmd_sim, "read": cmd_read}[a.cmd](a)


if __name__ == "__main__":
    main()
