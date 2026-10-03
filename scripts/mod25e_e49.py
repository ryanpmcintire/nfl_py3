import argparse
import os
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
for _k, _v in (("EGT", "1"), ("EGH", "1"), ("F2PR", "1"), ("QBC", "1"), ("OTY", "2009-2017"), ("KICK", "1"), ("GZ", "1"), ("SIM_FAST", "1"), ("OMP_NUM_THREADS", "1"), ("OPENBLAS_NUM_THREADS", "1"), ("MKL_NUM_THREADS", "1")):
    os.environ.setdefault(_k, _v)

import mod25d_variance as dv  # noqa: E402
import mod25e_crG as crG  # noqa: E402
import mod25e_crH as H  # noqa: E402

COLS = ["yl", "down", "qtr", "idx", "yg", "flip", "po", "pdf", "nyl", "ndown", "code", "pool_yg", "pool_flip", "pool_po", "pool_pdf", "pool_nyl", "home_off_dummy"]


def install_log():
    ns = dv._G["ns"]
    a = dv._G["tables"]["arrays"]
    pyg = np.asarray(a["yards_gained"], dtype=np.float64)
    pfl = np.asarray(a["possession_flip"]).astype(np.float64)
    ppo = np.asarray(a["points_off"], dtype=np.float64)
    ppd = np.asarray(a["points_def"], dtype=np.float64)
    pny = np.asarray(a["next_yardline"], dtype=np.float64)
    path = Path(os.environ["E49_OUT"]) / f"e49_{os.getpid()}.bin"
    path.parent.mkdir(parents=True, exist_ok=True)
    fh = open(path, "ab")
    inner = dv._G["pol"]

    def p2(down, distance, yardline, score_diff, qtr, clock_val, drawn):
        fr = sys._getframe(1)
        offense = fr.f_locals["offense"]
        off_to = fr.f_locals["off_to"]
        def_to = fr.f_locals["def_to"]
        out = inner(down, distance, yardline, score_diff, qtr, clock_val, drawn)
        i = int(out.get("idx", drawn.get("idx", -1)))
        ok = 0 <= i < len(pyg)
        row = [yardline, down, qtr, i, float(out["yards_gained"]), float(bool(out["flip"])), float(out["points_off"]), float(out["points_def"]), float(out["next_yardline"]), float(out["next_down"]), float(out["play_type_code"]),
               pyg[i] if ok else np.nan, pfl[i] if ok else np.nan, ppo[i] if ok else np.nan, ppd[i] if ok else np.nan, pny[i] if ok else np.nan, 0.0]
        fh.write(np.asarray(row, dtype=np.float64).tobytes())
        fh.flush()
        return out

    dv._G["pol"] = p2


def S(setting):
    H.H_ss(setting)
    install_log()


def cmd_run(a):
    import mod25e_scorestate as ss

    H.install()
    crG.register("")
    dv.DV["crHe49"] = dict(dv.DV["crG"])
    outd = Path(os.environ["E49_OUT"])
    outd.mkdir(parents=True, exist_ok=True)
    os.environ["BUD_OUT"] = str(outd)
    ss.DV["crHe49"] = dv.DV["crHe49"]
    ss.s_init = S
    a.variant = "crHe49"
    a.scale = 1.0
    ss.cmd_e5(a)


def cmd_analyze(a):
    import glob

    X = np.vstack([np.fromfile(f, dtype=np.float64).reshape(-1, len(COLS)) for f in sorted(glob.glob(str(Path(os.environ["E49_OUT"]) / "e49_*.bin")))])
    D = {c: X[:, i] for i, c in enumerate(COLS)}
    n = len(X)
    term = (D["flip"] > 0) | (D["po"] > 0) | (D["pdf"] > 0)
    pterm = (D["pool_flip"] > 0) | (D["pool_po"] > 0) | (D["pool_pdf"] > 0)
    chg = np.abs(D["yg"] - D["pool_yg"]) > 1e-9
    print("draws", n, "pool-terminal share", pterm.mean(), "final-terminal share", term.mean())
    print("pool terminal -> final terminal", int((pterm & term).sum()), "pool terminal -> final non-terminal", int((pterm & ~term).sum()), "pool non-terminal -> final terminal", int((~pterm & term).sum()))
    neg = (~term) & (D["yg"] < -20)
    print("final non-terminal with yards_gained < -20:", int(neg.sum()), "share", neg.mean(), "of which pool-terminal", int((neg & pterm).sum()), "yards changed vs pool", int((neg & chg).sum()))
    t = ~term & pterm
    if t.any():
        print("pool-terminal but final non-terminal: code counts", {int(k): int(v) for k, v in zip(*np.unique(D["code"][t], return_counts=True))}, "yg mean", D["yg"][t].mean(), "pool yg mean", D["pool_yg"][t].mean(), "pool_flip share", D["pool_flip"][t].mean(), "pool_po>0", (D["pool_po"][t] > 0).mean(), "pool_pdf>0", (D["pool_pdf"][t] > 0).mean(), "idx==-1", int((D["idx"][t] < 0).sum()))
        print("  field after (yl - yg clamp [1,99]) mean", np.clip(D["yl"][t] - D["yg"][t], 1, 99).mean(), "before", D["yl"][t].mean())
    print("non-terminal yards_gained quantiles", np.quantile(D["yg"][~term], [0.001, 0.01, 0.5, 0.99]).tolist())


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--worlds", type=int, default=2)
    r.add_argument("--seasons", type=int, default=4)
    r.add_argument("--workers", type=int, default=3)
    r.add_argument("--seed", type=int, default=49)
    sub.add_parser("analyze")
    a = ap.parse_args()
    {"run": cmd_run, "analyze": cmd_analyze}[a.cmd](a)


if __name__ == "__main__":
    main()
