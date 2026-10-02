import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

INT_VARIANTS = ("lbase", "lnh", "lconv", "lkl")


def register_int():
    import mod25d_variance as dv
    import sim09_int as si
    import sim09_u4f as u4f

    base = dv.DV["crzhk"]
    dv.DV["lbase"] = dict(base, intc=1)
    dv.DV["lnh"] = dict(base, e1=1, intc=1)
    dv.DV["lconv"] = dict(base, fdnb=0, e2=1, intc=1)
    dv.DV["lkl"] = dict(base, kl=u4f.REG, intc=1)
    return si


def run_e5(args):
    v = args.variant
    if v in INT_VARIANTS:
        si = register_int()
        si.cmd_e5(args)
    elif v == "u4g":
        import sim09_u4g as m

        m.cmd_e5(args)
    elif v == "f2":
        import sim09_f2 as m

        m.cmd_e5(args)
    elif v == "crzf3":
        import os

        import sim09_f3 as m

        outd = REPO / "artifacts" / "mod25e3" / f"e5_{v}_s{args.seed}"
        outd.mkdir(parents=True, exist_ok=True)
        os.environ["BUD_OUT"] = str(outd)
        m.cmd_e5(args)
    elif v == "crzhr":
        import mod25e_rz as m

        m.cmd_e5(args)
    elif v == "crzhk":
        import sim09_hk as m

        m.cmd_e5(args)
    else:
        raise SystemExit("unknown variant " + v)


METRICS = [
    ("margin_sd", "gate", "margin_sd"),
    ("nonstr", "gate", "nonstrength_var"),
    ("r2_10_18", "gate", "r2_w10_18"),
    ("strength", "budget", "strength_re_var"),
    ("noise", "budget", "noise_re_var"),
    ("xq_cov", "budget", "xq_cov_sum"),
    ("q4_slope", "budget", "q4_slope"),
    ("drives", "budget", "drives_g"),
    ("pts_g", "gate", "pts_game"),
]


def load(v, seed):
    import json

    p = REPO / "artifacts" / "mod25e3" / f"e5_{v}_s{seed}" / "e5.json"
    if not p.exists():
        return None
    d = json.loads(p.read_text())
    return [d[a][b] for _, a, b in METRICS]


def cmd_table(args):
    import numpy as np

    def runs(v):
        out = {}
        for s in args.seeds.split(","):
            r = load(v, int(s))
            if r is not None:
                out[int(s)] = r
        return out

    base = runs("crzhk")
    allb = np.array(list(base.values()))
    sd = allb.std(0, ddof=1) if len(allb) > 1 else np.zeros(len(METRICS))
    print("base crzhk seeds", sorted(base), "mean", np.round(allb.mean(0), 3).tolist())
    print("seed SD", np.round(sd, 3).tolist())
    print("cols", [m[0] for m in METRICS])
    for pair in args.pairs.split(";"):
        v, ref = pair.split(":")
        rv, rr = runs(v), runs(ref)
        common = sorted(set(rv) & set(rr))
        if not common:
            print(v, "no common seeds")
            continue
        d = np.array([np.array(rv[s]) - np.array(rr[s]) for s in common])
        m = d.mean(0)
        flag = ["*" if abs(x) > 2 * s_ and s_ > 0 else "" for x, s_ in zip(m, sd)]
        print(v, "vs", ref, "seeds", common, "n", len(rv))
        print("  delta", [f"{x:+.3f}{f}" for x, f in zip(m, flag)])
        print("  perseed", [[round(x, 3) for x in row] for row in d.tolist()])


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("e5")
    e.add_argument("--variant", required=True)
    e.add_argument("--worlds", type=int, default=8)
    e.add_argument("--seasons", type=int, default=8)
    e.add_argument("--workers", type=int, default=3)
    e.add_argument("--seed", type=int, default=11)
    t = sub.add_parser("table")
    t.add_argument("--seeds", default="11,12")
    t.add_argument("--pairs", required=True)
    a = ap.parse_args()
    if a.cmd == "e5":
        run_e5(a)
    else:
        cmd_table(a)


if __name__ == "__main__":
    main()
