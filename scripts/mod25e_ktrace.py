import atexit
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

import mod25e_crH as H  # noqa: E402
import mod25e_gz as GZ  # noqa: E402

LOG = REPO / "artifacts" / "mod25e3" / "e82" / "trace_rows.csv"
BUF = []
COLS = "pid,down,qtr,clock,sd,po0,pdf0,flip0,nyl0,el0,gpo,gpdf,gflip,gnyl,gpo2,gpdf2,gflip2,gnyl2,po1,pdf1,flip1,nyl1,el1,nd1,nq,nclock,nsd,nyl,ndown"
NAN = float("nan")
PEND = {"row": None}
CUR = {"g": None}


def flush():
    if not BUF:
        return
    LOG.parent.mkdir(parents=True, exist_ok=True)
    new = not LOG.exists()
    with open(LOG, "a") as f:
        if new:
            f.write(COLS + "\n")
        f.write("\n".join(BUF) + "\n")
    BUF.clear()


atexit.register(flush)
orig = H.run_with_flags
orig_gz = GZ.install_gz


def row_of(d):
    return (float(d["points_off"]), float(d["points_def"]), float(d["flip"]), float(d["next_yardline"]))


def install_gz_traced():
    orig_gz()
    dv = H.dv
    gbase = dv._G["pol"]

    def gpol(down, distance, yardline, score_diff, qtr, clock_val, drawn):
        before = row_of(drawn)
        out = gbase(down, distance, yardline, score_diff, qtr, clock_val, drawn)
        CUR["g"] = before + row_of(out)
        return out

    dv._G["pol"] = gpol


GZ.install_gz = install_gz_traced


def patched(init, setting):
    orig(init, setting)
    dv = H.dv
    base = dv._G["pol"]

    def pol(down, distance, yardline, score_diff, qtr, clock_val, drawn):
        loc = sys._getframe(1).f_locals
        offense = loc["offense"]  # noqa: F841
        off_to = loc["off_to"]  # noqa: F841
        def_to = loc["def_to"]  # noqa: F841
        gsr = loc.get("gsr")  # noqa: F841
        possessions = loc.get("possessions")  # noqa: F841
        if PEND["row"] is not None:
            BUF.append(PEND["row"] + ",".join(str(v) for v in (qtr, clock_val, score_diff, yardline)) + "," + str(down))
            PEND["row"] = None
            if len(BUF) >= 1:
                flush()
        po, pdf = float(drawn["points_off"]), float(drawn["points_def"])
        pre = (po, pdf, float(drawn["flip"]), float(drawn["next_yardline"]), float(drawn["clock_elapsed"]))
        CUR["g"] = None
        out = base(down, distance, yardline, score_diff, qtr, clock_val, drawn)
        po1, pdf1 = float(out["points_off"]), float(out["points_def"])
        sc = qtr <= 4 and ((po >= 3 and pdf == 0) or (pdf >= 6 and po == 0) or (po1 >= 3 and pdf1 == 0) or (pdf1 >= 6 and po1 == 0))
        if sc:
            g = CUR["g"] if CUR["g"] is not None else (NAN,) * 8
            PEND["row"] = ",".join(str(v) for v in (os.getpid(), down, qtr, clock_val, score_diff) + pre + g + (po1, pdf1, float(out["flip"]), float(out["next_yardline"]), float(out["clock_elapsed"]), float(out["next_down"]))) + ","
        return out

    dv._G["pol"] = pol


H.run_with_flags = patched

if __name__ == "__main__":
    H.main()
