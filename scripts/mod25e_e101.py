import io
import os
import sys
import contextlib
import glob
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
OUTD = REPO / "artifacts" / "mod25e3" / "e101"
NEW = "crHpqokgndecsmfwtjo2as2ypw2xfdcd"
BASE = "crHpqokgndecsmfwtjo2as2ypw2"

JOBS = {
    "e93": ("mod25e_e93.py", {'artifacts" / "mod25e3" / "e93"': 'artifacts" / "mod25e3" / "e101" / "e93"'}, {"A2_LABEL": NEW, "E93_TB": "cond", "E93_FG": "rule"}),
    "e95": ("mod25e_e95.py", {'artifacts" / "mod25e3" / "e95"': 'artifacts" / "mod25e3" / "e101" / "e95"'}, {"A2_LABEL": NEW}),
    "e97": ("mod25e_e97.py", {'LABEL = "crHpqokgndecsmfwtjo2as2"': 'LABEL = "' + NEW + '"', 'E3 / "e97"': 'E3 / "e101" / "e97"'}, {}),
    "e97b": ("mod25e_e97.py", {'LABEL = "crHpqokgndecsmfwtjo2as2"': 'LABEL = "' + BASE + '"', 'E3 / "e97"': 'E3 / "e101" / "e97b"'}, {}),
    "e99": ("mod25e_e99.py", {'LABEL = "' + BASE + '"': 'LABEL = "' + NEW + '"', 'OUTD = E3 / "e99"': 'OUTD = E3 / "e101" / "e99"'}, {}),
}


def run(name):
    fn, subs, env = JOBS[name]
    src = (REPO / "scripts" / fn).read_text(encoding="utf-8")
    for a, b in subs.items():
        assert a in src, (name, a)
        src = src.replace(a, b)
    os.environ.update(env)
    (OUTD / name).mkdir(parents=True, exist_ok=True)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        exec(compile(src, fn, "exec"), {"__name__": "__main__", "__file__": str(REPO / "scripts" / fn)})
    (OUTD / f"{name}_stdout.txt").write_text(buf.getvalue(), encoding="utf-8")


def snap():
    import numpy as np
    import pandas as pd
    import mod25e_endgame as eg
    T = eg.train_frame()
    def sel(d):
        return d[(d.qtr == 4) & (d.gsr <= 300) & (d.sd <= 0) & (d.sd >= -2) & (d.yl <= 40) & (d.down <= 3) & d.code.isin(eg.LATE_CODES) & (d.down >= 2)]
    rows = []
    r = sel(T[(T.season >= 2011) & (T.season <= 2017)])
    rows.append(("real", len(r), float((r.gsr <= 5).mean())))
    for lab in (BASE, NEW):
        S = []
        for sd in (11, 12, 13):
            for f in sorted(glob.glob(str(REPO / f"artifacts/mod25e3/e5_{lab}_s{sd}/play_*_*.parquet"))):
                w, s = (int(x) for x in Path(f).stem.split("_")[1:])
                if s >= 2:
                    S.append(pd.read_parquet(f))
        S = pd.concat(S, ignore_index=True)
        z = sel(S)
        rows.append((lab, len(z), float((z.gsr <= 5).mean())))
    txt = "\n".join(f"{a} n {b} P(gsr<=5 | Q4 gsr<=300, tied/trail<=2, yl<=40, downs 2-3) {c:.3f}" for a, b, c in rows)
    (OUTD / "snap.txt").write_text(txt, encoding="utf-8")
    print(txt)


if __name__ == "__main__":
    OUTD.mkdir(parents=True, exist_ok=True)
    for a in sys.argv[1:]:
        if a == "snap":
            snap()
        else:
            run(a)
