import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
OUTD = REPO / "artifacts" / "mod25e3" / "gfl"


def enabled():
    return os.environ.get("GFL") == "1"


def real_rows_fixed():
    import mod25e_clk as ck

    R = pd.read_parquet(ck.REAL)
    R = R[R.qtr.isin([2, 4]) & R.code.isin([0, 1, 4, 5]) & (R.penalty != 1)].copy()
    R["hs"] = np.where(R.qtr == 2, R.gsr - 1800.0, R.gsr.astype(float))
    R = R[R.hs > 0]
    R["el"] = np.minimum(R.el.astype(float), R.hs)
    fl = R.flip.to_numpy().copy()
    fl[R.next_qtr.isna().to_numpy()] = False
    R["flip"] = fl
    R["cls"] = ck.klass(R.code, R.yards, R.flip, R.po, R.pdf)
    return R[R.cls >= 0].reset_index(drop=True)


def patch():
    import mod25e_clk as ck

    ck.real_rows = real_rows_fixed
    ck.FIT = OUTD / "fit.json"
    ck.FIT2 = OUTD / "fit2.json"


def fit_all():
    import mod25e_clk as ck

    patch()
    ck.fit()
    ck.fit2()


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "fit":
        fit_all()
