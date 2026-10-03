import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

import mod25e_drill as dr  # noqa: E402
import mod25e_endgame as eg  # noqa: E402

OUT = REPO / "artifacts" / "mod25e3" / "to"
HB = [0, 15, 30, 60, 120]


def prep(d, sim):
    d = d.copy()
    if sim:
        d["oto"] = d["oto_sim"]
        d["dto"] = d["dto_sim"]
    d["sc"] = (d.po + d.pdf) > 0
    d["stp"] = eg.stop_after(d.code, d.yards, d.flip, d.sc, d.otu, d.dtu)
    d["half"] = np.where(d.qtr.to_numpy() >= 3, 2, 1)
    d = d.sort_values(["g", "play_id"] if "play_id" in d and not sim else ["g"], kind="stable").reset_index(drop=True)
    d["pstop"] = d.groupby(["g", "half"])["stp"].shift(1).fillna(1.0)
    d["hs"] = np.where(d.qtr == 2, d.gsr - 1800.0, np.where(d.qtr == 4, d.gsr, np.nan))
    d = d[d.qtr.isin((2, 4)) & (d.hs > 0) & (d.hs <= 120) & (d.code != 6)].copy()
    d["tb"] = pd.cut(d.hs, HB)
    d["st"] = np.where(d.sd < 0, "trail", np.where(d.sd > 0, "lead", "tied"))
    d["o"] = (d.otu > 0).astype(float)
    d["f"] = (d.dtu > 0).astype(float)
    d["otc"] = np.minimum(d.oto, 3)
    d["dtc"] = np.minimum(d.dto, 3)
    d["run"] = np.where(d.pstop == 0, "run", "stop")
    return d


def tab(d, by):
    g = d.groupby(by, observed=True)
    return pd.DataFrame({"n": g.size(), "off": g.o.mean(), "def": g.f.mean()}).round(3)


def left_at(d, T):
    x = d[d.hs <= T].sort_values("hs", ascending=False, kind="stable").groupby(["g", "half"]).head(1)
    tr = np.where(x.sd < 0, x.oto, np.where(x.sd > 0, x.dto, np.nan))
    ld = np.where(x.sd < 0, x.dto, np.where(x.sd > 0, x.oto, np.nan))
    return len(x), np.nanmean(tr), np.nanmean(ld), float((x.oto + x.dto).mean())


def main():
    path = sys.argv[1]
    R = prep(eg.train_frame(), False)
    S = prep(dr.sim_df(path), True)
    pd.set_option("display.width", 200, "display.max_rows", 500)
    L = []
    for q in (2, 4):
        r = R[R.qtr == q]
        s = S[S.qtr == q]
        L.append(f"==== Q{q} total snaps real {len(r)} sim {len(s)}")
        for by in (["tb"], ["tb", "st"], ["tb", "otc"], ["tb", "dtc"], ["tb", "run"]):
            t = pd.concat({"real": tab(r, by), "sim": tab(s, by)}, axis=1)
            L.append(t.to_string())
        for T in (60, 30, 15):
            a = left_at(r[r.hs > 0], T)
            b = left_at(s[s.hs > 0], T)
            L.append(f"Q{q} first snap hs<={T}: real n {a[0]} trailer {a[1]:.2f} leader {a[2]:.2f} both {a[3]:.2f} | sim n {b[0]} trailer {b[1]:.2f} leader {b[2]:.2f} both {b[3]:.2f}")
    txt = chr(10).join(L)
    print(txt)
    (OUT / (sys.argv[2] + ".txt")).write_text(txt)


if __name__ == "__main__":
    main()
