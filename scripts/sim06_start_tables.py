import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
for k_ in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(k_, "1")
os.environ.setdefault("SIM10_SEASONS", "2009-2017")
os.environ.setdefault("SIM10_OUT", str(REPO / "artifacts" / "sim10"))
sys.path.insert(0, str(REPO / "scripts"))

import sim10_drive as s10  # noqa: E402

MAP = {
    "Touchdown": "kickoff",
    "Field goal": "kickoff",
    "Opp touchdown": "kickoff",
    "Punt": "punt",
    "Turnover": "turnover",
    "Turnover on downs": "downs",
    "Missed field goal": "missed_fg",
    "Safety": "kickoff",
}
SLOPE_TYPES = ("punt", "turnover", "downs", "missed_fg")
OUTFILE = REPO / "artifacts" / "sim06" / "start_tables.npz"


def pbp(seasons):
    fr = []
    for s in seasons:
        f = pd.read_parquet(
            s10.PBP_DIR / f"season={s}" / "plays.parquet",
            columns=["game_id", "play_id", "season_type", "fixed_drive_result"],
        )
        fr.append(f[f["season_type"] == "REG"])
    return (
        pd.concat(fr, ignore_index=True)
        .sort_values(["game_id", "play_id"], kind="stable")
        .reset_index(drop=True)
    )


def end_types(seg, p):
    res = p["fixed_drive_result"].to_numpy(object)[seg["row"].to_numpy(int)]
    out = np.array([MAP.get(x, "none") for x in res], dtype=object)
    out[(seg["he"].to_numpy() == 1) | (seg["nsame"].to_numpy() == 1)] = "none"
    return out


def main():
    seg = pd.read_parquet(s10.OUT / "segments.parquet")
    typ = end_types(seg, pbp(range(2009, 2018)))
    x = seg["yl"].to_numpy(float)
    y = seg["nyl"].to_numpy(float)
    half = seg["half"].to_numpy()
    slope_seg = np.full(len(seg), -1.0)
    slopes = {}
    for t in SLOPE_TYPES:
        m = (typ == t) & (half < 3)
        slopes[t] = float(np.cov(x[m], y[m])[0, 1] / x[m].var(ddof=1))
        slope_seg[typ == t] = slopes[t]
    seg2 = pd.read_parquet(REPO / "artifacts" / "sim06" / "sim10" / "segments.parquet")
    typ2 = end_types(seg2, pbp(range(2018, 2026)))
    allseg = pd.concat([seg, seg2], ignore_index=True)
    alltyp = np.r_[typ, typ2]
    ko = (alltyp == "kickoff") & (allseg["half"].to_numpy() < 3)
    yrs = allseg["season"].to_numpy(int)[ko]
    vals = allseg["nyl"].to_numpy(float)[ko]
    order = np.argsort(yrs, kind="stable")
    np.savez(
        OUTFILE,
        slope_seg=slope_seg,
        ko_years=yrs[order],
        ko_values=vals[order],
        slope_types=np.array(SLOPE_TYPES),
        slope_values=np.array([slopes[t] for t in SLOPE_TYPES]),
    )
    print(slopes)
    for s in range(2009, 2026):
        v = vals[yrs == s]
        print(s, len(v), round(float(v.mean()), 2))


if __name__ == "__main__":
    main()
