import glob
import json

import numpy as np
import pandas as pd
import sim07_personnel as m

PBP = sorted(glob.glob(str(m.ROOT / "data/pbp/raw/*")))[-1]
OUT = m.OUT / "personnel_drive_map.json"
YD_BINS = (5, 10)
YL_BINS = (10, 20)
TARGETS = {"td": "Touchdown", "fg": "Field goal"}
SEED = 20261103
BOOT = 500


def load():
    cols = [
        "game_id",
        "season",
        "season_type",
        "fixed_drive",
        "fixed_drive_result",
        "down",
        "play_type",
        "success",
        "ydstogo",
        "yardline_100",
        "qb_kneel",
        "qb_spike",
    ]
    fr = []
    for s in m.SEASONS:
        x = pd.read_parquet(f"{PBP}/season={s}/plays.parquet", columns=cols)
        fr.append(x[x.season_type == "REG"])
    d = pd.concat(fr, ignore_index=True)
    d = d[
        d.play_type.isin(["pass", "run"])
        & (d.qb_kneel != 1)
        & (d.qb_spike != 1)
        & d.down.notna()
        & d.success.notna()
        & d.fixed_drive_result.notna()
    ].copy()
    d["drive"] = d.game_id + "_" + d.fixed_drive.astype(int).astype(str)
    return d


def slope(d, y, nyd, nyl):
    key = (
        d.down.astype(int).astype(str)
        + "|"
        + pd.qcut(d.ydstogo, nyd, duplicates="drop").astype(str)
        + "|"
        + pd.qcut(d.yardline_100, nyl, duplicates="drop").astype(str)
    )
    x = d.success.astype(float)
    xr = x - x.groupby(key).transform("mean")
    yr = y - y.groupby(key).transform("mean")
    dr = d.drive.values
    cx = pd.Series((xr * yr).values).groupby(dr).sum()
    vx = float((xr * xr).sum())
    return float((xr * yr).sum() / vx), cx, pd.Series((xr * xr).values).groupby(dr).sum()


def main():
    d = load()
    nbar = float(d.groupby("drive").size().mean())
    res = {
        "seasons": [int(m.SEASONS[0]), int(m.SEASONS[-1])],
        "plays": len(d),
        "drives": int(d.drive.nunique()),
        "plays_per_drive": nbar,
        "pbp_dir": PBP,
        "targets": {},
    }
    rng = np.random.default_rng(SEED)
    for t, lab in TARGETS.items():
        y = (d.fixed_drive_result == lab).astype(float)
        out = {}
        for nyd in YD_BINS:
            for nyl in YL_BINS:
                b, num, den = slope(d, y, nyd, nyl)
                ids = num.index.values
                nv, dv = num.values, den.values
                bs = []
                for _ in range(BOOT):
                    k = rng.integers(0, len(ids), len(ids))
                    bs.append(nv[k].sum() / dv[k].sum())
                bs = np.array(bs)
                out[f"yd{nyd}_yl{nyl}"] = {
                    "tau": b,
                    "lo": float(np.quantile(bs, 0.025)),
                    "hi": float(np.quantile(bs, 0.975)),
                    "p_positive": float((bs > 0).mean()),
                }
        res["targets"][t] = out
    res["chosen"] = "yd10_yl20"
    res["tau"] = {t: res["targets"][t][res["chosen"]]["tau"] for t in TARGETS}
    OUT.write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
