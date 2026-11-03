import glob
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import sim07_personnel as m
import sim07_power as p

FTN = sorted(glob.glob(str(m.ROOT / "data/raw/ftn_charting/*")))[-1]
SEASONS = (2022, 2023, 2024, 2025)
GRID = np.array([0.005, 0.01, 0.015, 0.02, 0.03, 0.045, 0.06, 0.08, 0.1, 0.14])
OLD_FULL = 0.044


def load_ftn():
    fr = []
    for s in SEASONS:
        x = pd.read_parquet(
            f"{FTN}/ftn_charting_{s}.parquet",
            columns=["nflverse_game_id", "nflverse_play_id", "n_defense_box", "n_blitzers"],
        )
        fr.append(x)
    f = pd.concat(fr, ignore_index=True)
    f = f.rename(columns={"nflverse_game_id": "game_id", "nflverse_play_id": "play_id"})
    f["play_id"] = f.play_id.astype(float)
    return f.drop_duplicates(["game_id", "play_id"])


def clip_rule(v):
    v = v.astype(int)
    share = v.value_counts(normalize=True)
    med = v.median()
    lo = [k for k in share.index if share[k] >= 1 - m.COVER and k < med]
    hi = [k for k in share.index if share[k] >= 1 - m.COVER and k >= med]
    return v.clip(lower=min(lo), upper=max(hi))


def instrument(d, ycol, cols, rng, label):
    f = p.Aug(d, cols)
    f.y[:] = d[ycol].values.astype(np.float64)
    cell, sw, *_ = f.fit(np.ones(len(f.y)))
    resid = f.t1.copy()
    s, n = p.cell_game_matrices(f, resid)
    mb = p.boot_means(s, n, rng, p.BOOT)
    se = mb.std(axis=0, ddof=1)
    proj, rank = p.projector(f.cells, sw)
    crit, pw, qf = p.power_curve(mb, proj, rank, rng, GRID)
    q_obs = float(qf(proj @ cell))
    p_obs = float((qf((mb - mb.mean(axis=0)) @ proj.T) >= q_obs).mean())
    out = dict(  # noqa: C408
        instrument=label,
        label=ycol,
        plays=len(d),
        cells=int(f.n_c),
        rank=int(rank),
        label_sd=float(resid.std()),
        cell_se_median=float(np.median(se)),
        cell_plays_min=int(sw.min()),
        detectable_at_power_0_8=p.smallest(GRID, pw),
        power_grid=dict(zip(map(str, GRID), pw)),  # noqa: B905
        omnibus_q=q_obs,
        omnibus_crit=float(crit),
        omnibus_p_boot=p_obs,
    )
    print(json.dumps(out), flush=True)
    return out


def main():
    t0 = time.time()
    raw = m.build()
    raw = raw[raw.season.isin(SEASONS)].copy()
    ftn = load_ftn()
    j = raw.merge(ftn, on=["game_id", "play_id"], how="left")
    match = (
        j.assign(
            ftn_row=j.n_defense_box.notna() | j.n_blitzers.notna(),
            box=j.n_defense_box.notna(),
            part=j.off.notna() & j.ndb.notna(),
        )
        .groupby("season")[["ftn_row", "box", "part"]]
        .mean()
    )
    print(match, flush=True)
    j = j[j.n_defense_box.notna()]
    j = j.merge(p.extra(), on=["game_id", "play_id"], how="left")
    j = j[j.xpass.notna() & j.wp.notna() & j.fumble_lost.notna()].reset_index(drop=True)
    d, _, _ = m.prepare(j)
    d = d.reset_index(drop=True)
    box = clip_rule(d.n_defense_box)
    blitz = (d.n_blitzers.fillna(0) >= 1).astype(int)
    states = {
        "old_db_count": d.dp.copy(),
        "ftn_box": box.astype(str),
        "ftn_box_x_blitz": box.astype(str) + "_b" + blitz.astype(str),
    }
    rng = np.random.default_rng(p.SEED)
    res = []
    for name, dp in states.items():
        x = d.copy()
        x["dp"] = dp
        for ycol, cols in (("epa", []), ("success", ["xpass", "wp"])):
            res.append(instrument(x, ycol, cols, rng, name))
    summ = dict(  # noqa: C408
        match_rate_by_season=match.to_dict(),
        ftn_dir=FTN,
        boot=p.BOOT,
        plant=p.PLANT,
        cells_planted=p.NPLANT,
        old_instrument_full_data_detectable=OLD_FULL,
        results=res,
        looks=len(res),
        seconds=time.time() - t0,
        ftn_hash={
            s: hashlib.sha256(Path(f"{FTN}/ftn_charting_{s}.parquet").read_bytes()).hexdigest()
            for s in SEASONS
        },
    )
    (m.OUT / "ftn_state_summary.json").write_text(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
