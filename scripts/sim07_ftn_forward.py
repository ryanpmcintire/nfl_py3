import argparse
import datetime
import glob
import hashlib
import json

import numpy as np
import pandas as pd
import sim07_ftn_loso as lo
import sim07_ftn_state as s
import sim07_personnel as m
import sim07_power as p

FROZEN = m.OUT / "ftn_frozen_2022_2025.json"
RESULT = m.OUT / "ftn_forward_2026.json"
FTN26 = m.ROOT / "data/raw/ftn_charting_2026/ftn_charting_2026.parquet"
SEASON = 2026
BOOT_GAME = 2000


def freeze():
    if FROZEN.exists():
        raise SystemExit("frozen file already exists")
    d = lo.build()
    f = lo.fitter(d)
    cell, sw, *_ = f.fit(np.ones(len(f.y)))
    cells = list(f.cells)
    proj, _ = p.projector(cells, sw)
    dev = proj @ cell
    box = s.clip_rule(d.n_defense_box)
    body = dict(  # noqa: C408
        seasons=list(s.SEASONS),
        plays=len(d),
        og_keep=sorted(set(d.og.unique()) - {"other"}),
        box_lo=int(box.min()),
        box_hi=int(box.max()),
        cells=cells,
        raw_cell=cell.tolist(),
        cell_plays=sw.tolist(),
        frozen_deviation=dev.tolist(),
        ftn_hash={
            y: hashlib.sha256(
                open(f"{s.FTN}/ftn_charting_{y}.parquet", "rb").read()  # noqa: SIM115
            ).hexdigest()
            for y in s.SEASONS
        },
    )
    payload = json.dumps(body, sort_keys=True)
    out = dict(  # noqa: C408
        frozen_utc=datetime.datetime.now(datetime.UTC).isoformat(),
        sha256=hashlib.sha256(payload.encode()).hexdigest(),
        body=body,
    )
    FROZEN.write_text(json.dumps(out, indent=1))
    print(out["frozen_utc"], out["sha256"], len(cells), flush=True)


def participation_2026():
    hits = sorted(
        glob.glob(str(m.ROOT / f"data/players/participation/raw/*/season={SEASON}/*.parquet"))
    )
    if not hits:
        return None
    return pd.read_parquet(
        hits[-1], columns=["game_id", "play_id", "offense_personnel", "defense_personnel"]
    )


def build_2026(fz):
    pt = participation_2026()
    if pt is None:
        return None
    pbp = sorted(glob.glob(str(m.ROOT / "data/pbp/raw/*")))[-1]
    cols = [
        *p.EXTRA,
        "game_id",
        "play_id",
        "season",
        "season_type",
        "week",
        "posteam",
        "defteam",
        "down",
        "play_type",
        "epa",
        "qtr",
        "ydstogo",
        "yardline_100",
        "game_seconds_remaining",
        "qb_kneel",
        "qb_spike",
        "score_differential",
    ]
    x = pd.read_parquet(f"{pbp}/season={SEASON}/plays.parquet", columns=cols)
    d = x[
        (x.season_type == "REG")
        & x.posteam.notna()
        & x.defteam.notna()
        & x.play_type.isin(["pass", "run"])
        & (x.qb_kneel != 1)
        & (x.qb_spike != 1)
        & (x.qtr <= 4)
        & x.down.notna()
        & x.score_differential.notna()
        & x.epa.notna()
        & x.xpass.notna()
        & x.wp.notna()
    ].copy()
    pt["play_id"] = pt.play_id.astype(d.play_id.dtype)
    pt = pt.drop_duplicates(["game_id", "play_id"])
    d = d.merge(pt, on=["game_id", "play_id"], how="left")
    d["off"] = d.offense_personnel.map(m.off_label)
    d["ndb"] = d.defense_personnel.map(m.db_count)
    f = pd.read_parquet(
        FTN26, columns=["nflverse_game_id", "nflverse_play_id", "n_defense_box"]
    ).rename(columns={"nflverse_game_id": "game_id", "nflverse_play_id": "play_id"})
    f["play_id"] = f.play_id.astype(float)
    d = d.merge(f.drop_duplicates(["game_id", "play_id"]), on=["game_id", "play_id"], how="left")
    d = d[d.off.notna() & d.ndb.notna() & d.n_defense_box.notna()].reset_index(drop=True)
    d["og"] = np.where(d.off.isin(fz["og_keep"]), d.off, "other")
    d["dp"] = d.n_defense_box.astype(int).clip(fz["box_lo"], fz["box_hi"]).astype(str)
    d["y"] = d.success.astype(float)
    return d


def score():
    fz = json.loads(FROZEN.read_text())["body"]
    d = build_2026(fz)
    if d is None:
        RESULT.write_text(
            json.dumps(
                {
                    "status": "participation_2026_not_published",
                    "ftn_2026_rows": len(pd.read_parquet(FTN26)),
                },
                indent=1,
            )
        )
        print("participation_2026_not_published", flush=True)
        return
    f = lo.fitter(d)
    rng = np.random.default_rng(p.SEED)
    frozen = dict(zip(fz["cells"], fz["frozen_deviation"], strict=True))
    weights = dict(zip(fz["cells"], fz["cell_plays"], strict=True))

    def run(w):
        ok, dev, sw = lo.deviations(f, w, [c for c in fz["cells"] if c in set(f.cells)])
        pred = np.array([frozen[c] for c in ok])
        pw = np.array([weights[c] for c in ok])
        return ok, pred, dev, np.minimum(pw, sw)

    ok, pred, dev, wt = run(np.ones(len(f.y)))
    r = m.wcorr(pred, dev, wt)[0]
    draws = []
    for _ in range(BOOT_GAME):
        _, a, b, w2 = run(f.boot(rng))
        draws.append(m.wcorr(a, b, w2)[0])
    draws = np.array(draws)
    draws = draws[~np.isnan(draws)]
    carry = ["13|5", "other|5", "13|8", "21|7", "12|5", "13|6"]
    n_cell = dict(zip(ok, dev.tolist(), strict=True))
    counts = pd.Series(f.c).map(dict(enumerate(f.cells))).value_counts().to_dict()
    out = dict(  # noqa: C408
        status="scored",
        plays=len(d),
        weeks=sorted(int(v) for v in d.week.unique()),
        games=len(f.games),
        cells=len(ok),
        corr=r,
        ci=[float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))],
        p_positive=float((draws > 0).mean()),
        sign_agree=int(((pred > 0) == (dev > 0)).sum()),
        carrying=[
            dict(  # noqa: C408
                cell=c,
                frozen_pp=float(100 * frozen[c]),
                dev_pp=float(100 * n_cell[c]) if c in n_cell else None,
                plays=int(counts.get(c, 0)),
            )
            for c in carry
        ],
        plays_per_cell=counts,
    )
    RESULT.write_text(json.dumps(out, indent=1))
    print(json.dumps({k: out[k] for k in ("status", "plays", "weeks", "corr", "ci", "p_positive")}))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["freeze", "score"])
    a = ap.parse_args()
    freeze() if a.mode == "freeze" else score()


if __name__ == "__main__":
    main()
